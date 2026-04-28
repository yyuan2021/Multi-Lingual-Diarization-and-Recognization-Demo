"""
Demo Pipeline: Two-Speaker Multilingual Transcription
======================================================
Input : mixed audio with two speakers (any sample rate)
Output: per-speaker transcript, with language-switch attribution fixed

Pipeline
--------
1. MossFormer2 speaker separation  →  spk0.wav / spk1.wav  (8 kHz)
2. WhisperX transcription          →  word-level timestamps + phoneme alignment
                                       + per-word language detection
3. Mixed-language segment check    →  TDNN same-speaker classifier
      same speaker  (code-switch)  →  keep text as-is
      diff speaker  (cross-talk)   →  move the interval in audio
4. Rewrite speaker audio           →  spk0_fixed.wav / spk1_fixed.wav
5. WhisperX transcription again    →  final transcript from fixed audio

Usage
-----
    python demo/demo_pipeline.py samples/mixed.wav
    python demo_pipeline.py path/to/mixed.wav --whisper-model small --device cuda
"""

from __future__ import annotations

import os
# Must be set before any torch/MKL import to prevent segfault in MossFormer2 on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import librosa
import numpy as np
import soundfile as sf

import torch


# ── make project root importable ─────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from dl_model.final_model.model import TDNNPredictor  # TDNN same-speaker model

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MOSSFORMER_SR = 8000   # MossFormer2 requires 8 kHz input
WHISPER_SR = 16000     # Whisper / TDNN require 16 kHz
MIN_SPAN_SEC = 0.10    # minimum audio span length for TDNN (100 ms)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Speaker Separation (MossFormer2)
# ─────────────────────────────────────────────────────────────────────────────

def _import_modelscope():
    """
    Import modelscope while ensuring the local datasets/ folder in the project
    root does NOT shadow the HuggingFace `datasets` package that modelscope needs.
    """
    import os

    _root_str = str(_ROOT)
    _cwd_strs = {"", ".", os.getcwd()}
    _hide = {p for p in sys.path if p in _cwd_strs or p == _root_str}

    saved_path = sys.path[:]
    for p in _hide:
        while p in sys.path:
            sys.path.remove(p)

    # Also evict any already-cached stub for the local datasets namespace pkg
    for key in list(sys.modules.keys()):
        if key == "datasets" or key.startswith("datasets."):
            del sys.modules[key]

    try:
        import resampy  # noqa: F401
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
    except ImportError as exc:
        raise ImportError(
            "ModelScope and resampy are required for speaker separation.\n"
            "Install them in the active conda env:\n"
            "  pip install modelscope resampy"
        ) from exc
    finally:
        sys.path[:] = saved_path

    return ms_pipeline, Tasks


def separate_speakers(audio_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Run MossFormer2 on *audio_path* and write spk0.wav / spk1.wav to output_dir.
    Returns (spk0_path, spk1_path).
    """
    ms_pipeline, Tasks = _import_modelscope()

    try:
        import resampy
    except ImportError as exc:
        raise ImportError("pip install resampy") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare mono 8 kHz audio
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != MOSSFORMER_SR:
        audio = resampy.resample(audio, sr, MOSSFORMER_SR)

    prep_path = output_dir / "_prepared_8k.wav"
    sf.write(str(prep_path), audio, MOSSFORMER_SR)

    separator = ms_pipeline(
        Tasks.speech_separation,
        model="iic/speech_mossformer2_separation_temporal_8k",
        revision="v0.9.0",
    )
    result = separator(str(prep_path))

    paths: List[Path] = []
    for i, pcm in enumerate(result["output_pcm_list"]):
        p = output_dir / f"spk{i}.wav"
        # sf.write(str(p), np.frombuffer(pcm, dtype=np.int16), MOSSFORMER_SR)
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(str(p), wav, MOSSFORMER_SR)
        paths.append(p)

    if len(paths) < 2:
        raise RuntimeError(f"MossFormer2 returned only {len(paths)} speaker(s); expected 2.")

    return paths[0], paths[1]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – WhisperX Transcription with word timestamps + language tagging
# ─────────────────────────────────────────────────────────────────────────────

def _detect_lang_by_char(ch: str) -> str:
    """Classify a character's language by Unicode range."""
    if not ch:
        return "other"

    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return "zh"
    if 0x0900 <= o <= 0x097F:
        return "hi"
    if "a" <= ch.lower() <= "z":
        return "en"
    return "other"


def _group_words_by_language(words: List[dict]) -> List[dict]:
    """
    Group a flat word list into consecutive same-language spans.
    Each span: {language, start, end, text, words}
    """
    spans: List[dict] = []
    cur: Optional[dict] = None

    def flush() -> None:
        nonlocal cur
        if cur:
            cur["text"] = " ".join(w["word"] for w in cur["words"])
            spans.append(cur)
            cur = None

    for w in words:
        token = w["word"].strip()
        if not token:
            continue
        lang = _detect_lang_by_char(next((c for c in token if c.isalpha()), token[0]))
        if lang == "other":
            flush()
            continue
        if cur is None or cur["language"] != lang:
            flush()
            cur = {
                "language": lang,
                "start": w["start"],
                "end": w["end"],
                "text": "",
                "words": [w],
            }
        else:
            cur["end"] = w["end"]
            cur["words"].append(w)

    flush()
    return spans


def transcribe_speaker(wx_model, audio_path: Path, device: str = "cpu") -> List[dict]:
    """
    Transcribe one speaker's audio with WhisperX.

    Returns a list of segment dicts:
        {segment_id, start, end, text, language_spans}

    Each language_span: {language, start, end, text, words}
    Each word:          {word, start, end, score, language}
    """
    import whisperx

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != WHISPER_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)

    result = wx_model.transcribe(audio, batch_size=8)
    detected_lang = result.get("language", "zh")

    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            align_meta,
            audio,
            device,
            return_char_alignments=False,
        )
        segments_src = aligned["segments"]
    except Exception:
        segments_src = result["segments"]

    segments_out: List[dict] = []

    for i, seg in enumerate(segments_src):
        t0 = float(seg.get("start", 0.0))
        t1 = float(seg.get("end", t0))

        words: List[dict] = []
        for w in seg.get("words", []):
            token = w.get("word", "").strip()
            w_start = w.get("start")
            w_end = w.get("end")
            if not token or w_start is None or w_end is None:
                continue
            words.append(
                {
                    "word": token,
                    "start": float(w_start),
                    "end": float(w_end),
                    "score": float(w.get("score", 0.0)),
                    "language": _detect_lang_by_char(token[0]),
                }
            )

        spans = _group_words_by_language(words)
        if not spans:
            spans = [
                {
                    "language": detected_lang,
                    "start": t0,
                    "end": t1,
                    "text": seg.get("text", "").strip(),
                    "words": [],
                }
            ]

        segments_out.append(
            {
                "segment_id": i,
                "start": t0,
                "end": t1,
                "text": seg.get("text", "").strip(),
                "language_spans": spans,
            }
        )

    return segments_out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – TDNN same-speaker check on mixed-language segments
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_CACHE = {}

def _load_window(audio_path: Path, t0: float, t1: float) -> np.ndarray:
    """Load a time window from a wav file, resampled to WHISPER_SR (16 kHz)."""
    key = str(audio_path)

    if key not in AUDIO_CACHE:
        audio, sr = sf.read(key)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != WHISPER_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)
        AUDIO_CACHE[key] = audio

    audio = AUDIO_CACHE[key]

    a = max(0, int(t0 * WHISPER_SR))
    b = min(len(audio), int(t1 * WHISPER_SR))
    chunk = audio[a:b]

    min_len = int(MIN_SPAN_SEC * WHISPER_SR)
    if len(chunk) < min_len:
        chunk = np.pad(chunk, (0, min_len - len(chunk)))

    return chunk


def check_mixed_segments(
    segments: List[dict],
    audio_path: Path,
    predictor: TDNNPredictor,
) -> List[dict]:
    """
    For each segment with adjacent spans of different language, run the TDNN.

    Returns:
        {segment_id, pair_idx, span_a, span_b, is_same_speaker, confidence}

    is_same_speaker=True  → code-switch (one speaker, two languages)
    is_same_speaker=False → cross-talk  (different speakers bled together)
    """
    results: List[dict] = []

    for seg in segments:
        spans = seg["language_spans"]
        for k in range(len(spans) - 1):
            sa, sb = spans[k], spans[k + 1]
            if sa["language"] == sb["language"]:
                continue

            wav_a = _load_window(audio_path, sa["start"], sa["end"])
            wav_b = _load_window(audio_path, sb["start"], sb["end"])

            is_same, conf = predictor.predict(wav_a, wav_b)
            if conf < 0.6:
                continue
            results.append(
                {
                    "segment_id": seg["segment_id"],
                    "pair_idx": k,
                    "span_a": sa,
                    "span_b": sb,
                    "is_same_speaker": is_same,
                    "confidence": conf,
                }
            )

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Rewrite speaker audio from confident TDNN decisions
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_16k_full(audio_path: Path) -> np.ndarray:
    """Load a full mono 16 kHz waveform."""
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != WHISPER_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)
    return audio


def _pad_to_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x
    return np.pad(x, (0, n - len(x)))


def _apply_audio_move(
    fixed0: np.ndarray,
    fixed1: np.ndarray,
    src0: np.ndarray,
    src1: np.ndarray,
    source_spk: int,
    target_spk: int,
    t0: float,
    t1: float,
) -> None:
    """
    Move [t0, t1] from source speaker track to target speaker track.
    Source interval becomes silence; target interval gets copied audio.
    """

    src = src0 if source_spk == 0 else src1

    a = max(0, int(t0 * WHISPER_SR))

    target_len = len(fixed0) if target_spk == 0 else len(fixed1)
    b = min(len(src), target_len, int(t1 * WHISPER_SR))

    if b <= a:
        return

    chunk = src[a:b]

    if source_spk == 0:
        fixed0[a:b] = 0.0
    else:
        fixed1[a:b] = 0.0

    # ✅ cleaner overwrite (no ghosting)
    if target_spk == 0:
        fixed0[a:b] = chunk
    else:
        fixed1[a:b] = chunk


def build_repair_plan(
    cross0: List[dict],
    cross1: List[dict],
) -> List[dict]:
    """
    Rule:
    - SAME speaker  → do nothing
    - DIFFERENT speaker → move span_b to the other speaker
    """

    moves: List[dict] = []
    seen: Set[Tuple[int, float, float]] = set()

    def _process(cross_results: List[dict], source_spk: int):
        for cr in cross_results:
            # ❌ SAME speaker → skip
            if cr["is_same_speaker"]:
                continue

            # ✅ DIFFERENT speaker → move second span
            span = cr["span_b"]

            t0 = float(span["start"])
            t1 = float(span["end"])

            key = (source_spk, round(t0, 2), round(t1, 2))
            if key in seen:
                continue
            seen.add(key)

            target_spk = 1 - source_spk  # switch speaker

            moves.append(
                {
                    "source_spk": source_spk,
                    "target_spk": target_spk,
                    "t0": t0,
                    "t1": t1,
                }
            )

    _process(cross0, 0)
    _process(cross1, 1)

    return moves

def rewrite_speaker_audio(
    spk0_path: Path,
    spk1_path: Path,
    moves: List[dict],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Create spk0_fixed.wav / spk1_fixed.wav by moving confident spans.
    """
    src0 = _load_audio_16k_full(spk0_path)
    src1 = _load_audio_16k_full(spk1_path)

    n = max(len(src0), len(src1))
    src0 = _pad_to_length(src0, n)
    src1 = _pad_to_length(src1, n)

    fixed0 = src0.copy()
    fixed1 = src1.copy()

    # ✅ FIX 3: prevent overlapping moves
    used = []

    for mv in sorted(moves, key=lambda x: (x["t0"], x["t1"])):
        overlap = any(not (mv["t1"] <= u[0] or mv["t0"] >= u[1]) for u in used)
        if overlap:
            continue

        used.append((mv["t0"], mv["t1"]))

        _apply_audio_move(
            fixed0=fixed0,
            fixed1=fixed1,
            src0=src0,
            src1=src1,
            source_spk=mv["source_spk"],
            target_spk=mv["target_spk"],
            t0=mv["t0"],
            t1=mv["t1"],
        )

    fixed0_path = output_dir / "spk0_fixed.wav"
    fixed1_path = output_dir / "spk1_fixed.wav"

    sf.write(str(fixed0_path), fixed0, WHISPER_SR)
    sf.write(str(fixed1_path), fixed1, WHISPER_SR)

    return fixed0_path, fixed1_path

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Format output
# ─────────────────────────────────────────────────────────────────────────────

def _extract_all_words(segments: List[dict]) -> List[dict]:
    words: List[dict] = []
    for seg in segments:
        for span in seg["language_spans"]:
            for w in span.get("words", []):
                words.append(w)
    return sorted(words, key=lambda x: x["start"])


def _format_transcript(words: List[dict], label: str) -> str:
    """Render a word list as a readable transcript string."""
    if not words:
        return f"[{label}]: (no speech detected)"

    lines: List[str] = []
    cur_lang = words[0].get("language", "?")
    buf: List[str] = []

    for w in words:
        lang = w.get("language", "?")
        if lang != cur_lang:
            lines.append(f"<{cur_lang}> {' '.join(buf)}")
            buf = [w["word"]]
            cur_lang = lang
        else:
            buf.append(w["word"])

    if buf:
        lines.append(f"<{cur_lang}> {' '.join(buf)}")

    return f"[{label}]:\n" + "\n".join(f"  {ln}" for ln in lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Translate transcript to English
# ─────────────────────────────────────────────────────────────────────────────


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_translation_cache = {}


def _get_translator(target_lang: str):
    """
    Load NLLB model once and cache it.
    """
    if "model" not in _translation_cache:
        model_name = "facebook/m2m100_418M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        _translation_cache["model"] = model
        _translation_cache["tokenizer"] = tokenizer

    return _translation_cache["model"], _translation_cache["tokenizer"]


# Map simple language codes → M2M100 codes
_LANG_MAP = {
    "en": "en",
    "zh": "zh",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ja": "ja",
}


def _join_words(words: List[dict]) -> str:
    """
    Fix broken tokenization:
    - Chinese → no spaces
    - English → proper spacing
    """
    text = ""
    for w in words:
        token = w["word"]

        if any('\u4e00' <= c <= '\u9fff' for c in token):
            text += token
        else:
            if text and not text.endswith(" "):
                text += " "
            text += token

    return text.strip()


def _is_mostly_chinese(text: str) -> bool:
    zh = sum('\u4e00' <= c <= '\u9fff' for c in text)
    return zh > len(text) * 0.2

def translate_text(text: str, target_lang: str = "en") -> str:
    if not text:
        return text

    # Skip translation if already English
    if target_lang == "en" and not _is_mostly_chinese(text):
        return text

    model, tokenizer = _get_translator(target_lang)

    src_lang = "zh" if _is_mostly_chinese(text) else "en"
    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt").to(device)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(target_lang),
        max_length=256,   # safer for M1
    )

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def build_translated_transcript(
    spk0_words: List[dict],
    spk1_words: List[dict],
    target_lang: str = "en",
) -> str:
    """
    Build translated transcript with selectable output language.
    """
    spk0_plain = _join_words(spk0_words)
    spk1_plain = _join_words(spk1_words)

    spk0_trans = translate_text(spk0_plain, target_lang)
    spk1_trans = translate_text(spk1_plain, target_lang)

    return (
        f"[Speaker 0]:\n{spk0_trans}\n\n"
        f"[Speaker 1]:\n{spk1_trans}\n"
    )


def _save_text(content: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    output_dir: str = "demo/demo_output",
    whisper_model_name: str = "small",
    device: str = "cpu",
    tdnn_weight: Optional[str] = None,
    target_lang: str = "en",
) -> None:
    import whisperx

    audio_path_obj = Path(audio_path).resolve()
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path_obj}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # WhisperX compute type: float16 on CUDA, int8 on CPU
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\n{'=' * 60}")
    print("  Two-Speaker Transcription Demo")
    print(f"  Input : {audio_path_obj.name}")
    print(f"  Device: {device}  |  WhisperX: {whisper_model_name}  |  compute: {compute_type}")
    print(f"{'=' * 60}\n")

    # ── 1. Speaker Separation ─────────────────────────────────────────────
    print("[1/6] Separating speakers (MossFormer2) ...")
    spk0_path, spk1_path = separate_speakers(audio_path_obj, out_dir)
    print(f"      spk0 → {spk0_path}")
    print(f"      spk1 → {spk1_path}\n")

    # ── 2. First-pass transcription ───────────────────────────────────────
    print(f"[2/6] First-pass WhisperX transcription ({whisper_model_name}) ...")
    wx_model = whisperx.load_model(
        whisper_model_name,
        device=device,
        compute_type=compute_type,
    )

    spk0_segs = transcribe_speaker(wx_model, spk0_path, device=device)
    spk1_segs = transcribe_speaker(wx_model, spk1_path, device=device)

    n_mixed0 = sum(1 for s in spk0_segs if len(s["language_spans"]) > 1)
    n_mixed1 = sum(1 for s in spk1_segs if len(s["language_spans"]) > 1)
    print(f"      spk0: {len(spk0_segs)} segments, {n_mixed0} with language switches")
    if spk0_segs:
        print(f"      text: {spk0_segs[0]['text']}")
    print(f"      spk1: {len(spk1_segs)} segments, {n_mixed1} with language switches\n")
    if spk1_segs:
        print(f"      text: {spk1_segs[0]['text']}")

    # ── 3. TDNN same-speaker check ────────────────────────────────────────
    print("[3/6] Checking mixed-language spans with TDNN ...")
    default_weight = "dl_model/final_model/tdnn_full_best_acc.pth"
    predictor = TDNNPredictor(
        device=device,
        weight_path=tdnn_weight if tdnn_weight else default_weight,
    )

    cross0 = check_mixed_segments(spk0_segs, spk0_path, predictor)
    cross1 = check_mixed_segments(spk1_segs, spk1_path, predictor)

    print("\n      TDNN Prediction Details:")
    print("      ----------------------------------------")

    for label, results in (("spk0", cross0), ("spk1", cross1)):
        if not results:
            print(f"      {label}: (no mixed-language pairs)")
            continue

        for r in results:
            is_same = r["is_same_speaker"]
            conf = r["confidence"]

            verdict = "SAME speaker (code-switch → KEEP)" if is_same else "DIFFERENT speakers (cross-talk → FIX)"

            print(
                f"""
        [{label}] Segment {r['segment_id']} | Pair {r['pair_idx']}
            Languages : {r['span_a']['language']} → {r['span_b']['language']}
            Time      : {r['span_a']['start']:.2f}s - {r['span_b']['end']:.2f}s
            Prediction: {is_same}
            Confidence: {conf:.4f}
            Result    : {verdict}
            ----------------------------------------
            """
            )

    
    # Check if ANY cross-talk (different speaker) exists
    has_crosstalk = any(not r["is_same_speaker"] for r in cross0 + cross1)

    if not has_crosstalk:
        print("      (no cross-talk detected)")
        print("\n[SKIP] Only code-switch detected → skipping audio rewrite and second WhisperX pass.\n")

        # Directly use first-pass results
        spk0_words = _extract_all_words(spk0_segs)
        spk1_words = _extract_all_words(spk1_segs)

        print(f"\n{'=' * 60}")
        print("  FINAL TRANSCRIPT (FIRST PASS)")
        print(f"{'=' * 60}")
        print(_format_transcript(spk0_words, "Speaker 0"))
        print()
        print(_format_transcript(spk1_words, "Speaker 1"))
        print(f"{'=' * 60}\n")

        final_text = (
            _format_transcript(spk0_words, "Speaker 0")
            + "\n\n"
            + _format_transcript(spk1_words, "Speaker 1")
        )

        _save_text(final_text, out_dir / "final_transcript.txt")

        # ── 6. Translate ──────────────────────────────────────────────────
        print(f"\n[6/6] Translating transcript → {target_lang} ...")
        translated_text = build_translated_transcript(spk0_words, spk1_words, target_lang)
        _save_text(translated_text, out_dir / f"final_transcript_{target_lang}.txt")

        return

    print()

    # ── 4. Rewrite audio using confident TDNN moves ──────────────────────
    print("[4/6] Rewriting speaker audio into fixed tracks ...")
    moves = build_repair_plan(cross0, cross1)

    if moves:
        print(f"      confident moves found: {len(moves)}")
    else:
        print("      no confident moves found; writing fixed copies anyway")

    spk0_fixed_path, spk1_fixed_path = rewrite_speaker_audio(
        spk0_path=spk0_path,
        spk1_path=spk1_path,
        moves=moves,
        output_dir=out_dir,
    )

    print(f"      spk0 fixed → {spk0_fixed_path}")
    print(f"      spk1 fixed → {spk1_fixed_path}\n")

    # ── 5. Second-pass transcription on fixed audio ──────────────────────
    print("[5/6] Second-pass WhisperX transcription on fixed audio ...")
    spk0_final_segs = transcribe_speaker(wx_model, spk0_fixed_path, device=device)
    spk1_final_segs = transcribe_speaker(wx_model, spk1_fixed_path, device=device)

    spk0_words = _extract_all_words(spk0_final_segs)
    spk1_words = _extract_all_words(spk1_final_segs)

    print(f"\n{'=' * 60}")
    print("  FINAL TRANSCRIPT")
    print(f"{'=' * 60}")
    print(_format_transcript(spk0_words, "Speaker 0"))
    print()
    print(_format_transcript(spk1_words, "Speaker 1"))
    print(f"{'=' * 60}\n")

    final_text = (
        _format_transcript(spk0_words, "Speaker 0")
        + "\n\n"
        + _format_transcript(spk1_words, "Speaker 1")
    )

    _save_text(final_text, out_dir / "final_transcript.txt")

    # ── 6. Translate ──────────────────────────────────────────────────────
    print(f"\n[6/6] Translating transcript → {target_lang} ...")
    translated_text = build_translated_transcript(
        spk0_words,
        spk1_words,
        target_lang=target_lang,
    )

    _save_text(translated_text, out_dir / f"final_transcript_{target_lang}.txt")
    


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end two-speaker multilingual transcription demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to the mixed two-speaker audio file (wav/mp3/flac/…).",
    )
    parser.add_argument(
        "--output-dir",
        default="demo/demo_output",
        help="Directory for intermediate and final outputs.",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper and TDNN inference.",
    )
    parser.add_argument(
        "--tdnn-weight",
        default=None,
        type=str,
        help=(
            "Path to TDNN checkpoint (.pth).  "
            "Defaults to dl_model/final_model/tdnn_full_best_acc.pth."
        ),
    )
    parser.add_argument(
        "--target-lang",
        default="en",
        choices=list(_LANG_MAP.keys()),
        help="Target language for the translated transcript.",
    )

    args = parser.parse_args()

    run_pipeline(
        audio_path=args.audio,
        output_dir=args.output_dir,
        whisper_model_name=args.whisper_model,
        device=args.device,
        tdnn_weight=args.tdnn_weight,
        target_lang=args.target_lang,
    )