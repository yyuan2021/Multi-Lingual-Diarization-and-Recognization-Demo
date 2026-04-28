import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
import whisper

from speechbrain.inference.speaker import SpeakerRecognition

# =========================
# 音频处理（完全一致）
# =========================
def load_audio(path: Path, sr=16000):
    wav, s = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if s != sr:
        wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    return wav.astype(np.float32), sr

def extract_window(wav: np.ndarray, sr: int, start_time: float, end_time: float):
    start_i = int(round(start_time * sr))
    end_i = int(round(end_time * sr))
    length = max(1, end_i - start_i)
    out = np.zeros(length, dtype=np.float32)

    src_start = max(0, start_i)
    src_end = min(len(wav), end_i)
    if src_end <= src_start:
        return out

    dst_start = src_start - start_i
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = wav[src_start:src_end]
    return out

# =========================
# 语言检测（完全一致）
# =========================
def detect_lang_by_char(ch: str):
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return "zh"
    if 0x0900 <= o <= 0x097F:
        return "hi"
    if "a" <= ch.lower() <= "z":
        return "en"
    return "other"

# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path.resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ===== 加载音频 =====
    wav, sr = load_audio(audio_path, sr=16000)

    # ===== Whisper（完全一致）=====
    whisper_model = whisper.load_model("base", device=device)

    asr_res = whisper_model.transcribe(
        str(audio_path),
        task="transcribe",
        language=None,
        verbose=False,
        fp16=False,
        word_timestamps=False,
    )

    print("\n=== ASR TEXT ===")
    print(asr_res.get("text", "").strip())

    segments = asr_res.get("segments", [])

    # ===== load full audio（完全一致）=====
    audio_full = whisper.load_audio(str(audio_path))
    sr_whisper = whisper.audio.SAMPLE_RATE

    language_spans = []

    # ===== segment → 再跑 whisper(word)（完全一致）=====
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        seg_audio = audio_full[int(seg_start * sr_whisper): int(seg_end * sr_whisper)]

        seg_result = whisper_model.transcribe(
            seg_audio,
            task="transcribe",
            language=None,
            verbose=False,
            fp16=False,
            temperature=0.2,
            beam_size=1,
            word_timestamps=True,
        )

        words = []
        for wseg in seg_result.get("segments", []):
            for w in wseg.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    continue
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": seg_start + w["start"],
                    "end": seg_start + w["end"],
                })

        # ===== 构建 language_spans（完全一致）=====
        cur = None

        def flush():
            nonlocal cur
            if cur:
                language_spans.append(cur)
                cur = None

        for w in words:
            if not w["word"]:
                continue

            lang = detect_lang_by_char(w["word"][0])

            if lang == "other":
                flush()
                continue

            if cur is None or cur["language"] != lang:
                flush()
                cur = {
                    "language": lang,
                    "start": w["start"],
                    "end": w["end"],
                }
            else:
                cur["end"] = w["end"]

        flush()

    # ===== 找 switch（完全一致）=====
    switch_time = None
    for i in range(1, len(language_spans)):
        if language_spans[i]["language"] != language_spans[i - 1]["language"]:
            switch_time = language_spans[i]["start"]
            break

    if switch_time is None:
        print("\nNo switch detected")
        print("Result: mix")
        return

    print(f"\nSwitch detected at: {switch_time:.3f} sec")

    # ===== 切窗口（完全一致）=====
    seg1 = extract_window(wav, sr, switch_time - 1.0, switch_time)
    seg2 = extract_window(wav, sr, switch_time, switch_time + 1.0)

    # ===== SpeechBrain（唯一替换部分）=====
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    wav1 = torch.tensor(seg1).unsqueeze(0).to(device)
    wav2 = torch.tensor(seg2).unsqueeze(0).to(device)

    with torch.no_grad():
        score, _ = verification.verify_batch(wav1, wav2)

    score = float(score.cpu().numpy()[0])

    # ===== 标签（按你定义）=====
    prob_code_switch = (score + 1) / 2   # same speaker
    prob_mix = 1 - prob_code_switch

    label = "code_switch" if prob_code_switch >= 0.5 else "mix"

    print(f"\nPrediction: {label}")
    print(f"Similarity score: {score:.4f}")
    print(f"Probabilities:")
    print(f"  code_switch (same speaker): {prob_code_switch:.4f}")
    print(f"  mix (different speaker): {prob_mix:.4f}")


if __name__ == "__main__":
    main()