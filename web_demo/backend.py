import sys
import uuid
import wave
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo.demo_pipeline import run_pipeline  # noqa: E402

OUTPUT_DIR     = Path("demo/demo_output")
UPLOAD_DIR     = Path("web_demo/uploads")
SAMPLE_OUT_DIR = Path("web_demo/sample_output")


def _ensure_sample():
    sample = Path("web_demo/samples/mixed.wav")
    if not sample.exists():
        sample.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(sample), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000 * 3)


@asynccontextmanager
async def lifespan(app):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_sample()
    yield


app = FastAPI(title="Speaker Diarization Demo", lifespan=lifespan)
app.mount("/outputs",       StaticFiles(directory="demo/demo_output"),       name="outputs")
app.mount("/sample_output", StaticFiles(directory="web_demo/sample_output"), name="sample_output")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_transcript(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("[Speaker 0]:")
    if len(parts) < 2:
        return "", ""
    spk_parts = parts[1].split("[Speaker 1]:")

    def _clean(block: str) -> str:
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("<") and ">" in line:
                line = line[line.index(">") + 1:].strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    spk0 = _clean(spk_parts[0])
    spk1 = _clean(spk_parts[1]) if len(spk_parts) > 1 else ""
    return spk0, spk1


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

from transformers import MarianMTModel, MarianTokenizer  # noqa: E402

_MT_NAME      = "Helsinki-NLP/opus-mt-zh-en"
_mt_tokenizer = MarianTokenizer.from_pretrained(_MT_NAME)
_mt_model     = MarianMTModel.from_pretrained(_MT_NAME)


def translate_text(text: str) -> str:
    if not text or not any("一" <= c <= "鿿" for c in text):
        return text
    inputs     = _mt_tokenizer([text], return_tensors="pt", padding=True)
    translated = _mt_model.generate(**inputs)
    return _mt_tokenizer.decode(translated[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Async job store
# ---------------------------------------------------------------------------

JOBS: dict = {}


def _step(job_id: str, step: int, message: str) -> None:
    JOBS[job_id]["step"]    = step
    JOBS[job_id]["message"] = message


def _run_job(job_id: str, audio_path: str, target_lang: str) -> None:
    try:
        from demo.demo_pipeline import (
            separate_speakers, transcribe_speaker,
            check_mixed_segments, build_repair_plan, rewrite_speaker_audio,
            _extract_all_words, _format_transcript,
        )
        from dl_model.final_model.model import TDNNPredictor
        import whisperx                           # noqa: E402

        audio_path_obj = Path(audio_path).resolve()
        out_dir        = OUTPUT_DIR
        device         = "cpu"
        compute_type   = "int8"

        # ── 1. Speaker separation ────────────────────────────────────────
        _step(job_id, 1, "[1/6] Separating speakers…")
        spk0_path, spk1_path = separate_speakers(audio_path_obj, out_dir)

        # ── 2. First-pass transcription ──────────────────────────────────
        _step(job_id, 2, "[2/6] Transcribing speakers (first pass)…")
        wx_model  = whisperx.load_model("small", device=device, compute_type=compute_type)
        spk0_segs = transcribe_speaker(wx_model, spk0_path, device=device)
        spk1_segs = transcribe_speaker(wx_model, spk1_path, device=device)
        spk0_w    = _extract_all_words(spk0_segs)
        spk1_w    = _extract_all_words(spk1_segs)

        # ── 3. TDNN same-speaker check ───────────────────────────────────
        _step(job_id, 3, "[3/6] Checking mixed-language spans (TDNN)…")
        predictor = TDNNPredictor(
            device=device,
            weight_path="dl_model/final_model/tdnn_full_best_acc.pth",
        )
        cross0    = check_mixed_segments(spk0_segs, spk0_path, predictor)
        cross1    = check_mixed_segments(spk1_segs, spk1_path, predictor)
        has_xtalk = any(not r["is_same_speaker"] for r in cross0 + cross1)

        spk0_audio = "/outputs/spk0.wav"
        spk1_audio = "/outputs/spk1.wav"

        if has_xtalk:
            # ── 4. Rewrite audio ─────────────────────────────────────────
            _step(job_id, 4, "[4/6] Rewriting speaker audio…")
            moves = build_repair_plan(cross0, cross1)
            spk0_fx, spk1_fx = rewrite_speaker_audio(spk0_path, spk1_path, moves, out_dir)
            spk0_audio = "/outputs/spk0_fixed.wav"
            spk1_audio = "/outputs/spk1_fixed.wav"

            # ── 5. Second-pass transcription ─────────────────────────────
            _step(job_id, 5, "[5/6] Transcribing fixed audio (second pass)…")
            spk0_segs = transcribe_speaker(wx_model, spk0_fx, device=device)
            spk1_segs = transcribe_speaker(wx_model, spk1_fx, device=device)
            spk0_w    = _extract_all_words(spk0_segs)
            spk1_w    = _extract_all_words(spk1_segs)
        else:
            _step(job_id, 5, "[5/6] No cross-talk — using first-pass transcript")

        # Write final transcript
        final_text = (
            _format_transcript(spk0_w, "Speaker 0")
            + "\n\n"
            + _format_transcript(spk1_w, "Speaker 1")
        )
        (out_dir / "final_transcript.txt").write_text(final_text, encoding="utf-8")
        spk0_text, spk1_text = _parse_transcript(out_dir / "final_transcript.txt")

        # ── 6. Translation ───────────────────────────────────────────────
        _step(job_id, 6, "[6/6] Translating transcript…")
        spk0_en = translate_text(spk0_text)
        spk1_en = translate_text(spk1_text)

        JOBS[job_id]["done"]   = True
        JOBS[job_id]["result"] = {
            "spk0_audio": spk0_audio,
            "spk1_audio": spk1_audio,
            "spk0_text":  spk0_text,
            "spk1_text":  spk1_text,
            "spk0_en":    spk0_en,
            "spk1_en":    spk1_en,
        }

    except Exception as exc:
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["done"]  = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/sample")
async def get_sample(target_lang: str = "en"):
    try:
        spk0_text, spk1_text = _parse_transcript(SAMPLE_OUT_DIR / "final_transcript.txt")
        trans_path = SAMPLE_OUT_DIR / f"final_transcript_{target_lang}.txt"
        if not trans_path.exists():
            trans_path = SAMPLE_OUT_DIR / "final_transcript_en.txt"
        spk0_en, spk1_en = _parse_transcript(trans_path)
        return JSONResponse({
            "spk0_audio": "/sample_output/spk0_fixed.wav",
            "spk1_audio": "/sample_output/spk1_fixed.wav",
            "spk0_text":  spk0_text,
            "spk1_text":  spk1_text,
            "spk0_en":    spk0_en,
            "spk1_en":    spk1_en,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/process")
async def process_audio(audio: UploadFile = File(...), target_lang: str = Form(default="en")):
    suffix   = Path(audio.filename).suffix or ".wav"
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"step": 0, "message": "Queued", "done": False, "error": None, "result": None}
    threading.Thread(target=_run_job, args=(job_id, str(tmp_path), target_lang), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "step":    job["step"],
        "message": job["message"],
        "done":    job["done"],
        "error":   job["error"],
        "result":  job["result"],
    })


# Serve static files — must be last
app.mount("/", StaticFiles(directory="web_demo", html=True), name="static")
