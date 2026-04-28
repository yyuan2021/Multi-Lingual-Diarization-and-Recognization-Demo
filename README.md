# Web Demo — Two-Speaker Multilingual Speech Transcription

A single-page web app that separates two speakers from a mixed audio recording, transcribes each speaker with language tagging, and optionally translates to English.

---

## Features

- **Three input modes** — microphone recording, file upload, or built-in sample
- **Speaker separation** — isolates two speakers from a mixed audio track
- **Multilingual transcription** — word-level timestamps with per-word language tags (zh / en / hi)
- **Code-switch vs. cross-talk detection** — TDNN classifier distinguishes a single speaker switching languages from two speakers bleeding into each other's track
- **Audio repair** — misattributed segments are moved to the correct speaker before final transcription
- **zh → en translation** — Chinese transcripts are automatically translated via MarianMT

---

## Pipeline

```
Mixed audio
    │
    ▼
[1] MossFormer2 speaker separation
        → spk0.wav / spk1.wav  (8 kHz)
    │
    ▼
[2] WhisperX first-pass transcription
        → word-level timestamps + per-word language tags
    │
    ▼
[3] TDNN same-speaker check (on mixed-language segment boundaries)
        same speaker  → code-switch, keep as-is
        diff speaker  → cross-talk, schedule audio move
    │
    ▼
[4] Audio repair
        → spk0_fixed.wav / spk1_fixed.wav
    │
    ▼
[5] WhisperX second-pass transcription on fixed audio
    │
    ▼
[6] MarianMT translation
    │
    ▼
JSON response  {spk0_audio, spk1_audio, spk0_text, spk1_text, spk0_en, spk1_en}
```

> If no cross-talk is detected in step 3, steps 4–5 are skipped and the first-pass transcript is used directly.

---

## Setup

**Requirements:** Python 3.9+, and the following installed in your environment:

```bash
pip install fastapi uvicorn python-multipart
pip install modelscope resampy soundfile librosa
pip install whisperx
pip install transformers torch
```

**Run the server:**

```bash
uvicorn web_demo.backend:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

> The MarianMT translation model (`Helsinki-NLP/opus-mt-zh-en`) is downloaded from HuggingFace on first startup.

