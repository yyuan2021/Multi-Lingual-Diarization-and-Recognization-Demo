import argparse
from pathlib import Path
import sys

# Add project root to sys.path for module imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
import whisper

from dl_model.old.functions import SpeakerFeatureExtractor

# =========================
# 模型（不变）
# =========================
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SwitchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        l = self.enc(x[:, 0].unsqueeze(1))
        r = self.enc(x[:, 1].unsqueeze(1))
        d = torch.abs(l - r)
        return self.fc(torch.cat([l, r, d], dim=1))


# =========================
# 音频
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
# 语言检测（训练同款）
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
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ===== 模型 =====
    model = SwitchCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ===== 音频 =====
    wav, sr = load_audio(audio_path, sr=16000)

    # ===== Whisper =====
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

    # ===== load full audio（训练同款）=====
    audio_full = whisper.load_audio(str(audio_path))
    sr_whisper = whisper.audio.SAMPLE_RATE

    language_spans = []

    # ===== segment → 再跑 whisper(word) =====
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

        # ===== 构建 language_spans =====
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

    # ===== 找 switch =====
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

    # ===== 切窗口 =====
    seg1 = extract_window(wav, sr, switch_time - 1.0, switch_time)
    seg2 = extract_window(wav, sr, switch_time, switch_time + 1.0)

    # ===== embedding =====
    extractor = SpeakerFeatureExtractor(sr=16000)
    emb1 = extractor.extract_embedding(seg1)
    emb2 = extractor.extract_embedding(seg2)

    pair = np.stack([emb1, emb2], axis=0)
    x = torch.from_numpy(pair).unsqueeze(0).float().to(device)

    # ===== 推理 =====
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    label = "code_switch" if pred == 1 else "mix"

    print(f"Prediction: {label}")
    print(f"Probabilities: mix={probs[0]:.4f}, code_switch={probs[1]:.4f}")


if __name__ == "__main__":
    main()