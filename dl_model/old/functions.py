import numpy as np
import soundfile as sf
import librosa
import whisper
import torch
import json
from pathlib import Path
import os


class SpeakerFeatureExtractor:
    def __init__(self, sr=16000, model_name="base"):
        self.sr = sr
        self.device = "cpu"
        self.model_name = model_name
        self.model = whisper.load_model(self.model_name, device=self.device)

    # =========================
    # Embedding
    # =========================
    def extract_embedding(self, wav: np.ndarray) -> np.ndarray:
        if len(wav) < int(0.05 * self.sr):
            return np.zeros(512, dtype=np.float32)

        wav = whisper.pad_or_trim(wav)
        mel = whisper.log_mel_spectrogram(wav).to(self.device)

        with torch.no_grad():
            enc = self.model.encoder(mel.unsqueeze(0))

        return enc.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)

    # =========================
    # Pitch
    # =========================
    def pitch_stats(self, seg: np.ndarray):
        if len(seg) < int(0.05 * self.sr):
            return 0.0, 0.0, 0.0

        f0 = librosa.yin(seg, fmin=50, fmax=400, sr=self.sr)
        f0 = f0[np.isfinite(f0)]
        if len(f0) == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(f0)), float(np.std(f0)), float(len(f0) / len(seg))

    def norm_diff(self, a, b):
        return abs(a - b) / (abs(a) + abs(b) + 1e-6)

    def extract_segment_features(self, seg: np.ndarray) -> dict:
        embedding = self.extract_embedding(seg)
        pitch_mean, pitch_std, voiced_ratio = self.pitch_stats(seg)
        duration = len(seg) / self.sr
        return {
            "embedding": embedding.tolist(),
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "voiced_ratio": float(voiced_ratio),
            "duration": float(duration),
        }

    # =========================
    # Core feature (8D)
    # numpy array
    # =========================
    def build_features(self, seg1: np.ndarray, seg2: np.ndarray,
                       t1_end: float = 1.0, t2_start: float = 1.0) -> np.ndarray:
        e1 = self.extract_embedding(seg1)
        e2 = self.extract_embedding(seg2)

        emb_cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        emb_l2 = np.linalg.norm(e1 - e2)

        emb_ratio = np.log(1 + emb_l2) * (1 - emb_cos)

        p1_mean, p1_std, v1 = self.pitch_stats(seg1)
        p2_mean, p2_std, v2 = self.pitch_stats(seg2)

        pitch_mean_diff = self.norm_diff(p1_mean, p2_mean)
        pitch_std_diff = self.norm_diff(p1_std, p2_std)
        voiced_diff = self.norm_diff(v1, v2)

        dur1 = len(seg1) / self.sr
        dur2 = len(seg2) / self.sr
        duration_diff = self.norm_diff(dur1, dur2)

        gap = max(0.0, t2_start - t1_end)
        time_gap = np.exp(-gap)

        feat = np.array([
            emb_cos,
            emb_l2,
            emb_ratio,
            pitch_mean_diff,
            pitch_std_diff,
            voiced_diff,
            duration_diff,
            time_gap
        ], dtype=np.float32)

        return feat

    def build_raw_features(self, seg1: np.ndarray, seg2: np.ndarray,
                           t1_end: float = 1.0, t2_start: float = 1.0) -> dict:
        gap = max(0.0, t2_start - t1_end)
        return {
            "left": self.extract_segment_features(seg1),
            "right": self.extract_segment_features(seg2),
            "time_gap_seconds": float(gap),
        }

    # =========================
    # dict for json.dumps
    # =========================
    def build_json(self, seg1: np.ndarray, seg2: np.ndarray,
                   label=None, meta: dict = None,
                   t1_end: float = 1.0, t2_start: float = 1.0) -> dict:
        feat = self.build_features(seg1, seg2, t1_end=t1_end, t2_start=t2_start)
        data = {
            "feature": feat.tolist()
        }
        if label is not None:
            data["label"] = int(label)
        if meta is not None:
            data["meta"] = meta
        return data

    # =========================
    # JSON string
    # =========================
    def build_json_str(self, seg1: np.ndarray, seg2: np.ndarray,
                       label=None, meta: dict = None,
                       t1_end: float = 1.0, t2_start: float = 1.0) -> str:
        data = self.build_json(seg1, seg2, label=label, meta=meta,
                               t1_end=t1_end, t2_start=t2_start)
        return json.dumps(data, ensure_ascii=False)


def _project_root():
    return Path(__file__).resolve().parents[1]


def _resolve_audio_path(path_like):
    path = Path(path_like)
    if path.exists():
        return path
    root = _project_root()
    candidates = [
        root / path_like,
        root / "samples" / path.name,
        root / "preprocess" / path.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_1s_segment_from_file(path_like, start=0.0, sr=16000):
    p = _resolve_audio_path(path_like)
    if p is None:
        return None
    wav, s = sf.read(str(p))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if s != sr:
        wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    start_i = int(start * sr)
    end_i = start_i + sr
    if end_i > len(wav):
        seg = np.zeros(sr, dtype=np.float32)
        seg[:max(0, len(wav) - start_i)] = wav[start_i:len(wav)]
        return seg
    return wav[start_i:end_i].astype(np.float32)


def synth_1s_sine(freq=220.0, sr=16000):
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


if __name__ == "__main__":
    print("cwd:", os.getcwd())
    extractor = SpeakerFeatureExtractor(sr=16000)

    a1 = load_1s_segment_from_file("speaker_en.wav", start=1.0, sr=16000)
    a2 = load_1s_segment_from_file("speaker_en.wav", start=2.0, sr=16000)
    b1 = load_1s_segment_from_file("same_voice_two_languages.wav", start=0.0, sr=16000)
    b2 = load_1s_segment_from_file("same_voice_two_languages.wav", start=3.0, sr=16000)
    c1 = load_1s_segment_from_file("two_languages.wav", start=0.0, sr=16000)
    c2 = load_1s_segment_from_file("two_languages.wav", start=3.0, sr=16000)

    prints = []
    js = extractor.build_json_str(a1, a2, label=1, meta={"case": "same_speaker"})
    print("\n=== same speaker JSON ===")
    print(js)
    prints.append(json.loads(js))

    js = extractor.build_json_str(b1, b2, label=1, meta={"case": "same_speaker_diff_lang"})
    print("\n=== same speaker diff lang JSON ===")
    print(js)
    prints.append(json.loads(js))

    js = extractor.build_json_str(c1, c2, label=0, meta={"case": "diff_speaker"})
    print("\n=== diff speaker JSON ===")
    print(js)
    prints.append(json.loads(js))

    print("\n=== parsed features ===")
    for item in prints:
        print(item["meta"]["case"], "->", item["feature"])
