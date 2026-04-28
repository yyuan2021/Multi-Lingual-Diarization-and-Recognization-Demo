from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import soundfile as sf
import librosa
import whisper

from dl_model.old.functions import SpeakerFeatureExtractor


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


def detect_lang_by_char(ch: str):
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return "zh"
    if 0x0900 <= o <= 0x097F:
        return "hi"
    if "a" <= ch.lower() <= "z":
        return "en"
    return "other"


class MLPWhisperSpeakerPredictor:
    def __init__(
        self,
        model_path: Path | None = None,
        model_class=None,
        device: str | None = None,
        sample_rate: int = 16000,
        window_sec: float = 1.0,
    ):
        if model_class is None:
            from dl_model.old.mlp.model1 import MLPModel1

            model_class = MLPModel1

        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path or (_project_root / "dl_model" / "checkpoints" / "MLPModel1_best.pth"))

        self.model = model_class().to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.extractor = SpeakerFeatureExtractor(sr=self.sample_rate, model_name="base")
        self._switch_detector = None

    def _prepare_audio(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        out = np.asarray(wav, dtype=np.float32).squeeze()
        if sample_rate != self.sample_rate:
            out = librosa.resample(out, orig_sr=sample_rate, target_sr=self.sample_rate)
        return out.astype(np.float32)

    def predict_pair(self, left_audio: np.ndarray, right_audio: np.ndarray, sample_rate: int = 16000) -> dict[str, Any]:
        left_audio = self._prepare_audio(left_audio, sample_rate)
        right_audio = self._prepare_audio(right_audio, sample_rate)

        emb1 = self.extractor.extract_embedding(left_audio)
        emb2 = self.extractor.extract_embedding(right_audio)

        pair = np.stack([emb1, emb2], axis=0)
        x = torch.from_numpy(pair).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred = int(np.argmax(probs))
        raw_score = float((logits[0, 1] - logits[0, 0]).detach().cpu().item())

        return {
            "prediction": pred,
            "same_speaker_score": float(probs[1]),
            "raw_score": raw_score,
            "probabilities": probs.tolist(),
        }

    def _load_switch_detector(self):
        if self._switch_detector is None:
            self._switch_detector = whisper.load_model("large-v3", device=self.device)
        return self._switch_detector

    def _build_language_spans(self, audio_path: Path) -> tuple[list[dict[str, Any]], str]:
        whisper_model = self._load_switch_detector()
        audio_full = whisper.load_audio(str(audio_path))
        sr_whisper = whisper.audio.SAMPLE_RATE

        base_result = whisper_model.transcribe(
            str(audio_path),
            task="transcribe",
            language=None,
            verbose=False,
            fp16=False,
            word_timestamps=False,
        )

        segments = base_result.get("segments", [])
        language_spans = []

        for seg in segments:
            seg_start = seg.get("start")
            seg_end = seg.get("end")

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

            seg_language = seg_result.get("language", "unknown")

            words = []
            for wseg in seg_result.get("segments", []):
                for w in wseg.get("words", []):
                    if w.get("start") is None or w.get("end") is None:
                        continue
                    words.append(
                        {
                            "word": w.get("word", "").strip(),
                            "start": seg_start + w["start"],
                            "end": seg_start + w["end"],
                            "score": w.get("probability", 0.0),
                        }
                    )

            cur = None

            def flush():
                nonlocal cur
                if cur:
                    cur["text"] = cur["text"].strip()
                    cur["score"] = cur["score_sum"] / max(cur["count"], 1)
                    cur.pop("score_sum")
                    cur.pop("count")
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
                        "text": w["word"],
                        "score_sum": w["score"] or 0.0,
                        "count": 1,
                    }
                else:
                    cur["end"] = w["end"]
                    cur["text"] += " " + w["word"]
                    cur["score_sum"] += w["score"] or 0.0
                    cur["count"] += 1

            flush()

            if not language_spans:
                language_spans.append(
                    {
                        "language": seg_language,
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_result.get("text", "").strip(),
                        "score": seg.get("avg_logprob"),
                    }
                )

        return language_spans, base_result.get("text", "").strip()

    def _select_switch(self, language_spans: list[dict[str, Any]]) -> dict[str, Any] | None:
        if len(language_spans) < 2:
            return None

        switches = []
        for i in range(1, len(language_spans)):
            if language_spans[i]["language"] != language_spans[i - 1]["language"]:
                before_dur = language_spans[i - 1]["end"] - language_spans[i - 1]["start"]
                after_dur = language_spans[i]["end"] - language_spans[i]["start"]
                switches.append(
                    {
                        "switch_time": language_spans[i - 1]["end"],
                        "from_lang": language_spans[i - 1]["language"],
                        "to_lang": language_spans[i]["language"],
                        "from_start": language_spans[i - 1]["start"],
                        "to_end": language_spans[i]["end"],
                        "before_dur": before_dur,
                        "after_dur": after_dur,
                    }
                )

        if not switches:
            return None

        valid_switches = [
            sw
            for sw in switches
            if sw["before_dur"] >= self.window_sec * 0.5 and sw["after_dur"] >= self.window_sec * 0.5
        ]

        if valid_switches:
            return max(valid_switches, key=lambda s: s["before_dur"] + s["after_dur"])
        return switches[0]

    def predict_audio(self, audio_path: Path) -> dict[str, Any] | None:
        wav, sr = load_audio(audio_path, sr=self.sample_rate)
        language_spans, asr_text = self._build_language_spans(audio_path)
        switch = self._select_switch(language_spans)

        if switch is None:
            print("\n=== ASR TEXT ===")
            print(asr_text)
            print("\nNo switch detected")
            print("Result: mix")
            return None

        switch_time = switch["switch_time"]
        print("device:", self.device)
        print("\n=== ASR TEXT ===")
        print(asr_text)
        print(f"\nSwitch detected at: {switch_time:.3f} sec (window={self.window_sec}s)")

        seg1 = extract_window(wav, sr, switch_time - self.window_sec, switch_time)
        seg2 = extract_window(wav, sr, switch_time, switch_time + self.window_sec)

        pair_result = self.predict_pair(seg1, seg2, sample_rate=sr)
        label = "code_switch" if pair_result["prediction"] == 1 else "mix"

        print(f"Prediction: {label}")
        print(
            f"Probabilities: mix={pair_result['probabilities'][0]:.4f}, "
            f"code_switch={pair_result['probabilities'][1]:.4f}"
        )

        return {
            "switch_time": switch_time,
            "prediction": label,
            "probabilities": pair_result["probabilities"],
            "same_speaker_score": pair_result["same_speaker_score"],
            "raw_score": pair_result["raw_score"],
        }


def predict(audio_path: Path, model_path: Path, model_class, device=None, window_sec: float = 1.0):
    # 固定随机种子
    torch.manual_seed(0)
    np.random.seed(0)

    predictor = MLPWhisperSpeakerPredictor(
        model_path=model_path,
        model_class=model_class,
        device=device,
        window_sec=window_sec,
    )
    return predictor.predict_audio(audio_path)


if __name__ == "__main__":
    from dl_model.old.mlp.model1 import MLPModel1

    audio_path = Path(r"samples\output_spk0.wav")
    model_path = Path(r"dl_model/checkpoints/MLPModel1_best.pth")
    model_class = MLPModel1

    predict(audio_path, model_path, model_class)
