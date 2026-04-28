from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


@dataclass
class EvalSample:
    test_row_index: int
    audio_path: str
    audio_abs_path: Path
    label: int
    left_start: float
    left_end: float
    right_start: float
    right_end: float


@dataclass
class SegmentPair:
    sample: EvalSample
    left_audio: np.ndarray
    right_audio: np.ndarray
    sample_rate: int


@dataclass
class PredictionResult:
    prediction: int
    same_speaker_score: float
    raw_score: float


@dataclass
class DatasetReport:
    csv_rows: int
    available_rows: int
    missing_rows: int
    positives: int
    negatives: int
    missing_examples: List[str]


class BaseSpeakerBaseline(ABC):
    model_name = "base_model"
    target_sample_rate = 16000

    def __init__(self, device: str = "cpu", cache_dir: Optional[Path] = None):
        self.device = device
        self.cache_dir = cache_dir

    @abstractmethod
    def predict(
        self, left_audio: np.ndarray, right_audio: np.ndarray, sample_rate: int
    ) -> PredictionResult:
        """Return same-speaker prediction for two waveform segments."""


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(labels: List[int], predictions: List[int]) -> Dict[str, Optional[float]]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")

    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)

    positives = sum(labels)
    negatives = len(labels) - positives

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = 0.5 * (recall + specificity)

    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "sample_count": len(labels),
        "positives": positives,
        "negatives": negatives,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(labels)),
        "positive_accuracy": safe_div(tp, positives),
        "negative_accuracy": safe_div(tn, negatives),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


class AudioPathResolver:
    def __init__(self, root: Path):
        self.root = root
        self._filename_cache: Dict[str, Optional[Path]] = {}

    def resolve(self, audio_path: str, audio_abs_path: Optional[str] = None) -> Optional[Path]:
        candidates: List[Path] = []
        if audio_abs_path:
            candidates.append(Path(audio_abs_path))
        candidates.append(self.root / audio_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        file_name = Path(audio_path).name
        if file_name not in self._filename_cache:
            matches = list((self.root / "datasets").glob(f"**/{file_name}"))
            self._filename_cache[file_name] = matches[0].resolve() if matches else None

        return self._filename_cache[file_name]


def load_audio(path: Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, source_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if source_sr != sr:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sr)
    return wav.astype(np.float32), sr


def extract_window(wav: np.ndarray, sr: int, start_time: float, end_time: float) -> np.ndarray:
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


def load_eval_samples(
    csv_path: Path, root: Path, max_samples: Optional[int] = None
) -> Tuple[List[EvalSample], DatasetReport]:
    resolver = AudioPathResolver(root)
    samples: List[EvalSample] = []
    missing_examples: List[str] = []
    csv_rows = 0
    missing_rows = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_rows += 1
            resolved = resolver.resolve(row["audio_path"], row.get("audio_abs_path"))
            if resolved is None:
                missing_rows += 1
                if len(missing_examples) < 10:
                    missing_examples.append(row["audio_path"])
                continue

            sample = EvalSample(
                test_row_index=int(row["test_row_index"]),
                audio_path=row["audio_path"],
                audio_abs_path=resolved,
                label=int(row["is_switch"]),
                left_start=float(row["left_start"]),
                left_end=float(row["left_end"]),
                right_start=float(row["right_start"]),
                right_end=float(row["right_end"]),
            )
            samples.append(sample)

            if max_samples is not None and len(samples) >= max_samples:
                break

    positives = sum(sample.label for sample in samples)
    negatives = len(samples) - positives
    report = DatasetReport(
        csv_rows=csv_rows,
        available_rows=len(samples),
        missing_rows=missing_rows,
        positives=positives,
        negatives=negatives,
        missing_examples=missing_examples,
    )
    return samples, report


def preload_segment_pairs(
    samples: List[EvalSample], target_sr: int = 16000
) -> List[SegmentPair]:
    audio_cache: Dict[Path, Tuple[np.ndarray, int]] = {}
    pairs: List[SegmentPair] = []

    for sample in samples:
        if sample.audio_abs_path not in audio_cache:
            audio_cache[sample.audio_abs_path] = load_audio(sample.audio_abs_path, sr=target_sr)

        wav, sr = audio_cache[sample.audio_abs_path]
        left_audio = extract_window(wav, sr, sample.left_start, sample.left_end)
        right_audio = extract_window(wav, sr, sample.right_start, sample.right_end)
        pairs.append(
            SegmentPair(
                sample=sample,
                left_audio=left_audio,
                right_audio=right_audio,
                sample_rate=sr,
            )
        )

    return pairs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)
