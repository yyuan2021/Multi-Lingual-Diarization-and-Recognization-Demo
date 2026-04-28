"""
Batch evaluate SpeechBrain speaker verification on one or more CSV label files.

This script keeps the evaluation logic from `predict_eval.py` and copies all wrong
predictions into `dl_model/speechbrain/error` for later analysis.

默认读取两个 CSV:
- dl_model/baseline_train_test_segments.csv
- dl_model/baseline_train_test_segments_switchlingua_seame.csv
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.final_model.model import TDNNPredictor


def load_audio(path: Path, sr: int = 16000):
    wav, file_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    wav = wav.astype(np.float32)
    return np.expand_dims(wav, axis=0)


def parse_bool(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def resolve_audio_path(row, root: Path, test_dir: Path):
    audio_path = str(row.get("audio_path", "") or "").strip()
    if audio_path:
        candidate = Path(audio_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate
        if test_dir is not None:
            fallback = test_dir / candidate.name
            if fallback.exists():
                return fallback

    if test_dir is not None:
        index_keys = [row.get("test_row_index"), row.get("csv_index"), row.get("id"), row.get("index")]
        for key in index_keys:
            if key is None:
                continue
            try:
                idx = int(str(key).strip())
            except ValueError:
                continue
            candidate = test_dir / f"{idx}.wav"
            if candidate.exists():
                return candidate

    return None


def build_error_filename(csv_name: str, sample_index: int, audio_path: Path, true_label: int, pred_label: int, score: float):
    base = audio_path.stem
    suffix = f"_{sample_index:05d}_true{true_label}_pred{pred_label}_score{score:.4f}{audio_path.suffix}"
    return f"{csv_name}_{base}{suffix}"


def evaluate_single_csv(csv_path: Path, test_dir: Path, error_root: Path, verification, threshold: float, sr: int):
    print(f"\nEvaluating CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("split", "").strip().lower() == "test"]

    print(f"  Test rows: {len(rows)}")

    csv_errors = []
    all_scores = []
    all_preds = []
    all_labels = []

    csv_name = csv_path.stem
    out_dir = error_root / csv_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            audio_file = resolve_audio_path(row, csv_path.parent.parent, test_dir)
            if audio_file is None:
                print(f"Warning: cannot resolve audio for row {idx} in {csv_path}")
                continue
            if not audio_file.exists():
                print(f"Warning: audio file not found: {audio_file}")
                continue

            wav_full = load_audio(audio_file, sr=sr)
            T = wav_full.shape[1]
            mid = T // 2
            wav_left = wav_full[:, :mid]
            wav_right = wav_full[:, mid:]
            min_len = min(wav_left.shape[1], wav_right.shape[1])
            wav_left = wav_left[:, :min_len]
            wav_right = wav_right[:, :min_len]

            is_switch_pred, confidence = predictor.predict(wav_left.squeeze(0), wav_right.squeeze(0))
            score_float = float(confidence)
            pred_label = 1 if score_float > threshold else 0

            is_switch_val = row.get("is_switch")
            true_label = None
            if is_switch_val is not None:
                try:
                    true_label = int(str(is_switch_val).strip())
                except ValueError:
                    true_label = 1 if parse_bool(is_switch_val) else 0
            if true_label is None:
                raise ValueError(f"Cannot parse is_switch for row {idx} in {csv_path}: {is_switch_val}")

            all_preds.append(pred_label)
            all_labels.append(true_label)
            all_scores.append(score_float)

            if pred_label != true_label:
                dest_name = build_error_filename(csv_name, idx, audio_file, true_label, pred_label, score_float)
                dest_path = out_dir / dest_name
                shutil.copy2(audio_file, dest_path)
                csv_errors.append({
                    "row": idx,
                    "audio_path": str(audio_file),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "score": f"{score_float:.4f}",
                    "copied_to": str(dest_path),
                })

    if all_labels:
        total = len(all_labels)
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        accuracy = correct / total
        print(f"  Total evaluated: {total}")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
        print(f"  Errors copied: {len(csv_errors)} into {out_dir}")
    else:
        print("  No evaluated rows found.")

    if csv_errors:
        error_csv = error_root / f"{csv_name}_errors.csv"
        with open(error_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["row", "audio_path", "true_label", "pred_label", "score", "copied_to"])
            writer.writeheader()
            writer.writerows(csv_errors)
        print(f"  Error summary saved to: {error_csv}")

    return len(csv_errors)


def main():
    parser = argparse.ArgumentParser(description="Batch run final_model TDNN prediction errors and copy wrong files")
    parser.add_argument(
        "--csv",
        nargs="*",
        default=[
            "dl_model/baseline_train_test_segments.csv",
            "dl_model/baseline_train_test_segments_switchlingua_seame.csv",
        ],
        help="CSV label files to evaluate. 默认读取两个 CSV",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Optional fallback test directory if audio_path cannot be resolved directly.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Similarity threshold used to decide same/different speaker.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate to resample audio to.",
    )
    parser.add_argument(
        "--weight",
        default="dl_model/final_model/tdnn_full_best_acc.pth",
        help="Path to final_model weight file.",
    )
    parser.add_argument(
        "--error-dir",
        default="dl_model/speechbrain/error",
        help="Directory to save wrong predictions.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    error_root = repo_root / args.error_dir
    error_root.mkdir(parents=True, exist_ok=True)

    test_dir = None
    if args.test_dir:
        test_dir = repo_root / args.test_dir
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print(f"repo_root: {repo_root}")
    print(f"error_root: {error_root}")
    if test_dir is not None:
        print(f"fallback test_dir: {test_dir}")
    print(f"weight: {args.weight}")

    predictor = TDNNPredictor(device="cpu", weight_path=str(repo_root / args.weight))

    total_errors = 0
    for csv_arg in args.csv:
        csv_path = repo_root / csv_arg
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        total_errors += evaluate_single_csv(csv_path, test_dir, error_root, verification, args.threshold, args.sr)

    print(f"\nBatch complete. Total error files copied: {total_errors}")


if __name__ == "__main__":
    main()
