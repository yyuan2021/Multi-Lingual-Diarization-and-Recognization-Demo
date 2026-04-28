"""
从 CSV 读取基线模型的 train/test 片段信息，提取 2 秒音频片段。
每个片段严格保证中间 1 秒处为切换点，长度不足则补 0。
输出到 datasets/mlp_train/train 和 datasets/mlp_train/test 目录。
文件名为 CSV 行号（从 1 开始）。
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm

try:
    import librosa  # type: ignore
except ModuleNotFoundError:
    librosa = None


def project_root():
    return Path(__file__).resolve().parents[1]


def resolve_case_insensitive(path: Path):
    """大小写不敏感地解析路径"""
    if path.exists():
        return path

    parts = path.parts
    current = Path(parts[0]) if path.is_absolute() else Path()

    for part in parts:
        if not current.exists():
            return None

        try:
            candidates = list(current.iterdir())
        except Exception:
            return None

        match = None
        for c in candidates:
            if c.name.lower() == part.lower():
                match = c
                break

        if match is None:
            return None

        current = match

    return current


def load_audio(path: Path, sr=16000):
    """加载音频并转换为单声道 16kHz"""
    wav, src_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if src_sr != sr:
        if librosa is not None:
            wav = librosa.resample(wav, orig_sr=src_sr, target_sr=sr)
        else:
            gcd = np.gcd(src_sr, sr)
            up = sr // gcd
            down = src_sr // gcd
            wav = signal.resample_poly(wav, up=up, down=down)
    return wav.astype(np.float32)


def extract_segment(wav: np.ndarray, sr: int, center_time: float, window_sec: float = 1.0):
    """
    提取 2 秒音频片段，center_time 为切换点（位于片段正中间 1 秒处）。
    片段范围：[center_time - window_sec, center_time + window_sec]
    如果超出音频边界则补 0。
    """
    start_time = center_time - window_sec
    end_time = center_time + window_sec

    start_i = int(round(start_time * sr))
    end_i = int(round(end_time * sr))
    segment_length = end_i - start_i  # 应为 2 * sr

    out = np.zeros(segment_length, dtype=np.float32)

    src_start = max(0, start_i)
    src_end = min(len(wav), end_i)

    if src_end <= src_start:
        return out

    dst_start = src_start - start_i
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = wav[src_start:src_end]

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2-second audio segments for baseline model training."
    )
    parser.add_argument(
        "--csv",
        default="dl_model/baseline_train_test_segments.csv",
        help="Input CSV file with segment information",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/mlp_train",
        help="Output directory (will create train/ and test/ subdirs)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate (default: 16000)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window size on each side of switch_time (default: 1.0, total 2.0s)",
    )
    args = parser.parse_args()

    root = project_root()
    csv_path = root / args.csv
    output_base = root / args.output_dir
    sr = args.sr
    window_sec = args.window_sec

    # 创建输出目录
    train_dir = output_base / "train"
    test_dir = output_base / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 读取 CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} segments...")
    print(f"Output directory: {output_base.resolve()}")
    print(f"Sample rate: {sr} Hz")
    print(f"Window: {window_sec}s per side (total {2 * window_sec}s)")

    processed = 0
    errors = 0

    for idx, row in tqdm(enumerate(rows, 1), desc="Extracting", total=len(rows)):
        audio_rel_path = row["audio_path"]
        is_switch = row["is_switch"].lower() == "true"
        split = row["split"].lower()
        switch_time = float(row["switch_time"])

        # 确定输出目录
        out_dir = train_dir if split == "train" else test_dir
        out_path = out_dir / f"{idx}.wav"

        # 解析音频路径
        audio_path = resolve_case_insensitive(root / audio_rel_path)
        if audio_path is None:
            tqdm.write(f"[WARN] Audio not found: {audio_rel_path}")
            errors += 1
            continue

        try:
            wav = load_audio(audio_path, sr=sr)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to load {audio_rel_path}: {e}")
            errors += 1
            continue

        # 提取片段
        segment = extract_segment(wav, sr, switch_time, window_sec=window_sec)

        # 保存
        sf.write(str(out_path), segment, sr)
        processed += 1

    print(f"\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Train files: {len(list(train_dir.glob('*.wav')))}")
    print(f"  Test files: {len(list(test_dir.glob('*.wav')))}")
    print(f"  Train dir: {train_dir}")
    print(f"  Test dir: {test_dir}")


if __name__ == "__main__":
    main()
