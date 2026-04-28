import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def resolve_case_insensitive(path: Path):
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
        for candidate in candidates:
            if candidate.name.lower() == part.lower():
                match = candidate
                break

        if match is None:
            return None

        current = match

    return current


def iter_switch_samples(item):
    segments = item.get("segments", [])

    for segment in segments:
        spans = segment.get("language_spans", [])
        if len(spans) < 2:
            continue

        for i in range(len(spans) - 1):
            left = spans[i]
            right = spans[i + 1]
            if left.get("language") == right.get("language"):
                continue

            yield {
                "segment_id": segment.get("segment_id"),
                "switch_index": i,
                "switch_time": float(left.get("end", 0.0)),
                "gap_start": float(left.get("end", 0.0)),
                "gap_end": float(right.get("start", left.get("end", 0.0))),
            }

    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]

        spans1 = seg1.get("language_spans", [])
        spans2 = seg2.get("language_spans", [])
        if (not spans1) or (not spans2):
            continue

        if spans1[-1].get("language") == spans2[0].get("language"):
            continue

        yield {
            "segment_id": f"{seg1.get('segment_id')}_{seg2.get('segment_id')}",
            "switch_index": i,
            "switch_time": float(seg1.get("end", 0.0)),
            "gap_start": float(seg1.get("end", 0.0)),
            "gap_end": float(seg2.get("start", seg1.get("end", 0.0))),
        }


def load_cache_records(cache_path: Path):
    cache_records = []
    json_order = []
    seen_json = set()
    audio_counter = Counter()

    with open(cache_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except Exception:
                continue

            audio_path = str(record["audio_path"]).replace("\\", "/")
            json_path = str(record["json_path"]).replace("\\", "/")
            audio_counter[audio_path] += 1

            cache_records.append(
                {
                    "cache_line_number": line_no,
                    "json_path": json_path,
                    "audio_path": audio_path,
                    "is_switch": bool(record["is_switch"]),
                    "audio_occurrence_in_cache": audio_counter[audio_path],
                }
            )

            if json_path not in seen_json:
                seen_json.add(json_path)
                json_order.append(json_path)

    return cache_records, json_order


def build_source_samples_by_json(root: Path, json_order, window_sec: float):
    samples_by_json = {}

    for json_rel in json_order:
        json_path = resolve_case_insensitive(root / json_rel)
        if json_path is None:
            raise FileNotFoundError(f"Cannot find source json: {json_rel}")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        samples = []
        audio_counter = Counter()

        for item in data:
            audio_path = str(item.get("path", "")).replace("\\", "/")
            if not audio_path:
                continue

            for sample in iter_switch_samples(item):
                audio_counter[audio_path] += 1
                switch_time = float(sample["switch_time"])
                samples.append(
                    {
                        "json_path": json_rel,
                        "audio_path": audio_path,
                        "segment_id": sample["segment_id"],
                        "switch_index": sample["switch_index"],
                        "switch_time": switch_time,
                        "window_sec": float(window_sec),
                        "gap_start": float(sample["gap_start"]),
                        "gap_end": float(sample["gap_end"]),
                        "left_start": switch_time - window_sec,
                        "left_end": switch_time,
                        "right_start": switch_time,
                        "right_end": switch_time + window_sec,
                        "audio_occurrence_in_source_json": audio_counter[audio_path],
                    }
                )

        samples_by_json[json_rel] = samples

    return samples_by_json


def align_cache_with_source(cache_records, samples_by_json):
    pointers = defaultdict(int)
    aligned = []

    for record in cache_records:
        json_path = record["json_path"]
        wanted_audio = record["audio_path"]
        samples = samples_by_json[json_path]

        idx = pointers[json_path]
        while idx < len(samples):
            if samples[idx]["audio_path"] == wanted_audio:
                break
            idx += 1

        if idx >= len(samples):
            raise RuntimeError(
                "Failed to align cache record with source sample: "
                f"cache_line={record['cache_line_number']} "
                f"json_path={json_path} audio_path={wanted_audio}"
            )

        source_sample = samples[idx]
        pointers[json_path] = idx + 1
        aligned.append({**record, **source_sample})

    return aligned


def split_train_test(records, seed=42):
    records = list(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    return records[: int(n * 0.8)], records[int(n * 0.8) :]


def balance(records):
    same = [r for r in records if not r["is_switch"]]
    switch = [r for r in records if r["is_switch"]]
    n = min(len(same), len(switch))
    balanced = same[:n] + switch[:n]
    random.shuffle(balanced)
    return balanced


def split_and_balance_records(aligned_records, seed=42):
    """划分 train/test 并分别平衡，返回带 split 标记的记录"""
    random.seed(seed)
    train_records, test_records = split_train_test(aligned_records, seed=seed)
    
    # 平衡
    train_balanced = balance(train_records)
    test_balanced = balance(test_records)
    
    # 添加 split 标记
    for r in train_balanced:
        r["split"] = "train"
    for r in test_balanced:
        r["split"] = "test"
    
    return train_balanced, test_balanced


def verify_against_test_paths(test_records, test_paths_csv: Path):
    with open(test_paths_csv, "r", encoding="utf-8") as f:
        expected_paths = [row["audio_path"] for row in csv.DictReader(f)]

    actual_paths = [row["audio_path"] for row in test_records]

    if actual_paths == expected_paths:
        return

    mismatch_index = None
    for i, (actual, expected) in enumerate(zip(actual_paths, expected_paths)):
        if actual == expected:
            continue
        mismatch_index = i
        break

    raise RuntimeError(
        "Generated test set does not match test_paths.csv. "
        f"first_mismatch_index={mismatch_index} "
        f"generated={actual_paths[mismatch_index]} "
        f"expected={expected_paths[mismatch_index]}"
    )


def write_baseline_csv(all_records, output_path: Path, root: Path, window_sec: float):
    """输出包含 train 和 test 的 CSV 文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "audio_path",         # 相对路径
        "is_switch",          # True / False
        "split",              # train / test
        "left_start",         # switch_time - window_sec
        "switch_time",        # 切换时间点（对齐 build_feature_json.py）
        "right_end",          # switch_time + window_sec
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in all_records:
            writer.writerow(
                {
                    "audio_path": record["audio_path"],
                    "is_switch": "True" if record["is_switch"] else "False",
                    "split": record["split"],
                    "left_start": f"{record['switch_time'] - window_sec:.6f}",
                    "switch_time": f"{record['switch_time']:.6f}",
                    "right_end": f"{record['switch_time'] + window_sec:.6f}",
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Export train and test segments used by train_net.py for baseline models."
    )
    parser.add_argument(
        "--cache",
        default="dl_model/mlp_feature_cache.jsonl",
        help="Feature cache generated for train_net.py",
    )
    parser.add_argument(
        "--test-paths",
        default="dl_model/test_paths.csv",
        help="Existing test_paths.csv to verify reproduced test order",
    )
    parser.add_argument(
        "--output",
        default="dl_model/baseline_train_test_segments.csv",
        help="Output CSV path (contains both train and test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by train_net.py",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Must match the --window-sec used by build_feature_json.py",
    )
    args = parser.parse_args()

    root = project_root()
    cache_path = root / args.cache
    test_paths_csv = root / args.test_paths
    output_path = root / args.output

    cache_records, json_order = load_cache_records(cache_path)
    source_samples_by_json = build_source_samples_by_json(
        root,
        json_order,
        window_sec=args.window_sec,
    )
    aligned_records = align_cache_with_source(cache_records, source_samples_by_json)
    
    # 划分 train/test 并平衡
    train_records, test_records = split_and_balance_records(aligned_records, seed=args.seed)
    
    # 验证 test 集
    verify_against_test_paths(test_records, test_paths_csv)
    
    # 合并所有记录并输出
    all_records = train_records + test_records
    write_baseline_csv(all_records, output_path, root, window_sec=args.window_sec)

    print(f"cache records       : {len(cache_records)}")
    print(f"aligned records     : {len(aligned_records)}")
    print(f"train records       : {len(train_records)}")
    print(f"test records        : {len(test_records)}")
    print(f"total records       : {len(all_records)}")
    print(f"window_sec          : {args.window_sec}")
    print(f"output csv          : {output_path.resolve()}")


if __name__ == "__main__":
    main()
