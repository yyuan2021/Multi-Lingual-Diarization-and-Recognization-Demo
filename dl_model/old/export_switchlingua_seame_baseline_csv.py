import argparse
import csv
import json
import random
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def iter_switch_samples(item: dict):
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
                "audio_path": str(item.get("path", "")).replace("\\", "/"),
                "switch_time": float(left.get("end", 0.0)),
                "source": "natural",
            }

    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        spans1 = seg1.get("language_spans", [])
        spans2 = seg2.get("language_spans", [])
        if not spans1 or not spans2:
            continue
        if spans1[-1].get("language") == spans2[0].get("language"):
            continue

        yield {
            "audio_path": str(item.get("path", "")).replace("\\", "/"),
            "switch_time": float(seg1.get("end", 0.0)),
            "source": "natural",
        }


def load_json_list(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_positive_records(root: Path) -> list[dict]:
    positive_jsons = [
        root / "datasets" / "seame_dev_sge_cn_en_2000" / "whisper_segment_seame_dev_sge_cn_en_2000_en_zh_only.json",
        root / "datasets" / "switchlingua_audio_2000" / "whisper_segment_switchlingua_audio_2000_en_hi_only.json",
    ]

    out = []
    for json_path in positive_jsons:
        data = load_json_list(json_path)
        for item in data:
            for sample in iter_switch_samples(item):
                out.append(
                    {
                        "audio_path": sample["audio_path"],
                        "is_switch": True,
                        "switch_time": float(sample["switch_time"]),
                        "dataset": json_path.stem,
                    }
                )
    return out


def load_negative_records(root: Path) -> list[dict]:
    true_only_path = (
        root
        / "datasets"
        / "crossfade_switchlingua_seame"
        / "whisper_language_switch_crossfade_switchlingua_seame_true_only.json"
    )
    data = load_json_list(true_only_path)

    out = []
    for item in data:
        audio_path = str(item.get("path", "")).replace("\\", "/")
        for sw in item.get("true_insert_switches", []):
            expected_time = sw.get("expected_time")
            if not isinstance(expected_time, (int, float)):
                continue
            out.append(
                {
                    "audio_path": audio_path,
                    "is_switch": False,
                    "switch_time": float(expected_time),
                    "dataset": "crossfade_switchlingua_seame",
                }
            )
    return out


def balance(records: list[dict], seed: int) -> list[dict]:
    same = [r for r in records if not r["is_switch"]]
    switch = [r for r in records if r["is_switch"]]
    n = min(len(same), len(switch))
    rng = random.Random(seed)
    rng.shuffle(same)
    rng.shuffle(switch)
    balanced = same[:n] + switch[:n]
    rng.shuffle(balanced)
    return balanced


def assign_split(records: list[dict], split: str) -> list[dict]:
    for record in records:
        record["split"] = split
    return records


def write_csv(records: list[dict], output_path: Path, window_sec: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_row_index",
                "audio_path",
                "is_switch",
                "split",
                "left_start",
                "left_end",
                "switch_time",
                "right_start",
                "right_end",
            ],
        )
        writer.writeheader()
        for idx, record in enumerate(records, start=1):
            switch_time = float(record["switch_time"])
            writer.writerow(
                {
                    "test_row_index": idx,
                    "audio_path": record["audio_path"],
                    "is_switch": 1 if record["is_switch"] else 0,
                    "split": record["split"],
                    "left_start": f"{switch_time - window_sec:.6f}",
                    "left_end": f"{switch_time:.6f}",
                    "switch_time": f"{switch_time:.6f}",
                    "right_start": f"{switch_time:.6f}",
                    "right_end": f"{switch_time + window_sec:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export baseline CSV for seame/switchlingua natural switches and crossfade negatives."
    )
    parser.add_argument(
        "--output",
        default="dl_model/baseline_train_test_segments_switchlingua_seame.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=1.0)
    args = parser.parse_args()

    root = project_root()
    positive_records = load_positive_records(root)
    negative_records = load_negative_records(root)
    all_records = positive_records + negative_records

    final_records = assign_split(balance(all_records, seed=args.seed), "test")

    write_csv(final_records, root / args.output, window_sec=float(args.window_sec))

    print(f"positive_records : {len(positive_records)}")
    print(f"negative_records : {len(negative_records)}")
    print(f"test_balanced    : {len(final_records)}")
    print(f"total_written    : {len(final_records)}")
    print(f"output           : {(root / args.output).as_posix()}")


if __name__ == "__main__":
    main()
