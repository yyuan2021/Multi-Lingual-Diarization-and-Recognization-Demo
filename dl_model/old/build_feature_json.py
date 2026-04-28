import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from dl_model.old.functions import SpeakerFeatureExtractor


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
        for c in candidates:
            if c.name.lower() == part.lower():
                match = c
                break

        if match is None:
            return None

        current = match

    return current


def iter_audio_path_candidates(path: Path):
    yield path

    path_str = path.as_posix()
    hinglish_rewrites = [
        ("/datasets/hinglish/data/train/", "/datasets/hinglish/data/train/train/"),
        ("/datasets/hinglish/data/test/", "/datasets/hinglish/data/test/test/"),
    ]

    for source, target in hinglish_rewrites:
        if source in path_str and target not in path_str:
            yield Path(path_str.replace(source, target, 1))


def resolve_audio_path(path: Path):
    for candidate in iter_audio_path_candidates(path):
        resolved = resolve_case_insensitive(candidate)
        if resolved is not None:
            return resolved
    return None


def load_used_json_list(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip().replace("\\", "/") for line in lines if line.strip()]


def load_audio(path: Path, sr=16000):
    wav, src_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if src_sr != sr:
        wav = librosa.resample(wav, orig_sr=src_sr, target_sr=sr)
    return wav.astype(np.float32)


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


def iter_switch_samples(item):
    segments = item.get("segments", [])

    # ===== 原来的逻辑（span 内切换）=====
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
                "audio_rel_path": item.get("path"),
                "segment_id": segment.get("segment_id"),
                "left_span": left,
                "right_span": right,
                "switch_time": float(left.get("end", 0.0)),
                "gap_start": float(left.get("end", 0.0)),
                "gap_end": float(right.get("start", left.get("end", 0.0))),
                "switch_index": i,
            }

    # ===== 新增：跨 segment 切换 =====
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]

        spans1 = seg1.get("language_spans", [])
        spans2 = seg2.get("language_spans", [])

        if not spans1 or not spans2:
            continue

        lang1 = spans1[-1].get("language")
        lang2 = spans2[0].get("language")

        if lang1 == lang2:
            continue

        yield {
            "audio_rel_path": item.get("path"),
            "segment_id": f"{seg1.get('segment_id')}_{seg2.get('segment_id')}",
            "left_span": spans1[-1],
            "right_span": spans2[0],
            "switch_time": float(seg1.get("end", 0.0)),
            "gap_start": float(seg1.get("end", 0.0)),
            "gap_end": float(seg2.get("start", seg1.get("end", 0.0))),
            "switch_index": i,
        }

def make_sample_key(source_json: Path, audio_path: Path, sample):
    switch_time = round(float(sample["switch_time"]), 6)
    segment_id = sample.get("segment_id")
    switch_index = sample.get("switch_index")
    return f"{source_json.as_posix()}|{audio_path.as_posix()}|{segment_id}|{switch_index}|{switch_time}"


def build_record(extractor, wav, audio_path: Path, source_json: Path, sample, window_sec=1.0):
    switch_time = sample["switch_time"]
    seg1 = extract_window(wav, extractor.sr, switch_time - window_sec, switch_time)
    seg2 = extract_window(wav, extractor.sr, switch_time, switch_time + window_sec)

    feature_values = extractor.build_raw_features(
        seg1,
        seg2,
        t1_end=sample["gap_start"],
        t2_start=sample["gap_end"],
    )

    root = project_root()
    source_json_str = source_json.relative_to(root).as_posix()
    audio_path_str = audio_path.relative_to(root).as_posix()
    is_switch = "crossfade_insertions" not in source_json_str.lower()

    return {
        "json_path": source_json_str,
        "json_name": source_json.name,
        "audio_path": audio_path_str,
        "audio_name": audio_path.name,
        "feature": feature_values,
        "is_switch": is_switch,
    }


def load_existing_state(output_path: Path, progress_path: Path):
    records = []
    completed_keys = set()

    if progress_path.exists():
        try:
            loaded = json.loads(progress_path.read_text(encoding="utf-8"))
            completed_keys = set(loaded.get("completed_keys", []))
        except Exception:
            completed_keys = set()

    return records, completed_keys


# =========================
# 🔴 修改1：去掉排序 + 只写progress
# =========================
def save_progress(progress_path: Path, completed_keys):
    progress = {"completed_keys": list(completed_keys)}  # 不排序
    progress_path.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")


def process_json_file(extractor, json_path: Path, records, completed_keys,
                      output_path: Path, progress_path: Path,
                      test_mode=False, window_sec=1.0, pbar=None):

    data = json.loads(json_path.read_text(encoding="utf-8"))
    root = project_root()
    new_count = 0

    with open(output_path, "a", encoding="utf-8") as f:  # 打开一次文件

        for item in data:
            audio_rel = item.get("path")
            if not audio_rel:
                continue

            audio_path = resolve_audio_path(root / audio_rel)
            if audio_path is None:
                print(f"missing audio: {audio_rel}")
                continue

            try:
                wav = load_audio(audio_path, sr=extractor.sr)
            except Exception:
                print(f"failed audio load: {audio_rel}")
                continue

            for sample in iter_switch_samples(item):
                sample_key = make_sample_key(json_path, audio_path, sample)
                if sample_key in completed_keys:
                    continue

                record = build_record(extractor, wav, audio_path, json_path, sample, window_sec=window_sec)

                records.append(record)
                completed_keys.add(sample_key)
                new_count += 1

                # JSONL写入（线性）
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # =========================
                # 🔴 修改2：降低progress写入频率
                # =========================
                if new_count % 100 == 0:
                    save_progress(progress_path, completed_keys)

                if pbar is not None:
                    pbar.update(1)

                if test_mode:
                    save_progress(progress_path, completed_keys)
                    return new_count

    # 最后写一次progress
    if new_count > 0:
        save_progress(progress_path, completed_keys)

    return new_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--used-json", default="dl_model/used_json.txt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--window-sec", type=float, default=1.0)
    args = parser.parse_args()

    root = project_root()
    used_json_path = root / args.used_json
    json_files = load_used_json_list(used_json_path)

    if args.output is None:
        output_name = "mlp_feature_cache.jsonl"
        output_path = root / "dl_model" / output_name
    else:
        output_path = root / args.output

    progress_path = output_path.with_name(output_path.name + ".progress.json")

    extractor = SpeakerFeatureExtractor(sr=16000, model_name="base")
    print(f"Embedding model      : {extractor.model_name}")
    all_records, completed_keys = load_existing_state(output_path, progress_path)

    total_remaining_samples = 0
    resolved_jsons_for_processing = []

    for rel_path in json_files:
        json_path = resolve_case_insensitive(root / rel_path)
        if json_path is None:
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for item in data:
            audio_rel = item.get("path")
            if not audio_rel:
                continue
            audio_path = resolve_audio_path(root / audio_rel)
            if audio_path is None:
                continue

            for sample in iter_switch_samples(item):
                sample_key = make_sample_key(json_path, audio_path, sample)
                if sample_key not in completed_keys:
                    total_remaining_samples += 1

        resolved_jsons_for_processing.append(json_path)

    print(f"Total remaining samples: {total_remaining_samples}")

    total_new = 0
    with tqdm(total=total_remaining_samples, desc="Processing") as pbar:
        for json_path in resolved_jsons_for_processing:
            total_new += process_json_file(
                extractor,
                json_path,
                all_records,
                completed_keys,
                output_path,
                progress_path,
                args.test,
                args.window_sec,
                pbar,
            )

    print(f"added {total_new} samples")


if __name__ == "__main__":
    main()
