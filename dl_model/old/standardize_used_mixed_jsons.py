import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_used_json_list(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def load_hinglish_name_map(root: Path) -> dict[str, str]:
    mapping = {}
    for manifest_name in ["manifest_train.jsonl", "manifest_test.jsonl"]:
        manifest_path = root / "datasets" / "hinglish" / "data" / manifest_name
        if not manifest_path.exists():
            continue
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = item.get("audio_filepath", "").replace("\\", "/")
            audio_name = Path(audio_path).name
            mapping[audio_name.lower()] = audio_name
    return mapping


def load_corpus_path_map(root: Path) -> dict[str, str]:
    mapping = {}
    corpus_root = root / "datasets" / "Corpus"
    for wav_path in corpus_root.rglob("*.wav"):
        rel_path = wav_path.relative_to(root).as_posix()
        mapping[rel_path.lower()] = rel_path
    return mapping


def load_crossfade_path_map(root: Path) -> dict[str, str]:
    mapping = {}
    crossfade_root = root / "datasets" / "crossfade_insertions"
    for manifest_path in crossfade_root.rglob("mixed_manifest.json"):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in data:
            rel_path = item.get("path", "").replace("\\", "/")
            if rel_path:
                mapping[rel_path.lower()] = rel_path
    for wav_path in crossfade_root.rglob("*.wav"):
        rel_path = wav_path.relative_to(root).as_posix()
        mapping.setdefault(rel_path.lower(), rel_path)
    return mapping


def normalize_dataset_path(raw_path: str, dataset_prefix: str) -> str:
    path = raw_path.replace("\\", "/")
    path_lower = path.lower()
    datasets_prefix = "datasets/"

    if datasets_prefix in path_lower:
        start = path_lower.index(datasets_prefix)
        rel_path = path[start:]
        parts = rel_path.split("/")
        if len(parts) >= 2:
            suffix = "/".join(parts[2:])
            return f"{dataset_prefix}/{suffix}" if suffix else dataset_prefix

    return f"{dataset_prefix}/{Path(path).name}"


def normalize_hinglish_path(raw_path: str, dataset_prefix: str, canonical_name_map: dict[str, str]) -> str:
    path = normalize_dataset_path(raw_path, dataset_prefix)
    path = path.replace("/data/train/train/", "/data/train/")
    path = path.replace("/data/test/test/", "/data/test/")

    path_obj = Path(path)
    canonical_name = canonical_name_map.get(path_obj.name.lower())
    if canonical_name:
        path = str(path_obj.with_name(canonical_name)).replace("\\", "/")
    return path


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def standardize_span(span: dict) -> dict:
    return {
        "language": span.get("language"),
        "start": to_float(span.get("start", 0.0)),
        "end": to_float(span.get("end", 0.0)),
        "text": span.get("text", ""),
        "score": to_float(span.get("score")),
    }


def standardize_segment(segment: dict) -> dict:
    spans = [standardize_span(span) for span in segment.get("language_spans", [])]
    return {
        "segment_id": segment.get("segment_id"),
        "start": to_float(segment.get("start", 0.0)),
        "end": to_float(segment.get("end", 0.0)),
        "text": segment.get("text", ""),
        "scores": segment.get("scores", {}),
        "language_spans": spans,
    }


def standardize_item(
    item: dict,
    dataset_prefix: str,
    hinglish_name_map: dict[str, str],
    corpus_path_map: dict[str, str],
    crossfade_path_map: dict[str, str],
) -> dict:
    if dataset_prefix == "datasets/hinglish":
        path = normalize_hinglish_path(item.get("path", ""), dataset_prefix, hinglish_name_map)
    elif dataset_prefix == "datasets/Corpus":
        path = normalize_dataset_path(item.get("path", ""), dataset_prefix)
        path = corpus_path_map.get(path.lower(), path)
    elif dataset_prefix == "datasets/crossfade_insertions":
        path = normalize_dataset_path(item.get("path", ""), dataset_prefix)
        path = crossfade_path_map.get(path.lower(), path)
    else:
        path = normalize_dataset_path(item.get("path", ""), dataset_prefix)
    return {
        "path": path,
        "audio_name": Path(path).name,
        "whisper_language": item.get("whisper_language", "mixed"),
        "segments": [standardize_segment(segment) for segment in item.get("segments", [])],
    }


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_old{path.suffix}")


def save_backup(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def main():
    root = project_root()
    used_json_path = root / "dl_model" / "used_json.txt"
    json_paths = load_used_json_list(used_json_path)
    hinglish_name_map = load_hinglish_name_map(root)
    corpus_path_map = load_corpus_path_map(root)
    crossfade_path_map = load_crossfade_path_map(root)

    for rel_path in json_paths:
        json_path = root / rel_path
        dataset_prefix = str(rel_path.parent).replace("\\", "/")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        standardized = [
            standardize_item(
                item,
                dataset_prefix,
                hinglish_name_map,
                corpus_path_map,
                crossfade_path_map,
            )
            for item in data
        ]

        old_path = backup_path(json_path)
        save_backup(json_path, old_path)
        json_path.write_text(
            json.dumps(standardized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"standardized: {json_path.as_posix()}")
        print(f"backup_saved: {old_path.as_posix()}")
        print(f"records: {len(standardized)}")
        if standardized:
            print(f"sample_path: {standardized[0]['path']}")
            print(f"sample_audio_name: {standardized[0]['audio_name']}")


if __name__ == "__main__":
    main()
