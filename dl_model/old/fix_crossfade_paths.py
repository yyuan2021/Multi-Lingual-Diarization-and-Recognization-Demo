import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_crossfade_path_map(root: Path) -> dict[str, str]:
    crossfade_root = root / "datasets" / "crossfade_insertions"
    mapping = {}
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


def main():
    root = project_root()
    json_path = root / "datasets" / "crossfade_insertions" / "crossfade_insertions_mixed_language_cleaned.json"
    old_path = root / "datasets" / "crossfade_insertions" / "crossfade_insertions_mixed_language_cleaned_old.json"

    crossfade_path_map = build_crossfade_path_map(root)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    fixed = []
    fixed_count = 0
    for item in data:
        path = item.get("path", "").replace("\\", "/")
        canonical_path = crossfade_path_map.get(path.lower(), path)
        if canonical_path != path:
            fixed_count += 1
        item["path"] = canonical_path
        item["audio_name"] = Path(canonical_path).name
        fixed.append(item)

    old_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    json_path.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"fixed: {json_path.as_posix()}")
    print(f"backup_refreshed: {old_path.as_posix()}")
    print(f"paths_fixed: {fixed_count}")
    if fixed:
        print(f"sample_path: {fixed[0]['path']}")
        print(f"sample_audio_name: {fixed[0]['audio_name']}")


if __name__ == "__main__":
    main()
