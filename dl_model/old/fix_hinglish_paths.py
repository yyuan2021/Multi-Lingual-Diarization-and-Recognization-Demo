import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_manifest_name_map(root: Path) -> dict[str, str]:
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


def normalize_hinglish_path(path: str, canonical_name_map: dict[str, str]) -> str:
    path = path.replace("\\", "/")
    path_lower = path.lower()

    if "datasets/hinglish/" in path_lower:
        start = path_lower.index("datasets/hinglish/")
        path = path[start:]

    path = path.replace("/data/train/train/", "/data/train/")
    path = path.replace("/data/test/test/", "/data/test/")

    path_obj = Path(path)
    canonical_name = canonical_name_map.get(path_obj.name.lower())
    if canonical_name:
        path = str(path_obj.with_name(canonical_name)).replace("\\", "/")

    return path


def main():
    root = project_root()
    json_path = root / "datasets" / "hinglish" / "hinglish_mixed_language.json"
    old_path = root / "datasets" / "hinglish" / "hinglish_mixed_language_old.json"

    canonical_name_map = load_manifest_name_map(root)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    fixed = []
    for item in data:
        item["path"] = normalize_hinglish_path(item.get("path", ""), canonical_name_map)
        item["audio_name"] = Path(item["path"]).name
        fixed.append(item)

    old_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    json_path.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"fixed: {json_path.as_posix()}")
    print(f"backup_refreshed: {old_path.as_posix()}")
    if fixed:
        print(f"sample_path: {fixed[0]['path']}")
        print(f"sample_audio_name: {fixed[0]['audio_name']}")


if __name__ == "__main__":
    main()
