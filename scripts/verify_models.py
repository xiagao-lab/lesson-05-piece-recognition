from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for asset in manifest["assets"]:
        if not str(asset["local_path"]).endswith(".onnx"):
            continue
        path = ROOT / asset["local_path"]
        if not path.is_file():
            failures.append(f"缺少 {path.relative_to(ROOT)}，原文件名应为 {asset['source_filename']}")
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        ok = actual_size == asset["bytes"] and actual_sha == asset["sha256"]
        print(f"{'OK' if ok else 'FAIL'} {path.relative_to(ROOT)} {actual_size} bytes {actual_sha}")
        if not ok:
            failures.append(f"{path.relative_to(ROOT)} 的大小或 SHA256 不匹配")
    if failures:
        for failure in failures:
            print(f"- {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
