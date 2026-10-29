#!/usr/bin/env python3
"""Print the deterministic identity of an extracted CI artifact tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def tree_identity(root: Path) -> tuple[int, str]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="+")
    arguments = parser.parse_args()
    results = []
    for raw in arguments.directory:
        root = raw.resolve()
        if not root.is_dir():
            parser.error(f"not a directory: {raw}")
        count, digest = tree_identity(root)
        results.append({
            "name": raw.name,
            "file_count": count,
            "extracted_tree_sha256": digest,
        })
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
