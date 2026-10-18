#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check or refresh file hashes pinned by the conformance corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


HASH_SECTIONS = (
    "glb_sha256",
    "lifecycle_profile_sha256",
    "quality_profile_sha256",
    "qualification_profile_sha256",
    "evidence_index_sha256",
)
NORMALIZED_SUFFIXES = {".json", ".xml"}


def contract_sha256(path: Path) -> str:
    """Match the installed C++ runner's cross-platform contract hashing."""
    data = path.read_bytes()
    if path.suffix.lower() in NORMALIZED_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes the repository: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(f"manifest input is missing: {relative}")
    return path


def synchronize_manifest(root: Path, *, write: bool) -> list[tuple[str, str, str, str]]:
    """Return stale pins and optionally replace them with current hashes."""
    root = root.resolve()
    manifest_path = root / "conformance/v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changes: list[tuple[str, str, str, str]] = []
    for section_name in HASH_SECTIONS:
        section = manifest.get(section_name)
        if not isinstance(section, dict) or not section:
            raise ValueError(f"manifest section is missing or empty: {section_name}")
        for relative, expected in section.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError(f"manifest section has an invalid entry: {section_name}")
            actual = contract_sha256(_checked_path(root, relative))
            if actual != expected:
                changes.append((section_name, relative, expected, actual))
                section[relative] = actual
    if write and changes:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return changes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the checkout containing this tool)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale pins without changing the manifest",
    )
    arguments = parser.parse_args(argv)
    changes = synchronize_manifest(arguments.root, write=not arguments.check)
    if not changes:
        print("PASS: conformance manifest hashes are current")
        return 0
    action = "STALE" if arguments.check else "UPDATED"
    for section, relative, _previous, current in changes:
        print(f"{action}: {section} {relative} {current}")
    return 1 if arguments.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
