#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify physical Gallery CTests pin the configured driver and validation layer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def _property(test: dict[str, Any], name: str) -> Any:
    for item in test.get("properties", []):
        if item.get("name") == name:
            return item.get("value")
    return []


def _normalized(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def audit_catalog(catalog: dict[str, Any], manifest: Path,
                  library: Path) -> list[str]:
    manifest = manifest.resolve()
    library = library.resolve()
    if not manifest.is_file():
        raise AssertionError(f"validation manifest is missing: {manifest}")
    if not library.is_file():
        raise AssertionError(f"validation layer is missing: {library}")
    try:
        layer = json.loads(manifest.read_text(encoding="utf-8"))["layer"]
        manifest_library = Path(layer["library_path"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AssertionError(f"invalid validation manifest {manifest}: {error}") from error
    if layer.get("name") != "PM_LAYER_PARALLEL_MATER_validation":
        raise AssertionError(f"unexpected validation layer name: {layer.get('name')}")
    if _normalized(manifest_library) != _normalized(library):
        raise AssertionError(
            f"validation manifest selects {manifest_library}, expected {library}")

    audited: list[str] = []
    failures: list[str] = []
    for test in catalog.get("tests", []):
        labels = set(_property(test, "LABELS"))
        environment = _property(test, "ENVIRONMENT")
        driver_entries = [entry for entry in environment
                          if entry.startswith("PM_DRIVER_FILES=")]
        if "gallery" not in labels or not driver_entries:
            continue
        name = test.get("name", "<unnamed>")
        audited.append(name)
        layer_files = [entry.split("=", 1)[1] for entry in environment
                       if entry.startswith("PM_LAYER_FILES=")]
        layer_paths = [entry for entry in environment
                       if entry.startswith("PM_LAYER_PATH=")]
        if len(driver_entries) != 1:
            failures.append(f"{name}: expected one PM_DRIVER_FILES entry")
        if len(layer_files) != 1:
            failures.append(f"{name}: expected one PM_LAYER_FILES entry")
        elif _normalized(Path(layer_files[0])) != _normalized(manifest):
            failures.append(
                f"{name}: selects {layer_files[0]}, expected {manifest}")
        if layer_paths:
            failures.append(
                f"{name}: PM_LAYER_PATH is not an exact build-artifact selection")
    if not audited:
        failures.append("no physical Gallery CTests were available to audit")
    if failures:
        raise AssertionError("\n".join(failures))
    return audited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctest", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    arguments = parser.parse_args()
    command = [arguments.ctest, "--test-dir", str(arguments.build),
               "--show-only=json-v1"]
    if arguments.config:
        command.extend(["--build-config", arguments.config])
    completed = subprocess.run(command, check=True, capture_output=True,
                               text=True)
    catalog = json.loads(completed.stdout)
    audited = audit_catalog(catalog, arguments.manifest, arguments.library)
    print(f"PASS: {len(audited)} physical Gallery CTests select "
          f"{arguments.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
