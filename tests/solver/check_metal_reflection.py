#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate single-entry Metal output without requiring Apple hardware."""

import json
from pathlib import Path
import re
import sys


def check(manifest: Path) -> None:
    records = manifest.read_text(encoding="utf-8").splitlines()
    if not records:
        raise ValueError("empty Metal manifest")
    seen = set()
    for record in records:
        module, entry, base = record.split("|")
        if entry in seen:
            raise ValueError(f"duplicate Metal entry: {entry}")
        seen.add(entry)
        source = Path(base + ".metal").read_text(encoding="utf-8")
        reflection = json.loads(Path(base + ".json").read_text(encoding="utf-8"))
        entries = reflection["entryPoints"]
        if len(entries) != 1 or entries[0]["name"] != entry:
            raise ValueError(f"{module}.{entry}: expected isolated entry reflection")
        kernels = re.findall(r"\[\[kernel\]\]\s+void\s+(\w+)", source)
        if kernels != [entry]:
            raise ValueError(f"{module}.{entry}: expected isolated native entry")
        if entry == "solve_avbd_stage_kernel" and re.search(
            r"\b(?:threadgroup_barrier|simdgroup_barrier|pmGridBarrier\w*)\s*\(", source
        ):
            raise ValueError("staged AVBD must synchronize at dispatch boundaries only")
        # Regression for Slang's multi-entry global-input lowering failure:
        # helper reads must have a native input at the kernel boundary.
        for helper, semantic in (
            ("pmThreadIndex", "thread_position_in_grid"),
            ("pmGroupThreadIndex", "thread_position_in_threadgroup"),
            ("pmBlockIndex", "threadgroup_position_in_grid"),
        ):
            if re.search(rf"\b{helper}_\d+\(", source) and f"[[{semantic}]]" not in source:
                raise ValueError(f"{module}.{entry}: missing native {semantic}")
        threads = entries[0]["threadGroupSize"]
        if threads[0] <= 0 or threads[1:] != [1, 1]:
            raise ValueError(f"{module}.{entry}: unsupported workgroup shape {threads}")
        end = 0
        for parameter in entries[0]["parameters"]:
            binding = parameter["binding"]
            if binding["kind"] != "uniform" or binding["offset"] < end:
                raise ValueError(f"{module}.{entry}: overlapping/invalid argument layout")
            end = binding["offset"] + binding["size"]
        if end > 4096:
            raise ValueError(f"{module}.{entry}: arguments exceed Metal setBytes limit")
        for parameter in reflection["parameters"]:
            if parameter["name"] != "pmMetalDispatch":
                continue
            fields = parameter["type"]["elementType"]["fields"]
            offsets = [field["binding"]["offset"] for field in fields]
            if offsets != [0, 4, 8, 12]:
                raise ValueError(f"{module}.{entry}: dispatch constants layout changed")
    print(f"Metal single-entry/reflection contracts: {len(seen)} passed")


if __name__ == "__main__":
    try:
        check(Path(sys.argv[1]))
    except (IndexError, KeyError, OSError, ValueError) as error:
        sys.exit(f"Metal reflection check failed: {error}")
