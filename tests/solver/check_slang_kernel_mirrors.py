#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
CUDA_KERNEL = re.compile(r"__global__\s+void\s+([A-Za-z0-9_]+)")
SLANG_ENTRY = re.compile(
    r'\[shader\("compute"\)\]\s*\[numthreads\([^]]+\)\]\s*'
    r'void\s+([A-Za-z0-9_]+)',
    re.MULTILINE,
)

native_sources = [ROOT / "src" / "world.cu"]
native_sources.extend(sorted((ROOT / "src").glob("*.cuh")))
failures: list[str] = []
kernel_count = 0

for native in native_sources:
    kernels = set(CUDA_KERNEL.findall(native.read_text()))
    if not kernels:
        continue
    mirror = ROOT / "src" / "slang" / f"{native.stem}.slang"
    if not mirror.exists():
        failures.append(f"{native.relative_to(ROOT)}: missing {mirror.relative_to(ROOT)}")
        continue
    entries = set(SLANG_ENTRY.findall(mirror.read_text()))
    missing = sorted(kernels - entries)
    if missing:
        failures.append(
            f"{mirror.relative_to(ROOT)}: missing entries {', '.join(missing)}"
        )
    kernel_count += len(kernels)

if failures:
    print("\n".join(failures), file=sys.stderr)
    raise SystemExit(1)

print(f"{kernel_count} CUDA physics kernels have 1:1 Slang mirrors")
