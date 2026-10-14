#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import json
import pathlib
import sys


EXPECTED = {
    "avbdInputs": 260,
    "avbdOutputs": 36,
    "avbdInputCount": 4,
}


def parameter_size(parameter):
    parameter_type = parameter["type"]
    if parameter_type.get("kind") == "resource":
        parameter_type = parameter_type["resultType"]
    return parameter_type["sizes"][0]["value"]


def check(path_text):
    path = pathlib.Path(path_text)
    reflection = json.loads(path.read_text(encoding="utf-8"))
    actual = {item["name"]: parameter_size(item)
              for item in reflection["parameters"]}
    if actual != EXPECTED:
        raise RuntimeError(f"{path.name}: AVBD ABI {actual}, expected {EXPECTED}")
    entry_points = reflection.get("entryPoints", [])
    if len(entry_points) != 1 or entry_points[0]["name"] != "avbdConformanceMain":
        raise RuntimeError(f"{path.name}: missing AVBD conformance entry point")
    if entry_points[0].get("threadGroupSize") != [64, 1, 1]:
        raise RuntimeError(f"{path.name}: unexpected thread-group size")


DXIL_RESOURCES = {
    "parameters": ("unorderedAccess", 0),
    "previous": ("unorderedAccess", 1),
    "states": ("unorderedAccess", 2),
    "manifoldContacts": ("unorderedAccess", 3),
    "pairs": ("unorderedAccess", 4),
    "pair_count": ("unorderedAccess", 5),
    "joints": ("unorderedAccess", 6),
    "ids": ("unorderedAccess", 7),
    "bodies": ("unorderedAccess", 8),
    "schedule": ("unorderedAccess", 9),
    "manifoldMetadata": ("unorderedAccess", 10),
    "avbdConstants": ("constantBuffer", 0),
}


def check_dxil(path_text, entry_name, group_size):
    path = pathlib.Path(path_text)
    reflection = json.loads(path.read_text(encoding="utf-8"))
    entry_points = reflection.get("entryPoints", [])
    if len(entry_points) != 1 or entry_points[0]["name"] != entry_name:
        raise RuntimeError(f"{path.name}: missing {entry_name}")
    entry = entry_points[0]
    if entry.get("threadGroupSize") != [group_size, 1, 1]:
        raise RuntimeError(f"{path.name}: unexpected thread-group size")
    actual = {
        item["name"]: (item["binding"]["kind"], item["binding"].get("index"))
        for item in entry.get("bindings", [])
        if item["name"] in DXIL_RESOURCES
    }
    if actual != DXIL_RESOURCES:
        raise RuntimeError(
            f"{path.name}: D3D12 binding ABI {actual}, expected {DXIL_RESOURCES}"
        )


if __name__ == "__main__":
    if len(sys.argv) not in (5, 7):
        raise SystemExit(
            "usage: check_slang_reflection.py CUDA_JSON METAL_JSON HLSL_JSON "
            "WGSL_JSON "
            "[PREPARE_DXIL_JSON SOLVE_DXIL_JSON]"
        )
    for argument in sys.argv[1:5]:
        check(argument)
    if len(sys.argv) == 7:
        check_dxil(sys.argv[5], "prepare_avbd_kernel", 128)
        check_dxil(sys.argv[6], "solve_avbd_kernel", 32)
    print("Slang AVBD CUDA/Metal/HLSL/WGSL/DXIL reflection contracts passed")
