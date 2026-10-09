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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_slang_reflection.py CUDA_JSON METAL_JSON HLSL_JSON")
    for argument in sys.argv[1:]:
        check(argument)
    print("Slang AVBD CUDA/Metal/HLSL reflection contracts passed")
