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


DXIL_PREPARE_BUFFERS = {
    "parameters", "previous", "states", "manifolds", "pairs",
    "pair_count", "joints", "ids", "bodies", "schedule",
}
DXIL_SOLVE_BUFFERS = DXIL_PREPARE_BUFFERS - {"ids"}


def check_dxil(path_text, entry_name, group_size, expected_buffers):
    path = pathlib.Path(path_text)
    reflection = json.loads(path.read_text(encoding="utf-8"))
    entry_points = reflection.get("entryPoints", [])
    if len(entry_points) != 1 or entry_points[0]["name"] != entry_name:
        raise RuntimeError(f"{path.name}: missing {entry_name}")
    entry = entry_points[0]
    if entry.get("threadGroupSize") != [group_size, 1, 1]:
        raise RuntimeError(f"{path.name}: unexpected thread-group size")
    if reflection.get("bindlessSpaceIndex") != 1:
        raise RuntimeError(f"{path.name}: bindless resource space missing")
    global_scope = reflection.get("globalScope", {})
    if global_scope.get("kind") != "constantBuffer" or \
            global_scope.get("binding", {}).get("index") != 0:
        raise RuntimeError(f"{path.name}: common dispatch constants not at b0")
    scope = entry.get("scope", {})
    if scope.get("kind") != "constantBuffer":
        raise RuntimeError(f"{path.name}: entry constants are not packed")
    parameters = {item["name"]: item
                  for item in scope.get("parameters", [])}
    actual_buffers = {
        name for name, item in parameters.items()
        if item.get("type", {}).get("name") == "PMGraphicsBuffer"
    }
    if actual_buffers != expected_buffers:
        raise RuntimeError(
            f"{path.name}: buffer ABI {actual_buffers}, "
            f"expected {expected_buffers}"
        )
    for name in expected_buffers:
        item = parameters[name]
        buffer_type = item["type"]
        fields = [field["name"] for field in buffer_type.get("fields", [])]
        size = buffer_type.get("sizes", [{}])[0].get("value")
        if item.get("binding", {}).get("kind") != "uniform" or \
                fields != ["descriptor_index", "offset", "available"] or \
                size != 12:
            raise RuntimeError(
                f"{path.name}: {name} is not a 12-byte PMGraphicsBuffer"
            )
    retired = {"manifoldContacts", "manifoldMetadata", "avbdConstants"}
    if retired.intersection(parameters):
        raise RuntimeError(f"{path.name}: legacy D3D12 AVBD resources remain")


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
        check_dxil(sys.argv[5], "prepare_avbd_kernel", 128,
                   DXIL_PREPARE_BUFFERS)
        check_dxil(sys.argv[6], "solve_avbd_kernel", 32,
                   DXIL_SOLVE_BUFFERS)
    print("Slang AVBD CUDA/Metal/HLSL/WGSL/DXIL reflection contracts passed")
