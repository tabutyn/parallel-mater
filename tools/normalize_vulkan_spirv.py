#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Order acyclic SPIR-V pointer types before use for affected Mali compilers.

Slang 2026.18 emits OpTypeForwardPointer even for nonrecursive structures.
On Pixel 7 (Mali-G710), a 1496-byte capture-only shader crashes in the vendor
compiler's apply_decorations; defining its pointer types before use fixes it.
This transport-only pass preserves IDs, decorations and executable instructions.
It hoists only type/constant dependencies and removes redundant declarations.
Recursive or unfamiliar dependencies fail closed rather than changing semantics.

Operand layouts: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
"""
import argparse
from pathlib import Path
import struct


def instructions(words):
    offset = 5
    while offset < len(words):
        count = words[offset] >> 16
        if not count or offset + count > len(words):
            raise ValueError(f"Malformed SPIR-V instruction at word {offset}")
        yield words[offset:offset + count]
        offset += count


def declaration(inst):
    """Return (result ID, dependency IDs) for understood movable declarations."""
    op = inst[0] & 0xffff
    if op in (19, 20, 21, 22, 26, 31, 34, 35, 36, 37, 38):
        return inst[1], ()  # scalar/opaque types: remaining operands are literals
    if op in (23, 24, 25, 27, 29):
        return inst[1], (inst[2],)  # vector/matrix/image/runtime array
    if op == 28:  # array element type and length constant
        return inst[1], inst[2:4]
    if op in (30, 33):  # struct / function types
        return inst[1], inst[2:]
    if op == 32:  # pointer: storage class is a literal
        return inst[1], (inst[3],)
    if op in (41, 42, 43, 45, 46, 48, 49, 50):
        return inst[2], (inst[1],)  # scalar/null constants (including spec constants)
    if op in (44, 51):  # constant composites
        return inst[2], (inst[1], *inst[3:])
    return None


def normalize(data):
    if len(data) < 20 or len(data) % 4:
        raise ValueError("Invalid SPIR-V size")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError("Expected little-endian SPIR-V")
    code = list(instructions(words))
    declarations = {}
    forwards = []
    for index, inst in enumerate(code):
        op = inst[0] & 0xffff
        if op == 54:  # OpFunction: leave all function bodies opaque
            break
        if op == 39:  # OpTypeForwardPointer
            if len(inst) != 3:
                raise ValueError("Malformed OpTypeForwardPointer")
            forwards.append(inst)
        item = declaration(inst)
        if item:
            result, dependencies = item
            if result in declarations:
                raise ValueError(f"Duplicate declaration %{result}")
            declarations[result] = (index, dependencies)
    if not forwards:
        return data
    for inst in forwards:
        target = declarations.get(inst[1])
        if target is None:
            raise ValueError(f"Missing pointer definition %{inst[1]}")
        pointer = code[target[0]]
        if pointer[0] & 0xffff != 32 or pointer[2] != inst[2]:
            raise ValueError(f"Mismatched pointer definition %{inst[1]}")

    output = list(words[:5])
    emitted, visiting = set(), set()

    def emit(result):
        if result not in declarations:
            raise ValueError(f"Cannot hoist unfamiliar dependency %{result}")
        index, dependencies = declarations[result]
        if index in emitted:
            return
        if result in visiting:
            raise ValueError(f"Recursive pointer type %{result}: Mali workaround requires acyclic types")
        visiting.add(result)
        for dependency in dependencies:
            emit(dependency)
        visiting.remove(result)
        output.extend(code[index])
        emitted.add(index)

    for index, inst in enumerate(code):
        if inst[0] & 0xffff == 39:
            emit(inst[1])
        elif index not in emitted:
            output.extend(inst)
            emitted.add(index)
    return struct.pack(f"<{len(output)}I", *output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_bytes(normalize(args.input.read_bytes()))


if __name__ == "__main__":
    main()
