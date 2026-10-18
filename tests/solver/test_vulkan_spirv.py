#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regression tests for Vulkan SPIR-V transport normalization."""
from collections import Counter
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from normalize_vulkan_spirv import (
    bind_entry_parameters, instructions, needs_parameter_buffer, normalize)


def op(opcode, *operands):
    return ((len(operands) + 1) << 16 | opcode, *operands)


def module(*code):
    words = (0x07230203, 0x00010400, 0, 100, 0, *(w for inst in code for w in inst))
    return struct.pack(f"<{len(words)}I", *words)


def decode(data):
    return list(instructions(struct.unpack(f"<{len(data) // 4}I", data)))


class NormalizeTests(unittest.TestCase):
    def test_nested_mesh_pointers_preserve_ids_layout_and_function(self):
        # Same dependency shape as the crashing TriangleMeshResource capture.
        source = module(op(71, 5, 6, 48), op(71, 6, 6, 12),  # ArrayStride
                        op(22, 1, 32), op(39, 5, 5349), op(39, 6, 5349),
                        op(30, 2, 5, 6, 6), op(30, 3, 1, 1, 1),
                        op(30, 4, 3, 3, 3, 3), op(32, 5, 5349, 4),
                        op(32, 6, 5349, 3), op(54, 9, 10, 0, 11), op(56))
        result = normalize(source)
        code = decode(result)
        self.assertEqual(Counter(code), Counter(i for i in decode(source) if i[0] & 0xffff != 39))
        self.assertEqual(code[:2], decode(source)[:2])
        self.assertLess(code.index(op(30, 3, 1, 1, 1)), code.index(op(32, 6, 5349, 3)))
        self.assertLess(code.index(op(32, 5, 5349, 4)), code.index(op(30, 2, 5, 6, 6)))
        self.assertLess(code.index(op(32, 6, 5349, 3)), code.index(op(30, 2, 5, 6, 6)))
        self.assertEqual(result[:20], source[:20])
        self.assertEqual(normalize(result), result)

    def test_array_length_is_a_dependency_not_a_literal(self):
        source = module(op(39, 5, 5349), op(21, 1, 32, 0), op(43, 1, 2, 10),
                        op(28, 3, 1, 2), op(30, 4, 3), op(32, 5, 5349, 4))
        self.assertEqual(decode(normalize(source)), decode(source)[1:])

    def test_recursive_type_fails_without_output(self):
        with self.assertRaisesRegex(ValueError, "Recursive pointer"):
            normalize(module(op(39, 2, 5349), op(30, 1, 2), op(32, 2, 5349, 1)))

    def test_unfamiliar_dependency_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unfamiliar dependency"):
            normalize(module(op(39, 2, 5349), op(30, 1, 99), op(32, 2, 5349, 1)))

    def test_no_forward_pointers_is_byte_identical(self):
        source = module(op(22, 1, 32), op(30, 2, 1), op(32, 3, 5349, 2))
        self.assertEqual(normalize(source), source)

    def test_rejects_malformed_input(self):
        for source in (b"", b"12345", module(op(22, 1, 32))[:-4],
                       module((0,)), module(op(39, 1, 5349)),
                       module(op(39, 2, 5349), op(30, 1), op(32, 2, 7, 1))):
            with self.subTest(source=source), self.assertRaises(ValueError):
                normalize(source)

    def test_entry_parameters_use_dynamic_uniform_binding(self):
        source = module(
            op(71, 2, 2),  # Block
            op(30, 2, 1),
            op(32, 3, 9, 2),  # pointer PushConstant
            op(59, 3, 4, 9))  # variable PushConstant
        result = bind_entry_parameters(source)
        code = decode(result)
        self.assertIn(op(32, 3, 2, 2), code)
        self.assertIn(op(59, 3, 4, 2), code)
        self.assertIn(op(71, 4, 34, 0), code)
        self.assertIn(op(71, 4, 33, 1), code)
        self.assertEqual(bind_entry_parameters(result), result)

    def test_parameter_binding_rejects_incomplete_module(self):
        with self.assertRaisesRegex(ValueError, "no variable"):
            bind_entry_parameters(module(op(30, 2, 1), op(32, 3, 9, 2)))

    def test_large_reflected_entry_selects_parameter_buffer(self):
        def reflection(size):
            return {"entryPoints": [{"parameters": [{
                "binding": {"kind": "uniform", "offset": 0, "size": size}}]}]}
        self.assertFalse(needs_parameter_buffer(reflection(256)))
        self.assertTrue(needs_parameter_buffer(reflection(260)))


if __name__ == "__main__":
    unittest.main()
