# SPDX-License-Identifier: MIT
import importlib.util
import pathlib
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "capture_reference", pathlib.Path(__file__).parents[1] / "capture_reference.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class ReferenceCaptureTests(unittest.TestCase):
    def test_only_timings_are_excluded(self):
        document = {"diagnostics": {"timings": [1.1], "state_hash": "abc"},
                    "checkpoints": [{"frame": 1, "position": [1.0, 2.0]}]}
        normalized = reference.deterministic_document(document)
        self.assertEqual(normalized["diagnostics"], {"state_hash": "abc"})
        self.assertIn("timings", document["diagnostics"])
        changed = reference.deterministic_document(document)
        changed["checkpoints"][0]["position"][1] += 1.e-12
        difference = reference.first_difference(normalized, changed)
        self.assertEqual(difference["path"], "$.checkpoints[0].position[1]")

    def test_discrete_values_types_order_and_missing_data_fail(self):
        for expected, actual in [([1, 2], [2, 1]), ([1], []), (1, 1.0),
                                 ({"id": 1}, {}), (-0.0, 0.0)]:
            self.assertIsNotNone(reference.first_difference(expected, actual))
        self.assertIsNone(reference.first_difference({"a": 1}, {"a": 1}))

    def test_reject_nonfinite_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            for text in ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}',
                         '{"id":1,"id":2}']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    reference.read_json(path)


if __name__ == "__main__":
    unittest.main()
