# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "compare.py"
SPEC = importlib.util.spec_from_file_location("conformance_compare", MODULE_PATH)
assert SPEC and SPEC.loader
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def case_document(abs_position: float = 0.002) -> dict:
    return {
        "world": {"chaotic_envelope": False},
        "tolerances": {
            "profile": "constraint_contact",
            "values": {
                "constraint_contact": {
                    "position": {"abs": abs_position, "rel": 0.0},
                    "velocity": {"abs": 0.02, "rel": 0.0},
                },
                "deformable": {
                    "position": {"abs": 0.005, "rel": 0.0},
                    "velocity": {"abs": 0.05, "rel": 0.0},
                },
                "direct": {
                    "position": {"abs": 1.0e-5, "rel": 1.0e-5},
                    "velocity": {"abs": 1.0e-5, "rel": 1.0e-5},
                },
                "quaternion_angular": {"abs": 0.002, "rel": 0.0},
                "chaotic_position": {"abs": 0.1, "rel": 0.05},
                "chaotic_momentum": {"abs": 25.0, "rel": 0.25},
                "chaotic_energy": {"abs": 25.0, "rel": 0.25},
                "chaotic_scalar": {"abs": 0.05, "rel": 0.1},
            },
        }
    }


def result(state: dict) -> dict:
    return {
        "backend": "cuda",
        "case_id": "sample",
        "case_sha256": "fixed",
        "checkpoints": [{"frame": 1, "resources": [state]}],
        "diagnostics": {"timing_ms": 1.0},
        "schema": "parallel-mater-conformance-result/v1",
    }


class ComparatorTests(unittest.TestCase):
    def compare(self, expected: dict, actual: dict,
                case: dict | None = None):
        return compare.compare_documents(case or case_document(),
                                         expected, actual)

    def test_tolerance_passes_and_deliberate_violation_fails(self):
        expected = result({"name": "body", "position": [1.0, 2.0, 3.0],
                           "tolerance_class": "constraint_contact"})
        near = result({"name": "body", "position": [1.001, 2.0, 3.0],
                       "tolerance_class": "constraint_contact"})
        far = result({"name": "body", "position": [1.003, 2.0, 3.0],
                      "tolerance_class": "constraint_contact"})
        self.assertTrue(self.compare(expected, near).passed)
        self.assertFalse(self.compare(expected, far).passed)

    def test_quaternion_signs_are_equivalent(self):
        expected = result({"name": "body", "orientation": [0.1, 0.2, 0.3, 0.9]})
        actual = result({"name": "body", "orientation": [-0.1, -0.2, -0.3, -0.9]})
        self.assertTrue(self.compare(expected, actual).passed)

    def test_missing_data_fails(self):
        expected = result({"name": "body", "position": [0.0, 0.0, 0.0]})
        actual = result({"name": "body"})
        comparison = self.compare(expected, actual)
        self.assertFalse(comparison.passed)
        self.assertTrue(any("missing" in item.message
                            for item in comparison.differences))

    def test_nonfinite_data_fails(self):
        expected = result({"name": "body", "position": [0.0, 0.0, 0.0]})
        actual = result({"name": "body", "position": [math.nan, 0.0, 0.0]})
        self.assertFalse(self.compare(expected, actual).passed)

    def test_contact_order_is_canonicalized(self):
        first = {"body": "a", "collider": "b", "normal_impulse": 1.0}
        second = {"body": "c", "collider": "d", "normal_impulse": 2.0}
        expected = result({"contacts": [first, second], "name": "contacts"})
        actual = result({"contacts": [second, first], "name": "contacts"})
        self.assertTrue(self.compare(expected, actual).passed)

    def test_fluid_contacts_align_by_stable_identity(self):
        expected = result({
            "fluid_contacts": [
                {"stable_particle_id": 1, "rigid_body": "floor",
                 "normal": [1.0, 0.0, 0.0]},
                {"stable_particle_id": 2, "rigid_body": "floor",
                 "normal": [0.0, 1.0, 0.0]},
            ],
            "name": "contacts",
        })
        actual = result({
            "fluid_contacts": [
                {"stable_particle_id": 2, "rigid_body": "floor",
                 "normal": [0.001, 1.0, 0.0]},
                {"stable_particle_id": 1, "rigid_body": "floor",
                 "normal": [0.999, 0.0, 0.0]},
            ],
            "name": "contacts",
        })
        self.assertTrue(self.compare(expected, actual).passed)

    def test_rigid_contacts_align_by_identity_and_nearest_geometry(self):
        expected = result({
            "rigid_contacts": [
                {"body": "a", "collider": "b",
                 "normal": [-0.98, 0.198, 1.0e-9],
                 "position": [0.0, 0.0, 0.0]},
                {"body": "a", "collider": "b",
                 "normal": [-0.98, 0.198, 0.0],
                 "position": [0.1, 0.0, 0.0]},
            ],
            "name": "contacts",
        })
        actual = result({
            "rigid_contacts": [
                {"body": "a", "collider": "b",
                 "normal": [-0.980001, 0.198001, 0.0],
                 "position": [0.100001, 0.0, 0.0]},
                {"body": "a", "collider": "b",
                 "normal": [-0.980001, 0.198001, 0.0],
                 "position": [0.000001, 0.0, 0.0]},
            ],
            "name": "contacts",
        })
        self.assertTrue(self.compare(expected, actual).passed)

    def test_topology_and_tolerance_provenance_are_exact(self):
        expected = result({"name": "cloth", "topology": {"indices": [0, 1, 2]}})
        topology = result({"name": "cloth", "topology": {"indices": [0, 2, 1]}})
        tolerance = result({"name": "cloth", "topology": {"indices": [0, 1, 2]}})
        tolerance["case_sha256"] = "changed-tolerance"
        self.assertFalse(self.compare(expected, topology).passed)
        self.assertFalse(self.compare(expected, tolerance).passed)

    def test_contact_rounding_cannot_change_correspondence(self):
        first = {"body": "a", "collider": "b", "position": [0.0, 0.0, 0.0],
                 "friction_impulse": [0.0001, 0.0, 0.0]}
        second = {"body": "a", "collider": "b", "position": [1.0, 0.0, 0.0],
                  "friction_impulse": [0.0002, 0.0, 0.0]}
        changed = dict(first, friction_impulse=[0.0003, 0.0, 0.0])
        expected = result({"contacts": [first, second]})
        self.assertTrue(self.compare(expected, result({"contacts": [second, changed]})).passed)
        self.assertFalse(self.compare(expected, result({"contacts": [changed, changed]})).passed)
        self.assertFalse(self.compare(expected, result({"contacts": [second,
            dict(changed, friction_impulse=[0.01, 0.0, 0.0])]})).passed)
        self.assertFalse(self.compare(expected, result({"contacts": [second,
            dict(changed, collider="other")]})).passed)

    def test_contact_matching_is_one_to_one_not_greedy(self):
        def contact(x):
            return {"body": "a", "collider": "b", "position": [x, 0.0, 0.0]}
        expected = result({"contacts": [contact(0.0), contact(0.002)]})
        actual = result({"contacts": [contact(0.001), contact(-0.002)]})
        self.assertTrue(self.compare(expected, actual).passed)
        self.assertFalse(self.compare(expected,
            result({"contacts": [contact(-0.002), contact(-0.002)]})).passed)

    def test_round_trip_preserves_binary64_text(self):
        value = 1.2345678901234567
        document = result({"name": "body", "position": [value, 0.0, 0.0]})
        restored = json.loads(json.dumps(document, allow_nan=False))
        self.assertEqual(restored["checkpoints"][0]["resources"][0]
                         ["position"][0], value)
        self.assertTrue(self.compare(document, restored).passed)

    def test_backend_and_diagnostics_never_gate(self):
        expected = result({"name": "body", "position": [0.0, 0.0, 0.0]})
        actual = result({"name": "body", "position": [0.0, 0.0, 0.0]})
        actual["backend"] = "metal"
        actual["diagnostics"] = {"timing_ms": 900.0, "state_hash": "different"}
        self.assertTrue(self.compare(expected, actual).passed)

    def test_chaotic_cases_gate_envelopes_not_samples(self):
        case = case_document()
        case["world"]["chaotic_envelope"] = True
        expected = result({
            "aggregate": {"bounds": {"maximum": [1.0, 1.0, 1.0],
                                      "minimum": [0.0, 0.0, 0.0]},
                          "kinetic_energy": 100.0,
                          "momentum": [10.0, 0.0, 0.0]},
            "name": "fluid", "samples": [{"id": 1, "position": [0.0, 0.0, 0.0]}]})
        actual = result({
            "aggregate": {"bounds": {"maximum": [1.1, 1.0, 1.0],
                                      "minimum": [0.0, 0.0, 0.0]},
                          "kinetic_energy": 120.0,
                          "momentum": [18.0, 0.0, 0.0]},
            "name": "fluid", "samples": [{"id": 1, "position": [50.0, 0.0, 0.0]}]})
        self.assertTrue(self.compare(expected, actual, case).passed)

    def test_chaotic_scalar_envelopes_use_scalar_tolerance(self):
        case = case_document()
        case["world"]["chaotic_envelope"] = True
        expected = result({
            "maximum_divergence": 0.5,
            "maximum_strain": 0.65,
            "minimum_clearance": 0.4,
            "pressure_relative_residual": 0.01,
            "rms_divergence": 0.2,
        })
        actual = result({
            "maximum_divergence": 0.58,
            "maximum_strain": 0.64,
            "minimum_clearance": 0.35,
            "pressure_relative_residual": 0.055,
            "rms_divergence": 0.24,
        })
        self.assertTrue(self.compare(expected, actual, case).passed)

        actual["checkpoints"][0]["resources"][0]["maximum_strain"] = 0.9
        comparison = self.compare(expected, actual, case)
        self.assertFalse(comparison.passed)
        self.assertTrue(any(item.path.endswith("maximum_strain")
                            for item in comparison.differences))

    def test_chaotic_quality_envelopes_are_directional(self):
        case = case_document()
        case["world"]["chaotic_envelope"] = True
        expected = result({
            "maximum_divergence": 0.5,
            "maximum_strain": 0.65,
            "minimum_clearance": 0.4,
            "pressure_relative_residual": 0.1,
            "rms_divergence": 0.2,
        })
        better = result({
            "maximum_divergence": 0.0,
            "maximum_strain": 0.0,
            "minimum_clearance": 1.0,
            "pressure_relative_residual": 0.0,
            "rms_divergence": 0.0,
        })
        self.assertTrue(self.compare(expected, better, case).passed)

        worse = result({
            "maximum_divergence": 0.7,
            "maximum_strain": 0.9,
            "minimum_clearance": 0.2,
            "pressure_relative_residual": 0.3,
            "rms_divergence": 0.4,
        })
        comparison = self.compare(expected, worse, case)
        self.assertFalse(comparison.passed)
        failed = {item.path.rsplit(".", 1)[-1]
                  for item in comparison.differences}
        self.assertEqual(failed, {
            "maximum_divergence", "maximum_strain", "minimum_clearance",
            "pressure_relative_residual", "rms_divergence",
        })

    def test_chaotic_rigid_trajectories_and_contact_events_do_not_gate(self):
        case = case_document()
        case["world"]["chaotic_envelope"] = True
        expected = result({
            "id": "rigid/body/1",
            "type": "rigid_body",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        })
        expected["checkpoints"][0]["contacts"] = {
            "fluid_contact_count": 61,
            "rigid_contact_count": 61,
            "fluid_contacts": [{"stable_particle_id": 7,
                                  "normal_impulse": 1.0}],
            "rigid_contacts": [{"body": "a", "collider": "b"}],
        }
        actual = json.loads(json.dumps(expected))
        body = actual["checkpoints"][0]["resources"][0]
        body["position"] = [50.0, -20.0, 3.0]
        body["orientation"] = [0.5, 0.5, 0.5, 0.5]
        body["linear_velocity"] = [100.0, 0.0, 0.0]
        body["angular_velocity"] = [0.0, 100.0, 0.0]
        actual["checkpoints"][0]["contacts"] = {
            "fluid_contact_count": 66,
            "rigid_contact_count": 66,
            "fluid_contacts": [],
            "rigid_contacts": [{"body": "different", "collider": "pair"},
                               {"body": "another", "collider": "pair"}],
        }
        self.assertTrue(self.compare(expected, actual, case).passed)

        actual["checkpoints"][0]["contacts"]["rigid_contact_count"] = 80
        comparison = self.compare(expected, actual, case)
        self.assertFalse(comparison.passed)
        self.assertTrue(any(item.path.endswith("rigid_contact_count")
                            for item in comparison.differences))

        case["world"]["chaotic_envelope"] = False
        self.assertFalse(self.compare(expected, actual, case).passed)


if __name__ == "__main__":
    unittest.main()
