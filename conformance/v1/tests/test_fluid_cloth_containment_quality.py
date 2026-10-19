# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "quality.py"
SPEC = importlib.util.spec_from_file_location("containment_quality", MODULE_PATH)
assert SPEC and SPEC.loader
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)
PROFILE_PATH = ROOT / "quality" / "fluid-cloth-containment-v1.json"


def fluid_state(position: list[float], velocity: list[float],
                acceleration: list[float]) -> dict:
    return {
        "stable_id": 0,
        "position_m": position,
        "velocity_m_s": velocity,
        "acceleration_m_s2": acceleration,
        "foam": 0.15,
    }


def cloth_state(index: int, position: list[float],
                force: list[float]) -> dict:
    return {
        "index": index,
        "position_m": list(position),
        "velocity_m_s": [0.0, 0.0, 0.0],
        "liquid_contact_force_n": force,
    }


def valid_result(profile: dict) -> dict:
    fixture = copy.deepcopy(profile["fixture"])
    vertices = fixture["cloth_vertices_m"]
    fixture["initial_fluid_state"] = [fluid_state(
        [0.05, 0.2, 0.2], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
    fixture["final_fluid_state"] = [fluid_state(
        [0.1, 0.2, 0.2], [0.0, 0.0, 0.0], [15.0, 0.0, 0.0])]
    fixture["initial_cloth_state"] = [
        cloth_state(index, position, [0.0, 0.0, 0.0])
        for index, position in enumerate(vertices)]
    fixture["final_cloth_state"] = [
        cloth_state(index, position,
                    [-5.0, 0.0, 0.0] if index < 3 else [0.0, 0.0, 0.0])
        for index, position in enumerate(vertices)]
    return {
        "schema": quality.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": quality.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "fixture": fixture,
    }


class FluidClothContainmentQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return quality.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_closed_containment_and_reciprocal_reaction_pass(self):
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["observations"]["leaked_particle_count"], 0)
        self.assertEqual(report["observations"][
            "reciprocal_force_residual_n"], 0.0)

    def test_open_or_inconsistently_oriented_boundary_fails(self):
        result = valid_result(self.profile)
        result["fixture"]["cloth_triangle_indices"][1:3] = [1, 2]
        self.assert_rejected(result, "cloth topology differs from profile")
        self.assert_rejected(result,
                             "not closed and consistently oriented")

    def test_leak_and_contact_envelope_penetration_fail(self):
        result = valid_result(self.profile)
        result["fixture"]["final_fluid_state"][0]["position_m"][0] = -0.01
        self.assert_rejected(result, "leaked outside")
        self.assert_rejected(result, "penetrated the containment envelope")

    def test_nonreciprocal_or_vacuous_reaction_fails(self):
        result = valid_result(self.profile)
        result["fixture"]["final_cloth_state"][0][
            "liquid_contact_force_n"][0] += 1.0
        self.assert_rejected(result, "reaction forces are not reciprocal")

        result = valid_result(self.profile)
        result["fixture"]["final_fluid_state"][0][
            "acceleration_m_s2"] = [0.0, 0.0, 0.0]
        for vertex in result["fixture"]["final_cloth_state"]:
            vertex["liquid_contact_force_n"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "reciprocal reaction is too small")

    def test_volume_drift_fails(self):
        result = valid_result(self.profile)
        result["fixture"]["final_cloth_state"][3]["position_m"][2] = 0.5
        self.assert_rejected(result, "volume drift exceeds profile")

        result = valid_result(self.profile)
        result["fixture"]["final_cloth_state"][3]["position_m"][2] = -1.0
        self.assert_rejected(result, "cloth orientation inverted")

    def test_changed_public_input_and_stable_id_fail(self):
        result = valid_result(self.profile)
        result["fixture"]["contact_distance_m"] = 0.11
        self.assert_rejected(result, "fixture setting differs from profile")

        result = valid_result(self.profile)
        result["fixture"]["final_fluid_state"][0]["stable_id"] = 7
        self.assert_rejected(result, "stable ID/order changed")


if __name__ == "__main__":
    unittest.main()
