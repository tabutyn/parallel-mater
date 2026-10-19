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
SPEC = importlib.util.spec_from_file_location("fluid_soft_quality", MODULE_PATH)
assert SPEC and SPEC.loader
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)
PROFILE_PATH = ROOT / "quality" / "fluid-soft-contact-v1.json"


def fluid_state(position: list[float], velocity: list[float],
                acceleration: list[float]) -> dict:
    return {
        "stable_id": 0,
        "position_m": position,
        "velocity_m_s": velocity,
        "acceleration_m_s2": acceleration,
        "foam": 0.1,
    }


def soft_state(index: int, position: list[float], velocity: list[float],
               force: list[float]) -> dict:
    return {
        "index": index,
        "position_m": list(position),
        "velocity_m_s": velocity,
        "liquid_contact_force_n": force,
    }


def valid_result(profile: dict) -> dict:
    fixture = copy.deepcopy(profile["fixture"])
    nodes = fixture["soft_nodes_m"]
    surface = fixture["surface_vertices_m"]
    response_inverse_mass = 1.0 + 0.5 * (0.6**2 + 0.2**2 + 0.2**2)
    denominator = 1.0 + 0.5
    impulse = (1.0 / response_inverse_mass) * (
        1.0 - (1.0 - response_inverse_mass / denominator)**4)
    fixture["initial_fluid_state"] = [fluid_state(
        [-0.1, 0.2, 0.2], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
    fixture["final_fluid_state"] = [fluid_state(
        [-0.1, 0.2, 0.2], [1.0 - impulse, 0.0, 0.0],
        [-100.0 * impulse, 0.0, 0.0])]
    weights = fixture["expected_contact_node_weights"]
    fixture["initial_soft_node_state"] = [
        soft_state(index, position, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for index, position in enumerate(nodes)]
    fixture["final_soft_node_state"] = [
        soft_state(index, position,
                   [0.5 * weight * impulse, 0.0, 0.0],
                   [100.0 * weight * impulse, 0.0, 0.0])
        for index, (position, weight) in enumerate(zip(nodes, weights))]
    fixture["initial_surface_state_m"] = copy.deepcopy(surface)
    fixture["final_surface_state_m"] = copy.deepcopy(surface)
    return {
        "schema": quality.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": quality.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "fixture": fixture,
    }


class FluidSoftContactQualityTests(unittest.TestCase):
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

    def test_contact_clearance_reciprocity_and_unique_mass_pass(self):
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["observations"][
            "minimum_binding_incidence_per_node"], 4)
        self.assertAlmostEqual(report["observations"][
            "expected_unique_bound_node_impulse_n_s"], 0.8186, places=3)
        self.assertAlmostEqual(report["observations"][
            "reciprocal_impulse_residual_n_s"], 0.0)

    def test_surface_overlap_fails(self):
        result = valid_result(self.profile)
        result["fixture"]["final_fluid_state"][0]["position_m"][0] = 0.01
        self.assert_rejected(result, "overlapped the soft-body surface")

    def test_nonreciprocal_impulse_and_force_fail(self):
        result = valid_result(self.profile)
        result["fixture"]["final_soft_node_state"][0]["velocity_m_s"][0] += 0.1
        self.assert_rejected(result, "impulses are not reciprocal")

        result = valid_result(self.profile)
        result["fixture"]["final_soft_node_state"][0][
            "liquid_contact_force_n"][0] += 1.0
        self.assert_rejected(result, "forces are not reciprocal")

    def test_surface_incidence_mass_multiplier_fails(self):
        result = valid_result(self.profile)
        result["fixture"]["final_fluid_state"][0]["velocity_m_s"][0] = 2.0 / 3.0
        result["fixture"]["final_fluid_state"][0][
            "acceleration_m_s2"][0] = -100.0 / 3.0
        weights = result["fixture"]["expected_contact_node_weights"]
        for node, weight in zip(
                result["fixture"]["final_soft_node_state"], weights):
            node["velocity_m_s"][0] = weight / 6.0
            node["liquid_contact_force_n"][0] = weight * 100.0 / 3.0
        self.assert_rejected(
            result, "does not use each uniquely bound node mass once")

    def test_surface_binding_reconstruction_fails(self):
        result = valid_result(self.profile)
        result["fixture"]["final_surface_state_m"][4][0] += 0.01
        self.assert_rejected(
            result, "public soft surface does not match its node bindings")

    def test_changed_public_input_and_stable_id_fail(self):
        result = valid_result(self.profile)
        result["fixture"]["contact_distance_m"] = 0.11
        self.assert_rejected(result, "fixture setting differs from profile")

        result = valid_result(self.profile)
        result["fixture"]["final_fluid_state"][0]["stable_id"] = 7
        self.assert_rejected(result, "stable ID/order changed")


if __name__ == "__main__":
    unittest.main()
