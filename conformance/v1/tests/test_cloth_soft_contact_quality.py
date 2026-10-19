#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "quality.py"
PROFILE_PATH = ROOT / "quality" / "cloth-soft-contact-v1.json"
SPEC = importlib.util.spec_from_file_location("cloth_soft_quality", MODULE_PATH)
assert SPEC and SPEC.loader
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def endpoint_state(index: int, position: list[float], velocity: list[float],
                   force: list[float], force_field: str) -> dict:
    return {
        "index": index,
        "position_m": list(position),
        "velocity_m_s": list(velocity),
        force_field: list(force),
    }


def reconstruct_surface(nodes: list[list[float]], bindings: list[dict]
                        ) -> list[list[float]]:
    return [[sum(nodes[node][axis] * weight
                 for node, weight in zip(binding["nodes"],
                                         binding["weights"]))
             for axis in range(3)]
            for binding in bindings]


def valid_result(profile: dict, impulse: float | None = None) -> dict:
    fixture = copy.deepcopy(profile["fixture"])
    weights = fixture["expected_cloth_contact_weights"]
    impulse = impulse if impulse is not None else 1.0 / (
        0.5 + sum(weight * weight for weight in weights))
    initial_nodes = copy.deepcopy(fixture["prepared_soft_nodes_m"])
    final_nodes = copy.deepcopy(initial_nodes)
    final_nodes[0][1] = 0.1
    initial_velocity = fixture["prepared_soft_velocity_m_s"]
    fixture["initial_cloth_state"] = [
        endpoint_state(index, position, [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0], "soft_body_contact_force_n")
        for index, position in enumerate(fixture["cloth_vertices_m"])]
    fixture["final_cloth_state"] = [
        endpoint_state(index, position, [0.0, -weight * impulse, 0.0],
                       [0.0, -100.0 * weight * impulse, 0.0],
                       "soft_body_contact_force_n")
        for index, (position, weight) in enumerate(zip(
            fixture["cloth_vertices_m"], weights))]
    fixture["initial_soft_node_state"] = [
        endpoint_state(index, position, initial_velocity,
                       [0.0, 0.0, 0.0], "cloth_contact_force_n")
        for index, position in enumerate(initial_nodes)]
    fixture["final_soft_node_state"] = [
        endpoint_state(index, position,
                       [1.0, -1.0 + (0.5 * impulse if index == 0 else 0.0),
                        0.0],
                       [0.0, 100.0 * impulse if index == 0 else 0.0, 0.0],
                       "cloth_contact_force_n")
        for index, position in enumerate(final_nodes)]
    fixture["initial_surface_state_m"] = reconstruct_surface(
        initial_nodes, fixture["surface_bindings"])
    fixture["final_surface_state_m"] = reconstruct_surface(
        final_nodes, fixture["surface_bindings"])
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "fixture": fixture,
    }


class ClothSoftContactQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = QUALITY.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return QUALITY.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_clearance_reciprocity_slip_and_unique_mass_pass(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["observations"][
            "minimum_binding_incidence_per_node"], 4)
        self.assertAlmostEqual(report["observations"][
            "expected_unique_bound_node_impulse_n_s"], 1.142857, places=5)
        self.assertAlmostEqual(report["observations"][
            "reciprocal_impulse_residual_n_s"], 0.0)
        self.assertGreater(report["observations"][
            "final_frictionless_slip_m_s"], 0.5)

    def test_contact_overlap_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final_soft_node_state"][0]["position_m"][1] = 0.05
        nodes = [state["position_m"] for state in
                 result["fixture"]["final_soft_node_state"]]
        result["fixture"]["final_surface_state_m"] = reconstruct_surface(
            nodes, result["fixture"]["surface_bindings"])
        self.assert_rejected(result, "overlapped the cloth contact envelope")

    def test_nonreciprocal_impulse_and_force_fail(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final_soft_node_state"][0]["velocity_m_s"][1] += 0.1
        self.assert_rejected(result, "impulses are not reciprocal")

        result = valid_result(self.profile)
        result["fixture"]["final_soft_node_state"][0][
            "cloth_contact_force_n"][1] += 1.0
        self.assert_rejected(result, "forces are not reciprocal")

    def test_surface_incidence_mass_multiplier_fails(self) -> None:
        wrong_impulse = 1.0 / (2.0 + 0.25**2 + 0.5**2 + 0.25**2)
        self.assert_rejected(
            valid_result(self.profile, wrong_impulse),
            "does not use the uniquely bound node mass once")

    def test_hidden_attachment_fails(self) -> None:
        result = valid_result(self.profile)
        soft = result["fixture"]["final_soft_node_state"][0]
        soft["velocity_m_s"][0] = 0.4
        soft["cloth_contact_force_n"][0] = -120.0
        for cloth in result["fixture"]["final_cloth_state"]:
            cloth["velocity_m_s"][0] = 0.4
            cloth["soft_body_contact_force_n"][0] = 40.0
        self.assert_rejected(result, "moved as a hidden attachment")

    def test_surface_binding_reconstruction_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final_surface_state_m"][4][0] += 0.01
        self.assert_rejected(
            result, "public soft surface does not match its node bindings")

    def test_changed_public_input_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["contact_distance_m"] = 0.11
        self.assert_rejected(result, "fixture setting differs from profile")


if __name__ == "__main__":
    unittest.main()
