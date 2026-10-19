#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "soft-rigid-contact-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_soft_rigid_contact_quality", ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def reconstruct(nodes: list[dict], bindings: list[dict]) -> list[list[float]]:
    surface = []
    for binding in bindings:
        value = [0.0, 0.0, 0.0]
        for node, weight in zip(binding["nodes"], binding["weights"]):
            for axis in range(3):
                value[axis] += weight * nodes[node]["position_m"][axis]
        surface.append(value)
    return surface


def snapshot(positions: list[list[float]], velocities: list[list[float]],
             bindings: list[dict], rigid_position: list[float],
             rigid_velocity: list[float]) -> dict:
    nodes = [{
        "id": index,
        "position_m": position,
        "velocity_m_s": velocities[index],
        "rigid_contact_force_n": [0.0, 0.0, 0.0],
    } for index, position in enumerate(positions)]
    return {
        "soft_nodes": nodes,
        "surface_vertices_m": reconstruct(nodes, bindings),
        "rigid": {
            "position_m": rigid_position,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity_m_s": rigid_velocity,
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        },
    }


def valid_result(profile: dict) -> dict:
    fixtures = []
    for definition in profile["fixtures"]:
        fixture = copy.deepcopy(definition)
        bindings = definition["surface_bindings"]
        initial_positions = copy.deepcopy(definition["prepared_soft_nodes_m"])
        initial_velocities = [[1.0, 0.0, 0.0] for _ in range(4)]
        if definition["id"] == "centered-bound-surface-impact":
            final_positions = copy.deepcopy(initial_positions)
            for index in range(1, 4):
                final_positions[index][0] += 0.01
            final_velocities = copy.deepcopy(initial_velocities)
            final_velocities[0][0] = 0.5
            rigid_position = [0.005, 0.0, 0.0]
            rigid_velocity = [0.5, 0.0, 0.0]
        else:
            final_positions = copy.deepcopy(initial_positions)
            for position in final_positions:
                position[0] += 0.01
            final_velocities = copy.deepcopy(initial_velocities)
            rigid_position = [0.0, 0.0, 0.0]
            rigid_velocity = [0.0, 0.0, 0.0]
        fixture["initial"] = snapshot(
            initial_positions, initial_velocities, bindings,
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        fixture["final"] = snapshot(
            final_positions, final_velocities, bindings,
            rigid_position, rigid_velocity)
        fixtures.append(fixture)
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "fixtures": fixtures,
    }


class SoftRigidContactQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = QUALITY.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return QUALITY.evaluate(self.profile, result, self.profile_hash)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in item["message"]
                            for item in report["failures"]), report)

    def test_centered_contact_and_separated_control_pass(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 2)
        impact = report["observations"]["centered-bound-surface-impact"]
        self.assertEqual(impact["unique_node_count"], 4)
        self.assertEqual(impact["surface_vertex_count"], 10)
        self.assertAlmostEqual(impact["expected_normal_impulse_n_s"], 1.0)
        self.assertAlmostEqual(impact["reciprocal_impulse_residual_n_s"], 0.0)

    def test_bound_surface_overlap_fails(self) -> None:
        result = valid_result(self.profile)
        fixture = result["fixtures"][0]
        fixture["final"]["soft_nodes"][0]["position_m"][0] = -0.09
        fixture["final"]["surface_vertices_m"] = reconstruct(
            fixture["final"]["soft_nodes"], fixture["surface_bindings"])
        self.assert_rejected(result, "overlapped the rigid surface")

    def test_one_way_response_fails_reciprocity(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "impulses are not reciprocal")

    def test_surface_incidence_cannot_duplicate_node_mass(self) -> None:
        result = valid_result(self.profile)
        fixture = result["fixtures"][0]
        fixture["final"]["soft_nodes"][0]["velocity_m_s"][0] = 0.75
        fixture["final"]["rigid"]["linear_velocity_m_s"][0] = 0.25
        self.assert_rejected(result, "unique-node reduced-mass oracle")

    def test_surface_must_reconstruct_from_bindings(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["surface_vertices_m"][4][1] += 0.01
        self.assert_rejected(result, "does not reconstruct from unique nodes")

    def test_separated_control_cannot_exchange_momentum(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][1]["final"]["rigid"][
            "linear_velocity_m_s"][0] = 0.1
        self.assert_rejected(result, "separated control exchanged momentum")

    def test_prepared_state_is_immutable(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["initial"]["soft_nodes"][0][
            "velocity_m_s"][0] = 0.9
        self.assert_rejected(result, "prepared soft velocity differs")


if __name__ == "__main__":
    unittest.main()
