#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "rope-rigid-contact-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_rope_rigid_contact_quality", ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def rigid_state(position: list[float], velocity: list[float],
                orientation: list[float] | None = None,
                angular: list[float] | None = None) -> dict:
    return {
        "position_m": list(position),
        "orientation_xyzw": list(orientation or [0.0, 0.0, 0.0, 1.0]),
        "linear_velocity_m_s": list(velocity),
        "angular_velocity_rad_s": list(angular or [0.0, 0.0, 0.0]),
    }


def rope_state(index: int, position: list[float], velocity: list[float],
               force: list[float]) -> dict:
    return {
        "index": index,
        "position_m": list(position),
        "velocity_m_s": list(velocity),
        "rigid_contact_force_n": list(force),
    }


def fixture_result(definition: dict) -> dict:
    fixture = copy.deepcopy(definition)
    initial_velocity = definition["prepared_rope_velocity_m_s"]
    initial_rope = [
        rope_state(index, position, initial_velocity, [0.0, 0.0, 0.0])
        for index, position in enumerate(definition["prepared_rope_nodes_m"])
    ]
    fixture["initial"] = {
        "rope": initial_rope,
        "rigid": rigid_state(definition["rigid_initial_position_m"],
                             [0.0, 0.0, 0.0]),
    }

    if definition["expected_contact"]:
        impulse = definition["expected_coherent_reduced_mass_impulse_n_s"]
        rope_mass = definition["rope_total_mass_kg"]
        rigid_mass = definition["rigid_mass_kg"]
        final_rope_velocity = [
            initial_velocity[0] - impulse / rope_mass,
            initial_velocity[1], initial_velocity[2],
        ]
        force = [-0.5 * impulse / definition["timestep_seconds"], 0.0, 0.0]
        final_positions = [
            [-0.12, -0.015, 0.0],
            [-0.12, 0.015, 0.0],
        ]
        fixture["final"] = {
            "rope": [
                rope_state(index, position, final_rope_velocity, force)
                for index, position in enumerate(final_positions)
            ],
            "rigid": rigid_state([0.0, 0.0, 0.0],
                                 [impulse / rigid_mass, 0.0, 0.0]),
        }
    else:
        timestep = definition["timestep_seconds"]
        final_positions = [[
            position[axis] + initial_velocity[axis] * timestep
            for axis in range(3)
        ] for position in definition["prepared_rope_nodes_m"]]
        fixture["final"] = {
            "rope": [
                rope_state(index, position, initial_velocity, [0.0, 0.0, 0.0])
                for index, position in enumerate(final_positions)
            ],
            "rigid": rigid_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        }
    return fixture


def valid_result(profile: dict) -> dict:
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "fixtures": [fixture_result(item) for item in profile["fixtures"]],
    }


class RopeRigidContactQualityTests(unittest.TestCase):
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

    def test_contact_and_separated_control_pass(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 2)
        self.assertEqual(report["sample_count"], 2)
        impact = report["observations"]["rope-segment-centered-cube-impact"]
        self.assertAlmostEqual(
            impact["minimum_capsule_geometry_clearance_m"], 0.02)
        self.assertAlmostEqual(
            impact["expected_normal_impulse_n_s"],
            self.profile["fixtures"][0][
                "expected_coherent_reduced_mass_impulse_n_s"])
        self.assertAlmostEqual(impact["reciprocal_impulse_residual_n_s"], 0.0)
        self.assertGreater(impact["final_frictionless_slip_m_s"], 0.5)
        control = report["observations"][
            "rope-segment-separated-cube-control"]
        self.assertAlmostEqual(control["control_position_error_m"], 0.0)
        self.assertAlmostEqual(control["control_momentum_change_n_s"], 0.0)

    def test_transformed_geometry_overlap_fails(self) -> None:
        result = valid_result(self.profile)
        for node in result["fixtures"][0]["final"]["rope"]:
            node["position_m"][0] = 0.0
        self.assert_rejected(result,
                             "overlapped transformed rigid geometry")

    def test_nonreciprocal_response_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "impulses are not reciprocal")

    def test_wrong_coherent_reduced_mass_impulse_fails(self) -> None:
        result = valid_result(self.profile)
        fixture = result["fixtures"][0]
        for node in fixture["final"]["rope"]:
            node["velocity_m_s"][0] = 0.2
            node["rigid_contact_force_n"][0] = -20.0
        fixture["final"]["rigid"]["linear_velocity_m_s"][0] = 0.004
        self.assert_rejected(result, "coherent reduced-mass oracle")

    def test_public_force_must_integrate_to_rope_momentum(self) -> None:
        result = valid_result(self.profile)
        for node in result["fixtures"][0]["final"]["rope"]:
            node["rigid_contact_force_n"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result,
                             "contact force does not integrate to momentum")

    def test_hidden_attachment_fails(self) -> None:
        result = valid_result(self.profile)
        for node in result["fixtures"][0]["final"]["rope"]:
            node["velocity_m_s"][1] = 0.0
        self.assert_rejected(result, "moved as a hidden attachment")

    def test_centered_impact_rotation_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rigid"][
            "orientation_xyzw"] = [0.0, 0.0, 0.01, 0.99994999875]
        result["fixtures"][0]["final"]["rigid"][
            "angular_velocity_rad_s"] = [0.0, 0.0, 0.01]
        self.assert_rejected(result, "centered impact generated angular speed")

    def test_separated_control_must_remain_free(self) -> None:
        result = valid_result(self.profile)
        control = result["fixtures"][1]
        control["final"]["rope"][0]["position_m"][0] += 0.01
        control["final"]["rope"][0][
            "rigid_contact_force_n"][0] = 1.0
        self.assert_rejected(result, "separated control is not ballistic")
        self.assert_rejected(result, "separated control reported contact force")

    def test_fixture_contract_is_immutable(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["rope_first_attached"] = True
        self.assert_rejected(result, "fixture setting differs from profile")

    def test_rope_speed_cap_is_enforced(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rope"][0][
            "velocity_m_s"] = [11.0, 0.0, 0.0]
        self.assert_rejected(result, "rope speed exceeds authored maximum")


if __name__ == "__main__":
    unittest.main()
