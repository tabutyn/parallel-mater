#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "soft-smoke-contact-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_soft_smoke_quality",
                                              ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def particle(position: list[float], velocity: list[float], age: float) -> dict:
    return {
        "slot": 0,
        "position_m": position,
        "velocity_m_s": velocity,
        "age_seconds": age,
        "number_density": 1.0,
        "pressure_m_s2": 0.0,
        "vorticity_per_s": [0.0, 0.0, 0.0],
    }


def snapshot(definition: dict, position: list[float],
             velocity: list[float], age: float) -> dict:
    return {
        "smoke_particles": [particle(position, velocity, age)],
        "soft_body_surface_vertices_m": copy.deepcopy(definition["soft_body_surface_vertices_m"]),
        "soft_body_surface_velocities_m_s": [
            [0.0, 0.0, 0.0]
            for _ in definition["soft_body_surface_vertices_m"]
        ],
    }


def fixture(definition: dict) -> dict:
    actual = copy.deepcopy(definition)
    initial_position = copy.deepcopy(
        definition["expected_initial_smoke_center_m"])
    initial_velocity = copy.deepcopy(
        definition["expected_initial_smoke_velocity_m_s"])
    timestep = definition["timestep_seconds"]
    control_final = [initial_position[axis] + initial_velocity[axis] * timestep
                     for axis in range(3)]
    coupled_final = copy.deepcopy(control_final)
    coupled_final[0] = -definition["contact_distance_m"]
    redirected = [0.0, initial_velocity[1], initial_velocity[2]]
    actual["runs"] = [
        {
            "id": "control",
            "initial": snapshot(definition, initial_position,
                                initial_velocity, 0.0),
            "final": snapshot(definition, control_final,
                              initial_velocity, timestep),
        },
        {
            "id": "coupled",
            "initial": snapshot(definition, initial_position,
                                initial_velocity, 0.0),
            "final": snapshot(definition, coupled_final,
                              redirected, timestep),
        },
    ]
    return actual


class SoftBodySmokeContactQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.result = {
            "schema": QUALITY.RESULT_SCHEMA,
            "profile_id": self.profile["id"],
            "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "fixtures": [fixture(item) for item in self.profile["fixtures"]],
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return QUALITY.evaluate(self.profile, result or self.result,
                                QUALITY.profile_sha256(PROFILE_PATH))

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"], report)
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_accepts_particle_and_grid_surface_redirection(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["fixture_count"], 2)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(set(report["observations"]), {
            "particle-soft-contact", "grid-soft-contact"})

    def test_rejects_control_without_surface_crossing(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][0]["final"]["smoke_particles"][0][
            "position_m"][0] = -0.004
        self.assert_rejected(
            result, "control smoke did not cross soft body surface")

    def test_rejects_wrong_contact_clearance_or_crossed_side(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][1]["final"]["smoke_particles"][0][
            "position_m"][0] = 0.002
        self.assert_rejected(result,
                             "contact failed to keep smoke on its incoming side")
        self.assert_rejected(result,
                             "contact clearance differs from declared surface")

    def test_rejects_normal_flow_or_missing_tangential_redirection(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [0.1, 0.0, 0.0]
        self.assert_rejected(result, "contact left excessive normal velocity")
        self.assert_rejected(result,
                             "contact did not retain redirected tangential motion")

    def test_rejects_unbounded_smoke_velocity(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [0.0, 3.0, 0.0]
        self.assert_rejected(result, "smoke speed exceeds profile bound")

    def test_rejects_reverse_mechanical_soft_response(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["soft_body_surface_velocities_m_s"][0][
            0] = 0.01
        self.assert_rejected(
            result,
            "soft body received an unadvertised reverse mechanical response")

    def test_rejects_noncanonical_setting_and_missing_representation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["smoke_grid_resolution"] = 8
        self.assert_rejected(result, "fixture setting differs from profile")

        result = copy.deepcopy(self.result)
        result["fixtures"].pop()
        self.assert_rejected(result, "missing required fixture")

    def test_rejects_forged_derived_separation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "signed_separation_m"] = 123.0
        self.assert_rejected(result,
                             "reported separation differs from public state")


if __name__ == "__main__":
    unittest.main()
