#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "rope-smoke-contact-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_rope_smoke_contact_quality", ROOT / "quality.py")
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
        "rope_nodes_m": copy.deepcopy(definition["rope_centerline_m"]),
        "rope_velocities_m_s": [
            [0.0, 0.0, 0.0] for _ in definition["rope_centerline_m"]
        ],
    }


def fixture(definition: dict) -> dict:
    actual = copy.deepcopy(definition)
    initial = [-0.020, 0.015, 0.0]
    initial_velocity = [1.0, 0.25, 0.0]
    actual["runs"] = [
        {
            "id": "control",
            "initial": snapshot(definition, initial, initial_velocity, 0.0),
            "final": snapshot(definition, [-0.010, 0.0175, 0.0],
                              initial_velocity, 0.01),
        },
        {
            "id": "coupled",
            "initial": snapshot(definition, initial, initial_velocity, 0.0),
            "final": snapshot(definition, [-0.015, 0.0175, 0.0],
                              [0.0, 0.25, 0.0], 0.01),
        },
    ]
    return actual


class RopeSmokeContactQualityTests(unittest.TestCase):
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
        self.assertTrue(any(message in item["message"]
                            for item in report["failures"]), report)

    def test_accepts_particle_and_grid_capsule_redirection(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["fixture_count"], 2)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(set(report["observations"]), {
            "particle-rope-contact", "grid-rope-contact"})

    def test_rejects_control_that_does_not_enter_radius(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][0]["final"]["smoke_particles"][0][
            "position_m"][0] = -0.018
        self.assert_rejected(result,
                             "control smoke did not enter rope contact radius")

    def test_rejects_wrong_clearance_or_side(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][1]["final"]["smoke_particles"][0][
            "position_m"][0] = 0.002
        self.assert_rejected(result,
                             "contact failed to keep smoke on its incoming side")
        self.assert_rejected(result,
                             "contact clearance differs from declared rope radius")

    def test_rejects_radial_flow_or_missing_tangent(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [0.1, 0.0, 0.0]
        self.assert_rejected(result, "contact left excessive normal velocity")
        self.assert_rejected(
            result, "contact did not retain redirected tangential motion")

    def test_rejects_reverse_rope_response(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"][
            "rope_velocities_m_s"][0][0] = 0.01
        self.assert_rejected(
            result, "unadvertised reverse mechanical response")

    def test_rejects_end_cap_and_speed_violation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "position_m"] = [-0.015, 0.001, 0.0]
        self.assert_rejected(result, "contact hit a rope end cap")
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [0.0, 3.0, 0.0]
        self.assert_rejected(result, "smoke speed exceeds profile bound")

    def test_rejects_noncanonical_setting_or_missing_representation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["rope_radius_m"] = 0.02
        self.assert_rejected(result, "fixture setting differs from profile")
        result = copy.deepcopy(self.result)
        result["fixtures"].pop()
        self.assert_rejected(result, "missing required fixture")

    def test_rejects_forged_public_geometry_derivatives(self) -> None:
        result = copy.deepcopy(self.result)
        smoke = result["fixtures"][0]["runs"][1]["final"][
            "smoke_particles"][0]
        smoke["closest_rope_distance_m"] = 123.0
        smoke["rope_contact_normal"] = [1.0, 0.0, 0.0]
        self.assert_rejected(
            result, "reported rope distance differs from public state")
        self.assert_rejected(
            result, "reported rope normal differs from public state")


if __name__ == "__main__":
    unittest.main()
