#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "smoke-rigid-contact-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_smoke_rigid_contact_quality", ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def rigid(linear: list[float] | None = None,
          angular: list[float] | None = None) -> dict:
    return {
        "position_m": [0.0, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity_m_s": linear or [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": angular or [0.0, 0.0, 0.0],
    }


def snapshot(position: list[float], velocity: list[float],
             rigid_state: dict) -> dict:
    return {
        "smoke_particles": [{
            "slot": 0,
            "position_m": position,
            "velocity_m_s": velocity,
            "age_seconds": 0.0,
            "number_density": 1.0,
            "pressure_m_s2": 0.0,
            "vorticity_per_s": [0.0, 0.0, 0.0],
        }],
        "rigid": rigid_state,
    }


def fixture_result(definition: dict) -> dict:
    result = copy.deepcopy(definition)
    authored_velocity = definition["smoke_initial_velocity_m_s"]
    authored_speed = math.sqrt(sum(value * value
                                   for value in authored_velocity))
    scale = min(1.0, definition["smoke_maximum_speed_m_s"] / authored_speed)
    velocity = [value * scale for value in authored_velocity]
    timestep = definition["timestep_seconds"]
    initial_position = copy.deepcopy(definition["smoke_emitter_center_m"])
    if definition["representation"] == "SMOKE_PARTICLE":
        initial_position = [initial_position[axis] + velocity[axis] * timestep
                            for axis in range(3)]
    control_position = [initial_position[axis] + velocity[axis] * timestep
                        for axis in range(3)]
    coupled_position = copy.deepcopy(control_position)
    coupled_position[0] = -0.035
    initial = snapshot(initial_position, velocity, rigid())
    result["runs"] = [
        {
            "id": "control",
            "initial": copy.deepcopy(initial),
            "final": snapshot(control_position, velocity, rigid()),
        },
        {
            "id": "coupled",
            "initial": copy.deepcopy(initial),
            "final": snapshot(
                coupled_position, [0.0, velocity[1], velocity[2]],
                rigid([0.1, 0.0, 0.0])),
        },
    ]
    return result


class SmokeRigidContactQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = QUALITY.profile_sha256(PROFILE_PATH)

    def valid_result(self) -> dict:
        return {
            "schema": QUALITY.RESULT_SCHEMA,
            "profile_id": self.profile["id"],
            "profile_sha256": self.profile_hash,
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "fixtures": [fixture_result(item)
                         for item in self.profile["fixtures"]],
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return QUALITY.evaluate(
            self.profile, result or self.valid_result(), self.profile_hash)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"], report)
        self.assertTrue(any(message in item["message"]
                            for item in report["failures"]), report)

    def test_both_representations_pass_public_oracle(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 2)
        self.assertEqual(report["sample_count"], 2)

    def test_control_must_cross_contact_shell(self) -> None:
        result = self.valid_result()
        result["fixtures"][0]["runs"][0]["final"]["smoke_particles"][0][
            "position_m"][0] = -0.037
        self.assert_rejected(result, "control smoke did not cross")

    def test_coupled_smoke_must_remain_at_declared_clearance(self) -> None:
        result = self.valid_result()
        result["fixtures"][1]["runs"][1]["final"]["smoke_particles"][0][
            "position_m"][0] = -0.033
        self.assert_rejected(result, "contact clearance differs")

    def test_rigid_body_must_receive_reciprocal_response(self) -> None:
        result = self.valid_result()
        result["fixtures"][0]["runs"][1]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "lacks the reciprocal normal response")

    def test_inward_relative_motion_is_rejected(self) -> None:
        result = self.valid_result()
        result["fixtures"][1]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [0.5, 0.25, 0.0]
        result["fixtures"][1]["runs"][1]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.1, 0.0, 0.0]
        self.assert_rejected(result, "inward relative normal motion")


if __name__ == "__main__":
    unittest.main()
