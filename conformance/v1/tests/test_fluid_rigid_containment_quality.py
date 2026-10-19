#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "fluid-rigid-containment-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_fluid_rigid_containment_quality", ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def fluid(position: list[float], velocity: list[float]) -> dict:
    return {
        "stable_id": 0,
        "position_m": position,
        "velocity_m_s": velocity,
        "acceleration_m_s2": [0.0, 0.0, 0.0],
    }


def rigid(linear_velocity: list[float]) -> dict:
    return {
        "position_m": [0.0, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity_m_s": linear_velocity,
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }


def valid_result(profile: dict) -> dict:
    fixture = copy.deepcopy(profile["fixtures"][0])
    impulse = 2.0 / 3.0
    fixture["initial"] = {
        "fluid": fluid([0.045, 0.0, 0.0], [1.0, 0.0, 0.0]),
        "rigid": rigid([0.0, 0.0, 0.0]),
    }
    fixture["final"] = {
        "fluid": fluid([0.05, 0.0, 0.0], [1.0 - impulse, 0.0, 0.0]),
        "rigid": rigid([impulse / 2.0, 0.0, 0.0]),
    }
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "fixtures": [fixture],
    }


class FluidRigidContainmentQualityTests(unittest.TestCase):
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

    def test_closed_cube_reciprocal_containment_passes(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 1)
        observation = report["observations"][
            "closed-cube-interior-outward-pulse"]
        self.assertAlmostEqual(observation[
            "reciprocal_momentum_residual_kg_m_s"], 0.0)
        self.assertEqual(observation["leaked_particle_count"], 0)

    def test_leaked_particle_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["fluid"]["position_m"][0] = 0.06
        self.assert_rejected(result, "leaked outside the closed rigid geometry")

    def test_nonreciprocal_response_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rigid"][
            "linear_velocity_m_s"][0] = 0.2
        self.assert_rejected(result, "momentum changes are not reciprocal")

    def test_reaction_must_be_non_vacuous(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["fluid"][
            "velocity_m_s"] = [1.0, 0.0, 0.0]
        result["fixtures"][0]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "reciprocal reaction is too small")

    def test_fixture_cannot_claim_exterior_contact(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["contact_claimed"] = True
        self.assert_rejected(result, "fixture setting differs from profile")

    def test_initial_public_state_is_immutable(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["initial"]["fluid"]["position_m"][0] = 0.044
        self.assert_rejected(result, "initial fluid state differs from profile")

    def test_initial_rigid_public_state_is_immutable(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["initial"]["rigid"][
            "linear_velocity_m_s"][0] = 0.1
        self.assert_rejected(result, "initial rigid state differs from profile")

    def test_rigid_orientation_must_be_normalized(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rigid"][
            "orientation_xyzw"][3] = 2.0
        self.assert_rejected(result, "unit quaternion")


if __name__ == "__main__":
    unittest.main()
