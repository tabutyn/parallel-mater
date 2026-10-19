#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "fluid-rigid-contact-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_fluid_rigid_contact_quality", ROOT / "quality.py")
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
    impulse = 11.0 / 15.0
    fixture["initial"] = {
        "fluid": fluid([-0.155, 0.0, 0.0], [1.0, 0.0, 0.0]),
        "rigid": rigid([0.0, 0.0, 0.0]),
    }
    fixture["final"] = {
        "fluid": fluid([-0.15, 0.0, 0.0], [1.0 - impulse, 0.0, 0.0]),
        "rigid": rigid([impulse / 2.0, 0.0, 0.0]),
        "contact_events": [{
            "type": "LIQUID_RIGID",
            "first": 0,
            "second": 0,
            "position_m": [-0.1, 0.0, 0.0],
            "normal": [-1.0, 0.0, 0.0],
            "normal_impulse_n_s": impulse,
        }],
    }
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "fixtures": [fixture],
    }


class FluidRigidContactQualityTests(unittest.TestCase):
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

    def test_centered_reciprocal_contact_passes(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 1)
        observation = report["observations"]["centered-exterior-cube-impact"]
        self.assertAlmostEqual(observation["expected_normal_impulse_n_s"],
                               11.0 / 15.0)
        self.assertAlmostEqual(observation[
            "reciprocal_impulse_residual_n_s"], 0.0)

    def test_surface_overlap_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["fluid"]["position_m"][0] = -0.12
        self.assert_rejected(result, "overlapped the rigid surface")

    def test_nonreciprocal_response_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rigid"][
            "linear_velocity_m_s"][0] = 0.25
        self.assert_rejected(result, "impulses are not reciprocal")

    def test_wrong_reduced_mass_impulse_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["fluid"][
            "velocity_m_s"][0] = 0.4
        result["fixtures"][0]["final"]["rigid"][
            "linear_velocity_m_s"][0] = 0.3
        result["fixtures"][0]["final"]["contact_events"][0][
            "normal_impulse_n_s"] = 0.6
        self.assert_rejected(result, "reduced-mass oracle")

    def test_contact_event_must_match_public_state(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["contact_events"][0][
            "normal_impulse_n_s"] = 0.5
        self.assert_rejected(result, "contact impulse differs")

        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["contact_events"][0]["normal"] = [
            1.0, 0.0, 0.0]
        self.assert_rejected(result, "not aligned with the exterior face")

    def test_fixture_cannot_claim_containment(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["containment_claimed"] = True
        self.assert_rejected(result, "fixture setting differs from profile")

    def test_initial_public_state_is_immutable(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["initial"]["fluid"]["position_m"][0] = -0.16
        self.assert_rejected(result, "initial fluid state differs from profile")


if __name__ == "__main__":
    unittest.main()
