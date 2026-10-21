#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "smoke-rigid-drag-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_smoke_rigid_drag_quality",
                                              ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def particle(velocity: float, pressure: float = 0.0,
             position: list[float] | None = None) -> dict:
    return {
        "slot": 0,
        "position_m": list(position or [0.0, 0.0, 0.12]),
        "velocity_m_s": [velocity, 0.0, 0.0],
        "age_seconds": 0.1,
        "number_density": 1.0,
        "pressure_m_s2": pressure,
        "vorticity_per_s": [0.0, 0.0, 0.0],
    }


def snapshot(rigid_velocity: float, smoke_velocity: float,
             rigid_position: list[float],
             pressure: float = 0.0,
             position: list[float] | None = None) -> dict:
    return {
        "rigid": {
            "position_m": list(rigid_position),
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity_m_s": [rigid_velocity, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        },
        "smoke_particles": [particle(smoke_velocity, pressure, position)],
    }


def fixture(item: dict) -> dict:
    value = copy.deepcopy(item)
    center = item["rigid_initial_position_m"]
    smoke_position = [center[0], center[1], center[2] + 0.12]

    def state(rigid_velocity: float, smoke_velocity: float) -> dict:
        return snapshot(rigid_velocity, smoke_velocity, center,
                        position=smoke_position)

    reciprocal_rigid_velocity = 0.0008
    value["runs"] = [
        {"id": "forward-control", "initial": state(0.0, 1.0),
         "final": state(0.0, 1.0)},
        {"id": "forward", "initial": state(0.0, 1.0),
         "final": state(reciprocal_rigid_velocity, 0.9)},
        {"id": "stationary-control", "initial": state(0.0, 0.0),
         "final": state(0.0, 0.0)},
        {"id": "stationary", "initial": state(0.0, 0.0),
         "final": state(0.0, 0.0)},
        {"id": "reverse-control", "initial": state(0.0, -1.0),
         "final": state(0.0, -1.0)},
        {"id": "reverse", "initial": state(0.0, -1.0),
         "final": state(-reciprocal_rigid_velocity, -0.9)},
    ]
    return value


class SmokeRigidDragQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.hash = QUALITY.profile_sha256(PROFILE_PATH)
        self.result = {
            "schema": "parallel-mater-conformance-quality-result/v1",
            "profile_id": self.profile["id"],
            "profile_sha256": self.hash,
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "fixtures": [fixture(item) for item in self.profile["fixtures"]],
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return QUALITY.evaluate(self.profile, result or self.result, self.hash)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in item["message"]
                            for item in report["failures"]), report)

    def test_accepts_particle_and_grid_noncontact_exchange(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["fixture_count"], 2)
        self.assertEqual(report["sample_count"], 6)
        particle_observation = report["observations"][
            "particle-smoke-rigid-drag"]
        self.assertAlmostEqual(
            particle_observation[
                "reciprocal_momentum_residuals_kg_m_s"]["forward"], 0.0)
        self.assertGreater(
            particle_observation["minimum_tracer_rigid_clearance_m"], 0.01)

    def test_rejects_vacuous_rigid_response(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][1]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "rigid drag response is outside bounds")

    def test_rejects_wrong_direction_smoke_response(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [1.1, 0.0, 0.0]
        self.assert_rejected(result, "smoke drag response is outside bounds")

    def test_rejects_vacuous_grid_tracer_delta(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [1.0, 0.0, 0.0]
        self.assert_rejected(result, "smoke drag response is outside bounds")

    def test_rejects_nonreciprocal_particle_momentum_and_force(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.003, 0.0, 0.0]
        self.assert_rejected(result, "momentum changes are not reciprocal")

    def test_rejects_stationary_exchange(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][3]["final"]["rigid"][
            "linear_velocity_m_s"] = [0.02, 0.0, 0.0]
        self.assert_rejected(result, "stationary smoke produced")

    def test_rejects_contact_or_projection(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "position_m"] = [0.25, 0.25, 0.45]
        self.assert_rejected(result, "entered rigid contact clearance")

    def test_rejects_energy_or_speed_growth(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["rigid"][
            "angular_velocity_rad_s"] = [0.0, 0.0, 2.0]
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        messages = {item["message"] for item in report["failures"]}
        self.assertIn("passive drag increased kinetic energy", messages)
        self.assertIn("rigid angular speed exceeds bound", messages)

    def test_rejects_nonexact_fixture_and_snapshot_shapes(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["uncontracted"] = True
        result["fixtures"][1]["runs"][1]["final"]["rigid"]["private_force"] = [
            1.0, 0.0, 0.0]
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        messages = {item["message"] for item in report["failures"]}
        self.assertIn("fixture setting differs from profile", messages)
        self.assertIn("rigid snapshot shape differs from contract", messages)

    def test_rejects_wrong_profile_hash_and_missing_representation(self) -> None:
        result = copy.deepcopy(self.result)
        result["profile_sha256"] = "b" * 64
        result["fixtures"].pop()
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        messages = {item["message"] for item in report["failures"]}
        self.assertIn("profile hash mismatch", messages)
        self.assertIn("missing required fixture", messages)


if __name__ == "__main__":
    unittest.main()
