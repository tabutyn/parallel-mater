#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "smoke-fluid-drag-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_smoke_fluid_quality",
                                              ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def particle() -> dict:
    return {
        "slot": 0,
        "position_m": [0.3, 0.25, 0.5],
        "velocity_m_s": [1.0, 0.0, 0.0],
        "age_seconds": 0.05,
        "number_density": 1.0,
        "pressure_m_s2": 0.0,
        "vorticity_per_s": [0.0, 0.0, 0.0],
    }


def snapshot(velocity: float) -> dict:
    return {
        "fluid": {
            "position_m": [0.25, 0.25, 0.5],
            "velocity_m_s": [velocity, 0.0, 0.0],
            "acceleration_m_s2": [0.0, 0.0, 0.0],
            "temperature_celsius": 20.0,
            "stable_id": 0,
        },
        "smoke_particles": [particle()],
    }


def fixture(definition: dict) -> dict:
    result = copy.deepcopy(definition)
    result["runs"] = [
        {"id": "control", "initial": snapshot(0.0), "final": snapshot(0.0)},
        {"id": "drag", "initial": snapshot(0.0), "final": snapshot(0.1)},
        {"id": "matched-control", "initial": snapshot(1.0),
         "final": snapshot(1.0)},
        {"id": "matched-drag", "initial": snapshot(1.0),
         "final": snapshot(1.0)},
    ]
    return result


class SmokeFluidDragQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.result = {
            "schema": "parallel-mater-conformance-quality-result/v1",
            "profile_id": self.profile["id"],
            "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "fixtures": [fixture(item) for item in self.profile["fixtures"]],
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return QUALITY.evaluate(self.profile, result or self.result,
                                QUALITY.profile_sha256(PROFILE_PATH))

    def test_accepts_one_way_velocity_dependent_drag(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["sample_count"], 4)

    def test_rejects_reverse_smoke_force(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"][0] = 0.9
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("reverse smoke force" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_drag_when_velocities_match(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][3]["final"]["fluid"][
            "velocity_m_s"][0] = 0.95
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("matched flow produced drag" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_missing_representation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"].pop()
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("missing required fixture" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_noncanonical_grid_fixture(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["smoke_grid_resolution"] = 32
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("fixture setting differs" in item["message"]
                            for item in report["failures"]), report)


if __name__ == "__main__":
    unittest.main()
