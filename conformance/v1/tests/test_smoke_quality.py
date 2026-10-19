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
MODULE_PATH = ROOT / "quality.py"
PROFILE_PATH = ROOT / "quality" / "smoke-self-pressure-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def particle(index: int, age: float, density: float, pressure: float) -> dict:
    return {
        "index": index,
        "position_m": [0.0, 0.0, 0.0],
        "velocity_m_s": [1.0, 0.0, 0.0],
        "age_seconds": age,
        "number_density": density,
        "pressure": pressure,
        "vorticity_per_s": [0.0, 0.0, 0.0],
    }


class SmokeQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        fixtures = []
        for definition in self.profile["fixtures"]:
            fixture = copy.deepcopy(definition)
            fixture["first_particles"] = [
                particle(index, 0.0, 0.0, 0.0) for index in range(32)]
            second_pressure = (4.0 if definition["representation"] == "particle"
                               else 0.0)
            fixture["second_particles"] = [
                particle(index, 0.015625, 2.0, second_pressure)
                for index in range(32)] + [
                particle(index, 0.0, 0.0, 0.0)
                for index in range(32, 64)]
            first_final_pressure = (
                4.0 + 4.0 * math.exp(-0.015625 / 0.05)
                if definition["representation"] == "particle" else 0.5)
            second_final_pressure = (
                4.0 if definition["representation"] == "particle" else 0.0)
            fixture["final_particles"] = [
                particle(index, 0.03125, 2.0, first_final_pressure)
                for index in range(32)] + [
                particle(index, 0.015625, 2.0, second_final_pressure)
                for index in range(32, 64)] + [
                particle(index, 0.0, 0.0, 0.0)
                for index in range(64, 96)]
            fixtures.append(fixture)
        self.result = {
            "schema": "parallel-mater-conformance-quality-result/v1",
            "profile_id": self.profile["id"],
            "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "fixtures": fixtures,
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return QUALITY.evaluate(self.profile, result or self.result,
                                QUALITY.profile_sha256(PROFILE_PATH))

    def test_accepts_particle_equation_and_grid_activity(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["sample_count"], 96)

    def test_rejects_particle_pressure_not_matching_density(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["final_particles"][0]["pressure"] = 3.0
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("density oracle" in failure["message"]
                            for failure in report["failures"]))

    def test_rejects_inactive_grid_pressure(self) -> None:
        result = copy.deepcopy(self.result)
        for item in result["fixtures"][1]["final_particles"]:
            item["pressure"] = 0.0
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("too small" in failure["message"]
                            for failure in report["failures"]))

    def test_rejects_wrong_particle_count(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["final_particles"].pop()
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("particle count" in failure["message"]
                            for failure in report["failures"]))

    def test_rejects_particle_net_velocity_drift(self) -> None:
        result = copy.deepcopy(self.result)
        for item in result["fixtures"][0]["final_particles"]:
            item["velocity_m_s"][0] += 0.1
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("net velocity drift" in failure["message"]
                            for failure in report["failures"]))


if __name__ == "__main__":
    unittest.main()
