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
PROFILE_PATH = ROOT / "quality" / "externally-heated-fluid-v1.json"
OBLIGATIONS_PATH = ROOT / "qualification" / "capability-obligations-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_external_heat_quality",
                                              ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def smoke_particle() -> dict:
    return {
        "slot": 0,
        "position_m": [0.3, 0.25, 0.5],
        "velocity_m_s": [1.0, 0.0, 0.0],
        "age_seconds": 0.05,
        "number_density": 1.0,
        "pressure_m_s2": 0.0,
        "vorticity_per_s": [0.0, 0.0, 0.0],
    }


def snapshot(temperature: float, position: list[float]) -> dict:
    return {
        "fluid": {
            "position_m": position,
            "velocity_m_s": [0.0, 0.0, 0.0],
            "acceleration_m_s2": [0.0, 0.0, 0.0],
            "temperature_celsius": temperature,
            "stable_id": 0,
        },
        "smoke_particles": [smoke_particle()],
    }


def fixture(definition: dict) -> dict:
    result = copy.deepcopy(definition)
    initial = definition["fluid_initial_temperature_celsius"]
    target = definition["heater_temperature_celsius"]
    rate = definition["test_heat_transfer_rate_per_s"]
    timestep = definition["timestep_seconds"]
    heated = initial + (target - initial) * (1.0 - math.exp(-rate * timestep))
    inside = definition["fluid_initial_position_m"]
    outside = definition["outside_fluid_position_m"]
    result["runs"] = [
        {"id": "control", "initial": snapshot(initial, inside),
         "final": snapshot(initial, inside)},
        {"id": "heated", "initial": snapshot(initial, inside),
         "final": snapshot(heated, inside)},
        {"id": "equilibrium", "initial": snapshot(target, inside),
         "final": snapshot(target, inside)},
        {"id": "outside", "initial": snapshot(initial, outside),
         "final": snapshot(initial, outside)},
    ]
    return result


class ExternallyHeatedFluidQualityTests(unittest.TestCase):
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

    def test_accepts_exponential_heating_and_retained_endpoints(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["sample_count"], 8)

    def test_rejects_wrong_heated_temperature(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["fluid"][
            "temperature_celsius"] += 0.5
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("heater oracle" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_heating_outside_plate(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][3]["final"]["fluid"][
            "temperature_celsius"] += 1.0
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("heater oracle" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_smoke_endpoint_mutation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"][0] = 0.5
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("smoke endpoint" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_missing_representation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"].pop()
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("missing required fixture" in item["message"]
                            for item in report["failures"]), report)

    def test_qualification_contracts_are_registered(self) -> None:
        obligations = json.loads(
            OBLIGATIONS_PATH.read_text(encoding="utf-8"))
        contracts = {item["id"] for item in obligations["contracts"]}
        self.assertTrue({
            "quality:externally-heated-fluid-v1",
            ("lifecycle:externally-heated-fluid-lifecycle-v1#"
             "particle-externally-heated-fluid-capacity-replacement-removal-rollback"),
            ("lifecycle:externally-heated-fluid-lifecycle-v1#"
             "grid-externally-heated-fluid-capacity-replacement-removal-rollback"),
        }.issubset(contracts))


if __name__ == "__main__":
    unittest.main()
