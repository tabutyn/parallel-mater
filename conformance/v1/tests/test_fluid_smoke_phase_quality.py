#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "fluid-smoke-phase-v1.json"
OBLIGATIONS_PATH = ROOT / "qualification" / "capability-obligations-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_fluid_smoke_phase_quality",
                                              ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def fluid(stable_id: int, velocity: list[float]) -> dict:
    return {
        "position_m": [0.25, 0.25, 0.25],
        "velocity_m_s": velocity,
        "acceleration_m_s2": [0.0, 0.0, 0.0],
        "temperature_celsius": 120.0,
        "stable_id": stable_id,
    }


def smoke(velocity: list[float], age: float) -> dict:
    return {
        "slot": 0,
        "position_m": [0.25, 0.25, 0.25],
        "velocity_m_s": velocity,
        "age_seconds": age,
        "number_density": 0.0,
        "pressure_m_s2": 0.0,
        "vorticity_per_s": [0.0, 0.0, 0.0],
    }


def snapshot(fluid_particles: list[dict], smoke_particles: list[dict]) -> dict:
    return {"fluid": fluid_particles, "smoke_particles": smoke_particles}


def fixture(definition: dict) -> dict:
    result = copy.deepcopy(definition)
    single = list(definition["single_source_velocity_m_s"])
    bounded = [list(value)
               for value in definition["bounded_source_velocities_m_s"]]
    timestep = definition["timestep_seconds"]
    result["runs"] = [
        {
            "id": "below-threshold-control",
            "initial": snapshot([fluid(0, single)], []),
            "final": snapshot([fluid(0, single)], []),
        },
        {
            "id": "single-transfer",
            "initial": snapshot([fluid(0, single)], []),
            "final": snapshot([], [smoke(single, timestep)]),
        },
        {
            "id": "capacity-bound",
            "initial": snapshot(
                [fluid(index, velocity)
                 for index, velocity in enumerate(bounded)], []),
            "final": snapshot(
                [fluid(2, bounded[2])],
                [smoke(bounded[0], timestep),
                 smoke(bounded[1], timestep)]),
        },
    ]
    return result


class FluidSmokePhaseQualityTests(unittest.TestCase):
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

    def test_accepts_bounded_phase_accounting(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["sample_count"], 4)

    def test_rejects_unbalanced_mass_count(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"].clear()
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("mass accounting" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_momentum_and_energy_creation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][1]["final"]["smoke_particles"][0][
            "velocity_m_s"] = [2.0, 0.0, 0.0]
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("momentum accounting" in item["message"]
                            for item in report["failures"]), report)
        self.assertTrue(any("kinetic-energy accounting" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_invalid_surviving_identity(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][2]["final"]["fluid"][0][
            "stable_id"] = 99
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("stable IDs" in item["message"]
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
            "quality:fluid-smoke-phase-v1",
            ("lifecycle:fluid-smoke-phase-lifecycle-v1#"
             "particle-fluid-smoke-phase-capacity-replacement-removal-rollback"),
            ("lifecycle:fluid-smoke-phase-lifecycle-v1#"
             "grid-fluid-smoke-phase-capacity-replacement-removal-rollback"),
        }.issubset(contracts))


if __name__ == "__main__":
    unittest.main()
