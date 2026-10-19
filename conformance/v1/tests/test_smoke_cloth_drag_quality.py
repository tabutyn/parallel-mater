#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "smoke-cloth-drag-v1.json"
OBLIGATIONS_PATH = ROOT / "qualification" / "capability-obligations-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_smoke_cloth_quality",
                                              ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def particle(velocity: float) -> dict:
    return {
        "slot": 0,
        "position_m": [0.25, 0.25, 0.5],
        "velocity_m_s": [velocity, 0.0, 0.0],
        "age_seconds": 0.06,
        "number_density": 1.0,
        "pressure_m_s2": 0.0,
        "vorticity_per_s": [0.0, 0.0, 0.0],
    }


def snapshot(cloth_velocity: float, smoke_velocity: float) -> dict:
    return {
        "cloth": [
            {"index": index,
             "position_m": [position[0], position[1], position[2]],
             "velocity_m_s": [cloth_velocity, 0.0, 0.0]}
            for index, position in enumerate(
                ([0.15, 0.15, 0.35], [0.35, 0.15, 0.35],
                 [0.25, 0.35, 0.35]))
        ],
        "smoke_particles": [particle(smoke_velocity)],
    }


def fixture(definition: dict) -> dict:
    result = copy.deepcopy(definition)
    result["runs"] = [
        {"id": "control", "initial": snapshot(0.0, 1.0),
         "final": snapshot(0.0, 1.0)},
        {"id": "forward", "initial": snapshot(0.0, 1.0),
         "final": snapshot(0.1, 1.0)},
        {"id": "stationary", "initial": snapshot(0.0, 0.0),
         "final": snapshot(0.0, 0.0)},
        {"id": "reverse", "initial": snapshot(0.0, -1.0),
         "final": snapshot(-0.1, -1.0)},
    ]
    return result


class SmokeClothDragQualityTests(unittest.TestCase):
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

    def test_accepts_bounded_one_way_drag_without_contact(self) -> None:
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

    def test_rejects_stationary_flow_response(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][2]["final"]["cloth"][0][
            "velocity_m_s"][0] = 0.1
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("stationary smoke" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_contact_clearance_violation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"]["smoke_particles"][0][
            "position_m"][2] = 0.36
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("contact clearance" in item["message"]
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

    def test_qualification_contracts_are_registered(self) -> None:
        obligations = json.loads(
            OBLIGATIONS_PATH.read_text(encoding="utf-8"))
        contracts = {item["id"] for item in obligations["contracts"]}
        self.assertTrue({
            "quality:smoke-cloth-drag-v1",
            ("lifecycle:smoke-cloth-grid-drag-lifecycle-v1#"
             "grid-smoke-cloth-drag-capacity-replacement-removal-rollback"),
        }.issubset(contracts))


if __name__ == "__main__":
    unittest.main()
