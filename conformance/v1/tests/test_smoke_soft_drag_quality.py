#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "smoke-soft-drag-v1.json"
OBLIGATIONS_PATH = ROOT / "qualification" / "capability-obligations-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_smoke_soft_quality",
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


def snapshot(soft_velocity: float, smoke_velocity: float) -> dict:
    nodes = ([0.15, 0.15, 0.35], [0.35, 0.15, 0.35],
             [0.25, 0.35, 0.35], [0.25, 0.216666667, 0.25])
    return {
        "soft_body": [
            {"index": index,
             "position_m": [position[0], position[1], position[2]],
             "velocity_m_s": [soft_velocity, 0.0, 0.0]}
            for index, position in enumerate(nodes)
        ],
        "soft_body_surface_vertices_m": [list(position)
                                          for position in nodes[:3]],
        "smoke_particles": [particle(smoke_velocity)],
    }


def fixture(definition: dict) -> dict:
    result = copy.deepcopy(definition)
    result["runs"] = [
        {"id": "forward-control", "initial": snapshot(0.0, 1.0),
         "final": snapshot(0.0, 1.0)},
        {"id": "forward", "initial": snapshot(0.0, 1.0),
         "final": snapshot(0.1, 1.0)},
        {"id": "stationary-control", "initial": snapshot(0.0, 0.0),
         "final": snapshot(0.0, 0.0)},
        {"id": "stationary", "initial": snapshot(0.0, 0.0),
         "final": snapshot(0.0, 0.0)},
        {"id": "reverse-control", "initial": snapshot(0.0, -1.0),
         "final": snapshot(0.0, -1.0)},
        {"id": "reverse", "initial": snapshot(0.0, -1.0),
         "final": snapshot(-0.1, -1.0)},
    ]
    return result


class SmokeSoftDragQualityTests(unittest.TestCase):
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
        self.assertEqual(report["sample_count"], 6)

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
        result["fixtures"][1]["runs"][3]["final"]["soft_body"][0][
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

    def test_rejects_surface_binding_mismatch(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][0]["runs"][1]["final"][
            "soft_body_surface_vertices_m"][0][0] += 0.01
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("surface" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_unpaired_initial_state(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][5]["initial"]["soft_body"][0][
            "position_m"][0] += 0.01
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("paired initial" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_reverse_run_smoke_mutation(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixtures"][1]["runs"][5]["final"]["smoke_particles"][0][
            "velocity_m_s"][0] = -0.8
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("reverse smoke force" in item["message"]
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
            "quality:smoke-soft-drag-v1",
            ("lifecycle:smoke-soft-drag-lifecycle-v1#"
             "particle-smoke-soft-drag-capacity-replacement-removal-rollback"),
            ("lifecycle:smoke-soft-drag-lifecycle-v1#"
             "grid-smoke-soft-drag-capacity-replacement-removal-rollback"),
        }.issubset(contracts))


if __name__ == "__main__":
    unittest.main()
