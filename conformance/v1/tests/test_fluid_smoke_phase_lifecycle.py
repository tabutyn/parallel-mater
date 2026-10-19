#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "lifecycle" / "fluid-smoke-phase-lifecycle-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_fluid_smoke_phase_lifecycle",
                                              ROOT / "lifecycle.py")
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIFECYCLE
SPEC.loader.exec_module(LIFECYCLE)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def resource(identity: str, generation: int, state: str,
             active: bool = True) -> dict:
    return {"id": identity, "active": active, "generation": generation,
            "state_sha256": digest(state)}


def case_result(definition: dict) -> dict:
    graph = definition["tracked_resources"][0]
    endpoints = definition["tracked_resources"][1]
    stable = resource(endpoints, 1, "stable-endpoints")
    checkpoints = [
        ("baseline", "PM_SUCCESS", resource(graph, 1, "baseline")),
        ("replacement", "PM_SUCCESS", resource(graph, 2, "replacement")),
        ("rejected-overflow", "PM_ERROR_CAPACITY_EXCEEDED",
         resource(graph, 2, "replacement")),
        ("removal", "PM_SUCCESS", resource(graph, 3, "removed", False)),
        ("recovery", "PM_SUCCESS", resource(graph, 4, "recovery")),
    ]
    return {
        "id": definition["id"],
        "execution_status": "completed",
        "families": definition["families"],
        "unrelated_resources": definition["unrelated_resources"],
        "observations": [
            {"id": identity, "outcome": outcome,
             "resources": [target, copy.deepcopy(stable)]}
            for identity, outcome, target in checkpoints
        ],
    }


class FluidSmokePhaseLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = LIFECYCLE.profile_sha256(PROFILE_PATH)

    def valid_result(self) -> dict:
        return {
            "schema": LIFECYCLE.RESULT_SCHEMA,
            "profile_id": self.profile["id"],
            "profile_sha256": self.profile_hash,
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "cases": [case_result(item) for item in self.profile["cases"]],
        }

    def test_both_representation_lifecycles_pass(self) -> None:
        report = LIFECYCLE.evaluate(self.profile, self.valid_result(),
                                    self.profile_hash)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["checkpoint_count"], 10)

    def test_overflow_must_preserve_phase_graph(self) -> None:
        result = self.valid_result()
        result["cases"][0]["observations"][2]["resources"][0][
            "state_sha256"] = digest("partial-install")
        report = LIFECYCLE.evaluate(self.profile, result, self.profile_hash)
        self.assertFalse(report["passed"])
        self.assertTrue(any("state_sha256 must match" in item["message"]
                            for item in report["failures"]), report)

    def test_endpoints_remain_stable(self) -> None:
        result = self.valid_result()
        result["cases"][1]["observations"][4]["resources"][1][
            "state_sha256"] = digest("mutated-endpoints")
        report = LIFECYCLE.evaluate(self.profile, result, self.profile_hash)
        self.assertFalse(report["passed"])
        self.assertTrue(any("state_sha256 must match" in item["message"]
                            for item in report["failures"]), report)


if __name__ == "__main__":
    unittest.main()
