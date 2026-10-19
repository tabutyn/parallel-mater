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
PROFILE_PATH = (ROOT / "lifecycle" /
                "fluid-rigid-containment-lifecycle-v1.json")
SPEC = importlib.util.spec_from_file_location(
    "pm_fluid_rigid_containment_lifecycle", ROOT / "lifecycle.py")
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


def valid_result(profile: dict) -> dict:
    endpoints = resource("fluid-rigid-endpoints", 1, "stable-endpoints")
    checkpoints = [
        ("baseline", "PM_SUCCESS", resource(
            "fluid-rigid-containment-graph", 1, "baseline")),
        ("replacement", "PM_SUCCESS", resource(
            "fluid-rigid-containment-graph", 2, "replacement")),
        ("rejected-overflow", "PM_ERROR_CAPACITY_EXCEEDED", resource(
            "fluid-rigid-containment-graph", 2, "replacement")),
        ("removal", "PM_SUCCESS", resource(
            "fluid-rigid-containment-graph", 3, "removed", False)),
        ("recovery", "PM_SUCCESS", resource(
            "fluid-rigid-containment-graph", 4, "recovery")),
    ]
    observations = [{"id": identity, "outcome": outcome,
                     "resources": [target, copy.deepcopy(endpoints)]}
                    for identity, outcome, target in checkpoints]
    return {
        "schema": LIFECYCLE.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": LIFECYCLE.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "cases": [{
            "id": "fluid-rigid-containment-capacity-replacement-removal-rollback",
            "execution_status": "completed",
            "families": ["fluid", "rigid_body"],
            "unrelated_resources": ["fluid-rigid-endpoints"],
            "observations": observations,
        }],
    }


class FluidRigidContainmentLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = LIFECYCLE.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return LIFECYCLE.evaluate(self.profile, result, self.profile_hash)

    def test_exact_lifecycle_passes(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["checkpoint_count"], 5)

    def test_overflow_must_preserve_installed_graph(self) -> None:
        result = valid_result(self.profile)
        result["cases"][0]["observations"][2]["resources"][0][
            "state_sha256"] = digest("partial-install")
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("state_sha256 must match" in item["message"]
                            for item in report["failures"]), report)

    def test_removal_and_recovery_are_required(self) -> None:
        result = valid_result(self.profile)
        result["cases"][0]["observations"][3]["resources"][0]["active"] = True
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("active state mismatch" in item["message"]
                            for item in report["failures"]), report)

    def test_endpoints_remain_stable_across_graph_mutations(self) -> None:
        result = valid_result(self.profile)
        result["cases"][0]["observations"][4]["resources"][1][
            "state_sha256"] = digest("mutated-endpoints")
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("state_sha256 must match" in item["message"]
                            for item in report["failures"]), report)


if __name__ == "__main__":
    unittest.main()
