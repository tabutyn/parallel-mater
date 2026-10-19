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
PROFILE_PATH = ROOT / "lifecycle" / "cloth-smoke-contact-lifecycle-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_cloth_smoke_contact_lifecycle", ROOT / "lifecycle.py")
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
    graph, endpoints = definition["tracked_resources"]
    contact_active = resource(endpoints, 1, "contact-active")
    contact_inactive = resource(endpoints, 1, "contact-inactive")
    checkpoints = [
        ("baseline", "PM_SUCCESS", resource(graph, 1, "baseline"),
         contact_active),
        ("replacement", "PM_SUCCESS", resource(graph, 2, "replacement"),
         contact_active),
        ("rejected-overflow", "PM_ERROR_CAPACITY_EXCEEDED",
         resource(graph, 2, "replacement"), contact_active),
        ("removal", "PM_SUCCESS", resource(graph, 3, "removed", False),
         contact_inactive),
        ("recovery", "PM_SUCCESS", resource(graph, 4, "recovery"),
         contact_active),
    ]
    return {
        "id": definition["id"],
        "execution_status": "completed",
        "families": definition["families"],
        "unrelated_resources": definition["unrelated_resources"],
        "observations": [
            {"id": identity, "outcome": outcome,
             "resources": [target, copy.deepcopy(endpoint)]}
            for identity, outcome, target, endpoint in checkpoints
        ],
    }


class ClothSmokeContactLifecycleTests(unittest.TestCase):
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
            "cases": [case_result(self.profile["cases"][0])],
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return LIFECYCLE.evaluate(
            self.profile, result or self.valid_result(), self.profile_hash)

    def test_grid_lifecycle_passes_all_five_checkpoints(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["checkpoint_count"], 5)

    def test_overflow_must_preserve_graph(self) -> None:
        result = self.valid_result()
        result["cases"][0]["observations"][2]["resources"][0][
            "state_sha256"] = digest("partial-install")
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("state_sha256 must match" in item["message"]
                            for item in report["failures"]), report)

    def test_removal_must_advance_graph_generation(self) -> None:
        result = self.valid_result()
        result["cases"][0]["observations"][3]["resources"][0][
            "generation"] = 2
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("generation must differ" in item["message"]
                            for item in report["failures"]), report)

    def test_endpoints_remain_stable(self) -> None:
        result = self.valid_result()
        result["cases"][0]["observations"][4]["resources"][1][
            "state_sha256"] = digest("mutated-endpoints")
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("state_sha256 must match" in item["message"]
                            for item in report["failures"]), report)


if __name__ == "__main__":
    unittest.main()
