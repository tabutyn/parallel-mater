# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "lifecycle.py"
SPEC = importlib.util.spec_from_file_location("containment_lifecycle", MODULE_PATH)
assert SPEC and SPEC.loader
lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lifecycle
SPEC.loader.exec_module(lifecycle)
PROFILE_PATH = (ROOT / "lifecycle" /
                "fluid-cloth-containment-lifecycle-v1.json")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def resources() -> list[dict]:
    return [
        {"id": "fluid", "active": True, "generation": 1,
         "state_sha256": digest("fluid-state")},
        {"id": "cloth", "active": True, "generation": 1,
         "state_sha256": digest("cloth-state")},
    ]


def observation(identity: str, outcome: str, graph: str) -> dict:
    return {
        "id": identity,
        "outcome": outcome,
        "active_graph_sha256": digest(graph),
        "resources": resources(),
    }


def valid_result(profile: dict) -> dict:
    return {
        "schema": lifecycle.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": lifecycle.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "cases": [{
            "id": "fluid-cloth-containment-capacity-replacement-removal-rollback",
            "execution_status": "completed",
            "families": ["fluid", "cloth"],
            "unrelated_resources": [],
            "observations": [
                observation("baseline", "PM_SUCCESS", "baseline"),
                observation("replacement", "PM_SUCCESS", "replacement"),
                observation("rejected-capacity",
                            "PM_ERROR_CAPACITY_EXCEEDED", "replacement"),
                observation("partial-failure", "PM_ERROR_INVALID_STATE",
                            "replacement"),
                observation("removal", "PM_SUCCESS", "empty"),
                observation("recovery", "PM_SUCCESS", "baseline"),
            ],
        }],
    }


class FluidClothContainmentLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = lifecycle.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return lifecycle.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_exact_graph_lifecycle_passes(self):
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["checkpoint_count"], 6)

    def test_capacity_and_runtime_failures_preserve_graph(self):
        for checkpoint in (2, 3):
            with self.subTest(checkpoint=checkpoint):
                result = valid_result(self.profile)
                result["cases"][0]["observations"][checkpoint][
                    "active_graph_sha256"] = digest("lost-active-graph")
                self.assert_rejected(result, "active graph must match")

    def test_rejected_mutation_preserves_endpoint_state(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][3]["resources"][1][
            "state_sha256"] = digest("mutated-cloth")
        self.assert_rejected(result, "state_sha256 must match")

    def test_removal_and_recovery_use_exact_graphs(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][4][
            "active_graph_sha256"] = digest("replacement")
        self.assert_rejected(result, "active graph must differ")

        result = valid_result(self.profile)
        result["cases"][0]["observations"][5][
            "active_graph_sha256"] = digest("not-baseline")
        self.assert_rejected(result, "active graph must match")

    def test_exact_outcomes_are_required(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][2]["outcome"] = "PM_SUCCESS"
        self.assert_rejected(result, "operation outcome mismatch")


if __name__ == "__main__":
    unittest.main()
