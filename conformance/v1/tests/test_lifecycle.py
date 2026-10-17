# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "lifecycle.py"
SPEC = importlib.util.spec_from_file_location("conformance_lifecycle", MODULE_PATH)
assert SPEC and SPEC.loader
lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lifecycle
SPEC.loader.exec_module(lifecycle)
PROFILE_PATH = ROOT / "lifecycle" / "adverse-capacity-three-family-v1.json"


def digest(label: str) -> str:
    import hashlib
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def resource(identity: str, generation: int, state: str,
             active: bool = True) -> dict:
    return {
        "id": identity,
        "active": active,
        "generation": generation,
        "state_sha256": digest(state),
    }


def observation(identity: str, outcome: str, resources: list[dict],
                graph: str | None = None) -> dict:
    result = {"id": identity, "outcome": outcome, "resources": resources}
    if graph is not None:
        result["active_graph_sha256"] = digest(graph)
    return result


def valid_result(profile: dict) -> dict:
    unrelated = resource("unrelated-fluid", 10, "unrelated")
    capacity = {
        "id": "capacity-replacement-recovery",
        "execution_status": "completed",
        "families": ["liquid"],
        "unrelated_resources": ["unrelated-fluid"],
        "observations": [
            observation("baseline", "PM_SUCCESS", [
                resource("target-fluid", 1, "target-1"), copy.deepcopy(unrelated)]),
            observation("replacement", "PM_SUCCESS", [
                resource("target-fluid", 2, "target-2"), copy.deepcopy(unrelated)]),
            observation("overflow", "PM_ERROR_CAPACITY_EXCEEDED", [
                resource("target-fluid", 3, "target-empty", False),
                copy.deepcopy(unrelated)]),
            observation("recovery", "PM_SUCCESS", [
                resource("target-fluid", 4, "target-4"), copy.deepcopy(unrelated)]),
        ],
    }

    installed = [
        resource("liquid", 1, "liquid-1"),
        resource("rope", 2, "rope-1"),
        resource("soft-body", 3, "soft-1"),
        resource("smoke", 4, "smoke-1"),
        resource("cloth", 5, "cloth-1"),
    ]
    replaced = copy.deepcopy(installed)
    replaced[0] = resource("liquid", 6, "liquid-2")
    graph = {
        "id": "three-family-partial-rollback",
        "execution_status": "completed",
        "families": ["liquid", "rope", "soft-body"],
        "unrelated_resources": ["smoke", "cloth"],
        "observations": [
            observation("installed", "PM_SUCCESS", copy.deepcopy(installed), "graph-1"),
            observation("updated", "PM_SUCCESS", copy.deepcopy(installed), "graph-2"),
            observation("partial-failure", "PM_ERROR_INVALID_ARGUMENT",
                        copy.deepcopy(installed), "graph-2"),
            observation("recovery", "PM_SUCCESS", copy.deepcopy(installed), "graph-1"),
            observation("replace-liquid", "PM_SUCCESS", replaced, "graph-1"),
        ],
    }
    return {
        "schema": lifecycle.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": lifecycle.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "cases": [capacity, graph],
    }


class LifecycleTests(unittest.TestCase):
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

    def test_complete_vendor_neutral_lifecycle_passes(self):
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["checkpoint_count"], 9)

    def test_profile_hash_is_checkout_line_ending_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            lf.write_bytes(b'{\n  "value": 1\n}\n')
            crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
            self.assertEqual(lifecycle.profile_sha256(lf),
                             lifecycle.profile_sha256(crlf))

    def test_suite_or_case_unsupported_never_passes(self):
        result = valid_result(self.profile)
        result["execution_status"] = "unsupported"
        self.assert_rejected(result, "suite must be completed")

        result = valid_result(self.profile)
        result["cases"][1]["execution_status"] = "unsupported"
        self.assert_rejected(result, "execution must be completed")

    def test_execution_failure_never_passes(self):
        result = valid_result(self.profile)
        result["cases"][0]["execution_status"] = "failed"
        self.assert_rejected(result, "execution must be completed")

    def test_wrong_capacity_outcome_fails(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][2]["outcome"] = "PM_SUCCESS"
        self.assert_rejected(result, "operation outcome mismatch")

    def test_replacement_and_recovery_require_new_generations(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][1]["resources"][0]["generation"] = 1
        self.assert_rejected(result, "generation must differ")

        result = valid_result(self.profile)
        result["cases"][0]["observations"][3]["resources"][0]["generation"] = 3
        self.assert_rejected(result, "generation must differ")

        result = valid_result(self.profile)
        result["cases"][0]["observations"][1]["resources"][0][
            "state_sha256"] = digest("target-1")
        self.assert_rejected(result, "state_sha256 must differ")

    def test_overflow_must_retire_target_and_preserve_unrelated(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][2]["resources"][0]["active"] = True
        self.assert_rejected(result, "active state mismatch")

        result = valid_result(self.profile)
        result["cases"][0]["observations"][2]["resources"][1][
            "state_sha256"] = digest("corrupt-unrelated")
        self.assert_rejected(result, "state_sha256 must match")

    def test_partial_failure_must_restore_prior_graph(self):
        result = valid_result(self.profile)
        result["cases"][1]["observations"][2][
            "active_graph_sha256"] = digest("partial-graph")
        self.assert_rejected(result, "active graph must match")

    def test_three_family_replacement_preserves_other_families(self):
        result = valid_result(self.profile)
        result["cases"][1]["observations"][4]["resources"][1][
            "generation"] += 1
        self.assert_rejected(result, "generation must match")

    def test_missing_checkpoint_or_resource_fails(self):
        result = valid_result(self.profile)
        result["cases"][1]["observations"].pop()
        self.assert_rejected(result, "missing required checkpoint")

        result = valid_result(self.profile)
        result["cases"][0]["observations"][0]["resources"].pop()
        self.assert_rejected(result, "tracked resource set mismatch")

    def test_profile_hash_driver_and_hash_format_are_required(self):
        result = valid_result(self.profile)
        result["profile_sha256"] = "0" * 64
        self.assert_rejected(result, "profile hash mismatch")

        result = valid_result(self.profile)
        result.pop("driver")
        self.assert_rejected(result, "driver name and device are required")

        result = valid_result(self.profile)
        result["cases"][0]["observations"][0]["resources"][0][
            "state_sha256"] = "not-a-hash"
        self.assert_rejected(result, "expected a lowercase SHA-256")

    def test_families_and_exact_case_set_are_required(self):
        result = valid_result(self.profile)
        result["cases"][1]["families"].pop()
        self.assert_rejected(result, "families mismatch")

        result = valid_result(self.profile)
        result["cases"].pop()
        self.assert_rejected(result, "missing required lifecycle case")


if __name__ == "__main__":
    unittest.main()
