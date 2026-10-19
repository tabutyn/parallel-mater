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
SPEC = importlib.util.spec_from_file_location("smoke_conformance_lifecycle", MODULE_PATH)
assert SPEC and SPEC.loader
lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lifecycle
SPEC.loader.exec_module(lifecycle)
PROFILE_PATH = ROOT / "lifecycle" / "smoke-self-pressure-lifecycle-v1.json"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def resource(identity: str, generation: int, state: str,
             active: bool = True) -> dict:
    return {
        "id": identity,
        "active": active,
        "generation": generation,
        "state_sha256": digest(state),
    }


def smoke_case(representation: str) -> dict:
    target = f"{representation}-target-smoke"
    control = f"{representation}-control-smoke"
    stable_control = resource(control, 1, f"{representation}-control")

    def observation(identity: str, outcome: str, target_state: dict) -> dict:
        return {
            "id": identity,
            "outcome": outcome,
            "resources": [target_state, copy.deepcopy(stable_control)],
        }

    return {
        "id": f"smoke-{representation}-capacity-replacement-removal-rollback",
        "execution_status": "completed",
        "families": ["smoke"],
        "unrelated_resources": [control],
        "observations": [
            observation("baseline", "PM_SUCCESS",
                        resource(target, 1, f"{representation}-baseline")),
            observation("replacement", "PM_SUCCESS",
                        resource(target, 2, f"{representation}-replacement")),
            observation("rejected-overflow", "PM_ERROR_CAPACITY_EXCEEDED",
                        resource(target, 2, f"{representation}-replacement")),
            observation("removal", "PM_SUCCESS",
                        resource(target, 3, f"{representation}-removed", False)),
            observation("recovery", "PM_SUCCESS",
                        resource(target, 4, f"{representation}-recovery")),
        ],
    }


def valid_result(profile: dict) -> dict:
    return {
        "schema": lifecycle.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": lifecycle.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "cases": [smoke_case("particle"), smoke_case("grid")],
    }


class SmokeLifecycleTests(unittest.TestCase):
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

    def test_particle_and_grid_lifecycles_pass(self):
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["checkpoint_count"], 10)

    def test_rejection_preserves_last_usable_state(self):
        result = valid_result(self.profile)
        rejected = result["cases"][0]["observations"][2]["resources"][0]
        rejected["generation"] += 1
        self.assert_rejected(result, "generation must match")

        result = valid_result(self.profile)
        rejected = result["cases"][1]["observations"][2]["resources"][0]
        rejected["state_sha256"] = digest("lost-grid-state")
        self.assert_rejected(result, "state_sha256 must match")

    def test_removal_and_recovery_are_exact(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][3]["resources"][0]["active"] = True
        self.assert_rejected(result, "active state mismatch")

        result = valid_result(self.profile)
        result["cases"][1]["observations"][4]["resources"][0]["generation"] = 3
        self.assert_rejected(result, "generation must differ")

    def test_control_smoke_is_not_mutated(self):
        result = valid_result(self.profile)
        result["cases"][0]["observations"][4]["resources"][1][
            "state_sha256"] = digest("changed-control")
        self.assert_rejected(result, "state_sha256 must match")

    def test_both_representations_are_required(self):
        result = valid_result(self.profile)
        result["cases"].pop()
        self.assert_rejected(result, "missing required lifecycle case")


if __name__ == "__main__":
    unittest.main()
