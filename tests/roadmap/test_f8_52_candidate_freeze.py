import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from check_realtime_candidate import (  # noqa: E402
    ACTIVE_ID,
    ACTIVE_SOURCE_REVISION,
    CANDIDATE_PATH,
    REQUIRED_CLOSURES,
    validate_candidate,
)

PLANNED_ID = "realtime-single-step-v1-5bacbe2"
PLANNED_SOURCE_REVISION = "5bacbe2b1e0fbfad565fb9477bb6873396e31d5d"
PLANNED_SOURCE_TREE = "67f5088c2e9f50938c55856e3683326e42675310"
PLANNED_MANIFEST_SHA256 = (
    "29e051d94a89a39d51c2128b212165d60da6da7de4b2c107a72ae7cc5b77f20a"
)
PLANNED_CI_RUN_ID = 41708083151
PLANNED_QUALIFICATION_TASKS = {f"F8-{number}" for number in range(26, 31)}


class F870CandidateFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    def test_rejected_candidate_is_immutable_after_refreeze(self):
        expected_transition_errors = {
            f"{task_id} {message}"
            for task_id in PLANNED_QUALIFICATION_TASKS
            for message in (
                "does not bind exact source identity",
                "does not bind exact candidate manifest",
                "does not bind exact protected CI run",
                "does not preserve candidate history",
            )
        }
        self.assertEqual(
            sorted(validate_candidate()),
            sorted(expected_transition_errors),
        )
        self.assertEqual(self.candidate["task"], "F8-70")
        rejection = json.loads((
            ROOT / "conformance/v1/qualification/"
            "realtime-candidate-v8-rejection.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(
            rejection["status"],
            "rejected-required-cpu-gallery-suite",
        )
        self.assertEqual(rejection["replacement"]["freeze_task"], "F8-70")
        self.assertEqual(rejection["replacement"]["candidate_id"], ACTIVE_ID)
        self.assertFalse(rejection["evidence_union_allowed"])

    def test_every_required_protected_drift_is_rejected(self):
        closures = {
            item["category"]: item
            for item in self.candidate["source"]["protected_closures"]
        }
        self.assertEqual(set(closures), REQUIRED_CLOSURES)
        for category in REQUIRED_CLOSURES:
            with self.subTest(category=category):
                tampered = copy.deepcopy(self.candidate)
                for closure in tampered["source"]["protected_closures"]:
                    if closure["category"] == category:
                        closure["sha256"] = "0" * 64
                        break
                errors = validate_candidate(tampered)
                self.assertIn(f"{category} closure hash drift", errors)

    def test_source_identity_drift_is_rejected(self):
        tampered = copy.deepcopy(self.candidate)
        tampered["source"]["revision"] = "0" * 40
        tampered["source"]["tree"] = "0" * 40
        errors = validate_candidate(tampered)
        self.assertIn("active source revision drift", errors)
        self.assertIn("active source tree drift", errors)

    def test_scenario_and_capture_contract_drift_is_rejected(self):
        tampered = copy.deepcopy(self.candidate)
        tampered["contract_inputs"]["scenario_fixtures"]["sha256"] = "0" * 64
        tampered["contract_inputs"]["capture_validator"]["sha256"] = "0" * 64
        tampered["contract_inputs"]["capture_producer"]["sha256"] = "0" * 64
        errors = validate_candidate(tampered)
        self.assertIn("scenario fixture contract hash drift", errors)
        self.assertIn("capture validator contract hash drift", errors)
        self.assertIn("capture producer hash drift", errors)

    def test_ipad_ci_does_not_dirty_the_source_checkout(self):
        workflow = (ROOT / ".github/workflows/mobile-gallery.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('$RUNNER_TEMP/xcode-version.txt', workflow)
        self.assertNotIn("tee xcode-version.txt", workflow)

    def test_only_final_candidate_can_receive_qualification_evidence(self):
        routing = self.candidate["qualification_routing"]
        self.assertEqual(routing["accepted_candidate"], ACTIVE_ID)
        self.assertFalse(routing["evidence_union_allowed"])
        self.assertEqual(
            self.candidate["supersedes"]["candidate_id"],
            "realtime-single-step-v1-c05d320",
        )
        for task_id in routing["tasks"]:
            task = json.loads(
                (ROOT / "docs/roadmap/tasks" / f"{task_id}.json")
                .read_text(encoding="utf-8")
            )
            if task_id in PLANNED_QUALIFICATION_TASKS:
                self.assertIn("F8-74", task["dependencies"])
                self.assertIn(PLANNED_ID, task["scope"])
                self.assertIn(PLANNED_SOURCE_REVISION, task["scope"])
                self.assertIn(PLANNED_SOURCE_TREE, task["scope"])
                self.assertIn(PLANNED_MANIFEST_SHA256, task["scope"])
                self.assertIn(str(PLANNED_CI_RUN_ID), task["scope"])
                self.assertIn(ACTIVE_ID, task["scope"])
            else:
                self.assertIn("F8-70", task["dependencies"])
                self.assertIn(ACTIVE_ID, task["scope"])
                self.assertIn(ACTIVE_SOURCE_REVISION, task["scope"])
            self.assertIn("cannot be unioned", task["scope"])


if __name__ == "__main__":
    unittest.main()
