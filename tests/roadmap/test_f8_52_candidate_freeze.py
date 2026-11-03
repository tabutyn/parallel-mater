import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from check_realtime_candidate import (  # noqa: E402
    ACTIVE_CI_RUN_ID,
    ACTIVE_ID,
    ACTIVE_SOURCE_REVISION,
    ACTIVE_SOURCE_TREE,
    CANDIDATE_PATH,
    REQUIRED_CLOSURES,
    V9_ID,
    V9_SHA256,
    validate_candidate,
)

class F874CandidateFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    def test_candidate_and_supersession_are_exact(self):
        self.assertEqual(validate_candidate(), [])
        self.assertEqual(self.candidate["task"], "F8-74")
        self.assertEqual(self.candidate["id"], ACTIVE_ID)
        self.assertEqual(self.candidate["source"]["revision"], ACTIVE_SOURCE_REVISION)
        self.assertEqual(self.candidate["source"]["tree"], ACTIVE_SOURCE_TREE)
        self.assertEqual(self.candidate["supersedes"]["candidate_id"], V9_ID)
        self.assertEqual(self.candidate["supersedes"]["manifest_sha256"], V9_SHA256)

    def test_every_protected_closure_drift_is_rejected(self):
        closures = {
            item["category"]: item
            for item in self.candidate["source"]["protected_closures"]
        }
        self.assertEqual(set(closures), REQUIRED_CLOSURES)
        for category in REQUIRED_CLOSURES:
            with self.subTest(category=category):
                tampered = copy.deepcopy(self.candidate)
                next(
                    item for item in tampered["source"]["protected_closures"]
                    if item["category"] == category
                )["sha256"] = "0" * 64
                self.assertIn(
                    f"{category} closure hash drift",
                    validate_candidate(tampered),
                )

    def test_source_and_contract_drift_are_rejected(self):
        tampered = copy.deepcopy(self.candidate)
        tampered["source"]["revision"] = "0" * 40
        tampered["source"]["tree"] = "0" * 40
        tampered["contract_inputs"]["scenario_fixtures"]["sha256"] = "0" * 64
        tampered["contract_inputs"]["capture_validator"]["sha256"] = "0" * 64
        errors = validate_candidate(tampered)
        self.assertIn("active source revision drift", errors)
        self.assertIn("active source tree drift", errors)
        self.assertIn("scenario fixtures contract hash drift", errors)
        self.assertIn("capture validator contract hash drift", errors)

    def test_protected_ci_drift_is_rejected(self):
        tampered = copy.deepcopy(self.candidate)
        actions = tampered["compile_evidence"]["github_actions"]
        actions["run_id"] = ACTIVE_CI_RUN_ID + 1
        actions["artifacts"].pop()
        actions["jobs"][0]["conclusion"] = "failure"
        errors = validate_candidate(tampered)
        self.assertIn("protected exact-source CI identity drift", errors)
        self.assertIn("protected CI artifact set drift", errors)
        self.assertIn("protected CI job set is incomplete or failed", errors)

    def test_timing_debt_cannot_be_promoted_to_a_pass(self):
        tampered = copy.deepcopy(self.candidate)
        tampered["timing_debt"]["final_performance_qualified"] = True
        tampered["timing_debt"]["windows_six_family_physics_p95_ms"]["plain"] = 9.0
        tampered["verdicts"]["final_performance"]["status"] = "passed"
        errors = validate_candidate(tampered)
        self.assertIn(
            "retained Windows timing debt was weakened or misclassified",
            errors,
        )
        self.assertIn(
            "integration and final-performance verdicts are not separated",
            errors,
        )

    def test_routing_cannot_union_candidate_v9_rows(self):
        tampered = copy.deepcopy(self.candidate)
        tampered["qualification_routing"]["evidence_union_allowed"] = True
        tampered["qualification_routing"]["tasks"].remove("F8-85")
        self.assertIn("qualification routing drift", validate_candidate(tampered))

    def test_ipad_ci_does_not_dirty_the_source_checkout(self):
        workflow = (ROOT / ".github/workflows/mobile-gallery.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('$RUNNER_TEMP/xcode-version.txt', workflow)
        self.assertNotIn("tee xcode-version.txt", workflow)

if __name__ == "__main__":
    unittest.main()
