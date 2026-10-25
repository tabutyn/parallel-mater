import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from check_realtime_candidate import (  # noqa: E402
    ACTIVE_ID,
    CANDIDATE_PATH,
    REQUIRED_CLOSURES,
    validate_candidate,
)


class F836CandidateFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    def test_repaired_candidate_is_frozen_without_drift(self):
        self.assertEqual(validate_candidate(), [])

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

    def test_only_replacement_can_receive_qualification_evidence(self):
        routing = self.candidate["qualification_routing"]
        self.assertEqual(routing["accepted_candidate"], ACTIVE_ID)
        self.assertFalse(routing["evidence_union_allowed"])
        for task_id in routing["tasks"]:
            task = json.loads(
                (ROOT / "docs/roadmap/tasks" / f"{task_id}.json")
                .read_text(encoding="utf-8")
            )
            self.assertIn(ACTIVE_ID, task["scope"])
            self.assertIn("F8-36", task["dependencies"])
            self.assertIn("F8-38", task["dependencies"])
            self.assertIn("rejected for qualification", task["scope"])


if __name__ == "__main__":
    unittest.main()
