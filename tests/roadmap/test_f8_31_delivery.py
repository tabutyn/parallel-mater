import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from check_f8_31_delivery import DELIVERY_PATH, validate_delivery  # noqa: E402


class F831DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delivery = json.loads(DELIVERY_PATH.read_text(encoding="utf-8"))

    def test_delivery_and_retained_artifacts_are_exact(self):
        self.assertEqual(validate_delivery(), [])

    def test_candidate_drift_is_rejected(self):
        tampered = copy.deepcopy(self.delivery)
        tampered["candidate"]["id"] = "realtime-single-step-v1-tampered"
        tampered["candidate"]["source_revision"] = "0" * 40
        errors = validate_delivery(tampered, verify_files=False)
        self.assertIn("candidate ID drift", errors)
        self.assertIn("candidate source revision drift", errors)

    def test_failed_rows_cannot_be_promoted(self):
        tampered = copy.deepcopy(self.delivery)
        rows = {row["target"]: row for row in tampered["device_rows"]}
        rows["android-pixel7-slang-vulkan"]["frozen_correctness"] = "passed"
        rows["windows-gtx1080-slang-d3d12"]["frozen_correctness"] = "passed"
        errors = validate_delivery(tampered, verify_files=False)
        self.assertIn("android-pixel7-slang-vulkan verdict drift", errors)
        self.assertIn("windows-gtx1080-slang-d3d12 verdict drift", errors)

    def test_timing_targets_and_dispatch_claim_cannot_be_weakened(self):
        tampered = copy.deepcopy(self.delivery)
        tampered["timing_policy"]["desktop_physics_p95_ms_maximum"] = 25.0
        tampered["dispatch_reduction"]["metal_rigid_public_fixture"][
            "final_total_dispatches"
        ] = 112
        errors = validate_delivery(tampered, verify_files=False)
        self.assertIn("timing targets were weakened", errors)
        self.assertIn("Metal rigid total dispatch reduction drift", errors)

    def test_broad_release_claims_remain_false(self):
        tampered = copy.deepcopy(self.delivery)
        tampered["release"]["broad_conformance_qualified"] = True
        tampered["release"]["broad_60hz_performance_qualified"] = True
        tampered["release"]["scene_quality_qualified"] = True
        errors = validate_delivery(tampered, verify_files=False)
        self.assertIn("broad conformance failure was promoted", errors)
        self.assertIn("broad 60 Hz failure was promoted", errors)
        self.assertIn("scene-quality debt was promoted", errors)


if __name__ == "__main__":
    unittest.main()
