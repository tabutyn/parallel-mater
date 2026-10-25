import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "docs/roadmap/evidence/F8-23-linux-cpu-qualification.json"
RAW_PATH = ROOT / "docs/roadmap/evidence/F8-23-CPU-QUALIFICATION.json"
CANDIDATE_PATH = ROOT / "conformance/v1/qualification/realtime-candidate-v3.json"
PROFILE_PATH = ROOT / "conformance/v1/qualification/realtime-profile-v1.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class F823CpuQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load(SUMMARY_PATH)
        cls.raw = load(RAW_PATH)
        cls.candidate = load(CANDIDATE_PATH)
        cls.profile = load(PROFILE_PATH)

    def test_exact_candidate_and_binary_are_bound(self):
        expected_hash = sha256(CANDIDATE_PATH)
        self.assertEqual(self.summary["candidate"]["id"], self.candidate["id"])
        self.assertEqual(
            self.summary["candidate"]["source_revision"],
            self.candidate["source"]["revision"],
        )
        self.assertEqual(self.summary["candidate"]["manifest_sha256"], expected_hash)
        self.assertEqual(self.raw["candidate"]["manifest_sha256"], expected_hash)
        self.assertEqual(
            self.raw["binary"]["driver_sha256"],
            self.summary["artifacts"]["production_driver_sha256"],
        )

    def test_every_protected_closure_is_bound(self):
        closures = {
            item["category"]: item["sha256"]
            for item in self.candidate["source"]["protected_closures"]
        }
        self.assertEqual(self.summary["candidate"]["protected_closures"], closures)

    def test_exact_cpu_workload_and_all_samples_are_retained(self):
        cpu_tier = next(
            tier for tier in self.profile["performance"]["tiers"]
            if tier["id"] == "cpu-reference-v1"
        )
        expected = cpu_tier["workloads"][0]
        workload = self.raw["workload"]
        self.assertEqual(workload["tier"], cpu_tier["id"])
        self.assertEqual(workload["id"], expected["id"])
        self.assertEqual(
            {key: workload[key] for key in expected["counts"]},
            expected["counts"],
        )
        self.assertEqual(len(self.raw["cold_start"]["host_physics_samples_ms"]), 120)
        self.assertEqual(
            len(self.raw["sustained"]["host_physics_samples_ms"]), 7200
        )
        self.assertEqual(self.raw["result"]["ticks"], 7800)
        self.assertEqual(self.raw["sustained"]["failed_ticks"], 0)

    def test_timing_and_fixed_work_pass_frozen_budgets(self):
        cpu_tier = next(
            tier for tier in self.profile["performance"]["tiers"]
            if tier["id"] == "cpu-reference-v1"
        )
        measurement = self.profile["performance"]["measurement"]
        self.assertLessEqual(
            self.raw["sustained"]["host_physics_summary"]["p95_ms"],
            cpu_tier["physics_p95_ms_maximum"],
        )
        self.assertGreaterEqual(
            self.raw["sustained"]["simulated_to_wall_time_ratio"],
            measurement["minimum_simulated_to_wall_time_ratio"],
        )
        self.assertEqual(self.raw["result"]["dispatches_per_tick"], 20)
        self.assertEqual(self.raw["result"]["workgroups_per_tick"], 1421)
        self.assertEqual(self.raw["result"]["rigid_iterations"], 8)
        self.assertEqual(self.raw["result"]["spatial_builds"], 1)
        self.assertEqual(self.raw["result"]["spatial_publications"], 1)
        self.assertEqual(self.raw["result"]["time_advances"], 1)

    def test_summary_hash_and_evidence_boundaries_are_exact(self):
        self.assertEqual(
            self.summary["artifacts"]["raw_capture_sha256"], sha256(RAW_PATH)
        )
        self.assertEqual(
            len(self.summary["coverage"]["unsupported_production_cpu_fixtures"]),
            15,
        )
        self.assertFalse(self.summary["legacy_separation"]["reused_as_realtime_evidence"])
        self.assertEqual(self.summary["status"], "passed")


if __name__ == "__main__":
    unittest.main()
