import hashlib
import json
import re
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "realtime-candidate-v1.json"
)
INVENTORY_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "kernel-device-obligations-v1.json"
)
PROFILE_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "realtime-profile-v1.json"
)
ORACLE_PATH = ROOT / "conformance" / "v1" / "realtime" / "oracle-suite-v1.json"
MANIFEST_PATH = ROOT / "cmake" / "SlangKernelManifest.cmake"
TASK_DIR = ROOT / "docs" / "roadmap" / "tasks"
REJECTION_PATH = (
    ROOT / "conformance" / "v1" / "qualification"
    / "realtime-candidate-v1-rejection.json"
)
OLD_MANIFEST_SHA256 = (
    "c25c010be3a0e908537e3c6542c1759b28b10cfb4a4169b51bb2834233411d43"
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmake_set(text, name):
    match = re.search(rf"set\({re.escape(name)}\s+(.*?)\)", text, re.DOTALL)
    if not match:
        raise AssertionError(f"missing CMake set {name}")
    return re.findall(r"[A-Za-z][A-Za-z0-9_]*", match.group(1))


def canonical_kernels():
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    modules = cmake_set(text, "PARALLEL_MATER_SLANG_KERNEL_MODULES")
    return {
        module: cmake_set(text, f"PARALLEL_MATER_SLANG_{module}_ENTRIES")
        for module in modules
    }


class F822CandidateFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = load_json(CANDIDATE_PATH)
        cls.inventory = load_json(INVENTORY_PATH)
        cls.profile = load_json(PROFILE_PATH)
        cls.oracle = load_json(ORACLE_PATH)

    def test_identity_profile_and_protected_hashes_are_frozen(self):
        self.assertEqual(
            self.candidate["schema"], "parallel-mater-realtime-candidate/v1"
        )
        self.assertEqual(self.candidate["task"], "F8-22")
        self.assertEqual(
            self.candidate["status"], "frozen-awaiting-physical-qualification"
        )
        self.assertRegex(self.candidate["source"]["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(self.candidate["source"]["tree"], r"^[0-9a-f]{40}$")

        frozen_profile = self.candidate["profile"]
        self.assertEqual(frozen_profile["id"], self.profile["id"])
        self.assertEqual(frozen_profile["semantic_name"], self.profile["profile"]["semantic_name"])
        self.assertEqual(frozen_profile["version"], self.profile["profile"]["version"])
        self.assertEqual(frozen_profile["timestep"], "1/60")

        # The rejected manifest is immutable history. Its old protected files
        # are deliberately not compared with the replacement source tree.
        self.assertEqual(sha256(CANDIDATE_PATH), OLD_MANIFEST_SHA256)
        rejection = load_json(REJECTION_PATH)
        self.assertEqual(rejection["candidate_id"], self.candidate["id"])
        self.assertEqual(rejection["candidate_manifest_sha256"],
                         OLD_MANIFEST_SHA256)
        self.assertEqual(rejection["status"], "rejected-non-executable")
        self.assertFalse(rejection["evidence_union_allowed"])

        # The rejected candidate's recorded closure is immutable provenance,
        # not a constraint on the active replacement source tree.
        closure = self.candidate["source"]["slang_source_closure"]
        self.assertEqual(closure["file_count"], 38)
        self.assertRegex(closure["sha256"], r"^[0-9a-f]{64}$")

    def test_every_canonical_entry_and_group_has_runtime_coverage(self):
        canonical = canonical_kernels()
        actual = defaultdict(list)
        realtime_cases = {
            case["id"] for case in self.oracle["public_trajectory_cases"]
        }
        for group in self.inventory["kernel_groups"]:
            self.assertTrue(group["existing_execution_tests"], group["id"])
            self.assertTrue(
                group["public_case_sets"] or group.get("f8_public_cases"),
                group["id"],
            )
            self.assertNotIn("coverage_gap", group)
            self.assertLessEqual(set(group.get("f8_public_cases", [])), realtime_cases)
            actual[group["module"]].extend(group["entries"])

        self.assertEqual(set(actual), set(canonical))
        for module, entries in canonical.items():
            self.assertEqual(len(actual[module]), len(set(actual[module])), module)
            self.assertEqual(set(actual[module]), set(entries), module)

        coverage = self.candidate["kernel_coverage"]
        self.assertEqual(coverage["canonical_module_count"], len(canonical))
        self.assertEqual(coverage["canonical_entry_count"], sum(map(len, canonical.values())))
        self.assertEqual(coverage["kernel_group_count"], len(self.inventory["kernel_groups"]))
        self.assertEqual(coverage["coverage_gaps"], [])
        self.assertIn("f8-paint-contact-stamping", realtime_cases)

    def test_runtime_variants_have_public_execution_cases_and_targets(self):
        inventory_variants = {
            item["id"]: item for item in self.inventory["compiled_variants"]
        }
        public_execution = set(self.oracle["public_api"]["execution_cases"])
        frozen_runtime = {
            item["id"]: item for item in self.candidate["runtime_variants"]
        }
        expected_runtime = {
            name for name, item in inventory_variants.items()
            if "compile-only" not in item["kind"]
        }
        self.assertEqual(set(frozen_runtime), expected_runtime)
        for variant_id, frozen in frozen_runtime.items():
            inventory = inventory_variants[variant_id]
            self.assertEqual(frozen["f8_execution_test"], inventory["f8_execution_test"])
            self.assertIn(frozen["f8_execution_test"], public_execution)
            self.assertEqual(set(frozen["applicable_targets"]), set(inventory["applicable_targets"]))
            self.assertEqual(set(frozen["qualification_owners"]), set(inventory["qualification_owners"]))
            self.assertEqual(frozen["status"], "awaiting-physical-qualification")

        frozen_compile_only = {
            item["id"]: item for item in self.candidate["compile_only_variants"]
        }
        expected_compile_only = set(inventory_variants) - expected_runtime
        self.assertEqual(set(frozen_compile_only), expected_compile_only)
        for variant_id, frozen in frozen_compile_only.items():
            self.assertFalse(frozen["qualification_allowed"])
            self.assertEqual(inventory_variants[variant_id]["applicable_targets"], [])
            self.assertTrue(inventory_variants[variant_id]["execution_not_applicable"])

    def test_physical_matrix_and_exit_dependencies_are_exact(self):
        inventory_targets = {
            item["id"]: item for item in self.inventory["physical_targets"]
        }
        frozen_targets = {
            item["id"]: item for item in self.candidate["physical_targets"]
        }
        self.assertEqual(set(frozen_targets), set(inventory_targets))
        self.assertEqual(len(frozen_targets), 8)
        exit_dependencies = set(load_json(TASK_DIR / "F8-99.json")["dependencies"])
        for target_id, frozen in frozen_targets.items():
            inventory = inventory_targets[target_id]
            self.assertEqual(frozen["variant"], inventory["compiled_variant"])
            self.assertEqual(frozen["qualification_owner"], inventory["qualification_owner"])
            self.assertEqual(frozen["status"], "pending")
            self.assertTrue(frozen["availability_at_freeze"])
            owner = frozen["qualification_owner"]
            owner_task = load_json(TASK_DIR / f"{owner}.json")
            self.assertIn("F8-22", owner_task["dependencies"])
            self.assertIn(owner, exit_dependencies)

        audit = self.candidate["device_task_audit"]
        self.assertEqual(audit["registered_backend_tuple_count"], len(frozen_targets))
        self.assertEqual(set(audit["qualification_tasks"]), {
            target["qualification_owner"] for target in frozen_targets.values()
        })
        self.assertEqual(audit["additional_tasks_added"], [])

    def test_compile_evidence_covers_backends_without_qualifying_them(self):
        evidence = self.candidate["compile_evidence"]
        self.assertFalse(evidence["qualification_claim"])
        self.assertIn("615/615", evidence["native_cuda_local"]["result"])
        self.assertEqual(evidence["native_cuda_local"]["runtime_execution"],
                         "not-run-driver-unavailable")
        self.assertEqual(evidence["github_actions"]["result"], "success")
        self.assertRegex(evidence["github_actions"]["head_revision"], r"^[0-9a-f]{40}$")
        artifact_text = " ".join(
            item["name"] + " " + item["toolchain"]
            for item in evidence["github_actions"]["artifacts"]
        ).lower()
        for backend in ("cuda", "vulkan", "d3d12", "metal"):
            self.assertIn(backend, artifact_text)
        for item in evidence["github_actions"]["artifacts"]:
            self.assertGreater(item["id"], 0)
            self.assertGreater(item["file_count"], 0)
            self.assertRegex(item["extracted_tree_sha256"], r"^[0-9a-f]{64}$")

    def test_invalidation_never_unions_candidate_evidence(self):
        invalidation = self.candidate["invalidation"]
        self.assertIn("never unioned", invalidation["policy"])
        trigger_text = " ".join(invalidation["triggers"])
        for required in ("kernel", "physics", "schedule", "profile", "oracle"):
            self.assertIn(required, trigger_text)
        self.assertIn("requalification task", invalidation["completed_row_rule"])
        self.assertIn("exact candidate", invalidation["exit_rule"])


if __name__ == "__main__":
    unittest.main()
