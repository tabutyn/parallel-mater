# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[1]
REPOSITORY = ROOT.parents[1]
MODULE_PATH = ROOT / "evidence_index.py"
SPEC = importlib.util.spec_from_file_location("conformance_evidence_index", MODULE_PATH)
assert SPEC and SPEC.loader
evidence_index = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evidence_index
SPEC.loader.exec_module(evidence_index)
PROFILE_PATH = ROOT / "qualification" / "capability-obligations-v1.json"
SOURCES_PATH = ROOT / "evidence" / "evidence-sources-v1.json"
INDEX_PATH = ROOT / "evidence" / "index-v1.json"
SOURCE_COMMIT = "a" * 40


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_result(profile: dict) -> dict:
    contracts = {item["id"]: item for item in profile["contracts"]}
    ids = {
        "compile": "registry:interaction-capabilities",
        "quality": "quality:core-conservation-stability-v1#equal-material-pressure-contact",
        "capacity": "lifecycle:adverse-capacity-three-family-v1#capacity-replacement-recovery",
        "graph": "lifecycle:adverse-capacity-three-family-v1#three-family-partial-rollback",
    }

    def evidence(scope: str, identity: str) -> dict:
        return {
            "scope": scope,
            "contract": identity,
            "status": "passed",
            "source_commit": SOURCE_COMMIT,
            "contract_sha256": contracts[identity]["sha256"],
            "artifact_sha256": digest(scope + identity),
        }

    return {
        "schema": "parallel-mater-conformance-qualification-result/v1",
        "obligations_id": profile["id"],
        "obligations_sha256": hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
        "execution_status": "completed",
        "source": {"commit": SOURCE_COMMIT, "dirty": False},
        "driver": {"name": "test-driver", "version": "1.0",
                   "binary_sha256": digest("driver")},
        "device": {"name": "test-device", "vendor_id": 1, "device_id": 2,
                   "driver_version": "3.0"},
        "toolchain": {"compiler": "test-compiler", "version": "4.0"},
        "advertised_capabilities": [{
            "first_family": "FLUID", "second_family": "FLUID",
            "scope": "SEPARATE_SYSTEM", "mode": "CONTACT",
            "direction": "BIDIRECTIONAL", "material_profile": "EQUAL",
            "max_graph_edges_per_simulation": "simulation-capacity",
        }],
        "evidence": [
            evidence("compilation", ids["compile"]),
            evidence("execution", ids["quality"]),
            evidence("lifecycle_invariants", ids["capacity"]),
            evidence("lifecycle_invariants", ids["graph"]),
            evidence("analytic_invariants", ids["quality"]),
        ],
    }


class EvidenceIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()

    def build(self, result: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            temporary = pathlib.Path(directory)
            result_path = temporary / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            sources = {
                "schema": evidence_index.SOURCES_SCHEMA,
                "id": "test-sources",
                "obligations": {"path": str(PROFILE_PATH),
                                "sha256": self.profile_hash},
                "expected_drivers": [{"id": "test-driver", "label": "Test"}],
                "qualification_results": [{
                    "id": "test-result", "path": str(result_path),
                    "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                }],
                "supplemental_reports": [],
            }
            sources_path = temporary / "sources.json"
            sources_path.write_text(json.dumps(sources), encoding="utf-8")
            return evidence_index.build_index(REPOSITORY, sources_path)

    def test_passing_result_preserves_provenance_and_qualifies(self):
        index = self.build(valid_result(self.profile))
        self.assertTrue(index["valid"])
        self.assertEqual(index["qualified_capability_count"], 1)
        self.assertTrue(index["capabilities"][0]["qualified"])
        report = index["reports"][0]
        self.assertEqual(report["source"]["commit"], SOURCE_COMMIT)
        self.assertEqual(report["driver"]["binary_sha256"], digest("driver"))
        self.assertEqual(report["device"]["name"], "test-device")
        self.assertEqual(report["toolchain"]["compiler"], "test-compiler")

    def test_stale_missing_and_unsupported_evidence_never_qualify(self):
        mutations = []
        stale = valid_result(self.profile)
        stale["evidence"][0]["contract_sha256"] = "0" * 64
        mutations.append((stale, "contract hash mismatch"))
        stale_source = valid_result(self.profile)
        stale_source["evidence"][0]["source_commit"] = "b" * 40
        mutations.append((stale_source, "source revision mismatch"))
        missing = valid_result(self.profile)
        missing["evidence"].pop()
        mutations.append((missing, "required evidence is missing"))
        unsupported = valid_result(self.profile)
        unsupported["evidence"][-1]["status"] = "unsupported"
        mutations.append((unsupported, "required evidence did not pass"))
        for result, message in mutations:
            with self.subTest(message=message):
                index = self.build(result)
                self.assertEqual(index["qualified_capability_count"], 0)
                self.assertFalse(index["capabilities"][0]["qualified"])
                self.assertTrue(any(message in failure["message"]
                                    for failure in index["known_failures"]))

    def test_checked_in_index_and_document_claims_are_current(self):
        index = evidence_index.build_index(REPOSITORY, SOURCES_PATH)
        self.assertEqual(index, json.loads(INDEX_PATH.read_text(encoding="utf-8")))
        sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
        document = sources["documentation"]
        text = (REPOSITORY / document["path"]).read_text(encoding="utf-8")
        begin = text.index(document["begin_marker"])
        end = text.index(document["end_marker"]) + len(document["end_marker"])
        self.assertEqual(text[begin:end], evidence_index.render_document_block(index))

    def test_legacy_report_retains_tuple_failures_and_no_certification(self):
        index = evidence_index.build_index(REPOSITORY, SOURCES_PATH)
        self.assertEqual(index["qualified_capability_count"], 0)
        cuda = next(report for report in index["reports"]
                    if report["driver"]["name"] == "slang-cuda")
        self.assertEqual(cuda["source"]["commit"],
                         "2a6819f14c0cd48dced13d5c1c80d6695326d6bf")
        self.assertEqual(cuda["device"]["name"], "NVIDIA GeForce GTX 1080")
        self.assertEqual(cuda["toolchain"]["slang"], "2026.18")
        kinds = {failure["kind"] for failure in index["known_failures"]}
        self.assertIn("historical_baseline_failure", kinds)
        self.assertIn("unsupported", kinds)
        self.assertIn("missing_qualification_result", kinds)

    def test_stale_report_registry_hash_is_a_validation_failure(self):
        sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
        sources["supplemental_reports"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sources.json"
            path.write_text(json.dumps(sources), encoding="utf-8")
            index = evidence_index.build_index(REPOSITORY, path)
        self.assertFalse(index["valid"])
        self.assertEqual(index["report_count"], 0)
        self.assertTrue(any(failure["message"] == "source SHA-256 mismatch"
                            for failure in index["validation_failures"]))


if __name__ == "__main__":
    unittest.main()
