# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[1]
REPOSITORY = ROOT.parents[1]
MODULE_PATH = ROOT / "qualification.py"
SPEC = importlib.util.spec_from_file_location("conformance_qualification", MODULE_PATH)
assert SPEC and SPEC.loader
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)
PROFILE_PATH = ROOT / "qualification" / "capability-obligations-v1.json"
sys.path.insert(0, str(REPOSITORY / "tools" / "registry"))
from api_registry import (  # noqa: E402
    INTERACTION_DIRECTIONS,
    INTERACTION_FAMILIES,
    INTERACTION_MODES,
    INTERACTION_PROFILES,
    INTERACTION_SCOPES,
    interaction_capabilities,
)


SOURCE_COMMIT = "a" * 40


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def contract_map(profile: dict) -> dict[str, dict]:
    return {contract["id"]: contract for contract in profile["contracts"]}


def evidence(profile: dict, scope: str, contract_id: str,
             status: str = "passed") -> dict:
    contract = contract_map(profile)[contract_id]
    return {
        "scope": scope,
        "contract": contract_id,
        "status": status,
        "source_commit": SOURCE_COMMIT,
        "contract_sha256": contract["sha256"],
        "artifact_sha256": digest(scope + ":" + contract_id + ":" + status),
    }


def capability(first: str, second: str, scope: str, mode: str,
               direction: str, profile: str,
               limit: str | int = "simulation-capacity") -> dict:
    return {
        "first_family": first,
        "second_family": second,
        "scope": scope,
        "mode": mode,
        "direction": direction,
        "material_profile": profile,
        "max_graph_edges_per_simulation": limit,
    }


def valid_equal_fluid_result(profile: dict) -> dict:
    quality_contract = (
        "quality:core-conservation-stability-v1#equal-material-pressure-contact")
    capacity_contract = (
        "lifecycle:adverse-capacity-three-family-v1#capacity-replacement-recovery")
    graph_contract = (
        "lifecycle:adverse-capacity-three-family-v1#three-family-partial-rollback")
    return {
        "schema": qualification.RESULT_SCHEMA,
        "obligations_id": profile["id"],
        "obligations_sha256": qualification.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "source": {"commit": SOURCE_COMMIT, "dirty": False},
        "driver": {
            "name": "test-driver",
            "version": "1.2.3",
            "binary_sha256": digest("driver"),
        },
        "device": {
            "name": "test-device",
            "vendor_id": 1,
            "device_id": 2,
            "driver_version": "4.5.6",
        },
        "toolchain": {"compiler": "test-compiler", "version": "7.8.9"},
        "advertised_capabilities": [
            capability("FLUID", "FLUID", "SEPARATE_SYSTEM", "CONTACT",
                       "BIDIRECTIONAL", "EQUAL")
        ],
        "evidence": [
            evidence(profile, "compilation", "registry:interaction-capabilities"),
            evidence(profile, "execution", quality_contract),
            evidence(profile, "lifecycle_invariants", capacity_contract),
            evidence(profile, "lifecycle_invariants", graph_contract),
            evidence(profile, "analytic_invariants", quality_contract),
        ],
    }


def rigid_result(profile: dict) -> dict:
    result = valid_equal_fluid_result(profile)
    result["advertised_capabilities"] = [
        capability("RIGID_BODY", "RIGID_BODY", "SAME_DOMAIN", "CONTACT",
                   "BIDIRECTIONAL", "ENDPOINT_NATIVE", "not-applicable")
    ]
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:rigid-direct"),
        evidence(profile, "reference_regression", "case:rigid-direct"),
        evidence(profile, "lifecycle_invariants", "case:rigid-direct"),
        evidence(profile, "analytic_invariants",
                 "quality:rigid-brick-penetration-v1"),
    ]
    return result


def rigid_attachment_result(profile: dict) -> dict:
    result = valid_equal_fluid_result(profile)
    result["advertised_capabilities"] = [
        capability("RIGID_BODY", "RIGID_BODY", "SAME_DOMAIN", "ATTACHMENT",
                   "BIDIRECTIONAL", "ENDPOINT_NATIVE", "not-applicable")
    ]
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:compound-weld-lifecycle"),
        evidence(profile, "reference_regression",
                 "case:compound-weld-lifecycle"),
        evidence(profile, "lifecycle_invariants",
                 "case:compound-weld-lifecycle"),
        evidence(profile, "analytic_invariants", "quality:rigid-attachment-v1"),
    ]
    return result


def smoke_viscosity_result(profile: dict) -> dict:
    result = valid_equal_fluid_result(profile)
    result["advertised_capabilities"] = [
        capability("SMOKE", "SMOKE", "SELF", "VISCOSITY",
                   "BIDIRECTIONAL", material, "not-applicable")
        for material in ("SMOKE_PARTICLE", "SMOKE_GRID")
    ]
    lifecycle_contract = (
        "lifecycle:smoke-self-lifecycle-v1#"
        "smoke-capacity-replacement-removal-rollback")
    quality_contract = "quality:smoke-self-viscosity-v1"
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:smoke-grid"),
        evidence(profile, "reference_regression", "case:smoke-grid"),
        evidence(profile, "lifecycle_invariants", lifecycle_contract),
        evidence(profile, "analytic_invariants", quality_contract),
    ]
    return result


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = qualification.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return qualification.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> dict:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)
        return report

    def test_every_registry_capability_matches_exactly_one_used_rule(self):
        rule_ids = [rule["id"] for rule in self.profile["rules"]]
        self.assertEqual(len(rule_ids), len(set(rule_ids)))

        used: set[str] = set()
        canonical: set[tuple[str, ...]] = set()
        for record in interaction_capabilities():
            key = {
                "first_family": record["first"],
                "second_family": record["second"],
                "scope": record["scope"],
                "mode": record["mode"],
                "direction": record["direction"],
                "material_profile": record["profile"],
            }
            matches = qualification.matching_rules(self.profile, key)
            self.assertEqual(len(matches), 1, (key, matches))
            used.add(matches[0]["id"])
            canonical.add(tuple(key[field] for field in qualification.KEY_FIELDS))
        self.assertEqual(len(interaction_capabilities()), 49)
        self.assertEqual(used, {rule["id"] for rule in self.profile["rules"]})

        matched: set[tuple[str, ...]] = set()
        for values in itertools.product(
                INTERACTION_FAMILIES, INTERACTION_FAMILIES,
                INTERACTION_SCOPES, INTERACTION_MODES,
                INTERACTION_DIRECTIONS, INTERACTION_PROFILES):
            key = dict(zip(qualification.KEY_FIELDS, values))
            matches = qualification.matching_rules(self.profile, key)
            self.assertLessEqual(len(matches), 1, (key, matches))
            if matches:
                matched.add(values)
        self.assertEqual(matched, canonical)

    def test_all_contract_hashes_match_repository_inputs(self):
        for contract in self.profile["contracts"]:
            path = REPOSITORY / contract["path"]
            with self.subTest(contract=contract["id"]):
                self.assertTrue(path.is_file())
                self.assertEqual(qualification.profile_sha256(path),
                                 contract["sha256"])

    def test_contract_hash_is_checkout_line_ending_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lf = root / "lf.xml"
            crlf = root / "crlf.xml"
            lf.write_bytes(b"<registry>\n</registry>\n")
            crlf.write_bytes(b"<registry>\r\n</registry>\r\n")
            self.assertEqual(qualification.profile_sha256(lf),
                             qualification.profile_sha256(crlf))

    def test_fully_evidenced_equal_fluid_mode_passes(self):
        report = self.evaluate(valid_equal_fluid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["qualified_count"], 1)
        scopes = report["capabilities"][0]["scopes"]
        self.assertEqual(scopes["compilation"]["status"], "passed")
        self.assertEqual(scopes["execution"]["status"], "passed")
        self.assertEqual(scopes["reference_regression"]["status"], "not_required")
        self.assertEqual(scopes["lifecycle_invariants"]["status"], "passed")
        self.assertEqual(scopes["analytic_invariants"]["status"], "passed")
        self.assertEqual(scopes["timing"]["status"], "optional")

    def test_missing_and_unsupported_evidence_do_not_pass(self):
        result = valid_equal_fluid_result(self.profile)
        result["evidence"].pop()
        self.assert_rejected(result, "required evidence is missing")

        result = valid_equal_fluid_result(self.profile)
        result["evidence"][-1]["status"] = "unsupported"
        self.assert_rejected(result, "required evidence did not pass")

    def test_scope_substitution_cannot_satisfy_physical_quality(self):
        result = valid_equal_fluid_result(self.profile)
        physical = result["evidence"].pop()
        physical["scope"] = "timing"
        result["evidence"].append(physical)
        report = self.assert_rejected(result, "required evidence is missing")
        self.assertEqual(report["evidence_summary"]["timing"]["passed"], 1)
        self.assertEqual(report["qualified_count"], 0)

    def test_rigid_capabilities_have_independent_complete_obligations(self):
        for result, rule_id in (
                (rigid_result(self.profile), "rigid-contact"),
                (rigid_attachment_result(self.profile), "rigid-attachment")):
            with self.subTest(rule_id=rule_id):
                report = self.evaluate(result)
                self.assertTrue(report["passed"], report)
                capability_report = report["capabilities"][0]
                self.assertEqual(capability_report["rule_id"], rule_id)
                for scope in ("execution", "reference_regression",
                              "lifecycle_invariants", "analytic_invariants"):
                    self.assertEqual(
                        capability_report["scopes"][scope]["status"], "passed")

    def test_smoke_viscosity_profiles_share_complete_exact_obligations(self):
        report = self.evaluate(smoke_viscosity_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["qualified_count"], 2)
        self.assertEqual(
            {item["key"]["material_profile"] for item in report["capabilities"]},
            {"SMOKE_PARTICLE", "SMOKE_GRID"})
        for capability_report in report["capabilities"]:
            self.assertEqual(capability_report["rule_id"],
                             "smoke-self-viscosity")
            for scope in ("execution", "reference_regression",
                          "lifecycle_invariants", "analytic_invariants"):
                self.assertEqual(
                    capability_report["scopes"][scope]["status"], "passed")

    def test_stale_contract_source_and_artifact_metadata_fail(self):
        result = valid_equal_fluid_result(self.profile)
        result["evidence"][0]["contract_sha256"] = "0" * 64
        report = self.assert_rejected(result, "contract hash mismatch")
        self.assertEqual(report["qualified_count"], 0)

        result = valid_equal_fluid_result(self.profile)
        result["evidence"][0]["source_commit"] = "b" * 40
        self.assert_rejected(result, "source revision mismatch")

        result = valid_equal_fluid_result(self.profile)
        result["evidence"][0]["artifact_sha256"] = "bad"
        self.assert_rejected(result, "artifact SHA-256 is required")

    def test_incomplete_evaluation_and_dirty_source_fail(self):
        result = valid_equal_fluid_result(self.profile)
        result["execution_status"] = "unsupported"
        self.assert_rejected(result, "evaluation must be completed")

        result = valid_equal_fluid_result(self.profile)
        result["source"]["dirty"] = True
        report = self.assert_rejected(result, "source must be clean")
        self.assertEqual(report["qualified_count"], 0)

    def test_unknown_or_duplicate_capability_fails(self):
        result = valid_equal_fluid_result(self.profile)
        result["advertised_capabilities"][0]["direction"] = "EXTERNAL_TO_FIRST"
        self.assert_rejected(result, "match exactly one obligation rule")

        result = valid_equal_fluid_result(self.profile)
        result["advertised_capabilities"].append(
            copy.deepcopy(result["advertised_capabilities"][0]))
        self.assert_rejected(result, "duplicate advertised capability")

    def test_duplicate_evidence_and_profile_hash_fail(self):
        result = valid_equal_fluid_result(self.profile)
        result["evidence"].append(copy.deepcopy(result["evidence"][0]))
        self.assert_rejected(result, "duplicate scoped evidence")

        result = valid_equal_fluid_result(self.profile)
        result["obligations_sha256"] = "0" * 64
        self.assert_rejected(result, "obligations hash mismatch")


if __name__ == "__main__":
    unittest.main()
