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


def cloth_smoke_contact_result(profile: dict, material: str,
                               mode: str = "CONTACT") -> dict:
    result = valid_equal_fluid_result(profile)
    direction = "FIRST_TO_SECOND" if mode == "CONTACT" else "SECOND_TO_FIRST"
    result["advertised_capabilities"] = [
        capability("CLOTH", "SMOKE", "DISTINCT_FAMILY", mode,
                   direction, material)
    ]
    lifecycle_contract = (
        "lifecycle:adverse-capacity-three-family-v1#"
        "three-family-partial-rollback"
        if material == "SMOKE_PARTICLE" else
        "lifecycle:cloth-smoke-contact-lifecycle-v1#"
        "grid-cloth-smoke-contact-capacity-replacement-removal-rollback")
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:smoke-cloth"),
        evidence(profile, "reference_regression", "case:smoke-cloth"),
        evidence(profile, "lifecycle_invariants", lifecycle_contract),
        evidence(profile, "analytic_invariants",
                 "quality:cloth-smoke-contact-v1"),
    ]
    return result


def soft_smoke_contact_result(profile: dict, material: str,
                              mode: str = "CONTACT") -> dict:
    result = valid_equal_fluid_result(profile)
    direction = "FIRST_TO_SECOND" if mode == "CONTACT" else "SECOND_TO_FIRST"
    result["advertised_capabilities"] = [
        capability("SOFT_BODY", "SMOKE", "DISTINCT_FAMILY", mode,
                   direction, material)
    ]
    if mode == "DRAG":
        prefix = material.removeprefix("SMOKE_").lower()
        lifecycle_contract = (
            "lifecycle:smoke-soft-drag-lifecycle-v1#" + prefix +
            "-smoke-soft-drag-capacity-replacement-removal-rollback")
        quality_contract = "quality:smoke-soft-drag-v1"
    else:
        lifecycle_contract = (
            "lifecycle:adverse-capacity-three-family-v1#"
            "three-family-partial-rollback"
            if material == "SMOKE_PARTICLE" else
            "lifecycle:soft-smoke-contact-lifecycle-v1#"
            "grid-soft-smoke-contact-capacity-replacement-removal-rollback")
        quality_contract = "quality:soft-smoke-contact-v1"
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:smoke-soft-body"),
        evidence(profile, "reference_regression", "case:smoke-soft-body"),
        evidence(profile, "lifecycle_invariants", lifecycle_contract),
        evidence(profile, "analytic_invariants", quality_contract),
    ]
    return result


def rope_smoke_contact_result(profile: dict, material: str,
                              mode: str = "CONTACT") -> dict:
    result = valid_equal_fluid_result(profile)
    direction = "FIRST_TO_SECOND" if mode == "CONTACT" else "SECOND_TO_FIRST"
    result["advertised_capabilities"] = [
        capability("ROPE", "SMOKE", "DISTINCT_FAMILY", mode,
                   direction, material)
    ]
    prefix = material.removeprefix("SMOKE_").lower()
    if mode == "DRAG":
        lifecycle_contract = (
            "lifecycle:smoke-rope-drag-lifecycle-v1#" + prefix +
            "-smoke-rope-drag-capacity-replacement-removal-rollback")
        quality_contract = "quality:smoke-rope-drag-v1"
    else:
        lifecycle_contract = (
            "lifecycle:rope-smoke-contact-lifecycle-v1#" + prefix +
            "-rope-smoke-contact-capacity-replacement-removal-rollback")
        quality_contract = "quality:rope-smoke-contact-v1"
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:smoke-rope"),
        evidence(profile, "reference_regression", "case:smoke-rope"),
        evidence(profile, "lifecycle_invariants", lifecycle_contract),
        evidence(profile, "analytic_invariants", quality_contract),
    ]
    return result


def soft_rope_attachment_result(profile: dict,
                                mode: str = "ATTACHMENT") -> dict:
    result = valid_equal_fluid_result(profile)
    result["advertised_capabilities"] = [
        capability("SOFT_BODY", "ROPE", "DISTINCT_FAMILY", mode,
                   "BIDIRECTIONAL", "ENDPOINT_NATIVE")
    ]
    lifecycle_contract = (
        "lifecycle:adverse-capacity-three-family-v1#"
        "three-family-partial-rollback")
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:rope-soft-body"),
        evidence(profile, "reference_regression", "case:rope-soft-body"),
        evidence(profile, "lifecycle_invariants", lifecycle_contract),
        evidence(profile, "analytic_invariants",
                 "quality:soft-rope-attachment-v1"),
    ]
    return result


def soft_rigid_contact_result(profile: dict,
                              first: str = "SOFT_BODY") -> dict:
    result = valid_equal_fluid_result(profile)
    result["advertised_capabilities"] = [
        capability(first, "RIGID_BODY", "DISTINCT_FAMILY", "CONTACT",
                   "BIDIRECTIONAL", "ENDPOINT_NATIVE", "not-applicable")
    ]
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:soft-body-core"),
        evidence(profile, "reference_regression", "case:soft-body-core"),
        evidence(profile, "lifecycle_invariants",
                 "lifecycle:soft-rigid-contact-lifecycle-v1#"
                 "soft-rigid-contact-capacity-replacement-removal-rollback"),
        evidence(profile, "analytic_invariants",
                 "quality:soft-rigid-contact-v1"),
    ]
    return result


def rope_rigid_contact_result(profile: dict, mode: str = "CONTACT") -> dict:
    result = valid_equal_fluid_result(profile)
    result["advertised_capabilities"] = [
        capability("ROPE", "RIGID_BODY", "DISTINCT_FAMILY", mode,
                   "BIDIRECTIONAL", "ENDPOINT_NATIVE", "not-applicable")
    ]
    result["evidence"] = [
        evidence(profile, "compilation", "registry:interaction-capabilities"),
        evidence(profile, "execution", "case:rope-core"),
        evidence(profile, "reference_regression", "case:rope-core"),
        evidence(profile, "lifecycle_invariants", "case:rope-core"),
        evidence(profile, "analytic_invariants",
                 "quality:rope-rigid-contact-v1"),
    ]
    return result


def fluid_soft_aggregate(profile: dict) -> tuple[dict, dict, list[dict]]:
    contracts = contract_map(profile)
    advertised = capability(
        "FLUID", "SOFT_BODY", "DISTINCT_FAMILY", "CONTACT",
        "BIDIRECTIONAL", "ENDPOINT_NATIVE")
    aggregate = {
        "schema": qualification.ROADMAP_EVIDENCE_SCHEMA,
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
            "driver_version": "456 (test)",
        },
        "toolchain": {"compiler": "test-compiler", "version": "7.8.9"},
        "advertised_capabilities": [advertised],
        "contracts": {
            "trajectory": {
                "id": "case:soft-body-fluid",
                "sha256": contracts["case:soft-body-fluid"]["sha256"],
            },
            "quality": {
                "id": "quality:fluid-soft-contact-v1",
                "sha256": contracts["quality:fluid-soft-contact-v1"]["sha256"],
            },
            "lifecycle": {
                "id": ("lifecycle:adverse-capacity-three-family-v1#"
                       "three-family-partial-rollback"),
                "sha256": contracts[
                    "lifecycle:adverse-capacity-three-family-v1#"
                    "three-family-partial-rollback"]["sha256"],
            },
        },
        "trajectory": {
            "execution_status": "completed",
            "reference_regression": {
                "passed": 1, "failed": 0, "unsupported": 0,
            },
        },
        "analytic_invariants": {
            "passed": True, "sample_count": 1, "failure_count": 0,
        },
        "lifecycle_invariants": {
            "passed": True, "profile_case_count": 2,
            "profile_checkpoint_count": 9, "failure_count": 0,
        },
    }
    provenance = {
        "source_commit": SOURCE_COMMIT,
        "device_name": "test-device",
        "vendor_id": 1,
        "device_id": 2,
        "driver_version": 456,
    }
    return aggregate, provenance, [advertised]


def rope_rigid_contact_aggregate(
        profile: dict) -> tuple[dict, dict, list[dict]]:
    contracts = contract_map(profile)
    advertised = capability(
        "ROPE", "RIGID_BODY", "DISTINCT_FAMILY", "CONTACT",
        "BIDIRECTIONAL", "ENDPOINT_NATIVE", "not-applicable")
    trajectory = {
        "id": "case:rope-core",
        "sha256": contracts["case:rope-core"]["sha256"],
    }
    aggregate = {
        "schema": qualification.ROADMAP_EVIDENCE_SCHEMA,
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
            "driver_version": "456 (test)",
        },
        "toolchain": {"compiler": "test-compiler", "version": "7.8.9"},
        "advertised_capabilities": [advertised],
        "contracts": {
            "trajectory": trajectory,
            "quality": {
                "id": "quality:rope-rigid-contact-v1",
                "sha256": contracts[
                    "quality:rope-rigid-contact-v1"]["sha256"],
            },
            "lifecycle": copy.deepcopy(trajectory),
        },
        "trajectory": {
            "execution_status": "completed",
            "reference_regression": {
                "passed": 1, "failed": 0, "unsupported": 0,
            },
        },
        "analytic_invariants": {
            "passed": True, "sample_count": 1, "failure_count": 0,
            "observations": {},
        },
        "lifecycle_invariants": {
            "passed": True, "failure_count": 0,
        },
    }
    provenance = {
        "source_commit": SOURCE_COMMIT,
        "device_name": "test-device",
        "vendor_id": 1,
        "device_id": 2,
        "driver_version": 456,
        "host_compiler": "test-compiler",
    }
    return aggregate, provenance, [advertised]


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

    def test_cloth_smoke_contact_contracts_do_not_qualify_drag(self):
        contracts = contract_map(self.profile)
        expected_contracts = {
            "quality:cloth-smoke-contact-v1":
                ROOT / "quality" / "cloth-smoke-contact-v1.json",
            ("lifecycle:cloth-smoke-contact-lifecycle-v1#"
             "grid-cloth-smoke-contact-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "cloth-smoke-contact-lifecycle-v1.json",
        }
        for contract_id, path in expected_contracts.items():
            with self.subTest(contract=contract_id):
                self.assertEqual(contracts[contract_id]["path"],
                                 path.relative_to(REPOSITORY).as_posix())
                self.assertEqual(contracts[contract_id]["sha256"],
                                 qualification.profile_sha256(path))

        for material in ("SMOKE_PARTICLE", "SMOKE_GRID"):
            with self.subTest(material=material, mode="CONTACT"):
                report = self.evaluate(
                    cloth_smoke_contact_result(self.profile, material))
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["qualified_count"], 1)
                self.assertEqual(report["capabilities"][0]["rule_id"],
                                 "cloth-smoke-contact-" +
                                 material.removeprefix("SMOKE_").lower())

            with self.subTest(material=material, mode="DRAG"):
                report = self.assert_rejected(
                    cloth_smoke_contact_result(self.profile, material, "DRAG"),
                    "required evidence is missing")
                self.assertEqual(report["qualified_count"], 0)
                self.assertEqual(report["capabilities"][0]["rule_id"],
                                 "cloth-smoke-drag-" +
                                 material.removeprefix("SMOKE_").lower())

    def test_soft_rope_attachment_keeps_exact_lifecycle_obligation(self):
        attachment_rule = next(
            rule for rule in self.profile["rules"]
            if rule["id"] == "soft-body-rope-attachment")
        lifecycle_contract = (
            "lifecycle:adverse-capacity-three-family-v1#"
            "three-family-partial-rollback")
        self.assertEqual(
            attachment_rule["required"]["lifecycle_invariants"],
            [lifecycle_contract])

        result = soft_rope_attachment_result(self.profile)
        report = self.evaluate(result)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["qualified_count"], 1)
        self.assertEqual(report["capabilities"][0]["rule_id"],
                         "soft-body-rope-attachment")

        result = soft_rope_attachment_result(self.profile)
        result["evidence"] = [item for item in result["evidence"]
                              if item["contract"] != lifecycle_contract]
        report = self.assert_rejected(result, "required evidence is missing")
        self.assertEqual(report["qualified_count"], 0)

    def test_soft_rigid_contact_has_complete_intrinsic_obligations(self):
        result = soft_rigid_contact_result(self.profile)
        report = self.evaluate(result)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["qualified_count"], 1)
        capability_report = report["capabilities"][0]
        self.assertEqual(capability_report["rule_id"], "soft-body-rigid")
        self.assertEqual(capability_report[
            "max_graph_edges_per_simulation"], "not-applicable")
        for scope in ("execution", "reference_regression",
                      "lifecycle_invariants", "analytic_invariants"):
            self.assertEqual(capability_report["scopes"][scope]["status"],
                             "passed")

        for missing_scope in ("lifecycle_invariants", "analytic_invariants"):
            with self.subTest(missing_scope=missing_scope):
                incomplete = soft_rigid_contact_result(self.profile)
                incomplete["evidence"] = [
                    item for item in incomplete["evidence"]
                    if item["scope"] != missing_scope]
                report = self.assert_rejected(
                    incomplete, "required evidence is missing")
                self.assertEqual(report["qualified_count"], 0)

    def test_soft_rigid_contact_rejects_nearby_contract_substitution(self):
        result = soft_rigid_contact_result(self.profile)
        result["evidence"][-2] = evidence(
            self.profile, "lifecycle_invariants",
            "lifecycle:adverse-capacity-three-family-v1#"
            "three-family-partial-rollback")
        result["evidence"][-1] = evidence(
            self.profile, "analytic_invariants",
            "quality:fluid-rigid-contact-v1")
        report = self.assert_rejected(result, "required evidence is missing")
        self.assertEqual(report["qualified_count"], 0)

        nearby = soft_rigid_contact_result(self.profile, "CLOTH")
        report = self.assert_rejected(nearby, "required evidence is missing")
        self.assertEqual(report["qualified_count"], 0)

    def test_rope_rigid_contact_is_exact_and_attachment_remains_missing(self):
        contracts = contract_map(self.profile)
        profile_path = ROOT / "quality" / "rope-rigid-contact-v1.json"
        self.assertEqual(
            contracts["quality:rope-rigid-contact-v1"]["path"],
            profile_path.relative_to(REPOSITORY).as_posix())
        self.assertEqual(
            contracts["quality:rope-rigid-contact-v1"]["sha256"],
            qualification.profile_sha256(profile_path))

        contact = self.evaluate(rope_rigid_contact_result(self.profile))
        self.assertTrue(contact["passed"], contact)
        self.assertEqual(contact["qualified_count"], 1)
        self.assertEqual(contact["capabilities"][0]["rule_id"],
                         "rope-rigid-contact")
        self.assertEqual(contact["capabilities"][0][
            "max_graph_edges_per_simulation"], "not-applicable")
        for scope in ("execution", "reference_regression",
                      "lifecycle_invariants", "analytic_invariants"):
            self.assertEqual(
                contact["capabilities"][0]["scopes"][scope]["status"],
                "passed")

        attachment = self.assert_rejected(
            rope_rigid_contact_result(self.profile, "ATTACHMENT"),
            "qualification contract is missing")
        self.assertEqual(attachment["qualified_count"], 0)
        self.assertEqual(attachment["capabilities"][0]["rule_id"],
                         "rope-rigid-attachment")
        self.assertEqual(attachment["capabilities"][0]["scopes"]
                         ["analytic_invariants"]["status"],
                         "missing_contract")
        self.assertIn(
            {"scope": "analytic_invariants",
             "contract": "quality:rope-rigid-contact-v1"},
            attachment["unused_evidence"])

    def test_rope_rigid_contact_rejects_contract_substitution(self):
        result = rope_rigid_contact_result(self.profile)
        result["evidence"][-2] = evidence(
            self.profile, "lifecycle_invariants", "case:rigid-direct")
        result["evidence"][-1] = evidence(
            self.profile, "analytic_invariants",
            "quality:fluid-rigid-contact-v1")
        report = self.assert_rejected(result, "required evidence is missing")
        self.assertEqual(report["qualified_count"], 0)
        self.assertEqual(
            {item["contract"] for item in report["unused_evidence"]},
            {"case:rigid-direct", "quality:fluid-rigid-contact-v1"})

    def test_soft_smoke_contact_and_drag_contracts_are_exact(self):
        contracts = contract_map(self.profile)
        expected_contracts = {
            "quality:soft-smoke-contact-v1":
                ROOT / "quality" / "soft-smoke-contact-v1.json",
            ("lifecycle:soft-smoke-contact-lifecycle-v1#"
             "grid-soft-smoke-contact-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "soft-smoke-contact-lifecycle-v1.json",
            "quality:smoke-soft-drag-v1":
                ROOT / "quality" / "smoke-soft-drag-v1.json",
            ("lifecycle:smoke-soft-drag-lifecycle-v1#"
             "particle-smoke-soft-drag-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "smoke-soft-drag-lifecycle-v1.json",
            ("lifecycle:smoke-soft-drag-lifecycle-v1#"
             "grid-smoke-soft-drag-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "smoke-soft-drag-lifecycle-v1.json",
        }
        for contract_id, path in expected_contracts.items():
            with self.subTest(contract=contract_id):
                self.assertEqual(contracts[contract_id]["path"],
                                 path.relative_to(REPOSITORY).as_posix())
                self.assertEqual(contracts[contract_id]["sha256"],
                                 qualification.profile_sha256(path))

        for material in ("SMOKE_PARTICLE", "SMOKE_GRID"):
            suffix = material.removeprefix("SMOKE_").lower()
            with self.subTest(material=material, mode="CONTACT"):
                report = self.evaluate(
                    soft_smoke_contact_result(self.profile, material))
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["qualified_count"], 1)
                self.assertEqual(report["capabilities"][0]["rule_id"],
                                 "soft-body-smoke-contact-" + suffix)

            with self.subTest(material=material, mode="DRAG"):
                report = self.evaluate(
                    soft_smoke_contact_result(self.profile, material, "DRAG"))
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["qualified_count"], 1)
                self.assertEqual(report["capabilities"][0]["rule_id"],
                                 "soft-body-smoke-drag-" + suffix)

                wrong = soft_smoke_contact_result(
                    self.profile, material, "DRAG")
                wrong["evidence"][-1] = evidence(
                    self.profile, "analytic_invariants",
                    "quality:soft-smoke-contact-v1")
                report = self.assert_rejected(
                    wrong, "required evidence is missing")
                self.assertEqual(report["qualified_count"], 0)
                self.assertIn(
                    "quality:soft-smoke-contact-v1",
                    {item["contract"] for item in report["unused_evidence"]})

    def test_rope_smoke_contact_contracts_are_exact_and_disjoint_from_drag(self):
        contracts = contract_map(self.profile)
        expected_contracts = {
            "quality:rope-smoke-contact-v1":
                ROOT / "quality" / "rope-smoke-contact-v1.json",
            ("lifecycle:rope-smoke-contact-lifecycle-v1#"
             "particle-rope-smoke-contact-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "rope-smoke-contact-lifecycle-v1.json",
            ("lifecycle:rope-smoke-contact-lifecycle-v1#"
             "grid-rope-smoke-contact-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "rope-smoke-contact-lifecycle-v1.json",
            "quality:smoke-rope-drag-v1":
                ROOT / "quality" / "smoke-rope-drag-v1.json",
            ("lifecycle:smoke-rope-drag-lifecycle-v1#"
             "particle-smoke-rope-drag-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "smoke-rope-drag-lifecycle-v1.json",
            ("lifecycle:smoke-rope-drag-lifecycle-v1#"
             "grid-smoke-rope-drag-capacity-replacement-removal-rollback"):
                ROOT / "lifecycle" / "smoke-rope-drag-lifecycle-v1.json",
        }
        for contract_id, path in expected_contracts.items():
            with self.subTest(contract=contract_id):
                self.assertEqual(contracts[contract_id]["path"],
                                 path.relative_to(REPOSITORY).as_posix())
                self.assertEqual(contracts[contract_id]["sha256"],
                                 qualification.profile_sha256(path))

        for material in ("SMOKE_PARTICLE", "SMOKE_GRID"):
            suffix = material.removeprefix("SMOKE_").lower()
            with self.subTest(material=material):
                report = self.evaluate(
                    rope_smoke_contact_result(self.profile, material))
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["qualified_count"], 1)
                self.assertEqual(report["capabilities"][0]["rule_id"],
                                 "rope-smoke-contact-" + suffix)

                wrong = rope_smoke_contact_result(self.profile, material)
                wrong["evidence"][-1] = evidence(
                    self.profile, "analytic_invariants",
                    "quality:smoke-rope-drag-v1")
                rejected = self.assert_rejected(
                    wrong, "required evidence is missing")
                self.assertEqual(rejected["qualified_count"], 0)

                drag = rope_smoke_contact_result(
                    self.profile, material, "DRAG")
                drag_report = self.evaluate(drag)
                self.assertTrue(drag_report["passed"], drag_report)
                self.assertEqual(drag_report["qualified_count"], 1)
                self.assertEqual(drag_report["capabilities"][0]["rule_id"],
                                 "rope-smoke-drag-" + suffix)

                wrong_drag = rope_smoke_contact_result(
                    self.profile, material, "DRAG")
                wrong_drag["evidence"][-1] = evidence(
                    self.profile, "analytic_invariants",
                    "quality:rope-smoke-contact-v1")
                rejected_drag = self.assert_rejected(
                    wrong_drag, "required evidence is missing")
                self.assertEqual(rejected_drag["qualified_count"], 0)

    def test_soft_rope_attachment_quality_cannot_qualify_contact(self):
        report = self.evaluate(
            soft_rope_attachment_result(self.profile, "CONTACT"))
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["qualified_count"], 0)
        self.assertEqual(report["capabilities"][0]["rule_id"],
                         "soft-body-rope-contact")
        self.assertEqual(report["unused_evidence"][0]["contract"],
                         "quality:soft-rope-attachment-v1")

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

    def test_checked_producer_derives_passing_fluid_soft_result(self):
        aggregate, provenance, capabilities = fluid_soft_aggregate(self.profile)
        result = qualification.derive_result(
            self.profile, aggregate, self.digest,
            "conformance/v1/evidence/test.json", digest("aggregate"),
            provenance, capabilities)
        report = self.evaluate(result)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["qualified_count"], 1)
        self.assertEqual(report["capabilities"][0]["rule_id"],
                         "fluid-soft-body")

    def test_checked_producer_derives_exact_rope_rigid_contact(self):
        aggregate, provenance, capabilities = rope_rigid_contact_aggregate(
            self.profile)
        result = qualification.derive_result(
            self.profile, aggregate, self.digest,
            "conformance/v1/evidence/test-rope-rigid.json",
            digest("rope-rigid-aggregate"), provenance, capabilities)
        report = self.evaluate(result)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["qualified_count"], 1)
        self.assertEqual(report["capabilities"][0]["rule_id"],
                         "rope-rigid-contact")

    def test_case_lifecycle_capture_requires_exact_trajectory_reuse(self):
        aggregate, provenance, _ = rope_rigid_contact_aggregate(self.profile)
        contracts = contract_map(self.profile)
        trajectory_result = {
            "schema": "parallel-mater-conformance-result/v1",
            "case_id": "rope-core",
            "case_sha256": contracts["case:rope-core"]["sha256"],
            "execution_status": "completed",
            "provenance": copy.deepcopy(provenance),
        }
        trajectory_report = {
            "schema": "parallel-mater-conformance-report/v1",
            "scope": "reference_regression",
            "passed": 1,
            "failed": 0,
            "unsupported": 0,
            "cases": [{
                "id": "rope-core", "passed": True,
                "status": "passed", "reason": "", "differences": [],
            }],
        }
        quality_sha = contracts["quality:rope-rigid-contact-v1"]["sha256"]
        quality_result = {
            "schema": "parallel-mater-conformance-quality-result/v1",
            "profile_id": "rope-rigid-contact-v1",
            "profile_sha256": quality_sha,
            "execution_status": "completed",
        }
        quality_report = {
            "schema": "parallel-mater-conformance-quality-report/v1",
            "profile_id": "rope-rigid-contact-v1",
            "profile_sha256": quality_sha,
            "passed": True,
            "failure_count": 0,
            "sample_count": 1,
            "observations": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)

            def write(name: str, value: dict) -> pathlib.Path:
                path = root / name
                path.write_text(json.dumps(value), encoding="utf-8")
                return path

            driver = root / "driver.bin"
            driver.write_bytes(b"driver")
            paths = {
                "driver_binary": driver,
                "provenance": write("provenance.json", provenance),
                "capabilities": write("capabilities.json", {}),
                "trajectory_result": write(
                    "trajectory.json", trajectory_result),
                "trajectory_report": write(
                    "trajectory-report.json", trajectory_report),
                "quality_result": write("quality.json", quality_result),
                "quality_report": write(
                    "quality-report.json", quality_report),
                "lifecycle_result": write(
                    "lifecycle.json", trajectory_result),
                "lifecycle_report": write(
                    "lifecycle-report.json", trajectory_report),
            }
            aggregate["driver"]["binary_sha256"] = (
                qualification.file_sha256(driver))
            aggregate["captures"] = {
                "provenance_sha256": qualification.file_sha256(
                    paths["provenance"]),
                "capabilities_sha256": qualification.file_sha256(
                    paths["capabilities"]),
            }
            aggregate["trajectory"].update({
                "execution_result_sha256": qualification.file_sha256(
                    paths["trajectory_result"]),
                "comparison_report_sha256": qualification.file_sha256(
                    paths["trajectory_report"]),
            })
            aggregate["analytic_invariants"].update({
                "result_sha256": qualification.file_sha256(
                    paths["quality_result"]),
                "report_sha256": qualification.file_sha256(
                    paths["quality_report"]),
            })
            aggregate["lifecycle_invariants"].update({
                "result_sha256": aggregate["trajectory"]
                ["execution_result_sha256"],
                "report_sha256": aggregate["trajectory"]
                ["comparison_report_sha256"],
            })

            qualification._verify_capture_hashes(aggregate, paths)
            qualification._verify_capture_results(aggregate, paths)

            for field in ("result_sha256", "report_sha256"):
                with self.subTest(field=field):
                    mismatched = copy.deepcopy(aggregate)
                    mismatched["lifecycle_invariants"][field] = digest(field)
                    with self.assertRaisesRegex(
                            ValueError,
                            "case lifecycle must exactly reuse its trajectory capture"):
                        qualification._verify_capture_results(
                            mismatched, paths)

            wrong_contract = copy.deepcopy(aggregate)
            wrong_contract["contracts"]["lifecycle"] = {
                "id": "case:rigid-direct",
                "sha256": contracts["case:rigid-direct"]["sha256"],
            }
            with self.assertRaisesRegex(
                    ValueError,
                    "case lifecycle must exactly reuse its trajectory capture"):
                qualification._verify_capture_results(wrong_contract, paths)

    def test_checked_producer_rejects_unproved_claims(self):
        aggregate, provenance, capabilities = fluid_soft_aggregate(self.profile)
        aggregate["analytic_invariants"]["passed"] = False
        with self.assertRaisesRegex(ValueError,
                                    "analytic invariants did not pass"):
            qualification.derive_result(
                self.profile, aggregate, self.digest, "test.json",
                digest("aggregate"), provenance, capabilities)

        aggregate, provenance, capabilities = fluid_soft_aggregate(self.profile)
        with self.assertRaisesRegex(ValueError,
                                    "capability was not captured"):
            qualification.derive_result(
                self.profile, aggregate, self.digest, "test.json",
                digest("aggregate"), provenance, [])


if __name__ == "__main__":
    unittest.main()
