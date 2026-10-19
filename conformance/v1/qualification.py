#!/usr/bin/env python3
"""Validate revisioned evidence for advertised interaction capabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "parallel-mater-conformance-obligations/v1"
RESULT_SCHEMA = "parallel-mater-conformance-qualification-result/v1"
REPORT_SCHEMA = "parallel-mater-conformance-qualification-report/v1"
ROADMAP_EVIDENCE_SCHEMA = "parallel-mater-roadmap-evidence/v1"
KEY_FIELDS = (
    "first_family", "second_family", "scope", "mode", "direction",
    "material_profile")
EVIDENCE_STATUSES = {"passed", "failed", "unsupported", "missing"}
HEX = re.compile(r"[0-9a-f]+")
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class Failure:
    path: str
    message: str
    actual: Any = None
    expected: Any = None


def profile_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".json", ".xml"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _index(values: Any, path: str, failures: list[Failure]) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        failures.append(Failure(path, "expected an array", values))
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            failures.append(Failure(item_path, "item needs a string id", value))
            continue
        identity = value["id"]
        if identity in result:
            failures.append(Failure(item_path + ".id", "duplicate id", identity))
            continue
        result[identity] = value
    return result


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _source_revision(value: Any) -> bool:
    return (isinstance(value, str) and len(value) in (40, 64) and
            HEX.fullmatch(value) is not None)


def _key_tuple(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(value.get(field) for field in KEY_FIELDS)


def _key_json(value: dict[str, Any]) -> dict[str, Any]:
    return {field: value.get(field) for field in KEY_FIELDS}


def _matches(rule: dict[str, Any], capability: dict[str, Any]) -> bool:
    match = rule.get("match")
    if not isinstance(match, dict):
        return False
    if any(capability.get(field) not in match.get(field, []) for field in KEY_FIELDS):
        return False
    pairs = rule.get("mode_directions")
    return (pairs is None or
            [capability.get("mode"), capability.get("direction")] in pairs)


def matching_rules(profile: dict[str, Any],
                   capability: dict[str, Any]) -> list[dict[str, Any]]:
    return [rule for rule in profile.get("rules", [])
            if isinstance(rule, dict) and _matches(rule, capability)]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required(rule: dict[str, Any], profile: dict[str, Any],
              failures: list[Failure], path: str) -> dict[str, list[str]]:
    scopes = set(profile.get("scopes", []))
    required: dict[str, list[str]] = {}
    for source_path, source in (("$.profile.default_required",
                                 profile.get("default_required", {})),
                                (path + ".required", rule.get("required", {}))):
        if not isinstance(source, dict):
            failures.append(Failure(source_path, "expected an object", source))
            continue
        for scope, contracts in source.items():
            if scope not in scopes:
                failures.append(Failure(source_path + "." + scope,
                                        "unknown evidence scope", scope))
                continue
            if not isinstance(contracts, list) or not all(
                    isinstance(item, str) and item for item in contracts):
                failures.append(Failure(source_path + "." + scope,
                                        "expected contract id array", contracts))
                continue
            required.setdefault(scope, []).extend(contracts)
    trajectory = rule.get("trajectory_contract")
    if trajectory is not None:
        if not isinstance(trajectory, str) or not trajectory:
            failures.append(Failure(path + ".trajectory_contract",
                                    "expected a contract id", trajectory))
        else:
            required.setdefault("execution", []).append(trajectory)
            required.setdefault("reference_regression", []).append(trajectory)
    for scope, contracts in required.items():
        if len(set(contracts)) != len(contracts):
            failures.append(Failure(path + ".required." + scope,
                                    "duplicate required contract", contracts))
    return required


def _metadata(result: dict[str, Any], failures: list[Failure]) -> str | None:
    source = result.get("source")
    commit = source.get("commit") if isinstance(source, dict) else None
    if not isinstance(source, dict) or not _source_revision(commit):
        failures.append(Failure("$.source.commit",
                                "expected a 40- or 64-digit lowercase revision",
                                commit))
        commit = None
    if not isinstance(source, dict) or source.get("dirty") is not False:
        failures.append(Failure("$.source.dirty",
                                "qualification source must be clean",
                                source.get("dirty") if isinstance(source, dict) else None,
                                False))

    driver = result.get("driver")
    if (not isinstance(driver, dict) or
            not _nonempty(driver.get("name")) or
            not _nonempty(driver.get("version")) or
            not _sha256(driver.get("binary_sha256"))):
        failures.append(Failure("$.driver",
                                "driver name, version and binary SHA-256 are required",
                                driver))
    device = result.get("device")
    if (not isinstance(device, dict) or not _nonempty(device.get("name")) or
            not isinstance(device.get("vendor_id"), int) or
            isinstance(device.get("vendor_id"), bool) or
            device["vendor_id"] < 0 or
            not isinstance(device.get("device_id"), int) or
            isinstance(device.get("device_id"), bool) or
            device["device_id"] < 0 or
            not _nonempty(device.get("driver_version"))):
        failures.append(Failure("$.device",
                                "device name, IDs and driver version are required",
                                device))
    toolchain = result.get("toolchain")
    if (not isinstance(toolchain, dict) or
            not _nonempty(toolchain.get("compiler")) or
            not _nonempty(toolchain.get("version"))):
        failures.append(Failure("$.toolchain",
                                "compiler name and version are required", toolchain))
    return commit


def _evidence(profile: dict[str, Any], result: dict[str, Any], commit: str | None,
              contracts: dict[str, dict[str, Any]], failures: list[Failure]
              ) -> tuple[dict[tuple[str, str], dict[str, Any]],
                         set[tuple[str, str]]]:
    scopes = set(profile.get("scopes", []))
    values = result.get("evidence")
    if not isinstance(values, list):
        failures.append(Failure("$.evidence", "expected an evidence array", values))
        return {}, set()
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    invalid: set[tuple[str, str]] = set()
    for index, value in enumerate(values):
        path = f"$.evidence[{index}]"
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected an object", value))
            continue
        scope = value.get("scope")
        contract_id = value.get("contract")
        if scope not in scopes:
            failures.append(Failure(path + ".scope", "unknown evidence scope", scope))
            continue
        if not isinstance(contract_id, str) or contract_id not in contracts:
            failures.append(Failure(path + ".contract",
                                    "unknown evidence contract", contract_id))
            continue
        key = (scope, contract_id)
        if key in indexed:
            failures.append(Failure(path, "duplicate scoped evidence", list(key)))
            continue
        indexed[key] = value
        status = value.get("status")
        if status not in EVIDENCE_STATUSES:
            invalid.add(key)
            failures.append(Failure(path + ".status",
                                    "invalid evidence status", status,
                                    sorted(EVIDENCE_STATUSES)))
        if value.get("source_commit") != commit:
            invalid.add(key)
            failures.append(Failure(path + ".source_commit",
                                    "evidence source revision mismatch",
                                    value.get("source_commit"), commit))
        expected_hash = contracts[contract_id].get("sha256")
        if value.get("contract_sha256") != expected_hash:
            invalid.add(key)
            failures.append(Failure(path + ".contract_sha256",
                                    "evidence contract hash mismatch",
                                    value.get("contract_sha256"), expected_hash))
        if not _sha256(value.get("artifact_sha256")):
            invalid.add(key)
            failures.append(Failure(path + ".artifact_sha256",
                                    "evidence artifact SHA-256 is required",
                                    value.get("artifact_sha256")))
    return indexed, invalid


def evaluate(profile: dict[str, Any], result: dict[str, Any],
             expected_hash: str) -> dict[str, Any]:
    failures: list[Failure] = []
    if profile.get("schema") != PROFILE_SCHEMA:
        failures.append(Failure("$.profile.schema", "unsupported profile schema",
                                profile.get("schema"), PROFILE_SCHEMA))
    if result.get("schema") != RESULT_SCHEMA:
        failures.append(Failure("$.schema", "unsupported result schema",
                                result.get("schema"), RESULT_SCHEMA))
    if result.get("obligations_id") != profile.get("id"):
        failures.append(Failure("$.obligations_id", "obligations identity mismatch",
                                result.get("obligations_id"), profile.get("id")))
    if result.get("obligations_sha256") != expected_hash:
        failures.append(Failure("$.obligations_sha256", "obligations hash mismatch",
                                result.get("obligations_sha256"), expected_hash))
    if result.get("execution_status") != "completed":
        failures.append(Failure("$.execution_status",
                                "qualification evaluation must be completed",
                                result.get("execution_status"), "completed"))
    commit = _metadata(result, failures)

    contracts = _index(profile.get("contracts"), "$.profile.contracts", failures)
    evidence, invalid_evidence = _evidence(
        profile, result, commit, contracts, failures)
    used_evidence: set[tuple[str, str]] = set()
    advertised = result.get("advertised_capabilities")
    if not isinstance(advertised, list):
        failures.append(Failure("$.advertised_capabilities",
                                "expected a capability array", advertised))
        advertised = []
    global_valid = not failures
    seen: set[tuple[Any, ...]] = set()
    capability_reports: list[dict[str, Any]] = []
    scopes = list(profile.get("scopes", []))
    optional_scopes = set(profile.get("optional_scopes", []))
    for index, capability in enumerate(advertised):
        path = f"$.advertised_capabilities[{index}]"
        before = len(failures)
        if not isinstance(capability, dict):
            failures.append(Failure(path, "expected an object", capability))
            continue
        key = _key_tuple(capability)
        if not all(isinstance(item, str) and item for item in key):
            failures.append(Failure(path, "capability key is incomplete",
                                    _key_json(capability)))
            continue
        if key in seen:
            failures.append(Failure(path, "duplicate advertised capability",
                                    _key_json(capability)))
            continue
        seen.add(key)
        limit = capability.get("max_graph_edges_per_simulation")
        if not ((isinstance(limit, int) and not isinstance(limit, bool) and limit > 0) or
                limit in ("not-applicable", "simulation-capacity")):
            failures.append(Failure(path + ".max_graph_edges_per_simulation",
                                    "invalid graph limit", limit))
        matches = matching_rules(profile, capability)
        scope_report = {scope: {"status": "not_required", "contracts": []}
                        for scope in scopes}
        rule_id = None
        if len(matches) != 1:
            failures.append(Failure(path, "capability must match exactly one obligation rule",
                                    [rule.get("id") for rule in matches]))
        else:
            rule = matches[0]
            rule_id = rule.get("id")
            required = _required(rule, profile, failures,
                                 f"$.profile.rules[{rule_id}]")
            missing = rule.get("missing", {})
            if not isinstance(missing, dict):
                failures.append(Failure(f"$.profile.rules[{rule_id}].missing",
                                        "expected an object", missing))
                missing = {}
            for scope in scopes:
                contracts_for_scope = required.get(scope, [])
                scope_report[scope]["contracts"] = contracts_for_scope
                if scope in missing:
                    scope_report[scope]["status"] = "missing_contract"
                    failures.append(Failure(path + "." + scope,
                                            "qualification contract is missing",
                                            missing[scope]))
                    continue
                if not contracts_for_scope:
                    continue
                statuses: list[str] = []
                for contract_id in contracts_for_scope:
                    if contract_id not in contracts:
                        statuses.append("missing_contract")
                        failures.append(Failure(path + "." + scope,
                                                "unknown required contract",
                                                contract_id))
                        continue
                    record = evidence.get((scope, contract_id))
                    if record is None:
                        statuses.append("missing_evidence")
                        failures.append(Failure(path + "." + scope,
                                                "required evidence is missing",
                                                contract_id))
                        continue
                    used_evidence.add((scope, contract_id))
                    if (scope, contract_id) in invalid_evidence:
                        statuses.append("invalid")
                        failures.append(Failure(path + "." + scope,
                                                "required evidence metadata is invalid",
                                                contract_id))
                        continue
                    status = str(record.get("status"))
                    statuses.append(status)
                    if status != "passed":
                        failures.append(Failure(path + "." + scope,
                                                "required evidence did not pass",
                                                status, "passed"))
                scope_report[scope]["status"] = (
                    "passed" if statuses and all(item == "passed" for item in statuses)
                    else next((item for item in statuses if item != "passed"),
                              "missing_evidence"))
            for scope in optional_scopes:
                if scope in scope_report and scope_report[scope]["status"] == "not_required":
                    scope_report[scope]["status"] = "optional"
        capability_reports.append({
            "key": _key_json(capability),
            "max_graph_edges_per_simulation": limit,
            "rule_id": rule_id,
            "qualified": global_valid and len(failures) == before,
            "scopes": scope_report,
        })

    evidence_summary = {
        scope: {status: sum(1 for (record_scope, _), record in evidence.items()
                           if record_scope == scope and record.get("status") == status)
                for status in sorted(EVIDENCE_STATUSES)}
        for scope in scopes
    }
    unused = [{"scope": scope, "contract": contract}
              for scope, contract in sorted(set(evidence) - used_evidence)]
    return {
        "schema": REPORT_SCHEMA,
        "scope": "capability_qualification",
        "obligations_id": profile.get("id"),
        "obligations_sha256": expected_hash,
        "source_commit": commit,
        "driver": result.get("driver"),
        "device": result.get("device"),
        "toolchain": result.get("toolchain"),
        "passed": not failures,
        "capability_count": len(capability_reports),
        "qualified_count": sum(item["qualified"] for item in capability_reports),
        "capabilities": capability_reports,
        "evidence_summary": evidence_summary,
        "unused_evidence": unused,
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }


def _aggregate_contract(aggregate: dict[str, Any], kind: str,
                        contract_id: str,
                        contracts: dict[str, dict[str, Any]]) -> None:
    aggregate_contracts = aggregate.get("contracts")
    value = (aggregate_contracts.get(kind)
             if isinstance(aggregate_contracts, dict) else None)
    if not isinstance(value, dict) or value.get("id") != contract_id:
        raise ValueError(
            f"physical aggregate does not bind {kind} contract {contract_id}")
    expected = contracts.get(contract_id, {}).get("sha256")
    if not expected or value.get("sha256") != expected:
        raise ValueError(
            f"physical aggregate has stale {kind} contract {contract_id}")


def _aggregate_scope_passed(aggregate: dict[str, Any], scope: str,
                            contract_id: str,
                            contracts: dict[str, dict[str, Any]]) -> None:
    if scope == "compilation":
        if contract_id != "registry:interaction-capabilities":
            raise ValueError(f"unsupported compilation contract {contract_id}")
        return
    if scope in {"execution", "reference_regression"}:
        _aggregate_contract(aggregate, "trajectory", contract_id, contracts)
        trajectory = aggregate.get("trajectory")
        if (not isinstance(trajectory, dict) or
                trajectory.get("execution_status") != "completed"):
            raise ValueError("physical trajectory execution did not complete")
        if scope == "reference_regression":
            regression = trajectory.get("reference_regression")
            if (not isinstance(regression, dict) or
                    regression.get("passed", 0) < 1 or
                    regression.get("failed") != 0 or
                    regression.get("unsupported") != 0):
                raise ValueError("physical trajectory regression did not pass")
        return
    if scope == "lifecycle_invariants":
        _aggregate_contract(aggregate, "lifecycle", contract_id, contracts)
        lifecycle = aggregate.get("lifecycle_invariants")
        if (not isinstance(lifecycle, dict) or
                lifecycle.get("passed") is not True or
                lifecycle.get("failure_count") != 0):
            raise ValueError("physical lifecycle invariants did not pass")
        return
    if scope == "analytic_invariants":
        _aggregate_contract(aggregate, "quality", contract_id, contracts)
        quality = aggregate.get("analytic_invariants")
        if (not isinstance(quality, dict) or quality.get("passed") is not True or
                quality.get("failure_count") != 0):
            raise ValueError("physical analytic invariants did not pass")
        return
    raise ValueError(f"physical aggregate cannot prove {scope}")


def derive_result(profile: dict[str, Any], aggregate: dict[str, Any],
                  obligations_sha256: str, artifact_path: str,
                  artifact_sha256: str, provenance: dict[str, Any],
                  captured_capabilities: list[Any]) -> dict[str, Any]:
    """Derive and evaluate a qualification input from a physical aggregate."""
    if aggregate.get("schema") != ROADMAP_EVIDENCE_SCHEMA:
        raise ValueError("unsupported physical evidence schema")
    source = aggregate.get("source")
    driver = aggregate.get("driver")
    device = aggregate.get("device")
    toolchain = aggregate.get("toolchain")
    advertised = aggregate.get("advertised_capabilities")
    if not all(isinstance(value, dict)
               for value in (source, driver, device, toolchain)):
        raise ValueError("physical aggregate provenance is incomplete")
    if not isinstance(advertised, list) or not advertised:
        raise ValueError("physical aggregate has no advertised capabilities")
    if source.get("dirty") is not False:
        raise ValueError("physical aggregate source is dirty")
    if provenance.get("source_commit") != source.get("commit"):
        raise ValueError("captured provenance source revision mismatch")
    for aggregate_key, provenance_key in (
            ("name", "device_name"), ("vendor_id", "vendor_id"),
            ("device_id", "device_id")):
        if device.get(aggregate_key) != provenance.get(provenance_key):
            raise ValueError(
                f"captured provenance device {aggregate_key} mismatch")
    if not str(device.get("driver_version", "")).startswith(
            str(provenance.get("driver_version", "missing"))):
        raise ValueError("captured provenance driver version mismatch")

    captured = {
        _key_tuple(value) + (value.get("max_graph_edges_per_simulation"),)
        for value in captured_capabilities if isinstance(value, dict)
    }
    for capability in advertised:
        if not isinstance(capability, dict):
            raise ValueError("physical aggregate capability is not an object")
        identity = (_key_tuple(capability) +
                    (capability.get("max_graph_edges_per_simulation"),))
        if identity not in captured:
            raise ValueError("physical aggregate capability was not captured")

    failures: list[Failure] = []
    contracts = _index(profile.get("contracts"),
                       "$.profile.contracts", failures)
    if failures:
        raise ValueError(failures[0].message)
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for index, capability in enumerate(advertised):
        rules = matching_rules(profile, capability)
        if len(rules) != 1:
            raise ValueError(
                f"advertised capability {index} does not match one obligation rule")
        rule = rules[0]
        rule_failures: list[Failure] = []
        required = _required(rule, profile, rule_failures,
                             f"$.profile.rules[{rule.get('id')}]")
        if rule_failures:
            raise ValueError(rule_failures[0].message)
        missing = rule.get("missing", {})
        if not isinstance(missing, dict):
            raise ValueError("obligation rule missing map is invalid")
        for scope, contract_ids in required.items():
            if scope in missing:
                raise ValueError(
                    f"obligation rule has no physical {scope} contract")
            for contract_id in contract_ids:
                _aggregate_scope_passed(
                    aggregate, scope, contract_id, contracts)
                contract_hash = contracts.get(contract_id, {}).get("sha256")
                evidence[(scope, contract_id)] = {
                    "scope": scope,
                    "contract": contract_id,
                    "status": "passed",
                    "source_commit": source.get("commit"),
                    "contract_sha256": contract_hash,
                    "artifact_path": artifact_path,
                    "artifact_sha256": artifact_sha256,
                }

    result = {
        "schema": RESULT_SCHEMA,
        "obligations_id": profile.get("id"),
        "obligations_sha256": obligations_sha256,
        "execution_status": "completed",
        "source": source,
        "driver": driver,
        "device": device,
        "toolchain": toolchain,
        "advertised_capabilities": advertised,
        "evidence": [evidence[key] for key in sorted(evidence)],
    }
    report = evaluate(profile, result, obligations_sha256)
    if not report["passed"]:
        first = report["failures"][0]
        raise ValueError(
            f"derived qualification failed: {first['path']}: "
            f"{first['message']}")
    return result


def _capture_paths(arguments: argparse.Namespace) -> dict[str, Path]:
    names = (
        "driver_binary", "provenance", "capabilities", "trajectory_result",
        "trajectory_report", "quality_result", "quality_report",
        "lifecycle_result", "lifecycle_report")
    values = {name: getattr(arguments, name) for name in names}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "--from-evidence requires " +
            ", ".join("--" + name.replace("_", "-") for name in missing))
    return values


def _verify_capture_hashes(aggregate: dict[str, Any],
                           paths: dict[str, Path]) -> None:
    captures = aggregate.get("captures")
    if not isinstance(captures, dict):
        raise ValueError("physical aggregate is missing capture hashes")
    fields = {
        "driver_binary": aggregate.get("driver", {}).get("binary_sha256"),
        "provenance": captures.get("provenance_sha256"),
        "capabilities": captures.get("capabilities_sha256"),
        "trajectory_result": aggregate.get("trajectory", {}).get(
            "execution_result_sha256"),
        "trajectory_report": aggregate.get("trajectory", {}).get(
            "comparison_report_sha256"),
        "quality_result": aggregate.get("analytic_invariants", {}).get(
            "result_sha256"),
        "quality_report": aggregate.get("analytic_invariants", {}).get(
            "report_sha256"),
        "lifecycle_result": aggregate.get("lifecycle_invariants", {}).get(
            "result_sha256"),
        "lifecycle_report": aggregate.get("lifecycle_invariants", {}).get(
            "report_sha256"),
    }
    for name, path in paths.items():
        expected = fields[name]
        actual = file_sha256(path)
        if not _sha256(expected) or actual != expected:
            raise ValueError(f"{name.replace('_', ' ')} SHA-256 mismatch")


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and
                all(key in actual and _is_subset(value, actual[key])
                    for key, value in expected.items()))
    return expected == actual


def _verify_capture_results(aggregate: dict[str, Any],
                            paths: dict[str, Path]) -> None:
    loaded = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items() if name != "driver_binary"
    }
    provenance = loaded["provenance"]
    trajectory_result = loaded["trajectory_result"]
    trajectory_report = loaded["trajectory_report"]
    quality_result = loaded["quality_result"]
    quality_report = loaded["quality_report"]
    lifecycle_result = loaded["lifecycle_result"]
    lifecycle_report = loaded["lifecycle_report"]
    trajectory_provenance = trajectory_result.get("provenance")
    shared_provenance_fields = (
        "source_commit", "device_id", "device_name", "driver_version",
        "host_compiler", "vendor_id")
    if (not isinstance(trajectory_provenance, dict) or
            any(trajectory_provenance.get(key) != provenance.get(key)
                for key in shared_provenance_fields)):
        raise ValueError("trajectory provenance does not match capture")
    trajectory_contract = aggregate.get("contracts", {}).get("trajectory", {})
    if (trajectory_result.get("execution_status") != "completed" or
            trajectory_result.get("case_sha256") !=
            trajectory_contract.get("sha256")):
        raise ValueError("trajectory result did not complete its bound contract")
    aggregate_regression = aggregate.get("trajectory", {}).get(
        "reference_regression")
    for field in ("passed", "failed", "unsupported"):
        if (not isinstance(aggregate_regression, dict) or
                trajectory_report.get(field) != aggregate_regression.get(field)):
            raise ValueError("trajectory report does not match physical aggregate")

    quality_contract = aggregate.get("contracts", {}).get("quality", {})
    aggregate_quality = aggregate.get("analytic_invariants", {})
    if (quality_result.get("execution_status") != "completed" or
            quality_result.get("profile_sha256") != quality_contract.get("sha256") or
            quality_report.get("profile_sha256") != quality_contract.get("sha256") or
            quality_report.get("passed") is not True or
            quality_report.get("failure_count") != 0 or
            quality_report.get("sample_count") !=
            aggregate_quality.get("sample_count")):
        raise ValueError("quality result does not prove the aggregate claim")
    report_observations = quality_report.get("observations")
    aggregate_observations = aggregate_quality.get("observations")
    if not _is_subset(aggregate_observations, report_observations):
        raise ValueError("quality observations do not match physical aggregate")

    lifecycle_contract = aggregate.get("contracts", {}).get("lifecycle", {})
    aggregate_lifecycle = aggregate.get("lifecycle_invariants", {})
    aggregate_case_count = aggregate_lifecycle.get(
        "profile_case_count", aggregate_lifecycle.get("case_count"))
    aggregate_checkpoint_count = aggregate_lifecycle.get(
        "profile_checkpoint_count", aggregate_lifecycle.get("checkpoint_count"))
    if (lifecycle_result.get("execution_status") != "completed" or
            lifecycle_result.get("profile_sha256") !=
            lifecycle_contract.get("sha256") or
            lifecycle_report.get("profile_sha256") !=
            lifecycle_contract.get("sha256") or
            lifecycle_report.get("passed") is not True or
            lifecycle_report.get("failure_count") != 0 or
            lifecycle_report.get("case_count") !=
            aggregate_case_count or
            lifecycle_report.get("checkpoint_count") !=
            aggregate_checkpoint_count):
        raise ValueError("lifecycle result does not prove the aggregate claim")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_profile = (Path(__file__).with_name("qualification") /
                       "capability-obligations-v1.json")
    parser.add_argument("--profile", type=Path, default=default_profile)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--actual", type=Path)
    source.add_argument("--from-evidence", type=Path)
    parser.add_argument("--artifact-path")
    parser.add_argument("--driver-binary", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--trajectory-result", type=Path)
    parser.add_argument("--trajectory-report", type=Path)
    parser.add_argument("--quality-result", type=Path)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--lifecycle-result", type=Path)
    parser.add_argument("--lifecycle-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    try:
        profile = json.loads(arguments.profile.read_text(encoding="utf-8"))
        digest = profile_sha256(arguments.profile)
        if arguments.actual is not None:
            actual = json.loads(arguments.actual.read_text(encoding="utf-8"))
        else:
            if not arguments.artifact_path or arguments.output is None:
                raise ValueError(
                    "--from-evidence requires --artifact-path and --output")
            artifact_file = Path(arguments.artifact_path)
            if not artifact_file.is_absolute():
                artifact_file = Path.cwd() / artifact_file
            if artifact_file.resolve() != arguments.from_evidence.resolve():
                raise ValueError(
                    "--artifact-path must identify the --from-evidence file")
            aggregate = json.loads(
                arguments.from_evidence.read_text(encoding="utf-8"))
            paths = _capture_paths(arguments)
            _verify_capture_hashes(aggregate, paths)
            _verify_capture_results(aggregate, paths)
            provenance = json.loads(paths["provenance"].read_text(
                encoding="utf-8"))
            capabilities = json.loads(paths["capabilities"].read_text(
                encoding="utf-8"))
            actual = derive_result(
                profile, aggregate, digest, arguments.artifact_path,
                file_sha256(arguments.from_evidence), provenance, capabilities)
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                json.dumps(actual, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
        report = evaluate(profile, actual, digest)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"qualification: {error}")
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text, encoding="utf-8")
    print(f"Qualification: {report['qualified_count']}/"
          f"{report['capability_count']} capabilities, "
          f"{report['failure_count']} failures")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
