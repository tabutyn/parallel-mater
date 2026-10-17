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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_profile = (Path(__file__).with_name("qualification") /
                       "capability-obligations-v1.json")
    parser.add_argument("--profile", type=Path, default=default_profile)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    try:
        profile = json.loads(arguments.profile.read_text(encoding="utf-8"))
        actual = json.loads(arguments.actual.read_text(encoding="utf-8"))
        report = evaluate(profile, actual, profile_sha256(arguments.profile))
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
