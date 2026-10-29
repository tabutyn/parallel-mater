#!/usr/bin/env python3
"""Load and validate the shared public fixed-tick scenario definitions.

The definitions are data, not backend code.  A device runner selects a runtime
variant and receives the same scene, events, measurements and tick outcomes as
every other runner.  This module deliberately does not alter the frozen oracle
suite; F8-61 owns the later candidate refreeze.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FIXTURES_PATH = ROOT / "conformance/v1/realtime/scenario-fixtures-v1.json"
SUITE_PATH = ROOT / "conformance/v1/realtime/oracle-suite-v1.json"
PROFILE_PATH = ROOT / "conformance/v1/qualification/realtime-profile-v1.json"
INVENTORY_PATH = (
    ROOT / "conformance/v1/qualification/kernel-device-obligations-v1.json"
)
ORACLE_PATH = ROOT / "conformance/v1/realtime_oracle.py"
SCHEMA = "parallel-mater-realtime-scenario-fixtures/v1"


def _load_oracle():
    spec = importlib.util.spec_from_file_location(
        "pm_realtime_scenario_oracle", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen realtime oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORACLE = _load_oracle()


@dataclass(frozen=True)
class ContractIssue:
    path: str
    message: str
    actual: Any = None
    expected: Any = None


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def fixture_sha256(path: Path = FIXTURES_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_identity(fixtures: Mapping[str, Any]) -> dict[str, str]:
    return {"id": str(fixtures.get("id")), "sha256": fixture_sha256()}


def _issue(path: str, message: str, actual: Any = None,
           expected: Any = None) -> ContractIssue:
    return ContractIssue(path, message, actual, expected)


def _measurement_assertions(
    fixtures: Mapping[str, Any], domains: list[str]
) -> list[str]:
    algorithms = fixtures.get("measurement_algorithms", {})
    return sorted(
        item.get("assertion")
        for domain in domains
        for item in algorithms.get(domain, [])
        if isinstance(item, dict) and isinstance(item.get("assertion"), str)
    )


def tick_outcome(definition: Mapping[str, Any], tick: int) -> dict[str, Any]:
    outcomes = definition["tick_outcomes"]
    result: dict[str, Any] = {
        "tick": tick,
        "status": outcomes["default"],
        "result": "PM_SUCCESS",
        "queue_submissions": 1,
        "spatial_index_publications": 1,
        "time_advances": 1,
        "intra_tick_host_readbacks": 0,
    }
    for override in outcomes["overrides"]:
        if override["tick"] == tick:
            result.update(override)
            break
    return result


def execution_plan(
    runtime_variant: str,
    fixtures: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the backend-neutral runner plan plus one runtime selection."""

    contract = load_json(FIXTURES_PATH) if fixtures is None else dict(fixtures)
    variants = contract["execution"]["runtime_variants"]
    if runtime_variant not in variants:
        raise ValueError(
            f"unknown runtime variant {runtime_variant!r}; expected {variants}")
    return {
        "schema": "parallel-mater-realtime-scenario-plan/v1",
        "fixture_contract": fixture_identity(contract),
        "runtime_variant": runtime_variant,
        "execution": copy.deepcopy(contract["execution"]),
        "semantic_assertions": copy.deepcopy(contract["semantic_assertions"]),
        "measurement_algorithms": copy.deepcopy(
            contract["measurement_algorithms"]),
        "scenarios": copy.deepcopy(contract["scenarios"]),
    }


def validate_contract(
    fixtures: Mapping[str, Any],
    suite: Mapping[str, Any],
    profile: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    if fixtures.get("schema") != SCHEMA:
        issues.append(_issue("$.schema", "unexpected fixture schema",
                             fixtures.get("schema"), SCHEMA))

    execution = fixtures.get("execution")
    if not isinstance(execution, dict):
        issues.append(_issue("$.execution", "execution contract is required"))
        return issues
    expected_dt = (
        profile["profile"]["timestep_seconds"]["numerator"]
        / profile["profile"]["timestep_seconds"]["denominator"]
    )
    exact_execution = {
        "timestep_seconds": expected_dt,
        "substeps": 1,
        "simulation_instances": 1,
        "spatial_publications_per_successful_tick": 1,
        "failed_tick_publications": 0,
        "failed_tick_time_advances": 0,
        "intra_tick_host_readbacks": 0,
        "readback": "after-completion-public-api",
    }
    for name, expected in exact_execution.items():
        if execution.get(name) != expected:
            issues.append(_issue(
                f"$.execution.{name}", "execution value is not exact",
                execution.get(name), expected))
    if execution.get("public_api_calls") != suite["public_api"]["operation_sequence"]:
        issues.append(_issue(
            "$.execution.public_api_calls",
            "public call sequence differs from the frozen suite",
            execution.get("public_api_calls"),
            suite["public_api"]["operation_sequence"]))
    expected_variants = sorted(
        item["id"] for item in inventory["compiled_variants"]
        if item["kind"] in {
            "runtime", "runtime-subset", "runtime-with-current-stubs"
        }
    )
    if sorted(execution.get("runtime_variants", [])) != expected_variants:
        issues.append(_issue(
            "$.execution.runtime_variants",
            "runtime variants differ from the shared inventory",
            sorted(execution.get("runtime_variants", [])), expected_variants))

    semantic = fixtures.get("semantic_assertions")
    expected_tags = {
        tag for item in suite.get("scenario_fixtures", [])
        for tag in item.get("tags", [])
    }
    if not isinstance(semantic, dict) or set(semantic) != expected_tags:
        issues.append(_issue(
            "$.semantic_assertions",
            "semantic assertion set does not cover the frozen tags exactly",
            sorted(semantic) if isinstance(semantic, dict) else semantic,
            sorted(expected_tags)))
    else:
        for name, value in semantic.items():
            if (not isinstance(value, dict)
                    or not isinstance(value.get("inputs"), list)
                    or not value["inputs"]
                    or not isinstance(value.get("algorithm"), str)
                    or not value["algorithm"].strip()):
                issues.append(_issue(
                    f"$.semantic_assertions.{name}",
                    "semantic assertion needs public inputs and an algorithm"))

    algorithms = fixtures.get("measurement_algorithms")
    profile_domains = {
        item["domain"] for item in profile["domain_quality"]
    }
    if not isinstance(algorithms, dict) or set(algorithms) != profile_domains:
        issues.append(_issue(
            "$.measurement_algorithms", "measurement domains are not exact",
            sorted(algorithms) if isinstance(algorithms, dict) else algorithms,
            sorted(profile_domains)))
    else:
        for domain, values in algorithms.items():
            path = f"$.measurement_algorithms.{domain}"
            actual = _measurement_assertions(fixtures, [domain])
            case = {"kind": "scenario", "domains": [domain]}
            expected = sorted(
                spec.assertion
                for spec in ORACLE.case_limit_specs(case, profile)
            )
            if actual != expected:
                issues.append(_issue(
                    path, "named measurement set is not the profile set",
                    actual, expected))
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                if (not isinstance(value, dict)
                        or not isinstance(value.get("inputs"), list)
                        or not value.get("inputs")
                        or not isinstance(value.get("algorithm"), str)
                        or not value["algorithm"].strip()):
                    issues.append(_issue(
                        f"{path}[{index}]",
                        "measurement needs public inputs and an algorithm"))

    suite_rows = suite.get("scenario_fixtures", [])
    definitions = fixtures.get("scenarios")
    if not isinstance(definitions, list):
        issues.append(_issue("$.scenarios", "scenario definitions are required"))
        return issues
    suite_by_id = {
        item["id"]: item for item in suite_rows if isinstance(item, dict)
    }
    fixture_by_id = {
        item.get("id"): item for item in definitions if isinstance(item, dict)
    }
    if len(fixture_by_id) != len(definitions) or set(fixture_by_id) != set(suite_by_id):
        issues.append(_issue(
            "$.scenarios", "scenario identities are not the frozen twelve rows",
            sorted(str(value) for value in fixture_by_id), sorted(suite_by_id)))

    for identity, frozen in suite_by_id.items():
        definition = fixture_by_id.get(identity)
        path = f"$.scenarios[{identity}]"
        if not isinstance(definition, dict):
            continue
        for name in ("domains", "ticks", "tags"):
            if definition.get(name) != frozen.get(name):
                issues.append(_issue(
                    f"{path}.{name}", "frozen scenario row changed",
                    definition.get(name), frozen.get(name)))
        fixture = definition.get("fixture")
        if (not isinstance(fixture, dict)
                or not isinstance(fixture.get("builder"), str)
                or not isinstance(fixture.get("initial_state"), str)
                or not isinstance(fixture.get("events"), list)):
            issues.append(_issue(
                path + ".fixture",
                "portable builder, deterministic state and events are required"))
            continue
        source = fixture.get("definition")
        digest = fixture.get("definition_sha256")
        if source is not None:
            source_path = ROOT / source
            if not source_path.is_file():
                issues.append(_issue(
                    path + ".fixture.definition", "source definition is absent",
                    source))
            elif (not isinstance(digest, str)
                  or hashlib.sha256(source_path.read_bytes()).hexdigest() != digest):
                issues.append(_issue(
                    path + ".fixture.definition_sha256",
                    "source definition digest changed", digest,
                    hashlib.sha256(source_path.read_bytes()).hexdigest()))
        for index, event in enumerate(fixture["events"]):
            if (not isinstance(event, dict)
                    or isinstance(event.get("tick"), bool)
                    or not isinstance(event.get("tick"), int)
                    or not 0 <= event["tick"] < frozen["ticks"]
                    or not isinstance(event.get("operation"), str)):
                issues.append(_issue(
                    f"{path}.fixture.events[{index}]",
                    "event needs a valid tick and public operation"))

        sets = definition.get("measurement_sets")
        if sets != frozen["domains"]:
            issues.append(_issue(
                path + ".measurement_sets",
                "measurement sets must follow the exact domain order",
                sets, frozen["domains"]))
        actual_measurements = _measurement_assertions(fixtures, sets or [])
        expected_measurements = sorted(
            spec.assertion for spec in ORACLE.case_limit_specs(
                {"kind": "scenario", **frozen}, profile)
        )
        if actual_measurements != expected_measurements:
            issues.append(_issue(
                path + ".measurements",
                "scenario named measurements are not exact",
                actual_measurements, expected_measurements))

        outcomes = definition.get("tick_outcomes")
        if (not isinstance(outcomes, dict)
                or outcomes.get("default") != "success"
                or not isinstance(outcomes.get("overrides"), list)):
            issues.append(_issue(
                path + ".tick_outcomes", "tick outcome contract is invalid"))
        else:
            seen: set[int] = set()
            for index, override in enumerate(outcomes["overrides"]):
                tick = override.get("tick") if isinstance(override, dict) else None
                if (isinstance(tick, bool) or not isinstance(tick, int)
                        or not 0 <= tick < frozen["ticks"] or tick in seen):
                    issues.append(_issue(
                        f"{path}.tick_outcomes.overrides[{index}]",
                        "override tick is invalid or duplicated", tick))
                    continue
                seen.add(tick)
                failure_fields = {
                    "status": "failure",
                    "queue_submissions": 1,
                    "spatial_index_updates": 0,
                    "spatial_index_publications": 0,
                    "time_advances": 0,
                }
                for name, expected in failure_fields.items():
                    if override.get(name) != expected:
                        issues.append(_issue(
                            f"{path}.tick_outcomes.overrides[{index}].{name}",
                            "failed tick accounting is not atomic",
                            override.get(name), expected))
        diagnostics = definition.get("failure_diagnostics")
        if not isinstance(diagnostics, list) or not diagnostics:
            issues.append(_issue(
                path + ".failure_diagnostics",
                "named public failure diagnostics are required"))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or print a shared fixed-tick scenario plan")
    parser.add_argument("--runtime-variant")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    fixtures = load_json(FIXTURES_PATH)
    issues = validate_contract(
        fixtures, load_json(SUITE_PATH), load_json(PROFILE_PATH),
        load_json(INVENTORY_PATH))
    if issues:
        for issue in issues:
            print(f"FAIL {issue.path}: {issue.message}", file=sys.stderr)
        return 1
    if arguments.runtime_variant:
        print(json.dumps(execution_plan(arguments.runtime_variant, fixtures),
                         indent=2, sort_keys=True))
    elif arguments.check:
        print(f"PASS {len(fixtures['scenarios'])} shared scenario fixtures")
    else:
        parser.error("choose --check or --runtime-variant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
