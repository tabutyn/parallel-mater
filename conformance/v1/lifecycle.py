#!/usr/bin/env python3
"""Evaluate backend-neutral capacity and interaction-lifecycle results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "parallel-mater-conformance-lifecycle-profile/v1"
RESULT_SCHEMA = "parallel-mater-conformance-lifecycle-result/v1"
REPORT_SCHEMA = "parallel-mater-conformance-lifecycle-report/v1"
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class Failure:
    path: str
    message: str
    actual: Any = None
    expected: Any = None


def profile_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _name_set(value: Any, path: str, failures: list[Failure]) -> set[str] | None:
    if (not isinstance(value, list) or
            not all(isinstance(item, str) and item for item in value) or
            len(set(value)) != len(value)):
        failures.append(Failure(path, "expected unique nonempty string names", value))
        return None
    return set(value)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _resource_snapshots(observation: dict[str, Any], path: str,
                        expected_resources: set[str],
                        failures: list[Failure]) -> dict[str, dict[str, Any]]:
    resources = _index(observation.get("resources"), path + ".resources", failures)
    actual_resources = set(resources)
    if actual_resources != expected_resources:
        failures.append(Failure(path + ".resources",
                                "tracked resource set mismatch",
                                sorted(actual_resources),
                                sorted(expected_resources)))
    for identity, resource in resources.items():
        resource_path = f"{path}.resources[{identity}]"
        if not isinstance(resource.get("active"), bool):
            failures.append(Failure(resource_path + ".active",
                                    "expected a boolean", resource.get("active")))
        generation = resource.get("generation")
        if (not isinstance(generation, int) or isinstance(generation, bool) or
                generation < 0):
            failures.append(Failure(resource_path + ".generation",
                                    "expected a nonnegative integer", generation))
        if not _valid_hash(resource.get("state_sha256")):
            failures.append(Failure(resource_path + ".state_sha256",
                                    "expected a lowercase SHA-256",
                                    resource.get("state_sha256")))
    return resources


def _compare_relation(current: Any, reference: Any, equal: bool, path: str,
                      label: str, failures: list[Failure]) -> None:
    if (current == reference) != equal:
        relation = "match" if equal else "differ from"
        failures.append(Failure(path, f"{label} must {relation} referenced checkpoint",
                                current, reference))


def _check_case(definition: dict[str, Any], actual: dict[str, Any], path: str,
                failures: list[Failure]) -> int:
    if actual.get("execution_status") != "completed":
        failures.append(Failure(path + ".execution_status",
                                "lifecycle execution must be completed",
                                actual.get("execution_status"), "completed"))

    for field in ("families", "unrelated_resources"):
        expected = _name_set(definition.get(field, []),
                             f"$.profile.cases[{definition.get('id')}].{field}",
                             failures)
        observed = _name_set(actual.get(field), path + "." + field, failures)
        if expected is not None and observed is not None and expected != observed:
            failures.append(Failure(path + "." + field, field + " mismatch",
                                    sorted(observed), sorted(expected)))

    tracked = _name_set(definition.get("tracked_resources"),
                        f"$.profile.cases[{definition.get('id')}].tracked_resources",
                        failures)
    if tracked is None:
        tracked = set()
    expected_checkpoints = _index(
        definition.get("checkpoints"),
        f"$.profile.cases[{definition.get('id')}].checkpoints", failures)
    observations = _index(actual.get("observations"), path + ".observations", failures)
    if set(observations) != set(expected_checkpoints):
        for missing in sorted(set(expected_checkpoints) - set(observations)):
            failures.append(Failure(path + ".observations",
                                    "missing required checkpoint", missing))
        for extra in sorted(set(observations) - set(expected_checkpoints)):
            failures.append(Failure(path + ".observations",
                                    "unexpected checkpoint", extra))

    snapshots: dict[str, dict[str, dict[str, Any]]] = {}
    graph_hashes: dict[str, str] = {}
    for identity, expected in expected_checkpoints.items():
        observation = observations.get(identity)
        if observation is None:
            continue
        observation_path = f"{path}.observations[{identity}]"
        expected_outcome = expected.get("expected_outcome")
        if observation.get("outcome") != expected_outcome:
            failures.append(Failure(observation_path + ".outcome",
                                    "operation outcome mismatch",
                                    observation.get("outcome"), expected_outcome))
        snapshots[identity] = _resource_snapshots(
            observation, observation_path, tracked, failures)
        if definition.get("track_graph"):
            graph_hash = observation.get("active_graph_sha256")
            if not _valid_hash(graph_hash):
                failures.append(Failure(observation_path + ".active_graph_sha256",
                                        "expected a lowercase SHA-256", graph_hash))
            else:
                graph_hashes[identity] = graph_hash

    for identity, expected in expected_checkpoints.items():
        if identity not in snapshots:
            continue
        observation_path = f"{path}.observations[{identity}]"
        for resource_id, rules in expected.get("resources", {}).items():
            current = snapshots[identity].get(resource_id)
            if current is None:
                continue
            resource_path = f"{observation_path}.resources[{resource_id}]"
            if "active" in rules and current.get("active") != rules["active"]:
                failures.append(Failure(resource_path + ".active",
                                        "active state mismatch", current.get("active"),
                                        rules["active"]))
            for rule, field, equal in (
                    ("generation_same_as", "generation", True),
                    ("generation_different_from", "generation", False),
                    ("state_same_as", "state_sha256", True),
                    ("state_different_from", "state_sha256", False)):
                reference_id = rules.get(rule)
                if reference_id is None:
                    continue
                reference = snapshots.get(reference_id, {}).get(resource_id)
                if reference is None:
                    failures.append(Failure(resource_path + "." + field,
                                            "referenced checkpoint/resource is missing",
                                            reference_id))
                    continue
                _compare_relation(current.get(field), reference.get(field), equal,
                                  resource_path + "." + field, field, failures)
        for rule, equal in (("graph_same_as", True),
                            ("graph_different_from", False)):
            reference_id = expected.get(rule)
            if reference_id is None:
                continue
            if identity not in graph_hashes or reference_id not in graph_hashes:
                failures.append(Failure(observation_path + ".active_graph_sha256",
                                        "referenced graph checkpoint is missing",
                                        reference_id))
                continue
            _compare_relation(graph_hashes[identity], graph_hashes[reference_id], equal,
                              observation_path + ".active_graph_sha256",
                              "active graph", failures)
    return len(observations)


def evaluate(profile: dict[str, Any], result: dict[str, Any],
             expected_hash: str) -> dict[str, Any]:
    failures: list[Failure] = []
    if profile.get("schema") != PROFILE_SCHEMA:
        failures.append(Failure("$.profile.schema", "unsupported profile schema",
                                profile.get("schema"), PROFILE_SCHEMA))
    if result.get("schema") != RESULT_SCHEMA:
        failures.append(Failure("$.schema", "unsupported result schema",
                                result.get("schema"), RESULT_SCHEMA))
    if result.get("profile_id") != profile.get("id"):
        failures.append(Failure("$.profile_id", "profile identity mismatch",
                                result.get("profile_id"), profile.get("id")))
    if result.get("profile_sha256") != expected_hash:
        failures.append(Failure("$.profile_sha256", "profile hash mismatch",
                                result.get("profile_sha256"), expected_hash))
    if result.get("execution_status") != "completed":
        failures.append(Failure("$.execution_status",
                                "lifecycle suite must be completed",
                                result.get("execution_status"), "completed"))
    driver = result.get("driver")
    if (not isinstance(driver, dict) or
            not isinstance(driver.get("name"), str) or not driver["name"].strip() or
            not isinstance(driver.get("device"), str) or not driver["device"].strip()):
        failures.append(Failure("$.driver", "driver name and device are required", driver))

    definitions = _index(profile.get("cases"), "$.profile.cases", failures)
    actual_cases = _index(result.get("cases"), "$.cases", failures)
    for missing in sorted(set(definitions) - set(actual_cases)):
        failures.append(Failure("$.cases", "missing required lifecycle case", missing))
    for extra in sorted(set(actual_cases) - set(definitions)):
        failures.append(Failure("$.cases", "unexpected lifecycle case", extra))

    checkpoint_count = 0
    for identity, definition in definitions.items():
        actual = actual_cases.get(identity)
        if actual is not None:
            checkpoint_count += _check_case(
                definition, actual, f"$.cases[{identity}]", failures)

    return {
        "schema": REPORT_SCHEMA,
        "scope": "lifecycle_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "case_count": len(actual_cases),
        "checkpoint_count": checkpoint_count,
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_profile = (Path(__file__).with_name("lifecycle") /
                       "adverse-capacity-three-family-v1.json")
    parser.add_argument("--profile", type=Path, default=default_profile)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    try:
        profile = json.loads(arguments.profile.read_text(encoding="utf-8"))
        actual = json.loads(arguments.actual.read_text(encoding="utf-8"))
        report = evaluate(profile, actual, profile_sha256(arguments.profile))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"lifecycle: {error}")
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text, encoding="utf-8")
    print(f"Lifecycle: {report['case_count']} cases, "
          f"{report['checkpoint_count']} checkpoints, "
          f"{report['failure_count']} failures")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
