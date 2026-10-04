#!/usr/bin/env python3
"""Compare backend-neutral Parallel Mater conformance results."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any


IGNORED_KEYS = {"backend", "diagnostics", "provenance"}
EXACT_KEYS = {
    "active_bonds",
    "broken",
    "case_id",
    "case_sha256",
    "count",
    "enabled",
    "frame",
    "id",
    "indices",
    "name",
    "schema",
    "stable_ids",
    "stable_particle_ids",
    "surface_source_indices",
    "topology",
    "type",
    "vertex_source_indices",
}


@dataclass
class Difference:
    path: str
    message: str
    expected: Any = None
    actual: Any = None

    def to_json(self) -> dict[str, Any]:
        return {
            "actual": self.actual,
            "expected": self.expected,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class Comparison:
    case_id: str
    differences: list[Difference] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.differences

    def fail(self, path: str, message: str, expected: Any = None,
             actual: Any = None) -> None:
        self.differences.append(Difference(path, message, expected, actual))


def load_json(path: pathlib.Path) -> Any:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}")

    with path.open("r", encoding="utf-8") as source:
        return json.load(source, parse_constant=invalid_constant)


def ensure_finite(value: Any, path: str, comparison: Comparison) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        comparison.fail(path, "NaN or infinity is forbidden", actual=value)
    elif isinstance(value, dict):
        for key, member in value.items():
            ensure_finite(member, f"{path}.{key}", comparison)
    elif isinstance(value, list):
        for index, member in enumerate(value):
            ensure_finite(member, f"{path}[{index}]", comparison)


def canonicalize_contacts(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: canonicalize_contacts(member)
                  for key, member in value.items()}
        for key in ("contacts", "rigid_contacts", "fluid_contacts"):
            if isinstance(result.get(key), list):
                result[key] = sorted(
                    result[key],
                    key=lambda member: json.dumps(
                        member, allow_nan=False, separators=(",", ":"),
                        sort_keys=True))
        return result
    if isinstance(value, list):
        return [canonicalize_contacts(member) for member in value]
    return value


def quaternion_error(expected: list[Any], actual: list[Any]) -> float:
    if len(expected) != 4 or len(actual) != 4:
        return math.inf
    expected_norm = math.sqrt(sum(float(value) ** 2 for value in expected))
    actual_norm = math.sqrt(sum(float(value) ** 2 for value in actual))
    if expected_norm == 0.0 or actual_norm == 0.0:
        return math.inf
    dot = sum(float(left) * float(right)
              for left, right in zip(expected, actual))
    dot = abs(dot / (expected_norm * actual_norm))
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


def tolerance_for(case: dict[str, Any], tolerance_class: str,
                  path: str) -> tuple[float, float]:
    values = case["tolerances"]["values"]
    leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
    if leaf == "orientation":
        selected = values["quaternion_angular"]
    elif case.get("world", {}).get("chaotic_envelope") and "momentum" in path:
        selected = values["chaotic_momentum"]
    elif case.get("world", {}).get("chaotic_envelope") and ".aggregate." in path:
        if ".momentum" in path:
            selected = values["chaotic_momentum"]
        elif ".kinetic_energy" in path:
            selected = values["chaotic_energy"]
        else:
            selected = values["chaotic_position"]
    elif case.get("world", {}).get("chaotic_envelope") and ".foam." in path:
        selected = values["chaotic_scalar"]
    else:
        profile = values.get(tolerance_class) or values[case["tolerances"]["profile"]]
        quantity = "velocity" if (
            "velocity" in leaf or "momentum" in leaf or
            "vorticity" in leaf) else "position"
        selected = profile[quantity]
    return float(selected["abs"]), float(selected["rel"])


def is_exact_path(path: str, expected: Any) -> bool:
    leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
    return (
        leaf in EXACT_KEYS or leaf.endswith("_count") or
        isinstance(expected, (bool, str, int)) or expected is None
    )


def compare_value(expected: Any, actual: Any, case: dict[str, Any],
                  comparison: Comparison, path: str = "$",
                  tolerance_class: str | None = None,
                  exact: bool = False) -> None:
    numeric = (isinstance(expected, (int, float)) and not isinstance(expected, bool)
               and isinstance(actual, (int, float)) and not isinstance(actual, bool))
    if type(expected) is not type(actual) and not numeric:
        comparison.fail(path, "type mismatch", type(expected).__name__,
                        type(actual).__name__)
        return
    if isinstance(expected, dict):
        ignored = set(IGNORED_KEYS)
        if case.get("world", {}).get("chaotic_envelope"):
            ignored.add("samples")
        expected_keys = set(expected) - ignored
        actual_keys = set(actual) - ignored
        for key in sorted(expected_keys - actual_keys):
            comparison.fail(f"{path}.{key}", "missing value",
                            expected[key], None)
        for key in sorted(actual_keys - expected_keys):
            comparison.fail(f"{path}.{key}", "unexpected value",
                            None, actual[key])
        child_class = str(expected.get("tolerance_class", tolerance_class or
                                       case["tolerances"]["profile"]))
        for key in sorted(expected_keys & actual_keys):
            compare_value(
                expected[key], actual[key], case, comparison,
                f"{path}.{key}", child_class,
                exact or key == "topology" or is_exact_path(key, expected[key]))
        return
    if isinstance(expected, list):
        if path.endswith(".orientation"):
            error = quaternion_error(expected, actual)
            absolute, relative = tolerance_for(
                case, tolerance_class or case["tolerances"]["profile"], path)
            limit = absolute + relative
            if error > limit:
                comparison.fail(path, f"quaternion angular error {error} > {limit}",
                                expected, actual)
            return
        if len(expected) != len(actual):
            comparison.fail(path, "length mismatch", len(expected), len(actual))
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_value(left, right, case, comparison, f"{path}[{index}]",
                          tolerance_class, exact)
        return
    if numeric and not exact:
        expected = float(expected)
        actual = float(actual)
        absolute, relative = tolerance_for(
            case, tolerance_class or case["tolerances"]["profile"], path)
        scale = max(abs(expected), abs(actual))
        limit = absolute + relative * scale
        error = abs(expected - actual)
        if error > limit:
            comparison.fail(path, f"absolute error {error} > {limit}",
                            expected, actual)
        return
    if expected != actual:
        comparison.fail(path, "exact value mismatch", expected, actual)


def compare_documents(case: dict[str, Any], expected: dict[str, Any],
                      actual: dict[str, Any]) -> Comparison:
    case_id = str(expected.get("case_id", actual.get("case_id", "unknown")))
    comparison = Comparison(case_id)
    ensure_finite(expected, "$expected", comparison)
    ensure_finite(actual, "$actual", comparison)
    expected = canonicalize_contacts(copy.deepcopy(expected))
    actual = canonicalize_contacts(copy.deepcopy(actual))
    compare_value(expected, actual, case, comparison)
    return comparison


def result_files(directory: pathlib.Path, selected: str) -> list[pathlib.Path]:
    if selected != "all":
        return [directory / f"{selected}.json"]
    return sorted(directory.glob("*.json"))


def compare_directories(cases: pathlib.Path, expected: pathlib.Path,
                        actual: pathlib.Path, selected: str) -> list[Comparison]:
    comparisons: list[Comparison] = []
    expected_files = result_files(expected, selected)
    if not expected_files:
        return [Comparison("all", [Difference("$", "no expected results")])]
    for expected_file in expected_files:
        case_id = expected_file.stem
        case_file = cases / f"{case_id}.json"
        actual_file = actual / expected_file.name
        comparison = Comparison(case_id)
        for path, label in ((case_file, "case input"),
                            (expected_file, "expected result"),
                            (actual_file, "actual result")):
            if not path.is_file():
                comparison.fail("$", f"missing {label}: {path}")
        if comparison.differences:
            comparisons.append(comparison)
            continue
        try:
            comparisons.append(compare_documents(
                load_json(case_file), load_json(expected_file),
                load_json(actual_file)))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            comparison.fail("$", f"invalid JSON: {error}")
            comparisons.append(comparison)
    return comparisons


def report_json(comparisons: list[Comparison]) -> dict[str, Any]:
    return {
        "cases": [
            {
                "differences": [difference.to_json()
                                for difference in comparison.differences],
                "id": comparison.case_id,
                "passed": comparison.passed,
            }
            for comparison in comparisons
        ],
        "failed": sum(not comparison.passed for comparison in comparisons),
        "passed": sum(comparison.passed for comparison in comparisons),
        "schema": "parallel-mater-conformance-report/v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=pathlib.Path, required=True)
    parser.add_argument("--expected", type=pathlib.Path, required=True)
    parser.add_argument("--actual", type=pathlib.Path, required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--report", type=pathlib.Path)
    arguments = parser.parse_args()
    comparisons = compare_directories(
        arguments.cases, arguments.expected, arguments.actual, arguments.case)
    report = report_json(comparisons)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    for comparison in comparisons:
        if comparison.passed:
            print(f"PASS {comparison.case_id}")
            continue
        print(f"FAIL {comparison.case_id}")
        for difference in comparison.differences:
            print(f"  {difference.path}: {difference.message}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
