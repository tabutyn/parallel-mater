#!/usr/bin/env python3
"""Evaluate backend-neutral conservation and stability results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "parallel-mater-conformance-quality-profile/v1"
RESULT_SCHEMA = "parallel-mater-conformance-quality-result/v1"
REPORT_SCHEMA = "parallel-mater-conformance-quality-report/v1"


@dataclass
class Failure:
    path: str
    message: str
    actual: Any = None
    limit: Any = None


def profile_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _scalar(value: Any, path: str, failures: list[Failure]) -> float | None:
    if not _number(value):
        failures.append(Failure(path, "expected a finite number", value))
        return None
    return float(value)


def _vector(value: Any, path: str,
            failures: list[Failure]) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3 or not all(
            _number(component) for component in value):
        failures.append(Failure(path, "expected three finite components", value))
        return None
    return float(value[0]), float(value[1]), float(value[2])


def _subtract(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _sum(vectors: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(sum(value[axis] for value in vectors) for axis in range(3))  # type: ignore[return-value]


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _limit(profile: dict[str, Any], name: str, scale: float) -> float:
    tolerance = profile["tolerances"][name]
    return max(float(tolerance["absolute"]),
               float(tolerance["relative"]) * scale)


def _check_system_exchange(profile: dict[str, Any], definition: dict[str, Any],
                           sample: dict[str, Any], checks: set[str], path: str,
                           failures: list[Failure]) -> None:
    systems = sample.get("systems")
    if not isinstance(systems, list) or len(systems) != 2:
        failures.append(Failure(path + ".systems",
                                "expected two endpoint aggregates", systems))
        return

    masses: list[float] = []
    linear_changes: list[tuple[float, float, float]] = []
    angular_changes: list[tuple[float, float, float]] = []
    energy_before = energy_after = external_work = 0.0
    valid_energy = True
    for index, system in enumerate(systems):
        system_path = f"{path}.systems[{index}]"
        if not isinstance(system, dict):
            failures.append(Failure(system_path, "expected an object", system))
            return
        mass = _scalar(system.get("mass"), system_path + ".mass", failures)
        if mass is not None:
            if mass <= 0.0:
                failures.append(Failure(system_path + ".mass",
                                        "mass must be positive", mass, "> 0"))
            else:
                masses.append(mass)

        if "linear_momentum" in checks:
            before = _vector(system.get("linear_momentum_before"),
                             system_path + ".linear_momentum_before", failures)
            after = _vector(system.get("linear_momentum_after"),
                            system_path + ".linear_momentum_after", failures)
            external = _vector(system.get("external_impulse"),
                               system_path + ".external_impulse", failures)
            if before is not None and after is not None and external is not None:
                linear_changes.append(_subtract(_subtract(after, before), external))
                if definition.get("external_terms") == "zero" and _norm(external) != 0.0:
                    failures.append(Failure(system_path + ".external_impulse",
                                            "closed fixture requires zero external impulse",
                                            list(external), [0.0, 0.0, 0.0]))

        if "angular_momentum" in checks:
            before = _vector(system.get("angular_momentum_before"),
                             system_path + ".angular_momentum_before", failures)
            after = _vector(system.get("angular_momentum_after"),
                            system_path + ".angular_momentum_after", failures)
            external = _vector(system.get("external_angular_impulse"),
                               system_path + ".external_angular_impulse", failures)
            if before is not None and after is not None and external is not None:
                angular_changes.append(_subtract(_subtract(after, before), external))
                if definition.get("external_terms") == "zero" and _norm(external) != 0.0:
                    failures.append(Failure(
                        system_path + ".external_angular_impulse",
                        "closed fixture requires zero external angular impulse",
                        list(external), [0.0, 0.0, 0.0]))

        if "dissipative_energy" in checks:
            before = _scalar(system.get("kinetic_energy_before"),
                             system_path + ".kinetic_energy_before", failures)
            after = _scalar(system.get("kinetic_energy_after"),
                            system_path + ".kinetic_energy_after", failures)
            work = _scalar(system.get("external_work"),
                           system_path + ".external_work", failures)
            if before is None or after is None or work is None:
                valid_energy = False
            elif before < 0.0 or after < 0.0:
                failures.append(Failure(system_path,
                                        "kinetic energy must be nonnegative"))
                valid_energy = False
            else:
                energy_before += before
                energy_after += after
                external_work += work
                if definition.get("external_terms") == "zero" and work != 0.0:
                    failures.append(Failure(system_path + ".external_work",
                                            "closed fixture requires zero external work",
                                            work, 0.0))

    declared_ratio = _scalar(sample.get("mass_ratio"), path + ".mass_ratio",
                             failures)
    if declared_ratio is not None and len(masses) == 2:
        actual_ratio = max(masses) / min(masses)
        if not math.isclose(actual_ratio, declared_ratio,
                            rel_tol=1.0e-9, abs_tol=1.0e-12):
            failures.append(Failure(path + ".mass_ratio",
                                    "declared ratio does not match endpoint masses",
                                    declared_ratio, actual_ratio))

    if len(linear_changes) == 2:
        activity = sum(_norm(value) for value in linear_changes)
        minimum = float(profile["tolerances"]["exchange_activity_minimum"])
        if activity < minimum:
            failures.append(Failure(path + ".systems",
                                    "linear exchange is too small to be a conservation fixture",
                                    activity, minimum))
        residual = _norm(_sum(linear_changes))
        limit = _limit(profile, "linear_momentum", activity)
        if residual > limit:
            failures.append(Failure(path + ".systems",
                                    "linear momentum residual exceeds profile",
                                    residual, limit))

    if len(angular_changes) == 2:
        activity = sum(_norm(value) for value in angular_changes)
        residual = _norm(_sum(angular_changes))
        limit = _limit(profile, "angular_momentum", activity)
        if residual > limit:
            failures.append(Failure(path + ".systems",
                                    "angular momentum residual exceeds profile",
                                    residual, limit))

    if "dissipative_energy" in checks and valid_energy:
        growth = energy_after - energy_before - external_work
        scale = max(abs(energy_before), abs(energy_after), abs(external_work))
        limit = _limit(profile, "dissipative_energy", scale)
        if growth > limit:
            failures.append(Failure(path + ".systems",
                                    "dissipative mode increased accounted kinetic energy",
                                    growth, limit))


def _state_map(value: Any, path: str,
               failures: list[Failure]) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, list):
        failures.append(Failure(path, "expected a particle array", value))
        return None
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("id"), (str, int)):
            failures.append(Failure(item_path, "particle needs a string/integer id", item))
            continue
        identity = str(item["id"])
        if identity in result:
            failures.append(Failure(item_path + ".id", "duplicate particle id",
                                    item["id"]))
            continue
        result[identity] = item
    return result


def _check_split_combined(profile: dict[str, Any], sample: dict[str, Any],
                          path: str, failures: list[Failure]) -> None:
    comparison = sample.get("split_combined")
    compare_path = path + ".split_combined"
    if not isinstance(comparison, dict):
        failures.append(Failure(compare_path, "expected split/combined states",
                                comparison))
        return
    support = _scalar(comparison.get("support_radius_m"),
                      compare_path + ".support_radius_m", failures)
    timestep = _scalar(sample.get("timestep_seconds"),
                       path + ".timestep_seconds", failures)
    split = _state_map(comparison.get("split"), compare_path + ".split", failures)
    combined = _state_map(comparison.get("combined"), compare_path + ".combined",
                          failures)
    if support is None or timestep is None or split is None or combined is None:
        return
    if support <= 0.0 or timestep <= 0.0:
        failures.append(Failure(compare_path,
                                "support radius and timestep must be positive"))
        return
    if set(split) != set(combined):
        failures.append(Failure(compare_path,
                                "split and combined particle identities differ",
                                sorted(split), sorted(combined)))
        return

    tolerance = profile["tolerances"]["split_combined"]
    scales = {"position": support, "velocity": support / timestep,
              "acceleration": support / (timestep * timestep)}
    for identity in sorted(split):
        for field, scale in scales.items():
            left = _vector(split[identity].get(field),
                           f"{compare_path}.split[{identity}].{field}", failures)
            right = _vector(combined[identity].get(field),
                            f"{compare_path}.combined[{identity}].{field}", failures)
            if left is None or right is None:
                continue
            error = max(abs(a - b) for a, b in zip(left, right)) / scale
            limit = float(tolerance[field + "_normalized"])
            if error > limit:
                failures.append(Failure(f"{compare_path}.{identity}.{field}",
                                        "split/combined normalized error exceeds profile",
                                        error, limit))
        left_temperature = _scalar(split[identity].get("temperature_celsius"),
                                   f"{compare_path}.split[{identity}].temperature_celsius",
                                   failures)
        right_temperature = _scalar(combined[identity].get("temperature_celsius"),
                                    f"{compare_path}.combined[{identity}].temperature_celsius",
                                    failures)
        if (left_temperature is not None and right_temperature is not None and
                left_temperature != right_temperature):
            failures.append(Failure(f"{compare_path}.{identity}.temperature_celsius",
                                    "split/combined temperature must be exact",
                                    left_temperature, right_temperature))
        left_foam = _scalar(split[identity].get("foam"),
                            f"{compare_path}.split[{identity}].foam", failures)
        right_foam = _scalar(combined[identity].get("foam"),
                             f"{compare_path}.combined[{identity}].foam", failures)
        if left_foam is not None and right_foam is not None:
            error = abs(left_foam - right_foam)
            limit = float(tolerance["foam_absolute"])
            if error > limit:
                failures.append(Failure(f"{compare_path}.{identity}.foam",
                                        "split/combined foam error exceeds profile",
                                        error, limit))


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
                                "quality execution must be completed",
                                result.get("execution_status"), "completed"))
    driver = result.get("driver")
    if (not isinstance(driver, dict) or
            not isinstance(driver.get("name"), str) or not driver["name"].strip() or
            not isinstance(driver.get("device"), str) or not driver["device"].strip()):
        failures.append(Failure("$.driver",
                                "driver name and device are required", driver))

    definitions = {fixture["id"]: fixture for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        if fixture["id"] in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", fixture["id"]))
            continue
        actual_by_id[fixture["id"]] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    timesteps = [float(value) for value in profile.get("sweep", {}).get(
        "timesteps_seconds", [])]
    ratios = [float(value) for value in profile.get("sweep", {}).get(
        "endpoint_mass_ratios", [])]
    expected_matrix = {(timestep, ratio) for timestep in timesteps for ratio in ratios}
    sample_count = 0
    for identity, definition in definitions.items():
        fixture = actual_by_id.get(identity)
        if fixture is None:
            continue
        samples = fixture.get("samples")
        fixture_path = f"$.fixtures[{identity}]"
        if not isinstance(samples, list):
            failures.append(Failure(fixture_path + ".samples",
                                    "expected a sample array", samples))
            continue
        indexed: dict[tuple[float, float], tuple[dict[str, Any], str]] = {}
        for index, sample in enumerate(samples):
            sample_path = f"{fixture_path}.samples[{index}]"
            if not isinstance(sample, dict):
                failures.append(Failure(sample_path, "expected an object", sample))
                continue
            timestep = _scalar(sample.get("timestep_seconds"),
                               sample_path + ".timestep_seconds", failures)
            ratio = _scalar(sample.get("mass_ratio"),
                            sample_path + ".mass_ratio", failures)
            if timestep is None or ratio is None:
                continue
            key = (timestep, ratio)
            if key in indexed:
                failures.append(Failure(sample_path, "duplicate sweep point", key))
                continue
            indexed[key] = sample, sample_path
        for missing in sorted(expected_matrix - set(indexed)):
            failures.append(Failure(fixture_path + ".samples",
                                    "missing timestep/mass-ratio point", list(missing)))
        for extra in sorted(set(indexed) - expected_matrix):
            failures.append(Failure(fixture_path + ".samples",
                                    "unexpected timestep/mass-ratio point", list(extra)))
        checks = set(definition.get("checks", []))
        for key in sorted(expected_matrix & set(indexed)):
            sample, sample_path = indexed[key]
            _check_system_exchange(profile, definition, sample, checks,
                                   sample_path, failures)
            if "split_combined" in checks:
                _check_split_combined(profile, sample, sample_path, failures)
            sample_count += 1

    report = {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_profile = Path(__file__).with_name("quality") / "core-conservation-v1.json"
    parser.add_argument("--profile", type=Path, default=default_profile)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    try:
        profile = json.loads(arguments.profile.read_text(encoding="utf-8"))
        actual = json.loads(arguments.actual.read_text(encoding="utf-8"))
        report = evaluate(profile, actual, profile_sha256(arguments.profile))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"quality: {error}")
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text, encoding="utf-8")
    print(f"Quality: {report['sample_count']} samples, "
          f"{report['failure_count']} failures")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
