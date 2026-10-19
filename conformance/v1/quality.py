#!/usr/bin/env python3
"""Evaluate backend-neutral conservation and stability results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "parallel-mater-conformance-quality-profile/v1"
RESULT_SCHEMA = "parallel-mater-conformance-quality-result/v1"
REPORT_SCHEMA = "parallel-mater-conformance-quality-report/v1"
CONFORMANCE_RESULT_SCHEMA = "parallel-mater-conformance-result/v1"


@dataclass
class Failure:
    path: str
    message: str
    actual: Any = None
    limit: Any = None


def profile_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".json", ".xml"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


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


def _quaternion(value: Any, path: str,
                failures: list[Failure]) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4 or not all(
            _number(component) for component in value):
        failures.append(Failure(path, "expected four finite components", value))
        return None
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def _subtract(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _sum(vectors: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(sum(value[axis] for value in vectors) for axis in range(3))  # type: ignore[return-value]


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _unit(value: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = _norm(value)
    if not length > 0.0:
        return None
    return tuple(component / length for component in value)  # type: ignore[return-value]


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


def _evaluate_fluid_self_pair(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure], *, mode: str) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture", "expected an object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected an object", actual))
        actual = {}
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances", "expected an object",
                                tolerances))
        tolerances = {}

    exact_fields = ("id", "particle_count", "substeps", "density_iterations",
                    "oracle")
    if mode == "pressure":
        exact_fields += ("density_estimator",)
    for field in exact_fields:
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting does not match profile",
                                    actual.get(field), definition.get(field)))

    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    scalar_fields = (
        "timestep_seconds", "particle_radius_m", "support_radius_m",
        "rest_density_kg_m3", "rest_particle_volume_m3",
        "pressure_stiffness", "kinematic_viscosity_m2_s", "normal_damping",
        "maximum_speed_m_s", "maximum_pair_acceleration_m_s2",
    )
    for field in scalar_fields:
        value = _scalar(actual.get(field), f"$.fixture.{field}", failures)
        expected = _scalar(definition.get(field),
                           f"$.profile.fixture.{field}", failures)
        if (value is not None and expected is not None and
                abs(value - expected) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    value, expected))
    gravity = _vector(actual.get("gravity_m_s2"),
                      "$.fixture.gravity_m_s2", failures)
    expected_gravity = _vector(definition.get("gravity_m_s2"),
                               "$.profile.fixture.gravity_m_s2", failures)
    if gravity is not None and expected_gravity is not None:
        error = max(abs(a - b) for a, b in zip(gravity, expected_gravity))
        if error > state_tolerance:
            failures.append(Failure("$.fixture.gravity_m_s2",
                                    "fixture setting differs from profile",
                                    list(gravity), list(expected_gravity)))

    def particle_map(value: Any, path: str) -> dict[int, dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a particle array", value))
            return {}
        mapped: dict[int, dict[str, Any]] = {}
        for index, particle in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(item_path, "expected an object", particle))
                continue
            identity = particle.get("stable_id")
            if (not isinstance(identity, int) or isinstance(identity, bool) or
                    identity in mapped):
                failures.append(Failure(item_path + ".stable_id",
                                        "stable ID must be a unique integer",
                                        identity))
                continue
            mapped[identity] = particle
        if set(mapped) != {0, 1}:
            failures.append(Failure(path,
                                    "fixture must preserve stable IDs 0 and 1",
                                    sorted(mapped), [0, 1]))
        return mapped

    expected_initial = particle_map(definition.get("initial_particles"),
                                    "$.profile.fixture.initial_particles")
    initial = particle_map(actual.get("initial_particles"),
                           "$.fixture.initial_particles")
    final = particle_map(actual.get("final_particles"),
                         "$.fixture.final_particles")

    initial_states: dict[int, tuple[tuple[float, float, float],
                                    tuple[float, float, float]]] = {}
    final_states: dict[int, tuple[tuple[float, float, float],
                                  tuple[float, float, float]]] = {}
    for identity in (0, 1):
        expected_particle = expected_initial.get(identity)
        initial_particle = initial.get(identity)
        if expected_particle is not None and initial_particle is not None:
            expected_position = _vector(
                expected_particle.get("position_m"),
                f"$.profile.fixture.initial_particles[{identity}].position_m",
                failures)
            expected_velocity = _vector(
                expected_particle.get("velocity_m_s"),
                f"$.profile.fixture.initial_particles[{identity}].velocity_m_s",
                failures)
            position = _vector(initial_particle.get("position_m"),
                               f"$.fixture.initial_particles[{identity}].position_m",
                               failures)
            velocity = _vector(initial_particle.get("velocity_m_s"),
                               f"$.fixture.initial_particles[{identity}].velocity_m_s",
                               failures)
            if position is not None and velocity is not None:
                initial_states[identity] = position, velocity
            for name, value, expected in (
                    ("position_m", position, expected_position),
                    ("velocity_m_s", velocity, expected_velocity)):
                if value is not None and expected is not None:
                    error = max(abs(a - b) for a, b in zip(value, expected))
                    if error > state_tolerance:
                        failures.append(Failure(
                            f"$.fixture.initial_particles[{identity}].{name}",
                            "initial public state differs from profile",
                            list(value), list(expected)))
        final_particle = final.get(identity)
        if final_particle is not None:
            position = _vector(final_particle.get("position_m"),
                               f"$.fixture.final_particles[{identity}].position_m",
                               failures)
            velocity = _vector(final_particle.get("velocity_m_s"),
                               f"$.fixture.final_particles[{identity}].velocity_m_s",
                               failures)
            if position is not None and velocity is not None:
                final_states[identity] = position, velocity

    observations: dict[str, Any] = {}
    if len(initial_states) == 2 and len(final_states) == 2:
        initial_separation = _subtract(initial_states[1][0], initial_states[0][0])
        final_separation = _subtract(final_states[1][0], final_states[0][0])
        separation = _norm(final_separation)
        observations["final_separation_m"] = separation
        if mode == "contact":
            radius = float(definition.get("particle_radius_m", 0.0))
            exclusion = (2.0 * radius * float(tolerances.get(
                "exclusion_distance_fraction", 0.0)))
            observations["minimum_separation_m"] = exclusion
            if separation < exclusion:
                failures.append(Failure(
                    "$.fixture.final_particles",
                    "fluid particles violate exclusion distance",
                    separation, exclusion))
        if sum(a * b for a, b in zip(initial_separation, final_separation)) <= 0.0:
            failures.append(Failure("$.fixture.final_particles",
                                    "fluid particles crossed stable-ID ordering"))

        mass = (float(definition.get("rest_density_kg_m3", 0.0)) *
                float(definition.get("rest_particle_volume_m3", 0.0)))
        if mode == "pressure":
            initial_spacing = _norm(initial_separation)
            rest_density = float(definition.get("rest_density_kg_m3", 0.0))
            if initial_spacing <= 0.0 or separation <= 0.0 or rest_density <= 0.0:
                failures.append(Failure(
                    "$.fixture.final_particles",
                    "density estimator requires positive spacing and rest density",
                    {
                        "initial_spacing_m": initial_spacing,
                        "final_spacing_m": separation,
                        "rest_density_kg_m3": rest_density,
                    }))
            else:
                initial_density = mass / (initial_spacing ** 3)
                final_density = mass / (separation ** 3)
                initial_error = abs(initial_density - rest_density) / rest_density
                final_error = abs(final_density - rest_density) / rest_density
                error_reduction = (1.0 - final_error / initial_error
                                   if initial_error > 0.0 else 0.0)
                observations.update({
                    "initial_density_kg_m3": initial_density,
                    "final_density_kg_m3": final_density,
                    "rest_density_kg_m3": rest_density,
                    "initial_density_relative_error": initial_error,
                    "final_density_relative_error": final_error,
                    "density_error_reduction_fraction": error_reduction,
                })
                initial_minimum = float(tolerances.get(
                    "initial_density_relative_error_minimum", 0.0))
                if initial_error < initial_minimum:
                    failures.append(Failure(
                        "$.fixture.initial_particles",
                        "initial density is not sufficiently displaced from rest",
                        initial_error, initial_minimum))
                final_maximum = float(tolerances.get(
                    "final_density_relative_error_maximum", 0.0))
                if final_error > final_maximum:
                    failures.append(Failure(
                        "$.fixture.final_particles",
                        "final density remains too far from rest",
                        final_error, final_maximum))
                reduction_minimum = float(tolerances.get(
                    "density_error_reduction_fraction_minimum", 0.0))
                if error_reduction < reduction_minimum:
                    failures.append(Failure(
                        "$.fixture.final_particles",
                        "pressure did not move density sufficiently toward rest",
                        error_reduction, reduction_minimum))
        if mode == "viscosity":
            initial_relative_velocity = _subtract(
                initial_states[1][1], initial_states[0][1])
            final_relative_velocity = _subtract(
                final_states[1][1], final_states[0][1])
            initial_relative_speed = _norm(initial_relative_velocity)
            final_relative_speed = _norm(final_relative_velocity)
            initial_energy = 0.5 * mass * sum(
                component * component
                for identity in (0, 1)
                for component in initial_states[identity][1])
            final_energy = 0.5 * mass * sum(
                component * component
                for identity in (0, 1)
                for component in final_states[identity][1])
            speed_reduction = (1.0 - final_relative_speed /
                               initial_relative_speed
                               if initial_relative_speed > 0.0 else 0.0)
            energy_reduction = (1.0 - final_energy / initial_energy
                                if initial_energy > 0.0 else 0.0)
            observations.update({
                "initial_relative_speed_m_s": initial_relative_speed,
                "final_relative_speed_m_s": final_relative_speed,
                "relative_speed_reduction_fraction": speed_reduction,
                "initial_kinetic_energy_j": initial_energy,
                "final_kinetic_energy_j": final_energy,
                "kinetic_energy_reduction_fraction": energy_reduction,
            })
            initial_speed_minimum = float(tolerances.get(
                "initial_relative_speed_minimum_m_s", 0.0))
            if initial_relative_speed < initial_speed_minimum:
                failures.append(Failure(
                    "$.fixture.initial_particles",
                    "initial relative speed is too small for viscosity",
                    initial_relative_speed, initial_speed_minimum))
            speed_reduction_minimum = float(tolerances.get(
                "relative_speed_reduction_fraction_minimum", 0.0))
            if speed_reduction < speed_reduction_minimum:
                failures.append(Failure(
                    "$.fixture.final_particles",
                    "viscosity did not dissipate relative motion",
                    speed_reduction, speed_reduction_minimum))
            energy_increase_maximum = float(tolerances.get(
                "kinetic_energy_relative_increase_maximum", 0.0))
            energy_limit = initial_energy * (1.0 + energy_increase_maximum)
            if final_energy > energy_limit:
                failures.append(Failure(
                    "$.fixture.final_particles",
                    "viscosity increased kinetic energy",
                    final_energy, energy_limit))
            energy_reduction_minimum = float(tolerances.get(
                "kinetic_energy_reduction_fraction_minimum", 0.0))
            if energy_reduction < energy_reduction_minimum:
                failures.append(Failure(
                    "$.fixture.final_particles",
                    "viscosity did not dissipate sufficient kinetic energy",
                    energy_reduction, energy_reduction_minimum))

        velocity_changes = [
            _subtract(final_states[identity][1], initial_states[identity][1])
            for identity in (0, 1)
        ]
        impulses = [tuple(mass * component for component in change)
                    for change in velocity_changes]
        activity = sum(_norm(impulse) for impulse in impulses)
        residual = _norm(_sum(impulses))
        momentum_limit = max(
            float(tolerances.get("linear_momentum_absolute_kg_m_s", 0.0)),
            float(tolerances.get("linear_momentum_relative", 0.0)) * activity)
        observations["exchange_impulse_kg_m_s"] = activity
        observations["net_internal_impulse_kg_m_s"] = residual
        if (mode == "contact" and activity < float(tolerances.get(
                "exchange_impulse_minimum_kg_m_s", 0.0))):
            failures.append(Failure("$.fixture.final_particles",
                                    "self-contact impulse activity is too small",
                                    activity, tolerances.get(
                                        "exchange_impulse_minimum_kg_m_s")))
        if residual > momentum_limit:
            failures.append(Failure("$.fixture.final_particles",
                                    "net internal impulse exceeds profile",
                                    residual, momentum_limit))

        initial_center = tuple(
            0.5 * (initial_states[0][0][axis] + initial_states[1][0][axis])
            for axis in range(3))
        final_center = tuple(
            0.5 * (final_states[0][0][axis] + final_states[1][0][axis])
            for axis in range(3))
        center_drift = _norm(_subtract(final_center, initial_center))
        observations["center_of_mass_drift_m"] = center_drift
        center_limit = float(tolerances.get(
            "center_of_mass_drift_absolute_m", 0.0))
        if center_drift > center_limit:
            failures.append(Failure("$.fixture.final_particles",
                                    "center-of-mass drift exceeds profile",
                                    center_drift, center_limit))

        transverse_position = max(
            abs(final_states[identity][0][component])
            for identity in (0, 1) for component in (1, 2))
        transverse_velocity = max(
            abs(final_states[identity][1][component])
            for identity in (0, 1) for component in (1, 2))
        observations["maximum_transverse_position_m"] = transverse_position
        observations["maximum_transverse_velocity_m_s"] = transverse_velocity
        transverse_position_limit = float(tolerances.get(
            "transverse_position_absolute_m", 0.0))
        transverse_velocity_limit = float(tolerances.get(
            "transverse_velocity_absolute_m_s", 0.0))
        if (transverse_position > transverse_position_limit or
                transverse_velocity > transverse_velocity_limit):
            failures.append(Failure("$.fixture.final_particles",
                                    "transverse symmetry leakage exceeds profile",
                                    {
                                        "position_m": transverse_position,
                                        "velocity_m_s": transverse_velocity,
                                    }, {
                                        "position_m": transverse_position_limit,
                                        "velocity_m_s": transverse_velocity_limit,
                                    }))

        if mode in ("contact", "pressure"):
            direction = _unit(final_separation)
            relative_velocity = _subtract(
                final_states[1][1], final_states[0][1])
            outward_speed = (sum(a * b for a, b in zip(
                relative_velocity, direction)) if direction is not None else 0.0)
            observations["outward_relative_speed_m_s"] = outward_speed
            outward_minimum = float(tolerances.get(
                "outward_relative_speed_minimum_m_s", 0.0))
            if outward_speed < outward_minimum:
                failures.append(Failure("$.fixture.final_particles",
                                        "self response is not outward",
                                        outward_speed, outward_minimum))

        maximum_speed = max(_norm(final_states[identity][1])
                            for identity in (0, 1))
        observations["maximum_speed_m_s"] = maximum_speed
        if maximum_speed > float(definition.get("maximum_speed_m_s", 0.0)):
            failures.append(Failure("$.fixture.final_particles",
                                    "fluid speed exceeds declared fixture bound",
                                    maximum_speed,
                                    definition.get("maximum_speed_m_s")))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": 1 if len(final_states) == 2 else 0,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_penetration(profile: dict[str, Any], result: dict[str, Any],
                          expected_hash: str, driver: Any,
                          failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture", "expected an object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected an object", actual))
        actual = {}

    exact_fields = ("id", "scene", "scene_sha256", "body_count", "frames",
                    "substeps", "timestep_seconds", "oracle")
    for field in exact_fields:
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting does not match profile",
                                    actual.get(field), definition.get(field)))

    frames = definition.get("frames")
    frame = actual.get("maximum_penetration_frame")
    if (not isinstance(frame, int) or isinstance(frame, bool) or
            not isinstance(frames, int) or frame < 0 or frame >= frames):
        failures.append(Failure("$.fixture.maximum_penetration_frame",
                                "maximum penetration frame is outside the run",
                                frame, f"0 <= frame < {frames}"))

    penetration = _scalar(actual.get("maximum_penetration_m"),
                          "$.fixture.maximum_penetration_m", failures)
    limit = profile.get("tolerances", {}).get("maximum_penetration_m")
    if not _number(limit) or float(limit) < 0.0:
        failures.append(Failure("$.profile.tolerances.maximum_penetration_m",
                                "expected a nonnegative finite limit", limit))
    elif penetration is not None:
        if penetration < 0.0:
            failures.append(Failure("$.fixture.maximum_penetration_m",
                                    "penetration cannot be negative", penetration, 0.0))
        elif penetration > float(limit):
            failures.append(Failure("$.fixture.maximum_penetration_m",
                                    "maximum penetration exceeds profile",
                                    penetration, float(limit)))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": 1 if penetration is not None else 0,
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_rigid_attachment(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture", "expected an object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected an object", actual))
        actual = {}

    exact_fields = ("id", "scene", "scene_sha256", "body_count",
                    "constraint_count", "frames", "substeps",
                    "timestep_seconds", "oracle")
    for field in exact_fields:
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting does not match profile",
                                    actual.get(field), definition.get(field)))

    maximum_metrics = (
        "maximum_hinge_anchor_error_m",
        "maximum_hinge_axis_error",
        "maximum_slider_sideways_drift_m",
        "maximum_slider_axis_error",
        "maximum_slider_travel_m",
    )
    measurements: dict[str, float] = {}
    tolerances = profile.get("tolerances", {})
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances", "expected an object",
                                tolerances))
        tolerances = {}
    for name in maximum_metrics:
        value = _scalar(actual.get(name), f"$.fixture.{name}", failures)
        limit = tolerances.get(name)
        if not _number(limit) or float(limit) < 0.0:
            failures.append(Failure(f"$.profile.tolerances.{name}",
                                    "expected a nonnegative finite limit", limit))
        elif value is not None:
            measurements[name] = value
            if value < 0.0:
                failures.append(Failure(f"$.fixture.{name}",
                                        "attachment measurement cannot be negative",
                                        value, 0.0))
            elif value >= float(limit):
                failures.append(Failure(f"$.fixture.{name}",
                                        "attachment maximum reaches or exceeds profile",
                                        value, float(limit)))

    travel = measurements.get("maximum_slider_travel_m")
    minimum_travel = tolerances.get("minimum_slider_travel_m")
    if not _number(minimum_travel) or float(minimum_travel) < 0.0:
        failures.append(Failure(
            "$.profile.tolerances.minimum_slider_travel_m",
            "expected a nonnegative finite limit", minimum_travel))
    elif travel is not None and travel <= float(minimum_travel):
        failures.append(Failure("$.fixture.maximum_slider_travel_m",
                                "slider travel does not exceed profile minimum",
                                travel, float(minimum_travel)))

    for field in ("constraints_enabled", "constraints_unbroken"):
        if actual.get(field) is not True:
            failures.append(Failure(f"$.fixture.{field}",
                                    "all authored constraints must remain active",
                                    actual.get(field), True))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": len(measurements),
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_rigid_kinematic_sweep(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        failures: list[Failure]) -> dict[str, Any]:
    fixture = profile.get("fixture")
    if not isinstance(fixture, dict):
        failures.append(Failure("$.profile.fixture", "expected an object", fixture))
        fixture = {}

    if result.get("schema") != CONFORMANCE_RESULT_SCHEMA:
        failures.append(Failure("$.schema", "unsupported conformance result schema",
                                result.get("schema"), CONFORMANCE_RESULT_SCHEMA))
    if result.get("execution_status") != "completed":
        failures.append(Failure("$.execution_status",
                                "quality execution must be completed",
                                result.get("execution_status"), "completed"))
    if result.get("case_id") != fixture.get("case_id"):
        failures.append(Failure("$.case_id", "case identity mismatch",
                                result.get("case_id"), fixture.get("case_id")))
    if result.get("case_sha256") != fixture.get("case_sha256"):
        failures.append(Failure("$.case_sha256", "case hash mismatch",
                                result.get("case_sha256"), fixture.get("case_sha256")))

    backend = result.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        failures.append(Failure("$.backend", "backend identity is required", backend))
        backend = "unknown"
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    driver = {
        "name": backend,
        "device": provenance.get("device_name", "not-reported"),
    }

    frame_count = fixture.get("frames")
    checkpoints = result.get("checkpoints")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
        failures.append(Failure("$.profile.fixture.frames",
                                "expected a positive frame count", frame_count))
        frame_count = 0
    if not isinstance(checkpoints, list):
        failures.append(Failure("$.checkpoints", "expected a checkpoint array",
                                checkpoints))
        checkpoints = []
    expected_frames = list(range(frame_count + 1))
    actual_frames = [checkpoint.get("frame") if isinstance(checkpoint, dict) else None
                     for checkpoint in checkpoints]
    if actual_frames != expected_frames:
        failures.append(Failure("$.checkpoints",
                                "every frame from zero through the configured run is required",
                                actual_frames, expected_frames))
    by_frame = {checkpoint.get("frame"): checkpoint for checkpoint in checkpoints
                if isinstance(checkpoint, dict) and isinstance(checkpoint.get("frame"), int)}

    def resource(frame: int, identity: str) -> dict[str, Any] | None:
        checkpoint = by_frame.get(frame)
        path = f"$.checkpoints[{frame}].resources"
        if not isinstance(checkpoint, dict):
            failures.append(Failure(path, "required checkpoint is missing", frame))
            return None
        resources = checkpoint.get("resources")
        if not isinstance(resources, list):
            failures.append(Failure(path, "expected a resource array", resources))
            return None
        matches = [item for item in resources
                   if isinstance(item, dict) and item.get("id") == identity]
        if len(matches) != 1:
            failures.append(Failure(path, "expected exactly one named rigid body",
                                    identity, 1))
            return None
        return matches[0]

    preimpact_frame = fixture.get("preimpact_frame")
    impact_frame = fixture.get("impact_frame")
    body_id = fixture.get("body")
    collider_id = fixture.get("collider")
    if (not isinstance(preimpact_frame, int) or isinstance(preimpact_frame, bool) or
            not isinstance(impact_frame, int) or isinstance(impact_frame, bool) or
            not isinstance(body_id, str) or not isinstance(collider_id, str)):
        failures.append(Failure("$.profile.fixture",
                                "sweep frames and rigid body identities are required"))
        preimpact_frame = impact_frame = -1
        body_id = collider_id = ""

    before_body = resource(preimpact_frame, body_id)
    before_collider = resource(preimpact_frame, collider_id)
    impact_body = resource(impact_frame, body_id)
    impact_collider = resource(impact_frame, collider_id)
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances", "expected an object",
                                tolerances))
        tolerances = {}

    exact_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    timestep = fixture.get("timestep_seconds")
    previous_target = _vector(fixture.get("kinematic_previous_position"),
                              "$.profile.fixture.kinematic_previous_position",
                              failures)
    target = _vector(fixture.get("kinematic_target_position"),
                     "$.profile.fixture.kinematic_target_position", failures)
    sweep = _vector(fixture.get("kinematic_sweep_displacement"),
                    "$.profile.fixture.kinematic_sweep_displacement", failures)
    half_extents = _vector(fixture.get("collider_half_extents"),
                           "$.profile.fixture.collider_half_extents", failures)
    response_axis = _vector(fixture.get("angular_response_axis"),
                            "$.profile.fixture.angular_response_axis", failures)

    def close_vector(actual: Any, expected: tuple[float, float, float] | None,
                     path: str) -> tuple[float, float, float] | None:
        value = _vector(actual, path, failures)
        if value is not None and expected is not None:
            error = max(abs(a - b) for a, b in zip(value, expected))
            if error > exact_tolerance:
                failures.append(Failure(path, "fixture state differs from profile",
                                        list(value), list(expected)))
        return value

    if before_collider is not None:
        close_vector(before_collider.get("position"), previous_target,
                     f"$.checkpoints[{preimpact_frame}].{collider_id}.position")
    impact_collider_position = None
    if impact_collider is not None:
        impact_collider_position = close_vector(
            impact_collider.get("position"), target,
            f"$.checkpoints[{impact_frame}].{collider_id}.position")
        if (_number(timestep) and float(timestep) > 0.0 and
                previous_target is not None and target is not None):
            expected_velocity = tuple((end - start) / float(timestep)
                                      for start, end in zip(previous_target, target))
            close_vector(impact_collider.get("linear_velocity"), expected_velocity,
                         f"$.checkpoints[{impact_frame}].{collider_id}.linear_velocity")

    def rigid_contacts(frame: int) -> list[dict[str, Any]]:
        checkpoint = by_frame.get(frame)
        path = f"$.checkpoints[{frame}].contacts"
        contacts = checkpoint.get("contacts") if isinstance(checkpoint, dict) else None
        if not isinstance(contacts, dict):
            failures.append(Failure(path, "expected contact diagnostics", contacts))
            return []
        values = contacts.get("rigid_contacts")
        count = contacts.get("rigid_contact_count")
        if not isinstance(values, list) or not all(isinstance(item, dict)
                                                   for item in values):
            failures.append(Failure(path + ".rigid_contacts",
                                    "expected a rigid-contact array", values))
            return []
        if count != len(values):
            failures.append(Failure(path + ".rigid_contact_count",
                                    "contact count does not match records",
                                    count, len(values)))
        return values

    if rigid_contacts(preimpact_frame):
        failures.append(Failure(f"$.checkpoints[{preimpact_frame}].contacts",
                                "preimpact frame must be contact-free"))
    contacts = rigid_contacts(impact_frame)
    if not contacts:
        failures.append(Failure(f"$.checkpoints[{impact_frame}].contacts",
                                "kinematic sweep produced no rigid contacts"))

    weighted_normal = (0.0, 0.0, 0.0)
    active_contacts = 0
    contact_point_tolerance = float(tolerances.get("contact_point_absolute_m", 0.0))
    friction_coefficient = fixture.get("friction_coefficient")
    friction_tolerance = float(tolerances.get("friction_cone_absolute", 0.0))
    for index, contact in enumerate(contacts):
        path = f"$.checkpoints[{impact_frame}].contacts.rigid_contacts[{index}]"
        if contact.get("body") != body_id or contact.get("collider") != collider_id:
            failures.append(Failure(path, "unexpected contact body pair",
                                    [contact.get("body"), contact.get("collider")],
                                    [body_id, collider_id]))
        normal = _vector(contact.get("normal"), path + ".normal", failures)
        point = _vector(contact.get("position"), path + ".position", failures)
        friction = _vector(contact.get("friction_impulse"),
                           path + ".friction_impulse", failures)
        normal_impulse = _scalar(contact.get("normal_impulse"),
                                 path + ".normal_impulse", failures)
        if normal is not None:
            length = _norm(normal)
            if abs(length - 1.0) > 1.0e-3:
                failures.append(Failure(path + ".normal",
                                        "contact normal is not unit length", length, 1.0))
        if normal_impulse is not None:
            if normal_impulse < 0.0:
                failures.append(Failure(path + ".normal_impulse",
                                        "normal impulse cannot be negative",
                                        normal_impulse, 0.0))
            elif normal_impulse > 0.0:
                active_contacts += 1
                if normal is not None:
                    weighted_normal = tuple(
                        total + component * normal_impulse
                        for total, component in zip(weighted_normal, normal))
        if (friction is not None and normal_impulse is not None and
                _number(friction_coefficient)):
            bound = float(friction_coefficient) * max(normal_impulse, 0.0)
            if _norm(friction) > bound + friction_tolerance:
                failures.append(Failure(path + ".friction_impulse",
                                        "friction impulse exceeds Coulomb cone",
                                        _norm(friction), bound + friction_tolerance))
        if (point is not None and impact_collider_position is not None and
                half_extents is not None):
            excess = max(abs(value - center) - half - contact_point_tolerance
                         for value, center, half in zip(
                             point, impact_collider_position, half_extents))
            if excess > 0.0:
                failures.append(Failure(path + ".position",
                                        "contact point lies outside the declared collider",
                                        list(point),
                                        {"center": list(impact_collider_position),
                                         "half_extents": list(half_extents),
                                         "absolute": contact_point_tolerance}))

    minimum_impulse = float(tolerances.get("normal_impulse_activity_minimum", 0.0))
    weighted_length = _norm(weighted_normal)
    if active_contacts == 0 or weighted_length < minimum_impulse:
        failures.append(Failure(f"$.checkpoints[{impact_frame}].contacts",
                                "normal impulse activity is too small for the sweep",
                                weighted_length, minimum_impulse))
    normal_alignment = None
    sweep_unit = _unit(sweep) if sweep is not None else None
    normal_unit = _unit(weighted_normal)
    if sweep_unit is not None and normal_unit is not None:
        normal_alignment = sum(a * b for a, b in zip(sweep_unit, normal_unit))
        minimum_alignment = float(tolerances.get("normal_alignment_cosine_minimum", 1.0))
        if normal_alignment < minimum_alignment:
            failures.append(Failure(f"$.checkpoints[{impact_frame}].contacts",
                                    "contact reaction is not aligned with the kinematic sweep",
                                    normal_alignment, minimum_alignment))

    planar = {}
    angular_response = None
    if impact_body is not None:
        position = _vector(impact_body.get("position"),
                           f"$.checkpoints[{impact_frame}].{body_id}.position",
                           failures)
        velocity = _vector(impact_body.get("linear_velocity"),
                           f"$.checkpoints[{impact_frame}].{body_id}.linear_velocity",
                           failures)
        angular = _vector(impact_body.get("angular_velocity"),
                          f"$.checkpoints[{impact_frame}].{body_id}.angular_velocity",
                          failures)
        orientation = _quaternion(impact_body.get("orientation"),
                                  f"$.checkpoints[{impact_frame}].{body_id}.orientation",
                                  failures)
        planar_limits = {
            "position_z": float(tolerances.get("planar_position_absolute_m", 0.0)),
            "linear_velocity_z": float(tolerances.get(
                "planar_linear_velocity_absolute_m_s", 0.0)),
            "orientation_xy": float(tolerances.get(
                "planar_orientation_component_absolute", 0.0)),
            "angular_velocity_xy": float(tolerances.get(
                "planar_angular_velocity_absolute_rad_s", 0.0)),
        }
        if position is not None:
            planar["position_z"] = abs(position[2])
        if velocity is not None:
            planar["linear_velocity_z"] = abs(velocity[2])
            sweep_projection = (sum(a * b for a, b in zip(velocity, sweep_unit))
                                if sweep_unit is not None else 0.0)
            minimum_speed = float(tolerances.get(
                "sweep_response_speed_minimum_m_s", 0.0))
            if sweep_projection < minimum_speed:
                failures.append(Failure(
                    f"$.checkpoints[{impact_frame}].{body_id}.linear_velocity",
                    "dynamic body did not respond along the kinematic sweep",
                    sweep_projection, minimum_speed))
        if orientation is not None:
            planar["orientation_xy"] = max(abs(orientation[0]),
                                            abs(orientation[1]))
        if angular is not None:
            planar["angular_velocity_xy"] = max(abs(angular[0]), abs(angular[1]))
            axis_unit = _unit(response_axis) if response_axis is not None else None
            if axis_unit is not None:
                angular_response = sum(a * b for a, b in zip(angular, axis_unit))
                minimum_angular = float(tolerances.get(
                    "angular_response_minimum_rad_s", 0.0))
                if angular_response < minimum_angular:
                    failures.append(Failure(
                        f"$.checkpoints[{impact_frame}].{body_id}.angular_velocity",
                        "off-center sweep lacks the required angular response",
                        angular_response, minimum_angular))
        for name, value in planar.items():
            if value > planar_limits[name]:
                failures.append(Failure(
                    f"$.checkpoints[{impact_frame}].{body_id}.{name}",
                    "planar symmetry leakage exceeds profile",
                    value, planar_limits[name]))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if checkpoints else 0,
        "sample_count": len(checkpoints),
        "failure_count": len(failures),
        "observations": {
            "active_contact_count": active_contacts,
            "angular_response_rad_s": angular_response,
            "normal_alignment_cosine": normal_alignment,
            "normal_impulse_vector": list(weighted_normal),
            "planar_leakage": planar,
        },
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_rope_self_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture", "expected an object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected an object", actual))
        actual = {}
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances", "expected an object",
                                tolerances))
        tolerances = {}

    exact_fields = (
        "id", "point_count", "sampled_node_count", "substeps",
        "solver_iterations", "self_collision",
        "topology_exclusion_index_distance", "excluded_probe_pair", "oracle",
    )
    for field in exact_fields:
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting does not match profile",
                                    actual.get(field), definition.get(field)))

    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    scalar_fields = (
        "timestep_seconds", "node_spacing_m", "radius_m", "mass_kg",
        "stretch_compliance_m_per_n", "velocity_damping_per_s",
        "maximum_substep_timestep_seconds", "friction", "maximum_speed_m_s",
    )
    for field in scalar_fields:
        value = _scalar(actual.get(field), f"$.fixture.{field}", failures)
        expected = _scalar(definition.get(field),
                           f"$.profile.fixture.{field}", failures)
        if (value is not None and expected is not None and
                abs(value - expected) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    value, expected))

    gravity = _vector(actual.get("gravity_m_s2"),
                      "$.fixture.gravity_m_s2", failures)
    expected_gravity = _vector(definition.get("gravity_m_s2"),
                               "$.profile.fixture.gravity_m_s2", failures)
    if (gravity is not None and expected_gravity is not None and
            max(abs(a - b) for a, b in zip(gravity, expected_gravity)) >
            state_tolerance):
        failures.append(Failure("$.fixture.gravity_m_s2",
                                "fixture setting differs from profile",
                                list(gravity), list(expected_gravity)))

    def vectors(value: Any, path: str) -> list[tuple[float, float, float]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a vector array", value))
            return []
        parsed: list[tuple[float, float, float]] = []
        for index, item in enumerate(value):
            vector = _vector(item, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    centerline = vectors(actual.get("initial_centerline_m"),
                         "$.fixture.initial_centerline_m")
    expected_centerline = vectors(definition.get("initial_centerline_m"),
                                  "$.profile.fixture.initial_centerline_m")
    if len(centerline) != len(expected_centerline):
        failures.append(Failure("$.fixture.initial_centerline_m",
                                "centerline point count differs from profile",
                                len(centerline), len(expected_centerline)))
    else:
        for index, (value, expected) in enumerate(zip(centerline,
                                                       expected_centerline)):
            if max(abs(a - b) for a, b in zip(value, expected)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_centerline_m[{index}]",
                    "initial public input differs from profile",
                    list(value), list(expected)))

    sampled_count = definition.get("sampled_node_count")
    expected_initial = vectors(definition.get("initial_sampled_nodes_m"),
                               "$.profile.fixture.initial_sampled_nodes_m")

    def nodes(value: Any, path: str) -> list[tuple[
            tuple[float, float, float], tuple[float, float, float]]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a node array", value))
            return []
        if not isinstance(sampled_count, int) or isinstance(sampled_count, bool):
            failures.append(Failure("$.profile.fixture.sampled_node_count",
                                    "expected an integer", sampled_count))
        elif len(value) != sampled_count:
            failures.append(Failure(path, "sampled node count differs from profile",
                                    len(value), sampled_count))
        parsed = []
        for index, node in enumerate(value):
            node_path = f"{path}[{index}]"
            if not isinstance(node, dict):
                failures.append(Failure(node_path, "expected a node object", node))
                continue
            if node.get("index") != index:
                failures.append(Failure(node_path + ".index",
                                        "node order/index changed", node.get("index"),
                                        index))
            position = _vector(node.get("position_m"), node_path + ".position_m",
                               failures)
            velocity = _vector(node.get("velocity_m_s"),
                               node_path + ".velocity_m_s", failures)
            if position is not None and velocity is not None:
                parsed.append((position, velocity))
        return parsed

    initial = nodes(actual.get("initial_nodes"), "$.fixture.initial_nodes")
    final = nodes(actual.get("final_nodes"), "$.fixture.final_nodes")
    if len(initial) == len(expected_initial):
        for index, ((position, velocity), expected) in enumerate(
                zip(initial, expected_initial)):
            if max(abs(a - b) for a, b in zip(position, expected)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_nodes[{index}].position_m",
                    "initial sampled public state differs from profile",
                    list(position), list(expected)))
            if _norm(velocity) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_nodes[{index}].velocity_m_s",
                    "initial sampled velocity must be zero", list(velocity), 0.0))
    elif initial:
        failures.append(Failure("$.fixture.initial_nodes",
                                "initial sampled profile count differs",
                                len(initial), len(expected_initial)))

    target_separation = None
    radius = _scalar(definition.get("radius_m"),
                     "$.profile.fixture.radius_m", failures)
    exclusion = definition.get("topology_exclusion_index_distance")
    eligible_initial = eligible_final = None
    overlapping_eligible_pairs = 0
    if (radius is not None and radius > 0.0 and
            isinstance(exclusion, int) and not isinstance(exclusion, bool) and
            exclusion >= 0 and len(initial) == len(final) and initial):
        target_separation = 2.0 * radius
        minimum_fraction = float(tolerances.get(
            "minimum_separation_fraction", 1.0))
        activity = float(tolerances.get(
            "initial_overlap_activity_minimum_m", 0.0))
        minimum_allowed = target_separation * minimum_fraction
        for first in range(len(initial)):
            for second in range(first + exclusion + 1, len(initial)):
                before = _norm(_subtract(initial[first][0], initial[second][0]))
                after = _norm(_subtract(final[first][0], final[second][0]))
                eligible_initial = before if eligible_initial is None else min(
                    eligible_initial, before)
                eligible_final = after if eligible_final is None else min(
                    eligible_final, after)
                if target_separation - before >= activity:
                    overlapping_eligible_pairs += 1
                    if after < minimum_allowed:
                        failures.append(Failure(
                            f"$.fixture.final_nodes[{first},{second}]",
                            "nonadjacent sampled nodes remain inside the radius separation",
                            after, minimum_allowed))
        if overlapping_eligible_pairs == 0:
            failures.append(Failure("$.fixture.initial_nodes",
                                    "fixture has no active nonadjacent overlap",
                                    eligible_initial,
                                    target_separation - activity))

        probe = definition.get("excluded_probe_pair")
        if (isinstance(probe, list) and len(probe) == 2 and
                all(isinstance(index, int) and not isinstance(index, bool)
                    for index in probe)):
            first, second = probe
            if (first < 0 or second >= len(initial) or first >= second or
                    second - first > exclusion):
                failures.append(Failure("$.profile.fixture.excluded_probe_pair",
                                        "excluded probe is not a valid topology neighbor",
                                        probe))
            else:
                excluded_limit = target_separation * float(tolerances.get(
                    "excluded_probe_maximum_fraction", 1.0))
                excluded_initial = _norm(_subtract(initial[first][0],
                                                    initial[second][0]))
                excluded_final = _norm(_subtract(final[first][0],
                                                  final[second][0]))
                if excluded_initial >= excluded_limit:
                    failures.append(Failure(
                        "$.fixture.initial_nodes",
                        "topology probe is not initially inside the exclusion radius",
                        excluded_initial, excluded_limit))
                if excluded_final >= excluded_limit:
                    failures.append(Failure(
                        "$.fixture.final_nodes",
                        "topology-neighbor probe was separated as self contact",
                        excluded_final, excluded_limit))
        else:
            failures.append(Failure("$.profile.fixture.excluded_probe_pair",
                                    "expected two topology indices", probe))

    center_of_mass_drift = None
    maximum_transverse = None
    maximum_speed = None
    if len(initial) == len(final) and initial:
        initial_center = tuple(sum(node[0][axis] for node in initial) / len(initial)
                               for axis in range(3))
        final_center = tuple(sum(node[0][axis] for node in final) / len(final)
                             for axis in range(3))
        center_of_mass_drift = _norm(_subtract(final_center, initial_center))
        center_limit = float(tolerances.get(
            "center_of_mass_drift_absolute_m", 0.0))
        if center_of_mass_drift > center_limit:
            failures.append(Failure("$.fixture.final_nodes",
                                    "center-of-mass drift exceeds profile",
                                    center_of_mass_drift, center_limit))
        maximum_transverse = max(abs(node[0][2]) for node in initial + final)
        transverse_limit = float(tolerances.get(
            "transverse_position_absolute_m", 0.0))
        if maximum_transverse > transverse_limit:
            failures.append(Failure("$.fixture.final_nodes",
                                    "transverse symmetry leakage exceeds profile",
                                    maximum_transverse, transverse_limit))
        maximum_speed = max(_norm(node[1]) for node in final)
        speed_limit = float(definition.get("maximum_speed_m_s", 0.0)) + float(
            tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure("$.fixture.final_nodes",
                                    "node speed exceeds authored maximum",
                                    maximum_speed, speed_limit))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": overlapping_eligible_pairs,
        "failure_count": len(failures),
        "observations": {
            "target_separation_m": target_separation,
            "minimum_eligible_initial_separation_m": eligible_initial,
            "minimum_eligible_final_separation_m": eligible_final,
            "overlapping_eligible_pair_count": overlapping_eligible_pairs,
            "center_of_mass_drift_m": center_of_mass_drift,
            "maximum_transverse_position_m": maximum_transverse,
            "maximum_node_speed_m_s": maximum_speed,
        },
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_fluid_cloth_containment(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture",
                                "expected a fixture object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected a fixture object", actual))
        actual = {}
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    for field in ("id", "substeps", "fluid_particle_count",
                  "cloth_solver_iterations", "cloth_preserve_volume",
                  "two_way", "oracle"):
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    actual.get(field), definition.get(field)))
    for field in (
            "timestep_seconds", "fluid_rest_density_kg_m3",
            "fluid_particle_radius_m", "fluid_support_radius_m",
            "fluid_rest_particle_volume_m3", "fluid_maximum_speed_m_s",
            "cloth_vertex_mass_kg", "cloth_thickness_m",
            "cloth_stretch_compliance_m_per_n",
            "cloth_bending_compliance_m_per_n", "cloth_target_volume_m3",
            "cloth_volume_compliance_m3_per_pa", "contact_distance_m",
            "stiffness_n_per_m", "damping_n_s_per_m",
            "tangential_drag_n_s_per_m", "maximum_force_n"):
        value = _scalar(actual.get(field), f"$.fixture.{field}", failures)
        expected = _scalar(definition.get(field),
                           f"$.profile.fixture.{field}", failures)
        if (value is not None and expected is not None and
                abs(value - expected) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    value, expected))
    gravity = _vector(actual.get("gravity_m_s2"),
                      "$.fixture.gravity_m_s2", failures)
    expected_gravity = _vector(definition.get("gravity_m_s2"),
                               "$.profile.fixture.gravity_m_s2", failures)
    if (gravity is not None and expected_gravity is not None and
            max(abs(a - b) for a, b in zip(gravity, expected_gravity)) >
            state_tolerance):
        failures.append(Failure("$.fixture.gravity_m_s2",
                                "fixture setting differs from profile",
                                list(gravity), list(expected_gravity)))

    def vectors(value: Any, path: str) -> list[tuple[float, float, float]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a vector array", value))
            return []
        parsed: list[tuple[float, float, float]] = []
        for index, item in enumerate(value):
            vector = _vector(item, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    vertices = vectors(actual.get("cloth_vertices_m"),
                       "$.fixture.cloth_vertices_m")
    expected_vertices = vectors(definition.get("cloth_vertices_m"),
                                "$.profile.fixture.cloth_vertices_m")
    if len(vertices) != len(expected_vertices):
        failures.append(Failure("$.fixture.cloth_vertices_m",
                                "cloth vertex count differs from profile",
                                len(vertices), len(expected_vertices)))
    else:
        for index, (value, expected) in enumerate(zip(vertices,
                                                       expected_vertices)):
            if max(abs(a - b) for a, b in zip(value, expected)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.cloth_vertices_m[{index}]",
                    "cloth public input differs from profile",
                    list(value), list(expected)))
    indices_value = actual.get("cloth_triangle_indices")
    expected_indices = definition.get("cloth_triangle_indices")
    if indices_value != expected_indices:
        failures.append(Failure("$.fixture.cloth_triangle_indices",
                                "cloth topology differs from profile",
                                indices_value, expected_indices))
    indices = indices_value if (isinstance(indices_value, list) and
                                all(isinstance(item, int) and
                                    not isinstance(item, bool)
                                    for item in indices_value)) else []
    if len(indices) == 0 or len(indices) % 3 != 0:
        failures.append(Failure("$.fixture.cloth_triangle_indices",
                                "expected complete triangle indices", indices))

    def fluid_states(value: Any, path: str) -> list[dict[str, Any]]:
        expected_count = definition.get("fluid_particle_count")
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a particle array", value))
            return []
        if isinstance(expected_count, int) and len(value) != expected_count:
            failures.append(Failure(path, "fluid particle count differs",
                                    len(value), expected_count))
        parsed: list[dict[str, Any]] = []
        for index, particle in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(item_path,
                                        "expected a particle object", particle))
                continue
            if particle.get("stable_id") != index:
                failures.append(Failure(item_path + ".stable_id",
                                        "fluid stable ID/order changed",
                                        particle.get("stable_id"), index))
            position = _vector(particle.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(particle.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            acceleration = _vector(particle.get("acceleration_m_s2"),
                                   item_path + ".acceleration_m_s2", failures)
            foam = _scalar(particle.get("foam"), item_path + ".foam", failures)
            if (position is not None and velocity is not None and
                    acceleration is not None and foam is not None):
                parsed.append({"position": position, "velocity": velocity,
                               "acceleration": acceleration, "foam": foam})
        return parsed

    def cloth_states(value: Any, path: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a cloth vertex array", value))
            return []
        if len(value) != len(expected_vertices):
            failures.append(Failure(path, "cloth state count differs",
                                    len(value), len(expected_vertices)))
        parsed: list[dict[str, Any]] = []
        for index, vertex in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(vertex, dict):
                failures.append(Failure(item_path,
                                        "expected a cloth vertex object", vertex))
                continue
            if vertex.get("index") != index:
                failures.append(Failure(item_path + ".index",
                                        "cloth vertex order/index changed",
                                        vertex.get("index"), index))
            position = _vector(vertex.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(vertex.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            force = _vector(vertex.get("liquid_contact_force_n"),
                            item_path + ".liquid_contact_force_n", failures)
            if position is not None and velocity is not None and force is not None:
                parsed.append({"position": position, "velocity": velocity,
                               "force": force})
        return parsed

    initial_profile = definition.get("initial_particles")
    initial_fluid = fluid_states(actual.get("initial_fluid_state"),
                                 "$.fixture.initial_fluid_state")
    final_fluid = fluid_states(actual.get("final_fluid_state"),
                               "$.fixture.final_fluid_state")
    if isinstance(initial_profile, list) and len(initial_fluid) == len(initial_profile):
        for index, (state, expected) in enumerate(zip(initial_fluid,
                                                       initial_profile)):
            expected_position = _vector(
                expected.get("position_m") if isinstance(expected, dict) else None,
                f"$.profile.fixture.initial_particles[{index}].position_m",
                failures)
            expected_velocity = _vector(
                expected.get("velocity_m_s") if isinstance(expected, dict) else None,
                f"$.profile.fixture.initial_particles[{index}].velocity_m_s",
                failures)
            if (expected_position is not None and
                    max(abs(a - b) for a, b in zip(
                        state["position"], expected_position)) > state_tolerance):
                failures.append(Failure(
                    f"$.fixture.initial_fluid_state[{index}].position_m",
                    "initial particle position differs from profile",
                    list(state["position"]), list(expected_position)))
            if (expected_velocity is not None and
                    max(abs(a - b) for a, b in zip(
                        state["velocity"], expected_velocity)) > state_tolerance):
                failures.append(Failure(
                    f"$.fixture.initial_fluid_state[{index}].velocity_m_s",
                    "initial particle velocity differs from profile",
                    list(state["velocity"]), list(expected_velocity)))
    else:
        failures.append(Failure("$.fixture.initial_fluid_state",
                                "initial particle profile count differs",
                                len(initial_fluid),
                                len(initial_profile) if
                                isinstance(initial_profile, list) else None))

    initial_cloth = cloth_states(actual.get("initial_cloth_state"),
                                 "$.fixture.initial_cloth_state")
    final_cloth = cloth_states(actual.get("final_cloth_state"),
                               "$.fixture.final_cloth_state")
    if len(initial_cloth) == len(expected_vertices):
        for index, (state, expected) in enumerate(zip(initial_cloth,
                                                       expected_vertices)):
            if max(abs(a - b) for a, b in zip(
                    state["position"], expected)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_cloth_state[{index}].position_m",
                    "initial cloth public state differs from profile",
                    list(state["position"]), list(expected)))
            if _norm(state["velocity"]) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_cloth_state[{index}].velocity_m_s",
                    "initial cloth velocity must be zero",
                    list(state["velocity"]), 0.0))
            if _norm(state["force"]) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_cloth_state[{index}].liquid_contact_force_n",
                    "initial cloth liquid force must be zero",
                    list(state["force"]), 0.0))

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def signed_volume(points: list[tuple[float, float, float]]) -> float | None:
        if (not indices or not points or
                any(index < 0 or index >= len(points) for index in indices)):
            failures.append(Failure("$.fixture.cloth_triangle_indices",
                                    "triangle index is outside cloth vertices",
                                    indices, len(points)))
            return None
        return sum(dot(points[indices[corner]], cross(
            points[indices[corner + 1]], points[indices[corner + 2]])) / 6.0
                   for corner in range(0, len(indices), 3))

    manifold = bool(indices and vertices)
    edge_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    if manifold:
        for corner in range(0, len(indices), 3):
            triangle = indices[corner:corner + 3]
            if len(set(triangle)) != 3:
                manifold = False
                failures.append(Failure(
                    f"$.fixture.cloth_triangle_indices[{corner}]",
                    "cloth triangle repeats a vertex", triangle))
                continue
            for offset in range(3):
                first = triangle[offset]
                second = triangle[(offset + 1) % 3]
                edge_directions.setdefault(tuple(sorted((first, second))), []).append(
                    (first, second))
        for edge, directions in edge_directions.items():
            if (len(directions) != 2 or directions[0] != directions[1][::-1]):
                manifold = False
                failures.append(Failure("$.fixture.cloth_triangle_indices",
                                        "cloth boundary is not closed and consistently oriented",
                                        {"edge": edge, "directions": directions}))

    initial_volume = signed_volume(vertices) if manifold else None
    final_positions = [state["position"] for state in final_cloth]
    final_volume = (signed_volume(final_positions)
                    if manifold and len(final_positions) == len(vertices) else None)
    volume_relative_error = None
    if initial_volume is not None:
        if initial_volume <= 0.0:
            failures.append(Failure("$.fixture.cloth_triangle_indices",
                                    "closed cloth orientation must have positive volume",
                                    initial_volume, "> 0"))
        if final_volume is not None and abs(initial_volume) > 0.0:
            if final_volume <= 0.0:
                failures.append(Failure("$.fixture.final_cloth_state",
                                        "closed cloth orientation inverted",
                                        final_volume, "> 0"))
            volume_relative_error = (
                abs(abs(final_volume) - abs(initial_volume)) /
                abs(initial_volume))
            volume_limit = float(tolerances.get(
                "maximum_cloth_volume_relative_error", 0.0))
            if volume_relative_error > volume_limit:
                failures.append(Failure("$.fixture.final_cloth_state",
                                        "closed cloth volume drift exceeds profile",
                                        volume_relative_error, volume_limit))

    maximum_signed_distance = None
    maximum_penetration = None
    leaked_particles = 0
    contact_distance = _scalar(definition.get("contact_distance_m"),
                               "$.profile.fixture.contact_distance_m", failures)
    if (manifold and len(final_positions) == len(vertices) and final_fluid and
            contact_distance is not None):
        face_planes: list[tuple[tuple[float, float, float],
                                tuple[float, float, float]]] = []
        for corner in range(0, len(indices), 3):
            first = final_positions[indices[corner]]
            second = final_positions[indices[corner + 1]]
            third = final_positions[indices[corner + 2]]
            normal = _unit(cross(_subtract(second, first),
                                 _subtract(third, first)))
            if normal is None:
                failures.append(Failure("$.fixture.final_cloth_state",
                                        "cloth triangle became degenerate",
                                        indices[corner:corner + 3]))
                continue
            face_planes.append((first, normal))
        signed_distances = [
            max(dot(_subtract(particle["position"], point), normal)
                for point, normal in face_planes)
            for particle in final_fluid
        ] if face_planes else []
        if signed_distances:
            maximum_signed_distance = max(signed_distances)
            leaked_particles = sum(distance > state_tolerance
                                   for distance in signed_distances)
            maximum_penetration = max(
                0.0, contact_distance + maximum_signed_distance)
            leak_limit = tolerances.get("maximum_leaked_particles", 0)
            if leaked_particles > leak_limit:
                failures.append(Failure("$.fixture.final_fluid_state",
                                        "fluid leaked outside the closed cloth",
                                        leaked_particles, leak_limit))
            penetration_limit = float(tolerances.get(
                "maximum_containment_penetration_m", 0.0))
            if maximum_penetration > penetration_limit:
                failures.append(Failure("$.fixture.final_fluid_state",
                                        "fluid penetrated the containment envelope",
                                        maximum_penetration, penetration_limit))

    fluid_force = cloth_force = residual = None
    reaction_activity = None
    density = _scalar(definition.get("fluid_rest_density_kg_m3"),
                      "$.profile.fixture.fluid_rest_density_kg_m3", failures)
    particle_volume = _scalar(definition.get("fluid_rest_particle_volume_m3"),
                              "$.profile.fixture.fluid_rest_particle_volume_m3",
                              failures)
    if density is not None and particle_volume is not None and final_fluid:
        mass = density * particle_volume
        fluid_force = tuple(mass * component for component in _sum(
            [state["acceleration"] for state in final_fluid]))
    if final_cloth:
        cloth_force = _sum([state["force"] for state in final_cloth])
    if fluid_force is not None and cloth_force is not None:
        residual_vector = tuple(a + b for a, b in zip(fluid_force, cloth_force))
        residual = _norm(residual_vector)
        reaction_activity = max(_norm(fluid_force), _norm(cloth_force))
        activity_limit = float(tolerances.get("minimum_reaction_force_n", 0.0))
        if reaction_activity < activity_limit:
            failures.append(Failure("$.fixture.final_cloth_state",
                                    "reciprocal reaction is too small",
                                    reaction_activity, activity_limit))
        residual_limit = float(tolerances.get(
            "maximum_reciprocal_force_residual_n", 0.0))
        if residual > residual_limit:
            failures.append(Failure("$.fixture.final_cloth_state",
                                    "fluid and cloth reaction forces are not reciprocal",
                                    residual, residual_limit))

    maximum_speed = None
    if final_fluid:
        maximum_speed = max(_norm(state["velocity"]) for state in final_fluid)
        speed_limit = float(definition.get("fluid_maximum_speed_m_s", 0.0)) + float(
            tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure("$.fixture.final_fluid_state",
                                    "fluid speed exceeds authored maximum",
                                    maximum_speed, speed_limit))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": len(final_fluid),
        "failure_count": len(failures),
        "observations": {
            "closed_consistently_oriented_manifold": manifold,
            "initial_signed_volume_m3": initial_volume,
            "final_signed_volume_m3": final_volume,
            "cloth_volume_relative_error": volume_relative_error,
            "maximum_particle_signed_distance_m": maximum_signed_distance,
            "maximum_containment_penetration_m": maximum_penetration,
            "leaked_particle_count": leaked_particles,
            "fluid_reaction_force_n": list(fluid_force)
                if fluid_force is not None else None,
            "cloth_reaction_force_n": list(cloth_force)
                if cloth_force is not None else None,
            "reciprocal_force_residual_n": residual,
            "reaction_activity_n": reaction_activity,
            "maximum_fluid_speed_m_s": maximum_speed,
        },
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_fluid_soft_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture",
                                "expected a fixture object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected a fixture object", actual))
        actual = {}
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    for field in ("id", "substeps", "fluid_particle_count",
                  "soft_solver_iterations", "coupling_solver_iterations",
                  "two_way",
                  "minimum_binding_incidence_per_node", "oracle"):
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    actual.get(field), definition.get(field)))
    for field in (
            "timestep_seconds", "fluid_rest_density_kg_m3",
            "fluid_particle_radius_m", "fluid_support_radius_m",
            "fluid_rest_particle_volume_m3", "fluid_maximum_speed_m_s",
            "soft_node_mass_kg", "soft_node_radius_m",
            "soft_stretch_compliance_m_per_n", "soft_velocity_damping",
            "soft_spring_damping", "soft_contact_friction",
            "soft_shape_matching_stiffness",
            "soft_maximum_projection_fraction",
            "soft_constraint_velocity_response", "soft_maximum_speed_m_s",
            "contact_distance_m", "friction"):
        value = _scalar(actual.get(field), f"$.fixture.{field}", failures)
        expected = _scalar(definition.get(field),
                           f"$.profile.fixture.{field}", failures)
        if (value is not None and expected is not None and
                abs(value - expected) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    value, expected))

    def vectors(value: Any, path: str) -> list[tuple[float, float, float]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a vector array", value))
            return []
        parsed: list[tuple[float, float, float]] = []
        for index, item in enumerate(value):
            vector = _vector(item, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    def compare_vectors(field: str) -> list[tuple[float, float, float]]:
        values = vectors(actual.get(field), f"$.fixture.{field}")
        expected = vectors(definition.get(field),
                           f"$.profile.fixture.{field}")
        if len(values) != len(expected):
            failures.append(Failure(f"$.fixture.{field}",
                                    "vector count differs from profile",
                                    len(values), len(expected)))
        else:
            for index, (value, wanted) in enumerate(zip(values, expected)):
                if max(abs(a - b) for a, b in zip(value, wanted)) > state_tolerance:
                    failures.append(Failure(
                        f"$.fixture.{field}[{index}]",
                        "public input differs from profile",
                        list(value), list(wanted)))
        return values

    gravity = _vector(actual.get("gravity_m_s2"),
                      "$.fixture.gravity_m_s2", failures)
    expected_gravity = _vector(definition.get("gravity_m_s2"),
                               "$.profile.fixture.gravity_m_s2", failures)
    if (gravity is not None and expected_gravity is not None and
            max(abs(a - b) for a, b in zip(gravity, expected_gravity)) >
            state_tolerance):
        failures.append(Failure("$.fixture.gravity_m_s2",
                                "fixture setting differs from profile",
                                list(gravity), list(expected_gravity)))

    nodes = compare_vectors("soft_nodes_m")
    surface_vertices = compare_vectors("surface_vertices_m")
    indices_value = actual.get("surface_triangle_indices")
    if indices_value != definition.get("surface_triangle_indices"):
        failures.append(Failure("$.fixture.surface_triangle_indices",
                                "surface topology differs from profile",
                                indices_value,
                                definition.get("surface_triangle_indices")))
    indices = indices_value if (isinstance(indices_value, list) and
                                all(isinstance(item, int) and
                                    not isinstance(item, bool)
                                    for item in indices_value)) else []
    if not indices or len(indices) % 3 != 0:
        failures.append(Failure("$.fixture.surface_triangle_indices",
                                "expected complete triangle indices", indices))

    inverse_value = actual.get("soft_inverse_masses_kg_inverse")
    inverse_masses: list[float] = []
    expected_inverse = definition.get("soft_inverse_masses_kg_inverse")
    if not isinstance(inverse_value, list):
        failures.append(Failure("$.fixture.soft_inverse_masses_kg_inverse",
                                "expected an inverse-mass array", inverse_value))
    else:
        for index, item in enumerate(inverse_value):
            value = _scalar(item,
                            f"$.fixture.soft_inverse_masses_kg_inverse[{index}]",
                            failures)
            if value is not None:
                if value <= 0.0:
                    failures.append(Failure(
                        f"$.fixture.soft_inverse_masses_kg_inverse[{index}]",
                        "quality fixture requires movable positive-mass nodes",
                        value, "> 0"))
                inverse_masses.append(value)
    if (not isinstance(expected_inverse, list) or
            len(inverse_masses) != len(expected_inverse)):
        failures.append(Failure("$.fixture.soft_inverse_masses_kg_inverse",
                                "inverse-mass count differs from profile",
                                len(inverse_masses),
                                len(expected_inverse) if
                                isinstance(expected_inverse, list) else None))
    else:
        for index, (value, wanted) in enumerate(zip(inverse_masses,
                                                     expected_inverse)):
            if (not _number(wanted) or
                    abs(value - float(wanted)) > state_tolerance):
                failures.append(Failure(
                    f"$.fixture.soft_inverse_masses_kg_inverse[{index}]",
                    "inverse mass differs from profile", value, wanted))

    bonds_value = actual.get("soft_bonds")
    expected_bonds = definition.get("soft_bonds")
    if (not isinstance(bonds_value, list) or
            not isinstance(expected_bonds, list) or
            len(bonds_value) != len(expected_bonds)):
        failures.append(Failure("$.fixture.soft_bonds",
                                "bond count differs from profile",
                                len(bonds_value) if isinstance(bonds_value, list)
                                else None,
                                len(expected_bonds) if
                                isinstance(expected_bonds, list) else None))
    else:
        for index, (bond, wanted) in enumerate(zip(bonds_value, expected_bonds)):
            path = f"$.fixture.soft_bonds[{index}]"
            if not isinstance(bond, dict) or not isinstance(wanted, dict):
                failures.append(Failure(path, "expected a bond object", bond))
                continue
            if (bond.get("first"), bond.get("second")) != (
                    wanted.get("first"), wanted.get("second")):
                failures.append(Failure(path,
                                        "bond endpoints differ from profile",
                                        bond, wanted))
            length = _scalar(bond.get("rest_length_m"),
                             path + ".rest_length_m", failures)
            wanted_length = _scalar(wanted.get("rest_length_m"),
                                    f"$.profile.fixture.soft_bonds[{index}].rest_length_m",
                                    failures)
            if (length is not None and wanted_length is not None and
                    abs(length - wanted_length) > state_tolerance):
                failures.append(Failure(path + ".rest_length_m",
                                        "bond length differs from profile",
                                        length, wanted_length))

    bindings_value = actual.get("surface_bindings")
    expected_bindings = definition.get("surface_bindings")
    bindings: list[list[tuple[int, float]]] = []
    incidence = [0 for _ in nodes]
    if not isinstance(bindings_value, list):
        failures.append(Failure("$.fixture.surface_bindings",
                                "expected a binding array", bindings_value))
        bindings_value = []
    if (not isinstance(expected_bindings, list) or
            len(bindings_value) != len(expected_bindings) or
            len(bindings_value) != len(surface_vertices)):
        failures.append(Failure("$.fixture.surface_bindings",
                                "binding count differs from profile/surface",
                                len(bindings_value), len(surface_vertices)))
    for index, item in enumerate(bindings_value):
        path = f"$.fixture.surface_bindings[{index}]"
        if not isinstance(item, dict):
            failures.append(Failure(path, "expected a binding object", item))
            bindings.append([])
            continue
        node_slots, weights = item.get("nodes"), item.get("weights")
        if (not isinstance(node_slots, list) or len(node_slots) != 4 or
                not isinstance(weights, list) or len(weights) != 4):
            failures.append(Failure(path,
                                    "binding needs four nodes and weights", item))
            bindings.append([])
            continue
        parsed: list[tuple[int, float]] = []
        for slot, (node, weight) in enumerate(zip(node_slots, weights)):
            if (not isinstance(node, int) or isinstance(node, bool) or
                    node < 0 or node >= len(nodes) or not _number(weight) or
                    float(weight) < 0.0):
                failures.append(Failure(f"{path}[{slot}]",
                                        "invalid node/weight binding",
                                        [node, weight]))
                continue
            parsed.append((node, float(weight)))
            if float(weight) > 0.0:
                incidence[node] += 1
        if abs(sum(weight for _, weight in parsed) - 1.0) > state_tolerance:
            failures.append(Failure(path + ".weights",
                                    "binding weights must sum to one",
                                    sum(weight for _, weight in parsed), 1.0))
        bindings.append(parsed)
        if (isinstance(expected_bindings, list) and
                index < len(expected_bindings) and
                item != expected_bindings[index]):
            failures.append(Failure(path,
                                    "binding differs from profile", item,
                                    expected_bindings[index]))
    minimum_incidence = min(incidence) if incidence else 0
    required_incidence = definition.get("minimum_binding_incidence_per_node")
    if (isinstance(required_incidence, int) and
            minimum_incidence < required_incidence):
        failures.append(Failure("$.fixture.surface_bindings",
                                "bound-node incidence is too small to detect mass multiplication",
                                minimum_incidence, required_incidence))

    def particle_states(value: Any, path: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a particle array", value))
            return []
        parsed: list[dict[str, Any]] = []
        for index, particle in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(item_path,
                                        "expected a particle object", particle))
                continue
            if particle.get("stable_id") != index:
                failures.append(Failure(item_path + ".stable_id",
                                        "fluid stable ID/order changed",
                                        particle.get("stable_id"), index))
            position = _vector(particle.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(particle.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            acceleration = _vector(particle.get("acceleration_m_s2"),
                                   item_path + ".acceleration_m_s2", failures)
            if position is not None and velocity is not None and acceleration is not None:
                parsed.append({"position": position, "velocity": velocity,
                               "acceleration": acceleration})
        return parsed

    def node_states(value: Any, path: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a soft-node array", value))
            return []
        parsed: list[dict[str, Any]] = []
        for index, node in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(node, dict):
                failures.append(Failure(item_path,
                                        "expected a soft-node object", node))
                continue
            if node.get("index") != index:
                failures.append(Failure(item_path + ".index",
                                        "soft-node order/index changed",
                                        node.get("index"), index))
            position = _vector(node.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(node.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            force = _vector(node.get("liquid_contact_force_n"),
                            item_path + ".liquid_contact_force_n", failures)
            if position is not None and velocity is not None and force is not None:
                parsed.append({"position": position, "velocity": velocity,
                               "force": force})
        return parsed

    initial_fluid = particle_states(actual.get("initial_fluid_state"),
                                    "$.fixture.initial_fluid_state")
    final_fluid = particle_states(actual.get("final_fluid_state"),
                                  "$.fixture.final_fluid_state")
    initial_nodes = node_states(actual.get("initial_soft_node_state"),
                                "$.fixture.initial_soft_node_state")
    final_nodes = node_states(actual.get("final_soft_node_state"),
                              "$.fixture.final_soft_node_state")
    initial_surface = vectors(actual.get("initial_surface_state_m"),
                              "$.fixture.initial_surface_state_m")
    final_surface = vectors(actual.get("final_surface_state_m"),
                            "$.fixture.final_surface_state_m")
    expected_count = definition.get("fluid_particle_count")
    if isinstance(expected_count, int):
        for path, states in (("initial", initial_fluid),
                             ("final", final_fluid)):
            if len(states) != expected_count:
                failures.append(Failure(f"$.fixture.{path}_fluid_state",
                                        "fluid particle count differs",
                                        len(states), expected_count))
    for path, states in (("initial", initial_nodes), ("final", final_nodes)):
        if len(states) != len(nodes):
            failures.append(Failure(f"$.fixture.{path}_soft_node_state",
                                    "soft-node count differs",
                                    len(states), len(nodes)))
    for path, states in (("initial", initial_surface),
                         ("final", final_surface)):
        if len(states) != len(surface_vertices):
            failures.append(Failure(f"$.fixture.{path}_surface_state_m",
                                    "surface vertex count differs",
                                    len(states), len(surface_vertices)))

    initial_profile = definition.get("initial_particles")
    if (isinstance(initial_profile, list) and initial_fluid and
            len(initial_profile) == len(initial_fluid)):
        for index, (state, wanted) in enumerate(zip(initial_fluid,
                                                     initial_profile)):
            wanted_position = _vector(
                wanted.get("position_m") if isinstance(wanted, dict) else None,
                f"$.profile.fixture.initial_particles[{index}].position_m",
                failures)
            wanted_velocity = _vector(
                wanted.get("velocity_m_s") if isinstance(wanted, dict) else None,
                f"$.profile.fixture.initial_particles[{index}].velocity_m_s",
                failures)
            if (wanted_position is not None and max(abs(a - b) for a, b in zip(
                    state["position"], wanted_position)) > state_tolerance):
                failures.append(Failure(
                    f"$.fixture.initial_fluid_state[{index}].position_m",
                    "initial particle position differs from profile",
                    list(state["position"]), list(wanted_position)))
            if (wanted_velocity is not None and max(abs(a - b) for a, b in zip(
                    state["velocity"], wanted_velocity)) > state_tolerance):
                failures.append(Failure(
                    f"$.fixture.initial_fluid_state[{index}].velocity_m_s",
                    "initial particle velocity differs from profile",
                    list(state["velocity"]), list(wanted_velocity)))

    for index, state in enumerate(initial_nodes):
        if index < len(nodes) and max(abs(a - b) for a, b in zip(
                state["position"], nodes[index])) > state_tolerance:
            failures.append(Failure(
                f"$.fixture.initial_soft_node_state[{index}].position_m",
                "initial soft-node state differs from profile",
                list(state["position"]), list(nodes[index])))
        if _norm(state["velocity"]) > state_tolerance:
            failures.append(Failure(
                f"$.fixture.initial_soft_node_state[{index}].velocity_m_s",
                "initial soft-node velocity must be zero",
                list(state["velocity"]), 0.0))
        if _norm(state["force"]) > state_tolerance:
            failures.append(Failure(
                f"$.fixture.initial_soft_node_state[{index}].liquid_contact_force_n",
                "initial soft-node liquid force must be zero",
                list(state["force"]), 0.0))

    maximum_reconstruction_error = 0.0
    for phase, node_state, skin in (("initial", initial_nodes, initial_surface),
                                    ("final", final_nodes, final_surface)):
        if len(node_state) != len(nodes) or len(skin) != len(bindings):
            continue
        for vertex, binding in enumerate(bindings):
            reconstructed = tuple(sum(node_state[node]["position"][axis] * weight
                                      for node, weight in binding)
                                  for axis in range(3))
            error = _norm(_subtract(skin[vertex], reconstructed))
            maximum_reconstruction_error = max(maximum_reconstruction_error,
                                               error)
            limit = float(tolerances.get(
                "maximum_surface_reconstruction_error_m", 0.0))
            if error > limit:
                failures.append(Failure(
                    f"$.fixture.{phase}_surface_state_m[{vertex}]",
                    "public soft surface does not match its node bindings",
                    error, limit))

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    manifold = bool(indices and surface_vertices)
    edge_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for corner in range(0, len(indices), 3):
        triangle = indices[corner:corner + 3]
        if (len(triangle) != 3 or len(set(triangle)) != 3 or
                any(index < 0 or index >= len(surface_vertices)
                    for index in triangle)):
            manifold = False
            failures.append(Failure(
                f"$.fixture.surface_triangle_indices[{corner}]",
                "invalid surface triangle", triangle))
            continue
        for offset in range(3):
            first, second = triangle[offset], triangle[(offset + 1) % 3]
            edge_directions.setdefault(tuple(sorted((first, second))), []).append(
                (first, second))
    for edge, directions in edge_directions.items():
        if len(directions) != 2 or directions[0] != directions[1][::-1]:
            manifold = False
            failures.append(Failure("$.fixture.surface_triangle_indices",
                                    "soft surface is not closed and consistently oriented",
                                    {"edge": edge, "directions": directions}))

    maximum_overlap = None
    final_signed_distance = None
    contact_normal = None
    if (manifold and final_fluid and
            len(final_surface) == len(surface_vertices)):
        planes: list[tuple[tuple[float, float, float],
                           tuple[float, float, float]]] = []
        for corner in range(0, len(indices), 3):
            first = final_surface[indices[corner]]
            second = final_surface[indices[corner + 1]]
            third = final_surface[indices[corner + 2]]
            normal = _unit(cross(_subtract(second, first),
                                 _subtract(third, first)))
            if normal is None:
                failures.append(Failure("$.fixture.final_surface_state_m",
                                        "soft surface triangle became degenerate",
                                        indices[corner:corner + 3]))
                continue
            planes.append((first, normal))
        if planes:
            distances = [(dot(_subtract(final_fluid[0]["position"], point),
                              normal), normal)
                         for point, normal in planes]
            final_signed_distance, contact_normal = max(
                distances, key=lambda item: item[0])
            maximum_overlap = max(0.0, -final_signed_distance)
            overlap_limit = float(tolerances.get(
                "maximum_overlap_penetration_m", 0.0))
            if maximum_overlap > overlap_limit:
                failures.append(Failure("$.fixture.final_fluid_state",
                                        "fluid overlapped the soft-body surface",
                                        maximum_overlap, overlap_limit))

    fluid_impulse = soft_impulse = impulse_residual = None
    contact_activity = expected_impulse = expected_impulse_error = None
    fluid_force = soft_force = force_residual = None
    force_impulse_residual = None
    density = _scalar(definition.get("fluid_rest_density_kg_m3"),
                      "$.profile.fixture.fluid_rest_density_kg_m3", failures)
    volume = _scalar(definition.get("fluid_rest_particle_volume_m3"),
                     "$.profile.fixture.fluid_rest_particle_volume_m3",
                     failures)
    timestep = _scalar(definition.get("timestep_seconds"),
                       "$.profile.fixture.timestep_seconds", failures)
    if (density is not None and volume is not None and initial_fluid and
            final_fluid):
        fluid_mass = density * volume
        fluid_impulse = tuple(fluid_mass * (after - before)
                              for before, after in zip(
                                  initial_fluid[0]["velocity"],
                                  final_fluid[0]["velocity"]))
        fluid_force = tuple(fluid_mass * component
                            for component in final_fluid[0]["acceleration"])
    if (len(initial_nodes) == len(final_nodes) == len(inverse_masses) and
            inverse_masses):
        soft_impulse = tuple(sum(
            (final_nodes[index]["velocity"][axis] -
             initial_nodes[index]["velocity"][axis]) / inverse_masses[index]
            for index in range(len(inverse_masses))) for axis in range(3))
        soft_force = _sum([node["force"] for node in final_nodes])
    if fluid_impulse is not None and soft_impulse is not None:
        impulse_residual = _norm(tuple(a + b for a, b in zip(
            fluid_impulse, soft_impulse)))
        contact_activity = max(_norm(fluid_impulse), _norm(soft_impulse))
        minimum_activity = float(tolerances.get(
            "minimum_contact_impulse_n_s", 0.0))
        if contact_activity < minimum_activity:
            failures.append(Failure("$.fixture.final_soft_node_state",
                                    "contact impulse is too small",
                                    contact_activity, minimum_activity))
        residual_limit = float(tolerances.get(
            "maximum_reciprocal_impulse_residual_n_s", 0.0))
        if impulse_residual > residual_limit:
            failures.append(Failure("$.fixture.final_soft_node_state",
                                    "fluid and soft-body impulses are not reciprocal",
                                    impulse_residual, residual_limit))
    if fluid_force is not None and soft_force is not None:
        force_residual = _norm(tuple(a + b for a, b in zip(
            fluid_force, soft_force)))
        force_limit = float(tolerances.get(
            "maximum_reciprocal_force_residual_n", 0.0))
        if force_residual > force_limit:
            failures.append(Failure("$.fixture.final_soft_node_state",
                                    "fluid and soft-body forces are not reciprocal",
                                    force_residual, force_limit))
        if soft_impulse is not None and timestep is not None:
            force_impulse_residual = _norm(tuple(
                impulse - force * timestep
                for impulse, force in zip(soft_impulse, soft_force)))
            consistency_limit = float(tolerances.get(
                "maximum_node_force_impulse_residual_n_s", 0.0))
            if force_impulse_residual > consistency_limit:
                failures.append(Failure(
                    "$.fixture.final_soft_node_state",
                    "soft-node masses do not map force to impulse once per bound node",
                    force_impulse_residual, consistency_limit))

    expected_weights_value = actual.get("expected_contact_node_weights")
    expected_weights_profile = definition.get("expected_contact_node_weights")
    expected_weights: list[float] = []
    if (not isinstance(expected_weights_value, list) or
            not isinstance(expected_weights_profile, list) or
            len(expected_weights_value) != len(inverse_masses)):
        failures.append(Failure("$.fixture.expected_contact_node_weights",
                                "contact weights differ from profile/nodes",
                                expected_weights_value,
                                expected_weights_profile))
    else:
        for index, (value, wanted) in enumerate(zip(
                expected_weights_value, expected_weights_profile)):
            if not _number(value) or not _number(wanted):
                failures.append(Failure(
                    f"$.fixture.expected_contact_node_weights[{index}]",
                    "expected a finite contact weight", value, wanted))
                continue
            parsed = float(value)
            expected_weights.append(parsed)
            if abs(parsed - float(wanted)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.expected_contact_node_weights[{index}]",
                    "contact weight differs from profile", parsed, wanted))
    if (len(expected_weights) == len(inverse_masses) and initial_fluid and
            fluid_impulse is not None and manifold and
            len(initial_surface) == len(surface_vertices)):
        initial_planes: list[tuple[float, tuple[float, float, float]]] = []
        for corner in range(0, len(indices), 3):
            first = initial_surface[indices[corner]]
            normal = _unit(cross(
                _subtract(initial_surface[indices[corner + 1]], first),
                _subtract(initial_surface[indices[corner + 2]], first)))
            if normal is not None:
                initial_planes.append((dot(_subtract(
                    initial_fluid[0]["position"], first), normal), normal))
        if initial_planes:
            _, initial_normal = max(initial_planes, key=lambda item: item[0])
            surface_velocity = tuple(sum(
                expected_weights[index] * initial_nodes[index]["velocity"][axis]
                for index in range(len(expected_weights))) for axis in range(3))
            relative_velocity = _subtract(initial_fluid[0]["velocity"],
                                          surface_velocity)
            closing_speed = max(0.0, -dot(relative_velocity, initial_normal))
            fluid_mass = float(density) * float(volume)
            effective_inverse_mass = sum(
                weight * inverse_mass for weight, inverse_mass in zip(
                    expected_weights, inverse_masses))
            denominator = 1.0 / fluid_mass + effective_inverse_mass
            response_inverse_mass = 1.0 / fluid_mass + sum(
                weight * weight * inverse_mass
                for weight, inverse_mass in zip(
                    expected_weights, inverse_masses))
            passes = definition.get("coupling_solver_iterations")
            if (not isinstance(passes, int) or isinstance(passes, bool) or
                    passes <= 0):
                failures.append(Failure(
                    "$.profile.fixture.coupling_solver_iterations",
                    "expected a positive solver pass count", passes))
                passes = 1
            remaining_fraction = 1.0 - response_inverse_mass / denominator
            expected_impulse = (closing_speed / response_inverse_mass) * (
                1.0 - remaining_fraction ** passes)
            expected_vector = tuple(component * expected_impulse
                                    for component in initial_normal)
            expected_impulse_error = _norm(_subtract(fluid_impulse,
                                                      expected_vector))
            expected_limit = float(tolerances.get(
                "maximum_expected_impulse_error_n_s", 0.0))
            if expected_impulse_error > expected_limit:
                failures.append(Failure(
                    "$.fixture.final_fluid_state",
                    "contact impulse does not use each uniquely bound node mass once",
                    expected_impulse_error, expected_limit))

    maximum_speed = None
    speeds = ([_norm(state["velocity"]) for state in final_fluid] +
              [_norm(state["velocity"]) for state in final_nodes])
    if speeds:
        maximum_speed = max(speeds)
        speed_limit = max(
            float(definition.get("fluid_maximum_speed_m_s", 0.0)),
            float(definition.get("soft_maximum_speed_m_s", 0.0))) + float(
                tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure("$.fixture",
                                    "contact speed exceeds authored maximum",
                                    maximum_speed, speed_limit))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": len(final_fluid),
        "failure_count": len(failures),
        "observations": {
            "closed_consistently_oriented_surface": manifold,
            "minimum_binding_incidence_per_node": minimum_incidence,
            "maximum_surface_reconstruction_error_m":
                maximum_reconstruction_error,
            "final_particle_signed_distance_m": final_signed_distance,
            "maximum_overlap_penetration_m": maximum_overlap,
            "fluid_contact_impulse_n_s": list(fluid_impulse)
                if fluid_impulse is not None else None,
            "soft_body_contact_impulse_n_s": list(soft_impulse)
                if soft_impulse is not None else None,
            "reciprocal_impulse_residual_n_s": impulse_residual,
            "contact_impulse_activity_n_s": contact_activity,
            "expected_unique_bound_node_impulse_n_s": expected_impulse,
            "expected_impulse_error_n_s": expected_impulse_error,
            "fluid_reaction_force_n": list(fluid_force)
                if fluid_force is not None else None,
            "soft_body_reaction_force_n": list(soft_force)
                if soft_force is not None else None,
            "reciprocal_force_residual_n": force_residual,
            "soft_force_impulse_residual_n_s": force_impulse_residual,
            "maximum_endpoint_speed_m_s": maximum_speed,
        },
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_cloth_soft_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture",
                                "expected a fixture object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected a fixture object", actual))
        actual = {}
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    for field in ("id", "substeps", "cloth_solver_iterations",
                  "cloth_preserve_volume", "soft_solver_iterations",
                  "coupling_solver_iterations", "two_way",
                  "minimum_binding_incidence_per_node",
                  "contact_soft_node_index", "oracle"):
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    actual.get(field), definition.get(field)))
    scalar_fields = (
        "preparation_timestep_seconds", "timestep_seconds",
        "cloth_vertex_mass_kg", "cloth_thickness_m",
        "cloth_stretch_compliance_m_per_n",
        "cloth_bending_compliance_m_per_n", "cloth_velocity_damping",
        "cloth_contact_friction", "soft_node_mass_kg",
        "soft_node_radius_m", "soft_stretch_compliance_m_per_n",
        "soft_velocity_damping", "soft_spring_damping",
        "soft_contact_friction", "soft_shape_matching_stiffness",
        "soft_maximum_projection_fraction",
        "soft_constraint_velocity_response", "soft_maximum_speed_m_s",
        "contact_distance_m", "friction")
    for field in scalar_fields:
        value = _scalar(actual.get(field), f"$.fixture.{field}", failures)
        wanted = _scalar(definition.get(field),
                         f"$.profile.fixture.{field}", failures)
        if (value is not None and wanted is not None and
                abs(value - wanted) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    value, wanted))

    def vectors(value: Any, path: str) -> list[tuple[float, float, float]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a vector array", value))
            return []
        parsed: list[tuple[float, float, float]] = []
        for index, item in enumerate(value):
            vector = _vector(item, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    def compare_vectors(field: str) -> list[tuple[float, float, float]]:
        values = vectors(actual.get(field), f"$.fixture.{field}")
        wanted = vectors(definition.get(field),
                         f"$.profile.fixture.{field}")
        if len(values) != len(wanted):
            failures.append(Failure(f"$.fixture.{field}",
                                    "vector count differs from profile",
                                    len(values), len(wanted)))
        else:
            for index, (value, expected) in enumerate(zip(values, wanted)):
                if _norm(_subtract(value, expected)) > state_tolerance:
                    failures.append(Failure(
                        f"$.fixture.{field}[{index}]",
                        "public input differs from profile",
                        list(value), list(expected)))
        return values

    def compare_vector(field: str) -> tuple[float, float, float] | None:
        value = _vector(actual.get(field), f"$.fixture.{field}", failures)
        wanted = _vector(definition.get(field),
                         f"$.profile.fixture.{field}", failures)
        if (value is not None and wanted is not None and
                _norm(_subtract(value, wanted)) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    list(value), list(wanted)))
        return value

    compare_vector("preparation_gravity_m_s2")
    compare_vector("gravity_m_s2")
    prepared_velocity = compare_vector("prepared_soft_velocity_m_s")
    cloth_vertices = compare_vectors("cloth_vertices_m")
    authored_nodes = compare_vectors("authored_soft_nodes_m")
    prepared_nodes = compare_vectors("prepared_soft_nodes_m")
    surface_vertices = compare_vectors("surface_vertices_m")

    def compare_float_array(field: str) -> list[float]:
        values = actual.get(field)
        wanted = definition.get(field)
        parsed: list[float] = []
        if not isinstance(values, list) or not isinstance(wanted, list):
            failures.append(Failure(f"$.fixture.{field}",
                                    "expected an array", values, wanted))
            return parsed
        if len(values) != len(wanted):
            failures.append(Failure(f"$.fixture.{field}",
                                    "array count differs from profile",
                                    len(values), len(wanted)))
        for index, item in enumerate(values):
            value = _scalar(item, f"$.fixture.{field}[{index}]", failures)
            if value is None:
                continue
            parsed.append(value)
            if (index >= len(wanted) or not _number(wanted[index]) or
                    abs(value - float(wanted[index])) > state_tolerance):
                failures.append(Failure(
                    f"$.fixture.{field}[{index}]",
                    "array value differs from profile", value,
                    wanted[index] if index < len(wanted) else None))
        return parsed

    cloth_inverse = compare_float_array(
        "cloth_inverse_masses_kg_inverse")
    soft_inverse = compare_float_array(
        "soft_inverse_masses_kg_inverse")
    expected_weights = compare_float_array(
        "expected_cloth_contact_weights")
    for field, values in (("cloth_inverse_masses_kg_inverse", cloth_inverse),
                          ("soft_inverse_masses_kg_inverse", soft_inverse)):
        for index, value in enumerate(values):
            if value <= 0.0:
                failures.append(Failure(f"$.fixture.{field}[{index}]",
                                        "quality fixture requires movable positive-mass endpoints",
                                        value, "> 0"))

    cloth_indices = actual.get("cloth_triangle_indices")
    if cloth_indices != definition.get("cloth_triangle_indices"):
        failures.append(Failure("$.fixture.cloth_triangle_indices",
                                "cloth topology differs from profile",
                                cloth_indices,
                                definition.get("cloth_triangle_indices")))
    if (not isinstance(cloth_indices, list) or len(cloth_indices) != 3 or
            any(not isinstance(item, int) or isinstance(item, bool)
                for item in cloth_indices)):
        failures.append(Failure("$.fixture.cloth_triangle_indices",
                                "expected one complete cloth triangle",
                                cloth_indices))
        cloth_indices = []
    surface_indices = actual.get("surface_triangle_indices")
    if surface_indices != definition.get("surface_triangle_indices"):
        failures.append(Failure("$.fixture.surface_triangle_indices",
                                "soft surface topology differs from profile",
                                surface_indices,
                                definition.get("surface_triangle_indices")))
    if (not isinstance(surface_indices, list) or
            len(surface_indices) == 0 or len(surface_indices) % 3 != 0):
        failures.append(Failure("$.fixture.surface_triangle_indices",
                                "expected complete surface triangles",
                                surface_indices))

    bonds_value = actual.get("soft_bonds")
    expected_bonds = definition.get("soft_bonds")
    if (not isinstance(bonds_value, list) or
            not isinstance(expected_bonds, list) or
            len(bonds_value) != len(expected_bonds)):
        failures.append(Failure("$.fixture.soft_bonds",
                                "bond count differs from profile",
                                len(bonds_value) if isinstance(bonds_value, list)
                                else None,
                                len(expected_bonds) if
                                isinstance(expected_bonds, list) else None))
    else:
        for index, (bond, wanted) in enumerate(zip(bonds_value,
                                                   expected_bonds)):
            path = f"$.fixture.soft_bonds[{index}]"
            if not isinstance(bond, dict) or not isinstance(wanted, dict):
                failures.append(Failure(path, "expected a bond object", bond))
                continue
            if (bond.get("first"), bond.get("second")) != (
                    wanted.get("first"), wanted.get("second")):
                failures.append(Failure(path,
                                        "bond endpoints differ from profile",
                                        bond, wanted))
            length = _scalar(bond.get("rest_length_m"),
                             path + ".rest_length_m", failures)
            wanted_length = _scalar(
                wanted.get("rest_length_m"),
                f"$.profile.fixture.soft_bonds[{index}].rest_length_m",
                failures)
            if (length is not None and wanted_length is not None and
                    abs(length - wanted_length) > state_tolerance):
                failures.append(Failure(path + ".rest_length_m",
                                        "bond length differs from profile",
                                        length, wanted_length))

    bindings_value = actual.get("surface_bindings")
    expected_bindings = definition.get("surface_bindings")
    bindings: list[list[tuple[int, float]]] = []
    incidence = [0 for _ in authored_nodes]
    if not isinstance(bindings_value, list):
        failures.append(Failure("$.fixture.surface_bindings",
                                "expected a binding array", bindings_value))
        bindings_value = []
    if (not isinstance(expected_bindings, list) or
            len(bindings_value) != len(expected_bindings) or
            len(bindings_value) != len(surface_vertices)):
        failures.append(Failure("$.fixture.surface_bindings",
                                "binding count differs from profile/surface",
                                len(bindings_value), len(surface_vertices)))
    for index, item in enumerate(bindings_value):
        path = f"$.fixture.surface_bindings[{index}]"
        if not isinstance(item, dict):
            failures.append(Failure(path, "expected a binding object", item))
            bindings.append([])
            continue
        nodes_value, weights_value = item.get("nodes"), item.get("weights")
        if (not isinstance(nodes_value, list) or len(nodes_value) != 4 or
                not isinstance(weights_value, list) or len(weights_value) != 4):
            failures.append(Failure(path,
                                    "binding needs four nodes and weights", item))
            bindings.append([])
            continue
        parsed: list[tuple[int, float]] = []
        for slot, (node, weight) in enumerate(zip(nodes_value, weights_value)):
            if (not isinstance(node, int) or isinstance(node, bool) or
                    node < 0 or node >= len(authored_nodes) or
                    not _number(weight) or float(weight) < 0.0):
                failures.append(Failure(f"{path}[{slot}]",
                                        "invalid node/weight binding",
                                        [node, weight]))
                continue
            parsed.append((node, float(weight)))
            if float(weight) > 0.0:
                incidence[node] += 1
        if abs(sum(weight for _, weight in parsed) - 1.0) > state_tolerance:
            failures.append(Failure(path + ".weights",
                                    "binding weights must sum to one",
                                    sum(weight for _, weight in parsed), 1.0))
        bindings.append(parsed)
        if (isinstance(expected_bindings, list) and
                index < len(expected_bindings) and
                item != expected_bindings[index]):
            failures.append(Failure(path, "binding differs from profile",
                                    item, expected_bindings[index]))
    minimum_incidence = min(incidence) if incidence else 0
    required_incidence = definition.get("minimum_binding_incidence_per_node")
    if (isinstance(required_incidence, int) and
            minimum_incidence < required_incidence):
        failures.append(Failure(
            "$.fixture.surface_bindings",
            "bound-node incidence is too small to detect mass multiplication",
            minimum_incidence, required_incidence))

    def endpoint_states(value: Any, path: str, force_field: str,
                        label: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, f"expected a {label} array", value))
            return []
        parsed: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(item, dict):
                failures.append(Failure(item_path,
                                        f"expected a {label} object", item))
                continue
            if item.get("index") != index:
                failures.append(Failure(item_path + ".index",
                                        f"{label} order/index changed",
                                        item.get("index"), index))
            position = _vector(item.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(item.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            force = _vector(item.get(force_field),
                            item_path + f".{force_field}", failures)
            if position is not None and velocity is not None and force is not None:
                parsed.append({"position": position, "velocity": velocity,
                               "force": force})
        return parsed

    initial_cloth = endpoint_states(
        actual.get("initial_cloth_state"), "$.fixture.initial_cloth_state",
        "soft_body_contact_force_n", "cloth vertex")
    final_cloth = endpoint_states(
        actual.get("final_cloth_state"), "$.fixture.final_cloth_state",
        "soft_body_contact_force_n", "cloth vertex")
    initial_soft = endpoint_states(
        actual.get("initial_soft_node_state"),
        "$.fixture.initial_soft_node_state", "cloth_contact_force_n",
        "soft node")
    final_soft = endpoint_states(
        actual.get("final_soft_node_state"),
        "$.fixture.final_soft_node_state", "cloth_contact_force_n",
        "soft node")
    initial_surface = vectors(actual.get("initial_surface_state_m"),
                              "$.fixture.initial_surface_state_m")
    final_surface = vectors(actual.get("final_surface_state_m"),
                            "$.fixture.final_surface_state_m")
    for path, states, wanted in (
            ("initial_cloth_state", initial_cloth, cloth_vertices),
            ("final_cloth_state", final_cloth, cloth_vertices),
            ("initial_soft_node_state", initial_soft, authored_nodes),
            ("final_soft_node_state", final_soft, authored_nodes)):
        if len(states) != len(wanted):
            failures.append(Failure(f"$.fixture.{path}",
                                    "endpoint state count differs",
                                    len(states), len(wanted)))
    for path, states in (("initial_surface_state_m", initial_surface),
                         ("final_surface_state_m", final_surface)):
        if len(states) != len(surface_vertices):
            failures.append(Failure(f"$.fixture.{path}",
                                    "surface state count differs",
                                    len(states), len(surface_vertices)))

    if len(initial_cloth) == len(cloth_vertices):
        for index, (state, wanted) in enumerate(zip(initial_cloth,
                                                     cloth_vertices)):
            if _norm(_subtract(state["position"], wanted)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_cloth_state[{index}].position_m",
                    "initial cloth state differs from profile",
                    list(state["position"]), list(wanted)))
            if _norm(state["velocity"]) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_cloth_state[{index}].velocity_m_s",
                    "initial cloth velocity must be zero",
                    list(state["velocity"]), 0.0))
            if _norm(state["force"]) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_cloth_state[{index}].soft_body_contact_force_n",
                    "initial cloth contact force must be zero",
                    list(state["force"]), 0.0))
    if (len(initial_soft) == len(prepared_nodes) and
            prepared_velocity is not None):
        for index, (state, wanted) in enumerate(zip(initial_soft,
                                                     prepared_nodes)):
            if _norm(_subtract(state["position"], wanted)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_soft_node_state[{index}].position_m",
                    "prepared soft-node state differs from profile",
                    list(state["position"]), list(wanted)))
            if (_norm(_subtract(state["velocity"], prepared_velocity)) >
                    state_tolerance):
                failures.append(Failure(
                    f"$.fixture.initial_soft_node_state[{index}].velocity_m_s",
                    "prepared soft-node velocity differs from profile",
                    list(state["velocity"]), list(prepared_velocity)))
            if _norm(state["force"]) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial_soft_node_state[{index}].cloth_contact_force_n",
                    "initial soft-node contact force must be zero",
                    list(state["force"]), 0.0))

    maximum_reconstruction_error = 0.0
    for phase, nodes, surface in (
            ("initial", initial_soft, initial_surface),
            ("final", final_soft, final_surface)):
        if len(nodes) != len(authored_nodes) or len(surface) != len(bindings):
            continue
        for vertex, binding in enumerate(bindings):
            reconstructed = tuple(sum(
                nodes[node]["position"][axis] * weight
                for node, weight in binding) for axis in range(3))
            error = _norm(_subtract(surface[vertex], reconstructed))
            maximum_reconstruction_error = max(maximum_reconstruction_error,
                                               error)
            limit = float(tolerances.get(
                "maximum_surface_reconstruction_error_m", 0.0))
            if error > limit:
                failures.append(Failure(
                    f"$.fixture.{phase}_surface_state_m[{vertex}]",
                    "public soft surface does not match its node bindings",
                    error, limit))

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def closest_triangle(point: tuple[float, float, float],
                         first: tuple[float, float, float],
                         second: tuple[float, float, float],
                         third: tuple[float, float, float]
                         ) -> tuple[float, float, float]:
        ab, ac, ap = (_subtract(second, first),
                      _subtract(third, first), _subtract(point, first))
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return first
        bp = _subtract(point, second)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return second
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            return add(first, scale(ab, d1 / (d1 - d3)))
        cp = _subtract(point, third)
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return third
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            return add(first, scale(ac, d2 / (d2 - d6)))
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            edge = _subtract(third, second)
            return add(second, scale(
                edge, (d4 - d3) / ((d4 - d3) + (d5 - d6))))
        denominator = 1.0 / (va + vb + vc)
        return add(first, add(scale(ab, vb * denominator),
                              scale(ac, vc * denominator)))

    contact_index = definition.get("contact_soft_node_index")
    final_separation = maximum_overlap = None
    initial_normal = final_normal = None
    if (isinstance(contact_index, int) and
            0 <= contact_index < len(final_soft) and
            len(final_cloth) == 3 and len(initial_cloth) == 3):
        initial_normal = _unit(cross(
            _subtract(initial_cloth[1]["position"],
                      initial_cloth[0]["position"]),
            _subtract(initial_cloth[2]["position"],
                      initial_cloth[0]["position"])))
        closest = closest_triangle(
            final_soft[contact_index]["position"],
            final_cloth[0]["position"], final_cloth[1]["position"],
            final_cloth[2]["position"])
        delta = _subtract(final_soft[contact_index]["position"], closest)
        final_separation = _norm(delta)
        final_normal = _unit(delta) or initial_normal
        distance = _scalar(definition.get("contact_distance_m"),
                           "$.profile.fixture.contact_distance_m", failures)
        if distance is not None:
            maximum_overlap = max(0.0, distance - final_separation)
            limit = float(tolerances.get("maximum_overlap_penetration_m", 0.0))
            if maximum_overlap > limit:
                failures.append(Failure(
                    "$.fixture.final_soft_node_state",
                    "soft node overlapped the cloth contact envelope",
                    maximum_overlap, limit))

    cloth_impulse = soft_impulse = impulse_residual = None
    cloth_force = soft_force = force_residual = None
    force_impulse_residual = None
    timestep = _scalar(definition.get("timestep_seconds"),
                       "$.profile.fixture.timestep_seconds", failures)
    if (len(initial_cloth) == len(final_cloth) == len(cloth_inverse) and
            cloth_inverse):
        cloth_impulse = tuple(sum(
            (final_cloth[index]["velocity"][axis] -
             initial_cloth[index]["velocity"][axis]) / cloth_inverse[index]
            for index in range(len(cloth_inverse))) for axis in range(3))
        cloth_force = _sum([state["force"] for state in final_cloth])
    if (len(initial_soft) == len(final_soft) == len(soft_inverse) and
            soft_inverse):
        soft_impulse = tuple(sum(
            (final_soft[index]["velocity"][axis] -
             initial_soft[index]["velocity"][axis]) / soft_inverse[index]
            for index in range(len(soft_inverse))) for axis in range(3))
        soft_force = _sum([state["force"] for state in final_soft])
    contact_activity = None
    if cloth_impulse is not None and soft_impulse is not None:
        impulse_residual = _norm(add(cloth_impulse, soft_impulse))
        contact_activity = max(_norm(cloth_impulse), _norm(soft_impulse))
        minimum = float(tolerances.get("minimum_contact_impulse_n_s", 0.0))
        if contact_activity < minimum:
            failures.append(Failure("$.fixture.final_soft_node_state",
                                    "contact impulse is too small",
                                    contact_activity, minimum))
        limit = float(tolerances.get(
            "maximum_reciprocal_impulse_residual_n_s", 0.0))
        if impulse_residual > limit:
            failures.append(Failure("$.fixture.final_soft_node_state",
                                    "cloth and soft-body impulses are not reciprocal",
                                    impulse_residual, limit))
    if cloth_force is not None and soft_force is not None:
        force_residual = _norm(add(cloth_force, soft_force))
        limit = float(tolerances.get(
            "maximum_reciprocal_force_residual_n", 0.0))
        if force_residual > limit:
            failures.append(Failure("$.fixture.final_soft_node_state",
                                    "cloth and soft-body forces are not reciprocal",
                                    force_residual, limit))
        if (cloth_impulse is not None and soft_impulse is not None and
                timestep is not None):
            cloth_error = _norm(_subtract(
                cloth_impulse, scale(cloth_force, timestep)))
            soft_error = _norm(_subtract(
                soft_impulse, scale(soft_force, timestep)))
            force_impulse_residual = max(cloth_error, soft_error)
            consistency_limit = float(tolerances.get(
                "maximum_endpoint_force_impulse_residual_n_s", 0.0))
            if force_impulse_residual > consistency_limit:
                failures.append(Failure(
                    "$.fixture.final_soft_node_state",
                    "endpoint masses do not map public force to impulse",
                    force_impulse_residual, consistency_limit))

    expected_impulse = expected_impulse_error = tangential_impulse = None
    initial_slip = final_slip = final_closing_speed = None
    if (isinstance(contact_index, int) and initial_normal is not None and
            0 <= contact_index < len(initial_soft) and
            contact_index < len(soft_inverse) and
            len(initial_cloth) == len(final_cloth) == len(expected_weights) ==
            len(cloth_inverse) and soft_impulse is not None):
        initial_cloth_velocity = tuple(sum(
            expected_weights[index] * initial_cloth[index]["velocity"][axis]
            for index in range(len(expected_weights))) for axis in range(3))
        initial_relative = _subtract(
            initial_soft[contact_index]["velocity"], initial_cloth_velocity)
        closing_speed = max(0.0, -dot(initial_relative, initial_normal))
        response_inverse_mass = soft_inverse[contact_index] + sum(
            weight * weight * inverse_mass
            for weight, inverse_mass in zip(expected_weights, cloth_inverse))
        expected_impulse = closing_speed / response_inverse_mass
        expected_vector = scale(initial_normal, expected_impulse)
        expected_impulse_error = _norm(_subtract(soft_impulse,
                                                  expected_vector))
        expected_limit = float(tolerances.get(
            "maximum_expected_impulse_error_n_s", 0.0))
        if expected_impulse_error > expected_limit:
            failures.append(Failure(
                "$.fixture.final_soft_node_state",
                "contact impulse does not use the uniquely bound node mass once",
                expected_impulse_error, expected_limit))
        normal_impulse = scale(initial_normal,
                               dot(soft_impulse, initial_normal))
        tangential_impulse = _norm(_subtract(soft_impulse, normal_impulse))
        tangential_limit = float(tolerances.get(
            "maximum_tangential_impulse_n_s", 0.0))
        if tangential_impulse > tangential_limit:
            failures.append(Failure(
                "$.fixture.final_soft_node_state",
                "frictionless contact introduced tangential impulse",
                tangential_impulse, tangential_limit))
        initial_tangent = _subtract(
            initial_relative, scale(initial_normal,
                                    dot(initial_relative, initial_normal)))
        initial_slip = _norm(initial_tangent)
        final_cloth_velocity = tuple(sum(
            expected_weights[index] * final_cloth[index]["velocity"][axis]
            for index in range(len(expected_weights))) for axis in range(3))
        final_relative = _subtract(
            final_soft[contact_index]["velocity"], final_cloth_velocity)
        normal = final_normal or initial_normal
        final_tangent = _subtract(
            final_relative, scale(normal, dot(final_relative, normal)))
        final_slip = _norm(final_tangent)
        slip_minimum = float(tolerances.get(
            "minimum_frictionless_slip_m_s", 0.0))
        if final_slip < slip_minimum:
            failures.append(Failure(
                "$.fixture.final_soft_node_state",
                "cloth and soft body moved as a hidden attachment",
                final_slip, slip_minimum))
        final_closing_speed = max(0.0, -dot(final_relative, normal))
        closing_limit = float(tolerances.get(
            "maximum_final_closing_speed_m_s", 0.0))
        if final_closing_speed > closing_limit:
            failures.append(Failure(
                "$.fixture.final_soft_node_state",
                "cloth-soft endpoints remain closing after contact",
                final_closing_speed, closing_limit))

    speeds = ([_norm(state["velocity"]) for state in final_cloth] +
              [_norm(state["velocity"]) for state in final_soft])
    maximum_speed = max(speeds) if speeds else None
    if maximum_speed is not None:
        speed_limit = max(
            20.0, float(definition.get("soft_maximum_speed_m_s", 0.0))) + float(
                tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure("$.fixture",
                                    "contact speed exceeds authored maximum",
                                    maximum_speed, speed_limit))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": len(final_soft),
        "failure_count": len(failures),
        "observations": {
            "minimum_binding_incidence_per_node": minimum_incidence,
            "maximum_surface_reconstruction_error_m":
                maximum_reconstruction_error,
            "final_contact_separation_m": final_separation,
            "maximum_overlap_penetration_m": maximum_overlap,
            "cloth_contact_impulse_n_s": list(cloth_impulse)
                if cloth_impulse is not None else None,
            "soft_body_contact_impulse_n_s": list(soft_impulse)
                if soft_impulse is not None else None,
            "reciprocal_impulse_residual_n_s": impulse_residual,
            "contact_impulse_activity_n_s": contact_activity,
            "expected_unique_bound_node_impulse_n_s": expected_impulse,
            "expected_impulse_error_n_s": expected_impulse_error,
            "tangential_impulse_n_s": tangential_impulse,
            "initial_frictionless_slip_m_s": initial_slip,
            "final_frictionless_slip_m_s": final_slip,
            "final_closing_speed_m_s": final_closing_speed,
            "cloth_reaction_force_n": list(cloth_force)
                if cloth_force is not None else None,
            "soft_body_reaction_force_n": list(soft_force)
                if soft_force is not None else None,
            "reciprocal_force_residual_n": force_residual,
            "endpoint_force_impulse_residual_n_s": force_impulse_residual,
            "maximum_endpoint_speed_m_s": maximum_speed,
        },
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_fluid_rope_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture",
                                "expected a fixture object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected a fixture object", actual))
        actual = {}
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    for field in ("id", "substeps", "fluid_particle_count",
                  "rope_node_count", "rope_solver_iterations",
                  "rope_self_collision", "rope_first_attached",
                  "rope_last_attached", "two_way", "oracle"):
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    actual.get(field), definition.get(field)))
    scalar_fields = (
        "timestep_seconds", "fluid_rest_density_kg_m3",
        "fluid_particle_radius_m", "fluid_support_radius_m",
        "fluid_rest_particle_volume_m3", "fluid_maximum_speed_m_s",
        "rope_node_spacing_m", "rope_radius_m", "rope_total_mass_kg",
        "rope_stretch_compliance_m_per_n", "rope_velocity_damping",
        "rope_maximum_substep_timestep_seconds", "rope_friction",
        "rope_maximum_speed_m_s", "contact_distance_m", "friction",
        "maximum_rope_acceleration_m_s2")
    for field in scalar_fields:
        value = _scalar(actual.get(field), f"$.fixture.{field}", failures)
        expected = _scalar(definition.get(field),
                           f"$.profile.fixture.{field}", failures)
        if (value is not None and expected is not None and
                abs(value - expected) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    value, expected))

    def vectors(value: Any, path: str) -> list[tuple[float, float, float]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a vector array", value))
            return []
        parsed: list[tuple[float, float, float]] = []
        for index, item in enumerate(value):
            vector = _vector(item, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    gravity = _vector(actual.get("gravity_m_s2"),
                      "$.fixture.gravity_m_s2", failures)
    expected_gravity = _vector(definition.get("gravity_m_s2"),
                               "$.profile.fixture.gravity_m_s2", failures)
    if (gravity is not None and expected_gravity is not None and
            _norm(_subtract(gravity, expected_gravity)) > state_tolerance):
        failures.append(Failure("$.fixture.gravity_m_s2",
                                "fixture setting differs from profile",
                                list(gravity), list(expected_gravity)))

    centerline = vectors(actual.get("rope_centerline_m"),
                         "$.fixture.rope_centerline_m")
    expected_centerline = vectors(definition.get("rope_centerline_m"),
                                  "$.profile.fixture.rope_centerline_m")
    if len(centerline) != len(expected_centerline):
        failures.append(Failure("$.fixture.rope_centerline_m",
                                "centerline count differs from profile",
                                len(centerline), len(expected_centerline)))
    else:
        for index, (value, expected) in enumerate(zip(
                centerline, expected_centerline)):
            if _norm(_subtract(value, expected)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.rope_centerline_m[{index}]",
                    "centerline differs from profile", list(value),
                    list(expected)))

    def particle_states(value: Any, path: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a particle array", value))
            return []
        parsed: list[dict[str, Any]] = []
        for index, particle in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(item_path,
                                        "expected a particle object", particle))
                continue
            if particle.get("stable_id") != index:
                failures.append(Failure(item_path + ".stable_id",
                                        "fluid stable ID/order changed",
                                        particle.get("stable_id"), index))
            position = _vector(particle.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(particle.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            acceleration = _vector(particle.get("acceleration_m_s2"),
                                   item_path + ".acceleration_m_s2", failures)
            if position is not None and velocity is not None and acceleration is not None:
                parsed.append({"position": position, "velocity": velocity,
                               "acceleration": acceleration})
        return parsed

    def node_states(value: Any, path: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a rope-node array", value))
            return []
        parsed: list[dict[str, Any]] = []
        for index, node in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(node, dict):
                failures.append(Failure(item_path,
                                        "expected a rope-node object", node))
                continue
            if node.get("index") != index:
                failures.append(Failure(item_path + ".index",
                                        "rope-node order/index changed",
                                        node.get("index"), index))
            position = _vector(node.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(node.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            force = _vector(node.get("liquid_contact_force_n"),
                            item_path + ".liquid_contact_force_n", failures)
            if position is not None and velocity is not None and force is not None:
                parsed.append({"position": position, "velocity": velocity,
                               "force": force})
        return parsed

    initial_fluid = particle_states(actual.get("initial_fluid_state"),
                                    "$.fixture.initial_fluid_state")
    final_fluid = particle_states(actual.get("final_fluid_state"),
                                  "$.fixture.final_fluid_state")
    initial_nodes = node_states(actual.get("initial_rope_node_state"),
                                "$.fixture.initial_rope_node_state")
    final_nodes = node_states(actual.get("final_rope_node_state"),
                              "$.fixture.final_rope_node_state")
    expected_particles = definition.get("initial_particles")
    if not isinstance(expected_particles, list):
        failures.append(Failure("$.profile.fixture.initial_particles",
                                "expected a particle array", expected_particles))
        expected_particles = []
    expected_particle_count = definition.get("fluid_particle_count")
    expected_node_count = definition.get("rope_node_count")
    for label, states in (("initial", initial_fluid), ("final", final_fluid)):
        if (isinstance(expected_particle_count, int) and
                len(states) != expected_particle_count):
            failures.append(Failure(f"$.fixture.{label}_fluid_state",
                                    "fluid particle count differs",
                                    len(states), expected_particle_count))
    for label, states in (("initial", initial_nodes), ("final", final_nodes)):
        if isinstance(expected_node_count, int) and len(states) != expected_node_count:
            failures.append(Failure(f"$.fixture.{label}_rope_node_state",
                                    "rope-node count differs",
                                    len(states), expected_node_count))

    if len(initial_fluid) == len(expected_particles):
        for index, (state, expected) in enumerate(zip(
                initial_fluid, expected_particles)):
            if not isinstance(expected, dict):
                failures.append(Failure(
                    f"$.profile.fixture.initial_particles[{index}]",
                    "expected a particle object", expected))
                continue
            wanted_position = _vector(expected.get("position_m"),
                f"$.profile.fixture.initial_particles[{index}].position_m",
                failures)
            wanted_velocity = _vector(expected.get("velocity_m_s"),
                f"$.profile.fixture.initial_particles[{index}].velocity_m_s",
                failures)
            if (wanted_position is not None and
                    _norm(_subtract(state["position"], wanted_position)) >
                    state_tolerance):
                failures.append(Failure(
                    f"$.fixture.initial_fluid_state[{index}].position_m",
                    "initial particle position differs from profile",
                    list(state["position"]), list(wanted_position)))
            if (wanted_velocity is not None and
                    _norm(_subtract(state["velocity"], wanted_velocity)) >
                    state_tolerance):
                failures.append(Failure(
                    f"$.fixture.initial_fluid_state[{index}].velocity_m_s",
                    "initial particle velocity differs from profile",
                    list(state["velocity"]), list(wanted_velocity)))
    for index, state in enumerate(initial_nodes):
        if index < len(centerline) and _norm(_subtract(
                state["position"], centerline[index])) > state_tolerance:
            failures.append(Failure(
                f"$.fixture.initial_rope_node_state[{index}].position_m",
                "initial rope-node position differs from centerline",
                list(state["position"]), list(centerline[index])))
        if _norm(state["velocity"]) > state_tolerance:
            failures.append(Failure(
                f"$.fixture.initial_rope_node_state[{index}].velocity_m_s",
                "rope fixture must start at rest", list(state["velocity"]),
                state_tolerance))
        if _norm(state["force"]) > state_tolerance:
            failures.append(Failure(
                f"$.fixture.initial_rope_node_state[{index}].liquid_contact_force_n",
                "rope fixture has contact force before stepping",
                list(state["force"]), state_tolerance))

    def scale(value: tuple[float, float, float], factor: float
              ) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)  # type: ignore[return-value]

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def segment_projection(point: tuple[float, float, float],
                           nodes: list[dict[str, Any]], path: str
                           ) -> tuple[float, tuple[float, float, float],
                                      tuple[float, float, float]] | None:
        if len(nodes) != 2:
            return None
        start, end = nodes[0]["position"], nodes[1]["position"]
        segment = _subtract(end, start)
        length_squared = dot(segment, segment)
        if length_squared <= 1.0e-12:
            failures.append(Failure(path, "rope segment is degenerate",
                                    length_squared, "> 0"))
            return None
        fraction = min(1.0, max(0.0,
            dot(_subtract(point, start), segment) / length_squared))
        closest = tuple(a + fraction * b for a, b in zip(start, segment))
        separation = _subtract(point, closest)
        normal = _unit(separation)
        if normal is None:
            failures.append(Failure(path,
                                    "particle lies on rope segment centerline"))
            return None
        return fraction, closest, normal

    final_separation = penetration = segment_fraction = None
    fluid_impulse = rope_impulse = reciprocal_residual = None
    expected_impulse_error = distribution_error = force_residual = None
    tangential_impulse = maximum_speed = None
    complete = (len(initial_fluid) == 1 and len(final_fluid) == 1 and
                len(initial_nodes) == 2 and len(final_nodes) == 2)
    if complete:
        initial_projection = segment_projection(
            initial_fluid[0]["position"], initial_nodes,
            "$.fixture.initial_rope_node_state")
        final_projection = segment_projection(
            final_fluid[0]["position"], final_nodes,
            "$.fixture.final_rope_node_state")
        if initial_projection is not None and final_projection is not None:
            segment_fraction = initial_projection[0]
            final_delta = _subtract(final_fluid[0]["position"],
                                    final_projection[1])
            final_separation = _norm(final_delta)
            contact_distance = float(definition.get("contact_distance_m", 0.0))
            penetration = max(0.0, contact_distance - final_separation)
            penetration_limit = float(tolerances.get(
                "maximum_overlap_penetration_m", 0.0))
            if penetration > penetration_limit:
                failures.append(Failure(
                    "$.fixture.final_fluid_state[0].position_m",
                    "fluid penetrates the rope contact radius",
                    penetration, penetration_limit))

            density = float(definition.get("fluid_rest_density_kg_m3", 0.0))
            volume = float(definition.get(
                "fluid_rest_particle_volume_m3", 0.0))
            fluid_mass = density * volume
            rope_mass = float(definition.get("rope_total_mass_kg", 0.0))
            node_mass = rope_mass / len(final_nodes)
            fluid_impulse = scale(_subtract(
                final_fluid[0]["velocity"], initial_fluid[0]["velocity"]),
                fluid_mass)
            node_impulses = [scale(_subtract(after["velocity"], before["velocity"]),
                                   node_mass)
                             for before, after in zip(initial_nodes, final_nodes)]
            rope_impulse = _sum(node_impulses)
            reciprocal_residual = _norm(_sum([fluid_impulse, rope_impulse]))
            reciprocal_limit = float(tolerances.get(
                "maximum_reciprocal_impulse_residual_n_s", 0.0))
            if reciprocal_residual > reciprocal_limit:
                failures.append(Failure(
                    "$.fixture.final_rope_node_state",
                    "fluid and rope contact impulses are not reciprocal",
                    reciprocal_residual, reciprocal_limit))
            activity = _norm(fluid_impulse)
            activity_limit = float(tolerances.get(
                "minimum_contact_impulse_n_s", 0.0))
            if activity < activity_limit:
                failures.append(Failure(
                    "$.fixture.final_fluid_state[0].velocity_m_s",
                    "fluid-rope contact impulse is too small",
                    activity, activity_limit))

            normal = initial_projection[2]
            rope_velocity = tuple(
                (1.0 - segment_fraction) * initial_nodes[0]["velocity"][axis] +
                segment_fraction * initial_nodes[1]["velocity"][axis]
                for axis in range(3))
            relative_velocity = _subtract(initial_fluid[0]["velocity"],
                                          rope_velocity)
            closing_speed = max(0.0, -dot(relative_velocity, normal))
            expected_fluid_impulse = scale(normal, fluid_mass * closing_speed)
            expected_impulse_error = _norm(_subtract(
                fluid_impulse, expected_fluid_impulse))
            expected_limit = float(tolerances.get(
                "maximum_expected_impulse_error_n_s", 0.0))
            if expected_impulse_error > expected_limit:
                failures.append(Failure(
                    "$.fixture.final_fluid_state[0].velocity_m_s",
                    "normal contact impulse differs from the public-state oracle",
                    expected_impulse_error, expected_limit))

            expected_weights = (1.0 - segment_fraction, segment_fraction)
            expected_rope_impulse = scale(expected_fluid_impulse, -1.0)
            distribution_errors = [_norm(_subtract(
                impulse, scale(expected_rope_impulse, weight)))
                for impulse, weight in zip(node_impulses, expected_weights)]
            distribution_error = max(distribution_errors)
            distribution_limit = float(tolerances.get(
                "maximum_node_distribution_error_n_s", 0.0))
            if distribution_error > distribution_limit:
                failures.append(Failure(
                    "$.fixture.final_rope_node_state",
                    "rope endpoint impulse distribution indicates hidden attachment or wrong segment weighting",
                    distribution_error, distribution_limit))

            timestep = float(definition.get("timestep_seconds", 0.0))
            force_errors = [_norm(_subtract(scale(node["force"], timestep),
                                                  impulse))
                            for node, impulse in zip(final_nodes, node_impulses)]
            force_residual = max(force_errors)
            force_limit = float(tolerances.get(
                "maximum_node_force_impulse_residual_n_s", 0.0))
            if force_residual > force_limit:
                failures.append(Failure(
                    "$.fixture.final_rope_node_state",
                    "rope liquid-contact force is inconsistent with impulse",
                    force_residual, force_limit))

            impulses = [fluid_impulse, *node_impulses]
            tangential_impulse = max(_norm(_subtract(
                impulse, scale(normal, dot(impulse, normal))))
                for impulse in impulses)
            tangent_limit = float(tolerances.get(
                "maximum_tangential_impulse_n_s", 0.0))
            if tangential_impulse > tangent_limit:
                failures.append(Failure(
                    "$.fixture.final_rope_node_state",
                    "frictionless contact introduced tangential impulse",
                    tangential_impulse, tangent_limit))

        speeds = [_norm(final_fluid[0]["velocity"])] + [
            _norm(node["velocity"]) for node in final_nodes]
        maximum_speed = max(speeds)
        speed_limit = max(
            float(definition.get("fluid_maximum_speed_m_s", 0.0)),
            float(definition.get("rope_maximum_speed_m_s", 0.0))) + float(
                tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure("$.fixture",
                                    "contact speed exceeds authored maximum",
                                    maximum_speed, speed_limit))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": 1 if complete else 0,
        "failure_count": len(failures),
        "observations": {
            "initial_segment_fraction": segment_fraction,
            "final_point_segment_separation_m": final_separation,
            "maximum_overlap_penetration_m": penetration,
            "fluid_contact_impulse_n_s": list(fluid_impulse)
                if fluid_impulse is not None else None,
            "rope_contact_impulse_n_s": list(rope_impulse)
                if rope_impulse is not None else None,
            "reciprocal_impulse_residual_n_s": reciprocal_residual,
            "expected_impulse_error_n_s": expected_impulse_error,
            "maximum_node_distribution_error_n_s": distribution_error,
            "maximum_node_force_impulse_residual_n_s": force_residual,
            "maximum_tangential_impulse_n_s": tangential_impulse,
            "maximum_endpoint_speed_m_s": maximum_speed,
        },
        "failures": [asdict(failure) for failure in failures],
    }


def _smoke_particle_states(
        value: Any, path: str, failures: list[Failure]
        ) -> dict[int, tuple[tuple[float, float, float],
                             tuple[float, float, float]]]:
    if not isinstance(value, list):
        failures.append(Failure(path, "expected a smoke particle array", value))
        return {}
    states: dict[int, tuple[tuple[float, float, float],
                            tuple[float, float, float]]] = {}
    for index, particle in enumerate(value):
        particle_path = f"{path}[{index}]"
        if not isinstance(particle, dict):
            failures.append(Failure(
                particle_path, "expected a smoke particle object", particle))
            continue
        slot = particle.get("slot")
        if (not isinstance(slot, int) or isinstance(slot, bool) or slot < 0):
            failures.append(Failure(
                particle_path + ".slot", "expected a nonnegative slot", slot))
            continue
        if slot != index:
            failures.append(Failure(
                particle_path + ".slot", "public smoke slot order changed",
                slot, index))
        if slot in states:
            failures.append(Failure(
                particle_path + ".slot", "duplicate smoke slot", slot))
            continue
        position = _vector(particle.get("position_m"),
                           particle_path + ".position_m", failures)
        velocity = _vector(particle.get("velocity_m_s"),
                           particle_path + ".velocity_m_s", failures)
        _scalar(particle.get("age_seconds"),
                particle_path + ".age_seconds", failures)
        _scalar(particle.get("number_density"),
                particle_path + ".number_density", failures)
        _scalar(particle.get("pressure_m_s2"),
                particle_path + ".pressure_m_s2", failures)
        _vector(particle.get("vorticity_per_s"),
                particle_path + ".vorticity_per_s", failures)
        if position is not None and velocity is not None:
            states[slot] = position, velocity
    return states


def _smoke_velocity_statistics(
        states: dict[int, tuple[tuple[float, float, float],
                                tuple[float, float, float]]]
        ) -> dict[str, Any] | None:
    if not states:
        return None
    velocities = [state[1] for _, state in sorted(states.items())]
    count = len(velocities)
    mean = tuple(sum(value[axis] for value in velocities) / count
                 for axis in range(3))
    relative_square = sum(
        sum((value[axis] - mean[axis]) ** 2 for axis in range(3))
        for value in velocities) / count
    total_square = sum(
        sum(component * component for component in value)
        for value in velocities) / count
    return {
        "mean_velocity_m_s": mean,
        "relative_rms_speed_m_s": math.sqrt(relative_square),
        "relative_specific_kinetic_energy_m2_s2": 0.5 * relative_square,
        "total_specific_kinetic_energy_m2_s2": 0.5 * total_square,
        "maximum_speed_m_s": max(_norm(value) for value in velocities),
    }


def _evaluate_smoke_self_viscosity(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    def index_by_id(value: list[Any], path: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                failures.append(Failure(
                    item_path, "item needs a string id", item))
                continue
            identity = item["id"]
            if identity in indexed:
                failures.append(Failure(
                    item_path + ".id", "duplicate id", identity))
                continue
            indexed[identity] = item
        return indexed

    definitions_value = profile.get("fixtures")
    actual_value = result.get("fixtures")
    if not isinstance(definitions_value, list):
        failures.append(Failure(
            "$.profile.fixtures", "expected a fixture array", definitions_value))
        definitions_value = []
    if not isinstance(actual_value, list):
        failures.append(Failure(
            "$.fixtures", "expected a fixture array", actual_value))
        actual_value = []
    definitions = index_by_id(definitions_value, "$.profile.fixtures")
    actuals = index_by_id(actual_value, "$.fixtures")
    if set(actuals) != set(definitions):
        failures.append(Failure(
            "$.fixtures", "smoke viscosity fixture identities differ",
            sorted(actuals), sorted(definitions)))
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure(
            "$.profile.tolerances", "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    exact_fields = (
        "id", "representation", "substeps", "initial_capture_step",
        "final_capture_step", "particle_capacity", "expected_particle_count",
        "grid_resolution", "grid_vertical_resolution",
        "grid_pressure_iterations", "oracle",
    )
    scalar_fields = (
        "timestep_seconds", "particles_per_second", "lifetime_seconds",
        "particle_radius_m", "buoyancy_m_s2", "wind_response_per_s",
        "rest_number_density", "pressure_stiffness_m_s2",
        "control_viscosity_per_s", "test_viscosity_per_s",
        "vorticity_confinement_m_s2", "maximum_speed_m_s",
        "grid_control_kinematic_viscosity_m2_s",
        "grid_test_kinematic_viscosity_m2_s", "grid_les_coefficient",
        "grid_pressure_tolerance_per_s", "grid_edge_length_m",
    )
    vector_fields = (
        "gravity_m_s2", "emitter_center_m", "initial_velocity_m_s",
        "wind_velocity_m_s", "grid_minimum_m",
    )
    observations: dict[str, Any] = {}
    sample_count = 0

    for identity, definition in definitions.items():
        actual = actuals.get(identity)
        if actual is None:
            continue
        fixture_path = f"$.fixtures[{identity}]"
        for field in exact_fields:
            if actual.get(field) != definition.get(field):
                failures.append(Failure(
                    f"{fixture_path}.{field}",
                    "fixture setting does not match profile",
                    actual.get(field), definition.get(field)))
        for field in scalar_fields:
            value = _scalar(actual.get(field),
                            f"{fixture_path}.{field}", failures)
            expected = _scalar(definition.get(field),
                               f"$.profile.fixtures[{identity}].{field}",
                               failures)
            if (value is not None and expected is not None and
                    abs(value - expected) > state_tolerance):
                failures.append(Failure(
                    f"{fixture_path}.{field}",
                    "fixture setting differs from profile", value, expected))
        for field in vector_fields:
            value = _vector(actual.get(field), f"{fixture_path}.{field}",
                            failures)
            expected = _vector(
                definition.get(field),
                f"$.profile.fixtures[{identity}].{field}", failures)
            if (value is not None and expected is not None and
                    max(abs(a - b) for a, b in zip(value, expected)) >
                    state_tolerance):
                failures.append(Failure(
                    f"{fixture_path}.{field}",
                    "fixture setting differs from profile",
                    list(value), list(expected)))
        extents = actual.get("emitter_half_extents_m")
        expected_extents = definition.get("emitter_half_extents_m")
        if (not isinstance(extents, list) or len(extents) != 2 or
                not all(_number(item) for item in extents)):
            failures.append(Failure(
                fixture_path + ".emitter_half_extents_m",
                "expected two finite emitter extents", extents))
        elif (not isinstance(expected_extents, list) or
              len(expected_extents) != 2 or
              not all(_number(item) for item in expected_extents)):
            failures.append(Failure(
                f"$.profile.fixtures[{identity}].emitter_half_extents_m",
                "expected two finite emitter extents", expected_extents))
        elif max(abs(float(a) - float(b))
                 for a, b in zip(extents, expected_extents)) > state_tolerance:
            failures.append(Failure(
                fixture_path + ".emitter_half_extents_m",
                "fixture setting differs from profile", extents,
                expected_extents))

        runs_value = actual.get("runs")
        if not isinstance(runs_value, list):
            failures.append(Failure(
                fixture_path + ".runs", "expected a run array", runs_value))
            runs_value = []
        runs = index_by_id(runs_value, fixture_path + ".runs")
        if set(runs) != {"control", "viscous"}:
            failures.append(Failure(
                fixture_path + ".runs",
                "expected exactly control and viscous runs",
                sorted(runs), ["control", "viscous"]))
        states: dict[str, dict[str, dict[int, tuple[
            tuple[float, float, float], tuple[float, float, float]]]]] = {}
        expected_count = definition.get("expected_particle_count")
        for run_id in ("control", "viscous"):
            run = runs.get(run_id)
            if run is None:
                continue
            run_path = f"{fixture_path}.runs[{run_id}]"
            initial = _smoke_particle_states(
                run.get("initial_particles"), run_path + ".initial_particles",
                failures)
            final = _smoke_particle_states(
                run.get("final_particles"), run_path + ".final_particles",
                failures)
            states[run_id] = {"initial": initial, "final": final}
            if set(initial) != set(final):
                failures.append(Failure(
                    run_path, "sampled smoke slots changed during measurement",
                    sorted(final), sorted(initial)))
            if (isinstance(expected_count, int) and
                    not isinstance(expected_count, bool) and
                    len(initial) != expected_count):
                failures.append(Failure(
                    run_path + ".initial_particles",
                    "smoke particle count differs from profile",
                    len(initial), expected_count))
            capacity = definition.get("particle_capacity")
            if (isinstance(capacity, int) and not isinstance(capacity, bool) and
                    len(final) > capacity):
                failures.append(Failure(
                    run_path + ".final_particles",
                    "smoke output exceeds declared capacity",
                    len(final), capacity))
        if set(states) != {"control", "viscous"}:
            continue
        if set(states["control"]["initial"]) != set(
                states["viscous"]["initial"]):
            failures.append(Failure(
                fixture_path + ".runs",
                "paired smoke runs sampled different slots"))

        statistics: dict[str, dict[str, dict[str, Any]]] = {}
        for run_id in ("control", "viscous"):
            statistics[run_id] = {}
            for phase in ("initial", "final"):
                measured = _smoke_velocity_statistics(states[run_id][phase])
                if measured is not None:
                    statistics[run_id][phase] = measured
        if any(set(statistics[run_id]) != {"initial", "final"}
               for run_id in statistics):
            continue

        initial_activity = statistics["viscous"]["initial"][
            "relative_rms_speed_m_s"]
        activity_minimum = float(tolerances.get(
            "initial_relative_rms_speed_minimum_m_s", 0.0))
        if initial_activity < activity_minimum:
            failures.append(Failure(
                fixture_path + ".runs[viscous].initial_particles",
                "initial smoke relative motion is too small for viscosity",
                initial_activity, activity_minimum))

        control_energy = statistics["control"]["final"][
            "relative_specific_kinetic_energy_m2_s2"]
        viscous_energy = statistics["viscous"]["final"][
            "relative_specific_kinetic_energy_m2_s2"]
        reduction = (1.0 - viscous_energy / control_energy
                     if control_energy > 0.0 else 0.0)
        minimum_reduction = float(tolerances.get(
            "relative_specific_energy_reduction_fraction_minimum", 0.0))
        if reduction < minimum_reduction:
            failures.append(Failure(
                fixture_path + ".runs[viscous].final_particles",
                "smoke viscosity did not dissipate relative motion",
                reduction, minimum_reduction))
        energy_slack_key = (
            "particle_specific_energy_relative_increase_maximum"
            if definition.get("representation") == "SMOKE_PARTICLE"
            else "driven_grid_specific_energy_relative_increase_maximum")
        energy_slack = float(tolerances.get(energy_slack_key, 0.0))
        control_total = statistics["control"]["final"][
            "total_specific_kinetic_energy_m2_s2"]
        viscous_total = statistics["viscous"]["final"][
            "total_specific_kinetic_energy_m2_s2"]
        total_limit = control_total * (1.0 + energy_slack)
        if viscous_total > total_limit:
            failures.append(Failure(
                fixture_path + ".runs[viscous].final_particles",
                "smoke viscosity increased specific kinetic energy",
                viscous_total, total_limit))

        maximum_speed = max(
            statistics[run_id][phase]["maximum_speed_m_s"]
            for run_id in ("control", "viscous")
            for phase in ("initial", "final"))
        maximum_authored = float(definition.get("maximum_speed_m_s", 0.0))
        speed_limit = maximum_authored * (1.0 + float(tolerances.get(
            "maximum_speed_relative_slack", 0.0)))
        if maximum_speed > speed_limit:
            failures.append(Failure(
                fixture_path + ".runs", "smoke speed exceeds authored maximum",
                maximum_speed, speed_limit))

        def mean_change(run_id: str) -> tuple[float, float, float]:
            before = statistics[run_id]["initial"]["mean_velocity_m_s"]
            after = statistics[run_id]["final"]["mean_velocity_m_s"]
            return _subtract(after, before)

        control_change = mean_change("control")
        viscous_change = mean_change("viscous")
        representation = definition.get("representation")
        if representation == "SMOKE_PARTICLE":
            impulse_residual = _norm(viscous_change)
            residual_limit = float(tolerances.get(
                "particle_mean_velocity_change_absolute_m_s", 0.0))
        else:
            impulse_residual = _norm(_subtract(
                viscous_change, control_change))
            residual_limit = float(tolerances.get(
                "grid_control_delta_absolute_m_s", 0.0))
        if impulse_residual > residual_limit:
            failures.append(Failure(
                fixture_path + ".runs[viscous]",
                "viscosity introduced an unexplained mean-velocity change",
                impulse_residual, residual_limit))

        observations[identity] = {
            "initial_relative_rms_speed_m_s": initial_activity,
            "control_final_relative_specific_kinetic_energy_m2_s2":
                control_energy,
            "viscous_final_relative_specific_kinetic_energy_m2_s2":
                viscous_energy,
            "relative_specific_energy_reduction_fraction": reduction,
            "control_final_total_specific_kinetic_energy_m2_s2": control_total,
            "viscous_final_total_specific_kinetic_energy_m2_s2": viscous_total,
            "mean_velocity_residual_m_s": impulse_residual,
            "maximum_speed_m_s": maximum_speed,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actuals),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }



def _evaluate_smoke_self_pressure(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    age_tolerance = float(tolerances.get(
        "aged_particle_age_absolute_seconds", state_tolerance))
    expected_per_step = tolerances.get("expected_particles_per_step")
    if not isinstance(expected_per_step, int) or isinstance(
            expected_per_step, bool) or expected_per_step <= 0:
        failures.append(Failure(
            "$.profile.tolerances.expected_particles_per_step",
            "expected a positive integer", expected_per_step))
        expected_per_step = 0

    integer_fields = ("step_count", "substeps", "capacity",
                      "grid_resolution", "grid_vertical_resolution",
                      "grid_pressure_iterations")
    scalar_fields = ("timestep_seconds", "particles_per_second",
                     "lifetime_seconds", "particle_radius_m",
                     "rest_number_density", "pressure_stiffness",
                     "maximum_speed_m_s", "grid_pressure_tolerance",
                     "grid_edge_length_m")
    vector_fields = ("gravity_m_s2", "emitter_center_m",
                     "initial_velocity_m_s", "wind_m_s",
                     "grid_minimum_m")

    def close_scalar(value: Any, expected: Any, path: str) -> None:
        actual_number = _scalar(value, path, failures)
        expected_number = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (actual_number is not None and expected_number is not None and
                abs(actual_number - expected_number) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual_number, expected_number))

    def close_vector(value: Any, expected: Any, path: str) -> None:
        actual_vector = _vector(value, path, failures)
        expected_vector = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (actual_vector is not None and expected_vector is not None and
                max(abs(a - b) for a, b in zip(
                    actual_vector, expected_vector)) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(actual_vector), list(expected_vector)))

    def close_extents(value: Any, expected: Any, path: str) -> None:
        if (not isinstance(value, list) or len(value) != 2 or
                not all(_number(item) for item in value)):
            failures.append(Failure(path, "expected two finite components", value))
            return
        if (not isinstance(expected, list) or len(expected) != 2 or
                not all(_number(item) for item in expected)):
            failures.append(Failure(path.replace(
                "$.fixtures", "$.profile.fixtures"),
                "expected two finite components", expected))
            return
        if max(abs(float(a) - float(b)) for a, b in zip(
                value, expected)) > state_tolerance:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, expected))

    def particles(value: Any, path: str, expected_count: int) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a particle array", value))
            return []
        if len(value) != expected_count:
            failures.append(Failure(path, "particle count differs from profile",
                                    len(value), expected_count))
        parsed: list[dict[str, Any]] = []
        for index, particle in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(item_path,
                                        "expected a particle object", particle))
                continue
            if particle.get("index") != index:
                failures.append(Failure(item_path + ".index",
                                        "particle order/index changed",
                                        particle.get("index"), index))
            valid = True
            for name in ("position_m", "velocity_m_s", "vorticity_per_s"):
                valid = _vector(particle.get(name), item_path + "." + name,
                                failures) is not None and valid
            for name in ("age_seconds", "number_density", "pressure"):
                valid = _scalar(particle.get(name), item_path + "." + name,
                                failures) is not None and valid
            if valid:
                parsed.append(particle)
        return parsed

    sample_count = 0
    observations: dict[str, Any] = {}
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("representation", "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in integer_fields:
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in scalar_fields:
            close_scalar(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in vector_fields:
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        close_extents(actual.get("emitter_half_extents_m"),
                      definition.get("emitter_half_extents_m"),
                      base + ".emitter_half_extents_m")

        first = particles(actual.get("first_particles"),
                          base + ".first_particles", expected_per_step)
        second = particles(actual.get("second_particles"),
                           base + ".second_particles", 2 * expected_per_step)
        final = particles(actual.get("final_particles"),
                          base + ".final_particles", 3 * expected_per_step)
        maximum_speed = float(definition.get("maximum_speed_m_s", 0.0)) + float(
            tolerances.get("maximum_speed_slack_m_s", 0.0))
        timestep = float(definition.get("timestep_seconds", 0.0))
        maximum_pressure = 0.0
        maximum_equation_error = 0.0
        for index, particle in enumerate(first):
            age = float(particle["age_seconds"])
            pressure = float(particle["pressure"])
            density = float(particle["number_density"])
            if abs(age) > age_tolerance:
                failures.append(Failure(
                    f"{base}.first_particles[{index}].age_seconds",
                    "new particle age must be zero", age, 0.0))
            if abs(pressure) > state_tolerance or abs(density) > state_tolerance:
                failures.append(Failure(
                    f"{base}.first_particles[{index}]",
                    "new particle pressure state must be zero",
                    {"pressure": pressure, "number_density": density}, 0.0))
        for index, particle in enumerate(second):
            expected_age = timestep if index < expected_per_step else 0.0
            age = float(particle["age_seconds"])
            if abs(age - expected_age) > age_tolerance:
                failures.append(Failure(
                    f"{base}.second_particles[{index}].age_seconds",
                    "second checkpoint particle age is incorrect",
                    age, expected_age))
        for index, particle in enumerate(final):
            age = float(particle["age_seconds"])
            pressure = float(particle["pressure"])
            density = float(particle["number_density"])
            velocity = tuple(float(component)
                             for component in particle["velocity_m_s"])
            if density < 0.0:
                failures.append(Failure(
                    f"{base}.final_particles[{index}].number_density",
                    "number density must be nonnegative", density, 0.0))
            if _norm(velocity) > maximum_speed:
                failures.append(Failure(
                    f"{base}.final_particles[{index}].velocity_m_s",
                    "particle speed exceeds authored maximum",
                    _norm(velocity), maximum_speed))
            if index < 2 * expected_per_step:
                expected_age = (2 * timestep if index < expected_per_step
                                else timestep)
                if abs(age - expected_age) > age_tolerance:
                    failures.append(Failure(
                        f"{base}.final_particles[{index}].age_seconds",
                        "aged particle did not advance to its checkpoint age",
                        age, expected_age))
                if (definition.get("representation") == "particle" or
                        index < expected_per_step):
                    maximum_pressure = max(maximum_pressure, abs(pressure))
                    sample_count += 1
            elif abs(age) > age_tolerance:
                failures.append(Failure(
                    f"{base}.final_particles[{index}].age_seconds",
                    "second emission must remain new", age, 0.0))

        mean_velocity_drift = 0.0
        if (definition.get("representation") == "particle" and final):
            initial_velocity = tuple(float(component) for component in
                                     definition["initial_velocity_m_s"])
            mean_velocity = tuple(
                sum(float(item["velocity_m_s"][component]) for item in final) /
                len(final) for component in range(3))
            mean_velocity_drift = _norm(tuple(
                actual - expected for actual, expected in
                zip(mean_velocity, initial_velocity)))
            drift_tolerance = float(tolerances.get(
                "particle_mean_velocity_drift_absolute_m_s", 0.0))
            if mean_velocity_drift > drift_tolerance:
                failures.append(Failure(
                    base + ".final_particles",
                    "closed particle pressure introduced net velocity drift",
                    mean_velocity_drift, drift_tolerance))

        if (definition.get("representation") == "particle" and second and
                final):
            rest = float(definition.get("rest_number_density", 0.0))
            stiffness = float(definition.get("pressure_stiffness", 0.0))
            equation_tolerance = float(tolerances.get(
                "particle_pressure_equation_absolute", 0.0))
            for index in range(min(2 * expected_per_step, len(second),
                                   len(final))):
                density = float(final[index]["number_density"])
                previous = float(second[index]["pressure"])
                expected = min(100.0, stiffness * max(0.0, density / rest - 1.0) +
                               previous * math.exp(-timestep / 0.05))
                actual_pressure = float(final[index]["pressure"])
                error = abs(actual_pressure - expected)
                maximum_equation_error = max(maximum_equation_error, error)
                if error > equation_tolerance:
                    failures.append(Failure(
                        f"{base}.final_particles[{index}].pressure",
                        "particle pressure differs from public density oracle",
                        actual_pressure, expected))
            minimum = float(tolerances.get(
                "particle_pressure_activity_minimum", 0.0))
        else:
            minimum = float(tolerances.get(
                "grid_pressure_activity_minimum", 0.0))
        if maximum_pressure < minimum:
            failures.append(Failure(base + ".final_particles",
                                    "pressure response is too small",
                                    maximum_pressure, minimum))
        observations[identity] = {
            "maximum_absolute_pressure": maximum_pressure,
            "maximum_particle_equation_error": maximum_equation_error,
            "particle_mean_velocity_drift_m_s": mean_velocity_drift,
        }

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_fluid_smoke_phase(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    control_tolerance = float(tolerances.get("control_state_absolute", 0.0))
    velocity_tolerance = float(tolerances.get("velocity_match_m_s", 0.0))
    momentum_tolerance = float(
        tolerances.get("momentum_residual_kg_m_s", 0.0))
    energy_tolerance = float(
        tolerances.get("kinetic_energy_residual_j", 0.0))
    age_tolerance = float(tolerances.get("smoke_age_seconds", 0.0))

    def sub(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(first, second))

    def sum_vectors(values: list[tuple[float, float, float]]) -> \
            tuple[float, float, float]:
        return tuple(sum(value[axis] for value in values) for axis in range(3))

    def close_number(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        reference = _scalar(expected, path + ".profile", failures)
        if (value is not None and reference is not None and
                abs(value - reference) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(actual, path, failures)
        reference = _vector(expected, path + ".profile", failures)
        if (value is not None and reference is not None and
                _norm(sub(value, reference)) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def close_vector_sequence(actual: Any, expected: Any, path: str) -> None:
        if (not isinstance(actual, list) or not isinstance(expected, list) or
                len(actual) != len(expected)):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))
            return
        for index, (value, reference) in enumerate(zip(actual, expected)):
            close_vector(value, reference, f"{path}[{index}]")

    def snapshot(value: Any, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        fluid = value.get("fluid")
        smoke = value.get("smoke_particles")
        if not isinstance(fluid, list):
            failures.append(Failure(path + ".fluid",
                                    "expected a fluid array", fluid))
            return None
        if not isinstance(smoke, list):
            failures.append(Failure(path + ".smoke_particles",
                                    "expected a smoke array", smoke))
            return None
        normalized_fluid: list[dict[str, Any]] = []
        for index, particle in enumerate(fluid):
            particle_path = f"{path}.fluid[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(particle_path, "expected an object",
                                        particle))
                continue
            position = _vector(particle.get("position_m"),
                               particle_path + ".position_m", failures)
            velocity = _vector(particle.get("velocity_m_s"),
                               particle_path + ".velocity_m_s", failures)
            temperature = _scalar(particle.get("temperature_celsius"),
                                  particle_path + ".temperature_celsius",
                                  failures)
            stable_id = particle.get("stable_id")
            if (not isinstance(stable_id, int) or isinstance(stable_id, bool) or
                    stable_id < 0):
                failures.append(Failure(particle_path + ".stable_id",
                                        "expected a non-negative integer",
                                        stable_id))
                stable_id = None
            if (position is not None and velocity is not None and
                    temperature is not None and stable_id is not None):
                normalized_fluid.append({
                    "position": position, "velocity": velocity,
                    "temperature": temperature, "stable_id": stable_id,
                })
        normalized_smoke: list[dict[str, Any]] = []
        for index, particle in enumerate(smoke):
            particle_path = f"{path}.smoke_particles[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(particle_path, "expected an object",
                                        particle))
                continue
            position = _vector(particle.get("position_m"),
                               particle_path + ".position_m", failures)
            velocity = _vector(particle.get("velocity_m_s"),
                               particle_path + ".velocity_m_s", failures)
            age = _scalar(particle.get("age_seconds"),
                          particle_path + ".age_seconds", failures)
            if position is not None and velocity is not None and age is not None:
                normalized_smoke.append({
                    "position": position, "velocity": velocity, "age": age,
                })
        if (len(normalized_fluid) != len(fluid) or
                len(normalized_smoke) != len(smoke)):
            return None
        return {"fluid": normalized_fluid, "smoke": normalized_smoke}

    def count_accounting(initial: dict[str, Any], final: dict[str, Any],
                         particle_mass: float, path: str) -> float:
        consumed = len(initial["fluid"]) - len(final["fluid"])
        emitted = len(final["smoke"]) - len(initial["smoke"])
        residual = abs(float(consumed - emitted) * particle_mass)
        if consumed < 0 or emitted < 0 or residual > state_tolerance:
            failures.append(Failure(path,
                                    "phase mass accounting is not balanced",
                                    {"consumed": consumed, "emitted": emitted,
                                     "residual_kg": residual}, 0.0))
        return residual

    def transfer_accounting(
            sources: list[dict[str, Any]], smoke: list[dict[str, Any]],
            particle_mass: float, path: str) -> tuple[float, float]:
        unmatched = list(range(len(smoke)))
        for source in sources:
            if not unmatched:
                break
            best = min(unmatched, key=lambda index: _norm(sub(
                smoke[index]["velocity"], source["velocity"])))
            difference = _norm(sub(smoke[best]["velocity"],
                                   source["velocity"]))
            if difference > velocity_tolerance:
                failures.append(Failure(path,
                                        "emitted smoke lost source identity",
                                        difference, velocity_tolerance))
            unmatched.remove(best)
        if len(sources) != len(smoke):
            failures.append(Failure(path,
                                    "source and emitted identity counts differ",
                                    len(smoke), len(sources)))
        source_momentum = sum_vectors([source["velocity"] for source in sources])
        smoke_momentum = sum_vectors([particle["velocity"] for particle in smoke])
        momentum_residual = particle_mass * _norm(sub(
            smoke_momentum, source_momentum))
        if momentum_residual > momentum_tolerance:
            failures.append(Failure(path,
                                    "phase momentum accounting is outside bounds",
                                    momentum_residual, momentum_tolerance))
        source_energy = 0.5 * particle_mass * sum(
            _norm(source["velocity"]) ** 2 for source in sources)
        smoke_energy = 0.5 * particle_mass * sum(
            _norm(particle["velocity"]) ** 2 for particle in smoke)
        energy_residual = abs(smoke_energy - source_energy)
        if energy_residual > energy_tolerance:
            failures.append(Failure(path,
                                    "phase kinetic-energy accounting is outside bounds",
                                    energy_residual, energy_tolerance))
        return momentum_residual, energy_residual

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "representation", "substeps",
                      "single_smoke_particle_capacity",
                      "bounded_fluid_particle_count",
                      "bounded_smoke_particle_capacity",
                      "smoke_grid_resolution",
                      "smoke_grid_vertical_resolution", "declared_mode_flags",
                      "exercised_mode", "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in ("timestep_seconds", "fluid_initial_temperature_celsius",
                      "fluid_particle_radius_m", "fluid_support_radius_m",
                      "fluid_rest_density_kg_m3",
                      "fluid_rest_particle_volume_m3", "fluid_particle_mass_kg",
                      "smoke_emission_rate_per_s", "smoke_lifetime_seconds",
                      "smoke_particle_radius_m", "smoke_rest_number_density",
                      "smoke_maximum_speed_m_s", "smoke_grid_edge_length_m",
                      "thermal_transfer_rate_per_s",
                      "control_boiling_temperature_celsius",
                      "transfer_boiling_temperature_celsius",
                      "wind_drag_per_s", "steam_rise_speed_m_s"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("gravity_m_s2", "single_source_position_m",
                      "single_source_velocity_m_s", "smoke_emitter_center_m",
                      "smoke_grid_minimum_m"):
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        close_vector_sequence(actual.get("bounded_source_positions_m"),
                              definition.get("bounded_source_positions_m"),
                              base + ".bounded_source_positions_m")
        close_vector_sequence(actual.get("bounded_source_velocities_m_s"),
                              definition.get("bounded_source_velocities_m_s"),
                              base + ".bounded_source_velocities_m_s")
        if actual.get("smoke_emitter_half_extents_m") != \
                definition.get("smoke_emitter_half_extents_m"):
            failures.append(Failure(base + ".smoke_emitter_half_extents_m",
                                    "fixture setting differs from profile",
                                    actual.get("smoke_emitter_half_extents_m"),
                                    definition.get("smoke_emitter_half_extents_m")))

        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array", runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        expected_runs = {"below-threshold-control", "single-transfer",
                         "capacity-bound"}
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(base + ".runs",
                                    "expected exactly the three phase runs",
                                    sorted(indexed), sorted(expected_runs)))
            continue
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial")
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final")
            if initial is not None and final is not None:
                states[run_id] = (initial, final)
        if set(states) != expected_runs:
            continue

        control_initial, control_final = states["below-threshold-control"]
        if (len(control_initial["fluid"]) != 1 or control_initial["smoke"] or
                len(control_final["fluid"]) != 1 or control_final["smoke"]):
            failures.append(Failure(base + ".runs[below-threshold-control]",
                                    "below-threshold control changed domain counts",
                                    [len(control_final["fluid"]),
                                     len(control_final["smoke"])], [1, 0]))
        elif (control_initial["fluid"][0]["stable_id"] !=
              control_final["fluid"][0]["stable_id"]):
            failures.append(Failure(base + ".runs[below-threshold-control]",
                                    "below-threshold control changed stable identity",
                                    control_final["fluid"][0]["stable_id"],
                                    control_initial["fluid"][0]["stable_id"]))
        elif (_norm(sub(control_initial["fluid"][0]["velocity"],
                        control_final["fluid"][0]["velocity"])) >
              control_tolerance):
            failures.append(Failure(base + ".runs[below-threshold-control]",
                                    "below-threshold control velocity drifted",
                                    control_final["fluid"][0]["velocity"],
                                    control_initial["fluid"][0]["velocity"]))

        particle_mass = float(definition["fluid_particle_mass_kg"])
        single_initial, single_final = states["single-transfer"]
        single_mass = count_accounting(
            single_initial, single_final, particle_mass,
            base + ".runs[single-transfer]")
        if (len(single_initial["fluid"]) != 1 or single_initial["smoke"] or
                single_final["fluid"] or len(single_final["smoke"]) != 1):
            failures.append(Failure(base + ".runs[single-transfer]",
                                    "single phase transfer must consume one fluid and emit one smoke particle",
                                    [len(single_final["fluid"]),
                                     len(single_final["smoke"])], [0, 1]))
            single_momentum = float("inf")
            single_energy = float("inf")
        else:
            single_momentum, single_energy = transfer_accounting(
                single_initial["fluid"], single_final["smoke"], particle_mass,
                base + ".runs[single-transfer]")

        bounded_initial, bounded_final = states["capacity-bound"]
        bounded_mass = count_accounting(
            bounded_initial, bounded_final, particle_mass,
            base + ".runs[capacity-bound]")
        initial_ids = {particle["stable_id"] for particle in bounded_initial["fluid"]}
        final_ids = {particle["stable_id"] for particle in bounded_final["fluid"]}
        if (len(initial_ids) != len(bounded_initial["fluid"]) or
                len(final_ids) != len(bounded_final["fluid"]) or
                not final_ids.issubset(initial_ids)):
            failures.append(Failure(base + ".runs[capacity-bound]",
                                    "surviving fluid stable IDs are not a unique source subset",
                                    sorted(final_ids), sorted(initial_ids)))
        expected_capacity = int(definition["bounded_smoke_particle_capacity"])
        if (len(bounded_initial["fluid"]) !=
                int(definition["bounded_fluid_particle_count"]) or
                bounded_initial["smoke"] or
                len(bounded_final["smoke"]) != expected_capacity or
                len(bounded_final["fluid"]) !=
                len(bounded_initial["fluid"]) - expected_capacity):
            failures.append(Failure(base + ".runs[capacity-bound]",
                                    "phase transfer did not stop at smoke capacity",
                                    [len(bounded_final["fluid"]),
                                     len(bounded_final["smoke"])], [1, 2]))
        removed = [particle for particle in bounded_initial["fluid"]
                   if particle["stable_id"] not in final_ids]
        bounded_momentum, bounded_energy = transfer_accounting(
            removed, bounded_final["smoke"], particle_mass,
            base + ".runs[capacity-bound]")
        for run_id in ("single-transfer", "capacity-bound"):
            final = states[run_id][1]
            for index, particle in enumerate(final["smoke"]):
                if abs(particle["age"] - float(definition["timestep_seconds"])) > \
                        age_tolerance:
                    failures.append(Failure(
                        base + f".runs[{run_id}].final.smoke_particles[{index}].age_seconds",
                        "emitted smoke age is outside the one-step boundary",
                        particle["age"], definition["timestep_seconds"]))
        observations[identity] = {
            "single_mass_residual_kg": single_mass,
            "single_momentum_residual_kg_m_s": single_momentum,
            "single_kinetic_energy_residual_j": single_energy,
            "bounded_mass_residual_kg": bounded_mass,
            "bounded_momentum_residual_kg_m_s": bounded_momentum,
            "bounded_kinetic_energy_residual_j": bounded_energy,
            "bounded_converted_stable_ids": sorted(initial_ids - final_ids),
            "bounded_surviving_stable_ids": sorted(final_ids),
        }
        sample_count += 2

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_smoke_fluid_drag(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    smoke_tolerance = float(tolerances.get("smoke_state_absolute", 0.0))

    def sub(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(first, second))

    def close_number(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        reference = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                abs(value - reference) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(list(actual) if isinstance(actual, tuple) else actual,
                        path, failures)
        reference = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                _norm(sub(value, reference)) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def close_sequence(actual: Any, expected: Any, path: str) -> None:
        if (not isinstance(actual, list) or not isinstance(expected, list) or
                len(actual) != len(expected)):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))
            return
        values = [_scalar(item, f"{path}[{index}]", failures)
                  for index, item in enumerate(actual)]
        references = [_scalar(item, f"{path}.profile[{index}]", failures)
                      for index, item in enumerate(expected)]
        if (all(item is not None for item in values + references) and
                any(abs(value - reference) > state_tolerance
                    for value, reference in zip(values, references))):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def snapshot(value: Any, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        fluid = value.get("fluid")
        particles = value.get("smoke_particles")
        if not isinstance(fluid, dict):
            failures.append(Failure(path + ".fluid",
                                    "expected a fluid object", fluid))
            return None
        if not isinstance(particles, list) or not particles:
            failures.append(Failure(path + ".smoke_particles",
                                    "expected emitted smoke particles", particles))
            return None
        position = _vector(fluid.get("position_m"), path +
                           ".fluid.position_m", failures)
        velocity = _vector(fluid.get("velocity_m_s"), path +
                           ".fluid.velocity_m_s", failures)
        if position is None or velocity is None:
            return None
        normalized_particles: list[dict[str, Any]] = []
        for index, particle in enumerate(particles):
            particle_path = f"{path}.smoke_particles[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(particle_path, "expected an object",
                                        particle))
                continue
            vectors = []
            for field in ("position_m", "velocity_m_s", "vorticity_per_s"):
                vector = _vector(particle.get(field), particle_path + "." +
                                 field, failures)
                vectors.append(vector)
            scalars = []
            for field in ("age_seconds", "number_density", "pressure_m_s2"):
                scalar = _scalar(particle.get(field), particle_path + "." +
                                 field, failures)
                scalars.append(scalar)
            if any(item is None for item in vectors + scalars):
                continue
            normalized_particles.append({
                "vectors": [item for item in vectors if item is not None],
                "scalars": [item for item in scalars if item is not None],
            })
        if len(normalized_particles) != len(particles):
            return None
        return {"position": position, "velocity": velocity,
                "smoke": normalized_particles}

    def compare_smoke(first: dict[str, Any], second: dict[str, Any],
                      path: str) -> float:
        a = first["smoke"]
        b = second["smoke"]
        if len(a) != len(b):
            failures.append(Failure(path, "drag changed smoke particle count",
                                    len(b), len(a)))
            return float("inf")
        maximum = 0.0
        for left, right in zip(a, b):
            for left_vector, right_vector in zip(
                    left["vectors"], right["vectors"]):
                maximum = max(maximum, _norm(sub(left_vector, right_vector)))
            for left_scalar, right_scalar in zip(
                    left["scalars"], right["scalars"]):
                maximum = max(maximum, abs(left_scalar - right_scalar))
        if maximum > smoke_tolerance:
            failures.append(Failure(path,
                                    "drag added an unadvertised reverse smoke force",
                                    maximum, smoke_tolerance))
        return maximum

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "representation", "warmup_step_count", "substeps",
                      "smoke_particle_capacity", "smoke_grid_resolution",
                      "smoke_grid_vertical_resolution", "declared_mode_flags",
                      "exercised_mode", "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in ("timestep_seconds", "fluid_initial_temperature_celsius",
                      "fluid_particle_radius_m", "fluid_support_radius_m",
                      "fluid_rest_density_kg_m3",
                      "fluid_rest_particle_volume_m3",
                      "smoke_emission_rate_per_s", "smoke_lifetime_seconds",
                      "smoke_particle_radius_m", "smoke_rest_number_density",
                      "smoke_maximum_speed_m_s", "smoke_grid_edge_length_m",
                      "thermal_transfer_rate_per_s",
                      "boiling_temperature_celsius", "control_wind_drag_per_s",
                      "test_wind_drag_per_s"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("gravity_m_s2", "fluid_initial_position_m",
                      "smoke_emitter_center_m", "smoke_grid_minimum_m",
                      "carrier_velocity_m_s",
                      "test_fluid_initial_velocity_m_s",
                      "matched_fluid_initial_velocity_m_s"):
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        close_sequence(actual.get("smoke_emitter_half_extents_m"),
                       definition.get("smoke_emitter_half_extents_m"),
                       base + ".smoke_emitter_half_extents_m")
        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array", runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        expected_runs = {"control", "drag", "matched-control", "matched-drag"}
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(base + ".runs",
                                    "expected exactly the four paired runs",
                                    sorted(indexed), sorted(expected_runs)))
            continue
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial")
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final")
            if initial is not None and final is not None:
                states[run_id] = (initial, final)
        if set(states) != expected_runs:
            continue

        control_initial, control_final = states["control"]
        drag_initial, drag_final = states["drag"]
        matched_control_initial, matched_control_final = states["matched-control"]
        matched_drag_initial, matched_drag_final = states["matched-drag"]
        close_vector(control_initial["velocity"],
                     definition["test_fluid_initial_velocity_m_s"],
                     base + ".runs[control].initial.fluid.velocity_m_s")
        close_vector(control_initial["position"],
                     definition["fluid_initial_position_m"],
                     base + ".runs[control].initial.fluid.position_m")
        close_vector(drag_initial["velocity"],
                     definition["test_fluid_initial_velocity_m_s"],
                     base + ".runs[drag].initial.fluid.velocity_m_s")
        close_vector(matched_control_initial["velocity"],
                     definition["matched_fluid_initial_velocity_m_s"],
                     base + ".runs[matched-control].initial.fluid.velocity_m_s")
        close_vector(matched_drag_initial["velocity"],
                     definition["matched_fluid_initial_velocity_m_s"],
                     base + ".runs[matched-drag].initial.fluid.velocity_m_s")

        control_drift = _norm(sub(control_final["velocity"],
                                  control_initial["velocity"]))
        allowed_control = float(tolerances.get(
            "maximum_control_velocity_drift_m_s", 0.0))
        if control_drift > allowed_control:
            failures.append(Failure(base + ".runs[control]",
                                    "zero-drag control velocity drifted",
                                    control_drift, allowed_control))
        delta = sub(drag_final["velocity"], control_final["velocity"])
        carrier = _vector(definition.get("carrier_velocity_m_s"),
                          base + ".carrier_velocity_m_s", failures)
        if carrier is not None:
            relative = sub(carrier, control_final["velocity"])
            denominator = sum(component * component for component in relative)
            response = (sum(component * axis for component, axis in
                            zip(delta, relative)) / denominator
                        if denominator > 1.0e-12 else 0.0)
            transverse = _norm(tuple(component - response * axis
                                     for component, axis in zip(delta, relative)))
            minimum = float(tolerances.get(
                "minimum_drag_response_fraction", 0.0))
            maximum = float(tolerances.get(
                "maximum_drag_response_fraction", 0.0))
            if response < minimum or response > maximum:
                failures.append(Failure(base + ".runs[drag]",
                                        "fluid drag response is outside bounds",
                                        response, [minimum, maximum]))
            allowed_transverse = float(tolerances.get(
                "maximum_transverse_velocity_m_s", 0.0))
            if transverse > allowed_transverse:
                failures.append(Failure(base + ".runs[drag]",
                                        "fluid drag is not aligned with smoke flow",
                                        transverse, allowed_transverse))
        matched_difference = _norm(sub(matched_drag_final["velocity"],
                                       matched_control_final["velocity"]))
        allowed_matched = float(tolerances.get(
            "maximum_matched_velocity_difference_m_s", 0.0))
        if matched_difference > allowed_matched:
            failures.append(Failure(base + ".runs[matched-drag]",
                                    "matched flow produced drag",
                                    matched_difference, allowed_matched))
        reverse_delta = compare_smoke(control_final, drag_final,
                                      base + ".runs[drag].final.smoke_particles")
        matched_reverse_delta = compare_smoke(
            matched_control_final, matched_drag_final,
            base + ".runs[matched-drag].final.smoke_particles")
        observations[identity] = {
            "control_velocity_drift_m_s": control_drift,
            "drag_velocity_delta_m_s": list(delta),
            "matched_velocity_difference_m_s": matched_difference,
            "maximum_reverse_smoke_state_delta": max(
                reverse_delta, matched_reverse_delta),
        }
        sample_count += 2

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_smoke_cloth_drag(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    soft_mode = profile.get("evaluation") == "smoke_soft_drag"
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    smoke_tolerance = float(tolerances.get("smoke_state_absolute", 0.0))

    def sub(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(first, second))

    def dot(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(first, second))

    def add(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(first, second))

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)

    def closest_on_triangle(
            point: tuple[float, float, float],
            first: tuple[float, float, float],
            second: tuple[float, float, float],
            third: tuple[float, float, float]) -> tuple[float, float, float]:
        first_second = sub(second, first)
        first_third = sub(third, first)
        first_point = sub(point, first)
        d1, d2 = dot(first_second, first_point), dot(first_third, first_point)
        if d1 <= 0.0 and d2 <= 0.0:
            return first
        second_point = sub(point, second)
        d3 = dot(first_second, second_point)
        d4 = dot(first_third, second_point)
        if d3 >= 0.0 and d4 <= d3:
            return second
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            return add(first, scale(first_second, d1 / (d1 - d3)))
        third_point = sub(point, third)
        d5 = dot(first_second, third_point)
        d6 = dot(first_third, third_point)
        if d6 >= 0.0 and d5 <= d6:
            return third
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            return add(first, scale(first_third, d2 / (d2 - d6)))
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            edge = sub(third, second)
            weight = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return add(second, scale(edge, weight))
        denominator = 1.0 / (va + vb + vc)
        return add(first, add(scale(first_second, vb * denominator),
                              scale(first_third, vc * denominator)))

    def close_contract(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, bool) or isinstance(expected, str):
            if actual != expected:
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
            return
        if isinstance(expected, (int, float)):
            value = _scalar(actual, path, failures)
            if value is not None and abs(value - float(expected)) > state_tolerance:
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        value, expected))
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
                return
            for index, (value, reference) in enumerate(zip(actual, expected)):
                close_contract(value, reference, f"{path}[{index}]")
            return
        if actual != expected:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def snapshot(value: Any, path: str,
                 definition: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        cloth = value.get("cloth")
        particles = value.get("smoke_particles")
        expected_nodes = definition.get("cloth_vertices_m")
        expected_count = len(expected_nodes) if isinstance(
            expected_nodes, list) else 0
        if (not isinstance(cloth, list) or len(cloth) != expected_count or
                expected_count == 0):
            failures.append(Failure(path + ".cloth",
                                    "unexpected deformable node count", cloth,
                                    expected_count))
            return None
        if not isinstance(particles, list) or not particles:
            failures.append(Failure(path + ".smoke_particles",
                                    "expected emitted smoke particles", particles))
            return None
        cloth_positions: list[tuple[float, float, float]] = []
        cloth_velocities: list[tuple[float, float, float]] = []
        for index, vertex in enumerate(cloth):
            vertex_path = f"{path}.cloth[{index}]"
            if not isinstance(vertex, dict):
                failures.append(Failure(vertex_path, "expected an object", vertex))
                continue
            position = _vector(vertex.get("position_m"), vertex_path +
                               ".position_m", failures)
            velocity = _vector(vertex.get("velocity_m_s"), vertex_path +
                               ".velocity_m_s", failures)
            if position is not None and velocity is not None:
                cloth_positions.append(position)
                cloth_velocities.append(velocity)
        normalized_particles: list[dict[str, Any]] = []
        for index, particle in enumerate(particles):
            particle_path = f"{path}.smoke_particles[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(particle_path, "expected an object",
                                        particle))
                continue
            vectors = [_vector(particle.get(field), particle_path + "." + field,
                               failures)
                       for field in ("position_m", "velocity_m_s",
                                     "vorticity_per_s")]
            scalars = [_scalar(particle.get(field), particle_path + "." + field,
                               failures)
                       for field in ("age_seconds", "number_density",
                                     "pressure_m_s2")]
            if any(item is None for item in vectors + scalars):
                continue
            normalized_particles.append({
                "vectors": [item for item in vectors if item is not None],
                "scalars": [item for item in scalars if item is not None],
            })
        if (len(cloth_positions) != len(cloth) or
                len(normalized_particles) != len(particles)):
            return None
        surface_positions = cloth_positions
        if soft_mode:
            raw_surface = value.get("cloth_surface_vertices_m")
            expected_surface = definition.get("cloth_surface_vertices_m")
            expected_surface_count = len(expected_surface) if isinstance(
                expected_surface, list) else 0
            if (not isinstance(raw_surface, list) or
                    len(raw_surface) != expected_surface_count or
                    expected_surface_count == 0):
                failures.append(Failure(
                    path + ".cloth_surface_vertices_m",
                    "unexpected public soft surface vertex count",
                    raw_surface, expected_surface_count))
                return None
            parsed_surface = [
                _vector(item,
                        f"{path}.cloth_surface_vertices_m[{index}]", failures)
                for index, item in enumerate(raw_surface)]
            if any(item is None for item in parsed_surface):
                return None
            surface_positions = [item for item in parsed_surface
                                 if item is not None]
            bindings = definition.get("cloth_surface_bindings")
            if not isinstance(bindings, list) or len(bindings) != len(
                    surface_positions):
                failures.append(Failure(
                    path + ".cloth_surface_vertices_m",
                    "soft surface bindings are incomplete", bindings))
                return None
            reconstruction_limit = float(tolerances.get(
                "maximum_surface_reconstruction_error_m", 0.0))
            for index, (surface, binding) in enumerate(zip(
                    surface_positions, bindings)):
                if (not isinstance(binding, dict) or
                        not isinstance(binding.get("nodes"), list) or
                        not isinstance(binding.get("weights"), list)):
                    failures.append(Failure(
                        path + f".cloth_surface_vertices_m[{index}]",
                        "invalid soft surface binding", binding))
                    continue
                reconstructed = (0.0, 0.0, 0.0)
                valid = True
                for node, weight in zip(binding["nodes"], binding["weights"]):
                    if (not isinstance(node, int) or isinstance(node, bool) or
                            node < 0 or node >= len(cloth_positions) or
                            not isinstance(weight, (int, float)) or
                            isinstance(weight, bool)):
                        valid = False
                        break
                    reconstructed = add(
                        reconstructed,
                        scale(cloth_positions[node], float(weight)))
                if not valid:
                    failures.append(Failure(
                        path + f".cloth_surface_vertices_m[{index}]",
                        "invalid soft surface binding", binding))
                    continue
                error = _norm(sub(surface, reconstructed))
                if error > reconstruction_limit:
                    failures.append(Failure(
                        path + f".cloth_surface_vertices_m[{index}]",
                        "public soft surface does not match node bindings",
                        error, reconstruction_limit))
        mean_velocity = tuple(sum(value[axis] for value in cloth_velocities) /
                              len(cloth_velocities) for axis in range(3))
        raw_indices = definition.get("cloth_triangle_indices")
        if (not isinstance(raw_indices, list) or not raw_indices or
                len(raw_indices) % 3 != 0 or
                any(not isinstance(index, int) or isinstance(index, bool) or
                    index < 0 or index >= len(surface_positions)
                    for index in raw_indices)):
            failures.append(Failure(path + ".cloth_triangle_indices",
                                    "invalid public surface triangles",
                                    raw_indices))
            return None
        clearance = float("inf")
        for particle in normalized_particles:
            point = particle["vectors"][0]
            for offset in range(0, len(raw_indices), 3):
                first, second, third = (surface_positions[raw_indices[offset + i]]
                                        for i in range(3))
                closest = closest_on_triangle(point, first, second, third)
                clearance = min(clearance, _norm(sub(point, closest)))
        return {"velocity": mean_velocity, "smoke": normalized_particles,
                "clearance": clearance, "positions": cloth_positions,
                "velocities": cloth_velocities,
                "surface": surface_positions}

    def compare_smoke(first: dict[str, Any], second: dict[str, Any],
                      path: str) -> float:
        left = first["smoke"]
        right = second["smoke"]
        if len(left) != len(right):
            failures.append(Failure(path, "drag changed smoke particle count",
                                    len(right), len(left)))
            return float("inf")
        maximum = 0.0
        for first_particle, second_particle in zip(left, right):
            for first_vector, second_vector in zip(
                    first_particle["vectors"], second_particle["vectors"]):
                maximum = max(maximum, _norm(sub(first_vector, second_vector)))
            for first_scalar, second_scalar in zip(
                    first_particle["scalars"], second_particle["scalars"]):
                maximum = max(maximum, abs(first_scalar - second_scalar))
        if maximum > smoke_tolerance:
            failures.append(Failure(path,
                                    "drag added an unadvertised reverse smoke force",
                                    maximum, smoke_tolerance))
        return maximum

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field, expected in definition.items():
            if field not in ("runs",):
                close_contract(actual.get(field), expected, base + "." + field)
        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array", runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        expected_runs = ({"forward-control", "forward",
                          "stationary-control", "stationary",
                          "reverse-control", "reverse"}
                         if soft_mode else
                         {"control", "forward", "stationary", "reverse"})
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(base + ".runs",
                                    "expected exactly the paired drag runs",
                                    sorted(indexed), sorted(expected_runs)))
            continue
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial", definition)
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final", definition)
            if initial is not None and final is not None:
                states[run_id] = (initial, final)
        if set(states) != expected_runs:
            continue
        for run_id, (initial, final) in states.items():
            initial_speed = _norm(initial["velocity"])
            if initial_speed > state_tolerance:
                failures.append(Failure(base + f".runs[{run_id}].initial.cloth",
                                        "cloth must start at rest", initial_speed,
                                        state_tolerance))
            minimum_clearance = float(tolerances.get(
                "minimum_tracer_plane_clearance_m", 0.0))
            observed_clearance = min(initial["clearance"], final["clearance"])
            if observed_clearance < minimum_clearance:
                failures.append(Failure(base + f".runs[{run_id}]",
                                        "smoke tracer entered the contact clearance",
                                        observed_clearance, minimum_clearance))

        if soft_mode:
            control_pairs = {
                "forward": "forward-control",
                "stationary": "stationary-control",
                "reverse": "reverse-control",
            }
            for run_id, control_id in control_pairs.items():
                control_initial = states[control_id][0]
                test_initial = states[run_id][0]
                paired_delta = max(
                    [_norm(sub(first, second)) for first, second in zip(
                        control_initial["positions"],
                        test_initial["positions"])] +
                    [_norm(sub(first, second)) for first, second in zip(
                        control_initial["velocities"],
                        test_initial["velocities"])] +
                    [_norm(sub(first, second)) for first, second in zip(
                        control_initial["surface"], test_initial["surface"])]
                )
                if paired_delta > state_tolerance:
                    failures.append(Failure(
                        base + f".runs[{run_id}].initial.cloth",
                        "paired initial soft body differs from control",
                        paired_delta, state_tolerance))
                compare_smoke(
                    control_initial, test_initial,
                    base + f".runs[{run_id}].initial.smoke_particles")
            forward_control_initial, forward_control_final = states[
                "forward-control"]
            _, stationary_control_final = states["stationary-control"]
            _, reverse_control_final = states["reverse-control"]
        else:
            forward_control_initial, forward_control_final = states["control"]
            stationary_control_final = forward_control_final
            reverse_control_final = forward_control_final
        _, forward_final = states["forward"]
        _, stationary_final = states["stationary"]
        _, reverse_final = states["reverse"]
        control_drift = _norm(sub(forward_control_final["velocity"],
                                  forward_control_initial["velocity"]))
        allowed_control = float(tolerances.get(
            "maximum_control_velocity_drift_m_s", 0.0))
        if control_drift > allowed_control:
            failures.append(Failure(base + ".runs[forward-control]",
                                    "zero-drag cloth velocity drifted",
                                    control_drift, allowed_control))

        forward_delta = sub(forward_final["velocity"],
                            forward_control_final["velocity"])
        reverse_delta = sub(reverse_final["velocity"],
                            reverse_control_final["velocity"])
        forward_carrier = _vector(definition.get("forward_carrier_velocity_m_s"),
                                  base + ".forward_carrier_velocity_m_s", failures)
        reverse_carrier = _vector(definition.get("reverse_carrier_velocity_m_s"),
                                  base + ".reverse_carrier_velocity_m_s", failures)

        def response(delta: tuple[float, float, float],
                     carrier: tuple[float, float, float] | None,
                     run_id: str) -> tuple[float, float]:
            if carrier is None:
                return 0.0, float("inf")
            denominator = sum(component * component for component in carrier)
            fraction = (sum(component * axis for component, axis in
                            zip(delta, carrier)) / denominator
                        if denominator > 1.0e-12 else 0.0)
            transverse = _norm(tuple(component - fraction * axis
                                     for component, axis in zip(delta, carrier)))
            minimum = float(tolerances.get("minimum_drag_response_fraction", 0.0))
            maximum = float(tolerances.get("maximum_drag_response_fraction", 0.0))
            if fraction < minimum or fraction > maximum:
                failures.append(Failure(base + f".runs[{run_id}]",
                                        "cloth drag response is outside bounds",
                                        fraction, [minimum, maximum]))
            allowed_transverse = float(tolerances.get(
                "maximum_transverse_velocity_m_s", 0.0))
            if transverse > allowed_transverse:
                failures.append(Failure(base + f".runs[{run_id}]",
                                        "cloth drag is not aligned with smoke flow",
                                        transverse, allowed_transverse))
            return fraction, transverse

        forward_response, forward_transverse = response(
            forward_delta, forward_carrier, "forward")
        reverse_response, reverse_transverse = response(
            reverse_delta, reverse_carrier, "reverse")
        stationary_speed = _norm(sub(
            stationary_final["velocity"],
            stationary_control_final["velocity"]))
        allowed_stationary = float(tolerances.get(
            "maximum_stationary_velocity_m_s", 0.0))
        if stationary_speed > allowed_stationary:
            failures.append(Failure(base + ".runs[stationary]",
                                    "stationary smoke produced cloth drag",
                                    stationary_speed, allowed_stationary))
        symmetry_error = abs(forward_response - reverse_response)
        allowed_symmetry = float(tolerances.get(
            "maximum_forward_reverse_response_difference", 0.0))
        if symmetry_error > allowed_symmetry:
            failures.append(Failure(base + ".runs",
                                    "forward and reverse drag responses differ",
                                    symmetry_error, allowed_symmetry))
        smoke_delta = compare_smoke(
            forward_control_final, forward_final,
            base + ".runs[forward].final.smoke_particles")
        if soft_mode:
            smoke_delta = max(
                smoke_delta,
                compare_smoke(
                    stationary_control_final, stationary_final,
                    base + ".runs[stationary].final.smoke_particles"),
                compare_smoke(
                    reverse_control_final, reverse_final,
                    base + ".runs[reverse].final.smoke_particles"))
            timestep = float(definition.get("timestep_seconds", 0.0))
            acceleration = max(_norm(forward_delta), _norm(reverse_delta)) / max(
                timestep, 1.0e-30)
            acceleration_limit = float(definition.get(
                "maximum_wind_acceleration_m_s2", 0.0)) + float(
                    tolerances.get("maximum_acceleration_slack_m_s2", 0.0))
            if acceleration > acceleration_limit:
                failures.append(Failure(
                    base + ".runs", "soft body drag exceeds acceleration cap",
                    acceleration, acceleration_limit))
            speed_limit = float(definition.get(
                "soft_body_maximum_speed_m_s", 0.0)) + float(
                    tolerances.get("maximum_speed_slack_m_s", 0.0))
            maximum_speed = max(
                _norm(velocity) for state in states.values()
                for snapshot_state in state
                for velocity in snapshot_state["velocities"])
            if maximum_speed > speed_limit:
                failures.append(Failure(
                    base + ".runs", "soft body node exceeds speed cap",
                    maximum_speed, speed_limit))
        else:
            acceleration = None
            maximum_speed = None
        observations[identity] = {
            "control_velocity_drift_m_s": control_drift,
            "forward_response_fraction": forward_response,
            "reverse_response_fraction": reverse_response,
            "forward_transverse_velocity_m_s": forward_transverse,
            "reverse_transverse_velocity_m_s": reverse_transverse,
            "stationary_velocity_m_s": stationary_speed,
            "forward_reverse_response_difference": symmetry_error,
            "minimum_tracer_plane_clearance_m": min(
                state[stage]["clearance"] for state in states.values()
                for stage in (0, 1)),
            "maximum_reverse_smoke_state_delta": smoke_delta,
            "maximum_drag_acceleration_m_s2": acceleration,
            "maximum_soft_body_node_speed_m_s": maximum_speed,
        }
        sample_count += 3 if soft_mode else 2

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_smoke_soft_drag(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    """Reuse the public deformable drag oracle with soft-body names."""
    key_map = {
        "soft_body": "cloth",
        "soft_body_nodes_m": "cloth_vertices_m",
        "soft_body_surface_vertices_m": "cloth_surface_vertices_m",
        "soft_body_surface_triangle_indices": "cloth_triangle_indices",
        "soft_body_surface_bindings": "cloth_surface_bindings",
        "soft_body_node_mass_kg": "cloth_vertex_mass_kg",
        "soft_body_node_radius_m": "cloth_thickness_m",
        "soft_body_velocity_damping_per_s": "cloth_velocity_damping_per_s",
        "soft_body_solver_iterations": "cloth_solver_iterations",
        "minimum_tracer_surface_clearance_m":
            "minimum_tracer_plane_clearance_m",
    }

    def remap(value: Any) -> Any:
        if isinstance(value, dict):
            return {key_map.get(key, key): remap(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [remap(item) for item in value]
        return copy.deepcopy(value)

    report = _evaluate_smoke_cloth_drag(
        remap(profile), remap(result), expected_hash, driver, failures)
    inverse = {target: source for source, target in key_map.items()}
    for failure in report["failures"]:
        path = failure["path"]
        for target, source in inverse.items():
            path = path.replace(target, source)
        failure["path"] = path
        failure["message"] = failure["message"].replace(
            "cloth", "soft body")
    for observation in report["observations"].values():
        if "minimum_tracer_plane_clearance_m" in observation:
            observation["minimum_tracer_surface_clearance_m"] = observation.pop(
                "minimum_tracer_plane_clearance_m")
    return report


def _evaluate_smoke_rope_drag(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if (not isinstance(fixture, dict) or
                not isinstance(fixture.get("id"), str)):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    smoke_tolerance = float(tolerances.get("smoke_state_absolute", 0.0))

    def close_contract(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                failures.append(Failure(path, "fixture setting differs from profile",
                                        actual, expected))
                return
            for key, value in expected.items():
                close_contract(actual.get(key), value, path + "." + key)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append(Failure(path, "fixture setting differs from profile",
                                        actual, expected))
                return
            for index, (item, value) in enumerate(zip(actual, expected)):
                close_contract(item, value, f"{path}[{index}]")
        elif _number(expected):
            if (not _number(actual) or
                    abs(float(actual) - float(expected)) > state_tolerance):
                failures.append(Failure(path, "fixture setting differs from profile",
                                        actual, expected))
        elif actual != expected:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def closest_segment_distance(
            point: tuple[float, float, float],
            first: tuple[float, float, float],
            second: tuple[float, float, float]) -> float:
        edge = _subtract(second, first)
        denominator = dot(edge, edge)
        fraction = (max(0.0, min(1.0, dot(_subtract(point, first), edge) /
                                 denominator))
                    if denominator > 1.0e-30 else 0.0)
        closest = tuple(first[axis] + fraction * edge[axis]
                        for axis in range(3))
        return _norm(_subtract(point, closest))

    def smoke_particles(value: Any, path: str) -> list[dict[str, Any]] | None:
        if not isinstance(value, list) or not value:
            failures.append(Failure(path, "expected emitted smoke particles", value))
            return None
        parsed: list[dict[str, Any]] = []
        slots: set[int] = set()
        for index, particle in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(item_path, "expected an object", particle))
                continue
            slot = particle.get("slot")
            if (not isinstance(slot, int) or isinstance(slot, bool) or slot < 0):
                failures.append(Failure(item_path + ".slot",
                                        "expected a nonnegative integer", slot))
                continue
            if slot in slots:
                failures.append(Failure(item_path + ".slot",
                                        "duplicate smoke slot", slot))
                continue
            slots.add(slot)
            position = _vector(particle.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(particle.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            vorticity = _vector(particle.get("vorticity_per_s"),
                                item_path + ".vorticity_per_s", failures)
            age = _scalar(particle.get("age_seconds"),
                          item_path + ".age_seconds", failures)
            density = _scalar(particle.get("number_density"),
                              item_path + ".number_density", failures)
            pressure = _scalar(particle.get("pressure_m_s2"),
                               item_path + ".pressure_m_s2", failures)
            if (position is None or velocity is None or vorticity is None or
                    age is None or density is None or pressure is None):
                continue
            parsed.append({
                "slot": slot,
                "vectors": (position, velocity, vorticity),
                "scalars": (age, density, pressure),
            })
        if len(parsed) != len(value):
            return None
        return sorted(parsed, key=lambda item: item["slot"])

    def snapshot(value: Any, path: str, definition: dict[str, Any]
                 ) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        expected_count = definition.get("rope_node_count")
        raw_rope = value.get("rope")
        if (not isinstance(expected_count, int) or
                isinstance(expected_count, bool) or expected_count < 2):
            failures.append(Failure("$.profile.fixture.rope_node_count",
                                    "expected at least two rope nodes",
                                    expected_count))
            return None
        if not isinstance(raw_rope, list) or len(raw_rope) != expected_count:
            failures.append(Failure(path + ".rope",
                                    "unexpected rope node count", raw_rope,
                                    expected_count))
            return None
        indexed_nodes: dict[int, tuple[tuple[float, float, float],
                                       tuple[float, float, float]]] = {}
        for offset, node in enumerate(raw_rope):
            node_path = f"{path}.rope[{offset}]"
            if not isinstance(node, dict):
                failures.append(Failure(node_path, "expected an object", node))
                continue
            index = node.get("index")
            if (not isinstance(index, int) or isinstance(index, bool) or
                    index < 0 or index >= expected_count):
                failures.append(Failure(node_path + ".index",
                                        "invalid rope node index", index))
                continue
            if index in indexed_nodes:
                failures.append(Failure(node_path + ".index",
                                        "duplicate rope node index", index))
                continue
            position = _vector(node.get("position_m"),
                               node_path + ".position_m", failures)
            velocity = _vector(node.get("velocity_m_s"),
                               node_path + ".velocity_m_s", failures)
            if position is not None and velocity is not None:
                indexed_nodes[index] = position, velocity
        particles = smoke_particles(value.get("smoke_particles"),
                                    path + ".smoke_particles")
        if len(indexed_nodes) != expected_count or particles is None:
            return None
        node_positions = [indexed_nodes[index][0]
                          for index in range(expected_count)]
        node_velocities = [indexed_nodes[index][1]
                           for index in range(expected_count)]
        centerline_raw = definition.get("rope_centerline_m")
        if (not isinstance(centerline_raw, list) or
                len(centerline_raw) != expected_count):
            failures.append(Failure("$.profile.fixture.rope_centerline_m",
                                    "unexpected rope centerline", centerline_raw,
                                    expected_count))
            return None
        centerline = [_vector(item,
                              f"$.profile.fixture.rope_centerline_m[{index}]",
                              failures)
                      for index, item in enumerate(centerline_raw)]
        if any(item is None for item in centerline):
            return None
        expected_positions = [item for item in centerline if item is not None]
        segment_error = max(
            abs(_norm(_subtract(node_positions[index + 1],
                                node_positions[index])) -
                _norm(_subtract(expected_positions[index + 1],
                                expected_positions[index])))
            for index in range(expected_count - 1))
        clearance = min(
            closest_segment_distance(
                particle["vectors"][0], node_positions[index],
                node_positions[index + 1])
            for particle in particles
            for index in range(expected_count - 1))
        mean_velocity = tuple(
            sum(item[axis] for item in node_velocities) / expected_count
            for axis in range(3))
        return {
            "positions": node_positions,
            "velocities": node_velocities,
            "smoke": particles,
            "mean_velocity": mean_velocity,
            "segment_error": segment_error,
            "clearance": clearance,
        }

    def rope_delta(first: dict[str, Any], second: dict[str, Any]) -> float:
        return max(
            [_norm(_subtract(left, right)) for left, right in zip(
                first["positions"], second["positions"])] +
            [_norm(_subtract(left, right)) for left, right in zip(
                first["velocities"], second["velocities"])])

    def compare_smoke(first: dict[str, Any], second: dict[str, Any],
                      path: str, message: str) -> float:
        left = first["smoke"]
        right = second["smoke"]
        if len(left) != len(right):
            failures.append(Failure(path, "drag changed smoke particle count",
                                    len(right), len(left)))
            return float("inf")
        maximum = 0.0
        for first_particle, second_particle in zip(left, right):
            if first_particle["slot"] != second_particle["slot"]:
                failures.append(Failure(path, "smoke slot identity changed",
                                        second_particle["slot"],
                                        first_particle["slot"]))
                maximum = float("inf")
                continue
            for first_vector, second_vector in zip(
                    first_particle["vectors"], second_particle["vectors"]):
                maximum = max(maximum,
                              _norm(_subtract(first_vector, second_vector)))
            for first_scalar, second_scalar in zip(
                    first_particle["scalars"], second_particle["scalars"]):
                maximum = max(maximum, abs(first_scalar - second_scalar))
        if maximum > smoke_tolerance:
            failures.append(Failure(path, message, maximum, smoke_tolerance))
        return maximum

    observations: dict[str, Any] = {}
    sample_count = 0
    expected_runs = {"forward-control", "forward", "stationary-control",
                     "stationary", "reverse-control", "reverse"}
    pairs = {"forward": "forward-control",
             "stationary": "stationary-control",
             "reverse": "reverse-control"}
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field, expected in definition.items():
            close_contract(actual.get(field), expected, base + "." + field)
        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array", runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(base + ".runs",
                                    "expected exactly the six paired drag runs",
                                    sorted(indexed), sorted(expected_runs)))
            continue
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial", definition)
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final", definition)
            if initial is not None and final is not None:
                states[run_id] = initial, final
        if set(states) != expected_runs:
            continue

        maximum_initial_speed = max(
            _norm(velocity) for state in states.values()
            for velocity in state[0]["velocities"])
        if maximum_initial_speed > state_tolerance:
            failures.append(Failure(base + ".runs",
                                    "rope must start at rest",
                                    maximum_initial_speed, state_tolerance))

        maximum_initial_rope_delta = 0.0
        maximum_initial_smoke_delta = 0.0
        for run_id, control_id in pairs.items():
            control_initial = states[control_id][0]
            test_initial = states[run_id][0]
            delta = rope_delta(control_initial, test_initial)
            maximum_initial_rope_delta = max(maximum_initial_rope_delta, delta)
            if delta > state_tolerance:
                failures.append(Failure(
                    base + f".runs[{run_id}].initial",
                    "paired initial rope state differs from control",
                    delta, state_tolerance))
            maximum_initial_smoke_delta = max(
                maximum_initial_smoke_delta,
                compare_smoke(
                    control_initial, test_initial,
                    base + f".runs[{run_id}].initial.smoke_particles",
                    "paired initial smoke state differs from control"))

        control_drift = max(
            _norm(_subtract(final_velocity, initial_velocity))
            for control_id in pairs.values()
            for initial_velocity, final_velocity in zip(
                states[control_id][0]["velocities"],
                states[control_id][1]["velocities"]))
        control_limit = float(tolerances.get(
            "maximum_control_velocity_drift_m_s", 0.0))
        if control_drift > control_limit:
            failures.append(Failure(base + ".runs",
                                    "zero-drag rope velocity drifted",
                                    control_drift, control_limit))

        forward_final = states["forward"][1]
        forward_control_final = states["forward-control"][1]
        stationary_final = states["stationary"][1]
        stationary_control_final = states["stationary-control"][1]
        reverse_final = states["reverse"][1]
        reverse_control_final = states["reverse-control"][1]
        forward_delta = _subtract(forward_final["mean_velocity"],
                                  forward_control_final["mean_velocity"])
        reverse_delta = _subtract(reverse_final["mean_velocity"],
                                  reverse_control_final["mean_velocity"])
        stationary_delta = _subtract(stationary_final["mean_velocity"],
                                     stationary_control_final["mean_velocity"])
        forward_carrier = _vector(definition.get("forward_carrier_velocity_m_s"),
                                  base + ".forward_carrier_velocity_m_s", failures)
        reverse_carrier = _vector(definition.get("reverse_carrier_velocity_m_s"),
                                  base + ".reverse_carrier_velocity_m_s", failures)

        def response(delta: tuple[float, float, float],
                     carrier: tuple[float, float, float] | None,
                     run_id: str) -> tuple[float, float]:
            if carrier is None:
                return 0.0, float("inf")
            denominator = dot(carrier, carrier)
            fraction = dot(delta, carrier) / denominator if denominator > 1e-30 else 0.0
            transverse = _norm(tuple(delta[axis] - fraction * carrier[axis]
                                     for axis in range(3)))
            minimum = float(tolerances.get(
                "minimum_drag_response_fraction", 0.0))
            maximum = float(tolerances.get(
                "maximum_drag_response_fraction", 0.0))
            if fraction < minimum or fraction > maximum:
                failures.append(Failure(base + f".runs[{run_id}]",
                                        "rope drag response is outside bounds",
                                        fraction, [minimum, maximum]))
            transverse_limit = float(tolerances.get(
                "maximum_transverse_velocity_m_s", 0.0))
            if transverse > transverse_limit:
                failures.append(Failure(base + f".runs[{run_id}]",
                                        "rope drag is not aligned with smoke flow",
                                        transverse, transverse_limit))
            return fraction, transverse

        forward_response, forward_transverse = response(
            forward_delta, forward_carrier, "forward")
        reverse_response, reverse_transverse = response(
            reverse_delta, reverse_carrier, "reverse")
        stationary_speed = _norm(stationary_delta)
        stationary_limit = float(tolerances.get(
            "maximum_stationary_velocity_m_s", 0.0))
        if stationary_speed > stationary_limit:
            failures.append(Failure(base + ".runs[stationary]",
                                    "stationary smoke produced rope drag",
                                    stationary_speed, stationary_limit))
        symmetry_error = abs(forward_response - reverse_response)
        symmetry_limit = float(tolerances.get(
            "maximum_forward_reverse_response_difference", 0.0))
        if symmetry_error > symmetry_limit:
            failures.append(Failure(base + ".runs",
                                    "forward and reverse rope responses differ",
                                    symmetry_error, symmetry_limit))

        smoke_delta = 0.0
        for run_id, control_id in pairs.items():
            smoke_delta = max(
                smoke_delta,
                compare_smoke(
                    states[control_id][1], states[run_id][1],
                    base + f".runs[{run_id}].final.smoke_particles",
                    "drag added an unadvertised reverse smoke force"))

        timestep = float(definition.get("timestep_seconds", 0.0))
        acceleration = max(
            _norm(_subtract(test_velocity, control_velocity)) /
            max(timestep, 1.0e-30)
            for run_id, control_id in pairs.items()
            for test_velocity, control_velocity in zip(
                states[run_id][1]["velocities"],
                states[control_id][1]["velocities"]))
        acceleration_limit = float(definition.get(
            "maximum_wind_acceleration_m_s2", 0.0)) + float(tolerances.get(
                "maximum_acceleration_slack_m_s2", 0.0))
        if acceleration > acceleration_limit:
            failures.append(Failure(base + ".runs",
                                    "rope drag exceeds acceleration cap",
                                    acceleration, acceleration_limit))

        maximum_speed = max(
            _norm(velocity) for state in states.values()
            for snapshot_state in state
            for velocity in snapshot_state["velocities"])
        speed_limit = float(definition.get("rope_maximum_speed_m_s", 0.0)) + float(
            tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure(base + ".runs",
                                    "rope node exceeds speed cap",
                                    maximum_speed, speed_limit))

        segment_error = max(
            snapshot_state["segment_error"] for state in states.values()
            for snapshot_state in state)
        segment_limit = float(tolerances.get(
            "maximum_segment_length_error_m", 0.0))
        if segment_error > segment_limit:
            failures.append(Failure(base + ".runs",
                                    "rope segment length changed beyond bounds",
                                    segment_error, segment_limit))

        clearance = min(
            snapshot_state["clearance"] for state in states.values()
            for snapshot_state in state)
        clearance_limit = float(tolerances.get(
            "minimum_tracer_segment_clearance_m", 0.0))
        if clearance < clearance_limit:
            failures.append(Failure(base + ".runs",
                                    "smoke tracer entered non-contact segment clearance",
                                    clearance, clearance_limit))

        observations[identity] = {
            "maximum_control_velocity_drift_m_s": control_drift,
            "maximum_paired_initial_rope_state_delta":
                maximum_initial_rope_delta,
            "maximum_paired_initial_smoke_state_delta":
                maximum_initial_smoke_delta,
            "forward_response_fraction": forward_response,
            "reverse_response_fraction": reverse_response,
            "forward_transverse_velocity_m_s": forward_transverse,
            "reverse_transverse_velocity_m_s": reverse_transverse,
            "stationary_velocity_m_s": stationary_speed,
            "forward_reverse_response_difference": symmetry_error,
            "maximum_reverse_smoke_state_delta": smoke_delta,
            "maximum_drag_acceleration_m_s2": acceleration,
            "maximum_rope_node_speed_m_s": maximum_speed,
            "maximum_segment_length_error_m": segment_error,
            "minimum_tracer_segment_clearance_m": clearance,
        }
        sample_count += 3

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_smoke_rigid_drag(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if (not isinstance(fixture, dict) or
                not isinstance(fixture.get("id"), str)):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def scale(value: tuple[float, float, float],
              amount: float) -> tuple[float, float, float]:
        return tuple(component * amount for component in value)  # type: ignore[return-value]

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def rotate(value: tuple[float, float, float],
               quaternion: tuple[float, float, float, float]
               ) -> tuple[float, float, float]:
        imaginary = quaternion[0], quaternion[1], quaternion[2]
        twice_cross = scale(cross(imaginary, value), 2.0)
        return add(value, add(scale(twice_cross, quaternion[3]),
                              cross(imaginary, twice_cross)))

    def normalize_quaternion(
            value: tuple[float, float, float, float], path: str
            ) -> tuple[float, float, float, float]:
        length = math.sqrt(sum(component * component for component in value))
        if abs(length - 1.0) > state_tolerance:
            failures.append(Failure(path,
                                    "rigid orientation is not normalized",
                                    length, 1.0))
        if length <= 1.0e-30:
            return 0.0, 0.0, 0.0, 1.0
        return tuple(component / length for component in value)  # type: ignore[return-value]

    def quaternion_distance(
            left: tuple[float, float, float, float],
            right: tuple[float, float, float, float]) -> float:
        direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
        negated = math.sqrt(sum((a + b) ** 2 for a, b in zip(left, right)))
        return min(direct, negated)

    def close_contract(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
                return
            for key, wanted in expected.items():
                close_contract(actual.get(key), wanted, path + "." + key)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
                return
            for index, (item, wanted) in enumerate(zip(actual, expected)):
                close_contract(item, wanted, f"{path}[{index}]")
        elif _number(expected):
            if (not _number(actual) or
                    abs(float(actual) - float(expected)) > state_tolerance):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
        elif actual != expected:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def point_triangle_distance(
            point: tuple[float, float, float],
            first: tuple[float, float, float],
            second: tuple[float, float, float],
            third: tuple[float, float, float]) -> float:
        ab = _subtract(second, first)
        ac = _subtract(third, first)
        ap = _subtract(point, first)
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return _norm(ap)
        bp = _subtract(point, second)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return _norm(bp)
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            fraction = d1 / (d1 - d3)
            return _norm(_subtract(point, add(first, scale(ab, fraction))))
        cp = _subtract(point, third)
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return _norm(cp)
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            fraction = d2 / (d2 - d6)
            return _norm(_subtract(point, add(first, scale(ac, fraction))))
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
            edge = _subtract(third, second)
            fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return _norm(_subtract(point, add(second, scale(edge, fraction))))
        denominator = va + vb + vc
        if abs(denominator) <= 1.0e-30:
            return min(_norm(ap), _norm(bp), _norm(cp))
        inverse = 1.0 / denominator
        closest = add(first, add(scale(ab, vb * inverse),
                                 scale(ac, vc * inverse)))
        return _norm(_subtract(point, closest))

    def point_inside_convex(
            point: tuple[float, float, float],
            vertices: list[tuple[float, float, float]],
            triangles: list[tuple[int, int, int]]) -> bool:
        center = tuple(sum(vertex[axis] for vertex in vertices) / len(vertices)
                       for axis in range(3))
        for first_index, second_index, third_index in triangles:
            first = vertices[first_index]
            normal = cross(_subtract(vertices[second_index], first),
                           _subtract(vertices[third_index], first))
            if dot(normal, _subtract(first, center)) < 0.0:
                normal = scale(normal, -1.0)
            if dot(normal, _subtract(point, first)) > 1.0e-10:
                return False
        return True

    def smoke_particles(value: Any, path: str) -> list[dict[str, Any]] | None:
        if not isinstance(value, list) or not value:
            failures.append(Failure(path, "expected emitted smoke particles", value))
            return None
        expected_keys = {"slot", "position_m", "velocity_m_s", "age_seconds",
                         "number_density", "pressure_m_s2", "vorticity_per_s"}
        parsed: list[dict[str, Any]] = []
        slots: set[int] = set()
        for index, particle in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(item_path, "expected an object", particle))
                continue
            if set(particle) != expected_keys:
                failures.append(Failure(item_path,
                                        "smoke snapshot shape differs from contract",
                                        sorted(particle), sorted(expected_keys)))
                continue
            slot = particle.get("slot")
            if (not isinstance(slot, int) or isinstance(slot, bool) or slot < 0):
                failures.append(Failure(item_path + ".slot",
                                        "expected a nonnegative integer", slot))
                continue
            if slot in slots:
                failures.append(Failure(item_path + ".slot",
                                        "duplicate smoke slot", slot))
                continue
            slots.add(slot)
            position = _vector(particle.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(particle.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            vorticity = _vector(particle.get("vorticity_per_s"),
                                item_path + ".vorticity_per_s", failures)
            age = _scalar(particle.get("age_seconds"),
                          item_path + ".age_seconds", failures)
            density = _scalar(particle.get("number_density"),
                              item_path + ".number_density", failures)
            pressure = _scalar(particle.get("pressure_m_s2"),
                               item_path + ".pressure_m_s2", failures)
            if (position is None or velocity is None or vorticity is None or
                    age is None or density is None or pressure is None):
                continue
            parsed.append({"slot": slot, "position": position,
                           "velocity": velocity, "vorticity": vorticity,
                           "age": age, "density": density,
                           "pressure": pressure})
        if len(parsed) != len(value):
            return None
        return sorted(parsed, key=lambda item: item["slot"])

    def geometry(definition: dict[str, Any], base: str
                 ) -> tuple[list[tuple[float, float, float]],
                            list[tuple[int, int, int]]] | None:
        vertices_raw = definition.get("rigid_geometry_vertices_m")
        indices = definition.get("rigid_geometry_triangle_indices")
        if not isinstance(vertices_raw, list) or not isinstance(indices, list):
            failures.append(Failure(base + ".rigid_geometry_vertices_m",
                                    "invalid rigid geometry", vertices_raw))
            return None
        parsed = [_vector(item, f"{base}.rigid_geometry_vertices_m[{index}]",
                          failures)
                  for index, item in enumerate(vertices_raw)]
        if any(vertex is None for vertex in parsed):
            return None
        vertices = [vertex for vertex in parsed if vertex is not None]
        if (not vertices or not indices or len(indices) % 3 != 0 or
                any(not isinstance(index, int) or isinstance(index, bool) or
                    index < 0 or index >= len(vertices) for index in indices)):
            failures.append(Failure(base + ".rigid_geometry_triangle_indices",
                                    "invalid rigid triangle topology", indices))
            return None
        triangles = [(indices[offset], indices[offset + 1], indices[offset + 2])
                     for offset in range(0, len(indices), 3)]
        return vertices, triangles

    def snapshot(value: Any, path: str,
                 mesh: tuple[list[tuple[float, float, float]],
                             list[tuple[int, int, int]]]
                 ) -> dict[str, Any] | None:
        if not isinstance(value, dict) or set(value) != {"rigid", "smoke_particles"}:
            failures.append(Failure(path,
                                    "snapshot shape differs from contract",
                                    value, ["rigid", "smoke_particles"]))
            return None
        raw_rigid = value.get("rigid")
        expected_rigid = {"position_m", "orientation_xyzw",
                          "linear_velocity_m_s", "angular_velocity_rad_s"}
        if not isinstance(raw_rigid, dict) or set(raw_rigid) != expected_rigid:
            failures.append(Failure(path + ".rigid",
                                    "rigid snapshot shape differs from contract",
                                    raw_rigid, sorted(expected_rigid)))
            return None
        position = _vector(raw_rigid.get("position_m"),
                           path + ".rigid.position_m", failures)
        orientation = _quaternion(raw_rigid.get("orientation_xyzw"),
                                  path + ".rigid.orientation_xyzw", failures)
        linear = _vector(raw_rigid.get("linear_velocity_m_s"),
                         path + ".rigid.linear_velocity_m_s", failures)
        angular = _vector(raw_rigid.get("angular_velocity_rad_s"),
                          path + ".rigid.angular_velocity_rad_s", failures)
        particles = smoke_particles(value.get("smoke_particles"),
                                    path + ".smoke_particles")
        if (position is None or orientation is None or linear is None or
                angular is None or particles is None):
            return None
        orientation = normalize_quaternion(
            orientation, path + ".rigid.orientation_xyzw")
        vertices, triangles = mesh
        transformed = [add(position, rotate(vertex, orientation))
                       for vertex in vertices]
        clearance = float("inf")
        for particle in particles:
            point = particle["position"]
            if point_inside_convex(point, transformed, triangles):
                clearance = 0.0
                break
            clearance = min(clearance, *(point_triangle_distance(
                point, transformed[first], transformed[second], transformed[third])
                for first, second, third in triangles))
        mean_smoke_velocity = scale(
            _sum([particle["velocity"] for particle in particles]),
            1.0 / len(particles))
        return {"position": position, "orientation": orientation,
                "linear": linear, "angular": angular,
                "smoke": particles, "mean_smoke_velocity": mean_smoke_velocity,
                "clearance": clearance}

    def smoke_delta(first: dict[str, Any], second: dict[str, Any], path: str,
                    check_all_fields: bool) -> tuple[float,
                                                     tuple[float, float, float]]:
        left = first["smoke"]
        right = second["smoke"]
        if len(left) != len(right):
            failures.append(Failure(path, "smoke particle count changed",
                                    len(right), len(left)))
            return float("inf"), (float("inf"),) * 3
        maximum = 0.0
        velocity_changes: list[tuple[float, float, float]] = []
        for before, after in zip(left, right):
            if before["slot"] != after["slot"]:
                failures.append(Failure(path, "smoke slot identity changed",
                                        after["slot"], before["slot"]))
                return float("inf"), (float("inf"),) * 3
            velocity_changes.append(_subtract(after["velocity"],
                                               before["velocity"]))
            if check_all_fields:
                maximum = max(maximum,
                              _norm(_subtract(after["position"], before["position"])),
                              _norm(_subtract(after["velocity"], before["velocity"])),
                              _norm(_subtract(after["vorticity"], before["vorticity"])),
                              abs(after["age"] - before["age"]),
                              abs(after["density"] - before["density"]),
                              abs(after["pressure"] - before["pressure"]))
        return maximum, _sum(velocity_changes)

    observations: dict[str, Any] = {}
    sample_count = 0
    expected_runs = {"forward-control", "forward", "stationary-control",
                     "stationary", "reverse-control", "reverse"}
    pairs = {"forward": "forward-control",
             "stationary": "stationary-control",
             "reverse": "reverse-control"}
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        settings = {key: value for key, value in actual.items() if key != "runs"}
        close_contract(settings, definition, base)
        mesh = geometry(definition, base)
        if mesh is None:
            continue
        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array", runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(base + ".runs",
                                    "expected exactly the six paired drag runs",
                                    sorted(indexed), sorted(expected_runs)))
            continue
        if any(set(run) != {"id", "initial", "final"}
               for run in indexed.values()):
            failures.append(Failure(base + ".runs",
                                    "run shape differs from contract"))
            continue
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial", mesh)
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final", mesh)
            if initial is not None and final is not None:
                states[run_id] = initial, final
        if set(states) != expected_runs:
            continue

        initial_position = _vector(
            definition.get("rigid_initial_position_m"),
            base + ".rigid_initial_position_m", failures)
        initial_orientation = _quaternion(
            definition.get("rigid_initial_orientation_xyzw"),
            base + ".rigid_initial_orientation_xyzw", failures)
        if initial_position is None or initial_orientation is None:
            continue
        initial_orientation = normalize_quaternion(
            initial_orientation, base + ".rigid_initial_orientation_xyzw")
        for run_id, state in states.items():
            rigid_initial = state[0]
            position_error = _norm(_subtract(
                rigid_initial["position"], initial_position))
            orientation_error = quaternion_distance(
                rigid_initial["orientation"], initial_orientation)
            rest_error = max(_norm(rigid_initial["linear"]),
                             _norm(rigid_initial["angular"]))
            if max(position_error, orientation_error, rest_error) > \
                    state_tolerance:
                failures.append(Failure(
                    base + f".runs[{run_id}].initial.rigid",
                    "initial rigid state differs from profile",
                    {"position_error_m": position_error,
                     "orientation_error": orientation_error,
                     "rest_velocity_error": rest_error}, state_tolerance))

        maximum_initial_state_delta = 0.0
        maximum_initial_smoke_delta = 0.0
        for run_id, control_id in pairs.items():
            control = states[control_id][0]
            test = states[run_id][0]
            rigid_delta = max(
                _norm(_subtract(test["position"], control["position"])),
                quaternion_distance(test["orientation"], control["orientation"]),
                _norm(_subtract(test["linear"], control["linear"])),
                _norm(_subtract(test["angular"], control["angular"])))
            maximum_initial_state_delta = max(maximum_initial_state_delta,
                                              rigid_delta)
            if rigid_delta > state_tolerance:
                failures.append(Failure(
                    base + f".runs[{run_id}].initial.rigid",
                    "paired initial rigid state differs from control",
                    rigid_delta, state_tolerance))
            smoke_state_delta, _ = smoke_delta(
                control, test, base + f".runs[{run_id}].initial.smoke_particles",
                True)
            maximum_initial_smoke_delta = max(maximum_initial_smoke_delta,
                                              smoke_state_delta)
            if smoke_state_delta > state_tolerance:
                failures.append(Failure(
                    base + f".runs[{run_id}].initial.smoke_particles",
                    "paired initial smoke state differs from control",
                    smoke_state_delta, state_tolerance))

        control_drift = max(
            _norm(_subtract(states[run_id][1]["linear"],
                            states[run_id][0]["linear"]))
            for run_id in pairs.values())
        control_smoke_drift = max(
            _norm(_subtract(states[run_id][1]["mean_smoke_velocity"],
                            states[run_id][0]["mean_smoke_velocity"]))
            for run_id in pairs.values())
        representation = definition.get("representation")
        control_limit = float(tolerances.get(
            "maximum_control_velocity_drift_m_s", 0.0))
        if control_drift > control_limit:
            failures.append(Failure(base + ".runs",
                                    "zero-drag rigid velocity drifted",
                                    control_drift, control_limit))
        if (representation == "SMOKE_PARTICLE" and
                control_smoke_drift > control_limit):
            failures.append(Failure(base + ".runs",
                                    "zero-drag smoke velocity drifted",
                                    control_smoke_drift, control_limit))

        rigid_responses: dict[str, float] = {}
        smoke_responses: dict[str, float] = {}
        maximum_transverse = 0.0
        rigid_impulses: dict[str, tuple[float, float, float]] = {}
        smoke_impulses: dict[str, tuple[float, float, float]] = {}
        reciprocal_residuals: dict[str, float] = {}
        force_residuals: dict[str, float] = {}
        pressure_delta = 0.0
        energy_increase = -float("inf")
        rigid_mass = float(definition.get("rigid_mass_kg", 0.0))
        particle_radius = float(definition.get("smoke_particle_radius_m", 0.0))
        air_density = float(definition.get(
            "test_air_density_kg_m3", definition.get("test_air_density", 0.0)))
        derived_particle_mass = air_density * (2.0 * particle_radius) ** 3
        particle_mass = float(definition.get(
            "effective_smoke_particle_mass_kg", derived_particle_mass))
        if abs(particle_mass - derived_particle_mass) > state_tolerance:
            failures.append(Failure(
                base + ".effective_smoke_particle_mass_kg",
                "effective smoke particle mass differs from public inputs",
                particle_mass, derived_particle_mass))
        timestep = float(definition.get("timestep_seconds", 0.0))
        drag_steps = definition.get("drag_step_count", 1)
        if (not isinstance(drag_steps, int) or isinstance(drag_steps, bool) or
                drag_steps <= 0):
            failures.append(Failure(base + ".drag_step_count",
                                    "expected a positive integer", drag_steps))
            drag_steps = 1
        drag_interval = timestep * drag_steps
        inertia_raw = definition.get("rigid_inertia_diagonal_kg_m2")
        inertia = (_vector(inertia_raw, base + ".rigid_inertia_diagonal_kg_m2",
                           failures) if inertia_raw is not None else None)
        def kinetic(state: dict[str, Any]) -> float:
            value = 0.5 * rigid_mass * dot(state["linear"], state["linear"])
            if inertia is not None:
                orientation = state["orientation"]
                inverse = (-orientation[0], -orientation[1], -orientation[2],
                           orientation[3])
                local_angular = rotate(state["angular"], inverse)
                value += 0.5 * sum(inertia[axis] * local_angular[axis] ** 2
                                   for axis in range(3))
            value += 0.5 * particle_mass * sum(
                dot(particle["velocity"], particle["velocity"])
                for particle in state["smoke"])
            return value

        for run_id, control_id in pairs.items():
            test_final = states[run_id][1]
            control_final = states[control_id][1]
            rigid_delta = _subtract(test_final["linear"],
                                    control_final["linear"])
            smoke_mean_delta = _subtract(test_final["mean_smoke_velocity"],
                                         control_final["mean_smoke_velocity"])
            carrier = _vector(
                definition.get(f"{run_id}_carrier_velocity_m_s"),
                base + f".{run_id}_carrier_velocity_m_s", failures)
            if carrier is None:
                continue
            denominator = dot(carrier, carrier)
            if denominator <= 1.0e-30:
                rigid_response = _norm(rigid_delta)
                smoke_response = _norm(smoke_mean_delta)
                stationary_limit = float(tolerances.get(
                    "maximum_stationary_velocity_m_s", 0.0))
                if max(rigid_response, smoke_response) > stationary_limit:
                    failures.append(Failure(
                        base + f".runs[{run_id}]",
                        "stationary smoke produced rigid-smoke drag",
                        max(rigid_response, smoke_response), stationary_limit))
            else:
                rigid_response = dot(rigid_delta, carrier) / denominator
                smoke_projection = dot(smoke_mean_delta, carrier) / denominator
                smoke_response = (-smoke_projection
                                  if representation == "SMOKE_PARTICLE"
                                  else abs(smoke_projection))
                rigid_transverse = _norm(_subtract(
                    rigid_delta, scale(carrier, rigid_response)))
                smoke_transverse = _norm(_subtract(
                    smoke_mean_delta, scale(carrier, smoke_projection)))
                maximum_transverse = max(maximum_transverse, rigid_transverse,
                                         smoke_transverse)
                minimum_rigid = float(tolerances.get(
                    "minimum_rigid_drag_response_fraction", 0.0))
                maximum_rigid = float(tolerances.get(
                    "maximum_rigid_drag_response_fraction", float("inf")))
                minimum_smoke = float(tolerances.get(
                    "minimum_smoke_drag_response_fraction", 0.0))
                maximum_smoke = float(tolerances.get(
                    "maximum_smoke_drag_response_fraction", float("inf")))
                if rigid_response < minimum_rigid or rigid_response > maximum_rigid:
                    failures.append(Failure(
                        base + f".runs[{run_id}]",
                        "rigid drag response is outside bounds",
                        rigid_response, [minimum_rigid, maximum_rigid]))
                if smoke_response < minimum_smoke or smoke_response > maximum_smoke:
                    failures.append(Failure(
                        base + f".runs[{run_id}]",
                        "smoke drag response is outside bounds",
                        smoke_response, [minimum_smoke, maximum_smoke]))
                transverse_limit = float(tolerances.get(
                    "maximum_transverse_velocity_m_s", 0.0))
                if max(rigid_transverse, smoke_transverse) > transverse_limit:
                    failures.append(Failure(
                        base + f".runs[{run_id}]",
                        "drag response is not aligned with relative flow",
                        max(rigid_transverse, smoke_transverse), transverse_limit))
            rigid_responses[run_id] = rigid_response
            smoke_responses[run_id] = smoke_response

            rigid_impulse = scale(rigid_delta, rigid_mass)
            _, smoke_velocity_sum = smoke_delta(
                control_final, test_final,
                base + f".runs[{run_id}].final.smoke_particles", False)
            rigid_impulses[run_id] = rigid_impulse
            if representation == "SMOKE_PARTICLE":
                smoke_impulse = scale(smoke_velocity_sum, particle_mass)
                smoke_impulses[run_id] = smoke_impulse
                reciprocal = _norm(add(rigid_impulse, smoke_impulse))
                reciprocal_residuals[run_id] = reciprocal
                force_residuals[run_id] = (
                    reciprocal / max(drag_interval, 1.0e-30))
                reciprocal_limit = float(tolerances.get(
                    "maximum_reciprocal_momentum_residual_kg_m_s", 0.0))
                if reciprocal > reciprocal_limit:
                    failures.append(Failure(
                        base + f".runs[{run_id}]",
                        "smoke and rigid momentum changes are not reciprocal",
                        reciprocal, reciprocal_limit))
                force_limit = float(tolerances.get(
                    "maximum_reciprocal_force_residual_n", float("inf")))
                if force_residuals[run_id] > force_limit:
                    failures.append(Failure(
                        base + f".runs[{run_id}]",
                        "smoke and rigid derived forces are not reciprocal",
                        force_residuals[run_id], force_limit))

            for control_particle, test_particle in zip(
                    control_final["smoke"], test_final["smoke"]):
                pressure_delta = max(
                    pressure_delta,
                    abs(test_particle["pressure"] - control_particle["pressure"]))
            if representation == "SMOKE_PARTICLE":
                energy_increase = max(
                    energy_increase,
                    kinetic(test_final) - kinetic(control_final))

        symmetry_error = max(
            abs(rigid_responses.get("forward", 0.0) -
                rigid_responses.get("reverse", 0.0)),
            abs(smoke_responses.get("forward", 0.0) -
                smoke_responses.get("reverse", 0.0)))
        symmetry_limit = float(tolerances.get(
            "maximum_forward_reverse_response_difference", 0.0))
        if symmetry_error > symmetry_limit:
            failures.append(Failure(base + ".runs",
                                    "forward and reverse drag responses differ",
                                    symmetry_error, symmetry_limit))

        minimum_clearance = min(
            snapshot_state["clearance"] for state in states.values()
            for snapshot_state in state)
        contact_distance = float(definition.get("contact_distance_m", 0.0))
        required_clearance = max(
            contact_distance,
            float(tolerances.get("minimum_non_contact_clearance_m", 0.0)))
        if minimum_clearance <= required_clearance + state_tolerance:
            failures.append(Failure(
                base + ".runs",
                "smoke tracer entered rigid contact clearance",
                minimum_clearance, f"> {required_clearance + state_tolerance}"))
        maximum_rigid_speed = max(
            _norm(snapshot_state["linear"]) for state in states.values()
            for snapshot_state in state)
        maximum_smoke_speed = max(
            _norm(particle["velocity"]) for state in states.values()
            for snapshot_state in state for particle in snapshot_state["smoke"])
        maximum_angular_speed = max(
            _norm(snapshot_state["angular"]) for state in states.values()
            for snapshot_state in state)
        speed_slack = float(tolerances.get("maximum_speed_slack_m_s", 0.0))
        rigid_speed_limit = float(definition.get(
            "rigid_maximum_linear_speed_m_s", 0.0)) + speed_slack
        smoke_speed_limit = float(definition.get(
            "smoke_maximum_speed_m_s", 0.0)) + speed_slack
        angular_limit = min(
            float(definition.get("rigid_maximum_angular_speed_rad_s",
                                 float("inf"))),
            float(tolerances.get("maximum_angular_speed_rad_s",
                                 float("inf"))))
        if maximum_rigid_speed > rigid_speed_limit:
            failures.append(Failure(base + ".runs",
                                    "rigid speed exceeds authored maximum",
                                    maximum_rigid_speed, rigid_speed_limit))
        if maximum_smoke_speed > smoke_speed_limit:
            failures.append(Failure(base + ".runs",
                                    "smoke speed exceeds authored maximum",
                                    maximum_smoke_speed, smoke_speed_limit))
        if maximum_angular_speed > angular_limit:
            failures.append(Failure(base + ".runs",
                                    "rigid angular speed exceeds bound",
                                    maximum_angular_speed, angular_limit))
        energy_limit = float(tolerances.get("maximum_energy_increase_j", 0.0))
        if (representation == "SMOKE_PARTICLE" and
                energy_increase > energy_limit):
            failures.append(Failure(base + ".runs",
                                    "passive drag increased kinetic energy",
                                    energy_increase, energy_limit))

        observations[identity] = {
            "maximum_paired_initial_rigid_state_delta":
                maximum_initial_state_delta,
            "maximum_paired_initial_smoke_state_delta":
                maximum_initial_smoke_delta,
            "maximum_zero_drag_rigid_velocity_drift_m_s": control_drift,
            "maximum_zero_drag_smoke_velocity_drift_m_s": control_smoke_drift,
            "rigid_response_fractions": rigid_responses,
            "smoke_response_fractions": smoke_responses,
            "maximum_transverse_velocity_m_s": maximum_transverse,
            "forward_reverse_response_difference": symmetry_error,
            "rigid_drag_impulses_kg_m_s": {
                key: list(value) for key, value in rigid_impulses.items()},
            "smoke_drag_impulses_kg_m_s": {
                key: list(value) for key, value in smoke_impulses.items()},
            "reciprocal_momentum_residuals_kg_m_s": reciprocal_residuals,
            "reciprocal_force_residuals_n": force_residuals,
            "minimum_tracer_rigid_clearance_m": minimum_clearance,
            "maximum_pressure_delta_m_s2": pressure_delta,
            "maximum_kinetic_energy_increase_j":
                energy_increase if representation == "SMOKE_PARTICLE" else None,
            "maximum_rigid_speed_m_s": maximum_rigid_speed,
            "maximum_smoke_speed_m_s": maximum_smoke_speed,
            "maximum_rigid_angular_speed_rad_s": maximum_angular_speed,
        }
        sample_count += 3

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_externally_heated_fluid(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    setting_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    temperature_tolerance = float(tolerances.get(
        "temperature_absolute_celsius", 0.0))
    energy_tolerance = float(tolerances.get(
        "sensible_energy_absolute_j", 0.0))
    fluid_tolerance = float(tolerances.get("fluid_state_absolute", 0.0))
    smoke_tolerance = float(tolerances.get("smoke_state_absolute", 0.0))

    def sub(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(first, second))

    def close_number(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        reference = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                abs(value - reference) > setting_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(actual, path, failures)
        reference = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                _norm(sub(value, reference)) > setting_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def close_sequence(actual: Any, expected: Any, path: str) -> None:
        if (not isinstance(actual, list) or not isinstance(expected, list) or
                len(actual) != len(expected)):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))
            return
        values = [_scalar(value, f"{path}[{index}]", failures)
                  for index, value in enumerate(actual)]
        references = [_scalar(value, f"{path}.profile[{index}]", failures)
                      for index, value in enumerate(expected)]
        if (all(value is not None for value in values + references) and
                any(abs(value - reference) > setting_tolerance
                    for value, reference in zip(values, references))):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def snapshot(value: Any, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        fluid = value.get("fluid")
        particles = value.get("smoke_particles")
        if not isinstance(fluid, dict):
            failures.append(Failure(path + ".fluid",
                                    "expected a fluid object", fluid))
            return None
        if not isinstance(particles, list) or not particles:
            failures.append(Failure(path + ".smoke_particles",
                                    "expected a retained smoke endpoint",
                                    particles))
            return None
        vectors = [_vector(fluid.get(field), path + ".fluid." + field,
                           failures)
                   for field in ("position_m", "velocity_m_s",
                                 "acceleration_m_s2")]
        temperature = _scalar(fluid.get("temperature_celsius"),
                              path + ".fluid.temperature_celsius", failures)
        stable_id = fluid.get("stable_id")
        if not isinstance(stable_id, int) or isinstance(stable_id, bool):
            failures.append(Failure(path + ".fluid.stable_id",
                                    "stable id must be an integer", stable_id))
        normalized_smoke: list[dict[str, Any]] = []
        for index, particle in enumerate(particles):
            particle_path = f"{path}.smoke_particles[{index}]"
            if not isinstance(particle, dict):
                failures.append(Failure(particle_path, "expected an object",
                                        particle))
                continue
            particle_vectors = [_vector(
                particle.get(field), particle_path + "." + field, failures)
                for field in ("position_m", "velocity_m_s", "vorticity_per_s")]
            particle_scalars = [_scalar(
                particle.get(field), particle_path + "." + field, failures)
                for field in ("age_seconds", "number_density", "pressure_m_s2")]
            slot = particle.get("slot")
            if (any(item is None for item in particle_vectors + particle_scalars) or
                    not isinstance(slot, int) or isinstance(slot, bool)):
                if not isinstance(slot, int) or isinstance(slot, bool):
                    failures.append(Failure(particle_path + ".slot",
                                            "slot must be an integer", slot))
                continue
            normalized_smoke.append({
                "slot": slot,
                "vectors": particle_vectors,
                "scalars": particle_scalars,
            })
        if (any(item is None for item in vectors) or temperature is None or
                not isinstance(stable_id, int) or isinstance(stable_id, bool) or
                len(normalized_smoke) != len(particles)):
            return None
        return {"position": vectors[0], "velocity": vectors[1],
                "acceleration": vectors[2], "temperature": temperature,
                "stable_id": stable_id, "smoke": normalized_smoke}

    def compare_smoke(first: dict[str, Any], second: dict[str, Any],
                      path: str) -> float:
        left = first["smoke"]
        right = second["smoke"]
        if len(left) != len(right):
            failures.append(Failure(path, "heating changed smoke endpoint count",
                                    len(right), len(left)))
            return float("inf")
        maximum = 0.0
        for left_particle, right_particle in zip(left, right):
            if left_particle["slot"] != right_particle["slot"]:
                failures.append(Failure(path, "heating changed smoke identity",
                                        right_particle["slot"],
                                        left_particle["slot"]))
            for left_vector, right_vector in zip(
                    left_particle["vectors"], right_particle["vectors"]):
                maximum = max(maximum, _norm(sub(left_vector, right_vector)))
            for left_scalar, right_scalar in zip(
                    left_particle["scalars"], right_particle["scalars"]):
                maximum = max(maximum, abs(left_scalar - right_scalar))
        if maximum > smoke_tolerance:
            failures.append(Failure(path,
                                    "external heating mutated the smoke endpoint",
                                    maximum, smoke_tolerance))
        return maximum

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "representation", "warmup_step_count", "substeps",
                      "smoke_particle_capacity", "smoke_grid_resolution",
                      "smoke_grid_vertical_resolution", "declared_mode_flags",
                      "exercised_mode", "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in ("timestep_seconds", "fluid_initial_temperature_celsius",
                      "equilibrium_temperature_celsius",
                      "fluid_particle_radius_m", "fluid_support_radius_m",
                      "fluid_rest_density_kg_m3",
                      "fluid_rest_particle_volume_m3",
                      "oracle_specific_heat_j_kg_k",
                      "smoke_emission_rate_per_s", "smoke_lifetime_seconds",
                      "smoke_particle_radius_m", "smoke_rest_number_density",
                      "smoke_maximum_speed_m_s", "smoke_grid_edge_length_m",
                      "heater_temperature_celsius",
                      "control_heat_transfer_rate_per_s",
                      "test_heat_transfer_rate_per_s", "wind_drag_per_s",
                      "boiling_temperature_celsius"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("gravity_m_s2", "fluid_initial_position_m",
                      "outside_fluid_position_m", "smoke_emitter_center_m",
                      "smoke_grid_minimum_m", "heater_center_m"):
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("smoke_emitter_half_extents_m", "heater_half_extents_m"):
            close_sequence(actual.get(field), definition.get(field),
                           base + "." + field)
        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array", runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        expected_runs = {"control", "heated", "equilibrium", "outside"}
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(base + ".runs",
                                    "expected exactly the four thermal runs",
                                    sorted(indexed), sorted(expected_runs)))
            continue
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial")
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final")
            if initial is not None and final is not None:
                states[run_id] = (initial, final)
        if set(states) != expected_runs:
            continue

        for run_id, (initial, final) in states.items():
            if initial["stable_id"] != final["stable_id"]:
                failures.append(Failure(base + f".runs[{run_id}]",
                                        "heating changed fluid identity",
                                        final["stable_id"], initial["stable_id"]))
            position_drift = _norm(sub(final["position"], initial["position"]))
            velocity_drift = _norm(sub(final["velocity"], initial["velocity"]))
            if max(position_drift, velocity_drift) > fluid_tolerance:
                failures.append(Failure(base + f".runs[{run_id}]",
                                        "thermal-only run changed fluid mechanics",
                                        max(position_drift, velocity_drift),
                                        fluid_tolerance))

        control_initial, control_final = states["control"]
        heated_initial, heated_final = states["heated"]
        equilibrium_initial, equilibrium_final = states["equilibrium"]
        outside_initial, outside_final = states["outside"]
        initial_temperature = float(definition["fluid_initial_temperature_celsius"])
        target_temperature = float(definition["heater_temperature_celsius"])
        rate = float(definition["test_heat_transfer_rate_per_s"])
        dt = float(definition["timestep_seconds"])
        expected_heated = initial_temperature + (
            target_temperature - initial_temperature) * (1.0 - math.exp(-rate * dt))
        checks = (
            ("control", control_initial["temperature"],
             control_final["temperature"], initial_temperature),
            ("heated", heated_initial["temperature"],
             heated_final["temperature"], expected_heated),
            ("equilibrium", equilibrium_initial["temperature"],
             equilibrium_final["temperature"], target_temperature),
            ("outside", outside_initial["temperature"],
             outside_final["temperature"], initial_temperature),
        )
        for run_id, observed_initial, observed_final, expected_final in checks:
            expected_initial = (target_temperature if run_id == "equilibrium"
                                else initial_temperature)
            if abs(observed_initial - expected_initial) > temperature_tolerance:
                failures.append(Failure(base + f".runs[{run_id}].initial",
                                        "initial temperature differs from profile",
                                        observed_initial, expected_initial))
            if abs(observed_final - expected_final) > temperature_tolerance:
                failures.append(Failure(base + f".runs[{run_id}].final",
                                        "temperature differs from heater oracle",
                                        observed_final, expected_final))
        minimum_heating = float(tolerances.get("minimum_heating_celsius", 0.0))
        observed_heating = heated_final["temperature"] - heated_initial["temperature"]
        if observed_heating < minimum_heating:
            failures.append(Failure(base + ".runs[heated]",
                                    "external heater response is too small",
                                    observed_heating, minimum_heating))
        if not (heated_initial["temperature"] <= heated_final["temperature"] <=
                target_temperature + temperature_tolerance):
            failures.append(Failure(base + ".runs[heated]",
                                    "external heater overshot its temperature",
                                    heated_final["temperature"], target_temperature))

        mass = (float(definition["fluid_rest_density_kg_m3"]) *
                float(definition["fluid_rest_particle_volume_m3"]))
        specific_heat = float(definition["oracle_specific_heat_j_kg_k"])
        actual_energy = mass * specific_heat * observed_heating
        expected_energy = mass * specific_heat * (
            expected_heated - initial_temperature)
        if abs(actual_energy - expected_energy) > energy_tolerance:
            failures.append(Failure(base + ".runs[heated]",
                                    "normalized sensible-energy accounting failed",
                                    actual_energy, expected_energy))
        smoke_delta = max(
            compare_smoke(control_final, heated_final,
                          base + ".runs[heated].final.smoke_particles"),
            compare_smoke(control_final, equilibrium_final,
                          base + ".runs[equilibrium].final.smoke_particles"),
            compare_smoke(control_final, outside_final,
                          base + ".runs[outside].final.smoke_particles"))
        observations[identity] = {
            "expected_heated_temperature_celsius": expected_heated,
            "observed_heated_temperature_celsius": heated_final["temperature"],
            "normalized_sensible_energy_change_j": actual_energy,
            "maximum_smoke_endpoint_delta": smoke_delta,
        }
        sample_count += 4

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_cloth_smoke_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    scalar_tolerance = float(tolerances.get(
        "smoke_scalar_state_absolute", 0.0))

    def add(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(first, second))

    def sub(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(first, second))

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)

    def dot(first: tuple[float, float, float],
            second: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(first, second))

    def cross(first: tuple[float, float, float],
              second: tuple[float, float, float]) -> tuple[float, float, float]:
        return (first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0])

    def close_number(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        reference = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                abs(value - reference) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(actual, path, failures)
        reference = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                _norm(sub(value, reference)) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def close_pair(actual: Any, expected: Any, path: str) -> None:
        if (not isinstance(actual, list) or len(actual) != 2 or
                not isinstance(expected, list) or len(expected) != 2):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))
            return
        values = [_scalar(value, f"{path}[{index}]", failures)
                  for index, value in enumerate(actual)]
        references = [_scalar(value, f"$.profile{path[1:]}[{index}]", failures)
                      for index, value in enumerate(expected)]
        if (all(value is not None for value in values + references) and
                any(abs(value - reference) > state_tolerance
                    for value, reference in zip(values, references))):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def snapshot(value: Any, path: str,
                 vertex_count: int) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        raw_smoke = value.get("smoke_particles")
        if not isinstance(raw_smoke, list) or len(raw_smoke) != 1:
            failures.append(Failure(path + ".smoke_particles",
                                    "expected exactly one smoke particle",
                                    raw_smoke, 1))
            return None
        raw_particle = raw_smoke[0]
        if not isinstance(raw_particle, dict):
            failures.append(Failure(path + ".smoke_particles[0]",
                                    "expected a smoke particle object",
                                    raw_particle))
            return None
        particle_path = path + ".smoke_particles[0]"
        slot = raw_particle.get("slot")
        if not isinstance(slot, int) or isinstance(slot, bool) or slot != 0:
            failures.append(Failure(particle_path + ".slot",
                                    "smoke slot/order changed", slot, 0))
        position = _vector(raw_particle.get("position_m"),
                           particle_path + ".position_m", failures)
        velocity = _vector(raw_particle.get("velocity_m_s"),
                           particle_path + ".velocity_m_s", failures)
        vorticity = _vector(raw_particle.get("vorticity_per_s"),
                            particle_path + ".vorticity_per_s", failures)
        scalars: dict[str, float] = {}
        for field in ("age_seconds", "number_density", "pressure_m_s2"):
            parsed = _scalar(raw_particle.get(field),
                             particle_path + "." + field, failures)
            if parsed is not None:
                scalars[field] = parsed
        for field in ("age_seconds", "number_density"):
            if field in scalars and scalars[field] < 0.0:
                failures.append(Failure(particle_path + "." + field,
                                        "smoke scalar must be nonnegative",
                                        scalars[field], 0.0))

        raw_vertices = value.get("cloth_vertices_m")
        raw_velocities = value.get("cloth_velocities_m_s")
        if not isinstance(raw_vertices, list) or len(raw_vertices) != vertex_count:
            failures.append(Failure(path + ".cloth_vertices_m",
                                    "unexpected cloth vertex count",
                                    raw_vertices, vertex_count))
            return None
        if (not isinstance(raw_velocities, list) or
                len(raw_velocities) != vertex_count):
            failures.append(Failure(path + ".cloth_velocities_m_s",
                                    "unexpected cloth velocity count",
                                    raw_velocities, vertex_count))
            return None
        vertices = [_vector(item, f"{path}.cloth_vertices_m[{index}]", failures)
                    for index, item in enumerate(raw_vertices)]
        velocities = [_vector(item,
                              f"{path}.cloth_velocities_m_s[{index}]", failures)
                      for index, item in enumerate(raw_velocities)]
        if (position is None or velocity is None or vorticity is None or
                len(scalars) != 3 or any(item is None for item in vertices) or
                any(item is None for item in velocities)):
            return None
        return {
            "particle_raw": raw_particle,
            "snapshot_raw": value,
            "position": position,
            "velocity": velocity,
            "vorticity": vorticity,
            "scalars": scalars,
            "slot": slot,
            "cloth_vertices": [item for item in vertices if item is not None],
            "cloth_velocities": [item for item in velocities if item is not None],
        }

    def closest_on_triangle(
            point: tuple[float, float, float],
            first: tuple[float, float, float],
            second: tuple[float, float, float],
            third: tuple[float, float, float]) -> tuple[
                tuple[float, float, float], tuple[float, float, float]]:
        # Closest-point regions from Real-Time Collision Detection.
        first_second = sub(second, first)
        first_third = sub(third, first)
        first_point = sub(point, first)
        d1 = dot(first_second, first_point)
        d2 = dot(first_third, first_point)
        if d1 <= 0.0 and d2 <= 0.0:
            return first, (1.0, 0.0, 0.0)
        second_point = sub(point, second)
        d3 = dot(first_second, second_point)
        d4 = dot(first_third, second_point)
        if d3 >= 0.0 and d4 <= d3:
            return second, (0.0, 1.0, 0.0)
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            weight = d1 / (d1 - d3)
            return add(first, scale(first_second, weight)), (
                1.0 - weight, weight, 0.0)
        third_point = sub(point, third)
        d5 = dot(first_second, third_point)
        d6 = dot(first_third, third_point)
        if d6 >= 0.0 and d5 <= d6:
            return third, (0.0, 0.0, 1.0)
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            weight = d2 / (d2 - d6)
            return add(first, scale(first_third, weight)), (
                1.0 - weight, 0.0, weight)
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            edge = sub(third, second)
            weight = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return add(second, scale(edge, weight)), (
                0.0, 1.0 - weight, weight)
        denominator = 1.0 / (va + vb + vc)
        second_weight = vb * denominator
        third_weight = vc * denominator
        return add(first, add(scale(first_second, second_weight),
                              scale(first_third, third_weight))), (
            1.0 - second_weight - third_weight,
            second_weight, third_weight)

    def surface_sample(state: dict[str, Any], indices: Any,
                       path: str) -> dict[str, Any] | None:
        vertices = state["cloth_vertices"]
        velocities = state["cloth_velocities"]
        if (not isinstance(indices, list) or not indices or len(indices) % 3 or
                any(not isinstance(index, int) or isinstance(index, bool) or
                    index < 0 or index >= len(vertices) for index in indices)):
            failures.append(Failure(path + ".cloth_triangle_indices",
                                    "invalid public cloth triangles", indices))
            return None
        best: dict[str, Any] | None = None
        for offset in range(0, len(indices), 3):
            triangle = indices[offset:offset + 3]
            first, second, third = (vertices[index] for index in triangle)
            normal = _unit(cross(sub(second, first), sub(third, first)))
            if normal is None:
                failures.append(Failure(path + ".cloth_vertices_m",
                                        "cloth triangle is degenerate", triangle))
                continue
            closest, weights = closest_on_triangle(
                state["position"], first, second, third)
            delta = sub(state["position"], closest)
            distance = _norm(delta)
            if best is None or distance < best["distance"]:
                surface_velocity = (0.0, 0.0, 0.0)
                for index, weight in zip(triangle, weights):
                    surface_velocity = add(
                        surface_velocity, scale(velocities[index], weight))
                best = {"distance": distance,
                        "signed_distance": dot(delta, normal),
                        "normal": normal,
                        "velocity": surface_velocity}
        if best is None:
            return None
        particle_raw = state["particle_raw"]
        reported_distance = particle_raw.get("signed_separation_m")
        if reported_distance is not None:
            value = _scalar(reported_distance,
                            path + ".smoke_particles[0].signed_separation_m",
                            failures)
            if (value is not None and
                    abs(value - best["signed_distance"]) > state_tolerance):
                failures.append(Failure(
                    path + ".smoke_particles[0].signed_separation_m",
                    "reported separation differs from public state",
                    value, best["signed_distance"]))
        reported_speed = particle_raw.get("speed_m_s")
        if reported_speed is not None:
            value = _scalar(reported_speed,
                            path + ".smoke_particles[0].speed_m_s", failures)
            speed = _norm(state["velocity"])
            if value is not None and abs(value - speed) > state_tolerance:
                failures.append(Failure(
                    path + ".smoke_particles[0].speed_m_s",
                    "reported speed differs from public state", value, speed))
        reported_normal = state["snapshot_raw"].get("cloth_normal")
        if reported_normal is not None:
            value = _vector(reported_normal, path + ".cloth_normal", failures)
            if (value is not None and
                    _norm(sub(value, best["normal"])) > state_tolerance):
                failures.append(Failure(
                    path + ".cloth_normal",
                    "reported normal differs from public state",
                    list(value), list(best["normal"])))
        return best

    def compare_smoke(first: dict[str, Any], second: dict[str, Any],
                      path: str) -> None:
        if first["slot"] != second["slot"]:
            failures.append(Failure(path + ".slot",
                                    "paired initial smoke slot differs",
                                    second["slot"], first["slot"]))
        for field in ("position", "velocity", "vorticity"):
            difference = _norm(sub(first[field], second[field]))
            if difference > state_tolerance:
                failures.append(Failure(path + "." + field,
                                        "paired initial smoke state differs",
                                        difference, state_tolerance))
        for field in ("age_seconds", "number_density", "pressure_m_s2"):
            difference = abs(first["scalars"][field] -
                             second["scalars"][field])
            if difference > scalar_tolerance:
                failures.append(Failure(path + "." + field,
                                        "paired initial smoke state differs",
                                        difference, scalar_tolerance))

    def compare_cloth(first: dict[str, Any], second: dict[str, Any],
                      path: str) -> None:
        position_limit = float(tolerances.get(
            "maximum_cloth_position_difference_m", 0.0))
        velocity_limit = float(tolerances.get(
            "maximum_cloth_velocity_difference_m_s", 0.0))
        for index, (left, right) in enumerate(zip(
                first["cloth_vertices"], second["cloth_vertices"])):
            difference = _norm(sub(left, right))
            if difference > position_limit:
                failures.append(Failure(
                    f"{path}.cloth_vertices_m[{index}]",
                    "cloth differs from no-contact control",
                    difference, position_limit))
        for index, (left, right) in enumerate(zip(
                first["cloth_velocities"], second["cloth_velocities"])):
            difference = _norm(sub(left, right))
            if difference > velocity_limit:
                failures.append(Failure(
                    f"{path}.cloth_velocities_m_s[{index}]",
                    "cloth received an unadvertised reverse mechanical response",
                    difference, velocity_limit))

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "representation", "warmup_step_count",
                      "step_count", "substeps", "declared_mode_flags",
                      "exercised_mode", "cloth_triangle_indices",
                      "cloth_dynamic", "smoke_particle_capacity",
                      "smoke_grid_resolution",
                      "smoke_grid_vertical_resolution", "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in ("timestep_seconds", "cloth_vertex_mass_kg",
                      "cloth_thickness_m", "smoke_emission_rate_per_s",
                      "smoke_lifetime_seconds", "smoke_particle_radius_m",
                      "smoke_rest_number_density",
                      "smoke_pressure_stiffness_m2_s2",
                      "smoke_viscosity_m2_s", "smoke_maximum_speed_m_s",
                      "smoke_grid_edge_length_m", "smoke_grid_spacing_m",
                      "contact_distance_m", "wind_drag_per_s",
                      "maximum_wind_acceleration_m_s2"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("gravity_m_s2", "smoke_emitter_center_m",
                      "smoke_initial_velocity_m_s", "smoke_wind_m_s",
                      "smoke_grid_minimum_m"):
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        close_pair(actual.get("smoke_emitter_half_extents_m"),
                   definition.get("smoke_emitter_half_extents_m"),
                   base + ".smoke_emitter_half_extents_m")
        actual_vertices = actual.get("cloth_vertices_m")
        expected_vertices = definition.get("cloth_vertices_m")
        if (not isinstance(actual_vertices, list) or
                not isinstance(expected_vertices, list) or
                len(actual_vertices) != len(expected_vertices)):
            failures.append(Failure(base + ".cloth_vertices_m",
                                    "fixture setting differs from profile",
                                    actual_vertices, expected_vertices))
            vertex_count = len(expected_vertices) if isinstance(
                expected_vertices, list) else 0
        else:
            vertex_count = len(expected_vertices)
            for index, (value, expected) in enumerate(zip(
                    actual_vertices, expected_vertices)):
                close_vector(value, expected,
                             f"{base}.cloth_vertices_m[{index}]")

        raw_runs = actual.get("runs")
        if not isinstance(raw_runs, list):
            failures.append(Failure(base + ".runs", "expected a run array",
                                    raw_runs))
            raw_runs = []
        runs: dict[str, dict[str, Any]] = {}
        for run_index, run in enumerate(raw_runs):
            run_path = f"{base}.runs[{run_index}]"
            if not isinstance(run, dict) or not isinstance(run.get("id"), str):
                failures.append(Failure(run_path,
                                        "run needs a string id", run))
                continue
            if run["id"] in runs:
                failures.append(Failure(run_path + ".id", "duplicate run id",
                                        run["id"]))
                continue
            runs[run["id"]] = run
        for missing in sorted({"control", "coupled"} - set(runs)):
            failures.append(Failure(base + ".runs", "missing required run",
                                    missing))
        for extra in sorted(set(runs) - {"control", "coupled"}):
            failures.append(Failure(base + ".runs", "unexpected run", extra))
        states: dict[str, dict[str, dict[str, Any]]] = {}
        for run_id in ("control", "coupled"):
            run = runs.get(run_id)
            if run is None:
                continue
            initial = snapshot(run.get("initial"),
                               f"{base}.runs[{run_id}].initial", vertex_count)
            final = snapshot(run.get("final"),
                             f"{base}.runs[{run_id}].final", vertex_count)
            if initial is not None and final is not None:
                states[run_id] = {"initial": initial, "final": final}
        if set(states) != {"control", "coupled"}:
            continue

        control = states["control"]
        coupled = states["coupled"]
        compare_smoke(control["initial"], coupled["initial"],
                      base + ".runs.coupled.initial.smoke_particles[0]")
        compare_cloth(control["initial"], coupled["initial"],
                      base + ".runs.coupled.initial")
        compare_cloth(control["final"], coupled["final"],
                      base + ".runs.coupled.final")
        compare_cloth(control["initial"], control["final"],
                      base + ".runs.control.final")
        compare_cloth(coupled["initial"], coupled["final"],
                      base + ".runs.coupled.final")

        canonical_vertices = definition.get("cloth_vertices_m", [])
        if isinstance(canonical_vertices, list):
            for label, state in (("control.initial", control["initial"]),
                                 ("coupled.initial", coupled["initial"])):
                for index, (value, expected) in enumerate(zip(
                        state["cloth_vertices"], canonical_vertices)):
                    reference = _vector(expected, "$.profile.cloth_vertices_m",
                                        failures)
                    if (reference is not None and
                            _norm(sub(value, reference)) > state_tolerance):
                        failures.append(Failure(
                            f"{base}.runs.{label}.cloth_vertices_m[{index}]",
                            "initial cloth state differs from canonical input",
                            list(value), list(reference)))
        expected_position = _vector(definition.get("smoke_emitter_center_m"),
                                    "$.profile.smoke_emitter_center_m", failures)
        expected_velocity = _vector(definition.get("smoke_initial_velocity_m_s"),
                                    "$.profile.smoke_initial_velocity_m_s",
                                    failures)
        half_extents = definition.get("smoke_emitter_half_extents_m")
        if (expected_position is not None and isinstance(half_extents, list) and
                len(half_extents) == 2 and all(_number(value)
                                               for value in half_extents)):
            actual_position = control["initial"]["position"]
            outside_emitter = (
                abs(actual_position[0] - expected_position[0]) > state_tolerance or
                abs(actual_position[1] - expected_position[1]) >
                float(half_extents[0]) + state_tolerance or
                abs(actual_position[2] - expected_position[2]) >
                float(half_extents[1]) + state_tolerance)
            if outside_emitter:
                failures.append(Failure(
                    base + ".runs.control.initial.smoke_particles[0].position_m",
                    "initial smoke state falls outside canonical emitter",
                    list(actual_position),
                    {"center_m": list(expected_position),
                     "half_extents_m": half_extents}))
        if (expected_velocity is not None and _norm(sub(
                control["initial"]["velocity"], expected_velocity)) >
                state_tolerance):
            failures.append(Failure(
                base + ".runs.control.initial.smoke_particles[0].velocity_m_s",
                "initial smoke state differs from canonical input",
                list(control["initial"]["velocity"]), list(expected_velocity)))

        indices = actual.get("cloth_triangle_indices")
        control_initial = surface_sample(
            control["initial"], indices, base + ".runs.control.initial")
        control_final = surface_sample(
            control["final"], indices, base + ".runs.control.final")
        coupled_initial = surface_sample(
            coupled["initial"], indices, base + ".runs.coupled.initial")
        coupled_final = surface_sample(
            coupled["final"], indices, base + ".runs.coupled.final")
        if any(item is None for item in (
                control_initial, control_final, coupled_initial, coupled_final)):
            continue
        assert control_initial is not None and control_final is not None
        assert coupled_initial is not None and coupled_final is not None
        initial_signed = control_initial["signed_distance"]
        final_control_signed = control_final["signed_distance"]
        minimum_crossing = float(tolerances.get(
            "minimum_control_crossing_m", 0.0))
        crossing = min(abs(initial_signed), abs(final_control_signed))
        if initial_signed * final_control_signed >= 0.0 or crossing < minimum_crossing:
            failures.append(Failure(
                base + ".runs.control.final.smoke_particles[0].position_m",
                "control smoke did not cross cloth surface",
                crossing, minimum_crossing))

        final_coupled_signed = coupled_final["signed_distance"]
        if coupled_initial["signed_distance"] * final_coupled_signed <= 0.0:
            failures.append(Failure(
                base + ".runs.coupled.final.smoke_particles[0].position_m",
                "contact failed to keep smoke on its incoming side"))
        contact_distance = float(definition.get("contact_distance_m", 0.0))
        clearance_error = abs(coupled_final["distance"] - contact_distance)
        clearance_limit = float(tolerances.get(
            "maximum_contact_clearance_error_m", 0.0))
        if clearance_error > clearance_limit:
            failures.append(Failure(
                base + ".runs.coupled.final.smoke_particles[0].position_m",
                "contact clearance differs from declared surface",
                clearance_error, clearance_limit))

        side = 1.0 if coupled_initial["signed_distance"] >= 0.0 else -1.0
        outward = scale(coupled_initial["normal"], side)
        final_relative = sub(coupled["final"]["velocity"],
                             coupled_final["velocity"])
        final_normal_velocity = dot(final_relative, outward)
        normal_limit = float(tolerances.get(
            "maximum_contact_normal_velocity_m_s", 0.0))
        if abs(final_normal_velocity) > normal_limit:
            failures.append(Failure(
                base + ".runs.coupled.final.smoke_particles[0].velocity_m_s",
                "contact left excessive normal velocity",
                abs(final_normal_velocity), normal_limit))
        final_tangent = sub(final_relative,
                            scale(outward, final_normal_velocity))
        tangent_speed = _norm(final_tangent)
        tangent_minimum = float(tolerances.get(
            "minimum_redirected_tangential_speed_m_s", 0.0))
        if tangent_speed < tangent_minimum:
            failures.append(Failure(
                base + ".runs.coupled.final.smoke_particles[0].velocity_m_s",
                "contact did not retain redirected tangential motion",
                tangent_speed, tangent_minimum))

        maximum_speed = float(definition.get("smoke_maximum_speed_m_s", 0.0))
        speed_error = float(tolerances.get(
            "maximum_smoke_speed_error_m_s", 0.0))
        observed_maximum_speed = 0.0
        for run_id, endpoints in states.items():
            for endpoint, state in endpoints.items():
                speed = _norm(state["velocity"])
                observed_maximum_speed = max(observed_maximum_speed, speed)
                if speed > maximum_speed + speed_error:
                    failures.append(Failure(
                        f"{base}.runs.{run_id}.{endpoint}.smoke_particles[0].velocity_m_s",
                        "smoke speed exceeds profile bound", speed,
                        maximum_speed + speed_error))

        observations[identity] = {
            "control_initial_signed_distance_m": initial_signed,
            "control_final_signed_distance_m": final_control_signed,
            "coupled_final_signed_distance_m": final_coupled_signed,
            "contact_clearance_error_m": clearance_error,
            "coupled_final_normal_velocity_m_s": final_normal_velocity,
            "coupled_final_tangential_speed_m_s": tangent_speed,
            "maximum_smoke_speed_m_s": observed_maximum_speed,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_rope_smoke_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    """Evaluate one-way smoke contact from public rope-node capsules."""
    definitions = {item["id"]: item for item in profile.get("fixtures", [])
                   if isinstance(item, dict) and
                   isinstance(item.get("id"), str)}
    raw_actual = result.get("fixtures")
    if not isinstance(raw_actual, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                raw_actual))
        raw_actual = []
    actuals: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_actual):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", item))
            continue
        identity = item["id"]
        if identity in actuals:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actuals[identity] = item
    for identity in sorted(set(definitions) - set(actuals)):
        failures.append(Failure("$.fixtures", "missing required fixture",
                                identity))
    for identity in sorted(set(actuals) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", identity))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    scalar_tolerance = float(tolerances.get(
        "smoke_scalar_state_absolute", 0.0))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]
            ) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))

    def scale(value: tuple[float, float, float], factor: float
              ) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def compare_setting(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, bool) or isinstance(expected, str):
            if actual != expected:
                failures.append(Failure(
                    path, "fixture setting differs from profile",
                    actual, expected))
            return
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            value = _scalar(actual, path, failures)
            if value is not None and abs(value - float(expected)) > state_tolerance:
                failures.append(Failure(
                    path, "fixture setting differs from profile",
                    value, expected))
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append(Failure(
                    path, "fixture setting differs from profile",
                    actual, expected))
                return
            for index, (value, reference) in enumerate(zip(actual, expected)):
                compare_setting(value, reference, f"{path}[{index}]")
            return
        if actual != expected:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def snapshot(value: Any, path: str, node_count: int
                 ) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        particles = value.get("smoke_particles")
        if not isinstance(particles, list) or len(particles) != 1:
            failures.append(Failure(path + ".smoke_particles",
                                    "expected exactly one smoke particle",
                                    particles, 1))
            return None
        particle = particles[0]
        if not isinstance(particle, dict):
            failures.append(Failure(path + ".smoke_particles[0]",
                                    "expected a smoke particle object",
                                    particle))
            return None
        particle_path = path + ".smoke_particles[0]"
        if particle.get("slot") != 0:
            failures.append(Failure(particle_path + ".slot",
                                    "smoke slot/order changed",
                                    particle.get("slot"), 0))
        position = _vector(particle.get("position_m"),
                           particle_path + ".position_m", failures)
        velocity = _vector(particle.get("velocity_m_s"),
                           particle_path + ".velocity_m_s", failures)
        vorticity = _vector(particle.get("vorticity_per_s"),
                            particle_path + ".vorticity_per_s", failures)
        scalars: dict[str, float] = {}
        for field in ("age_seconds", "number_density", "pressure_m_s2"):
            parsed = _scalar(particle.get(field), particle_path + "." + field,
                             failures)
            if parsed is not None:
                scalars[field] = parsed
        raw_nodes = value.get("rope_nodes_m")
        raw_velocities = value.get("rope_velocities_m_s")
        if not isinstance(raw_nodes, list) or len(raw_nodes) != node_count:
            failures.append(Failure(path + ".rope_nodes_m",
                                    "unexpected rope-node count",
                                    raw_nodes, node_count))
            return None
        if (not isinstance(raw_velocities, list) or
                len(raw_velocities) != node_count):
            failures.append(Failure(path + ".rope_velocities_m_s",
                                    "unexpected rope velocity count",
                                    raw_velocities, node_count))
            return None
        nodes = [_vector(item, f"{path}.rope_nodes_m[{index}]", failures)
                 for index, item in enumerate(raw_nodes)]
        velocities = [_vector(
            item, f"{path}.rope_velocities_m_s[{index}]", failures)
            for index, item in enumerate(raw_velocities)]
        if (position is None or velocity is None or vorticity is None or
                len(scalars) != 3 or any(item is None for item in nodes) or
                any(item is None for item in velocities)):
            return None
        return {
            "particle_raw": particle,
            "position": position,
            "velocity": velocity,
            "vorticity": vorticity,
            "scalars": scalars,
            "nodes": [item for item in nodes if item is not None],
            "node_velocities": [item for item in velocities
                                if item is not None],
        }

    def projection(point: tuple[float, float, float],
                   nodes: list[tuple[float, float, float]],
                   velocities: list[tuple[float, float, float]],
                   path: str) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for index in range(len(nodes) - 1):
            edge = _subtract(nodes[index + 1], nodes[index])
            length_squared = dot(edge, edge)
            if length_squared <= 1.0e-12:
                failures.append(Failure(
                    f"{path}.rope_nodes_m[{index}]",
                    "rope segment is degenerate", length_squared, "> 0"))
                continue
            fraction = min(1.0, max(0.0, dot(
                _subtract(point, nodes[index]), edge) / length_squared))
            closest = add(nodes[index], scale(edge, fraction))
            radial = _subtract(point, closest)
            distance = _norm(radial)
            rope_velocity = add(
                velocities[index], scale(
                    _subtract(velocities[index + 1], velocities[index]),
                    fraction))
            if best is None or distance < best["distance"]:
                best = {"distance": distance, "closest": closest,
                        "radial": radial, "rope_velocity": rope_velocity,
                        "fraction": fraction, "segment": index}
        if best is None:
            return None
        if best["distance"] <= 1.0e-12:
            failures.append(Failure(path + ".smoke_particles[0].position_m",
                                    "smoke tracer lies on rope centerline"))
            return None
        best["normal"] = scale(best["radial"], 1.0 / best["distance"])
        return best

    def compare_snapshots(first: dict[str, Any], second: dict[str, Any],
                          path: str) -> None:
        for field in ("position", "velocity", "vorticity"):
            if _norm(_subtract(first[field], second[field])) > state_tolerance:
                failures.append(Failure(
                    path, "paired initial smoke state differs",
                    list(second[field]), list(first[field])))
        for field in first["scalars"]:
            if abs(first["scalars"][field] - second["scalars"][field]) > \
                    scalar_tolerance:
                failures.append(Failure(
                    path, "paired initial smoke scalar state differs",
                    second["scalars"][field], first["scalars"][field]))
        for field in ("nodes", "node_velocities"):
            for left, right in zip(first[field], second[field]):
                if _norm(_subtract(left, right)) > state_tolerance:
                    failures.append(Failure(
                        path, "paired initial rope state differs",
                        list(right), list(left)))

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actuals.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field, expected in definition.items():
            compare_setting(actual.get(field), expected, base + "." + field)
        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array",
                                    runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        expected_runs = {"control", "coupled"}
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(
                base + ".runs", "expected exactly the paired runs",
                sorted(indexed), sorted(expected_runs)))
            continue
        node_count = int(definition.get("rope_node_count", 0))
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial", node_count)
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final", node_count)
            if initial is not None and final is not None:
                states[run_id] = initial, final
        if set(states) != expected_runs:
            continue
        control_initial, control_final = states["control"]
        coupled_initial, coupled_final = states["coupled"]
        compare_snapshots(control_initial, coupled_initial,
                          base + ".runs[coupled].initial")

        centerline = [_vector(item, base + ".rope_centerline_m", failures)
                      for item in definition.get("rope_centerline_m", [])]
        canonical = [item for item in centerline if item is not None]
        rope_position_limit = float(tolerances.get(
            "maximum_rope_position_difference_m", 0.0))
        rope_velocity_limit = float(tolerances.get(
            "maximum_rope_velocity_difference_m_s", 0.0))
        for run_id, (initial, final) in states.items():
            for index, (before, after) in enumerate(zip(
                    initial["nodes"], final["nodes"])):
                reference = canonical[index] if index < len(canonical) else before
                for label, value in (("initial", before), ("final", after)):
                    difference = _norm(_subtract(value, reference))
                    if difference > rope_position_limit:
                        failures.append(Failure(
                            base + f".runs[{run_id}].{label}.rope_nodes_m[{index}]",
                            "dynamic rope moved without an advertised reverse mechanical response",
                            difference, rope_position_limit))
                speed = max(_norm(initial["node_velocities"][index]),
                            _norm(final["node_velocities"][index]))
                if speed > rope_velocity_limit:
                    failures.append(Failure(
                        base + f".runs[{run_id}].final.rope_velocities_m_s[{index}]",
                        "dynamic rope received an unadvertised reverse mechanical response",
                        speed, rope_velocity_limit))
        for index in range(node_count):
            position_delta = _norm(_subtract(
                control_final["nodes"][index],
                coupled_final["nodes"][index]))
            velocity_delta = _norm(_subtract(
                control_final["node_velocities"][index],
                coupled_final["node_velocities"][index]))
            if (position_delta > rope_position_limit or
                    velocity_delta > rope_velocity_limit):
                failures.append(Failure(
                    base + ".runs[coupled].final.rope_nodes_m",
                    "rope received an unadvertised reverse mechanical response",
                    max(position_delta, velocity_delta),
                    max(rope_position_limit, rope_velocity_limit)))

        projections = {
            "initial": projection(control_initial["position"],
                                  control_initial["nodes"],
                                  control_initial["node_velocities"],
                                  base + ".runs[control].initial"),
            "control": projection(control_final["position"],
                                  control_final["nodes"],
                                  control_final["node_velocities"],
                                  base + ".runs[control].final"),
            "coupled": projection(coupled_final["position"],
                                  coupled_final["nodes"],
                                  coupled_final["node_velocities"],
                                  base + ".runs[coupled].final"),
        }
        if any(value is None for value in projections.values()):
            continue
        initial_projection = projections["initial"]
        control_projection = projections["control"]
        coupled_projection = projections["coupled"]
        assert initial_projection is not None
        assert control_projection is not None
        assert coupled_projection is not None
        radius = float(definition.get("expected_clearance_m", 0.0))
        minimum_crossing = float(tolerances.get(
            "minimum_control_crossing_m", 0.0))
        if initial_projection["distance"] - radius < minimum_crossing:
            failures.append(Failure(
                base + ".runs[control].initial.smoke_particles[0].position_m",
                "initial smoke did not start outside rope contact radius",
                initial_projection["distance"] - radius, minimum_crossing))
        if radius - control_projection["distance"] < minimum_crossing:
            failures.append(Failure(
                base + ".runs[control].final.smoke_particles[0].position_m",
                "control smoke did not enter rope contact radius",
                radius - control_projection["distance"], minimum_crossing))
        incoming_side = dot(coupled_projection["radial"],
                            initial_projection["normal"])
        if incoming_side <= 0.0:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].position_m",
                "contact failed to keep smoke on its incoming side",
                incoming_side, "> 0"))
        clearance_error = abs(coupled_projection["distance"] - radius)
        clearance_limit = float(tolerances.get(
            "maximum_contact_clearance_error_m", 0.0))
        if clearance_error > clearance_limit:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].position_m",
                "contact clearance differs from declared rope radius",
                clearance_error, clearance_limit))
        fraction = coupled_projection["fraction"]
        if fraction < 0.1 or fraction > 0.9:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].position_m",
                "contact hit a rope end cap instead of the segment interior",
                fraction, [0.1, 0.9]))
        relative_velocity = _subtract(
            coupled_final["velocity"],
            coupled_projection["rope_velocity"])
        radial_velocity = dot(relative_velocity,
                              coupled_projection["normal"])
        radial_limit = float(tolerances.get(
            "maximum_contact_normal_velocity_m_s", 0.0))
        if abs(radial_velocity) > radial_limit:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].velocity_m_s",
                "contact left excessive normal velocity",
                abs(radial_velocity), radial_limit))
        tangential_velocity = _norm(_subtract(
            relative_velocity,
            scale(coupled_projection["normal"], radial_velocity)))
        tangent_minimum = float(tolerances.get(
            "minimum_redirected_tangential_speed_m_s", 0.0))
        if tangential_velocity < tangent_minimum:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].velocity_m_s",
                "contact did not retain redirected tangential motion",
                tangential_velocity, tangent_minimum))
        speed_limit = float(definition.get("smoke_maximum_speed_m_s", 0.0)) + \
            float(tolerances.get("maximum_smoke_speed_error_m_s", 0.0))
        maximum_speed = max(_norm(state["velocity"])
                            for pair in states.values() for state in pair)
        if maximum_speed > speed_limit:
            failures.append(Failure(base + ".runs",
                                    "smoke speed exceeds profile bound",
                                    maximum_speed, speed_limit))

        for label, state, projected in (
                ("control.initial", control_initial, initial_projection),
                ("control.final", control_final, control_projection),
                ("coupled.initial", coupled_initial, initial_projection),
                ("coupled.final", coupled_final, coupled_projection)):
            raw = state["particle_raw"]
            if "closest_rope_distance_m" in raw:
                reported = _scalar(raw["closest_rope_distance_m"],
                                   base + f".runs[{label}].smoke_particles[0].closest_rope_distance_m",
                                   failures)
                if (reported is not None and
                        abs(reported - projected["distance"]) > state_tolerance):
                    failures.append(Failure(
                        base + f".runs[{label}].smoke_particles[0].closest_rope_distance_m",
                        "reported rope distance differs from public state",
                        reported, projected["distance"]))
            if "rope_contact_normal" in raw:
                reported_normal = _vector(
                    raw["rope_contact_normal"],
                    base + f".runs[{label}].smoke_particles[0].rope_contact_normal",
                    failures)
                if (reported_normal is not None and _norm(_subtract(
                        reported_normal, projected["normal"])) > state_tolerance):
                    failures.append(Failure(
                        base + f".runs[{label}].smoke_particles[0].rope_contact_normal",
                        "reported rope normal differs from public state",
                        list(reported_normal), list(projected["normal"])))

        observations[identity] = {
            "initial_rope_distance_m": initial_projection["distance"],
            "control_final_rope_distance_m": control_projection["distance"],
            "coupled_final_rope_distance_m": coupled_projection["distance"],
            "coupled_contact_segment_fraction": fraction,
            "coupled_relative_radial_velocity_m_s": radial_velocity,
            "coupled_relative_tangential_speed_m_s": tangential_velocity,
            "maximum_smoke_speed_m_s": maximum_speed,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actuals),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_soft_smoke_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    """Reuse the public triangulated-surface oracle with soft-body names."""
    key_map = {
        "soft_body_surface_vertices_m": "cloth_vertices_m",
        "soft_body_surface_velocities_m_s": "cloth_velocities_m_s",
        "soft_body_surface_triangle_indices": "cloth_triangle_indices",
        "soft_body_dynamic": "cloth_dynamic",
        "soft_body_node_mass_kg": "cloth_vertex_mass_kg",
        "soft_body_node_radius_m": "cloth_thickness_m",
        "maximum_soft_body_position_difference_m":
            "maximum_cloth_position_difference_m",
        "maximum_soft_body_velocity_difference_m_s":
            "maximum_cloth_velocity_difference_m_s",
        "soft_body_surface_normal": "cloth_normal",
    }

    def remap(value: Any) -> Any:
        if isinstance(value, dict):
            return {key_map.get(key, key): remap(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [remap(item) for item in value]
        return copy.deepcopy(value)

    report = _evaluate_cloth_smoke_contact(
        remap(profile), remap(result), expected_hash, driver, failures)
    inverse = {target: source for source, target in key_map.items()}
    for failure in report["failures"]:
        path = failure["path"]
        for target, source in inverse.items():
            path = path.replace(target, source)
        failure["path"] = path
        message = failure["message"]
        message = message.replace("cloth surface", "soft body surface")
        message = message.replace("cloth triangle", "soft body surface triangle")
        message = message.replace("cloth vertex", "soft body surface vertex")
        message = message.replace("cloth state", "soft body surface state")
        message = message.replace("cloth differs", "soft body surface differs")
        message = message.replace(
            "cloth received", "soft body received")
        failure["message"] = message
    return report


def _evaluate_fluid_cloth_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    def cross(a: tuple[float, float, float],
              b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def sub(a: tuple[float, float, float],
            b: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(first - second for first, second in zip(a, b))

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)

    def normal(vertices: list[tuple[float, float, float]], path: str
               ) -> tuple[float, float, float] | None:
        value = cross(sub(vertices[1], vertices[0]),
                      sub(vertices[2], vertices[0]))
        length = _norm(value)
        if length <= 1.0e-12:
            failures.append(Failure(path, "cloth triangle is degenerate", length))
            return None
        return scale(value, 1.0 / length)

    def close_number(actual: Any, expected: Any, path: str) -> None:
        number = _scalar(actual, path, failures)
        reference = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (number is not None and reference is not None and
                abs(number - reference) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    number, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(actual, path, failures)
        reference = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                _norm(sub(value, reference)) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def snapshot(value: Any, path: str,
                 definition: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        particle_position = _vector(value.get("particle_position_m"),
                                    path + ".particle_position_m", failures)
        particle_velocity = _vector(value.get("particle_velocity_m_s"),
                                    path + ".particle_velocity_m_s", failures)
        raw_vertices = value.get("cloth_vertices_m")
        raw_velocities = value.get("cloth_velocities_m_s")
        if not isinstance(raw_vertices, list) or len(raw_vertices) != 3:
            failures.append(Failure(path + ".cloth_vertices_m",
                                    "expected three cloth vertices", raw_vertices))
            return None
        if not isinstance(raw_velocities, list) or len(raw_velocities) != 3:
            failures.append(Failure(path + ".cloth_velocities_m_s",
                                    "expected three cloth velocities",
                                    raw_velocities))
            return None
        vertices = [_vector(item, f"{path}.cloth_vertices_m[{index}]", failures)
                    for index, item in enumerate(raw_vertices)]
        velocities = [_vector(item,
                              f"{path}.cloth_velocities_m_s[{index}]", failures)
                      for index, item in enumerate(raw_velocities)]
        if (particle_position is None or particle_velocity is None or
                any(item is None for item in vertices) or
                any(item is None for item in velocities)):
            return None
        typed_vertices = [item for item in vertices if item is not None]
        typed_velocities = [item for item in velocities if item is not None]
        surface_normal = normal(typed_vertices, path + ".cloth_normal")
        if surface_normal is None:
            return None
        separation = sum(component * axis for component, axis in zip(
            sub(particle_position, typed_vertices[0]), surface_normal))
        momentum = scale(particle_velocity, float(
            definition["particle_mass_kg"]))
        vertex_mass = float(definition["cloth_vertex_mass_kg"])
        for velocity in typed_velocities:
            momentum = tuple(first + second for first, second in zip(
                momentum, scale(velocity, vertex_mass)))
        reported_separation = _scalar(value.get("signed_separation_m"),
                                      path + ".signed_separation_m", failures)
        reported_momentum = _vector(value.get("endpoint_momentum_kg_m_s"),
                                    path + ".endpoint_momentum_kg_m_s", failures)
        if (reported_separation is not None and
                abs(reported_separation - separation) > state_tolerance):
            failures.append(Failure(path + ".signed_separation_m",
                                    "reported separation differs from public state",
                                    reported_separation, separation))
        if (reported_momentum is not None and
                _norm(sub(reported_momentum, momentum)) > state_tolerance):
            failures.append(Failure(path + ".endpoint_momentum_kg_m_s",
                                    "reported momentum differs from public state",
                                    list(reported_momentum), list(momentum)))
        return {"particle_position": particle_position,
                "particle_velocity": particle_velocity,
                "cloth_vertices": typed_vertices,
                "cloth_velocities": typed_velocities,
                "normal": surface_normal,
                "separation": separation,
                "momentum": momentum}

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "mode_flags", "cloth_preserve_volume",
                      "cloth_triangle_indices", "oracle", "step_count",
                      "substeps"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in ("timestep_seconds", "cloth_vertex_mass_kg",
                      "particle_mass_kg", "contact_distance_m",
                      "stiffness_n_m", "damping_n_s_m",
                      "tangential_drag_n_s_m", "maximum_force_n"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("gravity_m_s2", "particle_initial_position_m",
                      "particle_initial_velocity_m_s"):
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        actual_vertices = actual.get("cloth_vertices_m")
        expected_vertices = definition.get("cloth_vertices_m")
        if (not isinstance(actual_vertices, list) or
                not isinstance(expected_vertices, list) or
                len(actual_vertices) != len(expected_vertices)):
            failures.append(Failure(base + ".cloth_vertices_m",
                                    "fixture setting differs from profile",
                                    actual_vertices, expected_vertices))
        else:
            for index, (value, expected) in enumerate(zip(
                    actual_vertices, expected_vertices)):
                close_vector(value, expected,
                             f"{base}.cloth_vertices_m[{index}]")

        initial = snapshot(actual.get("initial"), base + ".initial",
                           definition)
        first = snapshot(actual.get("first_step"), base + ".first_step",
                         definition)
        final = snapshot(actual.get("final"), base + ".final", definition)
        if initial is None or first is None or final is None:
            continue
        sample_count += 2
        distance = float(definition["contact_distance_m"])
        first_change = initial["separation"] - first["separation"]
        maximum_change = float(tolerances.get(
            "first_step_maximum_separation_change_m", 0.0))
        if first_change < 0.0:
            failures.append(Failure(base + ".first_step",
                                    "contact did not reduce overlap",
                                    first_change, 0.0))
        if first_change > maximum_change:
            failures.append(Failure(base + ".first_step",
                                    "contact-only response snapped like containment",
                                    first_change, maximum_change))
        minimum_speed = float(tolerances.get(
            "first_step_minimum_endpoint_speed_m_s", 0.0))
        fluid_normal = sum(component * axis for component, axis in zip(
            first["particle_velocity"], first["normal"]))
        cloth_mean_velocity = tuple(sum(velocity[component]
            for velocity in first["cloth_velocities"]) / 3.0
            for component in range(3))
        cloth_normal = sum(component * axis for component, axis in zip(
            cloth_mean_velocity, first["normal"]))
        if fluid_normal >= -minimum_speed:
            failures.append(Failure(base + ".first_step.particle_velocity_m_s",
                                    "fluid lacks inward normal response",
                                    fluid_normal, -minimum_speed))
        if cloth_normal <= minimum_speed:
            failures.append(Failure(base + ".first_step.cloth_velocities_m_s",
                                    "cloth lacks reciprocal normal response",
                                    cloth_normal, minimum_speed))
        terminal_violation = max(0.0, distance + final["separation"])
        allowed_violation = float(tolerances.get(
            "terminal_contact_violation_m", 0.0))
        if terminal_violation > allowed_violation:
            failures.append(Failure(base + ".final.signed_separation_m",
                                    "terminal contact distance is violated",
                                    terminal_violation, allowed_violation))

        normal_momentum_tolerance = float(tolerances.get(
            "normal_momentum_error_kg_m_s", 0.0))
        tangential_momentum_tolerance = float(tolerances.get(
            "tangential_momentum_error_kg_m_s", 0.0))
        maximum_tangent = float(tolerances.get(
            "maximum_tangential_speed_m_s", 0.0))
        maximum_speed = float(tolerances.get(
            "maximum_endpoint_speed_m_s", 0.0))
        for label, state in (("first_step", first), ("final", final)):
            normal_momentum = sum(component * axis for component, axis in zip(
                state["momentum"], state["normal"]))
            tangent_momentum = sub(state["momentum"],
                                   scale(state["normal"], normal_momentum))
            if abs(normal_momentum) > normal_momentum_tolerance:
                failures.append(Failure(f"{base}.{label}.endpoint_momentum_kg_m_s",
                                        "reciprocal normal momentum is unbalanced",
                                        normal_momentum,
                                        normal_momentum_tolerance))
            if _norm(tangent_momentum) > tangential_momentum_tolerance:
                failures.append(Failure(f"{base}.{label}.endpoint_momentum_kg_m_s",
                                        "tangential momentum is unbalanced",
                                        _norm(tangent_momentum),
                                        tangential_momentum_tolerance))
            velocities = [state["particle_velocity"],
                          *state["cloth_velocities"]]
            for velocity_index, velocity in enumerate(velocities):
                speed = _norm(velocity)
                normal_speed = sum(component * axis for component, axis in zip(
                    velocity, state["normal"]))
                tangential_speed = _norm(sub(
                    velocity, scale(state["normal"], normal_speed)))
                if speed > maximum_speed:
                    failures.append(Failure(
                        f"{base}.{label}.velocity[{velocity_index}]",
                        "endpoint speed exceeds profile bound", speed,
                        maximum_speed))
                if tangential_speed > maximum_tangent:
                    failures.append(Failure(
                        f"{base}.{label}.velocity[{velocity_index}]",
                        "contact introduced tangential motion", tangential_speed,
                        maximum_tangent))
        observations[identity] = {
            "initial_signed_separation_m": initial["separation"],
            "first_step_signed_separation_m": first["separation"],
            "final_signed_separation_m": final["separation"],
            "terminal_contact_violation_m": terminal_violation,
            "first_step_fluid_normal_velocity_m_s": fluid_normal,
            "first_step_cloth_mean_normal_velocity_m_s": cloth_normal,
            "first_step_momentum_error_kg_m_s": _norm(first["momentum"]),
            "final_momentum_error_kg_m_s": _norm(final["momentum"]),
        }

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_smoke_rigid_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    """Evaluate paired smoke/rigid contact from public geometry and states."""
    definitions = {item["id"]: item for item in profile.get("fixtures", [])
                   if isinstance(item, dict) and
                   isinstance(item.get("id"), str)}
    raw_actual = result.get("fixtures")
    if not isinstance(raw_actual, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                raw_actual))
        raw_actual = []
    actuals: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_actual):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", item))
            continue
        identity = item["id"]
        if identity in actuals:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actuals[identity] = item
    for identity in sorted(set(definitions) - set(actuals)):
        failures.append(Failure("$.fixtures", "missing required fixture",
                                identity))
    for identity in sorted(set(actuals) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", identity))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))
    scalar_tolerance = float(tolerances.get(
        "smoke_scalar_state_absolute", 0.0))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]
            ) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))

    def sub(left: tuple[float, float, float],
            right: tuple[float, float, float]
            ) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(left, right))

    def scale(value: tuple[float, float, float], factor: float
              ) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]
              ) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def compare_setting(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, bool) or isinstance(expected, str):
            if actual != expected:
                failures.append(Failure(
                    path, "fixture setting differs from profile",
                    actual, expected))
            return
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            value = _scalar(actual, path, failures)
            if value is not None and abs(value - float(expected)) > state_tolerance:
                failures.append(Failure(
                    path, "fixture setting differs from profile",
                    value, expected))
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append(Failure(
                    path, "fixture setting differs from profile",
                    actual, expected))
                return
            for index, (value, reference) in enumerate(zip(actual, expected)):
                compare_setting(value, reference, f"{path}[{index}]")
            return
        if actual != expected:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def snapshot(value: Any, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        particles = value.get("smoke_particles")
        rigid = value.get("rigid")
        if not isinstance(particles, list) or len(particles) != 1:
            failures.append(Failure(path + ".smoke_particles",
                                    "expected exactly one smoke particle",
                                    particles, 1))
            return None
        if not isinstance(particles[0], dict):
            failures.append(Failure(path + ".smoke_particles[0]",
                                    "expected a smoke particle object",
                                    particles[0]))
            return None
        if not isinstance(rigid, dict):
            failures.append(Failure(path + ".rigid",
                                    "expected a rigid state object", rigid))
            return None
        particle = particles[0]
        particle_path = path + ".smoke_particles[0]"
        if particle.get("slot") != 0:
            failures.append(Failure(particle_path + ".slot",
                                    "smoke slot/order changed",
                                    particle.get("slot"), 0))
        smoke_position = _vector(particle.get("position_m"),
                                 particle_path + ".position_m", failures)
        smoke_velocity = _vector(particle.get("velocity_m_s"),
                                 particle_path + ".velocity_m_s", failures)
        smoke_vorticity = _vector(particle.get("vorticity_per_s"),
                                  particle_path + ".vorticity_per_s", failures)
        smoke_scalars: dict[str, float] = {}
        for field in ("age_seconds", "number_density", "pressure_m_s2"):
            parsed = _scalar(particle.get(field), particle_path + "." + field,
                             failures)
            if parsed is not None:
                smoke_scalars[field] = parsed
        rigid_position = _vector(rigid.get("position_m"),
                                 path + ".rigid.position_m", failures)
        rigid_orientation = _quaternion(rigid.get("orientation_xyzw"),
                                        path + ".rigid.orientation_xyzw",
                                        failures)
        rigid_linear = _vector(rigid.get("linear_velocity_m_s"),
                               path + ".rigid.linear_velocity_m_s", failures)
        rigid_angular = _vector(rigid.get("angular_velocity_rad_s"),
                                path + ".rigid.angular_velocity_rad_s",
                                failures)
        values = (smoke_position, smoke_velocity, smoke_vorticity,
                  rigid_position, rigid_orientation, rigid_linear,
                  rigid_angular)
        if any(item is None for item in values) or len(smoke_scalars) != 3:
            return None
        return {
            "smoke_position": smoke_position,
            "smoke_velocity": smoke_velocity,
            "smoke_vorticity": smoke_vorticity,
            "smoke_scalars": smoke_scalars,
            "rigid_position": rigid_position,
            "rigid_orientation": rigid_orientation,
            "rigid_linear": rigid_linear,
            "rigid_angular": rigid_angular,
        }

    def compare_initials(control: dict[str, Any],
                         coupled: dict[str, Any], path: str) -> None:
        for field in ("smoke_position", "smoke_velocity", "smoke_vorticity",
                      "rigid_position", "rigid_linear", "rigid_angular"):
            if _norm(sub(control[field], coupled[field])) > state_tolerance:
                failures.append(Failure(
                    path + "." + field, "paired initial state differs",
                    list(coupled[field]), list(control[field])))
        if _norm(tuple(a - b for a, b in zip(
                control["rigid_orientation"],
                coupled["rigid_orientation"]))) > state_tolerance:
            failures.append(Failure(
                path + ".rigid_orientation", "paired initial state differs",
                list(coupled["rigid_orientation"]),
                list(control["rigid_orientation"])))
        for field, value in control["smoke_scalars"].items():
            if abs(value - coupled["smoke_scalars"][field]) > scalar_tolerance:
                failures.append(Failure(
                    path + ".smoke_scalars." + field,
                    "paired initial smoke scalar state differs",
                    coupled["smoke_scalars"][field], value))

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actuals.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field, expected in definition.items():
            compare_setting(actual.get(field), expected, base + "." + field)
        runs = actual.get("runs")
        if not isinstance(runs, list):
            failures.append(Failure(base + ".runs", "expected a run array",
                                    runs))
            continue
        indexed = {run.get("id"): run for run in runs
                   if isinstance(run, dict) and isinstance(run.get("id"), str)}
        expected_runs = {"control", "coupled"}
        if set(indexed) != expected_runs or len(indexed) != len(runs):
            failures.append(Failure(
                base + ".runs", "expected exactly the paired runs",
                sorted(indexed), sorted(expected_runs)))
            continue
        states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for run_id in sorted(expected_runs):
            initial = snapshot(indexed[run_id].get("initial"),
                               base + f".runs[{run_id}].initial")
            final = snapshot(indexed[run_id].get("final"),
                             base + f".runs[{run_id}].final")
            if initial is not None and final is not None:
                states[run_id] = initial, final
        if set(states) != expected_runs:
            continue
        control_initial, control_final = states["control"]
        coupled_initial, coupled_final = states["coupled"]
        compare_initials(control_initial, coupled_initial,
                         base + ".runs[coupled].initial")

        raw_vertices = definition.get("rigid_geometry_vertices_m")
        face_indices = definition.get("impact_face_vertex_indices")
        if (not isinstance(raw_vertices, list) or
                not isinstance(face_indices, list) or len(face_indices) < 3):
            failures.append(Failure(base + ".impact_face_vertex_indices",
                                    "invalid public impact face", face_indices))
            continue
        vertices = [_vector(item, base + ".rigid_geometry_vertices_m",
                            failures) for item in raw_vertices]
        if any(item is None for item in vertices) or any(
                not isinstance(index, int) or isinstance(index, bool) or
                index < 0 or index >= len(vertices) for index in face_indices):
            failures.append(Failure(base + ".impact_face_vertex_indices",
                                    "invalid public impact face", face_indices))
            continue
        face = [vertices[index] for index in face_indices]
        assert all(item is not None for item in face)
        normal_raw = cross(sub(face[1], face[0]), sub(face[2], face[0]))
        normal_size = _norm(normal_raw)
        if normal_size <= 1.0e-12:
            failures.append(Failure(base + ".impact_face_vertex_indices",
                                    "impact face is degenerate", normal_size))
            continue
        outward = scale(normal_raw, 1.0 / normal_size)
        incoming = scale(outward, -1.0)

        def clearance(state: dict[str, Any]) -> float:
            face_point = add(face[0], state["rigid_position"])
            return dot(sub(state["smoke_position"], face_point), outward)

        initial_clearance = clearance(control_initial)
        control_clearance = clearance(control_final)
        coupled_clearance = clearance(coupled_final)
        contact_distance = float(definition.get(
            "coupling_contact_distance_m", 0.0))
        initial_margin = initial_clearance - contact_distance
        minimum_initial = float(tolerances.get(
            "minimum_initial_clearance_margin_m", 0.0))
        if initial_margin < minimum_initial:
            failures.append(Failure(
                base + ".runs[control].initial.smoke_particles[0].position_m",
                "smoke did not start outside the declared contact shell",
                initial_margin, minimum_initial))
        control_penetration = contact_distance - control_clearance
        minimum_penetration = float(tolerances.get(
            "minimum_control_penetration_m", 0.0))
        if control_penetration < minimum_penetration:
            failures.append(Failure(
                base + ".runs[control].final.smoke_particles[0].position_m",
                "control smoke did not cross the declared contact shell",
                control_penetration, minimum_penetration))
        clearance_error = abs(coupled_clearance - contact_distance)
        clearance_limit = float(tolerances.get(
            "maximum_contact_clearance_error_m", 0.0))
        if clearance_error > clearance_limit:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].position_m",
                "contact clearance differs from the declared distance",
                clearance_error, clearance_limit))

        relative_velocity = sub(coupled_final["smoke_velocity"],
                                coupled_final["rigid_linear"])
        inward_relative_speed = -dot(relative_velocity, outward)
        inward_limit = float(tolerances.get(
            "maximum_inward_relative_normal_velocity_m_s", 0.0))
        if inward_relative_speed > inward_limit:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].velocity_m_s",
                "contact left inward relative normal motion",
                inward_relative_speed, inward_limit))
        smoke_response = dot(sub(coupled_final["smoke_velocity"],
                                 control_final["smoke_velocity"]), outward)
        minimum_smoke_response = float(tolerances.get(
            "minimum_smoke_normal_speed_reduction_m_s", 0.0))
        if smoke_response < minimum_smoke_response:
            failures.append(Failure(
                base + ".runs[coupled].final.smoke_particles[0].velocity_m_s",
                "smoke lacks an outward normal contact response",
                smoke_response, minimum_smoke_response))
        rigid_response = dot(sub(coupled_final["rigid_linear"],
                                 control_final["rigid_linear"]), incoming)
        rigid_impulse = rigid_response * float(definition.get(
            "rigid_mass_kg", 0.0))
        minimum_rigid_impulse = float(tolerances.get(
            "minimum_rigid_normal_impulse_n_s", 0.0))
        if rigid_impulse < minimum_rigid_impulse:
            failures.append(Failure(
                base + ".runs[coupled].final.rigid.linear_velocity_m_s",
                "rigid body lacks the reciprocal normal response",
                rigid_impulse, minimum_rigid_impulse))
        control_rigid_speed = max(
            _norm(control_initial["rigid_linear"]),
            _norm(control_final["rigid_linear"]))
        control_limit = float(tolerances.get(
            "maximum_control_rigid_speed_m_s", 0.0))
        if control_rigid_speed > control_limit:
            failures.append(Failure(
                base + ".runs[control].final.rigid.linear_velocity_m_s",
                "uncoupled rigid body moved", control_rigid_speed,
                control_limit))
        angular_speed = _norm(coupled_final["rigid_angular"])
        angular_limit = float(tolerances.get(
            "maximum_rigid_angular_speed_rad_s", 0.0))
        if angular_speed > angular_limit:
            failures.append(Failure(
                base + ".runs[coupled].final.rigid.angular_velocity_rad_s",
                "rigid angular speed exceeds the declared fixture bound",
                angular_speed, angular_limit))
        expected_orientation = (0.0, 0.0, 0.0, 1.0)
        orientation_drift = _norm(tuple(
            value - reference for value, reference in zip(
                coupled_final["rigid_orientation"], expected_orientation)))
        orientation_limit = float(tolerances.get(
            "maximum_orientation_drift", 0.0))
        if orientation_drift > orientation_limit:
            failures.append(Failure(
                base + ".runs[coupled].final.rigid.orientation_xyzw",
                "contact changed rigid orientation during the impulse step",
                orientation_drift, orientation_limit))
        speed_limit = float(definition.get("smoke_maximum_speed_m_s", 0.0)) + \
            float(tolerances.get("maximum_smoke_speed_error_m_s", 0.0))
        maximum_smoke_speed = max(
            _norm(state["smoke_velocity"])
            for pair in states.values() for state in pair)
        if maximum_smoke_speed > speed_limit:
            failures.append(Failure(base + ".runs",
                                    "smoke speed exceeds profile bound",
                                    maximum_smoke_speed, speed_limit))

        observations[identity] = {
            "initial_face_clearance_m": initial_clearance,
            "control_final_face_clearance_m": control_clearance,
            "coupled_final_face_clearance_m": coupled_clearance,
            "coupled_clearance_error_m": clearance_error,
            "coupled_inward_relative_normal_velocity_m_s":
                inward_relative_speed,
            "smoke_outward_normal_response_m_s": smoke_response,
            "rigid_reciprocal_normal_impulse_n_s": rigid_impulse,
            "coupled_rigid_angular_speed_rad_s": angular_speed,
            "maximum_smoke_speed_m_s": maximum_smoke_speed,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actuals),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_fluid_rigid_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def sub(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)  # type: ignore[return-value]

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def close_number(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        reference = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                abs(value - reference) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(actual, path, failures)
        reference = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if (value is not None and reference is not None and
                _norm(sub(value, reference)) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def close_quaternion(actual: Any, expected: Any, path: str) -> None:
        value = _quaternion(actual, path, failures)
        reference = _quaternion(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if value is None or reference is None:
            return
        direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(value, reference)))
        negated = math.sqrt(sum((a + b) ** 2 for a, b in zip(value, reference)))
        if min(direct, negated) > state_tolerance:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def snapshot(value: Any, path: str,
                 include_contacts: bool) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        fluid = value.get("fluid")
        rigid = value.get("rigid")
        if not isinstance(fluid, dict):
            failures.append(Failure(path + ".fluid",
                                    "expected a fluid object", fluid))
            return None
        if not isinstance(rigid, dict):
            failures.append(Failure(path + ".rigid",
                                    "expected a rigid object", rigid))
            return None
        fluid_position = _vector(fluid.get("position_m"),
                                 path + ".fluid.position_m", failures)
        fluid_velocity = _vector(fluid.get("velocity_m_s"),
                                 path + ".fluid.velocity_m_s", failures)
        fluid_acceleration = _vector(fluid.get("acceleration_m_s2"),
                                     path + ".fluid.acceleration_m_s2", failures)
        rigid_position = _vector(rigid.get("position_m"),
                                 path + ".rigid.position_m", failures)
        rigid_orientation = _quaternion(rigid.get("orientation_xyzw"),
                                        path + ".rigid.orientation_xyzw", failures)
        rigid_linear = _vector(rigid.get("linear_velocity_m_s"),
                               path + ".rigid.linear_velocity_m_s", failures)
        rigid_angular = _vector(rigid.get("angular_velocity_rad_s"),
                                path + ".rigid.angular_velocity_rad_s", failures)
        stable_id = fluid.get("stable_id")
        if stable_id != 0:
            failures.append(Failure(path + ".fluid.stable_id",
                                    "fluid stable ID/order changed", stable_id, 0))
        values = (fluid_position, fluid_velocity, fluid_acceleration,
                  rigid_position, rigid_orientation, rigid_linear,
                  rigid_angular)
        if any(item is None for item in values):
            return None
        contacts: list[dict[str, Any]] = []
        if include_contacts:
            raw_contacts = value.get("contact_events")
            if not isinstance(raw_contacts, list):
                failures.append(Failure(path + ".contact_events",
                                        "expected a contact array", raw_contacts))
                return None
            for index, event in enumerate(raw_contacts):
                event_path = f"{path}.contact_events[{index}]"
                if not isinstance(event, dict):
                    failures.append(Failure(event_path,
                                            "expected a contact object", event))
                    continue
                position = _vector(event.get("position_m"),
                                   event_path + ".position_m", failures)
                normal = _vector(event.get("normal"),
                                 event_path + ".normal", failures)
                impulse = _scalar(event.get("normal_impulse_n_s"),
                                  event_path + ".normal_impulse_n_s", failures)
                if position is None or normal is None or impulse is None:
                    continue
                contacts.append({"type": event.get("type"),
                                 "first": event.get("first"),
                                 "second": event.get("second"),
                                 "position": position, "normal": normal,
                                 "impulse": impulse})
        return {"fluid_position": fluid_position,
                "fluid_velocity": fluid_velocity,
                "fluid_acceleration": fluid_acceleration,
                "rigid_position": rigid_position,
                "rigid_orientation": rigid_orientation,
                "rigid_linear": rigid_linear,
                "rigid_angular": rigid_angular,
                "contacts": contacts}

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "step_count", "substeps", "declared_mode_flags",
                      "exercised_mode", "containment_claimed",
                      "rigid_geometry_triangle_indices",
                      "impact_face_vertex_indices", "coupling_two_way",
                      "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in ("timestep_seconds", "fluid_rest_density_kg_m3",
                      "fluid_particle_radius_m", "fluid_support_radius_m",
                      "fluid_rest_particle_volume_m3", "fluid_particle_mass_kg",
                      "fluid_maximum_speed_m_s", "rigid_mass_kg",
                      "rigid_friction", "rigid_restitution",
                      "rigid_linear_damping_per_s",
                      "rigid_angular_damping_per_s",
                      "rigid_maximum_linear_speed_m_s",
                      "rigid_maximum_angular_speed_rad_s",
                      "rigid_collision_margin_m",
                      "coupling_contact_distance_m", "coupling_friction",
                      "coupling_maximum_acceleration_m_s2"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("gravity_m_s2", "particle_initial_position_m",
                      "particle_initial_velocity_m_s",
                      "rigid_initial_position_m",
                      "rigid_inertia_diagonal_kg_m2"):
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        close_quaternion(actual.get("rigid_initial_orientation_xyzw"),
                         definition.get("rigid_initial_orientation_xyzw"),
                         base + ".rigid_initial_orientation_xyzw")

        raw_vertices = actual.get("rigid_geometry_vertices_m")
        expected_vertices = definition.get("rigid_geometry_vertices_m")
        vertices: list[tuple[float, float, float]] = []
        if (not isinstance(raw_vertices, list) or
                not isinstance(expected_vertices, list) or
                len(raw_vertices) != len(expected_vertices)):
            failures.append(Failure(base + ".rigid_geometry_vertices_m",
                                    "fixture setting differs from profile",
                                    raw_vertices, expected_vertices))
        else:
            for index, (raw, expected) in enumerate(zip(
                    raw_vertices, expected_vertices)):
                path = f"{base}.rigid_geometry_vertices_m[{index}]"
                close_vector(raw, expected, path)
                parsed = _vector(raw, path, failures)
                if parsed is not None:
                    vertices.append(parsed)

        initial = snapshot(actual.get("initial"), base + ".initial", False)
        final = snapshot(actual.get("final"), base + ".final", True)
        if initial is None or final is None or len(vertices) != len(
                expected_vertices if isinstance(expected_vertices, list) else []):
            continue
        impact_indices = definition.get("impact_face_vertex_indices")
        if (not isinstance(impact_indices, list) or len(impact_indices) != 4 or
                any(not isinstance(index, int) or isinstance(index, bool) or
                    index < 0 or index >= len(vertices)
                    for index in impact_indices)):
            failures.append(Failure(base + ".impact_face_vertex_indices",
                                    "invalid impact face", impact_indices))
            continue
        face = [vertices[index] for index in impact_indices]
        face_cross = cross(sub(face[1], face[0]), sub(face[2], face[0]))
        face_normal = _unit(face_cross)
        if face_normal is None:
            failures.append(Failure(base + ".impact_face_vertex_indices",
                                    "impact face is degenerate", impact_indices))
            continue
        face_center = scale(_sum(face), 1.0 / len(face))
        initial_rigid_position = _vector(
            definition.get("rigid_initial_position_m"),
            base + ".rigid_initial_position_m", failures)
        expected_particle_position = _vector(
            definition.get("particle_initial_position_m"),
            base + ".particle_initial_position_m", failures)
        expected_particle_velocity = _vector(
            definition.get("particle_initial_velocity_m_s"),
            base + ".particle_initial_velocity_m_s", failures)
        if (initial_rigid_position is None or expected_particle_position is None or
                expected_particle_velocity is None):
            continue
        world_face_point = add(face[0], initial_rigid_position)
        world_face_center = add(face_center, initial_rigid_position)

        if _norm(sub(initial["fluid_position"],
                     expected_particle_position)) > state_tolerance:
            failures.append(Failure(base + ".initial.fluid.position_m",
                                    "initial fluid state differs from profile",
                                    list(initial["fluid_position"]),
                                    list(expected_particle_position)))
        if _norm(sub(initial["fluid_velocity"],
                     expected_particle_velocity)) > state_tolerance:
            failures.append(Failure(base + ".initial.fluid.velocity_m_s",
                                    "initial fluid state differs from profile",
                                    list(initial["fluid_velocity"]),
                                    list(expected_particle_velocity)))
        if _norm(sub(initial["rigid_position"],
                     initial_rigid_position)) > state_tolerance:
            failures.append(Failure(base + ".initial.rigid.position_m",
                                    "initial rigid state differs from profile",
                                    list(initial["rigid_position"]),
                                    list(initial_rigid_position)))
        for label, velocity in (("linear_velocity_m_s", initial["rigid_linear"]),
                                ("angular_velocity_rad_s", initial["rigid_angular"])):
            if _norm(velocity) > state_tolerance:
                failures.append(Failure(base + ".initial.rigid." + label,
                                        "initial rigid velocity must be zero",
                                        list(velocity), 0.0))

        density = float(definition["fluid_rest_density_kg_m3"])
        volume = float(definition["fluid_rest_particle_volume_m3"])
        fluid_mass = float(definition["fluid_particle_mass_kg"])
        derived_mass = density * volume
        if abs(fluid_mass - derived_mass) > state_tolerance:
            failures.append(Failure(base + ".fluid_particle_mass_kg",
                                    "particle mass differs from density times volume",
                                    fluid_mass, derived_mass))
        body_mass = float(definition["rigid_mass_kg"])
        radius = float(definition["fluid_particle_radius_m"])
        dt = float(definition["timestep_seconds"])
        relative_velocity = sub(initial["fluid_velocity"],
                                initial["rigid_linear"])
        unconstrained = add(initial["fluid_position"],
                            scale(relative_velocity, dt))
        predicted_penetration = max(
            0.0, radius - dot(sub(unconstrained, world_face_point), face_normal))
        incoming = dot(relative_velocity, face_normal)
        recovery_speed = min(
            float(definition["fluid_maximum_speed_m_s"]),
            min(0.5 * radius, 0.2 * predicted_penetration) / dt)
        expected_impulse = ((recovery_speed - incoming) /
                            (1.0 / fluid_mass + 1.0 / body_mass))

        final_clearance = dot(sub(final["fluid_position"], world_face_point),
                              face_normal) - radius
        overlap = max(0.0, -final_clearance)
        maximum_overlap = float(tolerances.get(
            "maximum_overlap_penetration_m", 0.0))
        if overlap > maximum_overlap:
            failures.append(Failure(base + ".final.fluid.position_m",
                                    "fluid overlapped the rigid surface",
                                    overlap, maximum_overlap))

        fluid_impulse = scale(sub(final["fluid_velocity"],
                                  initial["fluid_velocity"]), fluid_mass)
        rigid_impulse = scale(sub(final["rigid_linear"],
                                  initial["rigid_linear"]), body_mass)
        reciprocal_residual = _norm(add(fluid_impulse, rigid_impulse))
        maximum_reciprocal = float(tolerances.get(
            "maximum_reciprocal_impulse_residual_n_s", 0.0))
        if reciprocal_residual > maximum_reciprocal:
            failures.append(Failure(base + ".final",
                                    "fluid and rigid impulses are not reciprocal",
                                    reciprocal_residual, maximum_reciprocal))
        observed_impulse = dot(fluid_impulse, face_normal)
        minimum_impulse = float(tolerances.get(
            "minimum_contact_impulse_n_s", 0.0))
        if observed_impulse < minimum_impulse:
            failures.append(Failure(base + ".final.fluid.velocity_m_s",
                                    "fluid lacks a normal contact impulse",
                                    observed_impulse, minimum_impulse))
        expected_error = abs(observed_impulse - expected_impulse)
        maximum_expected_error = float(tolerances.get(
            "maximum_expected_impulse_error_n_s", 0.0))
        if expected_error > maximum_expected_error:
            failures.append(Failure(base + ".final.fluid.velocity_m_s",
                                    "normal impulse differs from the independent reduced-mass oracle",
                                    expected_error, maximum_expected_error))

        maximum_tangent = float(tolerances.get(
            "maximum_tangential_speed_m_s", 0.0))
        for label, velocity in (("fluid", final["fluid_velocity"]),
                                ("rigid", final["rigid_linear"])):
            normal_speed = dot(velocity, face_normal)
            tangent = sub(velocity, scale(face_normal, normal_speed))
            if _norm(tangent) > maximum_tangent:
                failures.append(Failure(base + f".final.{label}",
                                        "contact introduced tangential motion",
                                        _norm(tangent), maximum_tangent))
        maximum_angular = float(tolerances.get(
            "maximum_angular_speed_rad_s", 0.0))
        angular_speed = _norm(final["rigid_angular"])
        if angular_speed > maximum_angular:
            failures.append(Failure(base + ".final.rigid.angular_velocity_rad_s",
                                    "centered contact introduced angular motion",
                                    angular_speed, maximum_angular))
        translation = _norm(sub(final["rigid_position"],
                                initial["rigid_position"]))
        maximum_translation = float(tolerances.get(
            "maximum_rigid_translation_m", 0.0))
        if translation > maximum_translation:
            failures.append(Failure(base + ".final.rigid.position_m",
                                    "rigid translated before the contact impulse was applied",
                                    translation, maximum_translation))

        initial_energy = (0.5 * fluid_mass *
                          dot(initial["fluid_velocity"],
                              initial["fluid_velocity"]) +
                          0.5 * body_mass *
                          dot(initial["rigid_linear"], initial["rigid_linear"]))
        inertia = _vector(definition.get("rigid_inertia_diagonal_kg_m2"),
                          base + ".rigid_inertia_diagonal_kg_m2", failures)
        final_energy = (0.5 * fluid_mass *
                        dot(final["fluid_velocity"], final["fluid_velocity"]) +
                        0.5 * body_mass *
                        dot(final["rigid_linear"], final["rigid_linear"]))
        if inertia is not None:
            final_energy += 0.5 * sum(
                inertia[axis] * final["rigid_angular"][axis] ** 2
                for axis in range(3))
        energy_increase = final_energy - initial_energy
        maximum_energy_increase = float(tolerances.get(
            "maximum_energy_increase_j", 0.0))
        if energy_increase > maximum_energy_increase:
            failures.append(Failure(base + ".final",
                                    "inelastic contact increased kinetic energy",
                                    energy_increase, maximum_energy_increase))

        contacts = [event for event in final["contacts"]
                    if event["type"] == "LIQUID_RIGID" and
                    event["first"] == 0 and event["second"] == 0]
        if len(contacts) != 1:
            failures.append(Failure(base + ".final.contact_events",
                                    "expected exactly one public liquid-rigid contact",
                                    len(contacts), 1))
            contact_impulse_error = float("inf")
            contact_alignment = 0.0
            contact_point_error = float("inf")
        else:
            contact = contacts[0]
            normal_unit = _unit(contact["normal"])
            contact_alignment = (dot(normal_unit, face_normal)
                                 if normal_unit is not None else 0.0)
            minimum_alignment = float(tolerances.get(
                "minimum_contact_normal_alignment", 0.0))
            if contact_alignment < minimum_alignment:
                failures.append(Failure(
                    base + ".final.contact_events[0].normal",
                    "contact normal is not aligned with the exterior face",
                    contact_alignment, minimum_alignment))
            contact_point_error = _norm(sub(contact["position"],
                                            world_face_center))
            maximum_point_error = float(tolerances.get(
                "maximum_contact_point_error_m", 0.0))
            if contact_point_error > maximum_point_error:
                failures.append(Failure(
                    base + ".final.contact_events[0].position_m",
                    "contact point is not the centered cube-face impact",
                    contact_point_error, maximum_point_error))
            contact_impulse_error = abs(contact["impulse"] - observed_impulse)
            maximum_event_error = float(tolerances.get(
                "maximum_contact_event_impulse_error_n_s", 0.0))
            if contact_impulse_error > maximum_event_error:
                failures.append(Failure(
                    base + ".final.contact_events[0].normal_impulse_n_s",
                    "public contact impulse differs from endpoint momentum",
                    contact_impulse_error, maximum_event_error))

        observations[identity] = {
            "face_normal": list(face_normal),
            "predicted_penetration_m": predicted_penetration,
            "final_clearance_m": final_clearance,
            "observed_normal_impulse_n_s": observed_impulse,
            "expected_normal_impulse_n_s": expected_impulse,
            "expected_impulse_error_n_s": expected_error,
            "reciprocal_impulse_residual_n_s": reciprocal_residual,
            "contact_event_impulse_error_n_s": contact_impulse_error,
            "contact_normal_alignment": contact_alignment,
            "contact_point_error_m": contact_point_error,
            "kinetic_energy_change_j": energy_increase,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_soft_rigid_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def sub(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)  # type: ignore[return-value]

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def rotate(value: tuple[float, float, float],
               quaternion: tuple[float, float, float, float]
               ) -> tuple[float, float, float]:
        imaginary = (quaternion[0], quaternion[1], quaternion[2])
        twice_cross = scale(cross(imaginary, value), 2.0)
        return add(value, add(scale(twice_cross, quaternion[3]),
                              cross(imaginary, twice_cross)))

    def close_tree(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, bool) or expected is None or isinstance(expected, str):
            if actual != expected:
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
            return
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if (not isinstance(actual, (int, float)) or
                    isinstance(actual, bool) or
                    not math.isfinite(float(actual)) or
                    abs(float(actual) - float(expected)) > state_tolerance):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
                return
            for index, (item, wanted) in enumerate(zip(actual, expected)):
                close_tree(item, wanted, f"{path}[{index}]")
            return
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
                return
            for key, wanted in expected.items():
                close_tree(actual.get(key), wanted, f"{path}.{key}")
            return
        if actual != expected:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def parse_snapshot(value: Any, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        raw_nodes = value.get("soft_nodes")
        raw_surface = value.get("surface_vertices_m")
        raw_rigid = value.get("rigid")
        if not isinstance(raw_nodes, list) or len(raw_nodes) != 4:
            failures.append(Failure(path + ".soft_nodes",
                                    "expected four unique soft nodes", raw_nodes))
            return None
        nodes: list[dict[str, Any]] = []
        for index, node in enumerate(raw_nodes):
            item_path = f"{path}.soft_nodes[{index}]"
            if not isinstance(node, dict) or node.get("id") != index:
                failures.append(Failure(item_path,
                                        "soft-node ID/order changed", node, index))
                continue
            position = _vector(node.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(node.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            force = _vector(node.get("rigid_contact_force_n"),
                            item_path + ".rigid_contact_force_n", failures)
            if position is not None and velocity is not None and force is not None:
                nodes.append({"position": position, "velocity": velocity,
                              "force": force})
        if not isinstance(raw_surface, list) or len(raw_surface) != 10:
            failures.append(Failure(path + ".surface_vertices_m",
                                    "expected ten bound surface vertices",
                                    raw_surface))
            return None
        surface: list[tuple[float, float, float]] = []
        for index, vertex in enumerate(raw_surface):
            parsed = _vector(vertex, f"{path}.surface_vertices_m[{index}]",
                             failures)
            if parsed is not None:
                surface.append(parsed)
        if not isinstance(raw_rigid, dict):
            failures.append(Failure(path + ".rigid",
                                    "expected a rigid object", raw_rigid))
            return None
        rigid_position = _vector(raw_rigid.get("position_m"),
                                 path + ".rigid.position_m", failures)
        rigid_orientation = _quaternion(
            raw_rigid.get("orientation_xyzw"),
            path + ".rigid.orientation_xyzw", failures)
        rigid_linear = _vector(raw_rigid.get("linear_velocity_m_s"),
                               path + ".rigid.linear_velocity_m_s", failures)
        rigid_angular = _vector(raw_rigid.get("angular_velocity_rad_s"),
                                path + ".rigid.angular_velocity_rad_s", failures)
        if (len(nodes) != 4 or len(surface) != 10 or
                rigid_position is None or rigid_orientation is None or
                rigid_linear is None or rigid_angular is None):
            return None
        return {"nodes": nodes, "surface": surface,
                "rigid_position": rigid_position,
                "rigid_orientation": rigid_orientation,
                "rigid_linear": rigid_linear,
                "rigid_angular": rigid_angular}

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        settings = {key: value for key, value in actual.items()
                    if key not in {"initial", "final"}}
        close_tree(settings, definition, base)
        initial = parse_snapshot(actual.get("initial"), base + ".initial")
        final = parse_snapshot(actual.get("final"), base + ".final")
        if initial is None or final is None:
            continue

        expected_nodes = definition.get("prepared_soft_nodes_m")
        expected_velocity = _vector(
            definition.get("prepared_soft_velocity_m_s"),
            base + ".prepared_soft_velocity_m_s", failures)
        if not isinstance(expected_nodes, list) or len(expected_nodes) != 4 or \
                expected_velocity is None:
            failures.append(Failure(base + ".prepared_soft_nodes_m",
                                    "invalid prepared soft state", expected_nodes))
            continue
        for index, expected in enumerate(expected_nodes):
            wanted = _vector(expected,
                             f"{base}.prepared_soft_nodes_m[{index}]", failures)
            if wanted is not None and _norm(sub(
                    initial["nodes"][index]["position"], wanted)) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.soft_nodes[{index}].position_m",
                    "prepared soft position differs from profile",
                    list(initial["nodes"][index]["position"]), list(wanted)))
            if _norm(sub(initial["nodes"][index]["velocity"],
                         expected_velocity)) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.soft_nodes[{index}].velocity_m_s",
                    "prepared soft velocity differs from profile",
                    list(initial["nodes"][index]["velocity"]),
                    list(expected_velocity)))

        initial_rigid_position = _vector(
            definition.get("rigid_initial_position_m"),
            base + ".rigid_initial_position_m", failures)
        initial_rigid_orientation = _quaternion(
            definition.get("rigid_initial_orientation_xyzw"),
            base + ".rigid_initial_orientation_xyzw", failures)
        if (initial_rigid_position is None or initial_rigid_orientation is None):
            continue
        if _norm(sub(initial["rigid_position"], initial_rigid_position)) > \
                state_tolerance:
            failures.append(Failure(base + ".initial.rigid.position_m",
                                    "initial rigid position differs from profile",
                                    list(initial["rigid_position"]),
                                    list(initial_rigid_position)))
        direct_q = math.sqrt(sum((a - b) ** 2 for a, b in zip(
            initial["rigid_orientation"], initial_rigid_orientation)))
        negated_q = math.sqrt(sum((a + b) ** 2 for a, b in zip(
            initial["rigid_orientation"], initial_rigid_orientation)))
        if min(direct_q, negated_q) > state_tolerance:
            failures.append(Failure(base + ".initial.rigid.orientation_xyzw",
                                    "initial rigid orientation differs from profile",
                                    list(initial["rigid_orientation"]),
                                    list(initial_rigid_orientation)))
        if (_norm(initial["rigid_linear"]) > state_tolerance or
                _norm(initial["rigid_angular"]) > state_tolerance):
            failures.append(Failure(base + ".initial.rigid",
                                    "initial rigid endpoint must be stationary",
                                    {"linear": list(initial["rigid_linear"]),
                                     "angular": list(initial["rigid_angular"])}))

        bindings = definition.get("surface_bindings")
        inverse_masses = definition.get("soft_inverse_masses_kg_inverse")
        if (not isinstance(bindings, list) or len(bindings) != 10 or
                not isinstance(inverse_masses, list) or
                len(inverse_masses) != 4):
            failures.append(Failure(base, "invalid binding or mass profile",
                                    {"bindings": bindings,
                                     "inverse_masses": inverse_masses}))
            continue
        masses: list[float] = []
        for index, value in enumerate(inverse_masses):
            parsed = _scalar(value, f"{base}.soft_inverse_masses[{index}]",
                             failures)
            if parsed is None or parsed <= 0.0:
                failures.append(Failure(
                    f"{base}.soft_inverse_masses[{index}]",
                    "every oracle node must have positive inverse mass", value))
                parsed = 1.0
            masses.append(1.0 / parsed)
        incidence = [0, 0, 0, 0]
        reconstruction_error = 0.0
        for snapshot_name, snapshot_value in (("initial", initial),
                                               ("final", final)):
            for vertex, binding in enumerate(bindings):
                if not isinstance(binding, dict):
                    failures.append(Failure(
                        f"{base}.surface_bindings[{vertex}]",
                        "expected a binding object", binding))
                    continue
                node_ids = binding.get("nodes")
                weights = binding.get("weights")
                if (not isinstance(node_ids, list) or len(node_ids) != 4 or
                        not isinstance(weights, list) or len(weights) != 4):
                    failures.append(Failure(
                        f"{base}.surface_bindings[{vertex}]",
                        "binding needs four nodes and weights", binding))
                    continue
                reconstructed = (0.0, 0.0, 0.0)
                weight_sum = 0.0
                for slot, (node_id, weight_value) in enumerate(zip(
                        node_ids, weights)):
                    if (not isinstance(node_id, int) or isinstance(node_id, bool) or
                            node_id < 0 or node_id >= 4):
                        failures.append(Failure(
                            f"{base}.surface_bindings[{vertex}].nodes[{slot}]",
                            "binding node is out of range", node_id))
                        continue
                    weight = _scalar(
                        weight_value,
                        f"{base}.surface_bindings[{vertex}].weights[{slot}]",
                        failures)
                    if weight is None or weight < 0.0:
                        continue
                    if snapshot_name == "initial" and weight > 0.0:
                        incidence[node_id] += 1
                    weight_sum += weight
                    reconstructed = add(reconstructed, scale(
                        snapshot_value["nodes"][node_id]["position"], weight))
                if abs(weight_sum - 1.0) > state_tolerance:
                    failures.append(Failure(
                        f"{base}.surface_bindings[{vertex}].weights",
                        "binding weights must sum to one", weight_sum, 1.0))
                error = _norm(sub(snapshot_value["surface"][vertex],
                                  reconstructed))
                reconstruction_error = max(reconstruction_error, error)
                maximum_reconstruction = float(tolerances.get(
                    "maximum_surface_reconstruction_error_m", 0.0))
                if error > maximum_reconstruction:
                    failures.append(Failure(
                        f"{base}.{snapshot_name}.surface_vertices_m[{vertex}]",
                        "bound surface does not reconstruct from unique nodes",
                        error, maximum_reconstruction))
        minimum_incidence = int(definition.get(
            "minimum_surface_binding_incidence_per_node", 0))
        if min(incidence) < minimum_incidence:
            failures.append(Failure(base + ".surface_bindings",
                                    "surface does not repeatedly bind every unique node",
                                    incidence, minimum_incidence))

        soft_initial_momentum = (0.0, 0.0, 0.0)
        soft_final_momentum = (0.0, 0.0, 0.0)
        initial_energy = 0.0
        final_energy = 0.0
        for index, mass in enumerate(masses):
            initial_velocity = initial["nodes"][index]["velocity"]
            final_velocity = final["nodes"][index]["velocity"]
            soft_initial_momentum = add(
                soft_initial_momentum, scale(initial_velocity, mass))
            soft_final_momentum = add(
                soft_final_momentum, scale(final_velocity, mass))
            initial_energy += 0.5 * mass * dot(initial_velocity, initial_velocity)
            final_energy += 0.5 * mass * dot(final_velocity, final_velocity)
        rigid_mass_value = float(definition["rigid_mass_kg"])
        rigid_impulse = scale(sub(final["rigid_linear"],
                                  initial["rigid_linear"]), rigid_mass_value)
        soft_impulse = sub(soft_final_momentum, soft_initial_momentum)
        reciprocal_residual = _norm(add(soft_impulse, rigid_impulse))

        rigid_vertices = definition.get("rigid_geometry_vertices_m")
        face_indices = definition.get("impact_face_vertex_indices")
        if (not isinstance(rigid_vertices, list) or
                not isinstance(face_indices, list) or len(face_indices) != 4):
            failures.append(Failure(base, "invalid rigid impact face", face_indices))
            continue
        parsed_vertices = [_vector(
            vertex, f"{base}.rigid_geometry_vertices_m[{index}]", failures)
            for index, vertex in enumerate(rigid_vertices)]
        if any(vertex is None for vertex in parsed_vertices):
            continue
        face = [parsed_vertices[index] for index in face_indices]
        if any(vertex is None for vertex in face):
            continue
        face_values = [vertex for vertex in face if vertex is not None]
        local_normal = _unit(cross(sub(face_values[1], face_values[0]),
                                   sub(face_values[2], face_values[0])))
        if local_normal is None:
            failures.append(Failure(base + ".impact_face_vertex_indices",
                                    "impact face is degenerate", face_indices))
            continue
        normal = rotate(local_normal, final["rigid_orientation"])
        face_point = add(rotate(face_values[0], final["rigid_orientation"]),
                         final["rigid_position"])
        impact_node = int(definition.get("impact_node_index", -1))
        if impact_node < 0 or impact_node >= 4:
            failures.append(Failure(base + ".impact_node_index",
                                    "impact node is out of range", impact_node))
            continue
        radius = float(definition["soft_node_radius_m"])
        margin = float(definition["rigid_collision_margin_m"])
        clearance = dot(sub(final["nodes"][impact_node]["position"],
                            face_point), normal) - radius - margin
        overlap = max(0.0, -clearance)
        maximum_overlap = float(tolerances.get(
            "maximum_overlap_penetration_m", 0.0))
        if overlap > maximum_overlap:
            failures.append(Failure(
                f"{base}.final.soft_nodes[{impact_node}].position_m",
                "soft node overlapped the rigid surface", overlap,
                maximum_overlap))
        surface_penetration = max(
            0.0, max(-dot(sub(vertex, face_point), normal)
                     for vertex in final["surface"]))
        maximum_surface_penetration = float(tolerances.get(
            "maximum_surface_face_penetration_m", 0.0))
        if surface_penetration > maximum_surface_penetration:
            failures.append(Failure(base + ".final.surface_vertices_m",
                                    "bound surface crossed the rigid face",
                                    surface_penetration,
                                    maximum_surface_penetration))

        initial_face_point = add(
            rotate(face_values[0], initial["rigid_orientation"]),
            initial["rigid_position"])
        contact_point = sub(
            final["nodes"][impact_node]["position"],
            scale(normal, radius + margin))
        arm = sub(contact_point, final["rigid_position"])
        rigid_point_velocity = add(final["rigid_linear"],
                                   cross(final["rigid_angular"], arm))
        relative_final = sub(final["nodes"][impact_node]["velocity"],
                             rigid_point_velocity)
        closing_speed = max(0.0, -dot(relative_final, normal))
        maximum_closing = float(tolerances.get(
            "maximum_relative_closing_speed_m_s", 0.0))

        observed_impulse = dot(soft_impulse, normal)
        initial_contact_point = sub(
            initial["nodes"][impact_node]["position"],
            scale(rotate(local_normal, initial["rigid_orientation"]),
                  radius + margin))
        initial_arm = sub(initial_contact_point, initial["rigid_position"])
        initial_rigid_point_velocity = add(
            initial["rigid_linear"],
            cross(initial["rigid_angular"], initial_arm))
        incoming = dot(sub(initial["nodes"][impact_node]["velocity"],
                           initial_rigid_point_velocity),
                       rotate(local_normal, initial["rigid_orientation"]))
        expected_impulse = -incoming / (
            float(inverse_masses[impact_node]) + 1.0 / rigid_mass_value)
        expected_error = abs(observed_impulse - expected_impulse)
        reciprocal_limit = float(tolerances.get(
            "maximum_reciprocal_impulse_residual_n_s", 0.0))
        expected_limit = float(tolerances.get(
            "maximum_expected_impulse_error_n_s", 0.0))
        minimum_impulse = float(tolerances.get(
            "minimum_contact_impulse_n_s", 0.0))
        tangent_impulse = _norm(sub(
            soft_impulse, scale(normal, observed_impulse)))
        tangent_limit = float(tolerances.get(
            "maximum_tangential_impulse_n_s", 0.0))

        final_energy += 0.5 * rigid_mass_value * dot(
            final["rigid_linear"], final["rigid_linear"])
        initial_energy += 0.5 * rigid_mass_value * dot(
            initial["rigid_linear"], initial["rigid_linear"])
        inertia = definition.get("rigid_inertia_diagonal_kg_m2")
        if isinstance(inertia, list) and len(inertia) == 3:
            final_energy += 0.5 * sum(
                float(inertia[axis]) * final["rigid_angular"][axis] ** 2
                for axis in range(3))
            initial_energy += 0.5 * sum(
                float(inertia[axis]) * initial["rigid_angular"][axis] ** 2
                for axis in range(3))
        energy_change = final_energy - initial_energy

        is_control = identity == "separated-no-contact-control"
        if is_control:
            maximum_position = float(tolerances.get(
                "maximum_control_position_error_m", 0.0))
            maximum_velocity = float(tolerances.get(
                "maximum_control_velocity_error_m_s", 0.0))
            maximum_momentum = float(tolerances.get(
                "maximum_control_momentum_change_n_s", 0.0))
            maximum_force = float(tolerances.get(
                "maximum_control_contact_force_n", 0.0))
            position_error = 0.0
            velocity_error = 0.0
            contact_force = 0.0
            for index in range(4):
                expected_position = add(
                    initial["nodes"][index]["position"],
                    scale(initial["nodes"][index]["velocity"],
                          float(definition["timestep_seconds"])))
                position_error = max(position_error, _norm(sub(
                    final["nodes"][index]["position"], expected_position)))
                velocity_error = max(velocity_error, _norm(sub(
                    final["nodes"][index]["velocity"],
                    initial["nodes"][index]["velocity"])))
                contact_force = max(contact_force,
                                    _norm(final["nodes"][index]["force"]))
            momentum_change = _norm(add(soft_impulse, rigid_impulse)) + \
                _norm(soft_impulse) + _norm(rigid_impulse)
            if position_error > maximum_position:
                failures.append(Failure(base + ".final.soft_nodes",
                                        "separated control is not ballistic",
                                        position_error, maximum_position))
            if velocity_error > maximum_velocity:
                failures.append(Failure(base + ".final.soft_nodes",
                                        "separated control velocity changed",
                                        velocity_error, maximum_velocity))
            if momentum_change > maximum_momentum:
                failures.append(Failure(base + ".final",
                                        "separated control exchanged momentum",
                                        momentum_change, maximum_momentum))
            if contact_force > maximum_force:
                failures.append(Failure(base + ".final.soft_nodes",
                                        "separated control reported contact force",
                                        contact_force, maximum_force))
        else:
            if closing_speed > maximum_closing:
                failures.append(Failure(
                    f"{base}.final.soft_nodes[{impact_node}].velocity_m_s",
                    "soft and rigid endpoints retain closing speed",
                    closing_speed, maximum_closing))
            if observed_impulse < minimum_impulse:
                failures.append(Failure(base + ".final.soft_nodes",
                                        "soft-rigid contact lacks impulse activity",
                                        observed_impulse, minimum_impulse))
            if expected_error > expected_limit:
                failures.append(Failure(
                    base + ".final.soft_nodes",
                    "normal impulse differs from the unique-node reduced-mass oracle",
                    expected_error, expected_limit))
            if reciprocal_residual > reciprocal_limit:
                failures.append(Failure(base + ".final",
                                        "soft and rigid impulses are not reciprocal",
                                        reciprocal_residual, reciprocal_limit))
            if tangent_impulse > tangent_limit:
                failures.append(Failure(base + ".final.soft_nodes",
                                        "frictionless contact introduced tangential impulse",
                                        tangent_impulse, tangent_limit))
            angular_speed = _norm(final["rigid_angular"])
            angular_limit = float(tolerances.get(
                "maximum_angular_speed_rad_s", 0.0))
            if angular_speed > angular_limit:
                failures.append(Failure(
                    base + ".final.rigid.angular_velocity_rad_s",
                    "centered contact introduced angular motion",
                    angular_speed, angular_limit))
            energy_limit = float(tolerances.get(
                "maximum_energy_increase_j", 0.0))
            if energy_change > energy_limit:
                failures.append(Failure(base + ".final",
                                        "inelastic contact increased kinetic energy",
                                        energy_change, energy_limit))

        observations[identity] = {
            "unique_node_count": len(masses),
            "surface_vertex_count": len(final["surface"]),
            "minimum_binding_incidence": min(incidence),
            "maximum_surface_reconstruction_error_m": reconstruction_error,
            "final_node_clearance_m": clearance,
            "maximum_surface_face_penetration_m": surface_penetration,
            "observed_normal_impulse_n_s": observed_impulse,
            "expected_normal_impulse_n_s": expected_impulse,
            "expected_impulse_error_n_s": expected_error,
            "reciprocal_impulse_residual_n_s": reciprocal_residual,
            "relative_closing_speed_m_s": closing_speed,
            "tangential_impulse_n_s": tangent_impulse,
            "kinetic_energy_change_j": energy_change,
            "initial_face_point_m": list(initial_face_point),
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_fluid_rigid_containment(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances", "expected an object",
                                tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    definitions_value = profile.get("fixtures")
    if not isinstance(definitions_value, list):
        failures.append(Failure("$.profile.fixtures", "expected a fixture array",
                                definitions_value))
        definitions_value = []
    definitions = {fixture["id"]: fixture for fixture in definitions_value
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_value = result.get("fixtures")
    if not isinstance(actual_value, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_value))
        actual_value = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_value):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    def dot(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(component * factor for component in value)  # type: ignore[return-value]

    def rotate(value: tuple[float, float, float],
               quaternion: tuple[float, float, float, float]) -> tuple[
                   float, float, float]:
        vector = quaternion[:3]
        twice_cross = scale(cross(vector, value), 2.0)
        return add(value, add(scale(twice_cross, quaternion[3]),
                              cross(vector, twice_cross)))

    def close_scalar(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        wanted = _scalar(expected, "$.profile." + path[2:], failures)
        if (value is not None and wanted is not None and
                abs(value - wanted) > state_tolerance):
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, wanted))

    def close_vector(actual: Any, expected: Any, path: str,
                     message: str = "fixture setting differs from profile") -> None:
        value = _vector(actual, path, failures)
        wanted = _vector(expected, "$.profile." + path[2:], failures)
        if (value is not None and wanted is not None and
                max(abs(a - b) for a, b in zip(value, wanted)) >
                state_tolerance):
            failures.append(Failure(path, message, list(value), list(wanted)))

    def snapshot(value: Any, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a public endpoint snapshot",
                                    value))
            return None
        fluid = value.get("fluid")
        rigid = value.get("rigid")
        if not isinstance(fluid, dict):
            failures.append(Failure(path + ".fluid", "expected an object", fluid))
            fluid = {}
        if not isinstance(rigid, dict):
            failures.append(Failure(path + ".rigid", "expected an object", rigid))
            rigid = {}
        position = _vector(fluid.get("position_m"),
                           path + ".fluid.position_m", failures)
        velocity = _vector(fluid.get("velocity_m_s"),
                           path + ".fluid.velocity_m_s", failures)
        rigid_position = _vector(rigid.get("position_m"),
                                 path + ".rigid.position_m", failures)
        orientation = _quaternion(rigid.get("orientation_xyzw"),
                                  path + ".rigid.orientation_xyzw", failures)
        rigid_velocity = _vector(rigid.get("linear_velocity_m_s"),
                                 path + ".rigid.linear_velocity_m_s", failures)
        angular_velocity = _vector(rigid.get("angular_velocity_rad_s"),
                                   path + ".rigid.angular_velocity_rad_s",
                                   failures)
        if None in (position, velocity, rigid_position, orientation,
                    rigid_velocity, angular_velocity):
            return None
        quaternion_length = math.sqrt(sum(component * component
                                          for component in orientation))
        if abs(quaternion_length - 1.0) > state_tolerance:
            failures.append(Failure(path + ".rigid.orientation_xyzw",
                                    "rigid orientation must be a unit quaternion",
                                    quaternion_length, 1.0))
            return None
        normalized = tuple(component / quaternion_length
                           for component in orientation)
        return {"fluid_position": position, "fluid_velocity": velocity,
                "rigid_position": rigid_position,
                "orientation": normalized,
                "rigid_velocity": rigid_velocity,
                "angular_velocity": angular_velocity}

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "step_count", "substeps", "declared_mode_flags",
                      "exercised_mode", "contact_claimed",
                      "contact_event_capacity",
                      "rigid_geometry_triangle_indices", "coupling_two_way",
                      "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in (
                "timestep_seconds", "fluid_rest_density_kg_m3",
                "fluid_particle_radius_m", "fluid_support_radius_m",
                "fluid_rest_particle_volume_m3", "fluid_particle_mass_kg",
                "fluid_maximum_speed_m_s", "rigid_mass_kg", "rigid_friction",
                "rigid_restitution", "rigid_linear_damping_per_s",
                "rigid_angular_damping_per_s",
                "rigid_maximum_linear_speed_m_s",
                "rigid_maximum_angular_speed_rad_s", "rigid_collision_margin_m",
                "coupling_contact_distance_m", "coupling_friction",
                "coupling_maximum_acceleration_m_s2"):
            close_scalar(actual.get(field), definition.get(field),
                         base + "." + field)
        for field in ("gravity_m_s2", "particle_initial_position_m",
                      "particle_initial_velocity_m_s", "rigid_initial_position_m",
                      "rigid_initial_linear_velocity_m_s",
                      "rigid_initial_angular_velocity_rad_s",
                      "rigid_inertia_diagonal_kg_m2"):
            close_vector(actual.get(field), definition.get(field),
                         base + "." + field)
        actual_orientation = _quaternion(
            actual.get("rigid_initial_orientation_xyzw"),
            base + ".rigid_initial_orientation_xyzw", failures)
        wanted_orientation = _quaternion(
            definition.get("rigid_initial_orientation_xyzw"),
            "$.profile.fixtures.rigid_initial_orientation_xyzw", failures)
        if (actual_orientation is not None and wanted_orientation is not None and
                max(abs(a - b) for a, b in zip(actual_orientation,
                                                wanted_orientation)) >
                state_tolerance):
            failures.append(Failure(
                base + ".rigid_initial_orientation_xyzw",
                "fixture setting differs from profile",
                list(actual_orientation), list(wanted_orientation)))

        vertices_value = actual.get("rigid_geometry_vertices_m")
        expected_vertices_value = definition.get("rigid_geometry_vertices_m")
        vertices: list[tuple[float, float, float]] = []
        if (not isinstance(vertices_value, list) or
                not isinstance(expected_vertices_value, list) or
                len(vertices_value) != len(expected_vertices_value)):
            failures.append(Failure(base + ".rigid_geometry_vertices_m",
                                    "fixture setting differs from profile",
                                    vertices_value, expected_vertices_value))
        else:
            for index, (value, wanted) in enumerate(zip(
                    vertices_value, expected_vertices_value)):
                vertex = _vector(value,
                                 f"{base}.rigid_geometry_vertices_m[{index}]",
                                 failures)
                expected_vertex = _vector(
                    wanted,
                    f"$.profile.fixtures.rigid_geometry_vertices_m[{index}]",
                    failures)
                if vertex is not None:
                    vertices.append(vertex)
                if (vertex is not None and expected_vertex is not None and
                        max(abs(a - b) for a, b in zip(vertex,
                                                       expected_vertex)) >
                        state_tolerance):
                    failures.append(Failure(
                        f"{base}.rigid_geometry_vertices_m[{index}]",
                        "fixture setting differs from profile",
                        list(vertex), list(expected_vertex)))
        indices = actual.get("rigid_geometry_triangle_indices")
        manifold = True
        planes: list[tuple[tuple[float, float, float],
                           tuple[float, float, float]]] = []
        edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        if (not isinstance(indices, list) or len(indices) % 3 != 0 or
                any(not isinstance(index, int) or isinstance(index, bool) or
                    index < 0 or index >= len(vertices) for index in indices)):
            manifold = False
        else:
            for offset in range(0, len(indices), 3):
                triangle = indices[offset:offset + 3]
                first, second, third = (vertices[index] for index in triangle)
                normal = _unit(cross(_subtract(second, first),
                                     _subtract(third, first)))
                if normal is None:
                    manifold = False
                    continue
                planes.append((first, normal))
                for start, end in zip(triangle,
                                      (triangle[1], triangle[2], triangle[0])):
                    edges.setdefault(tuple(sorted((start, end))), []).append(
                        (start, end))
            manifold = manifold and bool(planes) and all(
                len(uses) == 2 and uses[0] == tuple(reversed(uses[1]))
                for uses in edges.values())
        if not manifold:
            failures.append(Failure(base + ".rigid_geometry_triangle_indices",
                                    "rigid geometry is not a closed consistently oriented manifold"))

        initial = snapshot(actual.get("initial"), base + ".initial")
        final = snapshot(actual.get("final"), base + ".final")
        if initial is None or final is None:
            continue
        initial_position = _vector(definition.get("particle_initial_position_m"),
                                   "$.profile.particle_initial_position_m",
                                   failures)
        initial_velocity = _vector(definition.get("particle_initial_velocity_m_s"),
                                   "$.profile.particle_initial_velocity_m_s",
                                   failures)
        rigid_initial = _vector(definition.get("rigid_initial_position_m"),
                                "$.profile.rigid_initial_position_m", failures)
        rigid_initial_velocity = _vector(
            definition.get("rigid_initial_linear_velocity_m_s"),
            "$.profile.rigid_initial_linear_velocity_m_s", failures)
        rigid_initial_angular = _vector(
            definition.get("rigid_initial_angular_velocity_rad_s"),
            "$.profile.rigid_initial_angular_velocity_rad_s", failures)
        expected_initial_orientation = _quaternion(
            definition.get("rigid_initial_orientation_xyzw"),
            "$.profile.rigid_initial_orientation_xyzw", failures)
        if (initial_position is not None and max(abs(a - b) for a, b in zip(
                initial["fluid_position"], initial_position)) > state_tolerance):
            failures.append(Failure(base + ".initial.fluid.position_m",
                                    "initial fluid state differs from profile",
                                    list(initial["fluid_position"]),
                                    list(initial_position)))
        if (initial_velocity is not None and max(abs(a - b) for a, b in zip(
                initial["fluid_velocity"], initial_velocity)) > state_tolerance):
            failures.append(Failure(base + ".initial.fluid.velocity_m_s",
                                    "initial fluid state differs from profile",
                                    list(initial["fluid_velocity"]),
                                    list(initial_velocity)))
        if (rigid_initial is not None and max(abs(a - b) for a, b in zip(
                initial["rigid_position"], rigid_initial)) > state_tolerance):
            failures.append(Failure(base + ".initial.rigid.position_m",
                                    "initial rigid state differs from profile",
                                    list(initial["rigid_position"]),
                                    list(rigid_initial)))
        if (rigid_initial_velocity is not None and max(abs(a - b)
                for a, b in zip(initial["rigid_velocity"],
                                rigid_initial_velocity)) > state_tolerance):
            failures.append(Failure(
                base + ".initial.rigid.linear_velocity_m_s",
                "initial rigid state differs from profile",
                list(initial["rigid_velocity"]), list(rigid_initial_velocity)))
        if (rigid_initial_angular is not None and max(abs(a - b)
                for a, b in zip(initial["angular_velocity"],
                                rigid_initial_angular)) > state_tolerance):
            failures.append(Failure(
                base + ".initial.rigid.angular_velocity_rad_s",
                "initial rigid state differs from profile",
                list(initial["angular_velocity"]), list(rigid_initial_angular)))
        if (expected_initial_orientation is not None and max(abs(a - b)
                for a, b in zip(initial["orientation"],
                                expected_initial_orientation)) > state_tolerance):
            failures.append(Failure(
                base + ".initial.rigid.orientation_xyzw",
                "initial rigid state differs from profile",
                list(initial["orientation"]),
                list(expected_initial_orientation)))

        particle_radius = float(definition.get("fluid_particle_radius_m", 0.0))
        signed_distance = None
        penetration = None
        leaked_count = 0
        active_normal = (1.0, 0.0, 0.0)
        if manifold and planes:
            inverse = tuple(-component for component in final["orientation"][:3]) + (
                final["orientation"][3],)
            local_particle = rotate(_subtract(final["fluid_position"],
                                              final["rigid_position"]), inverse)
            distances = [(dot(_subtract(local_particle, point), normal), normal)
                         for point, normal in planes]
            signed_distance, local_normal = max(distances, key=lambda item: item[0])
            active_normal = rotate(local_normal, final["orientation"])
            penetration = max(0.0, signed_distance + particle_radius)
            penetration_limit = float(tolerances.get(
                "maximum_containment_penetration_m", 0.0))
            leaked_count = int(penetration > penetration_limit)
            if penetration > penetration_limit:
                failures.append(Failure(base + ".final.fluid.position_m",
                                        "particle leaked outside the closed rigid geometry",
                                        penetration, penetration_limit))
            leak_limit = int(tolerances.get("maximum_leaked_particles", 0))
            if leaked_count > leak_limit:
                failures.append(Failure(base + ".final.fluid.position_m",
                                        "leaked particle count exceeds profile",
                                        leaked_count, leak_limit))

        fluid_mass = float(definition.get("fluid_particle_mass_kg", 0.0))
        body_mass = float(definition.get("rigid_mass_kg", 0.0))
        fluid_change = scale(_subtract(final["fluid_velocity"],
                                       initial["fluid_velocity"]), fluid_mass)
        rigid_change = scale(_subtract(final["rigid_velocity"],
                                       initial["rigid_velocity"]), body_mass)
        residual = _norm(add(fluid_change, rigid_change))
        residual_limit = float(tolerances.get(
            "maximum_reciprocal_momentum_residual_kg_m_s", 0.0))
        if residual > residual_limit:
            failures.append(Failure(base + ".final",
                                    "fluid and rigid momentum changes are not reciprocal",
                                    residual, residual_limit))
        reaction = min(_norm(fluid_change), _norm(rigid_change))
        reaction_minimum = float(tolerances.get(
            "minimum_reaction_impulse_n_s", 0.0))
        if reaction < reaction_minimum:
            failures.append(Failure(base + ".final",
                                    "reciprocal reaction is too small",
                                    reaction, reaction_minimum))
        normal_impulse = dot(fluid_change, active_normal)
        tangential = _norm(_subtract(fluid_change,
                                    scale(active_normal, normal_impulse)))
        tangent_limit = float(tolerances.get(
            "maximum_tangential_momentum_kg_m_s", 0.0))
        if tangential > tangent_limit:
            failures.append(Failure(base + ".final.fluid.velocity_m_s",
                                    "containment introduced tangential momentum",
                                    tangential, tangent_limit))
        angular_speed = _norm(final["angular_velocity"])
        angular_limit = float(tolerances.get(
            "maximum_angular_speed_rad_s", 0.0))
        if angular_speed > angular_limit:
            failures.append(Failure(base + ".final.rigid.angular_velocity_rad_s",
                                    "containment introduced rigid angular motion",
                                    angular_speed, angular_limit))
        rigid_translation = _norm(_subtract(final["rigid_position"],
                                            initial["rigid_position"]))
        translation_limit = float(tolerances.get(
            "maximum_rigid_translation_m", 0.0))
        if rigid_translation > translation_limit:
            failures.append(Failure(base + ".final.rigid.position_m",
                                    "rigid translation exceeds one-step bound",
                                    rigid_translation, translation_limit))

        inertia_value = definition.get("rigid_inertia_diagonal_kg_m2")
        inertia = _vector(inertia_value, "$.profile.rigid_inertia_diagonal_kg_m2",
                          failures)
        energy_increase = None
        if inertia is not None:
            def kinetic(state: dict[str, Any]) -> float:
                inverse = tuple(-component for component in state["orientation"][:3]) + (
                    state["orientation"][3],)
                local_angular = rotate(state["angular_velocity"], inverse)
                return (0.5 * fluid_mass * dot(state["fluid_velocity"],
                                               state["fluid_velocity"]) +
                        0.5 * body_mass * dot(state["rigid_velocity"],
                                              state["rigid_velocity"]) +
                        0.5 * sum(moment * speed * speed for moment, speed in zip(
                            inertia, local_angular)))
            energy_increase = kinetic(final) - kinetic(initial)
            energy_limit = float(tolerances.get("maximum_energy_increase_j", 0.0))
            if energy_increase > energy_limit:
                failures.append(Failure(base + ".final",
                                        "containment increased endpoint kinetic energy",
                                        energy_increase, energy_limit))
        observations[identity] = {
            "closed_consistently_oriented_manifold": manifold,
            "final_particle_signed_distance_m": signed_distance,
            "maximum_containment_penetration_m": penetration,
            "leaked_particle_count": leaked_count,
            "reaction_impulse_n_s": reaction,
            "reciprocal_momentum_residual_kg_m_s": residual,
            "tangential_momentum_kg_m_s": tangential,
            "rigid_angular_speed_rad_s": angular_speed,
            "rigid_translation_m": rigid_translation,
            "kinetic_energy_change_j": energy_increase,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_soft_rope_attachment(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def close_number(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        reference = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if value is not None and reference is not None and abs(
                value - reference) > state_tolerance:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(actual, path, failures)
        reference = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if value is not None and reference is not None and _norm(
                _subtract(value, reference)) > state_tolerance:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def vector_array(actual: Any, expected: Any, path: str) -> list[
            tuple[float, float, float]]:
        if not isinstance(actual, list) or not isinstance(expected, list) or len(
                actual) != len(expected):
            failures.append(Failure(path, "fixture vector array differs from profile",
                                    actual, expected))
            return []
        parsed = []
        for index, (value, reference) in enumerate(zip(actual, expected)):
            close_vector(value, reference, f"{path}[{index}]")
            vector = _vector(value, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    def scalar_array(actual: Any, expected: Any, path: str) -> list[float]:
        if not isinstance(actual, list) or not isinstance(expected, list) or len(
                actual) != len(expected):
            failures.append(Failure(path, "fixture scalar array differs from profile",
                                    actual, expected))
            return []
        parsed = []
        for index, (value, reference) in enumerate(zip(actual, expected)):
            item_path = f"{path}[{index}]"
            close_number(value, reference, item_path)
            scalar = _scalar(value, item_path, failures)
            if scalar is not None:
                parsed.append(scalar)
        return parsed

    def snapshot(value: Any, path: str, soft_count: int,
                 rope_count: int) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        soft = value.get("soft_body")
        surface = value.get("soft_body_surface_m")
        rope = value.get("rope")
        if not isinstance(soft, list) or len(soft) != soft_count:
            failures.append(Failure(path + ".soft_body",
                                    "soft-body state count differs from profile",
                                    len(soft) if isinstance(soft, list) else soft,
                                    soft_count))
            return None
        if not isinstance(surface, list) or len(surface) != soft_count:
            failures.append(Failure(path + ".soft_body_surface_m",
                                    "soft-body surface count differs from profile",
                                    len(surface) if isinstance(surface, list) else surface,
                                    soft_count))
            return None
        if not isinstance(rope, list) or len(rope) != rope_count:
            failures.append(Failure(path + ".rope",
                                    "rope state count differs from profile",
                                    len(rope) if isinstance(rope, list) else rope,
                                    rope_count))
            return None

        def states(items: list[Any], base: str,
                   force_name: str) -> list[dict[str, Any]]:
            parsed = []
            for index, item in enumerate(items):
                item_path = f"{base}[{index}]"
                if not isinstance(item, dict):
                    failures.append(Failure(item_path,
                                            "expected a state object", item))
                    continue
                if item.get("index") != index:
                    failures.append(Failure(item_path + ".index",
                                            "public state order changed",
                                            item.get("index"), index))
                position = _vector(item.get("position_m"),
                                   item_path + ".position_m", failures)
                velocity = _vector(item.get("velocity_m_s"),
                                   item_path + ".velocity_m_s", failures)
                force = _vector(item.get(force_name),
                                item_path + "." + force_name, failures)
                if position is not None and velocity is not None and force is not None:
                    parsed.append({"position": position, "velocity": velocity,
                                   "force": force})
            return parsed

        soft_states = states(soft, path + ".soft_body",
                             "rope_reaction_force_n")
        rope_states = states(rope, path + ".rope",
                             "soft_body_contact_force_n")
        surface_states = []
        for index, item in enumerate(surface):
            vector = _vector(item, f"{path}.soft_body_surface_m[{index}]",
                             failures)
            if vector is not None:
                surface_states.append(vector)
        if (len(soft_states) != soft_count or len(rope_states) != rope_count or
                len(surface_states) != soft_count):
            return None
        return {"soft": soft_states, "surface": surface_states,
                "rope": rope_states}

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "step_count", "substeps",
                      "surface_triangle_indices", "soft_solver_iterations",
                      "rope_solver_iterations", "attached_rope_end",
                      "attached_surface_triangle", "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        actual_bonds = actual.get("soft_bonds")
        expected_bonds = definition.get("soft_bonds")
        if (not isinstance(actual_bonds, list) or
                not isinstance(expected_bonds, list) or
                len(actual_bonds) != len(expected_bonds)):
            failures.append(Failure(base + ".soft_bonds",
                                    "fixture bond array differs from profile",
                                    actual_bonds, expected_bonds))
        else:
            for index, (bond, reference) in enumerate(zip(
                    actual_bonds, expected_bonds)):
                path = f"{base}.soft_bonds[{index}]"
                if (not isinstance(bond, list) or len(bond) != 3 or
                        not isinstance(reference, list) or len(reference) != 3 or
                        bond[:2] != reference[:2]):
                    failures.append(Failure(path,
                                            "fixture bond differs from profile",
                                            bond, reference))
                    continue
                close_number(bond[2], reference[2], path + "[2]")
        for field in ("timestep_seconds", "soft_node_mass_kg",
                      "soft_node_radius_m", "soft_stretch_compliance_m_per_n",
                      "soft_velocity_damping_per_s", "rope_node_spacing_m",
                      "rope_radius_m", "rope_total_mass_kg",
                      "rope_stretch_compliance_m_per_n",
                      "rope_velocity_damping_per_s",
                      "anchor_support_radius_scale",
                      "anchor_contact_support_radius_scale",
                      "maximum_soft_body_acceleration_m_s2"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        close_vector(actual.get("gravity_m_s2"),
                     definition.get("gravity_m_s2"),
                     base + ".gravity_m_s2")
        soft_nodes = vector_array(
            actual.get("soft_nodes_m"), definition.get("soft_nodes_m"),
            base + ".soft_nodes_m")
        inverse_masses = scalar_array(
            actual.get("soft_inverse_masses_per_kg"),
            definition.get("soft_inverse_masses_per_kg"),
            base + ".soft_inverse_masses_per_kg")
        rope_centerline = vector_array(
            actual.get("rope_centerline_m"),
            definition.get("rope_centerline_m"),
            base + ".rope_centerline_m")
        spacing = _scalar(definition.get("rope_node_spacing_m"),
                          base + ".rope_node_spacing_m", failures)
        rope_count = 0
        if len(rope_centerline) == 2 and spacing is not None and spacing > 0.0:
            rope_count = round(_norm(_subtract(
                rope_centerline[1], rope_centerline[0])) / spacing) + 1
        initial = snapshot(actual.get("initial"), base + ".initial",
                           len(soft_nodes), rope_count)
        previous = snapshot(actual.get("previous"), base + ".previous",
                            len(soft_nodes), rope_count)
        final = snapshot(actual.get("final"), base + ".final",
                         len(soft_nodes), rope_count)
        if (initial is None or previous is None or final is None or
                not soft_nodes or rope_count < 2):
            continue

        for index, expected in enumerate(soft_nodes):
            for state_name, position in (
                    ("node", initial["soft"][index]["position"]),
                    ("surface", initial["surface"][index])):
                if _norm(_subtract(position, expected)) > state_tolerance:
                    failures.append(Failure(
                        f"{base}.initial.soft_body_{state_name}[{index}]",
                        "initial soft-body state differs from profile",
                        list(position), list(expected)))
            if _norm(initial["soft"][index]["velocity"]) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.soft_body[{index}].velocity_m_s",
                    "initial soft-body velocity must be zero",
                    list(initial["soft"][index]["velocity"]), 0.0))
        initial_rope_positions = []
        for index in range(rope_count):
            fraction = index / (rope_count - 1)
            expected = tuple(rope_centerline[0][axis] + fraction * (
                rope_centerline[1][axis] - rope_centerline[0][axis])
                             for axis in range(3))
            initial_rope_positions.append(expected)
            position = initial["rope"][index]["position"]
            if _norm(_subtract(position, expected)) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.rope[{index}].position_m",
                    "initial rope state differs from the public centerline",
                    list(position), list(expected)))
            if _norm(initial["rope"][index]["velocity"]) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.rope[{index}].velocity_m_s",
                    "initial rope velocity must be zero",
                    list(initial["rope"][index]["velocity"]), 0.0))

        attached_triangle = definition.get("attached_surface_triangle")
        if (not isinstance(attached_triangle, list) or
                len(attached_triangle) != 3 or
                not all(isinstance(index, int) and not isinstance(index, bool) and
                        0 <= index < len(final["surface"])
                        for index in attached_triangle)):
            failures.append(Failure(base + ".attached_surface_triangle",
                                    "invalid attached surface triangle",
                                    attached_triangle))
            continue
        initial_anchor = tuple(value / 3.0 for value in _sum(
            [initial["surface"][index] for index in attached_triangle]))
        final_anchor = tuple(value / 3.0 for value in _sum(
            [final["surface"][index] for index in attached_triangle]))
        endpoint = 0 if definition.get("attached_rope_end") == "first" else rope_count - 1
        initial_drift = _norm(_subtract(
            initial_anchor,
            initial["rope"][endpoint]["position"]))
        final_drift = _norm(_subtract(
            final_anchor,
            final["rope"][endpoint]["position"]))
        drift_limit = float(tolerances.get("maximum_attachment_drift_m", 0.0))
        if initial_drift > state_tolerance:
            failures.append(Failure(base + ".initial",
                                    "attachment endpoints do not start coincident",
                                    initial_drift, state_tolerance))
        if final_drift > drift_limit:
            failures.append(Failure(base + ".final",
                                    "soft-body-rope attachment drift exceeds profile",
                                    final_drift, drift_limit))

        pinned_drift = 0.0
        for index, inverse_mass in enumerate(inverse_masses):
            if inverse_mass == 0.0:
                pinned_drift = max(pinned_drift, _norm(_subtract(
                    final["soft"][index]["position"], soft_nodes[index])))
        pinned_limit = float(tolerances.get("maximum_pinned_node_drift_m", 0.0))
        if pinned_drift > pinned_limit:
            failures.append(Failure(base + ".final.soft_body",
                                    "pinned soft-body support moved",
                                    pinned_drift, pinned_limit))

        soft_reaction = _sum([state["force"] for state in final["soft"]])
        rope_mass = float(definition.get("rope_total_mass_kg", 0.0))
        frame_dt = float(definition.get("timestep_seconds", 0.0))
        gravity = _vector(definition.get("gravity_m_s2"),
                          base + ".gravity_m_s2", failures)
        node_mass = rope_mass / rope_count
        dynamic_indices = [index for index in range(rope_count)
                           if index != endpoint]
        dynamic_mass = node_mass * len(dynamic_indices)
        previous_momentum = tuple(node_mass * value for value in _sum(
            [previous["rope"][index]["velocity"]
             for index in dynamic_indices]))
        final_momentum = tuple(node_mass * value for value in _sum(
            [final["rope"][index]["velocity"]
             for index in dynamic_indices]))
        if gravity is None or frame_dt <= 0.0:
            continue
        rope_reaction = tuple(
            (final_momentum[axis] - previous_momentum[axis]) / frame_dt -
            dynamic_mass * gravity[axis] for axis in range(3))
        reaction_activity = max(_norm(soft_reaction), _norm(rope_reaction))
        reaction_residual = _norm(add(soft_reaction, rope_reaction))
        activity_limit = float(tolerances.get("minimum_reaction_force_n", 0.0))
        residual_limit = float(tolerances.get(
            "maximum_reciprocal_force_residual_n", 0.0))
        if reaction_activity < activity_limit:
            failures.append(Failure(base + ".final",
                                    "attachment reaction is too small",
                                    reaction_activity, activity_limit))
        if reaction_residual > residual_limit:
            failures.append(Failure(base + ".final",
                                    "soft body and rope reactions are not reciprocal",
                                    reaction_residual, residual_limit))

        maximum_contact_force = max(
            _norm(state["force"]) for state in final["rope"])
        contact_limit = float(tolerances.get("maximum_contact_force_n", 0.0))
        if maximum_contact_force > contact_limit:
            failures.append(Failure(base + ".final.rope",
                                    "contact channel is active in attachment fixture",
                                    maximum_contact_force, contact_limit))

        transverse = 0.0
        for index, state in enumerate(final["soft"]):
            displacement = _subtract(state["position"], soft_nodes[index])
            transverse = max(transverse, abs(displacement[0]),
                             abs(displacement[2]))
        for index, state in enumerate(final["rope"]):
            displacement = _subtract(state["position"],
                                     initial_rope_positions[index])
            transverse = max(transverse, abs(displacement[0]),
                             abs(displacement[2]))
        transverse_limit = float(tolerances.get(
            "maximum_transverse_drift_m", 0.0))
        if transverse > transverse_limit:
            failures.append(Failure(base + ".final",
                                    "attachment introduced transverse drift",
                                    transverse, transverse_limit))
        maximum_speed = max(_norm(state["velocity"])
                            for state in final["soft"] + final["rope"])
        speed_limit = float(tolerances.get("maximum_speed_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure(base + ".final",
                                    "endpoint speed exceeds profile",
                                    maximum_speed, speed_limit))

        observations[identity] = {
            "initial_attachment_drift_m": initial_drift,
            "final_attachment_drift_m": final_drift,
            "soft_body_reaction_force_n": list(soft_reaction),
            "rope_reaction_force_n": list(rope_reaction),
            "dynamic_rope_mass_kg": dynamic_mass,
            "reaction_activity_n": reaction_activity,
            "reciprocal_force_residual_n": reaction_residual,
            "maximum_pinned_node_drift_m": pinned_drift,
            "maximum_contact_force_n": maximum_contact_force,
            "maximum_transverse_drift_m": transverse,
            "maximum_speed_m_s": maximum_speed,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_cloth_rope_attachment(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if not isinstance(fixture, dict) or not isinstance(
                fixture.get("id"), str):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def close_number(actual: Any, expected: Any, path: str) -> None:
        value = _scalar(actual, path, failures)
        reference = _scalar(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if value is not None and reference is not None and abs(
                value - reference) > state_tolerance:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    def close_vector(actual: Any, expected: Any, path: str) -> None:
        value = _vector(actual, path, failures)
        reference = _vector(expected, path.replace(
            "$.fixtures", "$.profile.fixtures"), failures)
        if value is not None and reference is not None and _norm(
                _subtract(value, reference)) > state_tolerance:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    list(value), list(reference)))

    def vector_array(actual: Any, expected: Any, path: str) -> list[
            tuple[float, float, float]]:
        if not isinstance(actual, list) or not isinstance(expected, list) or len(
                actual) != len(expected):
            failures.append(Failure(path, "fixture vector array differs from profile",
                                    actual, expected))
            return []
        parsed = []
        for index, (value, reference) in enumerate(zip(actual, expected)):
            close_vector(value, reference, f"{path}[{index}]")
            vector = _vector(value, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    def scalar_array(actual: Any, expected: Any, path: str) -> list[float]:
        if not isinstance(actual, list) or not isinstance(expected, list) or len(
                actual) != len(expected):
            failures.append(Failure(path, "fixture scalar array differs from profile",
                                    actual, expected))
            return []
        parsed = []
        for index, (value, reference) in enumerate(zip(actual, expected)):
            item_path = f"{path}[{index}]"
            close_number(value, reference, item_path)
            scalar = _scalar(value, item_path, failures)
            if scalar is not None:
                parsed.append(scalar)
        return parsed

    def snapshot(value: Any, path: str, cloth_count: int,
                 rope_count: int) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        cloth = value.get("cloth")
        rope = value.get("rope")
        if not isinstance(cloth, list) or len(cloth) != cloth_count:
            failures.append(Failure(path + ".cloth",
                                    "cloth state count differs from profile",
                                    len(cloth) if isinstance(cloth, list) else cloth,
                                    cloth_count))
            return None
        if not isinstance(rope, list) or len(rope) != rope_count:
            failures.append(Failure(path + ".rope",
                                    "rope state count differs from profile",
                                    len(rope) if isinstance(rope, list) else rope,
                                    rope_count))
            return None

        def states(items: list[Any], base: str,
                   force_name: str) -> list[dict[str, Any]]:
            parsed = []
            for index, item in enumerate(items):
                item_path = f"{base}[{index}]"
                if not isinstance(item, dict):
                    failures.append(Failure(item_path,
                                            "expected a state object", item))
                    continue
                if item.get("index") != index:
                    failures.append(Failure(item_path + ".index",
                                            "public state order changed",
                                            item.get("index"), index))
                position = _vector(item.get("position_m"),
                                   item_path + ".position_m", failures)
                velocity = _vector(item.get("velocity_m_s"),
                                   item_path + ".velocity_m_s", failures)
                force = _vector(item.get(force_name),
                                item_path + "." + force_name, failures)
                if position is not None and velocity is not None and force is not None:
                    parsed.append({"position": position, "velocity": velocity,
                                   "force": force})
            return parsed

        cloth_states = states(cloth, path + ".cloth",
                              "rope_reaction_force_n")
        rope_states = states(rope, path + ".rope", "constraint_force_n")
        if len(cloth_states) != cloth_count or len(rope_states) != rope_count:
            return None
        return {"cloth": cloth_states, "rope": rope_states}

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        for field in ("id", "step_count", "substeps",
                      "cloth_triangle_indices", "cloth_solver_iterations",
                      "rope_solver_iterations", "attached_rope_end",
                      "attached_cloth_vertex", "oracle"):
            if actual.get(field) != definition.get(field):
                failures.append(Failure(base + "." + field,
                                        "fixture setting differs from profile",
                                        actual.get(field), definition.get(field)))
        for field in ("timestep_seconds", "cloth_vertex_mass_kg",
                      "cloth_stretch_compliance_m_per_n",
                      "cloth_velocity_damping_per_s",
                      "rope_node_spacing_m", "rope_radius_m",
                      "rope_total_mass_kg",
                      "rope_stretch_compliance_m_per_n",
                      "rope_velocity_damping_per_s",
                      "anchor_effective_mass_kg",
                      "maximum_cloth_acceleration_m_s2"):
            close_number(actual.get(field), definition.get(field),
                         base + "." + field)
        close_vector(actual.get("gravity_m_s2"),
                     definition.get("gravity_m_s2"),
                     base + ".gravity_m_s2")
        cloth_vertices = vector_array(
            actual.get("cloth_vertices_m"), definition.get("cloth_vertices_m"),
            base + ".cloth_vertices_m")
        inverse_masses = scalar_array(
            actual.get("cloth_inverse_masses_per_kg"),
            definition.get("cloth_inverse_masses_per_kg"),
            base + ".cloth_inverse_masses_per_kg")
        rope_centerline = vector_array(
            actual.get("rope_centerline_m"),
            definition.get("rope_centerline_m"),
            base + ".rope_centerline_m")
        spacing = _scalar(definition.get("rope_node_spacing_m"),
                          base + ".rope_node_spacing_m", failures)
        rope_count = 0
        if len(rope_centerline) == 2 and spacing is not None and spacing > 0.0:
            rope_count = round(_norm(_subtract(
                rope_centerline[1], rope_centerline[0])) / spacing) + 1
        initial = snapshot(actual.get("initial"), base + ".initial",
                           len(cloth_vertices), rope_count)
        previous = snapshot(actual.get("previous"), base + ".previous",
                            len(cloth_vertices), rope_count)
        final = snapshot(actual.get("final"), base + ".final",
                         len(cloth_vertices), rope_count)
        if (initial is None or previous is None or final is None or
                not cloth_vertices or rope_count < 2):
            continue

        for index, expected in enumerate(cloth_vertices):
            position = initial["cloth"][index]["position"]
            if _norm(_subtract(position, expected)) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.cloth[{index}].position_m",
                    "initial cloth state differs from profile",
                    list(position), list(expected)))
            if _norm(initial["cloth"][index]["velocity"]) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.cloth[{index}].velocity_m_s",
                    "initial cloth velocity must be zero",
                    list(initial["cloth"][index]["velocity"]), 0.0))
        for index in range(rope_count):
            fraction = index / (rope_count - 1)
            expected = tuple(rope_centerline[0][axis] + fraction * (
                rope_centerline[1][axis] - rope_centerline[0][axis])
                             for axis in range(3))
            position = initial["rope"][index]["position"]
            if _norm(_subtract(position, expected)) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.rope[{index}].position_m",
                    "initial rope state differs from the public centerline",
                    list(position), list(expected)))
            if _norm(initial["rope"][index]["velocity"]) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.rope[{index}].velocity_m_s",
                    "initial rope velocity must be zero",
                    list(initial["rope"][index]["velocity"]), 0.0))

        attached_vertex = definition.get("attached_cloth_vertex")
        if not isinstance(attached_vertex, int) or isinstance(
                attached_vertex, bool) or attached_vertex < 0 or attached_vertex >= len(
                    final["cloth"]):
            failures.append(Failure(base + ".attached_cloth_vertex",
                                    "invalid attached cloth vertex",
                                    attached_vertex))
            continue
        endpoint = 0 if definition.get("attached_rope_end") == "first" else rope_count - 1
        initial_drift = _norm(_subtract(
            initial["cloth"][attached_vertex]["position"],
            initial["rope"][endpoint]["position"]))
        final_drift = _norm(_subtract(
            final["cloth"][attached_vertex]["position"],
            final["rope"][endpoint]["position"]))
        drift_limit = float(tolerances.get("maximum_attachment_drift_m", 0.0))
        if initial_drift > state_tolerance:
            failures.append(Failure(base + ".initial",
                                    "attachment endpoints do not start coincident",
                                    initial_drift, state_tolerance))
        if final_drift > drift_limit:
            failures.append(Failure(base + ".final",
                                    "cloth-rope attachment drift exceeds profile",
                                    final_drift, drift_limit))

        pinned_drift = 0.0
        for index, inverse_mass in enumerate(inverse_masses):
            if inverse_mass == 0.0:
                pinned_drift = max(pinned_drift, _norm(_subtract(
                    final["cloth"][index]["position"], cloth_vertices[index])))
        pinned_limit = float(tolerances.get("maximum_pinned_vertex_drift_m", 0.0))
        if pinned_drift > pinned_limit:
            failures.append(Failure(base + ".final.cloth",
                                    "pinned cloth support moved",
                                    pinned_drift, pinned_limit))

        cloth_reaction = _sum([state["force"] for state in final["cloth"]])
        rope_mass = float(definition.get("rope_total_mass_kg", 0.0))
        frame_dt = float(definition.get("timestep_seconds", 0.0))
        gravity = _vector(definition.get("gravity_m_s2"),
                          base + ".gravity_m_s2", failures)
        node_mass = rope_mass / rope_count
        dynamic_indices = [index for index in range(rope_count)
                           if index != endpoint]
        dynamic_mass = node_mass * len(dynamic_indices)
        previous_momentum = tuple(node_mass * value for value in _sum(
            [previous["rope"][index]["velocity"]
             for index in dynamic_indices]))
        final_momentum = tuple(node_mass * value for value in _sum(
            [final["rope"][index]["velocity"]
             for index in dynamic_indices]))
        if gravity is None or frame_dt <= 0.0:
            continue
        rope_reaction = tuple(
            (final_momentum[axis] - previous_momentum[axis]) / frame_dt -
            dynamic_mass * gravity[axis] for axis in range(3))
        reaction_activity = max(_norm(cloth_reaction), _norm(rope_reaction))
        reaction_residual = _norm(add(cloth_reaction, rope_reaction))
        activity_limit = float(tolerances.get("minimum_reaction_force_n", 0.0))
        residual_limit = float(tolerances.get(
            "maximum_reciprocal_force_residual_n", 0.0))
        if reaction_activity < activity_limit:
            failures.append(Failure(base + ".final",
                                    "attachment reaction is too small",
                                    reaction_activity, activity_limit))
        if reaction_residual > residual_limit:
            failures.append(Failure(base + ".final",
                                    "cloth and rope reactions are not reciprocal",
                                    reaction_residual, residual_limit))

        transverse = max(abs(state["position"][2])
                         for state in final["cloth"] + final["rope"])
        transverse_limit = float(tolerances.get(
            "maximum_transverse_drift_m", 0.0))
        if transverse > transverse_limit:
            failures.append(Failure(base + ".final",
                                    "attachment introduced transverse drift",
                                    transverse, transverse_limit))
        maximum_speed = max(_norm(state["velocity"])
                            for state in final["cloth"] + final["rope"])
        speed_limit = float(tolerances.get("maximum_speed_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure(base + ".final",
                                    "endpoint speed exceeds profile",
                                    maximum_speed, speed_limit))

        observations[identity] = {
            "initial_attachment_drift_m": initial_drift,
            "final_attachment_drift_m": final_drift,
            "cloth_reaction_force_n": list(cloth_reaction),
            "rope_reaction_force_n": list(rope_reaction),
            "dynamic_rope_mass_kg": dynamic_mass,
            "reaction_activity_n": reaction_activity,
            "reciprocal_force_residual_n": reaction_residual,
            "maximum_pinned_vertex_drift_m": pinned_drift,
            "maximum_transverse_drift_m": transverse,
            "maximum_speed_m_s": maximum_speed,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_rope_rigid_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definitions = {fixture["id"]: fixture
                   for fixture in profile.get("fixtures", [])
                   if isinstance(fixture, dict) and
                   isinstance(fixture.get("id"), str)}
    actual_fixtures = result.get("fixtures")
    if not isinstance(actual_fixtures, list):
        failures.append(Failure("$.fixtures", "expected a fixture array",
                                actual_fixtures))
        actual_fixtures = []
    actual_by_id: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(actual_fixtures):
        if (not isinstance(fixture, dict) or
                not isinstance(fixture.get("id"), str)):
            failures.append(Failure(f"$.fixtures[{index}]",
                                    "fixture needs a string id", fixture))
            continue
        identity = fixture["id"]
        if identity in actual_by_id:
            failures.append(Failure(f"$.fixtures[{index}].id",
                                    "duplicate fixture id", identity))
            continue
        actual_by_id[identity] = fixture
    for missing in sorted(set(definitions) - set(actual_by_id)):
        failures.append(Failure("$.fixtures", "missing required fixture", missing))
    for extra in sorted(set(actual_by_id) - set(definitions)):
        failures.append(Failure("$.fixtures", "unexpected fixture", extra))

    tolerances = profile.get("tolerances", {})
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    Vector = tuple[float, float, float]
    Quaternion = tuple[float, float, float, float]

    def add(left: Vector, right: Vector) -> Vector:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def scale(value: Vector, factor: float) -> Vector:
        return tuple(component * factor for component in value)  # type: ignore[return-value]

    def dot(left: Vector, right: Vector) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(left: Vector, right: Vector) -> Vector:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def quaternion_distance(left: Quaternion, right: Quaternion) -> float:
        direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
        negated = math.sqrt(sum((a + b) ** 2 for a, b in zip(left, right)))
        return min(direct, negated)

    def normalized_quaternion(value: Quaternion, path: str) -> Quaternion:
        length = math.sqrt(sum(component * component for component in value))
        if abs(length - 1.0) > state_tolerance:
            failures.append(Failure(path,
                                    "rigid orientation is not normalized",
                                    length, 1.0))
        if length <= 1.0e-30:
            return (0.0, 0.0, 0.0, 1.0)
        return tuple(component / length for component in value)  # type: ignore[return-value]

    def rotate(value: Vector, quaternion: Quaternion) -> Vector:
        imaginary = quaternion[0], quaternion[1], quaternion[2]
        twice_cross = scale(cross(imaginary, value), 2.0)
        return add(value, add(scale(twice_cross, quaternion[3]),
                              cross(imaginary, twice_cross)))

    def close_contract(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
                return
            for key, wanted in expected.items():
                close_contract(actual.get(key), wanted, path + "." + key)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
                return
            for index, (item, wanted) in enumerate(zip(actual, expected)):
                close_contract(item, wanted, f"{path}[{index}]")
        elif _number(expected):
            if (not _number(actual) or
                    abs(float(actual) - float(expected)) > state_tolerance):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        actual, expected))
        elif actual != expected:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    actual, expected))

    def snapshot(value: Any, path: str,
                 node_count: int) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return None
        raw_rope = value.get("rope")
        raw_rigid = value.get("rigid")
        if not isinstance(raw_rope, list) or len(raw_rope) != node_count:
            failures.append(Failure(path + ".rope",
                                    "rope state count differs from profile",
                                    raw_rope, node_count))
            return None
        nodes: list[dict[str, Any]] = []
        for index, node in enumerate(raw_rope):
            item_path = f"{path}.rope[{index}]"
            if not isinstance(node, dict) or node.get("index") != index:
                failures.append(Failure(item_path,
                                        "rope node order/index changed", node,
                                        index))
                continue
            position = _vector(node.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(node.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            force = _vector(node.get("rigid_contact_force_n"),
                            item_path + ".rigid_contact_force_n", failures)
            if position is not None and velocity is not None and force is not None:
                nodes.append({"position": position, "velocity": velocity,
                              "force": force})
        if not isinstance(raw_rigid, dict):
            failures.append(Failure(path + ".rigid",
                                    "expected a rigid state object", raw_rigid))
            return None
        rigid_position = _vector(raw_rigid.get("position_m"),
                                 path + ".rigid.position_m", failures)
        rigid_orientation = _quaternion(
            raw_rigid.get("orientation_xyzw"),
            path + ".rigid.orientation_xyzw", failures)
        rigid_linear = _vector(raw_rigid.get("linear_velocity_m_s"),
                               path + ".rigid.linear_velocity_m_s", failures)
        rigid_angular = _vector(raw_rigid.get("angular_velocity_rad_s"),
                                path + ".rigid.angular_velocity_rad_s", failures)
        if (len(nodes) != node_count or rigid_position is None or
                rigid_orientation is None or rigid_linear is None or
                rigid_angular is None):
            return None
        rigid_orientation = normalized_quaternion(
            rigid_orientation, path + ".rigid.orientation_xyzw")
        return {"nodes": nodes, "rigid_position": rigid_position,
                "rigid_orientation": rigid_orientation,
                "rigid_linear": rigid_linear,
                "rigid_angular": rigid_angular}

    def point_triangle_distance(point: Vector, first: Vector, second: Vector,
                                third: Vector) -> float:
        ab = _subtract(second, first)
        ac = _subtract(third, first)
        ap = _subtract(point, first)
        d1, d2 = dot(ab, ap), dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return _norm(ap)
        bp = _subtract(point, second)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return _norm(bp)
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            fraction = d1 / (d1 - d3)
            return _norm(_subtract(point, add(first, scale(ab, fraction))))
        cp = _subtract(point, third)
        d5, d6 = dot(ab, cp), dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return _norm(cp)
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            fraction = d2 / (d2 - d6)
            return _norm(_subtract(point, add(first, scale(ac, fraction))))
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
            edge = _subtract(third, second)
            fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return _norm(_subtract(point, add(second, scale(edge, fraction))))
        denominator = va + vb + vc
        if abs(denominator) <= 1.0e-30:
            return min(_norm(ap), _norm(bp), _norm(cp))
        inverse = 1.0 / denominator
        closest = add(first, add(scale(ab, vb * inverse),
                                 scale(ac, vc * inverse)))
        return _norm(_subtract(point, closest))

    def segment_segment_distance(first_a: Vector, last_a: Vector,
                                 first_b: Vector, last_b: Vector) -> float:
        edge_a = _subtract(last_a, first_a)
        edge_b = _subtract(last_b, first_b)
        offset = _subtract(first_a, first_b)
        a = dot(edge_a, edge_a)
        e = dot(edge_b, edge_b)
        f = dot(edge_b, offset)
        if a <= 1.0e-30 and e <= 1.0e-30:
            return _norm(offset)
        if a <= 1.0e-30:
            first_fraction = 0.0
            second_fraction = max(0.0, min(1.0, f / e))
        else:
            c = dot(edge_a, offset)
            if e <= 1.0e-30:
                second_fraction = 0.0
                first_fraction = max(0.0, min(1.0, -c / a))
            else:
                b = dot(edge_a, edge_b)
                denominator = a * e - b * b
                first_fraction = (max(0.0, min(1.0, (b * f - c * e) /
                                                    denominator))
                                  if abs(denominator) > 1.0e-30 else 0.0)
                second_fraction = (b * first_fraction + f) / e
                if second_fraction < 0.0:
                    second_fraction = 0.0
                    first_fraction = max(0.0, min(1.0, -c / a))
                elif second_fraction > 1.0:
                    second_fraction = 1.0
                    first_fraction = max(0.0, min(1.0, (b - c) / a))
        closest_a = add(first_a, scale(edge_a, first_fraction))
        closest_b = add(first_b, scale(edge_b, second_fraction))
        return _norm(_subtract(closest_a, closest_b))

    def segment_hits_triangle(first: Vector, last: Vector, a: Vector,
                              b: Vector, c: Vector) -> bool:
        direction = _subtract(last, first)
        first_edge = _subtract(b, a)
        second_edge = _subtract(c, a)
        trial = cross(direction, second_edge)
        determinant = dot(first_edge, trial)
        if abs(determinant) <= 1.0e-12:
            return False
        inverse = 1.0 / determinant
        offset = _subtract(first, a)
        u = inverse * dot(offset, trial)
        if u < 0.0 or u > 1.0:
            return False
        trial2 = cross(offset, first_edge)
        v = inverse * dot(direction, trial2)
        if v < 0.0 or u + v > 1.0:
            return False
        fraction = inverse * dot(second_edge, trial2)
        return 0.0 <= fraction <= 1.0

    def point_inside_convex(point: Vector, vertices: list[Vector],
                            triangles: list[tuple[int, int, int]]) -> bool:
        center = tuple(sum(vertex[axis] for vertex in vertices) / len(vertices)
                       for axis in range(3))
        for first_index, second_index, third_index in triangles:
            first = vertices[first_index]
            normal = cross(_subtract(vertices[second_index], first),
                           _subtract(vertices[third_index], first))
            if dot(normal, _subtract(first, center)) < 0.0:
                normal = scale(normal, -1.0)
            if dot(normal, _subtract(point, first)) > 1.0e-10:
                return False
        return True

    def segment_triangle_distance(first: Vector, last: Vector, a: Vector,
                                  b: Vector, c: Vector) -> float:
        if segment_hits_triangle(first, last, a, b, c):
            return 0.0
        return min(point_triangle_distance(first, a, b, c),
                   point_triangle_distance(last, a, b, c),
                   segment_segment_distance(first, last, a, b),
                   segment_segment_distance(first, last, b, c),
                   segment_segment_distance(first, last, c, a))

    observations: dict[str, Any] = {}
    sample_count = 0
    for identity, definition in definitions.items():
        actual = actual_by_id.get(identity)
        if actual is None:
            continue
        base = f"$.fixtures[{identity}]"
        settings = {key: value for key, value in actual.items()
                    if key not in {"initial", "final"}}
        close_contract(settings, definition, base)

        node_count = definition.get("rope_node_count")
        if (not isinstance(node_count, int) or isinstance(node_count, bool) or
                node_count < 2):
            failures.append(Failure(base + ".rope_node_count",
                                    "quality fixture needs at least two rope nodes",
                                    node_count))
            continue
        initial = snapshot(actual.get("initial"), base + ".initial", node_count)
        final = snapshot(actual.get("final"), base + ".final", node_count)
        if initial is None or final is None:
            continue

        prepared_raw = definition.get("prepared_rope_nodes_m")
        prepared_velocity = _vector(
            definition.get("prepared_rope_velocity_m_s"),
            base + ".prepared_rope_velocity_m_s", failures)
        if (not isinstance(prepared_raw, list) or
                len(prepared_raw) != node_count or prepared_velocity is None):
            failures.append(Failure(base + ".prepared_rope_nodes_m",
                                    "invalid prepared rope state", prepared_raw))
            continue
        prepared = [_vector(item, f"{base}.prepared_rope_nodes_m[{index}]",
                            failures)
                    for index, item in enumerate(prepared_raw)]
        if any(item is None for item in prepared):
            continue
        prepared_nodes = [item for item in prepared if item is not None]
        for index, wanted in enumerate(prepared_nodes):
            node = initial["nodes"][index]
            if _norm(_subtract(node["position"], wanted)) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.rope[{index}].position_m",
                    "prepared rope position differs from profile",
                    list(node["position"]), list(wanted)))
            if _norm(_subtract(node["velocity"], prepared_velocity)) > \
                    state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.rope[{index}].velocity_m_s",
                    "prepared rope velocity differs from profile",
                    list(node["velocity"]), list(prepared_velocity)))
            if _norm(node["force"]) > state_tolerance:
                failures.append(Failure(
                    f"{base}.initial.rope[{index}].rigid_contact_force_n",
                    "initial rope contact force must be zero",
                    list(node["force"]), 0.0))

        initial_rigid_position = _vector(
            definition.get("rigid_initial_position_m"),
            base + ".rigid_initial_position_m", failures)
        initial_rigid_orientation = _quaternion(
            definition.get("rigid_initial_orientation_xyzw"),
            base + ".rigid_initial_orientation_xyzw", failures)
        if initial_rigid_position is None or initial_rigid_orientation is None:
            continue
        if _norm(_subtract(initial["rigid_position"],
                           initial_rigid_position)) > state_tolerance:
            failures.append(Failure(base + ".initial.rigid.position_m",
                                    "initial rigid position differs from profile",
                                    list(initial["rigid_position"]),
                                    list(initial_rigid_position)))
        if quaternion_distance(initial["rigid_orientation"],
                               initial_rigid_orientation) > state_tolerance:
            failures.append(Failure(base + ".initial.rigid.orientation_xyzw",
                                    "initial rigid orientation differs from profile",
                                    list(initial["rigid_orientation"]),
                                    list(initial_rigid_orientation)))
        if (_norm(initial["rigid_linear"]) > state_tolerance or
                _norm(initial["rigid_angular"]) > state_tolerance):
            failures.append(Failure(base + ".initial.rigid",
                                    "initial rigid endpoint must be stationary",
                                    {"linear": list(initial["rigid_linear"]),
                                     "angular": list(initial["rigid_angular"])}))

        vertices_raw = definition.get("rigid_geometry_vertices_m")
        indices = definition.get("rigid_geometry_triangle_indices")
        if not isinstance(vertices_raw, list) or not isinstance(indices, list):
            failures.append(Failure(base + ".rigid_geometry_vertices_m",
                                    "invalid rigid geometry", vertices_raw))
            continue
        vertices = [_vector(item, f"{base}.rigid_geometry_vertices_m[{index}]",
                            failures)
                    for index, item in enumerate(vertices_raw)]
        if any(vertex is None for vertex in vertices):
            continue
        local_vertices = [vertex for vertex in vertices if vertex is not None]
        if (not indices or len(indices) % 3 != 0 or
                any(not isinstance(index, int) or isinstance(index, bool) or
                    index < 0 or index >= len(local_vertices) for index in indices)):
            failures.append(Failure(base + ".rigid_geometry_triangle_indices",
                                    "invalid rigid triangle topology", indices))
            continue
        triangles = [(indices[offset], indices[offset + 1], indices[offset + 2])
                     for offset in range(0, len(indices), 3)]
        transformed = [add(final["rigid_position"], rotate(
            vertex, final["rigid_orientation"])) for vertex in local_vertices]
        clearance = float("inf")
        for segment in range(node_count - 1):
            first = final["nodes"][segment]["position"]
            last = final["nodes"][segment + 1]["position"]
            if (point_inside_convex(first, transformed, triangles) or
                    point_inside_convex(last, transformed, triangles)):
                clearance = 0.0
                break
            for first_index, second_index, third_index in triangles:
                clearance = min(clearance, segment_triangle_distance(
                    first, last, transformed[first_index],
                    transformed[second_index], transformed[third_index]))
        combined_clearance = _scalar(
            definition.get("combined_clearance_m"),
            base + ".combined_clearance_m", failures)
        if combined_clearance is None or combined_clearance <= 0.0:
            continue
        penetration = max(0.0, combined_clearance - clearance)
        penetration_limit = float(tolerances.get(
            "maximum_overlap_penetration_m", 0.0))
        if penetration > penetration_limit:
            failures.append(Failure(base + ".final.rope",
                                    "rope capsule overlapped transformed rigid geometry",
                                    penetration, penetration_limit))

        rope_mass = _scalar(definition.get("rope_total_mass_kg"),
                            base + ".rope_total_mass_kg", failures)
        rigid_mass = _scalar(definition.get("rigid_mass_kg"),
                             base + ".rigid_mass_kg", failures)
        timestep = _scalar(definition.get("timestep_seconds"),
                           base + ".timestep_seconds", failures)
        normal = _vector(definition.get("contact_normal"),
                         base + ".contact_normal", failures)
        inertia_raw = _vector(definition.get("rigid_inertia_diagonal_kg_m2"),
                              base + ".rigid_inertia_diagonal_kg_m2", failures)
        if (rope_mass is None or rope_mass <= 0.0 or rigid_mass is None or
                rigid_mass <= 0.0 or timestep is None or timestep <= 0.0 or
                normal is None or inertia_raw is None):
            continue
        unit_normal = _unit(normal)
        if unit_normal is None:
            failures.append(Failure(base + ".contact_normal",
                                    "contact normal must be nonzero", normal))
            continue
        if abs(_norm(normal) - 1.0) > state_tolerance:
            failures.append(Failure(base + ".contact_normal",
                                    "contact normal must be unit length",
                                    _norm(normal), 1.0))

        node_mass = rope_mass / node_count
        initial_rope_momentum = scale(_sum([
            node["velocity"] for node in initial["nodes"]]), node_mass)
        final_rope_momentum = scale(_sum([
            node["velocity"] for node in final["nodes"]]), node_mass)
        rope_impulse = _subtract(final_rope_momentum, initial_rope_momentum)
        rigid_impulse = scale(_subtract(final["rigid_linear"],
                                        initial["rigid_linear"]), rigid_mass)
        reciprocal_residual = _norm(add(rope_impulse, rigid_impulse))
        observed_normal = dot(rope_impulse, unit_normal)
        impulse_activity = max(abs(observed_normal),
                               abs(dot(rigid_impulse, unit_normal)))
        tangent_impulse = _norm(_subtract(
            rope_impulse, scale(unit_normal, observed_normal)))
        initial_rope_velocity = scale(initial_rope_momentum, 1.0 / rope_mass)
        final_rope_velocity = scale(final_rope_momentum, 1.0 / rope_mass)
        initial_relative = _subtract(initial_rope_velocity,
                                     initial["rigid_linear"])
        final_relative = _subtract(final_rope_velocity, final["rigid_linear"])
        initial_closing = max(0.0, -dot(initial_relative, unit_normal))
        final_normal_speed = dot(final_relative, unit_normal)
        final_closing = max(0.0, -final_normal_speed)
        final_slip = _norm(_subtract(
            final_relative, scale(unit_normal, final_normal_speed)))
        expected_impulse = initial_closing / (1.0 / rope_mass +
                                               1.0 / rigid_mass)
        expected_error = abs(observed_normal - expected_impulse)
        force_impulse = scale(_sum([
            node["force"] for node in final["nodes"]]), timestep)
        force_error = _norm(_subtract(force_impulse, rope_impulse))
        orientation_drift = quaternion_distance(
            final["rigid_orientation"], initial["rigid_orientation"])
        angular_speed = _norm(final["rigid_angular"])

        def kinetic(snapshot_value: dict[str, Any]) -> float:
            rope_energy = sum(0.5 * node_mass * dot(
                node["velocity"], node["velocity"])
                              for node in snapshot_value["nodes"])
            rigid_energy = 0.5 * rigid_mass * dot(
                snapshot_value["rigid_linear"],
                snapshot_value["rigid_linear"])
            orientation = snapshot_value["rigid_orientation"]
            inverse = (-orientation[0], -orientation[1], -orientation[2],
                       orientation[3])
            local_angular = rotate(snapshot_value["rigid_angular"], inverse)
            rigid_energy += 0.5 * sum(
                inertia_raw[axis] * local_angular[axis] ** 2
                for axis in range(3))
            return rope_energy + rigid_energy

        energy_change = kinetic(final) - kinetic(initial)
        maximum_rope_speed = max(_norm(node["velocity"])
                                 for node in final["nodes"])
        rigid_linear_speed = _norm(final["rigid_linear"])
        rope_speed_limit = float(definition.get("rope_maximum_speed_m_s", 0.0)) + \
            float(tolerances.get("maximum_speed_slack_m_s", 0.0))
        rigid_speed_limit = float(definition.get(
            "rigid_maximum_linear_speed_m_s", 0.0)) + float(
                tolerances.get("maximum_speed_slack_m_s", 0.0))
        rigid_angular_limit = float(definition.get(
            "rigid_maximum_angular_speed_rad_s", 0.0)) + float(
                tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_rope_speed > rope_speed_limit:
            failures.append(Failure(base + ".final.rope",
                                    "rope speed exceeds authored maximum",
                                    maximum_rope_speed, rope_speed_limit))
        if rigid_linear_speed > rigid_speed_limit:
            failures.append(Failure(base + ".final.rigid.linear_velocity_m_s",
                                    "rigid speed exceeds authored maximum",
                                    rigid_linear_speed, rigid_speed_limit))
        if angular_speed > rigid_angular_limit:
            failures.append(Failure(base + ".final.rigid.angular_velocity_rad_s",
                                    "rigid angular speed exceeds authored maximum",
                                    angular_speed, rigid_angular_limit))

        expected_contact = definition.get("expected_contact") is True
        control_position_error = control_velocity_error = None
        control_momentum_change = control_contact_force = None
        if not expected_contact:
            control_position_error = max(_norm(_subtract(
                node["position"], add(before["position"], scale(
                    before["velocity"], timestep))))
                for before, node in zip(initial["nodes"], final["nodes"]))
            control_velocity_error = max(_norm(_subtract(
                node["velocity"], before["velocity"]))
                for before, node in zip(initial["nodes"], final["nodes"]))
            control_momentum_change = _norm(rope_impulse) + _norm(rigid_impulse)
            control_contact_force = max(_norm(node["force"])
                                        for node in final["nodes"])
            if control_position_error > float(tolerances.get(
                    "maximum_control_position_error_m", 0.0)):
                failures.append(Failure(base + ".final.rope",
                                        "separated control is not ballistic",
                                        control_position_error,
                                        tolerances.get(
                                            "maximum_control_position_error_m")))
            if control_velocity_error > float(tolerances.get(
                    "maximum_control_velocity_error_m_s", 0.0)):
                failures.append(Failure(base + ".final.rope",
                                        "separated control velocity changed",
                                        control_velocity_error,
                                        tolerances.get(
                                            "maximum_control_velocity_error_m_s")))
            if control_momentum_change > float(tolerances.get(
                    "maximum_control_momentum_change_n_s", 0.0)):
                failures.append(Failure(base + ".final",
                                        "separated control exchanged momentum",
                                        control_momentum_change,
                                        tolerances.get(
                                            "maximum_control_momentum_change_n_s")))
            if control_contact_force > float(tolerances.get(
                    "maximum_control_contact_force_n", 0.0)):
                failures.append(Failure(base + ".final.rope",
                                        "separated control reported contact force",
                                        control_contact_force,
                                        tolerances.get(
                                            "maximum_control_contact_force_n")))
            rigid_position_error = _norm(_subtract(
                final["rigid_position"], initial["rigid_position"]))
            rigid_velocity_error = _norm(final["rigid_linear"])
            rigid_orientation_error = quaternion_distance(
                final["rigid_orientation"], initial["rigid_orientation"])
            if (rigid_position_error > float(tolerances.get(
                    "maximum_control_position_error_m", 0.0)) or
                    rigid_velocity_error > float(tolerances.get(
                        "maximum_control_velocity_error_m_s", 0.0)) or
                    rigid_orientation_error > state_tolerance or
                    angular_speed > state_tolerance):
                failures.append(Failure(base + ".final.rigid",
                                        "separated control moved the rigid body",
                                        {"position_error_m": rigid_position_error,
                                         "velocity_error_m_s": rigid_velocity_error,
                                         "orientation_error": rigid_orientation_error,
                                         "angular_speed_rad_s": angular_speed}))
        else:
            declared_expected = _scalar(
                definition.get("expected_coherent_reduced_mass_impulse_n_s"),
                base + ".expected_coherent_reduced_mass_impulse_n_s", failures)
            if (declared_expected is not None and
                    abs(declared_expected - expected_impulse) > state_tolerance):
                failures.append(Failure(
                    base + ".expected_coherent_reduced_mass_impulse_n_s",
                    "declared impulse differs from coherent reduced-mass oracle",
                    declared_expected, expected_impulse))
            if impulse_activity < float(tolerances.get(
                    "minimum_contact_impulse_n_s", 0.0)):
                failures.append(Failure(base + ".final",
                                        "rope-rigid contact impulse is too small",
                                        impulse_activity,
                                        tolerances.get(
                                            "minimum_contact_impulse_n_s")))
            if reciprocal_residual > float(tolerances.get(
                    "maximum_reciprocal_impulse_residual_n_s", 0.0)):
                failures.append(Failure(base + ".final",
                                        "rope and rigid impulses are not reciprocal",
                                        reciprocal_residual,
                                        tolerances.get(
                                            "maximum_reciprocal_impulse_residual_n_s")))
            if expected_error > float(tolerances.get(
                    "maximum_expected_impulse_error_n_s", 0.0)):
                failures.append(Failure(
                    base + ".final.rope",
                    "normal impulse differs from coherent reduced-mass oracle",
                    expected_error,
                    tolerances.get("maximum_expected_impulse_error_n_s")))
            if force_error > float(tolerances.get(
                    "maximum_force_impulse_residual_n_s", 0.0)):
                failures.append(Failure(
                    base + ".final.rope",
                    "public rope contact force does not integrate to momentum",
                    force_error,
                    tolerances.get("maximum_force_impulse_residual_n_s")))
            if tangent_impulse > float(tolerances.get(
                    "maximum_tangential_impulse_n_s", 0.0)):
                failures.append(Failure(base + ".final.rope",
                                        "frictionless contact generated tangential impulse",
                                        tangent_impulse,
                                        tolerances.get(
                                            "maximum_tangential_impulse_n_s")))
            if final_slip < float(tolerances.get(
                    "minimum_frictionless_slip_m_s", 0.0)):
                failures.append(Failure(base + ".final",
                                        "rope and rigid moved as a hidden attachment",
                                        final_slip,
                                        tolerances.get(
                                            "minimum_frictionless_slip_m_s")))
            if final_closing > float(tolerances.get(
                    "maximum_final_closing_speed_m_s", 0.0)):
                failures.append(Failure(base + ".final",
                                        "rope and rigid remain closing after contact",
                                        final_closing,
                                        tolerances.get(
                                            "maximum_final_closing_speed_m_s")))
            if angular_speed > float(tolerances.get(
                    "maximum_angular_speed_rad_s", 0.0)):
                failures.append(Failure(
                    base + ".final.rigid.angular_velocity_rad_s",
                    "centered impact generated angular speed", angular_speed,
                    tolerances.get("maximum_angular_speed_rad_s")))
            if orientation_drift > float(tolerances.get(
                    "maximum_orientation_drift", 0.0)):
                failures.append(Failure(
                    base + ".final.rigid.orientation_xyzw",
                    "centered impact rotated the rigid body", orientation_drift,
                    tolerances.get("maximum_orientation_drift")))
            if energy_change > float(tolerances.get(
                    "maximum_energy_increase_j", 0.0)):
                failures.append(Failure(base + ".final",
                                        "passive contact increased kinetic energy",
                                        energy_change,
                                        tolerances.get(
                                            "maximum_energy_increase_j")))

        observations[identity] = {
            "minimum_capsule_geometry_clearance_m": clearance,
            "maximum_overlap_penetration_m": penetration,
            "rope_contact_impulse_n_s": list(rope_impulse),
            "rigid_contact_impulse_n_s": list(rigid_impulse),
            "contact_impulse_activity_n_s": impulse_activity,
            "reciprocal_impulse_residual_n_s": reciprocal_residual,
            "expected_normal_impulse_n_s": expected_impulse,
            "expected_impulse_error_n_s": expected_error,
            "rope_force_impulse_n_s": list(force_impulse),
            "force_impulse_residual_n_s": force_error,
            "tangential_impulse_n_s": tangent_impulse,
            "final_frictionless_slip_m_s": final_slip,
            "final_closing_speed_m_s": final_closing,
            "orientation_drift": orientation_drift,
            "angular_speed_rad_s": angular_speed,
            "kinetic_energy_change_j": energy_change,
            "maximum_rope_speed_m_s": maximum_rope_speed,
            "rigid_linear_speed_m_s": rigid_linear_speed,
            "control_position_error_m": control_position_error,
            "control_velocity_error_m_s": control_velocity_error,
            "control_momentum_change_n_s": control_momentum_change,
            "control_contact_force_n": control_contact_force,
        }
        sample_count += 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": len(actual_by_id),
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_rope_rigid_attachment(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture",
                                "profile needs one fixture object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected one fixture object", actual))
        actual = None

    tolerances = profile.get("tolerances", {})
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    def add(left: tuple[float, float, float],
            right: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]

    def scale(value: tuple[float, float, float],
              factor: float) -> tuple[float, float, float]:
        return tuple(factor * component for component in value)  # type: ignore[return-value]

    def cross(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0])

    def rotate(value: tuple[float, float, float],
               quaternion: tuple[float, float, float, float]) -> tuple[
                   float, float, float]:
        vector = quaternion[:3]
        scalar = quaternion[3]
        first = scale(vector, 2.0 * sum(a * b for a, b in zip(vector, value)))
        second = scale(value, scalar * scalar - sum(a * a for a in vector))
        third = scale(cross(vector, value), 2.0 * scalar)
        return add(add(first, second), third)

    def compare_contract(value: Any, reference: Any, path: str) -> None:
        if isinstance(reference, bool) or isinstance(reference, str):
            if value != reference or type(value) is not type(reference):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        value, reference))
            return
        if isinstance(reference, int):
            if (not isinstance(value, int) or isinstance(value, bool) or
                    value != reference):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        value, reference))
            return
        if isinstance(reference, float):
            number = _scalar(value, path, failures)
            if number is not None and abs(number - reference) > state_tolerance:
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        number, reference))
            return
        if isinstance(reference, list):
            if not isinstance(value, list) or len(value) != len(reference):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        value, reference))
                return
            for index, (item, expected) in enumerate(zip(value, reference)):
                compare_contract(item, expected, f"{path}[{index}]")
            return
        if isinstance(reference, dict):
            if not isinstance(value, dict):
                failures.append(Failure(path,
                                        "fixture setting differs from profile",
                                        value, reference))
                return
            for key, expected in reference.items():
                if key not in value:
                    failures.append(Failure(path + "." + key,
                                            "fixture setting is missing",
                                            None, expected))
                else:
                    compare_contract(value[key], expected, path + "." + key)
            return
        if value != reference:
            failures.append(Failure(path, "fixture setting differs from profile",
                                    value, reference))

    observations: dict[str, Any] = {}
    sample_count = 0
    if actual is not None and definition:
        base = "$.fixture"
        compare_contract(actual, definition, base)
        extras = sorted(set(actual) - set(definition) -
                        {"initial", "previous", "final"})
        if extras:
            failures.append(Failure(base, "unexpected fixture fields", extras))

        count_value = definition.get("rope_node_count")
        rope_count = (count_value if isinstance(count_value, int) and
                      not isinstance(count_value, bool) and count_value >= 2 else 0)

        def snapshot(value: Any, path: str) -> dict[str, Any] | None:
            if not isinstance(value, dict):
                failures.append(Failure(path, "expected a snapshot object", value))
                return None
            rope = value.get("rope")
            rigid = value.get("rigid")
            if not isinstance(rope, list) or len(rope) != rope_count:
                failures.append(Failure(
                    path + ".rope", "rope state count differs from profile",
                    len(rope) if isinstance(rope, list) else rope, rope_count))
                return None
            if not isinstance(rigid, dict):
                failures.append(Failure(path + ".rigid",
                                        "expected a rigid state object", rigid))
                return None
            rope_states = []
            for index, item in enumerate(rope):
                item_path = f"{path}.rope[{index}]"
                if not isinstance(item, dict):
                    failures.append(Failure(item_path,
                                            "expected a rope state object", item))
                    continue
                if item.get("index") != index:
                    failures.append(Failure(item_path + ".index",
                                            "public state order changed",
                                            item.get("index"), index))
                position = _vector(item.get("position_m"),
                                   item_path + ".position_m", failures)
                velocity = _vector(item.get("velocity_m_s"),
                                   item_path + ".velocity_m_s", failures)
                constraint = _vector(item.get("constraint_force_n"),
                                     item_path + ".constraint_force_n", failures)
                contact = _vector(item.get("contact_force_n"),
                                  item_path + ".contact_force_n", failures)
                if (position is not None and velocity is not None and
                        constraint is not None and contact is not None):
                    rope_states.append({"position": position,
                                        "velocity": velocity,
                                        "constraint": constraint,
                                        "contact": contact})
            position = _vector(rigid.get("position_m"),
                               path + ".rigid.position_m", failures)
            orientation = _quaternion(rigid.get("orientation_xyzw"),
                                      path + ".rigid.orientation_xyzw", failures)
            velocity = _vector(rigid.get("linear_velocity_m_s"),
                               path + ".rigid.linear_velocity_m_s", failures)
            angular = _vector(rigid.get("angular_velocity_rad_s"),
                              path + ".rigid.angular_velocity_rad_s", failures)
            if (len(rope_states) != rope_count or position is None or
                    orientation is None or velocity is None or angular is None):
                return None
            orientation_norm = math.sqrt(sum(value * value
                                             for value in orientation))
            if abs(orientation_norm - 1.0) > state_tolerance:
                failures.append(Failure(path + ".rigid.orientation_xyzw",
                                        "rigid orientation is not normalized",
                                        orientation_norm, 1.0))
            return {"rope": rope_states, "rigid": {
                "position": position, "orientation": orientation,
                "velocity": velocity, "angular": angular}}

        initial = snapshot(actual.get("initial"), base + ".initial")
        previous = snapshot(actual.get("previous"), base + ".previous")
        final = snapshot(actual.get("final"), base + ".final")

        centerline_value = definition.get("rope_centerline_m")
        centerline = []
        if isinstance(centerline_value, list):
            for index, value in enumerate(centerline_value):
                vector = _vector(value,
                                 f"$.profile.fixture.rope_centerline_m[{index}]",
                                 failures)
                if vector is not None:
                    centerline.append(vector)
        local_anchor = _vector(definition.get("attachment_local_anchor_m"),
                               "$.profile.fixture.attachment_local_anchor_m",
                               failures)
        initial_position = _vector(definition.get("rigid_initial_position_m"),
                                   "$.profile.fixture.rigid_initial_position_m",
                                   failures)
        initial_orientation = _quaternion(
            definition.get("rigid_initial_orientation_xyzw"),
            "$.profile.fixture.rigid_initial_orientation_xyzw", failures)
        initial_velocity = _vector(
            definition.get("rigid_initial_linear_velocity_m_s"),
            "$.profile.fixture.rigid_initial_linear_velocity_m_s", failures)
        initial_angular = _vector(
            definition.get("rigid_initial_angular_velocity_rad_s"),
            "$.profile.fixture.rigid_initial_angular_velocity_rad_s", failures)
        external_force = _vector(definition.get("rigid_external_force_n"),
                                 "$.profile.fixture.rigid_external_force_n",
                                 failures)
        external_torque = _vector(definition.get("rigid_external_torque_n_m"),
                                  "$.profile.fixture.rigid_external_torque_n_m",
                                  failures)
        inertia = _vector(definition.get("rigid_inertia_diagonal_kg_m2"),
                          "$.profile.fixture.rigid_inertia_diagonal_kg_m2",
                          failures)
        timestep = _scalar(definition.get("timestep_seconds"),
                           "$.profile.fixture.timestep_seconds", failures)
        body_mass = _scalar(definition.get("rigid_mass_kg"),
                            "$.profile.fixture.rigid_mass_kg", failures)
        rope_mass = _scalar(definition.get("rope_total_mass_kg"),
                            "$.profile.fixture.rope_total_mass_kg", failures)

        if (initial is not None and previous is not None and final is not None and
                len(centerline) == 2 and local_anchor is not None and
                initial_position is not None and initial_orientation is not None and
                initial_velocity is not None and initial_angular is not None and
                external_force is not None and external_torque is not None and
                inertia is not None and timestep is not None and timestep > 0.0 and
                body_mass is not None and body_mass > 0.0 and
                rope_mass is not None and rope_mass > 0.0):
            attached_end = definition.get("attached_rope_end")
            endpoint = 0 if attached_end == "first" else rope_count - 1
            for index in range(rope_count):
                fraction = index / (rope_count - 1)
                expected = tuple(centerline[0][axis] + fraction * (
                    centerline[1][axis] - centerline[0][axis])
                                 for axis in range(3))
                if _norm(_subtract(initial["rope"][index]["position"],
                                   expected)) > state_tolerance:
                    failures.append(Failure(
                        f"{base}.initial.rope[{index}].position_m",
                        "initial rope state differs from the public centerline",
                        list(initial["rope"][index]["position"]), list(expected)))
                if _norm(initial["rope"][index]["velocity"]) > state_tolerance:
                    failures.append(Failure(
                        f"{base}.initial.rope[{index}].velocity_m_s",
                        "initial rope velocity must be zero",
                        list(initial["rope"][index]["velocity"]), 0.0))
            for field, value, expected in (
                    ("position_m", initial["rigid"]["position"], initial_position),
                    ("linear_velocity_m_s", initial["rigid"]["velocity"],
                     initial_velocity),
                    ("angular_velocity_rad_s", initial["rigid"]["angular"],
                     initial_angular)):
                if _norm(_subtract(value, expected)) > state_tolerance:
                    failures.append(Failure(base + ".initial.rigid." + field,
                                            "initial rigid state differs from profile",
                                            list(value), list(expected)))
            if math.sqrt(sum((a - b) * (a - b) for a, b in zip(
                    initial["rigid"]["orientation"], initial_orientation))) > state_tolerance:
                failures.append(Failure(
                    base + ".initial.rigid.orientation_xyzw",
                    "initial rigid state differs from profile",
                    list(initial["rigid"]["orientation"]),
                    list(initial_orientation)))

            def anchor(state: dict[str, Any]) -> tuple[float, float, float]:
                return add(state["rigid"]["position"], rotate(
                    local_anchor, state["rigid"]["orientation"]))

            initial_drift = _norm(_subtract(
                initial["rope"][endpoint]["position"], anchor(initial)))
            previous_drift = _norm(_subtract(
                previous["rope"][endpoint]["position"], anchor(previous)))
            final_drift = _norm(_subtract(
                final["rope"][endpoint]["position"], anchor(final)))
            drift_limit = float(tolerances.get(
                "maximum_attachment_drift_m", 0.0))
            if initial_drift > state_tolerance:
                failures.append(Failure(base + ".initial",
                                        "attachment endpoint does not start at rigid anchor",
                                        initial_drift, state_tolerance))
            if max(previous_drift, final_drift) > drift_limit:
                failures.append(Failure(base + ".final",
                                        "rope-rigid attachment drift exceeds profile",
                                        max(previous_drift, final_drift), drift_limit))

            body_impulse = scale(_subtract(final["rigid"]["velocity"],
                                           previous["rigid"]["velocity"]),
                                 body_mass)
            external_impulse = scale(external_force, timestep)
            attachment_impulse = _subtract(body_impulse, external_impulse)
            node_mass = rope_mass / rope_count
            free_nodes = [index for index in range(rope_count)
                          if index != endpoint]
            declared_free_mass = _scalar(
                definition.get("free_rope_mass_kg"),
                "$.profile.fixture.free_rope_mass_kg", failures)
            derived_free_mass = node_mass * len(free_nodes)
            if (declared_free_mass is not None and
                    abs(declared_free_mass - derived_free_mass) > state_tolerance):
                failures.append(Failure(
                    "$.profile.fixture.free_rope_mass_kg",
                    "free rope mass does not match public node masses",
                    declared_free_mass, derived_free_mass))
            rope_impulse = scale(_sum([
                _subtract(final["rope"][index]["velocity"],
                          previous["rope"][index]["velocity"])
                for index in free_nodes]), node_mass)
            linear_residual = _norm(add(attachment_impulse, rope_impulse))
            reaction_limit = float(tolerances.get(
                "minimum_reaction_impulse_n_s", 0.0))
            if _norm(attachment_impulse) < reaction_limit:
                failures.append(Failure(base + ".final",
                                        "attachment reaction impulse is too small",
                                        _norm(attachment_impulse), reaction_limit))
            reciprocal_limit = float(tolerances.get(
                "maximum_reciprocal_linear_impulse_residual_n_s", 0.0))
            if linear_residual > reciprocal_limit:
                failures.append(Failure(
                    base + ".final",
                    "rigid and free-rope impulses are not reciprocal",
                    linear_residual, reciprocal_limit))

            force_impulse = scale(final["rope"][endpoint]["constraint"],
                                  timestep)
            force_residual = _norm(_subtract(force_impulse,
                                             attachment_impulse))
            force_limit = float(tolerances.get(
                "maximum_force_impulse_residual_n_s", 0.0))
            if force_residual > force_limit:
                failures.append(Failure(
                    base + f".final.rope[{endpoint}].constraint_force_n",
                    "public constraint force does not match rigid attachment impulse",
                    force_residual, force_limit))

            angular_impulse = tuple(inertia[axis] * (
                final["rigid"]["angular"][axis] -
                previous["rigid"]["angular"][axis]) for axis in range(3))
            external_angular_impulse = scale(external_torque, timestep)
            angular_reaction = _subtract(angular_impulse,
                                         external_angular_impulse)
            previous_arm = rotate(local_anchor,
                                  previous["rigid"]["orientation"])
            final_arm = rotate(local_anchor,
                               final["rigid"]["orientation"])
            midpoint_arm = scale(add(previous_arm, final_arm), 0.5)
            expected_angular_reaction = cross(midpoint_arm,
                                              attachment_impulse)
            angular_residual = _norm(_subtract(
                angular_reaction, expected_angular_reaction))
            angular_activity_limit = float(tolerances.get(
                "minimum_angular_reaction_impulse_n_m_s", 0.0))
            if _norm(angular_reaction) < angular_activity_limit:
                failures.append(Failure(base + ".final.rigid",
                                        "rigid angular reaction is too small",
                                        _norm(angular_reaction),
                                        angular_activity_limit))
            angular_limit = float(tolerances.get(
                "maximum_reciprocal_angular_impulse_residual_n_m_s", 0.0))
            if angular_residual > angular_limit:
                failures.append(Failure(
                    base + ".final.rigid",
                    "angular attachment impulse is not reciprocal",
                    angular_residual, angular_limit))

            maximum_contact = max(_norm(state["contact"])
                                  for snapshot_state in (initial, previous, final)
                                  for state in snapshot_state["rope"])
            contact_limit = float(tolerances.get("maximum_contact_force_n", 0.0))
            if maximum_contact > contact_limit:
                failures.append(Failure(base + ".final.rope",
                                        "contact channel is active in attachment fixture",
                                        maximum_contact, contact_limit))

            rest_length = _norm(_subtract(centerline[1], centerline[0])) / (
                rope_count - 1)
            length_error = max(abs(_norm(_subtract(
                snapshot_state["rope"][index + 1]["position"],
                snapshot_state["rope"][index]["position"])) - rest_length)
                for snapshot_state in (previous, final)
                for index in range(rope_count - 1))
            length_limit = float(tolerances.get(
                "maximum_rope_length_error_m", 0.0))
            if length_error > length_limit:
                failures.append(Failure(base + ".final.rope",
                                        "rope segment length differs from rest length",
                                        length_error, length_limit))

            speed_slack = float(tolerances.get("maximum_speed_slack_m_s", 0.0))
            maximum_rope_speed = max(_norm(state["velocity"])
                                     for state in final["rope"])
            rigid_linear_speed = _norm(final["rigid"]["velocity"])
            rigid_angular_speed = _norm(final["rigid"]["angular"])
            rope_speed_limit = float(definition.get("rope_maximum_speed_m_s", 0.0))
            rigid_speed_limit = float(definition.get(
                "rigid_maximum_linear_speed_m_s", 0.0))
            angular_speed_limit = float(definition.get(
                "rigid_maximum_angular_speed_rad_s", 0.0))
            if maximum_rope_speed > rope_speed_limit + speed_slack:
                failures.append(Failure(base + ".final.rope",
                                        "rope speed exceeds authored maximum",
                                        maximum_rope_speed,
                                        rope_speed_limit + speed_slack))
            if rigid_linear_speed > rigid_speed_limit + speed_slack:
                failures.append(Failure(base + ".final.rigid.linear_velocity_m_s",
                                        "rigid speed exceeds authored maximum",
                                        rigid_linear_speed,
                                        rigid_speed_limit + speed_slack))
            if rigid_angular_speed > angular_speed_limit + speed_slack:
                failures.append(Failure(base + ".final.rigid.angular_velocity_rad_s",
                                        "rigid angular speed exceeds authored maximum",
                                        rigid_angular_speed,
                                        angular_speed_limit + speed_slack))

            observations = {
                "initial_attachment_drift_m": initial_drift,
                "previous_attachment_drift_m": previous_drift,
                "final_attachment_drift_m": final_drift,
                "body_attachment_impulse_n_s": list(attachment_impulse),
                "free_rope_impulse_n_s": list(rope_impulse),
                "reciprocal_linear_impulse_residual_n_s": linear_residual,
                "endpoint_force_impulse_n_s": list(force_impulse),
                "force_impulse_residual_n_s": force_residual,
                "angular_reaction_impulse_n_m_s": list(angular_reaction),
                "expected_angular_reaction_impulse_n_m_s": list(
                    expected_angular_reaction),
                "reciprocal_angular_impulse_residual_n_m_s": angular_residual,
                "maximum_contact_force_n": maximum_contact,
                "maximum_rope_length_error_m": length_error,
                "maximum_rope_speed_m_s": maximum_rope_speed,
                "rigid_linear_speed_m_s": rigid_linear_speed,
                "rigid_angular_speed_rad_s": rigid_angular_speed,
            }
            sample_count = 1

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual is not None else 0,
        "sample_count": sample_count,
        "failure_count": len(failures),
        "observations": observations,
        "failures": [asdict(failure) for failure in failures],
    }


def _evaluate_cloth_rigid_contact(
        profile: dict[str, Any], result: dict[str, Any], expected_hash: str,
        driver: Any, failures: list[Failure]) -> dict[str, Any]:
    definition = profile.get("fixture")
    actual = result.get("fixture")
    if not isinstance(definition, dict):
        failures.append(Failure("$.profile.fixture",
                                "expected a fixture object", definition))
        definition = {}
    if not isinstance(actual, dict):
        failures.append(Failure("$.fixture", "expected a fixture object", actual))
        actual = {}
    tolerances = profile.get("tolerances")
    if not isinstance(tolerances, dict):
        failures.append(Failure("$.profile.tolerances",
                                "expected an object", tolerances))
        tolerances = {}
    state_tolerance = float(tolerances.get("fixture_state_absolute", 0.0))

    for field in ("id", "graph_policy", "substeps",
                  "cloth_solver_iterations", "cloth_preserve_volume",
                  "contact_cloth_vertex_index", "two_way", "oracle"):
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    actual.get(field), definition.get(field)))

    for field in (
            "preparation_timestep_seconds", "timestep_seconds",
            "cloth_vertex_mass_kg", "cloth_thickness_m",
            "cloth_stretch_compliance_m_per_n",
            "cloth_bending_compliance_m_per_n", "cloth_velocity_damping",
            "cloth_contact_friction", "rigid_mass_kg", "rigid_friction",
            "rigid_restitution", "rigid_linear_damping_per_s",
            "rigid_angular_damping_per_s", "rigid_maximum_linear_speed_m_s",
            "rigid_maximum_angular_speed_rad_s", "rigid_collision_margin_m"):
        value = _scalar(actual.get(field), f"$.fixture.{field}", failures)
        wanted = _scalar(definition.get(field),
                         f"$.profile.fixture.{field}", failures)
        if (value is not None and wanted is not None and
                abs(value - wanted) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    value, wanted))

    def vectors(value: Any, path: str) -> list[tuple[float, float, float]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a vector array", value))
            return []
        parsed: list[tuple[float, float, float]] = []
        for index, item in enumerate(value):
            vector = _vector(item, f"{path}[{index}]", failures)
            if vector is not None:
                parsed.append(vector)
        return parsed

    def compare_vector(field: str) -> tuple[float, float, float] | None:
        value = _vector(actual.get(field), f"$.fixture.{field}", failures)
        wanted = _vector(definition.get(field),
                         f"$.profile.fixture.{field}", failures)
        if (value is not None and wanted is not None and
                _norm(_subtract(value, wanted)) > state_tolerance):
            failures.append(Failure(f"$.fixture.{field}",
                                    "fixture setting differs from profile",
                                    list(value), list(wanted)))
        return value

    def compare_vectors(field: str) -> list[tuple[float, float, float]]:
        values = vectors(actual.get(field), f"$.fixture.{field}")
        wanted = vectors(definition.get(field),
                         f"$.profile.fixture.{field}")
        if len(values) != len(wanted):
            failures.append(Failure(f"$.fixture.{field}",
                                    "vector count differs from profile",
                                    len(values), len(wanted)))
        else:
            for index, (value, expected) in enumerate(zip(values, wanted)):
                if _norm(_subtract(value, expected)) > state_tolerance:
                    failures.append(Failure(
                        f"$.fixture.{field}[{index}]",
                        "public input differs from profile",
                        list(value), list(expected)))
        return values

    compare_vector("preparation_gravity_m_s2")
    compare_vector("gravity_m_s2")
    prepared_velocity = compare_vector("prepared_cloth_velocity_m_s")
    compare_vector("rigid_initial_position_m")
    authored_vertices = compare_vectors("authored_cloth_vertices_m")
    prepared_vertices = compare_vectors("prepared_cloth_vertices_m")
    rigid_vertices = compare_vectors("rigid_geometry_vertices_m")
    inertia = compare_vector("rigid_inertia_diagonal_kg_m2")

    inverse_value = actual.get("cloth_inverse_masses_kg_inverse")
    inverse_expected = definition.get("cloth_inverse_masses_kg_inverse")
    inverse_masses: list[float] = []
    if not isinstance(inverse_value, list) or not isinstance(
            inverse_expected, list) or len(inverse_value) != len(
                inverse_expected):
        failures.append(Failure(
            "$.fixture.cloth_inverse_masses_kg_inverse",
            "inverse-mass array differs from profile", inverse_value,
            inverse_expected))
    else:
        for index, (item, wanted) in enumerate(zip(inverse_value,
                                                   inverse_expected)):
            value = _scalar(item,
                            f"$.fixture.cloth_inverse_masses_kg_inverse[{index}]",
                            failures)
            expected = _scalar(
                wanted,
                f"$.profile.fixture.cloth_inverse_masses_kg_inverse[{index}]",
                failures)
            if value is None:
                continue
            inverse_masses.append(value)
            if value <= 0.0:
                failures.append(Failure(
                    f"$.fixture.cloth_inverse_masses_kg_inverse[{index}]",
                    "quality fixture requires positive inverse mass", value,
                    "> 0"))
            if expected is not None and abs(value - expected) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.cloth_inverse_masses_kg_inverse[{index}]",
                    "fixture setting differs from profile", value, expected))

    for field in ("cloth_triangle_indices",
                  "rigid_geometry_triangle_indices",
                  "impact_face_vertex_indices"):
        if actual.get(field) != definition.get(field):
            failures.append(Failure(f"$.fixture.{field}",
                                    "topology differs from profile",
                                    actual.get(field), definition.get(field)))

    initial_orientation = _quaternion(
        actual.get("rigid_initial_orientation_xyzw"),
        "$.fixture.rigid_initial_orientation_xyzw", failures)
    expected_orientation = _quaternion(
        definition.get("rigid_initial_orientation_xyzw"),
        "$.profile.fixture.rigid_initial_orientation_xyzw", failures)
    if (initial_orientation is not None and expected_orientation is not None and
            math.sqrt(sum((a - b) ** 2 for a, b in zip(
                initial_orientation, expected_orientation))) > state_tolerance):
        failures.append(Failure(
            "$.fixture.rigid_initial_orientation_xyzw",
            "fixture setting differs from profile", list(initial_orientation),
            list(expected_orientation)))

    def cloth_states(value: Any, path: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            failures.append(Failure(path, "expected a cloth state array", value))
            return []
        parsed: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(item, dict) or item.get("index") != index:
                failures.append(Failure(item_path,
                                        "cloth state index mismatch", item))
                continue
            position = _vector(item.get("position_m"),
                               item_path + ".position_m", failures)
            velocity = _vector(item.get("velocity_m_s"),
                               item_path + ".velocity_m_s", failures)
            force = _vector(item.get("rigid_contact_force_n"),
                            item_path + ".rigid_contact_force_n", failures)
            if position is not None and velocity is not None and force is not None:
                parsed.append({"position": position, "velocity": velocity,
                               "force": force})
        return parsed

    def rigid_state(value: Any, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a rigid state object", value))
            return None
        position = _vector(value.get("position_m"), path + ".position_m",
                           failures)
        orientation = _quaternion(value.get("orientation_xyzw"),
                                  path + ".orientation_xyzw", failures)
        linear = _vector(value.get("linear_velocity_m_s"),
                         path + ".linear_velocity_m_s", failures)
        angular = _vector(value.get("angular_velocity_rad_s"),
                          path + ".angular_velocity_rad_s", failures)
        if None in (position, orientation, linear, angular):
            return None
        return {"position": position, "orientation": orientation,
                "linear": linear, "angular": angular}

    def snapshot(value: Any, path: str) -> tuple[
            list[dict[str, Any]], dict[str, Any] | None]:
        if not isinstance(value, dict):
            failures.append(Failure(path, "expected a snapshot object", value))
            return [], None
        return (cloth_states(value.get("cloth"), path + ".cloth"),
                rigid_state(value.get("rigid"), path + ".rigid"))

    initial_cloth, initial_rigid = snapshot(actual.get("initial"),
                                             "$.fixture.initial")
    final_cloth, final_rigid = snapshot(actual.get("final"),
                                         "$.fixture.final")
    expected_count = len(prepared_vertices)
    for phase, states in (("initial", initial_cloth),
                          ("final", final_cloth)):
        if len(states) != expected_count:
            failures.append(Failure(f"$.fixture.{phase}.cloth",
                                    "cloth state count differs from profile",
                                    len(states), expected_count))

    if len(initial_cloth) == expected_count and prepared_velocity is not None:
        for index, state in enumerate(initial_cloth):
            if _norm(_subtract(state["position"],
                               prepared_vertices[index])) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial.cloth[{index}].position_m",
                    "initial cloth state differs from prepared public input",
                    list(state["position"]), list(prepared_vertices[index])))
            if _norm(_subtract(state["velocity"],
                               prepared_velocity)) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial.cloth[{index}].velocity_m_s",
                    "initial cloth velocity differs from preparation oracle",
                    list(state["velocity"]), list(prepared_velocity)))
            if _norm(state["force"]) > state_tolerance:
                failures.append(Failure(
                    f"$.fixture.initial.cloth[{index}].rigid_contact_force_n",
                    "initial contact force must be zero",
                    list(state["force"]), 0.0))

    zero = (0.0, 0.0, 0.0)
    if initial_rigid is not None and initial_orientation is not None:
        rigid_initial_position = _vector(
            definition.get("rigid_initial_position_m"),
            "$.profile.fixture.rigid_initial_position_m", failures)
        if (rigid_initial_position is not None and
                _norm(_subtract(initial_rigid["position"],
                                rigid_initial_position)) > state_tolerance):
            failures.append(Failure("$.fixture.initial.rigid.position_m",
                                    "initial rigid state differs from profile",
                                    list(initial_rigid["position"]),
                                    list(rigid_initial_position)))
        if _norm(initial_rigid["linear"]) > state_tolerance or _norm(
                initial_rigid["angular"]) > state_tolerance:
            failures.append(Failure("$.fixture.initial.rigid",
                                    "initial rigid velocity must be zero",
                                    initial_rigid, 0.0))

    contact_index = definition.get("contact_cloth_vertex_index")
    rigid_mass_value = _scalar(definition.get("rigid_mass_kg"),
                               "$.profile.fixture.rigid_mass_kg", failures)
    timestep = _scalar(definition.get("timestep_seconds"),
                       "$.profile.fixture.timestep_seconds", failures)
    normal = (-1.0, 0.0, 0.0)
    cloth_impulse = rigid_impulse = force_impulse = None
    impulse_activity = reciprocal_residual = expected_impulse = None
    expected_error = force_error = tangential_impulse = None
    final_penetration = final_slip = final_closing = None
    energy_increase = orientation_drift = maximum_speed = None

    valid_index = (isinstance(contact_index, int) and
                   not isinstance(contact_index, bool) and
                   0 <= contact_index < len(initial_cloth) and
                   contact_index < len(final_cloth) and
                   contact_index < len(inverse_masses))
    if not valid_index:
        failures.append(Failure("$.profile.fixture.contact_cloth_vertex_index",
                                "invalid contact cloth vertex", contact_index))
    elif initial_rigid is not None and final_rigid is not None and \
            rigid_mass_value is not None and rigid_mass_value > 0.0 and \
            timestep is not None and timestep > 0.0:
        cloth_impulses = []
        for index, (before, after) in enumerate(zip(initial_cloth, final_cloth)):
            mass = 1.0 / inverse_masses[index]
            cloth_impulses.append(tuple(mass * component for component in
                _subtract(after["velocity"], before["velocity"])))
        cloth_impulse = _sum(cloth_impulses)
        rigid_impulse = tuple(rigid_mass_value * component for component in
                              _subtract(final_rigid["linear"],
                                        initial_rigid["linear"]))
        reciprocal_residual = _norm(tuple(a + b for a, b in zip(
            cloth_impulse, rigid_impulse)))
        impulse_activity = max(_norm(cloth_impulse), _norm(rigid_impulse))
        if impulse_activity < float(tolerances.get(
                "minimum_contact_impulse_n_s", 0.0)):
            failures.append(Failure("$.fixture.final",
                                    "cloth-rigid contact impulse is too small",
                                    impulse_activity,
                                    tolerances.get("minimum_contact_impulse_n_s")))
        if reciprocal_residual > float(tolerances.get(
                "maximum_reciprocal_impulse_residual_n_s", 0.0)):
            failures.append(Failure("$.fixture.final",
                                    "cloth and rigid impulses are not reciprocal",
                                    reciprocal_residual,
                                    tolerances.get(
                                        "maximum_reciprocal_impulse_residual_n_s")))

        relative_initial = _subtract(
            initial_cloth[contact_index]["velocity"],
            initial_rigid["linear"])
        closing = max(0.0, -sum(a * b for a, b in zip(
            relative_initial, normal)))
        denominator = inverse_masses[contact_index] + 1.0 / rigid_mass_value
        expected_impulse = closing / denominator
        expected_vector = tuple(expected_impulse * component
                                for component in normal)
        expected_error = _norm(_subtract(cloth_impulse, expected_vector))
        if expected_error > float(tolerances.get(
                "maximum_expected_impulse_error_n_s", 0.0)):
            failures.append(Failure("$.fixture.final.cloth",
                                    "contact impulse differs from reduced-mass oracle",
                                    expected_error,
                                    tolerances.get(
                                        "maximum_expected_impulse_error_n_s")))

        total_force = _sum([state["force"] for state in final_cloth])
        force_impulse = tuple(component * timestep for component in total_force)
        force_error = _norm(_subtract(force_impulse, cloth_impulse))
        if force_error > float(tolerances.get(
                "maximum_force_impulse_residual_n_s", 0.0)):
            failures.append(Failure("$.fixture.final.cloth",
                                    "public contact force does not integrate to the cloth impulse",
                                    force_error,
                                    tolerances.get(
                                        "maximum_force_impulse_residual_n_s")))

        normal_component = sum(a * b for a, b in zip(cloth_impulse, normal))
        tangential_impulse = _norm(tuple(
            cloth_impulse[axis] - normal_component * normal[axis]
            for axis in range(3)))
        if tangential_impulse > float(tolerances.get(
                "maximum_tangential_impulse_n_s", 0.0)):
            failures.append(Failure("$.fixture.final.cloth",
                                    "frictionless contact generated tangential impulse",
                                    tangential_impulse,
                                    tolerances.get(
                                        "maximum_tangential_impulse_n_s")))

        relative_final = _subtract(
            final_cloth[contact_index]["velocity"], final_rigid["linear"])
        final_normal_speed = sum(a * b for a, b in zip(relative_final, normal))
        final_closing = max(0.0, -final_normal_speed)
        if final_closing > float(tolerances.get(
                "maximum_final_closing_speed_m_s", 0.0)):
            failures.append(Failure("$.fixture.final",
                                    "cloth and rigid remain closing after contact",
                                    final_closing,
                                    tolerances.get(
                                        "maximum_final_closing_speed_m_s")))
        tangent_velocity = tuple(
            relative_final[axis] - final_normal_speed * normal[axis]
            for axis in range(3))
        final_slip = _norm(tangent_velocity)
        if final_slip < float(tolerances.get(
                "minimum_frictionless_slip_m_s", 0.0)):
            failures.append(Failure("$.fixture.final",
                                    "cloth and rigid moved as a hidden attachment",
                                    final_slip,
                                    tolerances.get(
                                        "minimum_frictionless_slip_m_s")))

        face_indices = definition.get("impact_face_vertex_indices")
        if isinstance(face_indices, list) and face_indices and rigid_vertices:
            face_x = min(rigid_vertices[index][0] for index in face_indices
                         if isinstance(index, int) and
                         0 <= index < len(rigid_vertices))
            envelope_x = (final_rigid["position"][0] + face_x -
                          float(definition.get("rigid_collision_margin_m", 0.0)) -
                          float(definition.get("cloth_thickness_m", 0.0)))
            final_penetration = max(
                0.0, final_cloth[contact_index]["position"][0] - envelope_x)
            if final_penetration > float(tolerances.get(
                    "maximum_overlap_penetration_m", 0.0)):
                failures.append(Failure(
                    f"$.fixture.final.cloth[{contact_index}].position_m",
                    "cloth vertex overlapped the rigid contact envelope",
                    final_penetration,
                    tolerances.get("maximum_overlap_penetration_m")))

        orientation_drift = min(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(
                final_rigid["orientation"], initial_rigid["orientation"]))),
            math.sqrt(sum((a + b) ** 2 for a, b in zip(
                final_rigid["orientation"], initial_rigid["orientation"]))))
        if orientation_drift > float(tolerances.get(
                "maximum_orientation_drift", 0.0)):
            failures.append(Failure("$.fixture.final.rigid.orientation_xyzw",
                                    "centered impact rotated the rigid body",
                                    orientation_drift,
                                    tolerances.get("maximum_orientation_drift")))
        angular_speed = _norm(final_rigid["angular"])
        if angular_speed > float(tolerances.get(
                "maximum_angular_speed_rad_s", 0.0)):
            failures.append(Failure("$.fixture.final.rigid.angular_velocity_rad_s",
                                    "centered impact generated angular speed",
                                    angular_speed,
                                    tolerances.get(
                                        "maximum_angular_speed_rad_s")))

        def kinetic(cloth: list[dict[str, Any]],
                    rigid: dict[str, Any]) -> float:
            cloth_energy = sum(0.5 / inverse_masses[index] *
                               _norm(state["velocity"]) ** 2
                               for index, state in enumerate(cloth))
            rigid_energy = 0.5 * rigid_mass_value * _norm(rigid["linear"]) ** 2
            if inertia is not None:
                rigid_energy += 0.5 * sum(
                    inertia[axis] * rigid["angular"][axis] ** 2
                    for axis in range(3))
            return cloth_energy + rigid_energy
        energy_increase = kinetic(final_cloth, final_rigid) - kinetic(
            initial_cloth, initial_rigid)
        if energy_increase > float(tolerances.get(
                "maximum_energy_increase_j", 0.0)):
            failures.append(Failure("$.fixture.final",
                                    "passive contact increased kinetic energy",
                                    energy_increase,
                                    tolerances.get("maximum_energy_increase_j")))

        maximum_speed = max(
            [_norm(state["velocity"]) for state in final_cloth] +
            [_norm(final_rigid["linear"])])
        speed_limit = max(
            float(definition.get("rigid_maximum_linear_speed_m_s", 0.0)),
            20.0) + float(tolerances.get("maximum_speed_slack_m_s", 0.0))
        if maximum_speed > speed_limit:
            failures.append(Failure("$.fixture.final",
                                    "endpoint speed exceeds authored maximum",
                                    maximum_speed, speed_limit))

    return {
        "schema": REPORT_SCHEMA,
        "scope": "analytic_invariants",
        "profile_id": profile.get("id"),
        "profile_sha256": expected_hash,
        "driver": driver,
        "passed": not failures,
        "fixture_count": 1 if actual else 0,
        "sample_count": len(final_cloth),
        "failure_count": len(failures),
        "observations": {
            "maximum_overlap_penetration_m": final_penetration,
            "cloth_contact_impulse_n_s": list(cloth_impulse)
                if cloth_impulse is not None else None,
            "rigid_contact_impulse_n_s": list(rigid_impulse)
                if rigid_impulse is not None else None,
            "contact_impulse_activity_n_s": impulse_activity,
            "reciprocal_impulse_residual_n_s": reciprocal_residual,
            "expected_normal_impulse_n_s": expected_impulse,
            "expected_impulse_error_n_s": expected_error,
            "cloth_force_impulse_n_s": list(force_impulse)
                if force_impulse is not None else None,
            "force_impulse_residual_n_s": force_error,
            "tangential_impulse_n_s": tangential_impulse,
            "final_frictionless_slip_m_s": final_slip,
            "final_closing_speed_m_s": final_closing,
            "orientation_drift": orientation_drift,
            "energy_increase_j": energy_increase,
            "maximum_endpoint_speed_m_s": maximum_speed,
        },
        "failures": [asdict(failure) for failure in failures],
    }


def evaluate(profile: dict[str, Any], result: dict[str, Any],
             expected_hash: str) -> dict[str, Any]:
    failures: list[Failure] = []
    if profile.get("schema") != PROFILE_SCHEMA:
        failures.append(Failure("$.profile.schema", "unsupported profile schema",
                                profile.get("schema"), PROFILE_SCHEMA))
    evaluation = profile.get("evaluation", "system_exchange")
    if evaluation == "rigid_kinematic_sweep":
        return _evaluate_rigid_kinematic_sweep(
            profile, result, expected_hash, failures)
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

    if evaluation == "maximum_pairwise_penetration":
        return _evaluate_penetration(profile, result, expected_hash, driver, failures)
    if evaluation == "rigid_attachment":
        return _evaluate_rigid_attachment(
            profile, result, expected_hash, driver, failures)
    if evaluation in ("fluid_self_contact", "fluid_self_pressure",
                      "fluid_self_viscosity"):
        return _evaluate_fluid_self_pair(
            profile, result, expected_hash, driver, failures,
            mode=evaluation.removeprefix("fluid_self_"))
    if evaluation == "rope_self_contact":
        return _evaluate_rope_self_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "fluid_cloth_containment":
        return _evaluate_fluid_cloth_containment(
            profile, result, expected_hash, driver, failures)
    if evaluation == "fluid_soft_contact":
        return _evaluate_fluid_soft_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "cloth_soft_contact":
        return _evaluate_cloth_soft_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "cloth_rigid_contact":
        return _evaluate_cloth_rigid_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "rope_rigid_contact":
        return _evaluate_rope_rigid_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "rope_rigid_attachment":
        return _evaluate_rope_rigid_attachment(
            profile, result, expected_hash, driver, failures)
    if evaluation == "fluid_rope_contact":
        return _evaluate_fluid_rope_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_self_pressure":
        return _evaluate_smoke_self_pressure(
            profile, result, expected_hash, driver, failures)
    if evaluation == "cloth_smoke_contact":
        return _evaluate_cloth_smoke_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "soft_smoke_contact":
        return _evaluate_soft_smoke_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "rope_smoke_contact":
        return _evaluate_rope_smoke_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "fluid_cloth_contact":
        return _evaluate_fluid_cloth_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "fluid_rigid_contact":
        return _evaluate_fluid_rigid_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_rigid_contact":
        return _evaluate_smoke_rigid_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "soft_rigid_contact":
        return _evaluate_soft_rigid_contact(
            profile, result, expected_hash, driver, failures)
    if evaluation == "fluid_rigid_containment":
        return _evaluate_fluid_rigid_containment(
            profile, result, expected_hash, driver, failures)
    if evaluation == "cloth_rope_attachment":
        return _evaluate_cloth_rope_attachment(
            profile, result, expected_hash, driver, failures)
    if evaluation == "soft_rope_attachment":
        return _evaluate_soft_rope_attachment(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_self_viscosity":
        return _evaluate_smoke_self_viscosity(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_fluid_drag":
        return _evaluate_smoke_fluid_drag(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_cloth_drag":
        return _evaluate_smoke_cloth_drag(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_soft_drag":
        return _evaluate_smoke_soft_drag(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_rope_drag":
        return _evaluate_smoke_rope_drag(
            profile, result, expected_hash, driver, failures)
    if evaluation == "smoke_rigid_drag":
        return _evaluate_smoke_rigid_drag(
            profile, result, expected_hash, driver, failures)
    if evaluation == "fluid_smoke_phase":
        return _evaluate_fluid_smoke_phase(
            profile, result, expected_hash, driver, failures)
    if evaluation == "externally_heated_fluid":
        return _evaluate_externally_heated_fluid(
            profile, result, expected_hash, driver, failures)
    if evaluation != "system_exchange":
        failures.append(Failure("$.profile.evaluation",
                                "unsupported quality evaluation", evaluation))

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
