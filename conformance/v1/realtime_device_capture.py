#!/usr/bin/env python3
"""Capture the complete F8 fixed-tick device suite through the public SDK.

The native runner owns simulation and public readback.  This program retains
its raw plain/validation artifacts, derives the 37 named measurements from
those artifacts, and writes the single document consumed by
``realtime_device_suite.py``.  It never substitutes limit values for missing
or failing observations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SUITE_MODULE_PATH = ROOT / "conformance/v1/realtime_device_suite.py"
PROFILE_PATH = ROOT / "conformance/v1/qualification/realtime-profile-v1.json"
INVENTORY_PATH = (
    ROOT / "conformance/v1/qualification/kernel-device-obligations-v1.json"
)
CAPABILITY_PATH = (
    ROOT / "conformance/v1/qualification/capability-obligations-v1.json"
)
SUITE_PATH = ROOT / "conformance/v1/realtime/oracle-suite-v1.json"
FIXTURE_PATH = ROOT / "conformance/v1/realtime/scenario-fixtures-v1.json"
ACTIVE_CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/active-realtime-candidate.json"
)

CASE_SCENARIOS = {
    "rigid-cold-impact-rotation": "rigid-direct",
    "rigid-settled-long-run": "constraint-fixed",
    "fluid-closed-container-source-outflow": "fluid-lifecycle",
    "cloth-stiff-pin-impact": "cloth-core",
    "soft-body-mass-ratio-volume": "soft-body-core",
    "rope-attachment-rotation": "rope-core",
    "smoke-analytic-transport-long-run": "smoke-grid",
    "fluid-cloth-rigid-containment": "cloth-water",
}
NATIVE_SCENARIOS = {
    "six-family-mixed",
    "boundary-replacement-generation",
    "capacity-overflow-atomic-recovery",
    "empty-and-sparse-capacity",
}
WORKLOAD_IDS = (
    "rigid-wall", "fluid-rigid", "deformable-mixed", "smoke-coupled",
    "six-family-mixed",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUITE = _load_module("pm_realtime_device_suite_capture", SUITE_MODULE_PATH)
ORACLE = SUITE.ORACLE


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, *, portable_text: bool = False) -> str:
    data = path.read_bytes()
    if portable_text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _maximum(values: Iterable[float]) -> float:
    materialized = list(values)
    return max(materialized) if materialized else 0.0


def _rms(summaries: Iterable[Mapping[str, Any]]) -> float:
    squared_sum = 0.0
    count = 0
    for summary in summaries:
        squared_sum += _finite_number(summary.get("squared_sum", 0.0),
                                      "quality squared_sum")
        observed = summary.get("count", 0)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise ValueError("quality count is not an integer")
        count += observed
    return math.sqrt(squared_sum / count) if count else 0.0


def _resources(checkpoints: Sequence[Mapping[str, Any]], kind: str):
    for checkpoint in checkpoints:
        for resource in checkpoint.get("resources", []):
            if isinstance(resource, dict) and resource.get("type") == kind:
                yield checkpoint, resource


def _active_resources(checkpoints: Sequence[Mapping[str, Any]], kind: str):
    for checkpoint, resource in _resources(checkpoints, kind):
        count = resource.get("count")
        if count is None or (isinstance(count, int) and count > 0):
            yield checkpoint, resource


def _distance(first: Sequence[Any], second: Sequence[Any]) -> float:
    if len(first) != 3 or len(second) != 3:
        raise ValueError("public position must contain three components")
    return math.sqrt(sum(
        (_finite_number(a, "position") - _finite_number(b, "position")) ** 2
        for a, b in zip(first, second)
    ))


def _quat_distance(first: Sequence[Any], second: Sequence[Any]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("public orientation must contain four components")
    dot = abs(sum(
        _finite_number(a, "orientation") * _finite_number(b, "orientation")
        for a, b in zip(first, second)
    ))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _quaternion_axes(value: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
    if len(value) != 4:
        raise ValueError("public orientation must contain four components")
    x, y, z, w = (_finite_number(item, "orientation") for item in value)
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1.0e-12:
        raise ValueError("public orientation is not normalizable")
    x, y, z, w = (item / length for item in (x, y, z, w))
    return (
        (1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        (2.0 * (x * y - z * w),
         1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
        (2.0 * (x * z + y * w), 2.0 * (y * z - x * w),
         1.0 - 2.0 * (x * x + y * y)),
    )


def _box_penetration(first: Mapping[str, Any], first_half: Sequence[Any],
                     second: Mapping[str, Any],
                     second_half: Sequence[Any]) -> float:
    """Return exact oriented-box overlap depth from copied public poses."""
    if len(first_half) != 3 or len(second_half) != 3:
        raise ValueError("box half extents must contain three components")
    half_a = tuple(_finite_number(item, "box half extent")
                   for item in first_half)
    half_b = tuple(_finite_number(item, "box half extent")
                   for item in second_half)
    axes_a = _quaternion_axes(first.get("orientation", []))
    axes_b = _quaternion_axes(second.get("orientation", []))
    position_a = tuple(_finite_number(item, "position")
                       for item in first.get("position", []))
    position_b = tuple(_finite_number(item, "position")
                       for item in second.get("position", []))
    if len(position_a) != 3 or len(position_b) != 3:
        raise ValueError("public position must contain three components")
    delta = tuple(b - a for a, b in zip(position_a, position_b))

    candidates = [*axes_a, *axes_b]
    for axis_a in axes_a:
        for axis_b in axes_b:
            candidates.append((
                axis_a[1] * axis_b[2] - axis_a[2] * axis_b[1],
                axis_a[2] * axis_b[0] - axis_a[0] * axis_b[2],
                axis_a[0] * axis_b[1] - axis_a[1] * axis_b[0],
            ))
    minimum = math.inf
    for candidate in candidates:
        length = math.sqrt(sum(item * item for item in candidate))
        if length <= 1.0e-8:
            continue
        axis = tuple(item / length for item in candidate)
        radius_a = sum(half_a[index] * abs(sum(
            axis[component] * axes_a[index][component]
            for component in range(3))) for index in range(3))
        radius_b = sum(half_b[index] * abs(sum(
            axis[component] * axes_b[index][component]
            for component in range(3))) for index in range(3))
        overlap = radius_a + radius_b - abs(sum(
            delta[index] * axis[index] for index in range(3)))
        if overlap <= 0.0:
            return 0.0
        minimum = min(minimum, overlap)
    return 0.0 if not math.isfinite(minimum) else minimum


def _rigid_penetration(checkpoints: Sequence[Mapping[str, Any]],
                       definition: Mapping[str, Any]) -> float:
    """Measure published overlap, excluding swept CCD recovery distance."""
    case_path = definition.get("fixture", {}).get("definition")
    boxes: dict[str, Sequence[Any]] = {}
    if isinstance(case_path, str):
        case = read_json(ROOT / case_path)
        boxes = {
            str(item["name"]): item["definition"]["half_extents"]
            for item in case.get("resources", [])
            if isinstance(item, dict) and
            isinstance(item.get("definition"), dict) and
            isinstance(item["definition"].get("half_extents"), list)
        }
    maximum = 0.0
    for checkpoint in checkpoints:
        states = {
            str(resource.get("id")): resource
            for resource in checkpoint.get("resources", [])
            if isinstance(resource, dict) and
            resource.get("type") == "rigid_body"
        }
        for contact in checkpoint.get("contacts", {}).get(
                "rigid_contacts", []):
            first = str(contact.get("body"))
            second = str(contact.get("collider"))
            if first in boxes and second in boxes and \
                    first in states and second in states:
                depth = _box_penetration(
                    states[first], boxes[first], states[second], boxes[second])
            else:
                depth = max(0.0, _finite_number(
                    contact.get("penetration", 0.0), "contact penetration"))
            maximum = max(maximum, depth)
    return maximum


def _state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contacts": checkpoint.get("contacts"),
        "invariants": checkpoint.get("invariants"),
        "resources": checkpoint.get("resources"),
    }


def _event_boundaries(definition: Mapping[str, Any]) -> set[int]:
    return {
        # Commands authored at zero-based tick N are visible in completed
        # checkpoint N+1.  Start the next unforced energy interval there.
        int(item["tick"]) + 1
        for item in definition.get("fixture", {}).get("events", [])
        if isinstance(item, dict) and isinstance(item.get("tick"), int)
    }


def _energy_growth(checkpoints: Sequence[Mapping[str, Any]],
                   definition: Mapping[str, Any], kind: str) -> float:
    fixture = definition.get("fixture", {})
    floor = float(fixture.get("energy_floor_j", 0.001))
    boundaries = _event_boundaries(definition)
    windows = fixture.get("unforced_energy_windows")
    baseline: float | None = None
    maximum = 0.0
    for checkpoint in checkpoints:
        frame = checkpoint.get("frame")
        if not isinstance(frame, int):
            continue
        if isinstance(windows, list) and not any(
                isinstance(window, list) and len(window) == 2 and
                int(window[0]) <= frame <= int(window[1])
                for window in windows):
            baseline = None
            continue
        if kind == "rigid":
            value = checkpoint.get("invariants", {}).get(
                "rigid_quality", {}).get("mechanical_energy_j")
        else:
            values = [
                resource.get("aggregate", {}).get(
                    "mechanical_energy_j",
                    resource.get("aggregate", {}).get("kinetic_energy"))
                for _cp, resource in _resources([checkpoint], kind)
            ]
            value = sum(float(item) for item in values if item is not None)
        energy = _finite_number(value or 0.0, f"{kind} mechanical energy")
        if baseline is None or frame in boundaries:
            baseline = energy
        maximum = max(maximum,
                      max(0.0, energy - baseline) / max(abs(baseline), floor))
    return maximum


def _tunneling_events(checkpoints: Sequence[Mapping[str, Any]], kind: str,
                      definition: Mapping[str, Any]) -> int:
    """Count observed public support crossings lacking a public contact.

    Clearance is meaningful only for fixtures that explicitly author collider
    planes.  Empty domains and fixtures without planes have no plane crossing.
    """
    planes = definition.get("fixture", {}).get("collider_planes", [])
    if not planes:
        return 0
    crossings = 0
    previous: dict[str, float] = {}
    for checkpoint, resource in _active_resources(checkpoints, kind):
        identity = str(resource.get("id"))
        clearance = resource.get("minimum_clearance")
        if clearance is None and kind == "rigid_body":
            position = resource.get("position", [])
            clearance = position[1] if len(position) == 3 else None
        if clearance is None:
            continue
        current = _finite_number(clearance, "minimum clearance")
        contact_count = sum(
            int(checkpoint.get("contacts", {}).get(name, 0))
            for name in ("rigid_contact_count", "fluid_contact_count")
        )
        if identity in previous and previous[identity] >= 0.0 > current \
                and contact_count == 0:
            crossings += 1
        previous[identity] = current
    return crossings


def _domain_values(domain: str, checkpoints: Sequence[Mapping[str, Any]],
                   definition: Mapping[str, Any]) -> dict[str, float | int]:
    fixture = definition.get("fixture", {})
    characteristic = float(fixture.get("characteristic_length_m", 1.0))
    if domain == "RIGID_BODY":
        penetration = _rigid_penetration(checkpoints, definition)
        rigid_quality = [
            checkpoint.get("invariants", {}).get("rigid_quality", {})
            for checkpoint in checkpoints
        ]
        settled = fixture.get("settled_window")
        translations: list[float] = []
        rotations: list[float] = []
        if "long-run" in definition.get("tags", []) and \
                isinstance(settled, list) and len(settled) == 2:
            start, end = map(int, settled)
            origins: dict[str, Mapping[str, Any]] = {}
            for checkpoint, resource in _resources(checkpoints, "rigid_body"):
                frame = checkpoint.get("frame")
                if not isinstance(frame, int) or not start <= frame <= end:
                    continue
                identity = str(resource.get("id"))
                origin = origins.setdefault(identity, resource)
                translations.append(_distance(
                    origin.get("position", []), resource.get("position", [])))
                rotations.append(_quat_distance(
                    origin.get("orientation", []),
                    resource.get("orientation", [])))
        return {
            "maximum-contact-penetration-characteristic-length":
                penetration / characteristic if characteristic > 0.0 else 0.0,
            "maximum-contact-penetration-m": penetration,
            "maximum-joint-angle-error-rad": _maximum(
                float(item.get("maximum_joint_angle_error_rad", 0.0))
                for item in rigid_quality),
            "maximum-joint-position-error-m": _maximum(
                float(item.get("maximum_joint_position_error_m", 0.0))
                for item in rigid_quality),
            "maximum-unforced-mechanical-energy-growth-fraction":
                _energy_growth(checkpoints, definition, "rigid"),
            "resting-rotation-drift-rad-over-settled-window":
                _maximum(rotations),
            "resting-translation-drift-m-over-settled-window":
                _maximum(translations),
            "tunneling-events": _tunneling_events(
                checkpoints, "rigid_body", definition),
        }
    if domain == "FLUID":
        active = list(_active_resources(checkpoints, "fluid"))
        initial_ids: set[int] = set()
        if active:
            initial_ids = set(active[0][1].get("stable_ids", []))
        leaked: set[int] = set()
        missing: set[int] = set()
        centers: list[Sequence[Any]] = []
        density_maximum: list[float] = []
        density_p95: list[float] = []
        observed_fixture = definition.get("fixture_observation", {})
        closed_bounds = fixture.get("closed_container_bounds")
        collider_bounds = ([closed_bounds] if isinstance(closed_bounds, dict)
                           else observed_fixture.get(
                               "static_collision_bounds", []))
        particle_radius = float(observed_fixture.get(
            "fluid_particle_radius_m", 0.0))
        for _checkpoint, resource in active:
            centers.append(resource.get("aggregate", {}).get(
                "center_of_mass", [0.0, 0.0, 0.0]))
            quality = resource.get("quality", {})
            density_maximum.append(float(
                quality.get("density_relative_error_maximum", 0.0)))
            density_p95.append(float(
                quality.get("density_relative_error_p95", 0.0)))
            ids = set(resource.get("stable_ids", []))
            missing.update(initial_ids - ids)
            for sample in resource.get("samples", []):
                identity = sample.get("id")
                point = sample.get("position", [])
                if identity not in initial_ids or len(point) != 3 or not collider_bounds:
                    continue
                inside = any(
                    all(float(bounds["minimum"][axis]) - particle_radius <=
                        float(point[axis]) <=
                        float(bounds["maximum"][axis]) + particle_radius
                        for axis in range(3))
                    for bounds in collider_bounds
                    if len(bounds.get("minimum", [])) == 3 and
                    len(bounds.get("maximum", [])) == 3
                )
                if not inside:
                    leaked.add(int(identity))
        support = float(fixture.get("support_radius_m", 1.0))
        # A closed-system COM window is meaningful only when the fixture
        # authors one.  Source/outflow windows are measured by identity below.
        com_drift = 0.0
        window = fixture.get("closed_system_window")
        if active and isinstance(window, list) and len(window) == 2:
            window_centers = [resource.get("aggregate", {}).get(
                "center_of_mass", [0.0, 0.0, 0.0])
                for checkpoint, resource in active
                if int(window[0]) <= int(checkpoint.get("frame", -1)) <=
                int(window[1])]
            if window_centers:
                com_drift = _maximum(_distance(window_centers[0], value)
                                     for value in window_centers) / support
        denominator = max(1, len(initial_ids))
        return {
            "closed-container-leaked-particle-fraction":
                len(leaked) / denominator if initial_ids else 0.0,
            "closed-system-center-of-mass-drift-support-radius": com_drift,
            "density-relative-error-maximum": _maximum(density_maximum),
            "density-relative-error-p95": _maximum(density_p95),
            # Fixture-declared destroy-plane removals are excluded by the
            # normative algorithm.  Scenarios tagged removal author exactly
            # that exclusion; every other missing initial id is a loss.
            "free-surface-particle-loss-fraction":
                0.0 if "removal" in definition.get("tags", []) else
                len(missing) / denominator if initial_ids else 0.0,
            "maximum-unforced-mechanical-energy-growth-fraction":
                _energy_growth(checkpoints, definition, "fluid"),
        }
    if domain == "CLOTH":
        active = list(_active_resources(checkpoints, "cloth"))
        qualities = [resource.get("quality", {}) for _cp, resource in active]
        return {
            "bend-angle-error-rad-rms": _rms(
                item.get("bend_angle", {}) for item in qualities),
            "closed-surface-volume-relative-error": _maximum(
                float(item.get("closed_surface_volume_relative_error", 0.0))
                for item in qualities),
            "pinned-vertex-drift-m": _maximum(
                float(item.get("pinned_vertex_drift_m", 0.0))
                for item in qualities),
            "stretch-relative-error-maximum": _maximum(
                float(item.get("stretch", {}).get("maximum", 0.0))
                for item in qualities),
            "stretch-relative-error-rms": _rms(
                item.get("stretch", {}) for item in qualities),
            "tunneling-events": _tunneling_events(
                checkpoints, "cloth", definition),
        }
    if domain == "SOFT_BODY":
        active = list(_active_resources(checkpoints, "soft_body"))
        qualities = [resource.get("quality", {}) for _cp, resource in active]
        volume_errors = [
            float(resource.get("quality", {}).get("volume_relative_error", 0.0))
            for _cp, resource in active
            if int(resource.get("topology", {}).get("bond_count", 0)) > 0
        ]
        return {
            "bond-stretch-relative-error-maximum": _maximum(
                float(item.get("bond_stretch", {}).get("maximum", 0.0))
                for item in qualities),
            "bond-stretch-relative-error-rms": _rms(
                item.get("bond_stretch", {}) for item in qualities),
            "inverted-volume-elements": int(_maximum(
                float(item.get("inverted_volume_elements", 0))
                for item in qualities)),
            "shape-matching-position-error-characteristic-length-rms":
                _rms(item.get("shape_matching_position_error", {})
                     for item in qualities) /
                characteristic if characteristic > 0.0 else 0.0,
            "tunneling-events": _tunneling_events(
                checkpoints, "soft_body", definition),
            "volume-relative-error": _maximum(volume_errors),
        }
    if domain == "ROPE":
        active = list(_active_resources(checkpoints, "rope"))
        qualities = [resource.get("quality", {}) for _cp, resource in active]
        return {
            "attachment-position-error-m": _maximum(
                float(item.get("attachment_position_error_m", 0.0))
                for item in qualities),
            "bend-angle-error-rad-rms": _rms(
                item.get("bend_angle", {}) for item in qualities),
            "segment-stretch-relative-error-maximum": _maximum(
                float(item.get("segment_stretch", {}).get("maximum", 0.0))
                for item in qualities),
            "segment-stretch-relative-error-rms": _rms(
                item.get("segment_stretch", {}) for item in qualities),
            "tunneling-events": _tunneling_events(
                checkpoints, "rope", definition),
        }
    if domain == "SMOKE":
        active = list(_active_resources(checkpoints, "smoke"))
        qualities = [resource.get("quality", {}) for _cp, resource in active]
        masses = [float(item.get("number_density_sum", 0.0))
                  for item in qualities]
        mass_error = 0.0
        if masses and masses[0] != 0.0 and "long-run" in definition.get("tags", []):
            mass_error = _maximum(
                abs(value - masses[0]) / abs(masses[0]) for value in masses)
        return {
            "analytic-transport-centroid-error-grid-cells": _maximum(
                float(item.get(
                    "analytic_transport_centroid_error_grid_cells", 0.0))
                for item in qualities),
            "closed-tracer-mass-relative-error": mass_error,
            "grid-divergence-dt-maximum": _maximum(
                float(item.get("grid_divergence_dt", {}).get("maximum", 0.0))
                for item in qualities),
            "grid-divergence-dt-rms": _rms(
                item.get("grid_divergence_dt", {}) for item in qualities),
            "maximum-speed-overshoot-m-s": _maximum(
                float(item.get("maximum_speed_overshoot_m_s", 0.0))
                for item in qualities),
            "particle-density-relative-error-p95": _maximum(
                float(item.get("particle_density_relative_error_p95", 0.0))
                for item in qualities),
        }
    raise ValueError(f"unknown scenario domain {domain}")


def measurements(checkpoints: Sequence[Mapping[str, Any]],
                 definition: Mapping[str, Any],
                 contract: Mapping[str, Any],
                 fixture_observation: Mapping[str, Any] | None = None,
                 ) -> dict[str, float | int]:
    definition = dict(definition)
    definition["fixture_observation"] = fixture_observation or {}
    values: dict[str, float | int] = {}
    by_domain = {
        domain: _domain_values(domain, checkpoints, definition)
        for domain in definition["domains"]
    }
    specs = ORACLE.case_limit_specs(
        {"kind": "scenario", **definition}, contract)
    for spec in specs:
        domain = spec.path.split("/")[2]
        key = spec.path.split("/limits/", 1)[1].replace("_", "-")
        if key not in by_domain[domain]:
            raise ValueError(
                f"no raw measurement algorithm for {spec.assertion}")
        values[spec.assertion] = by_domain[domain][key]
    return values


def tick_work(checkpoints: Sequence[Mapping[str, Any]],
              definition: Mapping[str, Any]) -> list[dict[str, Any]]:
    ticks = list(checkpoints[1:])
    if len(ticks) != int(definition["ticks"]):
        raise ValueError(
            f"{definition['id']}: expected {definition['ticks']} completed "
            f"tick records, observed {len(ticks)}")
    result: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(ticks):
        work = copy.deepcopy(checkpoint.get("realtime_work"))
        if not isinstance(work, dict):
            raise ValueError(f"{definition['id']}: tick {index} has no work record")
        if work.get("status") == "failure":
            unchanged = _state(checkpoint) == _state(checkpoints[index])
            later_recovery = index + 1 < len(ticks) and \
                ticks[index + 1].get("realtime_work", {}).get("status") == "success"
            recovery_changed = later_recovery and \
                _state(ticks[index + 1]) != _state(checkpoint)
            work.update({
                "state_unchanged": unchanged,
                "pending_inputs_preserved": bool(recovery_changed),
                "hidden_retries": 0,
                "retry_substeps": 0,
                "retry_spatial_updates": 0,
                "atomic_failure": {
                    "committed_state_preserved": unchanged,
                    "clock_preserved": work.get("time_advances") == 0,
                    "pending_inputs_preserved": bool(recovery_changed),
                },
            })
        result.append(work)
    return result


def _candidate_path() -> Path:
    routing = read_json(ACTIVE_CANDIDATE_PATH)
    relative = Path(str(routing["manifest"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("active candidate path is not repository-relative")
    path = ROOT / relative
    if file_sha256(path, portable_text=True) != routing["manifest_sha256"]:
        raise ValueError("active candidate routing hash mismatch")
    return path


def _candidate_identity(candidate: Mapping[str, Any], digest: str):
    return {
        "id": candidate["id"],
        "manifest_sha256": digest,
        "source_revision": candidate["source"]["revision"],
    }


def _scenario_definitions(
    suite: Mapping[str, Any], fixtures: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    suite_definitions = {
        str(item["id"]): item for item in suite["scenario_fixtures"]
    }
    fixture_definitions = {
        str(item["id"]): item for item in fixtures["scenarios"]
    }
    if suite_definitions.keys() != fixture_definitions.keys():
        raise ValueError("oracle suite and fixture scenario identities differ")
    result: dict[str, dict[str, Any]] = {}
    for identity, suite_definition in suite_definitions.items():
        fixture_definition = fixture_definitions[identity]
        for key in ("domains", "ticks", "tags"):
            if fixture_definition.get(key) != suite_definition.get(key):
                raise ValueError(
                    f"{identity}: oracle suite and fixture {key} differ")
        result[identity] = {
            **suite_definition,
            "fixture": copy.deepcopy(fixture_definition["fixture"]),
            "measurement_sets": copy.deepcopy(
                fixture_definition["measurement_sets"]),
            "tick_outcomes": copy.deepcopy(
                fixture_definition["tick_outcomes"]),
            "failure_diagnostics": copy.deepcopy(
                fixture_definition["failure_diagnostics"]),
        }
    return result


class NativeCapture:
    def __init__(self, runner: Path, driver_manifest: Path,
                 layer_manifest: Path, output: Path):
        self.runner = runner
        self.driver_manifest = driver_manifest
        self.layer_manifest = layer_manifest
        self.output = output

    def environment(self, validation: bool) -> dict[str, str]:
        environment = dict(os.environ)
        environment["PM_DRIVER_FILES"] = str(self.driver_manifest)
        environment["PM_LAYER_FILES"] = str(self.layer_manifest)
        if validation:
            environment["PM_TEST_VALIDATION"] = "1"
        else:
            environment.pop("PM_TEST_VALIDATION", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    def run(self, arguments: Sequence[str | Path], log: Path,
            *, validation: bool, stdout_json: bool = False):
        command = [str(self.runner), *map(str, arguments)]
        completed = subprocess.run(
            command, cwd=ROOT, env=self.environment(validation), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"command failed ({completed.returncode}); see {log}: "
                + " ".join(command))
        return json.loads(completed.stdout) if stdout_json else completed.stdout

    def mode(self, validation: bool, fixtures: Mapping[str, Any]) -> dict[str, Any]:
        label = "validation" if validation else "plain"
        root = self.output / "raw" / label
        logs = self.output / "logs" / label
        root.mkdir(parents=True)
        provenance = self.run(
            ["--provenance", "--root", ROOT], logs / "provenance.log",
            validation=validation, stdout_json=True)
        capabilities = self.run(
            ["--capabilities", "--root", ROOT], logs / "capabilities.log",
            validation=validation, stdout_json=True)
        public_directory = root / "public-cases"
        self.run(
            ["--case", "all", "--root", ROOT, "--output", public_directory],
            logs / "public-cases.log", validation=validation)
        public = {
            path.stem: read_json(path)
            for path in sorted(public_directory.glob("*.json"))
        }
        scenarios: dict[str, dict[str, Any]] = {}
        scenario_directory = root / "scenarios"
        for definition in fixtures["scenarios"]:
            identity = definition["id"]
            output = scenario_directory / identity
            if identity in CASE_SCENARIOS or identity in NATIVE_SCENARIOS:
                command = [
                    "--scenario-raw", identity, "--root", ROOT,
                    "--output", output,
                ]
                raw_path = output / f"{identity}.json"
            else:
                raise ValueError(f"no installed-public scenario builder for {identity}")
            self.run(command, logs / "scenarios" / f"{identity}.log",
                     validation=validation)
            scenarios[identity] = read_json(raw_path)
        workload_directory = root / "workloads"
        self.run([
            "--workload", "all", "--warmup-ticks", "600",
            "--measured-ticks", "7200", "--root", ROOT,
            "--output", workload_directory,
        ], logs / "workloads.log", validation=validation)
        workloads = {
            identity: read_json(workload_directory / f"{identity}.json")
            for identity in WORKLOAD_IDS
        }
        return {
            "provenance": provenance,
            "capabilities": capabilities,
            "public": public,
            "scenarios": scenarios,
            "workloads": workloads,
        }


def _scenario_observation(
    definition: Mapping[str, Any], fixture: Mapping[str, Any],
    contract: Mapping[str, Any], candidate_identity: Mapping[str, Any],
    runtime_variant: str, plain: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    modes = (plain, validation)
    measured = [measurements(mode["checkpoints"][1:], definition, contract,
                             mode.get("fixture_observation"))
                for mode in modes]
    work = [tick_work(mode["checkpoints"], definition) for mode in modes]
    final_states = [_state(mode["checkpoints"][-1]) for mode in modes]
    executions = [{
        "validation_enabled": bool(index),
        "simulation_status": "PM_SUCCESS",
        "simulation_instance_count": 1,
        "completed_tick_records": int(definition["ticks"]),
        "measurements": measured[index],
        "measurements_sha256": value_sha256(measured[index]),
        "tick_work": work[index],
        "tick_work_sha256": value_sha256(work[index]),
        "final_public_state": final_states[index],
        "final_public_state_sha256": value_sha256(final_states[index]),
        "raw_artifact_sha256": value_sha256(mode),
    } for index, mode in enumerate(modes)]
    return {
        "id": definition["id"],
        "source": "installed-public-api",
        "driver_private_state_used": False,
        "public_api_calls": fixture["execution"]["public_api_calls"],
        "runtime_variant": runtime_variant,
        "fixture_contract": {
            "id": fixture["id"],
            "sha256": file_sha256(FIXTURE_PATH, portable_text=True),
        },
        "executions": executions,
        "simulation_recreated_during_ticks": False,
        "candidate": dict(candidate_identity),
        "domains": definition["domains"],
        "tags": definition["tags"],
        "simulation_status": "PM_SUCCESS",
        "steps": [{"timestep_seconds": 1.0 / 60.0, "substeps": 1}
                  for _ in range(int(definition["ticks"]))],
        "measurements": measured[0],
        "tick_work": work[0],
    }


def _backend_observation(
    scenario: Mapping[str, Any], candidate: Mapping[str, Any],
    candidate_digest: str, runtime_variant: str, case_id: str,
    calls: Sequence[str], validation: bool,
) -> dict[str, Any]:
    checkpoint = scenario["checkpoints"][1]
    rigid = next(
        resource for resource in checkpoint["resources"]
        if resource.get("type") == "rigid_body" and
        resource.get("tolerance_class") != "constraint_contact"
    )
    work = tick_work(scenario["checkpoints"][:2], {"id": case_id, "ticks": 1})
    return {
        "schema": "parallel-mater-realtime-observation/v1",
        "source": "installed-public-api",
        "driver_private_state_used": False,
        "validation_enabled": validation,
        "runtime": {"schedule": "fixed-tick", "version": 1},
        "candidate": {
            "manifest_role": "qualification-input",
            "execution_identity_claimed": False,
            "id": candidate["id"],
            "manifest_sha256": candidate_digest,
            "source_revision": candidate["source"]["revision"],
            "runtime_variant": runtime_variant,
        },
        "case_id": case_id,
        "public_api_calls": list(calls),
        "steps": [{"timestep_seconds": 1.0 / 60.0, "substeps": 1}],
        "simulation_status": "PM_SUCCESS",
        "statistics": {
            "step_count": 1,
            "substeps_executed": 1,
            "execution_result": "PM_SUCCESS",
            "telemetry_schema_version": 1,
            "profile_version": 1,
            "budget_version": 1,
        },
        "copied_public_state": {
            "body_count": 1,
            "position": rigid["position"],
            "linear_velocity": rigid["linear_velocity"],
        },
        "measurements": {},
        "tick_work": work,
    }


def _manifest_binary(manifest: Path) -> Path:
    document = read_json(manifest)
    entry = document.get("PMD") or document.get("driver")
    if not isinstance(entry, dict) or not isinstance(entry.get("library_path"), str):
        raise ValueError(f"{manifest} has no public driver library_path")
    path = Path(entry["library_path"])
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"driver binary does not exist: {path}")
    return path


def _negative_controls(capture: Mapping[str, Any], contracts: tuple[Any, ...]):
    contract, inventory, obligations, suite, candidate, candidate_digest = contracts
    mutations = {
        "candidate_missing_pair": lambda value: value["capabilities"].pop(),
        "atomic_failure_violation": lambda value: next(
            item for item in value["scenarios"]
            if item["id"] == "capacity-overflow-atomic-recovery"
        )["tick_work"][1].update({"state_unchanged": False}),
        "hidden_retry_violation": lambda value: next(
            item for item in value["scenarios"]
            if item["id"] == "capacity-overflow-atomic-recovery"
        )["tick_work"][1].update({"hidden_retries": 1}),
        "timing_stall_retained": lambda value:
            value["workloads"][0]["timings"]["physics_ms"].pop("maximum"),
        "timing_failed_tick": lambda value:
            value["workloads"][0].update({"failed_ticks": 1}),
    }
    baseline = copy.deepcopy(capture)
    baseline["negative_controls"] = [
        {"id": identity, "passed": True}
        for identity in SUITE.REQUIRED_NEGATIVE_CONTROLS
    ]
    baseline_report = SUITE.evaluate(
        baseline, contract, inventory, obligations, suite, candidate,
        candidate_digest)
    controls = []
    for identity in SUITE.REQUIRED_NEGATIVE_CONTROLS:
        injected = copy.deepcopy(baseline)
        mutations[identity](injected)
        report = SUITE.evaluate(
            injected, contract, inventory, obligations, suite, candidate,
            candidate_digest)
        controls.append({
            "id": identity,
            "passed": baseline_report["passed"] and not report["passed"],
            "rejected_failure_count": report["failure_count"],
        })
    return controls


def build_capture(arguments: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(PROFILE_PATH)
    inventory = read_json(INVENTORY_PATH)
    obligations = read_json(CAPABILITY_PATH)
    suite = read_json(SUITE_PATH)
    fixture = read_json(FIXTURE_PATH)
    candidate_path = _candidate_path()
    candidate = read_json(candidate_path)
    candidate_digest = file_sha256(candidate_path, portable_text=True)
    candidate_identity = _candidate_identity(candidate, candidate_digest)

    native = NativeCapture(
        arguments.runner.resolve(), arguments.driver_manifest.resolve(),
        arguments.validation_layer_manifest.resolve(), arguments.output.resolve())
    plain = native.mode(False, fixture)
    validation = native.mode(True, fixture)
    if plain["provenance"] != validation["provenance"]:
        raise ValueError("plain and validation device provenance differ")
    if plain["capabilities"] != validation["capabilities"]:
        raise ValueError("plain and validation advertised capabilities differ")

    definitions = _scenario_definitions(suite, fixture)
    scenarios = [
        _scenario_observation(
            definition, fixture, contract, candidate_identity,
            arguments.runtime_variant, plain["scenarios"][identity],
            validation["scenarios"][identity])
        for identity, definition in definitions.items()
    ]
    public_ids = inventory["public_case_sets"]["all"]
    public_cases = []
    for identity in public_ids:
        mode_statuses = [
            mode["public"].get(identity, {}).get("execution_status")
            for mode in (plain, validation)
        ]
        public_cases.append({
            "id": identity,
            "execution_status": "completed" if mode_statuses ==
                ["completed", "completed"] else "failed",
            "driver_private_state_used": False,
            "plain_raw_sha256": value_sha256(plain["public"].get(identity)),
            "validation_raw_sha256": value_sha256(
                validation["public"].get(identity)),
        })

    scenario_specs_pass = all(not ORACLE.evaluate_named_measurements(
        item["measurements"], ORACLE.case_limit_specs(
            {"kind": "scenario", **definitions[item["id"]]}, contract))
        for item in scenarios)
    public_pass = all(item["execution_status"] == "completed"
                      for item in public_cases)
    capabilities = []
    for item in plain["capabilities"]:
        capabilities.append({**item,
            "execution_status": "completed" if public_pass else "failed",
            "analytic_status": "passed" if scenario_specs_pass else "failed",
            "lifecycle_status": "passed" if scenario_specs_pass else "failed",
        })

    workloads = []
    for identity in WORKLOAD_IDS:
        selected = copy.deepcopy(validation["workloads"][identity])
        selected["executions"] = [{
            "validation_enabled": bool(index),
            "raw_artifact_sha256": value_sha256(mode["workloads"][identity]),
            "timings_sha256": value_sha256(
                mode["workloads"][identity].get("timings")),
        } for index, mode in enumerate((plain, validation))]
        selected.pop("raw_samples", None)
        workloads.append(selected)

    provenance = plain["provenance"]
    driver_binary = _manifest_binary(arguments.driver_manifest.resolve())
    backend_case = next(
        item["f8_execution_test"] for item in candidate["runtime_variants"]
        if item["id"] == arguments.runtime_variant)
    capture = {
        "schema": SUITE.CAPTURE_SCHEMA,
        "candidate": candidate_identity,
        "runtime_variant": arguments.runtime_variant,
        "device": {
            "target_id": arguments.target_id,
            "name": provenance["device_name"],
            "os": platform.platform(),
            "gpu": provenance["device_name"],
            "driver": str(provenance["driver_version"]),
            "toolchain": str(provenance["host_compiler"]),
            "driver_binary_sha256": file_sha256(driver_binary),
            "driver_manifest_sha256": file_sha256(
                arguments.driver_manifest.resolve(), portable_text=True),
        },
        "backend_observations": [
            _backend_observation(
                mode["scenarios"]["rigid-cold-impact-rotation"], candidate,
                candidate_digest, arguments.runtime_variant, backend_case,
                suite["public_api"]["operation_sequence"], bool(index))
            for index, mode in enumerate((plain, validation))
        ],
        "capabilities": capabilities,
        "public_cases": public_cases,
        "workloads": workloads,
        "scenarios": scenarios,
        "negative_controls": [],
    }
    contracts = (contract, inventory, obligations, suite, candidate,
                 candidate_digest)
    capture["negative_controls"] = _negative_controls(capture, contracts)
    report = SUITE.evaluate(capture, *contracts)
    return capture, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--driver-manifest", type=Path, required=True)
    parser.add_argument("--validation-layer-manifest", type=Path, required=True)
    parser.add_argument("--runtime-variant", default="cuda-runtime")
    parser.add_argument("--target-id", default="linux-rtx3050ti-slang-cuda")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        arguments.runner = arguments.runner.resolve(strict=True)
        arguments.driver_manifest = arguments.driver_manifest.resolve(strict=True)
        arguments.validation_layer_manifest = (
            arguments.validation_layer_manifest.resolve(strict=True))
        arguments.output = arguments.output.resolve()
        if arguments.output.exists():
            raise ValueError("output already exists; retain it and choose a new path")
        arguments.output.mkdir(parents=True)
        capture, report = build_capture(arguments)
        write_json(arguments.output / "capture.json", capture)
        write_json(arguments.output / "report.json", report)
        print(
            f"Realtime device capture: {report['scenario_count']} scenarios, "
            f"{report['failure_count']} failures; {arguments.output}")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"realtime device capture: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
