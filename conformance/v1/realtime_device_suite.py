#!/usr/bin/env python3
"""Validate one physical device against the frozen F8 fixed-tick suite.

This validator is part of the frozen public-oracle closure.  It connects
device-produced observations to that closure without changing the candidate,
physics, limits, or historical variable-timestep profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = ROOT / "conformance/v1/realtime_oracle.py"
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


def active_candidate_path() -> Path:
    routing = json.loads(ACTIVE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    relative = Path(str(routing["manifest"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("active candidate manifest must be repository-relative")
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"active candidate manifest does not exist: {relative}")
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != routing.get("manifest_sha256"):
        raise ValueError("active candidate manifest hash does not match routing")
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("id") != routing.get("candidate_id"):
        raise ValueError("active candidate id does not match routing")
    return path


CANDIDATE_PATH = active_candidate_path()

CAPTURE_SCHEMA = "parallel-mater-realtime-device-suite-capture/v1"
REPORT_SCHEMA = "parallel-mater-realtime-device-suite-report/v1"
REQUIRED_NEGATIVE_CONTROLS = (
    "candidate_missing_pair",
    "atomic_failure_violation",
    "hidden_retry_violation",
    "timing_stall_retained",
    "timing_failed_tick",
)
CAPABILITY_FIELDS = (
    "first_family", "second_family", "scope", "mode", "direction",
    "material_profile",
)


def _load_oracle():
    spec = importlib.util.spec_from_file_location("pm_realtime_oracle", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen realtime oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORACLE = _load_oracle()


@dataclass(frozen=True)
class Failure:
    path: str
    message: str
    actual: Any = None
    expected: Any = None


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_identity(candidate: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "manifest_sha256": digest,
        "source_revision": candidate.get("source", {}).get("revision"),
    }


def _failure(path: str, message: str, actual: Any = None,
             expected: Any = None) -> Failure:
    return Failure(path, message, actual, expected)


def _validate_candidate(
    value: Any,
    expected: Mapping[str, Any],
    path: str,
    failures: list[Failure],
) -> None:
    if value != expected:
        failures.append(_failure(
            path, "candidate identity is not exact", value, expected))


def _validate_step(step: Any, path: str, failures: list[Failure]) -> None:
    if not isinstance(step, dict):
        failures.append(_failure(path, "step must be an object", step))
        return
    timestep = step.get("timestep_seconds")
    if (isinstance(timestep, bool) or
            not isinstance(timestep, (int, float)) or
            not math.isfinite(float(timestep)) or
            ORACLE._binary32_ulp_distance(float(timestep), 1.0 / 60.0) > 1):
        failures.append(_failure(
            path + ".timestep_seconds",
            "every physical step must use the frozen binary32 1/60 timestep",
            timestep, 1.0 / 60.0))
    if step.get("substeps") != 1:
        failures.append(_failure(
            path + ".substeps", "every tick must contain one substep",
            step.get("substeps"), 1))


def _validate_tick_work(
    ticks: Any,
    expected_count: int,
    path: str,
    contract: Mapping[str, Any],
    fixture: Mapping[str, Any],
    failures: list[Failure],
) -> None:
    if not isinstance(ticks, list) or len(ticks) != expected_count:
        failures.append(_failure(
            path, "one public work record is required for every scenario tick",
            len(ticks) if isinstance(ticks, list) else ticks, expected_count))
        return
    outcomes = fixture.get("tick_outcomes", {})
    overrides = {
        item.get("tick"): item
        for item in outcomes.get("overrides", [])
        if isinstance(item, dict)
    }
    for index, tick in enumerate(ticks):
        if not isinstance(tick, dict):
            failures.append(_failure(
                f"{path}[{index}]", "tick work must be an object", tick))
            continue
        expected = {
            "status": outcomes.get("default", "success"),
            "result": "PM_SUCCESS",
            "queue_submissions": 1,
            "spatial_index_publications": 1,
            "time_advances": 1,
            "intra_tick_host_readbacks": 0,
        }
        expected.update(overrides.get(index, {}))
        expected.pop("tick", None)
        for name in (
            "status", "result", "queue_submissions",
            "spatial_index_publications", "time_advances",
            "intra_tick_host_readbacks",
        ):
            expected_value = expected.get(name)
            if tick.get(name) != expected_value:
                failures.append(_failure(
                    f"{path}[{index}].{name}",
                    "fixed-tick accounting is not exact",
                    tick.get(name), expected_value))
        if expected.get("status") == "failure":
            for name, expected_value in {
                "spatial_index_updates": 0,
                "state_unchanged": True,
                "pending_inputs_preserved": True,
                "hidden_retries": 0,
            }.items():
                if tick.get(name) != expected_value:
                    failures.append(_failure(
                        f"{path}[{index}].{name}",
                        "failed tick was not atomic",
                        tick.get(name), expected_value))
        for issue in ORACLE.evaluate_tick_work(tick, contract):
            failures.append(_failure(
                f"{path}[{index}].{issue.assertion}", issue.message,
                issue.actual, issue.expected))


def _validate_scenario(
    observation: Any,
    definition: Mapping[str, Any],
    fixture: Mapping[str, Any],
    contract: Mapping[str, Any],
    expected_candidate: Mapping[str, Any],
    runtime_variant: Any,
    fixture_identity: Mapping[str, Any],
    failures: list[Failure],
) -> None:
    identity = definition["id"]
    path = f"$.scenarios[{identity}]"
    if not isinstance(observation, dict):
        failures.append(_failure(path, "scenario observation must be an object"))
        return
    if observation.get("id") != identity:
        failures.append(_failure(
            path + ".id", "scenario identity mismatch",
            observation.get("id"), identity))
    if observation.get("source") != "installed-public-api":
        failures.append(_failure(
            path + ".source", "scenario must use the installed public API",
            observation.get("source"), "installed-public-api"))
    if observation.get("driver_private_state_used") is not False:
        failures.append(_failure(
            path + ".driver_private_state_used",
            "driver-private state cannot satisfy an assertion",
            observation.get("driver_private_state_used"), False))
    calls = observation.get("public_api_calls")
    required_calls = fixture_identity["public_api_calls"]
    if calls != required_calls:
        failures.append(_failure(
            path + ".public_api_calls",
            "scenario public call trace is not the shared fixture trace",
            calls, required_calls))
    if observation.get("runtime_variant") != runtime_variant:
        failures.append(_failure(
            path + ".runtime_variant",
            "scenario used a different runtime variant",
            observation.get("runtime_variant"), runtime_variant))
    if observation.get("fixture_contract") != fixture_identity["identity"]:
        failures.append(_failure(
            path + ".fixture_contract",
            "scenario did not consume the exact shared fixture definition",
            observation.get("fixture_contract"), fixture_identity["identity"]))
    executions = observation.get("executions")
    if not isinstance(executions, list) or len(executions) != 2:
        failures.append(_failure(
            path + ".executions",
            "plain and validation-enabled executions are both required",
            executions))
    if observation.get("simulation_recreated_during_ticks") is not False:
        failures.append(_failure(
            path + ".simulation_recreated_during_ticks",
            "sustained scenario ticks must advance one simulation instance",
            observation.get("simulation_recreated_during_ticks"), False))
    _validate_candidate(
        observation.get("candidate"), expected_candidate,
        path + ".candidate", failures)
    if observation.get("domains") != definition.get("domains"):
        failures.append(_failure(
            path + ".domains", "scenario domain set or order changed",
            observation.get("domains"), definition.get("domains")))
    if observation.get("tags") != definition.get("tags"):
        failures.append(_failure(
            path + ".tags", "scenario tag set or order changed",
            observation.get("tags"), definition.get("tags")))
    if observation.get("simulation_status") != "PM_SUCCESS":
        failures.append(_failure(
            path + ".simulation_status", "scenario did not complete",
            observation.get("simulation_status"), "PM_SUCCESS"))

    expected_ticks = int(definition["ticks"])
    steps = observation.get("steps")
    if not isinstance(steps, list) or len(steps) != expected_ticks:
        failures.append(_failure(
            path + ".steps", "scenario tick count is not exact",
            len(steps) if isinstance(steps, list) else steps, expected_ticks))
    else:
        for index, step in enumerate(steps):
            _validate_step(step, f"{path}.steps[{index}]", failures)

    case = {"kind": "scenario", **definition}
    specs = ORACLE.case_limit_specs(case, contract)
    measurements = observation.get("measurements")
    if not isinstance(measurements, dict):
        failures.append(_failure(
            path + ".measurements", "named measurements must be an object",
            measurements))
    else:
        for issue in ORACLE.evaluate_named_measurements(measurements, specs):
            failures.append(_failure(
                path + ".measurements." + issue.assertion,
                issue.message, issue.actual, issue.expected))
    _validate_tick_work(
        observation.get("tick_work"), expected_ticks,
        path + ".tick_work", contract, fixture, failures)
    if isinstance(executions, list) and len(executions) == 2:
        mode_measurements: list[Any] = []
        mode_tick_work: list[Any] = []
        mode_states: list[Any] = []
        state_hashes: list[Any] = []
        for index, execution in enumerate(executions):
            execution_path = f"{path}.executions[{index}]"
            if not isinstance(execution, dict):
                failures.append(_failure(
                    execution_path, "execution proof must be an object",
                    execution))
                continue
            expected_mode = index == 1
            exact = {
                "validation_enabled": expected_mode,
                "simulation_status": "PM_SUCCESS",
                "simulation_instance_count": 1,
                "completed_tick_records": expected_ticks,
            }
            for name, expected_value in exact.items():
                if execution.get(name) != expected_value:
                    failures.append(_failure(
                        f"{execution_path}.{name}",
                        "plain/validation execution proof is not exact",
                        execution.get(name), expected_value))
            execution_measurements = execution.get("measurements")
            execution_work = execution.get("tick_work")
            execution_state = execution.get("final_public_state")
            mode_measurements.append(execution_measurements)
            mode_tick_work.append(execution_work)
            mode_states.append(execution_state)
            for name, value in (
                ("measurements", execution_measurements),
                ("tick_work", execution_work),
                ("final_public_state", execution_state),
            ):
                digest_name = name + "_sha256"
                expected_digest = value_sha256(value)
                if execution.get(digest_name) != expected_digest:
                    failures.append(_failure(
                        f"{execution_path}.{digest_name}",
                        "execution proof digest does not match retained data",
                        execution.get(digest_name), expected_digest))
            state_hash = execution.get("final_public_state_sha256")
            if (not isinstance(state_hash, str)
                    or re.fullmatch(r"[0-9a-f]{64}", state_hash) is None):
                failures.append(_failure(
                    execution_path + ".final_public_state_sha256",
                    "canonical public-state SHA-256 is required", state_hash))
            state_hashes.append(state_hash)
        if len(mode_measurements) == 2 and mode_measurements[0] != mode_measurements[1]:
            failures.append(_failure(
                path + ".executions.measurements",
                "plain and validation-enabled measurements differ",
                mode_measurements[1], mode_measurements[0]))
        if len(mode_tick_work) == 2 and mode_tick_work[0] != mode_tick_work[1]:
            failures.append(_failure(
                path + ".executions.tick_work",
                "plain and validation-enabled tick work differs",
                mode_tick_work[1], mode_tick_work[0]))
        if len(mode_states) == 2 and mode_states[0] != mode_states[1]:
            failures.append(_failure(
                path + ".executions.final_public_state",
                "plain and validation-enabled public states differ",
                state_hashes[1] if len(state_hashes) == 2 else mode_states[1],
                state_hashes[0] if len(state_hashes) == 2 else mode_states[0]))
        if mode_measurements and mode_measurements[0] != measurements:
            failures.append(_failure(
                path + ".measurements",
                "scenario measurements do not match the plain execution",
                measurements, mode_measurements[0]))
        if mode_tick_work and mode_tick_work[0] != observation.get("tick_work"):
            failures.append(_failure(
                path + ".tick_work",
                "scenario tick work does not match the plain execution",
                observation.get("tick_work"), mode_tick_work[0]))


def _capability_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(value.get(field) for field in CAPABILITY_FIELDS)


def _validate_capabilities(
    values: Any,
    obligations: Mapping[str, Any],
    failures: list[Failure],
) -> None:
    expected_values = ORACLE.expand_capabilities(obligations)
    expected = {_capability_key(item): item for item in expected_values}
    observed_items = values if isinstance(values, list) else []
    observed = {
        _capability_key(item): item
        for item in observed_items if isinstance(item, dict)
    }
    if len(observed) != len(observed_items):
        failures.append(_failure(
            "$.capabilities", "capability rows must be unique objects"))
    if set(observed) != set(expected):
        failures.append(_failure(
            "$.capabilities", "physical capability matrix is not the frozen 49 rows",
            [list(key) for key in sorted(observed)],
            [list(key) for key in sorted(expected)]))
    for key, item in observed.items():
        path = "$.capabilities[" + ":".join(str(part) for part in key) + "]"
        if item.get("execution_status") != "completed":
            failures.append(_failure(
                path + ".execution_status",
                "advertised capability did not execute through the public API",
                item.get("execution_status"), "completed"))
        for field in ("analytic_status", "lifecycle_status"):
            if item.get(field) != "passed":
                failures.append(_failure(
                    path + "." + field,
                    "advertised capability is missing an independent assertion",
                    item.get(field), "passed"))


def _validate_public_cases(
    values: Any,
    inventory: Mapping[str, Any],
    failures: list[Failure],
) -> None:
    expected = set(inventory["public_case_sets"]["all"])
    items = values if isinstance(values, list) else []
    observed = {
        item.get("id"): item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if len(observed) != len(items) or set(observed) != expected:
        failures.append(_failure(
            "$.public_cases", "public case set is not the frozen 30 cases",
            sorted(observed), sorted(expected)))
    for identity, item in observed.items():
        if item.get("execution_status") != "completed":
            failures.append(_failure(
                f"$.public_cases[{identity}].execution_status",
                "a frozen public case was unsupported or failed",
                item.get("execution_status"), "completed"))
        if item.get("driver_private_state_used") is not False:
            failures.append(_failure(
                f"$.public_cases[{identity}].driver_private_state_used",
                "public case used driver-private state",
                item.get("driver_private_state_used"), False))


def _validate_distribution(
    value: Any,
    path: str,
    failures: list[Failure],
) -> None:
    names = ("p50", "p95", "p99", "maximum")
    if not isinstance(value, dict):
        failures.append(_failure(path, "timing distribution must be an object", value))
        return
    samples: list[float] = []
    for name in names:
        sample = value.get(name)
        if (isinstance(sample, bool) or not isinstance(sample, (int, float)) or
                not math.isfinite(float(sample)) or sample < 0):
            failures.append(_failure(
                path + "." + name, "timing sample must be finite and nonnegative",
                sample))
            return
        samples.append(float(sample))
    if samples != sorted(samples):
        failures.append(_failure(
            path, "timing percentiles must be monotonic", samples))


def _validate_workloads(
    values: Any,
    contract: Mapping[str, Any],
    target_id: Any,
    runtime_variant: Any,
    failures: list[Failure],
) -> None:
    performance = contract["performance"]
    assignments = {
        item["target"]: item["tier"]
        for item in performance["target_assignments"]
    }
    tier_id = assignments.get(target_id)
    tiers = {item["id"]: item for item in performance["tiers"]}
    tier = tiers.get(tier_id)
    if tier is None:
        failures.append(_failure(
            "$.device.target_id", "target has no frozen performance tier",
            target_id, sorted(assignments)))
        return
    expected = {item["id"]: item for item in tier["workloads"]}
    items = values if isinstance(values, list) else []
    observed = {
        item.get("id"): item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if len(observed) != len(items) or set(observed) != set(expected):
        failures.append(_failure(
            "$.workloads", "target workload set is not exact",
            sorted(observed), sorted(expected)))
    measurement = performance["measurement"]
    for identity, definition in expected.items():
        item = observed.get(identity)
        path = f"$.workloads[{identity}]"
        if not isinstance(item, dict):
            continue
        if item.get("counts") != definition["counts"]:
            failures.append(_failure(
                path + ".counts", "live and capacity counts are not exact",
                item.get("counts"), definition["counts"]))
        if item.get("warmup_ticks") != measurement["warmup_ticks"]:
            failures.append(_failure(
                path + ".warmup_ticks", "warmup prefix is not exact",
                item.get("warmup_ticks"), measurement["warmup_ticks"]))
        if item.get("measured_ticks") != measurement["measured_ticks"]:
            failures.append(_failure(
                path + ".measured_ticks", "sustained window is not exact",
                item.get("measured_ticks"), measurement["measured_ticks"]))
        if item.get("failed_ticks") != 0:
            failures.append(_failure(
                path + ".failed_ticks", "failed workload ticks are forbidden",
                item.get("failed_ticks"), 0))
        ratio = item.get("simulated_to_wall_time_ratio")
        minimum_ratio = measurement["minimum_simulated_to_wall_time_ratio"]
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or
                not math.isfinite(float(ratio)) or ratio < minimum_ratio):
            failures.append(_failure(
                path + ".simulated_to_wall_time_ratio",
                "sustained workload missed the realtime rate",
                ratio, minimum_ratio))
        timings = item.get("timings")
        if not isinstance(timings, dict):
            failures.append(_failure(
                path + ".timings", "CPU/GPU/physics/frame timings are required",
                timings))
            continue
        for name in ("cpu_ms", "physics_ms", "frame_ms"):
            _validate_distribution(timings.get(name), path + ".timings." + name,
                                   failures)
        gpu = timings.get("gpu_ms")
        if runtime_variant in {"cuda-runtime", "vulkan-runtime",
                               "d3d12-runtime", "metal-runtime"}:
            _validate_distribution(gpu, path + ".timings.gpu_ms", failures)
        elif gpu is not None:
            _validate_distribution(gpu, path + ".timings.gpu_ms", failures)
        physics = timings.get("physics_ms")
        if isinstance(physics, dict):
            p95 = physics.get("p95")
            if isinstance(p95, (int, float)) and not isinstance(p95, bool) and \
                    p95 > tier["physics_p95_ms_maximum"]:
                failures.append(_failure(
                    path + ".timings.physics_ms.p95",
                    "physics p95 exceeds the target tier",
                    p95, tier["physics_p95_ms_maximum"]))
        frame = timings.get("frame_ms")
        if isinstance(frame, dict):
            p95 = frame.get("p95")
            if isinstance(p95, (int, float)) and not isinstance(p95, bool) and \
                    p95 > performance["frame_budget_ms"]:
                failures.append(_failure(
                    path + ".timings.frame_ms.p95",
                    "frame p95 exceeds the 60 Hz budget",
                    p95, performance["frame_budget_ms"]))


def evaluate(
    capture: Mapping[str, Any],
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    obligations: Mapping[str, Any],
    suite: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_sha256: str,
) -> dict[str, Any]:
    failures: list[Failure] = []
    if capture.get("schema") != CAPTURE_SCHEMA:
        failures.append(_failure(
            "$.schema", "unexpected capture schema", capture.get("schema"),
            CAPTURE_SCHEMA))
    expected_candidate = _candidate_identity(candidate, candidate_sha256)
    _validate_candidate(
        capture.get("candidate"), expected_candidate, "$.candidate", failures)

    runtime_variant = capture.get("runtime_variant")
    variants = {
        item.get("id"): item
        for item in candidate.get("runtime_variants", [])
        if isinstance(item, dict)
    }
    if runtime_variant not in variants:
        failures.append(_failure(
            "$.runtime_variant", "runtime variant is not in the candidate",
            runtime_variant, sorted(variants)))

    _validate_capabilities(capture.get("capabilities"), obligations, failures)
    _validate_public_cases(capture.get("public_cases"), inventory, failures)

    device = capture.get("device")
    required_device_fields = (
        "target_id", "name", "os", "gpu", "driver", "toolchain",
        "driver_binary_sha256", "driver_manifest_sha256",
    )
    if not isinstance(device, dict):
        failures.append(_failure("$.device", "device identity is required", device))
    else:
        for field in required_device_fields:
            value = device.get(field)
            if not isinstance(value, str) or not value.strip():
                failures.append(_failure(
                    f"$.device.{field}", "nonempty exact identity is required",
                    value))
        for field in ("driver_binary_sha256", "driver_manifest_sha256"):
            value = device.get(field)
            if (not isinstance(value, str) or
                    re.fullmatch(r"[0-9a-f]{64}", value) is None):
                failures.append(_failure(
                    f"$.device.{field}", "lowercase SHA-256 is required", value))
    _validate_workloads(
        capture.get("workloads"), contract,
        device.get("target_id") if isinstance(device, dict) else None,
        runtime_variant, failures)

    backend_observations = capture.get("backend_observations")
    if not isinstance(backend_observations, list) or len(backend_observations) != 2:
        failures.append(_failure(
            "$.backend_observations",
            "plain and validation-enabled public observations are required"))
    else:
        validation_modes: set[bool] = set()
        for index, observation in enumerate(backend_observations):
            if isinstance(observation, dict):
                validation_enabled = observation.get("validation_enabled")
                if isinstance(validation_enabled, bool):
                    validation_modes.add(validation_enabled)
                if observation.get("candidate", {}).get("runtime_variant") != runtime_variant:
                    failures.append(_failure(
                        f"$.backend_observations[{index}].candidate.runtime_variant",
                        "observation used a different runtime variant",
                        observation.get("candidate", {}).get("runtime_variant"),
                        runtime_variant))
                for issue in ORACLE.validate_public_observation(
                        observation, contract, inventory, obligations, suite,
                        candidate, candidate_sha256):
                    failures.append(_failure(
                        f"$.backend_observations[{index}].{issue.assertion}",
                        issue.message, issue.actual, issue.expected))
            else:
                failures.append(_failure(
                    f"$.backend_observations[{index}]",
                    "backend observation must be an object", observation))
        if validation_modes != {False, True}:
            failures.append(_failure(
                "$.backend_observations.validation_enabled",
                "exactly one plain and one validation-enabled run are required",
                sorted(validation_modes), [False, True]))

    scenarios = capture.get("scenarios")
    definitions = {item["id"]: item for item in suite["scenario_fixtures"]}
    fixture_contract = load_json(FIXTURE_PATH)
    fixture_definitions = {
        item["id"]: item for item in fixture_contract.get("scenarios", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if set(fixture_definitions) != set(definitions):
        failures.append(_failure(
            "$.fixture_contract.scenarios",
            "shared executable fixture set differs from the frozen suite",
            sorted(fixture_definitions), sorted(definitions)))
    fixture_context = {
        "identity": {
            "id": fixture_contract.get("id"),
            "sha256": sha256(FIXTURE_PATH),
        },
        "public_api_calls": fixture_contract.get(
            "execution", {}).get("public_api_calls"),
    }
    observed = {
        item.get("id"): item for item in scenarios
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(scenarios, list) else {}
    if set(observed) != set(definitions):
        failures.append(_failure(
            "$.scenarios", "scenario set is not exact", sorted(observed),
            sorted(definitions)))
    for identity, definition in definitions.items():
        _validate_scenario(
            observed.get(identity), definition,
            fixture_definitions.get(identity, {}), contract,
            expected_candidate, runtime_variant, fixture_context, failures)

    controls = capture.get("negative_controls")
    indexed_controls = {
        item.get("id"): item for item in controls
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(controls, list) else {}
    if set(indexed_controls) != set(REQUIRED_NEGATIVE_CONTROLS):
        failures.append(_failure(
            "$.negative_controls", "negative-control set is not exact",
            sorted(indexed_controls), sorted(REQUIRED_NEGATIVE_CONTROLS)))
    for identity in REQUIRED_NEGATIVE_CONTROLS:
        control = indexed_controls.get(identity)
        if not isinstance(control, dict) or control.get("passed") is not True:
            failures.append(_failure(
                f"$.negative_controls[{identity}]",
                "negative control did not reject the injected violation",
                control))

    return {
        "schema": REPORT_SCHEMA,
        "candidate": expected_candidate,
        "runtime_variant": runtime_variant,
        "target_id": device.get("target_id") if isinstance(device, dict) else None,
        "scenario_count": len(observed),
        "negative_control_count": len(indexed_controls),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": [asdict(item) for item in failures],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        capture = load_json(arguments.capture)
        contract = load_json(PROFILE_PATH)
        inventory = load_json(INVENTORY_PATH)
        obligations = load_json(CAPABILITY_PATH)
        suite = load_json(SUITE_PATH)
        candidate = load_json(arguments.candidate_manifest)
        report = evaluate(
            capture, contract, inventory, obligations, suite, candidate,
            sha256(arguments.candidate_manifest))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"realtime-device-suite: {error}")
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    print(f"Realtime device suite: {report['scenario_count']} scenarios, "
          f"{report['failure_count']} failures")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
