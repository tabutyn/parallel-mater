#!/usr/bin/env python3
"""Independent oracles for the realtime-single-step v1 profile.

The module consumes public result data and immutable qualification contracts.  It
does not call driver-private code and does not claim that a runtime implements the
profile.  Physical-driver qualification remains owned by later F8 tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "conformance/v1/qualification/realtime-profile-v1.json"
INVENTORY_PATH = (
    ROOT / "conformance/v1/qualification/kernel-device-obligations-v1.json"
)
CAPABILITY_PATH = (
    ROOT / "conformance/v1/qualification/capability-obligations-v1.json"
)
SUITE_PATH = ROOT / "conformance/v1/realtime/oracle-suite-v1.json"
CANDIDATE_PATH = (
    ROOT / "conformance/v1/qualification/realtime-candidate-v1.json"
)

CAPABILITY_FIELDS = (
    "first_family",
    "second_family",
    "scope",
    "mode",
    "direction",
    "material_profile",
)


@dataclass(frozen=True)
class Failure:
    assertion: str
    message: str
    actual: Any = None
    expected: Any = None


@dataclass(frozen=True)
class LimitSpec:
    assertion: str
    path: str
    comparator: str
    expected: bool | int | float
    category: str


@dataclass(frozen=True)
class SpatialLeaf:
    stable_id: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float
    maximum_correction: float = 0.0


@dataclass
class BoundaryState:
    """Small deterministic model of the one-boundary mutation rules."""

    populations: dict[str, dict[int, int]] = field(default_factory=dict)
    clock_ticks: int = 0
    pending_inputs: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> tuple[int, tuple[tuple[str, tuple[tuple[int, int], ...]], ...]]:
        return (
            self.clock_ticks,
            tuple(
                (family, tuple(sorted(entries.items())))
                for family, entries in sorted(self.populations.items())
            ),
        )

    def submit(
        self,
        mutations: Sequence[Mapping[str, Any]],
        capacities: Mapping[str, int],
        *,
        discovered_error: str | None = None,
    ) -> dict[str, Any]:
        """Apply one boundary atomically, preserving inputs on failed ticks."""

        before = self.snapshot()
        staged = {
            family: dict(entries) for family, entries in self.populations.items()
        }
        rejected: list[dict[str, Any]] = []

        for mutation in mutations:
            operation = mutation["operation"]
            family = str(mutation["family"])
            entries = staged.setdefault(family, {})
            stable_id = int(mutation["stable_id"])
            if operation == "remove":
                entries.pop(stable_id, None)
            elif operation == "replace":
                if stable_id not in entries:
                    rejected.append(dict(mutation, reason="missing-source"))
                else:
                    entries[stable_id] += 1
            elif operation == "birth":
                # Identity ordering is the deterministic admission priority.
                continue
            elif operation == "phase-transfer":
                destination = str(mutation["destination_family"])
                destination_entries = staged.setdefault(destination, {})
                if stable_id not in entries:
                    rejected.append(dict(mutation, reason="missing-source"))
                elif len(destination_entries) >= capacities[destination]:
                    rejected.append(dict(mutation, reason="destination-capacity"))
                else:
                    generation = entries.pop(stable_id)
                    destination_entries[stable_id] = generation + 1
            else:
                raise ValueError(f"unknown mutation operation: {operation}")

        births = sorted(
            (mutation for mutation in mutations if mutation["operation"] == "birth"),
            key=lambda mutation: (str(mutation["family"]), int(mutation["stable_id"])),
        )
        for mutation in births:
            family = str(mutation["family"])
            entries = staged.setdefault(family, {})
            stable_id = int(mutation["stable_id"])
            if stable_id in entries:
                rejected.append(dict(mutation, reason="duplicate-id"))
            elif len(entries) >= capacities[family]:
                rejected.append(dict(mutation, reason="capacity"))
            else:
                entries[stable_id] = 0

        if discovered_error is not None:
            self.pending_inputs = [dict(mutation) for mutation in mutations]
            return {
                "success": False,
                "error": discovered_error,
                "before": before,
                "after": self.snapshot(),
                "pending_inputs_preserved": True,
                "spatial_publications": 0,
                "time_advances": 0,
                "rejected": rejected,
            }

        self.populations = staged
        self.clock_ticks += 1
        self.pending_inputs.clear()
        return {
            "success": True,
            "error": None,
            "before": before,
            "after": self.snapshot(),
            "pending_inputs_preserved": False,
            "spatial_publications": 1,
            "time_advances": 1,
            "rejected": rejected,
        }


class MockClock:
    """Integer-nanosecond clock used by deterministic timing tests."""

    def __init__(self) -> None:
        self._nanoseconds = 0

    def now_ns(self) -> int:
        return self._nanoseconds

    def advance_ms(self, milliseconds: float) -> None:
        if not math.isfinite(milliseconds) or milliseconds < 0.0:
            raise ValueError("clock increments must be finite and nonnegative")
        self._nanoseconds += round(milliseconds * 1_000_000.0)


class TimingCollector:
    """Retains every tick, excluding only the contract's warmup prefix."""

    def __init__(self, clock: MockClock, warmup_ticks: int) -> None:
        self.clock = clock
        self.warmup_ticks = warmup_ticks
        self.completed_ticks = 0
        self.failed_ticks = 0
        self.samples_ms: list[float] = []
        self._start_ns: int | None = None

    def begin_tick(self) -> None:
        if self._start_ns is not None:
            raise RuntimeError("a timing sample is already active")
        self._start_ns = self.clock.now_ns()

    def complete_tick(self, *, success: bool = True) -> None:
        if self._start_ns is None:
            raise RuntimeError("no timing sample is active")
        elapsed_ms = (self.clock.now_ns() - self._start_ns) / 1_000_000.0
        self._start_ns = None
        index = self.completed_ticks
        self.completed_ticks += 1
        if not success:
            self.failed_ticks += 1
        if index >= self.warmup_ticks:
            self.samples_ms.append(elapsed_ms)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json(PROFILE_PATH),
        load_json(INVENTORY_PATH),
        load_json(CAPABILITY_PATH),
        load_json(SUITE_PATH),
    )


def _slug(path: str) -> str:
    return ".".join(
        part.lower().replace("_", "-")
        for part in path.strip("/").split("/")
    )


def _numeric_leaves(value: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, bool) or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        yield prefix, value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield from _numeric_leaves(nested, f"{prefix}/{key}")


def _comparator(path: str, category: str, expected: Any) -> str:
    key = path.rsplit("/", 1)[-1]
    if isinstance(expected, bool):
        return "exact"
    if category in {"iterations", "workload-count", "measurement-window"}:
        return "exact"
    if key == "boundary_latency_ticks":
        return "exact"
    if key.endswith("maximum_exclusive"):
        return "less-than"
    if "minimum" in key or key.endswith("_min"):
        return "minimum"
    if "maximum" in key or key.endswith("_max"):
        return "maximum"
    if category in {"domain-quality", "coupling-quality"}:
        return "maximum"
    if category == "profile" and key in {
        "dynamic_spatial_updates_per_tick_maximum",
        "physics_ticks_per_render_callback_maximum",
    }:
        return "maximum"
    return "exact"


def named_limit_specs(contract: Mapping[str, Any]) -> list[LimitSpec]:
    """Translate every quantitative/bool F8-00 limit into a stable assertion."""

    raw: list[tuple[str, Any, str]] = []

    def add_tree(value: Any, prefix: str, category: str) -> None:
        raw.extend((path, leaf, category) for path, leaf in _numeric_leaves(value, prefix))

    add_tree(contract["profile"], "/profile", "profile")
    clock = contract["clock_and_presentation"]
    add_tree(clock["interpolation_alpha_range"],
             "/clock_and_presentation/interpolation_alpha_range", "clock")
    raw.append((
        "/clock_and_presentation/maximum_presentation_latency_ticks",
        clock["maximum_presentation_latency_ticks"], "clock",
    ))
    add_tree(contract["input_envelope"], "/input_envelope", "input-envelope")
    add_tree(contract["algebraic_iteration_budget"],
             "/algebraic_iteration_budget", "iterations")
    add_tree(contract["quality_measurement"],
             "/quality_measurement", "measurement-window")
    for domain in contract["domain_quality"]:
        add_tree(
            domain["limits"],
            f"/domain_quality/{domain['domain']}/limits",
            "domain-quality",
        )
    for mode in contract["coupling_quality_by_mode"]:
        add_tree(
            mode["limits"],
            f"/coupling_quality_by_mode/{mode['mode']}/limits",
            "coupling-quality",
        )
    failure_flags = {
        key: value for key, value in contract["failure_behavior"].items()
        if isinstance(value, bool)
    }
    add_tree(failure_flags, "/failure_behavior", "failure-behavior")
    add_tree(
        {
            key: value
            for key, value in contract["legacy_separation"].items()
            if isinstance(value, bool)
        },
        "/legacy_separation",
        "legacy-separation",
    )

    performance = contract["performance"]
    raw.append(("/performance/frame_budget_ms",
                performance["frame_budget_ms"], "performance"))
    measurement = performance["measurement"]
    for key in ("warmup_ticks", "measured_ticks",
                "minimum_simulated_to_wall_time_ratio"):
        category = "measurement-window" if key.endswith("ticks") else "performance"
        raw.append((f"/performance/measurement/{key}", measurement[key], category))
    for tier in performance["tiers"]:
        tier_prefix = f"/performance/tiers/{tier['id']}"
        raw.append((f"{tier_prefix}/physics_p95_ms_maximum",
                    tier["physics_p95_ms_maximum"], "performance"))
        raw.append((f"{tier_prefix}/render_input_reserve_ms_minimum",
                    tier["render_input_reserve_ms_minimum"], "performance"))
        for workload in tier["workloads"]:
            for key, value in workload["counts"].items():
                raw.append((
                    f"{tier_prefix}/workloads/{workload['id']}/counts/{key}",
                    value,
                    "workload-count",
                ))

    specs = [
        LimitSpec(
            assertion=f"f8.v1.{_slug(path)}.{_comparator(path, category, value)}",
            path=path,
            comparator=_comparator(path, category, value),
            expected=value,
            category=category,
        )
        for path, value, category in raw
    ]
    if len({spec.path for spec in specs}) != len(specs):
        raise ValueError("duplicate normative limit path")
    if len({spec.assertion for spec in specs}) != len(specs):
        raise ValueError("duplicate named assertion")
    return sorted(specs, key=lambda spec: spec.path)


def evaluate_limit(spec: LimitSpec, actual: Any) -> Failure | None:
    if isinstance(spec.expected, bool):
        if not isinstance(actual, bool) or actual is not spec.expected:
            return Failure(spec.assertion, "boolean contract mismatch",
                           actual, spec.expected)
        return None
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return Failure(spec.assertion, "measurement must be numeric",
                       actual, spec.expected)
    actual_float = float(actual)
    if not math.isfinite(actual_float):
        return Failure(spec.assertion, "measurement must be finite",
                       actual, spec.expected)
    expected = float(spec.expected)
    passed = {
        "exact": math.isclose(actual_float, expected, rel_tol=0.0, abs_tol=1.0e-12),
        "maximum": actual_float <= expected,
        "minimum": actual_float >= expected,
        "less-than": actual_float < expected,
    }[spec.comparator]
    if passed:
        return None
    return Failure(spec.assertion, f"failed {spec.comparator} limit",
                   actual, spec.expected)


def passing_value(spec: LimitSpec) -> bool | int | float:
    if spec.comparator == "less-than":
        return math.nextafter(float(spec.expected), -math.inf)
    return spec.expected


def violating_value(spec: LimitSpec) -> bool | int | float:
    if isinstance(spec.expected, bool):
        return not spec.expected
    if isinstance(spec.expected, int):
        if spec.comparator == "minimum":
            return spec.expected - 1
        return spec.expected + 1
    expected = float(spec.expected)
    if spec.comparator == "exact":
        return expected + max(1.0e-6, abs(expected) * 1.0e-6)
    if spec.comparator == "minimum":
        return math.nextafter(expected, -math.inf)
    return math.nextafter(expected, math.inf)


def expand_capabilities(obligations: Mapping[str, Any]) -> list[dict[str, str]]:
    """Expand the authoritative rule table without maintaining another pair list."""

    expanded: dict[tuple[str, ...], dict[str, str]] = {}
    for rule in obligations["rules"]:
        match = rule["match"]
        values = [match[field] for field in CAPABILITY_FIELDS]
        for combination in itertools.product(*values):
            capability = dict(zip(CAPABILITY_FIELDS, combination))
            key = tuple(capability[field] for field in CAPABILITY_FIELDS)
            capability["rule_id"] = rule["id"]
            expanded.setdefault(key, capability)
    return [expanded[key] for key in sorted(expanded)]


def derived_case_plan(
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    obligations: Mapping[str, Any],
    suite: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    domains = [
        {
            "id": f"domain:{entry['domain'].lower()}",
            "kind": "domain",
            "domain": entry["domain"],
        }
        for entry in contract["domain_quality"]
    ]
    capabilities = []
    for capability in expand_capabilities(obligations):
        identity = ":".join(capability[field].lower() for field in CAPABILITY_FIELDS)
        capabilities.append({
            "id": f"capability:{identity}",
            "kind": "capability",
            **capability,
        })
    workloads = [
        {
            "id": f"workload:{tier['id']}:{workload['id']}",
            "kind": "workload",
            "tier": tier["id"],
            "workload": workload["id"],
            "counts": workload["counts"],
        }
        for tier in contract["performance"]["tiers"]
        for workload in tier["workloads"]
    ]
    backends = [
        {
            "id": variant["f8_execution_test"],
            "kind": "backend",
            "variant": variant["id"],
            "driver": variant["driver"],
        }
        for variant in inventory["compiled_variants"]
        if variant["kind"] in {"runtime", "runtime-subset", "runtime-with-current-stubs"}
    ]
    suite_contract = load_json(SUITE_PATH) if suite is None else suite
    scenarios = [
        {"kind": "scenario", **scenario}
        for scenario in suite_contract["scenario_fixtures"]
    ]
    return {
        "domains": domains,
        "capabilities": capabilities,
        "workloads": workloads,
        "backends": backends,
        "scenarios": scenarios,
    }


def case_limit_specs(
    case: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[LimitSpec]:
    """Return the normative assertions applicable to one derived case."""

    specs = named_limit_specs(contract)
    prefixes: list[str] = []
    kind = case.get("kind")
    if kind == "domain":
        prefixes.append(f"/domain_quality/{case['domain']}/limits/")
    elif kind == "capability":
        prefixes.extend(
            f"/domain_quality/{domain}/limits/"
            for domain in sorted({case["first_family"], case["second_family"]})
        )
        prefixes.append(
            f"/coupling_quality_by_mode/{case['mode']}/limits/"
        )
    elif kind == "workload":
        prefixes.append(
            f"/performance/tiers/{case['tier']}/workloads/"
            f"{case['workload']}/counts/"
        )
    elif kind == "scenario":
        prefixes.extend(
            f"/domain_quality/{domain}/limits/"
            for domain in case.get("domains", [])
        )
    elif kind != "backend":
        raise ValueError(f"unknown realtime case kind: {kind}")
    return [spec for spec in specs if any(spec.path.startswith(p) for p in prefixes)]


def evaluate_named_measurements(
    measurements: Mapping[str, Any], specs: Sequence[LimitSpec]
) -> list[Failure]:
    """Evaluate a complete named result set; missing and invented metrics fail."""

    failures: list[Failure] = []
    expected = {spec.assertion: spec for spec in specs}
    for assertion in sorted(expected.keys() - measurements.keys()):
        failures.append(Failure(
            f"observation.missing.{assertion}",
            "required named measurement is absent",
        ))
    for assertion in sorted(measurements.keys() - expected.keys()):
        failures.append(Failure(
            f"observation.unexpected.{assertion}",
            "measurement is not part of this case",
        ))
    for assertion in sorted(expected.keys() & measurements.keys()):
        failure = evaluate_limit(expected[assertion], measurements[assertion])
        if failure is not None:
            failures.append(failure)
    return failures


def evaluate_workload_counts(
    actual: Mapping[str, Any], expected: Mapping[str, int]
) -> list[Failure]:
    """Require an exact frozen live/capacity count vector."""

    failures: list[Failure] = []
    for name in sorted(expected.keys() | actual.keys()):
        observed = actual.get(name)
        required = expected.get(name)
        if name not in expected:
            failures.append(Failure(
                f"workload.counts.{name}", "uncontracted workload count", observed
            ))
        elif name not in actual:
            failures.append(Failure(
                f"workload.counts.{name}", "required workload count is absent",
                None, required,
            ))
        elif (isinstance(observed, bool) or not isinstance(observed, int)
              or observed != required):
            failures.append(Failure(
                f"workload.counts.{name}", "workload count is not exact",
                observed, required,
            ))
    return failures


def _binary32_ulp_distance(first: float, second: float) -> int:
    if not math.isfinite(first) or not math.isfinite(second):
        return 2**32
    try:
        first_bits = struct.unpack("!I", struct.pack("!f", first))[0]
        second_bits = struct.unpack("!I", struct.pack("!f", second))[0]
    except (OverflowError, struct.error):
        return 2**32
    return abs(first_bits - second_bits)


def validate_public_observation(
    observation: Mapping[str, Any],
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    obligations: Mapping[str, Any],
    suite: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_sha256: str,
) -> list[Failure]:
    """Validate one future runtime result without accepting private telemetry."""

    failures: list[Failure] = []
    if observation.get("schema") != "parallel-mater-realtime-observation/v1":
        failures.append(Failure(
            "observation.schema", "unexpected observation schema",
            observation.get("schema"), "parallel-mater-realtime-observation/v1",
        ))
    if observation.get("source") != "installed-public-api":
        failures.append(Failure(
            "observation.source", "only installed public-API results are accepted",
            observation.get("source"), "installed-public-api",
        ))
    if observation.get("driver_private_state_used") is not False:
        failures.append(Failure(
            "observation.driver-private-state",
            "driver-private state cannot satisfy an oracle",
            observation.get("driver_private_state_used"), False,
        ))

    profile = observation.get("profile", {})
    required_profile = contract["profile"]
    expected_profile = {
        "name": required_profile["semantic_name"],
        "version": required_profile["version"],
    }
    if profile != expected_profile:
        failures.append(Failure(
            "observation.profile", "profile identity is not exact",
            profile, expected_profile,
        ))

    candidate_input = observation.get("candidate", {})
    variants = {
        item.get("id") for item in candidate.get("runtime_variants", [])
        if isinstance(item, dict)
    }
    expected_candidate = {
        "manifest_role": "qualification-input",
        "execution_identity_claimed": False,
        "id": candidate.get("id"),
        "manifest_sha256": candidate_sha256,
        "source_revision": candidate.get("source", {}).get("revision"),
        "runtime_variant": candidate_input.get("runtime_variant"),
    }
    if (candidate_input != expected_candidate or
            candidate_input.get("runtime_variant") not in variants):
        failures.append(Failure(
            "observation.candidate",
            "candidate manifest inputs are not exact",
            candidate_input, expected_candidate,
        ))

    plan = derived_case_plan(contract, inventory, obligations, suite)
    cases = {
        case["id"]: case
        for group in plan.values()
        for case in group
    }
    case_id = observation.get("case_id")
    case = cases.get(case_id)
    if case is None:
        failures.append(Failure(
            "observation.case-id", "case is not in the derived F8 registry", case_id
        ))
    elif (case["kind"] == "backend" and
          candidate_input.get("runtime_variant") != case["variant"]):
        failures.append(Failure(
            "observation.candidate-runtime",
            "candidate runtime variant does not match the backend case",
            candidate_input.get("runtime_variant"), case["variant"],
        ))

    required_calls = suite["public_api"]["operation_sequence"]
    calls = observation.get("public_api_calls")
    if calls != required_calls:
        failures.append(Failure(
            "observation.public-api-calls",
            "call trace does not match the frozen installed-API sequence",
            calls, required_calls,
        ))

    timestep = (
        required_profile["timestep_seconds"]["numerator"]
        / required_profile["timestep_seconds"]["denominator"]
    )
    steps = observation.get("steps")
    if not isinstance(steps, list) or not steps:
        failures.append(Failure(
            "observation.steps", "at least one completed public step is required"
        ))
    else:
        for index, step in enumerate(steps):
            actual_dt = step.get("timestep_seconds") if isinstance(step, dict) else None
            if (not isinstance(actual_dt, (int, float))
                    or isinstance(actual_dt, bool)
                    or _binary32_ulp_distance(float(actual_dt), timestep)
                    > required_profile["step_timestep_tolerance_ulps"]):
                failures.append(Failure(
                    f"observation.steps.{index}.timestep",
                    "step timestep is outside the binary32 ULP contract",
                    actual_dt, timestep,
                ))
            substeps = step.get("substeps") if isinstance(step, dict) else None
            if isinstance(substeps, bool) or not isinstance(substeps, int) \
                    or substeps != 1:
                failures.append(Failure(
                    f"observation.steps.{index}.substeps",
                    "realtime profile requires exactly one substep",
                    substeps, 1,
                ))

    if observation.get("simulation_status") != "PM_SUCCESS":
        failures.append(Failure(
            "observation.simulation-status",
            "completed public simulation status is required",
            observation.get("simulation_status"), "PM_SUCCESS",
        ))
    statistics = observation.get("statistics", {})
    expected_statistics = {
        "step_count": len(steps) if isinstance(steps, list) else 0,
        "substeps_executed": len(steps) if isinstance(steps, list) else 0,
        "execution_result": "PM_SUCCESS",
        "telemetry_schema_version": 1,
        "profile_version": 1,
        "budget_version": 1,
    }
    if statistics != expected_statistics:
        failures.append(Failure(
            "observation.statistics",
            "public step and telemetry statistics are not exact",
            statistics, expected_statistics,
        ))
    public_state = observation.get("copied_public_state", {})
    vectors = [public_state.get("position"),
               public_state.get("linear_velocity")]
    if (public_state.get("body_count") != 1 or any(
            not isinstance(vector, list) or len(vector) != 3 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in vector)
            for vector in vectors)):
        failures.append(Failure(
            "observation.copied-public-state",
            "one finite rigid state copied through the public API is required",
            public_state,
        ))

    if case is not None:
        specs = case_limit_specs(case, contract)
        measurements = observation.get("measurements", {})
        if not isinstance(measurements, dict):
            failures.append(Failure(
                "observation.measurements", "named measurements must be an object"
            ))
        else:
            failures.extend(evaluate_named_measurements(measurements, specs))
        if case["kind"] == "workload":
            counts = observation.get("workload_counts", {})
            if not isinstance(counts, dict):
                failures.append(Failure(
                    "observation.workload-counts", "workload counts must be an object"
                ))
            else:
                failures.extend(evaluate_workload_counts(counts, case["counts"]))

    work = observation.get("tick_work")
    if not isinstance(work, list) or not work:
        failures.append(Failure(
            "observation.tick-work", "per-tick work observations are required"
        ))
    else:
        for index, tick in enumerate(work):
            for failure in evaluate_tick_work(tick, contract):
                failures.append(Failure(
                    f"observation.tick-work.{index}.{failure.assertion}",
                    failure.message, failure.actual, failure.expected,
                ))
    return failures


def validate_suite(
    suite: Mapping[str, Any],
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    obligations: Mapping[str, Any],
) -> list[Failure]:
    failures: list[Failure] = []
    if suite.get("schema") != "parallel-mater-realtime-oracle-suite/v1":
        failures.append(Failure("suite.schema", "unexpected suite schema",
                                suite.get("schema")))
    if suite.get("profile") != contract["id"]:
        failures.append(Failure("suite.profile", "profile mismatch",
                                suite.get("profile"), contract["id"]))

    sources = suite.get("sources", {})
    expected_sources = {
        "profile": "conformance/v1/qualification/realtime-profile-v1.json",
        "inventory": "conformance/v1/qualification/kernel-device-obligations-v1.json",
        "capabilities": "conformance/v1/qualification/capability-obligations-v1.json",
    }
    if sources != expected_sources:
        failures.append(Failure("suite.sources", "source paths are not canonical",
                                sources, expected_sources))

    scenarios = suite.get("scenario_fixtures", [])
    ids = [scenario.get("id") for scenario in scenarios]
    if len(ids) != len(set(ids)) or not all(isinstance(item, str) for item in ids):
        failures.append(Failure("suite.scenario_fixtures", "scenario ids must be unique"))
    required_tags = {
        "cold-start", "long-run", "impact", "rotation", "stiff-system",
        "mass-ratio", "containment", "source", "removal", "replacement",
        "overflow", "recovery", "empty", "sparse",
    }
    observed_tags = {
        tag for scenario in scenarios for tag in scenario.get("tags", [])
    }
    missing_tags = sorted(required_tags - observed_tags)
    if missing_tags:
        failures.append(Failure("suite.scenario_fixtures",
                                "required adversarial scenarios are missing", missing_tags))

    expected_domains = {entry["domain"] for entry in contract["domain_quality"]}
    observed_domains = {
        domain for scenario in scenarios for domain in scenario.get("domains", [])
    }
    if not expected_domains <= observed_domains:
        failures.append(Failure("suite.scenario_fixtures",
                                "domain scenarios are incomplete",
                                sorted(observed_domains), sorted(expected_domains)))
    if not any(len(set(scenario.get("domains", []))) >= 3 for scenario in scenarios):
        failures.append(Failure("suite.scenario_fixtures",
                                "no three-plus-family scenario is defined"))

    plan = derived_case_plan(contract, inventory, obligations, suite)
    if {item["mode"] for item in plan["capabilities"]} != set(inventory["advertised_modes"]):
        failures.append(Failure("suite.capability_cases",
                                "derived modes differ from inventory"))
    configured_backends = set(suite.get("public_api", {}).get("execution_cases", []))
    expected_backends = {item["id"] for item in plan["backends"]}
    if configured_backends != expected_backends:
        failures.append(Failure("suite.public_api.execution_cases",
                                "backend execution cases are incomplete",
                                sorted(configured_backends), sorted(expected_backends)))

    operations = suite.get("public_api", {}).get("operation_sequence", [])
    if not operations or any(not isinstance(item, str) or not item.startswith("pm")
                             for item in operations):
        failures.append(Failure("suite.public_api.operation_sequence",
                                "only named public API operations are allowed", operations))

    trajectory_cases = suite.get("public_trajectory_cases", [])
    trajectory_ids = [case.get("id") for case in trajectory_cases]
    if (not trajectory_cases or len(trajectory_ids) != len(set(trajectory_ids))
            or not all(isinstance(item, str) for item in trajectory_ids)):
        failures.append(Failure(
            "suite.public_trajectory_cases",
            "public trajectory case ids must be present and unique",
        ))
    inventory_case_ids = {
        case_id
        for group in inventory["kernel_groups"]
        for case_id in group.get("f8_public_cases", [])
    } | {
        case_id
        for path in inventory["auxiliary_paths"]
        for case_id in path.get("f8_public_cases", [])
    }
    if set(trajectory_ids) != inventory_case_ids:
        failures.append(Failure(
            "suite.public_trajectory_cases",
            "trajectory registry and kernel inventory differ",
            sorted(trajectory_ids), sorted(inventory_case_ids),
        ))
    domain_names = {entry["domain"] for entry in contract["domain_quality"]}
    mode_names = set(inventory["advertised_modes"])
    for case in trajectory_cases:
        if (not case.get("inventory_groups") or case.get("ticks", 0) <= 0
                or not case.get("required_observations")):
            failures.append(Failure(
                f"suite.public_trajectory_cases.{case.get('id')}",
                "trajectory case is not executable",
            ))
        if not set(case.get("domains", [])) <= domain_names:
            failures.append(Failure(
                f"suite.public_trajectory_cases.{case.get('id')}.domains",
                "trajectory names an unknown domain",
            ))
        if not set(case.get("modes", [])) <= mode_names:
            failures.append(Failure(
                f"suite.public_trajectory_cases.{case.get('id')}.modes",
                "trajectory names an unadvertised mode",
            ))

    accounting = suite.get("work_accounting", {})
    if accounting.get("unit") != "per-tick":
        failures.append(Failure("suite.work_accounting.unit",
                                "work accounting must be per-tick",
                                accounting.get("unit"), "per-tick"))
    required_counters = {
        "dispatches", "workgroups", "barriers", "queue_submissions",
        "intra_tick_host_readbacks", "bytes_moved", "spatial_index_updates",
        "spatial_index_publications", "time_advances", "iterations",
    }
    counters = set(accounting.get("required_counters", []))
    if counters != required_counters:
        failures.append(Failure("suite.work_accounting.required_counters",
                                "work counter set is incomplete",
                                sorted(counters), sorted(required_counters)))
    if suite.get("negative_controls", {}).get("one_per_named_limit") is not True:
        failures.append(Failure("suite.negative_controls",
                                "each named limit needs a failing control"))
    if suite.get("legacy_separation", {}).get("accept_legacy_as_realtime") is not False:
        failures.append(Failure("suite.legacy_separation",
                                "legacy results must not pass as realtime"))
    return failures


def _vector_subtract(first: Sequence[float], second: Sequence[float]) -> tuple[float, ...]:
    return tuple(a - b for a, b in zip(first, second))


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def relative_residual(residual: Sequence[float], applied: Sequence[float]) -> float:
    return _norm(residual) / max(_norm(applied), 1.0e-6)


def rms(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("RMS needs at least one sample")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("RMS samples must be finite")
    return math.sqrt(sum(value * value for value in values) / len(values))


def sample_statistics(values: Sequence[float]) -> dict[str, float]:
    """Compute the common raw-sample quality statistics used by domain gates."""

    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("quality samples must be finite and nonempty")
    return {
        "rms": rms(values),
        "p95": percentile_nearest_rank(values, 0.95),
        "maximum": max(values),
    }


def normalized_state_error(
    reference: Sequence[float], observed: Sequence[float], scale: float
) -> float:
    """Euclidean split-versus-combined error normalized by a fixed SI scale."""

    if len(reference) != len(observed) or not reference or scale <= 0.0:
        raise ValueError("invalid normalized-state comparison")
    values = (*reference, *observed, scale)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("state comparison values must be finite")
    return _norm(_vector_subtract(observed, reference)) / scale


def centroid(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return an independent equal-weight centroid for transport fixtures."""

    if not points or not points[0]:
        raise ValueError("centroid needs at least one nonempty point")
    dimensions = len(points[0])
    if any(len(point) != dimensions for point in points):
        raise ValueError("centroid point dimensions differ")
    values = tuple(value for point in points for value in point)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("centroid values must be finite")
    return tuple(
        sum(point[axis] for point in points) / len(points)
        for axis in range(dimensions)
    )


def compliance_response(compliance: float, timestep: float,
                        effective_inverse_mass: float) -> tuple[float, float]:
    if compliance < 0.0 or timestep <= 0.0 or effective_inverse_mass <= 0.0:
        raise ValueError("invalid compliance conversion inputs")
    ratio = compliance / (timestep * timestep * effective_inverse_mass)
    return ratio, 1.0 / (1.0 + ratio)


def percentile_nearest_rank(samples: Sequence[float], percentile: float) -> float:
    if not samples or not 0.0 < percentile <= 1.0:
        raise ValueError("invalid percentile request")
    if not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("timing samples must be finite and nonnegative")
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def timing_summary(
    collector: TimingCollector,
    wall_seconds: float,
    *,
    gpu_timing_available: bool,
) -> dict[str, Any]:
    samples = collector.samples_ms
    if not samples:
        raise ValueError("no measured timing samples")
    simulated_seconds = collector.completed_ticks / 60.0
    return {
        "unit": "per-tick",
        "warmup_ticks": collector.warmup_ticks,
        "measured_ticks": len(samples),
        "retained_samples": len(samples),
        "failed_ticks": collector.failed_ticks,
        "p50_ms": percentile_nearest_rank(samples, 0.50),
        "p95_ms": percentile_nearest_rank(samples, 0.95),
        "p99_ms": percentile_nearest_rank(samples, 0.99),
        "maximum_ms": max(samples),
        "simulated_to_wall_time_ratio": simulated_seconds / wall_seconds,
        "gpu_timing": "available" if gpu_timing_available else "unavailable",
        "gpu_phase_times": [] if not gpu_timing_available else None,
    }


def evaluate_timing_summary(
    summary: Mapping[str, Any], tier: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[Failure]:
    failures: list[Failure] = []
    measurement = contract["performance"]["measurement"]
    expected = {
        "unit": "per-tick",
        "warmup_ticks": measurement["warmup_ticks"],
        "measured_ticks": measurement["measured_ticks"],
        "retained_samples": measurement["measured_ticks"],
        "failed_ticks": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(Failure(f"timing.{key}", "timing contract mismatch",
                                    summary.get(key), value))
    statistics = [summary.get(key) for key in
                  ("p50_ms", "p95_ms", "p99_ms", "maximum_ms")]
    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in statistics)
            or statistics != sorted(statistics)):
        failures.append(Failure("timing.statistics",
                                "timing statistics must be finite and ordered",
                                statistics))
    elif summary["p95_ms"] > tier["physics_p95_ms_maximum"]:
        failures.append(Failure("timing.p95", "physics p95 exceeds tier",
                                summary["p95_ms"], tier["physics_p95_ms_maximum"]))
    ratio = summary.get("simulated_to_wall_time_ratio")
    minimum_ratio = measurement["minimum_simulated_to_wall_time_ratio"]
    if not isinstance(ratio, (int, float)) or ratio < minimum_ratio:
        failures.append(Failure("timing.simulated_to_wall_time_ratio",
                                "simulation throughput is below contract",
                                ratio, minimum_ratio))
    if summary.get("gpu_timing") == "unavailable" and summary.get("gpu_phase_times"):
        failures.append(Failure("timing.gpu_phase_times",
                                "CPU timing cannot substitute for unavailable GPU timing"))
    if summary.get("gpu_timing") not in {"available", "unavailable"}:
        failures.append(Failure("timing.gpu_timing",
                                "GPU timing availability must be explicit"))
    return failures


def _closest_relative_distance_squared(first: SpatialLeaf, second: SpatialLeaf) -> float:
    relative_start = _vector_subtract(first.start, second.start)
    first_delta = _vector_subtract(first.end, first.start)
    second_delta = _vector_subtract(second.end, second.start)
    relative_delta = _vector_subtract(first_delta, second_delta)
    denominator = _dot(relative_delta, relative_delta)
    parameter = 0.0 if denominator == 0.0 else max(
        0.0, min(1.0, -_dot(relative_start, relative_delta) / denominator)
    )
    closest = tuple(
        relative_start[index] + parameter * relative_delta[index]
        for index in range(3)
    )
    return _dot(closest, closest)


def swept_spheres_intersect(first: SpatialLeaf, second: SpatialLeaf) -> bool:
    values = (*first.start, *first.end, first.radius, first.maximum_correction,
              *second.start, *second.end, second.radius, second.maximum_correction)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("spatial leaf values must be finite")
    if min(first.radius, first.maximum_correction,
           second.radius, second.maximum_correction) < 0.0:
        raise ValueError("spatial radii and corrections must be nonnegative")
    reach = (first.radius + first.maximum_correction +
             second.radius + second.maximum_correction)
    return _closest_relative_distance_squared(first, second) <= reach * reach


def brute_force_candidate_pairs(leaves: Sequence[SpatialLeaf]) -> set[tuple[int, int]]:
    if len({leaf.stable_id for leaf in leaves}) != len(leaves):
        raise ValueError("spatial leaf stable ids must be unique")
    return {
        tuple(sorted((first.stable_id, second.stable_id)))
        for index, first in enumerate(leaves)
        for second in leaves[index + 1:]
        if swept_spheres_intersect(first, second)
    }


def validate_candidate_pairs(
    leaves: Sequence[SpatialLeaf], candidates: Iterable[tuple[int, int]]
) -> list[Failure]:
    expected = brute_force_candidate_pairs(leaves)
    candidate_set = {tuple(sorted(pair)) for pair in candidates}
    missing = sorted(expected - candidate_set)
    if missing:
        return [Failure("spatial.candidate-completeness",
                        "candidate set missed swept overlaps", missing)]
    return []


def evaluate_tick_work(tick: Mapping[str, Any], contract: Mapping[str, Any]) -> list[Failure]:
    failures: list[Failure] = []
    if tick.get("unit") != "per-tick":
        failures.append(Failure("work.unit", "work must be reported per-tick",
                                tick.get("unit"), "per-tick"))
    integer_fields = (
        "dispatches", "workgroups", "barriers", "queue_submissions",
        "intra_tick_host_readbacks", "bytes_moved", "spatial_index_updates",
        "spatial_index_publications", "time_advances",
    )
    for key in integer_fields:
        value = tick.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            failures.append(Failure(f"work.{key}",
                                    "counter must be a nonnegative integer", value))

    phases = tick.get("phase_dispatches")
    if not isinstance(phases, dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in phases.values()
    ):
        failures.append(Failure("work.phase_dispatches",
                                "phase dispatch counts must be nonnegative integers",
                                phases))
    elif tick.get("dispatches") != sum(phases.values()):
        failures.append(Failure("work.dispatches",
                                "dispatch total differs from emitted phase work",
                                tick.get("dispatches"), sum(phases.values())))

    if tick.get("spatial_index_updates", 0) > 1:
        failures.append(Failure("work.spatial_index_updates",
                                "more than one dynamic index update was emitted",
                                tick.get("spatial_index_updates"), 1))
    if tick.get("intra_tick_host_readbacks") != 0:
        failures.append(Failure("work.intra_tick_host_readbacks",
                                "intra-tick host readback is forbidden",
                                tick.get("intra_tick_host_readbacks"), 0))

    status = tick.get("status")
    successful = status == "success"
    expected_commit = 1 if successful else 0
    for key in ("spatial_index_publications", "time_advances"):
        if tick.get(key) != expected_commit:
            failures.append(Failure(f"work.{key}",
                                    "tick commit accounting mismatch",
                                    tick.get(key), expected_commit))

    iterations = tick.get("iterations")
    budget = contract["algebraic_iteration_budget"]
    active = tick.get("active_iteration_budgets", [])
    if not isinstance(iterations, dict) or not isinstance(active, list):
        failures.append(Failure("work.iterations",
                                "iteration counters and active budget list are required"))
    else:
        for key in active:
            if key not in budget or not isinstance(budget[key], int):
                failures.append(Failure(f"work.iterations.{key}",
                                        "unknown iteration budget"))
            elif iterations.get(key) != budget[key]:
                failures.append(Failure(f"work.iterations.{key}",
                                        "iteration count is not exact",
                                        iterations.get(key), budget[key]))
        coupling_edges = tick.get("coupling_edges", 0)
        expected_coupling = coupling_edges * budget["coupling_sweeps_per_registered_edge"]
        if iterations.get("coupling_sweeps_total", 0) != expected_coupling:
            failures.append(Failure("work.iterations.coupling_sweeps_total",
                                    "coupling work is not exact per edge",
                                    iterations.get("coupling_sweeps_total"),
                                    expected_coupling))

    if not successful:
        atomic = tick.get("atomic_failure", {})
        for key in ("committed_state_preserved", "clock_preserved",
                    "pending_inputs_preserved"):
            if atomic.get(key) is not True:
                failures.append(Failure(f"work.atomic_failure.{key}",
                                        "failed tick was not atomic",
                                        atomic.get(key), True))
        if tick.get("retry_substeps", 0) != 0 or tick.get("retry_spatial_updates", 0) != 0:
            failures.append(Failure("work.failure_retry",
                                    "failed tick hid retry work"))
    return failures


def check_contract() -> list[Failure]:
    contract, inventory, obligations, suite = load_contracts()
    failures = validate_suite(suite, contract, inventory, obligations)
    specs = named_limit_specs(contract)
    if not specs:
        failures.append(Failure("limits", "no named limits were produced"))
    for spec in specs:
        failure = evaluate_limit(spec, passing_value(spec))
        if failure:
            failures.append(failure)
        if evaluate_limit(spec, violating_value(spec)) is None:
            failures.append(Failure(spec.assertion,
                                    "negative control did not fail", spec.path))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-contract", action="store_true",
                        help="validate the F8-03 oracle contract")
    parser.add_argument("--validate-observation", type=Path,
                        help="validate an installed public-API observation")
    parser.add_argument("--candidate-manifest", type=Path,
                        default=CANDIDATE_PATH,
                        help="candidate manifest named by the observation")
    args = parser.parse_args()
    if args.check_contract == bool(args.validate_observation):
        parser.error("select exactly one of --check-contract or "
                     "--validate-observation")
    if args.validate_observation:
        candidate_bytes = args.candidate_manifest.read_bytes()
        candidate = json.loads(candidate_bytes)
        contract, inventory, obligations, suite = load_contracts()
        observation = load_json(args.validate_observation)
        failures = validate_public_observation(
            observation, contract, inventory, obligations, suite, candidate,
            hashlib.sha256(candidate_bytes).hexdigest(),
        )
        if failures:
            for failure in failures:
                print(f"FAIL {failure.assertion}: {failure.message}")
            return 1
        print("PASS: installed public realtime observation")
        return 0
    failures = check_contract()
    if failures:
        for failure in failures:
            print(f"FAIL {failure.assertion}: {failure.message}")
        return 1
    contract, inventory, obligations, _ = load_contracts()
    plan = derived_case_plan(contract, inventory, obligations)
    print(
        "PASS: realtime oracle contract; "
        f"{len(named_limit_specs(contract))} named limits, "
        f"{len(plan['capabilities'])} advertised capability cases, "
        f"{len(plan['workloads'])} fixed workloads, "
        f"{len(plan['backends'])} runtime adapters"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
