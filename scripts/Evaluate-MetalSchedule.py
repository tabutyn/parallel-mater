#!/usr/bin/env python3
"""Evaluate native Metal schedule JSONL against fixed F8 ceilings."""

import argparse
import csv
import json
import pathlib
import statistics
import sys


PROFILE_SCHEMA = "parallel-mater-metal-schedule-profile/v1"
CEILINGS = {
    # Frozen independently in F8-54. Total workgroups outside the batched
    # solver scale with the body/contact workload, so the 386-body gate checks
    # the reviewed solve/prepare commands directly instead of reusing the
    # unrelated eight-body total of 1,445.
    "rigid-386": {
        "dispatches": 40,
        "dependency_barriers": 45,
        "declared_launches": 40,
    },
    # Frozen independently in F8-55 before native encoder capture existed.
    "mixed": {
        "dispatches": 512,
        "workgroups": 640,
        "dependency_barriers": 768,
        "declared_launches": 512,
    },
    # Representative installed-Gallery captures retain all native counters and
    # timing distributions. The smoke/soft-body scene also has a reviewed gate:
    # its fixed 20-body asset must stay below the rounded command/work limits
    # established before the Metal resource batches were implemented.
    "gallery-fluid": None,
    "gallery-cloth": None,
    "gallery-soft-body": None,
    "gallery-soft-body-fluid": None,
    "gallery-smoke-soft-body": {
        "dispatches": 288,
        "workgroups": 67000,
        "dependency_barriers": 672,
        "declared_launches": 288,
    },
}
KERNEL_GATES = {
    "rigid-386": {
        "required": {
            "prepare_avbd_kernel": {"dispatches": 1, "workgroups": 1},
            "solve_avbd_kernel": {"dispatches": 1, "workgroups": 1},
        },
        "forbidden": {"solve_avbd_stage_kernel"},
    },
    "mixed": {
        "required": {},
        "forbidden": {"solve_avbd_stage_kernel"},
    },
    "gallery-fluid": {"required": {}, "forbidden": {"solve_avbd_stage_kernel"}},
    "gallery-cloth": {"required": {}, "forbidden": {"solve_avbd_stage_kernel"}},
    "gallery-soft-body": {"required": {}, "forbidden": {"solve_avbd_stage_kernel"}},
    "gallery-soft-body-fluid": {"required": {}, "forbidden": {"solve_avbd_stage_kernel"}},
    "gallery-smoke-soft-body": {
        "required": {
            "deformable_collide_batch": {"dispatches": 8},
            "reduce_point_body_impulses_batch": {"dispatches": 8},
            "soft_body_apply_contact_friction_batch": {"dispatches": 8},
            "smoke_soft_body_contact_batch": {"dispatches": 8},
            "smoke_grid_soft_body_force_batch": {"dispatches": 8},
            "pm_spatial_publish_epoch": {"dispatches": 1},
        },
        "forbidden": {
            "solve_avbd_stage_kernel",
            "deformable_collide",
            "reduce_point_body_impulses",
            "soft_body_apply_contact_friction",
            "smoke_soft_body_contact",
            "smoke_grid_soft_body_force",
        },
    },
}
COUNTERS = (
    "dispatches",
    "workgroups",
    "dependency_barriers",
    "dependency_ranges",
    "native_barrier_commands",
    "transfer_commands",
    "readbacks",
    "host_waits",
    "bytes_moved",
    "declared_launches",
)
FIXED_TICK_HZ = 60


def load_profiles(paths):
    profiles = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    profile = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {error}") from error
                if profile.get("schema") != PROFILE_SCHEMA:
                    raise ValueError(
                        f"{path}:{line_number}: unexpected Metal profile schema")
                profiles.append(profile)
    return profiles


def load_frame_timings(path):
    required = {
        "frame", "scene", "mode", "physics_steps", "total_ms",
        "physics_ms", "render_ms", "present_ms", "other_ms",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != required:
            raise ValueError("unexpected frame timing CSV columns")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                rows.append({
                    "frame": int(row["frame"]),
                    "scene": row["scene"],
                    "mode": row["mode"],
                    "physics_steps": int(row["physics_steps"]),
                    "total_ms": float(row["total_ms"]),
                    "physics_ms": float(row["physics_ms"]),
                    "render_ms": float(row["render_ms"]),
                    "present_ms": (float(row["present_ms"])
                                   if row["present_ms"] else None),
                    "other_ms": float(row["other_ms"]),
                })
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line}: invalid frame timing value") from error
    return rows


def percentile(values, fraction):
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def distribution(values):
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "minimum": min(values),
        "maximum": max(values),
    }


def validate_profile(profile, workload, ceiling):
    if profile.get("final") is not True:
        raise ValueError("incomplete Metal schedule profile")
    if profile.get("gpu_timing_status") != "supported":
        raise ValueError("Metal GPU timing is not supported")
    if profile.get("submissions") != 1:
        raise ValueError("Metal realtime tick did not use one submission")
    if profile.get("native_barrier_commands") != 0:
        raise ValueError("Metal emitted an unexpected global barrier command")
    if profile.get("readbacks") != 0 or profile.get("host_waits") != 0:
        raise ValueError("Metal realtime tick used a host readback or wait")
    instrumentation = profile.get("instrumentation", {})
    if any(instrumentation.get(field) != 0 for field in
           ("timestamp_queries", "forced_waits", "result_bytes")):
        raise ValueError("Metal capture changed the measured command stream")
    if ceiling is not None:
        for field, maximum in ceiling.items():
            value = profile.get(field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Metal profile has invalid {field}")
            if value > maximum:
                raise ValueError(
                    f"Metal {field} ceiling exceeded: {value} > {maximum}")
    kernels = profile.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("Metal profile has no native kernel capture")
    by_name = {}
    for kernel in kernels:
        if (not isinstance(kernel, dict) or
                not isinstance(kernel.get("name"), str) or
                not kernel["name"] or kernel["name"] in by_name or
                not isinstance(kernel.get("dispatches"), int) or
                kernel["dispatches"] <= 0 or
                not isinstance(kernel.get("workgroups"), int) or
                kernel["workgroups"] <= 0):
            raise ValueError("Metal profile has invalid native kernel capture")
        by_name[kernel["name"]] = kernel
    if sum(item["dispatches"] for item in kernels) != profile["dispatches"]:
        raise ValueError("Metal kernel dispatch sum does not match native total")
    if sum(item["workgroups"] for item in kernels) != profile["workgroups"]:
        raise ValueError("Metal kernel workgroup sum does not match native total")
    gates = KERNEL_GATES[workload]
    for name, expected in gates["required"].items():
        if name not in by_name:
            raise ValueError(f"Metal profile is missing required kernel {name}")
        for field, value in expected.items():
            if by_name[name].get(field) != value:
                raise ValueError(
                    f"Metal {name} {field} differs from fixed gate: "
                    f"{by_name[name].get(field)} != {value}")
    forbidden = sorted(gates["forbidden"] & set(by_name))
    if forbidden:
        raise ValueError(
            "Metal profile used forbidden legacy kernels: " +
            ", ".join(forbidden))
    phases = profile.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Metal profile has no phase capture")
    for field in COUNTERS:
        total = sum(phase.get(field, 0) for phase in phases)
        if total != profile.get(field):
            raise ValueError(
                f"Metal {field} phase sum does not match the native total")


def align_frames(profiles, frames, warmup):
    if any(frame["physics_steps"] < 0 for frame in frames):
        raise ValueError("frame timing contains negative physics steps")
    if sum(frame["physics_steps"] for frame in frames) != len(profiles):
        raise ValueError(
            "frame timing physics-step total does not match native profiles")
    measured_frames = []
    profile_index = 0
    for frame in frames:
        begin = profile_index
        profile_index += frame["physics_steps"]
        if begin < warmup < profile_index:
            raise ValueError("warmup boundary splits a multi-step rendered frame")
        if begin < warmup or (frame["physics_steps"] > 0 and
                              profile_index == warmup):
            continue
        measured_frames.append((frame, profiles[begin:profile_index]))
    return measured_frames


def evaluate(profiles, workload, warmup=0, minimum_samples=1, frames=None,
             priming_samples=0):
    if warmup < 0 or priming_samples < 0:
        raise ValueError("warmup and priming samples must not be negative")
    if priming_samples > len(profiles):
        raise ValueError("priming sample count exceeds native profiles")
    runtime_profiles = profiles[priming_samples:]
    measured = runtime_profiles[warmup:]
    if len(measured) < minimum_samples:
        raise ValueError(
            f"Metal capture has {len(measured)} measured samples; "
            f"{minimum_samples} required")
    ceiling = CEILINGS[workload]
    # Fixed gates apply to cold priming, warmup and measured schedules. Only
    # distributions exclude priming/warmup.
    for profile in profiles:
        validate_profile(profile, workload, ceiling)
    counters = {
        field: distribution([profile[field] for profile in measured])
        for field in COUNTERS
    }
    timing = {
        field: distribution([profile[field] for profile in measured])
        for field in ("gpu_ms", "record_ms", "submit_ms", "wait_ms",
                      "submit_to_fence_ms")
    }
    reconciliation = [{
        "context": profile["context"],
        "schedule": profile["schedule"],
        "native_dispatches": profile["dispatches"],
        "declared_launches": profile["declared_launches"],
        "native_minus_declared": (
            profile["dispatches"] - profile["declared_launches"]),
    } for profile in measured]
    result = {
        "schema": "parallel-mater-metal-schedule-evaluation/v1",
        "profile_schema": PROFILE_SCHEMA,
        "workload": workload,
        "fixed_ceiling": ceiling,
        "fixed_ceiling_status": (
            "enforced" if ceiling is not None else
            "not-applicable; observational Gallery workload"),
        "warmup_samples_excluded": warmup,
        "priming_samples": priming_samples,
        "sample_count": len(measured),
        "passed": True,
        "counters_per_tick": counters,
        "timing_ms_per_tick": timing,
        "counters_per_simulated_second": {
            field: counters[field]["mean"] * FIXED_TICK_HZ
            for field in COUNTERS
        },
        "reconciliation": reconciliation,
        "capture_contract": {
            "native_encoder_counters": True,
            "one_submission_per_tick": True,
            "host_readbacks": 0,
            "host_waits": 0,
            "native_global_barriers": 0,
            "instrumentation_forced_waits": 0,
        },
    }
    if priming_samples:
        priming = profiles[:priming_samples]
        result["cold_priming_timing_ms"] = {
            field: distribution([profile[field] for profile in priming])
            for field in ("gpu_ms", "record_ms", "submit_ms", "wait_ms",
                          "submit_to_fence_ms")
        }
    if frames is not None:
        aligned = align_frames(runtime_profiles, frames, warmup)
        if not aligned:
            raise ValueError("frame timing has no measured frames after warmup")
        per_frame = {
            field: distribution([
                sum(profile[field] for profile in frame_profiles)
                for _, frame_profiles in aligned
            ]) for field in COUNTERS
        }
        present = [frame["present_ms"] for frame, _ in aligned
                   if frame["present_ms"] is not None]
        total_wall_ms = sum(frame["total_ms"] for frame, _ in aligned)
        result["rendered_frame_count"] = len(aligned)
        result["counters_per_rendered_frame"] = per_frame
        result["frame_timing_ms"] = {
            field: distribution([frame[field] for frame, _ in aligned])
            for field in ("total_ms", "physics_ms", "render_ms", "other_ms")
        }
        result["frame_timing_ms"]["present_ms"] = (
            distribution(present) if present else {"status": "unavailable"})
        result["simulation_to_wall_clock_rate"] = (
            (len(measured) / FIXED_TICK_HZ) / (total_wall_ms / 1000.0)
            if total_wall_ms > 0.0 else 0.0)
        result["fixed_tick_hz"] = FIXED_TICK_HZ
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=pathlib.Path)
    parser.add_argument("--workload", required=True, choices=sorted(CEILINGS))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--priming-samples", type=int, default=0,
                        help="unframed scene-priming schedules captured before frame 1")
    parser.add_argument("--minimum-samples", type=int, default=1)
    parser.add_argument("--frame-timings", type=pathlib.Path,
                        help="shared Gallery CSV used for per-frame and wall-rate evidence")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.minimum_samples < 1:
        parser.error("--minimum-samples must be positive")
    profiles = load_profiles(args.profiles)
    frames = (load_frame_timings(args.frame_timings)
              if args.frame_timings else None)
    result = evaluate(profiles, args.workload, args.warmup,
                      args.minimum_samples, frames, args.priming_samples)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Metal schedule evaluation failed: {error}") from error
