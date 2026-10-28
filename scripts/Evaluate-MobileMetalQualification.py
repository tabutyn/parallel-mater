#!/usr/bin/env python3
"""Reconcile a mobile Gallery report with native Metal schedule captures."""

import argparse
import collections
import hashlib
import json
import pathlib
import plistlib
import statistics
import sys


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mobile_gallery_report import validate_mobile_gallery_report  # noqa: E402


PROFILE_SCHEMA = "parallel-mater-metal-schedule-profile/v1"
REPORT_SCHEMA = "parallel-mater-mobile-metal-qualification/v1"
FIXED_TICK_HZ = 60
MEASUREMENT_OVERHEAD_SCHEDULES = 2  # initialization plus resume check
REQUIRED_COVERAGE = (
    "rigid-body",
    "fluid",
    "cloth",
    "soft-body",
    "soft-body-fluid",
    "smoke-soft-body",
)
FIXED_CEILINGS = {
    "rigid-body": {
        "dispatches": 40,
        "dependency_barriers": 45,
        "declared_launches": 40,
    },
    "smoke-soft-body": {
        "dispatches": 288,
        "workgroups": 67000,
        "dependency_barriers": 672,
        "declared_launches": 288,
    },
}
KERNEL_GATES = {
    "smoke-soft-body": {
        "required": {
            "deformable_collide_batch": 8,
            "reduce_point_body_impulses_batch": 8,
            "soft_body_apply_contact_friction_batch": 8,
            "smoke_soft_body_contact_batch": 8,
            "smoke_grid_soft_body_force_batch": 8,
            "pm_spatial_publish_epoch": 1,
        },
        "forbidden": {
            "deformable_collide",
            "reduce_point_body_impulses",
            "soft_body_apply_contact_friction",
            "smoke_soft_body_contact",
            "smoke_grid_soft_body_force",
        },
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    if not values:
        raise ValueError("percentile requires values")
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


def load_profiles(path):
    profiles = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}") from error
            if value.get("schema") != PROFILE_SCHEMA:
                raise ValueError(
                    f"{path}:{line_number}: unexpected Metal profile schema")
            profiles.append(value)
    if not profiles:
        raise ValueError("Metal profile is empty")
    return profiles


def expected_native(scene):
    work = scene["realtime_work"]
    return {
        "dispatches": work["dispatches"],
        "submissions": work["submissions"],
    }


def matches(profile, expected):
    return all(profile.get(field) == value for field, value in expected.items())


def validate_native_profile(profile, scene_id, expected):
    if profile.get("final") is not True:
        raise ValueError(f"{scene_id}: native profile is incomplete")
    if profile.get("gpu_timing_status") != "supported":
        raise ValueError(f"{scene_id}: native GPU timing is unavailable")
    if not matches(profile, expected):
        raise ValueError(
            f"{scene_id}: native/public dispatch or submission count disagrees")
    if profile.get("native_barrier_commands") != 0:
        raise ValueError(f"{scene_id}: unexpected native global barrier")
    if profile.get("readbacks") != 0 or profile.get("host_waits") != 0:
        raise ValueError(f"{scene_id}: native tick waited for or read from host")
    instrumentation = profile.get("instrumentation", {})
    if any(instrumentation.get(field) != 0 for field in
           ("timestamp_queries", "forced_waits", "result_bytes")):
        raise ValueError(f"{scene_id}: capture changed the measured stream")
    if profile.get("declared_launches") != profile.get("dispatches"):
        raise ValueError(f"{scene_id}: declared/native dispatches disagree")
    kernels = profile.get("kernels")
    if (not isinstance(kernels, list) or
            sum(item.get("dispatches", -1) for item in kernels) !=
            profile["dispatches"] or
            sum(item.get("workgroups", -1) for item in kernels) !=
            profile["workgroups"]):
        raise ValueError(f"{scene_id}: native kernel totals disagree")
    gate = KERNEL_GATES.get(scene_id)
    if gate:
        by_name = {item["name"]: item for item in kernels}
        for name, dispatches in gate["required"].items():
            if (name not in by_name or
                    by_name[name].get("dispatches") != dispatches):
                raise ValueError(
                    f"{scene_id}: required batch kernel gate failed: {name}")
        forbidden = sorted(gate["forbidden"] & set(by_name))
        if forbidden:
            raise ValueError(
                f"{scene_id}: forbidden legacy kernels: " +
                ", ".join(forbidden))
    phases = profile.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"{scene_id}: native phase capture is missing")
    for field in ("dispatches", "workgroups", "dependency_barriers",
                  "dependency_ranges", "transfer_commands", "readbacks",
                  "host_waits", "bytes_moved", "declared_launches"):
        if sum(phase.get(field, 0) for phase in phases) != profile.get(field):
            raise ValueError(f"{scene_id}: native {field} phase sum disagrees")


def reconcile(report, profiles):
    frames = report["frames_per_scene"]
    grouped = collections.OrderedDict()
    for profile in profiles:
        grouped.setdefault(profile["context"], []).append(profile)
    contexts = list(grouped.items())
    matched = {}
    search_at = 0
    for scene in (item for item in report["scenes"] if item["available"]):
        expected = expected_native(scene)
        found = None
        for position in range(search_at, len(contexts)):
            context, values = contexts[position]
            if len(values) != frames + MEASUREMENT_OVERHEAD_SCHEDULES:
                continue
            measured = values[1:1 + frames]
            if all(matches(value, expected) for value in measured):
                found = (position, context, values, measured)
                break
        if found is None:
            raise ValueError(
                f"{scene['id']}: no bounded native context matches the "
                "public qualification window")
        position, context, values, measured = found
        if [value.get("schedule") for value in values] != \
                list(range(1, len(values) + 1)):
            raise ValueError(f"{scene['id']}: native schedule sequence has gaps")
        for profile in measured:
            validate_native_profile(profile, scene["id"], expected)
        matched[scene["id"]] = (context, measured)
        search_at = position + 1
    return matched


def binary_identity(app):
    with (app / "Info.plist").open("rb") as stream:
        executable = plistlib.load(stream)["CFBundleExecutable"]
    return {
        "app_executable_sha256": sha256_file(app / executable),
        "driver_sha256": sha256_file(
            app / "Frameworks/ParallelMaterMetal.framework/ParallelMaterMetal"),
    }


def evaluate(report, profiles, assets):
    validate_mobile_gallery_report(report)
    if report["backend"] != "metal":
        raise ValueError("mobile qualification did not use Metal")
    matched = reconcile(report, profiles)
    by_id = {scene["id"]: scene for scene in report["scenes"]}
    missing = [scene for scene in REQUIRED_COVERAGE
               if scene not in by_id or not by_id[scene]["available"]]
    if missing:
        raise ValueError("required Gallery coverage unavailable: " +
                         ", ".join(missing))

    results = []
    for scene in (item for item in report["scenes"] if item["available"]):
        context, measured = matched[scene["id"]]
        work = scene["realtime_work"]
        if (work["time_advances"] != 1 or work["spatial_builds"] != 1 or
                work["spatial_publications"] != 1 or
                work["intra_tick_host_readbacks"] != 0 or
                work["intra_tick_host_waits"] != 0 or
                work["numerical_iterations"] <= 0):
            raise ValueError(
                f"{scene['id']}: public fixed-tick work contract failed")
        asset = assets / scene["scene_asset"]
        if not asset.is_file():
            raise ValueError(f"{scene['id']}: scene asset is missing: {asset}")
        native = {
            field: distribution([profile[field] for profile in measured])
            for field in (
                "dispatches", "workgroups", "dependency_barriers",
                "dependency_ranges", "transfer_commands", "bytes_moved",
                "declared_launches", "gpu_ms", "record_ms", "submit_ms",
                "wait_ms", "submit_to_fence_ms")
        }
        fixed_gate = None
        ceiling = FIXED_CEILINGS.get(scene["id"])
        if ceiling:
            failures = {
                field: {"observed": native[field]["maximum"], "maximum": maximum}
                for field, maximum in ceiling.items()
                if native[field]["maximum"] > maximum
            }
            fixed_gate = {
                "ceiling": ceiling,
                "passed": not failures,
                "failures": failures,
            }
            if failures:
                raise ValueError(
                    f"{scene['id']}: frozen native-work ceiling failed")
        results.append({
            "id": scene["id"],
            "scene": scene["scene"],
            "scene_asset": scene["scene_asset"],
            "scene_asset_sha256": sha256_file(asset),
            "requirements": scene["requirements"],
            "workload_profile": scene["workload_profile"],
            "workload": scene["workload"],
            "native_context": context,
            "measured_ticks": len(measured),
            "frame_timing_ms": scene["timing_ms"],
            "simulation_to_wall_clock_rate":
                scene["simulation_to_wall_clock_rate"],
            "thermal": {
                "before": scene["thermal_state_before"],
                "after": scene["thermal_state_after"],
            },
            "memory_footprint_bytes": scene["memory_footprint_bytes"],
            "public_work_per_tick": work,
            "native_per_tick": native,
            "native_minus_public_per_tick": {
                "dispatches": native["dispatches"]["mean"] -
                    work["dispatches"],
                "workgroups": native["workgroups"]["mean"] -
                    work["workgroups"],
                "barriers": native["dependency_barriers"]["mean"] -
                    work["barriers"],
                "bytes_moved": native["bytes_moved"]["mean"] -
                    work["bytes_moved"],
            },
            "native_per_simulated_second": {
                field: native[field]["mean"] * FIXED_TICK_HZ
                for field in ("dispatches", "workgroups",
                              "dependency_barriers", "bytes_moved")
            },
            "fixed_gate": fixed_gate,
        })
    return {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "source": {
            "revision": report["revision"],
            "platform": report["platform"],
            "backend": report["backend"],
            "execution_contract": report["execution_contract"],
        },
        "device": report["device"],
        "measurement_contract": {
            "fixed_tick_hz": FIXED_TICK_HZ,
            "frames_per_scene": report["frames_per_scene"],
            "native_profile_schema": PROFILE_SCHEMA,
            "initialization_samples_excluded_per_scene": 1,
            "resume_samples_excluded_per_scene": 1,
            "native_public_counter_reconciliation": True,
            "instrumentation_forced_waits": 0,
            "instrumentation_timestamp_queries": 0,
        },
        "required_coverage": list(REQUIRED_COVERAGE),
        "scenes": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=pathlib.Path)
    parser.add_argument("profile", type=pathlib.Path)
    parser.add_argument("--assets", type=pathlib.Path,
                        default=SCRIPT_DIR.parent / "examples/assets")
    parser.add_argument("--build-info", type=pathlib.Path)
    parser.add_argument("--app", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.qualification.read_text(encoding="utf-8"))
    result = evaluate(report, load_profiles(args.profile), args.assets)
    result["artifacts"] = {
        "qualification_sha256": sha256_file(args.qualification),
        "native_profile_sha256": sha256_file(args.profile),
    }
    if args.build_info:
        build = json.loads(args.build_info.read_text(encoding="utf-8"))
        if build.get("revision") != report["revision"]:
            raise ValueError("build-info revision differs from qualification")
        result["build_info"] = build
        result["artifacts"]["build_info_sha256"] = sha256_file(args.build_info)
    if args.app:
        result["installed_binary_identity"] = binary_identity(args.app)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Mobile Metal qualification: {len(result['scenes'])} scenes passed")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Mobile Metal qualification failed: {error}") from error
