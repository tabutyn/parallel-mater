#!/usr/bin/env python3
"""Capture matched Android Gallery frame windows and Vulkan schedule profiles."""

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import time


PACKAGE = "org.parallelmater.gallery"
ACTIVITY = PACKAGE + "/.GalleryActivity"
PROFILE_SCHEMA = "parallel-mater-vulkan-schedule-profile/v1"
PROFILE_PREFIX = "VulkanScheduleProfile "
SCENES = (
    {"index": 0, "id": "rigid-body", "asset": "RigidBody.glb"},
    {"index": 8, "id": "fluid", "asset": "Fluid.glb"},
    {"index": 24, "id": "smoke-soft-body", "asset": "SmokeSoftbody.glb"},
)
FRAME = re.compile(
    r"FrameWindow scene=(?P<scene>\S+) backend=(?P<backend>\S+) "
    r"frames=(?P<frames>\d+) count=(?P<count>\d+) "
    r"mean_ms=(?P<mean>[0-9.]+) median_ms=(?P<median>[0-9.]+) "
    r"p95_ms=(?P<p95>[0-9.]+) physics_ms=(?P<physics>[0-9.]+) "
    r"render_ms=(?P<render>[0-9.]+) present_ms=(?P<present>[0-9.]+) "
    r"fps=(?P<fps>[0-9.]+) particles=(?P<particles>\d+) "
    r"bodies=(?P<bodies>\d+)")
SCENE_READY = re.compile(r"Scene ready: (?P<scene>[^;]+); backend=(?P<backend>\S+);")


def parse_profiles(log):
    profiles = []
    for line in log.splitlines():
        marker = line.find(PROFILE_PREFIX)
        if marker < 0:
            continue
        profile = json.loads(line[marker + len(PROFILE_PREFIX):])
        if profile.get("schema") != PROFILE_SCHEMA:
            raise ValueError("unexpected Vulkan schedule profile schema")
        profiles.append(profile)
    return profiles


def aggregate_profile_schedules(profiles):
    grouped = {}
    for profile in profiles:
        grouped.setdefault(profile["schedule"], []).append(profile)
    schedules = []
    for schedule, submissions in sorted(grouped.items()):
        if not any(item["final"] for item in submissions):
            continue
        phases = {}
        for submission in submissions:
            for phase in submission["phases"]:
                value = phases.setdefault(phase["name"], {
                    "gpu_ms": 0.0, "dispatches": 0,
                    "declared_launches": 0, "occurrences": 0})
                for field in value:
                    value[field] += phase[field]
        schedules.append({
            "schedule": schedule,
            "submissions": len(submissions),
            "dispatches": sum(item["dispatches"] for item in submissions),
            "gpu_ms": sum(item["gpu_ms"] for item in submissions),
            "record_ms": sum(item["record_ms"] for item in submissions),
            "submit_to_fence_ms": sum(
                item["submit_to_fence_ms"] for item in submissions),
            "timestamp_overflow": any(
                item["timestamp_overflow"] for item in submissions),
            "phases": phases,
        })
    return schedules


def summarize_schedules(schedules):
    if not schedules:
        raise ValueError("no complete Vulkan schedule profiles")
    if any(item["timestamp_overflow"] for item in schedules):
        raise ValueError("Vulkan timestamp profile overflowed its bounded query pool")
    count = len(schedules)
    names = sorted({name for item in schedules for name in item["phases"]})
    phases = []
    for name in names:
        values = [item["phases"].get(name, {}) for item in schedules]
        phases.append({
            "name": name,
            "mean_gpu_ms": sum(item.get("gpu_ms", 0.0) for item in values) / count,
            "mean_dispatches": sum(
                item.get("dispatches", 0) for item in values) / count,
            "total_occurrences": sum(
                item.get("occurrences", 0) for item in values),
        })
    return {
        "schedule_count": count,
        "mean_submissions": sum(item["submissions"] for item in schedules) / count,
        "mean_dispatches": sum(item["dispatches"] for item in schedules) / count,
        "mean_gpu_ms": sum(item["gpu_ms"] for item in schedules) / count,
        "mean_record_ms": sum(item["record_ms"] for item in schedules) / count,
        "mean_submit_to_fence_ms": sum(
            item["submit_to_fence_ms"] for item in schedules) / count,
        "phases": phases,
    }


def parse_frame_windows(log):
    values = []
    for match in FRAME.finditer(log):
        item = match.groupdict()
        for field in ("frames", "count", "particles", "bodies"):
            item[field] = int(item[field])
        for field in ("mean", "median", "p95", "physics", "render", "present"):
            item[field + "_ms"] = float(item.pop(field))
        item["fps"] = float(item["fps"])
        values.append(item)
    return values


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--schedules", type=int, default=60)
    parser.add_argument("--scene-timeout", type=int, default=240)
    parser.add_argument("--apk", type=pathlib.Path, default=pathlib.Path(
        "examples/android/app/build/outputs/apk/debug/app-debug.apk"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(
        "out/android/vulkan-schedule-profile"))
    args = parser.parse_args()
    if not 1 <= args.schedules <= 600:
        parser.error("--schedules must be 1..600")
    args.output.mkdir(parents=True, exist_ok=True)

    def adb(*command):
        return subprocess.run(
            [args.adb, "-s", args.serial, *command], check=True,
            text=True, capture_output=True).stdout

    props = {
        "manufacturer": adb("shell", "getprop", "ro.product.manufacturer").strip(),
        "model": adb("shell", "getprop", "ro.product.model").strip(),
        "product": adb("shell", "getprop", "ro.product.name").strip(),
        "android_version": adb(
            "shell", "getprop", "ro.build.version.release").strip(),
        "api_level": adb("shell", "getprop", "ro.build.version.sdk").strip(),
    }
    records = []
    logs = []
    for scene in SCENES:
        adb("shell", "am", "force-stop", PACKAGE)
        adb("logcat", "-c")
        adb("shell", "am", "start", "-W", "-f", "0x20000000", "-n", ACTIVITY,
            "--ez", "pm_vulkan_profile", "true", "--ei", "pm_scene",
            str(scene["index"]), "--es", "pm_action", "unfreeze")
        deadline = time.monotonic() + args.scene_timeout
        selected = []
        windows = []
        log = ""
        while time.monotonic() < deadline:
            log = adb("logcat", "-d", "-s", "ParallelMater", "AndroidRuntime")
            ready = list(SCENE_READY.finditer(log))
            if ready and ready[-1].group("scene") == scene["asset"]:
                section = log[ready[-1].start():]
                windows = parse_frame_windows(section)
                schedules = aggregate_profile_schedules(parse_profiles(section))
                selected = schedules[-args.schedules:]
            if windows and len(selected) == args.schedules:
                break
            time.sleep(2)
        logs.append(f"===== {scene['id']} =====\n{log}")
        if not windows or len(selected) != args.schedules:
            raise RuntimeError(f"timed out profiling {scene['id']}")
        window = windows[-1]
        if (window["scene"] != scene["asset"] or window["backend"] != "vulkan" or
                window["count"] != 60):
            raise RuntimeError(f"invalid frame window for {scene['id']}")
        summary = summarize_schedules(selected)
        records.append({
            **scene,
            "frame_window": window,
            "effective_simulation_to_wall_ratio": window["fps"] / 30.0,
            "schedule_profile": summary,
        })
        print(f"{scene['id']}: physics {window['physics_ms']:.3f} ms, "
              f"GPU {summary['mean_gpu_ms']:.3f} ms, "
              f"{summary['mean_dispatches']:.1f} dispatches", flush=True)

    runtime = json.loads(adb(
        "shell", "run-as", PACKAGE, "cat", "files/runtime-info.json"))
    thermal = adb("shell", "dumpsys", "thermalservice")
    adb("shell", "am", "force-stop", PACKAGE)
    fatal_patterns = ("FATAL EXCEPTION", "Fatal signal", "SIGSEGV",
                      "Validation Error", "VUID-")
    failures = [line for line in "\n".join(logs).splitlines()
                if any(pattern in line for pattern in fatal_patterns)]
    report = {
        "schema": "parallel-mater-android-vulkan-schedule-diagnostic/v1",
        "revision": runtime.get("revision"),
        "device": props,
        "backend": runtime.get("backend"),
        "presentation": runtime.get("presentation"),
        "target_fps": 30,
        "solver_substeps_per_tick": 8,
        "schedules_per_scene": args.schedules,
        "apk_sha256": sha256(args.apk),
        "runner_sha256": sha256(pathlib.Path(__file__)),
        "scenes": records,
        "failures": failures,
        "passed": not failures and runtime.get("presentation") == "vulkan-direct",
        "notes": [
            "Timestamp queries are recorded in the production schedule command buffer; telemetry adds no queue submission or per-dispatch host wait.",
            "FrameWindow host physics/render times and schedule GPU times cover the same final 60-frame window.",
            "Effective simulation/wall rate is observed FPS divided by the unchanged 30 Hz fixed-step target.",
        ],
    }
    (args.output / "profile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "device.log").write_text("\n".join(logs))
    (args.output / "thermal.txt").write_text(thermal)
    if not report["passed"]:
        raise RuntimeError("Android Vulkan schedule diagnostic failed")


if __name__ == "__main__":
    main()
