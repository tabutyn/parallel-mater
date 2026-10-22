#!/usr/bin/env python3
"""Install, verify and capture fixed Pixel 7 legacy Vulkan baselines."""

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import android_deployment as deployment  # noqa: E402


PACKAGE = deployment.PACKAGE
ACTIVITY = PACKAGE + "/.GalleryActivity"
PROFILE_SCHEMA = "parallel-mater-vulkan-schedule-profile/v1"
PROFILE_PREFIX = "VulkanScheduleProfile "
CHUNK = re.compile(
    r"VulkanScheduleProfileChunk (?P<id>\d+) (?P<index>\d+)/(?P<count>\d+) (?P<data>.*)$")
SCENES = (
    {"index": 0, "id": "rigid-body", "asset": "RigidBody.glb"},
    {"index": 8, "id": "fluid", "asset": "Fluid.glb"},
    {"index": 25, "id": "smoke-soft-body", "asset": "SmokeSoftbody.glb"},
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
TEMPERATURE = re.compile(
    r"Temperature\{mValue=(?P<value>-?[0-9.]+), mType=(?P<type>-?\d+), "
    r"mName=(?P<name>[^,]+), mStatus=(?P<status>\d+)\}")


def parse_profiles(log):
    profiles = []
    chunks = {}
    for line in log.splitlines():
        marker = line.find(PROFILE_PREFIX)
        if marker >= 0:
            profiles.append(json.loads(line[marker + len(PROFILE_PREFIX):]))
            continue
        match = CHUNK.search(line)
        if not match:
            continue
        identifier = match.group("id")
        count = int(match.group("count"))
        state = chunks.setdefault(identifier, {"count": count, "parts": {}})
        if state["count"] != count:
            raise ValueError("inconsistent Vulkan schedule chunk count")
        index = int(match.group("index"))
        if not 1 <= index <= count or index in state["parts"]:
            raise ValueError("invalid or duplicate Vulkan schedule chunk")
        state["parts"][index] = match.group("data")
        if len(state["parts"]) == count:
            profiles.append(json.loads("".join(
                state["parts"][part] for part in range(1, count + 1))))
            del chunks[identifier]
    if chunks:
        raise ValueError("incomplete Vulkan schedule profile chunks")
    for profile in profiles:
        if profile.get("schema") != PROFILE_SCHEMA:
            raise ValueError("unexpected Vulkan schedule profile schema")
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


def summarize_schedules(schedules):
    if not schedules:
        raise ValueError("no complete Vulkan schedule profiles")
    if any(item["timestamp_overflow"] for item in schedules):
        raise ValueError("Vulkan timestamp profile overflowed its bounded query pool")
    names = sorted({name for item in schedules for name in item["phases"]})
    phases = []
    for name in names:
        values = [item["phases"].get(name, {}) for item in schedules]
        phases.append({
            "name": name,
            "gpu_ms": distribution([
                item.get("gpu_ms", 0.0) for item in values]),
            "dispatches": distribution([
                item.get("dispatches", 0) for item in values]),
            "total_occurrences": sum(
                item.get("occurrences", 0) for item in values),
        })
    return {
        "schedule_count": len(schedules),
        "submissions_per_tick": distribution([
            item["submissions"] for item in schedules]),
        "dispatches_per_tick": distribution([
            item["dispatches"] for item in schedules]),
        "gpu_ms_per_tick": distribution([
            item["gpu_ms"] for item in schedules]),
        "cpu_record_ms_per_tick": distribution([
            item["record_ms"] for item in schedules]),
        "cpu_submit_to_fence_ms_per_tick": distribution([
            item["submit_to_fence_ms"] for item in schedules]),
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


def parse_thermal(text):
    status = re.search(r"^Thermal Status:\s*(\d+)$", text, re.MULTILINE)
    current = text.split("Current temperatures from HAL:", 1)[-1]
    current = current.split("Current cooling devices from HAL:", 1)[0]
    temperatures = {}
    sensor_statuses = {}
    for match in TEMPERATURE.finditer(current):
        temperatures[match.group("name")] = float(match.group("value"))
        sensor_statuses[match.group("name")] = int(match.group("status"))
    if not status or not temperatures:
        raise RuntimeError("could not parse Android thermal-service state")
    return {"status": int(status.group(1)), "temperatures_c": temperatures,
            "sensor_statuses": sensor_statuses}


def read_json(adb, name):
    return json.loads(adb.text(
        "shell", "run-as", PACKAGE, "cat", f"files/{name}"))


def wait_for_report(adb, name, nonce, deadline):
    last_error = None
    while time.monotonic() < deadline:
        try:
            report = read_json(adb, name)
            if report.get("nonce") == nonce:
                return report
        except (json.JSONDecodeError, OSError, RuntimeError,
                subprocess.CalledProcessError) as error:
            last_error = error
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for fresh {name}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--schedules", type=int, default=60)
    parser.add_argument("--scene-timeout", type=int, default=240)
    parser.add_argument("--scene", choices=[scene["id"] for scene in SCENES],
                        action="append")
    parser.add_argument("--apk", type=pathlib.Path, default=pathlib.Path(
        "examples/android/app/build/outputs/apk/debug/app-debug.apk"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(
        "out/android/vulkan-schedule-profile"))
    args = parser.parse_args()
    if not 1 <= args.schedules <= 600:
        parser.error("--schedules must be 1..600")
    if args.scene_timeout < 10:
        parser.error("--scene-timeout must be at least 10 seconds")
    root = pathlib.Path(__file__).resolve().parents[1]
    source = deployment.source_identity(root)
    args.output.mkdir(parents=True, exist_ok=True)
    adb = deployment.Adb(args.adb, args.serial)
    installed = deployment.install_and_verify(adb, args.apk)
    run_nonce = deployment.fresh_nonce()
    chosen = [scene for scene in SCENES
              if not args.scene or scene["id"] in args.scene]

    def thermal():
        return parse_thermal(adb.text("shell", "dumpsys", "thermalservice"))

    thermal_start = thermal()
    records = []
    logs = []
    for scene in chosen:
        nonce = f"{run_nonce}-{scene['id']}"
        adb.text("shell", "am", "force-stop", PACKAGE)
        adb.text("logcat", "-c")
        adb.text("shell", "am", "start", "-W", "-f", "0x20000000",
                 "-n", ACTIVITY, "--ez", "pm_vulkan_profile", "true",
                 "--ei", "pm_scene", str(scene["index"]), "--es",
                 "pm_action", "unfreeze", "--es", "pm_nonce", nonce)
        deadline = time.monotonic() + args.scene_timeout
        selected = []
        windows = []
        log = ""
        while time.monotonic() < deadline:
            log = adb.text("logcat", "-d", "-s", "ParallelMater", "AndroidRuntime")
            ready = list(SCENE_READY.finditer(log))
            if ready and ready[-1].group("scene") == scene["asset"]:
                section = log[ready[-1].start():]
                window_matches = list(FRAME.finditer(section))
                if window_matches:
                    matched = section[:window_matches[-1].end()]
                    windows = parse_frame_windows(matched)
                    schedules = aggregate_profile_schedules(parse_profiles(matched))
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
        adb.text("shell", "am", "start", "-W", "-f", "0x20000000",
                 "-n", ACTIVITY, "--es", "pm_action", "freeze", "--ez",
                 "pm_capture_workload", "true", "--es", "pm_nonce", nonce)
        workload = wait_for_report(adb, "workload-info.json", nonce, deadline)
        runtime = wait_for_report(adb, "runtime-info.json", nonce, deadline)
        deployment.verify_runtime_identity(runtime, nonce, source)
        if workload.get("scene") != scene["asset"]:
            raise RuntimeError(f"workload snapshot is for the wrong scene: {scene['id']}")
        summary = summarize_schedules(selected)
        records.append({
            **scene,
            "nonce": nonce,
            "frame_window": window,
            "workload": workload,
            "effective_simulation_to_wall_ratio": window["fps"] / 30.0,
            "schedule_profile": summary,
            "schedule_samples": selected,
            "thermal": thermal(),
        })
        print(f"{scene['id']}: physics {window['physics_ms']:.3f} ms, "
              f"GPU {summary['gpu_ms_per_tick']['mean']:.3f} ms, "
              f"{summary['dispatches_per_tick']['mean']:.1f} dispatches", flush=True)

    thermal_end = thermal()
    adb.text("shell", "am", "force-stop", PACKAGE)
    fatal_patterns = ("FATAL EXCEPTION", "Fatal signal", "SIGSEGV",
                      "Validation Error", "VUID-")
    failures = [line for line in "\n".join(logs).splitlines()
                if any(pattern in line for pattern in fatal_patterns)]
    max_thermal_status = max(
        [thermal_start["status"], thermal_end["status"]] +
        [record["thermal"]["status"] for record in records])
    report = {
        "schema": "parallel-mater-android-vulkan-legacy-baseline/v2",
        "source": source,
        "deployment": installed,
        "run_nonce": run_nonce,
        "legacy_profile": {"target_fps": 30, "solver_substeps_per_tick": 8},
        "schedules_per_scene": args.schedules,
        "artifacts": {
            "runner_sha256": deployment.sha256_path(pathlib.Path(__file__)),
            "deployment_helper_sha256": deployment.sha256_path(
                pathlib.Path(deployment.__file__)),
        },
        "thermal_start": thermal_start,
        "thermal_end": thermal_end,
        "max_thermal_status": max_thermal_status,
        "scenes": records,
        "failures": failures,
        "passed": not failures and max_thermal_status == 0,
        "notes": [
            "The runner installed the named local APK and byte-compared the installed base APK and bundled Vulkan driver before launching it.",
            "Every scene and runtime report is bound to a fresh nonce and the APK-embedded revision/dirty-patch/toolchain identity.",
            "Timestamp queries use the production schedule command buffer; telemetry adds no queue submission or per-dispatch host wait.",
            "FrameWindow CPU timings and schedule GPU timings cover the same final 60-frame window.",
            "Workload readback occurs only after freezing and outside the measured window.",
            "Effective simulation/wall rate is observed FPS divided by the unchanged legacy 30 Hz fixed-step target.",
        ],
    }
    (args.output / "profile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "device.log").write_text("\n".join(logs))
    if not report["passed"]:
        raise RuntimeError("Android Vulkan legacy baseline failed")


if __name__ == "__main__":
    main()
