#!/usr/bin/env python3
"""Measure sustained ordinary Android Gallery presentation on every scene."""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
import zipfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import android_deployment as deployment  # noqa: E402


PACKAGE = "org.parallelmater.gallery"
ACTIVITY = PACKAGE + "/.GalleryActivity"
FRAME = re.compile(
    r"FrameWindow scene=(?P<scene>\S+) backend=(?P<backend>\S+) "
    r"frames=(?P<frames>\d+) count=(?P<count>\d+) "
    r"mean_ms=(?P<mean>[0-9.]+) median_ms=(?P<median>[0-9.]+) "
    r"p95_ms=(?P<p95>[0-9.]+) physics_ms=(?P<physics>[0-9.]+) "
    r"render_ms=(?P<render>[0-9.]+) present_ms=(?P<present>[0-9.]+) "
    r"fps=(?P<fps>[0-9.]+) particles=(?P<particles>\d+) "
    r"bodies=(?P<bodies>\d+)")
SCENE_READY = re.compile(
    r"Scene ready: (?P<scene>[^;]+); backend=(?P<backend>\S+);")
TEMPERATURE = re.compile(
    r"Temperature\{mValue=(?P<value>-?[0-9.]+), mType=(?P<type>-?\d+), "
    r"mName=(?P<name>[^,]+), mStatus=(?P<status>\d+)\}")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def parse_frame_windows(log):
    values = []
    for match in FRAME.finditer(log):
        item = match.groupdict()
        for field in ("frames", "count", "particles", "bodies"):
            item[field] = int(item[field])
        for field in ("mean", "median", "p95", "physics", "render",
                      "present", "fps"):
            item[field + "_ms" if field not in {"fps"} else field] = float(
                item.pop(field))
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
        raise RuntimeError("Could not parse Android thermal-service state")
    return {
        "status": int(status.group(1)),
        "temperatures_c": temperatures,
        "sensor_statuses": sensor_statuses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--scene-timeout", type=int, default=240)
    parser.add_argument("--qualification", type=pathlib.Path, required=True)
    parser.add_argument("--apk", type=pathlib.Path, default=pathlib.Path(
        "examples/android/app/build/outputs/apk/debug/app-debug.apk"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(
        "out/android/gallery-performance"))
    args = parser.parse_args()
    if not 1 <= args.windows <= 20:
        parser.error("--windows must be 1..20")
    if args.scene_timeout < 10:
        parser.error("--scene-timeout must be at least 10 seconds")

    qualification = json.loads(args.qualification.read_text())
    scenes = qualification.get("scenes", [])
    if len(scenes) != 28 or not all(scene.get("available") for scene in scenes):
        raise RuntimeError("Performance sweep requires 28 available qualified scenes")
    revision = qualification["revision"]
    root = pathlib.Path(__file__).resolve().parents[1]
    source = deployment.source_identity(root)
    if revision != source["revision"]:
        raise RuntimeError("qualification revision does not match current source")
    args.output.mkdir(parents=True, exist_ok=True)
    adb_client = deployment.Adb(args.adb, args.serial)
    installed = deployment.install_and_verify(adb_client, args.apk)
    nonce = deployment.fresh_nonce()

    def adb(*command):
        return adb_client.text(*command)

    def thermal():
        return parse_thermal(adb("shell", "dumpsys", "thermalservice"))

    props = {
        "manufacturer": adb("shell", "getprop", "ro.product.manufacturer").strip(),
        "model": adb("shell", "getprop", "ro.product.model").strip(),
        "product": adb("shell", "getprop", "ro.product.name").strip(),
        "android_version": adb("shell", "getprop", "ro.build.version.release").strip(),
        "api_level": adb("shell", "getprop", "ro.build.version.sdk").strip(),
        "security_patch": adb("shell", "getprop", "ro.build.version.security_patch").strip(),
    }
    with zipfile.ZipFile(args.apk) as archive:
        driver = archive.read(
            "lib/arm64-v8a/libparallel_mater_driver_slang_vulkan.so")

    start = thermal()
    records = []
    logs = []
    started = time.monotonic()
    for index, scene in enumerate(scenes):
        adb("logcat", "-c")
        adb("shell", "am", "start", "-f", "0x20000000", "-n", ACTIVITY,
            "--ei", "pm_scene", str(index), "--es", "pm_action", "unfreeze",
            "--es", "pm_nonce", nonce)
        deadline = time.monotonic() + args.scene_timeout
        windows = []
        log = ""
        asset = ""
        while time.monotonic() < deadline:
            log = adb("logcat", "-d", "-s", "ParallelMater", "AndroidRuntime")
            ready = list(SCENE_READY.finditer(log))
            if ready:
                asset = ready[-1].group("scene")
                windows = parse_frame_windows(log[ready[-1].start():])
            if len(windows) >= args.windows:
                break
            time.sleep(2)
        logs.append(f"===== {index:02d} {scene['id']} =====\n{log}")
        if len(windows) < args.windows:
            raise RuntimeError(
                f"Timed out waiting for {args.windows} frame windows in {scene['id']}")
        selected = windows[-args.windows:]
        if (not asset or any(item["scene"] != asset for item in selected) or
                any(item["backend"] != "vulkan" or item["count"] != 60 or
                    item["fps"] <= 0.0 for item in selected)):
            raise RuntimeError(f"Invalid frame window for {scene['id']}")
        sample = thermal()
        records.append({
            "index": index,
            "id": scene["id"],
            "scene": scene["scene"],
            "asset": asset,
            "windows": selected,
            "thermal": sample,
            "sustained_fps_ratio": selected[-1]["fps"] / selected[0]["fps"],
        })
        print(f"{index + 1}/28 {scene['id']}: "
              f"{selected[-1]['fps']:.2f} FPS, thermal {sample['status']}",
              flush=True)

    runtime = json.loads(adb(
        "shell", "run-as", PACKAGE, "cat", "files/runtime-info.json"))
    deployment.verify_runtime_identity(runtime, nonce, source)
    end = thermal()
    adb("shell", "input", "keyevent", "KEYCODE_BACK")
    all_logs = "\n".join(logs)
    fatal_patterns = ("FATAL EXCEPTION", "Fatal signal", "SIGSEGV",
                      "Validation Error", "VUID-")
    failures = [line for line in all_logs.splitlines()
                if any(pattern in line for pattern in fatal_patterns)]
    max_thermal_status = max(
        [start["status"], end["status"]] +
        [record["thermal"]["status"] for record in records])
    direct = (runtime.get("revision") == revision and
              runtime.get("backend") == "vulkan" and
              runtime.get("presentation") == "vulkan-direct" and
              runtime.get("ordinary_readback_pixels") == 0 and
              runtime.get("software_presentations") == 0)
    passed = not failures and max_thermal_status == 0 and direct
    report = {
        "schema": "parallel-mater-android-gallery-performance/v1",
        "revision": revision,
        "source": source,
        "deployment": installed,
        "run_nonce": nonce,
        "device": props,
        "backend": "vulkan",
        "artifacts": {
            "apk_sha256": sha256_bytes(args.apk.read_bytes()),
            "driver_sha256": sha256_bytes(driver),
            "runner_sha256": sha256_bytes(pathlib.Path(__file__).read_bytes()),
            "qualification_sha256": sha256_bytes(
                args.qualification.read_bytes()),
        },
        "windows_per_scene": args.windows,
        "frames_per_window": 60,
        "duration_seconds": time.monotonic() - started,
        "direct_presentation": direct,
        "max_thermal_status": max_thermal_status,
        "thermal_start": start,
        "thermal_end": end,
        "scenes": records,
        "failures": failures,
        "passed": passed,
        "notes": [
            "FrameWindow measures ordinary direct presentation without diagnostic readback.",
            "The profile target FPS controls the fixed simulation clock and is not a promise that every scene reaches that display rate.",
        ],
    }
    (args.output / "performance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "device.log").write_text(all_logs)
    if not passed:
        raise RuntimeError("Android Gallery sustained performance qualification failed")
    print(f"PASS: 28 scenes, {args.windows * 60} ordinary frames each, "
          "direct Vulkan presentation, no thermal throttling", revision)


if __name__ == "__main__":
    main()
