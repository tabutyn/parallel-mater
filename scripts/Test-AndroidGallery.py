#!/usr/bin/env python3
"""Exercise installed debug Gallery scenes and actual Android lifecycle callbacks."""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import uuid

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mobile_gallery_report import validate_mobile_gallery_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("out/android/gallery-tests"))
    args = parser.parse_args()
    if not 1 <= args.frames <= 100000:
        parser.error("--frames must be 1..100000")
    package = "org.parallelmater.gallery"
    nonce = str(uuid.uuid4())

    def adb(*command):
        return subprocess.run([args.adb, "-s", args.serial, *command], check=True,
                              text=True, capture_output=True).stdout

    def wait(name, predicate, timeout=60):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                value = json.loads(adb("shell", "run-as", package, "cat", "files/" + name))
                if value.get("nonce") == nonce and predicate(value):
                    return value
            except (subprocess.CalledProcessError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError(f"Timeout waiting for {name}; use a new debug APK and unlock the device")

    def direct(report):
        return (report.get("backend") != "vulkan" or
                (report.get("presentation") == "vulkan-direct" and
                 report.get("ordinary_readback_pixels") == 0 and
                 report.get("software_presentations") == 0))

    activity = package + "/.GalleryActivity"
    adb("logcat", "-c")
    adb("shell", "am", "force-stop", package)
    adb("shell", "am", "start", "-n", activity, "--ei", "pm_qualify_frames", str(args.frames), "--es", "pm_nonce", nonce)
    report = wait("qualification.json", lambda r: r.get("completed"), args.timeout)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    supported, unsupported = validate_mobile_gallery_report(report, nonce=nonce)
    scenes = report["scenes"]
    before = wait("runtime-info.json", lambda r: r["active"] and direct(r))
    adb("shell", "input", "keyevent", "KEYCODE_HOME")
    background = wait("runtime-info.json", lambda r: not r["active"] and r["backgrounds"] > before["backgrounds"])
    time.sleep(5)
    adb("shell", "am", "start", "-f", "0x20000000", "-n", activity)
    foreground = wait("runtime-info.json", lambda r:
                      r["foregrounds"] > background["foregrounds"])
    if foreground["frames"] != background["frames"]:
        raise RuntimeError("Simulation advanced while backgrounded")
    resumed = wait("runtime-info.json", lambda r:
                   r["foregrounds"] > background["foregrounds"] and
                   r.get("surface_ready") and direct(r))
    if (report.get("backend") == "vulkan" and
            resumed.get("surface_generation", 0) <=
            before.get("surface_generation", 0)):
        raise RuntimeError("Android surface was not recreated after backgrounding")
    (args.output / "lifecycle.json").write_text(json.dumps({
        "background": background, "foreground": foreground,
        "resumed_surface": resumed}, indent=2) + "\n")

    size_state = adb("shell", "wm", "size").strip()
    physical = re.search(r"Physical size:\s*(\d+)x(\d+)", size_state)
    if not physical:
        raise RuntimeError("Could not determine the physical Android display size")
    physical_size = tuple(map(int, physical.groups()))
    prior_override = re.search(r"Override size:\s*(\d+x\d+)", size_state)
    test_size = (max(640, physical_size[0] * 3 // 4),
                 max(360, physical_size[1] * 3 // 4))
    if f"{test_size[0]}x{test_size[1]}" == (prior_override.group(1)
                                              if prior_override else ""):
        test_size = (max(600, physical_size[0] * 2 // 3),
                     max(360, physical_size[1] * 2 // 3))
    resized = None
    try:
        adb("shell", "wm", "size", f"{test_size[0]}x{test_size[1]}")
        resized = wait("runtime-info.json", lambda r:
                       r.get("surface_ready") and direct(r) and
                       r.get("surface_generation", 0) >
                       resumed.get("surface_generation", 0))
    finally:
        adb("shell", "wm", "size",
            prior_override.group(1) if prior_override else "reset")
    resize_restored = wait("runtime-info.json", lambda r:
                           r.get("surface_ready") and direct(r) and
                           resized is not None and
                           r.get("surface_generation", 0) >
                           resized.get("surface_generation", 0))

    rotation_state = adb("shell", "wm", "user-rotation").strip()
    window_state = adb("shell", "dumpsys", "window", "displays")
    displayed_rotation = re.search(r"^\s*mRotation=(\d+)\b",
                                   window_state, re.MULTILINE)
    if not displayed_rotation:
        raise RuntimeError("Could not determine the current Android display rotation")
    current_rotation = int(displayed_rotation.group(1))
    test_rotation = 3 if current_rotation == 1 else 1
    rotated = None
    try:
        adb("shell", "wm", "user-rotation", "lock", str(test_rotation))
        rotated = wait("runtime-info.json", lambda r:
                       r.get("surface_ready") and direct(r) and
                       r.get("surface_generation", 0) >
                       resize_restored.get("surface_generation", 0))
    finally:
        prior_lock = re.fullmatch(r"lock\s+(\d+)", rotation_state)
        if prior_lock:
            adb("shell", "wm", "user-rotation", "lock", prior_lock.group(1))
        else:
            adb("shell", "wm", "user-rotation", "free")
    rotation_restored = wait("runtime-info.json", lambda r:
                             r.get("surface_ready") and direct(r) and
                             rotated is not None and
                             r.get("frames", 0) > rotated.get("frames", 0))
    (args.output / "surface-lifecycle.json").write_text(json.dumps({
        "initial": resumed,
        "display_size_before": size_state,
        "test_size": f"{test_size[0]}x{test_size[1]}",
        "resized": resized,
        "resize_restored": resize_restored,
        "rotation_before": {
            "wm_user_rotation": rotation_state,
            "display_rotation": current_rotation,
        },
        "test_rotation": test_rotation,
        "rotated": rotated,
        "rotation_restored": rotation_restored,
    }, indent=2) + "\n")

    pids = adb("shell", "pidof", package).strip().split()
    if not pids:
        device_log = adb("logcat", "-d")
        (args.output / "device.log").write_text(device_log)
        raise RuntimeError("Android Gallery process exited before teardown; see device.log")
    pid = pids[0]
    adb("shell", "input", "keyevent", "KEYCODE_BACK")
    teardown_deadline = time.monotonic() + 30
    device_log = ""
    while time.monotonic() < teardown_deadline:
        device_log = adb("logcat", "--pid", pid, "-d")
        if "Gallery session destroyed" in device_log:
            break
        time.sleep(1)
    (args.output / "device.log").write_text(device_log)
    if "Gallery session destroyed" not in device_log:
        raise RuntimeError("Android Gallery did not complete native teardown; "
                           "see device.log")
    fatal_patterns = ("Validation Error", "VUID-", "FATAL EXCEPTION",
                      "Fatal signal", "SIGSEGV")
    failures = [line for line in device_log.splitlines()
                if any(pattern in line for pattern in fatal_patterns)]
    if failures:
        raise RuntimeError("Android lifecycle produced validation or process errors; "
                           "see device.log")
    print(f"PASS: {len(scenes)} catalog scenes ({len(supported)} supported), "
          "direct presentation, pause/reset, background, resize, rotation, and teardown",
          report["revision"], report["backend"])


if __name__ == "__main__":
    main()
