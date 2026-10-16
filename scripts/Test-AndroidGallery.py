#!/usr/bin/env python3
"""Exercise installed debug Gallery scenes and actual Android lifecycle callbacks."""
import argparse
import json
import pathlib
import subprocess
import time
import uuid


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

    activity = package + "/.GalleryActivity"
    adb("shell", "am", "force-stop", package)
    adb("shell", "am", "start", "-n", activity, "--ei", "pm_qualify_frames", str(args.frames), "--es", "pm_nonce", nonce)
    report = wait("qualification.json", lambda r: r.get("completed"), args.timeout)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    if len(report["scenes"]) != 8 or not all(s["passed"] for s in report["scenes"]):
        raise RuntimeError("Scene checks failed; see qualification.json")
    before = wait("runtime-info.json", lambda r: r["active"])
    adb("shell", "input", "keyevent", "KEYCODE_HOME")
    background = wait("runtime-info.json", lambda r: not r["active"] and r["backgrounds"] > before["backgrounds"])
    time.sleep(5)
    adb("shell", "am", "start", "-f", "0x20000000", "-n", activity)
    foreground = wait("runtime-info.json", lambda r: r["foregrounds"] > background["foregrounds"])
    if foreground["frames"] != background["frames"]:
        raise RuntimeError("Simulation advanced while backgrounded")
    (args.output / "lifecycle.json").write_text(json.dumps({"background": background, "foreground": foreground}, indent=2) + "\n")
    print("PASS: 8 scenes, shared pause/reset checks, and OS background recovery", report["revision"], report["backend"])


if __name__ == "__main__":
    main()
