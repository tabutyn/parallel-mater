#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run existing public-API Vulkan tests on an explicitly selected Android device."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/data/local/tmp/parallel-mater-qualification"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--build-directory", type=Path, required=True,
                        help="NDK CMake directory built with -Pqualification=true")
    parser.add_argument("--adb", default=str(Path(os.environ.get(
        "ANDROID_HOME", Path.home() / "Android/Sdk")) / "platform-tools/adb"))
    parser.add_argument("--output", type=Path, default=ROOT / "out/android/native-vulkan-tests")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not Path(args.adb).is_file():
        raise SystemExit(f"ADB not found: {args.adb}; set ANDROID_HOME or pass --adb")
    build = args.build_directory.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    if "PARALLEL_MATER_ANDROID_QUALIFICATION:BOOL=ON" not in cache:
        raise SystemExit("Build with -Pqualification=true first")
    libraries = next(line.split("=", 1)[1] for line in cache.splitlines()
                     if line.startswith("CMAKE_LIBRARY_OUTPUT_DIRECTORY:"))
    stage = args.output / "payload"
    stage.mkdir(parents=True, exist_ok=True)
    tests = ("pm-android-command-tests", "pm-android-async-tests", "pm-android-liquid-tests")
    for name in tests:
        shutil.copy2(Path(libraries) / name, stage / name)
    for name in ("libparallel-mater.so", "libparallel_mater_driver_slang_vulkan.so",
                 "libparallel_mater_layer_validation.so"):
        shutil.copy2(Path(libraries) / name, stage / name)
    # Use the exact C++ runtime bundled with the just-built APK.
    apk = ROOT / "examples/android/app/build/outputs/apk/debug/app-debug.apk"
    with zipfile.ZipFile(apk) as archive:
        (stage / "libc++_shared.so").write_bytes(archive.read("lib/arm64-v8a/libc++_shared.so"))
    scenes = stage / "scenes"
    scenes.mkdir(exist_ok=True)
    for name in ("Fluid.glb", "FluidRigid.glb"):
        shutil.copy2(ROOT / "examples/assets" / name, scenes / name)
    shutil.copy2(ROOT / "config/device-profiles.json", stage / "device-profiles.json")
    (stage / "driver.json").write_text(json.dumps({"file_format_version": "1.0.0", "PMD": {
        "library_path": "./libparallel_mater_driver_slang_vulkan.so",
        "api_version": "0.2.0", "library_arch": "64"}}))
    (stage / "validation.json").write_text(json.dumps({"file_format_version": "1.0.0", "layer": {
        "name": "PM_LAYER_KHRONOS_validation", "library_path": "./libparallel_mater_layer_validation.so",
        "api_version": "0.2.0"}}))
    adb = [args.adb, "-s", args.serial]
    subprocess.run(adb + ["shell", "mkdir", "-p", REMOTE], check=True)
    subprocess.run(adb + ["push", str(stage) + "/.", REMOTE], check=True)
    environment = (f"LD_LIBRARY_PATH={REMOTE} PM_DRIVER_FILES={REMOTE}/driver.json "
                   f"PM_LAYER_FILES={REMOTE}/validation.json PM_VULKAN_CACHE_DIR={REMOTE}/cache")
    results = []
    for name in tests:
        print(f"Running {name}…", flush=True)
        subprocess.run(adb + ["shell", "chmod", "755", f"{REMOTE}/{name}"], check=True)
        command = f"cd {shlex.quote(REMOTE)} && {environment} ./{name}"
        try:
            result = subprocess.run(adb + ["shell", command], capture_output=True,
                                    text=True, timeout=args.timeout)
            output = result.stdout + result.stderr
            passed = result.returncode == 0 and "Validation Error" not in output and "VUID-" not in output
            results.append({"test": name, "passed": passed, "returncode": result.returncode})
        except subprocess.TimeoutExpired as error:
            # Stop only this test process, not unrelated phone work.
            subprocess.run(adb + ["shell", "pkill", "-f", f"^./{name}$"], check=False)
            output = f"Timed out after {args.timeout}s\n{error}"
            results.append({"test": name, "passed": False, "timeout": args.timeout})
        (args.output / f"{name}.log").write_text(output)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {'PASS' if results[-1]['passed'] else 'FAIL'}", flush=True)
        if not results[-1]["passed"]:
            print(output[-6000:])
            raise SystemExit(1)


if __name__ == "__main__":
    main()
