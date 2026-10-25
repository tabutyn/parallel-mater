#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run existing public-API Vulkan tests on an explicitly selected Android device."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/data/local/tmp/parallel-mater-qualification"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_value(cache, name):
    prefix = name + ":"
    for line in cache.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--build-directory", type=Path, required=True,
                        help="NDK CMake directory built with -Pqualification=true")
    parser.add_argument("--adb", default=str(Path(os.environ.get(
        "ANDROID_HOME", Path.home() / "Android/Sdk")) / "platform-tools/adb"))
    parser.add_argument("--output", type=Path, default=ROOT / "out/android/native-vulkan-tests")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--conformance-timeout", type=int, default=3600)
    parser.add_argument("--native-only", action="store_true",
                        help="Run required native adapter tests and capture provenance")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Stage and verify the device payload without using adb")
    args = parser.parse_args()
    if not args.prepare_only and not args.serial:
        parser.error("--serial is required unless --prepare-only is used")
    if not args.prepare_only and not Path(args.adb).is_file():
        raise SystemExit(f"ADB not found: {args.adb}; set ANDROID_HOME or pass --adb")
    build = args.build_directory.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    if "PARALLEL_MATER_ANDROID_QUALIFICATION:BOOL=ON" not in cache:
        raise SystemExit("Build with -Pqualification=true first")
    libraries = Path(next(line.split("=", 1)[1] for line in cache.splitlines()
                          if line.startswith("CMAKE_LIBRARY_OUTPUT_DIRECTORY:")))
    stage = args.output / "payload"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    tests = ("pm-android-command-tests", "pm-android-async-tests",
             "pm-android-liquid-tests", "pm-android-realtime-runtime-tests")
    tools = tests + ("pm-android-conformance", "pm-android-device-contracts")
    for name in tools:
        shutil.copy2(libraries / name, stage / name)
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
    corpus = stage / "conformance/v1"
    for name in ("cases", "lifecycle", "quality", "qualification"):
        shutil.copytree(ROOT / "conformance/v1" / name, corpus / name)
    shutil.copy2(ROOT / "conformance/v1/manifest.json", corpus / "manifest.json")
    (corpus / "evidence").mkdir()
    for name in ("evidence-sources-v1.json", "index-v1.json"):
        shutil.copy2(ROOT / "conformance/v1/evidence" / name,
                     corpus / "evidence" / name)
    manifest = json.loads((corpus / "manifest.json").read_text())
    for relative in manifest["glb_sha256"]:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (stage / "driver.json").write_text(json.dumps({"file_format_version": "1.0.0", "PMD": {
        "library_path": "./libparallel_mater_driver_slang_vulkan.so",
        "api_version": "0.2.0", "library_arch": "64"}}))
    (stage / "validation.json").write_text(json.dumps({"file_format_version": "1.0.0", "layer": {
        "name": "PM_LAYER_KHRONOS_validation", "library_path": "./libparallel_mater_layer_validation.so",
        "api_version": "0.2.0"}}))
    if args.prepare_only:
        print(f"Prepared Android Vulkan qualification payload: {stage}")
        return
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    source_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    if source_status:
        raise SystemExit("Physical qualification requires a clean source checkout")
    if cache_value(cache, "PARALLEL_MATER_SOURCE_REVISION") != source_revision:
        raise SystemExit("Qualification build is stale; rebuild it from the clean current commit")
    adb = [args.adb, "-s", args.serial]

    def prop(name):
        return subprocess.run(adb + ["shell", "getprop", name], check=True,
                              capture_output=True, text=True).stdout.strip()

    subprocess.run(adb + ["shell", "rm", "-rf", REMOTE], check=True)
    subprocess.run(adb + ["shell", "mkdir", "-p", REMOTE], check=True)
    subprocess.run(adb + ["push", str(stage) + "/.", REMOTE], check=True)
    environment = (f"LD_LIBRARY_PATH={REMOTE} PM_DRIVER_FILES={REMOTE}/driver.json "
                   f"PM_LAYER_FILES={REMOTE}/validation.json PM_VULKAN_CACHE_DIR={REMOTE}/cache "
                   "PM_VULKAN_TRACE=1")
    results = []

    def run_device(name, arguments=(), timeout=args.timeout, required=False):
        print(f"Running {name}…", flush=True)
        subprocess.run(adb + ["shell", "chmod", "755", f"{REMOTE}/{name}"], check=True)
        command = (f"cd {shlex.quote(REMOTE)} && {environment} " +
                   shlex.join([f"./{name}", *arguments]))
        try:
            result = subprocess.run(adb + ["shell", command], capture_output=True,
                                    text=True, timeout=timeout)
            stdout = result.stdout
            output = result.stdout + result.stderr
            passed = result.returncode == 0 and "Validation Error" not in output and "VUID-" not in output
            record = {"test": name, "arguments": list(arguments),
                      "passed": passed, "returncode": result.returncode}
        except subprocess.TimeoutExpired as error:
            # Stop only this test process, not unrelated phone work.
            subprocess.run(adb + ["shell", "pkill", "-f", f"^./{name}$"], check=False)
            output = f"Timed out after {timeout}s\n{error}"
            stdout = ""
            record = {"test": name, "arguments": list(arguments),
                      "passed": False, "timeout": timeout}
        results.append(record)
        suffix = ""
        for selector in ("--contract", "--case", "--quality",
                         "--lifecycle-profile"):
            if selector in arguments:
                suffix = "-" + arguments[arguments.index(selector) + 1]
                break
        if not suffix and arguments:
            suffix = "-" + arguments[0].removeprefix("--")
        (args.output / f"{name}{suffix}.log").write_text(output)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {'PASS' if results[-1]['passed'] else 'FAIL'}", flush=True)
        if required and not results[-1]["passed"]:
            print(output[-6000:])
            raise SystemExit(1)
        return record, stdout

    for name in tests:
        arguments = ("--production",) \
            if name == "pm-android-realtime-runtime-tests" else ()
        run_device(name, arguments, required=True)

    if args.native_only:
        apk = ROOT / "examples/android/app/build/outputs/apk/debug/app-debug.apk"
        capture = {
            "schema": "parallel-mater-android-native-capture/v1",
            "source_commit": source_revision,
            "apk_sha256": sha256(apk),
            "driver_sha256": sha256(
                stage / "libparallel_mater_driver_slang_vulkan.so"),
            "device": {
                "manufacturer": prop("ro.product.manufacturer"),
                "model": prop("ro.product.model"),
                "product": prop("ro.product.name"),
                "android_version": prop("ro.build.version.release"),
                "api_level": prop("ro.build.version.sdk"),
                "security_patch": prop("ro.build.version.security_patch"),
            },
            "toolchain": {
                "ndk": cache_value(cache, "CMAKE_ANDROID_NDK"),
                "slang": cache_value(cache, "PARALLEL_MATER_SLANG_VERSION"),
            },
            "native_runs": results,
        }
        (args.output / "capture.json").write_text(
            json.dumps(capture, indent=2) + "\n")
        print(f"Capture written to {args.output / 'capture.json'}", flush=True)
        return

    remote_results = f"{REMOTE}/results"
    subprocess.run(adb + ["shell", "mkdir", "-p", f"{remote_results}/cases",
                           f"{remote_results}/quality", f"{remote_results}/lifecycle"],
                   check=True)
    run_device("pm-android-conformance", ("--check-inputs", "--root", REMOTE),
               required=True)
    for mode, filename in (("--provenance", "provenance.json"),
                           ("--capabilities", "capabilities.json")):
        _, output = run_device("pm-android-conformance", (mode, "--root", REMOTE),
                               required=True)
        (args.output / filename).write_text(output.strip() + "\n")
    run_device("pm-android-conformance",
               ("--case", "all", "--root", REMOTE, "--output",
                f"{remote_results}/cases"),
               timeout=args.conformance_timeout)
    quality_profiles = (
        "rigid-brick-penetration-v1",
        "rigid-attachment-v1",
        "rope-self-contact-v1",
        "fluid-self-contact-v1",
        "fluid-self-pressure-v1",
        "fluid-self-viscosity-v1",
        "fluid-cloth-containment-v1",
        "fluid-soft-contact-v1",
        "fluid-rope-contact-v1",
        "smoke-self-pressure-v1",
        "fluid-cloth-contact-v1",
        "smoke-self-viscosity-v1",
    )
    for profile in quality_profiles:
        run_device("pm-android-conformance",
                   ("--quality", profile, "--root", REMOTE, "--output",
                    f"{remote_results}/quality"),
                   timeout=args.conformance_timeout)
    lifecycle_profiles = (
        "adverse-capacity-three-family-v1",
        "fluid-cloth-containment-lifecycle-v1",
        "smoke-self-pressure-lifecycle-v1",
        "fluid-cloth-contact-lifecycle-v1",
        "smoke-self-lifecycle-v1",
    )
    for profile in lifecycle_profiles:
        run_device("pm-android-conformance",
                   ("--lifecycle-profile", profile, "--root", REMOTE,
                    "--output", f"{remote_results}/lifecycle/{profile}.json"),
                   timeout=args.conformance_timeout)
    run_device("pm-android-device-contracts",
               ("--contract", "vulkan-info", "--root", REMOTE, "--output",
                f"{remote_results}/vulkan.json"), required=True)

    artifacts = args.output / "physics-qualification"
    if artifacts.exists():
        shutil.rmtree(artifacts)
    subprocess.run(adb + ["pull", f"{remote_results}/.", str(artifacts)], check=True)
    shutil.copy2(args.output / "provenance.json", artifacts / "provenance.json")
    shutil.copy2(args.output / "capabilities.json", artifacts / "capabilities.json")

    reports = args.output / "reports"
    quality_reports = reports / "quality"
    lifecycle_reports = reports / "lifecycle"
    quality_reports.mkdir(parents=True, exist_ok=True)
    lifecycle_reports.mkdir(parents=True, exist_ok=True)
    evaluations = [
        ("reference", "all-cases", reports / "reference.json",
         [str(ROOT / "conformance/v1/compare.py"),
          "--cases", str(ROOT / "conformance/v1/cases"), "--expected",
          str(ROOT / "conformance/v1/golden/cuda"), "--actual",
          str(artifacts / "cases"), "--report", str(reports / "reference.json")]),
    ]
    for profile in quality_profiles:
        report = quality_reports / f"{profile}.json"
        evaluations.append((
            "quality", profile, report,
            [str(ROOT / "conformance/v1/quality.py"),
             "--profile", str(ROOT / f"conformance/v1/quality/{profile}.json"),
             "--actual", str(artifacts / f"quality/{profile}.json"),
             "--report", str(report)]))
    for profile in lifecycle_profiles:
        report = lifecycle_reports / f"{profile}.json"
        evaluations.append((
            "lifecycle", profile, report,
            [str(ROOT / "conformance/v1/lifecycle.py"),
             "--profile", str(ROOT / f"conformance/v1/lifecycle/{profile}.json"),
             "--actual", str(artifacts / f"lifecycle/{profile}.json"),
             "--report", str(report)]))
    evaluation_results = []
    for kind, profile, report, command in evaluations:
        result = subprocess.run([shutil.which("python3") or "python3", *command],
                                capture_output=True, text=True)
        report.with_suffix(".log").write_text(result.stdout + result.stderr)
        evaluation_results.append({
            "kind": kind, "profile": profile,
            "report": str(report.relative_to(args.output)),
            "returncode": result.returncode, "passed": result.returncode == 0})

    apk = ROOT / "examples/android/app/build/outputs/apk/debug/app-debug.apk"
    provenance = json.loads((args.output / "provenance.json").read_text())
    vulkan = json.loads((artifacts / "vulkan.json").read_text())
    capture = {
        "schema": "parallel-mater-android-vulkan-qualification-capture/v1",
        "source_commit": provenance["source_commit"],
        "apk_sha256": sha256(apk),
        "driver_sha256": sha256(stage / "libparallel_mater_driver_slang_vulkan.so"),
        "device": {
            "manufacturer": prop("ro.product.manufacturer"),
            "model": prop("ro.product.model"),
            "product": prop("ro.product.name"),
            "android_version": prop("ro.build.version.release"),
            "api_level": prop("ro.build.version.sdk"),
            "security_patch": prop("ro.build.version.security_patch"),
            "gpu": provenance["device_name"],
            "vendor_id": provenance["vendor_id"],
            "device_id": provenance["device_id"],
            "vulkan_driver_version": provenance["driver_version"],
            "vulkan_api_version": vulkan["api_version"],
            "vulkan_instance_api_version": vulkan["instance_api_version"],
            "vulkan_driver_id": vulkan["driver_id"],
            "vulkan_driver_name": vulkan["driver_name"],
            "vulkan_driver_info": vulkan["driver_info"],
        },
        "toolchain": {
            "compiler": provenance["host_compiler"],
            "ndk": cache_value(cache, "CMAKE_ANDROID_NDK"),
            "slang": cache_value(cache, "PARALLEL_MATER_SLANG_VERSION"),
        },
        "contracts": {relative: digest for relative, digest in {
            **manifest.get("lifecycle_profile_sha256", {}),
            **manifest.get("quality_profile_sha256", {}),
            **manifest.get("qualification_profile_sha256", {}),
        }.items()},
        "native_runs": results,
        "evaluations": evaluation_results,
    }
    (args.output / "capture.json").write_text(json.dumps(capture, indent=2) + "\n")
    print(f"Capture written to {args.output / 'capture.json'}", flush=True)
    failed_runs = [record for record in results if not record["passed"]]
    if failed_runs:
        print(f"{len(failed_runs)} device execution(s) failed; capture retained",
              flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
