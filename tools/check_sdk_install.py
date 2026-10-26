#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install, relocate and consume an SDK without exposing private source paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def compiler_metadata(build_dir):
    """Read the compiler identity CMake used for the SDK under test."""
    compiler_files = sorted(
        (build_dir / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    cache_values = {}
    for line in (build_dir / "CMakeCache.txt").read_text(
            encoding="utf-8").splitlines():
        match = re.match(
            r"^CMAKE_CACHE_(MAJOR|MINOR|PATCH)_VERSION:INTERNAL=(.*)$",
            line,
        )
        if match:
            cache_values[match.group(1)] = match.group(2)
    if all(part in cache_values for part in ("MAJOR", "MINOR", "PATCH")):
        version = ".".join(cache_values[part]
                           for part in ("MAJOR", "MINOR", "PATCH"))
        current = build_dir / "CMakeFiles" / version / "CMakeCXXCompiler.cmake"
        if current.is_file():
            compiler_files = [current]
    if len(compiler_files) != 1:
        raise RuntimeError(
            "could not identify the active CMake C++ compiler record; found "
            f"{len(compiler_files)}")
    values = {}
    pattern = re.compile(r'^set\((CMAKE_CXX_COMPILER(?:_[A-Z_]+)?) "(.*)"\)$')
    for line in compiler_files[0].read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    required = ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID",
                "CMAKE_CXX_COMPILER_VERSION")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise RuntimeError("missing CMake compiler metadata: " + ", ".join(missing))
    result = {
        "path": values["CMAKE_CXX_COMPILER"],
        "id": values["CMAKE_CXX_COMPILER_ID"],
        "version": values["CMAKE_CXX_COMPILER_VERSION"],
        "architecture": (values.get("CMAKE_CXX_COMPILER_ARCHITECTURE_ID")
                         or platform.machine()),
    }
    if values.get("CMAKE_CXX_COMPILER_TARGET"):
        result["target"] = values["CMAKE_CXX_COMPILER_TARGET"]
    return result


def generator_arguments(build_dir):
    """Reuse the tested build generator when Ninja is unavailable."""
    if shutil.which("ninja"):
        return ["-G", "Ninja"]
    values = {}
    for line in (build_dir / "CMakeCache.txt").read_text(
            encoding="utf-8").splitlines():
        match = re.match(r"^(CMAKE_GENERATOR(?:_PLATFORM)?):INTERNAL=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2)
    generator = values.get("CMAKE_GENERATOR")
    if not generator:
        raise RuntimeError("could not identify a CMake generator")
    arguments = ["-G", generator]
    if values.get("CMAKE_GENERATOR_PLATFORM"):
        arguments += ["-A", values["CMAKE_GENERATOR_PLATFORM"]]
    return arguments


def windows_sdk_metadata(build_dir):
    """Read the Windows SDK selected by CMake for this build."""
    if os.name != "nt":
        return None
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"missing CMake cache: {cache}")
    resource_compiler = ""
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith("CMAKE_RC_COMPILER:"):
            resource_compiler = line.partition("=")[2]
            break
    match = re.search(
        r"[\\/]Windows Kits[\\/]10[\\/]bin[\\/]([^\\/]+)[\\/]",
        resource_compiler,
        re.IGNORECASE,
    )
    if match:
        version = match.group(1)
    else:
        # Visual Studio generators can cache the compiler as bare `rc` even
        # though every generated project records the selected SDK version.
        project = build_dir / "ZERO_CHECK.vcxproj"
        project_text = (project.read_text(encoding="utf-8")
                        if project.is_file() else "")
        project_match = re.search(
            r"<WindowsTargetPlatformVersion>([^<]+)</WindowsTargetPlatformVersion>",
            project_text,
        )
        if not project_match:
            raise RuntimeError(
                "could not identify the Windows SDK from the compiler or project")
        version = project_match.group(1)
    return {"version": version, "resource_compiler": resource_compiler}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New evidence directory (never overwritten)")
    parser.add_argument("--config", default="Release")
    parser.add_argument(
        "--require-realtime-driver", action="store_true",
        help="Fail unless the installed Slang CPU production driver is available",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original, sdk = output / "original-sdk", output / "relocated-sdk"
    report = {"schema": "parallel-mater-sdk-evidence/v1",
              "scope": "packaging_and_public_abi", "status": "failed",
              "physics_quality": "not_evaluated", "host": platform.platform(),
              "checks": []}
    environment = os.environ.copy()
    # Test exactly this SDK and this independent mock, not workstation installs.
    for key in ("PM_DRIVER_PATH", "PM_DRIVER_FILES", "PM_LAYER_PATH", "PM_LAYER_FILES"):
        environment.pop(key, None)
    environment["PM_LAYER_FILES"] = str(sdk / "share/parallel-mater/layers.d/parallel_mater_validation.json")
    environment["PATH"] = str(sdk / "bin") + os.pathsep + environment.get("PATH", "")
    generator = generator_arguments(args.build_dir.resolve())

    def run(name, command, expected=0, env=None):
        log = output / (name + ".log")
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run([str(part) for part in command], cwd=root,
                                     env=env or environment, stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
        passed = process.returncode == expected
        report["checks"].append({"name": name, "status": "passed" if passed else "failed",
                                 "exit_code": process.returncode, "expected": expected})
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
        if not passed:
            raise RuntimeError(f"{name}: see {log}")

    def configured_output(build, name):
        direct = build / name
        if direct.exists():
            return direct
        return build / args.config / name

    try:
        report["architecture"] = platform.machine()
        report["configuration"] = args.config
        report["compiler"] = compiler_metadata(args.build_dir.resolve())
        windows_sdk = windows_sdk_metadata(args.build_dir.resolve())
        if windows_sdk:
            report["windows_sdk"] = windows_sdk
        report["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        report["source_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip())
        run("install", ["cmake", "--install", args.build_dir.resolve(),
                        "--config", args.config, "--prefix", original])
        original.rename(sdk)
        for path in sdk.rglob("*.cmake"):
            content = path.read_text(encoding="utf-8")
            for forbidden in (str(root), root.as_posix(), str(original), original.as_posix()):
                if forbidden in content:
                    raise RuntimeError(f"non-relocatable path in {path}")
        source = output / "external-source"
        source.mkdir()
        for path in (root / "tests/sdk").iterdir():
            if path.is_file():
                shutil.copy2(path, source / path.name)
        for path in (root / "tests/api/mock_driver.cpp",
                     root / "tests/api/loader_driver_discovery_tests.cpp",
                     root / "examples/sdk/devices.c"):
            shutil.copy2(path, source / path.name)
        build = output / "external-build"
        observations = output / "realtime-observations"
        observations.mkdir()
        candidate_path = (
            root / "conformance/v1/qualification/realtime-candidate-v3.json")
        candidate_bytes = candidate_path.read_bytes()
        candidate = json.loads(candidate_bytes)
        installed_cpu_manifest = (
            sdk / "share/parallel-mater/drivers.d/parallel_mater_slang_cpu.json")
        realtime_available = installed_cpu_manifest.is_file()
        if args.require_realtime_driver and not realtime_available:
            raise RuntimeError(
                "installed Slang CPU production driver manifest is missing")
        report["realtime_profile"] = {
            "status": "tested" if realtime_available else "not_available",
            "required": args.require_realtime_driver,
        }
        run("consumer-configure", ["cmake", "-S", source, "-B", build,
            *generator, f"-DCMAKE_BUILD_TYPE={args.config}",
            f"-DCMAKE_PREFIX_PATH={sdk}",
            "-DPM_ENABLE_REALTIME_PROFILE_TESTS=" +
                ("ON" if realtime_available else "OFF"),
            f"-DPM_REALTIME_DRIVER_MANIFEST={installed_cpu_manifest}",
            f"-DPM_REALTIME_OBSERVATION_DIR={observations}",
            f"-DPM_REALTIME_CANDIDATE_ID={candidate['id']}",
            "-DPM_REALTIME_CANDIDATE_SHA256=" +
                hashlib.sha256(candidate_bytes).hexdigest(),
            "-DPM_REALTIME_CANDIDATE_REVISION=" +
                candidate["source"]["revision"]])
        run("consumer-build", ["cmake", "--build", build, "--config", args.config,
                               "--parallel", "1"])
        run("consumer-tests", ["ctest", "--test-dir", build, "--output-on-failure",
                                "--no-tests=error", "-C", args.config, "-j1"])
        if realtime_available:
            for mode in ("plain", "validation"):
                run("realtime-observation-" + mode,
                    [sys.executable, root / "conformance/v1/realtime_oracle.py",
                     "--validate-observation", observations / (mode + ".json"),
                     "--candidate-manifest", candidate_path])
            report["realtime_observations"] = {
                mode: hashlib.sha256(
                    (observations / (mode + ".json")).read_bytes()).hexdigest()
                for mode in ("plain", "validation")
            }
            capacity_path = observations / "capacity.json"
            report["realtime_capacity_regression"] = {
                "status": "passed",
                "sha256": hashlib.sha256(capacity_path.read_bytes()).hexdigest(),
                "workload": json.loads(capacity_path.read_text(encoding="utf-8"))[
                    "workload"],
            }
        for version in ("0.2", "0.4"):
            run("reject-version-" + version, ["cmake", "-S", source, "-B",
                output / ("version-" + version), *generator,
                f"-DCMAKE_PREFIX_PATH={sdk}", f"-DPM_REQUIRED_VERSION={version}"], expected=1)
        # Only conformance's own sources/headers; no repository src/ or examples/.
        suite = output / "conformance-source"
        suite.mkdir()
        for name in ("src", "include"):
            shutil.copytree(root / "conformance/v1" / name, suite / name)
        shutil.copy2(root / "conformance/v1/CMakeLists.txt", suite / "CMakeLists.txt")
        data = sdk / "share/parallel-mater"
        suite_build = output / "conformance-build"
        run("conformance-configure", ["cmake", "-S", suite, "-B", suite_build,
            *generator,
            f"-DCMAKE_BUILD_TYPE={args.config}", f"-DCMAKE_PREFIX_PATH={sdk}",
            f"-DPARALLEL_MATER_CONFORMANCE_DATA_ROOT={data}",
            f"-DPARALLEL_MATER_CONFORMANCE_REVISION={report['source_revision']}"])
        run("conformance-build", ["cmake", "--build", suite_build, "--config", args.config,
                                  "--parallel", "1"])
        run("conformance-inputs", ["ctest", "--test-dir", suite_build, "--output-on-failure",
                                  "--no-tests=error", "-C", args.config])
        suffix = ".exe" if os.name == "nt" else ""
        run("installed-runner-inputs", [sdk / "bin" / ("parallel-mater-conformance" + suffix),
                                        "--check-inputs", "--root", data])
        runner = configured_output(
            suite_build, "parallel-mater-conformance" + suffix)
        selected = dict(
            environment,
            PM_DRIVER_FILES=str(configured_output(build, "mock.json")),
        )
        run("unsupported-case", [runner, "--case", "cloth-core", "--output",
                                 output / "unsupported"], expected=77, env=selected)
        result = json.loads((output / "unsupported/cloth-core.json").read_text())
        if result["execution_status"] != "unsupported" or result["quality_status"] != "not_evaluated":
            raise RuntimeError("unsupported case misclassified")
        missing = dict(environment, PM_DRIVER_FILES=str(output / "missing-driver.json"))
        run("missing-driver", [runner, "--case", "rigid-direct", "--output",
                               output / "missing"], expected=2, env=missing)
        result = json.loads((output / "missing/rigid-direct.json").read_text())
        if result["execution_status"] != "failed":
            raise RuntimeError("missing driver misclassified")
        report["status"] = "passed"
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        report["error"] = str(error)
        print(error, file=sys.stderr)
    finally:
        report["sdk_sha256"] = {
            str(path.relative_to(sdk)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sdk.rglob("*")) if path.is_file()}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
