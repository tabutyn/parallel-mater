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
    if len(compiler_files) != 1:
        raise RuntimeError(
            f"expected one CMake C++ compiler record, found {len(compiler_files)}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New evidence directory (never overwritten)")
    parser.add_argument("--config", default="Release")
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

    try:
        report["architecture"] = platform.machine()
        report["compiler"] = compiler_metadata(args.build_dir.resolve())
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
        run("consumer-configure", ["cmake", "-S", source, "-B", build, "-G", "Ninja",
            f"-DCMAKE_BUILD_TYPE={args.config}", f"-DCMAKE_PREFIX_PATH={sdk}"])
        run("consumer-build", ["cmake", "--build", build, "--parallel", "1"])
        run("consumer-tests", ["ctest", "--test-dir", build, "--output-on-failure",
                                "--no-tests=error", "-j1"])
        for version in ("0.1", "0.3"):
            run("reject-version-" + version, ["cmake", "-S", source, "-B",
                output / ("version-" + version), "-G", "Ninja",
                f"-DCMAKE_PREFIX_PATH={sdk}", f"-DPM_REQUIRED_VERSION={version}"], expected=1)
        # Only conformance's own sources/headers; no repository src/ or examples/.
        suite = output / "conformance-source"
        suite.mkdir()
        for name in ("src", "include"):
            shutil.copytree(root / "conformance/v1" / name, suite / name)
        shutil.copy2(root / "conformance/v1/CMakeLists.txt", suite / "CMakeLists.txt")
        data = sdk / "share/parallel-mater"
        suite_build = output / "conformance-build"
        run("conformance-configure", ["cmake", "-S", suite, "-B", suite_build, "-G", "Ninja",
            f"-DCMAKE_PREFIX_PATH={sdk}", f"-DPARALLEL_MATER_CONFORMANCE_DATA_ROOT={data}",
            f"-DPARALLEL_MATER_CONFORMANCE_REVISION={report['source_revision']}"])
        run("conformance-build", ["cmake", "--build", suite_build, "--parallel", "1"])
        run("conformance-inputs", ["ctest", "--test-dir", suite_build, "--output-on-failure",
                                  "--no-tests=error"])
        suffix = ".exe" if os.name == "nt" else ""
        run("installed-runner-inputs", [sdk / "bin" / ("parallel-mater-conformance" + suffix),
                                        "--check-inputs", "--root", data])
        runner = suite_build / ("parallel-mater-conformance" + suffix)
        selected = dict(environment, PM_DRIVER_FILES=str(build / "mock.json"))
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
