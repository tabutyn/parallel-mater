#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install SDK components independently and consume every supported combination."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


KNOWN_COMPONENTS = (
    "loader",
    "validation",
    "preparation",
    "scene",
    "driver",
    "conformance",
    "benchmark",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New evidence directory (never overwritten)")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--expect-driver-target", action="append", default=[],
                        help="Imported driver target required when the driver component is present")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build_dir = args.build_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "parallel-mater-sdk-components/v1",
        "status": "failed",
        "scope": "relocated_component_exports",
        "host": platform.platform(),
        "configuration": args.config,
        "expected_driver_targets": args.expect_driver_target,
        "checks": [],
    }
    environment = os.environ.copy()
    for key in ("PM_DRIVER_PATH", "PM_DRIVER_FILES", "PM_LAYER_PATH", "PM_LAYER_FILES"):
        environment.pop(key, None)

    def run(name, command, expected=0, env=None):
        log = output / f"{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                [str(part) for part in command],
                cwd=root,
                env=env or environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        passed = process.returncode == expected
        report["checks"].append({
            "name": name,
            "status": "passed" if passed else "failed",
            "exit_code": process.returncode,
            "expected": expected,
        })
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
        if not passed:
            raise RuntimeError(f"{name}: see {log}")

    source = output / "external-source"
    shutil.copytree(root / "tests/sdk/components", source)
    shutil.copy2(root / "examples/sdk/devices.c", source / "devices.c")
    shutil.copy2(root / "tests/sdk/scene_consumer.cpp", source / "scene_consumer.cpp")

    install_script = (build_dir / "cmake_install.cmake").read_text(encoding="utf-8")
    available = [
        component for component in KNOWN_COMPONENTS
        if f'CMAKE_INSTALL_COMPONENT STREQUAL "{component}"' in install_script
    ]
    report["available_components"] = available

    def install(prefix, components, name):
        for component in components:
            run(
                f"{name}-install-{component}",
                ["cmake", "--install", build_dir, "--config", args.config,
                 "--prefix", prefix, "--component", component],
            )

    def configure(prefix, components, name, expected=0):
        build = output / name / "build"
        options = []
        if "driver" in components and args.expect_driver_target:
            options.append(
                f"-DPM_EXPECT_DRIVER_TARGETS={';'.join(args.expect_driver_target)}")
        run(
            f"{name}-configure",
            ["cmake", "-S", source, "-B", build, "-G", "Ninja",
             f"-DCMAKE_BUILD_TYPE={args.config}", f"-DCMAKE_PREFIX_PATH={prefix}",
             f"-DPM_COMPONENTS={';'.join(components)}", *options],
            expected=expected,
        )
        if expected == 0:
            run(
                f"{name}-build",
                ["cmake", "--build", build, "--config", args.config,
                 "--parallel", "1"],
            )

    try:
        report["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        report["source_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip())
        if "loader" not in available or "validation" not in available:
            raise RuntimeError("every SDK build must provide loader and validation components")

        supported = [
            ("loader", ["loader"]),
            ("loader-validation", ["loader", "validation"]),
        ]
        if "preparation" in available:
            supported.append(("loader-preparation", ["loader", "preparation"]))
        if "scene" in available:
            if "preparation" not in available:
                raise RuntimeError("scene is available without preparation")
            supported.append(("loader-scene", ["loader", "preparation", "scene"]))
        for component in ("driver", "conformance", "benchmark"):
            if component in available:
                supported.append((f"loader-{component}", ["loader", component]))
        supported.append(("all", available))

        for name, components in supported:
            prefix = output / name / "sdk"
            install(prefix, components, name)
            configure(prefix, components, name)
            if "conformance" in components:
                suffix = ".exe" if os.name == "nt" else ""
                selected = dict(environment)
                selected["PATH"] = str(prefix / "bin") + os.pathsep + selected.get("PATH", "")
                run(
                    f"{name}-conformance-inputs",
                    [prefix / "bin" / ("parallel-mater-conformance" + suffix),
                     "--check-inputs", "--root", prefix / "share/parallel-mater"],
                    env=selected,
                )

        loader_prefix = output / "negative-loader-only" / "sdk"
        install(loader_prefix, ["loader"], "negative-loader-only")
        for component in KNOWN_COMPONENTS[1:]:
            configure(
                loader_prefix,
                ["loader", component],
                f"reject-missing-{component}",
                expected=1,
            )
        if "scene" in available:
            dangling_prefix = output / "negative-scene-without-preparation" / "sdk"
            install(dangling_prefix, ["loader", "scene"],
                    "negative-scene-without-preparation")
            configure(
                dangling_prefix,
                ["loader", "scene"],
                "reject-scene-without-preparation",
                expected=1,
            )
        report["status"] = "passed"
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        report["error"] = str(error)
        print(error, file=sys.stderr)
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
