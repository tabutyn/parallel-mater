#!/usr/bin/env python3
"""Deploy an already-built preview; verify the installed revision and backend."""
import argparse
import hashlib
import json
import os
import pathlib
import plistlib
import subprocess
import sys
import tempfile
import time
import uuid

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mobile_gallery_report import validate_mobile_gallery_report  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = "com.ballrollergames.parallelmater.gallery"
EXECUTION_CONTRACT = "fixed-tick-60hz-v1"
MACOS_PROFILE_SUITE = (
    {"id": "rigid-386", "workload": "rigid-386",
     "arguments": ("--brick-count", "386")},
    {"id": "fluid", "workload": "gallery-fluid",
     "arguments": ("--fluid",)},
    {"id": "cloth", "workload": "gallery-cloth",
     "arguments": ("--cloth",)},
    {"id": "soft-body", "workload": "gallery-soft-body",
     "arguments": ("--soft-body",)},
    {"id": "soft-body-fluid", "workload": "gallery-soft-body-fluid",
     "arguments": ("--soft-body-fluid",)},
    {"id": "smoke-soft-body", "workload": "gallery-smoke-soft-body",
     "arguments": ("--smoke-softbody",)},
)


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, text=True, **kwargs)


def macos_thermal_state():
    measured_at = time.time()
    result = subprocess.run(["pmset", "-g", "therm"], text=True,
                            capture_output=True, check=False)
    return {
        "measured_at_unix": measured_at,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def build_info(app, platform, allow_dirty=False):
    resources = app / "Contents/Resources" if platform == "macos" else app
    info = json.loads((resources / "build-info.json").read_text())
    revision = run("git", "rev-parse", "HEAD", cwd=ROOT, capture_output=True).stdout.strip()
    dirty = run("git", "status", "--porcelain", cwd=ROOT, capture_output=True).stdout.strip()
    if info["revision"] != revision:
        raise RuntimeError("Built revision differs from HEAD; rebuild before deployment")
    if not allow_dirty and (info["dirty"] or dirty):
        raise RuntimeError("Release deployment requires a clean commit and clean rebuild (--allow-dirty is preview-only)")
    if info.get("schema_version") != 2 or \
            info.get("execution_contract") != EXECUTION_CONTRACT:
        raise RuntimeError("Expected the sole fixed-tick Gallery contract")
    if info["backend"] != "metal" or info["release_channel"] != "development-preview":
        raise RuntimeError("Expected explicitly labeled Metal development preview")
    plist_path = app / "Contents/Info.plist" if platform == "macos" else app / "Info.plist"
    with plist_path.open("rb") as stream:
        executable = plistlib.load(stream)["CFBundleExecutable"]
    executable_path = (app / "Contents/MacOS" / executable
                       if platform == "macos" else app / executable)
    if info["revision"].encode("ascii") not in executable_path.read_bytes():
        raise RuntimeError(
            "Compiled Gallery revision differs from build-info.json; rebuild "
            "before deployment")
    run("codesign", "--verify", "--deep", "--strict", app)
    if platform == "ipad" and not (app / "Assets.car").is_file():
        raise RuntimeError("iPad app icon asset catalog is missing")
    return info


def copy_report(device, name, directory):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    run("xcrun", "devicectl", "device", "copy", "from", "--device", device,
        "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE,
        "--source", "Documents/" + name, "--destination", path,
        "--timeout", "15",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(path.read_text())


def copy_artifact(device, name, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    run("xcrun", "devicectl", "device", "copy", "from", "--device", device,
        "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE,
        "--source", "Documents/" + name, "--destination", destination,
        "--timeout", "30", stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Collected device artifact is empty: " + name)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ipad_binary_identity(app):
    with (app / "Info.plist").open("rb") as stream:
        executable = plistlib.load(stream)["CFBundleExecutable"]
    return {
        "app_executable_sha256": sha256_file(app / executable),
        "driver_sha256": sha256_file(
            app / "Frameworks/ParallelMaterMetal.framework/ParallelMaterMetal"),
    }


def validate_physics_qualification(report, nonce, revision, binary_identity):
    if not isinstance(report, dict):
        raise RuntimeError("Physics qualification report must be an object")
    if report.get("schema") != "parallel-mater-ipados-qualification-run/v1":
        raise RuntimeError("Physics qualification schema is missing or unsupported")
    if report.get("report_kind") != "public-api-conformance" or report.get("gallery_scene_sweep") is not False:
        raise RuntimeError("Gallery-only reports cannot satisfy physics qualification")
    if report.get("nonce") != nonce:
        raise RuntimeError("Physics qualification report is stale")
    if report.get("revision") != revision:
        raise RuntimeError("Physics qualification revision differs from the installed app")
    if report.get("execution_contract") != EXECUTION_CONTRACT:
        raise RuntimeError("Physics qualification used the wrong execution contract")
    if report.get("completed") is not True:
        raise RuntimeError("Physics qualification report is partial")
    if report.get("backend") != "slang-metal":
        raise RuntimeError("Physics qualification did not use the Slang Metal driver")
    if report.get("binary_identity") != binary_identity:
        raise RuntimeError("Installed app or driver hash differs from the staged bundle")
    trajectory = report.get("trajectory")
    if not isinstance(trajectory, dict) or trajectory.get("case_count") != 30:
        raise RuntimeError("Physics qualification must account for all 30 canonical cases")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 36:
        raise RuntimeError("Physics qualification artifact list is incomplete")
    paths = [artifact.get("path") for artifact in artifacts if isinstance(artifact, dict)]
    if len(paths) != len(set(paths)) or any(not isinstance(path, str) or not path for path in paths):
        raise RuntimeError("Physics qualification artifact paths are invalid")
    kinds = [artifact.get("kind") for artifact in artifacts if isinstance(artifact, dict)]
    if (kinds.count("trajectory") != 30 or kinds.count("device-provenance") != 1 or
            kinds.count("advertised-capabilities") != 1 or
            kinds.count("physical-quality") != 3 or
            kinds.count("adverse-lifecycle") != 1):
        raise RuntimeError("Physics qualification artifact scopes are incomplete")
    trajectory_artifacts = [
        artifact for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("kind") == "trajectory"
    ]
    trajectory_statuses = {
        status: sum(artifact.get("status") == status
                    for artifact in trajectory_artifacts)
        for status in ("completed", "unsupported", "failed")
    }
    if (trajectory_statuses["failed"] != 0 or
            trajectory.get("failed") != 0):
        raise RuntimeError("Physics qualification contains failed trajectories")
    if any(trajectory.get(status) != trajectory_statuses[status]
           for status in ("completed", "unsupported", "failed")):
        raise RuntimeError("Physics qualification trajectory totals disagree")
    if trajectory_statuses["completed"] + trajectory_statuses["unsupported"] != 30:
        raise RuntimeError("Physics qualification trajectory coverage is incomplete")
    quality_statuses = [
        artifact.get("status") for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("kind") == "physical-quality"
    ]
    if any(status not in ("completed", "unsupported")
           for status in quality_statuses):
        raise RuntimeError("Physics qualification contains failed physical quality")
    device = report.get("device")
    toolchain = report.get("toolchain")
    if not isinstance(device, dict) or any(not device.get(key) for key in
            ("model", "gpu", "operating_system", "operating_system_version",
             "metal_api", "metal_version", "metal_gpu_family")):
        raise RuntimeError("Public iPadOS/Metal device metadata is incomplete")
    if not isinstance(toolchain, dict) or any(not toolchain.get(key) for key in
            ("compiler", "compiler_version", "slang_version")):
        raise RuntimeError("Qualification toolchain metadata is incomplete")
    forbidden = ("serial", "udid", "pairing", "coredevice")
    def reject_private(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(word in key.lower() for word in forbidden):
                    raise RuntimeError("Physics qualification contains private device identifiers")
                reject_private(child)
        elif isinstance(value, list):
            for child in value:
                reject_private(child)
    reject_private(report)
    return artifacts


def collect_physics_artifacts(device, report, output, temporary):
    output.mkdir(parents=True, exist_ok=True)
    for artifact in report["artifacts"]:
        relative = pathlib.PurePosixPath(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Qualification artifact path escapes its report directory")
        local = output.joinpath(*relative.parts)
        local.parent.mkdir(parents=True, exist_ok=True)
        copied = temporary.joinpath(*relative.parts)
        copied.parent.mkdir(parents=True, exist_ok=True)
        run("xcrun", "devicectl", "device", "copy", "from", "--device", device,
            "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE,
            "--source", "Documents/" + artifact["path"], "--destination", copied,
            "--timeout", "30", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if sha256_file(copied) != artifact.get("sha256"):
            raise RuntimeError("Qualification artifact hash mismatch: " + artifact["path"])
        local.write_bytes(copied.read_bytes())


def assert_process_running(device, pid, directory):
    path = directory / "processes.json"
    run("xcrun", "devicectl", "device", "info", "processes", "--device", device,
        "--json-output", path, "--timeout", "15", stdout=subprocess.DEVNULL)
    processes = json.loads(path.read_text())["result"]["runningProcesses"]
    if not any(p["processIdentifier"] == pid for p in processes):
        raise RuntimeError("Gallery exited before device qualification completed; inspect the saved partial report and device logs")


def wait_report(device, name, nonce, directory, predicate, timeout, health=None, save=None):
    end = time.monotonic() + timeout
    last_progress = -1
    while time.monotonic() < end:
        try:
            report = copy_report(device, name, directory)
            if isinstance(report, dict) and report.get("nonce") == nonce:
                if save:
                    save.write_text(json.dumps(report, indent=2) + "\n")
                if "scenes" in report and len(report["scenes"]) != last_progress:
                    last_progress = len(report["scenes"])
                    print(f"Device qualification: {last_progress}/28 scenes completed", flush=True)
                if predicate(report):
                    return report
        except (subprocess.CalledProcessError, OSError, ValueError):
            pass  # Reports may not exist yet; stale runs never satisfy the nonce.
        if health:
            health()
        time.sleep(2)
    raise RuntimeError(f"Timed out verifying fresh {name}; unlock iPad and inspect device logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=["macos", "ipad"])
    parser.add_argument("--app", type=pathlib.Path)
    parser.add_argument("--device", help="Explicit paired CoreDevice ID; required for iPad")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--qualify", type=int, metavar="FRAMES", default=0,
                        help="iPad: frames per scene plus pause/reset/lifecycle checks")
    parser.add_argument("--physics-qualification", action="store_true",
                        help="iPad: run and collect canonical public-API physics artifacts")
    parser.add_argument("--schedule-profile", action="store_true",
                        help="collect opt-in native Metal dispatch/timing evidence")
    parser.add_argument("--gallery-suite", action="store_true",
                        help="macOS: capture rigid, liquid, cloth, soft and mixed scenes")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "out/gallery-deployment")
    args = parser.parse_args()
    if args.qualify < 0 or args.qualify > 100000:
        parser.error("--qualify must be 1..100000, or omitted")
    if args.platform == "ipad" and not args.device:
        parser.error("--device is required for iPad")
    if args.physics_qualification and args.platform != "ipad":
        parser.error("--physics-qualification is available only for iPad")
    if args.schedule_profile and args.platform == "ipad" and not args.qualify:
        parser.error("iPad --schedule-profile requires --qualify")
    if args.gallery_suite and (args.platform != "macos" or
                               not args.schedule_profile):
        parser.error("--gallery-suite requires macOS --schedule-profile")
    default = ("build-macos-metal-preview/ParallelMater Gallery.app" if args.platform == "macos"
               else "build-ipad-gallery/Release-iphoneos/ParallelMater Gallery.app")
    app = (args.app or ROOT / default).resolve()
    info = build_info(app, args.platform, args.allow_dirty)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.platform == "macos":
        installed = pathlib.Path("/Applications/ParallelMater Gallery.app")
        run("cmake", f"-DSOURCE_APP={app}", f"-DDESTINATION_APP={installed}",
            "-DUPDATE_DOCK=ON", "-P", ROOT / "cmake/DeployMacOSBundle.cmake")
        if build_info(installed, "macos", args.allow_dirty) != info:
            raise RuntimeError("Installed Mac build identity differs from staged bundle")
        executable = installed / "Contents/MacOS/ParallelMater Gallery"
        captures = (MACOS_PROFILE_SUITE if args.gallery_suite else
                    ({"id": "rigid-386", "workload": "rigid-386",
                      "arguments": ("--brick-count", "386")},))
        if not args.schedule_profile:
            captures = ({"id": "smoke", "workload": None,
                         "arguments": ("--constraint-point",)},)
        suite_results = []
        for capture in captures:
            output = (args.output / capture["id"]
                      if args.gallery_suite else args.output)
            output.mkdir(parents=True, exist_ok=True)
            command = [executable, *capture["arguments"]]
            environment = None
            if args.schedule_profile:
                profile = (output / "macos-metal-schedule.jsonl").resolve()
                profile.unlink(missing_ok=True)
                environment = os.environ.copy()
                environment["PM_METAL_SCHEDULE_PROFILE"] = "1"
                environment["PM_METAL_SCHEDULE_PROFILE_FILE"] = str(profile)
                command += ["--frame-timings",
                            (output / "macos-frame-timings.csv").resolve(),
                            "--frames", "180"]
            else:
                command += ["--frames", "6"]
            command += ["--backend", "metal", "--interactive-smoke",
                        output.resolve() / "macos-smoke.ppm"]
            thermal_before = macos_thermal_state()
            result = run(*command, capture_output=True, env=environment)
            thermal_after = macos_thermal_state()
            thermal_report = {
                "schema": "parallel-mater-macos-thermal-capture/v1",
                "revision": info["revision"],
                "scene": capture["id"],
                "before": thermal_before,
                "after": thermal_after,
            }
            (output / "macos-thermal.json").write_text(
                json.dumps(thermal_report, indent=2) + "\n")
            if thermal_before["exit_code"] or thermal_after["exit_code"]:
                raise RuntimeError("pmset could not capture physical Mac thermal state")
            (output / "macos-smoke.log").write_text(result.stdout + result.stderr)
            if (f"Build revision: {info['revision']}" not in result.stdout or
                    f"Execution contract: {EXECUTION_CONTRACT}" not in result.stdout or
                    "renderer=metal-direct-fenced" not in result.stdout):
                raise RuntimeError(
                    "Installed Mac runtime did not prove the expected revision "
                    "and direct Metal renderer")
            if args.schedule_profile:
                evaluation = output / "macos-metal-schedule-evaluation.json"
                run(sys.executable, ROOT / "scripts/Evaluate-MetalSchedule.py",
                    output / "macos-metal-schedule.jsonl", "--workload",
                    capture["workload"], "--priming-samples", "1", "--warmup", "30",
                    "--minimum-samples", "60", "--frame-timings",
                    output / "macos-frame-timings.csv", "--output", evaluation)
                suite_results.append({
                    "id": capture["id"],
                    "workload": capture["workload"],
                    "evaluation": str(evaluation.relative_to(args.output)),
                })
        if args.gallery_suite:
            (args.output / "macos-gallery-suite.json").write_text(json.dumps({
                "schema": "parallel-mater-macos-gallery-suite/v1",
                "revision": info["revision"],
                "execution_contract": info["execution_contract"],
                "captures": suite_results,
            }, indent=2) + "\n")
        print("Installed bundle and native smoke verified:", info)
        run("open", "-a", installed)
    else:
        run("xcrun", "devicectl", "device", "install", "app", "--device", args.device, app)
        nonce = str(uuid.uuid4())
        isolate_physics = bool(args.qualify and args.physics_qualification)
        command = ["xcrun", "devicectl", "device", "process", "launch", "--device", args.device,
                   "--terminate-existing", BUNDLE, "--deployment-nonce", nonce]
        if args.qualify:
            command += ["--qualify", "--qualification-frames", str(args.qualify)]
        if args.physics_qualification and not isolate_physics:
            command += ["--physics-qualification"]
        if args.schedule_profile:
            command += ["--metal-schedule-profile"]
        run(*command)
        with tempfile.TemporaryDirectory(prefix="pm-gallery-verify-") as tmp:
            directory = pathlib.Path(tmp)
            runtime = wait_report(args.device, "runtime-info.json", nonce, directory, lambda r: True, 120)
            if (runtime["revision"] != info["revision"] or
                    runtime["backend"] != info["backend"] or
                    runtime.get("execution_contract") != info["execution_contract"]):
                raise RuntimeError("Installed runtime does not match built revision/backend")
            if args.qualify:
                report = wait_report(args.device, "smoke-report.json", nonce, directory,
                                     lambda r: r.get("completed"), args.timeout,
                                     health=lambda: assert_process_running(args.device, runtime["pid"], directory),
                                     save=args.output / "ipad-qualification.json")
                (args.output / "ipad-qualification.json").write_text(json.dumps(report, indent=2) + "\n")
                validate_mobile_gallery_report(report, nonce=nonce)
                # Exercise the OS lifecycle, separately from session-only checks.
                run("xcrun", "devicectl", "device", "process", "launch", "--device", args.device, "com.apple.Preferences")
                background = wait_report(args.device, "runtime-info.json", nonce, directory,
                                         lambda r: not r["active"] and r["backgrounds"] > runtime["backgrounds"], 60)
                # The native profile is append-only while the Gallery is active.
                # Snapshot it while the display link is suspended so CoreDevice
                # never tries to copy a file that is still growing.
                if args.schedule_profile:
                    copy_artifact(args.device, "metal-schedule-profile.jsonl",
                                  args.output / "ipad-metal-schedule.jsonl")
                time.sleep(5)
                run("xcrun", "devicectl", "device", "process", "launch", "--device", args.device, BUNDLE)
                foreground = wait_report(args.device, "runtime-info.json", nonce, directory,
                                         lambda r: r["active"] and r["foregrounds"] > background["foregrounds"], 60)
                if foreground["frames"] != background["frames"]:
                    raise RuntimeError("Simulation advanced while backgrounded")
                runtime = foreground
            if args.physics_qualification:
                physics_nonce = nonce
                physics_runtime = runtime
                if isolate_physics:
                    # The scene sweep intentionally backgrounds Gallery to test
                    # lifecycle recovery. Do not suspend the independent public-
                    # API runner halfway through a case; give it a fresh process
                    # and nonce after Gallery and schedule capture are complete.
                    physics_nonce = str(uuid.uuid4())
                    run("xcrun", "devicectl", "device", "process", "launch",
                        "--device", args.device, "--terminate-existing", BUNDLE,
                        "--deployment-nonce", physics_nonce,
                        "--physics-qualification")
                    physics_runtime = wait_report(
                        args.device, "runtime-info.json", physics_nonce,
                        directory, lambda r: True, 120)
                    if (physics_runtime["revision"] != info["revision"] or
                            physics_runtime["backend"] != info["backend"] or
                            physics_runtime.get("execution_contract") !=
                            info["execution_contract"]):
                        raise RuntimeError(
                            "Physics runtime does not match built revision/backend")
                    (args.output / "ipad-physics-runtime.json").write_text(
                        json.dumps(physics_runtime, indent=2) + "\n")
                physics = wait_report(args.device, "physics-qualification.json", physics_nonce,
                                      directory, lambda r: r.get("completed"), args.timeout,
                                      health=lambda: assert_process_running(
                                          args.device, physics_runtime["pid"], directory),
                                      save=args.output / "ipad-physics-qualification.json")
                identity = ipad_binary_identity(app)
                validate_physics_qualification(
                    physics, physics_nonce, info["revision"], identity)
                collect_physics_artifacts(args.device, physics, args.output, directory)
            if args.schedule_profile and not args.qualify:
                copy_artifact(args.device, "metal-schedule-profile.jsonl",
                              args.output / "ipad-metal-schedule.jsonl")
            (args.output / "ipad-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        print("Installed runtime verified:", runtime["revision"], runtime["backend"], runtime["device"])
    (args.output / f"{args.platform}-build-info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Gallery deployment failed: {error}") from error
