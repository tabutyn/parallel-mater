#!/usr/bin/env python3
"""Deploy an already-built preview; verify the installed revision and backend."""
import argparse
import hashlib
import json
import pathlib
import plistlib
import subprocess
import tempfile
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = "com.ballrollergames.parallelmater.gallery"


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, text=True, **kwargs)


def build_info(app, platform, allow_dirty=False):
    resources = app / "Contents/Resources" if platform == "macos" else app
    info = json.loads((resources / "build-info.json").read_text())
    revision = run("git", "rev-parse", "HEAD", cwd=ROOT, capture_output=True).stdout.strip()
    dirty = run("git", "status", "--porcelain", cwd=ROOT, capture_output=True).stdout.strip()
    if info["revision"] != revision:
        raise RuntimeError("Built revision differs from HEAD; rebuild before deployment")
    if not allow_dirty and (info["dirty"] or dirty):
        raise RuntimeError("Release deployment requires a clean commit and clean rebuild (--allow-dirty is preview-only)")
    if info["backend"] != "metal" or info["release_channel"] != "development-preview":
        raise RuntimeError("Expected explicitly labeled Metal development preview")
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
    if not isinstance(artifacts, list) or len(artifacts) != 35:
        raise RuntimeError("Physics qualification artifact list is incomplete")
    paths = [artifact.get("path") for artifact in artifacts if isinstance(artifact, dict)]
    if len(paths) != len(set(paths)) or any(not isinstance(path, str) or not path for path in paths):
        raise RuntimeError("Physics qualification artifact paths are invalid")
    kinds = [artifact.get("kind") for artifact in artifacts if isinstance(artifact, dict)]
    if (kinds.count("trajectory") != 30 or kinds.count("device-provenance") != 1 or
            kinds.count("advertised-capabilities") != 1 or
            kinds.count("physical-quality") != 2 or
            kinds.count("adverse-lifecycle") != 1):
        raise RuntimeError("Physics qualification artifact scopes are incomplete")
    device = report.get("device")
    toolchain = report.get("toolchain")
    if not isinstance(device, dict) or any(not device.get(key) for key in
            ("model", "gpu", "operating_system", "operating_system_version", "metal_api")):
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
                    print(f"Device qualification: {last_progress}/27 scenes completed", flush=True)
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
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "out/gallery-deployment")
    args = parser.parse_args()
    if args.qualify < 0 or args.qualify > 100000:
        parser.error("--qualify must be 1..100000, or omitted")
    if args.platform == "ipad" and not args.device:
        parser.error("--device is required for iPad")
    if args.physics_qualification and args.platform != "ipad":
        parser.error("--physics-qualification is available only for iPad")
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
        result = run(installed / "Contents/MacOS/ParallelMater Gallery", "--constraint-point",
                     "--backend", "metal", "--frames", "6", "--interactive-smoke", args.output.resolve() / "macos-smoke.ppm", capture_output=True)
        (args.output / "macos-smoke.log").write_text(result.stdout + result.stderr)
        if f"Build revision: {info['revision']}" not in result.stdout or "renderer=metal-direct-fenced" not in result.stdout:
            raise RuntimeError("Installed Mac runtime did not prove the expected revision and direct Metal renderer")
        print("Installed bundle and native smoke verified:", info)
        run("open", "-a", installed)
    else:
        run("xcrun", "devicectl", "device", "install", "app", "--device", args.device, app)
        nonce = str(uuid.uuid4())
        command = ["xcrun", "devicectl", "device", "process", "launch", "--device", args.device,
                   "--terminate-existing", BUNDLE, "--deployment-nonce", nonce]
        if args.qualify:
            command += ["--qualify", "--qualification-frames", str(args.qualify)]
        if args.physics_qualification:
            command += ["--physics-qualification"]
        run(*command)
        with tempfile.TemporaryDirectory(prefix="pm-gallery-verify-") as tmp:
            directory = pathlib.Path(tmp)
            runtime = wait_report(args.device, "runtime-info.json", nonce, directory, lambda r: True, 120)
            if runtime["revision"] != info["revision"] or runtime["backend"] != info["backend"]:
                raise RuntimeError("Installed runtime does not match built revision/backend")
            if args.qualify:
                report = wait_report(args.device, "smoke-report.json", nonce, directory,
                                     lambda r: r.get("completed"), args.timeout,
                                     health=lambda: assert_process_running(args.device, runtime["pid"], directory),
                                     save=args.output / "ipad-qualification.json")
                (args.output / "ipad-qualification.json").write_text(json.dumps(report, indent=2) + "\n")
                if len(report["scenes"]) != 27 or not all(s["passed"] for s in report["scenes"]):
                    raise RuntimeError("iPad scene qualification failed; see saved report")
                # Exercise the OS lifecycle, separately from session-only checks.
                run("xcrun", "devicectl", "device", "process", "launch", "--device", args.device, "com.apple.Preferences")
                background = wait_report(args.device, "runtime-info.json", nonce, directory,
                                         lambda r: not r["active"] and r["backgrounds"] > runtime["backgrounds"], 60)
                time.sleep(5)
                run("xcrun", "devicectl", "device", "process", "launch", "--device", args.device, BUNDLE)
                foreground = wait_report(args.device, "runtime-info.json", nonce, directory,
                                         lambda r: r["active"] and r["foregrounds"] > background["foregrounds"], 60)
                if foreground["frames"] != background["frames"]:
                    raise RuntimeError("Simulation advanced while backgrounded")
                runtime = foreground
            if args.physics_qualification:
                physics = wait_report(args.device, "physics-qualification.json", nonce,
                                      directory, lambda r: r.get("completed"), args.timeout,
                                      health=lambda: assert_process_running(
                                          args.device, runtime["pid"], directory),
                                      save=args.output / "ipad-physics-qualification.json")
                identity = ipad_binary_identity(app)
                validate_physics_qualification(
                    physics, nonce, info["revision"], identity)
                collect_physics_artifacts(args.device, physics, args.output, directory)
            (args.output / "ipad-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        print("Installed runtime verified:", runtime["revision"], runtime["backend"], runtime["device"])
    (args.output / f"{args.platform}-build-info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Gallery deployment failed: {error}") from error
