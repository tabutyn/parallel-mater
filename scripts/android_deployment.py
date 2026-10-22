#!/usr/bin/env python3
"""Build and deployment provenance for physical Android diagnostics."""

import argparse
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess
import uuid
import zipfile


PACKAGE = "org.parallelmater.gallery"
DRIVER_MEMBER = "lib/arm64-v8a/libparallel_mater_driver_slang_vulkan.so"
PIXEL_7 = {"manufacturer": "Google", "model": "Pixel 7", "product": "panther"}


class DeploymentError(RuntimeError):
    """A diagnostic would not be able to prove what ran."""


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_path(path):
    return sha256_bytes(path.read_bytes())


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True,
        capture_output=True).stdout


def source_identity(root):
    """Return commit plus a deterministic digest of tracked/untracked changes."""
    root = pathlib.Path(root).resolve()
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    status = _git(root, "status", "--porcelain=v1", "-z",
                  "--untracked-files=all")
    dirty = bool(status)
    if not dirty:
        return {"revision": revision, "dirty": False,
                "patch_sha256": "clean"}
    digest = hashlib.sha256()
    digest.update(b"parallel-mater-dirty-patch/v1\0")
    digest.update(_git(root, "diff", "--binary", "HEAD", "--"))
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    for raw_name in sorted(filter(None, untracked.split(b"\0"))):
        name = os.fsdecode(raw_name)
        path = root / name
        digest.update(b"untracked\0" + raw_name + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0" + os.fsencode(os.readlink(path)))
        else:
            digest.update(b"file\0" + path.read_bytes())
        digest.update(b"\0")
    return {"revision": revision, "dirty": True,
            "patch_sha256": digest.hexdigest()}


def inspect_apk_bytes(apk_bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(apk_bytes)) as archive:
            driver = archive.read(DRIVER_MEMBER)
    except (zipfile.BadZipFile, KeyError) as error:
        raise DeploymentError(
            f"APK does not contain {DRIVER_MEMBER}: {error}") from error
    return {"apk_sha256": sha256_bytes(apk_bytes),
            "driver_sha256": sha256_bytes(driver)}


def parse_adb_devices(text):
    devices = {}
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2:
            devices[fields[0]] = fields[1]
    return devices


def require_connected_serial(text, serial):
    devices = parse_adb_devices(text)
    if serial not in devices:
        raise DeploymentError(f"requested Android device {serial!r} is not connected")
    if devices[serial] != "device":
        raise DeploymentError(
            f"Android device {serial!r} is {devices[serial]!r}, not usable")


def verify_device_properties(properties):
    wrong = [f"{name}={properties.get(name)!r} (expected {value!r})"
             for name, value in PIXEL_7.items()
             if properties.get(name) != value]
    if wrong:
        raise DeploymentError("wrong physical baseline device: " + ", ".join(wrong))


def parse_package_path(text):
    paths = [line[len("package:"):].strip() for line in text.splitlines()
             if line.startswith("package:")]
    base = [path for path in paths if path.endswith("/base.apk")]
    if len(base) != 1:
        raise DeploymentError(
            "installed package is missing or has no unique base.apk")
    return base[0]


def verify_installed_artifact(local, installed):
    if local["apk_sha256"] != installed["apk_sha256"]:
        raise DeploymentError("installed APK hash does not match the local APK")
    if local["driver_sha256"] != installed["driver_sha256"]:
        raise DeploymentError("installed Vulkan driver hash does not match the local APK")


def verify_runtime_identity(runtime, nonce, source):
    if runtime.get("nonce") != nonce:
        raise DeploymentError("runtime report nonce is stale or belongs to another run")
    build = runtime.get("build")
    if not isinstance(build, dict):
        raise DeploymentError("runtime report has no embedded build identity")
    expected = {
        "source_revision": source["revision"],
        "source_dirty": source["dirty"],
        "source_patch_sha256": source["patch_sha256"],
    }
    mismatches = [name for name, value in expected.items()
                  if build.get(name) != value]
    if runtime.get("revision") != source["revision"]:
        mismatches.append("native_revision")
    if mismatches:
        raise DeploymentError(
            "runtime build identity mismatch: " + ", ".join(mismatches))
    if runtime.get("backend") != "vulkan" or runtime.get("presentation") != "vulkan-direct":
        raise DeploymentError("runtime is not the direct Vulkan backend")


class Adb:
    def __init__(self, executable, serial):
        self.executable = executable
        self.serial = serial

    def global_text(self, *arguments):
        return subprocess.run(
            [self.executable, *arguments], check=True, text=True,
            capture_output=True).stdout

    def text(self, *arguments):
        return subprocess.run(
            [self.executable, "-s", self.serial, *arguments], check=True,
            text=True, capture_output=True).stdout

    def bytes(self, *arguments):
        return subprocess.run(
            [self.executable, "-s", self.serial, *arguments], check=True,
            capture_output=True).stdout


def device_properties(adb):
    names = {
        "manufacturer": "ro.product.manufacturer",
        "model": "ro.product.model",
        "product": "ro.product.name",
        "android_version": "ro.build.version.release",
        "api_level": "ro.build.version.sdk",
        "security_patch": "ro.build.version.security_patch",
        "build_fingerprint": "ro.build.fingerprint",
    }
    return {name: adb.text("shell", "getprop", prop).strip()
            for name, prop in names.items()}


def install_and_verify(adb, apk):
    require_connected_serial(adb.global_text("devices", "-l"), adb.serial)
    properties = device_properties(adb)
    verify_device_properties(properties)
    apk_bytes = pathlib.Path(apk).read_bytes()
    local = inspect_apk_bytes(apk_bytes)
    result = adb.text("install", "-r", str(pathlib.Path(apk).resolve())).strip()
    if not result.endswith("Success"):
        raise DeploymentError(f"adb install did not report success: {result}")
    package_path = parse_package_path(adb.text("shell", "pm", "path", PACKAGE))
    installed_bytes = adb.bytes("exec-out", "cat", package_path)
    installed = inspect_apk_bytes(installed_bytes)
    verify_installed_artifact(local, installed)
    version = adb.text("shell", "dumpsys", "package", PACKAGE)
    version_name = re.search(r"\bversionName=([^\s]+)", version)
    version_code = re.search(r"\bversionCode=(\d+)", version)
    if not version_name or not version_code:
        raise DeploymentError("could not read installed package version")
    return {
        "serial": adb.serial,
        "device": properties,
        "local_apk_sha256": local["apk_sha256"],
        "installed_apk_sha256": installed["apk_sha256"],
        "local_driver_sha256": local["driver_sha256"],
        "installed_driver_sha256": installed["driver_sha256"],
        "installed_package_path": package_path,
        "package": PACKAGE,
        "version_name": version_name.group(1),
        "version_code": int(version_code.group(1)),
    }


def fresh_nonce():
    return uuid.uuid4().hex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-properties", help="print Gradle source-identity properties")
    build.add_argument("--root", type=pathlib.Path, required=True)
    identity = subparsers.add_parser(
        "source-identity", help="print source identity as JSON")
    identity.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    value = source_identity(args.root)
    if args.command == "build-properties":
        print(f"-PsourceRevision={value['revision']}")
        print(f"-PsourceDirty={str(value['dirty']).lower()}")
        print(f"-PsourcePatchSha256={value['patch_sha256']}")
    else:
        print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
