#!/usr/bin/env python3
"""Fetch and verify the pinned reference-document toolchain inputs."""
import argparse
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import tarfile
from urllib.request import urlopen

from provenance import ROOT, file_sha256

LOCK_PATH = Path(__file__).with_name("toolchain.json")
DEFAULT_FONT_DIR = ROOT / "out/reference-toolchain/fonts"


def load_lock():
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def verify_python(lock=None):
    lock = lock or load_lock()
    expected = lock["python"]
    actual = {"implementation": platform.python_implementation(),
              "version": platform.python_version()}
    if actual != expected:
        raise RuntimeError(f"reference Python mismatch: expected {expected}, found {actual}")
    return actual


def verify_packages(lock=None):
    lock = lock or load_lock()
    actual = {}
    for package, expected in lock["packages"].items():
        try:
            actual[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"missing reference package {package}=={expected}") from error
        if actual[package] != expected:
            raise RuntimeError(
                f"reference package mismatch: expected {package}=={expected}, "
                f"found {actual[package]}")
    return actual


def download(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def materialize(entry, archives):
    source = archives.setdefault(entry["url"], download(entry["url"]))
    expected_archive = entry.get("archive_sha256")
    if expected_archive:
        import hashlib
        actual = hashlib.sha256(source).hexdigest()
        if actual != expected_archive:
            raise RuntimeError(
                f"download hash mismatch for {entry['url']}: expected "
                f"{expected_archive}, found {actual}")
        with tarfile.open(fileobj=io.BytesIO(source), mode="r:bz2") as archive:
            member = archive.extractfile(entry["member"])
            if member is None:
                raise RuntimeError(f"missing {entry['member']} in {entry['url']}")
            return member.read()
    return source


def prepare_fonts(directory=DEFAULT_FONT_DIR, allow_download=False, lock=None):
    lock = lock or load_lock()
    directory = Path(directory)
    entries = lock["fonts"] + lock.get("licenses", [])
    missing = [entry for entry in entries if not (directory / entry["name"]).is_file()]
    if missing and not allow_download:
        names = ", ".join(entry["name"] for entry in missing)
        raise RuntimeError(
            f"missing pinned reference inputs ({names}); run "
            "'python tools/reference/bootstrap.py'")
    directory.mkdir(parents=True, exist_ok=True)
    archives = {}
    for entry in entries:
        target = directory / entry["name"]
        if not target.is_file():
            target.write_bytes(materialize(entry, archives))
        actual = file_sha256(target)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"reference input hash mismatch for {target}: expected "
                f"{entry['sha256']}, found {actual}")
    return directory


def verify_environment(font_dir=DEFAULT_FONT_DIR):
    lock = load_lock()
    python = verify_python(lock)
    packages = verify_packages(lock)
    directory = prepare_fonts(font_dir, lock=lock)
    return {
        "lock": str(LOCK_PATH.relative_to(ROOT)).replace("\\", "/"),
        "lock_sha256": file_sha256(LOCK_PATH),
        "python": python,
        "packages": packages,
        "fonts": {entry["name"]: entry["sha256"] for entry in lock["fonts"]},
        "font_dir": directory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, default=DEFAULT_FONT_DIR)
    parser.add_argument("--fetch-fonts", action="store_true")
    parser.add_argument("--packages", action="store_true",
                        help="Also verify installed Python packages")
    args = parser.parse_args()
    lock = load_lock()
    verify_python(lock)
    directory = prepare_fonts(args.font_dir, allow_download=args.fetch_fonts, lock=lock)
    if args.packages:
        verify_packages(lock)
    print(f"PASS: pinned reference toolchain inputs at {directory}")


if __name__ == "__main__":
    main()
