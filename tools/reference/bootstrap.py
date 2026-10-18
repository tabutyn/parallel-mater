#!/usr/bin/env python3
"""Create the pinned reference environment and optionally build or check artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from provenance import ROOT
from toolchain import DEFAULT_FONT_DIR, load_lock, prepare_fonts, verify_python


def venv_python(directory):
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(command):
    print("+", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / "out/reference-toolchain/venv")
    parser.add_argument("--build", action="store_true",
                        help="Regenerate the tracked release artifacts")
    parser.add_argument("--check", action="store_true",
                        help="Verify the tracked artifacts by clean regeneration")
    parser.add_argument("--skip-vulkan-header", action="store_true")
    args = parser.parse_args()
    lock = load_lock()
    verify_python(lock)
    args.venv = args.venv.resolve()
    python = venv_python(args.venv)
    if not python.is_file():
        run([sys.executable, "-m", "venv", args.venv])
    run([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
         "-r", ROOT / "tools/reference/requirements.txt"])
    prepare_fonts(DEFAULT_FONT_DIR, allow_download=True, lock=lock)
    run([python, "tools/reference/toolchain.py", "--packages"])
    if args.build:
        run([python, "tools/reference/build.py"])
    if args.check:
        command = [python, "tools/reference/check.py"]
        if args.skip_vulkan_header:
            command.append("--skip-vulkan-header")
        run(command)
    if not args.build and not args.check:
        print(json.dumps({
            "python": str(python),
            "font_dir": str(DEFAULT_FONT_DIR),
            "next": "add --build to regenerate or --check to verify",
        }, indent=2))


if __name__ == "__main__":
    main()
