#!/usr/bin/env python3
"""Regenerate CUDA results in a temporary directory and compare to goldens."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=pathlib.Path, required=True)
    parser.add_argument("--comparator", type=pathlib.Path, required=True)
    parser.add_argument("--cases", type=pathlib.Path, required=True)
    parser.add_argument("--golden", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    checked = subprocess.run([str(arguments.runner), "--check-inputs"],
                             check=False)
    if checked.returncode != 0:
        return checked.returncode
    with tempfile.TemporaryDirectory(prefix="parallel-mater-conformance-") as temp:
        generated = pathlib.Path(temp) / "cuda"
        run = subprocess.run([
            str(arguments.runner), "--case", "all", "--output", str(generated)
        ], check=False)
        if run.returncode != 0:
            return run.returncode
        report = pathlib.Path(temp) / "report.json"
        compared = subprocess.run([
            sys.executable, str(arguments.comparator),
            "--cases", str(arguments.cases),
            "--expected", str(arguments.golden),
            "--actual", str(generated),
            "--report", str(report),
        ], check=False)
        return compared.returncode


if __name__ == "__main__":
    sys.exit(main())
