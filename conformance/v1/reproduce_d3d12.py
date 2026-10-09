#!/usr/bin/env python3
"""Run the supported D3D12 rigid cases repeatedly against CUDA goldens."""
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
    parser.add_argument("--repetitions", type=int, default=10)
    arguments = parser.parse_args()
    checked = subprocess.run([str(arguments.runner), "--check-inputs"],
                             check=False)
    if checked.returncode != 0:
        return checked.returncode
    listed = subprocess.run([str(arguments.runner), "--list"], check=False,
                            capture_output=True, text=True)
    if listed.returncode != 0:
        return listed.returncode
    case_ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not case_ids:
        print("D3D12 runner exposed no rigid cases", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="parallel-mater-d3d12-") as temp:
        root = pathlib.Path(temp)
        for repetition in range(arguments.repetitions):
            actual = root / f"run-{repetition}"
            run = subprocess.run([
                str(arguments.runner), "--case", "all", "--output", str(actual)
            ], check=False)
            if run.returncode != 0:
                return run.returncode
            for case_id in case_ids:
                report = root / f"report-{repetition}-{case_id}.json"
                compared = subprocess.run([
                    sys.executable, str(arguments.comparator),
                    "--cases", str(arguments.cases),
                    "--expected", str(arguments.golden),
                    "--actual", str(actual), "--case", case_id,
                    "--report", str(report),
                ], check=False)
                if compared.returncode != 0:
                    return compared.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
