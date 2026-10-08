#!/usr/bin/env python3
"""Retain a complete CUDA reference package; never rewrite goldens or tolerances."""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    value = json.loads(path.read_text(), parse_constant=reject_constant,
                       object_pairs_hook=unique_keys)
    # Also reject finite-looking literals which overflow Python's float.
    json.dumps(value, allow_nan=False)
    return value


def deterministic_document(value: dict) -> dict:
    result = copy.deepcopy(value)
    # Only measured timings vary legitimately on the same backend/device.
    # Keep state_hash, provenance, ordering, counts, IDs, and every checkpoint.
    result.get("diagnostics", {}).pop("timings", None)
    return result


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def first_difference(expected, actual, path="$"):
    if type(expected) is not type(actual):
        return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return {"path": path, "expected_keys": sorted(expected),
                    "actual_keys": sorted(actual)}
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return {"path": path, "expected_length": len(expected),
                    "actual_length": len(actual)}
        for i, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{i}]")
            if difference:
                return difference
    elif canonical_bytes(expected) != canonical_bytes(actual):
        return {"path": path, "expected": expected, "actual": actual}
    return None


def command(args, *, cwd=None):
    completed = subprocess.run(list(map(str, args)), cwd=cwd, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return completed.returncode, completed.stdout


def git(source: Path, *args):
    code, output = command(["git", "-C", source, *args])
    if code:
        raise RuntimeError(output)
    return output.strip()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def capture(args) -> int:
    source, runner, output = args.source.resolve(), args.runner.resolve(), args.output.resolve()
    if args.runs < 10:
        raise ValueError("reference qualification requires at least ten complete runs")
    if output.is_relative_to(source):
        raise ValueError("output must be outside the source worktree")
    if output.exists():
        raise ValueError("output already exists; preserve it and choose a new directory")
    commit = git(source, "rev-parse", "HEAD")
    status = git(source, "status", "--porcelain", "--untracked-files=normal")
    if status and not args.allow_dirty:
        raise ValueError("reference source is dirty; commit scoped changes first")
    code, provenance_text = command([runner, "--provenance"])
    if code:
        raise RuntimeError(f"runner provenance failed ({code}): {provenance_text}")
    provenance = json.loads(provenance_text)
    if provenance["source_commit"] != commit:
        raise ValueError("runner/source revisions differ; reconfigure CMake and rebuild")
    output.mkdir(parents=True)
    report = {
        "schema": "parallel-mater-cuda-reference-package/v1",
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit": commit, "source_status": status, "clean_source": not bool(status),
        "device_and_toolchain": provenance, "runner_sha256": sha256(runner),
        "runs_requested": args.runs, "excluded_from_repeatability": ["diagnostics.timings"],
        "runs": [], "repeatable": False, "qualified": False,
        "first_divergence": None,
    }
    write_json(output / "report.json", report)
    for label, cmd in [
        ("cmake", ["cmake", "--version"]),
        ("gpu", ["nvidia-smi"]),
    ]:
        code, text = command(cmd)
        (output / f"{label}.log").write_text(text)
        report[label + "_returncode"] = code
    cache = runner.parent / "CMakeCache.txt"
    if cache.exists():
        (output / "CMakeCache.txt").write_bytes(cache.read_bytes())
    code, text = command([runner, "--check-inputs"])
    (output / "check-inputs.log").write_text(text)
    if code:
        raise RuntimeError(f"input validation failed ({code}); see check-inputs.log")
    code, text = command([runner, "--list"])
    if code:
        raise RuntimeError("cannot list cases")
    expected = sorted(line + ".json" for line in text.splitlines() if line)
    report["case_count"] = len(expected)
    reference = {}
    parity_reference = None
    for run in range(1, args.runs + 1):
        directory = output / f"run-{run:02d}"
        code, text = command([runner, "--case", "all", "--output", directory])
        (output / f"run-{run:02d}.log").write_text(text)
        item = {"run": run, "returncode": code, "cases": {}, "differences": []}
        names = sorted(path.name for path in directory.glob("*.json"))
        if code or names != expected:
            item["differences"].append({"path": "$.case_files", "expected": expected,
                                        "actual": names, "returncode": code})
        for name in names:
            path = directory / name
            try:
                document = deterministic_document(read_json(path))
                digest = hashlib.sha256(canonical_bytes(document)).hexdigest()
                item["cases"][name] = {"file_sha256": sha256(path),
                                       "deterministic_sha256": digest}
                if run == 1:
                    reference[name] = document
                elif name not in reference:
                    item["differences"].append({"path": name, "error": "missing first-run result"})
                else:
                    difference = first_difference(reference[name], document, name)
                    if difference:
                        item["differences"].append(difference)
            except (ValueError, TypeError) as error:
                item["differences"].append({"path": name, "error": str(error)})
        if args.parity_runner:
            parity_file = output / f"all-systems-{run:02d}.capture"
            parity_code, parity_log = command([args.parity_runner.resolve(), parity_file])
            (output / f"all-systems-{run:02d}.log").write_text(parity_log)
            if parity_code or not parity_file.exists():
                item["differences"].append({"path": "all-systems", "returncode": parity_code})
            else:
                contents = parity_file.read_bytes()
                item["parity_sha256"] = sha256(parity_file)
                if run == 1:
                    parity_reference = contents
                elif contents != parity_reference:
                    item["differences"].append({"path": "all-systems", "error": "capture bytes differ"})
        if item["differences"] and report["first_divergence"] is None:
            report["first_divergence"] = {"run": run, **item["differences"][0]}
        report["runs"].append(item)
        write_json(output / "report.json", report)
        print(f"Run {run}/{args.runs}: {len(names)} cases, "
              f"{len(item['differences'])} repeatability failures", flush=True)
    report["source_unchanged"] = (
        git(source, "rev-parse", "HEAD") == commit and
        git(source, "status", "--porcelain", "--untracked-files=normal") == status)
    report["repeatable"] = report["first_divergence"] is None
    report["qualified"] = report["repeatable"] and report["clean_source"] and report["source_unchanged"]
    write_json(output / "report.json", report)
    print(f"Reference package: {output}; qualified={report['qualified']}")
    return 0 if report["qualified"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--parity-runner", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="diagnostics only: mark package unqualified; never a clean reference")
    try:
        return capture(parser.parse_args())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"reference capture: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
