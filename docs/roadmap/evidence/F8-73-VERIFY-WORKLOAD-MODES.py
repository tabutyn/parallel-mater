#!/usr/bin/env python3
"""Evaluate both retained workload modes with the unchanged F8 suite oracle.

The capture builder selects validation timings for its aggregate report. F8-73
also requires plain-mode acceptance. This evidence helper evaluates each mode
independently and reruns all negative controls; it does not alter any limit or
combine passing observations from separate captures.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    module_path = args.repo / "conformance/v1/realtime_device_capture.py"
    spec = importlib.util.spec_from_file_location("f873_capture_oracle", module_path)
    assert spec is not None and spec.loader is not None
    capture_oracle = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = capture_oracle
    spec.loader.exec_module(capture_oracle)
    read = capture_oracle.read_json
    candidate_path = capture_oracle._candidate_path()
    contracts = (
        read(capture_oracle.PROFILE_PATH), read(capture_oracle.INVENTORY_PATH),
        read(capture_oracle.CAPABILITY_PATH), read(capture_oracle.SUITE_PATH),
        read(candidate_path), capture_oracle.file_sha256(candidate_path, portable_text=True),
    )
    aggregate_path = args.capture / "capture.json"
    aggregate = read(aggregate_path)
    reports = {}
    for mode in ("plain", "validation"):
        selected = copy.deepcopy(aggregate)
        raw_hashes = {}
        for index, template in enumerate(aggregate["workloads"]):
            raw_path = args.capture / "raw" / mode / "workloads" / (template["id"] + ".json")
            raw = read(raw_path)
            raw_hashes[template["id"]] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            raw.pop("raw_samples", None)
            raw["executions"] = copy.deepcopy(template["executions"])
            selected["workloads"][index] = raw
        selected["negative_controls"] = capture_oracle._negative_controls(selected, contracts)
        reports[mode] = {
            "report": capture_oracle.SUITE.evaluate(selected, *contracts),
            "negative_controls": selected["negative_controls"],
            "raw_artifact_sha256": raw_hashes,
        }
    result = {
        "schema": "parallel-mater-f8-73-both-workload-modes/v1",
        "passed": all(value["report"]["passed"] for value in reports.values()),
        "aggregate_capture_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(capture_oracle.PROFILE_PATH.read_bytes()).hexdigest(),
        "oracle_source_sha256": hashlib.sha256(capture_oracle.SUITE_MODULE_PATH.read_bytes()).hexdigest(),
        "modes": reports,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for mode, value in reports.items():
        print(mode, "passed=" + str(value["report"]["passed"]),
              "failures=" + str(value["report"]["failure_count"]))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
