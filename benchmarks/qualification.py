#!/usr/bin/env python3
"""Qualify headless benchmark measurements against physical-quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MEASUREMENT_SCHEMA = "parallel-mater-benchmark-measurement/v1"
REPORT_SCHEMA = "parallel-mater-benchmark-qualification-report/v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
MEMORY_FIELDS = (
    "application_allocation_bytes",
    "library_device_bytes",
    "library_host_bytes",
    "borrowed_buffer_bytes",
    "scratch_high_water_bytes",
)


@dataclass
class Failure:
    path: str
    message: str
    actual: Any = None
    expected: Any = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes | str) -> Any:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}")

    return json.loads(data, parse_constant=invalid_constant)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _revision(value: Any) -> bool:
    return isinstance(value, str) and REVISION.fullmatch(value) is not None


def _quality_module(root: Path):
    path = root / "conformance/v1/quality.py"
    spec = importlib.util.spec_from_file_location("pm_benchmark_quality", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metadata(measurement: dict[str, Any], quality_result: dict[str, Any] | None,
              failures: list[Failure]) -> None:
    source = measurement.get("source")
    if (not isinstance(source, dict) or not _revision(source.get("commit")) or
            source.get("dirty") is not False):
        failures.append(Failure("$.source",
                                "clean source revision is required", source))
    driver = measurement.get("driver")
    if (not isinstance(driver, dict) or not _nonempty(driver.get("name")) or
            not _nonempty(driver.get("version")) or
            not _digest(driver.get("binary_sha256"))):
        failures.append(Failure("$.driver",
                                "driver name, version and binary SHA-256 are required",
                                driver))
    device = measurement.get("device")
    if (not isinstance(device, dict) or not _nonempty(device.get("name")) or
            not isinstance(device.get("vendor_id"), int) or
            isinstance(device.get("vendor_id"), bool) or
            device["vendor_id"] < 0 or
            not isinstance(device.get("device_id"), int) or
            isinstance(device.get("device_id"), bool) or
            device["device_id"] < 0 or
            not _nonempty(device.get("driver_version"))):
        failures.append(Failure("$.device",
                                "device name, IDs and driver version are required",
                                device))
    toolchain = measurement.get("toolchain")
    if (not isinstance(toolchain, dict) or
            not _nonempty(toolchain.get("compiler")) or
            not _nonempty(toolchain.get("version"))):
        failures.append(Failure("$.toolchain",
                                "compiler name and version are required", toolchain))

    if quality_result is None:
        return
    quality_source = quality_result.get("source")
    quality_driver = quality_result.get("driver")
    quality_toolchain = quality_result.get("toolchain")
    if quality_source != source:
        failures.append(Failure("$.quality_result.source",
                                "quality source does not match benchmark source",
                                quality_source, source))
    expected_quality_driver = {
        "name": driver.get("name") if isinstance(driver, dict) else None,
        "device": device.get("name") if isinstance(device, dict) else None,
        "version": driver.get("version") if isinstance(driver, dict) else None,
        "binary_sha256": (driver.get("binary_sha256")
                           if isinstance(driver, dict) else None),
    }
    if quality_driver != expected_quality_driver:
        failures.append(Failure("$.quality_result.driver",
                                "quality driver/device does not match benchmark",
                                quality_driver, expected_quality_driver))
    if quality_toolchain != toolchain:
        failures.append(Failure("$.quality_result.toolchain",
                                "quality toolchain does not match benchmark",
                                quality_toolchain, toolchain))


def _nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _measurements(measurement: dict[str, Any], profile_hash: str,
                  failures: list[Failure]) -> dict[str, Any] | None:
    workload = measurement.get("workload")
    if (not isinstance(workload, dict) or not _nonempty(workload.get("id")) or
            not _digest(workload.get("revision_sha256")) or
            not _nonempty(workload.get("operation"))):
        failures.append(Failure("$.workload",
                                "workload id, revision hash and operation are required",
                                workload))
        workload = {}
    method = measurement.get("measurement_method")
    expected_method = {
        "timing": "host_monotonic",
        "memory": ["pmGetDeviceMemoryBudget",
                   "PmStepStatistics.scratchHighWaterBytes"],
    }
    if method != expected_method:
        failures.append(Failure("$.measurement_method",
                                "measurement method mismatch", method,
                                expected_method))
    samples = measurement.get("samples")
    if not isinstance(samples, list) or len(samples) < 20:
        failures.append(Failure("$.samples",
                                "at least 20 measured samples are required",
                                len(samples) if isinstance(samples, list) else samples,
                                ">= 20"))
        return None
    durations: list[float] = []
    total_duration_ns = 0
    total_operations = 0
    memory_peaks = {field: 0 for field in MEMORY_FIELDS}
    for index, sample in enumerate(samples):
        path = f"$.samples[{index}]"
        if not isinstance(sample, dict):
            failures.append(Failure(path, "expected an object", sample))
            continue
        duration = sample.get("duration_ns")
        operations = sample.get("operations")
        if (not isinstance(duration, int) or isinstance(duration, bool) or
                duration <= 0):
            failures.append(Failure(path + ".duration_ns",
                                    "duration must be a positive integer", duration))
            continue
        if (not isinstance(operations, int) or isinstance(operations, bool) or
                operations <= 0):
            failures.append(Failure(path + ".operations",
                                    "operations must be a positive integer", operations))
            continue
        memory = sample.get("memory")
        if not isinstance(memory, dict):
            failures.append(Failure(path + ".memory",
                                    "memory observation is required", memory))
            continue
        valid_memory = True
        for field in MEMORY_FIELDS:
            value = memory.get(field)
            if (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                failures.append(Failure(path + ".memory." + field,
                                        "memory bytes must be a nonnegative integer",
                                        value))
                valid_memory = False
            else:
                memory_peaks[field] = max(memory_peaks[field], value)
        if not valid_memory:
            continue
        total_duration_ns += duration
        total_operations += operations
        durations.append(duration / operations / 1_000_000.0)
    if len(durations) != len(samples):
        return None
    quality = {"profile_sha256": profile_hash}
    return {
        "throughput": {
            **quality,
            "unit": workload.get("operation", "operation") + "/second",
            "value": total_operations * 1_000_000_000.0 / total_duration_ns,
            "total_operations": total_operations,
        },
        "timing": {
            **quality,
            "unit": "milliseconds/" + workload.get("operation", "operation"),
            "median": _nearest_rank(durations, 0.50),
            "p95": _nearest_rank(durations, 0.95),
            "sample_count": len(durations),
        },
        "memory": {
            **quality,
            "unit": "bytes",
            "peak": memory_peaks,
        },
    }


def evaluate(root: Path, profile: dict[str, Any], measurement: dict[str, Any],
             quality_result: dict[str, Any] | None, profile_hash: str,
             quality_result_hash: str | None) -> dict[str, Any]:
    failures: list[Failure] = []
    if measurement.get("schema") != MEASUREMENT_SCHEMA:
        failures.append(Failure("$.schema", "unsupported measurement schema",
                                measurement.get("schema"), MEASUREMENT_SCHEMA))
    if measurement.get("execution_status") != "completed":
        failures.append(Failure("$.execution_status",
                                "benchmark execution must be completed",
                                measurement.get("execution_status"), "completed"))
    link = measurement.get("quality")
    if not isinstance(link, dict):
        failures.append(Failure("$.quality", "quality evidence link is required", link))
        link = {}
    if link.get("profile_id") != profile.get("id"):
        failures.append(Failure("$.quality.profile_id", "quality profile mismatch",
                                link.get("profile_id"), profile.get("id")))
    if link.get("profile_sha256") != profile_hash:
        failures.append(Failure("$.quality.profile_sha256",
                                "quality profile hash mismatch",
                                link.get("profile_sha256"), profile_hash))
    if quality_result is None:
        failures.append(Failure("$.quality.result_sha256",
                                "quality result is missing"))
    elif link.get("result_sha256") != quality_result_hash:
        failures.append(Failure("$.quality.result_sha256",
                                "quality result hash mismatch",
                                link.get("result_sha256"), quality_result_hash))
    _metadata(measurement, quality_result, failures)
    before_metrics = len(failures)
    metrics = _measurements(measurement, profile_hash, failures)
    if len(failures) != before_metrics:
        metrics = None

    quality_report = None
    if quality_result is not None:
        quality = _quality_module(root)
        quality_report = quality.evaluate(profile, quality_result, profile_hash)
        for failure in quality_report["failures"]:
            failures.append(Failure("$.quality_result" + failure["path"][1:],
                                    failure["message"], failure["actual"],
                                    failure["limit"]))
    qualified = not failures and quality_report is not None and quality_report["passed"]
    return {
        "schema": REPORT_SCHEMA,
        "scope": "quality_gated_performance",
        "qualified": qualified,
        "score_status": "qualified" if qualified else "unqualified",
        "source": measurement.get("source"),
        "driver": measurement.get("driver"),
        "device": measurement.get("device"),
        "toolchain": measurement.get("toolchain"),
        "workload": measurement.get("workload"),
        "measurement_method": measurement.get("measurement_method"),
        "quality_profile_id": profile.get("id"),
        "quality_profile_sha256": profile_hash,
        "quality_result_sha256": quality_result_hash,
        "quality_passed": bool(quality_report and quality_report["passed"]),
        "metrics": metrics,
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path,
                        default=root / "conformance/v1/quality/core-conservation-v1.json")
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--quality-result", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    try:
        profile_data = arguments.profile.read_bytes()
        measurement = _json(arguments.measurement.read_text(encoding="utf-8"))
        quality_data = (arguments.quality_result.read_bytes()
                        if arguments.quality_result else None)
        quality_result = _json(quality_data) if quality_data is not None else None
        profile = _json(profile_data)
        if (not isinstance(profile, dict) or not isinstance(measurement, dict) or
                (quality_result is not None and not isinstance(quality_result, dict))):
            raise ValueError("profile, measurement and quality result must be objects")
        report = evaluate(root, profile, measurement, quality_result,
                          _sha256(profile_data),
                          _sha256(quality_data) if quality_data is not None else None)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"benchmark qualification: {error}")
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text, encoding="utf-8")
    print(f"Benchmark: {report['score_status']}, {report['failure_count']} failures")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
