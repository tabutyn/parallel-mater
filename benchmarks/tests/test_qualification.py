# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import unittest


REPOSITORY = pathlib.Path(__file__).parents[2]
MODULE_PATH = REPOSITORY / "benchmarks" / "qualification.py"
SPEC = importlib.util.spec_from_file_location("benchmark_qualification", MODULE_PATH)
assert SPEC and SPEC.loader
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)
PROFILE_PATH = (REPOSITORY / "conformance/v1/quality" /
                "core-conservation-v1.json")
SOURCE = {"commit": "a" * 40, "dirty": False}
DRIVER = {"name": "test-driver", "version": "1.2.3",
          "binary_sha256": hashlib.sha256(b"driver").hexdigest()}
DEVICE = {"name": "test-device", "vendor_id": 1, "device_id": 2,
          "driver_version": "3.4.5"}
TOOLCHAIN = {"compiler": "test-compiler", "version": "6.7.8"}


def particle(identity: int) -> dict:
    return {
        "id": identity,
        "position": [0.1 * identity, 0.0, 0.0],
        "velocity": [0.25 - 0.1 * identity, 0.0, 0.0],
        "acceleration": [0.5 - 0.2 * identity, 0.0, 0.0],
        "temperature_celsius": 20.0,
        "foam": 0.0,
    }


def quality_sample(timestep: float, ratio: float) -> dict:
    systems = [
        {
            "mass": 1.0,
            "linear_momentum_before": [0.5, 0.0, 0.0],
            "linear_momentum_after": [0.4, 0.05, 0.0],
            "external_impulse": [0.0, 0.0, 0.0],
            "angular_momentum_before": [0.0, 0.0, 0.2],
            "angular_momentum_after": [0.0, 0.0, 0.1],
            "external_angular_impulse": [0.0, 0.0, 0.0],
            "kinetic_energy_before": 1.0,
            "kinetic_energy_after": 0.8,
            "external_work": 0.0,
        },
        {
            "mass": ratio,
            "linear_momentum_before": [-0.5, 0.0, 0.0],
            "linear_momentum_after": [-0.4, -0.05, 0.0],
            "external_impulse": [0.0, 0.0, 0.0],
            "angular_momentum_before": [0.0, 0.0, -0.2],
            "angular_momentum_after": [0.0, 0.0, -0.1],
            "external_angular_impulse": [0.0, 0.0, 0.0],
            "kinetic_energy_before": ratio,
            "kinetic_energy_after": 0.8 * ratio,
            "external_work": 0.0,
        },
    ]
    states = [particle(0), particle(1)]
    return {
        "timestep_seconds": timestep,
        "mass_ratio": ratio,
        "systems": systems,
        "split_combined": {
            "support_radius_m": 0.1,
            "split": copy.deepcopy(states),
            "combined": copy.deepcopy(states),
        },
    }


def valid_quality_result(profile: dict, profile_hash: str) -> dict:
    samples = [quality_sample(timestep, ratio)
               for timestep in profile["sweep"]["timesteps_seconds"]
               for ratio in profile["sweep"]["endpoint_mass_ratios"]]
    return {
        "schema": "parallel-mater-conformance-quality-result/v1",
        "profile_id": profile["id"],
        "profile_sha256": profile_hash,
        "execution_status": "completed",
        "source": copy.deepcopy(SOURCE),
        "driver": {"name": DRIVER["name"], "device": DEVICE["name"],
                   "version": DRIVER["version"],
                   "binary_sha256": DRIVER["binary_sha256"]},
        "toolchain": copy.deepcopy(TOOLCHAIN),
        "fixtures": [{"id": fixture["id"], "samples": copy.deepcopy(samples)}
                     for fixture in profile["fixtures"]],
    }


def result_bytes(result: dict) -> bytes:
    return (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()


def valid_measurement(profile: dict, profile_hash: str,
                      quality_hash: str) -> dict:
    return {
        "schema": qualification.MEASUREMENT_SCHEMA,
        "execution_status": "completed",
        "source": copy.deepcopy(SOURCE),
        "driver": copy.deepcopy(DRIVER),
        "device": copy.deepcopy(DEVICE),
        "toolchain": copy.deepcopy(TOOLCHAIN),
        "quality": {"profile_id": profile["id"],
                    "profile_sha256": profile_hash,
                    "result_sha256": quality_hash},
        "workload": {"id": "fluid-pair-step", "revision_sha256": "b" * 64,
                     "operation": "simulation-step"},
        "measurement_method": {
            "timing": "host_monotonic",
            "memory": ["pmGetDeviceMemoryBudget",
                       "PmStepStatistics.scratchHighWaterBytes"],
        },
        "samples": [
            {
                "duration_ns": 1_000_000 + index * 10_000,
                "operations": 10,
                "memory": {
                    "application_allocation_bytes": 1000,
                    "library_device_bytes": 2000 + index,
                    "library_host_bytes": 3000,
                    "borrowed_buffer_bytes": 4000,
                    "scratch_high_water_bytes": 5000 + index * 2,
                },
            }
            for index in range(20)
        ],
    }


class BenchmarkQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()

    def evaluate(self, measurement: dict, quality_result: dict | None,
                 quality_hash: str | None = None) -> dict:
        if quality_result is not None and quality_hash is None:
            quality_hash = hashlib.sha256(result_bytes(quality_result)).hexdigest()
        return qualification.evaluate(
            REPOSITORY, self.profile, measurement, quality_result,
            self.profile_hash, quality_hash)

    def pair(self) -> tuple[dict, dict, str]:
        quality_result = valid_quality_result(self.profile, self.profile_hash)
        quality_hash = hashlib.sha256(result_bytes(quality_result)).hexdigest()
        return (valid_measurement(self.profile, self.profile_hash, quality_hash),
                quality_result, quality_hash)

    def test_passing_quality_emits_hashed_metrics(self):
        measurement, quality_result, quality_hash = self.pair()
        report = self.evaluate(measurement, quality_result, quality_hash)
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["score_status"], "qualified")
        self.assertEqual(report["quality_result_sha256"], quality_hash)
        self.assertAlmostEqual(report["metrics"]["timing"]["median"], 0.109)
        self.assertAlmostEqual(report["metrics"]["timing"]["p95"], 0.118)
        self.assertAlmostEqual(report["metrics"]["throughput"]["value"],
                               200e9 / 21_900_000)
        self.assertEqual(report["metrics"]["memory"]["peak"][
            "scratch_high_water_bytes"], 5038)
        for metric in report["metrics"].values():
            self.assertEqual(metric["profile_sha256"], self.profile_hash)

    def test_failed_missing_and_unsupported_quality_block_scores(self):
        measurement, quality_result, quality_hash = self.pair()
        failed = copy.deepcopy(quality_result)
        failed["fixtures"][0]["samples"][0]["systems"][0][
            "linear_momentum_after"][0] += 0.1
        failed_hash = hashlib.sha256(result_bytes(failed)).hexdigest()
        failed_measurement = copy.deepcopy(measurement)
        failed_measurement["quality"]["result_sha256"] = failed_hash
        report = self.evaluate(failed_measurement, failed, failed_hash)
        self.assertFalse(report["qualified"])
        self.assertFalse(report["quality_passed"])

        report = self.evaluate(measurement, None, None)
        self.assertFalse(report["qualified"])
        self.assertTrue(any("quality result is missing" in failure["message"]
                            for failure in report["failures"]))

        unsupported = copy.deepcopy(quality_result)
        unsupported["execution_status"] = "unsupported"
        unsupported_hash = hashlib.sha256(result_bytes(unsupported)).hexdigest()
        unsupported_measurement = copy.deepcopy(measurement)
        unsupported_measurement["quality"]["result_sha256"] = unsupported_hash
        report = self.evaluate(unsupported_measurement, unsupported,
                               unsupported_hash)
        self.assertFalse(report["qualified"])
        self.assertTrue(any("must be completed" in failure["message"]
                            for failure in report["failures"]))

    def test_stale_profile_result_and_provenance_block_scores(self):
        measurement, quality_result, quality_hash = self.pair()
        for mutate, message in (
            (lambda item: item["quality"].__setitem__("profile_sha256", "0" * 64),
             "profile hash mismatch"),
            (lambda item: item["quality"].__setitem__("result_sha256", "0" * 64),
             "result hash mismatch"),
        ):
            candidate = copy.deepcopy(measurement)
            mutate(candidate)
            report = self.evaluate(candidate, quality_result, quality_hash)
            self.assertFalse(report["qualified"])
            self.assertTrue(any(message in failure["message"]
                                for failure in report["failures"]))
        mismatches = (
            (lambda item: item["source"].__setitem__("commit", "c" * 40),
             "quality source does not match"),
            (lambda item: item["driver"].__setitem__("device", "other-device"),
             "quality driver/device does not match"),
            (lambda item: item["toolchain"].__setitem__("version", "other"),
             "quality toolchain does not match"),
        )
        for mutate, message in mismatches:
            changed = copy.deepcopy(quality_result)
            mutate(changed)
            changed_hash = hashlib.sha256(result_bytes(changed)).hexdigest()
            changed_measurement = copy.deepcopy(measurement)
            changed_measurement["quality"]["result_sha256"] = changed_hash
            report = self.evaluate(changed_measurement, changed, changed_hash)
            self.assertFalse(report["qualified"])
            self.assertTrue(any(message in failure["message"]
                                for failure in report["failures"]))

    def test_incomplete_timing_or_memory_cannot_produce_metrics(self):
        measurement, quality_result, quality_hash = self.pair()
        measurement["samples"].pop()
        report = self.evaluate(measurement, quality_result, quality_hash)
        self.assertFalse(report["qualified"])
        self.assertIsNone(report["metrics"])

        measurement, quality_result, quality_hash = self.pair()
        del measurement["samples"][0]["memory"]["library_device_bytes"]
        report = self.evaluate(measurement, quality_result, quality_hash)
        self.assertFalse(report["qualified"])
        self.assertIsNone(report["metrics"])


if __name__ == "__main__":
    unittest.main()
