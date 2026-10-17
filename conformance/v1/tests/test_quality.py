# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "quality.py"
SPEC = importlib.util.spec_from_file_location("conformance_quality", MODULE_PATH)
assert SPEC and SPEC.loader
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)
PROFILE_PATH = ROOT / "quality" / "core-conservation-v1.json"
BRICK_PROFILE_PATH = ROOT / "quality" / "rigid-brick-penetration-v1.json"


def particle(identity: int) -> dict:
    return {
        "id": identity,
        "position": [0.1 * identity, 0.0, 0.0],
        "velocity": [0.25 - 0.1 * identity, 0.0, 0.0],
        "acceleration": [0.5 - 0.2 * identity, 0.0, 0.0],
        "temperature_celsius": 20.0,
        "foam": 0.0,
    }


def sample(timestep: float, ratio: float) -> dict:
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


def valid_result(profile: dict) -> dict:
    samples = [sample(timestep, ratio)
               for timestep in profile["sweep"]["timesteps_seconds"]
               for ratio in profile["sweep"]["endpoint_mass_ratios"]]
    return {
        "schema": quality.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": quality.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "fixtures": [
            {"id": fixture["id"], "samples": copy.deepcopy(samples)}
            for fixture in profile["fixtures"]
        ],
    }


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return quality.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_complete_vendor_neutral_matrix_passes(self):
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 2)
        self.assertEqual(report["sample_count"], 18)
        self.assertEqual(report["driver"]["name"], "test-driver")

    def test_profile_hash_is_checkout_line_ending_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            lf.write_bytes(b'{\n  "value": 1\n}\n')
            crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
            self.assertEqual(quality.profile_sha256(lf),
                             quality.profile_sha256(crlf))

    def test_closed_fixture_cannot_hide_errors_as_external_inputs(self):
        result = valid_result(self.profile)
        for fixture in result["fixtures"]:
            point = fixture["samples"][0]
            first = point["systems"][0]
            first["linear_momentum_after"][0] += 0.2
            first["external_impulse"][0] = 0.2
            first["angular_momentum_after"][2] += 0.3
            first["external_angular_impulse"][2] = 0.3
            first["kinetic_energy_after"] += 0.4
            first["external_work"] = 0.4
        self.assert_rejected(result, "closed fixture requires zero external")

    def test_wrong_linear_momentum_fails(self):
        result = valid_result(self.profile)
        result["fixtures"][0]["samples"][0]["systems"][0][
            "linear_momentum_after"][0] += 0.01
        self.assert_rejected(result, "linear momentum residual")

    def test_wrong_angular_momentum_fails(self):
        result = valid_result(self.profile)
        result["fixtures"][0]["samples"][0]["systems"][0][
            "angular_momentum_after"][2] += 0.01
        self.assert_rejected(result, "angular momentum residual")

    def test_vacuous_zero_exchange_fails(self):
        result = valid_result(self.profile)
        systems = result["fixtures"][0]["samples"][0]["systems"]
        for system in systems:
            system["linear_momentum_after"] = copy.deepcopy(
                system["linear_momentum_before"])
        self.assert_rejected(result, "too small to be a conservation fixture")

    def test_energy_injection_fails_dissipative_fixture(self):
        result = valid_result(self.profile)
        result["fixtures"][1]["samples"][0]["systems"][0][
            "kinetic_energy_after"] = 2.0
        self.assert_rejected(result, "increased accounted kinetic energy")

    def test_wrong_split_position_velocity_and_acceleration_fail(self):
        fields = ("position", "velocity", "acceleration")
        for field in fields:
            with self.subTest(field=field):
                result = valid_result(self.profile)
                point = result["fixtures"][0]["samples"][0]
                scale = {"position": 0.1, "velocity": 100.0,
                         "acceleration": 100000.0}[field]
                point["split_combined"]["split"][0][field][0] += scale * 0.001
                self.assert_rejected(result,
                                     "split/combined normalized error")

    def test_wrong_temperature_and_foam_fail(self):
        result = valid_result(self.profile)
        result["fixtures"][0]["samples"][0]["split_combined"][
            "split"][0]["temperature_celsius"] += 0.1
        self.assert_rejected(result, "temperature must be exact")

        result = valid_result(self.profile)
        result["fixtures"][0]["samples"][0]["split_combined"][
            "split"][0]["foam"] += 0.001
        self.assert_rejected(result, "foam error exceeds profile")

    def test_split_combined_identity_or_count_change_fails(self):
        result = valid_result(self.profile)
        result["fixtures"][0]["samples"][0]["split_combined"][
            "combined"].pop()
        self.assert_rejected(result, "particle identities differ")

    def test_missing_sweep_point_and_false_mass_ratio_fail(self):
        result = valid_result(self.profile)
        result["fixtures"][0]["samples"].pop()
        self.assert_rejected(result, "missing timestep/mass-ratio point")

        result = valid_result(self.profile)
        result["fixtures"][0]["samples"][0]["systems"][1]["mass"] = 3.0
        self.assert_rejected(result, "does not match endpoint masses")

    def test_profile_hash_and_incomplete_execution_fail(self):
        result = valid_result(self.profile)
        result["profile_sha256"] = "0" * 64
        self.assert_rejected(result, "profile hash mismatch")

        result = valid_result(self.profile)
        result["execution_status"] = "unsupported"
        self.assert_rejected(result, "must be completed")

    def test_nonfinite_state_and_missing_driver_fail(self):
        result = valid_result(self.profile)
        result["fixtures"][0]["samples"][0]["systems"][0]["mass"] = float("nan")
        self.assert_rejected(result, "expected a finite number")

        result = valid_result(self.profile)
        result.pop("driver")
        self.assert_rejected(result, "driver name and device are required")


class RigidPenetrationQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(BRICK_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(BRICK_PROFILE_PATH)

    def valid_result(self) -> dict:
        fixture = copy.deepcopy(self.profile["fixture"])
        fixture.update({
            "maximum_penetration_m": 0.000205278,
            "maximum_penetration_frame": 187,
        })
        return {
            "schema": quality.RESULT_SCHEMA,
            "profile_id": self.profile["id"],
            "profile_sha256": self.digest,
            "execution_status": "completed",
            "driver": {"name": "test-driver", "device": "test-device"},
            "fixture": fixture,
        }

    def evaluate(self, result: dict) -> dict:
        return quality.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_independent_brick_measurement_passes(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 1)
        self.assertEqual(report["sample_count"], 1)

    def test_excessive_or_negative_penetration_fails(self):
        result = self.valid_result()
        result["fixture"]["maximum_penetration_m"] = 0.0020001
        self.assert_rejected(result, "maximum penetration exceeds profile")

        result = self.valid_result()
        result["fixture"]["maximum_penetration_m"] = -0.001
        self.assert_rejected(result, "penetration cannot be negative")

    def test_fixture_configuration_and_frame_are_exact(self):
        for field, value in (("body_count", 7), ("frames", 299),
                             ("substeps", 1), ("scene", "Other.glb"),
                             ("oracle", "driver-contacts")):
            with self.subTest(field=field):
                result = self.valid_result()
                result["fixture"][field] = value
                self.assert_rejected(result,
                                     "fixture setting does not match profile")

        result = self.valid_result()
        result["fixture"]["maximum_penetration_frame"] = 300
        self.assert_rejected(result, "outside the run")

    def test_failed_or_nonfinite_measurement_cannot_pass(self):
        result = self.valid_result()
        result["execution_status"] = "failed"
        self.assert_rejected(result, "must be completed")

        result = self.valid_result()
        result["fixture"]["maximum_penetration_m"] = float("nan")
        self.assert_rejected(result, "expected a finite number")


if __name__ == "__main__":
    unittest.main()
