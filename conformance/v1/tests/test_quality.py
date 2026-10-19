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
ATTACHMENT_PROFILE_PATH = ROOT / "quality" / "rigid-attachment-v1.json"
SWEEP_PROFILE_PATH = ROOT / "quality" / "rigid-kinematic-sweep-v1.json"
FLUID_SELF_CONTACT_PROFILE_PATH = (
    ROOT / "quality" / "fluid-self-contact-v1.json")
FLUID_SELF_PRESSURE_PROFILE_PATH = (
    ROOT / "quality" / "fluid-self-pressure-v1.json")
FLUID_SELF_VISCOSITY_PROFILE_PATH = (
    ROOT / "quality" / "fluid-self-viscosity-v1.json")
ROPE_SELF_CONTACT_PROFILE_PATH = (
    ROOT / "quality" / "rope-self-contact-v1.json")
SMOKE_SELF_VISCOSITY_PROFILE_PATH = (
    ROOT / "quality" / "smoke-self-viscosity-v1.json")


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


class SmokeSelfViscosityQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(
            SMOKE_SELF_VISCOSITY_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(
            SMOKE_SELF_VISCOSITY_PROFILE_PATH)

    @staticmethod
    def particle(slot: int, velocity: list[float]) -> dict:
        return {
            "slot": slot,
            "position_m": [0.01 * slot, 0.0, 0.0],
            "velocity_m_s": velocity,
            "age_seconds": 0.25 + 0.01 * slot,
            "number_density": 1.0,
            "pressure_m_s2": 0.0,
            "vorticity_per_s": [0.0, 0.0, 0.0],
        }

    def make_run(self, identity: str, initial: list[list[float]],
                 final: list[list[float]]) -> dict:
        return {
            "id": identity,
            "initial_particles": [
                self.particle(slot, velocity)
                for slot, velocity in enumerate(initial)
            ],
            "final_particles": [
                self.particle(slot, velocity)
                for slot, velocity in enumerate(final)
            ],
        }

    def valid_result(self) -> dict:
        fixtures = []
        for definition in self.profile["fixtures"]:
            fixture = copy.deepcopy(definition)
            count = fixture["expected_particle_count"]
            if fixture["representation"] == "SMOKE_PARTICLE":
                control_initial = [[-1.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]]
                control_final = copy.deepcopy(control_initial)
                viscous_initial = copy.deepcopy(control_initial)
                viscous_final = [[-0.5, 0.0, 0.0],
                                 [0.5, 0.0, 0.0]]
            else:
                control_initial = [[0.5 * slot, 0.0, 0.0]
                                   for slot in range(count)]
                control_final = copy.deepcopy(control_initial)
                mean = 0.5 * (count - 1) / 2.0
                viscous_initial = copy.deepcopy(control_initial)
                viscous_final = [[mean + 0.1 * (slot - (count - 1) / 2.0),
                                  0.0, 0.0]
                                 for slot in range(count)]
            fixture["runs"] = [
                self.make_run("control", control_initial, control_final),
                self.make_run("viscous", viscous_initial, viscous_final),
            ]
            fixtures.append(fixture)
        return {
            "schema": quality.RESULT_SCHEMA,
            "profile_id": self.profile["id"],
            "profile_sha256": self.digest,
            "execution_status": "completed",
            "driver": {"name": "test-driver", "device": "test-device"},
            "fixtures": fixtures,
        }

    def evaluate(self, result: dict) -> dict:
        return quality.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_particle_and_grid_damping_pass(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 2)
        self.assertEqual(report["sample_count"], 2)
        for observation in report["observations"].values():
            self.assertGreater(
                observation[
                    "relative_specific_energy_reduction_fraction"], 0.01)

    def test_fixture_inputs_and_run_set_are_exact(self):
        result = self.valid_result()
        result["fixtures"][0]["particle_radius_m"] = 0.2
        self.assert_rejected(result, "fixture setting differs from profile")

        result = self.valid_result()
        result["fixtures"][0]["runs"].pop()
        self.assert_rejected(result, "expected exactly control and viscous runs")

    def test_missing_activity_and_damping_fail(self):
        result = self.valid_result()
        viscous = result["fixtures"][0]["runs"][1]
        for particle in viscous["initial_particles"]:
            particle["velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result,
                             "initial smoke relative motion is too small")

        result = self.valid_result()
        viscous = result["fixtures"][0]["runs"][1]
        viscous["final_particles"] = copy.deepcopy(
            result["fixtures"][0]["runs"][0]["final_particles"])
        self.assert_rejected(result,
                             "did not dissipate relative motion")

    def test_energy_growth_and_mean_velocity_residual_fail(self):
        result = self.valid_result()
        viscous = result["fixtures"][0]["runs"][1]
        viscous["final_particles"][0]["velocity_m_s"] = [-2.0, 0.0, 0.0]
        viscous["final_particles"][1]["velocity_m_s"] = [2.0, 0.0, 0.0]
        self.assert_rejected(result,
                             "increased specific kinetic energy")

        result = self.valid_result()
        viscous = result["fixtures"][0]["runs"][1]
        for particle in viscous["final_particles"]:
            particle["velocity_m_s"][1] += 0.01
        self.assert_rejected(result,
                             "unexplained mean-velocity change")

    def test_grid_control_delta_is_required(self):
        result = self.valid_result()
        viscous = result["fixtures"][1]["runs"][1]
        for particle in viscous["final_particles"]:
            particle["velocity_m_s"][1] += 0.2
        self.assert_rejected(result,
                             "unexplained mean-velocity change")

    def test_particle_count_slot_and_finite_state_are_required(self):
        result = self.valid_result()
        result["fixtures"][1]["runs"][0]["final_particles"].pop()
        self.assert_rejected(result, "sampled smoke slots changed")

        result = self.valid_result()
        result["fixtures"][0]["runs"][0]["initial_particles"][1]["slot"] = 0
        self.assert_rejected(result, "duplicate smoke slot")

        result = self.valid_result()
        result["fixtures"][0]["runs"][0]["initial_particles"][0][
            "velocity_m_s"][0] = float("nan")
        self.assert_rejected(result, "expected three finite components")


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


class FluidSelfContactQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(
            FLUID_SELF_CONTACT_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(FLUID_SELF_CONTACT_PROFILE_PATH)

    def valid_result(self) -> dict:
        fixture = copy.deepcopy(self.profile["fixture"])
        fixture["final_particles"] = [
            {
                "stable_id": 0,
                "position_m": [-0.105, 0.0, 0.0],
                "velocity_m_s": [-1.1, 0.0, 0.0],
            },
            {
                "stable_id": 1,
                "position_m": [0.105, 0.0, 0.0],
                "velocity_m_s": [1.1, 0.0, 0.0],
            },
        ]
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

    def test_public_state_exclusion_and_balance_pass(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 1)
        self.assertEqual(report["sample_count"], 1)
        self.assertGreaterEqual(report["observations"]["final_separation_m"],
                                0.19)
        self.assertEqual(report["observations"][
            "net_internal_impulse_kg_m_s"], 0.0)

    def test_fixture_inputs_and_stable_ids_are_exact(self):
        result = self.valid_result()
        result["fixture"]["particle_radius_m"] = 0.11
        self.assert_rejected(result, "fixture setting differs from profile")

        result = self.valid_result()
        result["fixture"]["initial_particles"][0]["position_m"][0] = -0.06
        self.assert_rejected(result, "initial public state differs from profile")

        result = self.valid_result()
        result["fixture"]["final_particles"][1]["stable_id"] = 0
        self.assert_rejected(result, "stable ID must be a unique integer")
        self.assert_rejected(result, "preserve stable IDs")

    def test_penetration_and_vacuous_response_fail(self):
        result = self.valid_result()
        result["fixture"]["final_particles"][0]["position_m"][0] = -0.08
        result["fixture"]["final_particles"][1]["position_m"][0] = 0.08
        self.assert_rejected(result, "violate exclusion distance")

        result = self.valid_result()
        for particle in result["fixture"]["final_particles"]:
            particle["velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "impulse activity is too small")
        self.assert_rejected(result, "response is not outward")

    def test_unbalanced_or_asymmetric_response_fails(self):
        result = self.valid_result()
        result["fixture"]["final_particles"][1]["velocity_m_s"][0] = 1.2
        self.assert_rejected(result, "net internal impulse exceeds profile")

        result = self.valid_result()
        result["fixture"]["final_particles"][1]["position_m"][1] = 0.001
        self.assert_rejected(result, "center-of-mass drift exceeds profile")
        self.assert_rejected(result, "transverse symmetry leakage")

    def test_crossing_nonfinite_and_incomplete_results_fail(self):
        result = self.valid_result()
        result["fixture"]["final_particles"][0]["position_m"][0] = 0.105
        result["fixture"]["final_particles"][1]["position_m"][0] = -0.105
        result["fixture"]["final_particles"][0]["velocity_m_s"][0] = 1.1
        result["fixture"]["final_particles"][1]["velocity_m_s"][0] = -1.1
        self.assert_rejected(result, "crossed stable-ID ordering")

        result = self.valid_result()
        result["fixture"]["final_particles"][0]["position_m"][0] = float("nan")
        self.assert_rejected(result, "expected three finite components")

        result = self.valid_result()
        result["execution_status"] = "failed"
        self.assert_rejected(result, "must be completed")


class FluidSelfPressureQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(
            FLUID_SELF_PRESSURE_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(FLUID_SELF_PRESSURE_PROFILE_PATH)

    def valid_result(self) -> dict:
        fixture = copy.deepcopy(self.profile["fixture"])
        fixture["final_particles"] = [
            {
                "stable_id": 0,
                "position_m": [-0.086, 0.0, 0.0],
                "velocity_m_s": [-0.9, 0.0, 0.0],
            },
            {
                "stable_id": 1,
                "position_m": [0.086, 0.0, 0.0],
                "velocity_m_s": [0.9, 0.0, 0.0],
            },
        ]
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

    def test_public_state_density_relaxation_and_balance_pass(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        observations = report["observations"]
        self.assertGreater(observations["initial_density_kg_m3"], 1000.0)
        self.assertLess(observations["final_density_relative_error"],
                        observations["initial_density_relative_error"])
        self.assertGreater(observations[
            "density_error_reduction_fraction"], 0.85)
        self.assertEqual(observations["net_internal_impulse_kg_m_s"], 0.0)

    def test_fixture_inputs_estimator_and_stable_ids_are_exact(self):
        result = self.valid_result()
        result["fixture"]["density_estimator"] = "driver-private-density"
        self.assert_rejected(result, "fixture setting does not match profile")

        result = self.valid_result()
        result["fixture"]["initial_particles"][0]["position_m"][0] = -0.06
        self.assert_rejected(result, "initial public state differs from profile")

        result = self.valid_result()
        result["fixture"]["final_particles"][1]["stable_id"] = 0
        self.assert_rejected(result, "stable ID must be a unique integer")

    def test_static_or_wrong_direction_density_response_fails(self):
        result = self.valid_result()
        result["fixture"]["final_particles"] = copy.deepcopy(
            result["fixture"]["initial_particles"])
        self.assert_rejected(result, "did not move density sufficiently")
        self.assert_rejected(result, "response is not outward")

        result = self.valid_result()
        result["fixture"]["final_particles"][0]["position_m"][0] = -0.045
        result["fixture"]["final_particles"][1]["position_m"][0] = 0.045
        self.assert_rejected(result, "final density remains too far from rest")
        self.assert_rejected(result, "did not move density sufficiently")

    def test_unbalanced_or_asymmetric_response_fails(self):
        result = self.valid_result()
        result["fixture"]["final_particles"][1]["velocity_m_s"][0] = 1.0
        self.assert_rejected(result, "net internal impulse exceeds profile")

        result = self.valid_result()
        result["fixture"]["final_particles"][1]["position_m"][1] = 0.001
        self.assert_rejected(result, "center-of-mass drift exceeds profile")
        self.assert_rejected(result, "transverse symmetry leakage")

    def test_nonfinite_and_incomplete_results_fail(self):
        result = self.valid_result()
        result["fixture"]["final_particles"][0]["position_m"][0] = float("nan")
        self.assert_rejected(result, "expected three finite components")

        result = self.valid_result()
        result["execution_status"] = "failed"
        self.assert_rejected(result, "must be completed")


class FluidSelfViscosityQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(
            FLUID_SELF_VISCOSITY_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(FLUID_SELF_VISCOSITY_PROFILE_PATH)

    def valid_result(self) -> dict:
        fixture = copy.deepcopy(self.profile["fixture"])
        fixture["final_particles"] = [
            {
                "stable_id": 0,
                "position_m": [-0.0925, 0.0, 0.0],
                "velocity_m_s": [-0.85, 0.0, 0.0],
            },
            {
                "stable_id": 1,
                "position_m": [0.0925, 0.0, 0.0],
                "velocity_m_s": [0.85, 0.0, 0.0],
            },
        ]
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

    def test_public_state_dissipation_and_balance_pass(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        observations = report["observations"]
        self.assertLess(observations["final_relative_speed_m_s"],
                        observations["initial_relative_speed_m_s"])
        self.assertLess(observations["final_kinetic_energy_j"],
                        observations["initial_kinetic_energy_j"])
        self.assertEqual(observations["net_internal_impulse_kg_m_s"], 0.0)

    def test_fixture_inputs_and_stable_ids_are_exact(self):
        result = self.valid_result()
        result["fixture"]["kinematic_viscosity_m2_s"] = 1.0
        self.assert_rejected(result, "fixture setting differs from profile")

        result = self.valid_result()
        result["fixture"]["initial_particles"][0]["velocity_m_s"][0] = -0.5
        self.assert_rejected(result, "initial public state differs from profile")

        result = self.valid_result()
        result["fixture"]["final_particles"][1]["stable_id"] = 0
        self.assert_rejected(result, "stable ID must be a unique integer")

    def test_static_or_energy_increasing_response_fails(self):
        result = self.valid_result()
        result["fixture"]["final_particles"] = copy.deepcopy(
            result["fixture"]["initial_particles"])
        self.assert_rejected(result, "did not dissipate relative motion")
        self.assert_rejected(result, "did not dissipate sufficient kinetic energy")

        result = self.valid_result()
        result["fixture"]["final_particles"][0]["velocity_m_s"][0] = -1.1
        result["fixture"]["final_particles"][1]["velocity_m_s"][0] = 1.1
        self.assert_rejected(result, "increased kinetic energy")

    def test_unbalanced_or_asymmetric_response_fails(self):
        result = self.valid_result()
        result["fixture"]["final_particles"][1]["velocity_m_s"][0] = 0.9
        self.assert_rejected(result, "net internal impulse exceeds profile")

        result = self.valid_result()
        result["fixture"]["final_particles"][1]["position_m"][1] = 0.001
        self.assert_rejected(result, "center-of-mass drift exceeds profile")
        self.assert_rejected(result, "transverse symmetry leakage")

    def test_nonfinite_and_incomplete_results_fail(self):
        result = self.valid_result()
        result["fixture"]["final_particles"][0]["velocity_m_s"][0] = float("nan")
        self.assert_rejected(result, "expected three finite components")

        result = self.valid_result()
        result["execution_status"] = "failed"
        self.assert_rejected(result, "must be completed")


class RopeSelfContactQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(
            ROPE_SELF_CONTACT_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(ROPE_SELF_CONTACT_PROFILE_PATH)

    @staticmethod
    def node(index: int, position: list[float],
             velocity: list[float]) -> dict:
        return {
            "index": index,
            "position_m": position,
            "velocity_m_s": velocity,
        }

    def valid_result(self) -> dict:
        fixture = copy.deepcopy(self.profile["fixture"])
        initial_positions = fixture.pop("initial_sampled_nodes_m")
        fixture["initial_nodes"] = [
            self.node(index, copy.deepcopy(position), [0.0, 0.0, 0.0])
            for index, position in enumerate(initial_positions)
        ]
        fixture["final_nodes"] = [
            self.node(0, [-0.05, 0.0, 0.0], [-5.0, 0.0, 0.0]),
            self.node(1, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]),
            self.node(2, [0.06, 0.091651514, 0.0], [0.0, 0.0, 0.0]),
            self.node(3, [0.07, 0.0, 0.0], [5.0, 0.0, 0.0]),
        ]
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

    def test_sampled_node_contact_and_topology_exclusion_pass(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 1)
        self.assertEqual(report["sample_count"], 1)
        observations = report["observations"]
        self.assertEqual(observations["target_separation_m"], 0.12)
        self.assertGreaterEqual(
            observations["minimum_eligible_final_separation_m"], 0.114)

    def test_fixture_inputs_and_sample_order_are_exact(self):
        result = self.valid_result()
        result["fixture"]["radius_m"] = 0.061
        self.assert_rejected(result, "fixture setting differs from profile")

        result = self.valid_result()
        result["fixture"]["initial_centerline_m"][0][0] = 0.01
        self.assert_rejected(result, "initial public input differs from profile")

        result = self.valid_result()
        result["fixture"]["initial_nodes"][1]["index"] = 2
        self.assert_rejected(result, "node order/index changed")

    def test_static_overlap_and_topology_overcorrection_fail(self):
        result = self.valid_result()
        result["fixture"]["final_nodes"] = copy.deepcopy(
            result["fixture"]["initial_nodes"])
        self.assert_rejected(result,
                             "nonadjacent sampled nodes remain inside")

        result = self.valid_result()
        result["fixture"]["final_nodes"][3]["position_m"] = [0.22, 0.0, 0.0]
        self.assert_rejected(result,
                             "topology-neighbor probe was separated")

    def test_asymmetry_nonfinite_and_incomplete_results_fail(self):
        result = self.valid_result()
        result["fixture"]["final_nodes"][2]["position_m"][2] = 0.001
        self.assert_rejected(result, "center-of-mass drift exceeds profile")
        self.assert_rejected(result, "transverse symmetry leakage")

        result = self.valid_result()
        result["fixture"]["final_nodes"][0]["position_m"][0] = float("nan")
        self.assert_rejected(result, "expected three finite components")

        result = self.valid_result()
        result["execution_status"] = "failed"
        self.assert_rejected(result, "must be completed")


class RigidAttachmentQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(
            ATTACHMENT_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(ATTACHMENT_PROFILE_PATH)

    def valid_result(self) -> dict:
        fixture = copy.deepcopy(self.profile["fixture"])
        fixture.update({
            "maximum_hinge_anchor_error_m": 0.00001,
            "maximum_hinge_axis_error": 0.00002,
            "maximum_slider_sideways_drift_m": 0.00003,
            "maximum_slider_axis_error": 0.00004,
            "maximum_slider_travel_m": 0.25,
            "constraints_enabled": True,
            "constraints_unbroken": True,
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

    def test_public_state_attachment_measurements_pass(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 1)
        self.assertEqual(report["sample_count"], 5)

    def test_each_existing_attachment_limit_is_enforced(self):
        for field in ("maximum_hinge_anchor_error_m",
                      "maximum_hinge_axis_error",
                      "maximum_slider_sideways_drift_m",
                      "maximum_slider_axis_error",
                      "maximum_slider_travel_m"):
            with self.subTest(field=field):
                result = self.valid_result()
                result["fixture"][field] = self.profile["tolerances"][field]
                self.assert_rejected(result, "reaches or exceeds profile")

        result = self.valid_result()
        result["fixture"]["maximum_slider_travel_m"] = 0.01
        self.assert_rejected(result, "does not exceed profile minimum")

    def test_fixture_identity_and_constraint_state_are_required(self):
        result = self.valid_result()
        result["fixture"]["scene_sha256"] = "0" * 64
        self.assert_rejected(result, "fixture setting does not match profile")

        for field in ("constraints_enabled", "constraints_unbroken"):
            with self.subTest(field=field):
                result = self.valid_result()
                result["fixture"][field] = False
                self.assert_rejected(result,
                                     "all authored constraints must remain active")

    def test_failed_or_nonfinite_measurement_cannot_pass(self):
        result = self.valid_result()
        result["execution_status"] = "failed"
        self.assert_rejected(result, "must be completed")

        result = self.valid_result()
        result["fixture"]["maximum_hinge_axis_error"] = float("nan")
        self.assert_rejected(result, "expected a finite number")


class RigidKinematicSweepQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(SWEEP_PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = quality.profile_sha256(SWEEP_PROFILE_PATH)

    def valid_result(self) -> dict:
        checkpoints = [
            {
                "frame": frame,
                "resources": [],
                "contacts": {"rigid_contact_count": 0, "rigid_contacts": []},
            }
            for frame in range(61)
        ]
        checkpoints[20]["resources"] = [
            {
                "id": "dynamic-box",
                "position": [0.25, 1.448, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [0.75, -3.27, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            },
            {
                "id": "kinematic-box",
                "position": [-1.0, 1.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            },
        ]
        checkpoints[21]["resources"] = [
            {
                "id": "dynamic-box",
                "position": [1.448, 1.733, 0.0],
                "orientation": [0.0, 0.0, 0.089, 0.996],
                "linear_velocity": [97.67, 21.47, 0.0],
                "angular_velocity": [0.0, 0.0, 16.53],
            },
            {
                "id": "kinematic-box",
                "position": [1.0, 1.5, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [120.0, 30.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            },
        ]
        contacts = [
            {
                "body": "dynamic-box",
                "collider": "kinematic-box",
                "position": [1.152, 1.689, z],
                "normal": [0.976187, 0.216930, 0.0],
                "normal_impulse": 24.0,
                "friction_impulse": [-0.21, 0.96, 0.0],
                "penetration": 0.099,
            }
            for z in (-0.2, 0.2)
        ]
        checkpoints[21]["contacts"] = {
            "rigid_contact_count": len(contacts),
            "rigid_contacts": contacts,
        }
        return {
            "schema": quality.CONFORMANCE_RESULT_SCHEMA,
            "backend": "slang-test",
            "case_id": "rigid-direct",
            "case_sha256": self.profile["fixture"]["case_sha256"],
            "execution_status": "completed",
            "checkpoints": checkpoints,
            "provenance": {"device_name": "test-device"},
        }

    def evaluate(self, result: dict) -> dict:
        return quality.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_off_center_sweep_oracle_passes_without_reference(self):
        report = self.evaluate(self.valid_result())
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 61)
        self.assertGreater(report["observations"]["normal_alignment_cosine"],
                           0.99)
        adjudication = self.profile["reference_adjudication"]
        self.assertFalse(adjudication["direct_tolerances_changed"])
        self.assertEqual(adjudication["golden_replacement"], "required")

    def test_case_identity_and_all_frames_are_required(self):
        result = self.valid_result()
        result["case_sha256"] = "0" * 64
        self.assert_rejected(result, "case hash mismatch")

        result = self.valid_result()
        result["checkpoints"].pop(10)
        self.assert_rejected(result, "every frame")

    def test_historical_nonplanar_wrong_axis_response_fails(self):
        result = self.valid_result()
        body = result["checkpoints"][21]["resources"][0]
        body["orientation"] = [-0.04, -0.007, 0.036, 0.998]
        body["angular_velocity"] = [-0.023, -0.004, -0.009]
        for contact in result["checkpoints"][21]["contacts"]["rigid_contacts"]:
            contact["normal"] = [0.0, -1.0, 0.0]
        self.assert_rejected(result, "not aligned with the kinematic sweep")
        self.assert_rejected(result, "lacks the required angular response")
        self.assert_rejected(result, "planar symmetry leakage")

    def test_declared_collider_bounds_are_independently_enforced(self):
        result = self.valid_result()
        result["checkpoints"][21]["contacts"]["rigid_contacts"][0][
            "position"][2] = 0.25
        self.assert_rejected(result, "outside the declared collider")

    def test_coulomb_cone_and_nonvacuous_response_are_required(self):
        result = self.valid_result()
        contact = result["checkpoints"][21]["contacts"]["rigid_contacts"][0]
        contact["friction_impulse"] = [20.0, 0.0, 0.0]
        self.assert_rejected(result, "exceeds Coulomb cone")

        result = self.valid_result()
        body = result["checkpoints"][21]["resources"][0]
        body["linear_velocity"] = [0.0, 0.0, 0.0]
        body["angular_velocity"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "did not respond along")
        self.assert_rejected(result, "lacks the required angular response")

    def test_incomplete_execution_and_backend_identity_fail(self):
        result = self.valid_result()
        result["execution_status"] = "failed"
        result["backend"] = ""
        self.assert_rejected(result, "must be completed")
        self.assert_rejected(result, "backend identity is required")


if __name__ == "__main__":
    unittest.main()
