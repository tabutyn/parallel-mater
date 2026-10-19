#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "cloth-rigid-contact-v1.json"
OBLIGATIONS_PATH = ROOT / "qualification" / "capability-obligations-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_cloth_rigid_quality",
                                              ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def cloth_state(index: int, position: list[float], velocity: list[float],
                force: list[float]) -> dict:
    return {
        "index": index,
        "position_m": list(position),
        "velocity_m_s": list(velocity),
        "rigid_contact_force_n": list(force),
    }


def rigid_state(position: list[float], velocity: list[float]) -> dict:
    return {
        "position_m": list(position),
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity_m_s": list(velocity),
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }


def valid_result(profile: dict) -> dict:
    fixture = copy.deepcopy(profile["fixture"])
    prepared = fixture["prepared_cloth_vertices_m"]
    initial_velocity = fixture["prepared_cloth_velocity_m_s"]
    impulse = 2.0 / 3.0
    initial_cloth = [
        cloth_state(index, position, initial_velocity, [0.0, 0.0, 0.0])
        for index, position in enumerate(prepared)
    ]
    final_cloth = copy.deepcopy(initial_cloth)
    final_cloth[0] = cloth_state(
        0, [-0.2, 0.0, 0.0], [1.0 - impulse, 1.0, 0.0],
        [-impulse / fixture["timestep_seconds"], 0.0, 0.0])
    fixture["initial"] = {
        "cloth": initial_cloth,
        "rigid": rigid_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    }
    fixture["final"] = {
        "cloth": final_cloth,
        "rigid": rigid_state([0.05, 0.0, 0.0], [impulse / 2.0, 0.0, 0.0]),
    }
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "fixture": fixture,
    }


class ClothRigidContactQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = QUALITY.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return QUALITY.evaluate(self.profile, result, self.profile_hash)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in item["message"]
                            for item in report["failures"]), report)

    def test_clearance_reciprocity_force_and_slip_pass(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 3)
        self.assertAlmostEqual(report["observations"][
            "expected_normal_impulse_n_s"], 2.0 / 3.0)
        self.assertAlmostEqual(report["observations"][
            "reciprocal_impulse_residual_n_s"], 0.0)
        self.assertGreater(report["observations"][
            "final_frictionless_slip_m_s"], 0.5)

    def test_surface_overlap_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["cloth"][0]["position_m"][0] = 0.0
        self.assert_rejected(result, "overlapped the rigid contact envelope")

    def test_nonreciprocal_response_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["rigid"][
            "linear_velocity_m_s"][0] = 0.2
        self.assert_rejected(result, "impulses are not reciprocal")

    def test_wrong_reduced_mass_impulse_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["cloth"][0][
            "velocity_m_s"][0] = 0.4
        result["fixture"]["final"]["rigid"][
            "linear_velocity_m_s"][0] = 0.3
        result["fixture"]["final"]["cloth"][0][
            "rigid_contact_force_n"][0] = -60.0
        self.assert_rejected(result, "reduced-mass oracle")

    def test_public_force_must_match_impulse(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["cloth"][0][
            "rigid_contact_force_n"][0] = -50.0
        self.assert_rejected(result, "contact force does not integrate")

    def test_hidden_attachment_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["cloth"][0][
            "velocity_m_s"][1] = 0.0
        self.assert_rejected(result, "moved as a hidden attachment")

    def test_public_input_is_immutable(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["cloth_thickness_m"] = 0.02
        self.assert_rejected(result, "fixture setting differs from profile")

    def test_exact_obligation_is_registered(self) -> None:
        obligations = json.loads(
            OBLIGATIONS_PATH.read_text(encoding="utf-8"))
        rule = next(item for item in obligations["rules"]
                    if item["id"] == "cloth-rigid")
        self.assertEqual(rule["match"], {
            "first_family": ["CLOTH"],
            "second_family": ["RIGID_BODY"],
            "scope": ["DISTINCT_FAMILY"],
            "mode": ["CONTACT"],
            "direction": ["BIDIRECTIONAL"],
            "material_profile": ["ENDPOINT_NATIVE"],
        })
        self.assertEqual(rule["required"], {
            "lifecycle_invariants": [
                "lifecycle:cloth-rigid-contact-lifecycle-v1#"
                "cloth-rigid-contact-capacity-replacement-removal-rollback"],
            "analytic_invariants": ["quality:cloth-rigid-contact-v1"],
        })
        self.assertEqual(rule["missing"], {})


if __name__ == "__main__":
    unittest.main()
