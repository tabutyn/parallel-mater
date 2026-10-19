#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "rope-rigid-attachment-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_rope_rigid_attachment_quality", ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def add(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def scale(value: list[float], factor: float) -> list[float]:
    return [factor * component for component in value]


def cross(left: list[float], right: list[float]) -> list[float]:
    return [left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]]


def rotate(value: list[float], quaternion: list[float]) -> list[float]:
    vector = quaternion[:3]
    scalar = quaternion[3]
    dot = sum(a * b for a, b in zip(vector, value))
    squared = sum(component * component for component in vector)
    return add(add(scale(vector, 2.0 * dot),
                   scale(value, scalar * scalar - squared)),
               scale(cross(vector, value), 2.0 * scalar))


def rope_state(index: int, position: list[float], velocity: list[float],
               constraint: list[float] | None = None,
               contact: list[float] | None = None) -> dict:
    return {
        "index": index,
        "position_m": list(position),
        "velocity_m_s": list(velocity),
        "constraint_force_n": list(constraint or [0.0, 0.0, 0.0]),
        "contact_force_n": list(contact or [0.0, 0.0, 0.0]),
    }


def rigid_state(position: list[float], orientation: list[float],
                velocity: list[float], angular: list[float]) -> dict:
    return {
        "position_m": list(position),
        "orientation_xyzw": list(orientation),
        "linear_velocity_m_s": list(velocity),
        "angular_velocity_rad_s": list(angular),
    }


def valid_result(profile: dict) -> dict:
    definition = profile["fixture"]
    fixture = copy.deepcopy(definition)
    count = definition["rope_node_count"]
    start, end = definition["rope_centerline_m"]
    initial_positions = [[
        start[axis] + index / (count - 1) * (end[axis] - start[axis])
        for axis in range(3)
    ] for index in range(count)]
    zero = [0.0, 0.0, 0.0]
    initial_rope = [rope_state(index, position, zero)
                    for index, position in enumerate(initial_positions)]
    initial_rigid = rigid_state(
        definition["rigid_initial_position_m"],
        definition["rigid_initial_orientation_xyzw"],
        definition["rigid_initial_linear_velocity_m_s"],
        definition["rigid_initial_angular_velocity_rad_s"])
    fixture["initial"] = {
        "rope": copy.deepcopy(initial_rope),
        "rigid": copy.deepcopy(initial_rigid),
    }
    fixture["previous"] = {
        "rope": copy.deepcopy(initial_rope),
        "rigid": copy.deepcopy(initial_rigid),
    }

    timestep = definition["timestep_seconds"]
    attachment_impulse = [0.0, -0.1, 0.0]
    external_impulse = scale(definition["rigid_external_force_n"], timestep)
    body_impulse = add(external_impulse, attachment_impulse)
    final_velocity = scale(body_impulse, 1.0 / definition["rigid_mass_kg"])

    angle = -0.02
    final_orientation = [0.0, 0.0, math.sin(0.5 * angle),
                         math.cos(0.5 * angle)]
    previous_arm = definition["attachment_local_anchor_m"]
    final_arm = rotate(previous_arm, final_orientation)
    midpoint_arm = scale(add(previous_arm, final_arm), 0.5)
    angular_impulse = cross(midpoint_arm, attachment_impulse)
    final_angular = [
        angular_impulse[axis] /
        definition["rigid_inertia_diagonal_kg_m2"][axis]
        for axis in range(3)
    ]
    final_position = [0.0, 0.002, 0.0]
    final_anchor = add(final_position, final_arm)
    rest_length = math.dist(start, end) / (count - 1)
    final_positions = [
        [final_anchor[0], final_anchor[1] - rest_length * index,
         final_anchor[2]]
        for index in range(count)
    ]
    free_velocity = scale(attachment_impulse,
                          -1.0 / definition["free_rope_mass_kg"])
    final_rope = [
        rope_state(
            index, position, zero if index == 0 else free_velocity,
            scale(attachment_impulse, 1.0 / timestep) if index == 0 else zero)
        for index, position in enumerate(final_positions)
    ]
    fixture["final"] = {
        "rope": final_rope,
        "rigid": rigid_state(final_position, final_orientation,
                             final_velocity, final_angular),
    }
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "fixture": fixture,
    }


class RopeRigidAttachmentQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile_hash = QUALITY.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return QUALITY.evaluate(self.profile, result, self.profile_hash)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_off_center_reciprocal_attachment_passes(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["fixture_count"], 1)
        self.assertEqual(report["sample_count"], 1)
        self.assertAlmostEqual(
            report["observations"]["final_attachment_drift_m"], 0.0)
        self.assertAlmostEqual(
            report["observations"][
                "reciprocal_linear_impulse_residual_n_s"], 0.0)
        self.assertAlmostEqual(
            report["observations"]["force_impulse_residual_n_s"], 0.0)
        self.assertGreater(abs(report["observations"][
            "angular_reaction_impulse_n_m_s"][2]), 0.001)

    def test_endpoint_drift_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["rope"][0]["position_m"][0] += 0.01
        self.assert_rejected(result, "attachment drift exceeds")

    def test_nonreciprocal_free_rope_impulse_fails(self) -> None:
        result = valid_result(self.profile)
        for node in result["fixture"]["final"]["rope"][1:]:
            node["velocity_m_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "impulses are not reciprocal")

    def test_constraint_force_must_match_body_impulse(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["rope"][0][
            "constraint_force_n"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result,
                             "constraint force does not match rigid attachment")

    def test_off_center_angular_reaction_is_required(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["rigid"][
            "angular_velocity_rad_s"] = [0.0, 0.0, 0.0]
        self.assert_rejected(result, "angular reaction is too small")
        self.assert_rejected(result,
                             "angular attachment impulse is not reciprocal")

    def test_contact_cannot_substitute_for_attachment(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["rope"][2][
            "contact_force_n"] = [0.0, 1.0, 0.0]
        self.assert_rejected(result, "contact channel is active")

    def test_rope_length_and_speed_caps_are_enforced(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["final"]["rope"][3]["position_m"][1] += 0.01
        self.assert_rejected(result, "segment length differs from rest length")

        result = valid_result(self.profile)
        result["fixture"]["final"]["rope"][3][
            "velocity_m_s"] = [11.0, 0.0, 0.0]
        self.assert_rejected(result, "rope speed exceeds authored maximum")

    def test_fixture_contract_and_singular_shape_are_exact(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["two_way"] = False
        self.assert_rejected(result, "fixture setting differs from profile")

        result = valid_result(self.profile)
        result["fixtures"] = [result.pop("fixture")]
        self.assert_rejected(result, "expected one fixture object")


if __name__ == "__main__":
    unittest.main()
