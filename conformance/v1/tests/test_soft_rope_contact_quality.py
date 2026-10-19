#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
PROFILE_PATH = ROOT / "quality" / "soft-rope-contact-v1.json"
OBLIGATIONS_PATH = (ROOT / "qualification" /
                    "capability-obligations-v1.json")
SPEC = importlib.util.spec_from_file_location(
    "soft_rope_quality", ROOT / "quality.py")
assert SPEC and SPEC.loader
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def endpoint(index: int, position: list[float], velocity: list[float],
             force: list[float], force_field: str) -> dict:
    return {
        "index": index,
        "position_m": list(position),
        "velocity_m_s": list(velocity),
        force_field: list(force),
    }


def reconstruct(nodes: list[list[float]], bindings: list[dict]
                ) -> list[list[float]]:
    return [[sum(nodes[node][axis] * weight
                 for node, weight in zip(binding["nodes"],
                                         binding["weights"]))
             for axis in range(3)]
            for binding in bindings]


def phase(rope_positions: list[list[float]], rope_velocity: list[float],
          rope_force: list[float], soft_positions: list[list[float]],
          soft_velocities: list[list[float]], soft_force: list[float],
          bindings: list[dict]) -> dict:
    return {
        "rope": [endpoint(index, position, rope_velocity, rope_force,
                          "soft_body_contact_force_n")
                 for index, position in enumerate(rope_positions)],
        "soft_nodes": [endpoint(
            index, position, soft_velocities[index],
            soft_force if index == 0 else [0.0, 0.0, 0.0],
            "rope_contact_force_n")
            for index, position in enumerate(soft_positions)],
        "surface_m": reconstruct(soft_positions, bindings),
    }


def valid_result(profile: dict) -> dict:
    fixture = copy.deepcopy(profile["fixture"])
    bindings = fixture["surface_bindings"]
    initial_rope = [[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0]]
    contact_rope = [[value[0], -0.2, value[2]] for value in initial_rope]
    released_rope = [[value[0], -0.202, value[2]] for value in initial_rope]
    initial_soft = copy.deepcopy(fixture["prepared_soft_nodes_m"])
    contact_soft = copy.deepcopy(initial_soft)
    released_soft = copy.deepcopy(initial_soft)
    released_soft[0][1] += 0.003
    initial_soft_velocity = [[1.0, -1.0, 0.0] for _ in initial_soft]
    contact_soft_velocity = copy.deepcopy(initial_soft_velocity)
    contact_soft_velocity[0][1] = -0.7
    released_soft_velocity = copy.deepcopy(contact_soft_velocity)
    fixture["initial"] = phase(
        initial_rope, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        initial_soft, initial_soft_velocity, [0.0, 0.0, 0.0], bindings)
    fixture["contact"] = phase(
        contact_rope, [0.0, -0.2, 0.0], [0.0, -20.0, 0.0],
        contact_soft, contact_soft_velocity, [0.0, 60.0, 0.0], bindings)
    fixture["released"] = phase(
        released_rope, [0.0, -0.2, 0.0], [0.0, 0.0, 0.0],
        released_soft, released_soft_velocity, [0.0, 0.0, 0.0], bindings)
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "test-driver", "device": "test-device"},
        "fixture": fixture,
    }


class SoftRopeContactQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.digest = QUALITY.profile_sha256(PROFILE_PATH)

    def evaluate(self, result: dict) -> dict:
        return QUALITY.evaluate(self.profile, result, self.digest)

    def assert_rejected(self, result: dict, message: str) -> None:
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(message in failure["message"]
                            for failure in report["failures"]), report)

    def test_clearance_reciprocity_slip_and_release_pass(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["sample_count"], 7)
        self.assertAlmostEqual(report["observations"][
            "reciprocal_impulse_residual_n_s"], 0.0)
        self.assertAlmostEqual(report["observations"][
            "reciprocal_force_residual_n"], 0.0)
        self.assertGreater(report["observations"][
            "contact_frictionless_slip_m_s"], 0.5)
        self.assertEqual(report["observations"][
            "released_contact_force_activity_n"], 0.0)

    def test_contact_overlap_fails(self) -> None:
        result = valid_result(self.profile)
        for node in result["fixture"]["contact"]["rope"]:
            node["position_m"][1] = 0.0
        self.assert_rejected(result, "remained inside the soft-body contact envelope")

    def test_nonreciprocal_impulse_and_force_fail(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["contact"]["soft_nodes"][0][
            "velocity_m_s"][1] += 0.1
        self.assert_rejected(result, "impulses are not reciprocal")

        result = valid_result(self.profile)
        result["fixture"]["contact"]["soft_nodes"][0][
            "rope_contact_force_n"][1] += 1.0
        self.assert_rejected(result, "forces are not reciprocal")

    def test_hidden_attachment_fails(self) -> None:
        result = valid_result(self.profile)
        for phase_name in ("contact", "released"):
            for node in result["fixture"][phase_name]["soft_nodes"]:
                node["velocity_m_s"][0] = 0.0
        self.assert_rejected(result, "hidden attachment")

    def test_release_force_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["released"]["rope"][0][
            "soft_body_contact_force_n"][1] = 1.0
        self.assert_rejected(result, "persisted after graph removal")

    def test_surface_reconstruction_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["contact"]["surface_m"][4][0] += 0.01
        self.assert_rejected(
            result, "public soft surface does not match its node bindings")

    def test_changed_public_input_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixture"]["attach_first"] = True
        self.assert_rejected(result, "fixture setting differs from profile")

    def test_obligation_binds_only_contact_and_retains_lifecycle(self) -> None:
        obligations = json.loads(OBLIGATIONS_PATH.read_text(encoding="utf-8"))
        rules = {rule["id"]: rule for rule in obligations["rules"]}
        contact = rules["soft-body-rope-contact"]
        attachment = rules["soft-body-rope-attachment"]
        self.assertEqual(contact["match"]["mode"], ["CONTACT"])
        self.assertEqual(contact["required"]["analytic_invariants"],
                         ["quality:soft-rope-contact-v1"])
        expected_lifecycle = [
            "lifecycle:adverse-capacity-three-family-v1#three-family-partial-rollback"]
        self.assertEqual(contact["required"]["lifecycle_invariants"],
                         expected_lifecycle)
        self.assertEqual(attachment["match"]["mode"], ["ATTACHMENT"])
        self.assertNotIn("analytic_invariants", attachment["required"])
        self.assertEqual(attachment["required"]["lifecycle_invariants"],
                         expected_lifecycle)


if __name__ == "__main__":
    unittest.main()
