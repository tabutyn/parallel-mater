#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "quality" / "soft-rope-attachment-v1.json"
SPEC = importlib.util.spec_from_file_location(
    "pm_soft_rope_attachment_quality", ROOT / "quality.py")
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def soft_state(index: int, position: list[float],
               reaction: list[float]) -> dict:
    return {"index": index, "position_m": position,
            "velocity_m_s": [0.0, 0.0, 0.0],
            "rope_reaction_force_n": reaction}


def rope_state(index: int, position: list[float],
               contact: list[float] | None = None) -> dict:
    return {"index": index, "position_m": position,
            "velocity_m_s": [0.0, 0.0, 0.0],
            "soft_body_contact_force_n": contact or [0.0, 0.0, 0.0]}


def valid_result(profile: dict) -> dict:
    fixture = copy.deepcopy(profile["fixtures"][0])
    initial_soft = [list(position) for position in fixture["soft_nodes_m"]]
    initial_rope = [[0.0, -0.05 * index, -1.0 / 30.0]
                    for index in range(5)]
    fixture["initial"] = {
        "soft_body": [soft_state(index, position, [0.0, 0.0, 0.0])
                      for index, position in enumerate(initial_soft)],
        "soft_body_surface_m": copy.deepcopy(initial_soft),
        "rope": [rope_state(index, position)
                 for index, position in enumerate(initial_rope)],
    }
    final_soft = copy.deepcopy(initial_soft)
    for position in final_soft[:3]:
        position[1] = -0.01
    fixture["final"] = {
        "soft_body": [
            *[soft_state(index, position, [0.0, -1.308, 0.0])
              for index, position in enumerate(final_soft[:3])],
            soft_state(3, final_soft[3], [0.0, 0.0, 0.0]),
        ],
        "soft_body_surface_m": copy.deepcopy(final_soft),
        "rope": [
            rope_state(0, [0.0, -0.01, -1.0 / 30.0]),
            *[rope_state(index, position)
              for index, position in enumerate(initial_rope[1:], start=1)],
        ],
    }
    fixture["previous"] = copy.deepcopy(fixture["final"])
    return {
        "schema": QUALITY.RESULT_SCHEMA,
        "profile_id": profile["id"],
        "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
        "execution_status": "completed",
        "driver": {"name": "slang-test", "device": "test-device"},
        "fixtures": [fixture],
    }


class SoftRopeAttachmentQualityTests(unittest.TestCase):
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

    def test_bounded_reciprocal_attachment_passes(self) -> None:
        report = self.evaluate(valid_result(self.profile))
        self.assertTrue(report["passed"], report)
        observation = report["observations"]["supported-hanging-rope"]
        self.assertEqual(observation["final_attachment_drift_m"], 0.0)
        self.assertAlmostEqual(observation["reciprocal_force_residual_n"], 0.0)
        self.assertEqual(observation["maximum_contact_force_n"], 0.0)

    def test_endpoint_drift_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rope"][0]["position_m"][0] = 0.01
        self.assert_rejected(result, "attachment drift exceeds")

    def test_nonreciprocal_reaction_fails(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["soft_body"][0][
            "rope_reaction_force_n"][1] = -2.0
        self.assert_rejected(result, "reactions are not reciprocal")

    def test_pinned_support_must_not_move(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["soft_body"][3][
            "position_m"][1] += 0.01
        self.assert_rejected(result, "pinned soft-body support moved")

    def test_contact_cannot_substitute_for_attachment(self) -> None:
        result = valid_result(self.profile)
        result["fixtures"][0]["final"]["rope"][2][
            "soft_body_contact_force_n"][1] = 1.0
        self.assert_rejected(result, "contact channel is active")


if __name__ == "__main__":
    unittest.main()
