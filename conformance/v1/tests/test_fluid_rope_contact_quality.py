#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "quality.py"
PROFILE_PATH = ROOT / "quality" / "fluid-rope-contact-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_fluid_rope_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def particle(position: list[float], velocity: list[float]) -> dict:
    return {
        "stable_id": 0,
        "position_m": position,
        "velocity_m_s": velocity,
        "acceleration_m_s2": [0.0, 0.0, 0.0],
    }


def node(index: int, position: list[float], velocity: list[float],
         force: list[float]) -> dict:
    return {
        "index": index,
        "position_m": position,
        "velocity_m_s": velocity,
        "liquid_contact_force_n": force,
    }


class FluidRopeContactQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        fixture = copy.deepcopy(self.profile["fixture"])
        fixture["initial_fluid_state"] = [
            particle([-0.09, 0.025, 0.0], [1.0, 0.0, 0.0])]
        fixture["final_fluid_state"] = [
            particle([-0.1, 0.025, 0.0], [0.0, 0.0, 0.0])]
        fixture["initial_rope_node_state"] = [
            node(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            node(1, [0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ]
        fixture["final_rope_node_state"] = [
            node(0, [0.0, 0.0, 0.0], [0.375, 0.0, 0.0],
                 [75.0, 0.0, 0.0]),
            node(1, [0.0, 0.1, 0.0], [0.125, 0.0, 0.0],
                 [25.0, 0.0, 0.0]),
        ]
        self.result = {
            "schema": "parallel-mater-conformance-quality-result/v1",
            "profile_id": self.profile["id"],
            "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "fixture": fixture,
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return QUALITY.evaluate(self.profile, result or self.result,
                                QUALITY.profile_sha256(PROFILE_PATH))

    def test_accepts_reciprocal_weighted_contact(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["sample_count"], 1)
        self.assertAlmostEqual(
            report["observations"]["initial_segment_fraction"], 0.25)

    def test_rejects_terminal_overlap(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixture"]["final_fluid_state"][0]["position_m"][0] = -0.08
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("penetrates" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_nonreciprocal_response(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixture"]["final_rope_node_state"][1]["velocity_m_s"][0] = 0.0
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("not reciprocal" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_hidden_endpoint_attachment(self) -> None:
        result = copy.deepcopy(self.result)
        nodes = result["fixture"]["final_rope_node_state"]
        nodes[0]["velocity_m_s"][0] = 0.0
        nodes[0]["liquid_contact_force_n"][0] = 0.0
        nodes[1]["velocity_m_s"][0] = 0.5
        nodes[1]["liquid_contact_force_n"][0] = 100.0
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("hidden attachment" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_force_impulse_mismatch(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixture"]["final_rope_node_state"][0][
            "liquid_contact_force_n"][0] = 0.0
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("inconsistent with impulse" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_fixture_setting_change(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixture"]["rope_first_attached"] = True
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any(item["path"] == "$.fixture.rope_first_attached"
                            for item in report["failures"]), report)

    def test_rejects_changed_stable_id(self) -> None:
        result = copy.deepcopy(self.result)
        result["fixture"]["final_fluid_state"][0]["stable_id"] = 7
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("stable ID" in item["message"]
                            for item in report["failures"]), report)


if __name__ == "__main__":
    unittest.main()
