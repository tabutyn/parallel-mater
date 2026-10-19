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
PROFILE_PATH = ROOT / "quality" / "fluid-cloth-contact-v1.json"
SPEC = importlib.util.spec_from_file_location("pm_fluid_cloth_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)


def snapshot(particle_y: float, particle_velocity_y: float,
             cloth_y: float, cloth_velocity_y: float) -> dict:
    momentum_y = particle_velocity_y + 3.0 * cloth_velocity_y
    separation = particle_y - cloth_y
    return {
        "particle_position_m": [0.0, particle_y, 0.0],
        "particle_velocity_m_s": [0.0, particle_velocity_y, 0.0],
        "particle_acceleration_m_s2": [0.0, 0.0, 0.0],
        "cloth_vertices_m": [
            [-1.0, cloth_y, -1.0], [0.0, cloth_y, 1.0],
            [1.0, cloth_y, -1.0]],
        "cloth_velocities_m_s": [[0.0, cloth_velocity_y, 0.0]] * 3,
        "cloth_liquid_contact_forces_n": [[0.0, 0.0, 0.0]] * 3,
        "cloth_normal": [0.0, 1.0, 0.0],
        "signed_separation_m": separation,
        "contact_violation_m": max(0.0, 0.05 + separation),
        "endpoint_momentum_kg_m_s": [0.0, momentum_y, 0.0],
        "fluid_normal_velocity_m_s": particle_velocity_y,
        "cloth_mean_normal_velocity_m_s": cloth_velocity_y,
    }


class FluidClothContactQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        definition = copy.deepcopy(self.profile["fixtures"][0])
        definition["initial"] = snapshot(0.01, 0.0, 0.0, 0.0)
        definition["first_step"] = snapshot(0.0097, -0.06, 0.0001, 0.02)
        definition["final"] = snapshot(-0.04, 0.0, 0.01, 0.0)
        self.result = {
            "schema": "parallel-mater-conformance-quality-result/v1",
            "profile_id": self.profile["id"],
            "profile_sha256": QUALITY.profile_sha256(PROFILE_PATH),
            "execution_status": "completed",
            "driver": {"name": "slang-test", "device": "test-device"},
            "fixtures": [definition],
        }

    def evaluate(self, result: dict | None = None) -> dict:
        return QUALITY.evaluate(self.profile, result or self.result,
                                QUALITY.profile_sha256(PROFILE_PATH))

    def test_accepts_incremental_reciprocal_contact(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["sample_count"], 2)

    def test_rejects_containment_snap(self) -> None:
        result = copy.deepcopy(self.result)
        first = result["fixtures"][0]["first_step"]
        first["particle_position_m"][1] = -0.05
        first["signed_separation_m"] = -0.0501
        first["contact_violation_m"] = 0.0
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("snapped like containment" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_terminal_contact_overlap(self) -> None:
        result = copy.deepcopy(self.result)
        final = result["fixtures"][0]["final"]
        final["particle_position_m"][1] = 0.0
        final["signed_separation_m"] = -0.01
        final["contact_violation_m"] = 0.04
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("terminal contact distance" in item["message"]
                            for item in report["failures"]), report)

    def test_rejects_nonreciprocal_momentum(self) -> None:
        result = copy.deepcopy(self.result)
        first = result["fixtures"][0]["first_step"]
        first["cloth_velocities_m_s"][0][1] = 0.05
        first["endpoint_momentum_kg_m_s"][1] = 0.03
        report = self.evaluate(result)
        self.assertFalse(report["passed"])
        self.assertTrue(any("normal momentum is unbalanced" in item["message"]
                            for item in report["failures"]), report)


if __name__ == "__main__":
    unittest.main()
