#!/usr/bin/env python3
"""Integrity checks for F6-10 capture/replay evidence."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
import unittest

from tests.roadmap.evidence_hash import byte_sha256, text_sha256


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/roadmap/evidence/F6-10-physics-capture-replay.json"
ARCHIVE = ROOT / "docs/roadmap/evidence/F6-07/snapshot-working-files.tar.gz"


class PhysicsCaptureReplayEvidenceTests(unittest.TestCase):
    def test_capture_and_candidate_scene_are_pinned(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        fixtures = report["fixtures"]
        capture = (ROOT / fixtures["capture"]).read_bytes()
        self.assertEqual(text_sha256(capture), fixtures["capture_sha256"])
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            candidate = archive.extractfile(
                "examples/assets/ConstraintHinge.glb").read()
        self.assertEqual(byte_sha256(candidate),
                         fixtures["candidate_scene_sha256"])

    def test_schema_and_known_quality_result_are_explicit(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        capture = (ROOT / report["fixtures"]["capture"]).read_text(
            encoding="utf-8")
        self.assertTrue(capture.startswith(
            "parallel_mater_physics_capture 5\n"))
        self.assertIn(
            'rigid_ids 4 "Gear" "Gear.001" "Gear.002" "Ground.002"',
            capture)
        self.assertIn("history_complete 0", capture)
        self.assertIn("timestep 0.00416666688 substeps 1", capture)
        replay = next(item for item in report["verification"]
                      if "--scenario replay" in item["check"])
        self.assertEqual(replay["exit_code"], 1)
        self.assertTrue(replay["metrics"]["cleared_top"])
        self.assertGreater(
            replay["metrics"]["maximum_gear_mesh_penetration_meters"],
            0.002)
        self.assertTrue(report["disposition"]["capture_replay_ready"])
        self.assertFalse(
            report["disposition"]["solver_or_fixture_changes_included"])


if __name__ == "__main__":
    unittest.main()
