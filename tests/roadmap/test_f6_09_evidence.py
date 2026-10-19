#!/usr/bin/env python3
"""Integrity checks for the rejected F6-09 hinge fixture candidate."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
import unittest

from tests.roadmap.evidence_hash import byte_sha256, text_sha256


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/roadmap/evidence/F6-09-hinge-fixture-qualification.json"
ARCHIVE = ROOT / "docs/roadmap/evidence/F6-07/snapshot-working-files.tar.gz"


class HingeFixtureQualificationTests(unittest.TestCase):
    def test_candidate_and_accepted_artifacts_are_distinct_and_immutable(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        source = report["source"]
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            candidate_blend = archive.extractfile(
                "examples/assets/ConstraintHinge.blend").read()
            candidate_glb = archive.extractfile(
                "examples/assets/ConstraintHinge.glb").read()
            candidate_test = archive.extractfile(
                "tests/blender/export_scene_tests.py").read()

        self.assertEqual(byte_sha256(candidate_blend),
                         source["candidate_blend_sha256"])
        self.assertEqual(byte_sha256(candidate_glb),
                         source["archived_glb_sha256"])
        self.assertEqual(byte_sha256(candidate_test),
                         source["candidate_blender_test_sha256"])
        self.assertEqual(source["archived_glb_sha256"],
                         source["current_exporter_glb_sha256"])
        accepted = source["accepted_assets"]
        accepted_blend = ROOT / "examples/assets/ConstraintHinge.blend"
        accepted_glb = ROOT / "examples/assets/ConstraintHinge.glb"
        accepted_golden = (
            ROOT / "conformance/v1/golden/cuda/constraint-hinge.json")
        self.assertEqual(byte_sha256(accepted_blend.read_bytes()),
                         accepted["blend_sha256"])
        self.assertEqual(byte_sha256(accepted_glb.read_bytes()),
                         accepted["glb_sha256"])
        self.assertEqual(text_sha256(accepted_golden.read_bytes()),
                         accepted["cuda_golden_sha256"])
        self.assertNotEqual(source["candidate_blend_sha256"],
                            accepted["blend_sha256"])
        self.assertNotEqual(source["archived_glb_sha256"], accepted["glb_sha256"])

    def test_failed_quality_gate_prevents_promotion_and_golden_update(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        full_rotation = next(check for check in report["verification"]
                             if "full-rotation" in check["check"])
        self.assertEqual(full_rotation["exit_code"], 1)
        self.assertIn("41.4778 mm", full_rotation["result"])
        self.assertIn("gear_mesh_penetration=0.0414778",
                      full_rotation["metrics"])
        self.assertFalse(report["disposition"]["candidate_promoted"])
        self.assertFalse(report["disposition"]["committed_gallery_assets_changed"])
        self.assertFalse(report["disposition"]["golden_generated"])


if __name__ == "__main__":
    unittest.main()
