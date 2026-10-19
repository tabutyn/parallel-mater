#!/usr/bin/env python3
"""Integrity checks for the retained F6-07 hinge/contact evidence."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tarfile
import unittest

from tests.roadmap.evidence_hash import byte_sha256, text_sha256


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/roadmap/evidence/F6-07"


class LegacyHingeEvidenceTests(unittest.TestCase):
    def test_archive_matches_metadata_and_reproduction_record(self):
        metadata_path = EVIDENCE / "snapshot-metadata.json"
        patch_path = EVIDENCE / "snapshot-unstaged.patch"
        archive_path = EVIDENCE / "snapshot-working-files.tar.gz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reproduction = json.loads(
            (EVIDENCE / "cuda-reproduction.json").read_text(encoding="utf-8"))
        snapshot = reproduction["snapshot"]

        self.assertEqual(text_sha256(metadata_path.read_bytes()),
                         snapshot["metadata_sha256"])
        self.assertEqual(text_sha256(patch_path.read_bytes()),
                         snapshot["unstaged_patch_sha256"])
        self.assertEqual(byte_sha256(archive_path.read_bytes()),
                         snapshot["working_files_archive_sha256"])

        with tarfile.open(archive_path, "r:gz") as archive:
            files = {
                member.name: byte_sha256(archive.extractfile(member).read())
                for member in archive.getmembers() if member.isfile()
            }
        self.assertEqual(metadata["sha256"], files)
        self.assertEqual(len(files), snapshot["verified_working_file_count"])

    def test_every_patch_hunk_and_untracked_capture_has_a_disposition(self):
        patch = (EVIDENCE / "snapshot-unstaged.patch").read_text(
            encoding="utf-8")
        evidence = json.loads(
            (EVIDENCE / "hunk-dispositions.json").read_text(encoding="utf-8"))
        items = {item["id"]: item for item in evidence["items"]}
        expected = {}

        sections = list(re.finditer(
            r"^diff --git a/(.+) b/(.+)$", patch, re.MULTILINE))
        for section_index, section_match in enumerate(sections):
            path = section_match.group(2)
            start = section_match.start()
            end = (sections[section_index + 1].start()
                   if section_index + 1 < len(sections) else len(patch))
            section = patch[start:end]
            hunks = list(re.finditer(r"^@@ .*?@@.*$", section, re.MULTILINE))
            if not hunks:
                expected[f"{path}#binary"] = byte_sha256(section.encode())
                continue
            for hunk_index, hunk in enumerate(hunks, 1):
                hunk_end = (hunks[hunk_index].start()
                            if hunk_index < len(hunks) else len(section))
                expected[f"{path}#h{hunk_index}"] = byte_sha256(
                    section[hunk.start():hunk_end].encode())

        capture_id = "tests/gallery/hinge_top_capture.hpp#untracked"
        self.assertEqual(set(expected) | {capture_id}, set(items))
        for item_id, expected_digest in expected.items():
            self.assertEqual(expected_digest, items[item_id]["patch_sha256"])
        self.assertEqual(evidence["inventory"]["total_items"], 62)
        self.assertEqual(items[capture_id]["content_sha256"],
                         json.loads((EVIDENCE / "snapshot-metadata.json").read_text(
                             encoding="utf-8"))["sha256"][
                                 "tests/gallery/hinge_top_capture.hpp"])
        for item in items.values():
            self.assertIn(item["action"], {
                "candidate", "quarantined", "superseded-review",
                "preserved-reproduction"})
            self.assertTrue(item["follow_up_tasks"])
            self.assertLessEqual(set(item["follow_up_tasks"]), {
                "F6-09", "F6-10", "F6-11"})


if __name__ == "__main__":
    unittest.main()
