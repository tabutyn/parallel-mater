#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Focused tests for release artifact comparison and path auditing."""

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "release_reproducibility", ROOT / "tools/check_release_reproducibility.py")
REPRO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPRO)


def pe_with_timestamp(timestamp):
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<I", data, 72, timestamp)
    return bytes(data)


def archive_with_timestamp(timestamp):
    member = b"payload"
    header = (
        b"file.o/         " + str(timestamp).encode("ascii").ljust(12) +
        b"0     " + b"0     " + b"100644  " + str(len(member)).encode("ascii").ljust(10) +
        b"`\n")
    return b"!<arch>\n" + header + member + b"\n"


class ReleaseReproducibilityTests(unittest.TestCase):
    def classify(self, first, second, kind):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left", root / "right"
            left.write_bytes(first)
            right.write_bytes(second)
            return REPRO.classify_difference(left, right, kind)

    def test_pe_timestamp_is_classified_separately(self):
        self.assertEqual(
            self.classify(pe_with_timestamp(1), pe_with_timestamp(2), "pe"),
            "pe_coff_timestamp")

    def test_archive_member_metadata_is_classified_separately(self):
        self.assertEqual(
            self.classify(archive_with_timestamp(1), archive_with_timestamp(2), "archive"),
            "archive_member_metadata")

    def test_binary_content_difference_is_not_called_reproducible(self):
        self.assertEqual(
            self.classify(b"\x7fELF-one", b"\x7fELF-two", "elf"),
            "elf_content")

    def test_embedded_build_path_is_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_build, second_build = root / "build-a", root / "build-b"
            first, second = root / "first.lib", root / "second.lib"
            first.write_bytes(b"prefix " + str(first_build).encode() + b" suffix")
            second.write_bytes(b"prefix " + str(second_build).encode() + b" suffix")
            self.assertEqual(
                REPRO.classify_difference(
                    first, second, "binary", [(first_build, second_build)]),
                "embedded_build_path")

    def test_inventory_reports_added_removed_and_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            first.mkdir()
            second.mkdir()
            (first / "same.txt").write_text("same", encoding="utf-8")
            (second / "same.txt").write_text("same", encoding="utf-8")
            (first / "changed.txt").write_text("first", encoding="utf-8")
            (second / "changed.txt").write_text("second", encoding="utf-8")
            (first / "removed.txt").write_text("removed", encoding="utf-8")
            (second / "added.txt").write_text("added", encoding="utf-8")
            comparison = REPRO.compare_inventory(
                REPRO.inventory(first), REPRO.inventory(second), first, second)
            self.assertFalse(comparison["artifact_sets_match"])
            self.assertFalse(comparison["bit_identical"])
            self.assertEqual(comparison["added"], ["added.txt"])
            self.assertEqual(comparison["removed"], ["removed.txt"])
            self.assertEqual(
                comparison["differences"][0]["classification"], "text_content")

    def test_runtime_metadata_cannot_require_a_workstation_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install, workstation = root / "install", root / "workstation"
            install.mkdir()
            workstation.mkdir()
            (install / "driver.json").write_text(
                '{"library_path":"' + workstation.as_posix() + '/driver"}',
                encoding="utf-8")
            audit = REPRO.audit_runtime_paths(install, [workstation])
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(audit["required_path_hits"][0]["path"], "driver.json")

    def test_cache_paths_are_tokenized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, build, install = root / "source", root / "build", root / "install"
            self.assertEqual(
                REPRO.portable_value(str(build / "tool.exe"), source, build, install),
                "<build>/tool.exe")


if __name__ == "__main__":
    unittest.main()
