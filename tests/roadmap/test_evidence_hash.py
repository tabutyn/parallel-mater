#!/usr/bin/env python3
"""Regression checks for retained evidence hash portability."""

from __future__ import annotations

import unittest

from tests.roadmap.evidence_hash import byte_sha256, text_sha256


class EvidenceHashTests(unittest.TestCase):
    def test_lf_and_crlf_text_have_the_same_digest(self):
        self.assertEqual(text_sha256(b"first\nsecond\n"),
                         text_sha256(b"first\r\nsecond\r\n"))

    def test_lone_carriage_return_and_content_changes_remain_visible(self):
        canonical = text_sha256(b"first\nsecond\n")
        self.assertNotEqual(canonical, text_sha256(b"first\rsecond\n"))
        self.assertNotEqual(canonical, text_sha256(b"first\nchanged\n"))

    def test_binary_hash_remains_byte_exact(self):
        self.assertNotEqual(byte_sha256(b"first\nsecond\n"),
                            byte_sha256(b"first\r\nsecond\r\n"))


if __name__ == "__main__":
    unittest.main()
