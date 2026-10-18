import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))

from build import RELEASE_ARTIFACTS
from provenance import registry_provenance


class ReferenceToolchainTests(unittest.TestCase):
    def setUp(self):
        self.lock_path = ROOT / "tools/reference/toolchain.json"
        self.lock = json.loads(self.lock_path.read_text(encoding="utf-8"))

    def test_requirements_match_locked_packages(self):
        requirements = {}
        for line in (ROOT / "tools/reference/requirements.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                package, version = line.split("==", 1)
                requirements[package] = version
        self.assertEqual(requirements, self.lock["packages"])

    def test_font_sources_and_hashes_are_pinned(self):
        entries = self.lock["fonts"] + self.lock["licenses"]
        self.assertEqual(len({entry["name"] for entry in entries}), len(entries))
        for entry in entries:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("/main/", entry["url"])
            if "archive_sha256" in entry:
                self.assertRegex(entry["archive_sha256"], r"^[0-9a-f]{64}$")
                self.assertIn("member", entry)

    def test_release_inventory_covers_every_output_class(self):
        expected = {
            ".h": 4,
            ".json": 3,
            ".md": 1,
            ".pdf": 2,
        }
        actual = {suffix: sum(Path(path).suffix == suffix for path in RELEASE_ARTIFACTS)
                  for suffix in expected}
        self.assertEqual(actual, expected)

    def test_tracked_provenance_matches_registry(self):
        digest = registry_provenance()["sha256"]
        for relative in (
                "docs/reference/api/parallel_mater.h",
                "docs/reference/api/interop_vulkan.h",
                "include/parallel_mater/parallel_mater.h",
                "include/parallel_mater/interop_vulkan.h",
                "docs/reference/parallel-mater-specification.md"):
            self.assertIn(digest, (ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
