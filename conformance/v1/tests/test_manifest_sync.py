# SPDX-License-Identifier: MIT
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "update_manifest.py"
SPEC = importlib.util.spec_from_file_location("conformance_manifest_sync", MODULE_PATH)
manifest_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_sync)


class ManifestSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "conformance/v1/manifest.json"
        self.manifest_path.parent.mkdir(parents=True)
        self.inputs = {
            "glb_sha256": "examples/assets/Fixture.glb",
            "lifecycle_profile_sha256": "conformance/v1/lifecycle/profile.json",
            "quality_profile_sha256": "conformance/v1/quality/profile.json",
            "qualification_profile_sha256": "conformance/v1/qualification/profile.json",
            "evidence_index_sha256": "conformance/v1/evidence/index.json",
        }
        manifest = {"schema": "fixture/v1", "source_commit": "unchanged"}
        for section, relative in self.inputs.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}\n" if path.suffix == ".json" else b"fixture")
            manifest[section] = {relative: "0" * 64}
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_refreshes_only_hash_pins_and_is_byte_idempotent(self):
        before = json.loads(self.manifest_path.read_text())
        changes = manifest_sync.synchronize_manifest(self.root, write=True)
        self.assertEqual(len(changes), len(self.inputs))
        after = json.loads(self.manifest_path.read_text())
        self.assertEqual(after["schema"], before["schema"])
        self.assertEqual(after["source_commit"], before["source_commit"])
        for section, relative in self.inputs.items():
            self.assertEqual(after[section][relative],
                             manifest_sync.contract_sha256(self.root / relative))
        first = self.manifest_path.read_bytes()
        self.assertEqual(manifest_sync.synchronize_manifest(self.root, write=True), [])
        self.assertEqual(self.manifest_path.read_bytes(), first)

    def test_check_rejects_stale_input_without_writing(self):
        manifest_sync.synchronize_manifest(self.root, write=True)
        relative = self.inputs["qualification_profile_sha256"]
        (self.root / relative).write_text('{"changed":true}\n')
        before = self.manifest_path.read_bytes()
        with redirect_stdout(io.StringIO()):
            result = manifest_sync.main(["--root", str(self.root), "--check"])
        self.assertEqual(result, 1)
        self.assertEqual(self.manifest_path.read_bytes(), before)
        changes = manifest_sync.synchronize_manifest(self.root, write=True)
        self.assertEqual([(section, path) for section, path, _, _ in changes],
                         [("qualification_profile_sha256", relative)])

    def test_json_hash_normalizes_crlf(self):
        path = self.root / self.inputs["quality_profile_sha256"]
        path.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
        expected = hashlib.sha256(b'{\n  "value": 1\n}\n').hexdigest()
        self.assertEqual(manifest_sync.contract_sha256(path), expected)

    def test_rejects_path_outside_repository(self):
        manifest = json.loads(self.manifest_path.read_text())
        section = copy.deepcopy(manifest["quality_profile_sha256"])
        manifest["quality_profile_sha256"] = {"../outside.json": next(iter(section.values()))}
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with self.assertRaisesRegex(ValueError, "escapes the repository"):
            manifest_sync.synchronize_manifest(self.root, write=False)


if __name__ == "__main__":
    unittest.main()
