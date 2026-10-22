#!/usr/bin/env python3
import importlib.util
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "android_deployment", ROOT / "scripts/android_deployment.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AndroidDeploymentTests(unittest.TestCase):
    def test_missing_installation_is_rejected(self):
        with self.assertRaisesRegex(MODULE.DeploymentError, "missing"):
            MODULE.parse_package_path("")

    def test_stale_installed_apk_is_rejected(self):
        local = {"apk_sha256": "a", "driver_sha256": "driver"}
        installed = {"apk_sha256": "b", "driver_sha256": "driver"}
        with self.assertRaisesRegex(MODULE.DeploymentError, "APK hash"):
            MODULE.verify_installed_artifact(local, installed)

    def test_wrong_device_is_rejected(self):
        with self.assertRaisesRegex(MODULE.DeploymentError, "wrong physical"):
            MODULE.verify_device_properties({
                "manufacturer": "Google", "model": "Pixel 8", "product": "shiba"})

    def test_stale_runtime_nonce_is_rejected(self):
        source = {"revision": "a" * 40, "dirty": False,
                  "patch_sha256": "clean"}
        runtime = {"nonce": "old", "revision": source["revision"],
                   "backend": "vulkan", "presentation": "vulkan-direct",
                   "build": {"source_revision": source["revision"],
                             "source_dirty": False,
                             "source_patch_sha256": "clean"}}
        with self.assertRaisesRegex(MODULE.DeploymentError, "nonce"):
            MODULE.verify_runtime_identity(runtime, "fresh", source)

    def test_dirty_identity_covers_untracked_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email",
                            "tests@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name",
                            "Tests"], check=True)
            (root / "tracked").write_text("base\n")
            subprocess.run(["git", "-C", str(root), "add", "tracked"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"],
                           check=True)
            clean = MODULE.source_identity(root)
            self.assertFalse(clean["dirty"])
            self.assertEqual(clean["patch_sha256"], "clean")
            (root / "new").write_bytes(b"one")
            first = MODULE.source_identity(root)
            (root / "new").write_bytes(b"two")
            second = MODULE.source_identity(root)
            self.assertTrue(first["dirty"])
            self.assertNotEqual(first["patch_sha256"], second["patch_sha256"])


if __name__ == "__main__":
    unittest.main()
