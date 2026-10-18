# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "ctest_manifest_contract", Path(__file__).with_name("ctest_manifest_contract.py"))
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


class CTestManifestContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.library = root / "libparallel_mater_layer_validation.so"
        self.library.touch()
        self.manifest = root / "validation.json"
        self.manifest.write_text(json.dumps({
            "layer": {
                "name": "PM_LAYER_PARALLEL_MATER_validation",
                "library_path": str(self.library),
            }
        }), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def catalog(self, environment):
        return {"tests": [{
            "name": "parallel-mater-gallery-physical",
            "properties": [
                {"name": "LABELS", "value": ["gallery", "vulkan"]},
                {"name": "ENVIRONMENT", "value": environment},
            ],
        }]}

    def test_accepts_exact_build_manifest(self):
        audited = contract.audit_catalog(self.catalog([
            "PM_DRIVER_FILES=/build/manifests/slang-vulkan.json",
            f"PM_LAYER_FILES={self.manifest}",
        ]), self.manifest, self.library)
        self.assertEqual(audited, ["parallel-mater-gallery-physical"])

    def test_rejects_missing_layer_manifest(self):
        with self.assertRaisesRegex(AssertionError, "expected one PM_LAYER_FILES"):
            contract.audit_catalog(self.catalog([
                "PM_DRIVER_FILES=/build/manifests/slang-vulkan.json",
            ]), self.manifest, self.library)

    def test_rejects_layer_directory_discovery(self):
        with self.assertRaisesRegex(AssertionError, "PM_LAYER_PATH"):
            contract.audit_catalog(self.catalog([
                "PM_DRIVER_FILES=/build/manifests/slang-vulkan.json",
                f"PM_LAYER_FILES={self.manifest}",
                f"PM_LAYER_PATH={self.manifest.parent}",
            ]), self.manifest, self.library)

    def test_rejects_another_manifest(self):
        with self.assertRaisesRegex(AssertionError, "expected"):
            contract.audit_catalog(self.catalog([
                "PM_DRIVER_FILES=/build/manifests/slang-vulkan.json",
                "PM_LAYER_FILES=/installed/manifests/validation.json",
            ]), self.manifest, self.library)


if __name__ == "__main__":
    unittest.main()
