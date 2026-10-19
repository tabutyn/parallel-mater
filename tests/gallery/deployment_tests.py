# SPDX-License-Identifier: MIT
"""Portable deployment guards; no device, signing or network side effects."""
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("deploy", pathlib.Path(__file__).resolve().parents[2] / "scripts/Deploy-Gallery.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def physics_report(self):
        artifacts = [
            {"path": f"physics-qualification/cases/case-{index}.json",
             "kind": "trajectory", "status": "completed", "sha256": "a" * 64}
            for index in range(30)]
        artifacts += [
            {"path": "physics-qualification/provenance.json", "kind": "device-provenance",
             "status": "completed", "sha256": "b" * 64},
            {"path": "physics-qualification/capabilities.json", "kind": "advertised-capabilities",
             "status": "completed", "sha256": "c" * 64},
            {"path": "physics-qualification/quality/rigid.json", "kind": "physical-quality",
             "status": "completed", "sha256": "d" * 64},
            {"path": "physics-qualification/quality/fluid.json", "kind": "physical-quality",
             "status": "unsupported", "sha256": "e" * 64},
            {"path": "physics-qualification/quality/fluid-rope.json", "kind": "physical-quality",
             "status": "completed", "sha256": "e" * 64},
            {"path": "physics-qualification/lifecycle/adverse.json", "kind": "adverse-lifecycle",
             "status": "captured", "sha256": "f" * 64},
        ]
        return {
            "schema": "parallel-mater-ipados-qualification-run/v1",
            "report_kind": "public-api-conformance", "gallery_scene_sweep": False,
            "nonce": "fresh", "revision": "revision", "backend": "slang-metal",
            "completed": True, "binary_identity": {"app_executable_sha256": "1" * 64,
                                                     "driver_sha256": "2" * 64},
            "trajectory": {"case_count": 30, "completed": 30, "unsupported": 0, "failed": 0},
            "device": {"model": "iPad13,8", "gpu": "Apple M1", "operating_system": "iPadOS",
                       "operating_system_version": "26.5", "metal_api": "Metal",
                       "metal_version": "Metal 4", "metal_gpu_family": "Apple7"},
            "toolchain": {"compiler": "AppleClang", "compiler_version": "21", "slang_version": "2026.18"},
            "artifacts": artifacts,
        }

    def test_physics_qualification_accepts_complete_public_api_report(self):
        report = self.physics_report()
        self.assertEqual(len(deploy.validate_physics_qualification(
            report, "fresh", "revision", report["binary_identity"])), 36)

    def test_physics_qualification_rejects_stale_partial_mismatched_and_gallery_only(self):
        for field, value, message in (
            ("nonce", "old", "stale"),
            ("completed", False, "partial"),
            ("revision", "old", "revision"),
            ("report_kind", "gallery-scene-sweep", "Gallery-only"),
        ):
            report = self.physics_report()
            report[field] = value
            with self.assertRaisesRegex(RuntimeError, message):
                deploy.validate_physics_qualification(
                    report, "fresh", "revision", report["binary_identity"])

    def test_physics_qualification_rejects_binary_mismatch_and_private_ids(self):
        report = self.physics_report()
        with self.assertRaisesRegex(RuntimeError, "hash differs"):
            deploy.validate_physics_qualification(
                report, "fresh", "revision", {"app_executable_sha256": "0" * 64,
                                                 "driver_sha256": "2" * 64})
        report["device"]["serial_number"] = "private"
        with self.assertRaisesRegex(RuntimeError, "private device"):
            deploy.validate_physics_qualification(
                report, "fresh", "revision", report["binary_identity"])

    def test_physics_qualification_requires_runtime_metal_version(self):
        for field in ("metal_version", "metal_gpu_family"):
            report = self.physics_report()
            del report["device"][field]
            with self.assertRaisesRegex(RuntimeError, "iPadOS/Metal"):
                deploy.validate_physics_qualification(
                    report, "fresh", "revision", report["binary_identity"])

    def test_dead_process_stops_wait_and_preserves_partial_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / "processes.json").write_text(json.dumps({"result": {"runningProcesses": []}}))
            partial = {"nonce": "new", "completed": False, "scenes": [{"passed": True}]}
            with patch.object(deploy, "copy_report", return_value=partial), patch.object(deploy, "run"):
                with self.assertRaisesRegex(RuntimeError, "Gallery exited"):
                    deploy.wait_report("device", "report", "new", directory, lambda r: r["completed"], 60,
                        health=lambda: deploy.assert_process_running("device", 123, directory),
                        save=directory / "partial.json")
            self.assertEqual(json.loads((directory / "partial.json").read_text()), partial)

    def test_old_reports_and_nonce_are_rejected(self):
        reports = [[{"passed": True}], {"nonce": "old", "completed": True},
                   {"nonce": "new", "completed": False}, {"nonce": "new", "completed": True}]
        with patch.object(deploy, "copy_report", side_effect=reports) as copy, patch.object(deploy.time, "sleep"):
            result = deploy.wait_report("device", "report", "new", pathlib.Path("unused"), lambda r: r["completed"], 2)
        self.assertTrue(result["completed"])
        self.assertEqual(copy.call_count, 4)

    def test_missing_report_times_out(self):
        with patch.object(deploy, "copy_report", side_effect=OSError), patch.object(deploy.time, "sleep"), \
             patch.object(deploy.time, "monotonic", side_effect=[0, 0, 2]):
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                deploy.wait_report("device", "report", "new", pathlib.Path("unused"), lambda r: True, 1)

    def test_build_identity_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = pathlib.Path(tmp)
            (app / "Assets.car").touch()
            info = dict(revision="current", dirty=False, backend="metal", release_channel="development-preview")
            def check(changes="", allow_dirty=False):
                (app / "build-info.json").write_text(json.dumps(info))
                def run(*args, **kwargs):
                    return types.SimpleNamespace(stdout="current\n" if "rev-parse" in args else changes)
                with patch.object(deploy, "run", side_effect=run):
                    return deploy.build_info(app, "ipad", allow_dirty)
            self.assertEqual(check(), info)
            info["revision"] = "old"
            with self.assertRaisesRegex(RuntimeError, "differs from HEAD"): check()
            info["revision"] = "current"
            with self.assertRaisesRegex(RuntimeError, "clean commit"): check("M changed.cpp")
            info["dirty"] = True
            with self.assertRaisesRegex(RuntimeError, "clean commit"): check()
            self.assertEqual(check(allow_dirty=True), info)
            info["dirty"] = False
            info["backend"] = "slang-cpu"
            with self.assertRaisesRegex(RuntimeError, "Metal development preview"): check()
            info["backend"] = "metal"
            (app / "Assets.car").unlink()
            with self.assertRaisesRegex(RuntimeError, "icon asset catalog"): check()


if __name__ == "__main__": unittest.main()
