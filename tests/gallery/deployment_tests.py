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
