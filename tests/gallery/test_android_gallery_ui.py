import importlib.util
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch


SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts/Test-AndroidGalleryUi.py"
SPEC = importlib.util.spec_from_file_location("android_gallery_ui", SCRIPT)
android_gallery_ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(android_gallery_ui)


def system_dialog(title):
    return ET.fromstring(f"""
        <hierarchy>
          <node package="android" text="{title}">
            <node package="android" resource-id="android:id/aerr_close"
                  text="Close app" bounds="[0,0][100,100]" />
            <node package="android" text="Wait" bounds="[0,100][100,200]" />
          </node>
        </hierarchy>
    """)


class AndroidGalleryUiTests(unittest.TestCase):
    def device(self, output):
        return android_gallery_ui.Device("adb", "emulator-5554", output, 1)

    def test_closes_quickstep_anr_and_retains_recovery_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = self.device(pathlib.Path(temporary))
            with patch.object(device, "tap_node") as tap:
                self.assertTrue(device.dismiss_recoverable_system_dialog(
                    system_dialog("Quickstep isn't responding")))
            tap.assert_called_once()
            self.assertEqual(tap.call_args.args[0].get("text"), "Close app")
            self.assertEqual(device.system_dialog_recoveries,
                             ["Quickstep isn't responding"])
            evidence = pathlib.Path(temporary) / "recovered-launcher-anr.xml"
            self.assertEqual(ET.parse(evidence).find("node").get("text"),
                             "Quickstep isn't responding")

    def test_requires_system_dialog_identity_and_close_button(self):
        for mismatch in ("package", "resource-id", "text"):
            with self.subTest(mismatch=mismatch), \
                    tempfile.TemporaryDirectory() as temporary:
                device = self.device(pathlib.Path(temporary))
                dialog = system_dialog("Quickstep isn't responding")
                close = dialog.find("node/node")
                close.set(mismatch, "unrecognized")
                with patch.object(device, "tap_node") as tap:
                    self.assertFalse(device.dismiss_recoverable_system_dialog(dialog))
                tap.assert_not_called()

    def test_preserves_gallery_anr_as_a_real_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = self.device(pathlib.Path(temporary))
            with patch.object(device, "tap_node") as tap:
                self.assertFalse(device.dismiss_recoverable_system_dialog(
                    system_dialog("Parallel Mater isn't responding")))
            tap.assert_not_called()

    def test_recurring_launcher_anr_fails_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            device = self.device(output)
            dialog = system_dialog("Quickstep isn't responding")
            with patch.object(device, "hierarchy", return_value=dialog), \
                    patch.object(device, "tap_node") as tap, \
                    patch.object(device, "run", side_effect=[b"screenshot", "logcat"]), \
                    patch.object(android_gallery_ui.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "launcher ANR recurred"):
                    device.wait_node("Resume")
            tap.assert_called_once()
            self.assertTrue((output / "failure-resume.xml").is_file())
            self.assertEqual((output / "failure-resume.png").read_bytes(), b"screenshot")
            self.assertEqual((output / "failure-resume-logcat.txt").read_text(), "logcat")

    def test_wait_node_retries_after_dismissing_quickstep(self):
        gallery = ET.fromstring(
            '<hierarchy><node content-desc="Choose scene: RIGID BODY" /></hierarchy>')
        with tempfile.TemporaryDirectory() as temporary:
            device = self.device(pathlib.Path(temporary))
            with patch.object(device, "hierarchy", side_effect=[
                    system_dialog("Quickstep isn't responding"), gallery]), \
                    patch.object(device, "tap_node") as tap, \
                    patch.object(android_gallery_ui.time, "sleep"):
                node, tree = device.wait_node("Choose scene: RIGID BODY")
            self.assertIs(tree, gallery)
            self.assertEqual(node.get("content-desc"), "Choose scene: RIGID BODY")
            tap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
