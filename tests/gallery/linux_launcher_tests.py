# SPDX-License-Identifier: MIT
"""Validate Linux launcher installation without touching the user profile."""

import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts/Install-DesktopGallery.sh"
LAUNCHER = ROOT / "scripts/Run-DesktopGallery.sh"


class LinuxGalleryLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="parallel-mater-linux-launcher-")
        self.root = pathlib.Path(self.temporary.name)
        self.home = self.root / "home"
        self.data_home = self.root / "data"
        self.home.mkdir()
        self.environment = os.environ.copy()
        self.environment.update(
            HOME=str(self.home), XDG_DATA_HOME=str(self.data_home))

    def tearDown(self):
        self.temporary.cleanup()

    @property
    def desktop(self):
        return self.data_home / "applications/parallel-mater.desktop"

    @property
    def command(self):
        return self.home / ".local/bin/parallel-mater"

    @property
    def legacy_root(self):
        return self.data_home / "parallel-mater"

    def install(self, check=True):
        return subprocess.run(
            [INSTALLER], check=check, capture_output=True, text=True,
            env=self.environment, timeout=30)

    def test_installs_one_canonical_launcher_and_command(self):
        self.install()
        desktop = self.desktop.read_text()
        self.assertIn("Name=Parallel Mater\n", desktop)
        self.assertIn(f'Exec="{LAUNCHER}"', desktop)
        self.assertIn("StartupWMClass=parallel-mater\n", desktop)
        self.assertIn("X-ParallelMater-Launcher=prepared-build\n", desktop)
        self.assertEqual(self.command.resolve(), LAUNCHER.resolve())
        self.assertEqual(
            [path.name for path in self.desktop.parent.glob("*parallel*mater*.desktop")],
            ["parallel-mater.desktop"],
        )

        build = self.root / "build"
        manifests = build / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "slang-vulkan.json").write_text("{}\n")
        gallery_output = self.root / "gallery-output"
        gallery = build / "parallel-mater-gallery"
        gallery.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$PM_DRIVER_FILES\" > '{gallery_output}'\n"
            f"printf '%s\\n' \"$*\" >> '{gallery_output}'\n")
        gallery.chmod(0o755)
        environment = self.environment.copy()
        environment["PM_GALLERY_BUILD_DIR"] = str(build)
        subprocess.run(
            [self.command, "--backend", "vulkan"], check=True,
            capture_output=True, text=True, env=environment, timeout=30)
        self.assertEqual(
            gallery_output.read_text().splitlines(),
            [str(manifests / "slang-vulkan.json"), "--backend vulkan"],
        )
        self.install()  # Reinstallation is an idempotent repair.

    def test_migrates_owned_legacy_launcher(self):
        self.legacy_root.mkdir(parents=True)
        (self.legacy_root / ".parallel-mater-installation").write_text(
            "Parallel Mater Linux Development Launcher\nsource=/old\n")
        legacy_launcher = self.legacy_root / "parallel-mater"
        legacy_launcher.write_text("#!/bin/sh\nexit 99\n")
        self.command.parent.mkdir(parents=True)
        self.command.symlink_to(legacy_launcher)
        self.desktop.parent.mkdir(parents=True)
        self.desktop.write_text(
            "[Desktop Entry]\nName=Parallel Mater\nX-ParallelMater-Managed=true\n")
        previous = self.desktop.with_name("parallel-mater-backend-test.desktop")
        previous.write_text(
            "[Desktop Entry]\nName=Parallel Mater Backend Test\n"
            "X-ParallelMater-BackendTest=true\n")

        self.install()

        self.assertFalse(self.legacy_root.exists())
        self.assertFalse(previous.exists())
        self.assertEqual(self.command.resolve(), LAUNCHER.resolve())
        self.assertEqual(self.desktop.read_text().count("Name=Parallel Mater\n"), 1)

    def test_refuses_unmanaged_desktop_or_command(self):
        self.desktop.parent.mkdir(parents=True)
        self.desktop.write_text("[Desktop Entry]\nName=Unrelated\n")
        result = self.install(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unmanaged desktop entry", result.stderr)

        self.desktop.unlink()
        self.command.parent.mkdir(parents=True)
        self.command.symlink_to(self.root / "unrelated")
        result = self.install(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unmanaged command link", result.stderr)
        self.assertFalse(self.desktop.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
