#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
source_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"
applications="${data_home}/applications"
bin_directory="${HOME}/.local/bin"
python3 - "$source_directory" "$applications" "$bin_directory" "$data_home" <<'PY'
import os
import pathlib
import shutil
import sys

source, applications, bin_directory, data_home = map(pathlib.Path, sys.argv[1:])
destination = applications / 'parallel-mater.desktop'
previous = applications / 'parallel-mater-backend-test.desktop'
launcher = source / 'scripts/Run-DesktopGallery.sh'
command = bin_directory / 'parallel-mater'
legacy_root = data_home / 'parallel-mater'
legacy_launcher = legacy_root / 'parallel-mater'
legacy_marker = legacy_root / '.parallel-mater-installation'
legacy_owned = False
if legacy_marker.is_file():
    marker = legacy_marker.read_text().splitlines()
    legacy_owned = bool(marker) and marker[0] in {
        'Parallel Mater Linux Gallery',
        'Parallel Mater Linux Development Launcher',
    }

# Only migrate entries installed by Parallel Mater; leave custom launchers alone.
for path, marker in ((destination, 'X-ParallelMater-Managed=true'),
                     (previous, 'X-ParallelMater-BackendTest=true')):
    if path.exists() and marker not in path.read_text().splitlines():
        raise SystemExit(f'Refusing to replace an unmanaged desktop entry: {path}')

if command.is_symlink():
    raw_target = pathlib.Path(os.readlink(command))
    command_target = (command.parent / raw_target).resolve() if not raw_target.is_absolute() else raw_target.resolve()
    allowed_targets = {launcher.resolve()}
    if legacy_owned:
        allowed_targets.add(legacy_launcher.resolve())
    if command_target not in allowed_targets:
        raise SystemExit(f'Refusing to replace an unmanaged command link: {command}')
elif command.exists():
    raise SystemExit(f'Refusing to replace an unmanaged command: {command}')

desktop_launcher = str(launcher)
desktop_launcher = desktop_launcher.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%')
applications.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix('.desktop.tmp')
temporary.write_text(f'''[Desktop Entry]
Type=Application
Version=1.0
Name=Parallel Mater
GenericName=GPU Physics Gallery
Comment=Choose CUDA or Vulkan physics
Exec="{desktop_launcher}"
Icon={source / 'examples/assets/branding/logo.png'}
Terminal=false
Categories=Science;Physics;
Keywords=Parallel Mater;Vulkan;CUDA;physics;
StartupNotify=true
StartupWMClass=parallel-mater
X-ParallelMater-Managed=true
X-ParallelMater-Launcher=prepared-build
''')
temporary.replace(destination)
previous.unlink(missing_ok=True)
bin_directory.mkdir(parents=True, exist_ok=True)
temporary_command = command.with_name(command.name + '.tmp')
temporary_command.unlink(missing_ok=True)
temporary_command.symlink_to(launcher)
temporary_command.replace(command)
if legacy_owned and legacy_root != source and legacy_root.exists():
    shutil.rmtree(legacy_root)
print(f'Installed: {destination}')
print(f'Command: {command} -> {launcher}')
PY
if command -v update-desktop-database >/dev/null; then
    update-desktop-database "$applications"
fi
