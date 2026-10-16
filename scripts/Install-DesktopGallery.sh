#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
source_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
applications="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
python3 - "$source_directory" "$applications" <<'PY'
import pathlib
import sys

source, applications = map(pathlib.Path, sys.argv[1:])
destination = applications / 'parallel-mater.desktop'
previous = applications / 'parallel-mater-backend-test.desktop'
# Only migrate entries installed by Parallel Mater; leave custom launchers alone.
for path, marker in ((destination, 'X-ParallelMater-Managed=true'),
                     (previous, 'X-ParallelMater-BackendTest=true')):
    if path.exists() and marker not in path.read_text().splitlines():
        raise SystemExit(f'Refusing to replace an unmanaged desktop entry: {path}')
launcher = str(source / 'scripts/Run-DesktopGallery.sh')
launcher = launcher.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%')
applications.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix('.desktop.tmp')
temporary.write_text(f'''[Desktop Entry]
Type=Application
Version=1.0
Name=Parallel Mater
GenericName=GPU Physics Gallery
Comment=Choose CUDA or Vulkan physics
Exec="{launcher}"
Icon={source / 'examples/assets/branding/logo.png'}
Terminal=false
Categories=Science;Physics;
Keywords=Parallel Mater;Vulkan;CUDA;physics;
StartupNotify=true
X-ParallelMater-Managed=true
X-ParallelMater-Launcher=prepared-build
''')
temporary.replace(destination)
previous.unlink(missing_ok=True)
print(f'Installed: {destination}')
PY
if command -v update-desktop-database >/dev/null; then
    update-desktop-database "$applications"
fi
