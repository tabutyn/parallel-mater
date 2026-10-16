#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
source_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
applications="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
desktop_file="${applications}/parallel-mater-backend-test.desktop"
if [[ -e "$desktop_file" ]] && ! rg -q '^X-ParallelMater-BackendTest=true$' "$desktop_file"; then
    printf 'Refusing to overwrite an unmanaged desktop entry: %s\n' "$desktop_file" >&2
    exit 1
fi
mkdir -p "$applications"
python3 - "$source_directory" "$desktop_file" <<'PY'
import pathlib, sys
source, destination = map(pathlib.Path, sys.argv[1:])
# Desktop Exec quoting has its own escaping rules; do not pass shell source.
launcher = str(source / 'scripts/Run-DesktopGallery.sh')
launcher = launcher.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%')
destination.write_text(f'''[Desktop Entry]
Type=Application
Version=1.0
Name=Parallel Mater Backend Test
Comment=Choose CUDA or Vulkan physics
Exec="{launcher}"
Icon={source / 'examples/assets/branding/logo.png'}
Terminal=false
Categories=Science;Physics;
Keywords=Parallel Mater;Vulkan;CUDA;physics;
StartupNotify=true
X-ParallelMater-BackendTest=true
''')
PY
if command -v update-desktop-database >/dev/null; then
    update-desktop-database "$applications"
fi
printf 'Installed: Parallel Mater Backend Test\n%s\n' "$desktop_file"
