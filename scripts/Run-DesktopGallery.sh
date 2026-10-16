#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
source_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_directory="${PM_GALLERY_BUILD_DIR:-${source_directory}/build-desktop-switch}"
gallery="${build_directory}/parallel-mater-gallery"
export PM_VULKAN_CACHE_DIR="${PM_VULKAN_CACHE_DIR:-${build_directory}/vulkan-cache}"
if [[ ! -x "$gallery" ]]; then
    printf 'Gallery executable missing: %s\nBuild the desktop gallery first.\n' "$gallery" >&2
    exit 1
fi
# Explicit selection also supports headless test runs without opening a dialog.
for argument in "$@"; do
    if [[ "$argument" == --backend || "$argument" == --help ]]; then
        exec "$gallery" "$@"
    fi
done
if ! command -v zenity >/dev/null; then
    printf 'Choose a backend: %s --backend vulkan (or cuda)\n' "$0" >&2
    exit 2
fi
choices=()
for backend in cuda vulkan; do
    if [[ -f "${build_directory}/Release/manifests/slang-${backend}.json" ]]; then
        if [[ "$backend" == cuda ]]; then
            choices+=(TRUE CUDA 'Full gallery')
        else
            choices+=(FALSE Vulkan 'Rigid bodies, constraints, and liquids • preview')
        fi
    fi
done
if [[ ${#choices[@]} == 0 ]]; then
    printf 'No desktop driver manifests found in %s\n' "$build_directory" >&2
    exit 1
fi
backend="$(zenity --list --radiolist --title='Parallel Mater' \
    --text='Choose the physics backend for this session.' \
    --column='' --column='Backend' --column='Available scenes' \
    --width=560 --height=250 --ok-label='Open Gallery' "${choices[@]}")" || exit 0
[[ -n "$backend" ]] || exit 0
exec "$gallery" --backend "${backend,,}" "$@"
