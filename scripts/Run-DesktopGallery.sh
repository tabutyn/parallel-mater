#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
script_path="${BASH_SOURCE[0]}"
while [[ -L "$script_path" ]]; do
    script_directory="$(cd -P -- "$(dirname -- "$script_path")" && pwd)"
    link_target="$(readlink -- "$script_path")"
    if [[ "$link_target" = /* ]]; then
        script_path="$link_target"
    else
        script_path="${script_directory}/${link_target}"
    fi
done
source_directory="$(cd -P -- "$(dirname -- "$script_path")/.." && pwd)"
build_directory="${PM_GALLERY_BUILD_DIR:-${source_directory}/build-desktop-switch}"
gallery="${build_directory}/parallel-mater-gallery"
export PM_VULKAN_CACHE_DIR="${PM_VULKAN_CACHE_DIR:-${build_directory}/vulkan-cache}"
if [[ ! -x "$gallery" ]]; then
    printf 'Gallery executable missing: %s\nBuild the desktop gallery first.\n' "$gallery" >&2
    exit 1
fi
# Match the Windows app: discover both loadable drivers and use the shared
# in-app picker. Explicit --backend remains available for automated runs.
manifest_directory="${build_directory}/Release/manifests"
if [[ ! -d "$manifest_directory" ]]; then
    manifest_directory="${build_directory}/manifests"
fi
if [[ -z "${PM_DRIVER_FILES+x}" && -z "${PM_DRIVER_PATH+x}" ]]; then
    manifests=()
    for backend in cuda vulkan; do
        manifest="${manifest_directory}/slang-${backend}.json"
        [[ ! -f "$manifest" ]] || manifests+=("$manifest")
    done
    if [[ ${#manifests[@]} == 0 ]]; then
        printf 'No desktop driver manifests found in %s\n' "$manifest_directory" >&2
        exit 1
    fi
    export PM_DRIVER_FILES="$(IFS=:; printf '%s' "${manifests[*]}")"
fi
if [[ -z "${PM_LAYER_FILES+x}" && -z "${PM_LAYER_PATH+x}" ]]; then
    export PM_LAYER_FILES="${manifest_directory}/validation.json"
fi
exec "$gallery" "$@"
