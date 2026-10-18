#!/bin/bash
# SPDX-License-Identifier: MIT
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${PM_IPAD_BUILD_DIR:-"$repo/build-ipad-gallery"}
: "${PM_APPLE_TEAM:?Set PM_APPLE_TEAM to your Apple development team ID}"
options=()
if [[ -n ${SLANGC:-} ]]; then options+=("-DPARALLEL_MATER_SLANGC_EXECUTABLE=$SLANGC"); fi
if [[ -n ${PM_CGLTF_SOURCE:-} ]]; then options+=("-DFETCHCONTENT_SOURCE_DIR_PARALLEL_MATER_CGLTF=$PM_CGLTF_SOURCE"); fi
if ((${#options[@]})); then
    cmake --preset ipad-metal-preview -S "$repo" -B "$build" "${options[@]}"
else
    # macOS still ships Bash 3, where expanding an empty array under `set -u`
    # reports it as unbound.
    cmake --preset ipad-metal-preview -S "$repo" -B "$build"
fi
cmake --build "$build" --config Release --target parallel-mater-ipad-gallery --parallel 6 -- \
    -allowProvisioningUpdates -allowProvisioningDeviceRegistration
app="$build/Release-iphoneos/ParallelMater Gallery.app"
shaders="$app/Frameworks/ParallelMaterMetal.framework/slang-metal"
[[ -s "$shaders/manifest.txt" ]] || { echo "Missing bundled Metal shader manifest" >&2; exit 1; }
while IFS='|' read -r module entry base; do
    [[ -s "$shaders/$base.metallib" && -s "$shaders/$base.json" ]] || {
        echo "Missing bundled Metal entry: $module.$entry" >&2; exit 1;
    }
done < "$shaders/manifest.txt"
codesign --verify --deep --strict "$app"
echo "Built and verified: $app"
