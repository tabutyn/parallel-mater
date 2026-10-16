#!/bin/bash
# SPDX-License-Identifier: MIT
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${PM_IPAD_BUILD_DIR:-"$repo/build-ipad-gallery"}
: "${PM_APPLE_TEAM:?Set PM_APPLE_TEAM to your Apple development team ID}"
options=()
if [[ -n ${SLANGC:-} ]]; then options+=("-DPARALLEL_MATER_SLANGC_EXECUTABLE=$SLANGC"); fi
if [[ -n ${PM_CGLTF_SOURCE:-} ]]; then options+=("-DFETCHCONTENT_SOURCE_DIR_PARALLEL_MATER_CGLTF=$PM_CGLTF_SOURCE"); fi
cmake -S "$repo" -B "$build" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="$PM_APPLE_TEAM" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_STYLE=Automatic \
    -DBUILD_TESTING=OFF -DPARALLEL_MATER_BUILD_CUDA=OFF \
    -DPARALLEL_MATER_BUILD_VULKAN=OFF -DPARALLEL_MATER_BUILD_D3D12=OFF \
    -DPARALLEL_MATER_BUILD_GALLERY=ON -DPARALLEL_MATER_BUILD_SLANG_CPU_DRIVER=OFF \
    -DPARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW=ON \
    -DPARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS=OFF "${options[@]}"
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
