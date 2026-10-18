#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sdk="$("$root/scripts/Setup-VulkanValidation.sh")"
build="${PM_GALLERY_BUILD_DIR:-$root/build-desktop-switch}"
export VK_LAYER_PATH="$sdk/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export PM_VULKAN_CACHE_DIR="${PM_VULKAN_CACHE_DIR:-$build/vulkan-cache}"
for module in geometry geometry_constraints primitives queries fluid geometry_fluid \
    cloth geometry_cloth fluid_cloth soft_body geometry_soft_body fluid_soft_body \
    soft_body_cloth rope fluid_rope cloth_rope soft_body_rope geometry_smoke \
    smoke_grid fluid_smoke cloth_smoke soft_body_smoke rope_smoke; do
    "$sdk/bin/spirv-val" --target-env vulkan1.3 --scalar-block-layout \
        "$build/generated/vulkan/$module.spv"
done
if [[ $# == 0 ]]; then
    set -- -R 'vulkan-|gallery-input|gallery-workload|^parallel-mater-command-api-tests$'
fi
ctest --test-dir "$build" --output-on-failure -j1 "$@"
