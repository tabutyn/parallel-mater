#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build="${PM_GALLERY_BUILD_DIR:-$root/build-desktop-switch}"
mode="${1:---pinned}"
case "$mode" in
--pinned)
    sdk="$("$root/scripts/Setup-VulkanValidation.sh")"
    layer_path="$sdk/share/vulkan/explicit_layer.d"
    layer_library="$sdk/lib/libVkLayer_khronos_validation.so"
    environment=("LD_LIBRARY_PATH=$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    ;;
--system)
    layer_path=/usr/share/vulkan/explicit_layer.d
    layer_library=/usr/lib/x86_64-linux-gnu/libVkLayer_khronos_validation.so
    environment=()
    ;;
*)
    echo "Usage: $0 [--pinned|--system]" >&2
    exit 2
    ;;
esac

shader="$build/generated/vulkan/fluid.spv"
manifest="$layer_path/VkLayer_khronos_validation.json"
for required in "$shader" "$manifest" "$layer_library"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

output="${PM_VULKAN_PROBE_DIR:-$root/out/qualification/vulkan-validation}"
mkdir -p "$output"
probe="$output/parallel-mater-vulkan-pipeline-probe"
"${CXX:-c++}" -std=c++20 -O2 "$root/tools/vulkan_pipeline_probe.cpp" \
    -lvulkan -o "$probe"

python3 - "$mode" "$manifest" <<'PY'
import json
import sys

layer = json.load(open(sys.argv[2], encoding="utf-8"))["layer"]
print(f"layer_mode={sys.argv[1]}")
print(f"layer_name={layer['name']}")
print(f"layer_api_version={layer['api_version']}")
print(f"layer_implementation_version={layer['implementation_version']}")
print(f"layer_library={layer['library_path']}")
PY
sha256sum "$manifest" "$layer_library" "$shader" "$probe"

env "${environment[@]}" \
    "VK_LAYER_PATH=$layer_path" PM_VULKAN_VALIDATION=1 \
    "$probe" "$shader" fluid_emit_cells 256
