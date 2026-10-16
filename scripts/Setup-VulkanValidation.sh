#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]]; then
    echo 'This SDK pin is for Linux x86_64 hosts.' >&2
    exit 1
fi
mapfile -t pin < <(python3 - "$root/config/vulkan-validation-sdk.json" <<'PY'
import json, sys
pin = json.load(open(sys.argv[1]))
for key in ('version', 'url', 'sha256'):
    print(pin[key])
PY
)
sdk="$root/out/toolchains/vulkan-${pin[0]}"
if [[ -f "$sdk/SHA256SUMS" && -f "$sdk/archive.sha256" ]] &&
   [[ "$(cat "$sdk/archive.sha256")" == "${pin[2]}" ]] &&
   (cd "$sdk" && sha256sum --check --status SHA256SUMS); then
    printf '%s\n' "$sdk"
    exit 0
fi
mkdir -p "$root/out/toolchains"
staging="$(mktemp -d "$root/out/toolchains/.vulkan-sdk.XXXXXX")"
trap 'rm -rf -- "$staging"' EXIT
archive="${PM_VULKAN_SDK_ARCHIVE:-$staging/sdk.tar.xz}"
if [[ -z "${PM_VULKAN_SDK_ARCHIVE:-}" ]]; then
    curl --fail --location --retry 3 --output "$archive" "${pin[1]}" >&2
fi
printf '%s  %s\n' "${pin[2]}" "$archive" | sha256sum --check >&2
mkdir "$staging/sdk"
files=(bin/spirv-val lib/libVkLayer_khronos_validation.so
       share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json)
members=()
for file in "${files[@]}"; do members+=("${pin[0]}/x86_64/$file"); done
tar -xJf "$archive" -C "$staging/sdk" --strip-components=2 "${members[@]}"
(cd "$staging/sdk" && sha256sum "${files[@]}" > SHA256SUMS)
printf '%s\n' "${pin[2]}" > "$staging/sdk/archive.sha256"
# Keep the previous installation until its replacement is verified and extracted.
if [[ -d "$sdk" ]]; then mv "$sdk" "$staging/previous"; fi
mv "$staging/sdk" "$sdk"
printf '%s\n' "$sdk"
