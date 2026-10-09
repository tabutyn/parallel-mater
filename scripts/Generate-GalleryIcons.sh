#!/bin/bash
# SPDX-License-Identifier: MIT
# Regenerate the committed platform containers without changing the artwork.
set -euo pipefail
script_directory="$(cd "$(dirname "$0")" && pwd)"
branding_directory="$script_directory/../examples/assets/branding"
for tool in magick iconutil; do
    command -v "$tool" >/dev/null || { echo "Required tool missing: $tool" >&2; exit 1; }
done
temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/parallel-mater-icons.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT
iconset="$temporary_directory/ParallelMater.iconset"
mkdir -p "$iconset"
# Preserve the full image if a future source is not square; never crop the logo.
magick "$branding_directory/logo.png" -resize 1024x1024 \
    -background '#07141b' -gravity center -extent 1024x1024 \
    "$temporary_directory/square.png"
for size in 16 32 128 256 512; do
    magick "$temporary_directory/square.png" -resize "${size}x${size}" \
        "$iconset/icon_${size}x${size}.png"
    retina_size=$((size * 2))
    magick "$temporary_directory/square.png" -resize "${retina_size}x${retina_size}" \
        "$iconset/icon_${size}x${size}@2x.png"
done
iconutil --convert icns --output "$branding_directory/ParallelMater.icns" "$iconset"
magick "$temporary_directory/square.png" \
    -define icon:auto-resize=256,128,64,48,32,24,16 \
    "$branding_directory/ParallelMater.ico"
echo "Updated macOS and Windows icons in $branding_directory"
