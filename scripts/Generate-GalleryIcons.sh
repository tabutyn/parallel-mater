#!/bin/bash
# SPDX-License-Identifier: MIT
# Regenerate the committed Windows icon without changing the artwork.
set -euo pipefail
script_directory="$(cd "$(dirname "$0")" && pwd)"
branding_directory="$script_directory/../examples/assets/branding"
command -v magick >/dev/null || {
    echo "Required tool missing: magick" >&2
    exit 1
}
temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/parallel-mater-icons.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT
# Preserve the full image if a future source is not square; never crop the logo.
magick "$branding_directory/logo.png" -resize 1024x1024 \
    -background '#07141b' -gravity center -extent 1024x1024 \
    "$temporary_directory/square.png"
magick "$temporary_directory/square.png" \
    -define icon:auto-resize=256,128,64,48,32,24,16 \
    "$branding_directory/ParallelMater.ico"
echo "Updated Windows icon in $branding_directory"
