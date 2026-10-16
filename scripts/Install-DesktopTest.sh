#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Compatibility entrypoint; installation now uses one canonical app entry.
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/Install-DesktopGallery.sh" "$@"
