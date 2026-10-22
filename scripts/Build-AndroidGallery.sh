#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
android_sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}"
if [[ ! -d "$android_sdk/ndk/29.0.14206865" ]]; then
    echo "Install Android NDK 29.0.14206865 in $android_sdk first." >&2
    exit 1
fi
export ANDROID_HOME="$android_sdk"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
arguments=(--no-daemon --max-workers=1 "-Pabis=${ANDROID_ABIS:-arm64-v8a}")
mapfile -t provenance_arguments < <(
    python3 "$project_root/scripts/android_deployment.py" \
        build-properties --root "$project_root")
arguments+=("${provenance_arguments[@]}")
if [[ -n "${SLANGC:-}" ]]; then
    arguments+=("-Pslangc=$SLANGC")
fi
cd "$project_root/examples/android"
./gradlew "${arguments[@]}" "$@" :app:assembleDebug
