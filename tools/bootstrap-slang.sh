#!/usr/bin/env sh
set -eu

version=2026.19
destination=${1:-}
if [ -z "${destination}" ]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

machine=$(uname -m)
case "${machine}" in
    x86_64)
        archive="slang-${version}-linux-x86_64-glibc-2.27.tar.gz"
        checksum="5899bd40c3d1ee60eadd1d1b37acc4e88ae65cc25ac61651534d55b71a85fe75"
        ;;
    aarch64|arm64)
        archive="slang-${version}-linux-aarch64-glibc-2.28.tar.gz"
        checksum="f49229eb9606b47b122e1d8789be46a7bcd6bb2ffa94b06fae6d544a4e6df637"
        ;;
    *)
        echo "unsupported host architecture: ${machine}" >&2
        exit 2
        ;;
esac

mkdir -p "${destination}"
download="${destination}/${archive}"
url="https://github.com/shader-slang/slang/releases/download/v${version}/${archive}"
curl -L --fail --silent --show-error "${url}" -o "${download}"
printf '%s  %s\n' "${checksum}" "${download}" | sha256sum --check --status
tar -xzf "${download}" -C "${destination}"
"${destination}/bin/slangc" -version
