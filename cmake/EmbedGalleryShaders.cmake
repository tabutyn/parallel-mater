# SPDX-License-Identifier: MIT
file(WRITE "${OUTPUT_FILE}" "// Generated from common gallery Slang. Do not edit.\n#pragma once\n#include <cstdint>\nnamespace parallel_mater::gallery::shaders {\n")
foreach(stage vertex fragment surfaceClear surfaceHash surfaceBounds surfaceField surfaceMesh surfaceFinalize)
    file(READ "${SHADER_DIR}/${stage}.dxil" bytes HEX)
    string(REGEX REPLACE "(..)" "0x\\1," bytes "${bytes}")
    file(APPEND "${OUTPUT_FILE}" "inline constexpr std::uint8_t ${stage}[] = {${bytes}};\n")
endforeach()
file(APPEND "${OUTPUT_FILE}" "}\n")
