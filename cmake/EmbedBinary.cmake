# SPDX-License-Identifier: MIT
if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
    message(FATAL_ERROR "EmbedBinary.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()

file(READ "${INPUT}" CONTENT HEX)
string(REGEX REPLACE "(..)" "0x\\1," CONTENT "${CONTENT}")
file(WRITE "${OUTPUT}"
    "// Generated from ${INPUT}; do not edit.\n"
    "#pragma once\n"
    "#include <cstddef>\n"
    "alignas(4) inline constexpr unsigned char ${SYMBOL}[] = {${CONTENT}};\n"
    "inline constexpr std::size_t ${SYMBOL}_size = sizeof(${SYMBOL});\n")
