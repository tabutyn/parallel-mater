# SPDX-License-Identifier: MIT

if(NOT DEFINED PREPARE_INPUT OR NOT DEFINED SOLVE_INPUT OR
   NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "PREPARE_INPUT, SOLVE_INPUT, and OUTPUT_FILE are required")
endif()

function(pm_binary_initializer input output_variable)
    file(READ "${input}" bytes HEX)
    string(REGEX REPLACE "(..)" "0x\\1," bytes "${bytes}")
    set(${output_variable} "${bytes}" PARENT_SCOPE)
endfunction()

pm_binary_initializer("${PREPARE_INPUT}" prepare_bytes)
pm_binary_initializer("${SOLVE_INPUT}" solve_bytes)

file(WRITE "${OUTPUT_FILE}"
"// Generated from the shared Slang schedule. Do not edit.\n"
"#pragma once\n"
"#include <cstddef>\n"
"#include <cstdint>\n"
"namespace parallel_mater::d3d12::generated {\n"
"inline constexpr std::uint8_t avbd_prepare[] = {${prepare_bytes}};\n"
"inline constexpr std::uint8_t avbd_solve[] = {${solve_bytes}};\n"
"inline constexpr std::size_t avbd_prepare_size = sizeof(avbd_prepare);\n"
"inline constexpr std::size_t avbd_solve_size = sizeof(avbd_solve);\n"
"}\n")
