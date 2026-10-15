# SPDX-License-Identifier: MIT

if(NOT DEFINED MANIFEST_FILE OR NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "MANIFEST_FILE and OUTPUT_FILE are required")
endif()

file(STRINGS "${MANIFEST_FILE}" records)
if(NOT records)
    message(FATAL_ERROR "DXIL manifest is empty: ${MANIFEST_FILE}")
endif()

set(declarations "")
set(table "")
set(index 0)
foreach(record IN LISTS records)
    string(REPLACE "|" ";" fields "${record}")
    list(LENGTH fields field_count)
    if(NOT field_count EQUAL 3)
        message(FATAL_ERROR "Malformed DXIL manifest record: ${record}")
    endif()
    list(GET fields 0 module)
    list(GET fields 1 entry)
    list(GET fields 2 input)
    if(NOT EXISTS "${input}")
        message(FATAL_ERROR "Missing DXIL schedule: ${input}")
    endif()
    file(READ "${input}" bytes HEX)
    string(REGEX REPLACE "(..)" "0x\\1," bytes "${bytes}")
    string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){16})" "\\1\n"
           bytes "${bytes}")
    string(APPEND declarations
        "inline constexpr std::uint8_t kernel_${index}[] = {\n${bytes}\n};\n")
    string(APPEND table
        "    {\"${module}\", \"${entry}\", kernel_${index}, sizeof(kernel_${index})},\n")
    math(EXPR index "${index} + 1")
endforeach()

file(WRITE "${OUTPUT_FILE}"
"// Generated from the common Slang schedule manifest. Do not edit.\n"
"#pragma once\n"
"#include <cstddef>\n"
"#include <cstdint>\n"
"namespace parallel_mater::d3d12::generated {\n"
"struct SlangKernelBytecode {\n"
"    const char* module;\n"
"    const char* entry;\n"
"    const std::uint8_t* code;\n"
"    std::size_t size;\n"
"};\n"
"${declarations}"
"inline constexpr SlangKernelBytecode slang_kernels[] = {\n${table}};\n"
"inline constexpr std::size_t slang_kernel_count =\n"
"    sizeof(slang_kernels) / sizeof(slang_kernels[0]);\n"
"}\n")
