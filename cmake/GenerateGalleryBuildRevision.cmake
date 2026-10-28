# SPDX-License-Identifier: MIT

foreach(required SOURCE_DIR OUTPUT)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE revision_result)
if(NOT revision_result EQUAL 0 OR NOT revision MATCHES "^[0-9a-fA-F]+$")
    set(revision unknown)
endif()

get_filename_component(output_directory "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
set(temporary_output "${OUTPUT}.tmp")
file(WRITE "${temporary_output}"
    "// Generated at build time; do not edit.\n"
    "#pragma once\n"
    "namespace parallel_mater::gallery {\n"
    "inline constexpr char gallery_build_revision[] = \"${revision}\";\n"
    "}\n")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${temporary_output}" "${OUTPUT}"
    RESULT_VARIABLE copy_result)
file(REMOVE "${temporary_output}")
if(NOT copy_result EQUAL 0)
    message(FATAL_ERROR "Failed to generate compiled Gallery revision: ${OUTPUT}")
endif()
