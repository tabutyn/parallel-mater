# SPDX-License-Identifier: MIT

foreach(required SOURCE_DIR OUTPUT BACKEND LOGO_PATH)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

if(DEFINED REVISION_OVERRIDE AND NOT REVISION_OVERRIDE STREQUAL "")
    set(revision "${REVISION_OVERRIDE}")
else()
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE revision_result)
    if(NOT revision_result EQUAL 0 OR revision STREQUAL "")
        set(revision unknown)
    endif()
endif()

execute_process(
    COMMAND git status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE changes
    ERROR_QUIET
    RESULT_VARIABLE status_result)
set(dirty false)
if(NOT status_result EQUAL 0 OR NOT changes STREQUAL "")
    set(dirty true)
endif()

execute_process(
    COMMAND git rev-list --count HEAD
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE number
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE number_result)
if(NOT number_result EQUAL 0 OR NOT number MATCHES "^[0-9]+$")
    set(number 1)
endif()

if(NOT EXISTS "${LOGO_PATH}")
    message(FATAL_ERROR "Gallery logo not found: ${LOGO_PATH}")
endif()
file(SHA256 "${LOGO_PATH}" logo_hash)

get_filename_component(output_directory "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
set(temporary_output "${OUTPUT}.tmp")
file(WRITE "${temporary_output}"
    "{\"schema_version\":2,\"revision\":\"${revision}\",\"dirty\":${dirty},\"backend\":\"${BACKEND}\",\"execution_contract\":\"fixed-tick-60hz-v1\",\"release_channel\":\"development-preview\",\"build_number\":${number},\"logo_sha256\":\"${logo_hash}\"}\n")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${temporary_output}" "${OUTPUT}"
    RESULT_VARIABLE copy_result)
file(REMOVE "${temporary_output}")
if(NOT copy_result EQUAL 0)
    message(FATAL_ERROR "Failed to generate Gallery build identity: ${OUTPUT}")
endif()
