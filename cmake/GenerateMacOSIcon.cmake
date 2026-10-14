# SPDX-License-Identifier: MIT

if(NOT DEFINED SOURCE_PNG OR NOT DEFINED OUTPUT_ICNS)
    message(FATAL_ERROR "SOURCE_PNG and OUTPUT_ICNS are required")
endif()
if(NOT EXISTS "${SOURCE_PNG}")
    message(FATAL_ERROR "Icon source not found: ${SOURCE_PNG}")
endif()

find_program(SIPS_EXECUTABLE sips REQUIRED)
find_program(ICONUTIL_EXECUTABLE iconutil REQUIRED)
get_filename_component(OUTPUT_DIRECTORY "${OUTPUT_ICNS}" DIRECTORY)
set(ICONSET "${OUTPUT_DIRECTORY}/ParallelMater.iconset")
file(REMOVE_RECURSE "${ICONSET}")
file(MAKE_DIRECTORY "${ICONSET}")

foreach(SIZE 16 32 128 256 512)
    math(EXPR RETINA_SIZE "${SIZE} * 2")
    execute_process(
        COMMAND "${SIPS_EXECUTABLE}" -z ${SIZE} ${SIZE} "${SOURCE_PNG}"
                --out "${ICONSET}/icon_${SIZE}x${SIZE}.png"
        RESULT_VARIABLE STANDARD_RESULT
        OUTPUT_QUIET ERROR_QUIET)
    execute_process(
        COMMAND "${SIPS_EXECUTABLE}" -z ${RETINA_SIZE} ${RETINA_SIZE}
                "${SOURCE_PNG}"
                --out "${ICONSET}/icon_${SIZE}x${SIZE}@2x.png"
        RESULT_VARIABLE RETINA_RESULT
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT STANDARD_RESULT EQUAL 0 OR NOT RETINA_RESULT EQUAL 0)
        file(REMOVE_RECURSE "${ICONSET}")
        message(FATAL_ERROR "sips could not generate the macOS icon set")
    endif()
endforeach()

execute_process(
    COMMAND "${ICONUTIL_EXECUTABLE}" --convert icns --output "${OUTPUT_ICNS}"
            "${ICONSET}"
    RESULT_VARIABLE ICONUTIL_RESULT)
file(REMOVE_RECURSE "${ICONSET}")
if(NOT ICONUTIL_RESULT EQUAL 0)
    message(FATAL_ERROR "iconutil could not generate ${OUTPUT_ICNS}")
endif()
