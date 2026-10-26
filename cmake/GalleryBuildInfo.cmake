# SPDX-License-Identifier: MIT
function(parallel_mater_gallery_build_info target backend)
    execute_process(COMMAND git rev-list --count HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE number OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT number MATCHES "^[0-9]+$")
        set(number 1)
    endif()
    set(info "${CMAKE_CURRENT_BINARY_DIR}/generated/${target}/build-info.json")
    add_custom_target(${target}-build-info
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DOUTPUT=${info}
            -DBACKEND=${backend}
            -DLOGO_PATH=${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/branding/logo.png
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateGalleryBuildInfo.cmake
        BYPRODUCTS "${info}"
        COMMENT "Refreshing ${target} build identity"
        VERBATIM)
    add_dependencies(${target} ${target}-build-info)
    target_sources(${target} PRIVATE "${info}")
    set_source_files_properties("${info}" PROPERTIES
        GENERATED TRUE
        MACOSX_PACKAGE_LOCATION Resources)
    set_target_properties(${target} PROPERTIES MACOSX_BUNDLE_BUNDLE_VERSION "${number}")
endfunction()
