# SPDX-License-Identifier: MIT
function(parallel_mater_gallery_build_info target backend)
    execute_process(COMMAND git status --porcelain --untracked-files=normal
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE changes ERROR_QUIET)
    set(dirty false)
    if(changes)
        set(dirty true)
    endif()
    execute_process(COMMAND git rev-list --count HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE number OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT number MATCHES "^[0-9]+$")
        set(number 1)
    endif()
    set(info "${CMAKE_CURRENT_BINARY_DIR}/generated/${target}/build-info.json")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/branding/logo.png" logo_hash)
    file(GENERATE OUTPUT "${info}" CONTENT
"{\"schema_version\":1,\"revision\":\"${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}\",\"dirty\":${dirty},\"backend\":\"${backend}\",\"release_channel\":\"development-preview\",\"build_number\":${number},\"logo_sha256\":\"${logo_hash}\"}\n")
    target_sources(${target} PRIVATE "${info}")
    set_source_files_properties("${info}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
    set_target_properties(${target} PROPERTIES MACOSX_BUNDLE_BUNDLE_VERSION "${number}")
endfunction()
