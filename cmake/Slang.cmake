# SPDX-License-Identifier: MIT

set(PARALLEL_MATER_SLANG_VERSION "2026.18" CACHE STRING
    "Pinned Slang compiler version used by shared physics kernels")
set(PARALLEL_MATER_SLANGC_EXECUTABLE "" CACHE FILEPATH
    "Path to slangc; downloaded from the official release when empty")

function(parallel_mater_require_slang)
    function(parallel_mater_check_slang_version executable output)
        execute_process(
            COMMAND "${executable}" -version
            RESULT_VARIABLE PARALLEL_MATER_SLANG_VERSION_RESULT
            OUTPUT_VARIABLE PARALLEL_MATER_SLANG_FOUND_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_VARIABLE PARALLEL_MATER_SLANG_ERROR_VERSION
            ERROR_STRIP_TRAILING_WHITESPACE)
        if(NOT PARALLEL_MATER_SLANG_FOUND_VERSION)
            set(PARALLEL_MATER_SLANG_FOUND_VERSION
                "${PARALLEL_MATER_SLANG_ERROR_VERSION}")
        endif()
        if(PARALLEL_MATER_SLANG_VERSION_RESULT EQUAL 0 AND
           PARALLEL_MATER_SLANG_FOUND_VERSION STREQUAL
               PARALLEL_MATER_SLANG_VERSION)
            set(${output} TRUE PARENT_SCOPE)
        else()
            set(${output} FALSE PARENT_SCOPE)
        endif()
    endfunction()

    if(PARALLEL_MATER_SLANGC_EXECUTABLE AND
       EXISTS "${PARALLEL_MATER_SLANGC_EXECUTABLE}")
        parallel_mater_check_slang_version(
            "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            PARALLEL_MATER_EXPLICIT_SLANG_MATCHES)
        if(NOT PARALLEL_MATER_EXPLICIT_SLANG_MATCHES)
            message(FATAL_ERROR
                "PARALLEL_MATER_SLANGC_EXECUTABLE must be Slang "
                "${PARALLEL_MATER_SLANG_VERSION}")
        endif()
        set(PARALLEL_MATER_SLANGC_EXECUTABLE
            "${PARALLEL_MATER_SLANGC_EXECUTABLE}" PARENT_SCOPE)
        return()
    endif()

    find_program(PARALLEL_MATER_SYSTEM_SLANGC NAMES slangc
                 HINTS "$ENV{SLANG_DIR}/bin")
    if(PARALLEL_MATER_SYSTEM_SLANGC)
        parallel_mater_check_slang_version(
            "${PARALLEL_MATER_SYSTEM_SLANGC}"
            PARALLEL_MATER_SYSTEM_SLANG_MATCHES)
        if(PARALLEL_MATER_SYSTEM_SLANG_MATCHES)
            set(PARALLEL_MATER_SLANGC_EXECUTABLE
                "${PARALLEL_MATER_SYSTEM_SLANGC}" CACHE FILEPATH
                "Path to slangc; downloaded from the official release when empty"
                FORCE)
            set(PARALLEL_MATER_SLANGC_EXECUTABLE
                "${PARALLEL_MATER_SYSTEM_SLANGC}" PARENT_SCOPE)
            return()
        endif()
    endif()

    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}"
           PARALLEL_MATER_SLANG_PROCESSOR)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        if(PARALLEL_MATER_SLANG_PROCESSOR MATCHES "^(arm64|aarch64)$")
            set(PARALLEL_MATER_SLANG_ARCHIVE
                "slang-${PARALLEL_MATER_SLANG_VERSION}-macos-aarch64.tar.gz")
            set(PARALLEL_MATER_SLANG_SHA256
                59833c5cfa12aad72c6fbad9cfcc06be8781ecd58c5983e6c79425e819e16bb2)
        elseif(PARALLEL_MATER_SLANG_PROCESSOR MATCHES "^(x86_64|amd64)$")
            set(PARALLEL_MATER_SLANG_ARCHIVE
                "slang-${PARALLEL_MATER_SLANG_VERSION}-macos-x86_64.tar.gz")
            set(PARALLEL_MATER_SLANG_SHA256
                8d27b7020b102beecaa769c1ff7342dd2534c14cfae505a36e1b71d595208739)
        endif()
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        if(PARALLEL_MATER_SLANG_PROCESSOR MATCHES "^(arm64|aarch64)$")
            set(PARALLEL_MATER_SLANG_ARCHIVE
                "slang-${PARALLEL_MATER_SLANG_VERSION}-linux-aarch64-glibc-2.28.tar.gz")
            set(PARALLEL_MATER_SLANG_SHA256
                5ab662d24241a963ccf7433198d8b89483721407398f035102a8dad5fc6dd501)
        elseif(PARALLEL_MATER_SLANG_PROCESSOR MATCHES "^(x86_64|amd64)$")
            set(PARALLEL_MATER_SLANG_ARCHIVE
                "slang-${PARALLEL_MATER_SLANG_VERSION}-linux-x86_64-glibc-2.27.tar.gz")
            set(PARALLEL_MATER_SLANG_SHA256
                e45ea4f117d51b8c1e84fa49f562081e73a9f29d02bd4f7fad20678603282829)
        endif()
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        if(PARALLEL_MATER_SLANG_PROCESSOR MATCHES "^(arm64|aarch64)$")
            set(PARALLEL_MATER_SLANG_ARCHIVE
                "slang-${PARALLEL_MATER_SLANG_VERSION}-windows-aarch64.zip")
            set(PARALLEL_MATER_SLANG_SHA256
                39ec2c02eba40ecd4d169599ae8c3cf00040729639b44be138dd82c62de4209e)
        elseif(PARALLEL_MATER_SLANG_PROCESSOR MATCHES "^(x86_64|amd64)$")
            set(PARALLEL_MATER_SLANG_ARCHIVE
                "slang-${PARALLEL_MATER_SLANG_VERSION}-windows-x86_64.zip")
            set(PARALLEL_MATER_SLANG_SHA256
                6ffa4827b519fd0a85b38407049d87ab0c1f045fe2289cb1e6831f965169f8a1)
        endif()
    endif()

    if(NOT PARALLEL_MATER_SLANG_ARCHIVE)
        message(FATAL_ERROR
            "No pinned Slang ${PARALLEL_MATER_SLANG_VERSION} package for "
            "${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_HOST_SYSTEM_PROCESSOR}; set "
            "PARALLEL_MATER_SLANGC_EXECUTABLE explicitly")
    endif()

    FetchContent_Declare(
        parallel_mater_slang
        URL "https://github.com/shader-slang/slang/releases/download/v${PARALLEL_MATER_SLANG_VERSION}/${PARALLEL_MATER_SLANG_ARCHIVE}"
        URL_HASH "SHA256=${PARALLEL_MATER_SLANG_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(parallel_mater_slang)

    if(WIN32)
        set(PARALLEL_MATER_SLANGC_NAME slangc.exe)
    else()
        set(PARALLEL_MATER_SLANGC_NAME slangc)
    endif()
    set(PARALLEL_MATER_DOWNLOADED_SLANGC
        "${parallel_mater_slang_SOURCE_DIR}/bin/${PARALLEL_MATER_SLANGC_NAME}")
    if(NOT EXISTS "${PARALLEL_MATER_DOWNLOADED_SLANGC}")
        message(FATAL_ERROR
            "The downloaded Slang package does not contain bin/${PARALLEL_MATER_SLANGC_NAME}")
    endif()
    parallel_mater_check_slang_version(
        "${PARALLEL_MATER_DOWNLOADED_SLANGC}"
        PARALLEL_MATER_DOWNLOADED_SLANG_MATCHES)
    if(NOT PARALLEL_MATER_DOWNLOADED_SLANG_MATCHES)
        message(FATAL_ERROR
            "The downloaded slangc is not version "
            "${PARALLEL_MATER_SLANG_VERSION}")
    endif()
    set(PARALLEL_MATER_SLANGC_EXECUTABLE
        "${PARALLEL_MATER_DOWNLOADED_SLANGC}" CACHE FILEPATH
        "Path to slangc; downloaded from the official release when empty"
        FORCE)
    set(PARALLEL_MATER_SLANGC_EXECUTABLE
        "${PARALLEL_MATER_DOWNLOADED_SLANGC}" PARENT_SCOPE)
endfunction()
