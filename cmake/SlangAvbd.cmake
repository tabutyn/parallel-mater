# SPDX-License-Identifier: MIT
include(ExternalProject)

set(PARALLEL_MATER_SLANG_VERSION "2026.18" CACHE STRING
    "Pinned Slang compiler version used for AVBD shaders")
set(PARALLEL_MATER_SLANGC_EXECUTABLE "" CACHE FILEPATH
    "Existing slangc executable; empty downloads the pinned release")

function(parallel_mater_add_slang_avbd)
    set(slang_dependency)
    set(slangc "${PARALLEL_MATER_SLANGC_EXECUTABLE}")
    if(NOT slangc)
        find_program(slangc NAMES slangc HINTS "$ENV{SLANG_DIR}/bin")
    endif()

    if(slangc)
        execute_process(
            COMMAND "${slangc}" -version
            OUTPUT_VARIABLE slangc_version
            ERROR_VARIABLE slangc_version_error
            RESULT_VARIABLE slangc_status)
        if(NOT slangc_version)
            set(slangc_version "${slangc_version_error}")
        endif()
        string(STRIP "${slangc_version}" slangc_version)
        if(NOT slangc_status EQUAL 0 OR
           NOT slangc_version STREQUAL PARALLEL_MATER_SLANG_VERSION)
            message(FATAL_ERROR
                "slangc reports '${slangc_version}', expected ${PARALLEL_MATER_SLANG_VERSION}")
        endif()
    else()
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" slang_architecture)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
           slang_architecture MATCHES "^(x86_64|amd64)$")
            set(slang_archive "slang-${PARALLEL_MATER_SLANG_VERSION}-linux-x86_64-glibc-2.27.tar.gz")
            set(slang_hash "e45ea4f117d51b8c1e84fa49f562081e73a9f29d02bd4f7fad20678603282829")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
               slang_architecture MATCHES "^(aarch64|arm64)$")
            set(slang_archive "slang-${PARALLEL_MATER_SLANG_VERSION}-linux-aarch64-glibc-2.28.tar.gz")
            set(slang_hash "5ab662d24241a963ccf7433198d8b89483721407398f035102a8dad5fc6dd501")
        elseif(APPLE AND slang_architecture MATCHES "^(aarch64|arm64)$")
            set(slang_archive "slang-${PARALLEL_MATER_SLANG_VERSION}-macos-aarch64.tar.gz")
            set(slang_hash "59833c5cfa12aad72c6fbad9cfcc06be8781ecd58c5983e6c79425e819e16bb2")
        elseif(APPLE AND slang_architecture MATCHES "^(x86_64|amd64)$")
            set(slang_archive "slang-${PARALLEL_MATER_SLANG_VERSION}-macos-x86_64.tar.gz")
            set(slang_hash "8d27b7020b102beecaa769c1ff7342dd2534c14cfae505a36e1b71d595208739")
        elseif(WIN32 AND slang_architecture MATCHES "^(x86_64|amd64)$")
            set(slang_archive "slang-${PARALLEL_MATER_SLANG_VERSION}-windows-x86_64.zip")
            set(slang_hash "6ffa4827b519fd0a85b38407049d87ab0c1f045fe2289cb1e6831f965169f8a1")
        elseif(WIN32 AND slang_architecture MATCHES "^(aarch64|arm64)$")
            set(slang_archive "slang-${PARALLEL_MATER_SLANG_VERSION}-windows-aarch64.zip")
            set(slang_hash "39ec2c02eba40ecd4d169599ae8c3cf00040729639b44be138dd82c62de4209e")
        else()
            message(FATAL_ERROR
                "No pinned Slang package for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}; "
                "set PARALLEL_MATER_SLANGC_EXECUTABLE")
        endif()

        set(slang_root
            "${CMAKE_CURRENT_BINARY_DIR}/_deps/parallel-mater-slang-${PARALLEL_MATER_SLANG_VERSION}")
        ExternalProject_Add(parallel-mater-slang-toolchain
            URL "https://github.com/shader-slang/slang/releases/download/v${PARALLEL_MATER_SLANG_VERSION}/${slang_archive}"
            URL_HASH "SHA256=${slang_hash}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            SOURCE_DIR "${slang_root}"
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            EXCLUDE_FROM_ALL TRUE)
        set(slangc "${slang_root}/bin/slangc${CMAKE_EXECUTABLE_SUFFIX}")
        set(slang_dependency parallel-mater-slang-toolchain)
    endif()

    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/slang")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/avbd.slang")
    set(cuda_source "${generated_dir}/parallel_mater_avbd.cu")
    set(cuda_reflection "${generated_dir}/parallel_mater_avbd.cuda.json")
    set(metal_source "${generated_dir}/parallel_mater_avbd.metal")
    set(metal_reflection "${generated_dir}/parallel_mater_avbd.metal.json")
    set(hlsl_source "${generated_dir}/parallel_mater_avbd.hlsl")
    set(hlsl_reflection "${generated_dir}/parallel_mater_avbd.hlsl.json")

    add_custom_command(
        OUTPUT
            "${cuda_source}" "${cuda_reflection}"
            "${metal_source}" "${metal_reflection}"
            "${hlsl_source}" "${hlsl_reflection}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
        COMMAND "${slangc}" "${source}"
            -entry avbdConformanceMain -stage compute
            -target cuda -fp-mode precise -warnings-as-errors all
            -o "${cuda_source}" -reflection-json "${cuda_reflection}"
        COMMAND "${slangc}" "${source}"
            -entry avbdConformanceMain -stage compute
            -target metal -fp-mode precise -warnings-as-errors all
            -o "${metal_source}" -reflection-json "${metal_reflection}"
        COMMAND "${slangc}" "${source}"
            -entry avbdConformanceMain -stage compute
            -target hlsl -profile sm_5_1 -fp-mode precise
            -warnings-as-errors all
            -o "${hlsl_source}" -reflection-json "${hlsl_reflection}"
        DEPENDS "${source}" ${slang_dependency}
        COMMENT "Compiling the AVBD numerical core with Slang ${PARALLEL_MATER_SLANG_VERSION}"
        VERBATIM)

    set(slang_outputs
        "${cuda_source}" "${cuda_reflection}"
        "${metal_source}" "${metal_reflection}"
        "${hlsl_source}" "${hlsl_reflection}")

    if(TARGET CUDA::nvrtc)
        set(ptx "${generated_dir}/parallel_mater_avbd.ptx")
        add_custom_command(
            OUTPUT "${ptx}"
            COMMAND "${slangc}" "${source}"
                -entry avbdConformanceMain -stage compute
                -target ptx -fp-mode precise -warnings-as-errors all
                -nvrtc-path "$<TARGET_FILE_DIR:CUDA::nvrtc>/nvrtc"
                -o "${ptx}"
            DEPENDS "${source}" ${slang_dependency} CUDA::nvrtc
            COMMENT "Compiling the Slang AVBD CUDA conformance kernel"
            VERBATIM)
        list(APPEND slang_outputs "${ptx}")
        set(PARALLEL_MATER_SLANG_AVBD_PTX "${ptx}" PARENT_SCOPE)
    endif()

    if(PARALLEL_MATER_BUILD_METAL)
        set(metal_air "${generated_dir}/parallel_mater_avbd.air")
        add_custom_command(
            OUTPUT "${metal_air}"
            COMMAND ${CMAKE_COMMAND} -E env
                CLANG_MODULE_CACHE_PATH=${generated_dir}/metal-module-cache
                "${PARALLEL_MATER_XCRUN_EXECUTABLE}" -sdk macosx metal
                -std=metal4.0 -mmacosx-version-min=26.0
                -fno-fast-math -ffp-contract=fast
                -fmodules-cache-path=${generated_dir}/metal-module-cache
                -c "${metal_source}" -o "${metal_air}"
            DEPENDS "${metal_source}"
            COMMENT "Validating Slang-generated AVBD with the native Metal compiler"
            VERBATIM)
        list(APPEND slang_outputs "${metal_air}")
    endif()

    if(PARALLEL_MATER_BUILD_D3D12)
        set(d3d12_binary "${generated_dir}/parallel_mater_avbd.cso")
        add_custom_command(
            OUTPUT "${d3d12_binary}"
            COMMAND "${PARALLEL_MATER_FXC_EXECUTABLE}"
                /nologo /T cs_5_1 /E avbdConformanceMain
                /O3 /Gis /Ges /WX /Fo "${d3d12_binary}" "${hlsl_source}"
            DEPENDS "${hlsl_source}"
            COMMENT "Validating Slang-generated AVBD with the native D3D12 compiler"
            VERBATIM)
        list(APPEND slang_outputs "${d3d12_binary}")
    endif()

    add_custom_target(parallel-mater-slang-avbd ALL DEPENDS ${slang_outputs})
    set(PARALLEL_MATER_SLANG_AVBD_CUDA_SOURCE "${cuda_source}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_AVBD_CUDA_REFLECTION "${cuda_reflection}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_AVBD_METAL_SOURCE "${metal_source}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_AVBD_METAL_REFLECTION "${metal_reflection}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_AVBD_HLSL_SOURCE "${hlsl_source}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_AVBD_HLSL_REFLECTION "${hlsl_reflection}" PARENT_SCOPE)
endfunction()
