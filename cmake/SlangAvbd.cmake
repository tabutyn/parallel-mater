# SPDX-License-Identifier: MIT

function(parallel_mater_add_slang_avbd)
    set(slangc "${PARALLEL_MATER_SLANGC_EXECUTABLE}")
    if(NOT slangc)
        message(FATAL_ERROR
            "parallel_mater_require_slang() must run before AVBD generation")
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
        DEPENDS "${source}" "${slangc}"
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
            DEPENDS "${source}" "${slangc}" CUDA::nvrtc
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
