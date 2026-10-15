# SPDX-License-Identifier: MIT

function(parallel_mater_add_slang_geometry_constraints)
    set(slangc "${PARALLEL_MATER_SLANGC_EXECUTABLE}")
    if(NOT slangc)
        message(FATAL_ERROR
            "parallel_mater_require_slang() must run before rigid constraint generation")
    endif()

    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/slang")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry_constraints.slang")
    set(cuda_source "${generated_dir}/parallel_mater_geometry_constraints.cu")
    set(cuda_reflection "${generated_dir}/parallel_mater_geometry_constraints.cuda.json")
    set(metal_source "${generated_dir}/parallel_mater_geometry_constraints.metal")
    set(metal_reflection "${generated_dir}/parallel_mater_geometry_constraints.metal.json")
    set(hlsl_source "${generated_dir}/parallel_mater_geometry_constraints.hlsl")
    set(hlsl_reflection "${generated_dir}/parallel_mater_geometry_constraints.hlsl.json")
    set(wgsl_source "${generated_dir}/parallel_mater_geometry_constraints.wgsl")
    set(wgsl_reflection "${generated_dir}/parallel_mater_geometry_constraints.wgsl.json")
    file(GLOB_RECURSE slang_dependencies CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slang")

    add_custom_command(
        OUTPUT
            "${cuda_source}" "${cuda_reflection}"
            "${metal_source}" "${metal_reflection}"
            "${hlsl_source}" "${hlsl_reflection}"
            "${wgsl_source}" "${wgsl_reflection}"
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
        COMMAND "${slangc}" "${source}"
            -entry avbdConformanceMain -stage compute
            -target wgsl -fp-mode precise -warnings-as-errors all
            -o "${wgsl_source}" -reflection-json "${wgsl_reflection}"
        DEPENDS ${slang_dependencies} "${slangc}"
        COMMENT "Compiling the rigid constraint numerical core with Slang ${PARALLEL_MATER_SLANG_VERSION}"
        VERBATIM)

    set(slang_outputs
        "${cuda_source}" "${cuda_reflection}"
        "${metal_source}" "${metal_reflection}"
        "${hlsl_source}" "${hlsl_reflection}"
        "${wgsl_source}" "${wgsl_reflection}")

    if(WIN32)
        set(dxil_prepare
            "${generated_dir}/parallel_mater_avbd_prepare.dxil")
        set(dxil_solve
            "${generated_dir}/parallel_mater_avbd_solve.dxil")
        set(dxil_prepare_reflection
            "${generated_dir}/parallel_mater_avbd_prepare.dxil.json")
        set(dxil_solve_reflection
            "${generated_dir}/parallel_mater_avbd_solve.dxil.json")
        set(dxil_arguments
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
            -target dxil -profile sm_6_6 -fp-mode precise -O2
            -warnings-as-errors all
            -DPM_TARGET_D3D12=1 -DPM_TARGET_CUDA=0
            -DPM_GEOMETRY_CONSTRAINTS_ADAPTER=1)
        add_custom_command(
            OUTPUT "${dxil_prepare}" "${dxil_prepare_reflection}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
            COMMAND "${slangc}" "${source}" ${dxil_arguments}
                -entry prepare_avbd_kernel
                -o "${dxil_prepare}"
                -reflection-json "${dxil_prepare_reflection}"
            DEPENDS ${slang_dependencies} "${slangc}"
            COMMENT "Compiling shared AVBD prepare schedule to DXIL"
            COMMAND_EXPAND_LISTS VERBATIM)
        add_custom_command(
            OUTPUT "${dxil_solve}" "${dxil_solve_reflection}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
            COMMAND "${slangc}" "${source}" ${dxil_arguments}
                -entry solve_avbd_kernel
                -o "${dxil_solve}"
                -reflection-json "${dxil_solve_reflection}"
            DEPENDS ${slang_dependencies} "${slangc}"
            COMMENT "Compiling shared AVBD solve schedule to DXIL"
            COMMAND_EXPAND_LISTS VERBATIM)
        add_custom_target(parallel-mater-slang-avbd-dxil
            DEPENDS "${dxil_prepare}" "${dxil_prepare_reflection}"
                    "${dxil_solve}" "${dxil_solve_reflection}")
        list(APPEND slang_outputs
            "${dxil_prepare}" "${dxil_prepare_reflection}"
            "${dxil_solve}" "${dxil_solve_reflection}")
        set(PARALLEL_MATER_SLANG_AVBD_PREPARE_DXIL
            "${dxil_prepare}" PARENT_SCOPE)
        set(PARALLEL_MATER_SLANG_AVBD_SOLVE_DXIL
            "${dxil_solve}" PARENT_SCOPE)
        set(PARALLEL_MATER_SLANG_AVBD_PREPARE_DXIL_REFLECTION
            "${dxil_prepare_reflection}" PARENT_SCOPE)
        set(PARALLEL_MATER_SLANG_AVBD_SOLVE_DXIL_REFLECTION
            "${dxil_solve_reflection}" PARENT_SCOPE)
    endif()

    if(CMAKE_CUDA_COMPILER)
        set(ptx "${generated_dir}/parallel_mater_geometry_constraints.ptx")
        if(CMAKE_CUDA_ARCHITECTURES)
            set(ptx_architectures ${CMAKE_CUDA_ARCHITECTURES})
            list(GET ptx_architectures 0 ptx_architecture)
            string(REGEX REPLACE "-(real|virtual)$" "" ptx_architecture
                "${ptx_architecture}")
        else()
            set(ptx_architecture 52)
        endif()
        get_filename_component(cuda_host_compiler_dir
            "${CMAKE_CXX_COMPILER}" DIRECTORY)
        add_custom_command(
            OUTPUT "${ptx}"
            COMMAND "${CMAKE_CUDA_COMPILER}"
                --ptx --std=c++17
                --gpu-architecture=compute_${ptx_architecture}
                --compiler-bindir "${cuda_host_compiler_dir}"
                "${cuda_source}" -o "${ptx}"
            DEPENDS "${cuda_source}" "${CMAKE_CUDA_COMPILER}"
            COMMENT "Compiling the Slang rigid constraint CUDA conformance kernel"
            VERBATIM)
        list(APPEND slang_outputs "${ptx}")
        set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_PTX
            "${ptx}" PARENT_SCOPE)
    endif()

    add_custom_target(parallel-mater-slang-geometry-constraints ALL
                      DEPENDS ${slang_outputs})
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_CUDA_SOURCE
        "${cuda_source}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_CUDA_REFLECTION
        "${cuda_reflection}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_METAL_SOURCE
        "${metal_source}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_METAL_REFLECTION
        "${metal_reflection}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_HLSL_SOURCE
        "${hlsl_source}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_HLSL_REFLECTION
        "${hlsl_reflection}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_WGSL_SOURCE
        "${wgsl_source}" PARENT_SCOPE)
    set(PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_WGSL_REFLECTION
        "${wgsl_reflection}" PARENT_SCOPE)
endfunction()
