# SPDX-License-Identifier: MIT

include(${CMAKE_CURRENT_LIST_DIR}/SlangKernelManifest.cmake)

function(parallel_mater_add_slang_kernel_mirrors)
    set(slangc "${PARALLEL_MATER_SLANGC_EXECUTABLE}")
    if(NOT slangc)
        message(FATAL_ERROR
            "parallel_mater_require_slang() must run before mirror generation")
    endif()

    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/slang/mirrors")
    file(GLOB_RECURSE slang_dependencies CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slang"
         "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slangh")
    set(outputs)

    set(nvrtc_path)
    if(TARGET CUDA::nvrtc)
        if(WIN32)
            file(GLOB nvrtc_candidates LIST_DIRECTORIES false
                 "${CUDAToolkit_BIN_DIR}/nvrtc64_*.dll")
            list(FILTER nvrtc_candidates EXCLUDE REGEX "\\.alt\\.dll$")
            list(SORT nvrtc_candidates COMPARE NATURAL ORDER DESCENDING)
            list(LENGTH nvrtc_candidates nvrtc_candidate_count)
            if(nvrtc_candidate_count GREATER 0)
                list(GET nvrtc_candidates 0 nvrtc_dll)
                cmake_path(GET nvrtc_dll PARENT_PATH nvrtc_directory)
                cmake_path(GET nvrtc_dll STEM nvrtc_name)
                set(nvrtc_path "${nvrtc_directory}/${nvrtc_name}")
            else()
                message(WARNING
                    "NVRTC runtime not found in ${CUDAToolkit_BIN_DIR}; "
                    "Slang PTX mirror validation is disabled")
            endif()
        else()
            set(nvrtc_path "$<TARGET_FILE_DIR:CUDA::nvrtc>/nvrtc")
        endif()
    endif()

    foreach(module IN LISTS PARALLEL_MATER_SLANG_KERNEL_MODULES)
        parallel_mater_slang_entry_arguments("${module}" MIRROR entry_arguments)
        parallel_mater_slang_definition_arguments(
            "${module}" MIRROR definition_arguments)
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/${module}.slang")

        set(cuda_output "${generated_dir}/${module}.cu")
        add_custom_command(
            OUTPUT "${cuda_output}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
            COMMAND "${slangc}" "${source}"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                -target cuda -fp-mode precise -warnings-as-errors all
                ${definition_arguments} ${entry_arguments} -o "${cuda_output}"
            COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${cuda_output}"
            DEPENDS ${slang_dependencies} "${slangc}"
            COMMENT "Validating ${module}.slang as CUDA"
            COMMAND_EXPAND_LISTS VERBATIM)
        list(APPEND outputs "${cuda_output}")

        set(spirv_output "${generated_dir}/${module}.spv")
        add_custom_command(
            OUTPUT "${spirv_output}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
            COMMAND "${slangc}" "${source}"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                -target spirv -profile sm_6_5 -fp-mode precise
                -warnings-as-errors all -DPM_TARGET_VULKAN=1
                ${definition_arguments} ${entry_arguments} -o "${spirv_output}"
            COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${spirv_output}"
            DEPENDS ${slang_dependencies} "${slangc}"
            COMMENT "Validating ${module}.slang as Vulkan SPIR-V"
            COMMAND_EXPAND_LISTS VERBATIM)
        list(APPEND outputs "${spirv_output}")

        if(nvrtc_path)
            set(ptx_output "${generated_dir}/${module}.ptx")
            add_custom_command(
                OUTPUT "${ptx_output}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
                COMMAND "${slangc}" "${source}"
                    -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                    -target ptx -fp-mode precise -warnings-as-errors all
                    -nvrtc-path "${nvrtc_path}"
                    ${definition_arguments} ${entry_arguments}
                    -o "${ptx_output}"
                COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${ptx_output}"
                DEPENDS ${slang_dependencies} "${slangc}" CUDA::nvrtc
                COMMENT "Validating ${module}.slang as PTX"
                COMMAND_EXPAND_LISTS VERBATIM)
            list(APPEND outputs "${ptx_output}")
        endif()
    endforeach()

    add_custom_target(parallel-mater-slang-kernel-mirrors ALL
        DEPENDS ${outputs})
endfunction()
