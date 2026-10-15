# SPDX-License-Identifier: MIT
include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/SlangKernelManifest.cmake)

# Compile each entry separately. Slang 2026.18's Metal global-input lowering
# can leave later entries' thread IDs uninitialized in multi-entry output.
# These are shader gates, not an installable/qualified physics driver.
function(parallel_mater_add_slang_metal_kernels)
    set(generated_dir "${PARALLEL_MATER_SLANG_GENERATED_DIR}/metal")
    file(GLOB_RECURSE dependencies CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slang")
    if(APPLE)
        find_program(PARALLEL_MATER_XCRUN_EXECUTABLE xcrun REQUIRED)
    endif()
    set(outputs)
    set(records "")
    set(modules ${PARALLEL_MATER_SLANG_KERNEL_MODULES})
    if(BUILD_TESTING)
        list(APPEND modules metal_device_contract)
        set(PARALLEL_MATER_SLANG_metal_device_contract_ENTRIES
            metal_device_contract)
    endif()
    foreach(module IN LISTS modules)
        set(source "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/${module}.slang")
        if(module STREQUAL "metal_device_contract")
            set(source "${CMAKE_CURRENT_SOURCE_DIR}/tests/solver/metal_device_contract.slang")
        endif()
        parallel_mater_slang_definition_arguments("${module}" METAL definitions)
        foreach(entry IN LISTS PARALLEL_MATER_SLANG_${module}_ENTRIES)
            set(base "${generated_dir}/${module}/${entry}")
            add_custom_command(
                OUTPUT "${base}.metal" "${base}.json"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}/${module}"
                COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}" "${source}"
                    -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
                    ${definitions} -DPM_TARGET_METAL=1
                    -DPM_DEVICE_LEAF_MANIFOLDS=1
                    -DPM_AVBD_PREPARE_SMALL_CACHE=1
                    -DPM_SOFT_NEIGHBOR_GROUP_SIZE=32
                    -entry "${entry}" -stage compute -target metal
                    -fp-mode precise -warnings-as-errors all
                    -line-directive-mode none
                    -o "${base}.metal" -reflection-json "${base}.json"
                COMMAND ${CMAKE_COMMAND} -E touch_nocreate
                    "${base}.metal" "${base}.json"
                DEPENDS ${dependencies} "${source}"
                    "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
                COMMENT "Compiling shared ${module}.${entry} to Metal"
                COMMAND_EXPAND_LISTS VERBATIM)
            list(APPEND outputs "${base}.metal" "${base}.json")
            string(APPEND records "${module}|${entry}|${base}\n")
            if(APPLE)
                add_custom_command(
                    OUTPUT "${base}.air" "${base}.metallib"
                    COMMAND "${PARALLEL_MATER_XCRUN_EXECUTABLE}" --sdk macosx metal
                        -std=metal3.0 -Werror -fno-fast-math -ffp-contract=off
                        # Slang retains unused atomic fetch results in MSL.
                        # Keep all other native diagnostics as errors.
                        -Wno-unused-variable
                        "-fmodules-cache-path=${generated_dir}/module-cache"
                        -c "${base}.metal" -o "${base}.air"
                    COMMAND "${PARALLEL_MATER_XCRUN_EXECUTABLE}" --sdk macosx metallib
                        "${base}.air" -o "${base}.metallib"
                    DEPENDS "${base}.metal" "${PARALLEL_MATER_XCRUN_EXECUTABLE}"
                    COMMENT "Native Metal compile/link: ${module}.${entry}"
                    VERBATIM)
                list(APPEND outputs "${base}.air" "${base}.metallib")
            endif()
        endforeach()
    endforeach()
    file(GENERATE OUTPUT "${generated_dir}/manifest.txt" CONTENT "${records}")
    add_custom_target(parallel-mater-slang-metal-kernels ALL DEPENDS ${outputs})
    set(PARALLEL_MATER_SLANG_METAL_MANIFEST
        "${generated_dir}/manifest.txt" PARENT_SCOPE)
endfunction()
