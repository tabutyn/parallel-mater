# SPDX-License-Identifier: MIT
find_package(Vulkan REQUIRED)
find_program(PARALLEL_MATER_SPIRV_VAL_EXECUTABLE NAMES spirv-val)
include(${CMAKE_CURRENT_LIST_DIR}/SlangKernelManifest.cmake)
set(vulkan_generated "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan")
file(MAKE_DIRECTORY "${vulkan_generated}")
file(GLOB_RECURSE vulkan_shader_dependencies CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slang" "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/*.slangh")
set(vulkan_sources)
foreach(module geometry geometry_constraints primitives queries fluid geometry_fluid)
    parallel_mater_slang_entry_arguments(${module} CUDA entries)
    parallel_mater_slang_definition_arguments(${module} CUDA definitions)
    set(wrapper "src/simulation/${module}_dispatch.cu")
    if(module STREQUAL "queries")
        set(wrapper src/simulation/exports.cu)
    elseif(module STREQUAL "primitives")
        set(wrapper src/simulation/primitives.cu)
    endif()
    set(base "${vulkan_generated}/${module}")
    add_custom_command(OUTPUT "${base}.cu" "${base}.json" "${base}.spv"
        COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/${module}.slang"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang" ${entries} ${definitions}
            -target cuda -fp-mode precise -line-directive-mode none -o "${base}.cu"
        COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/${module}.slang"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang" ${entries} ${definitions}
            -DPM_TARGET_VULKAN=1 -DPM_VULKAN_VARIABLE_GROUP_SIZE=1 -target spirv -profile sm_6_5 -fp-mode precise -O2
            -fvk-use-scalar-layout -fvk-use-entrypoint-name -separate-debug-info
            -reflection-json "${base}.json" -o "${base}.spv"
        DEPENDS ${vulkan_shader_dependencies} "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
        COMMENT "Compiling shared Slang ${module} for Vulkan" COMMAND_EXPAND_LISTS VERBATIM)
    add_custom_command(OUTPUT "${base}.cpp"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_vulkan_dispatch.py"
            --cuda "${base}.cu" --spirv "${base}.spv" --reflection "${base}.json"
            --wrapper "${CMAKE_CURRENT_SOURCE_DIR}/${wrapper}" --module "${module}" --output "${base}.cpp"
        DEPENDS "${base}.cu" "${base}.json" "${base}.spv" "${wrapper}" tools/generate_vulkan_dispatch.py
        COMMENT "Binding shared Slang ${module} to Vulkan" COMMAND_EXPAND_LISTS VERBATIM)
    list(APPEND vulkan_sources "${base}.cpp")
    if(BUILD_TESTING AND PARALLEL_MATER_SPIRV_VAL_EXECUTABLE)
        add_test(NAME parallel-mater-vulkan-spirv-${module}
            COMMAND "${PARALLEL_MATER_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.3 --scalar-block-layout "${base}.spv")
    endif()
endforeach()
set(vulkan_stub_headers
    src/simulation/cloth_dispatch.hpp src/simulation/fluid_cloth_dispatch.hpp
    src/simulation/fluid_rope_dispatch.hpp src/simulation/fluid_smoke_dispatch.hpp
    src/simulation/fluid_soft_body_dispatch.hpp src/simulation/geometry_cloth_dispatch.hpp
    src/simulation/geometry_smoke_dispatch.hpp src/simulation/geometry_soft_body_dispatch.hpp
    src/simulation/rope_coupling_dispatch.hpp src/simulation/rope_dispatch.hpp
    src/simulation/smoke_coupling_dispatch.hpp src/simulation/smoke_grid_dispatch.hpp
    src/simulation/soft_body_cloth_dispatch.hpp src/simulation/soft_body_dispatch.hpp)
add_custom_command(OUTPUT "${vulkan_generated}/unadvertised.cpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_dispatch_stubs.py"
        --output "${vulkan_generated}/unadvertised.cpp" ${vulkan_stub_headers}
    DEPENDS tools/generate_cpu_dispatch_stubs.py ${vulkan_stub_headers}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" VERBATIM)
add_library(parallel-mater-driver-slang-vulkan SHARED
    src/api/instance.cpp src/api/device.cpp src/api/memory.cpp src/api/resources.cpp
    src/api/domains.cpp src/api/pipeline.cpp src/api/commands.cpp src/api/synchronization.cpp
    src/api/queries.cpp src/api/interop.cpp src/driver/slang/entry.cpp
    src/driver/vulkan/schedule_context.cpp src/device/vulkan_device.cpp
    src/device/submission.cpp src/simulation/vulkan_runtime.cpp src/simulation/memory_arena.cpp
    ${vulkan_sources} "${vulkan_generated}/unadvertised.cpp")
target_compile_features(parallel-mater-driver-slang-vulkan PRIVATE cxx_std_20)
target_compile_definitions(parallel-mater-driver-slang-vulkan PRIVATE
    PARALLEL_MATER_DRIVER_IMPLEMENTATION=1 PARALLEL_MATER_VULKAN_DRIVER=1)
target_include_directories(parallel-mater-driver-slang-vulkan PRIVATE
    include src src/simulation src/device/vulkan_compat "${PARALLEL_MATER_API_GENERATED_DIR}")
target_link_libraries(parallel-mater-driver-slang-vulkan PRIVATE
    parallel-mater-host-preparation Threads::Threads Vulkan::Vulkan)
set_target_properties(parallel-mater-driver-slang-vulkan PROPERTIES
    OUTPUT_NAME parallel_mater_driver_slang_vulkan EXPORT_NAME slang_vulkan_driver
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES POSITION_INDEPENDENT_CODE ON)
add_library(ParallelMater::slang_vulkan_driver ALIAS parallel-mater-driver-slang-vulkan)
if(TARGET parallel-mater-gallery)
    add_dependencies(parallel-mater-gallery parallel-mater-driver-slang-vulkan)
endif()
list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater-driver-slang-vulkan)
file(GENERATE OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json"
    CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-driver-slang-vulkan>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
file(GENERATE OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-vulkan.json"
    CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"../$<TARGET_FILE_NAME:parallel-mater-driver-slang-vulkan>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
if(BUILD_TESTING)
    add_executable(parallel-mater-vulkan-command-api-tests tests/api/d3d12_command_api_tests.cpp)
    target_compile_features(parallel-mater-vulkan-command-api-tests PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-vulkan-command-api-tests PRIVATE PM_TEST_VULKAN=1)
    target_link_libraries(parallel-mater-vulkan-command-api-tests PRIVATE ParallelMater::parallel_mater)
    add_dependencies(parallel-mater-vulkan-command-api-tests parallel-mater-driver-slang-vulkan)
    add_test(NAME parallel-mater-vulkan-command-api-tests COMMAND parallel-mater-vulkan-command-api-tests)
    set_tests_properties(parallel-mater-vulkan-command-api-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180
        ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json")
    add_executable(parallel-mater-vulkan-async-tests tests/api/async_submission_tests.cpp)
    target_compile_features(parallel-mater-vulkan-async-tests PRIVATE cxx_std_20)
    target_link_libraries(parallel-mater-vulkan-async-tests PRIVATE ParallelMater::parallel_mater)
    add_dependencies(parallel-mater-vulkan-async-tests parallel-mater-driver-slang-vulkan)
    add_test(NAME parallel-mater-vulkan-async-tests COMMAND parallel-mater-vulkan-async-tests)
    set_tests_properties(parallel-mater-vulkan-async-tests PROPERTIES TIMEOUT 180
        ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json")
    if(PARALLEL_MATER_BUILD_GALLERY)
        add_executable(parallel-mater-vulkan-liquid-tests tests/gallery/mobile_liquid_tests.cpp)
        target_link_libraries(parallel-mater-vulkan-liquid-tests PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-vulkan-liquid-tests PRIVATE
            PM_TEST_BACKEND="vulkan"
            PM_PROFILE_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
            PM_SCENE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets")
        add_dependencies(parallel-mater-vulkan-liquid-tests parallel-mater-driver-slang-vulkan parallel-mater-layer-validation)
        add_test(NAME parallel-mater-vulkan-liquid-tests COMMAND parallel-mater-vulkan-liquid-tests)
        set_tests_properties(parallel-mater-vulkan-liquid-tests PROPERTIES TIMEOUT 180
            ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_PATH=${PARALLEL_MATER_BUILD_MANIFEST_DIR}")
    endif()
endif()

if(BUILD_TESTING)
    foreach(test parallel-mater-vulkan-command-api-tests parallel-mater-vulkan-async-tests
                 parallel-mater-vulkan-liquid-tests)
        if(TEST ${test})
            set_tests_properties(${test} PROPERTIES
                FAIL_REGULAR_EXPRESSION "Validation Error;VUID-")
        endif()
    endforeach()
endif()
