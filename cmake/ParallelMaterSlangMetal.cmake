# SPDX-License-Identifier: MIT
# Opt-in development driver. Only implemented rigid capabilities are advertised.
set(PM_METAL_UNSUPPORTED_HEADERS
    src/simulation/cloth_dispatch.hpp
    src/simulation/fluid_cloth_dispatch.hpp
    src/simulation/fluid_dispatch.hpp
    src/simulation/fluid_rope_dispatch.hpp
    src/simulation/fluid_smoke_dispatch.hpp
    src/simulation/fluid_soft_body_dispatch.hpp
    src/simulation/geometry_cloth_dispatch.hpp
    src/simulation/geometry_fluid_dispatch.hpp
    src/simulation/geometry_smoke_dispatch.hpp
    src/simulation/geometry_soft_body_dispatch.hpp
    src/simulation/rope_coupling_dispatch.hpp
    src/simulation/rope_dispatch.hpp
    src/simulation/smoke_coupling_dispatch.hpp
    src/simulation/smoke_grid_dispatch.hpp
    src/simulation/soft_body_cloth_dispatch.hpp
    src/simulation/soft_body_dispatch.hpp)
set(PM_METAL_STUB_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/generated/slang-metal-preview/unsupported.cpp")
add_custom_command(OUTPUT "${PM_METAL_STUB_SOURCE}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_dispatch_stubs.py"
        --output "${PM_METAL_STUB_SOURCE}" ${PM_METAL_UNSUPPORTED_HEADERS}
    DEPENDS tools/generate_cpu_dispatch_stubs.py ${PM_METAL_UNSUPPORTED_HEADERS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" VERBATIM)
set_source_files_properties("${PM_METAL_STUB_SOURCE}" PROPERTIES GENERATED TRUE COMPILE_OPTIONS "-w")
add_library(parallel-mater-driver-slang-metal SHARED
    src/api/instance.cpp src/api/device.cpp src/api/memory.cpp
    src/api/resources.cpp src/api/domains.cpp src/api/pipeline.cpp
    src/api/commands.cpp src/api/synchronization.cpp src/api/queries.cpp src/api/interop.cpp
    src/driver/slang/entry.cpp src/driver/metal/schedule_context.cpp
    src/device/metal_device.mm src/device/submission.cpp
    src/simulation/metal_runtime.cpp src/simulation/metal_geometry_dispatch.cpp
    src/simulation/metal_geometry_constraints_dispatch.cpp src/simulation/metal_primitives.cpp
    src/simulation/metal_exports.cpp src/simulation/memory_arena.cpp "${PM_METAL_STUB_SOURCE}")
target_compile_features(parallel-mater-driver-slang-metal PRIVATE cxx_std_20)
target_compile_definitions(parallel-mater-driver-slang-metal PRIVATE
    PARALLEL_MATER_DRIVER_IMPLEMENTATION=1 PARALLEL_MATER_METAL_DRIVER=1)
target_include_directories(parallel-mater-driver-slang-metal PRIVATE
    src/device/metal_compat src include "${PARALLEL_MATER_API_GENERATED_DIR}")
target_compile_options(parallel-mater-driver-slang-metal PRIVATE
    -Wall -Wextra -Wno-unused-parameter "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(parallel-mater-driver-slang-metal PRIVATE
    parallel-mater-device-metal parallel-mater-host-preparation Threads::Threads "-framework Metal")
set_target_properties(parallel-mater-driver-slang-metal PROPERTIES
    OUTPUT_NAME parallel_mater_driver_slang_metal CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES OBJCXX_STANDARD 20)
file(GENERATE OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-metal.json" CONTENT
"{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-driver-slang-metal>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
file(GENERATE OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-metal.json" CONTENT
"{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"../../Frameworks/$<TARGET_FILE_NAME:parallel-mater-driver-slang-metal>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
if(BUILD_TESTING)
    add_executable(parallel-mater-slang-metal-command-tests tests/api/native_command_api_tests.cpp)
    target_compile_features(parallel-mater-slang-metal-command-tests PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-slang-metal-command-tests PRIVATE
        PM_TEST_NATIVE_BACKEND=PM_NATIVE_BACKEND_METAL PM_TEST_METAL_PREVIEW=1)
    target_link_libraries(parallel-mater-slang-metal-command-tests PRIVATE parallel-mater)
    add_dependencies(parallel-mater-slang-metal-command-tests parallel-mater-driver-slang-metal)
    add_test(NAME parallel-mater-slang-metal-command-tests COMMAND parallel-mater-slang-metal-command-tests)
    set_tests_properties(parallel-mater-slang-metal-command-tests PROPERTIES TIMEOUT 120
        ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-metal.json")
    add_executable(parallel-mater-slang-metal-async-tests tests/api/async_submission_tests.cpp)
    target_compile_features(parallel-mater-slang-metal-async-tests PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-slang-metal-async-tests PRIVATE PM_TEST_METAL_PREVIEW=1)
    target_link_libraries(parallel-mater-slang-metal-async-tests PRIVATE parallel-mater)
    add_dependencies(parallel-mater-slang-metal-async-tests parallel-mater-driver-slang-metal)
    add_test(NAME parallel-mater-slang-metal-async-tests COMMAND parallel-mater-slang-metal-async-tests)
    set_tests_properties(parallel-mater-slang-metal-async-tests PROPERTIES TIMEOUT 120
        ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-metal.json")
endif()
