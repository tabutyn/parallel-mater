# SPDX-License-Identifier: MIT
# Serial host adapter for the shared Slang rigid-body schedule.  This is the
# first macOS driver rung: loader/ABI complete and equation-identical, while
# capabilities without host dispatch wrappers remain unadvertised.

get_filename_component(PARALLEL_MATER_SLANG_BIN_DIR
    "${PARALLEL_MATER_SLANGC_EXECUTABLE}" DIRECTORY)
get_filename_component(PARALLEL_MATER_SLANG_PACKAGE_DIR
    "${PARALLEL_MATER_SLANG_BIN_DIR}" DIRECTORY)
set(PARALLEL_MATER_SLANG_CPP_INCLUDE_DIR
    "${PARALLEL_MATER_SLANG_PACKAGE_DIR}/include")
if(NOT EXISTS "${PARALLEL_MATER_SLANG_CPP_INCLUDE_DIR}/slang-cpp-prelude.h")
    message(FATAL_ERROR
        "Slang C++ prelude is missing beside ${PARALLEL_MATER_SLANGC_EXECUTABLE}")
endif()

set(PARALLEL_MATER_SLANG_CPU_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/slang-cpu")
set(PARALLEL_MATER_GEOMETRY_CPU_SOURCE
    "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}/parallel_mater_geometry_runtime.cpp")
set(PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CPU_SOURCE
    "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}/parallel_mater_geometry_constraints_runtime.cpp")
set(PARALLEL_MATER_QUERIES_CPU_SOURCE
    "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}/parallel_mater_queries_runtime.cpp")
set(PARALLEL_MATER_CPU_STUB_SOURCE
    "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}/parallel_mater_unadvertised_dispatch.cpp")
set(PARALLEL_MATER_CPU_STUB_HEADERS
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

add_custom_command(
    OUTPUT "${PARALLEL_MATER_GEOMETRY_CPU_SOURCE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}"
    COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry.slang"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
            -DPM_TARGET_CPU=1
            -entry pm_integrate_rigid_bodies
            -entry pm_compute_rigid_world_bounds
            -entry pm_broad_phase_rigid_pairs
            -entry pm_generate_rigid_leaf_pairs
            -entry pm_evaluate_rigid_leaf_pairs
            -entry pm_reduce_rigid_leaf_manifolds
            -entry pm_finalize_rigid_contact_manifolds
            -entry pm_clear_rigid_inputs
            -entry pm_capture_rigid_inputs
            -entry pm_prepare_parallel_contact_events
            -entry pm_load_rigid_contact_cache
            -entry pm_save_rigid_contact_cache
            -entry pm_gather_rigid_debug_samples
            -entry pm_clamp_rigid_speeds
            -entry pm_build_fixed_collision_groups
            -target cpp -fp-mode precise -O2 -line-directive-mode none
            -o "${PARALLEL_MATER_GEOMETRY_CPU_SOURCE}"
    DEPENDS src/slang/geometry.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
    COMMENT "Compiling shared rigid geometry kernels from Slang to C++"
    VERBATIM)

add_custom_command(
    OUTPUT "${PARALLEL_MATER_CPU_STUB_SOURCE}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_dispatch_stubs.py"
            --output "${PARALLEL_MATER_CPU_STUB_SOURCE}"
            ${PARALLEL_MATER_CPU_STUB_HEADERS}
    DEPENDS tools/generate_cpu_dispatch_stubs.py
            ${PARALLEL_MATER_CPU_STUB_HEADERS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating guarded stubs for unadvertised CPU domains"
    VERBATIM)

add_custom_command(
    OUTPUT "${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CPU_SOURCE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}"
    COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry_constraints.slang"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
            -DPM_TARGET_CPU=1 -DPM_GEOMETRY_CONSTRAINTS_ADAPTER=1
            -entry prepare_avbd_kernel
            -entry solve_avbd_kernel
            -target cpp -fp-mode precise -O2 -line-directive-mode none
            -o "${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CPU_SOURCE}"
    DEPENDS src/slang/geometry_constraints.slang
            src/slang/geometry_constraints_solve.slang
            src/slang/avbd_stages.inc
            src/slang/constraint_math.slang src/slang/geometry.slang
            src/slang/types.slang src/slang/math.slang
            src/slang/device_intrinsics.slang
            "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
    COMMENT "Compiling shared rigid constraint solver from Slang to C++"
    VERBATIM)

add_custom_command(
    OUTPUT "${PARALLEL_MATER_QUERIES_CPU_SOURCE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}"
    COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/slang/queries.slang"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/slang"
            -DPM_TARGET_CPU=1
            -entry export_fluid_particles
            -entry export_cloth_vertices
            -entry export_surface_positions
            -entry export_rope_nodes
            -entry export_soft_body_nodes
            -entry export_smoke_particles
            -entry query_rigid_hit_box_kernel
            -entry query_particle_hit_box_kernel
            -target cpp -fp-mode precise -O2 -line-directive-mode none
            -o "${PARALLEL_MATER_QUERIES_CPU_SOURCE}"
    DEPENDS src/slang/queries.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
    COMMENT "Compiling shared query/export kernels from Slang to C++"
    VERBATIM)

set_source_files_properties(
    "${PARALLEL_MATER_GEOMETRY_CPU_SOURCE}"
    "${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CPU_SOURCE}"
    "${PARALLEL_MATER_QUERIES_CPU_SOURCE}"
    "${PARALLEL_MATER_CPU_STUB_SOURCE}"
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
set_source_files_properties("${PARALLEL_MATER_CPU_STUB_SOURCE}"
    PROPERTIES HEADER_FILE_ONLY FALSE COMPILE_OPTIONS "-w")
set_source_files_properties(
    src/simulation/cpu_geometry_dispatch.cpp
    src/simulation/cpu_geometry_constraints_dispatch.cpp
    src/simulation/cpu_exports.cpp
    PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter;-Wno-unused-variable")

add_library(parallel-mater-driver-slang-cpu SHARED
    src/api/instance.cpp
    src/api/device.cpp
    src/api/memory.cpp
    src/api/resources.cpp
    src/api/domains.cpp
    src/api/pipeline.cpp
    src/api/commands.cpp
    src/api/synchronization.cpp
    src/api/queries.cpp
    src/api/interop.cpp
    src/driver/slang/entry.cpp
    src/device/cpu_device.cpp
    src/simulation/cpu_runtime.cpp
    src/simulation/cpu_geometry_dispatch.cpp
    src/simulation/cpu_geometry_constraints_dispatch.cpp
    src/simulation/cpu_primitives.cpp
    src/simulation/cpu_exports.cpp
    src/simulation/memory_arena.cpp
    "${PARALLEL_MATER_GEOMETRY_CPU_SOURCE}"
    "${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CPU_SOURCE}"
    "${PARALLEL_MATER_QUERIES_CPU_SOURCE}")
target_sources(parallel-mater-driver-slang-cpu PRIVATE
    "${PARALLEL_MATER_CPU_STUB_SOURCE}")

target_compile_features(parallel-mater-driver-slang-cpu PRIVATE cxx_std_20)
target_compile_definitions(parallel-mater-driver-slang-cpu PRIVATE
    PARALLEL_MATER_DRIVER_IMPLEMENTATION=1
    PARALLEL_MATER_SLANG_CPU_DRIVER=1)
target_include_directories(parallel-mater-driver-slang-cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device/cpu_compat
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}
        ${PARALLEL_MATER_SLANG_CPP_INCLUDE_DIR}
        ${PARALLEL_MATER_API_GENERATED_DIR})
target_link_libraries(parallel-mater-driver-slang-cpu PRIVATE
    parallel-mater-host-preparation)
parallel_mater_enable_warnings(parallel-mater-driver-slang-cpu)
set_target_properties(parallel-mater-driver-slang-cpu PROPERTIES
    OUTPUT_NAME parallel_mater_driver_slang_cpu
    EXPORT_NAME slang_cpu_driver
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON)
add_library(ParallelMater::slang_cpu_driver ALIAS
            parallel-mater-driver-slang-cpu)
list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater-driver-slang-cpu)

file(GENERATE
    OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cpu.json"
    CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-driver-slang-cpu>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
file(GENERATE
    OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-cpu.json"
    CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"../../Frameworks/$<TARGET_FILE_NAME:parallel-mater-driver-slang-cpu>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")

if(BUILD_TESTING)
    add_executable(parallel-mater-slang-cpu-driver-abi-tests
        tests/api/driver_abi_tests.cpp)
    target_include_directories(parallel-mater-slang-cpu-driver-abi-tests
        PRIVATE include)
    target_compile_features(parallel-mater-slang-cpu-driver-abi-tests
        PRIVATE cxx_std_20)
    target_link_libraries(parallel-mater-slang-cpu-driver-abi-tests
        PRIVATE ${CMAKE_DL_LIBS})
    add_test(NAME parallel-mater-slang-cpu-driver-abi-tests
        COMMAND parallel-mater-slang-cpu-driver-abi-tests
            $<TARGET_FILE:parallel-mater-driver-slang-cpu>
            $<TARGET_FILE:parallel-mater>)
    set_tests_properties(parallel-mater-slang-cpu-driver-abi-tests PROPERTIES
        LABELS "api;driver;slang;macos")
endif()
