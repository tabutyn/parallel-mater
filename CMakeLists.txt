cmake_minimum_required(VERSION 3.25)

project(ParallelMater VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)
include(CTest)
include(CheckLanguage)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(NAME parallel-mater-generated-api-check
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/build.py"
            --check-generated)
set_tests_properties(parallel-mater-generated-api-check PROPERTIES
    LABELS "api;registry")

function(parallel_mater_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
    endif()
endfunction()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(PARALLEL_MATER_LIBRARY_ARCH 64)
else()
    set(PARALLEL_MATER_LIBRARY_ARCH 32)
endif()

set(PARALLEL_MATER_API_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/api")
file(MAKE_DIRECTORY "${PARALLEL_MATER_API_GENERATED_DIR}")
execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/registry/generate.py"
            --output "${PARALLEL_MATER_API_GENERATED_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)

if(NOT APPLE AND NOT WIN32)
    set(PARALLEL_MATER_CUDA_DEFAULT ON)
else()
    set(PARALLEL_MATER_CUDA_DEFAULT OFF)
endif()

option(PARALLEL_MATER_BUILD_CUDA
       "Build the CUDA Slang device adapter" ${PARALLEL_MATER_CUDA_DEFAULT})
option(PARALLEL_MATER_BUILD_OPTIX_GALLERY
       "Build the optional OptiX and glTF gallery" OFF)
option(PARALLEL_MATER_BUILD_MACOS_GALLERY
       "Build the loader-native macOS Gallery application" ${APPLE})
option(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER
       "Build the serial Slang CPU reference driver" ${APPLE})
option(PARALLEL_MATER_BUILD_BENCHMARKS
       "Build local performance benchmark executables" OFF)
if(PARALLEL_MATER_BUILD_MACOS_GALLERY AND
   PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    message(FATAL_ERROR
        "Select either the macOS software Gallery or the CUDA/OptiX Gallery")
endif()
if(PARALLEL_MATER_BUILD_METAL OR PARALLEL_MATER_BUILD_METAL_GALLERY)
    message(FATAL_ERROR
        "Backend-specific physics implementations are retired. Use the shared command API "
        "with a validated Slang device adapter.")
endif()
option(PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS
       "Build and validate the portable Slang rigid constraint core"
       ${BUILD_TESTING})
option(PARALLEL_MATER_BUILD_SLANG_MIRRORS
       "Build and validate every CUDA physics kernel mirror" ${BUILD_TESTING})
option(PARALLEL_MATER_SLANG_TEST_BUILD
       "Label the gallery as a production Slang runtime test build" OFF)

if(WIN32 AND PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    set(PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS ON CACHE BOOL
        "Build and validate the portable Slang rigid constraint core" FORCE)
endif()

if(PARALLEL_MATER_SLANG_TEST_BUILD)
    set(PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS ON CACHE BOOL
        "Build and validate the portable Slang rigid constraint core" FORCE)
    set(PARALLEL_MATER_BUILD_SLANG_MIRRORS ON CACHE BOOL
        "Build and validate every CUDA physics kernel mirror" FORCE)
    add_compile_definitions(PARALLEL_MATER_SLANG_TEST_BUILD=1)
endif()

# Ship prebuilt icons so ordinary builds need no image conversion tools.
if(WIN32 AND PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    enable_language(RC)
    set(PARALLEL_MATER_GALLERY_ICON
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/branding/ParallelMater.ico")
    set(PARALLEL_MATER_GALLERY_ICON_RESOURCE
        "${CMAKE_CURRENT_BINARY_DIR}/generated/parallel_mater_gallery_icon.rc")
    configure_file(examples/support/gallery_icon.rc.in
                   "${PARALLEL_MATER_GALLERY_ICON_RESOURCE}" @ONLY)
    set_source_files_properties("${PARALLEL_MATER_GALLERY_ICON_RESOURCE}"
        PROPERTIES OBJECT_DEPENDS "${PARALLEL_MATER_GALLERY_ICON}")
endif()

if(PARALLEL_MATER_BUILD_OPTIX_GALLERY AND NOT PARALLEL_MATER_BUILD_CUDA)
    message(FATAL_ERROR "The OptiX gallery requires PARALLEL_MATER_BUILD_CUDA")
endif()
if(PARALLEL_MATER_BUILD_BENCHMARKS AND
   NOT PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    message(FATAL_ERROR
            "Rigid scene benchmarks require PARALLEL_MATER_BUILD_OPTIX_GALLERY")
endif()

if(PARALLEL_MATER_BUILD_CUDA OR PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER OR
   PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS OR
   PARALLEL_MATER_BUILD_SLANG_MIRRORS)
    include(cmake/Slang.cmake)
    parallel_mater_require_slang()
endif()

# The conformance case registry intentionally has no CUDA or engine dependency.
# Every backend byte-checks and executes the same canonical inputs.
add_library(parallel-mater-conformance-registry STATIC
            conformance/v1/src/case_registry.cpp)
target_compile_features(parallel-mater-conformance-registry PUBLIC cxx_std_20)
target_include_directories(parallel-mater-conformance-registry
    PUBLIC conformance/v1/include)
if(MSVC)
    target_compile_options(parallel-mater-conformance-registry PRIVATE /W4)
else()
    target_compile_options(parallel-mater-conformance-registry
                           PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(parallel-mater-conformance-cases
               conformance/v1/src/case_files.cpp)
target_link_libraries(parallel-mater-conformance-cases
                      PRIVATE parallel-mater-conformance-registry)
if(MSVC)
    target_compile_options(parallel-mater-conformance-cases PRIVATE /W4)
else()
    target_compile_options(parallel-mater-conformance-cases
                           PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(PARALLEL_MATER_INSTALL_TARGETS)

include(cmake/ParallelMaterLoader.cmake)

if(PARALLEL_MATER_BUILD_OPTIX_GALLERY OR
   PARALLEL_MATER_BUILD_MACOS_GALLERY OR
   (PARALLEL_MATER_BUILD_CUDA AND BUILD_TESTING))
    FetchContent_Declare(
        parallel_mater_cgltf
        GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
        GIT_TAG 85cd62382dfea638278962690cf515023f33ed00
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(parallel_mater_cgltf)
endif()

if(PARALLEL_MATER_BUILD_CUDA OR PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER OR
   PARALLEL_MATER_BUILD_MACOS_GALLERY OR
   BUILD_TESTING)
    add_library(parallel-mater-host-preparation STATIC
        src/fluid_geometry.cpp
        src/fluid_source.cpp
        src/soft_body_geometry.cpp
        src/rope_geometry.cpp)
    target_compile_features(parallel-mater-host-preparation PUBLIC cxx_std_20)
    target_include_directories(parallel-mater-host-preparation
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(parallel-mater-host-preparation PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    parallel_mater_enable_warnings(parallel-mater-host-preparation)
endif()

if(PARALLEL_MATER_BUILD_CUDA)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "PARALLEL_MATER_BUILD_CUDA is ON but no CUDA compiler was found")
endif()
enable_language(CUDA)

# CMake 3.25 assumes the removed nvToolsExt library exists when discovering
# CUDA 13. NVTX is not used by the public target, so provide the compatibility
# target before FindCUDAToolkit evaluates that deprecated component.
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()
find_package(CUDAToolkit REQUIRED)

set(PARALLEL_MATER_SLANG_GENERATED_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/generated/slang)
set(PARALLEL_MATER_FLUID_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_fluid_runtime.cu)
set(PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_geometry_constraints_runtime.cu)
set(PARALLEL_MATER_QUERIES_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_queries_runtime.cu)
set(PARALLEL_MATER_GEOMETRY_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_geometry_runtime.cu)
set(PARALLEL_MATER_PRIMITIVES_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_primitives_runtime.cu)
set(PARALLEL_MATER_CLOTH_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_cloth_runtime.cu)
set(PARALLEL_MATER_SOFT_BODY_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_soft_body_runtime.cu)
set(PARALLEL_MATER_SOFT_BODY_CLOTH_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_soft_body_cloth_runtime.cu)
set(PARALLEL_MATER_CLOTH_ROPE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_cloth_rope_runtime.cu)
set(PARALLEL_MATER_SOFT_BODY_ROPE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_soft_body_rope_runtime.cu)
set(PARALLEL_MATER_FLUID_CLOTH_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_fluid_cloth_runtime.cu)
set(PARALLEL_MATER_FLUID_ROPE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_fluid_rope_runtime.cu)
set(PARALLEL_MATER_FLUID_SMOKE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_fluid_smoke_runtime.cu)
set(PARALLEL_MATER_CLOTH_SMOKE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_cloth_smoke_runtime.cu)
set(PARALLEL_MATER_SOFT_BODY_SMOKE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_soft_body_smoke_runtime.cu)
set(PARALLEL_MATER_ROPE_SMOKE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_rope_smoke_runtime.cu)
set(PARALLEL_MATER_FLUID_SOFT_BODY_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_fluid_soft_body_runtime.cu)
set(PARALLEL_MATER_GEOMETRY_CLOTH_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_geometry_cloth_runtime.cu)
set(PARALLEL_MATER_GEOMETRY_FLUID_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_geometry_fluid_runtime.cu)
set(PARALLEL_MATER_GEOMETRY_SMOKE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_geometry_smoke_runtime.cu)
set(PARALLEL_MATER_GEOMETRY_SOFT_BODY_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_geometry_soft_body_runtime.cu)
set(PARALLEL_MATER_SMOKE_GRID_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_smoke_grid_runtime.cu)
set(PARALLEL_MATER_ROPE_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_rope_runtime.cu)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_ROPE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/rope.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry rope_advance
            -target cuda -fp-mode precise -O3 -line-directive-mode none
            -o ${PARALLEL_MATER_ROPE_CUDA_SOURCE}
    DEPENDS src/slang/rope.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry.slang src/slang/geometry_fluid.slang
            src/slang/fluid_soft_body.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared rope simulation kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_FLUID_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/fluid.slang
            -DPM_TARGET_CUDA=1
            -entry pm_fluid_cell_keys
            -entry pm_fluid_forces
            -entry pm_fluid_integrate
            -entry fluid_reserve_contact_events
            -entry fluid_gather_contact_events
            -entry fluid_source_vacancies
            -entry fluid_source_emit
            -entry fluid_destroy_flags
            -entry fluid_gather
            -entry fluid_copy_initial
            -target cuda
            -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_FLUID_CUDA_SOURCE}
    DEPENDS src/slang/fluid.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared fluid kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry_constraints.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -DPM_GEOMETRY_CONSTRAINTS_ADAPTER=1
            -entry prepare_avbd_kernel
            -entry solve_avbd_kernel
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CUDA_SOURCE}
    DEPENDS src/slang/geometry_constraints.slang
            src/slang/geometry_constraints_solve.slang
            src/slang/constraint_math.slang
            src/slang/geometry.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared rigid constraint solver from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_QUERIES_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/queries.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry export_fluid_particles
            -entry export_cloth_vertices
            -entry export_surface_positions
            -entry export_rope_nodes
            -entry export_soft_body_nodes
            -entry export_smoke_particles
            -entry query_rigid_hit_box_kernel
            -entry query_particle_hit_box_kernel
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_QUERIES_CUDA_SOURCE}
    DEPENDS src/slang/queries.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared query and export kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_GEOMETRY_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
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
            -target cuda -fp-mode precise -O3 -line-directive-mode none
            -o ${PARALLEL_MATER_GEOMETRY_CUDA_SOURCE}
    DEPENDS src/slang/geometry.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared rigid geometry kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_PRIMITIVES_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/primitives.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry pm_exclusive_scan_u32
            -entry pm_compact_count_blocks
            -entry pm_compact_scatter_indices
            -entry pm_radix_histogram_u64
            -entry pm_radix_prefix_u32
            -entry pm_radix_scatter_u64_u32
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_PRIMITIVES_CUDA_SOURCE}
    DEPENDS src/slang/primitives.slang
            src/slang/primitives/scan.slang
            src/slang/primitives/compaction.slang
            src/slang/primitives/radix_sort.slang
            src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared scheduling primitives from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_CLOTH_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/cloth.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry cloth_project_volume
            -entry cloth_break_bonds
            -entry cloth_limit_strain
            -entry cloth_update_surface
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_CLOTH_CUDA_SOURCE}
    DEPENDS src/slang/cloth.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared cloth kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_SOFT_BODY_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/soft_body.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry deformable_predict
            -entry soft_body_measure_momentum
            -entry soft_body_restore_momentum
            -entry soft_body_project_rest_shape
            -entry deformable_project_links
            -entry deformable_project_links_warp
            -entry soft_body_finalize_velocities
            -entry soft_body_apply_contact_friction
            -entry soft_body_damp_springs
            -entry soft_body_damp_springs_warp
            -entry soft_body_update_surface
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_SOFT_BODY_CUDA_SOURCE}
    DEPENDS src/slang/soft_body.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared soft-body kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_SOFT_BODY_CLOTH_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/soft_body_cloth.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry soft_cloth_detect
            -entry soft_cloth_apply_soft
            -entry soft_cloth_apply_cloth
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_SOFT_BODY_CLOTH_CUDA_SOURCE}
    DEPENDS src/slang/soft_body_cloth.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_fluid.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared soft-body/cloth coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_CLOTH_ROPE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/cloth_rope.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry rope_cloth_sample_anchor
            -entry rope_cloth_apply_anchor
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_CLOTH_ROPE_CUDA_SOURCE}
    DEPENDS src/slang/cloth_rope.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared cloth/rope coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_SOFT_BODY_ROPE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/soft_body_rope.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry rope_soft_sample_anchor
            -entry rope_soft_scatter_anchor
            -entry rope_soft_apply
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_SOFT_BODY_ROPE_CUDA_SOURCE}
    DEPENDS src/slang/soft_body_rope.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared soft-body/rope coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_FLUID_CLOTH_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/fluid_cloth.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry fluid_cloth_containment_forces
            -entry cloth_apply_fluid_forces
            -entry fluid_project_inside_cloth
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_FLUID_CLOTH_CUDA_SOURCE}
    DEPENDS src/slang/fluid_cloth.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_fluid.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared fluid/cloth coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_FLUID_ROPE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/fluid_rope.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry fluid_rope_contacts
            -entry fluid_rope_apply
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_FLUID_ROPE_CUDA_SOURCE}
    DEPENDS src/slang/fluid_rope.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared fluid/rope coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_FLUID_SMOKE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/fluid_smoke.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry fluid_smoke_drag
            -entry fluid_smoke_grid_drag
            -entry fluid_smoke_exchange
            -entry fluid_clear_inactive_keep
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_FLUID_SMOKE_CUDA_SOURCE}
    DEPENDS src/slang/fluid_smoke.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_smoke.slang src/slang/smoke_grid.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared fluid/smoke coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_CLOTH_SMOKE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/cloth_smoke.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry smoke_cloth_wind
            -entry smoke_cloth_contact
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_CLOTH_SMOKE_CUDA_SOURCE}
    DEPENDS src/slang/cloth_smoke.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_smoke.slang src/slang/smoke_grid.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared cloth/smoke coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_SOFT_BODY_SMOKE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/soft_body_smoke.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry smoke_soft_body_wind
            -entry smoke_soft_body_contact
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_SOFT_BODY_SMOKE_CUDA_SOURCE}
    DEPENDS src/slang/soft_body_smoke.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_smoke.slang src/slang/smoke_grid.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared soft-body/smoke coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_ROPE_SMOKE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/rope_smoke.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry smoke_rope_wind
            -entry smoke_rope_contact
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_ROPE_SMOKE_CUDA_SOURCE}
    DEPENDS src/slang/rope_smoke.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/rope.slang src/slang/geometry_smoke.slang
            src/slang/smoke_grid.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared rope/smoke coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_FLUID_SOFT_BODY_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/fluid_soft_body.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry fluid_soft_refit
            -entry fluid_soft_detect
            -entry fluid_soft_solve
            -entry fluid_soft_apply
            -entry fluid_soft_recover
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_FLUID_SOFT_BODY_CUDA_SOURCE}
    DEPENDS src/slang/fluid_soft_body.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_fluid.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared fluid/soft-body coupling kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_GEOMETRY_CLOTH_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry_cloth.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry cloth_constrain_bodies
            -entry cloth_apply_body_corrections
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_GEOMETRY_CLOTH_CUDA_SOURCE}
    DEPENDS src/slang/geometry_cloth.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_fluid.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared cloth/rigid geometry kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_GEOMETRY_FLUID_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry_fluid.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry fluid_static_contacts
            -entry fluid_body_bounds_kernel
            -entry fluid_index_body_cells
            -entry fluid_detect_moving_contacts
            -entry fluid_resolve_moving_contacts
            -entry reduce_point_body_impulses
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_GEOMETRY_FLUID_CUDA_SOURCE}
    DEPENDS src/slang/geometry_fluid.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared fluid/rigid geometry kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_GEOMETRY_SMOKE_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry_smoke.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry smoke_emit_cells
            -entry smoke_density_pressure
            -entry smoke_pair_forces
            -entry smoke_compute_vorticity
            -entry smoke_advect
            -entry smoke_emit
            -entry smoke_deformable_bounds
            -entry smoke_rigid_contact
            -entry smoke_apply_rigid_impulses
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_GEOMETRY_SMOKE_CUDA_SOURCE}
    DEPENDS src/slang/geometry_smoke.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_fluid.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared smoke geometry kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_GEOMETRY_SOFT_BODY_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/geometry_soft_body.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry deformable_collide
            -entry soft_body_surface_contacts
            -entry soft_body_apply_surface_contacts
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_GEOMETRY_SOFT_BODY_CUDA_SOURCE}
    DEPENDS src/slang/geometry_soft_body.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_fluid.slang src/slang/geometry.slang
            ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared deformable/rigid geometry kernel from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_SMOKE_GRID_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/smoke_grid.slang
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/slang
            -entry smoke_grid_initialize_cells
            -entry smoke_grid_initialize_faces
            -entry smoke_grid_clear_obstacles
            -entry smoke_grid_raster_rigid
            -entry smoke_grid_raster_deformable
            -entry smoke_grid_raster_soft
            -entry smoke_grid_resolve_particle_fields
            -entry smoke_grid_splat_particles
            -entry smoke_grid_advect_face
            -entry smoke_grid_correct_face
            -entry smoke_grid_cell_diagnostics
            -entry smoke_grid_subgrid_force
            -entry smoke_grid_apply_face_forces
            -entry smoke_grid_apply_face_boundaries
            -entry smoke_grid_divergence
            -entry smoke_grid_pressure_smooth
            -entry smoke_grid_pressure_residual
            -entry smoke_grid_restrict_residual
            -entry smoke_grid_clear_scalar
            -entry smoke_grid_restrict_open
            -entry smoke_grid_mark_domain_boundaries
            -entry smoke_grid_prolong_add
            -entry smoke_grid_compare_residual
            -entry smoke_grid_reset_residual
            -entry smoke_grid_project_face
            -entry smoke_grid_trace
            -entry smoke_grid_soft_body_force
            -entry smoke_grid_cloth_force
            -entry smoke_grid_apply_cloth_force
            -entry smoke_grid_rigid_force
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_SMOKE_GRID_CUDA_SOURCE}
    DEPENDS src/slang/smoke_grid.slang src/slang/types.slang
            src/slang/math.slang src/slang/device_intrinsics.slang
            src/slang/geometry_fluid.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared Eulerian smoke-grid kernels from Slang to CUDA"
    VERBATIM)
add_library(
    parallel-mater-driver-slang-cuda SHARED
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
    src/device/cuda_device.cpp
    src/simulation/exports.cu
    src/simulation/geometry_dispatch.cu
    src/simulation/geometry_constraints_dispatch.cu
    src/simulation/cloth_dispatch.cu
    src/simulation/soft_body_dispatch.cu
    src/simulation/soft_body_cloth_dispatch.cu
    src/simulation/cloth_rope_dispatch.cu
    src/simulation/soft_body_rope_dispatch.cu
    src/simulation/rope_dispatch.cu
    src/simulation/fluid_dispatch.cu
    src/simulation/fluid_cloth_dispatch.cu
    src/simulation/fluid_rope_dispatch.cu
    src/simulation/fluid_smoke_dispatch.cu
    src/simulation/cloth_smoke_dispatch.cu
    src/simulation/soft_body_smoke_dispatch.cu
    src/simulation/rope_smoke_dispatch.cu
    src/simulation/fluid_soft_body_dispatch.cu
    src/simulation/geometry_cloth_dispatch.cu
    src/simulation/geometry_fluid_dispatch.cu
    src/simulation/geometry_smoke_dispatch.cu
    src/simulation/geometry_soft_body_dispatch.cu
    src/simulation/smoke_grid_dispatch.cu
    src/simulation/memory_arena.cpp
    src/simulation/primitives.cu
    src/simulation/runtime.cu
    src/simulation/geometry_storage.hpp
    src/simulation/rope_storage.hpp
    src/simulation/fluid_smoke_storage.hpp
    src/simulation/smoke_storage.hpp)
set_source_files_properties(
    ${PARALLEL_MATER_FLUID_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CUDA_SOURCE}
    ${PARALLEL_MATER_QUERIES_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_CUDA_SOURCE}
    ${PARALLEL_MATER_PRIMITIVES_CUDA_SOURCE}
    ${PARALLEL_MATER_CLOTH_CUDA_SOURCE}
    ${PARALLEL_MATER_SOFT_BODY_CUDA_SOURCE}
    ${PARALLEL_MATER_SOFT_BODY_CLOTH_CUDA_SOURCE}
    ${PARALLEL_MATER_CLOTH_ROPE_CUDA_SOURCE}
    ${PARALLEL_MATER_SOFT_BODY_ROPE_CUDA_SOURCE}
    ${PARALLEL_MATER_FLUID_CLOTH_CUDA_SOURCE}
    ${PARALLEL_MATER_FLUID_ROPE_CUDA_SOURCE}
    ${PARALLEL_MATER_FLUID_SMOKE_CUDA_SOURCE}
    ${PARALLEL_MATER_CLOTH_SMOKE_CUDA_SOURCE}
    ${PARALLEL_MATER_SOFT_BODY_SMOKE_CUDA_SOURCE}
    ${PARALLEL_MATER_ROPE_SMOKE_CUDA_SOURCE}
    ${PARALLEL_MATER_FLUID_SOFT_BODY_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_CLOTH_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_FLUID_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_SMOKE_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_SOFT_BODY_CUDA_SOURCE}
    ${PARALLEL_MATER_SMOKE_GRID_CUDA_SOURCE}
    ${PARALLEL_MATER_ROPE_CUDA_SOURCE}
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
target_sources(parallel-mater-driver-slang-cuda PRIVATE
    ${PARALLEL_MATER_FLUID_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_CONSTRAINTS_CUDA_SOURCE}
    ${PARALLEL_MATER_QUERIES_CUDA_SOURCE}
    ${PARALLEL_MATER_GEOMETRY_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_PRIMITIVES_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_CLOTH_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_SOFT_BODY_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_SOFT_BODY_CLOTH_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_CLOTH_ROPE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_SOFT_BODY_ROPE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_FLUID_CLOTH_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_FLUID_ROPE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_FLUID_SMOKE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_CLOTH_SMOKE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_SOFT_BODY_SMOKE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_ROPE_SMOKE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_FLUID_SOFT_BODY_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_GEOMETRY_CLOTH_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_GEOMETRY_FLUID_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_GEOMETRY_SMOKE_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_GEOMETRY_SOFT_BODY_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_SMOKE_GRID_CUDA_SOURCE})
target_sources(parallel-mater-driver-slang-cuda PRIVATE ${PARALLEL_MATER_ROPE_CUDA_SOURCE})

list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater-driver-slang-cuda)
set_target_properties(
    parallel-mater-driver-slang-cuda
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_RUNTIME_LIBRARY Shared
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        OUTPUT_NAME parallel_mater_driver_slang_cuda
        EXPORT_NAME slang_cuda_driver
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON)
set_property(TARGET parallel-mater-driver-slang-cuda PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_20)
target_include_directories(
    parallel-mater-driver-slang-cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PARALLEL_MATER_SLANG_GENERATED_DIR}
        ${PARALLEL_MATER_API_GENERATED_DIR})
target_compile_definitions(parallel-mater-driver-slang-cuda PRIVATE
    PARALLEL_MATER_DRIVER_IMPLEMENTATION=1)
target_link_libraries(parallel-mater-driver-slang-cuda PRIVATE
    parallel-mater-host-preparation
    CUDA::cuda_driver
    CUDA::cudart)
if(WIN32)
    find_file(PARALLEL_MATER_CUDART_RUNTIME
        NAMES "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll"
        PATHS "${CUDAToolkit_BIN_DIR}"
        NO_DEFAULT_PATH
        REQUIRED)
    add_custom_command(
        TARGET parallel-mater-driver-slang-cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PARALLEL_MATER_CUDART_RUNTIME}"
                "$<TARGET_FILE_DIR:parallel-mater-driver-slang-cuda>"
        COMMENT "Deploying the CUDA runtime beside the Slang CUDA driver"
        VERBATIM)
endif()
if(MSVC)
    target_compile_options(
        parallel-mater-driver-slang-cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
else()
    target_compile_options(
        parallel-mater-driver-slang-cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

add_library(ParallelMater::slang_cuda_driver ALIAS
            parallel-mater-driver-slang-cuda)
file(GENERATE
    OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json"
    CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-driver-slang-cuda>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
file(GENERATE
    OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-cuda.json"
    CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"../$<TARGET_FILE_NAME:parallel-mater-driver-slang-cuda>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")

if(BUILD_TESTING)
    add_executable(parallel-mater-driver-abi-tests
        tests/api/driver_abi_tests.cpp)
    target_include_directories(parallel-mater-driver-abi-tests PRIVATE include)
    target_compile_features(parallel-mater-driver-abi-tests PRIVATE cxx_std_20)
    target_link_libraries(parallel-mater-driver-abi-tests PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(parallel-mater-driver-abi-tests
        parallel-mater-driver-slang-cuda parallel-mater)
    add_test(NAME parallel-mater-driver-abi-tests
        COMMAND parallel-mater-driver-abi-tests
            $<TARGET_FILE:parallel-mater-driver-slang-cuda>
            $<TARGET_FILE:parallel-mater>)
    set_tests_properties(parallel-mater-driver-abi-tests PROPERTIES
        LABELS "api;driver")

    add_executable(parallel-mater-command-api-tests
        tests/api/command_api_tests.cpp)
    target_link_libraries(parallel-mater-command-api-tests
                          PRIVATE ParallelMater::parallel_mater)
    target_compile_features(parallel-mater-command-api-tests PRIVATE cxx_std_20)
    add_dependencies(parallel-mater-command-api-tests
        parallel-mater-driver-slang-cuda parallel-mater-layer-validation)
    add_test(NAME parallel-mater-command-api-tests
             COMMAND parallel-mater-command-api-tests)
    set_tests_properties(parallel-mater-command-api-tests
        PROPERTIES
            LABELS "api;cuda;loader"
            SKIP_RETURN_CODE 77
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
endif()

if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    add_library(parallel-mater-gallery-scene STATIC
                examples/gallery/scene_builder.cpp
                examples/gallery/render/render_scene.cpp
                examples/support/scene.cpp
                examples/support/brick_scene.cpp
                examples/support/device_profiles.cpp
                examples/support/rope_render.cpp
                examples/support/surface_query.cpp)
    target_compile_features(parallel-mater-gallery-scene PUBLIC cxx_std_20)
    target_include_directories(
        parallel-mater-gallery-scene
        PUBLIC examples/support/include examples/gallery
        PRIVATE src ${parallel_mater_cgltf_SOURCE_DIR})
    target_link_libraries(
        parallel-mater-gallery-scene
        PUBLIC ParallelMater::parallel_mater
        PRIVATE parallel-mater-host-preparation)
    parallel_mater_enable_warnings(parallel-mater-gallery-scene)

    if(BUILD_TESTING)
        add_executable(parallel-mater-command-gallery-tests
                       tests/gallery/command_gallery_tests.cpp)
        target_link_libraries(parallel-mater-command-gallery-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-command-gallery-tests PRIVATE
            PARALLEL_MATER_GALLERY_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets")
        add_dependencies(parallel-mater-command-gallery-tests
                         parallel-mater-driver-slang-cuda)
        parallel_mater_enable_warnings(parallel-mater-command-gallery-tests)
        add_test(NAME parallel-mater-command-gallery-tests
                 COMMAND parallel-mater-command-gallery-tests)
        set_tests_properties(parallel-mater-command-gallery-tests PROPERTIES
            LABELS "api;gallery;cuda"
            SKIP_RETURN_CODE 77
            TIMEOUT 300
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json")

        add_executable(parallel-mater-gallery-scene-tests
                       tests/gallery/scene_tests.cpp)
        target_link_libraries(parallel-mater-gallery-scene-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-gallery-scene-tests PRIVATE
            PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb"
            PARALLEL_MATER_PASSIVE_ACTIVE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/PassiveActive.glb"
            PARALLEL_MATER_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Fluid.glb"
            PARALLEL_MATER_FLUID_RIGID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/FluidRigid.glb"
            PARALLEL_MATER_PEGS_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Pegs.glb")
        add_dependencies(parallel-mater-gallery-scene-tests
                         parallel-mater-driver-slang-cuda)
        parallel_mater_enable_warnings(parallel-mater-gallery-scene-tests)
        add_test(NAME parallel-mater-gallery-scene-tests
                 COMMAND parallel-mater-gallery-scene-tests)

        add_executable(parallel-mater-device-profile-tests
                       tests/gallery/device_profiles_tests.cpp)
        target_link_libraries(parallel-mater-device-profile-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-device-profile-tests PRIVATE
            PARALLEL_MATER_DEVICE_PROFILES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
            PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb")
        add_dependencies(parallel-mater-device-profile-tests
                         parallel-mater-driver-slang-cuda)
        parallel_mater_enable_warnings(parallel-mater-device-profile-tests)
        add_test(NAME parallel-mater-device-profile-tests
                 COMMAND parallel-mater-device-profile-tests)

        add_executable(parallel-mater-dump-truck-tests
                       tests/gallery/dump_truck_tests.cpp)
        target_link_libraries(parallel-mater-dump-truck-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-dump-truck-tests PRIVATE
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb")
        add_dependencies(parallel-mater-dump-truck-tests
                         parallel-mater-driver-slang-cuda)
        parallel_mater_enable_warnings(parallel-mater-dump-truck-tests)
        add_test(NAME parallel-mater-dump-truck-tests
                 COMMAND parallel-mater-dump-truck-tests)

        set_tests_properties(
            parallel-mater-gallery-scene-tests
            parallel-mater-device-profile-tests
            parallel-mater-dump-truck-tests PROPERTIES
            LABELS "api;gallery;cuda"
            SKIP_RETURN_CODE 77
            TIMEOUT 300
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json")
    endif()

    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT)
        set(PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT unknown)
    endif()
    FetchContent_Declare(
        pm_optix
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pm_optix)

    if(NOT WIN32)
        find_package(OpenGL REQUIRED)
    endif()
    find_package(glfw3 3.3 QUIET)
    if(NOT TARGET glfw)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            parallel_mater_glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.4
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(parallel_mater_glfw)
    endif()

    set(PARALLEL_MATER_OPTIX_PTX
        ${CMAKE_CURRENT_BINARY_DIR}/generated/parallel_mater_optix.ptx)
    if(CMAKE_CUDA_ARCHITECTURES)
        set(parallel_mater_optix_architectures ${CMAKE_CUDA_ARCHITECTURES})
        list(GET parallel_mater_optix_architectures 0
            parallel_mater_optix_architecture)
        string(REGEX REPLACE "-(real|virtual)$" ""
            parallel_mater_optix_architecture
            "${parallel_mater_optix_architecture}")
    else()
        set(parallel_mater_optix_architecture 52)
    endif()
    add_custom_command(
        OUTPUT ${PARALLEL_MATER_OPTIX_PTX}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND
            ${CMAKE_CUDA_COMPILER}
            --ptx
            --std=c++17
            --gpu-architecture=compute_${parallel_mater_optix_architecture}
            --use_fast_math
            --generate-line-info
            -I${pm_optix_SOURCE_DIR}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/examples/support
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/support/optix_device.cu
            -o ${PARALLEL_MATER_OPTIX_PTX}
        DEPENDS
            examples/support/optix_device.cu
            examples/support/renderer_shared.hpp
        VERBATIM)
    add_custom_target(parallel-mater-optix-ptx
                      DEPENDS ${PARALLEL_MATER_OPTIX_PTX})

    add_library(parallel-mater-gallery-renderer STATIC
                examples/support/optix_renderer.cpp
                examples/support/foam_visuals.cpp
                examples/support/fluid_surface.cu
                examples/support/overlay.cpp
                examples/support/physics_debug.cpp)
    add_dependencies(parallel-mater-gallery-renderer parallel-mater-optix-ptx)
    target_compile_features(parallel-mater-gallery-renderer PUBLIC cxx_std_20)
    target_include_directories(
        parallel-mater-gallery-renderer
        PUBLIC examples/support/include
        PRIVATE
            examples/support
            ${pm_optix_SOURCE_DIR}/include)
    target_link_libraries(
        parallel-mater-gallery-renderer
        PUBLIC parallel-mater-gallery-scene
        PRIVATE CUDA::cuda_driver CUDA::cudart ${CMAKE_DL_LIBS})
    parallel_mater_enable_warnings(parallel-mater-gallery-renderer)
    target_compile_definitions(parallel-mater-gallery-renderer
        PRIVATE
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}"
            $<$<PLATFORM_ID:Windows>:NOMINMAX>)

    add_library(parallel-mater-gallery-window STATIC)
    target_compile_features(parallel-mater-gallery-window PUBLIC cxx_std_20)
    target_include_directories(parallel-mater-gallery-window
        PUBLIC examples/support/include)
    if(WIN32)
        target_sources(parallel-mater-gallery-window PRIVATE
            examples/support/d3d12_window.cpp)
        target_link_libraries(parallel-mater-gallery-window
            PRIVATE glfw d3d12 dxgi)
        target_compile_definitions(parallel-mater-gallery-window
            PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_sources(parallel-mater-gallery-window PRIVATE
            examples/support/opengl_window.cpp)
        target_link_libraries(parallel-mater-gallery-window
            PRIVATE glfw OpenGL::GL)
    endif()
    parallel_mater_enable_warnings(parallel-mater-gallery-window)

    if(PARALLEL_MATER_BUILD_BENCHMARKS)
        add_executable(parallel-mater-dump-truck-benchmark benchmarks/dump_truck.cpp)
        target_link_libraries(parallel-mater-dump-truck-benchmark PRIVATE parallel-mater-gallery-renderer)
        target_compile_definitions(parallel-mater-dump-truck-benchmark PRIVATE
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb"
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
        parallel_mater_enable_warnings(parallel-mater-dump-truck-benchmark)
        add_executable(parallel-mater-rigid-frame-benchmark benchmarks/rigid_frame.cpp)
        target_link_libraries(parallel-mater-rigid-frame-benchmark
                              PRIVATE parallel-mater-gallery-renderer)
        target_compile_definitions(parallel-mater-rigid-frame-benchmark PRIVATE
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
        parallel_mater_enable_warnings(parallel-mater-rigid-frame-benchmark)
    endif()

    add_library(parallel-mater-gallery-application STATIC
                examples/gallery/application.cpp
                examples/support/cuda_runtime_platform.cpp)
    target_compile_features(parallel-mater-gallery-application PUBLIC cxx_std_20)
    target_link_libraries(
        parallel-mater-gallery-application
        PUBLIC parallel-mater-gallery-renderer
        PRIVATE glfw parallel-mater-gallery-window CUDA::cudart)

    add_executable(parallel-mater-gallery examples/gallery/main.cpp)
    add_dependencies(parallel-mater-gallery
        parallel-mater-driver-slang-cuda parallel-mater-layer-validation)
    if(WIN32)
        target_sources(parallel-mater-gallery PRIVATE
                       "${PARALLEL_MATER_GALLERY_ICON_RESOURCE}")
        set_target_properties(parallel-mater-gallery PROPERTIES
            OUTPUT_NAME "Parallel-Mater")
    endif()
    set(PARALLEL_MATER_GALLERY_PROFILE_CATALOG
        "${CMAKE_CURRENT_BINARY_DIR}/config/device-profiles.json")
    configure_file(config/device-profiles.json
                   "${PARALLEL_MATER_GALLERY_PROFILE_CATALOG}" COPYONLY)
    target_link_libraries(
        parallel-mater-gallery
        PRIVATE parallel-mater-gallery-application)
    target_compile_definitions(
        parallel-mater-gallery-application
        PRIVATE
            PARALLEL_MATER_DEFAULT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb"
            PARALLEL_MATER_DEVICE_PROFILES_PATH="${PARALLEL_MATER_GALLERY_PROFILE_CATALOG}"
            PARALLEL_MATER_GALLERY_GIT_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}"
            PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintFixed.glb"
            PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Celestial.glb"
            PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintHinge.glb"
            PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPiston.glb"
            PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGeneric.glb"
            PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintMotorSpring.glb"
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb"
            PARALLEL_MATER_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Fluid.glb"
            PARALLEL_MATER_FLUID_RIGID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/FluidRigid.glb"
            PARALLEL_MATER_PEGS_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Pegs.glb"
            PARALLEL_MATER_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Cloth.glb"
            PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ClothTear.glb"
            PARALLEL_MATER_CLOTH_PAINT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ClothPaint.glb"
            PARALLEL_MATER_CLOTH_WATER_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ClothWater.glb"
            PARALLEL_MATER_SOFT_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Softbody.glb"
            PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SoftbodyRigidBody.glb"
            PARALLEL_MATER_SOFT_BODY_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SoftbodyCloth.glb"
            PARALLEL_MATER_SOFT_BODY_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SoftbodyFluid.glb"
            PARALLEL_MATER_ROPE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Rope.glb"
            PARALLEL_MATER_ROPE_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RopeFluid.glb"
            PARALLEL_MATER_ROPE_SOFT_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RopeSoftbody.glb"
            PARALLEL_MATER_ROPE_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RopeCloth.glb"
            PARALLEL_MATER_SMOKE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Smoke.glb"
            PARALLEL_MATER_SMOKE_WATER_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeWater.glb"
            PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeSoftbody.glb"
            PARALLEL_MATER_SMOKE_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeCloth.glb"
            PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeRope.glb")
    add_custom_command(
        TARGET parallel-mater-gallery POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:parallel-mater-gallery>/config"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
                "$<TARGET_FILE_DIR:parallel-mater-gallery>/config/device-profiles.json"
        VERBATIM)
    if(WIN32)
        add_custom_target(install-windows-gallery
            COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass
                -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Install-WindowsGallery.ps1"
                -BuildDirectory "$<TARGET_FILE_DIR:parallel-mater-gallery>"
                -Version "${PROJECT_VERSION}"
            DEPENDS parallel-mater-gallery
            COMMENT "Installing Parallel-Mater for the current user"
            USES_TERMINAL
            VERBATIM)
    endif()
endif()

endif() # PARALLEL_MATER_BUILD_CUDA

if(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
    if(NOT APPLE)
        message(FATAL_ERROR
            "PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER currently requires macOS")
    endif()
    include(cmake/ParallelMaterSlangCpu.cmake)
endif()

if(PARALLEL_MATER_BUILD_MACOS_GALLERY)
    if(NOT APPLE)
        message(FATAL_ERROR
            "PARALLEL_MATER_BUILD_MACOS_GALLERY requires macOS")
    endif()
    include(cmake/ParallelMaterMacOSGallery.cmake)
endif()

if(PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS)
    include(cmake/SlangGeometryConstraints.cmake)
    parallel_mater_add_slang_geometry_constraints()
    if(WIN32 AND TARGET parallel-mater-gallery-window)
        add_dependencies(parallel-mater-gallery-window
                         parallel-mater-slang-avbd-dxil)
    endif()
endif()

if(PARALLEL_MATER_BUILD_SLANG_MIRRORS)
    include(cmake/SlangKernelMirrors.cmake)
    parallel_mater_add_slang_kernel_mirrors()
    if(PARALLEL_MATER_SLANG_TEST_BUILD AND TARGET parallel-mater-gallery)
        add_dependencies(parallel-mater-gallery
                         parallel-mater-slang-kernel-mirrors)
    endif()
endif()

if(BUILD_TESTING)
    add_subdirectory(tests/solver)
    add_executable(parallel-mater-camera-controller-tests
        tests/gallery/camera_controller_tests.cpp)
    target_compile_features(parallel-mater-camera-controller-tests
        PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-camera-controller-tests PRIVATE
        include examples/support/include)
    add_test(NAME parallel-mater-camera-controller-tests
        COMMAND parallel-mater-camera-controller-tests)

    add_executable(parallel-mater-physics-frame-budget-tests
        tests/gallery/physics_frame_budget_tests.cpp)
    target_compile_features(parallel-mater-physics-frame-budget-tests
        PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-physics-frame-budget-tests
        PRIVATE examples/support/include)
    add_test(NAME parallel-mater-physics-frame-budget-tests
        COMMAND parallel-mater-physics-frame-budget-tests)

    add_executable(parallel-mater-gallery-registry-tests
        tests/gallery/gallery_registry_contract_tests.cpp)
    target_compile_features(parallel-mater-gallery-registry-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-gallery-registry-tests PRIVATE
        include examples/support/include)
    add_test(NAME parallel-mater-gallery-registry-tests
        COMMAND parallel-mater-gallery-registry-tests)
    add_executable(parallel-mater-gallery-input-tests
        tests/gallery/gallery_input_tests.cpp)
    target_compile_features(parallel-mater-gallery-input-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-gallery-input-tests PRIVATE
        include examples/support/include)
    add_test(NAME parallel-mater-gallery-input-tests
        COMMAND parallel-mater-gallery-input-tests)
    add_test(NAME parallel-mater-conformance-case-registry
        COMMAND parallel-mater-conformance-cases --check
                ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/cases)
    add_test(NAME parallel-mater-conformance-comparator-tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/tests
                -p test_*.py)

    add_executable(parallel-mater-types-contract-tests
                   tests/types_contract.cpp)
    target_compile_features(parallel-mater-types-contract-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-types-contract-tests
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME parallel-mater-types-contract-tests
             COMMAND parallel-mater-types-contract-tests)

    add_executable(parallel-mater-soft-body-geometry-tests
                   tests/soft_body_geometry_tests.cpp)
    target_link_libraries(parallel-mater-soft-body-geometry-tests
                          PRIVATE parallel-mater-host-preparation)
    parallel_mater_enable_warnings(parallel-mater-soft-body-geometry-tests)
    add_test(NAME parallel-mater-soft-body-geometry-tests
             COMMAND parallel-mater-soft-body-geometry-tests)

endif()

configure_package_config_file(
    cmake/ParallelMaterConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

set(PARALLEL_MATER_INSTALLED_LAYER_PATH
    "../../../${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_layer_validation${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(config/validation-layer.json.in
    ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_validation.json @ONLY)
if(PARALLEL_MATER_BUILD_CUDA)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_cuda${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cuda.json @ONLY)
endif()
if(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_cpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cpu.json @ONLY)
endif()

install(
    TARGETS ${PARALLEL_MATER_INSTALL_TARGETS}
    EXPORT ParallelMaterTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    FILES
        include/parallel_mater/parallel_mater.h
        include/parallel_mater/interop_vulkan.h
        include/parallel_mater/pm_driver.h
        include/parallel_mater/pm_layer.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parallel_mater)
if(PARALLEL_MATER_BUILD_CUDA)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cuda.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d)
    if(WIN32)
        install(FILES "${PARALLEL_MATER_CUDART_RUNTIME}"
                DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()
if(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cpu.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d)
endif()
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_validation.json
        DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/layers.d)
install(FILES tools/blender/export_parallel_mater_scene.py
        DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/blender)
install(
    EXPORT ParallelMaterTargets
    FILE ParallelMaterTargets.cmake
    NAMESPACE ParallelMater::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
