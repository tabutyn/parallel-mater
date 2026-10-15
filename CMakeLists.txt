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
if(WIN32)
    set(PARALLEL_MATER_D3D12_DEFAULT ON)
else()
    set(PARALLEL_MATER_D3D12_DEFAULT OFF)
endif()

option(PARALLEL_MATER_BUILD_CUDA
       "Build the CUDA Slang device adapter" ${PARALLEL_MATER_CUDA_DEFAULT})
option(PARALLEL_MATER_BUILD_D3D12
       "Build the D3D12 loader and shared-Slang qualification adapter"
       ${PARALLEL_MATER_D3D12_DEFAULT})
option(PARALLEL_MATER_BUILD_GALLERY
       "Build the backend-neutral gallery application" ${APPLE})
option(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER
       "Build the serial Slang CPU reference driver" ${APPLE})
option(PARALLEL_MATER_BUILD_OPTIX_GALLERY
       "Deprecated: build the gallery with the OptiX renderer" OFF)
set(PARALLEL_MATER_GALLERY_RENDERER "SOFTWARE" CACHE STRING
    "Gallery renderer adapter (SOFTWARE or OPTIX)")
set_property(CACHE PARALLEL_MATER_GALLERY_RENDERER PROPERTY STRINGS
             SOFTWARE OPTIX)
if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    set(PARALLEL_MATER_BUILD_GALLERY ON CACHE BOOL
        "Build the backend-neutral gallery application" FORCE)
    set(PARALLEL_MATER_GALLERY_RENDERER OPTIX CACHE STRING
        "Gallery renderer adapter (SOFTWARE or OPTIX)" FORCE)
endif()
string(TOUPPER "${PARALLEL_MATER_GALLERY_RENDERER}"
       PARALLEL_MATER_GALLERY_RENDERER)
if(NOT PARALLEL_MATER_GALLERY_RENDERER STREQUAL "SOFTWARE" AND
   NOT PARALLEL_MATER_GALLERY_RENDERER STREQUAL "OPTIX")
    message(FATAL_ERROR
        "PARALLEL_MATER_GALLERY_RENDERER must be SOFTWARE or OPTIX")
endif()
option(PARALLEL_MATER_BUILD_BENCHMARKS
       "Build local performance benchmark executables" OFF)
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

if(WIN32 AND (PARALLEL_MATER_BUILD_GALLERY OR
              PARALLEL_MATER_BUILD_D3D12))
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
if(WIN32 AND PARALLEL_MATER_BUILD_GALLERY)
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

if(PARALLEL_MATER_BUILD_GALLERY AND
   PARALLEL_MATER_GALLERY_RENDERER STREQUAL "OPTIX" AND
   NOT PARALLEL_MATER_BUILD_CUDA)
    message(FATAL_ERROR "The OptiX gallery requires PARALLEL_MATER_BUILD_CUDA")
endif()
if(PARALLEL_MATER_BUILD_BENCHMARKS AND
   NOT PARALLEL_MATER_BUILD_GALLERY)
    message(FATAL_ERROR
            "Rigid scene benchmarks require PARALLEL_MATER_BUILD_GALLERY")
endif()
if(PARALLEL_MATER_BUILD_BENCHMARKS AND NOT PARALLEL_MATER_BUILD_CUDA)
    message(FATAL_ERROR
        "The current profiler-backed benchmarks require PARALLEL_MATER_BUILD_CUDA")
endif()

if(PARALLEL_MATER_BUILD_CUDA OR PARALLEL_MATER_BUILD_D3D12 OR
   PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER OR
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

if(PARALLEL_MATER_BUILD_GALLERY OR
   (PARALLEL_MATER_BUILD_CUDA AND BUILD_TESTING))
    FetchContent_Declare(
        parallel_mater_cgltf
        GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
        GIT_TAG 85cd62382dfea638278962690cf515023f33ed00
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(parallel_mater_cgltf)
endif()

if(PARALLEL_MATER_BUILD_CUDA OR PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER OR
   PARALLEL_MATER_BUILD_GALLERY OR BUILD_TESTING)
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

include(cmake/SlangKernelManifest.cmake)
parallel_mater_compile_slang_cuda(PARALLEL_MATER_SLANG_CUDA_SOURCES)

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
    src/simulation/runtime_domains.cu.inc
    src/simulation/runtime_deformables.cu.inc
    src/simulation/runtime_rigid.cu.inc
    src/simulation/runtime_schedule.cu.inc
    src/simulation/runtime_diagnostics.cu.inc
    src/simulation/geometry_storage.hpp
    src/simulation/rope_storage.hpp
    src/simulation/fluid_smoke_storage.hpp
    src/simulation/smoke_storage.hpp
    ${PARALLEL_MATER_SLANG_CUDA_SOURCES})

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

endif() # PARALLEL_MATER_BUILD_CUDA

if(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
    if(NOT APPLE)
        message(FATAL_ERROR
            "PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER currently requires macOS")
    endif()
    include(cmake/ParallelMaterSlangCpu.cmake)
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

if(PARALLEL_MATER_BUILD_GALLERY)
    add_library(parallel-mater-gallery-scene STATIC
                examples/gallery/scene_builder.cpp
                examples/gallery/scene_builder_create.cpp.inc
                examples/gallery/scene_builder_output.cpp.inc
                examples/gallery/render/render_scene.cpp
                examples/support/scene.cpp
                examples/support/brick_scene.cpp
                examples/support/device_profiles.cpp
                examples/support/physics_debug.cpp
                examples/support/rope_render.cpp
                examples/support/surface_query.cpp)
    target_compile_features(parallel-mater-gallery-scene PUBLIC cxx_std_20)
    target_include_directories(
        parallel-mater-gallery-scene
        PUBLIC examples/support/include examples/gallery
               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        PRIVATE ${parallel_mater_cgltf_SOURCE_DIR})
    target_link_libraries(
        parallel-mater-gallery-scene
        PUBLIC ParallelMater::parallel_mater
        PRIVATE parallel-mater-host-preparation)
    parallel_mater_enable_warnings(parallel-mater-gallery-scene)

    if(BUILD_TESTING AND PARALLEL_MATER_BUILD_CUDA)
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

        add_executable(parallel-mater-rigid-constraint-command-tests
                       tests/gallery/rigid_constraint_command_tests.cpp)
        target_link_libraries(parallel-mater-rigid-constraint-command-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(
            parallel-mater-rigid-constraint-command-tests PRIVATE
            PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintHinge.glb")
        add_dependencies(parallel-mater-rigid-constraint-command-tests
                         parallel-mater-driver-slang-cuda)
        parallel_mater_enable_warnings(
            parallel-mater-rigid-constraint-command-tests)
        add_test(NAME parallel-mater-rigid-constraint-command-tests
                 COMMAND parallel-mater-rigid-constraint-command-tests)
        set_tests_properties(
            parallel-mater-rigid-constraint-command-tests PROPERTIES
            LABELS "api;gallery;cuda;constraints"
            SKIP_RETURN_CODE 77
            TIMEOUT 300
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json")

        add_executable(parallel-mater-physics-debug-tests
                       tests/gallery/physics_debug_tests.cpp)
        target_link_libraries(parallel-mater-physics-debug-tests
                              PRIVATE parallel-mater-gallery-scene)
        parallel_mater_enable_warnings(parallel-mater-physics-debug-tests)
        add_test(NAME parallel-mater-physics-debug-tests
                 COMMAND parallel-mater-physics-debug-tests)
        set_tests_properties(parallel-mater-physics-debug-tests PROPERTIES
            LABELS "gallery;capture")

        add_executable(parallel-mater-gallery-scene-tests
                       tests/gallery/scene_tests.cpp)
        target_link_libraries(parallel-mater-gallery-scene-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-gallery-scene-tests PRIVATE
            PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb"
            PARALLEL_MATER_PASSIVE_ACTIVE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/PassiveActive.glb"
            PARALLEL_MATER_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Fluid.glb"
            PARALLEL_MATER_FLUID_RIGID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/FluidRigid.glb"
            PARALLEL_MATER_PEGS_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Pegs.glb"
            PARALLEL_MATER_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Cloth.glb"
            PARALLEL_MATER_ALPHA_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/assets/material_alpha.gltf")
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

        add_executable(parallel-mater-conformance
                       conformance/v1/src/command_runner.cpp)
        target_link_libraries(parallel-mater-conformance PRIVATE
            parallel-mater-conformance-registry
            parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-conformance PRIVATE
            PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
            PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
        add_dependencies(parallel-mater-conformance
                         parallel-mater-driver-slang-cuda)
        parallel_mater_enable_warnings(parallel-mater-conformance)
        add_test(NAME parallel-mater-conformance-command-inputs
                 COMMAND parallel-mater-conformance --check-inputs)
        set_tests_properties(parallel-mater-conformance-command-inputs
            PROPERTIES LABELS "api;conformance")
    endif()

    if(PARALLEL_MATER_GALLERY_RENDERER STREQUAL "OPTIX")
        FetchContent_Declare(
            pm_optix
            GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
            GIT_TAG fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(pm_optix)
    endif()

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

    if(PARALLEL_MATER_GALLERY_RENDERER STREQUAL "OPTIX")
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
                    examples/support/fluid_surface.cu)
        add_dependencies(parallel-mater-gallery-renderer
                         parallel-mater-optix-ptx)
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
        target_compile_definitions(parallel-mater-gallery-renderer
            PRIVATE
                PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}"
                $<$<PLATFORM_ID:Windows>:NOMINMAX>)
    else()
        add_library(parallel-mater-gallery-renderer STATIC
                    examples/support/software_renderer.cpp)
        target_include_directories(parallel-mater-gallery-renderer
            PUBLIC examples/support/include
            PRIVATE examples/support examples/gallery)
        target_link_libraries(parallel-mater-gallery-renderer
            PUBLIC parallel-mater-gallery-scene)
    endif()
    target_compile_features(parallel-mater-gallery-renderer PUBLIC cxx_std_20)
    parallel_mater_enable_warnings(parallel-mater-gallery-renderer)

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
        target_link_libraries(parallel-mater-dump-truck-benchmark
                              PRIVATE parallel-mater-gallery-renderer
                                      CUDA::cudart)
        target_compile_definitions(parallel-mater-dump-truck-benchmark PRIVATE
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb")
        parallel_mater_enable_warnings(parallel-mater-dump-truck-benchmark)
        add_executable(parallel-mater-rigid-frame-benchmark benchmarks/rigid_frame.cpp)
        target_link_libraries(parallel-mater-rigid-frame-benchmark
                              PRIVATE parallel-mater-gallery-renderer)
        parallel_mater_enable_warnings(parallel-mater-rigid-frame-benchmark)
    endif()

    add_library(parallel-mater-gallery-application STATIC
                examples/gallery/application.cpp
                examples/gallery/application_runtime.cpp.inc
                examples/gallery/application_input.cpp.inc
                examples/gallery/application_validation.cpp.inc
                examples/gallery/application_calibration.cpp.inc
                examples/support/runtime_platform.cpp
                examples/support/overlay.cpp)
    target_compile_features(parallel-mater-gallery-application PUBLIC cxx_std_20)
    target_link_libraries(
        parallel-mater-gallery-application
        PUBLIC parallel-mater-gallery-renderer
        PRIVATE glfw parallel-mater-gallery-window)

    if(APPLE)
        include(cmake/ParallelMaterMacOSGallery.cmake)
    else()
        add_executable(parallel-mater-gallery examples/gallery/main.cpp)
    endif()
    add_dependencies(parallel-mater-gallery parallel-mater-layer-validation)
    if(PARALLEL_MATER_BUILD_CUDA)
        add_dependencies(parallel-mater-gallery
                         parallel-mater-driver-slang-cuda)
    endif()
    if(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
        add_dependencies(parallel-mater-gallery
                         parallel-mater-driver-slang-cpu)
    endif()
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
    if(APPLE)
        set(PARALLEL_MATER_GALLERY_ASSET_DIRECTORY "../Resources/assets")
        set(PARALLEL_MATER_GALLERY_PROFILE_PATH
            "../Resources/config/device-profiles.json")
    else()
        set(PARALLEL_MATER_GALLERY_ASSET_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets")
        set(PARALLEL_MATER_GALLERY_PROFILE_PATH
            "${PARALLEL_MATER_GALLERY_PROFILE_CATALOG}")
    endif()
    target_compile_definitions(
        parallel-mater-gallery-application
        PRIVATE
            PARALLEL_MATER_DEFAULT_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/RigidBody.glb"
            PARALLEL_MATER_DEVICE_PROFILES_PATH="${PARALLEL_MATER_GALLERY_PROFILE_PATH}"
            PARALLEL_MATER_GALLERY_GIT_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}"
            PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ConstraintFixed.glb"
            PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/Celestial.glb"
            PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ConstraintHinge.glb"
            PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ConstraintPiston.glb"
            PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ConstraintGeneric.glb"
            PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ConstraintMotorSpring.glb"
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/DumpTruck.glb"
            PARALLEL_MATER_FLUID_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/Fluid.glb"
            PARALLEL_MATER_FLUID_RIGID_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/FluidRigid.glb"
            PARALLEL_MATER_PEGS_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/Pegs.glb"
            PARALLEL_MATER_CLOTH_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/Cloth.glb"
            PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ClothTear.glb"
            PARALLEL_MATER_CLOTH_PAINT_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ClothPaint.glb"
            PARALLEL_MATER_CLOTH_WATER_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/ClothWater.glb"
            PARALLEL_MATER_SOFT_BODY_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/Softbody.glb"
            PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/SoftbodyRigidBody.glb"
            PARALLEL_MATER_SOFT_BODY_CLOTH_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/SoftbodyCloth.glb"
            PARALLEL_MATER_SOFT_BODY_FLUID_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/SoftbodyFluid.glb"
            PARALLEL_MATER_ROPE_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/Rope.glb"
            PARALLEL_MATER_ROPE_FLUID_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/RopeFluid.glb"
            PARALLEL_MATER_ROPE_SOFT_BODY_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/RopeSoftbody.glb"
            PARALLEL_MATER_ROPE_CLOTH_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/RopeCloth.glb"
            PARALLEL_MATER_SMOKE_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/Smoke.glb"
            PARALLEL_MATER_SMOKE_WATER_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/SmokeWater.glb"
            PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/SmokeSoftbody.glb"
            PARALLEL_MATER_SMOKE_CLOTH_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/SmokeCloth.glb"
            PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}/SmokeRope.glb")
    if(NOT APPLE)
        add_custom_command(
            TARGET parallel-mater-gallery POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:parallel-mater-gallery>/config"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
                    "$<TARGET_FILE_DIR:parallel-mater-gallery>/config/device-profiles.json"
            VERBATIM)
    endif()
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

if(PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS)
    include(cmake/SlangGeometryConstraints.cmake)
    parallel_mater_add_slang_geometry_constraints()
    if(WIN32 AND TARGET parallel-mater-gallery-window)
        add_dependencies(parallel-mater-gallery-window
                         parallel-mater-slang-avbd-dxil)
    endif()
endif()

if(WIN32 AND PARALLEL_MATER_BUILD_D3D12)
    include(cmake/SlangKernelManifest.cmake)
    parallel_mater_compile_slang_dxil(PARALLEL_MATER_SLANG_DXIL_SCHEDULES)
    add_custom_target(parallel-mater-slang-dxil-schedules
        DEPENDS ${PARALLEL_MATER_SLANG_DXIL_SCHEDULES})
    set(PARALLEL_MATER_D3D12_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/d3d12")
    file(MAKE_DIRECTORY "${PARALLEL_MATER_D3D12_GENERATED_DIR}")
    set(PARALLEL_MATER_D3D12_DXIL_MANIFEST
        "${PARALLEL_MATER_D3D12_GENERATED_DIR}/slang_dxil_manifest.txt")
    string(REPLACE ";" "\n" PARALLEL_MATER_D3D12_DXIL_MANIFEST_CONTENT
           "${PARALLEL_MATER_SLANG_DXIL_SCHEDULES_RECORDS}")
    file(GENERATE OUTPUT "${PARALLEL_MATER_D3D12_DXIL_MANIFEST}"
        CONTENT "${PARALLEL_MATER_D3D12_DXIL_MANIFEST_CONTENT}\n")
    set(PARALLEL_MATER_D3D12_DXIL_LIBRARY
        "${PARALLEL_MATER_D3D12_GENERATED_DIR}/parallel_mater_slang_dxil.hpp")
    add_custom_command(
        OUTPUT "${PARALLEL_MATER_D3D12_DXIL_LIBRARY}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${PARALLEL_MATER_D3D12_GENERATED_DIR}"
        COMMAND ${CMAKE_COMMAND}
                "-DMANIFEST_FILE=${PARALLEL_MATER_D3D12_DXIL_MANIFEST}"
                "-DOUTPUT_FILE=${PARALLEL_MATER_D3D12_DXIL_LIBRARY}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedDxilLibrary.cmake"
        DEPENDS
            ${PARALLEL_MATER_SLANG_DXIL_SCHEDULES_DXIL}
            "${PARALLEL_MATER_D3D12_DXIL_MANIFEST}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedDxilLibrary.cmake"
        COMMENT "Embedding common Slang DXIL schedules in the D3D12 driver"
        VERBATIM)
    add_library(parallel-mater-driver-slang-d3d12 SHARED
        src/driver/d3d12/entry.cpp
        src/driver/d3d12/slang_command.cpp
        "${PARALLEL_MATER_D3D12_DXIL_LIBRARY}")
    add_dependencies(parallel-mater-driver-slang-d3d12
                     parallel-mater-slang-dxil-schedules)
    target_compile_features(parallel-mater-driver-slang-d3d12 PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-driver-slang-d3d12 PRIVATE
        PARALLEL_MATER_DRIVER_IMPLEMENTATION=1
        NOMINMAX
        WIN32_LEAN_AND_MEAN)
    target_include_directories(parallel-mater-driver-slang-d3d12
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE "${PARALLEL_MATER_D3D12_GENERATED_DIR}")
    target_link_libraries(parallel-mater-driver-slang-d3d12 PRIVATE d3d12 dxgi)
    parallel_mater_enable_warnings(parallel-mater-driver-slang-d3d12)
    set_target_properties(parallel-mater-driver-slang-d3d12 PROPERTIES
        OUTPUT_NAME parallel_mater_driver_slang_d3d12
        EXPORT_NAME slang_d3d12_driver
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON)
    add_library(ParallelMater::slang_d3d12_driver ALIAS
                parallel-mater-driver-slang-d3d12)
    list(APPEND PARALLEL_MATER_INSTALL_TARGETS
         parallel-mater-driver-slang-d3d12)
    file(GENERATE
        OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-d3d12.json"
        CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-driver-slang-d3d12>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
    file(GENERATE
        OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-d3d12.json"
        CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"../$<TARGET_FILE_NAME:parallel-mater-driver-slang-d3d12>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
    if(BUILD_TESTING)
        add_executable(parallel-mater-d3d12-command-api-tests
            tests/api/d3d12_command_api_tests.cpp)
        target_compile_features(parallel-mater-d3d12-command-api-tests
            PRIVATE cxx_std_20)
        target_link_libraries(parallel-mater-d3d12-command-api-tests
            PRIVATE ParallelMater::parallel_mater)
        add_dependencies(parallel-mater-d3d12-command-api-tests
            parallel-mater-driver-slang-d3d12)
        parallel_mater_enable_warnings(parallel-mater-d3d12-command-api-tests)
        add_test(NAME parallel-mater-d3d12-command-api-tests
            COMMAND parallel-mater-d3d12-command-api-tests)
        set_tests_properties(parallel-mater-d3d12-command-api-tests PROPERTIES
            LABELS "api;d3d12;loader;slang"
            SKIP_RETURN_CODE 77
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-d3d12.json")
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
        include examples/support/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
        include examples/support/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME parallel-mater-gallery-registry-tests
        COMMAND parallel-mater-gallery-registry-tests)
    add_executable(parallel-mater-gallery-input-tests
        tests/gallery/gallery_input_tests.cpp)
    target_compile_features(parallel-mater-gallery-input-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-gallery-input-tests PRIVATE
        include examples/support/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
if(WIN32 AND PARALLEL_MATER_BUILD_D3D12)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_d3d12${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_d3d12.json @ONLY)
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
if(WIN32 AND PARALLEL_MATER_BUILD_D3D12)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_d3d12.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d)
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
