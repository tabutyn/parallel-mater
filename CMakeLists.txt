cmake_minimum_required(VERSION 3.25)

project(ParallelMater VERSION 0.2.0 LANGUAGES CXX)

set(PARALLEL_MATER_IOS OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(PARALLEL_MATER_IOS ON)
endif()

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
if(BUILD_TESTING)
    add_test(NAME parallel-mater-reflected-abi-tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/solver/test_slang_abi.py")
    set_tests_properties(parallel-mater-reflected-abi-tests PROPERTIES LABELS "slang;abi")
    add_test(NAME parallel-mater-dxil-binding-generator-tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/solver/test_dxil_bindings.py")
    set_tests_properties(parallel-mater-dxil-binding-generator-tests
        PROPERTIES LABELS "slang;abi;d3d12")
    add_test(NAME parallel-mater-release-reproducibility-tests
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/release/test_release_reproducibility.py")
    set_tests_properties(parallel-mater-release-reproducibility-tests PROPERTIES
        LABELS "release;reproducibility")
endif()

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

if(NOT APPLE AND NOT WIN32 AND NOT ANDROID)
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
option(PARALLEL_MATER_BUILD_VULKAN
       "Build the loadable shared-Slang Vulkan qualification driver" OFF)
option(PARALLEL_MATER_BUILD_SCENE_SDK
       "Build optional host scene authoring helpers" OFF)
option(PARALLEL_MATER_BUILD_CONFORMANCE
       "Build the headless conformance runner and scene input SDK" ${BUILD_TESTING})
option(PARALLEL_MATER_BUILD_GALLERY
       "Build the backend-neutral gallery application" ${APPLE})
if(WIN32)
    option(PARALLEL_MATER_WINDOWS_GALLERY_AUTO_DEPLOY
           "Refresh the current-user Start Menu gallery after every build" ON)
else()
    set(PARALLEL_MATER_WINDOWS_GALLERY_AUTO_DEPLOY OFF)
endif()
option(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER
       "Build the serial Slang CPU reference driver" ${APPLE})
option(PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW
       "Build the limited Apple-GPU Gallery development driver" OFF)
option(PARALLEL_MATER_ENABLE_IOS_QUALIFICATION
       "Expose the development-only iPadOS public-API qualification channel" OFF)
if(PARALLEL_MATER_ENABLE_IOS_QUALIFICATION AND NOT PARALLEL_MATER_IOS)
    message(FATAL_ERROR "The iPadOS qualification channel requires an iOS build")
endif()
option(PARALLEL_MATER_ENABLE_ANDROID_QUALIFICATION
       "Build the development-only Android public-API qualification runner" OFF)
if(PARALLEL_MATER_ENABLE_ANDROID_QUALIFICATION AND NOT ANDROID)
    message(FATAL_ERROR "The Android qualification runner requires an Android build")
endif()
if(PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW)
    if(NOT APPLE)
        message(FATAL_ERROR "The Metal preview requires an Apple platform")
    endif()
    set(PARALLEL_MATER_BUILD_SLANG_MIRRORS ON CACHE BOOL "Compile shared shaders" FORCE)
endif()
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
       "Build and validate the shared Slang kernel compiler gates" ${BUILD_TESTING})
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
        "Build and validate the shared Slang kernel compiler gates" FORCE)
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

# Enable at the common parent scope: the Gallery and native shader tests both
# contain Objective-C++ sources. Subdirectory-only enablement breaks generation
# when both targets are present.
if(APPLE AND (PARALLEL_MATER_BUILD_GALLERY OR PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW OR
              (BUILD_TESTING AND PARALLEL_MATER_BUILD_SLANG_MIRRORS)))
    enable_language(OBJCXX)
endif()

# Reusable native compute services for kernel tests and the opt-in preview.
# Runtime schedules consume the C++ compute boundary, never Objective-C types.
if(APPLE AND ((BUILD_TESTING AND PARALLEL_MATER_BUILD_SLANG_MIRRORS) OR PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW))
    add_library(parallel-mater-device-metal STATIC src/device/metal_compute.mm)
    target_compile_features(parallel-mater-device-metal PUBLIC cxx_std_20)
    target_include_directories(parallel-mater-device-metal PUBLIC src)
    target_compile_options(parallel-mater-device-metal PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_link_libraries(parallel-mater-device-metal PRIVATE "-framework Metal" "-framework Foundation")
endif()

if(PARALLEL_MATER_BUILD_CUDA OR PARALLEL_MATER_BUILD_D3D12 OR PARALLEL_MATER_BUILD_VULKAN OR
   PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER OR
   PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS OR
   PARALLEL_MATER_BUILD_SLANG_MIRRORS)
    include(cmake/Slang.cmake)
    parallel_mater_require_slang()
endif()

if(PARALLEL_MATER_BUILD_GALLERY OR BUILD_TESTING)
    # One portable clock policy, wall-time collector, and CSV serializer.
    add_library(parallel-mater-gallery-frame-timing STATIC
        examples/support/frame_timing.cpp
        examples/support/frame_policy.cpp)
    target_compile_features(parallel-mater-gallery-frame-timing PUBLIC cxx_std_20)
    set_target_properties(parallel-mater-gallery-frame-timing PROPERTIES POSITION_INDEPENDENT_CODE YES)
    target_include_directories(parallel-mater-gallery-frame-timing
        PUBLIC include examples/support/include)
    parallel_mater_enable_warnings(parallel-mater-gallery-frame-timing)

endif()

# The conformance case registry intentionally has no CUDA or engine dependency.
# Every backend byte-checks and executes the same canonical inputs.
if(PARALLEL_MATER_BUILD_CONFORMANCE OR BUILD_TESTING OR
   PARALLEL_MATER_ENABLE_IOS_QUALIFICATION OR
   PARALLEL_MATER_ENABLE_ANDROID_QUALIFICATION)
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

endif()

set(PARALLEL_MATER_LOADER_INSTALL_TARGETS)
set(PARALLEL_MATER_VALIDATION_INSTALL_TARGETS)
set(PARALLEL_MATER_PREPARATION_INSTALL_TARGETS)
set(PARALLEL_MATER_SCENE_INSTALL_TARGETS)
set(PARALLEL_MATER_DRIVER_INSTALL_TARGETS)

include(cmake/ParallelMaterLoader.cmake)

# Scene parsing is pinned in-tree and needs no workstation dependency cache.
set(parallel_mater_cgltf_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cgltf")

if(PARALLEL_MATER_BUILD_CUDA OR PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER OR
   PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW OR
   PARALLEL_MATER_BUILD_D3D12 OR PARALLEL_MATER_BUILD_VULKAN OR
   PARALLEL_MATER_BUILD_GALLERY OR PARALLEL_MATER_BUILD_SCENE_SDK OR
   PARALLEL_MATER_BUILD_CONFORMANCE OR BUILD_TESTING)
    add_library(parallel-mater-host-preparation STATIC
        src/fluid_geometry.cpp
        src/fluid_source.cpp
        src/soft_body_geometry.cpp
        src/rope_geometry.cpp)
    target_compile_features(parallel-mater-host-preparation PUBLIC cxx_std_20)
    target_include_directories(parallel-mater-host-preparation
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    set_target_properties(parallel-mater-host-preparation PROPERTIES
        EXPORT_NAME preparation POSITION_INDEPENDENT_CODE ON)
    add_library(ParallelMater::preparation ALIAS parallel-mater-host-preparation)
    list(APPEND PARALLEL_MATER_PREPARATION_INSTALL_TARGETS
        parallel-mater-host-preparation)
    parallel_mater_enable_warnings(parallel-mater-host-preparation)
endif()

find_package(Threads REQUIRED)

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
    src/device/submission.cpp
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
    src/simulation/runtime.cpp
    src/simulation/runtime_domains.cu.inc
    src/simulation/runtime_deformables.cu.inc
    src/simulation/runtime_rigid.cu.inc
    src/simulation/runtime_schedule.cu.inc
    src/simulation/reference_schedule.hpp
    src/simulation/native_schedule_backend.cu.inc
    src/simulation/runtime_diagnostics.cu.inc
    src/simulation/geometry_storage.hpp
    src/simulation/rope_storage.hpp
    src/simulation/fluid_smoke_storage.hpp
    src/simulation/smoke_storage.hpp
    ${PARALLEL_MATER_SLANG_CUDA_SOURCES})

list(APPEND PARALLEL_MATER_DRIVER_INSTALL_TARGETS
    parallel-mater-driver-slang-cuda)
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
# Keep Slang's ordered multiply/add operations through native CUDA compilation.
# Use GPU reciprocal/square-root arithmetic as on the graphics shader targets;
# this is deliberately not --use_fast_math (which also enables contraction/FTZ).
target_compile_options(parallel-mater-driver-slang-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=false;--prec-sqrt=false>)
target_link_libraries(parallel-mater-driver-slang-cuda PRIVATE Threads::Threads)
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
    PARALLEL_MATER_DRIVER_IMPLEMENTATION=1
    PARALLEL_MATER_QUALIFIED_EQUAL_FLUID=1)
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
    if(WIN32)
        message(FATAL_ERROR
            "PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER currently requires a POSIX host")
    endif()
    include(cmake/ParallelMaterSlangCpu.cmake)
endif()
if(PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW)
    include(cmake/ParallelMaterSlangMetal.cmake)
endif()

set(PARALLEL_MATER_SOURCE_REVISION "" CACHE STRING
    "Source commit embedded in built Gallery and qualification artifacts")
if(PARALLEL_MATER_SOURCE_REVISION)
    string(LENGTH "${PARALLEL_MATER_SOURCE_REVISION}"
           PARALLEL_MATER_SOURCE_REVISION_LENGTH)
    if(NOT PARALLEL_MATER_SOURCE_REVISION MATCHES "^[0-9a-fA-F]+$" OR
       NOT PARALLEL_MATER_SOURCE_REVISION_LENGTH EQUAL 40)
        message(FATAL_ERROR
            "PARALLEL_MATER_SOURCE_REVISION must be a full 40-character commit")
    endif()
    string(TOLOWER "${PARALLEL_MATER_SOURCE_REVISION}"
           PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT)
else()
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
if(NOT PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT)
    set(PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT unknown)
endif()
execute_process(
    COMMAND git rev-parse --git-path HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE PARALLEL_MATER_GIT_HEAD_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(PARALLEL_MATER_GIT_HEAD_PATH)
    if(NOT IS_ABSOLUTE "${PARALLEL_MATER_GIT_HEAD_PATH}")
        set(PARALLEL_MATER_GIT_HEAD_PATH
            "${CMAKE_CURRENT_SOURCE_DIR}/${PARALLEL_MATER_GIT_HEAD_PATH}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                 "${PARALLEL_MATER_GIT_HEAD_PATH}")
endif()
execute_process(
    COMMAND git symbolic-ref -q HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE PARALLEL_MATER_GIT_HEAD_REF
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(PARALLEL_MATER_GIT_HEAD_REF)
    execute_process(
        COMMAND git rev-parse --git-path "${PARALLEL_MATER_GIT_HEAD_REF}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE PARALLEL_MATER_GIT_HEAD_REF_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(PARALLEL_MATER_GIT_HEAD_REF_PATH)
        if(NOT IS_ABSOLUTE "${PARALLEL_MATER_GIT_HEAD_REF_PATH}")
            set(PARALLEL_MATER_GIT_HEAD_REF_PATH
                "${CMAKE_CURRENT_SOURCE_DIR}/${PARALLEL_MATER_GIT_HEAD_REF_PATH}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${PARALLEL_MATER_GIT_HEAD_REF_PATH}")
    endif()
endif()

include(cmake/ParallelMaterScene.cmake)
include(cmake/ParallelMaterConformance.cmake)

if(PARALLEL_MATER_BUILD_GALLERY)
    add_library(parallel-mater-gallery-scene STATIC
                examples/gallery/render/render_scene.cpp
                examples/support/physics_debug.cpp
                examples/support/surface_query.cpp)
    target_compile_features(parallel-mater-gallery-scene PUBLIC cxx_std_20)
    target_include_directories(parallel-mater-gallery-scene
        PUBLIC include examples/support/include examples/gallery)
    target_link_libraries(parallel-mater-gallery-scene PUBLIC ParallelMater::scene)
    parallel_mater_enable_warnings(parallel-mater-gallery-scene)

    if(BUILD_TESTING)
        add_executable(parallel-mater-command-gallery-tests
                       tests/gallery/command_gallery_tests.cpp)
        target_link_libraries(parallel-mater-command-gallery-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-command-gallery-tests PRIVATE
            PARALLEL_MATER_GALLERY_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets")
        parallel_mater_enable_warnings(parallel-mater-command-gallery-tests)

        add_executable(parallel-mater-rigid-constraint-command-tests
                       tests/gallery/rigid_constraint_command_tests.cpp)
        target_link_libraries(parallel-mater-rigid-constraint-command-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(
            parallel-mater-rigid-constraint-command-tests PRIVATE
            PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintHinge.glb"
            PARALLEL_MATER_HINGE_CAPTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/assets/hinge_over_top_v5.pmcapture")
        parallel_mater_enable_warnings(
            parallel-mater-rigid-constraint-command-tests)

    endif()

    if(BUILD_TESTING AND WIN32 AND PARALLEL_MATER_BUILD_D3D12)
        foreach(test_target parallel-mater-command-gallery-tests
                parallel-mater-rigid-constraint-command-tests)
            add_dependencies(${test_target} parallel-mater-driver-slang-d3d12
                parallel-mater-layer-validation)
        endforeach()
        foreach(scene RigidBody ConstraintFixed ConstraintPoint ConstraintHinge
                ConstraintPiston ConstraintGeneric ConstraintMotorSpring DumpTruck
                Fluid FluidRigid Cloth ClothTear ClothWater)
            add_test(NAME parallel-mater-d3d12-gallery-${scene}
                COMMAND parallel-mater-command-gallery-tests --backend d3d12
                    --scene ${scene}.glb --frames 300)
            set_tests_properties(parallel-mater-d3d12-gallery-${scene} PROPERTIES
                LABELS "api;gallery;d3d12;regression" SKIP_RETURN_CODE 77 TIMEOUT 1800
                RUN_SERIAL TRUE
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-d3d12.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
        endforeach()
        add_test(NAME parallel-mater-d3d12-rigid-constraints
            COMMAND parallel-mater-rigid-constraint-command-tests --backend d3d12)
        set_tests_properties(parallel-mater-d3d12-rigid-constraints PROPERTIES
            LABELS "api;gallery;d3d12;constraints" SKIP_RETURN_CODE 77 TIMEOUT 3600
            RUN_SERIAL TRUE
            ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-d3d12.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
    endif()

    if(BUILD_TESTING AND PARALLEL_MATER_BUILD_CUDA)
        foreach(test_target parallel-mater-command-gallery-tests
                parallel-mater-rigid-constraint-command-tests)
            add_dependencies(${test_target} parallel-mater-driver-slang-cuda
                parallel-mater-layer-validation)
        endforeach()
        add_test(NAME parallel-mater-command-gallery-tests
                 COMMAND parallel-mater-command-gallery-tests)
        set_tests_properties(parallel-mater-command-gallery-tests PROPERTIES
            LABELS "api;gallery;cuda" SKIP_RETURN_CODE 77 TIMEOUT 300
            ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
        add_test(NAME parallel-mater-rigid-constraint-command-tests
                 COMMAND parallel-mater-rigid-constraint-command-tests)
        set_tests_properties(
            parallel-mater-rigid-constraint-command-tests PROPERTIES
            LABELS "api;gallery;cuda;constraints"
            SKIP_RETURN_CODE 77
            TIMEOUT 300
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")

        add_executable(parallel-mater-physics-debug-tests
                       tests/gallery/physics_debug_tests.cpp)
        target_link_libraries(parallel-mater-physics-debug-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-physics-debug-tests PRIVATE
            PARALLEL_MATER_HINGE_CAPTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/assets/hinge_over_top_v5.pmcapture")
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
                         parallel-mater-driver-slang-cuda
                         parallel-mater-layer-validation)
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
                         parallel-mater-driver-slang-cuda
                         parallel-mater-layer-validation)
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
                         parallel-mater-driver-slang-cuda
                         parallel-mater-layer-validation)
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
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cuda.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")

    endif()

    if(PARALLEL_MATER_GALLERY_RENDERER STREQUAL "OPTIX")
        FetchContent_Declare(
            pm_optix
            GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
            GIT_TAG fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(pm_optix)
    endif()

    if(NOT ANDROID AND NOT PARALLEL_MATER_IOS)
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

        set(pm_surface_cuda "${CMAKE_CURRENT_BINARY_DIR}/generated/gallery/fluid_surface_generated.cu")
        add_custom_command(OUTPUT "${pm_surface_cuda}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/gallery"
            COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/render/fluid_surface_cuda.slang"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/render"
                -target cuda -entry surface_index_particles -entry surface_build_field
                -o "${pm_surface_cuda}"
            DEPENDS examples/gallery/render/fluid_surface_cuda.slang
                examples/gallery/render/fluid_surface_field.slang
            VERBATIM)
        set_source_files_properties("${pm_surface_cuda}" PROPERTIES HEADER_FILE_ONLY TRUE)
        set(pm_gallery_host_renderer_sources
            examples/support/optix_renderer.cpp
            examples/support/foam_visuals.cpp
            examples/support/fluid_surface.cu "${pm_surface_cuda}")
        set(pm_gallery_host_renderer_optix ON)
    else()
        set(pm_gallery_host_renderer_sources
            examples/support/software_renderer.cpp)
        set(pm_gallery_host_renderer_optix OFF)
    endif()

    add_library(parallel-mater-gallery-renderer-contract STATIC
                examples/support/gpu_renderer.cpp)
    set_target_properties(parallel-mater-gallery-renderer-contract PROPERTIES
        POSITION_INDEPENDENT_CODE YES)
    target_include_directories(parallel-mater-gallery-renderer-contract
        PUBLIC include examples/support/include
        PRIVATE examples/support examples/gallery)
    target_link_libraries(parallel-mater-gallery-renderer-contract
        PUBLIC parallel-mater-gallery-scene)
    target_compile_features(parallel-mater-gallery-renderer-contract
        PUBLIC cxx_std_20)
    parallel_mater_enable_warnings(parallel-mater-gallery-renderer-contract)

    add_library(parallel-mater-gallery-renderer-adapters STATIC
                examples/support/renderer_adapter.cpp
                ${pm_gallery_host_renderer_sources})
    set_target_properties(parallel-mater-gallery-renderer-adapters PROPERTIES
        POSITION_INDEPENDENT_CODE YES)
    target_include_directories(parallel-mater-gallery-renderer-adapters
        PUBLIC include examples/support/include
        PRIVATE examples/support examples/gallery)
    target_link_libraries(parallel-mater-gallery-renderer-adapters
        PUBLIC parallel-mater-gallery-renderer-contract
        PRIVATE parallel-mater-gallery-frame-timing)
    target_compile_features(parallel-mater-gallery-renderer-adapters
        PUBLIC cxx_std_20)
    if(pm_gallery_host_renderer_optix)
        add_dependencies(parallel-mater-gallery-renderer-adapters
                         parallel-mater-optix-ptx)
        target_include_directories(
            parallel-mater-gallery-renderer-adapters
            PRIVATE
                examples/support
                "${CMAKE_CURRENT_BINARY_DIR}/generated/gallery"
                ${pm_optix_SOURCE_DIR}/include)
        target_link_libraries(
            parallel-mater-gallery-renderer-adapters
            PRIVATE CUDA::cuda_driver CUDA::cudart ${CMAKE_DL_LIBS})
        target_compile_definitions(parallel-mater-gallery-renderer-adapters
            PRIVATE
                PM_GALLERY_HOST_RENDERER_OPTIX=1
                PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}"
                $<$<PLATFORM_ID:Windows>:NOMINMAX>)
    endif()
    if(WIN32 AND PARALLEL_MATER_BUILD_D3D12)
        include(cmake/ParallelMaterD3D12Renderer.cmake)
    endif()
    parallel_mater_enable_warnings(parallel-mater-gallery-renderer-adapters)

    add_library(parallel-mater-gallery-session STATIC
                examples/support/gallery_runtime.cpp
                examples/support/mobile_session.cpp
                examples/support/mobile_qualification.cpp
                examples/support/session.cpp)
    set_target_properties(parallel-mater-gallery-session PROPERTIES POSITION_INDEPENDENT_CODE YES)
    target_link_libraries(parallel-mater-gallery-session
        PUBLIC parallel-mater-gallery-renderer-contract
               parallel-mater-gallery-frame-timing
        PRIVATE parallel-mater-gallery-renderer-adapters)
    target_include_directories(parallel-mater-gallery-session PRIVATE examples/gallery examples/support)
    target_compile_definitions(parallel-mater-gallery-session PRIVATE
        PM_GALLERY_BUILD_REVISION="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
    parallel_mater_enable_warnings(parallel-mater-gallery-session)
    if(APPLE AND PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW)
        target_sources(parallel-mater-gallery-renderer-adapters PRIVATE
            examples/support/metal_gallery_renderer.mm)
        # Xcode does not distinguish CXX/OBJCXX target flags in mixed targets.
        # Keep ARC explicit there as well as for Makefiles/Ninja generators.
        set_target_properties(parallel-mater-gallery-renderer-adapters PROPERTIES
            XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
        target_compile_options(parallel-mater-gallery-renderer-adapters PRIVATE
            "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_compile_definitions(parallel-mater-gallery-renderer-adapters
            PRIVATE PM_GALLERY_NATIVE_RENDERER_METAL=1)
        target_link_libraries(parallel-mater-gallery-renderer-adapters PRIVATE
            "-framework Metal" "-framework QuartzCore" "-framework Foundation")
    endif()

    if(ANDROID)
        add_subdirectory(examples/android/app/src/main/cpp)
    elseif(PARALLEL_MATER_IOS)
        include(cmake/ParallelMaterIOSGallery.cmake)
    else()
    add_library(parallel-mater-gallery-window STATIC)
    target_compile_features(parallel-mater-gallery-window PUBLIC cxx_std_20)
    target_include_directories(parallel-mater-gallery-window
        PUBLIC include examples/support/include)
    target_link_libraries(parallel-mater-gallery-window PUBLIC ParallelMater::parallel_mater)
    if(WIN32)
        target_sources(parallel-mater-gallery-window PRIVATE
            examples/support/d3d12_window.cpp)
        target_link_libraries(parallel-mater-gallery-window
            PRIVATE glfw d3d12 dxgi)
        target_compile_definitions(parallel-mater-gallery-window
            PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    elseif(APPLE)
        target_sources(parallel-mater-gallery-window PRIVATE examples/support/metal_window.mm)
        target_compile_options(parallel-mater-gallery-window PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(parallel-mater-gallery-window PRIVATE glfw "-framework Metal" "-framework QuartzCore" "-framework AppKit")
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
                              PRIVATE parallel-mater-gallery-renderer-adapters
                                      CUDA::cudart)
        target_compile_definitions(parallel-mater-dump-truck-benchmark PRIVATE
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb")
        parallel_mater_enable_warnings(parallel-mater-dump-truck-benchmark)
        add_executable(parallel-mater-rigid-frame-benchmark benchmarks/rigid_frame.cpp)
        target_link_libraries(parallel-mater-rigid-frame-benchmark
                              PRIVATE parallel-mater-gallery-renderer-adapters)
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
        PRIVATE glfw parallel-mater-gallery-window
                parallel-mater-gallery-frame-timing
                parallel-mater-gallery-session)

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
    elseif(WIN32)
        set(PARALLEL_MATER_GALLERY_ASSET_DIRECTORY "assets")
        set(PARALLEL_MATER_GALLERY_PROFILE_PATH
            "config/device-profiles.json")
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
            PARALLEL_MATER_GALLERY_ASSET_DIRECTORY="${PARALLEL_MATER_GALLERY_ASSET_DIRECTORY}"
            PARALLEL_MATER_DEVICE_PROFILES_PATH="${PARALLEL_MATER_GALLERY_PROFILE_PATH}"
            PARALLEL_MATER_GALLERY_VERSION="${PROJECT_VERSION}"
            PARALLEL_MATER_GALLERY_GIT_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
    if(NOT APPLE)
        if(WIN32)
            file(GLOB PARALLEL_MATER_WINDOWS_GALLERY_ASSETS
                 CONFIGURE_DEPENDS
                 "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/*.glb")
            add_custom_command(
                TARGET parallel-mater-gallery POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        "$<TARGET_FILE_DIR:parallel-mater-gallery>/assets"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${PARALLEL_MATER_WINDOWS_GALLERY_ASSETS}
                        "$<TARGET_FILE_DIR:parallel-mater-gallery>/assets"
                VERBATIM)
        endif()
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
        set(PARALLEL_MATER_WINDOWS_GALLERY_INSTALL_COMMAND
            powershell.exe -NoProfile -ExecutionPolicy Bypass
            -File "${CMAKE_CURRENT_SOURCE_DIR}/scripts/Install-WindowsGallery.ps1"
            -BuildDirectory "$<TARGET_FILE_DIR:parallel-mater-gallery>"
            -BuildConfiguration "$<CONFIG>"
            -Version "${PROJECT_VERSION}")
        set(PARALLEL_MATER_WINDOWS_GALLERY_DEPLOY_TRIGGER
            "${CMAKE_CURRENT_BINARY_DIR}/generated/windows_gallery_deploy.cpp")
        file(WRITE "${PARALLEL_MATER_WINDOWS_GALLERY_DEPLOY_TRIGGER}"
             "extern \"C\" const char *parallel_mater_windows_gallery_build_stamp() noexcept {\n"
             "    return __DATE__ \" \" __TIME__;\n"
             "}\n")
        set_source_files_properties(
            "${PARALLEL_MATER_WINDOWS_GALLERY_DEPLOY_TRIGGER}"
            PROPERTIES GENERATED TRUE)
        target_sources(parallel-mater-gallery PRIVATE
            "${PARALLEL_MATER_WINDOWS_GALLERY_DEPLOY_TRIGGER}")
        if(PARALLEL_MATER_WINDOWS_GALLERY_AUTO_DEPLOY)
            add_custom_target(parallel-mater-windows-gallery-deploy-trigger
                COMMAND ${CMAKE_COMMAND} -E touch
                    "${PARALLEL_MATER_WINDOWS_GALLERY_DEPLOY_TRIGGER}"
                VERBATIM)
            add_dependencies(parallel-mater-gallery
                             parallel-mater-windows-gallery-deploy-trigger)
            add_custom_command(TARGET parallel-mater-gallery POST_BUILD
                COMMAND ${PARALLEL_MATER_WINDOWS_GALLERY_INSTALL_COMMAND}
                COMMENT "Refreshing the installed Parallel-Mater gallery"
                VERBATIM)
            add_custom_target(install-windows-gallery
                DEPENDS parallel-mater-gallery)
        else()
            add_custom_target(install-windows-gallery
                COMMAND ${PARALLEL_MATER_WINDOWS_GALLERY_INSTALL_COMMAND}
                DEPENDS parallel-mater-gallery
                COMMENT "Installing Parallel-Mater for the current user"
                USES_TERMINAL
                VERBATIM)
        endif()
    endif()
    endif() # Desktop gallery shell
endif()

if(PARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS)
    include(cmake/SlangGeometryConstraints.cmake)
    parallel_mater_add_slang_geometry_constraints()
    if(WIN32 AND TARGET parallel-mater-gallery-window)
        add_dependencies(parallel-mater-gallery-window
                         parallel-mater-slang-avbd-dxil)
    endif()
endif()

if(PARALLEL_MATER_BUILD_VULKAN)
    include(cmake/ParallelMaterVulkan.cmake)
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
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_dxil_library.py"
                --manifest "${PARALLEL_MATER_D3D12_DXIL_MANIFEST}"
                --output "${PARALLEL_MATER_D3D12_DXIL_LIBRARY}"
        DEPENDS
            ${PARALLEL_MATER_SLANG_DXIL_SCHEDULES_DXIL}
            "${PARALLEL_MATER_D3D12_DXIL_MANIFEST}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_dxil_library.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/slang_abi.py"
        COMMENT "Embedding common Slang DXIL schedules in the D3D12 driver"
        VERBATIM)
    set(PARALLEL_MATER_D3D12_STUB_SOURCE
        "${PARALLEL_MATER_D3D12_GENERATED_DIR}/parallel_mater_unadvertised_dispatch.cpp")
    set(PARALLEL_MATER_D3D12_STUB_HEADERS
        src/simulation/fluid_rope_dispatch.hpp
        src/simulation/fluid_smoke_dispatch.hpp
        src/simulation/fluid_soft_body_dispatch.hpp
        src/simulation/geometry_smoke_dispatch.hpp
        src/simulation/rope_coupling_dispatch.hpp
        src/simulation/rope_dispatch.hpp
        src/simulation/smoke_coupling_dispatch.hpp
        src/simulation/smoke_grid_dispatch.hpp
        src/simulation/soft_body_cloth_dispatch.hpp)
    set(PARALLEL_MATER_D3D12_DOMAIN_SOURCES
        src/simulation/fluid_dispatch.cu
        src/simulation/cloth_dispatch.cu
        src/simulation/soft_body_dispatch.cu
        src/simulation/fluid_cloth_dispatch.cu
        src/simulation/geometry_fluid_dispatch.cu
        src/simulation/geometry_cloth_dispatch.cu
        src/simulation/geometry_soft_body_dispatch.cu
        src/simulation/exports.cu)
    set(PARALLEL_MATER_D3D12_DOMAIN_DISPATCH
        "${PARALLEL_MATER_D3D12_GENERATED_DIR}/parallel_mater_domain_dispatch.cpp")
    add_custom_command(
        OUTPUT "${PARALLEL_MATER_D3D12_DOMAIN_DISPATCH}"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_dxil_dispatch.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --reflection "${PARALLEL_MATER_SLANG_GENERATED_DIR}/dxil"
            --output "${PARALLEL_MATER_D3D12_DOMAIN_DISPATCH}"
            ${PARALLEL_MATER_D3D12_DOMAIN_SOURCES}
        DEPENDS tools/generate_dxil_dispatch.py tools/slang_abi.py
            ${PARALLEL_MATER_D3D12_DOMAIN_SOURCES}
            ${PARALLEL_MATER_SLANG_DXIL_SCHEDULES_DXIL}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Binding shared domain schedules using Slang DXIL reflection"
        VERBATIM)
    add_custom_command(
        OUTPUT "${PARALLEL_MATER_D3D12_STUB_SOURCE}"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_dispatch_stubs.py"
                --output "${PARALLEL_MATER_D3D12_STUB_SOURCE}"
                ${PARALLEL_MATER_D3D12_STUB_HEADERS}
        DEPENDS tools/generate_cpu_dispatch_stubs.py
                ${PARALLEL_MATER_D3D12_STUB_HEADERS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating guarded stubs for unadvertised D3D12 domains"
        VERBATIM)
    set_source_files_properties("${PARALLEL_MATER_D3D12_STUB_SOURCE}"
        PROPERTIES GENERATED TRUE COMPILE_OPTIONS "/w")
    add_library(parallel-mater-driver-slang-d3d12 SHARED
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
        src/driver/d3d12/slang_command.cpp
        src/driver/d3d12/schedule_context.cpp
        src/device/d3d12_device.cpp
        src/device/submission.cpp
        src/simulation/runtime.cpp
        src/simulation/d3d12_geometry_dispatch.cpp
        src/simulation/memory_arena.cpp
        "${PARALLEL_MATER_D3D12_STUB_SOURCE}"
        "${PARALLEL_MATER_D3D12_DOMAIN_DISPATCH}"
        "${PARALLEL_MATER_D3D12_DXIL_LIBRARY}")
    add_dependencies(parallel-mater-driver-slang-d3d12
                     parallel-mater-slang-dxil-schedules)
    target_compile_features(parallel-mater-driver-slang-d3d12 PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-driver-slang-d3d12 PRIVATE
        PARALLEL_MATER_DRIVER_IMPLEMENTATION=1
        PARALLEL_MATER_D3D12_DRIVER=1
        NOMINMAX
        WIN32_LEAN_AND_MEAN)
    target_include_directories(parallel-mater-driver-slang-d3d12
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${PARALLEL_MATER_API_GENERATED_DIR}
            "${PARALLEL_MATER_D3D12_GENERATED_DIR}")
    target_link_libraries(parallel-mater-driver-slang-d3d12 PRIVATE
        parallel-mater-host-preparation Threads::Threads d3d12 dxgi)
    parallel_mater_enable_warnings(parallel-mater-driver-slang-d3d12)
    set_target_properties(parallel-mater-driver-slang-d3d12 PROPERTIES
        OUTPUT_NAME parallel_mater_driver_slang_d3d12
        EXPORT_NAME slang_d3d12_driver
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON)
    add_library(ParallelMater::slang_d3d12_driver ALIAS
                parallel-mater-driver-slang-d3d12)
    if(TARGET parallel-mater-gallery)
        add_dependencies(parallel-mater-gallery
                         parallel-mater-driver-slang-d3d12)
    endif()
    list(APPEND PARALLEL_MATER_DRIVER_INSTALL_TARGETS
         parallel-mater-driver-slang-d3d12)
    file(GENERATE
        OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-d3d12.json"
        CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-driver-slang-d3d12>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
    file(GENERATE
        OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-d3d12.json"
        CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"../$<TARGET_FILE_NAME:parallel-mater-driver-slang-d3d12>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
    if(BUILD_TESTING)
        add_executable(parallel-mater-d3d12-command-api-tests
            tests/api/native_command_api_tests.cpp)
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
    include(cmake/SlangMetal.cmake)
    parallel_mater_add_slang_metal_kernels()
    add_dependencies(parallel-mater-slang-kernel-mirrors
        parallel-mater-slang-metal-kernels)
    if(PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW)
        set(metal_resource_directory "$<TARGET_FILE_DIR:parallel-mater-driver-slang-metal>/slang-metal")
        set(metal_resource_arguments "-DDESTINATION=${metal_resource_directory}")
        if(PARALLEL_MATER_IOS)
            # Xcode's target-directory generator expressions contain a literal
            # EFFECTIVE_PLATFORM_NAME in custom targets. Resolve the
            # platform suffix from the build environment instead.
            set(metal_resource_arguments
                "-DDESTINATION=$<TARGET_BUNDLE_DIR:parallel-mater-driver-slang-metal>/slang-metal"
                "-DXCODE_RESOLVE_PLATFORM=ON")
        endif()
        add_custom_target(parallel-mater-slang-metal-resources ALL
            COMMAND ${CMAKE_COMMAND}
                -DMANIFEST=${PARALLEL_MATER_SLANG_METAL_MANIFEST}
                ${metal_resource_arguments}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackMetalKernels.cmake
            DEPENDS parallel-mater-slang-metal-kernels VERBATIM)
        add_dependencies(parallel-mater-driver-slang-metal parallel-mater-slang-metal-resources)
    endif()
    if(PARALLEL_MATER_SLANG_TEST_BUILD AND TARGET parallel-mater-gallery)
        add_dependencies(parallel-mater-gallery
                         parallel-mater-slang-kernel-mirrors)
    endif()
endif()

if(BUILD_TESTING)
    if(TARGET parallel-mater-gallery-renderer-contract)
        add_executable(parallel-mater-gallery-gpu-renderer-contract-tests
            tests/gallery/gpu_renderer_contract_tests.cpp)
        target_link_libraries(
            parallel-mater-gallery-gpu-renderer-contract-tests PRIVATE
            parallel-mater-gallery-renderer-contract)
        parallel_mater_enable_warnings(
            parallel-mater-gallery-gpu-renderer-contract-tests)
        add_test(NAME parallel-mater-gallery-gpu-renderer-contract-tests
            COMMAND parallel-mater-gallery-gpu-renderer-contract-tests)
        set_tests_properties(
            parallel-mater-gallery-gpu-renderer-contract-tests PROPERTIES
            LABELS "gallery;renderer;contract")
    endif()
    if(TARGET parallel-mater-gallery-renderer-adapters AND
       PARALLEL_MATER_GALLERY_RENDERER STREQUAL "SOFTWARE")
        add_executable(parallel-mater-software-renderer-tests
            tests/gallery/software_renderer_tests.cpp)
        target_compile_features(parallel-mater-software-renderer-tests
            PRIVATE cxx_std_20)
        target_compile_definitions(parallel-mater-software-renderer-tests PRIVATE
            PARALLEL_MATER_TEST_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Celestial.glb")
        target_link_libraries(parallel-mater-software-renderer-tests PRIVATE
            parallel-mater-gallery-renderer-adapters)
        parallel_mater_enable_warnings(parallel-mater-software-renderer-tests)
        add_test(NAME parallel-mater-software-renderer-tests
            COMMAND parallel-mater-software-renderer-tests)
        set_tests_properties(parallel-mater-software-renderer-tests PROPERTIES
            LABELS "gallery;renderer;host")
    endif()
    add_executable(parallel-mater-gallery-workload-tests
        tests/gallery/workload_profile_tests.cpp examples/support/device_profiles.cpp)
    target_compile_features(parallel-mater-gallery-workload-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-gallery-workload-tests PRIVATE include examples/support/include)
    target_compile_definitions(parallel-mater-gallery-workload-tests PRIVATE
        PM_PROFILE_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json")
    parallel_mater_enable_warnings(parallel-mater-gallery-workload-tests)
    add_test(NAME parallel-mater-gallery-workload-tests COMMAND parallel-mater-gallery-workload-tests)
    if(NOT CMAKE_CROSSCOMPILING AND TARGET parallel-mater-driver-slang-metal AND
       TARGET parallel-mater-gallery-renderer-adapters)
        set(ipad_test_resources "${CMAKE_CURRENT_BINARY_DIR}/ipad-test-resources")
        file(MAKE_DIRECTORY "${ipad_test_resources}/config")
        configure_file(config/device-profiles.json "${ipad_test_resources}/config/device-profiles.json" COPYONLY)
        if(NOT EXISTS "${ipad_test_resources}/assets")
            file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets" "${ipad_test_resources}/assets" SYMBOLIC)
        endif()
        add_executable(parallel-mater-ipad-session-tests
            tests/gallery/ipad_session_tests.cpp)
        target_link_libraries(parallel-mater-ipad-session-tests PRIVATE
            parallel-mater-gallery-session)
        target_compile_definitions(parallel-mater-ipad-session-tests PRIVATE
            PM_IPAD_TEST_RESOURCES="${ipad_test_resources}")
        add_dependencies(parallel-mater-ipad-session-tests parallel-mater-driver-slang-metal parallel-mater-layer-validation)
        add_test(NAME parallel-mater-ipad-session-tests COMMAND parallel-mater-ipad-session-tests)
        set_tests_properties(parallel-mater-ipad-session-tests PROPERTIES TIMEOUT 600
            LABELS "gallery;session;metal"
            ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-metal.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
    endif()
    if(NOT CMAKE_CROSSCOMPILING AND TARGET parallel-mater-gallery-session)
        unset(gallery_session_test_driver)
        unset(gallery_session_test_native_backend)
        if(TARGET parallel-mater-driver-slang-vulkan)
            set(gallery_session_test_driver parallel-mater-driver-slang-vulkan)
            set(gallery_session_test_manifest slang-vulkan.json)
            set(gallery_session_test_native_backend PM_NATIVE_BACKEND_VULKAN)
            set(gallery_session_test_backend vulkan)
        elseif(TARGET parallel-mater-driver-slang-d3d12)
            set(gallery_session_test_driver parallel-mater-driver-slang-d3d12)
            set(gallery_session_test_manifest slang-d3d12.json)
            set(gallery_session_test_native_backend PM_NATIVE_BACKEND_D3D12)
            set(gallery_session_test_backend d3d12)
        elseif(TARGET parallel-mater-driver-slang-metal)
            set(gallery_session_test_driver parallel-mater-driver-slang-metal)
            set(gallery_session_test_manifest slang-metal.json)
            set(gallery_session_test_native_backend PM_NATIVE_BACKEND_METAL)
            set(gallery_session_test_backend metal)
        elseif(TARGET parallel-mater-driver-slang-cuda)
            set(gallery_session_test_driver parallel-mater-driver-slang-cuda)
            set(gallery_session_test_manifest slang-cuda.json)
            set(gallery_session_test_native_backend PM_NATIVE_BACKEND_CUDA)
            set(gallery_session_test_backend cuda)
        elseif(TARGET parallel-mater-driver-slang-cpu)
            set(gallery_session_test_driver parallel-mater-driver-slang-cpu)
            set(gallery_session_test_manifest slang-cpu.json)
            set(gallery_session_test_backend slang-cpu)
        endif()
        if(gallery_session_test_driver)
            add_executable(parallel-mater-gallery-session-contract-tests
                tests/gallery/gallery_session_contract_tests.cpp)
            target_link_libraries(parallel-mater-gallery-session-contract-tests PRIVATE
                parallel-mater-gallery-session)
            target_compile_definitions(parallel-mater-gallery-session-contract-tests PRIVATE
                PM_GALLERY_SESSION_TEST_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
                PM_GALLERY_SESSION_TEST_BACKEND="${gallery_session_test_backend}")
            if(gallery_session_test_native_backend)
                target_compile_definitions(parallel-mater-gallery-session-contract-tests PRIVATE
                    PM_GALLERY_SESSION_TEST_NATIVE_BACKEND=${gallery_session_test_native_backend})
            endif()
            add_dependencies(parallel-mater-gallery-session-contract-tests
                ${gallery_session_test_driver} parallel-mater-layer-validation)
            add_test(NAME parallel-mater-gallery-session-contract-tests
                COMMAND parallel-mater-gallery-session-contract-tests)
            set_tests_properties(parallel-mater-gallery-session-contract-tests PROPERTIES
                LABELS "gallery;runtime;contract" TIMEOUT 120
                SKIP_RETURN_CODE 77
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/${gallery_session_test_manifest};PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
            if(gallery_session_test_backend STREQUAL "vulkan")
                set_tests_properties(parallel-mater-gallery-session-contract-tests
                    PROPERTIES TIMEOUT 600 RUN_SERIAL TRUE)
            endif()
        endif()
    endif()
    if(TARGET parallel-mater-command-gallery-tests AND
       TARGET parallel-mater-driver-slang-vulkan)
        add_dependencies(parallel-mater-command-gallery-tests
                         parallel-mater-driver-slang-vulkan
                         parallel-mater-layer-validation)
        foreach(scene Cloth ClothTear ClothPaint ClothWater)
            add_test(NAME parallel-mater-vulkan-gallery-${scene}
                COMMAND parallel-mater-command-gallery-tests --backend vulkan
                    --scene ${scene}.glb --frames 120)
            set_tests_properties(parallel-mater-vulkan-gallery-${scene} PROPERTIES
                LABELS "api;gallery;vulkan;cloth" SKIP_RETURN_CODE 77 TIMEOUT 300
                RUN_SERIAL TRUE
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
        endforeach()
        foreach(scene Softbody SoftbodyRigidBody SoftbodyFluid)
            add_test(NAME parallel-mater-vulkan-gallery-${scene}
                COMMAND parallel-mater-command-gallery-tests --backend vulkan
                    --scene ${scene}.glb --frames 120)
            set_tests_properties(parallel-mater-vulkan-gallery-${scene} PROPERTIES
                LABELS "api;gallery;vulkan;soft-body" SKIP_RETURN_CODE 77 TIMEOUT 300
                RUN_SERIAL TRUE
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
        endforeach()
        foreach(scene Rope RopeFluid)
            add_test(NAME parallel-mater-vulkan-gallery-${scene}
                COMMAND parallel-mater-command-gallery-tests --backend vulkan
                    --scene ${scene}.glb --frames 120)
            set_tests_properties(parallel-mater-vulkan-gallery-${scene} PROPERTIES
                LABELS "api;gallery;vulkan;rope" SKIP_RETURN_CODE 77 TIMEOUT 300
                RUN_SERIAL TRUE
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
        endforeach()
        foreach(scene Smoke SmokeWater)
            add_test(NAME parallel-mater-vulkan-gallery-${scene}
                COMMAND parallel-mater-command-gallery-tests --backend vulkan
                    --scene ${scene}.glb --frames 120)
            set_tests_properties(parallel-mater-vulkan-gallery-${scene} PROPERTIES
                LABELS "api;gallery;vulkan;smoke" SKIP_RETURN_CODE 77 TIMEOUT 300
                RUN_SERIAL TRUE
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
        endforeach()
        foreach(scene SoftbodyCloth RopeSoftbody RopeCloth SmokeSoftbody SmokeCloth SmokeRope)
            add_test(NAME parallel-mater-vulkan-gallery-${scene}
                COMMAND parallel-mater-command-gallery-tests --backend vulkan
                    --scene ${scene}.glb --frames 120)
            set_tests_properties(parallel-mater-vulkan-gallery-${scene} PROPERTIES
                LABELS "api;gallery;vulkan;cross-domain" SKIP_RETURN_CODE 77 TIMEOUT 300
                RUN_SERIAL TRUE
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
        endforeach()
    endif()
    if(PARALLEL_MATER_BUILD_GALLERY AND PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER AND
       PARALLEL_MATER_SLANG_CPU_LIQUIDS)
        add_executable(parallel-mater-mobile-liquid-tests tests/gallery/mobile_liquid_tests.cpp)
        target_link_libraries(parallel-mater-mobile-liquid-tests PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-mobile-liquid-tests PRIVATE
            PM_PROFILE_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
            PM_SCENE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets")
        add_dependencies(parallel-mater-mobile-liquid-tests
            parallel-mater-driver-slang-cpu parallel-mater-layer-validation)
        add_test(NAME parallel-mater-mobile-liquid-tests COMMAND parallel-mater-mobile-liquid-tests)
        set_tests_properties(parallel-mater-mobile-liquid-tests PROPERTIES TIMEOUT 120
            LABELS "gallery;mobile;slang-cpu"
            ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cpu.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
    endif()
    if(PARALLEL_MATER_BUILD_GALLERY AND NOT CMAKE_CROSSCOMPILING)
        set(parallel_mater_has_physical_gallery_driver FALSE)
        foreach(driver_target
                parallel-mater-driver-slang-cuda
                parallel-mater-driver-slang-d3d12
                parallel-mater-driver-slang-vulkan
                parallel-mater-driver-slang-metal
                parallel-mater-driver-slang-cpu)
            if(TARGET ${driver_target})
                set(parallel_mater_has_physical_gallery_driver TRUE)
            endif()
        endforeach()
        if(parallel_mater_has_physical_gallery_driver)
            add_test(NAME parallel-mater-gallery-runtime-manifest-contract
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/ctest_manifest_contract.py"
                    --ctest "${CMAKE_CTEST_COMMAND}"
                    --build "${CMAKE_BINARY_DIR}"
                    --config "$<CONFIG>"
                    --manifest "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json"
                    --library "$<TARGET_FILE:parallel-mater-layer-validation>")
            set_tests_properties(
                parallel-mater-gallery-runtime-manifest-contract PROPERTIES
                LABELS "gallery;contract;manifest")
        endif()
    endif()
    add_executable(parallel-mater-submission-queue-tests
        tests/api/submission_queue_tests.cpp src/device/submission.cpp)
    target_compile_features(parallel-mater-submission-queue-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-submission-queue-tests PRIVATE include src)
    target_link_libraries(parallel-mater-submission-queue-tests PRIVATE Threads::Threads)
    parallel_mater_enable_warnings(parallel-mater-submission-queue-tests)
    add_test(NAME parallel-mater-submission-queue-tests
        COMMAND parallel-mater-submission-queue-tests)
    set_tests_properties(parallel-mater-submission-queue-tests PROPERTIES TIMEOUT 30)
    add_subdirectory(tests/solver)
    add_subdirectory(tests/api)
    add_executable(parallel-mater-gallery-frame-timing-tests
        tests/gallery/frame_timing_tests.cpp)
    add_test(NAME parallel-mater-gallery-deployment-tests
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/deployment_tests.py")
    if(UNIX AND NOT APPLE)
        add_test(NAME parallel-mater-linux-gallery-launcher-tests
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/linux_launcher_tests.py")
        set_tests_properties(parallel-mater-linux-gallery-launcher-tests PROPERTIES
            LABELS "gallery;linux;install")
    endif()
    target_link_libraries(parallel-mater-gallery-frame-timing-tests PRIVATE
        parallel-mater-gallery-frame-timing)
    parallel_mater_enable_warnings(parallel-mater-gallery-frame-timing-tests)
    add_test(NAME parallel-mater-gallery-frame-timing-tests
        COMMAND parallel-mater-gallery-frame-timing-tests)
    if(TARGET parallel-mater-gallery-application)
        add_executable(parallel-mater-gallery-frame-timing-overlay-tests
            tests/gallery/frame_timing_overlay_tests.cpp)
        target_link_libraries(parallel-mater-gallery-frame-timing-overlay-tests PRIVATE
            parallel-mater-gallery-application parallel-mater-gallery-frame-timing)
        parallel_mater_enable_warnings(parallel-mater-gallery-frame-timing-overlay-tests)
        add_test(NAME parallel-mater-gallery-frame-timing-overlay-tests
            COMMAND parallel-mater-gallery-frame-timing-overlay-tests)
    endif()
    if(TARGET parallel-mater-gallery AND
       (TARGET parallel-mater-driver-slang-cpu OR TARGET parallel-mater-driver-slang-cuda OR
        TARGET parallel-mater-driver-slang-metal))
        add_test(NAME parallel-mater-gallery-frame-timing-headless
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/frame_timing_headless.py"
                $<TARGET_FILE:parallel-mater-gallery>)
        set_tests_properties(parallel-mater-gallery-frame-timing-headless PROPERTIES
            LABELS "gallery;timing;runtime" TIMEOUT 180)
    endif()
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
        PRIVATE include examples/support/include)
    add_test(NAME parallel-mater-physics-frame-budget-tests
        COMMAND parallel-mater-physics-frame-budget-tests)

    add_executable(parallel-mater-gallery-frame-policy-tests
        tests/gallery/frame_policy_tests.cpp)
    target_link_libraries(parallel-mater-gallery-frame-policy-tests PRIVATE
        parallel-mater-gallery-frame-timing)
    parallel_mater_enable_warnings(parallel-mater-gallery-frame-policy-tests)
    add_test(NAME parallel-mater-gallery-frame-policy-tests
        COMMAND parallel-mater-gallery-frame-policy-tests)

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
    add_test(NAME parallel-mater-mobile-qualification-schema-tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery
                -p test_mobile_qualification_report.py)
    set_tests_properties(parallel-mater-mobile-qualification-schema-tests
        PROPERTIES LABELS "gallery;mobile;qualification;contract")
    add_test(NAME parallel-mater-android-gallery-ui-tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery
                -p test_android_gallery_ui.py)
    set_tests_properties(parallel-mater-android-gallery-ui-tests
        PROPERTIES LABELS "gallery;mobile;android;ui")
    add_test(NAME parallel-mater-android-vulkan-schedule-profile-tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery
                -p android_vulkan_schedule_tests.py)
    set_tests_properties(parallel-mater-android-vulkan-schedule-profile-tests
        PROPERTIES LABELS "gallery;mobile;android;vulkan;performance")
    add_test(NAME parallel-mater-conformance-case-registry
        COMMAND parallel-mater-conformance-cases --check
                ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/cases)
    add_test(NAME parallel-mater-conformance-comparator-tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/tests
                -p test_*.py)
    add_test(NAME parallel-mater-benchmark-qualification-tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/tests
                -p test_*.py)
    set_tests_properties(parallel-mater-benchmark-qualification-tests PROPERTIES
        LABELS "benchmark;conformance")

    add_executable(parallel-mater-types-contract-tests
                   tests/types_contract.cpp)
    target_compile_features(parallel-mater-types-contract-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-types-contract-tests
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
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

if(PARALLEL_MATER_BUILD_CONFORMANCE OR PARALLEL_MATER_BUILD_BENCHMARKS)
    configure_file(cmake/ParallelMaterBenchmarkComponent.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterBenchmarkComponent.cmake"
        @ONLY)
    install(FILES benchmarks/qualification.py
        DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/benchmarks
        COMPONENT benchmark)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterBenchmarkComponent.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater
        COMPONENT benchmark)
endif()
if(TARGET parallel-mater-dump-truck-benchmark)
    install(TARGETS parallel-mater-dump-truck-benchmark
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT benchmark)
endif()
if(TARGET parallel-mater-rigid-frame-benchmark)
    install(TARGETS parallel-mater-rigid-frame-benchmark
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT benchmark)
endif()

configure_package_config_file(
    cmake/ParallelMaterConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

if(WIN32)
    set(PARALLEL_MATER_SHARED_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
else()
    set(PARALLEL_MATER_SHARED_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()
set(PARALLEL_MATER_INSTALLED_LAYER_PATH
    "../../../${PARALLEL_MATER_SHARED_INSTALL_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_layer_validation${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(config/validation-layer.json.in
    ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_validation.json @ONLY)
if(PARALLEL_MATER_BUILD_CUDA)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${PARALLEL_MATER_SHARED_INSTALL_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_cuda${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cuda.json @ONLY)
endif()
if(WIN32 AND PARALLEL_MATER_BUILD_D3D12)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${PARALLEL_MATER_SHARED_INSTALL_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_d3d12${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_d3d12.json @ONLY)
endif()
if(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${PARALLEL_MATER_SHARED_INSTALL_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_cpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cpu.json @ONLY)
endif()
if(PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${PARALLEL_MATER_SHARED_INSTALL_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_metal${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_metal.json @ONLY)
endif()

if(PARALLEL_MATER_BUILD_VULKAN)
    set(PARALLEL_MATER_INSTALLED_DRIVER_PATH
        "../../../${PARALLEL_MATER_SHARED_INSTALL_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}parallel_mater_driver_slang_vulkan${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(config/driver-manifest.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_vulkan.json @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_vulkan.json
        DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d
        COMPONENT driver)
endif()

function(parallel_mater_install_export component export_name)
    set(targets ${ARGN})
    if(NOT targets)
        return()
    endif()
    install(
        TARGETS ${targets}
        EXPORT ${export_name}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${component}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(
        EXPORT ${export_name}
        FILE ${export_name}.cmake
        NAMESPACE ParallelMater::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater
        COMPONENT ${component})
endfunction()

parallel_mater_install_export(loader ParallelMaterLoaderTargets
    ${PARALLEL_MATER_LOADER_INSTALL_TARGETS})
parallel_mater_install_export(validation ParallelMaterValidationTargets
    ${PARALLEL_MATER_VALIDATION_INSTALL_TARGETS})
parallel_mater_install_export(preparation ParallelMaterPreparationTargets
    ${PARALLEL_MATER_PREPARATION_INSTALL_TARGETS})
parallel_mater_install_export(scene ParallelMaterSceneTargets
    ${PARALLEL_MATER_SCENE_INSTALL_TARGETS})
parallel_mater_install_export(driver ParallelMaterDriverTargets
    ${PARALLEL_MATER_DRIVER_INSTALL_TARGETS})

install(
    FILES
        include/parallel_mater/parallel_mater.h
        include/parallel_mater/interop_vulkan.h
        include/parallel_mater/pm_driver.h
        include/parallel_mater/pm_layer.h
        ${PARALLEL_MATER_API_GENERATED_DIR}/parallel_mater/pm_layer_args.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parallel_mater
    COMPONENT loader)
if(TARGET parallel-mater-host-preparation)
    install(DIRECTORY include/parallel_mater_scene
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT preparation)
endif()
if(TARGET parallel-mater-scene)
    install(FILES third_party/cgltf/LICENSE third_party/cgltf/README.parallel-mater.md
        DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/licenses/cgltf
        COMPONENT scene)
endif()
if(PARALLEL_MATER_BUILD_CUDA)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cuda.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d
            COMPONENT driver)
    if(WIN32)
        install(FILES "${PARALLEL_MATER_CUDART_RUNTIME}"
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT driver)
    endif()
endif()
if(WIN32 AND PARALLEL_MATER_BUILD_D3D12)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_d3d12.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d
            COMPONENT driver)
endif()
if(PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_cpu.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d
            COMPONENT driver)
endif()
if(PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_slang_metal.json
            DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/drivers.d
            COMPONENT driver)
    if(NOT PARALLEL_MATER_IOS)
        install(DIRECTORY "${metal_resource_directory}/"
                DESTINATION ${PARALLEL_MATER_SHARED_INSTALL_DIR}/slang-metal
                COMPONENT driver)
    endif()
endif()
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parallel_mater_validation.json
        DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/layers.d
        COMPONENT validation)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/licenses
        COMPONENT loader)
install(FILES tools/blender/export_parallel_mater_scene.py
        DESTINATION ${CMAKE_INSTALL_DATADIR}/parallel-mater/blender
        COMPONENT scene)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater
    COMPONENT loader)
