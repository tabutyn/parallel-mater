cmake_minimum_required(VERSION 3.25)

project(ParallelMater VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)
include(CTest)
include(CheckLanguage)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

if(APPLE)
    set(PARALLEL_MATER_CUDA_DEFAULT OFF)
    set(PARALLEL_MATER_METAL_DEFAULT ON)
    set(PARALLEL_MATER_D3D12_DEFAULT OFF)
elseif(WIN32)
    set(PARALLEL_MATER_CUDA_DEFAULT OFF)
    set(PARALLEL_MATER_METAL_DEFAULT OFF)
    set(PARALLEL_MATER_D3D12_DEFAULT ON)
else()
    set(PARALLEL_MATER_CUDA_DEFAULT ON)
    set(PARALLEL_MATER_METAL_DEFAULT OFF)
    set(PARALLEL_MATER_D3D12_DEFAULT OFF)
endif()

option(PARALLEL_MATER_BUILD_CUDA
       "Build the CUDA physics backend" ${PARALLEL_MATER_CUDA_DEFAULT})
option(PARALLEL_MATER_BUILD_METAL
       "Build the Apple Metal 4 physics backend" ${PARALLEL_MATER_METAL_DEFAULT})
option(PARALLEL_MATER_BUILD_D3D12
       "Build the Windows D3D12 physics backend" ${PARALLEL_MATER_D3D12_DEFAULT})
option(PARALLEL_MATER_BUILD_METAL_GALLERY
       "Build the native Metal gallery" OFF)
option(PARALLEL_MATER_BUILD_D3D12_GALLERY
       "Build the native D3D12 gallery" OFF)
option(PARALLEL_MATER_BUILD_OPTIX_GALLERY
       "Build the optional OptiX and glTF gallery" OFF)
option(PARALLEL_MATER_BUILD_BENCHMARKS
       "Build local performance benchmark executables" OFF)
option(PARALLEL_MATER_BUILD_SLANG_AVBD
       "Build and validate the portable Slang AVBD core" ${BUILD_TESTING})

# Ship prebuilt icons so ordinary builds need no image conversion tools.
if(WIN32 AND (PARALLEL_MATER_BUILD_D3D12_GALLERY OR
              PARALLEL_MATER_BUILD_OPTIX_GALLERY))
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
if(PARALLEL_MATER_BUILD_METAL_GALLERY AND NOT PARALLEL_MATER_BUILD_METAL)
    message(FATAL_ERROR "The Metal gallery requires PARALLEL_MATER_BUILD_METAL")
endif()
if(PARALLEL_MATER_BUILD_D3D12_GALLERY AND NOT PARALLEL_MATER_BUILD_D3D12)
    message(FATAL_ERROR "The D3D12 gallery requires PARALLEL_MATER_BUILD_D3D12")
endif()
if(PARALLEL_MATER_BUILD_BENCHMARKS AND
   NOT PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    message(FATAL_ERROR
            "Rigid scene benchmarks require PARALLEL_MATER_BUILD_OPTIX_GALLERY")
endif()

if(PARALLEL_MATER_BUILD_CUDA OR PARALLEL_MATER_BUILD_METAL OR
   PARALLEL_MATER_BUILD_SLANG_AVBD)
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

if(PARALLEL_MATER_BUILD_OPTIX_GALLERY OR
   PARALLEL_MATER_BUILD_METAL_GALLERY OR
   PARALLEL_MATER_BUILD_D3D12_GALLERY OR
   (PARALLEL_MATER_BUILD_CUDA AND BUILD_TESTING))
    FetchContent_Declare(
        parallel_mater_cgltf
        GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
        GIT_TAG 85cd62382dfea638278962690cf515023f33ed00
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(parallel_mater_cgltf)
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
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_fluid.cu)
set(PARALLEL_MATER_AVBD_CUDA_SOURCE
    ${PARALLEL_MATER_SLANG_GENERATED_DIR}/parallel_mater_avbd_runtime.cu)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_FLUID_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/fluid_shared.slang
            -DPM_TARGET_METAL=0 -DPM_TARGET_CUDA=1
            -entry pm_fluid_cell_keys
            -entry pm_fluid_forces
            -entry pm_fluid_integrate
            -target cuda
            -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_FLUID_CUDA_SOURCE}
    DEPENDS src/slang/fluid_shared.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared fluid kernels from Slang to CUDA"
    VERBATIM)
add_custom_command(
    OUTPUT ${PARALLEL_MATER_AVBD_CUDA_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PARALLEL_MATER_SLANG_GENERATED_DIR}
    COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/avbd.slang
            -entry avbdConformanceMain -stage compute
            -target cuda -fp-mode precise -O2 -line-directive-mode none
            -o ${PARALLEL_MATER_AVBD_CUDA_SOURCE}
    DEPENDS src/slang/avbd.slang ${PARALLEL_MATER_SLANGC_EXECUTABLE}
    COMMENT "Compiling shared AVBD core from Slang to CUDA"
    VERBATIM)
add_library(
    parallel-mater STATIC
    src/world.cu
    src/geometry_constraints.cuh
    src/geometry_fluid.cuh
    src/geometry_soft_body.cuh
    src/geometry_cloth.cuh
    src/geometry_rope.cuh
    src/geometry_smoke.cuh
    src/fluid.cuh
    src/fluid_soft_body.cuh
    src/fluid_cloth.cuh
    src/fluid_rope.cuh
    src/fluid_smoke.cuh
    src/soft_body.cuh
    src/soft_body_cloth.cuh
    src/soft_body_rope.cuh
    src/soft_body_smoke.cuh
    src/cloth.cuh
    src/cloth_rope.cuh
    src/cloth_smoke.cuh
    src/rope_smoke.cuh
    src/smoke_grid.cuh
    src/fluid_geometry.cpp
    src/fluid_source.cpp
    src/soft_body_geometry.cpp
    src/rope_geometry.cpp)
set_source_files_properties(
    ${PARALLEL_MATER_FLUID_CUDA_SOURCE} ${PARALLEL_MATER_AVBD_CUDA_SOURCE}
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
target_sources(parallel-mater PRIVATE
    ${PARALLEL_MATER_FLUID_CUDA_SOURCE}
    ${PARALLEL_MATER_AVBD_CUDA_SOURCE})
add_library(ParallelMater::parallel_mater ALIAS parallel-mater)
add_library(ParallelMater::api ALIAS parallel-mater)
list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater)
set_target_properties(
    parallel-mater
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_RUNTIME_LIBRARY Shared
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        EXPORT_NAME parallel_mater
        POSITION_INDEPENDENT_CODE ON)
set_property(TARGET parallel-mater PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_20)
target_include_directories(
    parallel-mater
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${PARALLEL_MATER_SLANG_GENERATED_DIR})
target_link_libraries(parallel-mater PUBLIC CUDA::cudart)
if(MSVC)
    target_compile_options(
        parallel-mater
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
else()
    target_compile_options(
        parallel-mater
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    add_library(parallel-mater-gallery-scene STATIC
                examples/support/scene.cpp
                examples/support/brick_scene.cpp
                examples/support/device_profiles.cpp
                examples/support/rope_render.cpp
                examples/support/surface_query.cpp)
    target_compile_features(parallel-mater-gallery-scene PUBLIC cxx_std_20)
    target_include_directories(
        parallel-mater-gallery-scene
        PUBLIC examples/support/include
        PRIVATE ${parallel_mater_cgltf_SOURCE_DIR})
    target_link_libraries(
        parallel-mater-gallery-scene PUBLIC ParallelMater::parallel_mater)
    target_compile_options(
        parallel-mater-gallery-scene PRIVATE -Wall -Wextra -Wpedantic)

    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT)
        set(PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT unknown)
    endif()
    add_executable(parallel-mater-conformance
                   conformance/v1/src/cuda_runner.cpp)
    target_link_libraries(parallel-mater-conformance PRIVATE
        parallel-mater-conformance-registry
        parallel-mater-gallery-scene
        CUDA::cudart)
    target_compile_definitions(parallel-mater-conformance PRIVATE
        PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
    target_compile_options(parallel-mater-conformance
                           PRIVATE -Wall -Wextra -Wpedantic)

    if(PARALLEL_MATER_BUILD_BENCHMARKS)
        add_executable(parallel-mater-rigid-scene-benchmark
                       benchmarks/rigid_scene.cpp)
        target_link_libraries(
            parallel-mater-rigid-scene-benchmark
            PRIVATE parallel-mater-gallery-scene)
        target_compile_options(
            parallel-mater-rigid-scene-benchmark
            PRIVATE -Wall -Wextra -Wpedantic)

        add_executable(parallel-mater-rigid-capture-benchmark
                       benchmarks/rigid_capture.cpp)
        target_link_libraries(parallel-mater-rigid-capture-benchmark
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_options(parallel-mater-rigid-capture-benchmark
                               PRIVATE -Wall -Wextra -Wpedantic)

        add_executable(parallel-mater-dump-benchmark
                       benchmarks/dump_scene.cpp)
        target_link_libraries(
            parallel-mater-dump-benchmark
            PRIVATE parallel-mater-gallery-scene)
        target_compile_options(
            parallel-mater-dump-benchmark
            PRIVATE -Wall -Wextra -Wpedantic)

        add_executable(parallel-mater-fluid-rigid-benchmark
                       benchmarks/fluid_rigid_scene.cpp)
        target_link_libraries(
            parallel-mater-fluid-rigid-benchmark
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(
            parallel-mater-fluid-rigid-benchmark PRIVATE
            PARALLEL_MATER_FLUID_RIGID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/FluidRigid.glb")
        target_compile_options(
            parallel-mater-fluid-rigid-benchmark
            PRIVATE -Wall -Wextra -Wpedantic)

        add_executable(parallel-mater-pegs-benchmark
                       benchmarks/pegs_scene.cpp)
        target_link_libraries(
            parallel-mater-pegs-benchmark PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(
            parallel-mater-pegs-benchmark PRIVATE
            PARALLEL_MATER_PEGS_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Pegs.glb")
    endif()

    FetchContent_Declare(
        parallel_mater_optix_dev
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG f1f6dd803f3159992d248178f6e09421c6eb8b6d
        GIT_SHALLOW TRUE)
    FetchContent_GetProperties(parallel_mater_optix_dev)
    if(NOT parallel_mater_optix_dev_POPULATED)
        FetchContent_Populate(parallel_mater_optix_dev)
    endif()

    find_package(OpenGL REQUIRED)
    find_package(glfw3 3.3 REQUIRED)

    set(PARALLEL_MATER_OPTIX_PTX
        ${CMAKE_CURRENT_BINARY_DIR}/generated/parallel_mater_optix.ptx)
    add_custom_command(
        OUTPUT ${PARALLEL_MATER_OPTIX_PTX}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND
            ${CMAKE_CUDA_COMPILER}
            --ptx
            --std=c++17
            --gpu-architecture=compute_75
            --use_fast_math
            --generate-line-info
            -I${parallel_mater_optix_dev_SOURCE_DIR}/include
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
            ${parallel_mater_optix_dev_SOURCE_DIR}/include)
    target_link_libraries(
        parallel-mater-gallery-renderer
        PUBLIC parallel-mater-gallery-scene
        PRIVATE CUDA::cuda_driver CUDA::cudart ${CMAKE_DL_LIBS})
    target_compile_options(
        parallel-mater-gallery-renderer PRIVATE -Wall -Wextra -Wpedantic)

    if(PARALLEL_MATER_BUILD_BENCHMARKS)
        add_executable(parallel-mater-dump-truck-benchmark benchmarks/dump_truck.cpp)
        target_link_libraries(parallel-mater-dump-truck-benchmark PRIVATE parallel-mater-gallery-renderer)
        target_compile_definitions(parallel-mater-dump-truck-benchmark PRIVATE
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb"
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
        target_compile_options(parallel-mater-dump-truck-benchmark PRIVATE -Wall -Wextra -Wpedantic)
        add_executable(parallel-mater-rigid-frame-benchmark benchmarks/rigid_frame.cpp)
        target_link_libraries(parallel-mater-rigid-frame-benchmark
                              PRIVATE parallel-mater-gallery-renderer)
        target_compile_definitions(parallel-mater-rigid-frame-benchmark PRIVATE
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
        target_compile_options(parallel-mater-rigid-frame-benchmark
                               PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(parallel-mater-gallery examples/gallery/main.cpp)
    if(WIN32)
        target_sources(parallel-mater-gallery PRIVATE
                       "${PARALLEL_MATER_GALLERY_ICON_RESOURCE}")
    endif()
    set(PARALLEL_MATER_GALLERY_PROFILE_CATALOG
        "${CMAKE_CURRENT_BINARY_DIR}/config/device-profiles.json")
    configure_file(config/device-profiles.json
                   "${PARALLEL_MATER_GALLERY_PROFILE_CATALOG}" COPYONLY)
    target_link_libraries(
        parallel-mater-gallery
        PRIVATE
            parallel-mater-gallery-renderer
            glfw
            OpenGL::GL)
    target_compile_definitions(
        parallel-mater-gallery
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
            PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeRope.glb"
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
    add_custom_command(
        TARGET parallel-mater-gallery POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:parallel-mater-gallery>/config"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
                "$<TARGET_FILE_DIR:parallel-mater-gallery>/config/device-profiles.json"
        VERBATIM)
endif()

if(BUILD_TESTING)
    add_executable(parallel-mater-soft-body-geometry-tests tests/soft_body_geometry_tests.cpp)
    target_link_libraries(parallel-mater-soft-body-geometry-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-soft-body-geometry-tests COMMAND parallel-mater-soft-body-geometry-tests)
    add_executable(parallel-mater-fluid-soft-body-tests tests/fluid_soft_body_tests.cpp)
    target_link_libraries(parallel-mater-fluid-soft-body-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-fluid-soft-body-tests COMMAND parallel-mater-fluid-soft-body-tests)
    set_tests_properties(parallel-mater-fluid-soft-body-tests PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(parallel-mater-fluid-rope-tests tests/fluid_rope_tests.cpp)
    target_link_libraries(parallel-mater-fluid-rope-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-fluid-rope-tests COMMAND parallel-mater-fluid-rope-tests)
    set_tests_properties(parallel-mater-fluid-rope-tests PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(parallel-mater-api-contract-tests tests/api_contract.cpp)
    target_link_libraries(
        parallel-mater-api-contract-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-api-contract-tests
             COMMAND parallel-mater-api-contract-tests)

    add_executable(parallel-mater-cuda-parity-capture tests/parity/cuda_capture.cpp)
    target_link_libraries(parallel-mater-cuda-parity-capture PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-cuda-parity-capture-self-test
             COMMAND parallel-mater-cuda-parity-capture --self-test)
    set_tests_properties(parallel-mater-cuda-parity-capture-self-test
                         PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

    add_executable(parallel-mater-rigid-tests tests/rigid_tests.cpp)
    target_link_libraries(
        parallel-mater-rigid-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-rigid-tests COMMAND parallel-mater-rigid-tests)
    set_tests_properties(parallel-mater-rigid-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-avbd-rigid-tests tests/avbd_rigid_tests.cpp)
    target_link_libraries(parallel-mater-avbd-rigid-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-avbd-rigid-tests COMMAND parallel-mater-avbd-rigid-tests)
    set_tests_properties(parallel-mater-avbd-rigid-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

    add_executable(parallel-mater-hit-box-tests tests/hit_box_tests.cpp)
    target_link_libraries(
        parallel-mater-hit-box-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-hit-box-tests COMMAND parallel-mater-hit-box-tests)
    set_tests_properties(parallel-mater-hit-box-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-fluid-tests tests/fluid_tests.cpp)
    target_link_libraries(
        parallel-mater-fluid-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-fluid-tests COMMAND parallel-mater-fluid-tests)
    set_tests_properties(parallel-mater-fluid-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-fluid-source-tests tests/fluid_source_tests.cpp)
    target_link_libraries(parallel-mater-fluid-source-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-fluid-source-tests COMMAND parallel-mater-fluid-source-tests)
    set_tests_properties(parallel-mater-fluid-source-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-rope-api-tests tests/rope_api_tests.cpp)
    target_link_libraries(parallel-mater-rope-api-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-rope-api-tests COMMAND parallel-mater-rope-api-tests)
    set_tests_properties(parallel-mater-rope-api-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-paint-geometry-tests
                   tests/paint_geometry_tests.cpp)
    target_link_libraries(parallel-mater-paint-geometry-tests PRIVATE
        ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-paint-geometry-tests
             COMMAND parallel-mater-paint-geometry-tests)

    if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
        add_test(NAME parallel-mater-conformance-inputs
                 COMMAND parallel-mater-conformance --check-inputs)
        add_test(NAME parallel-mater-conformance-cuda-goldens
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/reproduce.py
                    --runner $<TARGET_FILE:parallel-mater-conformance>
                    --comparator ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/compare.py
                    --cases ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/cases
                    --golden ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/golden/cuda)
        set_tests_properties(parallel-mater-conformance-cuda-goldens
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 RUN_SERIAL TRUE)

        find_program(PARALLEL_MATER_BLENDER_EXECUTABLE NAMES blender)
        if(PARALLEL_MATER_BLENDER_EXECUTABLE)
            add_executable(parallel-mater-blender-scene-check
                           tests/gallery/blender_scene_check.cpp)
            target_link_libraries(parallel-mater-blender-scene-check
                                 PRIVATE parallel-mater-gallery-scene)
            add_test(NAME parallel-mater-blender-export-tests
                COMMAND ${PARALLEL_MATER_BLENDER_EXECUTABLE}
                        --background --factory-startup --threads 1
                        --python-exit-code 1
                        --python ${CMAKE_CURRENT_SOURCE_DIR}/tests/blender/export_scene_tests.py
                        -- --loader $<TARGET_FILE:parallel-mater-blender-scene-check>)
            set_tests_properties(parallel-mater-blender-export-tests
                                 PROPERTIES TIMEOUT 300)
        endif()
        add_executable(parallel-mater-camera-controller-tests
                       tests/gallery/camera_controller_tests.cpp)
        target_link_libraries(
            parallel-mater-camera-controller-tests
            PRIVATE parallel-mater-gallery-scene)
        add_test(NAME parallel-mater-camera-controller-tests
                 COMMAND parallel-mater-camera-controller-tests)

        add_executable(parallel-mater-gallery-scene-tests
                       tests/gallery/scene_tests.cpp)
        target_link_libraries(
            parallel-mater-gallery-scene-tests
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(
            parallel-mater-gallery-scene-tests
            PRIVATE
                PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb"
                PARALLEL_MATER_PASSIVE_ACTIVE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/PassiveActive.glb")
        target_compile_definitions(
            parallel-mater-gallery-scene-tests
            PRIVATE
                PARALLEL_MATER_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Fluid.glb")
        target_compile_definitions(
            parallel-mater-gallery-scene-tests
            PRIVATE
                PARALLEL_MATER_FLUID_RIGID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/FluidRigid.glb")
        target_compile_definitions(
            parallel-mater-gallery-scene-tests
            PRIVATE
                PARALLEL_MATER_PEGS_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Pegs.glb")
        add_test(NAME parallel-mater-gallery-scene-tests
                 COMMAND parallel-mater-gallery-scene-tests)
        set_tests_properties(parallel-mater-gallery-scene-tests
                             PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(parallel-mater-device-profile-tests
                       tests/gallery/device_profiles_tests.cpp)
        target_link_libraries(parallel-mater-device-profile-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-device-profile-tests PRIVATE
            PARALLEL_MATER_DEVICE_PROFILES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
            PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb")
        add_test(NAME parallel-mater-device-profile-tests
                 COMMAND parallel-mater-device-profile-tests)

        add_executable(parallel-mater-rigid-wall-tests
                       tests/gallery/rigid_wall_tests.cpp)
        target_link_libraries(parallel-mater-rigid-wall-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-rigid-wall-tests PRIVATE
            PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb")
        add_test(NAME parallel-mater-rigid-wall-tests
                 COMMAND parallel-mater-rigid-wall-tests)
        set_tests_properties(parallel-mater-rigid-wall-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

        add_executable(parallel-mater-physics-frame-budget-tests
                       tests/gallery/physics_frame_budget_tests.cpp)
        target_include_directories(parallel-mater-physics-frame-budget-tests
                                   PRIVATE examples/support/include)
        add_test(NAME parallel-mater-physics-frame-budget-tests
                 COMMAND parallel-mater-physics-frame-budget-tests)

        add_executable(parallel-mater-fixed-collector-tests
                       tests/gallery/fixed_collector_tests.cpp)
        target_link_libraries(parallel-mater-fixed-collector-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-fixed-collector-tests PRIVATE
            PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintFixed.glb")
        add_test(NAME parallel-mater-fixed-collector-tests
                 COMMAND parallel-mater-fixed-collector-tests)
        set_tests_properties(parallel-mater-fixed-collector-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

        add_executable(parallel-mater-rigid-constraint-scene-tests
                       tests/gallery/rigid_constraint_scene_tests.cpp)
        add_executable(parallel-mater-celestial-scene-tests
                       tests/gallery/celestial_scene_tests.cpp)
        add_executable(parallel-mater-dump-truck-tests tests/gallery/dump_truck_tests.cpp)
        target_link_libraries(parallel-mater-dump-truck-tests PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-dump-truck-tests PRIVATE
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb")
        add_test(NAME parallel-mater-dump-truck-tests COMMAND parallel-mater-dump-truck-tests)
        add_test(NAME parallel-mater-dump-payload-tests
                 COMMAND parallel-mater-dump-truck-tests --metadata-only)
        set_tests_properties(parallel-mater-dump-truck-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        target_link_libraries(parallel-mater-celestial-scene-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-celestial-scene-tests PRIVATE
            PARALLEL_MATER_CELESTIAL_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Celestial.glb")
        add_test(NAME parallel-mater-celestial-scene-tests
                 COMMAND parallel-mater-celestial-scene-tests)
        set_tests_properties(parallel-mater-celestial-scene-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        target_link_libraries(parallel-mater-rigid-constraint-scene-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(
            parallel-mater-rigid-constraint-scene-tests PRIVATE
            PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintFixed.glb"
            PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPoint.glb"
            PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintHinge.glb"
            PARALLEL_MATER_CONSTRAINT_SLIDER_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintSlider.glb"
            PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPiston.glb"
            PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGeneric.glb"
            PARALLEL_MATER_CONSTRAINT_GENERIC_SPRING_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGenericSpring.glb"
            PARALLEL_MATER_CONSTRAINT_MOTOR_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintMotor.glb"
            PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintMotorSpring.glb")
        add_test(NAME parallel-mater-rigid-constraint-scene-tests
                 COMMAND parallel-mater-rigid-constraint-scene-tests)
        set_tests_properties(parallel-mater-rigid-constraint-scene-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        add_test(NAME parallel-mater-hinge-slider-contact-tests
                 COMMAND parallel-mater-rigid-constraint-scene-tests --hinge-four-substeps)
        set_tests_properties(parallel-mater-hinge-slider-contact-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

        add_executable(parallel-mater-piston-scene-tests
                       tests/gallery/piston_scene_tests.cpp)
        target_link_libraries(parallel-mater-piston-scene-tests
                              PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-piston-scene-tests PRIVATE
            PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPiston.glb")
        add_test(NAME parallel-mater-piston-scene-tests
                 COMMAND parallel-mater-piston-scene-tests)
        set_tests_properties(parallel-mater-piston-scene-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

        add_executable(parallel-mater-arrow-force-tests tests/gallery/arrow_force_tests.cpp)
        target_link_libraries(parallel-mater-arrow-force-tests PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-arrow-force-tests PRIVATE
            PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGeneric.glb")
        add_test(NAME parallel-mater-arrow-force-tests COMMAND parallel-mater-arrow-force-tests)
        set_tests_properties(parallel-mater-arrow-force-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

        add_executable(parallel-mater-generic-contact-tests tests/gallery/generic_contact_tests.cpp)
        target_link_libraries(parallel-mater-generic-contact-tests PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-generic-contact-tests PRIVATE
            PARALLEL_MATER_GENERIC_TRAY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/assets/generic_tray.glb")
        add_test(NAME parallel-mater-generic-contact-tests COMMAND parallel-mater-generic-contact-tests)
        set_tests_properties(parallel-mater-generic-contact-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

        add_executable(parallel-mater-cloth-tests tests/gallery/cloth_tests.cpp)
        target_link_libraries(parallel-mater-cloth-tests
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-cloth-tests PRIVATE
            PARALLEL_MATER_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Cloth.glb")
        add_test(NAME parallel-mater-cloth-tests COMMAND parallel-mater-cloth-tests)
        set_tests_properties(parallel-mater-cloth-tests
            PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(parallel-mater-cloth-variants-tests
            tests/gallery/cloth_variants_tests.cpp)
        target_link_libraries(parallel-mater-cloth-variants-tests
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-cloth-variants-tests PRIVATE
            PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ClothTear.glb"
            PARALLEL_MATER_CLOTH_PAINT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ClothPaint.glb")
        add_test(NAME parallel-mater-cloth-variants-tests
            COMMAND parallel-mater-cloth-variants-tests)
        set_tests_properties(parallel-mater-cloth-variants-tests
            PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(parallel-mater-cloth-water-tests
            tests/gallery/cloth_water_tests.cpp)
        target_link_libraries(parallel-mater-cloth-water-tests
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-cloth-water-tests PRIVATE
            PARALLEL_MATER_CLOTH_WATER_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ClothWater.glb")
        add_test(NAME parallel-mater-cloth-water-tests
            COMMAND parallel-mater-cloth-water-tests)
        set_tests_properties(parallel-mater-cloth-water-tests
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

        add_executable(parallel-mater-soft-body-tests
            tests/gallery/soft_body_tests.cpp)
        target_link_libraries(parallel-mater-soft-body-tests
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-soft-body-tests PRIVATE
            PARALLEL_MATER_SOFT_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Softbody.glb")
        add_test(NAME parallel-mater-soft-body-tests
            COMMAND parallel-mater-soft-body-tests)
        set_tests_properties(parallel-mater-soft-body-tests
            PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(parallel-mater-soft-body-rigid-tests
            tests/gallery/soft_body_rigid_tests.cpp)
        target_link_libraries(parallel-mater-soft-body-rigid-tests
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-soft-body-rigid-tests PRIVATE
            PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SoftbodyRigidBody.glb")
        add_test(NAME parallel-mater-soft-body-rigid-tests
            COMMAND parallel-mater-soft-body-rigid-tests)
        set_tests_properties(parallel-mater-soft-body-rigid-tests
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        add_test(NAME parallel-mater-soft-body-contact-stress
            COMMAND parallel-mater-soft-body-rigid-tests --contact-stress)
        add_test(NAME parallel-mater-soft-body-shape-contact-stress
            COMMAND parallel-mater-soft-body-rigid-tests --shape-contact-stress)
        set_tests_properties(parallel-mater-soft-body-contact-stress
            parallel-mater-soft-body-shape-contact-stress
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)

        add_executable(parallel-mater-soft-body-cloth-tests
                       tests/gallery/soft_body_cloth_tests.cpp)
        target_link_libraries(parallel-mater-soft-body-cloth-tests PRIVATE
            parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-soft-body-cloth-tests PRIVATE
            PARALLEL_MATER_SOFT_BODY_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SoftbodyCloth.glb")
        add_test(NAME parallel-mater-soft-body-cloth-tests
                 COMMAND parallel-mater-soft-body-cloth-tests)
        set_tests_properties(parallel-mater-soft-body-cloth-tests PROPERTIES
                             SKIP_RETURN_CODE 77 TIMEOUT 180)

        add_executable(parallel-mater-cloth-debug-tests
                       tests/gallery/cloth_debug_tests.cpp)
        target_link_libraries(parallel-mater-cloth-debug-tests PRIVATE
            parallel-mater-gallery-renderer CUDA::cudart)
        add_test(NAME parallel-mater-cloth-debug-tests
                 COMMAND parallel-mater-cloth-debug-tests)
        set_tests_properties(parallel-mater-cloth-debug-tests PROPERTIES
                             SKIP_RETURN_CODE 77)

        add_executable(parallel-mater-smoke-debug-tests
                       tests/gallery/smoke_debug_tests.cpp)
        target_link_libraries(parallel-mater-smoke-debug-tests PRIVATE
            parallel-mater-gallery-renderer CUDA::cudart)
        add_test(NAME parallel-mater-smoke-debug-tests
                 COMMAND parallel-mater-smoke-debug-tests)
        set_tests_properties(parallel-mater-smoke-debug-tests PROPERTIES
                             SKIP_RETURN_CODE 77)

        add_executable(parallel-mater-cloth-fracture-tests
                       tests/gallery/cloth_fracture_tests.cpp)
        target_link_libraries(parallel-mater-cloth-fracture-tests PRIVATE
            parallel-mater CUDA::cudart)
        add_test(NAME parallel-mater-cloth-fracture-tests
                 COMMAND parallel-mater-cloth-fracture-tests)
        set_tests_properties(parallel-mater-cloth-fracture-tests PROPERTIES
                             SKIP_RETURN_CODE 77 TIMEOUT 60)

        add_executable(parallel-mater-soft-body-fluid-tests
                       tests/gallery/soft_body_fluid_tests.cpp)
        target_link_libraries(parallel-mater-soft-body-fluid-tests PRIVATE
            parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-soft-body-fluid-tests PRIVATE
            PARALLEL_MATER_SOFT_BODY_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SoftbodyFluid.glb")
        add_test(NAME parallel-mater-soft-body-fluid-tests
                 COMMAND parallel-mater-soft-body-fluid-tests)
        set_tests_properties(parallel-mater-soft-body-fluid-tests PROPERTIES
                             SKIP_RETURN_CODE 77 TIMEOUT 180)

        add_executable(parallel-mater-rope-tests tests/gallery/rope_tests.cpp)
        target_link_libraries(parallel-mater-rope-tests PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-rope-tests PRIVATE
            PARALLEL_MATER_ROPE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Rope.glb"
            PARALLEL_MATER_ROPE_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RopeFluid.glb")
        add_test(NAME parallel-mater-rope-tests COMMAND parallel-mater-rope-tests)
        set_tests_properties(parallel-mater-rope-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_executable(parallel-mater-rope-soft-body-tests tests/gallery/rope_soft_body_tests.cpp)
        target_link_libraries(parallel-mater-rope-soft-body-tests PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(parallel-mater-rope-soft-body-tests PRIVATE
            PARALLEL_MATER_ROPE_SOFT_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RopeSoftbody.glb")
        add_test(NAME parallel-mater-rope-soft-body-tests COMMAND parallel-mater-rope-soft-body-tests)
        set_tests_properties(parallel-mater-rope-soft-body-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_executable(parallel-mater-rope-cloth-tests tests/gallery/rope_cloth_tests.cpp)
        target_link_libraries(parallel-mater-rope-cloth-tests PRIVATE parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-rope-cloth-tests PRIVATE
            PARALLEL_MATER_ROPE_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RopeCloth.glb")
        add_test(NAME parallel-mater-rope-cloth-tests COMMAND parallel-mater-rope-cloth-tests)
        set_tests_properties(parallel-mater-rope-cloth-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_executable(parallel-mater-smoke-tests tests/gallery/smoke_tests.cpp)
        target_link_libraries(parallel-mater-smoke-tests PRIVATE parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-smoke-tests PRIVATE
            PARALLEL_MATER_SMOKE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Smoke.glb")
        add_test(NAME parallel-mater-smoke-tests COMMAND parallel-mater-smoke-tests)
        set_tests_properties(parallel-mater-smoke-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_executable(parallel-mater-smoke-grid-tests tests/gallery/smoke_grid_tests.cpp)
        target_link_libraries(parallel-mater-smoke-grid-tests PRIVATE parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-smoke-grid-tests PRIVATE
            PARALLEL_MATER_SMOKE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Smoke.glb"
            PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeSoftbody.glb")
        add_test(NAME parallel-mater-smoke-grid-tests COMMAND parallel-mater-smoke-grid-tests)
        set_tests_properties(parallel-mater-smoke-grid-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
        add_executable(parallel-mater-smoke-water-tests tests/gallery/smoke_water_tests.cpp)
        target_link_libraries(parallel-mater-smoke-water-tests PRIVATE parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-smoke-water-tests PRIVATE
            PARALLEL_MATER_SMOKE_WATER_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeWater.glb")
        add_test(NAME parallel-mater-smoke-water-tests COMMAND parallel-mater-smoke-water-tests)
        set_tests_properties(parallel-mater-smoke-water-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_executable(parallel-mater-smoke-soft-body-tests tests/gallery/smoke_soft_body_tests.cpp)
        target_link_libraries(parallel-mater-smoke-soft-body-tests PRIVATE parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-smoke-soft-body-tests PRIVATE
            PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeSoftbody.glb")
        add_test(NAME parallel-mater-smoke-soft-body-tests COMMAND parallel-mater-smoke-soft-body-tests)
        set_tests_properties(parallel-mater-smoke-soft-body-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_executable(parallel-mater-smoke-cloth-tests tests/gallery/smoke_cloth_tests.cpp)
        target_link_libraries(parallel-mater-smoke-cloth-tests PRIVATE parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-smoke-cloth-tests PRIVATE
            PARALLEL_MATER_SMOKE_CLOTH_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeCloth.glb")
        add_test(NAME parallel-mater-smoke-cloth-tests COMMAND parallel-mater-smoke-cloth-tests)
        set_tests_properties(parallel-mater-smoke-cloth-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_executable(parallel-mater-smoke-rope-tests tests/gallery/smoke_rope_tests.cpp)
        target_link_libraries(parallel-mater-smoke-rope-tests PRIVATE parallel-mater-gallery-scene CUDA::cudart)
        target_compile_definitions(parallel-mater-smoke-rope-tests PRIVATE
            PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/SmokeRope.glb")
        add_test(NAME parallel-mater-smoke-rope-tests COMMAND parallel-mater-smoke-rope-tests)
        set_tests_properties(parallel-mater-smoke-rope-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_test(NAME parallel-mater-rope-soft-body-winding-tests
                 COMMAND parallel-mater-rope-soft-body-tests 320 --wind)
        set_tests_properties(parallel-mater-rope-soft-body-winding-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_test(NAME parallel-mater-rope-soft-body-release-tests
                 COMMAND parallel-mater-rope-soft-body-tests 1000 --wind-release)
        set_tests_properties(parallel-mater-rope-soft-body-release-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        if(TARGET parallel-mater-dump-benchmark)
            add_test(NAME parallel-mater-dump-small-contact-tests
                     COMMAND parallel-mater-dump-benchmark 100 1200)
            set_tests_properties(parallel-mater-dump-small-contact-tests
                                 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
            add_test(NAME parallel-mater-dump-fused-limit-tests
                     COMMAND parallel-mater-dump-benchmark 250 240)
            set_tests_properties(parallel-mater-dump-fused-limit-tests
                                 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        endif()
        if(TARGET parallel-mater-fluid-rigid-benchmark)
            add_test(NAME parallel-mater-fluid-rigid-small-contact-tests
                     COMMAND parallel-mater-fluid-rigid-benchmark 1000)
            set_tests_properties(parallel-mater-fluid-rigid-small-contact-tests
                                 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        endif()
        add_test(NAME parallel-mater-rope-soft-body-pull-tests
                 COMMAND parallel-mater-rope-soft-body-tests 180 --pull)
        set_tests_properties(parallel-mater-rope-soft-body-pull-tests
                             PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_test(NAME parallel-mater-rope-settle-tests
                 COMMAND parallel-mater-rope-tests 1200 --settle)
        add_test(NAME parallel-mater-rope-release-tests
                 COMMAND parallel-mater-rope-tests 1200 --settle-after-motion)
        add_test(NAME parallel-mater-rope-wrap-tests
                 COMMAND parallel-mater-rope-tests 600 --wrapped)
        add_test(NAME parallel-mater-rope-winding-tests
                 COMMAND parallel-mater-rope-tests 500 --winding)
        add_test(NAME parallel-mater-rope-fluid-tests
                 COMMAND parallel-mater-rope-tests 180 --fluid)
        add_test(NAME parallel-mater-rope-fluid-winding-tests
                 COMMAND parallel-mater-rope-tests 600 --fluid-winding)
        set_tests_properties(parallel-mater-rope-fluid-tests
                             parallel-mater-rope-fluid-winding-tests PROPERTIES
                             SKIP_RETURN_CODE 77 TIMEOUT 240)
        set_tests_properties(parallel-mater-rope-settle-tests
                             parallel-mater-rope-release-tests
                             parallel-mater-rope-wrap-tests
                             parallel-mater-rope-winding-tests PROPERTIES
                             SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_test(NAME parallel-mater-soft-body-fluid-dry-test
                 COMMAND parallel-mater-soft-body-fluid-tests 600 --dry)
        set_tests_properties(parallel-mater-soft-body-fluid-dry-test PROPERTIES
                             SKIP_RETURN_CODE 77 TIMEOUT 180)

        add_executable(parallel-mater-material-visibility-tests
                       tests/gallery/material_visibility_tests.cpp)
        target_link_libraries(parallel-mater-material-visibility-tests PRIVATE
            parallel-mater-gallery-renderer CUDA::cudart)
        target_compile_definitions(parallel-mater-material-visibility-tests PRIVATE
            PARALLEL_MATER_ALPHA_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/assets/material_alpha.gltf"
            PARALLEL_MATER_PEGS_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Pegs.glb"
            PARALLEL_MATER_CLOTH_WATER_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ClothWater.glb"
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
        add_test(NAME parallel-mater-material-visibility-tests
                 COMMAND parallel-mater-material-visibility-tests)
        set_tests_properties(parallel-mater-material-visibility-tests
                             PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(parallel-mater-foam-visuals-tests
                       tests/gallery/foam_visuals_tests.cpp
                       examples/support/foam_visuals.cpp)
        target_include_directories(parallel-mater-foam-visuals-tests PRIVATE
            examples/support examples/support/include)
        target_link_libraries(parallel-mater-foam-visuals-tests PRIVATE
            ParallelMater::parallel_mater)
        target_compile_features(parallel-mater-foam-visuals-tests PRIVATE cxx_std_20)
        add_test(NAME parallel-mater-foam-visuals-tests
                 COMMAND parallel-mater-foam-visuals-tests)

        if(PARALLEL_MATER_BUILD_BENCHMARKS)
            add_test(NAME parallel-mater-pegs-stability-test
                     COMMAND parallel-mater-pegs-benchmark --verify-settling)
            set_tests_properties(parallel-mater-pegs-stability-test
                                 PROPERTIES SKIP_RETURN_CODE 77)
            add_test(NAME parallel-mater-pegs-tilt-test
                     COMMAND parallel-mater-pegs-benchmark --verify-tilt)
            set_tests_properties(parallel-mater-pegs-tilt-test
                                 PROPERTIES SKIP_RETURN_CODE 77)
            add_test(NAME parallel-mater-pegs-slosh-test
                     COMMAND parallel-mater-pegs-benchmark --verify-slosh)
            set_tests_properties(parallel-mater-pegs-slosh-test
                                 PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        add_executable(parallel-mater-fluid-surface-tests
                       tests/gallery/fluid_surface_tests.cpp)
        target_include_directories(parallel-mater-fluid-surface-tests PRIVATE
            examples/support ${parallel_mater_optix_dev_SOURCE_DIR}/include)
        target_link_libraries(parallel-mater-fluid-surface-tests PRIVATE
            parallel-mater-gallery-renderer CUDA::cudart)
        target_compile_definitions(parallel-mater-fluid-surface-tests PRIVATE
            PARALLEL_MATER_FLUID_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Fluid.glb")
        add_test(NAME parallel-mater-fluid-surface-tests
                 COMMAND parallel-mater-fluid-surface-tests)
        set_tests_properties(parallel-mater-fluid-surface-tests
                             PROPERTIES SKIP_RETURN_CODE 77)
        function(add_gallery_headless_test name output frames)
            add_test(
                NAME ${name}
                COMMAND parallel-mater-gallery
                        ${ARGN}
                        --headless ${CMAKE_CURRENT_BINARY_DIR}/${output}
                        --frames ${frames})
        endfunction()

        add_gallery_headless_test(
            parallel-mater-optix-headless-test
            parallel-mater-optix-test.ppm 30)
        add_gallery_headless_test(
            parallel-mater-constraint-fixed-headless-test
            parallel-mater-constraint-fixed.ppm 120 --constraint-fixed
            --gravity-tilt-degrees 30)
        add_gallery_headless_test(
            parallel-mater-constraint-point-headless-test
            parallel-mater-constraint-point.ppm 180 --constraint-point)
        add_gallery_headless_test(
            parallel-mater-constraint-point-release-headless-test
            parallel-mater-constraint-point-release.ppm 180 --constraint-point
            --constraint-action-after-frames 60)
        add_gallery_headless_test(
            parallel-mater-constraint-hinge-headless-test
            parallel-mater-constraint-hinge.ppm 120 --constraint-hinge)
        add_gallery_headless_test(
            parallel-mater-constraint-piston-headless-test
            parallel-mater-constraint-piston.ppm 120 --constraint-piston)
        add_gallery_headless_test(
            parallel-mater-constraint-generic-headless-test
            parallel-mater-constraint-generic.ppm 120 --constraint-generic)
        add_gallery_headless_test(
            parallel-mater-constraint-motor-spring-headless-test
            parallel-mater-constraint-motor-spring.ppm 120
            --constraint-motor-spring --motor-forward --motor-right)
        add_gallery_headless_test(
            parallel-mater-rope-headless-test
            parallel-mater-rope-settled.ppm 900 --rope)
        add_gallery_headless_test(
            parallel-mater-rope-fluid-headless-test
            parallel-mater-rope-fluid.ppm 180 --rope-fluid)
        set_tests_properties(parallel-mater-rope-fluid-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "foam_patches=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-rope-soft-body-headless-test
            parallel-mater-rope-soft-body.ppm 180 --rope-soft-body)
        add_gallery_headless_test(
            parallel-mater-rope-cloth-headless-test
            parallel-mater-rope-cloth.ppm 300 --rope-cloth)
        add_gallery_headless_test(
            parallel-mater-smoke-headless-test
            parallel-mater-smoke.ppm 240 --smoke)
        add_gallery_headless_test(
            parallel-mater-smoke-soft-body-headless-test
            parallel-mater-smoke-soft-body.ppm 180 --smoke-softbody)
        set_tests_properties(parallel-mater-smoke-soft-body-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Smoke particles=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-smoke-cloth-headless-test
            parallel-mater-smoke-cloth.ppm 180 --smoke-cloth)
        set_tests_properties(parallel-mater-smoke-cloth-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Smoke particles=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-smoke-rope-headless-test
            parallel-mater-smoke-rope.ppm 180 --smoke-rope)
        set_tests_properties(parallel-mater-smoke-rope-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Smoke particles=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-smoke-rope-tilt-headless-test
            parallel-mater-smoke-rope-tilt.ppm 90 --smoke-rope
            --gravity-tilt-degrees 45)
        set_tests_properties(parallel-mater-smoke-rope-tilt-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Smoke particles=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-dump-headless-test
            parallel-mater-dump-test.ppm 150 --dump-spheres 10)
        add_gallery_headless_test(
            parallel-mater-dump-tip-headless-test
            parallel-mater-dump-tip.ppm 480 --dump-spheres 100 --dump-after-frames 120)
        add_gallery_headless_test(
            parallel-mater-dump-drive-headless-test
            parallel-mater-dump-drive.ppm 60 --dump-spheres 10 --motor-forward --motor-right)
        add_gallery_headless_test(
            parallel-mater-fluid-headless-test
            parallel-mater-fluid-test.ppm 30 --fluid)
        set_tests_properties(parallel-mater-fluid-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "foam_patches=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-fluid-rigid-headless-test
            parallel-mater-fluid-rigid-test.ppm 120 --fluid-rigid)
        set_tests_properties(parallel-mater-fluid-rigid-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "foam_patches=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-peg-paint-headless-test
            parallel-mater-peg-paint-test.ppm 180 --peg-paint)
        add_gallery_headless_test(
            parallel-mater-cloth-headless-test
            parallel-mater-cloth-test.ppm 180 --cloth)
        add_gallery_headless_test(
            parallel-mater-cloth-tear-headless-test
            parallel-mater-cloth-tear-test.ppm 180 --cloth-tear)
        add_gallery_headless_test(
            parallel-mater-cloth-tear-rolled-render-test
            parallel-mater-cloth-tear-rolled-test.ppm 420
            --cloth-tear --cloth-tilt-degrees 45
            --cloth-tilt-after-frames 120)
        add_gallery_headless_test(
            parallel-mater-cloth-paint-headless-test
            parallel-mater-cloth-paint-test.ppm 180 --cloth-paint)
        set_tests_properties(parallel-mater-cloth-paint-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Painted texels=[1-9][0-9]*")
        add_gallery_headless_test(
            parallel-mater-water-cloth-headless-test
            parallel-mater-water-cloth-test.ppm 180 --water-cloth)
        set_tests_properties(parallel-mater-water-cloth-headless-test PROPERTIES
            TIMEOUT 120)
        add_gallery_headless_test(
            parallel-mater-water-cloth-debug-headless-test
            parallel-mater-water-cloth-debug-test.ppm 180 --water-cloth-debug)
        set_tests_properties(
            parallel-mater-water-cloth-debug-headless-test PROPERTIES
            TIMEOUT 120)
        add_gallery_headless_test(
            parallel-mater-water-cloth-left-regression-test
            parallel-mater-water-cloth-left-test.ppm 180
            --water-cloth --cloth-tilt-left)
        set_tests_properties(
            parallel-mater-water-cloth-left-regression-test PROPERTIES
            TIMEOUT 120)
        add_gallery_headless_test(
            parallel-mater-soft-body-headless-test
            parallel-mater-soft-body-test.ppm 180 --soft-body)
        add_gallery_headless_test(
            parallel-mater-soft-body-tilt-headless-test
            parallel-mater-soft-body-tilt-test.ppm 180
            --soft-body --cloth-tilt-degrees 45)
        add_gallery_headless_test(
            parallel-mater-soft-body-rigid-headless-test
            parallel-mater-soft-body-rigid-test.ppm 180
            --soft-body-rigid --cloth-tilt-degrees 45)
        add_gallery_headless_test(
            parallel-mater-soft-body-cloth-headless-test
            parallel-mater-soft-body-cloth-test.ppm 180 --soft-body-cloth)
        add_gallery_headless_test(
            parallel-mater-soft-body-fluid-headless-test
            parallel-mater-soft-body-fluid-test.ppm 240 --soft-body-fluid --fluid-particles 4000)
        add_gallery_headless_test(
            parallel-mater-soft-body-cloth-tear-debug-test
            parallel-mater-soft-body-cloth-tear-debug.ppm 600 --soft-body-cloth
            --cloth-tilt-after-frames 240 --cloth-tilt-degrees 45 --cloth-debug)
        set_tests_properties(parallel-mater-soft-body-cloth-tear-debug-test PROPERTIES
                             PASS_REGULAR_EXPRESSION "Cloth debug surfaces=2")
        add_gallery_headless_test(
            parallel-mater-physics-capture-headless-test
            parallel-mater-physics-capture-test.ppm 10
            --physics-capture
            ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-physics-capture-test.log)
        set_tests_properties(
            parallel-mater-physics-capture-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Physics capture")
        add_gallery_headless_test(
            parallel-mater-peg-paint-containment-test
            parallel-mater-peg-paint-containment-test.ppm 600
            --peg-paint --fluid-particle-view --trace-fluid-escapes)
        add_gallery_headless_test(
            parallel-mater-peg-paint-cap-test
            parallel-mater-peg-paint-cap-test.ppm 180
            --peg-paint --fluid-particles 100)
        add_gallery_headless_test(
            parallel-mater-fluid-containment-test
            parallel-mater-fluid-containment-test.ppm 120
            --fluid --trace-fluid-escapes)
        add_gallery_headless_test(
            parallel-mater-fluid-cap-headless-test
            parallel-mater-fluid-cap-test.ppm 30 --fluid-particles 100)
        set_tests_properties(parallel-mater-fluid-cap-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Fluid particles=100 ")
        add_gallery_headless_test(
            parallel-mater-fluid-particle-view-test
            parallel-mater-fluid-particles-test.ppm 30 --fluid-particle-view)
    endif()
endif()

if(BUILD_TESTING AND NOT PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    add_library(parallel-mater-cuda-conformance-scene STATIC
        examples/support/scene.cpp
        examples/support/rope_render.cpp
        examples/support/surface_query.cpp)
    target_compile_features(parallel-mater-cuda-conformance-scene
                            PUBLIC cxx_std_20)
    target_include_directories(parallel-mater-cuda-conformance-scene
        PUBLIC examples/support/include
        PRIVATE ${parallel_mater_cgltf_SOURCE_DIR})
    target_link_libraries(parallel-mater-cuda-conformance-scene
                          PUBLIC ParallelMater::parallel_mater)

    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE PARALLEL_MATER_CUDA_CONFORMANCE_SOURCE_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT PARALLEL_MATER_CUDA_CONFORMANCE_SOURCE_COMMIT)
        set(PARALLEL_MATER_CUDA_CONFORMANCE_SOURCE_COMMIT unknown)
    endif()
    add_executable(parallel-mater-cuda-conformance
                   conformance/v1/src/cuda_runner.cpp)
    target_link_libraries(parallel-mater-cuda-conformance PRIVATE
        parallel-mater-conformance-registry
        parallel-mater-cuda-conformance-scene
        CUDA::cudart)
    target_compile_features(parallel-mater-cuda-conformance PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-cuda-conformance PRIVATE
        PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CUDA_CONFORMANCE_SOURCE_COMMIT}")
endif()
endif() # PARALLEL_MATER_BUILD_CUDA

if(PARALLEL_MATER_BUILD_D3D12)
    if(NOT WIN32)
        message(FATAL_ERROR "PARALLEL_MATER_BUILD_D3D12 requires Windows")
    endif()

    set(PARALLEL_MATER_FXC_HINTS
        "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/x64"
        "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin")
    file(GLOB PARALLEL_MATER_WINDOWS_SDK_BIN_DIRS
         LIST_DIRECTORIES TRUE
         "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/*")
    foreach(sdk_bin_dir IN LISTS PARALLEL_MATER_WINDOWS_SDK_BIN_DIRS)
        if(IS_DIRECTORY "${sdk_bin_dir}/x64")
            list(APPEND PARALLEL_MATER_FXC_HINTS "${sdk_bin_dir}/x64")
        endif()
    endforeach()
    list(REVERSE PARALLEL_MATER_FXC_HINTS)
    find_program(PARALLEL_MATER_FXC_EXECUTABLE
        NAMES fxc fxc.exe
        HINTS ${PARALLEL_MATER_FXC_HINTS}
        REQUIRED)
    set(PARALLEL_MATER_D3D12_GENERATED_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/generated/d3d12)
    set(PARALLEL_MATER_D3D12_SHADER_HEADERS)
    foreach(shader IN ITEMS integrate prepare generate generate_guided generate_warp color finalize clear compact_count compact_prefix compact_scatter contacts contacts_warp contact_prepare contact_events)
        set(shader_header
            ${PARALLEL_MATER_D3D12_GENERATED_DIR}/pm_d3d12_${shader}.h)
        set(shader_binary ${PARALLEL_MATER_D3D12_GENERATED_DIR}/pm_d3d12_${shader}.cso)
        set(shader_optimization /O3)
        set(shader_source ${CMAKE_CURRENT_SOURCE_DIR}/src/d3d12/rigid.hlsl)
        if(shader MATCHES "^compact_")
            set(shader_source ${CMAKE_CURRENT_SOURCE_DIR}/src/d3d12/compact.hlsl)
        elseif(shader MATCHES "^contacts")
            set(shader_source ${CMAKE_CURRENT_SOURCE_DIR}/src/d3d12/contacts.hlsl)
        elseif(shader STREQUAL "contact_prepare" OR shader STREQUAL "contact_events")
            set(shader_source ${CMAKE_CURRENT_SOURCE_DIR}/src/d3d12/contact_prepare.hlsl)
        endif()
        if(shader STREQUAL "generate" OR shader STREQUAL "prepare" OR
           shader STREQUAL "color" OR shader STREQUAL "finalize" OR
           shader STREQUAL "contact_prepare" OR shader MATCHES "^contacts")
            # FXC's higher optimization levels take prohibitively long on the
            # regular BVH kernel and serial preparation/contact solver. The
            # guided sweep is compiled separately; /Gis preserves solve math.
            set(shader_optimization /O1)
        endif()
        add_custom_command(
            OUTPUT ${shader_header}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    ${PARALLEL_MATER_D3D12_GENERATED_DIR}
            COMMAND ${PARALLEL_MATER_FXC_EXECUTABLE}
                    /nologo /T cs_5_1 /E rigid_${shader}
                    ${shader_optimization} /Gis /Ges /WX
                    /Fo ${shader_binary}
                    ${shader_source}
            COMMAND ${CMAKE_COMMAND}
                    -DINPUT=${shader_binary} -DOUTPUT=${shader_header}
                    -DSYMBOL=pm_d3d12_${shader}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShader.cmake
            BYPRODUCTS ${shader_binary}
            DEPENDS src/d3d12/rigid.hlsl ${shader_source} cmake/EmbedShader.cmake
            VERBATIM)
        list(APPEND PARALLEL_MATER_D3D12_SHADER_HEADERS ${shader_header})
    endforeach()
    set_source_files_properties(${PARALLEL_MATER_D3D12_SHADER_HEADERS}
                                PROPERTIES GENERATED TRUE)

    add_library(parallel-mater-d3d12 STATIC
        src/d3d12/world.cpp
        src/d3d12/stubs.cpp
        src/fluid_geometry.cpp
        src/fluid_source.cpp
        src/soft_body_geometry.cpp
        src/rope_geometry.cpp
        ${PARALLEL_MATER_D3D12_SHADER_HEADERS})
    add_library(ParallelMater::d3d12 ALIAS parallel-mater-d3d12)
    target_compile_features(parallel-mater-d3d12 PUBLIC cxx_std_20)
    target_compile_definitions(parallel-mater-d3d12
        PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX PARALLEL_MATER_D3D12_GEOMETRY=1)
    target_include_directories(parallel-mater-d3d12
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE ${PARALLEL_MATER_D3D12_GENERATED_DIR})
    target_link_libraries(parallel-mater-d3d12
                          PUBLIC d3d12 dxgi dxguid)
    if(MSVC)
        target_compile_options(parallel-mater-d3d12 PRIVATE
                               /W4 /permissive- /EHsc /Zc:__cplusplus)
    endif()
    set_target_properties(parallel-mater-d3d12 PROPERTIES
        EXPORT_NAME d3d12
        OUTPUT_NAME parallel-mater-d3d12)
    list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater-d3d12)

    if(BUILD_TESTING)
        add_executable(parallel-mater-d3d12-api-contract-tests
                       tests/d3d12/api_contract.cpp)
        target_link_libraries(parallel-mater-d3d12-api-contract-tests
                              PRIVATE ParallelMater::d3d12)
        target_compile_features(parallel-mater-d3d12-api-contract-tests
                                PRIVATE cxx_std_20)
        add_test(NAME parallel-mater-d3d12-api-contract-tests
                 COMMAND parallel-mater-d3d12-api-contract-tests)

        add_executable(parallel-mater-d3d12-foundation-tests
                       tests/d3d12/foundation_tests.cpp)
        target_link_libraries(parallel-mater-d3d12-foundation-tests
                              PRIVATE ParallelMater::d3d12)
        target_compile_features(parallel-mater-d3d12-foundation-tests
                                PRIVATE cxx_std_20)
        add_test(NAME parallel-mater-d3d12-foundation-tests
                 COMMAND parallel-mater-d3d12-foundation-tests)
        set_tests_properties(parallel-mater-d3d12-foundation-tests
                             PROPERTIES LABELS "d3d12;rigid")
    endif()

    if(PARALLEL_MATER_BUILD_D3D12_GALLERY)
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
        set(PARALLEL_MATER_D3D12_GALLERY_SHADER_HEADERS)
        foreach(stage IN ITEMS vs ps ui_vs ui_ps)
            string(REGEX REPLACE "^ui_" "" shader_profile ${stage})
            set(gallery_shader_header
                ${PARALLEL_MATER_D3D12_GENERATED_DIR}/pm_d3d12_gallery_${stage}.h)
            add_custom_command(
                OUTPUT ${gallery_shader_header}
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        ${PARALLEL_MATER_D3D12_GENERATED_DIR}
                COMMAND ${PARALLEL_MATER_FXC_EXECUTABLE}
                        /nologo /T ${shader_profile}_5_1 /E gallery_${stage}
                        /O3 /Gis /Ges /WX
                        /Fh ${gallery_shader_header}
                        /Vn pm_d3d12_gallery_${stage}
                        ${CMAKE_CURRENT_SOURCE_DIR}/examples/d3d12_gallery/render.hlsl
                DEPENDS examples/d3d12_gallery/render.hlsl
                VERBATIM)
            list(APPEND PARALLEL_MATER_D3D12_GALLERY_SHADER_HEADERS
                 ${gallery_shader_header})
        endforeach()
        set_source_files_properties(
            ${PARALLEL_MATER_D3D12_GALLERY_SHADER_HEADERS}
            PROPERTIES GENERATED TRUE)

        add_library(parallel-mater-d3d12-gallery-scene STATIC
            examples/support/scene.cpp
            examples/support/rope_render.cpp)
        target_compile_features(parallel-mater-d3d12-gallery-scene
                                PUBLIC cxx_std_20)
        target_compile_definitions(parallel-mater-d3d12-gallery-scene
                                   PUBLIC PARALLEL_MATER_GALLERY_D3D12=1)
        target_include_directories(parallel-mater-d3d12-gallery-scene
            PUBLIC examples/support/include
            PRIVATE ${parallel_mater_cgltf_SOURCE_DIR})
        target_link_libraries(parallel-mater-d3d12-gallery-scene
                              PUBLIC ParallelMater::d3d12)
        if(MSVC)
            target_compile_options(parallel-mater-d3d12-gallery-scene
                                   PRIVATE /W4 /permissive- /EHsc)
        endif()

        add_executable(parallel-mater-d3d12-gallery
            examples/d3d12_gallery/main.cpp
            ${PARALLEL_MATER_D3D12_GALLERY_SHADER_HEADERS}
            "${PARALLEL_MATER_GALLERY_ICON_RESOURCE}")
        target_compile_features(parallel-mater-d3d12-gallery PRIVATE cxx_std_20)
        target_include_directories(parallel-mater-d3d12-gallery
                                   PRIVATE ${PARALLEL_MATER_D3D12_GENERATED_DIR})
        target_compile_definitions(parallel-mater-d3d12-gallery PRIVATE
            WIN32_LEAN_AND_MEAN NOMINMAX PARALLEL_MATER_GALLERY_D3D12=1
            PARALLEL_MATER_DEFAULT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb"
            PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintFixed.glb"
            PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPoint.glb"
            PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintHinge.glb"
            PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPiston.glb"
            PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGeneric.glb"
            PARALLEL_MATER_CONSTRAINT_GENERIC_SPRING_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGenericSpring.glb"
            PARALLEL_MATER_CONSTRAINT_MOTOR_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintMotor.glb"
            PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintMotorSpring.glb"
            PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/DumpTruck.glb")
        target_link_libraries(parallel-mater-d3d12-gallery PRIVATE
            parallel-mater-d3d12-gallery-scene glfw d3d12 dxgi dxguid)
        if(MSVC)
            target_compile_options(parallel-mater-d3d12-gallery
                                   PRIVATE /W4 /permissive- /EHsc)
        endif()

        add_executable(parallel-mater-d3d12-launcher WIN32
            examples/d3d12_gallery/launcher.cpp
            "${PARALLEL_MATER_GALLERY_ICON_RESOURCE}")
        target_compile_features(parallel-mater-d3d12-launcher PRIVATE cxx_std_20)
        target_link_libraries(parallel-mater-d3d12-launcher PRIVATE shell32 ole32 user32)
        set_target_properties(parallel-mater-d3d12-launcher PROPERTIES OUTPUT_NAME "Parallel-Mater")
        set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
        include(InstallRequiredSystemLibraries)
        set(PARALLEL_MATER_D3D12_RUNTIME_DIRECTORY "")
        if(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
            list(GET CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS 0 gallery_runtime)
            get_filename_component(PARALLEL_MATER_D3D12_RUNTIME_DIRECTORY "${gallery_runtime}" DIRECTORY)
            add_custom_command(TARGET parallel-mater-d3d12-launcher POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
                        $<TARGET_FILE_DIR:parallel-mater-d3d12-launcher>
                VERBATIM)
        endif()
        add_custom_target(install-d3d12-gallery
            COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/scripts/Install-WindowsGallery.ps1
                -BuildDirectory $<TARGET_FILE_DIR:parallel-mater-d3d12-gallery>
                -Version ${PROJECT_VERSION}
            DEPENDS parallel-mater-d3d12-gallery parallel-mater-d3d12-launcher
            COMMENT "Installing Parallel-Mater for the current user and adding its Start menu entry"
            VERBATIM)

        if(BUILD_TESTING)
            add_executable(parallel-mater-d3d12-gallery-controls tests/d3d12/gallery_controls.cpp)
            target_link_libraries(parallel-mater-d3d12-gallery-controls PRIVATE parallel-mater-d3d12-gallery-scene)
            add_test(NAME parallel-mater-d3d12-gallery-controls COMMAND parallel-mater-d3d12-gallery-controls)
            execute_process(
                COMMAND git rev-parse HEAD
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE PARALLEL_MATER_D3D12_SOURCE_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(NOT PARALLEL_MATER_D3D12_SOURCE_COMMIT)
                set(PARALLEL_MATER_D3D12_SOURCE_COMMIT unknown)
            endif()
            add_executable(parallel-mater-d3d12-conformance
                conformance/v1/src/d3d12_runner.cpp)
            target_link_libraries(parallel-mater-d3d12-conformance PRIVATE
                parallel-mater-conformance-registry
                parallel-mater-d3d12-gallery-scene d3d12 dxgi)
            target_compile_features(parallel-mater-d3d12-conformance
                                    PRIVATE cxx_std_20)
            target_compile_definitions(parallel-mater-d3d12-conformance PRIVATE
                PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
                PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_D3D12_SOURCE_COMMIT}")
            add_test(NAME parallel-mater-d3d12-conformance-inputs
                     COMMAND parallel-mater-d3d12-conformance --check-inputs)
            add_test(NAME parallel-mater-d3d12-conformance-cuda-goldens
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/reproduce_d3d12.py
                    --runner $<TARGET_FILE:parallel-mater-d3d12-conformance>
                    --comparator ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/compare.py
                    --cases ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/cases
                    --golden ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/golden/cuda
                    --repetitions 10)
            set_tests_properties(
                parallel-mater-d3d12-conformance-cuda-goldens
                PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 RUN_SERIAL TRUE
                           LABELS "d3d12;conformance;rigid")
            add_test(NAME parallel-mater-d3d12-rigid-gallery-validation
                COMMAND parallel-mater-d3d12-gallery
                    --warp --all-rigid-scenes --headless --validate
                    --repeat 10 --frames 1 --dump-spheres 32
                    --width 320 --height 240
                    --output ${CMAKE_CURRENT_BINARY_DIR}/d3d12-gallery-captures)
            set_tests_properties(parallel-mater-d3d12-rigid-gallery-validation
                PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE
                           LABELS "d3d12;gallery;rigid")
        endif()
    endif()
endif()

if(PARALLEL_MATER_BUILD_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "PARALLEL_MATER_BUILD_METAL requires Apple platforms")
    endif()
    enable_language(OBJCXX)
    find_library(PARALLEL_MATER_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(PARALLEL_MATER_METAL_FRAMEWORK Metal REQUIRED)
    find_program(PARALLEL_MATER_XCRUN_EXECUTABLE xcrun REQUIRED)
    find_program(PARALLEL_MATER_XXD_EXECUTABLE xxd REQUIRED)
    execute_process(
        COMMAND ${PARALLEL_MATER_XCRUN_EXECUTABLE} -f metal
        OUTPUT_VARIABLE PARALLEL_MATER_METAL_COMPILER
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    get_filename_component(PARALLEL_MATER_METAL_TOOLCHAIN_DIR
                           ${PARALLEL_MATER_METAL_COMPILER} DIRECTORY)
    set(PARALLEL_MATER_METALLIB_COMPILER
        ${PARALLEL_MATER_METAL_TOOLCHAIN_DIR}/metallib)
    if(NOT EXISTS ${PARALLEL_MATER_METALLIB_COMPILER})
        message(FATAL_ERROR "The Xcode Metal toolchain does not contain metallib")
    endif()

    set(PARALLEL_MATER_METAL_GENERATED_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/generated/metal)
    set(PARALLEL_MATER_METAL_AIR
        ${PARALLEL_MATER_METAL_GENERATED_DIR}/parallel_mater.air)
    set(PARALLEL_MATER_FLUID_MSL
        ${PARALLEL_MATER_METAL_GENERATED_DIR}/fluid.metal)
    set(PARALLEL_MATER_AVBD_MSL
        ${PARALLEL_MATER_METAL_GENERATED_DIR}/parallel_mater_avbd_runtime.metal)
    set(PARALLEL_MATER_FLUID_AIR
        ${PARALLEL_MATER_METAL_GENERATED_DIR}/fluid.air)
    set(PARALLEL_MATER_METALLIB
        ${PARALLEL_MATER_METAL_GENERATED_DIR}/parallel_mater.metallib)
    set(PARALLEL_MATER_METALLIB_INCLUDE
        ${PARALLEL_MATER_METAL_GENERATED_DIR}/parallel_mater_metallib.inc)

    add_custom_command(
        OUTPUT ${PARALLEL_MATER_METALLIB_INCLUDE}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${PARALLEL_MATER_METAL_GENERATED_DIR}
        COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/fluid_shared.slang
                -DPM_TARGET_METAL=1 -DPM_TARGET_CUDA=0
                -entry pm_fluid_cell_keys
                -entry pm_fluid_forces
                -entry pm_fluid_integrate
                -target metal
                -capability metallib_3_2
                -fp-mode precise -O2 -line-directive-mode none
                -o ${PARALLEL_MATER_FLUID_MSL}
        COMMAND ${PARALLEL_MATER_SLANGC_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/slang/avbd.slang
                -entry avbdConformanceMain -stage compute
                -target metal -capability metallib_3_2
                -fp-mode precise -O2 -line-directive-mode none
                -o ${PARALLEL_MATER_AVBD_MSL}
        COMMAND ${CMAKE_COMMAND} -E env
                CLANG_MODULE_CACHE_PATH=${PARALLEL_MATER_METAL_GENERATED_DIR}/module-cache
                ${PARALLEL_MATER_XCRUN_EXECUTABLE} -sdk macosx metal
                -std=metal4.0 -mmacosx-version-min=26.0
                -fno-fast-math -ffp-contract=fast
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                -I${PARALLEL_MATER_METAL_GENERATED_DIR}
                -fmodules-cache-path=${PARALLEL_MATER_METAL_GENERATED_DIR}/module-cache
                -c ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal
                -o ${PARALLEL_MATER_METAL_AIR}
        COMMAND ${CMAKE_COMMAND} -E env
                CLANG_MODULE_CACHE_PATH=${PARALLEL_MATER_METAL_GENERATED_DIR}/module-cache
                ${PARALLEL_MATER_XCRUN_EXECUTABLE} -sdk macosx metal
                -std=metal4.0 -mmacosx-version-min=26.0
                -fno-fast-math -ffp-contract=fast -Wno-unused-variable
                -fmodules-cache-path=${PARALLEL_MATER_METAL_GENERATED_DIR}/module-cache
                -c ${PARALLEL_MATER_FLUID_MSL}
                -o ${PARALLEL_MATER_FLUID_AIR}
        COMMAND ${PARALLEL_MATER_METALLIB_COMPILER}
                ${PARALLEL_MATER_METAL_AIR}
                ${PARALLEL_MATER_FLUID_AIR}
                -o ${PARALLEL_MATER_METALLIB}
        COMMAND ${PARALLEL_MATER_XXD_EXECUTABLE} -i
                -n parallel_mater_metallib_data
                ${PARALLEL_MATER_METALLIB}
                ${PARALLEL_MATER_METALLIB_INCLUDE}
        DEPENDS src/metal/kernels.metal src/metal/avbd.metal
                src/slang/fluid_shared.slang src/slang/avbd.slang
                include/parallel_mater/solver/avbd.hpp
                include/parallel_mater/solver/contact_friction.hpp
                ${PARALLEL_MATER_SLANGC_EXECUTABLE}
        BYPRODUCTS
                ${PARALLEL_MATER_FLUID_MSL}
                ${PARALLEL_MATER_AVBD_MSL}
                ${PARALLEL_MATER_METAL_AIR}
                ${PARALLEL_MATER_FLUID_AIR}
                ${PARALLEL_MATER_METALLIB}
        VERBATIM)
    set_source_files_properties(
        src/metal/world.mm
        PROPERTIES OBJECT_DEPENDS ${PARALLEL_MATER_METALLIB_INCLUDE})

    add_library(parallel-mater-metal STATIC
        src/metal/world.mm
        src/metal/systems.mm
        src/rope_geometry.cpp
        src/fluid_geometry.cpp
        src/fluid_source.cpp
        src/soft_body_geometry.cpp)
    add_library(ParallelMater::metal ALIAS parallel-mater-metal)
    set_target_properties(
        parallel-mater-metal
        PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED YES
            OBJCXX_STANDARD 20
            OBJCXX_STANDARD_REQUIRED YES
            OSX_DEPLOYMENT_TARGET 26.0
            EXPORT_NAME metal
            POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET parallel-mater-metal
                 PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_20)
    target_include_directories(
        parallel-mater-metal
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE ${PARALLEL_MATER_METAL_GENERATED_DIR})
    target_compile_options(
        parallel-mater-metal PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc;-ffp-contract=off;-Wall;-Wextra;-Wpedantic>)
    target_compile_definitions(
        parallel-mater-metal PRIVATE PARALLEL_MATER_METAL_GEOMETRY=1)
    target_link_libraries(
        parallel-mater-metal
        PUBLIC
            ${PARALLEL_MATER_FOUNDATION_FRAMEWORK}
            ${PARALLEL_MATER_METAL_FRAMEWORK})
    list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater-metal)

    if(BUILD_TESTING)
        add_executable(parallel-mater-metal-api-contract-tests
                       tests/metal/api_contract.cpp)
        target_link_libraries(parallel-mater-metal-api-contract-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-api-contract-tests
                 COMMAND parallel-mater-metal-api-contract-tests)

        add_executable(parallel-mater-metal-foundation-tests
                       tests/metal/foundation_tests.cpp)
        target_link_libraries(parallel-mater-metal-foundation-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-foundation-tests
                 COMMAND parallel-mater-metal-foundation-tests)

        add_executable(parallel-mater-metal-rigid-tests
                       tests/metal/rigid_tests.cpp)
        target_link_libraries(parallel-mater-metal-rigid-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-rigid-tests
                 COMMAND parallel-mater-metal-rigid-tests)

        add_executable(parallel-mater-metal-hit-box-tests
                       tests/metal/hit_box_tests.cpp)
        target_link_libraries(parallel-mater-metal-hit-box-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-hit-box-tests
                 COMMAND parallel-mater-metal-hit-box-tests)
        set_tests_properties(parallel-mater-metal-hit-box-tests
                             PROPERTIES LABELS "metal;rigid;hit-box")

        add_executable(parallel-mater-metal-constraint-tests
                       tests/metal/constraint_tests.cpp)
        target_link_libraries(parallel-mater-metal-constraint-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-constraint-tests
                 COMMAND parallel-mater-metal-constraint-tests)
        set_tests_properties(parallel-mater-metal-constraint-tests
                             PROPERTIES LABELS "metal;rigid;constraints")

        add_executable(parallel-mater-metal-particle-system-tests
                       tests/metal/particle_system_tests.mm)
        target_compile_options(parallel-mater-metal-particle-system-tests
                               PRIVATE -fobjc-arc)
        target_link_libraries(parallel-mater-metal-particle-system-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-particle-system-tests
                 COMMAND parallel-mater-metal-particle-system-tests)
        set_tests_properties(parallel-mater-metal-particle-system-tests
                             PROPERTIES LABELS "metal;couplings")

        add_executable(parallel-mater-metal-subsystem-tests
                       tests/metal/subsystem_tests.mm)
        target_compile_options(parallel-mater-metal-subsystem-tests
                               PRIVATE -fobjc-arc)
        target_link_libraries(parallel-mater-metal-subsystem-tests
                              PRIVATE ParallelMater::metal)
        foreach(PARALLEL_MATER_METAL_SUBSYSTEM
                fluid cloth soft-body rope smoke couplings paint debug)
            add_test(
                NAME parallel-mater-metal-${PARALLEL_MATER_METAL_SUBSYSTEM}-tests
                COMMAND parallel-mater-metal-subsystem-tests
                        ${PARALLEL_MATER_METAL_SUBSYSTEM})
            set_tests_properties(
                parallel-mater-metal-${PARALLEL_MATER_METAL_SUBSYSTEM}-tests
                PROPERTIES LABELS
                           "metal;subsystem;${PARALLEL_MATER_METAL_SUBSYSTEM}")
        endforeach()

        add_executable(parallel-mater-metal-buffer-upload-tests
                       tests/metal/buffer_upload_tests.mm)
        target_compile_options(parallel-mater-metal-buffer-upload-tests
                               PRIVATE -fobjc-arc)
        target_link_libraries(parallel-mater-metal-buffer-upload-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-buffer-upload-tests
                 COMMAND parallel-mater-metal-buffer-upload-tests)

        add_executable(parallel-mater-metal-primitive-tests
                       tests/metal/primitive_tests.mm)
        target_compile_options(parallel-mater-metal-primitive-tests
                               PRIVATE -fobjc-arc)
        target_compile_definitions(parallel-mater-metal-primitive-tests PRIVATE
            PARALLEL_MATER_TEST_METALLIB_PATH="${PARALLEL_MATER_METALLIB}")
        target_link_libraries(parallel-mater-metal-primitive-tests
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-primitive-tests
                 COMMAND parallel-mater-metal-primitive-tests)

        add_executable(parallel-mater-metal-parity-capture
                       tests/metal/parity_capture.mm)
        target_compile_options(parallel-mater-metal-parity-capture
                               PRIVATE -fobjc-arc)
        target_link_libraries(parallel-mater-metal-parity-capture
                              PRIVATE ParallelMater::metal)
        add_test(NAME parallel-mater-metal-parity-capture-self-test
                 COMMAND parallel-mater-metal-parity-capture --self-test)
        set_tests_properties(parallel-mater-metal-parity-capture-self-test
                             PROPERTIES LABELS "metal;parity;couplings"
                                        TIMEOUT 120)
    endif()

    if(PARALLEL_MATER_BUILD_METAL_GALLERY)
        find_package(glfw3 3.3 REQUIRED)
        find_library(PARALLEL_MATER_APPKIT_FRAMEWORK AppKit REQUIRED)
        find_library(PARALLEL_MATER_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

        set(PARALLEL_MATER_METAL_GALLERY_AIR
            ${PARALLEL_MATER_METAL_GENERATED_DIR}/parallel_mater_gallery.air)
        set(PARALLEL_MATER_METAL_GALLERY_METALLIB
            ${PARALLEL_MATER_METAL_GENERATED_DIR}/parallel_mater_gallery.metallib)
        set(PARALLEL_MATER_METAL_GALLERY_INCLUDE
            ${PARALLEL_MATER_METAL_GENERATED_DIR}/parallel_mater_gallery_metallib.inc)
        add_custom_command(
            OUTPUT ${PARALLEL_MATER_METAL_GALLERY_INCLUDE}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    ${PARALLEL_MATER_METAL_GENERATED_DIR}
            COMMAND ${CMAKE_COMMAND} -E env
                    CLANG_MODULE_CACHE_PATH=${PARALLEL_MATER_METAL_GENERATED_DIR}/module-cache
                    ${PARALLEL_MATER_XCRUN_EXECUTABLE} -sdk macosx metal
                    -std=metal4.0 -mmacosx-version-min=26.0
                    -fmodules-cache-path=${PARALLEL_MATER_METAL_GENERATED_DIR}/module-cache
                    -c ${CMAKE_CURRENT_SOURCE_DIR}/examples/metal_gallery/shaders.metal
                    -o ${PARALLEL_MATER_METAL_GALLERY_AIR}
            COMMAND ${PARALLEL_MATER_METALLIB_COMPILER}
                    ${PARALLEL_MATER_METAL_GALLERY_AIR}
                    -o ${PARALLEL_MATER_METAL_GALLERY_METALLIB}
            COMMAND ${PARALLEL_MATER_XXD_EXECUTABLE} -i
                    -n parallel_mater_metal_gallery_metallib_data
                    ${PARALLEL_MATER_METAL_GALLERY_METALLIB}
                    ${PARALLEL_MATER_METAL_GALLERY_INCLUDE}
            DEPENDS examples/metal_gallery/shaders.metal
            VERBATIM)
        set_source_files_properties(
            examples/metal_gallery/main.mm
            PROPERTIES OBJECT_DEPENDS ${PARALLEL_MATER_METAL_GALLERY_INCLUDE})

        add_library(parallel-mater-metal-gallery-scene STATIC
            examples/support/scene.cpp
            examples/support/brick_scene.cpp
            examples/support/device_profiles.cpp
            examples/support/rope_render.cpp)
        target_compile_features(parallel-mater-metal-gallery-scene
                                PUBLIC cxx_std_20)
        target_compile_definitions(parallel-mater-metal-gallery-scene
                                   PUBLIC PARALLEL_MATER_GALLERY_METAL=1)
        target_include_directories(
            parallel-mater-metal-gallery-scene
            PUBLIC examples/support/include
            PRIVATE ${parallel_mater_cgltf_SOURCE_DIR})
        target_link_libraries(parallel-mater-metal-gallery-scene
                              PUBLIC ParallelMater::metal)
        target_compile_options(parallel-mater-metal-gallery-scene
                               PRIVATE -Wall -Wextra -Wpedantic)

        add_executable(parallel-mater-metal-rigid-scene-benchmark
                       benchmarks/metal_rigid_scene.mm)
        target_compile_options(
            parallel-mater-metal-rigid-scene-benchmark
            PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
        target_compile_definitions(
            parallel-mater-metal-rigid-scene-benchmark PRIVATE
            PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb")
        target_link_libraries(
            parallel-mater-metal-rigid-scene-benchmark PRIVATE
            parallel-mater-metal-gallery-scene
            ${PARALLEL_MATER_METAL_FRAMEWORK})

        if(BUILD_TESTING)
            add_executable(parallel-mater-metal-device-profile-tests
                           tests/gallery/device_profiles_tests.cpp)
            target_compile_definitions(
                parallel-mater-metal-device-profile-tests PRIVATE
                PARALLEL_MATER_GALLERY_METAL=1
                PARALLEL_MATER_DEVICE_PROFILES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
                PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb")
            target_link_libraries(parallel-mater-metal-device-profile-tests
                                  PRIVATE parallel-mater-metal-gallery-scene)
            add_test(NAME parallel-mater-metal-device-profile-tests
                     COMMAND parallel-mater-metal-device-profile-tests)

            add_executable(parallel-mater-metal-rigid-wall-tests
                           tests/metal/rigid_wall_tests.mm)
            target_compile_options(parallel-mater-metal-rigid-wall-tests
                                   PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
            target_compile_definitions(
                parallel-mater-metal-rigid-wall-tests PRIVATE
                PARALLEL_MATER_RIGID_BODY_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb")
            target_link_libraries(
                parallel-mater-metal-rigid-wall-tests PRIVATE
                parallel-mater-metal-gallery-scene
                ${PARALLEL_MATER_METAL_FRAMEWORK})
            add_test(NAME parallel-mater-metal-rigid-wall-tests
                     COMMAND parallel-mater-metal-rigid-wall-tests)
            set_tests_properties(
                parallel-mater-metal-rigid-wall-tests
                PROPERTIES TIMEOUT 600 RUN_SERIAL TRUE LABELS "metal;gallery;rigid")
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
        add_executable(parallel-mater-metal-conformance
                       conformance/v1/src/metal_runner.mm)
        target_link_libraries(parallel-mater-metal-conformance PRIVATE
            parallel-mater-conformance-registry
            parallel-mater-metal-gallery-scene
            ${PARALLEL_MATER_METAL_FRAMEWORK})
        target_compile_definitions(parallel-mater-metal-conformance PRIVATE
            PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
            PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
        target_compile_options(parallel-mater-metal-conformance
                               PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)

        if(BUILD_TESTING)
            add_test(NAME parallel-mater-metal-conformance-inputs
                     COMMAND parallel-mater-metal-conformance --check-inputs)
            add_test(NAME parallel-mater-metal-conformance-cuda-goldens
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/reproduce.py
                        --runner $<TARGET_FILE:parallel-mater-metal-conformance>
                        --comparator ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/compare.py
                        --cases ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/cases
                        --golden ${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/golden/cuda)
            set_tests_properties(parallel-mater-metal-conformance-cuda-goldens
                PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 RUN_SERIAL TRUE
                           LABELS "metal;conformance")
        endif()

        set(PARALLEL_MATER_METAL_GALLERY_GENERATED_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_gallery")
        file(MAKE_DIRECTORY ${PARALLEL_MATER_METAL_GALLERY_GENERATED_DIR})
        set(PARALLEL_MATER_METAL_GALLERY_BUILD_INFO
            "${PARALLEL_MATER_METAL_GALLERY_GENERATED_DIR}/parallel_mater_gallery_build_info.hpp")
        add_custom_target(
            parallel-mater-metal-gallery-build-info
            COMMAND
                ${CMAKE_COMMAND}
                -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                -DOUTPUT_FILE=${PARALLEL_MATER_METAL_GALLERY_BUILD_INFO}
                -DPROJECT_VERSION=${PROJECT_VERSION}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuildInfo.cmake
            BYPRODUCTS ${PARALLEL_MATER_METAL_GALLERY_BUILD_INFO}
            COMMENT "Generating Metal gallery build identity"
            VERBATIM)

        add_executable(parallel-mater-metal-gallery MACOSX_BUNDLE
                       examples/metal_gallery/main.mm)
        set(PARALLEL_MATER_METAL_GALLERY_ICON
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/branding/ParallelMater.icns")
        target_sources(parallel-mater-metal-gallery PRIVATE
                       "${PARALLEL_MATER_METAL_GALLERY_ICON}")
        set_source_files_properties("${PARALLEL_MATER_METAL_GALLERY_ICON}"
            PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
        file(GLOB PARALLEL_MATER_METAL_GALLERY_ASSETS CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/*.glb")
        target_sources(parallel-mater-metal-gallery PRIVATE
                       ${PARALLEL_MATER_METAL_GALLERY_ASSETS})
        set_source_files_properties(
            ${PARALLEL_MATER_METAL_GALLERY_ASSETS}
            PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/assets")
        target_sources(parallel-mater-metal-gallery PRIVATE
                       ${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json)
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json
            PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/config")
        add_dependencies(parallel-mater-metal-gallery
                         parallel-mater-metal-gallery-build-info)
        set_target_properties(
            parallel-mater-metal-gallery
            PROPERTIES
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED YES
                OBJCXX_STANDARD 20
                OBJCXX_STANDARD_REQUIRED YES
                OUTPUT_NAME "ParallelMater Metal Gallery"
                MACOSX_BUNDLE_INFO_PLIST
                    "${CMAKE_CURRENT_SOURCE_DIR}/examples/metal_gallery/Info.plist.in"
                MACOSX_BUNDLE_BUNDLE_NAME "ParallelMater Metal Gallery"
                MACOSX_BUNDLE_ICON_FILE "ParallelMater.icns"
                MACOSX_BUNDLE_GUI_IDENTIFIER
                    "com.tabutyn.parallel-mater.metal-gallery"
                MACOSX_BUNDLE_BUNDLE_VERSION "1"
                MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
                OSX_DEPLOYMENT_TARGET 26.0)
        target_compile_options(parallel-mater-metal-gallery
                               PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
        target_include_directories(
            parallel-mater-metal-gallery
            PRIVATE
                ${PARALLEL_MATER_METAL_GENERATED_DIR}
                ${PARALLEL_MATER_METAL_GALLERY_GENERATED_DIR})
        target_compile_definitions(
            parallel-mater-metal-gallery
            PRIVATE
                PARALLEL_MATER_GALLERY_METAL=1
                PARALLEL_MATER_DEVICE_PROFILES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json"
                PARALLEL_MATER_DEFAULT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/RigidBody.glb"
                PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintFixed.glb"
                PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPoint.glb"
                PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintHinge.glb"
                PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintPiston.glb"
                PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGeneric.glb"
                PARALLEL_MATER_CONSTRAINT_GENERIC_SPRING_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintGenericSpring.glb"
                PARALLEL_MATER_CONSTRAINT_MOTOR_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/ConstraintMotor.glb"
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
        target_link_libraries(
            parallel-mater-metal-gallery
            PRIVATE
                parallel-mater-metal-gallery-scene
                glfw
                ${PARALLEL_MATER_APPKIT_FRAMEWORK}
                ${PARALLEL_MATER_QUARTZCORE_FRAMEWORK})
        add_custom_command(
            TARGET parallel-mater-metal-gallery POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-metal-gallery
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                    $<TARGET_FILE:parallel-mater-metal-gallery>
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-metal-gallery
            COMMENT "Creating Metal gallery command-line compatibility link"
            VERBATIM)

        set(PARALLEL_MATER_METAL_GALLERY_DEPLOY_DIRECTORY "/Applications"
            CACHE PATH "Directory containing the Dock-pinned Metal gallery")
        add_custom_target(
            deploy-metal-gallery
            COMMAND
                ${CMAKE_COMMAND}
                -DSOURCE_APP=$<TARGET_BUNDLE_DIR:parallel-mater-metal-gallery>
                -DDESTINATION_APP=${PARALLEL_MATER_METAL_GALLERY_DEPLOY_DIRECTORY}/ParallelMater\ Metal\ Gallery.app
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeployMacOSBundle.cmake
            DEPENDS parallel-mater-metal-gallery
            COMMENT "Deploying ParallelMater Metal Gallery to ${PARALLEL_MATER_METAL_GALLERY_DEPLOY_DIRECTORY}"
            USES_TERMINAL
            VERBATIM)

        if(BUILD_TESTING)
            add_test(
                NAME parallel-mater-metal-first-context
                COMMAND parallel-mater-metal-gallery
                        --headless --frames 120 --validate)
            set_tests_properties(parallel-mater-metal-first-context
                                 PROPERTIES TIMEOUT 180 LABELS "metal;gallery")
            add_test(
                NAME parallel-mater-metal-gallery-all-contexts
                COMMAND parallel-mater-metal-gallery
                        --all-scenes --frames 60 --validate
                        --width 320 --height 240
                        --output ${CMAKE_CURRENT_BINARY_DIR}/metal-gallery-test-captures)
            set_tests_properties(
                parallel-mater-metal-gallery-all-contexts
                PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE
                           LABELS "metal;gallery;subsystem")
        endif()
    endif()
endif()

if(PARALLEL_MATER_BUILD_SLANG_AVBD)
    include(cmake/SlangAvbd.cmake)
    parallel_mater_add_slang_avbd()
endif()

if(BUILD_TESTING)
    add_subdirectory(tests/solver)
    # Compile both registry/header contracts on Linux too; the native Metal
    # renderer and GPU kernels still require validation on an Apple host.
    foreach(registry_backend IN ITEMS cuda metal)
        if(registry_backend STREQUAL "cuda" AND NOT PARALLEL_MATER_BUILD_CUDA)
            continue()
        endif()
        set(registry_target parallel-mater-${registry_backend}-gallery-registry-tests)
        add_executable(${registry_target} tests/gallery/gallery_registry_contract_tests.cpp)
        target_compile_features(${registry_target} PRIVATE cxx_std_20)
        target_include_directories(${registry_target} PRIVATE include examples/support/include)
        if(registry_backend STREQUAL "metal")
            target_compile_definitions(${registry_target} PRIVATE PARALLEL_MATER_GALLERY_METAL=1)
        else()
            target_link_libraries(${registry_target} PRIVATE CUDA::cudart)
        endif()
        add_test(NAME ${registry_target} COMMAND ${registry_target})
    endforeach()
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

    add_executable(parallel-mater-parity-compare
                   tests/parity/compare.cpp)
    target_compile_features(parallel-mater-parity-compare PRIVATE cxx_std_20)
    add_test(NAME parallel-mater-parity-compare-self-test
             COMMAND parallel-mater-parity-compare --self-test)
endif()

if(NOT PARALLEL_MATER_INSTALL_TARGETS)
    message(FATAL_ERROR "At least one ParallelMater backend must be enabled")
endif()

set(PARALLEL_MATER_PACKAGE_NEEDS_CUDA ${PARALLEL_MATER_BUILD_CUDA})
configure_package_config_file(
    cmake/ParallelMaterConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    TARGETS ${PARALLEL_MATER_INSTALL_TARGETS}
    EXPORT ParallelMaterTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
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
