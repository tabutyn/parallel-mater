cmake_minimum_required(VERSION 3.25)

project(ParallelMater VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)
include(CTest)

# CMake 3.25 assumes the removed nvToolsExt library exists when discovering
# CUDA 13. NVTX is not used by the public target, so provide the compatibility
# target before FindCUDAToolkit evaluates that deprecated component.
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()
find_package(CUDAToolkit REQUIRED)

option(PARALLEL_MATER_BUILD_OPTIX_GALLERY
       "Build the optional OptiX and glTF gallery" OFF)
option(PARALLEL_MATER_BUILD_BENCHMARKS
       "Build local performance benchmark executables" OFF)
if(PARALLEL_MATER_BUILD_BENCHMARKS AND
   NOT PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    message(FATAL_ERROR
            "Rigid scene benchmarks require PARALLEL_MATER_BUILD_OPTIX_GALLERY")
endif()

add_library(parallel-mater STATIC src/world.cu src/fluid_geometry.cpp src/fluid_source.cpp src/soft_body_geometry.cpp)
add_library(ParallelMater::parallel_mater ALIAS parallel-mater)
add_library(ParallelMater::api ALIAS parallel-mater)
set_target_properties(
    parallel-mater
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        EXPORT_NAME parallel_mater
        POSITION_INDEPENDENT_CODE ON)
set_property(TARGET parallel-mater PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_20)
target_include_directories(
    parallel-mater
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(parallel-mater PUBLIC CUDA::cudart)
target_compile_options(
    parallel-mater
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
    FetchContent_Declare(
        parallel_mater_cgltf
        GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
        GIT_TAG 85cd62382dfea638278962690cf515023f33ed00
        GIT_SHALLOW TRUE)
    FetchContent_GetProperties(parallel_mater_cgltf)
    if(NOT parallel_mater_cgltf_POPULATED)
        FetchContent_Populate(parallel_mater_cgltf)
    endif()

    add_library(parallel-mater-gallery-scene STATIC
                examples/support/scene.cpp
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

    if(PARALLEL_MATER_BUILD_BENCHMARKS)
        add_executable(parallel-mater-rigid-scene-benchmark
                       benchmarks/rigid_scene.cpp)
        target_link_libraries(
            parallel-mater-rigid-scene-benchmark
            PRIVATE parallel-mater-gallery-scene)
        target_compile_options(
            parallel-mater-rigid-scene-benchmark
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

    add_executable(parallel-mater-gallery examples/gallery/main.cpp)
    target_link_libraries(
        parallel-mater-gallery
        PRIVATE
            parallel-mater-gallery-renderer
            glfw
            OpenGL::GL)
    target_compile_definitions(
        parallel-mater-gallery
        PRIVATE
            PARALLEL_MATER_DEFAULT_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/PassiveActive.glb"
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
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
endif()

if(BUILD_TESTING)
    add_executable(parallel-mater-soft-body-geometry-tests tests/soft_body_geometry_tests.cpp)
    target_link_libraries(parallel-mater-soft-body-geometry-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-soft-body-geometry-tests COMMAND parallel-mater-soft-body-geometry-tests)
    add_executable(parallel-mater-fluid-soft-body-tests tests/fluid_soft_body_tests.cpp)
    target_link_libraries(parallel-mater-fluid-soft-body-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-fluid-soft-body-tests COMMAND parallel-mater-fluid-soft-body-tests)
    set_tests_properties(parallel-mater-fluid-soft-body-tests PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(parallel-mater-api-contract-tests tests/api_contract.cpp)
    target_link_libraries(
        parallel-mater-api-contract-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-api-contract-tests
             COMMAND parallel-mater-api-contract-tests)

    add_executable(parallel-mater-rigid-tests tests/rigid_tests.cpp)
    target_link_libraries(
        parallel-mater-rigid-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-rigid-tests COMMAND parallel-mater-rigid-tests)
    set_tests_properties(parallel-mater-rigid-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-fluid-tests tests/fluid_tests.cpp)
    target_link_libraries(
        parallel-mater-fluid-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-fluid-tests COMMAND parallel-mater-fluid-tests)
    set_tests_properties(parallel-mater-fluid-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-fluid-source-tests tests/fluid_source_tests.cpp)
    target_link_libraries(parallel-mater-fluid-source-tests PRIVATE ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-fluid-source-tests COMMAND parallel-mater-fluid-source-tests)
    set_tests_properties(parallel-mater-fluid-source-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-paint-geometry-tests
                   tests/paint_geometry_tests.cpp)
    target_link_libraries(parallel-mater-paint-geometry-tests PRIVATE
        ParallelMater::parallel_mater)
    add_test(NAME parallel-mater-paint-geometry-tests
             COMMAND parallel-mater-paint-geometry-tests)

    if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
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
            parallel-mater-dump-headless-test
            parallel-mater-dump-test.ppm 150 --dump-spheres 10)
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

configure_package_config_file(
    cmake/ParallelMaterConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    TARGETS parallel-mater
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
