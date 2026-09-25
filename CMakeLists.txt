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

add_library(parallel-mater STATIC src/world.cu)
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

    add_library(parallel-mater-gallery-scene STATIC examples/support/scene.cpp)
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
                examples/support/fluid_surface.cu
                examples/support/contact_paint.cu
                examples/support/overlay.cpp)
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
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
endif()

if(BUILD_TESTING)
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

    if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
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

        add_executable(parallel-mater-material-visibility-tests
                       tests/gallery/material_visibility_tests.cpp)
        target_link_libraries(parallel-mater-material-visibility-tests PRIVATE
            parallel-mater-gallery-renderer CUDA::cudart)
        target_compile_definitions(parallel-mater-material-visibility-tests PRIVATE
            PARALLEL_MATER_ALPHA_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/assets/material_alpha.gltf"
            PARALLEL_MATER_PEGS_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Pegs.glb"
            PARALLEL_MATER_OPTIX_PTX_PATH="${PARALLEL_MATER_OPTIX_PTX}")
        add_test(NAME parallel-mater-material-visibility-tests
                 COMMAND parallel-mater-material-visibility-tests)
        set_tests_properties(parallel-mater-material-visibility-tests
                             PROPERTIES SKIP_RETURN_CODE 77)

        if(PARALLEL_MATER_BUILD_BENCHMARKS)
            add_test(NAME parallel-mater-pegs-stability-test
                     COMMAND parallel-mater-pegs-benchmark --verify-settling)
            set_tests_properties(parallel-mater-pegs-stability-test
                                 PROPERTIES SKIP_RETURN_CODE 77)
            add_test(NAME parallel-mater-pegs-tilt-test
                     COMMAND parallel-mater-pegs-benchmark --verify-tilt)
            set_tests_properties(parallel-mater-pegs-tilt-test
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

        add_executable(parallel-mater-contact-paint-tests
                       tests/gallery/contact_paint_tests.cu)
        target_include_directories(parallel-mater-contact-paint-tests PRIVATE
            examples/support ${parallel_mater_optix_dev_SOURCE_DIR}/include)
        target_link_libraries(parallel-mater-contact-paint-tests PRIVATE
            parallel-mater-gallery-renderer CUDA::cudart)
        add_test(NAME parallel-mater-contact-paint-tests
                 COMMAND parallel-mater-contact-paint-tests)
        set_tests_properties(parallel-mater-contact-paint-tests
                             PROPERTIES SKIP_RETURN_CODE 77)

        add_test(
            NAME parallel-mater-optix-headless-test
            COMMAND parallel-mater-gallery
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-optix-test.ppm
                    --frames 30)
        add_test(
            NAME parallel-mater-dump-headless-test
            COMMAND parallel-mater-gallery
                    --dump-spheres 10
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-dump-test.ppm
                    --frames 150)
        add_test(
            NAME parallel-mater-fluid-headless-test
            COMMAND parallel-mater-gallery
                    --fluid
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-fluid-test.ppm
                    --frames 30)
        add_test(
            NAME parallel-mater-fluid-rigid-headless-test
            COMMAND parallel-mater-gallery
                    --fluid-rigid
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-fluid-rigid-test.ppm
                    --frames 120)
        add_test(
            NAME parallel-mater-peg-paint-headless-test
            COMMAND parallel-mater-gallery
                    --peg-paint
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-peg-paint-test.ppm
                    --frames 180)
        add_test(
            NAME parallel-mater-peg-paint-containment-test
            COMMAND parallel-mater-gallery
                    --peg-paint
                    --fluid-particle-view
                    --trace-fluid-escapes
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-peg-paint-containment-test.ppm
                    --frames 600)
        add_test(
            NAME parallel-mater-peg-paint-cap-test
            COMMAND parallel-mater-gallery
                    --peg-paint
                    --fluid-particles 100
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-peg-paint-cap-test.ppm
                    --frames 180)
        add_test(
            NAME parallel-mater-fluid-containment-test
            COMMAND parallel-mater-gallery
                    --fluid
                    --trace-fluid-escapes
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-fluid-containment-test.ppm
                    --frames 120)
        add_test(
            NAME parallel-mater-fluid-cap-headless-test
            COMMAND parallel-mater-gallery
                    --fluid-particles 100
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-fluid-cap-test.ppm
                    --frames 30)
        set_tests_properties(parallel-mater-fluid-cap-headless-test PROPERTIES
            PASS_REGULAR_EXPRESSION "Fluid particles=100 ")
        add_test(
            NAME parallel-mater-fluid-particle-view-test
            COMMAND parallel-mater-gallery
                    --fluid-particle-view
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-fluid-particles-test.ppm
                    --frames 30)
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
