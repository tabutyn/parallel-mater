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

option(PARALLEL_MATER_BUILD_EXAMPLES "Build ParallelMater examples" ON)
option(PARALLEL_MATER_BUILD_OPTIX_GALLERY
       "Build the optional OptiX and glTF gallery" OFF)

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

if(PARALLEL_MATER_BUILD_EXAMPLES)
    add_executable(parallel-mater-rigid-sandbox examples/rigid_sandbox/main.cpp)
    target_link_libraries(
        parallel-mater-rigid-sandbox PRIVATE ParallelMater::parallel_mater)
endif()

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
                examples/support/optix_renderer.cpp)
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

    if(PARALLEL_MATER_BUILD_OPTIX_GALLERY)
        add_executable(parallel-mater-gallery-scene-tests
                       tests/gallery/scene_tests.cpp)
        target_link_libraries(
            parallel-mater-gallery-scene-tests
            PRIVATE parallel-mater-gallery-scene)
        target_compile_definitions(
            parallel-mater-gallery-scene-tests
            PRIVATE
                PARALLEL_MATER_PASSIVE_ACTIVE_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/PassiveActive.glb")
        add_test(NAME parallel-mater-gallery-scene-tests
                 COMMAND parallel-mater-gallery-scene-tests)
        set_tests_properties(parallel-mater-gallery-scene-tests
                             PROPERTIES SKIP_RETURN_CODE 77)

        add_test(
            NAME parallel-mater-optix-headless-test
            COMMAND parallel-mater-gallery
                    --headless
                    ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-optix-test.ppm
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
