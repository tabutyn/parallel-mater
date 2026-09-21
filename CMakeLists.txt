cmake_minimum_required(VERSION 3.25)

project(ParallelMater VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

# CMake 3.25 assumes the removed nvToolsExt library exists when discovering
# CUDA 13. NVTX is not used by the public target, so provide the compatibility
# target before FindCUDAToolkit evaluates that deprecated component.
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()
find_package(CUDAToolkit REQUIRED)

option(PARALLEL_MATER_BUILD_EXAMPLES "Build ParallelMater examples" ON)

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
