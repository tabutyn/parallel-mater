cmake_minimum_required(VERSION 3.25)

project(ParallelMater VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

find_path(
    PARALLEL_MATER_CUDA_INCLUDE_DIR
    cuda_runtime_api.h
    HINTS ENV CUDAToolkit_ROOT ENV CUDA_PATH
    PATHS /usr/local/cuda/include
    REQUIRED)

add_library(parallel-mater-api INTERFACE)
add_library(ParallelMater::api ALIAS parallel-mater-api)
set_target_properties(parallel-mater-api PROPERTIES EXPORT_NAME api)
target_compile_features(parallel-mater-api INTERFACE cxx_std_20)
target_include_directories(
    parallel-mater-api
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PARALLEL_MATER_CUDA_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(BUILD_TESTING)
    add_executable(parallel-mater-api-contract-tests tests/api_contract.cpp)
    target_link_libraries(parallel-mater-api-contract-tests PRIVATE ParallelMater::api)
    add_test(NAME parallel-mater-api-contract-tests COMMAND parallel-mater-api-contract-tests)
endif()

install(TARGETS parallel-mater-api EXPORT ParallelMaterTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT ParallelMaterTargets
    FILE ParallelMaterTargets.cmake
    NAMESPACE ParallelMater::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater)
