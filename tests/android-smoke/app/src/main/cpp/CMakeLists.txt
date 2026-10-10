cmake_minimum_required(VERSION 3.25)
project(ParallelMaterVulkanAndroidSmoke LANGUAGES CXX)

if(NOT PM_ROOT)
    message(FATAL_ERROR "PM_ROOT must point at the ParallelMater source tree")
endif()
if(NOT PARALLEL_MATER_SLANGC_EXECUTABLE)
    message(FATAL_ERROR
        "Pass host slangc with -PparallelMaterSlangc=/absolute/path/to/slangc")
endif()
if(NOT PARALLEL_MATER_SPIRV_VAL_EXECUTABLE)
    message(FATAL_ERROR
        "Pass current host spirv-val with -PparallelMaterSpirvVal=/absolute/path/to/spirv-val")
endif()
if(NOT PARALLEL_MATER_SPIRV_OPT_EXECUTABLE)
    message(FATAL_ERROR
        "Pass current host spirv-opt with -PparallelMaterSpirvOpt=/absolute/path/to/spirv-opt")
endif()

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(PARALLEL_MATER_BUILD_CUDA OFF CACHE BOOL "" FORCE)
set(PARALLEL_MATER_BUILD_METAL OFF CACHE BOOL "" FORCE)
set(PARALLEL_MATER_BUILD_VULKAN ON CACHE BOOL "" FORCE)
add_subdirectory("${PM_ROOT}" parallel-mater-build)

add_library(parallel_mater_vulkan_smoke SHARED smoke.cpp)
target_compile_features(parallel_mater_vulkan_smoke PRIVATE cxx_std_20)
target_link_libraries(parallel_mater_vulkan_smoke
    PRIVATE ParallelMater::vulkan android log)
