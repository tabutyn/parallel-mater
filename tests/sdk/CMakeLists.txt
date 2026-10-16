# SPDX-License-Identifier: MIT
# This directory is staged with ONLY the listed public consumer/driver sources.
cmake_minimum_required(VERSION 3.25)
project(ParallelMaterExternalSdkGate LANGUAGES C CXX)
include(CTest)
set(PM_REQUIRED_VERSION 0.2 CACHE STRING "Requested SDK compatibility line")
find_package(ParallelMater ${PM_REQUIRED_VERSION} CONFIG REQUIRED
    COMPONENTS loader scene preparation validation)
add_library(external-mock SHARED mock_driver.cpp)
target_include_directories(external-mock PRIVATE
    $<TARGET_PROPERTY:ParallelMater::api,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_features(external-mock PRIVATE cxx_std_20)
set_target_properties(external-mock PROPERTIES CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
# Intentionally no link to loader, reference driver or internal source.
file(GENERATE OUTPUT "$<TARGET_FILE_DIR:external-mock>/mock.json"
    CONTENT "{\"file_format_version\":\"1.0.0\",\"PMD\":{\"library_path\":\"$<TARGET_FILE_NAME:external-mock>\",\"api_version\":\"0.2.0\"}}\n")
add_executable(c-consumer devices.c)
target_compile_features(c-consumer PRIVATE c_std_11)
target_link_libraries(c-consumer PRIVATE ParallelMater::api)
add_executable(cpp-consumer loader_driver_discovery_tests.cpp)
target_compile_features(cpp-consumer PRIVATE cxx_std_20)
target_link_libraries(cpp-consumer PRIVATE ParallelMater::api)
add_executable(scene-consumer scene_consumer.cpp)
target_compile_features(scene-consumer PRIVATE cxx_std_20)
target_link_libraries(scene-consumer PRIVATE ParallelMater::scene)
foreach(consumer c-consumer cpp-consumer scene-consumer)
    add_test(NAME ${consumer} COMMAND ${consumer})
    set_tests_properties(${consumer} PROPERTIES ENVIRONMENT
        "PM_DRIVER_FILES=$<TARGET_FILE_DIR:external-mock>/mock.json")
endforeach()
