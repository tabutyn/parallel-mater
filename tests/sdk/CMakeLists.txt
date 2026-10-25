# SPDX-License-Identifier: MIT
# This directory is staged with ONLY the listed public consumer/driver sources.
cmake_minimum_required(VERSION 3.25)
project(ParallelMaterExternalSdkGate LANGUAGES C CXX)
include(CTest)
set(PM_REQUIRED_VERSION 0.2 CACHE STRING "Requested SDK compatibility line")
set(PM_REALTIME_DRIVER_MANIFEST "" CACHE FILEPATH
    "Installed production driver manifest used by the realtime SDK gate")
set(PM_REALTIME_OBSERVATION_DIR "" CACHE PATH
    "Directory receiving installed public realtime observations")
set(PM_REALTIME_CANDIDATE_ID "" CACHE STRING "Exact candidate manifest id")
set(PM_REALTIME_CANDIDATE_SHA256 "" CACHE STRING
    "Exact candidate manifest SHA-256")
set(PM_REALTIME_CANDIDATE_REVISION "" CACHE STRING
    "Exact candidate source revision")
option(PM_ENABLE_REALTIME_PROFILE_TESTS
    "Build and run the installed production realtime profile gate" ON)
if(PM_ENABLE_REALTIME_PROFILE_TESTS)
    foreach(required IN ITEMS PM_REALTIME_DRIVER_MANIFEST
                              PM_REALTIME_OBSERVATION_DIR
                              PM_REALTIME_CANDIDATE_ID
                              PM_REALTIME_CANDIDATE_SHA256
                              PM_REALTIME_CANDIDATE_REVISION)
        if("${${required}}" STREQUAL "")
            message(FATAL_ERROR "${required} is required for the realtime profile gate")
        endif()
    endforeach()
endif()
find_package(ParallelMater ${PM_REQUIRED_VERSION} CONFIG REQUIRED
    COMPONENTS loader scene preparation validation)
add_library(external-mock SHARED mock_driver.cpp)
target_include_directories(external-mock PRIVATE
    $<TARGET_PROPERTY:ParallelMater::api,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(external-mock PRIVATE
    PARALLEL_MATER_DRIVER_IMPLEMENTATION=1)
target_compile_features(external-mock PRIVATE cxx_std_20)
set_target_properties(external-mock PROPERTIES CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
# Intentionally no link to loader, reference driver or internal source.
file(GENERATE OUTPUT "$<TARGET_FILE_DIR:external-mock>/mock.json"
    CONTENT "{\"file_format_version\":\"1.0.0\",\"PMD\":{\"library_path\":\"$<TARGET_FILE_NAME:external-mock>\",\"api_version\":\"0.2.0\"}}\n")

# Compile negotiated ABI fixtures only from installed public C/C++ headers.
# No loader, layer implementation or solver source is staged into this project.
foreach(version IN ITEMS 1 2 3 0 999)
    add_library(external-layer-${version} SHARED mock_layer.c)
    target_compile_features(external-layer-${version} PRIVATE c_std_11)
    target_include_directories(external-layer-${version} PRIVATE
        $<TARGET_PROPERTY:ParallelMater::api,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(external-layer-${version} PRIVATE
        PM_TEST_LAYER_VERSION=${version})
    set_target_properties(external-layer-${version} PROPERTIES
        C_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
    file(GENERATE
        OUTPUT "$<TARGET_FILE_DIR:external-layer-${version}>/layer-${version}.json"
        CONTENT "{\"file_format_version\":\"1.0.0\",\"layer\":{\"library_path\":\"$<TARGET_FILE_NAME:external-layer-${version}>\",\"api_version\":\"0.2.0\"}}\n")
endforeach()

foreach(version IN ITEMS 0 999)
    add_library(external-driver-${version} SHARED mock_driver.cpp)
    target_compile_features(external-driver-${version} PRIVATE cxx_std_20)
    target_include_directories(external-driver-${version} PRIVATE
        $<TARGET_PROPERTY:ParallelMater::api,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(external-driver-${version} PRIVATE
        PARALLEL_MATER_DRIVER_IMPLEMENTATION=1
        PM_MOCK_INTERFACE_VERSION=${version})
    set_target_properties(external-driver-${version} PROPERTIES
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
    file(GENERATE
        OUTPUT "$<TARGET_FILE_DIR:external-driver-${version}>/driver-${version}.json"
        CONTENT "{\"file_format_version\":\"1.0.0\",\"PMD\":{\"library_path\":\"$<TARGET_FILE_NAME:external-driver-${version}>\",\"api_version\":\"0.2.0\"}}\n")
endforeach()

file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/installed-layer-3.json"
    CONTENT "{\"file_format_version\":\"1.0.0\",\"layer\":{\"library_path\":\"$<TARGET_FILE:ParallelMater::parallel-mater-layer-validation>\",\"api_version\":\"0.2.0\"}}\n")

add_executable(c-consumer devices.c)
target_compile_features(c-consumer PRIVATE c_std_11)
target_link_libraries(c-consumer PRIVATE ParallelMater::api)
add_executable(diagnostics-consumer diagnostics.c)
target_compile_features(diagnostics-consumer PRIVATE c_std_11)
target_link_libraries(diagnostics-consumer PRIVATE ParallelMater::api)
add_executable(cpp-consumer loader_driver_discovery_tests.cpp)
target_compile_features(cpp-consumer PRIVATE cxx_std_20)
target_link_libraries(cpp-consumer PRIVATE ParallelMater::api)
add_executable(scene-consumer scene_consumer.cpp)
target_compile_features(scene-consumer PRIVATE cxx_std_20)
target_link_libraries(scene-consumer PRIVATE ParallelMater::scene)
add_executable(compatibility-consumer compatibility.c)
target_compile_features(compatibility-consumer PRIVATE c_std_11)
target_link_libraries(compatibility-consumer PRIVATE ParallelMater::api)
if(PM_ENABLE_REALTIME_PROFILE_TESTS)
    add_executable(realtime-profile-consumer realtime_profile.c)
    target_compile_features(realtime-profile-consumer PRIVATE c_std_11)
    target_link_libraries(realtime-profile-consumer PRIVATE ParallelMater::api)
endif()
add_executable(legacy-0-2-consumer legacy_0_2_client.c)
target_compile_features(legacy-0-2-consumer PRIVATE c_std_11)
target_link_libraries(legacy-0-2-consumer PRIVATE ParallelMater::api)
foreach(consumer c-consumer diagnostics-consumer cpp-consumer scene-consumer)
    add_test(NAME ${consumer} COMMAND ${consumer})
    set_tests_properties(${consumer} PROPERTIES ENVIRONMENT
        "PM_DRIVER_FILES=$<TARGET_FILE_DIR:external-mock>/mock.json")
endforeach()
if(PM_ENABLE_REALTIME_PROFILE_TESTS)
    foreach(mode IN ITEMS plain validation)
        add_test(NAME realtime-profile-${mode}
            COMMAND realtime-profile-consumer ${mode}
                "${PM_REALTIME_OBSERVATION_DIR}/${mode}.json"
                "${PM_REALTIME_CANDIDATE_ID}"
                "${PM_REALTIME_CANDIDATE_SHA256}"
                "${PM_REALTIME_CANDIDATE_REVISION}")
        set_tests_properties(realtime-profile-${mode} PROPERTIES ENVIRONMENT
            "PM_DRIVER_FILES=${PM_REALTIME_DRIVER_MANIFEST}")
    endforeach()
endif()
add_test(NAME legacy-0-2-consumer COMMAND legacy-0-2-consumer)
set_tests_properties(legacy-0-2-consumer PROPERTIES ENVIRONMENT
    "PM_DRIVER_FILES=$<TARGET_FILE_DIR:external-mock>/mock.json")

function(add_compatibility_test name mode driver layer)
    add_test(NAME compatibility-${name} COMMAND compatibility-consumer ${mode})
    set_tests_properties(compatibility-${name} PROPERTIES ENVIRONMENT
        "PM_DRIVER_FILES=${driver};PM_LAYER_FILES=${layer}")
endfunction()

set(normal_driver "$<TARGET_FILE_DIR:external-mock>/mock.json")
foreach(mode IN ITEMS app-default app-current app-old app-future
                      instance-extension device-extension)
    add_compatibility_test(${mode} ${mode} "${normal_driver}" "")
endforeach()
add_compatibility_test(layer-current layer "${normal_driver}"
    "${CMAKE_CURRENT_BINARY_DIR}/installed-layer-3.json")
foreach(version IN ITEMS 1 2 3)
    add_compatibility_test(layer-abi-${version} layer "${normal_driver}"
        "$<TARGET_FILE_DIR:external-layer-${version}>/layer-${version}.json")
endforeach()
foreach(version IN ITEMS 0 999)
    add_compatibility_test(reject-driver-${version} reject-driver
        "$<TARGET_FILE_DIR:external-driver-${version}>/driver-${version}.json" "")
    add_compatibility_test(reject-layer-${version} reject-layer
        "${normal_driver}"
        "$<TARGET_FILE_DIR:external-layer-${version}>/layer-${version}.json")
endforeach()
