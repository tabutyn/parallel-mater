# Platform-neutral Parallel Mater loader, validation layer, and driver-discovery
# contract test. This module deliberately has no device-backend dependency.

find_package(Vulkan QUIET)

add_library(parallel-mater SHARED
    src/loader/driver.cpp
    src/loader/entry.cpp)
add_library(ParallelMater::parallel_mater ALIAS parallel-mater)
add_library(ParallelMater::api ALIAS parallel-mater)
set_target_properties(parallel-mater PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    EXPORT_NAME parallel_mater
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(parallel-mater
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${PARALLEL_MATER_API_GENERATED_DIR})
target_compile_definitions(parallel-mater PRIVATE
    PARALLEL_MATER_LOADER_IMPLEMENTATION=1
    PARALLEL_MATER_SYSTEM_DRIVER_MANIFEST_DIR="${CMAKE_INSTALL_FULL_DATADIR}/parallel-mater/drivers.d"
    PARALLEL_MATER_SYSTEM_LAYER_MANIFEST_DIR="${CMAKE_INSTALL_FULL_DATADIR}/parallel-mater/layers.d")
if(Vulkan_FOUND)
    target_include_directories(parallel-mater PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_compile_definitions(parallel-mater PRIVATE
        PARALLEL_MATER_HAS_VULKAN_INTEROP=1)
endif()
target_link_libraries(parallel-mater PRIVATE ${CMAKE_DL_LIBS})
parallel_mater_enable_warnings(parallel-mater)
list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater)

add_library(parallel-mater-layer-validation SHARED
    src/layers/validation.cpp)
set_target_properties(parallel-mater-layer-validation PROPERTIES
    OUTPUT_NAME parallel_mater_layer_validation
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
target_include_directories(parallel-mater-layer-validation PRIVATE include)
list(APPEND PARALLEL_MATER_INSTALL_TARGETS parallel-mater-layer-validation)

set(PARALLEL_MATER_BUILD_MANIFEST_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/manifests")
file(GENERATE
    OUTPUT "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json"
    CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"layer\": {\n    \"name\": \"PM_LAYER_KHRONOS_validation\",\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-layer-validation>\",\n    \"api_version\": \"0.2.0\"\n  }\n}\n")

if(BUILD_TESTING)
    add_library(parallel-mater-mock-driver SHARED tests/api/mock_driver.cpp)
    set_target_properties(parallel-mater-mock-driver PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    target_compile_definitions(parallel-mater-mock-driver PRIVATE
        PARALLEL_MATER_DRIVER_IMPLEMENTATION=1)
    target_include_directories(parallel-mater-mock-driver PRIVATE include)
    set(PARALLEL_MATER_TEST_DRIVER_MANIFEST_DIR
        "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/test-drivers")
    file(GENERATE
        OUTPUT "${PARALLEL_MATER_TEST_DRIVER_MANIFEST_DIR}/mock.json"
        CONTENT "{\n  \"file_format_version\": \"1.0.0\",\n  \"PMD\": {\n    \"library_path\": \"$<TARGET_FILE:parallel-mater-mock-driver>\",\n    \"api_version\": \"0.2.0\",\n    \"library_arch\": \"${PARALLEL_MATER_LIBRARY_ARCH}\"\n  }\n}\n")
    add_executable(parallel-mater-loader-driver-discovery-tests
        tests/api/loader_driver_discovery_tests.cpp)
    target_compile_features(parallel-mater-loader-driver-discovery-tests
        PRIVATE cxx_std_20)
    target_link_libraries(parallel-mater-loader-driver-discovery-tests
        PRIVATE ParallelMater::parallel_mater)
    add_dependencies(parallel-mater-loader-driver-discovery-tests
        parallel-mater-mock-driver)
    add_test(NAME parallel-mater-loader-driver-discovery-tests
        COMMAND parallel-mater-loader-driver-discovery-tests)
    set_tests_properties(parallel-mater-loader-driver-discovery-tests PROPERTIES
        LABELS "api;loader;driver"
        ENVIRONMENT
            "PM_DRIVER_PATH=${PARALLEL_MATER_TEST_DRIVER_MANIFEST_DIR}")
endif()
