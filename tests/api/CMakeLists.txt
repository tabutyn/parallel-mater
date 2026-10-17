# SDK-only tests can also be built against a separately installed loader.
cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ParallelMaterApiTests LANGUAGES CXX)
    include(CTest)
    find_package(ParallelMater CONFIG REQUIRED)
endif()
foreach(suite resource-lifecycle domain-lifecycle coupling-lifecycle interaction-graph-lifecycle graph-validation geometry-lifecycle paint-lifecycle endpoint-lifecycle)
    string(REPLACE "-" "_" source_name "${suite}")
    add_executable(parallel-mater-${suite}-tests ${source_name}_tests.cpp)
    target_compile_features(parallel-mater-${suite}-tests PRIVATE cxx_std_20)
    target_link_libraries(parallel-mater-${suite}-tests PRIVATE ParallelMater::parallel_mater)
    foreach(backend cpu cuda vulkan d3d12 metal)
        if(TARGET parallel-mater-driver-slang-${backend})
            add_dependencies(parallel-mater-${suite}-tests parallel-mater-driver-slang-${backend})
            add_test(NAME parallel-mater-${backend}-${suite}-tests COMMAND parallel-mater-${suite}-tests)
            set_tests_properties(parallel-mater-${backend}-${suite}-tests PROPERTIES
                LABELS "api;lifecycle;${backend}" SKIP_RETURN_CODE 77 TIMEOUT 120
                ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-${backend}.json")
            if(suite STREQUAL "resource-lifecycle")
                add_test(NAME parallel-mater-${backend}-validated-lifecycle-tests
                    COMMAND parallel-mater-${suite}-tests)
                set_tests_properties(parallel-mater-${backend}-validated-lifecycle-tests PROPERTIES
                    LABELS "api;validation;lifecycle;${backend}" SKIP_RETURN_CODE 77 TIMEOUT 120
                    ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-${backend}.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json;PM_TEST_VALIDATION=1")
                add_dependencies(parallel-mater-${suite}-tests parallel-mater-layer-validation)
            endif()
        endif()
    endforeach()
    if((suite STREQUAL "paint-lifecycle" OR
        suite STREQUAL "endpoint-lifecycle") AND
       TARGET parallel-mater-test-driver-slang-cpu-resources)
        add_dependencies(parallel-mater-${suite}-tests
            parallel-mater-test-driver-slang-cpu-resources)
        add_test(NAME parallel-mater-host-${suite}-tests
            COMMAND parallel-mater-${suite}-tests)
        set_tests_properties(
            parallel-mater-host-${suite}-tests PROPERTIES
            LABELS "api;lifecycle;host-fixture" TIMEOUT 120
            ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/test-slang-cpu-resources.json")
    endif()
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        add_test(NAME parallel-mater-external-${suite}-tests COMMAND parallel-mater-${suite}-tests)
        set_tests_properties(parallel-mater-external-${suite}-tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
endforeach()

if(TARGET parallel-mater-test-driver-slang-cpu-resources)
    add_executable(parallel-mater-failure-lifecycle-tests
        failure_lifecycle_tests.cpp)
    target_compile_features(parallel-mater-failure-lifecycle-tests PRIVATE
        cxx_std_20)
    target_link_libraries(parallel-mater-failure-lifecycle-tests PRIVATE
        ParallelMater::parallel_mater)
    add_dependencies(parallel-mater-failure-lifecycle-tests
        parallel-mater-test-driver-slang-cpu-resources)
    add_test(NAME parallel-mater-host-failure-lifecycle-tests
        COMMAND parallel-mater-failure-lifecycle-tests)
    set_tests_properties(parallel-mater-host-failure-lifecycle-tests PROPERTIES
        LABELS "api;lifecycle;failure;host-fixture" TIMEOUT 120
        ENVIRONMENT "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/test-slang-cpu-resources.json")
endif()

if(WIN32)
    # SDK-only executables live in this subdirectory, separate from the loader
    # DLL. Windows has no build-tree RPATH; use the selected SDK's DLL directory
    # for tests instead of depending on a developer's global PATH or installation.
    get_property(api_lifecycle_tests DIRECTORY PROPERTY TESTS)
    set_tests_properties(${api_lifecycle_tests} PROPERTIES
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:ParallelMater::parallel_mater>")
endif()
