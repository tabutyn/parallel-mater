# SDK-only tests can also be built against a separately installed loader.
cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ParallelMaterApiTests LANGUAGES CXX)
    include(CTest)
    find_package(ParallelMater CONFIG REQUIRED)
endif()
find_package(Threads REQUIRED)
foreach(suite resource-lifecycle domain-lifecycle coupling-lifecycle interaction-graph-lifecycle graph-validation geometry-lifecycle paint-lifecycle endpoint-lifecycle fluid-pair-graph-lifecycle)
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
        suite STREQUAL "endpoint-lifecycle" OR
        suite STREQUAL "fluid-pair-graph-lifecycle") AND
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

add_executable(parallel-mater-host-concurrency-tests
    host_concurrency_tests.cpp)
target_compile_features(parallel-mater-host-concurrency-tests PRIVATE cxx_std_20)
target_link_libraries(parallel-mater-host-concurrency-tests PRIVATE
    ParallelMater::parallel_mater Threads::Threads)
foreach(backend cuda vulkan d3d12 metal)
    if(TARGET parallel-mater-driver-slang-${backend} AND
       TARGET parallel-mater-layer-validation)
        add_dependencies(parallel-mater-host-concurrency-tests
            parallel-mater-driver-slang-${backend}
            parallel-mater-layer-validation)
        add_test(NAME parallel-mater-${backend}-host-concurrency-tests
            COMMAND parallel-mater-host-concurrency-tests ${backend})
        set_tests_properties(
            parallel-mater-${backend}-host-concurrency-tests PROPERTIES
            LABELS "api;threading;validation;${backend}"
            SKIP_RETURN_CODE 77 TIMEOUT 120
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-${backend}.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json")
    endif()
endforeach()

add_executable(parallel-mater-descriptor-snapshot-tests
    descriptor_snapshot_tests.cpp)
target_compile_features(parallel-mater-descriptor-snapshot-tests PRIVATE
    cxx_std_20)
target_link_libraries(parallel-mater-descriptor-snapshot-tests PRIVATE
    ParallelMater::parallel_mater)
foreach(backend cuda vulkan d3d12 metal)
    if(TARGET parallel-mater-driver-slang-${backend})
        add_dependencies(parallel-mater-descriptor-snapshot-tests
            parallel-mater-driver-slang-${backend})
        add_test(NAME parallel-mater-${backend}-descriptor-snapshot-tests
            COMMAND parallel-mater-descriptor-snapshot-tests)
        set_tests_properties(
            parallel-mater-${backend}-descriptor-snapshot-tests PROPERTIES
            LABELS "api;lifecycle;snapshot;${backend}"
            SKIP_RETURN_CODE 77 TIMEOUT 120
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-${backend}.json")
    endif()
endforeach()

if(TARGET parallel-mater-test-driver-slang-cpu-resources)
    add_dependencies(parallel-mater-descriptor-snapshot-tests
        parallel-mater-test-driver-slang-cpu-resources)
    add_test(NAME parallel-mater-host-descriptor-snapshot-tests
        COMMAND parallel-mater-descriptor-snapshot-tests --failure-fixture)
    set_tests_properties(parallel-mater-host-descriptor-snapshot-tests PROPERTIES
        LABELS "api;lifecycle;snapshot;failure;host-fixture" TIMEOUT 120
        ENVIRONMENT
            "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/test-slang-cpu-resources.json")
endif()

add_executable(parallel-mater-fluid-pair-quality-tests
    fluid_pair_quality_tests.cpp)
target_compile_features(parallel-mater-fluid-pair-quality-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-fluid-pair-quality-tests PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../conformance/v1/include")
target_compile_definitions(parallel-mater-fluid-pair-quality-tests PRIVATE
    PM_QUALITY_PROFILE_SHA256="a59a81c71b5ddf9ae1f0a47cd8e61e2074c24d929b15318852863b2c92cd9326")
target_link_libraries(parallel-mater-fluid-pair-quality-tests
    PRIVATE ParallelMater::parallel_mater)
foreach(backend cpu cuda vulkan d3d12 metal)
    if(TARGET parallel-mater-driver-slang-${backend})
        add_dependencies(parallel-mater-fluid-pair-quality-tests
            parallel-mater-driver-slang-${backend})
        add_test(NAME parallel-mater-${backend}-fluid-pair-quality-tests
            COMMAND parallel-mater-fluid-pair-quality-tests
                --output "${CMAKE_CURRENT_BINARY_DIR}/${backend}-fluid-pair-quality.json")
        set_tests_properties(parallel-mater-${backend}-fluid-pair-quality-tests
            PROPERTIES LABELS "api;quality;${backend}" SKIP_RETURN_CODE 77
            TIMEOUT 180
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-${backend}.json")
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

add_executable(parallel-mater-destruction-wait-tests
    destruction_wait_tests.cpp)
target_compile_features(parallel-mater-destruction-wait-tests PRIVATE
    cxx_std_20)
target_link_libraries(parallel-mater-destruction-wait-tests PRIVATE
    ParallelMater::parallel_mater)
foreach(backend cuda vulkan d3d12 metal)
    if(TARGET parallel-mater-driver-slang-${backend} AND
       TARGET parallel-mater-layer-validation)
        add_dependencies(parallel-mater-destruction-wait-tests
            parallel-mater-driver-slang-${backend}
            parallel-mater-layer-validation)
        add_test(NAME parallel-mater-${backend}-destruction-wait-tests
            COMMAND parallel-mater-destruction-wait-tests)
        set_tests_properties(
            parallel-mater-${backend}-destruction-wait-tests PROPERTIES
            LABELS "api;lifecycle;teardown;validation;${backend}"
            SKIP_RETURN_CODE 77 TIMEOUT 120
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-${backend}.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json;PM_TEST_VALIDATION=1")
    endif()
endforeach()

if(TARGET parallel-mater-test-driver-slang-cpu-resources AND
   TARGET parallel-mater-layer-validation)
    add_dependencies(parallel-mater-destruction-wait-tests
        parallel-mater-test-driver-slang-cpu-resources
        parallel-mater-layer-validation)
    add_test(NAME parallel-mater-host-destruction-wait-tests
        COMMAND parallel-mater-destruction-wait-tests --failure-fixture)
    set_tests_properties(parallel-mater-host-destruction-wait-tests PROPERTIES
        LABELS "api;lifecycle;teardown;validation;host-fixture" TIMEOUT 120
        ENVIRONMENT
            "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/test-slang-cpu-resources.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json;PM_TEST_VALIDATION=1")
endif()

add_executable(parallel-mater-capacity-recovery-tests
    capacity_recovery_tests.cpp)
target_compile_features(parallel-mater-capacity-recovery-tests PRIVATE
    cxx_std_20)
target_include_directories(parallel-mater-capacity-recovery-tests PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../conformance/v1/include")
target_compile_definitions(parallel-mater-capacity-recovery-tests PRIVATE
    PM_LIFECYCLE_PROFILE_PATH="${CMAKE_CURRENT_LIST_DIR}/../../conformance/v1/lifecycle/adverse-capacity-three-family-v1.json")
target_link_libraries(parallel-mater-capacity-recovery-tests PRIVATE
    ParallelMater::parallel_mater)
foreach(backend cuda vulkan d3d12 metal)
    if(TARGET parallel-mater-driver-slang-${backend} AND
       TARGET parallel-mater-layer-validation)
        add_dependencies(parallel-mater-capacity-recovery-tests
            parallel-mater-driver-slang-${backend}
            parallel-mater-layer-validation)
        add_test(NAME parallel-mater-${backend}-capacity-recovery-tests
            COMMAND parallel-mater-capacity-recovery-tests)
        set_tests_properties(
            parallel-mater-${backend}-capacity-recovery-tests PROPERTIES
            LABELS "api;lifecycle;capacity;validation;${backend}"
            SKIP_RETURN_CODE 77 TIMEOUT 120
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-${backend}.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json;PM_TEST_VALIDATION=1")
    endif()
endforeach()

if(PARALLEL_MATER_SLANG_CPU_LIQUIDS AND
   TARGET parallel-mater-test-driver-slang-cpu-resources AND
   TARGET parallel-mater-layer-validation)
    add_dependencies(parallel-mater-capacity-recovery-tests
        parallel-mater-test-driver-slang-cpu-resources
        parallel-mater-layer-validation)
    add_test(NAME parallel-mater-host-capacity-recovery-tests
        COMMAND parallel-mater-capacity-recovery-tests)
    set_tests_properties(parallel-mater-host-capacity-recovery-tests PROPERTIES
        LABELS "api;lifecycle;capacity;validation;host-fixture" TIMEOUT 120
        ENVIRONMENT
            "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/test-slang-cpu-resources.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json;PM_TEST_VALIDATION=1")
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
