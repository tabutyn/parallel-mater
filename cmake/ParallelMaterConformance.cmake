# SPDX-License-Identifier: MIT
if(NOT PARALLEL_MATER_BUILD_CONFORMANCE AND
   NOT PARALLEL_MATER_ENABLE_IOS_QUALIFICATION)
    return()
endif()

add_library(parallel-mater-conformance-runner STATIC
    conformance/v1/src/command_runner.cpp
    conformance/v1/src/rigid_quality.cpp)
target_link_libraries(parallel-mater-conformance-runner PRIVATE
    parallel-mater-conformance-registry ParallelMater::scene)
target_include_directories(parallel-mater-conformance-runner PUBLIC
    conformance/v1/include)
target_include_directories(parallel-mater-conformance-runner PRIVATE
    examples/support/include)
target_compile_features(parallel-mater-conformance-runner PUBLIC cxx_std_20)
target_compile_definitions(parallel-mater-conformance-runner PRIVATE
    PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
if(PARALLEL_MATER_ENABLE_IOS_QUALIFICATION)
    target_sources(parallel-mater-conformance-runner PRIVATE
        conformance/v1/src/device_fixtures.cpp
        tests/api/capacity_recovery_tests.cpp
        tests/api/fluid_pair_quality_tests.cpp
        tests/api/interaction_graph_lifecycle_tests.cpp
        tests/api/resource_lifecycle_tests.cpp)
    target_compile_definitions(parallel-mater-conformance-runner PRIVATE
        PM_CONFORMANCE_EMBEDDED=1
        PM_QUALITY_PROFILE_SHA256="a59a81c71b5ddf9ae1f0a47cd8e61e2074c24d929b15318852863b2c92cd9326")
    target_include_directories(parallel-mater-conformance-runner PRIVATE
        tests/api)
endif()
parallel_mater_enable_warnings(parallel-mater-conformance-runner)

if(NOT PARALLEL_MATER_BUILD_CONFORMANCE)
    return()
endif()
add_executable(parallel-mater-conformance
    conformance/v1/src/command_main.cpp)
target_link_libraries(parallel-mater-conformance PRIVATE
    parallel-mater-conformance-runner)
parallel_mater_enable_warnings(parallel-mater-conformance)
if(BUILD_TESTING)
    add_test(NAME parallel-mater-conformance-command-inputs
        COMMAND parallel-mater-conformance --check-inputs)
    set_tests_properties(parallel-mater-conformance-command-inputs PROPERTIES
        LABELS "api;conformance")
endif()
install(TARGETS parallel-mater-conformance
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT conformance)

if(APPLE)
    set_target_properties(parallel-mater-conformance PROPERTIES
        INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_target_properties(parallel-mater-conformance PROPERTIES
        INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set(corpus_install_root "${CMAKE_INSTALL_DATADIR}/parallel-mater")
install(DIRECTORY conformance/v1/cases conformance/v1/lifecycle
    conformance/v1/quality conformance/v1/qualification conformance/v1/evidence
    DESTINATION "${corpus_install_root}/conformance/v1"
    COMPONENT conformance)
install(FILES conformance/v1/manifest.json conformance/v1/compare.py
    conformance/v1/lifecycle.py conformance/v1/quality.py
    conformance/v1/qualification.py
    conformance/v1/README.md conformance/v1/SCHEMA.md
    DESTINATION "${corpus_install_root}/conformance/v1"
    COMPONENT conformance)
# Install only the immutable assets named by this corpus manifest.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/manifest.json" corpus_manifest)
string(JSON corpus_asset_count LENGTH "${corpus_manifest}" glb_sha256)
math(EXPR corpus_last_asset "${corpus_asset_count} - 1")
foreach(index RANGE 0 ${corpus_last_asset})
    string(JSON asset MEMBER "${corpus_manifest}" glb_sha256 ${index})
    get_filename_component(asset_directory "${asset}" DIRECTORY)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${asset}"
        DESTINATION "${corpus_install_root}/${asset_directory}"
        COMPONENT conformance)
endforeach()
configure_file(cmake/ParallelMaterConformanceComponent.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConformanceComponent.cmake"
    @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ParallelMaterConformanceComponent.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ParallelMater
    COMPONENT conformance)
