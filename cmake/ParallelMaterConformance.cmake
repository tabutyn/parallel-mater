# SPDX-License-Identifier: MIT
if(NOT PARALLEL_MATER_BUILD_CONFORMANCE)
    return()
endif()
add_executable(parallel-mater-conformance conformance/v1/src/command_runner.cpp)
target_link_libraries(parallel-mater-conformance PRIVATE
    parallel-mater-conformance-registry ParallelMater::scene)
target_compile_definitions(parallel-mater-conformance PRIVATE
    PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
parallel_mater_enable_warnings(parallel-mater-conformance)
if(BUILD_TESTING)
    add_test(NAME parallel-mater-conformance-command-inputs
        COMMAND parallel-mater-conformance --check-inputs)
    set_tests_properties(parallel-mater-conformance-command-inputs PROPERTIES
        LABELS "api;conformance")
endif()
install(TARGETS parallel-mater-conformance RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
