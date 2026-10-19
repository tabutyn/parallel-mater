# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(ParallelMaterConformance LANGUAGES CXX)
include(CTest)
find_package(ParallelMater 0.2 CONFIG REQUIRED COMPONENTS loader scene)
if(NOT TARGET ParallelMater::scene)
    message(FATAL_ERROR "Install the Parallel Mater scene SDK component before building conformance")
endif()
set(PARALLEL_MATER_CONFORMANCE_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE PATH "Root containing conformance/v1 and examples/assets (override at runtime with --root)")
set(PARALLEL_MATER_CONFORMANCE_REVISION "unknown" CACHE STRING "Conformance source revision")
add_library(parallel-mater-conformance-runner STATIC
    src/command_runner.cpp
    src/case_registry.cpp
    src/rigid_quality.cpp
    src/rope_quality.cpp
    src/fluid_cloth_containment_quality.cpp
    src/fluid_soft_contact_quality.cpp
    src/fluid_rope_contact_quality.cpp
    src/cloth_soft_contact.cpp
    src/fluid_rigid_containment.cpp
    src/fluid_rigid_contact.cpp
    src/cloth_rope_attachment.cpp
    src/smoke_quality.cpp
    src/fluid_cloth_contact.cpp
    src/smoke_fluid_drag.cpp
    src/fluid_smoke_phase.cpp
    src/smoke_viscosity_quality.cpp
    src/smoke_lifecycle_fixture.cpp
    src/lifecycle_fixtures.cpp
    src/fluid_cloth_containment_lifecycle.cpp
    src/smoke_lifecycle.cpp
    src/capacity_recovery_fixture.cpp
    src/interaction_graph_lifecycle_fixture.cpp
    src/resource_lifecycle_fixture.cpp)
target_compile_features(parallel-mater-conformance-runner PUBLIC cxx_std_20)
target_include_directories(parallel-mater-conformance-runner
    PUBLIC include)
target_compile_definitions(parallel-mater-conformance-runner PRIVATE
    PM_CONFORMANCE_EMBEDDED=1
    PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${PARALLEL_MATER_CONFORMANCE_DATA_ROOT}"
    PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CONFORMANCE_REVISION}")
target_link_libraries(parallel-mater-conformance-runner PRIVATE ParallelMater::scene)
add_executable(parallel-mater-conformance src/command_main.cpp)
target_compile_features(parallel-mater-conformance PRIVATE cxx_std_20)
target_link_libraries(parallel-mater-conformance PRIVATE
    parallel-mater-conformance-runner)
if(BUILD_TESTING)
    add_test(NAME parallel-mater-conformance-command-inputs
        COMMAND parallel-mater-conformance --check-inputs)
    add_test(NAME parallel-mater-conformance-lifecycle-output-required
        COMMAND parallel-mater-conformance --lifecycle)
    set_tests_properties(
        parallel-mater-conformance-lifecycle-output-required PROPERTIES
        WILL_FAIL TRUE)
endif()
install(TARGETS parallel-mater-conformance RUNTIME DESTINATION bin)
