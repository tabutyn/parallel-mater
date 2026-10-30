# SPDX-License-Identifier: MIT
if(NOT PARALLEL_MATER_BUILD_CONFORMANCE AND
   NOT PARALLEL_MATER_ENABLE_IOS_QUALIFICATION AND
   NOT PARALLEL_MATER_ENABLE_ANDROID_QUALIFICATION)
    return()
endif()

add_library(parallel-mater-conformance-runner STATIC
    conformance/v1/src/command_runner.cpp
    conformance/v1/src/realtime_workload.cpp
    conformance/v1/src/rigid_quality.cpp
    conformance/v1/src/rope_quality.cpp
    conformance/v1/src/fluid_cloth_containment_quality.cpp
    conformance/v1/src/fluid_soft_contact_quality.cpp
    conformance/v1/src/fluid_rope_contact_quality.cpp
    conformance/v1/src/cloth_soft_contact.cpp
    conformance/v1/src/soft_rope_contact.cpp
    conformance/v1/src/cloth_rigid_contact.cpp
    conformance/v1/src/rope_rigid_contact.cpp
    conformance/v1/src/rope_rigid_attachment.cpp
    conformance/v1/src/fluid_rigid_containment.cpp
    conformance/v1/src/fluid_rigid_contact.cpp
    conformance/v1/src/smoke_rigid_contact.cpp
    conformance/v1/src/soft_rigid_contact.cpp
    conformance/v1/src/cloth_rope_attachment.cpp
    conformance/v1/src/soft_rope_attachment.cpp
    conformance/v1/src/smoke_quality.cpp
    conformance/v1/src/fluid_cloth_contact.cpp
    conformance/v1/src/cloth_smoke_contact.cpp
    conformance/v1/src/soft_smoke_contact.cpp
    conformance/v1/src/rope_smoke_contact.cpp
    conformance/v1/src/smoke_fluid_drag.cpp
    conformance/v1/src/smoke_cloth_drag.cpp
    conformance/v1/src/smoke_soft_drag.cpp
    conformance/v1/src/smoke_rope_drag.cpp
    conformance/v1/src/smoke_rigid_drag.cpp
    conformance/v1/src/fluid_smoke_phase.cpp
    conformance/v1/src/smoke_viscosity_quality.cpp
    conformance/v1/src/smoke_lifecycle_fixture.cpp
    conformance/v1/src/lifecycle_fixtures.cpp
    conformance/v1/src/fluid_cloth_containment_lifecycle.cpp
    conformance/v1/src/smoke_lifecycle.cpp
    conformance/v1/src/capacity_recovery_fixture.cpp
    conformance/v1/src/interaction_graph_lifecycle_fixture.cpp
    conformance/v1/src/resource_lifecycle_fixture.cpp)
target_link_libraries(parallel-mater-conformance-runner PRIVATE
    parallel-mater-conformance-registry ParallelMater::scene)
target_include_directories(parallel-mater-conformance-runner PUBLIC
    conformance/v1/include)
target_compile_features(parallel-mater-conformance-runner PUBLIC cxx_std_20)
target_compile_definitions(parallel-mater-conformance-runner PRIVATE
    PM_CONFORMANCE_EMBEDDED=1
    PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
if(PARALLEL_MATER_BUILD_CONFORMANCE OR
   PARALLEL_MATER_ENABLE_IOS_QUALIFICATION OR
   PARALLEL_MATER_ENABLE_ANDROID_QUALIFICATION)
    target_sources(parallel-mater-conformance-runner PRIVATE
        conformance/v1/src/device_fixtures.cpp
        tests/api/fluid_pair_quality_tests.cpp)
    target_compile_definitions(parallel-mater-conformance-runner PRIVATE
        PM_CONFORMANCE_DEVICE_FIXTURES=1
        PM_QUALITY_PROFILE_SHA256="bbde06a7321e9cd0cdb9f3e547fccdb96caa3cd479e0781c40bc81c0426698c2"
        PM_FLUID_SELF_CONTACT_PROFILE_SHA256="28b9bb8f1dbe1a9696418f93108549b65a369dd4b81f69bf714312251b8e3af2"
        PM_FLUID_SELF_PRESSURE_PROFILE_SHA256="42825a5db3568c7d204f88958718f0a2b512880de865e036f8d82ecae0da93be"
        PM_FLUID_SELF_VISCOSITY_PROFILE_SHA256="a41f204deac0b08a1e4741542533e17922b1af1dce38220ab05f63bfcba2e5d5")
    if(PARALLEL_MATER_ENABLE_ANDROID_QUALIFICATION)
        target_compile_definitions(parallel-mater-conformance-runner PRIVATE
            PM_CONFORMANCE_DRIVER_NAME="slang-vulkan")
    elseif(PARALLEL_MATER_ENABLE_IOS_QUALIFICATION)
        target_compile_definitions(parallel-mater-conformance-runner PRIVATE
            PM_CONFORMANCE_DRIVER_NAME="slang-metal")
    else()
        target_compile_definitions(parallel-mater-conformance-runner PRIVATE
            PM_CONFORMANCE_DRIVER_NAME="selected-loadable-driver")
    endif()
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
    add_test(NAME parallel-mater-conformance-lifecycle-output-required
        COMMAND parallel-mater-conformance --lifecycle)
    set_tests_properties(parallel-mater-conformance-command-inputs PROPERTIES
        LABELS "api;conformance")
    set_tests_properties(
        parallel-mater-conformance-lifecycle-output-required PROPERTIES
        LABELS "api;conformance;lifecycle" WILL_FAIL TRUE)
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
    conformance/v1/realtime
    DESTINATION "${corpus_install_root}/conformance/v1"
    COMPONENT conformance)
install(FILES conformance/v1/manifest.json conformance/v1/compare.py
    conformance/v1/lifecycle.py conformance/v1/quality.py
    conformance/v1/qualification.py
    conformance/v1/realtime_device_capture.py
    conformance/v1/realtime_device_suite.py
    conformance/v1/realtime_oracle.py
    conformance/v1/realtime_scenario_fixtures.py
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
