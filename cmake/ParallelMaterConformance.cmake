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

if(APPLE)
    set_target_properties(parallel-mater-conformance PROPERTIES
        INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_target_properties(parallel-mater-conformance PROPERTIES
        INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set(corpus_install_root "${CMAKE_INSTALL_DATADIR}/parallel-mater")
install(DIRECTORY conformance/v1/cases conformance/v1/quality
    DESTINATION "${corpus_install_root}/conformance/v1")
install(FILES conformance/v1/manifest.json conformance/v1/compare.py
    conformance/v1/quality.py
    conformance/v1/README.md conformance/v1/SCHEMA.md
    DESTINATION "${corpus_install_root}/conformance/v1")
# Install only the immutable assets named by this corpus manifest.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/manifest.json" corpus_manifest)
string(JSON corpus_asset_count LENGTH "${corpus_manifest}" glb_sha256)
math(EXPR corpus_last_asset "${corpus_asset_count} - 1")
foreach(index RANGE 0 ${corpus_last_asset})
    string(JSON asset MEMBER "${corpus_manifest}" glb_sha256 ${index})
    get_filename_component(asset_directory "${asset}" DIRECTORY)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${asset}"
        DESTINATION "${corpus_install_root}/${asset_directory}")
endforeach()
