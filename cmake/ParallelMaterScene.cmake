# SPDX-License-Identifier: MIT
# Optional host-side scene authoring and public-command client. No renderer,
# native graphics API, GPU toolchain, or private simulation header dependency.
if(NOT PARALLEL_MATER_BUILD_GALLERY AND NOT PARALLEL_MATER_BUILD_SCENE_SDK
   AND NOT PARALLEL_MATER_BUILD_CONFORMANCE)
    return()
endif()
add_library(parallel-mater-scene STATIC
    examples/gallery/scene_builder.cpp
    examples/gallery/scene_builder_create.cpp.inc
    examples/gallery/scene_builder_output.cpp.inc
    examples/support/scene.cpp
    examples/support/rope_render.cpp
    examples/support/device_profiles.cpp
    examples/support/liquid_workload.cpp
    examples/support/fixed_tick_scene.cpp
    examples/support/brick_scene.cpp)
add_library(ParallelMater::scene ALIAS parallel-mater-scene)
set_target_properties(parallel-mater-scene PROPERTIES EXPORT_NAME scene
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(parallel-mater-scene PUBLIC cxx_std_20)
target_include_directories(parallel-mater-scene
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE examples/support/include examples/gallery ${parallel_mater_cgltf_SOURCE_DIR})
target_link_libraries(parallel-mater-scene PUBLIC ParallelMater::parallel_mater
    ParallelMater::preparation)
parallel_mater_enable_warnings(parallel-mater-scene)
list(APPEND PARALLEL_MATER_SCENE_INSTALL_TARGETS parallel-mater-scene)
