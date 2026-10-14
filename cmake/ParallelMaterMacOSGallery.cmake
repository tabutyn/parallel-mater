# SPDX-License-Identifier: MIT
# Loader-native macOS Gallery. Physics is supplied only by discovered Parallel
# Mater drivers; this target contains no backend-specific simulation runtime.

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 QUIET)
if(NOT TARGET glfw)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        parallel_mater_glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(parallel_mater_glfw)
endif()
find_library(PARALLEL_MATER_APPKIT_FRAMEWORK AppKit REQUIRED)

add_library(parallel-mater-gallery-scene STATIC
    examples/gallery/scene_builder.cpp
    examples/gallery/render/render_scene.cpp
    examples/support/scene.cpp
    examples/support/brick_scene.cpp
    examples/support/device_profiles.cpp
    examples/support/rope_render.cpp
    examples/support/surface_query.cpp)
target_compile_features(parallel-mater-gallery-scene PUBLIC cxx_std_20)
target_include_directories(parallel-mater-gallery-scene
    PUBLIC examples/support/include examples/gallery
    PRIVATE src ${parallel_mater_cgltf_SOURCE_DIR})
target_link_libraries(parallel-mater-gallery-scene
    PUBLIC ParallelMater::parallel_mater
    PRIVATE parallel-mater-host-preparation)
parallel_mater_enable_warnings(parallel-mater-gallery-scene)

add_library(parallel-mater-gallery-renderer STATIC
    examples/support/software_renderer.cpp
    examples/support/overlay.cpp
    examples/support/physics_debug.cpp)
target_compile_features(parallel-mater-gallery-renderer PUBLIC cxx_std_20)
target_include_directories(parallel-mater-gallery-renderer
    PUBLIC examples/support/include
    PRIVATE examples/support examples/gallery)
target_link_libraries(parallel-mater-gallery-renderer
    PUBLIC parallel-mater-gallery-scene)
parallel_mater_enable_warnings(parallel-mater-gallery-renderer)

add_library(parallel-mater-gallery-window STATIC
    examples/support/opengl_window.cpp)
target_compile_features(parallel-mater-gallery-window PUBLIC cxx_std_20)
target_include_directories(parallel-mater-gallery-window
    PUBLIC examples/support/include)
target_link_libraries(parallel-mater-gallery-window PRIVATE glfw OpenGL::GL)
parallel_mater_enable_warnings(parallel-mater-gallery-window)

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE PARALLEL_MATER_GALLERY_SOURCE_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT PARALLEL_MATER_GALLERY_SOURCE_COMMIT)
    set(PARALLEL_MATER_GALLERY_SOURCE_COMMIT unknown)
endif()

add_library(parallel-mater-gallery-application STATIC
    examples/gallery/application.cpp
    examples/support/loader_runtime_platform.cpp)
target_compile_features(parallel-mater-gallery-application PUBLIC cxx_std_20)
target_link_libraries(parallel-mater-gallery-application
    PUBLIC parallel-mater-gallery-renderer
    PRIVATE glfw parallel-mater-gallery-window)

set(PARALLEL_MATER_GALLERY_PROFILE_CATALOG
    "${CMAKE_CURRENT_BINARY_DIR}/config/device-profiles.json")
configure_file(config/device-profiles.json
               "${PARALLEL_MATER_GALLERY_PROFILE_CATALOG}" COPYONLY)
target_compile_definitions(parallel-mater-gallery-application PRIVATE
    PARALLEL_MATER_DEFAULT_SCENE_PATH="../Resources/assets/RigidBody.glb"
    PARALLEL_MATER_DEVICE_PROFILES_PATH="../Resources/config/device-profiles.json"
    PARALLEL_MATER_GALLERY_GIT_COMMIT="${PARALLEL_MATER_GALLERY_SOURCE_COMMIT}"
    PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH="../Resources/assets/ConstraintFixed.glb"
    PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH="../Resources/assets/Celestial.glb"
    PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH="../Resources/assets/ConstraintHinge.glb"
    PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH="../Resources/assets/ConstraintPiston.glb"
    PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH="../Resources/assets/ConstraintGeneric.glb"
    PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH="../Resources/assets/ConstraintMotorSpring.glb"
    PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH="../Resources/assets/DumpTruck.glb"
    PARALLEL_MATER_FLUID_SCENE_PATH="../Resources/assets/Fluid.glb"
    PARALLEL_MATER_FLUID_RIGID_SCENE_PATH="../Resources/assets/FluidRigid.glb"
    PARALLEL_MATER_PEGS_SCENE_PATH="../Resources/assets/Pegs.glb"
    PARALLEL_MATER_CLOTH_SCENE_PATH="../Resources/assets/Cloth.glb"
    PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH="../Resources/assets/ClothTear.glb"
    PARALLEL_MATER_CLOTH_PAINT_SCENE_PATH="../Resources/assets/ClothPaint.glb"
    PARALLEL_MATER_CLOTH_WATER_SCENE_PATH="../Resources/assets/ClothWater.glb"
    PARALLEL_MATER_SOFT_BODY_SCENE_PATH="../Resources/assets/Softbody.glb"
    PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH="../Resources/assets/SoftbodyRigidBody.glb"
    PARALLEL_MATER_SOFT_BODY_CLOTH_SCENE_PATH="../Resources/assets/SoftbodyCloth.glb"
    PARALLEL_MATER_SOFT_BODY_FLUID_SCENE_PATH="../Resources/assets/SoftbodyFluid.glb"
    PARALLEL_MATER_ROPE_SCENE_PATH="../Resources/assets/Rope.glb"
    PARALLEL_MATER_ROPE_FLUID_SCENE_PATH="../Resources/assets/RopeFluid.glb"
    PARALLEL_MATER_ROPE_SOFT_BODY_SCENE_PATH="../Resources/assets/RopeSoftbody.glb"
    PARALLEL_MATER_ROPE_CLOTH_SCENE_PATH="../Resources/assets/RopeCloth.glb"
    PARALLEL_MATER_SMOKE_SCENE_PATH="../Resources/assets/Smoke.glb"
    PARALLEL_MATER_SMOKE_WATER_SCENE_PATH="../Resources/assets/SmokeWater.glb"
    PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH="../Resources/assets/SmokeSoftbody.glb"
    PARALLEL_MATER_SMOKE_CLOTH_SCENE_PATH="../Resources/assets/SmokeCloth.glb"
    PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH="../Resources/assets/SmokeRope.glb")

set(PARALLEL_MATER_GALLERY_ICON
    "${CMAKE_CURRENT_BINARY_DIR}/generated/macos/ParallelMater.icns")
add_custom_command(
    OUTPUT "${PARALLEL_MATER_GALLERY_ICON}"
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_PNG=${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/branding/logo.png
        -DOUTPUT_ICNS=${PARALLEL_MATER_GALLERY_ICON}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateMacOSIcon.cmake
    DEPENDS examples/assets/branding/logo.png cmake/GenerateMacOSIcon.cmake
    COMMENT "Generating the ParallelMater macOS icon"
    VERBATIM)

add_executable(parallel-mater-gallery MACOSX_BUNDLE
    examples/gallery/macos_main.mm
    "${PARALLEL_MATER_GALLERY_ICON}")
set_source_files_properties("${PARALLEL_MATER_GALLERY_ICON}"
    PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
file(GLOB PARALLEL_MATER_GALLERY_ASSETS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/*.glb")
target_sources(parallel-mater-gallery PRIVATE
    ${PARALLEL_MATER_GALLERY_ASSETS}
    ${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json)
set_source_files_properties(${PARALLEL_MATER_GALLERY_ASSETS}
    PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/assets")
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/device-profiles.json
    PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/config")
target_link_libraries(parallel-mater-gallery PRIVATE
    parallel-mater-gallery-application
    ${PARALLEL_MATER_APPKIT_FRAMEWORK})
target_compile_options(parallel-mater-gallery PRIVATE
    -fobjc-arc -Wall -Wextra -Wpedantic)
add_dependencies(parallel-mater-gallery parallel-mater-layer-validation)
if(TARGET parallel-mater-driver-slang-cpu)
    add_dependencies(parallel-mater-gallery parallel-mater-driver-slang-cpu)
endif()
set_target_properties(parallel-mater-gallery PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED YES
    OUTPUT_NAME "ParallelMater Gallery"
    MACOSX_BUNDLE_INFO_PLIST
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/Info.plist.in"
    MACOSX_BUNDLE_BUNDLE_NAME "ParallelMater Gallery"
    MACOSX_BUNDLE_ICON_FILE "ParallelMater.icns"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.tabutyn.parallel-mater.gallery"
    MACOSX_BUNDLE_BUNDLE_VERSION "1"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    BUILD_RPATH "@executable_path/../Frameworks"
    INSTALL_RPATH "@executable_path/../Frameworks")

set(PARALLEL_MATER_BUNDLE_LAYER_MANIFEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/macos/validation.json")
file(GENERATE OUTPUT "${PARALLEL_MATER_BUNDLE_LAYER_MANIFEST}" CONTENT
"{\n  \"file_format_version\": \"1.0.0\",\n  \"layer\": {\n    \"name\": \"PM_LAYER_KHRONOS_validation\",\n    \"library_path\": \"../../Frameworks/$<TARGET_FILE_NAME:parallel-mater-layer-validation>\",\n    \"api_version\": \"0.2.0\"\n  }\n}\n")
add_custom_command(TARGET parallel-mater-gallery POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/drivers.d"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/layers.d"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:parallel-mater>"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_FILE_NAME:parallel-mater>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:parallel-mater-layer-validation>"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_FILE_NAME:parallel-mater-layer-validation>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:glfw>"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_FILE_NAME:glfw>"
    COMMAND ${CMAKE_COMMAND}
        -DAPPLICATION_EXECUTABLE=$<TARGET_FILE:parallel-mater-gallery>
        -DGLFW_LIBRARY=$<TARGET_FILE:glfw>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleMacOSRuntime.cmake
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PARALLEL_MATER_BUNDLE_LAYER_MANIFEST}"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/layers.d/validation.json"
    VERBATIM)

if(TARGET parallel-mater-driver-slang-cpu)
    add_custom_command(TARGET parallel-mater-gallery POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:parallel-mater-driver-slang-cpu>"
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_FILE_NAME:parallel-mater-driver-slang-cpu>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-cpu.json"
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/drivers.d/slang-cpu.json"
        VERBATIM)
endif()

set(PARALLEL_MATER_MACOS_GALLERY_DEPLOY_DIRECTORY "/Applications" CACHE PATH
    "Directory containing the continuously refreshed macOS Gallery")
option(PARALLEL_MATER_MACOS_GALLERY_AUTO_DEPLOY
       "Atomically refresh and Dock-pin the macOS Gallery after builds" OFF)
if(PARALLEL_MATER_MACOS_GALLERY_AUTO_DEPLOY)
    set(PARALLEL_MATER_MACOS_DEPLOY_KIND ALL)
else()
    set(PARALLEL_MATER_MACOS_DEPLOY_KIND)
endif()
add_custom_target(deploy-macos-gallery ${PARALLEL_MATER_MACOS_DEPLOY_KIND}
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_APP=$<TARGET_BUNDLE_DIR:parallel-mater-gallery>
        -DDESTINATION_APP=${PARALLEL_MATER_MACOS_GALLERY_DEPLOY_DIRECTORY}/ParallelMater\ Gallery.app
        -DUPDATE_DOCK=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeployMacOSBundle.cmake
    DEPENDS parallel-mater-gallery
    COMMENT "Refreshing the installed ParallelMater Gallery"
    USES_TERMINAL
    VERBATIM)

if(BUILD_TESTING)
    add_executable(parallel-mater-software-renderer-tests
        tests/gallery/software_renderer_tests.cpp)
    target_compile_features(parallel-mater-software-renderer-tests PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-software-renderer-tests PRIVATE
        PARALLEL_MATER_TEST_SCENE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/Celestial.glb")
    target_link_libraries(parallel-mater-software-renderer-tests PRIVATE
        parallel-mater-gallery-renderer)
    parallel_mater_enable_warnings(parallel-mater-software-renderer-tests)
    add_test(NAME parallel-mater-software-renderer-tests
             COMMAND parallel-mater-software-renderer-tests)
    set_tests_properties(parallel-mater-software-renderer-tests PROPERTIES
        LABELS "gallery;macos;renderer")
    add_test(NAME parallel-mater-macos-gallery-help
             COMMAND $<TARGET_FILE:parallel-mater-gallery> --help)
    set_tests_properties(parallel-mater-macos-gallery-help PROPERTIES
        LABELS "gallery;macos;bundle")
    if(TARGET parallel-mater-driver-slang-cpu)
        add_test(NAME parallel-mater-macos-slang-rigid-smoke
            COMMAND $<TARGET_FILE:parallel-mater-gallery>
                --constraint-point --frames 3 --headless
                ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-slang-rigid-smoke.ppm)
        set_tests_properties(parallel-mater-macos-slang-rigid-smoke PROPERTIES
            LABELS "driver;gallery;macos;slang;smoke"
            TIMEOUT 60)
    endif()
endif()
