# SPDX-License-Identifier: MIT
# Native macOS packaging for the backend-neutral Gallery targets created by
# the top-level build. Physics remains supplied only by loader-discovered
# drivers bundled under Resources/drivers.d.

find_library(PARALLEL_MATER_APPKIT_FRAMEWORK AppKit REQUIRED)

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
    ${PARALLEL_MATER_APPKIT_FRAMEWORK})
target_compile_options(parallel-mater-gallery PRIVATE
    -fobjc-arc -Wall -Wextra -Wpedantic)
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
    BUILD_WITH_INSTALL_RPATH YES
    INSTALL_RPATH "@executable_path/../Frameworks")

set(PARALLEL_MATER_BUNDLE_LAYER_MANIFEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/macos/validation.json")
file(GENERATE OUTPUT "${PARALLEL_MATER_BUNDLE_LAYER_MANIFEST}" CONTENT
"{\n  \"file_format_version\": \"1.0.0\",\n  \"layer\": {\n    \"name\": \"PM_LAYER_PARALLEL_MATER_validation\",\n    \"library_path\": \"../../Frameworks/$<TARGET_FILE_NAME:parallel-mater-layer-validation>\",\n    \"api_version\": \"0.3.0\"\n  }\n}\n")
add_custom_target(refresh-macos-gallery-runtime ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/drivers.d"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/layers.d"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:parallel-mater>"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_FILE_NAME:parallel-mater>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_SONAME_FILE:parallel-mater>"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_SONAME_FILE_NAME:parallel-mater>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_LINKER_FILE:parallel-mater>"
        "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_LINKER_FILE_NAME:parallel-mater>"
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
    DEPENDS parallel-mater-gallery parallel-mater parallel-mater-layer-validation
    VERBATIM)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GalleryBuildInfo.cmake")
if(TARGET parallel-mater-driver-slang-metal)
    set(PARALLEL_MATER_MACOS_GALLERY_BACKEND metal)
    parallel_mater_gallery_build_info(parallel-mater-gallery metal)
    add_dependencies(parallel-mater-gallery parallel-mater-driver-slang-metal)
    target_compile_definitions(parallel-mater-gallery-application PRIVATE PARALLEL_MATER_METAL_PREVIEW=1)
    add_dependencies(refresh-macos-gallery-runtime parallel-mater-driver-slang-metal)
    add_custom_command(TARGET refresh-macos-gallery-runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/drivers.d/slang-cpu.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:parallel-mater-driver-slang-metal>"
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_FILE_NAME:parallel-mater-driver-slang-metal>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "$<TARGET_FILE_DIR:parallel-mater-driver-slang-metal>/slang-metal"
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/slang-metal"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-metal.json"
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/drivers.d/slang-metal.json"
        VERBATIM)
elseif(TARGET parallel-mater-driver-slang-cpu)
    set(PARALLEL_MATER_MACOS_GALLERY_BACKEND slang-cpu)
    parallel_mater_gallery_build_info(parallel-mater-gallery slang-cpu)
    add_dependencies(refresh-macos-gallery-runtime parallel-mater-driver-slang-cpu)
    add_custom_command(TARGET refresh-macos-gallery-runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/drivers.d/slang-metal.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:parallel-mater-driver-slang-cpu>"
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Frameworks/$<TARGET_FILE_NAME:parallel-mater-driver-slang-cpu>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/portable/slang-cpu.json"
            "$<TARGET_BUNDLE_CONTENT_DIR:parallel-mater-gallery>/Resources/drivers.d/slang-cpu.json"
        VERBATIM)
endif()

# Runtime relocation and resource copies invalidate any earlier linker/Xcode
# signature. Seal only the finished bundle, so deployment can verify its input.
add_custom_command(TARGET refresh-macos-gallery-runtime POST_BUILD
    COMMAND /usr/bin/codesign --force --deep --sign -
        "$<TARGET_BUNDLE_DIR:parallel-mater-gallery>"
    COMMAND /usr/bin/codesign --verify --deep --strict
        "$<TARGET_BUNDLE_DIR:parallel-mater-gallery>"
    VERBATIM)

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
        -DEXPECTED_BACKEND=${PARALLEL_MATER_MACOS_GALLERY_BACKEND}
        -DUPDATE_DOCK=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeployMacOSBundle.cmake
    DEPENDS refresh-macos-gallery-runtime
    COMMENT "Refreshing the installed ParallelMater Gallery"
    USES_TERMINAL
    VERBATIM)

if(BUILD_TESTING)
    add_test(NAME parallel-mater-macos-gallery-signature
        COMMAND /usr/bin/codesign --verify --deep --strict
            "$<TARGET_BUNDLE_DIR:parallel-mater-gallery>")
    if(TARGET parallel-mater-driver-slang-metal)
        set(PM_PREVIEW_REFERENCE_ARGS)
        if(TARGET parallel-mater-driver-slang-cpu)
            list(APPEND PM_PREVIEW_REFERENCE_ARGS "${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-cpu.json")
        endif()
        add_test(NAME parallel-mater-slang-metal-gallery-preview
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/metal_preview_smoke.py"
                $<TARGET_FILE:parallel-mater-gallery> ${PM_PREVIEW_REFERENCE_ARGS})
        set_tests_properties(parallel-mater-slang-metal-gallery-preview PROPERTIES
            LABELS "gallery;metal;gpu;preview" TIMEOUT 900)
        add_test(NAME parallel-mater-slang-metal-gallery-domains
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery/metal_domain_smoke.py"
                $<TARGET_FILE:parallel-mater-gallery>)
        set_tests_properties(parallel-mater-slang-metal-gallery-domains PROPERTIES
            LABELS "gallery;metal;gpu;preview" TIMEOUT 1200)
        if(TARGET parallel-mater-conformance AND TARGET parallel-mater-layer-validation)
            set(PM_METAL_RIGID_QUALITY_DIRECTORY
                "${CMAKE_CURRENT_BINARY_DIR}/metal-rigid-brick-quality")
            file(MAKE_DIRECTORY "${PM_METAL_RIGID_QUALITY_DIRECTORY}")
            add_test(NAME parallel-mater-slang-metal-rigid-brick-quality-execution
                COMMAND "${CMAKE_COMMAND}" -E env
                    "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-metal.json"
                    "PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json"
                    $<TARGET_FILE:parallel-mater-conformance>
                    --quality rigid-brick-penetration-v1
                    --output "${PM_METAL_RIGID_QUALITY_DIRECTORY}")
            add_test(NAME parallel-mater-slang-metal-rigid-brick-quality
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/quality.py"
                    --profile "${CMAKE_CURRENT_SOURCE_DIR}/conformance/v1/quality/rigid-brick-penetration-v1.json"
                    --actual "${PM_METAL_RIGID_QUALITY_DIRECTORY}/rigid-brick-penetration-v1.json"
                    --report "${PM_METAL_RIGID_QUALITY_DIRECTORY}/report.json")
            set_tests_properties(
                parallel-mater-slang-metal-rigid-brick-quality-execution
                parallel-mater-slang-metal-rigid-brick-quality PROPERTIES
                LABELS "conformance;metal;gpu;quality" TIMEOUT 300)
            set_tests_properties(parallel-mater-slang-metal-rigid-brick-quality
                PROPERTIES DEPENDS
                    parallel-mater-slang-metal-rigid-brick-quality-execution)
        endif()
    endif()
    add_test(NAME parallel-mater-macos-gallery-help
             COMMAND $<TARGET_FILE:parallel-mater-gallery> --help)
    set_tests_properties(parallel-mater-macos-gallery-help PROPERTIES
        LABELS "gallery;macos;bundle")
    if(TARGET parallel-mater-driver-slang-cpu OR TARGET parallel-mater-driver-slang-metal)
        add_test(NAME parallel-mater-macos-slang-rigid-smoke
            COMMAND $<TARGET_FILE:parallel-mater-gallery>
                --constraint-point --frames 3 --headless
                ${CMAKE_CURRENT_BINARY_DIR}/parallel-mater-slang-rigid-smoke.ppm)
        set_tests_properties(parallel-mater-macos-slang-rigid-smoke PROPERTIES
            LABELS "driver;gallery;macos;slang;smoke"
            TIMEOUT 60)
    endif()
endif()
