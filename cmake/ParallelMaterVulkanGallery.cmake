# SPDX-License-Identifier: MIT
set(pm_vulkan_gallery_shader_dir
    "${CMAKE_CURRENT_BINARY_DIR}/generated/gallery-vulkan")
file(MAKE_DIRECTORY "${pm_vulkan_gallery_shader_dir}")

set(pm_vulkan_gallery_shaders)
foreach(shader vertex fragment overlay_vertex overlay_fragment)
    if(shader STREQUAL "vertex")
        set(stage vertex)
        set(entry vertexMain)
    elseif(shader STREQUAL "fragment")
        set(stage fragment)
        set(entry fragmentMain)
    elseif(shader STREQUAL "overlay_vertex")
        set(stage vertex)
        set(entry overlayVertex)
    else()
        set(stage fragment)
        set(entry overlayFragment)
    endif()
    set(output "${pm_vulkan_gallery_shader_dir}/${shader}.spv")
    add_custom_command(OUTPUT "${output}"
        COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/render/gallery.slang"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/render"
            -target spirv -profile glsl_460 -stage ${stage} -entry ${entry}
            -DPM_GALLERY_VULKAN=1
            -fvk-use-entrypoint-name -fvk-use-scalar-layout -O1
            -o "${output}"
        DEPENDS examples/gallery/render/gallery.slang
            examples/gallery/render/gallery_uniforms.slangh
            "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
        COMMENT "Compiling Gallery ${entry} to SPIR-V"
        VERBATIM)
    list(APPEND pm_vulkan_gallery_shaders "${output}")
endforeach()

set(pm_vulkan_gallery_header
    "${pm_vulkan_gallery_shader_dir}/vulkan_gallery_shaders.hpp")
add_custom_command(OUTPUT "${pm_vulkan_gallery_header}"
    COMMAND "${CMAKE_COMMAND}"
        -DSHADER_DIR=${pm_vulkan_gallery_shader_dir}
        -DOUTPUT_FILE=${pm_vulkan_gallery_header}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedVulkanGalleryShaders.cmake"
    DEPENDS ${pm_vulkan_gallery_shaders}
        cmake/EmbedVulkanGalleryShaders.cmake
    COMMENT "Embedding Gallery Vulkan shaders"
    VERBATIM)

set(pm_vulkan_gallery_renderer_source
    examples/support/vulkan_gallery_renderer.cpp)
# Vulkan create-info aggregate initialization deliberately supplies sType and
# zero-initializes every remaining member before setting the required fields.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(${pm_vulkan_gallery_renderer_source} PROPERTIES
        COMPILE_OPTIONS -Wno-missing-field-initializers)
endif()
target_sources(parallel-mater-gallery-renderer-adapters PRIVATE
    ${pm_vulkan_gallery_renderer_source}
    "${pm_vulkan_gallery_header}")
target_include_directories(parallel-mater-gallery-renderer-adapters PRIVATE
    "${pm_vulkan_gallery_shader_dir}")
target_compile_definitions(parallel-mater-gallery-renderer-adapters PRIVATE
    PM_GALLERY_NATIVE_RENDERER_VULKAN=1 PM_GALLERY_VULKAN_RENDERER=1)
target_link_libraries(parallel-mater-gallery-renderer-adapters PRIVATE Vulkan::Vulkan)
if(TARGET parallel-mater-gallery-window)
    target_compile_definitions(parallel-mater-gallery-window PRIVATE
        PM_GALLERY_VULKAN_WINDOW=1)
    target_link_libraries(parallel-mater-gallery-window PRIVATE Vulkan::Vulkan)
endif()

if(BUILD_TESTING)
    add_executable(parallel-mater-vulkan-gallery-renderer-tests
        tests/gallery/vulkan_gallery_renderer_tests.cpp)
    target_compile_features(parallel-mater-vulkan-gallery-renderer-tests
        PRIVATE cxx_std_20)
    target_compile_definitions(parallel-mater-vulkan-gallery-renderer-tests PRIVATE
        PM_GALLERY_TEST_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(parallel-mater-vulkan-gallery-renderer-tests PRIVATE
        parallel-mater-gallery-session)
    add_dependencies(parallel-mater-vulkan-gallery-renderer-tests
        parallel-mater-driver-slang-vulkan parallel-mater-layer-validation)
    add_test(NAME parallel-mater-vulkan-gallery-renderer-tests
        COMMAND parallel-mater-vulkan-gallery-renderer-tests)
    set_tests_properties(parallel-mater-vulkan-gallery-renderer-tests PROPERTIES
        LABELS "gallery;renderer;vulkan"
        SKIP_RETURN_CODE 77 TIMEOUT 240
        FAIL_REGULAR_EXPRESSION "Validation Error;VUID-"
        ENVIRONMENT
            "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json;PM_VULKAN_VALIDATION=1")

    if(TARGET parallel-mater-gallery-window)
        add_executable(parallel-mater-vulkan-gallery-surface-tests
            tests/gallery/vulkan_gallery_surface_tests.cpp)
        target_compile_features(parallel-mater-vulkan-gallery-surface-tests
            PRIVATE cxx_std_20)
        target_compile_definitions(parallel-mater-vulkan-gallery-surface-tests
            PRIVATE PM_GALLERY_TEST_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(parallel-mater-vulkan-gallery-surface-tests PRIVATE
            parallel-mater-gallery-session parallel-mater-gallery-window)
        add_dependencies(parallel-mater-vulkan-gallery-surface-tests
            parallel-mater-driver-slang-vulkan parallel-mater-layer-validation)
        add_test(NAME parallel-mater-vulkan-gallery-surface-tests
            COMMAND parallel-mater-vulkan-gallery-surface-tests)
        set_tests_properties(parallel-mater-vulkan-gallery-surface-tests PROPERTIES
            LABELS "gallery;renderer;surface;vulkan"
            SKIP_RETURN_CODE 77 TIMEOUT 240
            FAIL_REGULAR_EXPRESSION "Validation Error;VUID-"
            ENVIRONMENT
                "PM_DRIVER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/slang-vulkan.json;PM_LAYER_FILES=${PARALLEL_MATER_BUILD_MANIFEST_DIR}/validation.json;PM_VULKAN_VALIDATION=1")
    endif()
endif()
