# SPDX-License-Identifier: MIT
set(pm_gallery_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/gallery")
file(MAKE_DIRECTORY "${pm_gallery_shader_dir}")
foreach(stage vertex fragment)
    add_custom_command(OUTPUT "${pm_gallery_shader_dir}/${stage}.dxil"
        COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/render/gallery.slang"
            -target dxil -profile sm_6_6 -stage ${stage} -entry ${stage}Main
            -o "${pm_gallery_shader_dir}/${stage}.dxil"
        DEPENDS examples/gallery/render/gallery.slang examples/gallery/render/gallery_uniforms.slangh
        VERBATIM)
endforeach()
set(pm_surface_shaders surfaceClear surfaceHash surfaceBounds surfaceField surfaceMesh surfaceFinalize)
set(pm_surface_dxil)
foreach(entry IN LISTS pm_surface_shaders)
    set(shader "${pm_gallery_shader_dir}/${entry}.dxil")
    add_custom_command(OUTPUT "${shader}"
        COMMAND "${PARALLEL_MATER_SLANGC_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/render/fluid_surface_graphics.slang"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/examples/gallery/render"
            -target dxil -profile sm_6_6 -stage compute -entry ${entry} -O1
            -o "${shader}"
        DEPENDS examples/gallery/render/fluid_surface_graphics.slang
            examples/gallery/render/fluid_surface_field.slang examples/gallery/render/gallery_uniforms.slangh
        VERBATIM)
    list(APPEND pm_surface_dxil "${shader}")
endforeach()
add_custom_command(OUTPUT "${pm_gallery_shader_dir}/gallery_shaders.hpp"
    COMMAND "${CMAKE_COMMAND}" -DSHADER_DIR=${pm_gallery_shader_dir}
        -DOUTPUT_FILE=${pm_gallery_shader_dir}/gallery_shaders.hpp
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedGalleryShaders.cmake"
    DEPENDS "${pm_gallery_shader_dir}/vertex.dxil" "${pm_gallery_shader_dir}/fragment.dxil" ${pm_surface_dxil}
        cmake/EmbedGalleryShaders.cmake VERBATIM)
target_sources(parallel-mater-gallery-renderer PRIVATE
    examples/support/d3d12_renderer.cpp "${pm_gallery_shader_dir}/gallery_shaders.hpp")
target_include_directories(parallel-mater-gallery-renderer PRIVATE "${pm_gallery_shader_dir}")
target_compile_definitions(parallel-mater-gallery-renderer PRIVATE
    PARALLEL_MATER_GALLERY_D3D12_RENDERER=1 NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(parallel-mater-gallery-renderer PRIVATE d3d12 dxgi)
