// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/gpu_renderer.hpp>
#include <parallel_mater_gallery/renderer.hpp>

namespace parallel_mater::gallery {

[[nodiscard]] std::unique_ptr<Renderer> create_software_renderer(
    const SceneDefinition &scene, std::uint32_t width, std::uint32_t height,
    std::string &error);
[[nodiscard]] std::unique_ptr<Renderer> create_optix_renderer(
    const SceneDefinition &scene, const CommandSceneInstance &instance,
    std::uint32_t width, std::uint32_t height, std::string &error);
[[nodiscard]] std::unique_ptr<Renderer> create_d3d12_renderer(
    const SceneDefinition &scene, const GpuSceneBuffers &buffers,
    std::uint32_t width, std::uint32_t height, std::string &error);
[[nodiscard]] std::unique_ptr<GpuGalleryRenderer> create_vulkan_gallery_renderer(
    GpuRendererDevice device, const SceneDefinition &scene,
    const CommandSceneInstance &instance, std::string &error);
[[nodiscard]] std::unique_ptr<GpuGalleryRenderer> create_metal_gallery_renderer(
    GpuRendererDevice device, const SceneDefinition &scene,
    const CommandSceneInstance &instance, std::string &error);

} // namespace parallel_mater::gallery
