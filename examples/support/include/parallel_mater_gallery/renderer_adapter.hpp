// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/gpu_renderer.hpp>
#include <parallel_mater_gallery/renderer.hpp>

namespace parallel_mater::gallery {

// The only construction boundary used by Gallery clients. Concrete renderer
// types stay in the adapter target selected by the build.
class RendererAdapterFactory final {
  public:
    [[nodiscard]] static bool native_gpu_available() noexcept;
    [[nodiscard]] static bool create_host(
        const SceneDefinition &scene, const CommandSceneInstance &instance,
        std::uint32_t width, std::uint32_t height, RendererHandle &output,
        std::string &error);
    [[nodiscard]] static std::unique_ptr<GpuGalleryRenderer> create_native(
        GpuRendererDevice device, const SceneDefinition &scene,
        const CommandSceneInstance &instance, std::string &error);
};

} // namespace parallel_mater::gallery
