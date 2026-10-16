// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/renderer.hpp>

namespace parallel_mater::gallery {
[[nodiscard]] std::unique_ptr<Renderer> create_d3d12_renderer(
    const SceneDefinition &scene, const GpuSceneBuffers &buffers,
    std::uint32_t width, std::uint32_t height, std::string &error);
}
