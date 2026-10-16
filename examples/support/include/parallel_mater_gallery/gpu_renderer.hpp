// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/camera.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater/parallel_mater.h>
#include <memory>
#include <span>

namespace parallel_mater::gallery {
struct CommandSceneInstance;
struct CommandSceneGpuFrame;
class FrameTimingCollector;
// Shared Apple renderer: graphics only, no physics equations or platform input.
// A null layer renders offscreen. Pixel readback happens only when requested.
class GpuGalleryRenderer {
public:
    GpuGalleryRenderer(PmDevice device, const SceneDefinition &, const CommandSceneInstance &);
    ~GpuGalleryRenderer();
    void render(const CommandSceneGpuFrame &, Camera, unsigned width, unsigned height,
                void *metal_layer = nullptr, std::vector<std::uint32_t> *capture = nullptr,
                FrameTimingCollector *timing = nullptr,
                std::span<const std::uint32_t> overlay = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
