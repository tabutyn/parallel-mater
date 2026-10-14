// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

struct CommandSceneInstance;
struct RenderSceneFrame;

struct RendererTimings {
    float surface_gpu_milliseconds{};
    float raytrace_wall_milliseconds{};
    float foam_wall_milliseconds{};
    float total_wall_milliseconds{};
    std::uint32_t particle_count{};
    std::uint32_t smoke_particle_count{};
    std::uint32_t surface_excluded_particle_count{};
    std::uint32_t foam_patch_count{};
    bool particle_view{};
};

enum class FluidRenderMode : std::uint8_t {
    surface,
    particles,
    wireframe,
};

enum class SmokeDebugMode : std::uint8_t {
    none,
    grid,
    velocity,
    pressure,
    density_temperature,
    vorticity,
    divergence,
};

class GalleryRenderer {
  public:
    GalleryRenderer() noexcept;
    ~GalleryRenderer();
    GalleryRenderer(GalleryRenderer &&) noexcept;
    GalleryRenderer &operator=(GalleryRenderer &&) noexcept;
    GalleryRenderer(const GalleryRenderer &) = delete;
    GalleryRenderer &operator=(const GalleryRenderer &) = delete;

    [[nodiscard]] static bool create(const SceneDefinition &scene,
                                     const CommandSceneInstance &instance,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     GalleryRenderer &output,
                                     std::string &error);

    [[nodiscard]] bool render(const RenderSceneFrame &frame,
                              Camera camera,
                              std::vector<std::uint32_t> &rgba,
                              std::string &error,
                              RendererTimings *timings = nullptr,
                              FluidRenderMode fluid_mode =
                                  FluidRenderMode::surface,
                              bool show_smoke_particles = true,
                              float rigid_interpolation_alpha = 1.0F);

    // Advance render-only foam during unrendered headless steps.
    [[nodiscard]] bool advance_visuals(const RenderSceneFrame &frame,
                                       std::string &error);
    [[nodiscard]] bool paint_coverage(std::uint64_t &painted_texels,
                                      std::string &error) const;

    [[nodiscard]] std::uint32_t width() const noexcept;
    [[nodiscard]] std::uint32_t height() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::uint64_t gallery_device_available_memory() noexcept;
[[nodiscard]] HardwareIdentity gallery_device_hardware_identity();

} // namespace parallel_mater::gallery
