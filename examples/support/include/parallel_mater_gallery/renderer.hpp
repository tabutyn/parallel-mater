// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <cstdint>
#include <filesystem>
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

#if !defined(PARALLEL_MATER_GALLERY_METAL) && \
    !defined(PARALLEL_MATER_GALLERY_D3D12)
class OptixRenderer {
  public:
    OptixRenderer() noexcept;
    ~OptixRenderer();
    OptixRenderer(OptixRenderer &&) noexcept;
    OptixRenderer &operator=(OptixRenderer &&) noexcept;
    OptixRenderer(const OptixRenderer &) = delete;
    OptixRenderer &operator=(const OptixRenderer &) = delete;

    [[nodiscard]] static bool create(const SceneDefinition &scene,
                                     const CommandSceneInstance &instance,
                                     const std::filesystem::path &ptx_path,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     OptixRenderer &output,
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
#endif

} // namespace parallel_mater::gallery
