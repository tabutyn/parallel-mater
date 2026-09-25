// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

struct Camera {
    Vec3 eye{7.5F, 5.0F, 9.0F};
    Vec3 target{0.0F, 1.5F, 0.0F};
    Vec3 up{0.0F, 1.0F, 0.0F};
    float vertical_field_of_view_degrees{48.0F};
};

struct RendererTimings {
    float surface_gpu_milliseconds{};
    float raytrace_wall_milliseconds{};
    float foam_wall_milliseconds{};
    float total_wall_milliseconds{};
    std::uint32_t particle_count{};
    std::uint32_t surface_excluded_particle_count{};
    std::uint32_t foam_patch_count{};
    bool particle_view{};
};

enum class FluidRenderMode : std::uint8_t {
    surface,
    particles,
};

class OptixRenderer {
  public:
    OptixRenderer() noexcept;
    ~OptixRenderer();
    OptixRenderer(OptixRenderer &&) noexcept;
    OptixRenderer &operator=(OptixRenderer &&) noexcept;
    OptixRenderer(const OptixRenderer &) = delete;
    OptixRenderer &operator=(const OptixRenderer &) = delete;

    [[nodiscard]] static bool create(const SceneDefinition &scene,
                                     const World &world,
                                     const SceneInstance &instance,
                                     const std::filesystem::path &ptx_path,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     OptixRenderer &output,
                                     std::string &error);

    [[nodiscard]] bool render(const World &world,
                              const SceneInstance &instance,
                              Camera camera,
                              std::vector<std::uint32_t> &rgba,
                              std::string &error,
                              RendererTimings *timings = nullptr,
                              FluidRenderMode fluid_mode =
                                  FluidRenderMode::surface);

    // Advance render-only foam during unrendered headless steps.
    [[nodiscard]] bool advance_visuals(const World &world,
                                       const SceneInstance &instance,
                                       std::string &error);
    [[nodiscard]] bool paint_coverage(std::uint64_t &painted_texels,
                                      std::string &error) const;

    [[nodiscard]] std::uint32_t width() const noexcept;
    [[nodiscard]] std::uint32_t height() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::gallery
