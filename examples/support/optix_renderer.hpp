// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/renderer.hpp>

#include <filesystem>

namespace parallel_mater::gallery {

class OptixRenderer final : public Renderer {
  public:
    OptixRenderer() noexcept;
    ~OptixRenderer() override;
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
    [[nodiscard]] bool render(
        const RenderSceneFrame &frame, Camera camera,
        std::vector<std::uint32_t> &rgba, std::string &error,
        RendererTimings *timings = nullptr,
        FluidRenderMode fluid_mode = FluidRenderMode::surface,
        bool show_smoke_particles = true,
        float rigid_interpolation_alpha = 1.0F) override;
    [[nodiscard]] bool advance_visuals(
        const RenderSceneFrame &frame, std::string &error) override;
    [[nodiscard]] bool paint_coverage(
        std::uint64_t &painted_texels, std::string &error) const override;
    [[nodiscard]] std::uint32_t width() const noexcept override;
    [[nodiscard]] std::uint32_t height() const noexcept override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::gallery
