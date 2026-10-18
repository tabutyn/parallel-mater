// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/gpu_frame.hpp>

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

class Renderer {
  public:
    virtual ~Renderer() = default;
    [[nodiscard]] virtual NativeRenderImage native_image() const noexcept { return {}; }
    // Explicit capture only; interactive presentation must not read back pixels.
    [[nodiscard]] virtual bool read_image(std::vector<std::uint32_t> &, std::string &) { return true; }
    [[nodiscard]] virtual bool render(
        const RenderSceneFrame &frame, Camera camera,
        std::vector<std::uint32_t> &rgba, std::string &error,
        RendererTimings *timings, FluidRenderMode fluid_mode,
        bool show_smoke_particles, float rigid_interpolation_alpha) = 0;
    [[nodiscard]] virtual bool advance_visuals(
        const RenderSceneFrame &frame, std::string &error) = 0;
    [[nodiscard]] virtual bool paint_coverage(
        std::uint64_t &painted_texels, std::string &error) const = 0;
    [[nodiscard]] virtual std::uint32_t width() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t height() const noexcept = 0;
};

class RendererHandle {
  public:
    RendererHandle() noexcept = default;
    explicit RendererHandle(std::unique_ptr<Renderer> implementation) noexcept
        : implementation_(std::move(implementation)) {}
    ~RendererHandle() = default;
    RendererHandle(RendererHandle &&) noexcept = default;
    RendererHandle &operator=(RendererHandle &&) noexcept = default;
    RendererHandle(const RendererHandle &) = delete;
    RendererHandle &operator=(const RendererHandle &) = delete;
    [[nodiscard]] NativeRenderImage native_image() const noexcept {
        return implementation_ ? implementation_->native_image() : NativeRenderImage{};
    }
    [[nodiscard]] bool read_image(std::vector<std::uint32_t> &rgba, std::string &error) {
        return implementation_ && implementation_->read_image(rgba, error);
    }

    [[nodiscard]] bool render(
        const RenderSceneFrame &frame, Camera camera,
        std::vector<std::uint32_t> &rgba, std::string &error,
        RendererTimings *timings = nullptr,
        FluidRenderMode fluid_mode = FluidRenderMode::surface,
        bool show_smoke_particles = true,
        float rigid_interpolation_alpha = 1.0F) {
        if (!implementation_) {
            error = "renderer is not initialized";
            return false;
        }
        return implementation_->render(frame, camera, rgba, error, timings,
                                       fluid_mode, show_smoke_particles,
                                       rigid_interpolation_alpha);
    }
    [[nodiscard]] bool advance_visuals(const RenderSceneFrame &frame,
                                       std::string &error) {
        if (!implementation_) {
            error = "renderer is not initialized";
            return false;
        }
        return implementation_->advance_visuals(frame, error);
    }
    [[nodiscard]] bool paint_coverage(std::uint64_t &painted_texels,
                                      std::string &error) const {
        if (!implementation_) {
            error = "renderer is not initialized";
            return false;
        }
        return implementation_->paint_coverage(painted_texels, error);
    }
    [[nodiscard]] std::uint32_t width() const noexcept {
        return implementation_ ? implementation_->width() : 0U;
    }
    [[nodiscard]] std::uint32_t height() const noexcept {
        return implementation_ ? implementation_->height() : 0U;
    }

  private:
    std::unique_ptr<Renderer> implementation_;
};

} // namespace parallel_mater::gallery
