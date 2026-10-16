// SPDX-License-Identifier: MIT
#pragma once

#include "renderer_shared.hpp"

#include <parallel_mater_scene/types.hpp>

#include <memory>
#include <vector>

namespace parallel_mater::gallery {

// Example-only Zhu-Bridson particle surface; the physics API stays particle based.
class FluidSurface {
  public:
    explicit FluidSurface(std::uint32_t capacity);
    ~FluidSurface();
    FluidSurface(FluidSurface &&) noexcept;
    FluidSurface &operator=(FluidSurface &&) noexcept;
    FluidSurface(const FluidSurface &) = delete;
    FluidSurface &operator=(const FluidSurface &) = delete;

    [[nodiscard]] float update(const Vec3 *device_positions,
                               std::uint32_t particle_count,
                               float particle_radius,
                               float support_radius,
                               const std::vector<Vec3> &host_positions);
    [[nodiscard]] optix_shared::FluidSurfaceView view() const noexcept;
    [[nodiscard]] std::uint32_t excluded_particle_count() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::gallery
