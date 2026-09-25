// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/renderer.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace parallel_mater::gallery {

// Render-only foam shared by every fluid scene; never feeds back into physics.
class FoamVisuals {
  public:
    void advance(const std::vector<Vec3> &positions,
                 const std::vector<float> &foam,
                 const std::vector<std::uint32_t> &ids,
                 float particle_radius);

    void paint(float support_radius, Camera camera,
               std::uint32_t width, std::uint32_t height,
               const std::vector<float> &depth,
               const std::vector<float> &rigid_depth,
               std::vector<std::uint32_t> &rgba) const;

    [[nodiscard]] std::size_t patch_count() const noexcept {
        return patches_.size();
    }

  private:
    struct Patch {
        std::uint32_t particle_id{};
        Vec3 position{};
        float age{};
        float lifetime{};
        float radius{};
    };

    std::vector<Patch> patches_{};
    std::uint64_t tick_{};
};

} // namespace parallel_mater::gallery
