// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

enum class SurfaceSelection : std::uint8_t { lowest, highest };

// CPU-side XZ lookup for authored static triangle colliders. Diagnostics and
// renderer tests share this instead of each rebuilding transformed triangles.
class StaticTriangleSurface {
  public:
    [[nodiscard]] static bool create(const SceneDefinition &scene,
                                     StaticTriangleSurface &output,
                                     std::string &error);

    [[nodiscard]] std::optional<float> height(
        float x, float z, SurfaceSelection selection,
        float *normal_y = nullptr) const noexcept;
    [[nodiscard]] Vec3 minimum() const noexcept { return minimum_; }
    [[nodiscard]] Vec3 maximum() const noexcept { return maximum_; }
    [[nodiscard]] std::size_t triangle_count() const noexcept {
        return triangles_.size();
    }

  private:
    struct Triangle { Vec3 a{}, b{}, c{}; float normal_y{}; };
    float minimum_x_{};
    float minimum_z_{};
    float cell_size_{0.35F};
    std::size_t columns_{};
    std::size_t rows_{};
    Vec3 minimum_{};
    Vec3 maximum_{};
    std::vector<Triangle> triangles_{};
    std::vector<std::vector<std::uint32_t>> cells_{};
};

} // namespace parallel_mater::gallery
