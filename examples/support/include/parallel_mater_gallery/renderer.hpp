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

class OptixRenderer {
  public:
    OptixRenderer() noexcept;
    ~OptixRenderer();
    OptixRenderer(OptixRenderer &&) noexcept;
    OptixRenderer &operator=(OptixRenderer &&) noexcept;
    OptixRenderer(const OptixRenderer &) = delete;
    OptixRenderer &operator=(const OptixRenderer &) = delete;

    [[nodiscard]] static bool create(const SceneDefinition &scene,
                                     const std::filesystem::path &ptx_path,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     OptixRenderer &output,
                                     std::string &error);

    [[nodiscard]] bool render(const World &world,
                              const SceneInstance &instance,
                              Camera camera,
                              std::vector<std::uint32_t> &rgba,
                              std::string &error);

    [[nodiscard]] std::uint32_t width() const noexcept;
    [[nodiscard]] std::uint32_t height() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::gallery
