// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <parallel_mater_gallery/gpu_frame.hpp>

struct GLFWwindow;

namespace parallel_mater::gallery {

// Owns the native window and the API-specific presentation surface. Physics
// and scene code hand it an adapter-owned GPU image or a CPU image, plus UI.
class GalleryWindow {
  public:
    GalleryWindow() noexcept;
    ~GalleryWindow();
    GalleryWindow(GalleryWindow &&) noexcept;
    GalleryWindow &operator=(GalleryWindow &&) noexcept;
    GalleryWindow(const GalleryWindow &) = delete;
    GalleryWindow &operator=(const GalleryWindow &) = delete;

    [[nodiscard]] static bool create(
        std::uint32_t window_width, std::uint32_t window_height,
        std::uint32_t image_width, std::uint32_t image_height,
        std::string_view title, GalleryWindow &output, std::string &error);

    [[nodiscard]] bool present(const std::vector<std::uint32_t> &rgba,
                               std::string &error, NativeRenderImage image = {});
    void set_vsync(bool enabled) noexcept;
    [[nodiscard]] GLFWwindow *native_handle() const noexcept;
#if defined(__APPLE__)
    // Main-thread layer setup only. Shared graphics code owns rendering.
    [[nodiscard]] void *metal_layer(void *device);
#endif

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::gallery
