// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_gallery/gpu_frame.hpp>

struct GLFWwindow;

namespace parallel_mater::gallery {

struct GpuRenderTarget;

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
        std::string_view title, GalleryWindow &output, std::string &error,
        std::optional<PmNativeBackend> backend = std::nullopt);

    [[nodiscard]] bool present(const std::vector<std::uint32_t> &rgba,
                               std::string &error, NativeRenderImage image = {});
    void set_vsync(bool enabled) noexcept;
    [[nodiscard]] GLFWwindow *native_handle() const noexcept;
    // Main-thread surface/layer setup only. The renderer owns API resources;
    // the window adapter owns and outlives the returned native target.
    [[nodiscard]] GpuRenderTarget native_render_target(
        const PmNativeDeviceInfo &device, std::uint32_t width,
        std::uint32_t height, std::string &error);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::gallery
