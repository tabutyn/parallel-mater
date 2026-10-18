// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/session.hpp>

#include <cstdint>
#include <functional>
#include <string>

namespace parallel_mater::gallery {

enum class MobileGalleryCommand : std::uint8_t {
    select_scene,
    reset_scene,
    toggle_paused,
    set_paused,
    steer,
    orbit,
    zoom,
    primary_action,
    set_count,
    focus,
};

struct MobileGalleryState {
    GalleryPresentationState presentation{};
    bool foreground{};
    bool surface_ready{};
    bool failed{};
    std::uint64_t surface_generation{};
    std::string failure;
};

// Common mobile shell policy. Android and iPad translate OS lifecycle and
// input into this class; it alone owns simulation admission, controller state,
// and recovery after a failed command/frame. Native presenters remain outside.
class MobileGallerySession {
  public:
    using FrameClock = GallerySession::FrameClock;
    using Presenter = std::function<void(const GallerySession &)>;

    explicit MobileGallerySession(GallerySessionConfig config);

    [[nodiscard]] GallerySession &gallery() noexcept { return gallery_; }
    [[nodiscard]] const GallerySession &gallery() const noexcept {
        return gallery_;
    }

    [[nodiscard]] bool attach_surface(
        unsigned width, unsigned height,
        GpuRenderTarget target = {}) noexcept;
    void detach_surface() noexcept;
    [[nodiscard]] bool resize_surface(unsigned width,
                                      unsigned height) noexcept;
    void set_foreground(bool foreground) noexcept;

    [[nodiscard]] bool execute(MobileGalleryCommand command, float x = 0.0F,
                               float y = 0.0F) noexcept;
    [[nodiscard]] bool frame(const Presenter &present = {}) noexcept;
    [[nodiscard]] bool frame_at(FrameClock::time_point timestamp,
                                const Presenter &present = {}) noexcept;

    void report_failure(std::string failure) noexcept;
    [[nodiscard]] MobileGalleryState state() const;
    [[nodiscard]] bool should_frame() const noexcept {
        return foreground_ && surface_ready_ && !failed_;
    }

  private:
    GallerySession gallery_;
    bool foreground_{true};
    bool surface_ready_{};
    bool failed_{};
    std::uint64_t surface_generation_{};
    std::string failure_;

    void synchronize_activity() noexcept;
    void clear_failure() noexcept;
};

} // namespace parallel_mater::gallery
