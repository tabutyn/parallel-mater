// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>

namespace parallel_mater::gallery {

inline constexpr std::string_view gallery_execution_contract =
    "fixed-tick-60hz-v1";

// One deterministic display-to-physics clock for every Gallery adapter.
// Platform code supplies monotonic timestamps and lifecycle/input events; this
// policy owns fixed-step accumulation, overload slow motion and interpolation.
class GalleryFramePolicy {
  public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    static constexpr std::uint32_t default_target_fps = 60U;
    static constexpr double maximum_frame_delta_seconds = 0.25;

    GalleryFramePolicy();
    void set_paused(bool value) noexcept;
    void set_active(bool value) noexcept;

    // Reset without a baseline makes the next frame establish one. Supplying a
    // baseline is useful to adapters and deterministic trace tests.
    void reset() noexcept;
    void reset(TimePoint baseline) noexcept;

    void begin_frame(TimePoint now, bool simulation_enabled = true) noexcept;
    [[nodiscard]] bool can_step() const noexcept;
    void record_step(double elapsed_seconds) noexcept;
    void finish_frame() noexcept;

    [[nodiscard]] float timestep() const noexcept {
        return 1.0F / static_cast<float>(default_target_fps);
    }
    [[nodiscard]] std::uint32_t substeps() const noexcept { return 1U; }
    [[nodiscard]] std::uint32_t target_fps() const noexcept {
        return default_target_fps;
    }
    [[nodiscard]] std::uint32_t frame_steps() const noexcept {
        return frame_steps_;
    }
    [[nodiscard]] std::uint32_t dropped_steps() const noexcept {
        return dropped_steps_;
    }
    [[nodiscard]] float interpolation_alpha() const noexcept {
        return interpolation_alpha_;
    }
    [[nodiscard]] double simulation_to_wall_clock_rate() const noexcept {
        return active_wall_seconds_ > 0.0
            ? simulated_seconds_ / active_wall_seconds_ : 0.0;
    }
    [[nodiscard]] bool interpolation_history_valid() const noexcept {
        return interpolation_history_valid_;
    }
    [[nodiscard]] Clock::duration fixed_step_duration() const noexcept {
        return fixed_step_;
    }
    // Scripted qualification advances exactly one display interval. Interactive
    // adapters should instead pass their observed monotonic timestamp.
    [[nodiscard]] TimePoint next_fixed_frame_time() noexcept {
        if (!previous_) previous_ = Clock::now();
        return *previous_ + fixed_step_;
    }

  private:
    void clear_elapsed() noexcept;

    Clock::duration fixed_step_{};
    Clock::duration accumulator_{};
    std::optional<TimePoint> previous_{};
    std::uint32_t frame_steps_{};
    std::uint32_t dropped_steps_{};
    float interpolation_alpha_{1.0F};
    double active_wall_seconds_{};
    double simulated_seconds_{};
    bool paused_{};
    bool active_{true};
    bool frame_open_{};
    bool simulation_enabled_{};
    bool stalled_frame_{};
    bool interpolation_history_valid_{};
};

} // namespace parallel_mater::gallery
