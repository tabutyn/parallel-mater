// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/physics_frame_budget.hpp>
#include <parallel_mater_scene/device_profiles.hpp>

#include <chrono>
#include <cstdint>
#include <optional>

namespace parallel_mater::gallery {

// The client clock is selected explicitly so the legacy 240-substep cadence
// and its evidence are never silently reinterpreted as the F8 profile.
enum class GalleryFrameProfile : std::uint8_t {
    legacy_v1,
    realtime_single_step_v1,
};

[[nodiscard]] constexpr const char *gallery_frame_profile_name(
    GalleryFrameProfile profile) noexcept {
    return profile == GalleryFrameProfile::realtime_single_step_v1
        ? "realtime-single-step-v1" : "legacy-v1";
}

// One deterministic display-to-physics clock for every Gallery adapter.
// Platform code supplies monotonic timestamps and lifecycle/input events; this
// policy owns fixed-step accumulation, bounded catch-up, and profile cadence.
class GalleryFramePolicy {
  public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    static constexpr std::uint32_t default_target_fps = 60U;
    static constexpr std::uint32_t solver_substeps_per_second = 240U;
    static constexpr std::uint32_t maximum_catch_up_steps = 4U;
    static constexpr double maximum_frame_delta_seconds = 0.25;
    static constexpr double physics_budget_seconds = 0.008;

    explicit GalleryFramePolicy(
        const GalleryWorkloadLimits *limits = nullptr,
        GalleryFrameProfile profile = GalleryFrameProfile::legacy_v1);

    // Null limits select the portable 60 Hz workload. Applying a different
    // frame profile resets the clock so old cadence cannot become new work.
    void apply_workload(
        const GalleryWorkloadLimits *limits,
        GalleryFrameProfile profile = GalleryFrameProfile::legacy_v1);
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
        return profile_ == GalleryFrameProfile::realtime_single_step_v1
            ? 1.0F / 60.0F : 1.0F / static_cast<float>(target_fps_);
    }
    [[nodiscard]] std::uint32_t substeps() const noexcept {
        return profile_ == GalleryFrameProfile::realtime_single_step_v1
            ? 1U
            : (solver_substeps_per_second + target_fps_ - 1U) / target_fps_;
    }
    [[nodiscard]] std::uint32_t target_fps() const noexcept {
        return target_fps_;
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
    [[nodiscard]] GalleryFrameProfile profile() const noexcept {
        return profile_;
    }
    [[nodiscard]] bool interpolation_history_valid() const noexcept {
        return profile_ == GalleryFrameProfile::legacy_v1 ||
            interpolation_history_valid_;
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

    std::uint32_t target_fps_{default_target_fps};
    GalleryFrameProfile profile_{GalleryFrameProfile::legacy_v1};
    Clock::duration fixed_step_{};
    Clock::duration accumulator_{};
    std::optional<TimePoint> previous_{};
    PhysicsFrameBudget budget_{maximum_catch_up_steps,
                               physics_budget_seconds};
    std::uint32_t frame_steps_{};
    std::uint32_t dropped_steps_{};
    float interpolation_alpha_{1.0F};
    double active_wall_seconds_{};
    double simulated_seconds_{};
    bool paused_{};
    bool active_{true};
    bool frame_open_{};
    bool simulation_enabled_{};
    bool has_submitted_{};
    bool stalled_frame_{};
    bool interpolation_history_valid_{};
};

} // namespace parallel_mater::gallery
