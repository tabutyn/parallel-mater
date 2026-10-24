// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/frame_policy.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace parallel_mater::gallery {
namespace {
GalleryFramePolicy::Clock::duration make_fixed_step_duration(
    std::uint32_t target_fps) noexcept {
    using Seconds = std::chrono::duration<double>;
    auto duration = std::chrono::duration_cast<GalleryFramePolicy::Clock::duration>(
        Seconds(1.0 / static_cast<double>(target_fps)));
    return std::max(duration, GalleryFramePolicy::Clock::duration{1});
}
} // namespace

GalleryFramePolicy::GalleryFramePolicy(const GalleryWorkloadLimits *limits,
                                       GalleryFrameProfile profile) {
    apply_workload(limits, profile);
}

void GalleryFramePolicy::apply_workload(const GalleryWorkloadLimits *limits,
                                        GalleryFrameProfile profile) {
    const std::uint32_t target = limits ? limits->target_fps
                                        : default_target_fps;
    if (target == 0U || target > solver_substeps_per_second)
        throw std::invalid_argument(
            "Gallery workload target_fps must be between 1 and 240");
    target_fps_ = target;
    profile_ = profile;
    fixed_step_ = make_fixed_step_duration(
        profile_ == GalleryFrameProfile::realtime_single_step_v1
            ? default_target_fps : target_fps_);
    reset();
}

void GalleryFramePolicy::clear_elapsed() noexcept {
    accumulator_ = Clock::duration::zero();
    frame_steps_ = 0U;
    dropped_steps_ = 0U;
    interpolation_alpha_ =
        profile_ == GalleryFrameProfile::realtime_single_step_v1 ? 0.0F : 1.0F;
    active_wall_seconds_ = 0.0;
    simulated_seconds_ = 0.0;
    frame_open_ = false;
    simulation_enabled_ = false;
    has_submitted_ = false;
    stalled_frame_ = false;
    interpolation_history_valid_ = false;
    budget_ = PhysicsFrameBudget(maximum_catch_up_steps,
                                 physics_budget_seconds);
}

void GalleryFramePolicy::reset() noexcept {
    clear_elapsed();
    previous_.reset();
}

void GalleryFramePolicy::reset(TimePoint baseline) noexcept {
    clear_elapsed();
    previous_ = baseline;
}

void GalleryFramePolicy::set_paused(bool value) noexcept {
    if (paused_ == value) return;
    paused_ = value;
    reset();
}

void GalleryFramePolicy::set_active(bool value) noexcept {
    if (active_ == value) return;
    active_ = value;
    reset();
}

void GalleryFramePolicy::begin_frame(TimePoint now,
                                     bool simulation_enabled) noexcept {
    frame_steps_ = 0U;
    dropped_steps_ = 0U;
    budget_ = PhysicsFrameBudget(maximum_catch_up_steps,
                                 physics_budget_seconds);
    frame_open_ = active_;
    simulation_enabled_ = active_ && !paused_ && simulation_enabled;
    stalled_frame_ = false;
    if (!active_) return;

    if (!previous_ || now < *previous_) {
        previous_ = now;
        accumulator_ = Clock::duration::zero();
        if (profile_ == GalleryFrameProfile::realtime_single_step_v1)
            interpolation_history_valid_ = false;
    } else {
        const auto raw_elapsed = now - *previous_;
        const auto maximum_delta = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(maximum_frame_delta_seconds));
        previous_ = now;
        if (!paused_ && simulation_enabled) {
            active_wall_seconds_ +=
                std::chrono::duration<double>(raw_elapsed).count();
            if (profile_ == GalleryFrameProfile::realtime_single_step_v1 &&
                raw_elapsed > maximum_delta) {
                const std::uint64_t due = static_cast<std::uint64_t>(
                    raw_elapsed / fixed_step_);
                dropped_steps_ = due >
                        std::numeric_limits<std::uint32_t>::max()
                    ? std::numeric_limits<std::uint32_t>::max()
                    : static_cast<std::uint32_t>(due);
                accumulator_ = Clock::duration::zero();
                stalled_frame_ = true;
            } else {
                accumulator_ += std::min(raw_elapsed, maximum_delta);
            }
        }
    }

    if (paused_ || !simulation_enabled) {
        accumulator_ = Clock::duration::zero();
        if (profile_ == GalleryFrameProfile::realtime_single_step_v1)
            interpolation_history_valid_ = false;
        interpolation_alpha_ =
            profile_ == GalleryFrameProfile::realtime_single_step_v1
                ? 0.0F : 1.0F;
    }
}

bool GalleryFramePolicy::can_step() const noexcept {
    return frame_open_ && active_ && !paused_ &&
           !stalled_frame_ && accumulator_ >= fixed_step_ &&
           budget_.can_step() &&
           (profile_ != GalleryFrameProfile::realtime_single_step_v1 ||
            frame_steps_ == 0U);
}

void GalleryFramePolicy::record_step(double elapsed_seconds) noexcept {
    if (!can_step()) return;
    accumulator_ -= fixed_step_;
    budget_.record_step(std::max(0.0, elapsed_seconds));
    ++frame_steps_;
    has_submitted_ = true;
    interpolation_history_valid_ = true;
    simulated_seconds_ += static_cast<double>(timestep());
}

void GalleryFramePolicy::finish_frame() noexcept {
    if (!frame_open_) return;
    if (active_ && !paused_ && accumulator_ >= fixed_step_ &&
        (profile_ == GalleryFrameProfile::realtime_single_step_v1 ||
         !budget_.can_step())) {
        const std::uint64_t due = static_cast<std::uint64_t>(
            accumulator_ / fixed_step_);
        dropped_steps_ = due > std::numeric_limits<std::uint32_t>::max()
            ? std::numeric_limits<std::uint32_t>::max()
            : static_cast<std::uint32_t>(due);
        accumulator_ %= fixed_step_;
    }
    if (profile_ == GalleryFrameProfile::realtime_single_step_v1) {
        interpolation_alpha_ = !simulation_enabled_ || stalled_frame_
            ? 0.0F
            : static_cast<float>(static_cast<double>(accumulator_.count()) /
                                 static_cast<double>(fixed_step_.count()));
    } else {
        interpolation_alpha_ = !simulation_enabled_ || !has_submitted_
            ? 1.0F
            : static_cast<float>(static_cast<double>(accumulator_.count()) /
                                 static_cast<double>(fixed_step_.count()));
    }
    frame_open_ = false;
    simulation_enabled_ = false;
}

} // namespace parallel_mater::gallery
