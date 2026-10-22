// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/frame_policy.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Policy = parallel_mater::gallery::GalleryFramePolicy;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

Policy::TimePoint at(double seconds) {
    return Policy::TimePoint(std::chrono::duration_cast<Policy::Clock::duration>(
        std::chrono::duration<double>(seconds)));
}

std::pair<std::uint32_t, std::uint32_t> run_frame(
    Policy &policy, Policy::TimePoint time, double step_cost = 0.001,
    bool simulation_enabled = true) {
    policy.begin_frame(time, simulation_enabled);
    while (policy.can_step()) policy.record_step(step_cost);
    policy.finish_frame();
    return {policy.frame_steps(), policy.dropped_steps()};
}

void validate_realtime_rate(unsigned display_hz, unsigned seconds) {
    Policy policy(nullptr,
        parallel_mater::gallery::GalleryFrameProfile::realtime_single_step_v1);
    policy.reset(at(0.0));
    std::uint64_t ticks = 0U;
    bool saw_interpolated_render = false;
    for (unsigned frame = 1U; frame <= display_hz * seconds; ++frame) {
        const auto result = run_frame(
            policy, at(static_cast<double>(frame) / display_hz));
        require(result.first <= 1U,
                "Real-time callback submitted more than one tick");
        ticks += result.first;
        saw_interpolated_render |= result.first == 0U &&
            policy.interpolation_alpha() > 0.0F &&
            policy.interpolation_alpha() < 1.0F;
        require(policy.interpolation_alpha() >= 0.0F &&
                    policy.interpolation_alpha() < 1.0F,
                "Real-time interpolation extrapolated");
    }
    const std::uint64_t expected = std::min(display_hz, 60U) * seconds;
    require(ticks >= expected - 1U && ticks <= expected,
            "Display-rate trace produced the wrong real-time tick count");
    if (display_hz > 60U)
        require(saw_interpolated_render,
                "High-refresh trace did not render between ticks");
    const double expected_rate = std::min(
        1.0, static_cast<double>(display_hz) / 60.0);
    require(std::fabs(policy.simulation_to_wall_clock_rate() - expected_rate) <
                    0.002,
            "Simulation-to-wall-clock report hid overload");
}
} // namespace

int main() {
    try {
        Policy policy;
        policy.reset(at(0.0));
        require(run_frame(policy, at(1.0 / 120.0)).first == 0U,
                "A partial fixed step advanced physics");
        require(run_frame(policy, at(1.0 / 60.0)).first == 1U,
                "A complete fixed step did not advance physics");

        policy.reset(at(0.0));
        const auto catch_up = run_frame(policy, at(3.0 / 60.0));
        require(catch_up.first == 3U && catch_up.second == 0U,
                "Fast work did not use bounded catch-up");

        policy.reset(at(0.0));
        policy.begin_frame(at(4.0 / 60.0));
        require(policy.can_step(), "First slow catch-up step was not admitted");
        policy.record_step(0.044);
        require(!policy.can_step(), "Slow physics multiplied its stall");
        policy.finish_frame();
        require(policy.frame_steps() == 1U && policy.dropped_steps() == 3U,
                "Slow-frame backlog was not discarded deterministically");

        policy.reset(at(0.0));
        const auto stalled = run_frame(policy, at(5.0));
        require(stalled.first == Policy::maximum_catch_up_steps,
                "Long stall escaped the catch-up step cap");
        require(stalled.second > 0U,
                "Long stall did not report discarded clamped work");

        policy.set_paused(true);
        require(run_frame(policy, at(10.0)).first == 0U,
                "Paused clock submitted physics");
        policy.set_paused(false);
        require(run_frame(policy, at(20.0)).first == 0U,
                "Resume caught up paused time");
        require(run_frame(policy, at(20.0) + policy.fixed_step_duration()).first == 1U,
                "Resumed clock did not submit a new fixed step");

        policy.set_active(false);
        require(run_frame(policy, at(30.0)).first == 0U,
                "Background clock submitted physics");
        policy.set_active(true);
        require(run_frame(policy, at(40.0)).first == 0U,
                "Foreground clock caught up background time");

        parallel_mater::gallery::GalleryWorkloadLimits mobile;
        mobile.target_fps = 30U;
        policy.apply_workload(&mobile);
        require(policy.target_fps() == 30U && policy.substeps() == 8U,
                "Profile cadence was not admitted");
        require(run_frame(policy, at(50.0)).first == 0U,
                "Profile change retained an old clock baseline");
        require(run_frame(policy, at(50.0) + policy.fixed_step_duration()).first == 1U,
                "Profile cadence did not drive its fixed step");

        policy.reset(at(60.0));
        require(run_frame(policy, at(61.0), 0.001, false).first == 0U,
                "Modal frame submitted physics");
        require(policy.interpolation_alpha() == 1.0F,
                "Modal frame exposed stale interpolation");
        require(run_frame(policy, at(61.0) + policy.fixed_step_duration()).first == 1U,
                "Modal time leaked into the next frame");

        bool invalid_rejected = false;
        mobile.target_fps = 0U;
        try { policy.apply_workload(&mobile); }
        catch (const std::invalid_argument &) { invalid_rejected = true; }
        require(invalid_rejected, "Invalid profile cadence was accepted");

        Policy desktop, mobile_adapter;
        desktop.reset(at(0.0));
        mobile_adapter.reset(at(0.0));
        const std::vector<double> trace{0.005, 0.017, 0.050, 0.300, 0.317};
        for (double timestamp : trace) {
            const auto desktop_result = run_frame(desktop, at(timestamp));
            const auto mobile_result = run_frame(mobile_adapter, at(timestamp));
            require(desktop_result == mobile_result,
                    "Equal adapter traces produced different submissions");
        }

        for (const unsigned display_hz : {30U, 60U, 90U, 120U, 144U})
            validate_realtime_rate(display_hz, 10U);

        Policy realtime(nullptr,
            parallel_mater::gallery::GalleryFrameProfile::
                realtime_single_step_v1);
        require(realtime.timestep() == 1.0F / 60.0F &&
                    realtime.substeps() == 1U &&
                    realtime.target_fps() == 60U,
                "Real-time profile retained the 240-substep policy");
        parallel_mater::gallery::GalleryWorkloadLimits low_refresh;
        low_refresh.target_fps = 30U;
        Policy realtime_low_refresh(
            &low_refresh,
            parallel_mater::gallery::GalleryFrameProfile::
                realtime_single_step_v1);
        require(realtime_low_refresh.timestep() == 1.0F / 60.0F &&
                    realtime_low_refresh.substeps() == 1U,
                "30 Hz workload enlarged the real-time physics tick");
        realtime.reset(at(0.0));
        const auto overloaded = run_frame(realtime, at(1.0 / 30.0));
        require(overloaded.first == 1U && overloaded.second == 1U,
                "30 Hz overload caught up or enlarged the timestep");
        require(std::fabs(realtime.simulation_to_wall_clock_rate() - 0.5) <
                    0.0001,
                "30 Hz overload was not reported as half-speed");

        // Irregular display intervals may produce zero or one tick. Debt is
        // discarded after the callback, never repaid by a burst.
        const std::vector<double> jitter{
            0.040, 0.044, 0.061, 0.070, 0.105, 0.111, 0.129};
        for (double timestamp : jitter) {
            const auto result = run_frame(realtime, at(timestamp));
            require(result.first <= 1U,
                    "Jitter trace triggered real-time catch-up");
        }

        realtime.reset(at(1.0));
        const auto stall = run_frame(realtime, at(2.0));
        require(stall.first == 0U && stall.second >= 60U &&
                    realtime.interpolation_alpha() == 0.0F,
                "Stall debt was not discarded with zero presentation alpha");
        realtime.set_paused(true);
        require(run_frame(realtime, at(3.0)).first == 0U &&
                    realtime.interpolation_alpha() == 0.0F,
                "Real-time pause retained presentation debt");
        realtime.set_paused(false);
        require(run_frame(realtime, at(4.0)).first == 0U,
                "Real-time resume caught up paused time");
        realtime.set_active(false);
        require(run_frame(realtime, at(5.0)).first == 0U,
                "Real-time background submitted physics");
        realtime.set_active(true);
        require(run_frame(realtime, at(6.0)).first == 0U &&
                    realtime.interpolation_alpha() == 0.0F,
                "Real-time foreground caught up background time");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
