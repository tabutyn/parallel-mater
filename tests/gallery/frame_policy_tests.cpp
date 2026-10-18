// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/frame_policy.hpp>

#include <chrono>
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
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
