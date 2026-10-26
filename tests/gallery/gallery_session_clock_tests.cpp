// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/session.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace gallery = parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        const std::filesystem::path root = PM_GALLERY_SESSION_TEST_ROOT;
        gallery::GallerySessionConfig config{
            .assets = root / "examples/assets",
            .profiles = root /
                "tests/gallery/assets/session_contract_profiles.json",
            .platform = "ContractTest",
            .model = "test-host",
            .backend = PM_GALLERY_SESSION_TEST_BACKEND,
        };
        gallery::GallerySession session(std::move(config));
        using FrameClock = gallery::GallerySession::FrameClock;
        const auto baseline = FrameClock::time_point{};
        const auto overloaded_interval =
            std::chrono::duration_cast<FrameClock::duration>(
                std::chrono::duration<double>(1.0 / 30.0));
        gallery::GalleryFramePlan plan;
        plan.controls = {
            .right = 1.0F,
            .forward = -1.0F,
            .camera = session.camera().camera(),
            .gravity_override = parallel_mater::Vec3{1.0F, 2.0F, 3.0F},
        };
        unsigned renders = 0U;
        session.frame_at(baseline, plan,
            [&](gallery::GallerySession &, float) { ++renders; });
        const auto zero_tick_gravity = session.controller_state().gravity;
        const auto zero_tick_state = session.presentation();
        plan.controls.gravity_override =
            parallel_mater::Vec3{4.0F, 5.0F, 6.0F};
        session.frame_at(baseline + overloaded_interval, plan,
            [&](gallery::GallerySession &, float) { ++renders; });
        const auto state = session.presentation();
        require(session.frames == 1U && renders == 2U &&
                    session.frame_timestep() == 1.0F / 60.0F &&
                    session.frame_substeps() == 1U &&
                    state.execution_contract ==
                        gallery::gallery_execution_contract &&
                    state.simulation_to_wall_clock_rate > 0.49 &&
                    state.simulation_to_wall_clock_rate < 0.51 &&
                    state.interpolation_alpha >= 0.0F &&
                    state.interpolation_alpha < 1.0F &&
                    state.interpolation_history_valid &&
                    state.presentation_latency_ticks == 1U &&
                    !zero_tick_state.interpolation_history_valid &&
                    zero_tick_state.presentation_latency_ticks == 0U &&
                    zero_tick_gravity.x != 1.0F &&
                    state.controls.gravity.x == 4.0F &&
                    state.controls.gravity.y == 5.0F &&
                    state.controls.gravity.z == 6.0F,
                "Fixed-tick session hid overload or sampled a tick twice");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        if (std::string_view(error.what()) ==
            "no Parallel Mater physical device") {
            std::cout << "SKIP: no selected Gallery physics device\n";
            return 77;
        }
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
