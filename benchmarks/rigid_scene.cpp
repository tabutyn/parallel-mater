// SPDX-License-Identifier: MIT
#include "support.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::benchmark;
    using namespace parallel_mater::gallery;
    unsigned settle_frames = 360, measured_frames = 120;
    unsigned pass_limit = 0;
    const auto number = [](const char *text, unsigned &value) {
        const std::string_view input(text);
        const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
        return parsed.ec == std::errc{} && parsed.ptr == input.data() + input.size() && value <= 10000;
    };
    bool valid = argc >= 2;
    for (int i = 2; valid && i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--settle" && i + 1 < argc) valid = number(argv[++i], settle_frames);
        else if (option == "--frames" && i + 1 < argc) valid = number(argv[++i], measured_frames) && measured_frames > 0;
        else if (option == "--passes" && i + 1 < argc) valid = number(argv[++i], pass_limit) && pass_limit <= 64;
        else valid = false;
    }
    if (!valid) {
        std::cerr << "usage: parallel-mater-rigid-scene-benchmark scene.glb [--settle N] [--frames N] [--passes 0..64]\n";
        return 2;
    }

    SceneDefinition scene;
    std::string error;
    if (!load_glb_scene(argv[1], scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    World world;
    SceneInstance instance;
    if (!prepare_world(scene, world, instance)) {
        return 1;
    }

    StepOptions settle = standard_step_options(false);
    settle.rigid_contact_pass_limit = pass_limit;
    for (unsigned frame = 0; frame < settle_frames; ++frame) {
        if (!require(world.step(settle), "settle scene")) {
            return 1;
        }
    }
    StepOptions measured = standard_step_options(true);
    measured.rigid_contact_pass_limit = pass_limit;
    for (int frame = 0; frame < 10; ++frame) {
        if (!require(world.step(measured), "warm scene")) {
            return 1;
        }
    }

    TimingSamples samples{};
    unsigned solve_launches = 0;
    for (unsigned frame = 0; frame < measured_frames; ++frame) {
        WorldStepTimings timing{};
        double wall_milliseconds = 0.0;
        if (!measure_step(world, measured, "measure scene", timing,
                          wall_milliseconds)) {
            return 1;
        }
        add_timing_samples(samples, timing, wall_milliseconds);
        solve_launches = timing.rigid_contact_solve.launch_count;
    }
    constexpr std::array<const char *, k_timing_stage_count> stage_names{
        "integration", "bounds", "pair_filter", "pair_compaction",
        "leaf_pairs", "triangle_contacts", "solve", "clear",
        "gpu_total", "wall"};
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        samples[index].print_summary(std::cout, stage_names[index]);
    }
    WorldStatistics statistics{};
    if (!require(world.collect_statistics(statistics), "contact statistics")) return 1;
    std::cout << "contact pass_limit=" << pass_limit << " solve_launches=" << solve_launches
              << " islands=" << statistics.rigid_contact_island_count
              << " early_exits=" << statistics.rigid_contact_early_exit_count
              << " max_passes=" << statistics.rigid_contact_maximum_passes
              << " colors=" << statistics.rigid_contact_color_count
              << " overflow_pairs=" << statistics.rigid_contact_overflow_pairs
              << " grid_blocks=" << statistics.rigid_contact_grid_blocks
              << " candidates=" << statistics.rigid_contact_candidate_pairs
              << " live_pairs=" << statistics.rigid_contact_live_pairs << '\n';
    float maximum_drop = 0, maximum_displacement = 0, maximum_speed = 0;
    for (std::size_t i = 0; i < instance.rigid_bodies.size(); ++i) {
        if (scene.rigid_bodies[i].options.motion != MotionType::dynamic) continue;
        RigidBodyState state{};
        if (!require(world.read_rigid_body_state(instance.rigid_bodies[i], state), "quality readback")) return 1;
        const auto initial = scene.rigid_bodies[i].options.initial_state.position;
        const float dx = state.position.x - initial.x, dy = state.position.y - initial.y, dz = state.position.z - initial.z;
        maximum_drop = std::max(maximum_drop, -dy);
        maximum_displacement = std::max(maximum_displacement, std::sqrt(dx*dx + dy*dy + dz*dz));
        const auto v = state.linear_velocity;
        maximum_speed = std::max(maximum_speed, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
    }
    std::cout << "quality max_drop=" << maximum_drop << " max_displacement=" << maximum_displacement
              << " final_max_speed=" << maximum_speed << '\n';
    return 0;
}
