// SPDX-License-Identifier: MIT
#include "support.hpp"

#include <array>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::benchmark;
    using namespace parallel_mater::gallery;
    unsigned settle_frames = 360, measured_frames = 120;
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
        else valid = false;
    }
    if (!valid) {
        std::cerr << "usage: parallel-mater-rigid-scene-benchmark scene.glb [--settle N] [--frames N]\n";
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

    const StepOptions settle = standard_step_options(false);
    for (unsigned frame = 0; frame < settle_frames; ++frame) {
        if (!require(world.step(settle), "settle scene")) {
            return 1;
        }
    }
    const StepOptions measured = standard_step_options(true);
    for (int frame = 0; frame < 10; ++frame) {
        if (!require(world.step(measured), "warm scene")) {
            return 1;
        }
    }

    TimingSamples samples{};
    for (unsigned frame = 0; frame < measured_frames; ++frame) {
        WorldStepTimings timing{};
        double wall_milliseconds = 0.0;
        if (!measure_step(world, measured, "measure scene", timing,
                          wall_milliseconds)) {
            return 1;
        }
        add_timing_samples(samples, timing, wall_milliseconds);
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
    std::cout << "contact islands=" << statistics.rigid_contact_island_count
              << " early_exits=" << statistics.rigid_contact_early_exit_count
              << " max_passes=" << statistics.rigid_contact_maximum_passes
              << " colors=" << statistics.rigid_contact_color_count
              << " overflow_pairs=" << statistics.rigid_contact_overflow_pairs
              << " grid_blocks=" << statistics.rigid_contact_grid_blocks
              << " candidates=" << statistics.rigid_contact_candidate_pairs
              << " live_pairs=" << statistics.rigid_contact_live_pairs << '\n';
    return 0;
}
