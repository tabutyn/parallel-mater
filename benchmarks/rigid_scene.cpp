// SPDX-License-Identifier: MIT
#include "support.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::benchmark;
    using namespace parallel_mater::gallery;
    if (argc != 2) {
        std::cerr << "usage: parallel-mater-rigid-scene-benchmark scene.glb\n";
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
    if (!prepare_world(scene,
                       static_cast<std::uint32_t>(scene.rigid_bodies.size()),
                       world, instance)) {
        return 1;
    }

    const StepOptions settle = standard_step_options(false);
    for (int frame = 0; frame < 360; ++frame) {
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
    for (int frame = 0; frame < 120; ++frame) {
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
    return 0;
}
