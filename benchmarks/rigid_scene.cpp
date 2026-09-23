// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Samples {
    std::vector<float> values{};

    void add(float value) { values.push_back(value); }

    [[nodiscard]] float percentile(float fraction) {
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            std::round(fraction * static_cast<float>(values.size() - 1U)));
        return values[index];
    }

    void print(const char *name) {
        std::cout << name << " median=" << percentile(0.5F)
                  << " p5=" << percentile(0.05F)
                  << " p95=" << percentile(0.95F) << " ms\n";
    }
};

[[nodiscard]] bool require(parallel_mater::Status status,
                           const char *operation) {
    if (status) {
        return true;
    }
    std::cerr << operation << ": "
              << (status.message != nullptr ? status.message : "unknown")
              << '\n';
    return false;
}

} // namespace

int main(int argc, char **argv) {
    using namespace parallel_mater;
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
    if (!require(World::create(
                     {.rigid_body_capacity = static_cast<std::uint32_t>(
                          scene.rigid_bodies.size()),
                      .triangle_mesh_capacity = static_cast<std::uint32_t>(
                          scene.rigid_bodies.size())},
                     world),
                 "create world")) {
        return 1;
    }
    SceneInstance instance;
    if (!require(instantiate_scene(scene, world, instance),
                 "instantiate scene")) {
        return 1;
    }

    const StepOptions settle{.timestep = 1.0F / 60.0F,
                             .substeps = 4U,
                             .gravity = {0.0F, -9.81F, 0.0F}};
    for (int frame = 0; frame < 360; ++frame) {
        if (!require(world.step(settle), "settle scene")) {
            return 1;
        }
    }
    StepOptions measured = settle;
    measured.collect_kernel_timings = true;
    for (int frame = 0; frame < 10; ++frame) {
        if (!require(world.step(measured), "warm scene")) {
            return 1;
        }
    }

    Samples integration;
    Samples bounds;
    Samples pair_filter;
    Samples pair_compaction;
    Samples leaf_pairs;
    Samples triangle_contacts;
    Samples solve;
    Samples clear;
    Samples total;
    Samples wall;
    for (int frame = 0; frame < 120; ++frame) {
        const auto begin = std::chrono::steady_clock::now();
        if (!require(world.step(measured), "measure scene")) {
            return 1;
        }
        wall.add(std::chrono::duration<float, std::milli>(
                     std::chrono::steady_clock::now() - begin)
                     .count());
        WorldStepTimings timing{};
        if (!require(world.collect_step_timings(timing), "collect timings") ||
            !timing.available) {
            return 1;
        }
        integration.add(timing.rigid_integration.total_milliseconds);
        bounds.add(timing.rigid_world_bounds.total_milliseconds);
        pair_filter.add(timing.rigid_pair_filter.total_milliseconds);
        pair_compaction.add(timing.rigid_pair_compaction.total_milliseconds);
        leaf_pairs.add(timing.rigid_leaf_pair_generation.total_milliseconds);
        triangle_contacts.add(
            timing.rigid_contact_evaluation.total_milliseconds);
        solve.add(timing.rigid_contact_solve.total_milliseconds);
        clear.add(timing.rigid_input_clear.total_milliseconds);
        total.add(timing.total_gpu_milliseconds);
    }
    integration.print("integration");
    bounds.print("bounds");
    pair_filter.print("pair_filter");
    pair_compaction.print("pair_compaction");
    leaf_pairs.print("leaf_pairs");
    triangle_contacts.print("triangle_contacts");
    solve.print("solve");
    clear.print("clear");
    total.print("gpu_total");
    wall.print("wall");
    return 0;
}
