// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using parallel_mater::metal::StepOptions;
using parallel_mater::metal::World;
using parallel_mater::metal::WorldOptions;
using parallel_mater::metal::WorldStatistics;
using parallel_mater::metal::WorldStepTimings;
using parallel_mater::metal::RigidBodyDeviceView;
using parallel_mater::metal::RigidBodyState;
using parallel_mater::metal::gallery::SceneDefinition;
using parallel_mater::metal::gallery::SceneInstance;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    return values.size() % 2U != 0U
        ? values[middle]
        : 0.5 * (values[middle - 1U] + values[middle]);
}

double percentile95(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(
        0.95 * static_cast<double>(values.size() - 1U))];
}

bool step(World &world, bool timing, WorldStepTimings *output = nullptr,
          double *wall = nullptr) {
    const StepOptions options{.timestep = 1.0F / 60.0F,
                              .substeps = 4U,
                              .gravity = {0.0F, -9.81F, 0.0F},
                              .collect_kernel_timings = timing};
    const auto begin = std::chrono::steady_clock::now();
    const auto status = world.step(options);
    if (!status) {
        std::cerr << "step failed: " << status.message << '\n';
        return false;
    }
    if (wall != nullptr)
        *wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    if (output != nullptr) {
        const auto timing_status = world.collect_step_timings(*output);
        if (!timing_status || !output->available) {
            std::cerr << "timing collection failed\n";
            return false;
        }
    }
    return true;
}

bool measure(World &world, const char *mode, const char *phase,
             std::uint32_t frames) {
    std::vector<double> wall;
    std::vector<double> gpu;
    std::vector<double> solve;
    std::vector<double> evaluate;
    std::vector<double> filter;
    wall.reserve(frames);
    gpu.reserve(frames);
    solve.reserve(frames);
    evaluate.reserve(frames);
    filter.reserve(frames);
    for (std::uint32_t frame = 0U; frame < frames; ++frame) {
        WorldStepTimings timings{};
        double milliseconds = 0.0;
        if (!step(world, true, &timings, &milliseconds)) return false;
        wall.push_back(milliseconds);
        gpu.push_back(timings.total_gpu_milliseconds);
        solve.push_back(timings.rigid_contact_solve.total_milliseconds);
        evaluate.push_back(
            timings.rigid_contact_evaluation.total_milliseconds);
        filter.push_back(timings.rigid_pair_filter.total_milliseconds);
    }
    std::cout << "mode=" << mode << " phase=" << phase
              << " frames=" << frames
              << " wall_median_ms=" << median(wall)
              << " wall_p95_ms=" << percentile95(wall)
              << " gpu_median_ms=" << median(gpu)
              << " solve_median_ms=" << median(solve)
              << " contact_median_ms=" << median(evaluate)
              << " filter_median_ms=" << median(filter) << '\n';
    return true;
}

void print_motion(World &world, const char *mode) {
    RigidBodyDeviceView view{};
    if (!world.rigid_body_view(view) || view.states.buffer == nullptr) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)view.states.buffer;
    const auto *states = reinterpret_cast<const RigidBodyState *>(
        static_cast<const std::uint8_t *>(buffer.contents) +
        view.states.byte_offset);
    float maximum_linear = 0.0F;
    float maximum_angular = 0.0F;
    for (std::uint64_t body = 0U; body < view.states.size; ++body) {
        const auto &linear = states[body].linear_velocity;
        const auto &angular = states[body].angular_velocity;
        maximum_linear = std::max(maximum_linear, std::sqrt(
            linear.x * linear.x + linear.y * linear.y + linear.z * linear.z));
        maximum_angular = std::max(maximum_angular, std::sqrt(
            angular.x * angular.x + angular.y * angular.y +
            angular.z * angular.z));
    }
    std::cout << "mode=" << mode << " settled_max_linear="
              << maximum_linear << " settled_max_angular="
              << maximum_angular << '\n';
}

bool run(const SceneDefinition &scene, bool sleeping) {
    WorldOptions options{};
    auto status = parallel_mater::metal::gallery::scene_world_options(
        scene, options);
    if (!status) {
        std::cerr << "derive options failed: " << status.message << '\n';
        return false;
    }
    options.rigid_sleeping = sleeping;
    World world;
    SceneInstance instance;
    status = World::create(options, world);
    if (status)
        status = parallel_mater::metal::gallery::instantiate_scene(
            scene, world, instance);
    if (!status) {
        std::cerr << "create scene failed: " << status.message << '\n';
        return false;
    }
    const char *mode = sleeping ? "sleep" : "awake";
    if (!measure(world, mode, "startup", 24U)) return false;
    if (!measure(world, mode, "motion", 96U)) return false;
    for (std::uint32_t frame = 120U; frame < 360U; ++frame)
        if (!step(world, false)) return false;
    print_motion(world, mode);
    if (!measure(world, mode, "rest", 120U)) return false;
    WorldStatistics statistics{};
    if (!world.collect_statistics(statistics)) return false;
    std::cout << "mode=" << mode
              << " allocated_bytes=" << statistics.allocated_bytes
              << " sleeping_bodies="
              << statistics.sleeping_rigid_body_count << '\n';
    return true;
}

} // namespace

int main(int argc, char **argv) {
    const std::filesystem::path path = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path(PARALLEL_MATER_RIGID_BODY_SCENE_PATH);
    SceneDefinition scene;
    std::string error;
    if (!parallel_mater::metal::gallery::load_glb_scene(path, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    return run(scene, false) && run(scene, true) ? 0 : 1;
}
