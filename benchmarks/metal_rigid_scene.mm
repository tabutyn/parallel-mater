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
          double *wall = nullptr,
          parallel_mater::HostSpan<parallel_mater::RigidBodyId> vertical = {}) {
    const parallel_mater::Vec3 gravity = vertical.size == 0U
        ? parallel_mater::Vec3{0.0F, -9.81F, 0.0F}
        : parallel_mater::Vec3{4.905F, -8.495709F, 0.0F};
    if (vertical.size != 0U && !world.apply_central_acceleration(
            vertical, {-gravity.x, -9.81F - gravity.y, 0.0F})) return false;
    const StepOptions options{.timestep = 1.0F / 60.0F,
                              .substeps = 4U,
                              .gravity = gravity,
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
             std::uint32_t frames,
             parallel_mater::HostSpan<parallel_mater::RigidBodyId> vertical = {}) {
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
        if (!step(world, true, &timings, &milliseconds, vertical)) return false;
        wall.push_back(milliseconds);
        gpu.push_back(timings.total_gpu_milliseconds);
        solve.push_back(timings.rigid_contact_solve.total_milliseconds);
        evaluate.push_back(
            timings.rigid_contact_evaluation.total_milliseconds);
        filter.push_back(timings.rigid_pair_filter.total_milliseconds);
    }
    WorldStatistics statistics{};
    if (!world.collect_statistics(statistics)) return false;
    std::cout << "mode=" << mode << " phase=" << phase
              << " frames=" << frames
              << " wall_median_ms=" << median(wall)
              << " wall_p95_ms=" << percentile95(wall)
              << " wall_max_ms=" << *std::max_element(wall.begin(), wall.end())
              << " sleeping_bodies=" << statistics.sleeping_rigid_body_count
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

bool run(const SceneDefinition &scene, bool sleeping, std::string_view scenario) {
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
    if (scenario != "quiet") {
        for (unsigned frame = 0; frame < 120U; ++frame)
            if (!step(world, false)) return false;
        if (scenario == "steering") {
            std::vector<parallel_mater::RigidBodyId> vertical;
            for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i)
                if (!scene.rigid_bodies[i].follows_gravity_tilt)
                    vertical.push_back(instance.rigid_bodies[i]);
            const parallel_mater::HostSpan<parallel_mater::RigidBodyId> bodies{
                vertical.data(), vertical.size()};
            return measure(world, mode, "steering-start", 60U, bodies) &&
                   measure(world, mode, "steering-held", 120U, bodies);
        }
        std::size_t ball = scene.rigid_bodies.size();
        for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i)
            if (scene.rigid_bodies[i].source_name == "Icosphere") ball = i;
        if (ball == scene.rigid_bodies.size()) return false;
        RigidBodyState state = scene.rigid_bodies[ball].options.initial_state;
        state.position = {0.0F, 1.05F, 2.5F};
        state.linear_velocity = {0.0F, 0.0F, -10.0F};
        if (!world.set_rigid_body_state(instance.rigid_bodies[ball], state))
            return false;
        if (!measure(world, mode, "impact", 120U)) return false;
        unsigned displaced = 0U;
        for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
            RigidBodyState current{};
            if (!world.read_rigid_body_state(instance.rigid_bodies[i], current))
                return false;
            if (!std::isfinite(current.position.x) ||
                !std::isfinite(current.position.y) ||
                !std::isfinite(current.position.z)) return false;
            const auto initial = scene.rigid_bodies[i].options.initial_state.position;
            if (!scene.rigid_bodies[i].follows_gravity_tilt &&
                std::hypot(current.position.x - initial.x,
                           current.position.z - initial.z) > 0.05F) ++displaced;
        }
        std::cout << "mode=" << mode << " displaced_bricks=" << displaced << '\n';
        return displaced >= 16U;
    }
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
    std::filesystem::path path{PARALLEL_MATER_RIGID_BODY_SCENE_PATH};
    std::string_view scenario = "quiet";
    std::string_view mode = "both";
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if ((arg == "--scenario" || arg == "--mode") && i + 1 < argc) {
            if (arg == "--scenario") scenario = argv[++i];
            else mode = argv[++i];
        } else path = argv[i];
    }
    if ((scenario != "quiet" && scenario != "impact" && scenario != "steering") ||
        (mode != "awake" && mode != "sleep" && mode != "both")) return 2;
    SceneDefinition scene;
    std::string error;
    if (!parallel_mater::metal::gallery::load_glb_scene(path, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    return (mode == "sleep" || run(scene, false, scenario)) &&
           (mode == "awake" || run(scene, true, scenario)) ? 0 : 1;
}
