// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

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

[[nodiscard]] bool parse_count(const char *text, std::uint32_t minimum,
                               std::uint32_t maximum, std::uint32_t &value) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed < minimum || parsed > maximum) {
        return false;
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

} // namespace

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    std::uint32_t sphere_count = 1'000U;
    std::uint32_t frame_count = 240U;
    if (argc > 3 ||
        (argc > 1 && !parse_count(argv[1], 10U, 1'000U, sphere_count)) ||
        (argc > 2 && !parse_count(argv[2], 1U, 100'000U, frame_count))) {
        std::cerr << "usage: parallel-mater-dump-benchmark [10..1000 spheres] "
                     "[1..100000 frames]\n";
        return 2;
    }

    const SceneDefinition scene = make_dump_scene(sphere_count);
    World world;
    if (!require(World::create(
                     {.rigid_body_capacity =
                          static_cast<std::uint32_t>(scene.rigid_bodies.size()),
                      .triangle_mesh_capacity =
                          static_cast<std::uint32_t>(scene.meshes.size())},
                     world), "create DUMP world")) {
        return 1;
    }
    SceneInstance instance;
    if (!require(instantiate_scene(scene, world, instance),
                 "instantiate DUMP scene")) {
        return 1;
    }
    WorldStatistics statistics{};
    if (!require(world.collect_statistics(statistics), "collect memory")) {
        return 1;
    }
    std::cout << "spheres=" << sphere_count << " allocated_bytes="
              << statistics.allocated_bytes << '\n';
    std::cout << "frames,integration,bounds,filter,compact,leaf,triangle,"
                 "solve,clear,gpu,wall,gpu_p95 (ms)\n";

    constexpr float pi = 3.14159265358979323846F;
    float hopper_angle = pi * 0.25F;
    RigidBodyState hopper_target = scene.rigid_bodies[0].options.initial_state;
    const StepOptions options{.timestep = 1.0F / 60.0F,
                              .substeps = 4U,
                              .gravity = {0.0F, -9.81F, 0.0F},
                              .collect_kernel_timings = true};
    std::array<double, 10> sums{};
    std::vector<float> gpu_samples;
    std::uint32_t interval_start = 1U;
    std::cout << std::fixed << std::setprecision(3);
    for (std::uint32_t frame = 1U; frame <= frame_count; ++frame) {
        hopper_angle = std::max(
            -pi * 0.25F, hopper_angle - pi * 0.25F / 60.0F);
        hopper_target.orientation = {
            0.0F, 0.0F, std::sin(hopper_angle * 0.5F),
            std::cos(hopper_angle * 0.5F)};
        if (!require(world.set_kinematic_target(instance.rigid_bodies[0],
                                                hopper_target),
                     "rotate hopper")) {
            return 1;
        }
        const auto begin = std::chrono::steady_clock::now();
        if (!require(world.step(options), "step DUMP world")) {
            return 1;
        }
        const double wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        WorldStepTimings timing{};
        if (!require(world.collect_step_timings(timing), "collect timings") ||
            !timing.available) {
            return 1;
        }
        const std::array<double, 10> values{
            timing.rigid_integration.total_milliseconds,
            timing.rigid_world_bounds.total_milliseconds,
            timing.rigid_pair_filter.total_milliseconds,
            timing.rigid_pair_compaction.total_milliseconds,
            timing.rigid_leaf_pair_generation.total_milliseconds,
            timing.rigid_contact_evaluation.total_milliseconds,
            timing.rigid_contact_solve.total_milliseconds,
            timing.rigid_input_clear.total_milliseconds,
            timing.total_gpu_milliseconds, wall};
        for (std::size_t index = 0U; index < values.size(); ++index) {
            sums[index] += values[index];
        }
        gpu_samples.push_back(timing.total_gpu_milliseconds);
        if (frame % 30U == 0U || frame == frame_count) {
            std::sort(gpu_samples.begin(), gpu_samples.end());
            const double count = static_cast<double>(gpu_samples.size());
            std::cout << interval_start << '-' << frame;
            for (const double sum : sums) {
                std::cout << ',' << sum / count;
            }
            const std::size_t percentile = static_cast<std::size_t>(
                std::ceil(count * 0.95)) - 1U;
            std::cout << ',' << gpu_samples[percentile] << '\n';
            interval_start = frame + 1U;
            sums = {};
            gpu_samples.clear();
        }
    }

    RigidBodyDeviceView view{};
    if (!require(world.rigid_body_view(view), "view sphere states")) {
        return 1;
    }
    std::vector<RigidBodyState> states(view.states.size);
    const cudaError_t copy_error = cudaMemcpy(
        states.data(), view.states.data,
        states.size() * sizeof(RigidBodyState), cudaMemcpyDeviceToHost);
    if (copy_error != cudaSuccess) {
        std::cerr << "copy sphere states: "
                  << cudaGetErrorString(copy_error) << '\n';
        return 1;
    }
    std::uint32_t invalid = 0U;
    std::uint32_t below_receiver = 0U;
    std::uint32_t outside_receiver = 0U;
    float minimum_height = std::numeric_limits<float>::infinity();
    float maximum_speed = 0.0F;
    double linear_speed_squared = 0.0;
    double angular_speed_squared = 0.0;
    std::uint64_t state_hash = 1469598103934665603ULL;
    for (std::size_t index = 2U; index < states.size(); ++index) {
        const RigidBodyState &state = states[index];
        const Vec3 point = state.position;
        const Vec3 velocity = state.linear_velocity;
        const Vec3 angular = state.angular_velocity;
        invalid += !std::isfinite(point.x) || !std::isfinite(point.y) ||
                   !std::isfinite(point.z) ||
                   !std::isfinite(velocity.x) ||
                   !std::isfinite(velocity.y) ||
                   !std::isfinite(velocity.z) ||
                   !std::isfinite(angular.x) ||
                   !std::isfinite(angular.y) ||
                   !std::isfinite(angular.z) ||
                   !std::isfinite(state.orientation.x) ||
                   !std::isfinite(state.orientation.y) ||
                   !std::isfinite(state.orientation.z) ||
                   !std::isfinite(state.orientation.w);
        below_receiver += point.y < -2.6F;
        outside_receiver += point.x < -1.0F || point.x > 4.0F ||
                            point.y < -2.5F || point.y > 2.5F ||
                            point.z < -2.5F || point.z > 2.5F;
        minimum_height = std::min(minimum_height, point.y);
        const float speed_squared = velocity.x * velocity.x +
                                    velocity.y * velocity.y +
                                    velocity.z * velocity.z;
        linear_speed_squared += speed_squared;
        angular_speed_squared += angular.x * angular.x +
                                 angular.y * angular.y +
                                 angular.z * angular.z;
        maximum_speed = std::max(maximum_speed,
                                 std::sqrt(speed_squared));
        const std::array<float, 13> components{
            point.x, point.y, point.z,
            state.orientation.x, state.orientation.y,
            state.orientation.z, state.orientation.w,
            velocity.x, velocity.y, velocity.z,
            angular.x, angular.y, angular.z};
        for (const float component : components) {
            state_hash ^= std::bit_cast<std::uint32_t>(component);
            state_hash *= 1099511628211ULL;
        }
    }
    std::cout << "invalid_spheres=" << invalid
              << " below_receiver=" << below_receiver
              << " outside_receiver=" << outside_receiver
              << " minimum_height=" << minimum_height
              << " maximum_speed=" << maximum_speed
              << " linear_rms="
              << std::sqrt(linear_speed_squared / sphere_count)
              << " angular_rms="
              << std::sqrt(angular_speed_squared / sphere_count)
              << " state_hash=" << std::hex << state_hash << std::dec << '\n';
    return invalid == 0U && below_receiver == 0U &&
                   (frame_count < 480U || outside_receiver == 0U)
        ? 0 : 1;
}
