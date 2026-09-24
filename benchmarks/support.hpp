// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

namespace parallel_mater::benchmark {

[[nodiscard]] inline bool require(Status status, const char *operation) {
    if (status) {
        return true;
    }
    std::cerr << operation << ": "
              << (status.message != nullptr ? status.message : "unknown")
              << '\n';
    return false;
}

// The caller sets mesh capacity because GLB proxies and DUMP mesh reuse differ.
[[nodiscard]] inline bool prepare_world(const gallery::SceneDefinition &scene,
                                        std::uint32_t mesh_capacity,
                                        World &world,
                                        gallery::SceneInstance &instance) {
    const WorldOptions options{
        .rigid_body_capacity =
            static_cast<std::uint32_t>(scene.rigid_bodies.size()),
        .triangle_mesh_capacity = mesh_capacity};
    return require(World::create(options, world), "create world") &&
           require(gallery::instantiate_scene(scene, world, instance),
                   "instantiate scene");
}

[[nodiscard]] inline StepOptions standard_step_options(bool collect_timings) {
    return {.timestep = 1.0F / 60.0F, .substeps = 4U,
            .gravity = {0.0F, -9.81F, 0.0F},
            .collect_kernel_timings = collect_timings};
}

enum class PercentileMode { nearest_sample, nearest_rank };

struct Samples {
    std::vector<double> values{};

    void add(double value) { values.push_back(value); }
    void clear() { values.clear(); }

    [[nodiscard]] double mean() const {
        assert(!values.empty());
        return std::accumulate(values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());
    }

    [[nodiscard]] double percentile(
        double fraction,
        PercentileMode mode = PercentileMode::nearest_sample) {
        assert(!values.empty() && fraction >= 0.0 && fraction <= 1.0);
        assert(mode != PercentileMode::nearest_rank || fraction > 0.0);
        const auto rank = static_cast<std::size_t>(
            mode == PercentileMode::nearest_rank
                ? std::ceil(fraction * static_cast<double>(values.size())) - 1.0
                : std::round(fraction * static_cast<double>(values.size() - 1U)));
        std::sort(values.begin(), values.end());
        return values[rank];
    }

    void print_summary(std::ostream &output, const char *name) {
        output << name << " median=" << percentile(0.5)
               << " p5=" << percentile(0.05)
               << " p95=" << percentile(0.95) << " ms\n";
    }
};

constexpr std::size_t k_timing_stage_count = 10U;
constexpr std::size_t k_gpu_total_stage = 8U;
using TimingSamples = std::array<Samples, k_timing_stage_count>;

[[nodiscard]] inline std::array<double, k_timing_stage_count>
timing_values(const WorldStepTimings &timing, double wall_milliseconds) {
    return {timing.rigid_integration.total_milliseconds,
            timing.rigid_world_bounds.total_milliseconds,
            timing.rigid_pair_filter.total_milliseconds,
            timing.rigid_pair_compaction.total_milliseconds,
            timing.rigid_leaf_pair_generation.total_milliseconds,
            timing.rigid_contact_evaluation.total_milliseconds,
            timing.rigid_contact_solve.total_milliseconds,
            timing.rigid_input_clear.total_milliseconds,
            timing.total_gpu_milliseconds, wall_milliseconds};
}

inline void add_timing_samples(TimingSamples &samples,
                               const WorldStepTimings &timing,
                               double wall_milliseconds) {
    const auto values = timing_values(timing, wall_milliseconds);
    for (std::size_t index = 0U; index < values.size(); ++index) {
        samples[index].add(values[index]);
    }
}

[[nodiscard]] inline bool measure_step(World &world, StepOptions options,
                                       const char *operation,
                                       WorldStepTimings &timing,
                                       double &wall_milliseconds) {
    const auto begin = std::chrono::steady_clock::now();
    if (!require(world.step(options), operation)) {
        return false;
    }
    wall_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    if (!require(world.collect_step_timings(timing), "collect timings")) {
        return false;
    }
    if (!timing.available) {
        std::cerr << "collect timings: unavailable for measured frame\n";
        return false;
    }
    return true;
}

} // namespace parallel_mater::benchmark
