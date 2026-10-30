// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "simulation/smoke_realtime.hpp"
#include "parallel_mater_geometry_smoke_runtime.cu"

namespace {
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

template <typename T>
T *managed(std::size_t count) {
    T *result{};
    count = std::max<std::size_t>(count, 1U);
    if (cudaMallocManaged(&result, count * sizeof(T)) != cudaSuccess)
        return nullptr;
    std::fill(result, result + count, T{});
    return result;
}

float magnitude(const Vec3_0 &value) {
    return std::sqrt(value.x_0 * value.x_0 + value.y_0 * value.y_0 +
        value.z_0 * value.z_0);
}

bool finite(const Vec3_0 &value) {
    return std::isfinite(value.x_0) && std::isfinite(value.y_0) &&
        std::isfinite(value.z_0);
}

SmokeOptions_0 make_options(unsigned capacity) {
    SmokeOptions_0 result{};
    result.capacity_0 = capacity;
    result.emitter_half_extents_0 = {0.05F, 0.05F};
    result.initial_velocity_0 = {1.6F, 0.0F, 0.0F};
    result.wind_0 = {1.6F, 0.0F, 0.0F};
    result.particles_per_second_0 = 900.0F;
    result.lifetime_0 = 1'000.0F;
    result.particle_radius_0 = 0.085F;
    result.buoyancy_0 = 0.12F;
    result.response_0 = 0.5F;
    result.rest_number_density_0 = 1.0F;
    result.pressure_stiffness_0 = 2.0F;
    result.viscosity_0 = 0.02F;
    result.maximum_speed_0 = 4.0F;
    return result;
}

struct Fixture {
    unsigned capacity{};
    unsigned candidates_per_particle{};
    unsigned count{};
    Vec3_0 *positions{};
    Vec3_0 *previous{};
    Vec3_0 *velocities{};
    Vec3_0 *position_deltas{};
    Vec3_0 *accumulated_corrections{};
    float *ages{};
    float *thermal_lift{};
    float *densities{};
    float *pressures{};
    unsigned *candidate_counts{};
    unsigned *candidate_indices{};
    unsigned *time_advances{};
    unsigned *status{};
    SmokeOptions_0 options{};

    Fixture(unsigned particle_capacity, unsigned candidate_capacity)
        : capacity(particle_capacity),
          candidates_per_particle(candidate_capacity), count(capacity),
          positions(managed<Vec3_0>(capacity)),
          previous(managed<Vec3_0>(capacity)),
          velocities(managed<Vec3_0>(capacity)),
          position_deltas(managed<Vec3_0>(capacity)),
          accumulated_corrections(managed<Vec3_0>(capacity)),
          ages(managed<float>(capacity)),
          thermal_lift(managed<float>(capacity)),
          densities(managed<float>(capacity)),
          pressures(managed<float>(capacity)),
          candidate_counts(managed<unsigned>(capacity)),
          candidate_indices(managed<unsigned>(
              std::size_t{capacity} * candidate_capacity)),
          time_advances(managed<unsigned>(1U)),
          status(managed<unsigned>(1U)), options(make_options(capacity)) {}

    ~Fixture() {
        for (void *pointer : std::array<void *, 13>{positions, previous,
                 velocities, position_deltas, accumulated_corrections, ages,
                 thermal_lift, densities, pressures, candidate_counts,
                 candidate_indices, time_advances, status})
            if (pointer != nullptr) cudaFree(pointer);
    }

    bool allocated() const {
        return positions && previous && velocities && position_deltas &&
            accumulated_corrections && ages && thermal_lift && densities &&
            pressures && candidate_counts && candidate_indices &&
            time_advances && status;
    }

    void set_complete_candidates() {
        CHECK(candidates_per_particle >= count - 1U);
        for (unsigned particle = 0U; particle < count; ++particle) {
            unsigned written{};
            for (unsigned other = 0U; other < count; ++other)
                if (other != particle)
                    candidate_indices[particle * candidates_per_particle +
                                      written++] = other;
            candidate_counts[particle] = written;
        }
    }

    void emit(unsigned first_slot, unsigned emit_count,
              unsigned long long first_serial = 0U,
              unsigned profile = smoke_realtime_profile_version,
              float dt = smoke_realtime_timestep) {
        const unsigned blocks =
            std::max(1U, (emit_count + threads - 1U) / threads);
        smoke_realtime_particle_emit<<<blocks, threads>>>(positions, previous,
            velocities, ages, thermal_lift, densities, pressures, options,
            first_slot, emit_count, first_serial, dt, profile, status);
    }

    void run_tick(unsigned profile = smoke_realtime_profile_version,
                  float dt = smoke_realtime_timestep) {
        // Source collection is part of the fixed schedule even when this tick
        // emits no particles.
        emit(0U, 0U, 0U, profile, dt);
        const unsigned blocks = (count + threads - 1U) / threads;
        smoke_realtime_particle_predict<<<blocks, threads>>>(positions,
            previous, velocities, ages, thermal_lift, count, options,
            Vec3_0{0.0F, -9.81F, 0.0F}, dt, profile,
            accumulated_corrections, time_advances, status);
        for (unsigned sweep = 0U;
             sweep < smoke_realtime_particle_sweeps; ++sweep) {
            smoke_realtime_particle_density_pressure<<<blocks, threads>>>(
                positions, ages, count, options, candidate_counts,
                candidate_indices, candidates_per_particle, dt, profile,
                densities, pressures, position_deltas, status);
            smoke_realtime_particle_position_delta<<<blocks, threads>>>(
                positions, velocities, ages, count, options, candidate_counts,
                candidate_indices, candidates_per_particle, densities,
                pressures, accumulated_corrections, position_deltas, dt,
                profile, status);
            smoke_realtime_particle_apply_position_delta<<<blocks, threads>>>(
                positions, position_deltas, accumulated_corrections, ages,
                count, options, dt, profile, status);
        }
        smoke_realtime_particle_commit<<<blocks, threads>>>(positions,
            previous, velocities, ages, thermal_lift, count, options, dt,
            profile, status);
    }
};

void test_cold_source_pressure_and_rejection() {
    Fixture cold(8U, 8U);
    CHECK(cold.allocated());
    cold.count = 4U;
    std::fill(cold.ages, cold.ages + cold.capacity, cold.options.lifetime_0);
    cold.emit(0U, cold.count, 17U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    for (unsigned particle = 0U; particle < cold.count; ++particle) {
        CHECK(cold.ages[particle] == 0.0F);
        CHECK(finite(cold.positions[particle]));
        CHECK(magnitude(cold.velocities[particle]) <= 4.0001F);
    }
    cold.set_complete_candidates();
    cold.options.rest_number_density_0 = 2.0F;
    cold.run_tick();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*cold.status == 0U);
    CHECK(*cold.time_advances == 1U);
    float maximum_correction{};
    for (unsigned particle = 0U; particle < cold.count; ++particle) {
        CHECK(finite(cold.positions[particle]));
        CHECK(std::isfinite(cold.densities[particle]));
        CHECK(std::isfinite(cold.pressures[particle]));
        maximum_correction = std::max(maximum_correction,
            magnitude(cold.accumulated_corrections[particle]));
    }
    CHECK(maximum_correction <= 0.063751F);

    const Vec3_0 before = cold.positions[0];
    cold.run_tick(99U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*cold.status & smoke_realtime_status_invalid_profile) != 0U);
    CHECK(*cold.time_advances == 1U);
    CHECK(cold.positions[0].x_0 == before.x_0 &&
          cold.positions[0].y_0 == before.y_0 &&
          cold.positions[0].z_0 == before.z_0);
    std::printf(
        "F8_SMOKE_PARTICLE_COLD emitted=%u time_advances=%u "
        "maximum_correction_m=%.8f profile_rejection=pass\n",
        cold.count, *cold.time_advances, maximum_correction);
}

void test_high_velocity_long_run_and_timing() {
    constexpr unsigned count = 4'500U;
    Fixture smoke(count, 64U);
    CHECK(smoke.allocated());
    for (unsigned particle = 0U; particle < count; ++particle) {
        smoke.positions[particle] = {
            float(particle % 30U) * 0.3F,
            float(particle / 30U % 15U) * 0.3F,
            float(particle / 450U) * 0.3F};
        smoke.velocities[particle] = particle == 0U
            ? Vec3_0{4.0F, 0.0F, 0.0F}
            : Vec3_0{1.6F, 0.0F, 0.0F};
    }

    for (unsigned tick = 0U; tick < 120U; ++tick) smoke.run_tick();
    constexpr unsigned measured_ticks = 600U;
    std::vector<cudaEvent_t> starts(measured_ticks), ends(measured_ticks);
    for (unsigned tick = 0U; tick < measured_ticks; ++tick) {
        CHECK(cudaEventCreate(&starts[tick]) == cudaSuccess);
        CHECK(cudaEventCreate(&ends[tick]) == cudaSuccess);
        CHECK(cudaEventRecord(starts[tick]) == cudaSuccess);
        smoke.run_tick();
        CHECK(cudaEventRecord(ends[tick]) == cudaSuccess);
    }
    CHECK(cudaEventSynchronize(ends.back()) == cudaSuccess);
    std::vector<float> milliseconds(measured_ticks);
    for (unsigned tick = 0U; tick < measured_ticks; ++tick) {
        CHECK(cudaEventElapsedTime(
            &milliseconds[tick], starts[tick], ends[tick]) == cudaSuccess);
        cudaEventDestroy(starts[tick]);
        cudaEventDestroy(ends[tick]);
    }
    for (unsigned tick = 720U; tick < 3'600U; ++tick) smoke.run_tick();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaGetLastError() == cudaSuccess);
    std::sort(milliseconds.begin(), milliseconds.end());
    const auto percentile = [&](float quantile) {
        return milliseconds[static_cast<std::size_t>(
            quantile * float(milliseconds.size() - 1U))];
    };
    float maximum_speed{};
    std::vector<float> density_errors(count);
    for (unsigned particle = 0U; particle < count; ++particle) {
        CHECK(finite(smoke.positions[particle]));
        CHECK(finite(smoke.velocities[particle]));
        maximum_speed = std::max(maximum_speed,
            magnitude(smoke.velocities[particle]));
        density_errors[particle] =
            std::abs(smoke.densities[particle] - 1.0F);
    }
    std::sort(density_errors.begin(), density_errors.end());
    const float density_error_p95 = density_errors[static_cast<std::size_t>(
        0.95F * float(density_errors.size() - 1U))];
    CHECK(*smoke.status == 0U);
    CHECK(*smoke.time_advances == 3'600U);
    CHECK(maximum_speed <= smoke.options.maximum_speed_0 + 0.0001F);
    CHECK(density_error_p95 <= 0.25F);
    std::printf(
        "F8_SMOKE_PARTICLE_LONG particles=%u ticks=3600 "
        "density_error_p95=%.8f max_speed=%.8f p50_ms=%.6f "
        "p95_ms=%.6f p99_ms=%.6f max_ms=%.6f\n",
        count, density_error_p95, maximum_speed, percentile(0.50F),
        percentile(0.95F), percentile(0.99F), milliseconds.back());
}

} // namespace

int main() {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    test_cold_source_pressure_and_rejection();
    test_high_velocity_long_run_and_timing();
    if (failures == 0)
        std::puts("F8-16 particle smoke generated-Slang CUDA tests passed");
    return failures == 0 ? 0 : 1;
}
