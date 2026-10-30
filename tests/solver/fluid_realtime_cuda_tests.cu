// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "simulation/fluid_realtime.hpp"
#include "simulation/types.hpp"
#include "parallel_mater_fluid_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(Vec3) == sizeof(Vec3_0));
static_assert(sizeof(FluidOptions) == sizeof(FluidOptions_0));
static_assert(sizeof(SpatialLeafMetadata) ==
              sizeof(FluidSpatialLeafMetadata_0));

template <typename T>
T *managed(std::size_t count) {
    T *result{};
    if (cudaMallocManaged(&result,
            std::max<std::size_t>(count, 1U) * sizeof(T)) != cudaSuccess)
        return nullptr;
    std::fill(result, result + count, T{});
    return result;
}

FluidOptions realtime_options(unsigned capacity) {
    FluidOptions result{};
    result.capacity = capacity;
    result.particle_radius = 0.025F;
    result.rest_density = 1'000.0F;
    result.support_radius = 0.10F;
    result.solver_iterations = 97U; // realtime profile must ignore this
    result.maximum_neighbors = 128U;
    result.repulsion = 200.0F;
    result.viscosity = 12.0F;
    result.velocity_damping = 0.0F;
    result.maximum_speed = 2.9F;
    result.normal_damping = 6.0F;
    result.rest_particle_volume = 0.05F * 0.05F * 0.05F;
    result.realtime_commit_velocity_scale = capacity > 1U
        ? 1.0F - fluid_realtime_solver_damping : 1.0F;
    return result;
}

FluidOptions_0 generated_options(const FluidOptions &source) {
    FluidOptions_0 result{};
    std::memcpy(&result, &source, sizeof(result));
    return result;
}

float squared_length(const Vec3 &value) {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

float density_kernel(float squared, float support) {
    const float h2 = support * support;
    if (squared >= h2) return 0.0F;
    const float q2 = std::max(0.0F, 1.0F - squared / h2);
    return 1.56668147106F * q2 * q2 * q2 /
        (support * h2);
}

struct Fixture {
    unsigned capacity{};
    unsigned candidates_per_query{};
    std::size_t candidate_slots{};
    Vec3 *positions{};
    Vec3 *velocities{};
    Vec3 *previous{};
    Vec3 *accumulated{};
    Vec3 *position_deltas{};
    Vec3 *velocity_deltas{};
    float *densities{};
    float *lambdas{};
    SpatialLeafMetadata *metadata{};
    std::uint8_t *admission{};
    unsigned *query_sources{};
    unsigned *filtered_counts{};
    unsigned *filtered_sources{};
    unsigned *count{};
    unsigned *time_advances{};
    unsigned *neighbor_overflow{};
    unsigned *maximum_neighbor_count{};
    unsigned *status{};

    Fixture(unsigned capacity_value, unsigned candidate_capacity)
        : capacity(capacity_value),
          candidates_per_query(candidate_capacity),
          candidate_slots(std::size_t(capacity_value) * candidate_capacity),
          positions(managed<Vec3>(capacity)),
          velocities(managed<Vec3>(capacity)),
          previous(managed<Vec3>(capacity)),
          accumulated(managed<Vec3>(capacity)),
          position_deltas(managed<Vec3>(capacity)),
          velocity_deltas(managed<Vec3>(capacity)),
          densities(managed<float>(capacity)),
          lambdas(managed<float>(capacity)),
          metadata(managed<SpatialLeafMetadata>(capacity)),
          admission(managed<std::uint8_t>(capacity)),
          query_sources(managed<unsigned>(capacity)),
          filtered_counts(managed<unsigned>(capacity)),
          filtered_sources(managed<unsigned>(candidate_slots)),
          count(managed<unsigned>(1U)),
          time_advances(managed<unsigned>(1U)),
          neighbor_overflow(managed<unsigned>(1U)),
          maximum_neighbor_count(managed<unsigned>(1U)),
          status(managed<unsigned>(1U)) {
        for (unsigned index = 0U; index < capacity; ++index) {
            metadata[index].identity = {fluid_spatial_family, 7U, 3U, 11U,
                fluid_spatial_particle_primitive, 1000U + index, 1U};
            metadata[index].payload_ref = index;
            metadata[index].flags = spatial_leaf_retained;
            admission[index] = 1U;
            query_sources[index] = index;
        }
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 19>{positions, velocities,
                 previous, accumulated, position_deltas, velocity_deltas,
                 densities, lambdas, metadata, admission, query_sources,
                 filtered_counts, filtered_sources, count, time_advances,
                 neighbor_overflow, maximum_neighbor_count, status, nullptr})
            if (pointer != nullptr) cudaFree(pointer);
    }

    bool allocated() const {
        return positions && velocities && previous && accumulated &&
            position_deltas && velocity_deltas && densities && lambdas &&
            metadata && admission && query_sources && filtered_counts &&
            filtered_sources && count && time_advances && neighbor_overflow &&
            maximum_neighbor_count && status;
    }

    void configure_candidates(unsigned live, bool complete = true) {
        *count = live;
        for (unsigned query = 0U; query < capacity; ++query) {
            filtered_counts[query] = 0U;
            if (!complete || query >= live) continue;
            const auto base = std::size_t(query) * candidates_per_query;
            for (unsigned other = 0U; other < live; ++other) {
                if (other == query) continue;
                CHECK(filtered_counts[query] < candidates_per_query);
                filtered_sources[base + filtered_counts[query]++] = other;
            }
        }
    }

    void lattice(unsigned side, float spacing, float speed = 0.0F) {
        const unsigned live = side * side * side;
        CHECK(live <= capacity);
        const float center = 0.5F * float(side - 1U);
        for (unsigned z = 0U; z < side; ++z)
            for (unsigned y = 0U; y < side; ++y)
                for (unsigned x = 0U; x < side; ++x) {
                    const unsigned index = (z * side + y) * side + x;
                    positions[index] = {
                        (float(x) - center) * spacing,
                        (float(y) - center) * spacing,
                        (float(z) - center) * spacing};
                    const float sign = ((x + y + z) & 1U) != 0U
                        ? 1.0F : -1.0F;
                    velocities[index] = {sign * speed, 0.0F, 0.0F};
                }
        configure_candidates(live);
        *time_advances = 0U;
        *neighbor_overflow = 0U;
        *maximum_neighbor_count = 0U;
        *status = 0U;
    }

    void run_tick(const FluidOptions &host_options,
                  unsigned profile = fluid_realtime_profile_version) {
        const auto options = generated_options(host_options);
        const unsigned blocks = (capacity + threads - 1U) / threads;
        *time_advances = 0U;
        fluid_realtime_predict<<<blocks, threads>>>(
            reinterpret_cast<Vec3_0 *>(positions),
            reinterpret_cast<Vec3_0 *>(velocities),
            reinterpret_cast<Vec3_0 *>(previous), count, capacity, options,
            Vec3_0{}, fluid_realtime_timestep, profile,
            reinterpret_cast<Vec3_0 *>(accumulated), time_advances, status);
        for (unsigned sweep = 0U;
             sweep < fluid_realtime_density_sweeps; ++sweep) {
            fluid_realtime_density_lambda<<<blocks, threads>>>(
                reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
                admission, capacity, query_sources, *count,
                reinterpret_cast<Vec3_0 *>(positions), count, capacity,
                filtered_counts, filtered_sources, candidates_per_query,
                options, fluid_realtime_timestep, profile, densities, lambdas,
                sweep == 0U ? 1U : 0U, neighbor_overflow,
                maximum_neighbor_count,
                status);
            fluid_realtime_position_delta<<<blocks, threads>>>(
                reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
                admission, capacity, query_sources, *count,
                reinterpret_cast<Vec3_0 *>(positions), count, capacity,
                filtered_counts, filtered_sources, candidates_per_query,
                options, fluid_realtime_timestep, profile, lambdas,
                reinterpret_cast<Vec3_0 *>(accumulated),
                reinterpret_cast<Vec3_0 *>(position_deltas), status);
            fluid_realtime_apply_position_delta<<<blocks, threads>>>(
                reinterpret_cast<Vec3_0 *>(positions),
                reinterpret_cast<Vec3_0 *>(position_deltas),
                reinterpret_cast<Vec3_0 *>(accumulated), count, capacity,
                fluid_realtime_timestep, profile, options, status);
        }
        fluid_realtime_viscosity_delta<<<blocks, threads>>>(
            reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
            admission, capacity, query_sources, *count,
            reinterpret_cast<Vec3_0 *>(positions),
            reinterpret_cast<Vec3_0 *>(velocities), count, capacity,
            filtered_counts, filtered_sources, candidates_per_query, options,
            fluid_realtime_timestep, profile, densities,
            reinterpret_cast<Vec3_0 *>(velocity_deltas), status);
        fluid_realtime_commit<<<blocks, threads>>>(
            reinterpret_cast<Vec3_0 *>(positions),
            reinterpret_cast<Vec3_0 *>(previous),
            reinterpret_cast<Vec3_0 *>(velocities),
            reinterpret_cast<Vec3_0 *>(velocity_deltas), count, capacity,
            options, fluid_realtime_timestep, profile, status);
    }

    std::vector<float> compression_errors(
        const FluidOptions &options) const {
        std::vector<float> result(*count);
        const float mass = options.rest_density *
            options.rest_particle_volume;
        for (unsigned particle = 0U; particle < *count; ++particle) {
            float density = mass * density_kernel(
                0.0F, options.support_radius);
            for (unsigned other = 0U; other < *count; ++other) {
                if (other == particle) continue;
                const Vec3 delta{positions[particle].x - positions[other].x,
                    positions[particle].y - positions[other].y,
                    positions[particle].z - positions[other].z};
                density += mass * density_kernel(
                    squared_length(delta), options.support_radius);
            }
            result[particle] = std::max(
                density / options.rest_density - 1.0F, 0.0F);
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    Vec3 center() const {
        Vec3 result{};
        for (unsigned index = 0U; index < *count; ++index) {
            result.x += positions[index].x;
            result.y += positions[index].y;
            result.z += positions[index].z;
        }
        const float inverse = 1.0F / float(*count);
        result.x *= inverse;
        result.y *= inverse;
        result.z *= inverse;
        return result;
    }

    float kinetic_energy(const FluidOptions &options) const {
        const float mass = options.rest_density *
            options.rest_particle_volume;
        float result{};
        for (unsigned index = 0U; index < *count; ++index)
            result += 0.5F * mass * squared_length(velocities[index]);
        return result;
    }
};

void test_profile_and_compression() {
    Fixture fixture(64U, 64U);
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    auto options = realtime_options(fixture.capacity);
    fixture.lattice(4U, 0.045F);
    const auto initial_errors = fixture.compression_errors(options);
    const float initial_p95 =
        initial_errors[initial_errors.size() * 95U / 100U];
    const Vec3 before = fixture.positions[0];
    fixture.run_tick(options, 0U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & fluid_realtime_status_invalid_profile) != 0U);
    CHECK(*fixture.time_advances == 0U);
    CHECK(fixture.positions[0].x == before.x);

    *fixture.status = 0U;
    fixture.run_tick(options);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(*fixture.time_advances == 1U);
    const auto errors = fixture.compression_errors(options);
    const float p95 = errors[errors.size() * 95U / 100U];
    const float maximum = errors.back();
    float maximum_correction{};
    for (unsigned index = 0U; index < *fixture.count; ++index) {
        maximum_correction = std::max(maximum_correction,
            std::sqrt(squared_length(fixture.accumulated[index])));
        CHECK(std::isfinite(fixture.positions[index].x));
        CHECK(std::isfinite(fixture.velocities[index].x));
    }
    std::printf("F8_FLUID_60HZ_COMPRESSION particles=%u initial_p95=%.9g "
                "final_p95=%.9g final_maximum=%.9g "
                "maximum_correction_m=%.9g advances=%u sweeps=%u\n",
                *fixture.count, initial_p95, p95, maximum,
                maximum_correction, *fixture.time_advances,
                fluid_realtime_density_sweeps);
    CHECK(initial_p95 > 0.30F);
    CHECK(p95 <= 0.15F);
    CHECK(maximum <= 0.30F);
    CHECK(maximum_correction <=
          fluid_realtime_maximum_correction_fraction *
              options.support_radius + 1.0e-6F);
    CHECK(*fixture.count == 64U);
}

void test_viscosity_boundary_mutation_and_long_run() {
    Fixture fixture(65U, 65U);
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    auto options = realtime_options(fixture.capacity);
    options.repulsion = 0.0F;
    fixture.lattice(4U, 0.05F, 0.30F);
    const Vec3 initial_center = fixture.center();
    const float initial_energy = fixture.kinetic_energy(options);
    fixture.run_tick(options);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    const float first_energy = fixture.kinetic_energy(options);
    CHECK(first_energy < 0.95F * initial_energy);

    // Outflow and source admission happen at a tick boundary. The solve sees
    // one compacted count/list and performs no retry build or extra advance.
    fixture.configure_candidates(63U);
    fixture.positions[63] = {0.30F, 0.0F, 0.0F};
    fixture.velocities[63] = {};
    fixture.positions[64] = {-0.30F, 0.0F, 0.0F};
    fixture.velocities[64] = {};
    fixture.configure_candidates(65U);
    fixture.run_tick(options);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(*fixture.time_advances == 1U);
    CHECK(*fixture.count == 65U);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK(cudaEventCreate(&start) == cudaSuccess);
    CHECK(cudaEventCreate(&stop) == cudaSuccess);
    std::array<float, 720U> milliseconds{};
    float maximum_energy = initial_energy;
    unsigned leaked{};
    for (unsigned tick = 0U; tick < milliseconds.size(); ++tick) {
        CHECK(cudaEventRecord(start) == cudaSuccess);
        fixture.run_tick(options);
        CHECK(cudaEventRecord(stop) == cudaSuccess);
        CHECK(cudaEventSynchronize(stop) == cudaSuccess);
        CHECK(cudaEventElapsedTime(&milliseconds[tick], start, stop) ==
              cudaSuccess);
        CHECK(*fixture.status == 0U);
        CHECK(*fixture.time_advances == 1U);
        maximum_energy = std::max(maximum_energy,
            fixture.kinetic_energy(options));
        leaked = 0U;
        for (unsigned index = 0U; index < *fixture.count; ++index) {
            const auto &position = fixture.positions[index];
            if (std::abs(position.x) > 0.5F ||
                std::abs(position.y) > 0.5F ||
                std::abs(position.z) > 0.5F)
                ++leaked;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    const Vec3 final_center = fixture.center();
    const Vec3 center_delta{final_center.x - initial_center.x,
        final_center.y - initial_center.y,
        final_center.z - initial_center.z};
    const float center_drift = std::sqrt(squared_length(center_delta));
    auto sorted = milliseconds;
    std::sort(sorted.begin(), sorted.end());
    const float p50 = sorted[sorted.size() * 50U / 100U];
    const float p95 = sorted[sorted.size() * 95U / 100U];
    const float p99 = sorted[sorted.size() * 99U / 100U];
    std::printf("F8_FLUID_60HZ_WINDOW ticks=%zu particles=%u outflows=1 "
                "source_admissions=2 "
                "viscosity_energy_ratio=%.9g maximum_energy_ratio=%.9g "
                "center_drift_support=%.9g leaked=%u p50_ms=%.9g "
                "p95_ms=%.9g p99_ms=%.9g maximum_ms=%.9g\n",
                milliseconds.size(), *fixture.count,
                first_energy / initial_energy,
                maximum_energy / initial_energy,
                center_drift / options.support_radius, leaked,
                p50, p95, p99, sorted.back());
    CHECK(maximum_energy <= 1.05F * initial_energy);
    CHECK(center_drift <= 0.05F * options.support_radius);
    CHECK(leaked <= (*fixture.count + 99U) / 100U);
    CHECK(*fixture.count == 65U);
}

} // namespace

int main() {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    test_profile_and_compression();
    test_viscosity_boundary_mutation_and_long_run();
    if (failures == 0)
        std::puts("F8-12 physical fluid realtime kernels passed");
    return failures == 0 ? 0 : 1;
}
