// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "simulation/smoke_realtime.hpp"
#include "parallel_mater_smoke_grid_runtime.cu"

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

unsigned face_index(
    unsigned axis, unsigned x, unsigned y, unsigned z,
    unsigned n, unsigned h) {
    if (axis == 0U) return x + (n + 1U) * (y + h * z);
    if (axis == 1U) return x + n * (y + (h + 1U) * z);
    return x + n * (y + h * z);
}

unsigned face_count(unsigned axis, unsigned n, unsigned h) {
    if (axis == 0U) return (n + 1U) * h * n;
    if (axis == 1U) return n * (h + 1U) * n;
    return n * h * (n + 1U);
}

unsigned cell_index(
    unsigned x, unsigned y, unsigned z, unsigned n, unsigned h) {
    return x + n * (y + h * z);
}

float magnitude(const Vec3_0 &value) {
    return std::sqrt(value.x_0 * value.x_0 + value.y_0 * value.y_0 +
        value.z_0 * value.z_0);
}

struct Fixture {
    unsigned n{};
    unsigned h{};
    unsigned cells{};
    unsigned particles{};
    float spacing{0.1F};
    Vec3_0 *cell_velocity{};
    Vec3_0 *vorticity{};
    Vec3_0 *subgrid{};
    float *density{};
    float *density_next{};
    float *density_rate{};
    float *temperature{};
    float *temperature_next{};
    float *temperature_rate{};
    float *pressure_a{};
    float *pressure_b{};
    float *divergence{};
    float *strain{};
    std::array<float *, 3> faces{};
    std::array<float *, 3> face_scratch{};
    std::array<float *, 3> open{};
    std::array<float *, 3> wall{};
    Vec3_0 *positions{};
    Vec3_0 *previous{};
    Vec3_0 *velocities{};
    Vec3_0 *particle_vorticity{};
    float *ages{};
    float *thermal_lift{};
    float *particle_density{};
    float *particle_pressure{};
    unsigned *time_advances{};
    unsigned *status{};
    SmokeOptions_0 options{};
    SmokeGridField_0 grid{};

    Fixture(unsigned width, unsigned height, unsigned tracer_count)
        : n(width), h(height), cells(n * h * n), particles(tracer_count),
          cell_velocity(managed<Vec3_0>(cells)),
          vorticity(managed<Vec3_0>(cells)),
          subgrid(managed<Vec3_0>(cells)), density(managed<float>(cells)),
          density_next(managed<float>(cells)),
          density_rate(managed<float>(cells)),
          temperature(managed<float>(cells)),
          temperature_next(managed<float>(cells)),
          temperature_rate(managed<float>(cells)),
          pressure_a(managed<float>(cells)), pressure_b(managed<float>(cells)),
          divergence(managed<float>(cells)), strain(managed<float>(cells)),
          positions(managed<Vec3_0>(particles)),
          previous(managed<Vec3_0>(particles)),
          velocities(managed<Vec3_0>(particles)),
          particle_vorticity(managed<Vec3_0>(particles)),
          ages(managed<float>(particles)),
          thermal_lift(managed<float>(particles)),
          particle_density(managed<float>(particles)),
          particle_pressure(managed<float>(particles)),
          time_advances(managed<unsigned>(1U)), status(managed<unsigned>(1U)) {
        for (unsigned axis = 0U; axis < 3U; ++axis) {
            const unsigned count = face_count(axis, n, h);
            faces[axis] = managed<float>(count);
            face_scratch[axis] = managed<float>(count);
            open[axis] = managed<float>(count);
            wall[axis] = managed<float>(count);
            if (open[axis]) std::fill(open[axis], open[axis] + count, 1.0F);
        }
        options.capacity_0 = std::max(1U, particles);
        options.emitter_center_0 = {-100.0F, -100.0F, -100.0F};
        options.emitter_half_extents_0 = {0.05F, 0.05F};
        options.initial_velocity_0 = {1.0F, 0.0F, 0.0F};
        options.wind_2 = {1.0F, 0.0F, 0.0F};
        options.lifetime_0 = 1'000.0F;
        options.particle_radius_0 = 0.085F;
        options.buoyancy_0 = 0.12F;
        options.response_0 = 0.5F;
        options.rest_number_density_0 = 1.0F;
        options.pressure_stiffness_0 = 2.0F;
        options.viscosity_0 = 0.02F;
        options.maximum_speed_0 = 4.0F;
        options.grid_resolution_0 = n;
        options.grid_vertical_resolution_0 = h;
        options.grid_pressure_iterations_0 =
            smoke_realtime_grid_pressure_cycles;
        options.grid_kinematic_viscosity_0 = 1.5e-5F;
        options.grid_les_coefficient_0 = 0.12F;
        options.grid_minimum_0 = {0.0F, 0.0F, 0.0F};
        options.grid_edge_length_0 = float(n) * spacing;
        grid.n_2 = n;
        grid.height_2 = h;
        grid.minimum_1 = options.grid_minimum_0;
        grid.spacing_0 = spacing;
        grid.velocity_1 = cell_velocity;
        grid.density_1 = density;
        grid.pressure_1 = pressure_a;
        grid.face_1.m_data[0] = faces[0];
        grid.face_1.m_data[1] = faces[1];
        grid.face_1.m_data[2] = faces[2];
        grid.vorticity_1 = vorticity;
        grid.strain_1 = strain;
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 23>{cell_velocity, vorticity,
                 subgrid, density, density_next, density_rate, temperature,
                 temperature_next, temperature_rate, pressure_a, pressure_b,
                 divergence, strain, positions, previous, velocities,
                 particle_vorticity, ages, thermal_lift, particle_density,
                 particle_pressure, time_advances, status})
            if (pointer != nullptr) cudaFree(pointer);
        for (unsigned axis = 0U; axis < 3U; ++axis) {
            if (faces[axis]) cudaFree(faces[axis]);
            if (face_scratch[axis]) cudaFree(face_scratch[axis]);
            if (open[axis]) cudaFree(open[axis]);
            if (wall[axis]) cudaFree(wall[axis]);
        }
    }

    bool allocated() const {
        bool result = cell_velocity && vorticity && subgrid && density &&
            density_next && density_rate && temperature && temperature_next &&
            temperature_rate && pressure_a && pressure_b && divergence &&
            strain && time_advances && status;
        for (unsigned axis = 0U; axis < 3U; ++axis)
            result = result && faces[axis] && face_scratch[axis] &&
                open[axis] && wall[axis];
        return result && (particles == 0U || (positions && previous &&
            velocities && particle_vorticity && ages && thermal_lift &&
            particle_density && particle_pressure));
    }

    unsigned cell_blocks() const {
        return (cells + threads - 1U) / threads;
    }
    unsigned face_blocks(unsigned axis) const {
        return (face_count(axis, n, h) + threads - 1U) / threads;
    }

    void begin(unsigned profile = smoke_realtime_profile_version) {
        smoke_realtime_grid_begin<<<cell_blocks(), threads>>>(grid, options,
            smoke_realtime_timestep, profile, pressure_a, divergence,
            time_advances, status);
    }

    void divergence_pass() {
        smoke_realtime_grid_divergence<<<cell_blocks(), threads>>>(grid,
            options, divergence, smoke_realtime_timestep,
            smoke_realtime_profile_version, status);
    }

    void project_twelve() {
        float *source = pressure_a;
        float *destination = pressure_b;
        for (unsigned cycle = 0U;
             cycle < smoke_realtime_grid_pressure_cycles; ++cycle) {
            smoke_realtime_grid_pressure_cycle<<<cell_blocks(), threads>>>(
                grid, options, divergence, source, destination, open[0],
                open[1], open[2], smoke_realtime_timestep,
                smoke_realtime_profile_version, status);
            std::swap(source, destination);
        }
        grid.pressure_1 = source;
        for (unsigned axis = 0U; axis < 3U; ++axis)
            smoke_realtime_grid_project_face<<<face_blocks(axis), threads>>>(
                int(axis), grid, options, source, faces[axis], open[axis],
                wall[axis], smoke_realtime_timestep,
                smoke_realtime_profile_version, status);
    }

    void trace(unsigned first_emit_slot = 0U, unsigned emit_count = 0U,
               unsigned long long first_emit_serial = 0U) {
        if (particles == 0U) return;
        const unsigned blocks = (particles + threads - 1U) / threads;
        smoke_realtime_grid_trace<<<blocks, threads>>>(positions, previous,
            velocities, ages, thermal_lift, particle_density,
            particle_pressure, particle_vorticity, particles,
            first_emit_slot, emit_count, first_emit_serial, options, grid,
            smoke_realtime_timestep, smoke_realtime_profile_version, status);
    }

    void run_full_tick() {
        begin();
        smoke_realtime_grid_source<<<cell_blocks(), threads>>>(grid, options,
            density_rate, temperature_rate, density, temperature,
            smoke_realtime_timestep, smoke_realtime_profile_version, status);
        smoke_realtime_grid_advect_scalar<<<cell_blocks(), threads>>>(grid,
            options, density, density_next, smoke_realtime_timestep,
            smoke_realtime_profile_version, status);
        smoke_realtime_grid_advect_scalar<<<cell_blocks(), threads>>>(grid,
            options, temperature, temperature_next, smoke_realtime_timestep,
            smoke_realtime_profile_version, status);
        for (unsigned axis = 0U; axis < 3U; ++axis)
            smoke_realtime_grid_advect_face<<<face_blocks(axis), threads>>>(
                int(axis), grid, options, faces[axis], face_scratch[axis],
                smoke_realtime_timestep, smoke_realtime_profile_version,
                status);
        for (unsigned axis = 0U; axis < 3U; ++axis)
            grid.face_1.m_data[axis] = face_scratch[axis];
        grid.density_1 = density_next;
        for (unsigned axis = 0U; axis < 3U; ++axis)
            smoke_realtime_grid_face_forces<<<face_blocks(axis), threads>>>(
                int(axis), grid, options, face_scratch[axis], faces[axis],
                open[axis], wall[axis], strain, subgrid, temperature_next,
                smoke_realtime_timestep, Vec3_0{0.0F, -9.81F, 0.0F},
                smoke_realtime_profile_version, status);
        for (unsigned axis = 0U; axis < 3U; ++axis) {
            grid.face_1.m_data[axis] = faces[axis];
            smoke_realtime_grid_boundaries<<<face_blocks(axis), threads>>>(
                int(axis), grid, options, faces[axis], open[axis], wall[axis],
                smoke_realtime_timestep, smoke_realtime_profile_version,
                status);
        }
        divergence_pass();
        project_twelve();
        trace();
        std::swap(density, density_next);
        std::swap(temperature, temperature_next);
        grid.density_1 = density;
    }
};

std::pair<float, float> divergence_metrics(const Fixture &fixture) {
    double squared{};
    float maximum{};
    for (unsigned cell = 0U; cell < fixture.cells; ++cell) {
        const float dimensionless =
            std::abs(fixture.divergence[cell]) * smoke_realtime_timestep;
        squared += double(dimensionless) * dimensionless;
        maximum = std::max(maximum, dimensionless);
    }
    return {static_cast<float>(std::sqrt(squared / fixture.cells)), maximum};
}

void test_projection_obstacle_and_buoyancy() {
    Fixture smoke(8U, 8U, 1U);
    CHECK(smoke.allocated());
    // A compact high-gradient impulse exercises pressure rather than starting
    // from an already divergence-free constant field.
    smoke.faces[0][face_index(0U, 4U, 4U, 4U, smoke.n, smoke.h)] = 0.5F;
    smoke.begin();
    smoke.divergence_pass();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto before = divergence_metrics(smoke);
    smoke.project_twelve();
    smoke.divergence_pass();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto after = divergence_metrics(smoke);
    CHECK(*smoke.status == 0U);
    CHECK(*smoke.time_advances == 1U);
    CHECK(after.first < before.first);
    CHECK(after.first <= 0.025F);
    CHECK(after.second <= 0.1F);

    const unsigned obstacle =
        face_index(0U, 3U, 3U, 3U, smoke.n, smoke.h);
    smoke.open[0][obstacle] = 0.0F;
    smoke.wall[0][obstacle] = 0.25F;
    smoke.faces[0][obstacle] = 2.0F;
    smoke_realtime_grid_boundaries<<<smoke.face_blocks(0U), threads>>>(0,
        smoke.grid, smoke.options, smoke.faces[0], smoke.open[0],
        smoke.wall[0], smoke_realtime_timestep,
        smoke_realtime_profile_version, smoke.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(std::abs(smoke.faces[0][obstacle] - 0.25F) < 1.0e-7F);

    std::fill(smoke.density, smoke.density + smoke.cells, 1.0F);
    std::fill(smoke.faces[1],
        smoke.faces[1] + face_count(1U, smoke.n, smoke.h), 0.0F);
    smoke_realtime_grid_face_forces<<<smoke.face_blocks(1U), threads>>>(1,
        smoke.grid, smoke.options, smoke.faces[1], smoke.face_scratch[1],
        smoke.open[1], smoke.wall[1], smoke.strain, smoke.subgrid,
        smoke.temperature, smoke_realtime_timestep,
        Vec3_0{0.0F, -9.81F, 0.0F}, smoke_realtime_profile_version,
        smoke.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const unsigned buoyant =
        face_index(1U, 4U, 4U, 4U, smoke.n, smoke.h);
    CHECK(smoke.face_scratch[1][buoyant] > 0.0F);
    std::printf(
        "F8_SMOKE_GRID_PROJECT divergence_dt_rms_before=%.8f "
        "after=%.8f max_after=%.8f cycles=12 obstacle=pass buoyancy=pass\n",
        before.first, after.first, after.second);
}

void test_transport_cold_start_long_run_and_timing() {
    // Exercise the frozen desktop smoke workload: a 64 x 32 x 64 grid and
    // 4,500 tracers. The same fixture supplies the analytic source/transport
    // check, the 3,600-tick stability run and the complete-schedule timing.
    Fixture smoke(64U, 32U, 4'500U);
    CHECK(smoke.allocated());
    for (unsigned axis = 0U; axis < 3U; ++axis)
        std::fill(smoke.faces[axis], smoke.faces[axis] +
            face_count(axis, smoke.n, smoke.h), axis == 0U ? 1.0F : 0.0F);
    const unsigned source = cell_index(3U, 3U, 3U, smoke.n, smoke.h);
    smoke.density_rate[source] = 60.0F;
    smoke.grid.density_1 = smoke.density;
    smoke_realtime_grid_source<<<smoke.cell_blocks(), threads>>>(smoke.grid,
        smoke.options, smoke.density_rate, smoke.temperature_rate,
        smoke.density, smoke.temperature, smoke_realtime_timestep,
        smoke_realtime_profile_version, smoke.status);
    smoke_realtime_grid_advect_scalar<<<smoke.cell_blocks(), threads>>>(
        smoke.grid, smoke.options, smoke.density, smoke.density_next,
        smoke_realtime_timestep, smoke_realtime_profile_version, smoke.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    double mass{};
    double centroid{};
    for (unsigned z = 0U; z < smoke.n; ++z)
        for (unsigned y = 0U; y < smoke.h; ++y)
            for (unsigned x = 0U; x < smoke.n; ++x) {
                const float value = smoke.density_next[
                    cell_index(x, y, z, smoke.n, smoke.h)];
                mass += value;
                centroid += value * (double(x) + 0.5);
            }
    centroid /= std::max(mass, 1.0e-20);
    const double expected_centroid = 3.5 +
        smoke_realtime_timestep / smoke.spacing;
    CHECK(std::abs(mass - 1.0) <= 0.1);
    CHECK(std::abs(centroid - expected_centroid) <= 1.0);
    smoke.density_rate[source] = 0.0F;
    std::swap(smoke.density, smoke.density_next);
    smoke.grid.density_1 = smoke.density;

    for (unsigned particle = 0U; particle < smoke.particles; ++particle)
        smoke.positions[particle] = {0.25F, 0.25F, 0.25F};
    smoke.trace();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(std::abs(smoke.positions[0].x_0 -
                   (0.25F + smoke_realtime_timestep)) < 1.0e-5F);
    CHECK(smoke.ages[0] == smoke_realtime_timestep);

    smoke.options.emitter_center_0 = {0.5F, 0.6F, 0.7F};
    smoke.trace(17U, 1U, 42U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(smoke.positions[17].x_0 == 0.5F);
    CHECK(smoke.ages[17] == 0.0F);

    for (unsigned tick = 0U; tick < 120U; ++tick) {
        smoke.run_full_tick();
    }
    constexpr unsigned measured_ticks = 600U;
    std::vector<cudaEvent_t> starts(measured_ticks), ends(measured_ticks);
    for (unsigned tick = 0U; tick < measured_ticks; ++tick) {
        CHECK(cudaEventCreate(&starts[tick]) == cudaSuccess);
        CHECK(cudaEventCreate(&ends[tick]) == cudaSuccess);
        CHECK(cudaEventRecord(starts[tick]) == cudaSuccess);
        smoke.run_full_tick();
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
    for (unsigned tick = 720U; tick < 3'600U; ++tick) {
        smoke.run_full_tick();
    }
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::sort(milliseconds.begin(), milliseconds.end());
    const auto percentile = [&](float quantile) {
        return milliseconds[static_cast<std::size_t>(
            quantile * float(milliseconds.size() - 1U))];
    };
    float maximum_speed{};
    for (unsigned particle = 0U; particle < smoke.particles; ++particle) {
        CHECK(std::isfinite(smoke.positions[particle].x_0));
        CHECK(std::isfinite(smoke.positions[particle].y_0));
        CHECK(std::isfinite(smoke.positions[particle].z_0));
        maximum_speed = std::max(maximum_speed,
            magnitude(smoke.velocities[particle]));
    }
    CHECK(*smoke.status == 0U);
    CHECK(*smoke.time_advances == 3'600U);
    CHECK(maximum_speed <= smoke.options.maximum_speed_0 + 0.0001F);
    std::printf(
        "F8_SMOKE_GRID_TRANSPORT mass_error=%.8f centroid_error_cells=%.8f "
        "ticks=3600 max_speed=%.8f p50_ms=%.6f p95_ms=%.6f "
        "p99_ms=%.6f max_ms=%.6f\n",
        std::abs(mass - 1.0), std::abs(centroid - expected_centroid),
        maximum_speed, percentile(0.50F), percentile(0.95F),
        percentile(0.99F), milliseconds.back());

    const unsigned advances = *smoke.time_advances;
    smoke.begin(99U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*smoke.status & smoke_realtime_status_invalid_profile) != 0U);
    CHECK(*smoke.time_advances == advances);

    *smoke.status = 0U;
    smoke.options.grid_pressure_iterations_0 = 11U;
    smoke.begin();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*smoke.status & smoke_realtime_status_invalid_material) != 0U);
    CHECK(*smoke.time_advances == advances);
}

} // namespace

int main() {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    test_projection_obstacle_and_buoyancy();
    test_transport_cold_start_long_run_and_timing();
    if (failures == 0)
        std::puts("F8-16 grid smoke generated-Slang CUDA tests passed");
    return failures == 0 ? 0 : 1;
}
