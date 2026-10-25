// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

#include "simulation/geometry_storage.hpp"
#include "simulation/deformable_smoke_coupling_realtime.hpp"
#include "parallel_mater_deformable_smoke_coupling_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
constexpr unsigned count = 16U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(Vec3) == sizeof(Vec3_0));

template <typename T>
T *managed(std::size_t element_count) {
    T *result{};
    element_count = std::max<std::size_t>(element_count, 1U);
    if (cudaMallocManaged(&result, element_count * sizeof(T)) != cudaSuccess)
        return nullptr;
    std::fill(result, result + element_count, T{});
    return result;
}

Vec3 add_host(Vec3 left, Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 subtract_host(Vec3 left, Vec3 right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 multiply_host(Vec3 value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

float dot_host(Vec3 left, Vec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

float length_host(Vec3 value) {
    return std::sqrt(dot_host(value, value));
}

struct Fixture {
    Vec3 *first_positions{managed<Vec3>(count)};
    Vec3 *first_previous{managed<Vec3>(count)};
    Vec3 *first_velocities{managed<Vec3>(count)};
    float *first_inverse_masses{managed<float>(count)};
    float *first_correction_used{managed<float>(count)};
    unsigned *first_generations{managed<unsigned>(count)};

    Vec3 *second_positions{managed<Vec3>(count)};
    Vec3 *second_previous{managed<Vec3>(count)};
    Vec3 *second_velocities{managed<Vec3>(count)};
    float *second_inverse_masses{managed<float>(count)};
    float *second_correction_used{managed<float>(count)};
    unsigned *second_generations{managed<unsigned>(count)};

    unsigned *constraint_first{managed<unsigned>(count)};
    unsigned *constraint_second{managed<unsigned>(count)};
    unsigned *constraint_first_generations{managed<unsigned>(count)};
    unsigned *constraint_second_generations{managed<unsigned>(count)};
    unsigned *constraint_pairs{managed<unsigned>(count)};
    unsigned *constraint_modes{managed<unsigned>(count)};
    Vec3 *constraint_normals{managed<Vec3>(count)};
    float *constraint_distances{managed<float>(count)};
    float *constraint_frictions{managed<float>(count)};
    float *constraint_responses{managed<float>(count)};
    std::uint8_t *constraint_active{managed<std::uint8_t>(count)};

    Vec3 *first_position_deltas{managed<Vec3>(count)};
    Vec3 *second_position_deltas{managed<Vec3>(count)};
    Vec3 *first_impulses{managed<Vec3>(count)};
    Vec3 *second_impulses{managed<Vec3>(count)};
    unsigned *status{managed<unsigned>(1U)};

    Fixture() {
        constexpr std::array pairs{
            DeformableSmokeCouplingPair::cloth_soft_body,
            DeformableSmokeCouplingPair::cloth_rope,
            DeformableSmokeCouplingPair::cloth_smoke_particle,
            DeformableSmokeCouplingPair::cloth_smoke_particle,
            DeformableSmokeCouplingPair::cloth_smoke_grid,
            DeformableSmokeCouplingPair::cloth_smoke_grid,
            DeformableSmokeCouplingPair::soft_body_rope,
            DeformableSmokeCouplingPair::soft_body_rope,
            DeformableSmokeCouplingPair::soft_body_smoke_particle,
            DeformableSmokeCouplingPair::soft_body_smoke_particle,
            DeformableSmokeCouplingPair::soft_body_smoke_grid,
            DeformableSmokeCouplingPair::soft_body_smoke_grid,
            DeformableSmokeCouplingPair::rope_smoke_particle,
            DeformableSmokeCouplingPair::rope_smoke_particle,
            DeformableSmokeCouplingPair::rope_smoke_grid,
            DeformableSmokeCouplingPair::rope_smoke_grid,
        };
        constexpr std::array modes{
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::attachment,
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::drag,
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::drag,
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::attachment,
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::drag,
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::drag,
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::drag,
            DeformableSmokeCouplingMode::contact,
            DeformableSmokeCouplingMode::drag,
        };
        for (unsigned index = 0U; index < count; ++index) {
            first_inverse_masses[index] = 1.0F;
            second_inverse_masses[index] = 1.0F;
            first_generations[index] = 1U;
            second_generations[index] = 1U;
            constraint_first[index] = index;
            constraint_second[index] = index;
            constraint_first_generations[index] = 1U;
            constraint_second_generations[index] = 1U;
            constraint_pairs[index] = static_cast<unsigned>(pairs[index]);
            constraint_modes[index] = static_cast<unsigned>(modes[index]);
            constraint_normals[index] = {0.0F, 1.0F, 0.0F};
            constraint_active[index] = 1U;
            if (modes[index] == DeformableSmokeCouplingMode::contact) {
                first_previous[index] =
                    {0.2F * float(index), 0.015F, 0.0F};
                first_positions[index] =
                    {0.2F * float(index), -0.005F, 0.0F};
                first_velocities[index] = {0.0F, -1.2F, 0.0F};
                second_previous[index] = {0.2F * float(index), 0.0F, 0.0F};
                second_positions[index] = second_previous[index];
                constraint_distances[index] = 0.005F;
            } else if (modes[index] ==
                       DeformableSmokeCouplingMode::attachment) {
                first_positions[index] = {0.008F, 0.0F,
                    0.2F * float(index)};
                first_previous[index] = first_positions[index];
                second_positions[index] = {0.0F, 0.0F,
                    0.2F * float(index)};
                second_previous[index] = second_positions[index];
                constraint_responses[index] = 0.5F;
            } else {
                first_positions[index] = {0.2F * float(index), 0.0F, 0.0F};
                first_previous[index] = first_positions[index];
                second_positions[index] = first_positions[index];
                second_previous[index] = second_positions[index];
                first_velocities[index] = {
                    (index & 2U) == 0U ? 1.0F : -1.0F, 0.0F, 0.0F};
                constraint_responses[index] = 0.25F;
            }
        }
        // Exact 100:1 dynamic mass-ratio boundary for cloth/soft contact.
        second_inverse_masses[0] = 0.01F;
    }

    ~Fixture() {
        for (void *pointer : std::initializer_list<void *>{
                 first_positions, first_previous, first_velocities,
                 first_inverse_masses, first_correction_used,
                 first_generations, second_positions, second_previous,
                 second_velocities, second_inverse_masses,
                 second_correction_used, second_generations,
                 constraint_first, constraint_second,
                 constraint_first_generations,
                 constraint_second_generations, constraint_pairs,
                 constraint_modes, constraint_normals,
                 constraint_distances, constraint_frictions,
                 constraint_responses, constraint_active,
                 first_position_deltas, second_position_deltas,
                 first_impulses, second_impulses, status})
            if (pointer != nullptr) cudaFree(pointer);
    }

    bool allocated() const {
        return first_positions && first_previous && first_velocities &&
            first_inverse_masses && first_correction_used &&
            first_generations && second_positions && second_previous &&
            second_velocities && second_inverse_masses &&
            second_correction_used && second_generations &&
            constraint_first && constraint_second &&
            constraint_first_generations &&
            constraint_second_generations && constraint_pairs &&
            constraint_modes && constraint_normals &&
            constraint_distances && constraint_frictions &&
            constraint_responses && constraint_active &&
            first_position_deltas && second_position_deltas &&
            first_impulses && second_impulses && status;
    }

    void reset_correction_budget() {
        std::fill(first_correction_used, first_correction_used + count, 0.0F);
        std::fill(second_correction_used, second_correction_used + count, 0.0F);
    }

    void run(
        unsigned profile = deformable_smoke_coupling_profile_version,
        float dt = deformable_smoke_coupling_timestep) {
        for (unsigned sweep = 0U;
             sweep < deformable_smoke_coupling_sweeps; ++sweep) {
            deformable_smoke_coupling_project<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(first_positions),
                reinterpret_cast<Vec3_0 *>(first_previous),
                reinterpret_cast<Vec3_0 *>(first_velocities),
                first_inverse_masses, first_generations, count,
                reinterpret_cast<Vec3_0 *>(second_positions),
                reinterpret_cast<Vec3_0 *>(second_previous),
                reinterpret_cast<Vec3_0 *>(second_velocities),
                second_inverse_masses, second_generations, count,
                constraint_first, constraint_second,
                constraint_first_generations,
                constraint_second_generations, constraint_pairs,
                constraint_modes,
                reinterpret_cast<Vec3_0 *>(constraint_normals),
                constraint_distances, constraint_frictions,
                constraint_responses, constraint_active, count,
                0.1F, 2.9F, dt, profile, sweep,
                reinterpret_cast<Vec3_0 *>(first_position_deltas),
                reinterpret_cast<Vec3_0 *>(second_position_deltas),
                reinterpret_cast<Vec3_0 *>(first_impulses),
                reinterpret_cast<Vec3_0 *>(second_impulses), status);
            deformable_smoke_coupling_apply_first<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(first_positions),
                reinterpret_cast<Vec3_0 *>(first_velocities),
                first_inverse_masses, first_correction_used,
                first_generations, count, constraint_first,
                constraint_first_generations, constraint_active, count,
                reinterpret_cast<Vec3_0 *>(first_position_deltas),
                reinterpret_cast<Vec3_0 *>(first_impulses), 0.1F, 2.9F,
                dt, profile, sweep, status);
            deformable_smoke_coupling_apply_second<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(second_positions),
                reinterpret_cast<Vec3_0 *>(second_velocities),
                second_inverse_masses, second_correction_used,
                second_generations, count, constraint_second,
                constraint_second_generations, constraint_active, count,
                reinterpret_cast<Vec3_0 *>(second_position_deltas),
                reinterpret_cast<Vec3_0 *>(second_impulses), 0.1F, 2.9F,
                dt, profile, sweep, status);
        }
    }
};

void test_matrix_quality_and_direction() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    const Vec3 initial_first0 = fixture.first_positions[0];
    const Vec3 initial_second0 = fixture.second_positions[0];
    const Vec3 initial_velocity0 = fixture.first_velocities[0];
    const Vec3 initial_second_velocity0 = fixture.second_velocities[0];
    std::array<Vec3, count> smoke_contact_first{};
    std::array<Vec3, count> smoke_contact_second{};
    std::array<Vec3, count> smoke_drag_second_velocity{};
    for (unsigned index = 0U; index < count; ++index) {
        if (fixture.constraint_modes[index] ==
            static_cast<unsigned>(DeformableSmokeCouplingMode::contact) &&
            deformable_smoke_pair_has_smoke(
                static_cast<DeformableSmokeCouplingPair>(
                    fixture.constraint_pairs[index]))) {
            smoke_contact_first[index] = fixture.first_positions[index];
            smoke_contact_second[index] = fixture.second_positions[index];
        }
        if (fixture.constraint_modes[index] ==
            static_cast<unsigned>(DeformableSmokeCouplingMode::drag))
            smoke_drag_second_velocity[index] = fixture.second_velocities[index];
    }
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);

    float maximum_penetration{};
    float maximum_closing_speed{};
    unsigned tunnels{};
    for (unsigned index = 0U; index < count; ++index) {
        if (fixture.constraint_modes[index] !=
            static_cast<unsigned>(DeformableSmokeCouplingMode::contact))
            continue;
        const float separation = dot_host(subtract_host(
            fixture.first_positions[index], fixture.second_positions[index]),
            fixture.constraint_normals[index]);
        maximum_penetration = std::max(maximum_penetration,
            fixture.constraint_distances[index] - separation);
        const float closing = -dot_host(subtract_host(
            fixture.first_velocities[index],
            fixture.second_velocities[index]),
            fixture.constraint_normals[index]);
        maximum_closing_speed = std::max(maximum_closing_speed, closing);
        tunnels += separation < fixture.constraint_distances[index] - 1.0e-5F;
        if (deformable_smoke_pair_has_smoke(
            static_cast<DeformableSmokeCouplingPair>(
                fixture.constraint_pairs[index]))) {
            CHECK(length_host(subtract_host(fixture.first_positions[index],
                smoke_contact_first[index])) == 0.0F);
            CHECK(length_host(subtract_host(fixture.second_positions[index],
                smoke_contact_second[index])) > 0.0F);
        }
    }
    CHECK(maximum_penetration <= 1.0e-5F);
    CHECK(maximum_closing_speed <= 0.05F);
    CHECK(tunnels == 0U);

    const float first_mass = 1.0F / fixture.first_inverse_masses[0];
    const float second_mass = 1.0F / fixture.second_inverse_masses[0];
    const Vec3 initial_momentum = add_host(
        multiply_host(initial_velocity0, first_mass),
        multiply_host(initial_second_velocity0, second_mass));
    const Vec3 final_momentum = add_host(
        multiply_host(fixture.first_velocities[0], first_mass),
        multiply_host(fixture.second_velocities[0], second_mass));
    const float exchange = std::max(length_host(subtract_host(
        fixture.first_velocities[0], initial_velocity0)) * first_mass,
        1.0e-6F);
    const float linear_residual = length_host(subtract_host(
        final_momentum, initial_momentum)) / exchange;
    CHECK(linear_residual <= 1.0e-4F);
    CHECK(length_host(subtract_host(initial_first0,
        fixture.first_positions[0])) > 0.0F);
    CHECK(length_host(subtract_host(initial_second0,
        fixture.second_positions[0])) > 0.0F);

    float maximum_attachment_separation{};
    float maximum_drag_speed{};
    for (unsigned index = 0U; index < count; ++index) {
        if (fixture.constraint_modes[index] ==
            static_cast<unsigned>(DeformableSmokeCouplingMode::attachment))
            maximum_attachment_separation = std::max(
                maximum_attachment_separation, length_host(subtract_host(
                    fixture.first_positions[index],
                    fixture.second_positions[index])));
        if (fixture.constraint_modes[index] ==
            static_cast<unsigned>(DeformableSmokeCouplingMode::drag)) {
            maximum_drag_speed = std::max(maximum_drag_speed,
                length_host(subtract_host(fixture.first_velocities[index],
                    fixture.second_velocities[index])));
            CHECK(length_host(subtract_host(fixture.second_velocities[index],
                smoke_drag_second_velocity[index])) == 0.0F);
        }
    }
    CHECK(maximum_attachment_separation <= 0.00051F);
    CHECK(maximum_drag_speed <= 0.3165F);

    std::printf(
        "F8_DEFORMABLE_SMOKE_QUALITY pairs=9 modes=16 sweeps=4 "
        "penetration=%.8f closing_speed=%.8f tunnels=%u "
        "linear_residual=%.8f attachment_separation=%.8f "
        "drag_speed=%.8f mass_ratio=100 contact_direction=smoke-only "
        "drag_direction=deformable-only\n",
        std::max(maximum_penetration, 0.0F),
        std::max(maximum_closing_speed, 0.0F), tunnels, linear_residual,
        maximum_attachment_separation, maximum_drag_speed);
}

void test_generation_rebinding_and_profile_rejection() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    for (unsigned index = 1U; index < count; ++index)
        fixture.constraint_active[index] = 0U;
    fixture.first_generations[0] = 2U;
    const Vec3 stale_first = fixture.first_positions[0];
    const Vec3 stale_second = fixture.second_positions[0];
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(length_host(subtract_host(
        fixture.first_positions[0], stale_first)) == 0.0F);
    CHECK(length_host(subtract_host(
        fixture.second_positions[0], stale_second)) == 0.0F);

    fixture.constraint_first_generations[0] = 2U;
    fixture.reset_correction_budget();
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(fixture.first_positions[0].y > stale_first.y);
    CHECK(*fixture.status == 0U);

    const Vec3 rejected_first = fixture.first_positions[0];
    const Vec3 rejected_second = fixture.second_positions[0];
    fixture.reset_correction_budget();
    fixture.run(99U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status &
        deformable_smoke_coupling_status_invalid_profile) != 0U);
    CHECK(length_host(subtract_host(
        fixture.first_positions[0], rejected_first)) == 0.0F);
    CHECK(length_host(subtract_host(
        fixture.second_positions[0], rejected_second)) == 0.0F);
    std::puts("F8_DEFORMABLE_SMOKE_LIFECYCLE stale=ignored rebound=active "
              "invalid_profile=atomic topology=boundary-bound");
}

void test_correction_overflow_and_reset() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    for (unsigned index = 1U; index < count; ++index)
        fixture.constraint_active[index] = 0U;
    fixture.first_correction_used[0] = 0.025F;
    fixture.second_correction_used[0] = 0.025F;
    const Vec3 first_before = fixture.first_positions[0];
    const Vec3 second_before = fixture.second_positions[0];
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status &
        deformable_smoke_coupling_status_correction_exceeded) != 0U);
    CHECK(length_host(subtract_host(
        fixture.first_positions[0], first_before)) == 0.0F);
    CHECK(length_host(subtract_host(
        fixture.second_positions[0], second_before)) == 0.0F);

    *fixture.status = 0U;
    fixture.reset_correction_budget();
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(length_host(subtract_host(
        fixture.first_positions[0], first_before)) > 0.0F);
    CHECK(length_host(subtract_host(
        fixture.second_positions[0], second_before)) > 0.0F);
    std::puts("F8_DEFORMABLE_SMOKE_RECOVERY correction_overflow=reported "
              "reset=retry-passed");
}

void test_long_run_and_timing() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    cudaEvent_t start{}, stop{};
    CHECK(cudaEventCreate(&start) == cudaSuccess);
    CHECK(cudaEventCreate(&stop) == cudaSuccess);
    constexpr unsigned warmup = 120U;
    constexpr unsigned measured = 600U;
    std::vector<float> samples;
    samples.reserve(measured);
    for (unsigned tick = 0U; tick < warmup + measured; ++tick) {
        CHECK(cudaMemset(fixture.first_correction_used, 0,
            count * sizeof(float)) == cudaSuccess);
        CHECK(cudaMemset(fixture.second_correction_used, 0,
            count * sizeof(float)) == cudaSuccess);
        CHECK(cudaEventRecord(start) == cudaSuccess);
        fixture.run();
        CHECK(cudaEventRecord(stop) == cudaSuccess);
        CHECK(cudaEventSynchronize(stop) == cudaSuccess);
        float milliseconds{};
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop) == cudaSuccess);
        if (tick >= warmup) samples.push_back(milliseconds);
    }
    CHECK(*fixture.status == 0U);
    for (unsigned index = 0U; index < count; ++index) {
        CHECK(std::isfinite(fixture.first_positions[index].x));
        CHECK(std::isfinite(fixture.first_positions[index].y));
        CHECK(std::isfinite(fixture.first_positions[index].z));
        CHECK(std::isfinite(fixture.second_positions[index].x));
        CHECK(std::isfinite(fixture.second_positions[index].y));
        CHECK(std::isfinite(fixture.second_positions[index].z));
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](float fraction) {
        const std::size_t index = static_cast<std::size_t>(
            std::ceil(fraction * samples.size())) - 1U;
        return samples[index];
    };
    std::printf(
        "F8_DEFORMABLE_SMOKE_TIMING edges=16 constraints=16 warmup=%u "
        "samples=%u p50_ms=%.6f p95_ms=%.6f p99_ms=%.6f max_ms=%.6f\n",
        warmup, measured, percentile(0.50F), percentile(0.95F),
        percentile(0.99F), samples.back());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace

int main() {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no physical CUDA device");
        return 77;
    }
    test_matrix_quality_and_direction();
    test_generation_rebinding_and_profile_rejection();
    test_correction_overflow_and_reset();
    test_long_run_and_timing();
    if (failures == 0)
        std::puts("F8-19 deformable/smoke coupling physical CUDA tests passed");
    return failures == 0 ? 0 : 1;
}
