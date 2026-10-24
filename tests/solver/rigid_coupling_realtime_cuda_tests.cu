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
#include "simulation/rigid_coupling_realtime.hpp"
#include "parallel_mater_rigid_coupling_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
constexpr unsigned count = 8U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(Vec3) == sizeof(Vec3_0));
static_assert(sizeof(BodyParameters) == sizeof(BodyParameters_0));
static_assert(sizeof(RigidBodyState) == sizeof(RigidBodyState_0));

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

Vec3 cross_host(Vec3 left, Vec3 right) {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

float length_host(Vec3 value) {
    return std::sqrt(dot_host(value, value));
}

struct Fixture {
    Vec3 *endpoint_positions{managed<Vec3>(count)};
    Vec3 *endpoint_previous{managed<Vec3>(count)};
    Vec3 *endpoint_velocities{managed<Vec3>(count)};
    float *endpoint_inverse_masses{managed<float>(count)};
    float *endpoint_correction_used{managed<float>(count)};
    unsigned *endpoint_generations{managed<unsigned>(count)};

    BodyParameters *body_parameters{managed<BodyParameters>(count)};
    RigidBodyState *body_previous{managed<RigidBodyState>(count)};
    RigidBodyState *body_states{managed<RigidBodyState>(count)};
    float *body_correction_used{managed<float>(count)};
    unsigned *body_generations{managed<unsigned>(count)};

    unsigned *constraint_endpoints{managed<unsigned>(count)};
    unsigned *constraint_bodies{managed<unsigned>(count)};
    unsigned *constraint_endpoint_generations{managed<unsigned>(count)};
    unsigned *constraint_body_generations{managed<unsigned>(count)};
    unsigned *constraint_families{managed<unsigned>(count)};
    unsigned *constraint_modes{managed<unsigned>(count)};
    Vec3 *constraint_normals{managed<Vec3>(count)};
    Vec3 *constraint_body_local_points{managed<Vec3>(count)};
    float *constraint_contact_distances{managed<float>(count)};
    float *constraint_frictions{managed<float>(count)};
    float *constraint_responses{managed<float>(count)};
    std::uint8_t *constraint_active{managed<std::uint8_t>(count)};

    Vec3 *endpoint_position_deltas{managed<Vec3>(count)};
    Vec3 *body_position_deltas{managed<Vec3>(count)};
    Vec3 *endpoint_impulses{managed<Vec3>(count)};
    Vec3 *body_angular_impulses{managed<Vec3>(count)};
    unsigned *status{managed<unsigned>(1U)};

    Fixture() {
        constexpr std::array families{
            RigidCouplingFamily::cloth,
            RigidCouplingFamily::soft_body,
            RigidCouplingFamily::rope,
            RigidCouplingFamily::rope,
            RigidCouplingFamily::smoke_particle,
            RigidCouplingFamily::smoke_particle,
            RigidCouplingFamily::smoke_grid,
            RigidCouplingFamily::smoke_grid,
        };
        constexpr std::array modes{
            RigidCouplingMode::contact,
            RigidCouplingMode::contact,
            RigidCouplingMode::contact,
            RigidCouplingMode::attachment,
            RigidCouplingMode::contact,
            RigidCouplingMode::drag,
            RigidCouplingMode::contact,
            RigidCouplingMode::drag,
        };
        for (unsigned index = 0U; index < count; ++index) {
            endpoint_inverse_masses[index] = 1.0F;
            endpoint_generations[index] = 1U;
            body_parameters[index].motion = MotionType::dynamic;
            body_parameters[index].inverse_mass = 1.0F;
            body_parameters[index].inverse_inertia_local =
                {1.0F, 1.0F, 1.0F};
            body_parameters[index].maximum_linear_speed = 20.0F;
            body_parameters[index].maximum_angular_speed = 31.0F;
            body_previous[index].orientation.w = 1.0F;
            body_states[index] = body_previous[index];
            body_generations[index] = 1U;
            constraint_endpoints[index] = index;
            constraint_bodies[index] = index;
            constraint_endpoint_generations[index] = 1U;
            constraint_body_generations[index] = 1U;
            constraint_families[index] =
                static_cast<unsigned>(families[index]);
            constraint_modes[index] = static_cast<unsigned>(modes[index]);
            constraint_normals[index] = {0.0F, 1.0F, 0.0F};
            constraint_active[index] = 1U;
            if (modes[index] == RigidCouplingMode::contact) {
                endpoint_previous[index] =
                    {0.2F * float(index), 0.015F, 0.0F};
                endpoint_positions[index] =
                    {0.2F * float(index), -0.005F, 0.0F};
                endpoint_velocities[index] = {0.0F, -1.2F, 0.0F};
                body_previous[index].position.x = 0.2F * float(index);
                body_states[index] = body_previous[index];
                constraint_contact_distances[index] = 0.005F;
            }
        }
        // Exact 100:1 dynamic mass-ratio boundary plus off-centre impact.
        body_parameters[0].inverse_mass = 0.01F;
        body_parameters[0].inverse_inertia_local = {0.01F, 0.01F, 0.01F};
        constraint_body_local_points[0] = {0.02F, 0.0F, 0.0F};
        endpoint_previous[0].x = 0.02F;
        endpoint_positions[0].x = 0.02F;

        // Rope attachment and both supported smoke drag representations.
        endpoint_positions[3] = {0.008F, 0.0F, 0.0F};
        endpoint_previous[3] = endpoint_positions[3];
        constraint_responses[3] = 0.5F;
        endpoint_velocities[5] = {1.0F, 0.0F, 0.0F};
        endpoint_previous[5] = endpoint_positions[5];
        constraint_responses[5] = 0.25F;
        endpoint_velocities[7] = {-1.0F, 0.0F, 0.0F};
        endpoint_previous[7] = endpoint_positions[7];
        constraint_responses[7] = 0.25F;
    }

    ~Fixture() {
        for (void *pointer : std::initializer_list<void *>{
                 endpoint_positions, endpoint_previous, endpoint_velocities,
                 endpoint_inverse_masses, endpoint_correction_used,
                 endpoint_generations, body_parameters, body_previous,
                 body_states, body_correction_used, body_generations,
                 constraint_endpoints, constraint_bodies,
                 constraint_endpoint_generations,
                 constraint_body_generations, constraint_families,
                 constraint_modes, constraint_normals,
                 constraint_body_local_points, constraint_contact_distances,
                 constraint_frictions, constraint_responses,
                 constraint_active, endpoint_position_deltas,
                 body_position_deltas, endpoint_impulses,
                 body_angular_impulses, status})
            if (pointer != nullptr) cudaFree(pointer);
    }

    bool allocated() const {
        return endpoint_positions && endpoint_previous && endpoint_velocities &&
            endpoint_inverse_masses && endpoint_correction_used &&
            endpoint_generations && body_parameters && body_previous &&
            body_states && body_correction_used && body_generations &&
            constraint_endpoints && constraint_bodies &&
            constraint_endpoint_generations &&
            constraint_body_generations && constraint_families &&
            constraint_modes && constraint_normals &&
            constraint_body_local_points && constraint_contact_distances &&
            constraint_frictions && constraint_responses &&
            constraint_active && endpoint_position_deltas &&
            body_position_deltas && endpoint_impulses &&
            body_angular_impulses && status;
    }

    void reset_correction_budget() {
        std::fill(endpoint_correction_used,
            endpoint_correction_used + count, 0.0F);
        std::fill(body_correction_used, body_correction_used + count, 0.0F);
    }

    void run(unsigned profile = rigid_coupling_realtime_profile_version,
             float dt = rigid_coupling_realtime_timestep) {
        for (unsigned sweep = 0U;
             sweep < rigid_coupling_realtime_sweeps; ++sweep) {
            rigid_coupling_project<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(endpoint_positions),
                reinterpret_cast<Vec3_0 *>(endpoint_previous),
                reinterpret_cast<Vec3_0 *>(endpoint_velocities),
                endpoint_inverse_masses, endpoint_generations, count,
                reinterpret_cast<BodyParameters_0 *>(body_parameters),
                reinterpret_cast<RigidBodyState_0 *>(body_previous),
                reinterpret_cast<RigidBodyState_0 *>(body_states),
                body_generations, count, constraint_endpoints,
                constraint_bodies, constraint_endpoint_generations,
                constraint_body_generations, constraint_families,
                constraint_modes,
                reinterpret_cast<Vec3_0 *>(constraint_normals),
                reinterpret_cast<Vec3_0 *>(constraint_body_local_points),
                constraint_contact_distances, constraint_frictions,
                constraint_responses, constraint_active, count, 0.1F,
                2.9F, dt, profile, sweep,
                reinterpret_cast<Vec3_0 *>(endpoint_position_deltas),
                reinterpret_cast<Vec3_0 *>(body_position_deltas),
                reinterpret_cast<Vec3_0 *>(endpoint_impulses),
                reinterpret_cast<Vec3_0 *>(body_angular_impulses), status);
            rigid_coupling_apply_endpoints<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(endpoint_positions),
                reinterpret_cast<Vec3_0 *>(endpoint_velocities),
                endpoint_inverse_masses, endpoint_correction_used,
                endpoint_generations, count, constraint_endpoints,
                constraint_endpoint_generations, constraint_active, count,
                reinterpret_cast<Vec3_0 *>(endpoint_position_deltas),
                reinterpret_cast<Vec3_0 *>(endpoint_impulses), 0.1F, 2.9F,
                dt, profile, sweep, status);
            rigid_coupling_apply_bodies<<<1U, threads>>>(
                reinterpret_cast<BodyParameters_0 *>(body_parameters),
                reinterpret_cast<RigidBodyState_0 *>(body_states),
                body_correction_used, body_generations, count,
                constraint_bodies, constraint_body_generations,
                constraint_active, count,
                reinterpret_cast<Vec3_0 *>(body_position_deltas),
                reinterpret_cast<Vec3_0 *>(endpoint_impulses),
                reinterpret_cast<Vec3_0 *>(body_angular_impulses), 0.1F,
                dt, profile, sweep, status);
        }
    }
};

void test_balanced_modes_thin_contact_and_mass_ratio() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    const Vec3 previous_endpoint = fixture.endpoint_previous[0];
    const Vec3 initial_endpoint_position = fixture.endpoint_positions[0];
    const Vec3 initial_body_position = fixture.body_states[0].position;
    const Vec3 initial_endpoint_velocity = fixture.endpoint_velocities[0];
    const Vec3 initial_body_velocity = fixture.body_states[0].linear_velocity;
    const Vec3 initial_body_angular = fixture.body_states[0].angular_velocity;
    const float initial_drag_energy = 0.5F * (
        dot_host(fixture.endpoint_velocities[5],
            fixture.endpoint_velocities[5]) +
        dot_host(fixture.body_states[5].linear_velocity,
            fixture.body_states[5].linear_velocity));
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(length_host(subtract_host(
        fixture.endpoint_previous[0], previous_endpoint)) == 0.0F);

    float maximum_penetration{};
    float maximum_closing_speed{};
    unsigned tunnels{};
    for (unsigned constraint : {0U, 1U, 2U, 4U, 6U}) {
        const Vec3 body_point = add_host(fixture.body_states[constraint].position,
            fixture.constraint_body_local_points[constraint]);
        const float separation = dot_host(subtract_host(
            fixture.endpoint_positions[constraint], body_point),
            fixture.constraint_normals[constraint]);
        maximum_penetration = std::max(maximum_penetration,
            fixture.constraint_contact_distances[constraint] - separation);
        const Vec3 arm = fixture.constraint_body_local_points[constraint];
        const Vec3 point_velocity = add_host(
            fixture.body_states[constraint].linear_velocity,
            cross_host(fixture.body_states[constraint].angular_velocity, arm));
        const float closing = -dot_host(subtract_host(
            fixture.endpoint_velocities[constraint], point_velocity),
            fixture.constraint_normals[constraint]);
        maximum_closing_speed = std::max(maximum_closing_speed, closing);
        tunnels += separation <
            fixture.constraint_contact_distances[constraint] - 1.0e-5F;
    }
    CHECK(maximum_penetration <= 1.0e-5F);
    CHECK(maximum_closing_speed <= 0.05F);
    CHECK(tunnels == 0U);

    const float endpoint_mass = 1.0F / fixture.endpoint_inverse_masses[0];
    const float body_mass = 1.0F / fixture.body_parameters[0].inverse_mass;
    const Vec3 initial_momentum = add_host(
        multiply_host(initial_endpoint_velocity, endpoint_mass),
        multiply_host(initial_body_velocity, body_mass));
    const Vec3 final_momentum = add_host(
        multiply_host(fixture.endpoint_velocities[0], endpoint_mass),
        multiply_host(fixture.body_states[0].linear_velocity, body_mass));
    const float linear_residual = length_host(subtract_host(
        final_momentum, initial_momentum)) /
        std::max(length_host(subtract_host(
            fixture.endpoint_velocities[0], initial_endpoint_velocity)),
            1.0e-6F);
    const Vec3 endpoint_exchange = multiply_host(subtract_host(
        fixture.endpoint_velocities[0], initial_endpoint_velocity),
        endpoint_mass);
    const Vec3 body_exchange = multiply_host(subtract_host(
        fixture.body_states[0].linear_velocity, initial_body_velocity),
        body_mass);
    const Vec3 body_angular_exchange = multiply_host(subtract_host(
        fixture.body_states[0].angular_velocity, initial_body_angular),
        1.0F / fixture.body_parameters[0].inverse_inertia_local.z);
    const Vec3 angular_exchange_residual = add_host(
        cross_host(initial_endpoint_position, endpoint_exchange),
        add_host(cross_host(initial_body_position, body_exchange),
            body_angular_exchange));
    const float angular_residual = length_host(angular_exchange_residual) /
        std::max(length_host(cross_host(
            initial_endpoint_position, endpoint_exchange)), 1.0e-6F);
    CHECK(linear_residual <= 0.02F);
    CHECK(angular_residual <= 0.05F);

    const float attachment_separation = length_host(subtract_host(
        fixture.endpoint_positions[3], fixture.body_states[3].position));
    CHECK(attachment_separation <= 0.005F);
    const float particle_relative_speed = length_host(subtract_host(
        fixture.endpoint_velocities[5],
        fixture.body_states[5].linear_velocity));
    const float grid_relative_speed = length_host(subtract_host(
        fixture.endpoint_velocities[7],
        fixture.body_states[7].linear_velocity));
    CHECK(particle_relative_speed <= 0.95F);
    CHECK(grid_relative_speed <= 0.95F);
    const float final_drag_energy = 0.5F * (
        dot_host(fixture.endpoint_velocities[5],
            fixture.endpoint_velocities[5]) +
        dot_host(fixture.body_states[5].linear_velocity,
            fixture.body_states[5].linear_velocity));
    CHECK(final_drag_energy <= initial_drag_energy * 1.01F);

    std::printf(
        "F8_RIGID_COUPLING_QUALITY families=5 modes=8 sweeps=4 "
        "penetration=%.8f closing_speed=%.8f tunnels=%u "
        "linear_residual=%.8f angular_residual=%.8f "
        "attachment_separation=%.8f particle_drag_speed=%.8f "
        "grid_drag_speed=%.8f mass_ratio=100\n",
        std::max(maximum_penetration, 0.0F),
        std::max(maximum_closing_speed, 0.0F), tunnels, linear_residual,
        angular_residual, attachment_separation, particle_relative_speed,
        grid_relative_speed);
}

void test_generation_rebinding_and_profile_rejection() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    for (unsigned index = 1U; index < count; ++index)
        fixture.constraint_active[index] = 0U;
    fixture.endpoint_generations[0] = 2U;
    const Vec3 stale_position = fixture.endpoint_positions[0];
    const Vec3 stale_body = fixture.body_states[0].position;
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(length_host(subtract_host(
        fixture.endpoint_positions[0], stale_position)) == 0.0F);
    CHECK(length_host(subtract_host(
        fixture.body_states[0].position, stale_body)) == 0.0F);

    fixture.constraint_endpoint_generations[0] = 2U;
    fixture.reset_correction_budget();
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(fixture.endpoint_positions[0].y > stale_position.y);
    CHECK(*fixture.status == 0U);

    const Vec3 rejected_position = fixture.endpoint_positions[0];
    fixture.reset_correction_budget();
    fixture.run(99U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status &
        rigid_coupling_realtime_status_invalid_profile) != 0U);
    CHECK(length_host(subtract_host(
        fixture.endpoint_positions[0], rejected_position)) == 0.0F);
    std::puts("F8_RIGID_COUPLING_LIFECYCLE stale=ignored rebound=active "
              "invalid_profile=atomic");
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
        CHECK(cudaMemset(fixture.endpoint_correction_used, 0,
            count * sizeof(float)) == cudaSuccess);
        CHECK(cudaMemset(fixture.body_correction_used, 0,
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
        CHECK(std::isfinite(fixture.endpoint_positions[index].x));
        CHECK(std::isfinite(fixture.endpoint_positions[index].y));
        CHECK(std::isfinite(fixture.endpoint_positions[index].z));
        CHECK(std::isfinite(fixture.body_states[index].position.x));
        CHECK(std::isfinite(fixture.body_states[index].position.y));
        CHECK(std::isfinite(fixture.body_states[index].position.z));
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](float fraction) {
        const std::size_t index = static_cast<std::size_t>(
            std::ceil(fraction * samples.size())) - 1U;
        return samples[index];
    };
    std::printf(
        "F8_RIGID_COUPLING_TIMING edges=8 constraints=8 warmup=%u "
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
    test_balanced_modes_thin_contact_and_mass_ratio();
    test_generation_rebinding_and_profile_rejection();
    test_long_run_and_timing();
    if (failures == 0)
        std::puts("F8-17 rigid coupling physical CUDA tests passed");
    return failures == 0 ? 0 : 1;
}
