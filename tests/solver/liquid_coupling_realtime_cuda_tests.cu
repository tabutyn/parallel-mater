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
#include "simulation/liquid_coupling_realtime.hpp"
#include "parallel_mater_liquid_coupling_runtime.cu"

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
float length_host(Vec3 value) { return std::sqrt(dot_host(value, value)); }

struct Fixture {
    Vec3 *liquid_positions{managed<Vec3>(count)};
    Vec3 *liquid_previous{managed<Vec3>(count)};
    Vec3 *liquid_velocities{managed<Vec3>(count)};
    float *liquid_inverse_masses{managed<float>(count)};
    float *liquid_temperatures{managed<float>(count)};
    float *liquid_correction_used{managed<float>(count)};
    unsigned *liquid_generations{managed<unsigned>(count)};

    Vec3 *partner_positions{managed<Vec3>(count)};
    Vec3 *partner_previous{managed<Vec3>(count)};
    Vec3 *partner_velocities{managed<Vec3>(count)};
    float *partner_inverse_masses{managed<float>(count)};
    Vec3 *partner_angular_velocities{managed<Vec3>(count)};
    Vec3 *partner_inverse_inertias{managed<Vec3>(count)};
    float *partner_temperatures{managed<float>(count)};
    float *partner_correction_used{managed<float>(count)};
    unsigned *partner_generations{managed<unsigned>(count)};

    unsigned *constraint_liquids{managed<unsigned>(count)};
    unsigned *constraint_partners{managed<unsigned>(count)};
    unsigned *constraint_liquid_generations{managed<unsigned>(count)};
    unsigned *constraint_partner_generations{managed<unsigned>(count)};
    unsigned *constraint_families{managed<unsigned>(count)};
    unsigned *constraint_modes{managed<unsigned>(count)};
    Vec3 *constraint_normals{managed<Vec3>(count)};
    Vec3 *constraint_partner_arms{managed<Vec3>(count)};
    float *constraint_contact_distances{managed<float>(count)};
    float *constraint_frictions{managed<float>(count)};
    float *constraint_responses{managed<float>(count)};
    float *constraint_transfer_masses{managed<float>(count)};
    std::uint8_t *constraint_material_equal{managed<std::uint8_t>(count)};
    std::uint8_t *constraint_phase_capacity{managed<std::uint8_t>(count)};
    std::uint8_t *constraint_active{managed<std::uint8_t>(count)};

    Vec3 *liquid_position_deltas{managed<Vec3>(count)};
    Vec3 *partner_position_deltas{managed<Vec3>(count)};
    Vec3 *liquid_impulses{managed<Vec3>(count)};
    Vec3 *partner_angular_impulses{managed<Vec3>(count)};
    float *liquid_temperature_deltas{managed<float>(count)};
    float *phase_transfer_masses{managed<float>(count)};
    unsigned *status{managed<unsigned>(1U)};

    Fixture() {
        constexpr std::array families{
            LiquidCouplingFamily::liquid, LiquidCouplingFamily::liquid,
            LiquidCouplingFamily::liquid, LiquidCouplingFamily::liquid,
            LiquidCouplingFamily::rigid, LiquidCouplingFamily::rigid,
            LiquidCouplingFamily::cloth, LiquidCouplingFamily::cloth,
            LiquidCouplingFamily::soft_body, LiquidCouplingFamily::rope,
            LiquidCouplingFamily::smoke_particle,
            LiquidCouplingFamily::smoke_particle,
            LiquidCouplingFamily::smoke_particle,
            LiquidCouplingFamily::smoke_grid,
            LiquidCouplingFamily::smoke_grid,
            LiquidCouplingFamily::smoke_grid,
        };
        constexpr std::array modes{
            LiquidCouplingMode::contact, LiquidCouplingMode::contact,
            LiquidCouplingMode::pressure, LiquidCouplingMode::viscosity,
            LiquidCouplingMode::contact, LiquidCouplingMode::containment,
            LiquidCouplingMode::contact, LiquidCouplingMode::containment,
            LiquidCouplingMode::contact, LiquidCouplingMode::contact,
            LiquidCouplingMode::drag, LiquidCouplingMode::thermal,
            LiquidCouplingMode::phase, LiquidCouplingMode::drag,
            LiquidCouplingMode::thermal, LiquidCouplingMode::phase,
        };
        for (unsigned index = 0U; index < count; ++index) {
            liquid_inverse_masses[index] = 1.0F;
            liquid_temperatures[index] = 300.0F;
            liquid_generations[index] = 1U;
            partner_inverse_masses[index] = index >= 10U ? 0.0F : 1.0F;
            partner_inverse_inertias[index] = {1.0F, 1.0F, 1.0F};
            partner_temperatures[index] = 400.0F;
            partner_generations[index] = 1U;
            constraint_liquids[index] = index;
            constraint_partners[index] = index;
            constraint_liquid_generations[index] = 1U;
            constraint_partner_generations[index] = 1U;
            constraint_families[index] = static_cast<unsigned>(families[index]);
            constraint_modes[index] = static_cast<unsigned>(modes[index]);
            constraint_normals[index] = {0.0F, 1.0F, 0.0F};
            constraint_material_equal[index] = 1U;
            constraint_phase_capacity[index] = 1U;
            constraint_active[index] = 1U;
            if (modes[index] == LiquidCouplingMode::contact ||
                modes[index] == LiquidCouplingMode::containment) {
                liquid_previous[index] = {0.0F, 0.015F, 0.0F};
                liquid_positions[index] = {0.0F, -0.005F, 0.0F};
                liquid_velocities[index] = {0.0F, -1.2F, 0.0F};
                constraint_contact_distances[index] = 0.005F;
            } else if (modes[index] == LiquidCouplingMode::pressure) {
                liquid_positions[index] = {0.0F, -0.005F, 0.0F};
                liquid_previous[index] = liquid_positions[index];
                constraint_contact_distances[index] = 0.005F;
                constraint_responses[index] = 0.5F;
            } else if (modes[index] == LiquidCouplingMode::viscosity ||
                       modes[index] == LiquidCouplingMode::drag) {
                liquid_velocities[index] = {1.0F, 0.0F, 0.0F};
                constraint_contact_distances[index] = 0.05F;
                constraint_responses[index] = 0.25F;
            } else if (modes[index] == LiquidCouplingMode::thermal) {
                constraint_responses[index] = 0.25F;
            } else {
                constraint_transfer_masses[index] = 0.25F;
            }
        }
        // Exact 100:1 boundary and off-centre rigid contact.
        partner_inverse_masses[4] = 0.01F;
        partner_inverse_inertias[4] = {0.01F, 0.01F, 0.01F};
        constraint_partner_arms[4] = {0.02F, 0.0F, 0.0F};
        liquid_positions[4].x = 0.02F;
        liquid_previous[4].x = 0.02F;
        // Particle and grid heaters intentionally share one liquid endpoint.
        // The hotter grid proposal wins; the two responses must not sum.
        constraint_liquids[14] = 11U;
        partner_temperatures[14] = 450.0F;
    }

    ~Fixture() {
        for (void *pointer : std::initializer_list<void *>{
                 liquid_positions, liquid_previous, liquid_velocities,
                 liquid_inverse_masses, liquid_temperatures,
                 liquid_correction_used, liquid_generations,
                 partner_positions, partner_previous, partner_velocities,
                 partner_inverse_masses, partner_angular_velocities,
                 partner_inverse_inertias, partner_temperatures,
                 partner_correction_used, partner_generations,
                 constraint_liquids, constraint_partners,
                 constraint_liquid_generations,
                 constraint_partner_generations, constraint_families,
                 constraint_modes, constraint_normals,
                 constraint_partner_arms, constraint_contact_distances,
                 constraint_frictions, constraint_responses,
                 constraint_transfer_masses, constraint_material_equal,
                 constraint_phase_capacity, constraint_active,
                 liquid_position_deltas, partner_position_deltas,
                 liquid_impulses, partner_angular_impulses,
                 liquid_temperature_deltas, phase_transfer_masses, status})
            if (pointer != nullptr) cudaFree(pointer);
    }

    bool allocated() const {
        return liquid_positions && liquid_previous && liquid_velocities &&
            liquid_inverse_masses && liquid_temperatures &&
            liquid_correction_used && liquid_generations &&
            partner_positions && partner_previous && partner_velocities &&
            partner_inverse_masses && partner_angular_velocities &&
            partner_inverse_inertias && partner_temperatures &&
            partner_correction_used && partner_generations &&
            constraint_liquids && constraint_partners &&
            constraint_liquid_generations &&
            constraint_partner_generations && constraint_families &&
            constraint_modes && constraint_normals &&
            constraint_partner_arms && constraint_contact_distances &&
            constraint_frictions && constraint_responses &&
            constraint_transfer_masses && constraint_material_equal &&
            constraint_phase_capacity && constraint_active &&
            liquid_position_deltas && partner_position_deltas &&
            liquid_impulses && partner_angular_impulses &&
            liquid_temperature_deltas && phase_transfer_masses && status;
    }

    void reset_correction_budget() {
        std::fill(liquid_correction_used,
            liquid_correction_used + count, 0.0F);
        std::fill(partner_correction_used,
            partner_correction_used + count, 0.0F);
    }

    void run(unsigned profile = liquid_coupling_realtime_profile_version,
             float dt = liquid_coupling_realtime_timestep) {
        for (unsigned sweep = 0U;
             sweep < liquid_coupling_realtime_sweeps; ++sweep) {
            liquid_coupling_project<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(liquid_positions),
                reinterpret_cast<Vec3_0 *>(liquid_previous),
                reinterpret_cast<Vec3_0 *>(liquid_velocities),
                liquid_inverse_masses, liquid_temperatures,
                liquid_generations, count,
                reinterpret_cast<Vec3_0 *>(partner_positions),
                reinterpret_cast<Vec3_0 *>(partner_previous),
                reinterpret_cast<Vec3_0 *>(partner_velocities),
                partner_inverse_masses,
                reinterpret_cast<Vec3_0 *>(partner_angular_velocities),
                reinterpret_cast<Vec3_0 *>(partner_inverse_inertias),
                partner_temperatures, partner_generations, count,
                constraint_liquids, constraint_partners,
                constraint_liquid_generations,
                constraint_partner_generations, constraint_families,
                constraint_modes,
                reinterpret_cast<Vec3_0 *>(constraint_normals),
                reinterpret_cast<Vec3_0 *>(constraint_partner_arms),
                constraint_contact_distances, constraint_frictions,
                constraint_responses, constraint_transfer_masses,
                constraint_material_equal, constraint_phase_capacity,
                constraint_active, count, 0.1F, 2.9F, dt, profile, sweep,
                reinterpret_cast<Vec3_0 *>(liquid_position_deltas),
                reinterpret_cast<Vec3_0 *>(partner_position_deltas),
                reinterpret_cast<Vec3_0 *>(liquid_impulses),
                reinterpret_cast<Vec3_0 *>(partner_angular_impulses),
                liquid_temperature_deltas, phase_transfer_masses, status);
            liquid_coupling_apply_particles<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(liquid_positions),
                reinterpret_cast<Vec3_0 *>(liquid_velocities),
                liquid_inverse_masses, liquid_temperatures,
                liquid_correction_used, liquid_generations, count,
                constraint_liquids, constraint_liquid_generations,
                constraint_active, count,
                reinterpret_cast<Vec3_0 *>(liquid_position_deltas),
                reinterpret_cast<Vec3_0 *>(liquid_impulses),
                liquid_temperature_deltas, 0.1F, 2.9F, dt, profile,
                sweep, status);
            liquid_coupling_apply_partners<<<1U, threads>>>(
                reinterpret_cast<Vec3_0 *>(partner_positions),
                reinterpret_cast<Vec3_0 *>(partner_velocities),
                partner_inverse_masses,
                reinterpret_cast<Vec3_0 *>(partner_angular_velocities),
                reinterpret_cast<Vec3_0 *>(partner_inverse_inertias),
                partner_correction_used, partner_generations, count,
                constraint_partners, constraint_partner_generations,
                constraint_active, count,
                reinterpret_cast<Vec3_0 *>(partner_position_deltas),
                reinterpret_cast<Vec3_0 *>(liquid_impulses),
                reinterpret_cast<Vec3_0 *>(partner_angular_impulses),
                0.1F, 2.9F, 31.0F, dt, profile, sweep, status);
        }
    }
};

void test_all_advertised_modes_and_balance() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    std::array<Vec3, count> initial_liquid_velocity{};
    std::array<Vec3, count> initial_partner_velocity{};
    for (unsigned index = 0U; index < count; ++index) {
        initial_liquid_velocity[index] = fixture.liquid_velocities[index];
        initial_partner_velocity[index] = fixture.partner_velocities[index];
    }
    const float initial_viscosity_energy = 0.5F *
        dot_host(fixture.liquid_velocities[3], fixture.liquid_velocities[3]);
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);

    float maximum_penetration{};
    float maximum_closing_speed{};
    unsigned tunnels{};
    for (unsigned index : {0U, 1U, 4U, 5U, 6U, 7U, 8U, 9U}) {
        Vec3 point = add_host(fixture.partner_positions[index],
            fixture.constraint_partner_arms[index]);
        float separation = dot_host(subtract_host(
            fixture.liquid_positions[index], point),
            fixture.constraint_normals[index]);
        maximum_penetration = std::max(maximum_penetration,
            fixture.constraint_contact_distances[index] - separation);
        float closing = -dot_host(subtract_host(
            fixture.liquid_velocities[index], fixture.partner_velocities[index]),
            fixture.constraint_normals[index]);
        maximum_closing_speed = std::max(maximum_closing_speed, closing);
        tunnels += separation < fixture.constraint_contact_distances[index] -
            1.0e-5F;
    }
    CHECK(maximum_penetration <= 1.0e-5F);
    CHECK(maximum_closing_speed <= 0.05F);
    CHECK(tunnels == 0U);

    float maximum_linear_residual{};
    for (unsigned index = 0U; index < 10U; ++index) {
        float liquidMass = 1.0F / fixture.liquid_inverse_masses[index];
        float partnerMass = 1.0F / fixture.partner_inverse_masses[index];
        Vec3 initialMomentum = add_host(
            multiply_host(initial_liquid_velocity[index], liquidMass),
            multiply_host(initial_partner_velocity[index], partnerMass));
        Vec3 finalMomentum = add_host(
            multiply_host(fixture.liquid_velocities[index], liquidMass),
            multiply_host(fixture.partner_velocities[index], partnerMass));
        maximum_linear_residual = std::max(maximum_linear_residual,
            length_host(subtract_host(finalMomentum, initialMomentum)));
    }
    CHECK(maximum_linear_residual <= 1.0e-5F);

    const float pressure_center = 0.5F *
        (fixture.liquid_positions[2].y + fixture.partner_positions[2].y);
    const float pressure_separation = fixture.liquid_positions[2].y -
        fixture.partner_positions[2].y;
    // Equal-material ownership is mechanically invisible. The analytic
    // combined-population reference applies the same four 0.5-response
    // pressure projections to the unordered pair.
    constexpr float combined_reference_separation = 0.004375F;
    constexpr float combined_reference_center = -0.0025F;
    const float split_combined_error = std::max(
        std::abs(fixture.liquid_positions[2].y -
            (combined_reference_center +
             0.5F * combined_reference_separation)),
        std::abs(fixture.partner_positions[2].y -
            (combined_reference_center -
             0.5F * combined_reference_separation)));
    CHECK(std::abs(pressure_center + 0.0025F) <= 1.0e-6F);
    CHECK(pressure_separation > -0.005F);
    CHECK(split_combined_error <= 1.0e-6F);
    const float viscosity_speed = length_host(subtract_host(
        fixture.liquid_velocities[3], fixture.partner_velocities[3]));
    const float final_viscosity_energy = 0.5F * (
        dot_host(fixture.liquid_velocities[3], fixture.liquid_velocities[3]) +
        dot_host(fixture.partner_velocities[3], fixture.partner_velocities[3]));
    CHECK(viscosity_speed < 0.4F);
    CHECK(final_viscosity_energy <= initial_viscosity_energy);
    const float particle_drag = length_host(fixture.liquid_velocities[10]);
    const float grid_drag = length_host(fixture.liquid_velocities[13]);
    CHECK(particle_drag < 0.4F);
    CHECK(grid_drag < 0.4F);
    CHECK(fixture.liquid_temperatures[11] > 300.0F &&
          fixture.liquid_temperatures[11] < 450.0F);
    CHECK(fixture.liquid_temperatures[14] == 300.0F);
    CHECK(fixture.phase_transfer_masses[12] == 0.25F);
    CHECK(fixture.phase_transfer_masses[15] == 0.25F);
    CHECK(fixture.liquid_inverse_masses[12] == 1.0F);
    CHECK(fixture.liquid_inverse_masses[15] == 1.0F);

    std::printf(
        "F8_LIQUID_COUPLING_QUALITY families=7 combinations=15 sweeps=4 "
        "penetration=%.8f closing_speed=%.8f tunnels=%u "
        "linear_residual=%.8f pressure_separation=%.8f "
        "split_combined_error=%.8f "
        "viscosity_speed=%.8f particle_drag=%.8f grid_drag=%.8f "
        "thermal_shared=%.4f thermal_ceiling=450.0000 mass_ratio=100\n",
        std::max(maximum_penetration, 0.0F),
        std::max(maximum_closing_speed, 0.0F), tunnels,
        maximum_linear_residual, pressure_separation, split_combined_error,
        viscosity_speed, particle_drag, grid_drag,
        fixture.liquid_temperatures[11]);
}

void test_generation_capacity_and_profile_rejection() {
    Fixture fixture;
    CHECK(fixture.allocated());
    if (!fixture.allocated()) return;
    for (unsigned index = 1U; index < count; ++index)
        fixture.constraint_active[index] = 0U;
    fixture.liquid_generations[0] = 2U;
    Vec3 stale = fixture.liquid_positions[0];
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(length_host(subtract_host(fixture.liquid_positions[0], stale)) == 0.0F);
    fixture.constraint_liquid_generations[0] = 2U;
    fixture.reset_correction_budget();
    fixture.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(fixture.liquid_positions[0].y > stale.y);

    Vec3 rejected = fixture.liquid_positions[0];
    fixture.reset_correction_budget();
    fixture.run(99U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & liquid_coupling_realtime_status_invalid_profile) != 0U);
    CHECK(length_host(subtract_host(fixture.liquid_positions[0], rejected)) == 0.0F);

    Fixture capacity;
    CHECK(capacity.allocated());
    if (!capacity.allocated()) return;
    for (unsigned index = 0U; index < count; ++index)
        capacity.constraint_active[index] = index == 12U ? 1U : 0U;
    capacity.constraint_phase_capacity[12] = 0U;
    capacity.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*capacity.status &
        liquid_coupling_realtime_status_capacity_exceeded) != 0U);
    CHECK(capacity.phase_transfer_masses[12] == 0.0F);

    Fixture material;
    CHECK(material.allocated());
    if (!material.allocated()) return;
    for (unsigned index = 0U; index < count; ++index)
        material.constraint_active[index] = index == 2U ? 1U : 0U;
    material.constraint_material_equal[2] = 0U;
    Vec3 material_position = material.liquid_positions[2];
    material.run();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*material.status &
        liquid_coupling_realtime_status_invalid_material) != 0U);
    CHECK(length_host(subtract_host(
        material.liquid_positions[2], material_position)) == 0.0F);
    std::puts("F8_LIQUID_COUPLING_LIFECYCLE stale=ignored rebound=active "
              "invalid_profile=atomic phase_overflow=retained "
              "unequal_liquid_material=rejected");
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
        fixture.reset_correction_budget();
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
        CHECK(std::isfinite(fixture.liquid_positions[index].x));
        CHECK(std::isfinite(fixture.partner_positions[index].x));
        CHECK(std::isfinite(fixture.liquid_velocities[index].x));
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](float fraction) {
        return samples[static_cast<std::size_t>(
            std::ceil(fraction * samples.size())) - 1U];
    };
    std::printf(
        "F8_LIQUID_COUPLING_TIMING edges=16 constraints=16 warmup=%u "
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
    test_all_advertised_modes_and_balance();
    test_generation_capacity_and_profile_rejection();
    test_long_run_and_timing();
    if (failures == 0)
        std::puts("F8-18 liquid coupling physical CUDA tests passed");
    return failures == 0 ? 0 : 1;
}
