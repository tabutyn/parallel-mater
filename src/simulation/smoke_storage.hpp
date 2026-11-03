// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

struct SmokePressureLevel {
    std::uint32_t n{};
    std::uint32_t height{};
    std::uint32_t cell_count{};
    std::uint32_t face_count[3]{};
    float *pressure[2]{};
    float *rhs{};
    float *residual{};
    float *open[3]{};
    ~SmokePressureLevel() {
        for (auto &field : pressure) release_managed(field);
        release_managed(rhs);
        release_managed(residual);
        for (auto &field : open) release_managed(field);
    }
};

struct SmokeGridStorage {
    // Retirement reconstructs this owner without allocating. State that
    // contract explicitly for the nested array of owning pressure levels.
    SmokeGridStorage() noexcept = default;
    std::uint32_t resolution{};
    std::uint32_t height{};
    std::uint32_t cell_count{};
    std::uint32_t face_count[3]{};
    Vec3 minimum{};
    float spacing{};
    Vec3 *velocity{};
    float *face_velocity[3][2]{};
    float *face_reverse[3]{};
    float *face_open[3]{};
    float *face_wall_velocity[3]{};
    Vec3 *face_normal[3]{};
    unsigned int *face_nearest_triangle[3]{};
    float *density{};
    float *temperature{};
    float *realtime_density[2]{};
    float *realtime_temperature[2]{};
    float *realtime_density_rate{};
    float *realtime_temperature_rate{};
    unsigned long long *density_accumulator{};
    unsigned long long *temperature_accumulator{};
    float *pressure[2]{};
    float *divergence{};
    float *residual{};
    float *realtime_divergence{};
    float *realtime_face_velocity[3]{};
    Vec3 *vorticity{};
    float *strain{};
    Vec3 *subgrid_force{};
    unsigned int *solid{};
    float *rhs_max{};
    float *residual_max{};
    unsigned int *pressure_converged{};
    bool static_metadata_valid{};
    SmokePressureLevel coarse[3]{};
    ~SmokeGridStorage() {
        release_managed(velocity);
        for (int axis = 0; axis < 3; ++axis) {
            for (auto &field : face_velocity[axis]) release_managed(field);
            release_managed(face_reverse[axis]);
            release_managed(face_open[axis]);
            release_managed(face_wall_velocity[axis]);
            release_managed(face_normal[axis]);
            release_managed(face_nearest_triangle[axis]);
        }
        release_managed(density);
        release_managed(temperature);
        for (auto &field : realtime_density) release_managed(field);
        for (auto &field : realtime_temperature) release_managed(field);
        release_managed(realtime_density_rate);
        release_managed(realtime_temperature_rate);
        release_managed(density_accumulator);
        release_managed(temperature_accumulator);
        for (auto &field : pressure) release_managed(field);
        release_managed(divergence);
        release_managed(residual);
        release_managed(realtime_divergence);
        for (auto &field : realtime_face_velocity) release_managed(field);
        release_managed(vorticity);
        release_managed(strain);
        release_managed(subgrid_force);
        release_managed(solid);
        release_managed(rhs_max);
        release_managed(residual_max);
        release_managed(pressure_converged);
    }
};

[[nodiscard]] constexpr std::uint32_t smoke_grid_face_count(
    int axis, int resolution, int height) noexcept {
    if (axis == 0)
        return std::uint32_t((resolution + 1) * height * resolution);
    if (axis == 1)
        return std::uint32_t(resolution * (height + 1) * resolution);
    return std::uint32_t(resolution * height * (resolution + 1));
}

struct SmokeStorage {
    SmokeOptions options{};
    SmokeGridStorage grid{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *positions{};
    Vec3 *previous_positions{};
    Vec3 *velocities{};
    float *ages{};
    float *thermal_lift{};
    float *number_densities{};
    float *pressures{};
    Vec3 *accelerations{};
    Vec3 *vorticities{};
    float *vorticity_magnitudes{};
    std::uint64_t *keys[2]{};
    std::uint32_t *indices[2]{};
    std::uint8_t *sort_workspace{};
    std::size_t sort_workspace_size{};
    FluidBodyImpulse *rigid_impulses{};
    Vec3 *realtime_positions{};
    Vec3 *realtime_previous_positions{};
    Vec3 *realtime_velocities{};
    Vec3 *realtime_vorticities{};
    float *realtime_ages{};
    float *realtime_thermal_lift{};
    float *realtime_number_densities{};
    float *realtime_pressures{};
    std::uint32_t *realtime_stable_ids{};
    Vec3 *realtime_position_deltas{};
    Vec3 *realtime_accumulated_corrections{};
    std::uint32_t *realtime_candidate_counts{};
    std::uint32_t *realtime_candidate_indices{};
    std::uint32_t *realtime_count{};
    std::uint32_t *realtime_time_advances{};
    std::uint32_t realtime_leaf_offset{};
    std::uint32_t realtime_query_offset{};
    simulation::SmokeRealtimeTickContract realtime_contract{};
    std::uint32_t realtime_pending_count{};
    std::uint32_t realtime_pending_next_slot{};
    std::uint64_t realtime_pending_emitted{};
    std::uint32_t realtime_pending_emit_count{};
    float realtime_pending_emission_fraction{};
    float realtime_pending_time{};
    std::uint32_t count{};
    std::uint32_t next_slot{};
    std::uint64_t emitted{};
    bool index_dirty{};
    float emission_fraction{};
    float time{};
    ~SmokeStorage() {
        release_managed(positions);
        release_managed(previous_positions);
        release_managed(velocities);
        release_managed(ages);
        release_managed(thermal_lift);
        release_managed(number_densities);
        release_managed(pressures);
        release_managed(accelerations);
        release_managed(vorticities);
        release_managed(vorticity_magnitudes);
        for (auto &key : keys) release_managed(key);
        for (auto &index : indices) release_managed(index);
        release_managed(sort_workspace);
        release_managed(rigid_impulses);
        release_managed(realtime_positions);
        release_managed(realtime_previous_positions);
        release_managed(realtime_velocities);
        release_managed(realtime_vorticities);
        release_managed(realtime_ages);
        release_managed(realtime_thermal_lift);
        release_managed(realtime_number_densities);
        release_managed(realtime_pressures);
        release_managed(realtime_stable_ids);
        release_managed(realtime_position_deltas);
        release_managed(realtime_accumulated_corrections);
        release_managed(realtime_candidate_counts);
        release_managed(realtime_candidate_indices);
        release_managed(realtime_count);
        release_managed(realtime_time_advances);
    }
};

template <class Options> struct SmokeDeformableCouplingSlot {
    Options options{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *minimum{};
    Vec3 *maximum{};
    ~SmokeDeformableCouplingSlot() {
        release_managed(minimum);
        release_managed(maximum);
    }
};

using SmokeSoftBodyCouplingSlot =
    SmokeDeformableCouplingSlot<SmokeSoftBodyCouplingOptions>;
using SmokeClothCouplingSlot =
    SmokeDeformableCouplingSlot<SmokeClothCouplingOptions>;
using SmokeRopeCouplingSlot =
    SmokeDeformableCouplingSlot<SmokeRopeCouplingOptions>;

struct SmokeRigidCouplingSlot {
    SmokeRigidCouplingOptions options{};
    TriangleMeshId mesh{};
    std::uint32_t generation{1U};
    bool alive{};
};
