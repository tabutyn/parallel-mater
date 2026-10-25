// SPDX-License-Identifier: MIT
#include "soft_body_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

#include <cstring>

namespace parallel_mater::simulation {
namespace {
Vec3 vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}

} // namespace

void predict_soft_body_realtime(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, const float gravity[3],
    float velocity_damping, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "soft_body_realtime_predict", grid,
        positions, velocities, previous_positions, correction_used,
        inverse_masses, characteristic_lengths, count, capacity,
        vec3(gravity), velocity_damping, maximum_speed, timestep,
        profile_version, time_advances, status);
}

void project_soft_body_realtime_links(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, std::uint32_t capacity, float compliance,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "soft_body_realtime_project_links", grid,
        positions, position_deltas, correction_used, inverse_masses,
        characteristic_lengths, offsets, neighbors, count, capacity,
        compliance, timestep, profile_version, status);
}

void apply_soft_body_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "soft_body_realtime_apply_position_delta",
        grid, positions, position_deltas, correction_used, inverse_masses,
        characteristic_lengths, count, capacity, timestep, profile_version,
        status);
}

void project_soft_body_realtime_shape(
    device::ExecutionQueue stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    std::uint32_t count, std::uint32_t capacity, float movable_mass,
    const float rest_center[3], const void *inverse_rest,
    void *stored_orientation, float stiffness, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    std::array<Vec3, 3> matrix{};
    std::memcpy(&matrix, inverse_rest, sizeof matrix);
    (void)metal::launch(stream, "soft_body_realtime_project_shape", 1U,
        positions, corrections, rest_positions, inverse_masses,
        characteristic_lengths, correction_used, count, capacity,
        movable_mass, vec3(rest_center), matrix, stored_orientation,
        stiffness, timestep, profile_version, status);
}

void project_soft_body_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_radius, float friction,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "soft_body_realtime_project_contacts", grid,
        positions, previous_positions, inverse_masses,
        characteristic_lengths, correction_used, contact_normals,
        contact_offsets, contact_active, count, capacity, node_radius,
        friction, timestep, profile_version, status);
}

void commit_soft_body_realtime(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const float *characteristic_lengths,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_radius, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "soft_body_realtime_commit", grid,
        positions, previous_positions, velocities, inverse_masses,
        characteristic_lengths, contact_normals, contact_offsets,
        contact_active, count, capacity, node_radius, friction, maximum_speed,
        timestep, profile_version, status);
}

void predict_deformable(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, const float gravity[3], float timestep,
    float damping) noexcept {
    (void)metal::launch(stream, "deformable_predict", grid,
        positions, previous, velocities,
        inverse_masses, count, vec3(gravity), timestep,
        damping);
}

void measure_soft_body_momentum(
    device::DispatchSize block, device::ExecutionQueue stream, const void *velocities,
    const float *inverse_masses, std::uint32_t count, void *output) noexcept {
    (void)metal::launch(stream, "soft_body_measure_momentum", 1U,
        velocities,
        inverse_masses, count, output);
}

void restore_soft_body_momentum(
    device::DispatchSize block, device::ExecutionQueue stream, void *velocities,
    const float *inverse_masses, const void *contact_momentum_delta,
    std::uint32_t count, float movable_mass, const void *predicted_momentum,
    const std::uint32_t *dynamic_contact_flag, float maximum_speed) noexcept {
    (void)metal::launch(stream, "soft_body_restore_momentum", 1U,
        velocities, inverse_masses,
        contact_momentum_delta, count,
        movable_mass,
        predicted_momentum,
        dynamic_contact_flag, maximum_speed);
}

void project_soft_body_rest_shape(
    device::ExecutionQueue stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    std::uint32_t count, float movable_mass, const float rest_center[3],
    const void *inverse_rest, void *stored_orientation, float stiffness,
    float maximum_projection,
    const std::uint32_t *dynamic_contact_flag) noexcept {
    std::array<Vec3, 3> matrix{};
    std::memcpy(&matrix, inverse_rest, sizeof matrix);
    (void)metal::launch(stream, "soft_body_project_rest_shape", 1U,
        positions, corrections,
        rest_positions,
        inverse_masses, count, movable_mass,
        vec3(rest_center), matrix, stored_orientation,
        stiffness, maximum_projection,
        dynamic_contact_flag);
}

void project_deformable_links(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float timestep,
    float maximum_projection_fraction) noexcept {
    (void)metal::launch(stream, "deformable_project_links", grid,
        positions, scratch,
        inverse_masses,
        offsets,
        neighbors,
        bond_active, count, timestep,
        maximum_projection_fraction);
}

void project_deformable_links_warp(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const float *minimum_rest_lengths, std::uint32_t count,
    float compliance, float timestep,
    float maximum_projection_fraction) noexcept {
    // Native Metal uses a full 32-lane SIMD group: four nodes per 128 threads.
    // The shared scheduler's CUDA grid assumes eight nodes per workgroup.
    (void)metal::launch(stream, "deformable_project_links_warp", (count + 3U) / 4U,
        positions, scratch,
        inverse_masses,
        offsets,
        neighbors,
        minimum_rest_lengths, count, compliance, timestep,
        maximum_projection_fraction);
}

void finalize_soft_body_velocities(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, float inverse_timestep, float projection_response,
    float maximum_speed) noexcept {
    (void)metal::launch(stream, "soft_body_finalize_velocities", grid,
        positions,
        previous, velocities,
        inverse_masses, count, inverse_timestep,
        projection_response, maximum_speed);
}

void apply_soft_body_contact_friction(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses, void *contact_forces,
    const void *contact_normals, const void *contact_arms,
    void *body_impulses, const void *body_parameters,
    void *contact_momentum_delta, void *accumulated_friction_delta,
    const float *accumulated_normal_delta,
    const std::uint32_t *contact_count, std::uint32_t count,
    float movable_mass, const float gravity[3], float timestep,
    float contact_friction, float maximum_speed) noexcept {
    (void)metal::launch(stream, "soft_body_apply_contact_friction", grid,
        positions, velocities,
        inverse_masses, contact_forces,
        contact_normals,
        contact_arms,
        body_impulses,
        body_parameters,
        contact_momentum_delta,
        accumulated_friction_delta,
        accumulated_normal_delta,
        contact_count, count, movable_mass,
        vec3(gravity), timestep, contact_friction, maximum_speed);
}

void damp_soft_body_springs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float damping,
    float maximum_speed) noexcept {
    (void)metal::launch(stream, "soft_body_damp_springs", grid,
        positions,
        velocities, output,
        inverse_masses,
        offsets,
        neighbors,
        bond_active, count, damping,
        maximum_speed);
}

void damp_soft_body_springs_warp(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, float damping, float maximum_speed) noexcept {
    (void)metal::launch(stream, "soft_body_damp_springs_warp", (count + 3U) / 4U,
        positions,
        velocities, output,
        inverse_masses,
        offsets,
        neighbors,
        count, damping, maximum_speed);
}

void update_soft_body_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *rest_positions, const void *surface_rest_positions,
    const void *bindings, void *surface_positions,
    std::uint32_t surface_vertex_count) noexcept {
    (void)metal::launch(stream, "soft_body_update_surface", grid,
        positions,
        rest_positions,
        surface_rest_positions,
        bindings,
        surface_positions, surface_vertex_count);
}

} // namespace parallel_mater::simulation
