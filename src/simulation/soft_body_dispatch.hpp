// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"
#include "types.hpp"

#include <array>
#include <cstdint>

namespace parallel_mater::simulation {

// Metal records several independent soft-body resources in one native
// command. The records only describe resources; every item still executes the
// ordinary shared Slang entry point below. Keep this capacity small enough for
// Metal's inline argument-buffer limit while amortizing scene graph resources.
inline constexpr std::size_t soft_body_realtime_batch_capacity = 10U;

struct SoftBodyRealtimePredictBatchItem {
    void *positions{};
    void *velocities{};
    void *previous_positions{};
    float *correction_used{};
    const float *inverse_masses{};
    const float *characteristic_lengths{};
    std::uint32_t count{};
    std::uint32_t capacity{};
    float velocity_damping{};
    float maximum_speed{};
    std::uint32_t *time_advances{};
};

struct SoftBodyRealtimeLinksBatchItem {
    const void *positions{};
    void *position_deltas{};
    float *correction_used{};
    const float *inverse_masses{};
    const float *characteristic_lengths{};
    const std::uint32_t *offsets{};
    const void *neighbors{};
    std::uint32_t count{};
    std::uint32_t capacity{};
    float compliance{};
};

struct SoftBodyRealtimeApplyBatchItem {
    void *positions{};
    const void *position_deltas{};
    float *correction_used{};
    const float *inverse_masses{};
    const float *characteristic_lengths{};
    std::uint32_t count{};
    std::uint32_t capacity{};
};

struct SoftBodyRealtimeShapeBatchItem {
    void *positions{};
    void *corrections{};
    const void *rest_positions{};
    const float *inverse_masses{};
    const float *characteristic_lengths{};
    float *correction_used{};
    std::uint32_t count{};
    std::uint32_t capacity{};
    float movable_mass{};
    Vec3 rest_center{};
    std::array<Vec3, 3> inverse_rest{};
    void *stored_orientation{};
    float stiffness{};
};

struct SoftBodyRealtimeContactsBatchItem {
    void *positions{};
    const void *previous_positions{};
    const float *inverse_masses{};
    const float *characteristic_lengths{};
    float *correction_used{};
    const void *contact_normals{};
    const float *contact_offsets{};
    const std::uint8_t *contact_active{};
    std::uint32_t count{};
    std::uint32_t capacity{};
    float node_radius{};
    float friction{};
};

struct SoftBodyRealtimeCommitBatchItem {
    const void *positions{};
    const void *previous_positions{};
    void *velocities{};
    const float *inverse_masses{};
    const float *characteristic_lengths{};
    const void *contact_normals{};
    const float *contact_offsets{};
    const std::uint8_t *contact_active{};
    std::uint32_t count{};
    std::uint32_t capacity{};
    float node_radius{};
    float friction{};
    float maximum_speed{};
};

struct SoftBodySurfaceBatchItem {
    const void *positions{};
    const void *rest_positions{};
    const void *surface_rest_positions{};
    const void *bindings{};
    void *surface_positions{};
    std::uint32_t surface_vertex_count{};
};

struct SoftBodyContactFrictionBatchItem {
    void *positions{};
    void *velocities{};
    const float *inverse_masses{};
    void *contact_forces{};
    const void *contact_normals{};
    const void *contact_arms{};
    void *body_impulses{};
    void *contact_momentum_delta{};
    void *accumulated_friction_delta{};
    const float *accumulated_normal_delta{};
    const std::uint32_t *contact_count{};
    std::uint32_t count{};
    float movable_mass{};
    float contact_friction{};
    float maximum_speed{};
};

template <typename Item> struct SoftBodyRealtimeBatch {
    std::array<Item, soft_body_realtime_batch_capacity> items{};
};

using SoftBodyRealtimePredictBatch =
    SoftBodyRealtimeBatch<SoftBodyRealtimePredictBatchItem>;
using SoftBodyRealtimeLinksBatch =
    SoftBodyRealtimeBatch<SoftBodyRealtimeLinksBatchItem>;
using SoftBodyRealtimeApplyBatch =
    SoftBodyRealtimeBatch<SoftBodyRealtimeApplyBatchItem>;
using SoftBodyRealtimeShapeBatch =
    SoftBodyRealtimeBatch<SoftBodyRealtimeShapeBatchItem>;
using SoftBodyRealtimeContactsBatch =
    SoftBodyRealtimeBatch<SoftBodyRealtimeContactsBatchItem>;
using SoftBodyRealtimeCommitBatch =
    SoftBodyRealtimeBatch<SoftBodyRealtimeCommitBatchItem>;
using SoftBodySurfaceBatch = SoftBodyRealtimeBatch<SoftBodySurfaceBatchItem>;
using SoftBodyContactFrictionBatch =
    SoftBodyRealtimeBatch<SoftBodyContactFrictionBatchItem>;

void predict_soft_body_realtime_batch(
    device::DispatchSize grid, device::ExecutionQueue stream,
    const SoftBodyRealtimePredictBatch &batch, std::uint32_t batch_count,
    const float gravity[3], float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void project_soft_body_realtime_links_batch(
    device::DispatchSize grid, device::ExecutionQueue stream,
    const SoftBodyRealtimeLinksBatch &batch, std::uint32_t batch_count,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void apply_soft_body_realtime_position_delta_batch(
    device::DispatchSize grid, device::ExecutionQueue stream,
    const SoftBodyRealtimeApplyBatch &batch, std::uint32_t batch_count,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void project_soft_body_realtime_shape_batch(
    device::ExecutionQueue stream,
    const SoftBodyRealtimeShapeBatch &batch, std::uint32_t batch_count,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void project_soft_body_realtime_contacts_batch(
    device::DispatchSize grid, device::ExecutionQueue stream,
    const SoftBodyRealtimeContactsBatch &batch, std::uint32_t batch_count,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void commit_soft_body_realtime_batch(
    device::DispatchSize grid, device::ExecutionQueue stream,
    const SoftBodyRealtimeCommitBatch &batch, std::uint32_t batch_count,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void update_soft_body_surface_batch(
    device::DispatchSize grid, device::ExecutionQueue stream,
    const SoftBodySurfaceBatch &batch, std::uint32_t batch_count) noexcept;

void apply_soft_body_contact_friction_batch(
    device::DispatchSize grid, device::ExecutionQueue stream,
    const SoftBodyContactFrictionBatch &batch, std::uint32_t batch_count,
    const void *body_parameters, const float gravity[3], float timestep) noexcept;

void predict_soft_body_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, const float gravity[3],
    float velocity_damping, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept;

void project_soft_body_realtime_links(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, std::uint32_t capacity, float compliance,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void apply_soft_body_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void project_soft_body_realtime_shape(
    device::ExecutionQueue stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    std::uint32_t count, std::uint32_t capacity, float movable_mass,
    const float rest_center[3], const void *inverse_rest,
    void *stored_orientation, float stiffness, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void project_soft_body_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_radius, float friction,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void commit_soft_body_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const float *characteristic_lengths,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_radius, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void predict_deformable(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, const float gravity[3], float timestep,
    float damping) noexcept;

void measure_soft_body_momentum(
    device::DispatchSize block, device::ExecutionQueue stream, const void *velocities,
    const float *inverse_masses, std::uint32_t count, void *output) noexcept;

void restore_soft_body_momentum(
    device::DispatchSize block, device::ExecutionQueue stream, void *velocities,
    const float *inverse_masses, const void *contact_momentum_delta,
    std::uint32_t count, float movable_mass, const void *predicted_momentum,
    const std::uint32_t *dynamic_contact_flag, float maximum_speed) noexcept;

void project_soft_body_rest_shape(
    device::ExecutionQueue stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    std::uint32_t count, float movable_mass, const float rest_center[3],
    const void *inverse_rest, void *stored_orientation, float stiffness,
    float maximum_projection,
    const std::uint32_t *dynamic_contact_flag) noexcept;

void project_deformable_links(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float timestep,
    float maximum_projection_fraction) noexcept;

void project_deformable_links_warp(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const float *minimum_rest_lengths, std::uint32_t count,
    float compliance, float timestep,
    float maximum_projection_fraction) noexcept;

void finalize_soft_body_velocities(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, float inverse_timestep, float projection_response,
    float maximum_speed) noexcept;

void apply_soft_body_contact_friction(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses, void *contact_forces,
    const void *contact_normals, const void *contact_arms,
    void *body_impulses, const void *body_parameters,
    void *contact_momentum_delta, void *accumulated_friction_delta,
    const float *accumulated_normal_delta,
    const std::uint32_t *contact_count, std::uint32_t count,
    float movable_mass, const float gravity[3], float timestep,
    float contact_friction, float maximum_speed) noexcept;

void damp_soft_body_springs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float damping,
    float maximum_speed) noexcept;

void damp_soft_body_springs_warp(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, float damping, float maximum_speed) noexcept;

void update_soft_body_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *rest_positions, const void *surface_rest_positions,
    const void *bindings, void *surface_positions,
    std::uint32_t surface_vertex_count) noexcept;

} // namespace parallel_mater::simulation
