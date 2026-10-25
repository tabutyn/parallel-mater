// SPDX-License-Identifier: MIT
#pragma once

#include <simulation/runtime.hpp>

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

// Host descriptor mirrored by RopeData in shaders/types.slang. It contains
// resource bindings only; all rope physics remains in rope.slang.
struct RopeDispatchData {
    RopeOptions options{};
    std::uint32_t count{};
    Vec3 *positions{}, *previous{}, *velocities{}, *constraint_forces{};
    Vec3 *contact_forces{}, *fluid_contact_forces{};
    Vec3 *soft_body_contact_forces{};
    Vec3 *soft_anchor_positions{}, *soft_anchor_velocities{};
    Vec3 *soft_anchor_impulses{};
    float *soft_anchor_inverse_masses{};
    bool soft_first{}, soft_last{};
    Vec3 *directions{}, *scratch{}, *body_translation{}, *body_rotation{};
    Vec3 *normals{}, *normals2{};
    float *rest{}, *lambda{};
    std::uint32_t *solid_hint{};
    std::uint32_t body_capacity{};
    std::uint32_t first_attachment_contact_skip{2U};
    std::uint32_t last_attachment_contact_skip{2U};
};

struct RopeSoftTargetDispatch {
    const Vec3 *surface{}, *previous_surface{}, *velocities{};
    const Vec3 *bounds{};
    const void *tree{};
    const std::uint32_t *order{}, *indices{};
    const void *bindings{};
    const float *inverse_masses{};
    Vec3 *node_impulses{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    std::uint32_t triangle_count{}, surface_vertex_count{};
    float distance{}, friction{};
    float orientation{1.0F};
    bool first_anchor{}, last_anchor{};
};

void predict_rope_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const void *anchor_targets,
    const std::uint8_t *anchor_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    const float gravity[3], float velocity_damping, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *time_advances, std::uint32_t *status) noexcept;

void project_rope_realtime_constraints(
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *rest_lengths,
    const float *rest_bend_lengths, const std::uint8_t *anchor_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float structural_compliance, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void apply_rope_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void project_rope_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    float *correction_used, const std::uint8_t *anchor_active,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    float friction, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void commit_rope_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    float friction, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void advance_rope(device::ExecutionQueue stream, const RopeDispatchData &rope,
    float timestep, Vec3 gravity, int first, int last,
    const void *body_parameters, void *body_states,
    const void *previous_body_states, const void *meshes,
    std::uint32_t body_count, bool first_substep,
    const RopeSoftTargetDispatch &soft_a,
    const RopeSoftTargetDispatch &soft_b) noexcept;

} // namespace parallel_mater::simulation
