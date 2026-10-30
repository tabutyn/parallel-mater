// SPDX-License-Identifier: MIT
#include "rigid_coupling_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
Vec3 vector(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
} // namespace

void project_rigid_couplings(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const RigidCouplingEndpointBuffers &endpoints,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rigid_coupling_project", grid,
        endpoints.positions, endpoints.previous_positions,
        endpoints.velocities, endpoints.inverse_masses,
        endpoints.generations, endpoints.count, bodies.parameters,
        bodies.previous_states, bodies.states, bodies.generations,
        bodies.count, constraints.endpoints, constraints.bodies,
        constraints.endpoint_generations, constraints.body_generations,
        constraints.families, constraints.modes, constraints.normals,
        constraints.body_local_points, constraints.contact_distances,
        constraints.frictions, constraints.responses, constraints.active,
        constraints.count, characteristic_length, maximum_speed, timestep,
        profile_version, sweep, scratch.endpoint_position_deltas,
        scratch.body_position_deltas, scratch.endpoint_impulses,
        scratch.body_angular_impulses, status);
}

void apply_rigid_coupling_endpoints(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const RigidCouplingEndpointBuffers &endpoints,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rigid_coupling_apply_endpoints", grid,
        endpoints.positions, endpoints.velocities, endpoints.inverse_masses,
        endpoints.correction_used, endpoints.generations, endpoints.count,
        constraints.endpoints, constraints.endpoint_generations,
        constraints.active, constraints.count,
        scratch.endpoint_position_deltas, scratch.endpoint_impulses,
        characteristic_length, maximum_speed, timestep, profile_version,
        sweep, status);
}

void apply_rigid_coupling_bodies(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rigid_coupling_apply_bodies", grid,
        bodies.parameters, bodies.states, bodies.correction_used,
        bodies.generations, bodies.count, constraints.bodies,
        constraints.body_generations, constraints.active, constraints.count,
        scratch.body_position_deltas, scratch.endpoint_impulses,
        scratch.body_angular_impulses, characteristic_length, timestep,
        profile_version, sweep, status);
}

void sync_rigid_rope_attachments(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *rope_positions,
    void *rope_previous_positions, void *rope_velocities,
    std::uint32_t rope_count, const float *rope_rest_lengths,
    void *rope_anchor_targets,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    float timestep) noexcept {
    (void)metal::launch(stream, "rigid_rope_sync_attachments", grid,
        rope_positions, rope_previous_positions, rope_velocities, rope_count,
        rope_rest_lengths, rope_anchor_targets, bodies.parameters, bodies.states,
        bodies.generations, bodies.count, constraints.endpoints,
        constraints.bodies, constraints.body_generations,
        constraints.body_local_points, constraints.active, constraints.count,
        timestep);
}

void apply_rigid_rope_attachment_reactions(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream,
    const void *previous_rope_velocities,
    const void *current_rope_velocities,
    const float *rope_inverse_masses, std::uint32_t rope_count,
    const float *rope_rest_lengths, float rope_maximum_speed,
    std::uint32_t attached_end_mask, std::uint32_t attached_end_count,
    const float gravity[3], float timestep, void *rope_positions,
    void *rope_velocities, void *rope_constraint_forces,
    void *rope_anchor_targets, const void *body_accumulators,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints) noexcept {
    (void)metal::launch(stream,
        "rigid_rope_apply_attachment_reactions", grid,
        previous_rope_velocities, current_rope_velocities,
        rope_inverse_masses, rope_count, rope_rest_lengths,
        rope_maximum_speed, attached_end_mask,
        attached_end_count, vector(gravity), timestep, rope_positions,
        rope_velocities, rope_constraint_forces, rope_anchor_targets,
        bodies.parameters, body_accumulators, bodies.previous_states,
        bodies.states, bodies.generations, bodies.count,
        constraints.endpoints, constraints.bodies,
        constraints.body_generations, constraints.body_local_points,
        constraints.active, constraints.count);
}

} // namespace parallel_mater::simulation
