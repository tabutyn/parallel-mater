// SPDX-License-Identifier: MIT
#include "rigid_coupling_dispatch.hpp"

#include "parallel_mater_rigid_coupling_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
Vec3_0 vector(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
} // namespace

void project_rigid_couplings(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const RigidCouplingEndpointBuffers &endpoints,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    rigid_coupling_project<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(endpoints.positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(endpoints.previous_positions)),
        as<Vec3_0>(endpoints.velocities),
        const_cast<float *>(endpoints.inverse_masses),
        const_cast<std::uint32_t *>(endpoints.generations), endpoints.count,
        const_cast<BodyParameters_0 *>(
            as<BodyParameters_0>(bodies.parameters)),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(bodies.previous_states)),
        as<RigidBodyState_0>(bodies.states),
        const_cast<std::uint32_t *>(bodies.generations), bodies.count,
        const_cast<std::uint32_t *>(constraints.endpoints),
        const_cast<std::uint32_t *>(constraints.bodies),
        const_cast<std::uint32_t *>(constraints.endpoint_generations),
        const_cast<std::uint32_t *>(constraints.body_generations),
        const_cast<std::uint32_t *>(constraints.families),
        const_cast<std::uint32_t *>(constraints.modes),
        const_cast<Vec3_0 *>(as<Vec3_0>(constraints.normals)),
        const_cast<Vec3_0 *>(as<Vec3_0>(constraints.body_local_points)),
        const_cast<float *>(constraints.contact_distances),
        const_cast<float *>(constraints.frictions),
        const_cast<float *>(constraints.responses),
        const_cast<std::uint8_t *>(constraints.active), constraints.count,
        characteristic_length, maximum_speed, timestep, profile_version,
        sweep, as<Vec3_0>(scratch.endpoint_position_deltas),
        as<Vec3_0>(scratch.body_position_deltas),
        as<Vec3_0>(scratch.endpoint_impulses),
        as<Vec3_0>(scratch.body_angular_impulses), status);
}

void apply_rigid_coupling_endpoints(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const RigidCouplingEndpointBuffers &endpoints,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    rigid_coupling_apply_endpoints<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(endpoints.positions), as<Vec3_0>(endpoints.velocities),
        const_cast<float *>(endpoints.inverse_masses),
        endpoints.correction_used,
        const_cast<std::uint32_t *>(endpoints.generations), endpoints.count,
        const_cast<std::uint32_t *>(constraints.endpoints),
        const_cast<std::uint32_t *>(constraints.endpoint_generations),
        const_cast<std::uint8_t *>(constraints.active), constraints.count,
        as<Vec3_0>(scratch.endpoint_position_deltas),
        as<Vec3_0>(scratch.endpoint_impulses), characteristic_length,
        maximum_speed, timestep, profile_version, sweep, status);
}

void apply_rigid_coupling_bodies(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    rigid_coupling_apply_bodies<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<BodyParameters_0 *>(
            as<BodyParameters_0>(bodies.parameters)),
        as<RigidBodyState_0>(bodies.states), bodies.correction_used,
        const_cast<std::uint32_t *>(bodies.generations), bodies.count,
        const_cast<std::uint32_t *>(constraints.bodies),
        const_cast<std::uint32_t *>(constraints.body_generations),
        const_cast<std::uint8_t *>(constraints.active), constraints.count,
        as<Vec3_0>(scratch.body_position_deltas),
        as<Vec3_0>(scratch.endpoint_impulses),
        as<Vec3_0>(scratch.body_angular_impulses), characteristic_length,
        timestep, profile_version, sweep, status);
}

void sync_rigid_rope_attachments(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *rope_positions,
    void *rope_previous_positions, void *rope_velocities,
    std::uint32_t rope_count, const float *rope_rest_lengths,
    void *rope_anchor_targets,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    float timestep) noexcept {
    rigid_rope_sync_attachments<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(rope_positions), as<Vec3_0>(rope_previous_positions),
        as<Vec3_0>(rope_velocities), rope_count,
        const_cast<float *>(rope_rest_lengths),
        as<Vec3_0>(rope_anchor_targets),
        const_cast<BodyParameters_0 *>(
            as<BodyParameters_0>(bodies.parameters)),
        as<RigidBodyState_0>(bodies.states),
        const_cast<std::uint32_t *>(bodies.generations), bodies.count,
        const_cast<std::uint32_t *>(constraints.endpoints),
        const_cast<std::uint32_t *>(constraints.bodies),
        const_cast<std::uint32_t *>(constraints.body_generations),
        const_cast<Vec3_0 *>(as<Vec3_0>(constraints.body_local_points)),
        const_cast<std::uint8_t *>(constraints.active), constraints.count,
        timestep);
}

void apply_rigid_rope_attachment_reactions(
    device::DispatchSize grid, device::DispatchSize block,
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
    rigid_rope_apply_attachment_reactions<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_rope_velocities)),
        const_cast<Vec3_0 *>(as<Vec3_0>(current_rope_velocities)),
        const_cast<float *>(rope_inverse_masses), rope_count,
        const_cast<float *>(rope_rest_lengths), rope_maximum_speed,
        attached_end_mask, attached_end_count, vector(gravity), timestep,
        as<Vec3_0>(rope_positions), as<Vec3_0>(rope_velocities),
        as<Vec3_0>(rope_constraint_forces),
        as<Vec3_0>(rope_anchor_targets),
        const_cast<BodyParameters_0 *>(
            as<BodyParameters_0>(bodies.parameters)),
        const_cast<BodyAccumulator_0 *>(
            as<BodyAccumulator_0>(body_accumulators)),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(bodies.previous_states)),
        as<RigidBodyState_0>(bodies.states),
        const_cast<std::uint32_t *>(bodies.generations), bodies.count,
        const_cast<std::uint32_t *>(constraints.endpoints),
        const_cast<std::uint32_t *>(constraints.bodies),
        const_cast<std::uint32_t *>(constraints.body_generations),
        const_cast<Vec3_0 *>(as<Vec3_0>(constraints.body_local_points)),
        const_cast<std::uint8_t *>(constraints.active), constraints.count);
}

} // namespace parallel_mater::simulation
