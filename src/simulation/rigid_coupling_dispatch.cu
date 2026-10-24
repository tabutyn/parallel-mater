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

} // namespace parallel_mater::simulation
