// SPDX-License-Identifier: MIT
#include "rigid_coupling_dispatch.hpp"

#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {

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

} // namespace parallel_mater::simulation
