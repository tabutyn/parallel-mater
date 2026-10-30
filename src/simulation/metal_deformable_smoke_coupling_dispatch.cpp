// SPDX-License-Identifier: MIT
#include "deformable_smoke_coupling_dispatch.hpp"

#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {

void project_deformable_smoke_couplings(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &first,
    const DeformableSmokeCouplingEndpointBuffers &second,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "deformable_smoke_coupling_project", grid,
        first.positions, first.previous_positions, first.velocities,
        first.inverse_masses, first.generations, first.count,
        second.positions, second.previous_positions, second.velocities,
        second.inverse_masses, second.generations, second.count,
        constraints.first_endpoints, constraints.second_endpoints,
        constraints.first_generations, constraints.second_generations,
        constraints.pairs, constraints.modes, constraints.normals,
        constraints.contact_distances, constraints.frictions,
        constraints.responses, constraints.active, constraints.count,
        characteristic_length, maximum_speed, timestep, profile_version,
        sweep, scratch.first_position_deltas,
        scratch.second_position_deltas, scratch.first_impulses,
        scratch.second_impulses, status);
}

void apply_deformable_smoke_coupling_first(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &first,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "deformable_smoke_coupling_apply_first", grid,
        first.positions, first.previous_positions, first.velocities,
        first.inverse_masses,
        first.correction_used, first.generations, first.count,
        constraints.first_endpoints, constraints.first_generations,
        constraints.active, constraints.count, scratch.first_position_deltas,
        scratch.first_impulses, characteristic_length, maximum_speed,
        timestep, profile_version, sweep, status);
}

void apply_deformable_smoke_coupling_second(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &second,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "deformable_smoke_coupling_apply_second", grid,
        second.positions, second.previous_positions, second.velocities,
        second.inverse_masses,
        second.correction_used, second.generations, second.count,
        constraints.second_endpoints, constraints.second_generations,
        constraints.active, constraints.count,
        scratch.second_position_deltas, scratch.second_impulses,
        characteristic_length, maximum_speed, timestep, profile_version,
        sweep, status);
}

} // namespace parallel_mater::simulation
