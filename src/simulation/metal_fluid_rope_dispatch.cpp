// SPDX-License-Identifier: MIT
#include "fluid_rope_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
void detect_fluid_rope_contacts(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *particle_positions, void *particle_velocities,
    const std::uint32_t *particle_count, float particle_mass,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, float distance, float friction,
    std::uint32_t *contribution_nodes, void *contribution_impulses,
    std::uint32_t *contact_count,
    float *maximum_penetration, bool transfer_impulse) noexcept {
    (void)metal::launch(stream, "fluid_rope_contacts", grid,
        particle_positions, particle_velocities,
        particle_count, particle_mass,
        rope_positions,
        rope_velocities, rope_count,
        distance, friction, contribution_nodes, contribution_impulses,
        contact_count, maximum_penetration, transfer_impulse);
}

void apply_fluid_rope_impulses(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *rope_velocities, void *fluid_contact_forces,
    std::uint32_t rope_count, float rope_mass, float rope_maximum_speed,
    bool first_attached, bool last_attached,
    const std::uint32_t *particle_count,
    const std::uint32_t *contribution_nodes,
    const void *contribution_impulses,
    float maximum_acceleration, float timestep,
    float frame_inverse_timestep) noexcept {
    (void)metal::launch(stream, "fluid_rope_apply", grid,
        rope_velocities, fluid_contact_forces,
        rope_count, rope_mass, rope_maximum_speed, first_attached,
        last_attached, particle_count, contribution_nodes,
        contribution_impulses,
        maximum_acceleration, timestep, frame_inverse_timestep);
}

} // namespace parallel_mater::simulation
