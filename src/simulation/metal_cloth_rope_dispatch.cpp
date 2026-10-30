// SPDX-License-Identifier: MIT
#include "rope_coupling_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
void sample_rope_cloth_anchor(device::ExecutionQueue stream, void *anchor_positions,
    void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *positions,
    const void *velocities, const float *inverse_masses,
    std::uint32_t vertex, float effective_mass) noexcept {
    (void)metal::launch(stream, "rope_cloth_sample_anchor", 1U,
        anchor_positions, anchor_velocities,
        anchor_impulses, anchor_inverse_masses, end,
        positions,
        velocities,
        inverse_masses, vertex, effective_mass);
}

void apply_rope_cloth_anchor(device::ExecutionQueue stream, void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count, void *anchor_positions,
    void *anchor_velocities, const void *anchor_impulses,
    const void *rope_constraint_forces, std::uint32_t end, void *positions,
    void *velocities, void *forces, const float *inverse_masses,
    std::uint32_t vertex, float maximum_acceleration,
    float constraint_force_scale, float timestep,
    float frame_inverse_timestep) noexcept {
    (void)metal::launch(stream, "rope_cloth_apply_anchor", 1U,
        rope_positions, rope_velocities, rope_count,
        anchor_positions, anchor_velocities, anchor_impulses,
        rope_constraint_forces, end,
        positions, velocities, forces,
        inverse_masses, vertex, maximum_acceleration,
        constraint_force_scale, timestep, frame_inverse_timestep);
}

void accumulate_rope_cloth_reaction(device::ExecutionQueue stream,
    const void *previous_velocities, const void *current_velocities,
    const float *inverse_masses, std::uint32_t rope_count,
    std::uint32_t attached_end_mask, std::uint32_t attached_end_count,
    const float gravity[3], float timestep, void *cloth_forces,
    std::uint32_t cloth_vertex) noexcept {
    (void)metal::launch(stream, "rope_cloth_accumulate_reaction", 1U,
        previous_velocities, current_velocities, inverse_masses, rope_count,
        attached_end_mask, attached_end_count,
        Vec3{gravity[0], gravity[1], gravity[2]}, timestep, cloth_forces,
        cloth_vertex);
}

} // namespace parallel_mater::simulation
