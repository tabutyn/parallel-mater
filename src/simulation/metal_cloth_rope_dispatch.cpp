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
    std::uint32_t end, void *positions,
    void *velocities, void *forces, const float *inverse_masses,
    std::uint32_t vertex, float maximum_acceleration, float timestep,
    float frame_inverse_timestep) noexcept {
    (void)metal::launch(stream, "rope_cloth_apply_anchor", 1U,
        rope_positions, rope_velocities, rope_count,
        anchor_positions, anchor_velocities, anchor_impulses, end,
        positions, velocities, forces,
        inverse_masses, vertex, maximum_acceleration,
        timestep, frame_inverse_timestep);
}

} // namespace parallel_mater::simulation
