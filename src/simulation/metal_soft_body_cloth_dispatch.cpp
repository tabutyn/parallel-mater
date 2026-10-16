// SPDX-License-Identifier: MIT
#include "soft_body_cloth_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

namespace parallel_mater::simulation {
void detect_soft_body_cloth_contacts(dim3 grid, dim3 block,
    cudaStream_t stream, const void *positions, const void *previous,
    const void *velocities, const float *inverse_masses,
    std::uint32_t count, const void *cloth_positions,
    const void *cloth_previous, const void *cloth_velocities,
    const float *cloth_inverse_masses, const std::uint32_t *indices,
    const void *surface, std::uint32_t triangle_count, float distance,
    float friction, void *contacts, std::uint32_t *contact_counts) noexcept {
    (void)metal::launch(stream, "soft_cloth_detect", grid,
        positions,
        previous,
        velocities,
        inverse_masses, count,
        cloth_positions,
        cloth_previous,
        cloth_velocities,
        cloth_inverse_masses,
        indices,
        surface, triangle_count, distance,
        friction, contacts, contact_counts);
}

void apply_soft_body_cloth_contacts_to_soft_body(dim3 grid, dim3 block,
    cudaStream_t stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    const std::uint32_t *contact_counts, float timestep,
    float maximum_speed) noexcept {
    (void)metal::launch(stream, "soft_cloth_apply_soft", grid,
        positions, velocities, forces,
        inverse_masses, count,
        contacts,
        contact_counts, timestep, maximum_speed);
}

void apply_soft_body_cloth_contacts_to_cloth(dim3 grid, dim3 block,
    cudaStream_t stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    std::uint32_t contact_count, const std::uint32_t *contact_counts,
    float timestep) noexcept {
    (void)metal::launch(stream, "soft_cloth_apply_cloth", grid,
        positions, velocities, forces,
        inverse_masses, count,
        contacts,
        contact_count, contact_counts, timestep);
}

} // namespace parallel_mater::simulation
