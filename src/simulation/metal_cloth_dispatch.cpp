// SPDX-License-Identifier: MIT
#include "cloth_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"

namespace parallel_mater::simulation {
void project_cloth_volume(
    cudaStream_t stream, void *positions, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t vertex_count,
    std::uint32_t triangle_count, const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients, float target_volume,
    float orientation, float compliance, float timestep, float *lambda,
    bool reset_lambda) noexcept {
    (void)metal::launch(stream, "cloth_project_volume", 1U,
        positions, inverse_masses,
        indices, vertex_count, triangle_count,
        corner_offsets,
        corner_indices, gradients,
        target_volume, orientation, compliance, timestep, lambda, reset_lambda);
}

void break_cloth_bonds(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *bonds, std::uint8_t *active, std::uint8_t *damage,
    std::uint32_t count, float break_strain, std::uint32_t persistence,
    const void *contact_impulses, float impact_threshold) noexcept {
    (void)metal::launch(stream, "cloth_break_bonds", grid,
        positions,
        bonds, active, damage,
        count, break_strain, persistence,
        contact_impulses, impact_threshold);
}

void limit_cloth_strain(
    cudaStream_t stream, void *positions, void *scratch,
    const float *inverse_masses, const std::uint32_t *offsets,
    const void *neighbors, const std::uint8_t *free_nodes,
    std::uint32_t count) noexcept {
    (void)metal::launch(stream, "cloth_limit_strain", 1U,
        positions, scratch,
        inverse_masses,
        offsets,
        neighbors,
        free_nodes, count);
}

void update_cloth_surface(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const std::uint32_t *source_indices, void *surface_positions,
    std::uint32_t triangle_count) noexcept {
    (void)metal::launch(stream, "cloth_update_surface", grid,
        positions,
        source_indices,
        surface_positions, triangle_count);
}

} // namespace parallel_mater::simulation
