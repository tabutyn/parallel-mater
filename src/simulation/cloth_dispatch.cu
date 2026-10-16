// SPDX-License-Identifier: MIT
#include "cloth_dispatch.hpp"

#include "parallel_mater_cloth_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}

template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

} // namespace

void project_cloth_volume(
    device::ExecutionQueue stream, void *positions, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t vertex_count,
    std::uint32_t triangle_count, const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients, float target_volume,
    float orientation, float compliance, float timestep, float *lambda,
    bool reset_lambda) noexcept {
    cloth_project_volume<<<1U, 128U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(indices), vertex_count, triangle_count,
        const_cast<unsigned int *>(corner_offsets),
        const_cast<unsigned int *>(corner_indices), as<Vec3_0>(gradients),
        target_volume, orientation, compliance, timestep, lambda, reset_lambda);
}

void break_cloth_bonds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *bonds, std::uint8_t *active, std::uint8_t *damage,
    std::uint32_t count, float break_strain, std::uint32_t persistence,
    const void *contact_impulses, float impact_threshold) noexcept {
    cloth_break_bonds<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<ClothBond_0 *>(as<ClothBond_0>(bonds)), active, damage,
        count, break_strain, persistence,
        const_cast<FluidBodyImpulse_0 *>(
            as<FluidBodyImpulse_0>(contact_impulses)), impact_threshold);
}

void limit_cloth_strain(
    device::ExecutionQueue stream, void *positions, void *scratch,
    const float *inverse_masses, const std::uint32_t *offsets,
    const void *neighbors, const std::uint8_t *free_nodes,
    std::uint32_t count) noexcept {
    cloth_limit_strain<<<1U, 128U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(scratch),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<DeformableNeighbor_0 *>(as<DeformableNeighbor_0>(neighbors)),
        const_cast<unsigned char *>(free_nodes), count);
}

void update_cloth_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const std::uint32_t *source_indices, void *surface_positions,
    std::uint32_t triangle_count) noexcept {
    cloth_update_surface<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<unsigned int *>(source_indices),
        as<Vec3_0>(surface_positions), triangle_count);
}

} // namespace parallel_mater::simulation
