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

void predict_cloth_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, const float gravity[3],
    float velocity_damping, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept {
    cloth_realtime_predict<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        as<Vec3_0>(previous_positions), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), count, capacity,
        Vec3_0{gravity[0], gravity[1], gravity[2]}, velocity_damping,
        maximum_speed, timestep, profile_version, time_advances, status);
}

void project_cloth_realtime_links(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count,
    std::uint32_t capacity, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    cloth_realtime_project_links<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        as<Vec3_0>(position_deltas), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths),
        const_cast<unsigned int *>(offsets),
        const_cast<DeformableNeighbor_0 *>(
            as<DeformableNeighbor_0>(neighbors)),
        const_cast<unsigned char *>(bond_active), count, capacity, timestep,
        profile_version, status);
}

void apply_cloth_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    cloth_realtime_apply_position_delta<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(position_deltas)), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), count, capacity,
        timestep, profile_version, status);
}

void project_cloth_realtime_volume(
    device::ExecutionQueue stream, void *positions,
    const float *inverse_masses, const float *characteristic_lengths,
    float *correction_used, const std::uint32_t *indices,
    std::uint32_t vertex_count, std::uint32_t triangle_count,
    const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients,
    float target_volume, float orientation, float compliance, bool enabled,
    float timestep, std::uint32_t profile_version, float *lambda,
    bool reset_lambda, std::uint32_t *status) noexcept {
    cloth_realtime_project_volume<<<1U, 128U, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), correction_used,
        const_cast<unsigned int *>(indices), vertex_count, triangle_count,
        const_cast<unsigned int *>(corner_offsets),
        const_cast<unsigned int *>(corner_indices), as<Vec3_0>(gradients),
        target_volume, orientation, compliance, enabled, timestep,
        profile_version, lambda, reset_lambda, status);
}

void project_cloth_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float thickness, float friction,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    cloth_realtime_project_contacts<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), correction_used,
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_normals)),
        const_cast<float *>(contact_offsets),
        const_cast<unsigned char *>(contact_active), count, capacity,
        thickness, friction, timestep, profile_version, status);
}

void commit_cloth_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const float *characteristic_lengths,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, const void *bonds,
    std::uint8_t *bond_active, std::uint8_t *bond_damage,
    std::uint32_t bond_count, float break_strain,
    std::uint32_t fracture_persistence, const void *contact_impulses,
    float impact_break_threshold, std::uint32_t count,
    std::uint32_t capacity, float thickness, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    cloth_realtime_commit<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_normals)),
        const_cast<float *>(contact_offsets),
        const_cast<unsigned char *>(contact_active),
        const_cast<ClothBond_0 *>(as<ClothBond_0>(bonds)), bond_active,
        bond_damage, bond_count, break_strain, fracture_persistence,
        const_cast<FluidBodyImpulse_0 *>(
            as<FluidBodyImpulse_0>(contact_impulses)),
        impact_break_threshold, count, capacity, thickness, friction,
        maximum_speed, timestep, profile_version, status);
}

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
