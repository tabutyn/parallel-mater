// SPDX-License-Identifier: MIT
#include "smoke_coupling_dispatch.hpp"

#include "parallel_mater_cloth_smoke_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
SmokeOptions_0 converted(const SmokeDispatchOptions &value) noexcept {
    SmokeOptions_0 result{};
    result.capacity_0 = value.capacity;
    result.particle_radius_0 = value.particle_radius;
    result.lifetime_0 = value.lifetime;
    result.rest_number_density_0 = value.rest_number_density;
    result.pressure_stiffness_0 = value.pressure_stiffness;
    result.maximum_speed_0 = value.maximum_speed;
    return result;
}
SmokeGridField_0 converted(const SmokeGridDispatchField &value) noexcept {
    SmokeGridField_0 result{};
    result.n_0 = value.resolution;
    result.height_0 = value.height;
    result.minimum_0 = {value.minimum[0], value.minimum[1], value.minimum[2]};
    result.spacing_0 = value.spacing;
    result.velocity_0 = const_cast<Vec3_0 *>(as<Vec3_0>(value.velocity));
    result.density_0 = const_cast<float *>(value.density);
    result.pressure_0 = const_cast<float *>(value.pressure);
    for (int axis = 0; axis < 3; ++axis)
        result.face_0.m_data[axis] =
            const_cast<float *>(value.face_velocity[axis]);
    result.vorticity_0 = const_cast<Vec3_0 *>(as<Vec3_0>(value.vorticity));
    result.strain_0 = const_cast<float *>(value.strain);
    return result;
}
} // namespace

void apply_smoke_wind_to_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t vertex_count,
    const SmokeDispatchOptions &smoke, const void *smoke_positions,
    const void *smoke_velocities, std::uint32_t smoke_owner_slot,
    std::uint32_t smoke_owner_generation,
    std::uint32_t smoke_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const SmokeGridDispatchField &smoke_grid,
    float drag, float maximum_acceleration, float timestep) noexcept {
    smoke_cloth_wind<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses), vertex_count, converted(smoke),
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_velocities)),
        smoke_owner_slot, smoke_owner_generation, smoke_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        spatial_status, converted(smoke_grid), drag,
        maximum_acceleration, timestep);
}

void collide_smoke_with_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    float *pressures, std::uint32_t count,
    const SmokeDispatchOptions &smoke, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    std::uint32_t index_count, const void *minimum, const void *maximum,
    float clearance, const std::uint32_t *triangle_generations,
    std::uint32_t cloth_owner_slot, std::uint32_t cloth_owner_generation,
    std::uint32_t cloth_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const SmokeGridDispatchField &smoke_grid) noexcept {
    smoke_cloth_contact<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), const_cast<float *>(ages), pressures, count,
        smoke.lifetime,
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_velocities)),
        const_cast<std::uint32_t *>(cloth_indices), index_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(minimum)),
        const_cast<Vec3_0 *>(as<Vec3_0>(maximum)), clearance,
        const_cast<std::uint32_t *>(triangle_generations), cloth_owner_slot,
        cloth_owner_generation, cloth_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        spatial_status,
        smoke.pressure_stiffness, smoke.maximum_speed, converted(smoke_grid));
}

} // namespace parallel_mater::simulation
