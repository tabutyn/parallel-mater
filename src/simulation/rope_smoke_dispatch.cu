// SPDX-License-Identifier: MIT
#include "smoke_coupling_dispatch.hpp"

#include "parallel_mater_rope_smoke_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
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

void apply_smoke_wind_to_rope(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count,
    float rope_maximum_speed, bool first_attached, bool last_attached,
    bool soft_first, bool soft_last, const SmokeDispatchOptions &smoke,
    const void *smoke_positions, const void *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    const SmokeGridDispatchField &smoke_grid, float drag,
    float maximum_acceleration, float timestep) noexcept {
    smoke_rope_wind<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_positions)),
        as<Vec3_0>(rope_velocities), rope_count, rope_maximum_speed,
        first_attached, last_attached, soft_first, soft_last, smoke.capacity,
        smoke.particle_radius, smoke.rest_number_density,
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_velocities)),
        reinterpret_cast<ulonglong *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(indices), converted(smoke_grid), drag,
        maximum_acceleration, timestep);
}

void collide_smoke_with_rope(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    std::uint32_t count, const SmokeDispatchOptions &smoke,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, const void *minimum, const void *maximum,
    float clearance, const std::uint32_t *segment_generations,
    std::uint32_t rope_owner_slot, std::uint32_t rope_owner_generation,
    std::uint32_t rope_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status) noexcept {
    smoke_rope_contact<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), const_cast<float *>(ages), count,
        smoke.lifetime,
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_velocities)), rope_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(minimum)),
        const_cast<Vec3_0 *>(as<Vec3_0>(maximum)), clearance,
        const_cast<std::uint32_t *>(segment_generations), rope_owner_slot,
        rope_owner_generation, rope_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        spatial_status);
}

} // namespace parallel_mater::simulation
