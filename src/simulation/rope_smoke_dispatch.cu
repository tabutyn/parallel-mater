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

void apply_smoke_wind_to_rope(
    dim3 grid, dim3 block, cudaStream_t stream, const void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count,
    float rope_maximum_speed, bool first_attached, bool last_attached,
    bool soft_first, bool soft_last, const SmokeDispatchOptions &smoke,
    const void *smoke_positions, const void *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    const SmokeGridDispatchField &smoke_grid, float drag,
    float maximum_acceleration, float timestep) noexcept {
    smoke_rope_wind<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_positions)),
        as<Vec3_0>(rope_velocities), rope_count, rope_maximum_speed,
        first_attached, last_attached, soft_first, soft_last, converted(smoke),
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_velocities)),
        reinterpret_cast<ulonglong *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(indices), converted(smoke_grid), drag,
        maximum_acceleration, timestep);
}

void collide_smoke_with_rope(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    std::uint32_t count, const SmokeDispatchOptions &smoke,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, const void *minimum, const void *maximum,
    float clearance) noexcept {
    smoke_rope_contact<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), const_cast<float *>(ages), count,
        smoke.lifetime,
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_velocities)), rope_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(minimum)),
        const_cast<Vec3_0 *>(as<Vec3_0>(maximum)), clearance);
}

} // namespace parallel_mater::simulation
