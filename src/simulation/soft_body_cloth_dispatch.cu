// SPDX-License-Identifier: MIT
#include "soft_body_cloth_dispatch.hpp"

#include "parallel_mater_soft_body_cloth_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

} // namespace

void detect_soft_body_cloth_contacts(dim3 grid, dim3 block,
    cudaStream_t stream, const void *positions, const void *previous,
    const void *velocities, const float *inverse_masses,
    std::uint32_t count, const void *cloth_positions,
    const void *cloth_previous, const void *cloth_velocities,
    const float *cloth_inverse_masses, const std::uint32_t *indices,
    const void *surface, std::uint32_t triangle_count, float distance,
    float friction, void *contacts, std::uint32_t *contact_counts) noexcept {
    soft_cloth_detect<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<float *>(inverse_masses), count,
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_previous)),
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_velocities)),
        const_cast<float *>(cloth_inverse_masses),
        const_cast<std::uint32_t *>(indices),
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)), triangle_count, distance,
        friction, as<SoftClothContact_0>(contacts), contact_counts);
}

void apply_soft_body_cloth_contacts_to_soft_body(dim3 grid, dim3 block,
    cudaStream_t stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    const std::uint32_t *contact_counts, float timestep,
    float maximum_speed) noexcept {
    soft_cloth_apply_soft<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities), as<Vec3_0>(forces),
        const_cast<float *>(inverse_masses), count,
        const_cast<SoftClothContact_0 *>(as<SoftClothContact_0>(contacts)),
        const_cast<std::uint32_t *>(contact_counts), timestep, maximum_speed);
}

void apply_soft_body_cloth_contacts_to_cloth(dim3 grid, dim3 block,
    cudaStream_t stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    std::uint32_t contact_count, const std::uint32_t *contact_counts,
    float timestep) noexcept {
    soft_cloth_apply_cloth<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities), as<Vec3_0>(forces),
        const_cast<float *>(inverse_masses), count,
        const_cast<SoftClothContact_0 *>(as<SoftClothContact_0>(contacts)),
        contact_count, const_cast<std::uint32_t *>(contact_counts), timestep);
}

} // namespace parallel_mater::simulation
