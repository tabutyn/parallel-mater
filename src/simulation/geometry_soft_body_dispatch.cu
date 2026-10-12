// SPDX-License-Identifier: MIT
#include "geometry_soft_body_dispatch.hpp"

#include "parallel_mater_geometry_soft_body_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
} // namespace

void collide_deformable_with_rigid_geometry(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *velocities, const void *previous, const float *inverse_masses,
    std::uint32_t count, float thickness, float timestep,
    const void *body_parameters, const void *previous_body_states,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    void *impulses, void *contact_forces, void *contact_normals,
    void *contact_arms, void *contact_momentum_delta,
    float *accumulated_normal_delta,
    std::uint32_t *deformable_contact_count,
    std::uint32_t *dynamic_contact_flag, std::uint32_t *contact_flags,
    bool couple_dynamic, bool position_only, bool static_only,
    bool reconstruct_free_velocities, float maximum_speed,
    void *body_position_corrections) noexcept {
    deformable_collide<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<float *>(inverse_masses), count, thickness, timestep,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(previous_body_states)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, as<FluidBodyImpulse_0>(impulses),
        as<Vec3_0>(contact_forces), as<Vec3_0>(contact_normals),
        as<Vec3_0>(contact_arms), as<Vec3_0>(contact_momentum_delta),
        accumulated_normal_delta, deformable_contact_count,
        dynamic_contact_flag, contact_flags, couple_dynamic, position_only,
        static_only, reconstruct_free_velocities, maximum_speed,
        as<Vec3_0>(body_position_corrections));
}

void calculate_soft_body_surface_contacts(
    dim3 grid, dim3 block, cudaStream_t stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t triangle_count,
    const void *body_parameters, const void *body_states,
    const void *meshes, std::uint32_t body_count, void *corrections) noexcept {
    soft_body_surface_contacts<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<std::uint32_t *>(indices), triangle_count,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, as<Vec3_0>(corrections));
}

void apply_soft_body_surface_contacts(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    const void *corrections, const std::uint32_t *offsets,
    const void *influences, std::uint32_t node_count) noexcept {
    soft_body_apply_surface_contacts<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(corrections)),
        const_cast<std::uint32_t *>(offsets),
        const_cast<SoftSurfaceInfluence_0 *>(
            as<SoftSurfaceInfluence_0>(influences)),
        node_count);
}

} // namespace parallel_mater::simulation
