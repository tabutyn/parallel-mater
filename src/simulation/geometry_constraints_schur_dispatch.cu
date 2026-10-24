// SPDX-License-Identifier: MIT
#include "geometry_constraints_dispatch.hpp"

#include "parallel_mater_geometry_constraints_schur_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

} // namespace

PmResult solve_geometry_constraints_schur(
    device::ExecutionQueue stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *states,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep) noexcept {
    auto *shader_parameters =
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(parameters));
    auto *shader_previous = const_cast<RigidBodyState_0 *>(
        reinterpret_cast<const RigidBodyState_0 *>(previous));
    auto *shader_states = reinterpret_cast<RigidBodyState_0 *>(states);
    auto *shader_manifolds = as<ContactManifold_0>(manifolds);
    auto *shader_constraints = as<RigidConstraintResource_0>(constraints);
    auto *shader_bodies = reinterpret_cast<AvbdBody_0 *>(bodies);
    auto *shader_schedule = as<ContactSchedule_0>(schedule);
    solve_avbd_kernel<<<1U, 32U, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(shader_parameters,
        shader_previous, shader_states, body_count, shader_manifolds,
        active_pairs, active_pair_count, shader_constraints,
        constraint_capacity, shader_bodies, shader_schedule, timestep);
    return device::from_driver_error(cudaGetLastError());
}

} // namespace parallel_mater::simulation
