// SPDX-License-Identifier: MIT
#include "geometry_constraints_dispatch.hpp"
#include "geometry_storage.hpp"
#include "../device/cpu_dispatch.hpp"

#include "parallel_mater_geometry_constraints_runtime.cpp"

#include <type_traits>

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

static_assert(std::is_trivially_copyable_v<GeometryConstraintBody>);
static_assert(sizeof(GeometryConstraintBody) == sizeof(AvbdBody_0));
static_assert(alignof(GeometryConstraintBody) == alignof(AvbdBody_0));
static_assert(sizeof(RigidBodyState) == sizeof(RigidBodyState_0));
static_assert(sizeof(RigidBodyId) == sizeof(RigidBodyId_0));
static_assert(sizeof(ContactSchedule) == sizeof(ContactSchedule_0));
static_assert(sizeof(RigidConstraintResource) == sizeof(RigidConstraintResource_0));
static_assert(sizeof(ContactManifold) == sizeof(ContactManifold_0));

} // namespace

cudaError_t geometry_constraints_occupancy(
    int *resident_blocks, std::uint32_t block_size) noexcept {
    if (resident_blocks == nullptr || block_size == 0U)
        return cudaErrorInvalidValue;
    *resident_blocks = 1;
    return cudaSuccess;
}

cudaError_t prepare_geometry_constraints(
    cudaStream_t, const void *parameters, const RigidBodyState *previous,
    RigidBodyState *predicted, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity, RigidBodyId *ids,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations, std::uint32_t) noexcept {
    EntryPointParams_0 args{
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(parameters)),
        const_cast<RigidBodyState_0 *>(
            reinterpret_cast<const RigidBodyState_0 *>(previous)),
        reinterpret_cast<RigidBodyState_0 *>(predicted), body_count,
        as<ContactManifold_0>(manifolds), active_pairs, active_pair_count,
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        reinterpret_cast<RigidBodyId_0 *>(ids),
        reinterpret_cast<AvbdBody_0 *>(bodies),
        as<ContactSchedule_0>(schedule), timestep,
        Vec3_0{gravity.x, gravity.y, gravity.z}, base_iterations,
        requested_iterations, 1U};
    device::dispatch_cpu_serial(prepare_avbd_kernel_Thread, &args);
    return cudaSuccess;
}

cudaError_t solve_geometry_constraints(
    cudaStream_t, const void *parameters, const RigidBodyState *previous,
    RigidBodyState *states, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t) noexcept {
    EntryPointParams_1 args{
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(parameters)),
        const_cast<RigidBodyState_0 *>(
            reinterpret_cast<const RigidBodyState_0 *>(previous)),
        reinterpret_cast<RigidBodyState_0 *>(states), body_count,
        as<ContactManifold_0>(manifolds), active_pairs, active_pair_count,
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        reinterpret_cast<AvbdBody_0 *>(bodies),
        as<ContactSchedule_0>(schedule), timestep};
    device::dispatch_cpu_serial(solve_avbd_kernel_Thread, &args);
    return cudaSuccess;
}

} // namespace parallel_mater::simulation
