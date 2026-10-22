// SPDX-License-Identifier: MIT
#include "geometry_constraints_dispatch.hpp"
#include "geometry_storage.hpp"

#include "parallel_mater_geometry_constraints_runtime.cu"

#include <cooperative_groups.h>
#include <cstddef>
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
static_assert(alignof(RigidBodyState) == alignof(RigidBodyState_0));
static_assert(sizeof(RigidBodyId) == sizeof(RigidBodyId_0));
static_assert(alignof(RigidBodyId) == alignof(RigidBodyId_0));
static_assert(sizeof(ContactSchedule) == sizeof(ContactSchedule_0));
static_assert(alignof(ContactSchedule) == alignof(ContactSchedule_0));
static_assert(offsetof(GeometryConstraintBody, target) ==
              offsetof(AvbdBody_0, target_0));
static_assert(offsetof(GeometryConstraintBody, orientation_delta) ==
              offsetof(AvbdBody_0, orientation_delta_0));
static_assert(offsetof(GeometryConstraintBody, displacement) ==
              offsetof(AvbdBody_0, displacement_0));
static_assert(offsetof(GeometryConstraintBody, inertia) ==
              offsetof(AvbdBody_0, inertia_0));
static_assert(offsetof(GeometryConstraintBody, contact_head) ==
              offsetof(AvbdBody_0, contact_head_0));
static_assert(offsetof(GeometryConstraintBody, joint_head) ==
              offsetof(AvbdBody_0, joint_head_0));
static_assert(offsetof(GeometryConstraintBody, previous_velocity) ==
              offsetof(AvbdBody_0, previous_velocity_0));
static_assert(offsetof(GeometryConstraintBody, history_valid) ==
              offsetof(AvbdBody_0, history_valid_0));
static_assert(offsetof(RigidConstraintResource, options) ==
              offsetof(RigidConstraintResource_0, options_0));
static_assert(offsetof(RigidConstraintResource, state) ==
              offsetof(RigidConstraintResource_0, state_1));
static_assert(offsetof(RigidConstraintResource, geometry) ==
              offsetof(RigidConstraintResource_0, geometry_0));
static_assert(offsetof(RigidConstraintResource, avbd_rows) ==
              offsetof(RigidConstraintResource_0, avbd_rows_0));
static_assert(offsetof(RigidConstraintResource, avbd_count) ==
              offsetof(RigidConstraintResource_0, avbd_count_0));
static_assert(sizeof(RigidConstraintOptions) ==
              sizeof(RigidConstraintOptions_0));
static_assert(alignof(RigidConstraintOptions) ==
              alignof(RigidConstraintOptions_0));
static_assert(offsetof(RigidConstraintOptions, type) ==
              offsetof(RigidConstraintOptions_0, type_0));
static_assert(offsetof(RigidConstraintOptions, body_a) ==
              offsetof(RigidConstraintOptions_0, body_a_0));
static_assert(offsetof(RigidConstraintOptions, local_anchor_a) ==
              offsetof(RigidConstraintOptions_0, local_anchor_a_0));
static_assert(offsetof(RigidConstraintOptions, local_orientation_a) ==
              offsetof(RigidConstraintOptions_0, local_orientation_a_0));
static_assert(offsetof(RigidConstraintOptions, linear_limits) ==
              offsetof(RigidConstraintOptions_0, linear_limits_0));
static_assert(offsetof(RigidConstraintOptions, angular_limits) ==
              offsetof(RigidConstraintOptions_0, angular_limits_0));
static_assert(offsetof(RigidConstraintOptions, linear_springs) ==
              offsetof(RigidConstraintOptions_0, linear_springs_0));
static_assert(offsetof(RigidConstraintOptions, motor) ==
              offsetof(RigidConstraintOptions_0, motor_0));
static_assert(offsetof(RigidConstraintOptions, enabled) ==
              offsetof(RigidConstraintOptions_0, enabled_0));
static_assert(offsetof(RigidConstraintOptions, solver_iterations) ==
              offsetof(RigidConstraintOptions_0, solver_iterations_0));
static_assert(offsetof(ContactManifold, initial_relative_position) ==
              offsetof(ContactManifold_0, initial_relative_position_0));
static_assert(offsetof(ContactManifold, response_slot) ==
              offsetof(ContactManifold_0, response_slot_0));
static_assert(sizeof(HingeContactFrame) == sizeof(HingeContactFrame_0));
static_assert(alignof(HingeContactFrame) == alignof(HingeContactFrame_0));
static_assert(sizeof(Contact) == sizeof(Contact_0));
static_assert(alignof(Contact) == alignof(Contact_0));
static_assert(sizeof(ContactManifold) == sizeof(ContactManifold_0));
static_assert(alignof(ContactManifold) == alignof(ContactManifold_0));
static_assert(offsetof(Contact, body_hinge) ==
              offsetof(Contact_0, body_hinge_0));
static_assert(offsetof(Contact, accumulated_normal_impulse) ==
              offsetof(Contact_0, accumulated_normal_impulse_0));
static_assert(offsetof(Contact, impact_fraction) ==
              offsetof(Contact_0, impact_fraction_0));
static_assert(offsetof(Contact, avbd_dual) ==
              offsetof(Contact_0, avbd_dual_0));
static_assert(offsetof(Contact, avbd_error) ==
              offsetof(Contact_0, avbd_error_0));
static_assert(offsetof(Contact, avbd_anchor_a) ==
              offsetof(Contact_0, avbd_anchor_a_0));
static_assert(offsetof(Contact, avbd_cached) ==
              offsetof(Contact_0, avbd_cached_0));

} // namespace

PmResult geometry_constraints_occupancy(
    int *resident_blocks, std::uint32_t block_size) noexcept {
    return device::from_driver_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        resident_blocks, solve_avbd_kernel, block_size, 0U));
}

PmResult prepare_geometry_constraints(device::ExecutionQueue stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *predicted, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations,
    std::uint32_t profile_version,
    std::uint32_t grid_blocks) noexcept {
    const Vec3_0 shader_gravity{gravity.x, gravity.y, gravity.z};
    prepare_avbd_kernel<<<1U, 128U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(parameters)),
        const_cast<RigidBodyState_0 *>(
            reinterpret_cast<const RigidBodyState_0 *>(previous)),
        reinterpret_cast<RigidBodyState_0 *>(predicted), body_count,
        as<ContactManifold_0>(manifolds), active_pairs, active_pair_count,
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        reinterpret_cast<RigidBodyId_0 *>(ids),
        reinterpret_cast<AvbdBody_0 *>(bodies),
        as<ContactSchedule_0>(schedule), timestep, shader_gravity,
        base_iterations, requested_iterations, profile_version, grid_blocks);
    return device::from_driver_error(cudaGetLastError());
}

PmResult solve_geometry_constraints(device::ExecutionQueue stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *states, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t grid_blocks) noexcept {
    auto *shader_parameters =
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(parameters));
    auto *shader_previous = const_cast<RigidBodyState_0 *>(
        reinterpret_cast<const RigidBodyState_0 *>(previous));
    auto *shader_states = reinterpret_cast<RigidBodyState_0 *>(states);
    auto *shader_manifolds = as<ContactManifold_0>(manifolds);
    auto *shader_constraints = as<RigidConstraintResource_0>(constraints);
    auto *shader_bodies = reinterpret_cast<AvbdBody_0 *>(bodies);
    auto *shader_schedule = as<ContactSchedule_0>(schedule);

    if (grid_blocks == 1U) {
        solve_avbd_kernel<<<1U, 32U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(shader_parameters,
            shader_previous, shader_states, body_count, shader_manifolds,
            active_pairs, active_pair_count, shader_constraints,
            constraint_capacity, shader_bodies, shader_schedule, timestep);
        return device::from_driver_error(cudaGetLastError());
    }

    void *arguments[]{&shader_parameters, &shader_previous, &shader_states,
        &body_count, &shader_manifolds, &active_pairs, &active_pair_count,
        &shader_constraints, &constraint_capacity, &shader_bodies,
        &shader_schedule, &timestep};
    return device::from_driver_error(cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(solve_avbd_kernel), dim3(grid_blocks),
        dim3(32U), arguments, 0U, static_cast<cudaStream_t>(stream.native_handle)));
}

} // namespace parallel_mater::simulation
