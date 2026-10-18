// SPDX-License-Identifier: MIT
#include "rope_dispatch.hpp"

#include "parallel_mater_rope_runtime.cu"

#include <cstring>
#include <type_traits>

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

#if defined(PM_VULKAN_GENERATED_DISPATCH)
template <typename T>
T *device_pointer(vulkan::ScheduleContext &execution,
                  const void *value) noexcept {
    return reinterpret_cast<T *>(execution.address(value));
}

RopeAttachment_0 shader_attachment(const RopeAttachment &value) noexcept {
    return {{value.body.index, value.body.generation},
            {value.local_anchor.x, value.local_anchor.y, value.local_anchor.z},
            value.enabled};
}

RopeOptions_0 shader_options(const RopeOptions &value) noexcept {
    RopeOptions_0 result{};
    // Creation-only host input; the solver does not read it.
    result.centerline = {};
    result.node_spacing = value.node_spacing;
    result.radius = value.radius;
    result.mass = value.mass;
    result.stretch_compliance = value.stretch_compliance;
    result.velocity_damping = value.velocity_damping;
    result.maximum_substep_timestep = value.maximum_substep_timestep;
    result.friction = value.friction;
    result.maximum_speed = value.maximum_speed;
    result.solver_iterations = value.solver_iterations;
    result.self_collision = value.self_collision;
    result.first = shader_attachment(value.first);
    result.last = shader_attachment(value.last);
    return result;
}

RopeData_0 *shader_rope(const RopeDispatchData &value,
    cudaStream_t stream) noexcept {
    auto *execution = vulkan::context(stream);
    if (!execution) return nullptr;
    RopeData_0 result{};
    result.options = shader_options(value.options);
    result.count = value.count;
    result.positions = device_pointer<Vec3_0>(*execution, value.positions);
    result.previous = device_pointer<Vec3_0>(*execution, value.previous);
    result.velocities = device_pointer<Vec3_0>(*execution, value.velocities);
    result.constraint_forces =
        device_pointer<Vec3_0>(*execution, value.constraint_forces);
    result.contact_forces =
        device_pointer<Vec3_0>(*execution, value.contact_forces);
    result.fluid_contact_forces =
        device_pointer<Vec3_0>(*execution, value.fluid_contact_forces);
    result.soft_body_contact_forces =
        device_pointer<Vec3_0>(*execution, value.soft_body_contact_forces);
    result.soft_anchor_positions =
        device_pointer<Vec3_0>(*execution, value.soft_anchor_positions);
    result.soft_anchor_velocities =
        device_pointer<Vec3_0>(*execution, value.soft_anchor_velocities);
    result.soft_anchor_impulses =
        device_pointer<Vec3_0>(*execution, value.soft_anchor_impulses);
    result.soft_anchor_inverse_masses =
        device_pointer<float>(*execution, value.soft_anchor_inverse_masses);
    result.soft_first = value.soft_first;
    result.soft_last = value.soft_last;
    result.directions = device_pointer<Vec3_0>(*execution, value.directions);
    result.scratch = device_pointer<Vec3_0>(*execution, value.scratch);
    result.body_translation =
        device_pointer<Vec3_0>(*execution, value.body_translation);
    result.body_rotation =
        device_pointer<Vec3_0>(*execution, value.body_rotation);
    result.normals = device_pointer<Vec3_0>(*execution, value.normals);
    result.normals2 = device_pointer<Vec3_0>(*execution, value.normals2);
    result.rest = device_pointer<float>(*execution, value.rest);
    result.lambda = device_pointer<float>(*execution, value.lambda);
    result.solid_hint =
        device_pointer<std::uint32_t>(*execution, value.solid_hint);
    result.body_capacity = value.body_capacity;
    result.first_attachment_contact_skip = value.first_attachment_contact_skip;
    result.last_attachment_contact_skip = value.last_attachment_contact_skip;
    return as<RopeData_0>(execution->stage(
        std::as_bytes(std::span{&result, std::size_t{1}})));
}

RopeSoftTarget_0 *shader_target(
    const RopeSoftTargetDispatch &value, cudaStream_t stream) noexcept {
    auto *execution = vulkan::context(stream);
    if (!execution) return nullptr;
    RopeSoftTarget_0 result{};
    result.surface = device_pointer<Vec3_0>(*execution, value.surface);
    result.previous_surface =
        device_pointer<Vec3_0>(*execution, value.previous_surface);
    result.velocities = device_pointer<Vec3_0>(*execution, value.velocities);
    result.bounds = device_pointer<Vec3_0>(*execution, value.bounds);
    result.tree = device_pointer<BvhNode_0>(*execution, value.tree);
    result.order = device_pointer<std::uint32_t>(*execution, value.order);
    result.indices = device_pointer<std::uint32_t>(*execution, value.indices);
    result.bindings = device_pointer<SoftBodySurfaceBinding_0>(
        *execution, value.bindings);
    result.inverse_masses =
        device_pointer<float>(*execution, value.inverse_masses);
    result.node_impulses =
        device_pointer<Vec3_0>(*execution, value.node_impulses);
    result.contact_count =
        device_pointer<std::uint32_t>(*execution, value.contact_count);
    result.maximum_penetration =
        device_pointer<float>(*execution, value.maximum_penetration);
    result.triangle_count = value.triangle_count;
    result.surface_vertex_count = value.surface_vertex_count;
    result.distance = value.distance;
    result.friction = value.friction;
    result.orientation = value.orientation;
    result.first_anchor = value.first_anchor;
    result.last_anchor = value.last_anchor;
    return as<RopeSoftTarget_0>(execution->stage(
        std::as_bytes(std::span{&result, std::size_t{1}})));
}
#else
RopeData_0 shader_rope(const RopeDispatchData &value) noexcept {
    static_assert(std::is_trivially_copyable_v<RopeDispatchData>);
    static_assert(sizeof(RopeDispatchData) == sizeof(RopeData_0));
    static_assert(alignof(RopeDispatchData) == alignof(RopeData_0));
    RopeData_0 result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

RopeSoftTarget_0 shader_target(
    const RopeSoftTargetDispatch &value) noexcept {
    static_assert(std::is_trivially_copyable_v<RopeSoftTargetDispatch>);
    static_assert(sizeof(RopeSoftTargetDispatch) == sizeof(RopeSoftTarget_0));
    static_assert(alignof(RopeSoftTargetDispatch) == alignof(RopeSoftTarget_0));
    RopeSoftTarget_0 result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
#endif

} // namespace

void advance_rope(device::ExecutionQueue stream, const RopeDispatchData &rope,
    float timestep, Vec3 gravity, int first, int last,
    const void *body_parameters, void *body_states,
    const void *previous_body_states, const void *meshes,
    std::uint32_t body_count, bool first_substep,
    const RopeSoftTargetDispatch &soft_a,
    const RopeSoftTargetDispatch &soft_b) noexcept {
    const auto native = static_cast<cudaStream_t>(stream.native_handle);
#if defined(PM_VULKAN_GENERATED_DISPATCH)
    auto data = shader_rope(rope, native);
    auto first_target = shader_target(soft_a, native);
    auto second_target = shader_target(soft_b, native);
#else
    auto data = shader_rope(rope);
    auto first_target = shader_target(soft_a);
    auto second_target = shader_target(soft_b);
#endif
    const Vec3_0 shader_gravity{gravity.x, gravity.y, gravity.z};
    rope_advance<<<1U, 128U, 0U, native>>>(data, timestep, shader_gravity,
        first, last,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(previous_body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, first_substep, first_target, second_target);
}

} // namespace parallel_mater::simulation
