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

Vec3_0 vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
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

void predict_rope_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const void *anchor_targets,
    const std::uint8_t *anchor_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    const float gravity[3], float velocity_damping, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *time_advances, std::uint32_t *status) noexcept {
    const auto native = static_cast<cudaStream_t>(stream.native_handle);
    rope_realtime_predict<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U, native>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        as<Vec3_0>(previous_positions), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(anchor_targets)),
        const_cast<unsigned char *>(anchor_active), count, capacity,
        node_spacing, radius, vec3(gravity), velocity_damping, friction,
        maximum_speed, timestep, profile_version, time_advances, status);
}

void project_rope_realtime_constraints(
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, void *constraint_forces, float *correction_used,
    const float *inverse_masses, const float *rest_lengths,
    const float *rest_bend_lengths, const std::uint8_t *anchor_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float structural_compliance, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    rope_realtime_project_constraints<<<1U, 128U, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        as<Vec3_0>(position_deltas), as<Vec3_0>(constraint_forces),
        correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<float *>(rest_lengths),
        const_cast<float *>(rest_bend_lengths),
        const_cast<unsigned char *>(anchor_active), count, capacity,
        node_spacing, structural_compliance, timestep, profile_version,
        status);
}

void apply_rope_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    rope_realtime_apply_position_delta<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(position_deltas)), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<unsigned char *>(anchor_active), count, capacity,
        node_spacing, timestep, profile_version, status);
}

void project_rope_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    float *correction_used, const std::uint8_t *anchor_active,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    bool self_collision, float friction, float timestep,
    std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    rope_realtime_project_contacts<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        const_cast<float *>(inverse_masses), correction_used,
        const_cast<unsigned char *>(anchor_active),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_normals)),
        const_cast<float *>(contact_offsets),
        const_cast<unsigned char *>(contact_active), count, capacity,
        node_spacing, radius, static_cast<std::uint32_t>(self_collision),
        friction, timestep, profile_version, status);
}

void finalize_rope_realtime_self_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    const void *previous_positions, const void *previous_velocities,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    const std::uint8_t *contact_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float radius, bool self_collision, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    rope_realtime_finalize_self_contacts<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_velocities)),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned char *>(anchor_active),
        const_cast<unsigned char *>(contact_active), count, capacity,
        node_spacing, radius, static_cast<std::uint32_t>(self_collision),
        maximum_speed, timestep, profile_version, status);
}

void commit_rope_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    float friction, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    rope_realtime_commit<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), const_cast<float *>(inverse_masses),
        const_cast<unsigned char *>(anchor_active),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_normals)),
        const_cast<float *>(contact_offsets),
        const_cast<unsigned char *>(contact_active), count, capacity,
        node_spacing, radius, friction, maximum_speed, timestep,
        profile_version, status);
}

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
    const bool coupled = soft_a.surface != nullptr ||
        soft_b.surface != nullptr || soft_a.first_anchor || soft_a.last_anchor ||
        soft_b.first_anchor || soft_b.last_anchor;
    if (coupled)
        rope_advance_coupled<<<1U, 128U, 0U, native>>>(data, timestep,
            shader_gravity, first, last,
            const_cast<BodyParameters_0 *>(
                as<BodyParameters_0>(body_parameters)),
            as<RigidBodyState_0>(body_states),
            const_cast<RigidBodyState_0 *>(
                as<RigidBodyState_0>(previous_body_states)),
            const_cast<TriangleMeshResource_0 *>(
                as<TriangleMeshResource_0>(meshes)),
            body_count, first_substep, first_target, second_target);
    else
        rope_advance<<<1U, 128U, 0U, native>>>(data, timestep,
            shader_gravity, first, last,
            const_cast<BodyParameters_0 *>(
                as<BodyParameters_0>(body_parameters)),
            as<RigidBodyState_0>(body_states),
            const_cast<RigidBodyState_0 *>(
                as<RigidBodyState_0>(previous_body_states)),
            const_cast<TriangleMeshResource_0 *>(
                as<TriangleMeshResource_0>(meshes)),
            body_count, first_substep, first_target, second_target);
}

} // namespace parallel_mater::simulation
