// SPDX-License-Identifier: MIT
#pragma once
#include "schedule_context.hpp"
#include "simulation/rope_dispatch.hpp"
#include "simulation/smoke_dispatch.hpp"
#include "simulation/geometry_storage.hpp"
#include "simulation/paint_storage.hpp"

namespace parallel_mater::metal {
// Explicit host/native layouts, not reinterpret-casts of CPU pointer records.
#define PM_MEMBER(Type, member) {#member, offsetof(Type, member), sizeof(Type::member)}
inline RelocatedRecord rope_record(const simulation::RopeDispatchData &value) {
    using T = simulation::RopeDispatchData;
    static constexpr HostMember members[]{
        PM_MEMBER(T, options),
        PM_MEMBER(T, count),
        PM_MEMBER(T, positions),
        PM_MEMBER(T, previous),
        PM_MEMBER(T, velocities),
        PM_MEMBER(T, constraint_forces),
        PM_MEMBER(T, contact_forces),
        PM_MEMBER(T, fluid_contact_forces),
        PM_MEMBER(T, soft_body_contact_forces),
        PM_MEMBER(T, soft_anchor_positions),
        PM_MEMBER(T, soft_anchor_velocities),
        PM_MEMBER(T, soft_anchor_impulses),
        PM_MEMBER(T, soft_anchor_inverse_masses),
        PM_MEMBER(T, soft_first),
        PM_MEMBER(T, soft_last),
        PM_MEMBER(T, directions),
        PM_MEMBER(T, scratch),
        PM_MEMBER(T, body_translation),
        PM_MEMBER(T, body_rotation),
        PM_MEMBER(T, normals),
        PM_MEMBER(T, normals2),
        PM_MEMBER(T, rest),
        PM_MEMBER(T, lambda),
        PM_MEMBER(T, solid_hint),
        PM_MEMBER(T, body_capacity),
        PM_MEMBER(T, first_attachment_contact_skip),
        PM_MEMBER(T, last_attachment_contact_skip),
        {"options.centerline", offsetof(T, options) + offsetof(RopeOptions, centerline), sizeof(RopeOptions::centerline)},
        {"options.node_spacing", offsetof(T, options) + offsetof(RopeOptions, node_spacing), sizeof(RopeOptions::node_spacing)},
        {"options.radius", offsetof(T, options) + offsetof(RopeOptions, radius), sizeof(RopeOptions::radius)},
        {"options.mass", offsetof(T, options) + offsetof(RopeOptions, mass), sizeof(RopeOptions::mass)},
        {"options.stretch_compliance", offsetof(T, options) + offsetof(RopeOptions, stretch_compliance), sizeof(RopeOptions::stretch_compliance)},
        {"options.velocity_damping", offsetof(T, options) + offsetof(RopeOptions, velocity_damping), sizeof(RopeOptions::velocity_damping)},
        {"options.maximum_substep_timestep", offsetof(T, options) + offsetof(RopeOptions, maximum_substep_timestep), sizeof(RopeOptions::maximum_substep_timestep)},
        {"options.friction", offsetof(T, options) + offsetof(RopeOptions, friction), sizeof(RopeOptions::friction)},
        {"options.maximum_speed", offsetof(T, options) + offsetof(RopeOptions, maximum_speed), sizeof(RopeOptions::maximum_speed)},
        {"options.solver_iterations", offsetof(T, options) + offsetof(RopeOptions, solver_iterations), sizeof(RopeOptions::solver_iterations)},
        {"options.self_collision", offsetof(T, options) + offsetof(RopeOptions, self_collision), sizeof(RopeOptions::self_collision)},
        {"options.first", offsetof(T, options) + offsetof(RopeOptions, first), sizeof(RopeOptions::first)},
        {"options.last", offsetof(T, options) + offsetof(RopeOptions, last), sizeof(RopeOptions::last)},
        {"options.centerline.data", offsetof(T, options) + offsetof(RopeOptions, centerline) + offsetof(HostSpan<const Vec3>, data), sizeof(HostSpan<const Vec3>::data)},
        {"options.centerline.size", offsetof(T, options) + offsetof(RopeOptions, centerline) + offsetof(HostSpan<const Vec3>, size), sizeof(HostSpan<const Vec3>::size)},
        {"options.first.body", offsetof(T, options) + offsetof(RopeOptions, first) + offsetof(RopeAttachment, body), sizeof(RopeAttachment::body)},
        {"options.first.local_anchor", offsetof(T, options) + offsetof(RopeOptions, first) + offsetof(RopeAttachment, local_anchor), sizeof(RopeAttachment::local_anchor)},
        {"options.first.enabled", offsetof(T, options) + offsetof(RopeOptions, first) + offsetof(RopeAttachment, enabled), sizeof(RopeAttachment::enabled)},
        {"options.last.body", offsetof(T, options) + offsetof(RopeOptions, last) + offsetof(RopeAttachment, body), sizeof(RopeAttachment::body)},
        {"options.last.local_anchor", offsetof(T, options) + offsetof(RopeOptions, last) + offsetof(RopeAttachment, local_anchor), sizeof(RopeAttachment::local_anchor)},
        {"options.last.enabled", offsetof(T, options) + offsetof(RopeOptions, last) + offsetof(RopeAttachment, enabled), sizeof(RopeAttachment::enabled)},
    };
    return {std::as_bytes(std::span{&value, 1}), members};
}
inline RelocatedRecord target_record(const simulation::RopeSoftTargetDispatch &value) {
    using T = simulation::RopeSoftTargetDispatch;
    static constexpr HostMember members[]{
        PM_MEMBER(T, surface),
        PM_MEMBER(T, previous_surface),
        PM_MEMBER(T, velocities),
        PM_MEMBER(T, bounds),
        PM_MEMBER(T, tree),
        PM_MEMBER(T, order),
        PM_MEMBER(T, indices),
        PM_MEMBER(T, bindings),
        PM_MEMBER(T, inverse_masses),
        PM_MEMBER(T, node_impulses),
        PM_MEMBER(T, contact_count),
        PM_MEMBER(T, maximum_penetration),
        PM_MEMBER(T, triangle_count),
        PM_MEMBER(T, surface_vertex_count),
        PM_MEMBER(T, distance),
        PM_MEMBER(T, friction),
        PM_MEMBER(T, orientation),
        PM_MEMBER(T, first_anchor),
        PM_MEMBER(T, last_anchor),
    };
    return {std::as_bytes(std::span{&value, 1}), members};
}
inline RelocatedRecord mesh_record(const simulation::TriangleMeshResource &value) {
    using T = simulation::TriangleMeshResource;
    static constexpr HostMember members[]{
        PM_MEMBER(T, vertices),
        PM_MEMBER(T, indices),
        PM_MEMBER(T, vertex_count),
        PM_MEMBER(T, index_count),
        PM_MEMBER(T, generation),
        PM_MEMBER(T, alive),
        PM_MEMBER(T, minimum),
        PM_MEMBER(T, maximum),
        PM_MEMBER(T, bounding_center),
        PM_MEMBER(T, bounding_radius),
        PM_MEMBER(T, unit_inertia),
        PM_MEMBER(T, bvh_nodes),
        PM_MEMBER(T, bvh_node_count),
        PM_MEMBER(T, bvh_leaves),
        PM_MEMBER(T, bvh_leaf_count),
        PM_MEMBER(T, solid_planes),
        PM_MEMBER(T, solid_unique_plane_count),
        PM_MEMBER(T, shell_normals),
    };
    return {std::as_bytes(std::span{&value, 1}), members};
}
inline RelocatedRecord grid_record(const simulation::SmokeGridDispatchField &value) {
    using T = simulation::SmokeGridDispatchField;
    static constexpr HostMember members[]{
        {"n", offsetof(T, resolution), sizeof(T::resolution)},
        PM_MEMBER(T, height),
        PM_MEMBER(T, minimum),
        PM_MEMBER(T, spacing),
        PM_MEMBER(T, velocity),
        PM_MEMBER(T, density),
        PM_MEMBER(T, pressure),
        {"face", offsetof(T, face_velocity), sizeof(T::face_velocity)},
        PM_MEMBER(T, vorticity),
        PM_MEMBER(T, strain),
    };
    return {std::as_bytes(std::span{&value, 1}), members};
}
inline RelocatedRecord paint_record(const simulation::PaintFieldResource &value) {
    using T = simulation::PaintFieldResource;
    using Options = PaintFieldOptions;
    static constexpr HostMember members[]{
        PM_MEMBER(T, options), PM_MEMBER(T, uvs), PM_MEMBER(T, pixels),
        PM_MEMBER(T, generation), PM_MEMBER(T, alive),
        {"options.body", offsetof(T, options) + offsetof(Options, body), sizeof(Options::body)},
        {"options.mesh", offsetof(T, options) + offsetof(Options, mesh), sizeof(Options::mesh)},
        {"options.cloth", offsetof(T, options) + offsetof(Options, cloth), sizeof(Options::cloth)},
        {"options.vertex_uvs", offsetof(T, options) + offsetof(Options, vertex_uvs), sizeof(Options::vertex_uvs)},
        {"options.vertex_uvs.data", offsetof(T, options) + offsetof(Options, vertex_uvs) + offsetof(DeviceSpan<const Vec2>, data), sizeof(DeviceSpan<const Vec2>::data)},
        {"options.vertex_uvs.size", offsetof(T, options) + offsetof(Options, vertex_uvs) + offsetof(DeviceSpan<const Vec2>, size), sizeof(DeviceSpan<const Vec2>::size)},
        {"options.width", offsetof(T, options) + offsetof(Options, width), sizeof(Options::width)},
        {"options.height", offsetof(T, options) + offsetof(Options, height), sizeof(Options::height)},
    };
    return {std::as_bytes(std::span{&value, 1}), members};
}
#undef PM_MEMBER
} // namespace parallel_mater::metal
