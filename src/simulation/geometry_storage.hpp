// SPDX-License-Identifier: MIT
#pragma once

#include <simulation/runtime.hpp>
#include "constraint_layout.hpp"

#include <climits>
#include <cmath>
#include <cstdint>

// Backend-neutral host storage mirrored by shaders/types.slang and
// shaders/geometry.slang. This header intentionally contains no GPU kernels.

namespace parallel_mater::simulation {

// Host storage mirrored by AvbdBody in shaders/geometry_constraints.slang.
// It is scheduling state only; the constraint equations live in Slang.
struct GeometryConstraintBody {
    RigidBodyState target{};
    Quaternion orientation_delta{};
    avbd::Vector6 displacement{};
    avbd::Block inertia{};
    std::uint32_t contact_head{UINT32_MAX};
    std::uint32_t joint_head{UINT32_MAX};
    std::uint32_t color{UINT32_MAX};
    std::uint32_t parent{};
    std::uint32_t impact{};
    Vec3 previous_velocity{};
    std::uint32_t history_valid{};
};

struct BodyParameters {
    MotionType motion{};
    TriangleMeshId mesh{};
    float inverse_mass{};
    Vec3 inverse_inertia_local{};
    float friction{};
    float restitution{};
    float linear_damping{};
    float angular_damping{};
    float maximum_linear_speed{};
    float maximum_angular_speed{};
    float collision_margin{};
    std::uint64_t user_data{};
};

struct BodyAccumulator {
    Vec3 force{};
    Vec3 torque{};
    Vec3 impulse{};
    Vec3 angular_impulse{};
};

struct KinematicTarget {
    RigidBodyState state{};
    std::uint32_t active{};
};

struct RigidConstraintGeometry {
    std::uint32_t dense_a{};
    std::uint32_t dense_b{};
};

struct RigidConstraintResource {
    RigidConstraintOptions options{};
    RigidConstraintState state{};
    std::uint32_t generation{1U};
    std::uint32_t alive{};
    RigidConstraintGeometry geometry{};
    avbd::Row avbd_rows[18]{};
    std::uint32_t avbd_count{};
    std::uint32_t avbd_next_a{};
    std::uint32_t avbd_next_b{};
};

struct FixedContactProjection {
    std::uint32_t root{};
};

struct HingeContactFrame {
    Vec3 anchor{};
    Vec3 axis{};
    Vec3 local_anchor{};
    float inverse_mass{};
    float inverse_moment{};
    std::uint32_t present{};
    std::uint32_t fixed{};
    std::uint32_t axial{};
    std::uint32_t axial_rotation{};
    std::uint32_t fixed_member{};
    std::uint32_t point_member{};
    std::uint32_t static_body{};
};

struct Contact {
    Vec3 normal{};
    Vec3 point{};
    float penetration{};
    HingeContactFrame body_hinge{};
    HingeContactFrame collider_hinge{};
    float accumulated_normal_impulse{};
    Vec3 accumulated_friction_impulse{};
    float initial_normal_speed{};
    std::uint32_t warm_started{};
    std::uint32_t persistent{};
    float impact_fraction{1.0F};
    float reported_normal_impulse{};
    Vec3 reported_friction_impulse{};
    avbd::Dual avbd_dual[3]{};
    Vec3 avbd_error{};
    Vec3 avbd_anchor_a{};
    Vec3 avbd_anchor_b{};
    std::uint32_t avbd_cached{};
};

struct ContactManifold {
    Contact contacts[8]{};
    std::uint32_t count{};
    Vec3 initial_relative_position{};
    std::uint32_t face_patch{};
    std::uint32_t response_slot{UINT32_MAX};
    std::uint32_t avbd_next_a{};
    std::uint32_t avbd_next_b{};
};

struct CachedContact {
    Vec3 local_point{};
    Vec3 normal{};
    float normal_impulse{};
    Vec3 friction_impulse{};
    avbd::Dual avbd_dual[3]{};
    Vec3 avbd_anchor_a{};
    Vec3 avbd_anchor_b{};
};

struct CachedContactPair {
    std::uint32_t pair{};
    RigidBodyId body{};
    RigidBodyId collider{};
    std::uint64_t epoch{};
    float timestep{};
    std::uint32_t count{};
    CachedContact contacts[8]{};
};

struct ContactSchedule {
    std::uint32_t counts[29]{};
    std::uint32_t offsets[29]{};
    std::uint32_t budget{};
    std::uint32_t remaining{};
    std::uint32_t island_count{};
    std::uint32_t early_exit_count{};
    std::uint32_t maximum_passes{};
    std::uint32_t blocks{};
    std::uint32_t candidates{};
    std::uint32_t contacts{};
};

struct LeafPair {
    std::uint32_t body_first{};
    std::uint32_t body_count{};
    std::uint32_t collider_first{};
    std::uint32_t collider_count{};
};

constexpr std::uint32_t k_max_leaf_pairs_per_body_pair = 512U;
constexpr std::uint32_t k_small_rigid_leaf_body_capacity = 8U;
constexpr std::uint32_t k_small_rigid_leaf_pair_capacity = 4096U;
constexpr std::uint32_t k_shared_rigid_leaf_capacity = 1024U;
constexpr std::uint32_t k_rigid_leaf_blocks_per_pair = 16U;
constexpr std::uint32_t k_leaf_pair_cache_slots_per_body = 8U;
constexpr std::uint32_t k_minimum_leaf_pair_cache_slots = 4'096U;

struct BvhNode {
    Vec3 minimum{};
    Vec3 maximum{};
    std::uint32_t left{};
    std::uint32_t right{};
    std::uint32_t first_triangle{};
    std::uint32_t triangle_count{};
};

struct WorldAabb {
    Vec3 minimum{};
    Vec3 maximum{};
};

struct CollisionPlane {
    Vec3 normal{};
    float offset{};
};

struct TriangleMeshResource {
    Vec3 *vertices{};
    std::uint32_t *indices{};
    std::uint32_t vertex_count{};
    std::uint32_t index_count{};
    std::uint32_t generation{};
    std::uint32_t alive{};
    Vec3 minimum{};
    Vec3 maximum{};
    Vec3 bounding_center{};
    float bounding_radius{};
    Vec3 unit_inertia{};
    BvhNode *bvh_nodes{};
    std::uint32_t bvh_node_count{};
    std::uint32_t *bvh_leaves{};
    std::uint32_t bvh_leaf_count{};
    CollisionPlane *solid_planes{};
    std::uint32_t solid_unique_plane_count{};
    Vec3 *shell_normals{};
};

inline float component(Vec3 value, std::uint32_t axis) noexcept {
    return axis == 0U ? value.x : axis == 1U ? value.y : value.z;
}

inline Vec3 component_min(Vec3 first, Vec3 second) noexcept {
    return {fminf(first.x, second.x), fminf(first.y, second.y),
            fminf(first.z, second.z)};
}

inline Vec3 component_max(Vec3 first, Vec3 second) noexcept {
    return {fmaxf(first.x, second.x), fmaxf(first.y, second.y),
            fmaxf(first.z, second.z)};
}

inline bool bounds_overlap(Vec3 minimum_a, Vec3 maximum_a,
                           Vec3 minimum_b, Vec3 maximum_b) noexcept {
    return minimum_a.x <= maximum_b.x && maximum_a.x >= minimum_b.x &&
           minimum_a.y <= maximum_b.y && maximum_a.y >= minimum_b.y &&
           minimum_a.z <= maximum_b.z && maximum_a.z >= minimum_b.z;
}

inline Vec3 geometry_add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 geometry_subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 geometry_multiply(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline float geometry_dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 geometry_cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline float geometry_length_squared(Vec3 value) noexcept {
    return geometry_dot(value, value);
}

inline Vec3 geometry_rotate(Quaternion orientation, Vec3 value) noexcept {
    const Vec3 q{orientation.x, orientation.y, orientation.z};
    const Vec3 twice_cross = geometry_multiply(geometry_cross(q, value), 2.0F);
    return geometry_add(value, geometry_add(
        geometry_multiply(twice_cross, orientation.w),
        geometry_cross(q, twice_cross)));
}

inline Vec3 inverse_rotate(Quaternion orientation, Vec3 value) noexcept {
    return geometry_rotate(
        {-orientation.x, -orientation.y, -orientation.z, orientation.w},
        value);
}

inline Vec3 closest_on_triangle(Vec3 point, Vec3 a, Vec3 b, Vec3 c,
                                bool stable_face = false) noexcept {
    const Vec3 ab = geometry_subtract(b, a);
    const Vec3 ac = geometry_subtract(c, a);
    const Vec3 ap = geometry_subtract(point, a);
    const float d1 = geometry_dot(ab, ap);
    const float d2 = geometry_dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) return a;
    const Vec3 bp = geometry_subtract(point, b);
    const float d3 = geometry_dot(ab, bp);
    const float d4 = geometry_dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F)
        return geometry_add(a, geometry_multiply(ab, d1 / (d1 - d3)));
    const Vec3 cp = geometry_subtract(point, c);
    const float d5 = geometry_dot(ab, cp);
    const float d6 = geometry_dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F)
        return geometry_add(a, geometry_multiply(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F)
        return geometry_add(b, geometry_multiply(geometry_subtract(c, b),
            (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    if (stable_face) {
        const Vec3 normal = geometry_cross(ab, ac);
        return geometry_subtract(point, geometry_multiply(normal,
            geometry_dot(ap, normal) / geometry_length_squared(normal)));
    }
    const float inverse = 1.0F / (va + vb + vc);
    return geometry_add(a, geometry_add(
        geometry_multiply(ab, vb * inverse),
        geometry_multiply(ac, vc * inverse)));
}

inline Vec3 transform_point(const RigidBodyState &state, Vec3 point) noexcept {
    return geometry_add(state.position, geometry_rotate(state.orientation, point));
}

inline bool axis_enabled(std::uint32_t mask, std::uint32_t axis) noexcept {
    return (mask & static_cast<std::uint8_t>(1U << axis)) != 0U;
}

} // namespace parallel_mater::simulation
