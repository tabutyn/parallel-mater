// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace parallel_mater {
namespace {

constexpr float k_epsilon = 1.0e-6F;
constexpr std::uint32_t k_invalid_dense = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t k_soft_contact_cleanup_passes = 8U;

[[nodiscard]] Status success() noexcept { return {}; }

[[nodiscard]] Status failure(StatusCode code, const char *message,
                             cudaError_t cuda_error = cudaSuccess) noexcept {
    return {code, cuda_error, message};
}

[[nodiscard]] Status cuda_failure(cudaError_t error, const char *message) noexcept {
    return failure(StatusCode::cuda_failure, message, error);
}

__host__ __device__ Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vec3 multiply(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

__host__ __device__ float dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__host__ __device__ float length_squared(Vec3 value) noexcept {
    return dot(value, value);
}

__host__ __device__ float vector_length(Vec3 value) noexcept {
    return sqrtf(length_squared(value));
}

__host__ __device__ float clamp_scalar(float value, float minimum,
                                       float maximum) noexcept {
    return fminf(fmaxf(value, minimum), maximum);
}

__host__ __device__ Vec3 normalized_or(Vec3 value, Vec3 fallback) noexcept {
    const float squared = length_squared(value);
    if (squared <= k_epsilon * k_epsilon) {
        return fallback;
    }
    return multiply(value, rsqrtf(squared));
}

__host__ __device__ Quaternion conjugate(Quaternion value) noexcept {
    return {-value.x, -value.y, -value.z, value.w};
}

__host__ __device__ Quaternion quaternion_multiply(Quaternion a,
                                                   Quaternion b) noexcept {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

__host__ __device__ Quaternion normalized_quaternion(Quaternion value) noexcept {
    const float squared = value.x * value.x + value.y * value.y +
                          value.z * value.z + value.w * value.w;
    if (squared <= k_epsilon * k_epsilon) {
        return {};
    }
    const float inverse = rsqrtf(squared);
    return {value.x * inverse, value.y * inverse, value.z * inverse,
            value.w * inverse};
}

__host__ __device__ Vec3 rotate(Quaternion orientation, Vec3 value) noexcept {
    const Vec3 q{orientation.x, orientation.y, orientation.z};
    const Vec3 twice_cross = multiply(cross(q, value), 2.0F);
    return add(value,
               add(multiply(twice_cross, orientation.w), cross(q, twice_cross)));
}

__host__ __device__ Vec3 inverse_rotate(Quaternion orientation,
                                       Vec3 value) noexcept {
    return rotate(conjugate(orientation), value);
}

__host__ __device__ Vec3 clamp_length(Vec3 value, float maximum) noexcept {
    const float squared = length_squared(value);
    if (squared <= maximum * maximum || squared <= k_epsilon * k_epsilon) {
        return value;
    }
    return multiply(value, maximum * rsqrtf(squared));
}

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
    bool active{};
};

struct Contact {
    Vec3 normal{};
    Vec3 point{};
    float penetration{};
    bool hit{};
};

struct ContactManifold {
    Contact contacts[8]{};
    std::uint32_t count{};
};

struct LeafPair {
    std::uint32_t body_first{};
    std::uint32_t body_count{};
    std::uint32_t collider_first{};
    std::uint32_t collider_count{};
};

constexpr std::uint32_t k_max_leaf_pairs_per_body_pair = 512U;
// Keep the fast leaf-pair cache proportional to body capacity. Dense worlds
// retain exact contacts through the serial fallback instead of reserving one
// 512-entry cache for every possible body pair.
constexpr std::uint32_t k_leaf_pair_cache_slots_per_body = 8U;
constexpr std::uint32_t k_minimum_leaf_pair_cache_slots = 4'096U;
constexpr std::uint32_t k_leaf_pair_overflow =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t k_contact_color_count = 32U;
constexpr std::uint8_t k_contact_color_overflow = 0xffU;

struct AppliedContactImpulse {
    float normal{};
    Vec3 friction{};
};

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
    bool alive{};
    Vec3 minimum{};
    Vec3 maximum{};
    Vec3 bounding_center{};
    float bounding_radius{};
    Vec3 unit_inertia{};
    BvhNode *bvh_nodes{};
    std::uint32_t bvh_node_count{};
    std::uint32_t *bvh_leaves{};
    std::uint32_t bvh_leaf_count{};
    // Present only when the authored triangles form a closed convex solid.
    CollisionPlane *solid_planes{};
};

__device__ float rotational_motion_bound(
    const RigidBodyState &previous, const RigidBodyState &current,
    const TriangleMeshResource &mesh) noexcept {
    const float orientation_dot = clamp_scalar(
        fabsf(previous.orientation.x * current.orientation.x +
              previous.orientation.y * current.orientation.y +
              previous.orientation.z * current.orientation.z +
              previous.orientation.w * current.orientation.w),
        0.0F, 1.0F);
    const float sine_half_angle =
        sqrtf(fmaxf(0.0F, 1.0F - orientation_dot * orientation_dot));
    const Vec3 maximum_absolute{
        fmaxf(fabsf(mesh.minimum.x), fabsf(mesh.maximum.x)),
        fmaxf(fabsf(mesh.minimum.y), fabsf(mesh.maximum.y)),
        fmaxf(fabsf(mesh.minimum.z), fabsf(mesh.maximum.z))};
    return 2.0F * vector_length(maximum_absolute) * sine_half_angle;
}

__device__ bool requires_swept_contact(
    const RigidBodyState &previous, const RigidBodyState &current,
    const TriangleMeshResource &mesh, float threshold) noexcept {
    const float translation =
        vector_length(subtract(current.position, previous.position));
    return translation + rotational_motion_bound(previous, current, mesh) >
           threshold;
}

__device__ bool requires_swept_pair_contact(
    const RigidBodyState &previous_body, const RigidBodyState &body,
    const TriangleMeshResource &body_mesh,
    const RigidBodyState &previous_collider,
    const RigidBodyState &collider,
    const TriangleMeshResource &collider_mesh, float threshold) noexcept {
    const Vec3 relative_translation = subtract(
        subtract(body.position, previous_body.position),
        subtract(collider.position, previous_collider.position));
    const float body_rotation =
        rotational_motion_bound(previous_body, body, body_mesh);
    const float collider_rotation = rotational_motion_bound(
        previous_collider, collider, collider_mesh);
    return vector_length(relative_translation) + body_rotation +
               collider_rotation > threshold;
}

__device__ bool bounding_spheres_may_contact(
    const RigidBodyState &previous_body, const RigidBodyState &body,
    const TriangleMeshResource &body_mesh,
    const RigidBodyState &previous_collider,
    const RigidBodyState &collider,
    const TriangleMeshResource &collider_mesh, float margin) noexcept {
    const Vec3 previous_relative = subtract(
        add(previous_body.position,
            rotate(previous_body.orientation, body_mesh.bounding_center)),
        add(previous_collider.position,
            rotate(previous_collider.orientation,
                   collider_mesh.bounding_center)));
    const Vec3 current_relative = subtract(
        add(body.position, rotate(body.orientation, body_mesh.bounding_center)),
        add(collider.position,
            rotate(collider.orientation, collider_mesh.bounding_center)));
    const Vec3 movement = subtract(current_relative, previous_relative);
    const float squared_movement = length_squared(movement);
    const float time = squared_movement > k_epsilon * k_epsilon
        ? clamp_scalar(-dot(previous_relative, movement) / squared_movement,
                       0.0F, 1.0F)
        : 0.0F;
    const Vec3 nearest = add(previous_relative, multiply(movement, time));
    const float radius = body_mesh.bounding_radius +
        collider_mesh.bounding_radius + margin + 1.0e-5F +
        rotational_motion_bound(previous_body, body, body_mesh) +
        rotational_motion_bound(previous_collider, collider, collider_mesh);
    return length_squared(nearest) <= radius * radius;
}

__host__ __device__ void closest_segments(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2,
                                          Vec3 &c1, Vec3 &c2) noexcept {
    const Vec3 d1 = subtract(q1, p1);
    const Vec3 d2 = subtract(q2, p2);
    const Vec3 r = subtract(p1, p2);
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    float s = 0.0F;
    float t = 0.0F;

    const float squared_epsilon = k_epsilon * k_epsilon;
    if (a <= squared_epsilon && e <= squared_epsilon) {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= squared_epsilon) {
        t = clamp_scalar(f / e, 0.0F, 1.0F);
    } else {
        const float c = dot(d1, r);
        if (e <= squared_epsilon) {
            s = clamp_scalar(-c / a, 0.0F, 1.0F);
        } else {
            const float b = dot(d1, d2);
            const float denominator = a * e - b * b;
            // The denominator has units length^4. An absolute tolerance
            // classified ordinary centimetre-scale edges as parallel.
            if (fabsf(denominator) > k_epsilon * a * e) {
                s = clamp_scalar((b * f - c * e) / denominator, 0.0F, 1.0F);
            }
            t = (b * s + f) / e;
            if (t < 0.0F) {
                t = 0.0F;
                s = clamp_scalar(-c / a, 0.0F, 1.0F);
            } else if (t > 1.0F) {
                t = 1.0F;
                s = clamp_scalar((b - c) / a, 0.0F, 1.0F);
            }
        }
    }
    c1 = add(p1, multiply(d1, s));
    c2 = add(p2, multiply(d2, t));
}

__host__ __device__ Vec3 closest_on_triangle(Vec3 point, Vec3 a, Vec3 b,
                                             Vec3 c) noexcept {
    const Vec3 ab = subtract(b, a);
    const Vec3 ac = subtract(c, a);
    const Vec3 ap = subtract(point, a);
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) {
        return a;
    }
    const Vec3 bp = subtract(point, b);
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) {
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        return add(a, multiply(ab, d1 / (d1 - d3)));
    }
    const Vec3 cp = subtract(point, c);
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) {
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        return add(a, multiply(ac, d2 / (d2 - d6)));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F) {
        return add(b, multiply(subtract(c, b),
                               (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    const float inverse = 1.0F / (va + vb + vc);
    return add(a, add(multiply(ab, vb * inverse),
                      multiply(ac, vc * inverse)));
}

__host__ __device__ bool point_in_triangle(Vec3 point, Vec3 a, Vec3 b,
                                           Vec3 c, Vec3 normal) noexcept {
    constexpr float tolerance = -1.0e-5F;
    return dot(cross(subtract(b, a), subtract(point, a)), normal) >= tolerance &&
           dot(cross(subtract(c, b), subtract(point, b)), normal) >= tolerance &&
           dot(cross(subtract(a, c), subtract(point, c)), normal) >= tolerance;
}

__host__ __device__ void consider_closest_pair(
    Vec3 on_segment, Vec3 on_triangle, float &best_squared,
    Vec3 &segment_point, Vec3 &triangle_point) noexcept {
    const float squared = length_squared(subtract(on_segment, on_triangle));
    if (squared < best_squared) {
        best_squared = squared;
        segment_point = on_segment;
        triangle_point = on_triangle;
    }
}

__host__ __device__ Vec3 transform_point(const RigidBodyState &state,
                                         Vec3 point) noexcept {
    return add(state.position, rotate(state.orientation, point));
}

struct BoundsTransform {
    Vec3 position{};
    Vec3 axis_x{};
    Vec3 axis_y{};
    Vec3 axis_z{};
};

__device__ BoundsTransform bounds_transform(
    const RigidBodyState &state) noexcept {
    return {state.position,
            rotate(state.orientation, {1.0F, 0.0F, 0.0F}),
            rotate(state.orientation, {0.0F, 1.0F, 0.0F}),
            rotate(state.orientation, {0.0F, 0.0F, 1.0F})};
}

__host__ __device__ Vec3 component_min(Vec3 first, Vec3 second) noexcept {
    return {fminf(first.x, second.x), fminf(first.y, second.y),
            fminf(first.z, second.z)};
}

__host__ __device__ Vec3 component_max(Vec3 first, Vec3 second) noexcept {
    return {fmaxf(first.x, second.x), fmaxf(first.y, second.y),
            fmaxf(first.z, second.z)};
}

__device__ void transformed_bounds(Vec3 local_minimum, Vec3 local_maximum,
                                   const BoundsTransform &transform, float margin,
                                   Vec3 &minimum, Vec3 &maximum) noexcept {
    const Vec3 local_center = multiply(add(local_minimum, local_maximum), 0.5F);
    const Vec3 local_half = multiply(subtract(local_maximum, local_minimum), 0.5F);
    const Vec3 world_center =
        add(transform.position,
            add(multiply(transform.axis_x, local_center.x),
                add(multiply(transform.axis_y, local_center.y),
                    multiply(transform.axis_z, local_center.z))));
    const Vec3 world_half{
        fabsf(transform.axis_x.x) * local_half.x +
            fabsf(transform.axis_y.x) * local_half.y +
            fabsf(transform.axis_z.x) * local_half.z,
        fabsf(transform.axis_x.y) * local_half.x +
            fabsf(transform.axis_y.y) * local_half.y +
            fabsf(transform.axis_z.y) * local_half.z,
        fabsf(transform.axis_x.z) * local_half.x +
            fabsf(transform.axis_y.z) * local_half.y +
            fabsf(transform.axis_z.z) * local_half.z};
    const Vec3 expansion{margin, margin, margin};
    minimum = subtract(subtract(world_center, world_half), expansion);
    maximum = add(add(world_center, world_half), expansion);
}

__device__ void transformed_motion_bounds(
    Vec3 local_minimum, Vec3 local_maximum,
    const BoundsTransform &previous_transform,
    const BoundsTransform &current_transform, bool swept, float margin,
    Vec3 &minimum, Vec3 &maximum) noexcept {
    transformed_bounds(local_minimum, local_maximum, current_transform, margin,
                       minimum, maximum);
    if (!swept) {
        return;
    }
    Vec3 previous_minimum{};
    Vec3 previous_maximum{};
    transformed_bounds(local_minimum, local_maximum, previous_transform, margin,
                       previous_minimum, previous_maximum);
    minimum = component_min(minimum, previous_minimum);
    maximum = component_max(maximum, previous_maximum);
}

__host__ __device__ bool bounds_overlap(Vec3 minimum_a, Vec3 maximum_a,
                                        Vec3 minimum_b,
                                        Vec3 maximum_b) noexcept {
    return minimum_a.x <= maximum_b.x && maximum_a.x >= minimum_b.x &&
           minimum_a.y <= maximum_b.y && maximum_a.y >= minimum_b.y &&
           minimum_a.z <= maximum_b.z && maximum_a.z >= minimum_b.z;
}

__host__ __device__ bool triangle_bounds_overlap(
    Vec3 a0, Vec3 a1, Vec3 a2, Vec3 b0, Vec3 b1, Vec3 b2,
    float margin) noexcept {
    const Vec3 expansion{margin, margin, margin};
    const Vec3 minimum_a =
        subtract(component_min(a0, component_min(a1, a2)), expansion);
    const Vec3 maximum_a =
        add(component_max(a0, component_max(a1, a2)), expansion);
    const Vec3 minimum_b = component_min(b0, component_min(b1, b2));
    const Vec3 maximum_b = component_max(b0, component_max(b1, b2));
    return bounds_overlap(minimum_a, maximum_a, minimum_b, maximum_b);
}

__host__ __device__ bool segment_hits_triangle(
    Vec3 first, Vec3 second, Vec3 a, Vec3 b, Vec3 c,
    Vec3 &intersection) noexcept {
    const Vec3 normal = cross(subtract(b, a), subtract(c, a));
    const Vec3 direction = subtract(second, first);
    const float denominator = dot(normal, direction);
    if (length_squared(normal) <= k_epsilon * k_epsilon ||
        fabsf(denominator) <= k_epsilon) {
        return false;
    }
    const float amount = dot(normal, subtract(a, first)) / denominator;
    if (amount < 0.0F || amount > 1.0F) {
        return false;
    }
    intersection = add(first, multiply(direction, amount));
    return point_in_triangle(intersection, a, b, c, normal);
}

__host__ __device__ void closest_triangle_pair(
    Vec3 a0, Vec3 a1, Vec3 a2, Vec3 b0, Vec3 b1, Vec3 b2,
    Vec3 &point_a, Vec3 &point_b) noexcept {
    const Vec3 a[3]{a0, a1, a2};
    const Vec3 b[3]{b0, b1, b2};
    Vec3 intersection{};
    for (std::uint32_t edge = 0U; edge < 3U; ++edge) {
        if (segment_hits_triangle(
                a[edge], a[(edge + 1U) % 3U], b0, b1, b2,
                intersection)) {
            point_a = point_b = intersection;
            return;
        }
    }
    for (std::uint32_t edge = 0U; edge < 3U; ++edge) {
        if (segment_hits_triangle(
                b[edge], b[(edge + 1U) % 3U], a0, a1, a2,
                intersection)) {
            point_a = point_b = intersection;
            return;
        }
    }

    float best_squared = FLT_MAX;
    for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
        const Vec3 on_triangle =
            closest_on_triangle(a[vertex], b0, b1, b2);
        consider_closest_pair(a[vertex], on_triangle, best_squared,
                              point_a, point_b);
    }
    for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
        const Vec3 on_triangle =
            closest_on_triangle(b[vertex], a0, a1, a2);
        consider_closest_pair(on_triangle, b[vertex], best_squared,
                              point_a, point_b);
    }
    for (std::uint32_t edge_a = 0U; edge_a < 3U; ++edge_a) {
        for (std::uint32_t edge_b = 0U; edge_b < 3U; ++edge_b) {
            Vec3 on_a{};
            Vec3 on_b{};
            closest_segments(a[edge_a], a[(edge_a + 1U) % 3U],
                             b[edge_b], b[(edge_b + 1U) % 3U],
                             on_a, on_b);
            consider_closest_pair(on_a, on_b, best_squared,
                                  point_a, point_b);
        }
    }
}

__device__ void add_manifold_contact(ContactManifold &manifold,
                                     Contact candidate,
                                     float separation) noexcept {
    const float minimum_spacing_squared = separation * separation;
    for (std::uint32_t index = 0; index < manifold.count; ++index) {
        if (length_squared(subtract(candidate.point,
                                    manifold.contacts[index].point)) <
            minimum_spacing_squared) {
            if (candidate.penetration > manifold.contacts[index].penetration) {
                manifold.contacts[index] = candidate;
            }
            return;
        }
    }
    if (manifold.count < 8U) {
        manifold.contacts[manifold.count++] = candidate;
        return;
    }
    std::uint32_t shallowest = 0U;
    for (std::uint32_t index = 1; index < manifold.count; ++index) {
        if (manifold.contacts[index].penetration <
            manifold.contacts[shallowest].penetration) {
            shallowest = index;
        }
    }
    if (candidate.penetration > manifold.contacts[shallowest].penetration) {
        manifold.contacts[shallowest] = candidate;
    }
}

__device__ void collide_triangle_ranges(
    const RigidBodyState &body_state, const TriangleMeshResource &body_mesh,
    std::uint32_t body_first, std::uint32_t body_count,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh, std::uint32_t collider_first,
    std::uint32_t collider_count, float margin,
    ContactManifold &manifold) noexcept {
    for (std::uint32_t body_triangle = body_first;
         body_triangle < body_first + body_count; ++body_triangle) {
        const std::uint32_t body_index = body_triangle * 3U;
        const Vec3 a0 = transform_point(
            body_state, body_mesh.vertices[body_mesh.indices[body_index]]);
        const Vec3 a1 = transform_point(
            body_state, body_mesh.vertices[body_mesh.indices[body_index + 1U]]);
        const Vec3 a2 = transform_point(
            body_state, body_mesh.vertices[body_mesh.indices[body_index + 2U]]);
        for (std::uint32_t collider_triangle = collider_first;
             collider_triangle < collider_first + collider_count;
             ++collider_triangle) {
            const std::uint32_t collider_index = collider_triangle * 3U;
            const Vec3 b0 = transform_point(
                collider_state,
                collider_mesh.vertices[collider_mesh.indices[collider_index]]);
            const Vec3 b1 = transform_point(
                collider_state,
                collider_mesh.vertices[collider_mesh.indices[collider_index + 1U]]);
            const Vec3 b2 = transform_point(
                collider_state,
                collider_mesh.vertices[collider_mesh.indices[collider_index + 2U]]);
            if (!triangle_bounds_overlap(a0, a1, a2, b0, b1, b2, margin)) {
                continue;
            }
            Vec3 point_a{};
            Vec3 point_b{};
            closest_triangle_pair(a0, a1, a2, b0, b1, b2, point_a, point_b);
            const Vec3 delta = subtract(point_a, point_b);
            const float squared = length_squared(delta);
            if (squared > margin * margin) {
                continue;
            }
            const Vec3 collider_normal = normalized_or(
                cross(subtract(b1, b0), subtract(b2, b0)),
                {0.0F, 1.0F, 0.0F});
            const Vec3 center_delta = subtract(body_state.position, point_b);
            const Vec3 fallback =
                dot(collider_normal, center_delta) >= 0.0F
                    ? collider_normal
                    : multiply(collider_normal, -1.0F);
            const float distance = sqrtf(fmaxf(squared, 0.0F));
            const Contact contact{
                normalized_or(delta, fallback),
                multiply(add(point_a, point_b), 0.5F),
                margin - distance + 1.0e-5F,
                true};
            add_manifold_contact(manifold, contact, fmaxf(margin * 2.0F, 1.0e-4F));
        }
    }
}

__device__ void collide_triangle_ranges_swept(
    const RigidBodyState &previous_body_state,
    const RigidBodyState &body_state,
    const TriangleMeshResource &body_mesh, std::uint32_t body_first,
    std::uint32_t body_count,
    const RigidBodyState &previous_collider_state,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh,
    std::uint32_t collider_first, std::uint32_t collider_count, float margin,
    bool body_moves, bool collider_moves,
    ContactManifold &manifold) noexcept {
    if (!body_moves && !collider_moves) {
        return;
    }
    for (std::uint32_t body_triangle = body_first;
         body_triangle < body_first + body_count; ++body_triangle) {
        const std::uint32_t body_index = body_triangle * 3U;
        Vec3 previous_a[3]{};
        Vec3 current_a[3]{};
        for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
            const Vec3 local =
                body_mesh.vertices[body_mesh.indices[body_index + vertex]];
            previous_a[vertex] = transform_point(previous_body_state, local);
            current_a[vertex] = transform_point(body_state, local);
        }
        Vec3 swept_a_minimum = component_min(previous_a[0], current_a[0]);
        Vec3 swept_a_maximum = component_max(previous_a[0], current_a[0]);
        for (std::uint32_t vertex = 1U; vertex < 3U; ++vertex) {
            swept_a_minimum = component_min(
                swept_a_minimum,
                component_min(previous_a[vertex], current_a[vertex]));
            swept_a_maximum = component_max(
                swept_a_maximum,
                component_max(previous_a[vertex], current_a[vertex]));
        }
        for (std::uint32_t collider_triangle = collider_first;
             collider_triangle < collider_first + collider_count;
             ++collider_triangle) {
            const std::uint32_t collider_index = collider_triangle * 3U;
            Vec3 previous_b[3]{};
            Vec3 current_b[3]{};
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                const Vec3 local = collider_mesh.vertices[
                    collider_mesh.indices[collider_index + vertex]];
                previous_b[vertex] =
                    transform_point(previous_collider_state, local);
                current_b[vertex] = transform_point(collider_state, local);
            }
            Vec3 swept_b_minimum = component_min(previous_b[0], current_b[0]);
            Vec3 swept_b_maximum = component_max(previous_b[0], current_b[0]);
            for (std::uint32_t vertex = 1U; vertex < 3U; ++vertex) {
                swept_b_minimum = component_min(
                    swept_b_minimum,
                    component_min(previous_b[vertex], current_b[vertex]));
                swept_b_maximum = component_max(
                    swept_b_maximum,
                    component_max(previous_b[vertex], current_b[vertex]));
            }
            if (!bounds_overlap(
                    {swept_a_minimum.x - margin,
                     swept_a_minimum.y - margin,
                     swept_a_minimum.z - margin},
                    {swept_a_maximum.x + margin,
                     swept_a_maximum.y + margin,
                     swept_a_maximum.z + margin},
                    swept_b_minimum, swept_b_maximum)) {
                continue;
            }
            Vec3 delta_a[3]{};
            Vec3 delta_b[3]{};
            float speed_bound = 0.0F;
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                delta_a[vertex] = subtract(current_a[vertex], previous_a[vertex]);
                delta_b[vertex] = subtract(current_b[vertex], previous_b[vertex]);
                speed_bound = fmaxf(speed_bound,
                                    vector_length(delta_a[vertex]));
            }
            float collider_speed = 0.0F;
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                collider_speed = fmaxf(collider_speed,
                                        vector_length(delta_b[vertex]));
            }
            speed_bound += collider_speed;
            // Distance between moving triangles depends on their relative
            // motion. Subtracting any common translation preserves a safe
            // Lipschitz bound for all barycentric point pairs.
            const Vec3 common_motion = delta_b[0];
            float relative_body_speed = 0.0F;
            float relative_collider_speed = 0.0F;
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                relative_body_speed = fmaxf(
                    relative_body_speed,
                    vector_length(subtract(delta_a[vertex], common_motion)));
                relative_collider_speed = fmaxf(
                    relative_collider_speed,
                    vector_length(subtract(delta_b[vertex], common_motion)));
            }
            speed_bound = fminf(speed_bound,
                                relative_body_speed + relative_collider_speed);
            if (speed_bound <= k_epsilon) {
                continue;
            }

            float time = 0.0F;
            for (std::uint32_t iteration = 0U; iteration < 32U;
                 ++iteration) {
                Vec3 a[3]{};
                Vec3 b[3]{};
                for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                    a[vertex] = add(previous_a[vertex],
                                    multiply(delta_a[vertex], time));
                    b[vertex] = add(previous_b[vertex],
                                    multiply(delta_b[vertex], time));
                }
                Vec3 point_a{};
                Vec3 point_b{};
                closest_triangle_pair(a[0], a[1], a[2], b[0], b[1], b[2],
                                      point_a, point_b);
                const Vec3 delta = subtract(point_a, point_b);
                const float distance =
                    sqrtf(fmaxf(0.0F, length_squared(delta)));
                if (distance <= margin + 1.0e-5F) {
                    if (iteration == 0U) {
                        break;
                    }
                    const Vec3 collider_normal = normalized_or(
                        cross(subtract(b[1], b[0]), subtract(b[2], b[0])),
                        {0.0F, 1.0F, 0.0F});
                    const Vec3 body_center = add(
                        previous_body_state.position,
                        multiply(subtract(body_state.position,
                                          previous_body_state.position),
                                 time));
                    const Vec3 collider_center = add(
                        previous_collider_state.position,
                        multiply(subtract(collider_state.position,
                                          previous_collider_state.position),
                                 time));
                    const Vec3 fallback =
                        dot(collider_normal,
                            subtract(body_center, collider_center)) >= 0.0F
                            ? collider_normal
                            : multiply(collider_normal, -1.0F);
                    const Vec3 normal = normalized_or(delta, fallback);
                    const Vec3 relative_movement = subtract(
                        subtract(body_state.position,
                                 previous_body_state.position),
                        subtract(collider_state.position,
                                 previous_collider_state.position));
                    const float remaining = fmaxf(
                        0.0F,
                        -dot(multiply(relative_movement, 1.0F - time),
                             normal));
                    add_manifold_contact(
                        manifold,
                        {normal, multiply(add(point_a, point_b), 0.5F),
                         remaining + margin + 1.0e-5F, true},
                        fmaxf(margin * 2.0F, 1.0e-4F));
                    break;
                }
                float advancement =
                    (distance - margin) / (speed_bound + k_epsilon) * 0.9F;
                advancement = fmaxf(advancement, 1.0e-5F);
                time += advancement;
                if (time > 1.0F) {
                    break;
                }
            }
        }
    }
}

__device__ ContactManifold collide_meshes(
    const BodyParameters &body, const RigidBodyState &previous_body_state,
    const RigidBodyState &body_state,
    const TriangleMeshResource &body_mesh, const BodyParameters &collider,
    const RigidBodyState &previous_collider_state,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh) noexcept {
    ContactManifold manifold{};
    const float margin = body.collision_margin + collider.collision_margin;
    const bool swept = requires_swept_pair_contact(
        previous_body_state, body_state, body_mesh,
        previous_collider_state, collider_state, collider_mesh, margin);
    const BoundsTransform previous_body_transform =
        swept ? bounds_transform(previous_body_state) : BoundsTransform{};
    const BoundsTransform previous_collider_transform =
        swept ? bounds_transform(previous_collider_state) : BoundsTransform{};
    Vec3 body_minimum{};
    Vec3 body_maximum{};
    Vec3 collider_minimum{};
    Vec3 collider_maximum{};
    const BoundsTransform body_transform = bounds_transform(body_state);
    const BoundsTransform collider_transform = bounds_transform(collider_state);
    transformed_motion_bounds(body_mesh.minimum, body_mesh.maximum,
                              previous_body_transform, body_transform, swept,
                              margin, body_minimum, body_maximum);
    transformed_motion_bounds(collider_mesh.minimum, collider_mesh.maximum,
                              previous_collider_transform, collider_transform,
                              swept, 0.0F, collider_minimum, collider_maximum);
    if (!bounds_overlap(body_minimum, body_maximum, collider_minimum,
                        collider_maximum)) {
        return manifold;
    }

    struct NodePair {
        std::uint32_t body{};
        std::uint32_t collider{};
    };
    NodePair stack[256]{{0U, 0U}};
    std::uint32_t stack_size = 1U;
    bool overflow = body_mesh.bvh_node_count == 0U ||
                    collider_mesh.bvh_node_count == 0U;
    while (stack_size > 0U && !overflow) {
        const NodePair pair = stack[--stack_size];
        const BvhNode &body_node = body_mesh.bvh_nodes[pair.body];
        const BvhNode &collider_node = collider_mesh.bvh_nodes[pair.collider];
        transformed_motion_bounds(
            body_node.minimum, body_node.maximum, previous_body_transform,
            body_transform, swept, margin, body_minimum, body_maximum);
        transformed_motion_bounds(
            collider_node.minimum, collider_node.maximum,
            previous_collider_transform, collider_transform, swept, 0.0F,
            collider_minimum, collider_maximum);
        if (!bounds_overlap(body_minimum, body_maximum, collider_minimum,
                            collider_maximum)) {
            continue;
        }
        const bool body_leaf = body_node.triangle_count != 0U;
        const bool collider_leaf = collider_node.triangle_count != 0U;
        if (body_leaf && collider_leaf) {
            collide_triangle_ranges(
                body_state, body_mesh, body_node.first_triangle,
                body_node.triangle_count, collider_state, collider_mesh,
                collider_node.first_triangle, collider_node.triangle_count,
                margin, manifold);
            if (swept) {
                collide_triangle_ranges_swept(
                    previous_body_state, body_state, body_mesh,
                    body_node.first_triangle, body_node.triangle_count,
                    previous_collider_state, collider_state, collider_mesh,
                    collider_node.first_triangle, collider_node.triangle_count,
                    margin, true, true, manifold);
            }
            continue;
        }

        const std::uint32_t required = body_leaf || collider_leaf ? 2U : 4U;
        if (stack_size + required > 256U) {
            overflow = true;
            break;
        }
        if (body_leaf) {
            stack[stack_size++] = {pair.body, collider_node.right};
            stack[stack_size++] = {pair.body, collider_node.left};
        } else if (collider_leaf) {
            stack[stack_size++] = {body_node.right, pair.collider};
            stack[stack_size++] = {body_node.left, pair.collider};
        } else {
            stack[stack_size++] = {body_node.right, collider_node.right};
            stack[stack_size++] = {body_node.right, collider_node.left};
            stack[stack_size++] = {body_node.left, collider_node.right};
            stack[stack_size++] = {body_node.left, collider_node.left};
        }
    }
    if (overflow) {
        manifold = {};
        collide_triangle_ranges(
            body_state, body_mesh, 0U, body_mesh.index_count / 3U,
            collider_state, collider_mesh, 0U,
            collider_mesh.index_count / 3U, margin, manifold);
        if (swept) {
            collide_triangle_ranges_swept(
                previous_body_state, body_state, body_mesh, 0U,
                body_mesh.index_count / 3U, previous_collider_state,
                collider_state, collider_mesh, 0U,
                collider_mesh.index_count / 3U, margin, true, true, manifold);
        }
    }
    return manifold;
}

__host__ __device__ Vec3 inverse_inertia_world(
    const BodyParameters &parameters, const RigidBodyState &state,
    Vec3 world_vector) noexcept {
    const Vec3 local = inverse_rotate(state.orientation, world_vector);
    const Vec3 transformed{local.x * parameters.inverse_inertia_local.x,
                           local.y * parameters.inverse_inertia_local.y,
                           local.z * parameters.inverse_inertia_local.z};
    return rotate(state.orientation, transformed);
}

__device__ AppliedContactImpulse apply_contact_impulse(
    const BodyParameters &body, RigidBodyState &state,
    const BodyParameters &collider, RigidBodyState &collider_state,
    const Contact &contact) noexcept {
    AppliedContactImpulse applied{};
    const Vec3 body_arm = subtract(contact.point, state.position);
    const Vec3 collider_arm = subtract(contact.point, collider_state.position);
    const Vec3 body_velocity =
        add(state.linear_velocity, cross(state.angular_velocity, body_arm));
    const Vec3 collider_velocity = add(
        collider_state.linear_velocity,
        cross(collider_state.angular_velocity, collider_arm));
    Vec3 relative_velocity = subtract(body_velocity, collider_velocity);
    const float normal_speed = dot(relative_velocity, contact.normal);
    if (normal_speed >= 0.0F) {
        return applied;
    }

    const Vec3 body_cross = cross(body_arm, contact.normal);
    const Vec3 angular_term =
        cross(inverse_inertia_world(body, state, body_cross), body_arm);
    const Vec3 collider_cross = cross(collider_arm, contact.normal);
    const Vec3 collider_angular_term = cross(
        inverse_inertia_world(collider, collider_state, collider_cross),
        collider_arm);
    const float denominator =
        body.inverse_mass + collider.inverse_mass +
        dot(add(angular_term, collider_angular_term), contact.normal);
    if (denominator <= k_epsilon) {
        return applied;
    }

    const float restitution = fminf(body.restitution, collider.restitution);
    const float normal_impulse = -(1.0F + restitution) * normal_speed / denominator;
    applied.normal = normal_impulse;
    const Vec3 normal_vector = multiply(contact.normal, normal_impulse);
    state.linear_velocity =
        add(state.linear_velocity, multiply(normal_vector, body.inverse_mass));
    state.angular_velocity =
        add(state.angular_velocity,
            inverse_inertia_world(body, state, cross(body_arm, normal_vector)));
    if (collider.inverse_mass > 0.0F) {
        collider_state.linear_velocity = subtract(
            collider_state.linear_velocity,
            multiply(normal_vector, collider.inverse_mass));
        collider_state.angular_velocity = subtract(
            collider_state.angular_velocity,
            inverse_inertia_world(collider, collider_state,
                                  cross(collider_arm, normal_vector)));
    }

    relative_velocity = subtract(
        add(state.linear_velocity, cross(state.angular_velocity, body_arm)),
        add(collider_state.linear_velocity,
            cross(collider_state.angular_velocity, collider_arm)));
    Vec3 tangent = subtract(relative_velocity,
                            multiply(contact.normal,
                                     dot(relative_velocity, contact.normal)));
    const float tangent_length = vector_length(tangent);
    if (tangent_length <= k_epsilon) {
        return applied;
    }
    tangent = multiply(tangent, 1.0F / tangent_length);
    const Vec3 tangent_cross = cross(body_arm, tangent);
    const Vec3 collider_tangent_cross = cross(collider_arm, tangent);
    const float tangent_denominator = body.inverse_mass + collider.inverse_mass +
        dot(add(cross(inverse_inertia_world(body, state, tangent_cross), body_arm),
                cross(inverse_inertia_world(collider, collider_state,
                                            collider_tangent_cross),
                      collider_arm)),
            tangent);
    if (tangent_denominator <= k_epsilon) {
        return applied;
    }
    float tangent_impulse = -dot(relative_velocity, tangent) / tangent_denominator;
    const float friction_limit =
        sqrtf(body.friction * collider.friction) * normal_impulse;
    tangent_impulse =
        clamp_scalar(tangent_impulse, -friction_limit, friction_limit);
    const Vec3 tangent_vector = multiply(tangent, tangent_impulse);
    applied.friction = tangent_vector;
    state.linear_velocity =
        add(state.linear_velocity, multiply(tangent_vector, body.inverse_mass));
    state.angular_velocity =
        add(state.angular_velocity,
            inverse_inertia_world(body, state, cross(body_arm, tangent_vector)));
    if (collider.inverse_mass > 0.0F) {
        collider_state.linear_velocity = subtract(
            collider_state.linear_velocity,
            multiply(tangent_vector, collider.inverse_mass));
        collider_state.angular_velocity = subtract(
            collider_state.angular_velocity,
            inverse_inertia_world(collider, collider_state,
                                  cross(collider_arm, tangent_vector)));
    }
    return applied;
}

__device__ void resolve_contacts(
    const BodyParameters &body, RigidBodyState &state,
    const BodyParameters &collider, RigidBodyState &collider_state,
    const Contact *contacts, std::uint32_t contact_count,
    bool correct_position, RigidContactEvent *debug_events,
    std::uint32_t debug_event_count) noexcept {
    if (contact_count == 0U) {
        return;
    }
    const float inverse_mass_sum = body.inverse_mass + collider.inverse_mass;
    if (correct_position && inverse_mass_sum > k_epsilon) {
        const float contact_weight = 1.0F / static_cast<float>(contact_count);
        for (std::uint32_t index = 0; index < contact_count; ++index) {
            const Vec3 correction = multiply(
                contacts[index].normal,
                (contacts[index].penetration + 1.0e-5F) * contact_weight /
                    inverse_mass_sum);
            state.position = add(state.position,
                                 multiply(correction, body.inverse_mass));
            if (collider.inverse_mass > 0.0F) {
                collider_state.position = subtract(
                    collider_state.position,
                    multiply(correction, collider.inverse_mass));
            }
        }
    }
    for (std::uint32_t index = 0; index < contact_count; ++index) {
        const AppliedContactImpulse applied = apply_contact_impulse(
            body, state, collider, collider_state, contacts[index]);
        if (debug_events != nullptr && index < debug_event_count) {
            debug_events[index].normal_impulse += applied.normal;
            debug_events[index].friction_impulse =
                add(debug_events[index].friction_impulse, applied.friction);
        }
    }
}

__device__ Vec3 quaternion_delta_velocity(Quaternion from, Quaternion to,
                                          float timestep) noexcept {
    Quaternion delta = quaternion_multiply(to, conjugate(from));
    if (delta.w < 0.0F) {
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    }
    delta = normalized_quaternion(delta);
    const float vector_size = sqrtf(delta.x * delta.x + delta.y * delta.y +
                                    delta.z * delta.z);
    if (vector_size <= k_epsilon || timestep <= 0.0F) {
        return {};
    }
    const float angle = 2.0F * atan2f(vector_size, clamp_scalar(delta.w, -1.0F, 1.0F));
    const float scale = angle / (vector_size * timestep);
    return {delta.x * scale, delta.y * scale, delta.z * scale};
}

__global__ void integrate_rigid_bodies_kernel(
    const BodyParameters *parameters, const BodyAccumulator *accumulators,
    const KinematicTarget *targets, const RigidBodyState *input,
    RigidBodyState *output, std::uint32_t count, Vec3 gravity, float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const BodyParameters body = parameters[index];
    const RigidBodyState previous = input[index];
    RigidBodyState next = previous;
    if (body.motion == MotionType::static_body) {
        next.linear_velocity = {};
        next.angular_velocity = {};
        output[index] = next;
        return;
    }

    if (body.motion == MotionType::kinematic) {
        if (targets[index].active) {
            const float fraction = 1.0F / static_cast<float>(remaining_substeps);
            next.position = add(previous.position,
                                multiply(subtract(targets[index].state.position,
                                                  previous.position),
                                         fraction));
            Quaternion target = targets[index].state.orientation;
            const float orientation_dot = previous.orientation.x * target.x +
                                          previous.orientation.y * target.y +
                                          previous.orientation.z * target.z +
                                          previous.orientation.w * target.w;
            if (orientation_dot < 0.0F) {
                target = {-target.x, -target.y, -target.z, -target.w};
            }
            next.orientation = normalized_quaternion(
                {previous.orientation.x + (target.x - previous.orientation.x) * fraction,
                 previous.orientation.y + (target.y - previous.orientation.y) * fraction,
                 previous.orientation.z + (target.z - previous.orientation.z) * fraction,
                 previous.orientation.w + (target.w - previous.orientation.w) * fraction});
            next.linear_velocity = multiply(
                subtract(next.position, previous.position), 1.0F / timestep);
            next.angular_velocity = quaternion_delta_velocity(
                previous.orientation, next.orientation, timestep);
        } else {
            next.linear_velocity = {};
            next.angular_velocity = {};
        }
        output[index] = next;
        return;
    }

    const BodyAccumulator accumulator = accumulators[index];
    next.linear_velocity =
        add(next.linear_velocity,
            multiply(add(gravity, multiply(accumulator.force, body.inverse_mass)),
                     timestep));
    const Vec3 angular_acceleration =
        inverse_inertia_world(body, next, accumulator.torque);
    next.angular_velocity =
        add(next.angular_velocity, multiply(angular_acceleration, timestep));
    if (apply_impulses) {
        next.linear_velocity =
            add(next.linear_velocity,
                multiply(accumulator.impulse, body.inverse_mass));
        next.angular_velocity =
            add(next.angular_velocity,
                inverse_inertia_world(body, next, accumulator.angular_impulse));
    }

    next.linear_velocity = multiply(
        next.linear_velocity, 1.0F / (1.0F + body.linear_damping * timestep));
    next.angular_velocity = multiply(
        next.angular_velocity, 1.0F / (1.0F + body.angular_damping * timestep));
    next.linear_velocity =
        clamp_length(next.linear_velocity, body.maximum_linear_speed);
    next.angular_velocity =
        clamp_length(next.angular_velocity, body.maximum_angular_speed);
    next.position = add(next.position, multiply(next.linear_velocity, timestep));

    const Quaternion angular{next.angular_velocity.x, next.angular_velocity.y,
                             next.angular_velocity.z, 0.0F};
    const Quaternion derivative = quaternion_multiply(angular, next.orientation);
    next.orientation = normalized_quaternion(
        {next.orientation.x + 0.5F * derivative.x * timestep,
         next.orientation.y + 0.5F * derivative.y * timestep,
         next.orientation.z + 0.5F * derivative.z * timestep,
         next.orientation.w + 0.5F * derivative.w * timestep});
    output[index] = next;
}

__global__ void compute_rigid_world_bounds_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states,
    std::uint32_t count, const TriangleMeshResource *meshes,
    WorldAabb *world_bounds) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const TriangleMeshResource &mesh = meshes[parameters[index].mesh.index];
    const bool swept = requires_swept_contact(
        previous_states[index], states[index], mesh,
        parameters[index].collision_margin);
    const BoundsTransform current_transform = bounds_transform(states[index]);
    BoundsTransform previous_transform{};
    if (swept) {
        previous_transform = bounds_transform(previous_states[index]);
    }
    transformed_motion_bounds(
        mesh.minimum, mesh.maximum, previous_transform, current_transform,
        swept, 0.0F,
        world_bounds[index].minimum, world_bounds[index].maximum);
}

__global__ void broad_phase_rigid_pairs_kernel(
    const BodyParameters *parameters, const WorldAabb *world_bounds,
    const RigidBodyState *previous_states, const RigidBodyState *states,
    const TriangleMeshResource *meshes,
    std::uint32_t count, std::uint8_t *active_flags) {
    const std::uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= count * count) {
        return;
    }
    const std::uint32_t index = pair / count;
    const std::uint32_t collider_index = pair % count;
    bool active = parameters[index].motion == MotionType::dynamic &&
                  index != collider_index;
    if (active && parameters[collider_index].motion == MotionType::dynamic &&
        collider_index < index) {
        active = false;
    }
    if (active) {
        const float margin = parameters[index].collision_margin +
                             parameters[collider_index].collision_margin;
        const WorldAabb body = world_bounds[index];
        const WorldAabb collider = world_bounds[collider_index];
        active = bounds_overlap(
            {body.minimum.x - margin, body.minimum.y - margin,
             body.minimum.z - margin},
            {body.maximum.x + margin, body.maximum.y + margin,
             body.maximum.z + margin},
            collider.minimum, collider.maximum);
        if (active) {
            active = bounding_spheres_may_contact(
                previous_states[index], states[index],
                meshes[parameters[index].mesh.index],
                previous_states[collider_index], states[collider_index],
                meshes[parameters[collider_index].mesh.index], margin);
        }
    }
    active_flags[pair] = active ? 1U : 0U;
}

__global__ void generate_rigid_leaf_pairs_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states, std::uint32_t count,
    const TriangleMeshResource *meshes,
    std::uint32_t mesh_capacity, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, LeafPair *leaf_pairs,
    std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity) {
    for (std::uint32_t active_index = blockIdx.x;
         active_index < *active_pair_count; active_index += gridDim.x) {
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        if (threadIdx.x == 0U) {
            leaf_pair_counts[pair] = 0U;
        }
        __syncthreads();
        if (active_index >= leaf_pair_cache_slot_capacity) {
            if (threadIdx.x == 0U) {
                leaf_pair_counts[pair] = k_leaf_pair_overflow;
            }
            __syncthreads();
            continue;
        }
        const TriangleMeshId body_mesh_id = parameters[index].mesh;
        const TriangleMeshId collider_mesh_id =
            parameters[collider_index].mesh;
        if (body_mesh_id.index >= mesh_capacity ||
            collider_mesh_id.index >= mesh_capacity) {
            continue;
        }
        const TriangleMeshResource &body_mesh = meshes[body_mesh_id.index];
        const TriangleMeshResource &collider_mesh =
            meshes[collider_mesh_id.index];
        if (!body_mesh.alive ||
            body_mesh.generation != body_mesh_id.generation ||
            !collider_mesh.alive ||
            collider_mesh.generation != collider_mesh_id.generation) {
            continue;
        }

        const float margin = parameters[index].collision_margin +
                             parameters[collider_index].collision_margin;
        const BoundsTransform body_transform = bounds_transform(states[index]);
        const BoundsTransform collider_transform =
            bounds_transform(states[collider_index]);
        const bool swept = requires_swept_pair_contact(
            previous_states[index], states[index], body_mesh,
            previous_states[collider_index], states[collider_index],
            collider_mesh, margin);
        BoundsTransform previous_body_transform{};
        BoundsTransform previous_collider_transform{};
        if (swept) {
            previous_body_transform = bounds_transform(previous_states[index]);
            previous_collider_transform =
                bounds_transform(previous_states[collider_index]);
        }
        Vec3 body_minimum{};
        Vec3 body_maximum{};
        Vec3 collider_minimum{};
        Vec3 collider_maximum{};
        const std::uint64_t leaf_pair_count =
            static_cast<std::uint64_t>(body_mesh.bvh_leaf_count) *
            collider_mesh.bvh_leaf_count;
        std::uint32_t local_count = 0U;
        for (std::uint64_t leaf_pair = threadIdx.x;
             leaf_pair < leaf_pair_count; leaf_pair += blockDim.x) {
            const std::uint32_t body_leaf = static_cast<std::uint32_t>(
                leaf_pair / collider_mesh.bvh_leaf_count);
            const std::uint32_t collider_leaf = static_cast<std::uint32_t>(
                leaf_pair % collider_mesh.bvh_leaf_count);
            const BvhNode &body_node =
                body_mesh.bvh_nodes[body_mesh.bvh_leaves[body_leaf]];
            const BvhNode &collider_node = collider_mesh.bvh_nodes[
                collider_mesh.bvh_leaves[collider_leaf]];
            transformed_motion_bounds(
                body_node.minimum, body_node.maximum, previous_body_transform,
                body_transform, swept, margin, body_minimum,
                body_maximum);
            transformed_motion_bounds(
                collider_node.minimum, collider_node.maximum,
                previous_collider_transform, collider_transform,
                swept, 0.0F, collider_minimum, collider_maximum);
            if (bounds_overlap(body_minimum, body_maximum, collider_minimum,
                               collider_maximum)) {
                ++local_count;
            }
        }

        __shared__ std::uint32_t offsets[128];
        __shared__ std::uint32_t candidate_count;
        offsets[threadIdx.x] = local_count;
        __syncthreads();
        if (threadIdx.x == 0U) {
            std::uint32_t prefix = 0U;
            for (std::uint32_t thread = 0U; thread < blockDim.x; ++thread) {
                const std::uint32_t count_for_thread = offsets[thread];
                offsets[thread] = prefix;
                prefix += count_for_thread;
            }
            candidate_count = prefix;
            leaf_pair_counts[pair] =
                prefix > k_max_leaf_pairs_per_body_pair ? k_leaf_pair_overflow
                                                        : prefix;
        }
        __syncthreads();
        if (candidate_count > k_max_leaf_pairs_per_body_pair) {
            continue;
        }

        LeafPair *pair_candidates =
            leaf_pairs + static_cast<std::size_t>(active_index) *
                             k_max_leaf_pairs_per_body_pair;
        std::uint32_t output_index = offsets[threadIdx.x];
        for (std::uint64_t leaf_pair = threadIdx.x;
             leaf_pair < leaf_pair_count; leaf_pair += blockDim.x) {
            const std::uint32_t body_leaf = static_cast<std::uint32_t>(
                leaf_pair / collider_mesh.bvh_leaf_count);
            const std::uint32_t collider_leaf = static_cast<std::uint32_t>(
                leaf_pair % collider_mesh.bvh_leaf_count);
            const BvhNode &body_node =
                body_mesh.bvh_nodes[body_mesh.bvh_leaves[body_leaf]];
            const BvhNode &collider_node = collider_mesh.bvh_nodes[
                collider_mesh.bvh_leaves[collider_leaf]];
            transformed_motion_bounds(
                body_node.minimum, body_node.maximum, previous_body_transform,
                body_transform, swept, margin, body_minimum,
                body_maximum);
            transformed_motion_bounds(
                collider_node.minimum, collider_node.maximum,
                previous_collider_transform, collider_transform,
                swept, 0.0F, collider_minimum, collider_maximum);
            if (!bounds_overlap(body_minimum, body_maximum, collider_minimum,
                                collider_maximum)) {
                continue;
            }
            pair_candidates[output_index++] = {
                body_node.first_triangle, body_node.triangle_count,
                collider_node.first_triangle, collider_node.triangle_count};
        }
        __syncthreads();
    }
}

__global__ void evaluate_rigid_leaf_pairs_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    RigidBodyState *states, std::uint32_t count,
    const TriangleMeshResource *meshes, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const LeafPair *leaf_pairs, const std::uint32_t *leaf_pair_counts,
    ContactManifold *manifolds) {
    for (std::uint32_t active_index = blockIdx.x;
         active_index < *active_pair_count; active_index += gridDim.x) {
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        ContactManifold &output = manifolds[active_index];
        const std::uint32_t candidate_count = leaf_pair_counts[pair];
        if (candidate_count == 0U) {
            if (threadIdx.x == 0U) {
                output = {};
            }
            __syncthreads();
            continue;
        }
        const TriangleMeshResource &body_mesh =
            meshes[parameters[index].mesh.index];
        const TriangleMeshResource &collider_mesh =
            meshes[parameters[collider_index].mesh.index];
        if (candidate_count == k_leaf_pair_overflow) {
            if (threadIdx.x == 0U) {
                output = {};
            }
            __syncthreads();
            continue;
        }

        extern __shared__ ContactManifold partials[];
        __shared__ ContactManifold reduced;
        if (threadIdx.x == 0U) {
            reduced = {};
        }
        __syncthreads();
        const LeafPair *pair_candidates =
            leaf_pairs + static_cast<std::size_t>(active_index) *
                             k_max_leaf_pairs_per_body_pair;
        const float separation = fmaxf(
            (parameters[index].collision_margin +
             parameters[collider_index].collision_margin) *
                2.0F,
            1.0e-4F);
        const float collision_margin =
            parameters[index].collision_margin +
            parameters[collider_index].collision_margin;
        const bool swept = requires_swept_pair_contact(
            previous_states[index], states[index], body_mesh,
            previous_states[collider_index], states[collider_index],
            collider_mesh, collision_margin);
        for (std::uint32_t wave = 0U; wave < candidate_count;
             wave += blockDim.x) {
            ContactManifold local{};
            const std::uint32_t candidate_index = wave + threadIdx.x;
            if (candidate_index < candidate_count) {
                const LeafPair candidate = pair_candidates[candidate_index];
                collide_triangle_ranges(
                    states[index], body_mesh, candidate.body_first,
                    candidate.body_count, states[collider_index],
                    collider_mesh, candidate.collider_first,
                    candidate.collider_count, collision_margin, local);
                if (swept) {
                    collide_triangle_ranges_swept(
                        previous_states[index], states[index], body_mesh,
                        candidate.body_first, candidate.body_count,
                        previous_states[collider_index],
                        states[collider_index], collider_mesh,
                        candidate.collider_first, candidate.collider_count,
                        collision_margin, true, true, local);
                }
            }
            partials[threadIdx.x] = local;
            __syncthreads();
            if (threadIdx.x == 0U) {
                const std::uint32_t remaining = candidate_count - wave;
                const std::uint32_t wave_count =
                    blockDim.x < remaining ? blockDim.x : remaining;
                for (std::uint32_t item = 0U; item < wave_count; ++item) {
                    for (std::uint32_t contact = 0U;
                         contact < partials[item].count; ++contact) {
                        add_manifold_contact(reduced,
                                             partials[item].contacts[contact],
                                             separation);
                    }
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0U) {
            output = reduced;
        }
        __syncthreads();
    }
}

// Serial BVH traversal needs a large stack. Isolate it from normal pair work.
__global__ void evaluate_overflow_rigid_pairs_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states, std::uint32_t count,
    const TriangleMeshResource *meshes, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint32_t *leaf_pair_counts, ContactManifold *manifolds) {
    for (std::uint32_t active_index = blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        const std::uint32_t pair = active_pairs[active_index];
        if (leaf_pair_counts[pair] != k_leaf_pair_overflow) {
            continue;
        }
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        manifolds[active_index] = collide_meshes(
            parameters[index], previous_states[index], states[index],
            meshes[parameters[index].mesh.index], parameters[collider_index],
            previous_states[collider_index], states[collider_index],
            meshes[parameters[collider_index].mesh.index]);
    }
}

__global__ void prepare_parallel_contact_events_kernel(
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const RigidBodyId *ids,
    std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    if (reset_events) {
        *event_count = 0U;
    }
    if (!collect_events) {
        return;
    }
    std::uint32_t cursor = 0U;
    for (std::uint32_t active_index = 0U;
         active_index < *active_pair_count; ++active_index) {
        const ContactManifold &manifold = manifolds[active_index];
        event_offsets[active_index] = cursor;
        const std::uint32_t remaining = cursor < event_capacity
            ? event_capacity - cursor : 0U;
        const std::uint32_t retained = manifold.count < remaining
            ? manifold.count : remaining;
        if (retained > 0U) {
            const std::uint32_t pair = active_pairs[active_index];
            const std::uint32_t body_index = pair / count;
            const std::uint32_t collider_index = pair % count;
            for (std::uint32_t contact_index = 0U;
                 contact_index < retained; ++contact_index) {
                const Contact &contact = manifold.contacts[contact_index];
                events[cursor + contact_index] = {
                    ids[body_index], ids[collider_index], contact.point,
                    contact.normal, contact.penetration, 0.0F, {}};
            }
        }
        cursor += manifold.count;
    }
    // A later substep with no contacts must not erase an earlier event from
    // this frame; the original serial path retained it as well.
    if (cursor > 0U) {
        *event_count = cursor < event_capacity ? cursor : event_capacity;
    }
}

__global__ void initialize_parallel_colors_kernel(
    const std::uint32_t *active_pair_count, std::uint8_t *pair_colors,
    std::uint32_t *color_state) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        color_state[0] = 0U;
        color_state[1] = 0U;
    }
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        pair_colors[active_index] = k_contact_color_overflow;
    }
}

__global__ void reset_parallel_color_owners_kernel(
    std::uint32_t *owners, std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        owners[index] = 0xffffffffU;
    }
}

// Each body's lowest-priority uncolored contact wins this color round.
__host__ __device__ std::uint32_t contact_color_priority(
    std::uint32_t pair) noexcept {
    // Odd multiplication permutes uint32 values; nearby body IDs do not
    // monopolize all rounds, and priorities remain deterministic and unique.
    return pair * 2654435761U + 1013904223U;
}

__global__ void find_parallel_color_owners_kernel(
    const BodyParameters *parameters, std::uint32_t count,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const std::uint8_t *pair_colors,
    std::uint32_t *owners) {
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        if (pair_colors[active_index] != k_contact_color_overflow ||
            manifolds[active_index].count == 0U) {
            continue;
        }
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        const std::uint32_t priority = contact_color_priority(pair);
        atomicMin(&owners[index], priority);
        if (parameters[collider_index].motion == MotionType::dynamic) {
            atomicMin(&owners[collider_index], priority);
        }
    }
}

__global__ void assign_parallel_contact_colors_kernel(
    const BodyParameters *parameters, std::uint32_t count,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const std::uint32_t *owners,
    std::uint8_t *pair_colors, std::uint32_t *color_state,
    std::uint32_t color, std::uint32_t color_round_count) {
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        if (pair_colors[active_index] != k_contact_color_overflow ||
            manifolds[active_index].count == 0U) {
            continue;
        }
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        const bool dynamic_collider =
            parameters[collider_index].motion == MotionType::dynamic;
        const std::uint32_t priority = contact_color_priority(pair);
        if (owners[index] == priority &&
            (!dynamic_collider || owners[collider_index] == priority)) {
            pair_colors[active_index] = static_cast<std::uint8_t>(color);
            atomicMax(&color_state[0], color + 1U);
        } else if (color + 1U == color_round_count) {
            atomicAdd(&color_state[1], 1U);
        }
    }
}

__global__ void resolve_colored_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint8_t *pair_colors, const std::uint32_t *color_state,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity,
    std::uint32_t color, bool correct_position) {
    if (color >= color_state[0]) {
        return;
    }
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        if (pair_colors[active_index] != color) {
            continue;
        }
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        const ContactManifold &manifold = manifolds[active_index];
        RigidContactEvent *pair_events = nullptr;
        std::uint32_t retained = 0U;
        if (event_capacity > 0U) {
            const std::uint32_t offset = event_offsets[active_index];
            if (offset < event_capacity) {
                pair_events = events + offset;
                const std::uint32_t remaining = event_capacity - offset;
                retained = manifold.count < remaining
                    ? manifold.count : remaining;
            }
        }
        resolve_contacts(parameters[index], states[index],
                         parameters[collider_index], states[collider_index],
                         manifold.contacts, manifold.count, correct_position,
                         pair_events, retained);
    }
}

__global__ void resolve_uncolored_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint8_t *pair_colors, const std::uint32_t *color_state,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity,
    bool correct_position) {
    if (blockIdx.x != 0U || threadIdx.x != 0U || color_state[1] == 0U) {
        return;
    }
    for (std::uint32_t active_index = 0U;
         active_index < *active_pair_count; ++active_index) {
        if (pair_colors[active_index] != k_contact_color_overflow ||
            manifolds[active_index].count == 0U) {
            continue;
        }
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        const ContactManifold &manifold = manifolds[active_index];
        RigidContactEvent *pair_events = nullptr;
        std::uint32_t retained = 0U;
        if (event_capacity > 0U) {
            const std::uint32_t offset = event_offsets[active_index];
            if (offset < event_capacity) {
                pair_events = events + offset;
                const std::uint32_t remaining = event_capacity - offset;
                retained = manifold.count < remaining
                    ? manifold.count : remaining;
            }
        }
        resolve_contacts(parameters[index], states[index],
                         parameters[collider_index], states[collider_index],
                         manifold.contacts, manifold.count, correct_position,
                         pair_events, retained);
    }
}

__global__ void clamp_rigid_speeds_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || parameters[index].motion != MotionType::dynamic) {
        return;
    }
    states[index].linear_velocity = clamp_length(
        states[index].linear_velocity, parameters[index].maximum_linear_speed);
    states[index].angular_velocity = clamp_length(
        states[index].angular_velocity, parameters[index].maximum_angular_speed);
}

__global__ void clear_rigid_inputs_kernel(BodyAccumulator *accumulators,
                                          KinematicTarget *targets,
                                          std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    accumulators[index] = {};
    targets[index].active = false;
}

__global__ void capture_rigid_inputs_kernel(
    const BodyAccumulator *accumulators, Vec3 *forces, Vec3 *torques,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    forces[index] = accumulators[index].force;
    torques[index] = accumulators[index].torque;
}

struct CompletionState {
    cudaEvent_t event{};
    const std::uint32_t *fluid_neighbor_overflow{};
    bool acknowledged{};
    Status completion_status{};
    std::function<Status()> on_complete{};

    ~CompletionState() {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }
};

enum class TimingStage : std::uint8_t {
    rigid_integration,
    rigid_world_bounds,
    rigid_pair_filter,
    rigid_pair_compaction,
    rigid_leaf_pair_generation,
    rigid_contact_evaluation,
    rigid_contact_solve,
    rigid_input_clear,
    cloth_prediction,
    cloth_constraints,
    cloth_contacts,
    soft_body_prediction,
    soft_body_constraints,
    soft_body_contacts,
    soft_body_contact_cleanup,
    soft_body_cloth_contacts,
    fluid_soft_body_contacts,
    fluid_cloth_contacts,
    fluid_spawn,
    fluid_neighbor_sort,
    fluid_neighbor_forces,
    fluid_integration,
    fluid_static_contacts,
    fluid_body_index,
    fluid_moving_contacts,
    fluid_contact_events,
    fluid_outflow_compaction,
    rope_solve,
};

[[nodiscard]] Status wait_for_completion(
    const std::shared_ptr<CompletionState> &completion) noexcept {
    if (!completion) {
        return success();
    }
    if (completion->acknowledged) {
        return completion->completion_status;
    }
    const cudaError_t error = cudaEventSynchronize(completion->event);
    completion->completion_status = error == cudaSuccess
        ? success() : cuda_failure(error, "CUDA frame completion failed");
    Status debug_status{};
    if (error == cudaSuccess && completion->on_complete)
        debug_status = completion->on_complete();
    if (error == cudaSuccess && completion->fluid_neighbor_overflow != nullptr &&
        *completion->fluid_neighbor_overflow != 0U) {
        completion->completion_status = failure(
            StatusCode::capacity_exceeded,
            "fluid neighbor count exceeded maximum_neighbors");
    } else if (!debug_status) {
        completion->completion_status = debug_status;
    }
    completion->acknowledged = true;
    return completion->completion_status;
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }

[[nodiscard]] bool finite(Vec3 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

[[nodiscard]] bool finite(Quaternion value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

[[nodiscard]] bool finite(const RigidBodyState &state) noexcept {
    return finite(state.position) && finite(state.orientation) &&
           finite(state.linear_velocity) && finite(state.angular_velocity);
}

[[nodiscard]] bool zero(Vec3 value) noexcept {
    return value.x == 0.0F && value.y == 0.0F && value.z == 0.0F;
}

[[nodiscard]] Status validate_body_options(const RigidBodyOptions &options) noexcept {
    if (!finite(options.initial_state)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body state must contain finite values");
    }
    const float quaternion_size =
        options.initial_state.orientation.x * options.initial_state.orientation.x +
        options.initial_state.orientation.y * options.initial_state.orientation.y +
        options.initial_state.orientation.z * options.initial_state.orientation.z +
        options.initial_state.orientation.w * options.initial_state.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "rigid body orientation must be nonzero");
    }
    if (options.motion == MotionType::dynamic &&
        (!finite(options.mass) || options.mass <= 0.0F)) {
        return failure(StatusCode::invalid_argument,
                       "dynamic body mass must be finite and positive");
    }
    if (!finite(options.inertia_diagonal) ||
        (!zero(options.inertia_diagonal) &&
         (options.inertia_diagonal.x <= 0.0F ||
          options.inertia_diagonal.y <= 0.0F ||
          options.inertia_diagonal.z <= 0.0F))) {
        return failure(StatusCode::invalid_argument,
                       "inertia must be all zero or all positive");
    }
    if (!finite(options.friction) || options.friction < 0.0F ||
        !finite(options.restitution) || options.restitution < 0.0F ||
        options.restitution > 1.0F || !finite(options.linear_damping) ||
        options.linear_damping < 0.0F || !finite(options.angular_damping) ||
        options.angular_damping < 0.0F ||
        !finite(options.maximum_linear_speed) ||
        options.maximum_linear_speed <= 0.0F ||
        !finite(options.maximum_angular_speed) ||
        options.maximum_angular_speed <= 0.0F ||
        !finite(options.collision_margin) || options.collision_margin <= 0.0F) {
        return failure(StatusCode::invalid_argument,
                       "rigid body material and limits are invalid");
    }
    return success();
}

[[nodiscard]] BodyParameters make_parameters(
    const RigidBodyOptions &options, const TriangleMeshResource &mesh) noexcept {
    const Vec3 inertia = zero(options.inertia_diagonal)
                             ? multiply(mesh.unit_inertia, options.mass)
                             : options.inertia_diagonal;
    const float inverse_mass =
        options.motion == MotionType::dynamic ? 1.0F / options.mass : 0.0F;
    const Vec3 inverse_inertia = options.motion == MotionType::dynamic
                                     ? Vec3{1.0F / inertia.x, 1.0F / inertia.y,
                                            1.0F / inertia.z}
                                     : Vec3{};
    return {options.motion,
            options.mesh,
            inverse_mass,
            inverse_inertia,
            options.friction,
            options.restitution,
            options.linear_damping,
            options.angular_damping,
            options.maximum_linear_speed,
            options.maximum_angular_speed,
            options.collision_margin,
            options.user_data};
}

template <typename T>
[[nodiscard]] Status allocate_managed(T *&pointer, std::size_t count) noexcept {
    if (count == 0U) {
        pointer = nullptr;
        return success();
    }
    const cudaError_t error = cudaMallocManaged(
        reinterpret_cast<void **>(&pointer), sizeof(T) * count,
        cudaMemAttachGlobal);
    if (error != cudaSuccess) {
        pointer = nullptr;
        return cuda_failure(error, "CUDA managed allocation failed");
    }
    return success();
}

template <typename T> void release_managed(T *&pointer) noexcept {
    if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
    }
}

constexpr std::uint64_t k_fluid_empty_cell = ~std::uint64_t{0};
constexpr int k_fluid_cell_bias = 1 << 20;
constexpr std::uint32_t k_fluid_body_buckets = 4096U;

__host__ __device__ std::uint32_t fluid_body_bucket(
    int x, int y, int z) noexcept {
    const std::uint32_t hash =
        static_cast<std::uint32_t>(x) * 73856093U ^
        static_cast<std::uint32_t>(y) * 19349663U ^
        static_cast<std::uint32_t>(z) * 83492791U;
    return hash & (k_fluid_body_buckets - 1U);
}

struct FluidBodyImpulse {
    Vec3 linear{};
    Vec3 angular{};
    std::uint32_t body{k_invalid_dense};
};

struct ShapeMatrix {
    Vec3 columns[3]{};
};

struct DeformableNeighbor {
    std::uint32_t index{};
    float rest_length{};
    float compliance{};
    std::uint32_t bond{};
};

struct ClothBodyCorrection {
    Vec3 offset{};
    Vec3 impulse{};
    Vec3 contact{};
    float support_radius{};
    float weight_sum{};
    std::uint32_t vertices[3]{};
    bool active{};
};

struct ClothSeam {
    std::uint32_t corners[4]{}; // matching endpoints on the two incident faces
    std::uint32_t bond{};
    std::uint32_t bending{k_invalid_dense};
};

struct ClothStorage {
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t vertex_count{};
    std::uint32_t vertex_capacity{};
    std::uint32_t index_count{};
    float thickness{};
    float velocity_damping{};
    float contact_friction{};
    float break_strain{};
    std::uint32_t fracture_persistence_substeps{};
    float impact_break_impulse{};
    std::uint32_t solver_iterations{};
    bool preserve_volume{};
    float target_volume{};
    float volume_compliance{};
    float orientation{1.0F};
    Vec3 *positions{};
    Vec3 *scratch{};
    Vec3 *previous{};
    Vec3 *velocities{};
    float *inverse_masses{};
    std::uint32_t *indices{};
    std::uint32_t *source_indices{};
    std::uint32_t *vertex_sources{};
    std::uint8_t *free_triangle_nodes{};
    std::vector<ClothSeam> seams;
    std::vector<std::array<std::uint32_t, 2>> bond_corners;
    std::vector<float> source_inverse_masses;
    std::vector<std::uint32_t> source_degrees;
    std::vector<std::uint8_t> topology_active;
    float stretch_compliance{}, bending_compliance{};
    Vec3 *surface_positions{};
    std::uint32_t *surface_triangle_indices{};
    std::uint32_t *triangle_bonds{};
    ClothBond *bonds{};
    std::uint8_t *bond_active{};
    std::uint8_t *bond_damage{};
    std::uint32_t bond_count{};
    std::uint32_t *offsets{};
    DeformableNeighbor *neighbors{};
    std::uint32_t neighbor_count{};
    std::size_t neighbor_capacity{};
    FluidBodyImpulse *body_impulses{};
    Vec3 *rigid_contact_forces{};
    ClothBodyCorrection *body_corrections{};
    Vec3 *volume_gradients{};
    Vec3 *fluid_forces{};
    Vec3 *soft_body_forces{};
    float *volume_lambda{};
    std::uint32_t *count{};

    void release() noexcept {
        release_managed(positions);
        release_managed(scratch);
        release_managed(previous);
        release_managed(velocities);
        release_managed(inverse_masses);
        release_managed(indices);
        release_managed(source_indices);
        release_managed(vertex_sources);
        release_managed(free_triangle_nodes);
        release_managed(surface_positions);
        release_managed(surface_triangle_indices);
        release_managed(triangle_bonds);
        release_managed(bonds);
        release_managed(bond_active);
        release_managed(bond_damage);
        release_managed(offsets);
        release_managed(neighbors);
        release_managed(body_impulses);
        release_managed(rigid_contact_forces);
        release_managed(body_corrections);
        release_managed(volume_gradients);
        release_managed(fluid_forces);
        release_managed(soft_body_forces);
        release_managed(volume_lambda);
        release_managed(count);
    }
    ~ClothStorage() { release(); }
};

// Runs only at an idle frame boundary and only rebuilds when a bond changed.
// Split vertex fans across failed seams. Each new node inherits its parent's
// position/velocity; incident-face mass shares preserve total mass/momentum.
// No triangle is discarded or fitted to remote vertices after a tear.
static Status rebuild_cloth_topology(ClothStorage &cloth, bool initial = false) {
    if (!cloth.source_indices) return success();
    if (!initial && std::equal(cloth.topology_active.begin(),
                               cloth.topology_active.end(), cloth.bond_active))
        return success();
    try {
        const auto corners = cloth.index_count;
        auto count = cloth.vertex_count;
        std::vector<std::uint8_t> active(cloth.bond_active, cloth.bond_active + cloth.bond_count);
        std::vector<std::uint32_t> indices(cloth.indices, cloth.indices + corners);
        std::vector<std::array<std::uint32_t, 2>> copies;
        std::vector<std::uint32_t> parent(corners);
        for (std::uint32_t i = 0; i < corners; ++i) parent[i] = i;
        const auto root = [&](std::uint32_t i) {
            while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
            return i;
        };
        for (const auto &seam : cloth.seams) {
            if (active[seam.bond]) {
                parent[root(seam.corners[2])] = root(seam.corners[0]);
                parent[root(seam.corners[3])] = root(seam.corners[1]);
            } else if (seam.bending != k_invalid_dense) {
                active[seam.bending] = 0U;
            }
        }
        std::vector<std::uint32_t> nodes(corners, k_invalid_dense);
        std::vector<bool> used(cloth.vertex_capacity);
        std::vector<std::uint32_t> degrees(cloth.vertex_capacity);
        std::vector<std::uint8_t> free_nodes(cloth.vertex_capacity);
        for (std::uint32_t corner = 0; corner < corners; ++corner) {
            const auto group = root(corner);
            auto &node = nodes[group];
            if (node == k_invalid_dense) {
                const auto old = cloth.indices[corner];
                node = old;
                if (used[old]) {
                    if (count == cloth.vertex_capacity)
                        return failure(StatusCode::capacity_exceeded, "cloth split capacity exhausted");
                    node = count++;
                    copies.push_back({node,old});
                }
                used[node] = true;
            }
            indices[corner] = node;
            ++degrees[node];
        }
        for (std::uint32_t c = 0; c < corners; c += 3U) {
            bool detached = true;
            for (std::uint32_t k = 0; k < 3U; ++k)
                detached &= degrees[indices[c+k]] == 1U &&
                    cloth.source_inverse_masses[cloth.source_indices[c+k]] > 0.0F;
            if (detached) for (std::uint32_t k = 0; k < 3U; ++k)
                free_nodes[indices[c+k]] = 1U;
        }
        std::vector<std::vector<DeformableNeighbor>> adjacency(count);
        std::unordered_set<std::uint64_t> edges;
        const auto link = [&](std::uint32_t a, std::uint32_t b, float rest,
                              float compliance, std::uint32_t bond) {
            const auto key = (static_cast<std::uint64_t>(std::min(a,b)) << 32U) |
                             std::max(a,b);
            if (a == b || !edges.insert(key).second) return;
            adjacency[a].push_back({b, rest, compliance, bond});
            adjacency[b].push_back({a, rest, compliance, bond});
        };
        for (std::uint32_t corner = 0; corner < corners; ++corner) {
            const auto next = corner / 3U * 3U + (corner + 1U) % 3U;
            link(indices[corner], indices[next],
                 cloth.bonds[cloth.triangle_bonds[corner]].rest_length,
                 cloth.stretch_compliance, k_invalid_dense);
        }
        std::vector<ClothBond> bonds(cloth.bonds, cloth.bonds + cloth.bond_count);
        for (std::uint32_t i = 0; i < cloth.bond_count; ++i) {
            auto &bond = bonds[i];
            bond.first = indices[cloth.bond_corners[i][0]];
            bond.second = indices[cloth.bond_corners[i][1]];
            if (bond.bending && active[i])
                link(bond.first, bond.second, bond.rest_length,
                     cloth.bending_compliance, i);
        }
        // All allocations have succeeded; commit the prepared graph atomically
        // with respect to API calls. No device work is in flight at this point.
        for (const auto &copy : copies) {
            const auto node = copy[0], old = copy[1];
            cloth.positions[node] = cloth.positions[old];
            cloth.previous[node] = cloth.previous[old];
            cloth.scratch[node] = cloth.positions[old];
            cloth.velocities[node] = cloth.velocities[old];
            cloth.vertex_sources[node] = cloth.vertex_sources[old];
            cloth.rigid_contact_forces[node] = {};
            cloth.soft_body_forces[node] = {};
            if (cloth.fluid_forces) cloth.fluid_forces[node] = {};
        }
        std::copy(indices.begin(), indices.end(), cloth.indices);
        std::copy(bonds.begin(), bonds.end(), cloth.bonds);
        std::copy(active.begin(), active.end(), cloth.bond_active);
        std::copy_n(free_nodes.data(), count, cloth.free_triangle_nodes);
        std::uint32_t offset = 0;
        for (std::uint32_t node = 0; node < count; ++node) {
            const auto source = cloth.vertex_sources[node];
            cloth.inverse_masses[node] = degrees[node] == 0 ? cloth.source_inverse_masses[source] :
                cloth.source_inverse_masses[source] *
                static_cast<float>(cloth.source_degrees[source]) / degrees[node];
            cloth.offsets[node] = offset;
            for (const auto &neighbor : adjacency[node]) cloth.neighbors[offset++] = neighbor;
        }
        cloth.offsets[count] = offset;
        cloth.neighbor_count = offset;
        cloth.vertex_count = count;
        *cloth.count = count;
        cloth.topology_active.swap(active);
    } catch (...) {
        return failure(StatusCode::out_of_memory, "failed to split cloth topology");
    }
    return success();
}

struct SoftSurfaceInfluence {
    std::uint32_t corner{};
    float factor{};
};

struct SoftBodyStorage {
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t node_count{};
    std::uint32_t bond_count{};
    std::uint32_t neighbor_count{};
    std::uint32_t surface_vertex_count{};
    std::uint32_t surface_index_count{};
    float node_radius{};
    float velocity_damping{};
    float spring_damping{};
    float contact_friction{};
    float shape_matching_stiffness{};
    float shape_maximum_projection{};
    float maximum_projection_fraction{};
    float constraint_velocity_response{};
    float maximum_speed{};
    float movable_mass{};
    Vec3 shape_rest_center{};
    ShapeMatrix shape_inverse_rest{};
    std::uint32_t solver_iterations{};
    Vec3 *positions{};
    Vec3 *rest_positions{};
    Vec3 *scratch{};
    Vec3 *previous{};
    Vec3 *velocities{};
    Vec3 *velocity_scratch{};
    float *inverse_masses{};
    SoftBodyBond *bonds{};
    std::uint8_t *bond_active{};
    std::uint32_t *offsets{};
    DeformableNeighbor *neighbors{};
    Vec3 *surface_rest_positions{};
    Vec3 *surface_positions{};
    std::uint32_t *surface_indices{};
    SoftBodySurfaceBinding *surface_bindings{};
    Vec3 *surface_corner_corrections{};
    std::uint32_t *surface_node_offsets{};
    SoftSurfaceInfluence *surface_node_influences{};
    FluidBodyImpulse *body_impulses{};
    Vec3 *body_position_corrections{};
    Vec3 *cloth_forces{};
    Vec3 *fluid_forces{};
    Vec3 *rigid_contact_forces{};
    Vec3 *contact_normals{};
    Vec3 *contact_arms{};
    Vec3 *contact_momentum_delta{};
    Vec3 *contact_friction_delta{};
    float *contact_normal_delta{};
    Vec3 *predicted_momentum{};
    Quaternion *shape_orientation{};
    std::uint32_t *dynamic_contact_flag{};
    std::uint32_t *contact_count{};
    std::uint32_t *count{};

    void release() noexcept {
        release_managed(positions);
        release_managed(rest_positions);
        release_managed(scratch);
        release_managed(previous);
        release_managed(velocities);
        release_managed(velocity_scratch);
        release_managed(inverse_masses);
        release_managed(bonds);
        release_managed(bond_active);
        release_managed(offsets);
        release_managed(neighbors);
        release_managed(surface_rest_positions);
        release_managed(surface_positions);
        release_managed(surface_indices);
        release_managed(surface_bindings);
        release_managed(surface_corner_corrections);
        release_managed(surface_node_offsets);
        release_managed(surface_node_influences);
        release_managed(body_impulses);
        release_managed(body_position_corrections);
        release_managed(cloth_forces);
        release_managed(fluid_forces);
        release_managed(rigid_contact_forces);
        release_managed(contact_normals);
        release_managed(contact_arms);
        release_managed(contact_momentum_delta);
        release_managed(contact_friction_delta);
        release_managed(contact_normal_delta);
        release_managed(predicted_momentum);
        release_managed(shape_orientation);
        release_managed(dynamic_contact_flag);
        release_managed(contact_count);
        release_managed(count);
    }
    ~SoftBodyStorage() { release(); }
};

struct FluidClothCouplingResource {
    FluidClothCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};

struct FluidSoftContact {
    std::uint32_t nodes[12]{};
    float weights[12]{};
    std::uint32_t count{};
    Vec3 normal{}, relative_velocity{};
    float penetration{};
};

struct FluidSoftCouplingStorage {
    FluidSoftBodyCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    float orientation{1.0F};
    FluidSoftContact *contacts{};
    std::uint32_t *counts{};
    Vec3 *position_deltas{}, *impulses{}, *previous_surface{}, *bounds{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    BvhNode *tree{};
    std::uint32_t *triangle_order{}, *parents{}, *ready{};
    std::uint32_t tree_count{};
    void release() noexcept {
        release_managed(contacts);
        release_managed(counts);
        release_managed(position_deltas);
        release_managed(impulses);
        release_managed(previous_surface);
        release_managed(bounds);
        release_managed(contact_count);
        release_managed(maximum_penetration);
        release_managed(tree);
        release_managed(triangle_order);
        release_managed(parents);
        release_managed(ready);
    }
    ~FluidSoftCouplingStorage() { release(); }
};

struct SoftClothContact {
    std::uint32_t vertices[3]{};
    float weights[3]{};
    Vec3 position_impulse{};
    Vec3 velocity_impulse{};
    float soft_inverse_mass_fraction{};
    bool active{};
};

struct SoftClothCouplingStorage {
    SoftBodyClothCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    SoftClothContact *contacts{};
    std::uint32_t *cloth_contact_counts{};
    void release() noexcept {
        release_managed(contacts);
        release_managed(cloth_contact_counts);
    }
    ~SoftClothCouplingStorage() { release(); }
};

struct FluidContactSample {
    Vec3 position{};
    Vec3 normal{};
    float normal_impulse{};
    std::uint32_t body{k_invalid_dense};
};

struct PaintFieldResource {
    PaintFieldOptions options{};
    Vec2 *uvs{};
    std::uint32_t *pixels{};
    std::uint32_t generation{1U};
    bool alive{};
};

struct PaintRuleResource {
    PaintRuleOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};

struct FluidStorage {
    FluidOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t *count{};
    Vec3 *positions{};
    Vec3 *velocities{};
    Vec3 *previous{};
    std::uint32_t *ids{};
    float *foam{};
    float *foam_source{};
    Vec3 *next_positions{};
    Vec3 *next_velocities{};
    std::uint32_t *next_ids{};
    float *next_foam{};
    std::uint8_t *keep{};
    std::uint32_t *selected{};
    std::uint64_t *keys[2]{};
    std::uint32_t *indices[2]{};
    Vec3 *forces{};
    FluidBodyImpulse *body_impulses{};
    FluidContactSample *contact_samples{};
    std::uint8_t *contact_flags{};
    FluidContactSample *next_contact_samples{};
    std::uint8_t *next_contact_flags{};
    std::uint32_t *contact_count{};
    std::uint32_t *contact_offset{};
    std::uint8_t *sort_workspace{};
    std::size_t sort_workspace_size{};
    std::uint8_t *select_workspace{};
    std::size_t select_workspace_size{};
    std::uint32_t next_id{};
    std::uint64_t initial_count{};
    std::uint64_t emitted_count{};

    ~FluidStorage() {
        release_managed(count);
        release_managed(positions);
        release_managed(velocities);
        release_managed(previous);
        release_managed(ids);
        release_managed(foam);
        release_managed(foam_source);
        release_managed(next_positions);
        release_managed(next_velocities);
        release_managed(next_ids);
        release_managed(next_foam);
        release_managed(keep);
        release_managed(selected);
        release_managed(keys[0]);
        release_managed(keys[1]);
        release_managed(indices[0]);
        release_managed(indices[1]);
        release_managed(forces);
        release_managed(body_impulses);
        release_managed(contact_samples);
        release_managed(contact_flags);
        release_managed(next_contact_samples);
        release_managed(next_contact_flags);
        release_managed(contact_count);
        release_managed(contact_offset);
        release_managed(sort_workspace);
        release_managed(select_workspace);
    }
};

struct ParticleSourceData {
    Vec3 *points{};
    std::uint8_t *vacant{};
    std::uint32_t *capacity_misses{};
    std::uint32_t count{};
    float spacing{};
    ~ParticleSourceData() {
        release_managed(points);
        release_managed(vacant);
        release_managed(capacity_misses);
    }
};

struct ParticleSourceSlot {
    ParticleSourceOptions options{};
    std::uint32_t generation{1U};
    std::unique_ptr<ParticleSourceData> data{};
    bool alive{};
};

struct DestroyPlaneSlot {
    ParticleDestroyPlaneOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};

__host__ __device__ std::uint64_t fluid_cell_key(int x, int y, int z) noexcept {
    x = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, x));
    y = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, y));
    z = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, z));
    return (static_cast<std::uint64_t>(x + k_fluid_cell_bias) << 42U) |
           (static_cast<std::uint64_t>(y + k_fluid_cell_bias) << 21U) |
           static_cast<std::uint64_t>(z + k_fluid_cell_bias);
}

__device__ std::uint32_t fluid_lower_bound(const std::uint64_t *keys,
                                           std::uint32_t size,
                                           std::uint64_t key) noexcept {
    std::uint32_t lo = 0U, hi = size;
    while (lo < hi) {
        const std::uint32_t middle = lo + (hi - lo) / 2U;
        if (keys[middle] < key) lo = middle + 1U;
        else hi = middle;
    }
    return lo;
}

__global__ void fluid_emit_cells(const Vec3 *positions, const std::uint32_t *count,
                                 std::uint32_t capacity, float inverse_radius,
                                 std::uint64_t *keys,
                                 std::uint32_t *indices) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= capacity) return;
    indices[index] = index;
    if (index >= *count) {
        keys[index] = k_fluid_empty_cell;
        return;
    }
    const Vec3 position = positions[index];
    keys[index] = fluid_cell_key(
        __float2int_rd(position.x * inverse_radius),
        __float2int_rd(position.y * inverse_radius),
        __float2int_rd(position.z * inverse_radius));
}

__global__ void fluid_compute_forces(
    const Vec3 *positions, const Vec3 *velocities, const std::uint32_t *count,
    const std::uint64_t *keys, const std::uint32_t *indices,
    const float *foam, FluidOptions options, Vec3 up,
    Vec3 *forces, float *foam_source, std::uint32_t *overflow,
    std::uint32_t *maximum_neighbor_count) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    const Vec3 p = positions[particle], v = velocities[particle];
    const float inverse_radius = 1.0F / options.support_radius;
    const int cx = __float2int_rd(p.x * inverse_radius);
    const int cy = __float2int_rd(p.y * inverse_radius);
    const int cz = __float2int_rd(p.z * inverse_radius);
    Vec3 acceleration{};
    Vec3 outward{};
    float weight = 0.0F;
    float relative_speed_squared = 0.0F;
    float neighboring_foam = 0.0F;
    std::uint32_t neighbors = 0U;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (cx + dx < -k_fluid_cell_bias ||
                    cx + dx >= k_fluid_cell_bias ||
                    cy + dy < -k_fluid_cell_bias ||
                    cy + dy >= k_fluid_cell_bias ||
                    cz + dz < -k_fluid_cell_bias ||
                    cz + dz >= k_fluid_cell_bias) continue;
                const std::uint64_t key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (std::uint32_t item = fluid_lower_bound(keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item) {
                    const std::uint32_t other = indices[item];
                    if (other == particle) continue;
                    const Vec3 delta = subtract(p, positions[other]);
                    const float squared = length_squared(delta);
                    if (squared >= options.support_radius * options.support_radius)
                        continue;
                    ++neighbors;
                    const float distance = sqrtf(fmaxf(squared, 1.0e-12F));
                    // Stable fallback separates coincident particles without NaNs.
                    const Vec3 direction = squared > 1.0e-12F
                        ? multiply(delta, 1.0F / distance)
                        : (particle < other ? Vec3{-1.0F, 0.0F, 0.0F}
                                            : Vec3{1.0F, 0.0F, 0.0F});
                    const float q = 1.0F - distance * inverse_radius;
                    outward = add(outward, multiply(direction, q));
                    weight += q;
                    const Vec3 relative_velocity =
                        subtract(velocities[other], v);
                    relative_speed_squared +=
                        length_squared(relative_velocity) * q;
                    neighboring_foam = fmaxf(neighboring_foam,
                                               foam[other] * q);
                    const float radial_speed = dot(relative_velocity, direction);
                    acceleration = add(acceleration,
                        add(multiply(direction, options.repulsion *
                            (1'000.0F / options.rest_density) * q * q +
                            options.normal_damping * radial_speed),
                            multiply(relative_velocity,
                                     options.viscosity * q)));
                }
            }
        }
    }
    if (options.maximum_pair_acceleration > 0.0F)
        acceleration = clamp_length(acceleration,
                                    options.maximum_pair_acceleration);
    forces[particle] = acceleration;
    const float exposure = vector_length(outward) / fmaxf(weight, 1.0e-6F);
    const float upward = fmaxf(0.0F, dot(normalized_or(outward, up), up));
    const float agitation = sqrtf(relative_speed_squared /
                                  fmaxf(weight, 1.0e-6F));
    foam_source[particle] = fmaxf(
        clamp_scalar((exposure - 0.12F) * 2.0F, 0.0F, 1.0F) * upward *
            clamp_scalar((agitation - 0.15F) * 1.5F, 0.0F, 1.0F),
        neighboring_foam * upward * 0.9F);
    atomicMax(maximum_neighbor_count, neighbors);
    if (neighbors > options.maximum_neighbors) atomicAdd(overflow, 1U);
}

__global__ void fluid_integrate(Vec3 *positions, Vec3 *velocities,
                                Vec3 *previous, float *foam,
                                const Vec3 *forces, const float *foam_source,
                                const std::uint32_t *count,
                                FluidOptions options, Vec3 gravity, float dt) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= *count) return;
    previous[index] = positions[index];
    Vec3 velocity = add(velocities[index],
                        multiply(add(gravity, forces[index]), dt));
    velocity = clamp_length(
        multiply(velocity, expf(-options.velocity_damping * dt)),
        options.maximum_speed);
    const Vec3 position = add(positions[index], multiply(velocity, dt));
    if (isfinite(position.x) && isfinite(position.y) && isfinite(position.z) &&
        isfinite(velocity.x) && isfinite(velocity.y) && isfinite(velocity.z)) {
        positions[index] = position;
        velocities[index] = velocity;
    }
    foam[index] = fmaxf(fmaxf(0.0F, foam[index] - dt * 0.7F),
                        foam_source[index]);
}

__device__ Vec3 fluid_closest_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) noexcept {
    const Vec3 ab = subtract(b, a), ac = subtract(c, a), ap = subtract(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) return a;
    const Vec3 bp = subtract(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F)
        return add(a, multiply(ab, d1 / (d1 - d3)));
    const Vec3 cp = subtract(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F)
        return add(a, multiply(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F)
        return add(b, multiply(subtract(c, b),
                               (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    const float denominator = va + vb + vc;
    return denominator > 1.0e-12F
        ? add(a, add(multiply(ab, vb / denominator),
                     multiply(ac, vc / denominator))) : a;
}

__device__ Vec3 fluid_closest_triangle_barycentric(
    Vec3 p, Vec3 a, Vec3 b, Vec3 c, Vec3 &weights) noexcept {
    const Vec3 ab = subtract(b, a), ac = subtract(c, a), ap = subtract(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) {
        weights = {1.0F, 0.0F, 0.0F};
        return a;
    }
    const Vec3 bp = subtract(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) {
        weights = {0.0F, 1.0F, 0.0F};
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        const float v = d1 / (d1 - d3);
        weights = {1.0F - v, v, 0.0F};
        return add(a, multiply(ab, v));
    }
    const Vec3 cp = subtract(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) {
        weights = {0.0F, 0.0F, 1.0F};
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        const float w = d2 / (d2 - d6);
        weights = {1.0F - w, 0.0F, w};
        return add(a, multiply(ac, w));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        weights = {0.0F, 1.0F - w, w};
        return add(b, multiply(subtract(c, b), w));
    }
    const float denominator = va + vb + vc;
    if (denominator <= 1.0e-12F) {
        weights = {1.0F, 0.0F, 0.0F};
        return a;
    }
    const float inverse = 1.0F / denominator;
    const float v = vb * inverse, w = vc * inverse;
    weights = {1.0F - v - w, v, w};
    return add(a, add(multiply(ab, v), multiply(ac, w)));
}

__device__ void atomic_add(Vec3 *destination, Vec3 value) noexcept {
    atomicAdd(&destination->x, value.x);
    atomicAdd(&destination->y, value.y);
    atomicAdd(&destination->z, value.z);
}

__global__ void fluid_cloth_containment_forces(
    const Vec3 *particle_positions, const Vec3 *particle_velocities,
    const std::uint32_t *particle_count, Vec3 *particle_accelerations,
    float *foam_source, float particle_mass,
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *cloth_indices, std::uint32_t triangle_count,
    float cloth_orientation, FluidClothCouplingOptions options,
    Vec3 *cloth_forces) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *particle_count) return;
    const Vec3 point = particle_positions[particle];
    float best_squared = FLT_MAX;
    Vec3 best_point{}, best_normal{}, best_weights{};
    std::uint32_t best_triangle = 0U;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t a_index = cloth_indices[3U * triangle];
        const std::uint32_t b_index = cloth_indices[3U * triangle + 1U];
        const std::uint32_t c_index = cloth_indices[3U * triangle + 2U];
        const Vec3 a = cloth_positions[a_index];
        const Vec3 b = cloth_positions[b_index];
        const Vec3 c = cloth_positions[c_index];
        Vec3 weights{};
        const Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        const float squared = length_squared(subtract(point, nearest));
        if (squared >= best_squared) continue;
        const Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) <= 1.0e-14F) continue;
        best_squared = squared;
        best_point = nearest;
        best_normal = multiply(normalized_or(face, {0.0F, 1.0F, 0.0F}),
                               cloth_orientation);
        best_weights = weights;
        best_triangle = triangle;
    }
    if (best_squared == FLT_MAX) return;
    const float signed_distance = dot(subtract(point, best_point), best_normal);
    const float violation = options.contact_distance + signed_distance;
    const bool outside = signed_distance > 0.0F;
    if (violation <= 0.0F ||
        (!outside && best_squared >
            options.interaction_radius * options.interaction_radius)) return;
    const std::uint32_t a = cloth_indices[3U * best_triangle];
    const std::uint32_t b = cloth_indices[3U * best_triangle + 1U];
    const std::uint32_t c = cloth_indices[3U * best_triangle + 2U];
    const Vec3 surface_velocity = add(
        multiply(cloth_velocities[a], best_weights.x),
        add(multiply(cloth_velocities[b], best_weights.y),
            multiply(cloth_velocities[c], best_weights.z)));
    const Vec3 relative_velocity = subtract(
        particle_velocities[particle], surface_velocity);
    const float normal_speed = dot(relative_velocity, best_normal);
    const float magnitude = fminf(options.maximum_force,
        fmaxf(0.0F, options.stiffness * violation +
                     options.damping * normal_speed));
    const Vec3 tangent = subtract(relative_velocity,
                                  multiply(best_normal, normal_speed));
    const Vec3 force = clamp_length(add(
        multiply(best_normal, -magnitude),
        multiply(tangent, -options.tangential_drag)), options.maximum_force);
    particle_accelerations[particle] = add(
        particle_accelerations[particle], multiply(force, 1.0F / particle_mass));
    foam_source[particle] = fmaxf(foam_source[particle],
        clamp_scalar(magnitude / fmaxf(options.maximum_force, 1.0F),
                     0.0F, 1.0F));
    const Vec3 reaction = multiply(force, -1.0F);
    atomic_add(cloth_forces + a, multiply(reaction, best_weights.x));
    atomic_add(cloth_forces + b, multiply(reaction, best_weights.y));
    atomic_add(cloth_forces + c, multiply(reaction, best_weights.z));
}

__global__ void cloth_apply_fluid_forces(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    const Vec3 *forces, std::uint32_t count, float dt) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count || inverse_masses[vertex] == 0.0F) return;
    const Vec3 velocity_change = clamp_length(
        multiply(forces[vertex], inverse_masses[vertex] * dt), 2.0F);
    velocities[vertex] = clamp_length(
        add(velocities[vertex], velocity_change), 20.0F);
    positions[vertex] = add(positions[vertex], multiply(velocity_change, dt));
}

__global__ void fluid_project_inside_cloth(
    Vec3 *particle_positions, Vec3 *particle_velocities,
    const std::uint32_t *particle_count, float contact_distance,
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *cloth_indices, std::uint32_t triangle_count,
    float cloth_orientation) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *particle_count) return;
    const Vec3 point = particle_positions[particle];
    float best_squared = FLT_MAX;
    Vec3 best_point{}, best_normal{}, best_weights{};
    std::uint32_t best_triangle = 0U;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t a_index = cloth_indices[3U * triangle];
        const std::uint32_t b_index = cloth_indices[3U * triangle + 1U];
        const std::uint32_t c_index = cloth_indices[3U * triangle + 2U];
        const Vec3 a = cloth_positions[a_index];
        const Vec3 b = cloth_positions[b_index];
        const Vec3 c = cloth_positions[c_index];
        Vec3 weights{};
        const Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        const float squared = length_squared(subtract(point, nearest));
        if (squared >= best_squared) continue;
        const Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) <= 1.0e-14F) continue;
        best_squared = squared;
        best_point = nearest;
        best_normal = multiply(normalized_or(face, {0.0F, 1.0F, 0.0F}),
                               cloth_orientation);
        best_weights = weights;
        best_triangle = triangle;
    }
    if (best_squared == FLT_MAX) return;
    const float signed_distance = dot(subtract(point, best_point), best_normal);
    const float violation = contact_distance + signed_distance;
    if (violation <= 0.0F) return;
    const std::uint32_t a = cloth_indices[3U * best_triangle];
    const std::uint32_t b = cloth_indices[3U * best_triangle + 1U];
    const std::uint32_t c = cloth_indices[3U * best_triangle + 2U];
    particle_positions[particle] = subtract(
        particle_positions[particle], multiply(best_normal, violation));
    const Vec3 surface_velocity = add(
        multiply(cloth_velocities[a], best_weights.x),
        add(multiply(cloth_velocities[b], best_weights.y),
            multiply(cloth_velocities[c], best_weights.z)));
    Vec3 relative = subtract(particle_velocities[particle], surface_velocity);
    const float outward_speed = dot(relative, best_normal);
    if (outward_speed > 0.0F)
        relative = subtract(relative, multiply(best_normal, outward_speed));
    particle_velocities[particle] = add(surface_velocity, relative);
}

__device__ bool fluid_segment_bounds(Vec3 a, Vec3 b, const BvhNode &node,
                                     float radius) noexcept {
    float lower = 0.0F, upper = 1.0F;
    const float starts[3]{a.x, a.y, a.z};
    const float ends[3]{b.x, b.y, b.z};
    const float minima[3]{node.minimum.x, node.minimum.y, node.minimum.z};
    const float maxima[3]{node.maximum.x, node.maximum.y, node.maximum.z};
    for (int axis = 0; axis < 3; ++axis) {
        const float delta = ends[axis] - starts[axis];
        const float minimum = minima[axis] - radius;
        const float maximum = maxima[axis] + radius;
        if (fabsf(delta) < 1.0e-9F) {
            if (starts[axis] < minimum || starts[axis] > maximum) return false;
        } else {
            const float first = (minimum - starts[axis]) / delta;
            const float second = (maximum - starts[axis]) / delta;
            lower = fmaxf(lower, fminf(first, second));
            upper = fminf(upper, fmaxf(first, second));
            if (lower > upper) return false;
        }
    }
    return true;
}

__device__ __noinline__ void stamp_paint_at_contact(
    Vec3 local_particle, float particle_radius, FluidId source,
    RigidBodyId target, const TriangleMeshResource *meshes,
    const PaintFieldResource *fields, std::uint32_t field_capacity,
    const PaintRuleResource *rules, std::uint32_t rule_capacity) noexcept {
    for (std::uint32_t rule_index = 0; rule_index < rule_capacity;
         ++rule_index) {
        const PaintRuleResource rule = rules[rule_index];
        if (!rule.alive || !rule.options.enabled ||
            rule.options.source.index != source.index ||
            rule.options.source.generation != source.generation ||
            rule.options.target.index >= field_capacity) continue;
        const PaintFieldResource field = fields[rule.options.target.index];
        if (!field.alive || field.options.cloth.generation != 0U ||
            field.generation != rule.options.target.generation ||
            field.options.body.index != target.index ||
            field.options.body.generation != target.generation) continue;
        const TriangleMeshResource mesh = meshes[field.options.mesh.index];
        const float reach = particle_radius + rule.options.reach;
        const float reach_squared = reach * reach;
        float best_distance = reach_squared;
        Vec2 best_uv{};
        std::uint32_t best_side = 0U;
        std::uint32_t stack[64]{};
        int pending = mesh.bvh_node_count == 0U ? 0 : 1;
        while (pending != 0) {
            const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
            if (!fluid_segment_bounds(local_particle, local_particle,
                                      node, reach)) continue;
            if (node.triangle_count == 0U) {
                if (pending + 2 > 64) continue;
                stack[pending++] = node.right;
                stack[pending++] = node.left;
                continue;
            }
            for (std::uint32_t item = 0; item < node.triangle_count; ++item) {
                const std::uint32_t base =
                    (node.first_triangle + item) * 3U;
                const std::uint32_t ia = mesh.indices[base];
                const std::uint32_t ib = mesh.indices[base + 1U];
                const std::uint32_t ic = mesh.indices[base + 2U];
                const Vec3 a = mesh.vertices[ia], b = mesh.vertices[ib];
                const Vec3 c = mesh.vertices[ic];
                const Vec3 nearest = fluid_closest_triangle(
                    local_particle, a, b, c);
                const Vec3 delta = subtract(local_particle, nearest);
                const float distance = length_squared(delta);
                if (distance >= best_distance) continue;
                const Vec3 ab = subtract(b, a), ac = subtract(c, a);
                const Vec3 ap = subtract(nearest, a);
                const float d00 = dot(ab, ab), d01 = dot(ab, ac);
                const float d11 = dot(ac, ac), d20 = dot(ap, ab);
                const float d21 = dot(ap, ac);
                const float divisor = d00 * d11 - d01 * d01;
                if (divisor <= 1.0e-12F) continue;
                const float v = (d11*d20 - d01*d21) / divisor;
                const float w = (d00*d21 - d01*d20) / divisor;
                const float u = 1.0F - v - w;
                best_uv = {u*field.uvs[ia].x + v*field.uvs[ib].x +
                               w*field.uvs[ic].x,
                           u*field.uvs[ia].y + v*field.uvs[ib].y +
                               w*field.uvs[ic].y};
                best_side = dot(cross(ab, ac), delta) >= 0.0F ? 1U : 2U;
                best_distance = distance;
            }
        }
        if (best_side == 0U) continue;
        const int width = static_cast<int>(field.options.width);
        const int height = static_cast<int>(field.options.height);
        int x = static_cast<int>(floorf(best_uv.x * width)) % width;
        if (x < 0) x += width;
        const int y = max(0, min(height - 1,
            static_cast<int>(floorf(best_uv.y * height))));
        atomicOr(field.pixels + y * width + x, best_side);
    }
}

__global__ void fluid_static_contacts(
    Vec3 *positions, Vec3 *velocities, const Vec3 *previous, float *foam,
    const std::uint32_t *count, float radius, float spawn_clearance,
    std::uint32_t first_spawned, bool recover_spawn, Vec3 up,
    const BodyParameters *parameters, const RigidBodyState *states,
    const TriangleMeshResource *meshes, std::uint32_t body_index,
    float particle_mass, bool collect_contacts,
    FluidContactSample *samples, std::uint8_t *contact_flags,
    FluidId fluid_id, RigidBodyId body_id, bool apply_paint,
    const PaintFieldResource *paint_fields, std::uint32_t field_capacity,
    const PaintRuleResource *paint_rules, std::uint32_t rule_capacity) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    const BodyParameters body = parameters[body_index];
    const RigidBodyState state = states[body_index];
    const TriangleMeshResource mesh = meshes[body.mesh.index];
    if (particle >= *count || mesh.bvh_node_count == 0U) return;
    const Vec3 origin = inverse_rotate(state.orientation,
        subtract(previous[particle], state.position));
    Vec3 position = inverse_rotate(state.orientation,
        subtract(positions[particle], state.position));
    Vec3 velocity = inverse_rotate(state.orientation, velocities[particle]);
    const bool newly_spawned = recover_spawn && particle >= first_spawned;
    const float query_radius = newly_spawned
        ? fmaxf(radius, spawn_clearance) : radius;
    const Vec3 local_up = newly_spawned
        ? inverse_rotate(state.orientation, up) : Vec3{};
    float best_penetration = 0.0F;
    Vec3 best_normal{};
    Vec3 best_contact{};
    std::uint32_t stack[64]{};
    int pending = 1;
    while (pending != 0) {
        const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
        if (!fluid_segment_bounds(origin, position, node, query_radius)) continue;
        if (node.triangle_count == 0U) {
            if (pending + 2 > 64) continue;
            stack[pending++] = node.right;
            stack[pending++] = node.left;
            continue;
        }
        for (std::uint32_t item = 0U; item < node.triangle_count; ++item) {
            const std::uint32_t triangle =
                (node.first_triangle + item) * 3U;
            const Vec3 a = mesh.vertices[mesh.indices[triangle]];
            const Vec3 b = mesh.vertices[mesh.indices[triangle + 1U]];
            const Vec3 c = mesh.vertices[mesh.indices[triangle + 2U]];
            const Vec3 face = normalized_or(cross(subtract(b, a),
                                                  subtract(c, a)),
                                            {0.0F, 1.0F, 0.0F});
            const Vec3 closest = fluid_closest_triangle(position, a, b, c);
            const Vec3 delta = subtract(position, closest);
            const float distance = vector_length(delta);
            Vec3 normal = distance > 1.0e-6F
                ? multiply(delta, 1.0F / distance)
                : multiply(face, dot(subtract(origin, a), face) >= 0.0F
                                     ? 1.0F : -1.0F);
            float penetration = radius - distance;
            // An open triangle also catches a particle that crosses between
            // samples, even if its endpoint is already beyond the radius.
            const float before = dot(subtract(origin, a), face);
            const float after = dot(subtract(position, a), face);
            if (before * after < 0.0F) {
                const float fraction = before / (before - after);
                const Vec3 crossing = add(origin,
                    multiply(subtract(position, origin), fraction));
                if (length_squared(subtract(fluid_closest_triangle(
                        crossing, a, b, c), crossing)) < radius * radius) {
                    normal = multiply(face, before > 0.0F ? 1.0F : -1.0F);
                    penetration = fmaxf(penetration,
                        radius + fabsf(after));
                }
            }
            // A flow plane can overlap an open terrain mesh. Recover only
            // newly emitted particles close to a floor-facing triangle;
            // subsequent motion still uses swept, two-sided contacts.
            if (newly_spawned && fabsf(dot(face, local_up)) > 0.7F) {
                const Vec3 floor_normal = dot(face, local_up) > 0.0F
                    ? face : multiply(face, -1.0F);
                const float side = dot(subtract(position, a), floor_normal);
                if (side < radius && side > -spawn_clearance &&
                    distance < spawn_clearance) {
                    normal = floor_normal;
                    penetration = fmaxf(penetration, radius - side);
                }
            }
            if (penetration > best_penetration) {
                best_penetration = penetration;
                best_normal = normal;
                best_contact = closest;
            }
        }
    }
    if (best_penetration > 0.0F) {
        if (apply_paint && rule_capacity != 0U)
            stamp_paint_at_contact(position, radius, fluid_id, body_id,
                meshes, paint_fields, field_capacity, paint_rules,
                rule_capacity);
        position = add(position, multiply(best_normal, best_penetration));
        const float incoming = dot(velocity, best_normal);
        const float normal_impulse = incoming < 0.0F
            ? -incoming * (1.0F + body.restitution) * particle_mass : 0.0F;
        if (incoming < 0.0F) {
            velocity = subtract(velocity,
                multiply(best_normal, incoming * (1.0F + body.restitution)));
            const Vec3 tangent = subtract(velocity,
                multiply(best_normal, dot(velocity, best_normal)));
            const float tangent_speed = vector_length(tangent);
            if (tangent_speed > k_epsilon) {
                // Coulomb friction is limited by this contact's normal
                // impulse, as in fluid_moving_contacts. A fixed fractional
                // cut on every solver pass overdamps water at rest on walls.
                const float friction_speed = fminf(tangent_speed,
                    body.friction * normal_impulse / particle_mass);
                velocity = subtract(velocity, multiply(tangent,
                    friction_speed / tangent_speed));
            }
            foam[particle] = fmaxf(foam[particle],
                                  fminf(1.0F, -incoming * 0.35F));
        }
        positions[particle] = add(state.position,
                                  rotate(state.orientation, position));
        velocities[particle] = rotate(state.orientation, velocity);
        if (collect_contacts &&
            (contact_flags[particle] == 0U ||
             (contact_flags[particle] == 1U &&
              normal_impulse > samples[particle].normal_impulse))) {
            samples[particle] = {
                transform_point(state, best_contact),
                rotate(state.orientation, best_normal),
                normal_impulse, body_index};
            contact_flags[particle] = 1U;
        }
    }
}

__global__ void fluid_body_bounds_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous,
    const RigidBodyState *current, const TriangleMeshResource *meshes,
    std::uint32_t count, WorldAabb *bounds) {
    const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= count || parameters[body].motion == MotionType::static_body)
        return;
    const TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
    Vec3 minimum{}, maximum{};
    transformed_motion_bounds(mesh.minimum, mesh.maximum,
        bounds_transform(previous[body]), bounds_transform(current[body]),
        true, parameters[body].collision_margin, minimum, maximum);
    // Endpoint AABBs alone can miss the middle of a fast rotation.
    const float rotation_reach = rotational_motion_bound(
        previous[body], current[body], mesh);
    const Vec3 expansion{rotation_reach, rotation_reach, rotation_reach};
    minimum = subtract(minimum, expansion);
    maximum = add(maximum, expansion);
    bounds[body] = {minimum, maximum};
}

__global__ void fluid_index_body_cells(
    const BodyParameters *parameters, const WorldAabb *bounds,
    std::uint32_t body_count, std::uint32_t words, float padding,
    unsigned long long *masks, unsigned long long *global_masks) {
    const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= body_count || parameters[body].motion == MotionType::static_body)
        return;
    const WorldAabb box = bounds[body];
    const int x0 = __float2int_rd(box.minimum.x - padding);
    const int y0 = __float2int_rd(box.minimum.y - padding);
    const int z0 = __float2int_rd(box.minimum.z - padding);
    const int x1 = __float2int_rd(box.maximum.x + padding);
    const int y1 = __float2int_rd(box.maximum.y + padding);
    const int z1 = __float2int_rd(box.maximum.z + padding);
    const unsigned long long bit = 1ULL << (body & 63U);
    const std::uint32_t word = body / 64U;
    if (static_cast<std::int64_t>(x1) - x0 > 3 ||
        static_cast<std::int64_t>(y1) - y0 > 3 ||
        static_cast<std::int64_t>(z1) - z0 > 3) {
        atomicOr(global_masks + word, bit);
        return;
    }
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                atomicOr(masks + fluid_body_bucket(x, y, z) * words + word,
                         bit);
}

__global__ void fluid_moving_contacts(
    Vec3 *positions, Vec3 *velocities, const Vec3 *previous, float *foam,
    const std::uint32_t *count, float radius, float particle_mass,
    float timestep, float maximum_speed,
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states, const TriangleMeshResource *meshes,
    const WorldAabb *bounds, std::uint32_t body_count,
    const unsigned long long *masks, const unsigned long long *global_masks,
    std::uint32_t words, bool first_iteration, FluidBodyImpulse *impulses,
    std::uint32_t *contact_flags, bool collect_contacts,
    FluidContactSample *samples, std::uint8_t *particle_contact_flags,
    FluidId fluid_id, const RigidBodyId *body_ids, bool apply_paint,
    const PaintFieldResource *paint_fields, std::uint32_t field_capacity,
    const PaintRuleResource *paint_rules, std::uint32_t rule_capacity) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    impulses[particle] = {};
    const Vec3 start = previous[particle];
    const Vec3 end = positions[particle];
    float best_penetration = 0.0F;
    Vec3 best_normal{}, best_contact{};
    std::uint32_t best_body = k_invalid_dense;
    const std::uint32_t bucket = fluid_body_bucket(
        __float2int_rd(end.x), __float2int_rd(end.y),
        __float2int_rd(end.z));
    for (std::uint32_t word = 0U; word < words; ++word) {
        unsigned long long candidates =
            masks[bucket * words + word] | global_masks[word];
        while (candidates != 0ULL) {
            const std::uint32_t bit =
                static_cast<std::uint32_t>(__ffsll(candidates) - 1);
            candidates &= candidates - 1ULL;
            const std::uint32_t body_index = word * 64U + bit;
            if (body_index >= body_count) continue;
            const BodyParameters body = parameters[body_index];
            if (body.motion == MotionType::static_body) continue;
            const WorldAabb box = bounds[body_index];
            if (fmaxf(start.x, end.x) + radius < box.minimum.x ||
                fminf(start.x, end.x) - radius > box.maximum.x ||
                fmaxf(start.y, end.y) + radius < box.minimum.y ||
                fminf(start.y, end.y) - radius > box.maximum.y ||
                fmaxf(start.z, end.z) + radius < box.minimum.z ||
                fminf(start.z, end.z) - radius > box.maximum.z) continue;
            const RigidBodyState state = states[body_index];
            const RigidBodyState old = first_iteration
                ? previous_states[body_index] : state;
            const Vec3 origin = inverse_rotate(old.orientation,
                subtract(start, old.position));
        const Vec3 position = inverse_rotate(state.orientation,
            subtract(end, state.position));
        const TriangleMeshResource mesh = meshes[body.mesh.index];
        if (mesh.bvh_node_count == 0U) continue;
        std::uint32_t stack[64]{};
        int pending = 1;
        while (pending != 0) {
            const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
            if (!fluid_segment_bounds(origin, position, node, radius))
                continue;
            if (node.triangle_count == 0U) {
                if (pending + 2 > 64) continue;
                stack[pending++] = node.right;
                stack[pending++] = node.left;
                continue;
            }
            for (std::uint32_t item = 0U; item < node.triangle_count;
                 ++item) {
                const std::uint32_t triangle =
                    (node.first_triangle + item) * 3U;
                const Vec3 a = mesh.vertices[mesh.indices[triangle]];
                const Vec3 b = mesh.vertices[mesh.indices[triangle + 1U]];
                const Vec3 c = mesh.vertices[mesh.indices[triangle + 2U]];
                const Vec3 face = normalized_or(cross(subtract(b, a),
                    subtract(c, a)), {0.0F, 1.0F, 0.0F});
                const Vec3 closest = fluid_closest_triangle(position, a, b, c);
                const Vec3 delta = subtract(position, closest);
                const float distance = vector_length(delta);
                Vec3 normal = distance > 1.0e-6F
                    ? multiply(delta, 1.0F / distance)
                    : multiply(face, dot(subtract(origin, a), face) >= 0.0F
                                         ? 1.0F : -1.0F);
                float penetration = radius - distance;
                const float before = dot(subtract(origin, a), face);
                const float after = dot(subtract(position, a), face);
                if (before * after < 0.0F) {
                    const float fraction = before / (before - after);
                    const Vec3 crossing = add(origin,
                        multiply(subtract(position, origin), fraction));
                    if (length_squared(subtract(fluid_closest_triangle(
                            crossing, a, b, c), crossing)) < radius * radius) {
                        normal = multiply(face, before > 0.0F ? 1.0F : -1.0F);
                        penetration = fmaxf(penetration,
                                            radius + fabsf(after));
                    }
                }
                if (penetration > best_penetration) {
                    best_penetration = penetration;
                    best_normal = rotate(state.orientation, normal);
                    best_contact = transform_point(state, closest);
                    best_body = body_index;
                }
            }
        }
        }
    }
    if (best_body == k_invalid_dense) return;
    const BodyParameters body = parameters[best_body];
    const RigidBodyState state = states[best_body];
    if (apply_paint && rule_capacity != 0U) {
        const Vec3 local_particle = inverse_rotate(state.orientation,
            subtract(end, state.position));
        stamp_paint_at_contact(local_particle, radius, fluid_id,
            body_ids[best_body], meshes, paint_fields, field_capacity,
            paint_rules, rule_capacity);
    }
    positions[particle] = add(end, multiply(best_normal, best_penetration));
    if (collect_contacts && particle_contact_flags[particle] != 2U) {
        samples[particle] = {best_contact, best_normal, 0.0F, best_body};
        particle_contact_flags[particle] = 2U;
    }
    Vec3 velocity = velocities[particle];
    const Vec3 arm = subtract(best_contact, state.position);
    const Vec3 body_velocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    const Vec3 relative = subtract(velocity, body_velocity);
    const float incoming = dot(relative, best_normal);
    // A resting particle can be pushed into a moving surface without having
    // negative normal velocity. Share a bounded overlap-recovery impulse with
    // the body instead of silently moving only the particle.
    const float recovery_speed = fminf(maximum_speed,
        fminf(0.5F * radius, 0.2F * best_penetration) / timestep);
    if (incoming >= recovery_speed) return;
    const Vec3 normal_cross = cross(arm, best_normal);
    const float normal_denominator = 1.0F / particle_mass +
        body.inverse_mass + dot(cross(inverse_inertia_world(body, state,
            normal_cross), arm), best_normal);
    if (normal_denominator <= k_epsilon) return;
    const float normal_impulse =
        (recovery_speed - incoming) / normal_denominator;
    if (collect_contacts &&
        (particle_contact_flags[particle] != 2U ||
         normal_impulse > samples[particle].normal_impulse)) {
        samples[particle] = {best_contact, best_normal, normal_impulse,
                             best_body};
        particle_contact_flags[particle] = 2U;
    }
    Vec3 impulse = multiply(best_normal, normal_impulse);
    velocity = add(velocity, multiply(impulse, 1.0F / particle_mass));
    const Vec3 tangent_velocity = subtract(relative,
        multiply(best_normal, incoming));
    const float tangent_speed = vector_length(tangent_velocity);
    if (tangent_speed > k_epsilon) {
        const Vec3 tangent = multiply(tangent_velocity, 1.0F / tangent_speed);
        const float tangent_denominator = 1.0F / particle_mass +
            body.inverse_mass + dot(cross(inverse_inertia_world(body, state,
                cross(arm, tangent)), arm), tangent);
        if (tangent_denominator > k_epsilon) {
            const float tangent_impulse = fminf(
                tangent_speed / tangent_denominator,
                body.friction * normal_impulse);
            const Vec3 friction = multiply(tangent, -tangent_impulse);
            impulse = add(impulse, friction);
            velocity = add(velocity, multiply(friction,
                                              1.0F / particle_mass));
        }
    }
    velocities[particle] = velocity;
    foam[particle] = fmaxf(foam[particle],
                          fminf(1.0F, -incoming * 0.35F));
    impulses[particle] = {multiply(impulse, -1.0F),
                          multiply(cross(arm, impulse), -1.0F), best_body};
    atomicExch(contact_flags + best_body, 1U);
}

__global__ void reduce_point_body_impulses(
    const FluidBodyImpulse *impulses, const std::uint32_t *particle_count,
    const BodyParameters *parameters, RigidBodyState *states,
    const std::uint32_t *contact_flags, std::uint32_t body_count,
    const Vec3 *position_corrections = nullptr) {
    const std::uint32_t body = blockIdx.x;
    if (body >= body_count || contact_flags[body] == 0U ||
        parameters[body].motion != MotionType::dynamic)
        return;
    __shared__ Vec3 linear[128];
    __shared__ Vec3 angular[128];
    __shared__ Vec3 position[128];
    Vec3 local_linear{}, local_angular{}, local_position{};
    for (std::uint32_t particle = threadIdx.x; particle < *particle_count;
         particle += blockDim.x) {
        const FluidBodyImpulse impulse = impulses[particle];
        if (impulse.body != body) continue;
        local_linear = add(local_linear, impulse.linear);
        local_angular = add(local_angular, impulse.angular);
        if (position_corrections != nullptr)
            local_position = add(local_position, position_corrections[particle]);
    }
    linear[threadIdx.x] = local_linear;
    angular[threadIdx.x] = local_angular;
    position[threadIdx.x] = local_position;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
        if (threadIdx.x < stride) {
            linear[threadIdx.x] = add(linear[threadIdx.x],
                                     linear[threadIdx.x + stride]);
            angular[threadIdx.x] = add(angular[threadIdx.x],
                                       angular[threadIdx.x + stride]);
            position[threadIdx.x] = add(position[threadIdx.x],
                                        position[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        RigidBodyState state = states[body];
        const BodyParameters options = parameters[body];
        state.position = add(state.position, position[0]);
        state.linear_velocity = clamp_length(add(state.linear_velocity,
            multiply(linear[0], options.inverse_mass)),
            options.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(state.angular_velocity,
            inverse_inertia_world(options, state, angular[0])),
            options.maximum_angular_speed);
        states[body] = state;
    }
}

__global__ void deformable_predict(Vec3 *positions, Vec3 *previous,
                              Vec3 *velocities, const float *inverse_masses,
                              std::uint32_t count, Vec3 gravity, float dt,
                              float damping) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    const Vec3 old = positions[vertex];
    previous[vertex] = old;
    if (inverse_masses[vertex] == 0.0F) {
        velocities[vertex] = {};
        return;
    }
    Vec3 velocity = multiply(
        velocities[vertex], 1.0F / (1.0F + damping * dt));
    velocity = add(velocity, multiply(gravity, dt));
    positions[vertex] = add(old, multiply(velocity, dt));
    velocities[vertex] = velocity;
}

__global__ void soft_body_measure_momentum(
    const Vec3 *velocities, const float *inverse_masses,
    std::uint32_t count, Vec3 *output) {
    __shared__ Vec3 values[128];
    Vec3 local{};
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass > 0.0F)
            local = add(local, multiply(velocities[node], 1.0F / inverse_mass));
    }
    values[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U;
         stride /= 2U) {
        if (threadIdx.x < stride)
            values[threadIdx.x] = add(
                values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0U) *output = values[0];
}

__global__ void soft_body_restore_momentum(
    Vec3 *velocities, const float *inverse_masses,
    const Vec3 *contact_momentum_delta, std::uint32_t count,
    float movable_mass, const Vec3 *predicted_momentum,
    const std::uint32_t *dynamic_contact_flag, float maximum_speed) {
    if (*dynamic_contact_flag == 0U) return;
    __shared__ Vec3 actual_values[128];
    __shared__ Vec3 contact_values[128];
    __shared__ Vec3 correction;
    Vec3 actual{}, contact{};
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass <= 0.0F) continue;
        actual = add(actual,
            multiply(velocities[node], 1.0F / inverse_mass));
        contact = add(contact, contact_momentum_delta[node]);
    }
    actual_values[threadIdx.x] = actual;
    contact_values[threadIdx.x] = contact;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U;
         stride /= 2U) {
        if (threadIdx.x < stride) {
            actual_values[threadIdx.x] = add(
                actual_values[threadIdx.x],
                actual_values[threadIdx.x + stride]);
            contact_values[threadIdx.x] = add(
                contact_values[threadIdx.x],
                contact_values[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const Vec3 target = add(*predicted_momentum, contact_values[0]);
        correction = multiply(
            subtract(target, actual_values[0]), 1.0F / movable_mass);
    }
    __syncthreads();
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        if (inverse_masses[node] > 0.0F)
            velocities[node] = clamp_length(
                add(velocities[node], correction), maximum_speed);
    }
}

__device__ ShapeMatrix shape_matrix_multiply(
    const ShapeMatrix &first, const ShapeMatrix &second) {
    ShapeMatrix result{};
    for (std::uint32_t column = 0U; column < 3U; ++column) {
        const Vec3 weights = second.columns[column];
        result.columns[column] = add(
            multiply(first.columns[0], weights.x),
            add(multiply(first.columns[1], weights.y),
                multiply(first.columns[2], weights.z)));
    }
    return result;
}

// One deterministic thread computes the best-fit rigid transform of the rest
// lattice, then projects movable nodes toward it. The current center of mass
// and best-fit rotation keep this goal free to translate and roll; unlike a
// world-space tether it restores shape without pinning the body in place.
__global__ void soft_body_project_rest_shape(
    Vec3 *positions, const Vec3 *rest_positions,
    const float *inverse_masses, std::uint32_t count, float movable_mass,
    Vec3 rest_center, ShapeMatrix inverse_rest,
    Quaternion *stored_orientation, float stiffness,
    float maximum_projection, const std::uint32_t *dynamic_contact_flag) {
    if (blockIdx.x != 0U || threadIdx.x != 0U || stiffness <= 0.0F ||
        *dynamic_contact_flag != 0U) return;
    Vec3 current_center{};
    for (std::uint32_t node = 0U; node < count; ++node) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass > 0.0F)
            current_center = add(current_center,
                multiply(positions[node], 1.0F / inverse_mass));
    }
    current_center = multiply(current_center, 1.0F / movable_mass);

    ShapeMatrix covariance{};
    for (std::uint32_t node = 0U; node < count; ++node) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass <= 0.0F) continue;
        const float mass = 1.0F / inverse_mass;
        const Vec3 current = subtract(positions[node], current_center);
        const Vec3 rest = subtract(rest_positions[node], rest_center);
        covariance.columns[0] = add(covariance.columns[0],
            multiply(current, mass * rest.x));
        covariance.columns[1] = add(covariance.columns[1],
            multiply(current, mass * rest.y));
        covariance.columns[2] = add(covariance.columns[2],
            multiply(current, mass * rest.z));
    }
    const ShapeMatrix deformation = shape_matrix_multiply(
        covariance, inverse_rest);
    Quaternion orientation = normalized_quaternion(*stored_orientation);
    if (orientation.x == 0.0F && orientation.y == 0.0F &&
        orientation.z == 0.0F && orientation.w == 0.0F)
        orientation.w = 1.0F;
    for (std::uint32_t iteration = 0U; iteration < 12U; ++iteration) {
        const Vec3 axes[3]{
            rotate(orientation, {1.0F, 0.0F, 0.0F}),
            rotate(orientation, {0.0F, 1.0F, 0.0F}),
            rotate(orientation, {0.0F, 0.0F, 1.0F})};
        Vec3 angular = add(cross(axes[0], deformation.columns[0]),
            add(cross(axes[1], deformation.columns[1]),
                cross(axes[2], deformation.columns[2])));
        const float denominator = fabsf(
            dot(axes[0], deformation.columns[0]) +
            dot(axes[1], deformation.columns[1]) +
            dot(axes[2], deformation.columns[2])) + 1.0e-9F;
        angular = multiply(angular, 1.0F / denominator);
        const float magnitude = vector_length(angular);
        if (magnitude < 1.0e-6F) break;
        const float angle = fminf(magnitude, 0.5F);
        const float half = 0.5F * angle;
        const Vec3 axis = multiply(angular, 1.0F / magnitude);
        const Quaternion delta{axis.x * sinf(half), axis.y * sinf(half),
                               axis.z * sinf(half), cosf(half)};
        orientation = normalized_quaternion(
            quaternion_multiply(delta, orientation));
    }
    *stored_orientation = orientation;

    Vec3 weighted_correction{};
    for (std::uint32_t node = 0U; node < count; ++node) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass <= 0.0F) continue;
        const Vec3 target = add(current_center, rotate(orientation,
            subtract(rest_positions[node], rest_center)));
        const Vec3 correction = clamp_length(multiply(
            subtract(target, positions[node]), stiffness),
            maximum_projection);
        weighted_correction = add(weighted_correction,
            multiply(correction, 1.0F / inverse_mass));
    }
    const Vec3 center_correction = multiply(
        weighted_correction, 1.0F / movable_mass);
    for (std::uint32_t node = 0U; node < count; ++node) {
        if (inverse_masses[node] <= 0.0F) continue;
        const Vec3 target = add(current_center, rotate(orientation,
            subtract(rest_positions[node], rest_center)));
        const Vec3 correction = clamp_length(multiply(
            subtract(target, positions[node]), stiffness),
            maximum_projection);
        positions[node] = add(positions[node],
            subtract(correction, center_correction));
    }
}

__global__ void deformable_project_links(
    const Vec3 *positions, Vec3 *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const DeformableNeighbor *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float dt,
    float maximum_projection_fraction) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    const float self_mass = inverse_masses[vertex];
    const Vec3 position = positions[vertex];
    if (self_mass == 0.0F) {
        scratch[vertex] = position;
        return;
    }
    Vec3 correction{};
    float shortest_rest_length = FLT_MAX;
    const std::uint32_t first = offsets[vertex];
    const std::uint32_t last = offsets[vertex + 1U];
    for (std::uint32_t edge = first; edge < last; ++edge) {
        const DeformableNeighbor neighbor = neighbors[edge];
        if (bond_active != nullptr && neighbor.bond != k_invalid_dense &&
            bond_active[neighbor.bond] == 0U) continue;
        shortest_rest_length = fminf(shortest_rest_length,
                                     neighbor.rest_length);
        const Vec3 difference = subtract(position, positions[neighbor.index]);
        const float length = vector_length(difference);
        if (length < 1.0e-7F) continue;
        const float other_mass = inverse_masses[neighbor.index];
        const float denominator = self_mass + other_mass +
            neighbor.compliance / (dt * dt);
        const float amount = -self_mass * (length - neighbor.rest_length) /
            (denominator * length);
        correction = add(correction, multiply(difference, amount));
    }
    // Keep the rest-graph degree so a broken bond cannot make its surviving
    // neighbors abruptly stiffer and start a fracture cascade.
    const float divisor = static_cast<float>(max(1U, last - first));
    Vec3 proposal = multiply(correction, 1.0F / divisor);
    if (maximum_projection_fraction > 0.0F &&
        shortest_rest_length < FLT_MAX) {
        proposal = clamp_length(
            proposal, maximum_projection_fraction * shortest_rest_length);
    }
    scratch[vertex] = add(position, proposal);
}

// Closed-cloth volume is a global constraint. A single deterministic thread
// is preferable here to unordered floating-point atomics: authored pressure
// skins are small, while the work remains linear in vertices and triangles.
__global__ void cloth_project_volume(
    Vec3 *positions, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t vertex_count,
    std::uint32_t triangle_count, Vec3 *gradients, float target_volume,
    float orientation, float compliance, float dt, float *lambda,
    bool reset_lambda) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    if (reset_lambda) *lambda = 0.0F;
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
        gradients[vertex] = {};
    float signed_volume = 0.0F;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t a_index = indices[3U * triangle];
        const std::uint32_t b_index = indices[3U * triangle + 1U];
        const std::uint32_t c_index = indices[3U * triangle + 2U];
        const Vec3 a = positions[a_index];
        const Vec3 b = positions[b_index];
        const Vec3 c = positions[c_index];
        signed_volume += dot(a, cross(b, c)) / 6.0F;
        gradients[a_index] = add(
            gradients[a_index], multiply(cross(b, c), orientation / 6.0F));
        gradients[b_index] = add(
            gradients[b_index], multiply(cross(c, a), orientation / 6.0F));
        gradients[c_index] = add(
            gradients[c_index], multiply(cross(a, b), orientation / 6.0F));
    }
    float inverse_mass_sum = 0.0F;
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
        inverse_mass_sum += inverse_masses[vertex] *
            length_squared(gradients[vertex]);
    if (inverse_mass_sum <= 1.0e-12F) return;
    const float alpha = compliance / (dt * dt);
    const float constraint = orientation * signed_volume - target_volume;
    const float delta_lambda =
        (-constraint - alpha * *lambda) / (inverse_mass_sum + alpha);
    *lambda += delta_lambda;
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
        if (inverse_masses[vertex] == 0.0F) continue;
        positions[vertex] = add(positions[vertex], multiply(
            gradients[vertex], inverse_masses[vertex] * delta_lambda));
    }
}

__global__ void cloth_break_bonds(const Vec3 *positions,
    const ClothBond *bonds, std::uint8_t *active, std::uint8_t *damage,
    std::uint32_t count, float break_strain, std::uint32_t persistence,
    const FluidBodyImpulse *contact_impulses, float impact_threshold) {
    const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= count || active[edge] == 0U) return;
    const ClothBond bond = bonds[edge];
    if (impact_threshold > 0.0F && contact_impulses != nullptr &&
        vector_length(contact_impulses[bond.first].linear) +
        vector_length(contact_impulses[bond.second].linear) >
            impact_threshold) {
        active[edge] = 0U;
        damage[edge] = 0U;
        return;
    }
    if (break_strain <= 0.0F) return;
    const float length = vector_length(subtract(
        positions[bond.first], positions[bond.second]));
    if (!isfinite(length) || length <=
        bond.rest_length * (1.0F + break_strain)) {
        damage[edge] = 0U;
        return;
    }
    const std::uint32_t next = min(persistence,
        static_cast<std::uint32_t>(damage[edge]) + 1U);
    damage[edge] = static_cast<std::uint8_t>(next);
    if (next >= persistence) {
        active[edge] = 0U;
        damage[edge] = 0U;
    }
}

// Fracture releases seams, not the material within a face. Bound isolated
// triangles in physical space during contact, rather than fitting a render
// triangle to remote nodes. Do not erase attached material's tearing strain.
// One cooperative block performs
// deterministic Jacobi iterations without shared-vertex writes or host waits.
__global__ void cloth_limit_strain(Vec3 *positions, Vec3 *scratch,
    const float *inverse_masses, const std::uint32_t *offsets,
    const DeformableNeighbor *neighbors, const std::uint8_t *free_nodes,
    std::uint32_t count) {
    for (std::uint32_t pass = 0; pass < 32U; ++pass) {
        bool changed = false;
        for (std::uint32_t node = threadIdx.x; node < count; node += blockDim.x) {
            Vec3 correction{};
            std::uint32_t degree = 0U;
            for (auto item = offsets[node]; free_nodes[node] && item < offsets[node + 1U]; ++item) {
                const auto edge = neighbors[item];
                if (edge.bond != k_invalid_dense) continue;
                const float weight = inverse_masses[node] + inverse_masses[edge.index];
                const Vec3 delta = subtract(positions[edge.index], positions[node]);
                const float length = vector_length(delta);
                const float maximum = edge.rest_length * 1.10F;
                if (weight <= 0.0F || length <= maximum || inverse_masses[node] == 0.0F) continue;
                correction = add(correction, multiply(delta,
                    (length - maximum) * inverse_masses[node] / (length * weight)));
                changed |= length > maximum * 1.0001F;
                ++degree;
            }
            scratch[node] = add(positions[node], multiply(correction, 1.0F / max(1U,degree)));
        }
        __syncthreads();
        for (std::uint32_t node = threadIdx.x; node < count; node += blockDim.x)
            positions[node] = scratch[node];
        if (!__syncthreads_or(changed)) break;
    }
}

__global__ void cloth_update_surface(const Vec3 *positions,
    const std::uint32_t *source_indices,
    Vec3 *surface_positions, std::uint32_t triangle_count) {
    const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= triangle_count) return;
    const std::uint32_t base = 3U * triangle;
    for (std::uint32_t corner = 0U; corner < 3U; ++corner)
        surface_positions[base + corner] = positions[source_indices[base + corner]];
}

template <bool couple_dynamic, bool position_only = false>
__global__ void deformable_collide(
    Vec3 *positions, Vec3 *velocities, const Vec3 *previous,
    const float *inverse_masses, std::uint32_t count, float thickness,
    float dt, const BodyParameters *parameters,
    const RigidBodyState *previous_states, const RigidBodyState *states,
    const TriangleMeshResource *meshes,
    std::uint32_t body_count, FluidBodyImpulse *impulses,
    Vec3 *contact_forces, Vec3 *contact_normals, Vec3 *contact_arms,
    Vec3 *contact_momentum_delta,
    float *accumulated_normal_delta,
    std::uint32_t *deformable_contact_count,
    std::uint32_t *dynamic_contact_flag,
    std::uint32_t *contact_flags, bool static_only,
    bool reconstruct_free_velocities, float maximum_speed,
    Vec3 *body_position_corrections = nullptr) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    if constexpr (!position_only) {
        if (body_position_corrections != nullptr)
            body_position_corrections[vertex] = {};
        impulses[vertex] = {};
        contact_forces[vertex] = {};
        if (contact_normals != nullptr) contact_normals[vertex] = {};
        if constexpr (couple_dynamic) contact_arms[vertex] = {};
    }
    if (inverse_masses[vertex] == 0.0F) return;
    const Vec3 start = previous[vertex];
    Vec3 end = positions[vertex];
    float best_penetration = 0.0F;
    Vec3 best_normal{}, best_contact{};
    std::uint32_t best_body = k_invalid_dense;
    for (std::uint32_t body_index = 0U; body_index < body_count; ++body_index) {
        const BodyParameters body = parameters[body_index];
        if (static_only && body.motion != MotionType::static_body) continue;
        const RigidBodyState state = states[body_index];
        const RigidBodyState previous_state = previous_states != nullptr
            ? previous_states[body_index] : state;
        const TriangleMeshResource mesh = meshes[body.mesh.index];
        if (mesh.bvh_node_count == 0U) continue;
        // Sweep in collider-local space at both ends. Reusing the current body
        // transform for the start sample lets a moving closed body engulf a
        // node without the relative segment ever crossing its surface.
        const Vec3 local_start = inverse_rotate(previous_state.orientation,
            subtract(start, previous_state.position));
        const Vec3 local_end = inverse_rotate(state.orientation,
            subtract(end, state.position));
        if (!fluid_segment_bounds(local_start, local_end, mesh.bvh_nodes[0],
                                  thickness + body.collision_margin)) continue;
        if constexpr (couple_dynamic) {
            if (mesh.solid_planes != nullptr) {
                float nearest_side = -FLT_MAX;
                CollisionPlane nearest_plane{};
                for (std::uint32_t triangle = 0U;
                     triangle < mesh.index_count / 3U; ++triangle) {
                    const CollisionPlane plane = mesh.solid_planes[triangle];
                    const float side = dot(plane.normal, local_end) - plane.offset;
                    if (side > nearest_side) {
                        nearest_side = side;
                        nearest_plane = plane;
                    }
                    if (side > 0.0F) break;
                }
                if (nearest_side <= 0.0F) {
                    const float penetration = thickness +
                        body.collision_margin - nearest_side;
                    if (penetration > best_penetration) {
                        best_penetration = penetration;
                        best_normal = rotate(state.orientation,
                                              nearest_plane.normal);
                        best_contact = transform_point(state, subtract(local_end,
                            multiply(nearest_plane.normal, nearest_side)));
                        best_body = body_index;
                    }
                    // A solid's interior cannot choose the inward normal of
                    // its closest triangle, even after an earlier contact or
                    // spring projection left the node behind that triangle.
                    continue;
                }
            }
        }
        std::uint32_t stack[64]{};
        int pending = 1;
        while (pending != 0) {
            const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
            if (!fluid_segment_bounds(local_start, local_end, node,
                                      thickness + body.collision_margin)) continue;
            if (node.triangle_count == 0U) {
                if (pending + 2 > 64) continue;
                stack[pending++] = node.right;
                stack[pending++] = node.left;
                continue;
            }
            for (std::uint32_t item = 0U; item < node.triangle_count; ++item) {
                const std::uint32_t base = (node.first_triangle + item) * 3U;
                const Vec3 a = mesh.vertices[mesh.indices[base]];
                const Vec3 b = mesh.vertices[mesh.indices[base + 1U]];
                const Vec3 c = mesh.vertices[mesh.indices[base + 2U]];
                const Vec3 nearest = fluid_closest_triangle(local_end, a, b, c);
                const Vec3 delta = subtract(local_end, nearest);
                const float distance = vector_length(delta);
                const bool solid = couple_dynamic && mesh.solid_planes != nullptr;
                const Vec3 face = solid ? mesh.solid_planes[base / 3U].normal
                    : normalized_or(cross(subtract(b, a),
                        subtract(c, a)), {0.0F, 1.0F, 0.0F});
                Vec3 normal = distance > 1.0e-6F
                    ? multiply(delta, 1.0F / distance)
                    : multiply(face, dot(subtract(local_start, a), face) >= 0.0F
                                         ? 1.0F : -1.0F);
                float penetration = thickness + body.collision_margin - distance;
                const float before = dot(subtract(local_start, a), face);
                const float after = dot(subtract(local_end, a), face);
                if (before * after < 0.0F && (!solid || before > 0.0F)) {
                    const float fraction = before / (before - after);
                    const Vec3 crossing = add(local_start,
                        multiply(subtract(local_end, local_start), fraction));
                    if (length_squared(subtract(fluid_closest_triangle(
                            crossing, a, b, c), crossing)) <
                            thickness * thickness) {
                        normal = multiply(face, before > 0.0F ? 1.0F : -1.0F);
                        penetration = fmaxf(penetration,
                            thickness + body.collision_margin + fabsf(after));
                    }
                }
                if (penetration > best_penetration) {
                    best_penetration = penetration;
                    best_normal = rotate(state.orientation, normal);
                    best_contact = transform_point(state, nearest);
                    best_body = body_index;
                }
            }
        }
    }
    if (best_body != k_invalid_dense) {
        if constexpr (position_only) {
            positions[vertex] = add(end, multiply(best_normal, best_penetration));
            return;
        }
        const BodyParameters body = parameters[best_body];
        const RigidBodyState state = states[best_body];
        if constexpr (couple_dynamic) {
            if (body.motion == MotionType::dynamic)
                atomicExch(dynamic_contact_flag, 1U);
        }
        if (contact_normals != nullptr) contact_normals[vertex] = best_normal;
        if (accumulated_normal_delta != nullptr)
            accumulated_normal_delta[vertex] += best_penetration;
        if (deformable_contact_count != nullptr)
            atomicAdd(deformable_contact_count, 1U);
        Vec3 velocity = multiply(subtract(end, start), 1.0F / dt);
        float node_share = 1.0F;
        if constexpr (couple_dynamic)
            node_share = inverse_masses[vertex] /
                         (inverse_masses[vertex] + body.inverse_mass);
        end = add(end, multiply(best_normal, best_penetration * node_share));
        const Vec3 arm = subtract(best_contact, state.position);
        if constexpr (couple_dynamic) {
            impulses[vertex].body = best_body;
            contact_arms[vertex] = arm;
            atomicExch(contact_flags + best_body, 1U);
        }
        const Vec3 body_velocity = add(state.linear_velocity,
            cross(state.angular_velocity, arm));
        const Vec3 relative = subtract(velocity, body_velocity);
        const float incoming = dot(relative, best_normal);
        if (incoming < 0.0F) {
            const float mass = 1.0F / fmaxf(inverse_masses[vertex], 1.0e-6F);
            const Vec3 arm_cross = cross(arm, best_normal);
            const float denominator = 1.0F / mass + body.inverse_mass +
                dot(cross(inverse_inertia_world(body, state, arm_cross), arm),
                    best_normal);
            if (denominator > k_epsilon) {
                const Vec3 impulse = multiply(best_normal,
                    -(1.0F + body.restitution) * incoming / denominator);
                const Vec3 velocity_delta = multiply(impulse, 1.0F / mass);
                velocity = add(velocity, velocity_delta);
                if constexpr (couple_dynamic)
                    contact_momentum_delta[vertex] = add(
                        contact_momentum_delta[vertex], impulse);
                // Soft-body velocity is reconstructed from its projected
                // position. Encode dynamic-body impulses in that position so
                // the node retains the same impulse whose opposite is applied
                // to the rigid body after this contact pass.
                if constexpr (couple_dynamic) {
                    if (body.motion == MotionType::dynamic)
                        end = add(end, multiply(velocity_delta, dt));
                }
                impulses[vertex] = {multiply(impulse, -1.0F),
                    multiply(cross(arm, impulse), -1.0F), best_body};
                contact_forces[vertex] = multiply(impulse, 1.0F / dt);
                if constexpr (!couple_dynamic)
                    atomicExch(contact_flags + best_body, 1U);
            }
        }
        if constexpr (couple_dynamic)
            if (body_position_corrections != nullptr)
                body_position_corrections[vertex] = multiply(best_normal,
                    -best_penetration * body.inverse_mass /
                    (inverse_masses[vertex] + body.inverse_mass));
        positions[vertex] = end;
        velocities[vertex] = clamp_length(velocity, maximum_speed);
    } else if (reconstruct_free_velocities) {
        velocities[vertex] = clamp_length(
            multiply(subtract(end, start), 1.0F / dt), maximum_speed);
    }
}

__global__ void soft_body_finalize_velocities(
    const Vec3 *positions, const Vec3 *previous, Vec3 *velocities,
    const float *inverse_masses, std::uint32_t count, float inverse_dt,
    float projection_response, float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if (inverse_masses[node] == 0.0F) {
        velocities[node] = {};
        return;
    }
    const Vec3 projected_velocity = multiply(
        subtract(positions[node], previous[node]), inverse_dt);
    Vec3 velocity = add(velocities[node], multiply(
        subtract(projected_velocity, velocities[node]), projection_response));
    velocities[node] = clamp_length(velocity, maximum_speed);
}

__global__ void soft_body_apply_contact_friction(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    Vec3 *contact_forces, const Vec3 *contact_normals,
    const Vec3 *contact_arms, FluidBodyImpulse *body_impulses,
    const BodyParameters *body_parameters,
    Vec3 *contact_momentum_delta,
    Vec3 *accumulated_friction_delta,
    const float *accumulated_normal_delta,
    const std::uint32_t *contact_count, std::uint32_t count,
    float movable_mass, Vec3 gravity, float dt, float contact_friction,
    float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count || inverse_masses[node] == 0.0F ||
        contact_friction <= 0.0F) return;
    const std::uint32_t supported_nodes = *contact_count;
    if (supported_nodes == 0U) return;
    const float normal_squared = length_squared(contact_normals[node]);
    if (normal_squared <= 1.0e-10F) return;
    const std::uint32_t body_index = body_impulses[node].body;
    if (body_index == k_invalid_dense) return;
    const Vec3 normal = multiply(
        contact_normals[node], rsqrtf(normal_squared));
    const Vec3 original_velocity = velocities[node];
    Vec3 velocity = original_velocity;
    const Vec3 force = contact_forces[node];
    const float force_squared = length_squared(force);
    const Vec3 normal_velocity = multiply(normal, dot(velocity, normal));
    const Vec3 tangent = subtract(velocity, normal_velocity);
    const float impulse_acceleration = force_squared > 1.0e-10F
        ? sqrtf(force_squared) * inverse_masses[node] : 0.0F;
    // Contacts are generated before the graph solve, so a surface node's
    // direct impulse contains only its own predicted weight. Distribute the
    // whole movable mass across the active support nodes; otherwise a dense
    // lattice slides because its interior load never reaches friction.
    const float supported_acceleration = fabsf(dot(gravity, normal)) *
        movable_mass * inverse_masses[node] /
        static_cast<float>(supported_nodes);
    const float normal_acceleration =
        fmaxf(impulse_acceleration, supported_acceleration);
    const float constraint_delta = accumulated_normal_delta != nullptr
        ? accumulated_normal_delta[node] / dt : 0.0F;
    const BodyParameters body = body_parameters[body_index];
    const float friction = body.motion == MotionType::dynamic
        ? sqrtf(contact_friction * body.friction)
        : contact_friction;
    const float maximum_delta = friction *
        fmaxf(normal_acceleration * dt, constraint_delta);
    Vec3 accumulated = accumulated_friction_delta[node];
    accumulated = subtract(accumulated,
        multiply(normal, dot(accumulated, normal)));
    Vec3 next_accumulated = subtract(accumulated, tangent);
    next_accumulated = clamp_length(next_accumulated, maximum_delta);
    const Vec3 applied_delta = subtract(next_accumulated, accumulated);
    velocity = add(velocity, applied_delta);
    velocity = clamp_length(velocity, maximum_speed);
    positions[node] = add(positions[node], multiply(
        subtract(velocity, original_velocity), dt));
    velocities[node] = velocity;
    accumulated_friction_delta[node] = next_accumulated;
    const float mass = 1.0F / inverse_masses[node];
    const Vec3 node_impulse = multiply(applied_delta, mass);
    contact_momentum_delta[node] = add(
        contact_momentum_delta[node], node_impulse);
    contact_forces[node] = add(
        contact_forces[node], multiply(node_impulse, 1.0F / dt));
    const Vec3 reaction = multiply(node_impulse, -1.0F);
    body_impulses[node].linear = add(
        body_impulses[node].linear, reaction);
    body_impulses[node].angular = add(
        body_impulses[node].angular,
        cross(contact_arms[node], reaction));
}

__global__ void soft_body_damp_springs(
    const Vec3 *positions, const Vec3 *velocities, Vec3 *output,
    const float *inverse_masses, const std::uint32_t *offsets,
    const DeformableNeighbor *neighbors, const std::uint8_t *bond_active,
    std::uint32_t count, float damping, float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if (inverse_masses[node] == 0.0F) {
        output[node] = {};
        return;
    }
    Vec3 correction{};
    std::uint32_t active_count = 0U;
    for (std::uint32_t item = offsets[node]; item < offsets[node + 1U]; ++item) {
        const DeformableNeighbor neighbor = neighbors[item];
        if (bond_active[neighbor.bond] == 0U) continue;
        const Vec3 axis = normalized_or(
            subtract(positions[neighbor.index], positions[node]), {});
        correction = add(correction, multiply(axis,
            dot(subtract(velocities[neighbor.index], velocities[node]), axis)));
        ++active_count;
    }
    if (active_count != 0U)
        correction = multiply(correction,
            0.5F * damping / static_cast<float>(active_count));
    output[node] = clamp_length(add(velocities[node], correction), maximum_speed);
}

__global__ void soft_body_update_surface(
    const Vec3 *positions, const Vec3 *rest_positions,
    const Vec3 *surface_rest_positions,
    const SoftBodySurfaceBinding *bindings, Vec3 *surface_positions,
    std::uint32_t surface_vertex_count) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= surface_vertex_count) return;
    const SoftBodySurfaceBinding binding = bindings[vertex];
    Vec3 delta{};
    for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
        if (binding.weights[slot] == 0.0F) continue;
        delta = add(delta, multiply(subtract(positions[binding.nodes[slot]],
                                             rest_positions[binding.nodes[slot]]),
                                    binding.weights[slot]));
    }
    surface_positions[vertex] = add(surface_rest_positions[vertex], delta);
}

// Detection reads stable position buffers. Integer contact counts provide a
// shared relaxation for both sides; the later gathers use no float atomics.
#include "fluid_soft_body.cuh"

__global__ void soft_cloth_detect(
    const Vec3 *positions, const Vec3 *previous, const Vec3 *velocities,
    const float *inverse_masses, std::uint32_t count,
    const Vec3 *cloth_positions, const Vec3 *cloth_previous,
    const Vec3 *cloth_velocities, const float *cloth_inverse_masses,
    const std::uint32_t *indices, const Vec3 *surface,
    std::uint32_t triangle_count, float distance, float friction,
    SoftClothContact *contacts, std::uint32_t *contact_counts) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    contacts[node] = {};
    if (inverse_masses[node] == 0.0F) return;
    const Vec3 point = positions[node], start = previous[node];
    float best_squared = FLT_MAX;
    Vec3 best_normal{}, best_weights{}, best_point{};
    std::uint32_t best = k_invalid_dense;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t base = triangle * 3U;
        const Vec3 a = surface ? surface[base] : cloth_positions[indices[base]];
        const Vec3 b = surface ? surface[base + 1U] : cloth_positions[indices[base + 1U]];
        const Vec3 c = surface ? surface[base + 2U] : cloth_positions[indices[base + 2U]];
        const Vec3 old_a = cloth_previous[indices[base]];
        const Vec3 old_b = cloth_previous[indices[base + 1U]];
        const Vec3 old_c = cloth_previous[indices[base + 2U]];
        const Vec3 expansion{distance, distance, distance};
        if (!bounds_overlap(component_min(point, start), component_max(point, start),
            subtract(component_min(component_min(a, component_min(b, c)),
                component_min(old_a, component_min(old_b, old_c))), expansion),
            add(component_max(component_max(a, component_max(b, c)),
                component_max(old_a, component_max(old_b, old_c))), expansion))) continue;
        const Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) < 1.0e-14F) continue;
        const Vec3 normal = normalized_or(face, {});
        Vec3 weights{};
        const Vec3 nearest = fluid_closest_triangle_barycentric(point, a, b, c, weights);
        const Vec3 delta = subtract(point, nearest);
        const float squared = length_squared(delta);
        if (squared >= best_squared) continue;
        const Vec3 old_nearest = add(multiply(cloth_previous[indices[base]], weights.x),
            add(multiply(cloth_previous[indices[base + 1U]], weights.y),
                multiply(cloth_previous[indices[base + 2U]], weights.z)));
        const float before = dot(subtract(start, old_nearest), normal);
        const float after = dot(delta, normal);
        const float travel = vector_length(subtract(point, start)) +
                             vector_length(subtract(nearest, old_nearest));
        const bool crossed = before * after < 0.0F &&
            squared <= (travel + distance) * (travel + distance);
        if (!crossed && squared >= distance * distance) continue;
        best = triangle;
        best_squared = squared;
        best_point = nearest;
        best_weights = weights;
        best_normal = crossed || squared < 1.0e-14F
            ? multiply(normal, before >= 0.0F ? 1.0F : -1.0F)
            : multiply(delta, rsqrtf(squared));
    }
    if (best == k_invalid_dense) return;
    const float penetration = distance - dot(subtract(point, best_point), best_normal);
    if (penetration <= 0.0F) return;
    SoftClothContact contact{};
    contact.weights[0] = best_weights.x;
    contact.weights[1] = best_weights.y;
    contact.weights[2] = best_weights.z;
    float denominator = inverse_masses[node];
    Vec3 cloth_velocity{};
    for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
        const auto vertex = indices[3U * best + corner];
        const float weight = contact.weights[corner];
        contact.vertices[corner] = vertex;
        denominator += weight * weight * cloth_inverse_masses[vertex];
        cloth_velocity = add(cloth_velocity, multiply(cloth_velocities[vertex], weight));
        if (weight > 1.0e-6F && cloth_inverse_masses[vertex] > 0.0F)
            atomicAdd(contact_counts + vertex, 1U);
    }
    contact.position_impulse = multiply(best_normal,
        fminf(penetration, 2.0F * distance) / denominator);
    contact.soft_inverse_mass_fraction = inverse_masses[node] / denominator;
    const Vec3 relative = subtract(velocities[node], cloth_velocity);
    const float normal_speed = dot(relative, best_normal);
    const float normal_impulse = fmaxf(0.0F, -normal_speed) / denominator;
    const Vec3 tangent = subtract(relative, multiply(best_normal, normal_speed));
    contact.velocity_impulse = subtract(multiply(best_normal, normal_impulse),
        clamp_length(multiply(tangent, 1.0F / denominator), friction * normal_impulse));
    contact.active = true;
    contacts[node] = contact;
}

__device__ float soft_cloth_relaxation(
    const SoftClothContact &contact, const std::uint32_t *counts) {
    std::uint32_t degree = 1U;
    for (std::uint32_t corner = 0U; corner < 3U; ++corner)
        if (contact.weights[corner] > 1.0e-6F)
            degree = max(degree, counts[contact.vertices[corner]]);
    // Only cloth vertices are shared by several contacts. Multiplying the
    // soft node's independent inverse mass by that degree over-damps recovery
    // and allows a fast sheet to pass through it during impact.
    const float soft_fraction = contact.soft_inverse_mass_fraction;
    return 1.0F / (soft_fraction + static_cast<float>(degree) * (1.0F - soft_fraction));
}

__global__ void soft_cloth_apply_soft(
    Vec3 *positions, Vec3 *velocities, Vec3 *forces,
    const float *inverse_masses, std::uint32_t count,
    const SoftClothContact *contacts, const std::uint32_t *counts,
    float dt, float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count || !contacts[node].active) return;
    const SoftClothContact contact = contacts[node];
    const float relaxation = soft_cloth_relaxation(contact, counts);
    positions[node] = add(positions[node],
        multiply(contact.position_impulse, relaxation * inverse_masses[node]));
    const Vec3 impulse = multiply(contact.velocity_impulse, relaxation);
    velocities[node] = clamp_length(add(velocities[node],
        multiply(impulse, inverse_masses[node])), maximum_speed);
    forces[node] = add(forces[node], multiply(impulse, 1.0F / dt));
}

__global__ void soft_cloth_apply_cloth(
    Vec3 *positions, Vec3 *velocities, Vec3 *forces,
    const float *inverse_masses, std::uint32_t count,
    const SoftClothContact *contacts, std::uint32_t contact_count,
    const std::uint32_t *counts, float dt) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    Vec3 position_impulse{}, velocity_impulse{};
    for (std::uint32_t node = 0U; node < contact_count; ++node) {
        const SoftClothContact contact = contacts[node];
        if (!contact.active) continue;
        for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
            if (contact.vertices[corner] != vertex) continue;
            const float scale = -contact.weights[corner] *
                                soft_cloth_relaxation(contact, counts);
            position_impulse = add(position_impulse, multiply(contact.position_impulse, scale));
            velocity_impulse = add(velocity_impulse, multiply(contact.velocity_impulse, scale));
        }
    }
    positions[vertex] = add(positions[vertex], multiply(position_impulse, inverse_masses[vertex]));
    velocities[vertex] = clamp_length(add(velocities[vertex],
        multiply(velocity_impulse, inverse_masses[vertex])), 20.0F);
    forces[vertex] = add(forces[vertex], multiply(velocity_impulse, 1.0F / dt));
}

// Test the actual skin triangles against verified solid triangle meshes.
// Node contacts can leave a face cutting through a collider between its nodes.
// Select the least-displacing supporting plane, then constrain every corner
// to its outside half-space. The whole face is outside once all corners are.
__global__ void soft_body_surface_contacts(
    const Vec3 *surface, const std::uint32_t *indices,
    std::uint32_t triangle_count, const BodyParameters *parameters,
    const RigidBodyState *states, const TriangleMeshResource *meshes,
    std::uint32_t body_count, Vec3 *corrections) {
    const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= triangle_count) return;
    const std::uint32_t base = triangle * 3U;
    const Vec3 world[3]{surface[indices[base]], surface[indices[base + 1U]],
                        surface[indices[base + 2U]]};
    for (std::uint32_t corner = 0U; corner < 3U; ++corner)
        corrections[base + corner] = {};
    float deepest = 0.0F;
    for (std::uint32_t body = 0U; body < body_count; ++body) {
        const TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
        if (mesh.solid_planes == nullptr) continue;
        const RigidBodyState state = states[body];
        Vec3 p[3];
        for (std::uint32_t corner = 0U; corner < 3U; ++corner)
            p[corner] = inverse_rotate(state.orientation,
                                      subtract(world[corner], state.position));
        const float margin = parameters[body].collision_margin;
        const Vec3 expansion{margin, margin, margin};
        if (!bounds_overlap(subtract(component_min(p[0], component_min(p[1], p[2])), expansion),
                            add(component_max(p[0], component_max(p[1], p[2])), expansion),
                            mesh.minimum, mesh.maximum)) continue;
        float separation = -FLT_MAX;
        CollisionPlane support{};
        for (std::uint32_t face = 0U; face < mesh.index_count / 3U; ++face) {
            const CollisionPlane plane = mesh.solid_planes[face];
            const float minimum_side = fminf(dot(plane.normal, p[0]),
                fminf(dot(plane.normal, p[1]), dot(plane.normal, p[2]))) -
                plane.offset;
            if (minimum_side > separation) {
                separation = minimum_side;
                support = plane;
            }
            if (separation >= margin) break;
        }
        if (separation >= margin || margin - separation <= deepest) continue;
        // Face planes alone are conservative near edges. Require actual
        // triangle proximity, or a vertex inside the closed solid.
        bool contact = false;
        for (std::uint32_t corner = 0U; corner < 3U && !contact; ++corner) {
            bool inside = true;
            for (std::uint32_t face = 0U; face < mesh.index_count / 3U; ++face) {
                const CollisionPlane plane = mesh.solid_planes[face];
                if (dot(plane.normal, p[corner]) > plane.offset) {
                    inside = false;
                    break;
                }
            }
            contact = inside;
        }
        for (std::uint32_t face = 0U; face < mesh.index_count && !contact; face += 3U) {
            const Vec3 a = mesh.vertices[mesh.indices[face]];
            const Vec3 b = mesh.vertices[mesh.indices[face + 1U]];
            const Vec3 c = mesh.vertices[mesh.indices[face + 2U]];
            if (!triangle_bounds_overlap(p[0], p[1], p[2], a, b, c, margin)) continue;
            Vec3 on_soft{}, on_rigid{};
            closest_triangle_pair(p[0], p[1], p[2], a, b, c, on_soft, on_rigid);
            contact = length_squared(subtract(on_soft, on_rigid)) < margin * margin;
        }
        if (!contact) continue;
        deepest = margin - separation;
        const Vec3 normal = rotate(state.orientation, support.normal);
        for (std::uint32_t corner = 0U; corner < 3U; ++corner)
            corrections[base + corner] = multiply(normal, fmaxf(0.0F,
                margin + support.offset - dot(support.normal, p[corner])));
    }
}

__global__ void soft_body_apply_surface_contacts(
    Vec3 *positions, const Vec3 *corrections, const std::uint32_t *offsets,
    const SoftSurfaceInfluence *influences, std::uint32_t node_count) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) return;
    Vec3 correction{};
    for (std::uint32_t entry = offsets[node]; entry < offsets[node + 1U]; ++entry) {
        const SoftSurfaceInfluence influence = influences[entry];
        const Vec3 delta = multiply(corrections[influence.corner], influence.factor);
        const float squared = length_squared(delta);
        if (squared > 1.0e-16F) {
            // Satisfy adjacent face constraints together. Keeping only the
            // longest correction can discard a different contact normal.
            const float remaining = fmaxf(0.0F,
                1.0F - dot(correction, delta) / squared);
            correction = add(correction, multiply(delta, remaining));
        }
    }
    positions[node] = add(positions[node], correction);
}

// The vertex-side contact above deforms the sheet and transfers momentum.
// A second, triangle-side constraint keeps a fast rigid collider from slipping
// between intact, nonfracturing cloth vertices. The broad-phase sphere is
// conservative for any mesh; tearable cloth uses node contacts instead.
__device__ __noinline__ void stamp_rigid_cloth_paint(
    RigidBodyId rigid_id, ClothId cloth_id, Vec3 center, Vec3 contact,
    const Vec3 *positions, const std::uint32_t *indices,
    const std::uint32_t *vertex_sources,
    std::uint32_t triangle_count, const PaintFieldResource *fields,
    std::uint32_t field_capacity, const PaintRuleResource *rules,
    std::uint32_t rule_capacity) {
    for (std::uint32_t index = 0U; index < rule_capacity; ++index) {
        const PaintRuleResource rule = rules[index];
        if (!rule.alive || !rule.options.enabled ||
            rule.options.rigid_source.index != rigid_id.index ||
            rule.options.rigid_source.generation != rigid_id.generation ||
            rule.options.target.index >= field_capacity) continue;
        const PaintFieldResource field = fields[rule.options.target.index];
        if (!field.alive || field.generation != rule.options.target.generation ||
            field.options.cloth.index != cloth_id.index ||
            field.options.cloth.generation != cloth_id.generation) continue;
        const int width = static_cast<int>(field.options.width);
        const int height = static_cast<int>(field.options.height);
        const float radius_squared = rule.options.brush_radius *
                                     rule.options.brush_radius;
        for (std::uint32_t triangle = 0U; triangle < triangle_count;
             ++triangle) {
            const std::uint32_t base = triangle * 3U;
            const std::uint32_t ia = indices[base];
            const std::uint32_t ib = indices[base + 1U];
            const std::uint32_t ic = indices[base + 2U];
            if (ia == ib) continue;
            const Vec3 a = positions[ia], b = positions[ib], c = positions[ic];
            if (length_squared(subtract(contact,
                fluid_closest_triangle(contact, a, b, c))) > radius_squared)
                continue;
            const Vec2 ua = field.uvs[vertex_sources ? vertex_sources[ia] : ia],
                       ub = field.uvs[vertex_sources ? vertex_sources[ib] : ib],
                       uc = field.uvs[vertex_sources ? vertex_sources[ic] : ic];
            const float e0x = ub.x - ua.x, e0y = ub.y - ua.y;
            const float e1x = uc.x - ua.x, e1y = uc.y - ua.y;
            const float determinant = e0x * e1y - e0y * e1x;
            if (fabsf(determinant) < 1.0e-10F) continue;
            const float min_u = fminf(ua.x, fminf(ub.x, uc.x));
            const float max_u = fmaxf(ua.x, fmaxf(ub.x, uc.x));
            const float min_v = fminf(ua.y, fminf(ub.y, uc.y));
            const float max_v = fmaxf(ua.y, fmaxf(ub.y, uc.y));
            const int x0 = max(0, static_cast<int>(floorf(min_u * width)));
            const int x1 = min(width - 1,
                static_cast<int>(floorf(max_u * width)));
            const int y0 = max(0, static_cast<int>(floorf(min_v * height)));
            const int y1 = min(height - 1,
                static_cast<int>(floorf(max_v * height)));
            if (x0 > x1 || y0 > y1) continue;
            const std::uint32_t side = dot(cross(subtract(b, a),
                subtract(c, a)), subtract(center, contact)) >= 0.0F
                ? 1U : 2U;
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const float qx = (static_cast<float>(x) + 0.5F) /
                                     width - ua.x;
                    const float qy = (static_cast<float>(y) + 0.5F) /
                                     height - ua.y;
                    const float v = (qx * e1y - qy * e1x) / determinant;
                    const float w = (e0x * qy - e0y * qx) / determinant;
                    const float u = 1.0F - v - w;
                    if (u < -1.0e-4F || v < -1.0e-4F || w < -1.0e-4F)
                        continue;
                    const Vec3 point = add(multiply(a, u),
                        add(multiply(b, v), multiply(c, w)));
                    if (length_squared(subtract(point, contact)) <=
                        radius_squared)
                        atomicOr(field.pixels + y * width + x, side);
                }
            }
        }
    }
}

__global__ void cloth_constrain_bodies(
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *cloth_indices, const std::uint32_t *vertex_sources,
    const Vec3 *surface_positions,
    const float *cloth_inverse_masses, std::uint32_t vertex_count,
    std::uint32_t triangle_count, float thickness, float dt,
    float contact_friction,
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    RigidBodyState *states, const TriangleMeshResource *meshes,
    const RigidBodyId *body_ids, std::uint32_t body_count,
    ClothId cloth_id, const PaintFieldResource *paint_fields,
    std::uint32_t field_capacity, const PaintRuleResource *paint_rules,
    std::uint32_t rule_capacity, ClothBodyCorrection *corrections) {
    const std::uint32_t body_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (body_index >= body_count) return;
    corrections[body_index] = {};
    if (parameters[body_index].motion != MotionType::dynamic) return;
    const BodyParameters body = parameters[body_index];
    RigidBodyState state = states[body_index];
    const TriangleMeshResource mesh = meshes[body.mesh.index];
    const Vec3 center = transform_point(state, mesh.bounding_center);
    const Vec3 previous_center = transform_point(previous_states[body_index],
                                                  mesh.bounding_center);
    const float radius = mesh.bounding_radius + body.collision_margin + thickness;
    float best_penetration = 0.0F;
    Vec3 best_normal{}, best_contact{};
    std::uint32_t best_triangle = 0U;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t base = triangle * 3U;
        const Vec3 a = surface_positions != nullptr
            ? surface_positions[base] : cloth_positions[cloth_indices[base]];
        const Vec3 b = surface_positions != nullptr
            ? surface_positions[base + 1U]
            : cloth_positions[cloth_indices[base + 1U]];
        const Vec3 c = surface_positions != nullptr
            ? surface_positions[base + 2U]
            : cloth_positions[cloth_indices[base + 2U]];
        const Vec3 ab = subtract(b, a), ac = subtract(c, a);
        const Vec3 face = cross(ab, ac);
        if (length_squared(face) < 1.0e-12F) continue;
        const Vec3 normal = normalized_or(face, {0.0F, 0.0F, 1.0F});
        const Vec3 nearest = fluid_closest_triangle(center, a, b, c);
        const Vec3 delta = subtract(center, nearest);
        const float distance = vector_length(delta);
        Vec3 contact_normal = distance > 1.0e-6F
            ? multiply(delta, 1.0F / distance)
            : multiply(normal, dot(subtract(previous_center, a), normal) >= 0.0F
                                   ? 1.0F : -1.0F);
        float penetration = radius - distance;
        const float before = dot(subtract(previous_center, a), normal);
        const float after = dot(subtract(center, a), normal);
        if (before * after < 0.0F) {
            const float fraction = before / (before - after);
            const Vec3 crossing = add(previous_center,
                multiply(subtract(center, previous_center), fraction));
            if (length_squared(subtract(fluid_closest_triangle(
                    crossing, a, b, c), crossing)) < radius * radius) {
                contact_normal = multiply(normal, before > 0.0F ? 1.0F : -1.0F);
                penetration = fmaxf(penetration, radius + fabsf(after));
            }
        } else if (penetration > 0.0F && before * after > 0.0F) {
            contact_normal = multiply(normal, before > 0.0F ? 1.0F : -1.0F);
        }
        if (penetration > best_penetration) {
            best_penetration = penetration;
            best_normal = contact_normal;
            best_contact = nearest;
            best_triangle = triangle;
        }
    }
    if (best_penetration <= 0.0F) return;
    const std::uint32_t first = best_triangle * 3U;
    const std::uint32_t a = cloth_indices[first];
    const std::uint32_t b = cloth_indices[first + 1U];
    const std::uint32_t c = cloth_indices[first + 2U];
    if (rule_capacity != 0U)
        stamp_rigid_cloth_paint(body_ids[body_index], cloth_id, center,
            best_contact, cloth_positions, cloth_indices, vertex_sources, triangle_count,
            paint_fields, field_capacity, paint_rules, rule_capacity);
    // Fracturing cloth follows the node-contact response: the conservative
    // triangle-radius constraint otherwise cancels the body's incoming speed
    // on every substep while bonds fail. Keep the triangle query for paint.
    if (surface_positions != nullptr) return;
    const float free_fraction =
        ((cloth_inverse_masses[a] > 0.0F ? 1.0F : 0.0F) +
         (cloth_inverse_masses[b] > 0.0F ? 1.0F : 0.0F) +
         (cloth_inverse_masses[c] > 0.0F ? 1.0F : 0.0F)) / 3.0F;
    constexpr float cloth_share = 0.5F;
    const float cloth_shift = cloth_share * best_penetration;
    const Vec3 arm = subtract(best_contact, state.position);
    const float incoming = dot(state.linear_velocity, best_normal);
    const Vec3 angular_axis = cross(arm, best_normal);
    // The conservative body-radius constraint must remove inward center
    // velocity even if spin makes the contact-point velocity nearly zero.
    const float normal_impulse = incoming < 0.0F && body.inverse_mass > 0.0F
        ? -incoming / body.inverse_mass : 0.0F;
    const float support_radius = fmaxf(0.15F, mesh.bounding_radius * 0.8F);
    const float inverse_support_squared = 1.0F /
        (support_radius * support_radius);
    float weight_sum = 0.0F;
    float maximum_weighted_inverse_mass = 0.0F;
    float weighted_inverse_mass_squared = 0.0F;
    Vec3 weighted_cloth_velocity{};
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
        const float inverse_mass = cloth_inverse_masses[vertex];
        if (inverse_mass == 0.0F) continue;
        const float squared = length_squared(subtract(
            cloth_positions[vertex], best_contact));
        const float weight = fmaxf(0.0F,
            1.0F - squared * inverse_support_squared);
        const float weighted = weight * weight;
        weight_sum += weighted;
        weighted_inverse_mass_squared += inverse_mass * weighted * weighted;
        weighted_cloth_velocity = add(weighted_cloth_velocity,
                                      multiply(cloth_velocities[vertex], weighted));
        maximum_weighted_inverse_mass = fmaxf(
            maximum_weighted_inverse_mass, inverse_mass * weighted);
    }
    // Bound the velocity kick of every contacted cloth vertex to one tenth
    // of its collision thickness per substep. Excess impact is dissipated.
    const float maximum_cloth_impulse = maximum_weighted_inverse_mass > 0.0F
        ? 0.1F * thickness * weight_sum /
              (dt * maximum_weighted_inverse_mass)
        : normal_impulse;
    const float cloth_impulse = fminf(normal_impulse,
                                     maximum_cloth_impulse);
    Vec3 tangent_impulse{};
    if (weight_sum > 1.0e-8F && contact_friction > 0.0F) {
        const Vec3 cloth_velocity = multiply(weighted_cloth_velocity,
                                             1.0F / weight_sum);
        const Vec3 body_velocity = add(state.linear_velocity,
                                      cross(state.angular_velocity, arm));
        const Vec3 relative = subtract(body_velocity, cloth_velocity);
        const float separating_speed = dot(relative, best_normal);
        const Vec3 tangent = subtract(relative,
            multiply(best_normal, separating_speed));
        const float tangent_speed = vector_length(tangent);
        // Friction must not turn an outward-moving body into a cloth tether.
        const float release_weight = fmaxf(0.0F,
            1.0F - fmaxf(incoming, 0.0F) / 0.5F);
        if (release_weight > 0.0F && separating_speed <= 0.0F &&
            tangent_speed > 1.0e-6F) {
            const Vec3 direction = multiply(tangent, 1.0F / tangent_speed);
            const Vec3 angular_axis = cross(arm, direction);
            const float cloth_inverse_mass =
                weighted_inverse_mass_squared / (weight_sum * weight_sum);
            const float effective_inverse_mass = body.inverse_mass +
                dot(cross(inverse_inertia_world(body, state, angular_axis), arm),
                    direction) + cloth_inverse_mass;
            // Resting contact can have little incoming speed despite a finite
            // positional correction; use that correction as normal load.
            const float correction_impulse = best_penetration > 0.0F &&
                    body.inverse_mass > 0.0F
                ? 0.2F * best_penetration / (dt * body.inverse_mass) : 0.0F;
            const float friction_limit = contact_friction * release_weight *
                fmaxf(normal_impulse, correction_impulse);
            const float magnitude = effective_inverse_mass > k_epsilon
                ? fminf(tangent_speed / effective_inverse_mass, friction_limit)
                : 0.0F;
            tangent_impulse = multiply(direction, -magnitude);
        }
    }
    // Limit the cloth-side kick independently of the rigid-body friction.
    const Vec3 cloth_tangent_impulse = clamp_length(tangent_impulse,
                                                    maximum_cloth_impulse);
    corrections[body_index] = {
        multiply(best_normal, -cloth_shift),
        subtract(multiply(best_normal, -cloth_impulse),
                 cloth_tangent_impulse), best_contact,
        support_radius, weight_sum, {a, b, c}, true};
    state.position = add(state.position,
        multiply(best_normal,
                 best_penetration - cloth_shift * free_fraction));
    if (normal_impulse > 0.0F) {
        state.linear_velocity = clamp_length(add(state.linear_velocity,
            multiply(best_normal, normal_impulse * body.inverse_mass)),
            body.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(state.angular_velocity,
            inverse_inertia_world(body, state,
                multiply(angular_axis, normal_impulse))),
            body.maximum_angular_speed);
    }
    state.linear_velocity = clamp_length(add(state.linear_velocity,
        multiply(tangent_impulse, body.inverse_mass)),
        body.maximum_linear_speed);
    state.angular_velocity = clamp_length(add(state.angular_velocity,
        inverse_inertia_world(body, state, cross(arm, tangent_impulse))),
        body.maximum_angular_speed);
    states[body_index] = state;
}

__global__ void cloth_apply_body_corrections(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    std::uint32_t count,
    const ClothBodyCorrection *corrections, std::uint32_t body_count) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count || inverse_masses[vertex] == 0.0F) return;
    Vec3 offset{};
    Vec3 velocity = velocities[vertex];
    for (std::uint32_t body = 0U; body < body_count; ++body) {
        const ClothBodyCorrection correction = corrections[body];
        if (!correction.active) continue;
        if (correction.vertices[0] == vertex ||
            correction.vertices[1] == vertex ||
            correction.vertices[2] == vertex)
            offset = add(offset, correction.offset);
        if (correction.weight_sum > 1.0e-8F) {
            const float squared = length_squared(subtract(
                positions[vertex], correction.contact));
            const float support_squared = correction.support_radius *
                correction.support_radius;
            const float weight = fmaxf(0.0F, 1.0F - squared / support_squared);
            velocity = add(velocity, multiply(correction.impulse,
                inverse_masses[vertex] * weight * weight /
                    correction.weight_sum));
        }
    }
    positions[vertex] = add(positions[vertex], offset);
    velocities[vertex] = clamp_length(velocity, 20.0F);
}

__global__ void fluid_reserve_contact_events(
    const std::uint32_t *selected_count, std::uint32_t *offset,
    std::uint32_t *world_count, std::uint32_t *overflow,
    std::uint32_t capacity) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    const std::uint32_t used = *world_count;
    const std::uint32_t available = capacity - used;
    const std::uint32_t retained = min(*selected_count, available);
    *offset = used;
    *world_count = used + retained;
    *overflow += *selected_count - retained;
}

__global__ void fluid_gather_contact_events(
    const std::uint32_t *selected, const std::uint32_t *selected_count,
    const std::uint32_t *offset, const FluidContactSample *samples,
    const std::uint32_t *stable_ids, const RigidBodyId *body_ids,
    FluidId fluid, ContactEvent *events, std::uint32_t capacity) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= *selected_count || *offset + item >= capacity) return;
    const std::uint32_t particle = selected[item];
    const FluidContactSample sample = samples[particle];
    events[*offset + item] = {fluid, stable_ids[particle],
        body_ids[sample.body], sample.position, sample.normal,
        sample.normal_impulse};
}

__global__ void fluid_source_vacancies(
    const Vec3 *points, std::uint32_t amount, float spacing,
    const Vec3 *positions, const std::uint64_t *keys,
    const std::uint32_t *indices, std::uint32_t capacity,
    float inverse_cell_size, std::uint8_t *vacant) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= amount) return;
    const Vec3 p = points[item];
    const int cx = max(-k_fluid_cell_bias, min(k_fluid_cell_bias-1, __float2int_rd(p.x*inverse_cell_size)));
    const int cy = max(-k_fluid_cell_bias, min(k_fluid_cell_bias-1, __float2int_rd(p.y*inverse_cell_size)));
    const int cz = max(-k_fluid_cell_bias, min(k_fluid_cell_bias-1, __float2int_rd(p.z*inverse_cell_size)));
    vacant[item] = 0;
    for(int z=-1;z<=1;++z) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        const auto key = fluid_cell_key(cx+x, cy+y, cz+z);
        for(auto i=fluid_lower_bound(keys,capacity,key);i<capacity && keys[i]==key;++i)
            if(length_squared(subtract(p,positions[indices[i]])) < spacing*spacing)
                return;
    }
    vacant[item] = 1;
}

// A bounded deterministic commit pass avoids duplicate emission at seams and
// intersecting sources. Vacancies above are tested in parallel against the
// spatial index; only particles appended since that index need a direct check.
__global__ void fluid_source_emit(
    const Vec3 *points, const std::uint8_t *vacant, std::uint32_t amount,
    float spacing, Vec3 velocity, Vec3 *positions, Vec3 *velocities,
    std::uint32_t *ids, float *foam, std::uint32_t *count,
    std::uint32_t capacity, std::uint32_t first_spawned,
    std::uint32_t first_id, std::uint32_t *misses) {
    if(threadIdx.x || blockIdx.x) return;
    *misses=0;
    const auto previous_count=*count;
    for(std::uint32_t item=0;item<amount;++item) {
        if(!vacant[item]) continue;
        const Vec3 p=points[item];
        bool occupied=false;
        // This source's sites are already separated by the host sampler.
        // Only earlier sources need a cross-source conflict check.
        for(std::uint32_t i=first_spawned;i<previous_count;++i)
            if(length_squared(subtract(p,positions[i])) < spacing*spacing) {occupied=true;break;}
        if(occupied) continue;
        if(*count==capacity) {++*misses;continue;}
        const auto index=(*count)++;
        positions[index]=p;
        velocities[index]=velocity;
        ids[index]=first_id+index-first_spawned;
        foam[index]=0;
    }
}

__global__ void fluid_destroy_flags(const Vec3 *positions,
                                    const Vec3 *previous,
                                    const std::uint32_t *count,
                                    std::uint32_t capacity,
                                    ParticleDestroyPlaneOptions plane,
                                    bool combine,
                                    std::uint8_t *keep) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= capacity) return;
    if (item >= *count) { keep[item] = 0U; return; }
    const Vec3 before = inverse_rotate(plane.plane.orientation,
        subtract(previous[item], plane.plane.center));
    const Vec3 after = inverse_rotate(plane.plane.orientation,
        subtract(positions[item], plane.plane.center));
    const bool along = before.y < 0.0F && after.y >= 0.0F;
    const bool against = before.y > 0.0F && after.y <= 0.0F;
    const bool direction = plane.crossing == CrossingDirection::either
        ? along || against
        : plane.crossing == CrossingDirection::along_normal ? along : against;
    const float fraction = before.y / (before.y - after.y + 1.0e-20F);
    const Vec3 crossing = add(before, multiply(subtract(after, before), fraction));
    const bool in_bounds = fabsf(crossing.x) <= plane.plane.half_extents.x &&
                           fabsf(crossing.z) <= plane.plane.half_extents.y;
    const std::uint8_t survives = static_cast<std::uint8_t>(!direction || !in_bounds);
    keep[item] = combine ? keep[item] & survives : survives;
}

__global__ void fluid_gather(const Vec3 *positions, const Vec3 *velocities,
                             const std::uint32_t *ids, const float *foam,
                             const FluidContactSample *contact_samples,
                             const std::uint8_t *contact_flags,
                             bool copy_contacts,
                             const std::uint32_t *selected,
                             const std::uint32_t *count,
                             std::uint32_t capacity, Vec3 *next_positions,
                             Vec3 *next_velocities, std::uint32_t *next_ids,
                             float *next_foam,
                             FluidContactSample *next_contact_samples,
                             std::uint8_t *next_contact_flags) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= capacity || item >= *count) return;
    const std::uint32_t source = selected[item];
    next_positions[item] = positions[source];
    next_velocities[item] = velocities[source];
    next_ids[item] = ids[source];
    next_foam[item] = foam[source];
    if (copy_contacts) {
        next_contact_samples[item] = contact_samples[source];
        next_contact_flags[item] = contact_flags[source];
    }
}

__global__ void fluid_copy_initial(const FluidParticle *input,
                                   std::uint32_t count, Vec3 *positions,
                                   Vec3 *velocities, std::uint32_t *ids,
                                   float *foam, std::uint32_t *invalid) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (!isfinite(input[index].position.x) ||
        !isfinite(input[index].position.y) ||
        !isfinite(input[index].position.z) ||
        !isfinite(input[index].velocity.x) ||
        !isfinite(input[index].velocity.y) ||
        !isfinite(input[index].velocity.z)) {
        atomicExch(invalid, 1U);
        return;
    }
    positions[index] = input[index].position;
    velocities[index] = input[index].velocity;
    ids[index] = index;
    foam[index] = 0.0F;
}

} // namespace

namespace {
#include "rope.cuh"
} // namespace

struct FrameToken::Impl {
    std::shared_ptr<CompletionState> completion{};
};

struct World::Impl {
    struct Slot {
        std::uint32_t generation{1U};
        std::uint32_t dense_index{k_invalid_dense};
        bool alive{};
    };

    WorldOptions options{};
    int device_ordinal{-1};
    std::uint32_t rigid_body_count{};
    std::uint32_t triangle_mesh_count{};
    std::uint32_t fluid_count{};
    std::uint64_t emitted_particle_count{};
    std::uint64_t destroyed_particle_count{};
    std::uint64_t spawn_capacity_miss_count{};
    std::uint32_t current_state{};
    std::uint64_t frame_index{};
    std::uint64_t revision{};
    std::uint32_t rigid_solve_kernels_per_substep{1U};
    std::uint32_t soft_cloth_kernels_per_substep{};
    std::vector<Slot> slots{};
    std::vector<std::unique_ptr<FluidStorage>> fluids{};
    std::vector<std::unique_ptr<ClothStorage>> cloths{};
    std::vector<std::unique_ptr<SoftBodyStorage>> soft_bodies{};
    std::vector<std::unique_ptr<RopeStorage>> ropes{};
    std::vector<FluidClothCouplingResource> fluid_cloth_couplings{};
    std::vector<std::unique_ptr<FluidSoftCouplingStorage>> fluid_soft_couplings{};
    std::vector<std::unique_ptr<SoftClothCouplingStorage>> soft_cloth_couplings{};
    std::vector<ParticleSourceSlot> particle_sources{};
    std::vector<DestroyPlaneSlot> destroy_planes{};
    PaintFieldResource *paint_fields{};
    PaintRuleResource *paint_rules{};
    std::uint32_t paint_rule_count{};
    std::uint32_t *fluid_neighbor_overflow{};
    std::uint32_t *fluid_maximum_neighbor_count{};
    BodyParameters *parameters{};
    BodyAccumulator *accumulators{};
    KinematicTarget *targets{};
    RigidBodyId *ids{};
    RigidBodyState *states[2]{};
    Vec3 *debug_applied_forces{};
    Vec3 *debug_applied_torques{};
    RigidBodyState *fluid_previous_states{};
    ContactManifold *rigid_manifolds{};
    std::uint32_t *rigid_color_owners{};
    std::uint8_t *rigid_pair_colors{};
    std::uint32_t *rigid_color_state{};
    std::uint32_t *rigid_contact_event_offsets{};
    WorldAabb *rigid_world_bounds{};
    WorldAabb *fluid_body_bounds{};
    unsigned long long *fluid_body_masks{};
    unsigned long long *fluid_global_body_masks{};
    std::uint32_t *fluid_body_contact_flags{};
    ContactEvent *fluid_contact_events{};
    std::uint32_t *fluid_contact_count{};
    std::uint32_t *fluid_contact_overflow{};
    std::uint8_t *rigid_active_pair_flags{};
    std::uint32_t *rigid_active_pairs{};
    std::uint32_t *rigid_active_pair_count{};
    std::uint8_t *rigid_broad_phase_workspace{};
    std::size_t rigid_broad_phase_workspace_size{};
    LeafPair *rigid_leaf_pairs{};
    std::uint32_t *rigid_leaf_pair_counts{};
    std::uint32_t rigid_leaf_pair_slot_capacity{};
    std::size_t rigid_leaf_pair_capacity{};
    RigidContactEvent *rigid_contact_events{};
    std::uint32_t *rigid_contact_count{};
    std::uint32_t rigid_contact_capacity{};
    TriangleMeshResource *meshes{};
    std::shared_ptr<CompletionState> frame{};
    std::vector<cudaEvent_t> timing_events{};
    std::vector<TimingStage> timing_stages{};
    std::vector<std::uint32_t> timing_launch_counts{};
    std::size_t timing_boundary_count{};
    std::uint64_t timing_frame_index{};
    bool timing_available{};
    std::vector<PhysicsDebugFrame> debug_frames{};
    std::size_t debug_next_frame{};
    std::size_t debug_frame_count{};

    [[nodiscard]] Status record_debug_frame(
        StepOptions step_options) noexcept {
        if (options.physics_debug.frame_capacity == 0U ||
            frame_index % options.physics_debug.frame_stride != 0U)
            return success();
        try {
            PhysicsDebugFrame &output = debug_frames[debug_next_frame];
            output.frame_index = frame_index;
            output.timestep = step_options.timestep;
            output.gravity = step_options.gravity;
            output.maximum_fluid_neighbor_count =
                *fluid_maximum_neighbor_count;
            output.rigid_bodies.resize(rigid_body_count);
            for (std::uint32_t index = 0U; index < rigid_body_count; ++index) {
                output.rigid_bodies[index] = {
                    ids[index], states[current_state][index],
                    debug_applied_forces[index], debug_applied_torques[index]};
            }
            output.fluid_particles.clear();
            for (std::uint32_t slot = 0U; slot < fluids.size(); ++slot) {
                const auto &owner = fluids[slot];
                if (!owner || !owner->alive) continue;
                const FluidStorage &fluid = *owner;
                const std::uint32_t count = *fluid.count;
                output.fluid_particles.reserve(
                    output.fluid_particles.size() + count);
                for (std::uint32_t index = 0U; index < count; ++index)
                    output.fluid_particles.push_back({
                        {slot, fluid.generation}, fluid.ids[index],
                        fluid.positions[index], fluid.velocities[index],
                        fluid.forces[index], fluid.foam[index]});
            }
            output.cloth_vertices.clear();
            for (std::uint32_t slot = 0U; slot < cloths.size(); ++slot) {
                const auto &owner = cloths[slot];
                if (!owner || !owner->alive) continue;
                const ClothStorage &cloth = *owner;
                output.cloth_vertices.reserve(
                    output.cloth_vertices.size() + cloth.vertex_count);
                for (std::uint32_t index = 0U;
                     index < cloth.vertex_count; ++index)
                    output.cloth_vertices.push_back({
                        {slot, cloth.generation}, index,
                        cloth.positions[index], cloth.velocities[index],
                        cloth.rigid_contact_forces[index],
                        cloth.fluid_forces != nullptr
                            ? cloth.fluid_forces[index] : Vec3{},
                        cloth.soft_body_forces[index]});
            }
            output.soft_body_nodes.clear();
            output.rope_nodes.clear();
            for(unsigned slot=0;slot<ropes.size();++slot) {
                if(!ropes[slot] || !ropes[slot]->alive)continue;
                const auto &r=ropes[slot]->data;
                for(unsigned i=0;i<r.count;++i)output.rope_nodes.push_back({
                    {slot,ropes[slot]->generation},i,r.positions[i],r.velocities[i],r.constraint_forces[i],r.contact_forces[i]});
            }
            for (std::uint32_t slot = 0U; slot < soft_bodies.size(); ++slot) {
                const auto &owner = soft_bodies[slot];
                if (!owner || !owner->alive) continue;
                const SoftBodyStorage &body = *owner;
                output.soft_body_nodes.reserve(
                    output.soft_body_nodes.size() + body.node_count);
                for (std::uint32_t index = 0U; index < body.node_count; ++index)
                    output.soft_body_nodes.push_back({
                        {slot, body.generation}, index, body.positions[index],
                        body.velocities[index],
                        body.rigid_contact_forces[index], body.cloth_forces[index],
                        body.fluid_forces[index]});
            }
            const std::uint32_t retained_rigid_contacts = std::min(
                *rigid_contact_count, rigid_contact_capacity);
            output.rigid_contacts.assign(
                rigid_contact_events,
                rigid_contact_events + retained_rigid_contacts);
            const std::uint32_t retained_fluid_contacts = std::min(
                *fluid_contact_count, options.contact_capacity);
            output.fluid_contacts.assign(
                fluid_contact_events,
                fluid_contact_events + retained_fluid_contacts);
            debug_next_frame =
                (debug_next_frame + 1U) % debug_frames.size();
            debug_frame_count = std::min(
                debug_frame_count + 1U, debug_frames.size());
            return success();
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "physics debug frame allocation failed");
        }
    }

    ~Impl() {
        if (frame && !frame->acknowledged) {
            (void)wait_for_completion(frame);
        }
        if (meshes != nullptr) {
            for (std::uint32_t index = 0;
                 index < options.triangle_mesh_capacity; ++index) {
                release_managed(meshes[index].bvh_nodes);
                release_managed(meshes[index].bvh_leaves);
                release_managed(meshes[index].solid_planes);
                release_managed(meshes[index].indices);
                release_managed(meshes[index].vertices);
            }
        }
        if (paint_fields != nullptr) {
            for (std::uint32_t index = 0;
                 index < options.paint_field_capacity; ++index) {
                release_managed(paint_fields[index].uvs);
                release_managed(paint_fields[index].pixels);
            }
        }
        release_managed(paint_fields);
        release_managed(paint_rules);
        for (cudaEvent_t event : timing_events) {
            cudaEventDestroy(event);
        }
        release_managed(meshes);
        release_managed(fluid_neighbor_overflow);
        release_managed(fluid_maximum_neighbor_count);
        release_managed(rigid_contact_count);
        release_managed(rigid_contact_events);
        release_managed(rigid_leaf_pair_counts);
        release_managed(rigid_leaf_pairs);
        release_managed(rigid_broad_phase_workspace);
        release_managed(rigid_active_pair_count);
        release_managed(rigid_active_pairs);
        release_managed(rigid_active_pair_flags);
        release_managed(rigid_world_bounds);
        release_managed(fluid_body_bounds);
        release_managed(fluid_body_masks);
        release_managed(fluid_global_body_masks);
        release_managed(fluid_body_contact_flags);
        release_managed(fluid_contact_events);
        release_managed(fluid_contact_count);
        release_managed(fluid_contact_overflow);
        release_managed(rigid_contact_event_offsets);
        release_managed(rigid_color_state);
        release_managed(rigid_pair_colors);
        release_managed(rigid_color_owners);
        release_managed(rigid_manifolds);
        release_managed(states[1]);
        release_managed(debug_applied_forces);
        release_managed(debug_applied_torques);
        release_managed(fluid_previous_states);
        release_managed(states[0]);
        release_managed(ids);
        release_managed(targets);
        release_managed(accumulators);
        release_managed(parameters);
    }

    [[nodiscard]] Status prepare_timing_events(
        std::size_t boundary_count) noexcept {
        try {
            timing_events.reserve(boundary_count);
            timing_stages.clear();
            timing_launch_counts.clear();
            timing_launch_counts.reserve(boundary_count);
            timing_stages.reserve(boundary_count > 0U ? boundary_count - 1U
                                                       : 0U);
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "failed to allocate timing event storage");
        }
        while (timing_events.size() < boundary_count) {
            cudaEvent_t event = nullptr;
            const cudaError_t error = cudaEventCreate(&event);
            if (error != cudaSuccess) {
                return cuda_failure(error, "failed to create CUDA timing event");
            }
            try {
                timing_events.push_back(event);
            } catch (...) {
                cudaEventDestroy(event);
                return failure(StatusCode::out_of_memory,
                               "failed to retain CUDA timing event");
            }
        }
        return success();
    }

    [[nodiscard]] Status require_current_device() const noexcept {
        int current_device = -1;
        const cudaError_t error = cudaGetDevice(&current_device);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to query the current CUDA device");
        }
        if (current_device != device_ordinal) {
            return failure(StatusCode::invalid_argument,
                           "world used from a different CUDA device");
        }
        return success();
    }

    [[nodiscard]] Status require_idle() noexcept {
        Status status = require_current_device();
        if (!status) {
            return status;
        }
        if (frame && !frame->acknowledged) {
            return failure(StatusCode::busy,
                           "world still has an unacknowledged frame");
        }
        frame.reset();
        return success();
    }

    [[nodiscard]] Status validate_handle(RigidBodyId id,
                                         std::uint32_t &dense) const noexcept {
        if (id.index >= slots.size()) {
            return failure(StatusCode::invalid_handle,
                           "rigid body handle index is invalid");
        }
        const Slot &slot = slots[id.index];
        if (!slot.alive || slot.generation != id.generation) {
            return failure(StatusCode::invalid_handle,
                           "rigid body handle is stale");
        }
        dense = slot.dense_index;
        return success();
    }

    [[nodiscard]] Status validate_handle(TriangleMeshId id) const noexcept {
        if (id.index >= options.triangle_mesh_capacity || meshes == nullptr) {
            return failure(StatusCode::invalid_handle,
                           "triangle mesh handle index is invalid");
        }
        const TriangleMeshResource &mesh = meshes[id.index];
        if (!mesh.alive || mesh.generation != id.generation) {
            return failure(StatusCode::invalid_handle,
                           "triangle mesh handle is stale");
        }
        return success();
    }

    [[nodiscard]] Status validate_handle(FluidId id,
                                         FluidStorage *&fluid) const noexcept {
        if (id.index >= fluids.size() || !fluids[id.index] ||
            !fluids[id.index]->alive ||
            fluids[id.index]->generation != id.generation) {
            return failure(StatusCode::invalid_handle,
                           "fluid handle is invalid or stale");
        }
        fluid = fluids[id.index].get();
        return success();
    }
};

FrameToken::FrameToken() noexcept = default;

FrameToken::~FrameToken() {
    if (impl_ && impl_->completion && !impl_->completion->acknowledged) {
        (void)wait_for_completion(impl_->completion);
    }
}

FrameToken::FrameToken(FrameToken &&other) noexcept
    : impl_(std::move(other.impl_)) {}

FrameToken &FrameToken::operator=(FrameToken &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (impl_ && impl_->completion && !impl_->completion->acknowledged) {
        (void)wait_for_completion(impl_->completion);
    }
    impl_ = std::move(other.impl_);
    return *this;
}

bool FrameToken::pending() const noexcept {
    return impl_ && impl_->completion && !impl_->completion->acknowledged;
}

bool FrameToken::ready() const noexcept {
    if (!pending()) {
        return true;
    }
    return cudaEventQuery(impl_->completion->event) == cudaSuccess;
}

Status FrameToken::wait() noexcept {
    if (!impl_) {
        return success();
    }
    return wait_for_completion(impl_->completion);
}

World::World() noexcept = default;
World::~World() = default;
World::World(World &&) noexcept = default;
World &World::operator=(World &&) noexcept = default;

Status World::create(WorldOptions options, World &output,
                     cudaStream_t stream) noexcept {
    (void)stream;
    if (options.rigid_body_capacity == 0U ||
        options.triangle_mesh_capacity == 0U) {
        return failure(StatusCode::invalid_argument,
                       "rigid body and triangle mesh capacities must be positive");
    }
    if (options.physics_debug.frame_capacity > 3'600U ||
        options.physics_debug.frame_stride == 0U) {
        return failure(StatusCode::invalid_argument,
                       "physics debug frame capacity or stride is invalid");
    }
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to query the current CUDA device");
    }

    std::unique_ptr<Impl> implementation;
    try {
        implementation = std::make_unique<Impl>();
        implementation->slots.resize(options.rigid_body_capacity);
        implementation->fluids.resize(options.fluid_capacity);
        implementation->cloths.resize(options.cloth_capacity);
        implementation->soft_bodies.resize(options.soft_body_capacity);
        implementation->ropes.resize(options.rope_capacity);
        implementation->fluid_cloth_couplings.resize(
            options.fluid_cloth_coupling_capacity);
        implementation->fluid_soft_couplings.resize(
            options.fluid_soft_body_coupling_capacity);
        implementation->soft_cloth_couplings.resize(
            options.soft_body_cloth_coupling_capacity);
        implementation->particle_sources.resize(options.particle_source_capacity);
        implementation->destroy_planes.resize(options.particle_destroy_plane_capacity);
        implementation->debug_frames.resize(
            options.physics_debug.frame_capacity);
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate world host storage");
    }
    implementation->options = options;
    implementation->device_ordinal = device;
    Status status = allocate_managed(implementation->paint_fields,
                                     options.paint_field_capacity);
    if (!status) return status;
    for (std::uint32_t i = 0; i < options.paint_field_capacity; ++i)
        implementation->paint_fields[i] = {};
    status = allocate_managed(implementation->paint_rules,
                              options.paint_rule_capacity);
    if (!status) return status;
    for (std::uint32_t i = 0; i < options.paint_rule_capacity; ++i)
        implementation->paint_rules[i] = {};
    status = allocate_managed(implementation->fluid_neighbor_overflow, 1U);
    if (!status) return status;
    *implementation->fluid_neighbor_overflow = 0U;
    status = allocate_managed(
        implementation->fluid_maximum_neighbor_count, 1U);
    if (!status) return status;
    *implementation->fluid_maximum_neighbor_count = 0U;
    const std::size_t pair_capacity =
        static_cast<std::size_t>(options.rigid_body_capacity) *
        options.rigid_body_capacity;
    // At most n(n-1)/2 pairs can involve a dynamic body: dynamic/dynamic
    // pairs are unique, and every other pair needs exactly one dynamic body.
    const std::size_t manifold_count =
        static_cast<std::size_t>(options.rigid_body_capacity) *
        (options.rigid_body_capacity - 1U) / 2U;
    if (manifold_count >
        std::numeric_limits<std::size_t>::max() / sizeof(ContactManifold)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds contact cache range");
    }
    const std::size_t requested_leaf_pair_slots = std::max(
        static_cast<std::size_t>(k_minimum_leaf_pair_cache_slots),
        static_cast<std::size_t>(options.rigid_body_capacity) *
            k_leaf_pair_cache_slots_per_body);
    const std::size_t leaf_pair_slot_capacity =
        std::min(manifold_count, requested_leaf_pair_slots);
    if (leaf_pair_slot_capacity > std::numeric_limits<std::size_t>::max() /
                                      k_max_leaf_pairs_per_body_pair ||
        leaf_pair_slot_capacity >
            std::numeric_limits<std::uint32_t>::max()) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds leaf-pair cache range");
    }
    implementation->rigid_leaf_pair_slot_capacity =
        static_cast<std::uint32_t>(leaf_pair_slot_capacity);
    implementation->rigid_leaf_pair_capacity =
        leaf_pair_slot_capacity * k_max_leaf_pairs_per_body_pair;
    implementation->rigid_contact_capacity = options.contact_capacity;

    status = allocate_managed(implementation->parameters,
                                     options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->accumulators,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->targets,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->ids,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->states[0],
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->states[1],
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    if (options.physics_debug.frame_capacity != 0U) {
        status = allocate_managed(implementation->debug_applied_forces,
                                  options.rigid_body_capacity);
        if (!status) return status;
        status = allocate_managed(implementation->debug_applied_torques,
                                  options.rigid_body_capacity);
        if (!status) return status;
    }
    status = allocate_managed(implementation->fluid_previous_states,
                              options.rigid_body_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->rigid_manifolds, manifold_count);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_color_owners,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_pair_colors, manifold_count);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_color_state, 2U);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_contact_event_offsets,
                              manifold_count);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_world_bounds,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->fluid_body_bounds,
                              options.rigid_body_capacity);
    if (!status) return status;
    const std::size_t fluid_body_words =
        (static_cast<std::size_t>(options.rigid_body_capacity) + 63U) / 64U;
    status = allocate_managed(implementation->fluid_body_masks,
                              k_fluid_body_buckets * fluid_body_words);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_global_body_masks,
                              fluid_body_words);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_body_contact_flags,
                              options.rigid_body_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_contact_events,
                              options.contact_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_contact_count, 1U);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_contact_overflow, 1U);
    if (!status) return status;
    *implementation->fluid_contact_count = 0U;
    *implementation->fluid_contact_overflow = 0U;
    status = allocate_managed(implementation->rigid_active_pair_flags,
                              pair_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_active_pairs,
                              pair_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_active_pair_count, 1U);
    if (!status) {
        return status;
    }
    if (pair_capacity > static_cast<std::size_t>(
                             std::numeric_limits<int>::max())) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds broad-phase range");
    }
    const auto pair_indices = thrust::make_counting_iterator<std::uint32_t>(0U);
    cudaError_t broad_phase_error = cub::DeviceSelect::Flagged(
        nullptr, implementation->rigid_broad_phase_workspace_size,
        pair_indices, implementation->rigid_active_pair_flags,
        implementation->rigid_active_pairs,
        implementation->rigid_active_pair_count,
        static_cast<int>(pair_capacity));
    if (broad_phase_error != cudaSuccess) {
        return cuda_failure(broad_phase_error,
                            "failed to size broad-phase workspace");
    }
    status = allocate_managed(implementation->rigid_broad_phase_workspace,
                              implementation->rigid_broad_phase_workspace_size);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_leaf_pairs,
                              implementation->rigid_leaf_pair_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_leaf_pair_counts,
                              pair_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_contact_events,
                              implementation->rigid_contact_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_contact_count, 1U);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->meshes,
                              options.triangle_mesh_capacity);
    if (!status) {
        return status;
    }

    std::fill_n(implementation->parameters, options.rigid_body_capacity,
                BodyParameters{});
    std::fill_n(implementation->accumulators, options.rigid_body_capacity,
                BodyAccumulator{});
    std::fill_n(implementation->targets, options.rigid_body_capacity,
                KinematicTarget{});
    std::fill_n(implementation->ids, options.rigid_body_capacity, RigidBodyId{});
    std::fill_n(implementation->states[0], options.rigid_body_capacity,
                RigidBodyState{});
    std::fill_n(implementation->states[1], options.rigid_body_capacity,
                RigidBodyState{});
    if (implementation->debug_applied_forces != nullptr) {
        std::fill_n(implementation->debug_applied_forces,
                    options.rigid_body_capacity, Vec3{});
        std::fill_n(implementation->debug_applied_torques,
                    options.rigid_body_capacity, Vec3{});
    }
    std::fill_n(implementation->rigid_manifolds, manifold_count,
                ContactManifold{});
    std::fill_n(implementation->rigid_world_bounds,
                options.rigid_body_capacity, WorldAabb{});
    std::fill_n(implementation->rigid_active_pair_flags, pair_capacity, 0U);
    std::fill_n(implementation->rigid_active_pairs, pair_capacity, 0U);
    *implementation->rigid_active_pair_count = 0U;
    std::fill_n(implementation->rigid_leaf_pairs,
                implementation->rigid_leaf_pair_capacity, LeafPair{});
    std::fill_n(implementation->rigid_leaf_pair_counts, pair_capacity, 0U);
    std::fill_n(implementation->rigid_contact_events,
                implementation->rigid_contact_capacity, RigidContactEvent{});
    *implementation->rigid_contact_count = 0U;
    std::fill_n(implementation->meshes, options.triangle_mesh_capacity,
                TriangleMeshResource{});
    for (std::uint32_t index = 0; index < options.triangle_mesh_capacity;
         ++index) {
        implementation->meshes[index].generation = 1U;
    }

    output.impl_ = std::move(implementation);
    return success();
}

Status World::add_fluid(FluidOptions options,
                        DeviceSpan<const FluidParticle> initial_particles,
                        FluidId &output, cudaStream_t stream) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.capacity == 0U || options.capacity > INT_MAX ||
        initial_particles.size > options.capacity ||
        (initial_particles.size != 0U && initial_particles.data == nullptr) ||
        options.solver_iterations == 0U || options.solver_iterations > 16U ||
        options.maximum_neighbors == 0U ||
        !finite(options.particle_radius) || options.particle_radius <= 0.0F ||
        !finite(options.support_radius) ||
        options.support_radius < 2.0F * options.particle_radius ||
        !finite(options.rest_density) || options.rest_density <= 0.0F ||
        !finite(options.repulsion) || options.repulsion < 0.0F ||
        !finite(options.viscosity) || options.viscosity < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.maximum_speed) || options.maximum_speed <= 0.0F ||
        !finite(options.normal_damping) || options.normal_damping < 0.0F ||
        !finite(options.rest_particle_volume) ||
        options.rest_particle_volume < 0.0F ||
        !finite(options.maximum_pair_acceleration) ||
        options.maximum_pair_acceleration < 0.0F) {
        return failure(StatusCode::invalid_argument, "fluid options or initial particles are invalid");
    }
    std::uint32_t slot = 0U;
    for (; slot < impl_->fluids.size(); ++slot) {
        if (!impl_->fluids[slot] || !impl_->fluids[slot]->alive) break;
    }
    if (slot == impl_->fluids.size())
        return failure(StatusCode::capacity_exceeded, "fluid capacity exhausted");
    std::unique_ptr<FluidStorage> fluid;
    try { fluid = std::make_unique<FluidStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "fluid owner allocation failed"); }
    fluid->generation = impl_->fluids[slot] ? impl_->fluids[slot]->generation : 1U;
    fluid->options = options;
    const std::size_t capacity = options.capacity;
    if (!(status = allocate_managed(fluid->count, 1U)) ||
        !(status = allocate_managed(fluid->positions, capacity)) ||
        !(status = allocate_managed(fluid->velocities, capacity)) ||
        !(status = allocate_managed(fluid->previous, capacity)) ||
        !(status = allocate_managed(fluid->ids, capacity)) ||
        !(status = allocate_managed(fluid->foam, capacity)) ||
        !(status = allocate_managed(fluid->foam_source, capacity)) ||
        !(status = allocate_managed(fluid->next_positions, capacity)) ||
        !(status = allocate_managed(fluid->next_velocities, capacity)) ||
        !(status = allocate_managed(fluid->next_ids, capacity)) ||
        !(status = allocate_managed(fluid->next_foam, capacity)) ||
        !(status = allocate_managed(fluid->keep, capacity)) ||
        !(status = allocate_managed(fluid->selected, capacity)) ||
        !(status = allocate_managed(fluid->keys[0], capacity)) ||
        !(status = allocate_managed(fluid->keys[1], capacity)) ||
        !(status = allocate_managed(fluid->indices[0], capacity)) ||
        !(status = allocate_managed(fluid->indices[1], capacity)) ||
        !(status = allocate_managed(fluid->forces, capacity)) ||
        !(status = allocate_managed(fluid->body_impulses, capacity)) ||
        !(status = allocate_managed(fluid->contact_samples, capacity)) ||
        !(status = allocate_managed(fluid->contact_flags, capacity)) ||
        !(status = allocate_managed(fluid->next_contact_samples, capacity)) ||
        !(status = allocate_managed(fluid->next_contact_flags, capacity)) ||
        !(status = allocate_managed(fluid->contact_count, 1U)) ||
        !(status = allocate_managed(fluid->contact_offset, 1U))) return status;
    cudaError_t error = cub::DeviceRadixSort::SortPairs(
        nullptr, fluid->sort_workspace_size, fluid->keys[0], fluid->keys[1],
        fluid->indices[0], fluid->indices[1], options.capacity);
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid sort workspace query failed");
    const auto sequence = thrust::make_counting_iterator<std::uint32_t>(0U);
    error = cub::DeviceSelect::Flagged(
        nullptr, fluid->select_workspace_size, sequence, fluid->keep,
        fluid->selected, fluid->count, options.capacity);
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid compaction workspace query failed");
    if (!(status = allocate_managed(fluid->sort_workspace,
                                    fluid->sort_workspace_size)) ||
        !(status = allocate_managed(fluid->select_workspace,
                                    fluid->select_workspace_size))) return status;
    *fluid->count = static_cast<std::uint32_t>(initial_particles.size);
    fluid->next_id = static_cast<std::uint32_t>(initial_particles.size);
    fluid->initial_count = initial_particles.size;
    if (!initial_particles.empty()) {
        *impl_->fluid_neighbor_overflow = 0U;
        fluid_copy_initial<<<(initial_particles.size + 127U) / 128U,
                             128U, 0, stream>>>(
            initial_particles.data, static_cast<std::uint32_t>(initial_particles.size),
            fluid->positions, fluid->velocities, fluid->ids, fluid->foam,
            impl_->fluid_neighbor_overflow);
        error = cudaGetLastError();
        if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
            return cuda_failure(error, "initial fluid copy failed");
        if (*impl_->fluid_neighbor_overflow != 0U)
            return failure(StatusCode::invalid_argument,
                           "initial fluid particles must be finite");
    }
    fluid->alive = true;
    output = {slot, fluid->generation};
    impl_->fluids[slot] = std::move(fluid);
    ++impl_->fluid_count;
    ++impl_->revision;
    return success();
}

Status World::add_fluid_geometry(FluidOptions options,
                                 FluidGeometrySource source, FluidId &output,
                                 cudaStream_t stream) noexcept {
    std::vector<FluidParticle> sampled;
    Status status = sample_fluid_geometry(source, sampled);
    if (!status) return status;
    if (options.capacity == 0U)
        return failure(StatusCode::invalid_argument,
                       "fluid geometry requires a positive fluid capacity");
    if (sampled.size() > options.capacity) {
        std::vector<FluidParticle> selected;
        try {
            selected.reserve(options.capacity);
            for (std::size_t i = 0; i < options.capacity; ++i)
                selected.push_back(sampled[i * sampled.size() /
                                            options.capacity]);
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "fluid geometry selection allocation failed");
        }
        sampled.swap(selected);
    }
    FluidParticle *device = nullptr;
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&device),
                                    sampled.size() * sizeof(FluidParticle));
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid geometry upload allocation failed");
    error = cudaMemcpy(device, sampled.data(),
                       sampled.size() * sizeof(FluidParticle),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess)
        status = add_fluid(options, {device, sampled.size()}, output, stream);
    cudaFree(device);
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid geometry upload failed");
    return status;
}

Status World::remove_fluid(FluidId id, cudaStream_t) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(id, fluid))) return status;
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a soft-body coupling");
    for (std::uint32_t i = 0; i < impl_->options.paint_rule_capacity; ++i)
        if (impl_->paint_rules[i].alive &&
            impl_->paint_rules[i].options.source == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a paint rule");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.fluid == id)
            return failure(StatusCode::invalid_argument,
                "fluid is still referenced by a cloth coupling");
    std::unique_ptr<FluidStorage> tombstone(
        new (std::nothrow) FluidStorage());
    if (!tombstone)
        return failure(StatusCode::out_of_memory,
                       "fluid removal tombstone allocation failed");
    const std::uint32_t generation = fluid->generation + 1U;
    impl_->destroyed_particle_count +=
        fluid->initial_count + fluid->emitted_count - *fluid->count;
    tombstone->generation = generation;
    impl_->fluids[id.index] = std::move(tombstone);
    for (ParticleSourceSlot &plane : impl_->particle_sources)
        if (plane.alive && plane.options.fluid == id) {
            plane.alive = false; ++plane.generation; plane.data.reset();
        }
    for (DestroyPlaneSlot &plane : impl_->destroy_planes)
        if (plane.alive && plane.options.fluid == id) {
            plane.alive = false; ++plane.generation;
        }
    --impl_->fluid_count;
    ++impl_->revision;
    return success();
}

Status World::fluid_view(FluidId id, FluidDeviceView &output) const noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(id, fluid))) return status;
    if (impl_->frame && !impl_->frame->acknowledged)
        return failure(StatusCode::busy, "fluid view requires a completed frame");
    const std::uint32_t count = *fluid->count;
    output = {{fluid->positions, count}, {fluid->velocities, count},
              {fluid->forces, count}, {fluid->ids, count},
              {fluid->foam, count}, count, fluid->options.particle_radius,
              fluid->options.support_radius, impl_->revision};
    return success();
}

namespace {
[[nodiscard]] bool valid_particle_plane(ParticlePlane plane) noexcept {
    const float squared = plane.orientation.x * plane.orientation.x +
        plane.orientation.y * plane.orientation.y +
        plane.orientation.z * plane.orientation.z +
        plane.orientation.w * plane.orientation.w;
    return finite(plane.center) && finite(plane.orientation) &&
        finite(plane.half_extents.x) && plane.half_extents.x > 0.0F &&
        finite(plane.half_extents.y) && plane.half_extents.y > 0.0F &&
        squared > 0.25F && squared < 4.0F;
}
} // namespace

Status World::add_particle_source(ParticleSourceMesh mesh, ParticleSourceOptions options,
                                       ParticleSourceId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (mesh.spacing == 0.0F) mesh.spacing = fluid->options.support_radius;
    if (!finite(options.initial_velocity) || !finite(mesh.spacing) ||
        mesh.spacing < 2.0F * fluid->options.particle_radius)
        return failure(StatusCode::invalid_argument, "fluid source spacing or velocity is invalid");
    for (std::uint32_t index = 0; index < impl_->particle_sources.size(); ++index) {
        ParticleSourceSlot &plane = impl_->particle_sources[index];
        if (plane.alive) continue;
        std::vector<Vec3> points;
        if (!(status = sample_fluid_source(mesh, points))) return status;
        std::unique_ptr<ParticleSourceData> data(new (std::nothrow) ParticleSourceData());
        if (!data) return failure(StatusCode::out_of_memory, "fluid source allocation failed");
        data->count = static_cast<std::uint32_t>(points.size());
        data->spacing = mesh.spacing;
        if (!(status = allocate_managed(data->points, points.size())) ||
            !(status = allocate_managed(data->vacant, points.size())) ||
            !(status = allocate_managed(data->capacity_misses, 1))) return status;
        std::copy(points.begin(), points.end(), data->points);
        *data->capacity_misses = 0;
        plane.options = options;
        plane.data = std::move(data);
        plane.alive = true;
        output = {index, plane.generation};
        return success();
    }
    return failure(StatusCode::capacity_exceeded, "fluid source capacity exhausted");
}

Status World::update_particle_source(ParticleSourceId id,
                                          ParticleSourceOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->particle_sources.size() ||
        !impl_->particle_sources[id.index].alive ||
        impl_->particle_sources[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid source handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!finite(options.initial_velocity) ||
        !(options.fluid == impl_->particle_sources[id.index].options.fluid))
        return failure(StatusCode::invalid_argument, "fluid source destination is immutable or velocity is invalid");
    impl_->particle_sources[id.index].options = options;
    return success();
}

Status World::remove_particle_source(ParticleSourceId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->particle_sources.size() ||
        !impl_->particle_sources[id.index].alive ||
        impl_->particle_sources[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid source handle is stale");
    impl_->particle_sources[id.index].alive = false;
    impl_->particle_sources[id.index].data.reset();
    ++impl_->particle_sources[id.index].generation;
    return success();
}

Status World::add_particle_destroy_plane(ParticleDestroyPlaneOptions options,
                                         ParticleDestroyPlaneId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane))
        return failure(StatusCode::invalid_argument, "destroy plane options are invalid");
    for (std::uint32_t index = 0; index < impl_->destroy_planes.size(); ++index) {
        DestroyPlaneSlot &plane = impl_->destroy_planes[index];
        if (plane.alive) continue;
        plane.options = options;
        plane.alive = true;
        output = {index, plane.generation};
        return success();
    }
    return failure(StatusCode::capacity_exceeded, "destroy plane capacity exhausted");
}

Status World::update_particle_destroy_plane(ParticleDestroyPlaneId id,
                                            ParticleDestroyPlaneOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->destroy_planes.size() ||
        !impl_->destroy_planes[id.index].alive ||
        impl_->destroy_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "destroy plane handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane))
        return failure(StatusCode::invalid_argument, "destroy plane options are invalid");
    impl_->destroy_planes[id.index].options = options;
    return success();
}

Status World::remove_particle_destroy_plane(ParticleDestroyPlaneId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->destroy_planes.size() ||
        !impl_->destroy_planes[id.index].alive ||
        impl_->destroy_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "destroy plane handle is stale");
    impl_->destroy_planes[id.index].alive = false;
    ++impl_->destroy_planes[id.index].generation;
    return success();
}

// Rendering seams duplicate positions. Weld those positions only for the
// topology check; the original indexed triangle mesh remains unchanged.
static std::vector<CollisionPlane> closed_convex_planes(
    const Vec3 *vertices, std::uint32_t vertex_count,
    const std::vector<std::uint32_t> &indices) {
    std::map<std::array<float, 3>, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(vertex_count);
    Vec3 center{};
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
        const Vec3 p = vertices[vertex];
        const auto entry = welded.emplace(std::array<float, 3>{p.x, p.y, p.z},
            static_cast<std::uint32_t>(welded.size()));
        remap[vertex] = entry.first->second;
        if (entry.second) center = add(center, p);
    }
    if (welded.size() < 4U) return {};
    center = multiply(center, 1.0F / static_cast<float>(welded.size()));
    std::unordered_map<std::uint64_t, std::uint32_t> edges;
    for (std::size_t index = 0U; index < indices.size(); index += 3U)
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const auto a = remap[indices[index + edge]];
            const auto b = remap[indices[index + (edge + 1U) % 3U]];
            ++edges[(static_cast<std::uint64_t>(std::min(a, b)) << 32U) |
                     std::max(a, b)];
        }
    for (const auto &edge : edges)
        if (edge.second != 2U) return {};
    float scale = 0.0F;
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
        scale = std::max(scale, vector_length(subtract(vertices[vertex], center)));
    const float tolerance = std::max(1.0e-7F, scale * 1.0e-5F);
    std::vector<CollisionPlane> planes;
    planes.reserve(indices.size() / 3U);
    for (std::size_t index = 0U; index < indices.size(); index += 3U) {
        const Vec3 a = vertices[indices[index]];
        Vec3 normal = normalized_or(cross(
            subtract(vertices[indices[index + 1U]], a),
            subtract(vertices[indices[index + 2U]], a)), {});
        float side = dot(normal, subtract(a, center));
        if (fabsf(side) <= tolerance) return {};
        if (side < 0.0F) normal = multiply(normal, -1.0F);
        const float offset = dot(normal, a);
        for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
            if (dot(normal, vertices[vertex]) - offset > tolerance) return {};
        planes.push_back({normal, offset});
    }
    return planes;
}

Status World::add_triangle_mesh(
    DeviceSpan<const Vec3> vertices,
    DeviceSpan<const std::uint32_t> triangle_indices, TriangleMeshId &output,
    cudaStream_t stream) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (vertices.data == nullptr || triangle_indices.data == nullptr ||
        vertices.size < 3U || triangle_indices.size < 3U ||
        triangle_indices.size % 3U != 0U ||
        vertices.size > std::numeric_limits<std::uint32_t>::max() ||
        triangle_indices.size > std::numeric_limits<std::uint32_t>::max()) {
        return failure(StatusCode::invalid_argument,
                       "triangle mesh requires device vertices and triangle indices");
    }
    if (impl_->triangle_mesh_count >= impl_->options.triangle_mesh_capacity) {
        return failure(StatusCode::capacity_exceeded,
                       "triangle mesh capacity is exhausted");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0;
         index < impl_->options.triangle_mesh_capacity; ++index) {
        if (!impl_->meshes[index].alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense) {
        return failure(StatusCode::internal_error,
                       "no free triangle mesh slot was found");
    }

    Vec3 *owned_vertices = nullptr;
    std::uint32_t *owned_indices = nullptr;
    status = allocate_managed(owned_vertices,
                              static_cast<std::size_t>(vertices.size));
    if (!status) {
        return status;
    }
    status = allocate_managed(owned_indices,
                              static_cast<std::size_t>(triangle_indices.size));
    if (!status) {
        release_managed(owned_vertices);
        return status;
    }
    cudaError_t error = cudaMemcpyAsync(
        owned_vertices, vertices.data, sizeof(Vec3) * vertices.size,
        cudaMemcpyDefault, stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(owned_indices, triangle_indices.data,
                                sizeof(std::uint32_t) * triangle_indices.size,
                                cudaMemcpyDefault, stream);
    }
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error != cudaSuccess) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return cuda_failure(error, "triangle mesh upload failed");
    }
    for (std::uint64_t index = 0; index < vertices.size; ++index) {
        if (!finite(owned_vertices[index])) {
            release_managed(owned_indices);
            release_managed(owned_vertices);
            return failure(StatusCode::invalid_argument,
                           "triangle mesh contains a non-finite vertex");
        }
    }

    Vec3 minimum = owned_vertices[0];
    Vec3 maximum = owned_vertices[0];
    for (std::uint64_t index = 1; index < vertices.size; ++index) {
        minimum = component_min(minimum, owned_vertices[index]);
        maximum = component_max(maximum, owned_vertices[index]);
    }
    for (std::uint64_t index = 0; index < triangle_indices.size; index += 3U) {
        const std::uint32_t first = owned_indices[index];
        const std::uint32_t second = owned_indices[index + 1U];
        const std::uint32_t third = owned_indices[index + 2U];
        if (first >= vertices.size || second >= vertices.size ||
            third >= vertices.size) {
            release_managed(owned_indices);
            release_managed(owned_vertices);
            return failure(StatusCode::invalid_argument,
                           "triangle mesh index is outside the vertex buffer");
        }
        const Vec3 area = cross(subtract(owned_vertices[second],
                                         owned_vertices[first]),
                                subtract(owned_vertices[third],
                                         owned_vertices[first]));
        if (length_squared(area) <= k_epsilon * k_epsilon) {
            release_managed(owned_indices);
            release_managed(owned_vertices);
            return failure(StatusCode::invalid_argument,
                           "triangle mesh contains a degenerate triangle");
        }
    }

    std::vector<std::uint32_t> triangle_order;
    std::vector<BvhNode> bvh_nodes;
    std::vector<std::uint32_t> reordered_indices;
    try {
        const std::uint32_t triangle_count =
            static_cast<std::uint32_t>(triangle_indices.size / 3U);
        triangle_order.resize(triangle_count);
        std::iota(triangle_order.begin(), triangle_order.end(), 0U);
        bvh_nodes.reserve(triangle_count * 2U);
        const auto coordinate = [](Vec3 value, int axis) {
            return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
        };
        const auto centroid = [&](std::uint32_t triangle) {
            const std::uint32_t offset = triangle * 3U;
            return multiply(
                add(add(owned_vertices[owned_indices[offset]],
                        owned_vertices[owned_indices[offset + 1U]]),
                    owned_vertices[owned_indices[offset + 2U]]),
                1.0F / 3.0F);
        };
        std::function<std::uint32_t(std::uint32_t, std::uint32_t)> build =
            [&](std::uint32_t begin, std::uint32_t end) {
                BvhNode node{};
                node.minimum = {FLT_MAX, FLT_MAX, FLT_MAX};
                node.maximum = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (std::uint32_t item = begin; item < end; ++item) {
                    const std::uint32_t offset = triangle_order[item] * 3U;
                    for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                        const Vec3 vertex =
                            owned_vertices[owned_indices[offset + corner]];
                        node.minimum = component_min(node.minimum, vertex);
                        node.maximum = component_max(node.maximum, vertex);
                    }
                }
                const std::uint32_t node_index =
                    static_cast<std::uint32_t>(bvh_nodes.size());
                bvh_nodes.push_back(node);
                if (end - begin <= 4U) {
                    bvh_nodes[node_index].first_triangle = begin;
                    bvh_nodes[node_index].triangle_count = end - begin;
                    return node_index;
                }
                const Vec3 extent = subtract(node.maximum, node.minimum);
                const int axis = extent.x >= extent.y && extent.x >= extent.z
                                     ? 0
                                     : (extent.y >= extent.z ? 1 : 2);
                std::stable_sort(
                    triangle_order.begin() + begin, triangle_order.begin() + end,
                    [&](std::uint32_t first, std::uint32_t second) {
                        const float first_value = coordinate(centroid(first), axis);
                        const float second_value = coordinate(centroid(second), axis);
                        return first_value < second_value ||
                               (first_value == second_value && first < second);
                    });
                const std::uint32_t middle = begin + (end - begin) / 2U;
                const std::uint32_t left = build(begin, middle);
                const std::uint32_t right = build(middle, end);
                bvh_nodes[node_index].left = left;
                bvh_nodes[node_index].right = right;
                return node_index;
            };
        (void)build(0U, triangle_count);
        reordered_indices.resize(triangle_indices.size);
        for (std::uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
            const std::uint32_t source = triangle_order[triangle] * 3U;
            const std::uint32_t destination = triangle * 3U;
            reordered_indices[destination] = owned_indices[source];
            reordered_indices[destination + 1U] = owned_indices[source + 1U];
            reordered_indices[destination + 2U] = owned_indices[source + 2U];
        }
    } catch (...) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return failure(StatusCode::out_of_memory,
                       "failed to build triangle mesh acceleration data");
    }

    std::vector<std::uint32_t> bvh_leaves;
    std::vector<CollisionPlane> solid_planes;
    try {
        solid_planes = closed_convex_planes(owned_vertices,
            static_cast<std::uint32_t>(vertices.size), reordered_indices);
        for (std::uint32_t index = 0U; index < bvh_nodes.size(); ++index) {
            if (bvh_nodes[index].triangle_count != 0U) {
                bvh_leaves.push_back(index);
            }
        }
    } catch (...) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return failure(StatusCode::out_of_memory,
                       "failed to index triangle mesh BVH leaves");
    }

    BvhNode *owned_bvh_nodes = nullptr;
    status = allocate_managed(owned_bvh_nodes, bvh_nodes.size());
    if (!status) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    std::uint32_t *owned_bvh_leaves = nullptr;
    status = allocate_managed(owned_bvh_leaves, bvh_leaves.size());
    if (!status) {
        release_managed(owned_bvh_nodes);
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    std::copy(reordered_indices.begin(), reordered_indices.end(), owned_indices);
    std::copy(bvh_nodes.begin(), bvh_nodes.end(), owned_bvh_nodes);
    std::copy(bvh_leaves.begin(), bvh_leaves.end(), owned_bvh_leaves);

    CollisionPlane *owned_solid_planes = nullptr;
    status = allocate_managed(owned_solid_planes, solid_planes.size());
    if (!status) {
        release_managed(owned_bvh_leaves);
        release_managed(owned_bvh_nodes);
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    if (!solid_planes.empty())
        std::copy(solid_planes.begin(), solid_planes.end(), owned_solid_planes);

    TriangleMeshResource &mesh = impl_->meshes[slot];
    mesh.vertices = owned_vertices;
    mesh.indices = owned_indices;
    mesh.vertex_count = static_cast<std::uint32_t>(vertices.size);
    mesh.index_count = static_cast<std::uint32_t>(triangle_indices.size);
    mesh.minimum = minimum;
    mesh.maximum = maximum;
    mesh.bounding_center = multiply(add(minimum, maximum), 0.5F);
    float squared_radius = 0.0F;
    for (std::uint64_t index = 0U; index < vertices.size; ++index) {
        squared_radius = fmaxf(
            squared_radius,
            length_squared(subtract(owned_vertices[index],
                                    mesh.bounding_center)));
    }
    mesh.bounding_radius = sqrtf(squared_radius);
    const Vec3 half_extents = multiply(subtract(maximum, minimum), 0.5F);
    mesh.unit_inertia = {
        fmaxf((half_extents.y * half_extents.y +
               half_extents.z * half_extents.z) /
                  3.0F,
              k_epsilon),
        fmaxf((half_extents.x * half_extents.x +
               half_extents.z * half_extents.z) /
                  3.0F,
              k_epsilon),
        fmaxf((half_extents.x * half_extents.x +
               half_extents.y * half_extents.y) /
                  3.0F,
              k_epsilon)};
    mesh.bvh_nodes = owned_bvh_nodes;
    mesh.bvh_node_count = static_cast<std::uint32_t>(bvh_nodes.size());
    mesh.bvh_leaves = owned_bvh_leaves;
    mesh.bvh_leaf_count = static_cast<std::uint32_t>(bvh_leaves.size());
    mesh.solid_planes = owned_solid_planes;
    mesh.alive = true;
    ++impl_->triangle_mesh_count;
    ++impl_->revision;
    output = {slot, mesh.generation};
    return success();
}

Status World::remove_triangle_mesh(TriangleMeshId mesh_id) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    status = impl_->validate_handle(mesh_id);
    if (!status) {
        return status;
    }
    for (std::uint32_t index = 0; index < impl_->rigid_body_count; ++index) {
        if (impl_->parameters[index].mesh == mesh_id) {
            return failure(StatusCode::invalid_argument,
                           "triangle mesh is still referenced by a rigid body");
        }
    }
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index)
        if (impl_->paint_fields[index].alive &&
            impl_->paint_fields[index].options.mesh == mesh_id)
            return failure(StatusCode::invalid_argument,
                           "triangle mesh is still referenced by a paint field");
    TriangleMeshResource &mesh = impl_->meshes[mesh_id.index];
    release_managed(mesh.bvh_leaves);
    release_managed(mesh.bvh_nodes);
    release_managed(mesh.solid_planes);
    release_managed(mesh.indices);
    release_managed(mesh.vertices);
    mesh.vertex_count = 0U;
    mesh.index_count = 0U;
    mesh.bvh_node_count = 0U;
    mesh.bvh_leaf_count = 0U;
    mesh.alive = false;
    ++mesh.generation;
    if (mesh.generation == 0U) {
        mesh.generation = 1U;
    }
    --impl_->triangle_mesh_count;
    ++impl_->revision;
    return success();
}

Status World::add_paint_field(PaintFieldOptions options,
                              PaintFieldId &output,
                              cudaStream_t stream) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    const bool cloth_target = options.cloth.generation != 0U;
    std::uint32_t vertex_count = 0U;
    if (cloth_target) {
        if (options.body.generation != 0U || options.mesh.generation != 0U ||
            options.cloth.index >= impl_->cloths.size() ||
            !impl_->cloths[options.cloth.index] ||
            !impl_->cloths[options.cloth.index]->alive ||
            impl_->cloths[options.cloth.index]->generation != options.cloth.generation)
            return failure(StatusCode::invalid_handle, "paint cloth target is stale");
        const auto &cloth = *impl_->cloths[options.cloth.index];
        vertex_count = cloth.source_indices
            ? static_cast<std::uint32_t>(cloth.source_inverse_masses.size())
            : cloth.vertex_count;
    } else {
        std::uint32_t dense = 0U;
        if (!(status = impl_->validate_handle(options.body, dense)) ||
            !(status = impl_->validate_handle(options.mesh))) return status;
        vertex_count = impl_->meshes[options.mesh.index].vertex_count;
    }
    if (options.vertex_uvs.data == nullptr ||
        options.vertex_uvs.size != vertex_count ||
        options.width == 0U || options.height == 0U ||
        options.width > 4096U || options.height > 4096U ||
        static_cast<std::uint64_t>(options.width) * options.height >
            16'777'216U)
        return failure(StatusCode::invalid_argument,
                       "paint field UV count or dimensions are invalid");
    std::uint32_t slot = 0U;
    for (; slot < impl_->options.paint_field_capacity; ++slot)
        if (!impl_->paint_fields[slot].alive) break;
    if (slot == impl_->options.paint_field_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "paint field capacity exhausted");
    Vec2 *uvs = nullptr;
    std::uint32_t *pixels = nullptr;
    status = allocate_managed(uvs, vertex_count);
    if (!status) return status;
    status = allocate_managed(pixels,
        static_cast<std::size_t>(options.width) * options.height);
    if (!status) { release_managed(uvs); return status; }
    cudaError_t error = cudaMemcpyAsync(uvs, options.vertex_uvs.data,
        vertex_count * sizeof(Vec2), cudaMemcpyDefault, stream);
    if (error == cudaSuccess)
        error = cudaMemsetAsync(pixels, 0,
            static_cast<std::size_t>(options.width) * options.height *
                sizeof(std::uint32_t), stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        release_managed(pixels);
        release_managed(uvs);
        return cuda_failure(error, "paint field upload failed");
    }
    for (std::uint32_t i = 0; i < vertex_count; ++i)
        if (!finite(uvs[i].x) || !finite(uvs[i].y)) {
            release_managed(pixels);
            release_managed(uvs);
            return failure(StatusCode::invalid_argument,
                           "paint field contains non-finite UVs");
        }
    PaintFieldResource &field = impl_->paint_fields[slot];
    options.vertex_uvs = {uvs, vertex_count};
    field.options = options;
    field.uvs = uvs;
    field.pixels = pixels;
    field.alive = true;
    output = {slot, field.generation};
    ++impl_->revision;
    return success();
}

Status World::remove_paint_field(PaintFieldId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    for (std::uint32_t i = 0; i < impl_->options.paint_rule_capacity; ++i)
        if (impl_->paint_rules[i].alive &&
            impl_->paint_rules[i].options.target == id)
            return failure(StatusCode::invalid_argument,
                           "paint field is still referenced by a paint rule");
    PaintFieldResource &field = impl_->paint_fields[id.index];
    release_managed(field.pixels);
    release_managed(field.uvs);
    field.alive = false;
    ++field.generation;
    if (field.generation == 0U) field.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::clear_paint_field(PaintFieldId id, cudaStream_t stream) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    const PaintFieldResource &field = impl_->paint_fields[id.index];
    cudaError_t error = cudaMemsetAsync(field.pixels, 0,
        static_cast<std::size_t>(field.options.width) *
        field.options.height * sizeof(std::uint32_t), stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        return cuda_failure(error, "paint field clear failed");
    return success();
}

Status World::paint_field_view(PaintFieldId id,
                               PaintFieldDeviceView &output) const noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    const PaintFieldResource &field = impl_->paint_fields[id.index];
    output = {{field.pixels, static_cast<std::uint64_t>(field.options.width) *
                                 field.options.height},
              field.options.width, field.options.height, impl_->revision};
    return success();
}

Status World::add_paint_rule(PaintRuleOptions options,
                             PaintRuleId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    const bool fluid_source = options.source.generation != 0U;
    const bool rigid_source = options.rigid_source.generation != 0U;
    if (fluid_source == rigid_source)
        return failure(StatusCode::invalid_argument,
                       "paint rule needs exactly one source");
    if (fluid_source) {
        FluidStorage *fluid = nullptr;
        if (!(status = impl_->validate_handle(options.source, fluid))) return status;
    } else {
        std::uint32_t dense = 0U;
        if (!(status = impl_->validate_handle(options.rigid_source, dense)))
            return status;
    }
    if (options.target.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[options.target.index].alive ||
        impl_->paint_fields[options.target.index].generation !=
            options.target.generation)
        return failure(StatusCode::invalid_handle, "paint target field is stale");
    const PaintFieldResource &target =
        impl_->paint_fields[options.target.index];
    if ((fluid_source && target.options.cloth.generation != 0U) ||
        (rigid_source && target.options.cloth.generation == 0U))
        return failure(StatusCode::not_supported,
                       "paint source and target systems do not match");
    if (!finite(options.reach) || options.reach < 0.0F ||
        options.reach > 10.0F)
        return failure(StatusCode::invalid_argument, "paint reach is invalid");
    if (!finite(options.brush_radius) || options.brush_radius <= 0.0F ||
        options.brush_radius > 10.0F)
        return failure(StatusCode::invalid_argument,
                       "paint brush radius is invalid");
    std::uint32_t slot = 0U;
    for (; slot < impl_->options.paint_rule_capacity; ++slot)
        if (!impl_->paint_rules[slot].alive) break;
    if (slot == impl_->options.paint_rule_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "paint rule capacity exhausted");
    PaintRuleResource &rule = impl_->paint_rules[slot];
    rule.options = options;
    rule.alive = true;
    ++impl_->paint_rule_count;
    output = {slot, rule.generation};
    ++impl_->revision;
    return success();
}

Status World::remove_paint_rule(PaintRuleId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_rule_capacity ||
        !impl_->paint_rules[id.index].alive ||
        impl_->paint_rules[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint rule handle is stale");
    PaintRuleResource &rule = impl_->paint_rules[id.index];
    rule.alive = false;
    --impl_->paint_rule_count;
    ++rule.generation;
    if (rule.generation == 0U) rule.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::add_cloth(ClothOptions options, ClothId &output,
                        cudaStream_t stream) noexcept {
    (void)stream;
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.vertices.data == nullptr || options.vertices.size < 3U ||
        options.vertices.size > UINT32_MAX ||
        options.triangle_indices.data == nullptr ||
        options.triangle_indices.size == 0U ||
        options.triangle_indices.size % 3U != 0U ||
        options.triangle_indices.size > UINT32_MAX ||
        (options.inverse_masses.size != 0U &&
         (options.inverse_masses.data == nullptr ||
          options.inverse_masses.size != options.vertices.size)) ||
        !finite(options.vertex_mass) || options.vertex_mass <= 0.0F ||
        !finite(options.thickness) || options.thickness <= 0.0F ||
        !finite(options.stretch_compliance) || options.stretch_compliance < 0.0F ||
        !finite(options.bending_compliance) || options.bending_compliance < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.contact_friction) || options.contact_friction < 0.0F ||
        !finite(options.break_strain) || options.break_strain < 0.0F ||
        options.break_strain > 9.0F ||
        !finite(options.impact_break_impulse) ||
        options.impact_break_impulse < 0.0F ||
        !finite(options.target_volume) || options.target_volume < 0.0F ||
        !finite(options.volume_compliance) ||
        options.volume_compliance < 0.0F ||
        (options.preserve_volume &&
         (options.break_strain > 0.0F || options.impact_break_impulse > 0.0F)) ||
        options.fracture_persistence_substeps == 0U ||
        options.fracture_persistence_substeps > 64U ||
        options.solver_iterations == 0U || options.solver_iterations > 64U) {
        return failure(StatusCode::invalid_argument, "invalid cloth geometry or solver options");
    }
    for (std::uint64_t vertex = 0U; vertex < options.vertices.size; ++vertex) {
        if (!finite(options.vertices.data[vertex]) ||
            (options.inverse_masses.size != 0U &&
             (!finite(options.inverse_masses.data[vertex]) ||
              options.inverse_masses.data[vertex] < 0.0F)))
            return failure(StatusCode::invalid_argument, "invalid cloth vertex or inverse mass");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
        if (!impl_->cloths[index] || !impl_->cloths[index]->alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense)
        return failure(StatusCode::capacity_exceeded, "cloth capacity is exhausted");

    const std::uint32_t count = static_cast<std::uint32_t>(options.vertices.size);
    const bool fracture_enabled = options.break_strain > 0.0F ||
        options.impact_break_impulse > 0.0F;
    if (fracture_enabled && options.vertices.size + options.triangle_indices.size > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth split capacity exceeds uint32 range");
    const auto capacity = count + (fracture_enabled
        ? static_cast<std::uint32_t>(options.triangle_indices.size) : 0U);
    std::vector<std::vector<DeformableNeighbor>> adjacency;
    std::vector<std::uint32_t> offsets;
    std::vector<DeformableNeighbor> neighbors;
    std::vector<ClothBond> bonds;
    std::vector<Vec3> surface_rest;
    std::vector<std::uint32_t> surface_indices;
    std::vector<std::uint32_t> triangle_bonds;
    float initial_signed_volume = 0.0F;
    try {
        adjacency.resize(count);
        std::unordered_map<std::uint64_t, std::uint32_t> bond_ids;
        std::unordered_map<std::uint64_t, std::uint32_t> opposite;
        std::unordered_map<std::uint64_t, std::uint32_t> edge_counts;
        const auto key = [](std::uint32_t a, std::uint32_t b) {
            if (a > b) std::swap(a, b);
            return (static_cast<std::uint64_t>(a) << 32U) | b;
        };
        const auto link = [&](std::uint32_t a, std::uint32_t b,
                              float compliance, bool bending) {
            if (a == b || bond_ids.find(key(a, b)) != bond_ids.end()) return;
            const float rest = vector_length(subtract(
                options.vertices.data[a], options.vertices.data[b]));
            if (rest <= 1.0e-6F) return;
            const auto id = static_cast<std::uint32_t>(bonds.size());
            bond_ids.emplace(key(a, b), id);
            bonds.push_back({a, b, rest, bending});
            adjacency[a].push_back({b, rest, compliance, id});
            adjacency[b].push_back({a, rest, compliance, id});
        };
        for (std::uint64_t triangle = 0U;
             triangle < options.triangle_indices.size; triangle += 3U) {
            const std::uint32_t a = options.triangle_indices.data[triangle];
            const std::uint32_t b = options.triangle_indices.data[triangle + 1U];
            const std::uint32_t c = options.triangle_indices.data[triangle + 2U];
            if (a >= count || b >= count || c >= count || a == b || b == c || c == a)
                return failure(StatusCode::invalid_argument, "invalid cloth triangle index");
            if (vector_length(subtract(options.vertices.data[a],
                                       options.vertices.data[b])) <= 1.0e-6F ||
                vector_length(subtract(options.vertices.data[b],
                                       options.vertices.data[c])) <= 1.0e-6F ||
                vector_length(subtract(options.vertices.data[c],
                                       options.vertices.data[a])) <= 1.0e-6F)
                return failure(StatusCode::invalid_argument,
                               "cloth triangle has a zero-length edge");
            initial_signed_volume += dot(options.vertices.data[a], cross(
                options.vertices.data[b], options.vertices.data[c])) / 6.0F;
            const std::array<std::array<std::uint32_t, 3>, 3> edges{{
                {a, b, c}, {b, c, a}, {c, a, b}}};
            for (const auto &edge : edges) {
                const std::uint64_t edge_key = key(edge[0], edge[1]);
                ++edge_counts[edge_key];
                const auto previous = opposite.find(edge_key);
                if (previous == opposite.end()) {
                    opposite.emplace(edge_key, edge[2]);
                    link(edge[0], edge[1], options.stretch_compliance, false);
                } else {
                    link(previous->second, edge[2], options.bending_compliance,
                         true);
                }
            }
        }
        if (options.preserve_volume) {
            if (fabsf(initial_signed_volume) <= 1.0e-8F ||
                std::any_of(edge_counts.begin(), edge_counts.end(),
                    [](const auto &edge) { return edge.second != 2U; }))
                return failure(StatusCode::invalid_argument,
                    "volume-preserving cloth must be a closed manifold");
        }
        for (std::uint64_t triangle = 0U;
             triangle < options.triangle_indices.size; triangle += 3U) {
            const std::uint32_t corners[3]{
                options.triangle_indices.data[triangle],
                options.triangle_indices.data[triangle + 1U],
                options.triangle_indices.data[triangle + 2U]};
            for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
                surface_rest.push_back(options.vertices.data[corners[corner]]);
                surface_indices.push_back(static_cast<std::uint32_t>(triangle + corner));
                triangle_bonds.push_back(bond_ids.at(key(corners[corner],
                    corners[(corner + 1U) % 3U])));
            }
        }
        offsets.reserve(static_cast<std::size_t>(count) + 1U);
        offsets.push_back(0U);
        for (const auto &list : adjacency) {
            if (neighbors.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded, "cloth links exceed uint32 range");
            neighbors.insert(neighbors.end(), list.begin(), list.end());
            offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory, "failed to build cloth links");
    }
    std::unique_ptr<ClothStorage> cloth;
    try { cloth = std::make_unique<ClothStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "failed to allocate cloth"); }
    cloth->generation = impl_->cloths[slot] ? impl_->cloths[slot]->generation : 1U;
    cloth->vertex_count = count;
    cloth->vertex_capacity = capacity;
    cloth->stretch_compliance = options.stretch_compliance;
    cloth->bending_compliance = options.bending_compliance;
    cloth->index_count = static_cast<std::uint32_t>(options.triangle_indices.size);
    cloth->neighbor_count = static_cast<std::uint32_t>(neighbors.size());
    cloth->neighbor_capacity = fracture_enabled
        ? 2U * (options.triangle_indices.size + bonds.size()) : neighbors.size();
    if (cloth->neighbor_capacity > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth split links exceed uint32 range");
    if (bonds.size() > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth bonds exceed uint32 range");
    cloth->bond_count = static_cast<std::uint32_t>(bonds.size());
    cloth->thickness = options.thickness;
    cloth->velocity_damping = options.velocity_damping;
    cloth->contact_friction = options.contact_friction;
    cloth->break_strain = options.break_strain;
    cloth->fracture_persistence_substeps =
        options.fracture_persistence_substeps;
    cloth->impact_break_impulse = options.impact_break_impulse;
    cloth->solver_iterations = options.solver_iterations;
    cloth->preserve_volume = options.preserve_volume;
    cloth->target_volume = options.target_volume > 0.0F
        ? options.target_volume : fabsf(initial_signed_volume);
    cloth->volume_compliance = options.volume_compliance;
    cloth->orientation = initial_signed_volume < 0.0F ? -1.0F : 1.0F;
    status = allocate_managed(cloth->positions, capacity); if (!status) return status;
    status = allocate_managed(cloth->scratch, capacity); if (!status) return status;
    status = allocate_managed(cloth->previous, capacity); if (!status) return status;
    status = allocate_managed(cloth->velocities, capacity); if (!status) return status;
    status = allocate_managed(cloth->inverse_masses, capacity); if (!status) return status;
    status = allocate_managed(cloth->indices, cloth->index_count); if (!status) return status;
    if (fracture_enabled) {
        status = allocate_managed(cloth->source_indices, cloth->index_count);
        if (!status) return status;
        status = allocate_managed(cloth->vertex_sources, capacity);
        if (!status) return status;
        status = allocate_managed(cloth->free_triangle_nodes, capacity);
        if (!status) return status;
        status = allocate_managed(cloth->surface_positions, surface_rest.size());
        if (!status) return status;
        status = allocate_managed(cloth->surface_triangle_indices,
                                  surface_indices.size());
        if (!status) return status;
        status = allocate_managed(cloth->triangle_bonds, triangle_bonds.size());
        if (!status) return status;
    }
    status = allocate_managed(cloth->bonds, bonds.size()); if (!status) return status;
    status = allocate_managed(cloth->bond_active, bonds.size());
    if (!status) return status;
    status = allocate_managed(cloth->bond_damage, bonds.size());
    if (!status) return status;
    status = allocate_managed(cloth->offsets, static_cast<std::size_t>(capacity) + 1U); if (!status) return status;
    status = allocate_managed(cloth->neighbors, cloth->neighbor_capacity); if (!status) return status;
    status = allocate_managed(cloth->body_impulses, capacity); if (!status) return status;
    status = allocate_managed(cloth->rigid_contact_forces, capacity);
    if (!status) return status;
    status = allocate_managed(cloth->soft_body_forces, capacity);
    if (!status) return status;
    status = allocate_managed(cloth->body_corrections,
                              impl_->options.rigid_body_capacity);
    if (!status) return status;
    if (options.preserve_volume) {
        status = allocate_managed(cloth->volume_gradients, count);
        if (!status) return status;
        status = allocate_managed(cloth->volume_lambda, 1U);
        if (!status) return status;
        *cloth->volume_lambda = 0.0F;
    }
    if (impl_->options.fluid_cloth_coupling_capacity != 0U) {
        status = allocate_managed(cloth->fluid_forces, capacity);
        if (!status) return status;
    }
    status = allocate_managed(cloth->count, 1U); if (!status) return status;
    for (std::uint32_t index = 0U; index < count; ++index) {
        cloth->positions[index] = options.vertices.data[index];
        cloth->scratch[index] = options.vertices.data[index];
        cloth->previous[index] = options.vertices.data[index];
        cloth->velocities[index] = {};
        cloth->rigid_contact_forces[index] = {};
        cloth->soft_body_forces[index] = {};
        if (cloth->fluid_forces != nullptr) cloth->fluid_forces[index] = {};
        cloth->inverse_masses[index] = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[index] : 1.0F / options.vertex_mass;
    }
    std::copy_n(options.triangle_indices.data, cloth->index_count, cloth->indices);
    if (fracture_enabled) {
        std::copy_n(options.triangle_indices.data, cloth->index_count, cloth->source_indices);
        for (std::uint32_t i = 0; i < count; ++i) cloth->vertex_sources[i] = i;
        std::copy(surface_rest.begin(), surface_rest.end(), cloth->surface_positions);
        std::copy(surface_indices.begin(), surface_indices.end(),
                  cloth->surface_triangle_indices);
        std::copy(triangle_bonds.begin(), triangle_bonds.end(),
                  cloth->triangle_bonds);
    }
    std::copy(bonds.begin(), bonds.end(), cloth->bonds);
    std::fill_n(cloth->bond_active, bonds.size(), std::uint8_t{1U});
    std::fill_n(cloth->bond_damage, bonds.size(), std::uint8_t{0U});
    std::copy(offsets.begin(), offsets.end(), cloth->offsets);
    std::copy(neighbors.begin(), neighbors.end(), cloth->neighbors);
    *cloth->count = count;
    if (fracture_enabled) {
        try {
            cloth->source_inverse_masses.assign(cloth->inverse_masses, cloth->inverse_masses + count);
            cloth->source_degrees.resize(count);
            cloth->bond_corners.resize(bonds.size());
            std::unordered_map<std::uint64_t, std::uint32_t> ids, first_corners;
            const auto key = [](std::uint32_t a, std::uint32_t b) {
                return (static_cast<std::uint64_t>(std::min(a,b)) << 32U) | std::max(a,b);
            };
            for (std::uint32_t i = 0; i < bonds.size(); ++i)
                ids[key(bonds[i].first, bonds[i].second)] = i;
            for (std::uint32_t c = 0; c < cloth->index_count; ++c) {
                ++cloth->source_degrees[cloth->source_indices[c]];
                const auto next = c / 3U * 3U + (c + 1U) % 3U;
                const auto other = c / 3U * 3U + (c + 2U) % 3U;
                const auto a = cloth->source_indices[c], b = cloth->source_indices[next];
                const auto edge_key = key(a,b);
                auto [it, fresh] = first_corners.emplace(edge_key,c);
                if (fresh) cloth->bond_corners[ids.at(edge_key)] = {c,next};
                else {
                    const auto previous = it->second;
                    const auto previous_next = previous / 3U * 3U + (previous + 1U) % 3U;
                    const auto opposite = previous / 3U * 3U + (previous + 2U) % 3U;
                    const auto bending = ids.find(key(cloth->source_indices[opposite],
                                                     cloth->source_indices[other]));
                    const bool has_bend = bending != ids.end() && bonds[bending->second].bending;
                    if (has_bend) cloth->bond_corners[bending->second] = {opposite,other};
                    const bool same = cloth->source_indices[previous] == a;
                    cloth->seams.push_back({{previous,previous_next,
                        same ? c : next, same ? next : c}, ids.at(edge_key),
                        has_bend ? bending->second : k_invalid_dense});
                }
            }
        } catch (...) { return failure(StatusCode::out_of_memory, "failed to build cloth seams"); }
        status = rebuild_cloth_topology(*cloth, true);
        if (!status) return status;
    }
    cloth->alive = true;
    output = {slot, cloth->generation};
    impl_->cloths[slot] = std::move(cloth);
    ++impl_->revision;
    return success();
}

Status World::remove_cloth(ClothId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->cloths.size() || !impl_->cloths[id.index] ||
        !impl_->cloths[id.index]->alive ||
        impl_->cloths[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "cloth handle is stale");
    for (std::uint32_t field = 0U;
         field < impl_->options.paint_field_capacity; ++field)
        if (impl_->paint_fields[field].alive &&
            impl_->paint_fields[field].options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a paint field");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.cloth == id)
            return failure(StatusCode::invalid_argument,
                "cloth is still referenced by a fluid coupling");
    ClothStorage &cloth = *impl_->cloths[id.index];
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a soft-body coupling");
    cloth.alive = false;
    cloth.release();
    ++cloth.generation;
    if (cloth.generation == 0U) cloth.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::cloth_view(ClothId id, ClothDeviceView &output) const noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->cloths.size() || !impl_->cloths[id.index] ||
        !impl_->cloths[id.index]->alive ||
        impl_->cloths[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "cloth handle is stale");
    const ClothStorage &cloth = *impl_->cloths[id.index];
    output.positions = {cloth.positions, cloth.vertex_count};
    output.velocities = {cloth.velocities, cloth.vertex_count};
    output.triangle_indices = {cloth.indices, cloth.index_count};
    output.vertex_count = cloth.vertex_count;
    if (cloth.surface_positions != nullptr) {
        output.surface_positions = {cloth.surface_positions, cloth.index_count};
        output.surface_triangle_indices = {
            cloth.surface_triangle_indices, cloth.index_count};
        output.surface_source_indices = {cloth.source_indices, cloth.index_count};
        output.vertex_source_indices = {cloth.vertex_sources, cloth.vertex_count};
        output.inverse_masses = {cloth.inverse_masses, cloth.vertex_count};
    }
    output.bonds = {cloth.bonds, cloth.bond_count};
    output.active_bonds = {cloth.bond_active, cloth.bond_count};
    output.rigid_contact_forces = {
        cloth.rigid_contact_forces, cloth.vertex_count};
    output.soft_body_contact_forces = {cloth.soft_body_forces, cloth.vertex_count};
    if (cloth.fluid_forces != nullptr)
        output.fluid_contact_forces = {
            cloth.fluid_forces, cloth.vertex_count};
    return success();
}

Status World::add_soft_body(SoftBodyOptions options, SoftBodyId &output,
                            cudaStream_t stream) noexcept {
    (void)stream;
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.nodes.data == nullptr || options.nodes.size < 4U ||
        options.nodes.size > UINT32_MAX || options.bonds.data == nullptr ||
        options.bonds.size == 0U || options.bonds.size > UINT32_MAX ||
        options.surface_vertices.data == nullptr ||
        options.surface_vertices.size < 3U ||
        options.surface_vertices.size > UINT32_MAX ||
        options.surface_triangle_indices.data == nullptr ||
        options.surface_triangle_indices.size == 0U ||
        options.surface_triangle_indices.size % 3U != 0U ||
        options.surface_triangle_indices.size > UINT32_MAX ||
        options.surface_bindings.data == nullptr ||
        options.surface_bindings.size != options.surface_vertices.size ||
        (options.inverse_masses.size != 0U &&
         (options.inverse_masses.data == nullptr ||
          options.inverse_masses.size != options.nodes.size)) ||
        !finite(options.node_mass) || options.node_mass <= 0.0F ||
        !finite(options.node_radius) || options.node_radius <= 0.0F ||
        !finite(options.stretch_compliance) ||
        options.stretch_compliance < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.spring_damping) || options.spring_damping < 0.0F ||
        options.spring_damping > 1.0F ||
        !finite(options.contact_friction) || options.contact_friction < 0.0F ||
        !finite(options.shape_matching_stiffness) ||
        options.shape_matching_stiffness < 0.0F ||
        options.shape_matching_stiffness > 1.0F ||
        !finite(options.maximum_projection_fraction) ||
        options.maximum_projection_fraction <= 0.0F ||
        options.maximum_projection_fraction > 1.0F ||
        !finite(options.constraint_velocity_response) ||
        options.constraint_velocity_response < 0.0F ||
        options.constraint_velocity_response > 1.0F ||
        !finite(options.maximum_speed) || options.maximum_speed <= 0.0F ||
        options.solver_iterations == 0U || options.solver_iterations > 64U) {
        return failure(StatusCode::invalid_argument,
                       "invalid soft-body geometry or solver options");
    }
    const std::uint32_t node_count =
        static_cast<std::uint32_t>(options.nodes.size);
    float movable_mass = 0.0F;
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        if (!finite(options.nodes.data[node]) ||
            (options.inverse_masses.size != 0U &&
             (!finite(options.inverse_masses.data[node]) ||
              options.inverse_masses.data[node] < 0.0F)))
            return failure(StatusCode::invalid_argument,
                           "invalid soft-body node or inverse mass");
        const float inverse_mass = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
        if (inverse_mass > 0.0F) movable_mass += 1.0F / inverse_mass;
    }
    if (!finite(movable_mass) || movable_mass <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "soft body needs at least one movable node");
    for (std::uint64_t index = 0U;
         index < options.surface_triangle_indices.size; ++index) {
        if (options.surface_triangle_indices.data[index] >=
            options.surface_vertices.size)
            return failure(StatusCode::invalid_argument,
                           "soft-body surface index is out of range");
    }
    for (std::uint64_t vertex = 0U;
         vertex < options.surface_vertices.size; ++vertex) {
        if (!finite(options.surface_vertices.data[vertex]))
            return failure(StatusCode::invalid_argument,
                           "soft-body surface vertex is invalid");
        const SoftBodySurfaceBinding &binding =
            options.surface_bindings.data[vertex];
        float weight_sum = 0.0F;
        for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
            if (binding.nodes[slot] >= node_count ||
                !finite(binding.weights[slot]) || binding.weights[slot] < 0.0F)
                return failure(StatusCode::invalid_argument,
                               "soft-body surface binding is invalid");
            weight_sum += binding.weights[slot];
        }
        if (fabsf(weight_sum - 1.0F) > 1.0e-4F)
            return failure(StatusCode::invalid_argument,
                           "soft-body surface binding weights must sum to one");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
        if (!impl_->soft_bodies[index] || !impl_->soft_bodies[index]->alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense)
        return failure(StatusCode::capacity_exceeded,
                       "soft-body capacity is exhausted");

    std::vector<std::vector<DeformableNeighbor>> adjacency;
    std::vector<std::uint32_t> offsets;
    std::vector<DeformableNeighbor> neighbors;
    float minimum_bond_length = FLT_MAX;
    try {
        adjacency.resize(node_count);
        std::unordered_set<std::uint64_t> unique;
        for (std::uint32_t bond_index = 0U;
             bond_index < options.bonds.size; ++bond_index) {
            const SoftBodyBond bond = options.bonds.data[bond_index];
            if (bond.first >= node_count || bond.second >= node_count ||
                bond.first == bond.second || !finite(bond.rest_length) ||
                bond.rest_length <= 1.0e-6F)
                return failure(StatusCode::invalid_argument,
                               "soft-body bond is invalid");
            const std::uint32_t first = std::min(bond.first, bond.second);
            const std::uint32_t second = std::max(bond.first, bond.second);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(first) << 32U) | second;
            if (!unique.insert(key).second)
                return failure(StatusCode::invalid_argument,
                               "soft-body bond is duplicated");
            minimum_bond_length = std::min(
                minimum_bond_length, bond.rest_length);
            adjacency[bond.first].push_back({bond.second, bond.rest_length,
                                             options.stretch_compliance,
                                             bond_index});
            adjacency[bond.second].push_back({bond.first, bond.rest_length,
                                              options.stretch_compliance,
                                              bond_index});
        }
        offsets.reserve(static_cast<std::size_t>(node_count) + 1U);
        offsets.push_back(0U);
        for (const auto &list : adjacency) {
            if (neighbors.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded,
                               "soft-body neighbors exceed uint32 range");
            neighbors.insert(neighbors.end(), list.begin(), list.end());
            offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        }
        if (std::any_of(adjacency.begin(), adjacency.end(),
                        [](const auto &list) { return list.empty(); }))
            return failure(StatusCode::invalid_argument,
                           "every soft-body node needs at least one bond");
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to build soft-body adjacency");
    }

    Vec3 shape_rest_center{};
    ShapeMatrix shape_inverse_rest{};
    if (options.shape_matching_stiffness > 0.0F) {
        for (std::uint32_t node = 0U; node < node_count; ++node) {
            const float inverse_mass = options.inverse_masses.size != 0U
                ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
            if (inverse_mass > 0.0F)
                shape_rest_center = add(shape_rest_center,
                    multiply(options.nodes.data[node], 1.0F / inverse_mass));
        }
        shape_rest_center = multiply(shape_rest_center, 1.0F / movable_mass);
        ShapeMatrix rest_covariance{};
        for (std::uint32_t node = 0U; node < node_count; ++node) {
            const float inverse_mass = options.inverse_masses.size != 0U
                ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
            if (inverse_mass <= 0.0F) continue;
            const Vec3 rest = subtract(
                options.nodes.data[node], shape_rest_center);
            const float mass = 1.0F / inverse_mass;
            rest_covariance.columns[0] = add(
                rest_covariance.columns[0], multiply(rest, mass * rest.x));
            rest_covariance.columns[1] = add(
                rest_covariance.columns[1], multiply(rest, mass * rest.y));
            rest_covariance.columns[2] = add(
                rest_covariance.columns[2], multiply(rest, mass * rest.z));
        }
        const Vec3 row0 = cross(rest_covariance.columns[1],
                                rest_covariance.columns[2]);
        const Vec3 row1 = cross(rest_covariance.columns[2],
                                rest_covariance.columns[0]);
        const Vec3 row2 = cross(rest_covariance.columns[0],
                                rest_covariance.columns[1]);
        const float determinant = dot(rest_covariance.columns[0], row0);
        if (!finite(determinant) || fabsf(determinant) <= 1.0e-10F)
            return failure(StatusCode::invalid_argument,
                "shape-matched soft body needs a volumetric rest lattice");
        const float inverse_determinant = 1.0F / determinant;
        shape_inverse_rest.columns[0] = multiply(
            {row0.x, row1.x, row2.x}, inverse_determinant);
        shape_inverse_rest.columns[1] = multiply(
            {row0.y, row1.y, row2.y}, inverse_determinant);
        shape_inverse_rest.columns[2] = multiply(
            {row0.z, row1.z, row2.z}, inverse_determinant);
    }

    std::vector<std::uint32_t> surface_offsets;
    std::vector<SoftSurfaceInfluence> surface_influences;
    try {
        std::vector<std::vector<SoftSurfaceInfluence>> per_node(node_count);
        for (std::uint32_t corner = 0U;
             corner < options.surface_triangle_indices.size; ++corner) {
            const SoftBodySurfaceBinding binding = options.surface_bindings.data[
                options.surface_triangle_indices.data[corner]];
            float denominator = 0.0F;
            for (std::uint32_t a = 0U; a < 4U; ++a) {
                const float inverse = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[binding.nodes[a]]
                    : 1.0F / options.node_mass;
                for (std::uint32_t b = 0U; b < 4U; ++b)
                    if (binding.nodes[a] == binding.nodes[b])
                        denominator += binding.weights[a] * binding.weights[b] * inverse;
            }
            if (denominator <= 0.0F) continue;
            for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
                const std::uint32_t node = binding.nodes[slot];
                bool duplicate = false;
                for (std::uint32_t earlier = 0U; earlier < slot; ++earlier)
                    duplicate |= binding.nodes[earlier] == node;
                if (duplicate) continue;
                float weight = 0.0F;
                for (std::uint32_t other = slot; other < 4U; ++other)
                    if (binding.nodes[other] == node) weight += binding.weights[other];
                const float inverse = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
                if (weight * inverse > 0.0F)
                    per_node[node].push_back({corner, weight * inverse / denominator});
            }
        }
        surface_offsets.push_back(0U);
        for (const auto &list : per_node) {
            if (surface_influences.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded,
                               "soft-body surface influences exceed uint32 range");
            surface_influences.insert(surface_influences.end(), list.begin(), list.end());
            surface_offsets.push_back(static_cast<std::uint32_t>(surface_influences.size()));
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to build soft-body surface contact bindings");
    }

    std::unique_ptr<SoftBodyStorage> body;
    try {
        body = std::make_unique<SoftBodyStorage>();
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate soft-body storage");
    }
    body->generation = impl_->soft_bodies[slot]
        ? impl_->soft_bodies[slot]->generation : 1U;
    body->node_count = node_count;
    body->bond_count = static_cast<std::uint32_t>(options.bonds.size);
    body->neighbor_count = static_cast<std::uint32_t>(neighbors.size());
    body->surface_vertex_count =
        static_cast<std::uint32_t>(options.surface_vertices.size);
    body->surface_index_count =
        static_cast<std::uint32_t>(options.surface_triangle_indices.size);
    body->node_radius = options.node_radius;
    body->velocity_damping = options.velocity_damping;
    body->spring_damping = options.spring_damping;
    body->contact_friction = options.contact_friction;
    body->shape_matching_stiffness = options.shape_matching_stiffness;
    body->shape_maximum_projection = options.maximum_projection_fraction *
                                     minimum_bond_length;
    body->maximum_projection_fraction = options.maximum_projection_fraction;
    body->constraint_velocity_response = options.constraint_velocity_response;
    body->maximum_speed = options.maximum_speed;
    body->movable_mass = movable_mass;
    body->shape_rest_center = shape_rest_center;
    body->shape_inverse_rest = shape_inverse_rest;
    body->solver_iterations = options.solver_iterations;
#define PM_ALLOC_SOFT(member, count)                                            \
    status = allocate_managed(body->member, count);                             \
    if (!status) return status
    PM_ALLOC_SOFT(positions, body->node_count);
    PM_ALLOC_SOFT(rest_positions, body->node_count);
    PM_ALLOC_SOFT(scratch, body->node_count);
    PM_ALLOC_SOFT(previous, body->node_count);
    PM_ALLOC_SOFT(velocities, body->node_count);
    PM_ALLOC_SOFT(velocity_scratch, body->node_count);
    PM_ALLOC_SOFT(inverse_masses, body->node_count);
    PM_ALLOC_SOFT(bonds, body->bond_count);
    PM_ALLOC_SOFT(bond_active, body->bond_count);
    PM_ALLOC_SOFT(offsets, offsets.size());
    PM_ALLOC_SOFT(neighbors, neighbors.size());
    PM_ALLOC_SOFT(surface_rest_positions, body->surface_vertex_count);
    PM_ALLOC_SOFT(surface_positions, body->surface_vertex_count);
    PM_ALLOC_SOFT(surface_indices, body->surface_index_count);
    PM_ALLOC_SOFT(surface_bindings, body->surface_vertex_count);
    PM_ALLOC_SOFT(surface_corner_corrections, body->surface_index_count);
    PM_ALLOC_SOFT(surface_node_offsets, surface_offsets.size());
    PM_ALLOC_SOFT(surface_node_influences, surface_influences.size());
    PM_ALLOC_SOFT(body_impulses, body->node_count);
    PM_ALLOC_SOFT(body_position_corrections, body->node_count);
    PM_ALLOC_SOFT(cloth_forces, body->node_count);
    PM_ALLOC_SOFT(fluid_forces, body->node_count);
    PM_ALLOC_SOFT(rigid_contact_forces, body->node_count);
    PM_ALLOC_SOFT(contact_normals, body->node_count);
    PM_ALLOC_SOFT(contact_arms, body->node_count);
    PM_ALLOC_SOFT(contact_momentum_delta, body->node_count);
    PM_ALLOC_SOFT(contact_friction_delta, body->node_count);
    PM_ALLOC_SOFT(contact_normal_delta, body->node_count);
    PM_ALLOC_SOFT(predicted_momentum, 1U);
    PM_ALLOC_SOFT(shape_orientation, 1U);
    PM_ALLOC_SOFT(dynamic_contact_flag, 1U);
    PM_ALLOC_SOFT(contact_count, 1U);
    PM_ALLOC_SOFT(count, 1U);
#undef PM_ALLOC_SOFT
    for (std::uint32_t node = 0U; node < body->node_count; ++node) {
        body->positions[node] = options.nodes.data[node];
        body->rest_positions[node] = options.nodes.data[node];
        body->scratch[node] = options.nodes.data[node];
        body->previous[node] = options.nodes.data[node];
        body->velocities[node] = {};
        body->velocity_scratch[node] = {};
        body->inverse_masses[node] = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
        body->rigid_contact_forces[node] = {};
        body->cloth_forces[node] = {};
        body->fluid_forces[node] = {};
        body->contact_normals[node] = {};
        body->contact_arms[node] = {};
        body->contact_momentum_delta[node] = {};
        body->contact_friction_delta[node] = {};
        body->contact_normal_delta[node] = 0.0F;
    }
    std::copy_n(options.bonds.data, body->bond_count, body->bonds);
    std::fill_n(body->bond_active, body->bond_count, std::uint8_t{1U});
    std::copy(offsets.begin(), offsets.end(), body->offsets);
    std::copy(neighbors.begin(), neighbors.end(), body->neighbors);
    std::copy_n(options.surface_vertices.data, body->surface_vertex_count,
                body->surface_rest_positions);
    std::copy_n(options.surface_vertices.data, body->surface_vertex_count,
                body->surface_positions);
    std::copy_n(options.surface_triangle_indices.data,
                body->surface_index_count, body->surface_indices);
    std::copy_n(options.surface_bindings.data, body->surface_vertex_count,
                body->surface_bindings);
    std::copy(surface_offsets.begin(), surface_offsets.end(), body->surface_node_offsets);
    std::copy(surface_influences.begin(), surface_influences.end(), body->surface_node_influences);
    *body->count = body->node_count;
    *body->contact_count = 0U;
    *body->predicted_momentum = {};
    *body->shape_orientation = {0.0F, 0.0F, 0.0F, 1.0F};
    *body->dynamic_contact_flag = 0U;
    body->alive = true;
    output = {slot, body->generation};
    impl_->soft_bodies[slot] = std::move(body);
    ++impl_->revision;
    return success();
}

Status World::add_rope(RopeOptions options, RopeId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument,"world is not initialized");
    Status status=impl_->require_idle();
    if(!status)return status;
    if(!finite(options.radius)||options.radius<=0 || !finite(options.node_spacing)||
       options.node_spacing<=0 || options.node_spacing>2*options.radius ||
       !finite(options.mass)||options.mass<=0 || !finite(options.stretch_compliance)||options.stretch_compliance<0 ||
       !finite(options.velocity_damping)||options.velocity_damping<0 ||
       !finite(options.maximum_substep_timestep)||options.maximum_substep_timestep<=0 ||
       !finite(options.friction)||options.friction<0 ||
       !finite(options.maximum_speed)||options.maximum_speed<=0 || options.solver_iterations<1 || options.solver_iterations>128)
        return failure(StatusCode::invalid_argument,"invalid rope material or resolution");
    for(const auto anchor:{options.first,options.last}) {
        if(!finite(anchor.local_anchor))return failure(StatusCode::invalid_argument,"invalid rope anchor");
        if(anchor.enabled){unsigned dense;if(!(status=impl_->validate_handle(anchor.body,dense)))return status;}
    }
    if (options.first.enabled && options.last.enabled && options.first.body == options.last.body)
        return failure(StatusCode::invalid_argument,"rope endpoints must attach to distinct bodies");
    std::uint32_t slot=0;
    while(slot<impl_->ropes.size() && impl_->ropes[slot] && impl_->ropes[slot]->alive)++slot;
    if(slot==impl_->ropes.size())return failure(StatusCode::capacity_exceeded,"rope capacity exhausted");
    std::vector<Vec3> nodes;
    if(!(status=sample_rope_centerline(options.centerline,options.node_spacing,nodes)))return status;
    int attached[2]{-1, -1};
    // Rest curve endpoints define anchors, not an initial teleport/impulse.
    for(unsigned end=0;end<2;++end) {
        const auto anchor=end?options.last:options.first;
        if(!anchor.enabled)continue;
        unsigned dense;if(!(status=impl_->validate_handle(anchor.body,dense)))return status;
        attached[end] = static_cast<int>(dense);
        const Vec3 target=transform_point(impl_->states[impl_->current_state][dense],anchor.local_anchor);
        if(vector_length(subtract(target,end?nodes.back():nodes.front()))>1e-3F)
            return failure(StatusCode::invalid_argument,"rope endpoint must match its body-local attachment");
    }
    try {
        if (rope_rest_crosses_collider(nodes, attached[0], attached[1], impl_->parameters,
                impl_->states[impl_->current_state], impl_->meshes, impl_->rigid_body_count))
            return failure(StatusCode::invalid_argument,"rope rest centerline crosses a rigid collider");
    } catch (...) {
        return failure(StatusCode::out_of_memory,"rope rest collision validation allocation failed");
    }
    std::unique_ptr<RopeStorage> owner(new(std::nothrow)RopeStorage());
    if(!owner)return failure(StatusCode::out_of_memory,"rope allocation failed");
    owner->generation=impl_->ropes[slot]?impl_->ropes[slot]->generation:1;
    auto &r=owner->data;
    r.options=options;r.options.centerline={};r.count=static_cast<unsigned>(nodes.size());
    r.body_capacity=impl_->options.rigid_body_capacity;
    for(Vec3 **p:{&r.positions,&r.previous,&r.velocities,&r.constraint_forces,&r.contact_forces,&r.directions,&r.scratch,&r.normals,&r.normals2})
        if(!(status=allocate_managed(*p,r.count)))return status;
    for(float **p:{&r.rest,&r.lambda})
        if(!(status=allocate_managed(*p,r.count)))return status;
    if(!(status=allocate_managed(r.body_translation,impl_->options.rigid_body_capacity)) ||
       !(status=allocate_managed(r.body_rotation,impl_->options.rigid_body_capacity)))return status;
    if(!(status=allocate_managed(r.solid_hint,r.count*r.body_capacity)))return status;
    std::fill_n(r.solid_hint,r.count*r.body_capacity,~0U);
    for(unsigned i=0;i<r.count;++i){
        r.positions[i]=r.previous[i]=nodes[i];r.velocities[i]=r.constraint_forces[i]=r.contact_forces[i]={};
        if(i+1<r.count){r.rest[i]=vector_length(subtract(nodes[i+1],nodes[i]));
            if(!finite(r.rest[i]) || r.rest[i]<1e-6F)return failure(StatusCode::invalid_argument,"rope contains an invalid segment");}
    }
    owner->alive=true;output={slot,owner->generation};impl_->ropes[slot]=std::move(owner);++impl_->revision;
    return success();
}

Status World::remove_rope(RopeId id) noexcept {
    if(!impl_)return failure(StatusCode::invalid_argument,"world is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(id.index>=impl_->ropes.size() || !impl_->ropes[id.index] || !impl_->ropes[id.index]->alive || impl_->ropes[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope handle is stale");
    auto &rope=*impl_->ropes[id.index];rope.alive=false;rope.release();++rope.generation;
    if(rope.generation==0)rope.generation=1;
    ++impl_->revision;return success();
}

Status World::rope_view(RopeId id, RopeDeviceView &output) const noexcept {
    output={};
    if(!impl_)return failure(StatusCode::invalid_argument,"world is not initialized");
    auto status=impl_->require_current_device();if(!status)return status;
    if(impl_->frame && !impl_->frame->acknowledged)return failure(StatusCode::busy,"rope view requires completed frame");
    if(id.index>=impl_->ropes.size() || !impl_->ropes[id.index] || !impl_->ropes[id.index]->alive || impl_->ropes[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope handle is stale");
    const auto &r=impl_->ropes[id.index]->data;
    output={{r.positions,r.count},{r.velocities,r.count},{r.constraint_forces,r.count},{r.contact_forces,r.count},{r.rest,r.count-1},r.options.radius};
    return success();
}

Status World::remove_soft_body(SoftBodyId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_bodies.size() || !impl_->soft_bodies[id.index] ||
        !impl_->soft_bodies[id.index]->alive ||
        impl_->soft_bodies[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body handle is stale");
    SoftBodyStorage &body = *impl_->soft_bodies[id.index];
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a cloth coupling");
    body.alive = false;
    body.release();
    ++body.generation;
    if (body.generation == 0U) body.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::soft_body_view(SoftBodyId id,
                            SoftBodyDeviceView &output) const noexcept {
    output = {};
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->soft_bodies.size() || !impl_->soft_bodies[id.index] ||
        !impl_->soft_bodies[id.index]->alive ||
        impl_->soft_bodies[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body handle is stale");
    const SoftBodyStorage &body = *impl_->soft_bodies[id.index];
    output.positions = {body.positions, body.node_count};
    output.velocities = {body.velocities, body.node_count};
    output.bonds = {body.bonds, body.bond_count};
    output.surface_positions = {
        body.surface_positions, body.surface_vertex_count};
    output.surface_triangle_indices = {
        body.surface_indices, body.surface_index_count};
    output.rigid_contact_forces = {
        body.rigid_contact_forces, body.node_count};
    output.cloth_contact_forces = {body.cloth_forces, body.node_count};
    output.fluid_contact_forces = {body.fluid_forces, body.node_count};
    output.node_count = body.node_count;
    output.surface_vertex_count = body.surface_vertex_count;
    return success();
}

static bool valid_fluid_soft_options(FluidSoftBodyCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        options.solver_iterations > 0U && options.solver_iterations <= 16U;
}

Status World::add_fluid_soft_body_coupling(
    FluidSoftBodyCouplingOptions options, FluidSoftBodyCouplingId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    SoftBodyDeviceView view{};
    if (!(status = impl_->validate_handle(options.fluid, fluid)) ||
        !(status = soft_body_view(options.soft_body, view))) return status;
    if (!valid_fluid_soft_options(options))
        return failure(StatusCode::invalid_argument, "invalid fluid soft-body options");
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == options.fluid &&
            coupling->options.soft_body == options.soft_body)
            return failure(StatusCode::invalid_argument, "fluid and soft body are already coupled");
    std::uint32_t slot = 0;
    for (; slot < impl_->fluid_soft_couplings.size(); ++slot)
        if (!impl_->fluid_soft_couplings[slot] || !impl_->fluid_soft_couplings[slot]->alive) break;
    if (slot == impl_->fluid_soft_couplings.size())
        return failure(StatusCode::capacity_exceeded, "fluid soft-body coupling capacity exhausted");
    auto &body = *impl_->soft_bodies[options.soft_body.index];
    std::unique_ptr<FluidSoftCouplingStorage> coupling;
    std::vector<BvhNode> tree;
    std::vector<std::uint32_t> order, parents;
    double volume = 0;
    try {
        // glTF can duplicate a position at normal/UV seams. Validate geometric
        // edges, not rendering indices, without changing the skin bindings.
        std::map<std::array<float, 3>, std::uint32_t> vertices;
        std::vector<std::uint32_t> canonical(body.surface_vertex_count);
        for (std::uint32_t i = 0; i < body.surface_vertex_count; ++i) {
            const auto p = body.surface_rest_positions[i];
            canonical[i] = vertices.emplace(std::array{p.x, p.y, p.z}, vertices.size()).first->second;
        }
        struct Edge { std::uint32_t count{}; int winding{}; };
        std::map<std::pair<std::uint32_t, std::uint32_t>, Edge> edges;
        const Vec3 origin = body.surface_rest_positions[0];
        for (std::uint32_t base = 0; base < body.surface_index_count; base += 3) {
            const auto *tri = body.surface_indices + base;
            volume += dot(subtract(body.surface_rest_positions[tri[0]], origin),
                cross(subtract(body.surface_rest_positions[tri[1]], origin),
                      subtract(body.surface_rest_positions[tri[2]], origin))) / 6.0;
            for (std::uint32_t e = 0; e < 3; ++e) {
                auto a = canonical[tri[e]], b = canonical[tri[(e + 1) % 3]];
                auto &edge = edges[std::minmax(a, b)];
                ++edge.count;
                edge.winding += a < b ? 1 : -1;
            }
        }
        if (std::abs(volume) < 1.0e-10 || std::any_of(edges.begin(), edges.end(),
            [](const auto &edge) { return edge.second.count != 2 || edge.second.winding != 0; }))
            return failure(StatusCode::invalid_argument,
                           "fluid soft-body coupling needs a closed consistently wound surface");
        coupling = std::make_unique<FluidSoftCouplingStorage>();
        order.resize(body.surface_index_count / 3U);
        std::iota(order.begin(), order.end(), 0U);
        const auto centroid = [&](std::uint32_t triangle) {
            const auto *indices = body.surface_indices + triangle * 3U;
            return multiply(add(add(body.surface_rest_positions[indices[0]],
                body.surface_rest_positions[indices[1]]),
                body.surface_rest_positions[indices[2]]), 1.0F / 3.0F);
        };
        std::function<std::uint32_t(std::uint32_t, std::uint32_t, std::uint32_t)> build =
            [&](std::uint32_t first, std::uint32_t end, std::uint32_t parent) {
                const auto index = static_cast<std::uint32_t>(tree.size());
                BvhNode node{};
                node.minimum = {FLT_MAX, FLT_MAX, FLT_MAX};
                node.maximum = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (auto item = first; item < end; ++item) {
                    const Vec3 center = centroid(order[item]);
                    node.minimum = component_min(node.minimum, center);
                    node.maximum = component_max(node.maximum, center);
                }
                tree.push_back(node);
                parents.push_back(parent);
                if (end - first <= 4U) {
                    tree[index].first_triangle = first;
                    tree[index].triangle_count = end - first;
                } else {
                    const Vec3 extent = subtract(node.maximum, node.minimum);
                    const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 :
                        (extent.y >= extent.z ? 1 : 2);
                    const auto middle = first + (end - first) / 2U;
                    std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + end,
                        [&](auto a, auto b) {
                            const Vec3 ca = centroid(a), cb = centroid(b);
                            const float x = axis == 0 ? ca.x : (axis == 1 ? ca.y : ca.z);
                            const float y = axis == 0 ? cb.x : (axis == 1 ? cb.y : cb.z);
                            return x < y || (x == y && a < b);
                        });
                    const auto left = build(first, middle, index);
                    const auto right = build(middle, end, index);
                    tree[index].left = left;
                    tree[index].right = right;
                }
                return index;
            };
        build(0U, static_cast<std::uint32_t>(order.size()), k_invalid_dense);
    } catch (...) { return failure(StatusCode::out_of_memory, "failed to build fluid soft-body coupling"); }
#define PM_ALLOC_FLUID_SOFT(member, count) \
    status = allocate_managed(coupling->member, count); if (!status) return status
    PM_ALLOC_FLUID_SOFT(contacts, fluid->options.capacity);
    PM_ALLOC_FLUID_SOFT(counts, body.node_count);
    PM_ALLOC_FLUID_SOFT(position_deltas, body.node_count);
    PM_ALLOC_FLUID_SOFT(impulses, body.node_count);
    PM_ALLOC_FLUID_SOFT(previous_surface, body.surface_vertex_count);
    PM_ALLOC_FLUID_SOFT(bounds, 2U);
    PM_ALLOC_FLUID_SOFT(contact_count, 1U);
    PM_ALLOC_FLUID_SOFT(maximum_penetration, 1U);
    PM_ALLOC_FLUID_SOFT(tree, tree.size());
    PM_ALLOC_FLUID_SOFT(parents, parents.size());
    PM_ALLOC_FLUID_SOFT(ready, tree.size());
    PM_ALLOC_FLUID_SOFT(triangle_order, order.size());
#undef PM_ALLOC_FLUID_SOFT
    std::copy(tree.begin(), tree.end(), coupling->tree);
    std::copy(parents.begin(), parents.end(), coupling->parents);
    std::copy(order.begin(), order.end(), coupling->triangle_order);
    coupling->tree_count = static_cast<std::uint32_t>(tree.size());
    std::copy_n(body.surface_positions, body.surface_vertex_count, coupling->previous_surface);
    *coupling->contact_count = 0;
    *coupling->maximum_penetration = 0;
    coupling->orientation = volume < 0 ? -1.0F : 1.0F;
    if (impl_->fluid_soft_couplings[slot])
        coupling->generation = impl_->fluid_soft_couplings[slot]->generation;
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->fluid_soft_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status World::update_fluid_soft_body_coupling(
    FluidSoftBodyCouplingId id, FluidSoftBodyCouplingOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_soft_couplings.size() || !impl_->fluid_soft_couplings[id.index] ||
        !impl_->fluid_soft_couplings[id.index]->alive ||
        impl_->fluid_soft_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid soft-body coupling is stale");
    auto &coupling = *impl_->fluid_soft_couplings[id.index];
    if (!valid_fluid_soft_options(options) || !(options.fluid == coupling.options.fluid) ||
        !(options.soft_body == coupling.options.soft_body))
        return failure(StatusCode::invalid_argument, "invalid options or changed fluid soft-body endpoints");
    coupling.options = options;
    ++impl_->revision;
    return success();
}

Status World::remove_fluid_soft_body_coupling(FluidSoftBodyCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_soft_couplings.size() || !impl_->fluid_soft_couplings[id.index] ||
        !impl_->fluid_soft_couplings[id.index]->alive ||
        impl_->fluid_soft_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid soft-body coupling is stale");
    auto &coupling = *impl_->fluid_soft_couplings[id.index];
    coupling.release();
    coupling.alive = false;
    if (++coupling.generation == 0) coupling.generation = 1;
    ++impl_->revision;
    return success();
}

static bool valid_soft_cloth_options(SoftBodyClothCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        options.solver_iterations > 0U && options.solver_iterations <= 16U;
}

Status World::add_soft_body_cloth_coupling(
    SoftBodyClothCouplingOptions options,
    SoftBodyClothCouplingId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    SoftBodyDeviceView soft_view{};
    ClothDeviceView cloth_view_result{};
    if (!(status = soft_body_view(options.soft_body, soft_view)) ||
        !(status = cloth_view(options.cloth, cloth_view_result))) return status;
    if (!valid_soft_cloth_options(options))
        return failure(StatusCode::invalid_argument, "invalid soft-body cloth options");
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive &&
            coupling->options.soft_body == options.soft_body &&
            coupling->options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                           "soft body and cloth are already coupled");
    std::uint32_t slot = 0U;
    for (; slot < impl_->soft_cloth_couplings.size(); ++slot)
        if (!impl_->soft_cloth_couplings[slot] ||
            !impl_->soft_cloth_couplings[slot]->alive) break;
    if (slot == impl_->soft_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "soft-body cloth coupling capacity exhausted");
    std::unique_ptr<SoftClothCouplingStorage> coupling;
    try { coupling = std::make_unique<SoftClothCouplingStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "failed to allocate soft-body cloth coupling"); }
    status = allocate_managed(coupling->contacts, soft_view.node_count);
    if (!status) return status;
    status = allocate_managed(coupling->cloth_contact_counts,
                              impl_->cloths[options.cloth.index]->vertex_capacity);
    if (!status) return status;
    if (impl_->soft_cloth_couplings[slot])
        coupling->generation = impl_->soft_cloth_couplings[slot]->generation;
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->soft_cloth_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status World::update_soft_body_cloth_coupling(
    SoftBodyClothCouplingId id, SoftBodyClothCouplingOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_cloth_couplings.size() ||
        !impl_->soft_cloth_couplings[id.index] ||
        !impl_->soft_cloth_couplings[id.index]->alive ||
        impl_->soft_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body cloth coupling is stale");
    auto &coupling = *impl_->soft_cloth_couplings[id.index];
    if (!valid_soft_cloth_options(options) ||
        !(options.soft_body == coupling.options.soft_body) ||
        !(options.cloth == coupling.options.cloth))
        return failure(StatusCode::invalid_argument,
                       "invalid options or changed soft-body cloth endpoints");
    coupling.options = options;
    ++impl_->revision;
    return success();
}

Status World::remove_soft_body_cloth_coupling(SoftBodyClothCouplingId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_cloth_couplings.size() ||
        !impl_->soft_cloth_couplings[id.index] ||
        !impl_->soft_cloth_couplings[id.index]->alive ||
        impl_->soft_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body cloth coupling is stale");
    auto &coupling = *impl_->soft_cloth_couplings[id.index];
    coupling.alive = false;
    coupling.release();
    if (++coupling.generation == 0U) coupling.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::add_fluid_cloth_coupling(
    FluidClothCouplingOptions options,
    FluidClothCouplingId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation !=
            options.cloth.generation)
        return failure(StatusCode::invalid_handle,
                       "cloth coupling handle is invalid or stale");
    const ClothStorage &cloth = *impl_->cloths[options.cloth.index];
    if (!cloth.preserve_volume)
        return failure(StatusCode::invalid_argument,
            "contained fluid requires closed volume-preserving cloth");
    if (!finite(options.contact_distance) || options.contact_distance < 0.0F ||
        !finite(options.interaction_radius) || options.interaction_radius < 0.0F ||
        !finite(options.stiffness) || options.stiffness <= 0.0F ||
        !finite(options.damping) || options.damping < 0.0F ||
        !finite(options.tangential_drag) || options.tangential_drag < 0.0F ||
        !finite(options.maximum_force) || options.maximum_force <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid fluid-cloth coupling options");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.fluid == options.fluid &&
            coupling.options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                "fluid and cloth are already coupled");
    std::uint32_t slot = 0U;
    for (; slot < impl_->fluid_cloth_couplings.size(); ++slot)
        if (!impl_->fluid_cloth_couplings[slot].alive) break;
    if (slot == impl_->fluid_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "fluid-cloth coupling capacity exhausted");
    FluidClothCouplingResource &coupling =
        impl_->fluid_cloth_couplings[slot];
    coupling.options = options;
    coupling.alive = true;
    output = {slot, coupling.generation};
    ++impl_->revision;
    return success();
}

Status World::update_fluid_cloth_coupling(
    FluidClothCouplingId id,
    FluidClothCouplingOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_cloth_couplings.size() ||
        !impl_->fluid_cloth_couplings[id.index].alive ||
        impl_->fluid_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation !=
            options.cloth.generation ||
        !impl_->cloths[options.cloth.index]->preserve_volume)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling cloth is invalid or open");
    if (!finite(options.contact_distance) || options.contact_distance < 0.0F ||
        !finite(options.interaction_radius) || options.interaction_radius < 0.0F ||
        !finite(options.stiffness) || options.stiffness <= 0.0F ||
        !finite(options.damping) || options.damping < 0.0F ||
        !finite(options.tangential_drag) || options.tangential_drag < 0.0F ||
        !finite(options.maximum_force) || options.maximum_force <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid fluid-cloth coupling options");
    impl_->fluid_cloth_couplings[id.index].options = options;
    ++impl_->revision;
    return success();
}

Status World::remove_fluid_cloth_coupling(
    FluidClothCouplingId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_cloth_couplings.size() ||
        !impl_->fluid_cloth_couplings[id.index].alive ||
        impl_->fluid_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling handle is stale");
    FluidClothCouplingResource &coupling =
        impl_->fluid_cloth_couplings[id.index];
    coupling.alive = false;
    ++coupling.generation;
    if (coupling.generation == 0U) coupling.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::add_rigid_body(RigidBodyOptions options,
                             RigidBodyId &output) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    status = validate_body_options(options);
    if (!status) {
        return status;
    }
    status = impl_->validate_handle(options.mesh);
    if (!status) {
        return status;
    }
    if (impl_->rigid_body_count >= impl_->options.rigid_body_capacity) {
        return failure(StatusCode::capacity_exceeded,
                       "rigid body capacity is exhausted");
    }

    std::uint32_t slot_index = k_invalid_dense;
    for (std::uint32_t index = 0; index < impl_->slots.size(); ++index) {
        if (!impl_->slots[index].alive) {
            slot_index = index;
            break;
        }
    }
    if (slot_index == k_invalid_dense) {
        return failure(StatusCode::internal_error,
                       "no free rigid body handle slot was found");
    }

    RigidBodyState normalized_state = options.initial_state;
    normalized_state.orientation =
        normalized_quaternion(normalized_state.orientation);
    const TriangleMeshResource &mesh = impl_->meshes[options.mesh.index];

    Impl::Slot &slot = impl_->slots[slot_index];
    const std::uint32_t dense = impl_->rigid_body_count;
    slot.alive = true;
    slot.dense_index = dense;
    if (slot.generation == 0U) {
        slot.generation = 1U;
    }
    const RigidBodyId id{slot_index, slot.generation};
    impl_->parameters[dense] =
        make_parameters(options, mesh);
    impl_->accumulators[dense] = {};
    impl_->targets[dense] = {};
    impl_->ids[dense] = id;
    impl_->states[0][dense] = normalized_state;
    impl_->states[1][dense] = normalized_state;
    ++impl_->rigid_body_count;
    ++impl_->revision;
    output = id;
    return success();
}

Status World::remove_rigid_body(RigidBodyId body) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index)
        if (impl_->paint_fields[index].alive &&
            impl_->paint_fields[index].options.body == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a paint field");
    for (std::uint32_t index = 0;
         index < impl_->options.paint_rule_capacity; ++index)
        if (impl_->paint_rules[index].alive &&
            impl_->paint_rules[index].options.rigid_source == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a paint rule");
    for(const auto &rope:impl_->ropes)
        if(rope && rope->alive &&
           ((rope->data.options.first.enabled && rope->data.options.first.body==body) ||
            (rope->data.options.last.enabled && rope->data.options.last.body==body)))
            return failure(StatusCode::invalid_argument,"rigid body is still referenced by a rope attachment");
    const std::uint32_t last = impl_->rigid_body_count - 1U;
    if (dense != last) {
        impl_->parameters[dense] = impl_->parameters[last];
        impl_->accumulators[dense] = impl_->accumulators[last];
        impl_->targets[dense] = impl_->targets[last];
        impl_->ids[dense] = impl_->ids[last];
        impl_->states[0][dense] = impl_->states[0][last];
        impl_->states[1][dense] = impl_->states[1][last];
        impl_->slots[impl_->ids[dense].index].dense_index = dense;
    }
    --impl_->rigid_body_count;
    Impl::Slot &slot = impl_->slots[body.index];
    slot.alive = false;
    slot.dense_index = k_invalid_dense;
    ++slot.generation;
    if (slot.generation == 0U) {
        slot.generation = 1U;
    }
    ++impl_->revision;
    return success();
}

Status World::set_rigid_body_state(RigidBodyId body,
                                   RigidBodyState state) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(state)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body state must contain finite values");
    }
    const float quaternion_size = state.orientation.x * state.orientation.x +
                                  state.orientation.y * state.orientation.y +
                                  state.orientation.z * state.orientation.z +
                                  state.orientation.w * state.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "rigid body orientation must be nonzero");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    state.orientation = normalized_quaternion(state.orientation);
    impl_->states[0][dense] = state;
    impl_->states[1][dense] = state;
    impl_->accumulators[dense] = {};
    impl_->targets[dense] = {};
    ++impl_->revision;
    return success();
}

Status World::set_kinematic_target(RigidBodyId body,
                                   RigidBodyState target) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(target)) {
        return failure(StatusCode::invalid_argument,
                       "kinematic target must contain finite values");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::kinematic) {
        return failure(StatusCode::invalid_argument,
                       "kinematic targets require a kinematic body");
    }
    const float quaternion_size = target.orientation.x * target.orientation.x +
                                  target.orientation.y * target.orientation.y +
                                  target.orientation.z * target.orientation.z +
                                  target.orientation.w * target.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "kinematic orientation must be nonzero");
    }
    target.orientation = normalized_quaternion(target.orientation);
    impl_->targets[dense] = {target, true};
    return success();
}

Status World::apply_force(RigidBodyId body, Vec3 force,
                          Vec3 world_point) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(force) || !finite(world_point)) {
        return failure(StatusCode::invalid_argument,
                       "force and application point must be finite");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::dynamic) {
        return failure(StatusCode::invalid_argument,
                       "forces may only be applied to dynamic bodies");
    }
    BodyAccumulator &accumulator = impl_->accumulators[dense];
    accumulator.force = add(accumulator.force, force);
    const Vec3 arm = subtract(
        world_point, impl_->states[impl_->current_state][dense].position);
    accumulator.torque = add(accumulator.torque, cross(arm, force));
    return success();
}

Status World::apply_impulse(RigidBodyId body, Vec3 impulse,
                            Vec3 world_point) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(impulse) || !finite(world_point)) {
        return failure(StatusCode::invalid_argument,
                       "impulse and application point must be finite");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::dynamic) {
        return failure(StatusCode::invalid_argument,
                       "impulses may only be applied to dynamic bodies");
    }
    BodyAccumulator &accumulator = impl_->accumulators[dense];
    accumulator.impulse = add(accumulator.impulse, impulse);
    const Vec3 arm = subtract(
        world_point, impl_->states[impl_->current_state][dense].position);
    accumulator.angular_impulse =
        add(accumulator.angular_impulse, cross(arm, impulse));
    return success();
}

Status World::rigid_body_view(RigidBodyDeviceView &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    output.ids = {impl_->ids, impl_->rigid_body_count};
    output.states = {impl_->states[impl_->current_state],
                     impl_->rigid_body_count};
    if (impl_->debug_applied_forces != nullptr) {
        output.applied_forces = {
            impl_->debug_applied_forces, impl_->rigid_body_count};
        output.applied_torques = {
            impl_->debug_applied_torques, impl_->rigid_body_count};
    }
    output.revision = impl_->revision;
    return success();
}

Status World::read_rigid_body_state(RigidBodyId body, RigidBodyState &output,
                                    cudaStream_t stream) const noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) {
        return status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    RigidBodyState temporary{};
    cudaError_t error = cudaMemcpyAsync(
        &temporary, impl_->states[impl_->current_state] + dense,
        sizeof(RigidBodyState), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "rigid body state readback failed");
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "rigid body state readback synchronization failed");
    }
    output = temporary;
    return success();
}

Status World::step_async(StepOptions options, FrameToken &completion,
                         cudaStream_t stream) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (completion.pending()) {
        return failure(StatusCode::busy,
                       "completion token already represents a pending frame");
    }
    if (!finite(options.timestep) || options.timestep <= 0.0F ||
        options.substeps == 0U || options.substeps > 1'024U ||
        !finite(options.gravity)) {
        return failure(StatusCode::invalid_argument,
                       "step timestep, substeps, or gravity is invalid");
    }
    // Advance rigid attachments and ropes on the same clock. Splitting only
    // the rope after a coarse rigid step leaves endpoints discontinuous.
    for (const auto &rope : impl_->ropes) if (rope && rope->alive) {
        const float required = std::ceil(options.timestep /
            rope->data.options.maximum_substep_timestep);
        if (!finite(required) || required > 1'024.0F)
            return failure(StatusCode::invalid_argument,
                           "rope timestep limit requires more than 1024 substeps");
        options.substeps = std::max(options.substeps,
                                   static_cast<std::uint32_t>(required));
    }
    for (auto &cloth : impl_->cloths) if (cloth && cloth->alive) {
        status = rebuild_cloth_topology(*cloth);
        if (!status) return status;
    }
    const bool debug_enabled =
        impl_->options.physics_debug.frame_capacity != 0U;
    const bool collect_rigid_contacts =
        options.collect_rigid_contacts || debug_enabled;
    const bool collect_fluid_contacts =
        options.collect_fluid_contacts || debug_enabled;
    if (impl_->rigid_body_count == 0U) {
        *impl_->rigid_contact_count = 0U;
    }
    *impl_->fluid_neighbor_overflow = 0U;
    *impl_->fluid_maximum_neighbor_count = 0U;
    *impl_->fluid_contact_count = 0U;
    *impl_->fluid_contact_overflow = 0U;

    std::unique_ptr<FrameToken::Impl> token_impl;
    if (completion.impl_) {
        token_impl = std::move(completion.impl_);
    } else {
        try {
            token_impl = std::make_unique<FrameToken::Impl>();
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "failed to allocate completion token state");
        }
    }

    std::shared_ptr<CompletionState> frame;
    try {
        frame = std::make_shared<CompletionState>();
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate frame completion state");
    }
    frame->fluid_neighbor_overflow = impl_->fluid_neighbor_overflow;
    if (debug_enabled) {
        try {
            frame->on_complete = [implementation = impl_.get(), options]() {
                return implementation->record_debug_frame(options);
            };
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "physics debug completion allocation failed");
        }
    }
    cudaError_t error =
        cudaEventCreateWithFlags(&frame->event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to create frame completion event");
    }

    impl_->timing_available = false;
    impl_->timing_boundary_count = 0U;
    std::size_t timing_boundary = 0U;
    if (options.collect_kernel_timings) {
        std::size_t maximum_stages =
            static_cast<std::size_t>(options.substeps) * 8U + 1U;
        for (const auto &fluid : impl_->fluids) {
            if (fluid && fluid->alive) {
                maximum_stages += 2U +
                    static_cast<std::size_t>(options.substeps) *
                        fluid->options.solver_iterations *
                        (9U + 2U * impl_->fluid_soft_couplings.size());
            }
        }
        for (const auto &cloth : impl_->cloths) {
            if (cloth && cloth->alive)
                maximum_stages += static_cast<std::size_t>(options.substeps) *
                    (cloth->solver_iterations + 2U);
        }
        for (const auto &body : impl_->soft_bodies) {
            if (body && body->alive)
                maximum_stages += static_cast<std::size_t>(options.substeps) *
                    (body->solver_iterations + 3U +
                     (body->solver_iterations + 1U) / 2U);
        }
        status = impl_->prepare_timing_events(
            maximum_stages + options.substeps * impl_->ropes.size() + 2U);
        if (!status) {
            return status;
        }
        error = cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to begin kernel timing");
        }
    }

    const auto record_timing_stage = [&](TimingStage stage,
                                         std::uint32_t launches = 0U) noexcept -> Status {
        if (!options.collect_kernel_timings) {
            return success();
        }
        if (timing_boundary >= impl_->timing_events.size()) {
            cudaStreamSynchronize(stream);
            return failure(StatusCode::internal_error,
                           "kernel timing stage budget exhausted");
        }
        impl_->timing_stages.push_back(stage);
        impl_->timing_launch_counts.push_back(launches);
        const cudaError_t timing_error =
            cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        return timing_error == cudaSuccess
                   ? success()
                   : cuda_failure(timing_error,
                                  "failed to record kernel timing boundary");
    };

    constexpr std::uint32_t block_size = 128U;
    const std::uint32_t block_count =
        (impl_->rigid_body_count + block_size - 1U) / block_size;
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
        error = cudaMemcpyAsync(coupling->previous_surface, body.surface_positions,
            body.surface_vertex_count * sizeof(Vec3), cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) return cuda_failure(error, "previous fluid soft-body skin copy failed");
    }
    if (debug_enabled && impl_->rigid_body_count != 0U) {
        capture_rigid_inputs_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->accumulators, impl_->debug_applied_forces,
            impl_->debug_applied_torques, impl_->rigid_body_count);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return cuda_failure(error,
                                "rigid debug input capture launch failed");
    }
    const bool has_cloth = std::any_of(impl_->cloths.begin(), impl_->cloths.end(),
        [](const auto &cloth) { return cloth && cloth->alive; });
    const bool has_soft_body = std::any_of(
        impl_->soft_bodies.begin(), impl_->soft_bodies.end(),
        [](const auto &body) { return body && body->alive; });
    bool any_moving_body = false;
    if (impl_->fluid_count != 0U) {
        for (std::uint32_t body = 0U; body < impl_->rigid_body_count; ++body)
            any_moving_body |= impl_->parameters[body].motion !=
                               MotionType::static_body;
        if (any_moving_body) {
            error = cudaMemcpyAsync(impl_->fluid_previous_states,
                impl_->states[impl_->current_state],
                impl_->rigid_body_count * sizeof(RigidBodyState),
                cudaMemcpyDeviceToDevice, stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid body state copy failed");
        }
    }
    const float substep_timestep =
        options.timestep / static_cast<float>(options.substeps);
    // The lowest-priority remaining pair colors every round, so no small
    // world needs more rounds than its number of unordered body pairs.
    const std::uint32_t color_round_count =
        impl_->rigid_body_count <= 1U ? 1U
        : impl_->rigid_body_count < 9U
            ? impl_->rigid_body_count * (impl_->rigid_body_count - 1U) / 2U
            : k_contact_color_count;
    impl_->rigid_solve_kernels_per_substep =
        3U + 3U * color_round_count +
        8U * (color_round_count + 1U);
    const auto coupled_cloth = [&](std::uint32_t index) {
        return std::any_of(impl_->soft_cloth_couplings.begin(),
            impl_->soft_cloth_couplings.end(), [&](const auto &coupling) {
                return coupling && coupling->alive && coupling->options.enabled &&
                       coupling->options.cloth.index == index;
            });
    };
    const auto coupled_soft = [&](std::uint32_t index) {
        return std::any_of(impl_->soft_cloth_couplings.begin(),
            impl_->soft_cloth_couplings.end(), [&](const auto &coupling) {
                return coupling && coupling->alive && coupling->options.enabled &&
                       coupling->options.soft_body.index == index;
            });
    };
    const auto advance_cloth = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        for (std::uint32_t cloth_index = 0U;
             cloth_index < impl_->cloths.size(); ++cloth_index) {
            const auto &cloth_pointer = impl_->cloths[cloth_index];
            if (!cloth_pointer || !cloth_pointer->alive) continue;
            ClothStorage &cloth = *cloth_pointer;
            const std::uint32_t blocks =
                (cloth.vertex_count + block_size - 1U) / block_size;
            deformable_predict<<<blocks, block_size, 0, stream>>>(
                cloth.positions, cloth.previous, cloth.velocities,
                cloth.inverse_masses, cloth.vertex_count, options.gravity,
                substep_timestep, cloth.velocity_damping);
            Status cloth_status = record_timing_stage(
                TimingStage::cloth_prediction);
            if (!cloth_status) return cloth_status;
            for (std::uint32_t iteration = 0U;
                 iteration < cloth.solver_iterations; ++iteration) {
                deformable_project_links<<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.scratch, cloth.inverse_masses,
                    cloth.offsets, cloth.neighbors, cloth.bond_active,
                    cloth.vertex_count, substep_timestep, 0.0F);
                std::swap(cloth.positions, cloth.scratch);
                if (cloth.preserve_volume) {
                    cloth_project_volume<<<1U, 1U, 0, stream>>>(
                        cloth.positions, cloth.inverse_masses, cloth.indices,
                        cloth.vertex_count, cloth.index_count / 3U,
                        cloth.volume_gradients, cloth.target_volume,
                        cloth.orientation, cloth.volume_compliance,
                        substep_timestep, cloth.volume_lambda,
                        iteration == 0U);
                }
                cloth_status = record_timing_stage(
                    TimingStage::cloth_constraints);
                if (!cloth_status) return cloth_status;
            }
            if (impl_->rigid_body_count != 0U) {
                const cudaError_t clear_error = cudaMemsetAsync(
                    impl_->fluid_body_contact_flags, 0,
                    impl_->rigid_body_count * sizeof(std::uint32_t), stream);
                if (clear_error != cudaSuccess)
                    return cuda_failure(clear_error,
                                        "cloth contact flags clear failed");
                deformable_collide<false><<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.previous,
                    cloth.inverse_masses, cloth.vertex_count,
                    cloth.thickness, substep_timestep,
                    impl_->parameters, previous_states,
                    impl_->states[impl_->current_state],
                    impl_->meshes, impl_->rigid_body_count,
                    cloth.body_impulses, cloth.rigid_contact_forces,
                    nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr,
                    impl_->fluid_body_contact_flags,
                    false, true, 20.0F);
                reduce_point_body_impulses<<<impl_->rigid_body_count,
                                             block_size, 0, stream>>>(
                    cloth.body_impulses, cloth.count, impl_->parameters,
                    impl_->states[impl_->current_state],
                    impl_->fluid_body_contact_flags, impl_->rigid_body_count);
                if (cloth.surface_positions != nullptr) {
                    const std::uint32_t triangles = cloth.index_count / 3U;
                    cloth_update_surface<<<
                        (triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                }
                cloth_constrain_bodies<<<block_count, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.indices, cloth.vertex_sources,
                    cloth.surface_positions,
                    cloth.inverse_masses,
                    cloth.vertex_count, cloth.index_count / 3U,
                    cloth.thickness, substep_timestep,
                    cloth.contact_friction, impl_->parameters,
                    previous_states, impl_->states[impl_->current_state],
                    impl_->meshes, impl_->ids, impl_->rigid_body_count,
                    {cloth_index, cloth.generation}, impl_->paint_fields,
                    impl_->options.paint_field_capacity, impl_->paint_rules,
                    impl_->options.paint_rule_capacity,
                    cloth.body_corrections);
                cloth_apply_body_corrections<<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.inverse_masses,
                    cloth.vertex_count,
                    cloth.body_corrections, impl_->rigid_body_count);
            } else {
                deformable_collide<false><<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.previous,
                    cloth.inverse_masses, cloth.vertex_count,
                    cloth.thickness, substep_timestep,
                    impl_->parameters, previous_states,
                    impl_->states[impl_->current_state],
                    impl_->meshes, 0U, cloth.body_impulses,
                    cloth.rigid_contact_forces, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr,
                    impl_->fluid_body_contact_flags, false, true, 20.0F);
            }
            if (cloth.free_triangle_nodes)
                cloth_limit_strain<<<1U, 128U, 0, stream>>>(cloth.positions,
                    cloth.scratch, cloth.inverse_masses, cloth.offsets,
                    cloth.neighbors, cloth.free_triangle_nodes, cloth.vertex_count);
            cloth_status = record_timing_stage(TimingStage::cloth_contacts);
            if (!cloth_status) return cloth_status;
            if (cloth.surface_positions != nullptr) {
                const std::uint32_t triangles = cloth.index_count / 3U;
                if (!coupled_cloth(cloth_index)) cloth_break_bonds<<<
                    (cloth.bond_count + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions, cloth.bonds,
                    cloth.bond_active, cloth.bond_damage, cloth.bond_count,
                    cloth.break_strain, cloth.fracture_persistence_substeps,
                    cloth.body_impulses, cloth.impact_break_impulse);
                cloth_update_surface<<<
                    (triangles + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions,
                    cloth.indices,
                    cloth.surface_positions, triangles);
            }
        }
        const cudaError_t cloth_error = cudaPeekAtLastError();
        if (cloth_error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(cloth_error, "cloth kernel launch failed");
        }
        return success();
    };
    const auto advance_soft_bodies = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        for (const auto &body_pointer : impl_->soft_bodies) {
            if (!body_pointer || !body_pointer->alive) continue;
            SoftBodyStorage &body = *body_pointer;
            const std::uint32_t blocks =
                (body.node_count + block_size - 1U) / block_size;
            deformable_predict<<<blocks, block_size, 0, stream>>>(
                body.positions, body.previous, body.velocities,
                body.inverse_masses, body.node_count, options.gravity,
                substep_timestep, body.velocity_damping);
            cudaError_t friction_clear_error = cudaMemsetAsync(
                body.contact_momentum_delta, 0,
                body.node_count * sizeof(Vec3), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body momentum accumulator clear failed");
            friction_clear_error = cudaMemsetAsync(
                body.contact_friction_delta, 0,
                body.node_count * sizeof(Vec3), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body friction accumulator clear failed");
            friction_clear_error = cudaMemsetAsync(
                body.contact_normal_delta, 0,
                body.node_count * sizeof(float), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body normal accumulator clear failed");
            friction_clear_error = cudaMemsetAsync(
                body.dynamic_contact_flag, 0,
                sizeof(std::uint32_t), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body dynamic contact flag clear failed");
            soft_body_measure_momentum<<<1U, block_size, 0, stream>>>(
                body.velocities, body.inverse_masses, body.node_count,
                body.predicted_momentum);
            Status body_status = record_timing_stage(
                TimingStage::soft_body_prediction);
            if (!body_status) return body_status;
            const auto resolve_contacts = [&]() noexcept -> Status {
                cudaError_t clear_error = cudaMemsetAsync(
                    body.contact_count, 0, sizeof(std::uint32_t), stream);
                if (clear_error != cudaSuccess)
                    return cuda_failure(clear_error,
                        "soft-body contact count clear failed");
                if (impl_->rigid_body_count != 0U) {
                    clear_error = cudaMemsetAsync(
                        impl_->fluid_body_contact_flags, 0,
                        impl_->rigid_body_count * sizeof(std::uint32_t),
                        stream);
                    if (clear_error != cudaSuccess)
                        return cuda_failure(clear_error,
                            "soft-body contact flags clear failed");
                }
                deformable_collide<true><<<blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.previous,
                    body.inverse_masses, body.node_count, body.node_radius,
                    substep_timestep, impl_->parameters, previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->rigid_body_count, body.body_impulses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms,
                    body.contact_momentum_delta,
                    body.contact_normal_delta,
                    body.contact_count, body.dynamic_contact_flag,
                    impl_->fluid_body_contact_flags,
                    false, false, body.maximum_speed,
                    body.body_position_corrections);
                return success();
            };
            const auto apply_contact_traction = [&]() noexcept {
                soft_body_apply_contact_friction<<<
                    blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.inverse_masses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.body_impulses,
                    impl_->parameters,
                    body.contact_momentum_delta,
                    body.contact_friction_delta,
                    body.contact_normal_delta,
                    body.contact_count, body.node_count, body.movable_mass,
                    options.gravity, substep_timestep, body.contact_friction,
                    body.maximum_speed);
            };
            const auto finish_contact_pass = [&]() noexcept -> Status {
                apply_contact_traction();
                if (impl_->rigid_body_count != 0U)
                    reduce_point_body_impulses<<<
                        impl_->rigid_body_count, block_size, 0, stream>>>(
                        body.body_impulses, body.count, impl_->parameters,
                        impl_->states[impl_->current_state],
                        impl_->fluid_body_contact_flags,
                        impl_->rigid_body_count,
                        body.body_position_corrections);
                return record_timing_stage(TimingStage::soft_body_contacts);
            };
            // Contact and graph constraints form one position solve. Revisit
            // the triangle boundary after every two graph passes so spring
            // projection cannot strand nodes across a wall, while the next
            // passes distribute each contact correction through the volume.
            body_status = resolve_contacts();
            if (!body_status) return body_status;
            body_status = finish_contact_pass();
            if (!body_status) return body_status;
            for (std::uint32_t iteration = 0U;
                 iteration < body.solver_iterations; ++iteration) {
                deformable_project_links<<<blocks, block_size, 0, stream>>>(
                    body.positions, body.scratch, body.inverse_masses,
                    body.offsets, body.neighbors, body.bond_active,
                    body.node_count, substep_timestep,
                    body.maximum_projection_fraction);
                std::swap(body.positions, body.scratch);
                if (iteration + 1U == body.solver_iterations &&
                    body.shape_matching_stiffness > 0.0F) {
                    soft_body_project_rest_shape<<<1U, 1U, 0, stream>>>(
                        body.positions, body.rest_positions,
                        body.inverse_masses, body.node_count,
                        body.movable_mass, body.shape_rest_center,
                        body.shape_inverse_rest, body.shape_orientation,
                        body.shape_matching_stiffness,
                        body.shape_maximum_projection,
                        body.dynamic_contact_flag);
                }
                body_status = record_timing_stage(
                    TimingStage::soft_body_constraints);
                if (!body_status) return body_status;
                if ((iteration + 1U) % 2U == 0U ||
                    iteration + 1U == body.solver_iterations) {
                    body_status = resolve_contacts();
                    if (!body_status) return body_status;
                    if (iteration + 1U != body.solver_iterations) {
                        body_status = finish_contact_pass();
                        if (!body_status) return body_status;
                    }
                }
            }
            soft_body_finalize_velocities<<<blocks, block_size, 0, stream>>>(
                body.positions, body.previous, body.velocities,
                body.inverse_masses, body.node_count,
                1.0F / substep_timestep,
                body.constraint_velocity_response, body.maximum_speed);
            soft_body_damp_springs<<<blocks, block_size, 0, stream>>>(
                body.positions, body.velocities, body.velocity_scratch,
                body.inverse_masses, body.offsets, body.neighbors,
                body.bond_active, body.node_count, body.spring_damping,
                body.maximum_speed);
            std::swap(body.velocities, body.velocity_scratch);
            body_status = finish_contact_pass();
            if (!body_status) return body_status;
            soft_body_restore_momentum<<<1U, block_size, 0, stream>>>(
                body.velocities, body.inverse_masses,
                body.contact_momentum_delta, body.node_count,
                body.movable_mass, body.predicted_momentum,
                body.dynamic_contact_flag,
                body.maximum_speed);
            // Traction changes positions too. Finish with nonpenetration so
            // neither friction nor a competing collider becomes next step's
            // already-invalid sweep origin. This pass adds no second impulse.
            // Recovery enforces geometry with a small skin. The full node
            // radius is the normal solver's contact target, but overlapping
            // safety margins in a pinch must not push a node through a solid.
            for (std::uint32_t pass = 0U;
                 impl_->rigid_body_count != 0U &&
                 pass < k_soft_contact_cleanup_passes; ++pass) {
                if (pass % 2U == 0U) {
                    soft_body_update_surface<<<
                        (body.surface_vertex_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.positions, body.rest_positions,
                        body.surface_rest_positions, body.surface_bindings,
                        body.surface_positions, body.surface_vertex_count);
                    const std::uint32_t triangles = body.surface_index_count / 3U;
                    soft_body_surface_contacts<<<
                        (triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.surface_positions,
                        body.surface_indices, triangles, impl_->parameters,
                        impl_->states[impl_->current_state], impl_->meshes,
                        impl_->rigid_body_count, body.surface_corner_corrections);
                    soft_body_apply_surface_contacts<<<blocks, block_size, 0, stream>>>(
                        body.positions, body.surface_corner_corrections,
                        body.surface_node_offsets, body.surface_node_influences,
                        body.node_count);
                }
                deformable_collide<true, true><<<blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.previous,
                    body.inverse_masses, body.node_count,
                    body.node_radius * 0.01F,
                    substep_timestep, impl_->parameters, previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->rigid_body_count, body.body_impulses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.contact_momentum_delta,
                    body.contact_normal_delta, body.contact_count,
                    body.dynamic_contact_flag, impl_->fluid_body_contact_flags,
                    pass + 1U == k_soft_contact_cleanup_passes,
                    false, body.maximum_speed);
            }
            const std::uint32_t surface_blocks =
                (body.surface_vertex_count + block_size - 1U) / block_size;
            soft_body_update_surface<<<surface_blocks, block_size, 0, stream>>>(
                body.positions, body.rest_positions,
                body.surface_rest_positions, body.surface_bindings,
                body.surface_positions, body.surface_vertex_count);
            body_status = record_timing_stage(TimingStage::soft_body_contact_cleanup);
            if (!body_status) return body_status;
        }
        const cudaError_t body_error = cudaPeekAtLastError();
        if (body_error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(body_error, "soft-body kernel launch failed");
        }
        return success();
    };
    const auto advance_soft_cloth = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        std::uint32_t iterations = 0U;
        for (const auto &coupling : impl_->soft_cloth_couplings)
            if (coupling && coupling->alive && coupling->options.enabled)
                iterations = std::max(iterations, coupling->options.solver_iterations);
        for (const auto &body : impl_->soft_bodies) if (body && body->alive) {
            const auto clear = cudaMemsetAsync(body->cloth_forces, 0,
                body->node_count * sizeof(Vec3), stream);
            if (clear != cudaSuccess) return cuda_failure(clear, "soft-cloth force clear failed");
        }
        for (const auto &cloth : impl_->cloths) if (cloth && cloth->alive) {
            const auto clear = cudaMemsetAsync(cloth->soft_body_forces, 0,
                cloth->vertex_count * sizeof(Vec3), stream);
            if (clear != cudaSuccess) return cuda_failure(clear, "cloth-soft force clear failed");
        }
        if (iterations == 0U) return success();
        impl_->soft_cloth_kernels_per_substep = 0U;
        for (std::uint32_t pass = 0U; pass < iterations; ++pass) {
            for (const auto &owner : impl_->soft_cloth_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled) continue;
                auto &coupling = *owner;
                auto &body = *impl_->soft_bodies[coupling.options.soft_body.index];
                auto &cloth = *impl_->cloths[coupling.options.cloth.index];
                const auto clear = cudaMemsetAsync(coupling.cloth_contact_counts, 0,
                    cloth.vertex_count * sizeof(std::uint32_t), stream);
                if (clear != cudaSuccess) return cuda_failure(clear, "soft-cloth count clear failed");
                const float distance = coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance : body.node_radius + cloth.thickness;
                const auto blocks = (body.node_count + block_size - 1U) / block_size;
                if (cloth.surface_positions != nullptr) {
                    const auto triangles = cloth.index_count / 3U;
                    cloth_update_surface<<<(triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
                soft_cloth_detect<<<blocks, block_size, 0, stream>>>(
                    body.positions, body.previous, body.velocities, body.inverse_masses,
                    body.node_count, cloth.positions, cloth.previous, cloth.velocities,
                    cloth.inverse_masses, cloth.indices, cloth.surface_positions,
                    cloth.index_count / 3U, distance, coupling.options.friction,
                    coupling.contacts, coupling.cloth_contact_counts);
                soft_cloth_apply_soft<<<blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.cloth_forces, body.inverse_masses,
                    body.node_count, coupling.contacts, coupling.cloth_contact_counts,
                    substep_timestep, body.maximum_speed);
                soft_cloth_apply_cloth<<<
                    (cloth.vertex_count + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions, cloth.velocities,
                    cloth.soft_body_forces, cloth.inverse_masses, cloth.vertex_count,
                    coupling.contacts, body.node_count, coupling.cloth_contact_counts,
                    substep_timestep);
                impl_->soft_cloth_kernels_per_substep += 3U;
            }
            for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
                if (!coupled_cloth(index)) continue;
                auto &cloth = *impl_->cloths[index];
                // Sample once per substep, before projection erases impact strain.
                if (pass == 0U && cloth.surface_positions != nullptr) {
                    cloth_break_bonds<<<(cloth.bond_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions, cloth.bonds,
                        cloth.bond_active, cloth.bond_damage, cloth.bond_count,
                        cloth.break_strain, cloth.fracture_persistence_substeps,
                        cloth.body_impulses, cloth.impact_break_impulse);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
                if (pass + 1U < iterations) {
                    deformable_project_links<<<
                        (cloth.vertex_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions, cloth.scratch,
                        cloth.inverse_masses, cloth.offsets, cloth.neighbors,
                        cloth.bond_active, cloth.vertex_count, substep_timestep, 0.0F);
                    std::swap(cloth.positions, cloth.scratch);
                    ++impl_->soft_cloth_kernels_per_substep;
                    if (cloth.preserve_volume) {
                        cloth_project_volume<<<1U, 1U, 0, stream>>>(
                            cloth.positions, cloth.inverse_masses, cloth.indices,
                            cloth.vertex_count, cloth.index_count / 3U,
                            cloth.volume_gradients, cloth.target_volume,
                            cloth.orientation, cloth.volume_compliance,
                            substep_timestep, cloth.volume_lambda, false);
                        ++impl_->soft_cloth_kernels_per_substep;
                    }
                }
                if (cloth.surface_positions != nullptr) {
                    const auto triangles = cloth.index_count / 3U;
                    cloth_limit_strain<<<1U, 128U, 0, stream>>>(cloth.positions,
                        cloth.scratch, cloth.inverse_masses, cloth.offsets,
                        cloth.neighbors, cloth.free_triangle_nodes, cloth.vertex_count);
                    ++impl_->soft_cloth_kernels_per_substep;
                    cloth_update_surface<<<(triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
            }
            if (pass + 1U < iterations)
                for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
                    if (!coupled_soft(index)) continue;
                    auto &body = *impl_->soft_bodies[index];
                    deformable_project_links<<<
                        (body.node_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.positions, body.scratch,
                        body.inverse_masses, body.offsets, body.neighbors,
                        body.bond_active, body.node_count, substep_timestep,
                        body.maximum_projection_fraction);
                    std::swap(body.positions, body.scratch);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
        }
        for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
            if (!coupled_cloth(index) || impl_->rigid_body_count == 0U) continue;
            auto &cloth = *impl_->cloths[index];
            deformable_collide<false, true><<<
                (cloth.vertex_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
                cloth.positions, cloth.velocities, cloth.previous, cloth.inverse_masses,
                cloth.vertex_count, cloth.thickness, substep_timestep,
                impl_->parameters, previous_states, impl_->states[impl_->current_state],
                impl_->meshes, impl_->rigid_body_count, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true, false, 20.0F);
            ++impl_->soft_cloth_kernels_per_substep;
            if (cloth.surface_positions != nullptr) {
                const auto triangles = cloth.index_count / 3U;
                cloth_update_surface<<<(triangles + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions,
                    cloth.indices,
                    cloth.surface_positions, triangles);
                ++impl_->soft_cloth_kernels_per_substep;
            }
        }
        for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
            if (!coupled_soft(index)) continue;
            auto &body = *impl_->soft_bodies[index];
            if (impl_->rigid_body_count != 0U) {
                deformable_collide<true, true><<<
                    (body.node_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.previous, body.inverse_masses,
                    body.node_count, body.node_radius * 0.01F, substep_timestep,
                    impl_->parameters, previous_states, impl_->states[impl_->current_state],
                    impl_->meshes, impl_->rigid_body_count, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true, false,
                    body.maximum_speed);
                ++impl_->soft_cloth_kernels_per_substep;
            }
            soft_body_update_surface<<<
                (body.surface_vertex_count + block_size - 1U) / block_size,
                block_size, 0, stream>>>(body.positions, body.rest_positions,
                body.surface_rest_positions, body.surface_bindings,
                body.surface_positions, body.surface_vertex_count);
            ++impl_->soft_cloth_kernels_per_substep;
        }
        const auto launch_error = cudaPeekAtLastError();
        if (launch_error != cudaSuccess)
            return cuda_failure(launch_error, "soft-body cloth contact launch failed");
        return record_timing_stage(TimingStage::soft_body_cloth_contacts);
    };
    const auto advance_ropes = [&](const RigidBodyState *previous_states, bool first_substep) -> Status {
        for (const auto &rope : impl_->ropes) {
            if(!rope || !rope->alive)continue;
            int first=-1,last=-1;unsigned dense;
            if(rope->data.options.first.enabled) {
                auto status=impl_->validate_handle(rope->data.options.first.body,dense);
                if(!status)return status;first=int(dense);
            }
            if(rope->data.options.last.enabled) {
                auto status=impl_->validate_handle(rope->data.options.last.body,dense);
                if(!status)return status;last=int(dense);
            }
            rope_advance<<<1,128,0,stream>>>(rope->data,substep_timestep,options.gravity,first,last,
                impl_->parameters,impl_->states[impl_->current_state],previous_states,
                impl_->meshes,impl_->rigid_body_count,first_substep);
            auto error=cudaPeekAtLastError();
            if(error!=cudaSuccess)return cuda_failure(error,"rope solver launch failed");
            auto status=record_timing_stage(TimingStage::rope_solve);
            if(!status)return status;
        }
        return success();
    };
    for (std::uint32_t substep = 0; substep < options.substeps; ++substep) {
        if (impl_->rigid_body_count == 0U) {
            break;
        }
        const std::uint32_t output_state = 1U - impl_->current_state;
        integrate_rigid_bodies_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->accumulators, impl_->targets,
            impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, options.gravity, substep_timestep,
            options.substeps - substep, substep == 0U);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error, "rigid integration kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_integration);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        const std::uint32_t pair_count =
            impl_->rigid_body_count * impl_->rigid_body_count;
        const std::uint32_t pair_block_count =
            (pair_count + block_size - 1U) / block_size;
        const std::uint32_t contact_block_count =
            std::min(pair_count, 128U);
        compute_rigid_world_bounds_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->states[impl_->current_state],
            impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes,
            impl_->rigid_world_bounds);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid world-bounds kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_world_bounds);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        broad_phase_rigid_pairs_kernel<<<pair_block_count, block_size, 0,
                                         stream>>>(
            impl_->parameters, impl_->rigid_world_bounds,
            impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->meshes,
            impl_->rigid_body_count, impl_->rigid_active_pair_flags);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid broad-phase kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_pair_filter);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        const auto pair_indices =
            thrust::make_counting_iterator<std::uint32_t>(0U);
        error = cub::DeviceSelect::Flagged(
            impl_->rigid_broad_phase_workspace,
            impl_->rigid_broad_phase_workspace_size, pair_indices,
            impl_->rigid_active_pair_flags, impl_->rigid_active_pairs,
            impl_->rigid_active_pair_count, static_cast<int>(pair_count),
            stream);
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid broad-phase compaction failed");
        }
        status = record_timing_stage(TimingStage::rigid_pair_compaction);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        generate_rigid_leaf_pairs_kernel<<<contact_block_count, block_size, 0,
                                           stream>>>(
            impl_->parameters, impl_->states[impl_->current_state],
            impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes,
            impl_->options.triangle_mesh_capacity,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->rigid_leaf_pairs,
            impl_->rigid_leaf_pair_counts,
            impl_->rigid_leaf_pair_slot_capacity);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid leaf-pair kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_leaf_pair_generation);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        constexpr std::uint32_t contact_evaluation_threads = 64U;
        evaluate_rigid_leaf_pairs_kernel<<<
            contact_block_count, contact_evaluation_threads,
            contact_evaluation_threads * sizeof(ContactManifold),
            stream>>>(impl_->parameters,
                      impl_->states[impl_->current_state],
                      impl_->states[output_state],
                      impl_->rigid_body_count, impl_->meshes,
                      impl_->rigid_active_pairs,
                      impl_->rigid_active_pair_count,
                      impl_->rigid_leaf_pairs, impl_->rigid_leaf_pair_counts,
                      impl_->rigid_manifolds);
        evaluate_overflow_rigid_pairs_kernel<<<
            contact_block_count, block_size, 0, stream>>>(
                impl_->parameters, impl_->states[impl_->current_state],
                impl_->states[output_state], impl_->rigid_body_count,
                impl_->meshes, impl_->rigid_active_pairs,
                impl_->rigid_active_pair_count,
                impl_->rigid_leaf_pair_counts, impl_->rigid_manifolds);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid leaf contact kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_contact_evaluation);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        prepare_parallel_contact_events_kernel<<<1U, 1U, 0, stream>>>(
            impl_->rigid_body_count, impl_->rigid_manifolds,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->ids, impl_->rigid_contact_event_offsets,
            impl_->rigid_contact_events, impl_->rigid_contact_capacity,
            impl_->rigid_contact_count,
            collect_rigid_contacts, substep == 0U);
        initialize_parallel_colors_kernel<<<
            contact_block_count, block_size, 0, stream>>>(
                impl_->rigid_active_pair_count,
                impl_->rigid_pair_colors, impl_->rigid_color_state);
        for (std::uint32_t color = 0U;
             color < color_round_count; ++color) {
            reset_parallel_color_owners_kernel<<<block_count,
                block_size, 0, stream>>>(
                    impl_->rigid_color_owners,
                    impl_->rigid_body_count);
            find_parallel_color_owners_kernel<<<
                contact_block_count, block_size, 0, stream>>>(
                    impl_->parameters, impl_->rigid_body_count,
                    impl_->rigid_manifolds, impl_->rigid_active_pairs,
                    impl_->rigid_active_pair_count,
                    impl_->rigid_pair_colors,
                    impl_->rigid_color_owners);
            assign_parallel_contact_colors_kernel<<<
                contact_block_count, block_size, 0, stream>>>(
                    impl_->parameters, impl_->rigid_body_count,
                    impl_->rigid_manifolds, impl_->rigid_active_pairs,
                    impl_->rigid_active_pair_count,
                    impl_->rigid_color_owners,
                    impl_->rigid_pair_colors,
                    impl_->rigid_color_state, color, color_round_count);
        }
        for (std::uint32_t pass = 0U; pass < 8U; ++pass) {
            for (std::uint32_t color = 0U;
                 color < color_round_count; ++color) {
                resolve_colored_rigid_contacts_kernel<<<
                    contact_block_count, block_size, 0, stream>>>(
                    impl_->parameters, impl_->states[output_state],
                    impl_->rigid_body_count, impl_->rigid_manifolds,
                    impl_->rigid_active_pairs,
                    impl_->rigid_active_pair_count,
                    impl_->rigid_pair_colors, impl_->rigid_color_state,
                    impl_->rigid_contact_event_offsets,
                    impl_->rigid_contact_events,
                    collect_rigid_contacts
                        ? impl_->rigid_contact_capacity : 0U,
                    color, pass == 0U);
            }
            resolve_uncolored_rigid_contacts_kernel<<<1U, 1U, 0, stream>>>(
                impl_->parameters, impl_->states[output_state],
                impl_->rigid_body_count, impl_->rigid_manifolds,
                impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
                impl_->rigid_pair_colors, impl_->rigid_color_state,
                impl_->rigid_contact_event_offsets,
                impl_->rigid_contact_events,
                collect_rigid_contacts
                    ? impl_->rigid_contact_capacity : 0U,
                pass == 0U);
        }
        clamp_rigid_speeds_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->states[output_state],
            impl_->rigid_body_count);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid contact solve kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_contact_solve);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        impl_->current_state = output_state;
        // Couple the sheet to the rigid state from this same substep.
        if (has_cloth) {
            status = advance_cloth(impl_->states[1U - impl_->current_state]);
            if (!status) return status;
        }
        if (has_soft_body) {
            status = advance_soft_bodies(
                impl_->states[1U - impl_->current_state]);
            if (!status) return status;
        }
        status = advance_soft_cloth(impl_->states[1U - impl_->current_state]);
        if (!status) return status;
        status = advance_ropes(impl_->states[1U - impl_->current_state], substep == 0);
        if (!status) return status;
    }

    if (impl_->rigid_body_count > 0U) {
        clear_rigid_inputs_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->accumulators, impl_->targets, impl_->rigid_body_count);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error, "rigid input-clear kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_input_clear);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
    }

    if (impl_->rigid_body_count == 0U) {
        for (std::uint32_t substep = 0U; substep < options.substeps; ++substep) {
            if (has_cloth) {
                status = advance_cloth(nullptr);
                if (!status) return status;
            }
            if (has_soft_body) {
                status = advance_soft_bodies(nullptr);
                if (!status) return status;
            }
            status = advance_soft_cloth(nullptr);
            if (!status) return status;
            status = advance_ropes(nullptr, substep == 0);
            if (!status) return status;
        }
    }

    if (any_moving_body) {
        fluid_body_bounds_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->fluid_previous_states,
            impl_->states[impl_->current_state], impl_->meshes,
            impl_->rigid_body_count, impl_->fluid_body_bounds);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return cuda_failure(error, "fluid body bounds launch failed");
    }

    // Fluid storage is capacity-sized and all sort/selection workspaces were
    // reserved at creation. No CUDA allocations occur during a frame.
    for (const auto &body : impl_->soft_bodies) {
        if (!body || !body->alive) continue;
        error = cudaMemsetAsync(body->fluid_forces, 0, body->node_count * sizeof(Vec3), stream);
        if (error != cudaSuccess) return cuda_failure(error, "soft-body fluid force clear failed");
    }
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        error = cudaMemsetAsync(coupling->contact_count, 0, sizeof(std::uint32_t), stream);
        if (error == cudaSuccess) error = cudaMemsetAsync(coupling->maximum_penetration, 0,
            sizeof(float), stream);
        if (error != cudaSuccess) return cuda_failure(error, "fluid soft-body diagnostics clear failed");
    }
    for (std::uint32_t fluid_index = 0U;
         fluid_index < impl_->fluids.size(); ++fluid_index) {
        if (!impl_->fluids[fluid_index] || !impl_->fluids[fluid_index]->alive)
            continue;
        FluidStorage &fluid = *impl_->fluids[fluid_index];
        const FluidId fluid_id{fluid_index, fluid.generation};
        std::uint32_t live = *fluid.count;
        const std::uint32_t first_spawned = live;
        bool spawned = false;
        const std::uint32_t blocks =
            (fluid.options.capacity + block_size - 1U) / block_size;
        float source_cell_size = fluid.options.support_radius;
        bool has_sources = false;
        for (const auto &slot : impl_->particle_sources) {
            if (!slot.alive || !slot.options.enabled || !(slot.options.fluid == fluid_id)) continue;
            has_sources = true;
            source_cell_size = std::max(source_cell_size, slot.data->spacing);
        }
        if (has_sources) {
            if (fluid.next_id > UINT32_MAX - (fluid.options.capacity - live))
                return failure(StatusCode::capacity_exceeded, "stable fluid particle ID range exhausted");
            fluid_emit_cells<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.count, fluid.options.capacity,
                1.0F/source_cell_size, fluid.keys[0], fluid.indices[0]);
            error = cub::DeviceRadixSort::SortPairs(
                fluid.sort_workspace, fluid.sort_workspace_size,
                fluid.keys[0], fluid.keys[1], fluid.indices[0], fluid.indices[1],
                fluid.options.capacity, 0, 64, stream);
            if (error != cudaSuccess) return cuda_failure(error, "fluid source spatial index failed");
        }
        for (ParticleSourceSlot &slot : impl_->particle_sources) {
            if (!slot.alive || !slot.options.enabled ||
                !(slot.options.fluid == fluid_id)) continue;
            const auto &data = *slot.data;
            fluid_source_vacancies<<<(data.count+block_size-1U)/block_size,block_size,0,stream>>>(
                data.points, data.count, data.spacing, fluid.positions,
                fluid.keys[1], fluid.indices[1], fluid.options.capacity,
                1.0F/source_cell_size, data.vacant);
            fluid_source_emit<<<1U,1U,0,stream>>>(
                data.points, data.vacant, data.count, data.spacing,
                slot.options.initial_velocity, fluid.positions, fluid.velocities,
                fluid.ids, fluid.foam, fluid.count, fluid.options.capacity,
                first_spawned, fluid.next_id, data.capacity_misses);
        }
        if (has_sources) {
            // One readback per fluid, never one per sample. No frame allocation.
            error = cudaStreamSynchronize(stream);
            if (error != cudaSuccess) return cuda_failure(error, "fluid source emission failed");
            live = *fluid.count;
            const auto amount = live - first_spawned;
            spawned = amount != 0;
            fluid.next_id += amount;
            fluid.emitted_count += amount;
            impl_->emitted_particle_count += amount;
            for (const auto &slot : impl_->particle_sources)
                if (slot.alive && slot.options.enabled && slot.options.fluid == fluid_id)
                    impl_->spawn_capacity_miss_count += *slot.data->capacity_misses;
            status = record_timing_stage(TimingStage::fluid_spawn);
            if (!status) return status;
        }
        if (live == 0U) continue;
        const std::uint32_t iterations =
            options.substeps * fluid.options.solver_iterations;
        const float dt = options.timestep / static_cast<float>(iterations);
        const float diameter = 2.0F * fluid.options.particle_radius;
        const float particle_mass = fluid.options.rest_density *
            (fluid.options.rest_particle_volume > 0.0F
                ? fluid.options.rest_particle_volume
                : diameter * diameter * diameter);
        const std::uint32_t body_words =
            (impl_->options.rigid_body_capacity + 63U) / 64U;
        if (any_moving_body) {
            error = cudaMemsetAsync(impl_->fluid_body_masks, 0,
                k_fluid_body_buckets * body_words * sizeof(unsigned long long),
                stream);
            if (error == cudaSuccess)
                error = cudaMemsetAsync(impl_->fluid_global_body_masks, 0,
                    body_words * sizeof(unsigned long long), stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid body index clear failed");
            fluid_index_body_cells<<<block_count, block_size, 0, stream>>>(
                impl_->parameters, impl_->fluid_body_bounds,
                impl_->rigid_body_count, body_words,
                fluid.options.particle_radius + fluid.options.maximum_speed * dt,
                impl_->fluid_body_masks, impl_->fluid_global_body_masks);
            status = record_timing_stage(TimingStage::fluid_body_index);
            if (!status) return status;
        }
        if (collect_fluid_contacts) {
            error = cudaMemsetAsync(fluid.contact_flags, 0,
                fluid.options.capacity * sizeof(std::uint8_t), stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid event flags clear failed");
        }
        for (std::uint32_t iteration = 0U; iteration < iterations;
             ++iteration) {
            fluid_emit_cells<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.count, fluid.options.capacity,
                1.0F / fluid.options.support_radius,
                fluid.keys[0], fluid.indices[0]);
            error = cub::DeviceRadixSort::SortPairs(
                fluid.sort_workspace, fluid.sort_workspace_size,
                fluid.keys[0], fluid.keys[1], fluid.indices[0],
                fluid.indices[1], fluid.options.capacity, 0, 64, stream);
            if (error != cudaSuccess) {
                cudaStreamSynchronize(stream);
                return cuda_failure(error, "fluid neighbor cell sort failed");
            }
            status = record_timing_stage(TimingStage::fluid_neighbor_sort);
            if (!status) return status;
            fluid_compute_forces<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.velocities, fluid.count,
                fluid.keys[1], fluid.indices[1], fluid.foam,
                fluid.options, normalized_or(multiply(options.gravity, -1.0F),
                                             {0.0F, 1.0F, 0.0F}),
                fluid.forces, fluid.foam_source,
                impl_->fluid_neighbor_overflow,
                impl_->fluid_maximum_neighbor_count);
            status = record_timing_stage(TimingStage::fluid_neighbor_forces);
            if (!status) return status;
            for (FluidClothCouplingResource &coupling :
                 impl_->fluid_cloth_couplings) {
                if (!coupling.alive || !coupling.options.enabled ||
                    !(coupling.options.fluid == fluid_id)) continue;
                if (coupling.options.cloth.index >= impl_->cloths.size())
                    continue;
                const auto &cloth_pointer =
                    impl_->cloths[coupling.options.cloth.index];
                if (!cloth_pointer || !cloth_pointer->alive ||
                    cloth_pointer->generation !=
                        coupling.options.cloth.generation)
                    continue;
                ClothStorage &cloth = *cloth_pointer;
                FluidClothCouplingOptions resolved = coupling.options;
                if (resolved.contact_distance == 0.0F)
                    resolved.contact_distance =
                        fluid.options.particle_radius + cloth.thickness;
                if (resolved.interaction_radius == 0.0F)
                    resolved.interaction_radius = fluid.options.support_radius;
                error = cudaMemsetAsync(cloth.fluid_forces, 0,
                    cloth.vertex_count * sizeof(Vec3), stream);
                if (error != cudaSuccess)
                    return cuda_failure(error,
                        "fluid-cloth reaction clear failed");
                fluid_cloth_containment_forces<<<
                    blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.velocities, fluid.count,
                        fluid.forces, fluid.foam_source, particle_mass,
                        cloth.positions, cloth.velocities, cloth.indices,
                        cloth.index_count / 3U, cloth.orientation, resolved,
                        cloth.fluid_forces);
                const std::uint32_t cloth_blocks =
                    (cloth.vertex_count + block_size - 1U) / block_size;
                cloth_apply_fluid_forces<<<
                    cloth_blocks, block_size, 0, stream>>>(
                        cloth.positions, cloth.velocities,
                        cloth.inverse_masses, cloth.fluid_forces,
                        cloth.vertex_count, dt);
                deformable_project_links<<<cloth_blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.scratch, cloth.inverse_masses,
                    cloth.offsets, cloth.neighbors, cloth.bond_active,
                    cloth.vertex_count, dt, 0.0F);
                std::swap(cloth.positions, cloth.scratch);
                if (cloth.preserve_volume) {
                    cloth_project_volume<<<1U, 1U, 0, stream>>>(
                        cloth.positions, cloth.inverse_masses, cloth.indices,
                        cloth.vertex_count, cloth.index_count / 3U,
                        cloth.volume_gradients, cloth.target_volume,
                        cloth.orientation, cloth.volume_compliance, dt,
                        cloth.volume_lambda, true);
                }
                status = record_timing_stage(
                    TimingStage::fluid_cloth_contacts);
                if (!status) return status;
            }
            fluid_integrate<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.velocities, fluid.previous,
                fluid.foam, fluid.forces, fluid.foam_source,
                fluid.count, fluid.options,
                options.gravity, dt);
            status = record_timing_stage(TimingStage::fluid_integration);
            if (!status) return status;
            for (const auto &owner : impl_->fluid_soft_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    !(owner->options.fluid == fluid_id)) continue;
                auto &coupling = *owner;
                auto &body = *impl_->soft_bodies[coupling.options.soft_body.index];
                const auto node_blocks = (body.node_count + block_size - 1U) / block_size;
                const auto skin_blocks = (body.surface_vertex_count + block_size - 1U) / block_size;
                const float distance = coupling.options.contact_distance > 0
                    ? coupling.options.contact_distance : fluid.options.particle_radius;
                const auto refit_surface = [&]() -> Status {
                    const auto clear = cudaMemsetAsync(coupling.ready, 0,
                        coupling.tree_count * sizeof(std::uint32_t), stream);
                    if (clear != cudaSuccess) return cuda_failure(clear, "soft surface BVH clear failed");
                    fluid_soft_refit<<<(coupling.tree_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.surface_positions, coupling.previous_surface,
                        body.surface_indices, coupling.triangle_order, coupling.tree, coupling.parents,
                        coupling.ready, coupling.tree_count, coupling.bounds);
                    return success();
                };
                for (std::uint32_t pass = 0; pass < coupling.options.solver_iterations; ++pass) {
                    error = cudaMemsetAsync(coupling.counts, 0,
                        body.node_count * sizeof(std::uint32_t), stream);
                    if (error == cudaSuccess) error = cudaMemsetAsync(coupling.position_deltas, 0,
                        body.node_count * sizeof(Vec3), stream);
                    if (error == cudaSuccess) error = cudaMemsetAsync(coupling.impulses, 0,
                        body.node_count * sizeof(Vec3), stream);
                    if (error != cudaSuccess) return cuda_failure(error, "fluid soft-body scratch clear failed");
                    status = refit_surface();
                    if (!status) return status;
                    fluid_soft_detect<<<blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.previous, fluid.velocities, fluid.count,
                        body.surface_positions, coupling.previous_surface, body.surface_indices,
                        body.surface_index_count, body.surface_bindings, body.velocities,
                        coupling.orientation, distance, coupling.bounds, pass == 0,
                        coupling.tree, coupling.triangle_order,
                        coupling.contacts, coupling.counts, coupling.contact_count,
                        coupling.maximum_penetration);
                    fluid_soft_solve<<<blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.velocities, fluid.forces, fluid.foam, fluid.count,
                        coupling.contacts, coupling.counts, body.inverse_masses,
                        body.velocities, body.maximum_speed,
                        1.0F / particle_mass, coupling.options.friction,
                        0.2F * body.node_radius, dt, coupling.position_deltas, coupling.impulses);
                    fluid_soft_apply<<<node_blocks, block_size, 0, stream>>>(
                        body.positions, body.velocities, body.inverse_masses,
                        coupling.position_deltas, coupling.impulses, body.fluid_forces,
                        body.node_count, 1.0F / options.timestep);
                    deformable_project_links<<<node_blocks, block_size, 0, stream>>>(
                        body.positions, body.scratch, body.inverse_masses, body.offsets,
                        body.neighbors, body.bond_active, body.node_count, dt,
                        body.maximum_projection_fraction);
                    std::swap(body.positions, body.scratch);
                    // Keep fluid reaction corrections on the valid side of
                    // passive/dynamic rigid geometry before updating the skin.
                    if (impl_->rigid_body_count != 0U)
                        deformable_collide<true, true><<<node_blocks, block_size, 0, stream>>>(
                            body.positions, body.velocities, body.previous, body.inverse_masses,
                            body.node_count, body.node_radius * 0.01F, dt, impl_->parameters,
                            nullptr, impl_->states[impl_->current_state], impl_->meshes,
                            impl_->rigid_body_count, body.body_impulses, body.rigid_contact_forces,
                            body.contact_normals, body.contact_arms, body.contact_momentum_delta,
                            body.contact_normal_delta, body.contact_count, body.dynamic_contact_flag,
                            impl_->fluid_body_contact_flags, false, false, body.maximum_speed);
                    soft_body_update_surface<<<skin_blocks, block_size, 0, stream>>>(
                        body.positions, body.rest_positions, body.surface_rest_positions,
                        body.surface_bindings, body.surface_positions, body.surface_vertex_count);
                }
                status = refit_surface();
                if (!status) return status;
                fluid_soft_recover<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.previous, fluid.count, body.surface_positions,
                    coupling.previous_surface, body.surface_indices, body.surface_index_count,
                    coupling.orientation, distance, coupling.bounds, true,
                    coupling.tree, coupling.triangle_order);
                error = cudaMemcpyAsync(coupling.previous_surface, body.surface_positions,
                    body.surface_vertex_count * sizeof(Vec3), cudaMemcpyDeviceToDevice, stream);
                if (error != cudaSuccess) return cuda_failure(error, "fluid soft-body skin copy failed");
                status = record_timing_stage(TimingStage::fluid_soft_body_contacts,
                    (6U + (impl_->rigid_body_count != 0U ? 1U : 0U)) *
                    coupling.options.solver_iterations + 2U);
                if (!status) return status;
            }
            for (const FluidClothCouplingResource &coupling :
                 impl_->fluid_cloth_couplings) {
                if (!coupling.alive || !coupling.options.enabled ||
                    !(coupling.options.fluid == fluid_id) ||
                    coupling.options.cloth.index >= impl_->cloths.size())
                    continue;
                const auto &cloth_pointer =
                    impl_->cloths[coupling.options.cloth.index];
                if (!cloth_pointer || !cloth_pointer->alive ||
                    cloth_pointer->generation !=
                        coupling.options.cloth.generation)
                    continue;
                const ClothStorage &cloth = *cloth_pointer;
                const float contact_distance =
                    coupling.options.contact_distance > 0.0F
                        ? coupling.options.contact_distance
                        : fluid.options.particle_radius + cloth.thickness;
                fluid_project_inside_cloth<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.count,
                    contact_distance, cloth.positions, cloth.velocities,
                    cloth.indices, cloth.index_count / 3U,
                    cloth.orientation);
            }
            if (any_moving_body) {
                error = cudaMemsetAsync(impl_->fluid_body_contact_flags, 0,
                    impl_->rigid_body_count * sizeof(std::uint32_t), stream);
                if (error != cudaSuccess)
                    return cuda_failure(error, "fluid contact flags clear failed");
                fluid_moving_contacts<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.previous,
                    fluid.foam, fluid.count, fluid.options.particle_radius,
                    particle_mass, dt, fluid.options.maximum_speed,
                    impl_->parameters,
                    impl_->fluid_previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->fluid_body_bounds, impl_->rigid_body_count,
                    impl_->fluid_body_masks, impl_->fluid_global_body_masks,
                    body_words, iteration == 0U, fluid.body_impulses,
                    impl_->fluid_body_contact_flags,
                    collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags, fluid_id, impl_->ids,
                    impl_->paint_rule_count != 0U,
                    impl_->paint_fields, impl_->options.paint_field_capacity,
                    impl_->paint_rules, impl_->options.paint_rule_capacity);
                reduce_point_body_impulses<<<impl_->rigid_body_count,
                                             block_size, 0, stream>>>(
                    fluid.body_impulses, fluid.count, impl_->parameters,
                    impl_->states[impl_->current_state],
                    impl_->fluid_body_contact_flags,
                    impl_->rigid_body_count);
                status = record_timing_stage(TimingStage::fluid_moving_contacts);
                if (!status) return status;
            }
            bool any_static_body = false;
            for (std::uint32_t body = 0U; body < impl_->rigid_body_count;
                 ++body) {
                if (impl_->parameters[body].motion != MotionType::static_body)
                    continue;
                any_static_body = true;
                fluid_static_contacts<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.previous,
                    fluid.foam, fluid.count, fluid.options.particle_radius,
                    fluid.options.support_radius, first_spawned,
                    spawned && iteration == 0U,
                    normalized_or(multiply(options.gravity, -1.0F),
                                  {0.0F, 1.0F, 0.0F}),
                    impl_->parameters, impl_->states[impl_->current_state],
                    impl_->meshes, body, particle_mass,
                    collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags, fluid_id, impl_->ids[body],
                    impl_->paint_rule_count != 0U,
                    impl_->paint_fields, impl_->options.paint_field_capacity,
                    impl_->paint_rules, impl_->options.paint_rule_capacity);
            }
            if (any_static_body) {
                status = record_timing_stage(TimingStage::fluid_static_contacts);
                if (!status) return status;
            }
            // A rigid boundary can push water back into a neighboring soft
            // face. Finish with current-skin recovery, not the old surface or
            // an infinite triangle plane. No extra kinetic impulse is added.
            for (const auto &coupling : impl_->fluid_soft_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    !(coupling->options.fluid == fluid_id)) continue;
                const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
                const float distance = coupling->options.contact_distance > 0
                    ? coupling->options.contact_distance : fluid.options.particle_radius;
                for (std::uint32_t recovery = 0; recovery < 3; ++recovery)
                    fluid_soft_recover<<<blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.positions, fluid.count, body.surface_positions,
                        body.surface_positions, body.surface_indices, body.surface_index_count,
                        coupling->orientation, distance, coupling->bounds, false,
                        coupling->tree, coupling->triangle_order);
                status = record_timing_stage(TimingStage::fluid_soft_body_contacts, 3U);
                if (!status) return status;
            }
            bool any_destroy_plane = false;
            for (const DestroyPlaneSlot &slot : impl_->destroy_planes) {
                if (!slot.alive || !slot.options.enabled ||
                    !(slot.options.fluid == fluid_id)) continue;
                fluid_destroy_flags<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.previous, fluid.count,
                    fluid.options.capacity, slot.options,
                    any_destroy_plane, fluid.keep);
                any_destroy_plane = true;
            }
            if (any_destroy_plane) {
                const auto sequence =
                    thrust::make_counting_iterator<std::uint32_t>(0U);
                error = cub::DeviceSelect::Flagged(
                    fluid.select_workspace, fluid.select_workspace_size,
                    sequence, fluid.keep, fluid.selected, fluid.count,
                    fluid.options.capacity, stream);
                if (error != cudaSuccess) {
                    cudaStreamSynchronize(stream);
                    return cuda_failure(error, "fluid particle compaction failed");
                }
                fluid_gather<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.ids, fluid.foam,
                    fluid.contact_samples, fluid.contact_flags,
                    collect_fluid_contacts,
                    fluid.selected, fluid.count, fluid.options.capacity,
                    fluid.next_positions, fluid.next_velocities,
                    fluid.next_ids, fluid.next_foam,
                    fluid.next_contact_samples, fluid.next_contact_flags);
                std::swap(fluid.positions, fluid.next_positions);
                std::swap(fluid.velocities, fluid.next_velocities);
                std::swap(fluid.ids, fluid.next_ids);
                std::swap(fluid.foam, fluid.next_foam);
                std::swap(fluid.contact_samples, fluid.next_contact_samples);
                std::swap(fluid.contact_flags, fluid.next_contact_flags);
                status = record_timing_stage(
                    TimingStage::fluid_outflow_compaction);
                if (!status) return status;
            }
        }
        if (collect_fluid_contacts) {
            const auto sequence =
                thrust::make_counting_iterator<std::uint32_t>(0U);
            error = cub::DeviceSelect::Flagged(
                fluid.select_workspace, fluid.select_workspace_size,
                sequence, fluid.contact_flags, fluid.selected,
                fluid.contact_count, fluid.options.capacity, stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid contact selection failed");
            fluid_reserve_contact_events<<<1U, 1U, 0, stream>>>(
                fluid.contact_count, fluid.contact_offset,
                impl_->fluid_contact_count, impl_->fluid_contact_overflow,
                impl_->options.contact_capacity);
            fluid_gather_contact_events<<<blocks, block_size, 0, stream>>>(
                fluid.selected, fluid.contact_count, fluid.contact_offset,
                fluid.contact_samples, fluid.ids, impl_->ids, fluid_id,
                impl_->fluid_contact_events, impl_->options.contact_capacity);
            status = record_timing_stage(TimingStage::fluid_contact_events);
            if (!status) return status;
        }
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error, "fluid kernel launch failed");
        }
    }
    if (options.collect_kernel_timings && timing_boundary == 1U) {
        error = cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        if (error != cudaSuccess)
            return cuda_failure(error, "failed to finish empty kernel timing");
    }
    error = cudaEventRecord(frame->event, stream);
    if (error != cudaSuccess) {
        cudaStreamSynchronize(stream);
        return cuda_failure(error, "failed to record frame completion event");
    }

    ++impl_->frame_index;
    ++impl_->revision;
    if (options.collect_kernel_timings) {
        impl_->timing_available = true;
        impl_->timing_boundary_count = timing_boundary;
        impl_->timing_frame_index = impl_->frame_index;
    }
    impl_->frame = frame;
    token_impl->completion = std::move(frame);
    completion.impl_ = std::move(token_impl);
    return success();
}

Status World::step(StepOptions options, cudaStream_t stream) noexcept {
    FrameToken completion;
    Status status = step_async(options, completion, stream);
    if (!status) {
        return status;
    }
    return completion.wait();
}

ContactDeviceView World::contacts() const noexcept {
    if (!impl_ || (impl_->frame && !impl_->frame->acknowledged)) return {};
    const std::uint32_t count = *impl_->fluid_contact_count;
    return {{impl_->fluid_contact_events, count}, count,
            *impl_->fluid_contact_overflow != 0U, impl_->frame_index};
}

RigidContactDeviceView World::rigid_contacts() const noexcept {
    if (!impl_ || (impl_->frame && !impl_->frame->acknowledged)) {
        return {};
    }
    return {{impl_->rigid_contact_events, *impl_->rigid_contact_count},
            *impl_->rigid_contact_count, impl_->frame_index};
}

Status World::physics_debug_frame(
    PhysicsDebugFrameView &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->options.physics_debug.frame_capacity == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture was not enabled at world creation");
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    if (impl_->debug_frame_count == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture has no completed frame");
    }
    const std::size_t index =
        (impl_->debug_next_frame + impl_->debug_frames.size() - 1U) %
        impl_->debug_frames.size();
    const PhysicsDebugFrame &frame = impl_->debug_frames[index];
    output.frame_index = frame.frame_index;
    output.timestep = frame.timestep;
    output.gravity = frame.gravity;
    output.maximum_fluid_neighbor_count =
        frame.maximum_fluid_neighbor_count;
    output.rigid_bodies = {
        frame.rigid_bodies.data(), frame.rigid_bodies.size()};
    output.fluid_particles = {
        frame.fluid_particles.data(), frame.fluid_particles.size()};
    output.cloth_vertices = {
        frame.cloth_vertices.data(), frame.cloth_vertices.size()};
    output.soft_body_nodes = {
        frame.soft_body_nodes.data(), frame.soft_body_nodes.size()};
    output.rope_nodes = {frame.rope_nodes.data(), frame.rope_nodes.size()};
    output.rigid_contacts = {
        frame.rigid_contacts.data(), frame.rigid_contacts.size()};
    output.fluid_contacts = {
        frame.fluid_contacts.data(), frame.fluid_contacts.size()};
    return success();
}

Status World::copy_physics_debug_capture(
    PhysicsDebugCapture &output) const noexcept {
    output.frames.clear();
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->options.physics_debug.frame_capacity == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture was not enabled at world creation");
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    try {
        output.frames.reserve(impl_->debug_frame_count);
        const std::size_t first =
            (impl_->debug_next_frame + impl_->debug_frames.size() -
             impl_->debug_frame_count) % impl_->debug_frames.size();
        for (std::size_t offset = 0U;
             offset < impl_->debug_frame_count; ++offset) {
            output.frames.push_back(impl_->debug_frames[
                (first + offset) % impl_->debug_frames.size()]);
        }
    } catch (...) {
        output.frames.clear();
        return failure(StatusCode::out_of_memory,
                       "physics debug capture copy failed");
    }
    return success();
}

Status World::collect_step_timings(WorldStepTimings &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) {
        return status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    if (!impl_->timing_available || impl_->timing_boundary_count < 2U) {
        output.frame_index = impl_->frame_index;
        return success();
    }

    output.frame_index = impl_->timing_frame_index;
    output.available = true;
    cudaError_t error = cudaEventElapsedTime(
        &output.total_gpu_milliseconds, impl_->timing_events[0],
        impl_->timing_events[impl_->timing_boundary_count - 1U]);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to collect total kernel timing");
    }
    for (std::size_t index = 0; index < impl_->timing_stages.size(); ++index) {
        float milliseconds = 0.0F;
        error = cudaEventElapsedTime(&milliseconds, impl_->timing_events[index],
                                     impl_->timing_events[index + 1U]);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to collect kernel stage timing");
        }
        KernelTiming *timing = nullptr;
        bool contact_generation_stage = false;
        switch (impl_->timing_stages[index]) {
        case TimingStage::rigid_integration:
            timing = &output.rigid_integration;
            break;
        case TimingStage::rigid_world_bounds:
            timing = &output.rigid_world_bounds;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_pair_filter:
            timing = &output.rigid_pair_filter;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_pair_compaction:
            timing = &output.rigid_pair_compaction;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_leaf_pair_generation:
            timing = &output.rigid_leaf_pair_generation;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_contact_evaluation:
            timing = &output.rigid_contact_evaluation;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_contact_solve:
            timing = &output.rigid_contact_solve;
            break;
        case TimingStage::rigid_input_clear:
            timing = &output.rigid_input_clear;
            break;
        case TimingStage::rope_solve:
            timing = &output.rope_solve;
            break;
        case TimingStage::cloth_prediction:
            timing = &output.cloth_prediction;
            break;
        case TimingStage::cloth_constraints:
            timing = &output.cloth_constraints;
            break;
        case TimingStage::cloth_contacts:
            timing = &output.cloth_contacts;
            break;
        case TimingStage::soft_body_prediction:
            timing = &output.soft_body_prediction;
            break;
        case TimingStage::soft_body_constraints:
            timing = &output.soft_body_constraints;
            break;
        case TimingStage::soft_body_contacts:
        case TimingStage::soft_body_contact_cleanup:
            timing = &output.soft_body_contacts;
            break;
        case TimingStage::soft_body_cloth_contacts:
            timing = &output.soft_body_cloth_contacts;
            break;
        case TimingStage::fluid_cloth_contacts:
            timing = &output.fluid_cloth_contacts;
            break;
        case TimingStage::fluid_soft_body_contacts:
            timing = &output.fluid_soft_body_contacts;
            break;
        case TimingStage::fluid_spawn:
            timing = &output.fluid_spawn;
            break;
        case TimingStage::fluid_neighbor_sort:
            timing = &output.fluid_neighbor_sort;
            break;
        case TimingStage::fluid_neighbor_forces:
            timing = &output.fluid_neighbor_forces;
            break;
        case TimingStage::fluid_integration:
            timing = &output.fluid_integration;
            break;
        case TimingStage::fluid_static_contacts:
            timing = &output.fluid_static_contacts;
            break;
        case TimingStage::fluid_body_index:
            timing = &output.fluid_body_index;
            break;
        case TimingStage::fluid_moving_contacts:
            timing = &output.fluid_moving_contacts;
            break;
        case TimingStage::fluid_contact_events:
            timing = &output.fluid_contact_events;
            break;
        case TimingStage::fluid_outflow_compaction:
            timing = &output.fluid_outflow_compaction;
            break;
        }
        timing->total_milliseconds += milliseconds;
        const std::uint32_t launches =
            impl_->timing_launch_counts[index] != 0U ? impl_->timing_launch_counts[index] :
            impl_->timing_stages[index] == TimingStage::soft_body_cloth_contacts
                ? impl_->soft_cloth_kernels_per_substep
                : impl_->timing_stages[index] == TimingStage::soft_body_contact_cleanup
                ? 2U + (impl_->rigid_body_count != 0U
                    ? k_soft_contact_cleanup_passes * 5U / 2U : 0U)
                : impl_->timing_stages[index] == TimingStage::rigid_contact_solve
                ? impl_->rigid_solve_kernels_per_substep
                : impl_->timing_stages[index] ==
                          TimingStage::rigid_contact_evaluation
                    ? 2U : 1U;
        timing->launch_count += launches;
        if (contact_generation_stage) {
            output.rigid_contact_generation.total_milliseconds += milliseconds;
            output.rigid_contact_generation.launch_count += launches;
        }
    }
    return success();
}

Status World::collect_statistics(WorldStatistics &output,
                                 cudaStream_t stream) const noexcept {
    (void)stream;
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status device_status = impl_->require_current_device();
    if (!device_status) {
        return device_status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        Status status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    const std::size_t capacity = impl_->options.rigid_body_capacity;
    output = {};
    output.frame_index = impl_->frame_index;
    output.fluid_count = impl_->fluid_count;
    for (const auto &cloth : impl_->cloths) {
        if (!cloth || !cloth->alive) continue;
        ++output.cloth_count;
        output.cloth_vertex_count += cloth->vertex_count;
        output.allocated_bytes +=
            static_cast<std::size_t>(cloth->vertex_capacity) *
                (6U * sizeof(Vec3) + sizeof(float) + sizeof(FluidBodyImpulse)) +
            cloth->index_count * sizeof(std::uint32_t) +
            (static_cast<std::size_t>(cloth->vertex_capacity) + 1U) *
                sizeof(std::uint32_t) +
            cloth->neighbor_capacity * sizeof(DeformableNeighbor) +
            cloth->bond_count * (sizeof(ClothBond) +
                2U * sizeof(std::uint8_t)) +
            (cloth->surface_positions != nullptr
                ? cloth->index_count * (sizeof(Vec3) +
                    3U * sizeof(std::uint32_t)) +
                    cloth->vertex_capacity * (sizeof(std::uint32_t) + sizeof(std::uint8_t)) : 0U) +
            impl_->options.rigid_body_capacity *
                sizeof(ClothBodyCorrection) +
            (cloth->volume_gradients != nullptr
                ? static_cast<std::size_t>(cloth->vertex_count) * sizeof(Vec3) +
                    sizeof(float) : 0U) +
            (cloth->fluid_forces != nullptr
                ? static_cast<std::size_t>(cloth->vertex_capacity) * sizeof(Vec3)
                : 0U) +
            sizeof(std::uint32_t);
    }
    for (const auto &body : impl_->soft_bodies) {
        if (!body || !body->alive) continue;
        ++output.soft_body_count;
        output.soft_body_node_count += body->node_count;
        output.allocated_bytes +=
            static_cast<std::size_t>(body->node_count) *
                (14U * sizeof(Vec3) + 2U * sizeof(float) +
                 sizeof(FluidBodyImpulse)) +
            static_cast<std::size_t>(body->bond_count) *
                (sizeof(SoftBodyBond) + sizeof(std::uint8_t)) +
            (static_cast<std::size_t>(body->node_count) + 1U) *
                sizeof(std::uint32_t) +
            static_cast<std::size_t>(body->neighbor_count) *
                sizeof(DeformableNeighbor) +
            static_cast<std::size_t>(body->surface_vertex_count) *
                (2U * sizeof(Vec3) + sizeof(SoftBodySurfaceBinding)) +
            static_cast<std::size_t>(body->surface_index_count) *
                sizeof(std::uint32_t) + 3U * sizeof(std::uint32_t) +
                sizeof(Vec3) + sizeof(Quaternion);
    }
    for (const auto &coupling : impl_->soft_cloth_couplings) {
        if (!coupling || !coupling->alive) continue;
        output.allocated_bytes +=
            impl_->soft_bodies[coupling->options.soft_body.index]->node_count *
                sizeof(SoftClothContact) +
            impl_->cloths[coupling->options.cloth.index]->vertex_capacity *
                sizeof(std::uint32_t);
    }
    output.emitted_particle_count = impl_->emitted_particle_count;
    for(const auto &rope:impl_->ropes) {
        if(!rope || !rope->alive)continue;
        ++output.rope_count;output.rope_node_count+=rope->data.count;
        output.allocated_bytes+=rope->data.count*(9*sizeof(Vec3)+2*sizeof(float))+
            2*impl_->options.rigid_body_capacity*sizeof(Vec3);
    }
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
        const auto &fluid = *impl_->fluids[coupling->options.fluid.index];
        output.allocated_bytes += fluid.options.capacity * sizeof(FluidSoftContact) +
            body.node_count * (sizeof(std::uint32_t) + 2U * sizeof(Vec3)) +
            body.surface_vertex_count * sizeof(Vec3) + 2U * sizeof(Vec3) +
            sizeof(std::uint32_t) + sizeof(float);
        output.fluid_soft_body_contact_count += *coupling->contact_count;
        output.maximum_fluid_soft_body_penetration = std::max(
            output.maximum_fluid_soft_body_penetration, *coupling->maximum_penetration);
    }
    output.destroyed_particle_count = impl_->destroyed_particle_count;
    output.spawn_capacity_miss_count = impl_->spawn_capacity_miss_count;
    for (const auto &source : impl_->particle_sources)
        if (source.alive)
            output.allocated_bytes += source.data->count * (sizeof(Vec3) + sizeof(std::uint8_t)) + sizeof(std::uint32_t);
    output.contact_count = *impl_->fluid_contact_count;
    output.contact_overflow_count = *impl_->fluid_contact_overflow;
    output.maximum_fluid_neighbor_count =
        *impl_->fluid_maximum_neighbor_count;
    for (const auto &fluid : impl_->fluids) {
        if (!fluid || !fluid->alive) continue;
        const std::uint32_t live = *fluid->count;
        output.particle_count += live;
        output.destroyed_particle_count +=
            fluid->initial_count + fluid->emitted_count - live;
        const std::size_t capacity = fluid->options.capacity;
        output.allocated_bytes += capacity *
            (6U * sizeof(Vec3) + 5U * sizeof(std::uint32_t) +
             3U * sizeof(float) + 2U * sizeof(std::uint8_t) +
             2U * sizeof(std::uint64_t) + sizeof(FluidBodyImpulse) +
             2U * sizeof(FluidContactSample)) +
             fluid->sort_workspace_size + fluid->select_workspace_size +
             3U * sizeof(std::uint32_t);
    }
    output.rigid_body_count = impl_->rigid_body_count;
    output.triangle_mesh_count = impl_->triangle_mesh_count;
    output.allocated_bytes +=
        capacity * (sizeof(BodyParameters) + sizeof(BodyAccumulator) +
                    sizeof(KinematicTarget) + sizeof(RigidBodyId) +
                    3U * sizeof(RigidBodyState) +
                    (impl_->debug_applied_forces != nullptr
                         ? 2U * sizeof(Vec3) : 0U)) +
        capacity * (capacity - 1U) / 2U * sizeof(ContactManifold) +
        capacity * sizeof(std::uint32_t) +
        capacity * (capacity - 1U) / 2U * sizeof(std::uint8_t) +
        2U * sizeof(std::uint32_t) +
        capacity * (capacity - 1U) / 2U * sizeof(std::uint32_t) +
        capacity * (2U * sizeof(WorldAabb) + sizeof(std::uint32_t)) +
        capacity * capacity *
            (sizeof(std::uint8_t) + sizeof(std::uint32_t)) +
        sizeof(std::uint32_t) + impl_->rigid_broad_phase_workspace_size +
        impl_->rigid_leaf_pair_capacity * sizeof(LeafPair) +
        capacity * capacity * sizeof(std::uint32_t) +
        impl_->rigid_contact_capacity * sizeof(RigidContactEvent) +
        impl_->options.contact_capacity * sizeof(ContactEvent) +
        k_fluid_body_buckets *
            ((capacity + 63U) / 64U) * sizeof(unsigned long long) +
        ((capacity + 63U) / 64U) * sizeof(unsigned long long) +
        4U * sizeof(std::uint32_t) +
        impl_->options.triangle_mesh_capacity * sizeof(TriangleMeshResource) +
        impl_->options.paint_field_capacity * sizeof(PaintFieldResource) +
        impl_->options.paint_rule_capacity * sizeof(PaintRuleResource);
    output.allocated_bytes +=
        impl_->debug_frames.capacity() * sizeof(PhysicsDebugFrame);
    for (const PhysicsDebugFrame &frame : impl_->debug_frames) {
        output.allocated_bytes +=
            frame.rigid_bodies.capacity() * sizeof(PhysicsDebugRigidSample) +
            frame.fluid_particles.capacity() * sizeof(PhysicsDebugFluidSample) +
            frame.cloth_vertices.capacity() * sizeof(PhysicsDebugClothSample) +
            frame.soft_body_nodes.capacity() *
                sizeof(PhysicsDebugSoftBodySample) +
            frame.rope_nodes.capacity() * sizeof(PhysicsDebugRopeSample) +
            frame.rigid_contacts.capacity() * sizeof(RigidContactEvent) +
            frame.fluid_contacts.capacity() * sizeof(ContactEvent);
    }
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index) {
        const PaintFieldResource &field = impl_->paint_fields[index];
        if (!field.alive) continue;
        output.allocated_bytes +=
            field.options.vertex_uvs.size * sizeof(Vec2) +
            static_cast<std::size_t>(field.options.width) *
                field.options.height * sizeof(std::uint32_t);
    }
    for (std::uint32_t index = 0;
         index < impl_->options.triangle_mesh_capacity; ++index) {
        if (impl_->meshes[index].alive) {
            output.allocated_bytes +=
                impl_->meshes[index].vertex_count * sizeof(Vec3) +
                impl_->meshes[index].index_count * sizeof(std::uint32_t) +
                impl_->meshes[index].bvh_node_count * sizeof(BvhNode) +
                impl_->meshes[index].bvh_leaf_count * sizeof(std::uint32_t);
        }
    }
    return success();
}

int World::device_ordinal() const noexcept {
    return impl_ ? impl_->device_ordinal : -1;
}

} // namespace parallel_mater
