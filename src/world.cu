// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime.h>

#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

namespace parallel_mater {
namespace {

constexpr float k_epsilon = 1.0e-6F;
constexpr std::uint32_t k_invalid_dense = std::numeric_limits<std::uint32_t>::max();

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

    if (a <= k_epsilon && e <= k_epsilon) {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= k_epsilon) {
        t = clamp_scalar(f / e, 0.0F, 1.0F);
    } else {
        const float c = dot(d1, r);
        if (e <= k_epsilon) {
            s = clamp_scalar(-c / a, 0.0F, 1.0F);
        } else {
            const float b = dot(d1, d2);
            const float denominator = a * e - b * b;
            if (fabsf(denominator) > k_epsilon) {
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

struct CompletionState {
    cudaEvent_t event{};
    const std::uint32_t *fluid_neighbor_overflow{};
    bool acknowledged{};
    Status completion_status{};

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
    fluid_spawn,
    fluid_neighbor_sort,
    fluid_neighbor_forces,
    fluid_integration,
    fluid_static_contacts,
    fluid_body_index,
    fluid_moving_contacts,
    fluid_contact_events,
    fluid_outflow_compaction,
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
    completion->completion_status =
        error == cudaSuccess
            ? success()
            : cuda_failure(error, "CUDA frame completion failed");
    if (error == cudaSuccess && completion->fluid_neighbor_overflow != nullptr &&
        *completion->fluid_neighbor_overflow != 0U) {
        completion->completion_status = failure(
            StatusCode::capacity_exceeded,
            "fluid neighbor count exceeded maximum_neighbors");
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

struct FluidContactSample {
    Vec3 position{};
    Vec3 normal{};
    float normal_impulse{};
    std::uint32_t body{k_invalid_dense};
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

struct SpawnPlaneSlot {
    ParticleSpawnPlaneOptions options{};
    std::uint32_t generation{1U};
    double remainder{};
    std::uint32_t sequence{};
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
    Vec3 *forces, float *foam_source, std::uint32_t *overflow) {
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

__global__ void fluid_static_contacts(
    Vec3 *positions, Vec3 *velocities, const Vec3 *previous, float *foam,
    const std::uint32_t *count, float radius, float spawn_clearance,
    std::uint32_t first_spawned, bool recover_spawn, Vec3 up,
    const BodyParameters *parameters, const RigidBodyState *states,
    const TriangleMeshResource *meshes, std::uint32_t body_index,
    float particle_mass, bool collect_contacts,
    FluidContactSample *samples, std::uint8_t *contact_flags) {
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
        position = add(position, multiply(best_normal, best_penetration));
        const float incoming = dot(velocity, best_normal);
        const float normal_impulse = incoming < 0.0F
            ? -incoming * (1.0F + body.restitution) * particle_mass : 0.0F;
        if (incoming < 0.0F) {
            velocity = subtract(velocity,
                multiply(best_normal, incoming * (1.0F + body.restitution)));
            const Vec3 tangent = subtract(velocity,
                multiply(best_normal, dot(velocity, best_normal)));
            // Remove tangential energy on impact; the previous weak response
            // left shallow puddles cycling over narrow static surfaces.
            velocity = subtract(velocity, multiply(tangent,
                fminf(1.0F, body.friction * 1.5F)));
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
    FluidContactSample *samples, std::uint8_t *particle_contact_flags) {
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

__global__ void fluid_reduce_body_impulses(
    const FluidBodyImpulse *impulses, const std::uint32_t *particle_count,
    const BodyParameters *parameters, RigidBodyState *states,
    const std::uint32_t *contact_flags, std::uint32_t body_count) {
    const std::uint32_t body = blockIdx.x;
    if (body >= body_count || contact_flags[body] == 0U ||
        parameters[body].motion != MotionType::dynamic)
        return;
    __shared__ Vec3 linear[128];
    __shared__ Vec3 angular[128];
    Vec3 local_linear{}, local_angular{};
    for (std::uint32_t particle = threadIdx.x; particle < *particle_count;
         particle += blockDim.x) {
        const FluidBodyImpulse impulse = impulses[particle];
        if (impulse.body != body) continue;
        local_linear = add(local_linear, impulse.linear);
        local_angular = add(local_angular, impulse.angular);
    }
    linear[threadIdx.x] = local_linear;
    angular[threadIdx.x] = local_angular;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
        if (threadIdx.x < stride) {
            linear[threadIdx.x] = add(linear[threadIdx.x],
                                     linear[threadIdx.x + stride]);
            angular[threadIdx.x] = add(angular[threadIdx.x],
                                       angular[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        RigidBodyState state = states[body];
        const BodyParameters options = parameters[body];
        state.linear_velocity = clamp_length(add(state.linear_velocity,
            multiply(linear[0], options.inverse_mass)),
            options.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(state.angular_velocity,
            inverse_inertia_world(options, state, angular[0])),
            options.maximum_angular_speed);
        states[body] = state;
    }
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

__device__ float fluid_radical_inverse(std::uint32_t index,
                                      std::uint32_t base) noexcept {
    float result = 0.0F, factor = 1.0F / static_cast<float>(base);
    while (index != 0U) {
        result += factor * static_cast<float>(index % base);
        index /= base;
        factor /= static_cast<float>(base);
    }
    return result;
}

__global__ void fluid_spawn(Vec3 *positions, Vec3 *velocities,
                            std::uint32_t *ids, float *foam,
                            std::uint32_t first, std::uint32_t amount,
                            std::uint32_t first_id, std::uint32_t sequence,
                            ParticleSpawnPlaneOptions options) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= amount) return;
    const std::uint32_t sample = sequence + item + options.sequence_seed + 1U;
    const Vec3 local{
        (2.0F * fluid_radical_inverse(sample, 2U) - 1.0F) *
            options.plane.half_extents.x,
        0.0F,
        (2.0F * fluid_radical_inverse(sample, 3U) - 1.0F) *
            options.plane.half_extents.y};
    positions[first + item] = add(options.plane.center,
        rotate(options.plane.orientation, local));
    velocities[first + item] = options.initial_velocity;
    ids[first + item] = first_id + item;
    foam[first + item] = 0.0F;
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
    std::vector<Slot> slots{};
    std::vector<std::unique_ptr<FluidStorage>> fluids{};
    std::vector<SpawnPlaneSlot> spawn_planes{};
    std::vector<DestroyPlaneSlot> destroy_planes{};
    std::uint32_t *fluid_neighbor_overflow{};
    BodyParameters *parameters{};
    BodyAccumulator *accumulators{};
    KinematicTarget *targets{};
    RigidBodyId *ids{};
    RigidBodyState *states[2]{};
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
    std::size_t timing_boundary_count{};
    std::uint64_t timing_frame_index{};
    bool timing_available{};

    ~Impl() {
        if (frame && !frame->acknowledged) {
            (void)wait_for_completion(frame);
        }
        if (meshes != nullptr) {
            for (std::uint32_t index = 0;
                 index < options.triangle_mesh_capacity; ++index) {
                release_managed(meshes[index].bvh_nodes);
                release_managed(meshes[index].bvh_leaves);
                release_managed(meshes[index].indices);
                release_managed(meshes[index].vertices);
            }
        }
        for (cudaEvent_t event : timing_events) {
            cudaEventDestroy(event);
        }
        release_managed(meshes);
        release_managed(fluid_neighbor_overflow);
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
        implementation->spawn_planes.resize(options.particle_spawn_plane_capacity);
        implementation->destroy_planes.resize(options.particle_destroy_plane_capacity);
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate world host storage");
    }
    implementation->options = options;
    implementation->device_ordinal = device;
    Status status = allocate_managed(implementation->fluid_neighbor_overflow, 1U);
    if (!status) return status;
    *implementation->fluid_neighbor_overflow = 0U;
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

Status World::remove_fluid(FluidId id, cudaStream_t) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(id, fluid))) return status;
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
    for (SpawnPlaneSlot &plane : impl_->spawn_planes)
        if (plane.alive && plane.options.fluid == id) {
            plane.alive = false; ++plane.generation;
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
              {fluid->ids, count}, {fluid->foam, count}, count,
              fluid->options.particle_radius,
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

Status World::add_particle_spawn_plane(ParticleSpawnPlaneOptions options,
                                       ParticleSpawnPlaneId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane) ||
        !finite(options.particles_per_second) ||
        options.particles_per_second < 0.0F ||
        !finite(options.initial_velocity))
        return failure(StatusCode::invalid_argument, "spawn plane options are invalid");
    for (std::uint32_t index = 0; index < impl_->spawn_planes.size(); ++index) {
        SpawnPlaneSlot &plane = impl_->spawn_planes[index];
        if (plane.alive) continue;
        plane.options = options;
        plane.remainder = 0.0;
        plane.sequence = 0U;
        plane.alive = true;
        output = {index, plane.generation};
        return success();
    }
    return failure(StatusCode::capacity_exceeded, "spawn plane capacity exhausted");
}

Status World::update_particle_spawn_plane(ParticleSpawnPlaneId id,
                                          ParticleSpawnPlaneOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->spawn_planes.size() ||
        !impl_->spawn_planes[id.index].alive ||
        impl_->spawn_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "spawn plane handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane) ||
        !finite(options.particles_per_second) ||
        options.particles_per_second < 0.0F ||
        !finite(options.initial_velocity))
        return failure(StatusCode::invalid_argument, "spawn plane options are invalid");
    impl_->spawn_planes[id.index].options = options;
    return success();
}

Status World::remove_particle_spawn_plane(ParticleSpawnPlaneId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->spawn_planes.size() ||
        !impl_->spawn_planes[id.index].alive ||
        impl_->spawn_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "spawn plane handle is stale");
    impl_->spawn_planes[id.index].alive = false;
    ++impl_->spawn_planes[id.index].generation;
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
    try {
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
    TriangleMeshResource &mesh = impl_->meshes[mesh_id.index];
    release_managed(mesh.bvh_leaves);
    release_managed(mesh.bvh_nodes);
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
    if (impl_->rigid_body_count == 0U) {
        *impl_->rigid_contact_count = 0U;
    }
    *impl_->fluid_neighbor_overflow = 0U;
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
            static_cast<std::size_t>(options.substeps) * 7U + 1U;
        for (const auto &fluid : impl_->fluids) {
            if (fluid && fluid->alive) {
                maximum_stages += 2U +
                    static_cast<std::size_t>(options.substeps) *
                        fluid->options.solver_iterations * 7U;
            }
        }
        status = impl_->prepare_timing_events(
            maximum_stages + 2U);
        if (!status) {
            return status;
        }
        error = cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to begin kernel timing");
        }
    }

    const auto record_timing_stage = [&](TimingStage stage) noexcept -> Status {
        if (!options.collect_kernel_timings) {
            return success();
        }
        impl_->timing_stages.push_back(stage);
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
            options.collect_rigid_contacts, substep == 0U);
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
                    options.collect_rigid_contacts
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
                options.collect_rigid_contacts
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
        for (SpawnPlaneSlot &slot : impl_->spawn_planes) {
            if (!slot.alive || !slot.options.enabled ||
                !(slot.options.fluid == fluid_id)) continue;
            const double request = slot.remainder +
                static_cast<double>(slot.options.particles_per_second) *
                options.timestep;
            const std::uint64_t wanted = static_cast<std::uint64_t>(floor(request));
            slot.remainder = request - static_cast<double>(wanted);
            const std::uint32_t free = fluid.options.capacity - live;
            const std::uint32_t amount = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(wanted, free));
            impl_->spawn_capacity_miss_count += wanted - amount;
            if (amount != 0U) {
                spawned = true;
                if (fluid.next_id > UINT32_MAX - amount)
                    return failure(StatusCode::capacity_exceeded,
                                   "stable fluid particle ID range exhausted");
                fluid_spawn<<<(amount + block_size - 1U) / block_size,
                              block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.ids,
                    fluid.foam, live, amount, fluid.next_id,
                    slot.sequence, slot.options);
                live += amount;
                fluid.next_id += amount;
                fluid.emitted_count += amount;
                impl_->emitted_particle_count += amount;
            }
            slot.sequence += static_cast<std::uint32_t>(wanted);
        }
        *fluid.count = live;
        if (spawned) {
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
        if (options.collect_fluid_contacts) {
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
                impl_->fluid_neighbor_overflow);
            status = record_timing_stage(TimingStage::fluid_neighbor_forces);
            if (!status) return status;
            fluid_integrate<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.velocities, fluid.previous,
                fluid.foam, fluid.forces, fluid.foam_source,
                fluid.count, fluid.options,
                options.gravity, dt);
            status = record_timing_stage(TimingStage::fluid_integration);
            if (!status) return status;
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
                    options.collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags);
                fluid_reduce_body_impulses<<<impl_->rigid_body_count,
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
                    options.collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags);
            }
            if (any_static_body) {
                status = record_timing_stage(TimingStage::fluid_static_contacts);
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
                    options.collect_fluid_contacts,
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
        if (options.collect_fluid_contacts) {
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
            impl_->timing_stages[index] == TimingStage::rigid_contact_solve
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
    output.emitted_particle_count = impl_->emitted_particle_count;
    output.destroyed_particle_count = impl_->destroyed_particle_count;
    output.spawn_capacity_miss_count = impl_->spawn_capacity_miss_count;
    output.contact_count = *impl_->fluid_contact_count;
    output.contact_overflow_count = *impl_->fluid_contact_overflow;
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
                    3U * sizeof(RigidBodyState)) +
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
        3U * sizeof(std::uint32_t) +
        impl_->options.triangle_mesh_capacity * sizeof(TriangleMeshResource);
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
