// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
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

struct TriangleMeshResource {
    Vec3 *vertices{};
    std::uint32_t *indices{};
    std::uint32_t vertex_count{};
    std::uint32_t index_count{};
    std::uint32_t generation{};
    bool alive{};
    Vec3 minimum{};
    Vec3 maximum{};
    Vec3 unit_inertia{};
    BvhNode *bvh_nodes{};
    std::uint32_t bvh_node_count{};
};

__host__ __device__ Vec3 closest_on_segment(Vec3 point, Vec3 a, Vec3 b) noexcept {
    const Vec3 segment = subtract(b, a);
    const float denominator = length_squared(segment);
    if (denominator <= k_epsilon * k_epsilon) {
        return a;
    }
    const float t = clamp_scalar(dot(subtract(point, a), segment) / denominator,
                                 0.0F, 1.0F);
    return add(a, multiply(segment, t));
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

__host__ __device__ void closest_segment_triangle(
    Vec3 segment_a, Vec3 segment_b, Vec3 a, Vec3 b, Vec3 c,
    Vec3 &segment_point, Vec3 &triangle_point) noexcept {
    float best_squared = FLT_MAX;
    consider_closest_pair(segment_a, closest_on_triangle(segment_a, a, b, c),
                          best_squared, segment_point, triangle_point);
    consider_closest_pair(segment_b, closest_on_triangle(segment_b, a, b, c),
                          best_squared, segment_point, triangle_point);
    consider_closest_pair(closest_on_segment(a, segment_a, segment_b), a,
                          best_squared, segment_point, triangle_point);
    consider_closest_pair(closest_on_segment(b, segment_a, segment_b), b,
                          best_squared, segment_point, triangle_point);
    consider_closest_pair(closest_on_segment(c, segment_a, segment_b), c,
                          best_squared, segment_point, triangle_point);
    Vec3 first{};
    Vec3 second{};
    closest_segments(segment_a, segment_b, a, b, first, second);
    consider_closest_pair(first, second, best_squared, segment_point,
                          triangle_point);
    closest_segments(segment_a, segment_b, b, c, first, second);
    consider_closest_pair(first, second, best_squared, segment_point,
                          triangle_point);
    closest_segments(segment_a, segment_b, c, a, first, second);
    consider_closest_pair(first, second, best_squared, segment_point,
                          triangle_point);

    const Vec3 normal = cross(subtract(b, a), subtract(c, a));
    const float normal_squared = length_squared(normal);
    const Vec3 direction = subtract(segment_b, segment_a);
    const float denominator = dot(normal, direction);
    if (normal_squared > k_epsilon * k_epsilon &&
        fabsf(denominator) > k_epsilon) {
        const float amount = dot(normal, subtract(a, segment_a)) / denominator;
        if (amount >= 0.0F && amount <= 1.0F) {
            const Vec3 intersection = add(segment_a, multiply(direction, amount));
            if (point_in_triangle(intersection, a, b, c, normal)) {
                segment_point = intersection;
                triangle_point = intersection;
            }
        }
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

__host__ __device__ void closest_triangle_pair(
    Vec3 a0, Vec3 a1, Vec3 a2, Vec3 b0, Vec3 b1, Vec3 b2,
    Vec3 &point_a, Vec3 &point_b) noexcept {
    float best_squared = FLT_MAX;
    Vec3 on_edge{};
    Vec3 on_triangle{};
    closest_segment_triangle(a0, a1, b0, b1, b2, on_edge, on_triangle);
    consider_closest_pair(on_edge, on_triangle, best_squared, point_a, point_b);
    closest_segment_triangle(a1, a2, b0, b1, b2, on_edge, on_triangle);
    consider_closest_pair(on_edge, on_triangle, best_squared, point_a, point_b);
    closest_segment_triangle(a2, a0, b0, b1, b2, on_edge, on_triangle);
    consider_closest_pair(on_edge, on_triangle, best_squared, point_a, point_b);

    closest_segment_triangle(b0, b1, a0, a1, a2, on_edge, on_triangle);
    consider_closest_pair(on_triangle, on_edge, best_squared, point_a, point_b);
    closest_segment_triangle(b1, b2, a0, a1, a2, on_edge, on_triangle);
    consider_closest_pair(on_triangle, on_edge, best_squared, point_a, point_b);
    closest_segment_triangle(b2, b0, a0, a1, a2, on_edge, on_triangle);
    consider_closest_pair(on_triangle, on_edge, best_squared, point_a, point_b);
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

__device__ ContactManifold collide_meshes(
    const BodyParameters &body, const RigidBodyState &body_state,
    const TriangleMeshResource &body_mesh, const BodyParameters &collider,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh) noexcept {
    ContactManifold manifold{};
    const float margin = body.collision_margin + collider.collision_margin;
    Vec3 body_minimum{};
    Vec3 body_maximum{};
    Vec3 collider_minimum{};
    Vec3 collider_maximum{};
    const BoundsTransform body_transform = bounds_transform(body_state);
    const BoundsTransform collider_transform = bounds_transform(collider_state);
    transformed_bounds(body_mesh.minimum, body_mesh.maximum, body_transform,
                       margin, body_minimum, body_maximum);
    transformed_bounds(collider_mesh.minimum, collider_mesh.maximum,
                       collider_transform, 0.0F, collider_minimum,
                       collider_maximum);
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
        transformed_bounds(body_node.minimum, body_node.maximum, body_transform,
                           margin, body_minimum, body_maximum);
        transformed_bounds(collider_node.minimum, collider_node.maximum,
                           collider_transform, 0.0F, collider_minimum,
                           collider_maximum);
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
        collide_triangle_ranges(
            body_state, body_mesh, 0U, body_mesh.index_count / 3U,
            collider_state, collider_mesh, 0U,
            collider_mesh.index_count / 3U, margin, manifold);
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
    bool correct_position, RigidContactEvent *debug_events) noexcept {
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
        if (debug_events != nullptr) {
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

__global__ void generate_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, const TriangleMeshResource *meshes,
    std::uint32_t mesh_capacity, ContactManifold *manifolds,
    std::uint32_t manifold_stride) {
    const std::uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= count * count) {
        return;
    }
    const std::uint32_t index = pair / count;
    const std::uint32_t collider_index = pair % count;
    ContactManifold &manifold =
        manifolds[index * manifold_stride + collider_index];
    manifold = {};
    if (parameters[index].motion != MotionType::dynamic ||
        collider_index == index ||
        (parameters[collider_index].motion == MotionType::dynamic &&
         collider_index < index)) {
        return;
    }
    const TriangleMeshId body_mesh_id = parameters[index].mesh;
    const TriangleMeshId collider_mesh_id = parameters[collider_index].mesh;
    if (body_mesh_id.index >= mesh_capacity ||
        collider_mesh_id.index >= mesh_capacity) {
        return;
    }
    const TriangleMeshResource &body_mesh = meshes[body_mesh_id.index];
    const TriangleMeshResource &collider_mesh = meshes[collider_mesh_id.index];
    if (!body_mesh.alive || body_mesh.generation != body_mesh_id.generation ||
        !collider_mesh.alive ||
        collider_mesh.generation != collider_mesh_id.generation) {
        return;
    }
    manifold = collide_meshes(parameters[index], states[index], body_mesh,
                              parameters[collider_index],
                              states[collider_index], collider_mesh);
}

__global__ void resolve_cached_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    const RigidBodyId *ids, std::uint32_t count,
    const ContactManifold *manifolds, std::uint32_t manifold_stride,
    RigidContactEvent *debug_events, std::uint32_t debug_capacity,
    std::uint32_t *debug_count, bool reset_debug) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    if (reset_debug) {
        *debug_count = 0U;
    }
    for (int pass = 0; pass < 8; ++pass) {
        std::uint32_t event_cursor = 0U;
        for (std::uint32_t index = 0; index < count; ++index) {
            if (parameters[index].motion != MotionType::dynamic) {
                continue;
            }
            for (std::uint32_t collider_index = 0; collider_index < count;
                 ++collider_index) {
                if (collider_index == index ||
                    (parameters[collider_index].motion == MotionType::dynamic &&
                     collider_index < index)) {
                    continue;
                }
                const ContactManifold &manifold =
                    manifolds[index * manifold_stride + collider_index];
                if (manifold.count > 0U) {
                    RigidContactEvent *events = nullptr;
                    if (event_cursor <= debug_capacity &&
                        manifold.count <= debug_capacity - event_cursor) {
                        events = debug_events + event_cursor;
                        if (pass == 0) {
                            for (std::uint32_t contact_index = 0;
                                 contact_index < manifold.count;
                                 ++contact_index) {
                                const Contact &contact =
                                    manifold.contacts[contact_index];
                                events[contact_index] = {
                                    ids[index], ids[collider_index],
                                    contact.point, contact.normal,
                                    contact.penetration, 0.0F, {}};
                            }
                        }
                    }
                    resolve_contacts(parameters[index], states[index],
                                     parameters[collider_index],
                                     states[collider_index], manifold.contacts,
                                     manifold.count, pass == 0, events);
                    event_cursor += manifold.count;
                }
            }
        }
        if (pass == 0 && event_cursor > 0U) {
            *debug_count = event_cursor < debug_capacity ? event_cursor
                                                         : debug_capacity;
        }
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        if (parameters[index].motion != MotionType::dynamic) {
            continue;
        }
        states[index].linear_velocity = clamp_length(
            states[index].linear_velocity, parameters[index].maximum_linear_speed);
        states[index].angular_velocity = clamp_length(
            states[index].angular_velocity, parameters[index].maximum_angular_speed);
    }
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
    rigid_contact_generation,
    rigid_contact_solve,
    rigid_input_clear,
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
    std::uint32_t current_state{};
    std::uint64_t frame_index{};
    std::uint64_t revision{};
    std::vector<Slot> slots{};
    BodyParameters *parameters{};
    BodyAccumulator *accumulators{};
    KinematicTarget *targets{};
    RigidBodyId *ids{};
    RigidBodyState *states[2]{};
    ContactManifold *rigid_manifolds{};
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
                release_managed(meshes[index].indices);
                release_managed(meshes[index].vertices);
            }
        }
        for (cudaEvent_t event : timing_events) {
            cudaEventDestroy(event);
        }
        release_managed(meshes);
        release_managed(rigid_contact_count);
        release_managed(rigid_contact_events);
        release_managed(rigid_manifolds);
        release_managed(states[1]);
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
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate world host storage");
    }
    implementation->options = options;
    implementation->device_ordinal = device;
    const std::size_t manifold_count =
        static_cast<std::size_t>(options.rigid_body_capacity) *
        options.rigid_body_capacity;
    if (manifold_count >
        std::numeric_limits<std::size_t>::max() / sizeof(ContactManifold)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds contact cache range");
    }
    constexpr std::size_t contacts_per_manifold = 8U;
    if (manifold_count >
        std::numeric_limits<std::uint32_t>::max() / contacts_per_manifold) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds contact event range");
    }
    implementation->rigid_contact_capacity = static_cast<std::uint32_t>(
        manifold_count * contacts_per_manifold);

    Status status = allocate_managed(implementation->parameters,
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
    status = allocate_managed(implementation->rigid_manifolds, manifold_count);
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

Status World::add_fluid(FluidOptions, DeviceSpan<const FluidParticle>, FluidId &,
                        cudaStream_t) noexcept {
    return failure(StatusCode::not_supported,
                   "fluid implementation is scheduled for PR 5");
}

Status World::remove_fluid(FluidId, cudaStream_t) noexcept {
    return failure(StatusCode::not_supported,
                   "fluid implementation is scheduled for PR 5");
}

Status World::fluid_view(FluidId, FluidDeviceView &) const noexcept {
    return failure(StatusCode::not_supported,
                   "fluid implementation is scheduled for PR 5");
}

Status World::add_particle_spawn_plane(ParticleSpawnPlaneOptions,
                                       ParticleSpawnPlaneId &) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 5");
}

Status World::update_particle_spawn_plane(ParticleSpawnPlaneId,
                                          ParticleSpawnPlaneOptions) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 5");
}

Status World::remove_particle_spawn_plane(ParticleSpawnPlaneId) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 5");
}

Status World::add_particle_destroy_plane(ParticleDestroyPlaneOptions,
                                         ParticleDestroyPlaneId &) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 5");
}

Status World::update_particle_destroy_plane(ParticleDestroyPlaneId,
                                            ParticleDestroyPlaneOptions) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 5");
}

Status World::remove_particle_destroy_plane(ParticleDestroyPlaneId) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 5");
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

    BvhNode *owned_bvh_nodes = nullptr;
    status = allocate_managed(owned_bvh_nodes, bvh_nodes.size());
    if (!status) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    std::copy(reordered_indices.begin(), reordered_indices.end(), owned_indices);
    std::copy(bvh_nodes.begin(), bvh_nodes.end(), owned_bvh_nodes);

    TriangleMeshResource &mesh = impl_->meshes[slot];
    mesh.vertices = owned_vertices;
    mesh.indices = owned_indices;
    mesh.vertex_count = static_cast<std::uint32_t>(vertices.size);
    mesh.index_count = static_cast<std::uint32_t>(triangle_indices.size);
    mesh.minimum = minimum;
    mesh.maximum = maximum;
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
    release_managed(mesh.bvh_nodes);
    release_managed(mesh.indices);
    release_managed(mesh.vertices);
    mesh.vertex_count = 0U;
    mesh.index_count = 0U;
    mesh.bvh_node_count = 0U;
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

    Impl::Slot &slot = impl_->slots[slot_index];
    const std::uint32_t dense = impl_->rigid_body_count;
    slot.alive = true;
    slot.dense_index = dense;
    if (slot.generation == 0U) {
        slot.generation = 1U;
    }
    const RigidBodyId id{slot_index, slot.generation};
    RigidBodyState normalized_state = options.initial_state;
    normalized_state.orientation =
        normalized_quaternion(normalized_state.orientation);
    impl_->parameters[dense] =
        make_parameters(options, impl_->meshes[options.mesh.index]);
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
    cudaError_t error =
        cudaEventCreateWithFlags(&frame->event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to create frame completion event");
    }

    impl_->timing_available = false;
    impl_->timing_boundary_count = 0U;
    std::size_t timing_boundary = 0U;
    if (options.collect_kernel_timings) {
        status = impl_->prepare_timing_events(
            static_cast<std::size_t>(options.substeps) * 3U + 2U);
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
    const float substep_timestep =
        options.timestep / static_cast<float>(options.substeps);
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
        generate_rigid_contacts_kernel<<<pair_block_count, block_size, 0,
                                         stream>>>(
            impl_->parameters, impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes,
            impl_->options.triangle_mesh_capacity, impl_->rigid_manifolds,
            impl_->options.rigid_body_capacity);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid contact generation kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_contact_generation);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        resolve_cached_rigid_contacts_kernel<<<1U, 1U, 0, stream>>>(
            impl_->parameters, impl_->states[output_state],
            impl_->ids, impl_->rigid_body_count, impl_->rigid_manifolds,
            impl_->options.rigid_body_capacity, impl_->rigid_contact_events,
            options.collect_rigid_contacts ? impl_->rigid_contact_capacity : 0U,
            impl_->rigid_contact_count, substep == 0U);
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
    } else if (options.collect_kernel_timings) {
        error = cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to finish empty kernel timing");
        }
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
    if (!impl_) {
        return {};
    }
    return {{}, 0U, false, impl_->frame_index};
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
        switch (impl_->timing_stages[index]) {
        case TimingStage::rigid_integration:
            timing = &output.rigid_integration;
            break;
        case TimingStage::rigid_contact_generation:
            timing = &output.rigid_contact_generation;
            break;
        case TimingStage::rigid_contact_solve:
            timing = &output.rigid_contact_solve;
            break;
        case TimingStage::rigid_input_clear:
            timing = &output.rigid_input_clear;
            break;
        }
        timing->total_milliseconds += milliseconds;
        ++timing->launch_count;
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
    output.rigid_body_count = impl_->rigid_body_count;
    output.triangle_mesh_count = impl_->triangle_mesh_count;
    output.allocated_bytes =
        capacity * (sizeof(BodyParameters) + sizeof(BodyAccumulator) +
                    sizeof(KinematicTarget) + sizeof(RigidBodyId) +
                    2U * sizeof(RigidBodyState)) +
        capacity * capacity * sizeof(ContactManifold) +
        impl_->rigid_contact_capacity * sizeof(RigidContactEvent) +
        sizeof(std::uint32_t) +
        impl_->options.triangle_mesh_capacity * sizeof(TriangleMeshResource);
    for (std::uint32_t index = 0;
         index < impl_->options.triangle_mesh_capacity; ++index) {
        if (impl_->meshes[index].alive) {
            output.allocated_bytes +=
                impl_->meshes[index].vertex_count * sizeof(Vec3) +
                impl_->meshes[index].index_count * sizeof(std::uint32_t) +
                impl_->meshes[index].bvh_node_count * sizeof(BvhNode);
        }
    }
    return success();
}

int World::device_ordinal() const noexcept {
    return impl_ ? impl_->device_ordinal : -1;
}

} // namespace parallel_mater
