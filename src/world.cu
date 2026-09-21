// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
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

__host__ __device__ Vec3 clamp_components(Vec3 value, Vec3 minimum,
                                          Vec3 maximum) noexcept {
    return {clamp_scalar(value.x, minimum.x, maximum.x),
            clamp_scalar(value.y, minimum.y, maximum.y),
            clamp_scalar(value.z, minimum.z, maximum.z)};
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
    CollisionShape shape{};
    float inverse_mass{};
    Vec3 inverse_inertia_local{};
    float friction{};
    float restitution{};
    float linear_damping{};
    float angular_damping{};
    float maximum_linear_speed{};
    float maximum_angular_speed{};
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

__host__ __device__ Vec3 local_axis(Quaternion orientation,
                                   std::uint32_t axis) noexcept {
    if (axis == 0U) {
        return rotate(orientation, {1.0F, 0.0F, 0.0F});
    }
    if (axis == 1U) {
        return rotate(orientation, {0.0F, 1.0F, 0.0F});
    }
    return rotate(orientation, {0.0F, 0.0F, 1.0F});
}

__host__ __device__ float shape_support(const CollisionShape &shape,
                                        Quaternion orientation,
                                        Vec3 direction) noexcept {
    const Vec3 local = inverse_rotate(orientation, direction);
    switch (shape.type) {
    case ShapeType::sphere:
        return shape.dimensions.x;
    case ShapeType::box:
        return fabsf(local.x) * shape.dimensions.x +
               fabsf(local.y) * shape.dimensions.y +
               fabsf(local.z) * shape.dimensions.z;
    case ShapeType::capsule:
        return shape.dimensions.x + fabsf(local.y) * shape.dimensions.y;
    case ShapeType::plane:
        return 0.0F;
    }
    return 0.0F;
}

__host__ __device__ void capsule_segment(const BodyParameters &parameters,
                                         const RigidBodyState &state, Vec3 &a,
                                         Vec3 &b) noexcept {
    const Vec3 offset = multiply(local_axis(state.orientation, 1U),
                                 parameters.shape.dimensions.y);
    a = subtract(state.position, offset);
    b = add(state.position, offset);
}

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

__host__ __device__ Contact sphere_box_contact(
    Vec3 sphere_center, float sphere_radius, const BodyParameters &box,
    const RigidBodyState &box_state) noexcept {
    const Vec3 local_center =
        inverse_rotate(box_state.orientation, subtract(sphere_center, box_state.position));
    const Vec3 half = box.shape.dimensions;
    const Vec3 closest = clamp_components(local_center, multiply(half, -1.0F), half);
    const Vec3 local_delta = subtract(local_center, closest);
    const float squared = length_squared(local_delta);

    Vec3 local_normal{};
    float penetration = 0.0F;
    Vec3 local_point = closest;
    if (squared > k_epsilon * k_epsilon) {
        const float distance = sqrtf(squared);
        penetration = sphere_radius - distance;
        if (penetration <= 0.0F) {
            return {};
        }
        local_normal = multiply(local_delta, 1.0F / distance);
    } else {
        const float dx = half.x - fabsf(local_center.x);
        const float dy = half.y - fabsf(local_center.y);
        const float dz = half.z - fabsf(local_center.z);
        if (dx <= dy && dx <= dz) {
            local_normal = {local_center.x >= 0.0F ? 1.0F : -1.0F, 0.0F, 0.0F};
            local_point.x = local_normal.x * half.x;
            penetration = sphere_radius + dx;
        } else if (dy <= dz) {
            local_normal = {0.0F, local_center.y >= 0.0F ? 1.0F : -1.0F, 0.0F};
            local_point.y = local_normal.y * half.y;
            penetration = sphere_radius + dy;
        } else {
            local_normal = {0.0F, 0.0F, local_center.z >= 0.0F ? 1.0F : -1.0F};
            local_point.z = local_normal.z * half.z;
            penetration = sphere_radius + dz;
        }
    }
    return {rotate(box_state.orientation, local_normal),
            add(box_state.position, rotate(box_state.orientation, local_point)),
            penetration, true};
}

__host__ __device__ float point_box_squared(Vec3 point,
                                            const BodyParameters &box,
                                            const RigidBodyState &state) noexcept {
    const Vec3 local =
        inverse_rotate(state.orientation, subtract(point, state.position));
    const Vec3 closest = clamp_components(local, multiply(box.shape.dimensions, -1.0F),
                                          box.shape.dimensions);
    return length_squared(subtract(local, closest));
}

__host__ __device__ Contact capsule_box_contact(
    const BodyParameters &capsule, const RigidBodyState &capsule_state,
    const BodyParameters &box, const RigidBodyState &box_state) noexcept {
    Vec3 a{};
    Vec3 b{};
    capsule_segment(capsule, capsule_state, a, b);
    float low = 0.0F;
    float high = 1.0F;
    for (int iteration = 0; iteration < 20; ++iteration) {
        const float left = (2.0F * low + high) / 3.0F;
        const float right = (low + 2.0F * high) / 3.0F;
        const Vec3 left_point = add(a, multiply(subtract(b, a), left));
        const Vec3 right_point = add(a, multiply(subtract(b, a), right));
        if (point_box_squared(left_point, box, box_state) <
            point_box_squared(right_point, box, box_state)) {
            high = right;
        } else {
            low = left;
        }
    }
    const Vec3 centerline = add(a, multiply(subtract(b, a), 0.5F * (low + high)));
    return sphere_box_contact(centerline, capsule.shape.dimensions.x, box, box_state);
}

__host__ __device__ Contact box_box_contact(
    const BodyParameters &moving, const RigidBodyState &moving_state,
    const BodyParameters &collider, const RigidBodyState &collider_state) noexcept {
    Vec3 axes[15]{};
    axes[0] = local_axis(moving_state.orientation, 0U);
    axes[1] = local_axis(moving_state.orientation, 1U);
    axes[2] = local_axis(moving_state.orientation, 2U);
    axes[3] = local_axis(collider_state.orientation, 0U);
    axes[4] = local_axis(collider_state.orientation, 1U);
    axes[5] = local_axis(collider_state.orientation, 2U);
    int axis_count = 6;
    for (int first = 0; first < 3; ++first) {
        for (int second = 3; second < 6; ++second) {
            axes[axis_count++] = cross(axes[first], axes[second]);
        }
    }

    const Vec3 center_delta = subtract(moving_state.position, collider_state.position);
    float least_overlap = FLT_MAX;
    Vec3 best_normal{0.0F, 1.0F, 0.0F};
    for (int axis_index = 0; axis_index < axis_count; ++axis_index) {
        const float squared = length_squared(axes[axis_index]);
        if (squared <= k_epsilon * k_epsilon) {
            continue;
        }
        const Vec3 axis = multiply(axes[axis_index], rsqrtf(squared));
        const float radius_a = shape_support(moving.shape, moving_state.orientation, axis);
        const float radius_b = shape_support(collider.shape, collider_state.orientation, axis);
        const float signed_distance = dot(center_delta, axis);
        const float overlap = radius_a + radius_b - fabsf(signed_distance);
        if (overlap <= 0.0F) {
            return {};
        }
        if (overlap < least_overlap) {
            least_overlap = overlap;
            best_normal = signed_distance >= 0.0F ? axis : multiply(axis, -1.0F);
        }
    }
    const float support = shape_support(moving.shape, moving_state.orientation,
                                        best_normal);
    return {best_normal,
            subtract(moving_state.position, multiply(best_normal, support)),
            least_overlap, true};
}

__host__ __device__ Contact collide_shapes(
    const BodyParameters &moving, const RigidBodyState &moving_state,
    const BodyParameters &collider, const RigidBodyState &collider_state) noexcept {
    if (collider.shape.type == ShapeType::plane) {
        const Vec3 normal = local_axis(collider_state.orientation, 1U);
        const float center_distance =
            dot(subtract(moving_state.position, collider_state.position), normal);
        const float radius =
            shape_support(moving.shape, moving_state.orientation, normal);
        const float penetration = radius - center_distance;
        if (penetration <= 0.0F) {
            return {};
        }
        return {normal,
                subtract(moving_state.position, multiply(normal, radius)),
                penetration, true};
    }

    if (moving.shape.type == ShapeType::sphere) {
        const float moving_radius = moving.shape.dimensions.x;
        if (collider.shape.type == ShapeType::sphere) {
            const Vec3 delta = subtract(moving_state.position, collider_state.position);
            const float distance = vector_length(delta);
            const float radius_sum = moving_radius + collider.shape.dimensions.x;
            if (distance >= radius_sum) {
                return {};
            }
            const Vec3 normal = normalized_or(delta, {1.0F, 0.0F, 0.0F});
            return {normal,
                    subtract(moving_state.position, multiply(normal, moving_radius)),
                    radius_sum - distance, true};
        }
        if (collider.shape.type == ShapeType::box) {
            return sphere_box_contact(moving_state.position, moving_radius, collider,
                                      collider_state);
        }
        if (collider.shape.type == ShapeType::capsule) {
            Vec3 a{};
            Vec3 b{};
            capsule_segment(collider, collider_state, a, b);
            const Vec3 closest = closest_on_segment(moving_state.position, a, b);
            const Vec3 delta = subtract(moving_state.position, closest);
            const float distance = vector_length(delta);
            const float radius_sum = moving_radius + collider.shape.dimensions.x;
            if (distance >= radius_sum) {
                return {};
            }
            const Vec3 normal = normalized_or(
                delta, normalized_or(subtract(moving_state.position,
                                              collider_state.position),
                                     {1.0F, 0.0F, 0.0F}));
            return {normal,
                    subtract(moving_state.position, multiply(normal, moving_radius)),
                    radius_sum - distance, true};
        }
    }

    if (moving.shape.type == ShapeType::capsule) {
        if (collider.shape.type == ShapeType::sphere) {
            Vec3 a{};
            Vec3 b{};
            capsule_segment(moving, moving_state, a, b);
            const Vec3 closest = closest_on_segment(collider_state.position, a, b);
            const Vec3 delta = subtract(closest, collider_state.position);
            const float distance = vector_length(delta);
            const float radius_sum = moving.shape.dimensions.x +
                                     collider.shape.dimensions.x;
            if (distance >= radius_sum) {
                return {};
            }
            const Vec3 normal = normalized_or(
                delta, normalized_or(subtract(moving_state.position,
                                              collider_state.position),
                                     {1.0F, 0.0F, 0.0F}));
            return {normal,
                    subtract(closest,
                             multiply(normal, moving.shape.dimensions.x)),
                    radius_sum - distance, true};
        }
        if (collider.shape.type == ShapeType::capsule) {
            Vec3 moving_a{};
            Vec3 moving_b{};
            Vec3 collider_a{};
            Vec3 collider_b{};
            capsule_segment(moving, moving_state, moving_a, moving_b);
            capsule_segment(collider, collider_state, collider_a, collider_b);
            Vec3 moving_point{};
            Vec3 collider_point{};
            closest_segments(moving_a, moving_b, collider_a, collider_b,
                             moving_point, collider_point);
            const Vec3 delta = subtract(moving_point, collider_point);
            const float distance = vector_length(delta);
            const float radius_sum = moving.shape.dimensions.x +
                                     collider.shape.dimensions.x;
            if (distance >= radius_sum) {
                return {};
            }
            const Vec3 normal = normalized_or(
                delta, normalized_or(subtract(moving_state.position,
                                              collider_state.position),
                                     {1.0F, 0.0F, 0.0F}));
            return {normal,
                    subtract(moving_point,
                             multiply(normal, moving.shape.dimensions.x)),
                    radius_sum - distance, true};
        }
        if (collider.shape.type == ShapeType::box) {
            return capsule_box_contact(moving, moving_state, collider,
                                       collider_state);
        }
    }

    if (moving.shape.type == ShapeType::box) {
        if (collider.shape.type == ShapeType::box) {
            return box_box_contact(moving, moving_state, collider, collider_state);
        }
        if (collider.shape.type == ShapeType::sphere) {
            Contact reverse = sphere_box_contact(
                collider_state.position, collider.shape.dimensions.x, moving,
                moving_state);
            if (!reverse.hit) {
                return {};
            }
            reverse.normal = multiply(reverse.normal, -1.0F);
            const float support =
                shape_support(moving.shape, moving_state.orientation, reverse.normal);
            reverse.point = subtract(moving_state.position,
                                     multiply(reverse.normal, support));
            return reverse;
        }
        if (collider.shape.type == ShapeType::capsule) {
            Contact reverse = capsule_box_contact(collider, collider_state, moving,
                                                  moving_state);
            if (!reverse.hit) {
                return {};
            }
            reverse.normal = multiply(reverse.normal, -1.0F);
            const float support =
                shape_support(moving.shape, moving_state.orientation, reverse.normal);
            reverse.point = subtract(moving_state.position,
                                     multiply(reverse.normal, support));
            return reverse;
        }
    }
    return {};
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

__device__ void apply_contact_response(
    const BodyParameters &body, RigidBodyState &state,
    const BodyParameters &collider, const RigidBodyState &collider_state,
    const Contact &contact) noexcept {
    state.position = add(state.position,
                         multiply(contact.normal, contact.penetration + 1.0e-5F));

    const Vec3 body_arm = subtract(contact.point, state.position);
    const Vec3 collider_arm = subtract(contact.point, collider_state.position);
    const Vec3 body_velocity =
        add(state.linear_velocity, cross(state.angular_velocity, body_arm));
    const Vec3 collider_velocity =
        add(collider_state.linear_velocity,
            cross(collider_state.angular_velocity, collider_arm));
    Vec3 relative_velocity = subtract(body_velocity, collider_velocity);
    const float normal_speed = dot(relative_velocity, contact.normal);
    if (normal_speed >= 0.0F) {
        return;
    }

    const Vec3 body_cross = cross(body_arm, contact.normal);
    const Vec3 angular_term =
        cross(inverse_inertia_world(body, state, body_cross), body_arm);
    const float denominator = body.inverse_mass + dot(angular_term, contact.normal);
    if (denominator <= k_epsilon) {
        return;
    }

    const float restitution = fminf(body.restitution, collider.restitution);
    const float normal_impulse = -(1.0F + restitution) * normal_speed / denominator;
    const Vec3 normal_vector = multiply(contact.normal, normal_impulse);
    state.linear_velocity =
        add(state.linear_velocity, multiply(normal_vector, body.inverse_mass));
    state.angular_velocity =
        add(state.angular_velocity,
            inverse_inertia_world(body, state, cross(body_arm, normal_vector)));

    relative_velocity = subtract(
        add(state.linear_velocity, cross(state.angular_velocity, body_arm)),
        collider_velocity);
    Vec3 tangent = subtract(relative_velocity,
                            multiply(contact.normal,
                                     dot(relative_velocity, contact.normal)));
    const float tangent_length = vector_length(tangent);
    if (tangent_length <= k_epsilon) {
        return;
    }
    tangent = multiply(tangent, 1.0F / tangent_length);
    const Vec3 tangent_cross = cross(body_arm, tangent);
    const float tangent_denominator =
        body.inverse_mass +
        dot(cross(inverse_inertia_world(body, state, tangent_cross), body_arm),
            tangent);
    if (tangent_denominator <= k_epsilon) {
        return;
    }
    float tangent_impulse = -dot(relative_velocity, tangent) / tangent_denominator;
    const float friction_limit =
        sqrtf(body.friction * collider.friction) * normal_impulse;
    tangent_impulse =
        clamp_scalar(tangent_impulse, -friction_limit, friction_limit);
    const Vec3 tangent_vector = multiply(tangent, tangent_impulse);
    state.linear_velocity =
        add(state.linear_velocity, multiply(tangent_vector, body.inverse_mass));
    state.angular_velocity =
        add(state.angular_velocity,
            inverse_inertia_world(body, state, cross(body_arm, tangent_vector)));
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

__global__ void resolve_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || parameters[index].motion != MotionType::dynamic) {
        return;
    }

    RigidBodyState state = states[index];
    for (int pass = 0; pass < 4; ++pass) {
        bool found_contact = false;
        for (std::uint32_t collider_index = 0; collider_index < count;
             ++collider_index) {
            if (collider_index == index ||
                parameters[collider_index].motion == MotionType::dynamic) {
                continue;
            }
            const Contact contact = collide_shapes(
                parameters[index], state, parameters[collider_index],
                states[collider_index]);
            if (!contact.hit) {
                continue;
            }
            apply_contact_response(parameters[index], state,
                                   parameters[collider_index],
                                   states[collider_index], contact);
            found_contact = true;
        }
        if (!found_contact) {
            break;
        }
    }
    state.linear_velocity =
        clamp_length(state.linear_velocity, parameters[index].maximum_linear_speed);
    state.angular_velocity =
        clamp_length(state.angular_velocity, parameters[index].maximum_angular_speed);
    states[index] = state;
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

[[nodiscard]] Status validate_shape(const RigidBodyOptions &options) noexcept {
    const Vec3 dimensions = options.shape.dimensions;
    if (!finite(dimensions)) {
        return failure(StatusCode::invalid_argument,
                       "collision shape dimensions must be finite");
    }
    switch (options.shape.type) {
    case ShapeType::sphere:
        if (dimensions.x <= 0.0F) {
            return failure(StatusCode::invalid_argument,
                           "sphere radius must be positive");
        }
        break;
    case ShapeType::box:
        if (dimensions.x <= 0.0F || dimensions.y <= 0.0F ||
            dimensions.z <= 0.0F) {
            return failure(StatusCode::invalid_argument,
                           "box half extents must be positive");
        }
        break;
    case ShapeType::capsule:
        if (dimensions.x <= 0.0F || dimensions.y < 0.0F) {
            return failure(StatusCode::invalid_argument,
                           "capsule radius must be positive and half-height nonnegative");
        }
        break;
    case ShapeType::plane:
        if (options.motion == MotionType::dynamic) {
            return failure(StatusCode::invalid_argument,
                           "plane bodies cannot be dynamic");
        }
        break;
    }
    return success();
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
    Status shape_status = validate_shape(options);
    if (!shape_status) {
        return shape_status;
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
        options.maximum_angular_speed <= 0.0F) {
        return failure(StatusCode::invalid_argument,
                       "rigid body material and limits are invalid");
    }
    return success();
}

[[nodiscard]] Vec3 derived_inertia(const RigidBodyOptions &options) noexcept {
    const float mass = options.mass;
    const Vec3 dimensions = options.shape.dimensions;
    switch (options.shape.type) {
    case ShapeType::sphere: {
        const float moment = 0.4F * mass * dimensions.x * dimensions.x;
        return {moment, moment, moment};
    }
    case ShapeType::box:
        return {mass * (dimensions.y * dimensions.y +
                        dimensions.z * dimensions.z) /
                    3.0F,
                mass * (dimensions.x * dimensions.x +
                        dimensions.z * dimensions.z) /
                    3.0F,
                mass * (dimensions.x * dimensions.x +
                        dimensions.y * dimensions.y) /
                    3.0F};
    case ShapeType::capsule: {
        const float radius = dimensions.x;
        const float half_height = dimensions.y;
        const float cylinder_volume = 2.0F * half_height * radius * radius;
        const float sphere_volume = 4.0F * radius * radius * radius / 3.0F;
        const float total_volume = cylinder_volume + sphere_volume;
        const float cylinder_mass = mass * cylinder_volume / total_volume;
        const float cap_mass = mass - cylinder_mass;
        const float axial = 0.5F * cylinder_mass * radius * radius +
                            0.4F * cap_mass * radius * radius;
        const float transverse =
            cylinder_mass * (3.0F * radius * radius +
                             4.0F * half_height * half_height) /
                12.0F +
            cap_mass * (0.4F * radius * radius +
                        half_height * half_height);
        return {transverse, axial, transverse};
    }
    case ShapeType::plane:
        return {1.0F, 1.0F, 1.0F};
    }
    return {1.0F, 1.0F, 1.0F};
}

[[nodiscard]] BodyParameters make_parameters(
    const RigidBodyOptions &options) noexcept {
    const Vec3 inertia = zero(options.inertia_diagonal)
                             ? derived_inertia(options)
                             : options.inertia_diagonal;
    const float inverse_mass =
        options.motion == MotionType::dynamic ? 1.0F / options.mass : 0.0F;
    const Vec3 inverse_inertia = options.motion == MotionType::dynamic
                                     ? Vec3{1.0F / inertia.x, 1.0F / inertia.y,
                                            1.0F / inertia.z}
                                     : Vec3{};
    return {options.motion,
            options.shape,
            inverse_mass,
            inverse_inertia,
            options.friction,
            options.restitution,
            options.linear_damping,
            options.angular_damping,
            options.maximum_linear_speed,
            options.maximum_angular_speed,
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
    std::uint32_t current_state{};
    std::uint64_t frame_index{};
    std::uint64_t revision{};
    std::vector<Slot> slots{};
    BodyParameters *parameters{};
    BodyAccumulator *accumulators{};
    KinematicTarget *targets{};
    RigidBodyId *ids{};
    RigidBodyState *states[2]{};
    std::shared_ptr<CompletionState> frame{};

    ~Impl() {
        if (frame && !frame->acknowledged) {
            (void)wait_for_completion(frame);
        }
        release_managed(states[1]);
        release_managed(states[0]);
        release_managed(ids);
        release_managed(targets);
        release_managed(accumulators);
        release_managed(parameters);
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
    if (options.rigid_body_capacity == 0U) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity must be positive");
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

    output.impl_ = std::move(implementation);
    return success();
}

Status World::add_fluid(FluidOptions, DeviceSpan<const FluidParticle>, FluidId &,
                        cudaStream_t) noexcept {
    return failure(StatusCode::not_supported,
                   "fluid implementation is scheduled for PR 3");
}

Status World::remove_fluid(FluidId, cudaStream_t) noexcept {
    return failure(StatusCode::not_supported,
                   "fluid implementation is scheduled for PR 3");
}

Status World::fluid_view(FluidId, FluidDeviceView &) const noexcept {
    return failure(StatusCode::not_supported,
                   "fluid implementation is scheduled for PR 3");
}

Status World::add_particle_spawn_plane(ParticleSpawnPlaneOptions,
                                       ParticleSpawnPlaneId &) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 3");
}

Status World::update_particle_spawn_plane(ParticleSpawnPlaneId,
                                          ParticleSpawnPlaneOptions) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 3");
}

Status World::remove_particle_spawn_plane(ParticleSpawnPlaneId) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 3");
}

Status World::add_particle_destroy_plane(ParticleDestroyPlaneOptions,
                                         ParticleDestroyPlaneId &) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 3");
}

Status World::update_particle_destroy_plane(ParticleDestroyPlaneId,
                                            ParticleDestroyPlaneOptions) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 3");
}

Status World::remove_particle_destroy_plane(ParticleDestroyPlaneId) noexcept {
    return failure(StatusCode::not_supported,
                   "particle lifecycle implementation is scheduled for PR 3");
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
    impl_->parameters[dense] = make_parameters(options);
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
        resolve_rigid_contacts_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->states[output_state],
            impl_->rigid_body_count);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error, "rigid contact kernel launch failed");
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
    }
    error = cudaEventRecord(frame->event, stream);
    if (error != cudaSuccess) {
        cudaStreamSynchronize(stream);
        return cuda_failure(error, "failed to record frame completion event");
    }

    ++impl_->frame_index;
    ++impl_->revision;
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
    output.allocated_bytes =
        capacity * (sizeof(BodyParameters) + sizeof(BodyAccumulator) +
                    sizeof(KinematicTarget) + sizeof(RigidBodyId) +
                    2U * sizeof(RigidBodyState));
    return success();
}

int World::device_ordinal() const noexcept {
    return impl_ ? impl_->device_ordinal : -1;
}

} // namespace parallel_mater
