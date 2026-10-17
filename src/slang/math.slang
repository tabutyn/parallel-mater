// SPDX-License-Identifier: MIT
// Shared numerical helpers used by every physics domain.

import types;

Vec3 add(Vec3 a, Vec3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 subtract(Vec3 a, Vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 multiply(Vec3 value, float scalar)
{
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

float dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 component_min(Vec3 a, Vec3 b)
{
    return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

Vec3 component_max(Vec3 a, Vec3 b)
{
    return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}

bool bounds_overlap(Vec3 minimumA, Vec3 maximumA,
                    Vec3 minimumB, Vec3 maximumB)
{
    return minimumA.x <= maximumB.x && maximumA.x >= minimumB.x &&
           minimumA.y <= maximumB.y && maximumA.y >= minimumB.y &&
           minimumA.z <= maximumB.z && maximumA.z >= minimumB.z;
}

float length_squared(Vec3 value) { return dot(value, value); }
float vector_length(Vec3 value) { return sqrt(length_squared(value)); }

float clamp_scalar(float value, float minimum, float maximum)
{
    return min(max(value, minimum), maximum);
}

Vec3 normalized_or(Vec3 value, Vec3 fallback)
{
    float squared = length_squared(value);
    return squared <= kEpsilon * kEpsilon
        ? fallback : multiply(value, rsqrt(squared));
}

Quaternion conjugate(Quaternion value)
{
    return {-value.x, -value.y, -value.z, value.w};
}

Quaternion quaternion_multiply(Quaternion a, Quaternion b)
{
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

Quaternion normalized_quaternion(Quaternion value)
{
    float squared = value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w;
    if (squared <= kEpsilon * kEpsilon) return {};
    float inverse = rsqrt(squared);
    return {value.x * inverse, value.y * inverse, value.z * inverse,
            value.w * inverse};
}

Quaternion relative_quaternion(Quaternion from, Quaternion to, bool local)
{
    float cosine = from.x * to.x + from.y * to.y +
        from.z * to.z + from.w * to.w;
    if (cosine < 0.0f) to = {-to.x, -to.y, -to.z, -to.w};
    // Subtract before the Hamilton product. Directly multiplying nearly equal
    // orientations cancels large terms and invents a small rotation even when
    // the poses are identical (especially with fused multiply-add).
    Quaternion difference = {to.x - from.x, to.y - from.y,
        to.z - from.z, to.w - from.w};
    Quaternion result = local
        ? quaternion_multiply(conjugate(from), difference)
        : quaternion_multiply(difference, conjugate(from));
    result.w = abs(cosine);
    return normalized_quaternion(result);
}

Vec3 rotate(Quaternion orientation, Vec3 value)
{
    Vec3 q = {orientation.x, orientation.y, orientation.z};
    Vec3 twiceCross = multiply(cross(q, value), 2.0f);
    return add(value,
        add(multiply(twiceCross, orientation.w), cross(q, twiceCross)));
}

Vec3 inverse_rotate(Quaternion orientation, Vec3 value)
{
    return rotate(conjugate(orientation), value);
}

// R(to)*value - R(from)*value, evaluated from the small quaternion change.
// Subtracting two full rotated arms loses the displacement of resting contacts.
Vec3 rotation_displacement(Quaternion from, Quaternion to, Vec3 value)
{
    if (from.x * to.x + from.y * to.y + from.z * to.z + from.w * to.w < 0.0f)
        to = {-to.x, -to.y, -to.z, -to.w};
    Vec3 oldVector = {from.x, from.y, from.z};
    Vec3 newVector = {to.x, to.y, to.z};
    Vec3 difference = subtract(newVector, oldVector);
    Vec3 oldCross = cross(oldVector, value);
    Vec3 differenceCross = cross(difference, value);
    return multiply(add(add(multiply(oldCross, to.w - from.w),
                            multiply(differenceCross, to.w)),
                        add(cross(difference, oldCross),
                            cross(newVector, differenceCross))), 2.0f);
}

Vec3 transform_point(RigidBodyState state, Vec3 point)
{
    return add(state.position, rotate(state.orientation, point));
}

BoundsTransform bounds_transform(RigidBodyState state)
{
    BoundsTransform result;
    result.position = state.position;
    result.axis_x = rotate(state.orientation, {1.0f, 0.0f, 0.0f});
    result.axis_y = rotate(state.orientation, {0.0f, 1.0f, 0.0f});
    result.axis_z = rotate(state.orientation, {0.0f, 0.0f, 1.0f});
    return result;
}

void transformed_bounds(
    Vec3 local_minimum, Vec3 local_maximum, BoundsTransform transform,
    float margin, out Vec3 minimum, out Vec3 maximum)
{
    Vec3 localCenter = multiply(add(local_minimum, local_maximum), 0.5f);
    Vec3 localHalf = multiply(subtract(local_maximum, local_minimum), 0.5f);
    Vec3 worldCenter = add(transform.position,
        add(multiply(transform.axis_x, localCenter.x),
            add(multiply(transform.axis_y, localCenter.y),
                multiply(transform.axis_z, localCenter.z))));
    Vec3 worldHalf = {
        abs(transform.axis_x.x) * localHalf.x +
            abs(transform.axis_y.x) * localHalf.y +
            abs(transform.axis_z.x) * localHalf.z,
        abs(transform.axis_x.y) * localHalf.x +
            abs(transform.axis_y.y) * localHalf.y +
            abs(transform.axis_z.y) * localHalf.z,
        abs(transform.axis_x.z) * localHalf.x +
            abs(transform.axis_y.z) * localHalf.y +
            abs(transform.axis_z.z) * localHalf.z};
    Vec3 expansion = {margin, margin, margin};
    minimum = subtract(subtract(worldCenter, worldHalf), expansion);
    maximum = add(add(worldCenter, worldHalf), expansion);
}

void transformed_motion_bounds(
    Vec3 local_minimum, Vec3 local_maximum,
    BoundsTransform previous_transform, BoundsTransform current_transform,
    bool swept, float margin, out Vec3 minimum, out Vec3 maximum)
{
    transformed_bounds(local_minimum, local_maximum, current_transform,
        margin, minimum, maximum);
    if (!swept) return;
    Vec3 previousMinimum = {}, previousMaximum = {};
    transformed_bounds(local_minimum, local_maximum, previous_transform,
        margin, previousMinimum, previousMaximum);
    minimum = component_min(minimum, previousMinimum);
    maximum = component_max(maximum, previousMaximum);
}

float rotational_motion_bound(
    RigidBodyState previous, RigidBodyState current,
    TriangleMeshResource mesh)
{
    Quaternion delta = relative_quaternion(previous.orientation,
        current.orientation, false);
    // sqrt(1-dot(q0,q1)^2) invents motion for identical rounded unit
    // quaternions and can switch a resting pair into the swept narrow phase.
    float sineHalfAngle = vector_length({delta.x, delta.y, delta.z});
    Vec3 maximumAbsolute = {
        max(abs(mesh.minimum.x), abs(mesh.maximum.x)),
        max(abs(mesh.minimum.y), abs(mesh.maximum.y)),
        max(abs(mesh.minimum.z), abs(mesh.maximum.z))};
    return 2.0f * vector_length(maximumAbsolute) * sineHalfAngle;
}

Vec3 inverse_inertia_world(
    BodyParameters parameters, RigidBodyState state, Vec3 world_vector)
{
    Vec3 local = inverse_rotate(state.orientation, world_vector);
    Vec3 transformed = {
        local.x * parameters.inverse_inertia_local.x,
        local.y * parameters.inverse_inertia_local.y,
        local.z * parameters.inverse_inertia_local.z};
    return rotate(state.orientation, transformed);
}

// Logarithm of a normalized, shortest-arc quaternion. The small-angle series
// avoids a square root and target-specific atan approximation near identity.
Vec3 quaternion_rotation_vector(Quaternion delta)
{
    float squared = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    float scale = 2.0f + squared * (1.0f / 3.0f + squared *
        (3.0f / 20.0f + squared * (5.0f / 56.0f + squared * (35.0f / 576.0f))));
    if (squared > 0.01f)
    {
        float vectorSize = sqrt(squared);
        scale = 2.0f * atan2(vectorSize, clamp_scalar(delta.w, 0.0f, 1.0f)) /
            vectorSize;
    }
    return {delta.x * scale, delta.y * scale, delta.z * scale};
}

Vec3 quaternion_delta_velocity(Quaternion from, Quaternion to, float timestep)
{
    if (timestep <= 0.0f) return {};
    return multiply(quaternion_rotation_vector(relative_quaternion(from, to, false)),
        1.0f / timestep);
}

Vec3 clamp_length(Vec3 value, float maximum)
{
    float squared = length_squared(value);
    if (squared <= maximum * maximum || squared <= kEpsilon * kEpsilon)
        return value;
    return multiply(value, maximum * rsqrt(squared));
}
