// SPDX-License-Identifier: MIT

// Storage ABI uses scalar members deliberately. Host structs therefore match
// std430 without scalar-block-layout or target-specific vector padding.
struct GpuVec3 {
    float x;
    float y;
    float z;
};

struct GpuQuaternion {
    float x;
    float y;
    float z;
    float w;
};

struct GpuRigidBodyId {
    uint index;
    uint generation;
};

struct GpuRigidBodyState {
    GpuVec3 position;
    GpuQuaternion orientation;
    GpuVec3 linear_velocity;
    GpuVec3 angular_velocity;
};

struct GpuRigidParameters {
    uint motion;
    uint has_kinematic_target;
    float inverse_mass;
    float inverse_inertia_x;
    float inverse_inertia_y;
    float inverse_inertia_z;
    float linear_damping;
    float angular_damping;
    float maximum_linear_speed;
    float maximum_angular_speed;
};

struct StepPushConstants {
    uint body_count;
    uint substeps;
    float timestep;
    float gravity_x;
    float gravity_y;
    float gravity_z;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuRigidBodyId> rigid_ids;
[[vk::binding(1, 0)]] RWStructuredBuffer<GpuRigidBodyState> rigid_states;
[[vk::binding(2, 0)]] RWStructuredBuffer<GpuRigidBodyState> rigid_previous_states;
[[vk::binding(3, 0)]] RWStructuredBuffer<GpuRigidParameters> rigid_parameters;
[[vk::binding(4, 0)]] RWStructuredBuffer<GpuVec3> rigid_forces;
[[vk::binding(5, 0)]] RWStructuredBuffer<GpuVec3> rigid_torques;
[[vk::binding(6, 0)]] StructuredBuffer<GpuRigidBodyState> rigid_targets;
[[vk::binding(7, 0)]] RWStructuredBuffer<GpuVec3> rigid_impulses;
[[vk::binding(8, 0)]] RWStructuredBuffer<GpuVec3> rigid_angular_impulses;
[[vk::push_constant]] ConstantBuffer<StepPushConstants> step_data;

float3 unpack_vec3(GpuVec3 value) {
    return float3(value.x, value.y, value.z);
}

GpuVec3 pack_vec3(float3 value) {
    GpuVec3 result;
    result.x = value.x;
    result.y = value.y;
    result.z = value.z;
    return result;
}

float4 unpack_quaternion(GpuQuaternion value) {
    return float4(value.x, value.y, value.z, value.w);
}

GpuQuaternion pack_quaternion(float4 value) {
    GpuQuaternion result;
    result.x = value.x;
    result.y = value.y;
    result.z = value.z;
    result.w = value.w;
    return result;
}

float4 quaternion_multiply(float4 left, float4 right) {
    return float4(
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - dot(left.xyz, right.xyz));
}

float4 quaternion_normalize(float4 value) {
    return value * rsqrt(max(dot(value, value), 1.0e-20f));
}

float3 quaternion_rotate(float4 orientation, float3 value) {
    float3 q = orientation.xyz;
    float3 twice_cross = 2.0f * cross(q, value);
    return value + orientation.w * twice_cross + cross(q, twice_cross);
}

float3 quaternion_inverse_rotate(float4 orientation, float3 value) {
    return quaternion_rotate(float4(-orientation.xyz, orientation.w), value);
}

float3 apply_inverse_inertia(
    float4 orientation, float3 inverse_diagonal, float3 angular_momentum) {
    float3 local = quaternion_inverse_rotate(orientation, angular_momentum);
    local *= inverse_diagonal;
    return quaternion_rotate(orientation, local);
}

float3 clamp_length(float3 value, float maximum_length) {
    float squared = dot(value, value);
    float maximum_squared = maximum_length * maximum_length;
    return squared > maximum_squared
        ? value * (maximum_length * rsqrt(squared))
        : value;
}

float4 integrate_orientation(float4 orientation, float3 angular_velocity,
                             float timestep) {
    float4 omega = float4(angular_velocity, 0.0f);
    float4 derivative = 0.5f * quaternion_multiply(omega, orientation);
    return quaternion_normalize(orientation + derivative * timestep);
}

float3 target_angular_velocity(float4 start, float4 target, float timestep) {
    float4 delta = quaternion_multiply(target, float4(-start.xyz, start.w));
    if (delta.w < 0.0f) delta = -delta;
    delta = quaternion_normalize(delta);
    float half_angle = acos(clamp(delta.w, -1.0f, 1.0f));
    float sine = sin(half_angle);
    if (abs(sine) < 1.0e-7f || !(timestep > 0.0f)) return float3(0.0f);
    return delta.xyz * ((2.0f * half_angle) / (sine * timestep));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_rigid_integrate(uint3 thread_id : SV_DispatchThreadID) {
    uint index = thread_id.x;
    if (index >= step_data.body_count) return;

    GpuRigidBodyState state = rigid_states[index];
    rigid_previous_states[index] = state;
    GpuRigidParameters parameters = rigid_parameters[index];

    if (parameters.motion == 0u) {
        state.linear_velocity = pack_vec3(float3(0.0f));
        state.angular_velocity = pack_vec3(float3(0.0f));
        rigid_states[index] = state;
        return;
    }

    if (parameters.motion == 1u) {
        if (parameters.has_kinematic_target != 0u) {
            GpuRigidBodyState target = rigid_targets[index];
            float3 start_position = unpack_vec3(state.position);
            float3 target_position = unpack_vec3(target.position);
            float4 start_orientation = unpack_quaternion(state.orientation);
            float4 target_orientation = quaternion_normalize(
                unpack_quaternion(target.orientation));
            state.linear_velocity = pack_vec3(
                (target_position - start_position) / step_data.timestep);
            state.angular_velocity = pack_vec3(target_angular_velocity(
                start_orientation, target_orientation, step_data.timestep));
            state.position = target.position;
            state.orientation = pack_quaternion(target_orientation);
        } else {
            state.linear_velocity = pack_vec3(float3(0.0f));
            state.angular_velocity = pack_vec3(float3(0.0f));
        }
        rigid_states[index] = state;
        return;
    }

    float3 position = unpack_vec3(state.position);
    float4 orientation = quaternion_normalize(unpack_quaternion(state.orientation));
    float3 linear_velocity = unpack_vec3(state.linear_velocity);
    float3 angular_velocity = unpack_vec3(state.angular_velocity);
    float3 inverse_inertia = float3(parameters.inverse_inertia_x,
                                    parameters.inverse_inertia_y,
                                    parameters.inverse_inertia_z);
    linear_velocity += unpack_vec3(rigid_impulses[index]) * parameters.inverse_mass;
    angular_velocity += apply_inverse_inertia(
        orientation, inverse_inertia, unpack_vec3(rigid_angular_impulses[index]));

    float substep_timestep = step_data.timestep / float(step_data.substeps);
    float3 acceleration = float3(step_data.gravity_x,
                                 step_data.gravity_y,
                                 step_data.gravity_z) +
                          unpack_vec3(rigid_forces[index]) * parameters.inverse_mass;
    float3 torque = unpack_vec3(rigid_torques[index]);
    for (uint substep = 0u; substep < step_data.substeps; ++substep) {
        linear_velocity += acceleration * substep_timestep;
        angular_velocity += apply_inverse_inertia(
            orientation, inverse_inertia, torque) * substep_timestep;
        linear_velocity /= 1.0f + parameters.linear_damping * substep_timestep;
        angular_velocity /= 1.0f + parameters.angular_damping * substep_timestep;
        linear_velocity = clamp_length(linear_velocity,
                                       parameters.maximum_linear_speed);
        angular_velocity = clamp_length(angular_velocity,
                                        parameters.maximum_angular_speed);
        position += linear_velocity * substep_timestep;
        orientation = integrate_orientation(
            orientation, angular_velocity, substep_timestep);
    }

    state.position = pack_vec3(position);
    state.orientation = pack_quaternion(orientation);
    state.linear_velocity = pack_vec3(linear_velocity);
    state.angular_velocity = pack_vec3(angular_velocity);
    rigid_states[index] = state;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_rigid_clear_inputs(uint3 thread_id : SV_DispatchThreadID) {
    uint index = thread_id.x;
    if (index >= step_data.body_count) return;
    rigid_forces[index] = pack_vec3(float3(0.0f));
    rigid_torques[index] = pack_vec3(float3(0.0f));
    rigid_impulses[index] = pack_vec3(float3(0.0f));
    rigid_angular_impulses[index] = pack_vec3(float3(0.0f));
    rigid_parameters[index].has_kinematic_target = 0u;
}
