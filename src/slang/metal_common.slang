// SPDX-License-Identifier: MIT
// Packed ABI shared by Slang kernels dispatched through Metal 4 argument tables.

struct MetalPackedVec3
{
    float x;
    float y;
    float z;
};

struct MetalPackedVec2
{
    float x;
    float y;
};

struct MetalQuaternion
{
    float x;
    float y;
    float z;
    float w;
};

struct MetalRigidState
{
    MetalPackedVec3 position;
    MetalQuaternion orientation;
    MetalPackedVec3 linear_velocity;
    MetalPackedVec3 angular_velocity;
};

struct MetalRigidParameters
{
    uint motion;
    float inverse_mass;
    MetalPackedVec3 inverse_inertia;
    float linear_damping;
    float angular_damping;
    float maximum_linear_speed;
    float maximum_angular_speed;
    uint has_kinematic_target;
    MetalRigidState kinematic_target;
    uint mesh_index;
    float friction;
    float restitution;
    float collision_margin;
};

struct MetalTriangleMeshInfo
{
    uint vertex_offset;
    uint vertex_count;
    uint index_offset;
    uint index_count;
    MetalPackedVec3 minimum;
    MetalPackedVec3 maximum;
    MetalPackedVec3 bounding_center;
    float radius;
    uint bvh_node_offset;
    uint bvh_node_count;
    uint solid_plane_offset;
    uint solid_plane_count;
};

struct MetalCollisionPlane
{
    MetalPackedVec3 normal;
    float offset;
};

struct MetalRigidStepConstants
{
    float timestep;
    MetalPackedVec3 gravity;
    uint body_count;
    uint constraint_capacity;
    uint collect_rigid_contacts;
    uint rigid_event_capacity;
    uint substeps;
    uint cooperative_contact_generation;
    uint rigid_sleeping;
    uint rigid_contact_pass_limit;
};

struct MetalAvbdDual
{
    float lambda;
    float penalty;
};

struct MetalContactRecord
{
    MetalPackedVec3 point;
    MetalPackedVec3 normal;
    float penetration;
    uint found;
    float accumulated_normal_impulse;
    MetalPackedVec3 accumulated_friction_impulse;
    float initial_normal_speed;
    float impact_fraction;
    uint persistent;
    uint warm_started;
    MetalAvbdDual avbd_dual[3];
    MetalPackedVec3 avbd_anchor_a;
    MetalPackedVec3 avbd_anchor_b;
    MetalPackedVec3 avbd_error;
    uint avbd_cached;
};

struct MetalContactManifold
{
    MetalContactRecord contacts[8];
    uint count;
    uint event_offset;
    uint color;
    MetalPackedVec3 initial_relative_position;
    uint face_patch;
    uint body_fixed_member;
    uint collider_fixed_member;
    uint cached;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct MetalCachedContact
{
    MetalPackedVec3 local_point;
    MetalPackedVec3 collider_local_point;
    MetalPackedVec3 local_normal;
    float penetration;
    float normal_impulse;
    MetalPackedVec3 friction_impulse;
    MetalAvbdDual avbd_dual[3];
    MetalPackedVec3 avbd_anchor_a;
    MetalPackedVec3 avbd_anchor_b;
};

struct MetalCachedContactPair
{
    uint body_index;
    uint body_generation;
    uint collider_index;
    uint collider_generation;
    uint64_t epoch;
    float timestep;
    uint count;
    uint face_patch;
    uint color;
    MetalCachedContact contacts[8];
};

// Metal compatibility paths only inspect the header. Keep the legacy
// 1832-byte stride so buffer indexing remains ABI-compatible with world.mm.
struct MetalRigidConstraintResource
{
    uint generation;
    uint alive;
    uint type;
    uint body_a;
    uint body_b;
    uint enabled;
    uint broken;
    MetalPackedVec3 local_anchor_a;
    MetalPackedVec3 local_anchor_b;
    MetalQuaternion local_orientation_a;
    MetalQuaternion local_orientation_b;
    float breaking_impulse_threshold;
    float applied_impulse;
    uint linear_limit_axes;
    MetalPackedVec3 linear_limit_lower;
    MetalPackedVec3 linear_limit_upper;
    uint angular_limit_axes;
    MetalPackedVec3 angular_limit_lower;
    MetalPackedVec3 angular_limit_upper;
    uint linear_spring_axes;
    MetalPackedVec3 linear_spring_stiffness;
    MetalPackedVec3 linear_spring_damping;
    uint angular_spring_axes;
    MetalPackedVec3 angular_spring_stiffness;
    MetalPackedVec3 angular_spring_damping;
    uint linear_motor_enabled;
    uint angular_motor_enabled;
    float linear_target_velocity;
    float linear_maximum_impulse;
    float angular_target_velocity;
    float angular_maximum_impulse;
    uint solver_iterations;
    uint disable_collisions;
    uint reserved[399];
};

struct MetalDeformableConstants
{
    float timestep;
    MetalPackedVec3 gravity;
    uint count;
    uint bond_count;
    uint solver_iterations;
    float compliance;
    float velocity_damping;
    float maximum_speed;
    float radius;
    float break_strain;
    uint fracture_persistence;
    float impact_break_impulse;
    uint surface_count;
    uint triangle_index_count;
    uint preserve_volume;
    float target_volume;
    float volume_compliance;
    float shape_matching_stiffness;
    float maximum_projection_fraction;
    uint self_collision;
    float spring_damping;
    float constraint_velocity_response;
};

struct MetalBond
{
    uint first;
    uint second;
    float rest_length;
    float compliance;
    uint active;
};

struct MetalDeformableNeighbor
{
    uint index;
    float rest_length;
    float compliance;
    uint bond;
};

struct MetalSurfaceBinding
{
    uint nodes[4];
    float weights[4];
};

struct MetalClothRigidConstants
{
    float timestep;
    uint vertex_count;
    uint triangle_index_count;
    uint surface_count;
    float thickness;
    float friction;
    uint rigid_count;
    uint fracture_enabled;
};

struct MetalClothBodyCorrection
{
    MetalPackedVec3 offset;
    MetalPackedVec3 impulse;
    MetalPackedVec3 contact;
    float support_radius;
    float weight_sum;
    uint vertices[3];
    uint active;
};

struct MetalParticleMetadata
{
    uint count;
    uint next_stable_id;
    uint revision;
    uint capacity;
    uint64_t emitted;
    uint64_t destroyed;
    uint64_t boiled;
};

struct MetalSplitCounter
{
    uint low;
    uint high;
};

struct MetalHandle
{
    uint index;
    uint generation;
};

struct MetalContactEvent
{
    MetalHandle fluid;
    uint stable_particle_id;
    MetalHandle rigid_body;
    MetalPackedVec3 position;
    MetalPackedVec3 normal;
    float normal_impulse;
};

struct MetalParticleRigidConstants
{
    uint particle_count;
    uint rigid_count;
    float radius;
    float friction;
    float restitution;
    float maximum_reaction_speed;
    float timestep;
    uint diagnostic_is_acceleration;
    MetalHandle fluid;
    uint collect_contacts;
    float particle_inverse_mass;
    uint first_iteration;
    uint solid_contacts;
    uint share_position;
    uint first_spawned;
    uint recover_spawn;
    float spawn_clearance;
    MetalPackedVec3 up;
    MetalPackedVec3 gravity;
    float movable_mass;
};

struct MetalParticleRigidContact
{
    MetalPackedVec3 normal;
    MetalPackedVec3 point;
    float penetration;
    uint body;
};

struct MetalSoftContactAccumulator
{
    MetalPackedVec3 momentum_delta;
    MetalPackedVec3 friction_delta;
    MetalPackedVec3 arm;
    float normal_delta;
};

struct MetalSoftContactState
{
    uint dynamic_contact_flag;
    MetalPackedVec3 predicted_momentum;
};

struct MetalCouplingConstants
{
    float timestep;
    uint count_a;
    uint count_b;
    uint mode;
    float contact_distance;
    float stiffness;
    float damping;
    float friction;
    float maximum_force;
    uint first_vertex;
    uint last_vertex;
    uint enabled;
};

struct MetalRopeAnchorState
{
    MetalPackedVec3 position;
    MetalPackedVec3 velocity;
    MetalPackedVec3 impulse;
    float inverse_mass;
};

struct MetalRopeAttachmentConstants
{
    MetalHandle first_body;
    MetalPackedVec3 first_anchor;
    uint first_enabled;
    MetalHandle last_body;
    MetalPackedVec3 last_anchor;
    uint last_enabled;
    uint rigid_count;
    uint first_soft;
    uint last_soft;
    uint first_contact_skip;
    uint last_contact_skip;
    uint rigid_capacity;
};

[ForceInline]
void pmGroupBarrier()
{
    GroupMemoryBarrierWithGroupSync();
}

[ForceInline]
static MetalContactEvent metalZeroContactEvent()
{
    MetalContactEvent result;
    result.fluid.index = 0u;
    result.fluid.generation = 0u;
    result.stable_particle_id = 0u;
    result.rigid_body.index = 0u;
    result.rigid_body.generation = 0u;
    result.position = metalStore(float3(0.0f));
    result.normal = metalStore(float3(0.0f));
    result.normal_impulse = 0.0f;
    return result;
}

[ForceInline]
static float3 metalClosestPointTriangle(
    float3 point, float3 a, float3 b, float3 c)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = point - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    float3 bp = point - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        return a + (d1 / (d1 - d3)) * ab;
    float3 cp = point - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        return a + (d2 / (d2 - d6)) * ac;
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
    float inverse = 1.0f / (va + vb + vc);
    return a + ab * (vb * inverse) + ac * (vc * inverse);
}

[ForceInline]
static float3 metalTriangleWeights(
    float3 point, float3 a, float3 b, float3 c)
{
    float3 first = b - a;
    float3 second = c - a;
    float3 offset = point - a;
    float d00 = dot(first, first);
    float d01 = dot(first, second);
    float d11 = dot(second, second);
    float d20 = dot(offset, first);
    float d21 = dot(offset, second);
    float denominator = d00 * d11 - d01 * d01;
    if (abs(denominator) <= 1.0e-14f)
        return float3(1.0f, 0.0f, 0.0f);
    float secondWeight = (d11 * d20 - d01 * d21) / denominator;
    float thirdWeight = (d00 * d21 - d01 * d20) / denominator;
    float3 weights = max(float3(
        1.0f - secondWeight - thirdWeight, secondWeight, thirdWeight), 0.0f);
    return weights / max(weights.x + weights.y + weights.z, 1.0e-12f);
}

[ForceInline]
static MetalQuaternion metalQuaternionNormalize(MetalQuaternion value)
{
    float inverseLength = rsqrt(max(value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w, 1.0e-20f));
    value.x *= inverseLength;
    value.y *= inverseLength;
    value.z *= inverseLength;
    value.w *= inverseLength;
    return value;
}

[ForceInline]
static MetalQuaternion metalQuaternionMultiply(
    MetalQuaternion first, MetalQuaternion second)
{
    MetalQuaternion result;
    result.x = first.w * second.x + first.x * second.w +
        first.y * second.z - first.z * second.y;
    result.y = first.w * second.y - first.x * second.z +
        first.y * second.w + first.z * second.x;
    result.z = first.w * second.z + first.x * second.y -
        first.y * second.x + first.z * second.w;
    result.w = first.w * second.w - first.x * second.x -
        first.y * second.y - first.z * second.z;
    return result;
}

[ForceInline]
static MetalQuaternion metalQuaternionConjugate(MetalQuaternion value)
{
    value.x = -value.x;
    value.y = -value.y;
    value.z = -value.z;
    return value;
}

[ForceInline]
static float3 metalRotate(MetalQuaternion orientation, float3 value)
{
    float3 vector = float3(orientation.x, orientation.y, orientation.z);
    float3 twiceCross = 2.0f * cross(vector, value);
    return value + orientation.w * twiceCross + cross(vector, twiceCross);
}

[ForceInline]
static float3 metalWorldPoint(MetalRigidState state, float3 local)
{
    return metalLoad(state.position) + metalRotate(state.orientation, local);
}

[ForceInline]
static float3 metalInverseInertiaMul(
    MetalRigidParameters body, MetalRigidState state, float3 value)
{
    float3 local = metalRotate(
        metalQuaternionConjugate(state.orientation), value);
    return metalRotate(
        state.orientation, local * metalLoad(body.inverse_inertia));
}

[ForceInline]
static float3 metalLoad(MetalPackedVec3 value)
{
    return float3(value.x, value.y, value.z);
}

[ForceInline]
static MetalPackedVec3 metalStore(float3 value)
{
    MetalPackedVec3 result;
    result.x = value.x;
    result.y = value.y;
    result.z = value.z;
    return result;
}

[ForceInline]
static float3 metalLimit(float3 value, float maximumLength)
{
    float squaredLength = dot(value, value);
    return squaredLength > maximumLength * maximumLength
        ? value * (maximumLength * rsqrt(squaredLength))
        : value;
}

[ForceInline]
static void metalIntegrateDeformableNode(
    RWStructuredBuffer<MetalPackedVec3> positions,
    RWStructuredBuffer<MetalPackedVec3> previous,
    RWStructuredBuffer<MetalPackedVec3> velocities,
    StructuredBuffer<float> inverseMasses,
    MetalDeformableConstants constants, uint index)
{
    if (index >= constants.count) return;
    previous[index] = positions[index];
    if (inverseMasses[index] <= 0.0f)
    {
        velocities[index] = metalStore(float3(0.0f));
        return;
    }
    float3 velocity = metalLoad(velocities[index]) /
        (1.0f + constants.velocity_damping * constants.timestep);
    velocity += metalLoad(constants.gravity) * constants.timestep;
    velocities[index] = metalStore(velocity);
    positions[index] = metalStore(
        metalLoad(positions[index]) + velocity * constants.timestep);
}

[ForceInline]
static void metalFinalizeDeformableNode(
    RWStructuredBuffer<MetalPackedVec3> positions,
    RWStructuredBuffer<MetalPackedVec3> previous,
    RWStructuredBuffer<MetalPackedVec3> velocities,
    StructuredBuffer<float> inverseMasses,
    MetalDeformableConstants constants, uint index)
{
    if (index >= constants.count) return;
    if (inverseMasses[index] <= 0.0f)
    {
        velocities[index] = metalStore(float3(0.0f));
        return;
    }
    float3 projected = (metalLoad(positions[index]) -
        metalLoad(previous[index])) / constants.timestep;
    float response = clamp(
        constants.constraint_velocity_response, 0.0f, 1.0f);
    velocities[index] = metalStore(metalLimit(
        lerp(metalLoad(velocities[index]), projected, response),
        constants.maximum_speed));
}
