// SPDX-License-Identifier: MIT
// CUDA-facing ABI and numerical helpers shared by the Slang kernel mirrors.
// Keep aggregate fields scalar so their layouts match the public C++ structs.

static const float kEpsilon = 1.0e-6f;
static const uint kInvalidDense = 0xffffffffu;

struct Vec2
{
    float x;
    float y;
};

struct Vec3
{
    float x;
    float y;
    float z;
};

struct Quaternion
{
    float x;
    float y;
    float z;
    float w;
};

struct FluidId
{
    uint index;
    uint generation;
};

struct RigidBodyId
{
    uint index;
    uint generation;
};

struct TriangleMeshId
{
    uint index;
    uint generation;
};

enum MotionType : uint8_t
{
    static_body,
    kinematic,
    dynamic,
};

struct BodyParameters
{
    MotionType motion;
    TriangleMeshId mesh;
    float inverse_mass;
    Vec3 inverse_inertia_local;
    float friction;
    float restitution;
    float linear_damping;
    float angular_damping;
    float maximum_linear_speed;
    float maximum_angular_speed;
    float collision_margin;
    uint64_t user_data;
};

struct FluidParticle
{
    Vec3 position;
    Vec3 velocity;
    float temperature;
};

struct FluidOptions
{
    uint capacity;
    float particle_radius;
    float rest_density;
    float support_radius;
    uint solver_iterations;
    uint maximum_neighbors;
    float repulsion;
    float viscosity;
    float velocity_damping;
    float maximum_speed;
    float normal_damping;
    float rest_particle_volume;
    float maximum_pair_acceleration;
};

struct ClothId
{
    uint index;
    uint generation;
};

struct FluidClothCouplingOptions
{
    FluidId fluid;
    ClothId cloth;
    float contact_distance;
    float interaction_radius;
    float stiffness;
    float damping;
    float tangential_drag;
    float maximum_force;
    bool enabled;
};

struct ParticlePlane
{
    Vec3 center;
    Quaternion orientation;
    Vec2 half_extents;
};

enum CrossingDirection : uint8_t
{
    along_normal,
    against_normal,
    either,
};

struct ParticleDestroyPlaneOptions
{
    FluidId fluid;
    ParticlePlane plane;
    CrossingDirection crossing;
    bool enabled;
};

struct FluidContactSample
{
    Vec3 position;
    Vec3 normal;
    float normal_impulse;
    uint body;
};

struct ContactEvent
{
    FluidId fluid;
    uint stable_particle_id;
    RigidBodyId rigid_body;
    Vec3 position;
    Vec3 normal;
    float normal_impulse;
};

struct ClothBond
{
    uint first;
    uint second;
    float rest_length;
    bool bending;
};

struct FluidBodyImpulse
{
    Vec3 linear;
    Vec3 angular;
    uint body;
};

struct DeformableNeighbor
{
    uint index;
    float rest_length;
    float compliance;
    uint bond;
};

struct SoftBodyNeighbor
{
    uint index;
    float rest_length;
};

struct ShapeMatrix
{
    Vec3 columns[3];
};

struct SoftBodySurfaceBinding
{
    uint nodes[4];
    float weights[4];
};

struct SoftClothContact
{
    uint vertices[3];
    float weights[3];
    Vec3 position_impulse;
    Vec3 velocity_impulse;
    float soft_inverse_mass_fraction;
    bool active;
};

struct SoftSurfaceInfluence
{
    uint corner;
    float factor;
};

struct ClothBodyCorrection
{
    Vec3 offset;
    Vec3 impulse;
    Vec3 contact;
    float support_radius;
    float weight_sum;
    uint vertices[3];
    bool active;
};

struct RopeAttachment
{
    RigidBodyId body;
    Vec3 local_anchor;
    bool enabled;
};

struct HostVec3Span
{
    Vec3* data;
    uint64_t size;
};

struct RopeOptions
{
    HostVec3Span centerline;
    float node_spacing;
    float radius;
    float mass;
    float stretch_compliance;
    float velocity_damping;
    float maximum_substep_timestep;
    float friction;
    float maximum_speed;
    uint solver_iterations;
    bool self_collision;
    RopeAttachment first;
    RopeAttachment last;
};

struct RopeData
{
    RopeOptions options;
    uint count;
    Vec3* positions;
    Vec3* previous;
    Vec3* velocities;
    Vec3* constraint_forces;
    Vec3* contact_forces;
    Vec3* fluid_contact_forces;
    Vec3* soft_body_contact_forces;
    Vec3* soft_anchor_positions;
    Vec3* soft_anchor_velocities;
    Vec3* soft_anchor_impulses;
    float* soft_anchor_inverse_masses;
    bool soft_first;
    bool soft_last;
    Vec3* directions;
    Vec3* scratch;
    Vec3* body_translation;
    Vec3* body_rotation;
    Vec3* normals;
    Vec3* normals2;
    float* rest;
    float* lambda;
    uint* solid_hint;
    uint body_capacity;
    uint first_attachment_contact_skip;
    uint last_attachment_contact_skip;
};

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

Vec3 clamp_length(Vec3 value, float maximum)
{
    float squared = length_squared(value);
    if (squared <= maximum * maximum || squared <= kEpsilon * kEpsilon)
        return value;
    return multiply(value, maximum * rsqrt(squared));
}

uint pmThreadIndex()
{
    return cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
}

__target_intrinsic(cuda, "make_uint3(gridDim.x, gridDim.y, gridDim.z)")
uint3 pmGridDim();

__target_intrinsic(cuda, "atomicAdd") uint pmAtomicAdd(uint* value, uint addend);
__target_intrinsic(cuda, "atomicAdd") float pmAtomicAdd(float* value, float addend);
__target_intrinsic(cuda, "atomicMax") uint pmAtomicMax(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicExch") uint pmAtomicExchange(uint* value, uint replacement);
__target_intrinsic(cuda, "__syncwarp") void pmSyncWarp(uint mask);

void pmGroupBarrier() { GroupMemoryBarrierWithGroupSync(); }
