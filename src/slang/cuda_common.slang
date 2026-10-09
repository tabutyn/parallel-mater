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

void pmGroupBarrier() { GroupMemoryBarrierWithGroupSync(); }
