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

struct SmokeId
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

struct HitBox
{
    Vec3 center;
    Quaternion orientation;
    Vec3 half_extents;
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

struct RigidBodyState
{
    Vec3 position;
    Quaternion orientation;
    Vec3 linear_velocity;
    Vec3 angular_velocity;
};

struct BvhNode
{
    Vec3 minimum;
    Vec3 maximum;
    uint left;
    uint right;
    uint first_triangle;
    uint triangle_count;
};

struct CollisionPlane
{
    Vec3 normal;
    float offset;
};

struct TriangleMeshResource
{
    Vec3* vertices;
    uint* indices;
    uint vertex_count;
    uint index_count;
    uint generation;
    bool alive;
    Vec3 minimum;
    Vec3 maximum;
    Vec3 bounding_center;
    float bounding_radius;
    Vec3 unit_inertia;
    BvhNode* bvh_nodes;
    uint bvh_node_count;
    uint* bvh_leaves;
    uint bvh_leaf_count;
    CollisionPlane* solid_planes;
    uint solid_unique_plane_count;
    Vec3* shell_normals;
};

struct WorldAabb
{
    Vec3 minimum;
    Vec3 maximum;
};

struct BoundsTransform
{
    Vec3 position;
    Vec3 axis_x;
    Vec3 axis_y;
    Vec3 axis_z;
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

struct SmokeOptions
{
    uint capacity;
    Vec3 emitter_center;
    Vec2 emitter_half_extents;
    Vec3 initial_velocity;
    Vec3 wind;
    float particles_per_second;
    float lifetime;
    float particle_radius;
    float buoyancy;
    float response;
    float rest_number_density;
    float pressure_stiffness;
    float viscosity;
    float vorticity_confinement;
    float maximum_speed;
    uint grid_resolution;
    uint grid_vertical_resolution;
    uint grid_pressure_iterations;
    float grid_kinematic_viscosity;
    float grid_les_coefficient;
    float grid_pressure_tolerance;
    Vec3 grid_minimum;
    float grid_edge_length;
};

struct SmokeGridField
{
    uint n;
    uint height;
    Vec3 minimum;
    float spacing;
    Vec3* velocity;
    float* density;
    float* pressure;
    float* face[3];
    Vec3* vorticity;
    float* strain;
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

struct FluidSmokeCouplingOptions
{
    FluidId fluid;
    SmokeId smoke;
    ParticlePlane heater;
    float heater_temperature;
    float boiling_temperature;
    float heat_transfer_rate;
    float wind_drag;
    float steam_rise_speed;
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

struct FluidMovingContact
{
    Vec3 normal;
    Vec3 point;
    float penetration;
    uint body;
};

struct FluidSoftContact
{
    uint nodes[12];
    float weights[12];
    uint count;
    Vec3 normal;
    Vec3 relative_velocity;
    float penetration;
};

struct PaintFieldId
{
    uint index;
    uint generation;
};

struct DeviceVec2Span
{
    Vec2* data;
    uint64_t size;
};

struct PaintFieldOptions
{
    RigidBodyId body;
    TriangleMeshId mesh;
    ClothId cloth;
    DeviceVec2Span vertex_uvs;
    uint width;
    uint height;
};

struct PaintRuleOptions
{
    FluidId source;
    RigidBodyId rigid_source;
    PaintFieldId target;
    float reach;
    float brush_radius;
    bool enabled;
};

struct PaintFieldResource
{
    PaintFieldOptions options;
    Vec2* uvs;
    uint* pixels;
    uint generation;
    bool alive;
};

struct PaintRuleResource
{
    PaintRuleOptions options;
    uint generation;
    bool alive;
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

struct RopeSoftTarget
{
    Vec3* surface;
    Vec3* previous_surface;
    Vec3* velocities;
    Vec3* bounds;
    BvhNode* tree;
    uint* order;
    uint* indices;
    SoftBodySurfaceBinding* bindings;
    float* inverse_masses;
    Vec3* node_impulses;
    uint* contact_count;
    float* maximum_penetration;
    uint triangle_count;
    uint surface_vertex_count;
    float distance;
    float friction;
    float orientation;
    bool first_anchor;
    bool last_anchor;
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
    float orientationDot = clamp_scalar(abs(
        previous.orientation.x * current.orientation.x +
        previous.orientation.y * current.orientation.y +
        previous.orientation.z * current.orientation.z +
        previous.orientation.w * current.orientation.w), 0.0f, 1.0f);
    float sineHalfAngle = sqrt(max(
        0.0f, 1.0f - orientationDot * orientationDot));
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

Vec3 quaternion_delta_velocity(Quaternion from, Quaternion to, float timestep)
{
    Quaternion delta = quaternion_multiply(to, conjugate(from));
    if (delta.w < 0.0f)
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    delta = normalized_quaternion(delta);
    float vectorSize = sqrt(
        delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    if (timestep <= 0.0f) return {};
    if (vectorSize <= kEpsilon)
        return {delta.x * (2.0f / timestep),
                delta.y * (2.0f / timestep),
                delta.z * (2.0f / timestep)};
    float angle = 2.0f * atan2(vectorSize,
        clamp_scalar(delta.w, -1.0f, 1.0f));
    float scale = angle / (vectorSize * timestep);
    return {delta.x * scale, delta.y * scale, delta.z * scale};
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
__target_intrinsic(cuda, "atomicAdd") uint64_t pmAtomicAdd(
    uint64_t* value, uint64_t addend);
__target_intrinsic(cuda, "atomicAdd") float pmAtomicAdd(float* value, float addend);
__target_intrinsic(cuda, "atomicMax") uint pmAtomicMax(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicMin") uint pmAtomicMin(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicOr") uint pmAtomicOr(uint* value, uint candidate);
__target_intrinsic(cuda, "atomicOr") uint64_t pmAtomicOr(
    uint64_t* value, uint64_t candidate);
__target_intrinsic(cuda, "atomicExch") uint pmAtomicExchange(uint* value, uint replacement);
__target_intrinsic(cuda, "__syncwarp") void pmSyncWarp(uint mask);

void pmGroupBarrier() { GroupMemoryBarrierWithGroupSync(); }
