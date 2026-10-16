// SPDX-License-Identifier: MIT
// Shared physics resource and value layouts.
// Keep aggregate fields scalar so their layouts match host-visible API records.

import graphics_resources;

#ifndef PM_TARGET_D3D12
#define PM_TARGET_D3D12 0
#endif

// DXIL cannot perform memory access through i8 values. Keep logical enum
// values identical while exposing a legal structured-buffer scalar.
#if PM_TARGET_D3D12
#define PM_STORAGE_UINT8 uint
#else
#define PM_STORAGE_UINT8 uint8_t
#endif

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

enum MotionType : uint
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

// Portable command-API output layout. Keep scalar arrays tightly packed so
// the same record is consumed by C, CUDA, Metal, D3D12, and Vulkan clients.
struct PortableParticleState
{
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    float temperature_celsius;
    float foam;
    uint stable_id;
};

struct PortableDomainOutputHeader
{
    uint active_count;
    uint capacity;
    uint64_t revision;
};

struct PortableClothVertexState
{
    Vec3 position;
    Vec3 velocity;
    Vec3 rigid_contact_force;
    Vec3 liquid_contact_force;
    Vec3 soft_body_contact_force;
    Vec3 rope_contact_force;
};

struct PortableRopeNodeState
{
    Vec3 position;
    Vec3 velocity;
    Vec3 constraint_force;
    Vec3 contact_force;
    Vec3 liquid_contact_force;
    Vec3 soft_body_contact_force;
};

struct PortableSoftBodyNodeState
{
    Vec3 position;
    Vec3 velocity;
    Vec3 rigid_contact_force;
    Vec3 cloth_contact_force;
    Vec3 liquid_contact_force;
    Vec3 rope_contact_force;
};

struct PortableSmokeParticleState
{
    Vec3 position;
    Vec3 velocity;
    float age;
    float number_density;
    float pressure;
    Vec3 vorticity;
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
    PMBuffer<Vec3> vertices;
    PMBuffer<uint> indices;
    uint vertex_count;
    uint index_count;
    uint generation;
    uint alive;
    Vec3 minimum;
    Vec3 maximum;
    Vec3 bounding_center;
    float bounding_radius;
    Vec3 unit_inertia;
    PMBuffer<BvhNode> bvh_nodes;
    uint bvh_node_count;
    PMBuffer<uint> bvh_leaves;
    uint bvh_leaf_count;
    PMBuffer<CollisionPlane> solid_planes;
    uint solid_unique_plane_count;
    PMBuffer<Vec3> shell_normals;
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
    PMBuffer<Vec3> velocity;
    PMBuffer<float> density;
    PMBuffer<float> pressure;
    PMBuffer<float> face[3];
    PMBuffer<Vec3> vorticity;
    PMBuffer<float> strain;
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

enum CrossingDirection : PM_STORAGE_UINT8
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
    PMBuffer<Vec2> data;
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
    PMBuffer<Vec2> uvs;
    PMBuffer<uint> pixels;
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
    PMBuffer<Vec3> data;
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
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> previous;
    PMBuffer<Vec3> velocities;
    PMBuffer<Vec3> constraint_forces;
    PMBuffer<Vec3> contact_forces;
    PMBuffer<Vec3> fluid_contact_forces;
    PMBuffer<Vec3> soft_body_contact_forces;
    PMBuffer<Vec3> soft_anchor_positions;
    PMBuffer<Vec3> soft_anchor_velocities;
    PMBuffer<Vec3> soft_anchor_impulses;
    PMBuffer<float> soft_anchor_inverse_masses;
    bool soft_first;
    bool soft_last;
    PMBuffer<Vec3> directions;
    PMBuffer<Vec3> scratch;
    PMBuffer<Vec3> body_translation;
    PMBuffer<Vec3> body_rotation;
    PMBuffer<Vec3> normals;
    PMBuffer<Vec3> normals2;
    PMBuffer<float> rest;
    PMBuffer<float> lambda;
    PMBuffer<uint> solid_hint;
    uint body_capacity;
    uint first_attachment_contact_skip;
    uint last_attachment_contact_skip;
};

struct RopeSoftTarget
{
    PMBuffer<Vec3> surface;
    PMBuffer<Vec3> previous_surface;
    PMBuffer<Vec3> velocities;
    PMBuffer<Vec3> bounds;
    PMBuffer<BvhNode> tree;
    PMBuffer<uint> order;
    PMBuffer<uint> indices;
    PMBuffer<SoftBodySurfaceBinding> bindings;
    PMBuffer<float> inverse_masses;
    PMBuffer<Vec3> node_impulses;
    PMBuffer<uint> contact_count;
    PMBuffer<float> maximum_penetration;
    uint triangle_count;
    uint surface_vertex_count;
    float distance;
    float friction;
    float orientation;
    bool first_anchor;
    bool last_anchor;
};
