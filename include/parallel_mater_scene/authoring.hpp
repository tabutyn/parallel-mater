// SPDX-License-Identifier: MIT
#pragma once
#include "types.hpp"
// Tooling scene inputs, translated into public pm* descriptors before use.
// Defaults belong to the scene/corpus profile, not to the private solver ABI.
namespace parallel_mater::gallery {

struct RopeAttachment {
    Vec3 local_anchor{};
    bool enabled{};
};

struct RopeOptions {
    // World-space open polyline. API resamples it and copies all input data.
    HostSpan<const Vec3> centerline{};
    // At most two radii; collisions cover segments, not just sampled nodes.
    float node_spacing{0.02F};
    float radius{0.01F};
    float mass{0.1F}; // Total rope mass, distributed over sampled nodes.
    float stretch_compliance{0.0F};
    float velocity_damping{0.1F};
    float maximum_substep_timestep{1.0F / 480.0F};
    float friction{0.4F};
    float maximum_speed{8.0F};
    // Nominal budget; high-strain contact recovery allows up to 4x (max 32).
    std::uint32_t solver_iterations{24U};
    bool self_collision{true};
    RopeAttachment first{};
    RopeAttachment last{};
};

// Smoke is a tracer gas, separate from liquid. It may use a projected air grid
// or the older weakly compressible particle-only solver. Coupled obstacles
// are triangle meshes, not analytic sphere colliders.
struct SmokeOptions {
    std::uint32_t capacity{4'500U};
    Vec3 emitter_center{};
    Vec2 emitter_half_extents{0.25F, 0.25F}; // Y and Z on a world-X plane
    Vec3 initial_velocity{1.6F, 0.0F, 0.0F};
    Vec3 wind{1.6F, 0.0F, 0.0F};
    float particles_per_second{900.0F};
    float lifetime{5.0F};
    float particle_radius{0.085F};
    float buoyancy{0.12F};
    // Particle-only relaxation toward wind, inverse seconds; ignored by grid mode.
    float response{0.5F};
    float rest_number_density{12.0F};
    float pressure_stiffness{2.0F};
    float viscosity{0.02F};
    float vorticity_confinement{0.1F};
    float maximum_speed{4.0F};
    // Optional Eulerian air field. Zero retains the particle-only solver.
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{32U};
    std::uint32_t grid_pressure_iterations{24U};
    float grid_kinematic_viscosity{1.5e-5F};
    float grid_les_coefficient{0.12F};
    float grid_pressure_tolerance{1.0e-3F};
    // A zero edge length chooses a shallow domain around emitter travel.
    Vec3 grid_minimum{};
    float grid_edge_length{};
};

struct FluidOptions {
    std::uint32_t capacity{};
    float particle_radius{0.0225F};
    float rest_density{1'000.0F};
    float support_radius{0.09F};
    std::uint32_t solver_iterations{4U};
    std::uint32_t maximum_neighbors{128U};
    float repulsion{50.0F};
    float viscosity{0.02F};
    float velocity_damping{0.2F};
    float maximum_speed{8.0F};
    // Opposes pairwise approach/separation along the contact axis. Unlike
    // viscosity it does not damp tangential motion between particles.
    float normal_damping{0.0F};
    // Optional physical volume carried by one particle. Zero preserves the
    // historical cubic-diameter estimate for callers without a known lattice.
    float rest_particle_volume{0.0F};
    // Zero disables the cap. Useful when dense initial packs create a short
    // repulsion spike that would otherwise overwhelm contact resolution.
    float maximum_pair_acceleration{0.0F};
};

// A finite rectangle. orientation rotates local +Y into the plane normal;
// half_extents are measured along local X and Z.
struct ParticlePlane {
    Vec3 center{};
    Quaternion orientation{};
    Vec2 half_extents{0.5F, 0.5F};
};

struct ParticleSourceOptions {
    Vec3 initial_velocity{};
    float initial_temperature{20.0F};
    bool enabled{true};
};

enum class CrossingDirection : std::uint8_t {
    along_normal,
    against_normal,
    either,
};

struct ParticleDestroyPlaneOptions {
    ParticlePlane plane{};
    CrossingDirection crossing{CrossingDirection::either};
    bool enabled{true};
};

enum class MotionType : std::uint32_t {
    static_body,
    kinematic,
    dynamic,
};

// A non-colliding oriented trigger volume. Rigid bodies match when their
// collision triangles touch or enter the closed box. Fluid particles match
// when their centers are inside it, including the boundary.
struct HitBox {
    Vec3 center{};
    Quaternion orientation{};
    Vec3 half_extents{0.5F, 0.5F, 0.5F};
};

struct RigidBodyOptions {
    MotionType motion{MotionType::dynamic};
    RigidBodyState initial_state{};
    float mass{1.0F};
    // Zero requests an AABB inertia approximation derived from the mesh.
    Vec3 inertia_diagonal{};
    float friction{0.5F};
    float restitution{};
    float linear_damping{0.05F};
    float angular_damping{0.05F};
    float maximum_linear_speed{100.0F};
    float maximum_angular_speed{100.0F};
    // Broad/narrow-phase search distance. Solver uses at most 0.001 m of it
    // as a rest offset; larger values only enlarge contact search.
    float collision_margin{0.005F};
    std::uint64_t user_data{};
};

enum class RigidConstraintType : std::uint32_t {
    fixed,
    point,
    hinge,
    slider,
    piston,
    generic,
    generic_spring,
    motor,
};

struct RigidConstraintLimitOptions {
    // Bit mask of rigid_constraint_axis_* values. Disabled axes remain free.
    std::uint32_t axes{};
    Vec3 lower{};
    Vec3 upper{};
};

struct RigidConstraintSpringOptions {
    // Bit mask of rigid_constraint_axis_* values. Stiffness uses N/m for
    // translation and N*m/rad for rotation; damping uses matching SI units.
    std::uint32_t axes{};
    Vec3 stiffness{};
    Vec3 damping{};
};

struct RigidConstraintMotorOptions {
    std::uint32_t linear_enabled{};
    std::uint32_t angular_enabled{};
    // Linear motor follows frame X. Angular motor rotates around frame X.
    float linear_target_velocity{};
    float linear_maximum_impulse{1.0F};
    float angular_target_velocity{};
    float angular_maximum_impulse{1.0F};
};

struct RigidConstraintOptions {
    RigidConstraintType type{RigidConstraintType::fixed};
    Vec3 local_anchor_a{};
    Vec3 local_anchor_b{};
    Quaternion local_orientation_a{};
    Quaternion local_orientation_b{};
    RigidConstraintLimitOptions linear_limits{};
    RigidConstraintLimitOptions angular_limits{};
    RigidConstraintSpringOptions linear_springs{};
    RigidConstraintSpringOptions angular_springs{};
    RigidConstraintMotorOptions motor{};
    std::uint32_t enabled{1U};
    // Fixed joints suppress contacts throughout the component connected by
    // enabled, unbroken fixed joints with this flag. Other joints suppress
    // only their own body pair. Contacts outside the component are preserved.
    std::uint32_t disable_collisions{1U};
    // Zero disables breaking. Otherwise this is maximum accumulated impulse
    // accepted during one substep before constraint disables itself.
    float breaking_impulse_threshold{};
    std::uint32_t solver_iterations{8U};
};

} // namespace parallel_mater::gallery
