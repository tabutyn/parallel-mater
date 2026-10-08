// SPDX-License-Identifier: MIT
#pragma once

// Backend-neutral public value types. This header intentionally has no CUDA,
// Objective-C, or platform framework dependency.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace parallel_mater {

struct Vec2 {
    float x{};
    float y{};
};

// Three scalar floats are used instead of a SIMD type so CPU, CUDA, and MSL
// agree on the 12-byte storage contract.
struct Vec3 {
    float x{};
    float y{};
    float z{};
};

struct Quaternion {
    float x{};
    float y{};
    float z{};
    float w{1.0F};
};

template <typename T> struct HostSpan {
    const T *data{};
    std::uint64_t size{};

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0U; }
};

struct FluidId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(FluidId left,
                                                   FluidId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct SmokeId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(SmokeId left,
                                                   SmokeId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct FluidSmokeCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
};

struct SmokeSoftBodyCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
};

struct SmokeClothCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
};

struct ClothId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(ClothId left,
                                                   ClothId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct SoftBodyId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(SoftBodyId left,
                                                   SoftBodyId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct FluidClothCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(
        FluidClothCouplingId left, FluidClothCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct SoftBodyClothCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        SoftBodyClothCouplingId left,
        SoftBodyClothCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct FluidSoftBodyCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        FluidSoftBodyCouplingId left,
        FluidSoftBodyCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct RigidBodyId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(RigidBodyId left,
                                                   RigidBodyId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct SmokeRigidCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        SmokeRigidCouplingId left, SmokeRigidCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct RigidConstraintId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(
        RigidConstraintId left, RigidConstraintId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct TriangleMeshId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(
        TriangleMeshId left, TriangleMeshId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct RopeId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(RopeId left,
                                                   RopeId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct SmokeRopeCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        SmokeRopeCouplingId left, SmokeRopeCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct FluidRopeCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        FluidRopeCouplingId left, FluidRopeCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct RopeSoftBodyCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        RopeSoftBodyCouplingId left,
        RopeSoftBodyCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct RopeClothCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        RopeClothCouplingId left, RopeClothCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct PaintFieldId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(PaintFieldId left,
                                                   PaintFieldId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct PaintRuleId {
    std::uint32_t index{};
    std::uint32_t generation{};
};

struct ParticleSourceId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(
        ParticleSourceId left, ParticleSourceId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct ParticleDestroyPlaneId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(
        ParticleDestroyPlaneId left,
        ParticleDestroyPlaneId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct RopeAttachment {
    RigidBodyId body{};
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
    // World::step raises the shared substep count to respect live ropes' limits.
    float maximum_substep_timestep{1.0F / 480.0F};
    float friction{0.4F};
    float maximum_speed{8.0F};
    // Nominal budget; high-strain contact recovery allows up to 4x (max 32).
    std::uint32_t solver_iterations{24U};
    bool self_collision{true};
    RopeAttachment first{};
    RopeAttachment last{};
};

// Opt-in rolling physics history. A nonzero frame capacity records state and
// force diagnostics as frames complete, so it has intentional readback and
// host-memory cost. Frame stride one records every simulation frame.
struct PhysicsDebugOptions {
    std::uint32_t frame_capacity{};
    std::uint32_t frame_stride{1U};
};

struct WorldOptions {
    std::uint32_t fluid_capacity{1U};
    std::uint32_t smoke_capacity{1U};
    std::uint32_t fluid_smoke_coupling_capacity{1U};
    std::uint32_t smoke_soft_body_coupling_capacity{1U};
    std::uint32_t smoke_cloth_coupling_capacity{1U};
    std::uint32_t smoke_rope_coupling_capacity{1U};
    std::uint32_t smoke_rigid_coupling_capacity{1U};
    std::uint32_t rigid_body_capacity{64U};
    std::uint32_t rigid_constraint_capacity{64U};
    std::uint32_t triangle_mesh_capacity{16U};
    std::uint32_t particle_source_capacity{8U};
    std::uint32_t particle_destroy_plane_capacity{8U};
    std::uint32_t paint_field_capacity{8U};
    std::uint32_t paint_rule_capacity{8U};
    // Maximum diagnostic contact events retained for a requested frame.
    std::uint32_t contact_capacity{65'536U};
    bool deterministic{true};
    // Allows stable, unconstrained rigid islands to stop integrating after
    // they have remained below the velocity thresholds for half a second.
    // Metal preserves sleep under unchanged net loads (including gravity
    // compensation); changed loads, impulses and state edits wake bodies.
    bool rigid_sleeping{};
    std::uint32_t cloth_capacity{1U};
    std::uint32_t soft_body_capacity{1U};
    std::uint32_t fluid_cloth_coupling_capacity{1U};
    std::uint32_t soft_body_cloth_coupling_capacity{1U};
    std::uint32_t fluid_soft_body_coupling_capacity{1U};
    std::uint32_t rope_capacity{4U};
    std::uint32_t fluid_rope_coupling_capacity{1U};
    std::uint32_t rope_soft_body_coupling_capacity{1U};
    std::uint32_t rope_cloth_coupling_capacity{1U};
    PhysicsDebugOptions physics_debug{};
};

struct StepOptions {
    float timestep{1.0F / 60.0F};
    // Minimum count; live ropes may require smaller shared integration steps.
    std::uint32_t substeps{4U};
    Vec3 gravity{0.0F, -9.81F, 0.0F};
    // Records backend GPU timings for this frame. Disabled by default.
    bool collect_kernel_timings{};
    // Retains the frame's deterministic rigid contact diagnostics.
    bool collect_rigid_contacts{};
    // Retains the strongest contact per surviving fluid particle in stable
    // particle order; moving-body contacts take priority over static ones.
    bool collect_fluid_contacts{};
};

struct FluidParticle {
    Vec3 position{};
    Vec3 velocity{};
    float temperature{20.0F}; // degrees Celsius
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

// Particle smoke transfers local wind to a soft body. Grid smoke integrates
// projected pressure and viscous load over its skin; the skin deflects tracers.
struct SmokeSoftBodyCouplingOptions {
    SmokeId smoke{};
    SoftBodyId soft_body{};
    float wind_drag{0.5F};
    float maximum_wind_acceleration{2.0F};
    float contact_distance{};
    bool enabled{true};
};

// Particle smoke transfers local wind to cloth. Grid smoke integrates
// projected pressure and viscous load over current triangles.
struct SmokeClothCouplingOptions {
    SmokeId smoke{};
    ClothId cloth{};
    float wind_drag{2.0F};
    float maximum_wind_acceleration{20.0F};
    float contact_distance{};
    bool enabled{true};
};

struct SmokeRopeCouplingOptions {
    SmokeId smoke{};
    RopeId rope{};
    float wind_drag{2.0F};
    float maximum_wind_acceleration{20.0F};
    float contact_distance{};
    bool enabled{true};
};

struct SmokeRigidCouplingOptions {
    SmokeId smoke{};
    RigidBodyId body{};
    float air_density{1.5F};
    float drag_coefficient{4.0F};
    float contact_distance{};
    bool tracer_contact{true};
    bool enabled{true};
};

// Host geometry is copied at creation; inverse mass zero pins a vertex.
struct ClothBond {
    std::uint32_t first{};
    std::uint32_t second{};
    float rest_length{};
    bool bending{};
};

struct ClothOptions {
    HostSpan<Vec3> vertices{};
    HostSpan<std::uint32_t> triangle_indices{};
    HostSpan<float> inverse_masses{};
    float vertex_mass{0.02F};
    float thickness{0.025F};
    float stretch_compliance{1.0e-6F};
    float bending_compliance{0.1F};
    float velocity_damping{5.0F};
    float contact_friction{0.4F};
    std::uint32_t solver_iterations{8U};
    float break_strain{};
    std::uint32_t fracture_persistence_substeps{4U};
    float impact_break_impulse{};
    bool preserve_volume{};
    float target_volume{};
    float volume_compliance{1.0e-7F};
};

struct FluidClothCouplingOptions {
    FluidId fluid{};
    ClothId cloth{};
    float contact_distance{};
    float interaction_radius{};
    float stiffness{2'000.0F};
    float damping{12.0F};
    float tangential_drag{1.44F};
    float maximum_force{960.0F};
    bool enabled{true};
};

struct SoftBodyClothCouplingOptions {
    SoftBodyId soft_body{};
    ClothId cloth{};
    float contact_distance{};
    float friction{0.4F};
    std::uint32_t solver_iterations{4U};
    bool enabled{true};
};

struct FluidSoftBodyCouplingOptions {
    FluidId fluid{};
    SoftBodyId soft_body{};
    float contact_distance{};
    float friction{0.05F};
    std::uint32_t solver_iterations{4U};
    bool enabled{true};
};

struct FluidRopeCouplingOptions {
    FluidId fluid{};
    RopeId rope{};
    float contact_distance{};
    float friction{0.05F};
    float maximum_rope_acceleration{30.0F};
    bool enabled{true};
};

struct RopeSoftBodyCouplingOptions {
    RopeId rope{};
    SoftBodyId soft_body{};
    float contact_distance{};
    float friction{0.4F};
    float maximum_soft_body_acceleration{100.0F};
    float anchor_support_radius_scale{4.0F};
    float anchor_contact_support_radius_scale{1.5F};
    bool attach_first{};
    bool attach_last{};
    bool enabled{true};
};

struct RopeClothCouplingOptions {
    RopeId rope{};
    ClothId cloth{};
    std::uint32_t first_vertex{UINT32_MAX};
    std::uint32_t last_vertex{UINT32_MAX};
    float anchor_effective_mass{0.05F};
    float maximum_cloth_acceleration{20'000.0F};
    bool enabled{true};
};

struct SoftBodyBond {
    std::uint32_t first{};
    std::uint32_t second{};
    float rest_length{};
};

// Delta-skinning binding from one authored surface vertex to up to four
// physical lattice nodes. Weights must be finite, nonnegative, and sum to one.
struct SoftBodySurfaceBinding {
    std::uint32_t nodes[4]{};
    float weights[4]{};
};

// Mapping back to the input surface, for interpolating renderer-owned UVs,
// normals, or other vertex attributes after conforming triangle refinement.
struct SoftBodySurfaceSource {
    std::uint32_t vertices[3]{};
    float weights[3]{};
};

struct SoftBodyGeometrySource {
    HostSpan<Vec3> vertices{};
    HostSpan<std::uint32_t> triangle_indices{};
    // Optional full-weight Goal pins. Refined edges/faces interpolate weights;
    // only weight one is fixed. Rendering seams share one physical node.
    HostSpan<float> pin_weights{};
    float spacing{0.2F};
    float total_mass{1.0F};
    std::uint32_t maximum_nodes{100'000U};
    std::uint32_t maximum_surface_triangles{200'000U};
};

struct SoftBodyGeometry {
    std::vector<Vec3> nodes{};
    std::vector<SoftBodyBond> bonds{};
    std::vector<float> inverse_masses{};
    std::vector<Vec3> surface_vertices{};
    std::vector<std::uint32_t> surface_triangle_indices{};
    std::vector<SoftBodySurfaceBinding> surface_bindings{};
    std::vector<SoftBodySurfaceSource> surface_sources{};
    float node_mass{};
};

// Host buffers are copied during add_soft_body. Nodes and bonds describe the
// physical volume. The independently indexed surface is skinned for rendering
// and constrained against closed convex rigid triangle meshes through bindings.
struct SoftBodyOptions {
    HostSpan<Vec3> nodes{};
    HostSpan<SoftBodyBond> bonds{};
    HostSpan<float> inverse_masses{};
    HostSpan<Vec3> surface_vertices{};
    HostSpan<std::uint32_t> surface_triangle_indices{};
    HostSpan<SoftBodySurfaceBinding> surface_bindings{};
    float node_mass{0.02F};
    float node_radius{0.05F};
    float stretch_compliance{1.0e-7F};
    float velocity_damping{0.8F};
    float spring_damping{0.85F};
    float contact_friction{0.5F};
    // Rotation-invariant projection toward the best-fit rest shape. Zero
    // disables shape matching; one applies the full bounded correction each
    // substep without anchoring translation or rotation in world space.
    float shape_matching_stiffness{};
    // Clamp each graph projection relative to that node's shortest live bond.
    float maximum_projection_fraction{0.20F};
    // Fraction of projection displacement reconstructed as velocity.
    float constraint_velocity_response{0.70F};
    float maximum_speed{12.0F};
    std::uint32_t solver_iterations{8U};
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

// A finite hot plate heats nearby liquid. Boiling transfers the particle to
// smoke; particle or projected-grid smoke carrier flow also drags nearby
// liquid.
struct FluidSmokeCouplingOptions {
    FluidId fluid{};
    SmokeId smoke{};
    ParticlePlane heater{};
    float heater_temperature{500.0F};
    float boiling_temperature{100.0F};
    float heat_transfer_rate{0.2F}; // inverse seconds, at contact
    float wind_drag{2.0F}; // inverse seconds
    float steam_rise_speed{2.0F};
};

struct ParticleSourceOptions {
    FluidId fluid{};
    Vec3 initial_velocity{};
    float initial_temperature{20.0F};
    bool enabled{true};
};

// Arbitrary open or closed triangle surface, in world coordinates. Copied and
// sampled at registration; the caller may release these host buffers afterward.
struct ParticleSourceMesh {
    HostSpan<const Vec3> vertices{};
    HostSpan<const std::uint32_t> triangle_indices{};
    // Minimum distance between emission sites AND clearance from existing water.
    // Zero at registration selects the fluid support radius. Must be at least
    // the particle diameter. A site emits only when its clearance is empty.
    float spacing{};
};

enum class CrossingDirection : std::uint8_t {
    along_normal,
    against_normal,
    either,
};

struct ParticleDestroyPlaneOptions {
    FluidId fluid{};
    ParticlePlane plane{};
    CrossingDirection crossing{CrossingDirection::either};
    bool enabled{true};
};

enum class MotionType : std::uint8_t {
    static_body,
    kinematic,
    dynamic,
};

struct RigidBodyState {
    Vec3 position{};
    Quaternion orientation{};
    Vec3 linear_velocity{};
    Vec3 angular_velocity{};
};

// A non-colliding oriented trigger volume. Rigid bodies match when their
// collision triangles touch or enter the closed box. Fluid particles match
// when their centers are inside it, including the boundary.
struct HitBox {
    Vec3 center{};
    Quaternion orientation{};
    Vec3 half_extents{0.5F, 0.5F, 0.5F};
};

struct HitBoxParticle {
    FluidId fluid{};
    std::uint32_t stable_particle_id{};

    [[nodiscard]] friend constexpr bool operator==(
        HitBoxParticle left, HitBoxParticle right) noexcept {
        return left.fluid == right.fluid &&
               left.stable_particle_id == right.stable_particle_id;
    }
};

struct HitBoxResult {
    // Sorted by stable handles/IDs so repeated polls are deterministic.
    std::vector<RigidBodyId> rigid_bodies{};
    std::vector<HitBoxParticle> particles{};
};

// A closed, indexed, body-local triangle mesh sampled once into an HCP
// particle lattice. Host buffers are borrowed only for the duration of the
// call. A volume may have multiple disconnected closed components.
struct FluidGeometrySource {
    HostSpan<Vec3> vertices{};
    HostSpan<std::uint32_t> triangle_indices{};
    RigidBodyState transform{};
    Vec3 initial_velocity{};
    float spacing{0.06F};
};

struct RigidBodyOptions {
    MotionType motion{MotionType::dynamic};
    TriangleMeshId mesh{};
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

enum class RigidConstraintType : std::uint8_t {
    fixed,
    point,
    hinge,
    slider,
    piston,
    generic,
    generic_spring,
    motor,
};

inline constexpr std::uint8_t rigid_constraint_axis_x = 1U << 0U;
inline constexpr std::uint8_t rigid_constraint_axis_y = 1U << 1U;
inline constexpr std::uint8_t rigid_constraint_axis_z = 1U << 2U;
inline constexpr std::uint8_t rigid_constraint_all_axes =
    rigid_constraint_axis_x | rigid_constraint_axis_y | rigid_constraint_axis_z;

struct RigidConstraintLimitOptions {
    // Bit mask of rigid_constraint_axis_* values. Disabled axes remain free.
    std::uint8_t axes{};
    Vec3 lower{};
    Vec3 upper{};
};

struct RigidConstraintSpringOptions {
    // Bit mask of rigid_constraint_axis_* values. Stiffness uses N/m for
    // translation and N*m/rad for rotation; damping uses matching SI units.
    std::uint8_t axes{};
    Vec3 stiffness{};
    Vec3 damping{};
};

struct RigidConstraintMotorOptions {
    bool linear_enabled{};
    bool angular_enabled{};
    // Linear motor follows frame X. Angular motor rotates around frame X.
    float linear_target_velocity{};
    float linear_maximum_impulse{1.0F};
    float angular_target_velocity{};
    float angular_maximum_impulse{1.0F};
};

struct RigidConstraintOptions {
    RigidConstraintType type{RigidConstraintType::fixed};
    RigidBodyId body_a{};
    RigidBodyId body_b{};
    Vec3 local_anchor_a{};
    Vec3 local_anchor_b{};
    Quaternion local_orientation_a{};
    Quaternion local_orientation_b{};
    RigidConstraintLimitOptions linear_limits{};
    RigidConstraintLimitOptions angular_limits{};
    RigidConstraintSpringOptions linear_springs{};
    RigidConstraintSpringOptions angular_springs{};
    RigidConstraintMotorOptions motor{};
    bool enabled{true};
    // Fixed joints suppress contacts throughout the component connected by
    // enabled, unbroken fixed joints with this flag. Other joints suppress
    // only their own body pair. Contacts outside the component are preserved.
    bool disable_collisions{true};
    // Zero disables breaking. Otherwise this is maximum accumulated impulse
    // accepted during one substep before constraint disables itself.
    float breaking_impulse_threshold{};
    std::uint32_t solver_iterations{8U};
};

struct RigidConstraintState {
    bool enabled{};
    bool broken{};
    // Accumulated during the latest active substep. A broken constraint keeps
    // the impulse from the substep that exceeded its threshold.
    float applied_impulse{};
};

struct PaintRuleOptions {
    // Set exactly one source. Rigid sources paint cloth at rigid–cloth contact;
    // fluid sources paint rigid meshes at fluid–rigid contact.
    FluidId source{};
    RigidBodyId rigid_source{};
    PaintFieldId target{};
    // Additional reach beyond a fluid source's particle radius.
    float reach{0.025F};
    // World-space brush radius for rigid-to-cloth contact paint.
    float brush_radius{0.15F};
    bool enabled{true};
};

struct ContactEvent {
    FluidId fluid{};
    std::uint32_t stable_particle_id{};
    RigidBodyId rigid_body{};
    Vec3 position{};
    Vec3 normal{}; // Points from the rigid body toward the fluid particle.
    float normal_impulse{};
};

struct RigidContactEvent {
    RigidBodyId body{};
    RigidBodyId collider{};
    Vec3 position{};
    Vec3 normal{}; // Points from collider toward body.
    // Positional correction needed to restore solver's small rest offset.
    // Speculative contacts outside that offset report zero.
    float penetration{};
    float normal_impulse{};
    Vec3 friction_impulse{}; // Tangential impulse applied to body.
};

struct PhysicsDebugRigidSample {
    RigidBodyId id{};
    RigidBodyState state{};
    Vec3 applied_force{};
    Vec3 applied_torque{};
};

struct PhysicsDebugFluidSample {
    FluidId fluid{};
    std::uint32_t stable_particle_id{};
    Vec3 position{};
    Vec3 velocity{};
    Vec3 acceleration{};
    float foam{};
};

struct PhysicsDebugClothSample {
    ClothId cloth{};
    std::uint32_t vertex{};
    Vec3 position{};
    Vec3 velocity{};
    Vec3 rigid_contact_force{};
    Vec3 fluid_contact_force{};
    Vec3 soft_body_contact_force{};
};

struct PhysicsDebugSoftBodySample {
    SoftBodyId soft_body{};
    std::uint32_t node{};
    Vec3 position{};
    Vec3 velocity{};
    Vec3 rigid_contact_force{};
    Vec3 cloth_contact_force{};
    Vec3 fluid_contact_force{};
};

struct PhysicsDebugRopeSample {
    RopeId rope{};
    std::uint32_t node{};
    Vec3 position{}, velocity{}, constraint_force{}, contact_force{}, fluid_contact_force{};
};

struct PhysicsDebugFrame {
    std::uint64_t frame_index{};
    float timestep{};
    Vec3 gravity{};
    std::uint32_t maximum_fluid_neighbor_count{};
    std::vector<PhysicsDebugRigidSample> rigid_bodies{};
    std::vector<PhysicsDebugFluidSample> fluid_particles{};
    std::vector<PhysicsDebugClothSample> cloth_vertices{};
    std::vector<PhysicsDebugSoftBodySample> soft_body_nodes{};
    std::vector<PhysicsDebugRopeSample> rope_nodes{};
    std::vector<RigidContactEvent> rigid_contacts{};
    std::vector<ContactEvent> fluid_contacts{};
};

struct PhysicsDebugFrameView {
    std::uint64_t frame_index{};
    float timestep{};
    Vec3 gravity{};
    std::uint32_t maximum_fluid_neighbor_count{};
    HostSpan<PhysicsDebugRigidSample> rigid_bodies{};
    HostSpan<PhysicsDebugFluidSample> fluid_particles{};
    HostSpan<PhysicsDebugClothSample> cloth_vertices{};
    HostSpan<PhysicsDebugSoftBodySample> soft_body_nodes{};
    HostSpan<PhysicsDebugRopeSample> rope_nodes{};
    HostSpan<RigidContactEvent> rigid_contacts{};
    HostSpan<ContactEvent> fluid_contacts{};
};

struct PhysicsDebugCapture {
    std::vector<PhysicsDebugFrame> frames{};
};

struct KernelTiming {
    float total_milliseconds{};
    std::uint32_t launch_count{};
};

struct WorldStepTimings {
    std::uint64_t frame_index{};
    bool available{};
    KernelTiming rigid_integration{};
    KernelTiming rigid_world_bounds{};
    KernelTiming rigid_pair_filter{};
    KernelTiming rigid_pair_compaction{};
    KernelTiming rigid_leaf_pair_generation{};
    KernelTiming rigid_contact_evaluation{};
    // Aggregate of the five broad/narrow-phase stages above.
    KernelTiming rigid_contact_generation{};
    KernelTiming rigid_contact_solve{};
    KernelTiming rigid_input_clear{};
    KernelTiming fluid_spawn{};
    KernelTiming fluid_neighbor_sort{};
    KernelTiming fluid_neighbor_forces{};
    KernelTiming fluid_integration{};
    KernelTiming fluid_static_contacts{};
    KernelTiming fluid_body_index{};
    KernelTiming fluid_moving_contacts{};
    KernelTiming fluid_cloth_contacts{};
    KernelTiming fluid_contact_events{};
    KernelTiming fluid_outflow_compaction{};
    KernelTiming fluid_smoke_exchange{};
    float total_gpu_milliseconds{};
    KernelTiming cloth_prediction{};
    KernelTiming cloth_constraints{};
    KernelTiming cloth_contacts{};
    KernelTiming soft_body_prediction{};
    KernelTiming soft_body_constraints{};
    KernelTiming soft_body_contacts{};
    KernelTiming soft_body_cloth_contacts{};
    KernelTiming fluid_soft_body_contacts{};
    KernelTiming fluid_rope_contacts{};
    KernelTiming rope_solve{};
    KernelTiming rope_soft_body_contacts{};
    KernelTiming smoke_grid{};
    KernelTiming smoke_advection{};
    KernelTiming smoke_emission{};
};

struct WorldStatistics {
    std::uint64_t frame_index{};
    std::uint32_t fluid_count{};
    std::uint32_t particle_count{};
    std::uint32_t smoke_system_count{};
    // Occupied slots, including any expired slots awaiting reuse.
    std::uint32_t smoke_particle_count{};
    std::uint64_t emitted_smoke_particle_count{};
    std::uint64_t boiled_particle_count{};
    std::uint32_t rigid_body_count{};
    std::uint32_t sleeping_rigid_body_count{};
    std::uint32_t rigid_constraint_count{};
    std::uint32_t triangle_mesh_count{};
    std::uint32_t contact_count{};
    std::uint32_t contact_overflow_count{};
    std::uint32_t maximum_fluid_neighbor_count{};
    std::uint64_t emitted_particle_count{};
    std::uint64_t destroyed_particle_count{};
    std::uint64_t spawn_capacity_miss_count{};
    std::size_t allocated_bytes{};
    std::uint32_t cloth_count{};
    std::uint32_t cloth_vertex_count{};
    std::uint32_t soft_body_count{};
    std::uint32_t soft_body_node_count{};
    std::uint32_t rope_count{};
    std::uint32_t rope_node_count{};
    // Contact proposals (including repeated solver passes) and maximum
    // pre-correction penetration during the last frame, not residual overlap.
    std::uint32_t fluid_soft_body_contact_count{};
    float maximum_fluid_soft_body_penetration{};
    std::uint32_t fluid_rope_contact_count{};
    float maximum_fluid_rope_penetration{};
    std::uint32_t rope_soft_body_contact_count{};
    float maximum_rope_soft_body_penetration{};
};

static_assert(sizeof(Vec2) == 8U && alignof(Vec2) == 4U);
static_assert(sizeof(Vec3) == 12U && alignof(Vec3) == 4U);
static_assert(sizeof(Quaternion) == 16U && alignof(Quaternion) == 4U);
static_assert(std::is_standard_layout_v<Vec2>);
static_assert(std::is_standard_layout_v<Vec3>);
static_assert(std::is_standard_layout_v<Quaternion>);
static_assert(sizeof(FluidId) == 8U && alignof(FluidId) == 4U);
static_assert(sizeof(RigidBodyId) == 8U && alignof(RigidBodyId) == 4U);
static_assert(sizeof(TriangleMeshId) == 8U && alignof(TriangleMeshId) == 4U);
static_assert(std::is_standard_layout_v<RopeOptions>);
static_assert(std::is_standard_layout_v<WorldOptions>);
static_assert(std::is_standard_layout_v<StepOptions>);
static_assert(std::is_standard_layout_v<SmokeOptions>);
static_assert(std::is_standard_layout_v<SmokeSoftBodyCouplingOptions>);
static_assert(std::is_standard_layout_v<SmokeClothCouplingOptions>);
static_assert(std::is_standard_layout_v<SmokeRopeCouplingOptions>);
static_assert(std::is_standard_layout_v<SmokeRigidCouplingOptions>);
static_assert(std::is_standard_layout_v<ClothBond>);
static_assert(std::is_standard_layout_v<ClothOptions>);
static_assert(std::is_standard_layout_v<FluidClothCouplingOptions>);
static_assert(std::is_standard_layout_v<SoftBodyClothCouplingOptions>);
static_assert(std::is_standard_layout_v<FluidSoftBodyCouplingOptions>);
static_assert(std::is_standard_layout_v<FluidRopeCouplingOptions>);
static_assert(std::is_standard_layout_v<RopeSoftBodyCouplingOptions>);
static_assert(std::is_standard_layout_v<RopeClothCouplingOptions>);
static_assert(offsetof(Vec3, x) == 0U && offsetof(Vec3, y) == 4U &&
              offsetof(Vec3, z) == 8U);

} // namespace parallel_mater
