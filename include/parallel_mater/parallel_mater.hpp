// SPDX-License-Identifier: MIT
#pragma once

// Public CUDA physics API. Implementation milestones are tracked in
// docs/ROADMAP.md; a declaration may precede its solver implementation.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace parallel_mater {

struct Vec2 {
    float x{};
    float y{};
};

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

template <typename T> struct DeviceSpan {
    T *data{};
    std::uint64_t size{};

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0U; }
};

template <typename T> struct HostSpan {
    const T *data{};
    std::uint64_t size{};
};

enum class StatusCode : std::uint8_t {
    success,
    invalid_argument,
    not_supported,
    invalid_handle,
    capacity_exceeded,
    busy,
    out_of_memory,
    cuda_failure,
    internal_error,
};

struct Status {
    StatusCode code{StatusCode::success};
    cudaError_t cuda_error{cudaSuccess};
    const char *message{};

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == StatusCode::success;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

struct FluidId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(FluidId left,
                                                   FluidId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
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
        FluidClothCouplingId left,
        FluidClothCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct SoftBodyClothCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        SoftBodyClothCouplingId left, SoftBodyClothCouplingId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
};

struct FluidSoftBodyCouplingId {
    std::uint32_t index{};
    std::uint32_t generation{};
    [[nodiscard]] friend constexpr bool operator==(
        FluidSoftBodyCouplingId left, FluidSoftBodyCouplingId right) noexcept {
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
        RopeSoftBodyCouplingId left, RopeSoftBodyCouplingId right) noexcept {
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

struct RopeDeviceView {
    DeviceSpan<const Vec3> positions{};
    DeviceSpan<const Vec3> velocities{};
    DeviceSpan<const Vec3> constraint_forces{};
    DeviceSpan<const Vec3> contact_forces{};
    DeviceSpan<const Vec3> fluid_contact_forces{};
    DeviceSpan<const Vec3> soft_body_contact_forces{};
    DeviceSpan<const float> rest_lengths{};
    float radius{};
};

[[nodiscard]] Status sample_rope_centerline(
    HostSpan<const Vec3> centerline, float spacing,
    std::vector<Vec3> &output) noexcept;

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
        ParticleDestroyPlaneId left, ParticleDestroyPlaneId right) noexcept {
        return left.index == right.index && left.generation == right.generation;
    }
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
    std::uint32_t rigid_body_capacity{64U};
    std::uint32_t triangle_mesh_capacity{16U};
    std::uint32_t particle_source_capacity{8U};
    std::uint32_t particle_destroy_plane_capacity{8U};
    std::uint32_t paint_field_capacity{8U};
    std::uint32_t paint_rule_capacity{8U};
    // Maximum diagnostic contact events retained for a requested frame.
    std::uint32_t contact_capacity{65'536U};
    bool deterministic{true};
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
    // Records CUDA-event timings for this frame. Disabled by default so
    // production stepping does not pay profiling overhead.
    bool collect_kernel_timings{};
    // Retains the frame's deterministic rigid contact diagnostics for a
    // renderer or inspection tool. Disabled by default.
    bool collect_rigid_contacts{};
    // Retains the strongest contact per surviving fluid particle in stable
    // particle order; moving-body contacts take priority over static ones.
    bool collect_fluid_contacts{};
};

struct FluidParticle {
    Vec3 position{};
    Vec3 velocity{};
};

// Host geometry is copied at creation; vertex inverse mass zero pins a vertex
// exactly. Triangles define stretch, shear, and bending links; they may be open.
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
    // Coulomb coefficient for tangential rigid-body/cloth contact.
    float contact_friction{0.4F};
    std::uint32_t solver_iterations{8U};
    // Zero disables fracture. Otherwise a bond fails when its extension
    // exceeds this fraction of its rest length for the configured duration.
    float break_strain{};
    std::uint32_t fracture_persistence_substeps{4U};
    // Optional immediate bond failure from the sum of its endpoint contact
    // impulses. Zero disables this additional impact criterion.
    float impact_break_impulse{};
    // Preserves the signed volume of a closed cloth surface. A zero target
    // captures the authored initial volume. Compliance is inverse stiffness;
    // zero is a hard constraint.
    bool preserve_volume{};
    float target_volume{};
    float volume_compliance{1.0e-7F};
};

// Couples one fluid to one closed cloth surface. Containment treats the cloth
// winding as outward-facing, keeps particle centers inside it, and transfers
// equal-and-opposite forces back to the cloth.
struct FluidClothCouplingOptions {
    FluidId fluid{};
    ClothId cloth{};
    // Zero selects particle_radius + cloth thickness.
    float contact_distance{};
    // Zero selects the fluid support radius.
    float interaction_radius{};
    float stiffness{2'000.0F};
    float damping{12.0F};
    float tangential_drag{1.44F};
    float maximum_force{960.0F};
    bool enabled{true};
};

// Two-sided contact with the cloth's current triangles, including its torn
// surface. Fracture is configured on ClothOptions, independently per cloth.
struct SoftBodyClothCouplingOptions {
    SoftBodyId soft_body{};
    ClothId cloth{};
    // Zero selects soft node radius + cloth thickness.
    float contact_distance{};
    float friction{0.4F};
    // 1..16. The highest enabled request sets the shared contact pass count;
    // all pairs stay constrained while shared bodies continue projecting.
    std::uint32_t solver_iterations{4U};
    bool enabled{true};
};

// External fluid contact with the current closed, consistently wound soft
// surface. Reactions follow its skin bindings, including fixed nodes. No
// analytic collider or renderer geometry is involved.
struct FluidSoftBodyCouplingOptions {
    FluidId fluid{};
    SoftBodyId soft_body{};
    // Zero selects the fluid particle radius.
    float contact_distance{};
    float friction{0.05F};
    std::uint32_t solver_iterations{4U}; // 1..16
    bool enabled{true};
};

// Fluid particles collide with the moving capsule segments of an open rope.
// Their impulses push the rope, bounded per node to keep dense splashes stable.
struct FluidRopeCouplingOptions {
    FluidId fluid{};
    RopeId rope{};
    float contact_distance{}; // Zero selects particle radius + rope radius.
    float friction{0.05F};
    float maximum_rope_acceleration{30.0F};
    bool enabled{true};
};

// Two-way contact against the current closed soft-body skin. An endpoint may
// additionally follow its closest rest-surface point. Different soft bodies
// may bind the two ends of the same rope; multiple ropes may share a body.
struct RopeSoftBodyCouplingOptions {
    RopeId rope{};
    SoftBodyId soft_body{};
    float contact_distance{}; // Zero selects rope radius.
    float friction{0.4F};
    float maximum_soft_body_acceleration{100.0F};
    // Load an attached endpoint over nearby lattice nodes. The support
    // shrinks as rope/skin contacts distribute the wrap load. Zero for both
    // scales retains barycentric point loading; units are node radii.
    float anchor_support_radius_scale{4.0F};
    float anchor_contact_support_radius_scale{1.5F};
    bool attach_first{};
    bool attach_last{};
    bool enabled{true};
};

// Binds rope endpoints to authored cloth vertices. This is a physical,
// bidirectional joint; the endpoint follows the cloth and rope tension moves
// the vertex. UINT32_MAX leaves an endpoint unattached to this cloth.
struct RopeClothCouplingOptions {
    RopeId rope{};
    ClothId cloth{};
    std::uint32_t first_vertex{UINT32_MAX};
    std::uint32_t last_vertex{UINT32_MAX};
    // Effective mass of the cloth patch supported by one endpoint, not the
    // often much smaller mass of its single authored vertex.
    float anchor_effective_mass{0.05F};
    float maximum_cloth_acceleration{20'000.0F};
    bool enabled{true};
};

struct ClothDeviceView {
    // Physical nodes and current connectivity. Tearing can append nodes and
    // reindex faces at the next frame boundary; reacquire this view each frame.
    DeviceSpan<const Vec3> positions{};
    DeviceSpan<const Vec3> velocities{};
    DeviceSpan<const std::uint32_t> triangle_indices{};
    std::uint32_t vertex_count{};
    // Tearable cloth: authored vertex for each physical node, and split masses.
    // New nodes inherit velocity and share their source's original mass.
    DeviceSpan<const std::uint32_t> vertex_source_indices{};
    DeviceSpan<const float> inverse_masses{};
    // For tearable cloth, every triangle owns three surface corners. The
    // triangle count stays fixed as bonds fail; source indices preserve UVs.
    DeviceSpan<const Vec3> surface_positions{};
    DeviceSpan<const std::uint32_t> surface_triangle_indices{};
    DeviceSpan<const std::uint32_t> surface_source_indices{};
    DeviceSpan<const ClothBond> bonds{};
    DeviceSpan<const std::uint8_t> active_bonds{};
    // Last-substep forces applied at each physical node. These diagnostics
    // remain available without enabling contact event collection, allowing
    // renderers and tools to inspect cloth coupling through the public API.
    DeviceSpan<const Vec3> rigid_contact_forces{};
    DeviceSpan<const Vec3> fluid_contact_forces{};
    DeviceSpan<const Vec3> soft_body_contact_forces{};
    DeviceSpan<const Vec3> rope_contact_forces{};
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

// Host-side preparation; no CUDA device is needed. Refine long surface edges
// to lattice resolution (maximum edge 1.5 * spacing allows triangle diagonals),
// preserve the closed input shape/winding, and connect every new physical
// surface node to the same volumetric spring lattice. Output is transactional.
[[nodiscard]] Status build_soft_body_geometry(
    SoftBodyGeometrySource source, SoftBodyGeometry &output) noexcept;

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

struct SoftBodyDeviceView {
    DeviceSpan<const Vec3> positions{};
    DeviceSpan<const Vec3> velocities{};
    DeviceSpan<const SoftBodyBond> bonds{};
    DeviceSpan<const Vec3> surface_positions{};
    DeviceSpan<const std::uint32_t> surface_triangle_indices{};
    DeviceSpan<const Vec3> rigid_contact_forces{};
    DeviceSpan<const Vec3> cloth_contact_forces{};
    DeviceSpan<const Vec3> fluid_contact_forces{};
    DeviceSpan<const Vec3> rope_contact_forces{};
    std::uint32_t node_count{};
    std::uint32_t surface_vertex_count{};
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
    FluidId fluid{};
    Vec3 initial_velocity{};
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

// Host-only deterministic surface subdivision/thinning. Explicit spacing > 0.
// Transactional: replaces output on success, leaves it unchanged on failure.
[[nodiscard]] Status sample_fluid_source(
    ParticleSourceMesh mesh, std::vector<Vec3> &output) noexcept;

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

// Appends sampled particles to output. This CPU utility needs no CUDA device;
// callers can inspect or downsample before uploading to World.
[[nodiscard]] Status sample_fluid_geometry(
    FluidGeometrySource source, std::vector<FluidParticle> &output) noexcept;

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
    float collision_margin{0.005F};
    std::uint64_t user_data{};
};

// A field targets either one rigid-body mesh or one deforming cloth. Its UVs
// correspond to target vertices. Pixels hold two side bits:
// 1 for the winding/front side and 2 for the back side.
struct PaintFieldOptions {
    RigidBodyId body{};
    TriangleMeshId mesh{};
    ClothId cloth{}; // Set instead of body/mesh for a deforming cloth target.
    DeviceSpan<const Vec2> vertex_uvs{};
    std::uint32_t width{512U};
    std::uint32_t height{512U};
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

struct PaintFieldDeviceView {
    DeviceSpan<const std::uint32_t> pixels{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t revision{};
};

struct FluidDeviceView {
    DeviceSpan<const Vec3> positions{};
    DeviceSpan<const Vec3> velocities{};
    // Solver acceleration excluding the StepOptions gravity term.
    DeviceSpan<const Vec3> accelerations{};
    DeviceSpan<const std::uint32_t> stable_particle_ids{};
    // Short-lived impact/exposed-surface agitation for renderers; [0, 1].
    DeviceSpan<const float> foam{};
    std::uint32_t particle_count{};
    float particle_radius{};
    float support_radius{};
    std::uint64_t revision{};
};

struct RigidBodyDeviceView {
    DeviceSpan<const RigidBodyId> ids{};
    DeviceSpan<const RigidBodyState> states{};
    // Inputs captured at the beginning of the last completed frame. Empty
    // unless WorldOptions::physics_debug is enabled.
    DeviceSpan<const Vec3> applied_forces{};
    DeviceSpan<const Vec3> applied_torques{};
    std::uint64_t revision{};
};

struct ContactEvent {
    FluidId fluid{};
    std::uint32_t stable_particle_id{};
    RigidBodyId rigid_body{};
    Vec3 position{};
    Vec3 normal{}; // Points from the rigid body toward the fluid particle.
    float normal_impulse{};
};

struct ContactDeviceView {
    DeviceSpan<const ContactEvent> events{};
    std::uint32_t event_count{};
    bool overflowed{};
    std::uint64_t frame_index{};
};

struct RigidContactEvent {
    RigidBodyId body{};
    RigidBodyId collider{};
    Vec3 position{};
    Vec3 normal{}; // Points from collider toward body.
    float penetration{};
    float normal_impulse{};
    Vec3 friction_impulse{}; // Tangential impulse applied to body.
};

struct RigidContactDeviceView {
    DeviceSpan<const RigidContactEvent> events{};
    std::uint32_t event_count{};
    std::uint64_t frame_index{};
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
};

struct WorldStatistics {
    std::uint64_t frame_index{};
    std::uint32_t fluid_count{};
    std::uint32_t particle_count{};
    std::uint32_t rigid_body_count{};
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

class FrameToken {
  public:
    FrameToken() noexcept;
    ~FrameToken();
    FrameToken(FrameToken &&) noexcept;
    FrameToken &operator=(FrameToken &&) noexcept;
    FrameToken(const FrameToken &) = delete;
    FrameToken &operator=(const FrameToken &) = delete;

    [[nodiscard]] bool pending() const noexcept;
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] Status wait() noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class World;
};

class World {
  public:
    World() noexcept;
    ~World();
    World(World &&) noexcept;
    World &operator=(World &&) noexcept;
    World(const World &) = delete;
    World &operator=(const World &) = delete;

    [[nodiscard]] static Status create(WorldOptions options, World &output,
                                       cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] Status add_fluid(FluidOptions options,
                                   DeviceSpan<const FluidParticle> initial_particles,
                                   FluidId &output,
                                   cudaStream_t stream = nullptr) noexcept;
    // Samples one authored closed volume and creates a fluid. Continuous
    // inflow is configured separately with add_particle_source.
    [[nodiscard]] Status add_fluid_geometry(
        FluidOptions options, FluidGeometrySource source, FluidId &output,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status remove_fluid(FluidId fluid,
                                      cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status fluid_view(FluidId fluid, FluidDeviceView &output) const noexcept;

    [[nodiscard]] Status add_cloth(ClothOptions options, ClothId &output,
                                   cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status remove_cloth(ClothId cloth) noexcept;
    [[nodiscard]] Status cloth_view(ClothId cloth,
                                    ClothDeviceView &output) const noexcept;

    [[nodiscard]] Status add_soft_body(SoftBodyOptions options,
                                       SoftBodyId &output,
                                       cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status remove_soft_body(SoftBodyId soft_body) noexcept;
    [[nodiscard]] Status soft_body_view(
        SoftBodyId soft_body, SoftBodyDeviceView &output) const noexcept;

    [[nodiscard]] Status add_rope(RopeOptions options, RopeId &output) noexcept;
    [[nodiscard]] Status remove_rope(RopeId rope) noexcept;
    [[nodiscard]] Status rope_view(RopeId rope, RopeDeviceView &output) const noexcept;

    [[nodiscard]] Status add_fluid_rope_coupling(
        FluidRopeCouplingOptions options, FluidRopeCouplingId &output) noexcept;
    [[nodiscard]] Status update_fluid_rope_coupling(
        FluidRopeCouplingId coupling, FluidRopeCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_fluid_rope_coupling(
        FluidRopeCouplingId coupling) noexcept;

    [[nodiscard]] Status add_rope_soft_body_coupling(
        RopeSoftBodyCouplingOptions options,
        RopeSoftBodyCouplingId &output) noexcept;
    [[nodiscard]] Status update_rope_soft_body_coupling(
        RopeSoftBodyCouplingId coupling,
        RopeSoftBodyCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_rope_soft_body_coupling(
        RopeSoftBodyCouplingId coupling) noexcept;

    [[nodiscard]] Status add_rope_cloth_coupling(
        RopeClothCouplingOptions options, RopeClothCouplingId &output) noexcept;
    [[nodiscard]] Status update_rope_cloth_coupling(
        RopeClothCouplingId coupling, RopeClothCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_rope_cloth_coupling(
        RopeClothCouplingId coupling) noexcept;

    [[nodiscard]] Status add_fluid_cloth_coupling(
        FluidClothCouplingOptions options,
        FluidClothCouplingId &output) noexcept;
    [[nodiscard]] Status update_fluid_cloth_coupling(
        FluidClothCouplingId coupling,
        FluidClothCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_fluid_cloth_coupling(
        FluidClothCouplingId coupling) noexcept;

    [[nodiscard]] Status add_soft_body_cloth_coupling(
        SoftBodyClothCouplingOptions options,
        SoftBodyClothCouplingId &output) noexcept;
    // Endpoints are immutable; remove and add to bind another pair.
    [[nodiscard]] Status update_soft_body_cloth_coupling(
        SoftBodyClothCouplingId coupling,
        SoftBodyClothCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_soft_body_cloth_coupling(
        SoftBodyClothCouplingId coupling) noexcept;

    [[nodiscard]] Status add_fluid_soft_body_coupling(
        FluidSoftBodyCouplingOptions options,
        FluidSoftBodyCouplingId &output) noexcept;
    // Endpoints are immutable. Remove/re-add to change either endpoint.
    [[nodiscard]] Status update_fluid_soft_body_coupling(
        FluidSoftBodyCouplingId coupling,
        FluidSoftBodyCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_fluid_soft_body_coupling(
        FluidSoftBodyCouplingId coupling) noexcept;

    [[nodiscard]] Status add_particle_source(
        ParticleSourceMesh mesh, ParticleSourceOptions options,
        ParticleSourceId &output) noexcept;
    // Mesh/spacing and destination fluid are immutable; remove/re-add to change.
    [[nodiscard]] Status update_particle_source(
        ParticleSourceId plane, ParticleSourceOptions options) noexcept;
    [[nodiscard]] Status remove_particle_source(
        ParticleSourceId plane) noexcept;
    [[nodiscard]] Status add_particle_destroy_plane(
        ParticleDestroyPlaneOptions options, ParticleDestroyPlaneId &output) noexcept;
    [[nodiscard]] Status update_particle_destroy_plane(
        ParticleDestroyPlaneId plane, ParticleDestroyPlaneOptions options) noexcept;
    [[nodiscard]] Status remove_particle_destroy_plane(
        ParticleDestroyPlaneId plane) noexcept;

    [[nodiscard]] Status add_rigid_body(RigidBodyOptions options,
                                        RigidBodyId &output) noexcept;
    [[nodiscard]] Status remove_rigid_body(RigidBodyId body) noexcept;
    [[nodiscard]] Status set_rigid_body_state(RigidBodyId body,
                                              RigidBodyState state) noexcept;
    [[nodiscard]] Status set_kinematic_target(RigidBodyId body,
                                              RigidBodyState target) noexcept;
    [[nodiscard]] Status apply_force(RigidBodyId body, Vec3 force,
                                     Vec3 world_point) noexcept;
    [[nodiscard]] Status apply_impulse(RigidBodyId body, Vec3 impulse,
                                       Vec3 world_point) noexcept;
    [[nodiscard]] Status rigid_body_view(RigidBodyDeviceView &output) const noexcept;
    // Explicit synchronous readback for gameplay code that needs one body.
    [[nodiscard]] Status read_rigid_body_state(
        RigidBodyId body, RigidBodyState &output,
        cudaStream_t stream = nullptr) const noexcept;

    // Copies an indexed two-sided triangle soup into World-owned CUDA memory.
    // Triangles need not form a closed, manifold, or consistently wound mesh.
    [[nodiscard]] Status add_triangle_mesh(
        DeviceSpan<const Vec3> vertices,
        DeviceSpan<const std::uint32_t> triangle_indices,
        TriangleMeshId &output, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status remove_triangle_mesh(TriangleMeshId mesh) noexcept;

    [[nodiscard]] Status add_paint_field(
        PaintFieldOptions options, PaintFieldId &output,
        cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status remove_paint_field(PaintFieldId field) noexcept;
    [[nodiscard]] Status clear_paint_field(
        PaintFieldId field, cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status paint_field_view(
        PaintFieldId field, PaintFieldDeviceView &output) const noexcept;
    [[nodiscard]] Status add_paint_rule(
        PaintRuleOptions options, PaintRuleId &output) noexcept;
    [[nodiscard]] Status remove_paint_rule(PaintRuleId rule) noexcept;

    // Only one frame may be in flight per World in the initial release.
    [[nodiscard]] Status step_async(StepOptions options, FrameToken &completion,
                                    cudaStream_t stream = nullptr) noexcept;
    [[nodiscard]] Status step(StepOptions options,
                              cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] ContactDeviceView contacts() const noexcept;
    [[nodiscard]] RigidContactDeviceView rigid_contacts() const noexcept;
    // Borrow the latest host debug frame, or deep-copy the chronological ring.
    // Both require physics_debug.frame_capacity > 0 at World creation.
    [[nodiscard]] Status physics_debug_frame(
        PhysicsDebugFrameView &output) const noexcept;
    [[nodiscard]] Status copy_physics_debug_capture(
        PhysicsDebugCapture &output) const noexcept;
    [[nodiscard]] Status collect_step_timings(
        WorldStepTimings &output) const noexcept;
    [[nodiscard]] Status collect_statistics(WorldStatistics &output,
                                            cudaStream_t stream = nullptr) const noexcept;
    [[nodiscard]] int device_ordinal() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater
