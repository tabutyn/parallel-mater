// SPDX-License-Identifier: MIT
#pragma once

// Public D3D12 physics API. Implementation milestones are tracked in
// docs/ROADMAP.md; a declaration may precede its solver implementation.

#include <parallel_mater/types.hpp>

#include <memory>
#include <vector>

namespace parallel_mater::d3d12 {

using ::parallel_mater::HostSpan;
using ::parallel_mater::Quaternion;
using ::parallel_mater::Vec2;
using ::parallel_mater::Vec3;
using ::parallel_mater::ClothId;
using ::parallel_mater::ClothBond;
using ::parallel_mater::ClothOptions;
using ::parallel_mater::FluidClothCouplingId;
using ::parallel_mater::FluidClothCouplingOptions;
using ::parallel_mater::FluidId;
using ::parallel_mater::FluidRopeCouplingId;
using ::parallel_mater::FluidRopeCouplingOptions;
using ::parallel_mater::FluidSmokeCouplingId;
using ::parallel_mater::FluidSoftBodyCouplingId;
using ::parallel_mater::FluidSoftBodyCouplingOptions;
using ::parallel_mater::PaintFieldId;
using ::parallel_mater::PaintRuleId;
using ::parallel_mater::ParticleDestroyPlaneId;
using ::parallel_mater::ParticleSourceId;
using ::parallel_mater::PhysicsDebugOptions;
using ::parallel_mater::RigidBodyId;
using ::parallel_mater::RigidConstraintId;
using ::parallel_mater::RopeClothCouplingId;
using ::parallel_mater::RopeClothCouplingOptions;
using ::parallel_mater::RopeId;
using ::parallel_mater::RopeAttachment;
using ::parallel_mater::RopeOptions;
using ::parallel_mater::RopeSoftBodyCouplingId;
using ::parallel_mater::RopeSoftBodyCouplingOptions;
using ::parallel_mater::SmokeClothCouplingId;
using ::parallel_mater::SmokeClothCouplingOptions;
using ::parallel_mater::SmokeId;
using ::parallel_mater::SmokeOptions;
using ::parallel_mater::SmokeRigidCouplingId;
using ::parallel_mater::SmokeRigidCouplingOptions;
using ::parallel_mater::SmokeRopeCouplingId;
using ::parallel_mater::SmokeRopeCouplingOptions;
using ::parallel_mater::SmokeSoftBodyCouplingId;
using ::parallel_mater::SmokeSoftBodyCouplingOptions;
using ::parallel_mater::SoftBodyClothCouplingId;
using ::parallel_mater::SoftBodyClothCouplingOptions;
using ::parallel_mater::SoftBodyId;
using ::parallel_mater::SoftBodyBond;
using ::parallel_mater::SoftBodyGeometry;
using ::parallel_mater::SoftBodyGeometrySource;
using ::parallel_mater::SoftBodyOptions;
using ::parallel_mater::SoftBodySurfaceBinding;
using ::parallel_mater::SoftBodySurfaceSource;
using ::parallel_mater::TriangleMeshId;
using ::parallel_mater::FluidParticle;
using ::parallel_mater::FluidGeometrySource;
using ::parallel_mater::FluidOptions;
using ::parallel_mater::FluidSmokeCouplingOptions;
using ::parallel_mater::MotionType;
using ::parallel_mater::PaintRuleOptions;
using ::parallel_mater::ParticleDestroyPlaneOptions;
using ::parallel_mater::ParticlePlane;
using ::parallel_mater::ParticleSourceMesh;
using ::parallel_mater::ParticleSourceOptions;
using ::parallel_mater::CrossingDirection;
using ::parallel_mater::ContactEvent;
using ::parallel_mater::HitBox;
using ::parallel_mater::HitBoxParticle;
using ::parallel_mater::HitBoxResult;
using ::parallel_mater::KernelTiming;
using ::parallel_mater::PhysicsDebugCapture;
using ::parallel_mater::PhysicsDebugClothSample;
using ::parallel_mater::PhysicsDebugFluidSample;
using ::parallel_mater::PhysicsDebugFrame;
using ::parallel_mater::PhysicsDebugFrameView;
using ::parallel_mater::PhysicsDebugRigidSample;
using ::parallel_mater::PhysicsDebugRopeSample;
using ::parallel_mater::PhysicsDebugSoftBodySample;
using ::parallel_mater::RigidBodyOptions;
using ::parallel_mater::RigidBodyState;
using ::parallel_mater::RigidConstraintLimitOptions;
using ::parallel_mater::RigidConstraintMotorOptions;
using ::parallel_mater::RigidConstraintOptions;
using ::parallel_mater::RigidConstraintSpringOptions;
using ::parallel_mater::RigidConstraintState;
using ::parallel_mater::RigidConstraintType;
using ::parallel_mater::RigidContactEvent;
using ::parallel_mater::rigid_constraint_all_axes;
using ::parallel_mater::rigid_constraint_axis_x;
using ::parallel_mater::rigid_constraint_axis_y;
using ::parallel_mater::rigid_constraint_axis_z;
using ::parallel_mater::StepOptions;
using ::parallel_mater::WorldOptions;
using ::parallel_mater::WorldStatistics;
using ::parallel_mater::WorldStepTimings;

// Opaque COM object pointers. World AddRefs every non-null object passed to
// create(). A null device selects the first qualifying hardware adapter; a null
// queue creates a D3D12 direct queue on the selected device.
struct NativeContext {
    void *device{};       // ID3D12Device
    void *direct_queue{}; // ID3D12CommandQueue, type DIRECT
};

template <typename T> struct BufferSpan {
    void *resource{}; // Borrowed ID3D12Resource; inputs retained only while used.
    std::uint64_t byte_offset{};
    std::uint64_t size{};

    [[nodiscard]] constexpr bool empty() const noexcept { return size == 0U; }
};

enum class StatusCode : std::uint8_t {
    success,
    invalid_argument,
    not_supported,
    invalid_handle,
    capacity_exceeded,
    busy,
    out_of_memory,
    unsupported_adapter,
    device_removed,
    d3d12_failure,
    internal_error,
};

struct Status {
    StatusCode code{StatusCode::success};
    std::int64_t hresult{};
    const char *message{};

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == StatusCode::success;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

struct RopeDeviceView {
    BufferSpan<const Vec3> positions{};
    BufferSpan<const Vec3> velocities{};
    BufferSpan<const Vec3> constraint_forces{};
    BufferSpan<const Vec3> contact_forces{};
    BufferSpan<const Vec3> fluid_contact_forces{};
    BufferSpan<const Vec3> soft_body_contact_forces{};
    BufferSpan<const float> rest_lengths{};
    float radius{};
};

[[nodiscard]] Status sample_rope_centerline(
    HostSpan<const Vec3> centerline, float spacing,
    std::vector<Vec3> &output) noexcept;

struct SmokeDeviceView {
    BufferSpan<const Vec3> positions{};
    BufferSpan<const Vec3> velocities{};
    BufferSpan<const float> ages{};
    BufferSpan<const float> number_densities{};
    BufferSpan<const float> pressures{};
    BufferSpan<const Vec3> vorticities{};
    // Occupied ring slots; ages >= lifetime are expired and should not draw.
    std::uint32_t particle_count{};
    float lifetime{};
    float particle_radius{};
    std::uint64_t revision{};
    // Optional air field, indexed x + resolution *
    // (y + vertical_resolution*z).
    BufferSpan<const Vec3> grid_velocity{};
    BufferSpan<const float> grid_pressure{};
    BufferSpan<const float> grid_density{};
    // Density-weighted thermal acceleration deposited by smoke tracers.
    // Divide by grid_density where it is non-zero to recover the local mean.
    BufferSpan<const float> grid_temperature{};
    BufferSpan<const std::uint32_t> grid_solid{};
    BufferSpan<const Vec3> grid_vorticity{};
    BufferSpan<const float> grid_divergence{};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
    Vec3 grid_minimum{};
    float grid_spacing{};
    // Infinity-norm pressure residual divided by the pre-projection RHS norm.
    float grid_pressure_relative_residual{};
};

struct ClothDeviceView {
    // Physical nodes and current connectivity. Tearing can append nodes and
    // reindex faces at the next frame boundary; reacquire this view each frame.
    BufferSpan<const Vec3> positions{};
    BufferSpan<const Vec3> velocities{};
    BufferSpan<const std::uint32_t> triangle_indices{};
    std::uint32_t vertex_count{};
    // Tearable cloth: authored vertex for each physical node, and split masses.
    // New nodes inherit velocity and share their source's original mass.
    BufferSpan<const std::uint32_t> vertex_source_indices{};
    BufferSpan<const float> inverse_masses{};
    // For tearable cloth, every triangle owns three surface corners. The
    // triangle count stays fixed as bonds fail; source indices preserve UVs.
    BufferSpan<const Vec3> surface_positions{};
    BufferSpan<const std::uint32_t> surface_triangle_indices{};
    BufferSpan<const std::uint32_t> surface_source_indices{};
    BufferSpan<const ClothBond> bonds{};
    BufferSpan<const std::uint8_t> active_bonds{};
    // Last-substep forces applied at each physical node. These diagnostics
    // remain available without enabling contact event collection, allowing
    // renderers and tools to inspect cloth coupling through the public API.
    BufferSpan<const Vec3> rigid_contact_forces{};
    BufferSpan<const Vec3> fluid_contact_forces{};
    BufferSpan<const Vec3> soft_body_contact_forces{};
    BufferSpan<const Vec3> rope_contact_forces{};
};

// Host-side preparation; no CUDA device is needed. Refine long surface edges
// to lattice resolution (maximum edge 1.5 * spacing allows triangle diagonals),
// preserve the closed input shape/winding, and connect every new physical
// surface node to the same volumetric spring lattice. Output is transactional.
[[nodiscard]] Status build_soft_body_geometry(
    SoftBodyGeometrySource source, SoftBodyGeometry &output) noexcept;

struct SoftBodyDeviceView {
    BufferSpan<const Vec3> positions{};
    BufferSpan<const Vec3> velocities{};
    BufferSpan<const SoftBodyBond> bonds{};
    BufferSpan<const Vec3> surface_positions{};
    BufferSpan<const std::uint32_t> surface_triangle_indices{};
    BufferSpan<const Vec3> rigid_contact_forces{};
    BufferSpan<const Vec3> cloth_contact_forces{};
    BufferSpan<const Vec3> fluid_contact_forces{};
    BufferSpan<const Vec3> rope_contact_forces{};
    std::uint32_t node_count{};
    std::uint32_t surface_vertex_count{};
};

// Host-only deterministic surface subdivision/thinning. Explicit spacing > 0.
// Transactional: replaces output on success, leaves it unchanged on failure.
[[nodiscard]] Status sample_fluid_source(
    ParticleSourceMesh mesh, std::vector<Vec3> &output) noexcept;

// Appends sampled particles to output. This CPU utility needs no CUDA device;
// callers can inspect or downsample before uploading to World.
[[nodiscard]] Status sample_fluid_geometry(
    FluidGeometrySource source, std::vector<FluidParticle> &output) noexcept;

// A field targets either one rigid-body mesh or one deforming cloth. Its UVs
// correspond to target vertices. Pixels hold two side bits:
// 1 for the winding/front side and 2 for the back side.
struct PaintFieldOptions {
    RigidBodyId body{};
    TriangleMeshId mesh{};
    ClothId cloth{}; // Set instead of body/mesh for a deforming cloth target.
    BufferSpan<const Vec2> vertex_uvs{};
    std::uint32_t width{512U};
    std::uint32_t height{512U};
};

struct PaintFieldHostOptions {
    RigidBodyId body{};
    TriangleMeshId mesh{};
    ClothId cloth{};
    HostSpan<const Vec2> vertex_uvs{};
    std::uint32_t width{512U};
    std::uint32_t height{512U};
};

struct PaintFieldDeviceView {
    BufferSpan<const std::uint32_t> pixels{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t revision{};
};

struct FluidDeviceView {
    BufferSpan<const Vec3> positions{};
    BufferSpan<const Vec3> velocities{};
    // Solver acceleration excluding the StepOptions gravity term.
    BufferSpan<const Vec3> accelerations{};
    BufferSpan<const std::uint32_t> stable_particle_ids{};
    // Short-lived impact/exposed-surface agitation for renderers; [0, 1].
    BufferSpan<const float> foam{};
    BufferSpan<const float> temperatures{}; // degrees Celsius
    std::uint32_t particle_count{};
    float particle_radius{};
    float support_radius{};
    std::uint64_t revision{};
};

struct RigidBodyDeviceView {
    BufferSpan<const RigidBodyId> ids{};
    BufferSpan<const RigidBodyState> states{};
    // State at the start of the latest World::step call. Renderers may blend
    // previous_states toward states using a fixed-timestep accumulator.
    BufferSpan<const RigidBodyState> previous_states{};
    // Inputs captured at the beginning of the last completed frame. Empty
    // unless WorldOptions::physics_debug is enabled.
    BufferSpan<const Vec3> applied_forces{};
    BufferSpan<const Vec3> applied_torques{};
    std::uint64_t revision{};
};

struct ContactDeviceView {
    BufferSpan<const ContactEvent> events{};
    std::uint32_t event_count{};
    bool overflowed{};
    std::uint64_t frame_index{};
};

struct RigidContactDeviceView {
    BufferSpan<const RigidContactEvent> events{};
    std::uint32_t event_count{};
    std::uint64_t frame_index{};
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

    [[nodiscard]] static Status create(WorldOptions options, World &output) noexcept;
    [[nodiscard]] static Status create(WorldOptions options,
                                       NativeContext context,
                                       World &output) noexcept;

    [[nodiscard]] Status add_fluid(FluidOptions options,
                                   BufferSpan<const FluidParticle> initial_particles,
                                   FluidId &output) noexcept;
    [[nodiscard]] Status add_fluid(FluidOptions options,
                                   HostSpan<const FluidParticle> initial_particles,
                                   FluidId &output) noexcept;
    // Samples one authored closed volume and creates a fluid. Continuous
    // inflow is configured separately with add_particle_source.
    [[nodiscard]] Status add_fluid_geometry(
        FluidOptions options, FluidGeometrySource source, FluidId &output) noexcept;
    [[nodiscard]] Status remove_fluid(FluidId fluid) noexcept;
    [[nodiscard]] Status fluid_view(FluidId fluid, FluidDeviceView &output) const noexcept;
    [[nodiscard]] Status add_smoke(SmokeOptions options, SmokeId &output) noexcept;
    [[nodiscard]] Status remove_smoke(SmokeId smoke) noexcept;
    [[nodiscard]] Status smoke_view(SmokeId smoke, SmokeDeviceView &output) const noexcept;
    [[nodiscard]] Status add_fluid_smoke_coupling(
        FluidSmokeCouplingOptions options, FluidSmokeCouplingId &output) noexcept;
    [[nodiscard]] Status remove_fluid_smoke_coupling(
        FluidSmokeCouplingId coupling) noexcept;
    [[nodiscard]] Status add_smoke_soft_body_coupling(
        SmokeSoftBodyCouplingOptions options,
        SmokeSoftBodyCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_soft_body_coupling(
        SmokeSoftBodyCouplingId coupling) noexcept;
    [[nodiscard]] Status add_smoke_cloth_coupling(
        SmokeClothCouplingOptions options,
        SmokeClothCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_cloth_coupling(
        SmokeClothCouplingId coupling) noexcept;
    [[nodiscard]] Status add_smoke_rope_coupling(
        SmokeRopeCouplingOptions options,
        SmokeRopeCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_rope_coupling(
        SmokeRopeCouplingId coupling) noexcept;
    [[nodiscard]] Status add_smoke_rigid_coupling(
        SmokeRigidCouplingOptions options,
        SmokeRigidCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_rigid_coupling(
        SmokeRigidCouplingId coupling) noexcept;

    [[nodiscard]] Status add_cloth(ClothOptions options, ClothId &output) noexcept;
    [[nodiscard]] Status remove_cloth(ClothId cloth) noexcept;
    [[nodiscard]] Status cloth_view(ClothId cloth,
                                    ClothDeviceView &output) const noexcept;

    [[nodiscard]] Status add_soft_body(SoftBodyOptions options,
                                       SoftBodyId &output) noexcept;
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
    // Applies one acceleration at each body's center of mass without state
    // readback. Validation is transactional: no body changes on failure.
    [[nodiscard]] Status apply_central_acceleration(
        HostSpan<RigidBodyId> bodies, Vec3 acceleration) noexcept;
    [[nodiscard]] Status apply_impulse(RigidBodyId body, Vec3 impulse,
                                       Vec3 world_point) noexcept;
    [[nodiscard]] Status rigid_body_view(RigidBodyDeviceView &output) const noexcept;
    // Explicit synchronous readback for gameplay code that needs one body.
    [[nodiscard]] Status read_rigid_body_state(
        RigidBodyId body, RigidBodyState &output) const noexcept;
    // Synchronously polls the latest completed state. Results include every
    // live rigid body and fluid; callers can filter for a specific goal body.
    [[nodiscard]] Status query_hit_box(
        HitBox box, HitBoxResult &output) const noexcept;

    [[nodiscard]] Status add_rigid_constraint(
        RigidConstraintOptions options, RigidConstraintId &output) noexcept;
    [[nodiscard]] Status update_rigid_constraint(
        RigidConstraintId constraint, RigidConstraintOptions options) noexcept;
    [[nodiscard]] Status remove_rigid_constraint(
        RigidConstraintId constraint) noexcept;
    [[nodiscard]] Status read_rigid_constraint_state(
        RigidConstraintId constraint, RigidConstraintState &output) const noexcept;

    // Copies an indexed two-sided triangle soup into World-owned D3D12 memory.
    // Triangles need not form a closed, manifold, or consistently wound mesh.
    [[nodiscard]] Status add_triangle_mesh(
        BufferSpan<const Vec3> vertices,
        BufferSpan<const std::uint32_t> triangle_indices,
        TriangleMeshId &output) noexcept;
    [[nodiscard]] Status add_triangle_mesh(
        HostSpan<const Vec3> vertices,
        HostSpan<const std::uint32_t> triangle_indices,
        TriangleMeshId &output) noexcept;
    [[nodiscard]] Status remove_triangle_mesh(TriangleMeshId mesh) noexcept;

    [[nodiscard]] Status add_paint_field(
        PaintFieldOptions options, PaintFieldId &output) noexcept;
    [[nodiscard]] Status add_paint_field(
        PaintFieldHostOptions options, PaintFieldId &output) noexcept;
    [[nodiscard]] Status remove_paint_field(PaintFieldId field) noexcept;
    [[nodiscard]] Status clear_paint_field(
        PaintFieldId field) noexcept;
    [[nodiscard]] Status paint_field_view(
        PaintFieldId field, PaintFieldDeviceView &output) const noexcept;
    [[nodiscard]] Status add_paint_rule(
        PaintRuleOptions options, PaintRuleId &output) noexcept;
    [[nodiscard]] Status remove_paint_rule(PaintRuleId rule) noexcept;

    // Only one frame may be in flight per World in the initial release.
    [[nodiscard]] Status step_async(StepOptions options, FrameToken &completion) noexcept;
    [[nodiscard]] Status step(StepOptions options) noexcept;

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
    [[nodiscard]] Status collect_statistics(WorldStatistics &output) const noexcept;
    [[nodiscard]] NativeContext native_context() const noexcept;

  private:
    [[nodiscard]] void *systems_implementation() noexcept;
    [[nodiscard]] const void *systems_implementation() const noexcept;
    [[nodiscard]] bool systems_mutation_allowed() const noexcept;
    [[nodiscard]] Status systems_finish_mutation(Status status) noexcept;
    [[nodiscard]] Status systems_reserve_debug_samples(
        std::uint64_t fluid_particles, std::uint64_t cloth_vertices,
        std::uint64_t soft_body_nodes,
        std::uint64_t rope_nodes) noexcept;
    [[nodiscard]] std::uint64_t systems_revision() const noexcept;
    [[nodiscard]] bool systems_rigid_body_valid(RigidBodyId id) const noexcept;
    [[nodiscard]] Status systems_validate_rope_rest(
        HostSpan<const Vec3> nodes, RopeAttachment first,
        RopeAttachment last, std::uint32_t &first_contact_skip,
        std::uint32_t &last_contact_skip) const noexcept;
    [[nodiscard]] std::uint32_t systems_triangle_mesh_vertex_count(
        TriangleMeshId id) const noexcept;
    [[nodiscard]] Status copy_d3d12_buffer_to_host(
        void *buffer, std::uint64_t byte_offset, std::uint64_t byte_count,
        void *destination) noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::d3d12
