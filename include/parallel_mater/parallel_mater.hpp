// SPDX-License-Identifier: MIT
#pragma once

// Public CUDA physics API. Implementation milestones are tracked in
// docs/ROADMAP.md; a declaration may precede its solver implementation.

#include <cuda_runtime_api.h>

#include <parallel_mater/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace parallel_mater {

template <typename T> struct DeviceSpan {
    T *data{};
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

struct SmokeDeviceView {
    DeviceSpan<const Vec3> positions{};
    DeviceSpan<const Vec3> velocities{};
    DeviceSpan<const float> ages{};
    DeviceSpan<const float> number_densities{};
    DeviceSpan<const float> pressures{};
    DeviceSpan<const Vec3> vorticities{};
    // Occupied ring slots; ages >= lifetime are expired and should not draw.
    std::uint32_t particle_count{};
    float lifetime{};
    float particle_radius{};
    std::uint64_t revision{};
    // Optional air field, indexed x + resolution *
    // (y + vertical_resolution*z).
    DeviceSpan<const Vec3> grid_velocity{};
    DeviceSpan<const float> grid_pressure{};
    DeviceSpan<const float> grid_density{};
    // Density-weighted thermal acceleration deposited by smoke tracers.
    // Divide by grid_density where it is non-zero to recover the local mean.
    DeviceSpan<const float> grid_temperature{};
    DeviceSpan<const std::uint32_t> grid_solid{};
    DeviceSpan<const Vec3> grid_vorticity{};
    DeviceSpan<const float> grid_divergence{};
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

// Host-side preparation; no CUDA device is needed. Refine long surface edges
// to lattice resolution (maximum edge 1.5 * spacing allows triangle diagonals),
// preserve the closed input shape/winding, and connect every new physical
// surface node to the same volumetric spring lattice. Output is transactional.
[[nodiscard]] Status build_soft_body_geometry(
    SoftBodyGeometrySource source, SoftBodyGeometry &output) noexcept;

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
    DeviceSpan<const Vec2> vertex_uvs{};
    std::uint32_t width{512U};
    std::uint32_t height{512U};
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
    DeviceSpan<const float> temperatures{}; // degrees Celsius
    std::uint32_t particle_count{};
    float particle_radius{};
    float support_radius{};
    std::uint64_t revision{};
};

struct RigidBodyDeviceView {
    DeviceSpan<const RigidBodyId> ids{};
    DeviceSpan<const RigidBodyState> states{};
    // State at the start of the latest World::step call. Renderers may blend
    // previous_states toward states using a fixed-timestep accumulator.
    DeviceSpan<const RigidBodyState> previous_states{};
    // Inputs captured at the beginning of the last completed frame. Empty
    // unless WorldOptions::physics_debug is enabled.
    DeviceSpan<const Vec3> applied_forces{};
    DeviceSpan<const Vec3> applied_torques{};
    std::uint64_t revision{};
};

struct ContactDeviceView {
    DeviceSpan<const ContactEvent> events{};
    std::uint32_t event_count{};
    bool overflowed{};
    std::uint64_t frame_index{};
};

struct RigidContactDeviceView {
    DeviceSpan<const RigidContactEvent> events{};
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
    // Applies one acceleration at each body's center of mass without state
    // readback. Validation is transactional: no body changes on failure.
    [[nodiscard]] Status apply_central_acceleration(
        HostSpan<RigidBodyId> bodies, Vec3 acceleration) noexcept;
    [[nodiscard]] Status apply_impulse(RigidBodyId body, Vec3 impulse,
                                       Vec3 world_point) noexcept;
    [[nodiscard]] Status rigid_body_view(RigidBodyDeviceView &output) const noexcept;
    // Explicit synchronous readback for gameplay code that needs one body.
    [[nodiscard]] Status read_rigid_body_state(
        RigidBodyId body, RigidBodyState &output,
        cudaStream_t stream = nullptr) const noexcept;
    // Synchronously polls the latest completed state. Results include every
    // live rigid body and fluid; callers can filter for a specific goal body.
    [[nodiscard]] Status query_hit_box(
        HitBox box, HitBoxResult &output,
        cudaStream_t stream = nullptr) const noexcept;

    [[nodiscard]] Status add_rigid_constraint(
        RigidConstraintOptions options, RigidConstraintId &output) noexcept;
    [[nodiscard]] Status update_rigid_constraint(
        RigidConstraintId constraint, RigidConstraintOptions options) noexcept;
    [[nodiscard]] Status remove_rigid_constraint(
        RigidConstraintId constraint) noexcept;
    [[nodiscard]] Status read_rigid_constraint_state(
        RigidConstraintId constraint, RigidConstraintState &output,
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
