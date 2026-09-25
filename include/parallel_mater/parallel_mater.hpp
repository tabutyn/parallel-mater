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

struct ParticleSpawnPlaneId {
    std::uint32_t index{};
    std::uint32_t generation{};

    [[nodiscard]] friend constexpr bool operator==(
        ParticleSpawnPlaneId left, ParticleSpawnPlaneId right) noexcept {
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

struct WorldOptions {
    std::uint32_t fluid_capacity{1U};
    std::uint32_t rigid_body_capacity{64U};
    std::uint32_t triangle_mesh_capacity{16U};
    std::uint32_t particle_spawn_plane_capacity{8U};
    std::uint32_t particle_destroy_plane_capacity{8U};
    std::uint32_t paint_field_capacity{8U};
    std::uint32_t paint_rule_capacity{8U};
    // Maximum diagnostic contact events retained for a requested frame.
    std::uint32_t contact_capacity{65'536U};
    bool deterministic{true};
    std::uint32_t cloth_capacity{1U};
};

struct StepOptions {
    float timestep{1.0F / 60.0F};
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
struct ClothOptions {
    HostSpan<Vec3> vertices{};
    HostSpan<std::uint32_t> triangle_indices{};
    HostSpan<float> inverse_masses{};
    float vertex_mass{0.02F};
    float thickness{0.025F};
    float stretch_compliance{1.0e-6F};
    float bending_compliance{0.1F};
    float velocity_damping{0.02F};
    std::uint32_t solver_iterations{8U};
};

struct ClothDeviceView {
    DeviceSpan<const Vec3> positions{};
    DeviceSpan<const std::uint32_t> triangle_indices{};
    std::uint32_t vertex_count{};
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

struct ParticleSpawnPlaneOptions {
    FluidId fluid{};
    ParticlePlane plane{};
    float particles_per_second{};
    Vec3 initial_velocity{};
    std::uint32_t sequence_seed{};
    bool enabled{true};
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

// A field is owned per rigid-body instance, even if bodies share geometry.
// Its UVs correspond to mesh vertices; the mesh may differ from the body's
// collision mesh (for authored collision proxies). Pixels hold two side bits:
// 1 for the winding/front side and 2 for the back side.
struct PaintFieldOptions {
    RigidBodyId body{};
    TriangleMeshId mesh{};
    DeviceSpan<const Vec2> vertex_uvs{};
    std::uint32_t width{512U};
    std::uint32_t height{512U};
};

struct PaintRuleOptions {
    FluidId source{};
    PaintFieldId target{};
    // Additional reach beyond the source particle radius.
    float reach{0.025F};
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
    KernelTiming fluid_contact_events{};
    KernelTiming fluid_outflow_compaction{};
    float total_gpu_milliseconds{};
    KernelTiming cloth_prediction{};
    KernelTiming cloth_constraints{};
    KernelTiming cloth_contacts{};
};

struct WorldStatistics {
    std::uint64_t frame_index{};
    std::uint32_t fluid_count{};
    std::uint32_t particle_count{};
    std::uint32_t rigid_body_count{};
    std::uint32_t triangle_mesh_count{};
    std::uint32_t contact_count{};
    std::uint32_t contact_overflow_count{};
    std::uint64_t emitted_particle_count{};
    std::uint64_t destroyed_particle_count{};
    std::uint64_t spawn_capacity_miss_count{};
    std::size_t allocated_bytes{};
    std::uint32_t cloth_count{};
    std::uint32_t cloth_vertex_count{};
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
    // inflow is configured separately with add_particle_spawn_plane.
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

    [[nodiscard]] Status add_particle_spawn_plane(
        ParticleSpawnPlaneOptions options, ParticleSpawnPlaneId &output) noexcept;
    [[nodiscard]] Status update_particle_spawn_plane(
        ParticleSpawnPlaneId plane, ParticleSpawnPlaneOptions options) noexcept;
    [[nodiscard]] Status remove_particle_spawn_plane(
        ParticleSpawnPlaneId plane) noexcept;
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
