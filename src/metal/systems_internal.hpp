// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/metal.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace parallel_mater::metal::detail {

enum class MetalTimingStage : std::uint32_t {
    none,
    rigid_integration,
    rigid_world_bounds,
    rigid_pair_filter,
    rigid_pair_compaction,
    rigid_contact_evaluation,
    rigid_contact_solve,
    rigid_input_clear,
    fluid_spawn,
    fluid_neighbor_sort,
    fluid_neighbor_forces,
    fluid_integration,
    fluid_static_contacts,
    fluid_cloth_contacts,
    fluid_outflow_compaction,
    fluid_smoke_exchange,
    cloth_prediction,
    cloth_constraints,
    cloth_contacts,
    soft_body_prediction,
    soft_body_constraints,
    soft_body_contacts,
    soft_body_cloth_contacts,
    fluid_soft_body_contacts,
    fluid_rope_contacts,
    rope_solve,
    rope_soft_body_contacts,
    smoke_grid,
    smoke_advection,
    smoke_emission,
};

struct MetalTimingRecord {
    MetalTimingStage stage{MetalTimingStage::none};
    std::uint32_t begin_index{};
    std::uint32_t end_index{};
    std::uint32_t launch_count{};
};

struct MetalTimingContext {
    void *counter_heap{};
    MetalTimingRecord *records{};
    std::uint32_t record_capacity{};
    std::uint32_t record_count{};
    std::uint32_t next_index{1U};
    std::uint32_t final_index{};
    bool enabled{};
    bool overflowed{};
};

enum class MetalSystemPhase : std::uint8_t {
    frame_start,
    deformable_substep,
    frame_end,
};

class MetalSystems {
  public:
    MetalSystems() noexcept;
    ~MetalSystems();
    MetalSystems(MetalSystems &&) noexcept;
    MetalSystems &operator=(MetalSystems &&) noexcept;
    MetalSystems(const MetalSystems &) = delete;
    MetalSystems &operator=(const MetalSystems &) = delete;

    [[nodiscard]] static Status create(
        WorldOptions options, void *device, void *library, void *residency_set,
        MetalSystems &output) noexcept;

    [[nodiscard]] Status add_fluid(
        FluidOptions options, HostSpan<const FluidParticle> particles,
        FluidId &output) noexcept;
    [[nodiscard]] Status remove_fluid(FluidId id) noexcept;
    [[nodiscard]] Status fluid_view(FluidId id,
                                    FluidDeviceView &output) const noexcept;
    [[nodiscard]] Status append_hit_box_particles(
        HitBox box, std::vector<HitBoxParticle> &output) const noexcept;

    [[nodiscard]] Status add_smoke(SmokeOptions options,
                                   SmokeId &output) noexcept;
    [[nodiscard]] Status remove_smoke(SmokeId id) noexcept;
    [[nodiscard]] Status smoke_view(SmokeId id,
                                    SmokeDeviceView &output) const noexcept;

    [[nodiscard]] Status add_cloth(ClothOptions options,
                                   ClothId &output) noexcept;
    [[nodiscard]] Status remove_cloth(ClothId id) noexcept;
    [[nodiscard]] Status cloth_view(ClothId id,
                                    ClothDeviceView &output) const noexcept;

    [[nodiscard]] Status add_soft_body(SoftBodyOptions options,
                                       SoftBodyId &output) noexcept;
    [[nodiscard]] Status remove_soft_body(SoftBodyId id) noexcept;
    [[nodiscard]] Status soft_body_view(
        SoftBodyId id, SoftBodyDeviceView &output) const noexcept;

    [[nodiscard]] Status add_rope(RopeOptions options,
                                  std::uint32_t first_contact_skip,
                                  std::uint32_t last_contact_skip,
                                  RopeId &output) noexcept;
    [[nodiscard]] Status remove_rope(RopeId id) noexcept;
    [[nodiscard]] Status rope_view(RopeId id,
                                   RopeDeviceView &output) const noexcept;

    [[nodiscard]] Status add_particle_source(
        ParticleSourceMesh mesh, ParticleSourceOptions options,
        ParticleSourceId &output) noexcept;
    [[nodiscard]] Status update_particle_source(
        ParticleSourceId id, ParticleSourceOptions options) noexcept;
    [[nodiscard]] Status remove_particle_source(
        ParticleSourceId id) noexcept;
    [[nodiscard]] Status add_particle_destroy_plane(
        ParticleDestroyPlaneOptions options,
        ParticleDestroyPlaneId &output) noexcept;
    [[nodiscard]] Status update_particle_destroy_plane(
        ParticleDestroyPlaneId id,
        ParticleDestroyPlaneOptions options) noexcept;
    [[nodiscard]] Status remove_particle_destroy_plane(
        ParticleDestroyPlaneId id) noexcept;

    [[nodiscard]] Status add_paint_field(
        PaintFieldHostOptions options, std::uint32_t vertex_count,
        PaintFieldId &output) noexcept;
    [[nodiscard]] Status remove_paint_field(PaintFieldId id) noexcept;
    [[nodiscard]] Status clear_paint_field(PaintFieldId id) noexcept;
    [[nodiscard]] Status paint_field_view(
        PaintFieldId id, PaintFieldDeviceView &output) const noexcept;
    [[nodiscard]] Status add_paint_rule(
        PaintRuleOptions options, PaintRuleId &output) noexcept;
    [[nodiscard]] Status remove_paint_rule(PaintRuleId id) noexcept;
    [[nodiscard]] bool paint_field_exists(PaintFieldId id) const noexcept;
    [[nodiscard]] std::uint32_t cloth_source_vertex_count(
        ClothId id) const noexcept;

    [[nodiscard]] Status add_fluid_smoke_coupling(
        FluidSmokeCouplingOptions options,
        FluidSmokeCouplingId &output) noexcept;
    [[nodiscard]] Status remove_fluid_smoke_coupling(
        FluidSmokeCouplingId id) noexcept;
    [[nodiscard]] Status add_smoke_soft_body_coupling(
        SmokeSoftBodyCouplingOptions options,
        SmokeSoftBodyCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_soft_body_coupling(
        SmokeSoftBodyCouplingId id) noexcept;
    [[nodiscard]] Status add_smoke_cloth_coupling(
        SmokeClothCouplingOptions options,
        SmokeClothCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_cloth_coupling(
        SmokeClothCouplingId id) noexcept;
    [[nodiscard]] Status add_smoke_rope_coupling(
        SmokeRopeCouplingOptions options,
        SmokeRopeCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_rope_coupling(
        SmokeRopeCouplingId id) noexcept;
    [[nodiscard]] Status add_smoke_rigid_coupling(
        SmokeRigidCouplingOptions options,
        SmokeRigidCouplingId &output) noexcept;
    [[nodiscard]] Status remove_smoke_rigid_coupling(
        SmokeRigidCouplingId id) noexcept;

    [[nodiscard]] Status add_fluid_rope_coupling(
        FluidRopeCouplingOptions options,
        FluidRopeCouplingId &output) noexcept;
    [[nodiscard]] Status update_fluid_rope_coupling(
        FluidRopeCouplingId id, FluidRopeCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_fluid_rope_coupling(
        FluidRopeCouplingId id) noexcept;
    [[nodiscard]] Status add_rope_soft_body_coupling(
        RopeSoftBodyCouplingOptions options,
        RopeSoftBodyCouplingId &output) noexcept;
    [[nodiscard]] Status update_rope_soft_body_coupling(
        RopeSoftBodyCouplingId id,
        RopeSoftBodyCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_rope_soft_body_coupling(
        RopeSoftBodyCouplingId id) noexcept;
    [[nodiscard]] Status add_rope_cloth_coupling(
        RopeClothCouplingOptions options,
        RopeClothCouplingId &output) noexcept;
    [[nodiscard]] Status update_rope_cloth_coupling(
        RopeClothCouplingId id, RopeClothCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_rope_cloth_coupling(
        RopeClothCouplingId id) noexcept;
    [[nodiscard]] Status add_fluid_cloth_coupling(
        FluidClothCouplingOptions options,
        FluidClothCouplingId &output) noexcept;
    [[nodiscard]] Status update_fluid_cloth_coupling(
        FluidClothCouplingId id, FluidClothCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_fluid_cloth_coupling(
        FluidClothCouplingId id) noexcept;
    [[nodiscard]] Status add_soft_body_cloth_coupling(
        SoftBodyClothCouplingOptions options,
        SoftBodyClothCouplingId &output) noexcept;
    [[nodiscard]] Status update_soft_body_cloth_coupling(
        SoftBodyClothCouplingId id,
        SoftBodyClothCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_soft_body_cloth_coupling(
        SoftBodyClothCouplingId id) noexcept;
    [[nodiscard]] Status add_fluid_soft_body_coupling(
        FluidSoftBodyCouplingOptions options,
        FluidSoftBodyCouplingId &output) noexcept;
    [[nodiscard]] Status update_fluid_soft_body_coupling(
        FluidSoftBodyCouplingId id,
        FluidSoftBodyCouplingOptions options) noexcept;
    [[nodiscard]] Status remove_fluid_soft_body_coupling(
        FluidSoftBodyCouplingId id) noexcept;

    // Called once before encoding a frame. Buffers can move when rigid mesh
    // storage is rebuilt, so particle argument tables are refreshed here.
    void set_rigid_resources(void *ids, void *states, void *previous_states,
                             void *frame_states, void *parameters,
                             void *vertices, void *indices, void *meshes,
                             void *solid_planes,
                             std::uint32_t count) noexcept;
    // Explicit state edits can move an otherwise static smoke boundary.
    void invalidate_smoke_grid_static_metadata() noexcept;
    // Clears frame-scoped public diagnostics once before substep encoding so
    // every coupling can accumulate contributions across the whole frame.
    [[nodiscard]] Status begin_frame(bool collect_fluid_contacts,
                                     float frame_timestep) noexcept;
    // CUDA advances its Eulerian smoke grid before the rigid/deformable loop,
    // runs deformables on the rigid substep clock, then executes all fluid and
    // tracer work. Keeping those phases explicit prevents frame work from
    // accidentally being repeated or interleaved with world substeps.
    void encode(void *encoder, MetalSystemPhase phase, float timestep,
                std::uint32_t substeps, Vec3 gravity,
                MetalTimingContext *timings) noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool references_rigid_body(RigidBodyId id) const noexcept;
    [[nodiscard]] bool references_triangle_mesh(
        TriangleMeshId id) const noexcept;
    [[nodiscard]] ContactDeviceView contact_view(
        std::uint64_t frame_index) const noexcept;
    [[nodiscard]] std::uint32_t fluid_contact_overflow_count() const noexcept;
    [[nodiscard]] Status append_debug_samples(
        PhysicsDebugFrame &output, std::uint64_t frame_index) const noexcept;
    [[nodiscard]] std::uint32_t required_substeps(
        float timestep, std::uint32_t requested) const noexcept;
    void collect_statistics(WorldStatistics &output) const noexcept;
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;
    [[nodiscard]] void *fluid_neighbor_overflow_buffer() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parallel_mater::metal::detail
