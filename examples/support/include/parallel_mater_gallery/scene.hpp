// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

struct Vertex {
    Vec3 position{};
    Vec3 normal{};
    Vec2 uv{};
};

struct TriangleMesh {
    std::string name{};
    std::vector<Vertex> vertices{};
    std::vector<std::uint32_t> indices{};
    Vec3 base_color{0.7F, 0.7F, 0.7F};
    // Constant material alpha can hide render faces without removing collision.
    // Partial alpha blending and texture cutouts are not supported.
    bool visible{true};
};

struct RigidBodyDefinition {
    std::string name{};
    RigidBodyOptions options{};
    std::vector<std::uint32_t> mesh_indices{};
    // Empty means the render triangles also drive collision. Otherwise these
    // indices address SceneDefinition::collision_meshes.
    std::vector<std::uint32_t> collision_mesh_indices{};
    bool paintable{};
    bool smoke_collider{};
    // Gallery steering may tilt world gravity while selected scenery retains
    // authored vertical gravity. This does not make the body static: contacts
    // and impulses still move it normally.
    bool follows_gravity_tilt{true};
    // Per-object arrow force in newtons, applied at the center of mass.
    float arrow_force{};
    std::uint32_t paint_resolution{512U};
    std::string source_name{};
};

struct RigidConstraintDefinition {
    std::string name{};
    RigidConstraintOptions options{};
    std::uint32_t body_a{};
    std::uint32_t body_b{};
};

struct ClothDefinition {
    std::string name{};
    std::uint32_t mesh_index{};
    std::vector<float> inverse_masses{};
    float vertex_mass{0.001F};
    float thickness{0.025F};
    float break_strain{};
    std::uint32_t fracture_persistence_substeps{4U};
    float impact_break_impulse{};
    float stretch_compliance{1.0e-6F};
    float velocity_damping{5.0F};
    float contact_friction{0.4F};
    std::uint32_t solver_iterations{8U};
    bool preserve_volume{};
    float target_volume{};
    float volume_compliance{1.0e-7F};
    bool contains_fluid{};
    bool paintable{};
    std::uint32_t paint_resolution{512U};
    std::string paint_source{};
    float paint_brush_radius{0.15F};
};

struct SoftBodyDefinition {
    std::string name{};
    std::uint32_t mesh_index{};
    std::vector<Vec3> nodes{};
    std::vector<SoftBodyBond> bonds{};
    std::vector<float> inverse_masses{};
    std::vector<SoftBodySurfaceBinding> surface_bindings{};
    float node_mass{0.02F};
    float node_radius{0.05F};
    float stretch_compliance{1.0e-7F};
    float velocity_damping{0.8F};
    float spring_damping{0.85F};
    float contact_friction{0.5F};
    float shape_matching_stiffness{};
    float maximum_projection_fraction{0.20F};
    float constraint_velocity_response{0.70F};
    float maximum_speed{2.0F};
    std::uint32_t solver_iterations{16U};
};

struct ParticleSourceDefinition {
    std::vector<Vec3> vertices{};
    std::vector<std::uint32_t> indices{};
    float spacing{};
    ParticleSourceOptions options{};
};

struct RopeDefinition {
    std::string name{};
    std::vector<Vec3> centerline{};
    RopeOptions options{};
    std::int32_t first_body{-1}, last_body{-1};
    std::int32_t first_soft_body{-1}, last_soft_body{-1};
    std::int32_t first_cloth{-1}, last_cloth{-1};
    std::uint32_t first_cloth_vertex{UINT32_MAX}, last_cloth_vertex{UINT32_MAX};
    std::uint32_t mesh_index{};
};

struct HitBoxDefinition {
    std::string name{};
    HitBox box{};
};

// Presentation-only tube construction, shared by initial load and live drawing.
void update_rope_render_mesh(const std::vector<Vec3> &nodes, float radius,
                             TriangleMesh &mesh);

struct SceneDefinition {
    std::vector<TriangleMesh> meshes{};
    std::vector<TriangleMesh> collision_meshes{};
    std::vector<RigidBodyDefinition> rigid_bodies{};
    // Non-simulated templates at Blender Empty poses, expanded by the gallery.
    std::vector<RigidBodyDefinition> sphere_clusters{};
    std::vector<RigidConstraintDefinition> rigid_constraints{};
    std::vector<ClothDefinition> cloths{};
    std::vector<SoftBodyDefinition> soft_bodies{};
    std::vector<RopeDefinition> ropes{};
    std::vector<HitBoxDefinition> hit_boxes{};
    FluidOptions fluid_options{};
    float gravity_scale{1.0F};
    // Authored Flow/Geometry volumes are sampled once during scene loading.
    std::vector<FluidParticle> initial_particles{};
    std::vector<ParticleSourceDefinition> particle_sources{};
    std::vector<ParticleDestroyPlaneOptions> destroy_planes{};
    SmokeOptions smoke_options{};
    std::string smoke_obstacle_name{};
    struct ThermalSurfaceDefinition {
        ParticlePlane plane{};
        float temperature{500.0F};
        float heat_transfer_rate{0.2F};
        float smoke_drag{2.0F};
        float steam_rise_speed{2.0F};
    };
    std::vector<ThermalSurfaceDefinition> thermal_surfaces{};
    bool has_smoke{};
};

struct SceneInstance {
    struct PaintBinding {
        std::uint32_t body_index{};
        std::uint32_t mesh_index{};
        PaintFieldId field{};
    };
    std::vector<RigidBodyId> rigid_bodies{};
    std::vector<RigidConstraintId> rigid_constraints{};
    std::vector<ClothId> cloths{};
    std::vector<SoftBodyId> soft_bodies{};
    std::vector<RopeId> ropes{};
    std::vector<FluidClothCouplingId> fluid_cloth_couplings{};
    std::vector<SoftBodyClothCouplingId> soft_body_cloth_couplings{};
    std::vector<FluidSoftBodyCouplingId> fluid_soft_body_couplings{};
    std::vector<FluidRopeCouplingId> fluid_rope_couplings{};
    std::vector<RopeSoftBodyCouplingId> rope_soft_body_couplings{};
    std::vector<RopeClothCouplingId> rope_cloth_couplings{};
    std::vector<PaintBinding> paint_bindings{};
    FluidId fluid{};
    bool has_fluid{};
    SmokeId smoke{};
    bool has_smoke{};
    std::vector<FluidSmokeCouplingId> fluid_smoke_couplings{};
    std::vector<SmokeSoftBodyCouplingId> smoke_soft_body_couplings{};
    std::vector<SmokeClothCouplingId> smoke_cloth_couplings{};
    std::vector<SmokeRopeCouplingId> smoke_rope_couplings{};
    std::vector<SmokeRigidCouplingId> smoke_rigid_couplings{};
};

[[nodiscard]] bool load_glb_scene(const std::filesystem::path &path,
                                  SceneDefinition &output,
                                  std::string &error);

// Examples-only rigid stress scene. The installed physics API has no scene
// concepts; the legacy hopper fixture builds reusable triangle meshes through
// SceneDefinition. The gallery's DUMP entry now loads DumpTruck.glb instead.
[[nodiscard]] SceneDefinition make_dump_scene(std::uint32_t sphere_count);

// Derives the capacities needed by the shared scene instantiator. Gallery
// clients, tests, and benchmarks should use this instead of mirroring its
// mesh/paint/cloth accounting.
[[nodiscard]] Status scene_world_options(
    const SceneDefinition &scene, WorldOptions &output,
    PhysicsDebugOptions physics_debug = {}) noexcept;

[[nodiscard]] Status create_scene_world(
    const SceneDefinition &scene, World &world, SceneInstance &output,
    PhysicsDebugOptions physics_debug = {}) noexcept;

[[nodiscard]] Status instantiate_scene(const SceneDefinition &scene,
                                       World &world,
                                       SceneInstance &output) noexcept;

} // namespace parallel_mater::gallery
