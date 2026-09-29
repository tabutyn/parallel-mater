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
    bool checkerboard{};
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
    std::uint32_t paint_resolution{512U};
    std::string source_name{};
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
    std::uint32_t mesh_index{};
};

// Presentation-only tube construction, shared by initial load and live drawing.
void update_rope_render_mesh(const std::vector<Vec3> &nodes, float radius,
                             TriangleMesh &mesh);

struct SceneDefinition {
    std::vector<TriangleMesh> meshes{};
    std::vector<TriangleMesh> collision_meshes{};
    std::vector<RigidBodyDefinition> rigid_bodies{};
    std::vector<ClothDefinition> cloths{};
    std::vector<SoftBodyDefinition> soft_bodies{};
    std::vector<RopeDefinition> ropes{};
    FluidOptions fluid_options{};
    float gravity_scale{1.0F};
    // Authored Flow/Geometry volumes are sampled once during scene loading.
    std::vector<FluidParticle> initial_particles{};
    std::vector<ParticleSourceDefinition> particle_sources{};
    std::vector<ParticleDestroyPlaneOptions> destroy_planes{};
};

struct SceneInstance {
    struct PaintBinding {
        std::uint32_t body_index{};
        std::uint32_t mesh_index{};
        PaintFieldId field{};
    };
    std::vector<RigidBodyId> rigid_bodies{};
    std::vector<ClothId> cloths{};
    std::vector<SoftBodyId> soft_bodies{};
    std::vector<RopeId> ropes{};
    std::vector<FluidClothCouplingId> fluid_cloth_couplings{};
    std::vector<SoftBodyClothCouplingId> soft_body_cloth_couplings{};
    std::vector<FluidSoftBodyCouplingId> fluid_soft_body_couplings{};
    std::vector<FluidRopeCouplingId> fluid_rope_couplings{};
    std::vector<RopeSoftBodyCouplingId> rope_soft_body_couplings{};
    std::vector<PaintBinding> paint_bindings{};
    FluidId fluid{};
    bool has_fluid{};
};

[[nodiscard]] bool load_glb_scene(const std::filesystem::path &path,
                                  SceneDefinition &output,
                                  std::string &error);

// Examples-only rigid stress scene. The installed physics API has no scene
// concepts; DUMP builds reusable triangle meshes through SceneDefinition.
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
