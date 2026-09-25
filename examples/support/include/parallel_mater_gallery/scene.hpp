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
};

struct SceneDefinition {
    std::vector<TriangleMesh> meshes{};
    std::vector<TriangleMesh> collision_meshes{};
    std::vector<RigidBodyDefinition> rigid_bodies{};
    FluidOptions fluid_options{};
    float gravity_scale{1.0F};
    // Authored Flow/Geometry volumes are sampled once during scene loading.
    std::vector<FluidParticle> initial_particles{};
    std::vector<ParticleSpawnPlaneOptions> spawn_planes{};
    std::vector<ParticleDestroyPlaneOptions> destroy_planes{};
};

struct SceneInstance {
    std::vector<RigidBodyId> rigid_bodies{};
    FluidId fluid{};
    bool has_fluid{};
};

[[nodiscard]] bool load_glb_scene(const std::filesystem::path &path,
                                  SceneDefinition &output,
                                  std::string &error);

// Examples-only rigid stress scene. The installed physics API has no scene
// concepts; DUMP builds reusable triangle meshes through SceneDefinition.
[[nodiscard]] SceneDefinition make_dump_scene(std::uint32_t sphere_count);

[[nodiscard]] Status instantiate_scene(const SceneDefinition &scene,
                                       World &world,
                                       SceneInstance &output) noexcept;

} // namespace parallel_mater::gallery
