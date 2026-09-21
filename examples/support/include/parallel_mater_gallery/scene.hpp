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
};

struct TriangleMesh {
    std::string name{};
    std::vector<Vertex> vertices{};
    std::vector<std::uint32_t> indices{};
    Vec3 base_color{0.7F, 0.7F, 0.7F};
    bool checkerboard{};
};

struct RigidBodyDefinition {
    std::string name{};
    RigidBodyOptions options{};
    std::vector<std::uint32_t> mesh_indices{};
};

struct SceneDefinition {
    std::vector<TriangleMesh> meshes{};
    std::vector<RigidBodyDefinition> rigid_bodies{};
};

struct SceneInstance {
    std::vector<TriangleMeshId> collision_meshes{};
    std::vector<RigidBodyId> rigid_bodies{};
};

[[nodiscard]] bool load_glb_scene(const std::filesystem::path &path,
                                  SceneDefinition &output,
                                  std::string &error);

[[nodiscard]] Status instantiate_scene(const SceneDefinition &scene,
                                       World &world,
                                       SceneInstance &output) noexcept;

} // namespace parallel_mater::gallery
