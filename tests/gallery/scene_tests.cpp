// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_status(parallel_mater::Status status, const char *operation) {
    if (!status) {
        std::cerr << "FAIL: " << operation << ": "
                  << (status.message != nullptr ? status.message : "unknown")
                  << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    SceneDefinition scene;
    std::string error;
    check(load_glb_scene(PARALLEL_MATER_PASSIVE_ACTIVE_SCENE_PATH, scene, error),
          error.empty() ? "load PassiveActive GLB scene" : error.c_str());
    check(scene.rigid_bodies.size() == 5U,
          "PassiveActive must contain the bowl and four test bodies");
    check(scene.meshes.size() == 5U,
          "PassiveActive must contain five render meshes");
    check(scene.collision_meshes.size() == 3U,
          "three dynamic bodies must carry authored collision proxies");
    if (scene.rigid_bodies.size() == 5U) {
        check(scene.rigid_bodies[0].name == "Plane" &&
                  scene.rigid_bodies[0].options.motion ==
                      MotionType::static_body,
              "Blender PASSIVE plane must load as static");
        check(scene.rigid_bodies[1].name == "Cube" &&
                  scene.rigid_bodies[1].options.motion == MotionType::kinematic,
              "Blender ACTIVE Animated cube must load as kinematic");
        check(scene.rigid_bodies[2].name == "Icosphere" &&
                  scene.rigid_bodies[2].options.motion == MotionType::dynamic,
              "Blender ACTIVE sphere mesh must load as dynamic");
        check(scene.rigid_bodies[3].name == "Suzanne" &&
                  scene.rigid_bodies[3].options.motion == MotionType::dynamic,
              "Blender ACTIVE Suzanne must load as dynamic");
        check(scene.rigid_bodies[4].name == "Suzanne.001" &&
                  scene.rigid_bodies[4].options.motion == MotionType::dynamic,
              "the second Blender ACTIVE Suzanne must load as dynamic");
        check(scene.meshes[0].checkerboard,
              "exported passive ground must retain its checkerboard flag");
        for (const TriangleMesh &mesh : scene.meshes) {
            check(!mesh.vertices.empty() && !mesh.indices.empty() &&
                      mesh.indices.size() % 3U == 0U,
                  "every Blender object must be triangulated");
        }
        check(scene.meshes[0].indices.size() == 2'048U * 3U,
              "the Blender bowl quads must export as 2,048 triangles");
        check(scene.rigid_bodies[0].collision_mesh_indices.empty() &&
                  scene.rigid_bodies[1].collision_mesh_indices.empty(),
              "the detailed bowl and cube must use their render triangles");
        for (std::size_t index = 2U; index < 5U; ++index) {
            check(!scene.rigid_bodies[index].collision_mesh_indices.empty(),
                  "each detailed dynamic body must select a collision proxy");
            if (!scene.rigid_bodies[index].collision_mesh_indices.empty()) {
                const TriangleMesh &proxy = scene.collision_meshes[
                    scene.rigid_bodies[index].collision_mesh_indices.front()];
                check(proxy.indices.size() <
                          scene.meshes[index].indices.size(),
                      "a collision proxy must contain fewer triangles than its render mesh");
            }
        }
        float cube_extent = 0.0F;
        for (const Vertex &vertex : scene.meshes[1].vertices) {
            cube_extent = std::fmax(cube_extent, std::fabs(vertex.position.x));
            cube_extent = std::fmax(cube_extent, std::fabs(vertex.position.y));
            cube_extent = std::fmax(cube_extent, std::fabs(vertex.position.z));
        }
        check(std::fabs(cube_extent - 0.6837F) < 1.0e-3F,
              "the Blender cube scale must be baked into its vertices");
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: GLB structure passed; CUDA device unavailable\n";
        return failures == 0 ? 77 : 1;
    }

    World world;
    check_status(World::create(
                     {.rigid_body_capacity =
                          static_cast<std::uint32_t>(scene.rigid_bodies.size()),
                      .triangle_mesh_capacity =
                          static_cast<std::uint32_t>(scene.rigid_bodies.size())},
                     world),
                 "create PassiveActive world");
    SceneInstance instance;
    check_status(instantiate_scene(scene, world, instance),
                 "instantiate PassiveActive scene");
    check(instance.rigid_bodies.size() == 5U &&
              instance.collision_meshes.size() == 5U,
          "each rigid body must instantiate one selected collision mesh");
    WorldStatistics statistics{};
    check_status(world.collect_statistics(statistics), "collect scene statistics");
    check(statistics.rigid_body_count == 5U &&
              statistics.triangle_mesh_count == 5U,
          "world must own five bodies and five triangle meshes");

    if (instance.rigid_bodies.size() == 5U) {
        RigidBodyState before_cube{};
        RigidBodyState before_icosphere{};
        RigidBodyState before_suzanne{};
        check_status(world.read_rigid_body_state(instance.rigid_bodies[1],
                                                 before_cube),
                     "read cube before stepping");
        check_status(world.read_rigid_body_state(instance.rigid_bodies[2],
                                                 before_icosphere),
                     "read icosphere before stepping");
        check_status(world.read_rigid_body_state(instance.rigid_bodies[3],
                                                 before_suzanne),
                     "read Suzanne before stepping");
        RigidBodyState cube_target = before_cube;
        cube_target.position.x += 0.1F;
        check_status(world.set_kinematic_target(instance.rigid_bodies[1],
                                                cube_target),
                     "set authored cube target");
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 4U,
                                 .gravity = {4.905F, -8.495709F, 0.0F}}),
                     "step authored scene");
        RigidBodyState after_cube{};
        RigidBodyState after_icosphere{};
        check_status(world.read_rigid_body_state(instance.rigid_bodies[1],
                                                 after_cube),
                     "read cube after stepping");
        check_status(world.read_rigid_body_state(instance.rigid_bodies[2],
                                                 after_icosphere),
                     "read icosphere after stepping");
        check(std::fabs(after_cube.position.x - cube_target.position.x) <
                  1.0e-5F &&
                  std::fabs(after_cube.position.y - before_cube.position.y) <
                      1.0e-5F,
              "kinematic cube must follow its target and ignore gravity");
        check(after_icosphere.position.x > before_icosphere.position.x &&
                  after_icosphere.position.y < before_icosphere.position.y,
              "tilted gravity must affect the dynamic icosphere");

        for (int frame = 1; frame < 360; ++frame) {
            check_status(world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = 4U,
                                     .gravity = {0.0F, -9.81F, 0.0F}}),
                         "settle authored bowl scene");
        }
        RigidBodyState after_suzanne{};
        check_status(world.read_rigid_body_state(instance.rigid_bodies[3],
                                                 after_suzanne),
                     "read Suzanne after settling");
        const float orientation_dot = std::fabs(
            before_suzanne.orientation.x * after_suzanne.orientation.x +
            before_suzanne.orientation.y * after_suzanne.orientation.y +
            before_suzanne.orientation.z * after_suzanne.orientation.z +
            before_suzanne.orientation.w * after_suzanne.orientation.w);
        check(orientation_dot < 0.995F,
              "Suzanne must topple instead of remaining upright");
        float bowl_minimum_y = std::numeric_limits<float>::max();
        for (const std::uint32_t mesh_index :
             scene.rigid_bodies[0].mesh_indices) {
            for (const Vertex &vertex : scene.meshes[mesh_index].vertices) {
                bowl_minimum_y = std::fmin(
                    bowl_minimum_y,
                    scene.rigid_bodies[0].options.initial_state.position.y +
                        vertex.position.y);
            }
        }
        for (std::size_t index = 2; index < 5; ++index) {
            RigidBodyState settled{};
            check_status(world.read_rigid_body_state(instance.rigid_bodies[index],
                                                     settled),
                         "read contained bowl body");
            check(std::isfinite(settled.position.y) &&
                      settled.position.y >= bowl_minimum_y - 0.05F,
                  "dynamic body must not escape below the bowl");
        }
    }

    if (failures != 0) {
        std::cerr << failures << " gallery scene test(s) failed\n";
        return 1;
    }
    std::cout << "PassiveActive Blender scene tests passed\n";
    return 0;
}
