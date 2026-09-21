// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
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
    check(scene.rigid_bodies.size() == 4U,
          "PassiveActive must contain four triangle rigid bodies");
    check(scene.meshes.size() == 4U,
          "PassiveActive must contain four render meshes");
    if (scene.rigid_bodies.size() == 4U) {
        check(scene.rigid_bodies[0].name == "Plane" &&
                  scene.rigid_bodies[0].options.motion ==
                      MotionType::static_body,
              "Blender PASSIVE plane must load as static");
        check(scene.rigid_bodies[1].name == "Cube" &&
                  scene.rigid_bodies[1].options.motion == MotionType::dynamic,
              "Blender ACTIVE cube must load as dynamic");
        check(scene.rigid_bodies[2].name == "Icosphere" &&
                  scene.rigid_bodies[2].options.motion == MotionType::dynamic,
              "Blender ACTIVE sphere mesh must load as dynamic");
        check(scene.rigid_bodies[3].name == "Suzanne" &&
                  scene.rigid_bodies[3].options.motion == MotionType::dynamic,
              "Blender ACTIVE Suzanne must load as dynamic");
        check(scene.meshes[0].checkerboard,
              "exported passive ground must retain its checkerboard flag");
        for (const TriangleMesh &mesh : scene.meshes) {
            check(!mesh.vertices.empty() && !mesh.indices.empty() &&
                      mesh.indices.size() % 3U == 0U,
                  "every Blender object must be triangulated");
        }
        check(scene.meshes[0].indices.size() == 6U,
              "the Blender ground quad must export as two triangles");
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
    check(instance.rigid_bodies.size() == 4U &&
              instance.collision_meshes.size() == 4U,
          "render and collision must share one triangle mesh per body");
    WorldStatistics statistics{};
    check_status(world.collect_statistics(statistics), "collect scene statistics");
    check(statistics.rigid_body_count == 4U &&
              statistics.triangle_mesh_count == 4U,
          "world must own four bodies and four triangle meshes");

    if (instance.rigid_bodies.size() == 4U) {
        RigidBodyState before_cube{};
        RigidBodyState before_suzanne{};
        check_status(world.read_rigid_body_state(instance.rigid_bodies[1],
                                                 before_cube),
                     "read cube before stepping");
        check_status(world.read_rigid_body_state(instance.rigid_bodies[3],
                                                 before_suzanne),
                     "read Suzanne before stepping");
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 4U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "step authored scene");
        RigidBodyState after_cube{};
        check_status(world.read_rigid_body_state(instance.rigid_bodies[1],
                                                 after_cube),
                     "read cube after stepping");
        check(after_cube.position.y < before_cube.position.y,
              "Blender ACTIVE meshes must advance under gravity");

        for (int frame = 1; frame < 360; ++frame) {
            check_status(world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = 8U,
                                     .gravity = {0.0F, -9.81F, 0.0F}}),
                         "settle authored scene");
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
        check(after_suzanne.position.y > -0.5F,
              "Suzanne must remain supported by the triangle ground");
    }

    if (failures != 0) {
        std::cerr << failures << " gallery scene test(s) failed\n";
        return 1;
    }
    std::cout << "PassiveActive Blender scene tests passed\n";
    return 0;
}
