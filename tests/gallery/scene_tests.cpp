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
    check(load_glb_scene(PARALLEL_MATER_RIGID_SCENE_PATH, scene, error),
          error.empty() ? "load rigid GLB scene" : error.c_str());
    check(scene.rigid_bodies.size() == 4U,
          "authored scene must contain four rigid bodies");
    check(scene.meshes.size() == 4U,
          "authored scene must contain four render meshes");
    if (scene.rigid_bodies.size() == 4U) {
        check(scene.rigid_bodies[0].options.motion == MotionType::static_body &&
                  scene.rigid_bodies[0].options.shape.type == ShapeType::plane,
              "first authored body must be the static plane");
        check(scene.rigid_bodies[1].options.motion == MotionType::dynamic &&
                  scene.rigid_bodies[1].options.shape.type == ShapeType::sphere,
              "second authored body must be the dynamic sphere");
        check(scene.rigid_bodies[2].options.shape.type == ShapeType::box,
              "third authored body must be the dynamic box");
        check(scene.rigid_bodies[3].options.shape.type == ShapeType::capsule,
              "fourth authored body must be the dynamic capsule");
        check(std::fabs(scene.rigid_bodies[1].options.shape.dimensions.x - 0.7F) <
                  1.0e-3F,
              "sphere collider radius must be derived from GLB geometry");
        check(scene.meshes[0].checkerboard,
              "plane checkerboard flag must survive Blender extras export");
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: GLB structure passed; CUDA device unavailable\n";
        return failures == 0 ? 77 : 1;
    }

    World world;
    check_status(World::create(
                     {.rigid_body_capacity =
                          static_cast<std::uint32_t>(scene.rigid_bodies.size())},
                     world),
                 "create authored scene world");
    SceneInstance instance;
    check_status(instantiate_scene(scene, world, instance),
                 "instantiate authored scene");
    check(instance.rigid_bodies.size() == scene.rigid_bodies.size(),
          "every authored body must receive a runtime handle");
    if (instance.rigid_bodies.size() == scene.rigid_bodies.size()) {
        RigidBodyState before{};
        RigidBodyState after{};
        check_status(world.read_rigid_body_state(instance.rigid_bodies[1], before),
                     "read sphere before stepping");
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 4U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "step authored scene");
        check_status(world.read_rigid_body_state(instance.rigid_bodies[1], after),
                     "read sphere after stepping");
        check(after.position.y < before.position.y,
              "dynamic authored sphere must advance under gravity");
    }

    if (failures != 0) {
        std::cerr << failures << " gallery scene test(s) failed\n";
        return 1;
    }
    std::cout << "Authored GLB scene tests passed\n";
    return 0;
}
