// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <array>
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

struct ExpectedScene {
    const char *path;
    parallel_mater::RigidConstraintType type;
    std::size_t constraint_count;
};

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    constexpr std::array scenes{
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH,
                      RigidConstraintType::fixed, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH,
                      RigidConstraintType::point, 2U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH,
                      RigidConstraintType::hinge, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_SLIDER_SCENE_PATH,
                      RigidConstraintType::slider, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH,
                      RigidConstraintType::piston, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH,
                      RigidConstraintType::generic, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_GENERIC_SPRING_SCENE_PATH,
                      RigidConstraintType::generic_spring, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_MOTOR_SCENE_PATH,
                      RigidConstraintType::motor, 4U},
    };

    std::array<SceneDefinition, scenes.size()> definitions{};
    for (std::size_t index = 0U; index < scenes.size(); ++index) {
        std::string error;
        check(load_glb_scene(scenes[index].path, definitions[index], error),
              error.empty() ? "load rigid constraint scene" : error.c_str());
        const SceneDefinition &scene = definitions[index];
        check(scene.rigid_constraints.size() == scenes[index].constraint_count,
              "scene must contain expected constraint count");
        for (const auto &constraint : scene.rigid_constraints) {
            check(constraint.options.type == scenes[index].type,
                  "scene must retain Blender constraint type");
            check(constraint.body_a < scene.rigid_bodies.size() &&
                      constraint.body_b < scene.rigid_bodies.size(),
                  "constraint must resolve both rigid body names");
            check(constraint.options.solver_iterations == 16U,
                  "constraint must retain authored solver iterations");
        }
    }

    const auto &fixed = definitions[0].rigid_constraints.front().options;
    check(!fixed.enabled, "interactive fixed constraint must start released");
    const auto &point_constraints = definitions[1].rigid_constraints;
    check(point_constraints[0].options.enabled &&
              point_constraints[1].options.enabled,
          "point constraints must start enabled");
    check(point_constraints[0].body_a == point_constraints[1].body_a &&
              point_constraints[0].body_b != point_constraints[1].body_b,
          "point spheres must share one static anchor body");
    check(point_constraints[0].options.local_anchor_a.x ==
                  point_constraints[1].options.local_anchor_a.x &&
              point_constraints[0].options.local_anchor_a.y ==
                  point_constraints[1].options.local_anchor_a.y &&
              point_constraints[0].options.local_anchor_a.z ==
                  point_constraints[1].options.local_anchor_a.z &&
              point_constraints[0].options.local_anchor_b.x *
                  point_constraints[1].options.local_anchor_b.x < 0.0F,
          "point spheres must pivot around opposite sides of one point");
    const auto &point_a = definitions[1].rigid_bodies[
        point_constraints[0].body_b].options.initial_state;
    const auto &point_b = definitions[1].rigid_bodies[
        point_constraints[1].body_b].options.initial_state;
    check(point_a.linear_velocity.z * point_b.linear_velocity.z < 0.0F &&
              std::fabs(point_a.linear_velocity.z) > 2.0F &&
              std::fabs(point_b.linear_velocity.z) > 2.0F,
          "point spheres must start with opposite tangential velocities");
    const auto &hinge = definitions[2].rigid_constraints.front().options;
    check(hinge.angular_limits.axes == rigid_constraint_axis_z &&
              std::fabs(hinge.angular_limits.lower.z + 0.7853982F) < 1.0e-4F &&
              std::fabs(hinge.angular_limits.upper.z - 0.7853982F) < 1.0e-4F,
          "hinge must retain its +/-45 degree limit");
    for (std::size_t index : {3U, 4U}) {
        const auto &constraint =
            definitions[index].rigid_constraints.front().options;
        check(constraint.linear_limits.axes == rigid_constraint_axis_x &&
                  constraint.linear_limits.lower.x == -1.0F &&
                  constraint.linear_limits.upper.x == 1.0F,
              "slider and piston must retain their -1m to +1m travel");
    }
    const auto &generic = definitions[5].rigid_constraints.front().options;
    check(generic.linear_limits.axes == rigid_constraint_all_axes &&
              generic.angular_limits.axes == rigid_constraint_all_axes,
          "generic constraint must limit all six axes");
    const auto &spring = definitions[6].rigid_constraints.front().options;
    check(spring.linear_springs.axes == rigid_constraint_all_axes &&
              spring.angular_springs.axes == rigid_constraint_all_axes,
          "generic spring must spring all six axes");
    for (const auto &constraint : definitions[7].rigid_constraints)
        check(constraint.options.motor.angular_enabled &&
                  constraint.options.motor.angular_maximum_impulse == 8.0F,
              "each car wheel must retain its angular motor");

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: constraint GLB structure passed; CUDA unavailable\n";
        return failures == 0 ? 77 : 1;
    }
    for (const SceneDefinition &scene : definitions) {
        World world;
        SceneInstance instance;
        check_status(create_scene_world(scene, world, instance),
                     "instantiate rigid constraint scene");
        WorldStatistics statistics{};
        check_status(world.collect_statistics(statistics),
                     "collect rigid constraint statistics");
        check(statistics.rigid_constraint_count == scene.rigid_constraints.size(),
              "world statistics must expose instantiated constraints");
        for (int frame = 0; frame < 30; ++frame)
            check_status(world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = 4U,
                                     .gravity = {0.0F, -9.81F, 0.0F}}),
                         "step rigid constraint scene");
    }
    return failures == 0 ? 0 : 1;
}
