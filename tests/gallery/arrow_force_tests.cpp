// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/arrow_forces.hpp>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace parallel_mater;
using namespace parallel_mater::gallery;

namespace {
void check(bool okay, const char *message) {
    if (!okay) throw std::runtime_error(message);
}
void require(Status status) { check(bool(status), status.message ? status.message : "physics operation failed"); }
float length(Vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
bool near(Vec3 a, Vec3 b, float tolerance = 2.0e-4F) {
    return length({a.x-b.x, a.y-b.y, a.z-b.z}) < tolerance;
}

void check_camera_forces() {
    const Camera front{{0, 0, 5}, {0, 0, 0}};
    check(near(screen_space_force(front, 1, 0, 100), {100, 0, 0}), "Right must apply 100 N screen-right");
    check(near(screen_space_force(front, -1, 0, 100), {-100, 0, 0}), "Left must reverse force");
    check(near(screen_space_force(front, 0, 1, 100), {}), "Vertical screen direction has no ground projection");
    check(near(screen_space_force(front, 0, -1, 100), {}), "Reverse vertical direction has no ground projection");
    check(near(screen_space_force(front, 0, 0, 100), {}), "Released/opposing keys must produce no force");
    check(near(screen_space_force(front, 1, 1, 0), {}), "Zero pm_arrow must disable force");
    const Camera side{{5, 0, 0}, {0, 0, 0}};
    check(near(screen_space_force(side, 1, 0, 100), {0, 0, -100}), "Force must follow camera orbit");
    const Camera pitched{{0, 3, 4}, {0, 0, 0}};
    check(near(screen_space_force(pitched, 0, 1, 100), {0, 0, -100}), "Screen Up must project onto the ground");
    const auto diagonal = screen_space_force(pitched, 1, 1, 100);
    const float diagonal_scale = 100.0F/std::sqrt(1.0F + 0.6F*0.6F);
    check(near(diagonal, {diagonal_scale, 0, -0.6F*diagonal_scale}) &&
              std::fabs(length(diagonal)-100) < 1.0e-4F,
          "Projected diagonal must remain parallel to ground and total 100 N");
    const Camera rolled{{0, 0, 5}, {0, 0, 0}, {1, 0, 0}};
    check(near(screen_space_force(rolled, 0, 1, 100), {100, 0, 0}), "Force must follow camera roll");
}
}

int main() try {
    check_camera_forces();
    SceneDefinition authored;
    std::string error;
    check(load_glb_scene(PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH, authored, error), error.c_str());
    const auto driven = std::find_if(authored.rigid_bodies.begin(), authored.rigid_bodies.end(),
        [](const auto &body) { return body.source_name == "GenericBlockA"; });
    check(driven != authored.rigid_bodies.end() && driven->arrow_force == 100.0F &&
          driven->options.motion == MotionType::dynamic, "Authored GenericBlockA must load its 100 N force");
    check(std::count_if(authored.rigid_bodies.begin(), authored.rigid_bodies.end(),
        [](const auto &body) { return body.arrow_force > 0.0F; }) == 1, "Only the tagged body must be controlled");
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
        std::cout << "SKIP: camera/metadata checks passed; CUDA unavailable\n";
        return 77;
    }
    const auto prototype = *driven;
    for (const unsigned substeps : {1U, 4U, 8U}) {
        SceneDefinition scene = authored;
        scene.rigid_constraints.clear();
        scene.rigid_bodies.assign(4, prototype);
        for (std::size_t i = 0; i < 4; ++i) {
            auto &body = scene.rigid_bodies[i];
            body.options.mass = i == 1 ? 4.0F : 1.0F;
            if (i == 3) {
                body.options.mass = 2.0F;
                body.arrow_force = 200.0F; // Shares the first body's acceleration batch.
            }
            body.options.linear_damping = body.options.angular_damping = 0.0F;
            body.options.initial_state = {.position = {float(i)*100.0F, 20.0F, 0}};
            if (i == 2) body.arrow_force = 0.0F;
        }
        World world;
        SceneInstance instance;
        require(create_scene_world(scene, world, instance));
        ArrowForces controls;
        require(controls.initialize(scene, instance));
        check(controls.active(), "Authored force controls must activate");
        const Camera camera{{0, 3, 4}, {0, 0, 0}};
        const StepOptions step{.timestep = 1.0F/60.0F, .substeps = substeps, .gravity = {}};
        for (int frame = 0; frame < 6; ++frame) {
            require(controls.apply(world, camera, 1, 1));
            require(world.step(step));
        }
        std::array<RigidBodyState, 4> states;
        for (std::size_t i = 0; i < states.size(); ++i) {
            require(world.read_rigid_body_state(instance.rigid_bodies[i], states[i]));
            const auto expected = screen_space_force(camera, 1, 1,
                scene.rigid_bodies[i].arrow_force * 0.1F / scene.rigid_bodies[i].options.mass);
            check(near(states[i].linear_velocity, expected), "100 N must produce F*dt/m velocity, independent of substeps");
            check(near(states[i].angular_velocity, {}), "Center force must not add torque");
        }
        require(controls.apply(world, camera, 0, 0));
        require(world.step(step));
        for (std::size_t i = 0; i < states.size(); ++i) {
            RigidBodyState released;
            require(world.read_rigid_body_state(instance.rigid_bodies[i], released));
            check(near(released.linear_velocity, states[i].linear_velocity), "Release must stop adding force, not erase momentum");
        }
        require(controls.apply(world, camera, -1, -1));
        require(world.step(step));
        RigidBodyState reversed;
        require(world.read_rigid_body_state(instance.rigid_bodies[0], reversed));
        check(near(reversed.linear_velocity, screen_space_force(camera, 1, 1, 100.0F/12.0F)),
              "Opposite keys must reverse the applied force");
        scene.rigid_bodies[0].arrow_force = -1;
        check(!controls.initialize(scene, instance) && !controls.active(), "Invalid controls must fail without partial bindings");
        for (auto &body : scene.rigid_bodies) body.arrow_force = 0;
        require(controls.initialize(scene, instance));
        check(!controls.active(), "Absent/zero force must preserve ordinary gallery controls");
    }
    std::cout << "Arrow force: 100 N, mass scaling, release, reversal, and 1/4/8 substeps passed\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
