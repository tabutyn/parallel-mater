// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <iomanip>
#include <iostream>

namespace {

bool require(parallel_mater::Status status, const char *operation) {
    if (status) {
        return true;
    }
    std::cerr << operation << " failed: "
              << (status.message != nullptr ? status.message : "unknown") << '\n';
    return false;
}

} // namespace

int main() {
    using namespace parallel_mater;
    World world;
    if (!require(World::create({.rigid_body_capacity = 8U}, world),
                 "World::create")) {
        return 1;
    }

    RigidBodyId floor{};
    if (!require(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::plane(),
                      .friction = 0.9F},
                     floor),
                 "add floor")) {
        return 1;
    }
    RigidBodyId obstacle{};
    if (!require(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::box({0.25F, 0.75F, 1.0F}),
                      .initial_state = {.position = {2.0F, 0.75F, 0.0F}},
                      .friction = 0.8F},
                     obstacle),
                 "add obstacle")) {
        return 1;
    }
    RigidBodyId ball{};
    if (!require(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.35F),
                      .initial_state = {.position = {-2.0F, 1.5F, 0.0F}},
                      .mass = 2.0F,
                      .friction = 0.9F},
                     ball),
                 "add ball")) {
        return 1;
    }
    if (!require(world.apply_impulse(ball, {7.0F, 0.0F, 0.0F},
                                     {-2.0F, 1.5F, 0.0F}),
                 "launch ball")) {
        return 1;
    }

    std::cout << "frame      x       y       vx      vy\n";
    for (int frame = 0; frame < 180; ++frame) {
        if (!require(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 4U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "step")) {
            return 1;
        }
        if (frame % 30 == 0 || frame == 179) {
            RigidBodyState state{};
            if (!require(world.read_rigid_body_state(ball, state), "read ball")) {
                return 1;
            }
            std::cout << std::setw(5) << frame << std::fixed
                      << std::setprecision(3) << std::setw(8) << state.position.x
                      << std::setw(8) << state.position.y << std::setw(8)
                      << state.linear_velocity.x << std::setw(8)
                      << state.linear_velocity.y << '\n';
        }
    }
    return 0;
}
