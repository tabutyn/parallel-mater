// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace parallel_mater::metal;

namespace {

bool require(bool condition, const char *message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

bool require_status(Status status, const char *context) {
    if (status.ok()) return true;
    std::cerr << context;
    if (status.message != nullptr) std::cerr << ": " << status.message;
    std::cerr << '\n';
    return false;
}

bool exercise_constraint(RigidConstraintOptions options,
                         RigidBodyState initial,
                         RigidBodyState &output,
                         RigidConstraintState *constraint_output = nullptr) {
    World world;
    Status status = World::create(
        {.rigid_body_capacity = 2U,
         .rigid_constraint_capacity = 1U,
         .triangle_mesh_capacity = 1U},
        world);
    if (!require_status(status, "create constraint world")) return false;

    const std::array<Vec3, 8> vertices{{
        {-0.1F, -0.1F, -0.1F}, {0.1F, -0.1F, -0.1F},
        {0.1F, -0.1F, 0.1F},   {-0.1F, -0.1F, 0.1F},
        {-0.1F, 0.1F, -0.1F},  {0.1F, 0.1F, -0.1F},
        {0.1F, 0.1F, 0.1F},    {-0.1F, 0.1F, 0.1F},
    }};
    const std::array<std::uint32_t, 36> indices{
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0};
    TriangleMeshId mesh{};
    status = world.add_triangle_mesh(
        {vertices.data(), vertices.size()},
        {indices.data(), indices.size()}, mesh);
    if (!require_status(status, "upload constraint mesh")) return false;

    RigidBodyId anchor{}, body{};
    status = world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = mesh}, anchor);
    if (!require_status(status, "add constraint anchor")) return false;
    status = world.add_rigid_body(
        {.mesh = mesh,
         .initial_state = initial,
         .linear_damping = 0.0F,
         .angular_damping = 0.0F},
        body);
    if (!require_status(status, "add constrained body")) return false;

    options.body_a = anchor;
    options.body_b = body;
    RigidConstraintId constraint{};
    status = world.add_rigid_constraint(options, constraint);
    if (!require_status(status, "add rigid constraint")) return false;
    status = world.step(
        {.timestep = 1.0F / 60.0F, .substeps = 8U, .gravity = {}});
    if (!require_status(status, "step rigid constraint")) return false;
    status = world.read_rigid_body_state(body, output);
    if (!require_status(status, "read constrained body")) return false;
    if (constraint_output != nullptr) {
        status = world.read_rigid_constraint_state(
            constraint, *constraint_output);
        if (!require_status(status, "read rigid constraint")) return false;
    }
    return true;
}

bool exercise_compound_weld() {
    World world;
    Status status = World::create(
        {.rigid_body_capacity = 2U,
         .rigid_constraint_capacity = 1U,
         .triangle_mesh_capacity = 1U},
        world);
    if (!require_status(status, "create compound world")) return false;

    const std::array<Vec3, 8> vertices{{
        {-0.1F, -0.1F, -0.1F}, {0.1F, -0.1F, -0.1F},
        {0.1F, -0.1F, 0.1F},   {-0.1F, -0.1F, 0.1F},
        {-0.1F, 0.1F, -0.1F},  {0.1F, 0.1F, -0.1F},
        {0.1F, 0.1F, 0.1F},    {-0.1F, 0.1F, 0.1F},
    }};
    const std::array<std::uint32_t, 36> indices{
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0};
    TriangleMeshId mesh{};
    status = world.add_triangle_mesh(
        {vertices.data(), vertices.size()},
        {indices.data(), indices.size()}, mesh);
    if (!require_status(status, "upload compound mesh")) return false;

    RigidBodyId first{}, second{};
    status = world.add_rigid_body(
        {.mesh = mesh,
         .initial_state = {.position = {-0.1F, 1.0F, 0.0F}},
         .mass = 1.0F,
         .linear_damping = 0.0F,
         .angular_damping = 0.0F},
        first);
    if (!require_status(status, "add first compound body")) return false;
    status = world.add_rigid_body(
        {.mesh = mesh,
         .initial_state = {.position = {0.1F, 1.0F, 0.0F}},
         .mass = 1.0F,
         .linear_damping = 0.0F,
         .angular_damping = 0.0F},
        second);
    if (!require_status(status, "add second compound body")) return false;

    RigidConstraintId constraint{};
    status = world.add_rigid_constraint(
        {.type = RigidConstraintType::fixed,
         .body_a = first,
         .body_b = second,
         .local_anchor_a = {0.1F, 0.0F, 0.0F},
         .local_anchor_b = {-0.1F, 0.0F, 0.0F},
         .disable_collisions = true},
        constraint);
    if (!require_status(status, "add compound weld")) return false;
    status = world.apply_impulse(first, {1.0F, 0.0F, 0.0F},
                                 {-0.1F, 1.0F, 0.0F});
    if (!require_status(status, "impulse compound member")) return false;
    status = world.step(
        {.timestep = 1.0F / 60.0F, .substeps = 1U, .gravity = {}});
    if (!require_status(status, "step compound weld")) return false;

    RigidBodyState first_state{}, second_state{};
    RigidConstraintState constraint_state{};
    if (!require_status(world.read_rigid_body_state(first, first_state),
                        "read first compound member") ||
        !require_status(world.read_rigid_body_state(second, second_state),
                        "read second compound member") ||
        !require_status(
            world.read_rigid_constraint_state(constraint, constraint_state),
            "read absorbed compound weld"))
        return false;
    if (!require(
        std::abs(first_state.linear_velocity.x - 0.5F) < 1.0e-4F &&
            std::abs(second_state.linear_velocity.x - 0.5F) < 1.0e-4F &&
            std::abs((second_state.position.x - first_state.position.x) -
                     0.2F) < 1.0e-4F &&
            constraint_state.applied_impulse == 0.0F,
        "Compound weld did not preserve aggregate momentum and pose"))
        return false;

    if (!require_status(world.remove_rigid_constraint(constraint),
                        "remove final compound weld"))
        return false;
    status = world.step({.timestep = 1.0F / 60.0F,
                         .substeps = 1U,
                         .gravity = {},
                         .collect_rigid_contacts = true});
    return require_status(status, "step released compound members") &&
        require(world.rigid_contacts().event_count > 0U,
                "Released compound members remained collision-filtered");
}

} // namespace

int main() {
    if (!exercise_compound_weld()) return 1;

    RigidBodyState result{};
    if (!exercise_constraint(
            {.type = RigidConstraintType::fixed},
            {.linear_velocity = {2.0F, 2.0F, 0.0F},
             .angular_velocity = {2.0F, 0.0F, 0.0F}},
            result) ||
        !require(std::abs(result.linear_velocity.x) < 0.05F &&
                     std::abs(result.linear_velocity.y) < 0.05F &&
                     std::abs(result.angular_velocity.x) < 0.05F,
                 "Fixed constraint did not lock its frame"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::point},
            {.linear_velocity = {2.0F, 0.0F, 0.0F},
             .angular_velocity = {0.0F, 0.0F, 2.0F}},
            result) ||
        !require(std::abs(result.linear_velocity.x) < 0.05F &&
                     result.angular_velocity.z > 1.0F,
                 "Point constraint did not preserve free rotation"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::hinge,
             .angular_limits = {
                 .axes = rigid_constraint_axis_z,
                 .lower = {0.0F, 0.0F, -0.785398F},
                 .upper = {0.0F, 0.0F, 0.785398F}}},
            {.angular_velocity = {2.0F, 0.0F, 2.0F}}, result) ||
        !require(std::abs(result.angular_velocity.x) < 0.05F &&
                     result.angular_velocity.z > 1.0F,
                 "Hinge constraint did not retain only Z rotation"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::hinge,
             .angular_limits = {
                 .axes = rigid_constraint_axis_z,
                 .lower = {0.0F, 0.0F, -0.1F},
                 .upper = {0.0F, 0.0F, 0.1F}}},
            {.orientation = {
                 0.0F, 0.0F, std::sin(0.25F), std::cos(0.25F)}},
            result) ||
        !require(result.angular_velocity.z < 0.0F,
                 "Hinge angular limit did not oppose excess rotation"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::slider,
             .linear_limits = {
                 .axes = rigid_constraint_axis_x,
                 .lower = {-1.0F, 0.0F, 0.0F},
                 .upper = {1.0F, 0.0F, 0.0F}}},
            {.linear_velocity = {2.0F, 2.0F, 0.0F}}, result) ||
        !require(result.linear_velocity.x > 1.0F &&
                     std::abs(result.linear_velocity.y) < 0.05F,
                 "Slider constraint did not preserve only X translation"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::slider,
             .linear_limits = {
                 .axes = rigid_constraint_axis_x,
                 .lower = {-1.0F, 0.0F, 0.0F},
                 .upper = {1.0F, 0.0F, 0.0F}}},
            {.position = {2.0F, 0.0F, 0.0F}}, result) ||
        !require(result.linear_velocity.x < 0.0F,
                 "Slider linear limit did not oppose excess translation"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::piston,
             .linear_limits = {
                 .axes = rigid_constraint_axis_x,
                 .lower = {-1.0F, 0.0F, 0.0F},
                 .upper = {1.0F, 0.0F, 0.0F}}},
            {.linear_velocity = {2.0F, 2.0F, 0.0F},
             .angular_velocity = {2.0F, 2.0F, 0.0F}},
            result) ||
        !require(result.linear_velocity.x > 1.0F &&
                     std::abs(result.linear_velocity.y) < 0.05F &&
                     result.angular_velocity.x > 1.0F &&
                     std::abs(result.angular_velocity.y) < 0.05F,
                 "Piston constraint did not preserve X translation/rotation"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::piston,
             .linear_limits = {
                 .axes = rigid_constraint_axis_x,
                 .lower = {-1.0F, 0.0F, 0.0F},
                 .upper = {1.0F, 0.0F, 0.0F}},
             .angular_limits = {
                 .axes = rigid_constraint_axis_x,
                 .lower = {-0.1F, 0.0F, 0.0F},
                 .upper = {0.1F, 0.0F, 0.0F}}},
            {.position = {2.0F, 0.0F, 0.0F},
             .orientation = {
                 std::sin(0.25F), 0.0F, 0.0F, std::cos(0.25F)}},
            result) ||
        !require(result.linear_velocity.x < 0.0F &&
                     result.angular_velocity.x < 0.0F,
                 "Piston limits did not oppose excess frame motion"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::generic,
             .linear_limits = {.axes = rigid_constraint_axis_y},
             .angular_limits = {.axes = rigid_constraint_axis_z}},
            {.linear_velocity = {2.0F, 2.0F, 0.0F},
             .angular_velocity = {2.0F, 0.0F, 2.0F}},
            result) ||
        !require(result.linear_velocity.x > 1.0F &&
                     std::abs(result.linear_velocity.y) < 0.05F &&
                     result.angular_velocity.x > 1.0F &&
                     std::abs(result.angular_velocity.z) < 0.05F,
                 "Generic constraint did not enforce authored axes"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::generic_spring,
             .linear_springs = {
                 .axes = rigid_constraint_axis_x,
                 .stiffness = {40.0F, 0.0F, 0.0F},
                 .damping = {2.0F, 0.0F, 0.0F}}},
            {.position = {1.0F, 0.0F, 0.0F}}, result) ||
        !require(result.linear_velocity.x < -0.05F,
                 "Generic linear spring did not pull toward equilibrium"))
        return 1;

    const float half_angle = 0.25F;
    if (!exercise_constraint(
            {.type = RigidConstraintType::generic_spring,
             .angular_springs = {
                 .axes = rigid_constraint_axis_x,
                 .stiffness = {40.0F, 0.0F, 0.0F},
                 .damping = {2.0F, 0.0F, 0.0F}}},
            {.orientation = {
                 std::sin(half_angle), 0.0F, 0.0F,
                 std::cos(half_angle)}},
            result) ||
        !require(result.angular_velocity.x < 0.0F,
                 "Generic angular spring did not pull toward equilibrium"))
        return 1;

    if (!exercise_constraint(
            {.type = RigidConstraintType::motor,
             .motor = {.linear_enabled = true,
                       .angular_enabled = true,
                       .linear_target_velocity = 4.0F,
                       .linear_maximum_impulse = 10.0F,
                       .angular_target_velocity = 4.0F,
                       .angular_maximum_impulse = 10.0F}},
            {}, result) ||
        !require(result.linear_velocity.x > 1.0F &&
                     result.angular_velocity.x > 1.0F,
                 "Motor constraint did not drive both X axes"))
        return 1;

    RigidConstraintState broken{};
    if (!exercise_constraint(
            {.type = RigidConstraintType::fixed,
             .breaking_impulse_threshold = 0.0001F},
            {.linear_velocity = {5.0F, 0.0F, 0.0F}}, result, &broken) ||
        !require(broken.broken && !broken.enabled &&
                     broken.applied_impulse > 0.0F,
                 "Overloaded constraint did not preserve break diagnostics"))
        return 1;

    return 0;
}
