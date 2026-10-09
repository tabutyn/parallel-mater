// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using parallel_mater::TriangleMeshId;
using parallel_mater::Vec3;

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
                  << " (CUDA " << static_cast<int>(status.cuda_error) << ")\n";
        ++failures;
    }
}

bool near(float actual, float expected, float tolerance = 1.0e-4F) {
    return std::fabs(actual - expected) <= tolerance;
}

parallel_mater::Quaternion rotation_z(float angle) {
    return {0.0F, 0.0F, std::sin(angle * 0.5F), std::cos(angle * 0.5F)};
}

TriangleMeshId upload_mesh(parallel_mater::World &world,
                           const std::vector<Vec3> &vertices,
                           const std::vector<std::uint32_t> &indices,
                           const char *operation) {
    Vec3 *device_vertices = nullptr;
    std::uint32_t *device_indices = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&device_vertices),
                     vertices.size() * sizeof(Vec3)) == cudaSuccess,
          "allocate mesh vertices");
    check(cudaMalloc(reinterpret_cast<void **>(&device_indices),
                     indices.size() * sizeof(std::uint32_t)) == cudaSuccess,
          "allocate mesh indices");
    check(cudaMemcpy(device_vertices, vertices.data(),
                     vertices.size() * sizeof(Vec3), cudaMemcpyHostToDevice) ==
              cudaSuccess,
          "upload mesh vertices");
    check(cudaMemcpy(device_indices, indices.data(),
                     indices.size() * sizeof(std::uint32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload mesh indices");
    TriangleMeshId result{};
    check_status(world.add_triangle_mesh({device_vertices, vertices.size()},
                                         {device_indices, indices.size()}, result),
                 operation);
    cudaFree(device_indices);
    cudaFree(device_vertices);
    return result;
}

TriangleMeshId add_plane(parallel_mater::World &world) {
    return upload_mesh(world,
                       {{-6.0F, 0.0F, -6.0F}, {6.0F, 0.0F, -6.0F},
                        {6.0F, 0.0F, 6.0F}, {-6.0F, 0.0F, 6.0F}},
                       {0U, 2U, 1U, 0U, 3U, 2U}, "add plane mesh");
}

TriangleMeshId add_box(parallel_mater::World &world, Vec3 half) {
    const std::vector<Vec3> vertices{
        {-half.x, -half.y, -half.z}, {half.x, -half.y, -half.z},
        {half.x, half.y, -half.z},   {-half.x, half.y, -half.z},
        {-half.x, -half.y, half.z},  {half.x, -half.y, half.z},
        {half.x, half.y, half.z},    {-half.x, half.y, half.z}};
    const std::vector<std::uint32_t> indices{
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0};
    return upload_mesh(world, vertices, indices, "add box mesh");
}

void test_small_triangle_edge_clearance() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U}, world),
                 "create small triangle clearance world");
    const TriangleMeshId surface = upload_mesh(
        world, {{0.0F, 0.0F, 0.0F}, {0.04F, 0.0F, 0.0F},
                {0.0F, 0.04F, 0.0F}},
        {0U, 1U, 2U}, "add small triangle surface");
    const TriangleMeshId crossing = upload_mesh(
        world, {{0.0F, 0.0F, -0.01F}, {0.0F, 0.0F, 0.01F},
                {0.01F, 0.0F, 0.01F}},
        {0U, 1U, 2U}, "add crossing triangle");
    RigidBodyId fixed{}, moving{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = surface,
                      .collision_margin = 0.001F}, fixed),
                 "add small fixed triangle");
    check_status(world.add_rigid_body(
                     {.mesh = crossing,
                      .initial_state = {.position = {0.024F, 0.024F, 0.0F}},
                      .collision_margin = 0.001F}, moving),
                 "add triangle beyond diagonal edge");
    // The edge crosses the triangle's plane, but lies 5.7 mm outside its
    // diagonal. Overlapping AABBs must not turn that into a surface hit.
    check_status(world.step({.timestep = 1.0F / 480.0F, .gravity = {},
                             .collect_rigid_contacts = true}),
                 "step small triangle clearance");
    check(world.rigid_contacts().event_count == 0U,
          "small triangle edge clearance must not report a false intersection");
    check_status(world.set_rigid_body_state(
                     moving, {.position = {0.01F, 0.01F, 0.0F}}),
                 "move crossing edge inside small triangle");
    check_status(world.step({.timestep = 1.0F / 480.0F, .gravity = {},
                             .collect_rigid_contacts = true}),
                 "step real small triangle intersection");
    check(world.rigid_contacts().event_count > 0U,
          "small triangle must still detect a real intersection");
}

void test_mesh_lifetime_and_integration() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 1U},
                               world),
                 "create integration world");
    const TriangleMeshId mesh = add_box(world, {0.25F, 0.25F, 0.25F});
    RigidBodyId body{};
    check_status(world.add_rigid_body(
                     {.mesh = mesh,
                      .initial_state = {.position = {0.0F, 10.0F, 0.0F}},
                      .mass = 2.0F,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F,
                      .maximum_linear_speed = 1'000.0F},
                     body),
                 "add integration body");
    check(world.remove_triangle_mesh(mesh).code == StatusCode::invalid_argument,
          "a body must retain its triangle mesh resource");

    constexpr float timestep = 0.1F;
    constexpr std::uint32_t substeps = 4U;
    constexpr float substep = timestep / static_cast<float>(substeps);
    float expected_y = 10.0F;
    float expected_velocity = 0.0F;
    for (std::uint32_t index = 0; index < substeps; ++index) {
        expected_velocity += -9.81F * substep;
        expected_y += expected_velocity * substep;
    }
    check_status(world.step({.timestep = timestep,
                             .substeps = substeps,
                             .gravity = {0.0F, -9.81F, 0.0F}}),
                 "step integration world");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(body, state), "read integrated body");
    check(near(state.linear_velocity.y, expected_velocity),
          "GPU velocity must match semi-implicit CPU reference");
    check(near(state.position.y, expected_y),
          "GPU position must match semi-implicit CPU reference");

    check_status(world.remove_rigid_body(body), "remove integration body");
    check_status(world.remove_triangle_mesh(mesh), "remove unreferenced mesh");
    RigidBodyId invalid{};
    check(world.add_rigid_body({.mesh = mesh}, invalid).code ==
              StatusCode::invalid_handle,
          "stale mesh handles must be rejected");
}

void test_generation_and_kinematics() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 1U},
                               world),
                 "create handle world");
    const TriangleMeshId mesh = add_box(world, {0.5F, 0.5F, 0.5F});
    RigidBodyId first{};
    check_status(world.add_rigid_body({.mesh = mesh}, first), "add first body");
    check_status(world.remove_rigid_body(first), "remove first body");
    check(world.set_rigid_body_state(first, {}).code == StatusCode::invalid_handle,
          "removed body handles must remain invalid");
    RigidBodyId moving{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::kinematic, .mesh = mesh}, moving),
                 "add kinematic triangle body");
    check(first.index == moving.index && first.generation != moving.generation,
          "body slots must be reused with a new generation");
    RigidBodyState target{};
    target.position = {1.0F, 2.0F, 3.0F};
    check_status(world.set_kinematic_target(moving, target),
                 "set kinematic target");
    check_status(world.step({.timestep = 0.25F, .substeps = 4U, .gravity = {}}),
                 "step kinematic target");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(moving, state),
                 "read kinematic body");
    check(near(state.position.x, 1.0F) && near(state.position.y, 2.0F) &&
              near(state.position.z, 3.0F),
          "kinematic triangle body must reach its frame target");
}

void test_batched_central_acceleration_and_interpolation_snapshot() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 1U}, world),
                 "create acceleration batch world");
    const TriangleMeshId mesh = add_box(world, {0.1F, 0.1F, 0.1F});
    RigidBodyId bodies[2]{};
    for (std::uint32_t index = 0U; index < 2U; ++index)
        check_status(world.add_rigid_body(
                         {.mesh = mesh,
                          .initial_state = {
                              .position = {static_cast<float>(index),
                                           0.0F, 0.0F}},
                          .mass = 2.0F + 2.0F * index,
                          .linear_damping = 0.0F,
                          .angular_damping = 0.0F},
                         bodies[index]),
                     "add acceleration batch body");
    check_status(world.apply_central_acceleration(
                     {bodies, 2U}, {3.0F, 0.0F, 0.0F}),
                 "apply acceleration batch");
    check_status(world.step({.timestep = 0.5F, .substeps = 1U,
                             .gravity = {}}),
                 "step acceleration batch");
    RigidBodyDeviceView view{};
    check_status(world.rigid_body_view(view),
                 "borrow interpolation snapshot");
    check(view.previous_states.size == 2U,
          "rigid view must expose prior physics-tick states");
    for (std::uint32_t index = 0U; index < 2U; ++index) {
        RigidBodyState state{};
        check_status(world.read_rigid_body_state(bodies[index], state),
                     "read accelerated body");
        check(near(state.linear_velocity.x, 1.5F) &&
                  near(state.position.x,
                       static_cast<float>(index) + 0.75F),
              "central acceleration must be mass independent");
        RigidBodyState previous{};
        check(cudaMemcpy(&previous, view.previous_states.data + index,
                         sizeof(previous), cudaMemcpyDeviceToHost) ==
                  cudaSuccess,
              "read prior interpolation state");
        check(near(previous.position.x, static_cast<float>(index)),
              "interpolation snapshot must precede latest physics tick");
    }
}

void test_floor_contact_and_async_contract() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 8U,
                                .triangle_mesh_capacity = 2U},
                               world),
                 "create contact world");
    const TriangleMeshId plane_mesh = add_plane(world);
    const TriangleMeshId box_mesh = add_box(world, {0.5F, 0.5F, 0.5F});
    RigidBodyId floor{};
    RigidBodyId box{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .mesh = plane_mesh,
                      .friction = 0.8F},
                     floor),
                 "add triangle floor");
    check_status(world.add_rigid_body(
                     {.mesh = box_mesh,
                      .initial_state = {.position = {0.0F, 0.495F, 0.0F},
                                        .linear_velocity = {0.0F, -1.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     box),
                 "add contacting box");
    FrameToken token;
    check_status(world.step_async(
                     {.timestep = 1.0F / 60.0F,
                      .substeps = 4U,
                      .gravity = {},
                      .collect_kernel_timings = true,
                      .collect_rigid_contacts = true},
                     token),
                 "enqueue asynchronous contact frame");
    check(token.pending(), "frame token must remain pending until acknowledged");
    check(world.apply_force(box, {1.0F, 0.0F, 0.0F}, {}).code == StatusCode::busy,
          "mutation must reject an unacknowledged frame");
    check_status(token.wait(), "wait for asynchronous contact frame");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(box, state), "read contact box");
    check(state.position.y >= 0.499F,
          "contact cache must work when body capacity exceeds body count");
    check(state.linear_velocity.y >= -1.0e-3F,
          "triangle mesh contact must remove inward velocity");

    const RigidContactDeviceView contact_view = world.rigid_contacts();
    check(contact_view.event_count > 0U &&
              contact_view.events.size == contact_view.event_count,
          "requested rigid contact diagnostics must be available");
    if (contact_view.event_count > 0U) {
        RigidContactEvent contact{};
        check(cudaMemcpy(&contact, contact_view.events.data, sizeof(contact),
                         cudaMemcpyDeviceToHost) == cudaSuccess,
              "download rigid contact diagnostic");
        check(std::isfinite(contact.position.y) &&
                  std::isfinite(contact.normal.y) &&
                  contact.penetration >= 0.0F &&
                  contact.normal_impulse >= 0.0F,
              "rigid contact diagnostic must contain finite solver values");
    }

    WorldStepTimings timings{};
    check_status(world.collect_step_timings(timings),
                 "collect rigid kernel timings");
    check(timings.available && timings.rigid_integration.launch_count == 4U &&
              timings.rigid_world_bounds.launch_count == 4U &&
              timings.rigid_pair_filter.launch_count == 4U &&
              timings.rigid_pair_compaction.launch_count == 4U &&
              timings.rigid_leaf_pair_generation.launch_count == 4U &&
              // Capacity eight uses face preparation, leaf evaluation,
              // candidate-order reduction, and solver initialization.
              timings.rigid_contact_evaluation.launch_count == 16U &&
              timings.rigid_contact_generation.launch_count == 32U &&
              // Event/cache load, AVBD graph preparation and coupled solve,
              // speed limits and cache publication: six per substep.
              timings.rigid_contact_solve.launch_count == 4U * 6U &&
              timings.rigid_input_clear.launch_count == 1U &&
              timings.total_gpu_milliseconds > 0.0F,
          "requested timings must report every rigid kernel launch");

    check_status(world.remove_rigid_body(box), "remove diagnostic box");
    check_status(world.remove_rigid_body(floor), "remove diagnostic floor");
    check_status(world.step({.collect_rigid_contacts = true}),
                 "step empty diagnostic world");
    check(world.rigid_contacts().event_count == 0U,
          "an empty frame must clear stale rigid contacts");
}

void test_open_two_sided_surface() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 3U,
                                .triangle_mesh_capacity = 2U},
                               world),
                 "create two-sided world");
    const TriangleMeshId surface_mesh = upload_mesh(
        world,
        {{-4.0F, 0.0F, -4.0F}, {4.0F, 0.0F, -4.0F}, {0.0F, 0.0F, 4.0F}},
        {0U, 1U, 2U}, "add open triangle");
    const TriangleMeshId probe_mesh = add_box(world, {0.2F, 0.2F, 0.2F});
    RigidBodyId surface{};
    RigidBodyId above{};
    RigidBodyId below{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = surface_mesh},
                     surface),
                 "add open surface body");
    check_status(world.add_rigid_body(
                     {.mesh = probe_mesh,
                      .initial_state = {.position = {-1.0F, 0.195F, 0.0F},
                                        .linear_velocity = {0.0F, -1.0F, 0.0F}},
                      .linear_damping = 0.0F},
                     above),
                 "add front-side probe");
    check_status(world.add_rigid_body(
                     {.mesh = probe_mesh,
                      .initial_state = {.position = {1.0F, -0.195F, 0.0F},
                                        .linear_velocity = {0.0F, 1.0F, 0.0F}},
                      .linear_damping = 0.0F},
                     below),
                 "add back-side probe");
    check_status(world.step({.timestep = 0.01F, .substeps = 2U, .gravity = {}}),
                 "resolve two-sided contacts");
    RigidBodyState front{};
    RigidBodyState back{};
    check_status(world.read_rigid_body_state(above, front), "read front probe");
    check_status(world.read_rigid_body_state(below, back), "read back probe");
    check(front.position.y >= 0.199F && front.linear_velocity.y >= -1.0e-3F,
          "an open triangle must collide from its front");
    check(back.position.y <= -0.199F && back.linear_velocity.y <= 1.0e-3F,
          "an open triangle must collide from its back");
}

void test_rotation_dynamic_coupling_and_determinism() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U},
                               world),
                 "create rotational world");
    const TriangleMeshId plane_mesh = add_plane(world);
    const TriangleMeshId tall_mesh = add_box(world, {0.3F, 0.9F, 0.3F});
    RigidBodyId floor{};
    RigidBodyId tall{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .mesh = plane_mesh,
                      .friction = 0.9F},
                     floor),
                 "add rotational floor");
    const RigidBodyState initial{
        .position = {0.0F, 1.0F, 0.0F}, .orientation = rotation_z(0.25F)};
    check_status(world.add_rigid_body(
                     {.mesh = tall_mesh,
                      .initial_state = initial,
                      .friction = 0.8F,
                      .linear_damping = 0.08F,
                      .angular_damping = 0.25F},
                     tall),
                 "add leaning mesh");
    for (int frame = 0; frame < 240; ++frame) {
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 8U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "settle leaning mesh");
    }
    RigidBodyState toppled{};
    check_status(world.read_rigid_body_state(tall, toppled), "read leaning mesh");
    check(std::fabs(toppled.orientation.z - initial.orientation.z) > 0.05F,
          "off-center triangle contacts must create angular motion");

    RigidBodyState reference{};
    for (int repetition = 0; repetition < 20; ++repetition) {
        check_status(world.set_rigid_body_state(tall, initial),
                     "reset deterministic mesh");
        for (int frame = 0; frame < 20; ++frame) {
            check_status(world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = 4U,
                                     .gravity = {0.0F, -9.81F, 0.0F}}),
                         "step deterministic mesh");
        }
        RigidBodyState result{};
        check_status(world.read_rigid_body_state(tall, result),
                     "read deterministic mesh");
        if (repetition == 0) {
            reference = result;
        } else {
            check(std::memcmp(&reference, &result, sizeof(result)) == 0,
                  "identical triangle simulations must be bit-identical");
        }
    }

    World pair_world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 1U},
                               pair_world),
                 "create dynamic pair world");
    const TriangleMeshId cube = add_box(pair_world, {0.5F, 0.5F, 0.5F});
    RigidBodyId left{};
    RigidBodyId right{};
    check_status(pair_world.add_rigid_body(
                     {.mesh = cube,
                      .initial_state = {.position = {-0.49F, 0.0F, 0.0F},
                                        .linear_velocity = {1.0F, 0.0F, 0.0F}},
                      .restitution = 0.5F,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     left),
                 "add left dynamic mesh");
    check_status(pair_world.add_rigid_body(
                     {.mesh = cube,
                      .initial_state = {.position = {0.49F, 0.0F, 0.0F},
                                        .linear_velocity = {-1.0F, 0.0F, 0.0F}},
                      .restitution = 0.5F,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     right),
                 "add right dynamic mesh");
    check_status(pair_world.step(
                     {.timestep = 1.0F / 120.0F, .substeps = 4U, .gravity = {}}),
                 "resolve dynamic mesh pair");
    RigidBodyState left_state{};
    RigidBodyState right_state{};
    check_status(pair_world.read_rigid_body_state(left, left_state),
                 "read left mesh");
    check_status(pair_world.read_rigid_body_state(right, right_state),
                 "read right mesh");
    check(left_state.linear_velocity.x < 1.0F &&
              right_state.linear_velocity.x > -1.0F,
          "dynamic triangle bodies must exchange collision impulse");
    if (!near(left_state.linear_velocity.x + right_state.linear_velocity.x, 0.0F, 1.0e-4F))
        std::cerr << "pair momentum: vx=" << left_state.linear_velocity.x << ','
                  << right_state.linear_velocity.x << '\n';
    check(near(left_state.linear_velocity.x + right_state.linear_velocity.x,
               0.0F, 1.0e-4F),
          "dynamic triangle collision must preserve linear momentum");
}

// Overlapping siblings must not fight a weld through their common parent.
// Exercise runtime topology changes as well as the ground-contact exception.
void test_fixed_cluster_collision_filter() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 5U,
                                .rigid_constraint_capacity = 2U,
                                .triangle_mesh_capacity = 2U}, world),
                 "create welded collision-filter world");
    const auto plane = add_plane(world);
    const auto box = add_box(world, {0.2F, 0.2F, 0.2F});
    RigidBodyId floor{}, dummy{}, parent{}, left{}, right{};
    check_status(world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = plane}, floor), "add weld floor");
    check_status(world.add_rigid_body(
        {.mesh = box, .initial_state = {.position = {20.0F, 2.0F, 0.0F}}}, dummy),
        "add compaction body");
    const RigidBodyState parent_state{.position = {0.0F, 1.0F, 0.0F}};
    const RigidBodyState left_state{.position = {-0.1F, 0.199F, 0.0F}};
    const RigidBodyState right_state{.position = {0.1F, 0.199F, 0.0F}};
    check_status(world.add_rigid_body(
        {.mesh = box, .initial_state = parent_state, .mass = 100.0F}, parent),
        "add weld parent");
    check_status(world.add_rigid_body(
        {.mesh = box, .initial_state = left_state}, left), "add weld left");
    check_status(world.add_rigid_body(
        {.mesh = box, .initial_state = right_state}, right), "add weld right");
    RigidConstraintOptions left_options{
        .body_a = parent, .body_b = left,
        .local_anchor_a = {-0.1F, -0.801F, 0.0F}};
    RigidConstraintOptions right_options{
        .body_a = parent, .body_b = right,
        .local_anchor_a = {0.1F, -0.801F, 0.0F}};
    RigidConstraintId left_joint{}, right_joint{};
    check_status(world.add_rigid_constraint(left_options, left_joint), "weld left");
    check_status(world.add_rigid_constraint(right_options, right_joint), "weld right");
    const auto reset = [&](float lift = 0.0F) {
        auto a = parent_state, b = left_state, c = right_state;
        a.position.y += lift; b.position.y += lift; c.position.y += lift;
        check_status(world.set_rigid_body_state(parent, a), "reset weld parent");
        check_status(world.set_rigid_body_state(left, b), "reset weld left");
        check_status(world.set_rigid_body_state(right, c), "reset weld right");
    };
    const auto step = [&] {
        check_status(world.step({.timestep = 1.0F / 60.0F, .substeps = 1U,
                                 .gravity = {}, .collect_rigid_contacts = true}),
                     "step welded collision filter");
    };
    const auto contacts = [&](bool expect_siblings) {
        step();
        const auto view = world.rigid_contacts();
        std::vector<RigidContactEvent> events(view.event_count);
        if (!events.empty()) check(cudaMemcpy(events.data(), view.events.data,
            events.size() * sizeof(RigidContactEvent), cudaMemcpyDeviceToHost) ==
            cudaSuccess, "read welded contacts");
        bool siblings = false, ground = false;
        for (const auto &event : events) {
            siblings |= (event.body == left && event.collider == right) ||
                        (event.body == right && event.collider == left);
            ground |= event.body == floor || event.collider == floor;
        }
        check(siblings == expect_siblings, "weld topology must control sibling contacts");
        check(ground, "weld filtering must preserve external ground contacts");
    };
    // In free space, conflicting internal contacts previously accelerated an
    // initially motionless assembly even with gravity and external forces off.
    reset(2.0F);
    for (int frame = 0; frame < 12; ++frame) step();
    for (auto body : {parent, left, right}) {
        RigidBodyState state{};
        check_status(world.read_rigid_body_state(body, state), "read free weld");
        const auto v = state.linear_velocity, w = state.angular_velocity;
        check(v.x*v.x + v.y*v.y + v.z*v.z + w.x*w.x + w.y*w.y + w.z*w.z < 1.0e-6F,
              "overlapping welded siblings must not generate kinetic energy");
    }
    reset(); contacts(false);
    check_status(world.remove_rigid_body(dummy), "compact bodies with live welds");
    reset(); contacts(false);
    right_options.enabled = false;
    check_status(world.update_rigid_constraint(right_joint, right_options), "disable weld");
    reset(); contacts(true);
    right_options.enabled = true;
    right_options.disable_collisions = false;
    check_status(world.update_rigid_constraint(right_joint, right_options), "allow weld collisions");
    reset(); contacts(true);
    right_options.disable_collisions = true;
    right_options.type = RigidConstraintType::point;
    check_status(world.update_rigid_constraint(right_joint, right_options), "articulate weld");
    reset(); contacts(true);
    right_options.type = RigidConstraintType::fixed;
    right_options.breaking_impulse_threshold = 0.001F;
    check_status(world.update_rigid_constraint(right_joint, right_options), "make breakable weld");
    reset(2.0F);
    auto pulled = right_state;
    pulled.position.y += 2.0F;
    pulled.linear_velocity.x = 1.0F;
    check_status(world.set_rigid_body_state(right, pulled), "load breakable weld");
    step();
    RigidConstraintState broken{};
    check_status(world.read_rigid_constraint_state(right_joint, broken), "read broken weld");
    check(broken.broken, "test weld must break");
    reset(); contacts(true);
    check_status(world.remove_rigid_constraint(right_joint), "remove broken weld");
    right_options.breaking_impulse_threshold = 0.0F;
    check_status(world.add_rigid_constraint(right_options, right_joint), "reweld reused slot");
    reset(); contacts(false);
    check_status(world.remove_rigid_constraint(right_joint), "remove live weld");
    reset(); contacts(true);
}

void test_fixed_cluster_ground_support() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 5U,
                                .rigid_constraint_capacity = 3U,
                                .triangle_mesh_capacity = 3U},
                               world),
                 "create fixed ground-support world");
    const TriangleMeshId plane_mesh = add_plane(world);
    const TriangleMeshId large_mesh = add_box(world, {0.5F, 0.5F, 0.5F});
    const TriangleMeshId support_mesh = add_box(world, {0.2F, 0.2F, 0.2F});
    RigidBodyId floor{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .mesh = plane_mesh,
                      .friction = 0.8F,
                      .collision_margin = 0.006F},
                     floor),
                 "add fixed-cluster floor");
    const RigidBodyState large_initial{.position = {0.0F, 0.9F, 0.0F}};
    RigidBodyId large{};
    check_status(world.add_rigid_body(
                     {.mesh = large_mesh,
                      .initial_state = large_initial,
                      .mass = 100.0F,
                      .friction = 0.8F,
                      .collision_margin = 0.006F},
                     large),
                 "add fixed-cluster large body");
    constexpr std::array<Vec3, 3U> support_positions{{
        {-0.5F, 0.2F, -0.4F},
        {0.5F, 0.2F, -0.4F},
        {0.0F, 0.2F, 0.5F},
    }};
    std::array<RigidBodyId, support_positions.size()> supports{};
    for (std::size_t index = 0U; index < supports.size(); ++index) {
        check_status(world.add_rigid_body(
                         {.mesh = support_mesh,
                          .initial_state = {.position =
                                                support_positions[index]},
                          .mass = 1.0F,
                          .friction = 0.8F,
                          .collision_margin = 0.006F},
                         supports[index]),
                     "add fixed-cluster support");
        RigidConstraintId constraint{};
        check_status(world.add_rigid_constraint(
                         {.type = RigidConstraintType::fixed,
                          .body_a = large,
                          .body_b = supports[index],
                          .local_anchor_a = {
                              support_positions[index].x -
                                  large_initial.position.x,
                              support_positions[index].y -
                                  large_initial.position.y,
                              support_positions[index].z -
                                  large_initial.position.z},
                          .enabled = true,
                          .disable_collisions = true,
                          .solver_iterations = 32U},
                         constraint),
                     "fix ground support to large body");
    }

    float maximum_late_angular_speed = 0.0F;
    float maximum_late_support_vertical_speed = 0.0F;
    float minimum_support_clearance = 1.0F;
    for (int frame = 0; frame < 600; ++frame) {
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 8U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "settle fixed ground-support cluster");
        if (frame < 300) continue;
        RigidBodyState large_state{};
        check_status(world.read_rigid_body_state(large, large_state),
                     "read fixed-cluster large body");
        maximum_late_angular_speed = std::max(
            maximum_late_angular_speed,
            std::sqrt(large_state.angular_velocity.x *
                          large_state.angular_velocity.x +
                      large_state.angular_velocity.y *
                          large_state.angular_velocity.y +
                      large_state.angular_velocity.z *
                          large_state.angular_velocity.z));
        for (RigidBodyId support : supports) {
            RigidBodyState support_state{};
            check_status(world.read_rigid_body_state(support, support_state),
                         "read fixed-cluster support");
            const auto q = support_state.orientation;
            // The lowest corner of the rotated box must stay above the plane.
            const float vertical_extent = 0.2F * (
                std::fabs(2.0F * (q.x * q.y + q.w * q.z)) +
                std::fabs(1.0F - 2.0F * (q.x * q.x + q.z * q.z)) +
                std::fabs(2.0F * (q.y * q.z - q.w * q.x)));
            minimum_support_clearance = std::min(
                minimum_support_clearance,
                support_state.position.y - vertical_extent);
            maximum_late_support_vertical_speed = std::max(
                maximum_late_support_vertical_speed,
                std::fabs(support_state.linear_velocity.y));
        }
    }
    check(maximum_late_angular_speed < 0.02F &&
              maximum_late_support_vertical_speed < 0.1F,
          "fixed cluster must settle on multiple ground supports");
    check(minimum_support_clearance > -0.003F,
          "light fixed supports must carry the heavy body without sinking");
    if (maximum_late_angular_speed >= 0.02F ||
        maximum_late_support_vertical_speed >= 0.1F ||
        minimum_support_clearance <= -0.003F) {
        std::cerr << "fixed support angular_speed="
                  << maximum_late_angular_speed
                  << " support_vertical_speed="
                  << maximum_late_support_vertical_speed
                  << " minimum_clearance=" << minimum_support_clearance << '\n';
    }
}

void test_point_constraint_ground_support() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 6U,
                                .rigid_constraint_capacity = 4U,
                                .triangle_mesh_capacity = 3U},
                               world),
                 "create point ground-support world");
    const TriangleMeshId plane_mesh = add_plane(world);
    const TriangleMeshId anchor_mesh = add_box(world, {0.1F, 0.1F, 0.1F});
    const TriangleMeshId body_mesh = add_box(world, {0.4F, 0.4F, 0.4F});
    RigidBodyId floor{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .mesh = plane_mesh,
                      .friction = 0.8F,
                      .collision_margin = 0.006F},
                     floor),
                 "add point-support floor");
    RigidBodyId anchor{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .mesh = anchor_mesh,
                      .initial_state = {.position = {0.0F, 1.25F, 0.0F}}},
                     anchor),
                 "add point-support anchor");
    constexpr float radius = 1.2F;
    constexpr float height = 0.405F;
    constexpr std::array<Vec3, 4U> positions{{
        {radius, height, 0.0F}, {-radius, height, 0.0F},
        {0.0F, height, radius}, {0.0F, height, -radius},
    }};
    constexpr std::array<Vec3, 4U> velocities{{
        {0.0F, -3.0F, 0.0F}, {0.0F, 0.0F, 3.0F},
        {-3.0F, 0.0F, 0.0F}, {-3.0F, 0.0F, 0.0F},
    }};
    constexpr std::array<Vec3, 4U> angular_velocities{{
        {0.0F, 0.0F, -2.5F}, {}, {}, {},
    }};
    std::array<RigidBodyId, positions.size()> bodies{};
    for (std::size_t index = 0U; index < bodies.size(); ++index) {
        check_status(world.add_rigid_body(
                         {.mesh = body_mesh,
                          .initial_state = {.position = positions[index],
                                            .linear_velocity = velocities[index],
                                            .angular_velocity =
                                                angular_velocities[index]},
                          .mass = 1.0F,
                          .friction = 0.8F,
                          .linear_damping = 0.01F,
                          .angular_damping = 0.01F,
                          .collision_margin = 0.006F},
                         bodies[index]),
                     "add point-supported body");
        RigidConstraintId constraint{};
        check_status(world.add_rigid_constraint(
                         {.type = RigidConstraintType::point,
                          .body_a = anchor,
                          .body_b = bodies[index],
                          .local_anchor_a = {0.0F, height - 1.25F, 0.0F},
                          .local_anchor_b = {-positions[index].x, 0.0F,
                                             -positions[index].z},
                          .enabled = true,
                          .solver_iterations = 16U},
                         constraint),
                     "attach point-supported body");
    }

    float minimum_center_height = height;
    for (int frame = 0; frame < 600; ++frame) {
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 8U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "step point ground-support collision");
        for (RigidBodyId body : bodies) {
            RigidBodyState state{};
            check_status(world.read_rigid_body_state(body, state),
                         "read point-supported body");
            minimum_center_height = std::min(
                minimum_center_height, state.position.y);
        }
    }
    check(minimum_center_height > 0.2F,
          "point-constrained collisions must not tunnel through the ground");
    if (minimum_center_height <= 0.2F)
        std::cerr << "point support minimum_center_height="
                  << minimum_center_height << '\n';
}

void test_high_speed_swept_triangle_contact() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U},
                               world),
                 "create swept-contact world");
    const TriangleMeshId surface = upload_mesh(
        world,
        {{-5.0F, 0.0F, -5.0F}, {5.0F, 0.0F, -5.0F},
         {5.0F, 0.0F, 5.0F}, {-5.0F, 0.0F, 5.0F}},
        {0U, 2U, 1U, 0U, 3U, 2U}, "add swept surface");
    const TriangleMeshId projectile_mesh =
        add_box(world, {0.1F, 0.1F, 0.1F});
    RigidBodyId floor{};
    RigidBodyId projectile{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = surface},
                     floor),
                 "add swept floor");
    const RigidBodyState initial{
        .position = {0.0F, 1.5F, 0.0F},
        .linear_velocity = {0.0F, -120.0F, 0.0F}};
    check_status(world.add_rigid_body(
                     {.mesh = projectile_mesh,
                      .initial_state = initial,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F,
                      .maximum_linear_speed = 200.0F},
                     projectile),
                 "add swept projectile");

    RigidBodyState reference{};
    for (int repetition = 0; repetition < 20; ++repetition) {
        check_status(world.set_rigid_body_state(projectile, initial),
                     "reset swept projectile");
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 1U,
                                 .gravity = {}}),
                     "step swept projectile");
        RigidBodyState result{};
        check_status(world.read_rigid_body_state(projectile, result),
                     "read swept projectile");
        check(result.position.y >= 0.099F &&
                  result.linear_velocity.y >= -1.0e-3F,
              "one swept substep must stop a fast mesh above the surface");
        if (repetition == 0) {
            reference = result;
        } else {
            check(std::memcmp(&reference, &result, sizeof(result)) == 0,
                  "swept triangle contacts must be bit-identical");
        }
    }
}

void test_swept_contact_when_leaf_cache_overflows(std::uint32_t body_capacity) {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = body_capacity,
                                .triangle_mesh_capacity = 2U}, world),
                 "create overflow contact world");
    std::vector<std::uint32_t> floor_indices;
    // Exceed both the ordinary 512-entry cache and the 4096-entry small-world
    // cache (each floor leaf overlaps four projectile leaves).
    floor_indices.reserve(4100U * 3U);
    for (std::uint32_t index = 0U; index < 4100U; ++index) {
        floor_indices.insert(floor_indices.end(), {0U, 1U, 2U});
    }
    const TriangleMeshId floor_mesh = upload_mesh(
        world, {{-5.0F, 0.0F, -5.0F}, {0.0F, 0.0F, 5.0F},
                {5.0F, 0.0F, -5.0F}}, floor_indices,
        "add duplicated floor exceeding leaf cache");
    const TriangleMeshId projectile_mesh =
        add_box(world, {0.1F, 0.1F, 0.1F});
    RigidBodyId floor{};
    RigidBodyId projectile{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = floor_mesh},
                     floor),
                 "add overflow floor");
    check_status(world.add_rigid_body(
                     {.mesh = projectile_mesh,
                      .initial_state = {
                          .position = {0.0F, 1.5F, 0.0F},
                          .linear_velocity = {0.0F, -120.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F,
                      .maximum_linear_speed = 200.0F},
                     projectile),
                 "add overflow projectile");
    check_status(world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 1U, .gravity = {}}),
                 "step overflow contact world");
    RigidBodyState result{};
    check_status(world.read_rigid_body_state(projectile, result),
                 "read overflow projectile");
    if (result.position.y < 0.099F || result.linear_velocity.y < -1.0F) {
        std::cerr << "overflow projectile: y=" << result.position.y
                  << " velocity=" << result.linear_velocity.y << '\n';
    }
    check(result.position.y >= 0.099F &&
              result.linear_velocity.y >= -1.0F,
          "overflow fallback must preserve swept collision");
    check_status(world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 1U, .gravity = {}}),
                 "step overflow contact world again");
    check_status(world.read_rigid_body_state(projectile, result),
                 "read overflow projectile after follow-up step");
    check(result.position.y >= 0.099F,
          "overflow projectile must remain above the surface");
}

void test_small_rest_offset_speculative_contact() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U}, world),
                 "create rest-offset contact world");
    const TriangleMeshId floor_mesh = add_plane(world);
    const TriangleMeshId box_mesh = add_box(world, {0.5F, 0.5F, 0.5F});
    RigidBodyId floor{};
    RigidBodyId box{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .mesh = floor_mesh,
                      .collision_margin = 0.02F},
                     floor),
                 "add rest-offset floor");
    check_status(world.add_rigid_body(
                     {.mesh = box_mesh,
                      .initial_state = {
                          .position = {0.0F, 0.511F, 0.0F},
                          .linear_velocity = {0.0F, -1.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F,
                      .collision_margin = 0.02F},
                     box),
                 "add rest-offset box");
    constexpr float timestep = 0.005F;
    const StepOptions step{.timestep = timestep,
                           .substeps = 1U,
                           .gravity = {},
                           .collect_rigid_contacts = true};
    check_status(world.step(step), "approach rest-offset contact");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(box, state),
                 "read approaching box");
    check(near(state.position.y, 0.506F, 2.0e-4F) &&
              state.linear_velocity.y < -0.95F,
          "speculative contact must approach the small rest offset");
    check(world.rigid_contacts().event_count > 0U,
          "closing surface must become a speculative contact");

    check_status(world.step(step), "reach rest-offset contact");
    check_status(world.read_rigid_body_state(box, state),
                 "read touching box");
    check(near(state.position.y, 0.501F, 5.0e-4F) &&
              state.linear_velocity.y >= -2.0e-4F,
          "contact must stop at the small rest offset");
    check(world.rigid_contacts().event_count > 0U,
          "touching surfaces must retain one contact");

    check_status(world.step(step), "maintain rest-offset contact");
    check_status(world.read_rigid_body_state(box, state),
                 "read resting box");
    check(near(state.position.y, 0.501F, 5.0e-4F),
          "resolved surfaces must remain near the small rest offset");

    state.linear_velocity = {0.0F, 1.0F, 0.0F};
    check_status(world.set_rigid_body_state(box, state),
                 "release touching box");
    check_status(world.step(step), "separate rest-offset contact");
    check_status(world.read_rigid_body_state(box, state),
                 "read separating box");
    check(state.position.y > 0.505F && state.linear_velocity.y > 0.99F &&
              world.rigid_contacts().event_count == 0U,
          "a separating surface must release without a contact impulse");
}

void test_parallel_contact_coloring(std::uint32_t body_count) {
    using namespace parallel_mater;
    const std::uint32_t overlapping_bodies =
        body_count < 36U ? body_count : 36U;
    World world;
    check_status(World::create({.rigid_body_capacity = body_count,
                                .triangle_mesh_capacity = 1U,
                                .contact_capacity = 1U}, world),
                 "create contact-color overflow world");
    const TriangleMeshId mesh = add_box(world, {0.1F, 0.1F, 0.1F});
    std::vector<RigidBodyId> bodies;
    bodies.reserve(body_count);
    for (std::uint32_t index = 0U; index < body_count; ++index) {
        RigidBodyId body{};
        const bool dynamic = index < overlapping_bodies;
        check_status(world.add_rigid_body(
                         {.motion = dynamic ? MotionType::dynamic
                                            : MotionType::static_body,
                          .mesh = mesh,
                          .initial_state = {
                              .position = dynamic
                                  ? Vec3{}
                                  : Vec3{100.0F + index * 10.0F, 0.0F, 0.0F}}},
                         body),
                     "add contact-color overflow body");
        bodies.push_back(body);
    }
    RigidBodyState reference{};
    for (int repetition = 0; repetition < 2; ++repetition) {
        for (std::uint32_t index = 0U; index < overlapping_bodies; ++index) {
            check_status(world.set_rigid_body_state(bodies[index], {}),
                         "reset contact-color overflow body");
        }
        check_status(world.step({.timestep = 1.0F / 240.0F,
                                 .substeps = 1U,
                                 .gravity = {},
                                 .collect_rigid_contacts = repetition != 0}),
                     "step contact-color overflow world");
        RigidBodyState result{};
        check_status(world.read_rigid_body_state(bodies[0], result),
                     "read contact-color overflow body");
        check(std::isfinite(result.position.x) &&
                  std::isfinite(result.position.y) &&
                  std::isfinite(result.position.z),
              "high-degree contact graph must remain finite");
        if (repetition == 0) {
            reference = result;
        } else {
            check(world.rigid_contacts().event_count == 1U,
                  "parallel contacts must respect diagnostic capacity");
            check(std::memcmp(&reference, &result, sizeof(result)) == 0,
                  "contact-color overflow diagnostics must preserve motion");
        }
    }
}

void test_invalid_triangle_indices() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 1U,
                                .triangle_mesh_capacity = 1U},
                               world),
                 "create invalid mesh world");
    Vec3 vertices[3]{{}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}};
    std::uint32_t indices[3]{0U, 1U, 3U};
    Vec3 *device_vertices = nullptr;
    std::uint32_t *device_indices = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&device_vertices), sizeof(vertices));
    cudaMalloc(reinterpret_cast<void **>(&device_indices), sizeof(indices));
    cudaMemcpy(device_vertices, vertices, sizeof(vertices), cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices, indices, sizeof(indices), cudaMemcpyHostToDevice);
    TriangleMeshId mesh{};
    const Status status = world.add_triangle_mesh(
        {device_vertices, 3U}, {device_indices, 3U}, mesh);
    check(status.code == StatusCode::invalid_argument,
          "out-of-range triangle indices must be rejected");
    cudaFree(device_indices);
    cudaFree(device_vertices);
}

void test_opt_in_physics_debug_capture() {
    using namespace parallel_mater;
    World disabled;
    check_status(World::create({.rigid_body_capacity = 1U,
                                .triangle_mesh_capacity = 1U}, disabled),
                 "create debug-disabled world");
    PhysicsDebugFrameView disabled_frame{};
    check(disabled.physics_debug_frame(disabled_frame).code ==
              StatusCode::not_supported,
          "physics capture must be opt in");

    World world;
    check_status(World::create({.rigid_body_capacity = 1U,
                                .triangle_mesh_capacity = 1U,
                                .physics_debug = {.frame_capacity = 2U}},
                               world),
                 "create debug-enabled world");
    const TriangleMeshId mesh = add_box(world, {0.25F, 0.25F, 0.25F});
    RigidBodyId body{};
    check_status(world.add_rigid_body(
                     {.mesh = mesh,
                      .mass = 2.0F,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     body),
                 "add debug capture body");
    for (std::uint32_t frame = 1U; frame <= 3U; ++frame) {
        const Vec3 force{static_cast<float>(frame), 2.0F, -3.0F};
        check_status(world.apply_force(body, force, {}),
                     "apply captured force");
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 1U,
                                 .gravity = {}}),
                     "step debug capture world");
        PhysicsDebugFrameView latest{};
        check_status(world.physics_debug_frame(latest),
                     "borrow latest physics debug frame");
        check(latest.frame_index == frame && latest.rigid_bodies.size == 1U,
              "latest debug frame must match completed frame");
        if (latest.rigid_bodies.size == 1U) {
            check(near(latest.rigid_bodies.data[0].applied_force.x,
                       static_cast<float>(frame)),
                  "debug capture must preserve applied rigid force");
        }
    }
    PhysicsDebugCapture capture{};
    check_status(world.copy_physics_debug_capture(capture),
                 "copy rolling physics debug capture");
    check(capture.frames.size() == 2U &&
              capture.frames[0].frame_index == 2U &&
              capture.frames[1].frame_index == 3U,
          "debug capture ring must copy in chronological order");
    RigidBodyDeviceView view{};
    check_status(world.rigid_body_view(view), "borrow debug rigid view");
    check(view.applied_forces.size == 1U &&
              near(view.applied_forces.data[0].x, 3.0F),
          "debug rigid view must expose last frame input force");
}

parallel_mater::RigidBodyState exercise_constraint(
    parallel_mater::RigidConstraintOptions constraint,
    parallel_mater::RigidBodyState dynamic_state,
    parallel_mater::RigidConstraintState *constraint_state = nullptr) {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .rigid_constraint_capacity = 1U,
                                .triangle_mesh_capacity = 1U}, world),
                 "create rigid constraint world");
    const TriangleMeshId mesh = add_box(world, {0.1F, 0.1F, 0.1F});
    RigidBodyId anchor{};
    RigidBodyId body{};
    check_status(world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = mesh}, anchor),
        "add constraint anchor");
    check_status(world.add_rigid_body(
        {.mesh = mesh, .initial_state = dynamic_state,
         .linear_damping = 0.0F, .angular_damping = 0.0F}, body),
        "add constrained body");
    constraint.body_a = anchor;
    constraint.body_b = body;
    RigidConstraintId id{};
    check_status(world.add_rigid_constraint(constraint, id),
                 "add rigid constraint");
    check(world.remove_rigid_body(body).code == StatusCode::invalid_argument,
          "constraint must retain both rigid bodies");
    check_status(world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 8U, .gravity = {}}),
                 "step rigid constraint");
    RigidBodyState output{};
    check_status(world.read_rigid_body_state(body, output),
                 "read constrained body");
    if (constraint_state != nullptr)
        check_status(world.read_rigid_constraint_state(id, *constraint_state),
                     "read rigid constraint");
    check_status(world.remove_rigid_constraint(id), "remove rigid constraint");
    RigidConstraintState stale{};
    check(world.read_rigid_constraint_state(id, stale).code ==
              StatusCode::invalid_handle,
          "removed rigid constraint handle must become stale");
    check_status(world.remove_rigid_body(body),
                 "remove unreferenced constrained body");
    return output;
}

void test_rigid_constraint_types() {
    using namespace parallel_mater;
    const RigidBodyState translating{
        .linear_velocity = {2.0F, 2.0F, 0.0F}};
    RigidBodyState fixed = exercise_constraint(
        {.type = RigidConstraintType::fixed}, translating);
    check(std::fabs(fixed.linear_velocity.x) < 0.05F &&
              std::fabs(fixed.linear_velocity.y) < 0.05F,
          "fixed constraint must lock translation");

    RigidBodyState point = exercise_constraint(
        {.type = RigidConstraintType::point},
        {.linear_velocity = {2.0F, 0.0F, 0.0F},
         .angular_velocity = {0.0F, 0.0F, 2.0F}});
    check(std::fabs(point.linear_velocity.x) < 0.05F &&
              point.angular_velocity.z > 1.0F,
          "point constraint must lock its anchor and leave rotation free");

    RigidBodyState hinge = exercise_constraint(
        {.type = RigidConstraintType::hinge,
         .angular_limits = {.axes = rigid_constraint_axis_z,
                            .lower = {0.0F, 0.0F, -0.785398F},
                            .upper = {0.0F, 0.0F, 0.785398F}}},
        {.angular_velocity = {2.0F, 0.0F, 2.0F}});
    check(std::fabs(hinge.angular_velocity.x) < 0.05F &&
              hinge.angular_velocity.z > 1.0F,
          "hinge constraint must retain only its limited Z rotation");

    RigidBodyState slider = exercise_constraint(
        {.type = RigidConstraintType::slider,
         .linear_limits = {.axes = rigid_constraint_axis_x,
                           .lower = {-1.0F, 0.0F, 0.0F},
                           .upper = {1.0F, 0.0F, 0.0F}}}, translating);
    check(slider.linear_velocity.x > 1.0F &&
              std::fabs(slider.linear_velocity.y) < 0.05F,
          "slider constraint must allow only local X translation");

    RigidBodyState piston = exercise_constraint(
        {.type = RigidConstraintType::piston,
         .linear_limits = {.axes = rigid_constraint_axis_x,
                           .lower = {-1.0F, 0.0F, 0.0F},
                           .upper = {1.0F, 0.0F, 0.0F}}},
        {.linear_velocity = {2.0F, 2.0F, 0.0F},
         .angular_velocity = {2.0F, 2.0F, 0.0F}});
    check(piston.linear_velocity.x > 1.0F &&
              std::fabs(piston.linear_velocity.y) < 0.05F &&
              piston.angular_velocity.x > 1.0F &&
              std::fabs(piston.angular_velocity.y) < 0.05F,
          "piston constraint must allow local X translation and rotation");

    RigidBodyState generic = exercise_constraint(
        {.type = RigidConstraintType::generic,
         .linear_limits = {.axes = rigid_constraint_axis_y}}, translating);
    check(generic.linear_velocity.x > 1.0F &&
              std::fabs(generic.linear_velocity.y) < 0.05F,
          "generic constraint must lock only authored axes");

    RigidBodyState spring = exercise_constraint(
        {.type = RigidConstraintType::generic_spring,
         .linear_springs = {.axes = rigid_constraint_axis_x,
                            .stiffness = {40.0F, 0.0F, 0.0F},
                            .damping = {2.0F, 0.0F, 0.0F}}},
        {.position = {1.0F, 0.0F, 0.0F}});
    check(spring.linear_velocity.x < -0.05F,
          "generic spring must pull displaced bodies toward equilibrium");

    constexpr float spring_angle = 0.1F;
    RigidBodyState angular_spring = exercise_constraint(
        {.type = RigidConstraintType::generic_spring,
         .angular_limits = {.axes = rigid_constraint_axis_z,
                            .lower = {0.0F, 0.0F, -0.5F},
                            .upper = {0.0F, 0.0F, 0.5F}},
         .angular_springs = {.axes = rigid_constraint_axis_z,
                             .stiffness = {0.0F, 0.0F, 1.0F},
                             .damping = {0.0F, 0.0F, 0.01F}}},
        {.orientation = {0.0F, 0.0F, std::sin(spring_angle * 0.5F),
                         std::cos(spring_angle * 0.5F)},
         .angular_velocity = {0.0F, 0.0F, 2.0F}});
    check(angular_spring.angular_velocity.z > 1.0F,
          "generic angular spring must remain compliant under angular motion");

    constexpr float outside_limit_angle = 0.5F;
    RigidBodyState limited_spring = exercise_constraint(
        {.type = RigidConstraintType::generic_spring,
         .angular_limits = {.axes = rigid_constraint_axis_z,
                            .lower = {0.0F, 0.0F, -0.2F},
                            .upper = {0.0F, 0.0F, 0.2F}},
         .angular_springs = {.axes = rigid_constraint_axis_z}},
        {.orientation = {0.0F, 0.0F,
                         std::sin(outside_limit_angle * 0.5F),
                         std::cos(outside_limit_angle * 0.5F)}});
    const float limited_angle = 2.0F * std::atan2(
        std::fabs(limited_spring.orientation.z),
        std::fabs(limited_spring.orientation.w));
    if (!(limited_angle < outside_limit_angle - 0.05F))
        std::cerr << "angular limit: angle=" << limited_angle << '\n';
    check(limited_angle < outside_limit_angle - 0.05F,
          "generic angular spring must also enforce its angular limit");

    RigidBodyState motor = exercise_constraint(
        {.type = RigidConstraintType::motor,
         .motor = {.angular_enabled = true,
                   .angular_target_velocity = 4.0F,
                   .angular_maximum_impulse = 10.0F}}, {});
    check(motor.angular_velocity.x > 1.0F,
          "motor constraint must drive local X angular velocity");

    RigidConstraintState broken{};
    (void)exercise_constraint(
        {.type = RigidConstraintType::fixed,
         .breaking_impulse_threshold = 0.0001F},
        {.linear_velocity = {5.0F, 0.0F, 0.0F}}, &broken);
    check(broken.broken && !broken.enabled && broken.applied_impulse > 0.0F,
          "breaking threshold must disable an overloaded constraint");
}

void test_piston_free_twist_alignment() {
    using namespace parallel_mater;
    constexpr float pi = 3.14159265358979323846F;
    // Both sides of the quaternion branch cut, both signs, and a freely
    // spinning body. A tiny swing error must decay independently of X twist.
    for (const float degrees : {0.0F, 90.0F, 179.0F, 180.0F, 181.0F, 270.0F}) {
        for (const float sign : {-1.0F, 1.0F}) {
            World world;
            check_status(World::create({.rigid_body_capacity = 2U,
                .rigid_constraint_capacity = 1U,
                .triangle_mesh_capacity = 1U}, world), "create piston twist world");
            const auto mesh = add_box(world, {0.3F, 0.2F, 0.1F});
            RigidBodyId anchor{}, body{};
            check_status(world.add_rigid_body(
                {.motion = MotionType::static_body, .mesh = mesh}, anchor),
                "add piston twist anchor");
            const float half = degrees * pi / 360.0F;
            const float c = std::cos(0.00005F), s = std::sin(0.00005F);
            const Quaternion q{sign * c * std::sin(half),
                sign * s * std::cos(half), -sign * s * std::sin(half),
                sign * c * std::cos(half)};
            check_status(world.add_rigid_body({.mesh = mesh,
                .initial_state = {.orientation = q,
                                  .angular_velocity = {2.0F, 0.0F, 0.0F}},
                .linear_damping = 0.0F, .angular_damping = 0.0F}, body),
                "add freely spinning piston");
            RigidConstraintId joint{};
            check_status(world.add_rigid_constraint({.type = RigidConstraintType::piston,
                .body_a = anchor, .body_b = body,
                .breaking_impulse_threshold = sign < 0.0F ? 1000000.0F : 0.0F,
                .solver_iterations = 8U}, joint),
                "add piston twist joint");
            float maximum_swing = 0.0F, maximum_speed = 0.0F;
            RigidBodyState state{};
            for (int frame = 0; frame < 240; ++frame) {
                check_status(world.step({.timestep = 1.0F / 60.0F,
                    .substeps = 4U, .gravity = {}}), "step piston full revolution");
                check_status(world.read_rigid_body_state(body, state),
                    "read freely spinning piston");
                maximum_swing = std::max(maximum_swing,
                    2.0F * std::asin(std::min(1.0F,
                        std::hypot(state.orientation.y, state.orientation.z))));
                maximum_speed = std::max(maximum_speed,
                    std::sqrt(state.angular_velocity.x * state.angular_velocity.x +
                              state.angular_velocity.y * state.angular_velocity.y +
                              state.angular_velocity.z * state.angular_velocity.z));
                check(std::isfinite(state.orientation.w), "piston state must remain finite");
            }
            check(maximum_swing < 0.00011F,
                "piston swing must decay at any free-twist angle or quaternion sign");
            check(maximum_speed < 2.001F && std::fabs(state.angular_velocity.x - 2.0F) < 0.001F,
                "piston alignment must not add energy or damp permitted spin");
        }
    }
}

void test_rigid_constraint_toggle() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
                                .rigid_constraint_capacity = 1U,
                                .triangle_mesh_capacity = 1U}, world),
                 "create constraint toggle world");
    const TriangleMeshId mesh = add_box(world, {0.1F, 0.1F, 0.1F});
    RigidBodyId anchor{}, body{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = mesh}, anchor),
                 "add toggle anchor");
    check_status(world.add_rigid_body(
                     {.mesh = mesh,
                      .initial_state = {
                          .position = {1.0F, 0.0F, 0.0F},
                          .linear_velocity = {2.0F, 0.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F}, body),
                 "add toggle body");
    RigidConstraintOptions options{.type = RigidConstraintType::fixed,
                                   .body_a = anchor,
                                   .body_b = body,
                                   .local_anchor_b = {-1.0F, 0.0F, 0.0F},
                                   .enabled = false};
    RigidConstraintId constraint{};
    check_status(world.add_rigid_constraint(options, constraint),
                 "add disabled constraint");
    check_status(world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 4U, .gravity = {}}),
                 "step released constraint");
    RigidBodyState released{};
    check_status(world.read_rigid_body_state(body, released),
                 "read released constraint body");
    check(released.linear_velocity.x > 1.9F,
          "disabled constraint must leave its body released");
    options.enabled = true;
    options.local_anchor_b = {-released.position.x, -released.position.y,
                              -released.position.z};
    check_status(world.update_rigid_constraint(constraint, options),
                 "enable constraint");
    for (int frame = 0; frame < 30; ++frame)
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 4U, .gravity = {}}),
                     "step enabled constraint");
    RigidBodyState glued{};
    check_status(world.read_rigid_body_state(body, glued),
                 "read glued constraint body");
    check(std::fabs(glued.linear_velocity.x) < 0.05F,
          "updated fixed constraint must glue the body");
    RigidConstraintState state{};
    check_status(world.read_rigid_constraint_state(constraint, state),
                 "read toggled constraint");
    check(state.enabled && !state.broken,
          "enabled constraint state must be observable");
}

void test_static_axial_guide_lifecycle() {
    using namespace parallel_mater;
    for (const auto type : {RigidConstraintType::piston, RigidConstraintType::slider}) {
        for (const bool reverse : {false, true}) {
            World world;
            check_status(World::create({.rigid_body_capacity = 2U,
                .rigid_constraint_capacity = 1U, .triangle_mesh_capacity = 1U}, world),
                "create axial guide lifecycle world");
            const auto mesh = add_box(world, {0.1F,0.1F,0.1F});
            RigidBodyId anchor{}, body{};
            check_status(world.add_rigid_body(
                {.motion = MotionType::static_body, .mesh = mesh}, anchor), "add guide anchor");
            check_status(world.add_rigid_body({.mesh = mesh,
                .initial_state = {.position = {0.0F,0.5F,0.0F}},
                .linear_damping = 0.0F, .angular_damping = 0.0F}, body), "add offset guide body");
            RigidConstraintOptions options{.type = type,
                .body_a = reverse ? body : anchor, .body_b = reverse ? anchor : body,
                .local_anchor_a = reverse ? Vec3{0,-0.5F,0} : Vec3{},
                .local_anchor_b = reverse ? Vec3{} : Vec3{0,-0.5F,0}};
            RigidConstraintId joint{};
            check_status(world.add_rigid_constraint(options, joint), "add offset guide");
            // An off-axis COM must still feel gravitational torque around the
            // free piston axis; a slider must react it without rotation.
            check_status(world.step({.timestep = 1.0F/60, .substeps = 4U,
                .gravity = {0,0,9.81F}}), "step guide torque");
            RigidBodyState state{};
            check_status(world.read_rigid_body_state(body, state), "read offset guide");
            if (type == RigidConstraintType::piston)
                check(state.angular_velocity.x > 0.2F,
                      "guide projection must retain off-axis gravitational torque");
            else {
                if (!(std::fabs(state.angular_velocity.x) < 1.e-5F && std::fabs(state.position.z) < 1.e-5F))
                    std::cerr << "slider reaction: reverse=" << reverse << " omega="
                              << state.angular_velocity.x << " z=" << state.position.z << '\n';
                check(std::fabs(state.angular_velocity.x) < 1.e-5F &&
                      std::fabs(state.position.z) < 1.e-5F,
                      "slider must react forbidden gravitational torque");
            }
            options.enabled = false;
            check_status(world.update_rigid_constraint(joint, options), "disable guide");
            const float old_speed = state.linear_velocity.z;
            check_status(world.step({.timestep = 1.0F/60, .substeps = 4U,
                .gravity = {0,0,9.81F}}), "step disabled guide");
            check_status(world.read_rigid_body_state(body, state), "read released guide");
            check(state.linear_velocity.z > old_speed + 0.15F,
                  "disabled guide must release every body DOF immediately");
            options.enabled = true;
            options.breaking_impulse_threshold = 0.0001F;
            check_status(world.update_rigid_constraint(joint, options), "make guide breakable");
            check_status(world.step({.timestep = 1.0F/60, .substeps = 4U,
                .gravity = {0,0,9.81F}}), "break loaded guide");
            RigidConstraintState broken{};
            check_status(world.read_rigid_constraint_state(joint, broken), "read broken guide");
            check(broken.broken && broken.applied_impulse > 0.0F,
                  "breakable guide must retain general reaction-impulse semantics");
        }
    }
}

void test_guided_edge_clearance() {
    using namespace parallel_mater;
    for (const auto type : {RigidConstraintType::piston, RigidConstraintType::slider}) {
        for (const bool hit : {false, true}) {
            World world;
            check_status(World::create({.rigid_body_capacity = 3U,
                .rigid_constraint_capacity = 1U, .triangle_mesh_capacity = 2U}, world),
                "create guided edge world");
            const auto box = add_box(world, {0.1F,0.1F,0.1F});
            const auto plane = add_plane(world);
            RigidBodyId anchor{}, floor{}, body{};
            check_status(world.add_rigid_body({.motion = MotionType::static_body,
                .mesh = box, .initial_state = {.position = {0,10,0}}}, anchor),
                "add remote guide anchor");
            check_status(world.add_rigid_body({.motion = MotionType::static_body,
                .mesh = plane}, floor), "add finite guide obstacle");
            const float height = hit ? 0.05F : 0.1005F;
            check_status(world.add_rigid_body({.mesh = box,
                .initial_state = {.position = {-6.3F,height,0}, .linear_velocity = {1,0,0}},
                .linear_damping = 0, .angular_damping = 0}, body), "add edge traveller");
            RigidConstraintId joint{};
            check_status(world.add_rigid_constraint({.type = type, .body_a = anchor,
                .body_b = body, .local_anchor_a = {0,height-10,0}}, joint), "add edge guide");
            RigidBodyState state{};
            for (int frame = 0; frame < 60; ++frame) {
                check_status(world.step({.timestep = 1.0F/60, .substeps = 4U, .gravity = {}}),
                    "step guided edge approach");
                check_status(world.read_rigid_body_state(body, state), "read edge traveller");
                if (!hit)
                    check(std::fabs(state.linear_velocity.x-1.0F) < 1.e-4F &&
                          std::fabs(state.angular_velocity.x) < 1.e-4F,
                          "a 0.5 mm edge near-miss must not create speculative braking or spin");
            }
            const bool stopped_or_cleared = hit ? state.position.x < -6.098F : state.position.x > -5.301F;
            if (!stopped_or_cleared)
                std::cerr << "guided edge: hit=" << hit << " x=" << state.position.x
                          << " speed=" << state.linear_velocity.x << '\n';
            check(stopped_or_cleared,
                  "guided continuous contacts must block true hits but allow clear paths");
        }
    }
}

void test_piston_thin_rotational_stop() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 3U,
        .rigid_constraint_capacity = 1U, .triangle_mesh_capacity = 2U}, world),
        "create thin rotational stop world");
    const auto bar = add_box(world, {0.02F,0.5F,0.01F});
    const auto stop = add_box(world, {6.0F,0.1F,0.005F});
    RigidBodyId anchor{}, wall{}, rotor{};
    check_status(world.add_rigid_body({.motion = MotionType::static_body,
        .mesh = bar, .initial_state = {.position = {10,0,0}}}, anchor),
        "add rotational stop anchor");
    check_status(world.add_rigid_body({.motion = MotionType::static_body,
        .mesh = stop, .initial_state = {.position = {0,0.5F,0.03F}},
        .friction = 0, .restitution = 0}, wall), "add thin stop");
    check_status(world.add_rigid_body({.mesh = bar,
        .initial_state = {.linear_velocity = {1,0,0}, .angular_velocity = {5,0,0}},
        .friction = 0, .restitution = 0, .linear_damping = 0, .angular_damping = 0}, rotor),
        "add fast rotating piston bar");
    RigidConstraintId joint{};
    check_status(world.add_rigid_constraint({.type = RigidConstraintType::piston,
        .body_a = anchor, .body_b = rotor}, joint), "add rotational stop guide");
    for (int frame = 0; frame < 30; ++frame) {
        check_status(world.step({.timestep = 1.0F/60, .substeps = 1U, .gravity = {},
            .collect_rigid_contacts = true}),
            "step thin rotational stop");
        RigidBodyState state{};
        check_status(world.read_rigid_body_state(rotor, state), "read stopped rotor");
        const float angle = 2.0F * std::atan2(state.orientation.x, state.orientation.w);
        if (!(angle >= -0.001F && angle < 0.031F))
            std::cerr << "rotational stop: frame=" << frame << " angle=" << angle
                      << " omega=" << state.angular_velocity.x << '\n';
        check(angle >= -0.001F && angle < 0.031F,
              "swept entry contact must stop rotation before crossing a thin face");
        check(std::fabs(state.angular_velocity.x) < 0.01F &&
              std::fabs(state.linear_velocity.x-1.0F) < 0.001F,
              "frictionless rotational stop must preserve permitted axial travel");
    }
}

void test_contact_world_composition() {
    using namespace parallel_mater;
    RigidBodyState reference{};
    for (const std::uint32_t count : {4U, 31U, 32U, 256U, 257U}) {
        for (const bool joint : {false, true}) {
            World world;
            check_status(World::create({.rigid_body_capacity = count,
                .rigid_constraint_capacity = 1U, .triangle_mesh_capacity = 2U}, world),
                "create world-composition contact fixture");
            const auto plane = add_plane(world);
            const auto cube = add_box(world, {0.5F, 0.5F, 0.5F});
            RigidBodyId floor{}, box{}, anchor{}, bob{};
            check_status(world.add_rigid_body({.motion = MotionType::static_body,
                .mesh = plane, .friction = 0.8F}, floor), "add composition floor");
            check_status(world.add_rigid_body({.mesh = cube,
                .initial_state = {.position = {0, 0.495F, 0}},
                .friction = 0.8F, .restitution = 0}, box), "add composition box");
            check_status(world.add_rigid_body({.motion = MotionType::static_body,
                .mesh = cube, .initial_state = {.position = {100, 100, 0}}}, anchor),
                "add distant anchor");
            check_status(world.add_rigid_body({.mesh = cube,
                .initial_state = {.position = {100, 98, 0}}}, bob), "add distant bob");
            for (std::uint32_t i = 4; i < count; ++i) {
                RigidBodyId padding{};
                check_status(world.add_rigid_body({.motion = MotionType::static_body,
                    .mesh = cube, .initial_state = {.position = {200 + 2.0F*i, 100, 0}}}, padding),
                    "add noninteracting padding body");
            }
            if (joint) {
                RigidConstraintId id{};
                check_status(world.add_rigid_constraint({.type = RigidConstraintType::point,
                    .body_a = anchor, .body_b = bob, .local_anchor_a = {0, -1, 0},
                    .local_anchor_b = {0, 1, 0}}, id), "add unrelated point constraint");
            }
            for (unsigned frame = 0; frame < 30; ++frame)
                check_status(world.step({.substeps = 4U}), "step composition fixture");
            RigidBodyState state{};
            check_status(world.read_rigid_body_state(box, state), "read composition box");
            if (count == 4U && !joint) reference = state;
            check(near(state.position.y, reference.position.y, 0.002F) &&
                  near(state.linear_velocity.y, reference.linear_velocity.y, 0.02F) &&
                  near(state.orientation.w, reference.orientation.w, 0.001F),
                  "unrelated joints/body-count thresholds must preserve isolated support");
            check(state.position.y >= 0.499F && std::fabs(state.linear_velocity.y) < 0.02F,
                  "isolated support must remain stable under every contact schedule");
        }
    }
}

void test_contact_pass_limits() {
    using namespace parallel_mater;
    for (const unsigned count : {2U, 257U}) {
        World world;
        check_status(World::create({.rigid_body_capacity = count}, world), "create pass-limit world");
        const auto plane = add_plane(world);
        const auto cube = add_box(world, {0.5F, 0.5F, 0.5F});
        RigidBodyId floor{}, box{};
        check_status(world.add_rigid_body({.motion = MotionType::static_body, .mesh = plane}, floor),
                     "add pass-limit floor");
        check_status(world.add_rigid_body({.mesh = cube,
            .initial_state = {.position = {0, 0.5F, 0}}, .restitution = 0}, box), "add pass-limit box");
        for (unsigned i = 2; i < count; ++i) {
            RigidBodyId padding{};
            check_status(world.add_rigid_body({.motion = MotionType::static_body, .mesh = cube,
                .initial_state = {.position = {2.0F * i, 100, 0}}}, padding), "pad pass-limit world");
        }
        check(world.step({.rigid_contact_pass_limit = 65U}).code == StatusCode::invalid_argument,
              "reject out-of-range contact pass limit");
        for (const unsigned passes : {1U, 2U, 4U, 8U}) {
            check_status(world.step({.gravity = {}, .collect_kernel_timings = true,
                .rigid_contact_pass_limit = passes}), "step capped contacts");
            WorldStatistics statistics{};
            check_status(world.collect_statistics(statistics), "read capped pass count");
            check(statistics.rigid_contact_maximum_passes == passes &&
                  statistics.rigid_contact_early_exit_count == 0U,
                  "hard caps must finalize islands and count actual passes, excluding warm start");
            WorldStepTimings timing{};
            check_status(world.collect_step_timings(timing), "read capped launch count");
            check(timing.rigid_contact_solve.launch_count == 24U,
                  "launch count must not depend on world size or pass cap");
            RigidBodyState state{};
            check_status(world.read_rigid_body_state(box, state), "read capped support");
            check(near(state.position.y, 0.5F, 0.002F) && std::fabs(state.linear_velocity.y) < 0.001F,
                  "one-pass isolated support must remain finite and stationary without gravity");
        }
    }
}

void test_contact_island_convergence_and_diagnostics() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 104U,
        .rigid_constraint_capacity = 1U, .triangle_mesh_capacity = 3U}, world), "create island fixture");
    const auto plane = add_box(world, {10, 0.5F, 10});
    const auto cube = add_box(world, {0.5F, 0.5F, 0.5F});
    RigidBodyId floor{}, a{}, b{};
    check_status(world.add_rigid_body({.motion = MotionType::static_body, .mesh = plane,
        .initial_state = {.position = {0, -0.5F, 0}}}, floor), "add island floor");
    const auto add_supported = [&](float x, RigidBodyId &id) {
        check_status(world.add_rigid_body({.mesh = cube, .initial_state = {.position = {x, 0.5F, 0}},
            .mass = 2.0F, .restitution = 0, .linear_damping = 0, .angular_damping = 0}, id), "add island box");
    };
    // Disconnected tiny triangles have overlapping broad bounds but no actual
    // contacts. Put the supported boxes after >4096 candidate pairs: empty
    // candidates must not evict their response/history cache entries.
    const auto sparse = upload_mesh(world,
        {{-100, -100, -100}, {-99.9F, -100, -100}, {-100, -100, -99.9F},
         {100, 100, 100}, {100.1F, 100, 100}, {100, 100, 100.1F}},
        {0, 1, 2, 3, 4, 5}, "add sparse broad-bound mesh");
    for (unsigned index = 1; index < 102; ++index) {
        RigidBodyId id{};
        check_status(world.add_rigid_body({.mesh = sparse,
            .initial_state = {.position = {0, 0.04F * index, 0}},
            .collision_margin = 0.001F}, id), "pad island candidates");
    }
    add_supported(-2, a);
    add_supported(2, b);
    for (unsigned frame = 0; frame < 90; ++frame)
        check_status(world.step({.substeps = 4, .collect_rigid_contacts = true}), "settle independent islands");
    WorldStatistics statistics{};
    check_status(world.collect_statistics(statistics), "read island convergence");
    check(statistics.rigid_contact_candidate_pairs > 4096 && statistics.rigid_contact_live_pairs == 2,
          "sparse broad-phase candidates must not consume live contact cache slots");
    check(statistics.rigid_contact_island_count == 2, "shared static ground must not merge contact islands");
    // AVBD uses a bounded primal/dual budget, not the removed velocity-row
    // convergence policy. Island membership and support remain checked below.
    check(statistics.rigid_contact_early_exit_count == 0 && statistics.rigid_contact_maximum_passes == 10,
          "settled AVBD islands must report the ten-iteration pose phase without a spurious impact phase");
    const auto contacts = world.rigid_contacts();
    std::vector<RigidContactEvent> events(contacts.event_count);
    check(cudaMemcpy(events.data(), contacts.events.data, events.size() * sizeof(RigidContactEvent),
                     cudaMemcpyDeviceToHost) == cudaSuccess, "read deferred support impulses");
    float support = 0;
    for (const auto &event : events) support += event.normal_impulse;
    check(near(support, 4.0F * 9.81F / 240.0F, 0.002F),
          "deferred diagnostics include warm-start support exactly once for the final substep");
    RigidConstraintId joint{};
    check_status(world.add_rigid_constraint({.type = RigidConstraintType::point, .body_a = a, .body_b = b,
        .local_anchor_a = {2, 0, 0}, .local_anchor_b = {-2, 0, 0}}, joint), "connect supported islands");
    check_status(world.step({.substeps = 4}), "step connected islands");
    check_status(world.collect_statistics(statistics), "read joined island");
    check(statistics.rigid_contact_island_count == 1, "a live dynamic joint must merge convergence islands");
    check_status(world.remove_rigid_constraint(joint), "release convergence islands");
    check_status(world.apply_impulse(a, {2, 0, 0}, {}), "wake a converged island");
    check_status(world.step({.substeps = 4}), "step reactivated island");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(a, state), "read reactivated box");
    if (!(state.linear_velocity.x > 0.5F))
        std::cerr << "reactivated body: vx=" << state.linear_velocity.x << '\n';
    check(state.linear_velocity.x > 0.5F, "early stopping must not sleep or freeze an impacted body");
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    test_small_triangle_edge_clearance();
    test_mesh_lifetime_and_integration();
    test_generation_and_kinematics();
    test_batched_central_acceleration_and_interpolation_snapshot();
    test_floor_contact_and_async_contract();
    test_open_two_sided_surface();
    test_rotation_dynamic_coupling_and_determinism();
    test_fixed_cluster_collision_filter();
    test_fixed_cluster_ground_support();
    test_point_constraint_ground_support();
    test_high_speed_swept_triangle_contact();
    test_swept_contact_when_leaf_cache_overflows(2U);
    test_swept_contact_when_leaf_cache_overflows(17U);
    test_small_rest_offset_speculative_contact();
    test_parallel_contact_coloring(8U);
    test_parallel_contact_coloring(128U);
    test_parallel_contact_coloring(256U);
    test_invalid_triangle_indices();
    test_opt_in_physics_debug_capture();
    test_rigid_constraint_types();
    test_piston_free_twist_alignment();
    test_rigid_constraint_toggle();
    test_static_axial_guide_lifecycle();
    test_guided_edge_clearance();
    test_piston_thin_rotational_stop();
    test_contact_world_composition();
    test_contact_pass_limits();
    test_contact_island_convergence_and_diagnostics();
    if (failures != 0) {
        std::cerr << failures << " rigid test(s) failed\n";
        return 1;
    }
    std::cout << "All triangle rigid-body tests passed\n";
    return 0;
}
