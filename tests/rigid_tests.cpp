// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

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

float largest_vertical_axis(parallel_mater::Quaternion q) {
    const float first = std::fabs(2.0F * (q.x * q.y - q.w * q.z));
    const float second =
        std::fabs(1.0F - 2.0F * (q.x * q.x + q.z * q.z));
    const float third = std::fabs(2.0F * (q.y * q.z + q.w * q.x));
    return std::fmax(first, std::fmax(second, third));
}

void test_generation_checked_handles() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U}, world),
                 "create handle-test world");

    RigidBodyId first{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.5F)},
                     first),
                 "add first body");
    check_status(world.remove_rigid_body(first), "remove first body");
    const Status stale_status = world.set_rigid_body_state(first, {});
    check(stale_status.code == StatusCode::invalid_handle,
          "removed handle must remain invalid");

    RigidBodyId replacement{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.5F)},
                     replacement),
                 "add replacement body");
    check(first.index == replacement.index,
          "free handle slot should be reused deterministically");
    check(first.generation != replacement.generation,
          "reused handle slot must receive a new generation");
}

void test_cpu_reference_integration() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 1U}, world),
                 "create integration world");
    RigidBodyId body{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.25F),
                      .initial_state = {.position = {0.0F, 10.0F, 0.0F}},
                      .mass = 2.0F,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F,
                      .maximum_linear_speed = 1'000.0F},
                     body),
                 "add integration body");

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
    check_status(world.read_rigid_body_state(body, state),
                 "read integrated body");
    check(near(state.linear_velocity.y, expected_velocity),
          "GPU velocity must match semi-implicit CPU reference");
    check(near(state.position.y, expected_y),
          "GPU position must match semi-implicit CPU reference");

    check_status(world.apply_impulse(body, {2.0F, 0.0F, 0.0F}, state.position),
                 "queue center impulse");
    check_status(world.step({.timestep = 0.01F,
                             .substeps = 1U,
                             .gravity = {} }),
                 "step impulse world");
    check_status(world.read_rigid_body_state(body, state),
                 "read impulse body");
    check(near(state.linear_velocity.x, 1.0F),
          "impulse must change velocity by impulse divided by mass");
}

void test_contacts_and_async_contract() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 4U}, world),
                 "create contact world");
    RigidBodyId floor{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::plane(),
                      .friction = 0.8F},
                     floor),
                 "add floor");
    RigidBodyId sphere{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.5F),
                      .initial_state = {.position = {0.0F, 0.45F, 0.0F},
                                        .linear_velocity = {0.0F, -1.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     sphere),
                 "add contact sphere");

    FrameToken token;
    check_status(world.step_async({.timestep = 1.0F / 60.0F,
                                   .substeps = 2U,
                                   .gravity = {}},
                                  token),
                 "enqueue asynchronous frame");
    check(token.pending(), "new frame token must be pending until acknowledged");
    const Status busy = world.apply_force(sphere, {1.0F, 0.0F, 0.0F}, {});
    check(busy.code == StatusCode::busy,
          "world mutation must reject an unacknowledged frame");
    check_status(token.wait(), "wait for asynchronous frame");

    RigidBodyState state{};
    check_status(world.read_rigid_body_state(sphere, state),
                 "read contact sphere");
    check(state.position.y >= 0.499F,
          "sphere must be projected above the plane");
    check(state.linear_velocity.y >= -1.0e-4F,
          "plane contact must remove inward velocity");

    RigidBodyDeviceView view{};
    check_status(world.rigid_body_view(view), "borrow rigid device view");
    check(view.ids.size == 2U && view.states.size == 2U,
          "device view must contain all live bodies");

    WorldStatistics statistics{};
    check_status(world.collect_statistics(statistics), "collect statistics");
    check(statistics.rigid_body_count == 2U && statistics.frame_index == 1U,
          "statistics must report completed rigid frame");
}

void test_shape_validation_and_kinematics() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 5U}, world),
                 "create shape world");
    RigidBodyId invalid{};
    const Status plane_status = world.add_rigid_body(
        {.motion = MotionType::dynamic, .shape = CollisionShape::plane()}, invalid);
    check(plane_status.code == StatusCode::invalid_argument,
          "dynamic planes must be rejected");

    RigidBodyId moving_box{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::kinematic,
                      .shape = CollisionShape::box({0.5F, 0.5F, 0.5F})},
                     moving_box),
                 "add kinematic box");
    RigidBodyState target{};
    target.position = {1.0F, 2.0F, 3.0F};
    check_status(world.set_kinematic_target(moving_box, target),
                 "set kinematic target");
    check_status(world.step({.timestep = 0.25F,
                             .substeps = 4U,
                             .gravity = {}}),
                 "step kinematic target");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(moving_box, state),
                 "read kinematic box");
    check(near(state.position.x, 1.0F) && near(state.position.y, 2.0F) &&
              near(state.position.z, 3.0F),
          "kinematic body must reach its target exactly in one frame");
}

void test_approved_shape_contacts() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 10U}, world),
                 "create approved-shape world");

    RigidBodyId plane{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::plane()},
                     plane),
                 "add shape-test plane");
    RigidBodyId box{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::box({0.5F, 0.5F, 0.5F}),
                      .initial_state = {.position = {0.0F, 2.0F, 0.0F}}},
                     box),
                 "add shape-test box");
    RigidBodyId capsule{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::capsule(0.5F, 0.5F),
                      .initial_state = {.position = {3.0F, 2.0F, 0.0F}}},
                     capsule),
                 "add shape-test capsule");
    RigidBodyId fixed_sphere{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::sphere(0.5F),
                      .initial_state = {.position = {6.0F, 2.0F, 0.0F}}},
                     fixed_sphere),
                 "add shape-test sphere");

    RigidBodyId box_probe{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.6F),
                      .initial_state = {.position = {0.9F, 2.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     box_probe),
                 "add box probe");
    RigidBodyId capsule_probe{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.6F),
                      .initial_state = {.position = {3.9F, 2.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     capsule_probe),
                 "add capsule probe");
    RigidBodyId sphere_probe{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.6F),
                      .initial_state = {.position = {6.9F, 2.0F, 0.0F}},
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     sphere_probe),
                 "add sphere probe");
    RigidBodyId dynamic_box{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::box({0.3F, 0.4F, 0.3F}),
                      .initial_state = {.position = {9.0F, 0.2F, 0.0F}}},
                     dynamic_box),
                 "add dynamic box");
    RigidBodyId dynamic_capsule{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::capsule(0.25F, 0.5F),
                      .initial_state = {.position = {11.0F, 0.2F, 0.0F}}},
                     dynamic_capsule),
                 "add dynamic capsule");

    check_status(world.step({.timestep = 0.01F,
                             .substeps = 1U,
                             .gravity = {}}),
                 "resolve approved-shape contacts");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(box_probe, state), "read box probe");
    check(state.position.x >= 1.099F, "box must eject a sphere probe");
    check_status(world.read_rigid_body_state(capsule_probe, state),
                 "read capsule probe");
    check(state.position.x >= 4.099F, "capsule must eject a sphere probe");
    check_status(world.read_rigid_body_state(sphere_probe, state),
                 "read sphere probe");
    check(state.position.x >= 7.099F, "sphere must eject a sphere probe");
    check_status(world.read_rigid_body_state(dynamic_box, state),
                 "read dynamic box");
    check(state.position.y >= 0.399F, "plane must support a dynamic box");
    check_status(world.read_rigid_body_state(dynamic_capsule, state),
                 "read dynamic capsule");
    check(state.position.y >= 0.749F, "plane must support a dynamic capsule");
}

void test_same_gpu_determinism() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U}, world),
                 "create determinism world");
    RigidBodyId floor{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::plane(),
                      .friction = 0.7F},
                     floor),
                 "add determinism floor");
    RigidBodyId body{};
    const RigidBodyState initial{.position = {0.0F, 0.6F, 0.0F}};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::sphere(0.5F),
                      .initial_state = initial,
                      .mass = 2.0F,
                      .friction = 0.7F},
                     body),
                 "add determinism sphere");

    RigidBodyState reference{};
    for (int repetition = 0; repetition < 100; ++repetition) {
        check_status(world.set_rigid_body_state(body, initial),
                     "reset deterministic body");
        check_status(world.apply_impulse(body, {1.5F, -0.25F, 0.4F},
                                         {0.0F, 0.8F, 0.0F}),
                     "queue deterministic impulse");
        for (int frame = 0; frame < 12; ++frame) {
            check_status(world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = 4U,
                                     .gravity = {0.0F, -9.81F, 0.0F}}),
                         "step deterministic replay");
        }
        RigidBodyState result{};
        check_status(world.read_rigid_body_state(body, result),
                     "read deterministic replay");
        if (repetition == 0) {
            reference = result;
        } else {
            check(std::memcmp(&reference, &result, sizeof(result)) == 0,
                  "identical rigid replays must be bit-identical on one GPU");
        }
    }
}

void test_resting_rotation_and_dynamic_coupling() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 5U}, world),
                 "create rotational contact world");
    RigidBodyId floor{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::plane(),
                      .friction = 0.8F},
                     floor),
                 "add rotational floor");
    RigidBodyId box{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::box({0.55F, 0.35F, 0.45F}),
                      .initial_state = {.position = {-2.0F, 1.4F, 0.0F},
                                        .orientation = rotation_z(0.43F)},
                      .friction = 0.75F,
                      .linear_damping = 0.08F,
                      .angular_damping = 0.3F},
                     box),
                 "add tilted dynamic box");
    RigidBodyId capsule{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::dynamic,
                      .shape = CollisionShape::capsule(0.3F, 0.8F),
                      .initial_state = {.position = {2.0F, 1.5F, 0.0F},
                                        .orientation = rotation_z(0.22F)},
                      .friction = 0.75F,
                      .linear_damping = 0.08F,
                      .angular_damping = 0.3F},
                     capsule),
                 "add leaning dynamic capsule");
    for (int frame = 0; frame < 360; ++frame) {
        check_status(world.step({.timestep = 1.0F / 60.0F,
                                 .substeps = 8U,
                                 .gravity = {0.0F, -9.81F, 0.0F}}),
                     "settle rotational bodies");
    }
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(box, state), "read settled box");
    check(largest_vertical_axis(state.orientation) > 0.97F,
          "tilted box must settle onto one of its faces");
    check_status(world.read_rigid_body_state(capsule, state),
                 "read toppled capsule");
    const float capsule_axis_y =
        std::fabs(1.0F - 2.0F * (state.orientation.x * state.orientation.x +
                                 state.orientation.z * state.orientation.z));
    check(capsule_axis_y < 0.45F,
          "leaning capsule must topple instead of remaining upright");

    World pair_world;
    check_status(World::create({.rigid_body_capacity = 2U}, pair_world),
                 "create dynamic pair world");
    RigidBodyId left{};
    RigidBodyId right{};
    check_status(pair_world.add_rigid_body(
                     {.shape = CollisionShape::sphere(0.5F),
                      .initial_state = {.position = {-0.45F, 0.0F, 0.0F},
                                        .linear_velocity = {1.0F, 0.0F, 0.0F}},
                      .restitution = 0.5F,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     left),
                 "add left dynamic sphere");
    check_status(pair_world.add_rigid_body(
                     {.shape = CollisionShape::sphere(0.5F),
                      .initial_state = {.position = {0.45F, 0.0F, 0.0F},
                                        .linear_velocity = {-1.0F, 0.0F, 0.0F}},
                      .restitution = 0.5F,
                      .linear_damping = 0.0F,
                      .angular_damping = 0.0F},
                     right),
                 "add right dynamic sphere");
    check_status(pair_world.step(
                     {.timestep = 1.0F / 60.0F, .substeps = 4U, .gravity = {}}),
                 "resolve dynamic sphere pair");
    RigidBodyState left_state{};
    RigidBodyState right_state{};
    check_status(pair_world.read_rigid_body_state(left, left_state),
                 "read left dynamic sphere");
    check_status(pair_world.read_rigid_body_state(right, right_state),
                 "read right dynamic sphere");
    check(left_state.linear_velocity.x < 0.0F &&
              right_state.linear_velocity.x > 0.0F,
          "dynamic bodies must exchange equal-and-opposite collision impulse");
    check(near(left_state.linear_velocity.x + right_state.linear_velocity.x,
               0.0F, 1.0e-5F),
          "dynamic collision must preserve pair linear momentum");
}

void test_open_two_sided_triangle_mesh() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 3U,
                                .triangle_mesh_capacity = 1U},
                               world),
                 "create triangle contact world");
    constexpr Vec3 host_vertices[3]{{-4.0F, 0.0F, -4.0F},
                                    {4.0F, 0.0F, -4.0F},
                                    {0.0F, 0.0F, 4.0F}};
    constexpr std::uint32_t host_indices[3]{0U, 1U, 2U};
    constexpr std::uint32_t invalid_indices[3]{0U, 1U, 3U};
    Vec3 *vertices = nullptr;
    std::uint32_t *indices = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&vertices), sizeof(host_vertices)) ==
              cudaSuccess,
          "allocate triangle vertices");
    check(cudaMalloc(reinterpret_cast<void **>(&indices), sizeof(host_indices)) ==
              cudaSuccess,
          "allocate triangle indices");
    check(cudaMemcpy(vertices, host_vertices, sizeof(host_vertices),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload triangle vertices");
    check(cudaMemcpy(indices, invalid_indices, sizeof(invalid_indices),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload triangle indices");
    TriangleMeshId mesh{};
    const Status invalid_mesh =
        world.add_triangle_mesh({vertices, 3U}, {indices, 3U}, mesh);
    check(invalid_mesh.code == StatusCode::invalid_argument,
          "out-of-range triangle indices must be rejected");
    check(cudaMemcpy(indices, host_indices, sizeof(host_indices),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload valid triangle indices");
    check_status(world.add_triangle_mesh({vertices, 3U}, {indices, 3U}, mesh),
                 "add open triangle soup");
    cudaFree(indices);
    cudaFree(vertices);

    RigidBodyId surface{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body,
                      .shape = CollisionShape::triangle_mesh(mesh),
                      .friction = 0.7F},
                     surface),
                 "add open triangle body");
    RigidBodyId above{};
    RigidBodyId below{};
    check_status(world.add_rigid_body(
                     {.shape = CollisionShape::sphere(0.5F),
                      .initial_state = {.position = {-1.0F, 0.25F, 0.0F},
                                        .linear_velocity = {0.0F, -1.0F, 0.0F}},
                      .linear_damping = 0.0F},
                     above),
                 "add sphere above triangle front");
    check_status(world.add_rigid_body(
                     {.shape = CollisionShape::sphere(0.5F),
                      .initial_state = {.position = {1.0F, -0.25F, 0.0F},
                                        .linear_velocity = {0.0F, 1.0F, 0.0F}},
                      .linear_damping = 0.0F},
                     below),
                 "add sphere below triangle back");
    check_status(world.step(
                     {.timestep = 0.01F, .substeps = 2U, .gravity = {}}),
                 "resolve two-sided triangle contacts");
    RigidBodyState state{};
    check_status(world.read_rigid_body_state(above, state),
                 "read triangle front sphere");
    check(state.position.y >= 0.499F && state.linear_velocity.y >= -1.0e-4F,
          "triangle front must support an analytic body");
    check_status(world.read_rigid_body_state(below, state),
                 "read triangle back sphere");
    check(state.position.y <= -0.499F && state.linear_velocity.y <= 1.0e-4F,
          "triangle back must support an analytic body");
    check_status(world.remove_rigid_body(above), "remove front sphere");
    check_status(world.remove_rigid_body(below), "remove back sphere");

    RigidBodyId box{};
    RigidBodyId capsule{};
    check_status(world.add_rigid_body(
                     {.shape = CollisionShape::box({0.3F, 0.3F, 0.3F}),
                      .initial_state = {.position = {-1.0F, 0.2F, 0.0F},
                                        .linear_velocity = {0.0F, -1.0F, 0.0F}}},
                     box),
                 "add box above open triangle");
    check_status(world.add_rigid_body(
                     {.shape = CollisionShape::capsule(0.25F, 0.5F),
                      .initial_state = {.position = {1.0F, 0.5F, 0.0F},
                                        .linear_velocity = {0.0F, -1.0F, 0.0F}}},
                     capsule),
                 "add capsule above open triangle");
    check_status(world.step(
                     {.timestep = 0.01F, .substeps = 2U, .gravity = {}}),
                 "resolve box and capsule triangle contacts");
    check_status(world.read_rigid_body_state(box, state),
                 "read triangle-contact box");
    check(state.position.y >= 0.299F,
          "open triangle must support a box");
    check_status(world.read_rigid_body_state(capsule, state),
                 "read triangle-contact capsule");
    check(state.position.y >= 0.749F,
          "open triangle must support a capsule");
    check_status(world.remove_rigid_body(box), "remove triangle-contact box");
    check_status(world.remove_rigid_body(capsule),
                 "remove triangle-contact capsule");
    const Status referenced = world.remove_triangle_mesh(mesh);
    check(referenced.code == StatusCode::invalid_argument,
          "referenced triangle mesh must not be removable");
    check_status(world.remove_rigid_body(surface), "remove triangle body");
    check_status(world.remove_triangle_mesh(mesh), "remove unreferenced mesh");
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    test_generation_checked_handles();
    test_cpu_reference_integration();
    test_contacts_and_async_contract();
    test_shape_validation_and_kinematics();
    test_approved_shape_contacts();
    test_resting_rotation_and_dynamic_coupling();
    test_open_two_sided_triangle_mesh();
    test_same_gpu_determinism();
    if (failures != 0) {
        std::cerr << failures << " rigid test(s) failed\n";
        return 1;
    }
    std::cout << "All rigid-body tests passed\n";
    return 0;
}
