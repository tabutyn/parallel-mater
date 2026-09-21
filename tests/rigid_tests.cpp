// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

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

void test_floor_contact_and_async_contract() {
    using namespace parallel_mater;
    World world;
    check_status(World::create({.rigid_body_capacity = 2U,
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
          "triangle mesh contact must project the box above the plane");
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
                  contact.penetration > 0.0F && contact.normal_impulse >= 0.0F,
              "rigid contact diagnostic must contain finite solver values");
    }

    WorldStepTimings timings{};
    check_status(world.collect_step_timings(timings),
                 "collect rigid kernel timings");
    check(timings.available && timings.rigid_integration.launch_count == 4U &&
              timings.rigid_contact_generation.launch_count == 4U &&
              timings.rigid_contact_solve.launch_count == 4U &&
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
    check(near(left_state.linear_velocity.x + right_state.linear_velocity.x,
               0.0F, 1.0e-4F),
          "dynamic triangle collision must preserve linear momentum");
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

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    test_mesh_lifetime_and_integration();
    test_generation_and_kinematics();
    test_floor_contact_and_async_contract();
    test_open_two_sided_surface();
    test_rotation_dynamic_coupling_and_determinism();
    test_invalid_triangle_indices();
    if (failures != 0) {
        std::cerr << failures << " rigid test(s) failed\n";
        return 1;
    }
    std::cout << "All triangle rigid-body tests passed\n";
    return 0;
}
