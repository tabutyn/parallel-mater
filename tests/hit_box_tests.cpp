// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

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

parallel_mater::TriangleMeshId add_box_mesh(
    parallel_mater::World &world, parallel_mater::Vec3 half) {
    using parallel_mater::Vec3;
    const std::array<Vec3, 8> vertices{{
        {-half.x, -half.y, -half.z}, {half.x, -half.y, -half.z},
        {half.x, half.y, -half.z},   {-half.x, half.y, -half.z},
        {-half.x, -half.y, half.z},  {half.x, -half.y, half.z},
        {half.x, half.y, half.z},    {-half.x, half.y, half.z}}};
    const std::array<std::uint32_t, 36> indices{{
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7}};
    Vec3 *device_vertices = nullptr;
    std::uint32_t *device_indices = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&device_vertices),
                     sizeof(vertices)) == cudaSuccess,
          "allocate hit-box test vertices");
    check(cudaMalloc(reinterpret_cast<void **>(&device_indices),
                     sizeof(indices)) == cudaSuccess,
          "allocate hit-box test indices");
    check(cudaMemcpy(device_vertices, vertices.data(), sizeof(vertices),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload hit-box test vertices");
    check(cudaMemcpy(device_indices, indices.data(), sizeof(indices),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload hit-box test indices");
    parallel_mater::TriangleMeshId mesh{};
    check_status(world.add_triangle_mesh(
                     {device_vertices, vertices.size()},
                     {device_indices, indices.size()}, mesh),
                 "add hit-box test mesh");
    cudaFree(device_indices);
    cudaFree(device_vertices);
    return mesh;
}

bool contains(const std::vector<parallel_mater::RigidBodyId> &items,
              parallel_mater::RigidBodyId wanted) {
    return std::find(items.begin(), items.end(), wanted) != items.end();
}

} // namespace

int main() {
    using namespace parallel_mater;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    World world;
    check_status(World::create({.fluid_capacity = 1U,
                                .rigid_body_capacity = 3U,
                                .triangle_mesh_capacity = 1U}, world),
                 "create hit-box world");
    const TriangleMeshId mesh = add_box_mesh(world, {0.25F, 0.25F, 0.25F});
    RigidBodyId inside{}, touching{}, outside{};
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = mesh}, inside),
                 "add body inside hit box");
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = mesh,
                      .initial_state = {.position = {1.25F, 0.0F, 0.0F}}},
                     touching),
                 "add body touching hit box");
    check_status(world.add_rigid_body(
                     {.motion = MotionType::static_body, .mesh = mesh,
                      .initial_state = {.position = {2.0F, 0.0F, 0.0F}}},
                     outside),
                 "add body outside hit box");

    const std::array<FluidParticle, 4> particles{{
        {.position = {0.0F, 0.0F, 0.0F}},
        {.position = {1.0F, 0.0F, 0.0F}},
        {.position = {0.0F, 1.0F, 0.0F}},
        {.position = {1.01F, 0.0F, 0.0F}}}};
    FluidParticle *device_particles = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&device_particles),
                     sizeof(particles)) == cudaSuccess,
          "allocate hit-box particles");
    check(cudaMemcpy(device_particles, particles.data(), sizeof(particles),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          "upload hit-box particles");
    FluidId fluid{};
    check_status(world.add_fluid(
                     {.capacity = particles.size(),
                      .particle_radius = 0.05F,
                      .support_radius = 0.2F},
                     {device_particles, particles.size()}, fluid),
                 "add hit-box fluid");
    cudaFree(device_particles);

    const HitBox axis_aligned{.half_extents = {1.0F, 0.5F, 0.5F}};
    HitBoxResult result;
    check_status(world.query_hit_box(axis_aligned, result),
                 "query axis-aligned hit box");
    check(result.rigid_bodies.size() == 2U &&
              contains(result.rigid_bodies, inside) &&
              contains(result.rigid_bodies, touching) &&
              !contains(result.rigid_bodies, outside),
          "rigid query includes triangle overlap and boundary touch only");
    check(result.particles ==
              std::vector<HitBoxParticle>{{fluid, 0U}, {fluid, 1U}},
          "particle query uses stable IDs and includes the boundary");

    constexpr float half_sqrt_two = 0.70710678118F;
    HitBox rotated{.orientation = {0.0F, 0.0F, half_sqrt_two,
                                   half_sqrt_two},
                   .half_extents = {1.0F, 0.5F, 0.5F}};
    check_status(world.query_hit_box(rotated, result),
                 "query rotated hit box");
    check(result.particles ==
              std::vector<HitBoxParticle>{{fluid, 0U}, {fluid, 2U}},
          "particle query respects hit-box orientation");

    check_status(world.set_rigid_body_state(
                     touching, {.position = {2.0F, 0.0F, 0.0F}}),
                 "move touching body outside");
    check_status(world.query_hit_box(axis_aligned, result),
                 "query edited rigid state");
    check(result.rigid_bodies == std::vector<RigidBodyId>{inside},
          "hit-box polling follows lifecycle state edits");

    const HitBoxResult retained = result;
    HitBox invalid = axis_aligned;
    invalid.half_extents.x = 0.0F;
    check(world.query_hit_box(invalid, result).code ==
              StatusCode::invalid_argument &&
              result.rigid_bodies == retained.rigid_bodies &&
              result.particles == retained.particles,
          "invalid hit-box queries leave the previous result unchanged");

    if (failures != 0) {
        std::cerr << failures << " hit-box test(s) failed\n";
        return 1;
    }
    std::cout << "Hit-box API tests passed\n";
    return 0;
}
