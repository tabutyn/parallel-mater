// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <iomanip>
#include <iostream>
#include <vector>

namespace {

bool require(parallel_mater::Status status, const char *operation) {
    if (status) {
        return true;
    }
    std::cerr << operation << " failed: "
              << (status.message != nullptr ? status.message : "unknown") << '\n';
    return false;
}

parallel_mater::TriangleMeshId upload_mesh(
    parallel_mater::World &world,
    const std::vector<parallel_mater::Vec3> &vertices,
    const std::vector<std::uint32_t> &indices) {
    using namespace parallel_mater;
    Vec3 *device_vertices = nullptr;
    std::uint32_t *device_indices = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&device_vertices),
               vertices.size() * sizeof(Vec3));
    cudaMalloc(reinterpret_cast<void **>(&device_indices),
               indices.size() * sizeof(std::uint32_t));
    cudaMemcpy(device_vertices, vertices.data(), vertices.size() * sizeof(Vec3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices, indices.data(),
               indices.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    TriangleMeshId mesh{};
    require(world.add_triangle_mesh({device_vertices, vertices.size()},
                                    {device_indices, indices.size()}, mesh),
            "add triangle mesh");
    cudaFree(device_indices);
    cudaFree(device_vertices);
    return mesh;
}

} // namespace

int main() {
    using namespace parallel_mater;
    World world;
    if (!require(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U},
                               world),
                 "World::create")) {
        return 1;
    }
    const TriangleMeshId floor_mesh = upload_mesh(
        world,
        {{-5.0F, 0.0F, -5.0F}, {5.0F, 0.0F, -5.0F},
         {5.0F, 0.0F, 5.0F}, {-5.0F, 0.0F, 5.0F}},
        {0, 2, 1, 0, 3, 2});
    const TriangleMeshId tetrahedron = upload_mesh(
        world,
        {{0.0F, 0.55F, 0.0F}, {-0.5F, -0.35F, -0.35F},
         {0.5F, -0.35F, -0.35F}, {0.0F, -0.35F, 0.55F}},
        {0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2});
    RigidBodyId floor{};
    RigidBodyId body{};
    if (!require(world.add_rigid_body({.motion = MotionType::static_body,
                                       .mesh = floor_mesh,
                                       .friction = 0.9F},
                                      floor),
                 "add floor") ||
        !require(world.add_rigid_body(
                     {.mesh = tetrahedron,
                      .initial_state = {.position = {0.0F, 2.5F, 0.0F},
                                        .angular_velocity = {0.0F, 0.0F, 1.0F}},
                      .mass = 2.0F,
                      .friction = 0.8F},
                     body),
                 "add body")) {
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
            if (!require(world.read_rigid_body_state(body, state), "read body")) {
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
