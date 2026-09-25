// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace parallel_mater;

namespace {
bool check(Status status, const char *operation) {
    if (status) return true;
    std::cerr << operation << ": " << (status.message ? status.message : "unknown")
              << '\n';
    return false;
}
}

int main() {
    constexpr std::array<Vec3, 8> cube{{
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}}};
    constexpr std::array<std::uint32_t, 36> cube_indices{{
        0,2,1, 0,3,2, 4,5,6, 4,6,7, 0,1,5, 0,5,4,
        1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7}};
    std::vector<FluidParticle> particles;
    const FluidGeometrySource geometry{{cube.data(), cube.size()},
        {cube_indices.data(), cube_indices.size()}, {}, {1,2,3}, 0.2F};
    if (!check(sample_fluid_geometry(geometry, particles),
               "sample closed fluid geometry") || particles.empty() ||
        particles[0].velocity.x != 1.0F) return 1;
    const std::size_t count = particles.size();
    if (!check(sample_fluid_geometry(geometry, particles),
               "append second fluid geometry") ||
        particles.size() != 2U * count) return 1;
    FluidGeometrySource invalid = geometry;
    invalid.spacing = 0.0F;
    if (sample_fluid_geometry(invalid, particles).ok()) return 1;

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 0; // CPU sampling remains tested on machines without a GPU.

    World world;
    if (!check(World::create({.rigid_body_capacity = 1U,
                              .triangle_mesh_capacity = 1U}, world),
               "create paint world")) return 1;
    constexpr std::array<Vec3, 3> triangle{{
        {0,0,0}, {1,0,0}, {0,0,1}}};
    constexpr std::array<std::uint32_t, 3> indices{{0,1,2}};
    constexpr std::array<Vec2, 3> uvs{{
        {0.99F,0.5F}, {0.99F,0.5F}, {0.99F,0.5F}}};
    Vec3 *d_vertices = nullptr;
    std::uint32_t *d_indices = nullptr;
    Vec2 *d_uvs = nullptr;
    bool okay = cudaMalloc(reinterpret_cast<void **>(&d_vertices),
                           sizeof(triangle)) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void **>(&d_indices),
                   sizeof(indices)) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void **>(&d_uvs),
                   sizeof(uvs)) == cudaSuccess &&
        cudaMemcpy(d_vertices, triangle.data(), sizeof(triangle),
                   cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(d_indices, indices.data(), sizeof(indices),
                   cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(d_uvs, uvs.data(), sizeof(uvs),
                   cudaMemcpyHostToDevice) == cudaSuccess;
    TriangleMeshId mesh{};
    RigidBodyId body{};
    PaintFieldId field{};
    PaintRuleId rule{};
    FluidId fluid{};
    if (okay) okay = check(world.add_triangle_mesh(
        {d_vertices, triangle.size()}, {d_indices, indices.size()}, mesh),
        "add paint mesh");
    if (okay) okay = check(world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = mesh}, body),
        "add paint body");
    if (okay) okay = check(world.add_paint_field(
        {.body = body, .mesh = mesh, .vertex_uvs = {d_uvs, uvs.size()},
         .width = 64U, .height = 64U}, field), "add paint field");
    const FluidParticle initial{{0.2F, 0.02F, 0.2F}, {}};
    FluidParticle *d_initial = nullptr;
    if (okay) okay = cudaMalloc(reinterpret_cast<void **>(&d_initial),
                               sizeof(initial)) == cudaSuccess &&
        cudaMemcpy(d_initial, &initial, sizeof(initial),
                   cudaMemcpyHostToDevice) == cudaSuccess;
    if (okay) okay = check(world.add_fluid(
        {.capacity = 1U, .particle_radius = 0.05F,
         .support_radius = 0.12F, .solver_iterations = 1U},
        {d_initial, 1U}, fluid), "add paint fluid");
    if (okay) okay = check(world.add_paint_rule(
        {.source = fluid, .target = field}, rule), "add paint rule");
    if (okay) okay = check(world.step({.substeps = 1U, .gravity = {}}),
                            "step paint contact");
    PaintFieldDeviceView view{};
    if (okay) okay = check(world.paint_field_view(field, view),
                            "read paint view");
    std::uint32_t pixel = 0U;
    if (okay) okay = cudaMemcpy(&pixel, view.pixels.data + 32U*64U + 63U,
                                sizeof(pixel), cudaMemcpyDeviceToHost) ==
                     cudaSuccess && pixel == 2U;
    if (okay) okay = check(world.clear_paint_field(field), "clear paint field") &&
                     cudaDeviceSynchronize() == cudaSuccess &&
                     cudaMemcpy(&pixel, view.pixels.data + 32U*64U + 63U,
                                sizeof(pixel), cudaMemcpyDeviceToHost) ==
                         cudaSuccess && pixel == 0U;
    if (okay) okay = check(world.remove_paint_rule(rule),
                           "remove first paint rule") &&
                     check(world.remove_fluid(fluid),
                           "remove first paint fluid");
    const FluidParticle opposite{{0.2F, -0.02F, 0.2F}, {}};
    if (okay) okay = cudaMemcpy(d_initial, &opposite, sizeof(opposite),
                               cudaMemcpyHostToDevice) == cudaSuccess &&
                     check(world.add_fluid(
                         {.capacity = 1U, .particle_radius = 0.05F,
                          .support_radius = 0.12F, .solver_iterations = 1U},
                         {d_initial, 1U}, fluid), "add opposite-side fluid") &&
                     check(world.add_paint_rule(
                         {.source = fluid, .target = field}, rule),
                         "add opposite-side paint rule") &&
                     check(world.step({.substeps = 1U, .gravity = {}}),
                           "step opposite-side paint") &&
                     cudaMemcpy(&pixel, view.pixels.data + 32U*64U + 63U,
                                sizeof(pixel), cudaMemcpyDeviceToHost) ==
                         cudaSuccess && pixel == 1U;
    if (okay) okay = check(world.remove_paint_rule(rule), "remove paint rule") &&
                     check(world.remove_paint_field(field), "remove paint field");
    World geometry_world;
    FluidId geometry_fluid{};
    if (okay) okay = check(World::create({}, geometry_world),
                           "create geometry world") &&
                     check(geometry_world.add_fluid_geometry(
                         {.capacity = 10U}, geometry, geometry_fluid),
                         "add geometry fluid");
    FluidDeviceView geometry_view{};
    if (okay) okay = check(geometry_world.fluid_view(
                         geometry_fluid, geometry_view),
                         "view geometry fluid") &&
                     geometry_view.particle_count == 10U;
    cudaFree(d_initial);
    cudaFree(d_uvs);
    cudaFree(d_indices);
    cudaFree(d_vertices);
    if (!okay) {
        std::cerr << "physics-driven contact paint failed\n";
        return 1;
    }
    return 0;
}
