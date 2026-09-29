// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parallel_mater;

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void check(Status status, const char *message) {
    if (!status) throw std::runtime_error(std::string(message) + ": " + status.message);
}

struct Result {
    float rope_velocity{};
    float fluid_velocity{};
    float distance{};
    float fluid_force{};
    std::uint32_t contacts{};
};

static Result run(bool coupled) {
    World world;
    check(World::create({.rope_capacity = 1, .fluid_rope_coupling_capacity = 1,
                         .physics_debug = {.frame_capacity = 2}}, world),
          "create world");
    const std::array<Vec3, 2> line{{{-0.1F, 1.0F, 0.0F}, {0.1F, 1.0F, 0.0F}}};
    RopeId rope{};
    check(world.add_rope({.centerline = {line.data(), line.size()},
                          .node_spacing = 0.02F, .radius = 0.02F,
                          .self_collision = false}, rope), "add rope");
    const FluidParticle particle{{0.0F, 1.025F, 0.0F}, {0.0F, -1.0F, 0.0F}};
    FluidParticle *device = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&device), sizeof(particle)) == cudaSuccess,
          "allocate initial particle");
    check(cudaMemcpy(device, &particle, sizeof(particle), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload initial particle");
    FluidId fluid{};
    const Status added = world.add_fluid({.capacity = 1, .particle_radius = 0.02F,
        .support_radius = 0.08F, .solver_iterations = 2,
        .velocity_damping = 0.0F}, {device, 1}, fluid);
    cudaFree(device);
    check(added, "add fluid");
    FluidRopeCouplingId coupling{};
    if (coupled)
        check(world.add_fluid_rope_coupling({.fluid = fluid, .rope = rope}, coupling),
              "couple fluid and rope");
    check(world.step({.timestep = 1.0F / 60.0F, .substeps = 4,
                      .gravity = {0.0F, 0.0F, 0.0F},
                      .collect_kernel_timings = true}), "step");
    WorldStatistics statistics{};
    WorldStepTimings timings{};
    check(world.collect_statistics(statistics), "statistics");
    check(world.collect_step_timings(timings), "timings");
    RopeDeviceView rope_view{};
    FluidDeviceView fluid_view{};
    check(world.rope_view(rope, rope_view), "rope view");
    check(world.fluid_view(fluid, fluid_view), "fluid view");
    std::vector<Vec3> positions(rope_view.positions.size), velocities(positions.size()),
                      forces(positions.size());
    check(cudaMemcpy(positions.data(), rope_view.positions.data,
                     positions.size() * sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read rope positions");
    check(cudaMemcpy(velocities.data(), rope_view.velocities.data,
                     velocities.size() * sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read rope velocities");
    check(cudaMemcpy(forces.data(), rope_view.fluid_contact_forces.data,
                     forces.size() * sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read rope fluid forces");
    Vec3 water_position{}, water_velocity{};
    check(cudaMemcpy(&water_position, fluid_view.positions.data, sizeof(Vec3),
                     cudaMemcpyDeviceToHost) == cudaSuccess, "read water position");
    check(cudaMemcpy(&water_velocity, fluid_view.velocities.data, sizeof(Vec3),
                     cudaMemcpyDeviceToHost) == cudaSuccess, "read water velocity");
    const std::size_t middle = positions.size() / 2;
    Result result{velocities[middle].y, water_velocity.y,
                  water_position.y - positions[middle].y, forces[middle].y,
                  statistics.fluid_rope_contact_count};
    if (coupled) {
        check(timings.fluid_rope_contacts.launch_count == 32,
              "coupling did not run for every fluid iteration");
        PhysicsDebugCapture capture{};
        check(world.copy_physics_debug_capture(capture), "capture fluid rope forces");
        check(capture.frames.size() == 1 &&
              capture.frames[0].rope_nodes[middle].fluid_contact_force.y < 0.0F,
              "opt-in capture omitted rope water force");
        check(!world.remove_rope(rope) && !world.remove_fluid(fluid),
              "coupled resource removed");
        check(world.remove_fluid_rope_coupling(coupling), "remove coupling");
        check(!world.update_fluid_rope_coupling(coupling, {.fluid = fluid, .rope = rope}),
              "stale coupling accepted");
    }
    check(world.remove_fluid(fluid), "remove fluid");
    check(world.remove_rope(rope), "remove rope");
    return result;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        const Result dry = run(false), wet = run(true);
        check(dry.contacts == 0 && wet.contacts > 0, "missing rope-water contacts");
        check(wet.distance >= 0.039F, "particle penetrates rope segment");
        check(wet.fluid_velocity > dry.fluid_velocity + 0.5F,
              "rope failed to deflect water");
        check(wet.rope_velocity < dry.rope_velocity - 0.001F && wet.fluid_force < 0.0F,
              "water failed to push rope");
        check(std::isfinite(wet.rope_velocity) && std::isfinite(wet.fluid_velocity),
              "nonfinite contact response");
        std::cout << "fluid-rope contact: water vy " << dry.fluid_velocity << " -> "
                  << wet.fluid_velocity << ", rope vy " << dry.rope_velocity << " -> "
                  << wet.rope_velocity << ", contacts=" << wet.contacts << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
