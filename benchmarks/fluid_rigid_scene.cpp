// SPDX-License-Identifier: MIT
#include "support.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::benchmark;
    using namespace parallel_mater::gallery;

    std::uint32_t frames = 1'000U;
    if (argc > 2) return 2;
    if (argc == 2) {
        char *end = nullptr;
        const unsigned long value = std::strtoul(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || value == 0U ||
            value > 100'000U) return 2;
        frames = static_cast<std::uint32_t>(value);
    }
    SceneDefinition scene;
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_FLUID_RIGID_SCENE_PATH, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    World world;
    SceneInstance instance;
    if (!prepare_world(scene, static_cast<std::uint32_t>(
            scene.meshes.size() + scene.collision_meshes.size()),
            world, instance)) return 1;
    WorldStatistics statistics{};
    if (!require(world.collect_statistics(statistics), "collect allocation"))
        return 1;
    std::cout << "bodies=" << statistics.rigid_body_count
              << " allocated_bytes=" << statistics.allocated_bytes << '\n';
    std::cout << "frames,particles,contacts,overflow,rigid_solve,neighbor,static_tri,"
                 "body_index,moving_tri,events,gpu,wall (ms)\n";
    Samples rigid, neighbor, passive, index, moving, events, gpu, wall;
    std::uint64_t contact_total = 0U, overflow_total = 0U;
    const StepOptions base = standard_step_options(false);
    std::cout << std::fixed << std::setprecision(3);
    for (std::uint32_t frame = 1U; frame <= frames; ++frame) {
        StepOptions options = base;
        options.collect_kernel_timings = frame % 10U == 0U;
        options.collect_fluid_contacts = options.collect_kernel_timings;
        const auto begin = std::chrono::steady_clock::now();
        if (!require(world.step(options), "step FluidRigid world")) return 1;
        wall.add(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count());
        if (options.collect_kernel_timings) {
            WorldStepTimings timings{};
            if (!require(world.collect_step_timings(timings),
                         "collect FluidRigid timings")) return 1;
            rigid.add(timings.rigid_contact_solve.total_milliseconds);
            neighbor.add(timings.fluid_neighbor_forces.total_milliseconds);
            passive.add(timings.fluid_static_contacts.total_milliseconds);
            index.add(timings.fluid_body_index.total_milliseconds);
            moving.add(timings.fluid_moving_contacts.total_milliseconds);
            events.add(timings.fluid_contact_events.total_milliseconds);
            gpu.add(timings.total_gpu_milliseconds);
            const ContactDeviceView contacts = world.contacts();
            contact_total += contacts.event_count;
            overflow_total += contacts.overflowed;
        }
        if (frame % 100U != 0U && frame != frames) continue;
        if (rigid.values.empty()) continue;
        if (!require(world.collect_statistics(statistics),
                     "collect FluidRigid statistics")) return 1;
        std::cout << frame << ',' << statistics.particle_count << ','
                  << contact_total << ',' << overflow_total << ','
                  << rigid.mean() << ',' << neighbor.mean() << ','
                  << passive.mean() << ',' << index.mean() << ','
                  << moving.mean() << ',' << events.mean() << ','
                  << gpu.mean() << ',' << wall.mean() << '\n';
        rigid.clear(); neighbor.clear(); passive.clear(); index.clear();
        moving.clear(); events.clear(); gpu.clear(); wall.clear();
        contact_total = overflow_total = 0U;
    }

    RigidBodyDeviceView bodies{};
    FluidDeviceView fluid{};
    if (!require(world.rigid_body_view(bodies), "view rigid states") ||
        !require(world.fluid_view(instance.fluid, fluid), "view fluid particles"))
        return 1;
    std::vector<RigidBodyState> states(bodies.states.size);
    std::vector<Vec3> positions(fluid.positions.size);
    std::vector<std::uint32_t> particle_ids(fluid.particle_count);
    if (cudaMemcpy(states.data(), bodies.states.data,
                   states.size() * sizeof(RigidBodyState),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(positions.data(), fluid.positions.data,
                   positions.size() * sizeof(Vec3),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(particle_ids.data(), fluid.stable_particle_ids.data,
                   particle_ids.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "FluidRigid state readback failed\n";
        return 1;
    }
    std::uint32_t invalid = 0U, below_mesh = 0U;
    float maximum_body_speed = 0.0F;
    for (const RigidBodyState &state : states) {
        const Vec3 p = state.position, v = state.linear_velocity;
        invalid += !std::isfinite(p.x) || !std::isfinite(p.y) ||
                   !std::isfinite(p.z) || !std::isfinite(v.x) ||
                   !std::isfinite(v.y) || !std::isfinite(v.z);
        maximum_body_speed = std::max(maximum_body_speed,
            std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z));
    }
    for (std::size_t index = 0U; index < positions.size(); ++index) {
        const Vec3 p = positions[index];
        invalid += !std::isfinite(p.x) || !std::isfinite(p.y) ||
                   !std::isfinite(p.z);
        if (p.y < -2.72F) {
            ++below_mesh;
            if (below_mesh <= 4U)
                std::cout << "below id=" << particle_ids[index]
                          << " position=(" << p.x << ',' << p.y << ','
                          << p.z << ")\n";
        }
    }
    std::cout << "final invalid=" << invalid << " below_mesh=" << below_mesh
              << " max_body_speed=" << maximum_body_speed << '\n';
    return invalid == 0U && below_mesh == 0U ? 0 : 1;
}
