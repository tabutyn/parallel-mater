// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::gallery;

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void require(Status status, const char *message) {
    if (!status) throw std::runtime_error(std::string(message) + ": " + status.message);
}
template <class T> static std::vector<T> read(DeviceSpan<const T> source) {
    std::vector<T> values(source.size);
    if (!values.empty())
        require(cudaMemcpy(values.data(), source.data,
            values.size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess,
            "CUDA readback failed");
    return values;
}
static float distance(Vec3 a, Vec3 b) {
    return std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) +
                     (a.z-b.z)*(a.z-b.z));
}

int main() {
    try {
        SceneDefinition scene;
        std::string error;
        require(load_glb_scene(PARALLEL_MATER_SMOKE_CLOTH_SCENE_PATH,
                               scene, error), error.c_str());
        require(scene.has_smoke && scene.cloths.size() == 1U &&
                scene.rigid_bodies.size() == 2U,
                "SmokeCloth systems were not exported");
        const auto &definition = scene.cloths.front();
        const auto &mesh = scene.meshes[definition.mesh_index];
        const std::size_t pins = std::count(definition.inverse_masses.begin(),
                                             definition.inverse_masses.end(), 0.0F);
        require(mesh.indices.size() >= 300U && pins == 34U,
                "cloth mesh or authored Pin group is missing");
        std::cout << "cloth_vertices=" << mesh.vertices.size()
                  << " cloth_triangles=" << mesh.indices.size() / 3U
                  << " pins=" << pins << '\n';

        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "GPU runtime check skipped: CUDA unavailable\n";
            return 0;
        }
        World world, reference_world;
        SceneInstance coupled, reference;
        require(create_scene_world(scene, world, coupled),
                "create smoke-cloth world");
        require(create_scene_world(scene, reference_world, reference),
                "create uncoupled reference world");
        require(coupled.smoke_cloth_couplings.size() == 1U &&
                reference.smoke_cloth_couplings.size() == 1U,
                "smoke-cloth API coupling not registered");
        require(!world.remove_smoke(coupled.smoke),
                "referenced smoke removed");
        require(!world.remove_cloth(coupled.cloths.front()),
                "referenced cloth removed");
        SmokeClothCouplingId rejected{};
        require(!world.add_smoke_cloth_coupling(
                    {.smoke = coupled.smoke, .cloth = coupled.cloths.front(),
                     .wind_drag = -1.0F}, rejected),
                "negative smoke drag accepted");
        require(!world.add_smoke_cloth_coupling(
                    {.smoke = coupled.smoke, .cloth = coupled.cloths.front()},
                    rejected), "duplicate smoke-cloth coupling accepted");
        require(reference_world.remove_smoke_cloth_coupling(
                    reference.smoke_cloth_couplings.front()),
                "remove reference coupling");

        constexpr unsigned frames = 180U;
        double coupled_milliseconds = 0.0;
        double reference_milliseconds = 0.0;
        for (unsigned frame = 0; frame < frames; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                                   .substeps = 4U, .gravity = {}};
            auto begin = std::chrono::steady_clock::now();
            require(world.step(step), "step smoke-cloth scene");
            coupled_milliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            begin = std::chrono::steady_clock::now();
            require(reference_world.step(step), "step reference scene");
            reference_milliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        }
        ClothDeviceView cloth{}, still_cloth{};
        require(world.cloth_view(coupled.cloths.front(), cloth),
                "read coupled cloth");
        require(reference_world.cloth_view(reference.cloths.front(), still_cloth),
                "read reference cloth");
        const auto positions = read(cloth.positions);
        const auto velocities = read(cloth.velocities);
        const auto reference_positions = read(still_cloth.positions);
        const auto bonds = read(cloth.bonds);
        require(positions.size() == mesh.vertices.size(),
                "unexpected cloth vertex count");
        float cloth_difference = 0.0F, maximum_speed = 0.0F;
        float maximum_bond_strain = 0.0F;
        for (std::size_t vertex = 0; vertex < positions.size(); ++vertex) {
            const Vec3 p = positions[vertex], v = velocities[vertex];
            require(std::isfinite(p.x) && std::isfinite(p.y) &&
                    std::isfinite(p.z) && std::isfinite(v.x) &&
                    std::isfinite(v.y) && std::isfinite(v.z),
                    "nonfinite cloth state");
            if (definition.inverse_masses[vertex] == 0.0F)
                require(distance(p, mesh.vertices[vertex].position) < 1.0e-4F,
                        "pinned cloth vertex moved");
            cloth_difference += distance(p, reference_positions[vertex]);
            maximum_speed = std::max(maximum_speed, distance(v, {}));
        }
        for (const ClothBond &bond : bonds)
            maximum_bond_strain = std::max(maximum_bond_strain,
                std::abs(distance(positions[bond.first], positions[bond.second]) /
                         bond.rest_length - 1.0F));
        SmokeDeviceView smoke{}, reference_smoke{};
        require(world.smoke_view(coupled.smoke, smoke), "read coupled smoke");
        require(reference_world.smoke_view(reference.smoke, reference_smoke),
                "read reference smoke");
        const auto smoke_positions = read(smoke.positions);
        const auto reference_smoke_positions = read(reference_smoke.positions);
        float tracer_difference = 0.0F;
        for (std::size_t index = 0; index < smoke_positions.size(); ++index)
            tracer_difference += distance(smoke_positions[index],
                                          reference_smoke_positions[index]);
        std::cout << "frames=" << frames
                  << " coupled_ms_per_frame=" << coupled_milliseconds / frames
                  << " reference_ms_per_frame=" << reference_milliseconds / frames
                  << " cloth_difference=" << cloth_difference
                  << " tracer_difference=" << tracer_difference
                  << " max_cloth_speed=" << maximum_speed
                  << " max_bond_strain=" << maximum_bond_strain
                  << " smoke_particles=" << smoke.particle_count << '\n';
        require(cloth_difference > 0.1F && tracer_difference > 1.0F,
                "smoke and cloth did not influence each other");
        require(maximum_speed < 20.0F && maximum_bond_strain < 1.0F,
                "smoke-cloth simulation became unstable");

        // A fast tracer must not jump from the upstream side to beyond a
        // thin cloth triangle in one frame.
        SceneDefinition crossing_scene = scene;
        crossing_scene.smoke_options.capacity = 8U;
        crossing_scene.smoke_options.particles_per_second = 5.0F;
        crossing_scene.smoke_options.emitter_center = {1.0F, 0.7F, 0.0F};
        crossing_scene.smoke_options.emitter_half_extents = {0.01F, 0.01F};
        crossing_scene.smoke_options.initial_velocity = {6.0F, 0.0F, 0.0F};
        crossing_scene.smoke_options.wind = {6.0F, 0.0F, 0.0F};
        crossing_scene.smoke_options.maximum_speed = 8.0F;
        crossing_scene.smoke_options.wake_strength = 0.0F;
        World crossing_world;
        SceneInstance crossing;
        require(create_scene_world(crossing_scene, crossing_world, crossing),
                "create fast-tracer crossing world");
        require(crossing_world.remove_smoke_cloth_coupling(
                    crossing.smoke_cloth_couplings.front()),
                "remove default crossing coupling");
        SmokeClothCouplingId crossing_coupling{};
        require(crossing_world.add_smoke_cloth_coupling(
                    {.smoke = crossing.smoke, .cloth = crossing.cloths.front(),
                     .wind_drag = 0.0F}, crossing_coupling),
                "create contact-only crossing coupling");
        for (unsigned frame = 0; frame < 2U; ++frame)
            require(crossing_world.step({.timestep = 0.2F, .substeps = 1U,
                                         .gravity = {}}),
                    "step fast-tracer crossing world");
        SmokeDeviceView crossed_smoke{};
        require(crossing_world.smoke_view(crossing.smoke, crossed_smoke),
                "read fast-tracer smoke");
        const auto crossing_positions = read(crossed_smoke.positions);
        require(crossing_positions.size() >= 2U &&
                crossing_positions[0].x < 1.33F,
                "fast smoke tracer crossed the cloth sheet");

        require(world.remove_smoke_cloth_coupling(
                    coupled.smoke_cloth_couplings.front()),
                "remove smoke-cloth coupling");
        require(!world.remove_smoke_cloth_coupling(
                    coupled.smoke_cloth_couplings.front()),
                "stale coupling handle accepted");
        require(world.remove_smoke(coupled.smoke), "remove smoke");
        require(world.remove_cloth(coupled.cloths.front()), "remove cloth");
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
