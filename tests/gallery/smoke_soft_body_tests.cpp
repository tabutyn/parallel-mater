// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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
    std::vector<T> result(source.size);
    if (!result.empty())
        require(cudaMemcpy(result.data(), source.data,
            result.size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess,
            "device read failed");
    return result;
}

int main(int argc, char **argv) {
    try {
        SceneDefinition scene;
        std::string error;
        const char *scene_path = argc > 1 ? argv[1] :
            PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH;
        if (!load_glb_scene(scene_path, scene, error))
            throw std::runtime_error(error);
        require(scene.has_smoke && scene.soft_bodies.size() == 20U &&
                scene.rigid_bodies.size() == 2U,
                "SmokeSoftbody systems were not exported");
        std::size_t nodes = 0U, bonds = 0U, triangles = 0U, pins = 0U;
        for (const auto &body : scene.soft_bodies) {
            require(body.solver_iterations == 4U,
                    "lightweight authored solver budget was not exported");
            nodes += body.nodes.size();
            bonds += body.bonds.size();
            triangles += scene.meshes[body.mesh_index].indices.size() / 3U;
            for (const float inverse_mass : body.inverse_masses)
                pins += inverse_mass == 0.0F;
        }
        std::cout << "soft_bodies=" << scene.soft_bodies.size()
                  << " nodes=" << nodes << " bonds=" << bonds
                  << " triangles=" << triangles << " pins=" << pins << '\n';
        std::cout << "smoke_emitter=" << scene.smoke_options.emitter_center.x
                  << ',' << scene.smoke_options.emitter_center.y << ','
                  << scene.smoke_options.emitter_center.z
                  << " wind=" << scene.smoke_options.wind.x << ','
                  << scene.smoke_options.wind.y << ','
                  << scene.smoke_options.wind.z << '\n';
        Vec3 minimum = scene.soft_bodies.front().nodes.front();
        Vec3 maximum = minimum;
        for (const Vec3 p : scene.soft_bodies.front().nodes) {
            minimum.x = std::min(minimum.x, p.x);
            minimum.y = std::min(minimum.y, p.y);
            minimum.z = std::min(minimum.z, p.z);
            maximum.x = std::max(maximum.x, p.x);
            maximum.y = std::max(maximum.y, p.y);
            maximum.z = std::max(maximum.z, p.z);
        }
        std::cout << "first_body_bounds=" << minimum.x << ',' << minimum.y
                  << ',' << minimum.z << " to " << maximum.x << ','
                  << maximum.y << ',' << maximum.z << '\n';
        if (argc == 1)
            require(nodes < 10'000U && pins >= 20U,
                    "soft-body lattice is too dense or missing pins");
        if (argc != 1) return 0;
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "GPU runtime check skipped: CUDA device unavailable\n";
            return 0;
        }
        World world;
        SceneInstance instance;
        require(create_scene_world(scene, world, instance), "instantiate scene");
        require(instance.has_smoke && instance.soft_bodies.size() == 20U &&
                instance.smoke_soft_body_couplings.size() == 20U,
                "smoke/soft-body API couplings were not registered");
        require(!world.remove_smoke(instance.smoke),
                "referenced smoke could be removed");
        require(!world.remove_soft_body(instance.soft_bodies.front()),
                "referenced soft body could be removed");
        constexpr unsigned frames = 180U;
        const auto started = std::chrono::steady_clock::now();
        for (unsigned frame = 0; frame < frames; ++frame)
            require(world.step({.timestep = 1.0F / 60.0F,
                                .substeps = 4U}), "step smoke soft-body scene");
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        float maximum_speed = 0.0F;
        float maximum_strain = 0.0F;
        std::size_t displaced = 0U;
        for (std::size_t body_index = 0; body_index < instance.soft_bodies.size();
             ++body_index) {
            SoftBodyDeviceView body{};
            require(world.soft_body_view(instance.soft_bodies[body_index], body),
                    "read soft body");
            const auto positions = read(body.positions);
            const auto velocities = read(body.velocities);
            for (std::size_t node = 0; node < positions.size(); ++node) {
                const Vec3 p = positions[node], v = velocities[node];
                require(std::isfinite(p.x) && std::isfinite(p.y) &&
                        std::isfinite(p.z) && std::isfinite(v.x) &&
                        std::isfinite(v.y) && std::isfinite(v.z),
                        "nonfinite soft-body state");
                maximum_speed = std::max(maximum_speed,
                    std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
                const Vec3 rest = scene.soft_bodies[body_index].nodes[node];
                displaced += std::abs(p.x-rest.x) + std::abs(p.y-rest.y) +
                    std::abs(p.z-rest.z) > 0.01F;
                if (scene.soft_bodies[body_index].inverse_masses[node] == 0.0F)
                    require(std::abs(p.x-rest.x) + std::abs(p.y-rest.y) +
                            std::abs(p.z-rest.z) < 1.0e-4F,
                            "pinned top node moved");
            }
            for (const SoftBodyBond &bond : scene.soft_bodies[body_index].bonds) {
                const Vec3 a = positions[bond.first], b = positions[bond.second];
                const float length = std::sqrt((a.x-b.x)*(a.x-b.x) +
                    (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
                maximum_strain = std::max(maximum_strain,
                    std::abs(length / bond.rest_length - 1.0F));
            }
        }
        SmokeDeviceView smoke{};
        require(world.smoke_view(instance.smoke, smoke), "read smoke");
        require(smoke.particle_count > 0U, "smoke was not emitted");
        require(displaced > 100U && maximum_speed < 10.0F,
                "soft bodies did not move stably");
        std::cout << "frames=" << frames << " ms_per_frame=" << elapsed / frames
                  << " displaced_nodes=" << displaced
                  << " max_soft_speed=" << maximum_speed
                  << " max_bond_strain=" << maximum_strain
                  << " smoke_particles=" << smoke.particle_count << '\n';

        World scene_reference_world;
        SceneInstance scene_reference;
        require(create_scene_world(scene, scene_reference_world,
                                   scene_reference), "create gallery reference");
        for (const auto coupling : scene_reference.smoke_soft_body_couplings)
            require(scene_reference_world.remove_smoke_soft_body_coupling(coupling),
                    "disable gallery reference coupling");
        for (unsigned frame = 0; frame < frames; ++frame)
            require(scene_reference_world.step({.timestep = 1.0F / 60.0F,
                                                .substeps = 4U}),
                    "step gallery reference");
        SmokeDeviceView reference_smoke{};
        require(scene_reference_world.smoke_view(scene_reference.smoke,
                                                reference_smoke),
                "read gallery reference smoke");
        const auto gallery_smoke_positions = read(smoke.positions);
        const auto reference_smoke_positions = read(reference_smoke.positions);
        float gallery_contact_difference = 0.0F;
        for (std::size_t index = 0; index < gallery_smoke_positions.size(); ++index)
            gallery_contact_difference += std::abs(gallery_smoke_positions[index].x -
                                                  reference_smoke_positions[index].x) +
                std::abs(gallery_smoke_positions[index].y -
                         reference_smoke_positions[index].y) +
                std::abs(gallery_smoke_positions[index].z -
                         reference_smoke_positions[index].z);
        std::cout << "gallery_smoke_contact_difference="
                  << gallery_contact_difference << '\n';
        require(gallery_contact_difference > 1.0F,
                "authored gallery plume misses all soft bodies");

        // Isolate one column to prove both directions of the coupling: wind
        // bends it, and its moving skin changes the smoke tracer paths.
        SceneDefinition isolated = scene;
        isolated.soft_bodies.resize(1U);
        isolated.smoke_options.emitter_center = {-0.35F, 0.7F, 0.0F};
        isolated.smoke_options.emitter_half_extents = {0.1F, 0.05F};
        World coupled_world, uncoupled_world;
        SceneInstance coupled, uncoupled;
        require(create_scene_world(isolated, coupled_world, coupled),
                "create isolated coupled world");
        require(create_scene_world(isolated, uncoupled_world, uncoupled),
                "create isolated uncoupled world");
        require(uncoupled_world.remove_smoke_soft_body_coupling(
                    uncoupled.smoke_soft_body_couplings.front()),
                "disable reference coupling");
        for (unsigned frame = 0; frame < 90U; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                                   .substeps = 4U, .gravity = {}};
            require(coupled_world.step(step), "step isolated coupling");
            require(uncoupled_world.step(step), "step isolated reference");
        }
        SoftBodyDeviceView windy_body{}, still_body{};
        require(coupled_world.soft_body_view(coupled.soft_bodies[0], windy_body),
                "read wind-coupled body");
        require(uncoupled_world.soft_body_view(uncoupled.soft_bodies[0], still_body),
                "read reference body");
        const auto windy_positions = read(windy_body.positions);
        const auto still_positions = read(still_body.positions);
        float body_difference = 0.0F;
        for (std::size_t node = 0; node < windy_positions.size(); ++node)
            body_difference += std::abs(windy_positions[node].x -
                                        still_positions[node].x);
        SmokeDeviceView windy_smoke{}, still_smoke{};
        require(coupled_world.smoke_view(coupled.smoke, windy_smoke),
                "read coupled smoke");
        require(uncoupled_world.smoke_view(uncoupled.smoke, still_smoke),
                "read reference smoke");
        const auto windy_tracers = read(windy_smoke.positions);
        const auto still_tracers = read(still_smoke.positions);
        float smoke_difference = 0.0F;
        for (std::size_t index = 0; index < windy_tracers.size(); ++index)
            smoke_difference += std::abs(windy_tracers[index].x -
                                         still_tracers[index].x) +
                                std::abs(windy_tracers[index].y -
                                         still_tracers[index].y) +
                                std::abs(windy_tracers[index].z -
                                         still_tracers[index].z);
        std::cout << "body_wind_difference=" << body_difference
                  << " smoke_contact_difference=" << smoke_difference << '\n';
        require(body_difference > 0.1F && smoke_difference > 0.1F,
                "smoke and soft body did not influence each other");

        for (const auto coupling : instance.smoke_soft_body_couplings)
            require(world.remove_smoke_soft_body_coupling(coupling),
                    "remove smoke soft-body coupling");
        require(!world.remove_smoke_soft_body_coupling(
                    instance.smoke_soft_body_couplings.front()),
                "stale smoke soft-body coupling was accepted");
        require(world.remove_smoke(instance.smoke), "remove smoke");
        require(world.remove_soft_body(instance.soft_bodies.front()),
                "remove soft body");
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
