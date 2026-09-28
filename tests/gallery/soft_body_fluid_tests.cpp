// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include "../../examples/support/vector_math.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::gallery;
namespace math = parallel_mater::gallery::math;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void check(Status status, const char *message) {
    if (!status) throw std::runtime_error(std::string(message) + ": " + status.message);
}
template<class T> std::vector<T> read(DeviceSpan<const T> span) {
    std::vector<T> output(span.size);
    if (span.size) check(cudaMemcpy(output.data(), span.data, span.size * sizeof(T),
        cudaMemcpyDeviceToHost) == cudaSuccess, "device read");
    return output;
}
float length(Vec3 p) { return std::sqrt(math::dot(p,p)); }
bool inside(Vec3 p, const std::vector<Vec3> &skin, const std::vector<std::uint32_t> &indices) {
    Vec3 lo = skin[0], hi = skin[0];
    for (Vec3 v : skin) {
        lo = {std::min(lo.x,v.x), std::min(lo.y,v.y), std::min(lo.z,v.z)};
        hi = {std::max(hi.x,v.x), std::max(hi.y,v.y), std::max(hi.z,v.z)};
    }
    if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) return false;
    double angle = 0;
    for (std::size_t i = 0; i < indices.size(); i += 3) {
        const auto a = math::subtract(skin[indices[i]], p);
        const auto b = math::subtract(skin[indices[i+1]], p);
        const auto c = math::subtract(skin[indices[i+2]], p);
        const double x = length(a), y = length(b), z = length(c);
        angle += 2 * std::atan2(double(math::dot(a, math::cross(b,c))),
            x*y*z + math::dot(a,b)*z + math::dot(b,c)*x + math::dot(c,a)*y);
    }
    return std::abs(angle) > 6.283185307179586;
}

int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        SceneDefinition scene;
        std::string error;
        check(load_glb_scene(PARALLEL_MATER_SOFT_BODY_FLUID_SCENE_PATH, scene, error), error.c_str());
        check(scene.soft_bodies.size() == 1 && scene.spawn_planes.size() == 1 &&
              scene.destroy_planes.size() == 1, "authored systems missing");
        const auto &definition = scene.soft_bodies[0];
        const auto pins = std::count(definition.inverse_masses.begin(), definition.inverse_masses.end(), 0.0F);
        check(pins == 4, "Goal group must fix exactly four authored vertices");
        check(definition.shape_matching_stiffness == 0, "Goal group must not enable global shape matching");
        std::cout << "nodes=" << definition.nodes.size() << " bonds=" << definition.bonds.size()
                  << " pins=" << pins << std::endl;
        scene.fluid_options.capacity = argc > 3 ? std::stoul(argv[3]) : 4000;
        if (argc > 2 && std::string(argv[2]) == "--dry") {
            scene.fluid_options.capacity = 0;
            scene.spawn_planes.clear();
            scene.destroy_planes.clear();
            World dry;
            SceneInstance dry_instance;
            check(create_scene_world(scene, dry, dry_instance), "dry scene");
            for (int frame = 0; frame < 240; ++frame)
                check(dry.step({.timestep = 1.0F/60, .substeps = 4, .gravity = {0,-9.81F,0}}), "dry step");
            SoftBodyDeviceView dry_view;
            check(dry.soft_body_view(dry_instance.soft_bodies[0], dry_view), "dry view");
            const auto positions = read(dry_view.positions);
            float displacement = 0;
            for (std::size_t i = 0; i < positions.size(); ++i)
                displacement = std::max(displacement, length(math::subtract(positions[i], definition.nodes[i])));
            std::cout << "dry_displacement=" << displacement << std::endl;
            return 0;
        }
        World world;
        SceneInstance instance;
        check(create_scene_world(scene, world, instance), "create scene");
        check(instance.fluid_soft_body_couplings.size() == 1, "coupling missing");
        auto coupling = instance.fluid_soft_body_couplings[0];
        FluidSoftBodyCouplingOptions coupling_options{.fluid = instance.fluid,
                                                     .soft_body = instance.soft_bodies[0]};
        FluidSoftBodyCouplingId duplicate;
        check(!world.add_fluid_soft_body_coupling(coupling_options, duplicate), "duplicate accepted");
        check(!world.remove_soft_body(instance.soft_bodies[0]), "referenced soft body removed");
        check(!world.remove_fluid(instance.fluid), "referenced fluid removed");
        auto invalid = coupling_options; invalid.friction = -1;
        check(!world.update_fluid_soft_body_coupling(coupling, invalid), "invalid options accepted");
        const int frames = argc > 1 ? std::stoi(argv[1]) : 600;
        float max_pin = 0, max_speed = 0, max_force = 0, max_displacement = 0;
        double gpu = 0, coupling_gpu = 0;
        std::uint64_t contacts = 0, inside_count = 0;
        WorldStatistics stats{};
        for (int frame = 0; frame < frames; ++frame) {
            Vec3 gravity{0,-9.81F,0};
            if (argc > 2 && std::string(argv[2]) == "--tilt" && frame >= 240) {
                const float sign = ((frame - 240) / 180) % 2 == 0 ? 1.0F : -1.0F;
                gravity = {6.936718F * sign,-6.936718F,0};
            }
            check(world.step({.timestep = 1.0F / 60, .substeps = 4,
                .gravity = gravity, .collect_kernel_timings = true}), "step");
            SoftBodyDeviceView view;
            check(world.soft_body_view(instance.soft_bodies[0], view), "soft view");
            const auto positions = read(view.positions), velocities = read(view.velocities);
            const auto forces = read(view.fluid_contact_forces);
            for (std::size_t node = 0; node < positions.size(); ++node) {
                check(std::isfinite(length(positions[node])) && std::isfinite(length(velocities[node])) &&
                      std::isfinite(length(forces[node])), "nonfinite soft state");
                const auto displacement = length(math::subtract(positions[node], definition.nodes[node]));
                if (definition.inverse_masses[node] == 0) max_pin = std::max(max_pin, displacement);
                max_displacement = std::max(max_displacement, displacement);
                max_speed = std::max(max_speed, length(velocities[node]));
                max_force = std::max(max_force, length(forces[node]));
            }
            check(world.collect_statistics(stats), "stats");
            contacts += stats.fluid_soft_body_contact_count;
            WorldStepTimings timing{};
            check(world.collect_step_timings(timing), "timings");
            gpu += timing.total_gpu_milliseconds;
            coupling_gpu += timing.fluid_soft_body_contacts.total_milliseconds;
            if (frame % 10 == 0) {
                FluidDeviceView water;
                check(world.fluid_view(instance.fluid, water), "fluid view");
                const auto skin = read(view.surface_positions);
                const auto indices = read(view.surface_triangle_indices);
                for (Vec3 p : read(water.positions)) {
                    check(std::isfinite(length(p)), "nonfinite water position");
                    if (!inside(p, skin, indices)) continue;
                    if (inside_count < 5) std::cout << "inside frame=" << frame << " point="
                        << p.x << ',' << p.y << ',' << p.z << std::endl;
                    ++inside_count;
                }
            }
            if (frame % 120 == 119) std::cout << "frame=" << frame+1 << " particles=" << stats.particle_count
                << " contacts=" << contacts << " inside=" << inside_count << " speed=" << max_speed
                << " deflection=" << max_displacement << std::endl;
        }
        std::cout << "pins=" << max_pin << " speed=" << max_speed << " force=" << max_force
                  << " deflection=" << max_displacement << " inside=" << inside_count
                  << " outflow=" << stats.destroyed_particle_count << " gpu_ms=" << gpu/frames
                  << " coupling_ms=" << coupling_gpu/frames << std::endl;
        check(max_pin < 1.0e-6F, "Goal pins drifted");
        check(max_speed <= definition.maximum_speed + 1.0e-3F, "fluid drove soft body beyond speed bound");
        check(max_displacement < 3.0F, "anchored slab stretched without bound");
        check(contacts > 0 && max_force > 0, "no fluid soft reaction");
        check(inside_count == 0, "water crossed inside soft skin");
        check(stats.destroyed_particle_count > 0, "outflow removed no particles");
        check(world.remove_fluid_soft_body_coupling(coupling), "remove coupling");
        check(!world.update_fluid_soft_body_coupling(coupling, coupling_options), "stale handle accepted");
        check(world.add_fluid_soft_body_coupling(coupling_options, duplicate), "readd coupling");
        check(duplicate.generation != coupling.generation, "generation not advanced");
        check(world.remove_fluid_soft_body_coupling(duplicate), "remove replacement");
        check(world.remove_soft_body(instance.soft_bodies[0]), "remove body");
        check(world.remove_fluid(instance.fluid), "remove fluid");
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
