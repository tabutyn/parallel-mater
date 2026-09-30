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
        require(load_glb_scene(PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH,
                               scene, error), error.c_str());
        require(scene.has_smoke && scene.ropes.size() == 4U &&
                scene.rigid_bodies.size() == 5U,
                "SmokeRope must export four ropes, posts, panel, sphere, and ground");
        const int panel = scene.ropes.front().first_body;
        require(panel >= 0 && scene.rigid_bodies[panel].source_name == "Plane.001",
                "active panel endpoint was not inferred");
        for (const auto &rope : scene.ropes)
            require(rope.first_body == panel && rope.last_body >= 0 &&
                    rope.last_body != panel && rope.centerline.size() >= 2U,
                    "rope must join the active panel to a post");

        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "GPU runtime check skipped: CUDA unavailable\n";
            return 77;
        }
        World world, reference_world;
        SceneInstance coupled, reference;
        require(create_scene_world(scene, world, coupled),
                "create smoke-rope world");
        require(create_scene_world(scene, reference_world, reference),
                "create uncoupled reference world");
        require(coupled.smoke_rope_couplings.size() == 4U,
                "four smoke-rope API couplings were not registered");
        require(coupled.smoke_rigid_couplings.size() == 2U,
                "sphere and active panel smoke couplings were not registered");
        require(!world.remove_smoke(coupled.smoke),
                "referenced smoke was removed");
        require(!world.remove_rope(coupled.ropes.front()),
                "referenced rope was removed");
        SmokeRopeCouplingId rejected{};
        require(!world.add_smoke_rope_coupling(
                    {.smoke = coupled.smoke, .rope = coupled.ropes.front(),
                     .wind_drag = -1.0F}, rejected),
                "negative smoke drag accepted");
        require(!world.add_smoke_rope_coupling(
                    {.smoke = coupled.smoke, .rope = coupled.ropes.front()},
                    rejected), "duplicate smoke-rope coupling accepted");
        for (const auto coupling : reference.smoke_rope_couplings)
            require(reference_world.remove_smoke_rope_coupling(coupling),
                    "remove reference coupling");
        for (const auto coupling : reference.smoke_rigid_couplings)
            require(reference_world.remove_smoke_rigid_coupling(coupling),
                    "remove reference panel coupling");

        constexpr unsigned frames = 300U;
        double coupled_milliseconds = 0.0, reference_milliseconds = 0.0;
        for (unsigned frame = 0; frame < frames; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                                   .substeps = 4U,
                                   .gravity = {0.0F, -9.81F, 0.0F}};
            auto begin = std::chrono::steady_clock::now();
            require(world.step(step), "step smoke-rope scene");
            coupled_milliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            begin = std::chrono::steady_clock::now();
            require(reference_world.step(step), "step reference scene");
            reference_milliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        }
        float rope_difference = 0.0F, maximum_strain = 0.0F;
        for (std::size_t index = 0; index < coupled.ropes.size(); ++index) {
            RopeDeviceView rope{}, still_rope{};
            require(world.rope_view(coupled.ropes[index], rope), "read rope");
            require(reference_world.rope_view(reference.ropes[index], still_rope),
                    "read reference rope");
            const auto positions = read(rope.positions);
            const auto velocities = read(rope.velocities);
            const auto rest = read(rope.rest_lengths);
            const auto reference_positions = read(still_rope.positions);
            require(positions.size() > 2U && rest.size() + 1U == positions.size(),
                    "rope sampling failed");
            for (std::size_t node = 0; node < positions.size(); ++node) {
                const Vec3 p = positions[node], v = velocities[node];
                require(std::isfinite(p.x) && std::isfinite(p.y) &&
                        std::isfinite(p.z) && std::isfinite(v.x) &&
                        std::isfinite(v.y) && std::isfinite(v.z),
                        "nonfinite rope state");
                rope_difference += distance(p, reference_positions[node]);
                if (node + 1U < positions.size())
                    maximum_strain = std::max(maximum_strain,
                        std::abs(distance(p, positions[node + 1U]) /
                                 rest[node] - 1.0F));
            }
        }
        RigidBodyState panel_state{}, reference_panel_state{};
        require(world.read_rigid_body_state(coupled.rigid_bodies[panel], panel_state),
                "read coupled panel");
        require(reference_world.read_rigid_body_state(
                    reference.rigid_bodies[panel], reference_panel_state),
                "read reference panel");
        const float panel_difference = distance(
            panel_state.position, reference_panel_state.position);
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
                  << " rope_difference=" << rope_difference
                  << " panel_difference=" << panel_difference
                  << " tracer_difference=" << tracer_difference
                  << " max_rope_strain=" << maximum_strain
                  << " smoke_particles=" << smoke.particle_count << '\n';
        require(maximum_strain < 0.5F,
                "smoke-rope coupling stretched a rope unstably");
        require(rope_difference > 0.01F && tracer_difference > 1.0F,
                "smoke wind did not bend the suspended ropes");
        require(panel_difference > 0.03F,
                "smoke pressure did not move the suspended panel");

        // The authored narrow plume passes between the four corner ropes.
        // Aim a second, otherwise identical emitter at a lower rope to prove
        // live capsule contact independently of that scene composition.
        SceneDefinition contact_scene = scene;
        contact_scene.smoke_options.capacity = 1200U;
        contact_scene.smoke_options.emitter_center = {0.45F, 0.36F, -1.3F};
        contact_scene.smoke_options.emitter_half_extents = {0.16F, 0.22F};
        contact_scene.smoke_options.initial_velocity = {2.0F, 0.0F, 0.0F};
        contact_scene.smoke_options.wind = {2.0F, 0.0F, 0.0F};
        contact_scene.smoke_options.wake_strength = 0.0F;
        contact_scene.smoke_options.buoyancy = 0.0F;
        contact_scene.smoke_options.particles_per_second = 300.0F;
        World contact_world, pass_world;
        SceneInstance contact, pass;
        require(create_scene_world(contact_scene, contact_world, contact),
                "create focused smoke-rope contact world");
        require(create_scene_world(contact_scene, pass_world, pass),
                "create focused reference world");
        for (const auto coupling : contact.smoke_rigid_couplings)
            require(contact_world.remove_smoke_rigid_coupling(coupling),
                    "remove focused panel coupling");
        for (const auto coupling : pass.smoke_rigid_couplings)
            require(pass_world.remove_smoke_rigid_coupling(coupling),
                    "remove focused reference panel coupling");
        for (const auto coupling : pass.smoke_rope_couplings)
            require(pass_world.remove_smoke_rope_coupling(coupling),
                    "remove focused reference coupling");
        for (unsigned frame = 0; frame < 120U; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                                   .substeps = 4U, .gravity = {}};
            require(contact_world.step(step), "step focused contact world");
            require(pass_world.step(step), "step focused reference world");
        }
        SmokeDeviceView contact_smoke{}, pass_smoke{};
        require(contact_world.smoke_view(contact.smoke, contact_smoke),
                "read focused smoke");
        require(pass_world.smoke_view(pass.smoke, pass_smoke),
                "read focused reference smoke");
        const auto contact_positions = read(contact_smoke.positions);
        const auto pass_positions = read(pass_smoke.positions);
        float contact_difference = 0.0F;
        for (std::size_t index = 0; index < contact_positions.size(); ++index)
            contact_difference += distance(contact_positions[index],
                                           pass_positions[index]);
        std::cout << "focused_tracer_difference=" << contact_difference << '\n';
        require(contact_difference > 1.0F,
                "smoke tracers did not deflect from rope capsules");

        // A fast tracer must not jump through the thin active panel in one
        // frame; the rigid coupling uses the authored moving triangles.
        SceneDefinition crossing_scene = scene;
        crossing_scene.smoke_options.capacity = 8U;
        crossing_scene.smoke_options.particles_per_second = 30.0F;
        crossing_scene.smoke_options.emitter_center = {1.0F, 1.2F, 0.0F};
        crossing_scene.smoke_options.emitter_half_extents = {0.001F, 0.001F};
        crossing_scene.smoke_options.initial_velocity = {20.0F, 0.0F, 0.0F};
        crossing_scene.smoke_options.wind = {20.0F, 0.0F, 0.0F};
        crossing_scene.smoke_options.maximum_speed = 25.0F;
        crossing_scene.smoke_options.wake_strength = 0.0F;
        crossing_scene.smoke_options.buoyancy = 0.0F;
        World crossing_world;
        SceneInstance crossing;
        require(create_scene_world(crossing_scene, crossing_world, crossing),
                "create fast-tracer panel world");
        for (const auto coupling : crossing.smoke_rope_couplings)
            require(crossing_world.remove_smoke_rope_coupling(coupling),
                    "remove crossing rope coupling");
        for (const auto coupling : crossing.smoke_rigid_couplings)
            require(crossing_world.remove_smoke_rigid_coupling(coupling),
                    "remove default crossing rigid coupling");
        SmokeRigidCouplingId contact_only{};
        require(crossing_world.add_smoke_rigid_coupling(
                    {.smoke = crossing.smoke,
                     .body = crossing.rigid_bodies[panel],
                     .drag_coefficient = 0.0F}, contact_only),
                "create contact-only panel coupling");
        SmokeRigidCouplingId duplicate_panel{};
        require(!crossing_world.add_smoke_rigid_coupling(
                    {.smoke = crossing.smoke,
                     .body = crossing.rigid_bodies[panel]}, duplicate_panel),
                "duplicate smoke-rigid coupling accepted");
        require(!crossing_world.add_smoke_rigid_coupling(
                    {.smoke = crossing.smoke,
                     .body = crossing.rigid_bodies[panel],
                     .air_density = -1.0F}, duplicate_panel),
                "negative smoke density accepted");
        for (unsigned frame = 0; frame < 2U; ++frame)
            require(crossing_world.step({.timestep = 1.0F / 30.0F,
                                         .substeps = 4U, .gravity = {}}),
                    "step fast-tracer panel world");
        SmokeDeviceView crossed_smoke{};
        require(crossing_world.smoke_view(crossing.smoke, crossed_smoke),
                "read fast-tracer smoke");
        const auto crossing_positions = read(crossed_smoke.positions);
        require(!crossing_positions.empty() && crossing_positions.front().x < 1.33F,
                "fast tracer crossed the rigid panel");
        for (const auto coupling : coupled.smoke_rope_couplings)
            require(world.remove_smoke_rope_coupling(coupling),
                    "remove smoke-rope coupling");
        for (const auto coupling : coupled.smoke_rigid_couplings)
            require(world.remove_smoke_rigid_coupling(coupling),
                    "remove smoke-rigid coupling");
        require(!world.remove_smoke_rope_coupling(
                    coupled.smoke_rope_couplings.front()),
                "stale coupling handle accepted");
        require(world.remove_smoke(coupled.smoke), "remove smoke");
        for (const auto rope : coupled.ropes)
            require(world.remove_rope(rope), "remove rope");
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
