// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
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
    std::vector<T> result(source.size);
    if (!result.empty())
        require(cudaMemcpy(result.data(), source.data,
            result.size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess,
            "smoke device read failed");
    return result;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        SceneDefinition scene;
        std::string error;
        require(load_glb_scene(PARALLEL_MATER_SMOKE_SCENE_PATH, scene, error),
                error.c_str());
        require(scene.has_smoke && scene.rigid_bodies.size() == 1U &&
                scene.smoke_obstacle_name == "VortexSphere",
                "Smoke.blend physics metadata was not exported");
        require(scene.fluid_options.capacity == 0U,
                "smoke was incorrectly routed through liquid physics");
        SceneDefinition free_smoke = scene;
        free_smoke.rigid_bodies.clear();
        free_smoke.smoke_obstacle_name.clear();
        World free_world;
        SceneInstance free_instance;
        require(create_scene_world(free_smoke, free_world, free_instance),
                "create particle gas without any rigid obstacle");
        require(free_instance.smoke_rigid_couplings.empty(),
                "obstacle-free smoke registered an unwanted rigid coupling");
        require(free_world.step({.timestep = 1.0F / 60.0F,
                .substeps = 1U, .gravity = {}}),
                "emit smoke without a rigid obstacle");
        World world;
        SceneInstance instance;
        require(create_scene_world(scene, world, instance), "create smoke scene");
        require(instance.has_smoke && !instance.has_fluid,
                "smoke API resource was not instantiated");
        require(!world.remove_rigid_body(instance.rigid_bodies[0]),
                "smoke obstacle was removed while referenced");
        SmokeId invalid{};
        SmokeOptions bad = scene.smoke_options;
        bad.capacity = 0U;
        require(!world.add_smoke(bad, invalid), "zero-capacity smoke accepted");

        SceneDefinition unobstructed = scene;
        World reference_world;
        SceneInstance reference;
        require(create_scene_world(unobstructed, reference_world, reference),
                "create no-contact comparison");
        require(reference_world.remove_smoke_rigid_coupling(
                    reference.smoke_rigid_couplings.front()),
                "remove reference obstacle contact");
        constexpr unsigned frames = 300U;
        for (unsigned frame = 0; frame < frames; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                .substeps = 1U, .gravity = {0.0F, 0.0F, 0.0F}};
            require(world.step(step), "step smoke");
            require(reference_world.step(step), "step smoke comparison");
        }
        SmokeDeviceView smoke{}, no_wake{};
        require(world.smoke_view(instance.smoke, smoke), "borrow smoke");
        require(reference_world.smoke_view(reference.smoke, no_wake),
                "borrow smoke comparison");
        require(smoke.particle_count == scene.smoke_options.capacity &&
                no_wake.particle_count == smoke.particle_count,
                "smoke ring did not fill to its bounded capacity");
        const auto positions = read(smoke.positions);
        const auto velocities = read(smoke.velocities);
        const auto comparison = read(no_wake.velocities);
        const auto ages = read(smoke.ages);
        const auto reference_positions = read(no_wake.positions);
        const auto reference_ages = read(no_wake.ages);
        const auto vorticities = read(smoke.vorticities);
        const auto reference_vorticities = read(no_wake.vorticities);
        const auto densities = read(smoke.number_densities);
        const auto pressures = read(smoke.pressures);
        const Vec3 center = scene.rigid_bodies[0].options.initial_state.position;
        float radius = 0.0F;
        const auto &obstacle_mesh = scene.meshes[
            scene.rigid_bodies[0].mesh_indices[0]];
        for (const auto &vertex : obstacle_mesh.vertices)
            radius = std::max(radius, std::sqrt(
                vertex.position.x * vertex.position.x +
                vertex.position.y * vertex.position.y +
                vertex.position.z * vertex.position.z));
        unsigned wake_particles = 0U;
        float wake_difference = 0.0F;
        float maximum_speed = 0.0F;
        float maximum_pressure = 0.0F;
        float near_wall_speed = 0.0F;
        float near_wall_pressure = 0.0F, far_pressure = 0.0F;
        float far_speed = 0.0F;
        unsigned near_wall_particles = 0U;
        unsigned far_particles = 0U;
        float lee_vorticity = 0.0F, free_vorticity = 0.0F;
        unsigned lee_curl_count = 0U, free_curl_count = 0U;
        for (std::size_t index = 0; index < positions.size(); ++index) {
            const auto &p = positions[index];
            const auto &v = velocities[index];
            require(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
                    std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z),
                    "non-finite smoke state");
            if (ages[index] >= smoke.lifetime) continue;
            require(std::isfinite(densities[index]) &&
                    std::isfinite(pressures[index]) && pressures[index] >= 0.0F,
                    "non-finite smoke pressure state");
            maximum_pressure = std::max(maximum_pressure, pressures[index]);
            const float distance = std::sqrt((p.x-center.x)*(p.x-center.x) +
                (p.y-center.y)*(p.y-center.y) +
                (p.z-center.z)*(p.z-center.z));
            require(distance >= radius - 1.0e-3F,
                    "smoke particle penetrated spherical obstacle");
            const float speed = std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
            maximum_speed = std::max(maximum_speed, speed);
            if (distance < radius + 0.1F) {
                ++near_wall_particles;
                near_wall_speed += speed;
                near_wall_pressure += pressures[index];
            }
            if (p.x > center.x + 3.0F * radius &&
                p.x < center.x + 5.0F * radius) {
                ++far_particles;
                far_speed += speed;
                far_pressure += pressures[index];
            }
            if (p.x > center.x + radius &&
                p.x < center.x + 3.0F * radius) {
                lee_vorticity += std::sqrt(
                    vorticities[index].x * vorticities[index].x +
                    vorticities[index].y * vorticities[index].y +
                    vorticities[index].z * vorticities[index].z);
                ++lee_curl_count;
            }
            if (p.x < center.x + radius) continue;
            ++wake_particles;
            wake_difference += std::abs(v.y - comparison[index].y) +
                               std::abs(v.z - comparison[index].z);
        }
        for (std::size_t index = 0; index < reference_positions.size(); ++index) {
            if (reference_ages[index] >= no_wake.lifetime) continue;
            const Vec3 p = reference_positions[index];
            if (p.x <= center.x + radius ||
                p.x >= center.x + 3.0F * radius) continue;
            free_vorticity += std::sqrt(
                reference_vorticities[index].x * reference_vorticities[index].x +
                reference_vorticities[index].y * reference_vorticities[index].y +
                reference_vorticities[index].z * reference_vorticities[index].z);
            ++free_curl_count;
        }
        require(wake_particles > 100U, "smoke did not travel behind the sphere");
        std::cout << "wake_particles=" << wake_particles
                  << " mean_transverse_wake_delta="
                  << wake_difference / wake_particles
                  << " maximum_pressure=" << maximum_pressure
                  << " near_wall_particles=" << near_wall_particles
                  << " mean_near_wall_speed=" << near_wall_speed /
                     std::max(1U, near_wall_particles)
                  << " mean_near_wall_pressure=" << near_wall_pressure /
                     std::max(1U, near_wall_particles)
                  << " mean_far_speed=" << far_speed /
                     std::max(1U, far_particles)
                  << " mean_far_pressure=" << far_pressure /
                     std::max(1U, far_particles)
                  << " far_particles=" << far_particles
                  << " mean_lee_curl=" << lee_vorticity /
                     std::max(1U, lee_curl_count)
                  << " mean_free_curl=" << free_vorticity /
                     std::max(1U, free_curl_count) << '\n';
        require(wake_difference / wake_particles > 0.02F,
                "obstacle contact did not alter downstream particle motion");
        require(lee_curl_count > 100U && free_curl_count > 100U &&
                lee_vorticity / lee_curl_count >
                    1.25F * free_vorticity / free_curl_count,
                "obstacle did not increase locally measured downstream curl");
        require(maximum_pressure > 0.0F,
                "neighbor crowding did not generate positive pressure");
        require(near_wall_particles > 20U &&
                near_wall_speed / near_wall_particles <
                    0.3F * std::sqrt(
                        scene.smoke_options.wind.x * scene.smoke_options.wind.x +
                        scene.smoke_options.wind.y * scene.smoke_options.wind.y +
                        scene.smoke_options.wind.z * scene.smoke_options.wind.z),
                "smoke did not approach the rigid surface velocity");
        require(far_particles > 20U &&
                near_wall_speed / near_wall_particles <
                    0.3F * far_speed / far_particles &&
                near_wall_pressure / near_wall_particles >
                    far_pressure / far_particles,
                "stagnated near-wall smoke did not show lower speed and higher pressure");
        require(maximum_speed <= scene.smoke_options.maximum_speed + 0.01F,
                "smoke exceeded its speed cap");
        require(world.step({.timestep = 1.0F / 60.0F, .substeps = 1U,
                .gravity = {0.0F, 0.0F, 0.0F},
                .collect_kernel_timings = true}), "profile smoke step");
        WorldStepTimings timings{};
        WorldStatistics statistics{};
        require(world.collect_step_timings(timings), "smoke timings");
        require(world.collect_statistics(statistics), "smoke statistics");
        require(timings.available && timings.smoke_advection.launch_count == 1U &&
                timings.smoke_emission.launch_count == 1U &&
                statistics.smoke_particle_count == scene.smoke_options.capacity &&
                statistics.emitted_smoke_particle_count > 0U,
                "smoke timing or particle statistics were not exposed");
        require(instance.smoke_rigid_couplings.size() == 1U,
                "static mesh obstacle was not coupled to smoke");
        require(world.remove_smoke_rigid_coupling(
                    instance.smoke_rigid_couplings.front()),
                "remove static smoke-mesh coupling");
        require(world.remove_smoke(instance.smoke), "remove smoke");
        require(!world.smoke_view(instance.smoke, smoke),
                "stale smoke handle was accepted");
        require(world.remove_rigid_body(instance.rigid_bodies[0]),
                "remove former smoke obstacle");

        // A large external timestep must not let a tracer jump from the
        // upwind side to the lee side through the sphere.
        SceneDefinition swept = scene;
        swept.smoke_options.capacity = 16U;
        swept.smoke_options.particles_per_second = 1.0F;
        swept.smoke_options.lifetime = 10.0F;
        World swept_world;
        SceneInstance swept_instance;
        require(create_scene_world(swept, swept_world, swept_instance),
                "create swept-collision smoke");
        require(swept_world.step({.timestep = 1.0F, .substeps = 1U,
                .gravity = {0.0F, 0.0F, 0.0F}}), "emit swept tracer");
        require(swept_world.step({.timestep = 2.0F, .substeps = 1U,
                .gravity = {0.0F, 0.0F, 0.0F}}), "advance swept tracer");
        SmokeDeviceView swept_view{};
        require(swept_world.smoke_view(swept_instance.smoke, swept_view),
                "borrow swept tracer");
        const auto swept_positions = read(swept_view.positions);
        require(!swept_positions.empty() && swept_positions[0].x < center.x,
                "smoke tunneled through the sphere at a coarse timestep");

        // A rectangular, non-spherical authored obstacle must use its
        // triangles for the same swept contact, without a radius fallback.
        SceneDefinition box = swept;
        const std::uint32_t box_mesh_index =
            box.rigid_bodies[0].mesh_indices.front();
        auto &box_mesh = box.meshes[box_mesh_index];
        box_mesh.vertices.clear();
        for (const Vec3 point : {Vec3{-0.2F,-0.35F,-0.6F},
                 Vec3{0.2F,-0.35F,-0.6F}, Vec3{0.2F,0.35F,-0.6F},
                 Vec3{-0.2F,0.35F,-0.6F}, Vec3{-0.2F,-0.35F,0.6F},
                 Vec3{0.2F,-0.35F,0.6F}, Vec3{0.2F,0.35F,0.6F},
                 Vec3{-0.2F,0.35F,0.6F}})
            box_mesh.vertices.push_back({.position = point});
        box_mesh.indices = {0,2,1, 0,3,2, 4,5,6, 4,6,7,
                            0,4,7, 0,7,3, 1,2,6, 1,6,5,
                            0,1,5, 0,5,4, 3,7,6, 3,6,2};
        box.rigid_bodies[0].collision_mesh_indices.clear();
        World box_world;
        SceneInstance box_instance;
        require(create_scene_world(box, box_world, box_instance),
                "create non-spherical smoke obstacle");
        require(box_world.step({.timestep = 1.0F, .substeps = 1U,
                .gravity = {}}), "emit smoke toward rectangular obstacle");
        require(box_world.step({.timestep = 2.0F, .substeps = 1U,
                .gravity = {}}), "sweep smoke against rectangular obstacle");
        SmokeDeviceView box_smoke{};
        require(box_world.smoke_view(box_instance.smoke, box_smoke),
                "read rectangular-obstacle smoke");
        const auto box_positions = read(box_smoke.positions);
        require(!box_positions.empty() &&
                box_positions.front().x < center.x - 0.15F,
                "smoke tunneled through non-spherical triangles");

        // A translating active sphere must be accepted by the gallery and
        // remain solid to smoke moving against it.
        SceneDefinition moving = scene;
        moving.rigid_bodies[0].options.motion = MotionType::dynamic;
        moving.rigid_bodies[0].options.initial_state.linear_velocity =
            {-1.2F, 0.0F, 0.0F};
        moving.smoke_options.capacity = 256U;
        moving.smoke_options.particles_per_second = 200.0F;
        World moving_world;
        SceneInstance moving_instance;
        require(create_scene_world(moving, moving_world, moving_instance),
                "create moving-obstacle smoke");
        require(moving_instance.smoke_rigid_couplings.size() == 1U,
                "dynamic obstacle lacks reusable smoke-rigid coupling");
        require(moving_world.remove_smoke_rigid_coupling(
                    moving_instance.smoke_rigid_couplings.front()),
                "replace moving obstacle force with contact only");
        SmokeRigidCouplingId moving_contact{};
        require(moving_world.add_smoke_rigid_coupling(
                    {.smoke = moving_instance.smoke,
                     .body = moving_instance.rigid_bodies[0],
                     .air_density = 0.0F}, moving_contact),
                "restore moving triangle contact without carrier drag");
        float final_center_x = center.x;
        for (unsigned frame = 0U; frame < 60U; ++frame) {
            require(moving_world.step({.timestep = 1.0F / 60.0F,
                    .substeps = 1U, .gravity = {}}),
                    "advance moving-obstacle smoke");
            RigidBodyDeviceView bodies{};
            SmokeDeviceView moving_smoke{};
            require(moving_world.rigid_body_view(bodies),
                    "read moving smoke obstacle");
            require(moving_world.smoke_view(moving_instance.smoke, moving_smoke),
                    "read moving smoke tracers");
            const auto body_states = read(bodies.states);
            const auto moving_positions = read(moving_smoke.positions);
            const auto moving_ages = read(moving_smoke.ages);
            const Vec3 moving_center = body_states[0].position;
            final_center_x = moving_center.x;
            for (std::size_t i = 0; i < moving_positions.size(); ++i) {
                if (moving_ages[i] >= moving_smoke.lifetime) continue;
                const Vec3 p = moving_positions[i];
                const float distance = std::sqrt(
                    (p.x - moving_center.x) * (p.x - moving_center.x) +
                    (p.y - moving_center.y) * (p.y - moving_center.y) +
                    (p.z - moving_center.z) * (p.z - moving_center.z));
                require(distance >= radius - 1.0e-3F,
                        "smoke penetrated moving sphere");
            }
        }
        require(final_center_x < center.x - 0.5F,
                "active smoke sphere did not move");

        SceneDefinition pushed = moving;
        pushed.rigid_bodies[0].options.initial_state.linear_velocity = {};
        SceneDefinition no_particles = pushed;
        no_particles.smoke_options.particles_per_second = 1.0F;
        World empty_world;
        SceneInstance empty_instance;
        require(create_scene_world(no_particles, empty_world, empty_instance),
                "create pre-emission rigid scene");
        require(empty_world.step({.timestep = 1.0F / 60.0F,
                .substeps = 1U, .gravity = {}}),
                "advance pre-emission rigid scene");
        RigidBodyState empty_state{};
        require(empty_world.read_rigid_body_state(
                    empty_instance.rigid_bodies[0], empty_state),
                "read pre-emission rigid state");
        require(std::abs(empty_state.position.x - center.x) < 1.0e-5F &&
                std::abs(empty_state.linear_velocity.x) < 1.0e-5F,
                "remote smoke wind pushed a rigid body without particles");
        SceneDefinition fewer_particles = pushed;
        fewer_particles.smoke_options.particles_per_second = 50.0F;
        World pushed_world, sparse_world, unforced_world;
        SceneInstance pushed_instance, sparse_instance, unforced_instance;
        require(create_scene_world(pushed, pushed_world, pushed_instance),
                "create wind-pushed sphere");
        require(create_scene_world(fewer_particles, sparse_world, sparse_instance),
                "create sparse-particle sphere");
        require(create_scene_world(pushed, unforced_world, unforced_instance),
                "create unforced sphere reference");
        require(unforced_world.remove_smoke_rigid_coupling(
                    unforced_instance.smoke_rigid_couplings.front()),
                "disable reference sphere drag");
        SmokeRigidCouplingId unforced_contact{};
        require(unforced_world.add_smoke_rigid_coupling(
                    {.smoke = unforced_instance.smoke,
                     .body = unforced_instance.rigid_bodies[0],
                     .air_density = 0.0F}, unforced_contact),
                "retain reference triangle contact");
        for (unsigned frame = 0U; frame < 90U; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                                   .substeps = 1U, .gravity = {}};
            require(pushed_world.step(step), "step wind-pushed sphere");
            require(sparse_world.step(step), "step sparse-particle sphere");
            require(unforced_world.step(step), "step unforced sphere");
        }
        RigidBodyState pushed_state{}, sparse_state{}, unforced_state{};
        require(pushed_world.read_rigid_body_state(
                    pushed_instance.rigid_bodies[0], pushed_state),
                "read wind-pushed sphere");
        require(unforced_world.read_rigid_body_state(
                    unforced_instance.rigid_bodies[0], unforced_state),
                "read unforced sphere");
        require(sparse_world.read_rigid_body_state(
                    sparse_instance.rigid_bodies[0], sparse_state),
                "read sparse-particle sphere");
        const float carrier_displacement =
            pushed_state.position.x - unforced_state.position.x;
        std::cout << "sphere_wind_displacement=" << carrier_displacement << '\n';
        require(carrier_displacement > 0.2F,
                "local smoke particles did not push the closed rigid sphere");
        require(pushed_state.position.x > sparse_state.position.x + 0.03F,
                "additional smoke particles did not increase rigid push");
        SceneDefinition distant_body = pushed;
        distant_body.rigid_bodies[0].options.initial_state.position.y += 5.0F;
        World distant_body_world;
        SceneInstance distant_body_instance;
        require(create_scene_world(distant_body, distant_body_world,
                                   distant_body_instance),
                "create rigid body beyond the smoke plume");
        for (unsigned frame = 0U; frame < 90U; ++frame)
            require(distant_body_world.step({.timestep = 1.0F / 60.0F,
                    .substeps = 1U, .gravity = {}}),
                    "step distant rigid body");
        RigidBodyState distant_state{};
        require(distant_body_world.read_rigid_body_state(
                    distant_body_instance.rigid_bodies[0], distant_state),
                "read distant rigid body");
        require(std::abs(distant_state.position.x - center.x) < 1.0e-4F &&
                std::abs(distant_state.linear_velocity.x) < 1.0e-4F,
                "smoke pushed a rigid body outside the particle plume");

        SceneDefinition buoyant = scene;
        buoyant.smoke_options.capacity = 32U;
        buoyant.smoke_options.particles_per_second = 32.0F;
        buoyant.smoke_options.emitter_center = {-2.0F, 0.0F, 0.0F};
        buoyant.smoke_options.emitter_half_extents = {0.001F, 0.001F};
        buoyant.smoke_options.initial_velocity = {};
        buoyant.smoke_options.wind = {};
        buoyant.smoke_options.buoyancy = 0.8F;
        World vertical_world, tilted_world;
        SceneInstance vertical, tilted;
        require(create_scene_world(buoyant, vertical_world, vertical),
                "create vertical buoyancy scene");
        require(create_scene_world(buoyant, tilted_world, tilted),
                "create tilted buoyancy scene");
        for (unsigned frame = 0U; frame < 90U; ++frame) {
            require(vertical_world.step({.timestep = 1.0F / 60.0F,
                    .substeps = 1U, .gravity = {0.0F, -9.81F, 0.0F}}),
                    "step vertical smoke buoyancy");
            require(tilted_world.step({.timestep = 1.0F / 60.0F,
                    .substeps = 1U, .gravity = {6.93672F, -6.93672F, 0.0F}}),
                    "step tilted smoke buoyancy");
        }
        SmokeDeviceView vertical_smoke{}, tilted_smoke{};
        require(vertical_world.smoke_view(vertical.smoke, vertical_smoke),
                "read vertical smoke buoyancy");
        require(tilted_world.smoke_view(tilted.smoke, tilted_smoke),
                "read tilted smoke buoyancy");
        const auto vertical_positions = read(vertical_smoke.positions);
        const auto tilted_positions = read(tilted_smoke.positions);
        if (!vertical_positions.empty() && !tilted_positions.empty())
            std::cout << "buoyancy_vertical_x=" << vertical_positions.front().x
                      << " tilted_x=" << tilted_positions.front().x << '\n';
        require(!vertical_positions.empty() && !tilted_positions.empty() &&
                tilted_positions.front().x < vertical_positions.front().x - 0.12F,
                "smoke buoyancy did not follow tilted gravity");
        std::cout << "smoke_particles=" << positions.size()
                  << " wake_particles=" << wake_particles
                  << " mean_transverse_wake_delta="
                  << wake_difference / wake_particles
                  << " maximum_speed=" << maximum_speed << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
