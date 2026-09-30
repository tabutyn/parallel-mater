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
        World world;
        SceneInstance instance;
        require(create_scene_world(scene, world, instance), "create smoke scene");
        require(instance.has_smoke && !instance.has_fluid,
                "smoke API resource was not instantiated");
        require(!world.remove_rigid_body(instance.rigid_bodies[0]),
                "smoke obstacle was removed while referenced");
        SmokeId invalid{};
        SmokeOptions bad = scene.smoke_options;
        bad.obstacle = instance.rigid_bodies[0];
        bad.capacity = 0U;
        require(!world.add_smoke(bad, invalid), "zero-capacity smoke accepted");

        SceneDefinition still_air = scene;
        still_air.smoke_options.wake_strength = 0.0F;
        World reference_world;
        SceneInstance reference;
        require(create_scene_world(still_air, reference_world, reference),
                "create zero-wake comparison");
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
        for (std::size_t index = 0; index < positions.size(); ++index) {
            const auto &p = positions[index];
            const auto &v = velocities[index];
            require(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
                    std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z),
                    "non-finite smoke state");
            if (ages[index] >= smoke.lifetime) continue;
            const float distance = std::sqrt((p.x-center.x)*(p.x-center.x) +
                (p.y-center.y)*(p.y-center.y) +
                (p.z-center.z)*(p.z-center.z));
            require(distance >= radius - 1.0e-3F,
                    "smoke particle penetrated spherical obstacle");
            const float speed = std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
            maximum_speed = std::max(maximum_speed, speed);
            if (p.x < center.x + radius) continue;
            ++wake_particles;
            wake_difference += std::abs(v.y - comparison[index].y) +
                               std::abs(v.z - comparison[index].z);
        }
        require(wake_particles > 100U, "smoke did not travel behind the sphere");
        std::cout << "wake_particles=" << wake_particles
                  << " mean_transverse_wake_delta="
                  << wake_difference / wake_particles << '\n';
        require(wake_difference / wake_particles > 0.05F,
                "wake did not create measurable transverse vortices");
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
        swept.smoke_options.wake_strength = 0.0F;
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
