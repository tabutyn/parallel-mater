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

static void check(bool good, const char *message) {
    if (!good) throw std::runtime_error(message);
}
static void check(Status status, const char *message) {
    if (!status) throw std::runtime_error(std::string(message) + ": " + status.message);
}
template <class T> static std::vector<T> read(DeviceSpan<const T> span) {
    std::vector<T> result(span.size);
    if (!result.empty())
        check(cudaMemcpy(result.data(), span.data, result.size() * sizeof(T),
                         cudaMemcpyDeviceToHost) == cudaSuccess, "device read failed");
    return result;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        SceneDefinition authored;
        std::string error;
        check(load_glb_scene(PARALLEL_MATER_SMOKE_WATER_SCENE_PATH,
                             authored, error), error.c_str());
        check(authored.has_smoke && !authored.initial_particles.empty() &&
              authored.rigid_bodies.size() == 4U &&
              authored.thermal_surfaces.size() == 1U,
              "SmokeWater systems were not exported");
        bool active_smoke_sphere = false;
        for (const auto &body : authored.rigid_bodies)
            if ((body.name == authored.smoke_obstacle_name ||
                 body.source_name == authored.smoke_obstacle_name) &&
                body.options.motion == MotionType::dynamic)
                active_smoke_sphere = true;
        check(active_smoke_sphere,
              "SmokeWater active sphere was not exported");
        check(std::abs(authored.initial_particles.front().temperature - 80.0F) < 0.01F &&
              std::abs(authored.thermal_surfaces.front().temperature - 500.0F) < 0.01F,
              "authored temperatures were not preserved");

        // Isolated hot-plate contact: particle must leave water and enter smoke.
        SceneDefinition boiling = authored;
        boiling.initial_particles = {
            {{0.0F, 0.04F, 0.0F}, {}, 80.0F},
            {{-2.0F, 0.04F, 0.0F}, {}, 80.0F}};
        boiling.fluid_options.capacity = 32U;
        boiling.thermal_surfaces.front().heat_transfer_rate = 5.0F;
        World world;
        SceneInstance instance;
        check(create_scene_world(boiling, world, instance), "create boiling world");
        check(instance.has_fluid && instance.has_smoke &&
              instance.fluid_smoke_couplings.size() == 1U,
              "coupling was not registered in API");
        check(!world.remove_fluid(instance.fluid), "referenced fluid removed");
        check(!world.remove_smoke(instance.smoke), "referenced smoke removed");
        for (int frame = 0; frame < 20; ++frame)
            check(world.step({.timestep = 1.0F / 60.0F, .substeps = 1U,
                              .gravity = {}}), "step boiling world");
        FluidDeviceView water{};
        SmokeDeviceView smoke{};
        WorldStatistics statistics{};
        check(world.fluid_view(instance.fluid, water), "read water");
        check(world.smoke_view(instance.smoke, smoke), "read smoke");
        check(world.collect_statistics(statistics), "read statistics");
        check(water.particle_count == 1U && statistics.boiled_particle_count == 1U &&
              smoke.particle_count > 0U, "boiling did not transfer a water particle");
        const auto smoke_positions = read(smoke.positions);
        const auto smoke_velocities = read(smoke.velocities);
        bool rising_steam = false;
        for (std::size_t i = 0; i < smoke_positions.size(); ++i)
            if (smoke_positions[i].x > -0.5F &&
                smoke_positions[i].y > 0.2F && smoke_velocities[i].y > 0.05F)
                rising_steam = true;
        check(rising_steam, "transferred steam did not rise");
        const auto remaining_temperature = read(water.temperatures);
        const auto remaining_ids = read(water.stable_particle_ids);
        check(remaining_ids[0] == 1U &&
              std::abs(remaining_temperature[0] - 80.0F) < 0.01F,
              "water temperature or identity was lost during compaction");
        check(world.step({.timestep = 1.0F / 60.0F, .substeps = 1U,
                          .gravity = {}, .collect_kernel_timings = true}),
              "profile fluid smoke exchange");
        WorldStepTimings timings{};
        check(world.collect_step_timings(timings), "read fluid smoke timing");
        check(timings.available && timings.fluid_smoke_exchange.launch_count == 1U,
              "fluid smoke timing was not exposed");
        check(world.remove_fluid_smoke_coupling(instance.fluid_smoke_couplings[0]),
              "remove fluid smoke coupling");
        check(!world.remove_fluid_smoke_coupling(instance.fluid_smoke_couplings[0]),
              "stale coupling handle was accepted");
        check(world.remove_fluid(instance.fluid), "remove former coupled fluid");
        check(world.remove_smoke(instance.smoke), "remove former coupled smoke");

        // The carrier-gas field, not just the tracer visualization, pushes water.
        SceneDefinition blown = authored;
        blown.initial_particles = {{{-0.8F, 1.5F, 0.0F}, {}, 80.0F}};
        blown.fluid_options.capacity = 32U;
        blown.thermal_surfaces.front().heat_transfer_rate = 0.0F;
        SceneDefinition still = blown;
        still.thermal_surfaces.clear();
        World wind_world, still_world;
        SceneInstance wind, reference;
        check(create_scene_world(blown, wind_world, wind), "create wind world");
        check(create_scene_world(still, still_world, reference), "create reference world");
        for (int frame = 0; frame < 12; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                                   .substeps = 1U, .gravity = {}};
            check(wind_world.step(step), "step wind world");
            check(still_world.step(step), "step reference world");
        }
        FluidDeviceView windy{}, calm{};
        check(wind_world.fluid_view(wind.fluid, windy), "read windy water");
        check(still_world.fluid_view(reference.fluid, calm), "read calm water");
        const auto windy_velocity = read(windy.velocities);
        const auto calm_velocity = read(calm.velocities);
        check(windy_velocity.size() == 1U && calm_velocity.size() == 1U &&
              windy_velocity[0].x > calm_velocity[0].x + 0.1F,
              "smoke flow did not push water");

        SceneDefinition interaction = authored;
        interaction.fluid_options.capacity = 30'000U;
        World interaction_world;
        SceneInstance interaction_instance;
        check(create_scene_world(interaction, interaction_world,
                                 interaction_instance),
              "create active sphere water interaction");
        std::size_t obstacle_index = 0U;
        for (; obstacle_index < interaction.rigid_bodies.size(); ++obstacle_index)
            if (interaction.rigid_bodies[obstacle_index].source_name ==
                interaction.smoke_obstacle_name) break;
        check(obstacle_index < interaction.rigid_bodies.size(),
              "active sphere body index missing");
        Vec3 previous_velocity{};
        float maximum_speed = 0.0F;
        float maximum_velocity_jump = 0.0F;
        float maximum_angular_speed = 0.0F;
        const auto length = [](Vec3 value) {
            return std::sqrt(value.x * value.x + value.y * value.y +
                             value.z * value.z);
        };
        for (unsigned frame = 0U; frame < 180U; ++frame) {
            check(interaction_world.step({.timestep = 1.0F / 60.0F,
                  .substeps = 1U, .gravity = {0.0F, -9.81F, 0.0F}}),
                  "step active sphere water interaction");
            RigidBodyState state{};
            check(interaction_world.read_rigid_body_state(
                  interaction_instance.rigid_bodies[obstacle_index], state),
                  "read active sphere state");
            check(std::isfinite(length(state.linear_velocity)) &&
                  std::isfinite(length(state.angular_velocity)),
                  "active sphere velocity became non-finite");
            maximum_speed = std::max(maximum_speed,
                                     length(state.linear_velocity));
            maximum_velocity_jump = std::max(maximum_velocity_jump,
                length({state.linear_velocity.x - previous_velocity.x,
                        state.linear_velocity.y - previous_velocity.y,
                        state.linear_velocity.z - previous_velocity.z}));
            maximum_angular_speed = std::max(maximum_angular_speed,
                                             length(state.angular_velocity));
            previous_velocity = state.linear_velocity;
        }
        check(maximum_speed < 10.0F && maximum_velocity_jump < 10.0F &&
              maximum_angular_speed < 10.0F,
              "water impact destabilized the active sphere");
        check(maximum_angular_speed > 0.1F,
              "water no longer transfers angular momentum to the sphere");
        std::cout << "active_sphere_speed=" << maximum_speed
                  << " max_frame_delta_v=" << maximum_velocity_jump
                  << " max_angular_speed=" << maximum_angular_speed << '\n';
        std::cout << "boiled=" << statistics.boiled_particle_count
                  << " wind_delta=" << windy_velocity[0].x - calm_velocity[0].x
                  << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
