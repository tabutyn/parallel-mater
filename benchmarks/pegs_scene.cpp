// SPDX-License-Identifier: MIT
#include "support.hpp"
#include <parallel_mater_gallery/camera_controller.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::benchmark;
    using namespace parallel_mater::gallery;
    const bool verify_settling = argc == 2 &&
        std::string(argv[1]) == "--verify-settling";
    const bool verify_tilt = argc == 2 &&
        std::string(argv[1]) == "--verify-tilt";
    const bool verify_slosh = argc == 2 &&
        std::string(argv[1]) == "--verify-slosh";
    if (verify_settling || verify_tilt || verify_slosh) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
    }
    std::uint32_t frames = verify_settling ? 300U : verify_slosh ? 240U : 600U;
    if (argc > 14) return 2;
    if (argc >= 2 && !verify_settling && !verify_tilt && !verify_slosh) {
        char *end = nullptr;
        const unsigned long value = std::strtoul(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || value == 0U || value > 100'000U)
            return 2;
        frames = static_cast<std::uint32_t>(value);
    }
    SceneDefinition scene;
    std::string error;
    const char *scene_path = std::getenv("PM_PEGS_SCENE");
    if (scene_path == nullptr) scene_path = PARALLEL_MATER_PEGS_SCENE_PATH;
    if (!load_glb_scene(scene_path, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::size_t active = scene.rigid_bodies.size();
    for (std::size_t i = 0U; i < scene.rigid_bodies.size(); ++i)
        if (scene.rigid_bodies[i].options.motion == MotionType::dynamic)
            active = i;
    if (active == scene.rigid_bodies.size()) return 1;
    if (argc >= 3) {
        char *end = nullptr;
        const float mass = std::strtof(argv[2], &end);
        if (end == argv[2] || *end != '\0' || !std::isfinite(mass) || mass <= 0.0F)
            return 2;
        scene.rigid_bodies[active].options.mass = mass;
    }
    float gravity_scale = 1.0F;
    if (argc >= 4) {
        char *end = nullptr;
        gravity_scale = std::strtof(argv[3], &end);
        if (end == argv[3] || *end != '\0' || !std::isfinite(gravity_scale) ||
            gravity_scale <= 0.0F) return 2;
    }
    if (argc >= 5) {
        char *end = nullptr;
        const float repulsion = std::strtof(argv[4], &end);
        if (end == argv[4] || *end != '\0' || !std::isfinite(repulsion) ||
            repulsion < 0.0F) return 2;
        scene.fluid_options.repulsion = repulsion;
    }
    if (argc >= 6) {
        char *end = nullptr;
        const float initial_y = std::strtof(argv[5], &end);
        if (end == argv[5] || *end != '\0' || !std::isfinite(initial_y))
            return 2;
        scene.rigid_bodies[active].options.initial_state.position.y = initial_y;
    }
    std::uint32_t sample_interval = verify_slosh ? 10U : 100U;
    if (argc >= 7) {
        char *end = nullptr;
        const unsigned long value = std::strtoul(argv[6], &end, 10);
        if (end == argv[6] || *end != '\0' || value == 0U ||
            value > 100'000U) return 2;
        sample_interval = static_cast<std::uint32_t>(value);
    }
    if (argc >= 8) {
        char *end = nullptr;
        const float damping = std::strtof(argv[7], &end);
        if (end == argv[7] || *end != '\0' || !std::isfinite(damping) ||
            damping < 0.0F) return 2;
        scene.fluid_options.velocity_damping = damping;
    }
    if (argc >= 9) {
        char *end = nullptr;
        const float damping = std::strtof(argv[8], &end);
        if (end == argv[8] || *end != '\0' || !std::isfinite(damping) ||
            damping < 0.0F) return 2;
        scene.fluid_options.normal_damping = damping;
    }
    if (argc >= 10 && std::string(argv[9]) == "dry")
        scene.initial_particles.clear();
    if (argc >= 11) {
        char *end = nullptr;
        const float damping = std::strtof(argv[10], &end);
        if (end == argv[10] || *end != '\0' || !std::isfinite(damping) ||
            damping < 0.0F) return 2;
        scene.rigid_bodies[active].options.angular_damping = damping;
    }
    if (argc >= 12) {
        char *end = nullptr;
        const float damping = std::strtof(argv[11], &end);
        if (end == argv[11] || *end != '\0' || !std::isfinite(damping) ||
            damping < 0.0F) return 2;
        scene.rigid_bodies[active].options.linear_damping = damping;
    }
    if (argc >= 13) {
        char *end = nullptr;
        const float margin = std::strtof(argv[12], &end);
        if (end == argv[12] || *end != '\0' || !std::isfinite(margin) ||
            margin < 0.0F) return 2;
        for (RigidBodyDefinition &body : scene.rigid_bodies)
            if (body.name.starts_with("Sphere"))
                body.options.collision_margin = margin;
    }
    float gravity_tilt_degrees = verify_tilt
        ? peg_paint_gravity_tilt_degrees : 0.0F;
    if (argc >= 14) {
        char *end = nullptr;
        gravity_tilt_degrees = std::strtof(argv[13], &end);
        if (end == argv[13] || *end != '\0' ||
            !std::isfinite(gravity_tilt_degrees) ||
            gravity_tilt_degrees < 0.0F || gravity_tilt_degrees >= 90.0F)
            return 2;
    }
    std::uint32_t initial_cap_particles = 0U;
    for (const FluidParticle &particle : scene.initial_particles) {
        const Vec3 position = particle.position;
        const float dx = std::fabs(position.x) - 0.264F;
        const float dz = std::fabs(position.z) - 0.264F;
        initial_cap_particles += dx * dx + dz * dz < 0.04F * 0.04F &&
            position.y > -0.355F && position.y < -0.25F;
    }
    World world;
    SceneInstance instance;
    if (!prepare_world(scene, static_cast<std::uint32_t>(
            scene.meshes.size() + scene.collision_meshes.size()),
            world, instance)) return 1;
    std::cout << "initial_particles=" << scene.initial_particles.size()
              << " initial_cap_particles=" << initial_cap_particles
              << " active_mass=" << scene.rigid_bodies[active].options.mass
              << " initial_y="
              << scene.rigid_bodies[active].options.initial_state.position.y
              << '\n';
    for (const RigidBodyDefinition &body : scene.rigid_bodies)
        std::cout << "body=" << body.name << " margin="
                  << body.options.collision_margin << " friction="
                  << body.options.friction << '\n';
    Samples wall;
    bool settled = false;
    bool reversed_with_momentum = false;
    bool rebounded = false;
    std::cout << std::fixed << std::setprecision(3);
    for (std::uint32_t frame = 1U; frame <= frames; ++frame) {
        StepOptions options = standard_step_options(false);
        options.gravity.y *= scene.gravity_scale * gravity_scale;
        if (gravity_tilt_degrees > 0.0F) {
            const float radians = gravity_tilt_degrees * 0.017453292519943295F;
            const float magnitude = -options.gravity.y;
            options.gravity = {magnitude * std::sin(radians),
                               -magnitude * std::cos(radians), 0.0F};
        }
        if (verify_slosh) {
            const float radians = peg_paint_gravity_tilt_degrees *
                0.017453292519943295F;
            const float magnitude = -options.gravity.y;
            const float direction = frame <= 120U ? 1.0F : -1.0F;
            options.gravity = {direction * magnitude * std::sin(radians),
                               -magnitude * std::cos(radians), 0.0F};
        }
        options.collect_fluid_contacts = frame % sample_interval == 0U ||
            frame == frames;
        options.collect_rigid_contacts = options.collect_fluid_contacts;
        options.collect_kernel_timings = options.collect_fluid_contacts;
        const auto begin = std::chrono::steady_clock::now();
        if (!require(world.step(options), "step Pegs scene")) return 1;
        wall.add(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count());
        if (frame % sample_interval != 0U && frame != frames) continue;
        RigidBodyState state{};
        WorldStatistics statistics{};
        WorldStepTimings timings{};
        if (!require(world.read_rigid_body_state(instance.rigid_bodies[active],
                                                  state), "read active body") ||
            !require(world.collect_statistics(statistics), "read Pegs stats") ||
            !require(world.collect_step_timings(timings), "read Pegs timings"))
            return 1;
        FluidDeviceView fluid{};
        if (!require(world.fluid_view(instance.fluid, fluid), "view Pegs fluid"))
            return 1;
        std::vector<Vec3> positions(fluid.particle_count);
        std::vector<Vec3> velocities(fluid.particle_count);
        std::vector<float> foam(fluid.particle_count);
        if (cudaMemcpy(positions.data(), fluid.positions.data,
                       positions.size() * sizeof(Vec3),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
        if (cudaMemcpy(velocities.data(), fluid.velocities.data,
                       velocities.size() * sizeof(Vec3),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
        if (cudaMemcpy(foam.data(), fluid.foam.data,
                       foam.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
        float lowest = 1.0e10F;
        std::uint32_t outside = 0U;
        std::uint32_t cap_particles = 0U;
        std::uint32_t edge_particles = 0U;
        std::uint32_t high_outer_particles = 0U;
        float outer_top = -1.0e10F;
        float cap_speed = 0.0F;
        float cap_speed_sum = 0.0F;
        float cap_foam_sum = 0.0F;
        float cap_vertical_sum = 0.0F;
        float cap_horizontal_sum = 0.0F;
        float mean_vx = 0.0F;
        float mean_x = 0.0F;
        for (std::size_t i = 0U; i < positions.size(); ++i) {
            const Vec3 position = positions[i];
            mean_x += position.x;
            mean_vx += velocities[i].x;
            lowest = std::min(lowest, position.y);
            outside += position.y < -1.02F;
            const bool outer = position.x * position.x +
                position.z * position.z > 0.8F * 0.8F;
            edge_particles += outer;
            high_outer_particles += outer && position.y > -0.45F;
            if (outer) outer_top = std::max(outer_top, position.y);
            const float dx = std::fabs(position.x) - 0.264F;
            const float dz = std::fabs(position.z) - 0.264F;
            if (dx * dx + dz * dz < 0.04F * 0.04F &&
                position.y > -0.355F && position.y < -0.25F) {
                ++cap_particles;
                const Vec3 velocity = velocities[i];
                const float speed = std::sqrt(
                    velocity.x * velocity.x + velocity.y * velocity.y +
                    velocity.z * velocity.z);
                cap_speed = std::max(cap_speed, speed);
                cap_speed_sum += speed;
                cap_foam_sum += foam[i];
                cap_vertical_sum += std::fabs(velocity.y);
                cap_horizontal_sum += std::sqrt(
                    velocity.x * velocity.x + velocity.z * velocity.z);
            }
        }
        std::cout << frame << " particles=" << statistics.particle_count
                  << " rigid_contacts=" << world.rigid_contacts().event_count
                  << " x=" << state.position.x
                  << " y=" << state.position.y
                  << " z=" << state.position.z
                  << " vy=" << state.linear_velocity.y
                  << " omega=" << std::sqrt(
                      state.angular_velocity.x * state.angular_velocity.x +
                      state.angular_velocity.y * state.angular_velocity.y +
                      state.angular_velocity.z * state.angular_velocity.z)
                  << " min_particle_y=" << lowest
                  << " escaped=" << outside
                  << " contacts=" << statistics.contact_count
                  << " cap_particles=" << cap_particles
                  << " edge_particles=" << edge_particles
                  << " high_outer_particles=" << high_outer_particles
                  << " outer_top=" << (edge_particles ? outer_top : 0.0F)
                  << " mean_x=" << mean_x /
                     std::max<std::size_t>(positions.size(), 1U)
                  << " mean_vx=" << mean_vx /
                     std::max<std::size_t>(positions.size(), 1U)
                  << " cap_speed=" << cap_speed
                  << " cap_mean_speed=" << cap_speed_sum /
                     std::max(1U, cap_particles)
                  << " cap_mean_foam=" << cap_foam_sum /
                     std::max(1U, cap_particles)
                  << " cap_vertical=" << cap_vertical_sum /
                     std::max(1U, cap_particles)
                  << " cap_horizontal=" << cap_horizontal_sum /
                     std::max(1U, cap_particles)
                  << " overflow=" << statistics.contact_overflow_count
                  << " gpu_ms=" << timings.total_gpu_milliseconds
                  << " sort_ms=" << timings.fluid_neighbor_sort.total_milliseconds
                  << " neighbors_ms=" <<
                     timings.fluid_neighbor_forces.total_milliseconds
                  << " static_ms=" <<
                     timings.fluid_static_contacts.total_milliseconds
                  << " moving_ms=" <<
                     timings.fluid_moving_contacts.total_milliseconds
                  << " rigid_ms=" <<
                     timings.rigid_contact_generation.total_milliseconds +
                     timings.rigid_contact_solve.total_milliseconds
                  << " wall_ms=" << wall.mean() << '\n';
        if (frame == frames && verify_settling) {
            settled = std::fabs(state.position.y + 0.707F) < 0.025F &&
                std::fabs(state.linear_velocity.y) < 0.10F &&
                cap_particles <= 2U &&
                high_outer_particles <= 100U &&
                (cap_particles == 0U ||
                 cap_speed_sum / cap_particles < 0.12F) &&
                 outside == 0U && statistics.contact_overflow_count == 0U;
        }
        if (frame == frames && verify_tilt) {
            settled = outer_top < 0.75F &&
                high_outer_particles < 3000U && outside == 0U &&
                statistics.contact_overflow_count == 0U;
        }
        if (verify_slosh) {
            const float average_vx = mean_vx /
                std::max<std::size_t>(positions.size(), 1U);
            if (frame == 130U) reversed_with_momentum = average_vx < -1.7F;
            if (frame == 190U) rebounded = average_vx > 0.05F;
            if (frame == frames)
                settled = reversed_with_momentum && rebounded &&
                    statistics.particle_count == scene.initial_particles.size() &&
                    outside == 0U && statistics.contact_overflow_count == 0U;
        }
        wall.clear();
    }
    if ((verify_settling || verify_tilt || verify_slosh) && !settled) {
        std::cerr << (verify_slosh ? "Peg water lost slosh momentum\n"
                    : verify_tilt ? "Peg tilt stacked fluid excessively\n"
                                  : "Peg sphere or post-cap fluid did not settle\n");
        return 1;
    }
    return 0;
}
