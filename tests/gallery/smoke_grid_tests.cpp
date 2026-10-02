// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
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
            "grid readback failed");
    return result;
}

int main(int argc, char **argv) {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        const std::uint32_t resolution = argc > 1 ?
            static_cast<std::uint32_t>(std::stoul(argv[1])) : 128U;
        constexpr std::uint32_t height = 32U;
        const unsigned frames = argc > 2 ?
            static_cast<unsigned>(std::stoul(argv[2])) : 120U;
        SceneDefinition scene;
        std::string error;
        require(load_glb_scene(PARALLEL_MATER_SMOKE_SCENE_PATH, scene, error),
            error.c_str());
        scene.smoke_options.grid_resolution = resolution;
        scene.smoke_options.grid_vertical_resolution = height;
        scene.smoke_options.grid_pressure_iterations = 24U;
        World world;
        SceneInstance instance;
        require(create_scene_world(scene, world, instance), "create grid smoke");
        const auto started = std::chrono::steady_clock::now();
        for (unsigned frame = 0; frame < frames; ++frame)
            require(world.step({.timestep = 1.0F / 60.0F,
                .substeps = 1U, .gravity = {}}), "step grid smoke");
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        SmokeDeviceView view{};
        require(world.smoke_view(instance.smoke, view), "read grid smoke");
        require(view.grid_resolution == resolution &&
            view.grid_vertical_resolution == height &&
            view.grid_velocity.size == std::size_t(resolution) * height * resolution &&
            view.grid_temperature.size == view.grid_velocity.size &&
            view.grid_vorticity.size == view.grid_velocity.size &&
            view.grid_divergence.size == view.grid_velocity.size,
            "grid view has wrong dimensions");
        const auto velocity = read(view.grid_velocity);
        const auto density = read(view.grid_density);
        const auto temperature = read(view.grid_temperature);
        const auto pressure = read(view.grid_pressure);
        const auto solid = read(view.grid_solid);
        const auto vorticity = read(view.grid_vorticity);
        const auto divergence = read(view.grid_divergence);
        std::size_t solid_cells{}, dense_cells{}, altered_air{};
        float maximum_pressure{}, maximum_divergence{}, maximum_vorticity{};
        const Vec3 center = scene.rigid_bodies[0].options.initial_state.position;
        float radius{};
        for (const auto &vertex : scene.meshes[
            scene.rigid_bodies[0].mesh_indices[0]].vertices)
            radius = std::max(radius, std::sqrt(
                vertex.position.x*vertex.position.x +
                vertex.position.y*vertex.position.y +
                vertex.position.z*vertex.position.z));
        float front_lateral{}, side_lateral{}, lee_lateral{};
        float front_pressure{}, rear_pressure{}, side_speed{}, lee_enstrophy{};
        std::size_t recirculating_cells{};
        std::size_t front_cells{}, side_cells{}, lee_cells{};
        for (std::size_t cell = 0; cell < velocity.size(); ++cell) {
            require(std::isfinite(velocity[cell].x) &&
                std::isfinite(velocity[cell].y) &&
                std::isfinite(velocity[cell].z) &&
                std::isfinite(density[cell]) &&
                std::isfinite(pressure[cell]) &&
                std::isfinite(vorticity[cell].x) &&
                std::isfinite(vorticity[cell].y) &&
                std::isfinite(vorticity[cell].z) &&
                std::isfinite(divergence[cell]), "nonfinite air grid");
            solid_cells += solid[cell] != 0U;
            dense_cells += density[cell] > 0.05F;
            altered_air += !solid[cell] &&
                (std::abs(velocity[cell].x - scene.smoke_options.wind.x) +
                 std::abs(velocity[cell].y - scene.smoke_options.wind.y) +
                 std::abs(velocity[cell].z - scene.smoke_options.wind.z) > 0.05F);
            maximum_pressure = std::max(maximum_pressure,
                std::abs(pressure[cell]));
            maximum_divergence = std::max(maximum_divergence,
                std::abs(divergence[cell]));
            const float curl2 = vorticity[cell].x*vorticity[cell].x +
                vorticity[cell].y*vorticity[cell].y +
                vorticity[cell].z*vorticity[cell].z;
            maximum_vorticity = std::max(maximum_vorticity, std::sqrt(curl2));
            if (solid[cell]) continue;
            const float x = view.grid_minimum.x +
                (float(cell % resolution) + 0.5F) * view.grid_spacing - center.x;
            const float y = view.grid_minimum.y +
                (float(cell / resolution % height) + 0.5F) *
                view.grid_spacing - center.y;
            const float z = view.grid_minimum.z +
                (float(cell / (resolution * height)) + 0.5F) *
                view.grid_spacing - center.z;
            const float lateral = std::sqrt(velocity[cell].y*velocity[cell].y +
                                            velocity[cell].z*velocity[cell].z);
            if (x < -radius - 0.1F && x > -radius - 0.5F &&
                std::abs(y) < radius && std::abs(z) < radius) {
                front_lateral += lateral;
                front_pressure += pressure[cell];
                ++front_cells;
            }
            if (std::abs(x) < radius && std::abs(y) > radius &&
                std::abs(y) < radius + 0.3F && std::abs(z) < radius) {
                side_lateral += lateral;
                side_speed += std::sqrt(velocity[cell].x*velocity[cell].x +
                    velocity[cell].y*velocity[cell].y +
                    velocity[cell].z*velocity[cell].z);
                ++side_cells;
            }
            if (x > radius + 0.1F && x < radius + 1.0F &&
                std::abs(y) < radius && std::abs(z) < radius) {
                lee_lateral += lateral;
                rear_pressure += pressure[cell];
                lee_enstrophy += curl2;
                recirculating_cells += velocity[cell].x < -0.01F;
                ++lee_cells;
            }
        }
        const float normalized_divergence = maximum_divergence *
            view.grid_spacing / std::max(0.1F, std::abs(scene.smoke_options.wind.x));
        const float average_front_pressure = front_pressure /
            std::max<std::size_t>(1U, front_cells);
        const float average_rear_pressure = rear_pressure /
            std::max<std::size_t>(1U, lee_cells);
        const float average_side_speed = side_speed /
            std::max<std::size_t>(1U, side_cells);
        const float average_lee_enstrophy = lee_enstrophy /
            std::max<std::size_t>(1U, lee_cells);
        std::cout << "grid_resolution=" << resolution << 'x' << height
                  << 'x' << resolution
                  << " cells=" << velocity.size()
                  << " ms_per_frame=" << milliseconds / frames
                  << " solid_cells=" << solid_cells
                  << " dense_cells=" << dense_cells
                  << " altered_air=" << altered_air
                  << " max_abs_pressure=" << maximum_pressure
                  << " pressure_relative_residual="
                  << view.grid_pressure_relative_residual
                  << " normalized_divergence=" << normalized_divergence
                  << " max_vorticity=" << maximum_vorticity << '\n';
        std::cout << "front_lateral=" << front_lateral /
            std::max<std::size_t>(1U, front_cells)
            << " side_lateral=" << side_lateral /
            std::max<std::size_t>(1U, side_cells)
            << " lee_lateral=" << lee_lateral /
            std::max<std::size_t>(1U, lee_cells)
            << " front_pressure=" << average_front_pressure
            << " rear_pressure=" << average_rear_pressure
            << " side_speed=" << average_side_speed
            << " lee_enstrophy=" << average_lee_enstrophy
            << " recirculating_cells=" << recirculating_cells << '\n';
        require(solid_cells > 0U && dense_cells > 0U && altered_air > 0U,
            "grid failed to carry smoke around the triangle mesh");
        require(view.grid_pressure_relative_residual <=
                scene.smoke_options.grid_pressure_tolerance * 1.05F,
            "multigrid projection missed its relative residual tolerance");
        require(normalized_divergence < 0.02F,
            "projected MAC velocity retained excessive divergence");
        require(maximum_vorticity > 0.01F && average_front_pressure >
                average_rear_pressure && average_side_speed > 0.5F &&
                recirculating_cells > 0U,
            "triangle boundary did not produce pressure, side flow, and recirculation");

        if (resolution > 128U || frames < 60U) return 0;
        if (frames >= 120U) {
            World replay_world;
            SceneInstance replay_instance;
            require(create_scene_world(scene,replay_world,replay_instance),
                "create deterministic replay");
            for(unsigned frame=0;frame<frames;++frame)
                require(replay_world.step({.timestep=1.0F/60.0F,
                    .substeps=1U,.gravity={}}),"step deterministic replay");
            SmokeDeviceView replay_view{};
            require(replay_world.smoke_view(replay_instance.smoke,replay_view),
                "read deterministic replay");
            const auto bit_equal=[](const auto &a,const auto &b){
                return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),
                    b.data(),a.size()*sizeof(a.front()))==0);
            };
            require(bit_equal(velocity,read(replay_view.grid_velocity))&&
                    bit_equal(density,read(replay_view.grid_density))&&
                    bit_equal(temperature,read(replay_view.grid_temperature))&&
                    bit_equal(pressure,read(replay_view.grid_pressure))&&
                    bit_equal(vorticity,read(replay_view.grid_vorticity))&&
                    bit_equal(divergence,read(replay_view.grid_divergence))&&
                    bit_equal(read(view.positions),read(replay_view.positions)),
                "hybrid smoke replay was not bit deterministic");
        }
        SceneDefinition open_scene = scene;
        World open_world;
        SceneInstance open_instance;
        require(create_scene_world(open_scene, open_world, open_instance),
            "create unobstructed grid reference");
        require(open_world.remove_smoke_rigid_coupling(
            open_instance.smoke_rigid_couplings.front()),
            "remove reference triangle boundary");
        for (unsigned frame = 0; frame < frames; ++frame)
            require(open_world.step({.timestep = 1.0F / 60.0F,
                .substeps = 1U, .gravity = {}}), "step unobstructed grid");
        SmokeDeviceView open_view{};
        require(open_world.smoke_view(open_instance.smoke, open_view),
            "read unobstructed grid");
        const auto open_vorticity = read(open_view.grid_vorticity);
        float open_lee_enstrophy{};
        std::size_t open_lee_cells{};
        for (std::size_t cell = 0; cell < open_vorticity.size(); ++cell) {
            const float x = open_view.grid_minimum.x +
                (float(cell % resolution) + 0.5F) * open_view.grid_spacing - center.x;
            const float y = open_view.grid_minimum.y +
                (float(cell / resolution % height) + 0.5F) *
                open_view.grid_spacing - center.y;
            const float z = open_view.grid_minimum.z +
                (float(cell / (resolution * height)) + 0.5F) *
                open_view.grid_spacing - center.z;
            if (x <= radius + 0.1F || x >= radius + 1.0F ||
                std::abs(y) >= radius || std::abs(z) >= radius) continue;
            const auto curl = open_vorticity[cell];
            open_lee_enstrophy += curl.x*curl.x + curl.y*curl.y + curl.z*curl.z;
            ++open_lee_cells;
        }
        open_lee_enstrophy /= std::max<std::size_t>(1U, open_lee_cells);
        std::cout << "obstructed_lee_enstrophy=" << average_lee_enstrophy
                  << " open_lee_enstrophy=" << open_lee_enstrophy << '\n';
        require(average_lee_enstrophy > open_lee_enstrophy + 1.0e-4F,
            "triangle boundary did not increase lee enstrophy");

        const auto run_dynamic = [&](float rate, float air_density,
                                     unsigned run_frames) {
            SceneDefinition dynamic_scene = scene;
            dynamic_scene.rigid_bodies[0].options.motion =
                MotionType::dynamic;
            dynamic_scene.rigid_bodies[0].options.mass = 1.0F;
            dynamic_scene.smoke_options.particles_per_second = rate;
            World dynamic_world;
            SceneInstance dynamic_instance;
            require(create_scene_world(dynamic_scene, dynamic_world,
                dynamic_instance), "create dynamic grid smoke");
            if (air_density == 0.0F) {
                require(dynamic_world.remove_smoke_rigid_coupling(
                    dynamic_instance.smoke_rigid_couplings.front()),
                    "remove dynamic grid reaction");
                SmokeRigidCouplingId contact{};
                require(dynamic_world.add_smoke_rigid_coupling({
                    .smoke = dynamic_instance.smoke,
                    .body = dynamic_instance.rigid_bodies[0],
                    .air_density = 0.0F}, contact),
                    "restore containment-only grid coupling");
            }
            for (unsigned frame = 0; frame < run_frames; ++frame)
                require(dynamic_world.step({.timestep = 1.0F / 60.0F,
                    .substeps = 1U, .gravity = {}}),
                    "step dynamic grid smoke");
            RigidBodyState state{};
            require(dynamic_world.read_rigid_body_state(
                dynamic_instance.rigid_bodies[0], state),
                "read dynamic grid body");
            return state;
        };
        const auto prearrival = run_dynamic(900.0F, 1.5F, 30U);
        const auto prearrival_reference = run_dynamic(900.0F, 0.0F, 30U);
        require(std::abs(prearrival.position.x-prearrival_reference.position.x) <
                1.0e-3F,
            "grid smoke forced the body before the plume arrived");
        const auto dense_force = run_dynamic(900.0F, 1.5F, 120U);
        const auto sparse_force = run_dynamic(150.0F, 1.5F, 120U);
        const auto force_reference = run_dynamic(900.0F, 0.0F, 120U);
        const float dense_displacement = dense_force.position.x-
            force_reference.position.x;
        const float sparse_displacement = sparse_force.position.x-
            force_reference.position.x;
        std::cout << "grid_dense_force_displacement=" << dense_displacement
                  << " grid_sparse_force_displacement=" << sparse_displacement
                  << " grid_dense_vertical_force=" << dense_force.position.y-
                     force_reference.position.y << '\n';
        require(dense_displacement > 0.005F &&
                dense_displacement > sparse_displacement + 0.001F,
            "grid surface force was not local or density-scaled");

        SceneDefinition soft_scene;
        require(load_glb_scene(PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH,
            soft_scene, error), error.c_str());
        soft_scene.soft_bodies.resize(1U);
        soft_scene.smoke_options.emitter_center = {-0.35F, 0.7F, 0.0F};
        soft_scene.smoke_options.emitter_half_extents = {0.1F, 0.05F};
        soft_scene.smoke_options.grid_resolution = resolution;
        soft_scene.smoke_options.grid_vertical_resolution = height;
        soft_scene.smoke_options.grid_pressure_iterations = 24U;
        World soft_world, reference_world;
        SceneInstance soft_instance, reference_instance;
        require(create_scene_world(soft_scene, soft_world, soft_instance),
            "create grid soft body");
        require(create_scene_world(soft_scene, reference_world,
            reference_instance), "create grid reference");
        require(reference_world.remove_smoke_soft_body_coupling(
            reference_instance.smoke_soft_body_couplings.front()),
            "disable reference wind");
        const auto soft_started = std::chrono::steady_clock::now();
        for (unsigned frame = 0; frame < frames; ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F,
                .substeps = 4U, .gravity = {}};
            require(soft_world.step(step), "step grid soft body");
            require(reference_world.step(step), "step grid reference");
        }
        SmokeDeviceView soft_air{};
        require(soft_world.smoke_view(soft_instance.smoke, soft_air),
            "read soft-body air field");
        const auto soft_density = read(soft_air.grid_density);
        std::size_t dense_near_body{};
        float maximum_near_density{};
        for (std::size_t cell = 0; cell < soft_density.size(); ++cell) {
            const float x = soft_air.grid_minimum.x +
                (float(cell % resolution) + 0.5F) * soft_air.grid_spacing;
            const float y = soft_air.grid_minimum.y +
                (float(cell / resolution % height) + 0.5F) * soft_air.grid_spacing;
            const float z = soft_air.grid_minimum.z +
                (float(cell / (resolution * height)) + 0.5F) *
                    soft_air.grid_spacing;
            if (std::abs(x) < 0.5F && y > 0.0F && y < 1.5F &&
                std::abs(z) < 0.5F) {
                dense_near_body += soft_density[cell] > 0.01F;
                maximum_near_density = std::max(maximum_near_density,
                    soft_density[cell]);
            }
        }
        const double soft_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - soft_started).count();
        SoftBodyDeviceView soft{}, reference{};
        require(soft_world.soft_body_view(soft_instance.soft_bodies[0], soft),
            "read grid soft body");
        require(reference_world.soft_body_view(
            reference_instance.soft_bodies[0], reference),
            "read grid soft-body reference");
        const auto positions = read(soft.positions);
        const auto reference_positions = read(reference.positions);
        float displacement{}, maximum_strain{};
        for (std::size_t node = 0; node < positions.size(); ++node) {
            require(std::isfinite(positions[node].x) &&
                std::isfinite(positions[node].y) &&
                std::isfinite(positions[node].z), "nonfinite soft body");
            displacement += std::abs(positions[node].x - reference_positions[node].x);
        }
        for (const auto &bond : soft_scene.soft_bodies[0].bonds) {
            const Vec3 a = positions[bond.first], b = positions[bond.second];
            const float dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
            maximum_strain = std::max(maximum_strain,
                std::sqrt(dx*dx+dy*dy+dz*dz) / bond.rest_length - 1.0F);
        }
        std::cout << "softbody_grid_resolution=" << resolution << 'x'
                  << height << 'x' << resolution
                  << " ms_per_pair_frame=" << soft_ms / frames
                  << " body_x_difference=" << displacement
                  << " dense_near_body=" << dense_near_body
                  << " max_near_density=" << maximum_near_density
                  << " max_bond_stretch=" << maximum_strain << '\n';
        require(displacement > 0.1F && maximum_strain < 0.5F,
            "grid wind did not move the soft body stably");
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
