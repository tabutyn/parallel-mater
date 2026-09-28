// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/surface_query.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool require(parallel_mater::Status status, const char *operation) {
    if (status) return true;
    std::cerr << operation << ": "
              << (status.message != nullptr ? status.message : "unknown")
              << '\n';
    return false;
}

bool read_nodes(const parallel_mater::World &world,
                parallel_mater::SoftBodyId id,
                std::vector<parallel_mater::Vec3> &positions) {
    parallel_mater::SoftBodyDeviceView view{};
    if (!require(world.soft_body_view(id, view), "borrow soft body"))
        return false;
    positions.resize(view.node_count);
    return positions.empty() || cudaMemcpy(
        positions.data(), view.positions.data,
        positions.size() * sizeof(positions[0]),
        cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool read_state(const parallel_mater::World &world,
                parallel_mater::SoftBodyId id,
                std::vector<parallel_mater::Vec3> &positions,
                std::vector<parallel_mater::Vec3> &velocities) {
    parallel_mater::SoftBodyDeviceView view{};
    if (!require(world.soft_body_view(id, view), "borrow soft body state"))
        return false;
    positions.resize(view.node_count);
    velocities.resize(view.node_count);
    return (positions.empty() || cudaMemcpy(
        positions.data(), view.positions.data,
        positions.size() * sizeof(positions[0]),
        cudaMemcpyDeviceToHost) == cudaSuccess) &&
        (velocities.empty() || cudaMemcpy(
        velocities.data(), view.velocities.data,
        velocities.size() * sizeof(velocities[0]),
        cudaMemcpyDeviceToHost) == cudaSuccess);
}

float distance(parallel_mater::Vec3 a, parallel_mater::Vec3 b) {
    const float x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
    return std::sqrt(x * x + y * y + z * z);
}

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    SceneDefinition scene{};
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_SOFT_BODY_SCENE_PATH, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (scene.soft_bodies.size() != 1U || scene.rigid_bodies.empty() ||
        std::any_of(scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
            [](const RigidBodyDefinition &body) {
                return body.options.motion != MotionType::static_body;
            })) {
        std::cerr << "Softbody scene needs one soft body and passive rigid geometry\n";
        return 1;
    }
    const SoftBodyDefinition &definition = scene.soft_bodies.front();
    const TriangleMesh &surface_mesh = scene.meshes[definition.mesh_index];
    const bool has_interior_node = std::any_of(
        definition.nodes.begin(), definition.nodes.end(), [&](Vec3 node) {
            return std::none_of(
                surface_mesh.vertices.begin(), surface_mesh.vertices.end(),
                [&](const Vertex &vertex) {
                    return distance(node, vertex.position) <
                           definition.node_radius * 0.5F;
                });
        });
    if (!has_interior_node ||
        definition.bonds.size() < definition.nodes.size() ||
        definition.surface_bindings.size() !=
            surface_mesh.vertices.size() ||
        definition.maximum_speed > 2.01F ||
        std::fabs(definition.maximum_projection_fraction - 0.20F) > 1.0e-5F ||
        std::fabs(definition.constraint_velocity_response - 0.70F) > 1.0e-5F) {
        std::cerr << "Softbody export did not produce a volumetric lattice\n";
        return 1;
    }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    World world;
    SceneInstance instance{};
    if (!require(create_scene_world(
            scene, world, instance, {.frame_capacity = 2U, .frame_stride = 1U}),
                 "create soft-body scene world") ||
        instance.soft_bodies.size() != 1U) return 1;
    std::vector<Vec3> initial;
    if (!read_nodes(world, instance.soft_bodies.front(), initial)) return 1;

    const StepOptions step{.timestep = 1.0F / 60.0F,
                           .substeps = 4U,
                           .gravity = {0.0F, -9.81F, 0.0F}};
    for (int frame = 0; frame < 600; ++frame)
        if (!require(world.step(step), "step soft-body world")) return 1;
    std::vector<Vec3> final;
    if (!read_nodes(world, instance.soft_bodies.front(), final)) return 1;
    SoftBodyDeviceView settled_view{};
    if (!require(world.soft_body_view(instance.soft_bodies.front(), settled_view),
                 "borrow settled soft body")) return 1;
    if (settled_view.bonds.size != definition.bonds.size()) {
        std::cerr << "Soft-body API view omitted internal spring bonds\n";
        return 1;
    }
    std::vector<Vec3> settled_velocities(settled_view.node_count);
    if (cudaMemcpy(settled_velocities.data(), settled_view.velocities.data,
                   settled_velocities.size() * sizeof(Vec3),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Could not read settled soft-body velocities\n";
        return 1;
    }
    float peak_settled_speed = 0.0F;
    for (Vec3 velocity : settled_velocities)
        peak_settled_speed = std::max(
            peak_settled_speed, distance(velocity, {}));
    float maximum_motion = 0.0F;
    Vec3 initial_center{}, final_center{};
    for (std::size_t node = 0U; node < final.size(); ++node) {
        initial_center.x += initial[node].x;
        initial_center.y += initial[node].y;
        initial_center.z += initial[node].z;
        final_center.x += final[node].x;
        final_center.y += final[node].y;
        final_center.z += final[node].z;
    }
    const float inverse_count = 1.0F / static_cast<float>(final.size());
    initial_center = {initial_center.x * inverse_count,
                      initial_center.y * inverse_count,
                      initial_center.z * inverse_count};
    final_center = {final_center.x * inverse_count,
                    final_center.y * inverse_count,
                    final_center.z * inverse_count};
    float maximum_deformation = 0.0F;
    for (std::size_t node = 0U; node < final.size(); ++node) {
        if (!std::isfinite(final[node].x) || !std::isfinite(final[node].y) ||
            !std::isfinite(final[node].z)) {
            std::cerr << "Soft-body node became non-finite\n";
            return 1;
        }
        maximum_motion = std::max(maximum_motion,
                                  distance(initial[node], final[node]));
        maximum_deformation = std::max(maximum_deformation, std::fabs(
            distance(initial[node], initial_center) -
            distance(final[node], final_center)));
    }
    if (maximum_motion < 0.05F) {
        std::cerr << "Soft body did not respond to gravity\n";
        return 1;
    }
    float maximum_bond_strain = 0.0F;
    for (const SoftBodyBond &bond : definition.bonds) {
        maximum_bond_strain = std::max(maximum_bond_strain,
            std::fabs(distance(final[bond.first], final[bond.second]) /
                      bond.rest_length - 1.0F));
    }
    if (maximum_deformation < 0.01F || maximum_bond_strain > 0.35F ||
        peak_settled_speed > 0.25F) {
        std::cerr << "Soft-body deformation was absent or unstable: deformation="
                  << maximum_deformation << " strain=" << maximum_bond_strain
                  << " peak_speed=" << peak_settled_speed
                  << '\n';
        return 1;
    }
    StaticTriangleSurface passive_surface;
    if (!StaticTriangleSurface::create(scene, passive_surface, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::uint32_t escaped = 0U;
    float minimum_clearance = std::numeric_limits<float>::max();
    for (Vec3 node : final) {
        const auto floor = passive_surface.height(
            node.x, node.z, SurfaceSelection::lowest);
        if (floor) minimum_clearance = std::min(
            minimum_clearance, node.y - *floor);
        escaped += floor && node.y < *floor - definition.node_radius - 0.02F;
    }
    if (escaped != 0U) {
        std::cerr << escaped << " soft-body nodes crossed passive geometry\n";
        return 1;
    }
    StepOptions timed_step = step;
    timed_step.collect_kernel_timings = true;
    if (!require(world.step(timed_step), "step timed soft-body frame"))
        return 1;
    WorldStepTimings timings{};
    if (!require(world.collect_step_timings(timings),
                 "collect soft-body timings") || !timings.available) {
        std::cerr << "Soft-body timing sample is unavailable\n";
        return 1;
    }
    std::cout << "Soft body nodes=" << final.size()
              << " bonds=" << definition.bonds.size()
              << " deformation=" << maximum_deformation
              << " maximum_bond_strain=" << maximum_bond_strain
              << " escaped_nodes=" << escaped
              << " minimum_clearance=" << minimum_clearance
              << " peak_settled_speed=" << peak_settled_speed
              << " gpu_ms=" << timings.total_gpu_milliseconds << '\n';
    WorldStatistics statistics{};
    if (!require(world.collect_statistics(statistics),
                 "collect soft-body statistics") ||
        statistics.soft_body_count != 1U ||
        statistics.soft_body_node_count != definition.nodes.size()) {
        std::cerr << "Soft-body statistics do not match the scene\n";
        return 1;
    }
    PhysicsDebugFrameView debug{};
    if (!require(world.physics_debug_frame(debug),
                 "borrow soft-body debug frame") ||
        debug.soft_body_nodes.size != definition.nodes.size()) {
        std::cerr << "Soft-body physics capture omitted lattice nodes\n";
        return 1;
    }

    std::vector<Vec3> rolling_positions;
    std::vector<Vec3> rolling_velocities;
    const Vec3 rolling_start = final_center;
    float rolling_radius = 0.0F;
    for (Vec3 node : final)
        rolling_radius = std::max(
            rolling_radius, std::fabs(node.y - rolling_start.y));
    float peak_rolling_ratio = 0.0F;
    float rolling_ratio_sum = 0.0F;
    std::uint32_t rolling_ratio_samples = 0U;
    float peak_rolling_speed = 0.0F;
    float peak_node_speed = 0.0F;
    float rolling_distance_before_wall = 0.0F;
    float maximum_rolling_bond_strain = 0.0F;
    std::uint32_t wall_escape_count = 0U;
    Vec3 rolling_center = rolling_start;
    const StepOptions tilted_step{.timestep = 1.0F / 60.0F,
                                  .substeps = 4U,
                                  .gravity = {6.936718F, -6.936718F, 0.0F}};
    float minimum_wall_velocity = 0.0F;
    for (int frame = 0; frame < 180; ++frame) {
        if (!require(world.step(tilted_step), "step rolling soft body") ||
            !read_state(world, instance.soft_bodies.front(), rolling_positions,
                        rolling_velocities)) return 1;
        rolling_center = {};
        Vec3 center_velocity{};
        for (std::size_t node = 0U; node < rolling_positions.size(); ++node) {
            rolling_center.x += rolling_positions[node].x;
            rolling_center.y += rolling_positions[node].y;
            rolling_center.z += rolling_positions[node].z;
            center_velocity.x += rolling_velocities[node].x;
            center_velocity.y += rolling_velocities[node].y;
            center_velocity.z += rolling_velocities[node].z;
        }
        rolling_center = {rolling_center.x * inverse_count,
                          rolling_center.y * inverse_count,
                          rolling_center.z * inverse_count};
        center_velocity = {center_velocity.x * inverse_count,
                           center_velocity.y * inverse_count,
                           center_velocity.z * inverse_count};
        float angular_numerator = 0.0F;
        float angular_denominator = 0.0F;
        for (std::size_t node = 0U; node < rolling_positions.size(); ++node) {
            const float x = rolling_positions[node].x - rolling_center.x;
            const float y = rolling_positions[node].y - rolling_center.y;
            const float vx = rolling_velocities[node].x - center_velocity.x;
            const float vy = rolling_velocities[node].y - center_velocity.y;
            angular_numerator += x * vy - y * vx;
            angular_denominator += x * x + y * y;
        }
        const float angular_speed = angular_denominator > 1.0e-6F
            ? std::fabs(angular_numerator / angular_denominator) : 0.0F;
        const float rolling_speed = std::fabs(center_velocity.x);
        for (Vec3 velocity : rolling_velocities)
            peak_node_speed = std::max(
                peak_node_speed, distance(velocity, {}));
        const Vec3 passive_minimum = passive_surface.minimum();
        const Vec3 passive_maximum = passive_surface.maximum();
        wall_escape_count += static_cast<std::uint32_t>(std::count_if(
            rolling_positions.begin(), rolling_positions.end(),
            [&](Vec3 node) {
                return node.x < passive_minimum.x - 0.02F ||
                       node.x > passive_maximum.x + 0.02F ||
                       node.z < passive_minimum.z - 0.02F ||
                       node.z > passive_maximum.z + 0.02F;
            }));
        for (const SoftBodyBond &bond : definition.bonds)
            maximum_rolling_bond_strain = std::max(
                maximum_rolling_bond_strain,
                std::fabs(distance(rolling_positions[bond.first],
                                   rolling_positions[bond.second]) /
                          bond.rest_length - 1.0F));
        if (frame >= 60)
            minimum_wall_velocity = std::min(
                minimum_wall_velocity, center_velocity.x);
        peak_rolling_speed = std::max(peak_rolling_speed, rolling_speed);
        if (frame < 60 && rolling_speed > 0.1F) {
            const float rolling_ratio =
                angular_speed * rolling_radius / rolling_speed;
            peak_rolling_ratio = std::max(
                peak_rolling_ratio, rolling_ratio);
            if (frame >= 20) {
                rolling_ratio_sum += rolling_ratio;
                ++rolling_ratio_samples;
            }
        }
        if (frame == 59)
            rolling_distance_before_wall =
                rolling_center.x - rolling_start.x;
    }
    const float rolling_distance = rolling_center.x - rolling_start.x;
    const float mean_rolling_ratio = rolling_ratio_samples != 0U
        ? rolling_ratio_sum / static_cast<float>(rolling_ratio_samples) : 0.0F;
    if (rolling_distance_before_wall < 1.0F || mean_rolling_ratio < 0.75F ||
        peak_node_speed > definition.maximum_speed + 0.05F ||
        minimum_wall_velocity > -0.15F ||
        minimum_wall_velocity < -1.8F ||
        maximum_rolling_bond_strain > 0.5F || wall_escape_count != 0U) {
        std::cerr << "Soft-body traction or passive impact response regressed: "
                  << "distance=" << rolling_distance_before_wall
                  << " rolling_ratio=" << mean_rolling_ratio
                  << " node_speed=" << peak_node_speed
                  << " rebound=" << minimum_wall_velocity
                  << " strain=" << maximum_rolling_bond_strain
                  << " wall_escapes=" << wall_escape_count << '\n';
        return 1;
    }
    std::cout << "Soft body rolling distance=" << rolling_distance
              << " pre_wall_distance=" << rolling_distance_before_wall
              << " peak_speed=" << peak_rolling_speed
              << " peak_node_speed=" << peak_node_speed
              << " peak_ratio=" << peak_rolling_ratio
              << " mean_ratio=" << mean_rolling_ratio
              << " minimum_wall_velocity=" << minimum_wall_velocity
              << " maximum_strain=" << maximum_rolling_bond_strain
              << " wall_escapes=" << wall_escape_count << '\n';

    const SoftBodyId removed = instance.soft_bodies.front();
    if (!require(world.remove_soft_body(removed), "remove soft body")) return 1;
    SoftBodyDeviceView stale{};
    if (world.soft_body_view(removed, stale).code != StatusCode::invalid_handle) {
        std::cerr << "Removed soft-body handle remained valid\n";
        return 1;
    }
    return 0;
}
