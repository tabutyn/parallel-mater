// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/surface_query.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using parallel_mater::RigidBodyState;
using parallel_mater::SoftBodyId;
using parallel_mater::Vec3;
using parallel_mater::World;

bool require(parallel_mater::Status status, const char *operation) {
    if (status) return true;
    std::cerr << operation << ": "
              << (status.message != nullptr ? status.message : "unknown")
              << '\n';
    return false;
}

bool read_soft_state(const World &world, SoftBodyId id,
                     std::vector<Vec3> &positions,
                     std::vector<Vec3> &velocities) {
    parallel_mater::SoftBodyDeviceView view{};
    if (!require(world.soft_body_view(id, view), "borrow soft-body state"))
        return false;
    positions.resize(view.node_count);
    velocities.resize(view.node_count);
    return cudaMemcpy(positions.data(), view.positions.data,
                      positions.size() * sizeof(Vec3),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(velocities.data(), view.velocities.data,
                      velocities.size() * sizeof(Vec3),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

float length(Vec3 value) {
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

Vec3 center(const std::vector<Vec3> &positions) {
    Vec3 result{};
    for (Vec3 value : positions) {
        result.x += value.x;
        result.y += value.y;
        result.z += value.z;
    }
    const float inverse = 1.0F / static_cast<float>(positions.size());
    return {result.x * inverse, result.y * inverse, result.z * inverse};
}

float soft_momentum_x(
    const parallel_mater::gallery::SoftBodyDefinition &definition,
    const std::vector<Vec3> &velocities) {
    float momentum = 0.0F;
    for (std::size_t node = 0U; node < velocities.size(); ++node) {
        const float inverse_mass = definition.inverse_masses.empty()
            ? 1.0F / definition.node_mass
            : definition.inverse_masses[node];
        if (inverse_mass > 0.0F)
            momentum += velocities[node].x / inverse_mass;
    }
    return momentum;
}

float radial_shape_error(const std::vector<Vec3> &rest,
                         const std::vector<Vec3> &positions) {
    const Vec3 rest_center = center(rest);
    const Vec3 current_center = center(positions);
    float error_squared = 0.0F;
    float radius_squared = 0.0F;
    for (std::size_t node = 0U; node < rest.size(); ++node) {
        const float rest_radius = length({rest[node].x - rest_center.x,
            rest[node].y - rest_center.y, rest[node].z - rest_center.z});
        const float current_radius = length({
            positions[node].x - current_center.x,
            positions[node].y - current_center.y,
            positions[node].z - current_center.z});
        const float difference = current_radius - rest_radius;
        error_squared += difference * difference;
        radius_squared += rest_radius * rest_radius;
    }
    return std::sqrt(error_squared / std::max(radius_squared, 1.0e-12F));
}

struct RecoveryResult {
    float peak_error{};
    float loaded_error{};
    float recovered_error{};
    float recovered_speed{};
};

struct ContainmentResult {
    std::uint32_t escaped_nodes{};
    float minimum_z{std::numeric_limits<float>::max()};
    float maximum_z{-std::numeric_limits<float>::max()};
};

bool run_steered_containment(
    const parallel_mater::gallery::SceneDefinition &scene,
    ContainmentResult &output) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    World world;
    SceneInstance instance{};
    if (!require(create_scene_world(scene, world, instance),
                 "create steered soft-rigid world")) return false;
    constexpr float timestep = 1.0F / 60.0F;
    constexpr float gravity_magnitude = 9.81F;
    constexpr float diagonal = 0.70710678118F;
    CameraController camera;
    camera.set_preset(gallery_entry(GalleryContext::soft_body_rigid).camera);
    Vec3 gravity{0.0F, -gravity_magnitude * diagonal,
                 -gravity_magnitude * diagonal};
    for (std::uint32_t frame = 0U; frame < 300U; ++frame) {
        gravity = steer_gravity(gravity, camera.camera(), -1.0F, 0.0F,
                                gravity_magnitude, 45.0F, timestep);
        const StepOptions step{.timestep = timestep, .substeps = 4U,
                               .gravity = gravity};
        if (!require(world.step(step), "step steered soft-rigid world"))
            return false;
    }
    std::vector<Vec3> positions, velocities;
    if (!read_soft_state(world, instance.soft_bodies.front(), positions,
                         velocities)) return false;
    StaticTriangleSurface passive_surface;
    std::string error;
    if (!StaticTriangleSurface::create(scene, passive_surface, error)) {
        std::cerr << error << '\n';
        return false;
    }
    const Vec3 minimum = passive_surface.minimum();
    const Vec3 maximum = passive_surface.maximum();
    const float tolerance = scene.soft_bodies.front().node_radius + 0.02F;
    for (Vec3 node : positions) {
        output.minimum_z = std::min(output.minimum_z, node.z);
        output.maximum_z = std::max(output.maximum_z, node.z);
        output.escaped_nodes += node.x < minimum.x - tolerance ||
            node.x > maximum.x + tolerance ||
            node.z < minimum.z - tolerance ||
            node.z > maximum.z + tolerance;
    }
    return true;
}

bool run_symmetric_crush(parallel_mater::gallery::SceneDefinition scene,
                         float shape_stiffness, RecoveryResult &output) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    scene.soft_bodies.front().shape_matching_stiffness = shape_stiffness;
    scene.rigid_bodies.erase(std::remove_if(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const RigidBodyDefinition &body) {
            return body.options.motion != MotionType::dynamic;
        }), scene.rigid_bodies.end());
    if (scene.rigid_bodies.size() != 2U) return false;
    World world;
    SceneInstance instance{};
    if (!require(create_scene_world(scene, world, instance),
                 "create symmetric crush world")) return false;
    const Vec3 soft_center = center(scene.soft_bodies.front().nodes);
    const auto order = scene.rigid_bodies[0].options.initial_state.position.x <
            scene.rigid_bodies[1].options.initial_state.position.x
        ? std::array<std::size_t, 2U>{0U, 1U}
        : std::array<std::size_t, 2U>{1U, 0U};
    for (std::size_t side = 0U; side < 2U; ++side) {
        RigidBodyState state =
            scene.rigid_bodies[order[side]].options.initial_state;
        const float direction = side == 0U ? 1.0F : -1.0F;
        state.position = {soft_center.x - direction * 2.5F,
                          soft_center.y, soft_center.z};
        state.linear_velocity = {direction * 1.5F, 0.0F, 0.0F};
        state.angular_velocity = {};
        if (!require(world.set_rigid_body_state(
                instance.rigid_bodies[order[side]], state),
                "set symmetric crush sphere")) return false;
    }
    const StepOptions step{.timestep = 1.0F / 240.0F, .substeps = 1U,
                           .gravity = {}};
    std::vector<Vec3> positions, velocities;
    for (std::uint32_t frame = 0U; frame < 180U; ++frame) {
        if (!require(world.step(step), "step symmetric crush") ||
            !read_soft_state(world, instance.soft_bodies.front(), positions,
                             velocities)) return false;
        output.peak_error = std::max(output.peak_error,
            radial_shape_error(scene.soft_bodies.front().nodes, positions));
    }
    output.loaded_error = radial_shape_error(
        scene.soft_bodies.front().nodes, positions);
    for (RigidBodyId body : instance.rigid_bodies)
        if (!require(world.remove_rigid_body(body),
                     "remove symmetric crush sphere")) return false;
    for (std::uint32_t frame = 0U; frame < 60U; ++frame)
        if (!require(world.step(step), "recover symmetric crush")) return false;
    if (!read_soft_state(world, instance.soft_bodies.front(), positions,
                         velocities)) return false;
    output.recovered_error = radial_shape_error(
        scene.soft_bodies.front().nodes, positions);
    for (Vec3 velocity : velocities)
        output.recovered_speed = std::max(
            output.recovered_speed, length(velocity));
    return true;
}

struct ImpactResult {
    RigidBodyState rigid{};
    std::vector<Vec3> soft_positions{};
    std::vector<Vec3> soft_velocities{};
    float soft_displacement{};
    float peak_soft_speed{};
    float mean_balance_error{1.0F};
    float maximum_balance_error{};
    float minimum_separation{std::numeric_limits<float>::max()};
    std::uint32_t transfer_steps{};
};

bool run_impact(parallel_mater::gallery::SceneDefinition scene,
                float impact_speed, float projectile_mass,
                ImpactResult &output) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    const auto soft_center = center(scene.soft_bodies.front().nodes);
    scene.rigid_bodies.erase(std::remove_if(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const RigidBodyDefinition &body) {
            return body.options.motion != MotionType::dynamic;
        }), scene.rigid_bodies.end());
    const auto left = std::min_element(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const RigidBodyDefinition &a, const RigidBodyDefinition &b) {
            return a.options.initial_state.position.x <
                   b.options.initial_state.position.x;
        });
    if (left == scene.rigid_bodies.end()) return false;
    RigidBodyDefinition projectile = *left;
    projectile.options.mass = projectile_mass;
    scene.rigid_bodies.assign(1U, projectile);

    World world;
    SceneInstance instance{};
    if (!require(create_scene_world(scene, world, instance),
                 "create isolated soft-rigid world") ||
        instance.rigid_bodies.size() != 1U ||
        instance.soft_bodies.size() != 1U) return false;
    RigidBodyState state = projectile.options.initial_state;
    state.position = {soft_center.x - 2.50F, soft_center.y,
                      soft_center.z + 0.20F};
    state.linear_velocity = {impact_speed, 0.0F, 0.0F};
    state.angular_velocity = {};
    if (!require(world.set_rigid_body_state(instance.rigid_bodies.front(), state),
                 "set high-speed projectile")) return false;

    std::vector<Vec3> positions, velocities;
    if (!read_soft_state(world, instance.soft_bodies.front(), positions,
                         velocities)) return false;
    const Vec3 initial_soft_center = center(positions);
    const float rigid_mass = projectile.options.mass;
    float previous_rigid_momentum = rigid_mass * state.linear_velocity.x;
    float previous_soft_momentum = soft_momentum_x(
        scene.soft_bodies.front(), velocities);
    const StepOptions step{.timestep = 1.0F / 240.0F, .substeps = 1U,
                           .gravity = {}};
    float balance_error_sum = 0.0F;
    float maximum_balance_error = 0.0F;
    std::uint32_t transfer_steps = 0U;
    float minimum_separation = std::numeric_limits<float>::max();
    float peak_soft_speed = 0.0F;
    for (std::uint32_t frame = 0U; frame < 240U; ++frame) {
        if (!require(world.step(step), "step isolated soft-rigid impact") ||
            !require(world.read_rigid_body_state(
                instance.rigid_bodies.front(), state),
                "read projectile state") ||
            !read_soft_state(world, instance.soft_bodies.front(), positions,
                             velocities)) return false;
        const Vec3 soft_center_now = center(positions);
        minimum_separation = std::min(minimum_separation,
            length({state.position.x - soft_center_now.x,
                    state.position.y - soft_center_now.y,
                    state.position.z - soft_center_now.z}));
        for (Vec3 velocity : velocities) {
            if (!std::isfinite(velocity.x) || !std::isfinite(velocity.y) ||
                !std::isfinite(velocity.z)) return false;
            peak_soft_speed = std::max(peak_soft_speed, length(velocity));
        }
        const float rigid_momentum = rigid_mass * state.linear_velocity.x;
        const float soft_momentum = soft_momentum_x(
            scene.soft_bodies.front(), velocities);
        const float rigid_transfer = previous_rigid_momentum - rigid_momentum;
        const float soft_transfer = soft_momentum - previous_soft_momentum;
        const float transfer = std::max(std::fabs(rigid_transfer),
                                        std::fabs(soft_transfer));
        if (rigid_transfer > 0.01F && soft_transfer > 0.01F) {
            const float error =
                std::fabs(rigid_transfer - soft_transfer) / transfer;
            balance_error_sum += error;
            maximum_balance_error = std::max(maximum_balance_error, error);
            ++transfer_steps;
        }
        previous_rigid_momentum = rigid_momentum;
        previous_soft_momentum = soft_momentum;
    }
    output.rigid = state;
    output.soft_positions = std::move(positions);
    output.soft_velocities = std::move(velocities);
    output.soft_displacement = center(output.soft_positions).x -
                               initial_soft_center.x;
    output.peak_soft_speed = peak_soft_speed;
    output.mean_balance_error = transfer_steps != 0U
        ? balance_error_sum / static_cast<float>(transfer_steps) : 1.0F;
    output.maximum_balance_error = maximum_balance_error;
    output.minimum_separation = minimum_separation;
    output.transfer_steps = transfer_steps;
    return true;
}

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    SceneDefinition scene{};
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH,
                        scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const std::size_t dynamic_count = static_cast<std::size_t>(std::count_if(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const RigidBodyDefinition &body) {
            return body.options.motion == MotionType::dynamic;
        }));
    const std::size_t static_count = static_cast<std::size_t>(std::count_if(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const RigidBodyDefinition &body) {
            return body.options.motion == MotionType::static_body;
        }));
    const bool authored_heavy_spheres = std::all_of(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const RigidBodyDefinition &body) {
            return body.options.motion != MotionType::dynamic ||
                   std::fabs(body.options.mass - 100.0F) < 1.0e-4F;
        });
    if (scene.soft_bodies.size() != 1U || dynamic_count != 2U ||
        static_count != 1U || !authored_heavy_spheres ||
        scene.soft_bodies.front().shape_matching_stiffness <= 0.0F) {
        std::cerr << "SoftbodyRigidBody scene needs one soft body, two active "
                     "100 kg rigid bodies, and one passive arena\n";
        return 1;
    }

    World settled_world;
    SceneInstance settled_instance{};
    if (!require(create_scene_world(scene, settled_world, settled_instance),
                 "create authored soft-rigid scene")) return 1;
    const StepOptions settle_step{.timestep = 1.0F / 60.0F, .substeps = 4U,
                                  .gravity = {0.0F, -9.81F, 0.0F}};
    for (std::uint32_t frame = 0U; frame < 1'200U; ++frame)
        if (!require(settled_world.step(settle_step),
                     "settle authored soft-rigid scene")) return 1;
    float maximum_rigid_speed = 0.0F;
    for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
        if (scene.rigid_bodies[index].options.motion != MotionType::dynamic)
            continue;
        RigidBodyState state{};
        if (!require(settled_world.read_rigid_body_state(
            settled_instance.rigid_bodies[index], state),
            "read settled rigid sphere")) return 1;
        maximum_rigid_speed = std::max(maximum_rigid_speed,
                                       length(state.linear_velocity));
    }
    std::vector<Vec3> settled_positions, settled_velocities;
    if (!read_soft_state(settled_world,
                         settled_instance.soft_bodies.front(),
                         settled_positions, settled_velocities)) return 1;
    float maximum_soft_speed = 0.0F;
    for (Vec3 position : settled_positions) {
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) {
            std::cerr << "Soft-rigid contact produced a non-finite node\n";
            return 1;
        }
    }
    for (Vec3 velocity : settled_velocities)
        maximum_soft_speed = std::max(maximum_soft_speed, length(velocity));
    float maximum_bond_strain = 0.0F;
    for (const SoftBodyBond &bond : scene.soft_bodies.front().bonds) {
        maximum_bond_strain = std::max(maximum_bond_strain,
            std::fabs(length({
                settled_positions[bond.first].x -
                    settled_positions[bond.second].x,
                settled_positions[bond.first].y -
                    settled_positions[bond.second].y,
                settled_positions[bond.first].z -
                    settled_positions[bond.second].z}) /
                bond.rest_length - 1.0F));
    }
    StaticTriangleSurface passive_surface;
    if (!StaticTriangleSurface::create(scene, passive_surface, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::uint32_t escaped_nodes = 0U;
    for (Vec3 node : settled_positions) {
        const auto floor = passive_surface.height(
            node.x, node.z, SurfaceSelection::lowest);
        escaped_nodes += floor &&
            node.y < *floor - scene.soft_bodies.front().node_radius - 0.02F;
    }
    StepOptions timed_step = settle_step;
    timed_step.collect_kernel_timings = true;
    if (!require(settled_world.step(timed_step),
                 "step timed soft-rigid frame")) return 1;
    WorldStepTimings timings{};
    if (!require(settled_world.collect_step_timings(timings),
                 "collect soft-rigid timings") || !timings.available) {
        std::cerr << "Soft-rigid timing sample is unavailable\n";
        return 1;
    }
    if (maximum_rigid_speed > 0.35F || maximum_soft_speed > 0.35F ||
        maximum_bond_strain > 0.45F || escaped_nodes != 0U) {
        std::cerr << "Soft-rigid authored scene did not settle: rigid="
                  << maximum_rigid_speed << " soft=" << maximum_soft_speed
                  << " strain=" << maximum_bond_strain
                  << " escaped=" << escaped_nodes
                  << '\n';
        return 1;
    }

    RecoveryResult matched_recovery{}, spring_recovery{};
    ContainmentResult containment{};
    if (!run_symmetric_crush(scene,
            scene.soft_bodies.front().shape_matching_stiffness,
            matched_recovery) ||
        !run_symmetric_crush(scene, 0.0F, spring_recovery) ||
        !run_steered_containment(scene, containment)) return 1;
    if (matched_recovery.peak_error < 0.05F ||
        matched_recovery.loaded_error <
            spring_recovery.loaded_error * 0.98F ||
        matched_recovery.recovered_error >
            matched_recovery.loaded_error * 0.01F ||
        matched_recovery.recovered_error >
            spring_recovery.recovered_error * 0.25F ||
        matched_recovery.recovered_speed >
            scene.soft_bodies.front().maximum_speed + 0.05F) {
        std::cerr << "Soft-body rest-shape recovery regressed: peak="
                  << matched_recovery.peak_error
                  << " loaded=" << matched_recovery.loaded_error
                  << " recovered=" << matched_recovery.recovered_error
                  << " spring_recovered="
                  << spring_recovery.recovered_error
                  << " speed=" << matched_recovery.recovered_speed << '\n';
        return 1;
    }
    if (containment.escaped_nodes != 0U) {
        std::cerr << "Contact-aware shape restoration crossed the arena: "
                  << containment.escaped_nodes << " nodes, z="
                  << containment.minimum_z << ".."
                  << containment.maximum_z << '\n';
        return 1;
    }

    ImpactResult light{}, heavy{}, heavy_repeat{}, high_speed{};
    if (!run_impact(scene, 1.5F, 1.0F, light) ||
        !run_impact(scene, 1.5F, 100.0F, heavy) ||
        !run_impact(scene, 1.5F, 100.0F, heavy_repeat) ||
        !run_impact(scene, 4.0F, 100.0F, high_speed)) return 1;
    const bool deterministic =
        std::memcmp(&heavy.rigid, &heavy_repeat.rigid,
                    sizeof(heavy.rigid)) == 0 &&
        heavy.soft_positions.size() == heavy_repeat.soft_positions.size() &&
        std::memcmp(heavy.soft_positions.data(),
                    heavy_repeat.soft_positions.data(),
                    heavy.soft_positions.size() * sizeof(Vec3)) == 0 &&
        std::memcmp(heavy.soft_velocities.data(),
                    heavy_repeat.soft_velocities.data(),
                    heavy.soft_velocities.size() * sizeof(Vec3)) == 0;
    if (light.transfer_steps == 0U || light.mean_balance_error > 0.08F ||
        light.maximum_balance_error > 0.25F ||
        light.soft_displacement < 0.02F ||
        light.rigid.linear_velocity.x > 1.3F ||
        light.peak_soft_speed >
            scene.soft_bodies.front().maximum_speed + 0.05F ||
        light.minimum_separation < 1.25F ||
        heavy.soft_displacement < light.soft_displacement * 1.25F ||
        heavy.rigid.linear_velocity.x < light.rigid.linear_velocity.x + 0.5F ||
        heavy.peak_soft_speed >
            scene.soft_bodies.front().maximum_speed + 0.05F ||
        heavy.minimum_separation < 1.25F ||
        high_speed.transfer_steps == 0U ||
        high_speed.soft_displacement < 0.05F ||
        high_speed.peak_soft_speed >
            scene.soft_bodies.front().maximum_speed + 0.05F ||
        high_speed.minimum_separation < 1.25F || !deterministic) {
        std::cerr << "Soft-rigid coupling regression: light_transfers="
                  << light.transfer_steps
                  << " light_mean_balance_error=" << light.mean_balance_error
                  << " maximum_balance_error="
                  << light.maximum_balance_error
                  << " light_soft_dx=" << light.soft_displacement
                  << " light_rigid_vx=" << light.rigid.linear_velocity.x
                  << " heavy_soft_dx=" << heavy.soft_displacement
                  << " heavy_rigid_vx=" << heavy.rigid.linear_velocity.x
                  << " heavy_peak_soft_speed=" << heavy.peak_soft_speed
                  << " heavy_separation=" << heavy.minimum_separation
                  << " high_speed_dx=" << high_speed.soft_displacement
                  << " high_speed_separation="
                  << high_speed.minimum_separation
                  << " deterministic=" << deterministic << '\n';
        return 1;
    }
    std::cout << "Soft-rigid light_transfers=" << light.transfer_steps
              << " light_mean_balance_error=" << light.mean_balance_error
              << " light_maximum_balance_error="
              << light.maximum_balance_error
              << " light_soft_dx=" << light.soft_displacement
              << " light_rigid_vx=" << light.rigid.linear_velocity.x
              << " heavy_soft_dx=" << heavy.soft_displacement
              << " heavy_rigid_vx=" << heavy.rigid.linear_velocity.x
              << " heavy_peak_soft_speed=" << heavy.peak_soft_speed
              << " heavy_separation=" << heavy.minimum_separation
              << " high_speed_dx=" << high_speed.soft_displacement
              << " high_speed_separation=" << high_speed.minimum_separation
              << " settled_rigid_speed=" << maximum_rigid_speed
              << " settled_soft_speed=" << maximum_soft_speed
              << " settled_strain=" << maximum_bond_strain
              << " escaped_nodes=" << escaped_nodes
              << " matched_peak_error=" << matched_recovery.peak_error
              << " matched_loaded_error=" << matched_recovery.loaded_error
              << " matched_recovered_error="
              << matched_recovery.recovered_error
              << " matched_recovered_speed="
              << matched_recovery.recovered_speed
              << " spring_peak_error=" << spring_recovery.peak_error
              << " spring_recovered_error="
              << spring_recovery.recovered_error
              << " steered_escaped_nodes=" << containment.escaped_nodes
              << " steered_z=" << containment.minimum_z << ".."
              << containment.maximum_z
              << " gpu_ms=" << timings.total_gpu_milliseconds << '\n';
    return 0;
}
