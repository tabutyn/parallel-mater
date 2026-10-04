// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/fixed_collector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_status(parallel_mater::Status status, const char *operation) {
    if (!status) {
        std::cerr << "FAIL: " << operation << ": "
                  << (status.message != nullptr ? status.message : "unknown")
                  << '\n';
        ++failures;
    }
}

parallel_mater::Quaternion multiply(parallel_mater::Quaternion a,
                                    parallel_mater::Quaternion b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

parallel_mater::Vec3 local_z(parallel_mater::Quaternion orientation) {
    return {
        2.0F * (orientation.x * orientation.z +
                orientation.w * orientation.y),
        2.0F * (orientation.y * orientation.z -
                orientation.w * orientation.x),
        1.0F - 2.0F * (orientation.x * orientation.x +
                       orientation.y * orientation.y),
    };
}

parallel_mater::Vec3 rotate_vector(parallel_mater::Quaternion orientation,
                                   parallel_mater::Vec3 vector) {
    const parallel_mater::Quaternion rotated = multiply(
        multiply(orientation, {vector.x, vector.y, vector.z, 0.0F}),
        {-orientation.x, -orientation.y, -orientation.z, orientation.w});
    return {rotated.x, rotated.y, rotated.z};
}

bool same_id(parallel_mater::RigidBodyId first,
             parallel_mater::RigidBodyId second) {
    return first.index == second.index && first.generation == second.generation;
}

parallel_mater::Vec3 body_inertia(
    const parallel_mater::gallery::SceneDefinition &scene,
    const parallel_mater::gallery::RigidBodyDefinition &body) {
    using parallel_mater::Vec3;
    const bool collision = !body.collision_mesh_indices.empty();
    const auto &meshes = collision ? scene.collision_meshes : scene.meshes;
    const auto &indices = collision ? body.collision_mesh_indices
                                    : body.mesh_indices;
    Vec3 minimum{std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 maximum{-std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max()};
    for (const std::uint32_t mesh_index : indices) {
        for (const auto &vertex : meshes[mesh_index].vertices) {
            minimum.x = std::min(minimum.x, vertex.position.x);
            minimum.y = std::min(minimum.y, vertex.position.y);
            minimum.z = std::min(minimum.z, vertex.position.z);
            maximum.x = std::max(maximum.x, vertex.position.x);
            maximum.y = std::max(maximum.y, vertex.position.y);
            maximum.z = std::max(maximum.z, vertex.position.z);
        }
    }
    const Vec3 half{(maximum.x - minimum.x) * 0.5F,
                    (maximum.y - minimum.y) * 0.5F,
                    (maximum.z - minimum.z) * 0.5F};
    return {body.options.mass * (half.y * half.y + half.z * half.z) / 3.0F,
            body.options.mass * (half.x * half.x + half.z * half.z) / 3.0F,
            body.options.mass * (half.x * half.x + half.y * half.y) / 3.0F};
}

float mechanical_energy(const parallel_mater::RigidBodyOptions &options,
                        parallel_mater::Vec3 inertia,
                        const parallel_mater::RigidBodyState &state) {
    const parallel_mater::Vec3 local_angular = rotate_vector(
        {-state.orientation.x, -state.orientation.y,
         -state.orientation.z, state.orientation.w},
        state.angular_velocity);
    const float linear_squared =
        state.linear_velocity.x * state.linear_velocity.x +
        state.linear_velocity.y * state.linear_velocity.y +
        state.linear_velocity.z * state.linear_velocity.z;
    const float rotational =
        inertia.x * local_angular.x * local_angular.x +
        inertia.y * local_angular.y * local_angular.y +
        inertia.z * local_angular.z * local_angular.z;
    return options.mass * 9.81F * state.position.y +
           0.5F * options.mass * linear_squared + 0.5F * rotational;
}

struct ExpectedScene {
    const char *path;
    parallel_mater::RigidConstraintType type;
    std::size_t constraint_count;
    std::uint32_t solver_iterations{16U};
};

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    constexpr std::array scenes{
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH,
                      RigidConstraintType::fixed, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH,
                      RigidConstraintType::point, 4U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH,
                      RigidConstraintType::hinge, 3U, 64U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_SLIDER_SCENE_PATH,
                      RigidConstraintType::slider, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH,
                      RigidConstraintType::piston, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH,
                      RigidConstraintType::generic, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_GENERIC_SPRING_SCENE_PATH,
                      RigidConstraintType::generic_spring, 1U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_MOTOR_SCENE_PATH,
                      RigidConstraintType::motor, 4U},
    };

    std::array<SceneDefinition, scenes.size()> definitions{};
    for (std::size_t index = 0U; index < scenes.size(); ++index) {
        std::string error;
        check(load_glb_scene(scenes[index].path, definitions[index], error),
              error.empty() ? "load rigid constraint scene" : error.c_str());
        const SceneDefinition &scene = definitions[index];
        check(scene.rigid_constraints.size() == scenes[index].constraint_count,
              "scene must contain expected constraint count");
        for (const auto &constraint : scene.rigid_constraints) {
            check(constraint.options.type == scenes[index].type,
                  "scene must retain Blender constraint type");
            check(constraint.body_a < scene.rigid_bodies.size() &&
                      constraint.body_b < scene.rigid_bodies.size(),
                  "constraint must resolve both rigid body names");
            check(constraint.options.solver_iterations ==
                      scenes[index].solver_iterations,
                  "constraint must retain authored solver iterations");
        }
    }

    const SceneDefinition &fixed_scene = definitions[0];
    const auto &fixed_definition = fixed_scene.rigid_constraints.front();
    const auto &fixed = fixed_definition.options;
    check(fixed.enabled, "fixed collector seed constraint must start enabled");
    check(fixed_scene.rigid_bodies.size() == 51U,
          "fixed collector scene must contain Ground, Large, and 49 small spheres");
    check(fixed_scene.rigid_bodies[fixed_definition.body_a].source_name ==
                  "Large" &&
              fixed_scene.rigid_bodies[fixed_definition.body_b].source_name ==
                  "Small.048",
          "fixed collector seed must join Large to Small.048");
    std::size_t fixed_small_count = 0U;
    std::size_t fixed_large_index = fixed_scene.rigid_bodies.size();
    for (std::size_t index = 0U; index < fixed_scene.rigid_bodies.size();
         ++index) {
        const auto &body = fixed_scene.rigid_bodies[index];
        if (body.source_name == "Large") fixed_large_index = index;
        if (body.source_name == "Ground") {
            check(std::fabs(body.options.friction - 4.0F) < 1.0e-5F,
                  "fixed-scene ground must retain high rolling friction");
        }
        if (body.source_name.rfind("Small", 0U) != 0U) continue;
        ++fixed_small_count;
        check(std::fabs(body.options.friction - 16.0F) < 1.0e-5F,
              "fixed-scene small spheres must retain extreme rolling friction");
        const Vec3 velocity = body.options.initial_state.linear_velocity;
        check(velocity.x == 0.0F && velocity.y == 0.0F && velocity.z == 0.0F,
              "loose fixed-scene spheres must start at rest");
    }
    check(fixed_small_count == 49U,
          "fixed collector scene must retain all 49 small spheres");
    check(fixed_large_index < fixed_scene.rigid_bodies.size(),
          "fixed collector scene must resolve the large sphere");
    SceneInstance fixed_bindings{};
    fixed_bindings.rigid_bodies.resize(fixed_scene.rigid_bodies.size());
    for (std::size_t index = 0U; index < fixed_bindings.rigid_bodies.size();
         ++index) {
        fixed_bindings.rigid_bodies[index] = {
            static_cast<std::uint32_t>(index), 1U};
    }
    FixedContactCollector fixed_structure{};
    check_status(fixed_structure.initialize(fixed_scene, fixed_bindings),
                 "initialize fixed collector structure");
    check(fixed_structure.attached_count() == 2U,
          "fixed collector must seed Large and its authored small sphere");
    check(FixedContactCollector::constraint_capacity(fixed_scene) >= 49U,
          "fixed collector must reserve capacity for contact-created joints");
    const auto &point_constraints = definitions[1].rigid_constraints;
    check(std::all_of(point_constraints.begin(), point_constraints.end(),
                      [](const auto &constraint) {
                          return constraint.options.enabled;
                      }),
          "point constraints must start enabled");
    std::array<bool, 4U> point_sphere_seen{};
    const std::array<const char *, 4U> point_sphere_names{
        "PointSphereA", "PointSphereB", "PointSphereB.001",
        "PointSphereB.002"};
    const std::size_t point_post = point_constraints.front().body_a;
    for (const auto &constraint : point_constraints) {
        check(constraint.body_a == point_post,
              "all point spheres must share one static anchor body");
        const auto &body = definitions[1].rigid_bodies[constraint.body_b];
        for (std::size_t index = 0U; index < point_sphere_names.size(); ++index)
            if (body.source_name == point_sphere_names[index])
                point_sphere_seen[index] = true;
        const Vec3 radial = body.options.initial_state.position;
        const Vec3 velocity = body.options.initial_state.linear_velocity;
        const float radial_dot_velocity =
            radial.x * velocity.x + radial.z * velocity.z;
        check(std::fabs(radial_dot_velocity) < 1.0e-4F &&
                  std::sqrt(velocity.x * velocity.x +
                            velocity.z * velocity.z) > 2.0F,
              "point spheres must start with tangential velocity");
    }
    check(std::all_of(point_sphere_seen.begin(), point_sphere_seen.end(),
                      [](bool seen) { return seen; }),
          "point scene must retain all four orbiting spheres");
    const auto &point_x_negative = definitions[1].rigid_bodies[
        point_constraints[0].body_b].options.initial_state;
    const auto &point_x_positive = definitions[1].rigid_bodies[
        point_constraints[1].body_b].options.initial_state;
    check(point_x_negative.linear_velocity.z *
                  point_x_positive.linear_velocity.z < 0.0F,
          "X-axis point pair must use opposite tangential Z velocities");
    const auto point_velocity_for = [&](const char *name) {
        for (const auto &body : definitions[1].rigid_bodies)
            if (body.source_name == name)
                return body.options.initial_state.linear_velocity;
        return Vec3{};
    };
    check(point_velocity_for("PointSphereB.001").x *
                  point_velocity_for("PointSphereB.002").x < 0.0F,
          "Z-axis point pair must use opposite tangential X velocities");
    const SceneDefinition &hinge_scene = definitions[2];
    constexpr std::array<const char *, 3U> gear_names{
        "Gear", "Gear.001", "Gear.002"};
    std::array<std::size_t, 3U> gear_indices{
        hinge_scene.rigid_bodies.size(), hinge_scene.rigid_bodies.size(),
        hinge_scene.rigid_bodies.size()};
    std::size_t hinge_sphere_index = hinge_scene.rigid_bodies.size();
    std::array<Vec3, 3U> gear_local_anchors{};
    std::array<Quaternion, 3U> gear_local_orientations{};
    std::size_t hinge_count = 0U;
    std::array<bool, 3U> gear_on_shared_frame{};
    for (std::size_t index = 0U; index < hinge_scene.rigid_bodies.size();
         ++index) {
        if (hinge_scene.rigid_bodies[index].source_name == "HingeSphere")
            hinge_sphere_index = index;
    }
    for (const auto &constraint : hinge_scene.rigid_constraints) {
        const auto &body_a = hinge_scene.rigid_bodies[constraint.body_a];
        const auto &body_b = hinge_scene.rigid_bodies[constraint.body_b];
        if (constraint.options.type == RigidConstraintType::hinge) {
            ++hinge_count;
            check(constraint.options.angular_limits.axes == 0U,
                  "gear hinges must rotate continuously around local Z");
            const Vec3 world_axis = local_z(multiply(
                body_a.options.initial_state.orientation,
                constraint.options.local_orientation_a));
            check(std::fabs(world_axis.x) < 1.0e-4F &&
                      std::fabs(world_axis.y) < 1.0e-4F &&
                      world_axis.z > 0.9999F,
                  "Blender local Z must become the horizontal runtime hinge axis");
            for (std::size_t gear = 0U; gear < gear_names.size(); ++gear) {
                if (body_b.source_name != gear_names[gear]) continue;
                gear_on_shared_frame[gear] = body_a.source_name == "Ground";
                gear_indices[gear] = constraint.body_b;
                gear_local_anchors[gear] = constraint.options.local_anchor_b;
                gear_local_orientations[gear] =
                    constraint.options.local_orientation_b;
            }
        }
    }
    check(hinge_scene.rigid_bodies.size() == 5U && hinge_count == 3U,
          "expanded hinge scene must contain five bodies and three hinges");
    check(std::all_of(gear_on_shared_frame.begin(),
                      gear_on_shared_frame.end(),
                      [](bool attached) { return attached; }),
          "all active gears must hinge against one passive frame");
    check(std::all_of(gear_indices.begin(), gear_indices.end(),
                      [&](std::size_t index) {
                          return index < hinge_scene.rigid_bodies.size();
                      }) &&
              hinge_sphere_index < hinge_scene.rigid_bodies.size(),
          "hinge test must resolve all three gears and loose sphere");
    for (const auto &body : hinge_scene.rigid_bodies) {
        if (body.source_name == "Ground" ||
            body.source_name == "HingeSphere") {
            check(std::fabs(body.options.friction - 4.0F) < 1.0e-5F,
                  "hinge sphere and ground must retain high rolling friction");
        } else if (body.source_name == "Gear") {
            check(std::fabs(body.options.friction - 0.08F) < 1.0e-5F &&
                      body.options.restitution == 0.0F &&
                      std::fabs(body.options.angular_damping - 0.03F) <
                          1.0e-5F,
                  "driving gear teeth must roll without binding or bounce");
        } else if (body.source_name == "Gear.001" ||
                   body.source_name == "Gear.002") {
            check(std::fabs(body.options.mass - 1.0F) < 1.0e-5F &&
                      std::fabs(body.options.friction - 0.08F) < 1.0e-5F &&
                      body.options.restitution == 0.0F &&
                      std::fabs(body.options.angular_damping - 0.01F) <
                          1.0e-5F,
                  "follower gears must remain loose and non-bouncing");
        }
    }
    for (std::size_t index : {3U, 4U}) {
        const auto &constraint =
            definitions[index].rigid_constraints.front().options;
        check(constraint.linear_limits.axes == rigid_constraint_axis_x &&
                  constraint.linear_limits.lower.x == -1.0F &&
                  constraint.linear_limits.upper.x == 1.0F,
              "slider and piston must retain their -1m to +1m travel");
    }
    const auto &generic = definitions[5].rigid_constraints.front().options;
    check(generic.linear_limits.axes == rigid_constraint_all_axes &&
              generic.angular_limits.axes == rigid_constraint_all_axes,
          "generic constraint must limit all six axes");
    const auto &spring = definitions[6].rigid_constraints.front().options;
    check(spring.linear_springs.axes == rigid_constraint_all_axes &&
              spring.angular_springs.axes == rigid_constraint_all_axes,
          "generic spring must spring all six axes");
    for (const auto &constraint : definitions[7].rigid_constraints)
        check(constraint.options.motor.angular_enabled &&
                  constraint.options.motor.angular_maximum_impulse == 8.0F,
              "each car wheel must retain its angular motor");

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: constraint GLB structure passed; CUDA unavailable\n";
        return failures == 0 ? 77 : 1;
    }
    for (std::size_t scene_index = 0U; scene_index < definitions.size();
         ++scene_index) {
        const SceneDefinition &scene = definitions[scene_index];
        World world;
        SceneInstance instance;
        FixedContactCollector fixed_collector{};
        Status setup_status{};
        if (scene_index == 0U) {
            WorldOptions options{};
            setup_status = scene_world_options(scene, options);
            if (setup_status) {
                options.rigid_constraint_capacity =
                    FixedContactCollector::constraint_capacity(scene);
                setup_status = World::create(options, world);
            }
            if (setup_status)
                setup_status = instantiate_scene(scene, world, instance);
            if (setup_status)
                setup_status = fixed_collector.initialize(scene, instance);
        } else {
            setup_status = create_scene_world(scene, world, instance);
        }
        check_status(setup_status, "instantiate rigid constraint scene");
        if (!setup_status) continue;
        WorldStatistics statistics{};
        check_status(world.collect_statistics(statistics),
                     "collect rigid constraint statistics");
        check(statistics.rigid_constraint_count == scene.rigid_constraints.size(),
              "world statistics must expose instantiated constraints");
        if (scene_index == 1U) {
            for (std::size_t constraint_index = 0U;
                 constraint_index < scene.rigid_constraints.size();
                 ++constraint_index) {
                RigidConstraintOptions options =
                    scene.rigid_constraints[constraint_index].options;
                options.body_a = instance.rigid_bodies[
                    scene.rigid_constraints[constraint_index].body_a];
                options.body_b = instance.rigid_bodies[
                    scene.rigid_constraints[constraint_index].body_b];
                options.enabled = false;
                check_status(world.update_rigid_constraint(
                                 instance.rigid_constraints[constraint_index],
                                 options),
                             "disable point constraint through gallery action");
            }
            for (const RigidConstraintId id : instance.rigid_constraints) {
                RigidConstraintState state{};
                check_status(world.read_rigid_constraint_state(id, state),
                             "read disabled point constraint");
                check(!state.enabled,
                      "point action must disable all four constraints");
            }
            for (std::size_t constraint_index = 0U;
                 constraint_index < scene.rigid_constraints.size();
                 ++constraint_index) {
                RigidConstraintOptions options =
                    scene.rigid_constraints[constraint_index].options;
                options.body_a = instance.rigid_bodies[
                    scene.rigid_constraints[constraint_index].body_a];
                options.body_b = instance.rigid_bodies[
                    scene.rigid_constraints[constraint_index].body_b];
                options.enabled = true;
                check_status(world.update_rigid_constraint(
                                 instance.rigid_constraints[constraint_index],
                                 options),
                             "enable point constraint through gallery action");
            }
            for (const RigidConstraintId id : instance.rigid_constraints) {
                RigidConstraintState state{};
                check_status(world.read_rigid_constraint_state(id, state),
                             "read enabled point constraint");
                check(state.enabled,
                      "point action must enable all four constraints");
            }
        }
        std::array<bool, 2U> saw_gear_contact{};
        std::array<bool, 2U> saw_opposite_gear_rotation{};
        std::array<float, 2U> minimum_outward_dot{1.0e30F, 1.0e30F};
        std::array<float, 2U> maximum_gear_penetration{};
        int first_sphere_impact_frame = -1;
        float minimum_impact_rotation = 0.0F;
        float maximum_impact_rotation = 0.0F;
        float maximum_backward_recovery = 0.0F;
        float maximum_fixed_angular_speed = 0.0F;
        float maximum_fixed_roll_ratio = 0.0F;
        float maximum_fixed_height =
            fixed_large_index < fixed_scene.rigid_bodies.size()
                ? fixed_scene.rigid_bodies[fixed_large_index]
                      .options.initial_state.position.y
                : 0.0F;
        std::array<Vec3, 4U> hinge_inertias{};
        std::array<std::size_t, 4U> hinge_dynamic_indices{
            gear_indices[0], gear_indices[1], gear_indices[2],
            hinge_sphere_index};
        float initial_hinge_energy = 0.0F;
        float maximum_hinge_energy = 0.0F;
        if (scene_index == 2U) {
            for (std::size_t index = 0U;
                 index < hinge_dynamic_indices.size(); ++index) {
                const auto &body = hinge_scene.rigid_bodies[
                    hinge_dynamic_indices[index]];
                hinge_inertias[index] = body_inertia(hinge_scene, body);
                initial_hinge_energy += mechanical_energy(
                    body.options, hinge_inertias[index],
                    body.options.initial_state);
            }
            maximum_hinge_energy = initial_hinge_energy;
        }
        const int frame_count = scene_index == 2U ? 1200 :
                                scene_index == 0U ? 240 : 30;
        for (int frame = 0; frame < frame_count; ++frame) {
            if (scene_index == 0U)
                check_status(fixed_collector.apply_loose_gravity(
                                 world, scene, instance,
                                 {0.0F, -9.81F, 0.0F},
                                 {9.660964F, -1.703489F, 0.0F}),
                             "apply loose fixed-scene test gravity");
            check_status(world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = scene_index == 2U ? 8U : 4U,
                                     .gravity = scene_index == 0U
                                         ? Vec3{9.660964F, -1.703489F, 0.0F}
                                         : Vec3{0.0F, -9.81F, 0.0F},
                                     .collect_rigid_contacts =
                                         scene_index == 0U ||
                                         scene_index == 2U}),
                         "step rigid constraint scene");
            if (scene_index == 0U)
                check_status(fixed_collector.collect(world, scene, instance),
                             "collect fixed constraint contacts");
            if (scene_index == 0U &&
                fixed_large_index < instance.rigid_bodies.size()) {
                RigidBodyState state{};
                check_status(world.read_rigid_body_state(
                                 instance.rigid_bodies[fixed_large_index], state),
                             "read fixed collector state");
                maximum_fixed_angular_speed = std::max(
                    maximum_fixed_angular_speed,
                    std::fabs(state.angular_velocity.z));
                maximum_fixed_height = std::max(maximum_fixed_height,
                                                state.position.y);
                if (std::fabs(state.linear_velocity.x) > 0.25F) {
                    maximum_fixed_roll_ratio = std::max(
                        maximum_fixed_roll_ratio,
                        std::fabs(state.angular_velocity.z) * 1.05F /
                            std::fabs(state.linear_velocity.x));
                }
            }
            if (scene_index != 2U) continue;
            std::array<RigidBodyState, 4U> hinge_states{};
            float hinge_energy = 0.0F;
            for (std::size_t index = 0U;
                 index < hinge_dynamic_indices.size(); ++index) {
                check_status(world.read_rigid_body_state(
                                 instance.rigid_bodies[
                                     hinge_dynamic_indices[index]],
                                 hinge_states[index]),
                             "read natural hinge body state");
                const auto &body = hinge_scene.rigid_bodies[
                    hinge_dynamic_indices[index]];
                hinge_energy += mechanical_energy(
                    body.options, hinge_inertias[index], hinge_states[index]);
            }
            maximum_hinge_energy = std::max(maximum_hinge_energy,
                                             hinge_energy);
            for (std::size_t pair = 0U; pair < 2U; ++pair)
                saw_opposite_gear_rotation[pair] =
                    saw_opposite_gear_rotation[pair] ||
                    hinge_states[pair].angular_velocity.z *
                            hinge_states[pair + 1U].angular_velocity.z <
                        -1.0e-4F;
            const RigidContactDeviceView view = world.rigid_contacts();
            std::vector<RigidContactEvent> contacts(view.event_count);
            if (!contacts.empty())
                check(cudaMemcpy(contacts.data(), view.events.data,
                                 contacts.size() * sizeof(RigidContactEvent),
                                 cudaMemcpyDeviceToHost) == cudaSuccess,
                      "download hinge gear contacts");
            for (const RigidContactEvent &contact : contacts) {
                const bool sphere_hits_lever =
                    (same_id(contact.body,
                             instance.rigid_bodies[hinge_sphere_index]) &&
                     same_id(contact.collider,
                             instance.rigid_bodies[gear_indices[0]])) ||
                    (same_id(contact.collider,
                             instance.rigid_bodies[hinge_sphere_index]) &&
                     same_id(contact.body,
                             instance.rigid_bodies[gear_indices[0]]));
                if (sphere_hits_lever && first_sphere_impact_frame < 0)
                    first_sphere_impact_frame = frame;
                for (std::size_t pair = 0U; pair < 2U; ++pair) {
                    const bool first_is_body =
                        same_id(contact.body,
                                instance.rigid_bodies[gear_indices[pair]]) &&
                        same_id(contact.collider,
                                instance.rigid_bodies[gear_indices[pair + 1U]]);
                    const bool first_is_collider =
                        same_id(contact.collider,
                                instance.rigid_bodies[gear_indices[pair]]) &&
                        same_id(contact.body,
                                instance.rigid_bodies[gear_indices[pair + 1U]]);
                    if (!first_is_body && !first_is_collider) continue;
                    saw_gear_contact[pair] = true;
                    const Vec3 rotated_anchor = rotate_vector(
                        hinge_states[pair].orientation,
                        gear_local_anchors[pair]);
                    const Vec3 world_anchor{
                        hinge_states[pair].position.x + rotated_anchor.x,
                        hinge_states[pair].position.y + rotated_anchor.y,
                        hinge_states[pair].position.z + rotated_anchor.z};
                    const float direction = first_is_body ? 1.0F : -1.0F;
                    const float outward_dot = direction *
                        ((world_anchor.x - contact.position.x) *
                             contact.normal.x +
                         (world_anchor.y - contact.position.y) *
                             contact.normal.y +
                         (world_anchor.z - contact.position.z) *
                             contact.normal.z);
                    minimum_outward_dot[pair] = std::min(
                        minimum_outward_dot[pair], outward_dot);
                    maximum_gear_penetration[pair] = std::max(
                        maximum_gear_penetration[pair], contact.penetration);
                }
            }
            if (first_sphere_impact_frame >= 0 &&
                frame < first_sphere_impact_frame + 12) {
                const Quaternion initial = hinge_scene.rigid_bodies[
                    gear_indices[0]].options.initial_state.orientation;
                const Quaternion delta = multiply(
                    hinge_states[0].orientation,
                    {-initial.x, -initial.y, -initial.z, initial.w});
                // The ball approaches in +X below the hinge: r x impulse
                // must swing the lever in +Z, including position recovery.
                const float rotation = 2.0F * std::atan2(delta.z, delta.w);
                minimum_impact_rotation = std::min(minimum_impact_rotation,
                                                   rotation);
                maximum_impact_rotation = std::max(maximum_impact_rotation,
                                                   rotation);
                maximum_backward_recovery = std::max(
                    maximum_backward_recovery, maximum_impact_rotation - rotation);
            }
        }
        if (scene_index == 0U) {
            check(fixed_collector.attached_count() > 2U,
                  "fixed collector must attach a loose sphere after contact");
            check_status(world.collect_statistics(statistics),
                         "collect fixed collector statistics");
            check(statistics.rigid_constraint_count > 1U,
                  "fixed collector must create constraints through the API");
            check(maximum_fixed_angular_speed > 0.1F &&
                      maximum_fixed_roll_ratio > 0.5F,
                  "attached small spheres must grip the floor and roll the "
                  "collector");
            const float initial_height =
                fixed_large_index < fixed_scene.rigid_bodies.size()
                ? fixed_scene.rigid_bodies[fixed_large_index]
                      .options.initial_state.position.y
                : 0.0F;
            check(maximum_fixed_height > initial_height + 0.05F,
                  "tilted collector gravity must lift the large sphere onto "
                  "a small sphere");
            if (maximum_fixed_angular_speed <= 0.1F ||
                maximum_fixed_roll_ratio <= 0.5F ||
                maximum_fixed_height <= initial_height + 0.05F)
                std::cerr << "fixed angular speed="
                          << maximum_fixed_angular_speed
                          << " roll ratio=" << maximum_fixed_roll_ratio
                          << " height=" << maximum_fixed_height << '\n';
        }
        if (scene_index == 2U) {
            check(first_sphere_impact_frame >= 0,
                  "hinge sphere must strike the lever during natural replay");
            check(minimum_impact_rotation >= -0.002F &&
                      maximum_backward_recovery <= 0.002F &&
                      maximum_impact_rotation > 0.01F,
                  "ball impact must swing the lever forward without backward recovery");
            if (minimum_impact_rotation < -0.002F ||
                maximum_backward_recovery > 0.002F ||
                maximum_impact_rotation <= 0.01F)
                std::cerr << "impact rotation min=" << minimum_impact_rotation
                          << " max=" << maximum_impact_rotation
                          << " backward recovery=" << maximum_backward_recovery
                          << '\n';
            for (std::size_t pair = 0U; pair < 2U; ++pair) {
                check(saw_gear_contact[pair],
                      "both hinge gear interfaces must engage in natural replay");
                check(minimum_outward_dot[pair] > -1.0e-4F,
                      "hinge gear contact normals must point out of the driving gear");
                const float maximum_gear_recovery =
                    hinge_scene.rigid_bodies[gear_indices[pair]]
                        .options.collision_margin +
                    hinge_scene.rigid_bodies[gear_indices[pair + 1U]]
                        .options.collision_margin +
                    1.0e-4F;
                check(maximum_gear_penetration[pair] <= maximum_gear_recovery,
                      "gear contact recovery must stay inside its search shell");
                if (maximum_gear_penetration[pair] > maximum_gear_recovery)
                    std::cerr << "gear pair " << pair << " penetration="
                              << maximum_gear_penetration[pair] << '\n';
                check(saw_opposite_gear_rotation[pair],
                      "impact must propagate with alternating gear rotation");
            }
            if (maximum_hinge_energy > initial_hinge_energy + 0.5F)
                std::cerr << "natural hinge energy=" << initial_hinge_energy
                          << " peak=" << maximum_hinge_energy << '\n';
            check(maximum_hinge_energy <= initial_hinge_energy + 0.5F,
                  "hinge contacts must not inject mechanical energy");

            std::array<RigidBodyState, 3U> driven{
                scene.rigid_bodies[gear_indices[0]].options.initial_state,
                scene.rigid_bodies[gear_indices[1]].options.initial_state,
                scene.rigid_bodies[gear_indices[2]].options.initial_state};
            driven[0].angular_velocity = {0.0F, 0.0F, -12.0F};
            driven[1].angular_velocity = {0.0F, 0.0F, 6.0F};
            driven[2].angular_velocity = {0.0F, 0.0F, -6.0F};
            const auto hinge_linear_velocity = [](RigidBodyState state,
                                                  Vec3 local_anchor) {
                const Vec3 anchor_to_center = rotate_vector(
                    state.orientation,
                    {-local_anchor.x, -local_anchor.y, -local_anchor.z});
                return Vec3{
                    state.angular_velocity.y * anchor_to_center.z -
                        state.angular_velocity.z * anchor_to_center.y,
                    state.angular_velocity.z * anchor_to_center.x -
                        state.angular_velocity.x * anchor_to_center.z,
                    state.angular_velocity.x * anchor_to_center.y -
                        state.angular_velocity.y * anchor_to_center.x};
            };
            for (std::size_t gear = 0U; gear < driven.size(); ++gear)
                driven[gear].linear_velocity = hinge_linear_velocity(
                    driven[gear], gear_local_anchors[gear]);
            RigidBodyState isolated_sphere =
                scene.rigid_bodies[hinge_sphere_index].options.initial_state;
            isolated_sphere.position.x += 100.0F;
            isolated_sphere.linear_velocity = {};
            isolated_sphere.angular_velocity = {};
            check_status(world.set_rigid_body_state(
                             instance.rigid_bodies[hinge_sphere_index],
                             isolated_sphere),
                         "isolate driven hinge gears from the loose sphere");
            for (std::size_t gear = 0U; gear < driven.size(); ++gear)
                check_status(world.set_rigid_body_state(
                                 instance.rigid_bodies[gear_indices[gear]],
                                 driven[gear]),
                             "drive hinge gear through full rotations");
            std::array<Vec3, 3U> initial_anchors{};
            std::array<Vec3, 3U> initial_axes{};
            for (std::size_t gear = 0U; gear < driven.size(); ++gear) {
                const Vec3 anchor_offset = rotate_vector(
                    driven[gear].orientation, gear_local_anchors[gear]);
                initial_anchors[gear] = {
                    driven[gear].position.x + anchor_offset.x,
                    driven[gear].position.y + anchor_offset.y,
                    driven[gear].position.z + anchor_offset.z};
                initial_axes[gear] = local_z(multiply(
                    driven[gear].orientation,
                    gear_local_orientations[gear]));
            }
            Quaternion previous_orientation = driven[0].orientation;
            float accumulated_rotation = 0.0F;
            float maximum_anchor_drift = 0.0F;
            int maximum_anchor_drift_frame = 0;
            float minimum_axis_dot = 1.0F;
            std::array<bool, 2U> driven_opposite_rotation{};
            for (int frame = 0; frame < 360; ++frame) {
                check_status(world.apply_force(
                                 instance.rigid_bodies[gear_indices[0]],
                                 {0.0F, -10.0F, 0.0F},
                                 {driven[0].position.x + 1.0F,
                                  driven[0].position.y,
                                  driven[0].position.z}),
                             "drive first half of hinge torque couple");
                check_status(world.apply_force(
                                 instance.rigid_bodies[gear_indices[0]],
                                 {0.0F, 10.0F, 0.0F},
                                 {driven[0].position.x - 1.0F,
                                  driven[0].position.y,
                                  driven[0].position.z}),
                             "drive second half of hinge torque couple");
                check_status(world.step({.timestep = 1.0F / 60.0F,
                                         .substeps = 8U,
                                         .gravity = {}}),
                             "step full-rotation hinge regression");
                for (std::size_t gear = 0U; gear < driven.size(); ++gear)
                    check_status(world.read_rigid_body_state(
                                     instance.rigid_bodies[gear_indices[gear]],
                                     driven[gear]),
                                 "read driven hinge gear");
                for (std::size_t pair = 0U; pair < 2U; ++pair)
                    driven_opposite_rotation[pair] =
                        driven_opposite_rotation[pair] ||
                        driven[pair].angular_velocity.z *
                                driven[pair + 1U].angular_velocity.z <
                            -1.0e-4F;
                Quaternion delta = multiply(
                    driven[0].orientation,
                    {-previous_orientation.x, -previous_orientation.y,
                     -previous_orientation.z, previous_orientation.w});
                if (delta.w < 0.0F)
                    delta = {-delta.x, -delta.y, -delta.z, -delta.w};
                accumulated_rotation += 2.0F * std::atan2(
                    std::sqrt(delta.x * delta.x + delta.y * delta.y +
                              delta.z * delta.z),
                    std::max(delta.w, 0.0F));
                previous_orientation = driven[0].orientation;
                const auto drift = [](Vec3 current, Vec3 initial) {
                    const float x = current.x - initial.x;
                    const float y = current.y - initial.y;
                    const float z = current.z - initial.z;
                    return std::sqrt(x * x + y * y + z * z);
                };
                for (std::size_t gear = 0U; gear < driven.size(); ++gear) {
                    const Vec3 anchor_offset = rotate_vector(
                        driven[gear].orientation, gear_local_anchors[gear]);
                    const Vec3 anchor{
                        driven[gear].position.x + anchor_offset.x,
                        driven[gear].position.y + anchor_offset.y,
                        driven[gear].position.z + anchor_offset.z};
                    const float anchor_drift = drift(
                        anchor, initial_anchors[gear]);
                    if (anchor_drift > maximum_anchor_drift) {
                        maximum_anchor_drift = anchor_drift;
                        maximum_anchor_drift_frame = frame;
                    }
                    const Vec3 axis = local_z(multiply(
                        driven[gear].orientation,
                        gear_local_orientations[gear]));
                    minimum_axis_dot = std::min(
                        minimum_axis_dot,
                        axis.x * initial_axes[gear].x +
                            axis.y * initial_axes[gear].y +
                            axis.z * initial_axes[gear].z);
                }
            }
            check(accumulated_rotation > 2.0F * 3.14159265358979323846F,
                  "small hinge gear must complete a full rotation");
            if (accumulated_rotation <= 2.0F * 3.14159265358979323846F ||
                maximum_anchor_drift >= 0.002F ||
                minimum_axis_dot <= 0.9999F)
                std::cerr << "full rotation=" << accumulated_rotation
                          << " anchor_drift=" << maximum_anchor_drift
                          << " drift_frame=" << maximum_anchor_drift_frame
                          << " axis_dot=" << minimum_axis_dot
                          << " angular_velocity="
                          << driven[0].angular_velocity.z << '\n';
            check(maximum_anchor_drift < 0.002F,
                  "all three hinge anchors must remain within 2 mm");
            check(minimum_axis_dot > 0.9999F,
                  "all three hinge axes must remain aligned");
            check(std::all_of(driven_opposite_rotation.begin(),
                              driven_opposite_rotation.end(),
                              [](bool opposite) { return opposite; }),
                  "driven motion must alternate across both gear interfaces");
        }
    }
    return failures == 0 ? 0 : 1;
}
