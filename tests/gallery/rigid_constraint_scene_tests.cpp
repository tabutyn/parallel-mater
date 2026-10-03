// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

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
                      RigidConstraintType::point, 2U},
        ExpectedScene{PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH,
                      RigidConstraintType::hinge, 2U, 64U},
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

    const auto &fixed = definitions[0].rigid_constraints.front().options;
    check(!fixed.enabled, "interactive fixed constraint must start released");
    const auto &point_constraints = definitions[1].rigid_constraints;
    check(point_constraints[0].options.enabled &&
              point_constraints[1].options.enabled,
          "point constraints must start enabled");
    check(point_constraints[0].body_a == point_constraints[1].body_a &&
              point_constraints[0].body_b != point_constraints[1].body_b,
          "point spheres must share one static anchor body");
    check(point_constraints[0].options.local_anchor_a.x ==
                  point_constraints[1].options.local_anchor_a.x &&
              point_constraints[0].options.local_anchor_a.y ==
                  point_constraints[1].options.local_anchor_a.y &&
              point_constraints[0].options.local_anchor_a.z ==
                  point_constraints[1].options.local_anchor_a.z &&
              point_constraints[0].options.local_anchor_b.x *
                  point_constraints[1].options.local_anchor_b.x < 0.0F,
          "point spheres must pivot around opposite sides of one point");
    const auto &point_a = definitions[1].rigid_bodies[
        point_constraints[0].body_b].options.initial_state;
    const auto &point_b = definitions[1].rigid_bodies[
        point_constraints[1].body_b].options.initial_state;
    check(point_a.linear_velocity.z * point_b.linear_velocity.z < 0.0F &&
              std::fabs(point_a.linear_velocity.z) > 2.0F &&
              std::fabs(point_b.linear_velocity.z) > 2.0F,
          "point spheres must start with opposite tangential velocities");
    const SceneDefinition &hinge_scene = definitions[2];
    std::size_t small_gear_index = hinge_scene.rigid_bodies.size();
    std::size_t large_gear_index = hinge_scene.rigid_bodies.size();
    std::size_t hinge_sphere_index = hinge_scene.rigid_bodies.size();
    Vec3 small_gear_local_anchor{};
    Vec3 large_gear_local_anchor{};
    Quaternion small_gear_local_orientation{};
    Quaternion large_gear_local_orientation{};
    std::size_t hinge_count = 0U;
    bool small_gear_on_shared_frame = false;
    bool large_gear_on_shared_frame = false;
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
            small_gear_on_shared_frame = small_gear_on_shared_frame ||
                (body_a.source_name == "Ground" &&
                 body_b.source_name == "Gear");
            large_gear_on_shared_frame = large_gear_on_shared_frame ||
                (body_a.source_name == "Ground" &&
                 body_b.source_name == "Gear.001");
            if (body_b.source_name == "Gear") {
                small_gear_index = constraint.body_b;
                small_gear_local_anchor = constraint.options.local_anchor_b;
                small_gear_local_orientation =
                    constraint.options.local_orientation_b;
            }
            if (body_b.source_name == "Gear.001") {
                large_gear_index = constraint.body_b;
                large_gear_local_anchor = constraint.options.local_anchor_b;
                large_gear_local_orientation =
                    constraint.options.local_orientation_b;
            }
        }
    }
    check(hinge_scene.rigid_bodies.size() == 4U && hinge_count == 2U,
          "merged hinge scene must contain four bodies and two hinges");
    check(small_gear_on_shared_frame && large_gear_on_shared_frame,
          "both active gears must hinge against one passive frame");
    check(small_gear_index < hinge_scene.rigid_bodies.size() &&
              large_gear_index < hinge_scene.rigid_bodies.size() &&
              hinge_sphere_index < hinge_scene.rigid_bodies.size(),
          "hinge test must resolve the gears and loose sphere");
    for (const auto &body : hinge_scene.rigid_bodies) {
        if (body.source_name == "Gear") {
            check(std::fabs(body.options.friction - 0.08F) < 1.0e-5F &&
                      body.options.restitution == 0.0F &&
                      std::fabs(body.options.angular_damping - 0.03F) <
                          1.0e-5F,
                  "driving gear teeth must roll without binding or bounce");
        } else if (body.source_name == "Gear.001") {
            check(std::fabs(body.options.mass - 1.0F) < 1.0e-5F &&
                      std::fabs(body.options.friction - 0.08F) < 1.0e-5F &&
                      body.options.restitution == 0.0F &&
                      std::fabs(body.options.angular_damping - 0.01F) <
                          1.0e-5F,
                  "large gear must remain a loose non-bouncing follower");
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
        check_status(create_scene_world(scene, world, instance),
                     "instantiate rigid constraint scene");
        WorldStatistics statistics{};
        check_status(world.collect_statistics(statistics),
                     "collect rigid constraint statistics");
        check(statistics.rigid_constraint_count == scene.rigid_constraints.size(),
              "world statistics must expose instantiated constraints");
        bool saw_gear_contact = false;
        bool saw_opposite_gear_rotation = false;
        float minimum_outward_dot = 1.0e30F;
        float maximum_gear_penetration = 0.0F;
        int first_sphere_impact_frame = -1;
        float minimum_impact_rotation = 0.0F;
        float maximum_impact_rotation = 0.0F;
        float maximum_backward_recovery = 0.0F;
        const int frame_count = scene_index == 2U ? 1200 : 30;
        for (int frame = 0; frame < frame_count; ++frame) {
            check_status(world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = scene_index == 2U ? 8U : 4U,
                                     .gravity = {0.0F, -9.81F, 0.0F},
                                     .collect_rigid_contacts =
                                         scene_index == 2U}),
                         "step rigid constraint scene");
            if (scene_index != 2U) continue;
            RigidBodyState small_state{};
            RigidBodyState large_state{};
            check_status(world.read_rigid_body_state(
                             instance.rigid_bodies[small_gear_index],
                             small_state),
                         "read small hinge gear state");
            check_status(world.read_rigid_body_state(
                             instance.rigid_bodies[large_gear_index],
                             large_state),
                         "read large hinge gear state");
            saw_opposite_gear_rotation = saw_opposite_gear_rotation ||
                small_state.angular_velocity.z *
                        large_state.angular_velocity.z <
                    -1.0e-4F;
            const Vec3 rotated_anchor = rotate_vector(
                small_state.orientation, small_gear_local_anchor);
            const Vec3 world_anchor{
                small_state.position.x + rotated_anchor.x,
                small_state.position.y + rotated_anchor.y,
                small_state.position.z + rotated_anchor.z};
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
                             instance.rigid_bodies[small_gear_index])) ||
                    (same_id(contact.collider,
                             instance.rigid_bodies[hinge_sphere_index]) &&
                     same_id(contact.body,
                             instance.rigid_bodies[small_gear_index]));
                if (sphere_hits_lever && first_sphere_impact_frame < 0)
                    first_sphere_impact_frame = frame;
                const bool small_is_body =
                    same_id(contact.body,
                            instance.rigid_bodies[small_gear_index]) &&
                    same_id(contact.collider,
                            instance.rigid_bodies[large_gear_index]);
                const bool small_is_collider =
                    same_id(contact.collider,
                            instance.rigid_bodies[small_gear_index]) &&
                    same_id(contact.body,
                            instance.rigid_bodies[large_gear_index]);
                if (!small_is_body && !small_is_collider) continue;
                saw_gear_contact = true;
                const float direction = small_is_body ? 1.0F : -1.0F;
                const float outward_dot = direction *
                    ((world_anchor.x - contact.position.x) * contact.normal.x +
                     (world_anchor.y - contact.position.y) * contact.normal.y +
                     (world_anchor.z - contact.position.z) * contact.normal.z);
                minimum_outward_dot =
                    std::min(minimum_outward_dot, outward_dot);
                maximum_gear_penetration =
                    std::max(maximum_gear_penetration, contact.penetration);
            }
            if (first_sphere_impact_frame >= 0 &&
                frame < first_sphere_impact_frame + 12) {
                const Quaternion initial = hinge_scene.rigid_bodies[
                    small_gear_index].options.initial_state.orientation;
                const Quaternion delta = multiply(
                    small_state.orientation,
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
            check(saw_gear_contact,
                  "hinge gears must engage during the scene replay");
            check(minimum_outward_dot > -1.0e-4F,
                  "hinge gear contact normals must point out of the small gear");
            const float maximum_gear_recovery =
                hinge_scene.rigid_bodies[small_gear_index]
                    .options.collision_margin +
                hinge_scene.rigid_bodies[large_gear_index]
                    .options.collision_margin +
                1.0e-4F;
            check(maximum_gear_penetration <= maximum_gear_recovery,
                  "gear contact recovery must stay inside its search shell");
            if (maximum_gear_penetration > maximum_gear_recovery)
                std::cerr << "gear penetration=" << maximum_gear_penetration
                          << '\n';
            check(saw_opposite_gear_rotation,
                  "meshed hinge gears must rotate in opposite directions");

            // Regression for a user capture where a large gear-tooth impact
            // increased total mechanical energy by 22 J in two frames.  Zero
            // restitution, damping, and gravity work must not create energy.
            RigidBodyState captured_small{
                .position = {1.61932397F, 5.29441452F, 0.321598262F},
                .orientation = {0.489382893F, 0.510373354F,
                                0.51044178F, 0.4893592F},
                .linear_velocity = {0.096713528F, -2.27621722F,
                                    0.00183799304F},
                .angular_velocity = {-0.013401052F, 0.00615845667F,
                                     -1.40645504F}};
            RigidBodyState captured_large{
                .position = {-0.00471570203F, 2.17593265F, 1.51186085F},
                .orientation = {0.666613698F, -0.236210853F,
                                -0.236284733F, 0.666333199F},
                .linear_velocity = {1.33645872e-11F, 0.00528716994F, 0.0F},
                .angular_velocity = {-0.0588675253F, 0.00490533235F,
                                     0.554055095F}};
            check_status(world.set_rigid_body_state(
                             instance.rigid_bodies[small_gear_index],
                             captured_small),
                         "restore excited small hinge gear capture");
            check_status(world.set_rigid_body_state(
                             instance.rigid_bodies[large_gear_index],
                             captured_large),
                         "restore excited large hinge gear capture");
            const Vec3 small_inertia = body_inertia(
                hinge_scene, hinge_scene.rigid_bodies[small_gear_index]);
            const Vec3 large_inertia = body_inertia(
                hinge_scene, hinge_scene.rigid_bodies[large_gear_index]);
            const auto gear_energy = [&]() {
                return mechanical_energy(
                           hinge_scene.rigid_bodies[small_gear_index].options,
                           small_inertia, captured_small) +
                       mechanical_energy(
                           hinge_scene.rigid_bodies[large_gear_index].options,
                           large_inertia, captured_large);
            };
            const float initial_energy = gear_energy();
            float maximum_energy = initial_energy;
            for (int frame = 0; frame < 4; ++frame) {
                check_status(world.step({.timestep = 1.0F / 60.0F,
                                         .substeps = 8U,
                                         .gravity = {0.0F, -9.81F, 0.0F}}),
                             "step excited hinge energy regression");
                check_status(world.read_rigid_body_state(
                                 instance.rigid_bodies[small_gear_index],
                                 captured_small),
                             "read excited small hinge gear");
                check_status(world.read_rigid_body_state(
                                 instance.rigid_bodies[large_gear_index],
                                 captured_large),
                             "read excited large hinge gear");
                maximum_energy = std::max(maximum_energy, gear_energy());
            }
            if (maximum_energy > initial_energy + 0.5F)
                std::cerr << "excited hinge energy=" << initial_energy
                          << " peak=" << maximum_energy << '\n';
            check(maximum_energy <= initial_energy + 0.5F,
                  "hinge gear contacts must not amplify captured oscillation");

            RigidBodyState driven_small =
                scene.rigid_bodies[small_gear_index].options.initial_state;
            RigidBodyState driven_large =
                scene.rigid_bodies[large_gear_index].options.initial_state;
            driven_small.angular_velocity = {0.0F, 0.0F, -12.0F};
            driven_large.angular_velocity = {0.0F, 0.0F, 6.0F};
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
            driven_small.linear_velocity = hinge_linear_velocity(
                driven_small, small_gear_local_anchor);
            driven_large.linear_velocity = hinge_linear_velocity(
                driven_large, large_gear_local_anchor);
            RigidBodyState isolated_sphere =
                scene.rigid_bodies[hinge_sphere_index].options.initial_state;
            isolated_sphere.position.x += 100.0F;
            isolated_sphere.linear_velocity = {};
            isolated_sphere.angular_velocity = {};
            check_status(world.set_rigid_body_state(
                             instance.rigid_bodies[hinge_sphere_index],
                             isolated_sphere),
                         "isolate driven hinge gears from the loose sphere");
            check_status(world.set_rigid_body_state(
                             instance.rigid_bodies[small_gear_index],
                             driven_small),
                         "drive small hinge gear through full rotations");
            check_status(world.set_rigid_body_state(
                             instance.rigid_bodies[large_gear_index],
                             driven_large),
                         "drive large hinge gear through full rotations");
            const Vec3 initial_small_anchor_offset = rotate_vector(
                driven_small.orientation, small_gear_local_anchor);
            const Vec3 initial_small_anchor{
                driven_small.position.x + initial_small_anchor_offset.x,
                driven_small.position.y + initial_small_anchor_offset.y,
                driven_small.position.z + initial_small_anchor_offset.z};
            const Vec3 initial_large_anchor_offset = rotate_vector(
                driven_large.orientation, large_gear_local_anchor);
            const Vec3 initial_large_anchor{
                driven_large.position.x + initial_large_anchor_offset.x,
                driven_large.position.y + initial_large_anchor_offset.y,
                driven_large.position.z + initial_large_anchor_offset.z};
            const Vec3 initial_small_axis = local_z(multiply(
                driven_small.orientation, small_gear_local_orientation));
            const Vec3 initial_large_axis = local_z(multiply(
                driven_large.orientation, large_gear_local_orientation));
            Quaternion previous_orientation = driven_small.orientation;
            float accumulated_rotation = 0.0F;
            float maximum_anchor_drift = 0.0F;
            int maximum_anchor_drift_frame = 0;
            float minimum_axis_dot = 1.0F;
            for (int frame = 0; frame < 360; ++frame) {
                check_status(world.apply_force(
                                 instance.rigid_bodies[small_gear_index],
                                 {0.0F, -10.0F, 0.0F},
                                 {driven_small.position.x + 1.0F,
                                  driven_small.position.y,
                                  driven_small.position.z}),
                             "drive first half of hinge torque couple");
                check_status(world.apply_force(
                                 instance.rigid_bodies[small_gear_index],
                                 {0.0F, 10.0F, 0.0F},
                                 {driven_small.position.x - 1.0F,
                                  driven_small.position.y,
                                  driven_small.position.z}),
                             "drive second half of hinge torque couple");
                check_status(world.step({.timestep = 1.0F / 60.0F,
                                         .substeps = 8U,
                                         .gravity = {}}),
                             "step full-rotation hinge regression");
                check_status(world.read_rigid_body_state(
                                 instance.rigid_bodies[small_gear_index],
                                 driven_small),
                             "read driven small hinge gear");
                check_status(world.read_rigid_body_state(
                                 instance.rigid_bodies[large_gear_index],
                                 driven_large),
                             "read driven large hinge gear");
                Quaternion delta = multiply(
                    driven_small.orientation,
                    {-previous_orientation.x, -previous_orientation.y,
                     -previous_orientation.z, previous_orientation.w});
                if (delta.w < 0.0F)
                    delta = {-delta.x, -delta.y, -delta.z, -delta.w};
                accumulated_rotation += 2.0F * std::atan2(
                    std::sqrt(delta.x * delta.x + delta.y * delta.y +
                              delta.z * delta.z),
                    std::max(delta.w, 0.0F));
                previous_orientation = driven_small.orientation;
                const Vec3 small_anchor_offset = rotate_vector(
                    driven_small.orientation, small_gear_local_anchor);
                const Vec3 large_anchor_offset = rotate_vector(
                    driven_large.orientation, large_gear_local_anchor);
                const Vec3 small_anchor{
                    driven_small.position.x + small_anchor_offset.x,
                    driven_small.position.y + small_anchor_offset.y,
                    driven_small.position.z + small_anchor_offset.z};
                const Vec3 large_anchor{
                    driven_large.position.x + large_anchor_offset.x,
                    driven_large.position.y + large_anchor_offset.y,
                    driven_large.position.z + large_anchor_offset.z};
                const auto drift = [](Vec3 current, Vec3 initial) {
                    const float x = current.x - initial.x;
                    const float y = current.y - initial.y;
                    const float z = current.z - initial.z;
                    return std::sqrt(x * x + y * y + z * z);
                };
                const float anchor_drift = std::max(
                    drift(small_anchor, initial_small_anchor),
                    drift(large_anchor, initial_large_anchor));
                if (anchor_drift > maximum_anchor_drift) {
                    maximum_anchor_drift = anchor_drift;
                    maximum_anchor_drift_frame = frame;
                }
                const Vec3 small_axis = local_z(multiply(
                    driven_small.orientation, small_gear_local_orientation));
                const Vec3 large_axis = local_z(multiply(
                    driven_large.orientation, large_gear_local_orientation));
                minimum_axis_dot = std::min(
                    minimum_axis_dot,
                    std::min(small_axis.x * initial_small_axis.x +
                                 small_axis.y * initial_small_axis.y +
                                 small_axis.z * initial_small_axis.z,
                             large_axis.x * initial_large_axis.x +
                                 large_axis.y * initial_large_axis.y +
                                 large_axis.z * initial_large_axis.z));
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
                          << driven_small.angular_velocity.z << '\n';
            check(maximum_anchor_drift < 0.002F,
                  "hinge anchors must remain fixed through full rotations");
            check(minimum_axis_dot > 0.9999F,
                  "hinge axes must remain aligned through full rotations");
        }
    }
    return failures == 0 ? 0 : 1;
}
