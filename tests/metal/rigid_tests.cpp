// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace parallel_mater::metal;

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

bool finite(Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

} // namespace

int main() {
    World world;
    WorldOptions world_options{};
    world_options.rigid_body_capacity = 4;
    world_options.triangle_mesh_capacity = 2;
    Status status = World::create(world_options, world);
    if (!require(status.ok(), status.message == nullptr ? "World creation failed"
                                                        : status.message)) {
        return 1;
    }

    const std::array<Vec3, 4> vertices{{
        {-0.5F, 0.0F, -0.5F},
        {0.5F, 0.0F, -0.5F},
        {0.0F, 0.0F, 0.5F},
        {0.0F, 1.0F, 0.0F},
    }};
    const std::array<std::uint32_t, 12> indices{
        0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
    TriangleMeshId mesh{};
    status = world.add_triangle_mesh(
        HostSpan<const Vec3>{vertices.data(), vertices.size()},
        HostSpan<const std::uint32_t>{indices.data(), indices.size()}, mesh);
    if (!require(status.ok(), status.message == nullptr ? "Mesh upload failed"
                                                        : status.message)) {
        return 1;
    }

    RigidBodyOptions body_options{};
    body_options.mesh = mesh;
    body_options.initial_state.position = {0.0F, 2.0F, 0.0F};
    body_options.linear_damping = 0.0F;
    body_options.angular_damping = 0.0F;
    RigidBodyId body{};
    status = world.add_rigid_body(body_options, body);
    if (!require(status.ok(), status.message == nullptr ? "Body add failed"
                                                        : status.message)) {
        return 1;
    }

    RigidBodyDeviceView view{};
    status = world.rigid_body_view(view);
    if (!require(status.ok() && view.ids.size == 1 && view.states.size == 1 &&
                     view.ids.buffer != nullptr && view.states.buffer != nullptr,
                 "Rigid Metal buffer view is invalid")) {
        return 1;
    }

    status = world.step({.timestep = 0.1F,
                         .substeps = 2,
                         .gravity = {0.0F, -9.81F, 0.0F}});
    if (!require(status.ok(), status.message == nullptr ? "Rigid step failed"
                                                        : status.message)) {
        return 1;
    }
    RigidBodyState state{};
    status = world.read_rigid_body_state(body, state);
    if (!require(status.ok() && state.position.y < 2.0F &&
                     state.linear_velocity.y < -0.9F &&
                     std::isfinite(state.orientation.w),
                 "Rigid integration did not apply gravity")) {
        return 1;
    }

    status = world.apply_impulse(body, {1.0F, 0.0F, 0.0F}, state.position);
    if (!require(status.ok(), "Applying a rigid impulse failed")) {
        return 1;
    }
    RigidBodyState after_impulse{};
    status = world.read_rigid_body_state(body, after_impulse);
    if (!require(status.ok() && after_impulse.linear_velocity.x == 0.0F,
                 "Rigid impulse became visible before stepping")) {
        return 1;
    }
    status = world.step({.timestep = 0.01F,
                         .substeps = 1,
                         .gravity = {0.0F, 0.0F, 0.0F}});
    status = status ? world.read_rigid_body_state(body, after_impulse) : status;
    if (!require(status.ok() && after_impulse.linear_velocity.x > 0.9F,
                 "Rigid impulse did not update velocity during stepping")) {
        return 1;
    }

    status = world.remove_rigid_body(body);
    if (!require(status.ok(), "Rigid body removal failed")) {
        return 1;
    }
    status = world.read_rigid_body_state(body, state);
    if (!require(status.code == StatusCode::invalid_handle,
                 "A removed rigid body handle did not become stale")) {
        return 1;
    }
    status = world.remove_triangle_mesh(mesh);
    if (!require(status.ok(), "Triangle mesh removal failed")) {
        return 1;
    }

    const std::array<Vec3, 4> floor_vertices{{
        {-4.0F, 0.0F, -4.0F},
        {4.0F, 0.0F, -4.0F},
        {4.0F, 0.0F, 4.0F},
        {-4.0F, 0.0F, 4.0F},
    }};
    const std::array<std::uint32_t, 6> floor_indices{0, 2, 1, 0, 3, 2};
    const std::array<Vec3, 8> box_vertices{{
        {-0.5F, 0.0F, -0.5F}, {0.5F, 0.0F, -0.5F},
        {0.5F, 0.0F, 0.5F},   {-0.5F, 0.0F, 0.5F},
        {-0.5F, 1.0F, -0.5F}, {0.5F, 1.0F, -0.5F},
        {0.5F, 1.0F, 0.5F},   {-0.5F, 1.0F, 0.5F},
    }};
    const std::array<std::uint32_t, 36> box_indices{
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0};

    World motion_world;
    status = World::create(
        {.rigid_body_capacity = 3U, .triangle_mesh_capacity = 1U},
        motion_world);
    TriangleMeshId motion_mesh{};
    status = status ? motion_world.add_triangle_mesh(
                          {box_vertices.data(), box_vertices.size()},
                          {box_indices.data(), box_indices.size()},
                          motion_mesh)
                    : status;
    RigidBodyId static_motion{}, kinematic_motion{}, damped_motion{};
    status = status ? motion_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = motion_mesh,
                           .initial_state = {
                               .position = {-100.0F, 0.0F, 0.0F},
                               .linear_velocity = {1.0F, 2.0F, 3.0F},
                               .angular_velocity = {3.0F, 2.0F, 1.0F}}},
                          static_motion)
                    : status;
    status = status ? motion_world.add_rigid_body(
                          {.motion = MotionType::kinematic,
                           .mesh = motion_mesh},
                          kinematic_motion)
                    : status;
    status = status ? motion_world.add_rigid_body(
                          {.mesh = motion_mesh,
                           .initial_state = {
                               .position = {100.0F, 0.0F, 0.0F},
                               .orientation = {
                                   0.0F, 0.0F, 0.70710678F, 0.70710678F},
                               .linear_velocity = {10.0F, 0.0F, 0.0F}},
                           .inertia_diagonal = {1.0F, 2.0F, 4.0F},
                           .linear_damping = 1.0F,
                           .angular_damping = 0.0F},
                          damped_motion)
                    : status;
    status = status ? motion_world.apply_impulse(
                          damped_motion, {0.0F, 0.0F, 1.0F},
                          {100.0F, 1.0F, 0.0F})
                    : status;
    const float half_root = std::sqrt(0.5F);
    status = status ? motion_world.set_kinematic_target(
                          kinematic_motion,
                          {.position = {4.0F, 0.0F, 0.0F},
                           .orientation = {
                               0.0F, 0.0F, half_root, half_root}})
                    : status;
    status = status ? motion_world.step(
                          {.timestep = 0.4F,
                           .substeps = 4U,
                           .gravity = {}})
                    : status;
    RigidBodyState static_result{}, kinematic_result{}, damped_result{};
    status = status ? motion_world.read_rigid_body_state(
                          static_motion, static_result)
                    : status;
    status = status ? motion_world.read_rigid_body_state(
                          kinematic_motion, kinematic_result)
                    : status;
    status = status ? motion_world.read_rigid_body_state(
                          damped_motion, damped_result)
                    : status;
    const float expected_damped_velocity =
        10.0F / std::pow(1.0F + 0.1F, 4.0F);
    if (!require(
            status.ok() && static_result.linear_velocity.x == 0.0F &&
                static_result.linear_velocity.y == 0.0F &&
                static_result.angular_velocity.x == 0.0F &&
                std::abs(kinematic_result.position.x - 4.0F) < 1.0e-5F &&
                std::abs(kinematic_result.linear_velocity.x - 10.0F) <
                    1.0e-4F &&
                kinematic_result.angular_velocity.z > 0.0F &&
                std::abs(damped_result.linear_velocity.x -
                         expected_damped_velocity) < 1.0e-4F &&
                std::abs(damped_result.angular_velocity.x - 0.5F) < 1.0e-4F,
            "Metal rigid motion integration diverged from CUDA semantics"))
        return 1;
    status = motion_world.step(
        {.timestep = 0.1F, .substeps = 1U, .gravity = {}});
    status = status ? motion_world.read_rigid_body_state(
                          kinematic_motion, kinematic_result)
                    : status;
    if (!require(status.ok() &&
                     std::abs(kinematic_result.position.x - 4.0F) < 1.0e-5F &&
                     kinematic_result.linear_velocity.x == 0.0F &&
                     kinematic_result.angular_velocity.z == 0.0F,
                 "Completed kinematic target retained stale velocity"))
        return 1;

    TriangleMeshId floor_mesh{};
    status = world.add_triangle_mesh(
        HostSpan<const Vec3>{floor_vertices.data(), floor_vertices.size()},
        HostSpan<const std::uint32_t>{floor_indices.data(),
                                      floor_indices.size()},
        floor_mesh);
    if (!require(status.ok(), "Floor mesh upload failed")) return 1;

    TriangleMeshId falling_mesh{};
    status = world.add_triangle_mesh(
        HostSpan<const Vec3>{vertices.data(), vertices.size()},
        HostSpan<const std::uint32_t>{indices.data(), indices.size()},
        falling_mesh);
    if (!require(status.ok(), "Falling mesh upload failed")) return 1;

    RigidBodyId floor{};
    status = world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = floor_mesh}, floor);
    if (!require(status.ok(), "Static floor add failed")) return 1;
    RigidBodyId falling{};
    status = world.add_rigid_body(
        {.mesh = falling_mesh,
         .initial_state = {.position = {0.0F, 1.0F, 0.0F}},
         .linear_damping = 0.0F,
         .angular_damping = 0.0F,
         .collision_margin = 0.01F},
        falling);
    if (!require(status.ok(), "Falling body add failed")) return 1;
    for (std::uint32_t frame = 0; frame < 120U; ++frame) {
        status = world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 4U,
                             .gravity = {0.0F, -9.81F, 0.0F}});
        if (!require(status.ok(), "Rigid contact step failed")) return 1;
    }
    status = world.read_rigid_body_state(falling, state);
    if (!require(status.ok() && state.position.y > -0.1F &&
                     std::isfinite(state.linear_velocity.y),
                 "Rigid mesh contact did not retain the falling body")) {
        return 1;
    }
    status = world.step({.timestep = 1.0F / 60.0F,
                         .substeps = 1U,
                         .gravity = {0.0F, -9.81F, 0.0F},
                         .collect_kernel_timings = true,
                         .collect_rigid_contacts = true});
    const RigidContactDeviceView rigid_contacts = world.rigid_contacts();
    if (!require(status.ok() && rigid_contacts.event_count > 0U &&
                     rigid_contacts.events.size == rigid_contacts.event_count,
                 "Requested rigid contact diagnostics are unavailable"))
        return 1;
    WorldStepTimings rigid_timings{};
    status = world.collect_step_timings(rigid_timings);
    if (!require(status.ok() && rigid_timings.available &&
                     rigid_timings.rigid_integration.launch_count == 1U &&
                     rigid_timings.rigid_world_bounds.launch_count == 1U &&
                     rigid_timings.rigid_pair_filter.launch_count == 1U &&
                     rigid_timings.rigid_pair_compaction.launch_count == 3U &&
                     rigid_timings.rigid_contact_evaluation.launch_count ==
                         1U &&
                     rigid_timings.rigid_contact_generation.launch_count ==
                         6U &&
                     rigid_timings.rigid_contact_solve.launch_count >= 1U &&
                     rigid_timings.rigid_input_clear.launch_count == 1U &&
                     rigid_timings.total_gpu_milliseconds > 0.0F,
                 "Metal rigid per-phase timings are unavailable"))
        return 1;
    status = world.step({.timestep = 1.0F / 60.0F,
                         .substeps = 1U,
                         .gravity = {},
                         .collect_rigid_contacts = false});
    if (!require(status.ok() && world.rigid_contacts().event_count == 0U,
                 "Unrequested rigid contacts remained visible"))
        return 1;
    status = world.collect_step_timings(rigid_timings);
    if (!require(status.ok() && !rigid_timings.available,
                 "Unprofiled rigid frame retained stale timings"))
        return 1;

    World debug_world;
    status = World::create(
        {.rigid_body_capacity = 2U,
         .triangle_mesh_capacity = 2U,
         .physics_debug = {.frame_capacity = 1U, .frame_stride = 1U}},
        debug_world);
    TriangleMeshId debug_floor_mesh{}, debug_falling_mesh{};
    status = status ? debug_world.add_triangle_mesh(
                          {floor_vertices.data(), floor_vertices.size()},
                          {floor_indices.data(), floor_indices.size()},
                          debug_floor_mesh)
                    : status;
    status = status ? debug_world.add_triangle_mesh(
                          {box_vertices.data(), box_vertices.size()},
                          {box_indices.data(), box_indices.size()},
                          debug_falling_mesh)
                    : status;
    RigidBodyId debug_floor{}, debug_falling{};
    status = status ? debug_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = debug_floor_mesh},
                          debug_floor)
                    : status;
    status = status ? debug_world.add_rigid_body(
                          {.mesh = debug_falling_mesh,
                           .initial_state = {
                               .position = {0.0F, 0.02F, 0.0F},
                               .linear_velocity = {0.0F, -1.0F, 0.0F}},
                           .linear_damping = 0.0F,
                           .angular_damping = 0.0F,
                           .maximum_linear_speed = 200.0F,
                           .collision_margin = 0.01F},
                          debug_falling)
                    : status;
    status = status ? debug_world.step(
                          {.timestep = 0.01F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    PhysicsDebugFrameView debug_view{};
    status = status ? debug_world.physics_debug_frame(debug_view) : status;
    // Collision margins are a search bound; a valid predictive contact may
    // retain a negative penetration until the surfaces reach rest offset.
    bool valid_manifold = status.ok() &&
                          debug_view.rigid_contacts.size >= 1U;
    bool positive_impulse = false;
    for (std::size_t contact_index = 0;
         valid_manifold && contact_index < debug_view.rigid_contacts.size;
         ++contact_index) {
        const RigidContactEvent &event =
            debug_view.rigid_contacts.data[contact_index];
        valid_manifold = event.body == debug_falling &&
                         event.collider == debug_floor &&
                         finite(event.position) && finite(event.normal) &&
                         std::isfinite(event.penetration) &&
                         std::isfinite(event.normal_impulse) &&
                         finite(event.friction_impulse);
        positive_impulse = positive_impulse || event.normal_impulse > 0.0F;
    }
    if (!require(valid_manifold && positive_impulse,
                 "Rigid BVH manifold contact payload is invalid"))
        return 1;

    const RigidBodyState swept_initial{
        .position = {0.0F, 1.5F, 0.0F},
        .linear_velocity = {0.0F, -120.0F, 0.0F}};
    RigidBodyState swept_reference{};
    for (std::uint32_t repetition = 0U; repetition < 10U; ++repetition) {
        status = debug_world.set_rigid_body_state(debug_falling,
                                                   swept_initial);
        status = status ? debug_world.step(
                              {.timestep = 1.0F / 60.0F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        RigidBodyState swept_result{};
        status = status ? debug_world.read_rigid_body_state(
                              debug_falling, swept_result)
                        : status;
        // The collision margin bounds the search; CUDA's rigid rest offset is
        // capped at one millimetre.
        if (!require(status.ok() && swept_result.position.y >= 0.0009F &&
                         swept_result.linear_velocity.y >= -1.0e-3F,
                     "Swept BVH contact did not stop a fast rigid body"))
            return 1;
        if (repetition == 0U) {
            swept_reference = swept_result;
        } else if (!require(
                       std::memcmp(&swept_reference, &swept_result,
                                   sizeof(swept_result)) == 0,
                       "Swept BVH contact replay was not byte-identical")) {
            return 1;
        }
    }

    World colored_world;
    status = World::create(
        {.rigid_body_capacity = 6U,
         .triangle_mesh_capacity = 2U,
         .contact_capacity = 128U},
        colored_world);
    TriangleMeshId colored_floor_mesh{}, colored_box_mesh{};
    status = status ? colored_world.add_triangle_mesh(
                          {floor_vertices.data(), floor_vertices.size()},
                          {floor_indices.data(), floor_indices.size()},
                          colored_floor_mesh)
                    : status;
    status = status ? colored_world.add_triangle_mesh(
                          {box_vertices.data(), box_vertices.size()},
                          {box_indices.data(), box_indices.size()},
                          colored_box_mesh)
                    : status;
    RigidBodyId colored_floor{};
    status = status ? colored_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = colored_floor_mesh},
                          colored_floor)
                    : status;
    std::array<RigidBodyId, 5> colored_bodies{};
    std::array<RigidBodyState, 5> colored_initial{};
    for (std::size_t index = 0; status.ok() && index < colored_bodies.size();
         ++index) {
        colored_initial[index].position = {
            -1.6F + static_cast<float>(index) * 0.8F, 0.005F, 0.0F};
        colored_initial[index].linear_velocity = {0.0F, -0.5F, 0.0F};
        status = colored_world.add_rigid_body(
            {.mesh = colored_box_mesh,
             .initial_state = colored_initial[index],
             .linear_damping = 0.0F,
             .angular_damping = 0.0F},
            colored_bodies[index]);
    }
    if (!require(status.ok(), "Colored rigid fixture creation failed"))
        return 1;
    std::array<RigidBodyState, 5> colored_reference{};
    for (std::uint32_t repetition = 0U; repetition < 10U; ++repetition) {
        for (std::size_t index = 0;
             status.ok() && index < colored_bodies.size(); ++index)
            status = colored_world.set_rigid_body_state(
                colored_bodies[index], colored_initial[index]);
        status = status ? colored_world.step(
                              {.timestep = 0.005F,
                               .substeps = 1U,
                               .gravity = {},
                               .collect_rigid_contacts = true})
                        : status;
        std::array<RigidBodyState, 5> colored_result{};
        for (std::size_t index = 0;
             status.ok() && index < colored_bodies.size(); ++index)
            status = colored_world.read_rigid_body_state(
                colored_bodies[index], colored_result[index]);
        bool colored_finite = status.ok() &&
                              colored_world.rigid_contacts().event_count >=
                                  colored_bodies.size();
        for (const RigidBodyState &result : colored_result)
            colored_finite = colored_finite && finite(result.position) &&
                             finite(result.linear_velocity) &&
                             finite(result.angular_velocity);
        if (!require(colored_finite,
                     "Parallel colored rigid contacts are invalid"))
            return 1;
        if (repetition == 0U) {
            colored_reference = colored_result;
        } else if (!require(
                       std::memcmp(colored_reference.data(),
                                   colored_result.data(),
                                   sizeof(colored_result)) == 0,
                       "Parallel colored rigid replay was not byte-identical")) {
            return 1;
        }
    }

    RigidBodyId anchor{};
    status = world.add_rigid_body(
        {.motion = MotionType::static_body,
         .mesh = falling_mesh,
         .initial_state = {.position = {2.0F, 1.0F, 0.0F}}},
        anchor);
    if (!require(status.ok(), "Constraint anchor body add failed")) return 1;
    RigidBodyId constrained{};
    status = world.add_rigid_body(
        {.mesh = falling_mesh,
         .initial_state = {.position = {2.0F, 2.0F, 0.0F}},
         .linear_damping = 0.0F,
         .angular_damping = 0.0F},
        constrained);
    if (!require(status.ok(), "Constrained body add failed")) return 1;
    RigidConstraintId joint{};
    status = world.add_rigid_constraint(
        {.type = RigidConstraintType::point,
         .body_a = anchor,
         .body_b = constrained,
         .local_anchor_a = {0.0F, 1.0F, 0.0F}},
        joint);
    if (!require(status.ok(), "Rigid constraint add failed")) return 1;
    for (std::uint32_t frame = 0; frame < 30U; ++frame) {
        status = world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 4U,
                             .gravity = {0.0F, -9.81F, 0.0F}});
        if (!require(status.ok(), "Rigid constraint step failed")) return 1;
    }
    status = world.read_rigid_body_state(constrained, state);
    if (!require(status.ok() && std::abs(state.position.y - 2.0F) < 0.08F,
                 "Point constraint did not preserve its anchor")) {
        return 1;
    }
    RigidConstraintState constraint_state{};
    status = world.read_rigid_constraint_state(joint, constraint_state);
    if (!require(status.ok() && constraint_state.enabled &&
                     !constraint_state.broken &&
                     std::isfinite(constraint_state.applied_impulse),
                 "Rigid constraint state is invalid")) {
        return 1;
    }
    status = world.remove_rigid_body(constrained);
    if (!require(status.code == StatusCode::invalid_argument,
                 "A body referenced by a constraint was removed")) {
        return 1;
    }
    status = world.update_rigid_constraint(
        joint,
        {.type = RigidConstraintType::motor,
         .body_a = anchor,
         .body_b = constrained,
         .local_anchor_a = {0.0F, 1.0F, 0.0F},
         .motor = {.linear_enabled = true,
                   .linear_target_velocity = 1.0F,
                   .linear_maximum_impulse = 0.25F}});
    if (!require(status.ok(), "Rigid motor update failed")) return 1;
    status = world.step({.timestep = 1.0F / 60.0F,
                         .substeps = 1U,
                         .gravity = {}});
    status = status ? world.read_rigid_body_state(constrained, state) : status;
    if (!require(status.ok() && state.linear_velocity.x > 0.1F,
                 "Rigid motor did not drive its frame axis"))
        return 1;
    status = world.update_rigid_constraint(
        joint,
        {.type = RigidConstraintType::point,
         .body_a = anchor,
         .body_b = constrained,
         .local_anchor_a = {0.0F, 4.0F, 0.0F},
         .breaking_impulse_threshold = 0.001F});
    if (!require(status.ok(), "Breakable constraint update failed")) return 1;
    status = world.step({.timestep = 1.0F / 60.0F,
                         .substeps = 1U,
                         .gravity = {}});
    status = status ? world.read_rigid_constraint_state(joint,
                                                        constraint_state)
                    : status;
    if (!require(status.ok() && constraint_state.broken &&
                     !constraint_state.enabled,
                 "Rigid constraint did not report breakage"))
        return 1;
    status = world.update_rigid_constraint(
        joint,
        {.type = RigidConstraintType::fixed,
         .body_a = anchor,
         .body_b = constrained,
         .local_anchor_a = {0.0F, 1.0F, 0.0F}});
    if (!require(status.ok(), "Rigid constraint update failed")) return 1;
    status = world.remove_rigid_constraint(joint);
    if (!require(status.ok(), "Rigid constraint removal failed")) return 1;
    status = world.read_rigid_constraint_state(joint, constraint_state);
    if (!require(status.code == StatusCode::invalid_handle,
                 "Removed constraint handle did not become stale")) {
        return 1;
    }

    const std::array<Vec3, 8> centered_box_vertices{{
        {-0.1F, -0.1F, -0.1F}, {0.1F, -0.1F, -0.1F},
        {0.1F, -0.1F, 0.1F},   {-0.1F, -0.1F, 0.1F},
        {-0.1F, 0.1F, -0.1F},  {0.1F, 0.1F, -0.1F},
        {0.1F, 0.1F, 0.1F},    {-0.1F, 0.1F, 0.1F},
    }};

    World two_sided_world;
    status = World::create(
        {.rigid_body_capacity = 3U, .triangle_mesh_capacity = 2U},
        two_sided_world);
    const std::array<Vec3, 3> open_vertices{{
        {-4.0F, 0.0F, -4.0F},
        {4.0F, 0.0F, -4.0F},
        {0.0F, 0.0F, 4.0F},
    }};
    const std::array<std::uint32_t, 3> open_indices{0U, 1U, 2U};
    TriangleMeshId open_mesh{}, probe_mesh{};
    status = status ? two_sided_world.add_triangle_mesh(
                          {open_vertices.data(), open_vertices.size()},
                          {open_indices.data(), open_indices.size()},
                          open_mesh)
                    : status;
    status = status ? two_sided_world.add_triangle_mesh(
                          {centered_box_vertices.data(),
                           centered_box_vertices.size()},
                          {box_indices.data(), box_indices.size()},
                          probe_mesh)
                    : status;
    RigidBodyId open_surface{}, front_probe{}, back_probe{};
    status = status ? two_sided_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = open_mesh},
                          open_surface)
                    : status;
    status = status ? two_sided_world.add_rigid_body(
                          {.mesh = probe_mesh,
                           .initial_state = {
                               .position = {-1.0F, 0.095F, 0.0F},
                               .linear_velocity = {0.0F, -1.0F, 0.0F}},
                           .linear_damping = 0.0F,
                           .angular_damping = 0.0F},
                          front_probe)
                    : status;
    status = status ? two_sided_world.add_rigid_body(
                          {.mesh = probe_mesh,
                           .initial_state = {
                               .position = {1.0F, -0.095F, 0.0F},
                               .linear_velocity = {0.0F, 1.0F, 0.0F}},
                           .linear_damping = 0.0F,
                           .angular_damping = 0.0F},
                          back_probe)
                    : status;
    status = status ? two_sided_world.step(
                          {.timestep = 0.01F,
                           .substeps = 2U,
                           .gravity = {}})
                    : status;
    RigidBodyState front_state{}, back_state{};
    status = status ? two_sided_world.read_rigid_body_state(
                          front_probe, front_state)
                    : status;
    status = status ? two_sided_world.read_rigid_body_state(
                          back_probe, back_state)
                    : status;
    if (!require(status.ok() && front_state.position.y >= 0.099F &&
                     front_state.linear_velocity.y >= -1.0e-3F,
                 "An open triangle did not collide from its front"))
        return 1;
    if (!require(back_state.position.y <= -0.099F &&
                     back_state.linear_velocity.y <= 1.0e-3F,
                 "An open triangle did not collide from its back"))
        return 1;

    World dynamic_pair_world;
    status = World::create(
        {.rigid_body_capacity = 2U, .triangle_mesh_capacity = 1U},
        dynamic_pair_world);
    const std::array<Vec3, 8> pair_vertices{{
        {-0.5F, -0.5F, -0.5F}, {0.5F, -0.5F, -0.5F},
        {0.5F, -0.5F, 0.5F},   {-0.5F, -0.5F, 0.5F},
        {-0.5F, 0.5F, -0.5F},  {0.5F, 0.5F, -0.5F},
        {0.5F, 0.5F, 0.5F},    {-0.5F, 0.5F, 0.5F},
    }};
    TriangleMeshId pair_mesh{};
    status = status ? dynamic_pair_world.add_triangle_mesh(
                          {pair_vertices.data(), pair_vertices.size()},
                          {box_indices.data(), box_indices.size()}, pair_mesh)
                    : status;
    RigidBodyId left_body{}, right_body{};
    status = status ? dynamic_pair_world.add_rigid_body(
                          {.mesh = pair_mesh,
                           .initial_state = {
                               .position = {-0.49F, 0.0F, 0.0F},
                               .linear_velocity = {1.0F, 0.0F, 0.0F}},
                           .restitution = 0.5F,
                           .linear_damping = 0.0F,
                           .angular_damping = 0.0F},
                          left_body)
                    : status;
    status = status ? dynamic_pair_world.add_rigid_body(
                          {.mesh = pair_mesh,
                           .initial_state = {
                               .position = {0.49F, 0.0F, 0.0F},
                               .linear_velocity = {-1.0F, 0.0F, 0.0F}},
                           .restitution = 0.5F,
                           .linear_damping = 0.0F,
                           .angular_damping = 0.0F},
                          right_body)
                    : status;
    status = status ? dynamic_pair_world.step(
                          {.timestep = 1.0F / 120.0F,
                           .substeps = 4U,
                           .gravity = {}})
                    : status;
    RigidBodyState left_state{}, right_state{};
    status = status ? dynamic_pair_world.read_rigid_body_state(
                          left_body, left_state)
                    : status;
    status = status ? dynamic_pair_world.read_rigid_body_state(
                          right_body, right_state)
                    : status;
    if (!require(status.ok() && left_state.linear_velocity.x < 1.0F &&
                     right_state.linear_velocity.x > -1.0F,
                 "Dynamic rigid bodies did not exchange collision impulse"))
        return 1;
    if (!require(std::abs(left_state.linear_velocity.x +
                          right_state.linear_velocity.x) < 1.0e-4F,
                 "Dynamic rigid collision did not preserve linear momentum"))
        return 1;

    World angular_world;
    status = World::create(
        {.rigid_body_capacity = 2U, .triangle_mesh_capacity = 2U},
        angular_world);
    const std::array<Vec3, 8> tall_vertices{{
        {-0.3F, -0.9F, -0.3F}, {0.3F, -0.9F, -0.3F},
        {0.3F, -0.9F, 0.3F},   {-0.3F, -0.9F, 0.3F},
        {-0.3F, 0.9F, -0.3F},  {0.3F, 0.9F, -0.3F},
        {0.3F, 0.9F, 0.3F},    {-0.3F, 0.9F, 0.3F},
    }};
    TriangleMeshId angular_floor_mesh{}, tall_mesh{};
    status = status ? angular_world.add_triangle_mesh(
                          {floor_vertices.data(), floor_vertices.size()},
                          {floor_indices.data(), floor_indices.size()},
                          angular_floor_mesh)
                    : status;
    status = status ? angular_world.add_triangle_mesh(
                          {tall_vertices.data(), tall_vertices.size()},
                          {box_indices.data(), box_indices.size()}, tall_mesh)
                    : status;
    RigidBodyId angular_floor{}, tall_body{};
    status = status ? angular_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = angular_floor_mesh,
                           .friction = 0.9F},
                          angular_floor)
                    : status;
    const RigidBodyState tall_initial{
        .position = {0.0F, 1.0F, 0.0F},
        .orientation = {
            0.0F, 0.0F, std::sin(0.125F), std::cos(0.125F)}};
    status = status ? angular_world.add_rigid_body(
                          {.mesh = tall_mesh,
                           .initial_state = tall_initial,
                           .friction = 0.8F,
                           .linear_damping = 0.08F,
                           .angular_damping = 0.25F},
                          tall_body)
                    : status;
    for (std::uint32_t frame = 0U; status.ok() && frame < 240U; ++frame)
        status = angular_world.step({.timestep = 1.0F / 60.0F,
                                     .substeps = 8U,
                                     .gravity = {0.0F, -9.81F, 0.0F}});
    RigidBodyState toppled_state{};
    status = status ? angular_world.read_rigid_body_state(
                          tall_body, toppled_state)
                    : status;
    if (!require(status.ok() &&
                     std::abs(toppled_state.orientation.z -
                              tall_initial.orientation.z) > 0.05F,
                 "Off-center rigid contact did not create angular motion"))
        return 1;

    RigidBodyState angular_reference{};
    for (std::uint32_t repetition = 0U; repetition < 10U; ++repetition) {
        status = angular_world.set_rigid_body_state(tall_body, tall_initial);
        for (std::uint32_t frame = 0U; status.ok() && frame < 20U; ++frame)
            status = angular_world.step({.timestep = 1.0F / 60.0F,
                                         .substeps = 4U,
                                         .gravity = {0.0F, -9.81F, 0.0F}});
        RigidBodyState angular_result{};
        status = status ? angular_world.read_rigid_body_state(
                              tall_body, angular_result)
                        : status;
        if (!require(status.ok(),
                     "Angular rigid deterministic replay failed to step"))
            return 1;
        if (repetition == 0U) {
            angular_reference = angular_result;
        } else if (!require(
                       std::memcmp(&angular_reference, &angular_result,
                                   sizeof(angular_result)) == 0,
                       "Angular rigid contact replay was not byte-identical")) {
            return 1;
        }
    }

    World overflow_world;
    status = World::create(
        {.rigid_body_capacity = 2U, .triangle_mesh_capacity = 2U},
        overflow_world);
    std::vector<std::uint32_t> duplicate_floor_indices;
    duplicate_floor_indices.reserve(516U * 3U);
    for (std::uint32_t index = 0U; index < 516U; ++index)
        duplicate_floor_indices.insert(duplicate_floor_indices.end(),
                                       {0U, 1U, 2U});
    const std::array<Vec3, 3> duplicate_floor_vertices{{
        {-5.0F, 0.0F, -5.0F},
        {0.0F, 0.0F, 5.0F},
        {5.0F, 0.0F, -5.0F},
    }};
    TriangleMeshId duplicate_floor_mesh{}, overflow_projectile_mesh{};
    status = status ? overflow_world.add_triangle_mesh(
                          {duplicate_floor_vertices.data(),
                           duplicate_floor_vertices.size()},
                          {duplicate_floor_indices.data(),
                           duplicate_floor_indices.size()},
                          duplicate_floor_mesh)
                    : status;
    status = status ? overflow_world.add_triangle_mesh(
                          {centered_box_vertices.data(),
                           centered_box_vertices.size()},
                          {box_indices.data(), box_indices.size()},
                          overflow_projectile_mesh)
                    : status;
    RigidBodyId duplicate_floor{}, overflow_projectile{};
    status = status ? overflow_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = duplicate_floor_mesh},
                          duplicate_floor)
                    : status;
    status = status ? overflow_world.add_rigid_body(
                          {.mesh = overflow_projectile_mesh,
                           .initial_state = {
                               .position = {0.0F, 1.5F, 0.0F},
                               .linear_velocity = {0.0F, -120.0F, 0.0F}},
                           .linear_damping = 0.0F,
                           .angular_damping = 0.0F,
                           .maximum_linear_speed = 200.0F},
                          overflow_projectile)
                    : status;
    status = status ? overflow_world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    RigidBodyState overflow_state{};
    status = status ? overflow_world.read_rigid_body_state(
                          overflow_projectile, overflow_state)
                    : status;
    if (!require(status.ok() && overflow_state.position.y >= 0.099F &&
                     overflow_state.linear_velocity.y >= -1.0F,
                 "BVH leaf-cache overflow lost swept rigid collision"))
        return 1;
    status = overflow_world.step({.timestep = 1.0F / 60.0F,
                                  .substeps = 1U,
                                  .gravity = {}});
    status = status ? overflow_world.read_rigid_body_state(
                          overflow_projectile, overflow_state)
                    : status;
    if (!require(status.ok() && overflow_state.position.y >= 0.099F,
                 "BVH overflow projectile did not remain above the surface"))
        return 1;

    constexpr std::uint32_t high_degree_body_count = 36U;
    World high_degree_world;
    status = World::create(
        {.rigid_body_capacity = high_degree_body_count,
         .triangle_mesh_capacity = 1U,
         .contact_capacity = 1U},
        high_degree_world);
    TriangleMeshId high_degree_mesh{};
    status = status ? high_degree_world.add_triangle_mesh(
                          {centered_box_vertices.data(),
                           centered_box_vertices.size()},
                          {box_indices.data(), box_indices.size()},
                          high_degree_mesh)
                    : status;
    std::vector<RigidBodyId> high_degree_bodies;
    high_degree_bodies.reserve(high_degree_body_count);
    for (std::uint32_t index = 0U;
         status.ok() && index < high_degree_body_count; ++index) {
        RigidBodyId high_degree_body{};
        status = high_degree_world.add_rigid_body(
            {.mesh = high_degree_mesh,
             .linear_damping = 0.0F,
             .angular_damping = 0.0F},
            high_degree_body);
        high_degree_bodies.push_back(high_degree_body);
    }
    RigidBodyState high_degree_reference{};
    for (std::uint32_t repetition = 0U; repetition < 2U; ++repetition) {
        for (RigidBodyId high_degree_body : high_degree_bodies)
            status = status ? high_degree_world.set_rigid_body_state(
                                  high_degree_body, {})
                            : status;
        status = status ? high_degree_world.step(
                              {.timestep = 1.0F / 240.0F,
                               .substeps = 1U,
                               .gravity = {},
                               .collect_rigid_contacts = repetition != 0U})
                        : status;
        RigidBodyState high_degree_result{};
        status = status ? high_degree_world.read_rigid_body_state(
                              high_degree_bodies.front(),
                              high_degree_result)
                        : status;
        if (!require(status.ok() && finite(high_degree_result.position) &&
                         finite(high_degree_result.linear_velocity) &&
                         finite(high_degree_result.angular_velocity),
                     "High-degree rigid contact graph became non-finite"))
            return 1;
        if (repetition == 0U) {
            high_degree_reference = high_degree_result;
        } else {
            if (!require(high_degree_world.rigid_contacts().event_count == 1U,
                         "Rigid contact diagnostics exceeded capacity"))
                return 1;
            if (!require(
                    std::memcmp(&high_degree_reference,
                                &high_degree_result,
                                sizeof(high_degree_result)) == 0,
                    "High-degree colored rigid solve was not deterministic"))
                return 1;
        }
    }
    return 0;
}
