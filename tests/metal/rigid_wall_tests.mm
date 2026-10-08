// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
using namespace parallel_mater::metal;
using namespace parallel_mater::metal::gallery;

bool require(Status status) {
    if (status) return true;
    std::cerr << status.message << '\n';
    return false;
}

bool brick(const RigidBodyDefinition &body) {
    return body.source_name == "Layer1" || body.source_name == "Layer2";
}

float length(Vec3 value) {
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

Vec3 difference(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

bool read_states(World &world, const SceneInstance &instance,
                 std::vector<RigidBodyState> &states) {
    RigidBodyDeviceView view;
    if (!require(world.rigid_body_view(view)) || view.states.size != instance.rigid_bodies.size())
        return false;
    const auto contents = []<typename T>(BufferSpan<const T> span) -> const T * {
        if (span.size == 0U || span.buffer == nullptr) return nullptr;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)span.buffer;
        const std::uint64_t bytes = span.size * sizeof(T);
        if (buffer == nil || buffer.contents == nullptr ||
            span.byte_offset > buffer.length ||
            bytes > buffer.length - span.byte_offset) return nullptr;
        return reinterpret_cast<const T *>(
            static_cast<const std::uint8_t *>(buffer.contents) +
            span.byte_offset);
    };
    const RigidBodyId *ids = contents(view.ids);
    const RigidBodyState *device_states = contents(view.states);
    if (view.ids.size != view.states.size || ids == nullptr ||
        device_states == nullptr ||
        !std::equal(ids, ids + view.ids.size, instance.rigid_bodies.begin()))
        return false;
    states.assign(device_states, device_states + view.states.size);
    for (const auto &state : states) {
        if (!std::isfinite(length(state.position)) ||
            !std::isfinite(length(state.linear_velocity)) ||
            !std::isfinite(length(state.angular_velocity)) ||
            !std::isfinite(state.orientation.x) || !std::isfinite(state.orientation.y) ||
            !std::isfinite(state.orientation.z) || !std::isfinite(state.orientation.w))
            return false;
    }
    return true;
}

bool run_wall(const SceneDefinition &scene) {
    World world;
    SceneInstance instance;
    if (!require(create_scene_world(scene, world, instance))) return false;
    std::vector<RigidBodyState> states;
    // Touching faces must not invent margin-deep penetration or velocity.
    // The open floor retains its 1 mm rest skin.
    const StepOptions support_probe{.gravity = {},
                                    .collect_kernel_timings = true,
                                    .collect_rigid_contacts = true};
    if (!require(world.step(support_probe)) ||
        !read_states(world, instance, states)) return false;
    WorldStepTimings timings{};
    if (!require(world.collect_step_timings(timings)) || !timings.available ||
        timings.rigid_contact_evaluation.launch_count != 2U * support_probe.substeps ||
        timings.rigid_contact_solve.launch_count != 3U * support_probe.substeps) {
        std::cerr << "Ordinary-stack API timings omit dispatched kernels\n";
        return false;
    }
    const auto view = world.rigid_contacts();
    id<MTLBuffer> contact_buffer =
        (__bridge id<MTLBuffer>)view.events.buffer;
    if (view.event_count != 0U &&
        (contact_buffer == nil || contact_buffer.contents == nullptr)) return false;
    const auto *contact_data = view.event_count == 0U ? nullptr :
        reinterpret_cast<const RigidContactEvent *>(
            static_cast<const std::uint8_t *>(contact_buffer.contents) +
            view.events.byte_offset);
    std::vector<RigidContactEvent> contacts;
    if (contact_data != nullptr)
        contacts.assign(contact_data, contact_data + view.event_count);
    std::size_t bricks = 0;
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
        if (!brick(scene.rigid_bodies[i])) continue;
        ++bricks;
        const Vec3 initial = scene.rigid_bodies[i].options.initial_state.position;
        const bool supported = std::any_of(contacts.begin(), contacts.end(), [&](const auto &contact) {
            return (contact.body == instance.rigid_bodies[i] && contact.normal.y > 0.99F) ||
                   (contact.collider == instance.rigid_bodies[i] && contact.normal.y < -0.99F);
        });
        if (!supported || length(difference(states[i].position, initial)) > 0.0015F ||
            length(states[i].linear_velocity) > 1.0e-6F) {
            std::cerr << "Invalid first-frame support for brick " << i << '\n';
            return false;
        }
    }
    if (bricks != 384U || std::any_of(contacts.begin(), contacts.end(), [](const auto &contact) {
            return !std::isfinite(contact.penetration) || contact.penetration > 0.0015F;
        })) return false;

    // Begin gravity at the authored poses, not the already-projected probe.
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i)
        if (!require(world.set_rigid_body_state(instance.rigid_bodies[i],
                scene.rigid_bodies[i].options.initial_state))) return false;
    float maximum_drop = 0.0F, maximum_displacement = 0.0F;
    float maximum_angle = 0.0F, maximum_speed = 0.0F, resting_speed = 0.0F;
    float initial_energy = 0.0F, maximum_energy = 0.0F;
    float minimum_clearance = 1.0e30F;
    for (const auto &body : scene.rigid_bodies)
        if (brick(body)) initial_energy += body.options.mass * 9.81F *
            body.options.initial_state.position.y;
    constexpr unsigned settle_frames = 600U;
    for (unsigned frame = 0; frame < settle_frames; ++frame) {
        if (!require(world.step({})) || !read_states(world, instance, states)) return false;
        float energy = 0.0F;
        for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
            const auto &body = scene.rigid_bodies[i];
            if (!brick(body)) continue;
            const auto &state = states[i];
            const auto q = state.orientation;
            const float vertical_radius =
                0.4F * std::fabs(2.0F * (q.x*q.y + q.z*q.w)) +
                0.1F * std::fabs(1.0F - 2.0F * (q.x*q.x + q.z*q.z)) +
                0.2F * std::fabs(2.0F * (q.y*q.z - q.x*q.w));
            minimum_clearance = std::min(minimum_clearance, state.position.y - vertical_radius);
            const float speed = length(state.linear_velocity);
            const float angular_speed = length(state.angular_velocity);
            // Largest principal inertia bounds rotational energy from above.
            energy += body.options.mass * (9.81F * state.position.y +
                0.5F * speed * speed + (0.8F*0.8F + 0.4F*0.4F) / 24.0F *
                angular_speed * angular_speed);
            const Vec3 initial = body.options.initial_state.position;
            maximum_drop = std::max(maximum_drop, initial.y - states[i].position.y);
            maximum_displacement = std::max(maximum_displacement,
                length(difference(states[i].position, initial)));
            const auto initial_orientation = body.options.initial_state.orientation;
            const float orientation_dot = std::fabs(
                state.orientation.x * initial_orientation.x +
                state.orientation.y * initial_orientation.y +
                state.orientation.z * initial_orientation.z +
                state.orientation.w * initial_orientation.w);
            maximum_angle = std::max(maximum_angle, 2.0F * std::acos(
                std::clamp(orientation_dot, 0.0F, 1.0F)));
            maximum_speed = std::max(maximum_speed, length(states[i].linear_velocity));
            if (frame >= settle_frames - 120U)
                resting_speed = std::max(resting_speed, length(states[i].linear_velocity));
        }
        maximum_energy = std::max(maximum_energy, energy);
    }
    std::cout << "authored wall max_drop=" << maximum_drop
              << " max_displacement=" << maximum_displacement
              << " max_angle=" << maximum_angle
              << " peak_speed=" << maximum_speed << " resting_speed=" << resting_speed
              << " energy_ratio=" << maximum_energy / initial_energy
              << " clearance=" << minimum_clearance << '\n';
    if (maximum_drop > 0.002F || maximum_displacement > 0.005F ||
        maximum_angle > 0.01F ||
        maximum_speed > 0.12F || resting_speed > 0.02F ||
        maximum_energy > initial_energy * 1.01F || minimum_clearance < -0.002F) {
        std::cerr << "Brick wall lost static support under vertical gravity\n";
        return false;
    }

    // Arrow steering is intended to roll the ball, not push the authored wall
    // sideways. The wall remains dynamic and under vertical gravity, so an
    // actual impact can still displace individual bricks.
    std::vector<RigidBodyId> vertical_gravity_bodies;
    RigidBodyId ball{};
    bool found_ball = false;
    Vec3 initial_ball_position{};
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
        const auto &body = scene.rigid_bodies[i];
        if (brick(body)) {
            if (body.follows_gravity_tilt) {
                std::cerr << body.name << " unexpectedly follows gravity tilt\n";
                return false;
            }
            vertical_gravity_bodies.push_back(instance.rigid_bodies[i]);
        } else if (!body.follows_gravity_tilt) {
            std::cerr << body.name << " unexpectedly ignores gravity tilt\n";
            return false;
        }
        if (body.source_name == "Icosphere") {
            ball = instance.rigid_bodies[i];
            found_ball = true;
            initial_ball_position = body.options.initial_state.position;
        }
        if (!require(world.set_rigid_body_state(
                instance.rigid_bodies[i], body.options.initial_state))) return false;
    }
    if (vertical_gravity_bodies.size() != 384U || !found_ball) return false;
    constexpr float tilt = 0.5235987756F;
    const Vec3 tilted_gravity{
        9.81F * std::sin(tilt), -9.81F * std::cos(tilt), 0.0F};
    const Vec3 wall_compensation{
        -tilted_gravity.x, -9.81F - tilted_gravity.y, 0.0F};
    float tilted_wall_displacement = 0.0F;
    float tilted_wall_angle = 0.0F;
    for (unsigned frame = 0; frame < 600U; ++frame) {
        if (!require(world.apply_central_acceleration(
                {vertical_gravity_bodies.data(), vertical_gravity_bodies.size()},
                wall_compensation)) ||
            !require(world.step({.gravity = tilted_gravity})) ||
            !read_states(world, instance, states)) return false;
        for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
            const auto &body = scene.rigid_bodies[i];
            if (!brick(body)) continue;
            tilted_wall_displacement = std::max(tilted_wall_displacement,
                length(difference(states[i].position,
                                  body.options.initial_state.position)));
            const auto initial = body.options.initial_state.orientation;
            const float orientation_dot = std::fabs(
                states[i].orientation.x * initial.x +
                states[i].orientation.y * initial.y +
                states[i].orientation.z * initial.z +
                states[i].orientation.w * initial.w);
            tilted_wall_angle = std::max(tilted_wall_angle, 2.0F * std::acos(
                std::clamp(orientation_dot, 0.0F, 1.0F)));
        }
    }
    const auto ball_index = static_cast<std::size_t>(std::find(
        instance.rigid_bodies.begin(), instance.rigid_bodies.end(), ball) -
        instance.rigid_bodies.begin());
    std::cout << "tilted wall max_displacement=" << tilted_wall_displacement
              << " max_angle=" << tilted_wall_angle << '\n';
    if (tilted_wall_displacement > 0.005F || tilted_wall_angle > 0.01F ||
        ball_index == instance.rigid_bodies.size() ||
        states[ball_index].position.x - initial_ball_position.x < 0.25F) {
        std::cerr << "Gravity steering moved the wall instead of only the ball: "
                  << tilted_wall_displacement << " m, "
                  << tilted_wall_angle << " rad\n";
        return false;
    }

    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i)
        if (!require(world.set_rigid_body_state(instance.rigid_bodies[i],
                scene.rigid_bodies[i].options.initial_state))) return false;
    // The support diagnostic below measures the steady cached impulse, not
    // the deliberately stronger cold-start solve immediately after teleport.
    for (unsigned frame = 0; frame < 120U; ++frame)
        if (!require(world.step({}))) return false;

    // Batched diagnostics must include warm-started support, not just the
    // last incremental correction (which approaches zero at rest).
    const StepOptions diagnostic_step{.timestep = 1.0F / 60.0F, .substeps = 4U,
                                      .collect_rigid_contacts = true};
    if (!require(world.step(diagnostic_step)) || !read_states(world, instance, states)) return false;
    const auto support = world.rigid_contacts();
    contact_buffer = (__bridge id<MTLBuffer>)support.events.buffer;
    if (support.event_count != 0U &&
        (contact_buffer == nil || contact_buffer.contents == nullptr)) return false;
    contact_data = support.event_count == 0U ? nullptr :
        reinterpret_cast<const RigidContactEvent *>(
            static_cast<const std::uint8_t *>(contact_buffer.contents) +
            support.events.byte_offset);
    contacts.clear();
    if (contact_data != nullptr)
        contacts.assign(contact_data, contact_data + support.event_count);
    float expected_support = 0.0F, observed_support = 0.0F;
    RigidBodyId floor{};
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
        const auto &body = scene.rigid_bodies[i];
        if (body.source_name == "Ground") floor = instance.rigid_bodies[i];
        if (body.options.motion == MotionType::dynamic)
            expected_support += body.options.mass * 9.81F * diagnostic_step.timestep / diagnostic_step.substeps;
    }
    for (const auto &contact : contacts) {
        const float vertical = contact.normal.y * contact.normal_impulse + contact.friction_impulse.y;
        if (contact.collider == floor) observed_support += vertical;
        else if (contact.body == floor) observed_support -= vertical;
    }
    if (!std::isfinite(observed_support) || std::fabs(observed_support - expected_support) > 0.05F * expected_support) {
        std::cerr << "Contact diagnostics lost support impulse: " << observed_support
                  << " expected " << expected_support << '\n';
        return false;
    }

    // Stable does not mean frozen: a central top brick must respond to an
    // ordinary impulse independently of the rest of the wall.
    std::size_t target = 0U;
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i)
        if (brick(scene.rigid_bodies[i]) && std::fabs(states[i].position.x) < 0.8F &&
            states[i].position.y > states[target].position.y) target = i;
    const Vec3 before = states[target].position;
    if (!brick(scene.rigid_bodies[target]) ||
        !require(world.apply_impulse(instance.rigid_bodies[target], {0, 0, 200}, before)))
        return false;
    for (unsigned frame = 0; frame < 30; ++frame)
        if (!require(world.step({}))) return false;
    if (!read_states(world, instance, states) || states[target].position.z - before.z < 0.03F) {
        std::cerr << "Supported brick did not respond to impact\n";
        return false;
    }

    // Teleport the stack off the floor and stop gravity. Old support impulses
    // must not kick edited bodies. A changed timestep must also be safe.
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
        auto state = scene.rigid_bodies[i].options.initial_state;
        if (brick(scene.rigid_bodies[i])) state.position.y += 3.0F;
        if (!require(world.set_rigid_body_state(instance.rigid_bodies[i], state))) return false;
    }
    if (!require(world.step({.timestep = 1.0F / 120.0F, .substeps = 1U, .gravity = {}})) ||
        !read_states(world, instance, states)) return false;
    for (const auto &state : states)
        if (length(state.linear_velocity) > 1.0e-6F || length(state.angular_velocity) > 1.0e-6F)
            return false;
    return true;
}

bool run_sleeping_wall(const SceneDefinition &scene) {
    WorldOptions options{};
    if (!require(scene_world_options(scene, options))) return false;
    options.rigid_sleeping = true;
    World world;
    SceneInstance instance;
    if (!require(World::create(options, world)) ||
        !require(instantiate_scene(scene, world, instance))) return false;
    for (unsigned frame = 0; frame < 120U; ++frame)
        if (!require(world.step({}))) return false;
    WorldStatistics stats{};
    if (!require(world.collect_statistics(stats)) ||
        stats.sleeping_rigid_body_count != 385U) {
        std::cerr << "Quiet wall did not sleep\n";
        return false;
    }
    std::vector<RigidBodyId> vertical;
    std::size_t ball = 0U;
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
        if (brick(scene.rigid_bodies[i])) vertical.push_back(instance.rigid_bodies[i]);
        if (scene.rigid_bodies[i].source_name == "Icosphere") ball = i;
    }
    const Vec3 tilted{4.905F, -8.495709F, 0.0F};
    for (unsigned frame = 0; frame < 120U; ++frame) {
        if (!require(world.apply_central_acceleration(
                {vertical.data(), vertical.size()}, {-tilted.x, -9.81F - tilted.y, 0.0F})) ||
            !require(world.step({.gravity = tilted}))) return false;
    }
    std::vector<RigidBodyState> states;
    if (!read_states(world, instance, states) ||
        !require(world.collect_statistics(stats)) || stats.sleeping_rigid_body_count != 384U ||
        states[ball].position.x - scene.rigid_bodies[ball].options.initial_state.position.x < 0.25F) {
        std::cerr << "Gravity compensation woke the wall or froze the steered ball\n";
        return false;
    }
    std::size_t target = 0U;
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
        if (!brick(scene.rigid_bodies[i])) continue;
        if (length(difference(states[i].position,
                scene.rigid_bodies[i].options.initial_state.position)) > 0.005F) return false;
        if (std::fabs(states[i].position.x) < 0.8F &&
            states[i].position.y > states[target].position.y) target = i;
    }
    const Vec3 before = states[target].position;
    // Repeated changes smaller than the wake tolerance must accumulate
    // against the sleeping load, rather than drift forever unnoticed.
    for (unsigned frame = 1U; frame <= 12U; ++frame) {
        if (!require(world.apply_force(instance.rigid_bodies[target],
                {2.0e-6F * float(frame), 0.0F, 0.0F}, before)) ||
            !require(world.step({}))) return false;
    }
    if (!require(world.collect_statistics(stats)) ||
        stats.sleeping_rigid_body_count >= 384U) {
        std::cerr << "Slowly changing load failed to wake sleeping brick\n";
        return false;
    }
    if (!require(world.apply_impulse(instance.rigid_bodies[target], {0, 0, 20}, before)))
        return false;
    for (unsigned frame = 0; frame < 8U; ++frame)
        if (!require(world.step({}))) return false;
    if (!read_states(world, instance, states) || states[target].position.z - before.z < 0.03F) {
        std::cerr << "Sleeping brick failed to wake for impulse\n";
        return false;
    }
    if (!require(world.step({.gravity = {}})) ||
        !require(world.collect_statistics(stats)) || stats.sleeping_rigid_body_count != 0U) {
        std::cerr << "Changed net gravity failed to wake sleeping bodies\n";
        return false;
    }
    std::cout << "sleeping wall: compensated steering, impulse and gravity wake passed\n";
    return true;
}

bool run_impact_energy(const SceneDefinition &scene, bool sleeping) {
    WorldOptions options{};
    if (!require(scene_world_options(scene, options))) return false;
    options.rigid_sleeping = sleeping;
    World world;
    SceneInstance instance;
    if (!require(World::create(options, world)) ||
        !require(instantiate_scene(scene, world, instance))) return false;
    std::vector<RigidBodyState> states;
    for (unsigned frame = 0; frame < 120U; ++frame)
        if (!require(world.step({}))) return false;
    std::size_t ball = scene.rigid_bodies.size();
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i)
        if (scene.rigid_bodies[i].source_name == "Icosphere") ball = i;
    if (ball == scene.rigid_bodies.size()) return false;
    auto launched = scene.rigid_bodies[ball].options.initial_state;
    launched.position = {0.0F, 1.05F, 2.5F};
    launched.linear_velocity = {0.0F, 0.0F, -10.0F};
    if (!require(world.set_rigid_body_state(instance.rigid_bodies[ball], launched)) ||
        !read_states(world, instance, states)) return false;
    // No external work follows the launch. Translational kinetic energy plus
    // signed gravitational potential is a lower bound on total mechanical
    // energy. Keep the potential signed: bricks can fall off the finite floor.
    // Allow 10% numerical drift without accepting finite energy explosions.
    double available_energy = 0.0;
    for (std::size_t i = 0; i < states.size(); ++i) {
        const auto &body = scene.rigid_bodies[i].options;
        if (body.motion != MotionType::dynamic) continue;
        const double speed = length(states[i].linear_velocity);
        available_energy += body.mass *
            (9.81 * states[i].position.y + 0.5 * speed * speed);
    }
    double peak_kinetic = 0.0;
    double peak_mechanical = 0.0;
    float peak_speed = 0.0F;
    unsigned displaced = 0U;
    for (unsigned frame = 0; frame < 360U; ++frame) {
        if (!require(world.step({})) || !read_states(world, instance, states)) return false;
        double kinetic = 0.0;
        double potential = 0.0;
        displaced = 0U;
        for (std::size_t i = 0; i < states.size(); ++i) {
            const auto &body = scene.rigid_bodies[i];
            if (body.options.motion != MotionType::dynamic) continue;
            const double speed = length(states[i].linear_velocity);
            kinetic += 0.5 * body.options.mass * speed * speed;
            potential += body.options.mass * 9.81 * states[i].position.y;
            peak_speed = std::max(peak_speed, static_cast<float>(speed));
            if (brick(body) && std::hypot(
                    states[i].position.x - body.options.initial_state.position.x,
                    states[i].position.z - body.options.initial_state.position.z) > 0.05F)
                ++displaced;
        }
        peak_kinetic = std::max(peak_kinetic, kinetic);
        peak_mechanical = std::max(peak_mechanical, kinetic + potential);
        if (kinetic + potential > available_energy * 1.1) {
            std::cerr << "Impact injected energy: sleeping=" << sleeping
                      << " frame=" << frame << " kinetic=" << kinetic
                      << " potential=" << potential << " available=" << available_energy
                      << " peak_speed=" << peak_speed << '\n';
            return false;
        }
    }
    std::cout << "impact sleeping=" << sleeping
              << " peak_kinetic_ratio=" << peak_kinetic / available_energy
              << " peak_mechanical_ratio=" << peak_mechanical / available_energy
              << " peak_speed=" << peak_speed
              << " displaced_bricks=" << displaced << '\n';
    return displaced >= 16U;
}
} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        SceneDefinition scene;
        std::string error;
        if (!load_glb_scene(PARALLEL_MATER_RIGID_BODY_SCENE_PATH, scene, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        const bool impact_only = argc == 2 &&
            std::string_view(argv[1]) == "--impact-only";
        return (impact_only || (run_wall(scene) && run_sleeping_wall(scene))) &&
            run_impact_energy(scene, true) && run_impact_energy(scene, false)
            ? 0 : 1;
    }
}
