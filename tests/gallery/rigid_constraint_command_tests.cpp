// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/physics_debug.hpp>
#include <scene_builder.hpp>
#include "rigid_mesh_checks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using namespace parallel_mater;
using namespace parallel_mater::gallery;

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

[[nodiscard]] Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 scale(Vec3 value, float amount) noexcept {
    return {value.x * amount, value.y * amount, value.z * amount};
}

[[nodiscard]] float dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

[[nodiscard]] float length(Vec3 value) noexcept {
    return std::sqrt(dot(value, value));
}

[[nodiscard]] Vec3 normalize_or(Vec3 value, Vec3 fallback) noexcept {
    const float magnitude = length(value);
    return magnitude > 1.0e-8F ? scale(value, 1.0F / magnitude) : fallback;
}

[[nodiscard]] Quaternion quaternion(const PmRigidState &state) noexcept {
    return {state.orientation[0], state.orientation[1],
            state.orientation[2], state.orientation[3]};
}

[[nodiscard]] Vec3 position(const PmRigidState &state) noexcept {
    return {state.position[0], state.position[1], state.position[2]};
}

[[nodiscard]] RigidBodyState rigid_state(const PmRigidState &state) noexcept {
    return {position(state), quaternion(state),
            {state.linearVelocity[0], state.linearVelocity[1],
             state.linearVelocity[2]},
            {state.angularVelocity[0], state.angularVelocity[1],
             state.angularVelocity[2]}};
}

[[nodiscard]] Quaternion multiply(Quaternion a, Quaternion b) noexcept {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

[[nodiscard]] Vec3 rotate(Quaternion orientation, Vec3 value) noexcept {
    const Quaternion result = multiply(
        multiply(orientation, {value.x, value.y, value.z, 0.0F}),
        {-orientation.x, -orientation.y, -orientation.z, orientation.w});
    return {result.x, result.y, result.z};
}

[[nodiscard]] Vec3 world_anchor(const PmRigidState &state,
                                Vec3 local_anchor) noexcept {
    return add(position(state), rotate(quaternion(state), local_anchor));
}

[[nodiscard]] Vec3 world_axis(const PmRigidState &state,
                              Quaternion local_orientation,
                              Vec3 local_axis) noexcept {
    return normalize_or(
        rotate(multiply(quaternion(state), local_orientation), local_axis),
        local_axis);
}

[[nodiscard]] std::size_t body_index(const SceneDefinition &scene,
                                     const char *name) {
    const auto found = std::find_if(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [&](const RigidBodyDefinition &body) {
            return body.source_name == name;
        });
    check(found != scene.rigid_bodies.end(), "constraint scene body is missing");
    return static_cast<std::size_t>(found - scene.rigid_bodies.begin());
}

} // namespace

int main(int argc, char **argv) try {
    enum class Scenario { settling, steering, full_rotation, replay };
    Scenario scenario = Scenario::settling;
    std::optional<PmNativeBackend> backend;
    unsigned frame_count = 2400U;
    bool frame_count_set = false;
    std::string scene_path = PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH;
    std::string capture_path = PARALLEL_MATER_HINGE_CAPTURE_PATH;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--backend" && argument + 1 < argc) {
            const std::string value = argv[++argument];
            if (value == "d3d12") backend = PM_NATIVE_BACKEND_D3D12;
            else if (value == "cuda") backend = PM_NATIVE_BACKEND_CUDA;
            else throw std::runtime_error("unknown backend");
        } else if (option == "--frames" && argument + 1 < argc) {
            frame_count = std::stoul(argv[++argument]);
            frame_count_set = true;
            check(frame_count != 0U, "frame count must be positive");
        } else if (option == "--scenario" && argument + 1 < argc) {
            const std::string value = argv[++argument];
            if (value == "settling") scenario = Scenario::settling;
            else if (value == "steering") scenario = Scenario::steering;
            else if (value == "full-rotation")
                scenario = Scenario::full_rotation;
            else if (value == "replay") scenario = Scenario::replay;
            else throw std::runtime_error("unknown scenario");
        } else if (option == "--scene" && argument + 1 < argc) {
            scene_path = argv[++argument];
        } else if (option == "--capture" && argument + 1 < argc) {
            capture_path = argv[++argument];
        } else throw std::runtime_error(
            "expected --backend d3d12|cuda, --frames N, or "
            "--scenario settling|steering|full-rotation|replay, "
            "--scene PATH, or --capture PATH");
    }
    const bool steering = scenario == Scenario::steering;
    const bool full_rotation = scenario == Scenario::full_rotation;
    const bool replay = scenario == Scenario::replay;
    if (replay) {
        if (!frame_count_set) frame_count = 112U * 4U;
    }
    SceneDefinition scene;
    std::string error;
    check(load_glb_scene(scene_path, scene, error),
          error.empty() ? "load Hinge + Slider scene" : error.c_str());
    const bool candidate_fixture = scene.rigid_bodies.size() == 5U;
    check((candidate_fixture || scene.rigid_bodies.size() == 6U) &&
              scene.rigid_constraints.size() == 5U,
          "Hinge + Slider scene topology changed");
    check(candidate_fixture || (!steering && !full_rotation && !replay),
          "steering and replay scenarios require the five-body functional fixture");

    const std::size_t rod_index = body_index(scene, "Ground.002");
    const std::size_t ground_index = body_index(scene, "Ground.001");
    const std::array<std::size_t, 3U> gear_indices{
        body_index(scene, "Gear"), body_index(scene, "Gear.001"),
        body_index(scene, "Gear.002")};
    const RigidBodyState &rod_initial =
        scene.rigid_bodies[rod_index].options.initial_state;
    const PmRigidState initial_rod{
        {rod_initial.position.x, rod_initial.position.y, rod_initial.position.z},
        {rod_initial.orientation.x, rod_initial.orientation.y,
         rod_initial.orientation.z, rod_initial.orientation.w},
        {}, {}};
    const auto driver_hinge = std::find_if(
        scene.rigid_constraints.begin(), scene.rigid_constraints.end(),
        [&](const RigidConstraintDefinition &constraint) {
            return constraint.options.type == RigidConstraintType::hinge &&
                (constraint.body_a == gear_indices[0] ||
                 constraint.body_b == gear_indices[0]);
        });
    check(driver_hinge != scene.rigid_constraints.end(),
          "driver gear hinge is missing");
    const Vec3 driver_local_anchor = driver_hinge->body_a == gear_indices[0]
        ? driver_hinge->options.local_anchor_a
        : driver_hinge->options.local_anchor_b;
    check(candidate_fixture ? length(driver_local_anchor) > 1.9F
                            : length(driver_local_anchor) < 1.0e-4F,
          candidate_fixture
              ? "driver paddle must offset the compound body's center from its axle"
              : "driver gear hinge must be centered on the gear");
    const RigidBodyState &driver_initial =
        scene.rigid_bodies[gear_indices[0]].options.initial_state;
    const std::uint32_t frame_index = driver_hinge->body_a == gear_indices[0]
        ? driver_hinge->body_b : driver_hinge->body_a;
    const Vec3 frame_local_anchor = driver_hinge->body_a == frame_index
        ? driver_hinge->options.local_anchor_a
        : driver_hinge->options.local_anchor_b;
    const RigidBodyState &frame_initial =
        scene.rigid_bodies[frame_index].options.initial_state;
    check(length(subtract(
              add(driver_initial.position,
                  rotate(driver_initial.orientation, driver_local_anchor)),
              add(frame_initial.position,
                  rotate(frame_initial.orientation, frame_local_anchor)))) < 1.0e-4F,
          "driver hinge must meet the gear axle");
    const auto slider = std::find_if(
        scene.rigid_constraints.begin(), scene.rigid_constraints.end(),
        [&](const RigidConstraintDefinition &constraint) {
            return constraint.options.type == RigidConstraintType::slider &&
                ((constraint.body_a == ground_index &&
                  constraint.body_b == rod_index) ||
                 (constraint.body_a == rod_index &&
                  constraint.body_b == ground_index));
        });
    check(slider != scene.rigid_constraints.end() &&
              slider->options.linear_limits.axes == 0U,
          "rack slider must retain free axial travel");
    const auto return_spring = std::find_if(
        scene.rigid_constraints.begin(), scene.rigid_constraints.end(),
        [&](const RigidConstraintDefinition &constraint) {
            return constraint.options.type ==
                    RigidConstraintType::generic_spring &&
                ((constraint.body_a == ground_index &&
                  constraint.body_b == rod_index) ||
                 (constraint.body_a == rod_index &&
                  constraint.body_b == ground_index));
        });
    check(return_spring != scene.rigid_constraints.end() &&
              return_spring->options.linear_springs.axes == 1U &&
              std::fabs(return_spring->options.linear_springs.stiffness.x -
                        (candidate_fixture ? 5.0F : 50.0F)) < 1.0e-4F &&
              std::fabs(return_spring->options.linear_springs.damping.x -
                        (candidate_fixture ? 0.5F : 14.0F)) < 1.0e-4F,
          "rack return spring configuration changed");
    check(!candidate_fixture ||
              std::fabs(scene.rigid_bodies[rod_index].options.friction - 0.08F)
                  < 1.0e-5F,
          "rack tooth friction must match the gears, not the passive frame");

    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    if (!CommandScene::create(
            scene, 2U, simulation, instance, error, backend)) {
        if (error.find("physical device") != std::string::npos) {
            std::cout << "SKIP: no Parallel Mater device\n";
            return 77;
        }
        throw std::runtime_error(error);
    }
    PhysicsDebugCapture capture;
    if (replay) {
        check(read_physics_debug_capture(capture_path, capture, error),
              error.empty() ? "read rigid replay capture" : error.c_str());
        check(capture.format_version == 5U && !capture.history.empty(),
              "replay requires a v5 capture with rigid history");
        check(replay_rigid_physics_sample(
                  simulation, instance, scene, capture.history.back(),
                  capture.provenance.rigid_body_ids,
                  PhysicsReplayPolicy::allow_partial, error),
              error.empty() ? "apply rigid replay capture" : error.c_str());
    }

    float maximum_hinge_anchor_error = 0.0F;
    float maximum_hinge_axis_error = 0.0F;
    float maximum_slider_sideways_drift = 0.0F;
    float maximum_slider_axis_error = 0.0F;
    float maximum_slider_travel = 0.0F;
    float minimum_slider_axial_overlap = 1.0e30F;
    float maximum_slider_mesh_penetration = 0.0F;
    float slider_contact_impulse = 0.0F;
    float maximum_gear_mesh_penetration = 0.0F;
    float maximum_locked_axis_contact_impulse = 0.0F;
    std::array<bool, 2U> saw_gear_contact{};
    std::array<bool, 2U> saw_opposite_rotation{};
    const rigid_mesh_checks::MeshSamples slider_mesh(scene, rod_index);
    const rigid_mesh_checks::MeshSamples slider_gear_mesh(
        scene, gear_indices[2]);
    const rigid_mesh_checks::MeshSamples driver_mesh(scene, gear_indices[0]);
    const rigid_mesh_checks::MeshSamples middle_mesh(scene, gear_indices[1]);
    CameraController camera;
    camera.set_preset(gallery_entry(GalleryContext::constraint_hinge).camera);
    const Vec3 driver_anchor = add(driver_initial.position,
        rotate(driver_initial.orientation, driver_local_anchor));
    const PmRigidState *replay_driver = nullptr;
    if (replay) {
        const auto found = std::find(
            capture.provenance.rigid_body_ids.begin(),
            capture.provenance.rigid_body_ids.end(), "Gear");
        check(found != capture.provenance.rigid_body_ids.end(),
              "replay capture omits driver gear");
        replay_driver = &capture.history.back().bodies[static_cast<std::size_t>(
            found - capture.provenance.rigid_body_ids.begin())];
    }
    Vec3 driver_arm = replay
        ? subtract(position(*replay_driver), driver_anchor)
        : subtract(driver_initial.position, driver_anchor);
    float driver_speed = replay
        ? replay_driver->angularVelocity[2]
        : driver_initial.angular_velocity.z;
    float previous_driver_angle = std::atan2(driver_arm.x, -driver_arm.y);
    float driver_rotation = 0.0F;
    float maximum_driver_rotation = 0.0F;
    float maximum_abrupt_speed_loss = 0.0F;
    bool cleared_top = false;
    for (std::uint32_t frame_number = 0U;
         frame_number < frame_count; ++frame_number) {
        if (frame_number % 300U == 0U)
            std::cout << "constraint fixture "
                      << (steering ? "steering" :
                          full_rotation ? "full-rotation" :
                          replay ? "replay" : "settling")
                      << " frame " << frame_number << '\n' << std::flush;
        // Normal gallery arrows, reversed every 1.5 seconds, exposed tooth
        // tunnelling that a stationary-gravity settling run did not exercise.
        float right = steering
            ? ((frame_number / 90U) % 2U == 0U ? 1.0F : -1.0F) : 0.0F;
        // Pump the functional paddle with the same arrows/gravity as the
        // gallery. No prescribed pose, motor, initial spin, or detached rack.
        // Fixed-period rocking alone can pass even when drag prevents a turn.
        if (full_rotation)
            right = std::fabs(driver_speed) < 1.0e-4F ||
                -driver_arm.y * driver_speed >= 0.0F ? 1.0F : -1.0F;
        Vec3 gravity = screen_space_gravity(
            camera.camera(), right, 0.0F, 9.81F, 30.0F);
        if (replay)
            gravity = frame_number < 90U * 4U
                ? Vec3{capture.history.back().step.gravity[0],
                       capture.history.back().step.gravity[1],
                       capture.history.back().step.gravity[2]}
                : Vec3{0.0F, -9.81000042F, 0.0F};
        check(simulation.step(
                  replay ? capture.history.back().step.timestep : 1.0F / 60.0F,
                  replay ? capture.history.back().step.substeps : 1U,
                  gravity, frame, error, true),
              error.empty() ? "step Hinge + Slider scene" : error.c_str());
        const auto &driver_state = frame.rigid_states[gear_indices[0]];
        driver_arm = subtract(position(driver_state), driver_anchor);
        if (replay)
            maximum_abrupt_speed_loss = std::max(
                maximum_abrupt_speed_loss,
                driver_state.angularVelocity[2] - driver_speed);
        driver_speed = driver_state.angularVelocity[2];
        cleared_top = cleared_top || (driver_arm.x > 0.5F &&
            driver_arm.y > 0.0F && driver_speed < -0.1F);
        const float driver_angle = std::atan2(driver_arm.x, -driver_arm.y);
        constexpr float full_turn = 6.2831853071795864769F;
        driver_rotation += std::remainder(
            driver_angle - previous_driver_angle, full_turn);
        previous_driver_angle = driver_angle;
        maximum_driver_rotation = std::max(
            maximum_driver_rotation, std::fabs(driver_rotation));
        for (const RigidConstraintDefinition &constraint :
             scene.rigid_constraints) {
            const PmRigidState &a = frame.rigid_states[constraint.body_a];
            const PmRigidState &b = frame.rigid_states[constraint.body_b];
            const Vec3 anchor_error = subtract(
                world_anchor(b, constraint.options.local_anchor_b),
                world_anchor(a, constraint.options.local_anchor_a));
            if (constraint.options.type == RigidConstraintType::hinge) {
                maximum_hinge_anchor_error = std::max(
                    maximum_hinge_anchor_error, length(anchor_error));
                const Vec3 axis_a = world_axis(
                    a, constraint.options.local_orientation_a,
                    {0.0F, 0.0F, 1.0F});
                const Vec3 axis_b = world_axis(
                    b, constraint.options.local_orientation_b,
                    {0.0F, 0.0F, 1.0F});
                maximum_hinge_axis_error = std::max(
                    maximum_hinge_axis_error,
                    length(cross(axis_a, axis_b)));
            } else if (constraint.options.type == RigidConstraintType::slider) {
                const Vec3 axis_a = world_axis(
                    a, constraint.options.local_orientation_a,
                    {1.0F, 0.0F, 0.0F});
                const Vec3 axis_b = world_axis(
                    b, constraint.options.local_orientation_b,
                    {1.0F, 0.0F, 0.0F});
                const Vec3 sideways = subtract(
                    anchor_error, scale(axis_a, dot(anchor_error, axis_a)));
                maximum_slider_sideways_drift = std::max(
                    maximum_slider_sideways_drift, length(sideways));
                maximum_slider_axis_error = std::max(
                    maximum_slider_axis_error, length(cross(axis_a, axis_b)));
            }
        }
        const Vec3 rod_delta = subtract(position(frame.rigid_states[rod_index]),
                                        position(initial_rod));
        maximum_slider_travel = std::max(maximum_slider_travel,
                                         length(rod_delta));
        if (frame_number % 4U == 0U) {
            const RigidBodyState rod = rigid_state(frame.rigid_states[rod_index]);
            const RigidBodyState gear =
                rigid_state(frame.rigid_states[gear_indices[2]]);
            const auto rack_bounds = slider_mesh.projected_bounds(
                rod, {0.0F, 1.0F, 0.0F});
            const auto gear_bounds = slider_gear_mesh.projected_bounds(
                gear, {0.0F, 1.0F, 0.0F});
            minimum_slider_axial_overlap = std::min(
                minimum_slider_axial_overlap,
                std::min(rack_bounds[1], gear_bounds[1]) -
                    std::max(rack_bounds[0], gear_bounds[0]));
            maximum_slider_mesh_penetration = std::max({
                maximum_slider_mesh_penetration,
                slider_mesh.penetration(rod, slider_gear_mesh, gear),
                slider_gear_mesh.penetration(gear, slider_mesh, rod)});
            if (steering || full_rotation || replay) {
                const auto driver = rigid_state(frame.rigid_states[gear_indices[0]]);
                const auto middle = rigid_state(frame.rigid_states[gear_indices[1]]);
                maximum_gear_mesh_penetration = std::max({
                    maximum_gear_mesh_penetration,
                    driver_mesh.penetration(driver, middle_mesh, middle),
                    middle_mesh.penetration(middle, driver_mesh, driver),
                    middle_mesh.penetration(middle, slider_gear_mesh, gear),
                    slider_gear_mesh.penetration(gear, middle_mesh, middle)});
            }
        }
        for (std::size_t pair = 0U; pair < saw_opposite_rotation.size(); ++pair)
            saw_opposite_rotation[pair] = saw_opposite_rotation[pair] ||
                frame.rigid_states[gear_indices[pair]].angularVelocity[2] *
                    frame.rigid_states[gear_indices[pair + 1U]]
                        .angularVelocity[2] < -1.0e-4F;
        for (const PmContactEvent &contact : frame.contacts) {
            const auto matches = [&](std::size_t first, std::size_t second) {
                return (contact.first == first && contact.second == second) ||
                    (contact.first == second && contact.second == first);
            };
            if (matches(rod_index, gear_indices[2]))
                slider_contact_impulse += contact.normalImpulse;
            for (std::size_t pair = 0U; pair < saw_gear_contact.size(); ++pair) {
                if (!matches(gear_indices[pair], gear_indices[pair + 1U]))
                    continue;
                saw_gear_contact[pair] = true;
                if (std::fabs(contact.normal.z) > 0.9999999F)
                    maximum_locked_axis_contact_impulse = std::max(
                        maximum_locked_axis_contact_impulse,
                        contact.normalImpulse);
            }
        }
        if (!candidate_fixture) {
            // Stop the accepted-fixture regression at its first physical
            // failure instead of evolving an already invalid trajectory.
            check(maximum_hinge_anchor_error < 0.002F,
                  "hinge anchors drifted by more than 2 mm");
            check(maximum_hinge_axis_error < 0.014F,
                  "hinge axes lost alignment");
            check(maximum_slider_sideways_drift < 0.002F,
                  "slider drifted sideways by more than 2 mm");
            check(maximum_slider_axis_error < 1.0e-3F,
                  "slider axis lost alignment");
            check(maximum_slider_travel < 1.0F,
                  "return spring did not bound slider travel");
            check(maximum_slider_mesh_penetration < 0.002F,
                  "slider and gear meshes interpenetrated by more than 2 mm");
        }
    }

    std::cout << "fixture=" << (candidate_fixture ? "candidate" : "current")
              << " frames=" << frame_count
              << " hinge_anchor_error=" << maximum_hinge_anchor_error
              << " hinge_axis_error=" << maximum_hinge_axis_error
              << " slider_sideways_drift=" << maximum_slider_sideways_drift
              << " slider_axis_error=" << maximum_slider_axis_error
              << " slider_travel=" << maximum_slider_travel
              << " slider_axial_overlap=" << minimum_slider_axial_overlap
              << " slider_mesh_penetration="
              << maximum_slider_mesh_penetration
              << " gear_mesh_penetration=" << maximum_gear_mesh_penetration
              << " locked_axis_contact_impulse="
              << maximum_locked_axis_contact_impulse
              << " driver_rotation=" << maximum_driver_rotation
              << " abrupt_speed_loss=" << maximum_abrupt_speed_loss
              << " cleared_top=" << cleared_top
              << " slider_contact_impulse=" << slider_contact_impulse << '\n';
    check(maximum_hinge_anchor_error < 0.002F,
          "hinge anchors drifted by more than 2 mm");
    check(maximum_hinge_axis_error < 0.014F,
          "hinge axes lost alignment");
    check(maximum_slider_sideways_drift < 0.002F,
          "slider drifted sideways by more than 2 mm");
    check(maximum_slider_axis_error < 1.0e-3F,
          "slider axis lost alignment");
    check(maximum_slider_travel > 0.01F,
          "slider did not travel along its axis");
    if (candidate_fixture) {
        // A complete driver turn moves this rack about pi metres. Check that
        // the teeth remain within reach, rather than imposing a one-metre cap.
        check(minimum_slider_axial_overlap > 0.0F,
              "rack travelled beyond the follower gear's reach");
    } else {
        check(maximum_slider_travel < 1.0F,
              "return spring did not bound slider travel");
    }
    if (steering)
        check(maximum_driver_rotation > 0.5F,
              "alternating gallery input did not move the driver paddle");
    if (full_rotation)
        check(maximum_driver_rotation > 6.2831853071795864769F,
              "normal gravity controls could not produce a complete turn");
    if (replay) {
        check(maximum_abrupt_speed_loss < 0.8F,
              "replayed tooth-edge contact abruptly removed paddle momentum");
        check(cleared_top && driver_rotation < -1.5F,
              "replayed input did not carry the paddle over the top");
    }
    check(maximum_slider_mesh_penetration < 0.002F,
          "slider and gear meshes interpenetrated by more than 2 mm");
    check(maximum_gear_mesh_penetration < 0.002F,
          "gear meshes interpenetrated by more than 2 mm");
    check(maximum_locked_axis_contact_impulse < 1.0e-5F,
          "gear contact pushed along the hinges' locked axle direction");
    check(slider_contact_impulse > 0.1F,
          "slider and gear did not exchange contact impulse");
    check(std::all_of(saw_gear_contact.begin(), saw_gear_contact.end(),
                      [](bool value) { return value; }),
          "both gear interfaces must generate contacts");
    check(std::all_of(saw_opposite_rotation.begin(),
                      saw_opposite_rotation.end(),
                      [](bool value) { return value; }),
          "gear contacts must transfer alternating rotation");
    return 0;
} catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
}
