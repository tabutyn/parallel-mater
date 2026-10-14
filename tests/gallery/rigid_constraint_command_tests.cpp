// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <scene_builder.hpp>
#include "rigid_mesh_checks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

int main() try {
    SceneDefinition scene;
    std::string error;
    check(load_glb_scene(PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH,
                         scene, error),
          error.empty() ? "load Hinge + Slider scene" : error.c_str());
    check(scene.rigid_bodies.size() == 5U &&
              scene.rigid_constraints.size() == 4U,
          "Hinge + Slider scene topology changed");

    const std::size_t rod_index = body_index(scene, "Ground.002");
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

    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    if (!CommandScene::create(scene, 2U, simulation, instance, error)) {
        if (error.find("physical device") != std::string::npos) {
            std::cout << "SKIP: no Parallel Mater device\n";
            return 77;
        }
        throw std::runtime_error(error);
    }

    float maximum_hinge_anchor_error = 0.0F;
    float maximum_hinge_axis_error = 0.0F;
    float maximum_slider_sideways_drift = 0.0F;
    float maximum_slider_axis_error = 0.0F;
    float maximum_slider_travel = 0.0F;
    float maximum_slider_mesh_penetration = 0.0F;
    float slider_contact_impulse = 0.0F;
    std::array<bool, 2U> saw_gear_contact{};
    std::array<bool, 2U> saw_opposite_rotation{};
    const rigid_mesh_checks::MeshSamples slider_mesh(scene, rod_index);
    const rigid_mesh_checks::MeshSamples slider_gear_mesh(
        scene, gear_indices[2]);
    for (std::uint32_t frame_index = 0U; frame_index < 240U; ++frame_index) {
        check(simulation.step(1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F},
                              frame, error, true),
              error.empty() ? "step Hinge + Slider scene" : error.c_str());
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
        if (frame_index % 4U == 0U) {
            const RigidBodyState rod = rigid_state(frame.rigid_states[rod_index]);
            const RigidBodyState gear =
                rigid_state(frame.rigid_states[gear_indices[2]]);
            maximum_slider_mesh_penetration = std::max({
                maximum_slider_mesh_penetration,
                slider_mesh.penetration(rod, slider_gear_mesh, gear),
                slider_gear_mesh.penetration(gear, slider_mesh, rod)});
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
            for (std::size_t pair = 0U; pair < saw_gear_contact.size(); ++pair)
                saw_gear_contact[pair] = saw_gear_contact[pair] ||
                    matches(gear_indices[pair], gear_indices[pair + 1U]);
        }
    }

    std::cout << "hinge_anchor_error=" << maximum_hinge_anchor_error
              << " hinge_axis_error=" << maximum_hinge_axis_error
              << " slider_sideways_drift=" << maximum_slider_sideways_drift
              << " slider_axis_error=" << maximum_slider_axis_error
              << " slider_travel=" << maximum_slider_travel
              << " slider_mesh_penetration="
              << maximum_slider_mesh_penetration
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
          "unlimited slider did not travel along its axis");
    check(maximum_slider_mesh_penetration < 0.002F,
          "slider and gear meshes interpenetrated by more than 2 mm");
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
