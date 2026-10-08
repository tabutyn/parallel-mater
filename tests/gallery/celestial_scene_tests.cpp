// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace parallel_mater;
using namespace parallel_mater::gallery;

namespace {
void check(bool okay, const char *message) {
    if (!okay) throw std::runtime_error(message);
}
void require(Status status) {
    check(bool(status), status.message ? status.message : "physics operation failed");
}
Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
Vec3 subtract(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
float length(Vec3 v) { return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); }
Vec3 rotate(Quaternion q, Vec3 v) {
    const Vec3 t{2*(q.y*v.z-q.z*v.y), 2*(q.z*v.x-q.x*v.z), 2*(q.x*v.y-q.y*v.x)};
    return {v.x+q.w*t.x+q.y*t.z-q.z*t.y,
            v.y+q.w*t.y+q.z*t.x-q.x*t.z,
            v.z+q.w*t.z+q.x*t.y-q.y*t.x};
}
Vec3 anchor(const RigidBodyState &state, Vec3 local) {
    return add(state.position, rotate(state.orientation, local));
}
}

int main() try {
    SceneDefinition scene;
    std::string error;
    check(load_glb_scene(PARALLEL_MATER_CELESTIAL_SCENE_PATH, scene, error), error.c_str());
    check(scene.rigid_bodies.size() == 6 && scene.rigid_constraints.size() == 4,
          "Celestial needs four moving worlds, anchor, dais and four point joints");
    for (const auto &joint : scene.rigid_constraints) {
        check(joint.options.type == RigidConstraintType::point && joint.options.enabled,
              "every Celestial joint must start as an enabled point constraint");
        const auto &support = scene.rigid_bodies[joint.body_a];
        const auto &moving = scene.rigid_bodies[joint.body_b];
        check(support.source_name == "CelestialAnchor" &&
              support.options.motion == MotionType::static_body &&
              moving.options.motion == MotionType::dynamic &&
              !moving.collision_mesh_indices.empty(),
              "shared static anchor must carry four dynamic bodies with authored proxies");
        check(length(subtract(anchor(support.options.initial_state, joint.options.local_anchor_a),
                              Vec3{0, 6.3F, 0})) < 1.e-4F &&
              length(subtract(anchor(moving.options.initial_state, joint.options.local_anchor_b),
                              Vec3{0, 6.3F, 0})) < 1.e-4F,
              "all authored world-space anchor pairs must coincide at the solar core");
    }
    int count{};
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: Celestial metadata passed; CUDA unavailable\n";
        return 77;
    }
    World world;
    SceneInstance instance;
    require(create_scene_world(scene, world, instance));
    std::array<float, 4> travel{};
    float maximum_error{}, maximum_speed{}, release_error{};
    int elapsed_frame{};
    auto measure = [&](bool attached, bool enforce_anchor) {
        ++elapsed_frame;
        for (std::size_t i = 0; i < scene.rigid_constraints.size(); ++i) {
            const auto &joint = scene.rigid_constraints[i];
            RigidBodyState a{}, b{};
            require(world.read_rigid_body_state(instance.rigid_bodies[joint.body_a], a));
            require(world.read_rigid_body_state(instance.rigid_bodies[joint.body_b], b));
            if (!std::isfinite(length(b.position)) || length(b.position) >= 50 ||
                !std::isfinite(length(b.linear_velocity)) || length(b.linear_velocity) >= 100 ||
                !std::isfinite(length(b.angular_velocity)) || length(b.angular_velocity) >= 100)
                std::cerr << "frame=" << elapsed_frame << " body=" << scene.rigid_bodies[joint.body_b].name
                          << " position=" << length(b.position) << " speed=" << length(b.linear_velocity)
                          << " spin=" << length(b.angular_velocity) << '\n';
            check(std::isfinite(length(b.position)) && length(b.position) < 50 &&
                  std::isfinite(length(b.linear_velocity)) && length(b.linear_velocity) < 100 &&
                  std::isfinite(length(b.angular_velocity)) && length(b.angular_velocity) < 100,
                  "Celestial motion must remain finite and bounded");
            const float drift = length(subtract(anchor(a, joint.options.local_anchor_a),
                                               anchor(b, joint.options.local_anchor_b)));
            if (attached && enforce_anchor) {
                maximum_error = std::max(maximum_error, drift);
                check(drift < 0.06F, "point anchors drifted more than six centimetres");
            }
            if (!attached) release_error = std::max(release_error, drift);
            maximum_speed = std::max(maximum_speed, length(b.linear_velocity));
            travel[i] = std::max(travel[i], length(subtract(b.position,
                scene.rigid_bodies[joint.body_b].options.initial_state.position)));
            check(length(subtract(a.position, scene.rigid_bodies[joint.body_a].options.initial_state.position)) < 1.e-5F,
                  "central support must stay fixed");
        }
    };
    for (int frame = 0; frame < 600; ++frame) {
        // Same 60 Hz / four substeps as the gallery, including arrow tilt.
        const Vec3 gravity = frame < 360 ? Vec3{0, -9.81F, 0} : Vec3{4.905F, -8.49571F, 0};
        require(world.step({.timestep = 1.0F/60.0F, .substeps = 4U, .gravity = gravity}));
        measure(true, true);
    }
    check(std::all_of(travel.begin(), travel.end(), [](float v) { return v > 0.8F; }),
          "all four Celestial worlds must visibly move");
    auto toggle = [&](bool enabled) {
        if (enabled) {
            for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
                const auto &body = scene.rigid_bodies[i];
                if (body.options.motion == MotionType::dynamic)
                    require(world.set_rigid_body_state(instance.rigid_bodies[i], body.options.initial_state));
            }
        }
        for (std::size_t i = 0; i < scene.rigid_constraints.size(); ++i) {
            const auto &joint = scene.rigid_constraints[i];
            auto options = joint.options;
            options.body_a = instance.rigid_bodies[joint.body_a];
            options.body_b = instance.rigid_bodies[joint.body_b];
            options.enabled = enabled;
            require(world.update_rigid_constraint(instance.rigid_constraints[i], options));
        }
    };
    toggle(false);
    for (int frame = 0; frame < 60; ++frame) {
        require(world.step({.timestep = 1.0F/60.0F, .substeps = 4U}));
        measure(false, false);
    }
    check(release_error > 1, "Space release must free the bodies from the common pivot");
    toggle(true);
    for (int frame = 0; frame < 180; ++frame) {
        require(world.step({.timestep = 1.0F/60.0F, .substeps = 4U}));
        measure(true, true);
    }
    std::cout << "Celestial: 10s attached + tilt, 1s released, 3s restored; max anchor error="
              << maximum_error << "m, max speed=" << maximum_speed
              << "m/s, released anchor separation=" << release_error << "m\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
