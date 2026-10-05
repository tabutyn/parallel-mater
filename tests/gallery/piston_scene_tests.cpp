// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace parallel_mater;
using namespace parallel_mater::gallery;
constexpr float pi = 3.14159265358979323846F;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
void okay(Status status) {
    require(bool(status), status.message ? status.message : "physics operation failed");
}
Quaternion conjugate(Quaternion q) { return {-q.x, -q.y, -q.z, q.w}; }
Quaternion multiply(Quaternion a, Quaternion b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
            a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
            a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
            a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}
Vec3 rotate(Quaternion q, Vec3 v) {
    const auto result = multiply(multiply(q, {v.x,v.y,v.z,0}), conjugate(q));
    return {result.x,result.y,result.z};
}
float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 subtract(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
float length(Vec3 v) { return std::sqrt(dot(v,v)); }
float kinetic(const RigidBodyState &state, Vec3 inertia, float mass) {
    const Vec3 spin = rotate(conjugate(state.orientation), state.angular_velocity);
    return 0.5F * (mass * dot(state.linear_velocity, state.linear_velocity) +
        inertia.x*spin.x*spin.x+inertia.y*spin.y*spin.y+inertia.z*spin.z*spin.z);
}
Vec3 inertia_of(const SceneDefinition &scene, const RigidBodyDefinition &body) {
    Vec3 low{1.e30F,1.e30F,1.e30F}, high{-1.e30F,-1.e30F,-1.e30F};
    const auto &meshes = body.collision_mesh_indices.empty() ? scene.meshes : scene.collision_meshes;
    const auto &indices = body.collision_mesh_indices.empty() ? body.mesh_indices : body.collision_mesh_indices;
    for (auto index : indices) for (const auto &vertex : meshes[index].vertices) {
        const auto v = vertex.position;
        low = {std::min(low.x,v.x),std::min(low.y,v.y),std::min(low.z,v.z)};
        high = {std::max(high.x,v.x),std::max(high.y,v.y),std::max(high.z,v.z)};
    }
    const auto size = subtract(high, low);
    const float scale = body.options.mass / 12.0F;
    return {scale*(size.y*size.y+size.z*size.z),
            scale*(size.x*size.x+size.z*size.z),
            scale*(size.x*size.x+size.y*size.y)};
}

struct Fixture {
    SceneDefinition scene;
    SceneInstance instance;
    World world;
    std::size_t moving{};
    std::vector<double> step_milliseconds;
    Fixture() {
        std::string error;
        require(load_glb_scene(PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH, scene, error), error);
        require(scene.rigid_bodies.size() == 3 && scene.rigid_constraints.size() == 1,
                "piston fixture must contain shaft, sleeve, stationary teeth, and one joint");
        moving = scene.rigid_constraints.front().body_b;
        require(scene.rigid_bodies[moving].source_name == "Circle", "resolve moving sleeve");
        const auto &joint = scene.rigid_constraints.front().options;
        require(joint.type == RigidConstraintType::piston && joint.linear_limits.axes == 0 &&
                joint.angular_limits.axes == 0 && !joint.motor.linear_enabled &&
                !joint.motor.angular_enabled, "tooth contacts alone must index the piston");
        WorldOptions options;
        okay(scene_world_options(scene, options));
        okay(World::create(options, world));
        okay(instantiate_scene(scene, world, instance));
    }
    RigidBodyState state() {
        RigidBodyState result;
        okay(world.read_rigid_body_state(instance.rigid_bodies[moving], result));
        return result;
    }
    void step(Vec3 gravity) {
        const auto start = std::chrono::steady_clock::now();
        okay(world.step({.timestep = 1.0F/60.0F, .substeps = 4U, .gravity = gravity,
                         .collect_rigid_contacts = true}));
        step_milliseconds.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-start).count());
    }
};

void test_neutral_gravity_release() {
    Fixture fixture;
    // Seated pose from the user's release-kick capture. Start at rest so any
    // subsequent axial momentum is solver-generated, not residual inertia.
    RigidBodyState seated{.position = {0,0,-0.440304011F},
        .orientation = {0.261438668F,0.657000542F,0.657000601F,0.261438817F}};
    okay(fixture.world.set_rigid_body_state(fixture.instance.rigid_bodies[fixture.moving], seated));
    float maximum_speed = 0, maximum_drift = 0;
    for (int frame = 0; frame < 360; ++frame) {
        fixture.step({0,-9.81F,0});
        const auto state = fixture.state();
        maximum_speed = std::max(maximum_speed, length(state.linear_velocity));
        maximum_drift = std::max(maximum_drift, length(subtract(state.position, seated.position)));
        require(std::isfinite(state.position.z), "release state must remain finite");
    }
    require(maximum_speed < 1.e-4F, "perpendicular gravity must not kick a seated piston");
    require(maximum_drift < 0.002F, "release must retain the seated pose within contact clearance");
    std::cout << "Neutral release: peak speed " << maximum_speed
              << " m/s, drift " << maximum_drift << " m\n";
}

void test_eight_right_left_cycles(bool gallery_controls) {
    Fixture fixture;
    const auto &body = fixture.scene.rigid_bodies[fixture.moving];
    const auto inertia = inertia_of(fixture.scene, body);
    // Side-on camera makes screen right point down the authored shaft (-Z).
    // Use the same screen-space gravity mapping and 30-degree tilt as gallery.
    CameraController controller;
    const Camera camera = gallery_controls ? controller.camera() : Camera{{5,3,0},{0,0,0}};
    Vec3 gravity{0,-9.81F,0};
    // Also measure a complete turn from a seated endpoint, independent of the
    // authored pose's small initial angular offset. The abrupt test above
    // still covers startup directly from the unmodified GLB.
    if (gallery_controls) {
        for (int frame = 0; frame < 240; ++frame) {
            gravity = steer_gravity(gravity, camera, -1, 0, 9.81F, 30.0F, 1.0F/60.0F);
            fixture.step(gravity);
        }
        require(length(fixture.state().linear_velocity) < 0.02F,
                "initial left stroke must settle before measuring seated cycles");
    }
    const auto initial = fixture.state();
    std::cout << (gallery_controls ? "Gallery camera, smoothed input, seated start\n"
                                  : "Side camera, abrupt input, authored start\n");
    double total_angle = 0;
    float previous_angle = 0, maximum_swing = 0, maximum_drift = 0;
    float maximum_energy_gain = 0;
    float previous_cycle_angle = 0;
    auto previous = initial;
    for (int stroke = 0; stroke < 16; ++stroke) {
        const float direction = stroke % 2 == 0 ? 1.0F : -1.0F;
        const float starting_z = previous.position.z;
        RigidBodyState settle_start{};
        float settle_angle = 0;
        float settle_speed = 0, settle_spin = 0;
        for (int frame = 0; frame < 240; ++frame) {
            gravity = gallery_controls
                ? steer_gravity(gravity, camera, direction, 0, 9.81F, 30.0F, 1.0F/60.0F)
                : screen_space_gravity(camera, direction, 0, 9.81F, 30.0F);
            const float before_energy = kinetic(previous, inertia, body.options.mass) -
                body.options.mass * dot(gravity, previous.position);
            fixture.step(gravity);
            const auto state = fixture.state();
            const auto relative = multiply(state.orientation, conjugate(initial.orientation));
            const float angle = 2.0F * std::atan2(relative.z, relative.w);
            total_angle += std::remainder(angle - previous_angle, 2.0F*pi);
            previous_angle = angle;
            const float swing = 2.0F * std::asin(std::min(1.0F, std::hypot(relative.x,relative.y)));
            const float drift = std::hypot(state.position.x-initial.position.x,
                                           state.position.y-initial.position.y);
            maximum_swing = std::max(maximum_swing, swing);
            maximum_drift = std::max(maximum_drift, drift);
            require(std::isfinite(total_angle) && std::isfinite(length(state.linear_velocity)) &&
                    std::isfinite(length(state.angular_velocity)), "piston state must stay finite");
            require(swing < 0.002F && drift < 0.002F,
                    "piston must stay on its guide through every half-turn");
            const float energy = kinetic(state, inertia, body.options.mass) -
                body.options.mass * dot(gravity, state.position);
            maximum_energy_gain = std::max(maximum_energy_gain, energy-before_energy);
            require(energy <= before_energy + 0.005F,
                    "contact energy increase at stroke " + std::to_string(stroke+1) +
                    ", frame " + std::to_string(frame+1) + ": " +
                    std::to_string(energy-before_energy) + " J (limit 0.005 J)");
            if (frame == 209) { settle_start = state; settle_angle = total_angle; }
            if (frame >= 210) {
                settle_speed = std::max(settle_speed, length(state.linear_velocity));
                settle_spin = std::max(settle_spin, length(state.angular_velocity));
            }
            previous = state;
        }
        std::cout << "Stroke " << stroke+1 << (direction > 0 ? " right" : " left")
                  << ": z=" << previous.position.z << " m, turn="
                  << total_angle * 180.0 / pi << " deg, speed="
                  << length(previous.linear_velocity) << " m/s\n";
        require((previous.position.z-starting_z)*direction < -0.1F,
                "each gravity reversal must move the sleeve in the requested direction");
        require(length(subtract(previous.position,settle_start.position)) < 0.002F &&
                std::fabs(total_angle-settle_angle) < 0.002F &&
                settle_speed < 0.02F && settle_spin < 0.05F,
                "each stroke must settle before reversing gravity");
        if (stroke % 2 == 1) {
            require(std::fabs(total_angle-previous_cycle_angle-pi/4.0F) < 0.02F,
                    "each right-left cycle must index one tooth (45 degrees)");
            previous_cycle_angle = total_angle;
        }
    }
    require(std::fabs(total_angle-2*pi) < (gallery_controls ? 0.002F : 0.02F),
            "eight right-left cycles must make one complete revolution");
    std::sort(fixture.step_milliseconds.begin(), fixture.step_milliseconds.end());
    const auto p99 = fixture.step_milliseconds[static_cast<std::size_t>(
        std::ceil(0.99*fixture.step_milliseconds.size()))-1];
    std::cout << "Eight cycles: " << total_angle*180/pi << " deg; peak axis error "
              << maximum_swing << " rad, transverse drift " << maximum_drift
              << " m, energy increase " << maximum_energy_gain << " J; physics p99 "
              << p99 << " ms (diagnostic only, no rendering)\n";
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        test_neutral_gravity_release();
        test_eight_right_left_cycles(false);
        test_eight_right_left_cycles(true);
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
