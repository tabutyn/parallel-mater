// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/camera_controller.hpp>

#include <array>
#include <cmath>
#include <iostream>

namespace {

using parallel_mater::Vec3;
using parallel_mater::gallery::Camera;
using parallel_mater::gallery::CameraController;
using parallel_mater::gallery::CameraDragMode;
using parallel_mater::gallery::CameraPreset;
using parallel_mater::gallery::steer_gravity;
using parallel_mater::gallery::peg_paint_gravity_tilt_degrees;

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

Vec3 subtract(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 normalized(Vec3 p) {
    const float length = std::sqrt(dot(p, p));
    return {p.x / length, p.y / length, p.z / length};
}

bool near(float a, float b) {
    return std::fabs(a - b) < 1.0e-4F;
}

bool near(Vec3 a, Vec3 b) {
    return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z);
}

void check_pan(CameraController &controller, const char *message) {
    const Camera before = controller.camera();
    const Vec3 forward = normalized(subtract(before.target, before.eye));
    const Vec3 right = normalized(cross(forward, before.up));
    const Vec3 screen_up = cross(right, forward);
    const float distance = std::sqrt(dot(subtract(before.eye, before.target),
                                         subtract(before.eye, before.target)));
    const float units_per_pixel = 2.0F * distance *
        std::tan(before.vertical_field_of_view_degrees * 0.00872664626F) /
        720.0F;
    controller.begin_drag(CameraDragMode::pan, 200.0, 180.0);
    check(near(controller.camera().target, before.target),
          "Shift-click without movement does not pan");
    controller.move_cursor(260.0, 210.0, 720);
    const Camera after = controller.camera();
    const Vec3 movement = subtract(after.target, before.target);
    check(near(dot(movement, right), -60.0F * units_per_pixel) &&
              near(dot(movement, screen_up), 30.0F * units_per_pixel),
          message);
    check(near(subtract(after.eye, before.eye), movement),
          "panning translates eye and target together");
    controller.end_drag();
    controller.move_cursor(400.0, 400.0, 720);
    check(near(controller.camera().target, after.target),
          "released mouse does not pan");
}

} // namespace

int main() {
    const Camera steering_camera{{0.0F, 3.0F, 5.0F},
                                 {0.0F, 0.0F, 0.0F}};
    Vec3 tilted{0.0F, -19.62F, 0.0F};
    for (int step = 0; step < 60; ++step)
        tilted = steer_gravity(tilted, steering_camera, 1.0F, 0.0F,
                               19.62F, peg_paint_gravity_tilt_degrees,
                               1.0F / 60.0F);
    check(tilted.x > 14.9F && tilted.y < -12.5F &&
              std::fabs(tilted.z) < 1.0e-4F &&
              near(std::sqrt(dot(tilted, tilted)), 19.62F),
          "Peg arrow steering tilts authored gravity toward camera right");
    Vec3 forward_tilt{0.0F, -19.62F, 0.0F};
    for (int step = 0; step < 60; ++step)
        forward_tilt = steer_gravity(forward_tilt, steering_camera,
                                     0.0F, 1.0F, 19.62F,
                                     peg_paint_gravity_tilt_degrees,
                                     1.0F / 60.0F);
    check(forward_tilt.z < -14.9F &&
              std::fabs(forward_tilt.x) < 1.0e-4F,
          "Peg Up Arrow steers into the camera view");
    for (int step = 0; step < 120; ++step)
        tilted = steer_gravity(tilted, steering_camera, 0.0F, 0.0F,
                               19.62F, peg_paint_gravity_tilt_degrees,
                               1.0F / 60.0F);
    check(std::fabs(tilted.x) < 0.001F && near(tilted.y, -19.62F),
          "Peg gravity eases back to authored down when arrows release");
    CameraController controller;
    const Camera initial = controller.camera();
    check_pan(controller, "Rigid Body pan matches screen-space drag");

    const Vec3 pan_target = controller.camera().target;
    controller.begin_drag(CameraDragMode::orbit, 100.0, 100.0);
    controller.move_cursor(140.0, 120.0, 720);
    controller.end_drag();
    check(near(controller.camera().target, pan_target) &&
              !near(subtract(controller.camera().eye,
                             controller.camera().target),
                    subtract(initial.eye, initial.target)),
          "ordinary left-drag still orbits");

    const std::array<CameraPreset, 2> other_scenes{{
        {.target = {0.5F, 2.2F, 0.0F}},
        {.target = {-2.0F, -1.0F, -2.0F}, .distance_scale = 1.6F}}};
    for (std::size_t scene = 0; scene < other_scenes.size(); ++scene) {
        controller.set_preset(other_scenes[scene]);
        check(near(controller.camera().target, other_scenes[scene].target),
              "scene switch resets pan to its own target");
        check_pan(controller, scene == 0U
            ? "DUMP pan matches screen-space drag"
            : "Fluid pan matches screen-space drag at wider camera distance");
    }

    controller.set_preset({});
    controller.begin_drag(CameraDragMode::pan, 10.0, 10.0);
    controller.move_cursor(80.0, 80.0, 720, false);
    check(near(controller.camera().target, CameraPreset{}.target),
          "dialog blocks camera movement");
    controller.end_drag();
    controller.zoom(2.0);
    check(std::sqrt(dot(subtract(controller.camera().eye,
                                 controller.camera().target),
                        subtract(controller.camera().eye,
                                 controller.camera().target))) < 11.0F,
          "wheel zoom remains available");

    controller.set_preset({.target = {-2.0F, -1.0F, -2.0F},
                           .distance_scale = 1.6F});
    controller.begin_drag(CameraDragMode::orbit, 100.0, 100.0);
    controller.move_cursor(100.0, -1000.0, 720);
    controller.end_drag();
    check(controller.camera().eye.y < -3.0F,
          "Fluid camera can inspect the underside of the level");

    return failures == 0 ? 0 : 1;
}
