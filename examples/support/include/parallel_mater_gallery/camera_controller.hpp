// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/renderer.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

namespace parallel_mater::gallery {

enum class CameraDragMode { orbit, pan };

struct CameraPreset {
    Vec3 target{0.0F, 1.8F, 0.0F};
    float distance_scale{1.0F};
    std::optional<float> pitch{};
};

// Camera-relative gravity shared by scenes with a tiltable course. Screen
// right/up input maps to camera-right/forward ground directions in [-1, 1].
inline constexpr float peg_paint_gravity_tilt_degrees = 50.0F;
inline constexpr float collector_gravity_tilt_degrees = 30.0F;

// Project the requested screen direction onto the ground before applying it.
// A held diagonal has the same magnitude as one key.
[[nodiscard]] inline Vec3 screen_space_force(
    Camera camera, float right_input, float up_input, float magnitude) noexcept {
    const float input_length = std::hypot(right_input, up_input);
    if (input_length <= 1.0e-6F || magnitude <= 0.0F) return {};
    const auto cross = [](Vec3 a, Vec3 b) {
        return Vec3{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
    };
    const auto normalize = [](Vec3 value, Vec3 fallback) {
        const float length = std::sqrt(value.x*value.x + value.y*value.y + value.z*value.z);
        return length > 1.0e-6F
            ? Vec3{value.x/length, value.y/length, value.z/length} : fallback;
    };
    const Vec3 forward = normalize({camera.target.x-camera.eye.x,
        camera.target.y-camera.eye.y, camera.target.z-camera.eye.z}, {0, 0, -1});
    const Vec3 fallback_up = std::fabs(forward.y) > 0.99F ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
    const Vec3 right = normalize(cross(forward, camera.up),
                                normalize(cross(forward, fallback_up), {1, 0, 0}));
    const Vec3 up = normalize(cross(right, forward), {0, 1, 0});
    const Vec3 screen_direction{right.x*right_input + up.x*up_input,
                                right.y*right_input + up.y*up_input,
                                right.z*right_input + up.z*up_input};
    const float ground_length = std::hypot(screen_direction.x, screen_direction.z);
    if (ground_length <= 1.0e-6F) return {};
    const float scale = magnitude / ground_length;
    return {screen_direction.x*scale, 0.0F, screen_direction.z*scale};
}

[[nodiscard]] inline Vec3 screen_space_gravity(
    Camera camera, float right_input, float up_input, float magnitude,
    float tilt_degrees) noexcept {
    constexpr float radians = 0.017453292519943295F;
    Vec3 forward{camera.target.x - camera.eye.x, 0.0F,
                 camera.target.z - camera.eye.z};
    const float forward_length = std::hypot(forward.x, forward.z);
    if (forward_length > 1.0e-6F) {
        forward.x /= forward_length;
        forward.z /= forward_length;
    } else {
        forward = {0.0F, 0.0F, -1.0F};
    }
    const Vec3 right{-forward.z, 0.0F, forward.x};
    Vec3 steering{right.x * right_input + forward.x * up_input, 0.0F,
                  right.z * right_input + forward.z * up_input};
    const float length = std::hypot(steering.x, steering.z);
    const float tilt = std::clamp(tilt_degrees, 0.0F, 89.0F) * radians;
    Vec3 gravity{0.0F, -magnitude, 0.0F};
    if (length > 1.0e-6F) {
        const float horizontal = magnitude * std::sin(tilt) / length;
        gravity = {steering.x * horizontal, -magnitude * std::cos(tilt),
                   steering.z * horizontal};
    }
    return gravity;
}

[[nodiscard]] inline Vec3 steer_gravity(Vec3 current, Camera camera,
                                         float right_input, float up_input,
                                         float magnitude, float tilt_degrees,
                                         float timestep) noexcept {
    const Vec3 desired = screen_space_gravity(
        camera, right_input, up_input, magnitude, tilt_degrees);
    const float blend = 1.0F - std::exp(-timestep / 0.16F);
    Vec3 result{current.x + (desired.x - current.x) * blend,
                current.y + (desired.y - current.y) * blend,
                current.z + (desired.z - current.z) * blend};
    const float result_length = std::sqrt(result.x * result.x +
        result.y * result.y + result.z * result.z);
    if (result_length <= 1.0e-6F) return desired;
    const float scale = magnitude / result_length;
    return {result.x * scale, result.y * scale, result.z * scale};
}

class CameraController {
  public:
    void set_preset(CameraPreset preset) noexcept {
        preset_ = preset;
        if (preset.pitch) pitch_ = std::clamp(*preset.pitch, -1.35F, 1.35F);
        pan_ = {};
        dragging_ = false;
    }

    void begin_drag(CameraDragMode mode, double x, double y) noexcept {
        mode_ = mode;
        dragging_ = true;
        previous_x_ = x;
        previous_y_ = y;
    }

    void end_drag() noexcept { dragging_ = false; }

    void move_cursor(double x, double y, int viewport_height,
                     bool enabled = true) noexcept {
        const float dx = static_cast<float>(x - previous_x_);
        const float dy = static_cast<float>(y - previous_y_);
        previous_x_ = x;
        previous_y_ = y;
        if (!dragging_ || !enabled) return;
        if (mode_ == CameraDragMode::orbit) {
            yaw_ -= dx * 0.006F;
            pitch_ = std::clamp(pitch_ + dy * 0.006F, -1.35F, 1.35F);
            return;
        }
        if (viewport_height <= 0) return;
        const float radians = camera().vertical_field_of_view_degrees *
                              0.00872664626F;
        const float units_per_pixel = 2.0F * distance_ *
            preset_.distance_scale * std::tan(radians) /
            static_cast<float>(viewport_height);
        const float sin_yaw = std::sin(yaw_), cos_yaw = std::cos(yaw_);
        const float sin_pitch = std::sin(pitch_);
        const float cos_pitch = std::cos(pitch_);
        const Vec3 right{cos_yaw, 0.0F, -sin_yaw};
        const Vec3 screen_up{-sin_yaw * sin_pitch, cos_pitch,
                             -cos_yaw * sin_pitch};
        pan_.x += (-dx * right.x + dy * screen_up.x) * units_per_pixel;
        pan_.y += dy * screen_up.y * units_per_pixel;
        pan_.z += (-dx * right.z + dy * screen_up.z) * units_per_pixel;
    }

    void zoom(double offset) noexcept {
        distance_ = std::clamp(
            distance_ * std::exp(static_cast<float>(-offset) * 0.1F),
            3.0F, 30.0F);
    }

    [[nodiscard]] Camera camera() const noexcept {
        Camera result;
        result.target = {preset_.target.x + pan_.x,
                         preset_.target.y + pan_.y,
                         preset_.target.z + pan_.z};
        const float distance = distance_ * preset_.distance_scale;
        const float horizontal = distance * std::cos(pitch_);
        result.eye = {result.target.x + horizontal * std::sin(yaw_),
                      result.target.y + distance * std::sin(pitch_),
                      result.target.z + horizontal * std::cos(yaw_)};
        return result;
    }

  private:
    CameraPreset preset_{};
    Vec3 pan_{};
    float yaw_{0.62F};
    float pitch_{0.32F};
    float distance_{11.0F};
    CameraDragMode mode_{CameraDragMode::orbit};
    bool dragging_{};
    double previous_x_{};
    double previous_y_{};
};

} // namespace parallel_mater::gallery
