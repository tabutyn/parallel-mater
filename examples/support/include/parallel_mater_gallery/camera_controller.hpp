// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/renderer.hpp>

#include <algorithm>
#include <cmath>

namespace parallel_mater::gallery {

enum class CameraDragMode { orbit, pan };

struct CameraPreset {
    Vec3 target{0.0F, 1.8F, 0.0F};
    float distance_scale{1.0F};
};

class CameraController {
  public:
    void set_preset(CameraPreset preset) noexcept {
        preset_ = preset;
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
