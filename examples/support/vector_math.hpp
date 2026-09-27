// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.hpp>

#include <cmath>

namespace parallel_mater::gallery::math {

[[nodiscard]] inline Vec3 add(Vec3 first, Vec3 second) noexcept {
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

[[nodiscard]] inline Vec3 subtract(Vec3 first, Vec3 second) noexcept {
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

[[nodiscard]] inline Vec3 multiply(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

[[nodiscard]] inline Vec3 multiply(Vec3 value, Vec3 scale) noexcept {
    return {value.x * scale.x, value.y * scale.y, value.z * scale.z};
}

[[nodiscard]] inline float dot(Vec3 first, Vec3 second) noexcept {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

[[nodiscard]] inline Vec3 cross(Vec3 first, Vec3 second) noexcept {
    return {first.y * second.z - first.z * second.y,
            first.z * second.x - first.x * second.z,
            first.x * second.y - first.y * second.x};
}

[[nodiscard]] inline float length(Vec3 value) noexcept {
    return std::sqrt(dot(value, value));
}

[[nodiscard]] inline Vec3 normalize_or(Vec3 value, Vec3 fallback,
                                       float epsilon = 1.0e-8F) noexcept {
    const float size = length(value);
    return size > epsilon ? multiply(value, 1.0F / size) : fallback;
}

} // namespace parallel_mater::gallery::math
