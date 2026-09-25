// SPDX-License-Identifier: MIT
#include "foam_visuals.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace parallel_mater::gallery {
namespace {

[[nodiscard]] Vec3 subtract(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

[[nodiscard]] Vec3 normalize(Vec3 value) {
    const float length = std::sqrt(std::max(dot(value, value), 1.0e-20F));
    return {value.x / length, value.y / length, value.z / length};
}

[[nodiscard]] std::uint32_t foam_hash(std::uint32_t value) noexcept {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

[[nodiscard]] float foam_unit(std::uint32_t value) noexcept {
    return static_cast<float>(foam_hash(value) & 0x00ffffffU) /
           16777216.0F;
}

} // namespace

void FoamVisuals::advance(const std::vector<Vec3> &positions,
                          const std::vector<float> &foam,
                          const std::vector<std::uint32_t> &ids,
                          float particle_radius) {
    constexpr float dt = 1.0F / 60.0F;
    constexpr std::size_t capacity = 2048U;
    const std::size_t count = std::min({positions.size(), foam.size(), ids.size()});
    std::unordered_map<std::uint32_t, std::size_t> particle_by_id;
    particle_by_id.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        particle_by_id.emplace(ids[i], i);
    std::unordered_set<std::uint32_t> active;
    active.reserve(patches_.size());
    for (std::size_t i = 0; i < patches_.size();) {
        Patch &patch = patches_[i];
        patch.age += dt;
        const auto owner = particle_by_id.find(patch.particle_id);
        if (patch.age >= patch.lifetime || owner == particle_by_id.end()) {
            patches_[i] = patches_.back();
            patches_.pop_back();
            continue;
        }
        patch.position = positions[owner->second];
        active.insert(patch.particle_id);
        ++i;
    }
    // Peg's 0.03 m particles retain their authored bubble size. The larger
    // stream particles need visible patches at the wider Fluid camera.
    const float relative_radius = particle_radius / 0.03F;
    const float radius_scale = std::clamp(
        relative_radius * relative_radius, 0.75F, 2.5F);
    for (std::size_t i = 0; i < count && patches_.size() < capacity; ++i) {
        if (foam[i] <= 0.0F || active.contains(ids[i])) continue;
        const std::uint32_t seed = ids[i] ^
            static_cast<std::uint32_t>(tick_ * 104729ULL);
        if (foam_unit(seed) >= std::min(1.0F, 16.0F * foam[i] * dt))
            continue;
        const float radius = radius_scale * (0.0096F + 0.0128F *
            foam_unit(seed ^ 0x9e3779b9U));
        const float life = 1.5F + 2.0F *
            foam_unit(seed ^ 0x85ebca6bU);
        patches_.push_back({ids[i], positions[i],
                            0.12F * life * foam_unit(seed ^ 0xc2b2ae35U),
                            life, radius});
        active.insert(ids[i]);
    }
    ++tick_;
}

void FoamVisuals::paint(float support_radius, Camera camera,
                        std::uint32_t width, std::uint32_t height,
                        const std::vector<float> &depth,
                        const std::vector<float> &rigid_depth,
                        std::vector<std::uint32_t> &rgba) const {
    const Vec3 forward = normalize(subtract(camera.target, camera.eye));
    const Vec3 right = normalize(cross(forward, camera.up));
    const Vec3 up = normalize(cross(right, forward));
    constexpr float radians = 0.017453292519943295F;
    constexpr float pi = 3.14159265358979323846F;
    const float tangent = std::tan(camera.vertical_field_of_view_degrees *
                                   radians * 0.5F);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float relative_support = support_radius / 0.12F;
    const float minimum_pixel_radius = 0.75F * std::clamp(
        relative_support * relative_support, 1.0F, 2.5F);
    for (const Patch &patch : patches_) {
        const float progress = patch.age / patch.lifetime;
        const float fade = std::clamp((1.0F - progress) * 1.8F, 0.0F, 1.0F);
        const float grow = 0.58F + 0.42F *
            std::clamp(progress / 0.12F, 0.0F, 1.0F);
        const float spread = 0.64F + 0.56F *
            std::clamp(progress / 0.55F, 0.0F, 1.0F);
        const float phase = 2.0F * pi * foam_unit(patch.particle_id) +
                            static_cast<float>(tick_) * 0.015F;
        const Vec3 anchor{patch.position.x,
                          patch.position.y + 0.30F * support_radius,
                          patch.position.z};
        for (std::uint32_t lobe = 0U; lobe < 11U; ++lobe) {
            const std::uint32_t seed = patch.particle_id * 0x9e3779b9U +
                                       lobe * 0x85ebca6bU;
            const float first = foam_unit(seed ^ 0xc2b2ae35U);
            const float second = foam_unit(seed ^ 0x27d4eb2fU);
            const float third = foam_unit(seed ^ 0x165667b1U);
            const float angle = phase + 2.0F * pi * second;
            const float radial = lobe == 0U ? 0.0F :
                (0.08F + 1.35F * std::sqrt(first)) * patch.radius * spread;
            const Vec3 center{anchor.x + radial * std::cos(angle),
                              anchor.y + 0.16F * third * patch.radius,
                              anchor.z + radial * std::sin(angle)};
            const float radius = (lobe == 0U ? 0.56F :
                0.22F + 0.50F * third) * patch.radius * grow;
            const Vec3 offset = subtract(center, camera.eye);
            const float forward_distance = dot(offset, forward);
            if (forward_distance <= radius) continue;
            const float center_x = 0.5F * width *
                (1.0F + dot(offset, right) /
                    (forward_distance * tangent * aspect));
            const float center_y = 0.5F * height *
                (1.0F + dot(offset, up) / (forward_distance * tangent));
            const float pixel_radius = std::max(minimum_pixel_radius,
                radius * height / (2.0F * forward_distance * tangent));
            if (center_x + pixel_radius < 0.0F ||
                center_x - pixel_radius >= width ||
                center_y + pixel_radius < 0.0F ||
                center_y - pixel_radius >= height) continue;
            const int x0 = std::max(0, static_cast<int>(center_x - pixel_radius));
            const int y0 = std::max(0, static_cast<int>(center_y - pixel_radius));
            const int x1 = std::min(static_cast<int>(width) - 1,
                                    static_cast<int>(center_x + pixel_radius));
            const int y1 = std::min(static_cast<int>(height) - 1,
                                    static_cast<int>(center_y + pixel_radius));
            const float center_depth = std::sqrt(dot(offset, offset));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    const float dx = (x + 0.5F - center_x) / pixel_radius;
                    const float dy = (y + 0.5F - center_y) / pixel_radius;
                    const float squared = dx * dx + dy * dy;
                    if (squared >= 1.0F) continue;
                    const float cap = std::sqrt(1.0F - squared);
                    const float bubble_depth = center_depth - radius * cap;
                    const std::size_t pixel = static_cast<std::size_t>(y) *
                                              width + x;
                    if (bubble_depth > rigid_depth[pixel] + 0.002F ||
                        bubble_depth > depth[pixel] +
                            0.5F * support_radius) continue;
                    const float rim = std::pow(1.0F - cap, 2.6F);
                    const float glint = std::pow(std::max(0.0F,
                        0.70F * cap + 0.30F * (dx + dy)), 34.0F);
                    const float opacity = fade * fade * (3.0F - 2.0F * fade) *
                        std::min(0.96F, 0.055F + 0.82F * rim +
                                         0.26F * glint);
                    const std::uint32_t old = rgba[pixel];
                    const auto blend = [opacity](std::uint32_t previous,
                                                  float film) {
                        return static_cast<std::uint32_t>(std::clamp(
                            previous * (1.0F - opacity) + film * opacity,
                            0.0F, 255.0F));
                    };
                    const std::uint32_t red = blend(old & 255U,
                                                    204.0F + 40.0F * third);
                    const std::uint32_t green = blend((old >> 8U) & 255U,
                                                      229.0F + 24.0F * third);
                    const std::uint32_t blue = blend((old >> 16U) & 255U,
                                                     255.0F);
                    rgba[pixel] = 0xff000000U | (blue << 16U) |
                                  (green << 8U) | red;
                }
        }
    }
}

} // namespace parallel_mater::gallery
