// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#if defined(PARALLEL_MATER_GALLERY_D3D12)
namespace parallel_mater::d3d12::gallery {
using ::parallel_mater::gallery::Camera;
using ::parallel_mater::gallery::screen_space_force;
#else
namespace parallel_mater::gallery {
#endif

// Scene-authored force controls. Equal force/mass ratios share one GPU update;
// no per-body position readback is needed to apply force at the center of mass.
class ArrowForces {
  public:
    [[nodiscard]] Status initialize(const SceneDefinition &scene,
                                    const SceneInstance &instance) noexcept {
        groups_.clear();
        if (scene.rigid_bodies.size() != instance.rigid_bodies.size())
            return {StatusCode::invalid_argument, {}, "arrow force bindings do not match scene"};
        try {
            std::vector<Group> groups;
            for (std::size_t index = 0; index < scene.rigid_bodies.size(); ++index) {
                const auto &body = scene.rigid_bodies[index];
                if (body.arrow_force == 0.0F) continue;
                const float acceleration = body.arrow_force / body.options.mass;
                if (body.options.motion != MotionType::dynamic ||
                    !std::isfinite(body.arrow_force) || body.arrow_force < 0.0F ||
                    !std::isfinite(body.options.mass) || body.options.mass <= 0.0F ||
                    !std::isfinite(acceleration))
                    return {StatusCode::invalid_argument, {}, "invalid arrow force or body mass"};
                auto group = std::find_if(groups.begin(), groups.end(), [&](const auto &item) {
                    return item.acceleration == acceleration;
                });
                if (group == groups.end()) {
                    groups.push_back({acceleration, {}});
                    group = groups.end() - 1;
                }
                group->bodies.push_back(instance.rigid_bodies[index]);
            }
            groups_ = std::move(groups);
        } catch (...) {
            return {StatusCode::out_of_memory, {}, "arrow force allocation failed"};
        }
        return {};
    }

    [[nodiscard]] bool active() const noexcept { return !groups_.empty(); }

    // Call once before each fixed World::step, not once per displayed frame.
    [[nodiscard]] Status apply(World &world, Camera camera,
                                float right_input, float up_input) const noexcept {
        if (!active() || (right_input == 0.0F && up_input == 0.0F)) return {};
        for (const auto &group : groups_) {
            const auto status = world.apply_central_acceleration(
                {group.bodies.data(), group.bodies.size()},
                screen_space_force(camera, right_input, up_input, group.acceleration));
            if (!status) return status;
        }
        return {};
    }

  private:
    struct Group {
        float acceleration{};
        std::vector<RigidBodyId> bodies;
    };
    std::vector<Group> groups_;
};
} // namespace parallel_mater::gallery
