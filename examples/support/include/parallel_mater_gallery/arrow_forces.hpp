// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <scene_builder.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace parallel_mater::gallery {

// Scene-authored force controls. Equal force/mass ratios share one GPU update;
// no per-body position readback is needed to apply force at the center of mass.
class ArrowForces {
  public:
    [[nodiscard]] bool initialize(const SceneDefinition &scene,
                                  const CommandSceneInstance &instance,
                                  std::string &error) noexcept {
        command_bodies_.clear();
        if (scene.rigid_bodies.size() != instance.rigid_bodies.size()) {
            error = "arrow force bindings do not match scene";
            return false;
        }
        try {
            for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
                const auto &body = scene.rigid_bodies[index];
                if (body.arrow_force == 0.0F) continue;
                if (body.options.motion != MotionType::dynamic ||
                    !std::isfinite(body.arrow_force) || body.arrow_force < 0.0F ||
                    !std::isfinite(body.options.mass) || body.options.mass <= 0.0F) {
                    error = "invalid arrow force or body mass";
                    return false;
                }
                command_bodies_.push_back({
                    instance.rigid_bodies[index], body.arrow_force});
            }
        } catch (...) {
            error = "arrow force allocation failed";
            return false;
        }
        return true;
    }

    [[nodiscard]] bool active() const noexcept {
        return !command_bodies_.empty();
    }

    [[nodiscard]] bool apply(CommandScene &simulation, Camera camera,
                             float right_input, float up_input,
                             std::string &error) const noexcept {
        if (command_bodies_.empty() ||
            (right_input == 0.0F && up_input == 0.0F))
            return true;
        std::vector<PmRigidInput> inputs;
        try {
            inputs.reserve(command_bodies_.size());
            const Vec3 direction = screen_space_force(
                camera, right_input, up_input, 1.0F);
            for (const auto &body : command_bodies_) {
                PmRigidInput input{};
                input.bodyIndex = body.body;
                input.force[0] = direction.x * body.force;
                input.force[1] = direction.y * body.force;
                input.force[2] = direction.z * body.force;
                inputs.push_back(input);
            }
        } catch (...) {
            error = "arrow force command allocation failed";
            return false;
        }
        return simulation.apply_rigid_inputs(inputs, error);
    }

  private:
    struct CommandBody {
        std::uint32_t body{};
        float force{};
    };
    std::vector<CommandBody> command_bodies_;
};
} // namespace parallel_mater::gallery
