// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>
#include <scene_builder.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::gallery {
namespace fixed_collector_detail {

[[nodiscard]] inline Quaternion conjugate(Quaternion value) noexcept {
    return {-value.x, -value.y, -value.z, value.w};
}

[[nodiscard]] inline Quaternion multiply(Quaternion left,
                                         Quaternion right) noexcept {
    return {
        left.w * right.x + left.x * right.w + left.y * right.z -
            left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w +
            left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x +
            left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y -
            left.z * right.z};
}

[[nodiscard]] inline Vec3 rotate(Quaternion rotation, Vec3 point) noexcept {
    const Quaternion vector{point.x, point.y, point.z, 0.0F};
    const Quaternion result = multiply(
        multiply(rotation, vector), conjugate(rotation));
    return {result.x, result.y, result.z};
}

[[nodiscard]] inline Vec3 subtract(Vec3 left, Vec3 right) noexcept {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

} // namespace fixed_collector_detail

// Gameplay helper for the Fixed gallery scene. It consumes public contact
// output and records generated fixed constraints through CommandScene.
class FixedContactCollector {
  public:
    [[nodiscard]] bool initialize(
        const SceneDefinition &scene, const CommandSceneInstance &instance,
        std::string &error,
        std::string_view collector_source_name = "Large") noexcept {
        collector_body_index_ = invalid_index;
        generated_constraints_ = 0U;
        try {
            members_.assign(scene.rigid_bodies.size(), std::uint8_t{});
        } catch (...) {
            error = "fixed collector state allocation failed";
            return false;
        }
        if (scene.rigid_bodies.size() != instance.rigid_bodies.size()) {
            error = "fixed collector body bindings do not match scene";
            return false;
        }
        for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
            const auto &body = scene.rigid_bodies[index];
            if (body.source_name != collector_source_name &&
                body.name != collector_source_name)
                continue;
            if (collector_body_index_ != invalid_index) {
                error = "fixed collector body name is ambiguous";
                return false;
            }
            collector_body_index_ = index;
        }
        if (collector_body_index_ == invalid_index) {
            error = "fixed collector body was not found";
            return false;
        }
        members_[collector_body_index_] = 1U;

        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto &constraint : scene.rigid_constraints) {
                if (constraint.options.type != RigidConstraintType::fixed ||
                    !constraint.options.enabled ||
                    constraint.body_a >= members_.size() ||
                    constraint.body_b >= members_.size())
                    continue;
                const bool member_a = members_[constraint.body_a] != 0U;
                const bool member_b = members_[constraint.body_b] != 0U;
                if (member_a == member_b) continue;
                members_[member_a ? constraint.body_b : constraint.body_a] = 1U;
                changed = true;
            }
        }
        return true;
    }

    [[nodiscard]] bool active() const noexcept {
        return collector_body_index_ != invalid_index;
    }

    [[nodiscard]] std::size_t attached_count() const noexcept {
        return static_cast<std::size_t>(
            std::count(members_.begin(), members_.end(), std::uint8_t{1U}));
    }

    [[nodiscard]] std::size_t generated_constraint_count() const noexcept {
        return generated_constraints_;
    }

    [[nodiscard]] static std::uint32_t constraint_capacity(
        const SceneDefinition &scene) noexcept {
        return static_cast<std::uint32_t>(std::max<std::size_t>(
            1U, std::max(scene.rigid_constraints.size(),
                         scene.rigid_bodies.size())));
    }

    [[nodiscard]] bool apply_loose_gravity(
        CommandScene &simulation, const SceneDefinition &scene,
        const CommandSceneInstance &instance, Vec3 loose_gravity,
        Vec3 world_gravity, std::string &error) const noexcept {
        if (!active() || members_.size() != scene.rigid_bodies.size() ||
            members_.size() != instance.rigid_bodies.size()) {
            error = "fixed collector bindings are invalid";
            return false;
        }
        const Vec3 acceleration{loose_gravity.x - world_gravity.x,
                                loose_gravity.y - world_gravity.y,
                                loose_gravity.z - world_gravity.z};
        std::vector<PmRigidInput> inputs;
        try {
            for (std::size_t index = 0U; index < members_.size(); ++index) {
                if (members_[index] != 0U ||
                    scene.rigid_bodies[index].options.motion !=
                        MotionType::dynamic)
                    continue;
                PmRigidInput input{};
                input.bodyIndex = instance.rigid_bodies[index];
                const float mass = scene.rigid_bodies[index].options.mass;
                input.force[0] = acceleration.x * mass;
                input.force[1] = acceleration.y * mass;
                input.force[2] = acceleration.z * mass;
                inputs.push_back(input);
            }
        } catch (...) {
            error = "fixed collector gravity allocation failed";
            return false;
        }
        return inputs.empty() || simulation.apply_rigid_inputs(inputs, error);
    }

    // Call after a step recorded with contact collection enabled.
    [[nodiscard]] bool collect(CommandScene &simulation,
                               const SceneDefinition &scene,
                               const CommandSceneInstance &instance,
                               const CommandSceneFrame &frame,
                               std::string &error) noexcept {
        if (!active() || members_.size() != scene.rigid_bodies.size() ||
            frame.rigid_states.size() != members_.size()) {
            error = "fixed collector bindings are invalid";
            return false;
        }
        const auto state = [&](std::size_t index) {
            const PmRigidState &source = frame.rigid_states[index];
            return RigidBodyState{
                {source.position[0], source.position[1], source.position[2]},
                {source.orientation[0], source.orientation[1],
                 source.orientation[2], source.orientation[3]},
                {source.linearVelocity[0], source.linearVelocity[1],
                 source.linearVelocity[2]},
                {source.angularVelocity[0], source.angularVelocity[1],
                 source.angularVelocity[2]}};
        };
        for (const PmContactEvent &contact : frame.contacts) {
            if (contact.type != PM_CONTACT_RIGID_RIGID ||
                (contact.penetration <= 0.0F && contact.normalImpulse <= 0.0F))
                continue;
            const std::size_t first = contact.first;
            const std::size_t second = contact.second;
            if (first >= members_.size() || second >= members_.size() ||
                first == second ||
                (first != collector_body_index_ &&
                 second != collector_body_index_))
                continue;
            const std::size_t target = first == collector_body_index_
                ? second : first;
            if (members_[target] != 0U ||
                scene.rigid_bodies[target].options.motion != MotionType::dynamic)
                continue;

            using namespace fixed_collector_detail;
            const RigidBodyState member_state = state(collector_body_index_);
            const RigidBodyState target_state = state(target);
            const Vec3 position{contact.position.x, contact.position.y,
                                contact.position.z};
            RigidConstraintDefinition definition{};
            definition.name = "Collected";
            definition.body_a = static_cast<std::uint32_t>(collector_body_index_);
            definition.body_b = static_cast<std::uint32_t>(target);
            definition.options = {
                .type = RigidConstraintType::fixed,
                .local_anchor_a = rotate(conjugate(member_state.orientation),
                    subtract(position, member_state.position)),
                .local_anchor_b = rotate(conjugate(target_state.orientation),
                    subtract(position, target_state.position)),
                .local_orientation_a = {},
                .local_orientation_b = multiply(
                    conjugate(target_state.orientation),
                    member_state.orientation),
                .enabled = true,
                .disable_collisions = true,
                .solver_iterations = 16U};
            const std::uint32_t slot = static_cast<std::uint32_t>(
                scene.rigid_constraints.size() + generated_constraints_);
            if (!simulation.write_constraint(slot, definition, error))
                return false;
            ++generated_constraints_;
            members_[target] = 1U;
        }
        return true;
    }

  private:
    static constexpr std::size_t invalid_index =
        std::numeric_limits<std::size_t>::max();
    std::size_t collector_body_index_{invalid_index};
    std::vector<std::uint8_t> members_{};
    std::size_t generated_constraints_{};
};

} // namespace parallel_mater::gallery
