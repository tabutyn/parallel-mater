// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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

// Gameplay helper for the Fixed gallery scene. One authored fixed constraint
// seeds the collector. Actual collision results add each dynamic body that
// directly touches the collector body exactly once.
class FixedContactCollector {
  public:
    [[nodiscard]] Status initialize(
        const SceneDefinition &scene, const SceneInstance &instance,
        std::string_view collector_source_name = "Large") noexcept {
        collector_body_index_ = invalid_index;
        generated_constraints_.clear();
        try {
            members_.assign(scene.rigid_bodies.size(), std::uint8_t{});
            generated_constraints_.reserve(scene.rigid_bodies.size());
            gravity_bodies_.reserve(scene.rigid_bodies.size());
            dense_ids_.resize(scene.rigid_bodies.size());
            dense_states_.resize(scene.rigid_bodies.size());
        } catch (...) {
            return {StatusCode::out_of_memory, cudaSuccess,
                    "fixed collector state allocation failed"};
        }
        if (scene.rigid_bodies.size() != instance.rigid_bodies.size()) {
            return {StatusCode::invalid_argument, cudaSuccess,
                    "fixed collector body bindings do not match scene"};
        }
        for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
            const RigidBodyDefinition &body = scene.rigid_bodies[index];
            if (body.source_name == collector_source_name ||
                body.name == collector_source_name) {
                if (collector_body_index_ != invalid_index) {
                    return {StatusCode::invalid_argument, cudaSuccess,
                            "fixed collector body name is ambiguous"};
                }
                collector_body_index_ = index;
            }
        }
        if (collector_body_index_ == invalid_index) {
            return {StatusCode::invalid_argument, cudaSuccess,
                    "fixed collector body was not found"};
        }
        members_[collector_body_index_] = 1U;

        // Preserve any enabled authored fixed chain as the initial cluster.
        bool changed = true;
        while (changed) {
            changed = false;
            for (const RigidConstraintDefinition &constraint :
                 scene.rigid_constraints) {
                if (constraint.options.type != RigidConstraintType::fixed ||
                    !constraint.options.enabled ||
                    constraint.body_a >= members_.size() ||
                    constraint.body_b >= members_.size()) {
                    continue;
                }
                const bool member_a = members_[constraint.body_a] != 0U;
                const bool member_b = members_[constraint.body_b] != 0U;
                if (member_a == member_b) continue;
                members_[member_a ? constraint.body_b : constraint.body_a] = 1U;
                changed = true;
            }
        }
        return {};
    }

    [[nodiscard]] bool active() const noexcept {
        return collector_body_index_ != invalid_index;
    }

    [[nodiscard]] std::size_t attached_count() const noexcept {
        return static_cast<std::size_t>(
            std::count(members_.begin(), members_.end(), std::uint8_t{1U}));
    }

    [[nodiscard]] std::size_t generated_constraint_count() const noexcept {
        return generated_constraints_.size();
    }

    [[nodiscard]] static std::uint32_t constraint_capacity(
        const SceneDefinition &scene) noexcept {
        return static_cast<std::uint32_t>(std::max<std::size_t>(
            1U, std::max(scene.rigid_constraints.size(),
                         scene.rigid_bodies.size())));
    }

    // World gravity drives the attached cluster. Counter-force loose bodies at
    // their centers of mass so they retain downward gravity until collected.
    [[nodiscard]] Status apply_loose_gravity(
        World &world, const SceneDefinition &scene,
        const SceneInstance &instance, Vec3 loose_gravity,
        Vec3 world_gravity) const noexcept {
        if (!active() || members_.size() != scene.rigid_bodies.size() ||
            members_.size() != instance.rigid_bodies.size()) {
            return {StatusCode::invalid_argument, cudaSuccess,
                    "fixed collector bindings are invalid"};
        }
        const Vec3 acceleration{
            loose_gravity.x - world_gravity.x,
            loose_gravity.y - world_gravity.y,
            loose_gravity.z - world_gravity.z};
        if (acceleration.x == 0.0F && acceleration.y == 0.0F &&
            acceleration.z == 0.0F) {
            return {};
        }
        gravity_bodies_.clear();
        for (std::size_t index = 0U; index < members_.size(); ++index) {
            if (members_[index] != 0U ||
                scene.rigid_bodies[index].options.motion !=
                    MotionType::dynamic) {
                continue;
            }
            try {
                gravity_bodies_.push_back(instance.rigid_bodies[index]);
            } catch (...) {
                return {StatusCode::out_of_memory, cudaSuccess,
                        "fixed collector gravity batch allocation failed"};
            }
        }
        return world.apply_central_acceleration(
            {gravity_bodies_.data(), gravity_bodies_.size()}, acceleration);
    }

    // Call after a step made with collect_rigid_contacts=true.
    [[nodiscard]] Status collect(World &world, const SceneDefinition &scene,
                                 const SceneInstance &instance) noexcept {
        if (!active() || members_.size() != scene.rigid_bodies.size() ||
            members_.size() != instance.rigid_bodies.size()) {
            return {StatusCode::invalid_argument, cudaSuccess,
                    "fixed collector bindings are invalid"};
        }
        const RigidContactDeviceView view = world.rigid_contacts();
        std::vector<RigidContactEvent> contacts;
        try {
            contacts.resize(view.event_count);
        } catch (...) {
            return {StatusCode::out_of_memory, cudaSuccess,
                    "fixed collector contact allocation failed"};
        }
        if (!contacts.empty()) {
            const cudaError_t copied = cudaMemcpy(
                contacts.data(), view.events.data,
                contacts.size() * sizeof(RigidContactEvent),
                cudaMemcpyDeviceToHost);
            if (copied != cudaSuccess) {
                return {StatusCode::cuda_failure, copied,
                        "fixed collector contact readback failed"};
            }
        }

        const auto body_index = [&](RigidBodyId id) noexcept {
            const auto found = std::find(instance.rigid_bodies.begin(),
                                         instance.rigid_bodies.end(), id);
            return found == instance.rigid_bodies.end()
                ? invalid_index
                : static_cast<std::size_t>(found -
                                           instance.rigid_bodies.begin());
        };
        bool states_loaded = false;
        for (const RigidContactEvent &contact : contacts) {
            // Ignore speculative search-margin contacts until geometry touches
            // or an actual normal impulse is applied.
            if (contact.penetration <= 0.0F && contact.normal_impulse <= 0.0F)
                continue;
            const std::size_t first = body_index(contact.body);
            const std::size_t second = body_index(contact.collider);
            if (first == invalid_index || second == invalid_index ||
                first == second) continue;
            if (first != collector_body_index_ &&
                second != collector_body_index_) {
                continue;
            }
            const std::size_t member = collector_body_index_;
            const std::size_t target = first == collector_body_index_
                ? second
                : first;
            if (members_[target] != 0U) continue;
            if (scene.rigid_bodies[target].options.motion != MotionType::dynamic)
                continue;

            if (!states_loaded) {
                RigidBodyDeviceView state_view{};
                Status status = world.rigid_body_view(state_view);
                if (!status) return status;
                if (state_view.ids.size != dense_ids_.size() ||
                    state_view.states.size != dense_states_.size()) {
                    return {StatusCode::internal_error, cudaSuccess,
                            "fixed collector body count changed"};
                }
                cudaError_t copied = cudaMemcpy(
                    dense_ids_.data(), state_view.ids.data,
                    dense_ids_.size() * sizeof(RigidBodyId),
                    cudaMemcpyDeviceToHost);
                if (copied == cudaSuccess)
                    copied = cudaMemcpy(
                        dense_states_.data(), state_view.states.data,
                        dense_states_.size() * sizeof(RigidBodyState),
                        cudaMemcpyDeviceToHost);
                if (copied != cudaSuccess) {
                    return {StatusCode::cuda_failure, copied,
                            "fixed collector body readback failed"};
                }
                states_loaded = true;
            }
            const auto member_dense = std::find(
                dense_ids_.begin(), dense_ids_.end(),
                instance.rigid_bodies[member]);
            const auto target_dense = std::find(
                dense_ids_.begin(), dense_ids_.end(),
                instance.rigid_bodies[target]);
            if (member_dense == dense_ids_.end() ||
                target_dense == dense_ids_.end()) {
                return {StatusCode::invalid_handle, cudaSuccess,
                        "fixed collector body handle is stale"};
            }
            const RigidBodyState &member_state = dense_states_[
                static_cast<std::size_t>(member_dense - dense_ids_.begin())];
            const RigidBodyState &target_state = dense_states_[
                static_cast<std::size_t>(target_dense - dense_ids_.begin())];

            using namespace fixed_collector_detail;
            const Quaternion member_inverse = conjugate(member_state.orientation);
            const Quaternion target_inverse = conjugate(target_state.orientation);
            RigidConstraintOptions options{
                .type = RigidConstraintType::fixed,
                .body_a = instance.rigid_bodies[member],
                .body_b = instance.rigid_bodies[target],
                .local_anchor_a = rotate(
                    member_inverse,
                    subtract(contact.position, member_state.position)),
                .local_anchor_b = rotate(
                    target_inverse,
                    subtract(contact.position, target_state.position)),
                .local_orientation_a = {},
                .local_orientation_b = multiply(
                    target_inverse, member_state.orientation),
                .enabled = true,
                .disable_collisions = true,
                .solver_iterations = 16U};
            RigidConstraintId constraint{};
            Status status = world.add_rigid_constraint(options, constraint);
            if (!status) return status;
            try {
                generated_constraints_.push_back(constraint);
            } catch (...) {
                return {StatusCode::out_of_memory, cudaSuccess,
                        "fixed collector constraint tracking failed"};
            }
            members_[target] = 1U;
        }
        return {};
    }

  private:
    static constexpr std::size_t invalid_index =
        std::numeric_limits<std::size_t>::max();
    std::size_t collector_body_index_{invalid_index};
    std::vector<std::uint8_t> members_{};
    std::vector<RigidConstraintId> generated_constraints_{};
    mutable std::vector<RigidBodyId> gravity_bodies_{};
    std::vector<RigidBodyId> dense_ids_{};
    std::vector<RigidBodyState> dense_states_{};
};

} // namespace parallel_mater::gallery
