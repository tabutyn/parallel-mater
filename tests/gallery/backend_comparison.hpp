// SPDX-License-Identifier: MIT
#pragma once
#include <scene_builder.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <iostream>
#include <string>
#include <tuple>

namespace parallel_mater::gallery::test {

// Numerical conformance, not a visual/finite-value smoke test. These tolerances
// are deliberately fixed so a new backend cannot silently relax acceptance.
inline bool compare_frames(const CommandSceneFrame &actual,
                           const CommandSceneFrame &reference, std::string &error,
                           bool trace = false) {
    const auto scalar = [&](float a, float b, float absolute, const std::string &field) {
        if (std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= absolute) return true;
        error = field + ": actual=" + std::to_string(a) + " reference=" + std::to_string(b);
        return false;
    };
    const auto vector = [&](const auto &a, const auto &b, float tolerance,
                            const std::string &field) {
        return scalar(a.x, b.x, tolerance, field + ".x") &&
               scalar(a.y, b.y, tolerance, field + ".y") &&
               scalar(a.z, b.z, tolerance, field + ".z");
    };
    const auto count = [&](std::size_t a, std::size_t b, const char *field) {
        if (a == b) return true;
        error = std::string(field) + " count differs: " + std::to_string(a) + " / " + std::to_string(b);
        return false;
    };
    if (!count(actual.rigid_states.size(), reference.rigid_states.size(), "rigid")) return false;
    if (trace) {
        float position{}, velocity{}, orientation{};
        for (std::size_t body = 0U; body < actual.rigid_states.size(); ++body) {
            const auto &a = actual.rigid_states[body], &b = reference.rigid_states[body];
            for (unsigned axis = 0U; axis < 3U; ++axis) {
                position = std::max(position, std::abs(a.position[axis] - b.position[axis]));
                velocity = std::max(velocity, std::abs(a.linearVelocity[axis] - b.linearVelocity[axis]));
            }
            for (unsigned axis = 0U; axis < 4U; ++axis)
                orientation = std::max(orientation, std::abs(a.orientation[axis] - b.orientation[axis]));
        }
        std::cout << "comparison revision=" << actual.revision << " position=" << position
                  << " velocity=" << velocity << " orientation=" << orientation
                  << " contacts=" << actual.contacts.size() << '/' << reference.contacts.size() << '\n';
    }
    for (std::size_t body = 0; body < actual.rigid_states.size(); ++body) {
        const auto &a = actual.rigid_states[body], &b = reference.rigid_states[body];
        const auto name = "rigid[" + std::to_string(body) + "]";
        for (unsigned axis = 0; axis < 3; ++axis)
            if (!scalar(a.position[axis], b.position[axis], 0.005F, name + ".position") ||
                !scalar(a.linearVelocity[axis], b.linearVelocity[axis], 0.05F, name + ".velocity") ||
                !scalar(a.angularVelocity[axis], b.angularVelocity[axis], 0.05F, name + ".angularVelocity"))
                return false;
        float dot = 0.0F;
        for (unsigned axis = 0; axis < 4; ++axis) dot += a.orientation[axis] * b.orientation[axis];
        const float sign = dot < 0.0F ? -1.0F : 1.0F;
        for (unsigned axis = 0; axis < 4; ++axis)
            if (!scalar(a.orientation[axis], sign * b.orientation[axis], 0.005F, name + ".orientation"))
                return false;
    }
    if (!count(actual.constraint_states.size(), reference.constraint_states.size(), "constraint")) return false;
    for (std::size_t index = 0; index < actual.constraint_states.size(); ++index) {
        const auto &a = actual.constraint_states[index], &b = reference.constraint_states[index];
        if (a.enabled != b.enabled || a.broken != b.broken) {
            error = "constraint activation/breakage differs"; return false;
        }
        if (!scalar(a.appliedImpulse, b.appliedImpulse, 0.05F, "constraint impulse")) return false;
    }
    auto fluid_a = actual.fluid_particles, fluid_b = reference.fluid_particles;
    const auto by_id = [](const auto &a, const auto &b) { return a.stableId < b.stableId; };
    std::sort(fluid_a.begin(), fluid_a.end(), by_id);
    std::sort(fluid_b.begin(), fluid_b.end(), by_id);
    if (!count(fluid_a.size(), fluid_b.size(), "fluid")) return false;
    for (std::size_t index = 0; index < fluid_a.size(); ++index) {
        const auto &a = fluid_a[index], &b = fluid_b[index];
        if (a.stableId != b.stableId) { error = "fluid stable IDs differ"; return false; }
        for (unsigned axis = 0; axis < 3; ++axis)
            if (!scalar(a.position[axis], b.position[axis], 0.005F, "fluid position") ||
                !scalar(a.velocity[axis], b.velocity[axis], 0.05F, "fluid velocity")) return false;
    }
    if (!count(actual.cloth_vertices.size(), reference.cloth_vertices.size(), "cloth")) return false;
    for (std::size_t cloth = 0; cloth < actual.cloth_vertices.size(); ++cloth) {
        if (!count(actual.cloth_vertices[cloth].size(), reference.cloth_vertices[cloth].size(), "cloth vertex")) return false;
        for (std::size_t index = 0; index < actual.cloth_vertices[cloth].size(); ++index) {
            const auto &a = actual.cloth_vertices[cloth][index], &b = reference.cloth_vertices[cloth][index];
            if (!vector(a.position, b.position, 0.005F, "cloth position") ||
                !vector(a.velocity, b.velocity, 0.05F, "cloth velocity")) return false;
        }
    }
    // Atomic manifold order may differ. Compare every active body-pair's total
    // normal impulse and maximum penetration, independent of event ordering.
    using Key = std::tuple<int, std::uint32_t, std::uint32_t>;
    struct Contact { float impulse{}, penetration{}; };
    const auto contacts = [](const auto &events) {
        std::map<Key, Contact> result;
        for (const auto &event : events) {
            auto first = event.first, second = event.second;
            if (event.type == PM_CONTACT_RIGID_RIGID && first > second) std::swap(first, second);
            auto &value = result[{event.type, first, second}];
            value.impulse += event.normalImpulse;
            value.penetration = std::max(value.penetration, event.penetration);
        }
        return result;
    };
    const auto contact_a = contacts(actual.contacts), contact_b = contacts(reference.contacts);
    if (!count(contact_a.size(), contact_b.size(), "contact pair")) return false;
    for (const auto &[key, a] : contact_a) {
        const auto found = contact_b.find(key);
        if (found == contact_b.end()) { error = "contact body pair differs"; return false; }
        const auto name = "contact[" + std::to_string(std::get<0>(key)) + ":" +
            std::to_string(std::get<1>(key)) + "," + std::to_string(std::get<2>(key)) + "]";
        if (!scalar(a.impulse, found->second.impulse, 0.05F, name + " impulse") ||
            !scalar(a.penetration, found->second.penetration, 0.005F, name + " penetration")) return false;
    }
    return true;
}
} // namespace parallel_mater::gallery::test
