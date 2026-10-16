// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include <algorithm>

namespace parallel_mater::gallery {

bool apply_liquid_workload_limits(SceneDefinition &scene,
    const GalleryWorkloadLimits &limits, std::string &error) {
    if (limits.fluid_particles == 0 || limits.coupled_bodies == 0) {
        error = "Liquid workload budgets must be positive";
        return false;
    }
    const auto dynamic = [](const RigidBodyDefinition &body) {
        return body.options.motion == MotionType::dynamic;
    };
    const auto count = std::count_if(scene.rigid_bodies.begin(),
                                     scene.rigid_bodies.end(), dynamic);
    if (count > limits.coupled_bodies) {
        if (!scene.rigid_constraints.empty() || !scene.ropes.empty() ||
            !scene.cloths.empty() || !scene.soft_bodies.empty()) {
            error = "Cannot reduce bodies in a coupled scene with indexed attachments";
            return false;
        }
        // Keep a deterministic subset spanning the authored population.
        // Only body instances are removed; shared geometry/materials stay valid.
        std::uint64_t ordinal = 0, kept = 0;
        std::erase_if(scene.rigid_bodies, [&](const RigidBodyDefinition &body) {
            if (!dynamic(body)) return false;
            const bool keep = ordinal == kept * static_cast<std::uint64_t>(count) /
                                              limits.coupled_bodies;
            ++ordinal;
            if (keep) ++kept;
            return !keep;
        });
    }
    scene.fluid_options.capacity = limits.fluid_particles;
    error.clear();
    return true;
}

} // namespace parallel_mater::gallery
