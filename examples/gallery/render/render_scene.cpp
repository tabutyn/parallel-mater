// SPDX-License-Identifier: MIT
#include "render_scene.hpp"

#include <algorithm>
#include <cmath>
#include <new>

namespace parallel_mater::gallery {
namespace {

[[nodiscard]] Vec3 value(PmFloat3 source) noexcept {
    return {source.x, source.y, source.z};
}

[[nodiscard]] RigidBodyState value(const PmRigidState &source) noexcept {
    return {{source.position[0], source.position[1], source.position[2]},
            {source.orientation[0], source.orientation[1],
             source.orientation[2], source.orientation[3]},
            {source.linearVelocity[0], source.linearVelocity[1],
             source.linearVelocity[2]},
            {source.angularVelocity[0], source.angularVelocity[1],
             source.angularVelocity[2]}};
}

template <typename Source, typename Destination, typename Convert>
void convert_nested(const std::vector<std::vector<Source>> &source,
                    std::vector<std::vector<Destination>> &destination,
                    Convert convert) {
    destination.resize(source.size());
    for (std::size_t outer = 0U; outer < source.size(); ++outer) {
        destination[outer].reserve(source[outer].size());
        for (const Source &item : source[outer])
            destination[outer].push_back(convert(item));
    }
}

[[nodiscard]] float blend(float previous, float current, float alpha) noexcept {
    return previous + (current - previous) * std::clamp(alpha, 0.0F, 1.0F);
}

[[nodiscard]] Vec3 blend(Vec3 previous, Vec3 current, float alpha) noexcept {
    return {blend(previous.x, current.x, alpha),
            blend(previous.y, current.y, alpha),
            blend(previous.z, current.z, alpha)};
}

[[nodiscard]] Quaternion blend_orientation(Quaternion previous,
                                           Quaternion current,
                                           float alpha) noexcept {
    if (previous.x * current.x + previous.y * current.y +
            previous.z * current.z + previous.w * current.w < 0.0F) {
        previous = {-previous.x, -previous.y, -previous.z, -previous.w};
    }
    Quaternion result{
        blend(previous.x, current.x, alpha),
        blend(previous.y, current.y, alpha),
        blend(previous.z, current.z, alpha),
        blend(previous.w, current.w, alpha)};
    const float magnitude = std::sqrt(
        result.x * result.x + result.y * result.y +
        result.z * result.z + result.w * result.w);
    if (magnitude <= 1.0e-8F) return current;
    result.x /= magnitude;
    result.y /= magnitude;
    result.z /= magnitude;
    result.w /= magnitude;
    return result;
}

} // namespace

bool make_render_scene_frame(const SceneDefinition &scene,
                             const CommandSceneFrame &source,
                             RenderSceneFrame &output, std::string &error) {
    error.clear();
    try {
        output = {};
        output.rigid_states.reserve(source.rigid_states.size());
        for (const auto &state : source.rigid_states)
            output.rigid_states.push_back(value(state));
        output.previous_rigid_states.reserve(source.previous_rigid_states.size());
        for (const auto &state : source.previous_rigid_states)
            output.previous_rigid_states.push_back(value(state));
        output.fluid_particles.reserve(source.fluid_particles.size());
        for (const auto &particle : source.fluid_particles)
            output.fluid_particles.push_back({
                {particle.position[0], particle.position[1], particle.position[2]},
                {particle.velocity[0], particle.velocity[1], particle.velocity[2]},
                {particle.acceleration[0], particle.acceleration[1],
                 particle.acceleration[2]},
                particle.temperatureCelsius, particle.foam,
                particle.stableId});
        convert_nested(source.cloth_vertices, output.cloth_vertices,
            [](const PmClothVertexState &vertex) { return value(vertex.position); });
        convert_nested(source.cloth_surface_vertices,
                       output.cloth_surface_vertices,
                       [](PmFloat3 vertex) { return value(vertex); });
        convert_nested(source.rope_nodes, output.rope_nodes,
            [](const PmRopeNodeState &node) { return value(node.position); });
        convert_nested(source.soft_body_nodes, output.soft_body_nodes,
            [](const PmSoftBodyNodeState &node) { return value(node.position); });
        convert_nested(source.soft_body_surface_vertices,
                       output.soft_body_surface_vertices,
                       [](PmFloat3 vertex) { return value(vertex); });
        output.smoke_particles.reserve(source.smoke_particles.size());
        for (const auto &particle : source.smoke_particles)
            output.smoke_particles.push_back({
                value(particle.position), value(particle.velocity), particle.age,
                particle.numberDensity, particle.pressure,
                value(particle.vorticity)});
        output.paint_fields = source.paint_fields;
        output.fluid_particle_radius = scene.fluid_options.particle_radius;
        output.fluid_support_radius = scene.fluid_options.support_radius;
        output.smoke_particle_radius = scene.smoke_options.particle_radius;
        output.smoke_lifetime = scene.smoke_options.lifetime;
        output.revision = source.revision;
    } catch (const std::bad_alloc &) {
        error = "build renderer frame: out of host memory";
        return false;
    }
    return true;
}

void reset_render_scene_snapshots(RenderSceneFrame &frame) {
    frame.previous_rigid_states = frame.rigid_states;
    frame.previous_fluid_particles = frame.fluid_particles;
    frame.previous_cloth_vertices = frame.cloth_vertices;
    frame.previous_cloth_surface_vertices = frame.cloth_surface_vertices;
    frame.previous_rope_nodes = frame.rope_nodes;
    frame.previous_soft_body_nodes = frame.soft_body_nodes;
    frame.previous_soft_body_surface_vertices =
        frame.soft_body_surface_vertices;
    frame.previous_smoke_particles = frame.smoke_particles;
    frame.previous_revision = frame.revision;
    frame.previous_snapshot_available = true;
}

void advance_render_scene_snapshots(const RenderSceneFrame &previous,
                                    RenderSceneFrame &current) {
    if (!previous.previous_snapshot_available) {
        reset_render_scene_snapshots(current);
        return;
    }
    current.previous_rigid_states = previous.rigid_states;
    current.previous_fluid_particles = previous.fluid_particles;
    current.previous_cloth_vertices = previous.cloth_vertices;
    current.previous_cloth_surface_vertices = previous.cloth_surface_vertices;
    current.previous_rope_nodes = previous.rope_nodes;
    current.previous_soft_body_nodes = previous.soft_body_nodes;
    current.previous_soft_body_surface_vertices =
        previous.soft_body_surface_vertices;
    current.previous_smoke_particles = previous.smoke_particles;
    current.previous_revision = previous.revision;
    current.previous_snapshot_available = true;
}

RigidBodyState presentation_rigid_state(const RenderSceneFrame &frame,
                                        std::size_t index,
                                        float alpha) noexcept {
    if (index >= frame.rigid_states.size()) return {};
    RigidBodyState result = frame.rigid_states[index];
    if (!frame.previous_snapshot_available ||
        index >= frame.previous_rigid_states.size()) return result;
    const RigidBodyState &previous = frame.previous_rigid_states[index];
    result.position = blend(previous.position, result.position, alpha);
    result.orientation = blend_orientation(
        previous.orientation, result.orientation, alpha);
    return result;
}

Vec3 presentation_fluid_position(const RenderSceneFrame &frame,
                                 std::size_t index,
                                 float alpha) noexcept {
    if (index >= frame.fluid_particles.size()) return {};
    const RenderFluidParticle &current = frame.fluid_particles[index];
    if (!frame.previous_snapshot_available) return current.position;
    auto previous = frame.previous_fluid_particles.end();
    if (index < frame.previous_fluid_particles.size() &&
        frame.previous_fluid_particles[index].stable_id == current.stable_id)
        previous = frame.previous_fluid_particles.begin() + index;
    else previous = std::find_if(
        frame.previous_fluid_particles.begin(),
        frame.previous_fluid_particles.end(),
        [&](const RenderFluidParticle &candidate) {
            return candidate.stable_id == current.stable_id;
        });
    return previous == frame.previous_fluid_particles.end()
        ? current.position : blend(previous->position, current.position, alpha);
}

Vec3 presentation_domain_position(
    const std::vector<std::vector<Vec3>> &previous,
    const std::vector<std::vector<Vec3>> &current,
    std::size_t domain, std::size_t index, float alpha) noexcept {
    if (domain >= current.size() || index >= current[domain].size()) return {};
    // Without generation-tagged element identities, a topology change snaps
    // the entire affected domain to current rather than blending stale slots.
    if (domain >= previous.size() ||
        previous[domain].size() != current[domain].size())
        return current[domain][index];
    return blend(previous[domain][index], current[domain][index], alpha);
}

Vec3 presentation_smoke_position(const RenderSceneFrame &frame,
                                 std::size_t index,
                                 float alpha) noexcept {
    if (index >= frame.smoke_particles.size()) return {};
    // Legacy smoke output has no stable particle ID. Equal cardinality is the
    // strongest safe identity contract; spawn/removal snaps the whole domain.
    if (!frame.previous_snapshot_available ||
        frame.previous_smoke_particles.size() != frame.smoke_particles.size())
        return frame.smoke_particles[index].position;
    return blend(frame.previous_smoke_particles[index].position,
                 frame.smoke_particles[index].position, alpha);
}

} // namespace parallel_mater::gallery
