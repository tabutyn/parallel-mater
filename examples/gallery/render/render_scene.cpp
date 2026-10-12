// SPDX-License-Identifier: MIT
#include "render_scene.hpp"

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

} // namespace parallel_mater::gallery
