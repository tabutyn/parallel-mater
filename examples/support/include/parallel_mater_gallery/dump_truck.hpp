// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace parallel_mater::gallery {

// Gallery-only actuator: the public solver moves a dynamic bucket around its
// authored rear pivot, carrying reaction forces through the dynamic chassis.
class DumpTruckBed {
  public:
    static constexpr float maximum_angle = 100.0F * 3.14159265358979323846F / 180.0F;
    static constexpr float angular_speed = 30.0F * 3.14159265358979323846F / 180.0F;

    [[nodiscard]] bool initialize(const SceneDefinition &scene) noexcept {
        *this = {};
        for (std::size_t i = 0; i < scene.rigid_constraints.size(); ++i) {
            const auto &joint = scene.rigid_constraints[i];
            if (joint.name != "DumpLift") continue;
            if (joint.options.type != RigidConstraintType::generic) return false;
            index_ = i;
            authored_orientation_ = joint.options.local_orientation_a;
            return true;
        }
        return false;
    }
    void toggle() noexcept { raised_ = !raised_; }
    [[nodiscard]] float angle() const noexcept { return angle_; }
    [[nodiscard]] bool raised() const noexcept { return raised_; }

    [[nodiscard]] Status advance(World &world, SceneDefinition &scene,
                                 const SceneInstance &instance, float timestep) noexcept {
        if (index_ >= scene.rigid_constraints.size() || index_ >= instance.rigid_constraints.size())
            return {StatusCode::invalid_argument, cudaSuccess, "dump lift bindings are missing"};
        const float next = angle_ + std::clamp((raised_ ? maximum_angle : 0.0F) - angle_,
                                              -angular_speed*timestep, angular_speed*timestep);
        if (next == angle_) return {};
        auto &joint = scene.rigid_constraints[index_];
        auto options = joint.options;
        const float s = std::sin(next*0.5F), c = std::cos(next*0.5F);
        const auto q = authored_orientation_;
        options.local_orientation_a = {q.x*c+q.y*s, q.y*c-q.x*s, q.z*c+q.w*s, q.w*c-q.z*s};
        options.body_a = instance.rigid_bodies[joint.body_a];
        options.body_b = instance.rigid_bodies[joint.body_b];
        const auto status = world.update_rigid_constraint(instance.rigid_constraints[index_], options);
        if (status) { joint.options = options; angle_ = next; }
        return status;
    }

  private:
    std::size_t index_{std::numeric_limits<std::size_t>::max()};
    Quaternion authored_orientation_{};
    float angle_{};
    bool raised_{};
};

[[nodiscard]] inline std::uint32_t dump_payload_count(const SceneDefinition &scene) {
    return static_cast<std::uint32_t>(std::count_if(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const auto &body) { return body.source_name == "DumpPayload"; }));
}

inline constexpr std::uint32_t default_dump_payload_count = 100U;

// The Blender Empty owns the grid's center and orientation. Every count uses
// that same anchor, including repeated P edits; no load-volume helper is used.
[[nodiscard]] inline bool configure_dump_payload(SceneDefinition &scene,
                                                 std::uint32_t count,
                                                 std::string &error) {
    if (count < 10 || count > 1000) {
        error = "dump payload count must be 10..1000"; return false;
    }
    const auto prototype = std::find_if(scene.sphere_clusters.begin(), scene.sphere_clusters.end(),
        [](const auto &body) { return body.name == "SphereCluster"; });
    if (prototype == scene.sphere_clusters.end()) {
        error = "dump truck needs an exported Empty named SphereCluster"; return false;
    }
    const auto sphere = *prototype;
    float radius{};
    for (const auto index : sphere.mesh_indices)
        for (const auto &vertex : scene.meshes[index].vertices)
            radius = std::max(radius, std::sqrt(vertex.position.x*vertex.position.x +
                vertex.position.y*vertex.position.y + vertex.position.z*vertex.position.z));
    unsigned width = 1;
    while (width*width*width < count) ++width;
    const float spacing = 2.3F*radius;
    if (!(radius > 0) || !std::isfinite(radius)) {
        error = "SphereCluster needs a finite sphere template"; return false;
    }
    const unsigned depth = std::min(width, (count+width-1)/width);
    const unsigned height = (count+width*depth-1)/(width*depth);
    std::vector<RigidBodyDefinition> bodies;
    std::vector<std::uint32_t> remap(scene.rigid_bodies.size(), UINT32_MAX);
    bodies.reserve(scene.rigid_bodies.size()+count);
    for (std::size_t i = 0; i < scene.rigid_bodies.size(); ++i) {
        if (scene.rigid_bodies[i].source_name == "DumpPayload") continue;
        remap[i] = static_cast<std::uint32_t>(bodies.size());
        bodies.push_back(scene.rigid_bodies[i]);
    }
    auto joints = scene.rigid_constraints;
    for (auto &joint : joints) {
        if (joint.body_a >= remap.size() || joint.body_b >= remap.size() ||
            remap[joint.body_a] == UINT32_MAX || remap[joint.body_b] == UINT32_MAX) {
            error = "dump payload prototype cannot be constrained"; return false;
        }
        joint.body_a = remap[joint.body_a];
        joint.body_b = remap[joint.body_b];
    }
    for (unsigned i = 0; i < count; ++i) {
        auto body = sphere;
        body.name = "DumpPayload" + std::to_string(i);
        body.source_name = "DumpPayload";
        // Center the full occupied bounding box, including a partial top layer.
        const Vec3 local{(float(i%width)-float(width-1)*0.5F)*spacing,
                         (float(i/(width*depth))-float(height-1)*0.5F)*spacing,
                         (float((i/width)%depth)-float(depth-1)*0.5F)*spacing};
        const auto q = sphere.options.initial_state.orientation;
        const Vec3 t{2*(q.y*local.z-q.z*local.y), 2*(q.z*local.x-q.x*local.z), 2*(q.x*local.y-q.y*local.x)};
        const Vec3 rotated{local.x+q.w*t.x+q.y*t.z-q.z*t.y,
                           local.y+q.w*t.y+q.z*t.x-q.x*t.z,
                           local.z+q.w*t.z+q.x*t.y-q.y*t.x};
        const auto center = sphere.options.initial_state.position;
        body.options.initial_state.position = {center.x+rotated.x,
            center.y+rotated.y, center.z+rotated.z};
        bodies.push_back(std::move(body));
    }
    scene.rigid_bodies = std::move(bodies);
    scene.rigid_constraints = std::move(joints);
    return true;
}
} // namespace parallel_mater::gallery
