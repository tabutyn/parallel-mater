// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>
#include <scene_builder.hpp>
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
    static constexpr float maximum_tracking_correction =
        8.0F * 3.14159265358979323846F / 180.0F;
    static constexpr float maximum_return_correction =
        60.0F * 3.14159265358979323846F / 180.0F;
    static constexpr float tracking_gain = 3.0F;
    static constexpr float return_tracking_gain = 10.0F;

    [[nodiscard]] bool initialize(const SceneDefinition &scene) noexcept {
        *this = {};
        for (std::size_t i = 0; i < scene.rigid_constraints.size(); ++i) {
            const auto &joint = scene.rigid_constraints[i];
            if (joint.name != "DumpLift") continue;
            if (joint.options.type != RigidConstraintType::generic) return false;
            if (joint.body_a >= scene.rigid_bodies.size() ||
                joint.body_b >= scene.rigid_bodies.size()) return false;
            index_ = i;
            body_a_ = joint.body_a;
            body_b_ = joint.body_b;
            authored_orientation_ = joint.options.local_orientation_a;
            authored_relative_orientation_ = multiply(
                conjugate(scene.rigid_bodies[body_a_]
                              .options.initial_state.orientation),
                scene.rigid_bodies[body_b_]
                    .options.initial_state.orientation);
            return true;
        }
        return false;
    }
    void toggle() noexcept {
        raised_ = !raised_;
        returning_ = !raised_;
    }
    [[nodiscard]] float angle() const noexcept { return angle_; }
    [[nodiscard]] bool raised() const noexcept { return raised_; }

    [[nodiscard]] bool advance(CommandScene &simulation,
                               SceneDefinition &scene,
                               const CommandSceneInstance &instance,
                               const CommandSceneFrame &frame,
                               float timestep, std::string &error) noexcept {
        if (index_ >= scene.rigid_constraints.size() ||
            index_ >= instance.rigid_constraints.size()) {
            error = "dump lift bindings are missing";
            return false;
        }
        const float next = angle_ + std::clamp(
            (raised_ ? maximum_angle : 0.0F) - angle_,
            -angular_speed * timestep, angular_speed * timestep);
        float command = next;
        float observed{};
        if ((raised_ || returning_) && observed_angle(frame, observed)) {
            const float correction_limit = raised_
                ? maximum_tracking_correction : maximum_return_correction;
            const float gain = raised_ ? tracking_gain : return_tracking_gain;
            command += std::clamp(
                gain * (next - observed),
                -correction_limit, correction_limit);
        }
        if (next == angle_ && std::fabs(command - command_angle_) < 1.0e-6F)
            return true;
        auto &joint = scene.rigid_constraints[index_];
        auto updated = joint;
        const float s = std::sin(command * 0.5F);
        const float c = std::cos(command * 0.5F);
        const auto q = authored_orientation_;
        updated.options.local_orientation_a = {
            q.x*c+q.y*s, q.y*c-q.x*s, q.z*c+q.w*s, q.w*c-q.z*s};
        if (!simulation.write_constraint(
                instance.rigid_constraints[index_], updated, error))
            return false;
        joint.options = updated.options;
        angle_ = next;
        command_angle_ = command;
        return true;
    }

  private:
    [[nodiscard]] static Quaternion conjugate(Quaternion value) noexcept {
        return {-value.x, -value.y, -value.z, value.w};
    }

    [[nodiscard]] static Quaternion multiply(
        Quaternion a, Quaternion b) noexcept {
        return {
            a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
            a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
            a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
            a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
    }

    [[nodiscard]] bool observed_angle(
        const CommandSceneFrame &frame, float &value) const noexcept {
        if (body_a_ >= frame.rigid_states.size() ||
            body_b_ >= frame.rigid_states.size()) return false;
        const auto orientation = [&](std::size_t index) {
            const auto &state = frame.rigid_states[index];
            return Quaternion{state.orientation[0], state.orientation[1],
                              state.orientation[2], state.orientation[3]};
        };
        const Quaternion relative = multiply(
            conjugate(orientation(body_a_)), orientation(body_b_));
        const Quaternion delta = multiply(
            conjugate(authored_relative_orientation_), relative);
        value = 2.0F * std::atan2(
            std::sqrt(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z),
            std::fabs(delta.w));
        return std::isfinite(value);
    }

    std::size_t index_{std::numeric_limits<std::size_t>::max()};
    std::size_t body_a_{std::numeric_limits<std::size_t>::max()};
    std::size_t body_b_{std::numeric_limits<std::size_t>::max()};
    Quaternion authored_orientation_{};
    Quaternion authored_relative_orientation_{};
    float angle_{};
    float command_angle_{};
    bool raised_{};
    bool returning_{};
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
    for (auto &bucket : scene.rigid_bodies) {
        if (bucket.source_name != "DumpBucket") continue;
        for (const std::uint32_t mesh_index : bucket.mesh_indices) {
            auto &mesh = scene.meshes[mesh_index];
            if (mesh.name.ends_with("-sloped-liner")) continue;
            Vec3 minimum{std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
            Vec3 maximum{std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};
            for (const auto &vertex : mesh.vertices) {
                minimum.x = std::min(minimum.x, vertex.position.x);
                minimum.y = std::min(minimum.y, vertex.position.y);
                minimum.z = std::min(minimum.z, vertex.position.z);
                maximum.x = std::max(maximum.x, vertex.position.x);
                maximum.y = std::max(maximum.y, vertex.position.y);
                maximum.z = std::max(maximum.z, vertex.position.z);
            }
            constexpr float wall_thickness = 0.14F;
            constexpr float liner_run = 1.30F;
            const float inner = maximum.x - wall_thickness;
            const float height = maximum.y - minimum.y;
            for (auto &vertex : mesh.vertices) {
                if (vertex.position.x < inner - 1.0e-4F) continue;
                const float layer = std::clamp(
                    (vertex.position.x - inner) / wall_thickness,
                    0.0F, 1.0F);
                const float vertical = std::clamp(
                    (vertex.position.y - minimum.y) / height,
                    0.0F, 1.0F);
                const float sloped_inner = inner - liner_run *
                    (1.0F - vertical);
                vertex.position.x =
                    sloped_inner + layer * wall_thickness;
            }
            mesh.name += "-sloped-liner";
        }
    }
    float radius{};
    for (const auto index : sphere.mesh_indices)
        for (const auto &vertex : scene.meshes[index].vertices)
            radius = std::max(radius, std::sqrt(vertex.position.x*vertex.position.x +
                vertex.position.y*vertex.position.y + vertex.position.z*vertex.position.z));
    const float spacing = 2.3F*radius;
    if (!(radius > 0) || !std::isfinite(radius)) {
        error = "SphereCluster needs a finite sphere template"; return false;
    }
    unsigned width = 1;
    while (width*width*width < count) ++width;
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
        // Match the authored low-friction bed liner so payload slides through
        // the open top when the bed reaches its target.
        body.options.friction = std::min(body.options.friction, 0.02F);
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
