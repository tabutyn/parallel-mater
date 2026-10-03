// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/fixed_collector.hpp>
#include "assets/fixed_collector_capture.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace parallel_mater;
using namespace parallel_mater::gallery;
using namespace parallel_mater::gallery::fixed_collector_detail;

bool require(Status status, const char *operation) {
    if (status) return true;
    std::cerr << operation << ": " << status.message << '\n';
    return false;
}

struct Bounds {
    Vec3 minimum{1.0e30F, 1.0e30F, 1.0e30F};
    Vec3 maximum{-1.0e30F, -1.0e30F, -1.0e30F};
};

Bounds bounds(const SceneDefinition &scene, std::size_t index,
              const RigidBodyState &state) {
    Bounds result{};
    const auto &body = scene.rigid_bodies[index];
    const bool proxy = !body.collision_mesh_indices.empty();
    const auto &meshes = proxy ? scene.collision_meshes : scene.meshes;
    const auto &indices = proxy ? body.collision_mesh_indices : body.mesh_indices;
    for (auto mesh : indices) {
        for (const auto &vertex : meshes[mesh].vertices) {
            const Vec3 local = rotate(state.orientation, vertex.position);
            const Vec3 point{state.position.x + local.x,
                             state.position.y + local.y,
                             state.position.z + local.z};
            result.minimum.x = std::min(result.minimum.x, point.x);
            result.minimum.y = std::min(result.minimum.y, point.y);
            result.minimum.z = std::min(result.minimum.z, point.z);
            result.maximum.x = std::max(result.maximum.x, point.x);
            result.maximum.y = std::max(result.maximum.y, point.y);
            result.maximum.z = std::max(result.maximum.z, point.z);
        }
    }
    return result;
}

bool run_round_collector(SceneDefinition scene) {
    const auto large = std::find_if(scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const auto &body) { return body.source_name == "Large"; });
    if (large == scene.rigid_bodies.end()) return false;
    auto ball = *large;
    const bool proxy = !ball.collision_mesh_indices.empty();
    const auto &meshes = proxy ? scene.collision_meshes : scene.meshes;
    const auto &indices = proxy ? ball.collision_mesh_indices : ball.mesh_indices;
    const auto length = [](Vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); };
    float radius = 0.0F, minimum_radius = std::numeric_limits<float>::max();
    float support = std::numeric_limits<float>::max();
    for (auto mesh_index : indices) {
        const auto &mesh = meshes[mesh_index];
        for (const auto &vertex : mesh.vertices) {
            radius = std::max(radius, length(vertex.position));
            minimum_radius = std::min(minimum_radius, length(vertex.position));
        }
        for (std::size_t index = 0; index < mesh.indices.size(); index += 3) {
            const auto a = mesh.vertices[mesh.indices[index]].position;
            const auto b = subtract(mesh.vertices[mesh.indices[index + 1]].position, a);
            const auto c = subtract(mesh.vertices[mesh.indices[index + 2]].position, a);
            const Vec3 normal{b.y*c.z - b.z*c.y, b.z*c.x - b.x*c.z, b.x*c.y - b.y*c.x};
            support = std::min(support,
                std::fabs(normal.x*a.x + normal.y*a.y + normal.z*a.z) / length(normal));
        }
    }
    const auto ground = std::find_if(scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const auto &body) { return body.source_name == "Ground"; });
    if (ground == scene.rigid_bodies.end()) return false;
    const auto floor = *ground;
    ball.options.initial_state = {.position = {0.0F, radius + 0.001F, 0.0F},
        .linear_velocity = {0.65F, 0.0F, 0.0F},
        .angular_velocity = {0.0F, 0.0F, -0.65F / radius}};
    ball.options.linear_damping = ball.options.angular_damping = 0.0F;
    scene.rigid_bodies = {floor, ball};
    scene.rigid_constraints.clear();
    WorldOptions options{};
    World world;
    SceneInstance instance;
    if (!require(scene_world_options(scene, options), "round ball options") ||
        !require(World::create(options, world), "create round ball world") ||
        !require(instantiate_scene(scene, world, instance), "instantiate round ball"))
        return false;
    float low = radius, high = radius, vertical_speed = 0.0F;
    for (int frame = 0; frame < 240; ++frame) {
        if (!require(world.step({.timestep = 1.0F/60.0F, .substeps = 4U}),
                     "roll isolated collector")) return false;
        if (frame < 30) continue;
        RigidBodyState state{};
        if (!require(world.read_rigid_body_state(instance.rigid_bodies[1], state),
                     "read rolling collector")) return false;
        low = std::min(low, state.position.y);
        high = std::max(high, state.position.y);
        vertical_speed = std::max(vertical_speed, std::fabs(state.linear_velocity.y));
    }
    std::cout << "collector roundness radial_spread=" << radius - minimum_radius
              << " face_sag=" << radius - support << " rolling_height_range=" << high - low
              << " maximum_vertical_speed=" << vertical_speed << '\n';
    return proxy && radius - minimum_radius < 1.0e-4F && radius - support < 0.002F &&
           high - low < 0.004F && vertical_speed < 0.04F;
}

bool run_captured_assembly(SceneDefinition scene) {
    const auto definitions = scene.rigid_bodies;
    const auto ground = std::find_if(definitions.begin(), definitions.end(),
        [](const auto &body) { return body.source_name == "Ground"; });
    if (ground == definitions.end()) return false;
    scene.rigid_bodies = {*ground};
    scene.rigid_constraints.clear();
    for (const auto &pose : captured_collector_poses) {
        const auto body = std::find_if(definitions.begin(), definitions.end(),
            [&](const auto &item) { return item.source_name == pose.name; });
        if (body == definitions.end()) return false;
        scene.rigid_bodies.push_back(*body);
        scene.rigid_bodies.back().options.initial_state = pose.state;
        const auto index = static_cast<std::uint32_t>(scene.rigid_bodies.size() - 1);
        if (index == 1U) continue;
        const auto &parent = scene.rigid_bodies[1].options.initial_state;
        RigidConstraintDefinition joint{};
        joint.body_a = 1U;
        joint.body_b = index;
        joint.options.local_anchor_a = rotate(conjugate(parent.orientation),
                                             subtract(pose.state.position, parent.position));
        joint.options.local_orientation_b = multiply(conjugate(pose.state.orientation),
                                                    parent.orientation);
        joint.options.solver_iterations = 16U;
        scene.rigid_constraints.push_back(joint);
    }
    WorldOptions options{};
    World world;
    SceneInstance instance;
    if (!require(scene_world_options(scene, options), "captured assembly options") ||
        !require(World::create(options, world), "create captured assembly") ||
        !require(instantiate_scene(scene, world, instance), "instantiate captured assembly"))
        return false;
    std::vector<Vec3> inertia;
    float initial_energy = 0.0F;
    for (std::size_t index = 1; index < scene.rigid_bodies.size(); ++index) {
        const auto &body = scene.rigid_bodies[index];
        const auto shape = bounds(scene, index, {});
        const auto size = subtract(shape.maximum, shape.minimum);
        const auto mass = body.options.mass;
        inertia.push_back({mass * (size.y*size.y + size.z*size.z) / 12.0F,
                           mass * (size.x*size.x + size.z*size.z) / 12.0F,
                           mass * (size.x*size.x + size.y*size.y) / 12.0F});
        initial_energy += mass * 9.81F * body.options.initial_state.position.y;
    }
    float maximum_energy = initial_energy, minimum_clearance = 1.0e30F;
    std::size_t internal_contacts = 0U, ground_contacts = 0U;
    for (int frame = 0; frame < 360; ++frame) {
        if (!require(world.step({.timestep = 1.0F/60.0F, .substeps = 4U,
                                 .collect_rigid_contacts = true}),
                     "drop captured assembly")) return false;
        const auto view = world.rigid_contacts();
        std::vector<RigidContactEvent> events(view.event_count);
        if (!events.empty() && cudaMemcpy(events.data(), view.events.data,
            events.size() * sizeof(RigidContactEvent), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        for (const auto &event : events) {
            if (event.body == instance.rigid_bodies[0] ||
                event.collider == instance.rigid_bodies[0]) ++ground_contacts;
            else ++internal_contacts;
        }
        float energy = 0.0F;
        for (std::size_t index = 1; index < instance.rigid_bodies.size(); ++index) {
            RigidBodyState state{};
            if (!require(world.read_rigid_body_state(instance.rigid_bodies[index], state),
                         "read captured assembly")) return false;
            const auto v = state.linear_velocity;
            const auto w = rotate(conjugate(state.orientation), state.angular_velocity);
            const auto moment = inertia[index - 1];
            const float mass = scene.rigid_bodies[index].options.mass;
            energy += mass * (9.81F * state.position.y +
                                  0.5F * (v.x*v.x + v.y*v.y + v.z*v.z)) +
                0.5F * (moment.x*w.x*w.x + moment.y*w.y*w.y + moment.z*w.z*w.z);
            minimum_clearance = std::min(minimum_clearance,
                                          bounds(scene, index, state).minimum.y);
        }
        if (!std::isfinite(energy)) return false;
        maximum_energy = std::max(maximum_energy, energy);
    }
    std::cout << "captured assembly bodies=" << inertia.size()
              << " internal_contacts=" << internal_contacts
              << " ground_contacts=" << ground_contacts
              << " energy_ratio=" << maximum_energy / initial_energy
              << " minimum_clearance=" << minimum_clearance << '\n';
    return internal_contacts == 0U && ground_contacts > 0U &&
           maximum_energy < initial_energy * 1.02F && minimum_clearance > -0.003F;
}

bool run_collection(const SceneDefinition &scene) {
    World world;
    SceneInstance instance;
    WorldOptions options{};
    if (!require(scene_world_options(scene, options), "scene options")) return false;
    options.rigid_constraint_capacity = FixedContactCollector::constraint_capacity(scene);
    if (!require(World::create(options, world), "create world") ||
        !require(instantiate_scene(scene, world, instance), "instantiate scene"))
        return false;
    FixedContactCollector collector;
    if (!require(collector.initialize(scene, instance), "initialize collector")) return false;
    const auto ground = std::find_if(scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [](const auto &body) { return body.source_name == "Ground"; });
    if (ground == scene.rigid_bodies.end()) return false;
    const Bounds floor = bounds(scene, ground - scene.rigid_bodies.begin(),
                                ground->options.initial_state);
    float minimum_clearance = std::numeric_limits<float>::max();
    std::size_t worst_body{};
    int worst_frame{};
    bool escaped = false;
    // Roll onto the marbles, release input under full downward gravity, then
    // steer across the first track. Clearance is checked during motion and rest.
    for (int frame = 0; frame < 620; ++frame) {
        Vec3 gravity{0.0F, -9.81F, 0.0F};
        if (frame >= 60 && frame < 105) gravity = {9.660965F, -1.703488F, 0.0F};
        if (frame >= 345 && frame < 380) gravity = {0.0F, -1.703488F, 9.660965F};
        if (!require(collector.apply_loose_gravity(world, scene, instance,
                         {0.0F, -9.81F, 0.0F}, gravity), "loose gravity") ||
            !require(world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                                 .gravity = gravity, .collect_rigid_contacts = true}),
                     "step collector") ||
            !require(collector.collect(world, scene, instance), "collect contacts"))
            return false;
        for (std::size_t index = 0; index < instance.rigid_bodies.size(); ++index) {
            if (scene.rigid_bodies[index].options.motion != MotionType::dynamic) continue;
            RigidBodyState state{};
            if (!require(world.read_rigid_body_state(instance.rigid_bodies[index], state),
                         "read collector body")) return false;
            const Bounds shape = bounds(scene, index, state);
            escaped = escaped || shape.minimum.x < floor.minimum.x ||
                shape.maximum.x > floor.maximum.x || shape.minimum.z < floor.minimum.z ||
                shape.maximum.z > floor.maximum.z;
            const float clearance = shape.minimum.y - floor.maximum.y;
            if (clearance < minimum_clearance) {
                minimum_clearance = clearance;
                worst_body = index;
                worst_frame = frame;
            }
        }
    }
    std::cout << "fixed collector attached=" << collector.attached_count()
              << " minimum_clearance=" << minimum_clearance
              << " body=" << scene.rigid_bodies[worst_body].source_name
              << " frame=" << worst_frame << " escaped=" << escaped << '\n';
    return collector.generated_constraint_count() > 0U && !escaped &&
           minimum_clearance > -0.003F;
}
} // namespace

int main(int argc, char **argv) {
    int count{};
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    SceneDefinition scene;
    std::string error;
    if (!load_glb_scene(argc > 1 ? argv[1] : PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    // Extend only the test's flat floor so rolling off the gallery platform
    // cannot be confused with tunneling through its top surface.
    for (const auto &body : scene.rigid_bodies) {
        if (body.source_name != "Ground") continue;
        for (auto mesh : body.mesh_indices) {
            for (auto &vertex : scene.meshes[mesh].vertices) {
                vertex.position.x *= 10.0F;
                vertex.position.z *= 10.0F;
            }
        }
    }
    const bool round = run_round_collector(scene);
    const bool collected = run_collection(scene);
    const bool captured = run_captured_assembly(scene);
    return round && collected && captured ? 0 : 1;
}
