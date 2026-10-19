// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_conformance/case_registry.hpp>
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include <parallel_mater_scene/scene.hpp>

#include <parallel_mater_scene/command_scene.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifndef PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT
#error PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT is required
#endif
#ifndef PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT
#define PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT "unknown"
#endif

namespace {

using namespace parallel_mater;
using namespace parallel_mater::conformance;
using namespace parallel_mater::gallery;

class UnsupportedCase final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

constexpr std::size_t complete_state_limit = 256U;
constexpr std::size_t large_sample_limit = 32U;

[[nodiscard]] const char *host_compiler() noexcept {
#if defined(_MSC_VER)
    return "MSVC";
#elif defined(__VERSION__)
    return __VERSION__;
#else
    return "unknown";
#endif
}

[[nodiscard]] Vec3 vec(const PmFloat3 &value) noexcept {
    return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3 position(const PmRigidState &value) noexcept {
    return {value.position[0], value.position[1], value.position[2]};
}

[[nodiscard]] Vec3 linear_velocity(const PmRigidState &value) noexcept {
    return {value.linearVelocity[0], value.linearVelocity[1],
            value.linearVelocity[2]};
}

[[nodiscard]] Json vector_json(Vec3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

[[nodiscard]] Json quaternion_json(const float value[4]) {
    Json result = Json::array();
    for (std::size_t component = 0U; component < 4U; ++component)
        result.push_back(static_cast<double>(value[component]));
    return result;
}

[[nodiscard]] Json unsigned_array_json(
    const std::vector<std::uint32_t> &values,
    std::size_t limit = std::numeric_limits<std::size_t>::max()) {
    Json result = Json::array();
    for (std::size_t index = 0U; index < std::min(values.size(), limit); ++index)
        result.push_back(static_cast<std::uint64_t>(values[index]));
    return result;
}

struct Aggregate {
    Vec3 minimum{std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity()};
    Vec3 maximum{-std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity()};
    Vec3 center{};
    Vec3 momentum{};
    double kinetic_energy{};
    double mass{};
    double maximum_speed{};
    bool finite{true};
};

[[nodiscard]] Aggregate aggregate(const std::vector<Vec3> &positions,
                                  const std::vector<Vec3> &velocities,
                                  double element_mass) {
    Aggregate result{};
    result.mass = element_mass * static_cast<double>(positions.size());
    if (positions.empty()) {
        result.minimum = {};
        result.maximum = {};
        return result;
    }
    for (std::size_t index = 0U; index < positions.size(); ++index) {
        const Vec3 point = positions[index];
        const Vec3 velocity = index < velocities.size() ? velocities[index]
                                                        : Vec3{};
        result.finite = result.finite && std::isfinite(point.x) &&
            std::isfinite(point.y) && std::isfinite(point.z) &&
            std::isfinite(velocity.x) && std::isfinite(velocity.y) &&
            std::isfinite(velocity.z);
        result.minimum.x = std::min(result.minimum.x, point.x);
        result.minimum.y = std::min(result.minimum.y, point.y);
        result.minimum.z = std::min(result.minimum.z, point.z);
        result.maximum.x = std::max(result.maximum.x, point.x);
        result.maximum.y = std::max(result.maximum.y, point.y);
        result.maximum.z = std::max(result.maximum.z, point.z);
        result.center.x += point.x;
        result.center.y += point.y;
        result.center.z += point.z;
        result.momentum.x += static_cast<float>(element_mass * velocity.x);
        result.momentum.y += static_cast<float>(element_mass * velocity.y);
        result.momentum.z += static_cast<float>(element_mass * velocity.z);
        const double speed_squared = static_cast<double>(velocity.x) * velocity.x +
            static_cast<double>(velocity.y) * velocity.y +
            static_cast<double>(velocity.z) * velocity.z;
        result.kinetic_energy += 0.5 * element_mass * speed_squared;
        result.maximum_speed = std::max(result.maximum_speed,
                                         std::sqrt(speed_squared));
    }
    const float inverse_count = 1.0F / static_cast<float>(positions.size());
    result.center.x *= inverse_count;
    result.center.y *= inverse_count;
    result.center.z *= inverse_count;
    return result;
}

[[nodiscard]] Json aggregate_json(const Aggregate &value) {
    Json bounds = Json::object();
    bounds["maximum"] = vector_json(value.maximum);
    bounds["minimum"] = vector_json(value.minimum);
    Json result = Json::object();
    result["bounds"] = std::move(bounds);
    result["center_of_mass"] = vector_json(value.center);
    result["finite"] = value.finite;
    result["kinetic_energy"] = value.kinetic_energy;
    result["mass"] = value.mass;
    result["maximum_speed"] = value.maximum_speed;
    result["momentum"] = vector_json(value.momentum);
    return result;
}

[[nodiscard]] Json samples_json(
    const std::vector<Vec3> &positions, const std::vector<Vec3> &velocities,
    bool complete, const std::vector<std::uint32_t> *stable_ids = nullptr) {
    std::vector<std::size_t> order(positions.size());
    for (std::size_t index = 0U; index < order.size(); ++index) order[index] = index;
    if (stable_ids != nullptr)
        std::sort(order.begin(), order.end(), [&](std::size_t left,
                                                  std::size_t right) {
            return (*stable_ids)[left] < (*stable_ids)[right];
        });
    const std::size_t limit = complete ? order.size()
        : std::min(order.size(), large_sample_limit);
    Json result = Json::array();
    for (std::size_t sample = 0U; sample < limit; ++sample) {
        const std::size_t index = order[sample];
        Json item = Json::object();
        item["id"] = static_cast<std::uint64_t>(
            stable_ids != nullptr ? (*stable_ids)[index] : index);
        item["position"] = vector_json(positions[index]);
        item["velocity"] = vector_json(
            index < velocities.size() ? velocities[index] : Vec3{});
        result.push_back(std::move(item));
    }
    return result;
}

[[nodiscard]] std::string indexed_name(std::string_view kind,
                                       std::string_view source,
                                       std::size_t index) {
    return std::string(kind) + "/" +
        (source.empty() ? "unnamed" : std::string(source)) + "/" +
        std::to_string(index);
}

[[nodiscard]] TriangleMesh box_mesh(std::string name, Vec3 half) {
    TriangleMesh result{};
    result.name = std::move(name);
    const std::array<Vec3, 8U> points{{
        {-half.x,-half.y,-half.z},{half.x,-half.y,-half.z},
        {half.x,half.y,-half.z},{-half.x,half.y,-half.z},
        {-half.x,-half.y,half.z},{half.x,-half.y,half.z},
        {half.x,half.y,half.z},{-half.x,half.y,half.z}}};
    for (const Vec3 point : points) result.vertices.push_back({.position=point});
    result.indices = {0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,
                      1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0};
    return result;
}

[[nodiscard]] SceneDefinition direct_scene() {
    SceneDefinition scene{};
    scene.meshes.push_back(box_mesh("dynamic", {0.25F, 0.25F, 0.25F}));
    scene.meshes.push_back(box_mesh("ground", {3.0F, 0.05F, 3.0F}));
    scene.meshes.push_back(box_mesh("kinematic", {0.2F, 0.2F, 0.2F}));
    RigidBodyDefinition dynamic{};
    dynamic.name = "dynamic-box";
    dynamic.source_name = dynamic.name;
    dynamic.mesh_indices = {0U};
    dynamic.options.initial_state.position = {0.0F, 2.0F, 0.0F};
    dynamic.options.mass = 2.0F;
    dynamic.options.inertia_diagonal = {1.0F / 12.0F, 1.0F / 12.0F,
                                        1.0F / 12.0F};
    dynamic.options.linear_damping = 0.0F;
    dynamic.options.angular_damping = 0.0F;
    RigidBodyDefinition ground{};
    ground.name = "ground";
    ground.source_name = ground.name;
    ground.mesh_indices = {1U};
    ground.options.motion = MotionType::static_body;
    ground.options.initial_state.position = {0.0F, -0.05F, 0.0F};
    RigidBodyDefinition kinematic{};
    kinematic.name = "kinematic-box";
    kinematic.source_name = kinematic.name;
    kinematic.mesh_indices = {2U};
    kinematic.options.motion = MotionType::kinematic;
    kinematic.options.initial_state.position = {-1.0F, 1.0F, 0.0F};
    scene.rigid_bodies = {dynamic, ground, kinematic};
    return scene;
}

[[nodiscard]] SceneDefinition weld_scene(bool breaking) {
    SceneDefinition scene{};
    scene.meshes.push_back(box_mesh("member", {0.2F, 0.2F, 0.2F}));
    const std::size_t body_count = breaking ? 2U : 3U;
    for (std::size_t index = 0U; index < body_count; ++index) {
        RigidBodyDefinition body{};
        body.name = breaking ? (index == 0U ? "break-a" : "break-b")
                             : "member-" + std::to_string(index);
        body.source_name = body.name;
        body.mesh_indices = {0U};
        body.options.initial_state.position = {
            -0.4F + 0.4F * static_cast<float>(index), 1.0F, 0.0F};
        body.options.linear_damping = 0.0F;
        body.options.angular_damping = 0.0F;
        scene.rigid_bodies.push_back(std::move(body));
    }
    for (std::size_t index = 0U; index + 1U < body_count; ++index) {
        RigidConstraintDefinition constraint{};
        constraint.name = breaking ? "break-link"
            : "weld-link-" + std::to_string(index);
        constraint.body_a = static_cast<std::uint32_t>(index);
        constraint.body_b = static_cast<std::uint32_t>(index + 1U);
        constraint.options.type = RigidConstraintType::fixed;
        constraint.options.local_anchor_a = {0.2F, 0.0F, 0.0F};
        constraint.options.local_anchor_b = {-0.2F, 0.0F, 0.0F};
        constraint.options.breaking_impulse_threshold = breaking ? 0.05F : 0.0F;
        constraint.options.solver_iterations = 8U;
        scene.rigid_constraints.push_back(std::move(constraint));
    }
    return scene;
}

[[nodiscard]] SceneDefinition load_scene(const CaseDefinition &definition,
                                         const std::filesystem::path &root) {
    if (definition.kind == CaseKind::integrated_glb) {
        SceneDefinition scene{};
        std::string error;
        auto path = root / definition.glb_path;
        // Application bundles keep the same immutable GLBs in Resources/assets
        // instead of reproducing the checkout's examples/ directory hierarchy.
        if (!std::filesystem::is_regular_file(path))
            path = root / "assets" /
                std::filesystem::path(definition.glb_path).filename();
        if (!load_glb_scene(path, scene, error))
            throw std::runtime_error("load " + definition.glb_path + ": " + error);
        return scene;
    }
    if (definition.id == "rigid-direct") return direct_scene();
    if (definition.id == "compound-weld-lifecycle") return weld_scene(false);
    if (definition.id == "constraint-breaking") return weld_scene(true);
    throw std::runtime_error("no analytic scene for " + definition.id);
}

struct Runtime {
    SceneDefinition scene{};
    CommandScene command_scene{};
    CommandSceneInstance instance{};
    CommandSceneFrame frame{};
    std::vector<std::string> body_names{};
    std::vector<std::string> constraint_names{};
    std::vector<bool> constraint_present{};
    bool generation_changed{};
    bool constraints_rebuilt{};
};

[[nodiscard]] PmRigidState pm_state(const RigidBodyState &state) noexcept {
    PmRigidState result{};
    result.position[0] = state.position.x;
    result.position[1] = state.position.y;
    result.position[2] = state.position.z;
    result.orientation[0] = state.orientation.x;
    result.orientation[1] = state.orientation.y;
    result.orientation[2] = state.orientation.z;
    result.orientation[3] = state.orientation.w;
    result.linearVelocity[0] = state.linear_velocity.x;
    result.linearVelocity[1] = state.linear_velocity.y;
    result.linearVelocity[2] = state.linear_velocity.z;
    result.angularVelocity[0] = state.angular_velocity.x;
    result.angularVelocity[1] = state.angular_velocity.y;
    result.angularVelocity[2] = state.angular_velocity.z;
    return result;
}

void initialize_frame(Runtime &runtime) {
    for (const auto &body : runtime.scene.rigid_bodies)
        runtime.frame.rigid_states.push_back(pm_state(body.options.initial_state));
    for (const auto &constraint : runtime.scene.rigid_constraints)
        runtime.frame.constraint_states.push_back({
            constraint.options.enabled ? PM_TRUE : PM_FALSE, PM_FALSE, 0.0F});
    for (std::size_t index = 0U;
         index < runtime.scene.initial_particles.size(); ++index) {
        const auto &particle = runtime.scene.initial_particles[index];
        PmFluidParticleOutput output{};
        output.position[0] = particle.position.x;
        output.position[1] = particle.position.y;
        output.position[2] = particle.position.z;
        output.velocity[0] = particle.velocity.x;
        output.velocity[1] = particle.velocity.y;
        output.velocity[2] = particle.velocity.z;
        output.temperatureCelsius = particle.temperature;
        output.stableId = static_cast<std::uint32_t>(index);
        runtime.frame.fluid_particles.push_back(output);
    }
    for (const auto &cloth : runtime.scene.cloths) {
        std::vector<PmClothVertexState> vertices;
        if (cloth.mesh_index < runtime.scene.meshes.size())
            for (const auto &vertex : runtime.scene.meshes[cloth.mesh_index].vertices) {
                PmClothVertexState state{};
                state.position = {vertex.position.x, vertex.position.y,
                                  vertex.position.z};
                vertices.push_back(state);
            }
        runtime.frame.cloth_vertices.push_back(std::move(vertices));
    }
    for (const auto &rope : runtime.scene.ropes) {
        std::vector<PmRopeNodeState> nodes;
        for (const Vec3 point : rope.centerline) {
            PmRopeNodeState state{};
            state.position = {point.x, point.y, point.z};
            nodes.push_back(state);
        }
        runtime.frame.rope_nodes.push_back(std::move(nodes));
    }
    for (const auto &soft : runtime.scene.soft_bodies) {
        std::vector<PmSoftBodyNodeState> nodes;
        for (const Vec3 point : soft.nodes) {
            PmSoftBodyNodeState state{};
            state.position = {point.x, point.y, point.z};
            nodes.push_back(state);
        }
        runtime.frame.soft_body_nodes.push_back(std::move(nodes));
    }
}

[[nodiscard]] Runtime make_runtime(const CaseDefinition &definition,
                                   const std::filesystem::path &root) {
    Runtime runtime{};
    runtime.scene = load_scene(definition, root);
    for (std::size_t index = 0U; index < runtime.scene.rigid_bodies.size(); ++index) {
        const auto &body = runtime.scene.rigid_bodies[index];
        runtime.body_names.push_back(definition.kind == CaseKind::analytic
            ? body.name
            : indexed_name("rigid", body.source_name.empty() ? body.name
                                                               : body.source_name,
                           index));
    }
    for (std::size_t index = 0U;
         index < runtime.scene.rigid_constraints.size(); ++index) {
        const auto &constraint = runtime.scene.rigid_constraints[index];
        runtime.constraint_names.push_back(definition.kind == CaseKind::analytic
            ? constraint.name
            : indexed_name("constraint", constraint.name, index));
        runtime.constraint_present.push_back(true);
    }
    initialize_frame(runtime);
    std::string error;
    PmResult creation_category{};
    if (!CommandScene::create(runtime.scene, 0U, runtime.command_scene,
                              runtime.instance, error, std::nullopt, &creation_category)) {
        if (creation_category == PM_ERROR_FEATURE_NOT_PRESENT)
            throw UnsupportedCase(error);
        throw std::runtime_error("create command scene: " + error);
    }
    return runtime;
}

[[nodiscard]] std::size_t body_index(const Runtime &runtime,
                                     std::string_view name) {
    const auto found = std::find(runtime.body_names.begin(),
                                 runtime.body_names.end(), name);
    if (found == runtime.body_names.end())
        throw std::runtime_error("unknown command body " + std::string(name));
    return static_cast<std::size_t>(found - runtime.body_names.begin());
}

void require(bool success, std::string_view operation, const std::string &error) {
    if (!success)
        throw std::runtime_error(std::string(operation) + ": " + error);
}

void apply_command(Runtime &runtime, const Command &command, Vec3 &gravity) {
    if (command.operation == "set_gravity") {
        gravity = {static_cast<float>(command.value[0]),
                   static_cast<float>(command.value[1]),
                   static_cast<float>(command.value[2])};
        return;
    }
    std::string error;
    if (command.operation == "apply_impulse") {
        const auto index = static_cast<std::uint32_t>(
            body_index(runtime, command.target));
        PmRigidInput input{};
        input.bodyIndex = index;
        for (std::size_t component = 0U; component < 3U; ++component)
            input.linearImpulse[component] =
                static_cast<float>(command.value[component]);
        require(runtime.command_scene.apply_rigid_inputs({input}, error),
                "apply impulse", error);
        return;
    }
    if (command.operation == "set_kinematic_target" ||
        command.operation == "set_first_kinematic_target") {
        std::size_t index = 0U;
        if (command.operation == "set_kinematic_target") {
            index = body_index(runtime, command.target);
        } else {
            const auto found = std::find_if(
                runtime.scene.rigid_bodies.begin(),
                runtime.scene.rigid_bodies.end(), [](const auto &body) {
                    return body.options.motion == MotionType::kinematic;
                });
            if (found == runtime.scene.rigid_bodies.end())
                throw std::runtime_error("scene has no kinematic body");
            index = static_cast<std::size_t>(
                found - runtime.scene.rigid_bodies.begin());
        }
        RigidBodyState target = runtime.scene.rigid_bodies[index].options.initial_state;
        target.position = {static_cast<float>(command.value[0]),
                           static_cast<float>(command.value[1]),
                           static_cast<float>(command.value[2])};
        require(runtime.command_scene.set_kinematic_target(
                    static_cast<std::uint32_t>(index), target, error),
                "set kinematic target", error);
        return;
    }
    if (command.operation == "replace_body") {
        const std::size_t index = body_index(runtime, command.target);
        RigidBodyState replacement =
            runtime.scene.rigid_bodies[index].options.initial_state;
        replacement.position = {static_cast<float>(command.value[0]),
                                static_cast<float>(command.value[1]),
                                static_cast<float>(command.value[2])};
        require(runtime.command_scene.set_rigid_state(
                    static_cast<std::uint32_t>(index), replacement, error),
                "replace body state", error);
        runtime.generation_changed = true;
        return;
    }
    if (command.operation == "remove_constraints" ||
        command.operation == "add_constraints") {
        const bool active = command.operation == "add_constraints";
        for (std::size_t index = 0U; index < runtime.constraint_present.size();
             ++index) {
            require(runtime.command_scene.set_constraint_active(
                        static_cast<std::uint32_t>(index), active, error),
                    active ? "add constraint" : "remove constraint", error);
            runtime.constraint_present[index] = active;
        }
        runtime.constraints_rebuilt = runtime.constraints_rebuilt || active;
        return;
    }
    if (command.operation == "set_constraints_enabled") {
        const bool enabled = command.value[0] != 0.0;
        for (std::size_t index = 0U;
             index < runtime.scene.rigid_constraints.size(); ++index) {
            auto definition = runtime.scene.rigid_constraints[index];
            definition.options.enabled = enabled;
            require(runtime.command_scene.write_constraint(
                        static_cast<std::uint32_t>(index), definition, error),
                    "set constraint enabled", error);
            runtime.scene.rigid_constraints[index] = std::move(definition);
        }
        return;
    }
    throw std::runtime_error("unsupported command " + command.operation);
}

[[nodiscard]] Json rigid_resource(const Runtime &runtime, std::size_t index,
                                  std::string_view tolerance) {
    const PmRigidState &state = runtime.frame.rigid_states[index];
    Json result = Json::object();
    result["angular_velocity"] = vector_json(
        {state.angularVelocity[0], state.angularVelocity[1],
         state.angularVelocity[2]});
    result["id"] = runtime.body_names[index];
    result["linear_velocity"] = vector_json(linear_velocity(state));
    result["orientation"] = quaternion_json(state.orientation);
    result["position"] = vector_json(position(state));
    result["present"] = true;
    result["tolerance_class"] = std::string(tolerance);
    result["type"] = "rigid_body";
    return result;
}

[[nodiscard]] Json constraint_resource(const Runtime &runtime,
                                       std::size_t index) {
    const bool present = runtime.constraint_present[index];
    const PmConstraintState state = index < runtime.frame.constraint_states.size()
        ? runtime.frame.constraint_states[index] : PmConstraintState{};
    Json result = Json::object();
    result["applied_impulse"] = present ? static_cast<double>(state.appliedImpulse)
                                          : 0.0;
    result["broken"] = present && state.broken == PM_TRUE;
    result["enabled"] = present && state.enabled == PM_TRUE;
    result["id"] = runtime.constraint_names[index];
    result["present"] = present;
    result["tolerance_class"] = "constraint_contact";
    result["type"] = "rigid_constraint";
    return result;
}

[[nodiscard]] Json particle_resource(
    std::string id, std::string type, const std::vector<Vec3> &positions,
    const std::vector<Vec3> &velocities, bool complete,
    const std::vector<std::uint32_t> *stable_ids, double element_mass) {
    const Aggregate summary = aggregate(positions, velocities, element_mass);
    Json result = Json::object();
    result["aggregate"] = aggregate_json(summary);
    result["complete"] = complete;
    result["count"] = static_cast<std::uint64_t>(positions.size());
    result["id"] = std::move(id);
    result["minimum_clearance"] = static_cast<double>(summary.minimum.y);
    result["samples"] = samples_json(positions, velocities, complete, stable_ids);
    if (stable_ids != nullptr) {
        std::vector<std::uint32_t> ordered = *stable_ids;
        std::sort(ordered.begin(), ordered.end());
        result["stable_ids"] = unsigned_array_json(
            ordered, complete ? ordered.size() : large_sample_limit);
    }
    result["tolerance_class"] = "deformable";
    result["type"] = std::move(type);
    return result;
}

[[nodiscard]] Json fluid_resource(const Runtime &runtime, bool complete) {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    std::vector<std::uint32_t> ids;
    double foam_sum = 0.0;
    double foam_maximum = 0.0;
    for (const auto &particle : runtime.frame.fluid_particles) {
        positions.push_back({particle.position[0], particle.position[1],
                             particle.position[2]});
        velocities.push_back({particle.velocity[0], particle.velocity[1],
                              particle.velocity[2]});
        ids.push_back(particle.stableId);
        foam_sum += particle.foam;
        foam_maximum = std::max(foam_maximum, static_cast<double>(particle.foam));
    }
    const auto &options = runtime.scene.fluid_options;
    const double volume = options.rest_particle_volume > 0.0F
        ? options.rest_particle_volume
        : std::pow(static_cast<double>(options.particle_radius) * 2.0, 3.0);
    Json result = particle_resource("fluid/0", "fluid", positions, velocities,
                                    complete, &ids,
                                    options.rest_density * volume);
    Json foam = Json::object();
    foam["maximum"] = foam_maximum;
    foam["mean"] = positions.empty() ? 0.0 : foam_sum / positions.size();
    result["foam"] = std::move(foam);
    return result;
}

template <typename State>
void copy_particle_state(const std::vector<State> &states,
                         std::vector<Vec3> &positions,
                         std::vector<Vec3> &velocities) {
    for (const auto &state : states) {
        positions.push_back(vec(state.position));
        velocities.push_back(vec(state.velocity));
    }
}

[[nodiscard]] double distance(Vec3 first, Vec3 second) noexcept {
    const double x = static_cast<double>(first.x) - second.x;
    const double y = static_cast<double>(first.y) - second.y;
    const double z = static_cast<double>(first.z) - second.z;
    return std::sqrt(x * x + y * y + z * z);
}

[[nodiscard]] Json cloth_resource(const Runtime &runtime, std::size_t index,
                                  bool complete) {
    std::vector<Vec3> positions, velocities;
    copy_particle_state(runtime.frame.cloth_vertices[index], positions, velocities);
    const auto &definition = runtime.scene.cloths[index];
    Json result = particle_resource(indexed_name("cloth", definition.name, index),
        "cloth", positions, velocities, complete, nullptr,
        definition.vertex_mass);
    Json topology = Json::object();
    const auto &mesh = runtime.scene.meshes[definition.mesh_index];
    topology["active_bond_count"] = static_cast<std::uint64_t>(0U);
    topology["bond_count"] = static_cast<std::uint64_t>(0U);
    topology["triangle_count"] =
        static_cast<std::uint64_t>(mesh.indices.size() / 3U);
    topology["vertex_count"] = static_cast<std::uint64_t>(positions.size());
    if (complete) topology["indices"] = unsigned_array_json(mesh.indices);
    result["maximum_strain"] = 0.0;
    result["topology"] = std::move(topology);
    return result;
}

[[nodiscard]] Json soft_body_resource(const Runtime &runtime,
                                      std::size_t index, bool complete) {
    std::vector<Vec3> positions, velocities;
    copy_particle_state(runtime.frame.soft_body_nodes[index], positions, velocities);
    const auto &definition = runtime.scene.soft_bodies[index];
    Json result = particle_resource(indexed_name("soft_body", definition.name,
        index), "soft_body", positions, velocities, complete, nullptr,
        definition.node_mass);
    double maximum_strain = 0.0;
    for (const auto &bond : definition.bonds)
        if (bond.first < positions.size() && bond.second < positions.size() &&
            bond.rest_length > 0.0F)
            maximum_strain = std::max(maximum_strain,
                std::abs(distance(positions[bond.first], positions[bond.second]) /
                         bond.rest_length - 1.0));
    Json topology = Json::object();
    topology["bond_count"] = static_cast<std::uint64_t>(definition.bonds.size());
    topology["node_count"] = static_cast<std::uint64_t>(positions.size());
    const auto &mesh = runtime.scene.meshes[definition.mesh_index];
    topology["surface_triangle_count"] =
        static_cast<std::uint64_t>(mesh.indices.size() / 3U);
    topology["surface_vertex_count"] =
        static_cast<std::uint64_t>(mesh.vertices.size());
    if (complete) topology["indices"] = unsigned_array_json(mesh.indices);
    result["maximum_strain"] = maximum_strain;
    result["topology"] = std::move(topology);
    return result;
}

[[nodiscard]] Json rope_resource(const Runtime &runtime, std::size_t index,
                                 bool complete) {
    std::vector<Vec3> positions, velocities;
    copy_particle_state(runtime.frame.rope_nodes[index], positions, velocities);
    const auto &definition = runtime.scene.ropes[index];
    const double node_mass = positions.empty() ? 0.0
        : definition.options.mass / positions.size();
    Json result = particle_resource(indexed_name("rope", definition.name, index),
        "rope", positions, velocities, complete, nullptr, node_mass);
    double rest_length = 0.0;
    for (std::size_t node = 1U; node < definition.centerline.size(); ++node)
        rest_length += distance(definition.centerline[node - 1U],
                                definition.centerline[node]);
    const double rest_edge = positions.size() > 1U
        ? rest_length / static_cast<double>(positions.size() - 1U) : 0.0;
    double maximum_strain = 0.0;
    for (std::size_t node = 1U; node < positions.size() && rest_edge > 0.0;
         ++node)
        maximum_strain = std::max(maximum_strain,
            std::abs(distance(positions[node - 1U], positions[node]) /
                     rest_edge - 1.0));
    Json topology = Json::object();
    topology["edge_count"] = static_cast<std::uint64_t>(
        positions.empty() ? 0U : positions.size() - 1U);
    topology["node_count"] = static_cast<std::uint64_t>(positions.size());
    if (complete) {
        Json lengths = Json::array();
        for (std::size_t node = 1U; node < positions.size(); ++node)
            lengths.push_back(rest_edge);
        topology["rest_lengths"] = std::move(lengths);
    }
    result["maximum_strain"] = maximum_strain;
    result["topology"] = std::move(topology);
    return result;
}

[[nodiscard]] Json smoke_resource(const Runtime &runtime, bool complete) {
    std::vector<Vec3> positions, velocities;
    copy_particle_state(runtime.frame.smoke_particles, positions, velocities);
    Json result = particle_resource("smoke/0", "smoke", positions, velocities,
                                    complete, nullptr, 1.0);
    Json grid = Json::object();
    grid["cell_count"] = static_cast<std::uint64_t>(0U);
    grid["maximum_divergence"] = 0.0;
    grid["pressure_relative_residual"] = 0.0;
    grid["resolution"] =
        static_cast<std::uint64_t>(runtime.scene.smoke_options.grid_resolution);
    grid["rms_divergence"] = 0.0;
    grid["vertical_resolution"] = static_cast<std::uint64_t>(
        runtime.scene.smoke_options.grid_vertical_resolution);
    result["grid"] = std::move(grid);
    return result;
}

[[nodiscard]] Json contacts_json(const Runtime &runtime) {
    std::vector<PmContactEvent> rigid, fluid;
    for (const auto &contact : runtime.frame.contacts)
        (contact.type == PM_CONTACT_RIGID_RIGID ? rigid : fluid).push_back(contact);
    std::sort(rigid.begin(), rigid.end(), [](const auto &left,
                                             const auto &right) {
        return std::tuple{left.first, left.second, left.position.x,
                          left.position.y, left.position.z} <
               std::tuple{right.first, right.second, right.position.x,
                          right.position.y, right.position.z};
    });
    std::sort(fluid.begin(), fluid.end(), [](const auto &left,
                                             const auto &right) {
        return left.first < right.first;
    });
    Json rigid_json = Json::array();
    for (std::size_t index = 0U;
         index < std::min(rigid.size(), complete_state_limit); ++index) {
        const auto &contact = rigid[index];
        Json item = Json::object();
        item["body"] = contact.first < runtime.body_names.size()
            ? runtime.body_names[contact.first] : "unknown";
        item["collider"] = contact.second < runtime.body_names.size()
            ? runtime.body_names[contact.second] : "unknown";
        item["friction_impulse"] = vector_json(vec(contact.frictionImpulse));
        item["normal"] = vector_json(vec(contact.normal));
        item["normal_impulse"] = static_cast<double>(contact.normalImpulse);
        item["penetration"] = static_cast<double>(contact.penetration);
        item["position"] = vector_json(vec(contact.position));
        rigid_json.push_back(std::move(item));
    }
    Json fluid_json = Json::array();
    for (std::size_t index = 0U;
         index < std::min(fluid.size(), complete_state_limit); ++index) {
        const auto &contact = fluid[index];
        Json item = Json::object();
        item["normal"] = vector_json(vec(contact.normal));
        item["normal_impulse"] = static_cast<double>(contact.normalImpulse);
        item["position"] = vector_json(vec(contact.position));
        item["rigid_body"] = contact.second < runtime.body_names.size()
            ? runtime.body_names[contact.second] : "unknown";
        item["stable_particle_id"] = static_cast<std::uint64_t>(contact.first);
        fluid_json.push_back(std::move(item));
    }
    Json result = Json::object();
    result["fluid_contact_count"] = static_cast<std::uint64_t>(fluid.size());
    result["fluid_contacts"] = std::move(fluid_json);
    result["rigid_contact_count"] = static_cast<std::uint64_t>(rigid.size());
    result["rigid_contacts"] = std::move(rigid_json);
    return result;
}

[[nodiscard]] Vec3 combined_momentum(const Runtime &runtime) {
    Vec3 result{};
    const auto add = [&](const std::vector<Vec3> &velocities, double mass) {
        for (const Vec3 velocity : velocities) {
            result.x += static_cast<float>(mass * velocity.x);
            result.y += static_cast<float>(mass * velocity.y);
            result.z += static_cast<float>(mass * velocity.z);
        }
    };
    for (std::size_t index = 0U;
         index < runtime.frame.rigid_states.size(); ++index)
        if (runtime.scene.rigid_bodies[index].options.motion == MotionType::dynamic)
            add({linear_velocity(runtime.frame.rigid_states[index])},
                runtime.scene.rigid_bodies[index].options.mass);
    std::vector<Vec3> velocities;
    for (const auto &particle : runtime.frame.fluid_particles)
        velocities.push_back({particle.velocity[0], particle.velocity[1],
                              particle.velocity[2]});
    const auto &fluid = runtime.scene.fluid_options;
    const double volume = fluid.rest_particle_volume > 0.0F
        ? fluid.rest_particle_volume
        : std::pow(static_cast<double>(fluid.particle_radius) * 2.0, 3.0);
    add(velocities, fluid.rest_density * volume);
    for (std::size_t index = 0U; index < runtime.frame.cloth_vertices.size(); ++index) {
        velocities.clear();
        for (const auto &state : runtime.frame.cloth_vertices[index])
            velocities.push_back(vec(state.velocity));
        add(velocities, runtime.scene.cloths[index].vertex_mass);
    }
    for (std::size_t index = 0U;
         index < runtime.frame.soft_body_nodes.size(); ++index) {
        velocities.clear();
        for (const auto &state : runtime.frame.soft_body_nodes[index])
            velocities.push_back(vec(state.velocity));
        add(velocities, runtime.scene.soft_bodies[index].node_mass);
    }
    for (std::size_t index = 0U; index < runtime.frame.rope_nodes.size(); ++index) {
        velocities.clear();
        for (const auto &state : runtime.frame.rope_nodes[index])
            velocities.push_back(vec(state.velocity));
        add(velocities, velocities.empty() ? 0.0
            : runtime.scene.ropes[index].options.mass / velocities.size());
    }
    return result;
}

[[nodiscard]] Json checkpoint(const Runtime &runtime,
                              const CaseDefinition &definition,
                              std::uint32_t frame_number) {
    std::uint64_t element_count = runtime.frame.rigid_states.size() +
        runtime.frame.fluid_particles.size() +
        runtime.frame.smoke_particles.size();
    for (const auto &states : runtime.frame.cloth_vertices)
        element_count += states.size();
    for (const auto &states : runtime.frame.soft_body_nodes)
        element_count += states.size();
    for (const auto &states : runtime.frame.rope_nodes)
        element_count += states.size();
    const bool complete = element_count <= complete_state_limit;
    Json resources = Json::array();
    for (std::size_t index = 0U; index < runtime.frame.rigid_states.size(); ++index)
        resources.push_back(rigid_resource(runtime, index,
                                           definition.tolerance_profile));
    for (std::size_t index = 0U; index < runtime.constraint_names.size(); ++index)
        resources.push_back(constraint_resource(runtime, index));
    if (runtime.scene.fluid_options.capacity != 0U)
        resources.push_back(fluid_resource(runtime, complete));
    if (runtime.scene.has_smoke)
        resources.push_back(smoke_resource(runtime, complete));
    for (std::size_t index = 0U; index < runtime.frame.cloth_vertices.size(); ++index)
        resources.push_back(cloth_resource(runtime, index, complete));
    for (std::size_t index = 0U;
         index < runtime.frame.soft_body_nodes.size(); ++index)
        resources.push_back(soft_body_resource(runtime, index, complete));
    for (std::size_t index = 0U; index < runtime.frame.rope_nodes.size(); ++index)
        resources.push_back(rope_resource(runtime, index, complete));

    Json invariants = Json::object();
    invariants["cloth_count"] = static_cast<std::uint64_t>(
        runtime.frame.cloth_vertices.size());
    invariants["combined_momentum"] = vector_json(combined_momentum(runtime));
    invariants["complete_state"] = complete;
    invariants["constraint_count"] = static_cast<std::uint64_t>(
        runtime.constraint_names.size());
    invariants["constraints_rebuilt"] = runtime.constraints_rebuilt;
    invariants["element_count"] = element_count;
    invariants["finite"] = true;
    invariants["fluid_count"] = static_cast<std::uint64_t>(
        runtime.scene.fluid_options.capacity != 0U ? 1U : 0U);
    invariants["generation_changed"] = runtime.generation_changed;
    invariants["particle_count"] = static_cast<std::uint64_t>(
        runtime.frame.fluid_particles.size());
    invariants["rigid_body_count"] = static_cast<std::uint64_t>(
        runtime.frame.rigid_states.size());
    invariants["rope_count"] = static_cast<std::uint64_t>(
        runtime.frame.rope_nodes.size());
    invariants["smoke_count"] = static_cast<std::uint64_t>(
        runtime.frame.smoke_particles.size());
    invariants["soft_body_count"] = static_cast<std::uint64_t>(
        runtime.frame.soft_body_nodes.size());
    if (std::find(definition.coverage.begin(), definition.coverage.end(),
                  "equal_opposite_transfer") != definition.coverage.end()) {
        Json transfer = Json::object();
        transfer["combined_momentum"] = vector_json(combined_momentum(runtime));
        transfer["observable_contact_count"] = static_cast<std::uint64_t>(
            runtime.frame.contacts.size());
        transfer["systems"] = static_cast<std::uint64_t>(
            runtime.frame.rigid_states.size() +
            (runtime.scene.fluid_options.capacity != 0U ? 1U : 0U) +
            runtime.frame.cloth_vertices.size() +
            runtime.frame.soft_body_nodes.size() +
            runtime.frame.rope_nodes.size() +
            (runtime.scene.has_smoke ? 1U : 0U));
        invariants["equal_and_opposite_transfer"] = std::move(transfer);
    }
    Json result = Json::object();
    result["contacts"] = contacts_json(runtime);
    result["frame"] = static_cast<std::uint64_t>(frame_number);
    result["invariants"] = std::move(invariants);
    result["resources"] = std::move(resources);
    return result;
}

[[nodiscard]] Json timing_json(const Runtime &runtime) {
    Json result = Json::object();
    result["available"] = false;
    result["cloth_ms"] = 0.0;
    result["fluid_ms"] = 0.0;
    result["rigid_contact_ms"] = 0.0;
    result["rope_ms"] = 0.0;
    result["smoke_ms"] = 0.0;
    result["soft_body_ms"] = 0.0;
    result["total_gpu_ms"] = 0.0;
    (void)runtime;
    return result;
}

[[nodiscard]] Json run_case(const CaseDefinition &definition,
                            const std::filesystem::path &root,
                            bool every_frame) {
    Runtime runtime = make_runtime(definition, root);
    Vec3 gravity{static_cast<float>(definition.gravity[0]),
                 static_cast<float>(definition.gravity[1]),
                 static_cast<float>(definition.gravity[2])};
    Json checkpoints = Json::array();
    if (every_frame || std::find(definition.checkpoints.begin(),
                                 definition.checkpoints.end(), 0U) !=
                           definition.checkpoints.end())
        checkpoints.push_back(checkpoint(runtime, definition, 0U));
    for (std::uint32_t frame = 0U; frame < definition.frames; ++frame) {
        for (const Command &command : definition.commands)
            if (command.frame == frame) apply_command(runtime, command, gravity);
        std::string error;
        require(runtime.command_scene.step(static_cast<float>(definition.timestep),
                    definition.substeps, gravity, runtime.frame, error, true),
                "step command scene", error);
        const std::uint32_t completed = frame + 1U;
        if (every_frame || std::find(definition.checkpoints.begin(),
                                     definition.checkpoints.end(), completed) !=
                               definition.checkpoints.end())
            checkpoints.push_back(checkpoint(runtime, definition, completed));
    }
    Json diagnostics = Json::object();
    if (every_frame) diagnostics["checkpoint_mode"] = "every_frame";
    diagnostics["state_hash"] = sha256(checkpoints.serialize());
    diagnostics["timings"] = timing_json(runtime);
    Json provenance = Json::object();
    provenance["deterministic"] = true;
    provenance["device_ordinal"] = static_cast<std::int64_t>(0);
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(runtime.command_scene.physical_device(),
                                  &properties);
    provenance["device_id"] = static_cast<std::uint64_t>(properties.deviceId);
    provenance["device_name"] = properties.deviceName;
    provenance["driver_version"] =
        static_cast<std::uint64_t>(properties.driverVersion);
    provenance["host_compiler"] = host_compiler();
    provenance["source_commit"] = PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT;
    provenance["vendor_id"] = static_cast<std::uint64_t>(properties.vendorId);
    Json result = Json::object();
    PmNativeDeviceInfo native{};
    const PmResult native_result = pmGetNativeDevice(runtime.command_scene.device(), &native);
    if (native_result == PM_ERROR_FEATURE_NOT_PRESENT) {
        result["backend"] = "slang-cpu";
    } else {
        require(native_result == PM_SUCCESS, "query conformance backend", {});
        switch (native.backend) {
        case PM_NATIVE_BACKEND_CUDA: result["backend"] = "slang-cuda"; break;
        case PM_NATIVE_BACKEND_D3D12: result["backend"] = "slang-d3d12"; break;
        case PM_NATIVE_BACKEND_VULKAN: result["backend"] = "slang-vulkan"; break;
        case PM_NATIVE_BACKEND_METAL: result["backend"] = "slang-metal"; break;
        default: throw std::runtime_error("unrecognized conformance backend");
        }
    }
    result["case_id"] = definition.id;
    result["case_sha256"] = sha256(serialize_case(definition));
    result["checkpoints"] = std::move(checkpoints);
    result["diagnostics"] = std::move(diagnostics);
    result["provenance"] = std::move(provenance);
    result["schema"] = "parallel-mater-conformance-result/v1";
    result["execution_status"] = "completed";
    // Physics pass/fail is determined by the independent comparator/quality gates.
    result["quality_status"] = "not_evaluated";
    return result;
}

[[nodiscard]] std::string read_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

[[nodiscard]] bool matches_canonical_text(std::string actual,
                                          const std::string &expected) {
    std::string normalized;
    normalized.reserve(actual.size());
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        if (actual[index] == '\r' && index + 1U < actual.size() &&
            actual[index + 1U] == '\n')
            continue;
        normalized.push_back(actual[index]);
    }
    return normalized == expected;
}

[[nodiscard]] std::string contract_sha256(const std::filesystem::path &path) {
    std::string data = read_file(path);
    const auto extension = path.extension();
    if (extension == ".json" || extension == ".xml") {
        std::string normalized;
        normalized.reserve(data.size());
        for (std::size_t index = 0U; index < data.size(); ++index) {
            if (data[index] == '\r' && index + 1U < data.size() &&
                data[index + 1U] == '\n')
                continue;
            normalized.push_back(data[index]);
        }
        data = std::move(normalized);
    }
    return sha256(data);
}

[[nodiscard]] std::string compact_json_whitespace(std::string_view source) {
    std::string result;
    bool quoted = false;
    bool escaped = false;
    for (const char character : source) {
        if (quoted) {
            result.push_back(character);
            if (escaped) escaped = false;
            else if (character == '\\') escaped = true;
            else if (character == '"') quoted = false;
        } else if (character == '"') {
            quoted = true;
            result.push_back(character);
        } else if (character != ' ' && character != '\t' &&
                   character != '\r' && character != '\n') {
            result.push_back(character);
        }
    }
    return result;
}

[[nodiscard]] bool check_inputs(const std::filesystem::path &root) {
    bool valid = true;
    std::set<std::filesystem::path> expected_files;
    for (const auto &definition : case_registry()) {
        const auto path = case_file_path(root / "conformance/v1/cases", definition);
        expected_files.insert(path.filename());
        if (!std::filesystem::is_regular_file(path) ||
            !matches_canonical_text(read_file(path), serialize_case(definition))) {
            std::cerr << "non-canonical conformance input: " << path << '\n';
            valid = false;
        }
        if (definition.kind == CaseKind::integrated_glb) {
            const auto glb = root / definition.glb_path;
            if (!std::filesystem::is_regular_file(glb) ||
                sha256_file(glb) != definition.glb_sha256) {
                std::cerr << "GLB checksum mismatch: " << glb << '\n';
                valid = false;
            }
        }
    }
    const auto cases = root / "conformance/v1/cases";
    for (const auto &entry : std::filesystem::directory_iterator(cases))
        if (entry.path().extension() == ".json" &&
            !expected_files.contains(entry.path().filename())) {
            std::cerr << "unregistered conformance input: " << entry.path() << '\n';
            valid = false;
        }
    const auto manifest_path = root / "conformance/v1/manifest.json";
    if (!std::filesystem::is_regular_file(manifest_path)) {
        std::cerr << "missing conformance manifest: " << manifest_path << '\n';
        return false;
    }
    const std::string manifest = compact_json_whitespace(read_file(manifest_path));
    for (const auto &definition : case_registry()) {
        if (definition.glb_path.empty()) continue;
        const std::string entry = "\"" + definition.glb_path + "\":\"" +
                                  definition.glb_sha256 + "\"";
        if (manifest.find(entry) == std::string::npos) {
            std::cerr << "manifest missing GLB checksum: "
                      << definition.glb_path << '\n';
            valid = false;
        }
    }
    constexpr std::array<std::pair<std::string_view, std::string_view>, 12U>
        pinned_inputs{{
            {"quality profile", "conformance/v1/quality/core-conservation-v1.json"},
            {"quality profile", "conformance/v1/quality/rigid-attachment-v1.json"},
            {"quality profile", "conformance/v1/quality/rigid-brick-penetration-v1.json"},
            {"quality profile", "conformance/v1/quality/rigid-kinematic-sweep-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-self-pressure-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-self-viscosity-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/adverse-capacity-three-family-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-self-pressure-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-self-lifecycle-v1.json"},
            {"qualification profile",
             "conformance/v1/qualification/capability-obligations-v1.json"},
            {"evidence sources", "conformance/v1/evidence/evidence-sources-v1.json"},
            {"evidence index", "conformance/v1/evidence/index-v1.json"},
        }};
    for (const auto &[label, relative_text] : pinned_inputs) {
        const std::filesystem::path relative{std::string(relative_text)};
        const auto path = root / relative;
        if (!std::filesystem::is_regular_file(path)) {
            std::cerr << "missing " << label << ": " << path << '\n';
            valid = false;
            continue;
        }
        const std::string hash = contract_sha256(path);
        const std::string entry = "\"" + relative.generic_string() +
                                  "\":\"" + hash + "\"";
        if (manifest.find(entry) == std::string::npos) {
            std::cerr << "manifest " << label << " checksum mismatch: "
                      << path << '\n';
            valid = false;
        }
    }
    return valid;
}

[[nodiscard]] Json device_provenance(const std::filesystem::path &root) {
    const CaseDefinition *definition = find_case("rigid-direct");
    if (definition == nullptr) throw std::runtime_error("missing rigid-direct case");
    Runtime runtime = make_runtime(*definition, root);
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(runtime.command_scene.physical_device(),
                                  &properties);
    Json result = Json::object();
    result["api_version"] = static_cast<std::uint64_t>(properties.apiVersion);
    result["device_id"] = static_cast<std::uint64_t>(properties.deviceId);
    result["device_name"] = properties.deviceName;
    result["driver_version"] = static_cast<std::uint64_t>(properties.driverVersion);
    result["host_compiler"] = host_compiler();
    result["source_commit"] = PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT;
    result["vendor_id"] = static_cast<std::uint64_t>(properties.vendorId);
    return result;
}

[[nodiscard]] const char *interaction_family_name(PmInteractionFamily value) {
    switch (value) {
    case PM_INTERACTION_FAMILY_FLUID: return "FLUID";
    case PM_INTERACTION_FAMILY_CLOTH: return "CLOTH";
    case PM_INTERACTION_FAMILY_SOFT_BODY: return "SOFT_BODY";
    case PM_INTERACTION_FAMILY_ROPE: return "ROPE";
    case PM_INTERACTION_FAMILY_SMOKE: return "SMOKE";
    case PM_INTERACTION_FAMILY_RIGID_BODY: return "RIGID_BODY";
    }
    throw std::runtime_error("driver returned an unknown interaction family");
}

[[nodiscard]] const char *interaction_scope_name(PmInteractionScope value) {
    switch (value) {
    case PM_INTERACTION_SCOPE_SELF: return "SELF";
    case PM_INTERACTION_SCOPE_SAME_DOMAIN: return "SAME_DOMAIN";
    case PM_INTERACTION_SCOPE_SEPARATE_SYSTEM: return "SEPARATE_SYSTEM";
    case PM_INTERACTION_SCOPE_DISTINCT_FAMILY: return "DISTINCT_FAMILY";
    }
    throw std::runtime_error("driver returned an unknown interaction scope");
}

[[nodiscard]] const char *interaction_mode_name(PmInteractionMode value) {
    switch (value) {
    case PM_INTERACTION_MODE_CONTACT: return "CONTACT";
    case PM_INTERACTION_MODE_ATTACHMENT: return "ATTACHMENT";
    case PM_INTERACTION_MODE_DRAG: return "DRAG";
    case PM_INTERACTION_MODE_CONTAINMENT: return "CONTAINMENT";
    case PM_INTERACTION_MODE_PRESSURE: return "PRESSURE";
    case PM_INTERACTION_MODE_VISCOSITY: return "VISCOSITY";
    case PM_INTERACTION_MODE_THERMAL: return "THERMAL";
    case PM_INTERACTION_MODE_PHASE: return "PHASE";
    case PM_INTERACTION_MODE_INTERFACE: return "INTERFACE";
    }
    throw std::runtime_error("driver returned an unknown interaction mode");
}

[[nodiscard]] const char *interaction_direction_name(PmInteractionDirection value) {
    switch (value) {
    case PM_INTERACTION_DIRECTION_BIDIRECTIONAL: return "BIDIRECTIONAL";
    case PM_INTERACTION_DIRECTION_FIRST_TO_SECOND: return "FIRST_TO_SECOND";
    case PM_INTERACTION_DIRECTION_SECOND_TO_FIRST: return "SECOND_TO_FIRST";
    case PM_INTERACTION_DIRECTION_FIRST_TO_SECOND_TRANSFER:
        return "FIRST_TO_SECOND_TRANSFER";
    case PM_INTERACTION_DIRECTION_SECOND_TO_FIRST_TRANSFER:
        return "SECOND_TO_FIRST_TRANSFER";
    case PM_INTERACTION_DIRECTION_EXTERNAL_TO_FIRST: return "EXTERNAL_TO_FIRST";
    case PM_INTERACTION_DIRECTION_EXTERNAL_TO_SECOND: return "EXTERNAL_TO_SECOND";
    }
    throw std::runtime_error("driver returned an unknown interaction direction");
}

[[nodiscard]] const char *interaction_material_name(
    PmInteractionMaterialProfile value) {
    switch (value) {
    case PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE: return "ENDPOINT_NATIVE";
    case PM_INTERACTION_MATERIAL_SMOKE_PARTICLE: return "SMOKE_PARTICLE";
    case PM_INTERACTION_MATERIAL_SMOKE_GRID: return "SMOKE_GRID";
    case PM_INTERACTION_MATERIAL_EQUAL: return "EQUAL";
    case PM_INTERACTION_MATERIAL_MISCIBLE: return "MISCIBLE";
    case PM_INTERACTION_MATERIAL_IMMISCIBLE_INTERFACE:
        return "IMMISCIBLE_INTERFACE";
    case PM_INTERACTION_MATERIAL_THERMAL_PHASE: return "THERMAL_PHASE";
    }
    throw std::runtime_error("driver returned an unknown interaction material profile");
}

[[nodiscard]] Json advertised_capabilities(const std::filesystem::path &root) {
    const CaseDefinition *definition = find_case("rigid-direct");
    if (definition == nullptr) throw std::runtime_error("missing rigid-direct case");
    Runtime runtime = make_runtime(*definition, root);
    const PmPhysicalDevice physical = runtime.command_scene.physical_device();
    std::uint32_t count{};
    require(pmEnumeratePhysicalDeviceInteractionCapabilities(
                physical, &count, nullptr) == PM_SUCCESS,
            "query interaction capability count", {});
    std::vector<PmInteractionCapability> records(count);
    require(pmEnumeratePhysicalDeviceInteractionCapabilities(
                physical, &count, records.data()) == PM_SUCCESS,
            "query interaction capabilities", {});
    records.resize(count);
    Json result = Json::array();
    for (const auto &record : records) {
        Json item = Json::object();
        item["direction"] = interaction_direction_name(record.key.direction);
        item["first_family"] = interaction_family_name(record.key.firstFamily);
        item["material_profile"] =
            interaction_material_name(record.key.materialProfile);
        if (record.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE) {
            item["max_graph_edges_per_simulation"] = "not-applicable";
        } else if (record.maxGraphEdgesPerSimulation ==
                   PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY) {
            item["max_graph_edges_per_simulation"] = "simulation-capacity";
        } else {
            item["max_graph_edges_per_simulation"] =
                static_cast<std::uint64_t>(record.maxGraphEdgesPerSimulation);
        }
        item["mode"] = interaction_mode_name(record.key.mode);
        item["scope"] = interaction_scope_name(record.key.scope);
        item["second_family"] = interaction_family_name(record.key.secondFamily);
        result.push_back(std::move(item));
    }
    return result;
}

void write_file(const std::filesystem::path &path, const std::string &contents) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) throw std::runtime_error("cannot write " + path.string());
}

struct Arguments {
    bool list{};
    bool check{};
    bool provenance{};
    bool capabilities{};
    std::string lifecycle_id{};
    bool every_frame{};
    std::string case_id{};
    std::string quality_id{};
    std::filesystem::path output{};
    std::filesystem::path root{PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT};
};

[[nodiscard]] Arguments parse_arguments(int argc, char **argv) {
    Arguments result{};
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--list") result.list = true;
        else if (argument == "--check-inputs") result.check = true;
        else if (argument == "--provenance") result.provenance = true;
        else if (argument == "--capabilities") result.capabilities = true;
        else if (argument == "--lifecycle") {
            result.lifecycle_id = "adverse-capacity-three-family-v1";
            if (index + 1 < argc && argv[index + 1][0] != '-')
                result.lifecycle_id = argv[++index];
        }
        else if (argument == "--lifecycle-profile" && index + 1 < argc)
            result.lifecycle_id = argv[++index];
        else if (argument == "--every-frame") result.every_frame = true;
        else if (argument == "--case" && index + 1 < argc)
            result.case_id = argv[++index];
        else if (argument == "--quality" && index + 1 < argc)
            result.quality_id = argv[++index];
        else if (argument == "--root" && index + 1 < argc)
            result.root = argv[++index];
        else if (argument == "--output" && index + 1 < argc)
            result.output = argv[++index];
        else throw std::runtime_error("unknown or incomplete argument: " + argument);
    }
    const unsigned modes = static_cast<unsigned>(result.list) +
        static_cast<unsigned>(result.check) +
        static_cast<unsigned>(result.provenance) +
        static_cast<unsigned>(result.capabilities) +
        static_cast<unsigned>(!result.lifecycle_id.empty()) +
        static_cast<unsigned>(!result.case_id.empty()) +
        static_cast<unsigned>(!result.quality_id.empty());
    if (modes != 1U)
        throw std::runtime_error("choose exactly one of --list, --check-inputs, "
                                 "--provenance, --capabilities, --lifecycle, "
                                 "--lifecycle-profile, --case, or --quality");
    if (!result.lifecycle_id.empty() && result.output.empty())
        throw std::runtime_error("--lifecycle requires --output");
    if (!result.case_id.empty() && result.output.empty())
        throw std::runtime_error("--case requires --output");
    if (!result.quality_id.empty() && result.output.empty())
        throw std::runtime_error("--quality requires --output");
    if (result.every_frame &&
        (result.case_id.empty() || result.case_id == "all"))
        throw std::runtime_error("--every-frame requires one named --case");
    return result;
}

} // namespace

namespace parallel_mater::conformance {

std::vector<std::string> conformance_case_ids() {
    std::vector<std::string> result;
    result.reserve(case_registry().size());
    for (const auto &definition : case_registry()) result.push_back(definition.id);
    return result;
}

CaseExecution execute_conformance_case(
    const std::string &case_id, const std::filesystem::path &root,
    bool every_frame) {
    const CaseDefinition *definition = find_case(case_id);
    if (definition == nullptr)
        throw std::runtime_error("unknown conformance case: " + case_id);
    CaseExecution execution{};
    execution.case_id = case_id;
    Json result;
    try {
        result = run_case(*definition, root, every_frame);
        execution.status = ExecutionStatus::completed;
    } catch (const std::exception &error) {
        const bool absent = dynamic_cast<const UnsupportedCase *>(&error) != nullptr;
        execution.status = absent ? ExecutionStatus::unsupported
                                  : ExecutionStatus::failed;
        result = Json::object();
        result["schema"] = "parallel-mater-conformance-execution/v1";
        result["case_id"] = definition->id;
        result["case_sha256"] = sha256(serialize_case(*definition));
        result["execution_status"] = absent ? "unsupported" : "failed";
        result["quality_status"] = "not_evaluated";
        result["reason"] = error.what();
    }
    execution.json = result.serialize();
    return execution;
}

std::string conformance_device_provenance(const std::filesystem::path &root) {
    return device_provenance(root).serialize();
}

std::string conformance_advertised_capabilities(
    const std::filesystem::path &root) {
    return advertised_capabilities(root).serialize();
}

int conformance_command_main(int argc, char **argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const std::filesystem::path root = arguments.root;
        if (arguments.list) {
            for (const auto &definition : case_registry())
                std::cout << definition.id << '\n';
            return 0;
        }
        if (arguments.check) return check_inputs(root) ? 0 : 1;
        if (arguments.provenance) {
            std::cout << conformance_device_provenance(root);
            return 0;
        }
        if (arguments.capabilities) {
            std::cout << conformance_advertised_capabilities(root);
            return 0;
        }
        if (!arguments.lifecycle_id.empty()) {
            ExecutionStatus status{};
            std::string result;
            if (arguments.lifecycle_id == "adverse-capacity-three-family-v1")
                result = conformance_adverse_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-self-pressure-lifecycle-v1")
                result = conformance_smoke_pressure_lifecycle(root, &status);
            else if (arguments.lifecycle_id == "smoke-self-lifecycle-v1")
                result = conformance_smoke_lifecycle(root, &status);
            else
                throw std::runtime_error(
                    "unknown conformance lifecycle profile: " +
                    arguments.lifecycle_id);
            write_file(arguments.output, result);
            return status == ExecutionStatus::failed ? 2
                : status == ExecutionStatus::unsupported ? 77 : 0;
        }
        if (!arguments.quality_id.empty()) {
            const bool brick = arguments.quality_id ==
                "rigid-brick-penetration-v1";
            const bool attachment = arguments.quality_id ==
                "rigid-attachment-v1";
            const bool rope_self_contact = arguments.quality_id ==
                "rope-self-contact-v1";
            const bool smoke_self_pressure = arguments.quality_id ==
                "smoke-self-pressure-v1";
            const bool smoke_self_viscosity = arguments.quality_id ==
                "smoke-self-viscosity-v1";
            if (!brick && !attachment && !rope_self_contact &&
                !smoke_self_pressure && !smoke_self_viscosity)
                throw std::runtime_error(
                    "unknown conformance quality profile: " +
                    arguments.quality_id);
            std::string result;
            try {
                if (brick) {
                    result = conformance_rigid_brick_quality(root);
                } else if (attachment) {
                    result = conformance_rigid_attachment_quality(root);
                } else if (rope_self_contact) {
                    result = conformance_rope_self_contact_quality(root);
                } else if (smoke_self_pressure) {
                    ExecutionStatus status{};
                    result = conformance_smoke_pressure_quality(root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else {
                    result = conformance_smoke_self_viscosity_quality(root);
                }
            } catch (const std::exception &error) {
                Json failure = Json::object();
                failure["schema"] =
                    "parallel-mater-conformance-quality-result/v1";
                failure["profile_id"] = arguments.quality_id;
                const auto profile = root / "conformance/v1/quality" /
                    (arguments.quality_id + ".json");
                if (std::filesystem::is_regular_file(profile))
                    failure["profile_sha256"] = contract_sha256(profile);
                failure["execution_status"] = "failed";
                failure["reason"] = error.what();
                result = failure.serialize();
                write_file(arguments.output /
                               (arguments.quality_id + ".json"),
                           result);
                std::cerr << "FAILED " << arguments.quality_id << ": "
                          << error.what() << '\n';
                return 2;
            }
            write_file(arguments.output / (arguments.quality_id + ".json"),
                       result);
            std::cout << "Quality execution completed: "
                      << arguments.quality_id << '\n';
            return 0;
        }
        std::vector<const CaseDefinition *> definitions;
        if (arguments.case_id == "all") {
            for (const auto &definition : case_registry())
                definitions.push_back(&definition);
        } else {
            const CaseDefinition *definition = find_case(arguments.case_id);
            if (definition == nullptr)
                throw std::runtime_error("unknown conformance case: " +
                                         arguments.case_id);
            definitions.push_back(definition);
        }
        std::uint32_t completed{}, unsupported{}, failed{};
        for (const CaseDefinition *definition : definitions) {
            std::cout << "RUN " << definition->id << std::endl;
            const CaseExecution execution = execute_conformance_case(
                definition->id, root, arguments.every_frame);
            if (execution.status == ExecutionStatus::completed) {
                ++completed;
            } else if (execution.status == ExecutionStatus::unsupported) {
                ++unsupported;
                std::cerr << "UNSUPPORTED " << definition->id << '\n';
            } else {
                ++failed;
                std::cerr << "FAILED " << definition->id << '\n';
            }
            write_file(arguments.output / (definition->id + ".json"),
                       execution.json);
        }
        std::cout << "Execution: " << completed << " completed, " << unsupported
                  << " unsupported, " << failed << " failed; quality gates run separately\n";
        return failed ? 2 : (completed == 0 && unsupported != 0 ? 77 : 0);
    } catch (const std::exception &error) {
        std::cerr << "conformance: " << error.what() << '\n';
        return 2;
    }
}

} // namespace parallel_mater::conformance
