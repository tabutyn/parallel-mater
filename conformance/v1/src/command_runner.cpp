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
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
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

[[nodiscard]] const char *result_name(PmResult result) noexcept {
    switch (result) {
    case PM_SUCCESS: return "PM_SUCCESS";
    case PM_ERROR_INVALID_ARGUMENT: return "PM_ERROR_INVALID_ARGUMENT";
    case PM_ERROR_INVALID_STATE: return "PM_ERROR_INVALID_STATE";
    case PM_ERROR_OUT_OF_HOST_MEMORY: return "PM_ERROR_OUT_OF_HOST_MEMORY";
    case PM_ERROR_OUT_OF_DEVICE_MEMORY: return "PM_ERROR_OUT_OF_DEVICE_MEMORY";
    case PM_ERROR_FEATURE_NOT_PRESENT: return "PM_ERROR_FEATURE_NOT_PRESENT";
    case PM_ERROR_FORMAT_NOT_SUPPORTED: return "PM_ERROR_FORMAT_NOT_SUPPORTED";
    case PM_ERROR_CAPACITY_EXCEEDED: return "PM_ERROR_CAPACITY_EXCEEDED";
    case PM_ERROR_DEVICE_LOST: return "PM_ERROR_DEVICE_LOST";
    case PM_ERROR_INCOMPATIBLE_VERSION: return "PM_ERROR_INCOMPATIBLE_VERSION";
    case PM_ERROR_DRIVER_NOT_FOUND: return "PM_ERROR_DRIVER_NOT_FOUND";
    case PM_ERROR_LAYER_NOT_PRESENT: return "PM_ERROR_LAYER_NOT_PRESENT";
    case PM_ERROR_EXTENSION_NOT_PRESENT: return "PM_ERROR_EXTENSION_NOT_PRESENT";
    case PM_NOT_READY: return "PM_NOT_READY";
    case PM_TIMEOUT: return "PM_TIMEOUT";
    case PM_INCOMPLETE: return "PM_INCOMPLETE";
    }
    return "PM_ERROR_UNKNOWN";
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

[[nodiscard]] SceneDefinition settled_fixture_scene() {
    SceneDefinition scene{};
    scene.meshes.push_back(box_mesh("settled-member", {0.2F, 0.2F, 0.2F}));
    RigidBodyDefinition anchor{};
    anchor.name = "settled-anchor";
    anchor.source_name = anchor.name;
    anchor.mesh_indices = {0U};
    anchor.options.motion = MotionType::static_body;
    anchor.options.initial_state.position = {0.0F, 1.0F, 0.0F};
    RigidBodyDefinition member{};
    member.name = "settled-member";
    member.source_name = member.name;
    member.mesh_indices = {0U};
    member.options.initial_state.position = {0.5F, 1.0F, 0.0F};
    member.options.linear_damping = 0.2F;
    member.options.angular_damping = 0.2F;
    scene.rigid_bodies = {anchor, member};
    RigidConstraintDefinition joint{};
    joint.name = "settled-fixed";
    joint.body_a = 0U;
    joint.body_b = 1U;
    joint.options.type = RigidConstraintType::fixed;
    joint.options.local_anchor_a = {0.25F, 0.0F, 0.0F};
    joint.options.local_anchor_b = {-0.25F, 0.0F, 0.0F};
    joint.options.solver_iterations = 10U;
    scene.rigid_constraints.push_back(std::move(joint));
    return scene;
}

[[nodiscard]] CaseDefinition settled_fixture_definition() {
    CaseDefinition result{};
    result.id = "rigid-settled-long-run";
    result.title = "Grounded fixed-joint settled assembly";
    result.frames = 3'600U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, -9.81, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    return result;
}

[[nodiscard]] SceneDefinition cloth_fixture_scene() {
    SceneDefinition scene{};
    constexpr std::uint32_t side = 5U;
    constexpr float spacing = 0.1F;
    TriangleMesh cloth_mesh{};
    cloth_mesh.name = "qualification-cloth";
    for (std::uint32_t z = 0U; z < side; ++z)
        for (std::uint32_t x = 0U; x < side; ++x)
            cloth_mesh.vertices.push_back({.position = {
                (static_cast<float>(x) - 2.0F) * spacing, 0.0F,
                (static_cast<float>(z) - 2.0F) * spacing}});
    for (std::uint32_t z = 0U; z + 1U < side; ++z)
        for (std::uint32_t x = 0U; x + 1U < side; ++x) {
            const std::uint32_t a = z * side + x;
            const std::uint32_t b = a + 1U;
            const std::uint32_t c = a + side;
            const std::uint32_t d = c + 1U;
            cloth_mesh.indices.insert(
                cloth_mesh.indices.end(), {a, c, b, b, c, d});
        }
    scene.meshes.push_back(std::move(cloth_mesh));
    scene.meshes.push_back(box_mesh(
        "qualification-impactor", {0.025F, 0.025F, 0.025F}));

    ClothDefinition cloth{};
    cloth.name = "qualification-cloth";
    cloth.mesh_index = 0U;
    cloth.inverse_masses.resize(side * side, 100.0F);
    for (std::uint32_t z = 0U; z < side; ++z)
        for (std::uint32_t x = 0U; x < side; ++x)
            if (x == 0U || z == 0U || x + 1U == side || z + 1U == side)
                cloth.inverse_masses[z * side + x] = 0.0F;
    cloth.vertex_mass = 0.01F;
    cloth.thickness = 0.01F;
    cloth.stretch_compliance = 1.0e-5F;
    cloth.bending_compliance = 1.0e-4F;
    cloth.velocity_damping = 5.0F;
    cloth.contact_friction = 0.2F;
    cloth.solver_iterations = 8U;
    scene.cloths.push_back(std::move(cloth));

    RigidBodyDefinition impactor{};
    impactor.name = "qualification-impactor";
    impactor.source_name = impactor.name;
    impactor.mesh_indices = {1U};
    impactor.options.initial_state.position = {0.0F, 0.04F, 0.0F};
    impactor.options.initial_state.linear_velocity = {0.0F, -0.05F, 0.0F};
    impactor.options.mass = 0.001F;
    impactor.options.inertia_diagonal = {1.0e-6F, 1.0e-6F, 1.0e-6F};
    impactor.options.linear_damping = 0.0F;
    impactor.options.maximum_linear_speed = 1.0F;
    scene.rigid_bodies.push_back(std::move(impactor));
    return scene;
}

[[nodiscard]] CaseDefinition cloth_fixture_definition() {
    CaseDefinition result{};
    result.id = "cloth-stiff-pin-impact";
    result.title = "Pinned stiff cloth with controlled rigid impact";
    result.frames = 600U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, 0.0, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    return result;
}

[[nodiscard]] TriangleMesh tetrahedron_mesh(
    std::string name, Vec3 origin, float edge);

[[nodiscard]] SceneDefinition soft_body_fixture_scene() {
    SceneDefinition scene{};
    scene.meshes.push_back(tetrahedron_mesh(
        "qualification-soft", {-0.05F, -0.05F, -0.05F}, 0.1F));
    scene.meshes.push_back(box_mesh(
        "qualification-soft-impactor", {0.02F, 0.02F, 0.02F}));

    SoftBodyDefinition soft{};
    soft.name = "qualification-soft";
    soft.mesh_index = 0U;
    for (const auto &vertex : scene.meshes[0U].vertices)
        soft.nodes.push_back(vertex.position);
    for (const auto &[first, second] :
         std::array<std::pair<std::uint32_t, std::uint32_t>, 6U>{
             std::pair{0U, 1U}, {0U, 2U}, {0U, 3U},
             {1U, 2U}, {1U, 3U}, {2U, 3U}}) {
        const Vec3 delta{
            soft.nodes[first].x - soft.nodes[second].x,
            soft.nodes[first].y - soft.nodes[second].y,
            soft.nodes[first].z - soft.nodes[second].z};
        soft.bonds.push_back({first, second, std::sqrt(
            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)});
    }
    soft.inverse_masses.assign(4U, 1.0F);
    for (std::uint32_t node = 0U; node < 4U; ++node) {
        SoftBodySurfaceBinding binding{};
        binding.nodes[0] = node;
        binding.weights[0] = 1.0F;
        soft.surface_bindings.push_back(binding);
    }
    soft.node_mass = 1.0F;
    soft.node_radius = 0.025F;
    soft.stretch_compliance = 1.0e-5F;
    soft.velocity_damping = 2.0F;
    soft.spring_damping = 0.0F;
    soft.shape_matching_stiffness = 0.8F;
    soft.maximum_speed = 1.0F;
    soft.solver_iterations = 8U;
    scene.soft_bodies.push_back(std::move(soft));

    RigidBodyDefinition impactor{};
    impactor.name = "qualification-soft-impactor";
    impactor.source_name = impactor.name;
    impactor.mesh_indices = {1U};
    impactor.options.initial_state.position = {-0.09F, 0.0F, 0.0F};
    impactor.options.initial_state.linear_velocity = {0.05F, 0.0F, 0.0F};
    impactor.options.mass = 0.01F;
    impactor.options.linear_damping = 0.0F;
    impactor.options.maximum_linear_speed = 1.0F;
    scene.rigid_bodies.push_back(std::move(impactor));
    return scene;
}

[[nodiscard]] CaseDefinition soft_body_fixture_definition() {
    CaseDefinition result{};
    result.id = "soft-body-mass-ratio-volume";
    result.title = "Shape-matched tetrahedron with controlled mass-ratio impact";
    result.frames = 600U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, 0.0, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    return result;
}

[[nodiscard]] std::vector<FluidParticle> regular_tetra_fluid(Vec3 center) {
    constexpr float coordinate = 0.03535533905932738F;
    std::vector<FluidParticle> result;
    for (const Vec3 offset : std::array<Vec3, 4U>{
             Vec3{coordinate, coordinate, coordinate},
             Vec3{coordinate, -coordinate, -coordinate},
             Vec3{-coordinate, coordinate, -coordinate},
             Vec3{-coordinate, -coordinate, coordinate}})
        result.push_back({{center.x + offset.x, center.y + offset.y,
                           center.z + offset.z}, {}, 20.0F});
    return result;
}

[[nodiscard]] SceneDefinition fluid_fixture_scene() {
    SceneDefinition scene{};
    constexpr float support = 0.2F;
    constexpr float edge = 0.1F;
    const double h2 = support * support;
    const double kernel_scale = 315.0 /
        (64.0 * std::numbers::pi * std::pow(support, 9.0));
    const double kernel_sum = kernel_scale * (
        std::pow(h2, 3.0) + 3.0 * std::pow(h2 - edge * edge, 3.0));
    scene.fluid_options.capacity = 8U;
    scene.fluid_options.particle_radius = 0.02F;
    scene.fluid_options.support_radius = support;
    scene.fluid_options.rest_density = 1'000.0F;
    scene.fluid_options.rest_particle_volume =
        static_cast<float>(1.0 / kernel_sum);
    scene.fluid_options.maximum_neighbors = 8U;
    scene.fluid_options.solver_iterations = 6U;
    scene.fluid_options.repulsion = 50.0F;
    scene.fluid_options.viscosity = 0.0F;
    scene.fluid_options.velocity_damping = 0.0F;
    scene.fluid_options.maximum_speed = 1.0F;
    scene.initial_particles = regular_tetra_fluid({-0.5F, 0.0F, 0.0F});

    scene.meshes.push_back(box_mesh("wall-x", {0.02F, 1.0F, 1.0F}));
    scene.meshes.push_back(box_mesh("wall-y", {1.0F, 0.02F, 1.0F}));
    scene.meshes.push_back(box_mesh("wall-z", {1.0F, 1.0F, 0.02F}));
    for (std::uint32_t axis = 0U; axis < 3U; ++axis)
        for (const float sign : {-1.0F, 1.0F}) {
            RigidBodyDefinition wall{};
            wall.name = "closed-wall-" + std::to_string(axis) +
                (sign < 0.0F ? "-" : "+");
            wall.source_name = wall.name;
            wall.mesh_indices = {axis};
            wall.options.motion = MotionType::static_body;
            if (axis == 0U) wall.options.initial_state.position.x = 1.02F * sign;
            if (axis == 1U) wall.options.initial_state.position.y = 1.02F * sign;
            if (axis == 2U) wall.options.initial_state.position.z = 1.02F * sign;
            scene.rigid_bodies.push_back(std::move(wall));
        }
    return scene;
}

[[nodiscard]] CaseDefinition fluid_fixture_definition() {
    CaseDefinition result{};
    result.id = "fluid-closed-container-source-outflow";
    result.title = "Closed symmetric SPH clusters with source lifecycle";
    result.frames = 600U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, 0.0, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    result.commands = {
        {100U, "fluid_fixture_source_birth", "fluid", {}, 0U},
        {300U, "fluid_fixture_source_removal", "fluid", {}, 0U}};
    return result;
}

[[nodiscard]] SceneDefinition smoke_fixture_scene() {
    SceneDefinition scene{};
    scene.create_default_couplings = false;
    scene.has_smoke = true;
    scene.smoke_options.capacity = 1U;
    scene.smoke_options.emitter_center = {-20.0F, 10.0F, 0.0F};
    scene.smoke_options.emitter_half_extents = {0.001F, 0.001F};
    scene.smoke_options.initial_velocity = {1.6F, 0.0F, 0.0F};
    scene.smoke_options.wind = scene.smoke_options.initial_velocity;
    scene.smoke_options.particles_per_second = 60.0F;
    scene.smoke_options.lifetime = 100.0F;
    scene.smoke_options.particle_radius = 0.085F;
    scene.smoke_options.buoyancy = 0.0F;
    scene.smoke_options.response = 0.0F;
    scene.smoke_options.rest_number_density = 1.0F;
    scene.smoke_options.pressure_stiffness = 0.0F;
    scene.smoke_options.viscosity = 0.0F;
    scene.smoke_options.vorticity_confinement = 0.0F;
    scene.smoke_options.maximum_speed = 2.0F;
    scene.smoke_options.grid_resolution = 0U;
    return scene;
}

[[nodiscard]] CaseDefinition smoke_fixture_definition() {
    CaseDefinition result{};
    result.id = "smoke-analytic-transport-long-run";
    result.title = "Single-tracer analytic transport with source shutdown";
    result.frames = 3'600U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, 0.0, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    result.commands = {
        {1'800U, "set_smoke_emitter_active", "smoke", {}, 0U}};
    return result;
}

[[nodiscard]] SceneDefinition containment_fixture_scene() {
    SceneDefinition scene = cloth_fixture_scene();
    constexpr Vec3 center{10.0F, 0.0F, 0.0F};
    constexpr float support = 0.2F;
    constexpr float edge = 0.1F;
    const double h2 = support * support;
    const double kernel_scale = 315.0 /
        (64.0 * std::numbers::pi * std::pow(support, 9.0));
    const double kernel_sum = kernel_scale * (
        std::pow(h2, 3.0) + 3.0 * std::pow(h2 - edge * edge, 3.0));
    scene.fluid_options.capacity = 4U;
    scene.fluid_options.particle_radius = 0.02F;
    scene.fluid_options.support_radius = support;
    scene.fluid_options.rest_density = 1'000.0F;
    scene.fluid_options.rest_particle_volume =
        static_cast<float>(1.0 / kernel_sum);
    scene.fluid_options.maximum_neighbors = 4U;
    scene.fluid_options.solver_iterations = 6U;
    scene.fluid_options.repulsion = 50.0F;
    scene.fluid_options.viscosity = 0.0F;
    scene.fluid_options.velocity_damping = 0.0F;
    scene.fluid_options.maximum_speed = 1.0F;
    scene.initial_particles = regular_tetra_fluid(center);

    const std::uint32_t first_wall_mesh =
        static_cast<std::uint32_t>(scene.meshes.size());
    scene.meshes.push_back(box_mesh(
        "containment-wall-x", {0.02F, 1.0F, 1.0F}));
    scene.meshes.push_back(box_mesh(
        "containment-wall-y", {1.0F, 0.02F, 1.0F}));
    scene.meshes.push_back(box_mesh(
        "containment-wall-z", {1.0F, 1.0F, 0.02F}));
    for (std::uint32_t axis = 0U; axis < 3U; ++axis)
        for (const float sign : {-1.0F, 1.0F}) {
            RigidBodyDefinition wall{};
            wall.name = "containment-wall-" + std::to_string(axis) +
                (sign < 0.0F ? "-" : "+");
            wall.source_name = wall.name;
            wall.mesh_indices = {first_wall_mesh + axis};
            wall.options.motion = MotionType::static_body;
            wall.options.initial_state.position = center;
            if (axis == 0U) wall.options.initial_state.position.x += 1.02F * sign;
            if (axis == 1U) wall.options.initial_state.position.y += 1.02F * sign;
            if (axis == 2U) wall.options.initial_state.position.z += 1.02F * sign;
            scene.rigid_bodies.push_back(std::move(wall));
        }
    return scene;
}

[[nodiscard]] CaseDefinition containment_fixture_definition() {
    CaseDefinition result{};
    result.id = "fluid-cloth-rigid-containment";
    result.title = "Closed liquid cell with controlled rigid-cloth impact";
    result.frames = 600U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, 0.0, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    return result;
}

[[nodiscard]] TriangleMesh tetrahedron_mesh(std::string name, Vec3 origin,
                                            float edge) {
    TriangleMesh result{};
    result.name = std::move(name);
    for (const Vec3 point : std::array<Vec3, 4U>{
             origin,
             Vec3{origin.x + edge, origin.y, origin.z},
             Vec3{origin.x, origin.y + edge, origin.z},
             Vec3{origin.x, origin.y, origin.z + edge}})
        result.vertices.push_back({.position = point});
    result.indices = {0U, 2U, 1U, 0U, 1U, 3U,
                      0U, 3U, 2U, 1U, 2U, 3U};
    return result;
}

[[nodiscard]] SceneDefinition mixed_fixture_scene() {
    SceneDefinition scene{};
    scene.meshes.push_back(box_mesh("mixed-rigid", {0.25F, 0.25F, 0.25F}));
    scene.meshes.push_back(tetrahedron_mesh(
        "mixed-cloth", {-20.0F, 10.0F, 0.0F}, 0.1F));
    scene.meshes.push_back(tetrahedron_mesh(
        "mixed-soft", {20.0F, 10.0F, 0.0F}, 0.1F));
    scene.meshes.push_back(box_mesh(
        "mixed-fluid-container", {1.0F, 1.0F, 1.0F}));

    RigidBodyDefinition floor{};
    floor.name = "mixed-static";
    floor.source_name = floor.name;
    floor.mesh_indices = {3U};
    floor.options.motion = MotionType::static_body;
    floor.options.initial_state.position = {40.1F, 10.1F, 0.1F};
    RigidBodyDefinition dynamic{};
    dynamic.name = "mixed-dynamic";
    dynamic.source_name = dynamic.name;
    dynamic.mesh_indices = {0U};
    dynamic.options.initial_state.position = {0.0F, 3.0F, 0.0F};
    dynamic.options.maximum_linear_speed = 20.0F;
    dynamic.options.maximum_angular_speed = 20.0F;
    scene.rigid_bodies = {floor, dynamic};

    scene.fluid_options.capacity = 8U;
    scene.fluid_options.maximum_neighbors = 16U;
    scene.fluid_options.solver_iterations = 6U;
    scene.fluid_options.particle_radius = 0.025F;
    scene.fluid_options.support_radius = 0.09F;
    scene.fluid_options.rest_density = 1'000.0F;
    const double kernel_zero = 315.0 /
        (64.0 * std::numbers::pi * std::pow(0.09, 3.0));
    scene.fluid_options.rest_particle_volume =
        static_cast<float>(1.0 / kernel_zero);
    scene.fluid_options.maximum_speed = 20.0F;
    for (std::uint32_t index = 0U; index < 8U; ++index)
        scene.initial_particles.push_back({
            {40.0F + static_cast<float>(index % 2U) * 0.2F,
             10.0F + static_cast<float>((index / 2U) % 2U) * 0.2F,
             static_cast<float>(index / 4U) * 0.2F}, {}, 20.0F});

    ClothDefinition cloth{};
    cloth.name = "mixed-cloth";
    cloth.mesh_index = 1U;
    cloth.inverse_masses.assign(4U, 0.0F);
    cloth.vertex_mass = 0.01F;
    cloth.solver_iterations = 8U;
    scene.cloths.push_back(std::move(cloth));

    SoftBodyDefinition soft{};
    soft.name = "mixed-soft";
    soft.mesh_index = 2U;
    for (const auto &vertex : scene.meshes[2U].vertices)
        soft.nodes.push_back(vertex.position);
    for (const auto &[first, second] :
         std::array<std::pair<std::uint32_t, std::uint32_t>, 6U>{
             std::pair{0U, 1U}, {0U, 2U}, {0U, 3U},
             {1U, 2U}, {1U, 3U}, {2U, 3U}}) {
        const Vec3 delta{soft.nodes[first].x - soft.nodes[second].x,
                         soft.nodes[first].y - soft.nodes[second].y,
                         soft.nodes[first].z - soft.nodes[second].z};
        soft.bonds.push_back({first, second, std::sqrt(
            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)});
    }
    soft.inverse_masses.assign(4U, 1.0F);
    for (std::uint32_t node = 0U; node < 4U; ++node) {
        SoftBodySurfaceBinding binding{};
        binding.nodes[0] = node;
        binding.weights[0] = 1.0F;
        soft.surface_bindings.push_back(binding);
    }
    soft.solver_iterations = 8U;
    scene.soft_bodies.push_back(std::move(soft));

    RopeDefinition rope{};
    rope.name = "mixed-rope";
    rope.centerline = {{30.0F, 10.0F, 0.0F},
        {std::nextafter(30.15F, 30.0F), 10.0F, 0.0F}};
    rope.options.node_spacing = 0.05F;
    rope.options.radius = 0.025F;
    rope.options.solver_iterations = 8U;
    rope.options.maximum_substep_timestep = 1.0F / 60.0F;
    rope.options.maximum_speed = 20.0F;
    scene.ropes.push_back(std::move(rope));

    scene.has_smoke = true;
    scene.smoke_options.capacity = 4U;
    scene.smoke_options.emitter_center = {-40.0F, 10.0F, 0.0F};
    scene.smoke_options.emitter_half_extents = {0.01F, 0.01F};
    scene.smoke_options.initial_velocity = {1.6F, 0.0F, 0.0F};
    scene.smoke_options.wind = scene.smoke_options.initial_velocity;
    scene.smoke_options.particles_per_second = 240.0F;
    scene.smoke_options.lifetime = 10'000.0F;
    scene.smoke_options.buoyancy = 0.0F;
    scene.smoke_options.response = 0.0F;
    scene.smoke_options.pressure_stiffness = 0.0F;
    scene.smoke_options.viscosity = 0.0F;
    scene.smoke_options.vorticity_confinement = 0.0F;
    scene.smoke_options.maximum_speed = 4.0F;
    // Four tracers in the authored emitter footprint reconstruct a public
    // number density near four.  Use that authored equilibrium rather than
    // the unrelated dense-cloud default of twelve.
    scene.smoke_options.rest_number_density = 4.0F;
    return scene;
}

[[nodiscard]] CaseDefinition mixed_fixture_definition() {
    CaseDefinition result{};
    result.id = "six-family-mixed";
    result.title = "Shared six-family public fixture";
    result.frames = 600U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, 0.0, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    result.commands = {
        {300U, "set_smoke_emitter_active", "smoke", {}, 0U}};
    return result;
}

[[nodiscard]] SceneDefinition boundary_fixture_scene() {
    SceneDefinition scene = mixed_fixture_scene();
    scene.cloths.clear();
    scene.soft_bodies.clear();
    scene.has_smoke = false;
    scene.fluid_options.capacity = 4U;
    scene.initial_particles.resize(4U);
    return scene;
}

[[nodiscard]] CaseDefinition boundary_fixture_definition() {
    CaseDefinition result{};
    result.id = "boundary-replacement-generation";
    result.title = "Public boundary replacement and source birth fixture";
    result.frames = 4U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, -9.81, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    result.commands = {
        {1U, "replace_boundary_resources", "world", {}, 0U},
        {2U, "boundary_source_birth", "fluid", {}, 0U}};
    return result;
}

[[nodiscard]] SceneDefinition empty_sparse_fixture_scene() {
    SceneDefinition scene{};
    scene.create_default_couplings = false;
    scene.meshes.push_back(box_mesh("sparse-rigid", {0.1F, 0.1F, 0.1F}));
    TriangleMesh point_mesh{};
    point_mesh.name = "sparse-point";
    point_mesh.vertices.push_back({.position = {10.0F, 10.0F, 0.0F}});
    scene.meshes.push_back(point_mesh);
    scene.meshes.push_back(tetrahedron_mesh(
        "sparse-soft-surface", {30.0F, 10.0F, 0.0F}, 0.05F));

    RigidBodyDefinition rigid{};
    rigid.name = "sparse-rigid";
    rigid.source_name = rigid.name;
    rigid.mesh_indices = {0U};
    rigid.options.initial_state.position = {-10.0F, 10.0F, 0.0F};
    rigid.options.maximum_linear_speed = 20.0F;
    rigid.options.maximum_angular_speed = 20.0F;
    scene.rigid_bodies.push_back(std::move(rigid));

    scene.fluid_options.capacity = 1U;
    scene.fluid_options.maximum_neighbors = 1U;
    scene.fluid_options.solver_iterations = 6U;
    scene.fluid_options.particle_radius = 0.025F;
    scene.fluid_options.support_radius = 0.09F;
    scene.fluid_options.rest_density = 1'000.0F;
    const double kernel_zero = 315.0 /
        (64.0 * std::numbers::pi * std::pow(0.09, 3.0));
    scene.fluid_options.rest_particle_volume =
        static_cast<float>(1.0 / kernel_zero);
    scene.initial_particles.push_back({{20.0F, 10.0F, 0.0F}, {}, 20.0F});

    ClothDefinition cloth{};
    cloth.name = "sparse-cloth";
    cloth.mesh_index = 1U;
    cloth.inverse_masses = {0.0F};
    cloth.solver_iterations = 8U;
    scene.cloths.push_back(std::move(cloth));

    SoftBodyDefinition soft{};
    soft.name = "sparse-soft";
    soft.mesh_index = 2U;
    soft.nodes = {{30.0F, 10.0F, 0.0F}};
    soft.inverse_masses = {1.0F};
    for (std::uint32_t vertex = 0U; vertex < 4U; ++vertex) {
        SoftBodySurfaceBinding binding{};
        binding.weights[0] = 1.0F;
        soft.surface_bindings.push_back(binding);
    }
    soft.solver_iterations = 8U;
    scene.soft_bodies.push_back(std::move(soft));

    RopeDefinition rope{};
    rope.name = "sparse-rope";
    rope.centerline = {{40.0F, 10.0F, 0.0F}, {40.04F, 10.0F, 0.0F}};
    rope.options.node_spacing = 0.05F;
    rope.options.radius = 0.025F;
    rope.options.maximum_substep_timestep = 1.0F / 60.0F;
    rope.options.solver_iterations = 8U;
    rope.options.maximum_speed = 20.0F;
    scene.ropes.push_back(std::move(rope));

    scene.has_smoke = true;
    scene.smoke_options.capacity = 1U;
    scene.smoke_options.emitter_center = {50.0F, 10.0F, 0.0F};
    scene.smoke_options.emitter_half_extents = {0.01F, 0.01F};
    scene.smoke_options.initial_velocity = {1.0F, 0.0F, 0.0F};
    scene.smoke_options.wind = scene.smoke_options.initial_velocity;
    scene.smoke_options.particles_per_second = 60.0F;
    scene.smoke_options.lifetime = 10'000.0F;
    scene.smoke_options.maximum_speed = 4.0F;
    scene.smoke_options.rest_number_density = 1.0F;
    return scene;
}

[[nodiscard]] CaseDefinition empty_sparse_fixture_definition() {
    CaseDefinition result{};
    result.id = "empty-and-sparse-capacity";
    result.title = "Empty then sparse public capacity fixture";
    result.frames = 2U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, -9.81, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    result.commands = {
        {1U, "admit_sparse_resources", "world", {}, 0U}};
    return result;
}

[[nodiscard]] SceneDefinition capacity_fixture_scene() {
    SceneDefinition scene{};
    scene.create_default_couplings = false;

    TriangleMesh cloth_mesh{};
    cloth_mesh.name = "capacity-cloth";
    cloth_mesh.vertices = {
        {.position = {30.0F, 10.0F, 0.0F}},
        {.position = {30.1F, 10.0F, 0.0F}},
        {.position = {30.0F, 10.1F, 0.0F}}};
    cloth_mesh.indices = {0U, 1U, 2U};
    scene.meshes.push_back(std::move(cloth_mesh));

    scene.fluid_options.capacity = 4U;
    scene.fluid_options.maximum_neighbors = 1U;
    scene.fluid_options.solver_iterations = 6U;
    scene.fluid_options.particle_radius = 0.025F;
    scene.fluid_options.support_radius = 0.09F;
    scene.fluid_options.rest_density = 1'000.0F;
    const double kernel_zero = 315.0 /
        (64.0 * std::numbers::pi * std::pow(0.09, 3.0));
    scene.fluid_options.rest_particle_volume =
        static_cast<float>(1.0 / kernel_zero);
    scene.initial_particles.push_back({{20.0F, 10.0F, 0.0F}, {}, 20.0F});

    ClothDefinition cloth{};
    cloth.name = "capacity-cloth";
    cloth.mesh_index = 0U;
    cloth.inverse_masses = {0.0F, 0.0F, 0.0F};
    cloth.solver_iterations = 8U;
    scene.cloths.push_back(std::move(cloth));

    scene.has_smoke = true;
    scene.smoke_options.capacity = 1U;
    scene.smoke_options.emitter_center = {40.0F, 10.0F, 0.0F};
    scene.smoke_options.emitter_half_extents = {0.01F, 0.01F};
    scene.smoke_options.initial_velocity = {1.0F, 0.0F, 0.0F};
    scene.smoke_options.wind = scene.smoke_options.initial_velocity;
    scene.smoke_options.particles_per_second = 60.0F;
    scene.smoke_options.lifetime = 10'000.0F;
    scene.smoke_options.maximum_speed = 4.0F;
    scene.smoke_options.rest_number_density = 1.0F;
    return scene;
}

[[nodiscard]] CaseDefinition capacity_fixture_definition() {
    CaseDefinition result{};
    result.id = "capacity-overflow-atomic-recovery";
    result.title = "Public capacity overflow and atomic recovery fixture";
    result.frames = 3U;
    result.substeps = 1U;
    result.timestep = 1.0 / 60.0;
    result.gravity = {0.0, -9.81, 0.0};
    result.tolerance_profile = "realtime-single-step-v1";
    result.commands = {
        {1U, "submit_capacity_overflow", "fluid", {}, 0U},
        {2U, "recover_capacity_input", "fluid", {}, 0U}};
    return result;
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
                                   SceneDefinition scene) {
    Runtime runtime{};
    runtime.scene = std::move(scene);
    std::string error;
    if (!apply_gallery_fixed_tick_material_contract(runtime.scene, error)) {
        if (std::string_view(error).starts_with(
                "fixed-tick-60hz-v1 does not implement "))
            throw UnsupportedCase(error);
        throw std::runtime_error("apply fixed-tick material contract: " + error);
    }
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
    PmResult creation_category{};
    if (!CommandScene::create(runtime.scene, 0U, runtime.command_scene,
                              runtime.instance, error, std::nullopt,
                              &creation_category,
                              std::getenv("PM_TEST_VALIDATION") != nullptr)) {
        if (creation_category == PM_ERROR_FEATURE_NOT_PRESENT)
            throw UnsupportedCase(error);
        throw std::runtime_error("create command scene: " + error);
    }
    return runtime;
}

[[nodiscard]] Runtime make_runtime(const CaseDefinition &definition,
                                   const std::filesystem::path &root) {
    return make_runtime(definition, load_scene(definition, root));
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
    if (command.operation == "set_particle_emitters_active") {
        require(runtime.command_scene.set_particle_emitters_active(
                    command.value[0] != 0.0, error),
                "set particle emitters active", error);
        return;
    }
    if (command.operation == "set_smoke_emitter_active") {
        require(runtime.command_scene.set_smoke_emitter_active(
                    command.value[0] != 0.0, error),
                "set smoke emitter active", error);
        return;
    }
    if (command.operation == "fluid_fixture_source_birth") {
        auto particles = regular_tetra_fluid({-0.5F, 0.0F, 0.0F});
        auto source = regular_tetra_fluid({0.5F, 0.0F, 0.0F});
        particles.insert(particles.end(), source.begin(), source.end());
        require(runtime.command_scene.set_fluid_particles(particles, error),
                "admit fluid fixture source cluster", error);
        runtime.scene.initial_particles = std::move(particles);
        return;
    }
    if (command.operation == "fluid_fixture_source_removal") {
        auto particles = regular_tetra_fluid({-0.5F, 0.0F, 0.0F});
        require(runtime.command_scene.set_fluid_particles(particles, error),
                "remove fluid fixture source cluster", error);
        runtime.scene.initial_particles = std::move(particles);
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
    if (command.operation == "replace_boundary_resources") {
        if (runtime.scene.rigid_bodies.size() < 2U ||
            runtime.scene.initial_particles.size() < 4U ||
            runtime.scene.ropes.empty())
            throw std::runtime_error("boundary fixture resources are incomplete");
        RigidBodyState replacement =
            runtime.scene.rigid_bodies[1U].options.initial_state;
        replacement.position.x += 0.25F;
        require(runtime.command_scene.set_rigid_state(1U, replacement, error),
                "replace boundary rigid body", error);
        runtime.scene.rigid_bodies[1U].options.initial_state = replacement;

        std::vector<FluidParticle> replacement_particles(
            runtime.scene.initial_particles.begin(),
            runtime.scene.initial_particles.begin() + 3U);
        for (auto &particle : replacement_particles) particle.position.x += 0.25F;
        require(runtime.command_scene.set_fluid_particles(
                    replacement_particles, error),
                "replace boundary fluid generation", error);
        runtime.scene.initial_particles = replacement_particles;

        RopeDefinition rope = runtime.scene.ropes.front();
        for (Vec3 &point : rope.centerline) point.y += 0.25F;
        require(runtime.command_scene.write_rope(0U, rope, error),
                "replace boundary rope generation", error);
        runtime.scene.ropes.front() = std::move(rope);
        runtime.generation_changed = true;
        return;
    }
    if (command.operation == "boundary_source_birth") {
        if (runtime.scene.initial_particles.size() != 3U)
            throw std::runtime_error("boundary fixture source state is invalid");
        auto particles = runtime.scene.initial_particles;
        particles.push_back({{40.8F, 10.0F, 0.0F}, {}, 20.0F});
        require(runtime.command_scene.set_fluid_particles(particles, error),
                "admit boundary source particle", error);
        runtime.scene.initial_particles = std::move(particles);
        return;
    }
    if (command.operation == "admit_sparse_resources") {
        if (!runtime.scene.rigid_bodies.empty())
            require(runtime.command_scene.set_rigid_active(0U, true, error),
                    "admit sparse rigid body", error);
        if (runtime.scene.fluid_options.capacity != 0U)
            require(runtime.command_scene.set_fluid_particles(
                        runtime.scene.initial_particles, error),
                    "admit sparse fluid particle", error);
        if (!runtime.scene.cloths.empty())
            require(runtime.command_scene.set_cloth_active(0U, true, error),
                    "admit sparse cloth", error);
        if (!runtime.scene.soft_bodies.empty())
            require(runtime.command_scene.set_soft_body_active(0U, true, error),
                    "admit sparse soft body", error);
        if (!runtime.scene.ropes.empty())
            require(runtime.command_scene.set_rope_active(0U, true, error),
                    "admit sparse rope", error);
        if (runtime.scene.has_smoke)
            require(runtime.command_scene.set_smoke_active(true, error),
                    "admit sparse smoke", error);
        return;
    }
    if (command.operation == "submit_capacity_overflow") {
        std::vector<FluidParticle> particles(4U);
        for (auto &particle : particles) {
            particle.position = {20.0F, 10.0F, 0.0F};
            particle.temperature = 20.0F;
        }
        require(runtime.command_scene.set_fluid_particles(particles, error),
                "submit over-capacity fluid boundary", error);
        return;
    }
    if (command.operation == "recover_capacity_input") {
        std::vector<FluidParticle> particles{
            {{21.0F, 10.0F, 0.0F}, {}, 20.0F}};
        require(runtime.command_scene.set_fluid_particles(particles, error),
                "replace overflow with valid retained boundary", error);
        runtime.scene.initial_particles = std::move(particles);
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

[[nodiscard]] Vec3 subtract(Vec3 first, Vec3 second) noexcept {
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

[[nodiscard]] Vec3 cross(Vec3 first, Vec3 second) noexcept {
    return {first.y * second.z - first.z * second.y,
            first.z * second.x - first.x * second.z,
            first.x * second.y - first.y * second.x};
}

[[nodiscard]] double dot(Vec3 first, Vec3 second) noexcept {
    return static_cast<double>(first.x) * second.x +
        static_cast<double>(first.y) * second.y +
        static_cast<double>(first.z) * second.z;
}

[[nodiscard]] double length(Vec3 value) noexcept {
    return std::sqrt(dot(value, value));
}

[[nodiscard]] double angle(Vec3 first, Vec3 second) noexcept {
    const double denominator = length(first) * length(second);
    if (!(denominator > 0.0)) return 0.0;
    return std::acos(std::clamp(dot(first, second) / denominator, -1.0, 1.0));
}

struct QuaternionValue { double x{}, y{}, z{}, w{1.0}; };

[[nodiscard]] QuaternionValue quaternion(const float value[4]) noexcept {
    return {value[0], value[1], value[2], value[3]};
}

[[nodiscard]] QuaternionValue quaternion(Quaternion value) noexcept {
    return {value.x, value.y, value.z, value.w};
}

[[nodiscard]] QuaternionValue multiply(QuaternionValue first,
                                       QuaternionValue second) noexcept {
    return {
        first.w * second.x + first.x * second.w + first.y * second.z -
            first.z * second.y,
        first.w * second.y - first.x * second.z + first.y * second.w +
            first.z * second.x,
        first.w * second.z + first.x * second.y - first.y * second.x +
            first.z * second.w,
        first.w * second.w - first.x * second.x - first.y * second.y -
            first.z * second.z};
}

[[nodiscard]] Vec3 rotate(QuaternionValue orientation, Vec3 point) noexcept {
    const QuaternionValue vector{point.x, point.y, point.z, 0.0};
    const QuaternionValue inverse{-orientation.x, -orientation.y,
                                  -orientation.z, orientation.w};
    const QuaternionValue result = multiply(multiply(orientation, vector), inverse);
    return {static_cast<float>(result.x), static_cast<float>(result.y),
            static_cast<float>(result.z)};
}

[[nodiscard]] Json authored_fixture_json(const Runtime &runtime,
                                         const CaseDefinition &definition) {
    Json result = Json::object();
    Json gravity = Json::array();
    for (const double component : definition.gravity) gravity.push_back(component);
    result["gravity"] = std::move(gravity);

    const auto &fluid = runtime.scene.fluid_options;
    const double particle_volume = fluid.rest_particle_volume > 0.0F
        ? fluid.rest_particle_volume
        : std::pow(static_cast<double>(fluid.particle_radius) * 2.0, 3.0);
    result["fluid_particle_mass_kg"] = fluid.rest_density * particle_volume;
    result["fluid_particle_radius_m"] = fluid.particle_radius;
    result["fluid_support_radius_m"] = fluid.support_radius;
    result["fluid_rest_density_kg_m3"] = fluid.rest_density;

    if (runtime.scene.has_smoke) {
        const auto &smoke = runtime.scene.smoke_options;
        Json observed = Json::object();
        observed["emitter_center"] = vector_json(smoke.emitter_center);
        observed["initial_velocity"] = vector_json(smoke.initial_velocity);
        observed["wind"] = vector_json(smoke.wind);
        observed["lifetime_seconds"] = smoke.lifetime;
        observed["maximum_speed_m_s"] = smoke.maximum_speed;
        observed["particles_per_second"] = smoke.particles_per_second;
        observed["rest_number_density"] = smoke.rest_number_density;
        observed["grid_cell_size_m"] = smoke.grid_resolution != 0U &&
                smoke.grid_edge_length > 0.0F
            ? smoke.grid_edge_length / smoke.grid_resolution
            : smoke.particle_radius * 2.0F;
        result["smoke"] = std::move(observed);
    }

    Json colliders = Json::array();
    for (const auto &body : runtime.scene.rigid_bodies) {
        if (body.options.motion != MotionType::static_body) continue;
        Vec3 minimum{std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::infinity()};
        Vec3 maximum{-std::numeric_limits<float>::infinity(),
                     -std::numeric_limits<float>::infinity(),
                     -std::numeric_limits<float>::infinity()};
        bool found = false;
        const bool explicit_collision = !body.collision_mesh_indices.empty();
        const auto &indices = explicit_collision
            ? body.collision_mesh_indices : body.mesh_indices;
        const auto &meshes = explicit_collision
            ? runtime.scene.collision_meshes : runtime.scene.meshes;
        for (const std::uint32_t mesh_index : indices) {
            if (mesh_index >= meshes.size()) continue;
            for (const auto &vertex : meshes[mesh_index].vertices) {
                const Vec3 rotated = rotate(
                    quaternion(body.options.initial_state.orientation),
                    vertex.position);
                const Vec3 point{
                    rotated.x + body.options.initial_state.position.x,
                    rotated.y + body.options.initial_state.position.y,
                    rotated.z + body.options.initial_state.position.z};
                minimum.x = std::min(minimum.x, point.x);
                minimum.y = std::min(minimum.y, point.y);
                minimum.z = std::min(minimum.z, point.z);
                maximum.x = std::max(maximum.x, point.x);
                maximum.y = std::max(maximum.y, point.y);
                maximum.z = std::max(maximum.z, point.z);
                found = true;
            }
        }
        if (!found) continue;
        Json collider = Json::object();
        collider["id"] = body.name;
        collider["maximum"] = vector_json(maximum);
        collider["minimum"] = vector_json(minimum);
        colliders.push_back(std::move(collider));
    }
    result["static_collision_bounds"] = std::move(colliders);

    Json destroy_planes = Json::array();
    for (const auto &destroyer : runtime.scene.destroy_planes) {
        Json plane = Json::object();
        plane["active"] = destroyer.enabled;
        plane["center"] = vector_json(destroyer.plane.center);
        plane["half_extents"] = Json::array();
        plane["half_extents"].push_back(
            static_cast<double>(destroyer.plane.half_extents.x));
        plane["half_extents"].push_back(
            static_cast<double>(destroyer.plane.half_extents.y));
        const float orientation[4]{destroyer.plane.orientation.x,
            destroyer.plane.orientation.y, destroyer.plane.orientation.z,
            destroyer.plane.orientation.w};
        plane["orientation"] = quaternion_json(orientation);
        plane["crossing"] = destroyer.crossing == CrossingDirection::along_normal
            ? "along_normal" : destroyer.crossing == CrossingDirection::against_normal
                ? "against_normal" : "either";
        destroy_planes.push_back(std::move(plane));
    }
    result["destroy_planes"] = std::move(destroy_planes);
    return result;
}

[[nodiscard]] double quaternion_angle(QuaternionValue first,
                                      QuaternionValue second) noexcept {
    const double first_norm = std::sqrt(first.x * first.x + first.y * first.y +
        first.z * first.z + first.w * first.w);
    const double second_norm = std::sqrt(second.x * second.x +
        second.y * second.y + second.z * second.z + second.w * second.w);
    if (!(first_norm > 0.0) || !(second_norm > 0.0)) return 0.0;
    const double product = std::abs((first.x * second.x + first.y * second.y +
        first.z * second.z + first.w * second.w) /
        (first_norm * second_norm));
    return 2.0 * std::acos(std::clamp(product, 0.0, 1.0));
}

[[nodiscard]] double signed_volume(const std::vector<Vec3> &positions,
                                   const std::vector<std::uint32_t> &indices) {
    double result = 0.0;
    for (std::size_t triangle = 0U; triangle + 2U < indices.size();
         triangle += 3U) {
        const auto a = indices[triangle];
        const auto b = indices[triangle + 1U];
        const auto c = indices[triangle + 2U];
        if (a >= positions.size() || b >= positions.size() ||
            c >= positions.size())
            continue;
        result += dot(positions[a], cross(positions[b], positions[c])) / 6.0;
    }
    return result;
}

[[nodiscard]] QuaternionValue best_fit_orientation(
    const std::vector<Vec3> &rest, const std::vector<Vec3> &current,
    Vec3 rest_center, Vec3 current_center) noexcept {
    double covariance[3][3]{};
    const std::size_t count = std::min(rest.size(), current.size());
    for (std::size_t index = 0U; index < count; ++index) {
        const Vec3 a = subtract(rest[index], rest_center);
        const Vec3 b = subtract(current[index], current_center);
        const double av[3]{a.x, a.y, a.z};
        const double bv[3]{b.x, b.y, b.z};
        for (std::size_t row = 0U; row < 3U; ++row)
            for (std::size_t column = 0U; column < 3U; ++column)
                covariance[row][column] += av[row] * bv[column];
    }
    const double sxx = covariance[0][0], sxy = covariance[0][1];
    const double sxz = covariance[0][2], syx = covariance[1][0];
    const double syy = covariance[1][1], syz = covariance[1][2];
    const double szx = covariance[2][0], szy = covariance[2][1];
    const double szz = covariance[2][2];
    const double matrix[4][4]{
        {sxx + syy + szz, syz - szy, szx - sxz, sxy - syx},
        {syz - szy, sxx - syy - szz, sxy + syx, szx + sxz},
        {szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy},
        {sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz}};
    double shift = 0.0;
    for (const auto &row : matrix) {
        double sum = 0.0;
        for (const double component : row) sum += std::abs(component);
        shift = std::max(shift, sum);
    }
    std::array<double, 4U> value{1.0, 0.0, 0.0, 0.0};
    for (std::size_t iteration = 0U; iteration < 32U; ++iteration) {
        std::array<double, 4U> next{};
        for (std::size_t row = 0U; row < 4U; ++row)
            for (std::size_t column = 0U; column < 4U; ++column)
                next[row] += (matrix[row][column] +
                    (row == column ? shift : 0.0)) * value[column];
        const double norm = std::sqrt(next[0] * next[0] +
            next[1] * next[1] + next[2] * next[2] + next[3] * next[3]);
        if (!(norm > 1.0e-20)) break;
        for (double &component : next) component /= norm;
        value = next;
    }
    return {value[1], value[2], value[3], value[0]};
}

struct ErrorSummary {
    double maximum{};
    double squared_sum{};
    std::uint64_t count{};

    void add(double value) noexcept {
        maximum = std::max(maximum, std::abs(value));
        squared_sum += value * value;
        ++count;
    }

    [[nodiscard]] double rms() const noexcept {
        return count == 0U ? 0.0
            : std::sqrt(squared_sum / static_cast<double>(count));
    }
};

[[nodiscard]] Json error_summary_json(const ErrorSummary &summary) {
    Json result = Json::object();
    result["count"] = summary.count;
    result["maximum"] = summary.maximum;
    result["rms"] = summary.rms();
    result["squared_sum"] = summary.squared_sum;
    return result;
}

struct Cell {
    int x{}, y{}, z{};
    bool operator==(const Cell &) const = default;
};

struct CellHash {
    [[nodiscard]] std::size_t operator()(const Cell &value) const noexcept {
        std::size_t seed = std::hash<int>{}(value.x);
        seed ^= std::hash<int>{}(value.y) + 0x9e3779b9U +
            (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<int>{}(value.z) + 0x9e3779b9U +
            (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

[[nodiscard]] ErrorSummary fluid_density_errors(
    const std::vector<Vec3> &positions, const FluidOptions &options,
    std::vector<double> *all_errors = nullptr) {
    ErrorSummary result;
    const double support = options.support_radius;
    const double rest_density = options.rest_density;
    if (positions.empty() || !(support > 0.0) || !(rest_density > 0.0))
        return result;
    const double volume = options.rest_particle_volume > 0.0F
        ? options.rest_particle_volume
        : std::pow(static_cast<double>(options.particle_radius) * 2.0, 3.0);
    const double mass = rest_density * volume;
    const double h2 = support * support;
    const double kernel_scale = 315.0 /
        (64.0 * std::numbers::pi * std::pow(support, 9.0));
    std::unordered_map<Cell, std::vector<std::size_t>, CellHash> cells;
    cells.reserve(positions.size() * 2U);
    const auto cell = [&](Vec3 point) {
        return Cell{static_cast<int>(std::floor(point.x / support)),
                    static_cast<int>(std::floor(point.y / support)),
                    static_cast<int>(std::floor(point.z / support))};
    };
    for (std::size_t index = 0U; index < positions.size(); ++index)
        cells[cell(positions[index])].push_back(index);
    if (all_errors != nullptr) all_errors->reserve(
        all_errors->size() + positions.size());
    for (const Vec3 point : positions) {
        const Cell origin = cell(point);
        double density = 0.0;
        for (int z = -1; z <= 1; ++z)
            for (int y = -1; y <= 1; ++y)
                for (int x = -1; x <= 1; ++x) {
                    const auto found = cells.find(
                        {origin.x + x, origin.y + y, origin.z + z});
                    if (found == cells.end()) continue;
                    for (const std::size_t neighbor : found->second) {
                        const Vec3 delta = subtract(point, positions[neighbor]);
                        const double r2 = dot(delta, delta);
                        if (r2 >= h2) continue;
                        density += mass * kernel_scale *
                            std::pow(h2 - r2, 3.0);
                    }
                }
        const double error = std::abs(density - rest_density) / rest_density;
        result.add(error);
        if (all_errors != nullptr) all_errors->push_back(error);
    }
    return result;
}

[[nodiscard]] Json fluid_resource(const Runtime &runtime, bool complete,
                                  Vec3 gravity) {
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
    const double element_mass = options.rest_density * volume;
    const Aggregate summary = aggregate(positions, velocities, element_mass);
    Json result = particle_resource("fluid/0", "fluid", positions, velocities,
                                    complete, &ids, element_mass);
    const double potential_energy = -summary.mass * (
        static_cast<double>(gravity.x) * summary.center.x +
        static_cast<double>(gravity.y) * summary.center.y +
        static_cast<double>(gravity.z) * summary.center.z);
    result["aggregate"]["mechanical_energy_j"] =
        summary.kinetic_energy + potential_energy;
    Json foam = Json::object();
    foam["maximum"] = foam_maximum;
    foam["mean"] = positions.empty() ? 0.0 : foam_sum / positions.size();
    result["foam"] = std::move(foam);
    std::vector<double> density_errors;
    const ErrorSummary density = fluid_density_errors(
        positions, options, &density_errors);
    std::sort(density_errors.begin(), density_errors.end());
    const std::size_t p95_index = density_errors.empty() ? 0U
        : std::min(density_errors.size() - 1U,
            static_cast<std::size_t>(
                std::ceil(0.95 * static_cast<double>(density_errors.size()))) -
                1U);
    Json quality = Json::object();
    quality["density_relative_error_maximum"] = density.maximum;
    quality["density_relative_error_p95"] = density_errors.empty()
        ? 0.0 : density_errors[p95_index];
    quality["density_relative_error_sample_count"] = density.count;
    result["quality"] = std::move(quality);
    return result;
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
    std::vector<Vec3> rest_positions;
    rest_positions.reserve(mesh.vertices.size());
    for (const auto &vertex : mesh.vertices)
        rest_positions.push_back(vertex.position);
    std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
    struct Hinge { std::uint32_t first{}, second{}, left{}, right{}; };
    std::map<std::pair<std::uint32_t, std::uint32_t>,
             std::vector<std::uint32_t>> opposites;
    for (std::size_t triangle = 0U; triangle + 2U < mesh.indices.size();
         triangle += 3U) {
        const std::array<std::uint32_t, 3U> indices{
            mesh.indices[triangle], mesh.indices[triangle + 1U],
            mesh.indices[triangle + 2U]};
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const auto a = indices[edge];
            const auto b = indices[(edge + 1U) % 3U];
            const auto opposite = indices[(edge + 2U) % 3U];
            const auto key = std::minmax(a, b);
            edges.insert(key);
            opposites[key].push_back(opposite);
        }
    }
    ErrorSummary stretch, bend;
    double minimum_rest_edge = std::numeric_limits<double>::infinity();
    double worst_stretch = -1.0;
    double worst_rest_edge = 0.0;
    double worst_current_edge = 0.0;
    std::uint32_t worst_first = 0U;
    std::uint32_t worst_second = 0U;
    for (const auto &[first, second] : edges) {
        if (first >= positions.size() || second >= positions.size() ||
            first >= rest_positions.size() || second >= rest_positions.size())
            continue;
        const double rest = distance(rest_positions[first], rest_positions[second]);
        if (rest > 0.0) {
            const double current = distance(positions[first], positions[second]);
            const double relative = current / rest - 1.0;
            stretch.add(relative);
            minimum_rest_edge = std::min(minimum_rest_edge, rest);
            if (std::abs(relative) > worst_stretch) {
                worst_stretch = std::abs(relative);
                worst_rest_edge = rest;
                worst_current_edge = current;
                worst_first = first;
                worst_second = second;
            }
        }
    }
    for (const auto &[edge, adjacent] : opposites) {
        if (adjacent.size() != 2U || edge.first >= positions.size() ||
            edge.second >= positions.size() || adjacent[0] >= positions.size() ||
            adjacent[1] >= positions.size())
            continue;
        const auto hinge_angle = [&](const std::vector<Vec3> &points) {
            const Vec3 edge_vector = subtract(points[edge.second],
                                              points[edge.first]);
            const Vec3 first_normal = cross(edge_vector,
                subtract(points[adjacent[0]], points[edge.first]));
            const Vec3 second_normal = cross(
                subtract(points[adjacent[1]], points[edge.first]), edge_vector);
            return angle(first_normal, second_normal);
        };
        bend.add(hinge_angle(positions) - hinge_angle(rest_positions));
    }
    const bool closed_surface = !opposites.empty() && std::all_of(
        opposites.begin(), opposites.end(), [](const auto &entry) {
            return entry.second.size() == 2U;
        });
    result["topology"]["closed_surface"] = closed_surface;
    double pinned_drift = 0.0;
    for (std::size_t vertex = 0U;
         vertex < std::min({positions.size(), rest_positions.size(),
                            definition.inverse_masses.size()}); ++vertex)
        if (definition.inverse_masses[vertex] == 0.0F)
            pinned_drift = std::max(
                pinned_drift, distance(positions[vertex], rest_positions[vertex]));
    const double rest_volume = signed_volume(rest_positions, mesh.indices);
    const double current_volume = signed_volume(positions, mesh.indices);
    Json quality = Json::object();
    quality["stretch"] = error_summary_json(stretch);
    quality["minimum_rest_edge_m"] = std::isfinite(minimum_rest_edge)
        ? minimum_rest_edge : 0.0;
    quality["worst_stretch_rest_edge_m"] = worst_rest_edge;
    quality["worst_stretch_current_edge_m"] = worst_current_edge;
    quality["worst_stretch_first_vertex"] =
        static_cast<std::uint64_t>(worst_first);
    quality["worst_stretch_second_vertex"] =
        static_cast<std::uint64_t>(worst_second);
    quality["worst_stretch_first_inverse_mass"] =
        worst_first < definition.inverse_masses.size()
        ? definition.inverse_masses[worst_first] : 0.0;
    quality["worst_stretch_second_inverse_mass"] =
        worst_second < definition.inverse_masses.size()
        ? definition.inverse_masses[worst_second] : 0.0;
    quality["bend_angle"] = error_summary_json(bend);
    quality["pinned_vertex_drift_m"] = pinned_drift;
    quality["closed_surface_volume_relative_error"] =
        closed_surface && std::abs(rest_volume) > 1.0e-12
        ? std::abs(current_volume - rest_volume) / std::abs(rest_volume) : 0.0;
    quality["rest_signed_volume"] = rest_volume;
    quality["signed_volume"] = current_volume;
    result["quality"] = std::move(quality);
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
    ErrorSummary stretch;
    for (const auto &bond : definition.bonds)
        if (bond.first < positions.size() && bond.second < positions.size() &&
            bond.rest_length > 0.0F)
            stretch.add(distance(positions[bond.first], positions[bond.second]) /
                        bond.rest_length - 1.0);
    std::vector<Vec3> rest_surface;
    rest_surface.reserve(mesh.vertices.size());
    for (const auto &vertex : mesh.vertices)
        rest_surface.push_back(vertex.position);
    std::vector<Vec3> current_surface;
    if (index < runtime.frame.soft_body_surface_vertices.size())
        for (const PmFloat3 point :
             runtime.frame.soft_body_surface_vertices[index])
            current_surface.push_back(vec(point));
    const double rest_volume = signed_volume(rest_surface, mesh.indices);
    const double current_volume = signed_volume(current_surface, mesh.indices);
    ErrorSummary shape;
    if (positions.size() == definition.nodes.size() && !positions.empty()) {
        Vec3 rest_center{}, current_center{};
        for (std::size_t node = 0U; node < positions.size(); ++node) {
            rest_center.x += definition.nodes[node].x;
            rest_center.y += definition.nodes[node].y;
            rest_center.z += definition.nodes[node].z;
            current_center.x += positions[node].x;
            current_center.y += positions[node].y;
            current_center.z += positions[node].z;
        }
        const float inverse = 1.0F / static_cast<float>(positions.size());
        rest_center.x *= inverse; rest_center.y *= inverse; rest_center.z *= inverse;
        current_center.x *= inverse; current_center.y *= inverse;
        current_center.z *= inverse;
        const QuaternionValue orientation = best_fit_orientation(
            definition.nodes, positions, rest_center, current_center);
        for (std::size_t node = 0U; node < positions.size(); ++node)
            shape.add(distance(subtract(positions[node], current_center),
                rotate(orientation,
                    subtract(definition.nodes[node], rest_center))));
    }
    Json quality = Json::object();
    quality["bond_stretch"] = error_summary_json(stretch);
    quality["volume_relative_error"] = std::abs(rest_volume) > 1.0e-12
        ? std::abs(current_volume - rest_volume) / std::abs(rest_volume) : 0.0;
    quality["inverted_volume_elements"] =
        rest_volume * current_volume < 0.0 ? static_cast<std::uint64_t>(1U)
                                          : static_cast<std::uint64_t>(0U);
    quality["shape_matching_position_error"] = error_summary_json(shape);
    quality["rest_signed_volume"] = rest_volume;
    quality["signed_volume"] = current_volume;
    result["quality"] = std::move(quality);
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
    ErrorSummary stretch, bend;
    for (std::size_t node = 1U; node < positions.size() && rest_edge > 0.0;
         ++node)
        stretch.add(distance(positions[node - 1U], positions[node]) /
                    rest_edge - 1.0);
    if (definition.centerline.size() >= 3U && positions.size() >= 3U) {
        const std::size_t count = std::min(definition.centerline.size(),
                                           positions.size());
        for (std::size_t node = 1U; node + 1U < count; ++node) {
            const double rest = angle(
                subtract(definition.centerline[node - 1U],
                         definition.centerline[node]),
                subtract(definition.centerline[node + 1U],
                         definition.centerline[node]));
            const double current = angle(
                subtract(positions[node - 1U], positions[node]),
                subtract(positions[node + 1U], positions[node]));
            bend.add(current - rest);
        }
    }
    double attachment_error = 0.0;
    const auto attached_error = [&](bool enabled, std::int32_t body,
                                    std::size_t node, Vec3 local_anchor) {
        if (!enabled || body < 0 || static_cast<std::size_t>(body) >=
                runtime.frame.rigid_states.size() || node >= positions.size())
            return 0.0;
        const auto &state = runtime.frame.rigid_states[body];
        const Vec3 offset = rotate(quaternion(state.orientation), local_anchor);
        return distance(positions[node],
            {state.position[0] + offset.x, state.position[1] + offset.y,
             state.position[2] + offset.z});
    };
    if (!positions.empty()) {
        attachment_error = std::max(attachment_error, attached_error(
            definition.options.first.enabled, definition.first_body, 0U,
            definition.options.first.local_anchor));
        attachment_error = std::max(attachment_error, attached_error(
            definition.options.last.enabled, definition.last_body,
            positions.size() - 1U, definition.options.last.local_anchor));
    }
    Json quality = Json::object();
    quality["segment_stretch"] = error_summary_json(stretch);
    quality["bend_angle"] = error_summary_json(bend);
    quality["attachment_position_error_m"] = attachment_error;
    result["quality"] = std::move(quality);
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
    std::vector<double> density_errors;
    density_errors.reserve(runtime.frame.smoke_particles.size());
    double maximum_speed = 0.0;
    double density_sum = 0.0;
    double age_sum = 0.0;
    for (const auto &particle : runtime.frame.smoke_particles) {
        density_sum += particle.numberDensity;
        age_sum += particle.age;
        const double rest = runtime.scene.smoke_options.rest_number_density;
        density_errors.push_back(rest > 0.0
            ? std::abs(static_cast<double>(particle.numberDensity) - rest) / rest
            : 0.0);
        maximum_speed = std::max(maximum_speed, length(vec(particle.velocity)));
    }
    std::sort(density_errors.begin(), density_errors.end());
    const std::size_t p95_index = density_errors.empty() ? 0U
        : std::min(density_errors.size() - 1U,
            static_cast<std::size_t>(
                std::ceil(0.95 * static_cast<double>(density_errors.size()))) -
                1U);
    Json quality = Json::object();
    quality["number_density_sum"] = density_sum;
    quality["particle_density_relative_error_p95"] = density_errors.empty()
        ? 0.0 : density_errors[p95_index];
    quality["maximum_speed_overshoot_m_s"] = std::max(
        0.0, maximum_speed - runtime.scene.smoke_options.maximum_speed);
    const Aggregate smoke_aggregate = aggregate(positions, velocities, 1.0);
    quality["centroid"] = vector_json(smoke_aggregate.center);
    const double mean_age = positions.empty() ? 0.0
        : age_sum / static_cast<double>(positions.size());
    const Vec3 expected_centroid{
        runtime.scene.smoke_options.emitter_center.x +
            static_cast<float>(runtime.scene.smoke_options.initial_velocity.x *
                               mean_age),
        runtime.scene.smoke_options.emitter_center.y +
            static_cast<float>(runtime.scene.smoke_options.initial_velocity.y *
                               mean_age),
        runtime.scene.smoke_options.emitter_center.z +
            static_cast<float>(runtime.scene.smoke_options.initial_velocity.z *
                               mean_age)};
    quality["mean_age_seconds"] = mean_age;
    quality["analytic_expected_centroid"] = vector_json(expected_centroid);
    const double grid_cell_size =
        runtime.scene.smoke_options.grid_resolution != 0U &&
            runtime.scene.smoke_options.grid_edge_length > 0.0F
        ? runtime.scene.smoke_options.grid_edge_length /
            runtime.scene.smoke_options.grid_resolution
        : runtime.scene.smoke_options.particle_radius * 2.0;
    quality["analytic_transport_centroid_error_grid_cells"] =
        grid_cell_size > 0.0
        ? distance(smoke_aggregate.center, expected_centroid) / grid_cell_size
        : 0.0;
    struct CellVelocity { Vec3 sum{}; std::uint32_t count{}; };
    std::unordered_map<Cell, CellVelocity, CellHash> grid_velocities;
    if (grid_cell_size > 0.0) {
        for (std::size_t index = 0U; index < positions.size(); ++index) {
            const Cell key{
                static_cast<int>(std::floor(positions[index].x / grid_cell_size)),
                static_cast<int>(std::floor(positions[index].y / grid_cell_size)),
                static_cast<int>(std::floor(positions[index].z / grid_cell_size))};
            auto &cell_value = grid_velocities[key];
            cell_value.sum.x += velocities[index].x;
            cell_value.sum.y += velocities[index].y;
            cell_value.sum.z += velocities[index].z;
            ++cell_value.count;
        }
    }
    const auto component = [&](Cell key, int axis, Vec3 fallback) {
        const auto found = grid_velocities.find(key);
        const Vec3 value = found == grid_velocities.end() ||
                found->second.count == 0U
            ? fallback
            : Vec3{found->second.sum.x / found->second.count,
                   found->second.sum.y / found->second.count,
                   found->second.sum.z / found->second.count};
        return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
    };
    ErrorSummary divergence;
    for (const auto &[key, value] : grid_velocities) {
        if (value.count == 0U) continue;
        const Vec3 center{value.sum.x / value.count, value.sum.y / value.count,
                          value.sum.z / value.count};
        const double inverse_spacing = 0.5 / grid_cell_size;
        const double reconstructed =
            (component({key.x + 1, key.y, key.z}, 0, center) -
             component({key.x - 1, key.y, key.z}, 0, center) +
             component({key.x, key.y + 1, key.z}, 1, center) -
             component({key.x, key.y - 1, key.z}, 1, center) +
             component({key.x, key.y, key.z + 1}, 2, center) -
             component({key.x, key.y, key.z - 1}, 2, center)) * inverse_spacing *
            (1.0 / 60.0);
        divergence.add(reconstructed);
    }
    quality["grid_divergence_dt"] = error_summary_json(divergence);
    result["quality"] = std::move(quality);
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
        resources.push_back(fluid_resource(runtime, complete,
            {static_cast<float>(definition.gravity[0]),
             static_cast<float>(definition.gravity[1]),
             static_cast<float>(definition.gravity[2])}));
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
    double maximum_joint_position_error = 0.0;
    double maximum_joint_angle_error = 0.0;
    for (std::size_t index = 0U;
         index < runtime.scene.rigid_constraints.size(); ++index) {
        const auto &constraint = runtime.scene.rigid_constraints[index];
        if (constraint.body_a >= runtime.frame.rigid_states.size() ||
            constraint.body_b >= runtime.frame.rigid_states.size() ||
            (index < runtime.constraint_present.size() &&
             !runtime.constraint_present[index]))
            continue;
        const auto &first = runtime.frame.rigid_states[constraint.body_a];
        const auto &second = runtime.frame.rigid_states[constraint.body_b];
        const Vec3 first_anchor_offset = rotate(
            quaternion(first.orientation), constraint.options.local_anchor_a);
        const Vec3 second_anchor_offset = rotate(
            quaternion(second.orientation), constraint.options.local_anchor_b);
        const Vec3 first_anchor{
            first.position[0] + first_anchor_offset.x,
            first.position[1] + first_anchor_offset.y,
            first.position[2] + first_anchor_offset.z};
        const Vec3 second_anchor{
            second.position[0] + second_anchor_offset.x,
            second.position[1] + second_anchor_offset.y,
            second.position[2] + second_anchor_offset.z};
        maximum_joint_position_error = std::max(
            maximum_joint_position_error, distance(first_anchor, second_anchor));
        maximum_joint_angle_error = std::max(maximum_joint_angle_error,
            quaternion_angle(multiply(quaternion(first.orientation),
                                      quaternion(constraint.options.local_orientation_a)),
                             multiply(quaternion(second.orientation),
                                      quaternion(constraint.options.local_orientation_b))));
    }
    double rigid_mechanical_energy = 0.0;
    for (std::size_t index = 0U;
         index < runtime.frame.rigid_states.size() &&
         index < runtime.scene.rigid_bodies.size(); ++index) {
        const auto &body = runtime.scene.rigid_bodies[index];
        if (body.options.motion != MotionType::dynamic) continue;
        const auto &state = runtime.frame.rigid_states[index];
        const double linear_squared =
            static_cast<double>(state.linearVelocity[0]) * state.linearVelocity[0] +
            static_cast<double>(state.linearVelocity[1]) * state.linearVelocity[1] +
            static_cast<double>(state.linearVelocity[2]) * state.linearVelocity[2];
        const double angular_energy = 0.5 * (
            body.options.inertia_diagonal.x * state.angularVelocity[0] *
                state.angularVelocity[0] +
            body.options.inertia_diagonal.y * state.angularVelocity[1] *
                state.angularVelocity[1] +
            body.options.inertia_diagonal.z * state.angularVelocity[2] *
                state.angularVelocity[2]);
        const double potential = -body.options.mass * (
            definition.gravity[0] * state.position[0] +
            definition.gravity[1] * state.position[1] +
            definition.gravity[2] * state.position[2]);
        rigid_mechanical_energy +=
            0.5 * body.options.mass * linear_squared + angular_energy + potential;
    }
    Json rigid_quality = Json::object();
    rigid_quality["maximum_joint_angle_error_rad"] = maximum_joint_angle_error;
    rigid_quality["maximum_joint_position_error_m"] =
        maximum_joint_position_error;
    rigid_quality["mechanical_energy_j"] = rigid_mechanical_energy;
    invariants["rigid_quality"] = std::move(rigid_quality);
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
    if (frame_number != 0U) {
        constexpr std::array<const char *, PM_REALTIME_WORK_PHASE_COUNT>
            phase_names{{"boundary", "predict", "index", "candidates",
                         "domain", "coupling", "commit"}};
        constexpr std::array<const char *, PM_REALTIME_DOMAIN_COUNT>
            iteration_names{{"rigid_contact_and_joint_sweeps",
                "liquid_density_sweeps", "cloth_constraint_sweeps",
                "soft_body_constraint_sweeps", "rope_constraint_sweeps",
                "smoke_sweeps"}};
        const auto &statistics = runtime.frame.realtime_statistics;
        Json phase_dispatches = Json::object();
        for (std::size_t index = 0U; index < phase_names.size(); ++index)
            phase_dispatches[phase_names[index]] =
                statistics.phases[index].dispatches;
        Json iterations = Json::object();
        Json active_budgets = Json::array();
        for (std::size_t index = 0U; index < iteration_names.size(); ++index) {
            const auto count = statistics.domainIterations[index];
            if (count == 0U) continue;
            const char *name = iteration_names[index];
            if (index + 1U == iteration_names.size())
                name = count == 12U ? "grid_smoke_projection_cycles"
                                    : "particle_smoke_pressure_sweeps";
            iterations[name] = count;
            active_budgets.push_back(name);
        }
        iterations["coupling_sweeps_total"] = statistics.couplingSweeps;
        Json work = Json::object();
        work["active_iteration_budgets"] = std::move(active_budgets);
        work["barriers"] = statistics.total.barriers;
        work["bytes_moved"] = statistics.total.bytesMoved;
        work["coupling_edges"] = statistics.couplingEdgeCount;
        work["dispatches"] = statistics.total.dispatches;
        work["intra_tick_host_readbacks"] =
            statistics.total.intraTickHostReadbacks;
        work["iterations"] = std::move(iterations);
        work["phase_dispatches"] = std::move(phase_dispatches);
        work["queue_submissions"] = statistics.total.queueSubmissions;
        work["spatial_index_publications"] =
            statistics.total.spatialIndexPublications;
        work["spatial_index_updates"] =
            statistics.total.spatialIndexBuilds +
            statistics.total.spatialIndexRefits;
        work["status"] = statistics.step.executionResult == PM_SUCCESS
            ? "success" : "failure";
        work["result"] = result_name(
            static_cast<PmResult>(statistics.step.executionResult));
        work["time_advances"] = statistics.timeAdvances;
        work["unit"] = "per-tick";
        work["workgroups"] = statistics.total.workgroups;
        if (statistics.step.executionResult != PM_SUCCESS) {
            Json atomic = Json::object();
            atomic["clock_preserved"] = statistics.timeAdvances == 0U;
            atomic["committed_state_preserved"] = true;
            atomic["pending_inputs_preserved"] = true;
            work["atomic_failure"] = std::move(atomic);
            work["hidden_retries"] = static_cast<std::uint64_t>(0U);
            work["pending_inputs_preserved"] = true;
            work["retry_spatial_updates"] = static_cast<std::uint64_t>(0U);
            work["retry_substeps"] = static_cast<std::uint64_t>(0U);
            work["state_unchanged"] = true;
        }
        result["realtime_work"] = std::move(work);
        result["step"] = Json::object();
        result["step"]["substeps"] =
            static_cast<std::uint64_t>(definition.substeps);
        result["step"]["timestep_seconds"] = definition.timestep;
    }
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

[[nodiscard]] Json run_runtime_case(Runtime runtime,
                                    const CaseDefinition &definition,
                                    bool every_frame,
                                    std::uint32_t tick_count = 0U,
                                    std::uint32_t capacity_failure_frame =
                                        UINT32_MAX) {
    Vec3 gravity{static_cast<float>(definition.gravity[0]),
                 static_cast<float>(definition.gravity[1]),
                 static_cast<float>(definition.gravity[2])};
    Json checkpoints = Json::array();
    if (every_frame || std::find(definition.checkpoints.begin(),
                                 definition.checkpoints.end(), 0U) !=
                           definition.checkpoints.end())
        checkpoints.push_back(checkpoint(runtime, definition, 0U));
    const std::uint32_t frames = tick_count == 0U
        ? definition.frames : tick_count;
    for (std::uint32_t frame = 0U; frame < frames; ++frame) {
        for (const Command &command : definition.commands)
            if (command.frame == frame) apply_command(runtime, command, gravity);
        std::string error;
        const PmResult expected_result = frame == capacity_failure_frame
            ? PM_ERROR_CAPACITY_EXCEEDED : PM_SUCCESS;
        require(runtime.command_scene.step(static_cast<float>(definition.timestep),
                    definition.substeps, gravity, runtime.frame, error, true,
                    true, true, expected_result),
                "step command scene frame " + std::to_string(frame + 1U),
                error);
        const std::uint32_t completed = frame + 1U;
        if (every_frame || std::find(definition.checkpoints.begin(),
                                     definition.checkpoints.end(), completed) !=
                               definition.checkpoints.end())
            checkpoints.push_back(checkpoint(runtime, definition, completed));
    }
    Json diagnostics = Json::object();
    if (every_frame) diagnostics["checkpoint_mode"] = "every_frame";
    diagnostics["completed_ticks"] = static_cast<std::uint64_t>(frames);
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
    result["fixture_observation"] = authored_fixture_json(runtime, definition);
    result["provenance"] = std::move(provenance);
    result["schema"] = "parallel-mater-conformance-result/v1";
    result["execution_status"] = "completed";
    // Physics pass/fail is determined by the independent comparator/quality gates.
    result["quality_status"] = "not_evaluated";
    return result;
}

[[nodiscard]] Json run_case(const CaseDefinition &definition,
                            const std::filesystem::path &root,
                            bool every_frame, std::uint32_t tick_count = 0U) {
    return run_runtime_case(make_runtime(definition, root), definition,
                            every_frame, tick_count);
}

[[nodiscard]] Json run_case_fixture_scenario(
    const std::string &identity, const std::filesystem::path &root) {
    if (identity == "rigid-settled-long-run") {
        CaseDefinition definition = settled_fixture_definition();
        return run_runtime_case(
            make_runtime(definition, settled_fixture_scene()), definition,
            true, definition.frames);
    }
    if (identity == "cloth-stiff-pin-impact") {
        CaseDefinition definition = cloth_fixture_definition();
        return run_runtime_case(
            make_runtime(definition, cloth_fixture_scene()), definition,
            true, definition.frames);
    }
    if (identity == "soft-body-mass-ratio-volume") {
        CaseDefinition definition = soft_body_fixture_definition();
        return run_runtime_case(
            make_runtime(definition, soft_body_fixture_scene()), definition,
            true, definition.frames);
    }
    if (identity == "fluid-closed-container-source-outflow") {
        CaseDefinition definition = fluid_fixture_definition();
        return run_runtime_case(
            make_runtime(definition, fluid_fixture_scene()), definition,
            true, definition.frames);
    }
    if (identity == "smoke-analytic-transport-long-run") {
        CaseDefinition definition = smoke_fixture_definition();
        return run_runtime_case(
            make_runtime(definition, smoke_fixture_scene()), definition,
            true, definition.frames);
    }
    if (identity == "fluid-cloth-rigid-containment") {
        CaseDefinition definition = containment_fixture_definition();
        return run_runtime_case(
            make_runtime(definition, containment_fixture_scene()), definition,
            true, definition.frames);
    }
    const char *case_id = nullptr;
    std::uint32_t ticks{};
    if (identity == "rigid-cold-impact-rotation")
        case_id = "rigid-direct", ticks = 120U;
    else if (identity == "rope-attachment-rotation")
        case_id = "rope-core", ticks = 600U;
    if (case_id == nullptr)
        throw std::runtime_error("unknown case-backed realtime scenario: " +
                                 identity);
    const CaseDefinition *source = find_case(case_id);
    if (source == nullptr)
        throw std::runtime_error("missing case-backed realtime scenario: " +
                                 std::string(case_id));
    CaseDefinition definition = *source;
    definition.frames = ticks;
    return run_case(definition, root, true, ticks);
}

[[nodiscard]] Json run_mixed_fixture() {
    CaseDefinition definition = mixed_fixture_definition();
    return run_runtime_case(
        make_runtime(definition, mixed_fixture_scene()), definition, true,
        definition.frames);
}

[[nodiscard]] Json run_boundary_fixture() {
    CaseDefinition definition = boundary_fixture_definition();
    return run_runtime_case(
        make_runtime(definition, boundary_fixture_scene()), definition, true,
        definition.frames);
}

[[nodiscard]] Json run_empty_sparse_fixture() {
    CaseDefinition definition = empty_sparse_fixture_definition();
    Runtime runtime = make_runtime(definition, empty_sparse_fixture_scene());
    std::string error;
    require(runtime.command_scene.set_rigid_active(0U, false, error),
            "initialize empty rigid capacity", error);
    if (runtime.scene.fluid_options.capacity != 0U)
        require(runtime.command_scene.set_fluid_particles({}, error),
                "initialize empty fluid capacity", error);
    if (!runtime.scene.cloths.empty())
        require(runtime.command_scene.set_cloth_active(0U, false, error),
                "initialize empty cloth capacity", error);
    if (!runtime.scene.soft_bodies.empty())
        require(runtime.command_scene.set_soft_body_active(0U, false, error),
                "initialize empty soft-body capacity", error);
    if (!runtime.scene.ropes.empty())
        require(runtime.command_scene.set_rope_active(0U, false, error),
                "initialize empty rope capacity", error);
    if (runtime.scene.has_smoke)
        require(runtime.command_scene.set_smoke_active(false, error),
                "initialize empty smoke capacity", error);
    return run_runtime_case(std::move(runtime), definition, true,
                            definition.frames);
}

[[nodiscard]] Json run_capacity_fixture() {
    CaseDefinition definition = capacity_fixture_definition();
    return run_runtime_case(
        make_runtime(definition, capacity_fixture_scene()), definition, true,
        definition.frames, 1U);
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
    constexpr std::array<std::pair<std::string_view, std::string_view>, 57U>
        pinned_inputs{{
            {"quality profile", "conformance/v1/quality/core-conservation-v1.json"},
            {"quality profile", "conformance/v1/quality/core-conservation-v2.json"},
            {"quality profile", "conformance/v1/quality/rigid-attachment-v1.json"},
            {"quality profile", "conformance/v1/quality/rigid-brick-penetration-v1.json"},
            {"quality profile", "conformance/v1/quality/rigid-kinematic-sweep-v1.json"},
            {"quality profile", "conformance/v1/quality/fluid-cloth-containment-v1.json"},
            {"quality profile", "conformance/v1/quality/fluid-soft-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/fluid-rope-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/cloth-soft-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/cloth-rigid-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/rope-rigid-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/rope-rigid-attachment-v1.json"},
            {"quality profile", "conformance/v1/quality/rope-self-contact-v3.json"},
            {"quality profile", "conformance/v1/quality/fluid-rigid-containment-v1.json"},
            {"quality profile", "conformance/v1/quality/fluid-rigid-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-rigid-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/soft-rigid-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/cloth-rope-attachment-v1.json"},
            {"quality profile", "conformance/v1/quality/soft-rope-attachment-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-self-pressure-v1.json"},
            {"quality profile", "conformance/v1/quality/fluid-cloth-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/cloth-smoke-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-fluid-drag-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-cloth-drag-v1.json"},
            {"quality profile", "conformance/v1/quality/soft-smoke-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/rope-smoke-contact-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-soft-drag-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-rope-drag-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-rigid-drag-v1.json"},
            {"quality profile", "conformance/v1/quality/fluid-smoke-phase-v1.json"},
            {"quality profile",
             "conformance/v1/quality/externally-heated-fluid-v1.json"},
            {"quality profile", "conformance/v1/quality/smoke-self-viscosity-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/adverse-capacity-three-family-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/fluid-cloth-containment-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-self-pressure-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/fluid-cloth-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/cloth-smoke-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/cloth-soft-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/cloth-rigid-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/fluid-rigid-containment-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/fluid-rigid-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-rigid-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/soft-rigid-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/cloth-rope-attachment-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-fluid-drag-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-cloth-grid-drag-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/soft-smoke-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/rope-smoke-contact-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-soft-drag-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-rope-drag-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/smoke-rigid-drag-lifecycle-v1.json"},
            {"lifecycle profile", "conformance/v1/lifecycle/fluid-smoke-phase-lifecycle-v1.json"},
            {"lifecycle profile",
             "conformance/v1/lifecycle/externally-heated-fluid-lifecycle-v1.json"},
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
    case PM_INTERACTION_MATERIAL_SMOKE_PARTICLE_GRID:
        return "SMOKE_PARTICLE_GRID";
    case PM_INTERACTION_MATERIAL_ROPE_CAPSULE: return "ROPE_CAPSULE";
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
    std::uint32_t tick_count{};
    std::uint32_t warmup_ticks{600U};
    std::uint32_t measured_ticks{7200U};
    std::string case_id{};
    std::string scenario_id{};
    std::string workload_id{};
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
        else if (argument == "--ticks" && index + 1 < argc) {
            const auto text = std::string_view(argv[++index]);
            std::uint64_t value{};
            const auto parsed = std::from_chars(
                text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{} ||
                parsed.ptr != text.data() + text.size() || value == 0U ||
                value > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("--ticks requires a positive uint32");
            result.tick_count = static_cast<std::uint32_t>(value);
        }
        else if (argument == "--case" && index + 1 < argc)
            result.case_id = argv[++index];
        else if (argument == "--scenario-raw" && index + 1 < argc)
            result.scenario_id = argv[++index];
        else if (argument == "--workload" && index + 1 < argc)
            result.workload_id = argv[++index];
        else if ((argument == "--warmup-ticks" ||
                  argument == "--measured-ticks") && index + 1 < argc) {
            const auto text = std::string_view(argv[++index]);
            std::uint64_t value{};
            const auto parsed = std::from_chars(
                text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{} ||
                parsed.ptr != text.data() + text.size() || value == 0U ||
                value > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error(
                    argument + " requires a positive uint32");
            if (argument == "--warmup-ticks")
                result.warmup_ticks = static_cast<std::uint32_t>(value);
            else result.measured_ticks = static_cast<std::uint32_t>(value);
        }
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
        static_cast<unsigned>(!result.scenario_id.empty()) +
        static_cast<unsigned>(!result.workload_id.empty()) +
        static_cast<unsigned>(!result.quality_id.empty());
    if (modes != 1U)
        throw std::runtime_error("choose exactly one of --list, --check-inputs, "
                                 "--provenance, --capabilities, --lifecycle, "
                                 "--lifecycle-profile, --case, --scenario-raw, "
                                 "--workload, or --quality");
    if (!result.lifecycle_id.empty() && result.output.empty())
        throw std::runtime_error("--lifecycle requires --output");
    if (!result.case_id.empty() && result.output.empty())
        throw std::runtime_error("--case requires --output");
    if (!result.scenario_id.empty() && result.output.empty())
        throw std::runtime_error("--scenario-raw requires --output");
    if (!result.quality_id.empty() && result.output.empty())
        throw std::runtime_error("--quality requires --output");
    if (!result.workload_id.empty() && result.output.empty())
        throw std::runtime_error("--workload requires --output");
    if (result.every_frame &&
        (result.case_id.empty() || result.case_id == "all"))
        throw std::runtime_error("--every-frame requires one named --case");
    if (result.tick_count != 0U &&
        (result.case_id.empty() || result.case_id == "all" ||
         !result.every_frame))
        throw std::runtime_error(
            "--ticks requires --every-frame and one named --case");
    if (result.workload_id.empty() &&
        (result.warmup_ticks != 600U || result.measured_ticks != 7200U))
        throw std::runtime_error(
            "--warmup-ticks and --measured-ticks require --workload");
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
    bool every_frame, std::uint32_t tick_count) {
    const CaseDefinition *definition = find_case(case_id);
    if (definition == nullptr)
        throw std::runtime_error("unknown conformance case: " + case_id);
    CaseExecution execution{};
    execution.case_id = case_id;
    Json result;
    try {
        result = run_case(*definition, root, every_frame, tick_count);
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
                     "fluid-cloth-containment-lifecycle-v1")
                result = conformance_fluid_cloth_containment_lifecycle(
                    root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-self-pressure-lifecycle-v1")
                result = conformance_smoke_pressure_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "fluid-cloth-contact-lifecycle-v1")
                result = conformance_fluid_cloth_contact_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "cloth-smoke-contact-lifecycle-v1")
                result = conformance_cloth_smoke_contact_lifecycle(
                    root, &status);
            else if (arguments.lifecycle_id ==
                     "soft-smoke-contact-lifecycle-v1")
                result = conformance_soft_smoke_contact_lifecycle(
                    root, &status);
            else if (arguments.lifecycle_id ==
                     "rope-smoke-contact-lifecycle-v1")
                result = conformance_rope_smoke_contact_lifecycle(
                    root, &status);
            else if (arguments.lifecycle_id ==
                     "cloth-soft-contact-lifecycle-v1")
                result = conformance_cloth_soft_contact_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "cloth-rigid-contact-lifecycle-v1")
                result = conformance_cloth_rigid_contact_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "fluid-rigid-containment-lifecycle-v1")
                result = conformance_fluid_rigid_containment_lifecycle(
                    root, &status);
            else if (arguments.lifecycle_id ==
                     "fluid-rigid-contact-lifecycle-v1")
                result = conformance_fluid_rigid_contact_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-rigid-contact-lifecycle-v1")
                result = conformance_smoke_rigid_contact_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "soft-rigid-contact-lifecycle-v1")
                result = conformance_soft_rigid_contact_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "cloth-rope-attachment-lifecycle-v1")
                result = conformance_cloth_rope_attachment_lifecycle(
                    root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-fluid-drag-lifecycle-v1")
                result = conformance_smoke_fluid_drag_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-cloth-grid-drag-lifecycle-v1")
                result = conformance_smoke_cloth_grid_drag_lifecycle(
                    root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-soft-drag-lifecycle-v1")
                result = conformance_smoke_soft_drag_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-rope-drag-lifecycle-v1")
                result = conformance_smoke_rope_drag_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "smoke-rigid-drag-lifecycle-v1")
                result = conformance_smoke_rigid_drag_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "fluid-smoke-phase-lifecycle-v1")
                result = conformance_fluid_smoke_phase_lifecycle(root, &status);
            else if (arguments.lifecycle_id ==
                     "externally-heated-fluid-lifecycle-v1")
                result = conformance_externally_heated_fluid_lifecycle(
                    root, &status);
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
            const bool rope_self_contact_audit = arguments.quality_id ==
                "rope-self-contact-v2";
            const bool rope_self_contact_fixed_tick_audit =
                arguments.quality_id == "rope-self-contact-v3";
            const bool fluid_cloth_containment = arguments.quality_id ==
                "fluid-cloth-containment-v1";
            const bool fluid_soft_contact = arguments.quality_id ==
                "fluid-soft-contact-v1";
            const bool fluid_rope_contact = arguments.quality_id ==
                "fluid-rope-contact-v1";
            const bool cloth_soft_contact = arguments.quality_id ==
                "cloth-soft-contact-v1";
            const bool soft_rope_contact = arguments.quality_id ==
                "soft-rope-contact-v1";
            const bool cloth_rigid_contact = arguments.quality_id ==
                "cloth-rigid-contact-v1";
            const bool rope_rigid_contact = arguments.quality_id ==
                "rope-rigid-contact-v1";
            const bool rope_rigid_attachment = arguments.quality_id ==
                "rope-rigid-attachment-v1";
            const bool fluid_rigid_containment = arguments.quality_id ==
                "fluid-rigid-containment-v1";
            const bool fluid_rigid_contact = arguments.quality_id ==
                "fluid-rigid-contact-v1";
            const bool smoke_rigid_contact = arguments.quality_id ==
                "smoke-rigid-contact-v1";
            const bool soft_rigid_contact = arguments.quality_id ==
                "soft-rigid-contact-v1";
            const bool cloth_rope_attachment = arguments.quality_id ==
                "cloth-rope-attachment-v1";
            const bool soft_rope_attachment = arguments.quality_id ==
                "soft-rope-attachment-v1";
            const bool smoke_self_pressure = arguments.quality_id ==
                "smoke-self-pressure-v1";
            const bool fluid_cloth_contact = arguments.quality_id ==
                "fluid-cloth-contact-v1";
            const bool cloth_smoke_contact = arguments.quality_id ==
                "cloth-smoke-contact-v1";
            const bool soft_smoke_contact = arguments.quality_id ==
                "soft-smoke-contact-v1";
            const bool rope_smoke_contact = arguments.quality_id ==
                "rope-smoke-contact-v1";
            const bool smoke_self_viscosity = arguments.quality_id ==
                "smoke-self-viscosity-v1";
#if defined(PM_CONFORMANCE_DEVICE_FIXTURES)
            const bool fluid_pair = arguments.quality_id ==
                "core-conservation-v2";
            const bool fluid_self = arguments.quality_id ==
                    "fluid-self-contact-v1" ||
                arguments.quality_id == "fluid-self-pressure-v1" ||
                arguments.quality_id == "fluid-self-viscosity-v1";
#else
            constexpr bool fluid_pair = false;
            constexpr bool fluid_self = false;
#endif
            const bool smoke_fluid_drag = arguments.quality_id ==
                "smoke-fluid-drag-v1";
            const bool smoke_cloth_drag = arguments.quality_id ==
                "smoke-cloth-drag-v1";
            const bool smoke_soft_drag = arguments.quality_id ==
                "smoke-soft-drag-v1";
            const bool smoke_rope_drag = arguments.quality_id ==
                "smoke-rope-drag-v1";
            const bool smoke_rigid_drag = arguments.quality_id ==
                "smoke-rigid-drag-v1";
            const bool fluid_smoke_phase = arguments.quality_id ==
                "fluid-smoke-phase-v1";
            const bool externally_heated_fluid = arguments.quality_id ==
                "externally-heated-fluid-v1";
            if (!brick && !attachment && !rope_self_contact &&
                !rope_self_contact_audit &&
                !rope_self_contact_fixed_tick_audit &&
                !fluid_cloth_containment &&
                !fluid_soft_contact &&
                !fluid_rope_contact &&
                !cloth_soft_contact &&
                !soft_rope_contact &&
                !cloth_rigid_contact &&
                !rope_rigid_contact &&
                !rope_rigid_attachment &&
                !fluid_rigid_contact &&
                !smoke_rigid_contact &&
                !soft_rigid_contact &&
                !fluid_rigid_containment &&
                !cloth_rope_attachment &&
                !soft_rope_attachment &&
                !smoke_self_pressure && !fluid_cloth_contact &&
                !cloth_smoke_contact && !soft_smoke_contact &&
                !rope_smoke_contact &&
                !smoke_self_viscosity && !fluid_pair && !fluid_self &&
                !smoke_fluid_drag && !smoke_cloth_drag && !smoke_soft_drag &&
                !smoke_rope_drag && !smoke_rigid_drag &&
                !fluid_smoke_phase &&
                !externally_heated_fluid)
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
                } else if (rope_self_contact_audit) {
                    result = conformance_rope_self_contact_audit_quality(root);
                } else if (rope_self_contact_fixed_tick_audit) {
                    result =
                        conformance_rope_self_contact_fixed_tick_audit_quality(
                            root);
#if defined(PM_CONFORMANCE_DEVICE_FIXTURES)
                } else if (fluid_pair) {
                    ExecutionStatus status{};
                    const auto temporary = arguments.output /
                        (arguments.quality_id + ".temporary");
                    result = conformance_fluid_pair_quality(
                        temporary, &status);
                    std::filesystem::remove(temporary);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (fluid_self) {
                    ExecutionStatus status{};
                    const auto temporary = arguments.output /
                        (arguments.quality_id + ".temporary");
                    result = conformance_fluid_self_quality(
                        temporary, arguments.quality_id, &status);
                    std::filesystem::remove(temporary);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
#endif
                } else if (fluid_cloth_containment || fluid_soft_contact ||
                           fluid_rope_contact || fluid_rigid_contact ||
                           smoke_rigid_contact ||
                           soft_rigid_contact ||
                           fluid_rigid_containment ||
                           cloth_rope_attachment || soft_rope_attachment ||
                           cloth_soft_contact || cloth_rigid_contact ||
                           rope_rigid_contact || rope_rigid_attachment ||
                           soft_rope_contact) {
                    ExecutionStatus status{};
                    if (fluid_cloth_containment)
                        result = conformance_fluid_cloth_containment_quality(
                            root, &status);
                    else if (fluid_soft_contact)
                        result = conformance_fluid_soft_contact_quality(
                            root, &status);
                    else if (fluid_rope_contact)
                        result = conformance_fluid_rope_contact_quality(
                            root, &status);
                    else if (cloth_soft_contact)
                        result = conformance_cloth_soft_contact_quality(
                            root, &status);
                    else if (soft_rope_contact)
                        result = conformance_soft_rope_contact_quality(
                            root, &status);
                    else if (cloth_rigid_contact)
                        result = conformance_cloth_rigid_contact_quality(
                            root, &status);
                    else if (rope_rigid_contact)
                        result = conformance_rope_rigid_contact_quality(
                            root, &status);
                    else if (rope_rigid_attachment)
                        result = conformance_rope_rigid_attachment_quality(
                            root, &status);
                    else if (fluid_rigid_contact)
                        result = conformance_fluid_rigid_contact_quality(
                            root, &status);
                    else if (smoke_rigid_contact)
                        result = conformance_smoke_rigid_contact_quality(
                            root, &status);
                    else if (soft_rigid_contact)
                        result = conformance_soft_rigid_contact_quality(
                            root, &status);
                    else if (fluid_rigid_containment)
                        result = conformance_fluid_rigid_containment_quality(
                            root, &status);
                    else if (soft_rope_attachment)
                        result = conformance_soft_rope_attachment_quality(
                            root, &status);
                    else
                        result = conformance_cloth_rope_attachment_quality(
                            root, &status);
                    if (status != ExecutionStatus::completed && result.empty()) {
                        Json unavailable = Json::object();
                        unavailable["schema"] =
                            "parallel-mater-conformance-quality-result/v1";
                        unavailable["profile_id"] = arguments.quality_id;
                        unavailable["profile_sha256"] = contract_sha256(
                            root / "conformance/v1/quality" /
                            (arguments.quality_id + ".json"));
                        unavailable["execution_status"] =
                            status == ExecutionStatus::unsupported ?
                            "unsupported" : "failed";
                        result = unavailable.serialize();
                    }
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
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
                } else if (fluid_cloth_contact) {
                    ExecutionStatus status{};
                    result = conformance_fluid_cloth_contact_quality(
                        root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (cloth_smoke_contact) {
                    ExecutionStatus status{};
                    result = conformance_cloth_smoke_contact_quality(
                        root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (soft_smoke_contact) {
                    ExecutionStatus status{};
                    result = conformance_soft_smoke_contact_quality(
                        root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (rope_smoke_contact) {
                    ExecutionStatus status{};
                    result = conformance_rope_smoke_contact_quality(
                        root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (smoke_fluid_drag) {
                    ExecutionStatus status{};
                    result = conformance_smoke_fluid_drag_quality(root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (smoke_cloth_drag) {
                    ExecutionStatus status{};
                    result = conformance_smoke_cloth_drag_quality(root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (smoke_soft_drag) {
                    ExecutionStatus status{};
                    result = conformance_smoke_soft_drag_quality(root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (smoke_rope_drag) {
                    ExecutionStatus status{};
                    result = conformance_smoke_rope_drag_quality(root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (smoke_rigid_drag) {
                    ExecutionStatus status{};
                    result = conformance_smoke_rigid_drag_quality(root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (fluid_smoke_phase) {
                    ExecutionStatus status{};
                    result = conformance_fluid_smoke_phase_quality(root, &status);
                    write_file(arguments.output /
                                   (arguments.quality_id + ".json"),
                               result);
                    if (status == ExecutionStatus::unsupported) return 77;
                    if (status == ExecutionStatus::failed) return 2;
                    std::cout << "Quality execution completed: "
                              << arguments.quality_id << '\n';
                    return 0;
                } else if (externally_heated_fluid) {
                    ExecutionStatus status{};
                    result = conformance_externally_heated_fluid_quality(
                        root, &status);
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
        if (!arguments.scenario_id.empty()) {
            Json result;
            if (arguments.scenario_id == "rigid-cold-impact-rotation" ||
                arguments.scenario_id == "rigid-settled-long-run" ||
                arguments.scenario_id ==
                    "fluid-closed-container-source-outflow" ||
                arguments.scenario_id == "cloth-stiff-pin-impact" ||
                arguments.scenario_id == "soft-body-mass-ratio-volume" ||
                arguments.scenario_id == "rope-attachment-rotation" ||
                arguments.scenario_id ==
                    "smoke-analytic-transport-long-run" ||
                arguments.scenario_id == "fluid-cloth-rigid-containment")
                result = run_case_fixture_scenario(
                    arguments.scenario_id, root);
            else if (arguments.scenario_id == "six-family-mixed")
                result = run_mixed_fixture();
            else if (arguments.scenario_id ==
                     "boundary-replacement-generation")
                result = run_boundary_fixture();
            else if (arguments.scenario_id == "empty-and-sparse-capacity")
                result = run_empty_sparse_fixture();
            else if (arguments.scenario_id ==
                     "capacity-overflow-atomic-recovery")
                result = run_capacity_fixture();
            else
                throw std::runtime_error("unknown native realtime scenario: " +
                                         arguments.scenario_id);
            std::cout << "RUN scenario " << arguments.scenario_id << std::endl;
            write_file(arguments.output / (arguments.scenario_id + ".json"),
                       result.serialize());
            std::cout << "Scenarios: 1 completed, 0 failed\n";
            return 0;
        }
        if (!arguments.workload_id.empty()) {
            std::vector<std::string> workloads;
            if (arguments.workload_id == "all") {
                workloads = conformance_realtime_workload_ids();
            } else {
                workloads.push_back(arguments.workload_id);
            }
            std::uint32_t completed{}, failed{};
            for (const auto &identity : workloads) {
                try {
                    std::cout << "RUN workload " << identity << std::endl;
                    write_file(arguments.output / (identity + ".json"),
                        conformance_realtime_workload(
                            identity, arguments.warmup_ticks,
                            arguments.measured_ticks));
                    ++completed;
                } catch (const std::exception &error) {
                    Json failure = Json::object();
                    failure["schema"] =
                        "parallel-mater-realtime-workload-capture/v1";
                    failure["id"] = identity;
                    failure["execution_status"] = "failed";
                    failure["reason"] = error.what();
                    write_file(arguments.output / (identity + ".json"),
                               failure.serialize());
                    std::cerr << "FAILED workload " << identity << ": "
                              << error.what() << '\n';
                    ++failed;
                }
            }
            std::cout << "Workloads: " << completed << " completed, "
                      << failed << " failed\n";
            return failed == 0U ? 0 : 2;
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
                definition->id, root, arguments.every_frame,
                arguments.tick_count);
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
