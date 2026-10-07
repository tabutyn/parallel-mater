// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/case_registry.hpp>
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include <parallel_mater_gallery/scene.hpp>

#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
#import <Metal/Metal.h>
#else
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#ifndef PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT
#error PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT is required
#endif
#ifndef PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT
#define PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT "unknown"
#endif

namespace {

#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
using namespace parallel_mater::metal;
using namespace parallel_mater::metal::gallery;
#else
using namespace parallel_mater;
using namespace parallel_mater::gallery;
#endif
using namespace parallel_mater::conformance;

constexpr std::size_t complete_state_limit = 256U;
constexpr std::size_t large_sample_limit = 32U;

void require(Status status, std::string_view operation) {
    if (status) return;
    throw std::runtime_error(std::string(operation) + ": " +
        (status.message != nullptr ? status.message : "unknown error"));
}

#if !defined(PARALLEL_MATER_CONFORMANCE_METAL)
void require_cuda(cudaError_t status, std::string_view operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
#endif

template <typename T>
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
std::vector<T> download(BufferSpan<const T> span) {
    std::vector<T> result(static_cast<std::size_t>(span.size));
    if (!result.empty()) {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)span.buffer;
        if (buffer == nil || buffer.contents == nullptr)
            throw std::runtime_error("Metal conformance buffer is not host visible");
        const auto *source = reinterpret_cast<const T *>(
            static_cast<const std::byte *>(buffer.contents) + span.byte_offset);
        std::copy_n(source, result.size(), result.data());
    }
    return result;
}
#else
std::vector<T> download(DeviceSpan<const T> span) {
    std::vector<T> result(static_cast<std::size_t>(span.size));
    if (!result.empty())
        require_cuda(cudaMemcpy(result.data(), span.data,
                                result.size() * sizeof(T),
                                cudaMemcpyDeviceToHost),
                     "download device state");
    return result;
}
#endif

Json vector_json(Vec3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

Json quaternion_json(Quaternion value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    result.push_back(static_cast<double>(value.w));
    return result;
}

Json unsigned_array_json(const std::vector<std::uint32_t> &values,
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

Aggregate aggregate(const std::vector<Vec3> &positions,
                    const std::vector<Vec3> &velocities,
                    double element_mass = 1.0) {
    Aggregate result{};
    result.mass = element_mass * static_cast<double>(positions.size());
    if (positions.empty()) {
        result.minimum = {};
        result.maximum = {};
        return result;
    }
    for (std::size_t index = 0U; index < positions.size(); ++index) {
        const Vec3 position = positions[index];
        const Vec3 velocity = index < velocities.size() ? velocities[index] : Vec3{};
        result.finite = result.finite &&
            std::isfinite(position.x) && std::isfinite(position.y) &&
            std::isfinite(position.z) && std::isfinite(velocity.x) &&
            std::isfinite(velocity.y) && std::isfinite(velocity.z);
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
        result.center.x += position.x;
        result.center.y += position.y;
        result.center.z += position.z;
        result.momentum.x += static_cast<float>(element_mass * velocity.x);
        result.momentum.y += static_cast<float>(element_mass * velocity.y);
        result.momentum.z += static_cast<float>(element_mass * velocity.z);
        const double speed_squared =
            static_cast<double>(velocity.x) * velocity.x +
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

Json aggregate_json(const Aggregate &value) {
    Json result = Json::object();
    Json bounds = Json::object();
    bounds["maximum"] = vector_json(value.maximum);
    bounds["minimum"] = vector_json(value.minimum);
    result["bounds"] = std::move(bounds);
    result["center_of_mass"] = vector_json(value.center);
    result["finite"] = value.finite;
    result["kinetic_energy"] = value.kinetic_energy;
    result["mass"] = value.mass;
    result["maximum_speed"] = value.maximum_speed;
    result["momentum"] = vector_json(value.momentum);
    return result;
}

Json samples_json(const std::vector<Vec3> &positions,
                  const std::vector<Vec3> &velocities,
                  bool complete,
                  const std::vector<std::uint32_t> *stable_ids = nullptr) {
    Json result = Json::array();
    const std::size_t limit = complete
        ? positions.size() : std::min(positions.size(), large_sample_limit);
    std::vector<std::size_t> order(positions.size());
    for (std::size_t index = 0U; index < order.size(); ++index) order[index] = index;
    if (stable_ids != nullptr)
        std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
            return (*stable_ids)[left] < (*stable_ids)[right];
        });
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

TriangleMeshId upload_mesh(World &world, const std::vector<Vec3> &vertices,
                           const std::vector<std::uint32_t> &indices) {
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
    TriangleMeshId mesh{};
    require(world.add_triangle_mesh({vertices.data(), vertices.size()},
                                    {indices.data(), indices.size()}, mesh),
            "add conformance triangle mesh");
    return mesh;
#else
    Vec3 *device_vertices = nullptr;
    std::uint32_t *device_indices = nullptr;
    require_cuda(cudaMalloc(reinterpret_cast<void **>(&device_vertices),
                            vertices.size() * sizeof(Vec3)),
                 "allocate conformance mesh vertices");
    require_cuda(cudaMalloc(reinterpret_cast<void **>(&device_indices),
                            indices.size() * sizeof(std::uint32_t)),
                 "allocate conformance mesh indices");
    require_cuda(cudaMemcpy(device_vertices, vertices.data(),
                            vertices.size() * sizeof(Vec3),
                            cudaMemcpyHostToDevice),
                 "upload conformance mesh vertices");
    require_cuda(cudaMemcpy(device_indices, indices.data(),
                            indices.size() * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice),
                 "upload conformance mesh indices");
    TriangleMeshId mesh{};
    const Status status = world.add_triangle_mesh(
        {device_vertices, vertices.size()}, {device_indices, indices.size()}, mesh);
    cudaFree(device_indices);
    cudaFree(device_vertices);
    require(status, "add conformance triangle mesh");
    return mesh;
#endif
}

TriangleMeshId add_box_mesh(World &world, Vec3 half) {
    const std::vector<Vec3> vertices{
        {-half.x,-half.y,-half.z},{half.x,-half.y,-half.z},
        {half.x,half.y,-half.z},{-half.x,half.y,-half.z},
        {-half.x,-half.y,half.z},{half.x,-half.y,half.z},
        {half.x,half.y,half.z},{-half.x,half.y,half.z}};
    const std::vector<std::uint32_t> indices{
        0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,
        1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0};
    return upload_mesh(world, vertices, indices);
}

struct Runtime {
    World world{};
    SceneDefinition scene{};
    SceneInstance instance{};
    std::vector<std::string> body_names{};
    std::vector<RigidBodyId> bodies{};
    std::vector<RigidBodyOptions> body_options{};
    std::vector<std::string> constraint_names{};
    std::vector<RigidConstraintId> constraints{};
    std::vector<RigidConstraintOptions> constraint_options{};
    std::vector<std::array<std::size_t, 2U>> constraint_bodies{};
    std::vector<bool> constraint_present{};
    std::vector<TriangleMeshId> meshes{};
    bool integrated{};
    bool generation_changed{};
    bool constraints_rebuilt{};
};

std::string indexed_name(std::string_view kind, std::string_view source,
                         std::size_t index) {
    return std::string(kind) + "/" +
        (source.empty() ? "unnamed" : std::string(source)) + "/" +
        std::to_string(index);
}

Runtime make_integrated(const CaseDefinition &definition,
                        const std::filesystem::path &source_root) {
    Runtime runtime{};
    runtime.integrated = true;
    std::string error;
    if (!load_glb_scene(source_root / definition.glb_path, runtime.scene, error))
        throw std::runtime_error("load " + definition.glb_path + ": " + error);
    require(create_scene_world(runtime.scene, runtime.world, runtime.instance),
            "create integrated conformance world");
    runtime.bodies = runtime.instance.rigid_bodies;
    for (std::size_t index = 0U; index < runtime.scene.rigid_bodies.size(); ++index) {
        const auto &body = runtime.scene.rigid_bodies[index];
        runtime.body_names.push_back(indexed_name(
            "rigid", body.source_name.empty() ? body.name : body.source_name, index));
        runtime.body_options.push_back(body.options);
    }
    runtime.constraints = runtime.instance.rigid_constraints;
    for (std::size_t index = 0U; index < runtime.scene.rigid_constraints.size(); ++index) {
        const auto &constraint = runtime.scene.rigid_constraints[index];
        runtime.constraint_names.push_back(indexed_name(
            "constraint", constraint.name, index));
        runtime.constraint_options.push_back(constraint.options);
        runtime.constraint_bodies.push_back({constraint.body_a, constraint.body_b});
        runtime.constraint_present.push_back(true);
    }
    return runtime;
}

Runtime make_direct() {
    Runtime runtime{};
    require(World::create({.rigid_body_capacity = 4U,
                           .triangle_mesh_capacity = 2U,
                           .contact_capacity = 512U}, runtime.world),
            "create direct conformance world");
    const TriangleMeshId body_mesh = add_box_mesh(runtime.world, {0.25F,0.25F,0.25F});
    const TriangleMeshId ground_mesh = add_box_mesh(runtime.world, {3.0F,0.05F,3.0F});
    runtime.meshes = {body_mesh, ground_mesh};
    runtime.body_names = {"dynamic-box", "ground", "kinematic-box"};
    runtime.body_options = {
        {.mesh=body_mesh,.initial_state={.position={0,2,0}},.mass=2.0F,
         .inertia_diagonal={1.0F/12.0F,1.0F/12.0F,1.0F/12.0F},
         .linear_damping=0,.angular_damping=0},
        {.motion=MotionType::static_body,.mesh=ground_mesh,
         .initial_state={.position={0,-0.05F,0}}},
        {.motion=MotionType::kinematic,.mesh=body_mesh,
         .initial_state={.position={-1,1,0}}}};
    for (const auto &options : runtime.body_options) {
        RigidBodyId body{};
        require(runtime.world.add_rigid_body(options, body), "add direct body");
        runtime.bodies.push_back(body);
    }
    return runtime;
}

Runtime make_weld(bool breaking) {
    Runtime runtime{};
    require(World::create({.rigid_body_capacity = 4U,
                           .rigid_constraint_capacity = 4U,
                           .triangle_mesh_capacity = 1U,
                           .contact_capacity = 512U}, runtime.world),
            "create weld conformance world");
    const TriangleMeshId mesh = add_box_mesh(runtime.world, {0.2F,0.2F,0.2F});
    runtime.meshes = {mesh};
    const std::size_t body_count = breaking ? 2U : 3U;
    for (std::size_t index = 0U; index < body_count; ++index) {
        runtime.body_names.push_back(breaking
            ? (index == 0U ? "break-a" : "break-b")
            : "member-" + std::to_string(index));
        RigidBodyOptions options{.mesh=mesh,
            .initial_state={.position={-0.4F + 0.4F * static_cast<float>(index),
                                      1.0F,0.0F}},
            .mass=1.0F,.linear_damping=0,.angular_damping=0};
        runtime.body_options.push_back(options);
        RigidBodyId body{};
        require(runtime.world.add_rigid_body(options, body), "add weld body");
        runtime.bodies.push_back(body);
    }
    const std::size_t constraint_count = breaking ? 1U : 2U;
    for (std::size_t index = 0U; index < constraint_count; ++index) {
        RigidConstraintOptions options{
            .type=RigidConstraintType::fixed,
            .body_a=runtime.bodies[index],.body_b=runtime.bodies[index+1U],
            .local_anchor_a={0.2F,0,0},.local_anchor_b={-0.2F,0,0},
            .breaking_impulse_threshold=breaking?0.05F:0.0F,
            .solver_iterations=8U};
        RigidConstraintId constraint{};
        require(runtime.world.add_rigid_constraint(options, constraint),
                "add weld constraint");
        runtime.constraint_names.push_back(
            breaking ? "break-link" : "weld-link-" + std::to_string(index));
        runtime.constraints.push_back(constraint);
        runtime.constraint_options.push_back(options);
        runtime.constraint_bodies.push_back({index, index + 1U});
        runtime.constraint_present.push_back(true);
    }
    return runtime;
}

Runtime make_runtime(const CaseDefinition &definition,
                     const std::filesystem::path &source_root) {
    if (definition.kind == CaseKind::integrated_glb)
        return make_integrated(definition, source_root);
    if (definition.id == "rigid-direct") return make_direct();
    if (definition.id == "compound-weld-lifecycle") return make_weld(false);
    if (definition.id == "constraint-breaking") return make_weld(true);
    throw std::runtime_error("no analytic runner for " + definition.id);
}

std::size_t body_index(const Runtime &runtime, std::string_view name) {
    const auto found = std::find(runtime.body_names.begin(),
                                 runtime.body_names.end(), name);
    if (found == runtime.body_names.end())
        throw std::runtime_error("unknown command body " + std::string(name));
    return static_cast<std::size_t>(found - runtime.body_names.begin());
}

void update_constraint(Runtime &runtime, std::size_t index, bool enabled) {
    auto options = runtime.constraint_options[index];
    options.body_a = runtime.bodies[runtime.constraint_bodies[index][0]];
    options.body_b = runtime.bodies[runtime.constraint_bodies[index][1]];
    options.enabled = enabled;
    require(runtime.world.update_rigid_constraint(runtime.constraints[index], options),
            "update conformance constraint");
    runtime.constraint_options[index] = options;
}

void apply_command(Runtime &runtime, const Command &command, Vec3 &gravity) {
    if (command.operation == "set_gravity") {
        gravity = {static_cast<float>(command.value[0]),
                   static_cast<float>(command.value[1]),
                   static_cast<float>(command.value[2])};
        return;
    }
    if (command.operation == "apply_impulse") {
        const std::size_t index = body_index(runtime, command.target);
        RigidBodyState state{};
        require(runtime.world.read_rigid_body_state(runtime.bodies[index], state),
                "read impulse target");
        require(runtime.world.apply_impulse(
            runtime.bodies[index],
            {static_cast<float>(command.value[0]),
             static_cast<float>(command.value[1]),
             static_cast<float>(command.value[2])}, state.position),
            "apply conformance impulse");
        return;
    }
    if (command.operation == "set_kinematic_target") {
        const std::size_t index = body_index(runtime, command.target);
        RigidBodyState target = runtime.body_options[index].initial_state;
        target.position = {static_cast<float>(command.value[0]),
                           static_cast<float>(command.value[1]),
                           static_cast<float>(command.value[2])};
        require(runtime.world.set_kinematic_target(runtime.bodies[index], target),
                "set conformance kinematic target");
        return;
    }
    if (command.operation == "set_first_kinematic_target") {
        for (std::size_t index = 0U; index < runtime.body_options.size(); ++index) {
            if (runtime.body_options[index].motion != MotionType::kinematic) continue;
            RigidBodyState target = runtime.body_options[index].initial_state;
            target.position = {static_cast<float>(command.value[0]),
                               static_cast<float>(command.value[1]),
                               static_cast<float>(command.value[2])};
            require(runtime.world.set_kinematic_target(runtime.bodies[index], target),
                    "set first authored kinematic target");
            return;
        }
        throw std::runtime_error("integrated case has no kinematic body");
    }
    if (command.operation == "replace_body") {
        const std::size_t index = body_index(runtime, command.target);
        const RigidBodyId previous = runtime.bodies[index];
        require(runtime.world.remove_rigid_body(previous), "remove lifecycle body");
        auto options = runtime.body_options[index];
        options.initial_state.position = {static_cast<float>(command.value[0]),
                                          static_cast<float>(command.value[1]),
                                          static_cast<float>(command.value[2])};
        require(runtime.world.add_rigid_body(options, runtime.bodies[index]),
                "replace lifecycle body");
        runtime.generation_changed =
            previous.index == runtime.bodies[index].index &&
            previous.generation != runtime.bodies[index].generation;
        runtime.body_options[index] = options;
        return;
    }
    if (command.operation == "remove_constraints") {
        for (std::size_t index = 0U; index < runtime.constraints.size(); ++index) {
            if (!runtime.constraint_present[index]) continue;
            require(runtime.world.remove_rigid_constraint(runtime.constraints[index]),
                    "remove lifecycle constraint");
            runtime.constraint_present[index] = false;
        }
        return;
    }
    if (command.operation == "add_constraints") {
        for (std::size_t index = 0U; index < runtime.constraints.size(); ++index) {
            if (runtime.constraint_present[index]) continue;
            auto options = runtime.constraint_options[index];
            options.body_a = runtime.bodies[runtime.constraint_bodies[index][0]];
            options.body_b = runtime.bodies[runtime.constraint_bodies[index][1]];
            require(runtime.world.add_rigid_constraint(options,
                                                        runtime.constraints[index]),
                    "rebuild lifecycle constraint");
            runtime.constraint_present[index] = true;
            runtime.constraints_rebuilt = true;
        }
        return;
    }
    if (command.operation == "set_constraints_enabled") {
        const bool enabled = command.value[0] != 0.0;
        for (std::size_t index = 0U; index < runtime.constraints.size(); ++index)
            if (runtime.constraint_present[index])
                update_constraint(runtime, index, enabled);
        return;
    }
    throw std::runtime_error("unsupported command " + command.operation);
}

Json rigid_resource(Runtime &runtime, std::size_t index,
                    std::string_view tolerance) {
    RigidBodyState state{};
    require(runtime.world.read_rigid_body_state(runtime.bodies[index], state),
            "read rigid conformance state");
    Json result = Json::object();
    result["angular_velocity"] = vector_json(state.angular_velocity);
    result["id"] = runtime.body_names[index];
    result["linear_velocity"] = vector_json(state.linear_velocity);
    result["orientation"] = quaternion_json(state.orientation);
    result["position"] = vector_json(state.position);
    result["present"] = true;
    result["tolerance_class"] = std::string(tolerance);
    result["type"] = "rigid_body";
    return result;
}

Json constraint_resource(Runtime &runtime, std::size_t index) {
    Json result = Json::object();
    result["id"] = runtime.constraint_names[index];
    result["present"] = static_cast<bool>(runtime.constraint_present[index]);
    result["tolerance_class"] = "constraint_contact";
    result["type"] = "rigid_constraint";
    if (runtime.constraint_present[index]) {
        RigidConstraintState state{};
        require(runtime.world.read_rigid_constraint_state(
                    runtime.constraints[index], state),
                "read conformance constraint state");
        result["applied_impulse"] = static_cast<double>(state.applied_impulse);
        result["broken"] = state.broken;
        result["enabled"] = state.enabled;
    } else {
        result["applied_impulse"] = 0.0;
        result["broken"] = false;
        result["enabled"] = false;
    }
    return result;
}

Json particle_resource(const std::string &id, std::string_view type,
                       const std::vector<Vec3> &positions,
                       const std::vector<Vec3> &velocities,
                       bool complete,
                       const std::vector<std::uint32_t> *stable_ids = nullptr,
                       double element_mass = 1.0) {
    Json result = Json::object();
    const Aggregate summary = aggregate(positions, velocities, element_mass);
    result["aggregate"] = aggregate_json(summary);
    result["complete"] = complete;
    result["count"] = static_cast<std::uint64_t>(positions.size());
    result["id"] = id;
    result["minimum_clearance"] = static_cast<double>(summary.minimum.y);
    result["samples"] = samples_json(positions, velocities, complete, stable_ids);
    if (stable_ids != nullptr) {
        std::vector<std::uint32_t> ordered = *stable_ids;
        std::sort(ordered.begin(), ordered.end());
        result["stable_ids"] = unsigned_array_json(
            ordered, complete ? ordered.size() : large_sample_limit);
    }
    result["tolerance_class"] = "deformable";
    result["type"] = std::string(type);
    return result;
}

Json fluid_resource(Runtime &runtime, bool complete) {
    FluidDeviceView view{};
    require(runtime.world.fluid_view(runtime.instance.fluid, view),
            "read conformance fluid view");
    auto positions = download(view.positions);
    auto velocities = download(view.velocities);
    auto ids = download(view.stable_particle_ids);
    const auto &options = runtime.scene.fluid_options;
    const double volume = options.rest_particle_volume > 0.0F
        ? options.rest_particle_volume
        : std::pow(static_cast<double>(options.particle_radius) * 2.0, 3.0);
    Json result = particle_resource("fluid/0", "fluid", positions, velocities,
                                    complete, &ids, options.rest_density * volume);
    auto foam = download(view.foam);
    double foam_sum = 0.0;
    double foam_maximum = 0.0;
    for (const float value : foam) {
        foam_sum += value;
        foam_maximum = std::max(foam_maximum, static_cast<double>(value));
    }
    Json foam_summary = Json::object();
    foam_summary["maximum"] = foam_maximum;
    foam_summary["mean"] = foam.empty() ? 0.0 : foam_sum / foam.size();
    result["foam"] = std::move(foam_summary);
    return result;
}

Json smoke_resource(Runtime &runtime, bool complete) {
    SmokeDeviceView view{};
    require(runtime.world.smoke_view(runtime.instance.smoke, view),
            "read conformance smoke view");
    auto positions = download(view.positions);
    auto velocities = download(view.velocities);
    if (positions.size() > view.particle_count) positions.resize(view.particle_count);
    if (velocities.size() > view.particle_count) velocities.resize(view.particle_count);
    Json result = particle_resource("smoke/0", "smoke", positions, velocities,
                                    complete);
    Json grid = Json::object();
    grid["cell_count"] = static_cast<std::uint64_t>(view.grid_divergence.size);
    grid["pressure_relative_residual"] =
        static_cast<double>(view.grid_pressure_relative_residual);
    grid["resolution"] = static_cast<std::uint64_t>(view.grid_resolution);
    grid["vertical_resolution"] =
        static_cast<std::uint64_t>(view.grid_vertical_resolution);
    const auto divergence = download(view.grid_divergence);
    double maximum_divergence = 0.0;
    double squared_divergence = 0.0;
    for (const float value : divergence) {
        maximum_divergence = std::max(maximum_divergence,
                                      std::abs(static_cast<double>(value)));
        squared_divergence += static_cast<double>(value) * value;
    }
    grid["maximum_divergence"] = maximum_divergence;
    grid["rms_divergence"] = divergence.empty() ? 0.0 :
        std::sqrt(squared_divergence / divergence.size());
    result["grid"] = std::move(grid);
    return result;
}

Json cloth_resource(Runtime &runtime, std::size_t index, bool complete) {
    ClothDeviceView view{};
    require(runtime.world.cloth_view(runtime.instance.cloths[index], view),
            "read conformance cloth view");
    auto positions = download(view.positions);
    auto velocities = download(view.velocities);
    Json result = particle_resource(indexed_name("cloth",
        runtime.scene.cloths[index].name, index), "cloth", positions, velocities,
        complete, nullptr, runtime.scene.cloths[index].vertex_mass);
    const auto indices = download(view.triangle_indices);
    const auto sources = download(view.vertex_source_indices);
    const auto active = download(view.active_bonds);
    const auto bonds = download(view.bonds);
    double maximum_strain = 0.0;
    for (std::size_t bond = 0U; bond < bonds.size(); ++bond) {
        if (bond >= active.size() || active[bond] == 0U ||
            bonds[bond].first >= positions.size() ||
            bonds[bond].second >= positions.size() ||
            bonds[bond].rest_length <= 0.0F) continue;
        const Vec3 a = positions[bonds[bond].first];
        const Vec3 b = positions[bonds[bond].second];
        const double dx = static_cast<double>(a.x) - b.x;
        const double dy = static_cast<double>(a.y) - b.y;
        const double dz = static_cast<double>(a.z) - b.z;
        maximum_strain = std::max(maximum_strain,
            std::abs(std::sqrt(dx*dx+dy*dy+dz*dz) / bonds[bond].rest_length - 1.0));
    }
    result["maximum_strain"] = maximum_strain;
    Json topology = Json::object();
    topology["active_bond_count"] = static_cast<std::uint64_t>(
        std::count(active.begin(), active.end(), std::uint8_t{1U}));
    topology["bond_count"] = static_cast<std::uint64_t>(active.size());
    topology["triangle_count"] = static_cast<std::uint64_t>(indices.size() / 3U);
    topology["vertex_count"] = static_cast<std::uint64_t>(view.vertex_count);
    if (complete) {
        Json bond_json = Json::array();
        for (const ClothBond bond : bonds) {
            Json item = Json::object();
            item["first"] = static_cast<std::uint64_t>(bond.first);
            item["rest_length"] = static_cast<double>(bond.rest_length);
            item["second"] = static_cast<std::uint64_t>(bond.second);
            bond_json.push_back(std::move(item));
        }
        topology["active_bonds"] = unsigned_array_json(
            std::vector<std::uint32_t>(active.begin(), active.end()));
        topology["bonds"] = std::move(bond_json);
        topology["indices"] = unsigned_array_json(indices);
        topology["vertex_source_indices"] = unsigned_array_json(sources);
    }
    result["topology"] = std::move(topology);
    return result;
}

Json soft_body_resource(Runtime &runtime, std::size_t index, bool complete) {
    SoftBodyDeviceView view{};
    require(runtime.world.soft_body_view(runtime.instance.soft_bodies[index], view),
            "read conformance soft-body view");
    auto positions = download(view.positions);
    auto velocities = download(view.velocities);
    Json result = particle_resource(indexed_name("soft_body",
        runtime.scene.soft_bodies[index].name, index), "soft_body",
        positions, velocities, complete, nullptr,
        runtime.scene.soft_bodies[index].node_mass);
    const auto bonds = download(view.bonds);
    const auto surface_indices = download(view.surface_triangle_indices);
    Json topology = Json::object();
    topology["bond_count"] = static_cast<std::uint64_t>(bonds.size());
    topology["node_count"] = static_cast<std::uint64_t>(view.node_count);
    topology["surface_triangle_count"] =
        static_cast<std::uint64_t>(surface_indices.size() / 3U);
    topology["surface_vertex_count"] =
        static_cast<std::uint64_t>(view.surface_vertex_count);
    double maximum_strain = 0.0;
    for (const SoftBodyBond bond : bonds) {
        if (bond.first >= positions.size() || bond.second >= positions.size() ||
            bond.rest_length <= 0.0F) continue;
        const Vec3 a = positions[bond.first], b = positions[bond.second];
        const double dx = static_cast<double>(a.x) - b.x;
        const double dy = static_cast<double>(a.y) - b.y;
        const double dz = static_cast<double>(a.z) - b.z;
        maximum_strain = std::max(maximum_strain,
            std::abs(std::sqrt(dx*dx+dy*dy+dz*dz) / bond.rest_length - 1.0));
    }
    result["maximum_strain"] = maximum_strain;
    if (complete) {
        Json bond_json = Json::array();
        for (const SoftBodyBond bond : bonds) {
            Json item = Json::object();
            item["first"] = static_cast<std::uint64_t>(bond.first);
            item["rest_length"] = static_cast<double>(bond.rest_length);
            item["second"] = static_cast<std::uint64_t>(bond.second);
            bond_json.push_back(std::move(item));
        }
        topology["bonds"] = std::move(bond_json);
        topology["indices"] = unsigned_array_json(surface_indices);
    }
    result["topology"] = std::move(topology);
    return result;
}

Json rope_resource(Runtime &runtime, std::size_t index, bool complete) {
    RopeDeviceView view{};
    require(runtime.world.rope_view(runtime.instance.ropes[index], view),
            "read conformance rope view");
    auto positions = download(view.positions);
    auto velocities = download(view.velocities);
    const double node_mass = positions.empty() ? 0.0 :
        runtime.scene.ropes[index].options.mass / positions.size();
    Json result = particle_resource(indexed_name("rope",
        runtime.scene.ropes[index].name, index), "rope", positions, velocities,
        complete, nullptr, node_mass);
    const auto rest = download(view.rest_lengths);
    double maximum_strain = 0.0;
    for (std::size_t edge = 0U; edge < rest.size() && edge + 1U < positions.size();
         ++edge) {
        const Vec3 difference{positions[edge + 1U].x - positions[edge].x,
                              positions[edge + 1U].y - positions[edge].y,
                              positions[edge + 1U].z - positions[edge].z};
        const double length = std::sqrt(
            static_cast<double>(difference.x) * difference.x +
            static_cast<double>(difference.y) * difference.y +
            static_cast<double>(difference.z) * difference.z);
        if (rest[edge] > 0.0F)
            maximum_strain = std::max(maximum_strain,
                                      std::abs(length / rest[edge] - 1.0));
    }
    Json topology = Json::object();
    topology["edge_count"] = static_cast<std::uint64_t>(rest.size());
    topology["node_count"] = static_cast<std::uint64_t>(positions.size());
    if (complete) {
        Json rest_json = Json::array();
        for (const float value : rest)
            rest_json.push_back(static_cast<double>(value));
        topology["rest_lengths"] = std::move(rest_json);
    }
    result["maximum_strain"] = maximum_strain;
    result["topology"] = std::move(topology);
    return result;
}

std::string rigid_name(const Runtime &runtime, RigidBodyId id) {
    for (std::size_t index = 0U; index < runtime.bodies.size(); ++index)
        if (runtime.bodies[index] == id) return runtime.body_names[index];
    return "unknown";
}

Json contacts_json(Runtime &runtime) {
    Json result = Json::object();
    const RigidContactDeviceView rigid_view = runtime.world.rigid_contacts();
    auto rigid = download(rigid_view.events);
    if (rigid.size() > rigid_view.event_count) rigid.resize(rigid_view.event_count);
    std::sort(rigid.begin(), rigid.end(), [&](const auto &left, const auto &right) {
        return std::tuple{rigid_name(runtime, left.body),
                          rigid_name(runtime, left.collider), left.position.x,
                          left.position.y, left.position.z} <
               std::tuple{rigid_name(runtime, right.body),
                          rigid_name(runtime, right.collider), right.position.x,
                          right.position.y, right.position.z};
    });
    Json rigid_json = Json::array();
    for (std::size_t index = 0U;
         index < std::min(rigid.size(), complete_state_limit); ++index) {
        Json contact = Json::object();
        contact["body"] = rigid_name(runtime, rigid[index].body);
        contact["collider"] = rigid_name(runtime, rigid[index].collider);
        contact["friction_impulse"] = vector_json(rigid[index].friction_impulse);
        contact["normal"] = vector_json(rigid[index].normal);
        contact["normal_impulse"] = static_cast<double>(rigid[index].normal_impulse);
        contact["penetration"] = static_cast<double>(rigid[index].penetration);
        contact["position"] = vector_json(rigid[index].position);
        rigid_json.push_back(std::move(contact));
    }
    result["rigid_contact_count"] = static_cast<std::uint64_t>(rigid.size());
    result["rigid_contacts"] = std::move(rigid_json);
    const ContactDeviceView fluid_view = runtime.world.contacts();
    auto fluid = download(fluid_view.events);
    if (fluid.size() > fluid_view.event_count) fluid.resize(fluid_view.event_count);
    std::sort(fluid.begin(), fluid.end(), [](const auto &left, const auto &right) {
        return left.stable_particle_id < right.stable_particle_id;
    });
    Json fluid_json = Json::array();
    for (std::size_t index = 0U;
         index < std::min(fluid.size(), complete_state_limit); ++index) {
        Json contact = Json::object();
        contact["normal"] = vector_json(fluid[index].normal);
        contact["normal_impulse"] = static_cast<double>(fluid[index].normal_impulse);
        contact["position"] = vector_json(fluid[index].position);
        contact["rigid_body"] = rigid_name(runtime, fluid[index].rigid_body);
        contact["stable_particle_id"] =
            static_cast<std::uint64_t>(fluid[index].stable_particle_id);
        fluid_json.push_back(std::move(contact));
    }
    result["fluid_contact_count"] = static_cast<std::uint64_t>(fluid.size());
    result["fluid_contacts"] = std::move(fluid_json);
    return result;
}

Vec3 combined_momentum(Runtime &runtime) {
    Vec3 result{};
    const auto add_velocities = [&](const std::vector<Vec3> &velocities,
                                    double mass) {
        for (const Vec3 velocity : velocities) {
            result.x += static_cast<float>(mass * velocity.x);
            result.y += static_cast<float>(mass * velocity.y);
            result.z += static_cast<float>(mass * velocity.z);
        }
    };
    for (std::size_t index = 0U; index < runtime.bodies.size(); ++index) {
        if (runtime.body_options[index].motion != MotionType::dynamic) continue;
        RigidBodyState state{};
        require(runtime.world.read_rigid_body_state(runtime.bodies[index], state),
                "read momentum rigid state");
        add_velocities({state.linear_velocity}, runtime.body_options[index].mass);
    }
    if (runtime.instance.has_fluid) {
        FluidDeviceView view{};
        require(runtime.world.fluid_view(runtime.instance.fluid, view),
                "read momentum fluid view");
        const auto &options = runtime.scene.fluid_options;
        const double volume = options.rest_particle_volume > 0.0F
            ? options.rest_particle_volume
            : std::pow(static_cast<double>(options.particle_radius) * 2.0, 3.0);
        add_velocities(download(view.velocities), options.rest_density * volume);
    }
    for (std::size_t index = 0U; index < runtime.instance.cloths.size(); ++index) {
        ClothDeviceView view{};
        require(runtime.world.cloth_view(runtime.instance.cloths[index], view),
                "read momentum cloth view");
        add_velocities(download(view.velocities),
                       runtime.scene.cloths[index].vertex_mass);
    }
    for (std::size_t index = 0U; index < runtime.instance.soft_bodies.size(); ++index) {
        SoftBodyDeviceView view{};
        require(runtime.world.soft_body_view(runtime.instance.soft_bodies[index], view),
                "read momentum soft-body view");
        add_velocities(download(view.velocities),
                       runtime.scene.soft_bodies[index].node_mass);
    }
    for (std::size_t index = 0U; index < runtime.instance.ropes.size(); ++index) {
        RopeDeviceView view{};
        require(runtime.world.rope_view(runtime.instance.ropes[index], view),
                "read momentum rope view");
        const double node_mass = view.positions.size == 0U ? 0.0 :
            runtime.scene.ropes[index].options.mass / view.positions.size;
        add_velocities(download(view.velocities), node_mass);
    }
    return result;
}

Json checkpoint(Runtime &runtime, const CaseDefinition &definition,
                std::uint32_t frame) {
    WorldStatistics statistics{};
    require(runtime.world.collect_statistics(statistics),
            "collect conformance statistics");
    const std::uint64_t element_count =
        static_cast<std::uint64_t>(statistics.rigid_body_count) +
        statistics.particle_count + statistics.smoke_particle_count +
        statistics.cloth_vertex_count + statistics.soft_body_node_count +
        statistics.rope_node_count;
    const bool complete = element_count <= complete_state_limit;
    Json resources = Json::array();
    for (std::size_t index = 0U; index < runtime.bodies.size(); ++index)
        resources.push_back(rigid_resource(runtime, index,
            definition.tolerance_profile));
    for (std::size_t index = 0U; index < runtime.constraints.size(); ++index)
        resources.push_back(constraint_resource(runtime, index));
    if (runtime.instance.has_fluid)
        resources.push_back(fluid_resource(runtime, complete));
    if (runtime.instance.has_smoke)
        resources.push_back(smoke_resource(runtime, complete));
    for (std::size_t index = 0U; index < runtime.instance.cloths.size(); ++index)
        resources.push_back(cloth_resource(runtime, index, complete));
    for (std::size_t index = 0U; index < runtime.instance.soft_bodies.size(); ++index)
        resources.push_back(soft_body_resource(runtime, index, complete));
    for (std::size_t index = 0U; index < runtime.instance.ropes.size(); ++index)
        resources.push_back(rope_resource(runtime, index, complete));
    Json invariants = Json::object();
    invariants["cloth_count"] = static_cast<std::uint64_t>(statistics.cloth_count);
    invariants["constraint_count"] =
        static_cast<std::uint64_t>(statistics.rigid_constraint_count);
    invariants["complete_state"] = complete;
    invariants["element_count"] = element_count;
    invariants["combined_momentum"] = vector_json(combined_momentum(runtime));
    invariants["constraints_rebuilt"] = runtime.constraints_rebuilt;
    invariants["finite"] = true;
    invariants["fluid_count"] = static_cast<std::uint64_t>(statistics.fluid_count);
    invariants["generation_changed"] = runtime.generation_changed;
    invariants["particle_count"] =
        static_cast<std::uint64_t>(statistics.particle_count);
    invariants["rigid_body_count"] =
        static_cast<std::uint64_t>(statistics.rigid_body_count);
    invariants["rope_count"] = static_cast<std::uint64_t>(statistics.rope_count);
    invariants["smoke_count"] =
        static_cast<std::uint64_t>(statistics.smoke_particle_count);
    invariants["soft_body_count"] =
        static_cast<std::uint64_t>(statistics.soft_body_count);
    if (std::find(definition.coverage.begin(), definition.coverage.end(),
                  "equal_opposite_transfer") != definition.coverage.end()) {
        Json transfer = Json::object();
        transfer["combined_momentum"] = vector_json(combined_momentum(runtime));
        transfer["observable_contact_count"] = static_cast<std::uint64_t>(
            statistics.contact_count + statistics.fluid_soft_body_contact_count +
            statistics.fluid_rope_contact_count +
            statistics.rope_soft_body_contact_count);
        transfer["systems"] = static_cast<std::uint64_t>(
            statistics.fluid_count + statistics.rigid_body_count +
            statistics.cloth_count + statistics.soft_body_count +
            statistics.rope_count + statistics.smoke_system_count);
        invariants["equal_and_opposite_transfer"] = std::move(transfer);
    }
    Json result = Json::object();
    result["contacts"] = contacts_json(runtime);
    result["frame"] = static_cast<std::uint64_t>(frame);
    result["invariants"] = std::move(invariants);
    result["resources"] = std::move(resources);
    return result;
}

Json timing_json(Runtime &runtime) {
    WorldStepTimings timing{};
    require(runtime.world.collect_step_timings(timing),
            "collect conformance timings");
    Json result = Json::object();
    result["available"] = timing.available;
    result["cloth_ms"] = static_cast<double>(timing.cloth_constraints.total_milliseconds);
    result["fluid_ms"] = static_cast<double>(
        timing.fluid_neighbor_forces.total_milliseconds +
        timing.fluid_integration.total_milliseconds);
    result["rigid_contact_ms"] = static_cast<double>(
        timing.rigid_contact_generation.total_milliseconds +
        timing.rigid_contact_solve.total_milliseconds);
    result["rope_ms"] = static_cast<double>(timing.rope_solve.total_milliseconds);
    result["smoke_ms"] = static_cast<double>(
        timing.smoke_grid.total_milliseconds +
        timing.smoke_advection.total_milliseconds);
    result["soft_body_ms"] =
        static_cast<double>(timing.soft_body_constraints.total_milliseconds);
    result["total_gpu_ms"] = static_cast<double>(timing.total_gpu_milliseconds);
    return result;
}

Json run_case(const CaseDefinition &definition,
              const std::filesystem::path &source_root,
              bool every_frame = false) {
    Runtime runtime = make_runtime(definition, source_root);
    Vec3 gravity{static_cast<float>(definition.gravity[0]),
                 static_cast<float>(definition.gravity[1]),
                 static_cast<float>(definition.gravity[2])};
    Json checkpoints = Json::array();
    if (every_frame ||
        std::find(definition.checkpoints.begin(), definition.checkpoints.end(), 0U)
            != definition.checkpoints.end())
        checkpoints.push_back(checkpoint(runtime, definition, 0U));
    for (std::uint32_t frame = 0U; frame < definition.frames; ++frame) {
        for (const Command &command : definition.commands)
            if (command.frame == frame) apply_command(runtime, command, gravity);
        require(runtime.world.step({
            .timestep=static_cast<float>(definition.timestep),
            .substeps=definition.substeps,.gravity=gravity,
            .collect_kernel_timings=true,.collect_rigid_contacts=true,
            .collect_fluid_contacts=true}), "step conformance world");
        const std::uint32_t completed = frame + 1U;
        if (every_frame ||
            std::find(definition.checkpoints.begin(), definition.checkpoints.end(),
                      completed) != definition.checkpoints.end())
            checkpoints.push_back(checkpoint(runtime, definition, completed));
    }
    const std::string canonical_case = serialize_case(definition);
    Json diagnostics = Json::object();
    if (every_frame) diagnostics["checkpoint_mode"] = "every_frame";
    diagnostics["timings"] = timing_json(runtime);
    diagnostics["state_hash"] = sha256(checkpoints.serialize());
    Json provenance = Json::object();
    provenance["deterministic"] = true;
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
    id<MTLDevice> device =
        (__bridge id<MTLDevice>)runtime.world.native_context().device;
    provenance["device_name"] =
        device != nil ? std::string(device.name.UTF8String) : "unknown";
    provenance["registry_id"] = static_cast<std::uint64_t>(
        device != nil ? device.registryID : 0U);
#else
    provenance["device_ordinal"] =
        static_cast<std::int64_t>(runtime.world.device_ordinal());
#endif
    Json result = Json::object();
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
    result["backend"] = "metal";
#else
    result["backend"] = "cuda";
#endif
    result["case_id"] = definition.id;
    result["case_sha256"] = sha256(canonical_case);
    result["checkpoints"] = std::move(checkpoints);
    result["diagnostics"] = std::move(diagnostics);
    result["provenance"] = std::move(provenance);
    result["schema"] = "parallel-mater-conformance-result/v1";
    return result;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path &path, const std::string &contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) throw std::runtime_error("cannot write " + path.string());
}

bool check_inputs(const std::filesystem::path &source_root) {
    bool valid = true;
    std::set<std::filesystem::path> expected_files;
    for (const auto &definition : case_registry()) {
        const auto path = case_file_path(source_root / "conformance/v1/cases",
                                         definition);
        expected_files.insert(path.filename());
        const std::string canonical = serialize_case(definition);
        if (!std::filesystem::is_regular_file(path) || read_file(path) != canonical) {
            std::cerr << "non-canonical conformance input: " << path << '\n';
            valid = false;
        }
        if (definition.kind == CaseKind::integrated_glb) {
            const auto glb = source_root / definition.glb_path;
            if (!std::filesystem::is_regular_file(glb) ||
                sha256_file(glb) != definition.glb_sha256) {
                std::cerr << "GLB checksum mismatch: " << glb << '\n';
                valid = false;
            }
        }
    }
    const auto cases = source_root / "conformance/v1/cases";
    if (std::filesystem::is_directory(cases))
        for (const auto &entry : std::filesystem::directory_iterator(cases))
            if (entry.path().extension() == ".json" &&
                !expected_files.contains(entry.path().filename())) {
                std::cerr << "unregistered conformance input: " << entry.path() << '\n';
                valid = false;
            }
    const auto manifest_path = source_root / "conformance/v1/manifest.json";
    if (!std::filesystem::is_regular_file(manifest_path)) {
        std::cerr << "missing conformance manifest: " << manifest_path << '\n';
        valid = false;
    } else {
        const std::string manifest = read_file(manifest_path);
        for (const auto &definition : case_registry()) {
            if (definition.glb_path.empty()) continue;
            const std::string entry = "\"" + definition.glb_path +
                                      "\":\"" + definition.glb_sha256 + "\"";
            if (manifest.find(entry) == std::string::npos) {
                std::cerr << "manifest missing GLB checksum: "
                          << definition.glb_path << '\n';
                valid = false;
            }
        }
    }
    return valid;
}

Json device_provenance() {
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) throw std::runtime_error("no Metal device available");
    Json result = Json::object();
    result["device_name"] = std::string(device.name.UTF8String);
    result["host_compiler"] = __VERSION__;
    result["registry_id"] = static_cast<std::uint64_t>(device.registryID);
    return result;
#else
    int device = 0;
    require_cuda(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
                 "get CUDA device properties");
    int driver = 0, runtime = 0;
    require_cuda(cudaDriverGetVersion(&driver), "get CUDA driver version");
    require_cuda(cudaRuntimeGetVersion(&runtime), "get CUDA runtime version");
    Json result = Json::object();
    result["compute_capability"] = std::to_string(properties.major) + "." +
                                   std::to_string(properties.minor);
    result["cuda_driver_version"] = static_cast<std::int64_t>(driver);
    result["cuda_runtime_version"] = static_cast<std::int64_t>(runtime);
    result["device_name"] = properties.name;
    result["host_compiler"] = __VERSION__;
    return result;
#endif
}

void write_manifest(const std::filesystem::path &source_root) {
    Json assets = Json::object();
    for (const auto &definition : case_registry())
        if (!definition.glb_path.empty())
            assets[definition.glb_path] = definition.glb_sha256;
    Json deterministic = Json::object();
    deterministic["contact_order"] = "canonical";
    deterministic["enabled"] = true;
    deterministic["timings_gate_correctness"] = false;
    Json manifest = Json::object();
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
    manifest["backend"] = "metal";
#else
    manifest["backend"] = "cuda";
#endif
    manifest["case_count"] = static_cast<std::uint64_t>(case_registry().size());
    manifest["deterministic_settings"] = std::move(deterministic);
    manifest["device_and_toolchain"] = device_provenance();
    manifest["glb_sha256"] = std::move(assets);
    manifest["result_schema"] = "parallel-mater-conformance-result/v1";
    manifest["schema"] = "parallel-mater-conformance-manifest/v1";
    manifest["source_commit"] = PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT;
    write_file(source_root / "conformance/v1/manifest.json", manifest.serialize());
}

struct Arguments {
    bool list{};
    bool check{};
    bool provenance{};
    bool update{};
    bool every_frame{};
    std::string case_id{};
    std::filesystem::path output{};
};

Arguments parse_arguments(int argc, char **argv) {
    Arguments result{};
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--list") result.list = true;
        else if (argument == "--check-inputs") result.check = true;
        else if (argument == "--provenance") result.provenance = true;
        else if (argument == "--update-goldens") result.update = true;
        else if (argument == "--every-frame") result.every_frame = true;
        else if (argument == "--case" && index + 1 < argc)
            result.case_id = argv[++index];
        else if (argument == "--output" && index + 1 < argc)
            result.output = argv[++index];
        else throw std::runtime_error("unknown or incomplete argument: " + argument);
    }
    const unsigned modes = static_cast<unsigned>(result.list) +
        static_cast<unsigned>(result.check) +
        static_cast<unsigned>(result.provenance) +
        static_cast<unsigned>(result.update || !result.case_id.empty());
    if (modes != 1U)
        throw std::runtime_error("choose exactly one of --list, --check-inputs, "
                                 "--provenance, --case, or --update-goldens");
    if (result.update && !result.output.empty())
        throw std::runtime_error("--update-goldens writes only to golden/cuda; "
                                 "do not combine it with --output");
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
    if (result.update)
        throw std::runtime_error(
            "Metal runner cannot update the CUDA reference goldens");
#endif
    if (result.update && !result.case_id.empty() && result.case_id != "all")
        throw std::runtime_error("golden refreshes must cover --case all");
    if (result.every_frame &&
        (result.update || result.case_id.empty() || result.case_id == "all"))
        throw std::runtime_error(
            "--every-frame requires one named --case and cannot update goldens");
    if (!result.update && !result.case_id.empty() && result.output.empty())
        throw std::runtime_error("--case requires --output");
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const std::filesystem::path source_root =
            PARALLEL_MATER_CONFORMANCE_SOURCE_ROOT;
        if (arguments.list) {
            for (const auto &definition : case_registry())
                std::cout << definition.id << '\n';
            return 0;
        }
        if (arguments.check)
            return check_inputs(source_root) ? 0 : 1;
#if defined(PARALLEL_MATER_CONFORMANCE_METAL)
        if (MTLCreateSystemDefaultDevice() == nil) return 77;
#else
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
#endif
        const bool update = arguments.update;
        if (arguments.provenance) {
            Json provenance = device_provenance();
            provenance["source_commit"] = PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT;
            std::cout << provenance.serialize();
            return 0;
        }
        const std::string selected = update || arguments.case_id.empty()
            ? "all" : arguments.case_id;
        const std::filesystem::path output = update
            ? source_root / "conformance/v1/golden/cuda" : arguments.output;
        std::filesystem::create_directories(output);
        std::vector<const CaseDefinition *> definitions;
        if (selected == "all") {
            for (const auto &definition : case_registry())
                definitions.push_back(&definition);
        } else {
            const CaseDefinition *definition = find_case(selected);
            if (definition == nullptr)
                throw std::runtime_error("unknown conformance case: " + selected);
            definitions.push_back(definition);
        }
        for (const CaseDefinition *definition : definitions) {
            std::cout << "RUN " << definition->id << std::endl;
            const Json result = run_case(
                *definition, source_root, arguments.every_frame);
            write_file(output / (definition->id + ".json"), result.serialize());
        }
        if (update) write_manifest(source_root);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "conformance: " << error.what() << '\n';
        return 2;
    }
}
