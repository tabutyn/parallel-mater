// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#include "systems_internal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <atomic>
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace parallel_mater::metal {
namespace {

#include "parallel_mater_metallib.inc"

constexpr std::uint32_t metal_timestamp_count = 2048U;
constexpr std::uint32_t metal_timing_record_capacity =
    (metal_timestamp_count - 2U) / 2U;
constexpr std::uint32_t metal_final_timestamp = metal_timestamp_count - 1U;

constexpr Status success() noexcept { return {}; }

constexpr Status invalid_argument(const char *message) noexcept {
    return {StatusCode::invalid_argument, 0, message};
}

constexpr Status busy(const char *message) noexcept {
    return {StatusCode::busy, 0, message};
}

constexpr Status out_of_memory(const char *message) noexcept {
    return {StatusCode::out_of_memory, 0, message};
}

constexpr Status capacity_exceeded(const char *message) noexcept {
    return {StatusCode::capacity_exceeded, 0, message};
}

constexpr Status invalid_handle(const char *message) noexcept {
    return {StatusCode::invalid_handle, 0, message};
}

Status metal_failure(NSError *error, const char *message) noexcept {
    return {StatusCode::metal_failure,
            error == nil ? 0 : static_cast<std::int64_t>(error.code), message};
}

struct CompletionState {
    mutable std::mutex mutex{};
    std::condition_variable completed_condition{};
    bool completed{};
    std::int64_t error_code{};
    double gpu_start_time{};
    double gpu_end_time{};
    __strong id<MTLBuffer> fluid_neighbor_overflow{nil};
    std::uint32_t fluid_neighbor_overflow_count{};
};

struct RigidParameters {
    std::uint32_t motion{};
    float inverse_mass{};
    Vec3 inverse_inertia{};
    float linear_damping{};
    float angular_damping{};
    float maximum_linear_speed{};
    float maximum_angular_speed{};
    std::uint32_t has_kinematic_target{};
    RigidBodyState kinematic_target{};
    std::uint32_t mesh_index{};
    float friction{};
    float restitution{};
    float collision_margin{};
};

struct TriangleMeshInfo {
    std::uint32_t vertex_offset{};
    std::uint32_t vertex_count{};
    std::uint32_t index_offset{};
    std::uint32_t index_count{};
    Vec3 minimum{};
    Vec3 maximum{};
    Vec3 bounding_center{};
    float radius{};
    std::uint32_t bvh_node_offset{};
    std::uint32_t bvh_node_count{};
    std::uint32_t solid_plane_offset{};
    std::uint32_t solid_plane_count{};
};

struct ContactRecord {
    Vec3 point{};
    Vec3 normal{};
    float penetration{};
    std::uint32_t found{};
};

struct ContactManifold {
    ContactRecord contacts[8]{};
    std::uint32_t count{};
    std::uint32_t event_offset{};
    std::uint32_t color{};
};

struct BvhNode {
    Vec3 minimum{};
    Vec3 maximum{};
    std::uint32_t left{};
    std::uint32_t right{};
    std::uint32_t first_triangle{};
    std::uint32_t triangle_count{};
};

struct MeshLeafInfo {
    std::uint32_t offset{};
    std::uint32_t count{};
};

struct WorldAabb {
    Vec3 minimum{};
    Vec3 maximum{};
};

struct StepConstants {
    float timestep{};
    Vec3 gravity{};
    std::uint32_t body_count{};
    std::uint32_t constraint_capacity{};
    std::uint32_t collect_rigid_contacts{};
    std::uint32_t rigid_event_capacity{};
    std::uint32_t substeps{};
};

struct RigidConstraintResource {
    std::uint32_t generation{1U};
    std::uint32_t alive{};
    std::uint32_t type{};
    std::uint32_t body_a{};
    std::uint32_t body_b{};
    std::uint32_t enabled{};
    std::uint32_t broken{};
    Vec3 local_anchor_a{};
    Vec3 local_anchor_b{};
    Quaternion local_orientation_a{};
    Quaternion local_orientation_b{};
    float breaking_impulse_threshold{};
    float applied_impulse{};
    std::uint32_t linear_limit_axes{};
    Vec3 linear_limit_lower{};
    Vec3 linear_limit_upper{};
    std::uint32_t angular_limit_axes{};
    Vec3 angular_limit_lower{};
    Vec3 angular_limit_upper{};
    std::uint32_t linear_spring_axes{};
    Vec3 linear_spring_stiffness{};
    Vec3 linear_spring_damping{};
    std::uint32_t angular_spring_axes{};
    Vec3 angular_spring_stiffness{};
    Vec3 angular_spring_damping{};
    std::uint32_t linear_motor_enabled{};
    std::uint32_t angular_motor_enabled{};
    float linear_target_velocity{};
    float linear_maximum_impulse{};
    float angular_target_velocity{};
    float angular_maximum_impulse{};
    std::uint32_t solver_iterations{};
    std::uint32_t disable_collisions{};
};

struct RigidConstraintAxisGeometry {
    Vec3 axis{};
    Vec3 inverse_angular_a{};
    Vec3 inverse_angular_b{};
    float angular_denominator{};
    float linear_denominator{};
};

struct RigidConstraintGeometry {
    std::uint32_t valid{};
    Vec3 arm_a{};
    Vec3 arm_b{};
    Vec3 anchor_error{};
    Vec3 rotation_error{};
    Vec3 hinge_alignment_error{};
    RigidConstraintAxisGeometry axes[3]{};
};

struct RigidCompound {
    std::uint32_t root{};
    std::uint32_t member_count{};
    std::uint32_t eligible{};
    std::uint32_t blocked{};
    Vec3 center{};
    float inverse_mass{};
    Vec3 inverse_inertia[3]{};
    std::uint32_t projection_root{};
    std::uint32_t projection_movable{};
    Vec3 projection_translation{};
};

static_assert(sizeof(RigidCompound) == 88U);

struct HandleSlot {
    std::uint32_t generation{1};
    std::uint32_t dense_index{};
    bool alive{};
};

struct CollisionPlane {
    Vec3 normal{};
    float offset{};
};

struct TriangleMeshStorage {
    std::uint32_t generation{1};
    bool alive{};
    std::vector<Vec3> vertices{};
    std::vector<std::uint32_t> indices{};
    std::vector<BvhNode> bvh_nodes{};
    std::vector<CollisionPlane> solid_planes{};
    Vec3 minimum{};
    Vec3 maximum{};
};

static_assert(sizeof(RigidBodyState) == 52U);
static_assert(sizeof(RigidParameters) == 108U);
static_assert(sizeof(TriangleMeshInfo) == 72U);
static_assert(sizeof(ContactRecord) == 32U);
static_assert(sizeof(ContactManifold) == 268U);
static_assert(sizeof(BvhNode) == 40U);
static_assert(sizeof(MeshLeafInfo) == 8U);
static_assert(sizeof(WorldAabb) == 24U);
static_assert(sizeof(StepConstants) == 36U);
static_assert(sizeof(RigidBodyId) == 8U);
static_assert(sizeof(RigidContactEvent) == 60U);
static_assert(sizeof(RigidConstraintResource) == 236U);
static_assert(sizeof(RigidConstraintAxisGeometry) == 44U);
static_assert(sizeof(RigidConstraintGeometry) == 196U);

bool completion_ready(const std::shared_ptr<CompletionState> &state) noexcept {
    if (!state) {
        return true;
    }
    std::lock_guard lock(state->mutex);
    return state->completed;
}

Status wait_for_completion(const std::shared_ptr<CompletionState> &state) noexcept {
    if (!state) {
        return success();
    }
    std::unique_lock lock(state->mutex);
    state->completed_condition.wait(lock, [&state] { return state->completed; });
    if (state->error_code != 0) {
        return {StatusCode::metal_failure, state->error_code,
                "Metal 4 command submission failed"};
    }
    if (state->fluid_neighbor_overflow_count != 0U) {
        return {StatusCode::capacity_exceeded, 0,
                "fluid neighbor count exceeded maximum_neighbors"};
    }
    return success();
}

bool finite(Vec3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool finite(Quaternion value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

bool finite(RigidBodyState value) noexcept {
    return finite(value.position) && finite(value.orientation) &&
           finite(value.linear_velocity) && finite(value.angular_velocity);
}

bool valid_constraint_options(const RigidConstraintOptions &options) noexcept {
    const auto quaternion_valid = [](Quaternion value) {
        const float size = value.x * value.x + value.y * value.y +
                           value.z * value.z + value.w * value.w;
        return finite(value) && size > 1.0e-12F;
    };
    const auto axes_valid = [](std::uint8_t axes) {
        return (axes & ~rigid_constraint_all_axes) == 0U;
    };
    const auto nonnegative = [](Vec3 value) {
        return finite(value) && value.x >= 0.0F && value.y >= 0.0F &&
               value.z >= 0.0F;
    };
    const auto limits_valid = [&](const RigidConstraintLimitOptions &limits) {
        return axes_valid(limits.axes) && finite(limits.lower) &&
               finite(limits.upper) && limits.lower.x <= limits.upper.x &&
               limits.lower.y <= limits.upper.y &&
               limits.lower.z <= limits.upper.z;
    };
    const auto springs_valid = [&](const RigidConstraintSpringOptions &springs) {
        return axes_valid(springs.axes) && nonnegative(springs.stiffness) &&
               nonnegative(springs.damping);
    };
    return options.body_a != options.body_b && finite(options.local_anchor_a) &&
           finite(options.local_anchor_b) &&
           quaternion_valid(options.local_orientation_a) &&
           quaternion_valid(options.local_orientation_b) &&
           limits_valid(options.linear_limits) &&
           limits_valid(options.angular_limits) &&
           springs_valid(options.linear_springs) &&
           springs_valid(options.angular_springs) &&
           std::isfinite(options.breaking_impulse_threshold) &&
           options.breaking_impulse_threshold >= 0.0F &&
           options.solver_iterations != 0U &&
           options.solver_iterations <= 64U &&
           std::isfinite(options.motor.linear_target_velocity) &&
           std::isfinite(options.motor.linear_maximum_impulse) &&
           options.motor.linear_maximum_impulse >= 0.0F &&
           std::isfinite(options.motor.angular_target_velocity) &&
           std::isfinite(options.motor.angular_maximum_impulse) &&
           options.motor.angular_maximum_impulse >= 0.0F;
}

Vec3 add(Vec3 left, Vec3 right) noexcept {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 subtract(Vec3 left, Vec3 right) noexcept {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 multiply(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

Vec3 cross(Vec3 left, Vec3 right) noexcept {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

Vec3 rotate(Quaternion orientation, Vec3 value) noexcept {
    const Vec3 vector{orientation.x, orientation.y, orientation.z};
    const Vec3 first = cross(vector, value);
    const Vec3 second = cross(vector, add(first, multiply(
        value, orientation.w)));
    return add(value, multiply(second, 2.0F));
}

Vec3 inverse_inertia_world(const RigidParameters &parameters,
                           const RigidBodyState &state,
                           Vec3 value) noexcept {
    const Quaternion inverse{-state.orientation.x, -state.orientation.y,
                             -state.orientation.z, state.orientation.w};
    const Vec3 local = rotate(inverse, value);
    return rotate(state.orientation,
                  {local.x * parameters.inverse_inertia.x,
                   local.y * parameters.inverse_inertia.y,
                   local.z * parameters.inverse_inertia.z});
}

float length_squared(Vec3 value) noexcept {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

Vec3 component_min(Vec3 first, Vec3 second) noexcept {
    return {std::min(first.x, second.x), std::min(first.y, second.y),
            std::min(first.z, second.z)};
}

Vec3 component_max(Vec3 first, Vec3 second) noexcept {
    return {std::max(first.x, second.x), std::max(first.y, second.y),
            std::max(first.z, second.z)};
}

float dot(Vec3 first, Vec3 second) noexcept {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

Vec3 normalized_or(Vec3 value, Vec3 fallback) noexcept {
    const float squared = length_squared(value);
    return squared <= 1.0e-12F
               ? fallback
               : multiply(value, 1.0F / std::sqrt(squared));
}

Vec3 inverse_rotate(Quaternion orientation, Vec3 value) noexcept {
    return rotate({-orientation.x, -orientation.y, -orientation.z,
                   orientation.w},
                  value);
}

bool hit_box_separates_triangle(Vec3 axis, Vec3 first, Vec3 second,
                                Vec3 third, Vec3 half_extents) noexcept {
    if (length_squared(axis) <= 1.0e-12F) return false;
    const float first_projection = dot(axis, first);
    const float second_projection = dot(axis, second);
    const float third_projection = dot(axis, third);
    const float minimum = std::min(first_projection,
                                   std::min(second_projection,
                                            third_projection));
    const float maximum = std::max(first_projection,
                                   std::max(second_projection,
                                            third_projection));
    const float radius = std::abs(axis.x) * half_extents.x +
                         std::abs(axis.y) * half_extents.y +
                         std::abs(axis.z) * half_extents.z;
    return minimum > radius || maximum < -radius;
}

bool hit_box_overlaps_triangle(Vec3 first, Vec3 second, Vec3 third,
                               Vec3 half_extents) noexcept {
    const Vec3 minimum = component_min(first, component_min(second, third));
    const Vec3 maximum = component_max(first, component_max(second, third));
    if (minimum.x > half_extents.x || maximum.x < -half_extents.x ||
        minimum.y > half_extents.y || maximum.y < -half_extents.y ||
        minimum.z > half_extents.z || maximum.z < -half_extents.z) {
        return false;
    }
    const Vec3 edges[3]{subtract(second, first), subtract(third, second),
                        subtract(first, third)};
    if (hit_box_separates_triangle(cross(edges[0], edges[1]), first, second,
                                   third, half_extents)) {
        return false;
    }
    constexpr Vec3 axes[3]{{1.0F, 0.0F, 0.0F},
                           {0.0F, 1.0F, 0.0F},
                           {0.0F, 0.0F, 1.0F}};
    for (const Vec3 edge : edges) {
        for (const Vec3 axis : axes) {
            if (hit_box_separates_triangle(cross(edge, axis), first, second,
                                           third, half_extents)) {
                return false;
            }
        }
    }
    return true;
}

Vec3 hit_box_local_point(HitBox box, Vec3 point) noexcept {
    return inverse_rotate(box.orientation, subtract(point, box.center));
}

bool bounds_overlap(Vec3 minimum_a, Vec3 maximum_a, Vec3 minimum_b,
                    Vec3 maximum_b) noexcept {
    return minimum_a.x <= maximum_b.x && maximum_a.x >= minimum_b.x &&
           minimum_a.y <= maximum_b.y && maximum_a.y >= minimum_b.y &&
           minimum_a.z <= maximum_b.z && maximum_a.z >= minimum_b.z;
}

Vec3 closest_on_triangle(Vec3 point, Vec3 a, Vec3 b, Vec3 c) noexcept {
    const Vec3 ab = subtract(b, a);
    const Vec3 ac = subtract(c, a);
    const Vec3 ap = subtract(point, a);
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) return a;
    const Vec3 bp = subtract(point, b);
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F)
        return add(a, multiply(ab, d1 / (d1 - d3)));
    const Vec3 cp = subtract(point, c);
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F)
        return add(a, multiply(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F)
        return add(b, multiply(subtract(c, b),
                               (d4 - d3) /
                                   ((d4 - d3) + (d5 - d6))));
    const float inverse = 1.0F / (va + vb + vc);
    return add(a, add(multiply(ab, vb * inverse),
                      multiply(ac, vc * inverse)));
}

std::vector<CollisionPlane> closed_convex_planes(
    const std::vector<Vec3> &vertices,
    const std::vector<std::uint32_t> &indices) {
    std::map<std::array<float, 3>, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(vertices.size());
    Vec3 center{};
    for (std::uint32_t vertex = 0U; vertex < vertices.size(); ++vertex) {
        const Vec3 point = vertices[vertex];
        const auto entry = welded.emplace(
            std::array<float, 3>{point.x, point.y, point.z},
            static_cast<std::uint32_t>(welded.size()));
        remap[vertex] = entry.first->second;
        if (entry.second) center = add(center, point);
    }
    if (welded.size() < 4U) return {};
    center = multiply(center, 1.0F / static_cast<float>(welded.size()));
    std::unordered_map<std::uint64_t, std::uint32_t> edges;
    for (std::size_t index = 0U; index < indices.size(); index += 3U) {
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const std::uint32_t a = remap[indices[index + edge]];
            const std::uint32_t b =
                remap[indices[index + (edge + 1U) % 3U]];
            ++edges[(static_cast<std::uint64_t>(std::min(a, b)) << 32U) |
                    std::max(a, b)];
        }
    }
    for (const auto &edge : edges)
        if (edge.second != 2U) return {};
    float scale = 0.0F;
    for (const Vec3 vertex : vertices)
        scale = std::max(
            scale, std::sqrt(length_squared(subtract(vertex, center))));
    const float tolerance = std::max(1.0e-7F, scale * 1.0e-5F);
    std::vector<CollisionPlane> planes;
    planes.reserve(indices.size() / 3U);
    for (std::size_t index = 0U; index < indices.size(); index += 3U) {
        const Vec3 a = vertices[indices[index]];
        Vec3 normal = normalized_or(
            cross(subtract(vertices[indices[index + 1U]], a),
                  subtract(vertices[indices[index + 2U]], a)),
            {});
        const float side = dot(normal, subtract(a, center));
        if (std::abs(side) <= tolerance) return {};
        if (side < 0.0F) normal = multiply(normal, -1.0F);
        const float offset = dot(normal, a);
        for (const Vec3 vertex : vertices)
            if (dot(normal, vertex) - offset > tolerance) return {};
        planes.push_back({normal, offset});
    }
    return planes;
}

void build_mesh_bvh(TriangleMeshStorage &mesh) {
    const std::uint32_t triangle_count =
        static_cast<std::uint32_t>(mesh.indices.size() / 3U);
    std::vector<std::uint32_t> order(triangle_count);
    std::iota(order.begin(), order.end(), 0U);
    mesh.bvh_nodes.clear();
    mesh.bvh_nodes.reserve(triangle_count * 2U);
    const auto coordinate = [](Vec3 value, int axis) {
        return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
    };
    const auto centroid = [&](std::uint32_t triangle) {
        const std::uint32_t base = triangle * 3U;
        return multiply(
            add(add(mesh.vertices[mesh.indices[base]],
                    mesh.vertices[mesh.indices[base + 1U]]),
                mesh.vertices[mesh.indices[base + 2U]]),
            1.0F / 3.0F);
    };
    std::function<std::uint32_t(std::uint32_t, std::uint32_t)> build =
        [&](std::uint32_t begin, std::uint32_t end) {
            BvhNode node{};
            const float maximum = std::numeric_limits<float>::max();
            node.minimum = {maximum, maximum, maximum};
            node.maximum = {-maximum, -maximum, -maximum};
            for (std::uint32_t item = begin; item < end; ++item) {
                const std::uint32_t base = order[item] * 3U;
                for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
                    const Vec3 vertex =
                        mesh.vertices[mesh.indices[base + corner]];
                    node.minimum = component_min(node.minimum, vertex);
                    node.maximum = component_max(node.maximum, vertex);
                }
            }
            const std::uint32_t node_index =
                static_cast<std::uint32_t>(mesh.bvh_nodes.size());
            mesh.bvh_nodes.push_back(node);
            if (end - begin <= 4U) {
                mesh.bvh_nodes[node_index].first_triangle = begin;
                mesh.bvh_nodes[node_index].triangle_count = end - begin;
                return node_index;
            }
            const Vec3 extent = subtract(node.maximum, node.minimum);
            const int axis = extent.x >= extent.y && extent.x >= extent.z
                                 ? 0
                                 : (extent.y >= extent.z ? 1 : 2);
            std::stable_sort(
                order.begin() + begin, order.begin() + end,
                [&](std::uint32_t first, std::uint32_t second) {
                    const float first_value = coordinate(centroid(first), axis);
                    const float second_value =
                        coordinate(centroid(second), axis);
                    return first_value < second_value ||
                           (first_value == second_value && first < second);
                });
            const std::uint32_t middle = begin + (end - begin) / 2U;
            const std::uint32_t left = build(begin, middle);
            const std::uint32_t right = build(middle, end);
            mesh.bvh_nodes[node_index].left = left;
            mesh.bvh_nodes[node_index].right = right;
            return node_index;
        };
    (void)build(0U, triangle_count);

    std::vector<std::uint32_t> reordered(mesh.indices.size());
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t source = order[triangle] * 3U;
        const std::uint32_t destination = triangle * 3U;
        reordered[destination] = mesh.indices[source];
        reordered[destination + 1U] = mesh.indices[source + 1U];
        reordered[destination + 2U] = mesh.indices[source + 2U];
    }
    mesh.indices.swap(reordered);
}

Quaternion normalized(Quaternion value) noexcept {
    const float squared = value.x * value.x + value.y * value.y +
                          value.z * value.z + value.w * value.w;
    const float scale = 1.0F / std::sqrt(squared);
    return {value.x * scale, value.y * scale, value.z * scale,
            value.w * scale};
}

detail::MetalTimingRecord *begin_timing(
    id<MTL4ComputeCommandEncoder> encoder,
    detail::MetalTimingContext &context, detail::MetalTimingStage stage,
    std::uint32_t launch_count = 1U) {
    if (!context.enabled || stage == detail::MetalTimingStage::none ||
        launch_count == 0U)
        return nullptr;
    if (context.counter_heap == nullptr || context.records == nullptr ||
        context.record_count >= context.record_capacity ||
        context.next_index + 1U >= context.final_index) {
        context.overflowed = true;
        return nullptr;
    }
    detail::MetalTimingRecord &record =
        context.records[context.record_count++];
    record = {stage, context.next_index, context.next_index + 1U,
              launch_count};
    context.next_index += 2U;
    id<MTL4CounterHeap> heap =
        (__bridge id<MTL4CounterHeap>)context.counter_heap;
    [encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                  intoHeap:heap
                                   atIndex:record.begin_index];
    return &record;
}

void end_timing(id<MTL4ComputeCommandEncoder> encoder,
                detail::MetalTimingContext &context,
                const detail::MetalTimingRecord *record) {
    if (record == nullptr) return;
    id<MTL4CounterHeap> heap =
        (__bridge id<MTL4CounterHeap>)context.counter_heap;
    [encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                  intoHeap:heap
                                   atIndex:record->end_index];
}

KernelTiming *timing_for_stage(WorldStepTimings &output,
                               detail::MetalTimingStage stage) noexcept {
    using detail::MetalTimingStage;
    switch (stage) {
    case MetalTimingStage::rigid_integration:
        return &output.rigid_integration;
    case MetalTimingStage::rigid_world_bounds:
        return &output.rigid_world_bounds;
    case MetalTimingStage::rigid_pair_filter:
        return &output.rigid_pair_filter;
    case MetalTimingStage::rigid_pair_compaction:
        return &output.rigid_pair_compaction;
    case MetalTimingStage::rigid_contact_evaluation:
        return &output.rigid_contact_evaluation;
    case MetalTimingStage::rigid_contact_solve:
        return &output.rigid_contact_solve;
    case MetalTimingStage::rigid_input_clear:
        return &output.rigid_input_clear;
    case MetalTimingStage::fluid_spawn:
        return &output.fluid_spawn;
    case MetalTimingStage::fluid_neighbor_sort:
        return &output.fluid_neighbor_sort;
    case MetalTimingStage::fluid_neighbor_forces:
        return &output.fluid_neighbor_forces;
    case MetalTimingStage::fluid_integration:
        return &output.fluid_integration;
    case MetalTimingStage::fluid_static_contacts:
        return &output.fluid_static_contacts;
    case MetalTimingStage::fluid_cloth_contacts:
        return &output.fluid_cloth_contacts;
    case MetalTimingStage::fluid_outflow_compaction:
        return &output.fluid_outflow_compaction;
    case MetalTimingStage::fluid_smoke_exchange:
        return &output.fluid_smoke_exchange;
    case MetalTimingStage::cloth_prediction:
        return &output.cloth_prediction;
    case MetalTimingStage::cloth_constraints:
        return &output.cloth_constraints;
    case MetalTimingStage::cloth_contacts:
        return &output.cloth_contacts;
    case MetalTimingStage::soft_body_prediction:
        return &output.soft_body_prediction;
    case MetalTimingStage::soft_body_constraints:
        return &output.soft_body_constraints;
    case MetalTimingStage::soft_body_contacts:
        return &output.soft_body_contacts;
    case MetalTimingStage::soft_body_cloth_contacts:
        return &output.soft_body_cloth_contacts;
    case MetalTimingStage::fluid_soft_body_contacts:
        return &output.fluid_soft_body_contacts;
    case MetalTimingStage::fluid_rope_contacts:
        return &output.fluid_rope_contacts;
    case MetalTimingStage::rope_solve:
        return &output.rope_solve;
    case MetalTimingStage::rope_soft_body_contacts:
        return &output.rope_soft_body_contacts;
    case MetalTimingStage::smoke_grid:
        return &output.smoke_grid;
    case MetalTimingStage::smoke_advection:
        return &output.smoke_advection;
    case MetalTimingStage::smoke_emission:
        return &output.smoke_emission;
    case MetalTimingStage::none:
        return nullptr;
    }
    return nullptr;
}

} // namespace

struct FrameToken::Impl {
    id<MTLSharedEvent> event{nil};
    std::uint64_t event_value{};
    std::shared_ptr<CompletionState> completion{};
    __strong MTL4CommitOptions *commit_options{nil};
};

struct World::Impl {
    WorldOptions options{};
    id<MTLDevice> device{nil};
    id<MTL4CommandQueue> command_queue{nil};
    id<MTL4CommandAllocator> command_allocator{nil};
    id<MTL4CommandBuffer> command_buffer{nil};
    id<MTLSharedEvent> completion_event{nil};
    id<MTLLibrary> library{nil};
    id<MTLComputePipelineState> noop_pipeline{nil};
    id<MTLComputePipelineState> rigid_integrate_pipeline{nil};
    id<MTLComputePipelineState> rigid_compound_pipeline{nil};
    id<MTLComputePipelineState> rigid_constraint_pipeline{nil};
    id<MTLComputePipelineState> rigid_world_bounds_pipeline{nil};
    id<MTLComputePipelineState> rigid_pair_filter_pipeline{nil};
    id<MTLComputePipelineState> rigid_pair_count_rows_pipeline{nil};
    id<MTLComputePipelineState> rigid_pair_prefix_rows_pipeline{nil};
    id<MTLComputePipelineState> rigid_pair_scatter_rows_pipeline{nil};
    id<MTLComputePipelineState> rigid_contact_generate_pipeline{nil};
    id<MTLComputePipelineState> rigid_contact_reduce_pipeline{nil};
    id<MTLComputePipelineState> rigid_advance_substep_pipeline{nil};
    id<MTLComputePipelineState> rigid_clear_pipeline{nil};
    id<MTL4ArgumentTable> rigid_argument_table{nil};
    id<MTL4CounterHeap> timestamp_heap{nil};
    id<MTLResidencySet> residency_set{nil};

    id<MTLBuffer> rigid_ids{nil};
    id<MTLBuffer> rigid_states{nil};
    id<MTLBuffer> rigid_previous_states{nil};
    id<MTLBuffer> rigid_frame_states{nil};
    id<MTLBuffer> rigid_parameters{nil};
    id<MTLBuffer> rigid_forces{nil};
    id<MTLBuffer> rigid_torques{nil};
    id<MTLBuffer> step_constants{nil};
    id<MTLBuffer> mesh_vertices{nil};
    id<MTLBuffer> mesh_indices{nil};
    id<MTLBuffer> mesh_infos{nil};
    id<MTLBuffer> mesh_bvh_nodes{nil};
    id<MTLBuffer> mesh_leaf_infos{nil};
    id<MTLBuffer> mesh_bvh_leaves{nil};
    id<MTLBuffer> mesh_solid_planes{nil};
    id<MTLBuffer> contact_records{nil};
    id<MTLBuffer> rigid_color_owners{nil};
    id<MTLBuffer> rigid_world_bounds{nil};
    id<MTLBuffer> rigid_pair_flags{nil};
    id<MTLBuffer> rigid_active_pairs{nil};
    id<MTLBuffer> rigid_active_pair_count{nil};
    id<MTLBuffer> rigid_pair_row_offsets{nil};
    id<MTLBuffer> rigid_substep_index{nil};
    id<MTLBuffer> rigid_constraints{nil};
    id<MTLBuffer> rigid_constraint_geometry{nil};
    id<MTLBuffer> rigid_compounds{nil};
    id<MTLBuffer> rigid_contact_events{nil};
    id<MTLBuffer> rigid_contact_count_buffer{nil};

    std::vector<HandleSlot> rigid_slots{};
    std::vector<TriangleMeshId> rigid_meshes{};
    std::vector<Vec3> pending_impulses{};
    std::vector<Vec3> pending_angular_impulses{};
    std::vector<TriangleMeshStorage> meshes{};
    std::vector<RigidConstraintOptions> rigid_constraint_options{};
    std::vector<Vec3> debug_input_forces{};
    std::vector<Vec3> debug_input_torques{};
    std::vector<PhysicsDebugFrame> debug_frames{};
    std::uint32_t rigid_body_count{};
    std::uint32_t rigid_constraint_count{};
    std::uint32_t triangle_mesh_count{};
    std::uint64_t revision{};
    detail::MetalSystems systems{};

    std::mutex submission_mutex{};
    FrameToken synchronous_completion{};
    std::shared_ptr<CompletionState> latest_completion{};
    std::uint64_t next_event_value{1};
    std::uint64_t frame_index{};
    bool last_frame_profiled{};
    bool last_timing_overflow{};
    std::array<detail::MetalTimingRecord, metal_timing_record_capacity>
        timing_records{};
    std::uint32_t timing_record_count{};
    std::size_t debug_next_frame{};
    std::size_t debug_frame_count{};
    std::uint64_t debug_captured_frame{};
    StepOptions last_step_options{};

    ~Impl() {
        (void)wait_for_completion(latest_completion);
    }

    [[nodiscard]] bool idle() const noexcept {
        return completion_ready(latest_completion);
    }

    [[nodiscard]] bool valid(TriangleMeshId id) const noexcept {
        return id.index < meshes.size() && meshes[id.index].alive &&
               meshes[id.index].generation == id.generation;
    }

    [[nodiscard]] bool valid(RigidBodyId id,
                             std::uint32_t &dense_index) const noexcept {
        if (id.index >= rigid_slots.size()) {
            return false;
        }
        const HandleSlot &slot = rigid_slots[id.index];
        if (!slot.alive || slot.generation != id.generation) {
            return false;
        }
        dense_index = slot.dense_index;
        return true;
    }

    [[nodiscard]] bool valid(RigidConstraintId id,
                             RigidConstraintResource *&resource) const noexcept {
        if (id.index >= options.rigid_constraint_capacity) return false;
        auto *resources = static_cast<RigidConstraintResource *>(
            rigid_constraints.contents);
        RigidConstraintResource &candidate = resources[id.index];
        if (candidate.alive == 0U || candidate.generation != id.generation) {
            return false;
        }
        resource = &candidate;
        return true;
    }

    [[nodiscard]] std::uint32_t rigid_contact_event_count() const noexcept {
        if (rigid_contact_count_buffer == nil) return 0U;
        return std::min(
            *static_cast<const std::uint32_t *>(
                rigid_contact_count_buffer.contents),
            options.contact_capacity);
    }

    [[nodiscard]] Status record_debug_frame() noexcept {
        if (options.physics_debug.frame_capacity == 0U || frame_index == 0U ||
            debug_captured_frame == frame_index)
            return success();
        if (frame_index % options.physics_debug.frame_stride != 0U) {
            debug_captured_frame = frame_index;
            return success();
        }
        try {
            PhysicsDebugFrame &output = debug_frames[debug_next_frame];
            output.frame_index = frame_index;
            output.timestep = last_step_options.timestep;
            output.gravity = last_step_options.gravity;
            output.rigid_bodies.resize(rigid_body_count);
            const auto *ids =
                static_cast<const RigidBodyId *>(rigid_ids.contents);
            const auto *states =
                static_cast<const RigidBodyState *>(rigid_states.contents);
            for (std::uint32_t index = 0U; index < rigid_body_count; ++index)
                output.rigid_bodies[index] = {
                    ids[index], states[index], debug_input_forces[index],
                    debug_input_torques[index]};
            Status status = systems.append_debug_samples(output, frame_index);
            if (!status) return status;
            const std::uint32_t rigid_contact_count =
                rigid_contact_event_count();
            const auto *rigid_events = static_cast<const RigidContactEvent *>(
                rigid_contact_events.contents);
            output.rigid_contacts.assign(
                rigid_events, rigid_events + rigid_contact_count);
            debug_next_frame = (debug_next_frame + 1U) % debug_frames.size();
            debug_frame_count = std::min(
                debug_frame_count + 1U, debug_frames.size());
            debug_captured_frame = frame_index;
            return success();
        } catch (const std::bad_alloc &) {
            return out_of_memory("Could not record Metal physics debug frame");
        } catch (...) {
            return {StatusCode::internal_error, 0,
                    "Unexpected Metal physics debug capture failure"};
        }
    }

    [[nodiscard]] Status rebuild_mesh_buffers() noexcept {
        @autoreleasepool {
            try {
                std::vector<Vec3> vertices;
                std::vector<std::uint32_t> indices;
                std::vector<BvhNode> bvh_nodes;
                std::vector<std::uint32_t> bvh_leaves;
                std::vector<CollisionPlane> solid_planes;
                std::vector<TriangleMeshInfo> infos(meshes.size());
                std::vector<MeshLeafInfo> leaf_infos(meshes.size());
                for (std::uint32_t slot = 0; slot < meshes.size(); ++slot) {
                    const TriangleMeshStorage &mesh = meshes[slot];
                    if (!mesh.alive) continue;
                    if (vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
                        indices.size() > std::numeric_limits<std::uint32_t>::max() ||
                        bvh_nodes.size() >
                            std::numeric_limits<std::uint32_t>::max() ||
                        bvh_leaves.size() >
                            std::numeric_limits<std::uint32_t>::max() ||
                        solid_planes.size() >
                            std::numeric_limits<std::uint32_t>::max() ||
                        mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
                        mesh.indices.size() > std::numeric_limits<std::uint32_t>::max() ||
                        mesh.bvh_nodes.size() >
                            std::numeric_limits<std::uint32_t>::max() ||
                        mesh.solid_planes.size() >
                            std::numeric_limits<std::uint32_t>::max()) {
                        return capacity_exceeded(
                            "Metal rigid mesh data exceeds uint32 ranges");
                    }
                    TriangleMeshInfo &info = infos[slot];
                    info.vertex_offset = static_cast<std::uint32_t>(vertices.size());
                    info.vertex_count =
                        static_cast<std::uint32_t>(mesh.vertices.size());
                    info.index_offset = static_cast<std::uint32_t>(indices.size());
                    info.index_count =
                        static_cast<std::uint32_t>(mesh.indices.size());
                    info.minimum = mesh.minimum;
                    info.maximum = mesh.maximum;
                    info.bounding_center = multiply(
                        add(mesh.minimum, mesh.maximum), 0.5F);
                    info.bvh_node_offset =
                        static_cast<std::uint32_t>(bvh_nodes.size());
                    info.bvh_node_count =
                        static_cast<std::uint32_t>(mesh.bvh_nodes.size());
                    MeshLeafInfo &leaf_info = leaf_infos[slot];
                    leaf_info.offset =
                        static_cast<std::uint32_t>(bvh_leaves.size());
                    info.solid_plane_offset =
                        static_cast<std::uint32_t>(solid_planes.size());
                    info.solid_plane_count =
                        static_cast<std::uint32_t>(mesh.solid_planes.size());
                    for (const Vec3 vertex : mesh.vertices) {
                        info.radius = std::max(
                            info.radius,
                            std::sqrt(length_squared(subtract(
                                vertex, info.bounding_center))));
                    }
                    vertices.insert(vertices.end(), mesh.vertices.begin(),
                                    mesh.vertices.end());
                    indices.insert(indices.end(), mesh.indices.begin(),
                                   mesh.indices.end());
                    for (std::uint32_t node_index = 0U;
                         node_index < mesh.bvh_nodes.size(); ++node_index) {
                        BvhNode node = mesh.bvh_nodes[node_index];
                        if (node.triangle_count == 0U) {
                            node.left += info.bvh_node_offset;
                            node.right += info.bvh_node_offset;
                        } else {
                            bvh_leaves.push_back(
                                info.bvh_node_offset + node_index);
                        }
                        bvh_nodes.push_back(node);
                    }
                    leaf_info.count = static_cast<std::uint32_t>(
                        bvh_leaves.size() - leaf_info.offset);
                    solid_planes.insert(solid_planes.end(),
                                        mesh.solid_planes.begin(),
                                        mesh.solid_planes.end());
                }

                const NSUInteger vertex_bytes = std::max<NSUInteger>(
                    sizeof(Vec3), vertices.size() * sizeof(Vec3));
                const NSUInteger index_bytes = std::max<NSUInteger>(
                    sizeof(std::uint32_t),
                    indices.size() * sizeof(std::uint32_t));
                const NSUInteger info_bytes = std::max<NSUInteger>(
                    sizeof(TriangleMeshInfo),
                    infos.size() * sizeof(TriangleMeshInfo));
                const NSUInteger bvh_bytes = std::max<NSUInteger>(
                    sizeof(BvhNode), bvh_nodes.size() * sizeof(BvhNode));
                const NSUInteger leaf_info_bytes = std::max<NSUInteger>(
                    sizeof(MeshLeafInfo),
                    leaf_infos.size() * sizeof(MeshLeafInfo));
                const NSUInteger bvh_leaf_bytes = std::max<NSUInteger>(
                    sizeof(std::uint32_t),
                    bvh_leaves.size() * sizeof(std::uint32_t));
                const NSUInteger plane_bytes = std::max<NSUInteger>(
                    sizeof(CollisionPlane),
                    solid_planes.size() * sizeof(CollisionPlane));
                id<MTLBuffer> next_vertices = [device
                    newBufferWithLength:vertex_bytes
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> next_indices = [device
                    newBufferWithLength:index_bytes
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> next_infos = [device
                    newBufferWithLength:info_bytes
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> next_bvh_nodes = [device
                    newBufferWithLength:bvh_bytes
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> next_leaf_infos = [device
                    newBufferWithLength:leaf_info_bytes
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> next_bvh_leaves = [device
                    newBufferWithLength:bvh_leaf_bytes
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> next_solid_planes = [device
                    newBufferWithLength:plane_bytes
                               options:MTLResourceStorageModeShared];
                if (next_vertices == nil || next_indices == nil ||
                    next_infos == nil || next_bvh_nodes == nil ||
                    next_leaf_infos == nil || next_bvh_leaves == nil ||
                    next_solid_planes == nil) {
                    return metal_failure(
                        nil, "Could not allocate Metal rigid mesh buffers");
                }
                if (!vertices.empty()) {
                    std::memcpy(next_vertices.contents, vertices.data(),
                                vertices.size() * sizeof(Vec3));
                }
                if (!indices.empty()) {
                    std::memcpy(next_indices.contents, indices.data(),
                                indices.size() * sizeof(std::uint32_t));
                }
                std::memset(next_infos.contents, 0, info_bytes);
                if (!infos.empty()) {
                    std::memcpy(next_infos.contents, infos.data(),
                                infos.size() * sizeof(TriangleMeshInfo));
                }
                std::memset(next_bvh_nodes.contents, 0, bvh_bytes);
                if (!bvh_nodes.empty()) {
                    std::memcpy(next_bvh_nodes.contents, bvh_nodes.data(),
                                bvh_nodes.size() * sizeof(BvhNode));
                }
                std::memset(next_leaf_infos.contents, 0, leaf_info_bytes);
                if (!leaf_infos.empty()) {
                    std::memcpy(next_leaf_infos.contents, leaf_infos.data(),
                                leaf_infos.size() * sizeof(MeshLeafInfo));
                }
                std::memset(next_bvh_leaves.contents, 0, bvh_leaf_bytes);
                if (!bvh_leaves.empty()) {
                    std::memcpy(next_bvh_leaves.contents, bvh_leaves.data(),
                                bvh_leaves.size() * sizeof(std::uint32_t));
                }
                std::memset(next_solid_planes.contents, 0, plane_bytes);
                if (!solid_planes.empty()) {
                    std::memcpy(next_solid_planes.contents,
                                solid_planes.data(),
                                solid_planes.size() * sizeof(CollisionPlane));
                }

                [residency_set addAllocation:next_vertices];
                [residency_set addAllocation:next_indices];
                [residency_set addAllocation:next_infos];
                [residency_set addAllocation:next_bvh_nodes];
                [residency_set addAllocation:next_leaf_infos];
                [residency_set addAllocation:next_bvh_leaves];
                [residency_set addAllocation:next_solid_planes];
                [rigid_argument_table setAddress:next_vertices.gpuAddress
                                          atIndex:5];
                [rigid_argument_table setAddress:next_indices.gpuAddress
                                          atIndex:6];
                [rigid_argument_table setAddress:next_infos.gpuAddress
                                          atIndex:7];
                [rigid_argument_table setAddress:next_bvh_nodes.gpuAddress
                                          atIndex:13];
                [rigid_argument_table setAddress:next_leaf_infos.gpuAddress
                                          atIndex:23];
                [rigid_argument_table setAddress:next_bvh_leaves.gpuAddress
                                          atIndex:24];
                if (mesh_vertices != nil)
                    [residency_set removeAllocation:mesh_vertices];
                if (mesh_indices != nil)
                    [residency_set removeAllocation:mesh_indices];
                if (mesh_infos != nil)
                    [residency_set removeAllocation:mesh_infos];
                if (mesh_bvh_nodes != nil)
                    [residency_set removeAllocation:mesh_bvh_nodes];
                if (mesh_leaf_infos != nil)
                    [residency_set removeAllocation:mesh_leaf_infos];
                if (mesh_bvh_leaves != nil)
                    [residency_set removeAllocation:mesh_bvh_leaves];
                if (mesh_solid_planes != nil)
                    [residency_set removeAllocation:mesh_solid_planes];
                [residency_set commit];
                mesh_vertices = next_vertices;
                mesh_indices = next_indices;
                mesh_infos = next_infos;
                mesh_bvh_nodes = next_bvh_nodes;
                mesh_leaf_infos = next_leaf_infos;
                mesh_bvh_leaves = next_bvh_leaves;
                mesh_solid_planes = next_solid_planes;
                return success();
            } catch (const std::bad_alloc &) {
                return out_of_memory("Could not flatten Metal rigid meshes");
            } catch (...) {
                return {StatusCode::internal_error, 0,
                        "Unexpected Metal rigid mesh rebuild failure"};
            }
        }
    }
};

FrameToken::FrameToken() noexcept {
    try {
        auto implementation = std::make_unique<Impl>();
        implementation->completion = std::make_shared<CompletionState>();
        implementation->completion->completed = true;
        implementation->commit_options = [[MTL4CommitOptions alloc] init];
        if (implementation->commit_options == nil) return;
        impl_ = std::move(implementation);
    } catch (...) {
        impl_.reset();
    }
}
FrameToken::~FrameToken() = default;
FrameToken::FrameToken(FrameToken &&) noexcept = default;
FrameToken &FrameToken::operator=(FrameToken &&) noexcept = default;

bool FrameToken::pending() const noexcept {
    return impl_ != nullptr && !completion_ready(impl_->completion);
}

bool FrameToken::ready() const noexcept {
    return impl_ == nullptr || completion_ready(impl_->completion);
}

Status FrameToken::wait() noexcept {
    return impl_ == nullptr ? success() : wait_for_completion(impl_->completion);
}

World::World() noexcept = default;
World::~World() = default;
World::World(World &&) noexcept = default;
World &World::operator=(World &&) noexcept = default;

Status World::create(WorldOptions options, World &output) noexcept {
    return create(options, {}, output);
}

Status World::create(WorldOptions options, NativeContext context,
                     World &output) noexcept {
    if (options.rigid_body_capacity == 0 ||
        options.triangle_mesh_capacity == 0 ||
        options.physics_debug.frame_capacity > 3'600U ||
        options.physics_debug.frame_stride == 0U) {
        return invalid_argument(
            "Metal World capacities or physics-debug options are invalid");
    }

    @autoreleasepool {
        try {
            auto impl = std::make_unique<Impl>();
            impl->options = options;
            impl->rigid_slots.resize(options.rigid_body_capacity);
            impl->rigid_meshes.resize(options.rigid_body_capacity);
            impl->pending_impulses.resize(options.rigid_body_capacity);
            impl->pending_angular_impulses.resize(options.rigid_body_capacity);
            impl->meshes.resize(options.triangle_mesh_capacity);
            impl->rigid_constraint_options.resize(
                options.rigid_constraint_capacity);
            impl->debug_input_forces.resize(options.rigid_body_capacity);
            impl->debug_input_torques.resize(options.rigid_body_capacity);
            impl->debug_frames.resize(options.physics_debug.frame_capacity);
            for (PhysicsDebugFrame &frame : impl->debug_frames) {
                frame.rigid_bodies.reserve(options.rigid_body_capacity);
                frame.rigid_contacts.reserve(options.contact_capacity);
                frame.fluid_contacts.reserve(options.contact_capacity);
            }

            if (context.command_queue != nullptr) {
                impl->command_queue =
                    (__bridge id<MTL4CommandQueue>)context.command_queue;
            }
            if (context.device != nullptr) {
                impl->device = (__bridge id<MTLDevice>)context.device;
            } else if (impl->command_queue != nil) {
                impl->device = impl->command_queue.device;
            } else {
                impl->device = MTLCreateSystemDefaultDevice();
            }
            if (impl->device == nil) {
                return metal_failure(nil, "No Metal device is available");
            }
            if (impl->command_queue != nil &&
                impl->command_queue.device != impl->device) {
                return invalid_argument(
                    "NativeContext device and Metal 4 command queue do not match");
            }
            if (impl->command_queue == nil) {
                impl->command_queue = [impl->device newMTL4CommandQueue];
            }
            if (impl->command_queue == nil) {
                return metal_failure(nil, "Could not create a Metal 4 command queue");
            }

            impl->command_allocator = [impl->device newCommandAllocator];
            impl->command_buffer = [impl->device newCommandBuffer];
            impl->completion_event = [impl->device newSharedEvent];
            if (impl->command_allocator == nil || impl->command_buffer == nil ||
                impl->completion_event == nil) {
                return metal_failure(nil, "Could not allocate Metal 4 command objects");
            }

            dispatch_data_t metallib_data = dispatch_data_create(
                parallel_mater_metallib_data,
                static_cast<std::size_t>(parallel_mater_metallib_data_len),
                nullptr,
                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *error = nil;
            impl->library =
                [impl->device newLibraryWithData:metallib_data error:&error];
            if (impl->library == nil) {
                return metal_failure(error, "Could not load embedded Metal shaders");
            }

            id<MTLFunction> noop_function =
                [impl->library newFunctionWithName:@"pm_noop"];
            if (noop_function == nil) {
                return metal_failure(nil,
                                     "Embedded Metal no-op kernel is missing");
            }
            impl->noop_pipeline =
                [impl->device newComputePipelineStateWithFunction:noop_function
                                                             error:&error];
            if (impl->noop_pipeline == nil) {
                return metal_failure(error,
                                     "Could not create Metal foundation pipeline");
            }

            id<MTLFunction> rigid_integrate_function =
                [impl->library newFunctionWithName:@"pm_rigid_integrate"];
            id<MTLFunction> rigid_compound_function =
                [impl->library newFunctionWithName:@"pm_build_rigid_compounds"];
            id<MTLFunction> rigid_world_bounds_function = [impl->library
                newFunctionWithName:@"pm_rigid_world_bounds"];
            id<MTLFunction> rigid_pair_filter_function = [impl->library
                newFunctionWithName:@"pm_rigid_pair_filter"];
            id<MTLFunction> rigid_pair_count_rows_function = [impl->library
                newFunctionWithName:@"pm_rigid_pair_count_rows"];
            id<MTLFunction> rigid_pair_prefix_rows_function = [impl->library
                newFunctionWithName:@"pm_rigid_pair_prefix_rows"];
            id<MTLFunction> rigid_pair_scatter_rows_function = [impl->library
                newFunctionWithName:@"pm_rigid_pair_scatter_rows"];
            id<MTLFunction> rigid_advance_substep_function = [impl->library
                newFunctionWithName:@"pm_rigid_advance_substep"];
            id<MTLFunction> rigid_contact_generate_function = [impl->library
                newFunctionWithName:@"pm_rigid_contact_generate"];
            id<MTLFunction> rigid_contact_reduce_function = [impl->library
                newFunctionWithName:@"pm_rigid_contact_reduce"];
            id<MTLFunction> rigid_clear_function = [impl->library
                newFunctionWithName:@"pm_rigid_clear_accumulators"];
            id<MTLFunction> rigid_constraint_function = [impl->library
                newFunctionWithName:@"pm_rigid_constraints_serial"];
            if (rigid_integrate_function == nil ||
                rigid_compound_function == nil ||
                rigid_world_bounds_function == nil ||
                rigid_pair_filter_function == nil ||
                rigid_pair_count_rows_function == nil ||
                rigid_pair_prefix_rows_function == nil ||
                rigid_pair_scatter_rows_function == nil ||
                rigid_advance_substep_function == nil ||
                rigid_contact_generate_function == nil ||
                rigid_contact_reduce_function == nil ||
                rigid_clear_function == nil ||
                rigid_constraint_function == nil) {
                return metal_failure(nil,
                                     "Embedded Metal rigid kernels are missing");
            }
            impl->rigid_integrate_pipeline = [impl->device
                newComputePipelineStateWithFunction:rigid_integrate_function
                                             error:&error];
            if (impl->rigid_integrate_pipeline == nil) {
                return metal_failure(error,
                                     "Could not create rigid integration pipeline");
            }
            impl->rigid_compound_pipeline = [impl->device
                newComputePipelineStateWithFunction:rigid_compound_function
                                             error:&error];
            if (impl->rigid_compound_pipeline == nil) {
                return metal_failure(
                    error, "Could not create rigid compound pipeline");
            }
            impl->rigid_world_bounds_pipeline = [impl->device
                newComputePipelineStateWithFunction:rigid_world_bounds_function
                                             error:&error];
            impl->rigid_pair_filter_pipeline = [impl->device
                newComputePipelineStateWithFunction:rigid_pair_filter_function
                                             error:&error];
            impl->rigid_pair_count_rows_pipeline = [impl->device
                newComputePipelineStateWithFunction:
                    rigid_pair_count_rows_function
                                             error:&error];
            impl->rigid_pair_prefix_rows_pipeline = [impl->device
                newComputePipelineStateWithFunction:
                    rigid_pair_prefix_rows_function
                                             error:&error];
            impl->rigid_pair_scatter_rows_pipeline = [impl->device
                newComputePipelineStateWithFunction:
                    rigid_pair_scatter_rows_function
                                             error:&error];
            if (impl->rigid_world_bounds_pipeline == nil ||
                impl->rigid_pair_filter_pipeline == nil ||
                impl->rigid_pair_count_rows_pipeline == nil ||
                impl->rigid_pair_prefix_rows_pipeline == nil ||
                impl->rigid_pair_scatter_rows_pipeline == nil) {
                return metal_failure(
                    error, "Could not create rigid broad-phase pipelines");
            }
            impl->rigid_advance_substep_pipeline = [impl->device
                newComputePipelineStateWithFunction:
                    rigid_advance_substep_function
                                             error:&error];
            if (impl->rigid_advance_substep_pipeline == nil) {
                return metal_failure(
                    error, "Could not create rigid substep pipeline");
            }
            impl->rigid_constraint_pipeline = [impl->device
                newComputePipelineStateWithFunction:rigid_constraint_function
                                             error:&error];
            if (impl->rigid_constraint_pipeline == nil) {
                return metal_failure(
                    error, "Could not create rigid constraint pipeline");
            }
            impl->rigid_contact_generate_pipeline = [impl->device
                newComputePipelineStateWithFunction:
                    rigid_contact_generate_function
                                             error:&error];
            if (impl->rigid_contact_generate_pipeline == nil) {
                return metal_failure(error,
                    "Could not create rigid contact generation pipeline");
            }
            impl->rigid_contact_reduce_pipeline = [impl->device
                newComputePipelineStateWithFunction:rigid_contact_reduce_function
                                             error:&error];
            if (impl->rigid_contact_reduce_pipeline == nil) {
                return metal_failure(error,
                    "Could not create rigid contact reduction pipeline");
            }
            impl->rigid_clear_pipeline = [impl->device
                newComputePipelineStateWithFunction:rigid_clear_function
                                             error:&error];
            if (impl->rigid_clear_pipeline == nil) {
                return metal_failure(error,
                                     "Could not create rigid clear pipeline");
            }

            MTL4CounterHeapDescriptor *counter_descriptor =
                [[MTL4CounterHeapDescriptor alloc] init];
            counter_descriptor.type = MTL4CounterHeapTypeTimestamp;
            counter_descriptor.count = metal_timestamp_count;
            impl->timestamp_heap = [impl->device
                newCounterHeapWithDescriptor:counter_descriptor
                                         error:&error];
            if (impl->timestamp_heap == nil) {
                return metal_failure(error,
                                     "Could not create Metal timestamp heap");
            }

            const NSUInteger body_capacity = options.rigid_body_capacity;
            const MTLResourceOptions shared = MTLResourceStorageModeShared;
            impl->rigid_ids = [impl->device
                newBufferWithLength:body_capacity * sizeof(RigidBodyId)
                           options:shared];
            impl->rigid_states = [impl->device
                newBufferWithLength:body_capacity * sizeof(RigidBodyState)
                           options:shared];
            impl->rigid_previous_states = [impl->device
                newBufferWithLength:body_capacity * sizeof(RigidBodyState)
                           options:shared];
            impl->rigid_frame_states = [impl->device
                newBufferWithLength:body_capacity * sizeof(RigidBodyState)
                           options:shared];
            impl->rigid_parameters = [impl->device
                newBufferWithLength:body_capacity * sizeof(RigidParameters)
                           options:shared];
            impl->rigid_forces = [impl->device
                newBufferWithLength:body_capacity * sizeof(Vec3)
                           options:shared];
            impl->rigid_torques = [impl->device
                newBufferWithLength:body_capacity * sizeof(Vec3)
                           options:shared];
            impl->step_constants =
                [impl->device newBufferWithLength:sizeof(StepConstants)
                                           options:shared];
            impl->mesh_vertices =
                [impl->device newBufferWithLength:sizeof(Vec3) options:shared];
            impl->mesh_indices = [impl->device
                newBufferWithLength:sizeof(std::uint32_t) options:shared];
            impl->mesh_infos = [impl->device
                newBufferWithLength:std::max<NSUInteger>(
                                        sizeof(TriangleMeshInfo),
                                        options.triangle_mesh_capacity *
                                            sizeof(TriangleMeshInfo))
                           options:shared];
            impl->mesh_bvh_nodes = [impl->device
                newBufferWithLength:sizeof(BvhNode) options:shared];
            impl->mesh_leaf_infos = [impl->device
                newBufferWithLength:sizeof(MeshLeafInfo) options:shared];
            impl->mesh_bvh_leaves = [impl->device
                newBufferWithLength:sizeof(std::uint32_t) options:shared];
            impl->mesh_solid_planes = [impl->device
                newBufferWithLength:sizeof(CollisionPlane) options:shared];
            const std::uint64_t record_count =
                static_cast<std::uint64_t>(body_capacity) * body_capacity;
            if (record_count >
                std::numeric_limits<NSUInteger>::max() /
                    sizeof(ContactManifold)) {
                return capacity_exceeded(
                    "Metal rigid contact scratch capacity is too large");
            }
            impl->contact_records = [impl->device
                newBufferWithLength:static_cast<NSUInteger>(record_count) *
                                    sizeof(ContactManifold)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_color_owners = [impl->device
                newBufferWithLength:body_capacity * sizeof(std::uint32_t)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_world_bounds = [impl->device
                newBufferWithLength:body_capacity * sizeof(WorldAabb)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_pair_flags = [impl->device
                newBufferWithLength:static_cast<NSUInteger>(record_count) *
                                    sizeof(std::uint32_t)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_active_pairs = [impl->device
                newBufferWithLength:static_cast<NSUInteger>(record_count) *
                                    sizeof(std::uint32_t)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_active_pair_count = [impl->device
                newBufferWithLength:sizeof(std::uint32_t)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_pair_row_offsets = [impl->device
                newBufferWithLength:(body_capacity + 1U) *
                                    sizeof(std::uint32_t)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_substep_index = [impl->device
                newBufferWithLength:sizeof(std::uint32_t)
                           options:shared];
            impl->rigid_constraints = [impl->device
                newBufferWithLength:std::max<NSUInteger>(
                                        sizeof(RigidConstraintResource),
                                        static_cast<NSUInteger>(
                                            options.rigid_constraint_capacity) *
                                            sizeof(RigidConstraintResource))
                           options:shared];
            impl->rigid_constraint_geometry = [impl->device
                newBufferWithLength:std::max<NSUInteger>(
                                        sizeof(RigidConstraintGeometry),
                                        static_cast<NSUInteger>(
                                            options.rigid_constraint_capacity) *
                                            sizeof(RigidConstraintGeometry))
                           options:MTLResourceStorageModePrivate];
            impl->rigid_compounds = [impl->device
                newBufferWithLength:body_capacity * sizeof(RigidCompound)
                           options:MTLResourceStorageModePrivate];
            impl->rigid_contact_events = [impl->device
                newBufferWithLength:std::max<NSUInteger>(
                                        sizeof(RigidContactEvent),
                                        static_cast<NSUInteger>(
                                            options.contact_capacity) *
                                            sizeof(RigidContactEvent))
                           options:shared];
            impl->rigid_contact_count_buffer = [impl->device
                newBufferWithLength:sizeof(std::uint32_t) options:shared];
            if (impl->rigid_ids == nil || impl->rigid_states == nil ||
                impl->rigid_previous_states == nil ||
                impl->rigid_frame_states == nil ||
                impl->rigid_parameters == nil || impl->rigid_forces == nil ||
                impl->rigid_torques == nil || impl->step_constants == nil ||
                impl->mesh_vertices == nil || impl->mesh_indices == nil ||
                impl->mesh_infos == nil || impl->mesh_bvh_nodes == nil ||
                impl->mesh_leaf_infos == nil ||
                impl->mesh_bvh_leaves == nil ||
                impl->mesh_solid_planes == nil ||
                impl->contact_records == nil ||
                impl->rigid_color_owners == nil ||
                impl->rigid_world_bounds == nil ||
                impl->rigid_pair_flags == nil ||
                impl->rigid_active_pairs == nil ||
                impl->rigid_active_pair_count == nil ||
                impl->rigid_pair_row_offsets == nil ||
                impl->rigid_substep_index == nil ||
                impl->rigid_constraint_geometry == nil ||
                impl->rigid_compounds == nil) {
                return metal_failure(nil,
                                     "Could not allocate fixed rigid buffers");
            }
            if (impl->rigid_contact_events == nil ||
                impl->rigid_contact_count_buffer == nil) {
                return metal_failure(
                    nil, "Could not allocate rigid contact event buffers");
            }
            std::memset(impl->rigid_ids.contents, 0, impl->rigid_ids.length);
            std::memset(impl->rigid_states.contents, 0,
                        impl->rigid_states.length);
            std::memset(impl->rigid_previous_states.contents, 0,
                        impl->rigid_previous_states.length);
            std::memset(impl->rigid_frame_states.contents, 0,
                        impl->rigid_frame_states.length);
            std::memset(impl->rigid_parameters.contents, 0,
                        impl->rigid_parameters.length);
            std::memset(impl->rigid_forces.contents, 0,
                        impl->rigid_forces.length);
            std::memset(impl->rigid_torques.contents, 0,
                        impl->rigid_torques.length);
            std::memset(impl->mesh_vertices.contents, 0,
                        impl->mesh_vertices.length);
            std::memset(impl->mesh_indices.contents, 0,
                        impl->mesh_indices.length);
            std::memset(impl->mesh_infos.contents, 0,
                        impl->mesh_infos.length);
            std::memset(impl->mesh_bvh_nodes.contents, 0,
                        impl->mesh_bvh_nodes.length);
            std::memset(impl->mesh_leaf_infos.contents, 0,
                        impl->mesh_leaf_infos.length);
            std::memset(impl->mesh_bvh_leaves.contents, 0,
                        impl->mesh_bvh_leaves.length);
            std::memset(impl->mesh_solid_planes.contents, 0,
                        impl->mesh_solid_planes.length);
            std::memset(impl->rigid_substep_index.contents, 0,
                        impl->rigid_substep_index.length);
            if (impl->rigid_constraints == nil) {
                return metal_failure(
                    nil, "Could not allocate fixed rigid constraint buffer");
            }
            std::memset(impl->rigid_contact_events.contents, 0,
                        impl->rigid_contact_events.length);
            std::memset(impl->rigid_contact_count_buffer.contents, 0,
                        impl->rigid_contact_count_buffer.length);
            auto *constraints = static_cast<RigidConstraintResource *>(
                impl->rigid_constraints.contents);
            for (std::uint32_t index = 0;
                 index < options.rigid_constraint_capacity; ++index) {
                constraints[index] = {};
                constraints[index].generation = 1U;
            }

            MTL4ArgumentTableDescriptor *argument_descriptor =
                [[MTL4ArgumentTableDescriptor alloc] init];
            argument_descriptor.maxBufferBindCount = 26;
            argument_descriptor.initializeBindings = YES;
            argument_descriptor.label = @"ParallelMater rigid arguments";
            impl->rigid_argument_table = [impl->device
                newArgumentTableWithDescriptor:argument_descriptor
                                         error:&error];
            if (impl->rigid_argument_table == nil) {
                return metal_failure(error,
                                     "Could not create rigid argument table");
            }
            [impl->rigid_argument_table setAddress:impl->rigid_states.gpuAddress
                                           atIndex:0];
            [impl->rigid_argument_table
                setAddress:impl->rigid_parameters.gpuAddress
                   atIndex:1];
            [impl->rigid_argument_table setAddress:impl->rigid_forces.gpuAddress
                                           atIndex:2];
            [impl->rigid_argument_table setAddress:impl->rigid_torques.gpuAddress
                                           atIndex:3];
            [impl->rigid_argument_table
                setAddress:impl->step_constants.gpuAddress
                   atIndex:4];
            [impl->rigid_argument_table
                setAddress:impl->mesh_vertices.gpuAddress
                   atIndex:5];
            [impl->rigid_argument_table
                setAddress:impl->mesh_indices.gpuAddress
                   atIndex:6];
            [impl->rigid_argument_table setAddress:impl->mesh_infos.gpuAddress
                                           atIndex:7];
            [impl->rigid_argument_table
                setAddress:impl->contact_records.gpuAddress
                   atIndex:8];
            [impl->rigid_argument_table
                setAddress:impl->rigid_constraints.gpuAddress
                   atIndex:9];
            [impl->rigid_argument_table setAddress:impl->rigid_ids.gpuAddress
                                           atIndex:10];
            [impl->rigid_argument_table
                setAddress:impl->rigid_contact_events.gpuAddress
                   atIndex:11];
            [impl->rigid_argument_table
                setAddress:impl->rigid_contact_count_buffer.gpuAddress
                   atIndex:12];
            [impl->rigid_argument_table
                setAddress:impl->mesh_bvh_nodes.gpuAddress
                   atIndex:13];
            [impl->rigid_argument_table
                setAddress:impl->rigid_previous_states.gpuAddress
                   atIndex:14];
            [impl->rigid_argument_table
                setAddress:impl->rigid_color_owners.gpuAddress
                   atIndex:15];
            [impl->rigid_argument_table
                setAddress:impl->rigid_world_bounds.gpuAddress
                   atIndex:16];
            [impl->rigid_argument_table
                setAddress:impl->rigid_pair_flags.gpuAddress
                   atIndex:17];
            [impl->rigid_argument_table
                setAddress:impl->rigid_active_pairs.gpuAddress
                   atIndex:18];
            [impl->rigid_argument_table
                setAddress:impl->rigid_active_pair_count.gpuAddress
                   atIndex:19];
            [impl->rigid_argument_table
                setAddress:impl->rigid_substep_index.gpuAddress
                   atIndex:20];
            [impl->rigid_argument_table
                setAddress:impl->rigid_pair_row_offsets.gpuAddress
                   atIndex:21];
            [impl->rigid_argument_table
                setAddress:impl->rigid_constraint_geometry.gpuAddress
                   atIndex:22];
            [impl->rigid_argument_table
                setAddress:impl->mesh_leaf_infos.gpuAddress
                   atIndex:23];
            [impl->rigid_argument_table
                setAddress:impl->mesh_bvh_leaves.gpuAddress
                   atIndex:24];
            [impl->rigid_argument_table
                setAddress:impl->rigid_compounds.gpuAddress
                   atIndex:25];

            MTLResidencySetDescriptor *residency_descriptor =
                [[MTLResidencySetDescriptor alloc] init];
            residency_descriptor.label = @"ParallelMater fixed resources";
            residency_descriptor.initialCapacity = 28;
            impl->residency_set = [impl->device
                newResidencySetWithDescriptor:residency_descriptor
                                          error:&error];
            if (impl->residency_set == nil) {
                return metal_failure(error,
                                     "Could not create Metal residency set");
            }
            const id<MTLAllocation> allocations[] = {
                impl->rigid_ids,        impl->rigid_states,
                impl->rigid_previous_states,
                impl->rigid_frame_states,
                impl->rigid_parameters, impl->rigid_forces,
                impl->rigid_torques,    impl->step_constants,
                impl->mesh_vertices,    impl->mesh_indices,
                impl->mesh_infos,       impl->contact_records,
                impl->rigid_color_owners,
                impl->rigid_world_bounds, impl->rigid_pair_flags,
                impl->rigid_active_pairs, impl->rigid_active_pair_count,
                impl->rigid_pair_row_offsets,
                impl->rigid_substep_index,
                impl->rigid_constraints, impl->rigid_constraint_geometry,
                impl->rigid_contact_events,
                impl->rigid_contact_count_buffer, impl->mesh_bvh_nodes,
                impl->mesh_solid_planes, impl->mesh_leaf_infos,
                impl->mesh_bvh_leaves, impl->rigid_compounds};
            [impl->residency_set addAllocations:allocations count:28];
            [impl->residency_set commit];

            const Status system_status = detail::MetalSystems::create(
                options, (__bridge void *)impl->device,
                (__bridge void *)impl->library,
                (__bridge void *)impl->residency_set, impl->systems);
            if (!system_status) return system_status;

            impl->command_buffer.label = @"ParallelMater reusable frame";
            output.impl_ = std::move(impl);
            return success();
        } catch (const std::bad_alloc &) {
            return out_of_memory("Could not allocate Metal World state");
        } catch (...) {
            return {StatusCode::internal_error, 0,
                    "Unexpected failure while creating Metal World"};
        }
    }
}

Status World::add_triangle_mesh(
    BufferSpan<const Vec3> vertices,
    BufferSpan<const std::uint32_t> triangle_indices,
    TriangleMeshId &output) noexcept {
    output = {};
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Triangle meshes cannot change while a frame is in flight");
    }
    if (vertices.buffer == nullptr || triangle_indices.buffer == nullptr ||
        vertices.size < 3 || triangle_indices.size < 3 ||
        triangle_indices.size % 3 != 0 ||
        vertices.byte_offset % alignof(Vec3) != 0 ||
        triangle_indices.byte_offset % alignof(std::uint32_t) != 0 ||
        vertices.size > std::numeric_limits<NSUInteger>::max() / sizeof(Vec3) ||
        triangle_indices.size >
            std::numeric_limits<NSUInteger>::max() / sizeof(std::uint32_t)) {
        return invalid_argument("Metal triangle mesh spans are invalid");
    }

    id<MTLBuffer> vertex_buffer = (__bridge id<MTLBuffer>)vertices.buffer;
    id<MTLBuffer> index_buffer =
        (__bridge id<MTLBuffer>)triangle_indices.buffer;
    const NSUInteger vertex_bytes =
        static_cast<NSUInteger>(vertices.size * sizeof(Vec3));
    const NSUInteger index_bytes = static_cast<NSUInteger>(
        triangle_indices.size * sizeof(std::uint32_t));
    if (vertex_buffer.device != impl_->device ||
        index_buffer.device != impl_->device ||
        vertices.byte_offset > vertex_buffer.length ||
        vertex_bytes > vertex_buffer.length - vertices.byte_offset ||
        triangle_indices.byte_offset > index_buffer.length ||
        index_bytes > index_buffer.length - triangle_indices.byte_offset) {
        return invalid_argument(
            "Metal triangle mesh buffers have the wrong device or range");
    }

    const auto *vertex_contents =
        vertex_buffer.storageMode == MTLStorageModePrivate
            ? nullptr
            : static_cast<const std::byte *>(vertex_buffer.contents);
    const auto *index_contents =
        index_buffer.storageMode == MTLStorageModePrivate
            ? nullptr
            : static_cast<const std::byte *>(index_buffer.contents);
    if (vertex_contents != nullptr && index_contents != nullptr) {
        return add_triangle_mesh(
            {reinterpret_cast<const Vec3 *>(vertex_contents +
                                            vertices.byte_offset),
             vertices.size},
            {reinterpret_cast<const std::uint32_t *>(
                 index_contents + triangle_indices.byte_offset),
             triangle_indices.size},
            output);
    }

    @autoreleasepool {
        try {
            id<MTLBuffer> staged_vertices =
                [impl_->device newBufferWithLength:vertex_bytes
                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> staged_indices =
                [impl_->device newBufferWithLength:index_bytes
                                           options:MTLResourceStorageModeShared];
            if (staged_vertices == nil || staged_indices == nil) {
                return metal_failure(nil,
                                     "Could not allocate mesh staging buffers");
            }

            {
                std::lock_guard submission_lock(impl_->submission_mutex);
                if (!impl_->idle()) {
                    return busy(
                        "Triangle meshes cannot change while a frame is in flight");
                }
                const bool add_vertex =
                    ![impl_->residency_set containsAllocation:vertex_buffer];
                const bool add_index =
                    ![impl_->residency_set containsAllocation:index_buffer];
                if (add_vertex) {
                    [impl_->residency_set addAllocation:vertex_buffer];
                }
                if (add_index) {
                    [impl_->residency_set addAllocation:index_buffer];
                }
                [impl_->residency_set addAllocation:staged_vertices];
                [impl_->residency_set addAllocation:staged_indices];
                [impl_->residency_set commit];

                [impl_->command_allocator reset];
                [impl_->command_buffer
                    beginCommandBufferWithAllocator:impl_->command_allocator];
                [impl_->command_buffer useResidencySet:impl_->residency_set];
                id<MTL4ComputeCommandEncoder> encoder =
                    [impl_->command_buffer computeCommandEncoder];
                if (encoder == nil) {
                    [impl_->command_buffer endCommandBuffer];
                    [impl_->residency_set removeAllocation:staged_vertices];
                    [impl_->residency_set removeAllocation:staged_indices];
                    if (add_vertex) {
                        [impl_->residency_set removeAllocation:vertex_buffer];
                    }
                    if (add_index) {
                        [impl_->residency_set removeAllocation:index_buffer];
                    }
                    [impl_->residency_set commit];
                    return metal_failure(
                        nil, "Could not create mesh upload encoder");
                }
                [encoder copyFromBuffer:vertex_buffer
                           sourceOffset:vertices.byte_offset
                               toBuffer:staged_vertices
                      destinationOffset:0
                                   size:vertex_bytes];
                [encoder copyFromBuffer:index_buffer
                           sourceOffset:triangle_indices.byte_offset
                               toBuffer:staged_indices
                      destinationOffset:0
                                   size:index_bytes];
                [encoder endEncoding];
                [impl_->command_buffer endCommandBuffer];

                auto state = std::make_shared<CompletionState>();
                MTL4CommitOptions *commit_options =
                    [[MTL4CommitOptions alloc] init];
                [commit_options
                    addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
                      {
                          std::lock_guard state_lock(state->mutex);
                          state->error_code =
                              feedback.error == nil
                                  ? 0
                                  : static_cast<std::int64_t>(
                                        feedback.error.code);
                          state->completed = true;
                      }
                      state->completed_condition.notify_all();
                    }];
                const id<MTL4CommandBuffer> command_buffers[] = {
                    impl_->command_buffer};
                [impl_->command_queue commit:command_buffers
                                       count:1
                                     options:commit_options];
                const Status copy_status = wait_for_completion(state);

                [impl_->residency_set removeAllocation:staged_vertices];
                [impl_->residency_set removeAllocation:staged_indices];
                if (add_vertex) {
                    [impl_->residency_set removeAllocation:vertex_buffer];
                }
                if (add_index) {
                    [impl_->residency_set removeAllocation:index_buffer];
                }
                [impl_->residency_set commit];
                if (!copy_status) {
                    return copy_status;
                }
            }

            return add_triangle_mesh(
                {static_cast<const Vec3 *>(staged_vertices.contents),
                 vertices.size},
                {static_cast<const std::uint32_t *>(staged_indices.contents),
                 triangle_indices.size},
                output);
        } catch (const std::bad_alloc &) {
            return out_of_memory("Could not allocate mesh upload state");
        } catch (...) {
            return {StatusCode::internal_error, 0,
                    "Unexpected Metal mesh upload failure"};
        }
    }
}

Status World::add_triangle_mesh(
    HostSpan<const Vec3> vertices,
    HostSpan<const std::uint32_t> triangle_indices,
    TriangleMeshId &output) noexcept {
    output = {};
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Triangle meshes cannot change while a frame is in flight");
    }
    if (vertices.data == nullptr || triangle_indices.data == nullptr ||
        vertices.size < 3 || triangle_indices.size < 3 ||
        triangle_indices.size % 3 != 0) {
        return invalid_argument(
            "Triangle mesh requires vertices and complete triangle indices");
    }
    if (impl_->triangle_mesh_count >= impl_->options.triangle_mesh_capacity) {
        return capacity_exceeded("Triangle mesh capacity is exhausted");
    }

    std::uint32_t slot_index = impl_->options.triangle_mesh_capacity;
    for (std::uint32_t index = 0; index < impl_->meshes.size(); ++index) {
        if (!impl_->meshes[index].alive) {
            slot_index = index;
            break;
        }
    }
    if (slot_index == impl_->options.triangle_mesh_capacity) {
        return {StatusCode::internal_error, 0,
                "No free triangle mesh slot was found"};
    }

    for (std::uint64_t index = 0; index < vertices.size; ++index) {
        if (!finite(vertices.data[index])) {
            return invalid_argument("Triangle mesh contains a non-finite vertex");
        }
    }
    for (std::uint64_t index = 0; index < triangle_indices.size; index += 3) {
        const std::uint32_t first = triangle_indices.data[index];
        const std::uint32_t second = triangle_indices.data[index + 1];
        const std::uint32_t third = triangle_indices.data[index + 2];
        if (first >= vertices.size || second >= vertices.size ||
            third >= vertices.size) {
            return invalid_argument(
                "Triangle mesh index is outside the vertex span");
        }
        const Vec3 area = cross(subtract(vertices.data[second],
                                         vertices.data[first]),
                                subtract(vertices.data[third],
                                         vertices.data[first]));
        if (length_squared(area) <= 1.0e-12F) {
            return invalid_argument(
                "Triangle mesh contains a degenerate triangle");
        }
    }

    try {
        TriangleMeshStorage &mesh = impl_->meshes[slot_index];
        mesh.vertices.assign(vertices.data, vertices.data + vertices.size);
        mesh.indices.assign(triangle_indices.data,
                            triangle_indices.data + triangle_indices.size);
        mesh.minimum = mesh.maximum = mesh.vertices.front();
        for (const Vec3 vertex : mesh.vertices) {
            mesh.minimum.x = std::min(mesh.minimum.x, vertex.x);
            mesh.minimum.y = std::min(mesh.minimum.y, vertex.y);
            mesh.minimum.z = std::min(mesh.minimum.z, vertex.z);
            mesh.maximum.x = std::max(mesh.maximum.x, vertex.x);
            mesh.maximum.y = std::max(mesh.maximum.y, vertex.y);
            mesh.maximum.z = std::max(mesh.maximum.z, vertex.z);
        }
        build_mesh_bvh(mesh);
        mesh.solid_planes = closed_convex_planes(mesh.vertices, mesh.indices);
        mesh.alive = true;
        const Status buffer_status = impl_->rebuild_mesh_buffers();
        if (!buffer_status) {
            mesh.vertices.clear();
            mesh.indices.clear();
            mesh.bvh_nodes.clear();
            mesh.solid_planes.clear();
            mesh.alive = false;
            return buffer_status;
        }
        ++impl_->triangle_mesh_count;
        ++impl_->revision;
        output = {slot_index, mesh.generation};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not copy triangle mesh host data");
    } catch (...) {
        return {StatusCode::internal_error, 0,
                "Unexpected triangle mesh upload failure"};
    }
}

Status World::remove_triangle_mesh(TriangleMeshId mesh_id) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Triangle meshes cannot change while a frame is in flight");
    }
    if (!impl_->valid(mesh_id)) {
        return invalid_handle("Triangle mesh handle is stale");
    }
    for (std::uint32_t index = 0; index < impl_->rigid_body_count; ++index) {
        if (impl_->rigid_meshes[index] == mesh_id) {
            return invalid_argument(
                "Triangle mesh is still referenced by a rigid body");
        }
    }
    if (impl_->systems.references_triangle_mesh(mesh_id)) {
        return invalid_argument(
            "Triangle mesh is still referenced by a paint field");
    }
    TriangleMeshStorage &mesh = impl_->meshes[mesh_id.index];
    TriangleMeshStorage previous = std::move(mesh);
    mesh.vertices.clear();
    mesh.indices.clear();
    mesh.bvh_nodes.clear();
    mesh.solid_planes.clear();
    mesh.alive = false;
    const Status buffer_status = impl_->rebuild_mesh_buffers();
    if (!buffer_status) {
        mesh = std::move(previous);
        return buffer_status;
    }
    mesh.generation = previous.generation;
    if (++mesh.generation == 0) {
        mesh.generation = 1;
    }
    --impl_->triangle_mesh_count;
    ++impl_->revision;
    return success();
}

Status World::add_rigid_body(RigidBodyOptions options,
                             RigidBodyId &output) noexcept {
    output = {};
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Rigid bodies cannot change while a frame is in flight");
    }
    if (!impl_->valid(options.mesh)) {
        return invalid_handle("Rigid body triangle mesh handle is stale");
    }
    const float orientation_size =
        options.initial_state.orientation.x * options.initial_state.orientation.x +
        options.initial_state.orientation.y * options.initial_state.orientation.y +
        options.initial_state.orientation.z * options.initial_state.orientation.z +
        options.initial_state.orientation.w * options.initial_state.orientation.w;
    if (!finite(options.initial_state) || !(orientation_size > 1.0e-12F) ||
        !std::isfinite(options.mass) || options.mass <= 0.0F ||
        !finite(options.inertia_diagonal) || options.inertia_diagonal.x < 0.0F ||
        options.inertia_diagonal.y < 0.0F ||
        options.inertia_diagonal.z < 0.0F ||
        !std::isfinite(options.friction) || options.friction < 0.0F ||
        !std::isfinite(options.restitution) || options.restitution < 0.0F ||
        options.restitution > 1.0F ||
        !std::isfinite(options.collision_margin) ||
        options.collision_margin < 0.0F ||
        !std::isfinite(options.linear_damping) || options.linear_damping < 0.0F ||
        !std::isfinite(options.angular_damping) || options.angular_damping < 0.0F ||
        !std::isfinite(options.maximum_linear_speed) ||
        options.maximum_linear_speed <= 0.0F ||
        !std::isfinite(options.maximum_angular_speed) ||
        options.maximum_angular_speed <= 0.0F) {
        return invalid_argument("Rigid body options are invalid");
    }
    if (impl_->rigid_body_count >= impl_->options.rigid_body_capacity) {
        return capacity_exceeded("Rigid body capacity is exhausted");
    }

    std::uint32_t slot_index = impl_->options.rigid_body_capacity;
    for (std::uint32_t index = 0; index < impl_->rigid_slots.size(); ++index) {
        if (!impl_->rigid_slots[index].alive) {
            slot_index = index;
            break;
        }
    }
    if (slot_index == impl_->options.rigid_body_capacity) {
        return {StatusCode::internal_error, 0,
                "No free rigid body handle slot was found"};
    }

    const std::uint32_t dense_index = impl_->rigid_body_count;
    HandleSlot &slot = impl_->rigid_slots[slot_index];
    slot.alive = true;
    slot.dense_index = dense_index;
    if (slot.generation == 0) {
        slot.generation = 1;
    }
    const RigidBodyId id{slot_index, slot.generation};

    auto *ids = static_cast<RigidBodyId *>(impl_->rigid_ids.contents);
    auto *states = static_cast<RigidBodyState *>(impl_->rigid_states.contents);
    auto *parameters =
        static_cast<RigidParameters *>(impl_->rigid_parameters.contents);
    auto *forces = static_cast<Vec3 *>(impl_->rigid_forces.contents);
    auto *torques = static_cast<Vec3 *>(impl_->rigid_torques.contents);

    ids[dense_index] = id;
    states[dense_index] = options.initial_state;
    states[dense_index].orientation =
        normalized(states[dense_index].orientation);
    const TriangleMeshStorage &mesh = impl_->meshes[options.mesh.index];
    const Vec3 half_extent = multiply(subtract(mesh.maximum, mesh.minimum), 0.5F);
    Vec3 inertia = options.inertia_diagonal;
    if (inertia.x == 0.0F && inertia.y == 0.0F && inertia.z == 0.0F) {
        inertia = {
            options.mass * std::max((half_extent.y * half_extent.y +
                                     half_extent.z * half_extent.z) /
                                        3.0F,
                                    1.0e-6F),
            options.mass * std::max((half_extent.x * half_extent.x +
                                     half_extent.z * half_extent.z) /
                                        3.0F,
                                    1.0e-6F),
            options.mass * std::max((half_extent.x * half_extent.x +
                                     half_extent.y * half_extent.y) /
                                        3.0F,
                                    1.0e-6F)};
    } else if (!(inertia.x > 0.0F && inertia.y > 0.0F && inertia.z > 0.0F)) {
        slot.alive = false;
        return invalid_argument(
            "Explicit rigid body inertia must be positive on every axis");
    }
    parameters[dense_index] = {
        static_cast<std::uint32_t>(options.motion),
        options.motion == MotionType::dynamic ? 1.0F / options.mass : 0.0F,
        options.motion == MotionType::dynamic
            ? Vec3{1.0F / inertia.x, 1.0F / inertia.y, 1.0F / inertia.z}
            : Vec3{},
        options.linear_damping,
        options.angular_damping,
        options.maximum_linear_speed,
        options.maximum_angular_speed,
        0U,
        {},
        options.mesh.index,
        options.friction,
        options.restitution,
        options.collision_margin};
    forces[dense_index] = {};
    torques[dense_index] = {};
    impl_->pending_impulses[dense_index] = {};
    impl_->pending_angular_impulses[dense_index] = {};
    impl_->rigid_meshes[dense_index] = options.mesh;
    ++impl_->rigid_body_count;
    ++impl_->revision;
    output = id;
    return success();
}

Status World::remove_rigid_body(RigidBodyId body) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Rigid bodies cannot change while a frame is in flight");
    }
    if (impl_->systems.references_rigid_body(body)) {
        return invalid_argument(
            "Rigid body is referenced by a particle-system coupling");
    }
    std::uint32_t dense_index = 0;
    if (!impl_->valid(body, dense_index)) {
        return invalid_handle("Rigid body handle is stale");
    }
    auto *constraint_resources = static_cast<RigidConstraintResource *>(
        impl_->rigid_constraints.contents);
    for (std::uint32_t index = 0;
         index < impl_->options.rigid_constraint_capacity; ++index) {
        if (constraint_resources[index].alive != 0U) {
            const RigidConstraintOptions &options =
                impl_->rigid_constraint_options[index];
            if (options.body_a == body || options.body_b == body) {
                return invalid_argument(
                    "Rigid body is still referenced by a constraint");
            }
        }
    }

    auto *ids = static_cast<RigidBodyId *>(impl_->rigid_ids.contents);
    auto *states = static_cast<RigidBodyState *>(impl_->rigid_states.contents);
    auto *parameters =
        static_cast<RigidParameters *>(impl_->rigid_parameters.contents);
    auto *forces = static_cast<Vec3 *>(impl_->rigid_forces.contents);
    auto *torques = static_cast<Vec3 *>(impl_->rigid_torques.contents);
    const std::uint32_t last = impl_->rigid_body_count - 1;
    if (dense_index != last) {
        ids[dense_index] = ids[last];
        states[dense_index] = states[last];
        parameters[dense_index] = parameters[last];
        forces[dense_index] = forces[last];
        torques[dense_index] = torques[last];
        impl_->rigid_meshes[dense_index] = impl_->rigid_meshes[last];
        impl_->pending_impulses[dense_index] = impl_->pending_impulses[last];
        impl_->pending_angular_impulses[dense_index] =
            impl_->pending_angular_impulses[last];
        impl_->rigid_slots[ids[dense_index].index].dense_index = dense_index;
        for (std::uint32_t index = 0;
             index < impl_->options.rigid_constraint_capacity; ++index) {
            if (constraint_resources[index].alive == 0U) continue;
            if (constraint_resources[index].body_a == last)
                constraint_resources[index].body_a = dense_index;
            if (constraint_resources[index].body_b == last)
                constraint_resources[index].body_b = dense_index;
        }
    }
    --impl_->rigid_body_count;
    HandleSlot &slot = impl_->rigid_slots[body.index];
    slot.alive = false;
    if (++slot.generation == 0) {
        slot.generation = 1;
    }
    ++impl_->revision;
    return success();
}

Status World::set_rigid_body_state(RigidBodyId body,
                                   RigidBodyState state) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Rigid body state cannot change during a frame");
    }
    const float orientation_size = state.orientation.x * state.orientation.x +
                                   state.orientation.y * state.orientation.y +
                                   state.orientation.z * state.orientation.z +
                                   state.orientation.w * state.orientation.w;
    if (!finite(state) || !(orientation_size > 1.0e-12F)) {
        return invalid_argument("Rigid body state is invalid");
    }
    std::uint32_t dense_index = 0;
    if (!impl_->valid(body, dense_index)) {
        return invalid_handle("Rigid body handle is stale");
    }
    state.orientation = normalized(state.orientation);
    static_cast<RigidBodyState *>(impl_->rigid_states.contents)[dense_index] =
        state;
    static_cast<Vec3 *>(impl_->rigid_forces.contents)[dense_index] = {};
    static_cast<Vec3 *>(impl_->rigid_torques.contents)[dense_index] = {};
    impl_->pending_impulses[dense_index] = {};
    impl_->pending_angular_impulses[dense_index] = {};
    static_cast<RigidParameters *>(impl_->rigid_parameters.contents)
        [dense_index]
            .has_kinematic_target = 0;
    impl_->systems.invalidate_smoke_grid_static_metadata();
    ++impl_->revision;
    return success();
}

Status World::set_kinematic_target(RigidBodyId body,
                                   RigidBodyState target) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Kinematic targets cannot change during a frame");
    }
    const float orientation_size = target.orientation.x * target.orientation.x +
                                   target.orientation.y * target.orientation.y +
                                   target.orientation.z * target.orientation.z +
                                   target.orientation.w * target.orientation.w;
    if (!finite(target) || !(orientation_size > 1.0e-12F)) {
        return invalid_argument("Kinematic target is invalid");
    }
    std::uint32_t dense_index = 0;
    if (!impl_->valid(body, dense_index)) {
        return invalid_handle("Rigid body handle is stale");
    }
    auto *parameters =
        static_cast<RigidParameters *>(impl_->rigid_parameters.contents);
    if (parameters[dense_index].motion !=
        static_cast<std::uint32_t>(MotionType::kinematic)) {
        return invalid_argument("Rigid body is not kinematic");
    }
    target.orientation = normalized(target.orientation);
    parameters[dense_index].kinematic_target = target;
    parameters[dense_index].has_kinematic_target = 1;
    return success();
}

Status World::apply_force(RigidBodyId body, Vec3 force,
                          Vec3 world_point) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Forces cannot change during a frame");
    }
    if (!finite(force) || !finite(world_point)) {
        return invalid_argument("Force and world point must be finite");
    }
    std::uint32_t dense_index = 0;
    if (!impl_->valid(body, dense_index)) {
        return invalid_handle("Rigid body handle is stale");
    }
    auto *states = static_cast<RigidBodyState *>(impl_->rigid_states.contents);
    const auto *parameters =
        static_cast<const RigidParameters *>(impl_->rigid_parameters.contents);
    if (parameters[dense_index].motion !=
        static_cast<std::uint32_t>(MotionType::dynamic)) {
        return invalid_argument("Forces require a dynamic rigid body");
    }
    auto *forces = static_cast<Vec3 *>(impl_->rigid_forces.contents);
    auto *torques = static_cast<Vec3 *>(impl_->rigid_torques.contents);
    forces[dense_index] = add(forces[dense_index], force);
    torques[dense_index] =
        add(torques[dense_index],
            cross(subtract(world_point, states[dense_index].position), force));
    return success();
}

Status World::apply_central_acceleration(
    HostSpan<RigidBodyId> bodies, Vec3 acceleration) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Accelerations cannot change during a frame");
    }
    if (!finite(acceleration) ||
        (bodies.size != 0U && bodies.data == nullptr) ||
        bodies.size > impl_->rigid_body_count) {
        return invalid_argument("Central acceleration batch is invalid");
    }
    const auto *parameters =
        static_cast<const RigidParameters *>(impl_->rigid_parameters.contents);
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense_index = 0U;
        if (!impl_->valid(bodies.data[index], dense_index)) {
            return invalid_handle("Rigid body handle is stale");
        }
        if (parameters[dense_index].motion !=
            static_cast<std::uint32_t>(MotionType::dynamic)) {
            return invalid_argument(
                "Central acceleration requires dynamic bodies");
        }
        for (std::uint64_t prior = 0U; prior < index; ++prior) {
            if (bodies.data[prior] == bodies.data[index]) {
                return invalid_argument(
                    "Central acceleration body is duplicated");
            }
        }
    }
    auto *forces = static_cast<Vec3 *>(impl_->rigid_forces.contents);
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense_index = 0U;
        if (!impl_->valid(bodies.data[index], dense_index)) {
            return invalid_handle("Rigid body handle is stale");
        }
        const float mass = 1.0F / parameters[dense_index].inverse_mass;
        forces[dense_index] = add(
            forces[dense_index], multiply(acceleration, mass));
    }
    return success();
}

Status World::apply_impulse(RigidBodyId body, Vec3 impulse,
                            Vec3 world_point) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Impulses cannot change during a frame");
    }
    if (!finite(impulse) || !finite(world_point)) {
        return invalid_argument("Impulse and world point must be finite");
    }
    std::uint32_t dense_index = 0;
    if (!impl_->valid(body, dense_index)) {
        return invalid_handle("Rigid body handle is stale");
    }
    auto *parameters =
        static_cast<RigidParameters *>(impl_->rigid_parameters.contents);
    if (parameters[dense_index].motion !=
        static_cast<std::uint32_t>(MotionType::dynamic)) {
        return invalid_argument("Impulses require a dynamic rigid body");
    }
    const auto *states =
        static_cast<const RigidBodyState *>(impl_->rigid_states.contents);
    impl_->pending_impulses[dense_index] =
        add(impl_->pending_impulses[dense_index], impulse);
    const Vec3 angular_impulse =
        cross(subtract(world_point, states[dense_index].position), impulse);
    impl_->pending_angular_impulses[dense_index] =
        add(impl_->pending_angular_impulses[dense_index], angular_impulse);
    return success();
}

Status World::rigid_body_view(RigidBodyDeviceView &output) const noexcept {
    output = {};
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Rigid body view requires a completed frame");
    }
    output.ids = {(__bridge void *)impl_->rigid_ids, 0,
                  impl_->rigid_body_count};
    output.states = {(__bridge void *)impl_->rigid_states, 0,
                     impl_->rigid_body_count};
    output.previous_states = {(__bridge void *)impl_->rigid_previous_states, 0,
                              impl_->rigid_body_count};
    if (impl_->options.physics_debug.frame_capacity != 0) {
        output.applied_forces = {(__bridge void *)impl_->rigid_forces, 0,
                                 impl_->rigid_body_count};
        output.applied_torques = {(__bridge void *)impl_->rigid_torques, 0,
                                  impl_->rigid_body_count};
    }
    output.revision = impl_->revision;
    return success();
}

Status World::read_rigid_body_state(RigidBodyId body,
                                    RigidBodyState &output) const noexcept {
    output = {};
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Rigid body readback requires a completed frame");
    }
    std::uint32_t dense_index = 0;
    if (!impl_->valid(body, dense_index)) {
        return invalid_handle("Rigid body handle is stale");
    }
    output = static_cast<RigidBodyState *>(impl_->rigid_states.contents)
        [dense_index];
    return success();
}

Status World::query_hit_box(HitBox box,
                            HitBoxResult &output) const noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle()) {
        return busy("Hit-box query requires a completed frame");
    }
    const float orientation_length_squared =
        box.orientation.x * box.orientation.x +
        box.orientation.y * box.orientation.y +
        box.orientation.z * box.orientation.z +
        box.orientation.w * box.orientation.w;
    if (!finite(box.center) || !finite(box.orientation) ||
        !finite(box.half_extents) || box.half_extents.x <= 0.0F ||
        box.half_extents.y <= 0.0F || box.half_extents.z <= 0.0F ||
        orientation_length_squared <= 1.0e-12F) {
        return invalid_argument(
            "Hit-box transform and half extents are invalid");
    }
    box.orientation = normalized(box.orientation);

    HitBoxResult temporary;
    try {
        temporary.rigid_bodies.reserve(impl_->rigid_body_count);
        const auto *states = static_cast<const RigidBodyState *>(
            impl_->rigid_states.contents);
        const auto *ids = static_cast<const RigidBodyId *>(
            impl_->rigid_ids.contents);
        for (std::uint32_t body = 0U; body < impl_->rigid_body_count; ++body) {
            const TriangleMeshId mesh_id = impl_->rigid_meshes[body];
            if (mesh_id.index >= impl_->meshes.size()) continue;
            const TriangleMeshStorage &mesh = impl_->meshes[mesh_id.index];
            if (!mesh.alive || mesh.generation != mesh_id.generation) continue;
            const RigidBodyState state = states[body];
            bool overlap = false;
            for (std::size_t index = 0U; index < mesh.indices.size();
                 index += 3U) {
                const Vec3 first = hit_box_local_point(
                    box, add(state.position,
                             rotate(state.orientation,
                                    mesh.vertices[mesh.indices[index]])));
                const Vec3 second = hit_box_local_point(
                    box, add(state.position,
                             rotate(state.orientation,
                                    mesh.vertices[mesh.indices[index + 1U]])));
                const Vec3 third = hit_box_local_point(
                    box, add(state.position,
                             rotate(state.orientation,
                                    mesh.vertices[mesh.indices[index + 2U]])));
                if (hit_box_overlaps_triangle(first, second, third,
                                              box.half_extents)) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap && !mesh.solid_planes.empty()) {
                const Vec3 center = inverse_rotate(
                    state.orientation, subtract(box.center, state.position));
                overlap = true;
                for (const CollisionPlane plane : mesh.solid_planes) {
                    if (dot(plane.normal, center) > plane.offset + 1.0e-6F) {
                        overlap = false;
                        break;
                    }
                }
            }
            if (overlap) temporary.rigid_bodies.push_back(ids[body]);
        }
        const Status particle_status =
            impl_->systems.append_hit_box_particles(box, temporary.particles);
        if (!particle_status) return particle_status;
        std::sort(temporary.rigid_bodies.begin(),
                  temporary.rigid_bodies.end(),
                  [](RigidBodyId left, RigidBodyId right) {
                      return left.index < right.index ||
                          (left.index == right.index &&
                           left.generation < right.generation);
                  });
        std::sort(temporary.particles.begin(), temporary.particles.end(),
                  [](HitBoxParticle left, HitBoxParticle right) {
                      if (left.fluid.index != right.fluid.index) {
                          return left.fluid.index < right.fluid.index;
                      }
                      if (left.fluid.generation != right.fluid.generation) {
                          return left.fluid.generation <
                                 right.fluid.generation;
                      }
                      return left.stable_particle_id <
                             right.stable_particle_id;
                  });
    } catch (const std::bad_alloc &) {
        return out_of_memory("Hit-box result allocation failed");
    } catch (...) {
        return out_of_memory("Hit-box result allocation failed");
    }
    output = std::move(temporary);
    return success();
}

Status World::add_rigid_constraint(
    RigidConstraintOptions options, RigidConstraintId &output) noexcept {
    output = {};
    if (impl_ == nullptr) return invalid_argument("World is not initialized");
    if (!impl_->idle())
        return busy("Rigid constraints cannot change during a frame");
    if (!valid_constraint_options(options)) {
        return invalid_argument("Rigid constraint options are invalid");
    }
    std::uint32_t body_a = 0U;
    std::uint32_t body_b = 0U;
    if (!impl_->valid(options.body_a, body_a) ||
        !impl_->valid(options.body_b, body_b)) {
        return invalid_handle("Rigid constraint body handle is stale");
    }
    const auto *parameters = static_cast<const RigidParameters *>(
        impl_->rigid_parameters.contents);
    if (parameters[body_a].inverse_mass == 0.0F &&
        parameters[body_b].inverse_mass == 0.0F) {
        return invalid_argument(
            "Rigid constraint needs at least one dynamic body");
    }
    if (impl_->rigid_constraint_count >=
        impl_->options.rigid_constraint_capacity) {
        return capacity_exceeded("Rigid constraint capacity is exhausted");
    }
    auto *resources = static_cast<RigidConstraintResource *>(
        impl_->rigid_constraints.contents);
    std::uint32_t slot = impl_->options.rigid_constraint_capacity;
    for (std::uint32_t index = 0;
         index < impl_->options.rigid_constraint_capacity; ++index) {
        if (resources[index].alive == 0U) {
            slot = index;
            break;
        }
    }
    if (slot == impl_->options.rigid_constraint_capacity) {
        return {StatusCode::internal_error, 0,
                "No free rigid constraint slot was found"};
    }
    options.local_orientation_a = normalized(options.local_orientation_a);
    options.local_orientation_b = normalized(options.local_orientation_b);
    RigidConstraintResource &resource = resources[slot];
    if (resource.generation == 0U) resource.generation = 1U;
    resource.alive = 1U;
    resource.type = static_cast<std::uint32_t>(options.type);
    resource.body_a = body_a;
    resource.body_b = body_b;
    resource.enabled = options.enabled ? 1U : 0U;
    resource.broken = 0U;
    resource.local_anchor_a = options.local_anchor_a;
    resource.local_anchor_b = options.local_anchor_b;
    resource.local_orientation_a = options.local_orientation_a;
    resource.local_orientation_b = options.local_orientation_b;
    resource.breaking_impulse_threshold =
        options.breaking_impulse_threshold;
    resource.applied_impulse = 0.0F;
    resource.linear_limit_axes = options.linear_limits.axes;
    resource.linear_limit_lower = options.linear_limits.lower;
    resource.linear_limit_upper = options.linear_limits.upper;
    resource.angular_limit_axes = options.angular_limits.axes;
    resource.angular_limit_lower = options.angular_limits.lower;
    resource.angular_limit_upper = options.angular_limits.upper;
    resource.linear_spring_axes = options.linear_springs.axes;
    resource.linear_spring_stiffness = options.linear_springs.stiffness;
    resource.linear_spring_damping = options.linear_springs.damping;
    resource.angular_spring_axes = options.angular_springs.axes;
    resource.angular_spring_stiffness = options.angular_springs.stiffness;
    resource.angular_spring_damping = options.angular_springs.damping;
    resource.linear_motor_enabled = options.motor.linear_enabled ? 1U : 0U;
    resource.angular_motor_enabled = options.motor.angular_enabled ? 1U : 0U;
    resource.linear_target_velocity = options.motor.linear_target_velocity;
    resource.linear_maximum_impulse = options.motor.linear_maximum_impulse;
    resource.angular_target_velocity = options.motor.angular_target_velocity;
    resource.angular_maximum_impulse = options.motor.angular_maximum_impulse;
    resource.solver_iterations = options.solver_iterations;
    resource.disable_collisions = options.disable_collisions ? 1U : 0U;
    impl_->rigid_constraint_options[slot] = options;
    ++impl_->rigid_constraint_count;
    ++impl_->revision;
    output = {slot, resource.generation};
    return success();
}

Status World::update_rigid_constraint(
    RigidConstraintId constraint, RigidConstraintOptions options) noexcept {
    if (impl_ == nullptr) return invalid_argument("World is not initialized");
    if (!impl_->idle())
        return busy("Rigid constraints cannot change during a frame");
    RigidConstraintResource *resource = nullptr;
    if (!impl_->valid(constraint, resource))
        return invalid_handle("Rigid constraint handle is stale");
    if (!valid_constraint_options(options)) {
        return invalid_argument("Rigid constraint options are invalid");
    }
    std::uint32_t body_a = 0U;
    std::uint32_t body_b = 0U;
    if (!impl_->valid(options.body_a, body_a) ||
        !impl_->valid(options.body_b, body_b)) {
        return invalid_handle("Rigid constraint body handle is stale");
    }
    const auto *parameters = static_cast<const RigidParameters *>(
        impl_->rigid_parameters.contents);
    if (parameters[body_a].inverse_mass == 0.0F &&
        parameters[body_b].inverse_mass == 0.0F) {
        return invalid_argument(
            "Rigid constraint needs at least one dynamic body");
    }
    options.local_orientation_a = normalized(options.local_orientation_a);
    options.local_orientation_b = normalized(options.local_orientation_b);
    resource->type = static_cast<std::uint32_t>(options.type);
    resource->body_a = body_a;
    resource->body_b = body_b;
    resource->enabled = options.enabled ? 1U : 0U;
    resource->broken = 0U;
    resource->local_anchor_a = options.local_anchor_a;
    resource->local_anchor_b = options.local_anchor_b;
    resource->local_orientation_a = options.local_orientation_a;
    resource->local_orientation_b = options.local_orientation_b;
    resource->breaking_impulse_threshold = options.breaking_impulse_threshold;
    resource->applied_impulse = 0.0F;
    resource->linear_limit_axes = options.linear_limits.axes;
    resource->linear_limit_lower = options.linear_limits.lower;
    resource->linear_limit_upper = options.linear_limits.upper;
    resource->angular_limit_axes = options.angular_limits.axes;
    resource->angular_limit_lower = options.angular_limits.lower;
    resource->angular_limit_upper = options.angular_limits.upper;
    resource->linear_spring_axes = options.linear_springs.axes;
    resource->linear_spring_stiffness = options.linear_springs.stiffness;
    resource->linear_spring_damping = options.linear_springs.damping;
    resource->angular_spring_axes = options.angular_springs.axes;
    resource->angular_spring_stiffness = options.angular_springs.stiffness;
    resource->angular_spring_damping = options.angular_springs.damping;
    resource->linear_motor_enabled = options.motor.linear_enabled ? 1U : 0U;
    resource->angular_motor_enabled = options.motor.angular_enabled ? 1U : 0U;
    resource->linear_target_velocity = options.motor.linear_target_velocity;
    resource->linear_maximum_impulse = options.motor.linear_maximum_impulse;
    resource->angular_target_velocity = options.motor.angular_target_velocity;
    resource->angular_maximum_impulse = options.motor.angular_maximum_impulse;
    resource->solver_iterations = options.solver_iterations;
    resource->disable_collisions = options.disable_collisions ? 1U : 0U;
    impl_->rigid_constraint_options[constraint.index] = options;
    ++impl_->revision;
    return success();
}

Status World::remove_rigid_constraint(RigidConstraintId constraint) noexcept {
    if (impl_ == nullptr) return invalid_argument("World is not initialized");
    if (!impl_->idle())
        return busy("Rigid constraints cannot change during a frame");
    RigidConstraintResource *resource = nullptr;
    if (!impl_->valid(constraint, resource))
        return invalid_handle("Rigid constraint handle is stale");
    resource->alive = 0U;
    resource->enabled = 0U;
    resource->broken = 0U;
    resource->applied_impulse = 0.0F;
    if (++resource->generation == 0U) resource->generation = 1U;
    --impl_->rigid_constraint_count;
    // Compound data is rebuilt by the GPU while any constraint remains.
    // Removing the last constraint skips that pass, so explicitly discard
    // the derived state before the next broad phase. Otherwise bodies from
    // the former compound remain collision-filtered indefinitely.
    if (impl_->rigid_constraint_count == 0U) {
        std::memset(impl_->rigid_compounds.contents, 0,
                    impl_->rigid_compounds.length);
    }
    ++impl_->revision;
    return success();
}

Status World::read_rigid_constraint_state(
    RigidConstraintId constraint, RigidConstraintState &output) const noexcept {
    output = {};
    if (impl_ == nullptr) return invalid_argument("World is not initialized");
    if (!impl_->idle())
        return busy("Rigid constraint readback requires a completed frame");
    RigidConstraintResource *resource = nullptr;
    if (!impl_->valid(constraint, resource))
        return invalid_handle("Rigid constraint handle is stale");
    output.enabled = resource->enabled != 0U;
    output.broken = resource->broken != 0U;
    output.applied_impulse = resource->applied_impulse;
    return success();
}

Status World::step_async(StepOptions options, FrameToken &completion) noexcept {
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (completion.pending()) {
        return busy("Completion token already represents a pending frame");
    }
    if (completion.impl_ == nullptr ||
        completion.impl_->completion == nullptr ||
        completion.impl_->commit_options == nil) {
        return out_of_memory(
            "Frame token completion resources are unavailable");
    }
    if (!(options.timestep > 0.0F) || !std::isfinite(options.timestep) ||
        options.substeps == 0 || options.substeps > 1'024U ||
        !finite(options.gravity)) {
        return invalid_argument(
            "StepOptions require a finite positive timestep and substeps");
    }
    options.substeps =
        impl_->systems.required_substeps(options.timestep, options.substeps);
    if (options.substeps > 1'024U)
        return invalid_argument(
            "Rope timestep limit requires more than 1024 substeps");

    @autoreleasepool {
        try {
            std::lock_guard lock(impl_->submission_mutex);
            if (!completion_ready(impl_->latest_completion)) {
                return busy("Only one Metal frame may be in flight per World");
            }
            Status debug_status = impl_->record_debug_frame();
            if (!debug_status) return debug_status;

            const bool debug_enabled =
                impl_->options.physics_debug.frame_capacity != 0U;
            impl_->last_step_options = options;
            if (debug_enabled) {
                const auto *forces =
                    static_cast<const Vec3 *>(impl_->rigid_forces.contents);
                const auto *torques =
                    static_cast<const Vec3 *>(impl_->rigid_torques.contents);
                std::copy_n(forces, impl_->rigid_body_count,
                            impl_->debug_input_forces.begin());
                std::copy_n(torques, impl_->rigid_body_count,
                            impl_->debug_input_torques.begin());
            }

            auto *states =
                static_cast<RigidBodyState *>(impl_->rigid_states.contents);
            const auto *parameters = static_cast<const RigidParameters *>(
                impl_->rigid_parameters.contents);
            for (std::uint32_t index = 0; index < impl_->rigid_body_count;
                 ++index) {
                states[index].linear_velocity = add(
                    states[index].linear_velocity,
                    multiply(impl_->pending_impulses[index],
                             parameters[index].inverse_mass));
                const Vec3 angular = impl_->pending_angular_impulses[index];
                states[index].angular_velocity =
                    add(states[index].angular_velocity,
                        inverse_inertia_world(
                            parameters[index], states[index], angular));
                impl_->pending_impulses[index] = {};
                impl_->pending_angular_impulses[index] = {};
            }
            std::memcpy(impl_->rigid_frame_states.contents, states,
                        static_cast<std::size_t>(impl_->rigid_body_count) *
                            sizeof(RigidBodyState));

            impl_->systems.set_rigid_resources(
                (__bridge void *)impl_->rigid_ids,
                (__bridge void *)impl_->rigid_states,
                (__bridge void *)impl_->rigid_previous_states,
                (__bridge void *)impl_->rigid_frame_states,
                (__bridge void *)impl_->rigid_parameters,
                (__bridge void *)impl_->mesh_vertices,
                (__bridge void *)impl_->mesh_indices,
                (__bridge void *)impl_->mesh_infos,
                (__bridge void *)impl_->mesh_solid_planes,
                impl_->rigid_body_count);
            *static_cast<std::uint32_t *>(
                impl_->rigid_contact_count_buffer.contents) = 0U;
            *static_cast<std::uint32_t *>(
                impl_->rigid_substep_index.contents) = 0U;
            const Status systems_status = impl_->systems.begin_frame(
                options.collect_fluid_contacts || debug_enabled,
                options.timestep);
            if (!systems_status) return systems_status;

            detail::MetalTimingContext timing_context{};
            timing_context.counter_heap =
                (__bridge void *)impl_->timestamp_heap;
            timing_context.records = impl_->timing_records.data();
            timing_context.record_capacity = metal_timing_record_capacity;
            timing_context.final_index = metal_final_timestamp;
            timing_context.enabled = options.collect_kernel_timings;

            [impl_->command_allocator reset];
            [impl_->command_buffer
                beginCommandBufferWithAllocator:impl_->command_allocator];
            [impl_->command_buffer useResidencySet:impl_->residency_set];
            if (options.collect_kernel_timings) {
                [impl_->timestamp_heap invalidateCounterRange:
                    NSMakeRange(0, metal_timestamp_count)];
                [impl_->command_buffer writeTimestampIntoHeap:impl_->timestamp_heap
                                                      atIndex:0];
            }
            id<MTL4ComputeCommandEncoder> encoder =
                [impl_->command_buffer computeCommandEncoder];
            if (encoder == nil) {
                [impl_->command_buffer endCommandBuffer];
                return metal_failure(nil,
                                     "Could not create Metal 4 compute encoder");
            }
            if (impl_->rigid_body_count == 0 && impl_->systems.empty()) {
                [encoder setComputePipelineState:impl_->noop_pipeline];
                [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } else {
                const float substep_timestep =
                    options.timestep / static_cast<float>(options.substeps);
                const NSUInteger thread_count = impl_->rigid_body_count;
                const NSUInteger group_size = std::min<NSUInteger>(
                    64, impl_->rigid_integrate_pipeline.maxTotalThreadsPerThreadgroup);
                if (impl_->rigid_body_count != 0U) {
                    auto *constants = static_cast<StepConstants *>(
                        impl_->step_constants.contents);
                    *constants = {substep_timestep, options.gravity,
                                  impl_->rigid_body_count,
                                  impl_->options.rigid_constraint_capacity,
                                  options.collect_rigid_contacts || debug_enabled
                                      ? 1U
                                      : 0U,
                                  impl_->options.contact_capacity,
                                  options.substeps};
                    [encoder setArgumentTable:impl_->rigid_argument_table];
                }
                impl_->systems.encode(
                    (__bridge void *)encoder,
                    detail::MetalSystemPhase::frame_start,
                    substep_timestep, options.substeps, options.gravity,
                    &timing_context);
                for (std::uint32_t substep = 0; substep < options.substeps;
                    ++substep) {
                    if (impl_->rigid_body_count != 0U) {
                        detail::MetalTimingRecord *integration_timing =
                            begin_timing(
                                encoder, timing_context,
                                detail::MetalTimingStage::rigid_integration);
                        [encoder setArgumentTable:impl_->rigid_argument_table];
                        [encoder setComputePipelineState:
                                     impl_->rigid_integrate_pipeline];
                        [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        end_timing(encoder, timing_context,
                                   integration_timing);
                        if (impl_->rigid_constraint_count != 0U) {
                            [encoder setComputePipelineState:
                                         impl_->rigid_compound_pipeline];
                            [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            [encoder barrierAfterEncoderStages:MTLStageDispatch
                                           beforeEncoderStages:MTLStageDispatch
                                             visibilityOptions:
                                                 MTL4VisibilityOptionDevice];
                        }
                        detail::MetalTimingRecord *bounds_timing =
                            begin_timing(
                                encoder, timing_context,
                                detail::MetalTimingStage::rigid_world_bounds);
                        [encoder setComputePipelineState:
                                     impl_->rigid_world_bounds_pipeline];
                        [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        end_timing(encoder, timing_context, bounds_timing);
                        const NSUInteger contact_thread_count =
                            thread_count * thread_count;
                        const NSUInteger pair_group_size =
                            std::min<NSUInteger>(
                                64, impl_->rigid_pair_filter_pipeline
                                        .maxTotalThreadsPerThreadgroup);
                        detail::MetalTimingRecord *filter_timing =
                            begin_timing(
                                encoder, timing_context,
                                detail::MetalTimingStage::rigid_pair_filter);
                        [encoder setComputePipelineState:
                                     impl_->rigid_pair_filter_pipeline];
                        [encoder dispatchThreads:
                                     MTLSizeMake(contact_thread_count, 1, 1)
                            threadsPerThreadgroup:
                                MTLSizeMake(pair_group_size, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        end_timing(encoder, timing_context, filter_timing);
                        detail::MetalTimingRecord *compaction_timing =
                            begin_timing(
                                encoder, timing_context,
                                detail::MetalTimingStage::
                                    rigid_pair_compaction,
                                3U);
                        [encoder setComputePipelineState:
                                     impl_->rigid_pair_count_rows_pipeline];
                        const NSUInteger row_group_size =
                            std::min<NSUInteger>(
                                64, impl_->rigid_pair_count_rows_pipeline
                                        .maxTotalThreadsPerThreadgroup);
                        [encoder dispatchThreads:
                                     MTLSizeMake(thread_count, 1, 1)
                            threadsPerThreadgroup:
                                MTLSizeMake(row_group_size, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        [encoder setComputePipelineState:
                                     impl_->rigid_pair_prefix_rows_pipeline];
                        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        [encoder setComputePipelineState:
                                     impl_->rigid_pair_scatter_rows_pipeline];
                        const NSUInteger scatter_group_size =
                            std::min<NSUInteger>(
                                64, impl_->rigid_pair_scatter_rows_pipeline
                                        .maxTotalThreadsPerThreadgroup);
                        [encoder dispatchThreads:
                                     MTLSizeMake(thread_count, 1, 1)
                            threadsPerThreadgroup:
                                MTLSizeMake(scatter_group_size, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        end_timing(encoder, timing_context,
                                   compaction_timing);
                        detail::MetalTimingRecord *generation_timing =
                            begin_timing(
                                encoder, timing_context,
                                detail::MetalTimingStage::
                                    rigid_contact_evaluation);
                        [encoder setComputePipelineState:
                                     impl_->rigid_contact_generate_pipeline];
                        const NSUInteger contact_group_size =
                            std::min<NSUInteger>(
                                64, impl_->rigid_contact_generate_pipeline
                                        .maxTotalThreadsPerThreadgroup);
                        [encoder dispatchThreads:
                                     MTLSizeMake(contact_thread_count, 1, 1)
                            threadsPerThreadgroup:
                                MTLSizeMake(contact_group_size, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        end_timing(encoder, timing_context,
                                   generation_timing);
                        detail::MetalTimingRecord *solve_timing =
                            begin_timing(
                                encoder, timing_context,
                                detail::MetalTimingStage::
                                    rigid_contact_solve);
                        [encoder setComputePipelineState:
                                     impl_->rigid_contact_reduce_pipeline];
                        const NSUInteger solve_group_size =
                            std::min<NSUInteger>(
                                256, impl_->rigid_contact_reduce_pipeline
                                         .maxTotalThreadsPerThreadgroup);
                        [encoder dispatchThreads:
                                     MTLSizeMake(solve_group_size, 1, 1)
                            threadsPerThreadgroup:
                                MTLSizeMake(solve_group_size, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                        end_timing(encoder, timing_context, solve_timing);
                        if (impl_->rigid_constraint_count != 0U) {
                            detail::MetalTimingRecord *constraint_timing =
                                begin_timing(
                                    encoder, timing_context,
                                    detail::MetalTimingStage::
                                        rigid_contact_solve);
                            [encoder setComputePipelineState:
                                         impl_->rigid_constraint_pipeline];
                            [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            [encoder barrierAfterEncoderStages:MTLStageDispatch
                                           beforeEncoderStages:MTLStageDispatch
                                             visibilityOptions:
                                                 MTL4VisibilityOptionDevice];
                            end_timing(encoder, timing_context,
                                       constraint_timing);
                        }
                        [encoder setComputePipelineState:
                                     impl_->rigid_advance_substep_pipeline];
                        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                       beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:
                                             MTL4VisibilityOptionDevice];
                    }
                    impl_->systems.encode(
                        (__bridge void *)encoder,
                        detail::MetalSystemPhase::deformable_substep,
                        substep_timestep, options.substeps, options.gravity,
                        &timing_context);
                }
                if (impl_->rigid_body_count != 0U) {
                    detail::MetalTimingRecord *clear_timing = begin_timing(
                        encoder, timing_context,
                        detail::MetalTimingStage::rigid_input_clear);
                    [encoder setArgumentTable:impl_->rigid_argument_table];
                    [encoder setComputePipelineState:impl_->rigid_clear_pipeline];
                    [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
                    [encoder barrierAfterEncoderStages:MTLStageDispatch
                                   beforeEncoderStages:MTLStageDispatch
                                     visibilityOptions:
                                         MTL4VisibilityOptionDevice];
                    end_timing(encoder, timing_context, clear_timing);
                }
                impl_->systems.encode(
                    (__bridge void *)encoder,
                    detail::MetalSystemPhase::frame_end,
                    options.timestep, options.substeps, options.gravity,
                    &timing_context);
            }
            [encoder endEncoding];
            if (options.collect_kernel_timings) {
                [impl_->command_buffer writeTimestampIntoHeap:impl_->timestamp_heap
                                                      atIndex:
                                                          metal_final_timestamp];
            }
            [impl_->command_buffer endCommandBuffer];

            const std::shared_ptr<CompletionState> state =
                completion.impl_->completion;
            {
                std::lock_guard state_lock(state->mutex);
                state->completed = false;
                state->error_code = 0;
                state->gpu_start_time = 0.0;
                state->gpu_end_time = 0.0;
                state->fluid_neighbor_overflow =
                    (__bridge id<MTLBuffer>)
                        impl_->systems.fluid_neighbor_overflow_buffer();
                state->fluid_neighbor_overflow_count = 0U;
            }
            [completion.impl_->commit_options
                addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
                  {
                      std::lock_guard state_lock(state->mutex);
                      state->error_code = feedback.error == nil
                                              ? 0
                                              : static_cast<std::int64_t>(
                                                    feedback.error.code);
                      state->gpu_start_time = feedback.GPUStartTime;
                      state->gpu_end_time = feedback.GPUEndTime;
                      state->fluid_neighbor_overflow_count =
                          state->fluid_neighbor_overflow == nil
                              ? 0U
                              : *static_cast<const std::uint32_t *>(
                                    state->fluid_neighbor_overflow.contents);
                      state->completed = true;
                  }
                  state->completed_condition.notify_all();
                }];

            const id<MTL4CommandBuffer> command_buffers[] = {
                impl_->command_buffer};
            [impl_->command_queue commit:command_buffers
                                   count:1
                                 options:completion.impl_->commit_options];
            const std::uint64_t event_value = impl_->next_event_value++;
            [impl_->command_queue signalEvent:impl_->completion_event
                                        value:event_value];

            completion.impl_->event = impl_->completion_event;
            completion.impl_->event_value = event_value;
            impl_->latest_completion = state;
            impl_->last_frame_profiled = options.collect_kernel_timings;
            impl_->last_timing_overflow = timing_context.overflowed;
            impl_->timing_record_count = timing_context.record_count;
            ++impl_->frame_index;
            ++impl_->revision;
            return success();
        } catch (const std::bad_alloc &) {
            return out_of_memory("Could not allocate Metal frame completion state");
        } catch (...) {
            return {StatusCode::internal_error, 0,
                    "Unexpected failure while submitting Metal frame"};
        }
    }
}

Status World::step(StepOptions options) noexcept {
    if (impl_ == nullptr)
        return invalid_argument("World is not initialized");
    Status status = step_async(options, impl_->synchronous_completion);
    return status ? impl_->synchronous_completion.wait() : status;
}

ContactDeviceView World::contacts() const noexcept {
    if (impl_ == nullptr || !impl_->idle()) return {};
    return impl_->systems.contact_view(impl_->frame_index);
}

RigidContactDeviceView World::rigid_contacts() const noexcept {
    if (impl_ == nullptr || !impl_->idle()) return {};
    const std::uint32_t rigid_contact_count =
        impl_->rigid_contact_event_count();
    return {{(__bridge void *)impl_->rigid_contact_events, 0U,
             rigid_contact_count},
            rigid_contact_count, impl_->frame_index};
}

Status World::physics_debug_frame(
    PhysicsDebugFrameView &output) const noexcept {
    output = {};
    if (impl_ == nullptr)
        return invalid_argument("World is not initialized");
    if (impl_->options.physics_debug.frame_capacity == 0U)
        return {StatusCode::not_supported, 0,
                "Physics debug capture was not enabled at World creation"};
    if (!impl_->idle())
        return busy("Physics debug view requires a completed frame");
    Status status = impl_->record_debug_frame();
    if (!status) return status;
    if (impl_->debug_frame_count == 0U)
        return {StatusCode::not_supported, 0,
                "Physics debug capture has no completed frame"};
    const std::size_t index =
        (impl_->debug_next_frame + impl_->debug_frames.size() - 1U) %
        impl_->debug_frames.size();
    const PhysicsDebugFrame &frame = impl_->debug_frames[index];
    output.frame_index = frame.frame_index;
    output.timestep = frame.timestep;
    output.gravity = frame.gravity;
    output.maximum_fluid_neighbor_count =
        frame.maximum_fluid_neighbor_count;
    output.rigid_bodies = {frame.rigid_bodies.data(),
                           frame.rigid_bodies.size()};
    output.fluid_particles = {frame.fluid_particles.data(),
                              frame.fluid_particles.size()};
    output.cloth_vertices = {frame.cloth_vertices.data(),
                             frame.cloth_vertices.size()};
    output.soft_body_nodes = {frame.soft_body_nodes.data(),
                              frame.soft_body_nodes.size()};
    output.rope_nodes = {frame.rope_nodes.data(), frame.rope_nodes.size()};
    output.rigid_contacts = {frame.rigid_contacts.data(),
                             frame.rigid_contacts.size()};
    output.fluid_contacts = {frame.fluid_contacts.data(),
                             frame.fluid_contacts.size()};
    return success();
}

Status World::copy_physics_debug_capture(
    PhysicsDebugCapture &output) const noexcept {
    output.frames.clear();
    if (impl_ == nullptr)
        return invalid_argument("World is not initialized");
    if (impl_->options.physics_debug.frame_capacity == 0U)
        return {StatusCode::not_supported, 0,
                "Physics debug capture was not enabled at World creation"};
    if (!impl_->idle())
        return busy("Physics debug copy requires a completed frame");
    Status status = impl_->record_debug_frame();
    if (!status) return status;
    try {
        output.frames.reserve(impl_->debug_frame_count);
        const std::size_t first =
            (impl_->debug_next_frame + impl_->debug_frames.size() -
             impl_->debug_frame_count) %
            impl_->debug_frames.size();
        for (std::size_t offset = 0U;
             offset < impl_->debug_frame_count; ++offset)
            output.frames.push_back(impl_->debug_frames[
                (first + offset) % impl_->debug_frames.size()]);
        return success();
    } catch (const std::bad_alloc &) {
        output.frames.clear();
        return out_of_memory("Could not copy Metal physics debug capture");
    } catch (...) {
        output.frames.clear();
        return {StatusCode::internal_error, 0,
                "Unexpected Metal physics debug copy failure"};
    }
}

Status World::collect_step_timings(WorldStepTimings &output) const noexcept {
    output = {};
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    output.frame_index = impl_->frame_index;
    output.available = impl_->last_frame_profiled &&
                       !impl_->last_timing_overflow &&
                       completion_ready(impl_->latest_completion);
    if (!output.available || !impl_->latest_completion) {
        return success();
    }
    std::lock_guard lock(impl_->latest_completion->mutex);
    double seconds = impl_->latest_completion->gpu_end_time -
                     impl_->latest_completion->gpu_start_time;
    const std::uint64_t frequency = [impl_->device queryTimestampFrequency];
    @autoreleasepool {
        NSData *timestamps =
            [impl_->timestamp_heap resolveCounterRange:
                NSMakeRange(0, metal_timestamp_count)];
        if (timestamps.length >=
            metal_timestamp_count * sizeof(MTL4TimestampHeapEntry)) {
            const auto *entries = static_cast<const MTL4TimestampHeapEntry *>(
                timestamps.bytes);
            if (frequency != 0 &&
                entries[metal_final_timestamp].timestamp >=
                    entries[0].timestamp) {
                seconds = static_cast<double>(
                              entries[metal_final_timestamp].timestamp -
                              entries[0].timestamp) /
                          static_cast<double>(frequency);
                for (std::uint32_t index = 0U;
                     index < impl_->timing_record_count; ++index) {
                    const detail::MetalTimingRecord &record =
                        impl_->timing_records[index];
                    const std::uint64_t begin =
                        entries[record.begin_index].timestamp;
                    const std::uint64_t end =
                        entries[record.end_index].timestamp;
                    if (end < begin) continue;
                    KernelTiming *timing =
                        timing_for_stage(output, record.stage);
                    if (timing == nullptr) continue;
                    const float milliseconds = static_cast<float>(
                        static_cast<double>(end - begin) * 1000.0 /
                        static_cast<double>(frequency));
                    timing->total_milliseconds += milliseconds;
                    timing->launch_count += record.launch_count;
                    if (record.stage ==
                            detail::MetalTimingStage::rigid_world_bounds ||
                        record.stage ==
                            detail::MetalTimingStage::rigid_pair_filter ||
                        record.stage ==
                            detail::MetalTimingStage::rigid_pair_compaction ||
                        record.stage == detail::MetalTimingStage::
                                            rigid_contact_evaluation) {
                        output.rigid_contact_generation.total_milliseconds +=
                            milliseconds;
                        output.rigid_contact_generation.launch_count +=
                            record.launch_count;
                    }
                }
            }
        }
    }
    output.total_gpu_milliseconds =
        static_cast<float>(seconds > 0.0 ? seconds * 1000.0 : 0.0);
    return success();
}

Status World::collect_statistics(WorldStatistics &output) const noexcept {
    output = {};
    if (impl_ == nullptr) {
        return invalid_argument("World is not initialized");
    }
    if (!impl_->idle())
        return busy("Statistics require a completed frame");
    output.frame_index = impl_->frame_index;
    output.rigid_body_count = impl_->rigid_body_count;
    output.rigid_constraint_count = impl_->rigid_constraint_count;
    output.triangle_mesh_count = impl_->triangle_mesh_count;
    impl_->systems.collect_statistics(output);
    const ContactDeviceView fluid_contacts =
        impl_->systems.contact_view(impl_->frame_index);
    output.contact_count = fluid_contacts.event_count;
    output.contact_overflow_count =
        impl_->systems.fluid_contact_overflow_count();
    output.allocated_bytes = static_cast<std::size_t>(
        [impl_->command_allocator allocatedSize] + impl_->rigid_ids.length +
        impl_->rigid_states.length + impl_->rigid_previous_states.length +
        impl_->rigid_frame_states.length +
        impl_->rigid_parameters.length +
        impl_->rigid_forces.length + impl_->rigid_torques.length +
        impl_->step_constants.length + impl_->mesh_vertices.length +
        impl_->mesh_indices.length + impl_->mesh_infos.length +
        impl_->mesh_bvh_nodes.length + impl_->mesh_leaf_infos.length +
        impl_->mesh_bvh_leaves.length + impl_->mesh_solid_planes.length +
        impl_->contact_records.length + impl_->rigid_color_owners.length +
        impl_->rigid_world_bounds.length + impl_->rigid_pair_flags.length +
        impl_->rigid_active_pairs.length +
        impl_->rigid_active_pair_count.length +
        impl_->rigid_pair_row_offsets.length +
        impl_->rigid_substep_index.length +
        impl_->rigid_constraints.length +
        impl_->rigid_constraint_geometry.length +
        impl_->rigid_compounds.length +
        impl_->rigid_contact_events.length +
        impl_->rigid_contact_count_buffer.length +
        impl_->systems.allocated_bytes());
    output.allocated_bytes +=
        impl_->debug_input_forces.capacity() * sizeof(Vec3) +
        impl_->debug_input_torques.capacity() * sizeof(Vec3) +
        impl_->debug_frames.capacity() * sizeof(PhysicsDebugFrame);
    for (const PhysicsDebugFrame &frame : impl_->debug_frames) {
        output.allocated_bytes +=
            frame.rigid_bodies.capacity() * sizeof(PhysicsDebugRigidSample) +
            frame.fluid_particles.capacity() * sizeof(PhysicsDebugFluidSample) +
            frame.cloth_vertices.capacity() * sizeof(PhysicsDebugClothSample) +
            frame.soft_body_nodes.capacity() *
                sizeof(PhysicsDebugSoftBodySample) +
            frame.rope_nodes.capacity() * sizeof(PhysicsDebugRopeSample) +
            frame.rigid_contacts.capacity() * sizeof(RigidContactEvent) +
            frame.fluid_contacts.capacity() * sizeof(ContactEvent);
    }
    return success();
}

NativeContext World::native_context() const noexcept {
    if (impl_ == nullptr) {
        return {};
    }
    return {(__bridge void *)impl_->device,
            (__bridge void *)impl_->command_queue};
}

void *World::systems_implementation() noexcept {
    return impl_ == nullptr ? nullptr : &impl_->systems;
}

const void *World::systems_implementation() const noexcept {
    return impl_ == nullptr ? nullptr : &impl_->systems;
}

bool World::systems_mutation_allowed() const noexcept {
    return impl_ != nullptr && impl_->idle();
}

Status World::systems_finish_mutation(Status status) noexcept {
    if (status && impl_ != nullptr) ++impl_->revision;
    return status;
}

Status World::systems_reserve_debug_samples(
    std::uint64_t fluid_particles, std::uint64_t cloth_vertices,
    std::uint64_t soft_body_nodes, std::uint64_t rope_nodes) noexcept {
    if (impl_ == nullptr)
        return invalid_argument("World is not initialized");
    if (impl_->debug_frames.empty()) return success();
    const auto reserve_more = [](auto &values,
                                 std::uint64_t additional) {
        if (additional > values.max_size() - values.capacity())
            throw std::bad_alloc{};
        values.reserve(values.capacity() +
                       static_cast<std::size_t>(additional));
    };
    try {
        for (PhysicsDebugFrame &frame : impl_->debug_frames) {
            reserve_more(frame.fluid_particles, fluid_particles);
            reserve_more(frame.cloth_vertices, cloth_vertices);
            reserve_more(frame.soft_body_nodes, soft_body_nodes);
            reserve_more(frame.rope_nodes, rope_nodes);
        }
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory(
            "Could not reserve fixed Metal physics-debug storage");
    } catch (...) {
        return {StatusCode::internal_error, 0,
                "Unexpected Metal physics-debug reservation failure"};
    }
}

std::uint64_t World::systems_revision() const noexcept {
    return impl_ == nullptr ? 0U : impl_->revision;
}

bool World::systems_rigid_body_valid(RigidBodyId id) const noexcept {
    if (impl_ == nullptr) return false;
    std::uint32_t dense_index = 0U;
    return impl_->valid(id, dense_index);
}

Status World::systems_validate_rope_rest(
    HostSpan<const Vec3> nodes, RopeAttachment first,
    RopeAttachment last, std::uint32_t &first_contact_skip,
    std::uint32_t &last_contact_skip) const noexcept {
    first_contact_skip = 2U;
    last_contact_skip = 2U;
    if (impl_ == nullptr || nodes.data == nullptr || nodes.size < 2U)
        return invalid_argument("Rope rest centerline is invalid");
    int attached[2]{-1, -1};
    const RopeAttachment attachments[2]{first, last};
    for (std::uint32_t end = 0U; end < 2U; ++end) {
        if (!attachments[end].enabled) continue;
        std::uint32_t dense = 0U;
        if (!impl_->valid(attachments[end].body, dense))
            return invalid_handle("Rope attachment rigid-body handle is stale");
        attached[end] = static_cast<int>(dense);
    }
    const auto *states =
        static_cast<const RigidBodyState *>(impl_->rigid_states.contents);
    const auto attachment_skip = [&](bool from_first,
                                     int body) -> std::uint32_t {
        if (body < 0) return 2U;
        const TriangleMeshStorage &mesh =
            impl_->meshes[impl_->rigid_meshes[body].index];
        if (mesh.solid_planes.empty()) return 2U;
        std::uint32_t skip = 2U;
        for (std::uint64_t index = 0U; index < nodes.size / 2U; ++index) {
            const Vec3 point = nodes.data[
                from_first ? index : nodes.size - 1U - index];
            const Vec3 local = inverse_rotate(
                states[body].orientation,
                subtract(point, states[body].position));
            bool inside = true;
            for (const CollisionPlane plane : mesh.solid_planes) {
                if (dot(plane.normal, local) - plane.offset > 0.0F) {
                    inside = false;
                    break;
                }
            }
            if (!inside) break;
            skip = std::max(skip,
                            static_cast<std::uint32_t>(index + 2U));
        }
        return skip;
    };
    const std::uint32_t first_skip = attachment_skip(true, attached[0]);
    const std::uint32_t last_skip = attachment_skip(false, attached[1]);
    first_contact_skip = first_skip;
    last_contact_skip = last_skip;
    try {
        for (std::uint32_t body = 0U; body < impl_->rigid_body_count;
             ++body) {
            const TriangleMeshStorage &mesh =
                impl_->meshes[impl_->rigid_meshes[body].index];
            const RigidBodyState &state = states[body];
            for (std::uint64_t segment = 0U; segment + 1U < nodes.size;
                 ++segment) {
                if ((static_cast<int>(body) == attached[0] &&
                     segment < first_skip) ||
                    (static_cast<int>(body) == attached[1] &&
                     segment + last_skip + 1U >= nodes.size))
                    continue;
                const Vec3 a = inverse_rotate(
                    state.orientation,
                    subtract(nodes.data[segment], state.position));
                const Vec3 b = inverse_rotate(
                    state.orientation,
                    subtract(nodes.data[segment + 1U], state.position));
                const Vec3 low = component_min(a, b);
                const Vec3 high = component_max(a, b);
                if (!bounds_overlap(low, high, mesh.minimum, mesh.maximum))
                    continue;
                std::vector<std::uint32_t> pending{0U};
                while (!pending.empty()) {
                    const BvhNode node = mesh.bvh_nodes[pending.back()];
                    pending.pop_back();
                    if (!bounds_overlap(low, high, node.minimum,
                                        node.maximum))
                        continue;
                    if (node.triangle_count == 0U) {
                        pending.push_back(node.left);
                        pending.push_back(node.right);
                        continue;
                    }
                    for (std::uint32_t triangle = node.first_triangle;
                         triangle < node.first_triangle + node.triangle_count;
                         ++triangle) {
                        const Vec3 x =
                            mesh.vertices[mesh.indices[3U * triangle]];
                        const Vec3 y =
                            mesh.vertices[mesh.indices[3U * triangle + 1U]];
                        const Vec3 z =
                            mesh.vertices[mesh.indices[3U * triangle + 2U]];
                        const Vec3 normal = normalized_or(
                            cross(subtract(y, x), subtract(z, x)), {});
                        const float before = dot(normal, subtract(a, x));
                        const float after = dot(normal, subtract(b, x));
                        if (before * after > 0.0F ||
                            std::abs(before - after) < 1.0e-7F)
                            continue;
                        const Vec3 point = add(
                            a, multiply(subtract(b, a),
                                        before / (before - after)));
                        if (length_squared(subtract(
                                point, closest_on_triangle(point, x, y, z))) <
                            1.0e-12F)
                            return invalid_argument(
                                "Rope rest centerline crosses a rigid collider");
                    }
                }
            }
        }
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory(
            "Could not validate rope rest collision topology");
    } catch (...) {
        return {StatusCode::internal_error, 0,
                "Unexpected rope rest collision validation failure"};
    }
}

std::uint32_t World::systems_triangle_mesh_vertex_count(
    TriangleMeshId id) const noexcept {
    return impl_ != nullptr && impl_->valid(id)
               ? static_cast<std::uint32_t>(
                     impl_->meshes[id.index].vertices.size())
               : 0U;
}

Status World::copy_metal_buffer_to_host(
    void *opaque_buffer, std::uint64_t byte_offset, std::uint64_t byte_count,
    void *destination) noexcept {
    if (impl_ == nullptr || opaque_buffer == nullptr || destination == nullptr ||
        byte_count == 0U)
        return invalid_argument("Metal buffer copy arguments are invalid");
    if (!impl_->idle())
        return busy("Metal buffer upload cannot run during a frame");
    id<MTLBuffer> source = (__bridge id<MTLBuffer>)opaque_buffer;
    if (source.device != impl_->device || byte_offset > source.length ||
        byte_count > source.length - byte_offset)
        return invalid_argument("Metal buffer copy range or device is invalid");
    @autoreleasepool {
        id<MTLBuffer> staging = [impl_->device
            newBufferWithLength:static_cast<NSUInteger>(byte_count)
                         options:MTLResourceStorageModeShared];
        if (staging == nil)
            return metal_failure(nil, "Could not allocate Metal staging buffer");
        std::lock_guard lock(impl_->submission_mutex);
        const bool add_source =
            ![impl_->residency_set containsAllocation:source];
        if (add_source) [impl_->residency_set addAllocation:source];
        [impl_->residency_set addAllocation:staging];
        [impl_->residency_set commit];
        [impl_->command_allocator reset];
        [impl_->command_buffer
            beginCommandBufferWithAllocator:impl_->command_allocator];
        [impl_->command_buffer useResidencySet:impl_->residency_set];
        id<MTL4ComputeCommandEncoder> encoder =
            [impl_->command_buffer computeCommandEncoder];
        if (encoder == nil) {
            [impl_->command_buffer endCommandBuffer];
            [impl_->residency_set removeAllocation:staging];
            if (add_source) [impl_->residency_set removeAllocation:source];
            [impl_->residency_set commit];
            return metal_failure(nil, "Could not encode Metal buffer copy");
        }
        [encoder copyFromBuffer:source
                   sourceOffset:static_cast<NSUInteger>(byte_offset)
                       toBuffer:staging
              destinationOffset:0
                           size:static_cast<NSUInteger>(byte_count)];
        [encoder endEncoding];
        [impl_->command_buffer endCommandBuffer];
        auto state = std::make_shared<CompletionState>();
        MTL4CommitOptions *commit_options = [[MTL4CommitOptions alloc] init];
        [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
          {
              std::lock_guard state_lock(state->mutex);
              state->error_code = feedback.error == nil
                                      ? 0
                                      : static_cast<std::int64_t>(
                                            feedback.error.code);
              state->completed = true;
          }
          state->completed_condition.notify_all();
        }];
        const id<MTL4CommandBuffer> command_buffers[] = {impl_->command_buffer};
        [impl_->command_queue commit:command_buffers
                               count:1
                             options:commit_options];
        const Status status = wait_for_completion(state);
        if (status)
            std::memcpy(destination, staging.contents,
                        static_cast<std::size_t>(byte_count));
        [impl_->residency_set removeAllocation:staging];
        if (add_source) [impl_->residency_set removeAllocation:source];
        [impl_->residency_set commit];
        return status;
    }
}

} // namespace parallel_mater::metal
