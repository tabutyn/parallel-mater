// SPDX-License-Identifier: MIT
#include <parallel_mater/d3d12.hpp>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "pm_d3d12_clear.h"
#include "pm_d3d12_compact_count.h"
#include "pm_d3d12_compact_prefix.h"
#include "pm_d3d12_compact_scatter.h"
#include "pm_d3d12_contacts.h"
#include "pm_d3d12_contacts_warp.h"
#include "pm_d3d12_generate.h"
#include "pm_d3d12_generate_guided.h"
#include "pm_d3d12_generate_warp.h"
#include "pm_d3d12_integrate.h"
#include "pm_d3d12_prepare.h"
#include "pm_d3d12_color.h"
#include "pm_d3d12_finalize.h"
#include "pm_d3d12_contact_prepare.h"
#include "pm_d3d12_contact_events.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace parallel_mater::d3d12 {
namespace {

using Microsoft::WRL::ComPtr;

[[nodiscard]] constexpr Status ok() noexcept { return {}; }
[[nodiscard]] constexpr Status fail(StatusCode code, const char *message,
                                    HRESULT hr = S_OK) noexcept {
    return {code, static_cast<std::int64_t>(hr), message};
}
[[nodiscard]] constexpr Status invalid(const char *message) noexcept {
    return fail(StatusCode::invalid_argument, message);
}
[[nodiscard]] constexpr Status invalid_handle(const char *message) noexcept {
    return fail(StatusCode::invalid_handle, message);
}
[[nodiscard]] constexpr Status capacity(const char *message) noexcept {
    return fail(StatusCode::capacity_exceeded, message);
}
[[nodiscard]] constexpr Status busy_status() noexcept {
    return fail(StatusCode::busy, "A physics frame is still in flight");
}
[[nodiscard]] constexpr Status unsupported(const char *message) noexcept {
    return fail(StatusCode::not_supported, message);
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }
[[nodiscard]] bool finite(Vec3 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z);
}
[[nodiscard]] bool finite(Quaternion value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}
[[nodiscard]] Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
[[nodiscard]] Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
[[nodiscard]] Vec3 multiply(Vec3 value, float scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}
[[nodiscard]] Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
[[nodiscard]] Vec3 rotate(Quaternion q, Vec3 value) noexcept {
    const Vec3 vector{q.x, q.y, q.z};
    const Vec3 twice = multiply(cross(vector, value), 2.0F);
    return add(value, add(multiply(twice, q.w), cross(vector, twice)));
}
[[nodiscard]] float length_squared(Vec3 value) noexcept {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

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

struct ConstraintResource {
    std::uint32_t generation{};
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

struct ConstraintAxisGeometry {
    Vec3 axis{};
    Vec3 inverse_angular_a{};
    Vec3 inverse_angular_b{};
    float angular_denominator{};
    float linear_denominator{};
};

struct ConstraintGeometry {
    std::uint32_t valid{};
    Vec3 arm_a{};
    Vec3 arm_b{};
    Vec3 anchor_error{};
    Vec3 rotation_error{};
    Vec3 hinge_alignment_error{};
    Vec3 piston_alignment_error{};
    ConstraintAxisGeometry axes[3]{};
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

struct MeshInfo {
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
    std::uint32_t alive{};
    std::uint32_t generation{1U};
};

struct CollisionPlane {
    Vec3 normal{};
    float offset{};
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

struct ContactRecord {
    Vec3 point{};
    Vec3 normal{};
    float penetration{};
    std::uint32_t found{};
    std::uint32_t persistent{};
    std::uint32_t warm_started{};
    float impact_fraction{};
    float initial_normal_speed{};
    float accumulated_normal_impulse{};
    Vec3 accumulated_friction_impulse{};
};

struct ContactManifold {
    ContactRecord contacts[8]{};
    std::uint32_t count{};
    std::uint32_t event_offset{};
    std::uint32_t color{};
    std::uint32_t cached{};
    Vec3 initial_relative_position{};
};

struct CachedContact {
    Vec3 local_point{};
    Vec3 normal{};
    float normal_impulse{};
    Vec3 friction_impulse{};
};

struct ContactCacheHeader {
    std::uint32_t valid{};
    std::uint32_t epoch{};
    RigidBodyId body{};
    RigidBodyId collider{};
    float timestep{};
    std::uint32_t count{};
};

struct HingeContactFrame {
    Vec3 anchor{};
    Vec3 axis{};
    Vec3 local_anchor{};
    float inverse_mass{};
    float inverse_moment{};
    std::uint32_t present{};
    std::uint32_t fixed{};
    std::uint32_t axial{};
    std::uint32_t axial_rotation{};
    std::uint32_t fixed_member{};
    std::uint32_t static_body{};
};

struct StepConstants {
    float timestep{};
    Vec3 gravity{};
    std::uint32_t body_count{};
    std::uint32_t constraint_capacity{};
    std::uint32_t collect_contacts{};
    std::uint32_t event_capacity{};
    std::uint32_t substeps{};
    std::uint32_t substep_index{};
    std::uint32_t use_mesh_bvh{1U};
    std::uint32_t contact_epoch{};
    std::uint32_t solver_phase{};
    std::uint32_t solver_pass_begin{};
    std::uint32_t solver_pass_count{};
    std::uint32_t solver_color{};
};

static_assert(sizeof(RigidBodyState) == 52U);
static_assert(sizeof(RigidParameters) == 108U);
static_assert(sizeof(ConstraintResource) == 236U);
static_assert(sizeof(ConstraintAxisGeometry) == 44U);
static_assert(sizeof(ConstraintGeometry) == 208U);
static_assert(sizeof(RigidCompound) == 88U);
static_assert(sizeof(MeshInfo) == 80U);
static_assert(sizeof(CollisionPlane) == 16U);
static_assert(sizeof(BvhNode) == 40U);
static_assert(sizeof(MeshLeafInfo) == 8U);
static_assert(sizeof(ContactRecord) == 64U);
static_assert(sizeof(ContactManifold) == 540U);
static_assert(sizeof(CachedContact) == 40U);
static_assert(sizeof(ContactCacheHeader) == 32U);
static_assert(sizeof(HingeContactFrame) == 68U);
static_assert(sizeof(RigidContactEvent) == 60U);
static_assert(sizeof(StepConstants) == 64U);

struct BodySlot {
    std::uint32_t generation{1U};
    std::uint32_t dense{};
    bool alive{};
};

struct MeshRecord {
    std::uint32_t generation{1U};
    bool alive{};
    std::vector<Vec3> vertices{};
    std::vector<std::uint32_t> indices{};
    std::vector<BvhNode> bvh_nodes{};
    std::vector<CollisionPlane> solid_planes{};
    std::vector<Vec3> shell_normals{};
};

[[nodiscard]] Vec3 normalized_or(Vec3 value, Vec3 fallback) noexcept {
    const float squared = length_squared(value);
    return squared > 1.0e-12F ? multiply(value, 1.0F / std::sqrt(squared))
                              : fallback;
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
        const float side = normal.x * (a.x - center.x) +
                           normal.y * (a.y - center.y) +
                           normal.z * (a.z - center.z);
        if (std::abs(side) <= tolerance) return {};
        if (side < 0.0F) normal = multiply(normal, -1.0F);
        const float offset =
            normal.x * a.x + normal.y * a.y + normal.z * a.z;
        for (const Vec3 vertex : vertices)
            if (normal.x * vertex.x + normal.y * vertex.y +
                    normal.z * vertex.z - offset > tolerance)
                return {};
        planes.push_back({normal, offset});
    }
    return planes;
}

std::vector<Vec3> convex_shell_normals(
    const std::vector<Vec3> &vertices,
    const std::vector<std::uint32_t> &indices) {
    std::map<std::array<float, 3>, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(vertices.size());
    std::vector<std::uint32_t> parent(vertices.size());
    for (std::uint32_t i = 0U; i < vertices.size(); ++i) {
        const Vec3 point = vertices[i];
        remap[i] = welded.emplace(
            std::array<float, 3>{point.x, point.y, point.z}, i).first->second;
        parent[i] = i;
    }
    const auto root = [&](std::uint32_t value) {
        while (parent[value] != value) {
            parent[value] = parent[parent[value]];
            value = parent[value];
        }
        return value;
    };
    for (std::size_t index = 0U; index < indices.size(); index += 3U) {
        for (std::size_t corner = 1U; corner < 3U; ++corner) {
            const std::uint32_t a = root(remap[indices[index]]);
            const std::uint32_t b = root(remap[indices[index + corner]]);
            parent[b] = a;
        }
    }
    std::map<std::uint32_t, std::vector<std::uint32_t>> shells;
    for (std::uint32_t triangle = 0U;
         triangle < indices.size() / 3U; ++triangle)
        shells[root(remap[indices[triangle * 3U]])].push_back(triangle);
    std::vector<Vec3> normals(indices.size() / 3U);
    bool found = false;
    for (const auto &[id, triangles] : shells) {
        (void)id;
        std::map<std::uint32_t, std::uint32_t> local;
        std::vector<Vec3> points;
        std::vector<std::uint32_t> faces;
        for (const std::uint32_t triangle : triangles) {
            for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
                const std::uint32_t vertex =
                    remap[indices[triangle * 3U + corner]];
                const auto [entry, inserted] = local.emplace(
                    vertex, static_cast<std::uint32_t>(points.size()));
                if (inserted) points.push_back(vertices[vertex]);
                faces.push_back(entry->second);
            }
        }
        const std::vector<CollisionPlane> planes =
            closed_convex_planes(points, faces);
        if (planes.empty()) continue;
        found = true;
        for (std::size_t index = 0U; index < triangles.size(); ++index)
            normals[triangles[index]] = planes[index].normal;
    }
    return found ? normals : std::vector<Vec3>{};
}

void build_mesh_bvh(MeshRecord &mesh) {
    const std::uint32_t triangle_count =
        static_cast<std::uint32_t>(mesh.indices.size() / 3U);
    std::vector<std::uint32_t> order(triangle_count);
    std::iota(order.begin(), order.end(), 0U);
    mesh.bvh_nodes.clear();
    mesh.bvh_nodes.reserve(static_cast<std::size_t>(triangle_count) * 2U);
    const auto minimum = [](Vec3 first, Vec3 second) noexcept {
        return Vec3{std::min(first.x, second.x), std::min(first.y, second.y),
                    std::min(first.z, second.z)};
    };
    const auto maximum = [](Vec3 first, Vec3 second) noexcept {
        return Vec3{std::max(first.x, second.x), std::max(first.y, second.y),
                    std::max(first.z, second.z)};
    };
    const auto coordinate = [](Vec3 value, int axis) noexcept {
        return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
    };
    const auto centroid = [&](std::uint32_t triangle) noexcept {
        const std::uint32_t base = triangle * 3U;
        return multiply(add(add(mesh.vertices[mesh.indices[base]],
                                mesh.vertices[mesh.indices[base + 1U]]),
                            mesh.vertices[mesh.indices[base + 2U]]),
                        1.0F / 3.0F);
    };
    std::function<std::uint32_t(std::uint32_t, std::uint32_t)> build =
        [&](std::uint32_t begin, std::uint32_t end) {
            BvhNode node{};
            const float limit = std::numeric_limits<float>::max();
            node.minimum = {limit, limit, limit};
            node.maximum = {-limit, -limit, -limit};
            for (std::uint32_t item = begin; item < end; ++item) {
                const std::uint32_t base = order[item] * 3U;
                for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
                    const Vec3 vertex = mesh.vertices[mesh.indices[base + corner]];
                    node.minimum = minimum(node.minimum, vertex);
                    node.maximum = maximum(node.maximum, vertex);
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
            std::stable_sort(order.begin() + begin, order.begin() + end,
                [&](std::uint32_t first, std::uint32_t second) {
                    const float first_value = coordinate(centroid(first), axis);
                    const float second_value = coordinate(centroid(second), axis);
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

struct GpuBuffer {
    ComPtr<ID3D12Resource> resource{};
    ComPtr<ID3D12Resource> upload{};
    ComPtr<ID3D12Resource> readback{};
    void *upload_data{};
    void *readback_data{};
    std::uint64_t bytes{};
    std::uint32_t stride{};
    std::uint32_t elements{};
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COPY_DEST};

    GpuBuffer() = default;
    GpuBuffer(const GpuBuffer &) = delete;
    GpuBuffer &operator=(const GpuBuffer &) = delete;
    GpuBuffer(GpuBuffer &&other) noexcept { *this = std::move(other); }
    GpuBuffer &operator=(GpuBuffer &&other) noexcept {
        if (this == &other) return *this;
        reset();
        resource = std::move(other.resource);
        upload = std::move(other.upload);
        readback = std::move(other.readback);
        upload_data = std::exchange(other.upload_data, nullptr);
        readback_data = std::exchange(other.readback_data, nullptr);
        bytes = std::exchange(other.bytes, 0U);
        stride = std::exchange(other.stride, 0U);
        elements = std::exchange(other.elements, 0U);
        state = std::exchange(other.state, D3D12_RESOURCE_STATE_COPY_DEST);
        return *this;
    }
    ~GpuBuffer() { reset(); }

    void reset() noexcept {
        if (upload && upload_data != nullptr) upload->Unmap(0, nullptr);
        if (readback && readback_data != nullptr) readback->Unmap(0, nullptr);
        upload_data = nullptr;
        readback_data = nullptr;
        resource.Reset();
        upload.Reset();
        readback.Reset();
    }
};

[[nodiscard]] D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    result.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    result.CreationNodeMask = 1U;
    result.VisibleNodeMask = 1U;
    return result;
}

[[nodiscard]] D3D12_RESOURCE_DESC buffer_description(
    std::uint64_t bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC result{};
    result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    result.Alignment = 0U;
    result.Width = std::max<std::uint64_t>(bytes, 4U);
    result.Height = 1U;
    result.DepthOrArraySize = 1U;
    result.MipLevels = 1U;
    result.Format = DXGI_FORMAT_UNKNOWN;
    result.SampleDesc = {1U, 0U};
    result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    result.Flags = flags;
    return result;
}

[[nodiscard]] bool valid_state(RigidBodyState state) noexcept {
    const float orientation_length = state.orientation.x * state.orientation.x +
        state.orientation.y * state.orientation.y +
        state.orientation.z * state.orientation.z +
        state.orientation.w * state.orientation.w;
    return finite(state.position) && finite(state.orientation) &&
        finite(state.linear_velocity) && finite(state.angular_velocity) &&
        orientation_length > 1.0e-12F;
}

[[nodiscard]] bool valid_constraint_options(
    const RigidConstraintOptions &value) noexcept {
    const auto limits_valid = [](const RigidConstraintLimitOptions &limits) {
        if ((limits.axes & ~rigid_constraint_all_axes) != 0U ||
            !finite(limits.lower) || !finite(limits.upper)) return false;
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if ((limits.axes & (1U << axis)) == 0U) continue;
            const float lower = axis == 0 ? limits.lower.x :
                                axis == 1 ? limits.lower.y : limits.lower.z;
            const float upper = axis == 0 ? limits.upper.x :
                                axis == 1 ? limits.upper.y : limits.upper.z;
            if (lower > upper) return false;
        }
        return true;
    };
    const auto spring_valid = [](const RigidConstraintSpringOptions &spring) {
        return (spring.axes & ~rigid_constraint_all_axes) == 0U &&
            finite(spring.stiffness) && finite(spring.damping) &&
            spring.stiffness.x >= 0.0F && spring.stiffness.y >= 0.0F &&
            spring.stiffness.z >= 0.0F && spring.damping.x >= 0.0F &&
            spring.damping.y >= 0.0F && spring.damping.z >= 0.0F;
    };
    return static_cast<std::uint32_t>(value.type) <=
            static_cast<std::uint32_t>(RigidConstraintType::motor) &&
        finite(value.local_anchor_a) && finite(value.local_anchor_b) &&
        finite(value.local_orientation_a) && finite(value.local_orientation_b) &&
        limits_valid(value.linear_limits) && limits_valid(value.angular_limits) &&
        spring_valid(value.linear_springs) && spring_valid(value.angular_springs) &&
        finite(value.motor.linear_target_velocity) &&
        finite(value.motor.angular_target_velocity) &&
        finite(value.motor.linear_maximum_impulse) &&
        finite(value.motor.angular_maximum_impulse) &&
        value.motor.linear_maximum_impulse >= 0.0F &&
        value.motor.angular_maximum_impulse >= 0.0F &&
        finite(value.breaking_impulse_threshold) &&
        value.breaking_impulse_threshold >= 0.0F &&
        value.solver_iterations >= 1U && value.solver_iterations <= 64U;
}

[[nodiscard]] ConstraintResource make_constraint(
    const RigidConstraintOptions &source, std::uint32_t generation,
    std::uint32_t dense_a, std::uint32_t dense_b) noexcept {
    ConstraintResource result{};
    result.generation = generation;
    result.alive = 1U;
    result.type = static_cast<std::uint32_t>(source.type);
    result.body_a = dense_a;
    result.body_b = dense_b;
    result.enabled = source.enabled ? 1U : 0U;
    result.local_anchor_a = source.local_anchor_a;
    result.local_anchor_b = source.local_anchor_b;
    result.local_orientation_a = source.local_orientation_a;
    result.local_orientation_b = source.local_orientation_b;
    result.breaking_impulse_threshold = source.breaking_impulse_threshold;
    result.linear_limit_axes = source.linear_limits.axes;
    result.linear_limit_lower = source.linear_limits.lower;
    result.linear_limit_upper = source.linear_limits.upper;
    result.angular_limit_axes = source.angular_limits.axes;
    result.angular_limit_lower = source.angular_limits.lower;
    result.angular_limit_upper = source.angular_limits.upper;
    result.linear_spring_axes = source.linear_springs.axes;
    result.linear_spring_stiffness = source.linear_springs.stiffness;
    result.linear_spring_damping = source.linear_springs.damping;
    result.angular_spring_axes = source.angular_springs.axes;
    result.angular_spring_stiffness = source.angular_springs.stiffness;
    result.angular_spring_damping = source.angular_springs.damping;
    result.linear_motor_enabled = source.motor.linear_enabled ? 1U : 0U;
    result.angular_motor_enabled = source.motor.angular_enabled ? 1U : 0U;
    result.linear_target_velocity = source.motor.linear_target_velocity;
    result.linear_maximum_impulse = source.motor.linear_maximum_impulse;
    result.angular_target_velocity = source.motor.angular_target_velocity;
    result.angular_maximum_impulse = source.motor.angular_maximum_impulse;
    result.solver_iterations = source.solver_iterations;
    result.disable_collisions = source.disable_collisions ? 1U : 0U;
    return result;
}

} // namespace

struct FrameToken::Impl {
    ComPtr<ID3D12Fence> fence{};
    ComPtr<ID3D12Device> device{};
    std::uint64_t value{};
    HANDLE event_handle{};

    ~Impl() { if (event_handle != nullptr) CloseHandle(event_handle); }
};

struct World::Impl {
    WorldOptions options{};
    ComPtr<ID3D12Device> device{};
    ComPtr<ID3D12CommandQueue> queue{};
    ComPtr<ID3D12CommandAllocator> allocator{};
    ComPtr<ID3D12GraphicsCommandList> commands{};
    ComPtr<ID3D12Fence> fence{};
    HANDLE fence_event{};
    std::uint64_t fence_value{};
    std::uint64_t submitted_value{};
    ComPtr<ID3D12DescriptorHeap> descriptors{};
    std::uint32_t descriptor_size{};
    ComPtr<ID3D12RootSignature> root_signature{};
    ComPtr<ID3D12PipelineState> integrate_pipeline{};
    ComPtr<ID3D12PipelineState> prepare_pipeline{};
    ComPtr<ID3D12PipelineState> generate_pipeline{};
    ComPtr<ID3D12PipelineState> generate_guided_pipeline{};
    ComPtr<ID3D12PipelineState> generate_warp_pipeline{};
    ComPtr<ID3D12PipelineState> color_pipeline{},finalize_pipeline{},contact_prepare_pipeline{};
    ComPtr<ID3D12PipelineState> clear_pipeline{};
    ComPtr<ID3D12PipelineState> compact_count_pipeline{},compact_prefix_pipeline{},compact_scatter_pipeline{};
    ComPtr<ID3D12PipelineState> contacts_pipeline{},contacts_warp_pipeline{};
    ComPtr<ID3D12PipelineState> contact_events_pipeline{};
    ComPtr<ID3D12QueryHeap> timestamp_heap{};
    ComPtr<ID3D12Resource> timestamp_readback{};
    std::uint64_t timestamp_frequency{};
    std::uint32_t timestamp_count{};
    std::uint32_t timed_solver_launches{};
    std::uint32_t timed_compaction_launches{};

    GpuBuffer states{};
    GpuBuffer parameters{};
    GpuBuffer forces{};
    GpuBuffer torques{};
    GpuBuffer previous_states{};
    GpuBuffer constraints{};
    GpuBuffer constraint_geometry{};
    GpuBuffer meshes{};
    GpuBuffer body_ids{};
    GpuBuffer contact_events{};
    GpuBuffer counters{};
    GpuBuffer active_pairs{};
    GpuBuffer compact_blocks{};
    GpuBuffer compounds{};
    GpuBuffer mesh_vertices{};
    GpuBuffer mesh_indices{};
    GpuBuffer mesh_bvh_nodes{};
    GpuBuffer mesh_leaf_infos{};
    GpuBuffer mesh_bvh_leaves{};
    GpuBuffer mesh_solid_planes{};
    GpuBuffer mesh_shell_normals{};
    GpuBuffer manifolds{};
    GpuBuffer face_clip_scratch{};
    GpuBuffer contact_cache_rows{};
    GpuBuffer contact_cache_headers{};
    GpuBuffer hinge_frames{};
    GpuBuffer pair_colors{};
    GpuBuffer color_owners{};

    std::vector<RigidBodyState> host_states{};
    std::vector<RigidParameters> host_parameters{};
    std::vector<Vec3> host_forces{};
    std::vector<Vec3> host_torques{};
    std::vector<RigidBodyId> host_ids{};
    std::vector<BodySlot> body_slots{};
    std::vector<ConstraintResource> host_constraints{};
    std::vector<MeshRecord> mesh_records{};
    std::vector<MeshInfo> host_meshes{};
    std::vector<Vec3> host_mesh_vertices{};
    std::vector<std::uint32_t> host_mesh_indices{};
    std::vector<BvhNode> host_mesh_bvh_nodes{};
    std::vector<MeshLeafInfo> host_mesh_leaf_infos{};
    std::vector<std::uint32_t> host_mesh_bvh_leaves{};
    std::vector<CollisionPlane> host_mesh_solid_planes{};
    std::vector<Vec3> host_mesh_shell_normals{};
    std::uint64_t revision{};
    std::uint64_t frame_index{};
    std::uint32_t contact_epoch{};
    std::uint32_t rigid_event_count{};
    bool contact_overflow{};
    bool host_results_current{true};
    bool timings_requested{};
    bool use_mesh_bvh{true};
    WorldStepTimings timings{};
    std::size_t allocated_bytes{};

    ~Impl() { if (fence_event != nullptr) CloseHandle(fence_event); }

    [[nodiscard]] bool idle() const noexcept {
        return submitted_value == 0U ||
            fence->GetCompletedValue() >= submitted_value;
    }

    [[nodiscard]] Status hr_status(HRESULT hr, const char *message) const noexcept {
        if (SUCCEEDED(hr)) return ok();
        const HRESULT removed = device ? device->GetDeviceRemovedReason() : S_OK;
        if (FAILED(removed))
            return fail(StatusCode::device_removed,
                        "The D3D12 device was removed", removed);
        return fail(StatusCode::d3d12_failure, message, hr);
    }

    [[nodiscard]] Status wait_value(std::uint64_t value) noexcept {
        const std::uint64_t completed = fence->GetCompletedValue();
        if (completed == UINT64_MAX)
            return fail(StatusCode::device_removed,
                        "The D3D12 device was removed",
                        device->GetDeviceRemovedReason());
        if (value == 0U || completed >= value) return ok();
        HRESULT hr = fence->SetEventOnCompletion(value, fence_event);
        if (FAILED(hr)) return hr_status(hr, "Could not arm the D3D12 fence");
        const DWORD result = WaitForSingleObject(fence_event, INFINITE);
        if (result != WAIT_OBJECT_0)
            return fail(StatusCode::d3d12_failure,
                        "Waiting for the D3D12 fence failed",
                        HRESULT_FROM_WIN32(GetLastError()));
        return ok();
    }

    [[nodiscard]] Status begin_commands() noexcept {
        if (!idle()) return busy_status();
        HRESULT hr = allocator->Reset();
        if (FAILED(hr)) return hr_status(hr, "Could not reset command allocator");
        hr = commands->Reset(allocator.Get(), nullptr);
        if (FAILED(hr)) return hr_status(hr, "Could not reset command list");
        return ok();
    }

    [[nodiscard]] Status submit(bool wait) noexcept {
        HRESULT hr = commands->Close();
        if (FAILED(hr)) return hr_status(hr, "Could not close command list");
        ID3D12CommandList *lists[] = {commands.Get()};
        queue->ExecuteCommandLists(1U, lists);
        submitted_value = ++fence_value;
        hr = queue->Signal(fence.Get(), submitted_value);
        if (FAILED(hr)) return hr_status(hr, "Could not signal D3D12 queue");
        return wait ? wait_value(submitted_value) : ok();
    }

    void transition(GpuBuffer &buffer, D3D12_RESOURCE_STATES target) noexcept {
        if (buffer.state == target) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = buffer.resource.Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = buffer.state;
        barrier.Transition.StateAfter = target;
        commands->ResourceBarrier(1U, &barrier);
        buffer.state = target;
    }

    void uav_barrier(GpuBuffer &buffer) noexcept {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = buffer.resource.Get();
        commands->ResourceBarrier(1U, &barrier);
    }

    [[nodiscard]] Status make_buffer(std::uint32_t elements,
                                     std::uint32_t stride,
                                     GpuBuffer &output) noexcept {
        output.elements = std::max(elements, 1U);
        output.stride = stride;
        output.bytes = static_cast<std::uint64_t>(output.elements) * stride;
        const D3D12_HEAP_PROPERTIES default_heap =
            heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        const D3D12_HEAP_PROPERTIES upload_heap =
            heap_properties(D3D12_HEAP_TYPE_UPLOAD);
        const D3D12_HEAP_PROPERTIES readback_heap =
            heap_properties(D3D12_HEAP_TYPE_READBACK);
        const D3D12_RESOURCE_DESC gpu_desc = buffer_description(
            output.bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const D3D12_RESOURCE_DESC staging_desc = buffer_description(output.bytes);
        HRESULT hr = device->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &gpu_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(output.resource.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return hr_status(hr, "Could not allocate D3D12 buffer");
        hr = device->CreateCommittedResource(
            &upload_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(output.upload.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return hr_status(hr, "Could not allocate upload buffer");
        hr = device->CreateCommittedResource(
            &readback_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(output.readback.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return hr_status(hr, "Could not allocate readback buffer");
        D3D12_RANGE no_read{0U, 0U};
        hr = output.upload->Map(0U, &no_read, &output.upload_data);
        if (FAILED(hr)) return hr_status(hr, "Could not map upload buffer");
        D3D12_RANGE no_write{0U, 0U};
        hr = output.readback->Map(0U, &no_write, &output.readback_data);
        if (FAILED(hr)) return hr_status(hr, "Could not map readback buffer");
        std::memset(output.upload_data, 0, static_cast<std::size_t>(output.bytes));
        allocated_bytes += static_cast<std::size_t>(output.bytes);
        return ok();
    }

    [[nodiscard]] Status make_scratch_buffer(std::uint32_t elements,
                                             std::uint32_t stride,
                                             GpuBuffer &output) noexcept {
        output.elements = std::max(elements, 1U);
        output.stride = stride;
        output.bytes = static_cast<std::uint64_t>(output.elements) * stride;
        const D3D12_HEAP_PROPERTIES default_heap =
            heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        const D3D12_RESOURCE_DESC description = buffer_description(
            output.bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const HRESULT hr = device->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &description,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(output.resource.ReleaseAndGetAddressOf()));
        if (FAILED(hr))
            return hr_status(hr, "Could not allocate D3D12 scratch buffer");
        output.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        allocated_bytes += static_cast<std::size_t>(output.bytes);
        return ok();
    }

    void write_uav_descriptor(std::uint32_t slot,
                              const GpuBuffer &buffer) noexcept {
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor =
            descriptors->GetCPUDescriptorHandleForHeapStart();
        descriptor.ptr += static_cast<SIZE_T>(slot) * descriptor_size;
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Format = DXGI_FORMAT_UNKNOWN;
        view.Buffer.NumElements = buffer.elements;
        view.Buffer.StructureByteStride = buffer.stride;
        device->CreateUnorderedAccessView(
            buffer.resource.Get(), nullptr, &view, descriptor);
    }

    void write_srv_descriptor(std::uint32_t slot,
                              const GpuBuffer &buffer) noexcept {
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor =
            descriptors->GetCPUDescriptorHandleForHeapStart();
        descriptor.ptr += static_cast<SIZE_T>(slot) * descriptor_size;
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        view.Format = DXGI_FORMAT_UNKNOWN;
        view.Buffer.NumElements = buffer.elements;
        view.Buffer.StructureByteStride = buffer.stride;
        device->CreateShaderResourceView(buffer.resource.Get(), &view, descriptor);
    }

    void record_upload(GpuBuffer &buffer, const void *source,
                       std::uint64_t bytes) noexcept {
        bytes = std::min(bytes, buffer.bytes);
        if (source != nullptr && bytes != 0U)
            std::memcpy(buffer.upload_data, source, static_cast<std::size_t>(bytes));
        transition(buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyBufferRegion(buffer.resource.Get(), 0U,
                                   buffer.upload.Get(), 0U, buffer.bytes);
        transition(buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    [[nodiscard]] Status upload_host_data() noexcept {
        Status status = begin_commands();
        if (!status) return status;
        record_upload(states, host_states.data(),
            host_states.size() * sizeof(RigidBodyState));
        record_upload(parameters, host_parameters.data(),
            host_parameters.size() * sizeof(RigidParameters));
        record_upload(forces, host_forces.data(), host_forces.size() * sizeof(Vec3));
        record_upload(torques, host_torques.data(), host_torques.size() * sizeof(Vec3));
        record_upload(previous_states, host_states.data(),
            host_states.size() * sizeof(RigidBodyState));
        record_upload(constraints, host_constraints.data(),
            host_constraints.size() * sizeof(ConstraintResource));
        record_upload(meshes, host_meshes.data(), host_meshes.size() * sizeof(MeshInfo));
        record_upload(body_ids, host_ids.data(), host_ids.size() * sizeof(RigidBodyId));
        record_upload(mesh_vertices, host_mesh_vertices.data(),
            host_mesh_vertices.size() * sizeof(Vec3));
        record_upload(mesh_indices, host_mesh_indices.data(),
            host_mesh_indices.size() * sizeof(std::uint32_t));
        record_upload(mesh_bvh_nodes, host_mesh_bvh_nodes.data(),
            host_mesh_bvh_nodes.size() * sizeof(BvhNode));
        record_upload(mesh_leaf_infos, host_mesh_leaf_infos.data(),
            host_mesh_leaf_infos.size() * sizeof(MeshLeafInfo));
        record_upload(mesh_bvh_leaves, host_mesh_bvh_leaves.data(),
            host_mesh_bvh_leaves.size() * sizeof(std::uint32_t));
        record_upload(mesh_solid_planes, host_mesh_solid_planes.data(),
            host_mesh_solid_planes.size() * sizeof(CollisionPlane));
        record_upload(mesh_shell_normals, host_mesh_shell_normals.data(),
            host_mesh_shell_normals.size() * sizeof(Vec3));
        record_upload(contact_cache_rows, nullptr, 0U);
        record_upload(contact_cache_headers, nullptr, 0U);
        const std::uint32_t zero[33]{};
        record_upload(counters, zero, sizeof(zero));
        status = submit(true);
        if (status) host_results_current = true;
        return status;
    }

    [[nodiscard]] Status rebuild_mesh_buffers() noexcept {
        std::uint64_t vertex_count = 0U;
        std::uint64_t index_count = 0U;
        std::uint64_t node_count = 0U;
        std::uint64_t leaf_count = 0U;
        std::uint64_t plane_count = 0U;
        std::uint64_t normal_count = 0U;
        for (const MeshRecord &record : mesh_records) {
            if (!record.alive) continue;
            vertex_count += record.vertices.size();
            index_count += record.indices.size();
            node_count += record.bvh_nodes.size();
            plane_count += record.solid_planes.size();
            normal_count += record.indices.size() / 3U;
            for (const BvhNode &node : record.bvh_nodes)
                if (node.triangle_count != 0U) ++leaf_count;
        }
        if (vertex_count > UINT32_MAX || index_count > UINT32_MAX ||
            node_count > UINT32_MAX || leaf_count > UINT32_MAX ||
            plane_count > UINT32_MAX || normal_count > UINT32_MAX)
            return capacity("Triangle mesh storage exceeds D3D12 limits");
        host_mesh_vertices.clear();
        host_mesh_indices.clear();
        host_mesh_bvh_nodes.clear();
        host_mesh_bvh_leaves.clear();
        host_mesh_solid_planes.clear();
        host_mesh_shell_normals.clear();
        host_mesh_leaf_infos.assign(mesh_records.size(), {});
        host_mesh_vertices.reserve(static_cast<std::size_t>(vertex_count));
        host_mesh_indices.reserve(static_cast<std::size_t>(index_count));
        host_mesh_bvh_nodes.reserve(static_cast<std::size_t>(node_count));
        host_mesh_bvh_leaves.reserve(static_cast<std::size_t>(leaf_count));
        host_mesh_solid_planes.reserve(static_cast<std::size_t>(plane_count));
        host_mesh_shell_normals.reserve(static_cast<std::size_t>(normal_count));
        for (std::uint32_t slot = 0U; slot < mesh_records.size(); ++slot) {
            const MeshRecord &record = mesh_records[slot];
            MeshInfo &mesh = host_meshes[slot];
            if (!record.alive) continue;
            mesh.vertex_offset = static_cast<std::uint32_t>(
                host_mesh_vertices.size());
            mesh.vertex_count = static_cast<std::uint32_t>(record.vertices.size());
            mesh.index_offset = static_cast<std::uint32_t>(
                host_mesh_indices.size());
            mesh.index_count = static_cast<std::uint32_t>(record.indices.size());
            mesh.bvh_node_offset = static_cast<std::uint32_t>(
                host_mesh_bvh_nodes.size());
            mesh.bvh_node_count = static_cast<std::uint32_t>(
                record.bvh_nodes.size());
            mesh.solid_plane_offset = static_cast<std::uint32_t>(
                host_mesh_solid_planes.size());
            mesh.solid_plane_count = static_cast<std::uint32_t>(
                record.solid_planes.size());
            MeshLeafInfo &leaf_info = host_mesh_leaf_infos[slot];
            leaf_info.offset = static_cast<std::uint32_t>(
                host_mesh_bvh_leaves.size());
            host_mesh_vertices.insert(host_mesh_vertices.end(),
                record.vertices.begin(), record.vertices.end());
            host_mesh_indices.insert(host_mesh_indices.end(),
                record.indices.begin(), record.indices.end());
            host_mesh_solid_planes.insert(host_mesh_solid_planes.end(),
                record.solid_planes.begin(), record.solid_planes.end());
            const std::size_t triangle_count = record.indices.size() / 3U;
            if (record.shell_normals.size() == triangle_count)
                host_mesh_shell_normals.insert(host_mesh_shell_normals.end(),
                    record.shell_normals.begin(), record.shell_normals.end());
            else
                host_mesh_shell_normals.insert(host_mesh_shell_normals.end(),
                    triangle_count, Vec3{});
            for (std::uint32_t node_index = 0U;
                 node_index < record.bvh_nodes.size(); ++node_index) {
                BvhNode node = record.bvh_nodes[node_index];
                if (node.triangle_count == 0U) {
                    node.left += mesh.bvh_node_offset;
                    node.right += mesh.bvh_node_offset;
                } else {
                    host_mesh_bvh_leaves.push_back(
                        mesh.bvh_node_offset + node_index);
                }
                host_mesh_bvh_nodes.push_back(node);
            }
            leaf_info.count = static_cast<std::uint32_t>(
                host_mesh_bvh_leaves.size() - leaf_info.offset);
        }
        GpuBuffer vertices;
        GpuBuffer indices;
        GpuBuffer nodes;
        GpuBuffer leaf_infos;
        GpuBuffer leaves;
        GpuBuffer planes;
        GpuBuffer shell_normals;
        const std::size_t allocation_start = allocated_bytes;
        Status status = make_buffer(
            static_cast<std::uint32_t>(vertex_count), sizeof(Vec3), vertices);
        if (!status) { allocated_bytes = allocation_start; return status; }
        status = make_buffer(static_cast<std::uint32_t>(index_count),
                             sizeof(std::uint32_t), indices);
        if (!status) { allocated_bytes = allocation_start; return status; }
        status = make_buffer(static_cast<std::uint32_t>(node_count),
                             sizeof(BvhNode), nodes);
        if (!status) { allocated_bytes = allocation_start; return status; }
        status = make_buffer(static_cast<std::uint32_t>(mesh_records.size()),
                             sizeof(MeshLeafInfo), leaf_infos);
        if (!status) { allocated_bytes = allocation_start; return status; }
        status = make_buffer(static_cast<std::uint32_t>(leaf_count),
                             sizeof(std::uint32_t), leaves);
        if (!status) { allocated_bytes = allocation_start; return status; }
        status = make_buffer(static_cast<std::uint32_t>(plane_count),
                             sizeof(CollisionPlane), planes);
        if (!status) { allocated_bytes = allocation_start; return status; }
        status = make_buffer(static_cast<std::uint32_t>(normal_count),
                             sizeof(Vec3), shell_normals);
        if (!status) { allocated_bytes = allocation_start; return status; }
        allocated_bytes -= static_cast<std::size_t>(
            mesh_vertices.bytes + mesh_indices.bytes + mesh_bvh_nodes.bytes +
            mesh_leaf_infos.bytes + mesh_bvh_leaves.bytes +
            mesh_solid_planes.bytes + mesh_shell_normals.bytes);
        mesh_vertices = std::move(vertices);
        mesh_indices = std::move(indices);
        mesh_bvh_nodes = std::move(nodes);
        mesh_leaf_infos = std::move(leaf_infos);
        mesh_bvh_leaves = std::move(leaves);
        mesh_solid_planes = std::move(planes);
        mesh_shell_normals = std::move(shell_normals);
        write_srv_descriptor(64U, mesh_bvh_nodes);
        write_srv_descriptor(65U, mesh_leaf_infos);
        write_srv_descriptor(66U, mesh_bvh_leaves);
        write_srv_descriptor(67U, mesh_solid_planes);
        write_srv_descriptor(68U, mesh_shell_normals);
        return upload_host_data();
    }

    [[nodiscard]] Status sync_results(bool wait) noexcept {
        if (host_results_current) return ok();
        if (fence->GetCompletedValue() == UINT64_MAX)
            return fail(StatusCode::device_removed,
                        "The D3D12 device was removed",
                        device->GetDeviceRemovedReason());
        if (!idle()) {
            if (!wait) return busy_status();
            Status status = wait_value(submitted_value);
            if (!status) return status;
        }
        if (!host_states.empty())
            std::memcpy(host_states.data(), states.readback_data,
                        host_states.size() * sizeof(RigidBodyState));
        if (!host_parameters.empty())
            std::memcpy(host_parameters.data(), parameters.readback_data,
                        host_parameters.size() * sizeof(RigidParameters));
        if (!host_constraints.empty())
            std::memcpy(host_constraints.data(), constraints.readback_data,
                        host_constraints.size() * sizeof(ConstraintResource));
        const auto *counts = static_cast<const std::uint32_t *>(counters.readback_data);
        rigid_event_count = std::min(counts[0], options.contact_capacity);
        contact_overflow = counts[1] != 0U || counts[0] > options.contact_capacity;
        if (timings_requested) {
            void *mapped = nullptr;
            D3D12_RANGE range{0U, timestamp_count * sizeof(std::uint64_t)};
            if (timestamp_readback && SUCCEEDED(timestamp_readback->Map(0U, &range, &mapped))) {
                const auto *values = static_cast<const std::uint64_t *>(mapped);
                const float scale=timestamp_frequency==0U ? 0.0F :
                    1000.0F/static_cast<float>(timestamp_frequency);
                const auto elapsed=[&](std::uint32_t from,std::uint32_t to) {
                    return static_cast<float>(values[to]-values[from])*scale;
                };
                timings.available = true;
                timings.total_gpu_milliseconds=elapsed(0U,timestamp_count-1U);
                const std::uint32_t substeps=(timestamp_count-2U)/6U;
                for(std::uint32_t substep=0U;substep<substeps;++substep) {
                    const auto base=substep*6U;
                    timings.rigid_integration.total_milliseconds+=elapsed(base,base+1U);
                    timings.rigid_world_bounds.total_milliseconds+=elapsed(base+1U,base+2U);
                    timings.rigid_contact_evaluation.total_milliseconds+=elapsed(base+2U,base+3U);
                    timings.rigid_pair_compaction.total_milliseconds+=elapsed(base+3U,base+4U);
                    timings.rigid_contact_solve.total_milliseconds+=elapsed(base+4U,base+6U);
                }
                timings.rigid_integration.launch_count=substeps;
                timings.rigid_world_bounds.launch_count=substeps;
                timings.rigid_contact_evaluation.launch_count=substeps*(use_mesh_bvh ? 2U : 1U);
                timings.rigid_pair_compaction.launch_count=timed_compaction_launches;
                timings.rigid_contact_solve.launch_count=timed_solver_launches;
                timings.rigid_contact_generation={
                    timings.rigid_world_bounds.total_milliseconds+
                    timings.rigid_contact_evaluation.total_milliseconds+
                    timings.rigid_pair_compaction.total_milliseconds,
                    timings.rigid_world_bounds.launch_count+
                    timings.rigid_contact_evaluation.launch_count+
                    timings.rigid_pair_compaction.launch_count};
                timings.rigid_input_clear={elapsed(timestamp_count-2U,timestamp_count-1U),1U};
                timestamp_readback->Unmap(0U, nullptr);
            }
        }
        host_results_current = true;
        return ok();
    }

    [[nodiscard]] std::uint32_t dense_index(RigidBodyId id) const noexcept {
        if (id.index >= body_slots.size()) return UINT32_MAX;
        const BodySlot &slot = body_slots[id.index];
        return slot.alive && slot.generation == id.generation
            ? slot.dense : UINT32_MAX;
    }

    [[nodiscard]] bool valid_mesh(TriangleMeshId id) const noexcept {
        return id.index < mesh_records.size() && mesh_records[id.index].alive &&
            mesh_records[id.index].generation == id.generation;
    }

    [[nodiscard]] bool valid_constraint(RigidConstraintId id) const noexcept {
        return id.index < host_constraints.size() &&
            host_constraints[id.index].alive != 0U &&
            host_constraints[id.index].generation == id.generation;
    }
};

FrameToken::FrameToken() noexcept
    : impl_(new (std::nothrow) Impl{}) {
    if (impl_) impl_->event_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
}
FrameToken::~FrameToken() = default;
FrameToken::FrameToken(FrameToken &&) noexcept = default;
FrameToken &FrameToken::operator=(FrameToken &&) noexcept = default;

bool FrameToken::pending() const noexcept {
    return impl_ != nullptr && impl_->fence &&
        impl_->fence->GetCompletedValue() < impl_->value;
}
bool FrameToken::ready() const noexcept { return !pending(); }
Status FrameToken::wait() noexcept {
    if (!impl_ || !impl_->fence) return ok();
    if (impl_->fence->GetCompletedValue() == UINT64_MAX) {
        const HRESULT removed = impl_->device
            ? impl_->device->GetDeviceRemovedReason() : DXGI_ERROR_DEVICE_REMOVED;
        return fail(StatusCode::device_removed, "The D3D12 device was removed",
                    removed);
    }
    if (!pending()) return ok();
    HRESULT hr = impl_->fence->SetEventOnCompletion(impl_->value,
                                                    impl_->event_handle);
    if (FAILED(hr))
        return fail(StatusCode::d3d12_failure, "Could not arm frame fence", hr);
    if (WaitForSingleObject(impl_->event_handle, INFINITE) != WAIT_OBJECT_0)
        return fail(StatusCode::d3d12_failure, "Waiting for frame failed",
                    HRESULT_FROM_WIN32(GetLastError()));
    if (impl_->fence->GetCompletedValue() == UINT64_MAX) {
        const HRESULT removed = impl_->device
            ? impl_->device->GetDeviceRemovedReason() : DXGI_ERROR_DEVICE_REMOVED;
        return fail(StatusCode::device_removed, "The D3D12 device was removed",
                    removed);
    }
    return ok();
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
    if (context.device == nullptr && context.direct_queue != nullptr)
        return invalid("A direct queue requires its D3D12 device");
    try {
        auto impl = std::make_unique<Impl>();
        impl->options = options;
        D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_11_0;
        if (context.device != nullptr) {
            impl->device = static_cast<ID3D12Device *>(context.device);
        } else {
            ComPtr<IDXGIFactory6> factory;
            HRESULT hr = CreateDXGIFactory2(0U,
                IID_PPV_ARGS(factory.ReleaseAndGetAddressOf()));
            if (FAILED(hr))
                return fail(StatusCode::d3d12_failure,
                            "Could not create DXGI factory", hr);
            constexpr D3D_FEATURE_LEVEL levels[] = {
                D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_0,
                D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
            for (UINT adapter_index = 0U; !impl->device; ++adapter_index) {
                ComPtr<IDXGIAdapter1> adapter;
                hr = factory->EnumAdapterByGpuPreference(
                    adapter_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf()));
                if (hr == DXGI_ERROR_NOT_FOUND) break;
                if (FAILED(hr)) continue;
                DXGI_ADAPTER_DESC1 description{};
                if (FAILED(adapter->GetDesc1(&description)) ||
                    (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0U)
                    continue;
                ComPtr<ID3D12Device> candidate_device;
                D3D_FEATURE_LEVEL candidate_level = D3D_FEATURE_LEVEL_11_0;
                for (D3D_FEATURE_LEVEL candidate : levels) {
                    hr = D3D12CreateDevice(adapter.Get(), candidate,
                        IID_PPV_ARGS(candidate_device.ReleaseAndGetAddressOf()));
                    if (SUCCEEDED(hr)) { candidate_level = candidate; break; }
                }
                if (!candidate_device) continue;
                D3D12_FEATURE_DATA_D3D12_OPTIONS candidate_binding{};
                D3D12_FEATURE_DATA_SHADER_MODEL candidate_shader{
                    D3D_SHADER_MODEL_5_1};
                if (FAILED(candidate_device->CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS, &candidate_binding,
                        sizeof(candidate_binding))) ||
                    FAILED(candidate_device->CheckFeatureSupport(
                        D3D12_FEATURE_SHADER_MODEL, &candidate_shader,
                        sizeof(candidate_shader))) ||
                    candidate_shader.HighestShaderModel < D3D_SHADER_MODEL_5_1 ||
                    (candidate_binding.ResourceBindingTier <
                         D3D12_RESOURCE_BINDING_TIER_2 &&
                     candidate_level < D3D_FEATURE_LEVEL_11_1))
                    continue;
                feature_level = candidate_level;
                impl->device = std::move(candidate_device);
            }
            if (!impl->device)
                return fail(StatusCode::unsupported_adapter,
                    "No qualifying hardware D3D12 adapter was found");
        }

        ComPtr<IDXGIFactory6> adapter_factory;
        if (SUCCEEDED(CreateDXGIFactory2(0U,
                IID_PPV_ARGS(adapter_factory.ReleaseAndGetAddressOf())))) {
            ComPtr<IDXGIAdapter1> selected_adapter;
            if (SUCCEEDED(adapter_factory->EnumAdapterByLuid(
                    impl->device->GetAdapterLuid(),
                    IID_PPV_ARGS(selected_adapter.ReleaseAndGetAddressOf())))) {
                DXGI_ADAPTER_DESC1 selected_description{};
                if (SUCCEEDED(selected_adapter->GetDesc1(&selected_description)))
                    impl->use_mesh_bvh =
                        (selected_description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0U;
            }
        }
        constexpr D3D_FEATURE_LEVEL requested_levels[] = {
            D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        D3D12_FEATURE_DATA_FEATURE_LEVELS level_query{
            static_cast<UINT>(std::size(requested_levels)), requested_levels,
            D3D_FEATURE_LEVEL_11_0};
        if (SUCCEEDED(impl->device->CheckFeatureSupport(
                D3D12_FEATURE_FEATURE_LEVELS, &level_query,
                sizeof(level_query))))
            feature_level = level_query.MaxSupportedFeatureLevel;
        D3D12_FEATURE_DATA_D3D12_OPTIONS binding{};
        HRESULT hr = impl->device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS, &binding, sizeof(binding));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not query binding tier");
        D3D12_FEATURE_DATA_SHADER_MODEL shader_model{D3D_SHADER_MODEL_5_1};
        hr = impl->device->CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model));
        if (FAILED(hr) || shader_model.HighestShaderModel < D3D_SHADER_MODEL_5_1)
            return fail(StatusCode::unsupported_adapter,
                        "Shader Model 5.1 is required", hr);
        const bool has_64_uavs =
            binding.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_2 ||
            feature_level >= D3D_FEATURE_LEVEL_11_1;
        if (!has_64_uavs)
            return fail(StatusCode::unsupported_adapter,
                "Adapter lacks 64 UAV slots (requires binding tier 2 or FL11_1)");

        if (context.direct_queue != nullptr) {
            impl->queue = static_cast<ID3D12CommandQueue *>(context.direct_queue);
            if (impl->queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
                return invalid("Native queue must be a D3D12 direct queue");
            ComPtr<ID3D12Device> queue_device;
            hr = impl->queue->GetDevice(
                IID_PPV_ARGS(queue_device.ReleaseAndGetAddressOf()));
            if (FAILED(hr) || queue_device.Get() != impl->device.Get())
                return invalid("Native queue belongs to a different D3D12 device");
        } else {
            D3D12_COMMAND_QUEUE_DESC queue_desc{};
            queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            hr = impl->device->CreateCommandQueue(&queue_desc,
                IID_PPV_ARGS(impl->queue.ReleaseAndGetAddressOf()));
            if (FAILED(hr)) return impl->hr_status(hr, "Could not create direct queue");
        }
        hr = impl->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(impl->allocator.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create command allocator");
        hr = impl->device->CreateCommandList(0U, D3D12_COMMAND_LIST_TYPE_DIRECT,
            impl->allocator.Get(), nullptr,
            IID_PPV_ARGS(impl->commands.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create command list");
        impl->commands->SetName(L"Parallel Mater physics");
        impl->commands->Close();
        hr = impl->device->CreateFence(0U, D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(impl->fence.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create fence");
        impl->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (impl->fence_event == nullptr)
            return fail(StatusCode::d3d12_failure, "Could not create fence event",
                        HRESULT_FROM_WIN32(GetLastError()));

        impl->body_slots.resize(options.rigid_body_capacity);
        impl->host_states.reserve(options.rigid_body_capacity);
        impl->host_parameters.reserve(options.rigid_body_capacity);
        impl->host_forces.reserve(options.rigid_body_capacity);
        impl->host_torques.reserve(options.rigid_body_capacity);
        impl->host_ids.reserve(options.rigid_body_capacity);
        impl->host_constraints.resize(options.rigid_constraint_capacity);
        for (ConstraintResource &item : impl->host_constraints)
            item.generation = 1U;
        impl->mesh_records.resize(options.triangle_mesh_capacity);
        impl->host_meshes.resize(options.triangle_mesh_capacity);
        impl->host_mesh_leaf_infos.resize(options.triangle_mesh_capacity);
        const std::uint64_t pair_capacity =
            static_cast<std::uint64_t>(options.rigid_body_capacity) *
            options.rigid_body_capacity;
        if (pair_capacity > UINT32_MAX / 72U)
            return capacity("Rigid pair scratch capacity exceeds D3D12 limits");

        const struct BufferRequest { std::uint32_t count, stride; GpuBuffer *buffer; }
            requests[] = {
                {options.rigid_body_capacity, sizeof(RigidBodyState), &impl->states},
                {options.rigid_body_capacity, sizeof(RigidParameters), &impl->parameters},
                {options.rigid_body_capacity, sizeof(Vec3), &impl->forces},
                {options.rigid_body_capacity, sizeof(Vec3), &impl->torques},
                {options.rigid_body_capacity, sizeof(RigidBodyState), &impl->previous_states},
                {options.rigid_constraint_capacity, sizeof(ConstraintResource), &impl->constraints},
                {options.rigid_constraint_capacity, sizeof(ConstraintGeometry), &impl->constraint_geometry},
                {options.triangle_mesh_capacity, sizeof(MeshInfo), &impl->meshes},
                {options.rigid_body_capacity, sizeof(RigidBodyId), &impl->body_ids},
                {options.contact_capacity, sizeof(RigidContactEvent), &impl->contact_events},
                {33U, sizeof(std::uint32_t), &impl->counters},
                {options.rigid_body_capacity, sizeof(RigidCompound), &impl->compounds},
                {1U, sizeof(Vec3), &impl->mesh_vertices},
                {1U, sizeof(std::uint32_t), &impl->mesh_indices},
                {1U, sizeof(BvhNode), &impl->mesh_bvh_nodes},
                {options.triangle_mesh_capacity, sizeof(MeshLeafInfo),
                 &impl->mesh_leaf_infos},
                {1U, sizeof(std::uint32_t), &impl->mesh_bvh_leaves},
                {1U, sizeof(CollisionPlane), &impl->mesh_solid_planes},
                {1U, sizeof(Vec3), &impl->mesh_shell_normals},
                {static_cast<std::uint32_t>(pair_capacity * 8U),
                 sizeof(CachedContact), &impl->contact_cache_rows},
                {static_cast<std::uint32_t>(pair_capacity),
                 sizeof(ContactCacheHeader), &impl->contact_cache_headers}};
        for (const BufferRequest &request : requests) {
            Status status = impl->make_buffer(request.count, request.stride,
                                              *request.buffer);
            if (!status) return status;
        }
        Status scratch_status = impl->make_scratch_buffer(
            static_cast<std::uint32_t>(pair_capacity),
            sizeof(ContactManifold), impl->manifolds);
        if (!scratch_status) return scratch_status;
        scratch_status = impl->make_scratch_buffer(
            static_cast<std::uint32_t>(pair_capacity * 72U),
            sizeof(Vec3), impl->face_clip_scratch);
        if (!scratch_status) return scratch_status;
        scratch_status = impl->make_scratch_buffer(
            options.rigid_body_capacity,
            sizeof(HingeContactFrame), impl->hinge_frames);
        if (!scratch_status) return scratch_status;
        scratch_status = impl->make_scratch_buffer(
            static_cast<std::uint32_t>(pair_capacity),
            sizeof(std::uint32_t), impl->pair_colors);
        if (!scratch_status) return scratch_status;
        scratch_status = impl->make_scratch_buffer(
            options.rigid_body_capacity,
            sizeof(std::uint32_t), impl->color_owners);
        if (!scratch_status) return scratch_status;

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        scratch_status = impl->make_scratch_buffer(
            static_cast<std::uint32_t>(pair_capacity * 2U),
            sizeof(std::uint32_t), impl->active_pairs);
        if (!scratch_status) return scratch_status;
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        scratch_status=impl->make_scratch_buffer(
            static_cast<std::uint32_t>((pair_capacity+63U)/64U+1U),
            sizeof(std::uint32_t),impl->compact_blocks);
        if(!scratch_status) return scratch_status;
        heap_desc.NumDescriptors = 69U;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = impl->device->CreateDescriptorHeap(&heap_desc,
            IID_PPV_ARGS(impl->descriptors.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create UAV heap");
        impl->descriptor_size = impl->device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor =
            impl->descriptors->GetCPUDescriptorHandleForHeapStart();
        const GpuBuffer *actual[] = {&impl->states, &impl->parameters,
            &impl->forces, &impl->torques, &impl->previous_states,
            &impl->constraints, &impl->constraint_geometry, &impl->body_ids,
            &impl->contact_events, &impl->counters, &impl->compounds,
            nullptr, &impl->manifolds, &impl->face_clip_scratch,
            &impl->contact_cache_rows, &impl->contact_cache_headers,
            &impl->hinge_frames, &impl->pair_colors, &impl->color_owners,
            &impl->active_pairs,&impl->compact_blocks};
        for (std::uint32_t slot = 0U; slot < 64U; ++slot) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
            view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            if (slot < std::size(actual) && actual[slot] != nullptr) {
                view.Format = DXGI_FORMAT_UNKNOWN;
                view.Buffer.NumElements = actual[slot]->elements;
                view.Buffer.StructureByteStride = actual[slot]->stride;
                impl->device->CreateUnorderedAccessView(
                    actual[slot]->resource.Get(), nullptr, &view, descriptor);
            } else {
                view.Format = DXGI_FORMAT_R32_TYPELESS;
                view.Buffer.NumElements = 1U;
                view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                impl->device->CreateUnorderedAccessView(nullptr, nullptr,
                                                        &view, descriptor);
            }
            descriptor.ptr += impl->descriptor_size;
        }
        impl->write_srv_descriptor(64U, impl->mesh_bvh_nodes);
        impl->write_srv_descriptor(65U, impl->mesh_leaf_infos);
        impl->write_srv_descriptor(66U, impl->mesh_bvh_leaves);
        impl->write_srv_descriptor(67U, impl->mesh_solid_planes);
        impl->write_srv_descriptor(68U, impl->mesh_shell_normals);

        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[0].NumDescriptors = 64U;
        ranges[0].BaseShaderRegister = 0U;
        ranges[0].RegisterSpace = 0U;
        ranges[0].OffsetInDescriptorsFromTableStart = 0U;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[1].NumDescriptors = 5U;
        ranges[1].BaseShaderRegister = 3U;
        ranges[1].RegisterSpace = 0U;
        ranges[1].OffsetInDescriptorsFromTableStart = 64U;
        D3D12_ROOT_PARAMETER root_parameters[5]{};
        root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_parameters[0].DescriptorTable = {2U, ranges};
        root_parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        root_parameters[1].Constants = {0U, 0U,
            static_cast<UINT>(sizeof(StepConstants) / sizeof(std::uint32_t))};
        root_parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        for (std::uint32_t parameter = 2U; parameter < 5U; ++parameter) {
            root_parameters[parameter].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_SRV;
            root_parameters[parameter].Descriptor.ShaderRegister = parameter - 2U;
            root_parameters[parameter].Descriptor.RegisterSpace = 0U;
            root_parameters[parameter].ShaderVisibility =
                D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.NumParameters = 5U;
        root_desc.pParameters = root_parameters;
        ComPtr<ID3DBlob> root_blob;
        ComPtr<ID3DBlob> root_error;
        hr = D3D12SerializeRootSignature(&root_desc,
            D3D_ROOT_SIGNATURE_VERSION_1_0,
            root_blob.ReleaseAndGetAddressOf(), root_error.ReleaseAndGetAddressOf());
        if (FAILED(hr)) return impl->hr_status(hr, "Could not serialize root signature");
        hr = impl->device->CreateRootSignature(0U, root_blob->GetBufferPointer(),
            root_blob->GetBufferSize(),
            IID_PPV_ARGS(impl->root_signature.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create root signature");
        const auto make_pipeline = [&](const unsigned char *shader,
                                       std::size_t size,
                                       ComPtr<ID3D12PipelineState> &pipeline) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = impl->root_signature.Get();
            desc.CS = {shader, size};
            return impl->device->CreateComputePipelineState(
                &desc, IID_PPV_ARGS(pipeline.ReleaseAndGetAddressOf()));
        };
        hr = make_pipeline(pm_d3d12_integrate, sizeof(pm_d3d12_integrate),
                           impl->integrate_pipeline);
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create integration pipeline");
        hr = make_pipeline(pm_d3d12_prepare, sizeof(pm_d3d12_prepare),
                           impl->prepare_pipeline);
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create rigid preparation pipeline");
        if(impl->use_mesh_bvh) {
        hr = make_pipeline(pm_d3d12_generate, sizeof(pm_d3d12_generate),
                           impl->generate_pipeline);
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create contact pipeline");
        hr = make_pipeline(pm_d3d12_generate_guided,
                           sizeof(pm_d3d12_generate_guided),
                           impl->generate_guided_pipeline);
        if (FAILED(hr))
            return impl->hr_status(hr,
                                   "Could not create guided contact pipeline");
        } else {
        hr = make_pipeline(pm_d3d12_generate_warp,
                           sizeof(pm_d3d12_generate_warp),
                           impl->generate_warp_pipeline);
        if (FAILED(hr))
            return impl->hr_status(hr,
                                   "Could not create WARP contact pipeline");
        }
        hr=make_pipeline(pm_d3d12_color,sizeof(pm_d3d12_color),impl->color_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create coloring pipeline");
        hr=make_pipeline(pm_d3d12_finalize,sizeof(pm_d3d12_finalize),impl->finalize_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create finalization pipeline");
        hr=make_pipeline(pm_d3d12_contact_prepare,sizeof(pm_d3d12_contact_prepare),impl->contact_prepare_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create contact preparation pipeline");
        hr=make_pipeline(pm_d3d12_contact_events,sizeof(pm_d3d12_contact_events),impl->contact_events_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create contact events pipeline");
        hr = make_pipeline(pm_d3d12_clear, sizeof(pm_d3d12_clear),
                           impl->clear_pipeline);
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create clear pipeline");
        hr=make_pipeline(pm_d3d12_compact_count,sizeof(pm_d3d12_compact_count),impl->compact_count_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create compaction count pipeline");
        hr=make_pipeline(pm_d3d12_compact_prefix,sizeof(pm_d3d12_compact_prefix),impl->compact_prefix_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create compaction prefix pipeline");
        hr=make_pipeline(pm_d3d12_compact_scatter,sizeof(pm_d3d12_compact_scatter),impl->compact_scatter_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create compaction scatter pipeline");
        if(impl->use_mesh_bvh) {
        hr=make_pipeline(pm_d3d12_contacts,sizeof(pm_d3d12_contacts),impl->contacts_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create contact solve pipeline");
        } else {
        hr=make_pipeline(pm_d3d12_contacts_warp,sizeof(pm_d3d12_contacts_warp),impl->contacts_warp_pipeline);
        if(FAILED(hr)) return impl->hr_status(hr,"Could not create software contact solve pipeline");
        }

        D3D12_QUERY_HEAP_DESC query_desc{};
        query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        query_desc.Count = 6U*1024U+2U;
        hr = impl->device->CreateQueryHeap(&query_desc,
            IID_PPV_ARGS(impl->timestamp_heap.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create timestamp heap");
        const D3D12_HEAP_PROPERTIES readback_heap =
            heap_properties(D3D12_HEAP_TYPE_READBACK);
        const D3D12_RESOURCE_DESC timestamp_desc = buffer_description(
            query_desc.Count * sizeof(std::uint64_t));
        hr = impl->device->CreateCommittedResource(&readback_heap,
            D3D12_HEAP_FLAG_NONE, &timestamp_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(impl->timestamp_readback.ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return impl->hr_status(hr, "Could not create timestamp readback");
        impl->queue->GetTimestampFrequency(&impl->timestamp_frequency);
        Status status = impl->upload_host_data();
        if (!status) return status;
        output.impl_ = std::move(impl);
        return ok();
    } catch (const std::bad_alloc &) {
        return fail(StatusCode::out_of_memory, "Host allocation failed");
    } catch (...) {
        return fail(StatusCode::internal_error, "Unexpected D3D12 initialization failure");
    }
}

Status World::add_triangle_mesh(HostSpan<const Vec3> vertices,
                                HostSpan<const std::uint32_t> indices,
                                TriangleMeshId &output) noexcept {
    if (!impl_) return invalid("World is not initialized");
    output = {};
    if (!impl_->idle()) return busy_status();
    if (vertices.size < 3U || indices.size < 3U || indices.size % 3U != 0U)
        return invalid("Mesh requires vertices and triangle indices");
    for (std::uint64_t i = 0; i < vertices.size; ++i)
        if (!finite(vertices.data[i])) return invalid("Mesh vertex is not finite");
    for (std::uint64_t i = 0; i < indices.size; ++i)
        if (indices.data[i] >= vertices.size) return invalid("Mesh index is out of range");
    for (std::uint64_t i = 0; i < indices.size; i += 3U) {
        const Vec3 edge_a = subtract(vertices.data[indices.data[i + 1U]],
                                     vertices.data[indices.data[i]]);
        const Vec3 edge_b = subtract(vertices.data[indices.data[i + 2U]],
                                     vertices.data[indices.data[i]]);
        if (length_squared(cross(edge_a, edge_b)) <= 1.0e-12F)
            return invalid("Mesh contains a degenerate triangle");
    }
    try {
        Status status = impl_->sync_results(false);
        if (!status) return status;
        std::uint32_t slot = UINT32_MAX;
        for (std::uint32_t i = 0; i < impl_->mesh_records.size(); ++i)
            if (!impl_->mesh_records[i].alive) { slot = i; break; }
        if (slot == UINT32_MAX) return capacity("Triangle mesh capacity exceeded");
        MeshRecord &record = impl_->mesh_records[slot];
        std::vector<Vec3> vertex_copy(vertices.data, vertices.data + vertices.size);
        std::vector<std::uint32_t> index_copy(indices.data, indices.data + indices.size);
        Vec3 minimum = vertex_copy.front();
        Vec3 maximum = minimum;
        for (Vec3 value : vertex_copy) {
            minimum.x = std::min(minimum.x, value.x);
            minimum.y = std::min(minimum.y, value.y);
            minimum.z = std::min(minimum.z, value.z);
            maximum.x = std::max(maximum.x, value.x);
            maximum.y = std::max(maximum.y, value.y);
            maximum.z = std::max(maximum.z, value.z);
        }
        record.vertices = std::move(vertex_copy);
        record.indices = std::move(index_copy);
        build_mesh_bvh(record);
        record.solid_planes =
            closed_convex_planes(record.vertices, record.indices);
        record.shell_normals =
            convex_shell_normals(record.vertices, record.indices);
        record.alive = true;
        MeshInfo &gpu = impl_->host_meshes[slot];
        gpu = {};
        gpu.minimum = minimum;
        gpu.maximum = maximum;
        gpu.bounding_center = multiply(add(minimum, maximum), 0.5F);
        for (Vec3 value : record.vertices)
            gpu.radius = std::max(gpu.radius, std::sqrt(length_squared(
                subtract(value, gpu.bounding_center))));
        gpu.alive = 1U;
        gpu.generation = record.generation;
        Status rebuild_status = impl_->rebuild_mesh_buffers();
        if (!rebuild_status) {
            record.vertices.clear();
            record.indices.clear();
            record.bvh_nodes.clear();
            record.solid_planes.clear();
            record.shell_normals.clear();
            record.alive = false;
            gpu = {};
            gpu.generation = record.generation;
            return rebuild_status;
        }
        output = {slot, record.generation};
        ++impl_->revision;
        return ok();
    } catch (const std::bad_alloc &) {
        return fail(StatusCode::out_of_memory, "Mesh allocation failed");
    } catch (...) {
        return fail(StatusCode::internal_error, "Unexpected mesh failure");
    }
}

Status World::add_triangle_mesh(BufferSpan<const Vec3> vertices,
                                BufferSpan<const std::uint32_t> indices,
                                TriangleMeshId &output) noexcept {
    output = {};
    if (!impl_) return invalid("World is not initialized");
    if (vertices.resource == nullptr || indices.resource == nullptr ||
        vertices.size == 0U || indices.size == 0U)
        return invalid("D3D12 mesh spans are empty");
    try {
        std::vector<Vec3> host_vertices(static_cast<std::size_t>(vertices.size));
        std::vector<std::uint32_t> host_indices(static_cast<std::size_t>(indices.size));
        Status status = copy_d3d12_buffer_to_host(vertices.resource,
            vertices.byte_offset, vertices.size * sizeof(Vec3), host_vertices.data());
        if (!status) return status;
        status = copy_d3d12_buffer_to_host(indices.resource, indices.byte_offset,
            indices.size * sizeof(std::uint32_t), host_indices.data());
        if (!status) return status;
        return add_triangle_mesh(
            HostSpan<const Vec3>{host_vertices.data(), host_vertices.size()},
            HostSpan<const std::uint32_t>{host_indices.data(), host_indices.size()},
            output);
    } catch (const std::bad_alloc &) {
        return fail(StatusCode::out_of_memory, "Mesh readback allocation failed");
    }
}

Status World::remove_triangle_mesh(TriangleMeshId mesh) noexcept {
    if (!impl_) return invalid("World is not initialized");
    if (!impl_->idle()) return busy_status();
    if (!impl_->valid_mesh(mesh)) return invalid_handle("Invalid triangle mesh");
    for (const RigidParameters &body : impl_->host_parameters)
        if (body.mesh_index == mesh.index)
            return invalid("Cannot remove a mesh used by a rigid body");
    MeshRecord &record = impl_->mesh_records[mesh.index];
    record.alive = false;
    record.vertices.clear();
    record.indices.clear();
    record.bvh_nodes.clear();
    record.solid_planes.clear();
    record.shell_normals.clear();
    ++record.generation;
    if (record.generation == 0U) ++record.generation;
    impl_->host_meshes[mesh.index] = {};
    impl_->host_meshes[mesh.index].generation = record.generation;
    Status rebuild_status = impl_->rebuild_mesh_buffers();
    if (!rebuild_status) return rebuild_status;
    ++impl_->revision;
    return ok();
}

Status World::add_rigid_body(RigidBodyOptions options,
                             RigidBodyId &output) noexcept {
    output = {};
    if (!impl_) return invalid("World is not initialized");
    if (!impl_->idle()) return busy_status();
    if (!impl_->valid_mesh(options.mesh)) return invalid_handle("Invalid body mesh");
    if (static_cast<std::uint32_t>(options.motion) >
            static_cast<std::uint32_t>(MotionType::dynamic) ||
        !valid_state(options.initial_state) || !finite(options.mass) ||
        !finite(options.inertia_diagonal) || !finite(options.friction) ||
        !finite(options.restitution) || !finite(options.linear_damping) ||
        !finite(options.angular_damping) || !finite(options.maximum_linear_speed) ||
        !finite(options.maximum_angular_speed) || !finite(options.collision_margin) ||
        options.mass <= 0.0F || options.friction < 0.0F ||
        options.restitution < 0.0F || options.linear_damping < 0.0F ||
        options.angular_damping < 0.0F || options.maximum_linear_speed <= 0.0F ||
        options.maximum_angular_speed <= 0.0F || options.collision_margin < 0.0F)
        return invalid("Invalid rigid body options");
    Status status = impl_->sync_results(false);
    if (!status) return status;
    std::uint32_t slot_index = UINT32_MAX;
    for (std::uint32_t i = 0; i < impl_->body_slots.size(); ++i)
        if (!impl_->body_slots[i].alive) { slot_index = i; break; }
    if (slot_index == UINT32_MAX) return capacity("Rigid body capacity exceeded");
    const std::uint32_t dense = static_cast<std::uint32_t>(impl_->host_states.size());
    BodySlot &slot = impl_->body_slots[slot_index];
    slot.alive = true;
    slot.dense = dense;
    const RigidBodyId candidate{slot_index, slot.generation};
    RigidParameters parameters{};
    parameters.motion = static_cast<std::uint32_t>(options.motion);
    parameters.inverse_mass = options.motion == MotionType::dynamic
        ? 1.0F / options.mass : 0.0F;
    Vec3 inertia = options.inertia_diagonal;
    if (length_squared(inertia) <= 1.0e-16F) {
        const MeshInfo &mesh = impl_->host_meshes[options.mesh.index];
        const Vec3 size = subtract(mesh.maximum, mesh.minimum);
        inertia = {options.mass * (size.y*size.y + size.z*size.z) / 12.0F,
                   options.mass * (size.x*size.x + size.z*size.z) / 12.0F,
                   options.mass * (size.x*size.x + size.y*size.y) / 12.0F};
    }
    if (inertia.x < 0.0F || inertia.y < 0.0F || inertia.z < 0.0F) {
        slot.alive = false;
        return invalid("Inertia diagonal cannot be negative");
    }
    parameters.inverse_inertia = options.motion == MotionType::dynamic
        ? Vec3{inertia.x > 0.0F ? 1.0F/inertia.x : 0.0F,
               inertia.y > 0.0F ? 1.0F/inertia.y : 0.0F,
               inertia.z > 0.0F ? 1.0F/inertia.z : 0.0F} : Vec3{};
    parameters.linear_damping = options.linear_damping;
    parameters.angular_damping = options.angular_damping;
    parameters.maximum_linear_speed = options.maximum_linear_speed;
    parameters.maximum_angular_speed = options.maximum_angular_speed;
    parameters.mesh_index = options.mesh.index;
    parameters.friction = options.friction;
    parameters.restitution = options.restitution;
    parameters.collision_margin = options.collision_margin;
    // All five vectors reserve the fixed body capacity during World creation,
    // so these POD insertions cannot allocate or partially fail here.
    impl_->host_states.push_back(options.initial_state);
    impl_->host_parameters.push_back(parameters);
    impl_->host_forces.push_back({});
    impl_->host_torques.push_back({});
    impl_->host_ids.push_back(candidate);
    output = candidate;
    ++impl_->revision;
    return impl_->upload_host_data();
}

Status World::remove_rigid_body(RigidBodyId body) noexcept {
    if (!impl_) return invalid("World is not initialized");
    if (!impl_->idle()) return busy_status();
    Status status = impl_->sync_results(false);
    if (!status) return status;
    const std::uint32_t dense = impl_->dense_index(body);
    if (dense == UINT32_MAX) return invalid_handle("Invalid rigid body");
    for (const ConstraintResource &joint : impl_->host_constraints)
        if (joint.alive != 0U && (joint.body_a == dense || joint.body_b == dense))
            return invalid("Remove constraints before removing their rigid bodies");
    const std::uint32_t last = static_cast<std::uint32_t>(impl_->host_states.size()-1U);
    if (dense != last) {
        impl_->host_states[dense] = impl_->host_states[last];
        impl_->host_parameters[dense] = impl_->host_parameters[last];
        impl_->host_forces[dense] = impl_->host_forces[last];
        impl_->host_torques[dense] = impl_->host_torques[last];
        impl_->host_ids[dense] = impl_->host_ids[last];
        impl_->body_slots[impl_->host_ids[dense].index].dense = dense;
        for (ConstraintResource &joint : impl_->host_constraints) {
            if (joint.alive == 0U) continue;
            if (joint.body_a == last) joint.body_a = dense;
            if (joint.body_b == last) joint.body_b = dense;
        }
    }
    impl_->host_states.pop_back();
    impl_->host_parameters.pop_back();
    impl_->host_forces.pop_back();
    impl_->host_torques.pop_back();
    impl_->host_ids.pop_back();
    BodySlot &slot = impl_->body_slots[body.index];
    slot.alive = false;
    ++slot.generation;
    if (slot.generation == 0U) ++slot.generation;
    ++impl_->revision;
    return impl_->upload_host_data();
}

Status World::set_rigid_body_state(RigidBodyId body,
                                   RigidBodyState state) noexcept {
    if (!impl_) return invalid("World is not initialized");
    if (!valid_state(state)) return invalid("Rigid body state is invalid");
    if (!impl_->idle()) return busy_status();
    Status status = impl_->sync_results(false);
    if (!status) return status;
    const std::uint32_t dense = impl_->dense_index(body);
    if (dense == UINT32_MAX) return invalid_handle("Invalid rigid body");
    impl_->host_states[dense] = state;
    ++impl_->revision;
    return impl_->upload_host_data();
}

Status World::set_kinematic_target(RigidBodyId body,
                                   RigidBodyState target) noexcept {
    if (!impl_) return invalid("World is not initialized");
    if (!valid_state(target)) return invalid("Kinematic target is invalid");
    if (!impl_->idle()) return busy_status();
    Status status = impl_->sync_results(false);
    if (!status) return status;
    const std::uint32_t dense = impl_->dense_index(body);
    if (dense == UINT32_MAX) return invalid_handle("Invalid rigid body");
    if (impl_->host_parameters[dense].motion !=
        static_cast<std::uint32_t>(MotionType::kinematic))
        return invalid("Body is not kinematic");
    impl_->host_parameters[dense].kinematic_target = target;
    impl_->host_parameters[dense].has_kinematic_target = 1U;
    return impl_->upload_host_data();
}

Status World::apply_force(RigidBodyId body, Vec3 force,
                          Vec3 world_point) noexcept {
    if (!impl_) return invalid("World is not initialized");
    if (!finite(force) || !finite(world_point)) return invalid("Force is not finite");
    if (!impl_->idle()) return busy_status();
    Status status = impl_->sync_results(false);
    if (!status) return status;
    const std::uint32_t dense = impl_->dense_index(body);
    if (dense == UINT32_MAX) return invalid_handle("Invalid rigid body");
    impl_->host_forces[dense] = add(impl_->host_forces[dense], force);
    impl_->host_torques[dense] = add(impl_->host_torques[dense],
        cross(subtract(world_point, impl_->host_states[dense].position), force));
    return impl_->upload_host_data();
}

Status World::apply_central_acceleration(HostSpan<RigidBodyId> bodies,
                                         Vec3 acceleration) noexcept {
    if (!impl_) return invalid("World is not initialized");
    if (!finite(acceleration)) return invalid("Acceleration is not finite");
    if (!impl_->idle()) return busy_status();
    Status status = impl_->sync_results(false);
    if (!status) return status;
    for (std::uint64_t i=0;i<bodies.size;++i)
        if (impl_->dense_index(bodies.data[i]) == UINT32_MAX)
            return invalid_handle("Acceleration contains an invalid body");
    for (std::uint64_t i=0;i<bodies.size;++i) {
        const std::uint32_t dense=impl_->dense_index(bodies.data[i]);
        const float inverse_mass=impl_->host_parameters[dense].inverse_mass;
        if (inverse_mass>0.0F)
            impl_->host_forces[dense]=add(impl_->host_forces[dense],
                                          multiply(acceleration,1.0F/inverse_mass));
    }
    return impl_->upload_host_data();
}

Status World::apply_impulse(RigidBodyId body, Vec3 impulse,
                            Vec3 world_point) noexcept {
    if (!impl_) return invalid("World is not initialized");
    if (!finite(impulse) || !finite(world_point)) return invalid("Impulse is not finite");
    if (!impl_->idle()) return busy_status();
    Status status=impl_->sync_results(false);
    if (!status) return status;
    const std::uint32_t dense=impl_->dense_index(body);
    if (dense==UINT32_MAX) return invalid_handle("Invalid rigid body");
    RigidParameters &parameters=impl_->host_parameters[dense];
    RigidBodyState &state=impl_->host_states[dense];
    if(parameters.inverse_mass>0.0F) {
        state.linear_velocity=add(state.linear_velocity,
                                  multiply(impulse,parameters.inverse_mass));
        const Vec3 angular=cross(subtract(world_point,state.position),impulse);
        const Quaternion inverse{-state.orientation.x,-state.orientation.y,
                                 -state.orientation.z,state.orientation.w};
        const Vec3 local=rotate(inverse,angular);
        state.angular_velocity=add(state.angular_velocity,rotate(state.orientation,
            Vec3{local.x*parameters.inverse_inertia.x,
                 local.y*parameters.inverse_inertia.y,
                 local.z*parameters.inverse_inertia.z}));
    }
    return impl_->upload_host_data();
}

Status World::rigid_body_view(RigidBodyDeviceView &output) const noexcept {
    output={};
    if(!impl_) return invalid("World is not initialized");
    if(!impl_->idle()) return busy_status();
    output.ids={impl_->body_ids.resource.Get(),0U,impl_->host_ids.size()};
    output.states={impl_->states.resource.Get(),0U,impl_->host_states.size()};
    output.previous_states={impl_->previous_states.resource.Get(),0U,
                            impl_->host_states.size()};
    if(impl_->options.physics_debug.frame_capacity!=0U) {
        output.applied_forces={impl_->forces.resource.Get(),0U,impl_->host_states.size()};
        output.applied_torques={impl_->torques.resource.Get(),0U,impl_->host_states.size()};
    }
    output.revision=impl_->revision;
    return ok();
}

Status World::read_rigid_body_state(RigidBodyId body,
                                    RigidBodyState &output) const noexcept {
    output={};
    if(!impl_) return invalid("World is not initialized");
    Status status=impl_->sync_results(true);
    if(!status) return status;
    const std::uint32_t dense=impl_->dense_index(body);
    if(dense==UINT32_MAX) return invalid_handle("Invalid rigid body");
    output=impl_->host_states[dense];
    return ok();
}

Status World::add_rigid_constraint(RigidConstraintOptions options,
                                   RigidConstraintId &output) noexcept {
    output={};
    if(!impl_) return invalid("World is not initialized");
    if(!impl_->idle()) return busy_status();
    if(!valid_constraint_options(options)) return invalid("Invalid constraint options");
    Status status=impl_->sync_results(false);
    if(!status) return status;
    const std::uint32_t a=impl_->dense_index(options.body_a);
    const std::uint32_t b=impl_->dense_index(options.body_b);
    if(a==UINT32_MAX || b==UINT32_MAX || a==b)
        return invalid_handle("Constraint body handle is invalid");
    std::uint32_t slot=UINT32_MAX;
    for(std::uint32_t i=0;i<impl_->host_constraints.size();++i)
        if(impl_->host_constraints[i].alive==0U) { slot=i; break; }
    if(slot==UINT32_MAX) return capacity("Rigid constraint capacity exceeded");
    std::uint32_t generation=impl_->host_constraints[slot].generation;
    if(generation==0U) generation=1U;
    impl_->host_constraints[slot]=make_constraint(options,generation,a,b);
    output={slot,generation};
    ++impl_->revision;
    return impl_->upload_host_data();
}

Status World::update_rigid_constraint(RigidConstraintId constraint,
                                      RigidConstraintOptions options) noexcept {
    if(!impl_) return invalid("World is not initialized");
    if(!impl_->idle()) return busy_status();
    if(!valid_constraint_options(options)) return invalid("Invalid constraint options");
    Status status=impl_->sync_results(false);
    if(!status) return status;
    if(!impl_->valid_constraint(constraint)) return invalid_handle("Invalid constraint");
    const std::uint32_t a=impl_->dense_index(options.body_a);
    const std::uint32_t b=impl_->dense_index(options.body_b);
    if(a==UINT32_MAX || b==UINT32_MAX || a==b)
        return invalid_handle("Constraint body handle is invalid");
    const ConstraintResource old=impl_->host_constraints[constraint.index];
    ConstraintResource updated=make_constraint(options,constraint.generation,a,b);
    updated.broken=old.broken;
    updated.applied_impulse=old.applied_impulse;
    if(updated.broken!=0U) updated.enabled=0U;
    impl_->host_constraints[constraint.index]=updated;
    ++impl_->revision;
    return impl_->upload_host_data();
}

Status World::remove_rigid_constraint(RigidConstraintId constraint) noexcept {
    if(!impl_) return invalid("World is not initialized");
    if(!impl_->idle()) return busy_status();
    Status status=impl_->sync_results(false);
    if(!status) return status;
    if(!impl_->valid_constraint(constraint)) return invalid_handle("Invalid constraint");
    ConstraintResource &resource=impl_->host_constraints[constraint.index];
    const std::uint32_t generation=resource.generation+1U==0U ? 1U : resource.generation+1U;
    resource={};
    resource.generation=generation;
    ++impl_->revision;
    return impl_->upload_host_data();
}

Status World::read_rigid_constraint_state(RigidConstraintId constraint,
                                          RigidConstraintState &output) const noexcept {
    output={};
    if(!impl_) return invalid("World is not initialized");
    Status status=impl_->sync_results(true);
    if(!status) return status;
    if(!impl_->valid_constraint(constraint)) return invalid_handle("Invalid constraint");
    const ConstraintResource &resource=impl_->host_constraints[constraint.index];
    output.enabled=resource.enabled!=0U;
    output.broken=resource.broken!=0U;
    output.applied_impulse=resource.applied_impulse;
    return ok();
}

Status World::step_async(StepOptions options, FrameToken &completion) noexcept {
    if(!impl_) return invalid("World is not initialized");
    if(!completion.impl_ || completion.impl_->event_handle==nullptr)
        return fail(StatusCode::out_of_memory,"Frame token has no wait event");
    if(completion.pending()) return busy_status();
    if(!finite(options.timestep) || options.timestep<=0.0F ||
       options.substeps==0U || options.substeps>1024U || !finite(options.gravity))
        return invalid("Invalid step options");
    if(!impl_->idle()) return busy_status();
    Status status=impl_->sync_results(false);
    if(!status) return status;
    status=impl_->begin_commands();
    if(!status) return status;
    ID3D12DescriptorHeap *heaps[]={impl_->descriptors.Get()};
    impl_->commands->SetDescriptorHeaps(1U,heaps);
    impl_->commands->SetComputeRootSignature(impl_->root_signature.Get());
    impl_->commands->SetComputeRootDescriptorTable(
        0U,impl_->descriptors->GetGPUDescriptorHandleForHeapStart());
    impl_->transition(impl_->meshes,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->transition(impl_->mesh_vertices,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->transition(impl_->mesh_indices,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->transition(impl_->mesh_bvh_nodes,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->transition(impl_->mesh_leaf_infos,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->transition(impl_->mesh_bvh_leaves,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->transition(impl_->mesh_solid_planes,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->transition(impl_->mesh_shell_normals,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    impl_->commands->SetComputeRootShaderResourceView(
        2U,impl_->meshes.resource->GetGPUVirtualAddress());
    impl_->commands->SetComputeRootShaderResourceView(
        3U,impl_->mesh_vertices.resource->GetGPUVirtualAddress());
    impl_->commands->SetComputeRootShaderResourceView(
        4U,impl_->mesh_indices.resource->GetGPUVirtualAddress());
    std::uint32_t timestamp_index=0U;
    const auto stamp=[&]() {
        if(options.collect_kernel_timings)
            impl_->commands->EndQuery(impl_->timestamp_heap.Get(),
                D3D12_QUERY_TYPE_TIMESTAMP,timestamp_index++);
    };
    stamp();
    StepConstants constants{};
    constants.timestep=options.timestep/static_cast<float>(options.substeps);
    constants.gravity=options.gravity;
    constants.body_count=static_cast<std::uint32_t>(impl_->host_states.size());
    constants.constraint_capacity=impl_->options.rigid_constraint_capacity;
    constants.collect_contacts=options.collect_rigid_contacts ? 1U : 0U;
    constants.event_capacity=impl_->options.contact_capacity;
    constants.substeps=options.substeps;
    constants.use_mesh_bvh=impl_->use_mesh_bvh ? 1U : 0U;
    const UINT groups=std::max(1U,(constants.body_count+63U)/64U);
    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(constants.body_count) * constants.body_count;
    const UINT pair_groups=std::max<UINT>(
        1U,static_cast<UINT>((pair_count+63U)/64U));
    // Only unconstrained stacks with at least 32 bodies can select the
    // prepared-contact path. Every other pair returns immediately after pass
    // eight, so omitting those empty dispatches cannot change solver results.
    const bool prepared_contacts_possible=constants.body_count>=32U &&
        std::none_of(impl_->host_constraints.begin(),impl_->host_constraints.end(),
            [](const ConstraintResource &joint) { return joint.alive!=0U; });
    const std::uint32_t maximum_contact_pass=prepared_contacts_possible ? 64U : 8U;
    for(std::uint32_t substep=0;substep<options.substeps;++substep) {
        constants.substep_index=substep;
        constants.contact_epoch=++impl_->contact_epoch;
        if(constants.contact_epoch==0U)
            constants.contact_epoch=++impl_->contact_epoch;
        impl_->commands->SetComputeRoot32BitConstants(1U,
            static_cast<UINT>(sizeof(constants)/sizeof(std::uint32_t)),&constants,0U);
        impl_->commands->SetPipelineState(impl_->integrate_pipeline.Get());
        impl_->commands->Dispatch(groups,1U,1U);
        impl_->uav_barrier(impl_->states);
        impl_->uav_barrier(impl_->previous_states);
        stamp();
        impl_->uav_barrier(impl_->hinge_frames);
        impl_->commands->SetPipelineState(impl_->prepare_pipeline.Get());
        impl_->commands->Dispatch(1U,1U,1U);
        impl_->uav_barrier(impl_->states);
        impl_->uav_barrier(impl_->compounds);
        impl_->uav_barrier(impl_->hinge_frames);
        impl_->uav_barrier(impl_->color_owners);
        stamp();
        if(impl_->use_mesh_bvh) {
            impl_->commands->SetPipelineState(impl_->generate_pipeline.Get());
            impl_->commands->Dispatch(pair_groups,1U,1U);
            impl_->uav_barrier(impl_->manifolds);
            impl_->commands->SetPipelineState(
                impl_->generate_guided_pipeline.Get());
            impl_->commands->Dispatch(pair_groups,1U,1U);
        } else {
            impl_->commands->SetPipelineState(
                impl_->generate_warp_pipeline.Get());
            impl_->commands->Dispatch(pair_groups,1U,1U);
        }
        impl_->uav_barrier(impl_->manifolds);
        stamp();
        impl_->commands->SetPipelineState(impl_->compact_count_pipeline.Get());
        impl_->commands->Dispatch(pair_groups,1U,1U);
        impl_->uav_barrier(impl_->compact_blocks);
        impl_->uav_barrier(impl_->active_pairs);
        impl_->commands->SetPipelineState(impl_->compact_prefix_pipeline.Get());
        impl_->commands->Dispatch(1U,1U,1U);
        impl_->uav_barrier(impl_->compact_blocks);
        impl_->uav_barrier(impl_->counters);
        impl_->commands->SetPipelineState(impl_->compact_scatter_pipeline.Get());
        impl_->commands->Dispatch(pair_groups,1U,1U);
        impl_->uav_barrier(impl_->active_pairs);
        impl_->commands->SetPipelineState(impl_->contact_prepare_pipeline.Get());
        impl_->commands->Dispatch(groups,1U,1U);
        impl_->uav_barrier(impl_->manifolds);
        impl_->uav_barrier(impl_->face_clip_scratch);
        impl_->commands->SetPipelineState(impl_->color_pipeline.Get());
        constants.solver_phase=0U;
        constants.solver_pass_begin=0U;
        constants.solver_pass_count=0U;
        impl_->commands->SetComputeRoot32BitConstants(1U,
            static_cast<UINT>(sizeof(constants)/sizeof(std::uint32_t)),&constants,0U);
        impl_->commands->Dispatch(1U,1U,1U);
        impl_->uav_barrier(impl_->manifolds);
        impl_->uav_barrier(impl_->face_clip_scratch);
        impl_->uav_barrier(impl_->contact_events);
        impl_->uav_barrier(impl_->counters);
        impl_->uav_barrier(impl_->pair_colors);
        impl_->uav_barrier(impl_->color_owners);
        impl_->uav_barrier(impl_->active_pairs);
        if(options.collect_rigid_contacts) {
            impl_->commands->SetPipelineState(impl_->contact_events_pipeline.Get());
            impl_->commands->Dispatch(groups,1U,1U);
            impl_->uav_barrier(impl_->contact_events);
        }
        stamp();
        constants.solver_phase=impl_->use_mesh_bvh ? 1U : 3U;
        impl_->commands->SetPipelineState(impl_->use_mesh_bvh
            ? impl_->contacts_pipeline.Get() : impl_->contacts_warp_pipeline.Get());
        const UINT contact_groups=impl_->use_mesh_bvh ? std::min(32U,groups) : 1U;
        constants.solver_pass_count=impl_->use_mesh_bvh ? contact_groups*64U : 1U;
        const std::uint32_t maximum_colors=constants.body_count<9U
            ? std::min(24U,std::max(1U,constants.body_count*
                (constants.body_count>0U ? constants.body_count-1U : 0U)/2U)) : 24U;
        impl_->commands->SetComputeRoot32BitConstants(1U,
            static_cast<UINT>(sizeof(constants)/sizeof(std::uint32_t)),&constants,0U);
        for(std::uint32_t solver_pass=0U;solver_pass<=maximum_contact_pass;++solver_pass) {
            impl_->commands->SetComputeRoot32BitConstant(1U,solver_pass,
                offsetof(StepConstants,solver_pass_begin)/sizeof(std::uint32_t));
            const auto rounds=maximum_colors+1U;
            for(std::uint32_t round=0U;round<rounds;++round) {
                constants.solver_color=round+1U==rounds ? 24U : round;
                impl_->commands->SetComputeRoot32BitConstant(1U,constants.solver_color,
                    offsetof(StepConstants,solver_color)/sizeof(std::uint32_t));
                // Pairs in one color have disjoint dynamic owners. Overflow
                // retains the serial ascending-pair order in one group.
                impl_->commands->Dispatch(constants.solver_color==24U
                    ? 1U : contact_groups,1U,1U);
                D3D12_RESOURCE_BARRIER solver_barrier{};
                solver_barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
                // All writable solver buffers share the same queue.
                // One global UAV barrier orders them between color rounds.
                impl_->commands->ResourceBarrier(1U,&solver_barrier);
            }
        }
        impl_->uav_barrier(impl_->states);
        impl_->uav_barrier(impl_->manifolds);
        impl_->uav_barrier(impl_->face_clip_scratch);
        impl_->uav_barrier(impl_->contact_events);
        constants.solver_phase=2U;
        stamp();
        impl_->commands->SetPipelineState(impl_->finalize_pipeline.Get());
        constants.solver_pass_begin=0U;
        constants.solver_pass_count=0U;
        impl_->commands->SetComputeRoot32BitConstants(1U,
            static_cast<UINT>(sizeof(constants)/sizeof(std::uint32_t)),&constants,0U);
        impl_->commands->Dispatch(1U,1U,1U);
        impl_->uav_barrier(impl_->states);
        impl_->uav_barrier(impl_->constraints);
        impl_->uav_barrier(impl_->constraint_geometry);
        impl_->uav_barrier(impl_->compounds);
        impl_->uav_barrier(impl_->contact_events);
        impl_->uav_barrier(impl_->counters);
        impl_->uav_barrier(impl_->contact_cache_rows);
        impl_->uav_barrier(impl_->contact_cache_headers);
        impl_->uav_barrier(impl_->hinge_frames);
        impl_->uav_barrier(impl_->pair_colors);
        impl_->uav_barrier(impl_->color_owners);
        stamp();
    }
    impl_->commands->SetPipelineState(impl_->clear_pipeline.Get());
    impl_->commands->Dispatch(groups,1U,1U);
    impl_->uav_barrier(impl_->forces);
    impl_->uav_barrier(impl_->torques);
    if(options.collect_kernel_timings) {
        stamp();
        impl_->commands->ResolveQueryData(impl_->timestamp_heap.Get(),
            D3D12_QUERY_TYPE_TIMESTAMP,0U,timestamp_index,impl_->timestamp_readback.Get(),0U);
    }
    GpuBuffer *readbacks[]={&impl_->states,&impl_->parameters,
                            &impl_->constraints,&impl_->counters};
    for(GpuBuffer *buffer:readbacks) {
        impl_->transition(*buffer,D3D12_RESOURCE_STATE_COPY_SOURCE);
        impl_->commands->CopyBufferRegion(buffer->readback.Get(),0U,
                                          buffer->resource.Get(),0U,buffer->bytes);
        impl_->transition(*buffer,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    status=impl_->submit(false);
    if(!status) return status;
    completion.impl_->fence=impl_->fence;
    completion.impl_->device=impl_->device;
    completion.impl_->value=impl_->submitted_value;
    std::fill(impl_->host_forces.begin(),impl_->host_forces.end(),Vec3{});
    std::fill(impl_->host_torques.begin(),impl_->host_torques.end(),Vec3{});
    impl_->host_results_current=false;
    impl_->timings_requested=options.collect_kernel_timings;
    impl_->timestamp_count=timestamp_index;
    const std::uint32_t timed_colors=constants.body_count<9U
        ? std::min(24U,std::max(1U,constants.body_count*
            (constants.body_count>0U ? constants.body_count-1U : 0U)/2U)) : 24U;
    impl_->timed_solver_launches=options.substeps*((maximum_contact_pass+1U)*
        (timed_colors+1U)+1U);
    impl_->timed_compaction_launches=options.substeps*(options.collect_rigid_contacts ? 6U : 5U);
    impl_->timings={};
    impl_->timings.frame_index=++impl_->frame_index;
    ++impl_->revision;
    return ok();
}

Status World::step(StepOptions options) noexcept {
    FrameToken token;
    Status status=step_async(options,token);
    if(!status) return status;
    status=token.wait();
    if(!status) return status;
    return impl_->sync_results(true);
}

RigidContactDeviceView World::rigid_contacts() const noexcept {
    if(!impl_ || !impl_->idle()) return {};
    (void)impl_->sync_results(false);
    return {{impl_->contact_events.resource.Get(),0U,impl_->rigid_event_count},
            impl_->rigid_event_count,impl_->frame_index};
}

ContactDeviceView World::contacts() const noexcept { return {}; }

Status World::collect_step_timings(WorldStepTimings &output) const noexcept {
    output={};
    if(!impl_) return invalid("World is not initialized");
    Status status=impl_->sync_results(false);
    if(!status) return status;
    output=impl_->timings;
    return ok();
}

Status World::collect_statistics(WorldStatistics &output) const noexcept {
    output={};
    if(!impl_) return invalid("World is not initialized");
    Status status=impl_->sync_results(false);
    if(!status) return status;
    output.frame_index=impl_->frame_index;
    output.rigid_body_count=static_cast<std::uint32_t>(impl_->host_states.size());
    output.rigid_constraint_count=static_cast<std::uint32_t>(std::count_if(
        impl_->host_constraints.begin(),impl_->host_constraints.end(),
        [](const ConstraintResource &item){ return item.alive!=0U; }));
    output.triangle_mesh_count=static_cast<std::uint32_t>(std::count_if(
        impl_->mesh_records.begin(),impl_->mesh_records.end(),
        [](const MeshRecord &item){ return item.alive; }));
    output.contact_count=impl_->rigid_event_count;
    output.contact_overflow_count=impl_->contact_overflow ? 1U : 0U;
    if(impl_->frame_index!=0U) {
        const auto *counts=static_cast<const std::uint32_t *>(impl_->counters.readback_data);
        output.rigid_contact_color_count=counts[2];
        output.rigid_contact_maximum_passes=counts[5];
        output.rigid_contact_live_pairs=counts[6];
    }
    output.allocated_bytes=impl_->allocated_bytes;
    return ok();
}

NativeContext World::native_context() const noexcept {
    return impl_ ? NativeContext{impl_->device.Get(),impl_->queue.Get()} : NativeContext{};
}

Status World::query_hit_box(HitBox box, HitBoxResult &output) const noexcept {
    output={};
    if(!impl_) return invalid("World is not initialized");
    if(!finite(box.center) || !finite(box.orientation) ||
       !finite(box.half_extents) || box.half_extents.x<0.0F ||
       box.half_extents.y<0.0F || box.half_extents.z<0.0F)
        return invalid("Invalid hit box");
    Status status=impl_->sync_results(true);
    if(!status) return status;
    try {
        for(std::size_t i=0;i<impl_->host_states.size();++i) {
            const Vec3 position=impl_->host_states[i].position;
            const Quaternion inverse{-box.orientation.x,-box.orientation.y,
                                     -box.orientation.z,box.orientation.w};
            const Vec3 relative=subtract(position,box.center);
            const Vec3 qv{inverse.x,inverse.y,inverse.z};
            const Vec3 twice=multiply(cross(qv,relative),2.0F);
            const Vec3 local=add(relative,
                add(multiply(twice,inverse.w),cross(qv,twice)));
            if(std::abs(local.x)<=box.half_extents.x &&
               std::abs(local.y)<=box.half_extents.y &&
               std::abs(local.z)<=box.half_extents.z)
                output.rigid_bodies.push_back(impl_->host_ids[i]);
        }
        return ok();
    } catch(const std::bad_alloc &) {
        return fail(StatusCode::out_of_memory,"Hit-box allocation failed");
    }
}

Status World::copy_d3d12_buffer_to_host(void *buffer,
    std::uint64_t byte_offset,std::uint64_t byte_count,void *destination) noexcept {
    if(!impl_ || buffer==nullptr || destination==nullptr || byte_count==0U)
        return invalid("Invalid buffer readback arguments");
    if(!impl_->idle()) return busy_status();
    ID3D12Resource *resource=static_cast<ID3D12Resource *>(buffer);
    const D3D12_RESOURCE_DESC description=resource->GetDesc();
    if(description.Dimension!=D3D12_RESOURCE_DIMENSION_BUFFER ||
       byte_offset>description.Width || byte_count>description.Width-byte_offset)
        return invalid("Buffer readback range is out of bounds");
    ComPtr<ID3D12Resource> readback;
    const D3D12_HEAP_PROPERTIES heap=heap_properties(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC readback_desc=buffer_description(byte_count);
    HRESULT hr=impl_->device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,
        &readback_desc,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,
        IID_PPV_ARGS(readback.ReleaseAndGetAddressOf()));
    if(FAILED(hr)) return impl_->hr_status(hr,"Could not allocate temporary readback");
    Status status=impl_->begin_commands();
    if(!status) return status;
    // External resources are required to be in COMMON when passed through the
    // opaque BufferSpan contract. COMMON promotes to COPY_SOURCE here.
    impl_->commands->CopyBufferRegion(readback.Get(),0U,resource,byte_offset,byte_count);
    status=impl_->submit(true);
    if(!status) return status;
    void *mapped=nullptr;
    D3D12_RANGE range{0U,static_cast<SIZE_T>(byte_count)};
    hr=readback->Map(0U,&range,&mapped);
    if(FAILED(hr)) return impl_->hr_status(hr,"Could not map temporary readback");
    std::memcpy(destination,mapped,static_cast<std::size_t>(byte_count));
    D3D12_RANGE written{0U,0U};
    readback->Unmap(0U,&written);
    return ok();
}

void *World::systems_implementation() noexcept { return impl_.get(); }
const void *World::systems_implementation() const noexcept { return impl_.get(); }
bool World::systems_mutation_allowed() const noexcept { return impl_ && impl_->idle(); }
Status World::systems_finish_mutation(Status status) noexcept { return status; }
std::uint64_t World::systems_revision() const noexcept {
    return impl_ ? impl_->revision : 0U;
}
bool World::systems_rigid_body_valid(RigidBodyId id) const noexcept {
    return impl_ && impl_->dense_index(id)!=UINT32_MAX;
}
std::uint32_t World::systems_triangle_mesh_vertex_count(TriangleMeshId id) const noexcept {
    return impl_ && impl_->valid_mesh(id)
        ? static_cast<std::uint32_t>(impl_->mesh_records[id.index].vertices.size()) : 0U;
}

} // namespace parallel_mater::d3d12
