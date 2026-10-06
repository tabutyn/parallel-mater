// SPDX-License-Identifier: MIT
#include "systems_internal.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace parallel_mater::metal::detail {
namespace {

constexpr Status success() noexcept { return {}; }
constexpr Status invalid_argument(const char *message) noexcept {
    return {StatusCode::invalid_argument, 0, message};
}
constexpr Status invalid_handle(const char *message) noexcept {
    return {StatusCode::invalid_handle, 0, message};
}
constexpr Status capacity_exceeded(const char *message) noexcept {
    return {StatusCode::capacity_exceeded, 0, message};
}
constexpr Status out_of_memory(const char *message) noexcept {
    return {StatusCode::out_of_memory, 0, message};
}
Status metal_failure(NSError *error, const char *message) noexcept {
    return {StatusCode::metal_failure,
            error == nil ? 0 : static_cast<std::int64_t>(error.code), message};
}

bool finite(Vec3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

float dot(Vec3 first, Vec3 second) noexcept {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

Vec3 cross(Vec3 first, Vec3 second) noexcept {
    return {first.y * second.z - first.z * second.y,
            first.z * second.x - first.x * second.z,
            first.x * second.y - first.y * second.x};
}

Vec3 subtract(Vec3 first, Vec3 second) noexcept {
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

Vec3 add(Vec3 first, Vec3 second) noexcept {
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

Vec3 multiply(Vec3 value, float scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 closest_point_triangle(Vec3 point, Vec3 a, Vec3 b, Vec3 c,
                            Vec3 &weights) noexcept {
    const Vec3 ab = subtract(b, a);
    const Vec3 ac = subtract(c, a);
    const Vec3 ap = subtract(point, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) {
        weights = {1.0F, 0.0F, 0.0F};
        return a;
    }
    const Vec3 bp = subtract(point, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) {
        weights = {0.0F, 1.0F, 0.0F};
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        const float value = d1 / (d1 - d3);
        weights = {1.0F - value, value, 0.0F};
        return add(a, multiply(ab, value));
    }
    const Vec3 cp = subtract(point, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) {
        weights = {0.0F, 0.0F, 1.0F};
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        const float value = d2 / (d2 - d6);
        weights = {1.0F - value, 0.0F, value};
        return add(a, multiply(ac, value));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F) {
        const float value = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        weights = {0.0F, 1.0F - value, value};
        return add(b, multiply(subtract(c, b), value));
    }
    const float inverse = 1.0F / (va + vb + vc);
    const float second = vb * inverse, third = vc * inverse;
    weights = {1.0F - second - third, second, third};
    return add(a, add(multiply(ab, second), multiply(ac, third)));
}

bool surface_volume_orientation(HostSpan<Vec3> vertices,
                                HostSpan<std::uint32_t> indices,
                                float &orientation) noexcept {
    if (!vertices.data || vertices.size == 0U || !indices.data ||
        indices.size == 0U || indices.size % 3U != 0U)
        return false;
    const Vec3 origin = vertices.data[0];
    double volume = 0.0;
    for (std::uint64_t base = 0; base < indices.size; base += 3U) {
        if (indices.data[base] >= vertices.size ||
            indices.data[base + 1U] >= vertices.size ||
            indices.data[base + 2U] >= vertices.size)
            return false;
        const Vec3 a = subtract(vertices.data[indices.data[base]], origin);
        const Vec3 b = subtract(vertices.data[indices.data[base + 1U]], origin);
        const Vec3 c = subtract(vertices.data[indices.data[base + 2U]], origin);
        volume += (static_cast<double>(a.x) *
                       (static_cast<double>(b.y) * c.z -
                        static_cast<double>(b.z) * c.y) +
                   static_cast<double>(a.y) *
                       (static_cast<double>(b.z) * c.x -
                        static_cast<double>(b.x) * c.z) +
                   static_cast<double>(a.z) *
                       (static_cast<double>(b.x) * c.y -
                        static_cast<double>(b.y) * c.x)) /
                  6.0;
    }
    if (std::abs(volume) <= 1.0e-10) return false;
    orientation = volume < 0.0 ? -1.0F : 1.0F;
    return true;
}

bool closed_surface_orientation(HostSpan<Vec3> vertices,
                                HostSpan<std::uint32_t> indices,
                                float &orientation) {
    if (!vertices.data || vertices.size < 4U || !indices.data ||
        indices.size == 0U || indices.size % 3U != 0U)
        return false;
    std::map<std::array<float, 3>, std::uint32_t> unique_vertices;
    std::vector<std::uint32_t> canonical(vertices.size);
    for (std::uint64_t index = 0; index < vertices.size; ++index) {
        const Vec3 point = vertices.data[index];
        canonical[index] = unique_vertices
                               .emplace(std::array{point.x, point.y, point.z},
                                        unique_vertices.size())
                               .first->second;
    }
    struct EdgeUse {
        std::uint32_t count{};
        int winding{};
    };
    std::map<std::pair<std::uint32_t, std::uint32_t>, EdgeUse> edges;
    const Vec3 origin = vertices.data[0];
    double volume = 0.0;
    for (std::uint64_t base = 0; base < indices.size; base += 3U) {
        const std::uint32_t triangle[3]{indices.data[base],
                                        indices.data[base + 1U],
                                        indices.data[base + 2U]};
        if (triangle[0] >= vertices.size || triangle[1] >= vertices.size ||
            triangle[2] >= vertices.size)
            return false;
        const Vec3 a{vertices.data[triangle[0]].x - origin.x,
                     vertices.data[triangle[0]].y - origin.y,
                     vertices.data[triangle[0]].z - origin.z};
        const Vec3 b{vertices.data[triangle[1]].x - origin.x,
                     vertices.data[triangle[1]].y - origin.y,
                     vertices.data[triangle[1]].z - origin.z};
        const Vec3 c{vertices.data[triangle[2]].x - origin.x,
                     vertices.data[triangle[2]].y - origin.y,
                     vertices.data[triangle[2]].z - origin.z};
        volume += (static_cast<double>(a.x) *
                       (static_cast<double>(b.y) * c.z -
                        static_cast<double>(b.z) * c.y) +
                   static_cast<double>(a.y) *
                       (static_cast<double>(b.z) * c.x -
                        static_cast<double>(b.x) * c.z) +
                   static_cast<double>(a.z) *
                       (static_cast<double>(b.x) * c.y -
                        static_cast<double>(b.y) * c.x)) /
                  6.0;
        for (std::uint32_t edge = 0; edge < 3U; ++edge) {
            const std::uint32_t first = canonical[triangle[edge]];
            const std::uint32_t second =
                canonical[triangle[(edge + 1U) % 3U]];
            if (first == second) return false;
            auto &use = edges[std::minmax(first, second)];
            ++use.count;
            use.winding += first < second ? 1 : -1;
        }
    }
    if (std::abs(volume) <= 1.0e-10 ||
        std::any_of(edges.begin(), edges.end(), [](const auto &entry) {
            return entry.second.count != 2U || entry.second.winding != 0;
        }))
        return false;
    orientation = volume < 0.0 ? -1.0F : 1.0F;
    return true;
}

struct FluidConstants {
    float timestep{};
    Vec3 gravity{};
    std::uint32_t count{};
    float particle_radius{};
    float support_radius{};
    float repulsion{};
    float viscosity{};
    float normal_damping{};
    float velocity_damping{};
    float maximum_speed{};
    float maximum_pair_acceleration{};
    float rest_density{};
    std::uint32_t maximum_neighbors{};
};

struct ParticleMetadata {
    std::uint32_t count{};
    std::uint32_t next_stable_id{};
    std::uint32_t revision{};
    std::uint32_t capacity{};
    std::uint64_t emitted{};
    std::uint64_t destroyed{};
    std::uint64_t boiled{};
};

struct SplitCounter {
    std::uint32_t low{};
    std::uint32_t high{};
};

struct FluidSmokeConstants {
    float timestep{};
    Vec3 gravity{};
    Vec3 heater_center{};
    Quaternion heater_orientation{};
    Vec2 heater_half_extents{};
    float heater_temperature{};
    float boiling_temperature{};
    float heat_transfer_rate{};
    float wind_drag{};
    float steam_rise_speed{};
    float water_radius{};
    float smoke_radius{};
    float smoke_rest_number_density{};
    float smoke_maximum_speed{};
    float smoke_lifetime{};
    std::uint32_t fluid_capacity{};
    std::uint32_t smoke_capacity{};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
    float grid_spacing{};
    Vec3 grid_minimum{};
};

struct SourceConstants {
    std::uint32_t site_count{};
    std::uint32_t capacity{};
    Vec3 initial_velocity{};
    float initial_temperature{};
    float clearance{};
    std::uint32_t enabled{};
};

struct DestroyConstants {
    Vec3 center{};
    Quaternion orientation{};
    Vec2 half_extents{};
    std::uint32_t crossing{};
    std::uint32_t enabled{};
};

struct SmokeMetadata {
    std::uint32_t count{};
    std::uint32_t next_particle{};
    float emission_remainder{};
    std::uint32_t revision{};
    std::uint64_t emitted{};
};

struct SmokeConstants {
    float timestep{};
    Vec3 gravity{};
    Vec3 emitter_center{};
    Vec3 initial_velocity{};
    Vec3 wind{};
    Vec2 emitter_half_extents{};
    std::uint32_t command{};
    float lifetime{};
    float particle_radius{};
    float buoyancy{};
    float response{};
    float maximum_speed{};
    float rest_number_density{};
    float pressure_stiffness{};
    float viscosity{};
    float vorticity_confinement{};
    std::uint32_t capacity{};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
    std::uint32_t grid_pressure_iterations{};
    Vec3 grid_minimum{};
    float grid_spacing{};
    float grid_kinematic_viscosity{};
    float grid_les_coefficient{};
    float grid_pressure_tolerance{};
};

struct SmokeGridContribution {
    std::uint64_t density{};
    std::uint64_t temperature{};
};

struct DeformableConstants {
    float timestep{};
    Vec3 gravity{};
    std::uint32_t count{};
    std::uint32_t bond_count{};
    std::uint32_t solver_iterations{};
    float compliance{};
    float velocity_damping{};
    float maximum_speed{};
    float radius{};
    float break_strain{};
    std::uint32_t fracture_persistence{};
    float impact_break_impulse{};
    std::uint32_t surface_count{};
    std::uint32_t triangle_index_count{};
    std::uint32_t preserve_volume{};
    float target_volume{};
    float volume_compliance{};
    float shape_matching_stiffness{};
    float maximum_projection_fraction{};
    std::uint32_t self_collision{};
    float spring_damping{};
    float constraint_velocity_response{1.0F};
};

struct MetalBond {
    std::uint32_t first{};
    std::uint32_t second{};
    float rest_length{};
    float compliance{};
    std::uint32_t active{1U};
};

struct DeformableNeighbor {
    std::uint32_t index{};
    float rest_length{};
    float compliance{};
    std::uint32_t bond{UINT32_MAX};
};

struct MetalSurfaceBinding {
    std::uint32_t nodes[4]{};
    float weights[4]{};
};

struct CouplingConstants {
    float timestep{};
    std::uint32_t count_a{};
    std::uint32_t count_b{};
    std::uint32_t mode{};
    float contact_distance{};
    float stiffness{};
    float damping{};
    float friction{};
    float maximum_force{};
    std::uint32_t first_vertex{UINT32_MAX};
    std::uint32_t last_vertex{UINT32_MAX};
    std::uint32_t enabled{1U};
};

struct FluidClothConstants {
    float timestep{};
    std::uint32_t surface_index_count{};
    float contact_distance{};
    float interaction_radius{};
    float stiffness{};
    float damping{};
    float tangential_drag{};
    float maximum_force{};
    float particle_mass{};
    float orientation{1.0F};
    float maximum_particle_speed{};
    std::uint32_t surface_count{};
    std::uint32_t enabled{1U};
    std::uint32_t cloth_count{};
};

struct FluidSoftConstants {
    float timestep{};
    std::uint32_t node_count{};
    std::uint32_t surface_count{};
    std::uint32_t surface_index_count{};
    float contact_distance{};
    float friction{};
    float particle_mass{};
    float maximum_particle_speed{};
    float maximum_soft_speed{};
    float orientation{1.0F};
    std::uint32_t enabled{1U};
    float maximum_projection{};
    float frame_inverse_timestep{};
    std::uint32_t rigid_count{};
    float rigid_recovery_radius{};
};

struct SoftClothConstants {
    float timestep{};
    std::uint32_t soft_count{};
    std::uint32_t soft_surface_count{};
    std::uint32_t cloth_count{};
    std::uint32_t cloth_surface_count{};
    std::uint32_t cloth_surface_index_count{};
    float contact_distance{};
    float friction{};
    float maximum_soft_speed{};
    std::uint32_t solver_iterations{};
    std::uint32_t enabled{1U};
};

struct SoftClothContact {
    std::uint32_t vertices[3]{};
    float weights[3]{};
    float soft_inverse_mass_fraction{};
    std::uint32_t active{};
    Vec3 position_impulse{};
    Vec3 velocity_impulse{};
};

struct SmokeSurfaceConstants {
    float timestep{};
    std::uint32_t target_count{};
    std::uint32_t surface_index_count{};
    std::uint32_t mode{};
    float lifetime{};
    float contact_distance{};
    float wind_radius{};
    float rest_number_density{};
    float wind_drag{};
    float maximum_wind_acceleration{};
    float maximum_target_speed{};
    float maximum_smoke_speed{};
    std::uint32_t enabled{1U};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
    float grid_spacing{};
    Vec3 grid_minimum{};
    float grid_kinematic_viscosity{};
    float grid_les_coefficient{};
    std::uint32_t command{};
    std::uint32_t raster_triangle_base{};
};

struct SmokeRigidRasterEntry {
    RigidBodyId body{};
    std::uint32_t triangle_base{};
};

struct RasterRigidParameters {
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

struct RasterTriangleMeshInfo {
    std::uint32_t vertex_offset{};
    std::uint32_t vertex_count{};
    std::uint32_t index_offset{};
    std::uint32_t index_count{};
    Vec3 minimum{};
    Vec3 maximum{};
    float radius{};
    std::uint32_t bvh_node_offset{};
    std::uint32_t bvh_node_count{};
    std::uint32_t solid_plane_offset{};
    std::uint32_t solid_plane_count{};
};

struct SmokeRopeConstants {
    float timestep{};
    std::uint32_t smoke_capacity{};
    std::uint32_t rope_count{};
    float lifetime{};
    float contact_distance{};
    float wind_radius{};
    float rest_number_density{};
    float wind_drag{};
    float maximum_wind_acceleration{};
    float maximum_rope_speed{};
    std::uint32_t skip_first{};
    std::uint32_t skip_last{};
    std::uint32_t enabled{1U};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
    float grid_spacing{};
    Vec3 grid_minimum{};
};

struct RopeSoftConstants {
    float timestep{};
    std::uint32_t rope_count{};
    std::uint32_t soft_count{};
    std::uint32_t surface_count{};
    std::uint32_t surface_index_count{};
    float contact_distance{};
    float friction{};
    float maximum_soft_acceleration{};
    float maximum_rope_speed{};
    float maximum_soft_speed{};
    float node_radius{};
    float anchor_support_radius_scale{};
    float anchor_contact_support_radius_scale{};
    float orientation{1.0F};
    std::uint32_t attach_first{};
    std::uint32_t attach_last{};
    std::uint32_t first_triangle{UINT32_MAX};
    Vec3 first_weights{};
    Vec3 first_offset{};
    std::uint32_t last_triangle{UINT32_MAX};
    Vec3 last_weights{};
    Vec3 last_offset{};
    std::uint32_t enabled{1U};
    float frame_inverse_timestep{60.0F};
};

struct ParticleRigidConstants {
    std::uint32_t particle_count{};
    std::uint32_t rigid_count{};
    float radius{};
    float friction{};
    float restitution{};
    float maximum_reaction_speed{};
    float timestep{};
    std::uint32_t diagnostic_is_acceleration{};
    FluidId fluid{};
    std::uint32_t collect_contacts{};
    float particle_inverse_mass{};
    std::uint32_t first_iteration{};
    std::uint32_t solid_contacts{};
    std::uint32_t share_position{};
    std::uint32_t first_spawned{};
    std::uint32_t recover_spawn{};
    float spawn_clearance{};
    Vec3 up{0.0F, 1.0F, 0.0F};
    Vec3 gravity{};
    float movable_mass{};
};

struct ParticleRigidContact {
    Vec3 normal{};
    Vec3 point{};
    float penetration{};
    std::uint32_t body{UINT32_MAX};
};

struct SoftContactAccumulator {
    Vec3 momentum_delta{};
    Vec3 friction_delta{};
    Vec3 arm{};
    float normal_delta{};
};

struct SoftContactState {
    std::uint32_t dynamic_contact_flag{};
    Vec3 predicted_momentum{};
};

struct ClothRigidConstants {
    float timestep{};
    std::uint32_t vertex_count{};
    std::uint32_t triangle_index_count{};
    std::uint32_t surface_count{};
    float thickness{};
    float friction{};
    std::uint32_t rigid_count{};
    std::uint32_t fracture_enabled{};
};

struct ClothBodyCorrection {
    Vec3 offset{};
    Vec3 impulse{};
    Vec3 contact{};
    float support_radius{};
    float weight_sum{};
    std::uint32_t vertices[3]{};
    std::uint32_t active{};
};

struct SmokeRigidConstants {
    RigidBodyId body{};
    float particle_radius{};
    float lifetime{};
    float air_density{};
    float drag_coefficient{};
    float contact_distance{};
    std::uint32_t tracer_contact{};
    std::uint32_t enabled{};
    std::uint32_t rigid_count{};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
    float grid_spacing{};
    Vec3 grid_minimum{};
    float density_scale{};
    float kinematic_viscosity{};
    float les_coefficient{};
    float timestep{};
    float maximum_speed{};
    float pressure_stiffness{};
};

struct RopeAttachmentConstants {
    RigidBodyId first_body{};
    Vec3 first_anchor{};
    std::uint32_t first_enabled{};
    RigidBodyId last_body{};
    Vec3 last_anchor{};
    std::uint32_t last_enabled{};
    std::uint32_t rigid_count{};
    std::uint32_t first_soft{};
    std::uint32_t last_soft{};
    std::uint32_t first_contact_skip{2U};
    std::uint32_t last_contact_skip{2U};
    std::uint32_t rigid_capacity{};
};

struct RopeAnchorState {
    Vec3 position{};
    Vec3 velocity{};
    Vec3 impulse{};
    float inverse_mass{};
};

struct PaintConstants {
    std::uint32_t mode{};
    RigidBodyId source{};
    RigidBodyId target{};
    std::uint32_t mesh_index{};
    std::uint32_t width{};
    std::uint32_t height{};
    float reach{};
    float particle_radius{};
    float cloth_thickness{};
    std::uint32_t rigid_count{};
    std::uint32_t cloth_count{};
    std::uint32_t cloth_index_count{};
    std::uint32_t enabled{};
};

static_assert(sizeof(FluidConstants) == 60U);
static_assert(sizeof(ParticleMetadata) == 40U);
static_assert(sizeof(SplitCounter) == 8U);
static_assert(sizeof(FluidSmokeConstants) == 124U);
static_assert(sizeof(SourceConstants) == 32U);
static_assert(sizeof(DestroyConstants) == 44U);
static_assert(sizeof(SmokeMetadata) == 24U);
static_assert(sizeof(SmokeConstants) == 144U);
static_assert(sizeof(SmokeGridContribution) == 16U);
static_assert(sizeof(DeformableConstants) == 96U);
static_assert(sizeof(MetalBond) == 20U);
static_assert(sizeof(DeformableNeighbor) == 16U);
static_assert(sizeof(MetalSurfaceBinding) == 32U);
static_assert(sizeof(CouplingConstants) == 48U);
static_assert(sizeof(FluidClothConstants) == 56U);
static_assert(sizeof(FluidSoftConstants) == 60U);
static_assert(sizeof(SoftClothConstants) == 44U);
static_assert(sizeof(SoftClothContact) == 56U);
static_assert(sizeof(SmokeSurfaceConstants) == 92U);
static_assert(sizeof(SmokeRigidRasterEntry) == 12U);
static_assert(sizeof(RasterRigidParameters) == 108U);
static_assert(sizeof(RasterTriangleMeshInfo) == 60U);
static_assert(sizeof(SmokeRopeConstants) == 76U);
static_assert(sizeof(RopeSoftConstants) == 128U);
static_assert(sizeof(ParticleRigidConstants) == 100U);
static_assert(sizeof(ParticleRigidContact) == 32U);
static_assert(sizeof(SoftContactAccumulator) == 40U);
static_assert(sizeof(SoftContactState) == 16U);
static_assert(sizeof(ClothRigidConstants) == 32U);
static_assert(sizeof(ClothBodyCorrection) == 60U);
static_assert(sizeof(FluidId) == 8U);
static_assert(sizeof(RigidBodyId) == 8U);
static_assert(sizeof(ContactEvent) == 48U);
static_assert(sizeof(SmokeRigidConstants) == 88U);
static_assert(sizeof(RopeAttachmentConstants) == 72U);
static_assert(sizeof(RopeAnchorState) == 40U);
static_assert(sizeof(PaintConstants) == 60U);

template <typename Resource> struct Slots {
    std::vector<std::unique_ptr<Resource>> entries{};
    std::vector<std::uint32_t> generations{};
    std::uint32_t count{};

    explicit Slots(std::uint32_t capacity = 0)
        : entries(capacity), generations(capacity, 1U) {}

    [[nodiscard]] std::uint32_t free_slot() const noexcept {
        for (std::uint32_t index = 0; index < entries.size(); ++index) {
            if (!entries[index]) return index;
        }
        return static_cast<std::uint32_t>(entries.size());
    }

    template <typename Id>
    [[nodiscard]] Resource *get(Id id) noexcept {
        return id.index < entries.size() && entries[id.index] &&
                       generations[id.index] == id.generation
                   ? entries[id.index].get()
                   : nullptr;
    }
    template <typename Id>
    [[nodiscard]] const Resource *get(Id id) const noexcept {
        return id.index < entries.size() && entries[id.index] &&
                       generations[id.index] == id.generation
                   ? entries[id.index].get()
                   : nullptr;
    }

    template <typename Id> void erase(Id id) noexcept {
        entries[id.index].reset();
        if (++generations[id.index] == 0U) generations[id.index] = 1U;
        --count;
    }
};

struct ParticleBuffers {
    id<MTLBuffer> positions{nil};
    id<MTLBuffer> previous{nil};
    id<MTLBuffer> velocities{nil};
    id<MTLBuffer> inverse_masses{nil};
    id<MTLBuffer> rigid_contacts{nil};
    id<MTLBuffer> rigid_contact_counts{nil};
    id<MTLBuffer> rigid_linear_impulses{nil};
    id<MTLBuffer> rigid_angular_impulses{nil};
    id<MTLBuffer> rigid_position_corrections{nil};
    id<MTLBuffer> rigid_particle_linear_impulses{nil};
    id<MTLBuffer> rigid_particle_angular_impulses{nil};
    id<MTLBuffer> rigid_particle_position_corrections{nil};
};

struct FluidResource {
    FluidOptions options{};
    std::uint32_t count{};
    std::uint32_t initial_count{};
    std::uint64_t revision{1U};
    ParticleBuffers particles{};
    id<MTLBuffer> accelerations{nil};
    id<MTLBuffer> stable_ids{nil};
    id<MTLBuffer> foam{nil};
    id<MTLBuffer> foam_sources{nil};
    id<MTLBuffer> temperatures{nil};
    id<MTLBuffer> cell_keys[2]{nil, nil};
    id<MTLBuffer> sorted_indices[2]{nil, nil};
    id<MTLBuffer> radix_histograms{nil};
    id<MTLBuffer> radix_bucket_offsets{nil};
    id<MTLBuffer> metadata{nil};
    id<MTLBuffer> spawn_baseline{nil};
    id<MTLBuffer> constants{nil};
    id<MTL4ArgumentTable> table{nil};
    id<MTLBuffer> rigid_constants{nil};
    id<MTL4ArgumentTable> rigid_table{nil};
    id<MTLBuffer> rigid_first_constants{nil};
    id<MTL4ArgumentTable> rigid_first_table{nil};
    id<MTLBuffer> contact_samples{nil};
    id<MTLBuffer> contact_flags{nil};
};

struct SmokeResource {
    SmokeOptions options{};
    std::uint64_t revision{1U};
    float emission_fraction{};
    bool static_metadata_valid{};
    std::uint32_t raster_triangle_count{};
    std::uint32_t rigid_raster_triangle_count{};
    ParticleBuffers particles{};
    id<MTLBuffer> ages{nil};
    id<MTLBuffer> densities{nil};
    id<MTLBuffer> pressures{nil};
    id<MTLBuffer> vorticities{nil};
    id<MTLBuffer> metadata{nil};
    id<MTLBuffer> constants{nil};
    id<MTLBuffer> advection_constants{nil};
    id<MTLBuffer> emission_constants{nil};
    id<MTLBuffer> grid_velocity{nil};
    id<MTLBuffer> grid_pressure{nil};
    id<MTLBuffer> grid_density{nil};
    id<MTLBuffer> grid_temperature{nil};
    id<MTLBuffer> grid_solid{nil};
    id<MTLBuffer> grid_vorticity{nil};
    id<MTLBuffer> grid_divergence{nil};
    id<MTLBuffer> grid_scratch{nil};
    id<MTLBuffer> grid_pressure_relative_residual{nil};
    id<MTLBuffer> grid_deformable_solid{nil};
    id<MTLBuffer> grid_rigid_bodies{nil};
    id<MTLBuffer> grid_rigid_body_count{nil};
    id<MTLBuffer> grid_face_velocity{nil};
    id<MTLBuffer> grid_face_advection{nil};
    id<MTLBuffer> grid_face_boundary{nil};
    id<MTLBuffer> grid_face_nearest_triangle{nil};
    id<MTLBuffer> grid_face_normal{nil};
    id<MTLBuffer> grid_splat_keys[2]{nil, nil};
    id<MTLBuffer> grid_splat_contributions[2]{nil, nil};
    id<MTLBuffer> grid_splat_histograms{nil};
    id<MTLBuffer> grid_splat_bucket_offsets{nil};
    id<MTLBuffer> grid_pressure_state{nil};
    id<MTL4ArgumentTable> table{nil};
    id<MTL4ArgumentTable> grid_pressure_table{nil};
    id<MTL4ArgumentTable> advection_table{nil};
    id<MTL4ArgumentTable> emission_table{nil};
    id<MTL4ArgumentTable> grid_splat_table{nil};
    id<MTL4ArgumentTable> grid_raster_table{nil};
};

struct ClothSeam {
    std::uint32_t corners[4]{};
    std::uint32_t bond{};
    std::uint32_t bending{UINT32_MAX};
};

struct ClothGraphEdge {
    std::uint32_t first{};
    std::uint32_t second{};
    float rest_length{};
    float compliance{};
    std::uint32_t bond{UINT32_MAX};
};

struct ClothResource {
    ClothOptions options{};
    std::uint64_t revision{1U};
    std::uint32_t count{};
    std::uint32_t capacity{};
    std::uint32_t triangle_index_count{};
    std::uint32_t bond_count{};
    std::uint32_t surface_count{};
    float orientation{1.0F};
    ParticleBuffers particles{};
    id<MTLBuffer> triangle_indices{nil};
    id<MTLBuffer> source_indices{nil};
    id<MTLBuffer> surface_positions{nil};
    id<MTLBuffer> surface_indices{nil};
    id<MTLBuffer> surface_source_indices{nil};
    id<MTLBuffer> surface_physical_indices{nil};
    id<MTLBuffer> public_bonds{nil};
    id<MTLBuffer> bonds{nil};
    id<MTLBuffer> active_bonds{nil};
    id<MTLBuffer> bond_damage{nil};
    id<MTLBuffer> neighbor_offsets{nil};
    id<MTLBuffer> neighbors{nil};
    id<MTLBuffer> free_triangle_nodes{nil};
    std::uint32_t neighbor_capacity{};
    id<MTLBuffer> corrections{nil};
    id<MTLBuffer> rigid_forces{nil};
    id<MTLBuffer> fluid_forces{nil};
    id<MTLBuffer> soft_forces{nil};
    id<MTLBuffer> rope_forces{nil};
    id<MTLBuffer> smoke_forces{nil};
    id<MTLBuffer> constants{nil};
    id<MTL4ArgumentTable> table{nil};
    id<MTLBuffer> rigid_constants{nil};
    id<MTL4ArgumentTable> rigid_table{nil};
    id<MTLBuffer> body_corrections{nil};
    id<MTLBuffer> surface_rigid_constants{nil};
    id<MTL4ArgumentTable> surface_rigid_table{nil};
    std::vector<ClothSeam> seams{};
    std::vector<std::array<std::uint32_t, 2>> bond_corners{};
    std::vector<std::uint32_t> triangle_bonds{};
    std::vector<float> source_inverse_masses{};
    std::vector<std::uint32_t> source_degrees{};
    std::vector<std::uint8_t> topology_active{};
    std::vector<std::uint8_t> topology_scratch_active{};
    std::vector<std::uint32_t> topology_scratch_old_indices{};
    std::vector<std::uint32_t> topology_scratch_indices{};
    std::vector<std::uint32_t> topology_scratch_parent{};
    std::vector<std::uint32_t> topology_scratch_nodes{};
    std::vector<std::uint8_t> topology_scratch_used{};
    std::vector<std::uint32_t> topology_scratch_degrees{};
    std::vector<std::array<std::uint32_t, 2>> topology_scratch_copies{};
    std::vector<ClothGraphEdge> topology_scratch_edges{};
    std::vector<std::uint32_t> topology_scratch_graph_degrees{};
    std::vector<std::uint32_t> topology_scratch_offsets{};
    std::vector<std::uint32_t> topology_scratch_cursors{};
    std::vector<DeformableNeighbor> topology_scratch_neighbors{};
    std::vector<std::uint8_t> topology_scratch_free_nodes{};
};

Status rebuild_cloth_topology(ClothResource &cloth,
                              bool initial = false) noexcept;

struct SoftResource {
    SoftBodyOptions options{};
    std::uint64_t revision{1U};
    std::uint32_t count{};
    std::uint32_t bond_count{};
    std::uint32_t surface_count{};
    std::uint32_t surface_index_count{};
    bool surface_closed{};
    bool surface_has_volume{};
    float surface_orientation{1.0F};
    float movable_mass{};
    ParticleBuffers particles{};
    id<MTLBuffer> public_bonds{nil};
    id<MTLBuffer> bonds{nil};
    id<MTLBuffer> rest_positions{nil};
    id<MTLBuffer> corrections{nil};
    id<MTLBuffer> surface_positions{nil};
    id<MTLBuffer> surface_rest_positions{nil};
    id<MTLBuffer> surface_indices{nil};
    id<MTLBuffer> surface_bindings{nil};
    id<MTLBuffer> shape_orientation{nil};
    id<MTLBuffer> contact_accumulators{nil};
    id<MTLBuffer> contact_state{nil};
    id<MTLBuffer> rigid_forces{nil};
    id<MTLBuffer> cloth_forces{nil};
    id<MTLBuffer> fluid_forces{nil};
    id<MTLBuffer> rope_forces{nil};
    id<MTLBuffer> smoke_forces{nil};
    id<MTLBuffer> constants{nil};
    id<MTL4ArgumentTable> table{nil};
    id<MTLBuffer> rigid_constants{nil};
    id<MTL4ArgumentTable> rigid_table{nil};
};

struct RopeResource {
    RopeOptions options{};
    std::uint64_t revision{1U};
    std::uint32_t count{};
    ParticleBuffers particles{};
    id<MTLBuffer> rest_lengths{nil};
    id<MTLBuffer> constraint_forces{nil};
    id<MTLBuffer> contact_forces{nil};
    id<MTLBuffer> fluid_forces{nil};
    id<MTLBuffer> soft_forces{nil};
    id<MTLBuffer> bonds{nil};
    id<MTLBuffer> directions{nil};
    id<MTLBuffer> diagonal{nil};
    id<MTLBuffer> upper{nil};
    id<MTLBuffer> rhs{nil};
    id<MTLBuffer> lambdas{nil};
    id<MTLBuffer> scratch{nil};
    id<MTLBuffer> contact_normals{nil};
    id<MTLBuffer> contact_normals2{nil};
    id<MTLBuffer> body_translation{nil};
    id<MTLBuffer> body_rotation{nil};
    id<MTLBuffer> anchor_states{nil};
    id<MTLBuffer> empty_soft_target{nil};
    id<MTLBuffer> constants{nil};
    id<MTLBuffer> attachment_constants{nil};
    id<MTL4ArgumentTable> table{nil};
};

struct SourceResource {
    ParticleSourceOptions options{};
    FluidId fluid{};
    float clearance{};
    std::uint32_t site_count{};
    id<MTLBuffer> sites{nil};
    id<MTLBuffer> constants{nil};
    id<MTL4ArgumentTable> table{nil};
};

struct DestroyResource {
    ParticleDestroyPlaneOptions options{};
    id<MTLBuffer> constants{nil};
    id<MTL4ArgumentTable> table{nil};
};

struct PaintFieldResource {
    PaintFieldHostOptions options{};
    std::uint32_t vertex_count{};
    std::uint64_t revision{1U};
    id<MTLBuffer> uvs{nil};
    id<MTLBuffer> pixels{nil};
};

struct PaintRuleResource {
    PaintRuleOptions options{};
    id<MTLBuffer> constants{nil};
    id<MTL4ArgumentTable> table{nil};
};

template <typename Options> struct CouplingResource {
    Options options{};
    bool enabled{true};
    id<MTLBuffer> constants{nil};
    id<MTLBuffer> phase_constants{nil};
    id<MTLBuffer> contact_count{nil};
    id<MTLBuffer> maximum_penetration{nil};
    id<MTLBuffer> packed_state{nil};
    id<MTLBuffer> previous_surface{nil};
    id<MTLBuffer> contribution_nodes{nil};
    id<MTLBuffer> contribution_positions{nil};
    id<MTLBuffer> contribution_changes{nil};
    id<MTLBuffer> contribution_forces{nil};
    id<MTL4ArgumentTable> table{nil};
    id<MTL4ArgumentTable> phase_table{nil};
};

template <typename T>
void copy_to(id<MTLBuffer> buffer, const T *source, std::size_t count) {
    if (count != 0U) {
        std::memcpy(buffer.contents, source, count * sizeof(T));
    }
}

template <typename T>
BufferSpan<const T> span(id<MTLBuffer> buffer, std::uint64_t count) noexcept {
    return {(__bridge void *)buffer, 0U, count};
}

} // namespace

struct MetalSystems::Impl {
    WorldOptions options{};
    id<MTLDevice> device{nil};
    id<MTLLibrary> library{nil};
    id<MTLResidencySet> residency{nil};
    id<MTLComputePipelineState> fluid_forces_pipeline{nil};
    id<MTLComputePipelineState> fluid_cell_keys_pipeline{nil};
    id<MTLComputePipelineState> fluid_radix_histogram_pipelines[8]{};
    id<MTLComputePipelineState> fluid_radix_prefix_blocks_pipeline{nil};
    id<MTLComputePipelineState> fluid_radix_prefix_buckets_pipeline{nil};
    id<MTLComputePipelineState> fluid_radix_scatter_pipelines[8]{};
    id<MTLComputePipelineState> fluid_integrate_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_splat_generate_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_splat_histogram_pipelines[4]{};
    id<MTLComputePipelineState> smoke_grid_splat_prefix_blocks_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_splat_prefix_buckets_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_splat_scatter_pipelines[4]{};
    id<MTLComputePipelineState> smoke_grid_splat_resolve_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_raster_clear_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_raster_rigid_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_select_rigid_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_resolve_rigid_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_merge_obstacles_pipeline{nil};
    id<MTLComputePipelineState> smoke_pressure_restrict_open_pipelines[3]{};
    id<MTLComputePipelineState> smoke_pressure_smooth_pipelines[4][2]{};
    id<MTLComputePipelineState> smoke_pressure_residual_pipelines[3]{};
    id<MTLComputePipelineState> smoke_pressure_restrict_residual_pipelines[3]{};
    id<MTLComputePipelineState> smoke_pressure_clear_pipelines[3]{};
    id<MTLComputePipelineState> smoke_pressure_prolong_pipelines[3]{};
    id<MTLComputePipelineState> smoke_pressure_cycle_begin_pipeline{nil};
    id<MTLComputePipelineState> smoke_pressure_cycle_end_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_pressure_begin_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_mark_boundaries_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_advect_forward_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_advect_reverse_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_correct_face_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_diagnostics_pre_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_subgrid_force_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_apply_forces_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_apply_boundaries_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_divergence_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_project_face_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_diagnostics_post_pipeline{nil};
    id<MTLComputePipelineState> smoke_particle_density_pipeline{nil};
    id<MTLComputePipelineState> smoke_particle_vorticity_pipeline{nil};
    id<MTLComputePipelineState> smoke_particle_forces_pipeline{nil};
    id<MTLComputePipelineState> smoke_particle_integrate_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_advect_particles_pipeline{nil};
    id<MTLComputePipelineState> smoke_emission_pipeline{nil};
    id<MTLComputePipelineState> smoke_grid_clear_pipeline{nil};
    id<MTLComputePipelineState> cloth_prediction_pipeline{nil};
    id<MTLComputePipelineState> cloth_damage_pipeline{nil};
    id<MTLComputePipelineState> cloth_project_pipeline{nil};
    id<MTLComputePipelineState> cloth_apply_pipeline{nil};
    id<MTLComputePipelineState> cloth_limit_strain_pipeline{nil};
    id<MTLComputePipelineState> cloth_pipeline{nil};
    id<MTLComputePipelineState> cloth_finalize_pipeline{nil};
    id<MTLComputePipelineState> cloth_surface_update_pipeline{nil};
    id<MTLComputePipelineState> cloth_impact_pipeline{nil};
    id<MTLComputePipelineState> cloth_rigid_surface_pipeline{nil};
    id<MTLComputePipelineState> soft_prediction_pipeline{nil};
    id<MTLComputePipelineState> soft_project_pipeline{nil};
    id<MTLComputePipelineState> soft_apply_pipeline{nil};
    id<MTLComputePipelineState> soft_pipeline{nil};
    id<MTLComputePipelineState> soft_finalize_pipeline{nil};
    id<MTLComputePipelineState> soft_damping_prepare_pipeline{nil};
    id<MTLComputePipelineState> soft_damping_apply_pipeline{nil};
    id<MTLComputePipelineState> soft_surface_update_pipeline{nil};
    id<MTLComputePipelineState> soft_contact_clear_pipeline{nil};
    id<MTLComputePipelineState> soft_measure_momentum_pipeline{nil};
    id<MTLComputePipelineState> soft_contact_finish_pipeline{nil};
    id<MTLComputePipelineState> soft_restore_momentum_pipeline{nil};
    id<MTLComputePipelineState> rope_prediction_pipeline{nil};
    id<MTLComputePipelineState> rope_pipeline{nil};
    id<MTLComputePipelineState> rope_cloth_sample_pipeline{nil};
    id<MTLComputePipelineState> rope_soft_sample_pipeline{nil};
    id<MTLComputePipelineState> coupling_pipeline{nil};
    id<MTLComputePipelineState> fluid_rope_contact_pipeline{nil};
    id<MTLComputePipelineState> fluid_rope_apply_pipeline{nil};
    id<MTLComputePipelineState> fluid_cloth_pipeline{nil};
    id<MTLComputePipelineState> fluid_cloth_apply_pipeline{nil};
    id<MTLComputePipelineState> fluid_cloth_surface_pipeline{nil};
    id<MTLComputePipelineState> fluid_cloth_project_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_solve_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_apply_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_surface_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_sweep_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_recover_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_recover_current_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_copy_pipeline{nil};
    id<MTLComputePipelineState> fluid_soft_rigid_recover_pipeline{nil};
    id<MTLComputePipelineState> soft_contact_friction_pipeline{nil};
    id<MTLComputePipelineState> soft_cloth_clear_pipeline{nil};
    id<MTLComputePipelineState> soft_cloth_pipeline{nil};
    id<MTLComputePipelineState> soft_cloth_apply_soft_pipeline{nil};
    id<MTLComputePipelineState> soft_cloth_apply_pipeline{nil};
    id<MTLComputePipelineState> soft_cloth_surface_pipeline{nil};
    id<MTLComputePipelineState> soft_cloth_soft_surface_pipeline{nil};
    id<MTLComputePipelineState> smoke_surface_pipeline{nil};
    id<MTLComputePipelineState> smoke_surface_grid_raster_pipeline{nil};
    id<MTLComputePipelineState> smoke_surface_grid_select_pipeline{nil};
    id<MTLComputePipelineState> smoke_surface_grid_resolve_pipeline{nil};
    id<MTLComputePipelineState> smoke_surface_grid_force_pipeline{nil};
    id<MTLComputePipelineState> smoke_rope_pipeline{nil};
    id<MTLComputePipelineState> smoke_rope_wind_pipeline{nil};
    id<MTLComputePipelineState> rope_soft_anchor_weight_pipeline{nil};
    id<MTLComputePipelineState> rope_soft_pipeline{nil};
    id<MTLComputePipelineState> rope_soft_surface_pipeline{nil};
    id<MTLComputePipelineState> rope_soft_endpoint_pipeline{nil};
    id<MTLComputePipelineState> rope_soft_finalize_pipeline{nil};
    id<MTLComputePipelineState> rope_soft_pack_pipeline{nil};
    id<MTLComputePipelineState> fluid_smoke_pipeline{nil};
    id<MTLComputePipelineState> fluid_smoke_compact_pipeline{nil};
    id<MTLComputePipelineState> particle_rigid_pipeline{nil};
    id<MTLComputePipelineState> smoke_rigid_pipeline{nil};
    id<MTLComputePipelineState> fluid_capture_spawn_baseline_pipeline{nil};
    id<MTLComputePipelineState> source_pipeline{nil};
    id<MTLComputePipelineState> destroy_pipeline{nil};
    id<MTLComputePipelineState> paint_pipeline{nil};

    Slots<FluidResource> fluids{};
    Slots<SmokeResource> smokes{};
    Slots<ClothResource> cloths{};
    Slots<SoftResource> soft_bodies{};
    Slots<RopeResource> ropes{};
    Slots<SourceResource> sources{};
    Slots<DestroyResource> destroy_planes{};
    Slots<PaintFieldResource> paint_fields{};
    Slots<PaintRuleResource> paint_rules{};
    Slots<CouplingResource<FluidSmokeCouplingOptions>> fluid_smoke{};
    Slots<CouplingResource<SmokeSoftBodyCouplingOptions>> smoke_soft{};
    Slots<CouplingResource<SmokeClothCouplingOptions>> smoke_cloth{};
    Slots<CouplingResource<SmokeRopeCouplingOptions>> smoke_rope{};
    Slots<CouplingResource<SmokeRigidCouplingOptions>> smoke_rigid{};
    Slots<CouplingResource<FluidRopeCouplingOptions>> fluid_rope{};
    Slots<CouplingResource<RopeSoftBodyCouplingOptions>> rope_soft{};
    Slots<CouplingResource<RopeClothCouplingOptions>> rope_cloth{};
    Slots<CouplingResource<FluidClothCouplingOptions>> fluid_cloth{};
    Slots<CouplingResource<SoftBodyClothCouplingOptions>> soft_cloth{};
    Slots<CouplingResource<FluidSoftBodyCouplingOptions>> fluid_soft{};

    id<MTLBuffer> rigid_ids{nil};
    id<MTLBuffer> rigid_states{nil};
    id<MTLBuffer> rigid_previous_states{nil};
    id<MTLBuffer> rigid_frame_states{nil};
    id<MTLBuffer> rigid_parameters{nil};
    id<MTLBuffer> mesh_vertices{nil};
    id<MTLBuffer> mesh_indices{nil};
    id<MTLBuffer> mesh_infos{nil};
    id<MTLBuffer> mesh_solid_planes{nil};
    std::uint32_t rigid_count{};
    id<MTLBuffer> fluid_contact_events{nil};
    id<MTLBuffer> fluid_neighbor_overflow{nil};
    id<MTLBuffer> fluid_maximum_neighbor_count{nil};
    id<MTLBuffer> spawn_capacity_miss_count{nil};
    std::uint64_t archived_emitted_particle_count{};
    std::uint64_t archived_destroyed_particle_count{};
    std::uint64_t archived_boiled_particle_count{};
    bool collect_fluid_contacts{};
    float frame_timestep{1.0F / 60.0F};
    float frame_inverse_timestep{60.0F};
    mutable std::uint32_t fluid_contact_count{};
    mutable std::uint32_t fluid_contact_overflow{};
    mutable std::uint64_t contact_cache_frame{UINT64_MAX};

    Impl() = default;
    explicit Impl(WorldOptions value)
        : options(value), fluids(value.fluid_capacity),
          smokes(value.smoke_capacity), cloths(value.cloth_capacity),
          soft_bodies(value.soft_body_capacity), ropes(value.rope_capacity),
          sources(value.particle_source_capacity),
          destroy_planes(value.particle_destroy_plane_capacity),
          paint_fields(value.paint_field_capacity),
          paint_rules(value.paint_rule_capacity),
          fluid_smoke(value.fluid_smoke_coupling_capacity),
          smoke_soft(value.smoke_soft_body_coupling_capacity),
          smoke_cloth(value.smoke_cloth_coupling_capacity),
          smoke_rope(value.smoke_rope_coupling_capacity),
          smoke_rigid(value.smoke_rigid_coupling_capacity),
          fluid_rope(value.fluid_rope_coupling_capacity),
          rope_soft(value.rope_soft_body_coupling_capacity),
          rope_cloth(value.rope_cloth_coupling_capacity),
          fluid_cloth(value.fluid_cloth_coupling_capacity),
          soft_cloth(value.soft_body_cloth_coupling_capacity),
          fluid_soft(value.fluid_soft_body_coupling_capacity) {}

    [[nodiscard]] id<MTLBuffer> buffer(std::size_t bytes) {
        id<MTLBuffer> result = [device
            newBufferWithLength:std::max<std::size_t>(bytes, 4U)
                         options:MTLResourceStorageModeShared];
        if (result != nil) {
            std::memset(result.contents, 0, result.length);
            [residency addAllocation:result];
        }
        return result;
    }

    [[nodiscard]] id<MTL4ArgumentTable> table(std::uint32_t bindings) {
        MTL4ArgumentTableDescriptor *descriptor =
            [[MTL4ArgumentTableDescriptor alloc] init];
        descriptor.maxBufferBindCount = bindings;
        descriptor.initializeBindings = YES;
        NSError *error = nil;
        return [device newArgumentTableWithDescriptor:descriptor error:&error];
    }

    void bind(id<MTL4ArgumentTable> table, std::uint32_t index,
              id<MTLBuffer> buffer) noexcept {
        [table setAddress:buffer.gpuAddress atIndex:index];
    }

    template <typename Options, typename Id>
    [[nodiscard]] Status add_coupling(
        Slots<CouplingResource<Options>> &slots, Options options,
        ParticleBuffers &first, std::uint32_t first_count,
        ParticleBuffers &second, std::uint32_t second_count,
        id<MTLBuffer> first_diagnostics, id<MTLBuffer> second_diagnostics,
        std::uint32_t mode, float contact_distance, float stiffness,
        float damping, float friction, float maximum_force,
        std::uint32_t first_vertex, std::uint32_t last_vertex, bool enabled,
        Id &output) {
        output = {};
        const std::uint32_t slot = slots.free_slot();
        if (slot == slots.entries.size())
            return capacity_exceeded("Particle coupling capacity is exhausted");
        auto resource = std::make_unique<CouplingResource<Options>>();
        resource->options = options;
        resource->enabled = enabled;
        resource->constants = buffer(sizeof(CouplingConstants));
        resource->contact_count = buffer(sizeof(std::uint32_t));
        resource->maximum_penetration = buffer(sizeof(float));
        resource->contribution_nodes = buffer(
            static_cast<std::size_t>(first_count) * sizeof(std::uint32_t) *
            2U);
        resource->contribution_changes = buffer(
            static_cast<std::size_t>(first_count) * sizeof(Vec3) * 2U);
        resource->contribution_forces = buffer(
            static_cast<std::size_t>(first_count) * sizeof(Vec3) * 2U);
        resource->table = table(16U);
        if (resource->constants == nil || resource->contact_count == nil ||
            resource->maximum_penetration == nil ||
            resource->contribution_nodes == nil ||
            resource->contribution_changes == nil ||
            resource->contribution_forces == nil || resource->table == nil)
            return metal_failure(nil, "Could not allocate Metal coupling state");
        *static_cast<CouplingConstants *>(resource->constants.contents) = {
            0.0F, first_count, second_count, mode, contact_distance, stiffness,
            damping, friction, maximum_force, first_vertex, last_vertex,
            enabled ? 1U : 0U};
        bind(resource->table, 0U, first.positions);
        bind(resource->table, 1U, first.velocities);
        bind(resource->table, 2U, first.inverse_masses);
        bind(resource->table, 3U, second.positions);
        bind(resource->table, 4U, second.velocities);
        bind(resource->table, 5U, second.inverse_masses);
        bind(resource->table, 6U, resource->constants);
        bind(resource->table, 7U, first_diagnostics);
        bind(resource->table, 8U, second_diagnostics);
        bind(resource->table, 9U, resource->constants);
        bind(resource->table, 10U, resource->contact_count);
        bind(resource->table, 11U, resource->maximum_penetration);
        bind(resource->table, 13U, resource->contribution_nodes);
        bind(resource->table, 14U, resource->contribution_changes);
        bind(resource->table, 15U, resource->contribution_forces);
        [residency commit];
        slots.entries[slot] = std::move(resource);
        ++slots.count;
        output = {slot, slots.generations[slot]};
        return success();
    }

    [[nodiscard]] Status setup_rigid_contact_table(
        ParticleBuffers &particles, id<MTLBuffer> __strong &constants,
        id<MTL4ArgumentTable> __strong &argument_table,
        id<MTLBuffer> diagnostics, std::uint32_t particle_capacity) {
        (void)particle_capacity;
        constants = buffer(sizeof(ParticleRigidConstants));
        argument_table = table(30U);
        if (constants == nil || argument_table == nil)
            return metal_failure(nil,
                                 "Could not allocate particle-rigid table");
        bind(argument_table, 0U, particles.positions);
        bind(argument_table, 1U, particles.velocities);
        bind(argument_table, 2U, particles.inverse_masses);
        bind(argument_table, 8U, constants);
        bind(argument_table, 9U, diagnostics);
        bind(argument_table, 10U, constants);
        bind(argument_table, 11U, constants);
        bind(argument_table, 12U, constants);
        bind(argument_table, 13U, constants);
        bind(argument_table, 14U, constants);
        bind(argument_table, 15U, particles.previous);
        bind(argument_table, 16U, constants);
        bind(argument_table, 17U, constants);
        bind(argument_table, 18U, particles.rigid_contacts);
        bind(argument_table, 19U, particles.rigid_contact_counts);
        bind(argument_table, 20U, particles.rigid_linear_impulses);
        bind(argument_table, 21U, particles.rigid_angular_impulses);
        bind(argument_table, 22U, constants);
        bind(argument_table, 23U, particles.rigid_position_corrections);
        bind(argument_table, 24U,
             particles.rigid_particle_linear_impulses);
        bind(argument_table, 25U,
             particles.rigid_particle_angular_impulses);
        bind(argument_table, 26U,
             particles.rigid_particle_position_corrections);
        for (std::uint32_t slot = 27U; slot < 30U; ++slot)
            bind(argument_table, slot, constants);
        return success();
    }

    [[nodiscard]] Status setup_rigid_contact(
        ParticleBuffers &particles, id<MTLBuffer> __strong &constants,
        id<MTL4ArgumentTable> __strong &argument_table,
        id<MTLBuffer> diagnostics, std::uint32_t particle_capacity) {
        particles.rigid_contacts = buffer(
            static_cast<std::size_t>(particle_capacity) *
            sizeof(ParticleRigidContact));
        particles.rigid_contact_counts = buffer(
            static_cast<std::size_t>(options.rigid_body_capacity) *
            sizeof(std::uint32_t));
        particles.rigid_linear_impulses = buffer(
            static_cast<std::size_t>(options.rigid_body_capacity) *
            sizeof(Vec3));
        particles.rigid_angular_impulses = buffer(
            static_cast<std::size_t>(options.rigid_body_capacity) *
            sizeof(Vec3));
        particles.rigid_position_corrections = buffer(
            static_cast<std::size_t>(options.rigid_body_capacity) *
            sizeof(Vec3));
        particles.rigid_particle_linear_impulses = buffer(
            static_cast<std::size_t>(particle_capacity) * sizeof(Vec3));
        particles.rigid_particle_angular_impulses = buffer(
            static_cast<std::size_t>(particle_capacity) * sizeof(Vec3));
        particles.rigid_particle_position_corrections = buffer(
            static_cast<std::size_t>(particle_capacity) * sizeof(Vec3));
        if (particles.rigid_contacts == nil ||
            particles.rigid_contact_counts == nil ||
            particles.rigid_linear_impulses == nil ||
            particles.rigid_angular_impulses == nil ||
            particles.rigid_position_corrections == nil ||
            particles.rigid_particle_linear_impulses == nil ||
            particles.rigid_particle_angular_impulses == nil ||
            particles.rigid_particle_position_corrections == nil)
            return metal_failure(nil,
                                 "Could not allocate particle-rigid state");
        return setup_rigid_contact_table(
            particles, constants, argument_table, diagnostics,
            particle_capacity);
    }
};

MetalSystems::MetalSystems() noexcept = default;
MetalSystems::~MetalSystems() = default;
MetalSystems::MetalSystems(MetalSystems &&) noexcept = default;
MetalSystems &MetalSystems::operator=(MetalSystems &&) noexcept = default;

Status MetalSystems::create(WorldOptions options, void *device, void *library,
                            void *residency_set,
                            MetalSystems &output) noexcept {
    @autoreleasepool {
        try {
            auto impl = std::make_unique<Impl>(options);
            impl->device = (__bridge id<MTLDevice>)device;
            impl->library = (__bridge id<MTLLibrary>)library;
            impl->residency = (__bridge id<MTLResidencySet>)residency_set;
            if (impl->device == nil || impl->library == nil ||
                impl->residency == nil) {
                return invalid_argument("Metal particle runtime needs native resources");
            }
            NSError *error = nil;
            const auto pipeline = [&](NSString *name,
                                      id<MTLComputePipelineState> __strong &target)
                -> Status {
                id<MTLFunction> function =
                    [impl->library newFunctionWithName:name];
                if (function == nil)
                    return metal_failure(nil, "Embedded particle kernel is missing");
                target = [impl->device
                    newComputePipelineStateWithFunction:function error:&error];
                return target == nil
                           ? metal_failure(error,
                                           "Could not create particle pipeline")
                           : success();
            };
            Status status = pipeline(@"pm_fluid_cell_keys",
                                     impl->fluid_cell_keys_pipeline);
            for (std::uint32_t pass = 0U; status && pass < 8U; ++pass) {
                status = pipeline(
                    [NSString stringWithFormat:@"pm_fluid_radix_histogram_%u",
                                               pass],
                    impl->fluid_radix_histogram_pipelines[pass]);
            }
            if (status)
                status = pipeline(@"pm_fluid_radix_prefix_blocks",
                                  impl->fluid_radix_prefix_blocks_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_radix_prefix_buckets",
                                  impl->fluid_radix_prefix_buckets_pipeline);
            for (std::uint32_t pass = 0U; status && pass < 8U; ++pass) {
                status = pipeline(
                    [NSString stringWithFormat:@"pm_fluid_radix_scatter_%u",
                                               pass],
                    impl->fluid_radix_scatter_pipelines[pass]);
            }
            if (status) status = pipeline(@"pm_fluid_forces", impl->fluid_forces_pipeline);
            if (status) status = pipeline(@"pm_fluid_integrate", impl->fluid_integrate_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_splat_generate",
                                  impl->smoke_grid_splat_generate_pipeline);
            for (std::uint32_t pass = 0U; status && pass < 4U; ++pass) {
                status = pipeline(
                    [NSString stringWithFormat:
                        @"pm_smoke_grid_splat_histogram_%u", pass],
                    impl->smoke_grid_splat_histogram_pipelines[pass]);
            }
            if (status)
                status = pipeline(@"pm_smoke_grid_splat_prefix_blocks",
                                  impl->smoke_grid_splat_prefix_blocks_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_splat_prefix_buckets",
                                  impl->smoke_grid_splat_prefix_buckets_pipeline);
            for (std::uint32_t pass = 0U; status && pass < 4U; ++pass) {
                status = pipeline(
                    [NSString stringWithFormat:
                        @"pm_smoke_grid_splat_scatter_%u", pass],
                    impl->smoke_grid_splat_scatter_pipelines[pass]);
            }
            if (status)
                status = pipeline(@"pm_smoke_grid_splat_resolve",
                                  impl->smoke_grid_splat_resolve_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_raster_clear",
                                  impl->smoke_grid_raster_clear_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_raster_rigid",
                                  impl->smoke_grid_raster_rigid_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_select_rigid",
                                  impl->smoke_grid_select_rigid_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_resolve_rigid",
                                  impl->smoke_grid_resolve_rigid_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_merge_obstacles",
                                  impl->smoke_grid_merge_obstacles_pipeline);
            for (std::uint32_t level = 0U; status && level < 3U; ++level) {
                status = pipeline(
                    [NSString stringWithFormat:
                        @"pm_smoke_pressure_restrict_open_%u", level + 1U],
                    impl->smoke_pressure_restrict_open_pipelines[level]);
            }
            for (std::uint32_t level = 0U; status && level < 4U; ++level) {
                status = pipeline(
                    [NSString stringWithFormat:
                        @"pm_smoke_pressure_smooth_%u_forward", level],
                    impl->smoke_pressure_smooth_pipelines[level][0]);
                if (status)
                    status = pipeline(
                        [NSString stringWithFormat:
                            @"pm_smoke_pressure_smooth_%u_backward", level],
                        impl->smoke_pressure_smooth_pipelines[level][1]);
            }
            for (std::uint32_t level = 0U; status && level < 3U; ++level) {
                status = pipeline(
                    [NSString stringWithFormat:
                        @"pm_smoke_pressure_residual_%u", level],
                    impl->smoke_pressure_residual_pipelines[level]);
                if (status)
                    status = pipeline(
                        [NSString stringWithFormat:
                            @"pm_smoke_pressure_restrict_residual_%u", level],
                        impl->smoke_pressure_restrict_residual_pipelines[level]);
                if (status)
                    status = pipeline(
                        [NSString stringWithFormat:
                            @"pm_smoke_pressure_clear_%u", level + 1U],
                        impl->smoke_pressure_clear_pipelines[level]);
                if (status)
                    status = pipeline(
                        [NSString stringWithFormat:
                            @"pm_smoke_pressure_prolong_%u", level + 1U],
                        impl->smoke_pressure_prolong_pipelines[level]);
            }
            if (status)
                status = pipeline(@"pm_smoke_pressure_cycle_begin",
                                  impl->smoke_pressure_cycle_begin_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_pressure_cycle_end",
                                  impl->smoke_pressure_cycle_end_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_pressure_begin",
                                  impl->smoke_grid_pressure_begin_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_mark_domain_boundaries",
                                  impl->smoke_grid_mark_boundaries_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_advect_forward",
                                  impl->smoke_grid_advect_forward_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_advect_reverse",
                                  impl->smoke_grid_advect_reverse_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_correct_face",
                                  impl->smoke_grid_correct_face_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_cell_diagnostics_pre",
                                  impl->smoke_grid_diagnostics_pre_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_subgrid_force",
                                  impl->smoke_grid_subgrid_force_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_apply_face_forces",
                                  impl->smoke_grid_apply_forces_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_apply_face_boundaries",
                                  impl->smoke_grid_apply_boundaries_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_divergence_parallel",
                                  impl->smoke_grid_divergence_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_project_face_parallel",
                                  impl->smoke_grid_project_face_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_cell_diagnostics_post",
                                  impl->smoke_grid_diagnostics_post_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_particle_density",
                                  impl->smoke_particle_density_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_particle_vorticity",
                                  impl->smoke_particle_vorticity_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_particle_forces",
                                  impl->smoke_particle_forces_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_particle_integrate",
                                  impl->smoke_particle_integrate_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_advect_particles",
                                  impl->smoke_grid_advect_particles_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_emit",
                                  impl->smoke_emission_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_grid_clear",
                                  impl->smoke_grid_clear_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_predict",
                                  impl->cloth_prediction_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_damage",
                                  impl->cloth_damage_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_project_bonds",
                                  impl->cloth_project_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_apply_bonds",
                                  impl->cloth_apply_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_limit_strain_serial",
                                  impl->cloth_limit_strain_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_project_volume",
                                  impl->cloth_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_finalize",
                                  impl->cloth_finalize_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_surface_update",
                                  impl->cloth_surface_update_pipeline);
            if (status)
                status = pipeline(@"pm_cloth_impact",
                                  impl->cloth_impact_pipeline);
            if (status) status = pipeline(@"pm_cloth_rigid_surface_serial", impl->cloth_rigid_surface_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_predict",
                                  impl->soft_prediction_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_project_bonds",
                                  impl->soft_project_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_apply_bonds",
                                  impl->soft_apply_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_shape_matching",
                                  impl->soft_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_finalize",
                                  impl->soft_finalize_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_damping_prepare",
                                  impl->soft_damping_prepare_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_damping_apply",
                                  impl->soft_damping_apply_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_surface_update",
                                  impl->soft_surface_update_pipeline);
            if (status)
                status = pipeline(@"pm_soft_contact_accumulators_clear",
                                  impl->soft_contact_clear_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_measure_momentum",
                                  impl->soft_measure_momentum_pipeline);
            if (status)
                status = pipeline(@"pm_rope_predict",
                                  impl->rope_prediction_pipeline);
            if (status) status = pipeline(@"pm_rope_step_serial", impl->rope_pipeline);
            if (status)
                status = pipeline(@"pm_rope_cloth_sample",
                                  impl->rope_cloth_sample_pipeline);
            if (status)
                status = pipeline(@"pm_rope_soft_sample",
                                  impl->rope_soft_sample_pipeline);
            if (status) status = pipeline(@"pm_particle_coupling_serial", impl->coupling_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_rope_contacts",
                                  impl->fluid_rope_contact_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_rope_apply",
                                  impl->fluid_rope_apply_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_cloth_forces",
                                  impl->fluid_cloth_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_cloth_apply",
                                  impl->fluid_cloth_apply_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_cloth_surface_update",
                                  impl->fluid_cloth_surface_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_cloth_project",
                                  impl->fluid_cloth_project_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_contact",
                                  impl->fluid_soft_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_solve",
                                  impl->fluid_soft_solve_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_apply",
                                  impl->fluid_soft_apply_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_surface_update",
                                  impl->fluid_soft_surface_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_sweep_on",
                                  impl->fluid_soft_sweep_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_recover",
                                  impl->fluid_soft_recover_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_recover_current",
                                  impl->fluid_soft_recover_current_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_copy_surface",
                                  impl->fluid_soft_copy_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_soft_rigid_recover",
                                  impl->fluid_soft_rigid_recover_pipeline);
            if (status)
                status = pipeline(@"pm_soft_cloth_clear_counts",
                                  impl->soft_cloth_clear_pipeline);
            if (status)
                status = pipeline(@"pm_soft_cloth_contact",
                                  impl->soft_cloth_pipeline);
            if (status)
                status = pipeline(@"pm_soft_cloth_apply_soft",
                                  impl->soft_cloth_apply_soft_pipeline);
            if (status)
                status = pipeline(@"pm_soft_cloth_apply",
                                  impl->soft_cloth_apply_pipeline);
            if (status)
                status = pipeline(@"pm_soft_cloth_surface_update",
                                  impl->soft_cloth_surface_pipeline);
            if (status)
                status = pipeline(@"pm_soft_cloth_soft_surface_update",
                                  impl->soft_cloth_soft_surface_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_surface",
                                  impl->smoke_surface_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_surface_grid_raster",
                                  impl->smoke_surface_grid_raster_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_surface_grid_select",
                                  impl->smoke_surface_grid_select_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_surface_grid_resolve",
                                  impl->smoke_surface_grid_resolve_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_surface_grid_force",
                                  impl->smoke_surface_grid_force_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_rope",
                                  impl->smoke_rope_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_rope_wind",
                                  impl->smoke_rope_wind_pipeline);
            if (status)
                status = pipeline(@"pm_rope_soft_anchor_weights",
                                  impl->rope_soft_anchor_weight_pipeline);
            if (status)
                status = pipeline(@"pm_rope_soft_apply",
                                  impl->rope_soft_pipeline);
            if (status)
                status = pipeline(@"pm_rope_soft_surface_update",
                                  impl->rope_soft_surface_pipeline);
            if (status)
                status = pipeline(@"pm_rope_soft_update_endpoint",
                                  impl->rope_soft_endpoint_pipeline);
            if (status)
                status = pipeline(@"pm_rope_soft_finalize",
                                  impl->rope_soft_finalize_pipeline);
            if (status)
                status = pipeline(@"pm_rope_soft_pack",
                                  impl->rope_soft_pack_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_smoke_update",
                                  impl->fluid_smoke_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_smoke_compact",
                                  impl->fluid_smoke_compact_pipeline);
            if (status)
                status = pipeline(@"pm_particles_rigid",
                                  impl->particle_rigid_pipeline);
            if (status)
                status = pipeline(@"pm_soft_contact_friction",
                                  impl->soft_contact_friction_pipeline);
            if (status)
                status = pipeline(@"pm_soft_contact_finish",
                                  impl->soft_contact_finish_pipeline);
            if (status)
                status = pipeline(@"pm_soft_body_restore_momentum",
                                  impl->soft_restore_momentum_pipeline);
            if (status)
                status = pipeline(@"pm_smoke_rigid",
                                  impl->smoke_rigid_pipeline);
            if (status)
                status = pipeline(
                    @"pm_fluid_capture_spawn_baseline",
                    impl->fluid_capture_spawn_baseline_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_source", impl->source_pipeline);
            if (status)
                status = pipeline(@"pm_fluid_destroy", impl->destroy_pipeline);
            if (status)
                status = pipeline(@"pm_paint", impl->paint_pipeline);
            if (!status) return status;
            impl->fluid_contact_events = impl->buffer(
                static_cast<std::size_t>(options.contact_capacity) *
                sizeof(ContactEvent));
            impl->fluid_neighbor_overflow =
                impl->buffer(sizeof(std::uint32_t));
            impl->fluid_maximum_neighbor_count =
                impl->buffer(sizeof(std::uint32_t));
            impl->spawn_capacity_miss_count =
                impl->buffer(sizeof(SplitCounter));
            if (impl->fluid_contact_events == nil ||
                impl->fluid_neighbor_overflow == nil ||
                impl->fluid_maximum_neighbor_count == nil ||
                impl->spawn_capacity_miss_count == nil)
                return metal_failure(
                    nil, "Could not allocate Metal fluid diagnostics");
            [impl->residency commit];
            output.impl_ = std::move(impl);
            return success();
        } catch (const std::bad_alloc &) {
            return out_of_memory("Could not allocate Metal particle runtime");
        } catch (...) {
            return {StatusCode::internal_error, 0,
                    "Unexpected Metal particle runtime creation failure"};
        }
    }
}

Status MetalSystems::add_fluid(FluidOptions options,
                               HostSpan<const FluidParticle> particles,
                               FluidId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    if (options.capacity == 0U || options.capacity > INT_MAX ||
        particles.size > options.capacity ||
        (particles.size != 0U && particles.data == nullptr) ||
        options.solver_iterations == 0U || options.solver_iterations > 16U ||
        options.maximum_neighbors == 0U ||
        !std::isfinite(options.particle_radius) ||
        !(options.particle_radius > 0.0F) ||
        !std::isfinite(options.support_radius) ||
        !(options.support_radius >= options.particle_radius * 2.0F) ||
        !std::isfinite(options.rest_density) ||
        !(options.rest_density > 0.0F) ||
        !std::isfinite(options.repulsion) || options.repulsion < 0.0F ||
        !std::isfinite(options.viscosity) || options.viscosity < 0.0F ||
        !std::isfinite(options.velocity_damping) ||
        options.velocity_damping < 0.0F ||
        !std::isfinite(options.maximum_speed) ||
        !(options.maximum_speed > 0.0F) ||
        !std::isfinite(options.normal_damping) ||
        options.normal_damping < 0.0F ||
        !std::isfinite(options.rest_particle_volume) ||
        options.rest_particle_volume < 0.0F ||
        !std::isfinite(options.maximum_pair_acceleration) ||
        options.maximum_pair_acceleration < 0.0F) {
        return invalid_argument("Fluid options or initial particles are invalid");
    }
    const std::uint32_t slot = impl_->fluids.free_slot();
    if (slot == impl_->fluids.entries.size())
        return capacity_exceeded("Fluid capacity is exhausted");
    try {
        auto resource = std::make_unique<FluidResource>();
        resource->options = options;
        resource->count = static_cast<std::uint32_t>(particles.size);
        resource->initial_count = resource->count;
        const std::size_t vec_bytes = options.capacity * sizeof(Vec3);
        resource->particles.positions = impl_->buffer(vec_bytes);
        resource->particles.previous = impl_->buffer(vec_bytes);
        resource->particles.velocities = impl_->buffer(vec_bytes);
        resource->particles.inverse_masses =
            impl_->buffer(options.capacity * sizeof(float));
        resource->accelerations = impl_->buffer(vec_bytes);
        resource->stable_ids =
            impl_->buffer(options.capacity * sizeof(std::uint32_t));
        resource->foam = impl_->buffer(options.capacity * sizeof(float));
        resource->foam_sources =
            impl_->buffer(options.capacity * sizeof(float));
        resource->temperatures = impl_->buffer(options.capacity * sizeof(float));
        resource->cell_keys[0] =
            impl_->buffer(options.capacity * sizeof(std::uint64_t));
        resource->cell_keys[1] =
            impl_->buffer(options.capacity * sizeof(std::uint64_t));
        resource->sorted_indices[0] =
            impl_->buffer(options.capacity * sizeof(std::uint32_t));
        resource->sorted_indices[1] =
            impl_->buffer(options.capacity * sizeof(std::uint32_t));
        const std::size_t radix_block_count =
            (static_cast<std::size_t>(options.capacity) + 255U) / 256U;
        resource->radix_histograms = impl_->buffer(
            256U * radix_block_count * sizeof(std::uint32_t));
        resource->radix_bucket_offsets =
            impl_->buffer(256U * sizeof(std::uint32_t));
        resource->contact_samples =
            impl_->buffer(options.capacity * sizeof(ContactEvent));
        resource->contact_flags =
            impl_->buffer(options.capacity * sizeof(std::uint32_t));
        resource->metadata = impl_->buffer(sizeof(ParticleMetadata));
        resource->spawn_baseline = impl_->buffer(sizeof(std::uint32_t));
        resource->constants = impl_->buffer(sizeof(FluidConstants));
        resource->table = impl_->table(19U);
        if (resource->particles.positions == nil ||
            resource->particles.velocities == nil ||
            resource->particles.inverse_masses == nil ||
            resource->accelerations == nil || resource->stable_ids == nil ||
            resource->foam == nil || resource->foam_sources == nil ||
            resource->temperatures == nil ||
            resource->cell_keys[0] == nil ||
            resource->cell_keys[1] == nil ||
            resource->sorted_indices[0] == nil ||
            resource->sorted_indices[1] == nil ||
            resource->radix_histograms == nil ||
            resource->radix_bucket_offsets == nil ||
            resource->contact_samples == nil ||
            resource->contact_flags == nil ||
            resource->metadata == nil || resource->spawn_baseline == nil ||
            resource->constants == nil || resource->table == nil) {
            return metal_failure(nil, "Could not allocate Metal fluid buffers");
        }
        auto *positions = static_cast<Vec3 *>(resource->particles.positions.contents);
        auto *velocities = static_cast<Vec3 *>(resource->particles.velocities.contents);
        auto *inverse = static_cast<float *>(resource->particles.inverse_masses.contents);
        auto *ids = static_cast<std::uint32_t *>(resource->stable_ids.contents);
        auto *temperatures = static_cast<float *>(resource->temperatures.contents);
        for (std::uint32_t index = 0; index < resource->count; ++index) {
            if (!finite(particles.data[index].position) ||
                !finite(particles.data[index].velocity) ||
                !std::isfinite(particles.data[index].temperature)) {
                return invalid_argument("Fluid particle contains non-finite data");
            }
            positions[index] = particles.data[index].position;
            velocities[index] = particles.data[index].velocity;
            inverse[index] = 1.0F;
            ids[index] = index;
            temperatures[index] = particles.data[index].temperature;
        }
        std::memcpy(resource->particles.previous.contents,
                    resource->particles.positions.contents,
                    resource->count * sizeof(Vec3));
        impl_->bind(resource->table, 0U, resource->particles.positions);
        impl_->bind(resource->table, 1U, resource->particles.velocities);
        impl_->bind(resource->table, 2U, resource->accelerations);
        impl_->bind(resource->table, 3U, resource->stable_ids);
        impl_->bind(resource->table, 4U, resource->foam);
        impl_->bind(resource->table, 5U, resource->temperatures);
        impl_->bind(resource->table, 6U, resource->constants);
        *static_cast<ParticleMetadata *>(resource->metadata.contents) = {
            resource->count, resource->count, 1U, options.capacity,
            0U, 0U, 0U};
        *static_cast<std::uint32_t *>(resource->spawn_baseline.contents) =
            resource->count;
        impl_->bind(resource->table, 7U, resource->metadata);
        impl_->bind(resource->table, 8U, resource->particles.previous);
        impl_->bind(resource->table, 9U, resource->foam_sources);
        impl_->bind(resource->table, 10U, resource->cell_keys[0]);
        impl_->bind(resource->table, 11U, resource->cell_keys[1]);
        impl_->bind(resource->table, 12U, resource->sorted_indices[0]);
        impl_->bind(resource->table, 13U, resource->sorted_indices[1]);
        impl_->bind(resource->table, 14U,
                    impl_->fluid_neighbor_overflow);
        impl_->bind(resource->table, 15U,
                    impl_->fluid_maximum_neighbor_count);
        impl_->bind(resource->table, 16U, resource->radix_histograms);
        impl_->bind(resource->table, 17U,
                    resource->radix_bucket_offsets);
        impl_->bind(resource->table, 18U, resource->spawn_baseline);
        Status rigid_status = impl_->setup_rigid_contact(
            resource->particles, resource->rigid_constants,
            resource->rigid_table, resource->accelerations,
            options.capacity);
        if (rigid_status)
            rigid_status = impl_->setup_rigid_contact_table(
                resource->particles, resource->rigid_first_constants,
                resource->rigid_first_table, resource->accelerations,
                options.capacity);
        if (!rigid_status) return rigid_status;
        const auto bind_fluid_contact = [&](id<MTL4ArgumentTable> table) {
            impl_->bind(table, 10U, resource->stable_ids);
            impl_->bind(table, 12U, resource->contact_samples);
            impl_->bind(table, 13U, resource->contact_flags);
            impl_->bind(table, 14U, resource->metadata);
            impl_->bind(table, 17U, resource->foam);
            impl_->bind(table, 29U, resource->spawn_baseline);
        };
        bind_fluid_contact(resource->rigid_table);
        bind_fluid_contact(resource->rigid_first_table);
        [impl_->residency commit];
        impl_->fluids.entries[slot] = std::move(resource);
        ++impl_->fluids.count;
        output = {slot, impl_->fluids.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate Metal fluid state");
    }
}

Status MetalSystems::remove_fluid(FluidId id) noexcept {
    FluidResource *fluid = impl_ ? impl_->fluids.get(id) : nullptr;
    if (!fluid)
        return invalid_handle("Fluid handle is stale");
    for (const auto &entry : impl_->fluid_smoke.entries)
        if (entry && entry->options.fluid == id)
            return invalid_argument("Fluid is referenced by a smoke coupling");
    for (const auto &entry : impl_->fluid_rope.entries)
        if (entry && entry->options.fluid == id)
            return invalid_argument("Fluid is referenced by a rope coupling");
    for (const auto &entry : impl_->fluid_cloth.entries)
        if (entry && entry->options.fluid == id)
            return invalid_argument("Fluid is referenced by a cloth coupling");
    for (const auto &entry : impl_->fluid_soft.entries)
        if (entry && entry->options.fluid == id)
            return invalid_argument("Fluid is referenced by a soft-body coupling");
    for (const auto &entry : impl_->paint_rules.entries)
        if (entry && entry->options.source == id)
            return invalid_argument("Fluid is referenced by a paint rule");
    for (std::uint32_t index = 0U;
         index < impl_->sources.entries.size(); ++index)
        if (impl_->sources.entries[index] &&
            impl_->sources.entries[index]->fluid == id)
            impl_->sources.erase(ParticleSourceId{
                index, impl_->sources.generations[index]});
    for (std::uint32_t index = 0U;
         index < impl_->destroy_planes.entries.size(); ++index)
        if (impl_->destroy_planes.entries[index] &&
            impl_->destroy_planes.entries[index]->options.fluid == id)
            impl_->destroy_planes.erase(ParticleDestroyPlaneId{
                index, impl_->destroy_planes.generations[index]});
    const auto *metadata = static_cast<const ParticleMetadata *>(
        fluid->metadata.contents);
    const std::uint64_t live = std::min(
        metadata->count, fluid->options.capacity);
    impl_->archived_emitted_particle_count += metadata->emitted;
    impl_->archived_destroyed_particle_count +=
        static_cast<std::uint64_t>(fluid->initial_count) +
        metadata->emitted - live;
    impl_->archived_boiled_particle_count += metadata->boiled;
    impl_->fluids.erase(id);
    return success();
}

Status MetalSystems::fluid_view(FluidId id,
                                FluidDeviceView &output) const noexcept {
    output = {};
    const FluidResource *resource = impl_ ? impl_->fluids.get(id) : nullptr;
    if (!resource) return invalid_handle("Fluid handle is stale");
    const std::uint32_t count = std::min(
        static_cast<const ParticleMetadata *>(resource->metadata.contents)->count,
        resource->options.capacity);
    output.positions = span<Vec3>(resource->particles.positions, count);
    output.velocities = span<Vec3>(resource->particles.velocities, count);
    output.accelerations = span<Vec3>(resource->accelerations, count);
    output.stable_particle_ids =
        span<std::uint32_t>(resource->stable_ids, count);
    output.foam = span<float>(resource->foam, count);
    output.temperatures = span<float>(resource->temperatures, count);
    output.particle_count = count;
    output.particle_radius = resource->options.particle_radius;
    output.support_radius = resource->options.support_radius;
    output.revision = resource->revision;
    return success();
}

Status MetalSystems::append_hit_box_particles(
    HitBox box, std::vector<HitBoxParticle> &output) const noexcept {
    if (!impl_) return invalid_argument("World is not initialized");
    const Quaternion inverse{-box.orientation.x, -box.orientation.y,
                             -box.orientation.z, box.orientation.w};
    try {
        for (std::uint32_t slot = 0U;
             slot < impl_->fluids.entries.size(); ++slot) {
            const FluidResource *resource = impl_->fluids.entries[slot].get();
            if (!resource) continue;
            const std::uint32_t count = std::min(
                static_cast<const ParticleMetadata *>(
                    resource->metadata.contents)->count,
                resource->options.capacity);
            const auto *positions = static_cast<const Vec3 *>(
                resource->particles.positions.contents);
            const auto *stable_ids = static_cast<const std::uint32_t *>(
                resource->stable_ids.contents);
            const FluidId fluid{slot, impl_->fluids.generations[slot]};
            for (std::uint32_t particle = 0U; particle < count; ++particle) {
                const Vec3 relative = subtract(positions[particle], box.center);
                const Vec3 local = add(
                    relative,
                    multiply(cross(Vec3{inverse.x, inverse.y, inverse.z},
                                   add(cross(Vec3{inverse.x, inverse.y,
                                                  inverse.z}, relative),
                                       multiply(relative, inverse.w))),
                             2.0F));
                constexpr float boundary_epsilon = 1.0e-6F;
                if (std::abs(local.x) <=
                        box.half_extents.x + boundary_epsilon &&
                    std::abs(local.y) <=
                        box.half_extents.y + boundary_epsilon &&
                    std::abs(local.z) <=
                        box.half_extents.z + boundary_epsilon) {
                    output.push_back({fluid, stable_ids[particle]});
                }
            }
        }
    } catch (const std::bad_alloc &) {
        return out_of_memory("Hit-box particle result allocation failed");
    } catch (...) {
        return out_of_memory("Hit-box particle result allocation failed");
    }
    return success();
}

Status MetalSystems::add_smoke(SmokeOptions options, SmokeId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    if (options.capacity == 0U || options.capacity > 1'000'000U ||
        !finite(options.emitter_center) || !finite(options.initial_velocity) ||
        !finite(options.wind) ||
        !std::isfinite(options.emitter_half_extents.x) ||
        !std::isfinite(options.emitter_half_extents.y) ||
        options.emitter_half_extents.x <= 0.0F ||
        options.emitter_half_extents.y <= 0.0F ||
        !std::isfinite(options.particles_per_second) ||
        options.particles_per_second <= 0.0F ||
        options.particles_per_second > 1'000'000.0F ||
        !std::isfinite(options.lifetime) || !(options.lifetime > 0.0F) ||
        !std::isfinite(options.particle_radius) ||
        !(options.particle_radius > 0.0F) ||
        !std::isfinite(options.buoyancy) || !std::isfinite(options.response) ||
        options.response < 0.0F ||
        !std::isfinite(options.rest_number_density) ||
        options.rest_number_density <= 0.0F ||
        !std::isfinite(options.pressure_stiffness) ||
        options.pressure_stiffness < 0.0F ||
        !std::isfinite(options.viscosity) || options.viscosity < 0.0F ||
        !std::isfinite(options.vorticity_confinement) ||
        options.vorticity_confinement < 0.0F ||
        !std::isfinite(options.maximum_speed) ||
        !(options.maximum_speed > 0.0F) ||
        (options.grid_resolution != 0U &&
         (options.grid_resolution < 16U || options.grid_resolution > 256U ||
          options.grid_vertical_resolution < 8U ||
          options.grid_vertical_resolution > 256U ||
          options.grid_pressure_iterations < 4U ||
          options.grid_pressure_iterations > 128U ||
          !std::isfinite(options.grid_kinematic_viscosity) ||
          options.grid_kinematic_viscosity < 0.0F ||
          !std::isfinite(options.grid_les_coefficient) ||
          options.grid_les_coefficient < 0.0F ||
          !std::isfinite(options.grid_pressure_tolerance) ||
          options.grid_pressure_tolerance <= 0.0F ||
          options.grid_pressure_tolerance > 1.0F)) ||
        !finite(options.grid_minimum) ||
        !std::isfinite(options.grid_edge_length) ||
        options.grid_edge_length < 0.0F) {
        return invalid_argument("Smoke options are invalid");
    }
    const std::uint32_t slot = impl_->smokes.free_slot();
    if (slot == impl_->smokes.entries.size())
        return capacity_exceeded("Smoke capacity is exhausted");
    try {
        auto resource = std::make_unique<SmokeResource>();
        resource->options = options;
        if (resource->options.grid_resolution != 0U) {
            if (resource->options.grid_vertical_resolution == 0U ||
                resource->options.grid_pressure_iterations == 0U)
                return invalid_argument("Smoke grid dimensions are invalid");
            if (!(resource->options.grid_edge_length > 0.0F)) {
                const Vec3 end{
                    options.emitter_center.x +
                        options.wind.x * options.lifetime,
                    options.emitter_center.y +
                        options.wind.y * options.lifetime,
                    options.emitter_center.z +
                        options.wind.z * options.lifetime};
                const Vec3 low{
                    std::min(options.emitter_center.x, end.x) - 1.5F,
                    options.emitter_center.y,
                    std::min(options.emitter_center.z, end.z) - 1.5F};
                const Vec3 high{
                    std::max(options.emitter_center.x, end.x) + 1.5F,
                    options.emitter_center.y,
                    std::max(options.emitter_center.z, end.z) + 1.5F};
                const float edge = std::max(high.x - low.x,
                                            high.z - low.z);
                const Vec3 center{0.5F * (low.x + high.x),
                                  0.5F * (low.y + high.y),
                                  0.5F * (low.z + high.z)};
                const float spacing = edge / static_cast<float>(
                    resource->options.grid_resolution);
                resource->options.grid_edge_length = edge;
                resource->options.grid_minimum = {
                    center.x - 0.5F * edge,
                    center.y - 0.5F * spacing * static_cast<float>(
                        resource->options.grid_vertical_resolution),
                    center.z - 0.5F * edge};
            }
            const float spacing = resource->options.grid_edge_length /
                static_cast<float>(resource->options.grid_resolution);
            if (!std::isfinite(spacing) || spacing < 1.0e-4F)
                return invalid_argument("Smoke grid cell spacing is invalid");
        }
        const std::size_t vec_bytes = options.capacity * sizeof(Vec3);
        const std::size_t scalar_bytes = options.capacity * sizeof(float);
        resource->particles.positions = impl_->buffer(vec_bytes);
        resource->particles.previous = impl_->buffer(vec_bytes);
        resource->particles.velocities = impl_->buffer(vec_bytes);
        resource->particles.inverse_masses = impl_->buffer(scalar_bytes);
        resource->ages = impl_->buffer(scalar_bytes);
        resource->densities = impl_->buffer(scalar_bytes);
        resource->pressures = impl_->buffer(scalar_bytes);
        resource->vorticities = impl_->buffer(vec_bytes);
        resource->metadata = impl_->buffer(sizeof(SmokeMetadata));
        resource->constants = impl_->buffer(sizeof(SmokeConstants));
        resource->advection_constants =
            impl_->buffer(sizeof(SmokeConstants));
        resource->emission_constants = impl_->buffer(sizeof(SmokeConstants));
        const std::uint64_t cell_count =
            static_cast<std::uint64_t>(options.grid_resolution) *
            options.grid_vertical_resolution * options.grid_resolution;
        const std::uint64_t face_count_x =
            static_cast<std::uint64_t>(options.grid_resolution + 1U) *
            options.grid_vertical_resolution * options.grid_resolution;
        const std::uint64_t face_count_y =
            static_cast<std::uint64_t>(options.grid_resolution) *
            (options.grid_vertical_resolution + 1U) * options.grid_resolution;
        const std::uint64_t face_count_z =
            static_cast<std::uint64_t>(options.grid_resolution) *
            options.grid_vertical_resolution * (options.grid_resolution + 1U);
        const std::uint64_t face_count =
            face_count_x + face_count_y + face_count_z;
        const auto coarse_counts = [](std::uint64_t n,
                                      std::uint64_t height) {
            n = std::max<std::uint64_t>(2U, n / 2U);
            height = std::max<std::uint64_t>(2U, height / 2U);
            const std::uint64_t cells = n * height * n;
            const std::uint64_t faces =
                (n + 1U) * height * n +
                n * (height + 1U) * n +
                n * height * (n + 1U);
            return std::array<std::uint64_t, 4>{n, height, cells, faces};
        };
        const auto level1 = coarse_counts(options.grid_resolution,
                                          options.grid_vertical_resolution);
        const auto level2 = coarse_counts(level1[0], level1[1]);
        const auto level3 = coarse_counts(level2[0], level2[1]);
        const std::uint64_t pressure_scratch_count =
            2U * cell_count + 4U * level1[2] + level1[3] +
            4U * level2[2] + level2[3] +
            3U * level3[2] + level3[3];
        const std::uint64_t advection_scratch_count =
            std::max(2U * face_count, pressure_scratch_count);
        const std::uint64_t splat_contribution_count =
            options.grid_resolution == 0U
                ? 0U
                : 27U * static_cast<std::uint64_t>(options.capacity);
        const std::uint64_t splat_block_count =
            (splat_contribution_count + 255U) / 256U;
        const std::uint64_t splat_histogram_count =
            256U * splat_block_count;
        if (cell_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3))
            return capacity_exceeded("Smoke grid is too large");
        if (advection_scratch_count >
            std::numeric_limits<std::size_t>::max() / sizeof(float))
            return capacity_exceeded("Smoke face grid is too large");
        if (splat_contribution_count >
                std::numeric_limits<std::size_t>::max() /
                    sizeof(SmokeGridContribution) ||
            splat_histogram_count >
                std::numeric_limits<std::size_t>::max() /
                    sizeof(std::uint32_t))
            return capacity_exceeded("Smoke splat grid is too large");
        resource->grid_velocity = impl_->buffer(cell_count * sizeof(Vec3));
        resource->grid_pressure = impl_->buffer(cell_count * sizeof(float));
        resource->grid_density = impl_->buffer(cell_count * sizeof(float));
        resource->grid_temperature = impl_->buffer(cell_count * sizeof(float));
        resource->grid_solid = impl_->buffer(cell_count * sizeof(std::uint32_t));
        resource->grid_vorticity = impl_->buffer(cell_count * sizeof(Vec3));
        resource->grid_divergence = impl_->buffer(cell_count * sizeof(float));
        resource->grid_scratch = impl_->buffer(cell_count * sizeof(Vec3));
        resource->grid_pressure_relative_residual =
            impl_->buffer(sizeof(float));
        resource->grid_deformable_solid =
            impl_->buffer(cell_count * sizeof(std::uint32_t));
        resource->grid_rigid_bodies = impl_->buffer(
            std::max<std::size_t>(1U,
                impl_->options.smoke_rigid_coupling_capacity) *
            sizeof(SmokeRigidRasterEntry));
        resource->grid_rigid_body_count =
            impl_->buffer(4U * sizeof(std::uint32_t));
        resource->grid_face_velocity =
            impl_->buffer(face_count * sizeof(float));
        resource->grid_face_advection =
            impl_->buffer(advection_scratch_count * sizeof(float));
        resource->grid_face_boundary =
            impl_->buffer(2U * face_count * sizeof(float));
        resource->grid_face_nearest_triangle =
            impl_->buffer(face_count * sizeof(std::uint32_t));
        resource->grid_face_normal =
            impl_->buffer(face_count * sizeof(Vec3));
        for (std::uint32_t index = 0U; index < 2U; ++index) {
            resource->grid_splat_keys[index] = impl_->buffer(
                splat_contribution_count * sizeof(std::uint32_t));
            resource->grid_splat_contributions[index] = impl_->buffer(
                splat_contribution_count * sizeof(SmokeGridContribution));
        }
        resource->grid_splat_histograms = impl_->buffer(
            splat_histogram_count * sizeof(std::uint32_t));
        resource->grid_splat_bucket_offsets =
            impl_->buffer(256U * sizeof(std::uint32_t));
        resource->grid_pressure_state =
            impl_->buffer(3U * sizeof(std::uint32_t));
        resource->table = impl_->table(31U);
        resource->grid_pressure_table = impl_->table(13U);
        resource->advection_table = impl_->table(31U);
        resource->emission_table = impl_->table(10U);
        resource->grid_splat_table = impl_->table(13U);
        resource->grid_raster_table = impl_->table(16U);
        const bool missing_grid_buffers = options.grid_resolution != 0U &&
            (resource->grid_velocity == nil ||
             resource->grid_pressure == nil ||
             resource->grid_density == nil ||
             resource->grid_temperature == nil ||
             resource->grid_solid == nil ||
             resource->grid_vorticity == nil ||
             resource->grid_divergence == nil ||
             resource->grid_scratch == nil);
        if (resource->particles.positions == nil ||
            resource->particles.previous == nil ||
            resource->particles.velocities == nil || resource->ages == nil ||
            resource->densities == nil || resource->pressures == nil ||
            resource->vorticities == nil || resource->metadata == nil ||
            resource->constants == nil ||
            resource->advection_constants == nil ||
            resource->emission_constants == nil || resource->table == nil ||
            resource->grid_pressure_table == nil ||
            resource->advection_table == nil ||
            resource->emission_table == nil ||
            resource->grid_pressure_relative_residual == nil ||
            resource->grid_deformable_solid == nil ||
            resource->grid_rigid_bodies == nil ||
            resource->grid_rigid_body_count == nil ||
            resource->grid_face_velocity == nil ||
            resource->grid_face_advection == nil ||
            resource->grid_face_boundary == nil ||
            resource->grid_face_nearest_triangle == nil ||
            resource->grid_face_normal == nil ||
            resource->grid_splat_keys[0] == nil ||
            resource->grid_splat_keys[1] == nil ||
            resource->grid_splat_contributions[0] == nil ||
            resource->grid_splat_contributions[1] == nil ||
            resource->grid_splat_histograms == nil ||
            resource->grid_splat_bucket_offsets == nil ||
            resource->grid_pressure_state == nil ||
            resource->grid_splat_table == nil ||
            resource->grid_raster_table == nil ||
            missing_grid_buffers) {
            return metal_failure(nil, "Could not allocate Metal smoke buffers");
        }
        const auto bind_smoke_step = [&](id<MTL4ArgumentTable> table,
                                         id<MTLBuffer> constants) {
            impl_->bind(table, 0U, resource->particles.positions);
            impl_->bind(table, 1U, resource->particles.velocities);
            impl_->bind(table, 2U, resource->ages);
            impl_->bind(table, 3U, resource->densities);
            impl_->bind(table, 4U, resource->pressures);
            impl_->bind(table, 5U, resource->vorticities);
            impl_->bind(table, 6U, resource->metadata);
            impl_->bind(table, 7U, constants);
            impl_->bind(table, 8U, resource->particles.inverse_masses);
            impl_->bind(table, 9U, resource->grid_velocity);
            impl_->bind(table, 10U, resource->grid_pressure);
            impl_->bind(table, 11U, resource->grid_density);
            impl_->bind(table, 12U, resource->grid_temperature);
            impl_->bind(table, 13U, resource->grid_solid);
            impl_->bind(table, 14U, resource->grid_vorticity);
            impl_->bind(table, 15U, resource->grid_divergence);
            impl_->bind(table, 16U, resource->particles.previous);
            impl_->bind(table, 17U, resource->grid_scratch);
            impl_->bind(table, 18U,
                        resource->grid_pressure_relative_residual);
            impl_->bind(table, 19U, resource->grid_pressure_state);
            impl_->bind(table, 20U, resource->grid_rigid_body_count);
            impl_->bind(table, 27U, resource->grid_deformable_solid);
            impl_->bind(table, 28U, resource->grid_face_velocity);
            impl_->bind(table, 29U, resource->grid_face_advection);
            impl_->bind(table, 30U, resource->grid_face_boundary);
        };
        bind_smoke_step(resource->table, resource->constants);
        bind_smoke_step(resource->advection_table,
                        resource->advection_constants);
        impl_->bind(resource->grid_pressure_table, 0U,
                    resource->constants);
        impl_->bind(resource->grid_pressure_table, 1U,
                    resource->grid_pressure);
        impl_->bind(resource->grid_pressure_table, 2U,
                    resource->grid_divergence);
        impl_->bind(resource->grid_pressure_table, 3U,
                    resource->grid_face_boundary);
        impl_->bind(resource->grid_pressure_table, 4U,
                    resource->grid_face_advection);
        impl_->bind(resource->grid_pressure_table, 5U,
                    resource->grid_pressure_relative_residual);
        impl_->bind(resource->grid_pressure_table, 6U,
                    resource->grid_pressure_state);
        impl_->bind(resource->grid_pressure_table, 7U,
                    resource->grid_face_velocity);
        impl_->bind(resource->grid_pressure_table, 8U,
                    resource->grid_velocity);
        impl_->bind(resource->grid_pressure_table, 9U,
                    resource->grid_vorticity);
        impl_->bind(resource->grid_pressure_table, 10U,
                    resource->grid_density);
        impl_->bind(resource->grid_pressure_table, 11U,
                    resource->grid_temperature);
        impl_->bind(resource->grid_pressure_table, 12U,
                    resource->grid_scratch);
        impl_->bind(resource->grid_splat_table, 0U,
                    resource->particles.positions);
        impl_->bind(resource->grid_splat_table, 1U, resource->ages);
        impl_->bind(resource->grid_splat_table, 2U,
                    resource->particles.inverse_masses);
        impl_->bind(resource->grid_splat_table, 3U, resource->metadata);
        impl_->bind(resource->grid_splat_table, 4U, resource->constants);
        impl_->bind(resource->grid_splat_table, 5U,
                    resource->grid_splat_keys[0]);
        impl_->bind(resource->grid_splat_table, 6U,
                    resource->grid_splat_keys[1]);
        impl_->bind(resource->grid_splat_table, 7U,
                    resource->grid_splat_contributions[0]);
        impl_->bind(resource->grid_splat_table, 8U,
                    resource->grid_splat_contributions[1]);
        impl_->bind(resource->grid_splat_table, 9U,
                    resource->grid_splat_histograms);
        impl_->bind(resource->grid_splat_table, 10U,
                    resource->grid_splat_bucket_offsets);
        impl_->bind(resource->grid_splat_table, 11U,
                    resource->grid_density);
        impl_->bind(resource->grid_splat_table, 12U,
                    resource->grid_temperature);
        impl_->bind(resource->grid_raster_table, 0U, resource->constants);
        impl_->bind(resource->grid_raster_table, 1U,
                    resource->grid_deformable_solid);
        impl_->bind(resource->grid_raster_table, 2U,
                    resource->grid_face_boundary);
        impl_->bind(resource->grid_raster_table, 3U,
                    resource->grid_face_nearest_triangle);
        impl_->bind(resource->grid_raster_table, 4U,
                    resource->grid_face_normal);
        impl_->bind(resource->grid_raster_table, 5U,
                    resource->grid_rigid_bodies);
        impl_->bind(resource->grid_raster_table, 6U,
                    resource->grid_rigid_body_count);
        impl_->bind(resource->grid_raster_table, 13U,
                    resource->grid_velocity);
        impl_->bind(resource->grid_raster_table, 14U,
                    resource->grid_scratch);
        impl_->bind(resource->grid_raster_table, 15U,
                    resource->grid_solid);
        impl_->bind(resource->emission_table, 0U,
                    resource->particles.positions);
        impl_->bind(resource->emission_table, 1U,
                    resource->particles.previous);
        impl_->bind(resource->emission_table, 2U,
                    resource->particles.velocities);
        impl_->bind(resource->emission_table, 3U, resource->ages);
        impl_->bind(resource->emission_table, 4U, resource->densities);
        impl_->bind(resource->emission_table, 5U, resource->pressures);
        impl_->bind(resource->emission_table, 6U, resource->vorticities);
        impl_->bind(resource->emission_table, 7U,
                    resource->particles.inverse_masses);
        impl_->bind(resource->emission_table, 8U, resource->metadata);
        impl_->bind(resource->emission_table, 9U,
                    resource->emission_constants);
        if (face_count != 0U) {
            auto *face_velocity = static_cast<float *>(
                resource->grid_face_velocity.contents);
            std::fill_n(face_velocity, face_count_x, options.wind.x);
            std::fill_n(face_velocity + face_count_x, face_count_y,
                        options.wind.y);
            std::fill_n(face_velocity + face_count_x + face_count_y,
                        face_count_z, options.wind.z);
            auto *face_boundary = static_cast<float *>(
                resource->grid_face_boundary.contents);
            std::fill_n(face_boundary, face_count, 1.0F);
            std::fill_n(face_boundary + face_count, face_count, 0.0F);
        }
        [impl_->residency commit];
        impl_->smokes.entries[slot] = std::move(resource);
        ++impl_->smokes.count;
        output = {slot, impl_->smokes.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate Metal smoke state");
    }
}

Status MetalSystems::remove_smoke(SmokeId id) noexcept {
    if (!impl_ || !impl_->smokes.get(id))
        return invalid_handle("Smoke handle is stale");
    for (const auto &entry : impl_->fluid_smoke.entries)
        if (entry && entry->options.smoke == id)
            return invalid_argument("Smoke is referenced by a fluid coupling");
    for (const auto &entry : impl_->smoke_soft.entries)
        if (entry && entry->options.smoke == id)
            return invalid_argument("Smoke is referenced by a soft-body coupling");
    for (const auto &entry : impl_->smoke_cloth.entries)
        if (entry && entry->options.smoke == id)
            return invalid_argument("Smoke is referenced by a cloth coupling");
    for (const auto &entry : impl_->smoke_rope.entries)
        if (entry && entry->options.smoke == id)
            return invalid_argument("Smoke is referenced by a rope coupling");
    for (const auto &entry : impl_->smoke_rigid.entries)
        if (entry && entry->options.smoke == id)
            return invalid_argument("Smoke is referenced by a rigid coupling");
    impl_->smokes.erase(id);
    return success();
}

Status MetalSystems::smoke_view(SmokeId id,
                                SmokeDeviceView &output) const noexcept {
    output = {};
    const SmokeResource *resource = impl_ ? impl_->smokes.get(id) : nullptr;
    if (!resource) return invalid_handle("Smoke handle is stale");
    const auto *metadata =
        static_cast<const SmokeMetadata *>(resource->metadata.contents);
    const std::uint32_t count = std::min(metadata->count, resource->options.capacity);
    output.positions = span<Vec3>(resource->particles.positions, count);
    output.velocities = span<Vec3>(resource->particles.velocities, count);
    output.ages = span<float>(resource->ages, count);
    output.number_densities = span<float>(resource->densities, count);
    output.pressures = span<float>(resource->pressures, count);
    output.vorticities = span<Vec3>(resource->vorticities, count);
    output.particle_count = count;
    output.lifetime = resource->options.lifetime;
    output.particle_radius = resource->options.particle_radius;
    output.revision = resource->revision + metadata->revision;
    const std::uint64_t cells =
        static_cast<std::uint64_t>(resource->options.grid_resolution) *
        resource->options.grid_vertical_resolution *
        resource->options.grid_resolution;
    output.grid_velocity = span<Vec3>(resource->grid_velocity, cells);
    output.grid_pressure = span<float>(resource->grid_pressure, cells);
    output.grid_density = span<float>(resource->grid_density, cells);
    output.grid_temperature = span<float>(resource->grid_temperature, cells);
    output.grid_solid = span<std::uint32_t>(resource->grid_solid, cells);
    output.grid_vorticity = span<Vec3>(resource->grid_vorticity, cells);
    output.grid_divergence = span<float>(resource->grid_divergence, cells);
    output.grid_resolution = resource->options.grid_resolution;
    output.grid_vertical_resolution = resource->options.grid_vertical_resolution;
    output.grid_minimum = resource->options.grid_minimum;
    output.grid_spacing = resource->options.grid_resolution == 0U
                              ? 0.0F
                              : resource->options.grid_edge_length /
                                    resource->options.grid_resolution;
    output.grid_pressure_relative_residual =
        *static_cast<const float *>(
            resource->grid_pressure_relative_residual.contents);
    return success();
}

Status MetalSystems::add_cloth(ClothOptions options, ClothId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    if (!options.vertices.data || options.vertices.size < 3U ||
        options.vertices.size > UINT32_MAX ||
        !options.triangle_indices.data || options.triangle_indices.size < 3U ||
        options.triangle_indices.size > UINT32_MAX ||
        options.triangle_indices.size % 3U != 0U ||
        (options.inverse_masses.size != 0U &&
         (!options.inverse_masses.data ||
          options.inverse_masses.size != options.vertices.size)) ||
        !std::isfinite(options.vertex_mass) || !(options.vertex_mass > 0.0F) ||
        !std::isfinite(options.thickness) || !(options.thickness > 0.0F) ||
        !std::isfinite(options.stretch_compliance) ||
        options.stretch_compliance < 0.0F ||
        !std::isfinite(options.bending_compliance) ||
        options.bending_compliance < 0.0F ||
        !std::isfinite(options.velocity_damping) ||
        options.velocity_damping < 0.0F ||
        !std::isfinite(options.contact_friction) ||
        options.contact_friction < 0.0F ||
        !std::isfinite(options.break_strain) || options.break_strain < 0.0F ||
        options.break_strain > 9.0F ||
        options.fracture_persistence_substeps == 0U ||
        options.fracture_persistence_substeps > 64U ||
        !std::isfinite(options.impact_break_impulse) ||
        options.impact_break_impulse < 0.0F ||
        !std::isfinite(options.target_volume) || options.target_volume < 0.0F ||
        !std::isfinite(options.volume_compliance) ||
        options.volume_compliance < 0.0F ||
        (options.preserve_volume &&
         (options.break_strain > 0.0F ||
          options.impact_break_impulse > 0.0F)) ||
        options.solver_iterations == 0U || options.solver_iterations > 64U) {
        return invalid_argument("Cloth options are invalid");
    }
    const std::uint32_t slot = impl_->cloths.free_slot();
    if (slot == impl_->cloths.entries.size())
        return capacity_exceeded("Cloth capacity is exhausted");
    try {
        auto resource = std::make_unique<ClothResource>();
        resource->options = options;
        resource->count = static_cast<std::uint32_t>(options.vertices.size);
        resource->triangle_index_count =
            static_cast<std::uint32_t>(options.triangle_indices.size);
        const bool fracture_enabled = options.break_strain > 0.0F ||
                                      options.impact_break_impulse > 0.0F;
        for (std::uint32_t vertex = 0; vertex < resource->count; ++vertex) {
            if (!finite(options.vertices.data[vertex]) ||
                (options.inverse_masses.size != 0U &&
                 (!std::isfinite(options.inverse_masses.data[vertex]) ||
                  options.inverse_masses.data[vertex] < 0.0F)))
                return invalid_argument(
                    "Cloth vertex or inverse mass is invalid");
        }
        std::vector<MetalBond> bonds;
        std::vector<ClothBond> public_bonds;
        std::vector<std::vector<DeformableNeighbor>> adjacency(
            resource->count);
        std::vector<std::uint32_t> triangle_bonds;
        std::unordered_map<std::uint64_t, std::uint32_t> bond_ids;
        std::unordered_map<std::uint64_t, std::uint32_t> opposite;
        std::unordered_map<std::uint64_t, std::uint32_t> edge_counts;
        const auto edge_key = [](std::uint32_t first,
                                 std::uint32_t second) {
            if (first > second) std::swap(first, second);
            return (static_cast<std::uint64_t>(first) << 32U) | second;
        };
        const auto link = [&](std::uint32_t first, std::uint32_t second,
                              float compliance, bool bending) {
            if (first == second ||
                bond_ids.find(edge_key(first, second)) != bond_ids.end())
                return;
            const Vec3 a = options.vertices.data[first];
            const Vec3 b = options.vertices.data[second];
            const float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
            const float rest = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (!(rest > 1.0e-6F)) return;
            bond_ids.emplace(edge_key(first, second),
                             static_cast<std::uint32_t>(bonds.size()));
            const std::uint32_t bond =
                static_cast<std::uint32_t>(bonds.size());
            bonds.push_back(
                {first, second, rest, compliance, 1U});
            public_bonds.push_back({first, second, rest, bending});
            adjacency[first].push_back(
                {second, rest, compliance, bond});
            adjacency[second].push_back(
                {first, rest, compliance, bond});
        };
        float signed_volume = 0.0F;
        for (std::uint64_t triangle = 0; triangle < options.triangle_indices.size;
             triangle += 3U) {
            const std::uint32_t vertices[3]{
                options.triangle_indices.data[triangle],
                options.triangle_indices.data[triangle + 1U],
                options.triangle_indices.data[triangle + 2U]};
            if (vertices[0] >= resource->count ||
                vertices[1] >= resource->count ||
                vertices[2] >= resource->count ||
                vertices[0] == vertices[1] || vertices[1] == vertices[2] ||
                vertices[2] == vertices[0])
                return invalid_argument("Cloth triangle is invalid");
            const Vec3 a = options.vertices.data[vertices[0]];
            const Vec3 b = options.vertices.data[vertices[1]];
            const Vec3 c = options.vertices.data[vertices[2]];
            if (dot(subtract(a, b), subtract(a, b)) <= 1.0e-12F ||
                dot(subtract(b, c), subtract(b, c)) <= 1.0e-12F ||
                dot(subtract(c, a), subtract(c, a)) <= 1.0e-12F)
                return invalid_argument(
                    "Cloth triangle has a zero-length edge");
            for (std::uint32_t edge = 0; edge < 3U; ++edge) {
                const std::uint32_t first = vertices[edge];
                const std::uint32_t second = vertices[(edge + 1U) % 3U];
                const std::uint32_t across = vertices[(edge + 2U) % 3U];
                if (!finite(options.vertices.data[first]) ||
                    !finite(options.vertices.data[second]))
                    return invalid_argument("Cloth contains invalid geometry");
                const std::uint64_t key = edge_key(first, second);
                ++edge_counts[key];
                const auto previous = opposite.find(key);
                if (previous == opposite.end()) {
                    opposite.emplace(key, across);
                    link(first, second, options.stretch_compliance, false);
                } else {
                    link(previous->second, across,
                         options.bending_compliance, true);
                }
                triangle_bonds.push_back(bond_ids.at(key));
            }
            signed_volume += dot(a, cross(b, c)) / 6.0F;
        }
        resource->orientation = signed_volume < 0.0F ? -1.0F : 1.0F;
        if (options.preserve_volume) {
            if (std::abs(signed_volume) <= 1.0e-8F ||
                std::any_of(edge_counts.begin(), edge_counts.end(),
                            [](const auto &entry) {
                                return entry.second != 2U;
                            }))
                return invalid_argument(
                    "Volume-preserving cloth must be a closed manifold");
            const float target = options.target_volume > 0.0F
                                     ? options.target_volume
                                     : std::abs(signed_volume);
            resource->options.target_volume =
                std::copysign(target, signed_volume);
        }
        resource->bond_count = static_cast<std::uint32_t>(bonds.size());
        if (fracture_enabled &&
            options.vertices.size + options.triangle_indices.size >
                UINT32_MAX)
            return capacity_exceeded("Cloth split capacity is too large");
        resource->capacity = resource->count +
            (fracture_enabled ? resource->triangle_index_count : 0U);
        resource->surface_count = fracture_enabled
                                      ? resource->triangle_index_count
                                      : resource->count;
        const std::size_t vec_bytes = resource->capacity * sizeof(Vec3);
        resource->particles.positions = impl_->buffer(vec_bytes);
        resource->particles.previous = impl_->buffer(vec_bytes);
        resource->particles.velocities = impl_->buffer(vec_bytes);
        resource->particles.inverse_masses =
            impl_->buffer(resource->capacity * sizeof(float));
        resource->triangle_indices =
            impl_->buffer(resource->triangle_index_count * sizeof(std::uint32_t));
        resource->source_indices =
            impl_->buffer(resource->capacity * sizeof(std::uint32_t));
        resource->surface_positions =
            impl_->buffer(resource->surface_count * sizeof(Vec3));
        resource->surface_indices =
            impl_->buffer(resource->triangle_index_count * sizeof(std::uint32_t));
        resource->surface_source_indices =
            impl_->buffer(resource->surface_count * sizeof(std::uint32_t));
        resource->surface_physical_indices =
            impl_->buffer(resource->surface_count * sizeof(std::uint32_t));
        resource->public_bonds = impl_->buffer(bonds.size() * sizeof(ClothBond));
        resource->bonds = impl_->buffer(bonds.size() * sizeof(MetalBond));
        resource->active_bonds = impl_->buffer(bonds.size() * sizeof(std::uint8_t));
        resource->bond_damage = impl_->buffer(bonds.size() * sizeof(std::uint8_t));
        const std::uint64_t neighbor_capacity = 2U *
            (fracture_enabled
                 ? static_cast<std::uint64_t>(
                       resource->triangle_index_count) + bonds.size()
                 : bonds.size());
        if (neighbor_capacity > UINT32_MAX)
            return capacity_exceeded("Cloth split links exceed uint32 range");
        resource->neighbor_capacity =
            static_cast<std::uint32_t>(neighbor_capacity);
        resource->neighbor_offsets = impl_->buffer(
            (static_cast<std::size_t>(resource->capacity) + 1U) *
            sizeof(std::uint32_t));
        resource->neighbors = impl_->buffer(
            static_cast<std::size_t>(resource->neighbor_capacity) *
            sizeof(DeformableNeighbor));
        resource->free_triangle_nodes = impl_->buffer(
            static_cast<std::size_t>(resource->capacity) *
            sizeof(std::uint8_t));
        resource->corrections = impl_->buffer(vec_bytes);
        resource->rigid_forces = impl_->buffer(vec_bytes);
        resource->fluid_forces = impl_->buffer(vec_bytes);
        resource->soft_forces = impl_->buffer(vec_bytes);
        resource->rope_forces = impl_->buffer(vec_bytes);
        resource->smoke_forces = impl_->buffer(vec_bytes);
        resource->constants = impl_->buffer(sizeof(DeformableConstants));
        resource->table = impl_->table(17U);
        resource->body_corrections = impl_->buffer(
            static_cast<std::size_t>(impl_->options.rigid_body_capacity) *
            sizeof(ClothBodyCorrection));
        resource->surface_rigid_constants =
            impl_->buffer(sizeof(ClothRigidConstants));
        resource->surface_rigid_table = impl_->table(13U);
        if (resource->particles.positions == nil ||
            resource->particles.previous == nil ||
            resource->particles.velocities == nil || resource->bonds == nil ||
            resource->bond_damage == nil ||
            resource->neighbor_offsets == nil || resource->neighbors == nil ||
            resource->free_triangle_nodes == nil ||
            resource->corrections == nil ||
            resource->surface_physical_indices == nil ||
            resource->smoke_forces == nil ||
            resource->constants == nil || resource->table == nil ||
            resource->body_corrections == nil ||
            resource->surface_rigid_constants == nil ||
            resource->surface_rigid_table == nil)
            return metal_failure(nil, "Could not allocate Metal cloth buffers");
        copy_to(resource->particles.positions, options.vertices.data, resource->count);
        copy_to(resource->particles.previous, options.vertices.data, resource->count);
        copy_to(resource->triangle_indices, options.triangle_indices.data,
                resource->triangle_index_count);
        copy_to(resource->bonds, bonds.data(), bonds.size());
        copy_to(resource->public_bonds, public_bonds.data(), public_bonds.size());
        auto *inverse = static_cast<float *>(resource->particles.inverse_masses.contents);
        auto *sources = static_cast<std::uint32_t *>(resource->source_indices.contents);
        auto *surface_positions =
            static_cast<Vec3 *>(resource->surface_positions.contents);
        auto *surface_indices =
            static_cast<std::uint32_t *>(resource->surface_indices.contents);
        auto *surface_sources = static_cast<std::uint32_t *>(
            resource->surface_source_indices.contents);
        auto *surface_physical = static_cast<std::uint32_t *>(
            resource->surface_physical_indices.contents);
        auto *active = static_cast<std::uint8_t *>(resource->active_bonds.contents);
        for (std::uint32_t index = 0; index < resource->count; ++index) {
            inverse[index] = options.inverse_masses.data &&
                                     index < options.inverse_masses.size
                                 ? options.inverse_masses.data[index]
                                 : 1.0F / options.vertex_mass;
            sources[index] = index;
        }
        if (fracture_enabled) {
            for (std::uint32_t index = 0; index < resource->surface_count;
                 ++index) {
                const std::uint32_t source = options.triangle_indices.data[index];
                surface_positions[index] = options.vertices.data[source];
                surface_indices[index] = index;
                surface_sources[index] = source;
                surface_physical[index] = source;
            }
        } else {
            copy_to(resource->surface_positions, options.vertices.data,
                    resource->count);
            copy_to(resource->surface_indices, options.triangle_indices.data,
                    resource->triangle_index_count);
            for (std::uint32_t index = 0; index < resource->surface_count;
                 ++index) {
                surface_sources[index] = index;
                surface_physical[index] = index;
            }
        }
        std::fill(active, active + bonds.size(), std::uint8_t{1U});
        std::memset(resource->bond_damage.contents, 0, bonds.size());
        std::memset(resource->free_triangle_nodes.contents, 0,
                    resource->capacity * sizeof(std::uint8_t));
        resource->triangle_bonds = std::move(triangle_bonds);
        if (fracture_enabled) {
            resource->source_inverse_masses.assign(
                inverse, inverse + resource->count);
            resource->source_degrees.resize(resource->count);
            resource->bond_corners.resize(bonds.size());
            resource->topology_active.assign(bonds.size(), std::uint8_t{1U});
            resource->topology_scratch_active.resize(bonds.size());
            resource->topology_scratch_old_indices.resize(
                resource->triangle_index_count);
            resource->topology_scratch_indices.resize(
                resource->triangle_index_count);
            resource->topology_scratch_parent.resize(
                resource->triangle_index_count);
            resource->topology_scratch_nodes.resize(
                resource->triangle_index_count);
            resource->topology_scratch_used.resize(resource->capacity);
            resource->topology_scratch_degrees.resize(resource->capacity);
            resource->topology_scratch_copies.reserve(resource->capacity);
            resource->topology_scratch_edges.reserve(
                resource->triangle_index_count + resource->bond_count);
            resource->topology_scratch_graph_degrees.resize(
                resource->capacity);
            resource->topology_scratch_offsets.resize(
                static_cast<std::size_t>(resource->capacity) + 1U);
            resource->topology_scratch_cursors.resize(resource->capacity);
            resource->topology_scratch_neighbors.resize(
                resource->neighbor_capacity);
            resource->topology_scratch_free_nodes.resize(
                resource->capacity);
            std::unordered_map<std::uint64_t, std::uint32_t> ids;
            std::unordered_map<std::uint64_t, std::uint32_t> first_corners;
            for (std::uint32_t index = 0; index < public_bonds.size(); ++index)
                ids[edge_key(public_bonds[index].first,
                             public_bonds[index].second)] = index;
            for (std::uint32_t corner = 0;
                 corner < resource->triangle_index_count; ++corner) {
                const std::uint32_t next =
                    corner / 3U * 3U + (corner + 1U) % 3U;
                const std::uint32_t other =
                    corner / 3U * 3U + (corner + 2U) % 3U;
                const std::uint32_t first =
                    options.triangle_indices.data[corner];
                const std::uint32_t second =
                    options.triangle_indices.data[next];
                ++resource->source_degrees[first];
                const std::uint64_t key = edge_key(first, second);
                const auto [found, fresh] =
                    first_corners.emplace(key, corner);
                const std::uint32_t stretch = ids.at(key);
                if (fresh) {
                    resource->bond_corners[stretch] = {corner, next};
                    continue;
                }
                const std::uint32_t previous = found->second;
                const std::uint32_t previous_next =
                    previous / 3U * 3U + (previous + 1U) % 3U;
                const std::uint32_t opposite_corner =
                    previous / 3U * 3U + (previous + 2U) % 3U;
                const std::uint32_t opposite_source =
                    options.triangle_indices.data[opposite_corner];
                const std::uint32_t other_source =
                    options.triangle_indices.data[other];
                const auto bending =
                    ids.find(edge_key(opposite_source, other_source));
                const bool has_bending =
                    bending != ids.end() &&
                    public_bonds[bending->second].bending;
                if (has_bending)
                    resource->bond_corners[bending->second] = {
                        opposite_corner, other};
                const bool same =
                    options.triangle_indices.data[previous] == first;
                ClothSeam seam{};
                seam.corners[0] = previous;
                seam.corners[1] = previous_next;
                seam.corners[2] = same ? corner : next;
                seam.corners[3] = same ? next : corner;
                seam.bond = stretch;
                seam.bending =
                    has_bending ? bending->second : UINT32_MAX;
                resource->seams.push_back(seam);
            }
            Status topology_status = rebuild_cloth_topology(*resource, true);
            if (!topology_status) return topology_status;
        } else {
            auto *offsets = static_cast<std::uint32_t *>(
                resource->neighbor_offsets.contents);
            auto *neighbors = static_cast<DeformableNeighbor *>(
                resource->neighbors.contents);
            std::uint32_t cursor = 0U;
            offsets[0] = 0U;
            for (std::uint32_t node = 0U; node < resource->count; ++node) {
                for (const DeformableNeighbor neighbor : adjacency[node])
                    neighbors[cursor++] = neighbor;
                offsets[node + 1U] = cursor;
            }
        }
        impl_->bind(resource->table, 0U, resource->particles.positions);
        impl_->bind(resource->table, 1U, resource->particles.previous);
        impl_->bind(resource->table, 2U, resource->particles.velocities);
        impl_->bind(resource->table, 3U, resource->particles.inverse_masses);
        impl_->bind(resource->table, 4U, resource->bonds);
        impl_->bind(resource->table, 5U, resource->active_bonds);
        impl_->bind(resource->table, 6U, resource->surface_positions);
        impl_->bind(resource->table, 7U, resource->constants);
        impl_->bind(resource->table, 8U, resource->triangle_indices);
        impl_->bind(resource->table, 9U, resource->bond_damage);
        impl_->bind(resource->table, 10U,
                    resource->surface_physical_indices);
        impl_->bind(resource->table, 11U, resource->rigid_forces);
        impl_->bind(resource->table, 12U, resource->corrections);
        impl_->bind(resource->table, 13U, resource->neighbor_offsets);
        impl_->bind(resource->table, 14U, resource->neighbors);
        impl_->bind(resource->table, 15U, resource->free_triangle_nodes);
        impl_->bind(resource->surface_rigid_table, 0U,
                    resource->particles.positions);
        impl_->bind(resource->surface_rigid_table, 1U,
                    resource->particles.velocities);
        impl_->bind(resource->surface_rigid_table, 2U,
                    resource->particles.inverse_masses);
        impl_->bind(resource->surface_rigid_table, 3U,
                    resource->triangle_indices);
        impl_->bind(resource->surface_rigid_table, 9U,
                    resource->surface_rigid_constants);
        impl_->bind(resource->surface_rigid_table, 10U,
                    resource->body_corrections);
        impl_->bind(resource->surface_rigid_table, 11U,
                    resource->surface_positions);
        impl_->bind(resource->surface_rigid_table, 12U,
                    resource->surface_physical_indices);
        Status rigid_status = impl_->setup_rigid_contact(
            resource->particles, resource->rigid_constants,
            resource->rigid_table, resource->rigid_forces,
            resource->capacity);
        if (!rigid_status) return rigid_status;
        impl_->bind(resource->table, 16U,
                    resource->particles.rigid_particle_linear_impulses);
        [impl_->residency commit];
        impl_->cloths.entries[slot] = std::move(resource);
        ++impl_->cloths.count;
        output = {slot, impl_->cloths.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate Metal cloth state");
    }
}

Status MetalSystems::remove_cloth(ClothId id) noexcept {
    if (!impl_ || !impl_->cloths.get(id))
        return invalid_handle("Cloth handle is stale");
    for (const auto &entry : impl_->smoke_cloth.entries)
        if (entry && entry->options.cloth == id)
            return invalid_argument("Cloth is referenced by smoke");
    for (const auto &entry : impl_->fluid_cloth.entries)
        if (entry && entry->options.cloth == id)
            return invalid_argument("Cloth is referenced by fluid");
    for (const auto &entry : impl_->soft_cloth.entries)
        if (entry && entry->options.cloth == id)
            return invalid_argument("Cloth is referenced by a soft body");
    for (const auto &entry : impl_->rope_cloth.entries)
        if (entry && entry->options.cloth == id)
            return invalid_argument("Cloth is referenced by a rope");
    for (const auto &entry : impl_->paint_fields.entries)
        if (entry && entry->options.cloth == id)
            return invalid_argument("Cloth is referenced by a paint field");
    impl_->cloths.erase(id);
    return success();
}

Status MetalSystems::cloth_view(ClothId id,
                                ClothDeviceView &output) const noexcept {
    output = {};
    const ClothResource *r = impl_ ? impl_->cloths.get(id) : nullptr;
    if (!r) return invalid_handle("Cloth handle is stale");
    output.positions = span<Vec3>(r->particles.positions, r->count);
    output.velocities = span<Vec3>(r->particles.velocities, r->count);
    output.triangle_indices =
        span<std::uint32_t>(r->triangle_indices, r->triangle_index_count);
    output.vertex_count = r->count;
    output.vertex_source_indices = span<std::uint32_t>(r->source_indices, r->count);
    output.inverse_masses = span<float>(r->particles.inverse_masses, r->count);
    output.surface_positions = span<Vec3>(r->surface_positions, r->surface_count);
    output.surface_triangle_indices =
        span<std::uint32_t>(r->surface_indices, r->triangle_index_count);
    output.surface_source_indices =
        span<std::uint32_t>(r->surface_source_indices, r->surface_count);
    output.bonds = span<ClothBond>(r->public_bonds, r->bond_count);
    output.active_bonds = span<std::uint8_t>(r->active_bonds, r->bond_count);
    output.rigid_contact_forces = span<Vec3>(r->rigid_forces, r->count);
    output.fluid_contact_forces = span<Vec3>(r->fluid_forces, r->count);
    output.soft_body_contact_forces = span<Vec3>(r->soft_forces, r->count);
    output.rope_contact_forces = span<Vec3>(r->rope_forces, r->count);
    return success();
}

Status MetalSystems::add_soft_body(SoftBodyOptions options,
                                   SoftBodyId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    if (!options.nodes.data || options.nodes.size < 4U ||
        options.nodes.size > UINT32_MAX || !options.bonds.data ||
        options.bonds.size == 0U || options.bonds.size > UINT32_MAX ||
        !options.surface_vertices.data || options.surface_vertices.size < 3U ||
        options.surface_vertices.size > UINT32_MAX ||
        !options.surface_triangle_indices.data ||
        options.surface_triangle_indices.size == 0U ||
        options.surface_triangle_indices.size % 3U != 0U ||
        options.surface_triangle_indices.size > UINT32_MAX ||
        !options.surface_bindings.data ||
        options.surface_bindings.size != options.surface_vertices.size ||
        (options.inverse_masses.size != 0U &&
         (!options.inverse_masses.data ||
          options.inverse_masses.size != options.nodes.size)) ||
        !std::isfinite(options.node_mass) || !(options.node_mass > 0.0F) ||
        !std::isfinite(options.node_radius) || !(options.node_radius > 0.0F) ||
        !std::isfinite(options.stretch_compliance) ||
        options.stretch_compliance < 0.0F ||
        !std::isfinite(options.velocity_damping) ||
        options.velocity_damping < 0.0F ||
        !std::isfinite(options.spring_damping) ||
        options.spring_damping < 0.0F || options.spring_damping > 1.0F ||
        !std::isfinite(options.contact_friction) ||
        options.contact_friction < 0.0F ||
        !std::isfinite(options.shape_matching_stiffness) ||
        options.shape_matching_stiffness < 0.0F ||
        options.shape_matching_stiffness > 1.0F ||
        !std::isfinite(options.maximum_projection_fraction) ||
        options.maximum_projection_fraction <= 0.0F ||
        options.maximum_projection_fraction > 1.0F ||
        !std::isfinite(options.constraint_velocity_response) ||
        options.constraint_velocity_response < 0.0F ||
        options.constraint_velocity_response > 1.0F ||
        !std::isfinite(options.maximum_speed) ||
        !(options.maximum_speed > 0.0F) || options.solver_iterations == 0U ||
        options.solver_iterations > 64U) {
        return invalid_argument("Soft-body options are invalid");
    }
    const std::uint32_t slot = impl_->soft_bodies.free_slot();
    if (slot == impl_->soft_bodies.entries.size())
        return capacity_exceeded("Soft-body capacity is exhausted");
    try {
        auto r = std::make_unique<SoftResource>();
        r->options = options;
        r->count = static_cast<std::uint32_t>(options.nodes.size);
        r->bond_count = static_cast<std::uint32_t>(options.bonds.size);
        r->surface_count = static_cast<std::uint32_t>(options.surface_vertices.size);
        r->surface_index_count =
            static_cast<std::uint32_t>(options.surface_triangle_indices.size);
        std::vector<MetalBond> bonds(r->bond_count);
        std::vector<std::uint64_t> bond_keys;
        std::vector<std::uint32_t> bond_degrees(r->count, 0U);
        bond_keys.reserve(r->bond_count);
        for (std::uint32_t index = 0; index < r->bond_count; ++index) {
            const SoftBodyBond &bond = options.bonds.data[index];
            if (bond.first >= r->count || bond.second >= r->count ||
                bond.first == bond.second || !std::isfinite(bond.rest_length) ||
                !(bond.rest_length > 1.0e-6F))
                return invalid_argument("Soft-body bond is invalid");
            const std::uint32_t first = std::min(bond.first, bond.second);
            const std::uint32_t second = std::max(bond.first, bond.second);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(first) << 32U) | second;
            if (std::find(bond_keys.begin(), bond_keys.end(), key) !=
                bond_keys.end())
                return invalid_argument("Soft-body bond is duplicated");
            bond_keys.push_back(key);
            ++bond_degrees[bond.first];
            ++bond_degrees[bond.second];
            bonds[index] = {bond.first, bond.second, bond.rest_length,
                            options.stretch_compliance, 1U};
        }
        if (std::any_of(bond_degrees.begin(), bond_degrees.end(),
                        [](std::uint32_t degree) { return degree == 0U; }))
            return invalid_argument(
                "Every soft-body node needs at least one bond");
        float movable_mass = 0.0F;
        for (std::uint32_t index = 0; index < r->count; ++index) {
            if (!finite(options.nodes.data[index]))
                return invalid_argument("Soft-body node is invalid");
            if (options.inverse_masses.size != 0U &&
                (!std::isfinite(options.inverse_masses.data[index]) ||
                 options.inverse_masses.data[index] < 0.0F))
                return invalid_argument("Soft-body inverse mass is invalid");
            const float inverse_mass = options.inverse_masses.size != 0U
                ? options.inverse_masses.data[index]
                : 1.0F / options.node_mass;
            if (inverse_mass > 0.0F) movable_mass += 1.0F / inverse_mass;
        }
        if (!std::isfinite(movable_mass) || !(movable_mass > 0.0F))
            return invalid_argument(
                "Soft body needs at least one movable node");
        r->movable_mass = movable_mass;
        if (options.shape_matching_stiffness > 0.0F) {
            Vec3 rest_center{};
            for (std::uint32_t index = 0; index < r->count; ++index) {
                const float inverse_mass = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[index]
                    : 1.0F / options.node_mass;
                if (inverse_mass > 0.0F)
                    rest_center = add(
                        rest_center,
                        multiply(options.nodes.data[index],
                                 1.0F / inverse_mass));
            }
            rest_center = multiply(rest_center, 1.0F / movable_mass);
            Vec3 covariance[3]{};
            for (std::uint32_t index = 0; index < r->count; ++index) {
                const float inverse_mass = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[index]
                    : 1.0F / options.node_mass;
                if (inverse_mass <= 0.0F) continue;
                const Vec3 rest = subtract(options.nodes.data[index],
                                           rest_center);
                const float mass = 1.0F / inverse_mass;
                covariance[0] = add(covariance[0],
                                    multiply(rest, mass * rest.x));
                covariance[1] = add(covariance[1],
                                    multiply(rest, mass * rest.y));
                covariance[2] = add(covariance[2],
                                    multiply(rest, mass * rest.z));
            }
            const float determinant = dot(
                covariance[0], cross(covariance[1], covariance[2]));
            if (!std::isfinite(determinant) ||
                std::abs(determinant) <= 1.0e-10F)
                return invalid_argument(
                    "Shape-matched soft body needs a volumetric rest lattice");
        }
        for (std::uint32_t index = 0; index < r->surface_index_count; ++index)
            if (options.surface_triangle_indices.data[index] >= r->surface_count)
                return invalid_argument("Soft-body surface index is invalid");
        for (std::uint32_t vertex = 0; vertex < r->surface_count; ++vertex) {
            if (!finite(options.surface_vertices.data[vertex]))
                return invalid_argument("Soft-body surface vertex is invalid");
            float weight_sum = 0.0F;
            for (std::uint32_t item = 0; item < 4U; ++item) {
                const std::uint32_t node = options.surface_bindings.data[vertex]
                                               .nodes[item];
                const float weight = options.surface_bindings.data[vertex]
                                         .weights[item];
                if (node >= r->count || !std::isfinite(weight) || weight < 0.0F)
                    return invalid_argument(
                        "Soft-body surface binding is invalid");
                weight_sum += weight;
            }
            if (std::abs(weight_sum - 1.0F) > 1.0e-4F)
                return invalid_argument(
                    "Soft-body surface binding weights must sum to one");
        }
        r->surface_has_volume = surface_volume_orientation(
            options.surface_vertices, options.surface_triangle_indices,
            r->surface_orientation);
        float closed_orientation = r->surface_orientation;
        r->surface_closed = closed_surface_orientation(
            options.surface_vertices, options.surface_triangle_indices,
            closed_orientation);
        if (r->surface_closed) r->surface_orientation = closed_orientation;
        const std::size_t vec_bytes = r->count * sizeof(Vec3);
        r->particles.positions = impl_->buffer(vec_bytes);
        r->particles.previous = impl_->buffer(vec_bytes);
        r->particles.velocities = impl_->buffer(vec_bytes);
        r->particles.inverse_masses = impl_->buffer(r->count * sizeof(float));
        r->public_bonds =
            impl_->buffer(r->bond_count * sizeof(SoftBodyBond));
        r->bonds = impl_->buffer(bonds.size() * sizeof(MetalBond));
        r->rest_positions = impl_->buffer(vec_bytes);
        r->corrections = impl_->buffer(vec_bytes);
        r->surface_positions = impl_->buffer(r->surface_count * sizeof(Vec3));
        r->surface_rest_positions = impl_->buffer(r->surface_count * sizeof(Vec3));
        r->surface_indices =
            impl_->buffer(r->surface_index_count * sizeof(std::uint32_t));
        r->surface_bindings =
            impl_->buffer(r->surface_count * sizeof(MetalSurfaceBinding));
        r->shape_orientation = impl_->buffer(sizeof(Quaternion));
        r->contact_accumulators = impl_->buffer(
            r->count * sizeof(SoftContactAccumulator));
        r->contact_state = impl_->buffer(sizeof(SoftContactState));
        r->rigid_forces = impl_->buffer(vec_bytes);
        r->cloth_forces = impl_->buffer(vec_bytes);
        r->fluid_forces = impl_->buffer(vec_bytes);
        r->rope_forces = impl_->buffer(vec_bytes);
        r->smoke_forces = impl_->buffer(vec_bytes);
        r->constants = impl_->buffer(sizeof(DeformableConstants));
        r->table = impl_->table(13U);
        if (r->particles.positions == nil || r->particles.previous == nil ||
            r->particles.velocities == nil || r->bonds == nil ||
            r->rest_positions == nil || r->corrections == nil ||
            r->surface_positions == nil || r->surface_bindings == nil ||
            r->shape_orientation == nil ||
            r->contact_accumulators == nil || r->contact_state == nil ||
            r->smoke_forces == nil ||
            r->constants == nil || r->table == nil)
            return metal_failure(nil, "Could not allocate Metal soft-body buffers");
        copy_to(r->particles.positions, options.nodes.data, r->count);
        copy_to(r->particles.previous, options.nodes.data, r->count);
        copy_to(r->rest_positions, options.nodes.data, r->count);
        copy_to(r->public_bonds, options.bonds.data, r->bond_count);
        copy_to(r->bonds, bonds.data(), bonds.size());
        *static_cast<Quaternion *>(r->shape_orientation.contents) =
            {0.0F, 0.0F, 0.0F, 1.0F};
        auto *inverse = static_cast<float *>(r->particles.inverse_masses.contents);
        for (std::uint32_t index = 0; index < r->count; ++index) {
            inverse[index] = options.inverse_masses.data &&
                                     index < options.inverse_masses.size
                                 ? options.inverse_masses.data[index]
                                 : 1.0F / options.node_mass;
        }
        if (r->surface_count != 0U) {
            if (!options.surface_vertices.data || !options.surface_bindings.data ||
                options.surface_bindings.size != r->surface_count)
                return invalid_argument("Soft-body surface bindings are invalid");
            copy_to(r->surface_positions, options.surface_vertices.data,
                    r->surface_count);
            copy_to(r->surface_rest_positions, options.surface_vertices.data,
                    r->surface_count);
            copy_to(r->surface_indices, options.surface_triangle_indices.data,
                    r->surface_index_count);
            copy_to(r->surface_bindings,
                    reinterpret_cast<const MetalSurfaceBinding *>(
                        options.surface_bindings.data),
                    r->surface_count);
        }
        impl_->bind(r->table, 0U, r->particles.positions);
        impl_->bind(r->table, 1U, r->particles.previous);
        impl_->bind(r->table, 2U, r->particles.velocities);
        impl_->bind(r->table, 3U, r->particles.inverse_masses);
        impl_->bind(r->table, 4U, r->bonds);
        impl_->bind(r->table, 5U, r->surface_positions);
        impl_->bind(r->table, 6U, r->surface_rest_positions);
        impl_->bind(r->table, 7U, r->surface_bindings);
        impl_->bind(r->table, 8U, r->constants);
        impl_->bind(r->table, 9U, r->rest_positions);
        impl_->bind(r->table, 10U, r->corrections);
        impl_->bind(r->table, 11U, r->shape_orientation);
        impl_->bind(r->table, 12U, r->contact_state);
        Status rigid_status = impl_->setup_rigid_contact(
            r->particles, r->rigid_constants, r->rigid_table,
            r->rigid_forces, r->count);
        if (!rigid_status) return rigid_status;
        impl_->bind(r->rigid_table, 27U, r->contact_accumulators);
        impl_->bind(r->rigid_table, 28U, r->contact_state);
        [impl_->residency commit];
        impl_->soft_bodies.entries[slot] = std::move(r);
        ++impl_->soft_bodies.count;
        output = {slot, impl_->soft_bodies.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate Metal soft-body state");
    }
}

Status MetalSystems::remove_soft_body(SoftBodyId id) noexcept {
    if (!impl_ || !impl_->soft_bodies.get(id))
        return invalid_handle("Soft-body handle is stale");
    for (const auto &entry : impl_->smoke_soft.entries)
        if (entry && entry->options.soft_body == id)
            return invalid_argument("Soft body is referenced by smoke");
    for (const auto &entry : impl_->fluid_soft.entries)
        if (entry && entry->options.soft_body == id)
            return invalid_argument("Soft body is referenced by fluid");
    for (const auto &entry : impl_->soft_cloth.entries)
        if (entry && entry->options.soft_body == id)
            return invalid_argument("Soft body is referenced by cloth");
    for (const auto &entry : impl_->rope_soft.entries)
        if (entry && entry->options.soft_body == id)
            return invalid_argument("Soft body is referenced by rope");
    impl_->soft_bodies.erase(id);
    return success();
}

Status MetalSystems::soft_body_view(SoftBodyId id,
                                    SoftBodyDeviceView &output) const noexcept {
    output = {};
    const SoftResource *r = impl_ ? impl_->soft_bodies.get(id) : nullptr;
    if (!r) return invalid_handle("Soft-body handle is stale");
    output.positions = span<Vec3>(r->particles.positions, r->count);
    output.velocities = span<Vec3>(r->particles.velocities, r->count);
    output.bonds = span<SoftBodyBond>(r->public_bonds, r->bond_count);
    output.surface_positions = span<Vec3>(r->surface_positions, r->surface_count);
    output.surface_triangle_indices =
        span<std::uint32_t>(r->surface_indices, r->surface_index_count);
    output.rigid_contact_forces = span<Vec3>(r->rigid_forces, r->count);
    output.cloth_contact_forces = span<Vec3>(r->cloth_forces, r->count);
    output.fluid_contact_forces = span<Vec3>(r->fluid_forces, r->count);
    output.rope_contact_forces = span<Vec3>(r->rope_forces, r->count);
    output.node_count = r->count;
    output.surface_vertex_count = r->surface_count;
    return success();
}

Status MetalSystems::add_rope(RopeOptions options,
                              std::uint32_t first_contact_skip,
                              std::uint32_t last_contact_skip,
                              RopeId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    if (!std::isfinite(options.radius) || !(options.radius > 0.0F) ||
        !std::isfinite(options.node_spacing) ||
        !(options.node_spacing > 0.0F) ||
        options.node_spacing > 2.0F * options.radius ||
        !std::isfinite(options.mass) || !(options.mass > 0.0F) ||
        !std::isfinite(options.stretch_compliance) ||
        options.stretch_compliance < 0.0F ||
        !std::isfinite(options.velocity_damping) ||
        options.velocity_damping < 0.0F ||
        !std::isfinite(options.maximum_substep_timestep) ||
        !(options.maximum_substep_timestep > 0.0F) ||
        !std::isfinite(options.friction) || options.friction < 0.0F ||
        !std::isfinite(options.maximum_speed) ||
        !(options.maximum_speed > 0.0F) ||
        options.solver_iterations == 0U ||
        options.solver_iterations > 128U ||
        !finite(options.first.local_anchor) ||
        !finite(options.last.local_anchor) ||
        (options.first.enabled && options.last.enabled &&
         options.first.body == options.last.body))
        return invalid_argument("Rope options are invalid");
    std::vector<Vec3> nodes;
    Status sampled = sample_rope_centerline(options.centerline,
                                            options.node_spacing, nodes);
    if (!sampled) return sampled;
    const std::uint32_t slot = impl_->ropes.free_slot();
    if (slot == impl_->ropes.entries.size())
        return capacity_exceeded("Rope capacity is exhausted");
    try {
        auto r = std::make_unique<RopeResource>();
        r->options = options;
        r->count = static_cast<std::uint32_t>(nodes.size());
        std::vector<MetalBond> bonds(r->count - 1U);
        std::vector<float> rest_lengths(r->count - 1U);
        for (std::uint32_t index = 0; index + 1U < r->count; ++index) {
            const Vec3 a = nodes[index], b = nodes[index + 1U];
            const float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
            const float length = std::sqrt(dx * dx + dy * dy + dz * dz);
            rest_lengths[index] = length;
            bonds[index] = {index, index + 1U, length,
                            options.stretch_compliance, 1U};
        }
        const std::size_t vec_bytes = r->count * sizeof(Vec3);
        r->particles.positions = impl_->buffer(vec_bytes);
        r->particles.previous = impl_->buffer(vec_bytes);
        r->particles.velocities = impl_->buffer(vec_bytes);
        r->particles.inverse_masses = impl_->buffer(r->count * sizeof(float));
        r->rest_lengths = impl_->buffer(rest_lengths.size() * sizeof(float));
        r->constraint_forces = impl_->buffer(vec_bytes);
        r->contact_forces = impl_->buffer(vec_bytes);
        r->fluid_forces = impl_->buffer(vec_bytes);
        r->soft_forces = impl_->buffer(vec_bytes);
        r->bonds = impl_->buffer(bonds.size() * sizeof(MetalBond));
        r->directions = impl_->buffer(bonds.size() * sizeof(Vec3));
        r->diagonal = impl_->buffer(bonds.size() * sizeof(float));
        r->upper = impl_->buffer(bonds.size() * sizeof(float));
        r->rhs = impl_->buffer(bonds.size() * sizeof(float));
        r->lambdas = impl_->buffer(bonds.size() * sizeof(float));
        r->scratch = impl_->buffer(r->count * sizeof(Vec3));
        r->contact_normals = impl_->buffer(vec_bytes);
        r->contact_normals2 = impl_->buffer(vec_bytes);
        const std::size_t body_bytes =
            impl_->options.rigid_body_capacity * sizeof(Vec3);
        const std::size_t hint_bytes =
            static_cast<std::size_t>(r->count) *
            impl_->options.rigid_body_capacity * sizeof(std::uint32_t);
        r->body_translation = impl_->buffer(body_bytes);
        // Metal exposes 31 buffer slots. Store the CUDA-compatible convex
        // plane hints after the per-body rotation scratch in the same slot.
        r->body_rotation = impl_->buffer(body_bytes + hint_bytes);
        r->anchor_states = impl_->buffer(2U * sizeof(RopeAnchorState));
        r->empty_soft_target = impl_->buffer(32U * sizeof(std::uint32_t));
        r->constants = impl_->buffer(sizeof(DeformableConstants));
        r->attachment_constants =
            impl_->buffer(sizeof(RopeAttachmentConstants));
        r->table = impl_->table(31U);
        if (r->particles.positions == nil || r->particles.previous == nil ||
            r->particles.velocities == nil || r->bonds == nil ||
            r->directions == nil || r->diagonal == nil || r->upper == nil ||
            r->rhs == nil || r->lambdas == nil || r->scratch == nil ||
            r->contact_normals == nil || r->contact_normals2 == nil ||
            r->body_translation == nil || r->body_rotation == nil ||
            r->anchor_states == nil || r->empty_soft_target == nil ||
            r->constants == nil || r->attachment_constants == nil ||
            r->table == nil)
            return metal_failure(nil, "Could not allocate Metal rope buffers");
        copy_to(r->particles.positions, nodes.data(), nodes.size());
        copy_to(r->particles.previous, nodes.data(), nodes.size());
        copy_to(r->bonds, bonds.data(), bonds.size());
        copy_to(r->rest_lengths, rest_lengths.data(), rest_lengths.size());
        const float inverse_mass = r->count * (1.0F / options.mass);
        auto *inverse = static_cast<float *>(r->particles.inverse_masses.contents);
        std::fill(inverse, inverse + r->count, inverse_mass);
        std::memset(static_cast<unsigned char *>(r->body_rotation.contents) +
                        body_bytes,
                    0xff, hint_bytes);
        impl_->bind(r->table, 0U, r->particles.positions);
        impl_->bind(r->table, 1U, r->particles.previous);
        impl_->bind(r->table, 2U, r->particles.velocities);
        impl_->bind(r->table, 3U, r->particles.inverse_masses);
        impl_->bind(r->table, 4U, r->bonds);
        impl_->bind(r->table, 5U, r->constants);
        *static_cast<RopeAttachmentConstants *>(
            r->attachment_constants.contents) = {
            options.first.body, options.first.local_anchor,
            options.first.enabled ? 1U : 0U, options.last.body,
            options.last.local_anchor, options.last.enabled ? 1U : 0U, 0U,
            0U, 0U, first_contact_skip, last_contact_skip,
            impl_->options.rigid_body_capacity};
        auto *anchors = static_cast<RopeAnchorState *>(
            r->anchor_states.contents);
        anchors[0].position = nodes.front();
        anchors[1].position = nodes.back();
        impl_->bind(r->table, 6U, r->attachment_constants);
        impl_->bind(r->table, 9U, r->directions);
        impl_->bind(r->table, 10U, r->diagonal);
        impl_->bind(r->table, 11U, r->upper);
        impl_->bind(r->table, 12U, r->rhs);
        impl_->bind(r->table, 13U, r->lambdas);
        impl_->bind(r->table, 14U, r->scratch);
        impl_->bind(r->table, 15U, r->constraint_forces);
        impl_->bind(r->table, 17U, r->anchor_states);
        impl_->bind(r->table, 18U, r->contact_forces);
        impl_->bind(r->table, 23U, r->contact_normals);
        impl_->bind(r->table, 24U, r->contact_normals2);
        impl_->bind(r->table, 25U, r->body_translation);
        impl_->bind(r->table, 26U, r->body_rotation);
        impl_->bind(r->table, 27U, r->empty_soft_target);
        impl_->bind(r->table, 28U, r->empty_soft_target);
        impl_->bind(r->table, 29U, r->soft_forces);
        [impl_->residency commit];
        impl_->ropes.entries[slot] = std::move(r);
        ++impl_->ropes.count;
        output = {slot, impl_->ropes.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate Metal rope state");
    }
}

Status MetalSystems::remove_rope(RopeId id) noexcept {
    if (!impl_ || !impl_->ropes.get(id))
        return invalid_handle("Rope handle is stale");
    for (const auto &entry : impl_->smoke_rope.entries)
        if (entry && entry->options.rope == id)
            return invalid_argument("Rope is referenced by smoke");
    for (const auto &entry : impl_->fluid_rope.entries)
        if (entry && entry->options.rope == id)
            return invalid_argument("Rope is referenced by fluid");
    for (const auto &entry : impl_->rope_soft.entries)
        if (entry && entry->options.rope == id)
            return invalid_argument("Rope is referenced by a soft body");
    for (const auto &entry : impl_->rope_cloth.entries)
        if (entry && entry->options.rope == id)
            return invalid_argument("Rope is referenced by cloth");
    impl_->ropes.erase(id);
    return success();
}

Status MetalSystems::rope_view(RopeId id,
                               RopeDeviceView &output) const noexcept {
    output = {};
    const RopeResource *r = impl_ ? impl_->ropes.get(id) : nullptr;
    if (!r) return invalid_handle("Rope handle is stale");
    output.positions = span<Vec3>(r->particles.positions, r->count);
    output.velocities = span<Vec3>(r->particles.velocities, r->count);
    output.constraint_forces = span<Vec3>(r->constraint_forces, r->count);
    output.contact_forces = span<Vec3>(r->contact_forces, r->count);
    output.fluid_contact_forces = span<Vec3>(r->fluid_forces, r->count);
    output.soft_body_contact_forces = span<Vec3>(r->soft_forces, r->count);
    output.rest_lengths = span<float>(r->rest_lengths, r->count - 1U);
    output.radius = r->options.radius;
    return success();
}

Status MetalSystems::add_particle_source(
    ParticleSourceMesh mesh, ParticleSourceOptions options,
    ParticleSourceId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    if (!fluid) return invalid_handle("Particle source fluid handle is stale");
    if (!finite(options.initial_velocity) ||
        !std::isfinite(options.initial_temperature))
        return invalid_argument("Particle source options are invalid");
    if (mesh.spacing == 0.0F) mesh.spacing = fluid->options.support_radius;
    if (mesh.spacing < fluid->options.particle_radius * 2.0F)
        return invalid_argument("Particle source spacing is too small");
    std::vector<Vec3> sites;
    Status sampled = sample_fluid_source(mesh, sites);
    if (!sampled) return sampled;
    const std::uint32_t slot = impl_->sources.free_slot();
    if (slot == impl_->sources.entries.size())
        return capacity_exceeded("Particle source capacity is exhausted");
    try {
        auto resource = std::make_unique<SourceResource>();
        resource->options = options;
        resource->fluid = options.fluid;
        resource->clearance = mesh.spacing;
        resource->site_count = static_cast<std::uint32_t>(sites.size());
        resource->sites = impl_->buffer(sites.size() * sizeof(Vec3));
        resource->constants = impl_->buffer(sizeof(SourceConstants));
        resource->table = impl_->table(16U);
        if (resource->sites == nil || resource->constants == nil ||
            resource->table == nil)
            return metal_failure(nil, "Could not allocate particle source");
        copy_to(resource->sites, sites.data(), sites.size());
        *static_cast<SourceConstants *>(resource->constants.contents) = {
            resource->site_count, fluid->options.capacity,
            options.initial_velocity, options.initial_temperature,
            resource->clearance, options.enabled ? 1U : 0U};
        impl_->bind(resource->table, 0U, fluid->particles.positions);
        impl_->bind(resource->table, 1U, fluid->particles.velocities);
        impl_->bind(resource->table, 2U, fluid->stable_ids);
        impl_->bind(resource->table, 3U, fluid->temperatures);
        impl_->bind(resource->table, 4U, fluid->metadata);
        impl_->bind(resource->table, 5U, resource->sites);
        impl_->bind(resource->table, 6U, resource->constants);
        impl_->bind(resource->table, 7U, fluid->particles.previous);
        impl_->bind(resource->table, 8U, fluid->foam);
        impl_->bind(resource->table, 9U, fluid->foam_sources);
        impl_->bind(resource->table, 10U, fluid->accelerations);
        impl_->bind(resource->table, 11U, fluid->contact_samples);
        impl_->bind(resource->table, 12U, fluid->contact_flags);
        impl_->bind(resource->table, 13U,
                    impl_->spawn_capacity_miss_count);
        [impl_->residency commit];
        impl_->sources.entries[slot] = std::move(resource);
        ++impl_->sources.count;
        output = {slot, impl_->sources.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate particle source state");
    }
}

Status MetalSystems::update_particle_source(
    ParticleSourceId id, ParticleSourceOptions options) noexcept {
    SourceResource *resource = impl_ ? impl_->sources.get(id) : nullptr;
    if (!resource) return invalid_handle("Particle source handle is stale");
    if (resource->fluid != options.fluid)
        return invalid_argument("Particle source fluid is immutable");
    if (!finite(options.initial_velocity) ||
        !std::isfinite(options.initial_temperature))
        return invalid_argument("Particle source options are invalid");
    resource->options = options;
    auto *constants =
        static_cast<SourceConstants *>(resource->constants.contents);
    constants->initial_velocity = options.initial_velocity;
    constants->initial_temperature = options.initial_temperature;
    constants->enabled = options.enabled ? 1U : 0U;
    return success();
}

Status MetalSystems::remove_particle_source(ParticleSourceId id) noexcept {
    if (!impl_ || !impl_->sources.get(id))
        return invalid_handle("Particle source handle is stale");
    impl_->sources.erase(id);
    return success();
}

Status MetalSystems::add_particle_destroy_plane(
    ParticleDestroyPlaneOptions options,
    ParticleDestroyPlaneId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    if (!fluid)
        return invalid_handle("Particle destroy-plane fluid handle is stale");
    const Quaternion q = options.plane.orientation;
    const float q_size = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (!finite(options.plane.center) ||
        !(q_size > 0.25F) || !(q_size < 4.0F) ||
        !std::isfinite(options.plane.half_extents.x) ||
        !(options.plane.half_extents.x > 0.0F) ||
        !std::isfinite(options.plane.half_extents.y) ||
        !(options.plane.half_extents.y > 0.0F))
        return invalid_argument("Particle destroy plane is invalid");
    const std::uint32_t slot = impl_->destroy_planes.free_slot();
    if (slot == impl_->destroy_planes.entries.size())
        return capacity_exceeded("Particle destroy-plane capacity is exhausted");
    try {
        auto resource = std::make_unique<DestroyResource>();
        resource->options = options;
        resource->constants = impl_->buffer(sizeof(DestroyConstants));
        resource->table = impl_->table(12U);
        if (resource->constants == nil || resource->table == nil)
            return metal_failure(nil, "Could not allocate particle destroy plane");
        *static_cast<DestroyConstants *>(resource->constants.contents) = {
            options.plane.center, options.plane.orientation,
            options.plane.half_extents,
            static_cast<std::uint32_t>(options.crossing),
            options.enabled ? 1U : 0U};
        impl_->bind(resource->table, 0U, fluid->particles.positions);
        impl_->bind(resource->table, 1U, fluid->particles.previous);
        impl_->bind(resource->table, 2U, fluid->particles.velocities);
        impl_->bind(resource->table, 3U, fluid->accelerations);
        impl_->bind(resource->table, 4U, fluid->stable_ids);
        impl_->bind(resource->table, 5U, fluid->foam);
        impl_->bind(resource->table, 6U, fluid->temperatures);
        impl_->bind(resource->table, 7U, fluid->metadata);
        impl_->bind(resource->table, 8U, fluid->foam_sources);
        impl_->bind(resource->table, 9U, resource->constants);
        impl_->bind(resource->table, 10U, fluid->contact_samples);
        impl_->bind(resource->table, 11U, fluid->contact_flags);
        [impl_->residency commit];
        impl_->destroy_planes.entries[slot] = std::move(resource);
        ++impl_->destroy_planes.count;
        output = {slot, impl_->destroy_planes.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate particle destroy-plane state");
    }
}

Status MetalSystems::update_particle_destroy_plane(
    ParticleDestroyPlaneId id,
    ParticleDestroyPlaneOptions options) noexcept {
    DestroyResource *resource = impl_ ? impl_->destroy_planes.get(id) : nullptr;
    if (!resource)
        return invalid_handle("Particle destroy-plane handle is stale");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    if (!fluid)
        return invalid_handle("Particle destroy-plane fluid handle is stale");
    const Quaternion q = options.plane.orientation;
    const float q_size = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (!finite(options.plane.center) ||
        !(q_size > 0.25F) || !(q_size < 4.0F) ||
        !std::isfinite(options.plane.half_extents.x) ||
        !(options.plane.half_extents.x > 0.0F) ||
        !std::isfinite(options.plane.half_extents.y) ||
        !(options.plane.half_extents.y > 0.0F))
        return invalid_argument("Particle destroy plane is invalid");
    resource->options = options;
    *static_cast<DestroyConstants *>(resource->constants.contents) = {
        options.plane.center, options.plane.orientation,
        options.plane.half_extents,
        static_cast<std::uint32_t>(options.crossing),
        options.enabled ? 1U : 0U};
    impl_->bind(resource->table, 0U, fluid->particles.positions);
    impl_->bind(resource->table, 1U, fluid->particles.previous);
    impl_->bind(resource->table, 2U, fluid->particles.velocities);
    impl_->bind(resource->table, 3U, fluid->accelerations);
    impl_->bind(resource->table, 4U, fluid->stable_ids);
    impl_->bind(resource->table, 5U, fluid->foam);
    impl_->bind(resource->table, 6U, fluid->temperatures);
    impl_->bind(resource->table, 7U, fluid->metadata);
    impl_->bind(resource->table, 8U, fluid->foam_sources);
    impl_->bind(resource->table, 10U, fluid->contact_samples);
    impl_->bind(resource->table, 11U, fluid->contact_flags);
    [impl_->residency commit];
    return success();
}

Status MetalSystems::remove_particle_destroy_plane(
    ParticleDestroyPlaneId id) noexcept {
    if (!impl_ || !impl_->destroy_planes.get(id))
        return invalid_handle("Particle destroy-plane handle is stale");
    impl_->destroy_planes.erase(id);
    return success();
}

Status MetalSystems::add_paint_field(
    PaintFieldHostOptions options, std::uint32_t vertex_count,
    PaintFieldId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    if (vertex_count == 0U || options.vertex_uvs.data == nullptr ||
        options.vertex_uvs.size != vertex_count || options.width == 0U ||
        options.height == 0U || options.width > 4096U ||
        options.height > 4096U ||
        static_cast<std::uint64_t>(options.width) * options.height >
            16'777'216U)
        return invalid_argument("Paint field UV count or dimensions are invalid");
    for (std::uint32_t index = 0U; index < vertex_count; ++index)
        if (!std::isfinite(options.vertex_uvs.data[index].x) ||
            !std::isfinite(options.vertex_uvs.data[index].y))
            return invalid_argument("Paint field contains non-finite UVs");
    const std::uint32_t slot = impl_->paint_fields.free_slot();
    if (slot == impl_->paint_fields.entries.size())
        return capacity_exceeded("Paint field capacity is exhausted");
    try {
        auto resource = std::make_unique<PaintFieldResource>();
        resource->options = options;
        resource->options.vertex_uvs = {};
        resource->vertex_count = vertex_count;
        resource->uvs = impl_->buffer(
            static_cast<std::size_t>(vertex_count) * sizeof(Vec2));
        resource->pixels = impl_->buffer(
            static_cast<std::size_t>(options.width) * options.height *
            sizeof(std::uint32_t));
        if (resource->uvs == nil || resource->pixels == nil)
            return metal_failure(nil, "Could not allocate Metal paint field");
        copy_to(resource->uvs, options.vertex_uvs.data, vertex_count);
        [impl_->residency commit];
        impl_->paint_fields.entries[slot] = std::move(resource);
        ++impl_->paint_fields.count;
        output = {slot, impl_->paint_fields.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate Metal paint field state");
    }
}

Status MetalSystems::remove_paint_field(PaintFieldId id) noexcept {
    PaintFieldResource *field = impl_ ? impl_->paint_fields.get(id) : nullptr;
    if (!field) return invalid_handle("Paint field handle is stale");
    for (const auto &entry : impl_->paint_rules.entries)
        if (entry && entry->options.target == id)
            return invalid_argument("Paint field is referenced by a paint rule");
    [impl_->residency removeAllocation:field->uvs];
    [impl_->residency removeAllocation:field->pixels];
    [impl_->residency commit];
    impl_->paint_fields.erase(id);
    return success();
}

Status MetalSystems::clear_paint_field(PaintFieldId id) noexcept {
    PaintFieldResource *field = impl_ ? impl_->paint_fields.get(id) : nullptr;
    if (!field) return invalid_handle("Paint field handle is stale");
    std::memset(field->pixels.contents, 0, field->pixels.length);
    return success();
}

Status MetalSystems::paint_field_view(
    PaintFieldId id, PaintFieldDeviceView &output) const noexcept {
    output = {};
    const PaintFieldResource *field =
        impl_ ? impl_->paint_fields.get(id) : nullptr;
    if (!field) return invalid_handle("Paint field handle is stale");
    output.pixels = span<std::uint32_t>(
        field->pixels,
        static_cast<std::uint64_t>(field->options.width) *
            field->options.height);
    output.width = field->options.width;
    output.height = field->options.height;
    output.revision = field->revision;
    return success();
}

Status MetalSystems::add_paint_rule(
    PaintRuleOptions options, PaintRuleId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    const bool fluid_source = options.source.generation != 0U;
    const bool rigid_source = options.rigid_source.generation != 0U;
    if (fluid_source == rigid_source)
        return invalid_argument("Paint rule needs exactly one source");
    if (fluid_source && !impl_->fluids.get(options.source))
        return invalid_handle("Paint rule fluid source is stale");
    PaintFieldResource *field = impl_->paint_fields.get(options.target);
    if (!field) return invalid_handle("Paint target field is stale");
    const bool cloth_target = field->options.cloth.generation != 0U;
    if ((fluid_source && cloth_target) || (rigid_source && !cloth_target))
        return {StatusCode::not_supported, 0,
                "Paint source and target systems do not match"};
    if (!std::isfinite(options.reach) || options.reach < 0.0F ||
        options.reach > 10.0F)
        return invalid_argument("Paint reach is invalid");
    if (!std::isfinite(options.brush_radius) ||
        !(options.brush_radius > 0.0F) || options.brush_radius > 10.0F)
        return invalid_argument("Paint brush radius is invalid");
    const std::uint32_t slot = impl_->paint_rules.free_slot();
    if (slot == impl_->paint_rules.entries.size())
        return capacity_exceeded("Paint rule capacity is exhausted");
    try {
        auto resource = std::make_unique<PaintRuleResource>();
        resource->options = options;
        resource->constants = impl_->buffer(sizeof(PaintConstants));
        resource->table = impl_->table(14U);
        if (resource->constants == nil || resource->table == nil)
            return metal_failure(nil, "Could not allocate Metal paint rule");
        impl_->bind(resource->table, 2U, field->uvs);
        impl_->bind(resource->table, 3U, field->pixels);
        impl_->bind(resource->table, 13U, resource->constants);
        [impl_->residency commit];
        impl_->paint_rules.entries[slot] = std::move(resource);
        ++impl_->paint_rules.count;
        output = {slot, impl_->paint_rules.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate Metal paint rule state");
    }
}

Status MetalSystems::remove_paint_rule(PaintRuleId id) noexcept {
    PaintRuleResource *rule = impl_ ? impl_->paint_rules.get(id) : nullptr;
    if (!rule) return invalid_handle("Paint rule handle is stale");
    [impl_->residency removeAllocation:rule->constants];
    [impl_->residency commit];
    impl_->paint_rules.erase(id);
    return success();
}

bool MetalSystems::paint_field_exists(PaintFieldId id) const noexcept {
    return impl_ && impl_->paint_fields.get(id) != nullptr;
}

std::uint32_t MetalSystems::cloth_source_vertex_count(ClothId id) const noexcept {
    const ClothResource *cloth = impl_ ? impl_->cloths.get(id) : nullptr;
    return cloth == nullptr
               ? 0U
               : cloth->source_inverse_masses.empty()
                     ? cloth->count
                     : static_cast<std::uint32_t>(
                           cloth->source_inverse_masses.size());
}

Status MetalSystems::add_fluid_smoke_coupling(
    FluidSmokeCouplingOptions options,
    FluidSmokeCouplingId &output) noexcept {
    if (!impl_) return invalid_argument("World is not initialized");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    SmokeResource *smoke = impl_->smokes.get(options.smoke);
    if (!fluid || !smoke)
        return invalid_handle("Fluid-smoke coupling endpoint is stale");
    const Quaternion orientation = options.heater.orientation;
    const float rotation_norm_squared =
        orientation.x * orientation.x + orientation.y * orientation.y +
        orientation.z * orientation.z + orientation.w * orientation.w;
    if (!finite(options.heater.center) ||
        !std::isfinite(rotation_norm_squared) ||
        rotation_norm_squared < 0.25F || rotation_norm_squared > 4.0F ||
        !std::isfinite(options.heater.half_extents.x) ||
        !std::isfinite(options.heater.half_extents.y) ||
        options.heater.half_extents.x <= 0.0F ||
        options.heater.half_extents.y <= 0.0F ||
        !std::isfinite(options.heater_temperature) ||
        !std::isfinite(options.boiling_temperature) ||
        !std::isfinite(options.heat_transfer_rate) ||
        options.heat_transfer_rate < 0.0F ||
        !std::isfinite(options.wind_drag) || options.wind_drag < 0.0F ||
        !std::isfinite(options.steam_rise_speed) ||
        options.steam_rise_speed < 0.0F)
        return invalid_argument("Fluid-smoke coupling options are invalid");
    const std::uint32_t slot = impl_->fluid_smoke.free_slot();
    if (slot == impl_->fluid_smoke.entries.size())
        return capacity_exceeded("Fluid-smoke coupling capacity is exhausted");
    try {
        auto resource =
            std::make_unique<CouplingResource<FluidSmokeCouplingOptions>>();
        resource->options = options;
        resource->constants = impl_->buffer(sizeof(FluidSmokeConstants));
        resource->table = impl_->table(23U);
        if (resource->constants == nil || resource->table == nil)
            return metal_failure(nil,
                                 "Could not allocate fluid-smoke coupling");
        impl_->bind(resource->table, 0U, fluid->particles.positions);
        impl_->bind(resource->table, 1U, fluid->particles.previous);
        impl_->bind(resource->table, 2U, fluid->particles.velocities);
        impl_->bind(resource->table, 3U, fluid->accelerations);
        impl_->bind(resource->table, 4U, fluid->stable_ids);
        impl_->bind(resource->table, 5U, fluid->foam);
        impl_->bind(resource->table, 6U, fluid->foam_sources);
        impl_->bind(resource->table, 7U, fluid->temperatures);
        impl_->bind(resource->table, 8U, fluid->metadata);
        impl_->bind(resource->table, 9U, smoke->particles.positions);
        impl_->bind(resource->table, 10U, smoke->particles.previous);
        impl_->bind(resource->table, 11U, smoke->particles.velocities);
        impl_->bind(resource->table, 12U, smoke->ages);
        impl_->bind(resource->table, 13U, smoke->densities);
        impl_->bind(resource->table, 14U, smoke->pressures);
        impl_->bind(resource->table, 15U, smoke->vorticities);
        impl_->bind(resource->table, 16U,
                    smoke->particles.inverse_masses);
        impl_->bind(resource->table, 17U, smoke->metadata);
        impl_->bind(resource->table, 18U, resource->constants);
        impl_->bind(resource->table, 19U, fluid->contact_samples);
        impl_->bind(resource->table, 20U, fluid->contact_flags);
        impl_->bind(resource->table, 21U, smoke->grid_density);
        impl_->bind(resource->table, 22U, smoke->grid_face_velocity);
        [impl_->residency commit];
        impl_->fluid_smoke.entries[slot] = std::move(resource);
        ++impl_->fluid_smoke.count;
        output = {slot, impl_->fluid_smoke.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate fluid-smoke coupling");
    }
}

Status MetalSystems::remove_fluid_smoke_coupling(
    FluidSmokeCouplingId id) noexcept {
    if (!impl_ || !impl_->fluid_smoke.get(id))
        return invalid_handle("Fluid-smoke coupling handle is stale");
    impl_->fluid_smoke.erase(id);
    return success();
}

Status MetalSystems::add_smoke_soft_body_coupling(
    SmokeSoftBodyCouplingOptions options,
    SmokeSoftBodyCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    SmokeResource *smoke = impl_->smokes.get(options.smoke);
    SoftResource *soft = impl_->soft_bodies.get(options.soft_body);
    if (!smoke || !soft)
        return invalid_handle("Smoke-soft-body coupling endpoint is stale");
    if (!std::isfinite(options.wind_drag) || options.wind_drag < 0.0F ||
        !std::isfinite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F)
        return invalid_argument("Smoke-soft-body coupling options are invalid");
    for (const auto &entry : impl_->smoke_soft.entries)
        if (entry && entry->options.smoke == options.smoke &&
            entry->options.soft_body == options.soft_body)
            return invalid_argument("Smoke and soft body are already coupled");
    const std::uint32_t slot = impl_->smoke_soft.free_slot();
    if (slot == impl_->smoke_soft.entries.size())
        return capacity_exceeded(
            "Smoke-soft-body coupling capacity is exhausted");
    try {
        auto resource = std::make_unique<
            CouplingResource<SmokeSoftBodyCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        resource->constants = impl_->buffer(sizeof(SmokeSurfaceConstants));
        resource->phase_constants =
            impl_->buffer(sizeof(SmokeSurfaceConstants));
        resource->table = impl_->table(22U);
        resource->phase_table = impl_->table(22U);
        if (resource->constants == nil ||
            resource->phase_constants == nil || resource->table == nil ||
            resource->phase_table == nil)
            return metal_failure(
                nil, "Could not allocate smoke-soft-body coupling");
        const auto bind_surface = [&](id<MTL4ArgumentTable> table,
                                      id<MTLBuffer> constants) {
            impl_->bind(table, 0U, smoke->particles.positions);
            impl_->bind(table, 1U, smoke->particles.previous);
            impl_->bind(table, 2U, smoke->particles.velocities);
            impl_->bind(table, 3U, smoke->ages);
            impl_->bind(table, 4U, smoke->metadata);
            impl_->bind(table, 5U, soft->particles.positions);
            impl_->bind(table, 6U, soft->particles.velocities);
            impl_->bind(table, 7U, soft->particles.inverse_masses);
            impl_->bind(table, 8U, soft->surface_positions);
            impl_->bind(table, 9U, soft->surface_indices);
            impl_->bind(table, 10U, soft->surface_indices);
            impl_->bind(table, 11U, soft->surface_bindings);
            impl_->bind(table, 12U, constants);
            impl_->bind(table, 13U, smoke->grid_scratch);
            impl_->bind(table, 14U, smoke->grid_deformable_solid);
            impl_->bind(table, 15U, smoke->grid_face_boundary);
            impl_->bind(table, 16U, smoke->grid_pressure);
            impl_->bind(table, 17U, smoke->grid_density);
            impl_->bind(table, 18U, smoke->grid_face_velocity);
            impl_->bind(table, 19U, soft->smoke_forces);
            impl_->bind(table, 20U, smoke->grid_face_nearest_triangle);
            impl_->bind(table, 21U, smoke->grid_face_normal);
        };
        bind_surface(resource->table, resource->constants);
        bind_surface(resource->phase_table, resource->phase_constants);
        [impl_->residency commit];
        smoke->static_metadata_valid = false;
        impl_->smoke_soft.entries[slot] = std::move(resource);
        ++impl_->smoke_soft.count;
        output = {slot, impl_->smoke_soft.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate smoke-soft-body coupling");
    }
}

Status MetalSystems::remove_smoke_soft_body_coupling(
    SmokeSoftBodyCouplingId id) noexcept {
    if (!impl_) return invalid_handle("Smoke-soft-body coupling handle is stale");
    auto *coupling = impl_->smoke_soft.get(id);
    if (!coupling)
        return invalid_handle("Smoke-soft-body coupling handle is stale");
    SmokeResource *smoke = impl_->smokes.get(coupling->options.smoke);
    impl_->smoke_soft.erase(id);
    if (smoke) smoke->static_metadata_valid = false;
    return success();
}

Status MetalSystems::add_smoke_cloth_coupling(
    SmokeClothCouplingOptions options,
    SmokeClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    SmokeResource *smoke = impl_->smokes.get(options.smoke);
    ClothResource *cloth = impl_->cloths.get(options.cloth);
    if (!smoke || !cloth)
        return invalid_handle("Smoke-cloth coupling endpoint is stale");
    if (!std::isfinite(options.wind_drag) || options.wind_drag < 0.0F ||
        !std::isfinite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F)
        return invalid_argument("Smoke-cloth coupling options are invalid");
    for (const auto &entry : impl_->smoke_cloth.entries)
        if (entry && entry->options.smoke == options.smoke &&
            entry->options.cloth == options.cloth)
            return invalid_argument("Smoke and cloth are already coupled");
    const std::uint32_t slot = impl_->smoke_cloth.free_slot();
    if (slot == impl_->smoke_cloth.entries.size())
        return capacity_exceeded("Smoke-cloth coupling capacity is exhausted");
    try {
        auto resource =
            std::make_unique<CouplingResource<SmokeClothCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        resource->constants = impl_->buffer(sizeof(SmokeSurfaceConstants));
        resource->phase_constants =
            impl_->buffer(sizeof(SmokeSurfaceConstants));
        resource->table = impl_->table(22U);
        resource->phase_table = impl_->table(22U);
        if (resource->constants == nil ||
            resource->phase_constants == nil || resource->table == nil ||
            resource->phase_table == nil)
            return metal_failure(nil, "Could not allocate smoke-cloth coupling");
        const auto bind_surface = [&](id<MTL4ArgumentTable> table,
                                      id<MTLBuffer> constants) {
            impl_->bind(table, 0U, smoke->particles.positions);
            impl_->bind(table, 1U, smoke->particles.previous);
            impl_->bind(table, 2U, smoke->particles.velocities);
            impl_->bind(table, 3U, smoke->ages);
            impl_->bind(table, 4U, smoke->metadata);
            impl_->bind(table, 5U, cloth->particles.positions);
            impl_->bind(table, 6U, cloth->particles.velocities);
            impl_->bind(table, 7U, cloth->particles.inverse_masses);
            impl_->bind(table, 8U, cloth->surface_positions);
            impl_->bind(table, 9U, cloth->surface_indices);
            impl_->bind(table, 10U, cloth->surface_physical_indices);
            impl_->bind(table, 11U, cloth->surface_source_indices);
            impl_->bind(table, 12U, constants);
            impl_->bind(table, 13U, smoke->grid_scratch);
            impl_->bind(table, 14U, smoke->grid_deformable_solid);
            impl_->bind(table, 15U, smoke->grid_face_boundary);
            impl_->bind(table, 16U, smoke->grid_pressure);
            impl_->bind(table, 17U, smoke->grid_density);
            impl_->bind(table, 18U, smoke->grid_face_velocity);
            impl_->bind(table, 19U, cloth->smoke_forces);
            impl_->bind(table, 20U, smoke->grid_face_nearest_triangle);
            impl_->bind(table, 21U, smoke->grid_face_normal);
        };
        bind_surface(resource->table, resource->constants);
        bind_surface(resource->phase_table, resource->phase_constants);
        [impl_->residency commit];
        smoke->static_metadata_valid = false;
        impl_->smoke_cloth.entries[slot] = std::move(resource);
        ++impl_->smoke_cloth.count;
        output = {slot, impl_->smoke_cloth.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate smoke-cloth coupling");
    }
}

Status MetalSystems::remove_smoke_cloth_coupling(
    SmokeClothCouplingId id) noexcept {
    if (!impl_) return invalid_handle("Smoke-cloth coupling handle is stale");
    auto *coupling = impl_->smoke_cloth.get(id);
    if (!coupling)
        return invalid_handle("Smoke-cloth coupling handle is stale");
    SmokeResource *smoke = impl_->smokes.get(coupling->options.smoke);
    impl_->smoke_cloth.erase(id);
    if (smoke) smoke->static_metadata_valid = false;
    return success();
}

Status MetalSystems::add_smoke_rope_coupling(
    SmokeRopeCouplingOptions options,
    SmokeRopeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    SmokeResource *smoke = impl_->smokes.get(options.smoke);
    RopeResource *rope = impl_->ropes.get(options.rope);
    if (!smoke || !rope)
        return invalid_handle("Smoke-rope coupling endpoint is stale");
    if (!std::isfinite(options.wind_drag) || options.wind_drag < 0.0F ||
        !std::isfinite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F)
        return invalid_argument("Smoke-rope coupling options are invalid");
    for (const auto &entry : impl_->smoke_rope.entries)
        if (entry && entry->options.smoke == options.smoke &&
            entry->options.rope == options.rope)
            return invalid_argument("Smoke and rope are already coupled");
    const std::uint32_t slot = impl_->smoke_rope.free_slot();
    if (slot == impl_->smoke_rope.entries.size())
        return capacity_exceeded("Smoke-rope coupling capacity is exhausted");
    const float distance = options.contact_distance > 0.0F
                               ? options.contact_distance
                               : smoke->options.particle_radius +
                                     rope->options.radius;
    try {
        auto resource =
            std::make_unique<CouplingResource<SmokeRopeCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        resource->constants = impl_->buffer(sizeof(SmokeRopeConstants));
        resource->phase_constants =
            impl_->buffer(sizeof(SmokeRopeConstants));
        resource->table = impl_->table(11U);
        resource->phase_table = impl_->table(11U);
        if (resource->constants == nil ||
            resource->phase_constants == nil || resource->table == nil ||
            resource->phase_table == nil)
            return metal_failure(nil, "Could not allocate smoke-rope coupling");
        *static_cast<SmokeRopeConstants *>(resource->constants.contents) = {
            0.0F, smoke->options.capacity, rope->count,
            smoke->options.lifetime, distance,
            3.0F * smoke->options.particle_radius,
            smoke->options.rest_number_density, options.wind_drag,
            options.maximum_wind_acceleration, rope->options.maximum_speed,
            0U, 0U, options.enabled ? 1U : 0U,
            smoke->options.grid_resolution,
            smoke->options.grid_vertical_resolution,
            smoke->options.grid_resolution == 0U
                ? 0.0F
                : smoke->options.grid_edge_length /
                      smoke->options.grid_resolution,
            smoke->options.grid_minimum};
        const auto bind_rope = [&](id<MTL4ArgumentTable> table,
                                   id<MTLBuffer> constants) {
            impl_->bind(table, 0U, smoke->particles.positions);
            impl_->bind(table, 1U, smoke->particles.previous);
            impl_->bind(table, 2U, smoke->particles.velocities);
            impl_->bind(table, 3U, smoke->ages);
            impl_->bind(table, 4U, rope->particles.positions);
            impl_->bind(table, 5U, rope->particles.velocities);
            impl_->bind(table, 6U, rope->particles.inverse_masses);
            impl_->bind(table, 7U, constants);
            impl_->bind(table, 8U, smoke->metadata);
            impl_->bind(table, 9U, smoke->grid_density);
            impl_->bind(table, 10U, smoke->grid_face_velocity);
        };
        bind_rope(resource->table, resource->constants);
        bind_rope(resource->phase_table, resource->phase_constants);
        [impl_->residency commit];
        impl_->smoke_rope.entries[slot] = std::move(resource);
        ++impl_->smoke_rope.count;
        output = {slot, impl_->smoke_rope.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate smoke-rope coupling");
    }
}

Status MetalSystems::remove_smoke_rope_coupling(
    SmokeRopeCouplingId id) noexcept {
    if (!impl_ || !impl_->smoke_rope.get(id))
        return invalid_handle("Smoke-rope coupling handle is stale");
    impl_->smoke_rope.erase(id);
    return success();
}

Status MetalSystems::add_smoke_rigid_coupling(
    SmokeRigidCouplingOptions options,
    SmokeRigidCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    if (!impl_->smokes.get(options.smoke) || options.body.generation == 0U)
        return invalid_handle("Smoke-rigid coupling endpoint is stale");
    if (!std::isfinite(options.air_density) || options.air_density < 0.0F ||
        !std::isfinite(options.drag_coefficient) ||
        options.drag_coefficient < 0.0F ||
        !std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F)
        return invalid_argument("Smoke-rigid coupling options are invalid");
    for (const auto &entry : impl_->smoke_rigid.entries)
        if (entry && entry->options.smoke == options.smoke &&
            entry->options.body == options.body)
            return invalid_argument("Smoke and rigid body are already coupled");
    const std::uint32_t slot = impl_->smoke_rigid.free_slot();
    if (slot == impl_->smoke_rigid.entries.size())
        return capacity_exceeded("Smoke-rigid coupling capacity is exhausted");
    try {
        auto resource =
            std::make_unique<CouplingResource<SmokeRigidCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        SmokeResource *smoke = impl_->smokes.get(options.smoke);
        resource->constants = impl_->buffer(sizeof(SmokeRigidConstants));
        resource->contribution_changes = impl_->buffer(
            static_cast<std::size_t>(smoke->options.capacity) *
            sizeof(Vec3));
        resource->contribution_forces = impl_->buffer(
            static_cast<std::size_t>(smoke->options.capacity) *
            sizeof(Vec3));
        resource->table = impl_->table(19U);
        if (resource->constants == nil ||
            resource->contribution_changes == nil ||
            resource->contribution_forces == nil || resource->table == nil)
            return metal_failure(nil, "Could not allocate smoke-rigid coupling");
        *static_cast<SmokeRigidConstants *>(resource->constants.contents) = {
            options.body, smoke->options.particle_radius,
            smoke->options.lifetime, options.air_density,
            options.drag_coefficient,
            options.contact_distance > 0.0F
                ? options.contact_distance
                : smoke->options.particle_radius,
            options.tracer_contact ? 1U : 0U, options.enabled ? 1U : 0U, 0U,
            smoke->options.grid_resolution,
            smoke->options.grid_vertical_resolution,
            smoke->options.grid_resolution == 0U
                ? 0.0F
                : smoke->options.grid_edge_length /
                      smoke->options.grid_resolution,
            smoke->options.grid_minimum,
            smoke->options.rest_number_density,
            smoke->options.grid_kinematic_viscosity,
            smoke->options.grid_les_coefficient, 0.0F,
            smoke->options.maximum_speed,
            smoke->options.pressure_stiffness};
        impl_->bind(resource->table, 0U, smoke->particles.positions);
        impl_->bind(resource->table, 1U, smoke->particles.velocities);
        impl_->bind(resource->table, 2U, smoke->ages);
        impl_->bind(resource->table, 3U, smoke->metadata);
        impl_->bind(resource->table, 10U, resource->constants);
        impl_->bind(resource->table, 11U, smoke->grid_pressure);
        impl_->bind(resource->table, 12U, smoke->grid_density);
        impl_->bind(resource->table, 13U, smoke->grid_face_velocity);
        impl_->bind(resource->table, 14U, smoke->particles.previous);
        impl_->bind(resource->table, 15U, resource->constants);
        impl_->bind(resource->table, 16U, smoke->pressures);
        impl_->bind(resource->table, 17U,
                    resource->contribution_changes);
        impl_->bind(resource->table, 18U,
                    resource->contribution_forces);
        [impl_->residency commit];
        smoke->static_metadata_valid = false;
        impl_->smoke_rigid.entries[slot] = std::move(resource);
        ++impl_->smoke_rigid.count;
        output = {slot, impl_->smoke_rigid.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate smoke-rigid coupling");
    }
}

Status MetalSystems::remove_smoke_rigid_coupling(
    SmokeRigidCouplingId id) noexcept {
    if (!impl_) return invalid_handle("Smoke-rigid coupling handle is stale");
    auto *coupling = impl_->smoke_rigid.get(id);
    if (!coupling)
        return invalid_handle("Smoke-rigid coupling handle is stale");
    SmokeResource *smoke = impl_->smokes.get(coupling->options.smoke);
    impl_->smoke_rigid.erase(id);
    if (smoke) smoke->static_metadata_valid = false;
    return success();
}

Status MetalSystems::add_fluid_rope_coupling(
    FluidRopeCouplingOptions options,
    FluidRopeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    RopeResource *rope = impl_->ropes.get(options.rope);
    if (!fluid || !rope)
        return invalid_handle("Fluid-rope coupling endpoint is stale");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F ||
        !std::isfinite(options.friction) || options.friction < 0.0F ||
        options.friction > 1.0F ||
        !std::isfinite(options.maximum_rope_acceleration) ||
        !(options.maximum_rope_acceleration > 0.0F))
        return invalid_argument("Fluid-rope coupling options are invalid");
    for (const auto &entry : impl_->fluid_rope.entries)
        if (entry && entry->options.fluid == options.fluid &&
            entry->options.rope == options.rope)
            return invalid_argument("Fluid and rope are already coupled");
    const float distance = options.contact_distance > 0.0F
                               ? options.contact_distance
                               : fluid->options.particle_radius +
                                     rope->options.radius;
    const float diameter = 2.0F * fluid->options.particle_radius;
    const float particle_mass = fluid->options.rest_density *
        (fluid->options.rest_particle_volume > 0.0F
             ? fluid->options.rest_particle_volume
             : diameter * diameter * diameter);
    Status status = impl_->add_coupling(
        impl_->fluid_rope, options, fluid->particles,
        fluid->options.capacity,
        rope->particles, rope->count, fluid->accelerations,
        rope->fluid_forces, 4U, distance, particle_mass, 1.0F,
        options.friction, options.maximum_rope_acceleration,
        rope->options.first.enabled ? 0U : UINT32_MAX,
        rope->options.last.enabled ? rope->count - 1U : UINT32_MAX,
        options.enabled, output);
    if (!status) return status;
    auto *resource = impl_->fluid_rope.get(output);
    impl_->bind(resource->table, 9U, fluid->metadata);
    [impl_->residency commit];
    return success();
}

Status MetalSystems::update_fluid_rope_coupling(
    FluidRopeCouplingId id, FluidRopeCouplingOptions options) noexcept {
    auto *resource = impl_ ? impl_->fluid_rope.get(id) : nullptr;
    if (!resource) return invalid_handle("Fluid-rope coupling handle is stale");
    if (resource->options.fluid != options.fluid ||
        resource->options.rope != options.rope)
        return invalid_argument("Fluid-rope endpoints are immutable");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F ||
        !std::isfinite(options.friction) || options.friction < 0.0F ||
        options.friction > 1.0F ||
        !std::isfinite(options.maximum_rope_acceleration) ||
        !(options.maximum_rope_acceleration > 0.0F))
        return invalid_argument("Fluid-rope coupling options are invalid");
    resource->options = options;
    resource->enabled = options.enabled;
    auto *c = static_cast<CouplingConstants *>(resource->constants.contents);
    c->contact_distance = options.contact_distance > 0.0F
                              ? options.contact_distance
                              : impl_->fluids.get(options.fluid)->options.particle_radius +
                                    impl_->ropes.get(options.rope)->options.radius;
    c->friction = options.friction;
    c->maximum_force = options.maximum_rope_acceleration;
    c->enabled = options.enabled ? 1U : 0U;
    return success();
}

Status MetalSystems::remove_fluid_rope_coupling(
    FluidRopeCouplingId id) noexcept {
    if (!impl_ || !impl_->fluid_rope.get(id))
        return invalid_handle("Fluid-rope coupling handle is stale");
    impl_->fluid_rope.erase(id);
    return success();
}

Status MetalSystems::add_rope_soft_body_coupling(
    RopeSoftBodyCouplingOptions options,
    RopeSoftBodyCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    RopeResource *rope = impl_->ropes.get(options.rope);
    SoftResource *soft = impl_->soft_bodies.get(options.soft_body);
    if (!rope || !soft)
        return invalid_handle("Rope-soft-body coupling endpoint is stale");
    if (!soft->surface_has_volume)
        return invalid_argument("Rope-soft-body skin has zero volume");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F || !std::isfinite(options.friction) ||
        options.friction < 0.0F ||
        !std::isfinite(options.maximum_soft_body_acceleration) ||
        !(options.maximum_soft_body_acceleration > 0.0F) ||
        !std::isfinite(options.anchor_support_radius_scale) ||
        options.anchor_support_radius_scale < 0.0F ||
        !std::isfinite(options.anchor_contact_support_radius_scale) ||
        options.anchor_contact_support_radius_scale < 0.0F ||
        options.anchor_contact_support_radius_scale >
            options.anchor_support_radius_scale)
        return invalid_argument("Rope-soft-body coupling options are invalid");
    if ((options.attach_first && rope->options.first.enabled) ||
        (options.attach_last && rope->options.last.enabled))
        return invalid_argument("Rope endpoint already has a rigid attachment");
    std::uint32_t existing_targets = 0U;
    for (const auto &entry : impl_->rope_soft.entries) {
        if (!entry || entry->options.rope != options.rope) continue;
        ++existing_targets;
        if (entry->options.soft_body == options.soft_body)
            return invalid_argument("Rope and soft body are already coupled");
        if ((options.attach_first && entry->options.attach_first) ||
            (options.attach_last && entry->options.attach_last))
            return invalid_argument("Rope endpoint already has an attachment");
    }
    for (const auto &entry : impl_->rope_cloth.entries)
        if (entry && entry->options.rope == options.rope &&
            ((options.attach_first &&
              entry->options.first_vertex != UINT32_MAX) ||
             (options.attach_last &&
              entry->options.last_vertex != UINT32_MAX)))
            return invalid_argument("Rope endpoint already has an attachment");
    if (existing_targets >= 2U)
        return capacity_exceeded(
            "A rope supports at most two soft-body contact targets");
    const std::uint32_t slot = impl_->rope_soft.free_slot();
    if (slot == impl_->rope_soft.entries.size())
        return capacity_exceeded(
            "Rope-soft-body coupling capacity is exhausted");
    try {
        auto resource = std::make_unique<
            CouplingResource<RopeSoftBodyCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        resource->constants = impl_->buffer(sizeof(RopeSoftConstants));
        resource->contact_count = impl_->buffer(sizeof(std::uint32_t));
        resource->maximum_penetration = impl_->buffer(sizeof(float));
        constexpr std::size_t packed_header_words = 32U;
        const std::size_t packed_words = packed_header_words +
            14U * soft->surface_count + 7U * soft->count +
            soft->surface_index_count;
        resource->packed_state =
            impl_->buffer(packed_words * sizeof(std::uint32_t));
        resource->previous_surface =
            impl_->buffer(soft->surface_count * sizeof(Vec3));
        resource->table = impl_->table(20U);
        if (resource->constants == nil || resource->contact_count == nil ||
            resource->maximum_penetration == nil ||
            resource->packed_state == nil ||
            resource->previous_surface == nil || resource->table == nil)
            return metal_failure(nil,
                                 "Could not allocate rope-soft-body coupling");
        copy_to(resource->previous_surface,
                static_cast<const Vec3 *>(soft->surface_positions.contents),
                soft->surface_count);
        auto *constants = static_cast<RopeSoftConstants *>(
            resource->constants.contents);
        *constants = {};
        constants->first_triangle = UINT32_MAX;
        constants->last_triangle = UINT32_MAX;
        constants->attach_first = options.attach_first ? 1U : 0U;
        constants->attach_last = options.attach_last ? 1U : 0U;
        const auto *rope_positions = static_cast<const Vec3 *>(
            rope->particles.positions.contents);
        const auto *surface = static_cast<const Vec3 *>(
            soft->surface_rest_positions.contents);
        const auto *indices = static_cast<const std::uint32_t *>(
            soft->surface_indices.contents);
        const auto find_anchor = [&](Vec3 point, std::uint32_t &triangle,
                                     Vec3 &weights, Vec3 &offset) {
            float best = std::numeric_limits<float>::infinity();
            for (std::uint32_t base = 0U;
                 base + 2U < soft->surface_index_count; base += 3U) {
                Vec3 candidate_weights{};
                const Vec3 nearest = closest_point_triangle(
                    point, surface[indices[base]], surface[indices[base + 1U]],
                    surface[indices[base + 2U]], candidate_weights);
                const Vec3 delta = subtract(point, nearest);
                const float squared = dot(delta, delta);
                if (squared >= best) continue;
                best = squared;
                triangle = base;
                weights = candidate_weights;
                offset = delta;
            }
            return triangle != UINT32_MAX;
        };
        if (options.attach_first &&
            !find_anchor(rope_positions[0], constants->first_triangle,
                         constants->first_weights, constants->first_offset))
            return invalid_argument("Soft body has no usable anchor triangle");
        if (options.attach_last &&
            !find_anchor(rope_positions[rope->count - 1U],
                         constants->last_triangle, constants->last_weights,
                         constants->last_offset))
            return invalid_argument("Soft body has no usable anchor triangle");
        impl_->bind(resource->table, 0U, rope->particles.positions);
        impl_->bind(resource->table, 1U, rope->particles.previous);
        impl_->bind(resource->table, 2U, rope->particles.velocities);
        impl_->bind(resource->table, 3U, rope->particles.inverse_masses);
        impl_->bind(resource->table, 4U, rope->soft_forces);
        impl_->bind(resource->table, 5U, soft->particles.positions);
        impl_->bind(resource->table, 6U, soft->particles.velocities);
        impl_->bind(resource->table, 7U, soft->particles.inverse_masses);
        impl_->bind(resource->table, 8U, soft->surface_positions);
        impl_->bind(resource->table, 9U, soft->surface_indices);
        impl_->bind(resource->table, 10U, soft->surface_bindings);
        impl_->bind(resource->table, 11U, soft->rope_forces);
        impl_->bind(resource->table, 12U, soft->rest_positions);
        impl_->bind(resource->table, 13U, soft->surface_rest_positions);
        impl_->bind(resource->table, 14U, resource->constants);
        impl_->bind(resource->table, 15U, resource->contact_count);
        impl_->bind(resource->table, 16U, resource->maximum_penetration);
        impl_->bind(resource->table, 17U, rope->anchor_states);
        impl_->bind(resource->table, 18U, resource->packed_state);
        impl_->bind(resource->table, 19U, resource->previous_surface);
        [impl_->residency commit];
        impl_->rope_soft.entries[slot] = std::move(resource);
        ++impl_->rope_soft.count;
        output = {slot, impl_->rope_soft.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate rope-soft-body coupling");
    }
}

Status MetalSystems::update_rope_soft_body_coupling(
    RopeSoftBodyCouplingId id, RopeSoftBodyCouplingOptions options) noexcept {
    auto *r = impl_ ? impl_->rope_soft.get(id) : nullptr;
    if (!r) return invalid_handle("Rope-soft-body coupling handle is stale");
    if (r->options.rope != options.rope ||
        r->options.soft_body != options.soft_body ||
        r->options.attach_first != options.attach_first ||
        r->options.attach_last != options.attach_last)
        return invalid_argument("Rope-soft-body endpoints are immutable");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F || !std::isfinite(options.friction) ||
        options.friction < 0.0F ||
        !std::isfinite(options.maximum_soft_body_acceleration) ||
        !(options.maximum_soft_body_acceleration > 0.0F) ||
        !std::isfinite(options.anchor_support_radius_scale) ||
        options.anchor_support_radius_scale < 0.0F ||
        !std::isfinite(options.anchor_contact_support_radius_scale) ||
        options.anchor_contact_support_radius_scale < 0.0F ||
        options.anchor_contact_support_radius_scale >
            options.anchor_support_radius_scale)
        return invalid_argument("Rope-soft-body coupling options are invalid");
    r->options = options;
    r->enabled = options.enabled;
    return success();
}

Status MetalSystems::remove_rope_soft_body_coupling(
    RopeSoftBodyCouplingId id) noexcept {
    if (!impl_ || !impl_->rope_soft.get(id))
        return invalid_handle("Rope-soft-body coupling handle is stale");
    impl_->rope_soft.erase(id);
    return success();
}

Status MetalSystems::add_rope_cloth_coupling(
    RopeClothCouplingOptions options,
    RopeClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    RopeResource *rope = impl_->ropes.get(options.rope);
    ClothResource *cloth = impl_->cloths.get(options.cloth);
    if (!rope || !cloth)
        return invalid_handle("Rope-cloth coupling endpoint is stale");
    if ((options.first_vertex != UINT32_MAX &&
         options.first_vertex >= cloth->count) ||
        (options.last_vertex != UINT32_MAX &&
         options.last_vertex >= cloth->count) ||
        (options.first_vertex == UINT32_MAX &&
         options.last_vertex == UINT32_MAX) ||
        !std::isfinite(options.anchor_effective_mass) ||
        !(options.anchor_effective_mass > 0.0F) ||
        !std::isfinite(options.maximum_cloth_acceleration) ||
        !(options.maximum_cloth_acceleration > 0.0F))
        return invalid_argument("Rope-cloth coupling options are invalid");
    const std::uint32_t vertices[2]{options.first_vertex,
                                    options.last_vertex};
    const auto *rope_positions = static_cast<const Vec3 *>(
        rope->particles.positions.contents);
    const auto *cloth_positions = static_cast<const Vec3 *>(
        cloth->particles.positions.contents);
    for (std::uint32_t end = 0U; end < 2U; ++end) {
        if (vertices[end] == UINT32_MAX) continue;
        if (end == 0U ? rope->options.first.enabled
                      : rope->options.last.enabled)
            return invalid_argument(
                "Rope endpoint already has a rigid attachment");
        const Vec3 delta = subtract(
            rope_positions[end == 0U ? 0U : rope->count - 1U],
            cloth_positions[vertices[end]]);
        if (dot(delta, delta) > 1.0e-6F)
            return invalid_argument(
                "Rope endpoint must coincide with its cloth vertex");
    }
    for (const auto &entry : impl_->rope_soft.entries)
        if (entry && entry->options.rope == options.rope &&
            ((options.first_vertex != UINT32_MAX &&
              entry->options.attach_first) ||
             (options.last_vertex != UINT32_MAX &&
              entry->options.attach_last)))
            return invalid_argument(
                "Rope endpoint already has a soft-body attachment");
    for (const auto &entry : impl_->rope_cloth.entries)
        if (entry && entry->options.rope == options.rope &&
            (entry->options.cloth == options.cloth ||
             (options.first_vertex != UINT32_MAX &&
              entry->options.first_vertex != UINT32_MAX) ||
             (options.last_vertex != UINT32_MAX &&
              entry->options.last_vertex != UINT32_MAX)))
            return invalid_argument(
                "Rope endpoint already has a cloth attachment");
    Status status = impl_->add_coupling(
        impl_->rope_cloth, options, rope->particles, rope->count,
        cloth->particles, cloth->count, rope->contact_forces,
        cloth->rope_forces, 1U, 0.0F,
        1.0F / options.anchor_effective_mass, 0.0F, 0.0F,
        options.maximum_cloth_acceleration, options.first_vertex,
        options.last_vertex, options.enabled, output);
    if (!status) return status;
    auto *resource = impl_->rope_cloth.get(output);
    impl_->bind(resource->table, 12U, rope->anchor_states);
    [impl_->residency commit];
    return success();
}

Status MetalSystems::update_rope_cloth_coupling(
    RopeClothCouplingId id, RopeClothCouplingOptions options) noexcept {
    auto *r = impl_ ? impl_->rope_cloth.get(id) : nullptr;
    if (!r) return invalid_handle("Rope-cloth coupling handle is stale");
    if (r->options.rope != options.rope ||
        r->options.cloth != options.cloth ||
        r->options.first_vertex != options.first_vertex ||
        r->options.last_vertex != options.last_vertex)
        return invalid_argument("Rope-cloth endpoints are immutable");
    const ClothResource *cloth = impl_->cloths.get(options.cloth);
    if ((options.first_vertex != UINT32_MAX && options.first_vertex >= cloth->count) ||
        (options.last_vertex != UINT32_MAX && options.last_vertex >= cloth->count) ||
        !std::isfinite(options.anchor_effective_mass) ||
        !(options.anchor_effective_mass > 0.0F) ||
        !std::isfinite(options.maximum_cloth_acceleration) ||
        !(options.maximum_cloth_acceleration > 0.0F))
        return invalid_argument("Rope-cloth coupling options are invalid");
    r->options = options;
    r->enabled = options.enabled;
    auto *c = static_cast<CouplingConstants *>(r->constants.contents);
    c->stiffness = 1.0F / options.anchor_effective_mass;
    c->maximum_force = options.maximum_cloth_acceleration;
    c->first_vertex = options.first_vertex;
    c->last_vertex = options.last_vertex;
    c->enabled = options.enabled ? 1U : 0U;
    return success();
}

Status MetalSystems::remove_rope_cloth_coupling(
    RopeClothCouplingId id) noexcept {
    if (!impl_ || !impl_->rope_cloth.get(id))
        return invalid_handle("Rope-cloth coupling handle is stale");
    impl_->rope_cloth.erase(id);
    return success();
}

Status MetalSystems::add_fluid_cloth_coupling(
    FluidClothCouplingOptions options,
    FluidClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    ClothResource *cloth = impl_->cloths.get(options.cloth);
    if (!fluid || !cloth)
        return invalid_handle("Fluid-cloth coupling endpoint is stale");
    if (!cloth->options.preserve_volume)
        return invalid_argument(
            "Fluid-cloth containment requires closed volume-preserving cloth");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F ||
        !std::isfinite(options.interaction_radius) ||
        options.interaction_radius < 0.0F ||
        !std::isfinite(options.stiffness) || !(options.stiffness > 0.0F) ||
        !std::isfinite(options.damping) || options.damping < 0.0F ||
        !std::isfinite(options.tangential_drag) ||
        options.tangential_drag < 0.0F ||
        !std::isfinite(options.maximum_force) ||
        !(options.maximum_force > 0.0F))
        return invalid_argument("Fluid-cloth coupling options are invalid");
    for (const auto &entry : impl_->fluid_cloth.entries)
        if (entry && entry->options.fluid == options.fluid &&
            entry->options.cloth == options.cloth)
            return invalid_argument("Fluid and cloth are already coupled");
    const std::uint32_t slot = impl_->fluid_cloth.free_slot();
    if (slot == impl_->fluid_cloth.entries.size())
        return capacity_exceeded("Fluid-cloth coupling capacity is exhausted");
    try {
        auto resource =
            std::make_unique<CouplingResource<FluidClothCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        resource->constants = impl_->buffer(sizeof(FluidClothConstants));
        resource->phase_constants =
            impl_->buffer(sizeof(DeformableConstants));
        resource->contribution_nodes = impl_->buffer(
            static_cast<std::size_t>(fluid->options.capacity) *
            sizeof(std::uint32_t) * 3U);
        resource->contribution_forces = impl_->buffer(
            static_cast<std::size_t>(fluid->options.capacity) *
            sizeof(Vec3) * 3U);
        resource->table = impl_->table(15U);
        resource->phase_table = impl_->table(16U);
        if (resource->constants == nil ||
            resource->phase_constants == nil ||
            resource->contribution_nodes == nil ||
            resource->contribution_forces == nil || resource->table == nil ||
            resource->phase_table == nil)
            return metal_failure(nil,
                                 "Could not allocate fluid-cloth coupling");
        impl_->bind(resource->table, 0U, fluid->particles.positions);
        impl_->bind(resource->table, 1U, fluid->particles.velocities);
        impl_->bind(resource->table, 2U, fluid->metadata);
        impl_->bind(resource->table, 3U, fluid->foam_sources);
        impl_->bind(resource->table, 4U, cloth->particles.positions);
        impl_->bind(resource->table, 5U, cloth->particles.velocities);
        impl_->bind(resource->table, 6U, cloth->particles.inverse_masses);
        impl_->bind(resource->table, 7U, cloth->surface_positions);
        impl_->bind(resource->table, 8U, cloth->surface_indices);
        impl_->bind(resource->table, 9U, cloth->surface_physical_indices);
        impl_->bind(resource->table, 10U, cloth->fluid_forces);
        impl_->bind(resource->table, 11U, resource->constants);
        impl_->bind(resource->table, 12U, fluid->accelerations);
        impl_->bind(resource->table, 13U,
                    resource->contribution_nodes);
        impl_->bind(resource->table, 14U,
                    resource->contribution_forces);
        impl_->bind(resource->phase_table, 0U, cloth->particles.positions);
        impl_->bind(resource->phase_table, 1U, cloth->particles.previous);
        impl_->bind(resource->phase_table, 2U, cloth->particles.velocities);
        impl_->bind(resource->phase_table, 3U,
                    cloth->particles.inverse_masses);
        impl_->bind(resource->phase_table, 4U, cloth->bonds);
        impl_->bind(resource->phase_table, 5U, cloth->active_bonds);
        impl_->bind(resource->phase_table, 6U, cloth->surface_positions);
        impl_->bind(resource->phase_table, 7U, resource->phase_constants);
        impl_->bind(resource->phase_table, 8U, cloth->triangle_indices);
        impl_->bind(resource->phase_table, 9U, cloth->bond_damage);
        impl_->bind(resource->phase_table, 10U,
                    cloth->surface_source_indices);
        impl_->bind(resource->phase_table, 11U, cloth->rigid_forces);
        impl_->bind(resource->phase_table, 12U, cloth->corrections);
        impl_->bind(resource->phase_table, 13U, cloth->neighbor_offsets);
        impl_->bind(resource->phase_table, 14U, cloth->neighbors);
        impl_->bind(resource->phase_table, 15U,
                    cloth->free_triangle_nodes);
        [impl_->residency commit];
        impl_->fluid_cloth.entries[slot] = std::move(resource);
        ++impl_->fluid_cloth.count;
        output = {slot, impl_->fluid_cloth.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate fluid-cloth coupling");
    }
}

Status MetalSystems::update_fluid_cloth_coupling(
    FluidClothCouplingId id, FluidClothCouplingOptions options) noexcept {
    auto *r = impl_ ? impl_->fluid_cloth.get(id) : nullptr;
    if (!r) return invalid_handle("Fluid-cloth coupling handle is stale");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    ClothResource *cloth = impl_->cloths.get(options.cloth);
    if (!fluid || !cloth || !cloth->options.preserve_volume)
        return invalid_handle(
            "Fluid-cloth coupling endpoint is stale or open");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F ||
        !std::isfinite(options.interaction_radius) ||
        options.interaction_radius < 0.0F ||
        !std::isfinite(options.stiffness) || !(options.stiffness > 0.0F) ||
        !std::isfinite(options.damping) || options.damping < 0.0F ||
        !std::isfinite(options.tangential_drag) ||
        options.tangential_drag < 0.0F ||
        !std::isfinite(options.maximum_force) ||
        !(options.maximum_force > 0.0F))
        return invalid_argument("Fluid-cloth coupling options are invalid");
    r->options = options;
    r->enabled = options.enabled;
    impl_->bind(r->table, 0U, fluid->particles.positions);
    impl_->bind(r->table, 1U, fluid->particles.velocities);
    impl_->bind(r->table, 2U, fluid->metadata);
    impl_->bind(r->table, 3U, fluid->foam_sources);
    impl_->bind(r->table, 4U, cloth->particles.positions);
    impl_->bind(r->table, 5U, cloth->particles.velocities);
    impl_->bind(r->table, 6U, cloth->particles.inverse_masses);
    impl_->bind(r->table, 7U, cloth->surface_positions);
    impl_->bind(r->table, 8U, cloth->surface_indices);
    impl_->bind(r->table, 9U, cloth->surface_physical_indices);
    impl_->bind(r->table, 10U, cloth->fluid_forces);
    impl_->bind(r->table, 12U, fluid->accelerations);
    impl_->bind(r->phase_table, 0U, cloth->particles.positions);
    impl_->bind(r->phase_table, 1U, cloth->particles.previous);
    impl_->bind(r->phase_table, 2U, cloth->particles.velocities);
    impl_->bind(r->phase_table, 3U, cloth->particles.inverse_masses);
    impl_->bind(r->phase_table, 4U, cloth->bonds);
    impl_->bind(r->phase_table, 5U, cloth->active_bonds);
    impl_->bind(r->phase_table, 6U, cloth->surface_positions);
    impl_->bind(r->phase_table, 8U, cloth->triangle_indices);
    impl_->bind(r->phase_table, 9U, cloth->bond_damage);
    impl_->bind(r->phase_table, 10U, cloth->surface_source_indices);
    impl_->bind(r->phase_table, 11U, cloth->rigid_forces);
    impl_->bind(r->phase_table, 12U, cloth->corrections);
    impl_->bind(r->phase_table, 13U, cloth->neighbor_offsets);
    impl_->bind(r->phase_table, 14U, cloth->neighbors);
    impl_->bind(r->phase_table, 15U, cloth->free_triangle_nodes);
    [impl_->residency commit];
    return success();
}

Status MetalSystems::remove_fluid_cloth_coupling(
    FluidClothCouplingId id) noexcept {
    if (!impl_ || !impl_->fluid_cloth.get(id))
        return invalid_handle("Fluid-cloth coupling handle is stale");
    impl_->fluid_cloth.erase(id);
    return success();
}

Status MetalSystems::add_soft_body_cloth_coupling(
    SoftBodyClothCouplingOptions options,
    SoftBodyClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    SoftResource *soft = impl_->soft_bodies.get(options.soft_body);
    ClothResource *cloth = impl_->cloths.get(options.cloth);
    if (!soft || !cloth)
        return invalid_handle("Soft-body-cloth coupling endpoint is stale");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F || !std::isfinite(options.friction) ||
        options.friction < 0.0F ||
        options.solver_iterations == 0U || options.solver_iterations > 16U)
        return invalid_argument("Soft-body-cloth coupling options are invalid");
    for (const auto &entry : impl_->soft_cloth.entries)
        if (entry && entry->options.soft_body == options.soft_body &&
            entry->options.cloth == options.cloth)
            return invalid_argument("Soft body and cloth are already coupled");
    const std::uint32_t slot = impl_->soft_cloth.free_slot();
    if (slot == impl_->soft_cloth.entries.size())
        return capacity_exceeded(
            "Soft-body-cloth coupling capacity is exhausted");
    try {
        auto resource = std::make_unique<
            CouplingResource<SoftBodyClothCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        resource->constants = impl_->buffer(sizeof(SoftClothConstants));
        resource->packed_state = impl_->buffer(
            static_cast<std::size_t>(soft->count) *
            sizeof(SoftClothContact));
        resource->contact_count = impl_->buffer(
            static_cast<std::size_t>(cloth->count) *
            sizeof(std::uint32_t));
        resource->table = impl_->table(24U);
        if (resource->constants == nil || resource->packed_state == nil ||
            resource->contact_count == nil || resource->table == nil)
            return metal_failure(
                nil, "Could not allocate soft-body-cloth coupling");
        impl_->bind(resource->table, 0U, soft->particles.positions);
        impl_->bind(resource->table, 1U, soft->particles.previous);
        impl_->bind(resource->table, 2U, soft->particles.velocities);
        impl_->bind(resource->table, 3U, soft->particles.inverse_masses);
        impl_->bind(resource->table, 4U, soft->cloth_forces);
        impl_->bind(resource->table, 5U, cloth->particles.positions);
        impl_->bind(resource->table, 6U, cloth->particles.previous);
        impl_->bind(resource->table, 7U, cloth->particles.velocities);
        impl_->bind(resource->table, 8U, cloth->particles.inverse_masses);
        impl_->bind(resource->table, 9U, cloth->surface_positions);
        impl_->bind(resource->table, 10U, cloth->surface_indices);
        impl_->bind(resource->table, 11U, cloth->surface_physical_indices);
        impl_->bind(resource->table, 12U, cloth->soft_forces);
        impl_->bind(resource->table, 13U, resource->constants);
        impl_->bind(resource->table, 14U, soft->surface_positions);
        impl_->bind(resource->table, 15U, soft->surface_bindings);
        impl_->bind(resource->table, 16U, soft->rest_positions);
        impl_->bind(resource->table, 17U, soft->surface_rest_positions);
        impl_->bind(resource->table, 22U, resource->packed_state);
        impl_->bind(resource->table, 23U, resource->contact_count);
        [impl_->residency commit];
        impl_->soft_cloth.entries[slot] = std::move(resource);
        ++impl_->soft_cloth.count;
        output = {slot, impl_->soft_cloth.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate soft-body-cloth coupling");
    }
}

Status MetalSystems::update_soft_body_cloth_coupling(
    SoftBodyClothCouplingId id,
    SoftBodyClothCouplingOptions options) noexcept {
    auto *r = impl_ ? impl_->soft_cloth.get(id) : nullptr;
    if (!r) return invalid_handle("Soft-body-cloth coupling handle is stale");
    if (r->options.soft_body != options.soft_body ||
        r->options.cloth != options.cloth)
        return invalid_argument("Soft-body-cloth endpoints are immutable");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F || !std::isfinite(options.friction) ||
        options.friction < 0.0F || options.solver_iterations == 0U ||
        options.solver_iterations > 16U)
        return invalid_argument("Soft-body-cloth coupling options are invalid");
    r->options = options;
    r->enabled = options.enabled;
    return success();
}

Status MetalSystems::remove_soft_body_cloth_coupling(
    SoftBodyClothCouplingId id) noexcept {
    if (!impl_ || !impl_->soft_cloth.get(id))
        return invalid_handle("Soft-body-cloth coupling handle is stale");
    impl_->soft_cloth.erase(id);
    return success();
}

Status MetalSystems::add_fluid_soft_body_coupling(
    FluidSoftBodyCouplingOptions options,
    FluidSoftBodyCouplingId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("World is not initialized");
    FluidResource *fluid = impl_->fluids.get(options.fluid);
    SoftResource *soft = impl_->soft_bodies.get(options.soft_body);
    if (!fluid || !soft)
        return invalid_handle("Fluid-soft-body coupling endpoint is stale");
    if (!soft->surface_closed)
        return invalid_argument(
            "Fluid-soft-body coupling requires a closed consistently wound skin");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F || !std::isfinite(options.friction) ||
        options.friction < 0.0F ||
        options.solver_iterations == 0U || options.solver_iterations > 16U)
        return invalid_argument("Fluid-soft-body coupling options are invalid");
    for (const auto &entry : impl_->fluid_soft.entries)
        if (entry && entry->options.fluid == options.fluid &&
            entry->options.soft_body == options.soft_body)
            return invalid_argument("Fluid and soft body are already coupled");
    const std::uint32_t slot = impl_->fluid_soft.free_slot();
    if (slot == impl_->fluid_soft.entries.size())
        return capacity_exceeded(
            "Fluid-soft-body coupling capacity is exhausted");
    try {
        auto resource = std::make_unique<
            CouplingResource<FluidSoftBodyCouplingOptions>>();
        resource->options = options;
        resource->enabled = options.enabled;
        resource->constants = impl_->buffer(sizeof(FluidSoftConstants));
        resource->phase_constants =
            impl_->buffer(sizeof(DeformableConstants));
        resource->contact_count = impl_->buffer(sizeof(std::uint32_t));
        resource->maximum_penetration = impl_->buffer(sizeof(float));
        resource->packed_state = impl_->buffer(
            (static_cast<std::size_t>(soft->count) + 1U) *
            sizeof(std::uint32_t));
        resource->previous_surface =
            impl_->buffer(soft->surface_count * sizeof(Vec3));
        const std::size_t contribution_count =
            static_cast<std::size_t>(fluid->options.capacity) * 12U;
        resource->contribution_nodes = impl_->buffer(
            contribution_count * sizeof(std::uint32_t));
        resource->contribution_positions = impl_->buffer(
            contribution_count * sizeof(Vec3));
        resource->contribution_changes = impl_->buffer(
            contribution_count * sizeof(Vec3));
        resource->contribution_forces = impl_->buffer(
            contribution_count * sizeof(Vec3));
        resource->table = impl_->table(31U);
        resource->phase_table = impl_->table(12U);
        if (resource->constants == nil || resource->phase_constants == nil ||
            resource->contact_count == nil ||
            resource->maximum_penetration == nil ||
            resource->packed_state == nil ||
            resource->previous_surface == nil ||
            resource->contribution_nodes == nil ||
            resource->contribution_positions == nil ||
            resource->contribution_changes == nil ||
            resource->contribution_forces == nil || resource->table == nil ||
            resource->phase_table == nil)
            return metal_failure(
                nil, "Could not allocate fluid-soft-body coupling");
        impl_->bind(resource->table, 0U, fluid->particles.positions);
        impl_->bind(resource->table, 1U, fluid->particles.previous);
        impl_->bind(resource->table, 2U, fluid->particles.velocities);
        impl_->bind(resource->table, 3U, fluid->accelerations);
        impl_->bind(resource->table, 4U, fluid->foam);
        impl_->bind(resource->table, 5U, fluid->metadata);
        impl_->bind(resource->table, 6U, soft->particles.positions);
        impl_->bind(resource->table, 7U, soft->particles.velocities);
        impl_->bind(resource->table, 8U, soft->particles.inverse_masses);
        impl_->bind(resource->table, 9U, soft->surface_positions);
        impl_->bind(resource->table, 10U, soft->surface_indices);
        impl_->bind(resource->table, 11U, soft->surface_bindings);
        impl_->bind(resource->table, 12U, soft->fluid_forces);
        impl_->bind(resource->table, 13U, resource->constants);
        impl_->bind(resource->table, 14U, soft->rest_positions);
        impl_->bind(resource->table, 15U, soft->surface_rest_positions);
        impl_->bind(resource->table, 16U, resource->contact_count);
        impl_->bind(resource->table, 17U,
                    resource->maximum_penetration);
        impl_->bind(resource->table, 18U,
                    resource->contribution_nodes);
        impl_->bind(resource->table, 19U,
                    resource->contribution_positions);
        impl_->bind(resource->table, 20U,
                    resource->contribution_changes);
        impl_->bind(resource->table, 21U,
                    resource->contribution_forces);
        impl_->bind(resource->table, 22U, resource->packed_state);
        impl_->bind(resource->table, 23U,
                    resource->previous_surface);
        impl_->bind(resource->table, 24U, soft->particles.previous);
        impl_->bind(resource->table, 25U, impl_->rigid_states);
        impl_->bind(resource->table, 26U, impl_->rigid_parameters);
        impl_->bind(resource->table, 27U, impl_->mesh_vertices);
        impl_->bind(resource->table, 28U, impl_->mesh_indices);
        impl_->bind(resource->table, 29U, impl_->mesh_infos);
        impl_->bind(resource->table, 30U, impl_->mesh_solid_planes);
        impl_->bind(resource->phase_table, 0U, soft->particles.positions);
        impl_->bind(resource->phase_table, 1U, soft->particles.previous);
        impl_->bind(resource->phase_table, 2U, soft->particles.velocities);
        impl_->bind(resource->phase_table, 3U,
                    soft->particles.inverse_masses);
        impl_->bind(resource->phase_table, 4U, soft->bonds);
        impl_->bind(resource->phase_table, 5U, soft->surface_positions);
        impl_->bind(resource->phase_table, 6U,
                    soft->surface_rest_positions);
        impl_->bind(resource->phase_table, 7U, soft->surface_bindings);
        impl_->bind(resource->phase_table, 8U, resource->phase_constants);
        impl_->bind(resource->phase_table, 9U, soft->rest_positions);
        impl_->bind(resource->phase_table, 10U, soft->corrections);
        impl_->bind(resource->phase_table, 11U, soft->shape_orientation);
        std::memset(resource->packed_state.contents, 0,
                    resource->packed_state.length);
        copy_to(resource->previous_surface,
                static_cast<const Vec3 *>(soft->surface_positions.contents),
                soft->surface_count);
        [impl_->residency commit];
        impl_->fluid_soft.entries[slot] = std::move(resource);
        ++impl_->fluid_soft.count;
        output = {slot, impl_->fluid_soft.generations[slot]};
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not allocate fluid-soft-body coupling");
    }
}

Status MetalSystems::update_fluid_soft_body_coupling(
    FluidSoftBodyCouplingId id,
    FluidSoftBodyCouplingOptions options) noexcept {
    auto *r = impl_ ? impl_->fluid_soft.get(id) : nullptr;
    if (!r) return invalid_handle("Fluid-soft-body coupling handle is stale");
    if (r->options.fluid != options.fluid ||
        r->options.soft_body != options.soft_body)
        return invalid_argument("Fluid-soft-body endpoints are immutable");
    if (!std::isfinite(options.contact_distance) ||
        options.contact_distance < 0.0F || !std::isfinite(options.friction) ||
        options.friction < 0.0F || options.solver_iterations == 0U ||
        options.solver_iterations > 16U)
        return invalid_argument("Fluid-soft-body coupling options are invalid");
    r->options = options;
    r->enabled = options.enabled;
    return success();
}

Status MetalSystems::remove_fluid_soft_body_coupling(
    FluidSoftBodyCouplingId id) noexcept {
    if (!impl_ || !impl_->fluid_soft.get(id))
        return invalid_handle("Fluid-soft-body coupling handle is stale");
    impl_->fluid_soft.erase(id);
    return success();
}

void MetalSystems::set_rigid_resources(
    void *ids, void *states, void *previous_states, void *frame_states,
    void *parameters, void *vertices, void *indices, void *meshes,
    void *solid_planes, std::uint32_t count) noexcept {
    if (!impl_) return;
    impl_->rigid_ids = (__bridge id<MTLBuffer>)ids;
    impl_->rigid_states = (__bridge id<MTLBuffer>)states;
    impl_->rigid_previous_states =
        (__bridge id<MTLBuffer>)previous_states;
    impl_->rigid_frame_states = (__bridge id<MTLBuffer>)frame_states;
    impl_->rigid_parameters = (__bridge id<MTLBuffer>)parameters;
    impl_->mesh_vertices = (__bridge id<MTLBuffer>)vertices;
    impl_->mesh_indices = (__bridge id<MTLBuffer>)indices;
    impl_->mesh_infos = (__bridge id<MTLBuffer>)meshes;
    impl_->mesh_solid_planes = (__bridge id<MTLBuffer>)solid_planes;
    impl_->rigid_count = count;
}

void MetalSystems::invalidate_smoke_grid_static_metadata() noexcept {
    if (!impl_) return;
    for (auto &entry : impl_->smokes.entries)
        if (entry) entry->static_metadata_valid = false;
}

namespace {
Status rebuild_cloth_topology(ClothResource &cloth, bool initial) noexcept {
    if (cloth.topology_active.empty()) return success();
    const auto *live = static_cast<const std::uint8_t *>(
        cloth.active_bonds.contents);
    if (!initial && std::equal(cloth.topology_active.begin(),
                               cloth.topology_active.end(), live))
        return success();
    try {
        const std::uint32_t corners = cloth.triangle_index_count;
        std::uint32_t count = cloth.count;
        auto &active = cloth.topology_scratch_active;
        auto &old_indices = cloth.topology_scratch_old_indices;
        auto &indices = cloth.topology_scratch_indices;
        auto &parent = cloth.topology_scratch_parent;
        auto &nodes = cloth.topology_scratch_nodes;
        auto &used = cloth.topology_scratch_used;
        auto &degrees = cloth.topology_scratch_degrees;
        auto &copies = cloth.topology_scratch_copies;
        std::copy_n(live, cloth.bond_count, active.begin());
        const auto *current_indices = static_cast<const std::uint32_t *>(
            cloth.triangle_indices.contents);
        std::copy_n(current_indices, corners, old_indices.begin());
        std::copy_n(current_indices, corners, indices.begin());
        for (std::uint32_t corner = 0U; corner < corners; ++corner)
            parent[corner] = corner;
        const auto root = [&](std::uint32_t corner) {
            while (parent[corner] != corner) {
                parent[corner] = parent[parent[corner]];
                corner = parent[corner];
            }
            return corner;
        };
        for (const ClothSeam &seam : cloth.seams) {
            if (active[seam.bond] != 0U) {
                parent[root(seam.corners[2])] = root(seam.corners[0]);
                parent[root(seam.corners[3])] = root(seam.corners[1]);
            } else if (seam.bending != UINT32_MAX) {
                active[seam.bending] = 0U;
            }
        }
        std::fill(nodes.begin(), nodes.end(), UINT32_MAX);
        std::fill(used.begin(), used.end(), std::uint8_t{});
        std::fill(degrees.begin(), degrees.end(), 0U);
        copies.clear();
        for (std::uint32_t corner = 0U; corner < corners; ++corner) {
            const std::uint32_t group = root(corner);
            std::uint32_t &node = nodes[group];
            if (node == UINT32_MAX) {
                const std::uint32_t old = old_indices[corner];
                node = old;
                if (used[old] != 0U) {
                    if (count == cloth.capacity)
                        return capacity_exceeded(
                            "Cloth split capacity is exhausted");
                    node = count++;
                    copies.push_back({node, old});
                }
                used[node] = 1U;
            }
            indices[corner] = node;
            ++degrees[node];
        }

        auto *metal_bonds =
            static_cast<MetalBond *>(cloth.bonds.contents);
        auto *public_bonds =
            static_cast<ClothBond *>(cloth.public_bonds.contents);
        for (std::uint32_t bond = 0U; bond < cloth.bond_count; ++bond) {
            const auto corners_for_bond = cloth.bond_corners[bond];
            const std::uint32_t first = indices[corners_for_bond[0]];
            const std::uint32_t second = indices[corners_for_bond[1]];
            metal_bonds[bond].first = first;
            metal_bonds[bond].second = second;
            metal_bonds[bond].active = active[bond] != 0U ? 1U : 0U;
            public_bonds[bond].first = first;
            public_bonds[bond].second = second;
        }

        auto &edges = cloth.topology_scratch_edges;
        auto &graph_degrees = cloth.topology_scratch_graph_degrees;
        auto &offsets = cloth.topology_scratch_offsets;
        auto &cursors = cloth.topology_scratch_cursors;
        auto &neighbors = cloth.topology_scratch_neighbors;
        auto &free_nodes = cloth.topology_scratch_free_nodes;
        edges.clear();
        std::fill(graph_degrees.begin(), graph_degrees.end(), 0U);
        std::fill(free_nodes.begin(), free_nodes.end(), std::uint8_t{});
        const auto link = [&](std::uint32_t first, std::uint32_t second,
                              float rest_length, float compliance,
                              std::uint32_t bond) {
            if (first == second) return;
            const std::uint32_t low = std::min(first, second);
            const std::uint32_t high = std::max(first, second);
            for (const ClothGraphEdge &edge : edges)
                if (std::min(edge.first, edge.second) == low &&
                    std::max(edge.first, edge.second) == high)
                    return;
            edges.push_back(
                {first, second, rest_length, compliance, bond});
            ++graph_degrees[first];
            ++graph_degrees[second];
        };
        for (std::uint32_t corner = 0U; corner < corners; ++corner) {
            const std::uint32_t next =
                corner / 3U * 3U + (corner + 1U) % 3U;
            const ClothBond &bond = public_bonds[cloth.triangle_bonds[corner]];
            link(indices[corner], indices[next], bond.rest_length,
                 cloth.options.stretch_compliance, UINT32_MAX);
        }
        for (std::uint32_t bond = 0U; bond < cloth.bond_count; ++bond) {
            const ClothBond &item = public_bonds[bond];
            if (item.bending && active[bond] != 0U)
                link(item.first, item.second, item.rest_length,
                     cloth.options.bending_compliance, bond);
        }
        std::uint32_t neighbor_count = 0U;
        offsets[0] = 0U;
        for (std::uint32_t node = 0U; node < count; ++node) {
            if (graph_degrees[node] >
                cloth.neighbor_capacity - neighbor_count)
                return capacity_exceeded(
                    "Cloth split links exceed allocated capacity");
            neighbor_count += graph_degrees[node];
            offsets[node + 1U] = neighbor_count;
            cursors[node] = offsets[node];
        }
        for (const ClothGraphEdge &edge : edges) {
            neighbors[cursors[edge.first]++] = {
                edge.second, edge.rest_length, edge.compliance, edge.bond};
            neighbors[cursors[edge.second]++] = {
                edge.first, edge.rest_length, edge.compliance, edge.bond};
        }
        const auto *corner_sources = static_cast<const std::uint32_t *>(
            cloth.surface_source_indices.contents);
        for (std::uint32_t corner = 0U; corner < corners; corner += 3U) {
            bool detached = true;
            for (std::uint32_t item = 0U; item < 3U; ++item) {
                const std::uint32_t node = indices[corner + item];
                detached &= degrees[node] == 1U &&
                    cloth.source_inverse_masses[
                        corner_sources[corner + item]] > 0.0F;
            }
            if (detached)
                for (std::uint32_t item = 0U; item < 3U; ++item)
                    free_nodes[indices[corner + item]] = 1U;
        }

        auto *positions =
            static_cast<Vec3 *>(cloth.particles.positions.contents);
        auto *previous =
            static_cast<Vec3 *>(cloth.particles.previous.contents);
        auto *velocities =
            static_cast<Vec3 *>(cloth.particles.velocities.contents);
        auto *inverse_masses = static_cast<float *>(
            cloth.particles.inverse_masses.contents);
        auto *sources =
            static_cast<std::uint32_t *>(cloth.source_indices.contents);
        auto *rigid_forces =
            static_cast<Vec3 *>(cloth.rigid_forces.contents);
        auto *fluid_forces =
            static_cast<Vec3 *>(cloth.fluid_forces.contents);
        auto *soft_forces =
            static_cast<Vec3 *>(cloth.soft_forces.contents);
        auto *rope_forces =
            static_cast<Vec3 *>(cloth.rope_forces.contents);
        for (const auto copy : copies) {
            const std::uint32_t node = copy[0], old = copy[1];
            positions[node] = positions[old];
            previous[node] = previous[old];
            velocities[node] = velocities[old];
            sources[node] = sources[old];
            rigid_forces[node] = {};
            fluid_forces[node] = {};
            soft_forces[node] = {};
            rope_forces[node] = {};
        }
        for (std::uint32_t node = 0U; node < count; ++node) {
            const std::uint32_t source = sources[node];
            inverse_masses[node] = degrees[node] == 0U
                ? cloth.source_inverse_masses[source]
                : cloth.source_inverse_masses[source] *
                      static_cast<float>(cloth.source_degrees[source]) /
                      static_cast<float>(degrees[node]);
        }
        std::memcpy(cloth.triangle_indices.contents, indices.data(),
                    corners * sizeof(std::uint32_t));
        std::memcpy(cloth.surface_physical_indices.contents, indices.data(),
                    corners * sizeof(std::uint32_t));
        std::memcpy(cloth.active_bonds.contents, active.data(),
                    cloth.bond_count * sizeof(std::uint8_t));
        std::memcpy(cloth.neighbor_offsets.contents, offsets.data(),
                    (static_cast<std::size_t>(count) + 1U) *
                        sizeof(std::uint32_t));
        std::memcpy(cloth.neighbors.contents, neighbors.data(),
                    neighbor_count * sizeof(DeformableNeighbor));
        std::memcpy(cloth.free_triangle_nodes.contents, free_nodes.data(),
                    static_cast<std::size_t>(count) * sizeof(std::uint8_t));
        cloth.count = count;
        std::copy(active.begin(), active.end(),
                  cloth.topology_active.begin());
        if (!initial) ++cloth.revision;
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not rebuild torn cloth topology");
    } catch (...) {
        return {StatusCode::internal_error, 0,
                "Unexpected torn cloth topology failure"};
    }
}
} // namespace

Status MetalSystems::begin_frame(bool collect_fluid_contacts,
                                 float frame_timestep) noexcept {
    if (!impl_) return invalid_argument("World is not initialized");
    impl_->frame_timestep = frame_timestep;
    impl_->frame_inverse_timestep = 1.0F / frame_timestep;
    impl_->collect_fluid_contacts = collect_fluid_contacts;
    impl_->fluid_contact_count = 0U;
    impl_->fluid_contact_overflow = 0U;
    impl_->contact_cache_frame = UINT64_MAX;
    *static_cast<std::uint32_t *>(
        impl_->fluid_neighbor_overflow.contents) = 0U;
    *static_cast<std::uint32_t *>(
        impl_->fluid_maximum_neighbor_count.contents) = 0U;
    const auto clear = [](id<MTLBuffer> buffer, std::uint32_t count) {
        if (buffer != nil && count != 0U)
            std::memset(buffer.contents, 0, count * sizeof(Vec3));
    };
    if (collect_fluid_contacts) {
        for (auto &entry : impl_->fluids.entries) {
            if (!entry) continue;
            std::memset(entry->contact_flags.contents, 0,
                        entry->options.capacity * sizeof(std::uint32_t));
        }
    }
    for (auto &entry : impl_->cloths.entries) {
        if (!entry) continue;
        Status status = rebuild_cloth_topology(*entry);
        if (!status) return status;
        clear(entry->rigid_forces, entry->count);
        clear(entry->fluid_forces, entry->count);
        clear(entry->soft_forces, entry->count);
        clear(entry->rope_forces, entry->count);
    }
    for (auto &entry : impl_->soft_bodies.entries) {
        if (!entry) continue;
        clear(entry->rigid_forces, entry->count);
        clear(entry->cloth_forces, entry->count);
        clear(entry->fluid_forces, entry->count);
        clear(entry->rope_forces, entry->count);
    }
    for (auto &entry : impl_->ropes.entries) {
        if (!entry) continue;
        clear(entry->constraint_forces, entry->count);
        clear(entry->contact_forces, entry->count);
        clear(entry->fluid_forces, entry->count);
        clear(entry->soft_forces, entry->count);
    }
    const auto clear_coupling_diagnostics = [](auto &slots) {
        for (auto &entry : slots.entries) {
            if (!entry) continue;
            *static_cast<std::uint32_t *>(entry->contact_count.contents) = 0U;
            *static_cast<float *>(entry->maximum_penetration.contents) = 0.0F;
        }
    };
    clear_coupling_diagnostics(impl_->fluid_rope);
    clear_coupling_diagnostics(impl_->fluid_soft);
    clear_coupling_diagnostics(impl_->rope_soft);
    return success();
}

namespace {
void barrier(id<MTL4ComputeCommandEncoder> encoder) {
    [encoder barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
}

MetalTimingRecord *begin_timing(
    id<MTL4ComputeCommandEncoder> encoder, MetalTimingContext *context,
    MetalTimingStage stage, std::uint32_t launch_count = 1U) {
    if (context == nullptr || !context->enabled ||
        stage == MetalTimingStage::none || launch_count == 0U)
        return nullptr;
    if (context->counter_heap == nullptr || context->records == nullptr ||
        context->record_count >= context->record_capacity ||
        context->next_index + 1U >= context->final_index) {
        context->overflowed = true;
        return nullptr;
    }
    MetalTimingRecord &record =
        context->records[context->record_count++];
    record = {stage, context->next_index, context->next_index + 1U,
              launch_count};
    context->next_index += 2U;
    id<MTL4CounterHeap> heap =
        (__bridge id<MTL4CounterHeap>)context->counter_heap;
    [encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                  intoHeap:heap
                                   atIndex:record.begin_index];
    return &record;
}

void end_timing(id<MTL4ComputeCommandEncoder> encoder,
                MetalTimingContext *context,
                const MetalTimingRecord *record) {
    if (record == nullptr || context == nullptr) return;
    id<MTL4CounterHeap> heap =
        (__bridge id<MTL4CounterHeap>)context->counter_heap;
    [encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                  intoHeap:heap
                                   atIndex:record->end_index];
}

} // namespace

void MetalSystems::encode(void *opaque_encoder, MetalSystemPhase phase,
                          float timestep, std::uint32_t substeps,
                          Vec3 gravity,
                          MetalTimingContext *timings) noexcept {
    if (!impl_) return;
    id<MTL4ComputeCommandEncoder> encoder =
        (__bridge id<MTL4ComputeCommandEncoder>)opaque_encoder;
    const auto dispatch = [&](MetalTimingStage stage,
                              id<MTL4ArgumentTable> table,
                              id<MTLComputePipelineState> pipeline,
                              MTLSize threads, MTLSize group) {
        MetalTimingRecord *record =
            begin_timing(encoder, timings, stage);
        [encoder setArgumentTable:table];
        [encoder setComputePipelineState:pipeline];
        [encoder dispatchThreads:threads threadsPerThreadgroup:group];
        barrier(encoder);
        end_timing(encoder, timings, record);
    };
    const auto dispatch_untimed = [&](id<MTL4ArgumentTable> table,
                                      id<MTLComputePipelineState> pipeline,
                                      MTLSize threads, MTLSize group) {
        [encoder setArgumentTable:table];
        [encoder setComputePipelineState:pipeline];
        [encoder dispatchThreads:threads threadsPerThreadgroup:group];
        barrier(encoder);
    };
    const auto encode_fluid_sort = [&](FluidResource &resource) {
        MetalTimingRecord *record = begin_timing(
            encoder, timings, MetalTimingStage::fluid_neighbor_sort);
        [encoder setArgumentTable:resource.table];
        const auto run = [&](id<MTLComputePipelineState> pipeline,
                             NSUInteger thread_count) {
            const NSUInteger group_size = std::min<NSUInteger>(
                64U, pipeline.maxTotalThreadsPerThreadgroup);
            [encoder setComputePipelineState:pipeline];
            [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
            barrier(encoder);
        };
        run(impl_->fluid_cell_keys_pipeline, resource.options.capacity);
        const NSUInteger block_count =
            (static_cast<NSUInteger>(resource.options.capacity) + 255U) /
            256U;
        for (std::uint32_t pass = 0U; pass < 8U; ++pass) {
            run(impl_->fluid_radix_histogram_pipelines[pass], block_count);
            run(impl_->fluid_radix_prefix_blocks_pipeline, 256U);
            run(impl_->fluid_radix_prefix_buckets_pipeline, 1U);
            run(impl_->fluid_radix_scatter_pipelines[pass], block_count);
        }
        end_timing(encoder, timings, record);
    };
    const auto encode_smoke_advection = [&](SmokeResource &resource) {
        MetalTimingRecord *record = begin_timing(
            encoder, timings, MetalTimingStage::smoke_advection);
        [encoder setArgumentTable:resource.advection_table];
        const auto run = [&](id<MTLComputePipelineState> pipeline) {
            const NSUInteger group_size = std::min<NSUInteger>(
                64U, pipeline.maxTotalThreadsPerThreadgroup);
            [encoder setComputePipelineState:pipeline];
            [encoder dispatchThreads:
                         MTLSizeMake(resource.options.capacity, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
            barrier(encoder);
        };
        if (resource.options.grid_resolution == 0U) {
            run(impl_->smoke_particle_density_pipeline);
            run(impl_->smoke_particle_vorticity_pipeline);
            run(impl_->smoke_particle_forces_pipeline);
            run(impl_->smoke_particle_integrate_pipeline);
        } else {
            run(impl_->smoke_grid_advect_particles_pipeline);
        }
        end_timing(encoder, timings, record);
    };
    const auto encode_smoke_grid = [&](SmokeResource &resource) {
        MetalTimingRecord *record = begin_timing(
            encoder, timings, MetalTimingStage::smoke_grid);
        const auto run = [&](id<MTLComputePipelineState> pipeline,
                             NSUInteger thread_count) {
            if (thread_count == 0U) return;
            const NSUInteger group_size = std::min<NSUInteger>(
                64U, pipeline.maxTotalThreadsPerThreadgroup);
            [encoder setComputePipelineState:pipeline];
            [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
            barrier(encoder);
        };
        const NSUInteger contribution_count =
            static_cast<NSUInteger>(resource.options.capacity) * 27U;
        const NSUInteger block_count =
            (contribution_count + 255U) / 256U;
        const NSUInteger cell_count =
            static_cast<NSUInteger>(resource.options.grid_resolution) *
            resource.options.grid_vertical_resolution *
            resource.options.grid_resolution;
        [encoder setArgumentTable:resource.grid_raster_table];
        run(impl_->smoke_grid_merge_obstacles_pipeline, cell_count);
        [encoder setArgumentTable:resource.grid_splat_table];
        run(impl_->smoke_grid_splat_generate_pipeline,
            contribution_count);
        for (std::uint32_t pass = 0U; pass < 4U; ++pass) {
            run(impl_->smoke_grid_splat_histogram_pipelines[pass],
                block_count);
            run(impl_->smoke_grid_splat_prefix_blocks_pipeline, 256U);
            run(impl_->smoke_grid_splat_prefix_buckets_pipeline, 1U);
            run(impl_->smoke_grid_splat_scatter_pipelines[pass],
                block_count);
        }
        run(impl_->smoke_grid_splat_resolve_pipeline, cell_count);
        std::array<NSUInteger, 4U> level_n{
            resource.options.grid_resolution, 0U, 0U, 0U};
        std::array<NSUInteger, 4U> level_height{
            resource.options.grid_vertical_resolution, 0U, 0U, 0U};
        std::array<NSUInteger, 4U> level_cells{};
        std::array<NSUInteger, 4U> level_faces{};
        for (std::uint32_t level = 1U; level < 4U; ++level) {
            level_n[level] = std::max<NSUInteger>(
                2U, level_n[level - 1U] / 2U);
            level_height[level] = std::max<NSUInteger>(
                2U, level_height[level - 1U] / 2U);
        }
        for (std::uint32_t level = 0U; level < 4U; ++level) {
            const NSUInteger n = level_n[level];
            const NSUInteger height = level_height[level];
            level_cells[level] = n * height * n;
            level_faces[level] =
                (n + 1U) * height * n + n * (height + 1U) * n +
                n * height * (n + 1U);
        }
        [encoder setArgumentTable:resource.grid_pressure_table];
        run(impl_->smoke_grid_pressure_begin_pipeline, 1U);
        run(impl_->smoke_grid_mark_boundaries_pipeline, level_faces[0]);
        run(impl_->smoke_grid_advect_forward_pipeline, level_faces[0]);
        run(impl_->smoke_grid_advect_reverse_pipeline, level_faces[0]);
        run(impl_->smoke_grid_correct_face_pipeline, level_faces[0]);
        run(impl_->smoke_grid_diagnostics_pre_pipeline, cell_count);
        run(impl_->smoke_grid_subgrid_force_pipeline, cell_count);
        run(impl_->smoke_grid_apply_forces_pipeline, level_faces[0]);
        run(impl_->smoke_grid_apply_boundaries_pipeline, level_faces[0]);
        run(impl_->smoke_grid_divergence_pipeline, cell_count);
        for (std::uint32_t level = 0U; level < 3U; ++level)
            run(impl_->smoke_pressure_restrict_open_pipelines[level],
                level_faces[level + 1U]);
        const auto smooth = [&](std::uint32_t level,
                                std::uint32_t pairs) {
            for (std::uint32_t pair = 0U; pair < pairs; ++pair) {
                run(impl_->smoke_pressure_smooth_pipelines[level][0],
                    level_cells[level]);
                run(impl_->smoke_pressure_smooth_pipelines[level][1],
                    level_cells[level]);
            }
        };
        const std::uint32_t cycles = std::max(
            1U, (resource.options.grid_pressure_iterations + 4U) / 5U);
        for (std::uint32_t cycle = 0U; cycle < cycles; ++cycle) {
            for (std::uint32_t level = 0U; level < 3U; ++level) {
                smooth(level, 1U);
                run(impl_->smoke_pressure_residual_pipelines[level],
                    level_cells[level]);
                run(impl_->smoke_pressure_restrict_residual_pipelines[level],
                    level_cells[level + 1U]);
                run(impl_->smoke_pressure_clear_pipelines[level],
                    level_cells[level + 1U]);
            }
            smooth(3U, 6U);
            for (std::int32_t level = 2; level >= 0; --level) {
                run(impl_->smoke_pressure_prolong_pipelines[level],
                    level_cells[static_cast<std::uint32_t>(level)]);
                smooth(static_cast<std::uint32_t>(level), 1U);
            }
            run(impl_->smoke_pressure_cycle_begin_pipeline, 1U);
            run(impl_->smoke_pressure_residual_pipelines[0], level_cells[0]);
            run(impl_->smoke_pressure_cycle_end_pipeline, 1U);
        }
        run(impl_->smoke_grid_project_face_pipeline, level_faces[0]);
        run(impl_->smoke_grid_diagnostics_post_pipeline, cell_count);
        end_timing(encoder, timings, record);
    };
    const auto encode_cloth_constraints = [&](ClothResource &resource) {
        MetalTimingRecord *record = begin_timing(
            encoder, timings, MetalTimingStage::cloth_constraints);
        [encoder setArgumentTable:resource.table];
        const auto run = [&](id<MTLComputePipelineState> pipeline,
                             NSUInteger thread_count) {
            if (thread_count == 0U) return;
            const NSUInteger group_size = std::min<NSUInteger>(
                thread_count,
                std::min<NSUInteger>(
                    64U, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder setComputePipelineState:pipeline];
            [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
            barrier(encoder);
        };
        for (std::uint32_t iteration = 0U;
             iteration < resource.options.solver_iterations; ++iteration) {
            run(impl_->cloth_project_pipeline, resource.count);
            run(impl_->cloth_apply_pipeline, resource.count);
            if (resource.options.preserve_volume) {
                [encoder setComputePipelineState:impl_->cloth_pipeline];
                [encoder dispatchThreads:MTLSizeMake(128U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128U, 1, 1)];
                barrier(encoder);
            }
        }
        run(impl_->cloth_finalize_pipeline, resource.count);
        run(impl_->cloth_surface_update_pipeline, resource.surface_count);
        end_timing(encoder, timings, record);
    };
    const auto encode_rope_solve = [&](RopeResource &resource) {
        MetalTimingRecord *record = begin_timing(
            encoder, timings, MetalTimingStage::rope_solve);
        [encoder setArgumentTable:resource.table];
        const NSUInteger group_size = std::min<NSUInteger>(
            resource.count,
            std::min<NSUInteger>(
                64U, impl_->rope_prediction_pipeline
                         .maxTotalThreadsPerThreadgroup));
        [encoder setComputePipelineState:impl_->rope_prediction_pipeline];
        [encoder dispatchThreads:MTLSizeMake(resource.count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
        barrier(encoder);
        [encoder setComputePipelineState:impl_->rope_pipeline];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        barrier(encoder);
        end_timing(encoder, timings, record);
    };
    const auto encode_rigid_contacts =
        [&](ParticleBuffers &particles, std::uint32_t count, float radius,
            float friction, id<MTLBuffer> constants,
            id<MTL4ArgumentTable> table, bool acceleration_diagnostic,
            FluidId fluid, id<MTLBuffer> stable_ids,
            id<MTLBuffer> contact_samples, id<MTLBuffer> contact_flags,
            MetalTimingStage stage, float contact_timestep,
            float particle_inverse_mass, bool first_iteration,
            float maximum_reaction_speed, bool solid_contacts,
            bool share_position, std::uint32_t first_spawned,
            bool recover_spawn, float spawn_clearance, Vec3 up,
            Vec3 contact_gravity, float movable_mass) {
            if (impl_->rigid_count == 0U || table == nil ||
                (count == 0U && fluid.generation == 0U))
                return;
            *static_cast<ParticleRigidConstants *>(constants.contents) = {
                count, impl_->rigid_count, radius, friction, 0.0F,
                maximum_reaction_speed,
                contact_timestep, acceleration_diagnostic ? 1U : 0U, fluid,
                impl_->collect_fluid_contacts && fluid.generation != 0U
                    ? 1U
                    : 0U,
                particle_inverse_mass, first_iteration ? 1U : 0U};
            auto *rigid_constants =
                static_cast<ParticleRigidConstants *>(constants.contents);
            rigid_constants->solid_contacts = solid_contacts ? 1U : 0U;
            rigid_constants->share_position = share_position ? 1U : 0U;
            rigid_constants->first_spawned = first_spawned;
            rigid_constants->recover_spawn = recover_spawn ? 1U : 0U;
            rigid_constants->spawn_clearance = spawn_clearance;
            rigid_constants->up = up;
            rigid_constants->gravity = contact_gravity;
            rigid_constants->movable_mass = movable_mass;
            impl_->bind(table, 0U, particles.positions);
            impl_->bind(table, 1U, particles.velocities);
            impl_->bind(table, 2U, particles.inverse_masses);
            impl_->bind(table, 3U, impl_->rigid_states);
            impl_->bind(table, 4U, impl_->rigid_parameters);
            impl_->bind(table, 5U, impl_->mesh_vertices);
            impl_->bind(table, 6U, impl_->mesh_indices);
            impl_->bind(table, 7U, impl_->mesh_infos);
            impl_->bind(table, 8U, constants);
            impl_->bind(table, 10U, stable_ids);
            impl_->bind(table, 11U, impl_->rigid_ids);
            impl_->bind(table, 12U, contact_samples);
            impl_->bind(table, 13U, contact_flags);
            impl_->bind(table, 15U, particles.previous);
            impl_->bind(
                table, 16U,
                phase == MetalSystemPhase::frame_end
                    ? impl_->rigid_frame_states
                    : impl_->rigid_previous_states);
            impl_->bind(table, 22U, impl_->mesh_solid_planes);
            const NSUInteger group = std::min<NSUInteger>(
                128U, impl_->particle_rigid_pipeline
                          .maxTotalThreadsPerThreadgroup);
            dispatch(stage, table, impl_->particle_rigid_pipeline,
                     MTLSizeMake(group, 1, 1),
                     MTLSizeMake(group, 1, 1));
        };
    const auto encode_soft_constraints = [&](SoftResource &resource) {
        MetalTimingRecord *record = begin_timing(
            encoder, timings, MetalTimingStage::soft_body_constraints);
        const auto run = [&](id<MTLComputePipelineState> pipeline,
                             NSUInteger thread_count) {
            if (thread_count == 0U) return;
            const NSUInteger group_size = std::min<NSUInteger>(
                thread_count,
                std::min<NSUInteger>(
                    64U, pipeline.maxTotalThreadsPerThreadgroup));
            [encoder setArgumentTable:resource.table];
            [encoder setComputePipelineState:pipeline];
            [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
            barrier(encoder);
        };
        const auto run_rigid = [&](id<MTLComputePipelineState> pipeline,
                                   NSUInteger thread_count,
                                   NSUInteger requested_group = 64U) {
            if (thread_count == 0U) return;
            const NSUInteger group_size = std::min<NSUInteger>(
                requested_group, pipeline.maxTotalThreadsPerThreadgroup);
            dispatch_untimed(resource.rigid_table, pipeline,
                             MTLSizeMake(thread_count, 1, 1),
                             MTLSizeMake(group_size, 1, 1));
        };
        auto *soft_contact_constants =
            static_cast<ParticleRigidConstants *>(
                resource.rigid_constants.contents);
        *soft_contact_constants = {};
        soft_contact_constants->particle_count = resource.count;
        soft_contact_constants->rigid_count = impl_->rigid_count;
        soft_contact_constants->maximum_reaction_speed =
            resource.options.maximum_speed;
        soft_contact_constants->movable_mass = resource.movable_mass;
        run_rigid(impl_->soft_contact_clear_pipeline, resource.count);
        run_rigid(impl_->soft_measure_momentum_pipeline, 128U, 128U);
        const auto contacts = [&] {
            encode_rigid_contacts(
                resource.particles, resource.count,
                resource.options.node_radius,
                resource.options.contact_friction,
                resource.rigid_constants, resource.rigid_table, false, {},
                resource.rigid_constants, resource.rigid_constants,
                resource.rigid_constants,
                MetalTimingStage::soft_body_contacts, timestep, 0.0F, true,
                resource.options.maximum_speed, true, true, 0U, false,
                0.0F, {0.0F, 1.0F, 0.0F}, gravity,
                resource.movable_mass);
        };
        const auto finish_contact_pass = [&] {
            if (impl_->rigid_count == 0U) return;
            run_rigid(impl_->soft_contact_friction_pipeline,
                      resource.count);
            run_rigid(impl_->soft_contact_finish_pipeline, 128U, 128U);
        };
        // CUDA solves soft-body contacts as part of graph projection: once
        // before the first spring pass, then after every two passes. Matching
        // that cadence keeps the graph solve from pulling separated nodes
        // back through a rigid boundary.
        contacts();
        finish_contact_pass();
        for (std::uint32_t iteration = 0U;
             iteration < resource.options.solver_iterations; ++iteration) {
            run(impl_->soft_project_pipeline, resource.count);
            run(impl_->soft_apply_pipeline, resource.count);
            if (iteration + 1U == resource.options.solver_iterations &&
                resource.options.shape_matching_stiffness > 0.0F) {
                [encoder setArgumentTable:resource.table];
                [encoder setComputePipelineState:impl_->soft_pipeline];
                [encoder dispatchThreads:MTLSizeMake(128U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128U, 1, 1)];
                barrier(encoder);
            }
            if ((iteration + 1U) % 2U == 0U ||
                iteration + 1U == resource.options.solver_iterations) {
                contacts();
                if (iteration + 1U != resource.options.solver_iterations)
                    finish_contact_pass();
            }
        }
        run(impl_->soft_finalize_pipeline, resource.count);
        if (resource.options.spring_damping > 0.0F) {
            run(impl_->soft_damping_prepare_pipeline, resource.count);
            run(impl_->soft_damping_apply_pipeline, resource.count);
        }
        finish_contact_pass();
        run_rigid(impl_->soft_restore_momentum_pipeline, 128U, 128U);
        run(impl_->soft_surface_update_pipeline, resource.surface_count);
        end_timing(encoder, timings, record);
    };
    const auto encode_paint = [&](bool select_fluid_source) {
        for (auto &entry : impl_->paint_rules.entries) {
            if (!entry || !entry->options.enabled) continue;
            PaintFieldResource *field =
                impl_->paint_fields.get(entry->options.target);
            if (!field) continue;
            const bool fluid_source =
                entry->options.source.generation != 0U;
            if (fluid_source != select_fluid_source) continue;
            PaintConstants constants{};
            constants.mode = fluid_source ? 0U : 1U;
            constants.source = fluid_source
                                   ? RigidBodyId{
                                         entry->options.source.index,
                                         entry->options.source.generation}
                                   : entry->options.rigid_source;
            constants.target = field->options.cloth.generation != 0U
                                   ? RigidBodyId{
                                         field->options.cloth.index,
                                         field->options.cloth.generation}
                                   : field->options.body;
            constants.mesh_index = field->options.mesh.index;
            constants.width = field->options.width;
            constants.height = field->options.height;
            constants.reach = fluid_source ? entry->options.reach
                                           : entry->options.brush_radius;
            constants.rigid_count = impl_->rigid_count;
            constants.enabled = 1U;
            if (fluid_source) {
                FluidResource *fluid =
                    impl_->fluids.get(entry->options.source);
                if (!fluid) continue;
                constants.particle_radius = fluid->options.particle_radius;
                impl_->bind(entry->table, 0U, fluid->particles.positions);
                impl_->bind(entry->table, 1U, fluid->metadata);
                impl_->bind(entry->table, 10U, entry->constants);
                impl_->bind(entry->table, 11U, entry->constants);
                impl_->bind(entry->table, 12U, entry->constants);
            } else {
                ClothResource *cloth =
                    impl_->cloths.get(field->options.cloth);
                if (!cloth) continue;
                constants.cloth_thickness = cloth->options.thickness;
                constants.cloth_count = cloth->count;
                constants.cloth_index_count = cloth->triangle_index_count;
                impl_->bind(entry->table, 0U, entry->constants);
                impl_->bind(entry->table, 1U, entry->constants);
                impl_->bind(entry->table, 10U,
                            cloth->particles.positions);
                impl_->bind(entry->table, 11U, cloth->triangle_indices);
                impl_->bind(entry->table, 12U, cloth->source_indices);
            }
            impl_->bind(entry->table, 4U, impl_->rigid_ids);
            impl_->bind(entry->table, 5U, impl_->rigid_states);
            impl_->bind(entry->table, 6U, impl_->rigid_parameters);
            impl_->bind(entry->table, 7U, impl_->mesh_vertices);
            impl_->bind(entry->table, 8U, impl_->mesh_indices);
            impl_->bind(entry->table, 9U, impl_->mesh_infos);
            *static_cast<PaintConstants *>(entry->constants.contents) =
                constants;
            [encoder setArgumentTable:entry->table];
            [encoder setComputePipelineState:impl_->paint_pipeline];
            [encoder dispatchThreads:MTLSizeMake(64, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            barrier(encoder);
        }
    };
    const auto encode_fluid_cloth =
        [&](FluidId fluid_id, FluidResource &fluid, float fluid_timestep,
            bool projection) {
            for (auto &entry : impl_->fluid_cloth.entries) {
                if (!entry || !entry->enabled ||
                    entry->options.fluid != fluid_id)
                    continue;
                ClothResource *cloth =
                    impl_->cloths.get(entry->options.cloth);
                if (!cloth) continue;
                const auto &options = entry->options;
                const float diameter =
                    2.0F * fluid.options.particle_radius;
                const float particle_mass = fluid.options.rest_density *
                    (fluid.options.rest_particle_volume > 0.0F
                         ? fluid.options.rest_particle_volume
                         : diameter * diameter * diameter);
                *static_cast<FluidClothConstants *>(
                    entry->constants.contents) = {
                    fluid_timestep,
                    cloth->triangle_index_count,
                    options.contact_distance > 0.0F
                        ? options.contact_distance
                        : fluid.options.particle_radius +
                              cloth->options.thickness,
                    options.interaction_radius > 0.0F
                        ? options.interaction_radius
                        : fluid.options.support_radius,
                    options.stiffness,
                    options.damping,
                    options.tangential_drag,
                    options.maximum_force,
                    particle_mass,
                    cloth->orientation,
                    fluid.options.maximum_speed,
                    cloth->surface_count,
                    options.enabled ? 1U : 0U,
                    cloth->count};
                if (projection) {
                    const NSUInteger group = std::min<NSUInteger>(
                        64U, impl_->fluid_cloth_project_pipeline
                                 .maxTotalThreadsPerThreadgroup);
                    dispatch(MetalTimingStage::fluid_cloth_contacts,
                             entry->table,
                             impl_->fluid_cloth_project_pipeline,
                             MTLSizeMake(fluid.options.capacity, 1, 1),
                             MTLSizeMake(group, 1, 1));
                    continue;
                }
                const auto run = [&](id<MTLComputePipelineState> pipeline,
                                     NSUInteger threads) {
                    const NSUInteger group = std::min<NSUInteger>(
                        64U, pipeline.maxTotalThreadsPerThreadgroup);
                    dispatch_untimed(entry->table, pipeline,
                                     MTLSizeMake(threads, 1, 1),
                                     MTLSizeMake(group, 1, 1));
                };
                MetalTimingRecord *record = begin_timing(
                    encoder, timings,
                    MetalTimingStage::fluid_cloth_contacts);
                run(impl_->fluid_cloth_pipeline,
                    fluid.options.capacity);
                run(impl_->fluid_cloth_apply_pipeline, cloth->count);
                *static_cast<DeformableConstants *>(
                    entry->phase_constants.contents) = {
                    fluid_timestep, {}, cloth->count, cloth->bond_count, 1U,
                    0.0F, 0.0F, 100.0F, cloth->options.thickness, 0.0F, 0U,
                    0.0F, cloth->surface_count,
                    cloth->triangle_index_count,
                    cloth->options.preserve_volume ? 1U : 0U,
                    cloth->options.target_volume,
                    cloth->options.volume_compliance, 0.0F, 0.0F, 0U,
                    0.0F, 1.0F};
                const auto run_cloth =
                    [&](id<MTLComputePipelineState> pipeline,
                        NSUInteger threads, NSUInteger group) {
                        dispatch_untimed(
                            entry->phase_table, pipeline,
                            MTLSizeMake(threads, 1, 1),
                            MTLSizeMake(group, 1, 1));
                    };
                const NSUInteger cloth_group = std::min<NSUInteger>(
                    cloth->count,
                    std::min<NSUInteger>(
                        64U, impl_->cloth_project_pipeline
                                 .maxTotalThreadsPerThreadgroup));
                if (cloth->count != 0U) {
                    run_cloth(impl_->cloth_project_pipeline, cloth->count,
                              cloth_group);
                    run_cloth(impl_->cloth_apply_pipeline, cloth->count,
                              cloth_group);
                }
                if (cloth->options.preserve_volume)
                    run_cloth(impl_->cloth_pipeline, 128U, 128U);
                run(impl_->fluid_cloth_surface_pipeline,
                    cloth->surface_count);
                end_timing(encoder, timings, record);
            }
        };
    const auto encode_fluid_rope =
        [&](FluidId fluid_id, FluidResource &fluid, float fluid_timestep) {
            for (auto &entry : impl_->fluid_rope.entries) {
                if (!entry || !entry->enabled ||
                    entry->options.fluid != fluid_id)
                    continue;
                auto *constants = static_cast<CouplingConstants *>(
                    entry->constants.contents);
                constants->timestep = fluid_timestep;
                constants->damping = impl_->frame_inverse_timestep;
                constants->count_a = fluid.count;
                const NSUInteger contact_group = std::min<NSUInteger>(
                    64U, impl_->fluid_rope_contact_pipeline
                             .maxTotalThreadsPerThreadgroup);
                MetalTimingRecord *record = begin_timing(
                    encoder, timings,
                    MetalTimingStage::fluid_rope_contacts);
                dispatch_untimed(
                    entry->table, impl_->fluid_rope_contact_pipeline,
                    MTLSizeMake(fluid.options.capacity, 1, 1),
                    MTLSizeMake(contact_group, 1, 1));
                const NSUInteger apply_group = std::min<NSUInteger>(
                    64U, impl_->fluid_rope_apply_pipeline
                             .maxTotalThreadsPerThreadgroup);
                dispatch_untimed(
                    entry->table, impl_->fluid_rope_apply_pipeline,
                    MTLSizeMake(constants->count_b, 1, 1),
                    MTLSizeMake(apply_group, 1, 1));
                end_timing(encoder, timings, record);
            }
        };
    const auto encode_fluid_soft =
        [&](FluidId fluid_id, FluidResource &fluid, float fluid_timestep) {
            for (auto &entry : impl_->fluid_soft.entries) {
                if (!entry || !entry->enabled ||
                    entry->options.fluid != fluid_id)
                    continue;
                SoftResource *soft =
                    impl_->soft_bodies.get(entry->options.soft_body);
                if (!soft) continue;
                const auto &options = entry->options;
                const float diameter =
                    2.0F * fluid.options.particle_radius;
                const float particle_mass = fluid.options.rest_density *
                    (fluid.options.rest_particle_volume > 0.0F
                         ? fluid.options.rest_particle_volume
                         : diameter * diameter * diameter);
                *static_cast<FluidSoftConstants *>(
                    entry->constants.contents) = {
                    fluid_timestep,
                    soft->count,
                    soft->surface_count,
                    soft->surface_index_count,
                    options.contact_distance > 0.0F
                        ? options.contact_distance
                        : fluid.options.particle_radius,
                    options.friction,
                    particle_mass,
                    fluid.options.maximum_speed,
                    soft->options.maximum_speed,
                    soft->surface_orientation,
                    options.enabled ? 1U : 0U,
                    0.2F * soft->options.node_radius,
                    1.0F / std::max(timestep, 1.0e-12F),
                    impl_->rigid_count,
                    0.01F * soft->options.node_radius};
                *static_cast<DeformableConstants *>(
                    entry->phase_constants.contents) = {
                    fluid_timestep, {}, soft->count, soft->bond_count, 1U,
                    soft->options.stretch_compliance, 0.0F,
                    soft->options.maximum_speed,
                    soft->options.node_radius, 0.0F, 0U, 0.0F,
                    soft->surface_count, 0U, 0U, 0.0F, 0.0F,
                    soft->options.shape_matching_stiffness,
                    soft->options.maximum_projection_fraction, 0U,
                    soft->options.spring_damping,
                    soft->options.constraint_velocity_response};
                const auto run = [&](id<MTLComputePipelineState> pipeline,
                                     NSUInteger threads) {
                    const NSUInteger group = std::min<NSUInteger>(
                        64U, pipeline.maxTotalThreadsPerThreadgroup);
                    dispatch_untimed(entry->table, pipeline,
                                     MTLSizeMake(threads, 1, 1),
                                     MTLSizeMake(group, 1, 1));
                };
                MetalTimingRecord *record = begin_timing(
                    encoder, timings,
                    MetalTimingStage::fluid_soft_body_contacts,
                    (6U + (impl_->rigid_count != 0U ? 1U : 0U)) *
                            options.solver_iterations +
                        3U);
                run(impl_->fluid_soft_sweep_pipeline, 1U);
                for (std::uint32_t pass = 0U;
                     pass < options.solver_iterations; ++pass) {
                    run(impl_->fluid_soft_pipeline,
                        fluid.options.capacity);
                    run(impl_->fluid_soft_solve_pipeline,
                        fluid.options.capacity);
                    run(impl_->fluid_soft_apply_pipeline, soft->count);
                    const NSUInteger soft_group = std::min<NSUInteger>(
                        soft->count,
                        std::min<NSUInteger>(
                            64U, impl_->soft_project_pipeline
                                     .maxTotalThreadsPerThreadgroup));
                    if (soft->count != 0U) {
                        dispatch_untimed(
                            entry->phase_table,
                            impl_->soft_project_pipeline,
                            MTLSizeMake(soft->count, 1, 1),
                            MTLSizeMake(soft_group, 1, 1));
                        dispatch_untimed(
                            entry->phase_table,
                            impl_->soft_apply_pipeline,
                            MTLSizeMake(soft->count, 1, 1),
                            MTLSizeMake(soft_group, 1, 1));
                    }
                    if (impl_->rigid_count != 0U)
                        run(impl_->fluid_soft_rigid_recover_pipeline,
                            soft->count);
                    run(impl_->fluid_soft_surface_pipeline,
                        soft->surface_count);
                }
                run(impl_->fluid_soft_recover_pipeline,
                    fluid.options.capacity);
                run(impl_->fluid_soft_copy_pipeline,
                    soft->surface_count);
                end_timing(encoder, timings, record);
            }
    };
    if (phase == MetalSystemPhase::frame_end) {
        for (std::uint32_t fluid_slot = 0U;
             fluid_slot < impl_->fluids.entries.size(); ++fluid_slot) {
            auto &fluid = impl_->fluids.entries[fluid_slot];
            if (!fluid) continue;
            const FluidId fluid_id{
                fluid_slot, impl_->fluids.generations[fluid_slot]};
            bool has_enabled_source = false;
            for (const auto &source : impl_->sources.entries)
                has_enabled_source = has_enabled_source ||
                    (source && source->options.enabled &&
                     source->fluid == fluid_id);
            if (has_enabled_source)
                dispatch_untimed(
                    fluid->table,
                    impl_->fluid_capture_spawn_baseline_pipeline,
                    MTLSizeMake(1U, 1U, 1U),
                    MTLSizeMake(1U, 1U, 1U));
        }
        for (auto &entry : impl_->sources.entries) {
            if (!entry || !entry->options.enabled) continue;
            dispatch(MetalTimingStage::fluid_spawn, entry->table,
                     impl_->source_pipeline, MTLSizeMake(64, 1, 1),
                     MTLSizeMake(64, 1, 1));
        }
        for (std::uint32_t fluid_slot = 0U;
             fluid_slot < impl_->fluids.entries.size(); ++fluid_slot) {
            auto &entry = impl_->fluids.entries[fluid_slot];
            if (!entry) continue;
            FluidResource &r = *entry;
            const FluidId fluid_id{
                fluid_slot, impl_->fluids.generations[fluid_slot]};
            r.count = std::min(
                static_cast<const ParticleMetadata *>(r.metadata.contents)
                    ->count,
                r.options.capacity);
            const std::uint32_t first_spawned = r.count;
            bool recover_spawn = false;
            for (const auto &source : impl_->sources.entries)
                recover_spawn = recover_spawn ||
                    (source && source->options.enabled &&
                     source->fluid == fluid_id);
            const float gravity_length = std::sqrt(
                gravity.x * gravity.x + gravity.y * gravity.y +
                gravity.z * gravity.z);
            const Vec3 up = gravity_length > 1.0e-6F
                ? Vec3{-gravity.x / gravity_length,
                       -gravity.y / gravity_length,
                       -gravity.z / gravity_length}
                : Vec3{0.0F, 1.0F, 0.0F};
            const std::uint32_t iterations =
                substeps * r.options.solver_iterations;
            const float fluid_timestep =
                timestep / static_cast<float>(iterations);
            *static_cast<FluidConstants *>(r.constants.contents) = {
                fluid_timestep, gravity, r.count, r.options.particle_radius,
                r.options.support_radius, r.options.repulsion,
                r.options.viscosity, r.options.normal_damping,
                r.options.velocity_damping, r.options.maximum_speed,
                r.options.maximum_pair_acceleration, r.options.rest_density,
                r.options.maximum_neighbors};
            const NSUInteger group = std::min<NSUInteger>(
                64U,
                impl_->fluid_forces_pipeline.maxTotalThreadsPerThreadgroup);
            for (std::uint32_t iteration = 0U; iteration < iterations;
                 ++iteration) {
                encode_fluid_sort(r);
                dispatch(MetalTimingStage::fluid_neighbor_forces, r.table,
                         impl_->fluid_forces_pipeline,
                         MTLSizeMake(r.options.capacity, 1, 1),
                         MTLSizeMake(group, 1, 1));
                encode_fluid_cloth(fluid_id, r, fluid_timestep, false);
                dispatch(MetalTimingStage::fluid_integration, r.table,
                         impl_->fluid_integrate_pipeline,
                         MTLSizeMake(r.options.capacity, 1, 1),
                         MTLSizeMake(group, 1, 1));
                encode_fluid_rope(fluid_id, r, fluid_timestep);
                encode_fluid_soft(fluid_id, r, fluid_timestep);
                encode_fluid_cloth(fluid_id, r, fluid_timestep, true);
                encode_rigid_contacts(
                    r.particles, r.count, r.options.particle_radius, 0.05F,
                    iteration == 0U ? r.rigid_first_constants
                                    : r.rigid_constants,
                    iteration == 0U ? r.rigid_first_table : r.rigid_table,
                    true, fluid_id,
                    r.stable_ids, r.contact_samples, r.contact_flags,
                    MetalTimingStage::fluid_static_contacts, fluid_timestep,
                    1.0F /
                        (r.options.rest_density *
                         (r.options.rest_particle_volume > 0.0F
                              ? r.options.rest_particle_volume
                              : 8.0F * r.options.particle_radius *
                                    r.options.particle_radius *
                                    r.options.particle_radius)),
                    iteration == 0U, r.options.maximum_speed, false, false,
                    first_spawned, recover_spawn && iteration == 0U,
                    r.options.support_radius, up, {}, 0.0F);
                // CUDA performs three current-skin position-only passes after
                // rigid contacts. A rigid boundary can otherwise push water
                // back through a neighboring soft face after the swept
                // fluid/soft solve has already finished.
                for (auto &coupling : impl_->fluid_soft.entries) {
                    if (!coupling || !coupling->enabled ||
                        coupling->options.fluid != fluid_id)
                        continue;
                    const NSUInteger recovery_group = std::min<NSUInteger>(
                        64U, impl_->fluid_soft_recover_current_pipeline
                                 .maxTotalThreadsPerThreadgroup);
                    for (std::uint32_t recovery = 0U; recovery < 3U;
                         ++recovery)
                        dispatch_untimed(
                            coupling->table,
                            impl_->fluid_soft_recover_current_pipeline,
                            MTLSizeMake(r.options.capacity, 1, 1),
                            MTLSizeMake(recovery_group, 1, 1));
                }
                for (auto &destroy : impl_->destroy_planes.entries) {
                    if (!destroy || !destroy->options.enabled ||
                        destroy->options.fluid != fluid_id)
                        continue;
                    dispatch(MetalTimingStage::fluid_outflow_compaction,
                             destroy->table, impl_->destroy_pipeline,
                             MTLSizeMake(64, 1, 1),
                             MTLSizeMake(64, 1, 1));
                }
            }
            ++r.revision;
        }
        encode_paint(true);
    }
    const auto smoke_boundary_moves = [&](SmokeResource &smoke) {
        for (const auto &coupling : impl_->smoke_soft.entries)
            if (coupling && coupling->enabled &&
                impl_->smokes.get(coupling->options.smoke) == &smoke)
                return true;
        for (const auto &coupling : impl_->smoke_cloth.entries)
            if (coupling && coupling->enabled &&
                impl_->smokes.get(coupling->options.smoke) == &smoke)
                return true;
        const auto *rigid_ids = static_cast<const RigidBodyId *>(
            impl_->rigid_ids.contents);
        const auto *rigid_parameters =
            static_cast<const RasterRigidParameters *>(
                impl_->rigid_parameters.contents);
        for (const auto &coupling : impl_->smoke_rigid.entries) {
            if (!coupling || !coupling->enabled ||
                impl_->smokes.get(coupling->options.smoke) != &smoke)
                continue;
            std::uint32_t dense = impl_->rigid_count;
            for (std::uint32_t body = 0U; body < impl_->rigid_count; ++body)
                if (rigid_ids[body] == coupling->options.body) {
                    dense = body;
                    break;
                }
            if (dense == impl_->rigid_count ||
                rigid_parameters[dense].motion !=
                    static_cast<std::uint32_t>(MotionType::static_body))
                return true;
        }
        return false;
    };
    if (phase != MetalSystemPhase::frame_end) {
    for (auto &entry : impl_->smokes.entries) {
        if (!entry) continue;
        SmokeResource &r = *entry;
        if (phase == MetalSystemPhase::frame_start) {
            r.raster_triangle_count = 0U;
            r.rigid_raster_triangle_count = 0U;
        }
        *static_cast<SmokeConstants *>(r.constants.contents) = {
            impl_->frame_timestep, gravity, r.options.emitter_center,
            r.options.initial_velocity,
            r.options.wind, r.options.emitter_half_extents,
            1U, r.options.lifetime,
            r.options.particle_radius, r.options.buoyancy, r.options.response,
            r.options.maximum_speed, r.options.rest_number_density,
            r.options.pressure_stiffness, r.options.viscosity,
            r.options.vorticity_confinement, r.options.capacity,
            r.options.grid_resolution, r.options.grid_vertical_resolution,
            r.options.grid_pressure_iterations, r.options.grid_minimum,
            r.options.grid_resolution == 0U
                ? 0.0F
                : r.options.grid_edge_length / r.options.grid_resolution,
            r.options.grid_kinematic_viscosity,
            r.options.grid_les_coefficient,
            r.options.grid_pressure_tolerance};
        if (phase == MetalSystemPhase::frame_start &&
            r.options.grid_resolution != 0U) {
            auto *grid_rigid_counts = static_cast<std::uint32_t *>(
                r.grid_rigid_body_count.contents);
            grid_rigid_counts[3] =
                !smoke_boundary_moves(r) && r.static_metadata_valid ? 1U : 0U;
            const NSUInteger n = r.options.grid_resolution;
            const NSUInteger height = r.options.grid_vertical_resolution;
            const NSUInteger cells = n * height * n;
            const NSUInteger faces =
                (n + 1U) * height * n + n * (height + 1U) * n +
                n * height * (n + 1U);
            const NSUInteger group = std::min<NSUInteger>(
                64U, impl_->smoke_grid_clear_pipeline
                         .maxTotalThreadsPerThreadgroup);
            dispatch(MetalTimingStage::smoke_grid, r.table,
                     impl_->smoke_grid_clear_pipeline,
                     MTLSizeMake(std::max(cells, faces), 1, 1),
                     MTLSizeMake(group, 1, 1));
            const NSUInteger raster_group = std::min<NSUInteger>(
                64U, impl_->smoke_grid_raster_clear_pipeline
                         .maxTotalThreadsPerThreadgroup);
            [encoder setArgumentTable:r.grid_raster_table];
            [encoder setComputePipelineState:
                         impl_->smoke_grid_raster_clear_pipeline];
            [encoder dispatchThreads:MTLSizeMake(std::max(cells, faces), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(raster_group, 1, 1)];
            barrier(encoder);
        }
    }
    // Raster deforming smoke boundaries before the Eulerian step.  Their
    // current surface buffers still describe the beginning of this substep,
    // matching the CUDA ordering where grid projection precedes deformable
    // integration.
    for (auto &entry : impl_->smoke_soft.entries) {
        if (!entry || !entry->enabled) continue;
        SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
        SoftResource *soft = impl_->soft_bodies.get(entry->options.soft_body);
        if (!smoke || !soft) continue;
        const auto &options = entry->options;
        auto *surface_constants = static_cast<SmokeSurfaceConstants *>(
            entry->constants.contents);
        const std::uint32_t raster_triangle_base =
            surface_constants->raster_triangle_base;
        *surface_constants = {
            timestep, soft->count, soft->surface_index_count, 1U,
            smoke->options.lifetime,
            options.contact_distance > 0.0F
                ? options.contact_distance
                : smoke->options.particle_radius + soft->options.node_radius,
            3.0F * smoke->options.particle_radius,
            smoke->options.rest_number_density, options.wind_drag,
            options.maximum_wind_acceleration, soft->options.maximum_speed,
            smoke->options.maximum_speed, options.enabled ? 1U : 0U,
            smoke->options.grid_resolution,
            smoke->options.grid_vertical_resolution,
            smoke->options.grid_resolution == 0U
                ? 0.0F
                : smoke->options.grid_edge_length /
                      smoke->options.grid_resolution,
            smoke->options.grid_minimum,
            smoke->options.grid_kinematic_viscosity,
            smoke->options.grid_les_coefficient,
            1U};
        surface_constants->raster_triangle_base = raster_triangle_base;
        if (phase == MetalSystemPhase::frame_start &&
            smoke->options.grid_resolution != 0U) {
            const NSUInteger triangles = soft->surface_index_count / 3U;
            surface_constants->raster_triangle_base =
                smoke->raster_triangle_count;
            smoke->raster_triangle_count +=
                static_cast<std::uint32_t>(triangles);
            const NSUInteger group = std::min<NSUInteger>(
                triangles,
                std::min<NSUInteger>(
                    64U, impl_->smoke_surface_grid_raster_pipeline
                             .maxTotalThreadsPerThreadgroup));
            if (triangles != 0U) {
                dispatch(MetalTimingStage::smoke_grid, entry->table,
                         impl_->smoke_surface_grid_raster_pipeline,
                         MTLSizeMake(triangles, 1, 1),
                         MTLSizeMake(group, 1, 1));
            }
        }
        if (smoke->options.grid_resolution == 0U &&
            options.wind_drag != 0.0F)
            dispatch(MetalTimingStage::soft_body_constraints, entry->table,
                     impl_->smoke_surface_pipeline,
                     MTLSizeMake(soft->count, 1, 1),
                     MTLSizeMake(std::min<NSUInteger>(
                         64U, impl_->smoke_surface_pipeline
                                  .maxTotalThreadsPerThreadgroup),
                                 1, 1));
    }
    for (auto &entry : impl_->smoke_cloth.entries) {
        if (!entry || !entry->enabled) continue;
        SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
        ClothResource *cloth = impl_->cloths.get(entry->options.cloth);
        if (!smoke || !cloth) continue;
        const auto &options = entry->options;
        auto *surface_constants = static_cast<SmokeSurfaceConstants *>(
            entry->constants.contents);
        const std::uint32_t raster_triangle_base =
            surface_constants->raster_triangle_base;
        *surface_constants = {
            timestep, cloth->count, cloth->triangle_index_count, 0U,
            smoke->options.lifetime,
            options.contact_distance > 0.0F
                ? options.contact_distance
                : smoke->options.particle_radius + cloth->options.thickness,
            3.0F * smoke->options.particle_radius,
            smoke->options.rest_number_density, options.wind_drag,
            options.maximum_wind_acceleration, 20.0F,
            smoke->options.maximum_speed, options.enabled ? 1U : 0U,
            smoke->options.grid_resolution,
            smoke->options.grid_vertical_resolution,
            smoke->options.grid_resolution == 0U
                ? 0.0F
                : smoke->options.grid_edge_length /
                      smoke->options.grid_resolution,
            smoke->options.grid_minimum,
            smoke->options.grid_kinematic_viscosity,
            smoke->options.grid_les_coefficient,
            1U};
        surface_constants->raster_triangle_base = raster_triangle_base;
        if (phase == MetalSystemPhase::frame_start &&
            smoke->options.grid_resolution != 0U) {
            const NSUInteger triangles = cloth->triangle_index_count / 3U;
            surface_constants->raster_triangle_base =
                smoke->raster_triangle_count;
            smoke->raster_triangle_count +=
                static_cast<std::uint32_t>(triangles);
            const NSUInteger group = std::min<NSUInteger>(
                triangles,
                std::min<NSUInteger>(
                    64U, impl_->smoke_surface_grid_raster_pipeline
                             .maxTotalThreadsPerThreadgroup));
            if (triangles != 0U) {
                dispatch(MetalTimingStage::smoke_grid, entry->table,
                         impl_->smoke_surface_grid_raster_pipeline,
                         MTLSizeMake(triangles, 1, 1),
                         MTLSizeMake(group, 1, 1));
            }
        }
        if (smoke->options.grid_resolution == 0U &&
            options.wind_drag != 0.0F)
            dispatch(MetalTimingStage::cloth_constraints, entry->table,
                     impl_->smoke_surface_pipeline,
                     MTLSizeMake(cloth->count, 1, 1),
                     MTLSizeMake(std::min<NSUInteger>(
                         64U, impl_->smoke_surface_pipeline
                                  .maxTotalThreadsPerThreadgroup),
                                 1, 1));
    }
    for (std::uint32_t smoke_slot = 0U;
         smoke_slot < impl_->smokes.entries.size(); ++smoke_slot) {
        auto &entry = impl_->smokes.entries[smoke_slot];
        if (!entry) continue;
        SmokeResource &r = *entry;
        const SmokeId smoke_id{
            smoke_slot, impl_->smokes.generations[smoke_slot]};
        auto *grid_rigid_bodies = static_cast<SmokeRigidRasterEntry *>(
            r.grid_rigid_bodies.contents);
        std::uint32_t grid_rigid_body_count = 0U;
        const auto *rigid_ids = static_cast<const RigidBodyId *>(
            impl_->rigid_ids.contents);
        const auto *rigid_parameters =
            static_cast<const RasterRigidParameters *>(
                impl_->rigid_parameters.contents);
        const auto *mesh_infos = static_cast<const RasterTriangleMeshInfo *>(
            impl_->mesh_infos.contents);
        if (phase == MetalSystemPhase::frame_start &&
            r.options.grid_resolution != 0U) {
            r.rigid_raster_triangle_count = 0U;
            for (const auto &coupling : impl_->smoke_rigid.entries) {
                if (!coupling || !coupling->enabled ||
                    coupling->options.smoke != smoke_id)
                    continue;
                if (grid_rigid_body_count >=
                    impl_->options.smoke_rigid_coupling_capacity)
                    break;
                std::uint32_t body_index = impl_->rigid_count;
                for (std::uint32_t body = 0U; body < impl_->rigid_count;
                     ++body) {
                    if (rigid_ids[body] == coupling->options.body) {
                        body_index = body;
                        break;
                    }
                }
                if (body_index == impl_->rigid_count) continue;
                const std::uint32_t triangle_count =
                    mesh_infos[rigid_parameters[body_index].mesh_index]
                        .index_count /
                    3U;
                grid_rigid_bodies[grid_rigid_body_count++] = {
                    coupling->options.body, r.raster_triangle_count};
                r.raster_triangle_count += triangle_count;
                r.rigid_raster_triangle_count += triangle_count;
            }
        }
        auto *grid_rigid_counts = static_cast<std::uint32_t *>(
            r.grid_rigid_body_count.contents);
        if (phase == MetalSystemPhase::frame_start &&
            r.options.grid_resolution != 0U) {
            grid_rigid_counts[0] = grid_rigid_body_count;
            grid_rigid_counts[1] = impl_->rigid_count;
            grid_rigid_counts[2] = r.rigid_raster_triangle_count;
        }
        const auto bind_rigid_grid = [&](id<MTL4ArgumentTable> table) {
            impl_->bind(table, 21U, impl_->rigid_ids);
            impl_->bind(table, 22U, impl_->rigid_states);
            impl_->bind(table, 23U, impl_->rigid_parameters);
            impl_->bind(table, 24U, impl_->mesh_vertices);
            impl_->bind(table, 25U, impl_->mesh_indices);
            impl_->bind(table, 26U, impl_->mesh_infos);
        };
        bind_rigid_grid(r.table);
        bind_rigid_grid(r.advection_table);
        impl_->bind(r.grid_raster_table, 7U, impl_->rigid_ids);
        impl_->bind(r.grid_raster_table, 8U, impl_->rigid_states);
        impl_->bind(r.grid_raster_table, 9U, impl_->rigid_parameters);
        impl_->bind(r.grid_raster_table, 10U, impl_->mesh_vertices);
        impl_->bind(r.grid_raster_table, 11U, impl_->mesh_indices);
        impl_->bind(r.grid_raster_table, 12U, impl_->mesh_infos);
        if (phase == MetalSystemPhase::frame_start &&
            r.options.grid_resolution != 0U) {
            const auto dispatch_triangles =
                [&](id<MTL4ArgumentTable> table,
                    id<MTLComputePipelineState> pipeline,
                    NSUInteger triangles) {
                    if (triangles == 0U) return;
                    const NSUInteger group = std::min<NSUInteger>(
                        triangles,
                        std::min<NSUInteger>(
                            64U, pipeline.maxTotalThreadsPerThreadgroup));
                    dispatch(MetalTimingStage::smoke_grid, table, pipeline,
                             MTLSizeMake(triangles, 1, 1),
                             MTLSizeMake(group, 1, 1));
                };
            dispatch_triangles(
                r.grid_raster_table,
                impl_->smoke_grid_raster_rigid_pipeline,
                r.rigid_raster_triangle_count);
            const bool preserve_static_metadata = grid_rigid_counts[3] != 0U;
            if (!preserve_static_metadata) {
                for (auto &surface : impl_->smoke_soft.entries) {
                    if (!surface || !surface->enabled ||
                        surface->options.smoke != smoke_id)
                        continue;
                    SoftResource *soft =
                        impl_->soft_bodies.get(surface->options.soft_body);
                    if (!soft) continue;
                    dispatch_triangles(
                        surface->table,
                        impl_->smoke_surface_grid_select_pipeline,
                        soft->surface_index_count / 3U);
                }
                for (auto &surface : impl_->smoke_cloth.entries) {
                    if (!surface || !surface->enabled ||
                        surface->options.smoke != smoke_id)
                        continue;
                    ClothResource *cloth =
                        impl_->cloths.get(surface->options.cloth);
                    if (!cloth) continue;
                    dispatch_triangles(
                        surface->table,
                        impl_->smoke_surface_grid_select_pipeline,
                        cloth->triangle_index_count / 3U);
                }
                dispatch_triangles(
                    r.grid_raster_table,
                    impl_->smoke_grid_select_rigid_pipeline,
                    r.rigid_raster_triangle_count);
            }
            for (auto &surface : impl_->smoke_soft.entries) {
                if (!surface || !surface->enabled ||
                    surface->options.smoke != smoke_id)
                    continue;
                SoftResource *soft =
                    impl_->soft_bodies.get(surface->options.soft_body);
                if (!soft) continue;
                dispatch_triangles(
                    surface->table,
                    impl_->smoke_surface_grid_resolve_pipeline,
                    soft->surface_index_count / 3U);
            }
            for (auto &surface : impl_->smoke_cloth.entries) {
                if (!surface || !surface->enabled ||
                    surface->options.smoke != smoke_id)
                    continue;
                ClothResource *cloth =
                    impl_->cloths.get(surface->options.cloth);
                if (!cloth) continue;
                dispatch_triangles(
                    surface->table,
                    impl_->smoke_surface_grid_resolve_pipeline,
                    cloth->triangle_index_count / 3U);
            }
            dispatch_triangles(
                r.grid_raster_table,
                impl_->smoke_grid_resolve_rigid_pipeline,
                r.rigid_raster_triangle_count);
            r.static_metadata_valid = !smoke_boundary_moves(r);
            encode_smoke_grid(r);
        }
    }
    if (phase == MetalSystemPhase::frame_start) return;
    // Apply the newly projected Eulerian field before deformable prediction,
    // matching CUDA's smoke-grid/deformable phase order. The force kernel
    // accumulates deterministically into fixed per-node buffers.
    for (auto &entry : impl_->smoke_soft.entries) {
        if (!entry || !entry->enabled || entry->options.wind_drag == 0.0F)
            continue;
        SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
        SoftResource *soft =
            impl_->soft_bodies.get(entry->options.soft_body);
        if (!smoke || !soft || smoke->options.grid_resolution == 0U) continue;
        const NSUInteger group = std::min<NSUInteger>(
            64U, impl_->smoke_surface_grid_force_pipeline
                     .maxTotalThreadsPerThreadgroup);
        dispatch(MetalTimingStage::soft_body_constraints, entry->table,
                 impl_->smoke_surface_grid_force_pipeline,
                 MTLSizeMake(soft->count, 1, 1),
                 MTLSizeMake(group, 1, 1));
    }
    for (auto &entry : impl_->smoke_cloth.entries) {
        if (!entry || !entry->enabled || entry->options.wind_drag == 0.0F)
            continue;
        SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
        ClothResource *cloth = impl_->cloths.get(entry->options.cloth);
        if (!smoke || !cloth || smoke->options.grid_resolution == 0U)
            continue;
        const NSUInteger group = std::min<NSUInteger>(
            64U, impl_->smoke_surface_grid_force_pipeline
                     .maxTotalThreadsPerThreadgroup);
        dispatch(MetalTimingStage::cloth_constraints, entry->table,
                 impl_->smoke_surface_grid_force_pipeline,
                 MTLSizeMake(cloth->count, 1, 1),
                 MTLSizeMake(group, 1, 1));
    }
    for (auto &entry : impl_->cloths.entries) {
        if (!entry) continue;
        ClothResource &r = *entry;
        *static_cast<DeformableConstants *>(r.constants.contents) = {
            timestep, gravity, r.count, r.bond_count,
            r.options.solver_iterations, r.options.stretch_compliance,
            r.options.velocity_damping, 100.0F, r.options.thickness,
            r.options.break_strain, r.options.fracture_persistence_substeps,
            r.options.impact_break_impulse, r.surface_count,
            r.triangle_index_count,
            r.options.preserve_volume ? 1U : 0U,
            r.options.target_volume, r.options.volume_compliance, 0.0F, 0.0F,
            0U, 0.0F, 1.0F};
        dispatch(MetalTimingStage::cloth_prediction, r.table,
                 impl_->cloth_prediction_pipeline,
                 MTLSizeMake(r.count, 1, 1),
                 MTLSizeMake(std::min<NSUInteger>(
                     64U, impl_->cloth_prediction_pipeline
                              .maxTotalThreadsPerThreadgroup),
                             1, 1));
        encode_cloth_constraints(r);
        ++r.revision;
    }
    for (auto &entry : impl_->soft_bodies.entries) {
        if (!entry) continue;
        SoftResource &r = *entry;
        *static_cast<DeformableConstants *>(r.constants.contents) = {
            timestep, gravity, r.count, r.bond_count,
            r.options.solver_iterations, r.options.stretch_compliance,
            r.options.velocity_damping, r.options.maximum_speed,
            r.options.node_radius, 0.0F, 0U, 0.0F, r.surface_count, 0U, 0U,
            0.0F, 0.0F,
            r.options.shape_matching_stiffness,
            r.options.maximum_projection_fraction, 0U,
            r.options.spring_damping,
            r.options.constraint_velocity_response};
        dispatch(MetalTimingStage::soft_body_prediction, r.table,
                 impl_->soft_prediction_pipeline,
                 MTLSizeMake(r.count, 1, 1),
                 MTLSizeMake(std::min<NSUInteger>(
                     64U, impl_->soft_prediction_pipeline
                              .maxTotalThreadsPerThreadgroup),
                             1, 1));
        encode_soft_constraints(r);
        ++r.revision;
    }
    for (auto &entry : impl_->rope_cloth.entries) {
        if (!entry || !entry->enabled) continue;
        dispatch(MetalTimingStage::rope_solve, entry->table,
                 impl_->rope_cloth_sample_pipeline,
                 MTLSizeMake(2, 1, 1), MTLSizeMake(2, 1, 1));
    }
    for (auto &entry : impl_->rope_soft.entries) {
        if (!entry || !entry->enabled) continue;
        RopeResource *rope = impl_->ropes.get(entry->options.rope);
        SoftResource *soft = impl_->soft_bodies.get(entry->options.soft_body);
        if (!rope || !soft) continue;
        const auto &options = entry->options;
        auto *constants = static_cast<RopeSoftConstants *>(
            entry->constants.contents);
        constants->timestep = timestep;
        constants->rope_count = rope->count;
        constants->soft_count = soft->count;
        constants->surface_count = soft->surface_count;
        constants->surface_index_count = soft->surface_index_count;
        constants->contact_distance = options.contact_distance > 0.0F
                                          ? options.contact_distance
                                          : rope->options.radius;
        constants->friction = options.friction;
        constants->maximum_soft_acceleration =
            options.maximum_soft_body_acceleration;
        constants->maximum_rope_speed = rope->options.maximum_speed;
        constants->maximum_soft_speed = soft->options.maximum_speed;
        constants->node_radius = soft->options.node_radius;
        constants->anchor_support_radius_scale =
            options.anchor_support_radius_scale;
        constants->anchor_contact_support_radius_scale =
            options.anchor_contact_support_radius_scale;
        constants->orientation = soft->surface_orientation;
        constants->attach_first = options.attach_first ? 1U : 0U;
        constants->attach_last = options.attach_last ? 1U : 0U;
        constants->enabled = 1U;
        constants->frame_inverse_timestep = impl_->frame_inverse_timestep;
        if (options.attach_first || options.attach_last)
            dispatch(MetalTimingStage::rope_solve, entry->table,
                     impl_->rope_soft_sample_pipeline,
                     MTLSizeMake(2, 1, 1), MTLSizeMake(2, 1, 1));
        dispatch(MetalTimingStage::rope_solve, entry->table,
                 impl_->rope_soft_pack_pipeline,
                 MTLSizeMake(64, 1, 1), MTLSizeMake(64, 1, 1));
    }
    for (auto &entry : impl_->ropes.entries) {
        if (!entry) continue;
        RopeResource &r = *entry;
        std::uint32_t packed_target = 0U;
        for (const auto &coupling : impl_->rope_soft.entries) {
            if (!coupling || !coupling->enabled ||
                impl_->ropes.get(coupling->options.rope) != &r)
                continue;
            if (packed_target >= 2U) break;
            impl_->bind(r.table, 27U + packed_target,
                        coupling->packed_state);
            ++packed_target;
        }
        while (packed_target < 2U) {
            impl_->bind(r.table, 27U + packed_target,
                        r.empty_soft_target);
            ++packed_target;
        }
        for (auto &coupling : impl_->smoke_rope.entries) {
            if (!coupling || !coupling->enabled ||
                coupling->options.wind_drag == 0.0F ||
                impl_->ropes.get(coupling->options.rope) != &r)
                continue;
            SmokeResource *smoke = impl_->smokes.get(coupling->options.smoke);
            if (!smoke) continue;
            bool skip_first = r.options.first.enabled;
            bool skip_last = r.options.last.enabled;
            for (const auto &anchor : impl_->rope_soft.entries) {
                if (!anchor || !anchor->enabled ||
                    anchor->options.rope != coupling->options.rope)
                    continue;
                skip_first |= anchor->options.attach_first;
                skip_last |= anchor->options.attach_last;
            }
            for (const auto &anchor : impl_->rope_cloth.entries) {
                if (!anchor || !anchor->enabled ||
                    anchor->options.rope != coupling->options.rope)
                    continue;
                skip_first |= anchor->options.first_vertex != UINT32_MAX;
                skip_last |= anchor->options.last_vertex != UINT32_MAX;
            }
            const auto &options = coupling->options;
            *static_cast<SmokeRopeConstants *>(
                coupling->constants.contents) = {
                timestep, smoke->options.capacity, r.count,
                smoke->options.lifetime,
                options.contact_distance > 0.0F
                    ? options.contact_distance
                    : smoke->options.particle_radius + r.options.radius,
                3.0F * smoke->options.particle_radius,
                smoke->options.rest_number_density, options.wind_drag,
                options.maximum_wind_acceleration, r.options.maximum_speed,
                skip_first ? 1U : 0U, skip_last ? 1U : 0U,
                options.enabled ? 1U : 0U,
                smoke->options.grid_resolution,
                smoke->options.grid_vertical_resolution,
                smoke->options.grid_resolution == 0U
                    ? 0.0F
                    : smoke->options.grid_edge_length /
                          smoke->options.grid_resolution,
                smoke->options.grid_minimum};
            dispatch(MetalTimingStage::rope_solve, coupling->table,
                     impl_->smoke_rope_wind_pipeline,
                     MTLSizeMake(r.count, 1, 1),
                     MTLSizeMake(std::min<NSUInteger>(
                         r.count,
                         std::min<NSUInteger>(
                             64U, impl_->smoke_rope_wind_pipeline
                                      .maxTotalThreadsPerThreadgroup)),
                                 1, 1));
        }
        *static_cast<DeformableConstants *>(r.constants.contents) = {
            timestep, gravity, r.count, r.count - 1U,
            r.options.solver_iterations, r.options.stretch_compliance,
            r.options.velocity_damping, r.options.maximum_speed,
            r.options.radius, 0.0F, 0U, 0.0F, 0U, 0U, 0U, 0.0F, 0.0F,
            0.0F, 0.0F,
            r.options.self_collision ? 1U : 0U, r.options.friction, 1.0F};
        auto *attachments = static_cast<RopeAttachmentConstants *>(
            r.attachment_constants.contents);
        attachments->rigid_count = impl_->rigid_count;
        attachments->first_soft = 0U;
        attachments->last_soft = 0U;
        for (const auto &coupling : impl_->rope_soft.entries) {
            if (!coupling || !coupling->enabled ||
                impl_->ropes.get(coupling->options.rope) != &r)
                continue;
            attachments->first_soft |=
                coupling->options.attach_first ? 1U : 0U;
            attachments->last_soft |=
                coupling->options.attach_last ? 1U : 0U;
        }
        for (const auto &coupling : impl_->rope_cloth.entries) {
            if (!coupling || !coupling->enabled ||
                impl_->ropes.get(coupling->options.rope) != &r)
                continue;
            attachments->first_soft |=
                coupling->options.first_vertex != UINT32_MAX ? 1U : 0U;
            attachments->last_soft |=
                coupling->options.last_vertex != UINT32_MAX ? 1U : 0U;
        }
        if (impl_->rigid_count != 0U) {
            impl_->bind(r.table, 7U, impl_->rigid_ids);
            impl_->bind(r.table, 8U, impl_->rigid_states);
            impl_->bind(r.table, 16U, impl_->rigid_parameters);
            impl_->bind(r.table, 19U, impl_->mesh_vertices);
            impl_->bind(r.table, 20U, impl_->mesh_indices);
            impl_->bind(r.table, 21U, impl_->mesh_infos);
            impl_->bind(r.table, 22U, impl_->rigid_previous_states);
            impl_->bind(r.table, 30U, impl_->mesh_solid_planes);
        } else {
            impl_->bind(r.table, 7U, r.attachment_constants);
            impl_->bind(r.table, 8U, r.attachment_constants);
            impl_->bind(r.table, 16U, r.attachment_constants);
            impl_->bind(r.table, 19U, r.attachment_constants);
            impl_->bind(r.table, 20U, r.attachment_constants);
            impl_->bind(r.table, 21U, r.attachment_constants);
            impl_->bind(r.table, 22U, r.attachment_constants);
            impl_->bind(r.table, 30U, r.attachment_constants);
        }
        encode_rope_solve(r);
        ++r.revision;
    }

    for (auto &entry : impl_->cloths.entries)
        if (entry)
            encode_rigid_contacts(
                entry->particles, entry->count, entry->options.thickness,
                entry->options.contact_friction, entry->rigid_constants,
                entry->rigid_table, false, {}, entry->rigid_constants,
                entry->rigid_constants, entry->rigid_constants,
                MetalTimingStage::cloth_contacts, timestep, 0.0F, true,
                20.0F, false, false, 0U, false, 0.0F,
                {0.0F, 1.0F, 0.0F}, {}, 0.0F);
    // CUDA stamps rigid-cloth paint from the detected contact before the
    // conservative body/triangle correction separates the geometry.
    encode_paint(false);
    for (auto &entry : impl_->cloths.entries) {
        if (!entry || impl_->rigid_count == 0U) continue;
        const bool fracture_enabled = entry->options.break_strain > 0.0F ||
                                      entry->options.impact_break_impulse > 0.0F;
        *static_cast<ClothRigidConstants *>(
            entry->surface_rigid_constants.contents) = {
            timestep, entry->count, entry->triangle_index_count,
            entry->surface_count, entry->options.thickness,
            entry->options.contact_friction, impl_->rigid_count,
            fracture_enabled ? 1U : 0U};
        impl_->bind(entry->surface_rigid_table, 4U, impl_->rigid_states);
        impl_->bind(entry->surface_rigid_table, 5U,
                    impl_->rigid_parameters);
        impl_->bind(entry->surface_rigid_table, 6U, impl_->mesh_vertices);
        impl_->bind(entry->surface_rigid_table, 7U, impl_->mesh_infos);
        impl_->bind(entry->surface_rigid_table, 8U,
                    impl_->rigid_previous_states);
        dispatch(MetalTimingStage::cloth_contacts,
                 entry->surface_rigid_table,
                 impl_->cloth_rigid_surface_pipeline,
                 MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    }
    const auto soft_cloth_coupled = [&](const ClothResource *cloth) {
        for (const auto &coupling : impl_->soft_cloth.entries)
            if (coupling && coupling->enabled &&
                impl_->cloths.get(coupling->options.cloth) == cloth)
                return true;
        return false;
    };
    for (auto &entry : impl_->cloths.entries) {
        if (!entry ||
            soft_cloth_coupled(entry.get()) ||
            (entry->options.break_strain <= 0.0F &&
             entry->options.impact_break_impulse <= 0.0F))
            continue;
        dispatch(MetalTimingStage::cloth_contacts, entry->table,
                 impl_->cloth_limit_strain_pipeline,
                 MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
        const NSUInteger group = std::min<NSUInteger>(
            entry->bond_count,
            std::min<NSUInteger>(
                64U,
                impl_->cloth_damage_pipeline.maxTotalThreadsPerThreadgroup));
        dispatch(MetalTimingStage::cloth_contacts, entry->table,
                 impl_->cloth_damage_pipeline,
                 MTLSizeMake(entry->bond_count, 1, 1),
                 MTLSizeMake(group, 1, 1));
    }

    const auto encode_couplings = [&](auto &slots, MetalTimingStage stage) {
        for (auto &entry : slots.entries) {
            if (!entry || !entry->enabled) continue;
            auto *constants =
                static_cast<CouplingConstants *>(entry->constants.contents);
            constants->timestep = timestep;
            constants->damping = impl_->frame_inverse_timestep;
            dispatch(stage, entry->table, impl_->coupling_pipeline,
                     MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
        }
    };
    encode_couplings(impl_->rope_cloth, MetalTimingStage::cloth_contacts);
    }
    const auto encode_smoke_contacts = [&] {
    for (auto &entry : impl_->smoke_rope.entries) {
        if (!entry || !entry->enabled) continue;
        SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
        RopeResource *rope = impl_->ropes.get(entry->options.rope);
        if (!smoke || !rope) continue;
        bool skip_first = rope->options.first.enabled;
        bool skip_last = rope->options.last.enabled;
        for (const auto &anchor : impl_->rope_soft.entries) {
            if (!anchor || !anchor->enabled ||
                anchor->options.rope != entry->options.rope)
                continue;
            skip_first |= anchor->options.attach_first;
            skip_last |= anchor->options.attach_last;
        }
        for (const auto &anchor : impl_->rope_cloth.entries) {
            if (!anchor || !anchor->enabled ||
                anchor->options.rope != entry->options.rope)
                continue;
            skip_first |= anchor->options.first_vertex != UINT32_MAX;
            skip_last |= anchor->options.last_vertex != UINT32_MAX;
        }
        const auto &options = entry->options;
        *static_cast<SmokeRopeConstants *>(
            entry->phase_constants.contents) = {
            impl_->frame_timestep,
            smoke->options.capacity,
            rope->count,
            smoke->options.lifetime,
            options.contact_distance > 0.0F
                ? options.contact_distance
                : smoke->options.particle_radius + rope->options.radius,
            3.0F * smoke->options.particle_radius,
            smoke->options.rest_number_density,
            options.wind_drag,
            options.maximum_wind_acceleration,
            rope->options.maximum_speed,
            skip_first ? 1U : 0U,
            skip_last ? 1U : 0U,
            options.enabled ? 1U : 0U,
            smoke->options.grid_resolution,
            smoke->options.grid_vertical_resolution,
            smoke->options.grid_resolution == 0U
                ? 0.0F
                : smoke->options.grid_edge_length /
                      smoke->options.grid_resolution,
            smoke->options.grid_minimum};
        dispatch(MetalTimingStage::smoke_advection, entry->phase_table,
                 impl_->smoke_rope_pipeline,
                 MTLSizeMake(smoke->options.capacity, 1, 1),
                 MTLSizeMake(std::min<NSUInteger>(
                     64U,
                     impl_->smoke_rope_pipeline
                         .maxTotalThreadsPerThreadgroup),
                             1, 1));
    }
    for (auto &entry : impl_->smoke_soft.entries) {
        if (!entry || !entry->enabled) continue;
        SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
        SoftResource *soft = impl_->soft_bodies.get(entry->options.soft_body);
        if (!smoke || !soft) continue;
        const auto &options = entry->options;
        *static_cast<SmokeSurfaceConstants *>(
            entry->phase_constants.contents) = {
            impl_->frame_timestep,
            soft->count,
            soft->surface_index_count,
            1U,
            smoke->options.lifetime,
            options.contact_distance > 0.0F
                ? options.contact_distance
                : smoke->options.particle_radius + soft->options.node_radius,
            3.0F * smoke->options.particle_radius,
            smoke->options.rest_number_density,
            options.wind_drag,
            options.maximum_wind_acceleration,
            soft->options.maximum_speed,
            smoke->options.maximum_speed,
            options.enabled ? 1U : 0U,
            smoke->options.grid_resolution,
            smoke->options.grid_vertical_resolution,
            smoke->options.grid_resolution == 0U
                ? 0.0F
                : smoke->options.grid_edge_length /
                      smoke->options.grid_resolution,
            smoke->options.grid_minimum,
            smoke->options.grid_kinematic_viscosity,
            smoke->options.grid_les_coefficient,
            2U};
        dispatch(MetalTimingStage::smoke_advection, entry->phase_table,
                 impl_->smoke_surface_pipeline,
                 MTLSizeMake(smoke->options.capacity, 1, 1),
                 MTLSizeMake(std::min<NSUInteger>(
                     64U, impl_->smoke_surface_pipeline
                              .maxTotalThreadsPerThreadgroup),
                             1, 1));
    }
    for (auto &entry : impl_->smoke_cloth.entries) {
        if (!entry || !entry->enabled) continue;
        SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
        ClothResource *cloth = impl_->cloths.get(entry->options.cloth);
        if (!smoke || !cloth) continue;
        const auto &options = entry->options;
        *static_cast<SmokeSurfaceConstants *>(
            entry->phase_constants.contents) = {
            impl_->frame_timestep,
            cloth->count,
            cloth->triangle_index_count,
            0U,
            smoke->options.lifetime,
            options.contact_distance > 0.0F
                ? options.contact_distance
                : smoke->options.particle_radius + cloth->options.thickness,
            3.0F * smoke->options.particle_radius,
            smoke->options.rest_number_density,
            options.wind_drag,
            options.maximum_wind_acceleration,
            20.0F,
            smoke->options.maximum_speed,
            options.enabled ? 1U : 0U,
            smoke->options.grid_resolution,
            smoke->options.grid_vertical_resolution,
            smoke->options.grid_resolution == 0U
                ? 0.0F
                : smoke->options.grid_edge_length /
                      smoke->options.grid_resolution,
            smoke->options.grid_minimum,
            smoke->options.grid_kinematic_viscosity,
            smoke->options.grid_les_coefficient,
            2U};
        dispatch(MetalTimingStage::smoke_advection, entry->phase_table,
                 impl_->smoke_surface_pipeline,
                 MTLSizeMake(smoke->options.capacity, 1, 1),
                 MTLSizeMake(std::min<NSUInteger>(
                     64U, impl_->smoke_surface_pipeline
                              .maxTotalThreadsPerThreadgroup),
                             1, 1));
    }
    };
    if (phase != MetalSystemPhase::frame_end) {
    std::uint32_t soft_cloth_iterations = 0U;
    for (auto &entry : impl_->soft_cloth.entries) {
        if (!entry || !entry->enabled) continue;
        SoftResource *soft = impl_->soft_bodies.get(entry->options.soft_body);
        ClothResource *cloth = impl_->cloths.get(entry->options.cloth);
        if (!soft || !cloth) continue;
        const auto &options = entry->options;
        soft_cloth_iterations = std::max(
            soft_cloth_iterations, options.solver_iterations);
        *static_cast<SoftClothConstants *>(entry->constants.contents) = {
            timestep,
            soft->count,
            soft->surface_count,
            cloth->count,
            cloth->surface_count,
            cloth->triangle_index_count,
            options.contact_distance > 0.0F
                ? options.contact_distance
                : soft->options.node_radius + cloth->options.thickness,
            options.friction,
            soft->options.maximum_speed,
            options.solver_iterations,
            options.enabled ? 1U : 0U};
    }
    const auto soft_cloth_run =
        [&](id<MTL4ArgumentTable> table,
            id<MTLComputePipelineState> pipeline, NSUInteger threads) {
            if (threads == 0U) return;
            const NSUInteger group = std::min<NSUInteger>(
                threads,
                std::min<NSUInteger>(
                    64U, pipeline.maxTotalThreadsPerThreadgroup));
            dispatch_untimed(table, pipeline,
                             MTLSizeMake(threads, 1, 1),
                             MTLSizeMake(group, 1, 1));
        };
    const auto soft_body_cloth_coupled = [&](const SoftResource *soft) {
        for (const auto &coupling : impl_->soft_cloth.entries)
            if (coupling && coupling->enabled &&
                impl_->soft_bodies.get(coupling->options.soft_body) == soft)
                return true;
        return false;
    };
    const auto cloth_soft_body_coupled = [&](const ClothResource *cloth) {
        for (const auto &coupling : impl_->soft_cloth.entries)
            if (coupling && coupling->enabled &&
                impl_->cloths.get(coupling->options.cloth) == cloth)
                return true;
        return false;
    };
    MetalTimingRecord *soft_cloth_record = nullptr;
    if (soft_cloth_iterations != 0U)
        soft_cloth_record = begin_timing(
            encoder, timings, MetalTimingStage::soft_body_cloth_contacts,
            soft_cloth_iterations);
    for (std::uint32_t pass = 0U;
         pass < soft_cloth_iterations; ++pass) {
        for (auto &entry : impl_->soft_cloth.entries) {
            if (!entry || !entry->enabled) continue;
            SoftResource *soft =
                impl_->soft_bodies.get(entry->options.soft_body);
            ClothResource *cloth =
                impl_->cloths.get(entry->options.cloth);
            if (!soft || !cloth) continue;
            soft_cloth_run(entry->table,
                           impl_->soft_cloth_clear_pipeline,
                           cloth->count);
            soft_cloth_run(entry->table, impl_->soft_cloth_pipeline,
                           soft->count);
            soft_cloth_run(entry->table,
                           impl_->soft_cloth_apply_soft_pipeline,
                           soft->count);
            soft_cloth_run(entry->table, impl_->soft_cloth_apply_pipeline,
                           cloth->count);
            soft_cloth_run(entry->table,
                           impl_->soft_cloth_surface_pipeline,
                           cloth->surface_count);
        }
        for (auto &entry : impl_->cloths.entries) {
            if (!entry || !cloth_soft_body_coupled(entry.get())) continue;
            ClothResource &cloth = *entry;
            if (pass == 0U &&
                (cloth.options.break_strain > 0.0F ||
                 cloth.options.impact_break_impulse > 0.0F))
                soft_cloth_run(cloth.table, impl_->cloth_damage_pipeline,
                               cloth.bond_count);
            if (pass + 1U < soft_cloth_iterations) {
                soft_cloth_run(cloth.table, impl_->cloth_project_pipeline,
                               cloth.count);
                soft_cloth_run(cloth.table, impl_->cloth_apply_pipeline,
                               cloth.count);
                if (cloth.options.preserve_volume)
                    dispatch_untimed(
                        cloth.table, impl_->cloth_pipeline,
                        MTLSizeMake(128U, 1, 1),
                        MTLSizeMake(128U, 1, 1));
            }
            if (cloth.surface_count != 0U) {
                dispatch_untimed(
                    cloth.table, impl_->cloth_limit_strain_pipeline,
                    MTLSizeMake(1U, 1, 1), MTLSizeMake(1U, 1, 1));
                soft_cloth_run(cloth.table,
                               impl_->cloth_surface_update_pipeline,
                               cloth.surface_count);
            }
        }
        if (pass + 1U < soft_cloth_iterations)
            for (auto &entry : impl_->soft_bodies.entries) {
                if (!entry ||
                    !soft_body_cloth_coupled(entry.get()))
                    continue;
                soft_cloth_run(entry->table, impl_->soft_project_pipeline,
                               entry->count);
                soft_cloth_run(entry->table, impl_->soft_apply_pipeline,
                               entry->count);
            }
    }
    for (auto &entry : impl_->soft_cloth.entries) {
        if (!entry || !entry->enabled) continue;
        SoftResource *soft = impl_->soft_bodies.get(entry->options.soft_body);
        if (!soft) continue;
        soft_cloth_run(entry->table,
                       impl_->soft_cloth_soft_surface_pipeline,
                       soft->surface_count);
    }
    if (soft_cloth_record != nullptr)
        end_timing(encoder, timings, soft_cloth_record);
    for (auto &entry : impl_->rope_soft.entries) {
        if (!entry || !entry->enabled) continue;
        RopeResource *rope = impl_->ropes.get(entry->options.rope);
        SoftResource *soft = impl_->soft_bodies.get(entry->options.soft_body);
        if (!rope || !soft) continue;
        const auto &options = entry->options;
        auto *constants = static_cast<RopeSoftConstants *>(
            entry->constants.contents);
        constants->timestep = timestep;
        constants->rope_count = rope->count;
        constants->soft_count = soft->count;
        constants->surface_count = soft->surface_count;
        constants->surface_index_count = soft->surface_index_count;
        constants->contact_distance = options.contact_distance > 0.0F
                                          ? options.contact_distance
                                          : rope->options.radius;
        constants->friction = options.friction;
        constants->maximum_soft_acceleration =
            options.maximum_soft_body_acceleration;
        constants->maximum_rope_speed = rope->options.maximum_speed;
        constants->maximum_soft_speed = soft->options.maximum_speed;
        constants->node_radius = soft->options.node_radius;
        constants->anchor_support_radius_scale =
            options.anchor_support_radius_scale;
        constants->anchor_contact_support_radius_scale =
            options.anchor_contact_support_radius_scale;
        constants->orientation = soft->surface_orientation;
        constants->attach_first = options.attach_first ? 1U : 0U;
        constants->attach_last = options.attach_last ? 1U : 0U;
        constants->enabled = options.enabled ? 1U : 0U;
        constants->frame_inverse_timestep = impl_->frame_inverse_timestep;
        MetalTimingRecord *record = begin_timing(
            encoder, timings,
            MetalTimingStage::rope_soft_body_contacts);
        const auto run = [&](id<MTLComputePipelineState> pipeline,
                             NSUInteger threads) {
            const NSUInteger group = std::min<NSUInteger>(
                64U, pipeline.maxTotalThreadsPerThreadgroup);
            dispatch_untimed(entry->table, pipeline,
                             MTLSizeMake(threads, 1, 1),
                             MTLSizeMake(group, 1, 1));
        };
        run(impl_->rope_soft_anchor_weight_pipeline, 1U);
        run(impl_->rope_soft_pipeline, soft->count);
        run(impl_->rope_soft_surface_pipeline, soft->surface_count);
        run(impl_->rope_soft_endpoint_pipeline, 2U);
        run(impl_->rope_soft_finalize_pipeline, 1U);
        end_timing(encoder, timings, record);
    }
    return;
    }
    for (auto &entry : impl_->fluid_smoke.entries) {
            if (!entry) continue;
            FluidResource *fluid = impl_->fluids.get(entry->options.fluid);
            SmokeResource *smoke = impl_->smokes.get(entry->options.smoke);
            if (!fluid || !smoke) continue;
            const auto &options = entry->options;
            *static_cast<FluidSmokeConstants *>(entry->constants.contents) = {
                impl_->frame_timestep,
                gravity,
                options.heater.center,
                options.heater.orientation,
                options.heater.half_extents,
                options.heater_temperature,
                options.boiling_temperature,
                options.heat_transfer_rate,
                options.wind_drag,
                options.steam_rise_speed,
                fluid->options.particle_radius,
                smoke->options.particle_radius,
                smoke->options.rest_number_density,
                smoke->options.maximum_speed,
                smoke->options.lifetime,
                fluid->options.capacity,
                smoke->options.capacity,
                smoke->options.grid_resolution,
                smoke->options.grid_vertical_resolution,
                smoke->options.grid_resolution == 0U
                    ? 0.0F
                    : smoke->options.grid_edge_length /
                          static_cast<float>(smoke->options.grid_resolution),
                smoke->options.grid_minimum};
            MetalTimingRecord *record = begin_timing(
                encoder, timings,
                MetalTimingStage::fluid_smoke_exchange);
            const NSUInteger group = std::min<NSUInteger>(
                64U, impl_->fluid_smoke_pipeline
                         .maxTotalThreadsPerThreadgroup);
            dispatch_untimed(entry->table, impl_->fluid_smoke_pipeline,
                             MTLSizeMake(fluid->options.capacity, 1, 1),
                             MTLSizeMake(group, 1, 1));
            dispatch_untimed(entry->table,
                             impl_->fluid_smoke_compact_pipeline,
                             MTLSizeMake(64, 1, 1),
                             MTLSizeMake(64, 1, 1));
            end_timing(encoder, timings, record);
    }
    for (auto &entry : impl_->smokes.entries) {
            if (!entry) continue;
            SmokeResource &r = *entry;
            *static_cast<SmokeConstants *>(
                r.advection_constants.contents) = {
                impl_->frame_timestep, gravity, r.options.emitter_center,
                r.options.initial_velocity, r.options.wind,
                r.options.emitter_half_extents, 2U, r.options.lifetime,
                r.options.particle_radius, r.options.buoyancy,
                r.options.response, r.options.maximum_speed,
                r.options.rest_number_density,
                r.options.pressure_stiffness, r.options.viscosity,
                r.options.vorticity_confinement, r.options.capacity,
                r.options.grid_resolution,
                r.options.grid_vertical_resolution,
                r.options.grid_pressure_iterations,
                r.options.grid_minimum,
                r.options.grid_resolution == 0U
                    ? 0.0F
                    : r.options.grid_edge_length /
                          r.options.grid_resolution,
                r.options.grid_kinematic_viscosity,
                r.options.grid_les_coefficient,
                r.options.grid_pressure_tolerance};
            encode_smoke_advection(r);
    }
    encode_smoke_contacts();
    for (auto &entry : impl_->smoke_rigid.entries) {
        if (!entry || !entry->enabled || impl_->rigid_count == 0U)
            continue;
        auto *constants = static_cast<SmokeRigidConstants *>(
            entry->constants.contents);
        constants->rigid_count = impl_->rigid_count;
        constants->timestep = impl_->frame_timestep;
        impl_->bind(entry->table, 4U, impl_->rigid_ids);
        impl_->bind(entry->table, 5U, impl_->rigid_states);
        impl_->bind(entry->table, 6U, impl_->rigid_parameters);
        impl_->bind(entry->table, 7U, impl_->mesh_vertices);
        impl_->bind(entry->table, 8U, impl_->mesh_indices);
        impl_->bind(entry->table, 9U, impl_->mesh_infos);
        impl_->bind(entry->table, 15U, impl_->rigid_frame_states);
        const NSUInteger group = std::min<NSUInteger>(
            128U,
            impl_->smoke_rigid_pipeline.maxTotalThreadsPerThreadgroup);
        dispatch(MetalTimingStage::smoke_advection, entry->table,
                 impl_->smoke_rigid_pipeline,
                 MTLSizeMake(group, 1, 1),
                 MTLSizeMake(group, 1, 1));
    }
    {
        for (auto &entry : impl_->smokes.entries) {
            if (!entry) continue;
            SmokeResource &r = *entry;
            const double exact =
                static_cast<double>(r.emission_fraction) +
                static_cast<double>(r.options.particles_per_second) *
                    impl_->frame_timestep;
            const std::uint32_t requested =
                static_cast<std::uint32_t>(std::min(
                    std::floor(exact),
                    static_cast<double>(r.options.capacity)));
            r.emission_fraction = static_cast<float>(
                exact - std::floor(exact));
            static_cast<SmokeMetadata *>(r.metadata.contents)
                ->emission_remainder = r.emission_fraction;
            *static_cast<SmokeConstants *>(
                r.emission_constants.contents) = {
                impl_->frame_timestep, gravity, r.options.emitter_center,
                r.options.initial_velocity, r.options.wind,
                r.options.emitter_half_extents,
                requested, r.options.lifetime,
                r.options.particle_radius, r.options.buoyancy,
                r.options.response, r.options.maximum_speed,
                r.options.rest_number_density,
                r.options.pressure_stiffness, r.options.viscosity,
                r.options.vorticity_confinement, r.options.capacity,
                r.options.grid_resolution,
                r.options.grid_vertical_resolution,
                r.options.grid_pressure_iterations,
                r.options.grid_minimum,
                r.options.grid_resolution == 0U
                    ? 0.0F
                    : r.options.grid_edge_length /
                          r.options.grid_resolution,
                r.options.grid_kinematic_viscosity,
                r.options.grid_les_coefficient,
                r.options.grid_pressure_tolerance};
            if (requested != 0U)
                dispatch(MetalTimingStage::smoke_emission,
                         r.emission_table, impl_->smoke_emission_pipeline,
                         MTLSizeMake(128, 1, 1),
                         MTLSizeMake(128, 1, 1));
        }
    }
}

bool MetalSystems::empty() const noexcept {
    return !impl_ || (impl_->fluids.count == 0U && impl_->smokes.count == 0U &&
                      impl_->cloths.count == 0U &&
                      impl_->soft_bodies.count == 0U && impl_->ropes.count == 0U);
}

bool MetalSystems::references_rigid_body(RigidBodyId id) const noexcept {
    if (!impl_) return false;
    for (const auto &entry : impl_->smoke_rigid.entries)
        if (entry && entry->options.body == id) return true;
    for (const auto &entry : impl_->ropes.entries)
        if (entry && ((entry->options.first.enabled &&
                       entry->options.first.body == id) ||
                      (entry->options.last.enabled &&
                       entry->options.last.body == id)))
            return true;
    for (const auto &entry : impl_->paint_fields.entries)
        if (entry && entry->options.body == id) return true;
    for (const auto &entry : impl_->paint_rules.entries)
        if (entry && entry->options.rigid_source == id) return true;
    return false;
}

bool MetalSystems::references_triangle_mesh(TriangleMeshId id) const noexcept {
    if (!impl_) return false;
    for (const auto &entry : impl_->paint_fields.entries)
        if (entry && entry->options.mesh == id) return true;
    return false;
}

ContactDeviceView MetalSystems::contact_view(
    std::uint64_t frame_index) const noexcept {
    if (!impl_ || impl_->fluid_contact_events == nil) return {};
    if (impl_->contact_cache_frame != frame_index) {
        impl_->fluid_contact_count = 0U;
        impl_->fluid_contact_overflow = 0U;
        if (impl_->collect_fluid_contacts) {
            auto *events = static_cast<ContactEvent *>(
                impl_->fluid_contact_events.contents);
            const std::uint32_t capacity = impl_->options.contact_capacity;
            for (const auto &entry : impl_->fluids.entries) {
                if (!entry) continue;
                const FluidResource &fluid = *entry;
                const std::uint32_t count = std::min(
                    static_cast<const ParticleMetadata *>(
                        fluid.metadata.contents)->count,
                    fluid.options.capacity);
                const auto *samples = static_cast<const ContactEvent *>(
                    fluid.contact_samples.contents);
                const auto *flags = static_cast<const std::uint32_t *>(
                    fluid.contact_flags.contents);
                for (std::uint32_t particle = 0U; particle < count;
                     ++particle) {
                    if (flags[particle] == 0U) continue;
                    if (impl_->fluid_contact_count < capacity) {
                        events[impl_->fluid_contact_count++] =
                            samples[particle];
                    } else {
                        ++impl_->fluid_contact_overflow;
                    }
                }
            }
        }
        impl_->contact_cache_frame = frame_index;
    }
    return {span<ContactEvent>(impl_->fluid_contact_events,
                               impl_->fluid_contact_count),
            impl_->fluid_contact_count,
            impl_->fluid_contact_overflow != 0U,
            frame_index};
}

std::uint32_t MetalSystems::fluid_contact_overflow_count() const noexcept {
    return impl_ == nullptr ? 0U : impl_->fluid_contact_overflow;
}

Status MetalSystems::append_debug_samples(
    PhysicsDebugFrame &output, std::uint64_t frame_index) const noexcept {
    if (!impl_) return invalid_argument("World is not initialized");
    try {
        output.fluid_particles.clear();
        output.cloth_vertices.clear();
        output.soft_body_nodes.clear();
        output.rope_nodes.clear();
        output.maximum_fluid_neighbor_count =
            *static_cast<const std::uint32_t *>(
                impl_->fluid_maximum_neighbor_count.contents);
        for (std::uint32_t slot = 0U;
             slot < impl_->fluids.entries.size(); ++slot) {
            const auto &entry = impl_->fluids.entries[slot];
            if (!entry) continue;
            const FluidResource &fluid = *entry;
            const std::uint32_t count = std::min(
                static_cast<const ParticleMetadata *>(fluid.metadata.contents)
                    ->count,
                fluid.options.capacity);
            const auto *positions = static_cast<const Vec3 *>(
                fluid.particles.positions.contents);
            const auto *velocities = static_cast<const Vec3 *>(
                fluid.particles.velocities.contents);
            const auto *accelerations = static_cast<const Vec3 *>(
                fluid.accelerations.contents);
            const auto *stable_ids = static_cast<const std::uint32_t *>(
                fluid.stable_ids.contents);
            const auto *foam = static_cast<const float *>(fluid.foam.contents);
            output.fluid_particles.reserve(
                output.fluid_particles.size() + count);
            for (std::uint32_t particle = 0U; particle < count; ++particle)
                output.fluid_particles.push_back({
                    {slot, impl_->fluids.generations[slot]},
                    stable_ids[particle], positions[particle],
                    velocities[particle], accelerations[particle],
                    foam[particle]});
        }
        for (std::uint32_t slot = 0U;
             slot < impl_->cloths.entries.size(); ++slot) {
            const auto &entry = impl_->cloths.entries[slot];
            if (!entry) continue;
            const ClothResource &cloth = *entry;
            const auto *positions = static_cast<const Vec3 *>(
                cloth.particles.positions.contents);
            const auto *velocities = static_cast<const Vec3 *>(
                cloth.particles.velocities.contents);
            const auto *rigid = static_cast<const Vec3 *>(
                cloth.rigid_forces.contents);
            const auto *fluid = static_cast<const Vec3 *>(
                cloth.fluid_forces.contents);
            const auto *soft = static_cast<const Vec3 *>(
                cloth.soft_forces.contents);
            output.cloth_vertices.reserve(output.cloth_vertices.size() +
                                          cloth.count);
            for (std::uint32_t vertex = 0U; vertex < cloth.count; ++vertex)
                output.cloth_vertices.push_back({
                    {slot, impl_->cloths.generations[slot]}, vertex,
                    positions[vertex], velocities[vertex], rigid[vertex],
                    fluid[vertex], soft[vertex]});
        }
        for (std::uint32_t slot = 0U;
             slot < impl_->soft_bodies.entries.size(); ++slot) {
            const auto &entry = impl_->soft_bodies.entries[slot];
            if (!entry) continue;
            const SoftResource &soft = *entry;
            const auto *positions = static_cast<const Vec3 *>(
                soft.particles.positions.contents);
            const auto *velocities = static_cast<const Vec3 *>(
                soft.particles.velocities.contents);
            const auto *rigid = static_cast<const Vec3 *>(
                soft.rigid_forces.contents);
            const auto *cloth = static_cast<const Vec3 *>(
                soft.cloth_forces.contents);
            const auto *fluid = static_cast<const Vec3 *>(
                soft.fluid_forces.contents);
            output.soft_body_nodes.reserve(output.soft_body_nodes.size() +
                                           soft.count);
            for (std::uint32_t node = 0U; node < soft.count; ++node)
                output.soft_body_nodes.push_back({
                    {slot, impl_->soft_bodies.generations[slot]}, node,
                    positions[node], velocities[node], rigid[node],
                    cloth[node], fluid[node]});
        }
        for (std::uint32_t slot = 0U;
             slot < impl_->ropes.entries.size(); ++slot) {
            const auto &entry = impl_->ropes.entries[slot];
            if (!entry) continue;
            const RopeResource &rope = *entry;
            const auto *positions = static_cast<const Vec3 *>(
                rope.particles.positions.contents);
            const auto *velocities = static_cast<const Vec3 *>(
                rope.particles.velocities.contents);
            const auto *constraint = static_cast<const Vec3 *>(
                rope.constraint_forces.contents);
            const auto *contact = static_cast<const Vec3 *>(
                rope.contact_forces.contents);
            const auto *fluid = static_cast<const Vec3 *>(
                rope.fluid_forces.contents);
            output.rope_nodes.reserve(output.rope_nodes.size() + rope.count);
            for (std::uint32_t node = 0U; node < rope.count; ++node)
                output.rope_nodes.push_back({
                    {slot, impl_->ropes.generations[slot]}, node,
                    positions[node], velocities[node], constraint[node],
                    contact[node], fluid[node]});
        }
        const ContactDeviceView contacts = contact_view(frame_index);
        const auto *events = static_cast<const ContactEvent *>(
            impl_->fluid_contact_events.contents);
        output.fluid_contacts.assign(events,
                                     events + contacts.event_count);
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory("Could not record Metal physics debug samples");
    } catch (...) {
        return {StatusCode::internal_error, 0,
                "Unexpected Metal physics debug capture failure"};
    }
}

std::uint32_t MetalSystems::required_substeps(
    float timestep, std::uint32_t requested) const noexcept {
    if (!impl_) return requested;
    std::uint32_t result = requested;
    for (const auto &entry : impl_->ropes.entries) {
        if (!entry || !(entry->options.maximum_substep_timestep > 0.0F))
            continue;
        const float required = std::ceil(
            timestep / entry->options.maximum_substep_timestep);
        if (!std::isfinite(required) || required > 1'024.0F)
            return 1'025U;
        const auto needed = static_cast<std::uint32_t>(required);
        result = std::max(result, needed);
    }
    return result;
}

void MetalSystems::collect_statistics(WorldStatistics &output) const noexcept {
    if (!impl_) return;
    output.fluid_count = impl_->fluids.count;
    output.smoke_system_count = impl_->smokes.count;
    output.cloth_count = impl_->cloths.count;
    output.soft_body_count = impl_->soft_bodies.count;
    output.rope_count = impl_->ropes.count;
    output.maximum_fluid_neighbor_count =
        *static_cast<const std::uint32_t *>(
            impl_->fluid_maximum_neighbor_count.contents);
    output.emitted_particle_count =
        impl_->archived_emitted_particle_count;
    output.destroyed_particle_count =
        impl_->archived_destroyed_particle_count;
    output.boiled_particle_count =
        impl_->archived_boiled_particle_count;
    const auto *capacity_misses = static_cast<const SplitCounter *>(
        impl_->spawn_capacity_miss_count.contents);
    output.spawn_capacity_miss_count =
        (static_cast<std::uint64_t>(capacity_misses->high) << 32U) |
        capacity_misses->low;
    for (const auto &entry : impl_->fluids.entries)
        if (entry) {
            const std::uint32_t count = std::min(
                static_cast<const ParticleMetadata *>(entry->metadata.contents)
                    ->count,
                entry->options.capacity);
            output.particle_count += count;
        }
    for (const auto &entry : impl_->smokes.entries)
        if (entry)
            output.smoke_particle_count +=
                static_cast<const SmokeMetadata *>(entry->metadata.contents)->count;
    for (const auto &entry : impl_->smokes.entries)
        if (entry)
            output.emitted_smoke_particle_count +=
                static_cast<const SmokeMetadata *>(entry->metadata.contents)
                    ->emitted;
    for (const auto &entry : impl_->fluids.entries)
        if (entry) {
            const auto *metadata = static_cast<const ParticleMetadata *>(
                entry->metadata.contents);
            output.emitted_particle_count += metadata->emitted;
            const std::uint64_t live = std::min(
                metadata->count, entry->options.capacity);
            output.destroyed_particle_count +=
                static_cast<std::uint64_t>(entry->initial_count) +
                metadata->emitted - live;
            output.boiled_particle_count += metadata->boiled;
        }
    for (const auto &entry : impl_->cloths.entries)
        if (entry) output.cloth_vertex_count += entry->count;
    for (const auto &entry : impl_->soft_bodies.entries)
        if (entry) output.soft_body_node_count += entry->count;
    for (const auto &entry : impl_->ropes.entries)
        if (entry) output.rope_node_count += entry->count;
    for (const auto &entry : impl_->fluid_soft.entries) {
        if (!entry) continue;
        output.fluid_soft_body_contact_count +=
            *static_cast<const std::uint32_t *>(
                entry->contact_count.contents);
        output.maximum_fluid_soft_body_penetration = std::max(
            output.maximum_fluid_soft_body_penetration,
            *static_cast<const float *>(
                entry->maximum_penetration.contents));
    }
    for (const auto &entry : impl_->fluid_rope.entries) {
        if (!entry) continue;
        output.fluid_rope_contact_count +=
            *static_cast<const std::uint32_t *>(
                entry->contact_count.contents);
        output.maximum_fluid_rope_penetration = std::max(
            output.maximum_fluid_rope_penetration,
            *static_cast<const float *>(
                entry->maximum_penetration.contents));
    }
    for (const auto &entry : impl_->rope_soft.entries) {
        if (!entry) continue;
        output.rope_soft_body_contact_count +=
            *static_cast<const std::uint32_t *>(
                entry->contact_count.contents);
        output.maximum_rope_soft_body_penetration = std::max(
            output.maximum_rope_soft_body_penetration,
            *static_cast<const float *>(
                entry->maximum_penetration.contents));
    }
}

std::size_t MetalSystems::allocated_bytes() const noexcept {
    if (!impl_) return 0U;
    std::size_t result = 0U;
    const auto add = [&result](id<MTLBuffer> buffer) {
        if (buffer != nil) result += buffer.length;
    };
    const auto add_particles = [&add](const ParticleBuffers &particles) {
        add(particles.positions); add(particles.previous);
        add(particles.velocities); add(particles.inverse_masses);
        add(particles.rigid_contacts); add(particles.rigid_contact_counts);
        add(particles.rigid_linear_impulses);
        add(particles.rigid_angular_impulses);
        add(particles.rigid_position_corrections);
        add(particles.rigid_particle_linear_impulses);
        add(particles.rigid_particle_angular_impulses);
        add(particles.rigid_particle_position_corrections);
    };
    add(impl_->fluid_contact_events);
    add(impl_->fluid_neighbor_overflow);
    add(impl_->fluid_maximum_neighbor_count);
    add(impl_->spawn_capacity_miss_count);
    for (const auto &entry : impl_->fluids.entries) if (entry) {
        add_particles(entry->particles);
        add(entry->accelerations); add(entry->stable_ids); add(entry->foam);
        add(entry->foam_sources); add(entry->temperatures);
        add(entry->cell_keys[0]); add(entry->cell_keys[1]);
        add(entry->sorted_indices[0]); add(entry->sorted_indices[1]);
        add(entry->radix_histograms); add(entry->radix_bucket_offsets);
        add(entry->contact_samples); add(entry->contact_flags);
        add(entry->metadata); add(entry->constants); add(entry->rigid_constants);
        add(entry->rigid_first_constants);
    }
    for (const auto &entry : impl_->smokes.entries) if (entry) {
        add_particles(entry->particles);
        add(entry->ages); add(entry->densities); add(entry->pressures);
        add(entry->vorticities); add(entry->metadata); add(entry->constants);
        add(entry->advection_constants); add(entry->emission_constants);
        add(entry->grid_velocity); add(entry->grid_pressure);
        add(entry->grid_density); add(entry->grid_temperature);
        add(entry->grid_solid); add(entry->grid_vorticity);
        add(entry->grid_divergence); add(entry->grid_scratch);
        add(entry->grid_pressure_relative_residual);
        add(entry->grid_deformable_solid);
        add(entry->grid_rigid_bodies); add(entry->grid_rigid_body_count);
        add(entry->grid_face_velocity); add(entry->grid_face_advection);
        add(entry->grid_face_boundary);
        add(entry->grid_pressure_state);
    }
    for (const auto &entry : impl_->cloths.entries) if (entry) {
        add_particles(entry->particles);
        add(entry->triangle_indices); add(entry->source_indices);
        add(entry->surface_positions); add(entry->surface_indices);
        add(entry->surface_source_indices); add(entry->surface_physical_indices);
        add(entry->public_bonds); add(entry->bonds); add(entry->active_bonds);
        add(entry->bond_damage); add(entry->neighbor_offsets);
        add(entry->neighbors); add(entry->free_triangle_nodes);
        add(entry->corrections);
        add(entry->rigid_forces);
        add(entry->fluid_forces); add(entry->soft_forces);
        add(entry->rope_forces); add(entry->smoke_forces);
        add(entry->constants);
        add(entry->rigid_constants);
        add(entry->body_corrections);
        add(entry->surface_rigid_constants);
    }
    for (const auto &entry : impl_->soft_bodies.entries) if (entry) {
        add_particles(entry->particles);
        add(entry->public_bonds); add(entry->bonds);
        add(entry->rest_positions); add(entry->corrections);
        add(entry->surface_positions); add(entry->surface_rest_positions);
        add(entry->surface_indices); add(entry->surface_bindings);
        add(entry->shape_orientation);
        add(entry->contact_accumulators); add(entry->contact_state);
        add(entry->rigid_forces); add(entry->cloth_forces);
        add(entry->fluid_forces); add(entry->rope_forces);
        add(entry->smoke_forces);
        add(entry->constants); add(entry->rigid_constants);
    }
    for (const auto &entry : impl_->ropes.entries) if (entry) {
        add_particles(entry->particles);
        add(entry->rest_lengths); add(entry->constraint_forces);
        add(entry->contact_forces); add(entry->fluid_forces);
        add(entry->soft_forces); add(entry->bonds); add(entry->directions);
        add(entry->diagonal); add(entry->upper); add(entry->rhs);
        add(entry->lambdas); add(entry->scratch); add(entry->constants);
        add(entry->contact_normals); add(entry->contact_normals2);
        add(entry->body_translation); add(entry->body_rotation);
        add(entry->anchor_states); add(entry->empty_soft_target);
        add(entry->attachment_constants);
    }
    for (const auto &entry : impl_->sources.entries) if (entry) {
        add(entry->sites); add(entry->constants);
    }
    for (const auto &entry : impl_->destroy_planes.entries) if (entry)
        add(entry->constants);
    const auto add_couplings = [&add](const auto &slots) {
        for (const auto &entry : slots.entries)
            if (entry) {
                add(entry->constants);
                add(entry->phase_constants);
                add(entry->contact_count);
                add(entry->maximum_penetration);
                add(entry->packed_state);
                add(entry->previous_surface);
                add(entry->contribution_nodes);
                add(entry->contribution_positions);
                add(entry->contribution_changes);
                add(entry->contribution_forces);
            }
    };
    add_couplings(impl_->fluid_smoke);
    add_couplings(impl_->smoke_soft);
    add_couplings(impl_->smoke_cloth);
    add_couplings(impl_->smoke_rope);
    add_couplings(impl_->smoke_rigid);
    add_couplings(impl_->fluid_rope);
    add_couplings(impl_->rope_soft);
    add_couplings(impl_->rope_cloth);
    add_couplings(impl_->fluid_cloth);
    add_couplings(impl_->soft_cloth);
    add_couplings(impl_->fluid_soft);
    for (const auto &entry : impl_->paint_fields.entries) if (entry) {
        add(entry->uvs); add(entry->pixels);
    }
    for (const auto &entry : impl_->paint_rules.entries) if (entry)
        add(entry->constants);
    return result;
}

void *MetalSystems::fluid_neighbor_overflow_buffer() const noexcept {
    return impl_ == nullptr
               ? nullptr
               : (__bridge void *)impl_->fluid_neighbor_overflow;
}

} // namespace parallel_mater::metal::detail

namespace parallel_mater::metal {
namespace {
Status system_busy() noexcept {
    return {StatusCode::busy, 0,
            "Particle systems cannot change while a frame is in flight"};
}
Status no_world() noexcept {
    return {StatusCode::invalid_argument, 0, "World is not initialized"};
}
} // namespace

Status World::add_paint_field(
    PaintFieldHostOptions options, PaintFieldId &output) noexcept {
    output = {};
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    const bool cloth_target = options.cloth.generation != 0U;
    std::uint32_t vertex_count = 0U;
    if (cloth_target) {
        if (options.body.generation != 0U ||
            options.mesh.generation != 0U)
            return {StatusCode::invalid_handle, 0,
                    "Paint cloth target cannot also name a rigid body"};
        vertex_count = systems->cloth_source_vertex_count(options.cloth);
        if (vertex_count == 0U)
            return {StatusCode::invalid_handle, 0,
                    "Paint cloth target is stale"};
    } else {
        if (!systems_rigid_body_valid(options.body))
            return {StatusCode::invalid_handle, 0,
                    "Paint rigid-body target is stale"};
        vertex_count = systems_triangle_mesh_vertex_count(options.mesh);
        if (vertex_count == 0U)
            return {StatusCode::invalid_handle, 0,
                    "Paint triangle-mesh target is stale"};
    }
    return systems_finish_mutation(
        systems->add_paint_field(options, vertex_count, output));
}

Status World::add_paint_field(
    PaintFieldOptions options, PaintFieldId &output) noexcept {
    output = {};
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    if (options.vertex_uvs.buffer == nullptr ||
        options.vertex_uvs.byte_offset % alignof(Vec2) != 0U ||
        options.vertex_uvs.size == 0U ||
        options.vertex_uvs.size >
            std::numeric_limits<NSUInteger>::max() / sizeof(Vec2))
        return {StatusCode::invalid_argument, 0,
                "Paint UV Metal span is invalid"};
    id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)options.vertex_uvs.buffer;
    const NSUInteger bytes = static_cast<NSUInteger>(
        options.vertex_uvs.size * sizeof(Vec2));
    if (buffer.device != (__bridge id<MTLDevice>)native_context().device ||
        options.vertex_uvs.byte_offset > buffer.length ||
        bytes > buffer.length - options.vertex_uvs.byte_offset)
        return {StatusCode::invalid_argument, 0,
                "Paint UV Metal span has the wrong device or range"};
    PaintFieldHostOptions host{options.body, options.mesh, options.cloth, {},
                               options.width, options.height};
    const auto *contents =
        buffer.storageMode == MTLStorageModePrivate
            ? nullptr
            : static_cast<const std::byte *>(buffer.contents);
    if (contents != nullptr) {
        host.vertex_uvs = {
            reinterpret_cast<const Vec2 *>(
                contents + options.vertex_uvs.byte_offset),
            options.vertex_uvs.size};
        return add_paint_field(host, output);
    }
    try {
        std::vector<Vec2> staged(options.vertex_uvs.size);
        Status status = copy_metal_buffer_to_host(
            options.vertex_uvs.buffer, options.vertex_uvs.byte_offset, bytes,
            staged.data());
        if (!status) return status;
        host.vertex_uvs = {staged.data(), staged.size()};
        return add_paint_field(host, output);
    } catch (const std::bad_alloc &) {
        return {StatusCode::out_of_memory, 0,
                "Could not allocate private paint UV upload staging"};
    }
}

Status World::remove_paint_field(PaintFieldId id) noexcept {
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    return systems_finish_mutation(systems->remove_paint_field(id));
}

Status World::clear_paint_field(PaintFieldId id) noexcept {
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    return systems->clear_paint_field(id);
}

Status World::paint_field_view(
    PaintFieldId id, PaintFieldDeviceView &output) const noexcept {
    output = {};
    const auto *systems = static_cast<const detail::MetalSystems *>(
        systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    Status status = systems->paint_field_view(id, output);
    if (status) output.revision = systems_revision();
    return status;
}

Status World::add_paint_rule(
    PaintRuleOptions options, PaintRuleId &output) noexcept {
    output = {};
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    if (options.rigid_source.generation != 0U &&
        !systems_rigid_body_valid(options.rigid_source))
        return {StatusCode::invalid_handle, 0,
                "Paint rule rigid source is stale"};
    return systems_finish_mutation(systems->add_paint_rule(options, output));
}

Status World::remove_paint_rule(PaintRuleId id) noexcept {
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    return systems_finish_mutation(systems->remove_paint_rule(id));
}

Status World::add_fluid(FluidOptions options,
                        HostSpan<const FluidParticle> particles,
                        FluidId &output) noexcept {
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    Status reserve_status = systems_reserve_debug_samples(
        options.capacity, 0U, 0U, 0U);
    if (!reserve_status) return reserve_status;
    return systems_finish_mutation(
        systems->add_fluid(options, particles, output));
}

Status World::add_fluid(FluidOptions options,
                        BufferSpan<const FluidParticle> particles,
                        FluidId &output) noexcept {
    output = {};
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    if (particles.size == 0U)
        return add_fluid(options, HostSpan<const FluidParticle>{}, output);
    if (particles.buffer == nullptr ||
        particles.byte_offset % alignof(FluidParticle) != 0U ||
        particles.size >
            std::numeric_limits<NSUInteger>::max() / sizeof(FluidParticle))
        return {StatusCode::invalid_argument, 0,
                "Fluid particle Metal span is invalid"};
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)particles.buffer;
    const NSUInteger bytes =
        static_cast<NSUInteger>(particles.size * sizeof(FluidParticle));
    if (particles.byte_offset > buffer.length ||
        bytes > buffer.length - particles.byte_offset)
        return {StatusCode::invalid_argument, 0,
                "Fluid particle Metal span is out of range"};
    if ((__bridge void *)buffer.device != native_context().device)
        return {StatusCode::invalid_argument, 0,
                "Fluid particle buffer belongs to another Metal device"};
    const auto *contents =
        buffer.storageMode == MTLStorageModePrivate
            ? nullptr
            : static_cast<const std::byte *>(buffer.contents);
    if (contents == nullptr) {
        try {
            std::vector<FluidParticle> staged(particles.size);
            Status copy_status = copy_metal_buffer_to_host(
                particles.buffer, particles.byte_offset, bytes, staged.data());
            return copy_status
                       ? add_fluid(
                             options,
                             HostSpan<const FluidParticle>{staged.data(),
                                                           staged.size()},
                             output)
                       : copy_status;
        } catch (const std::bad_alloc &) {
            return {StatusCode::out_of_memory, 0,
                    "Could not allocate private fluid upload staging"};
        }
    }
    return add_fluid(
        options,
        {reinterpret_cast<const FluidParticle *>(contents +
                                                 particles.byte_offset),
         particles.size},
        output);
}

Status World::add_fluid_geometry(FluidOptions options,
                                 FluidGeometrySource source,
                                 FluidId &output) noexcept {
    std::vector<FluidParticle> particles;
    Status status = sample_fluid_geometry(source, particles);
    if (!status) return status;
    if (options.capacity == 0U)
        return {StatusCode::invalid_argument, 0,
                "Fluid geometry requires a positive fluid capacity"};
    if (particles.size() > options.capacity) {
        try {
            std::vector<FluidParticle> selected;
            selected.reserve(options.capacity);
            for (std::size_t index = 0U; index < options.capacity; ++index)
                selected.push_back(
                    particles[index * particles.size() / options.capacity]);
            particles.swap(selected);
        } catch (const std::bad_alloc &) {
            return {StatusCode::out_of_memory, 0,
                    "Could not select sampled fluid geometry particles"};
        }
    }
    return add_fluid(
        options,
        HostSpan<const FluidParticle>{
            particles.data(), static_cast<std::uint64_t>(particles.size())},
        output);
}

Status World::add_smoke_rigid_coupling(
    SmokeRigidCouplingOptions options,
    SmokeRigidCouplingId &output) noexcept {
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    if (!systems_rigid_body_valid(options.body))
        return {StatusCode::invalid_handle, 0,
                "Smoke-rigid body handle is stale"};
    return systems_finish_mutation(
        systems->add_smoke_rigid_coupling(options, output));
}

Status World::add_rope(RopeOptions options, RopeId &output) noexcept {
    output = {};
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    if ((options.first.enabled &&
         !systems_rigid_body_valid(options.first.body)) ||
        (options.last.enabled &&
         !systems_rigid_body_valid(options.last.body))) {
        return {StatusCode::invalid_handle, 0,
                "Rope attachment rigid-body handle is stale"};
    }
    if (options.first.enabled && options.last.enabled &&
        options.first.body == options.last.body)
        return {StatusCode::invalid_argument, 0,
                "Rope endpoints must attach to distinct bodies"};
    std::vector<Vec3> nodes;
    Status status = sample_rope_centerline(options.centerline,
                                           options.node_spacing, nodes);
    if (!status) return status;
    const auto rotate = [](Quaternion orientation, Vec3 value) {
        const Vec3 q{orientation.x, orientation.y, orientation.z};
        const Vec3 first{q.y * value.z - q.z * value.y,
                         q.z * value.x - q.x * value.z,
                         q.x * value.y - q.y * value.x};
        const Vec3 second_input{
            first.x + orientation.w * value.x,
            first.y + orientation.w * value.y,
            first.z + orientation.w * value.z};
        const Vec3 second{
            q.y * second_input.z - q.z * second_input.y,
            q.z * second_input.x - q.x * second_input.z,
            q.x * second_input.y - q.y * second_input.x};
        return Vec3{value.x + 2.0F * second.x,
                    value.y + 2.0F * second.y,
                    value.z + 2.0F * second.z};
    };
    const RopeAttachment attachments[2]{options.first, options.last};
    for (std::uint32_t end = 0U; end < 2U; ++end) {
        const RopeAttachment attachment = attachments[end];
        if (!attachment.enabled) continue;
        RigidBodyState body{};
        status = read_rigid_body_state(attachment.body, body);
        if (!status) return status;
        const Vec3 local = rotate(body.orientation, attachment.local_anchor);
        const Vec3 target{body.position.x + local.x,
                          body.position.y + local.y,
                          body.position.z + local.z};
        const Vec3 point = end == 0U ? nodes.front() : nodes.back();
        const float dx = target.x - point.x;
        const float dy = target.y - point.y;
        const float dz = target.z - point.z;
        if (dx * dx + dy * dy + dz * dz > 1.0e-6F)
            return {StatusCode::invalid_argument, 0,
                    "Rope endpoint must match its body-local attachment"};
    }
    std::uint32_t first_contact_skip = 2U;
    std::uint32_t last_contact_skip = 2U;
    status = systems_validate_rope_rest(
        {nodes.data(), static_cast<std::uint64_t>(nodes.size())},
        options.first, options.last, first_contact_skip, last_contact_skip);
    if (!status) return status;
    status = systems_reserve_debug_samples(
        0U, 0U, 0U, static_cast<std::uint64_t>(nodes.size()));
    if (!status) return status;
    return systems_finish_mutation(systems->add_rope(
        options, first_contact_skip, last_contact_skip, output));
}

Status World::add_cloth(ClothOptions options, ClothId &output) noexcept {
    output = {};
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    const bool fracture_enabled = options.break_strain > 0.0F ||
                                  options.impact_break_impulse > 0.0F;
    if (fracture_enabled &&
        options.triangle_indices.size >
            UINT64_MAX - options.vertices.size)
        return {StatusCode::capacity_exceeded, 0,
                "Cloth debug capacity exceeds uint64 range"};
    const std::uint64_t maximum_vertices = options.vertices.size +
        (fracture_enabled ? options.triangle_indices.size : 0U);
    Status status = systems_reserve_debug_samples(
        0U, maximum_vertices, 0U, 0U);
    if (!status) return status;
    return systems_finish_mutation(systems->add_cloth(options, output));
}

Status World::add_soft_body(
    SoftBodyOptions options, SoftBodyId &output) noexcept {
    output = {};
    auto *systems =
        static_cast<detail::MetalSystems *>(systems_implementation());
    if (!systems) return no_world();
    if (!systems_mutation_allowed()) return system_busy();
    Status status = systems_reserve_debug_samples(
        0U, 0U, options.nodes.size, 0U);
    if (!status) return status;
    return systems_finish_mutation(systems->add_soft_body(options, output));
}

#define PM_SYSTEM_FORWARD_1(name, Type)                                      \
    Status World::name(Type value) noexcept {                                \
        auto *systems = static_cast<detail::MetalSystems *>(                 \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        return systems_finish_mutation(systems->name(value));                \
    }
#define PM_SYSTEM_FORWARD_2(name, Type1, Type2)                              \
    Status World::name(Type1 first, Type2 second) noexcept {                 \
        auto *systems = static_cast<detail::MetalSystems *>(                 \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        return systems_finish_mutation(systems->name(first, second));        \
    }
#define PM_SYSTEM_FORWARD_3(name, Type1, Type2, Type3)                       \
    Status World::name(Type1 first, Type2 second, Type3 third) noexcept {    \
        auto *systems = static_cast<detail::MetalSystems *>(                 \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        return systems_finish_mutation(                                      \
            systems->name(first, second, third));                            \
    }
#define PM_SYSTEM_VIEW(name, Id, View)                                       \
    Status World::name(Id id, View &output) const noexcept {                 \
        output = {};                                                         \
        auto *systems = static_cast<const detail::MetalSystems *>(           \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        return systems->name(id, output);                                    \
    }
#define PM_SYSTEM_REVISION_VIEW(name, Id, View)                              \
    Status World::name(Id id, View &output) const noexcept {                 \
        output = {};                                                         \
        auto *systems = static_cast<const detail::MetalSystems *>(           \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        Status status = systems->name(id, output);                           \
        if (status) output.revision = systems_revision();                    \
        return status;                                                       \
    }

PM_SYSTEM_FORWARD_1(remove_fluid, FluidId)
PM_SYSTEM_REVISION_VIEW(fluid_view, FluidId, FluidDeviceView)
PM_SYSTEM_FORWARD_2(add_smoke, SmokeOptions, SmokeId &)
PM_SYSTEM_FORWARD_1(remove_smoke, SmokeId)
PM_SYSTEM_REVISION_VIEW(smoke_view, SmokeId, SmokeDeviceView)
PM_SYSTEM_FORWARD_1(remove_cloth, ClothId)
PM_SYSTEM_VIEW(cloth_view, ClothId, ClothDeviceView)
PM_SYSTEM_FORWARD_1(remove_soft_body, SoftBodyId)
PM_SYSTEM_VIEW(soft_body_view, SoftBodyId, SoftBodyDeviceView)
PM_SYSTEM_FORWARD_1(remove_rope, RopeId)
PM_SYSTEM_VIEW(rope_view, RopeId, RopeDeviceView)
#define PM_SYSTEM_UNREVISED_FORWARD_1(name, Type)                            \
    Status World::name(Type value) noexcept {                                \
        auto *systems = static_cast<detail::MetalSystems *>(                 \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        return systems->name(value);                                         \
    }
#define PM_SYSTEM_UNREVISED_FORWARD_2(name, Type1, Type2)                    \
    Status World::name(Type1 first, Type2 second) noexcept {                 \
        auto *systems = static_cast<detail::MetalSystems *>(                 \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        return systems->name(first, second);                                 \
    }
#define PM_SYSTEM_UNREVISED_FORWARD_3(name, Type1, Type2, Type3)             \
    Status World::name(Type1 first, Type2 second, Type3 third) noexcept {    \
        auto *systems = static_cast<detail::MetalSystems *>(                 \
            systems_implementation());                                       \
        if (!systems) return no_world();                                     \
        if (!systems_mutation_allowed()) return system_busy();               \
        return systems->name(first, second, third);                          \
    }

PM_SYSTEM_UNREVISED_FORWARD_3(add_particle_source, ParticleSourceMesh,
                              ParticleSourceOptions, ParticleSourceId &)
PM_SYSTEM_UNREVISED_FORWARD_2(update_particle_source, ParticleSourceId,
                              ParticleSourceOptions)
PM_SYSTEM_UNREVISED_FORWARD_1(remove_particle_source, ParticleSourceId)
PM_SYSTEM_UNREVISED_FORWARD_2(add_particle_destroy_plane,
                              ParticleDestroyPlaneOptions,
                              ParticleDestroyPlaneId &)
PM_SYSTEM_UNREVISED_FORWARD_2(update_particle_destroy_plane,
                              ParticleDestroyPlaneId,
                              ParticleDestroyPlaneOptions)
PM_SYSTEM_UNREVISED_FORWARD_1(remove_particle_destroy_plane,
                              ParticleDestroyPlaneId)

PM_SYSTEM_FORWARD_2(add_fluid_smoke_coupling, FluidSmokeCouplingOptions,
                    FluidSmokeCouplingId &)
PM_SYSTEM_FORWARD_1(remove_fluid_smoke_coupling, FluidSmokeCouplingId)
PM_SYSTEM_FORWARD_2(add_smoke_soft_body_coupling,
                    SmokeSoftBodyCouplingOptions, SmokeSoftBodyCouplingId &)
PM_SYSTEM_FORWARD_1(remove_smoke_soft_body_coupling,
                    SmokeSoftBodyCouplingId)
PM_SYSTEM_FORWARD_2(add_smoke_cloth_coupling, SmokeClothCouplingOptions,
                    SmokeClothCouplingId &)
PM_SYSTEM_FORWARD_1(remove_smoke_cloth_coupling, SmokeClothCouplingId)
PM_SYSTEM_FORWARD_2(add_smoke_rope_coupling, SmokeRopeCouplingOptions,
                    SmokeRopeCouplingId &)
PM_SYSTEM_FORWARD_1(remove_smoke_rope_coupling, SmokeRopeCouplingId)
PM_SYSTEM_FORWARD_1(remove_smoke_rigid_coupling, SmokeRigidCouplingId)

PM_SYSTEM_FORWARD_2(add_fluid_rope_coupling, FluidRopeCouplingOptions,
                    FluidRopeCouplingId &)
PM_SYSTEM_FORWARD_2(update_fluid_rope_coupling, FluidRopeCouplingId,
                    FluidRopeCouplingOptions)
PM_SYSTEM_FORWARD_1(remove_fluid_rope_coupling, FluidRopeCouplingId)
PM_SYSTEM_FORWARD_2(add_rope_soft_body_coupling,
                    RopeSoftBodyCouplingOptions, RopeSoftBodyCouplingId &)
PM_SYSTEM_FORWARD_2(update_rope_soft_body_coupling, RopeSoftBodyCouplingId,
                    RopeSoftBodyCouplingOptions)
PM_SYSTEM_FORWARD_1(remove_rope_soft_body_coupling,
                    RopeSoftBodyCouplingId)
PM_SYSTEM_FORWARD_2(add_rope_cloth_coupling, RopeClothCouplingOptions,
                    RopeClothCouplingId &)
PM_SYSTEM_FORWARD_2(update_rope_cloth_coupling, RopeClothCouplingId,
                    RopeClothCouplingOptions)
PM_SYSTEM_FORWARD_1(remove_rope_cloth_coupling, RopeClothCouplingId)
PM_SYSTEM_FORWARD_2(add_fluid_cloth_coupling, FluidClothCouplingOptions,
                    FluidClothCouplingId &)
PM_SYSTEM_FORWARD_2(update_fluid_cloth_coupling, FluidClothCouplingId,
                    FluidClothCouplingOptions)
PM_SYSTEM_FORWARD_1(remove_fluid_cloth_coupling, FluidClothCouplingId)
PM_SYSTEM_FORWARD_2(add_soft_body_cloth_coupling,
                    SoftBodyClothCouplingOptions,
                    SoftBodyClothCouplingId &)
PM_SYSTEM_FORWARD_2(update_soft_body_cloth_coupling,
                    SoftBodyClothCouplingId,
                    SoftBodyClothCouplingOptions)
PM_SYSTEM_FORWARD_1(remove_soft_body_cloth_coupling,
                    SoftBodyClothCouplingId)
PM_SYSTEM_FORWARD_2(add_fluid_soft_body_coupling,
                    FluidSoftBodyCouplingOptions,
                    FluidSoftBodyCouplingId &)
PM_SYSTEM_FORWARD_2(update_fluid_soft_body_coupling,
                    FluidSoftBodyCouplingId,
                    FluidSoftBodyCouplingOptions)
PM_SYSTEM_FORWARD_1(remove_fluid_soft_body_coupling,
                    FluidSoftBodyCouplingId)

#undef PM_SYSTEM_VIEW
#undef PM_SYSTEM_REVISION_VIEW
#undef PM_SYSTEM_UNREVISED_FORWARD_3
#undef PM_SYSTEM_UNREVISED_FORWARD_2
#undef PM_SYSTEM_UNREVISED_FORWARD_1
#undef PM_SYSTEM_FORWARD_3
#undef PM_SYSTEM_FORWARD_2
#undef PM_SYSTEM_FORWARD_1

} // namespace parallel_mater::metal
