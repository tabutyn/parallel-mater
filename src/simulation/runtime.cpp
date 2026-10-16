// SPDX-License-Identifier: MIT
#include <simulation/runtime.hpp>
#include "convex_plane_dedup.hpp"
#include "simulation/geometry_dispatch.hpp"
#include "simulation/geometry_constraints_dispatch.hpp"
#include "simulation/geometry_storage.hpp"
#include "simulation/paint_storage.hpp"
#include "simulation/cloth_dispatch.hpp"
#include "simulation/soft_body_dispatch.hpp"
#include "simulation/soft_body_cloth_dispatch.hpp"
#include "simulation/rope_coupling_dispatch.hpp"
#include "simulation/fluid_dispatch.hpp"
#include "simulation/fluid_cloth_dispatch.hpp"
#include "simulation/fluid_rope_dispatch.hpp"
#include "simulation/fluid_smoke_dispatch.hpp"
#include "simulation/smoke_coupling_dispatch.hpp"
#include "simulation/fluid_soft_body_dispatch.hpp"
#include "simulation/geometry_cloth_dispatch.hpp"
#include "simulation/geometry_fluid_dispatch.hpp"
#include "simulation/geometry_smoke_dispatch.hpp"
#include "simulation/geometry_soft_body_dispatch.hpp"
#include "simulation/smoke_grid_dispatch.hpp"
#include "simulation/rope_dispatch.hpp"
#include "simulation/exports.hpp"
#include "simulation/memory_arena.hpp"
#include "simulation/resource_plan.hpp"
#include "simulation/primitives.hpp"
#include "simulation/reference_schedule.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include <vector>

namespace parallel_mater {
namespace {

using simulation::BodyAccumulator;
using simulation::BodyParameters;
using simulation::BvhNode;
using simulation::CachedContactPair;
using simulation::CollisionPlane;
using simulation::Contact;
using simulation::ContactManifold;
using simulation::ContactSchedule;
using simulation::FixedContactProjection;
using simulation::KinematicTarget;
using simulation::LeafPair;
using simulation::PaintFieldResource;
using simulation::PaintRuleResource;
using simulation::RigidConstraintResource;
using simulation::TriangleMeshResource;
using simulation::WorldAabb;
using simulation::axis_enabled;
using simulation::bounds_overlap;
using simulation::closest_on_triangle;
using simulation::component;
using simulation::component_max;
using simulation::component_min;
using simulation::inverse_rotate;
using simulation::k_leaf_pair_cache_slots_per_body;
using simulation::k_max_leaf_pairs_per_body_pair;
using simulation::k_minimum_leaf_pair_cache_slots;
using simulation::k_rigid_leaf_blocks_per_pair;
using simulation::k_shared_rigid_leaf_capacity;
using simulation::k_small_rigid_leaf_body_capacity;
using simulation::k_small_rigid_leaf_pair_capacity;
using simulation::transform_point;

constexpr float k_epsilon = 1.0e-6F;
constexpr std::uint32_t k_invalid_dense = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] Status success() noexcept { return {}; }

[[nodiscard]] Status failure(StatusCode code, const char *message,
                             PmResult error = PM_SUCCESS) noexcept {
    return {code, static_cast<std::int32_t>(error), message};
}

[[nodiscard]] Status backend_failure(PmResult error, const char *message) noexcept {
    return failure(StatusCode::backend_failure, message, error);
}

[[nodiscard]] PmResult copy_bytes(void *destination, const void *source,
    std::size_t size, device::CopyDirection direction,
    device::ExecutionQueue queue) noexcept {
    return queue.services == nullptr ? PM_ERROR_INVALID_STATE
        : queue.services->copy(destination, source, size, direction);
}

[[nodiscard]] PmResult wait_queue(device::ExecutionQueue queue) noexcept {
    return queue.services == nullptr ? PM_ERROR_INVALID_STATE
        : queue.services->wait_idle();
}

[[nodiscard]] PmResult copy_bytes_wait(void *destination, const void *source,
    std::size_t size, device::CopyDirection direction,
    device::ExecutionQueue queue) noexcept {
    const auto result = copy_bytes(destination, source, size, direction, queue);
    return result == PM_SUCCESS ? wait_queue(queue) : result;
}

[[nodiscard]] PmResult clear_bytes(void *destination, int value,
    std::size_t size, device::ExecutionQueue queue) noexcept {
    return queue.services == nullptr ? PM_ERROR_INVALID_STATE
        : queue.services->clear(destination, value, size);
}

[[nodiscard]] float reciprocal_sqrt(float value) noexcept {
    return 1.0F / std::sqrt(value);
}

template <typename T>
[[nodiscard]] Status read_host_span(device::ExecutionQueue queue, T *pointer,
                                    std::size_t count) noexcept {
    if (!pointer || count == 0U) return success();
    const auto result = queue.services->read_host(pointer, count * sizeof(T));
    return result == PM_SUCCESS ? success()
        : backend_failure(result, "host range readback failed");
}

template <typename T>
[[nodiscard]] Status write_host_span(device::ExecutionQueue queue, T *pointer,
                                     std::size_t count) noexcept {
    if (!pointer || count == 0U) return success();
    const auto result = queue.services->write_host(pointer, count * sizeof(T));
    return result == PM_SUCCESS ? success()
        : backend_failure(result, "host range upload failed");
}

template <typename... T>
[[nodiscard]] Status read_host_objects(device::ExecutionQueue queue, T *...pointers) noexcept {
    Status status;
    ((status ? static_cast<void>(status = read_host_span(queue, pointers, 1U)) : void()), ...);
    return status;
}

template <typename... T>
[[nodiscard]] Status write_host_objects(device::ExecutionQueue queue, T *...pointers) noexcept {
    Status status;
    ((status ? static_cast<void>(status = write_host_span(queue, pointers, 1U)) : void()), ...);
    return status;
}

template <typename T>
[[nodiscard]] Status read_device_value(device::ExecutionQueue stream,
                                       const T *source,
                                       T &destination,
                                       const char *message) noexcept {
    PmResult error = copy_bytes(&destination, source,
        sizeof(destination), device::CopyDirection::device_to_host, stream);
    if (error == PM_SUCCESS) error = wait_queue(stream);
    return error == PM_SUCCESS ? success() : backend_failure(error, message);
}

[[nodiscard]] Status read_device_u32(device::ExecutionQueue stream,
                                     const std::uint32_t *source,
                                     std::uint32_t &destination,
                                     const char *message) noexcept {
    return read_device_value(stream, source, destination, message);
}

Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 multiply(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

float dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float length_squared(Vec3 value) noexcept {
    return dot(value, value);
}

float vector_length(Vec3 value) noexcept {
    return sqrtf(length_squared(value));
}

Vec3 normalized_or(Vec3 value, Vec3 fallback) noexcept {
    const float squared = length_squared(value);
    if (squared <= k_epsilon * k_epsilon) {
        return fallback;
    }
    return multiply(value, reciprocal_sqrt(squared));
}

Quaternion normalized_quaternion(Quaternion value) noexcept {
    const float squared = value.x * value.x + value.y * value.y +
                          value.z * value.z + value.w * value.w;
    if (squared <= k_epsilon * k_epsilon) {
        return {};
    }
    const float inverse = reciprocal_sqrt(squared);
    return {value.x * inverse, value.y * inverse, value.z * inverse,
            value.w * inverse};
}

struct CompletionState {
    device::ExecutionQueue execution{};
    device::Event event{};
    const std::uint32_t *fluid_neighbor_overflow{};
    bool acknowledged{};
    Status completion_status{};
    std::function<Status()> on_complete{};

    ~CompletionState() {
        if (event && execution.services != nullptr) {
            execution.services->destroy_event(event);
        }
    }
};

using simulation::TimingStage;
using simulation::k_soft_contact_cleanup_passes;

[[nodiscard]] Status wait_for_completion(
    const std::shared_ptr<CompletionState> &completion) noexcept {
    if (!completion) {
        return success();
    }
    if (completion->acknowledged) {
        return completion->completion_status;
    }
    const PmResult error = completion->execution.services->wait_event(completion->event);
    completion->completion_status = error == PM_SUCCESS
        ? success() : backend_failure(error, "frame completion failed");
    Status debug_status{};
    if (error == PM_SUCCESS && completion->on_complete)
        debug_status = completion->on_complete();
    if (error == PM_SUCCESS && completion->fluid_neighbor_overflow != nullptr) {
        std::uint32_t overflow = 0U;
        const Status overflow_status = read_device_u32(
            completion->execution, completion->fluid_neighbor_overflow, overflow,
            "fluid neighbor overflow readback failed");
        if (!overflow_status) {
            completion->completion_status = overflow_status;
        } else if (overflow != 0U) {
            completion->completion_status = failure(
                StatusCode::capacity_exceeded,
                "fluid neighbor count exceeded maximum_neighbors");
        } else if (!debug_status) {
            completion->completion_status = debug_status;
        }
    } else if (!debug_status) {
        completion->completion_status = debug_status;
    }
    completion->acknowledged = true;
    return completion->completion_status;
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }

[[nodiscard]] bool finite(Vec3 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

[[nodiscard]] bool finite(Quaternion value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

[[nodiscard]] bool finite(const RigidBodyState &state) noexcept {
    return finite(state.position) && finite(state.orientation) &&
           finite(state.linear_velocity) && finite(state.angular_velocity);
}

[[nodiscard]] bool zero(Vec3 value) noexcept {
    return value.x == 0.0F && value.y == 0.0F && value.z == 0.0F;
}

[[nodiscard]] Status validate_body_options(const RigidBodyOptions &options) noexcept {
    if (!finite(options.initial_state)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body state must contain finite values");
    }
    const float quaternion_size =
        options.initial_state.orientation.x * options.initial_state.orientation.x +
        options.initial_state.orientation.y * options.initial_state.orientation.y +
        options.initial_state.orientation.z * options.initial_state.orientation.z +
        options.initial_state.orientation.w * options.initial_state.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "rigid body orientation must be nonzero");
    }
    if (options.motion == MotionType::dynamic &&
        (!finite(options.mass) || options.mass <= 0.0F)) {
        return failure(StatusCode::invalid_argument,
                       "dynamic body mass must be finite and positive");
    }
    if (!finite(options.inertia_diagonal) ||
        (!zero(options.inertia_diagonal) &&
         (options.inertia_diagonal.x <= 0.0F ||
          options.inertia_diagonal.y <= 0.0F ||
          options.inertia_diagonal.z <= 0.0F))) {
        return failure(StatusCode::invalid_argument,
                       "inertia must be all zero or all positive");
    }
    if (!finite(options.friction) || options.friction < 0.0F ||
        !finite(options.restitution) || options.restitution < 0.0F ||
        options.restitution > 1.0F || !finite(options.linear_damping) ||
        options.linear_damping < 0.0F || !finite(options.angular_damping) ||
        options.angular_damping < 0.0F ||
        !finite(options.maximum_linear_speed) ||
        options.maximum_linear_speed <= 0.0F ||
        !finite(options.maximum_angular_speed) ||
        options.maximum_angular_speed <= 0.0F ||
        !finite(options.collision_margin) || options.collision_margin <= 0.0F) {
        return failure(StatusCode::invalid_argument,
                       "rigid body material and limits are invalid");
    }
    return success();
}

[[nodiscard]] bool valid_constraint_mask(std::uint32_t axes) noexcept {
    return (axes & ~rigid_constraint_all_axes) == 0U;
}

[[nodiscard]] Status validate_constraint_options(
    const RigidConstraintOptions &options) noexcept {
    if (static_cast<std::uint32_t>(options.type) >
        static_cast<std::uint32_t>(RigidConstraintType::motor))
        return failure(StatusCode::invalid_argument,
                       "rigid constraint type is invalid");
    if (options.body_a == options.body_b)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint bodies must be distinct");
    if (!finite(options.local_anchor_a) || !finite(options.local_anchor_b) ||
        !finite(options.local_orientation_a) ||
        !finite(options.local_orientation_b))
        return failure(StatusCode::invalid_argument,
                       "rigid constraint frames must contain finite values");
    const auto quaternion_size = [](Quaternion value) {
        return value.x * value.x + value.y * value.y + value.z * value.z +
               value.w * value.w;
    };
    if (quaternion_size(options.local_orientation_a) <= k_epsilon * k_epsilon ||
        quaternion_size(options.local_orientation_b) <= k_epsilon * k_epsilon)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint orientations must be nonzero");
    if (!valid_constraint_mask(options.linear_limits.axes) ||
        !valid_constraint_mask(options.angular_limits.axes) ||
        !valid_constraint_mask(options.linear_springs.axes) ||
        !valid_constraint_mask(options.angular_springs.axes))
        return failure(StatusCode::invalid_argument,
                       "rigid constraint axis mask is invalid");
    const auto valid_limits = [](const RigidConstraintLimitOptions &limits) {
        if (!finite(limits.lower) || !finite(limits.upper)) return false;
        for (std::uint32_t axis = 0U; axis < 3U; ++axis)
            if (axis_enabled(limits.axes, axis) &&
                component(limits.lower, axis) > component(limits.upper, axis))
                return false;
        return true;
    };
    const auto valid_springs = [](const RigidConstraintSpringOptions &springs) {
        if (!finite(springs.stiffness) || !finite(springs.damping)) return false;
        for (std::uint32_t axis = 0U; axis < 3U; ++axis)
            if (axis_enabled(springs.axes, axis) &&
                (component(springs.stiffness, axis) < 0.0F ||
                 component(springs.damping, axis) < 0.0F)) return false;
        return true;
    };
    if (!valid_limits(options.linear_limits) ||
        !valid_limits(options.angular_limits))
        return failure(StatusCode::invalid_argument,
                       "rigid constraint limits are invalid");
    if (!valid_springs(options.linear_springs) ||
        !valid_springs(options.angular_springs))
        return failure(StatusCode::invalid_argument,
                       "rigid constraint springs are invalid");
    if (!finite(options.motor.linear_target_velocity) ||
        !finite(options.motor.angular_target_velocity) ||
        !finite(options.motor.linear_maximum_impulse) ||
        options.motor.linear_maximum_impulse < 0.0F ||
        !finite(options.motor.angular_maximum_impulse) ||
        options.motor.angular_maximum_impulse < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint motor is invalid");
    if (!finite(options.breaking_impulse_threshold) ||
        options.breaking_impulse_threshold < 0.0F ||
        options.solver_iterations == 0U || options.solver_iterations > 64U)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint breaking threshold or iterations are invalid");
    return success();
}

[[nodiscard]] BodyParameters make_parameters(
    const RigidBodyOptions &options, const TriangleMeshResource &mesh) noexcept {
    const Vec3 inertia = zero(options.inertia_diagonal)
                             ? multiply(mesh.unit_inertia, options.mass)
                             : options.inertia_diagonal;
    const float inverse_mass =
        options.motion == MotionType::dynamic ? 1.0F / options.mass : 0.0F;
    const Vec3 inverse_inertia = options.motion == MotionType::dynamic
                                     ? Vec3{1.0F / inertia.x, 1.0F / inertia.y,
                                            1.0F / inertia.z}
                                     : Vec3{};
    return {options.motion,
            options.mesh,
            inverse_mass,
            inverse_inertia,
            options.friction,
            options.restitution,
            options.linear_damping,
            options.angular_damping,
            options.maximum_linear_speed,
            options.maximum_angular_speed,
            options.collision_margin,
            options.user_data};
}

template <typename T>
[[nodiscard]] Status allocate_managed(T *&pointer, std::size_t count) noexcept {
    pointer = nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return failure(StatusCode::out_of_memory, "simulation allocation size overflow");
    if (count == 0U) {
        pointer = nullptr;
        return success();
    }
    if (simulation::has_active_arena()) {
        pointer = static_cast<T *>(simulation::allocate_bound(
            sizeof(T) * count, alignof(T),
            simulation::MemoryClass::persistent));
        return pointer != nullptr
            ? success()
            : failure(StatusCode::out_of_memory,
                      "bound persistent simulation memory exhausted");
    }
    pointer = nullptr;
    return failure(StatusCode::invalid_argument,
                   "simulation allocation requires a bound memory arena");
}

template <typename T> void release_managed(T *&pointer) noexcept {
    if (pointer != nullptr) {
        // Resource owners retire queued work before releasing their storage.
        // Locate the owning arena: geometry can live outside the active arena.
        (void)simulation::release_bound(pointer);
        pointer = nullptr;
    }
}

// Keep the host slot allocation when retiring an owner. Cleanup must not
// require fresh host memory while recovering from an allocation failure.
template<class T> void retire_owner(T &owner) noexcept {
    static_assert(std::is_nothrow_default_constructible_v<T>);
    auto generation = owner.generation + 1U;
    if (!generation) generation = 1U;
    std::destroy_at(&owner);
    std::construct_at(&owner);
    owner.generation = generation;
}

template <typename T>
[[nodiscard]] Status allocate_scratch(T *&pointer, std::size_t count) noexcept {
    pointer = nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return failure(StatusCode::out_of_memory, "simulation scratch size overflow");
    if (count == 0U) {
        pointer = nullptr;
        return success();
    }
    if (!simulation::has_active_arena()) return allocate_managed(pointer, count);
    pointer = static_cast<T *>(simulation::allocate_bound(
        sizeof(T) * count, alignof(T), simulation::MemoryClass::scratch));
    return pointer != nullptr
        ? success()
        : failure(StatusCode::out_of_memory,
                  "bound scratch simulation memory exhausted");
}

struct StorageAllocator {
    static constexpr bool materialize = true;
    template<class T> Status operator()(T *&pointer, std::size_t count,
                                       simulation::MemoryClass kind) noexcept {
        return kind == simulation::MemoryClass::persistent
            ? allocate_managed(pointer, count) : allocate_scratch(pointer, count);
    }
};

using simulation::k_fluid_body_buckets;

struct FluidBodyImpulse {
    Vec3 linear{};
    Vec3 angular{};
    std::uint32_t body{k_invalid_dense};
};

struct ShapeMatrix {
    Vec3 columns[3]{};
};

struct DeformableNeighbor {
    std::uint32_t index{};
    float rest_length{};
    float compliance{};
    std::uint32_t bond{};
};

struct SoftBodyNeighbor {
    std::uint32_t index{};
    float rest_length{};
};

struct ClothBodyCorrection {
    Vec3 offset{};
    Vec3 impulse{};
    Vec3 contact{};
    float support_radius{};
    float weight_sum{};
    std::uint32_t vertices[3]{};
    bool active{};
};

struct ClothSeam {
    std::uint32_t corners[4]{}; // matching endpoints on the two incident faces
    std::uint32_t bond{};
    std::uint32_t bending{k_invalid_dense};
};

struct ClothStorage {
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t vertex_count{};
    std::uint32_t vertex_capacity{};
    std::uint32_t index_count{};
    float thickness{};
    float velocity_damping{};
    float contact_friction{};
    float break_strain{};
    std::uint32_t fracture_persistence_substeps{};
    float impact_break_impulse{};
    std::uint32_t solver_iterations{};
    bool preserve_volume{};
    float target_volume{};
    float volume_compliance{};
    float orientation{1.0F};
    Vec3 *positions{};
    Vec3 *scratch{};
    Vec3 *previous{};
    Vec3 *velocities{};
    float *inverse_masses{};
    std::uint32_t *indices{};
    std::uint32_t *source_indices{};
    std::uint32_t *vertex_sources{};
    std::uint8_t *free_triangle_nodes{};
    std::vector<ClothSeam> seams;
    std::vector<std::array<std::uint32_t, 2>> bond_corners;
    std::vector<float> source_inverse_masses;
    std::vector<std::uint32_t> source_degrees;
    std::vector<std::uint8_t> topology_active;
    float stretch_compliance{}, bending_compliance{};
    Vec3 *surface_positions{};
    std::uint32_t *surface_triangle_indices{};
    std::uint32_t *triangle_bonds{};
    ClothBond *bonds{};
    std::uint8_t *bond_active{};
    std::uint8_t *bond_damage{};
    std::uint32_t bond_count{};
    std::uint32_t *offsets{};
    DeformableNeighbor *neighbors{};
    std::uint32_t neighbor_count{};
    std::size_t neighbor_capacity{};
    FluidBodyImpulse *body_impulses{};
    Vec3 *rigid_contact_forces{};
    ClothBodyCorrection *body_corrections{};
    Vec3 *volume_gradients{};
    std::uint32_t *volume_corner_offsets{};
    std::uint32_t *volume_corner_indices{};
    Vec3 *fluid_forces{};
    Vec3 *soft_body_forces{};
    Vec3 *rope_forces{};
    Vec3 *smoke_forces{};
    float *volume_lambda{};
    std::uint32_t *count{};

    void release() noexcept {
        release_managed(positions);
        release_managed(scratch);
        release_managed(previous);
        release_managed(velocities);
        release_managed(inverse_masses);
        release_managed(indices);
        release_managed(source_indices);
        release_managed(vertex_sources);
        release_managed(free_triangle_nodes);
        release_managed(surface_positions);
        release_managed(surface_triangle_indices);
        release_managed(triangle_bonds);
        release_managed(bonds);
        release_managed(bond_active);
        release_managed(bond_damage);
        release_managed(offsets);
        release_managed(neighbors);
        release_managed(body_impulses);
        release_managed(rigid_contact_forces);
        release_managed(body_corrections);
        release_managed(volume_gradients);
        release_managed(volume_corner_offsets);
        release_managed(volume_corner_indices);
        release_managed(fluid_forces);
        release_managed(soft_body_forces);
        release_managed(rope_forces);
        release_managed(smoke_forces);
        release_managed(volume_lambda);
        release_managed(count);
    }
    ~ClothStorage() { release(); }
};

// Runs only at an idle frame boundary and only rebuilds when a bond changed.
// Split vertex fans across failed seams. Each new node inherits its parent's
// position/velocity; incident-face mass shares preserve total mass/momentum.
// No triangle is discarded or fitted to remote vertices after a tear.
static Status rebuild_cloth_topology(ClothStorage &cloth, device::ExecutionQueue queue,
                                     bool initial = false) {
    if (!cloth.source_indices) return success();
    if (!initial) {
        const auto status = read_host_span(queue, cloth.bond_active, cloth.bond_count);
        if (!status) return status;
    }
    if (!initial && std::equal(cloth.topology_active.begin(),
                               cloth.topology_active.end(), cloth.bond_active))
        return success();
    try {
        const auto corners = cloth.index_count;
        auto count = cloth.vertex_count;
        std::vector<std::uint8_t> active(cloth.bond_active, cloth.bond_active + cloth.bond_count);
        std::vector<std::uint32_t> indices(cloth.indices, cloth.indices + corners);
        std::vector<std::array<std::uint32_t, 2>> copies;
        std::vector<std::uint32_t> parent(corners);
        for (std::uint32_t i = 0; i < corners; ++i) parent[i] = i;
        const auto root = [&](std::uint32_t i) {
            while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
            return i;
        };
        for (const auto &seam : cloth.seams) {
            if (active[seam.bond]) {
                parent[root(seam.corners[2])] = root(seam.corners[0]);
                parent[root(seam.corners[3])] = root(seam.corners[1]);
            } else if (seam.bending != k_invalid_dense) {
                active[seam.bending] = 0U;
            }
        }
        std::vector<std::uint32_t> nodes(corners, k_invalid_dense);
        std::vector<bool> used(cloth.vertex_capacity);
        std::vector<std::uint32_t> degrees(cloth.vertex_capacity);
        std::vector<std::uint8_t> free_nodes(cloth.vertex_capacity);
        for (std::uint32_t corner = 0; corner < corners; ++corner) {
            const auto group = root(corner);
            auto &node = nodes[group];
            if (node == k_invalid_dense) {
                const auto old = cloth.indices[corner];
                node = old;
                if (used[old]) {
                    if (count == cloth.vertex_capacity)
                        return failure(StatusCode::capacity_exceeded, "cloth split capacity exhausted");
                    node = count++;
                    copies.push_back({node,old});
                }
                used[node] = true;
            }
            indices[corner] = node;
            ++degrees[node];
        }
        for (std::uint32_t c = 0; c < corners; c += 3U) {
            bool detached = true;
            for (std::uint32_t k = 0; k < 3U; ++k)
                detached &= degrees[indices[c+k]] == 1U &&
                    cloth.source_inverse_masses[cloth.source_indices[c+k]] > 0.0F;
            if (detached) for (std::uint32_t k = 0; k < 3U; ++k)
                free_nodes[indices[c+k]] = 1U;
        }
        std::vector<std::vector<DeformableNeighbor>> adjacency(count);
        std::unordered_set<std::uint64_t> edges;
        const auto link = [&](std::uint32_t a, std::uint32_t b, float rest,
                              float compliance, std::uint32_t bond) {
            const auto key = (static_cast<std::uint64_t>(std::min(a,b)) << 32U) |
                             std::max(a,b);
            if (a == b || !edges.insert(key).second) return;
            adjacency[a].push_back({b, rest, compliance, bond});
            adjacency[b].push_back({a, rest, compliance, bond});
        };
        for (std::uint32_t corner = 0; corner < corners; ++corner) {
            const auto next = corner / 3U * 3U + (corner + 1U) % 3U;
            link(indices[corner], indices[next],
                 cloth.bonds[cloth.triangle_bonds[corner]].rest_length,
                 cloth.stretch_compliance, k_invalid_dense);
        }
        std::vector<ClothBond> bonds(cloth.bonds, cloth.bonds + cloth.bond_count);
        for (std::uint32_t i = 0; i < cloth.bond_count; ++i) {
            auto &bond = bonds[i];
            bond.first = indices[cloth.bond_corners[i][0]];
            bond.second = indices[cloth.bond_corners[i][1]];
            if (bond.bending && active[i])
                link(bond.first, bond.second, bond.rest_length,
                     cloth.bending_compliance, i);
        }
        // All allocations have succeeded; commit the prepared graph atomically
        // with respect to API calls. No device work is in flight at this point.
        for (const auto &copy : copies) {
            const auto node = copy[0], old = copy[1];
            auto status = read_host_objects(queue, cloth.positions + old,
                cloth.previous + old, cloth.velocities + old);
            if (!status) return status;
            cloth.positions[node] = cloth.positions[old];
            cloth.previous[node] = cloth.previous[old];
            cloth.scratch[node] = cloth.positions[old];
            cloth.velocities[node] = cloth.velocities[old];
            cloth.vertex_sources[node] = cloth.vertex_sources[old];
            cloth.rigid_contact_forces[node] = {};
            cloth.soft_body_forces[node] = {};
            if (cloth.fluid_forces) cloth.fluid_forces[node] = {};
            status = write_host_objects(queue, cloth.positions + node,
                cloth.previous + node, cloth.scratch + node, cloth.velocities + node,
                cloth.rigid_contact_forces + node, cloth.soft_body_forces + node);
            if (!status) return status;
            if (cloth.fluid_forces) {
                status = write_host_objects(queue, cloth.fluid_forces + node);
                if (!status) return status;
            }
        }
        std::copy(indices.begin(), indices.end(), cloth.indices);
        std::copy(bonds.begin(), bonds.end(), cloth.bonds);
        std::copy(active.begin(), active.end(), cloth.bond_active);
        std::copy_n(free_nodes.data(), count, cloth.free_triangle_nodes);
        std::uint32_t offset = 0;
        for (std::uint32_t node = 0; node < count; ++node) {
            const auto source = cloth.vertex_sources[node];
            cloth.inverse_masses[node] = degrees[node] == 0 ? cloth.source_inverse_masses[source] :
                cloth.source_inverse_masses[source] *
                static_cast<float>(cloth.source_degrees[source]) / degrees[node];
            cloth.offsets[node] = offset;
            for (const auto &neighbor : adjacency[node]) cloth.neighbors[offset++] = neighbor;
        }
        cloth.offsets[count] = offset;
        cloth.neighbor_count = offset;
        cloth.vertex_count = count;
        *cloth.count = count;
        cloth.topology_active.swap(active);
        const auto status = write_host_span(queue, cloth.bond_active, cloth.bond_count);
        if (!status) return status;
    } catch (...) {
        return failure(StatusCode::out_of_memory, "failed to split cloth topology");
    }
    return success();
}

struct SoftSurfaceInfluence {
    std::uint32_t corner{};
    float factor{};
};

struct SoftBodyStorage {
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t node_count{};
    std::uint32_t bond_count{};
    std::uint32_t neighbor_count{};
    std::uint32_t surface_vertex_count{};
    std::uint32_t surface_index_count{};
    float node_radius{};
    float stretch_compliance{};
    float velocity_damping{};
    float spring_damping{};
    float contact_friction{};
    float shape_matching_stiffness{};
    float shape_maximum_projection{};
    float maximum_projection_fraction{};
    float constraint_velocity_response{};
    float maximum_speed{};
    float movable_mass{};
    Vec3 shape_rest_center{};
    ShapeMatrix shape_inverse_rest{};
    std::uint32_t solver_iterations{};
    Vec3 *positions{};
    Vec3 *rest_positions{};
    Vec3 *scratch{};
    Vec3 *previous{};
    Vec3 *velocities{};
    Vec3 *velocity_scratch{};
    float *inverse_masses{};
    SoftBodyBond *bonds{};
    std::uint8_t *bond_active{};
    std::uint32_t *offsets{};
    DeformableNeighbor *neighbors{};
    SoftBodyNeighbor *warp_neighbors{};
    float *minimum_rest_lengths{};
    Vec3 *surface_rest_positions{};
    Vec3 *surface_positions{};
    std::uint32_t *surface_indices{};
    SoftBodySurfaceBinding *surface_bindings{};
    Vec3 *surface_corner_corrections{};
    std::uint32_t *surface_node_offsets{};
    SoftSurfaceInfluence *surface_node_influences{};
    FluidBodyImpulse *body_impulses{};
    Vec3 *body_position_corrections{};
    Vec3 *cloth_forces{};
    Vec3 *fluid_forces{};
    Vec3 *rope_forces{};
    Vec3 *rigid_contact_forces{};
    Vec3 *contact_normals{};
    Vec3 *contact_arms{};
    Vec3 *contact_momentum_delta{};
    Vec3 *contact_friction_delta{};
    float *contact_normal_delta{};
    Vec3 *predicted_momentum{};
    Quaternion *shape_orientation{};
    std::uint32_t *dynamic_contact_flag{};
    std::uint32_t *contact_count{};
    std::uint32_t *count{};

    void release() noexcept {
        release_managed(positions);
        release_managed(rest_positions);
        release_managed(scratch);
        release_managed(previous);
        release_managed(velocities);
        release_managed(velocity_scratch);
        release_managed(inverse_masses);
        release_managed(bonds);
        release_managed(bond_active);
        release_managed(offsets);
        release_managed(neighbors);
        release_managed(warp_neighbors);
        release_managed(minimum_rest_lengths);
        release_managed(surface_rest_positions);
        release_managed(surface_positions);
        release_managed(surface_indices);
        release_managed(surface_bindings);
        release_managed(surface_corner_corrections);
        release_managed(surface_node_offsets);
        release_managed(surface_node_influences);
        release_managed(body_impulses);
        release_managed(body_position_corrections);
        release_managed(cloth_forces);
        release_managed(fluid_forces);
        release_managed(rope_forces);
        release_managed(rigid_contact_forces);
        release_managed(contact_normals);
        release_managed(contact_arms);
        release_managed(contact_momentum_delta);
        release_managed(contact_friction_delta);
        release_managed(contact_normal_delta);
        release_managed(predicted_momentum);
        release_managed(shape_orientation);
        release_managed(dynamic_contact_flag);
        release_managed(contact_count);
        release_managed(count);
    }
    ~SoftBodyStorage() { release(); }
};

struct FluidClothCouplingResource {
    FluidClothCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};

struct FluidSoftContact {
    std::uint32_t nodes[12]{};
    float weights[12]{};
    std::uint32_t count{};
    Vec3 normal{}, relative_velocity{};
    float penetration{};
};

struct FluidSoftCouplingStorage {
    FluidSoftBodyCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    float orientation{1.0F};
    FluidSoftContact *contacts{};
    std::uint32_t *counts{};
    Vec3 *position_deltas{}, *impulses{}, *previous_surface{}, *bounds{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    BvhNode *tree{};
    std::uint32_t *triangle_order{}, *parents{}, *ready{};
    std::uint32_t tree_count{};
    void release() noexcept {
        release_managed(contacts);
        release_managed(counts);
        release_managed(position_deltas);
        release_managed(impulses);
        release_managed(previous_surface);
        release_managed(bounds);
        release_managed(contact_count);
        release_managed(maximum_penetration);
        release_managed(tree);
        release_managed(triangle_order);
        release_managed(parents);
        release_managed(ready);
    }
    ~FluidSoftCouplingStorage() { release(); }
};

struct SoftClothContact {
    std::uint32_t vertices[3]{};
    float weights[3]{};
    Vec3 position_impulse{};
    Vec3 velocity_impulse{};
    float soft_inverse_mass_fraction{};
    bool active{};
};

struct SoftClothCouplingStorage {
    SoftBodyClothCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    SoftClothContact *contacts{};
    std::uint32_t *cloth_contact_counts{};
    void release() noexcept {
        release_managed(contacts);
        release_managed(cloth_contact_counts);
    }
    ~SoftClothCouplingStorage() { release(); }
};

struct FluidContactSample {
    Vec3 position{};
    Vec3 normal{};
    float normal_impulse{};
    std::uint32_t body{k_invalid_dense};
};

struct FluidMovingContact {
    Vec3 normal{};
    Vec3 point{};
    float penetration{};
    std::uint32_t body{k_invalid_dense};
};

struct FluidStorage {
    FluidOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t *count{};
    Vec3 *positions{};
    Vec3 *velocities{};
    Vec3 *previous{};
    std::uint32_t *ids{};
    float *foam{};
    float *temperatures{};
    float *foam_source{};
    Vec3 *next_positions{};
    Vec3 *next_velocities{};
    std::uint32_t *next_ids{};
    float *next_foam{};
    float *next_temperatures{};
    std::uint8_t *keep{};
    std::uint32_t *selected{};
    std::uint64_t *keys[2]{};
    std::uint32_t *indices[2]{};
    Vec3 *forces{};
    FluidBodyImpulse *body_impulses{};
    FluidMovingContact *moving_contacts{};
    FluidContactSample *contact_samples{};
    std::uint8_t *contact_flags{};
    std::uint8_t *hit_box_flags{};
    FluidContactSample *next_contact_samples{};
    std::uint8_t *next_contact_flags{};
    std::uint32_t *contact_count{};
    std::uint32_t *contact_offset{};
    std::uint8_t *sort_workspace{};
    std::size_t sort_workspace_size{};
    std::uint8_t *select_workspace{};
    std::size_t select_workspace_size{};
    std::uint32_t next_id{};
    std::uint64_t initial_count{};
    std::uint64_t emitted_count{};

    ~FluidStorage() {
        release_managed(count);
        release_managed(positions);
        release_managed(velocities);
        release_managed(previous);
        release_managed(ids);
        release_managed(foam);
        release_managed(temperatures);
        release_managed(foam_source);
        release_managed(next_positions);
        release_managed(next_velocities);
        release_managed(next_ids);
        release_managed(next_foam);
        release_managed(next_temperatures);
        release_managed(keep);
        release_managed(selected);
        release_managed(keys[0]);
        release_managed(keys[1]);
        release_managed(indices[0]);
        release_managed(indices[1]);
        release_managed(forces);
        release_managed(body_impulses);
        release_managed(moving_contacts);
        release_managed(contact_samples);
        release_managed(contact_flags);
        release_managed(hit_box_flags);
        release_managed(next_contact_samples);
        release_managed(next_contact_flags);
        release_managed(contact_count);
        release_managed(contact_offset);
        release_managed(sort_workspace);
        release_managed(select_workspace);
    }
};

struct ParticleSourceData {
    Vec3 *points{};
    std::uint8_t *vacant{};
    std::uint32_t *capacity_misses{};
    std::uint32_t count{};
    float spacing{};
    ~ParticleSourceData() {
        release_managed(points);
        release_managed(vacant);
        release_managed(capacity_misses);
    }
};

struct ParticleSourceSlot {
    ParticleSourceOptions options{};
    std::uint32_t generation{1U};
    std::unique_ptr<ParticleSourceData> data{};
    bool alive{};
};

struct DestroyPlaneSlot {
    ParticleDestroyPlaneOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};

simulation::FluidDispatchConstants slang_fluid_constants(
    const FluidOptions &options, Vec3 gravity, float timestep,
    std::uint32_t count) noexcept {
    return {timestep,
            {gravity.x, gravity.y, gravity.z},
            count,
            options.particle_radius,
            options.support_radius,
            options.repulsion,
            options.viscosity,
            options.normal_damping,
            options.velocity_damping,
            options.maximum_speed,
            options.maximum_pair_acceleration,
            options.rest_density,
            options.maximum_neighbors};
}

} // namespace

namespace {
#include "simulation/rope_storage.hpp"
#include "simulation/rope_cloth_storage.hpp"
#include "simulation/fluid_rope_storage.hpp"
#include "simulation/fluid_smoke_storage.hpp"
#include "simulation/smoke_storage.hpp"
[[nodiscard]] simulation::SmokeDispatchOptions smoke_dispatch_options(
    const SmokeOptions &options) noexcept {
    return {options.capacity, options.particle_radius, options.lifetime,
            options.rest_number_density, options.pressure_stiffness,
            options.maximum_speed};
}

[[nodiscard]] simulation::SmokeDomainDispatchOptions smoke_domain_options(
    const SmokeOptions &options) noexcept {
    simulation::SmokeDomainDispatchOptions result{};
    result.capacity = options.capacity;
    result.emitter_center[0] = options.emitter_center.x;
    result.emitter_center[1] = options.emitter_center.y;
    result.emitter_center[2] = options.emitter_center.z;
    result.emitter_half_extents[0] = options.emitter_half_extents.x;
    result.emitter_half_extents[1] = options.emitter_half_extents.y;
    result.initial_velocity[0] = options.initial_velocity.x;
    result.initial_velocity[1] = options.initial_velocity.y;
    result.initial_velocity[2] = options.initial_velocity.z;
    result.wind[0] = options.wind.x;
    result.wind[1] = options.wind.y;
    result.wind[2] = options.wind.z;
    result.particles_per_second = options.particles_per_second;
    result.lifetime = options.lifetime;
    result.particle_radius = options.particle_radius;
    result.buoyancy = options.buoyancy;
    result.response = options.response;
    result.rest_number_density = options.rest_number_density;
    result.pressure_stiffness = options.pressure_stiffness;
    result.viscosity = options.viscosity;
    result.vorticity_confinement = options.vorticity_confinement;
    result.maximum_speed = options.maximum_speed;
    result.grid_resolution = options.grid_resolution;
    result.grid_vertical_resolution = options.grid_vertical_resolution;
    result.grid_pressure_iterations = options.grid_pressure_iterations;
    result.grid_kinematic_viscosity = options.grid_kinematic_viscosity;
    result.grid_les_coefficient = options.grid_les_coefficient;
    result.grid_pressure_tolerance = options.grid_pressure_tolerance;
    result.grid_minimum[0] = options.grid_minimum.x;
    result.grid_minimum[1] = options.grid_minimum.y;
    result.grid_minimum[2] = options.grid_minimum.z;
    result.grid_edge_length = options.grid_edge_length;
    return result;
}

[[nodiscard]] simulation::SmokeGridDispatchField smoke_dispatch_grid(
    const SmokeGridStorage &grid) noexcept {
    simulation::SmokeGridDispatchField result{};
    result.resolution = grid.resolution;
    result.height = grid.height;
    result.minimum[0] = grid.minimum.x;
    result.minimum[1] = grid.minimum.y;
    result.minimum[2] = grid.minimum.z;
    result.spacing = grid.spacing;
    result.velocity = grid.velocity;
    result.density = grid.density;
    result.pressure = grid.pressure[0];
    result.face_velocity[0] = grid.face_velocity[0][0];
    result.face_velocity[1] = grid.face_velocity[1][0];
    result.face_velocity[2] = grid.face_velocity[2][0];
    result.vorticity = grid.vorticity;
    result.strain = grid.strain;
    return result;
}
} // namespace

struct FrameToken::Impl {
    std::shared_ptr<CompletionState> completion{};
};

struct SimulationRuntime::Impl {
    using RopeSoftCoupling = RopeSoftBodyCouplingStorage;
    struct Slot {
        std::uint32_t generation{1U};
        std::uint32_t dense_index{k_invalid_dense};
        bool alive{};
    };

    SimulationOptions options{};
    device::ExecutionQueue execution{};
    std::uint32_t rigid_body_count{};
    std::uint32_t rigid_constraint_count{};
    std::uint32_t triangle_mesh_count{};
    std::uint32_t fluid_count{};
    std::uint64_t emitted_particle_count{};
    std::uint64_t destroyed_particle_count{};
    std::uint64_t spawn_capacity_miss_count{};
    std::uint32_t current_state{};
    std::uint64_t frame_index{};
    std::uint64_t revision{};
    std::uint32_t rigid_solve_kernels_per_substep{1U};
    std::uint32_t soft_cloth_kernels_per_substep{};
    std::vector<Slot> slots{};
    std::vector<std::unique_ptr<FluidStorage>> fluids{};
    std::vector<std::unique_ptr<SmokeStorage>> smokes{};
    std::vector<std::unique_ptr<FluidSmokeCouplingStorage>> fluid_smoke_couplings{};
    std::vector<std::unique_ptr<SmokeSoftBodyCouplingSlot>> smoke_soft_body_couplings{};
    std::vector<std::unique_ptr<SmokeClothCouplingSlot>> smoke_cloth_couplings{};
    std::vector<std::unique_ptr<SmokeRopeCouplingSlot>> smoke_rope_couplings{};
    std::vector<std::unique_ptr<SmokeRigidCouplingSlot>> smoke_rigid_couplings{};
    std::uint64_t boiled_particle_count{};
    std::vector<std::unique_ptr<ClothStorage>> cloths{};
    std::vector<std::unique_ptr<SoftBodyStorage>> soft_bodies{};
    std::vector<std::unique_ptr<RopeStorage>> ropes{};
    std::vector<std::unique_ptr<FluidRopeCouplingStorage>> fluid_rope_couplings{};
    std::vector<std::unique_ptr<RopeSoftBodyCouplingStorage>> rope_soft_body_couplings{};
    std::vector<RopeClothCouplingStorage> rope_cloth_couplings{};
    std::vector<FluidClothCouplingResource> fluid_cloth_couplings{};
    std::vector<std::unique_ptr<FluidSoftCouplingStorage>> fluid_soft_couplings{};
    std::vector<std::unique_ptr<SoftClothCouplingStorage>> soft_cloth_couplings{};
    std::vector<ParticleSourceSlot> particle_sources{};
    std::vector<DestroyPlaneSlot> destroy_planes{};
    PaintFieldResource *paint_fields{};
    PaintRuleResource *paint_rules{};
    std::uint32_t paint_rule_count{};
    std::uint32_t *fluid_neighbor_overflow{};
    std::uint32_t *fluid_maximum_neighbor_count{};
    BodyParameters *parameters{};
    BodyAccumulator *accumulators{};
    KinematicTarget *targets{};
    RigidConstraintResource *rigid_constraints{};
    FixedContactProjection *fixed_contact_projection{};
    RigidBodyId *ids{};
    RigidBodyState *states[2]{};
    RigidBodyState *render_previous_states{};
    Vec3 *debug_applied_forces{};
    Vec3 *debug_applied_torques{};
    PhysicsDebugRigidSample *debug_rigid_samples{};
    RigidBodyState *fluid_previous_states{};
    ContactManifold *rigid_manifolds{};
    simulation::GeometryConstraintBody *avbd_bodies{};
    CachedContactPair *rigid_contact_cache{};
    std::uint32_t *rigid_contact_cache_slots{};
    std::uint64_t rigid_contact_epoch{};
    std::uint64_t rigid_contact_revision{};
    ContactSchedule *rigid_contact_schedule{};
    std::uint32_t rigid_contact_grid_limit{1U};
    std::uint32_t rigid_contact_block_size{8U};
    std::uint32_t *rigid_contact_event_offsets{};
    WorldAabb *rigid_world_bounds{};
    std::uint8_t *hit_box_rigid_flags{};
    WorldAabb *fluid_body_bounds{};
    unsigned long long *fluid_body_masks{};
    unsigned long long *fluid_global_body_masks{};
    std::uint32_t *fluid_body_contact_flags{};
    ContactEvent *fluid_contact_events{};
    std::uint32_t *fluid_contact_count{};
    std::uint32_t *fluid_contact_overflow{};
    std::uint8_t *rigid_active_pair_flags{};
    std::uint32_t *rigid_active_pairs{};
    std::uint32_t *rigid_active_pair_count{};
    std::uint8_t *rigid_broad_phase_workspace{};
    std::size_t rigid_broad_phase_workspace_size{};
    LeafPair *rigid_leaf_pairs{};
    ContactManifold *rigid_leaf_manifolds{};
    std::uint32_t *rigid_leaf_pair_counts{};
    std::uint32_t rigid_leaf_pair_slot_capacity{};
    std::uint32_t rigid_leaf_pairs_per_slot{};
    std::size_t rigid_leaf_pair_capacity{};
    RigidContactEvent *rigid_contact_events{};
    std::uint32_t *rigid_contact_count{};
    std::uint32_t rigid_contact_capacity{};
    TriangleMeshResource *meshes{};
    std::shared_ptr<CompletionState> frame{};
    std::vector<device::Event> timing_events{};
    std::vector<TimingStage> timing_stages{};
    std::vector<std::uint32_t> timing_launch_counts{};
    std::size_t timing_boundary_count{};
    std::uint64_t timing_frame_index{};
    bool timing_available{};
    std::vector<PhysicsDebugFrame> debug_frames{};
    std::size_t debug_next_frame{};
    std::size_t debug_frame_count{};

    [[nodiscard]] Status record_debug_frame(
        StepOptions step_options, device::ExecutionQueue queue) noexcept {
        if (options.physics_debug.frame_capacity == 0U ||
            frame_index % options.physics_debug.frame_stride != 0U)
            return success();
        try {
            PhysicsDebugFrame &output = debug_frames[debug_next_frame];
            const device::ExecutionQueue stream = queue;
            const auto download = [stream](void *destination,
                                           const void *source,
                                           std::size_t bytes) -> Status {
                if (bytes == 0U) return success();
                auto error = copy_bytes(destination, source, bytes,
                                             device::CopyDirection::device_to_host, stream);
                if (error == PM_SUCCESS)
                    error = wait_queue(stream);
                return error == PM_SUCCESS ? success()
                    : backend_failure(error, "physics debug readback failed");
            };
            output.frame_index = frame_index;
            output.timestep = step_options.timestep;
            output.gravity = step_options.gravity;
            output.maximum_fluid_neighbor_count = 0U;
            if (fluid_count != 0U) {
                const auto status = download(&output.maximum_fluid_neighbor_count,
                    fluid_maximum_neighbor_count, sizeof(output.maximum_fluid_neighbor_count));
                if (!status) return status;
            }
            output.rigid_bodies.resize(rigid_body_count);
            if (rigid_body_count != 0U) {
                // Explicit readback leaves live simulation pages on the GPU;
                // direct CPU dereferences migrate them back and forth each frame.
                const auto gather_error = simulation::gather_rigid_debug_samples(
                    (rigid_body_count + 127U) / 128U, 128U, stream,
                    ids, states[current_state], debug_applied_forces,
                    debug_applied_torques, rigid_body_count, debug_rigid_samples);
                if (gather_error != PM_SUCCESS)
                    return backend_failure(gather_error,
                        "rigid debug sample gather launch failed");
                const auto status = download(output.rigid_bodies.data(), debug_rigid_samples,
                    rigid_body_count * sizeof(PhysicsDebugRigidSample));
                if (!status) return status;
            }
            output.fluid_particles.clear();
            for (std::uint32_t slot = 0U; slot < fluids.size(); ++slot) {
                const auto &owner = fluids[slot];
                if (!owner || !owner->alive) continue;
                const FluidStorage &fluid = *owner;
                std::uint32_t count{};
                const auto count_status = read_device_u32(
                    stream, fluid.count, count,
                    "physics debug fluid count readback failed");
                if (!count_status) return count_status;
                if (count > fluid.options.capacity)
                    return failure(StatusCode::internal_error, "fluid output count exceeds capacity");
                for (auto *values : {fluid.positions, fluid.velocities, fluid.forces}) {
                    const auto status = read_host_span(stream, values, count);
                    if (!status) return status;
                }
                auto status = read_host_span(stream, fluid.ids, count);
                if (!status) return status;
                status = read_host_span(stream, fluid.foam, count);
                if (!status) return status;
                output.fluid_particles.reserve(
                    output.fluid_particles.size() + count);
                for (std::uint32_t index = 0U; index < count; ++index)
                    output.fluid_particles.push_back({
                        {slot, fluid.generation}, fluid.ids[index],
                        fluid.positions[index], fluid.velocities[index],
                        fluid.forces[index], fluid.foam[index]});
            }
            output.cloth_vertices.clear();
            for (std::uint32_t slot = 0U; slot < cloths.size(); ++slot) {
                const auto &owner = cloths[slot];
                if (!owner || !owner->alive) continue;
                const ClothStorage &cloth = *owner;
                for (auto *values : {cloth.positions, cloth.velocities, cloth.rigid_contact_forces,
                                     cloth.fluid_forces, cloth.soft_body_forces}) {
                    const auto status = read_host_span(stream, values, cloth.vertex_count);
                    if (!status) return status;
                }
                output.cloth_vertices.reserve(
                    output.cloth_vertices.size() + cloth.vertex_count);
                for (std::uint32_t index = 0U;
                     index < cloth.vertex_count; ++index)
                    output.cloth_vertices.push_back({
                        {slot, cloth.generation}, index,
                        cloth.positions[index], cloth.velocities[index],
                        cloth.rigid_contact_forces[index],
                        cloth.fluid_forces != nullptr
                            ? cloth.fluid_forces[index] : Vec3{},
                        cloth.soft_body_forces[index]});
            }
            output.soft_body_nodes.clear();
            output.rope_nodes.clear();
            for(unsigned slot=0;slot<ropes.size();++slot) {
                if(!ropes[slot] || !ropes[slot]->alive)continue;
                const auto &r=ropes[slot]->data;
                for (auto *values : {r.positions, r.velocities, r.constraint_forces,
                                     r.contact_forces, r.fluid_contact_forces}) {
                    const auto status = read_host_span(stream, values, r.count);
                    if (!status) return status;
                }
                for(unsigned i=0;i<r.count;++i)output.rope_nodes.push_back({
                    {slot,ropes[slot]->generation},i,r.positions[i],r.velocities[i],
                    r.constraint_forces[i],r.contact_forces[i],r.fluid_contact_forces[i]});
            }
            for (std::uint32_t slot = 0U; slot < soft_bodies.size(); ++slot) {
                const auto &owner = soft_bodies[slot];
                if (!owner || !owner->alive) continue;
                const SoftBodyStorage &body = *owner;
                for (auto *values : {body.positions, body.velocities, body.rigid_contact_forces,
                                     body.cloth_forces, body.fluid_forces}) {
                    const auto status = read_host_span(stream, values, body.node_count);
                    if (!status) return status;
                }
                output.soft_body_nodes.reserve(
                    output.soft_body_nodes.size() + body.node_count);
                for (std::uint32_t index = 0U; index < body.node_count; ++index)
                    output.soft_body_nodes.push_back({
                        {slot, body.generation}, index, body.positions[index],
                        body.velocities[index],
                        body.rigid_contact_forces[index], body.cloth_forces[index],
                        body.fluid_forces[index]});
            }
            std::uint32_t rigid_count{}, fluid_events{};
            auto status = download(&rigid_count, rigid_contact_count, sizeof(rigid_count));
            if (!status) return status;
            if (fluid_count != 0U) {
                status = download(&fluid_events, fluid_contact_count, sizeof(fluid_events));
                if (!status) return status;
            }
            output.rigid_contacts.resize(std::min(rigid_count, rigid_contact_capacity));
            status = download(output.rigid_contacts.data(), rigid_contact_events,
                              output.rigid_contacts.size() * sizeof(RigidContactEvent));
            if (!status) return status;
            output.fluid_contacts.resize(std::min(fluid_events, options.contact_capacity));
            status = download(output.fluid_contacts.data(), fluid_contact_events,
                              output.fluid_contacts.size() * sizeof(output.fluid_contacts[0]));
            if (!status) return status;
            debug_next_frame =
                (debug_next_frame + 1U) % debug_frames.size();
            debug_frame_count = std::min(
                debug_frame_count + 1U, debug_frames.size());
            return success();
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "physics debug frame allocation failed");
        }
    }

    ~Impl() {
        if (frame && !frame->acknowledged) {
            (void)wait_for_completion(frame);
        }
        if (meshes != nullptr) {
            for (std::uint32_t index = 0;
                 index < options.triangle_mesh_capacity; ++index) {
                release_managed(meshes[index].bvh_nodes);
                release_managed(meshes[index].bvh_leaves);
                release_managed(meshes[index].solid_planes);
                release_managed(meshes[index].shell_normals);
                release_managed(meshes[index].indices);
                release_managed(meshes[index].vertices);
            }
        }
        if (paint_fields != nullptr) {
            for (std::uint32_t index = 0;
                 index < options.paint_field_capacity; ++index) {
                release_managed(paint_fields[index].uvs);
                release_managed(paint_fields[index].pixels);
            }
        }
        release_managed(paint_fields);
        release_managed(paint_rules);
        for (device::Event event : timing_events) {
            execution.services->destroy_event(event);
        }
        release_managed(meshes);
        release_managed(fluid_neighbor_overflow);
        release_managed(fluid_maximum_neighbor_count);
        release_managed(rigid_contact_count);
        release_managed(fixed_contact_projection);
        release_managed(rigid_contact_events);
        release_managed(rigid_leaf_pair_counts);
        release_managed(rigid_leaf_pairs);
        release_managed(rigid_leaf_manifolds);
        release_managed(rigid_broad_phase_workspace);
        release_managed(rigid_active_pair_count);
        release_managed(rigid_active_pairs);
        release_managed(rigid_active_pair_flags);
        release_managed(rigid_world_bounds);
        release_managed(hit_box_rigid_flags);
        release_managed(fluid_body_bounds);
        release_managed(fluid_body_masks);
        release_managed(fluid_global_body_masks);
        release_managed(fluid_body_contact_flags);
        release_managed(fluid_contact_events);
        release_managed(fluid_contact_count);
        release_managed(fluid_contact_overflow);
        release_managed(rigid_contact_event_offsets);
        release_managed(rigid_contact_schedule);
        release_managed(rigid_manifolds);
        release_managed(avbd_bodies);
        release_managed(rigid_contact_cache);
        release_managed(rigid_contact_cache_slots);
        release_managed(states[1]);
        release_managed(debug_applied_forces);
        release_managed(debug_applied_torques);
        release_managed(debug_rigid_samples);
        release_managed(fluid_previous_states);
        release_managed(render_previous_states);
        release_managed(states[0]);
        release_managed(ids);
        release_managed(rigid_constraints);
        release_managed(targets);
        release_managed(accumulators);
        release_managed(parameters);
    }

    [[nodiscard]] Status prepare_timing_events(
        std::size_t boundary_count) noexcept {
        try {
            timing_events.reserve(boundary_count);
            timing_stages.clear();
            timing_launch_counts.clear();
            timing_launch_counts.reserve(boundary_count);
            timing_stages.reserve(boundary_count > 0U ? boundary_count - 1U
                                                       : 0U);
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "failed to allocate timing event storage");
        }
        while (timing_events.size() < boundary_count) {
            device::Event event{};
            const PmResult error = execution.services->create_event(true, event);
            if (error != PM_SUCCESS) {
                return backend_failure(error, "failed to create timing event");
            }
            try {
                timing_events.push_back(event);
            } catch (...) {
                execution.services->destroy_event(event);
                return failure(StatusCode::out_of_memory,
                               "failed to retain timing event");
            }
        }
        return success();
    }

    [[nodiscard]] Status require_current_device() const noexcept {
        if (execution.services == nullptr)
            return failure(StatusCode::invalid_argument, "missing simulation device services");
        const auto result = execution.services->activate();
        return result == PM_SUCCESS ? success()
            : backend_failure(result, "failed to activate simulation device");
    }

    [[nodiscard]] Status require_idle() noexcept {
        Status status = require_current_device();
        if (!status) {
            return status;
        }
        if (frame && !frame->acknowledged) {
            return failure(StatusCode::busy,
                           "world still has an unacknowledged frame");
        }
        frame.reset();
        return success();
    }

    [[nodiscard]] Status validate_handle(RigidBodyId id,
                                         std::uint32_t &dense) const noexcept {
        if (id.index >= slots.size()) {
            return failure(StatusCode::invalid_handle,
                           "rigid body handle index is invalid");
        }
        const Slot &slot = slots[id.index];
        if (!slot.alive || slot.generation != id.generation) {
            return failure(StatusCode::invalid_handle,
                           "rigid body handle is stale");
        }
        dense = slot.dense_index;
        return success();
    }

    [[nodiscard]] Status validate_handle(
        RigidConstraintId id, RigidConstraintResource *&constraint) const noexcept {
        if (id.index >= options.rigid_constraint_capacity ||
            rigid_constraints == nullptr) {
            return failure(StatusCode::invalid_handle,
                           "rigid constraint handle index is invalid");
        }
        RigidConstraintResource &resource = rigid_constraints[id.index];
        if (!resource.alive || resource.generation != id.generation) {
            return failure(StatusCode::invalid_handle,
                           "rigid constraint handle is stale");
        }
        constraint = &resource;
        return success();
    }

    [[nodiscard]] Status validate_handle(TriangleMeshId id) const noexcept {
        if (id.index >= options.triangle_mesh_capacity || meshes == nullptr) {
            return failure(StatusCode::invalid_handle,
                           "triangle mesh handle index is invalid");
        }
        const TriangleMeshResource &mesh = meshes[id.index];
        if (!mesh.alive || mesh.generation != id.generation) {
            return failure(StatusCode::invalid_handle,
                           "triangle mesh handle is stale");
        }
        return success();
    }

    [[nodiscard]] Status validate_handle(FluidId id,
                                         FluidStorage *&fluid) const noexcept {
        if (id.index >= fluids.size() || !fluids[id.index] ||
            !fluids[id.index]->alive ||
            fluids[id.index]->generation != id.generation) {
            return failure(StatusCode::invalid_handle,
                           "fluid handle is invalid or stale");
        }
        fluid = fluids[id.index].get();
        return success();
    }
};

FrameToken::FrameToken() noexcept = default;

FrameToken::~FrameToken() {
    if (impl_ && impl_->completion && !impl_->completion->acknowledged) {
        (void)wait_for_completion(impl_->completion);
    }
}

FrameToken::FrameToken(FrameToken &&other) noexcept
    : impl_(std::move(other.impl_)) {}

FrameToken &FrameToken::operator=(FrameToken &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (impl_ && impl_->completion && !impl_->completion->acknowledged) {
        (void)wait_for_completion(impl_->completion);
    }
    impl_ = std::move(other.impl_);
    return *this;
}

bool FrameToken::pending() const noexcept {
    return impl_ && impl_->completion && !impl_->completion->acknowledged;
}

bool FrameToken::ready() const noexcept {
    if (!pending()) {
        return true;
    }
    bool ready{};
    const auto &frame = *impl_->completion;
    return frame.execution.services->query_event(frame.event, ready) == PM_SUCCESS && ready;
}

Status FrameToken::wait() noexcept {
    if (!impl_) {
        return success();
    }
    return wait_for_completion(impl_->completion);
}

SimulationRuntime::SimulationRuntime() noexcept = default;
SimulationRuntime::~SimulationRuntime() = default;
SimulationRuntime::SimulationRuntime(SimulationRuntime &&) noexcept = default;
SimulationRuntime &SimulationRuntime::operator=(SimulationRuntime &&) noexcept = default;

Status SimulationRuntime::cloth_memory_requirements(std::uint32_t vertices,
    std::uint32_t indices, std::uint32_t bonds, std::uint32_t rigid,
    simulation::StorageRequirements &output) noexcept {
    output = {};
    ClothStorage state;
    state.vertex_count = state.vertex_capacity = vertices;
    state.index_count = indices;
    state.bond_count = bonds;
    state.neighbor_capacity = 2 * (std::size_t(indices) + bonds);
    if (state.neighbor_capacity > UINT32_MAX)
        return failure(StatusCode::invalid_argument, "cloth neighbor capacity overflow");
    simulation::StoragePlanner planner;
    auto status = simulation::visit_cloth_storage(state, rigid,
        true, true, true, true, true, planner);
    if (status) output = planner.requirements;
    return status;
}
Status SimulationRuntime::rope_memory_requirements(std::uint32_t nodes,
    std::uint32_t rigid, simulation::StorageRequirements &output) noexcept {
    output = {};
    RopeStorage state;
    simulation::StoragePlanner planner;
    auto status = simulation::visit_rope_storage(state.data, nodes, rigid, planner);
    if (status) output = planner.requirements;
    return status;
}
Status SimulationRuntime::soft_body_memory_requirements(std::uint32_t nodes,
    std::uint32_t bonds, std::uint32_t vertices, std::uint32_t indices,
    simulation::StorageRequirements &output) noexcept {
    output = {};
    SoftBodyStorage state;
    state.node_count = nodes;
    state.bond_count = bonds;
    state.surface_vertex_count = vertices;
    state.surface_index_count = indices;
    const std::size_t neighbors = 2 * std::size_t(bonds);
    const std::size_t influences = 4 * std::size_t(indices);
    if (neighbors > UINT32_MAX || influences > UINT32_MAX)
        return failure(StatusCode::invalid_argument, "soft-body adjacency capacity overflow");
    simulation::StoragePlanner planner;
    auto status = simulation::visit_soft_body_storage(state, neighbors,
        neighbors, influences, planner);
    if (status) output = planner.requirements;
    return status;
}

Status SimulationRuntime::smoke_memory_requirements(std::uint32_t particles,
    std::uint32_t cells, simulation::StorageRequirements &output) noexcept {
    output = {};
    if (particles > 1'000'000U || cells > 256U * 256U * 256U)
        return failure(StatusCode::invalid_argument, "smoke capacity exceeds implementation limits");
    SmokeStorage state;
    simulation::StoragePlanner planner;
    state.grid.cell_count = cells;
    // The query declares cell count, not grid shape. Bound staggered faces
    // using the minimum supported dimension (8); coarse dimensions stop at 2.
    for (auto &count : state.grid.face_count) count = cells + (cells + 7) / 8;
    auto coarse_cells = cells;
    for (auto &level : state.grid.coarse) {
        // At least two dimensions still halve until the final 2x2x2 level.
        coarse_cells = std::max(8U, coarse_cells / 4U);
        level.cell_count = coarse_cells;
        for (auto &count : level.face_count) count = coarse_cells + (coarse_cells + 1) / 2;
    }
    auto status = simulation::visit_smoke_grid_storage(state.grid, planner);
    if (status) status = simulation::visit_smoke_particles_storage(state, particles, planner);
    if (status) output = planner.requirements;
    return status;
}

Status SimulationRuntime::coupling_memory_requirements(const simulation::CouplingStorageLimits &limits,
    simulation::StorageRequirements &output) noexcept {
    output = {};
    simulation::StoragePlanner fluid_soft, rope_soft, fluid_rope, soft_cloth, smoke;
    FluidSoftCouplingStorage fs;
    RopeSoftBodyCouplingStorage rs;
    FluidRopeCouplingStorage fr;
    SoftClothCouplingStorage sc;
    const auto triangles = limits.soft_indices / 3;
    const auto tree_nodes = 2 * triangles; // Binary BVH has at most 2T-1 nodes.
    Status status;
    if (!(status = simulation::visit_fluid_soft_storage(fs, limits.fluid_particles, limits.soft_nodes,
            limits.soft_vertices, tree_nodes, triangles, fluid_soft)) ||
        !(status = simulation::visit_rope_soft_tree_storage(rs, tree_nodes, triangles, rope_soft)) ||
        !(status = simulation::visit_rope_soft_data_storage(rs, limits.soft_nodes, limits.soft_vertices, rope_soft)) ||
        !(status = simulation::visit_fluid_rope_storage(fr, limits.rope_nodes, fluid_rope)) ||
        !(status = simulation::visit_soft_cloth_storage(sc, limits.soft_nodes, limits.cloth_vertices, soft_cloth)))
        return status;
    Vec3 *bounds = nullptr;
    std::uint32_t *converted = nullptr;
    if (!(status = smoke(bounds, 2, simulation::MemoryClass::persistent)) ||
        !(status = smoke(converted, 1, simulation::MemoryClass::persistent))) return status;
    for (const auto *plan : {&fluid_soft, &rope_soft, &fluid_rope, &soft_cloth, &smoke})
        output.persistent = std::max(output.persistent, plan->requirements.persistent);
    return {};
}
Status SimulationRuntime::emitter_memory_requirements(simulation::StorageRequirements &output) noexcept {
    output = {};
    ParticleSourceData state;
    simulation::StoragePlanner planner;
    const auto status = simulation::visit_emitter_storage(state, maximum_fluid_source_sites, planner);
    if (status) output = planner.requirements;
    return status;
}

Status SimulationRuntime::base_memory_requirements(SimulationOptions options,
    simulation::StorageRequirements &output) noexcept {
    output = {};
    Impl state;
    state.options = options;
    simulation::StoragePlanner planner;
    const auto status = simulation::visit_global_storage(state, options, planner);
    if (status) output = planner.requirements;
    return status;
}
Status SimulationRuntime::fluid_memory_requirements(std::uint32_t capacity,
    simulation::StorageRequirements &output) noexcept {
    output = {};
    FluidStorage state;
    simulation::StoragePlanner planner;
    const auto status = simulation::visit_fluid_storage(state, capacity, planner);
    if (status) output = planner.requirements;
    return status;
}

Status SimulationRuntime::create(SimulationOptions options, SimulationRuntime &output,
                     device::ExecutionQueue queue) noexcept {
    if (queue.services == nullptr)
        return failure(StatusCode::invalid_argument, "simulation requires device services");
    if (options.rigid_body_capacity == 0U ||
        options.triangle_mesh_capacity == 0U) {
        return failure(StatusCode::invalid_argument,
                       "rigid body and triangle mesh capacities must be positive");
    }
    if (options.physics_debug.frame_capacity > 3'600U ||
        options.physics_debug.frame_stride == 0U) {
        return failure(StatusCode::invalid_argument,
                       "physics debug frame capacity or stride is invalid");
    }
    PmResult error = queue.services->activate();
    if (error != PM_SUCCESS) {
        return backend_failure(error, "failed to activate simulation device");
    }

    std::unique_ptr<Impl> implementation;
    try {
        implementation = std::make_unique<Impl>();
        implementation->slots.resize(options.rigid_body_capacity);
        implementation->fluids.resize(options.fluid_capacity);
        implementation->smokes.resize(options.smoke_capacity);
        implementation->fluid_smoke_couplings.resize(
            options.fluid_smoke_coupling_capacity);
        implementation->smoke_soft_body_couplings.resize(
            options.smoke_soft_body_coupling_capacity);
        implementation->smoke_cloth_couplings.resize(
            options.smoke_cloth_coupling_capacity);
        implementation->smoke_rope_couplings.resize(
            options.smoke_rope_coupling_capacity);
        implementation->smoke_rigid_couplings.resize(
            options.smoke_rigid_coupling_capacity);
        implementation->cloths.resize(options.cloth_capacity);
        implementation->soft_bodies.resize(options.soft_body_capacity);
        implementation->ropes.resize(options.rope_capacity);
        implementation->fluid_rope_couplings.resize(
            options.fluid_rope_coupling_capacity);
        implementation->rope_soft_body_couplings.resize(
            options.rope_soft_body_coupling_capacity);
        implementation->rope_cloth_couplings.resize(
            options.rope_cloth_coupling_capacity);
        implementation->fluid_cloth_couplings.resize(
            options.fluid_cloth_coupling_capacity);
        implementation->fluid_soft_couplings.resize(
            options.fluid_soft_body_coupling_capacity);
        implementation->soft_cloth_couplings.resize(
            options.soft_body_cloth_coupling_capacity);
        implementation->particle_sources.resize(options.particle_source_capacity);
        implementation->destroy_planes.resize(options.particle_destroy_plane_capacity);
        implementation->debug_frames.resize(
            options.physics_debug.frame_capacity);
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate world host storage");
    }
    implementation->options = options;
    implementation->execution = queue;
    std::uint32_t scheduling_units = 1U;
    error = queue.services->scheduling_units(scheduling_units);
    if (error != PM_SUCCESS) return backend_failure(error, "failed to query contact multiprocessors");
    int resident_blocks = 1;
    implementation->rigid_contact_block_size = 32U;
    error = simulation::geometry_constraints_occupancy(
        &resident_blocks, implementation->rigid_contact_block_size);
    if (error != PM_SUCCESS) return backend_failure(error, "query contact kernel occupancy");
    // One resident block per SM avoids over-subscribing grid barriers. Launch
    // geometry changes scheduling only; equations and budgets stay shared.
    implementation->rigid_contact_grid_limit = resident_blocks > 0
        ? scheduling_units : 1U;
    StorageAllocator allocator;
    Status status = simulation::visit_global_storage(*implementation, options, allocator);
    if (!status) return status;
    for (std::uint32_t i = 0; i < options.paint_field_capacity; ++i)
        implementation->paint_fields[i] = {};
    for (std::uint32_t i = 0; i < options.paint_rule_capacity; ++i)
        implementation->paint_rules[i] = {};
    *implementation->fluid_neighbor_overflow = 0U;
    *implementation->fluid_maximum_neighbor_count = 0U;
    const std::size_t pair_capacity = std::size_t(options.rigid_body_capacity) * options.rigid_body_capacity;
    const std::size_t manifold_count = std::size_t(options.rigid_body_capacity) * (options.rigid_body_capacity - 1) / 2;
    const auto leaf_pair_slot_capacity = implementation->rigid_leaf_pair_slot_capacity;
    std::fill_n(implementation->avbd_bodies, options.rigid_body_capacity,
                simulation::GeometryConstraintBody{});
    std::fill_n(implementation->rigid_contact_cache, leaf_pair_slot_capacity, CachedContactPair{});
    std::fill_n(implementation->rigid_contact_cache_slots, pair_capacity, k_invalid_dense);
    *implementation->rigid_contact_schedule = {};
    *implementation->fluid_contact_count = 0U;
    *implementation->fluid_contact_overflow = 0U;
    std::fill_n(implementation->parameters, options.rigid_body_capacity,
                BodyParameters{});
    std::fill_n(implementation->accumulators, options.rigid_body_capacity,
                BodyAccumulator{});
    std::fill_n(implementation->targets, options.rigid_body_capacity,
                KinematicTarget{});
    std::fill_n(implementation->rigid_constraints,
                options.rigid_constraint_capacity, RigidConstraintResource{});
    std::fill_n(implementation->ids, options.rigid_body_capacity, RigidBodyId{});
    std::fill_n(implementation->states[0], options.rigid_body_capacity,
                RigidBodyState{});
    std::fill_n(implementation->states[1], options.rigid_body_capacity,
                RigidBodyState{});
    std::fill_n(implementation->render_previous_states,
                options.rigid_body_capacity, RigidBodyState{});
    if (implementation->debug_applied_forces != nullptr) {
        std::fill_n(implementation->debug_applied_forces,
                    options.rigid_body_capacity, Vec3{});
        std::fill_n(implementation->debug_applied_torques,
                    options.rigid_body_capacity, Vec3{});
    }
    std::fill_n(implementation->rigid_manifolds, manifold_count,
                ContactManifold{});
    std::fill_n(implementation->rigid_world_bounds,
                options.rigid_body_capacity, WorldAabb{});
    std::fill_n(implementation->hit_box_rigid_flags,
                options.rigid_body_capacity, 0U);
    std::fill_n(implementation->rigid_active_pair_flags, pair_capacity, 0U);
    std::fill_n(implementation->rigid_active_pairs, pair_capacity, 0U);
    *implementation->rigid_active_pair_count = 0U;
    std::fill_n(implementation->rigid_leaf_pairs,
                implementation->rigid_leaf_pair_capacity, LeafPair{});
    std::fill_n(implementation->rigid_leaf_pair_counts, pair_capacity, 0U);
    std::fill_n(implementation->rigid_contact_events,
                implementation->rigid_contact_capacity, RigidContactEvent{});
    *implementation->rigid_contact_count = 0U;
    std::fill_n(implementation->meshes, options.triangle_mesh_capacity,
                TriangleMeshResource{});
    for (std::uint32_t index = 0; index < options.triangle_mesh_capacity;
         ++index) {
        implementation->meshes[index].generation = 1U;
    }

    output.impl_ = std::move(implementation);
    return success();
}

#include "runtime_domains.cu.inc"
#include "runtime_deformables.cu.inc"
#include "runtime_rigid.cu.inc"
#include "native_schedule_backend.cu.inc"
#include "runtime_schedule.cu.inc"
#include "runtime_diagnostics.cu.inc"

} // namespace parallel_mater
