// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>
#include <parallel_mater/solver/contact_friction.hpp>
#include <parallel_mater/solver/avbd.hpp>
#include <parallel_mater/solver/halfspace.hpp>
#include "convex_plane_dedup.hpp"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>

#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace parallel_mater {
namespace {

constexpr float k_epsilon = 1.0e-6F;
constexpr std::uint32_t k_invalid_dense = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t k_soft_contact_cleanup_passes = 8U;

[[nodiscard]] Status success() noexcept { return {}; }

[[nodiscard]] Status failure(StatusCode code, const char *message,
                             cudaError_t cuda_error = cudaSuccess) noexcept {
    return {code, cuda_error, message};
}

[[nodiscard]] Status cuda_failure(cudaError_t error, const char *message) noexcept {
    return failure(StatusCode::cuda_failure, message, error);
}

__host__ __device__ Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vec3 multiply(Vec3 value, float scalar) noexcept {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

__host__ __device__ float dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__host__ __device__ float length_squared(Vec3 value) noexcept {
    return dot(value, value);
}

__host__ __device__ float vector_length(Vec3 value) noexcept {
    return sqrtf(length_squared(value));
}

__host__ __device__ float clamp_scalar(float value, float minimum,
                                       float maximum) noexcept {
    return fminf(fmaxf(value, minimum), maximum);
}

__host__ __device__ Vec3 normalized_or(Vec3 value, Vec3 fallback) noexcept {
    const float squared = length_squared(value);
    if (squared <= k_epsilon * k_epsilon) {
        return fallback;
    }
    return multiply(value, rsqrtf(squared));
}

__host__ __device__ Quaternion conjugate(Quaternion value) noexcept {
    return {-value.x, -value.y, -value.z, value.w};
}

__host__ __device__ Quaternion quaternion_multiply(Quaternion a,
                                                   Quaternion b) noexcept {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

__host__ __device__ Quaternion normalized_quaternion(Quaternion value) noexcept {
    const float squared = value.x * value.x + value.y * value.y +
                          value.z * value.z + value.w * value.w;
    if (squared <= k_epsilon * k_epsilon) {
        return {};
    }
    const float inverse = rsqrtf(squared);
    return {value.x * inverse, value.y * inverse, value.z * inverse,
            value.w * inverse};
}

__host__ __device__ Vec3 rotate(Quaternion orientation, Vec3 value) noexcept {
    const Vec3 q{orientation.x, orientation.y, orientation.z};
    const Vec3 twice_cross = multiply(cross(q, value), 2.0F);
    return add(value,
               add(multiply(twice_cross, orientation.w), cross(q, twice_cross)));
}

__host__ __device__ Vec3 inverse_rotate(Quaternion orientation,
                                       Vec3 value) noexcept {
    return rotate(conjugate(orientation), value);
}

__host__ __device__ Vec3 clamp_length(Vec3 value, float maximum) noexcept {
    const float squared = length_squared(value);
    if (squared <= maximum * maximum || squared <= k_epsilon * k_epsilon) {
        return value;
    }
    return multiply(value, maximum * rsqrtf(squared));
}

#include "geometry_constraints.cuh"
#include "avbd_cuda.cuh"

__device__ bool hit_box_separates_triangle(
    Vec3 axis, Vec3 first, Vec3 second, Vec3 third,
    Vec3 half_extents) noexcept {
    if (length_squared(axis) <= k_epsilon * k_epsilon) return false;
    const float first_projection = dot(axis, first);
    const float second_projection = dot(axis, second);
    const float third_projection = dot(axis, third);
    const float minimum = fminf(first_projection,
                                fminf(second_projection, third_projection));
    const float maximum = fmaxf(first_projection,
                                fmaxf(second_projection, third_projection));
    const float radius = fabsf(axis.x) * half_extents.x +
                         fabsf(axis.y) * half_extents.y +
                         fabsf(axis.z) * half_extents.z;
    return minimum > radius || maximum < -radius;
}

__device__ bool hit_box_overlaps_triangle(
    Vec3 first, Vec3 second, Vec3 third, Vec3 half_extents) noexcept {
    const Vec3 minimum = component_min(first, component_min(second, third));
    const Vec3 maximum = component_max(first, component_max(second, third));
    if (minimum.x > half_extents.x || maximum.x < -half_extents.x ||
        minimum.y > half_extents.y || maximum.y < -half_extents.y ||
        minimum.z > half_extents.z || maximum.z < -half_extents.z)
        return false;
    const Vec3 edges[3]{subtract(second, first), subtract(third, second),
                        subtract(first, third)};
    if (hit_box_separates_triangle(cross(edges[0], edges[1]), first,
                                   second, third, half_extents))
        return false;
    for (const Vec3 edge : edges)
        for (std::uint32_t axis = 0U; axis < 3U; ++axis)
            if (hit_box_separates_triangle(
                    cross(edge, basis_axis(axis)), first, second, third,
                    half_extents))
                return false;
    return true;
}

__device__ Vec3 hit_box_local_point(HitBox box, Vec3 point) noexcept {
    return inverse_rotate(box.orientation, subtract(point, box.center));
}

__device__ bool hit_box_contains_point(HitBox box, Vec3 point) noexcept {
    const Vec3 local = hit_box_local_point(box, point);
    return fabsf(local.x) <= box.half_extents.x &&
           fabsf(local.y) <= box.half_extents.y &&
           fabsf(local.z) <= box.half_extents.z;
}

__global__ void query_rigid_hit_box_kernel(
    HitBox box, const BodyParameters *parameters,
    const RigidBodyState *states, const TriangleMeshResource *meshes,
    std::uint32_t body_count, std::uint8_t *matches) {
    const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= body_count) return;
    const TriangleMeshResource &mesh = meshes[parameters[body].mesh.index];
    const RigidBodyState state = states[body];
    bool overlap = false;
    for (std::uint32_t index = 0U; index < mesh.index_count; index += 3U) {
        const Vec3 first = hit_box_local_point(
            box, transform_point(state, mesh.vertices[mesh.indices[index]]));
        const Vec3 second = hit_box_local_point(
            box, transform_point(state, mesh.vertices[mesh.indices[index + 1U]]));
        const Vec3 third = hit_box_local_point(
            box, transform_point(state, mesh.vertices[mesh.indices[index + 2U]]));
        if (hit_box_overlaps_triangle(first, second, third,
                                      box.half_extents)) {
            overlap = true;
            break;
        }
    }
    // A closed convex body can fully contain a small hit box without either
    // surface crossing. Convex-plane data is already built for such meshes.
    if (!overlap && mesh.solid_planes != nullptr) {
        const Vec3 center = inverse_rotate(
            state.orientation, subtract(box.center, state.position));
        overlap = true;
        for (std::uint32_t triangle = 0U;
             triangle < mesh.index_count / 3U; ++triangle) {
            const CollisionPlane plane = mesh.solid_planes[triangle];
            if (dot(plane.normal, center) > plane.offset + k_epsilon) {
                overlap = false;
                break;
            }
        }
    }
    matches[body] = overlap ? 1U : 0U;
}

__global__ void query_particle_hit_box_kernel(
    HitBox box, const Vec3 *positions, std::uint32_t particle_count,
    std::uint8_t *matches) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particle_count) return;
    matches[particle] =
        hit_box_contains_point(box, positions[particle]) ? 1U : 0U;
}

struct CompletionState {
    cudaEvent_t event{};
    const std::uint32_t *fluid_neighbor_overflow{};
    bool acknowledged{};
    Status completion_status{};
    std::function<Status()> on_complete{};

    ~CompletionState() {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }
};

enum class TimingStage : std::uint8_t {
    rigid_integration,
    rigid_world_bounds,
    rigid_pair_filter,
    rigid_pair_compaction,
    rigid_leaf_pair_generation,
    rigid_contact_evaluation,
    rigid_contact_solve,
    rigid_input_clear,
    cloth_prediction,
    cloth_constraints,
    cloth_contacts,
    soft_body_prediction,
    soft_body_constraints,
    soft_body_contacts,
    soft_body_contact_cleanup,
    soft_body_cloth_contacts,
    fluid_soft_body_contacts,
    fluid_rope_contacts,
    rope_soft_body_contacts,
    fluid_cloth_contacts,
    fluid_spawn,
    fluid_neighbor_sort,
    fluid_neighbor_forces,
    fluid_integration,
    fluid_static_contacts,
    fluid_body_index,
    fluid_moving_contacts,
    fluid_contact_events,
    fluid_outflow_compaction,
    fluid_smoke_exchange,
    rope_solve,
    smoke_grid,
    smoke_advection,
    smoke_emission,
};

[[nodiscard]] Status wait_for_completion(
    const std::shared_ptr<CompletionState> &completion) noexcept {
    if (!completion) {
        return success();
    }
    if (completion->acknowledged) {
        return completion->completion_status;
    }
    const cudaError_t error = cudaEventSynchronize(completion->event);
    completion->completion_status = error == cudaSuccess
        ? success() : cuda_failure(error, "CUDA frame completion failed");
    Status debug_status{};
    if (error == cudaSuccess && completion->on_complete)
        debug_status = completion->on_complete();
    if (error == cudaSuccess && completion->fluid_neighbor_overflow != nullptr &&
        *completion->fluid_neighbor_overflow != 0U) {
        completion->completion_status = failure(
            StatusCode::capacity_exceeded,
            "fluid neighbor count exceeded maximum_neighbors");
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

[[nodiscard]] bool valid_constraint_mask(std::uint8_t axes) noexcept {
    return (axes & ~rigid_constraint_all_axes) == 0U;
}

[[nodiscard]] Status validate_constraint_options(
    const RigidConstraintOptions &options) noexcept {
    if (static_cast<std::uint8_t>(options.type) >
        static_cast<std::uint8_t>(RigidConstraintType::motor))
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
    if (count == 0U) {
        pointer = nullptr;
        return success();
    }
    const cudaError_t error = cudaMallocManaged(
        reinterpret_cast<void **>(&pointer), sizeof(T) * count,
        cudaMemAttachGlobal);
    if (error != cudaSuccess) {
        pointer = nullptr;
        return cuda_failure(error, "CUDA managed allocation failed");
    }
    return success();
}

template <typename T> void release_managed(T *&pointer) noexcept {
    if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
    }
}

constexpr std::uint64_t k_fluid_empty_cell = ~std::uint64_t{0};
constexpr int k_fluid_cell_bias = 1 << 20;
constexpr std::uint32_t k_fluid_body_buckets = 4096U;

__host__ __device__ std::uint32_t fluid_body_bucket(
    int x, int y, int z) noexcept {
    const std::uint32_t hash =
        static_cast<std::uint32_t>(x) * 73856093U ^
        static_cast<std::uint32_t>(y) * 19349663U ^
        static_cast<std::uint32_t>(z) * 83492791U;
    return hash & (k_fluid_body_buckets - 1U);
}

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
static Status rebuild_cloth_topology(ClothStorage &cloth, bool initial = false) {
    if (!cloth.source_indices) return success();
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
            cloth.positions[node] = cloth.positions[old];
            cloth.previous[node] = cloth.previous[old];
            cloth.scratch[node] = cloth.positions[old];
            cloth.velocities[node] = cloth.velocities[old];
            cloth.vertex_sources[node] = cloth.vertex_sources[old];
            cloth.rigid_contact_forces[node] = {};
            cloth.soft_body_forces[node] = {};
            if (cloth.fluid_forces) cloth.fluid_forces[node] = {};
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

struct PaintFieldResource {
    PaintFieldOptions options{};
    Vec2 *uvs{};
    std::uint32_t *pixels{};
    std::uint32_t generation{1U};
    bool alive{};
};

struct PaintRuleResource {
    PaintRuleOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
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

#include "fluid.cuh"

#include "geometry_fluid.cuh"

#include "fluid_cloth.cuh"

#include "soft_body.cuh"

#include "cloth.cuh"

#include "geometry_soft_body.cuh"

#include "fluid_soft_body.cuh"

#include "soft_body_cloth.cuh"

#include "geometry_cloth.cuh"


} // namespace

namespace {
#include "geometry_rope.cuh"
#include "cloth_rope.cuh"
#include "fluid_rope.cuh"
#include "geometry_smoke.cuh"
#include "soft_body_smoke.cuh"
#include "cloth_smoke.cuh"
#include "rope_smoke.cuh"
#include "fluid_smoke.cuh"
} // namespace

struct FrameToken::Impl {
    std::shared_ptr<CompletionState> completion{};
};

struct World::Impl {
    struct Slot {
        std::uint32_t generation{1U};
        std::uint32_t dense_index{k_invalid_dense};
        bool alive{};
    };

    WorldOptions options{};
    int device_ordinal{-1};
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
    std::vector<std::unique_ptr<FluidSmokeCouplingSlot>> fluid_smoke_couplings{};
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
    AvbdBody *avbd_bodies{};
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
    std::vector<cudaEvent_t> timing_events{};
    std::vector<TimingStage> timing_stages{};
    std::vector<std::uint32_t> timing_launch_counts{};
    std::size_t timing_boundary_count{};
    std::uint64_t timing_frame_index{};
    bool timing_available{};
    std::vector<PhysicsDebugFrame> debug_frames{};
    std::size_t debug_next_frame{};
    std::size_t debug_frame_count{};

    [[nodiscard]] Status record_debug_frame(
        StepOptions step_options) noexcept {
        if (options.physics_debug.frame_capacity == 0U ||
            frame_index % options.physics_debug.frame_stride != 0U)
            return success();
        try {
            PhysicsDebugFrame &output = debug_frames[debug_next_frame];
            const auto download = [](void *destination, const void *source,
                                     std::size_t bytes) -> Status {
                if (bytes == 0U) return success();
                const auto error = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
                return error == cudaSuccess ? success()
                    : cuda_failure(error, "physics debug readback failed");
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
                gather_rigid_debug_samples_kernel<<<(rigid_body_count + 127U) / 128U, 128U>>>(
                    ids, states[current_state], debug_applied_forces,
                    debug_applied_torques, rigid_body_count, debug_rigid_samples);
                const auto status = download(output.rigid_bodies.data(), debug_rigid_samples,
                    rigid_body_count * sizeof(PhysicsDebugRigidSample));
                if (!status) return status;
            }
            output.fluid_particles.clear();
            for (std::uint32_t slot = 0U; slot < fluids.size(); ++slot) {
                const auto &owner = fluids[slot];
                if (!owner || !owner->alive) continue;
                const FluidStorage &fluid = *owner;
                const std::uint32_t count = *fluid.count;
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
                for(unsigned i=0;i<r.count;++i)output.rope_nodes.push_back({
                    {slot,ropes[slot]->generation},i,r.positions[i],r.velocities[i],
                    r.constraint_forces[i],r.contact_forces[i],r.fluid_contact_forces[i]});
            }
            for (std::uint32_t slot = 0U; slot < soft_bodies.size(); ++slot) {
                const auto &owner = soft_bodies[slot];
                if (!owner || !owner->alive) continue;
                const SoftBodyStorage &body = *owner;
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
        for (cudaEvent_t event : timing_events) {
            cudaEventDestroy(event);
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
            cudaEvent_t event = nullptr;
            const cudaError_t error = cudaEventCreate(&event);
            if (error != cudaSuccess) {
                return cuda_failure(error, "failed to create CUDA timing event");
            }
            try {
                timing_events.push_back(event);
            } catch (...) {
                cudaEventDestroy(event);
                return failure(StatusCode::out_of_memory,
                               "failed to retain CUDA timing event");
            }
        }
        return success();
    }

    [[nodiscard]] Status require_current_device() const noexcept {
        int current_device = -1;
        const cudaError_t error = cudaGetDevice(&current_device);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to query the current CUDA device");
        }
        if (current_device != device_ordinal) {
            return failure(StatusCode::invalid_argument,
                           "world used from a different CUDA device");
        }
        return success();
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
    return cudaEventQuery(impl_->completion->event) == cudaSuccess;
}

Status FrameToken::wait() noexcept {
    if (!impl_) {
        return success();
    }
    return wait_for_completion(impl_->completion);
}

World::World() noexcept = default;
World::~World() = default;
World::World(World &&) noexcept = default;
World &World::operator=(World &&) noexcept = default;

Status World::create(WorldOptions options, World &output,
                     cudaStream_t stream) noexcept {
    (void)stream;
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
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to query the current CUDA device");
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
    implementation->device_ordinal = device;
    int multiprocessors = 0;
    error = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess) return cuda_failure(error, "failed to query contact multiprocessors");
    int cooperative = 0;
    error = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
    if (error != cudaSuccess) return cuda_failure(error, "query cooperative contact launch support");
    int resident_blocks = 1;
    implementation->rigid_contact_block_size = 32U;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,
        solve_avbd_kernel, implementation->rigid_contact_block_size, 0);
    if (error != cudaSuccess) return cuda_failure(error, "query contact kernel occupancy");
    // One resident block per SM avoids over-subscribing grid barriers. Launch
    // geometry changes scheduling only; equations and budgets stay shared.
    implementation->rigid_contact_grid_limit = cooperative && resident_blocks > 0
        ? static_cast<unsigned>(std::max(1, multiprocessors)) : 1U;
    Status status = allocate_managed(implementation->paint_fields,
                                     options.paint_field_capacity);
    if (!status) return status;
    for (std::uint32_t i = 0; i < options.paint_field_capacity; ++i)
        implementation->paint_fields[i] = {};
    status = allocate_managed(implementation->paint_rules,
                              options.paint_rule_capacity);
    if (!status) return status;
    for (std::uint32_t i = 0; i < options.paint_rule_capacity; ++i)
        implementation->paint_rules[i] = {};
    status = allocate_managed(implementation->fluid_neighbor_overflow, 1U);
    if (!status) return status;
    *implementation->fluid_neighbor_overflow = 0U;
    status = allocate_managed(
        implementation->fluid_maximum_neighbor_count, 1U);
    if (!status) return status;
    *implementation->fluid_maximum_neighbor_count = 0U;
    const std::size_t pair_capacity =
        static_cast<std::size_t>(options.rigid_body_capacity) *
        options.rigid_body_capacity;
    // At most n(n-1)/2 pairs can involve a dynamic body: dynamic/dynamic
    // pairs are unique, and every other pair needs exactly one dynamic body.
    const std::size_t manifold_count =
        static_cast<std::size_t>(options.rigid_body_capacity) *
        (options.rigid_body_capacity - 1U) / 2U;
    if (manifold_count >
        std::numeric_limits<std::size_t>::max() / sizeof(ContactManifold)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds contact cache range");
    }
    const std::size_t requested_leaf_pair_slots = std::max(
        static_cast<std::size_t>(k_minimum_leaf_pair_cache_slots),
        static_cast<std::size_t>(options.rigid_body_capacity) *
            k_leaf_pair_cache_slots_per_body);
    const std::size_t leaf_pair_slot_capacity =
        std::min(manifold_count, requested_leaf_pair_slots);
    const std::uint32_t leaf_pairs_per_slot =
        options.rigid_body_capacity <= k_small_rigid_leaf_body_capacity
            ? k_small_rigid_leaf_pair_capacity : k_max_leaf_pairs_per_body_pair;
    if (leaf_pair_slot_capacity > std::numeric_limits<std::size_t>::max() /
                                      leaf_pairs_per_slot ||
        leaf_pair_slot_capacity >
            std::numeric_limits<std::uint32_t>::max()) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds leaf-pair cache range");
    }
    implementation->rigid_leaf_pair_slot_capacity =
        static_cast<std::uint32_t>(leaf_pair_slot_capacity);
    implementation->rigid_leaf_pair_capacity =
        leaf_pair_slot_capacity * leaf_pairs_per_slot;
    implementation->rigid_leaf_pairs_per_slot = leaf_pairs_per_slot;
    implementation->rigid_contact_capacity = options.contact_capacity;

    status = allocate_managed(implementation->parameters,
                                     options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->accumulators,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->targets,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_constraints,
                              options.rigid_constraint_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->ids,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->states[0],
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->states[1],
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->render_previous_states,
                              options.rigid_body_capacity);
    if (!status) return status;
    if (options.physics_debug.frame_capacity != 0U) {
        status = allocate_managed(implementation->debug_applied_forces,
                                  options.rigid_body_capacity);
        if (!status) return status;
        status = allocate_managed(implementation->debug_applied_torques,
                                  options.rigid_body_capacity);
        if (!status) return status;
        status = allocate_managed(implementation->debug_rigid_samples,
                                  options.rigid_body_capacity);
        if (!status) return status;
    }
    status = allocate_managed(implementation->fluid_previous_states,
                              options.rigid_body_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->rigid_manifolds, manifold_count);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->avbd_bodies, options.rigid_body_capacity);
    if (!status) return status;
    std::fill_n(implementation->avbd_bodies, options.rigid_body_capacity, AvbdBody{});
    status = allocate_managed(implementation->rigid_contact_cache, leaf_pair_slot_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->rigid_contact_cache_slots, pair_capacity);
    if (!status) return status;
    std::fill_n(implementation->rigid_contact_cache, leaf_pair_slot_capacity, CachedContactPair{});
    std::fill_n(implementation->rigid_contact_cache_slots, pair_capacity, k_invalid_dense);
    status = allocate_managed(implementation->rigid_contact_schedule, 1U);
    if (!status) return status;
    *implementation->rigid_contact_schedule = {};
    status = allocate_managed(implementation->rigid_contact_event_offsets,
                              manifold_count);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_world_bounds,
                              options.rigid_body_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->hit_box_rigid_flags,
                              options.rigid_body_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_body_bounds,
                              options.rigid_body_capacity);
    if (!status) return status;
    if (options.rigid_constraint_capacity > 0U) {
        status = allocate_managed(implementation->fixed_contact_projection,
                                  options.rigid_body_capacity);
        if (!status) return status;
    }
    const std::size_t fluid_body_words =
        (static_cast<std::size_t>(options.rigid_body_capacity) + 63U) / 64U;
    status = allocate_managed(implementation->fluid_body_masks,
                              k_fluid_body_buckets * fluid_body_words);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_global_body_masks,
                              fluid_body_words);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_body_contact_flags,
                              options.rigid_body_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_contact_events,
                              options.contact_capacity);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_contact_count, 1U);
    if (!status) return status;
    status = allocate_managed(implementation->fluid_contact_overflow, 1U);
    if (!status) return status;
    *implementation->fluid_contact_count = 0U;
    *implementation->fluid_contact_overflow = 0U;
    status = allocate_managed(implementation->rigid_active_pair_flags,
                              pair_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_active_pairs,
                              pair_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_active_pair_count, 1U);
    if (!status) {
        return status;
    }
    if (pair_capacity > static_cast<std::size_t>(
                             std::numeric_limits<int>::max())) {
        return failure(StatusCode::invalid_argument,
                       "rigid body capacity exceeds broad-phase range");
    }
    const auto pair_indices = thrust::make_counting_iterator<std::uint32_t>(0U);
    cudaError_t broad_phase_error = cub::DeviceSelect::Flagged(
        nullptr, implementation->rigid_broad_phase_workspace_size,
        pair_indices, implementation->rigid_active_pair_flags,
        implementation->rigid_active_pairs,
        implementation->rigid_active_pair_count,
        static_cast<int>(pair_capacity));
    if (broad_phase_error != cudaSuccess) {
        return cuda_failure(broad_phase_error,
                            "failed to size broad-phase workspace");
    }
    status = allocate_managed(implementation->rigid_broad_phase_workspace,
                              implementation->rigid_broad_phase_workspace_size);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_leaf_pairs,
                              implementation->rigid_leaf_pair_capacity);
    if (!status) {
        return status;
    }
    if (options.rigid_body_capacity <= k_small_rigid_leaf_body_capacity) {
        status = allocate_managed(implementation->rigid_leaf_manifolds,
                                  implementation->rigid_leaf_pair_capacity);
        if (!status) return status;
    }
    status = allocate_managed(implementation->rigid_leaf_pair_counts,
                              pair_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_contact_events,
                              implementation->rigid_contact_capacity);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->rigid_contact_count, 1U);
    if (!status) {
        return status;
    }
    status = allocate_managed(implementation->meshes,
                              options.triangle_mesh_capacity);
    if (!status) {
        return status;
    }

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

Status World::add_fluid(FluidOptions options,
                        DeviceSpan<const FluidParticle> initial_particles,
                        FluidId &output, cudaStream_t stream) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.capacity == 0U || options.capacity > INT_MAX ||
        initial_particles.size > options.capacity ||
        (initial_particles.size != 0U && initial_particles.data == nullptr) ||
        options.solver_iterations == 0U || options.solver_iterations > 16U ||
        options.maximum_neighbors == 0U ||
        !finite(options.particle_radius) || options.particle_radius <= 0.0F ||
        !finite(options.support_radius) ||
        options.support_radius < 2.0F * options.particle_radius ||
        !finite(options.rest_density) || options.rest_density <= 0.0F ||
        !finite(options.repulsion) || options.repulsion < 0.0F ||
        !finite(options.viscosity) || options.viscosity < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.maximum_speed) || options.maximum_speed <= 0.0F ||
        !finite(options.normal_damping) || options.normal_damping < 0.0F ||
        !finite(options.rest_particle_volume) ||
        options.rest_particle_volume < 0.0F ||
        !finite(options.maximum_pair_acceleration) ||
        options.maximum_pair_acceleration < 0.0F) {
        return failure(StatusCode::invalid_argument, "fluid options or initial particles are invalid");
    }
    std::uint32_t slot = 0U;
    for (; slot < impl_->fluids.size(); ++slot) {
        if (!impl_->fluids[slot] || !impl_->fluids[slot]->alive) break;
    }
    if (slot == impl_->fluids.size())
        return failure(StatusCode::capacity_exceeded, "fluid capacity exhausted");
    std::unique_ptr<FluidStorage> fluid;
    try { fluid = std::make_unique<FluidStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "fluid owner allocation failed"); }
    fluid->generation = impl_->fluids[slot] ? impl_->fluids[slot]->generation : 1U;
    fluid->options = options;
    const std::size_t capacity = options.capacity;
    if (!(status = allocate_managed(fluid->count, 1U)) ||
        !(status = allocate_managed(fluid->positions, capacity)) ||
        !(status = allocate_managed(fluid->velocities, capacity)) ||
        !(status = allocate_managed(fluid->previous, capacity)) ||
        !(status = allocate_managed(fluid->ids, capacity)) ||
        !(status = allocate_managed(fluid->foam, capacity)) ||
        !(status = allocate_managed(fluid->temperatures, capacity)) ||
        !(status = allocate_managed(fluid->foam_source, capacity)) ||
        !(status = allocate_managed(fluid->next_positions, capacity)) ||
        !(status = allocate_managed(fluid->next_velocities, capacity)) ||
        !(status = allocate_managed(fluid->next_ids, capacity)) ||
        !(status = allocate_managed(fluid->next_foam, capacity)) ||
        !(status = allocate_managed(fluid->next_temperatures, capacity)) ||
        !(status = allocate_managed(fluid->keep, capacity)) ||
        !(status = allocate_managed(fluid->selected, capacity)) ||
        !(status = allocate_managed(fluid->keys[0], capacity)) ||
        !(status = allocate_managed(fluid->keys[1], capacity)) ||
        !(status = allocate_managed(fluid->indices[0], capacity)) ||
        !(status = allocate_managed(fluid->indices[1], capacity)) ||
        !(status = allocate_managed(fluid->forces, capacity)) ||
        !(status = allocate_managed(fluid->body_impulses, capacity)) ||
        !(status = allocate_managed(fluid->moving_contacts, capacity)) ||
        !(status = allocate_managed(fluid->contact_samples, capacity)) ||
        !(status = allocate_managed(fluid->contact_flags, capacity)) ||
        !(status = allocate_managed(fluid->hit_box_flags, capacity)) ||
        !(status = allocate_managed(fluid->next_contact_samples, capacity)) ||
        !(status = allocate_managed(fluid->next_contact_flags, capacity)) ||
        !(status = allocate_managed(fluid->contact_count, 1U)) ||
        !(status = allocate_managed(fluid->contact_offset, 1U))) return status;
    cudaError_t error = cub::DeviceRadixSort::SortPairs(
        nullptr, fluid->sort_workspace_size, fluid->keys[0], fluid->keys[1],
        fluid->indices[0], fluid->indices[1], options.capacity);
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid sort workspace query failed");
    const auto sequence = thrust::make_counting_iterator<std::uint32_t>(0U);
    error = cub::DeviceSelect::Flagged(
        nullptr, fluid->select_workspace_size, sequence, fluid->keep,
        fluid->selected, fluid->count, options.capacity);
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid compaction workspace query failed");
    if (!(status = allocate_managed(fluid->sort_workspace,
                                    fluid->sort_workspace_size)) ||
        !(status = allocate_managed(fluid->select_workspace,
                                    fluid->select_workspace_size))) return status;
    *fluid->count = static_cast<std::uint32_t>(initial_particles.size);
    fluid->next_id = static_cast<std::uint32_t>(initial_particles.size);
    fluid->initial_count = initial_particles.size;
    if (!initial_particles.empty()) {
        *impl_->fluid_neighbor_overflow = 0U;
        fluid_copy_initial<<<(initial_particles.size + 127U) / 128U,
                             128U, 0, stream>>>(
            initial_particles.data, static_cast<std::uint32_t>(initial_particles.size),
            fluid->positions, fluid->velocities, fluid->ids, fluid->foam,
            fluid->temperatures,
            impl_->fluid_neighbor_overflow);
        error = cudaGetLastError();
        if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
            return cuda_failure(error, "initial fluid copy failed");
        if (*impl_->fluid_neighbor_overflow != 0U)
            return failure(StatusCode::invalid_argument,
                           "initial fluid particles must be finite");
    }
    fluid->alive = true;
    output = {slot, fluid->generation};
    impl_->fluids[slot] = std::move(fluid);
    ++impl_->fluid_count;
    ++impl_->revision;
    return success();
}

Status World::add_fluid_geometry(FluidOptions options,
                                 FluidGeometrySource source, FluidId &output,
                                 cudaStream_t stream) noexcept {
    std::vector<FluidParticle> sampled;
    Status status = sample_fluid_geometry(source, sampled);
    if (!status) return status;
    if (options.capacity == 0U)
        return failure(StatusCode::invalid_argument,
                       "fluid geometry requires a positive fluid capacity");
    if (sampled.size() > options.capacity) {
        std::vector<FluidParticle> selected;
        try {
            selected.reserve(options.capacity);
            for (std::size_t i = 0; i < options.capacity; ++i)
                selected.push_back(sampled[i * sampled.size() /
                                            options.capacity]);
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "fluid geometry selection allocation failed");
        }
        sampled.swap(selected);
    }
    FluidParticle *device = nullptr;
    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&device),
                                    sampled.size() * sizeof(FluidParticle));
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid geometry upload allocation failed");
    error = cudaMemcpy(device, sampled.data(),
                       sampled.size() * sizeof(FluidParticle),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess)
        status = add_fluid(options, {device, sampled.size()}, output, stream);
    cudaFree(device);
    if (error != cudaSuccess)
        return cuda_failure(error, "fluid geometry upload failed");
    return status;
}

Status World::remove_fluid(FluidId id, cudaStream_t) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(id, fluid))) return status;
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->fluid_rope_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a rope coupling");
    for (const auto &coupling : impl_->fluid_smoke_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a smoke coupling");
    for (std::uint32_t i = 0; i < impl_->options.paint_rule_capacity; ++i)
        if (impl_->paint_rules[i].alive &&
            impl_->paint_rules[i].options.source == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a paint rule");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.fluid == id)
            return failure(StatusCode::invalid_argument,
                "fluid is still referenced by a cloth coupling");
    std::unique_ptr<FluidStorage> tombstone(
        new (std::nothrow) FluidStorage());
    if (!tombstone)
        return failure(StatusCode::out_of_memory,
                       "fluid removal tombstone allocation failed");
    const std::uint32_t generation = fluid->generation + 1U;
    impl_->destroyed_particle_count +=
        fluid->initial_count + fluid->emitted_count - *fluid->count;
    tombstone->generation = generation;
    impl_->fluids[id.index] = std::move(tombstone);
    for (ParticleSourceSlot &plane : impl_->particle_sources)
        if (plane.alive && plane.options.fluid == id) {
            plane.alive = false; ++plane.generation; plane.data.reset();
        }
    for (DestroyPlaneSlot &plane : impl_->destroy_planes)
        if (plane.alive && plane.options.fluid == id) {
            plane.alive = false; ++plane.generation;
        }
    --impl_->fluid_count;
    ++impl_->revision;
    return success();
}

Status World::fluid_view(FluidId id, FluidDeviceView &output) const noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(id, fluid))) return status;
    if (impl_->frame && !impl_->frame->acknowledged)
        return failure(StatusCode::busy, "fluid view requires a completed frame");
    const std::uint32_t count = *fluid->count;
    output = {{fluid->positions, count}, {fluid->velocities, count},
              {fluid->forces, count}, {fluid->ids, count},
              {fluid->foam, count}, {fluid->temperatures, count},
              count, fluid->options.particle_radius,
              fluid->options.support_radius, impl_->revision};
    return success();
}

Status World::add_smoke(SmokeOptions options, SmokeId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.capacity == 0U || options.capacity > 1'000'000U ||
        !finite(options.emitter_center) || !finite(options.initial_velocity) ||
        !finite(options.wind) || !finite(options.emitter_half_extents.x) ||
        !finite(options.emitter_half_extents.y) ||
        options.emitter_half_extents.x <= 0.0F ||
        options.emitter_half_extents.y <= 0.0F ||
        !finite(options.particles_per_second) || options.particles_per_second <= 0.0F ||
        options.particles_per_second > 1'000'000.0F ||
        !finite(options.lifetime) || options.lifetime <= 0.0F ||
        !finite(options.particle_radius) || options.particle_radius <= 0.0F ||
        !finite(options.buoyancy) || !finite(options.response) ||
        options.response < 0.0F || !finite(options.rest_number_density) ||
        options.rest_number_density <= 0.0F ||
        !finite(options.pressure_stiffness) ||
        options.pressure_stiffness < 0.0F ||
        !finite(options.viscosity) || options.viscosity < 0.0F ||
        !finite(options.vorticity_confinement) ||
        options.vorticity_confinement < 0.0F ||
        !finite(options.maximum_speed) ||
        options.maximum_speed <= 0.0F ||
        (options.grid_resolution != 0U &&
         (options.grid_resolution < 16U || options.grid_resolution > 256U ||
          options.grid_vertical_resolution < 8U ||
          options.grid_vertical_resolution > 256U ||
          options.grid_pressure_iterations < 4U ||
          options.grid_pressure_iterations > 128U ||
          !finite(options.grid_kinematic_viscosity) ||
          options.grid_kinematic_viscosity < 0.0F ||
          !finite(options.grid_les_coefficient) ||
          options.grid_les_coefficient < 0.0F ||
          !finite(options.grid_pressure_tolerance) ||
          options.grid_pressure_tolerance <= 0.0F ||
          options.grid_pressure_tolerance > 1.0F)) ||
        !finite(options.grid_minimum) ||
        !finite(options.grid_edge_length) || options.grid_edge_length < 0.0F)
        return failure(StatusCode::invalid_argument, "invalid smoke options");
    std::uint32_t slot = 0U;
    while (slot < impl_->smokes.size() && impl_->smokes[slot] &&
           impl_->smokes[slot]->alive) ++slot;
    if (slot == impl_->smokes.size())
        return failure(StatusCode::capacity_exceeded, "smoke capacity exhausted");
    std::unique_ptr<SmokeStorage> smoke;
    try { smoke = std::make_unique<SmokeStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "smoke owner allocation failed"); }
    smoke->generation = impl_->smokes[slot] ? impl_->smokes[slot]->generation : 1U;
    smoke->options = options;
    if (options.grid_resolution != 0U) {
        auto &grid = smoke->grid;
        grid.resolution = options.grid_resolution;
        grid.height = options.grid_vertical_resolution;
        grid.cell_count = options.grid_resolution * options.grid_resolution *
                          options.grid_vertical_resolution;
        if (options.grid_edge_length > 0.0F) {
            grid.minimum = options.grid_minimum;
            grid.spacing = options.grid_edge_length / float(grid.resolution);
        } else {
            const Vec3 end = add(options.emitter_center,
                multiply(options.wind, options.lifetime));
            const Vec3 low{fminf(options.emitter_center.x, end.x) - 1.5F,
                options.emitter_center.y,
                fminf(options.emitter_center.z, end.z) - 1.5F};
            const Vec3 high{fmaxf(options.emitter_center.x, end.x) + 1.5F,
                options.emitter_center.y,
                fmaxf(options.emitter_center.z, end.z) + 1.5F};
            const float edge = fmaxf(high.x-low.x, high.z-low.z);
            const Vec3 center = multiply(add(low, high), 0.5F);
            grid.spacing = edge / float(grid.resolution);
            grid.minimum = subtract(center, {edge*0.5F,
                grid.spacing * float(grid.height) * 0.5F, edge*0.5F});
        }
        if (!finite(grid.spacing) || grid.spacing < 1.0e-4F)
            return failure(StatusCode::invalid_argument,
                "invalid smoke grid cell spacing");
        for (int axis = 0; axis < 3; ++axis)
            grid.face_count[axis] = smoke_grid_face_count(axis,
                int(grid.resolution), int(grid.height));
        if (!(status = allocate_managed(grid.velocity, grid.cell_count)) ||
            !(status = allocate_managed(grid.density, grid.cell_count)) ||
            !(status = allocate_managed(grid.temperature, grid.cell_count)) ||
            !(status = allocate_managed(grid.density_accumulator,
                                        grid.cell_count)) ||
            !(status = allocate_managed(grid.temperature_accumulator,
                                        grid.cell_count)) ||
            !(status = allocate_managed(grid.pressure[0], grid.cell_count)) ||
            !(status = allocate_managed(grid.pressure[1], grid.cell_count)) ||
            !(status = allocate_managed(grid.divergence, grid.cell_count)) ||
            !(status = allocate_managed(grid.residual, grid.cell_count)) ||
            !(status = allocate_managed(grid.vorticity, grid.cell_count)) ||
            !(status = allocate_managed(grid.strain, grid.cell_count)) ||
            !(status = allocate_managed(grid.subgrid_force, grid.cell_count)) ||
            !(status = allocate_managed(grid.rhs_max, 1U)) ||
            !(status = allocate_managed(grid.residual_max, 1U)) ||
            !(status = allocate_managed(grid.pressure_converged, 1U)) ||
            !(status = allocate_managed(grid.solid, grid.cell_count)))
            return status;
        for (int axis = 0; axis < 3; ++axis) {
            if (!(status = allocate_managed(grid.face_velocity[axis][0],
                                            grid.face_count[axis])) ||
                !(status = allocate_managed(grid.face_velocity[axis][1],
                                            grid.face_count[axis])) ||
                !(status = allocate_managed(grid.face_reverse[axis],
                                            grid.face_count[axis])) ||
                !(status = allocate_managed(grid.face_open[axis],
                                            grid.face_count[axis])) ||
                !(status = allocate_managed(grid.face_wall_velocity[axis],
                                            grid.face_count[axis])) ||
                !(status = allocate_managed(grid.face_normal[axis],
                                            grid.face_count[axis])) ||
                !(status = allocate_managed(grid.face_nearest_triangle[axis],
                                            grid.face_count[axis])))
                return status;
        }
        std::uint32_t level_n = grid.resolution;
        std::uint32_t level_h = grid.height;
        for (auto &level : grid.coarse) {
            level_n = std::max(2U, level_n / 2U);
            level_h = std::max(2U, level_h / 2U);
            level.n = level_n;
            level.height = level_h;
            level.cell_count = level_n * level_h * level_n;
            if (!(status = allocate_managed(level.pressure[0], level.cell_count)) ||
                !(status = allocate_managed(level.pressure[1], level.cell_count)) ||
                !(status = allocate_managed(level.rhs, level.cell_count)) ||
                !(status = allocate_managed(level.residual, level.cell_count)))
                return status;
            for (int axis = 0; axis < 3; ++axis)
                if (!(status = allocate_managed(level.open[axis],
                    smoke_grid_face_count(axis, int(level.n),
                                          int(level.height)))))
                    return status;
        }
        const auto blocks = (grid.cell_count + 255U) / 256U;
        smoke_grid_initialize_cells<<<blocks, 256U>>>(grid.velocity,
            grid.density, grid.temperature, grid.pressure[0],
            grid.divergence, grid.vorticity, grid.strain,
            grid.subgrid_force, grid.solid, grid.cell_count, options.wind);
        cudaMemset(grid.pressure[1], 0, grid.cell_count * sizeof(float));
        cudaMemset(grid.residual, 0, grid.cell_count * sizeof(float));
        cudaMemset(grid.rhs_max, 0, sizeof(float));
        cudaMemset(grid.residual_max, 0, sizeof(float));
        cudaMemset(grid.pressure_converged, 0, sizeof(unsigned int));
        for (int axis = 0; axis < 3; ++axis) {
            const float component = axis == 0 ? options.wind.x :
                                    axis == 1 ? options.wind.y : options.wind.z;
            smoke_grid_initialize_faces<<<
                (grid.face_count[axis] + 255U) / 256U, 256U>>>(
                grid.face_velocity[axis][0], grid.face_reverse[axis],
                grid.face_open[axis], grid.face_wall_velocity[axis],
                grid.face_normal[axis], grid.face_nearest_triangle[axis],
                grid.face_count[axis], component);
            cudaMemcpy(grid.face_velocity[axis][1],
                grid.face_velocity[axis][0],
                grid.face_count[axis] * sizeof(float),
                cudaMemcpyDeviceToDevice);
        }
        for (auto &level : grid.coarse) {
            cudaMemset(level.pressure[0], 0,
                level.cell_count * sizeof(float));
            cudaMemset(level.pressure[1], 0,
                level.cell_count * sizeof(float));
            cudaMemset(level.rhs, 0, level.cell_count * sizeof(float));
            cudaMemset(level.residual, 0,
                level.cell_count * sizeof(float));
        }
        cudaError_t grid_error = cudaPeekAtLastError();
        if (grid_error != cudaSuccess)
            return cuda_failure(grid_error, "smoke grid initialization failed");
        grid_error = cudaDeviceSynchronize();
        if (grid_error != cudaSuccess)
            return cuda_failure(grid_error, "smoke grid initialization failed");
    }
    if (!(status = allocate_managed(smoke->positions, options.capacity)) ||
        !(status = allocate_managed(smoke->previous_positions, options.capacity)) ||
        !(status = allocate_managed(smoke->velocities, options.capacity)) ||
        !(status = allocate_managed(smoke->ages, options.capacity)) ||
        !(status = allocate_managed(smoke->thermal_lift, options.capacity)) ||
        !(status = allocate_managed(smoke->number_densities, options.capacity)) ||
        !(status = allocate_managed(smoke->pressures, options.capacity)) ||
        !(status = allocate_managed(smoke->accelerations, options.capacity)) ||
        !(status = allocate_managed(smoke->vorticities, options.capacity)) ||
        !(status = allocate_managed(smoke->vorticity_magnitudes,
                                    options.capacity)) ||
        !(status = allocate_managed(smoke->keys[0], options.capacity)) ||
        !(status = allocate_managed(smoke->keys[1], options.capacity)) ||
        !(status = allocate_managed(smoke->indices[0], options.capacity)) ||
        !(status = allocate_managed(smoke->indices[1], options.capacity)) ||
        !(status = allocate_managed(smoke->rigid_impulses, options.capacity)))
        return status;
    cudaError_t sort_error = cub::DeviceRadixSort::SortPairs(
        nullptr, smoke->sort_workspace_size, smoke->keys[0], smoke->keys[1],
        smoke->indices[0], smoke->indices[1], options.capacity);
    if (sort_error != cudaSuccess)
        return cuda_failure(sort_error, "smoke sort workspace query failed");
    if (!(status = allocate_managed(smoke->sort_workspace,
                                    smoke->sort_workspace_size))) return status;
    smoke->alive = true;
    output = {slot, smoke->generation};
    impl_->smokes[slot] = std::move(smoke);
    ++impl_->revision;
    return success();
}

Status World::remove_smoke(SmokeId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smokes.size() || !impl_->smokes[id.index] ||
        !impl_->smokes[id.index]->alive ||
        impl_->smokes[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "smoke handle is stale");
    for (const auto &coupling : impl_->fluid_smoke_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->smoke_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->smoke_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a cloth coupling");
    for (const auto &coupling : impl_->smoke_rope_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a rope coupling");
    for (const auto &coupling : impl_->smoke_rigid_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a rigid coupling");
    std::unique_ptr<SmokeStorage> tombstone;
    try { tombstone = std::make_unique<SmokeStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "smoke tombstone allocation failed"); }
    tombstone->generation = impl_->smokes[id.index]->generation + 1U;
    if (tombstone->generation == 0U) tombstone->generation = 1U;
    impl_->smokes[id.index] = std::move(tombstone);
    ++impl_->revision;
    return success();
}

Status World::smoke_view(SmokeId id, SmokeDeviceView &output) const noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->smokes.size() || !impl_->smokes[id.index] ||
        !impl_->smokes[id.index]->alive ||
        impl_->smokes[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "smoke handle is stale");
    if (impl_->frame && !impl_->frame->acknowledged)
        return failure(StatusCode::busy, "smoke view requires a completed frame");
    const auto &smoke = *impl_->smokes[id.index];
    output = {{smoke.positions, smoke.count}, {smoke.velocities, smoke.count},
        {smoke.ages, smoke.count}, {smoke.number_densities, smoke.count},
        {smoke.pressures, smoke.count}, {smoke.vorticities, smoke.count},
        smoke.count,
        smoke.options.lifetime, smoke.options.particle_radius, impl_->revision,
        {smoke.grid.velocity, smoke.grid.cell_count},
        {smoke.grid.pressure[0], smoke.grid.cell_count},
        {smoke.grid.density, smoke.grid.cell_count},
        {smoke.grid.temperature, smoke.grid.cell_count},
        {smoke.grid.solid, smoke.grid.cell_count},
        {smoke.grid.vorticity, smoke.grid.cell_count},
        {smoke.grid.divergence, smoke.grid.cell_count},
        smoke.grid.resolution, smoke.grid.height,
        smoke.grid.minimum, smoke.grid.spacing,
        smoke.grid.resolution != 0U
            ? *smoke.grid.residual_max /
                fmaxf(*smoke.grid.rhs_max, 1.0e-5F)
            : 0.0F};
    return success();
}

Status World::add_fluid_smoke_coupling(
    FluidSmokeCouplingOptions options, FluidSmokeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation)
        return failure(StatusCode::invalid_handle, "smoke handle is invalid or stale");
    const float rotation_norm_squared =
        options.heater.orientation.x * options.heater.orientation.x +
        options.heater.orientation.y * options.heater.orientation.y +
        options.heater.orientation.z * options.heater.orientation.z +
        options.heater.orientation.w * options.heater.orientation.w;
    if (!finite(options.heater.center) || !finite(options.heater.orientation) ||
        rotation_norm_squared < 0.25F || rotation_norm_squared > 4.0F ||
        !finite(options.heater.half_extents.x) ||
        !finite(options.heater.half_extents.y) ||
        options.heater.half_extents.x <= 0.0F ||
        options.heater.half_extents.y <= 0.0F ||
        !finite(options.heater_temperature) ||
        !finite(options.boiling_temperature) ||
        !finite(options.heat_transfer_rate) || options.heat_transfer_rate < 0.0F ||
        !finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.steam_rise_speed) || options.steam_rise_speed < 0.0F)
        return failure(StatusCode::invalid_argument, "invalid fluid smoke coupling options");
    std::uint32_t slot = 0U;
    while (slot < impl_->fluid_smoke_couplings.size() &&
           impl_->fluid_smoke_couplings[slot] &&
           impl_->fluid_smoke_couplings[slot]->alive) ++slot;
    if (slot == impl_->fluid_smoke_couplings.size())
        return failure(StatusCode::capacity_exceeded, "fluid smoke coupling capacity exhausted");
    std::unique_ptr<FluidSmokeCouplingSlot> coupling;
    try { coupling = std::make_unique<FluidSmokeCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "fluid smoke coupling allocation failed"); }
    coupling->generation = impl_->fluid_smoke_couplings[slot] ?
        impl_->fluid_smoke_couplings[slot]->generation : 1U;
    coupling->options = options;
    if (!(status = allocate_managed(coupling->converted, 1U))) return status;
    *coupling->converted = 0U;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->fluid_smoke_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status World::remove_fluid_smoke_coupling(FluidSmokeCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_smoke_couplings.size() ||
        !impl_->fluid_smoke_couplings[id.index] ||
        !impl_->fluid_smoke_couplings[id.index]->alive ||
        impl_->fluid_smoke_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid smoke coupling handle is stale");
    std::unique_ptr<FluidSmokeCouplingSlot> tombstone;
    try { tombstone = std::make_unique<FluidSmokeCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "fluid smoke coupling removal failed"); }
    tombstone->generation = id.generation + 1U;
    if (tombstone->generation == 0U) tombstone->generation = 1U;
    impl_->fluid_smoke_couplings[id.index] = std::move(tombstone);
    ++impl_->revision;
    return success();
}

Status World::add_smoke_soft_body_coupling(
    SmokeSoftBodyCouplingOptions options,
    SmokeSoftBodyCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation ||
        options.soft_body.index >= impl_->soft_bodies.size() ||
        !impl_->soft_bodies[options.soft_body.index] ||
        !impl_->soft_bodies[options.soft_body.index]->alive ||
        impl_->soft_bodies[options.soft_body.index]->generation != options.soft_body.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke or soft-body coupling handle is stale");
    if (!finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke soft-body coupling parameters");
    for (const auto &existing : impl_->smoke_soft_body_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.soft_body == options.soft_body)
            return failure(StatusCode::invalid_argument,
                           "smoke soft-body coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_soft_body_couplings.size() &&
           impl_->smoke_soft_body_couplings[slot] &&
           impl_->smoke_soft_body_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_soft_body_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke soft-body coupling capacity exhausted");
    std::unique_ptr<SmokeSoftBodyCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeSoftBodyCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke soft-body coupling allocation failed"); }
    coupling->generation = impl_->smoke_soft_body_couplings[slot] ?
        impl_->smoke_soft_body_couplings[slot]->generation : 1U;
    coupling->options = options;
    if (!(status = allocate_managed(coupling->minimum, 1U)) ||
        !(status = allocate_managed(coupling->maximum, 1U))) return status;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_soft_body_couplings[slot] = std::move(coupling);
    impl_->smokes[options.smoke.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status World::remove_smoke_soft_body_coupling(
    SmokeSoftBodyCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_soft_body_couplings.size() ||
        !impl_->smoke_soft_body_couplings[id.index] ||
        !impl_->smoke_soft_body_couplings[id.index]->alive ||
        impl_->smoke_soft_body_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke soft-body coupling handle is stale");
    std::unique_ptr<SmokeSoftBodyCouplingSlot> tombstone;
    try { tombstone = std::make_unique<SmokeSoftBodyCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke soft-body coupling removal failed"); }
    tombstone->generation = id.generation + 1U;
    if (tombstone->generation == 0U) tombstone->generation = 1U;
    const auto smoke_id = impl_->smoke_soft_body_couplings[id.index]->options.smoke;
    impl_->smoke_soft_body_couplings[id.index] = std::move(tombstone);
    impl_->smokes[smoke_id.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status World::add_smoke_cloth_coupling(
    SmokeClothCouplingOptions options, SmokeClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation ||
        options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation != options.cloth.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke or cloth coupling handle is stale");
    if (!finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke cloth coupling parameters");
    for (const auto &existing : impl_->smoke_cloth_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                           "smoke cloth coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_cloth_couplings.size() &&
           impl_->smoke_cloth_couplings[slot] &&
           impl_->smoke_cloth_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke cloth coupling capacity exhausted");
    std::unique_ptr<SmokeClothCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeClothCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke cloth coupling allocation failed"); }
    coupling->generation = impl_->smoke_cloth_couplings[slot] ?
        impl_->smoke_cloth_couplings[slot]->generation : 1U;
    coupling->options = options;
    if (!(status = allocate_managed(coupling->minimum, 1U)) ||
        !(status = allocate_managed(coupling->maximum, 1U))) return status;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_cloth_couplings[slot] = std::move(coupling);
    impl_->smokes[options.smoke.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status World::remove_smoke_cloth_coupling(SmokeClothCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_cloth_couplings.size() ||
        !impl_->smoke_cloth_couplings[id.index] ||
        !impl_->smoke_cloth_couplings[id.index]->alive ||
        impl_->smoke_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke cloth coupling handle is stale");
    std::unique_ptr<SmokeClothCouplingSlot> tombstone;
    try { tombstone = std::make_unique<SmokeClothCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke cloth coupling removal failed"); }
    tombstone->generation = id.generation + 1U;
    if (tombstone->generation == 0U) tombstone->generation = 1U;
    const auto smoke_id = impl_->smoke_cloth_couplings[id.index]->options.smoke;
    impl_->smoke_cloth_couplings[id.index] = std::move(tombstone);
    impl_->smokes[smoke_id.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status World::add_smoke_rope_coupling(
    SmokeRopeCouplingOptions options, SmokeRopeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation ||
        options.rope.index >= impl_->ropes.size() ||
        !impl_->ropes[options.rope.index] ||
        !impl_->ropes[options.rope.index]->alive ||
        impl_->ropes[options.rope.index]->generation != options.rope.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke or rope coupling handle is stale");
    if (!finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke rope coupling parameters");
    for (const auto &existing : impl_->smoke_rope_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.rope == options.rope)
            return failure(StatusCode::invalid_argument,
                           "smoke rope coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_rope_couplings.size() &&
           impl_->smoke_rope_couplings[slot] &&
           impl_->smoke_rope_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_rope_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke rope coupling capacity exhausted");
    std::unique_ptr<SmokeRopeCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeRopeCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke rope coupling allocation failed"); }
    coupling->generation = impl_->smoke_rope_couplings[slot]
        ? impl_->smoke_rope_couplings[slot]->generation : 1U;
    coupling->options = options;
    if (!(status = allocate_managed(coupling->minimum, 1U)) ||
        !(status = allocate_managed(coupling->maximum, 1U))) return status;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_rope_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status World::remove_smoke_rope_coupling(SmokeRopeCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_rope_couplings.size() ||
        !impl_->smoke_rope_couplings[id.index] ||
        !impl_->smoke_rope_couplings[id.index]->alive ||
        impl_->smoke_rope_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke rope coupling handle is stale");
    std::unique_ptr<SmokeRopeCouplingSlot> tombstone;
    try { tombstone = std::make_unique<SmokeRopeCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke rope coupling removal failed"); }
    tombstone->generation = id.generation + 1U;
    if (tombstone->generation == 0U) tombstone->generation = 1U;
    impl_->smoke_rope_couplings[id.index] = std::move(tombstone);
    ++impl_->revision;
    return success();
}

Status World::add_smoke_rigid_coupling(
    SmokeRigidCouplingOptions options, SmokeRigidCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation)
        return failure(StatusCode::invalid_handle, "smoke coupling handle is stale");
    std::uint32_t dense = 0U;
    if (!(status = impl_->validate_handle(options.body, dense))) return status;
    if (!finite(options.air_density) || options.air_density < 0.0F ||
        !finite(options.drag_coefficient) || options.drag_coefficient < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke rigid coupling parameters");
    for (const auto &existing : impl_->smoke_rigid_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.body == options.body)
            return failure(StatusCode::invalid_argument,
                           "smoke rigid coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_rigid_couplings.size() &&
           impl_->smoke_rigid_couplings[slot] &&
           impl_->smoke_rigid_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_rigid_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke rigid coupling capacity exhausted");
    std::unique_ptr<SmokeRigidCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeRigidCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke rigid coupling allocation failed"); }
    coupling->generation = impl_->smoke_rigid_couplings[slot]
        ? impl_->smoke_rigid_couplings[slot]->generation : 1U;
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_rigid_couplings[slot] = std::move(coupling);
    impl_->smokes[options.smoke.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status World::remove_smoke_rigid_coupling(SmokeRigidCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_rigid_couplings.size() ||
        !impl_->smoke_rigid_couplings[id.index] ||
        !impl_->smoke_rigid_couplings[id.index]->alive ||
        impl_->smoke_rigid_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke rigid coupling handle is stale");
    std::unique_ptr<SmokeRigidCouplingSlot> tombstone;
    try { tombstone = std::make_unique<SmokeRigidCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke rigid coupling removal failed"); }
    tombstone->generation = id.generation + 1U;
    if (tombstone->generation == 0U) tombstone->generation = 1U;
    const auto smoke_id = impl_->smoke_rigid_couplings[id.index]->options.smoke;
    impl_->smoke_rigid_couplings[id.index] = std::move(tombstone);
    impl_->smokes[smoke_id.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

namespace {
[[nodiscard]] bool valid_particle_plane(ParticlePlane plane) noexcept {
    const float squared = plane.orientation.x * plane.orientation.x +
        plane.orientation.y * plane.orientation.y +
        plane.orientation.z * plane.orientation.z +
        plane.orientation.w * plane.orientation.w;
    return finite(plane.center) && finite(plane.orientation) &&
        finite(plane.half_extents.x) && plane.half_extents.x > 0.0F &&
        finite(plane.half_extents.y) && plane.half_extents.y > 0.0F &&
        squared > 0.25F && squared < 4.0F;
}
} // namespace

Status World::add_particle_source(ParticleSourceMesh mesh, ParticleSourceOptions options,
                                       ParticleSourceId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (mesh.spacing == 0.0F) mesh.spacing = fluid->options.support_radius;
    if (!finite(options.initial_velocity) || !finite(options.initial_temperature) ||
        !finite(mesh.spacing) ||
        mesh.spacing < 2.0F * fluid->options.particle_radius)
        return failure(StatusCode::invalid_argument, "fluid source spacing or velocity is invalid");
    for (std::uint32_t index = 0; index < impl_->particle_sources.size(); ++index) {
        ParticleSourceSlot &plane = impl_->particle_sources[index];
        if (plane.alive) continue;
        std::vector<Vec3> points;
        if (!(status = sample_fluid_source(mesh, points))) return status;
        std::unique_ptr<ParticleSourceData> data(new (std::nothrow) ParticleSourceData());
        if (!data) return failure(StatusCode::out_of_memory, "fluid source allocation failed");
        data->count = static_cast<std::uint32_t>(points.size());
        data->spacing = mesh.spacing;
        if (!(status = allocate_managed(data->points, points.size())) ||
            !(status = allocate_managed(data->vacant, points.size())) ||
            !(status = allocate_managed(data->capacity_misses, 1))) return status;
        std::copy(points.begin(), points.end(), data->points);
        *data->capacity_misses = 0;
        plane.options = options;
        plane.data = std::move(data);
        plane.alive = true;
        output = {index, plane.generation};
        return success();
    }
    return failure(StatusCode::capacity_exceeded, "fluid source capacity exhausted");
}

Status World::update_particle_source(ParticleSourceId id,
                                          ParticleSourceOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->particle_sources.size() ||
        !impl_->particle_sources[id.index].alive ||
        impl_->particle_sources[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid source handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!finite(options.initial_velocity) ||
        !finite(options.initial_temperature) ||
        !(options.fluid == impl_->particle_sources[id.index].options.fluid))
        return failure(StatusCode::invalid_argument, "fluid source destination is immutable or velocity is invalid");
    impl_->particle_sources[id.index].options = options;
    return success();
}

Status World::remove_particle_source(ParticleSourceId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->particle_sources.size() ||
        !impl_->particle_sources[id.index].alive ||
        impl_->particle_sources[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid source handle is stale");
    impl_->particle_sources[id.index].alive = false;
    impl_->particle_sources[id.index].data.reset();
    ++impl_->particle_sources[id.index].generation;
    return success();
}

Status World::add_particle_destroy_plane(ParticleDestroyPlaneOptions options,
                                         ParticleDestroyPlaneId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane))
        return failure(StatusCode::invalid_argument, "destroy plane options are invalid");
    for (std::uint32_t index = 0; index < impl_->destroy_planes.size(); ++index) {
        DestroyPlaneSlot &plane = impl_->destroy_planes[index];
        if (plane.alive) continue;
        plane.options = options;
        plane.alive = true;
        output = {index, plane.generation};
        return success();
    }
    return failure(StatusCode::capacity_exceeded, "destroy plane capacity exhausted");
}

Status World::update_particle_destroy_plane(ParticleDestroyPlaneId id,
                                            ParticleDestroyPlaneOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->destroy_planes.size() ||
        !impl_->destroy_planes[id.index].alive ||
        impl_->destroy_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "destroy plane handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane))
        return failure(StatusCode::invalid_argument, "destroy plane options are invalid");
    impl_->destroy_planes[id.index].options = options;
    return success();
}

Status World::remove_particle_destroy_plane(ParticleDestroyPlaneId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->destroy_planes.size() ||
        !impl_->destroy_planes[id.index].alive ||
        impl_->destroy_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "destroy plane handle is stale");
    impl_->destroy_planes[id.index].alive = false;
    ++impl_->destroy_planes[id.index].generation;
    return success();
}

// Rendering seams duplicate positions. Weld those positions only for the
// topology check; the original indexed triangle mesh remains unchanged.
static std::vector<CollisionPlane> closed_convex_planes(
    const Vec3 *vertices, std::uint32_t vertex_count,
    const std::vector<std::uint32_t> &indices) {
    std::map<std::array<float, 3>, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(vertex_count);
    Vec3 center{};
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
        const Vec3 p = vertices[vertex];
        const auto entry = welded.emplace(std::array<float, 3>{p.x, p.y, p.z},
            static_cast<std::uint32_t>(welded.size()));
        remap[vertex] = entry.first->second;
        if (entry.second) center = add(center, p);
    }
    if (welded.size() < 4U) return {};
    center = multiply(center, 1.0F / static_cast<float>(welded.size()));
    std::unordered_map<std::uint64_t, std::uint32_t> edges;
    for (std::size_t index = 0U; index < indices.size(); index += 3U)
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const auto a = remap[indices[index + edge]];
            const auto b = remap[indices[index + (edge + 1U) % 3U]];
            ++edges[(static_cast<std::uint64_t>(std::min(a, b)) << 32U) |
                     std::max(a, b)];
        }
    for (const auto &edge : edges)
        if (edge.second != 2U) return {};
    float scale = 0.0F;
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
        scale = std::max(scale, vector_length(subtract(vertices[vertex], center)));
    const float tolerance = std::max(1.0e-7F, scale * 1.0e-5F);
    std::vector<CollisionPlane> planes;
    planes.reserve(indices.size() / 3U);
    for (std::size_t index = 0U; index < indices.size(); index += 3U) {
        const Vec3 a = vertices[indices[index]];
        Vec3 normal = normalized_or(cross(
            subtract(vertices[indices[index + 1U]], a),
            subtract(vertices[indices[index + 2U]], a)), {});
        float side = dot(normal, subtract(a, center));
        if (fabsf(side) <= tolerance) return {};
        if (side < 0.0F) normal = multiply(normal, -1.0F);
        const float offset = dot(normal, a);
        for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
            if (dot(normal, vertices[vertex]) - offset > tolerance) return {};
        planes.push_back({normal, offset});
    }
    return planes;
}

// A concave compound can contain individually closed convex shells. Keep
// their outward normals for one-sided contact validation, without replacing
// any triangles or treating the entire compound as its convex hull.
static std::vector<Vec3> convex_shell_normals(
    const Vec3 *vertices, std::uint32_t vertex_count,
    const std::vector<std::uint32_t> &indices) {
    std::map<std::array<float, 3>, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(vertex_count), parent(vertex_count);
    for (std::uint32_t i = 0; i < vertex_count; ++i) {
        const auto p = vertices[i];
        remap[i] = welded.emplace(
            std::array<float, 3>{p.x, p.y, p.z}, i).first->second;
        parent[i] = i;
    }
    const auto root = [&](std::uint32_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (std::size_t i = 0; i < indices.size(); i += 3)
        for (std::size_t corner = 1; corner < 3; ++corner) {
            const auto a = root(remap[indices[i]]);
            const auto b = root(remap[indices[i + corner]]);
            parent[b] = a;
        }
    std::map<std::uint32_t, std::vector<std::uint32_t>> shells;
    for (std::uint32_t t = 0; t < indices.size() / 3U; ++t)
        shells[root(remap[indices[t * 3U]])].push_back(t);
    std::vector<Vec3> normals(indices.size() / 3U);
    bool found = false;
    for (const auto &[id, triangles] : shells) {
        (void)id;
        std::map<std::uint32_t, std::uint32_t> local;
        std::vector<Vec3> points;
        std::vector<std::uint32_t> faces;
        for (auto t : triangles) {
            for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                const auto vertex = remap[indices[t * 3U + corner]];
                const auto [entry, inserted] = local.emplace(
                    vertex, static_cast<std::uint32_t>(points.size()));
                if (inserted) points.push_back(vertices[vertex]);
                faces.push_back(entry->second);
            }
        }
        const auto planes = closed_convex_planes(
            points.data(), static_cast<std::uint32_t>(points.size()), faces);
        if (planes.empty()) continue;
        found = true;
        for (std::size_t i = 0; i < triangles.size(); ++i)
            normals[triangles[i]] = planes[i].normal;
    }
    return found ? normals : std::vector<Vec3>{};
}

Status World::add_triangle_mesh(
    DeviceSpan<const Vec3> vertices,
    DeviceSpan<const std::uint32_t> triangle_indices, TriangleMeshId &output,
    cudaStream_t stream) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (vertices.data == nullptr || triangle_indices.data == nullptr ||
        vertices.size < 3U || triangle_indices.size < 3U ||
        triangle_indices.size % 3U != 0U ||
        vertices.size > std::numeric_limits<std::uint32_t>::max() ||
        triangle_indices.size > std::numeric_limits<std::uint32_t>::max()) {
        return failure(StatusCode::invalid_argument,
                       "triangle mesh requires device vertices and triangle indices");
    }
    if (impl_->triangle_mesh_count >= impl_->options.triangle_mesh_capacity) {
        return failure(StatusCode::capacity_exceeded,
                       "triangle mesh capacity is exhausted");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0;
         index < impl_->options.triangle_mesh_capacity; ++index) {
        if (!impl_->meshes[index].alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense) {
        return failure(StatusCode::internal_error,
                       "no free triangle mesh slot was found");
    }

    Vec3 *owned_vertices = nullptr;
    std::uint32_t *owned_indices = nullptr;
    status = allocate_managed(owned_vertices,
                              static_cast<std::size_t>(vertices.size));
    if (!status) {
        return status;
    }
    status = allocate_managed(owned_indices,
                              static_cast<std::size_t>(triangle_indices.size));
    if (!status) {
        release_managed(owned_vertices);
        return status;
    }
    cudaError_t error = cudaMemcpyAsync(
        owned_vertices, vertices.data, sizeof(Vec3) * vertices.size,
        cudaMemcpyDefault, stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(owned_indices, triangle_indices.data,
                                sizeof(std::uint32_t) * triangle_indices.size,
                                cudaMemcpyDefault, stream);
    }
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error != cudaSuccess) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return cuda_failure(error, "triangle mesh upload failed");
    }
    for (std::uint64_t index = 0; index < vertices.size; ++index) {
        if (!finite(owned_vertices[index])) {
            release_managed(owned_indices);
            release_managed(owned_vertices);
            return failure(StatusCode::invalid_argument,
                           "triangle mesh contains a non-finite vertex");
        }
    }

    Vec3 minimum = owned_vertices[0];
    Vec3 maximum = owned_vertices[0];
    for (std::uint64_t index = 1; index < vertices.size; ++index) {
        minimum = component_min(minimum, owned_vertices[index]);
        maximum = component_max(maximum, owned_vertices[index]);
    }
    for (std::uint64_t index = 0; index < triangle_indices.size; index += 3U) {
        const std::uint32_t first = owned_indices[index];
        const std::uint32_t second = owned_indices[index + 1U];
        const std::uint32_t third = owned_indices[index + 2U];
        if (first >= vertices.size || second >= vertices.size ||
            third >= vertices.size) {
            release_managed(owned_indices);
            release_managed(owned_vertices);
            return failure(StatusCode::invalid_argument,
                           "triangle mesh index is outside the vertex buffer");
        }
        const Vec3 area = cross(subtract(owned_vertices[second],
                                         owned_vertices[first]),
                                subtract(owned_vertices[third],
                                         owned_vertices[first]));
        if (length_squared(area) <= k_epsilon * k_epsilon) {
            release_managed(owned_indices);
            release_managed(owned_vertices);
            return failure(StatusCode::invalid_argument,
                           "triangle mesh contains a degenerate triangle");
        }
    }

    std::vector<std::uint32_t> triangle_order;
    std::vector<BvhNode> bvh_nodes;
    std::vector<std::uint32_t> reordered_indices;
    try {
        const std::uint32_t triangle_count =
            static_cast<std::uint32_t>(triangle_indices.size / 3U);
        triangle_order.resize(triangle_count);
        std::iota(triangle_order.begin(), triangle_order.end(), 0U);
        bvh_nodes.reserve(triangle_count * 2U);
        const auto coordinate = [](Vec3 value, int axis) {
            return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
        };
        const auto centroid = [&](std::uint32_t triangle) {
            const std::uint32_t offset = triangle * 3U;
            return multiply(
                add(add(owned_vertices[owned_indices[offset]],
                        owned_vertices[owned_indices[offset + 1U]]),
                    owned_vertices[owned_indices[offset + 2U]]),
                1.0F / 3.0F);
        };
        std::function<std::uint32_t(std::uint32_t, std::uint32_t)> build =
            [&](std::uint32_t begin, std::uint32_t end) {
                BvhNode node{};
                node.minimum = {FLT_MAX, FLT_MAX, FLT_MAX};
                node.maximum = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (std::uint32_t item = begin; item < end; ++item) {
                    const std::uint32_t offset = triangle_order[item] * 3U;
                    for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                        const Vec3 vertex =
                            owned_vertices[owned_indices[offset + corner]];
                        node.minimum = component_min(node.minimum, vertex);
                        node.maximum = component_max(node.maximum, vertex);
                    }
                }
                const std::uint32_t node_index =
                    static_cast<std::uint32_t>(bvh_nodes.size());
                bvh_nodes.push_back(node);
                if (end - begin <= 4U) {
                    bvh_nodes[node_index].first_triangle = begin;
                    bvh_nodes[node_index].triangle_count = end - begin;
                    return node_index;
                }
                const Vec3 extent = subtract(node.maximum, node.minimum);
                const int axis = extent.x >= extent.y && extent.x >= extent.z
                                     ? 0
                                     : (extent.y >= extent.z ? 1 : 2);
                std::stable_sort(
                    triangle_order.begin() + begin, triangle_order.begin() + end,
                    [&](std::uint32_t first, std::uint32_t second) {
                        const float first_value = coordinate(centroid(first), axis);
                        const float second_value = coordinate(centroid(second), axis);
                        return first_value < second_value ||
                               (first_value == second_value && first < second);
                    });
                const std::uint32_t middle = begin + (end - begin) / 2U;
                const std::uint32_t left = build(begin, middle);
                const std::uint32_t right = build(middle, end);
                bvh_nodes[node_index].left = left;
                bvh_nodes[node_index].right = right;
                return node_index;
            };
        (void)build(0U, triangle_count);
        reordered_indices.resize(triangle_indices.size);
        for (std::uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
            const std::uint32_t source = triangle_order[triangle] * 3U;
            const std::uint32_t destination = triangle * 3U;
            reordered_indices[destination] = owned_indices[source];
            reordered_indices[destination + 1U] = owned_indices[source + 1U];
            reordered_indices[destination + 2U] = owned_indices[source + 2U];
        }
    } catch (...) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return failure(StatusCode::out_of_memory,
                       "failed to build triangle mesh acceleration data");
    }

    std::vector<std::uint32_t> bvh_leaves;
    std::vector<CollisionPlane> solid_planes;
    std::vector<CollisionPlane> unique_solid_planes;
    std::vector<Vec3> shell_normals;
    try {
        solid_planes = closed_convex_planes(owned_vertices,
            static_cast<std::uint32_t>(vertices.size), reordered_indices);
        unique_solid_planes = detail::unique_convex_planes(solid_planes);
        shell_normals = convex_shell_normals(owned_vertices,
            static_cast<std::uint32_t>(vertices.size), reordered_indices);
        for (std::uint32_t index = 0U; index < bvh_nodes.size(); ++index) {
            if (bvh_nodes[index].triangle_count != 0U) {
                bvh_leaves.push_back(index);
            }
        }
    } catch (...) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return failure(StatusCode::out_of_memory,
                       "failed to index triangle mesh BVH leaves");
    }

    BvhNode *owned_bvh_nodes = nullptr;
    status = allocate_managed(owned_bvh_nodes, bvh_nodes.size());
    if (!status) {
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    std::uint32_t *owned_bvh_leaves = nullptr;
    status = allocate_managed(owned_bvh_leaves, bvh_leaves.size());
    if (!status) {
        release_managed(owned_bvh_nodes);
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    std::copy(reordered_indices.begin(), reordered_indices.end(), owned_indices);
    std::copy(bvh_nodes.begin(), bvh_nodes.end(), owned_bvh_nodes);
    std::copy(bvh_leaves.begin(), bvh_leaves.end(), owned_bvh_leaves);

    CollisionPlane *owned_solid_planes = nullptr;
    status = allocate_managed(owned_solid_planes,
        solid_planes.size() + unique_solid_planes.size());
    if (!status) {
        release_managed(owned_bvh_leaves);
        release_managed(owned_bvh_nodes);
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    if (!solid_planes.empty())
        std::copy(solid_planes.begin(), solid_planes.end(), owned_solid_planes);
    if (!unique_solid_planes.empty())
        std::copy(unique_solid_planes.begin(), unique_solid_planes.end(),
                  owned_solid_planes + solid_planes.size());

    Vec3 *owned_shell_normals = nullptr;
    status = allocate_managed(owned_shell_normals, shell_normals.size());
    if (!status) {
        release_managed(owned_solid_planes);
        release_managed(owned_bvh_leaves);
        release_managed(owned_bvh_nodes);
        release_managed(owned_indices);
        release_managed(owned_vertices);
        return status;
    }
    if (!shell_normals.empty())
        std::copy(shell_normals.begin(), shell_normals.end(), owned_shell_normals);

    TriangleMeshResource &mesh = impl_->meshes[slot];
    mesh.vertices = owned_vertices;
    mesh.indices = owned_indices;
    mesh.vertex_count = static_cast<std::uint32_t>(vertices.size);
    mesh.index_count = static_cast<std::uint32_t>(triangle_indices.size);
    mesh.minimum = minimum;
    mesh.maximum = maximum;
    mesh.bounding_center = multiply(add(minimum, maximum), 0.5F);
    float squared_radius = 0.0F;
    for (std::uint64_t index = 0U; index < vertices.size; ++index) {
        squared_radius = fmaxf(
            squared_radius,
            length_squared(subtract(owned_vertices[index],
                                    mesh.bounding_center)));
    }
    mesh.bounding_radius = sqrtf(squared_radius);
    const Vec3 half_extents = multiply(subtract(maximum, minimum), 0.5F);
    mesh.unit_inertia = {
        fmaxf((half_extents.y * half_extents.y +
               half_extents.z * half_extents.z) /
                  3.0F,
              k_epsilon),
        fmaxf((half_extents.x * half_extents.x +
               half_extents.z * half_extents.z) /
                  3.0F,
              k_epsilon),
        fmaxf((half_extents.x * half_extents.x +
               half_extents.y * half_extents.y) /
                  3.0F,
              k_epsilon)};
    mesh.bvh_nodes = owned_bvh_nodes;
    mesh.bvh_node_count = static_cast<std::uint32_t>(bvh_nodes.size());
    mesh.bvh_leaves = owned_bvh_leaves;
    mesh.bvh_leaf_count = static_cast<std::uint32_t>(bvh_leaves.size());
    mesh.solid_planes = owned_solid_planes;
    mesh.solid_unique_plane_count = static_cast<std::uint32_t>(unique_solid_planes.size());
    mesh.shell_normals = owned_shell_normals;
    mesh.alive = true;
    ++impl_->triangle_mesh_count;
    ++impl_->revision;
    output = {slot, mesh.generation};
    return success();
}

Status World::remove_triangle_mesh(TriangleMeshId mesh_id) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    status = impl_->validate_handle(mesh_id);
    if (!status) {
        return status;
    }
    for (std::uint32_t index = 0; index < impl_->rigid_body_count; ++index) {
        if (impl_->parameters[index].mesh == mesh_id) {
            return failure(StatusCode::invalid_argument,
                           "triangle mesh is still referenced by a rigid body");
        }
    }
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index)
        if (impl_->paint_fields[index].alive &&
            impl_->paint_fields[index].options.mesh == mesh_id)
            return failure(StatusCode::invalid_argument,
                           "triangle mesh is still referenced by a paint field");
    TriangleMeshResource &mesh = impl_->meshes[mesh_id.index];
    release_managed(mesh.bvh_leaves);
    release_managed(mesh.bvh_nodes);
    release_managed(mesh.solid_planes);
    release_managed(mesh.shell_normals);
    release_managed(mesh.indices);
    release_managed(mesh.vertices);
    mesh.vertex_count = 0U;
    mesh.index_count = 0U;
    mesh.bvh_node_count = 0U;
    mesh.bvh_leaf_count = 0U;
    mesh.alive = false;
    ++mesh.generation;
    if (mesh.generation == 0U) {
        mesh.generation = 1U;
    }
    --impl_->triangle_mesh_count;
    ++impl_->revision;
    return success();
}

Status World::add_paint_field(PaintFieldOptions options,
                              PaintFieldId &output,
                              cudaStream_t stream) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    const bool cloth_target = options.cloth.generation != 0U;
    std::uint32_t vertex_count = 0U;
    if (cloth_target) {
        if (options.body.generation != 0U || options.mesh.generation != 0U ||
            options.cloth.index >= impl_->cloths.size() ||
            !impl_->cloths[options.cloth.index] ||
            !impl_->cloths[options.cloth.index]->alive ||
            impl_->cloths[options.cloth.index]->generation != options.cloth.generation)
            return failure(StatusCode::invalid_handle, "paint cloth target is stale");
        const auto &cloth = *impl_->cloths[options.cloth.index];
        vertex_count = cloth.source_indices
            ? static_cast<std::uint32_t>(cloth.source_inverse_masses.size())
            : cloth.vertex_count;
    } else {
        std::uint32_t dense = 0U;
        if (!(status = impl_->validate_handle(options.body, dense)) ||
            !(status = impl_->validate_handle(options.mesh))) return status;
        vertex_count = impl_->meshes[options.mesh.index].vertex_count;
    }
    if (options.vertex_uvs.data == nullptr ||
        options.vertex_uvs.size != vertex_count ||
        options.width == 0U || options.height == 0U ||
        options.width > 4096U || options.height > 4096U ||
        static_cast<std::uint64_t>(options.width) * options.height >
            16'777'216U)
        return failure(StatusCode::invalid_argument,
                       "paint field UV count or dimensions are invalid");
    std::uint32_t slot = 0U;
    for (; slot < impl_->options.paint_field_capacity; ++slot)
        if (!impl_->paint_fields[slot].alive) break;
    if (slot == impl_->options.paint_field_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "paint field capacity exhausted");
    Vec2 *uvs = nullptr;
    std::uint32_t *pixels = nullptr;
    status = allocate_managed(uvs, vertex_count);
    if (!status) return status;
    status = allocate_managed(pixels,
        static_cast<std::size_t>(options.width) * options.height);
    if (!status) { release_managed(uvs); return status; }
    cudaError_t error = cudaMemcpyAsync(uvs, options.vertex_uvs.data,
        vertex_count * sizeof(Vec2), cudaMemcpyDefault, stream);
    if (error == cudaSuccess)
        error = cudaMemsetAsync(pixels, 0,
            static_cast<std::size_t>(options.width) * options.height *
                sizeof(std::uint32_t), stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        release_managed(pixels);
        release_managed(uvs);
        return cuda_failure(error, "paint field upload failed");
    }
    for (std::uint32_t i = 0; i < vertex_count; ++i)
        if (!finite(uvs[i].x) || !finite(uvs[i].y)) {
            release_managed(pixels);
            release_managed(uvs);
            return failure(StatusCode::invalid_argument,
                           "paint field contains non-finite UVs");
        }
    PaintFieldResource &field = impl_->paint_fields[slot];
    options.vertex_uvs = {uvs, vertex_count};
    field.options = options;
    field.uvs = uvs;
    field.pixels = pixels;
    field.alive = true;
    output = {slot, field.generation};
    ++impl_->revision;
    return success();
}

Status World::remove_paint_field(PaintFieldId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    for (std::uint32_t i = 0; i < impl_->options.paint_rule_capacity; ++i)
        if (impl_->paint_rules[i].alive &&
            impl_->paint_rules[i].options.target == id)
            return failure(StatusCode::invalid_argument,
                           "paint field is still referenced by a paint rule");
    PaintFieldResource &field = impl_->paint_fields[id.index];
    release_managed(field.pixels);
    release_managed(field.uvs);
    field.alive = false;
    ++field.generation;
    if (field.generation == 0U) field.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::clear_paint_field(PaintFieldId id, cudaStream_t stream) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    const PaintFieldResource &field = impl_->paint_fields[id.index];
    cudaError_t error = cudaMemsetAsync(field.pixels, 0,
        static_cast<std::size_t>(field.options.width) *
        field.options.height * sizeof(std::uint32_t), stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        return cuda_failure(error, "paint field clear failed");
    return success();
}

Status World::paint_field_view(PaintFieldId id,
                               PaintFieldDeviceView &output) const noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    const PaintFieldResource &field = impl_->paint_fields[id.index];
    output = {{field.pixels, static_cast<std::uint64_t>(field.options.width) *
                                 field.options.height},
              field.options.width, field.options.height, impl_->revision};
    return success();
}

Status World::add_paint_rule(PaintRuleOptions options,
                             PaintRuleId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    const bool fluid_source = options.source.generation != 0U;
    const bool rigid_source = options.rigid_source.generation != 0U;
    if (fluid_source == rigid_source)
        return failure(StatusCode::invalid_argument,
                       "paint rule needs exactly one source");
    if (fluid_source) {
        FluidStorage *fluid = nullptr;
        if (!(status = impl_->validate_handle(options.source, fluid))) return status;
    } else {
        std::uint32_t dense = 0U;
        if (!(status = impl_->validate_handle(options.rigid_source, dense)))
            return status;
    }
    if (options.target.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[options.target.index].alive ||
        impl_->paint_fields[options.target.index].generation !=
            options.target.generation)
        return failure(StatusCode::invalid_handle, "paint target field is stale");
    const PaintFieldResource &target =
        impl_->paint_fields[options.target.index];
    if ((fluid_source && target.options.cloth.generation != 0U) ||
        (rigid_source && target.options.cloth.generation == 0U))
        return failure(StatusCode::not_supported,
                       "paint source and target systems do not match");
    if (!finite(options.reach) || options.reach < 0.0F ||
        options.reach > 10.0F)
        return failure(StatusCode::invalid_argument, "paint reach is invalid");
    if (!finite(options.brush_radius) || options.brush_radius <= 0.0F ||
        options.brush_radius > 10.0F)
        return failure(StatusCode::invalid_argument,
                       "paint brush radius is invalid");
    std::uint32_t slot = 0U;
    for (; slot < impl_->options.paint_rule_capacity; ++slot)
        if (!impl_->paint_rules[slot].alive) break;
    if (slot == impl_->options.paint_rule_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "paint rule capacity exhausted");
    PaintRuleResource &rule = impl_->paint_rules[slot];
    rule.options = options;
    rule.alive = true;
    ++impl_->paint_rule_count;
    output = {slot, rule.generation};
    ++impl_->revision;
    return success();
}

Status World::remove_paint_rule(PaintRuleId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_rule_capacity ||
        !impl_->paint_rules[id.index].alive ||
        impl_->paint_rules[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint rule handle is stale");
    PaintRuleResource &rule = impl_->paint_rules[id.index];
    rule.alive = false;
    --impl_->paint_rule_count;
    ++rule.generation;
    if (rule.generation == 0U) rule.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::add_cloth(ClothOptions options, ClothId &output,
                        cudaStream_t stream) noexcept {
    (void)stream;
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.vertices.data == nullptr || options.vertices.size < 3U ||
        options.vertices.size > UINT32_MAX ||
        options.triangle_indices.data == nullptr ||
        options.triangle_indices.size == 0U ||
        options.triangle_indices.size % 3U != 0U ||
        options.triangle_indices.size > UINT32_MAX ||
        (options.inverse_masses.size != 0U &&
         (options.inverse_masses.data == nullptr ||
          options.inverse_masses.size != options.vertices.size)) ||
        !finite(options.vertex_mass) || options.vertex_mass <= 0.0F ||
        !finite(options.thickness) || options.thickness <= 0.0F ||
        !finite(options.stretch_compliance) || options.stretch_compliance < 0.0F ||
        !finite(options.bending_compliance) || options.bending_compliance < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.contact_friction) || options.contact_friction < 0.0F ||
        !finite(options.break_strain) || options.break_strain < 0.0F ||
        options.break_strain > 9.0F ||
        !finite(options.impact_break_impulse) ||
        options.impact_break_impulse < 0.0F ||
        !finite(options.target_volume) || options.target_volume < 0.0F ||
        !finite(options.volume_compliance) ||
        options.volume_compliance < 0.0F ||
        (options.preserve_volume &&
         (options.break_strain > 0.0F || options.impact_break_impulse > 0.0F)) ||
        options.fracture_persistence_substeps == 0U ||
        options.fracture_persistence_substeps > 64U ||
        options.solver_iterations == 0U || options.solver_iterations > 64U) {
        return failure(StatusCode::invalid_argument, "invalid cloth geometry or solver options");
    }
    for (std::uint64_t vertex = 0U; vertex < options.vertices.size; ++vertex) {
        if (!finite(options.vertices.data[vertex]) ||
            (options.inverse_masses.size != 0U &&
             (!finite(options.inverse_masses.data[vertex]) ||
              options.inverse_masses.data[vertex] < 0.0F)))
            return failure(StatusCode::invalid_argument, "invalid cloth vertex or inverse mass");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
        if (!impl_->cloths[index] || !impl_->cloths[index]->alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense)
        return failure(StatusCode::capacity_exceeded, "cloth capacity is exhausted");

    const std::uint32_t count = static_cast<std::uint32_t>(options.vertices.size);
    const bool fracture_enabled = options.break_strain > 0.0F ||
        options.impact_break_impulse > 0.0F;
    if (fracture_enabled && options.vertices.size + options.triangle_indices.size > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth split capacity exceeds uint32 range");
    const auto capacity = count + (fracture_enabled
        ? static_cast<std::uint32_t>(options.triangle_indices.size) : 0U);
    std::vector<std::vector<DeformableNeighbor>> adjacency;
    std::vector<std::uint32_t> offsets;
    std::vector<DeformableNeighbor> neighbors;
    std::vector<ClothBond> bonds;
    std::vector<Vec3> surface_rest;
    std::vector<std::uint32_t> surface_indices;
    std::vector<std::uint32_t> triangle_bonds;
    std::vector<std::uint32_t> volume_corner_offsets;
    std::vector<std::uint32_t> volume_corner_indices;
    float initial_signed_volume = 0.0F;
    try {
        adjacency.resize(count);
        std::unordered_map<std::uint64_t, std::uint32_t> bond_ids;
        std::unordered_map<std::uint64_t, std::uint32_t> opposite;
        std::unordered_map<std::uint64_t, std::uint32_t> edge_counts;
        const auto key = [](std::uint32_t a, std::uint32_t b) {
            if (a > b) std::swap(a, b);
            return (static_cast<std::uint64_t>(a) << 32U) | b;
        };
        const auto link = [&](std::uint32_t a, std::uint32_t b,
                              float compliance, bool bending) {
            if (a == b || bond_ids.find(key(a, b)) != bond_ids.end()) return;
            const float rest = vector_length(subtract(
                options.vertices.data[a], options.vertices.data[b]));
            if (rest <= 1.0e-6F) return;
            const auto id = static_cast<std::uint32_t>(bonds.size());
            bond_ids.emplace(key(a, b), id);
            bonds.push_back({a, b, rest, bending});
            adjacency[a].push_back({b, rest, compliance, id});
            adjacency[b].push_back({a, rest, compliance, id});
        };
        for (std::uint64_t triangle = 0U;
             triangle < options.triangle_indices.size; triangle += 3U) {
            const std::uint32_t a = options.triangle_indices.data[triangle];
            const std::uint32_t b = options.triangle_indices.data[triangle + 1U];
            const std::uint32_t c = options.triangle_indices.data[triangle + 2U];
            if (a >= count || b >= count || c >= count || a == b || b == c || c == a)
                return failure(StatusCode::invalid_argument, "invalid cloth triangle index");
            if (vector_length(subtract(options.vertices.data[a],
                                       options.vertices.data[b])) <= 1.0e-6F ||
                vector_length(subtract(options.vertices.data[b],
                                       options.vertices.data[c])) <= 1.0e-6F ||
                vector_length(subtract(options.vertices.data[c],
                                       options.vertices.data[a])) <= 1.0e-6F)
                return failure(StatusCode::invalid_argument,
                               "cloth triangle has a zero-length edge");
            initial_signed_volume += dot(options.vertices.data[a], cross(
                options.vertices.data[b], options.vertices.data[c])) / 6.0F;
            const std::array<std::array<std::uint32_t, 3>, 3> edges{{
                {a, b, c}, {b, c, a}, {c, a, b}}};
            for (const auto &edge : edges) {
                const std::uint64_t edge_key = key(edge[0], edge[1]);
                ++edge_counts[edge_key];
                const auto previous = opposite.find(edge_key);
                if (previous == opposite.end()) {
                    opposite.emplace(edge_key, edge[2]);
                    link(edge[0], edge[1], options.stretch_compliance, false);
                } else {
                    link(previous->second, edge[2], options.bending_compliance,
                         true);
                }
            }
        }
        if (options.preserve_volume) {
            if (fabsf(initial_signed_volume) <= 1.0e-8F ||
                std::any_of(edge_counts.begin(), edge_counts.end(),
                    [](const auto &edge) { return edge.second != 2U; }))
                return failure(StatusCode::invalid_argument,
                    "volume-preserving cloth must be a closed manifold");
            std::vector<std::vector<std::uint32_t>> corners(count);
            for (std::uint64_t corner = 0U;
                 corner < options.triangle_indices.size; ++corner)
                corners[options.triangle_indices.data[corner]].push_back(
                    static_cast<std::uint32_t>(corner));
            volume_corner_offsets.reserve(static_cast<std::size_t>(count) + 1U);
            volume_corner_indices.reserve(options.triangle_indices.size);
            volume_corner_offsets.push_back(0U);
            for (const auto &list : corners) {
                volume_corner_indices.insert(volume_corner_indices.end(),
                    list.begin(), list.end());
                volume_corner_offsets.push_back(
                    static_cast<std::uint32_t>(volume_corner_indices.size()));
            }
        }
        for (std::uint64_t triangle = 0U;
             triangle < options.triangle_indices.size; triangle += 3U) {
            const std::uint32_t corners[3]{
                options.triangle_indices.data[triangle],
                options.triangle_indices.data[triangle + 1U],
                options.triangle_indices.data[triangle + 2U]};
            for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
                surface_rest.push_back(options.vertices.data[corners[corner]]);
                surface_indices.push_back(static_cast<std::uint32_t>(triangle + corner));
                triangle_bonds.push_back(bond_ids.at(key(corners[corner],
                    corners[(corner + 1U) % 3U])));
            }
        }
        offsets.reserve(static_cast<std::size_t>(count) + 1U);
        offsets.push_back(0U);
        for (const auto &list : adjacency) {
            if (neighbors.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded, "cloth links exceed uint32 range");
            neighbors.insert(neighbors.end(), list.begin(), list.end());
            offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory, "failed to build cloth links");
    }
    std::unique_ptr<ClothStorage> cloth;
    try { cloth = std::make_unique<ClothStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "failed to allocate cloth"); }
    cloth->generation = impl_->cloths[slot] ? impl_->cloths[slot]->generation : 1U;
    cloth->vertex_count = count;
    cloth->vertex_capacity = capacity;
    cloth->stretch_compliance = options.stretch_compliance;
    cloth->bending_compliance = options.bending_compliance;
    cloth->index_count = static_cast<std::uint32_t>(options.triangle_indices.size);
    cloth->neighbor_count = static_cast<std::uint32_t>(neighbors.size());
    cloth->neighbor_capacity = fracture_enabled
        ? 2U * (options.triangle_indices.size + bonds.size()) : neighbors.size();
    if (cloth->neighbor_capacity > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth split links exceed uint32 range");
    if (bonds.size() > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth bonds exceed uint32 range");
    cloth->bond_count = static_cast<std::uint32_t>(bonds.size());
    cloth->thickness = options.thickness;
    cloth->velocity_damping = options.velocity_damping;
    cloth->contact_friction = options.contact_friction;
    cloth->break_strain = options.break_strain;
    cloth->fracture_persistence_substeps =
        options.fracture_persistence_substeps;
    cloth->impact_break_impulse = options.impact_break_impulse;
    cloth->solver_iterations = options.solver_iterations;
    cloth->preserve_volume = options.preserve_volume;
    cloth->target_volume = options.target_volume > 0.0F
        ? options.target_volume : fabsf(initial_signed_volume);
    cloth->volume_compliance = options.volume_compliance;
    cloth->orientation = initial_signed_volume < 0.0F ? -1.0F : 1.0F;
    status = allocate_managed(cloth->positions, capacity); if (!status) return status;
    status = allocate_managed(cloth->scratch, capacity); if (!status) return status;
    status = allocate_managed(cloth->previous, capacity); if (!status) return status;
    status = allocate_managed(cloth->velocities, capacity); if (!status) return status;
    status = allocate_managed(cloth->inverse_masses, capacity); if (!status) return status;
    status = allocate_managed(cloth->indices, cloth->index_count); if (!status) return status;
    if (fracture_enabled) {
        status = allocate_managed(cloth->source_indices, cloth->index_count);
        if (!status) return status;
        status = allocate_managed(cloth->vertex_sources, capacity);
        if (!status) return status;
        status = allocate_managed(cloth->free_triangle_nodes, capacity);
        if (!status) return status;
        status = allocate_managed(cloth->surface_positions, surface_rest.size());
        if (!status) return status;
        status = allocate_managed(cloth->surface_triangle_indices,
                                  surface_indices.size());
        if (!status) return status;
        status = allocate_managed(cloth->triangle_bonds, triangle_bonds.size());
        if (!status) return status;
    }
    status = allocate_managed(cloth->bonds, bonds.size()); if (!status) return status;
    status = allocate_managed(cloth->bond_active, bonds.size());
    if (!status) return status;
    status = allocate_managed(cloth->bond_damage, bonds.size());
    if (!status) return status;
    status = allocate_managed(cloth->offsets, static_cast<std::size_t>(capacity) + 1U); if (!status) return status;
    status = allocate_managed(cloth->neighbors, cloth->neighbor_capacity); if (!status) return status;
    status = allocate_managed(cloth->body_impulses, capacity); if (!status) return status;
    status = allocate_managed(cloth->rigid_contact_forces, capacity);
    if (!status) return status;
    status = allocate_managed(cloth->soft_body_forces, capacity);
    if (!status) return status;
    if (impl_->options.rope_cloth_coupling_capacity != 0U) {
        status = allocate_managed(cloth->rope_forces, capacity);
        if (!status) return status;
    }
    status = allocate_managed(cloth->body_corrections,
                              impl_->options.rigid_body_capacity);
    if (!status) return status;
    if (options.preserve_volume) {
        status = allocate_managed(cloth->volume_gradients, count);
        if (!status) return status;
        status = allocate_managed(cloth->volume_corner_offsets,
            volume_corner_offsets.size());
        if (!status) return status;
        status = allocate_managed(cloth->volume_corner_indices,
            volume_corner_indices.size());
        if (!status) return status;
        status = allocate_managed(cloth->volume_lambda, 1U);
        if (!status) return status;
        *cloth->volume_lambda = 0.0F;
        std::copy(volume_corner_offsets.begin(), volume_corner_offsets.end(),
            cloth->volume_corner_offsets);
        std::copy(volume_corner_indices.begin(), volume_corner_indices.end(),
            cloth->volume_corner_indices);
    }
    if (impl_->options.fluid_cloth_coupling_capacity != 0U) {
        status = allocate_managed(cloth->fluid_forces, capacity);
        if (!status) return status;
    }
    if (impl_->options.smoke_cloth_coupling_capacity != 0U) {
        status = allocate_managed(cloth->smoke_forces, capacity);
        if (!status) return status;
    }
    status = allocate_managed(cloth->count, 1U); if (!status) return status;
    for (std::uint32_t index = 0U; index < count; ++index) {
        cloth->positions[index] = options.vertices.data[index];
        cloth->scratch[index] = options.vertices.data[index];
        cloth->previous[index] = options.vertices.data[index];
        cloth->velocities[index] = {};
        cloth->rigid_contact_forces[index] = {};
        cloth->soft_body_forces[index] = {};
        if (cloth->rope_forces != nullptr) cloth->rope_forces[index] = {};
        if (cloth->fluid_forces != nullptr) cloth->fluid_forces[index] = {};
        if (cloth->smoke_forces != nullptr) cloth->smoke_forces[index] = {};
        cloth->inverse_masses[index] = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[index] : 1.0F / options.vertex_mass;
    }
    std::copy_n(options.triangle_indices.data, cloth->index_count, cloth->indices);
    if (fracture_enabled) {
        std::copy_n(options.triangle_indices.data, cloth->index_count, cloth->source_indices);
        for (std::uint32_t i = 0; i < count; ++i) cloth->vertex_sources[i] = i;
        std::copy(surface_rest.begin(), surface_rest.end(), cloth->surface_positions);
        std::copy(surface_indices.begin(), surface_indices.end(),
                  cloth->surface_triangle_indices);
        std::copy(triangle_bonds.begin(), triangle_bonds.end(),
                  cloth->triangle_bonds);
    }
    std::copy(bonds.begin(), bonds.end(), cloth->bonds);
    std::fill_n(cloth->bond_active, bonds.size(), std::uint8_t{1U});
    std::fill_n(cloth->bond_damage, bonds.size(), std::uint8_t{0U});
    std::copy(offsets.begin(), offsets.end(), cloth->offsets);
    std::copy(neighbors.begin(), neighbors.end(), cloth->neighbors);
    *cloth->count = count;
    if (fracture_enabled) {
        try {
            cloth->source_inverse_masses.assign(cloth->inverse_masses, cloth->inverse_masses + count);
            cloth->source_degrees.resize(count);
            cloth->bond_corners.resize(bonds.size());
            std::unordered_map<std::uint64_t, std::uint32_t> ids, first_corners;
            const auto key = [](std::uint32_t a, std::uint32_t b) {
                return (static_cast<std::uint64_t>(std::min(a,b)) << 32U) | std::max(a,b);
            };
            for (std::uint32_t i = 0; i < bonds.size(); ++i)
                ids[key(bonds[i].first, bonds[i].second)] = i;
            for (std::uint32_t c = 0; c < cloth->index_count; ++c) {
                ++cloth->source_degrees[cloth->source_indices[c]];
                const auto next = c / 3U * 3U + (c + 1U) % 3U;
                const auto other = c / 3U * 3U + (c + 2U) % 3U;
                const auto a = cloth->source_indices[c], b = cloth->source_indices[next];
                const auto edge_key = key(a,b);
                auto [it, fresh] = first_corners.emplace(edge_key,c);
                if (fresh) cloth->bond_corners[ids.at(edge_key)] = {c,next};
                else {
                    const auto previous = it->second;
                    const auto previous_next = previous / 3U * 3U + (previous + 1U) % 3U;
                    const auto opposite = previous / 3U * 3U + (previous + 2U) % 3U;
                    const auto bending = ids.find(key(cloth->source_indices[opposite],
                                                     cloth->source_indices[other]));
                    const bool has_bend = bending != ids.end() && bonds[bending->second].bending;
                    if (has_bend) cloth->bond_corners[bending->second] = {opposite,other};
                    const bool same = cloth->source_indices[previous] == a;
                    cloth->seams.push_back({{previous,previous_next,
                        same ? c : next, same ? next : c}, ids.at(edge_key),
                        has_bend ? bending->second : k_invalid_dense});
                }
            }
        } catch (...) { return failure(StatusCode::out_of_memory, "failed to build cloth seams"); }
        status = rebuild_cloth_topology(*cloth, true);
        if (!status) return status;
    }
    cloth->alive = true;
    output = {slot, cloth->generation};
    impl_->cloths[slot] = std::move(cloth);
    ++impl_->revision;
    return success();
}

Status World::remove_cloth(ClothId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->cloths.size() || !impl_->cloths[id.index] ||
        !impl_->cloths[id.index]->alive ||
        impl_->cloths[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "cloth handle is stale");
    for (std::uint32_t field = 0U;
         field < impl_->options.paint_field_capacity; ++field)
        if (impl_->paint_fields[field].alive &&
            impl_->paint_fields[field].options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a paint field");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.cloth == id)
            return failure(StatusCode::invalid_argument,
                "cloth is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->rope_cloth_couplings)
        if (coupling.alive && coupling.options.cloth == id)
            return failure(StatusCode::invalid_argument,
                "cloth is still referenced by a rope coupling");
    ClothStorage &cloth = *impl_->cloths[id.index];
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->smoke_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a smoke coupling");
    cloth.alive = false;
    cloth.release();
    ++cloth.generation;
    if (cloth.generation == 0U) cloth.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::cloth_view(ClothId id, ClothDeviceView &output) const noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->cloths.size() || !impl_->cloths[id.index] ||
        !impl_->cloths[id.index]->alive ||
        impl_->cloths[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "cloth handle is stale");
    const ClothStorage &cloth = *impl_->cloths[id.index];
    output.positions = {cloth.positions, cloth.vertex_count};
    output.velocities = {cloth.velocities, cloth.vertex_count};
    output.triangle_indices = {cloth.indices, cloth.index_count};
    output.vertex_count = cloth.vertex_count;
    if (cloth.surface_positions != nullptr) {
        output.surface_positions = {cloth.surface_positions, cloth.index_count};
        output.surface_triangle_indices = {
            cloth.surface_triangle_indices, cloth.index_count};
        output.surface_source_indices = {cloth.source_indices, cloth.index_count};
        output.vertex_source_indices = {cloth.vertex_sources, cloth.vertex_count};
        output.inverse_masses = {cloth.inverse_masses, cloth.vertex_count};
    }
    output.bonds = {cloth.bonds, cloth.bond_count};
    output.active_bonds = {cloth.bond_active, cloth.bond_count};
    output.rigid_contact_forces = {
        cloth.rigid_contact_forces, cloth.vertex_count};
    output.soft_body_contact_forces = {cloth.soft_body_forces, cloth.vertex_count};
    if (cloth.rope_forces != nullptr)
        output.rope_contact_forces = {cloth.rope_forces, cloth.vertex_count};
    if (cloth.fluid_forces != nullptr)
        output.fluid_contact_forces = {
            cloth.fluid_forces, cloth.vertex_count};
    return success();
}

Status World::add_soft_body(SoftBodyOptions options, SoftBodyId &output,
                            cudaStream_t stream) noexcept {
    (void)stream;
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.nodes.data == nullptr || options.nodes.size < 4U ||
        options.nodes.size > UINT32_MAX || options.bonds.data == nullptr ||
        options.bonds.size == 0U || options.bonds.size > UINT32_MAX ||
        options.surface_vertices.data == nullptr ||
        options.surface_vertices.size < 3U ||
        options.surface_vertices.size > UINT32_MAX ||
        options.surface_triangle_indices.data == nullptr ||
        options.surface_triangle_indices.size == 0U ||
        options.surface_triangle_indices.size % 3U != 0U ||
        options.surface_triangle_indices.size > UINT32_MAX ||
        options.surface_bindings.data == nullptr ||
        options.surface_bindings.size != options.surface_vertices.size ||
        (options.inverse_masses.size != 0U &&
         (options.inverse_masses.data == nullptr ||
          options.inverse_masses.size != options.nodes.size)) ||
        !finite(options.node_mass) || options.node_mass <= 0.0F ||
        !finite(options.node_radius) || options.node_radius <= 0.0F ||
        !finite(options.stretch_compliance) ||
        options.stretch_compliance < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.spring_damping) || options.spring_damping < 0.0F ||
        options.spring_damping > 1.0F ||
        !finite(options.contact_friction) || options.contact_friction < 0.0F ||
        !finite(options.shape_matching_stiffness) ||
        options.shape_matching_stiffness < 0.0F ||
        options.shape_matching_stiffness > 1.0F ||
        !finite(options.maximum_projection_fraction) ||
        options.maximum_projection_fraction <= 0.0F ||
        options.maximum_projection_fraction > 1.0F ||
        !finite(options.constraint_velocity_response) ||
        options.constraint_velocity_response < 0.0F ||
        options.constraint_velocity_response > 1.0F ||
        !finite(options.maximum_speed) || options.maximum_speed <= 0.0F ||
        options.solver_iterations == 0U || options.solver_iterations > 64U) {
        return failure(StatusCode::invalid_argument,
                       "invalid soft-body geometry or solver options");
    }
    const std::uint32_t node_count =
        static_cast<std::uint32_t>(options.nodes.size);
    float movable_mass = 0.0F;
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        if (!finite(options.nodes.data[node]) ||
            (options.inverse_masses.size != 0U &&
             (!finite(options.inverse_masses.data[node]) ||
              options.inverse_masses.data[node] < 0.0F)))
            return failure(StatusCode::invalid_argument,
                           "invalid soft-body node or inverse mass");
        const float inverse_mass = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
        if (inverse_mass > 0.0F) movable_mass += 1.0F / inverse_mass;
    }
    if (!finite(movable_mass) || movable_mass <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "soft body needs at least one movable node");
    for (std::uint64_t index = 0U;
         index < options.surface_triangle_indices.size; ++index) {
        if (options.surface_triangle_indices.data[index] >=
            options.surface_vertices.size)
            return failure(StatusCode::invalid_argument,
                           "soft-body surface index is out of range");
    }
    for (std::uint64_t vertex = 0U;
         vertex < options.surface_vertices.size; ++vertex) {
        if (!finite(options.surface_vertices.data[vertex]))
            return failure(StatusCode::invalid_argument,
                           "soft-body surface vertex is invalid");
        const SoftBodySurfaceBinding &binding =
            options.surface_bindings.data[vertex];
        float weight_sum = 0.0F;
        for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
            if (binding.nodes[slot] >= node_count ||
                !finite(binding.weights[slot]) || binding.weights[slot] < 0.0F)
                return failure(StatusCode::invalid_argument,
                               "soft-body surface binding is invalid");
            weight_sum += binding.weights[slot];
        }
        if (fabsf(weight_sum - 1.0F) > 1.0e-4F)
            return failure(StatusCode::invalid_argument,
                           "soft-body surface binding weights must sum to one");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
        if (!impl_->soft_bodies[index] || !impl_->soft_bodies[index]->alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense)
        return failure(StatusCode::capacity_exceeded,
                       "soft-body capacity is exhausted");

    std::vector<std::vector<DeformableNeighbor>> adjacency;
    std::vector<std::uint32_t> offsets;
    std::vector<DeformableNeighbor> neighbors;
    std::vector<SoftBodyNeighbor> warp_neighbors;
    std::vector<float> minimum_rest_lengths;
    float minimum_bond_length = FLT_MAX;
    try {
        adjacency.resize(node_count);
        std::unordered_set<std::uint64_t> unique;
        for (std::uint32_t bond_index = 0U;
             bond_index < options.bonds.size; ++bond_index) {
            const SoftBodyBond bond = options.bonds.data[bond_index];
            if (bond.first >= node_count || bond.second >= node_count ||
                bond.first == bond.second || !finite(bond.rest_length) ||
                bond.rest_length <= 1.0e-6F)
                return failure(StatusCode::invalid_argument,
                               "soft-body bond is invalid");
            const std::uint32_t first = std::min(bond.first, bond.second);
            const std::uint32_t second = std::max(bond.first, bond.second);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(first) << 32U) | second;
            if (!unique.insert(key).second)
                return failure(StatusCode::invalid_argument,
                               "soft-body bond is duplicated");
            minimum_bond_length = std::min(
                minimum_bond_length, bond.rest_length);
            adjacency[bond.first].push_back({bond.second, bond.rest_length,
                                             options.stretch_compliance,
                                             bond_index});
            adjacency[bond.second].push_back({bond.first, bond.rest_length,
                                              options.stretch_compliance,
                                              bond_index});
        }
        offsets.reserve(static_cast<std::size_t>(node_count) + 1U);
        offsets.push_back(0U);
        for (const auto &list : adjacency) {
            if (neighbors.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded,
                               "soft-body neighbors exceed uint32 range");
            neighbors.insert(neighbors.end(), list.begin(), list.end());
            offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        }
        if (std::any_of(adjacency.begin(), adjacency.end(),
                        [](const auto &list) { return list.empty(); }))
            return failure(StatusCode::invalid_argument,
                           "every soft-body node needs at least one bond");
        std::size_t maximum_degree=0U;
        for(const auto &list:adjacency)
            maximum_degree=std::max(maximum_degree,list.size());
        const std::size_t ell_count=maximum_degree*node_count;
        // The transpose helps dense near-uniform graphs, but sparse hubs
        // should not pay for mostly empty rows or an unbounded extra buffer.
        if(node_count>=128U && maximum_degree>=64U &&
            maximum_degree<=UINT32_MAX/node_count &&
            ell_count<=2U*neighbors.size() && ell_count<=4'194'304U) {
            warp_neighbors.reserve(neighbors.size());
            for(const DeformableNeighbor neighbor:neighbors)
                warp_neighbors.push_back({neighbor.index,neighbor.rest_length});
            minimum_rest_lengths.reserve(node_count);
            for(const auto &list:adjacency) {
                float shortest=FLT_MAX;
                for(const DeformableNeighbor neighbor:list)
                    shortest=std::min(shortest,neighbor.rest_length);
                minimum_rest_lengths.push_back(shortest);
            }
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to build soft-body adjacency");
    }

    Vec3 shape_rest_center{};
    ShapeMatrix shape_inverse_rest{};
    if (options.shape_matching_stiffness > 0.0F) {
        for (std::uint32_t node = 0U; node < node_count; ++node) {
            const float inverse_mass = options.inverse_masses.size != 0U
                ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
            if (inverse_mass > 0.0F)
                shape_rest_center = add(shape_rest_center,
                    multiply(options.nodes.data[node], 1.0F / inverse_mass));
        }
        shape_rest_center = multiply(shape_rest_center, 1.0F / movable_mass);
        ShapeMatrix rest_covariance{};
        for (std::uint32_t node = 0U; node < node_count; ++node) {
            const float inverse_mass = options.inverse_masses.size != 0U
                ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
            if (inverse_mass <= 0.0F) continue;
            const Vec3 rest = subtract(
                options.nodes.data[node], shape_rest_center);
            const float mass = 1.0F / inverse_mass;
            rest_covariance.columns[0] = add(
                rest_covariance.columns[0], multiply(rest, mass * rest.x));
            rest_covariance.columns[1] = add(
                rest_covariance.columns[1], multiply(rest, mass * rest.y));
            rest_covariance.columns[2] = add(
                rest_covariance.columns[2], multiply(rest, mass * rest.z));
        }
        const Vec3 row0 = cross(rest_covariance.columns[1],
                                rest_covariance.columns[2]);
        const Vec3 row1 = cross(rest_covariance.columns[2],
                                rest_covariance.columns[0]);
        const Vec3 row2 = cross(rest_covariance.columns[0],
                                rest_covariance.columns[1]);
        const float determinant = dot(rest_covariance.columns[0], row0);
        if (!finite(determinant) || fabsf(determinant) <= 1.0e-10F)
            return failure(StatusCode::invalid_argument,
                "shape-matched soft body needs a volumetric rest lattice");
        const float inverse_determinant = 1.0F / determinant;
        shape_inverse_rest.columns[0] = multiply(
            {row0.x, row1.x, row2.x}, inverse_determinant);
        shape_inverse_rest.columns[1] = multiply(
            {row0.y, row1.y, row2.y}, inverse_determinant);
        shape_inverse_rest.columns[2] = multiply(
            {row0.z, row1.z, row2.z}, inverse_determinant);
    }

    std::vector<std::uint32_t> surface_offsets;
    std::vector<SoftSurfaceInfluence> surface_influences;
    try {
        std::vector<std::vector<SoftSurfaceInfluence>> per_node(node_count);
        for (std::uint32_t corner = 0U;
             corner < options.surface_triangle_indices.size; ++corner) {
            const SoftBodySurfaceBinding binding = options.surface_bindings.data[
                options.surface_triangle_indices.data[corner]];
            float denominator = 0.0F;
            for (std::uint32_t a = 0U; a < 4U; ++a) {
                const float inverse = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[binding.nodes[a]]
                    : 1.0F / options.node_mass;
                for (std::uint32_t b = 0U; b < 4U; ++b)
                    if (binding.nodes[a] == binding.nodes[b])
                        denominator += binding.weights[a] * binding.weights[b] * inverse;
            }
            if (denominator <= 0.0F) continue;
            for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
                const std::uint32_t node = binding.nodes[slot];
                bool duplicate = false;
                for (std::uint32_t earlier = 0U; earlier < slot; ++earlier)
                    duplicate |= binding.nodes[earlier] == node;
                if (duplicate) continue;
                float weight = 0.0F;
                for (std::uint32_t other = slot; other < 4U; ++other)
                    if (binding.nodes[other] == node) weight += binding.weights[other];
                const float inverse = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
                if (weight * inverse > 0.0F)
                    per_node[node].push_back({corner, weight * inverse / denominator});
            }
        }
        surface_offsets.push_back(0U);
        for (const auto &list : per_node) {
            if (surface_influences.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded,
                               "soft-body surface influences exceed uint32 range");
            surface_influences.insert(surface_influences.end(), list.begin(), list.end());
            surface_offsets.push_back(static_cast<std::uint32_t>(surface_influences.size()));
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to build soft-body surface contact bindings");
    }

    std::unique_ptr<SoftBodyStorage> body;
    try {
        body = std::make_unique<SoftBodyStorage>();
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate soft-body storage");
    }
    body->generation = impl_->soft_bodies[slot]
        ? impl_->soft_bodies[slot]->generation : 1U;
    body->node_count = node_count;
    body->bond_count = static_cast<std::uint32_t>(options.bonds.size);
    body->neighbor_count = static_cast<std::uint32_t>(neighbors.size());
    body->surface_vertex_count =
        static_cast<std::uint32_t>(options.surface_vertices.size);
    body->surface_index_count =
        static_cast<std::uint32_t>(options.surface_triangle_indices.size);
    body->node_radius = options.node_radius;
    body->stretch_compliance = options.stretch_compliance;
    body->velocity_damping = options.velocity_damping;
    body->spring_damping = options.spring_damping;
    body->contact_friction = options.contact_friction;
    body->shape_matching_stiffness = options.shape_matching_stiffness;
    body->shape_maximum_projection = options.maximum_projection_fraction *
                                     minimum_bond_length;
    body->maximum_projection_fraction = options.maximum_projection_fraction;
    body->constraint_velocity_response = options.constraint_velocity_response;
    body->maximum_speed = options.maximum_speed;
    body->movable_mass = movable_mass;
    body->shape_rest_center = shape_rest_center;
    body->shape_inverse_rest = shape_inverse_rest;
    body->solver_iterations = options.solver_iterations;
#define PM_ALLOC_SOFT(member, count)                                            \
    status = allocate_managed(body->member, count);                             \
    if (!status) return status
    PM_ALLOC_SOFT(positions, body->node_count);
    PM_ALLOC_SOFT(rest_positions, body->node_count);
    PM_ALLOC_SOFT(scratch, body->node_count);
    PM_ALLOC_SOFT(previous, body->node_count);
    PM_ALLOC_SOFT(velocities, body->node_count);
    PM_ALLOC_SOFT(velocity_scratch, body->node_count);
    PM_ALLOC_SOFT(inverse_masses, body->node_count);
    PM_ALLOC_SOFT(bonds, body->bond_count);
    PM_ALLOC_SOFT(bond_active, body->bond_count);
    PM_ALLOC_SOFT(offsets, offsets.size());
    PM_ALLOC_SOFT(neighbors, neighbors.size());
    if(!warp_neighbors.empty())PM_ALLOC_SOFT(warp_neighbors,warp_neighbors.size());
    if(!minimum_rest_lengths.empty())
        PM_ALLOC_SOFT(minimum_rest_lengths,minimum_rest_lengths.size());
    PM_ALLOC_SOFT(surface_rest_positions, body->surface_vertex_count);
    PM_ALLOC_SOFT(surface_positions, body->surface_vertex_count);
    PM_ALLOC_SOFT(surface_indices, body->surface_index_count);
    PM_ALLOC_SOFT(surface_bindings, body->surface_vertex_count);
    PM_ALLOC_SOFT(surface_corner_corrections, body->surface_index_count);
    PM_ALLOC_SOFT(surface_node_offsets, surface_offsets.size());
    PM_ALLOC_SOFT(surface_node_influences, surface_influences.size());
    PM_ALLOC_SOFT(body_impulses, body->node_count);
    PM_ALLOC_SOFT(body_position_corrections, body->node_count);
    PM_ALLOC_SOFT(cloth_forces, body->node_count);
    PM_ALLOC_SOFT(rope_forces, body->node_count);
    PM_ALLOC_SOFT(fluid_forces, body->node_count);
    PM_ALLOC_SOFT(rigid_contact_forces, body->node_count);
    PM_ALLOC_SOFT(contact_normals, body->node_count);
    PM_ALLOC_SOFT(contact_arms, body->node_count);
    PM_ALLOC_SOFT(contact_momentum_delta, body->node_count);
    PM_ALLOC_SOFT(contact_friction_delta, body->node_count);
    PM_ALLOC_SOFT(contact_normal_delta, body->node_count);
    PM_ALLOC_SOFT(predicted_momentum, 1U);
    PM_ALLOC_SOFT(shape_orientation, 1U);
    PM_ALLOC_SOFT(dynamic_contact_flag, 1U);
    PM_ALLOC_SOFT(contact_count, 1U);
    PM_ALLOC_SOFT(count, 1U);
#undef PM_ALLOC_SOFT
    for (std::uint32_t node = 0U; node < body->node_count; ++node) {
        body->positions[node] = options.nodes.data[node];
        body->rest_positions[node] = options.nodes.data[node];
        body->scratch[node] = options.nodes.data[node];
        body->previous[node] = options.nodes.data[node];
        body->velocities[node] = {};
        body->velocity_scratch[node] = {};
        body->inverse_masses[node] = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
        body->rigid_contact_forces[node] = {};
        body->cloth_forces[node] = {};
        body->fluid_forces[node] = {};
        body->rope_forces[node] = {};
        body->contact_normals[node] = {};
        body->contact_arms[node] = {};
        body->contact_momentum_delta[node] = {};
        body->contact_friction_delta[node] = {};
        body->contact_normal_delta[node] = 0.0F;
    }
    std::copy_n(options.bonds.data, body->bond_count, body->bonds);
    std::fill_n(body->bond_active, body->bond_count, std::uint8_t{1U});
    std::copy(offsets.begin(), offsets.end(), body->offsets);
    std::copy(neighbors.begin(), neighbors.end(), body->neighbors);
    if(!warp_neighbors.empty())
        std::copy(warp_neighbors.begin(),warp_neighbors.end(),body->warp_neighbors);
    if(!minimum_rest_lengths.empty())
        std::copy(minimum_rest_lengths.begin(),minimum_rest_lengths.end(),
            body->minimum_rest_lengths);
    std::copy_n(options.surface_vertices.data, body->surface_vertex_count,
                body->surface_rest_positions);
    std::copy_n(options.surface_vertices.data, body->surface_vertex_count,
                body->surface_positions);
    std::copy_n(options.surface_triangle_indices.data,
                body->surface_index_count, body->surface_indices);
    std::copy_n(options.surface_bindings.data, body->surface_vertex_count,
                body->surface_bindings);
    std::copy(surface_offsets.begin(), surface_offsets.end(), body->surface_node_offsets);
    std::copy(surface_influences.begin(), surface_influences.end(), body->surface_node_influences);
    *body->count = body->node_count;
    *body->contact_count = 0U;
    *body->predicted_momentum = {};
    *body->shape_orientation = {0.0F, 0.0F, 0.0F, 1.0F};
    *body->dynamic_contact_flag = 0U;
    body->alive = true;
    output = {slot, body->generation};
    impl_->soft_bodies[slot] = std::move(body);
    ++impl_->revision;
    return success();
}

Status World::add_rope(RopeOptions options, RopeId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument,"world is not initialized");
    Status status=impl_->require_idle();
    if(!status)return status;
    if(!finite(options.radius)||options.radius<=0 || !finite(options.node_spacing)||
       options.node_spacing<=0 || options.node_spacing>2*options.radius ||
       !finite(options.mass)||options.mass<=0 || !finite(options.stretch_compliance)||options.stretch_compliance<0 ||
       !finite(options.velocity_damping)||options.velocity_damping<0 ||
       !finite(options.maximum_substep_timestep)||options.maximum_substep_timestep<=0 ||
       !finite(options.friction)||options.friction<0 ||
       !finite(options.maximum_speed)||options.maximum_speed<=0 || options.solver_iterations<1 || options.solver_iterations>128)
        return failure(StatusCode::invalid_argument,"invalid rope material or resolution");
    for(const auto anchor:{options.first,options.last}) {
        if(!finite(anchor.local_anchor))return failure(StatusCode::invalid_argument,"invalid rope anchor");
        if(anchor.enabled){unsigned dense;if(!(status=impl_->validate_handle(anchor.body,dense)))return status;}
    }
    if (options.first.enabled && options.last.enabled && options.first.body == options.last.body)
        return failure(StatusCode::invalid_argument,"rope endpoints must attach to distinct bodies");
    std::uint32_t slot=0;
    while(slot<impl_->ropes.size() && impl_->ropes[slot] && impl_->ropes[slot]->alive)++slot;
    if(slot==impl_->ropes.size())return failure(StatusCode::capacity_exceeded,"rope capacity exhausted");
    std::vector<Vec3> nodes;
    if(!(status=sample_rope_centerline(options.centerline,options.node_spacing,nodes)))return status;
    int attached[2]{-1, -1};
    // Rest curve endpoints define anchors, not an initial teleport/impulse.
    for(unsigned end=0;end<2;++end) {
        const auto anchor=end?options.last:options.first;
        if(!anchor.enabled)continue;
        unsigned dense;if(!(status=impl_->validate_handle(anchor.body,dense)))return status;
        attached[end] = static_cast<int>(dense);
        const Vec3 target=transform_point(impl_->states[impl_->current_state][dense],anchor.local_anchor);
        if(vector_length(subtract(target,end?nodes.back():nodes.front()))>1e-3F)
            return failure(StatusCode::invalid_argument,"rope endpoint must match its body-local attachment");
    }
    const unsigned first_skip=rope_attachment_contact_skip(nodes,true,attached[0],
        impl_->parameters,impl_->states[impl_->current_state],impl_->meshes);
    const unsigned last_skip=rope_attachment_contact_skip(nodes,false,attached[1],
        impl_->parameters,impl_->states[impl_->current_state],impl_->meshes);
    try {
        if (rope_rest_crosses_collider(nodes, attached[0], attached[1],
                first_skip,last_skip,impl_->parameters,
                impl_->states[impl_->current_state], impl_->meshes, impl_->rigid_body_count))
            return failure(StatusCode::invalid_argument,"rope rest centerline crosses a rigid collider");
    } catch (...) {
        return failure(StatusCode::out_of_memory,"rope rest collision validation allocation failed");
    }
    std::unique_ptr<RopeStorage> owner(new(std::nothrow)RopeStorage());
    if(!owner)return failure(StatusCode::out_of_memory,"rope allocation failed");
    owner->generation=impl_->ropes[slot]?impl_->ropes[slot]->generation:1;
    auto &r=owner->data;
    r.options=options;r.options.centerline={};r.count=static_cast<unsigned>(nodes.size());
    r.first_attachment_contact_skip=first_skip;
    r.last_attachment_contact_skip=last_skip;
    r.body_capacity=impl_->options.rigid_body_capacity;
    for(Vec3 **p:{&r.positions,&r.previous,&r.velocities,&r.constraint_forces,&r.contact_forces,&r.fluid_contact_forces,&r.soft_body_contact_forces,&r.directions,&r.scratch,&r.normals,&r.normals2})
        if(!(status=allocate_managed(*p,r.count)))return status;
    for(Vec3 **p:{&r.soft_anchor_positions,&r.soft_anchor_velocities,&r.soft_anchor_impulses})
        if(!(status=allocate_managed(*p,2U)))return status;
    if(!(status=allocate_managed(r.soft_anchor_inverse_masses,2U)))return status;
    r.soft_anchor_inverse_masses[0]=r.soft_anchor_inverse_masses[1]=0.0F;
    for(float **p:{&r.rest,&r.lambda})
        if(!(status=allocate_managed(*p,r.count)))return status;
    if(!(status=allocate_managed(r.body_translation,impl_->options.rigid_body_capacity)) ||
       !(status=allocate_managed(r.body_rotation,impl_->options.rigid_body_capacity)))return status;
    if(!(status=allocate_managed(r.solid_hint,r.count*r.body_capacity)))return status;
    std::fill_n(r.solid_hint,r.count*r.body_capacity,~0U);
    for(unsigned i=0;i<r.count;++i){
        r.positions[i]=r.previous[i]=nodes[i];
        r.velocities[i]=r.constraint_forces[i]=r.contact_forces[i]=r.fluid_contact_forces[i]=r.soft_body_contact_forces[i]={};
        if(i+1<r.count){r.rest[i]=vector_length(subtract(nodes[i+1],nodes[i]));
            if(!finite(r.rest[i]) || r.rest[i]<1e-6F)return failure(StatusCode::invalid_argument,"rope contains an invalid segment");}
    }
    owner->alive=true;output={slot,owner->generation};impl_->ropes[slot]=std::move(owner);++impl_->revision;
    return success();
}

Status World::remove_rope(RopeId id) noexcept {
    if(!impl_)return failure(StatusCode::invalid_argument,"world is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(id.index>=impl_->ropes.size() || !impl_->ropes[id.index] || !impl_->ropes[id.index]->alive || impl_->ropes[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope handle is stale");
    for (const auto &coupling : impl_->fluid_rope_couplings)
        if (coupling && coupling->alive && coupling->options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->smoke_rope_couplings)
        if (coupling && coupling->alive && coupling->options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a smoke coupling");
    for (const auto &coupling : impl_->rope_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->rope_cloth_couplings)
        if (coupling.alive && coupling.options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a cloth coupling");
    auto &rope=*impl_->ropes[id.index];rope.alive=false;rope.release();++rope.generation;
    if(rope.generation==0)rope.generation=1;
    ++impl_->revision;return success();
}

Status World::rope_view(RopeId id, RopeDeviceView &output) const noexcept {
    output={};
    if(!impl_)return failure(StatusCode::invalid_argument,"world is not initialized");
    auto status=impl_->require_current_device();if(!status)return status;
    if(impl_->frame && !impl_->frame->acknowledged)return failure(StatusCode::busy,"rope view requires completed frame");
    if(id.index>=impl_->ropes.size() || !impl_->ropes[id.index] || !impl_->ropes[id.index]->alive || impl_->ropes[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope handle is stale");
    const auto &r=impl_->ropes[id.index]->data;
    output={{r.positions,r.count},{r.velocities,r.count},{r.constraint_forces,r.count},
        {r.contact_forces,r.count},{r.fluid_contact_forces,r.count},
        {r.soft_body_contact_forces,r.count},
        {r.rest,r.count-1},r.options.radius};
    return success();
}

static bool valid_fluid_rope_options(FluidRopeCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F && options.friction <= 1.0F &&
        finite(options.maximum_rope_acceleration) && options.maximum_rope_acceleration > 0.0F;
}

Status World::add_fluid_rope_coupling(
    FluidRopeCouplingOptions options, FluidRopeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.rope.index >= impl_->ropes.size() ||
        !impl_->ropes[options.rope.index] || !impl_->ropes[options.rope.index]->alive ||
        impl_->ropes[options.rope.index]->generation != options.rope.generation)
        return failure(StatusCode::invalid_handle, "rope handle is stale");
    if (!valid_fluid_rope_options(options))
        return failure(StatusCode::invalid_argument, "invalid fluid rope options");
    for (const auto &item : impl_->fluid_rope_couplings)
        if (item && item->alive && item->options.fluid == options.fluid &&
            item->options.rope == options.rope)
            return failure(StatusCode::invalid_argument, "fluid and rope are already coupled");
    std::uint32_t slot = 0;
    while (slot < impl_->fluid_rope_couplings.size() &&
           impl_->fluid_rope_couplings[slot] && impl_->fluid_rope_couplings[slot]->alive)
        ++slot;
    if (slot == impl_->fluid_rope_couplings.size())
        return failure(StatusCode::capacity_exceeded, "fluid rope coupling capacity exhausted");
    std::unique_ptr<FluidRopeCouplingStorage> owner(
        new (std::nothrow) FluidRopeCouplingStorage());
    if (!owner) return failure(StatusCode::out_of_memory, "fluid rope coupling allocation failed");
    if (!(status = allocate_managed(owner->node_impulses,
            impl_->ropes[options.rope.index]->data.count)) ||
        !(status = allocate_managed(owner->contact_count, 1U)) ||
        !(status = allocate_managed(owner->maximum_penetration, 1U))) return status;
    *owner->contact_count = 0U;
    *owner->maximum_penetration = 0.0F;
    owner->options = options;
    owner->generation = impl_->fluid_rope_couplings[slot]
        ? impl_->fluid_rope_couplings[slot]->generation : 1U;
    owner->alive = true;
    output = {slot, owner->generation};
    impl_->fluid_rope_couplings[slot] = std::move(owner);
    ++impl_->revision;
    return success();
}

Status World::update_fluid_rope_coupling(
    FluidRopeCouplingId id, FluidRopeCouplingOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_rope_couplings.size() ||
        !impl_->fluid_rope_couplings[id.index] ||
        !impl_->fluid_rope_couplings[id.index]->alive ||
        impl_->fluid_rope_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid rope coupling is stale");
    auto &item = *impl_->fluid_rope_couplings[id.index];
    if (!valid_fluid_rope_options(options) ||
        !(options.fluid == item.options.fluid) || !(options.rope == item.options.rope))
        return failure(StatusCode::invalid_argument, "invalid options or changed fluid rope endpoints");
    item.options = options;
    ++impl_->revision;
    return success();
}

Status World::remove_fluid_rope_coupling(FluidRopeCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_rope_couplings.size() ||
        !impl_->fluid_rope_couplings[id.index] ||
        !impl_->fluid_rope_couplings[id.index]->alive ||
        impl_->fluid_rope_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid rope coupling is stale");
    auto &item = *impl_->fluid_rope_couplings[id.index];
    item.release();
    item.alive = false;
    if (++item.generation == 0U) item.generation = 1U;
    ++impl_->revision;
    return success();
}

static bool valid_rope_soft_options(RopeSoftBodyCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        finite(options.maximum_soft_body_acceleration) &&
        options.maximum_soft_body_acceleration > 0.0F &&
        finite(options.anchor_support_radius_scale) &&
        options.anchor_support_radius_scale >= 0.0F &&
        finite(options.anchor_contact_support_radius_scale) &&
        options.anchor_contact_support_radius_scale >= 0.0F &&
        options.anchor_contact_support_radius_scale <=
            options.anchor_support_radius_scale;
}

Status World::add_rope_soft_body_coupling(RopeSoftBodyCouplingOptions options,
    RopeSoftBodyCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument,"world is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(options.rope.index>=impl_->ropes.size() || !impl_->ropes[options.rope.index] ||
       !impl_->ropes[options.rope.index]->alive ||
       impl_->ropes[options.rope.index]->generation!=options.rope.generation ||
       options.soft_body.index>=impl_->soft_bodies.size() ||
       !impl_->soft_bodies[options.soft_body.index] ||
       !impl_->soft_bodies[options.soft_body.index]->alive ||
       impl_->soft_bodies[options.soft_body.index]->generation!=options.soft_body.generation)
        return failure(StatusCode::invalid_handle,"rope or soft body handle is stale");
    if(!valid_rope_soft_options(options))
        return failure(StatusCode::invalid_argument,"invalid rope soft-body options");
    auto &rope=impl_->ropes[options.rope.index]->data;
    auto &body=*impl_->soft_bodies[options.soft_body.index];
    if((options.attach_first && rope.options.first.enabled) ||
       (options.attach_last && rope.options.last.enabled))
        return failure(StatusCode::invalid_argument,"rope endpoint already has an attachment");
    unsigned existing_targets=0;
    for(const auto &item:impl_->rope_soft_body_couplings)
        if(item && item->alive && item->options.rope==options.rope) {
            ++existing_targets;
            if(item->options.soft_body==options.soft_body)
                return failure(StatusCode::invalid_argument,"rope and soft body are already coupled");
            if((options.attach_first && item->options.attach_first) ||
               (options.attach_last && item->options.attach_last))
                return failure(StatusCode::invalid_argument,"rope endpoint already has an attachment");
        }
    for(const auto &item:impl_->rope_cloth_couplings)
        if(item.alive && item.options.rope==options.rope &&
           ((options.attach_first && item.options.first_vertex!=UINT32_MAX) ||
            (options.attach_last && item.options.last_vertex!=UINT32_MAX)))
            return failure(StatusCode::invalid_argument,
                "rope endpoint already has a cloth attachment");
    if(existing_targets>=2U)
        return failure(StatusCode::capacity_exceeded,
            "a rope supports at most two soft-body contact targets");
    unsigned slot=0;
    while(slot<impl_->rope_soft_body_couplings.size() &&
          impl_->rope_soft_body_couplings[slot] &&
          impl_->rope_soft_body_couplings[slot]->alive)++slot;
    if(slot==impl_->rope_soft_body_couplings.size())
        return failure(StatusCode::capacity_exceeded,"rope soft-body coupling capacity exhausted");
    std::unique_ptr<RopeSoftBodyCouplingStorage> owner(
        new(std::nothrow)RopeSoftBodyCouplingStorage());
    if(!owner)return failure(StatusCode::out_of_memory,"rope soft-body coupling allocation failed");
    std::vector<BvhNode> tree;
    std::vector<std::uint32_t> order,parents;
    double signed_volume=0.0;
    const Vec3 volume_origin=body.surface_rest_positions[0];
    for(unsigned t=0;t<body.surface_index_count/3U;++t) {
        const auto *idx=body.surface_indices+3U*t;
        signed_volume+=dot(subtract(body.surface_rest_positions[idx[0]],volume_origin),
            cross(subtract(body.surface_rest_positions[idx[1]],volume_origin),
                  subtract(body.surface_rest_positions[idx[2]],volume_origin)))/6.0;
    }
    if(std::abs(signed_volume)<1e-10)
        return failure(StatusCode::invalid_argument,"rope soft-body skin has zero volume");
    owner->orientation=signed_volume>=0.0?1.0F:-1.0F;
    try {
        order.resize(body.surface_index_count/3U);
        std::iota(order.begin(),order.end(),0U);
        tree.reserve(order.size()*2U);
        parents.reserve(order.size()*2U);
        const auto centroid=[&](unsigned triangle) {
            const auto *idx=body.surface_indices+triangle*3U;
            return multiply(add(add(body.surface_rest_positions[idx[0]],
                body.surface_rest_positions[idx[1]]),
                body.surface_rest_positions[idx[2]]),1.0F/3.0F);
        };
        std::function<unsigned(unsigned,unsigned,unsigned)> build=
            [&](unsigned begin,unsigned end,unsigned parent) {
                const unsigned index=static_cast<unsigned>(tree.size());
                BvhNode node{};
                node.minimum={FLT_MAX,FLT_MAX,FLT_MAX};
                node.maximum={-FLT_MAX,-FLT_MAX,-FLT_MAX};
                for(unsigned item=begin;item<end;++item) {
                    const auto point=centroid(order[item]);
                    node.minimum=component_min(node.minimum,point);
                    node.maximum=component_max(node.maximum,point);
                }
                tree.push_back(node);parents.push_back(parent);
                if(end-begin<=4U) {
                    tree[index].first_triangle=begin;
                    tree[index].triangle_count=end-begin;
                } else {
                    const auto extent=subtract(node.maximum,node.minimum);
                    const unsigned axis=extent.x>=extent.y && extent.x>=extent.z?0U:
                        extent.y>=extent.z?1U:2U;
                    const unsigned middle=begin+(end-begin)/2U;
                    std::nth_element(order.begin()+begin,order.begin()+middle,
                        order.begin()+end,[&](unsigned left,unsigned right) {
                            const auto a=centroid(left),b=centroid(right);
                            const float x=axis==0?a.x:axis==1?a.y:a.z;
                            const float y=axis==0?b.x:axis==1?b.y:b.z;
                            return x<y || (x==y && left<right);
                        });
                    const unsigned left=build(begin,middle,index);
                    const unsigned right=build(middle,end,index);
                    tree[index].left=left;tree[index].right=right;
                }
                return index;
            };
        build(0U,static_cast<unsigned>(order.size()),k_invalid_dense);
    } catch (...) {
        return failure(StatusCode::out_of_memory,"rope soft-body BVH build failed");
    }
    owner->tree_count=static_cast<unsigned>(tree.size());
    if(!(status=allocate_managed(owner->tree,tree.size())) ||
       !(status=allocate_managed(owner->order,order.size())) ||
       !(status=allocate_managed(owner->parents,parents.size())) ||
       !(status=allocate_managed(owner->ready,tree.size())))return status;
    std::copy(tree.begin(),tree.end(),owner->tree);
    std::copy(order.begin(),order.end(),owner->order);
    std::copy(parents.begin(),parents.end(),owner->parents);
    for(unsigned end=0;end<2;++end) {
        if(!(end?options.attach_last:options.attach_first))continue;
        const Vec3 point=rope.positions[end?rope.count-1U:0U];
        float best=FLT_MAX;
        for(unsigned t=0;t<body.surface_index_count/3U;++t) {
            const auto *idx=body.surface_indices+3U*t;
            const Vec3 a=body.surface_rest_positions[idx[0]],
                       b=body.surface_rest_positions[idx[1]],
                       c=body.surface_rest_positions[idx[2]];
            const Vec3 ab=subtract(b,a),ac=subtract(c,a);
            const float d00=dot(ab,ab),d01=dot(ab,ac),d11=dot(ac,ac);
            const float denominator=d00*d11-d01*d01;
            if(denominator<=1e-14F)continue;
            const Vec3 nearest=closest_on_triangle(point,a,b,c);
            const float distance=length_squared(subtract(point,nearest));
            if(distance>=best)continue;
            const Vec3 ap=subtract(nearest,a);
            const float d20=dot(ap,ab),d21=dot(ap,ac);
            const float v=(d11*d20-d01*d21)/denominator;
            const float w=(d00*d21-d01*d20)/denominator;
            owner->anchor_triangle[end]=t;
            owner->anchor_weights[end]={1.0F-v-w,v,w};
            owner->anchor_offset[end]=subtract(point,nearest);
            best=distance;
        }
        if(owner->anchor_triangle[end]==~0U)
            return failure(StatusCode::invalid_argument,"soft body has no usable anchor triangle");
    }
    if(!(status=allocate_managed(owner->node_impulses,body.node_count)) ||
       !(status=allocate_managed(owner->previous_surface,body.surface_vertex_count)) ||
       !(status=allocate_managed(owner->bounds,2U)) ||
       !(status=allocate_managed(owner->contact_count,1U)) ||
       !(status=allocate_managed(owner->maximum_penetration,1U)))return status;
    std::fill_n(owner->node_impulses,body.node_count,Vec3{});
    std::copy_n(body.surface_positions,body.surface_vertex_count,
        owner->previous_surface);
    *owner->contact_count=0U;*owner->maximum_penetration=0.0F;
    owner->options=options;
    owner->generation=impl_->rope_soft_body_couplings[slot]
        ? impl_->rope_soft_body_couplings[slot]->generation:1U;
    owner->alive=true;
    if(options.enabled) {
        rope.soft_first|=options.attach_first;
        rope.soft_last|=options.attach_last;
    }
    output={slot,owner->generation};
    impl_->rope_soft_body_couplings[slot]=std::move(owner);
    ++impl_->revision;return success();
}

Status World::update_rope_soft_body_coupling(RopeSoftBodyCouplingId id,
    RopeSoftBodyCouplingOptions options) noexcept {
    if(!impl_)return failure(StatusCode::invalid_argument,"world is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(id.index>=impl_->rope_soft_body_couplings.size() ||
       !impl_->rope_soft_body_couplings[id.index] ||
       !impl_->rope_soft_body_couplings[id.index]->alive ||
       impl_->rope_soft_body_couplings[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope soft-body coupling is stale");
    auto &item=*impl_->rope_soft_body_couplings[id.index];
    if(!valid_rope_soft_options(options) || !(options.rope==item.options.rope) ||
       !(options.soft_body==item.options.soft_body) ||
       options.attach_first!=item.options.attach_first ||
       options.attach_last!=item.options.attach_last)
        return failure(StatusCode::invalid_argument,"cannot change rope soft-body endpoints");
    item.options=options;
    auto &rope=impl_->ropes[options.rope.index]->data;
    rope.soft_first=rope.soft_last=false;
    for(const auto &coupling:impl_->rope_soft_body_couplings)
        if(coupling && coupling->alive && coupling->options.enabled &&
           coupling->options.rope==options.rope) {
            rope.soft_first|=coupling->options.attach_first;
            rope.soft_last|=coupling->options.attach_last;
        }
    for(const auto &coupling:impl_->rope_cloth_couplings)
        if(coupling.alive && coupling.options.enabled &&
           coupling.options.rope==options.rope) {
            rope.soft_first|=coupling.options.first_vertex!=UINT32_MAX;
            rope.soft_last|=coupling.options.last_vertex!=UINT32_MAX;
        }
    ++impl_->revision;return success();
}

Status World::remove_rope_soft_body_coupling(RopeSoftBodyCouplingId id) noexcept {
    if(!impl_)return failure(StatusCode::invalid_argument,"world is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(id.index>=impl_->rope_soft_body_couplings.size() ||
       !impl_->rope_soft_body_couplings[id.index] ||
       !impl_->rope_soft_body_couplings[id.index]->alive ||
       impl_->rope_soft_body_couplings[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope soft-body coupling is stale");
    auto &item=*impl_->rope_soft_body_couplings[id.index];
    const auto rope_id=item.options.rope;
    item.release();item.alive=false;
    if(++item.generation==0U)item.generation=1U;
    auto &rope=impl_->ropes[rope_id.index]->data;
    rope.soft_first=rope.soft_last=false;
    for(const auto &coupling:impl_->rope_soft_body_couplings)
        if(coupling && coupling->alive && coupling->options.enabled &&
           coupling->options.rope==rope_id) {
            rope.soft_first|=coupling->options.attach_first;
            rope.soft_last|=coupling->options.attach_last;
        }
    for(const auto &coupling:impl_->rope_cloth_couplings)
        if(coupling.alive && coupling.options.enabled &&
           coupling.options.rope==rope_id) {
            rope.soft_first|=coupling.options.first_vertex!=UINT32_MAX;
            rope.soft_last|=coupling.options.last_vertex!=UINT32_MAX;
        }
    ++impl_->revision;return success();
}

Status World::add_rope_cloth_coupling(RopeClothCouplingOptions options,
    RopeClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    auto status = impl_->require_idle(); if (!status) return status;
    if (options.rope.index >= impl_->ropes.size() ||
        !impl_->ropes[options.rope.index] ||
        !impl_->ropes[options.rope.index]->alive ||
        impl_->ropes[options.rope.index]->generation != options.rope.generation ||
        options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation != options.cloth.generation)
        return failure(StatusCode::invalid_handle, "rope or cloth handle is stale");
    if (!finite(options.anchor_effective_mass) ||
        options.anchor_effective_mass <= 0.0F ||
        !finite(options.maximum_cloth_acceleration) ||
        options.maximum_cloth_acceleration <= 0.0F ||
        (options.first_vertex == UINT32_MAX && options.last_vertex == UINT32_MAX))
        return failure(StatusCode::invalid_argument, "invalid rope cloth options");
    auto &rope = impl_->ropes[options.rope.index]->data;
    auto &cloth = *impl_->cloths[options.cloth.index];
    const std::uint32_t vertices[2]{options.first_vertex, options.last_vertex};
    for (unsigned end = 0; end < 2; ++end) {
        if (vertices[end] == UINT32_MAX) continue;
        if (vertices[end] >= cloth.vertex_count ||
            (end ? rope.options.last.enabled : rope.options.first.enabled))
            return failure(StatusCode::invalid_argument,
                "rope cloth vertex is invalid or endpoint has a rigid attachment");
        const Vec3 delta = subtract(rope.positions[end ? rope.count - 1U : 0U],
                                    cloth.positions[vertices[end]]);
        if (length_squared(delta) > 1.0e-6F)
            return failure(StatusCode::invalid_argument,
                "rope endpoint must coincide with its cloth vertex");
    }
    for (const auto &item : impl_->rope_soft_body_couplings)
        if (item && item->alive && item->options.rope == options.rope &&
            ((options.first_vertex != UINT32_MAX && item->options.attach_first) ||
             (options.last_vertex != UINT32_MAX && item->options.attach_last)))
            return failure(StatusCode::invalid_argument,
                "rope endpoint already has a soft-body attachment");
    for (const auto &item : impl_->rope_cloth_couplings)
        if (item.alive && item.options.rope == options.rope &&
            (item.options.cloth == options.cloth ||
             (options.first_vertex != UINT32_MAX && item.options.first_vertex != UINT32_MAX) ||
             (options.last_vertex != UINT32_MAX && item.options.last_vertex != UINT32_MAX)))
            return failure(StatusCode::invalid_argument,
                "rope endpoint already has a cloth attachment");
    unsigned slot = 0U;
    while (slot < impl_->rope_cloth_couplings.size() &&
           impl_->rope_cloth_couplings[slot].alive) ++slot;
    if (slot == impl_->rope_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
            "rope cloth coupling capacity exhausted");
    auto &item = impl_->rope_cloth_couplings[slot];
    item.options = options;
    item.alive = true;
    if (options.enabled) {
        rope.soft_first |= options.first_vertex != UINT32_MAX;
        rope.soft_last |= options.last_vertex != UINT32_MAX;
    }
    output = {slot, item.generation};
    ++impl_->revision;
    return success();
}

Status World::update_rope_cloth_coupling(RopeClothCouplingId id,
    RopeClothCouplingOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    auto status = impl_->require_idle(); if (!status) return status;
    if (id.index >= impl_->rope_cloth_couplings.size() ||
        !impl_->rope_cloth_couplings[id.index].alive ||
        impl_->rope_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "rope cloth coupling is stale");
    auto &item = impl_->rope_cloth_couplings[id.index];
    if (!(item.options.rope == options.rope) ||
        !(item.options.cloth == options.cloth) ||
        item.options.first_vertex != options.first_vertex ||
        item.options.last_vertex != options.last_vertex ||
        !finite(options.anchor_effective_mass) ||
        options.anchor_effective_mass <= 0.0F ||
        !finite(options.maximum_cloth_acceleration) ||
        options.maximum_cloth_acceleration <= 0.0F)
        return failure(StatusCode::invalid_argument,
            "cannot change rope cloth endpoints or use invalid acceleration");
    item.options = options;
    auto &rope = impl_->ropes[options.rope.index]->data;
    rope.soft_first = rope.soft_last = false;
    for (const auto &soft : impl_->rope_soft_body_couplings)
        if (soft && soft->alive && soft->options.enabled &&
            soft->options.rope == options.rope) {
            rope.soft_first |= soft->options.attach_first;
            rope.soft_last |= soft->options.attach_last;
        }
    for (const auto &cloth : impl_->rope_cloth_couplings)
        if (cloth.alive && cloth.options.enabled && cloth.options.rope == options.rope) {
            rope.soft_first |= cloth.options.first_vertex != UINT32_MAX;
            rope.soft_last |= cloth.options.last_vertex != UINT32_MAX;
        }
    ++impl_->revision;
    return success();
}

Status World::remove_rope_cloth_coupling(RopeClothCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    auto status = impl_->require_idle(); if (!status) return status;
    if (id.index >= impl_->rope_cloth_couplings.size() ||
        !impl_->rope_cloth_couplings[id.index].alive ||
        impl_->rope_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "rope cloth coupling is stale");
    auto &item = impl_->rope_cloth_couplings[id.index];
    const RopeId rope_id = item.options.rope;
    item.alive = false;
    if (++item.generation == 0U) item.generation = 1U;
    auto &rope = impl_->ropes[rope_id.index]->data;
    rope.soft_first = rope.soft_last = false;
    for (const auto &soft : impl_->rope_soft_body_couplings)
        if (soft && soft->alive && soft->options.enabled &&
            soft->options.rope == rope_id) {
            rope.soft_first |= soft->options.attach_first;
            rope.soft_last |= soft->options.attach_last;
        }
    for (const auto &cloth : impl_->rope_cloth_couplings)
        if (cloth.alive && cloth.options.enabled && cloth.options.rope == rope_id) {
            rope.soft_first |= cloth.options.first_vertex != UINT32_MAX;
            rope.soft_last |= cloth.options.last_vertex != UINT32_MAX;
        }
    ++impl_->revision;
    return success();
}

Status World::remove_soft_body(SoftBodyId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_bodies.size() || !impl_->soft_bodies[id.index] ||
        !impl_->soft_bodies[id.index]->alive ||
        impl_->soft_bodies[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body handle is stale");
    SoftBodyStorage &body = *impl_->soft_bodies[id.index];
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->smoke_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a smoke coupling");
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a cloth coupling");
    for (const auto &coupling : impl_->rope_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a rope coupling");
    body.alive = false;
    body.release();
    ++body.generation;
    if (body.generation == 0U) body.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::soft_body_view(SoftBodyId id,
                            SoftBodyDeviceView &output) const noexcept {
    output = {};
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->soft_bodies.size() || !impl_->soft_bodies[id.index] ||
        !impl_->soft_bodies[id.index]->alive ||
        impl_->soft_bodies[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body handle is stale");
    const SoftBodyStorage &body = *impl_->soft_bodies[id.index];
    output.positions = {body.positions, body.node_count};
    output.velocities = {body.velocities, body.node_count};
    output.bonds = {body.bonds, body.bond_count};
    output.surface_positions = {
        body.surface_positions, body.surface_vertex_count};
    output.surface_triangle_indices = {
        body.surface_indices, body.surface_index_count};
    output.rigid_contact_forces = {
        body.rigid_contact_forces, body.node_count};
    output.cloth_contact_forces = {body.cloth_forces, body.node_count};
    output.fluid_contact_forces = {body.fluid_forces, body.node_count};
    output.rope_contact_forces = {body.rope_forces, body.node_count};
    output.node_count = body.node_count;
    output.surface_vertex_count = body.surface_vertex_count;
    return success();
}

static bool valid_fluid_soft_options(FluidSoftBodyCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        options.solver_iterations > 0U && options.solver_iterations <= 16U;
}

Status World::add_fluid_soft_body_coupling(
    FluidSoftBodyCouplingOptions options, FluidSoftBodyCouplingId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    SoftBodyDeviceView view{};
    if (!(status = impl_->validate_handle(options.fluid, fluid)) ||
        !(status = soft_body_view(options.soft_body, view))) return status;
    if (!valid_fluid_soft_options(options))
        return failure(StatusCode::invalid_argument, "invalid fluid soft-body options");
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == options.fluid &&
            coupling->options.soft_body == options.soft_body)
            return failure(StatusCode::invalid_argument, "fluid and soft body are already coupled");
    std::uint32_t slot = 0;
    for (; slot < impl_->fluid_soft_couplings.size(); ++slot)
        if (!impl_->fluid_soft_couplings[slot] || !impl_->fluid_soft_couplings[slot]->alive) break;
    if (slot == impl_->fluid_soft_couplings.size())
        return failure(StatusCode::capacity_exceeded, "fluid soft-body coupling capacity exhausted");
    auto &body = *impl_->soft_bodies[options.soft_body.index];
    std::unique_ptr<FluidSoftCouplingStorage> coupling;
    std::vector<BvhNode> tree;
    std::vector<std::uint32_t> order, parents;
    double volume = 0;
    try {
        // glTF can duplicate a position at normal/UV seams. Validate geometric
        // edges, not rendering indices, without changing the skin bindings.
        std::map<std::array<float, 3>, std::uint32_t> vertices;
        std::vector<std::uint32_t> canonical(body.surface_vertex_count);
        for (std::uint32_t i = 0; i < body.surface_vertex_count; ++i) {
            const auto p = body.surface_rest_positions[i];
            canonical[i] = vertices.emplace(std::array{p.x, p.y, p.z}, vertices.size()).first->second;
        }
        struct Edge { std::uint32_t count{}; int winding{}; };
        std::map<std::pair<std::uint32_t, std::uint32_t>, Edge> edges;
        const Vec3 origin = body.surface_rest_positions[0];
        for (std::uint32_t base = 0; base < body.surface_index_count; base += 3) {
            const auto *tri = body.surface_indices + base;
            volume += dot(subtract(body.surface_rest_positions[tri[0]], origin),
                cross(subtract(body.surface_rest_positions[tri[1]], origin),
                      subtract(body.surface_rest_positions[tri[2]], origin))) / 6.0;
            for (std::uint32_t e = 0; e < 3; ++e) {
                auto a = canonical[tri[e]], b = canonical[tri[(e + 1) % 3]];
                auto &edge = edges[std::minmax(a, b)];
                ++edge.count;
                edge.winding += a < b ? 1 : -1;
            }
        }
        if (std::abs(volume) < 1.0e-10 || std::any_of(edges.begin(), edges.end(),
            [](const auto &edge) { return edge.second.count != 2 || edge.second.winding != 0; }))
            return failure(StatusCode::invalid_argument,
                           "fluid soft-body coupling needs a closed consistently wound surface");
        coupling = std::make_unique<FluidSoftCouplingStorage>();
        order.resize(body.surface_index_count / 3U);
        std::iota(order.begin(), order.end(), 0U);
        const auto centroid = [&](std::uint32_t triangle) {
            const auto *indices = body.surface_indices + triangle * 3U;
            return multiply(add(add(body.surface_rest_positions[indices[0]],
                body.surface_rest_positions[indices[1]]),
                body.surface_rest_positions[indices[2]]), 1.0F / 3.0F);
        };
        std::function<std::uint32_t(std::uint32_t, std::uint32_t, std::uint32_t)> build =
            [&](std::uint32_t first, std::uint32_t end, std::uint32_t parent) {
                const auto index = static_cast<std::uint32_t>(tree.size());
                BvhNode node{};
                node.minimum = {FLT_MAX, FLT_MAX, FLT_MAX};
                node.maximum = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (auto item = first; item < end; ++item) {
                    const Vec3 center = centroid(order[item]);
                    node.minimum = component_min(node.minimum, center);
                    node.maximum = component_max(node.maximum, center);
                }
                tree.push_back(node);
                parents.push_back(parent);
                if (end - first <= 4U) {
                    tree[index].first_triangle = first;
                    tree[index].triangle_count = end - first;
                } else {
                    const Vec3 extent = subtract(node.maximum, node.minimum);
                    const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 :
                        (extent.y >= extent.z ? 1 : 2);
                    const auto middle = first + (end - first) / 2U;
                    std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + end,
                        [&](auto a, auto b) {
                            const Vec3 ca = centroid(a), cb = centroid(b);
                            const float x = axis == 0 ? ca.x : (axis == 1 ? ca.y : ca.z);
                            const float y = axis == 0 ? cb.x : (axis == 1 ? cb.y : cb.z);
                            return x < y || (x == y && a < b);
                        });
                    const auto left = build(first, middle, index);
                    const auto right = build(middle, end, index);
                    tree[index].left = left;
                    tree[index].right = right;
                }
                return index;
            };
        build(0U, static_cast<std::uint32_t>(order.size()), k_invalid_dense);
    } catch (...) { return failure(StatusCode::out_of_memory, "failed to build fluid soft-body coupling"); }
#define PM_ALLOC_FLUID_SOFT(member, count) \
    status = allocate_managed(coupling->member, count); if (!status) return status
    PM_ALLOC_FLUID_SOFT(contacts, fluid->options.capacity);
    PM_ALLOC_FLUID_SOFT(counts, body.node_count);
    PM_ALLOC_FLUID_SOFT(position_deltas, body.node_count);
    PM_ALLOC_FLUID_SOFT(impulses, body.node_count);
    PM_ALLOC_FLUID_SOFT(previous_surface, body.surface_vertex_count);
    PM_ALLOC_FLUID_SOFT(bounds, 2U);
    PM_ALLOC_FLUID_SOFT(contact_count, 1U);
    PM_ALLOC_FLUID_SOFT(maximum_penetration, 1U);
    PM_ALLOC_FLUID_SOFT(tree, tree.size());
    PM_ALLOC_FLUID_SOFT(parents, parents.size());
    PM_ALLOC_FLUID_SOFT(ready, tree.size());
    PM_ALLOC_FLUID_SOFT(triangle_order, order.size());
#undef PM_ALLOC_FLUID_SOFT
    std::copy(tree.begin(), tree.end(), coupling->tree);
    std::copy(parents.begin(), parents.end(), coupling->parents);
    std::copy(order.begin(), order.end(), coupling->triangle_order);
    coupling->tree_count = static_cast<std::uint32_t>(tree.size());
    std::copy_n(body.surface_positions, body.surface_vertex_count, coupling->previous_surface);
    *coupling->contact_count = 0;
    *coupling->maximum_penetration = 0;
    coupling->orientation = volume < 0 ? -1.0F : 1.0F;
    if (impl_->fluid_soft_couplings[slot])
        coupling->generation = impl_->fluid_soft_couplings[slot]->generation;
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->fluid_soft_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status World::update_fluid_soft_body_coupling(
    FluidSoftBodyCouplingId id, FluidSoftBodyCouplingOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_soft_couplings.size() || !impl_->fluid_soft_couplings[id.index] ||
        !impl_->fluid_soft_couplings[id.index]->alive ||
        impl_->fluid_soft_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid soft-body coupling is stale");
    auto &coupling = *impl_->fluid_soft_couplings[id.index];
    if (!valid_fluid_soft_options(options) || !(options.fluid == coupling.options.fluid) ||
        !(options.soft_body == coupling.options.soft_body))
        return failure(StatusCode::invalid_argument, "invalid options or changed fluid soft-body endpoints");
    coupling.options = options;
    ++impl_->revision;
    return success();
}

Status World::remove_fluid_soft_body_coupling(FluidSoftBodyCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_soft_couplings.size() || !impl_->fluid_soft_couplings[id.index] ||
        !impl_->fluid_soft_couplings[id.index]->alive ||
        impl_->fluid_soft_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid soft-body coupling is stale");
    auto &coupling = *impl_->fluid_soft_couplings[id.index];
    coupling.release();
    coupling.alive = false;
    if (++coupling.generation == 0) coupling.generation = 1;
    ++impl_->revision;
    return success();
}

static bool valid_soft_cloth_options(SoftBodyClothCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        options.solver_iterations > 0U && options.solver_iterations <= 16U;
}

Status World::add_soft_body_cloth_coupling(
    SoftBodyClothCouplingOptions options,
    SoftBodyClothCouplingId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    SoftBodyDeviceView soft_view{};
    ClothDeviceView cloth_view_result{};
    if (!(status = soft_body_view(options.soft_body, soft_view)) ||
        !(status = cloth_view(options.cloth, cloth_view_result))) return status;
    if (!valid_soft_cloth_options(options))
        return failure(StatusCode::invalid_argument, "invalid soft-body cloth options");
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive &&
            coupling->options.soft_body == options.soft_body &&
            coupling->options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                           "soft body and cloth are already coupled");
    std::uint32_t slot = 0U;
    for (; slot < impl_->soft_cloth_couplings.size(); ++slot)
        if (!impl_->soft_cloth_couplings[slot] ||
            !impl_->soft_cloth_couplings[slot]->alive) break;
    if (slot == impl_->soft_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "soft-body cloth coupling capacity exhausted");
    std::unique_ptr<SoftClothCouplingStorage> coupling;
    try { coupling = std::make_unique<SoftClothCouplingStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "failed to allocate soft-body cloth coupling"); }
    status = allocate_managed(coupling->contacts, soft_view.node_count);
    if (!status) return status;
    status = allocate_managed(coupling->cloth_contact_counts,
                              impl_->cloths[options.cloth.index]->vertex_capacity);
    if (!status) return status;
    if (impl_->soft_cloth_couplings[slot])
        coupling->generation = impl_->soft_cloth_couplings[slot]->generation;
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->soft_cloth_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status World::update_soft_body_cloth_coupling(
    SoftBodyClothCouplingId id, SoftBodyClothCouplingOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_cloth_couplings.size() ||
        !impl_->soft_cloth_couplings[id.index] ||
        !impl_->soft_cloth_couplings[id.index]->alive ||
        impl_->soft_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body cloth coupling is stale");
    auto &coupling = *impl_->soft_cloth_couplings[id.index];
    if (!valid_soft_cloth_options(options) ||
        !(options.soft_body == coupling.options.soft_body) ||
        !(options.cloth == coupling.options.cloth))
        return failure(StatusCode::invalid_argument,
                       "invalid options or changed soft-body cloth endpoints");
    coupling.options = options;
    ++impl_->revision;
    return success();
}

Status World::remove_soft_body_cloth_coupling(SoftBodyClothCouplingId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_cloth_couplings.size() ||
        !impl_->soft_cloth_couplings[id.index] ||
        !impl_->soft_cloth_couplings[id.index]->alive ||
        impl_->soft_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body cloth coupling is stale");
    auto &coupling = *impl_->soft_cloth_couplings[id.index];
    coupling.alive = false;
    coupling.release();
    if (++coupling.generation == 0U) coupling.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::add_fluid_cloth_coupling(
    FluidClothCouplingOptions options,
    FluidClothCouplingId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation !=
            options.cloth.generation)
        return failure(StatusCode::invalid_handle,
                       "cloth coupling handle is invalid or stale");
    const ClothStorage &cloth = *impl_->cloths[options.cloth.index];
    if (!cloth.preserve_volume)
        return failure(StatusCode::invalid_argument,
            "contained fluid requires closed volume-preserving cloth");
    if (!finite(options.contact_distance) || options.contact_distance < 0.0F ||
        !finite(options.interaction_radius) || options.interaction_radius < 0.0F ||
        !finite(options.stiffness) || options.stiffness <= 0.0F ||
        !finite(options.damping) || options.damping < 0.0F ||
        !finite(options.tangential_drag) || options.tangential_drag < 0.0F ||
        !finite(options.maximum_force) || options.maximum_force <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid fluid-cloth coupling options");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.fluid == options.fluid &&
            coupling.options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                "fluid and cloth are already coupled");
    std::uint32_t slot = 0U;
    for (; slot < impl_->fluid_cloth_couplings.size(); ++slot)
        if (!impl_->fluid_cloth_couplings[slot].alive) break;
    if (slot == impl_->fluid_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "fluid-cloth coupling capacity exhausted");
    FluidClothCouplingResource &coupling =
        impl_->fluid_cloth_couplings[slot];
    coupling.options = options;
    coupling.alive = true;
    output = {slot, coupling.generation};
    ++impl_->revision;
    return success();
}

Status World::update_fluid_cloth_coupling(
    FluidClothCouplingId id,
    FluidClothCouplingOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_cloth_couplings.size() ||
        !impl_->fluid_cloth_couplings[id.index].alive ||
        impl_->fluid_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation !=
            options.cloth.generation ||
        !impl_->cloths[options.cloth.index]->preserve_volume)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling cloth is invalid or open");
    if (!finite(options.contact_distance) || options.contact_distance < 0.0F ||
        !finite(options.interaction_radius) || options.interaction_radius < 0.0F ||
        !finite(options.stiffness) || options.stiffness <= 0.0F ||
        !finite(options.damping) || options.damping < 0.0F ||
        !finite(options.tangential_drag) || options.tangential_drag < 0.0F ||
        !finite(options.maximum_force) || options.maximum_force <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid fluid-cloth coupling options");
    impl_->fluid_cloth_couplings[id.index].options = options;
    ++impl_->revision;
    return success();
}

Status World::remove_fluid_cloth_coupling(
    FluidClothCouplingId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_cloth_couplings.size() ||
        !impl_->fluid_cloth_couplings[id.index].alive ||
        impl_->fluid_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling handle is stale");
    FluidClothCouplingResource &coupling =
        impl_->fluid_cloth_couplings[id.index];
    coupling.alive = false;
    ++coupling.generation;
    if (coupling.generation == 0U) coupling.generation = 1U;
    ++impl_->revision;
    return success();
}

Status World::add_rigid_body(RigidBodyOptions options,
                             RigidBodyId &output) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    status = validate_body_options(options);
    if (!status) {
        return status;
    }
    status = impl_->validate_handle(options.mesh);
    if (!status) {
        return status;
    }
    if (impl_->rigid_body_count >= impl_->options.rigid_body_capacity) {
        return failure(StatusCode::capacity_exceeded,
                       "rigid body capacity is exhausted");
    }

    std::uint32_t slot_index = k_invalid_dense;
    for (std::uint32_t index = 0; index < impl_->slots.size(); ++index) {
        if (!impl_->slots[index].alive) {
            slot_index = index;
            break;
        }
    }
    if (slot_index == k_invalid_dense) {
        return failure(StatusCode::internal_error,
                       "no free rigid body handle slot was found");
    }

    RigidBodyState normalized_state = options.initial_state;
    normalized_state.orientation =
        normalized_quaternion(normalized_state.orientation);
    const TriangleMeshResource &mesh = impl_->meshes[options.mesh.index];

    Impl::Slot &slot = impl_->slots[slot_index];
    const std::uint32_t dense = impl_->rigid_body_count;
    slot.alive = true;
    slot.dense_index = dense;
    if (slot.generation == 0U) {
        slot.generation = 1U;
    }
    const RigidBodyId id{slot_index, slot.generation};
    impl_->parameters[dense] =
        make_parameters(options, mesh);
    impl_->accumulators[dense] = {};
    impl_->targets[dense] = {};
    impl_->ids[dense] = id;
    impl_->avbd_bodies[dense] = {};
    impl_->states[0][dense] = normalized_state;
    impl_->states[1][dense] = normalized_state;
    impl_->render_previous_states[dense] = normalized_state;
    ++impl_->rigid_body_count;
    ++impl_->revision;
    output = id;
    return success();
}

Status World::remove_rigid_body(RigidBodyId body) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    for (std::uint32_t index = 0U;
         index < impl_->options.rigid_constraint_capacity; ++index) {
        const RigidConstraintResource &constraint =
            impl_->rigid_constraints[index];
        if (constraint.alive &&
            (constraint.options.body_a == body ||
             constraint.options.body_b == body))
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a constraint");
    }
    for (const auto &coupling : impl_->smoke_rigid_couplings)
        if (coupling && coupling->alive && coupling->options.body == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by smoke coupling");
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index)
        if (impl_->paint_fields[index].alive &&
            impl_->paint_fields[index].options.body == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a paint field");
    for (std::uint32_t index = 0;
         index < impl_->options.paint_rule_capacity; ++index)
        if (impl_->paint_rules[index].alive &&
            impl_->paint_rules[index].options.rigid_source == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a paint rule");
    for(const auto &rope:impl_->ropes)
        if(rope && rope->alive &&
           ((rope->data.options.first.enabled && rope->data.options.first.body==body) ||
            (rope->data.options.last.enabled && rope->data.options.last.body==body)))
            return failure(StatusCode::invalid_argument,"rigid body is still referenced by a rope attachment");
    const std::uint32_t last = impl_->rigid_body_count - 1U;
    if (dense != last) {
        impl_->parameters[dense] = impl_->parameters[last];
        impl_->accumulators[dense] = impl_->accumulators[last];
        impl_->targets[dense] = impl_->targets[last];
        impl_->ids[dense] = impl_->ids[last];
        impl_->states[0][dense] = impl_->states[0][last];
        impl_->states[1][dense] = impl_->states[1][last];
        impl_->avbd_bodies[dense] = impl_->avbd_bodies[last];
        impl_->render_previous_states[dense] =
            impl_->render_previous_states[last];
        impl_->slots[impl_->ids[dense].index].dense_index = dense;
    }
    --impl_->rigid_body_count;
    Impl::Slot &slot = impl_->slots[body.index];
    slot.alive = false;
    slot.dense_index = k_invalid_dense;
    ++slot.generation;
    if (slot.generation == 0U) {
        slot.generation = 1U;
    }
    ++impl_->revision;
    return success();
}

Status World::set_rigid_body_state(RigidBodyId body,
                                   RigidBodyState state) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(state)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body state must contain finite values");
    }
    const float quaternion_size = state.orientation.x * state.orientation.x +
                                  state.orientation.y * state.orientation.y +
                                  state.orientation.z * state.orientation.z +
                                  state.orientation.w * state.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "rigid body orientation must be nonzero");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    state.orientation = normalized_quaternion(state.orientation);
    impl_->states[0][dense] = state;
    impl_->states[1][dense] = state;
    impl_->render_previous_states[dense] = state;
    impl_->avbd_bodies[dense] = {};
    impl_->accumulators[dense] = {};
    impl_->targets[dense] = {};
    for (unsigned index = 0; index < impl_->options.rigid_constraint_capacity; ++index) {
        auto &joint = impl_->rigid_constraints[index];
        if (joint.alive && (joint.options.body_a == body || joint.options.body_b == body))
            std::fill_n(joint.avbd_rows, 18U, avbd::Row{});
    }
    for (auto &smoke : impl_->smokes)
        if (smoke && smoke->alive)
            smoke->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status World::set_kinematic_target(RigidBodyId body,
                                   RigidBodyState target) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(target)) {
        return failure(StatusCode::invalid_argument,
                       "kinematic target must contain finite values");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::kinematic) {
        return failure(StatusCode::invalid_argument,
                       "kinematic targets require a kinematic body");
    }
    const float quaternion_size = target.orientation.x * target.orientation.x +
                                  target.orientation.y * target.orientation.y +
                                  target.orientation.z * target.orientation.z +
                                  target.orientation.w * target.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "kinematic orientation must be nonzero");
    }
    target.orientation = normalized_quaternion(target.orientation);
    impl_->targets[dense] = {target, true};
    return success();
}

Status World::apply_force(RigidBodyId body, Vec3 force,
                          Vec3 world_point) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(force) || !finite(world_point)) {
        return failure(StatusCode::invalid_argument,
                       "force and application point must be finite");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::dynamic) {
        return failure(StatusCode::invalid_argument,
                       "forces may only be applied to dynamic bodies");
    }
    BodyAccumulator &accumulator = impl_->accumulators[dense];
    accumulator.force = add(accumulator.force, force);
    const Vec3 arm = subtract(
        world_point, impl_->states[impl_->current_state][dense].position);
    accumulator.torque = add(accumulator.torque, cross(arm, force));
    return success();
}

Status World::apply_central_acceleration(
    HostSpan<RigidBodyId> bodies, Vec3 acceleration) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument,
                       "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (!finite(acceleration) ||
        (bodies.size != 0U && bodies.data == nullptr) ||
        bodies.size > impl_->rigid_body_count)
        return failure(StatusCode::invalid_argument,
                       "central acceleration batch is invalid");
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense = 0U;
        status = impl_->validate_handle(bodies.data[index], dense);
        if (!status) return status;
        if (impl_->parameters[dense].motion != MotionType::dynamic)
            return failure(StatusCode::invalid_argument,
                           "central acceleration requires dynamic bodies");
        for (std::uint64_t prior = 0U; prior < index; ++prior)
            if (bodies.data[prior] == bodies.data[index])
                return failure(StatusCode::invalid_argument,
                               "central acceleration body is duplicated");
    }
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense = 0U;
        status = impl_->validate_handle(bodies.data[index], dense);
        if (!status) return status;
        const float mass = 1.0F / impl_->parameters[dense].inverse_mass;
        impl_->accumulators[dense].force = add(
            impl_->accumulators[dense].force,
            multiply(acceleration, mass));
    }
    return success();
}

Status World::apply_impulse(RigidBodyId body, Vec3 impulse,
                            Vec3 world_point) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(impulse) || !finite(world_point)) {
        return failure(StatusCode::invalid_argument,
                       "impulse and application point must be finite");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::dynamic) {
        return failure(StatusCode::invalid_argument,
                       "impulses may only be applied to dynamic bodies");
    }
    BodyAccumulator &accumulator = impl_->accumulators[dense];
    accumulator.impulse = add(accumulator.impulse, impulse);
    const Vec3 arm = subtract(
        world_point, impl_->states[impl_->current_state][dense].position);
    accumulator.angular_impulse =
        add(accumulator.angular_impulse, cross(arm, impulse));
    return success();
}

Status World::rigid_body_view(RigidBodyDeviceView &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    output.ids = {impl_->ids, impl_->rigid_body_count};
    output.states = {impl_->states[impl_->current_state],
                     impl_->rigid_body_count};
    output.previous_states = {
        impl_->render_previous_states, impl_->rigid_body_count};
    if (impl_->debug_applied_forces != nullptr) {
        output.applied_forces = {
            impl_->debug_applied_forces, impl_->rigid_body_count};
        output.applied_torques = {
            impl_->debug_applied_torques, impl_->rigid_body_count};
    }
    output.revision = impl_->revision;
    return success();
}

Status World::read_rigid_body_state(RigidBodyId body, RigidBodyState &output,
                                    cudaStream_t stream) const noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) {
        return status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    RigidBodyState temporary{};
    cudaError_t error = cudaMemcpyAsync(
        &temporary, impl_->states[impl_->current_state] + dense,
        sizeof(RigidBodyState), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "rigid body state readback failed");
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "rigid body state readback synchronization failed");
    }
    output = temporary;
    return success();
}

Status World::query_hit_box(HitBox box, HitBoxResult &output,
                            cudaStream_t stream) const noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument,
                       "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    const float orientation_length_squared =
        box.orientation.x * box.orientation.x +
        box.orientation.y * box.orientation.y +
        box.orientation.z * box.orientation.z +
        box.orientation.w * box.orientation.w;
    if (!finite(box.center) || !finite(box.orientation) ||
        !finite(box.half_extents) || box.half_extents.x <= 0.0F ||
        box.half_extents.y <= 0.0F || box.half_extents.z <= 0.0F ||
        orientation_length_squared <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "hit box transform and half extents are invalid");
    }
    box.orientation = normalized_quaternion(box.orientation);

    constexpr std::uint32_t block_size = 128U;
    if (impl_->rigid_body_count != 0U) {
        query_rigid_hit_box_kernel<<<
            (impl_->rigid_body_count + block_size - 1U) / block_size,
            block_size, 0, stream>>>(
            box, impl_->parameters, impl_->states[impl_->current_state],
            impl_->meshes, impl_->rigid_body_count,
            impl_->hit_box_rigid_flags);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return cuda_failure(error, "rigid hit-box query launch failed");
    }
    for (const auto &owner : impl_->fluids) {
        if (!owner || !owner->alive || *owner->count == 0U) continue;
        query_particle_hit_box_kernel<<<
            (*owner->count + block_size - 1U) / block_size,
            block_size, 0, stream>>>(
            box, owner->positions, *owner->count, owner->hit_box_flags);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return cuda_failure(error, "particle hit-box query launch failed");
    }
    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        return cuda_failure(error, "hit-box query synchronization failed");

    HitBoxResult temporary;
    try {
        temporary.rigid_bodies.reserve(impl_->rigid_body_count);
        std::size_t particle_count = 0U;
        for (const auto &owner : impl_->fluids)
            if (owner && owner->alive) particle_count += *owner->count;
        temporary.particles.reserve(particle_count);
        for (std::uint32_t body = 0U; body < impl_->rigid_body_count; ++body)
            if (impl_->hit_box_rigid_flags[body] != 0U)
                temporary.rigid_bodies.push_back(impl_->ids[body]);
        for (std::uint32_t slot = 0U; slot < impl_->fluids.size(); ++slot) {
            const auto &owner = impl_->fluids[slot];
            if (!owner || !owner->alive) continue;
            const FluidId fluid{slot, owner->generation};
            for (std::uint32_t particle = 0U; particle < *owner->count;
                 ++particle)
                if (owner->hit_box_flags[particle] != 0U)
                    temporary.particles.push_back(
                        {fluid, owner->ids[particle]});
        }
        std::sort(temporary.rigid_bodies.begin(),
                  temporary.rigid_bodies.end(),
                  [](RigidBodyId left, RigidBodyId right) {
                      return left.index < right.index ||
                          (left.index == right.index &&
                           left.generation < right.generation);
                  });
        std::sort(temporary.particles.begin(), temporary.particles.end(),
                  [](HitBoxParticle left, HitBoxParticle right) {
                      if (left.fluid.index != right.fluid.index)
                          return left.fluid.index < right.fluid.index;
                      if (left.fluid.generation != right.fluid.generation)
                          return left.fluid.generation < right.fluid.generation;
                      return left.stable_particle_id <
                             right.stable_particle_id;
                  });
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "hit-box query result allocation failed");
    }
    output = std::move(temporary);
    return success();
}

Status World::add_rigid_constraint(
    RigidConstraintOptions options, RigidConstraintId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    status = validate_constraint_options(options);
    if (!status) return status;
    std::uint32_t dense_a = 0U;
    std::uint32_t dense_b = 0U;
    if (!(status = impl_->validate_handle(options.body_a, dense_a)) ||
        !(status = impl_->validate_handle(options.body_b, dense_b))) return status;
    if (impl_->parameters[dense_a].motion != MotionType::dynamic &&
        impl_->parameters[dense_b].motion != MotionType::dynamic)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint needs at least one dynamic body");
    if (impl_->rigid_constraint_count >= impl_->options.rigid_constraint_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "rigid constraint capacity is exhausted");
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U;
         index < impl_->options.rigid_constraint_capacity; ++index)
        if (!impl_->rigid_constraints[index].alive) {
            slot = index;
            break;
        }
    if (slot == k_invalid_dense)
        return failure(StatusCode::internal_error,
                       "no free rigid constraint slot was found");
    options.local_orientation_a = normalized_quaternion(
        options.local_orientation_a);
    options.local_orientation_b = normalized_quaternion(
        options.local_orientation_b);
    RigidConstraintResource &resource = impl_->rigid_constraints[slot];
    if (resource.generation == 0U) resource.generation = 1U;
    resource.options = options;
    resource.state = {};
    resource.state.enabled = options.enabled;
    std::fill_n(resource.avbd_rows, 18U, avbd::Row{});
    resource.alive = true;
    ++impl_->rigid_constraint_count;
    ++impl_->revision;
    output = {slot, resource.generation};
    return success();
}

Status World::update_rigid_constraint(
    RigidConstraintId id, RigidConstraintOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    status = validate_constraint_options(options);
    if (!status) return status;
    RigidConstraintResource *resource = nullptr;
    status = impl_->validate_handle(id, resource);
    if (!status) return status;
    std::uint32_t dense_a = 0U;
    std::uint32_t dense_b = 0U;
    if (!(status = impl_->validate_handle(options.body_a, dense_a)) ||
        !(status = impl_->validate_handle(options.body_b, dense_b))) return status;
    if (impl_->parameters[dense_a].motion != MotionType::dynamic &&
        impl_->parameters[dense_b].motion != MotionType::dynamic)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint needs at least one dynamic body");
    options.local_orientation_a = normalized_quaternion(
        options.local_orientation_a);
    options.local_orientation_b = normalized_quaternion(
        options.local_orientation_b);
    const bool contact_topology_changed = !(resource->options.body_a == options.body_a)
        || !(resource->options.body_b == options.body_b)
        || resource->options.enabled != options.enabled
        || resource->options.disable_collisions != options.disable_collisions
        || resource->options.type != options.type || resource->state.broken;
    const bool contact_cache_current = impl_->rigid_contact_revision == impl_->revision;
    resource->options = options;
    resource->state = {};
    resource->state.enabled = options.enabled;
    std::fill_n(resource->avbd_rows, 18U, avbd::Row{});
    ++impl_->revision;
    if (!contact_topology_changed && contact_cache_current)
        impl_->rigid_contact_revision = impl_->revision;
    return success();
}

Status World::remove_rigid_constraint(RigidConstraintId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    RigidConstraintResource *resource = nullptr;
    status = impl_->validate_handle(id, resource);
    if (!status) return status;
    resource->alive = false;
    resource->state = {};
    ++resource->generation;
    if (resource->generation == 0U) resource->generation = 1U;
    --impl_->rigid_constraint_count;
    ++impl_->revision;
    return success();
}

Status World::read_rigid_constraint_state(
    RigidConstraintId id, RigidConstraintState &output,
    cudaStream_t stream) const noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "world is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    RigidConstraintResource *resource = nullptr;
    status = impl_->validate_handle(id, resource);
    if (!status) return status;
    RigidConstraintState temporary{};
    cudaError_t error = cudaMemcpyAsync(
        &temporary, &resource->state, sizeof(temporary),
        cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        return cuda_failure(error, "rigid constraint state readback failed");
    output = temporary;
    return success();
}

Status World::step_async(StepOptions options, FrameToken &completion,
                         cudaStream_t stream) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (completion.pending()) {
        return failure(StatusCode::busy,
                       "completion token already represents a pending frame");
    }
    if (!finite(options.timestep) || options.timestep <= 0.0F ||
        options.substeps == 0U || options.substeps > 1'024U ||
        !finite(options.gravity) || options.rigid_contact_pass_limit > 64U) {
        return failure(StatusCode::invalid_argument,
                       "step timestep, substeps, gravity, or contact pass limit is invalid");
    }
    // Advance rigid attachments and ropes on the same clock. Splitting only
    // the rope after a coarse rigid step leaves endpoints discontinuous.
    for (const auto &rope : impl_->ropes) if (rope && rope->alive) {
        const float required = std::ceil(options.timestep /
            rope->data.options.maximum_substep_timestep);
        if (!finite(required) || required > 1'024.0F)
            return failure(StatusCode::invalid_argument,
                           "rope timestep limit requires more than 1024 substeps");
        options.substeps = std::max(options.substeps,
                                   static_cast<std::uint32_t>(required));
    }
    for (auto &cloth : impl_->cloths) if (cloth && cloth->alive) {
        status = rebuild_cloth_topology(*cloth);
        if (!status) return status;
    }
    const bool debug_enabled =
        impl_->options.physics_debug.frame_capacity != 0U;
    const bool collect_rigid_contacts =
        options.collect_rigid_contacts || debug_enabled;
    const bool collect_fluid_contacts =
        options.collect_fluid_contacts || debug_enabled;
    if (impl_->rigid_body_count == 0U) {
        *impl_->rigid_contact_count = 0U;
    }
    *impl_->fluid_neighbor_overflow = 0U;
    *impl_->fluid_maximum_neighbor_count = 0U;
    *impl_->fluid_contact_count = 0U;
    *impl_->fluid_contact_overflow = 0U;

    std::unique_ptr<FrameToken::Impl> token_impl;
    if (completion.impl_) {
        token_impl = std::move(completion.impl_);
    } else {
        try {
            token_impl = std::make_unique<FrameToken::Impl>();
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "failed to allocate completion token state");
        }
    }

    std::shared_ptr<CompletionState> frame;
    try {
        frame = std::make_shared<CompletionState>();
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate frame completion state");
    }
    frame->fluid_neighbor_overflow = impl_->fluid_neighbor_overflow;
    if (debug_enabled) {
        try {
            frame->on_complete = [implementation = impl_.get(), options]() {
                return implementation->record_debug_frame(options);
            };
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "physics debug completion allocation failed");
        }
    }
    cudaError_t error =
        cudaEventCreateWithFlags(&frame->event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to create frame completion event");
    }
    if (impl_->rigid_body_count != 0U) {
        error = cudaMemcpyAsync(
            impl_->render_previous_states,
            impl_->states[impl_->current_state],
            impl_->rigid_body_count * sizeof(RigidBodyState),
            cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess)
            return cuda_failure(error,
                                "rigid interpolation snapshot failed");
    }

    impl_->timing_available = false;
    impl_->timing_boundary_count = 0U;
    std::size_t timing_boundary = 0U;
    if (options.collect_kernel_timings) {
        std::size_t maximum_stages =
            static_cast<std::size_t>(options.substeps) * 8U + 1U;
        for (const auto &fluid : impl_->fluids) {
            if (fluid && fluid->alive) {
                maximum_stages += 2U +
                    static_cast<std::size_t>(options.substeps) *
                        fluid->options.solver_iterations *
                        (9U + 2U * impl_->fluid_soft_couplings.size() +
                         2U * impl_->fluid_rope_couplings.size());
            }
        }
        for (const auto &smoke : impl_->smokes)
            if (smoke && smoke->alive) maximum_stages += 3U;
        for (const auto &cloth : impl_->cloths) {
            if (cloth && cloth->alive)
                maximum_stages += static_cast<std::size_t>(options.substeps) *
                    (cloth->solver_iterations + 2U);
        }
        for (const auto &body : impl_->soft_bodies) {
            if (body && body->alive)
                maximum_stages += static_cast<std::size_t>(options.substeps) *
                    (body->solver_iterations + 3U +
                     (body->solver_iterations + 1U) / 2U);
        }
        status = impl_->prepare_timing_events(
            maximum_stages + options.substeps *
                (impl_->ropes.size()+impl_->rope_soft_body_couplings.size()) + 2U);
        if (!status) {
            return status;
        }
        error = cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to begin kernel timing");
        }
    }

    const auto record_timing_stage = [&](TimingStage stage,
                                         std::uint32_t launches = 0U) noexcept -> Status {
        if (!options.collect_kernel_timings) {
            return success();
        }
        if (timing_boundary >= impl_->timing_events.size()) {
            cudaStreamSynchronize(stream);
            return failure(StatusCode::internal_error,
                           "kernel timing stage budget exhausted");
        }
        impl_->timing_stages.push_back(stage);
        impl_->timing_launch_counts.push_back(launches);
        const cudaError_t timing_error =
            cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        return timing_error == cudaSuccess
                   ? success()
                   : cuda_failure(timing_error,
                                  "failed to record kernel timing boundary");
    };

    constexpr std::uint32_t block_size = 128U;
    const std::uint32_t block_count =
        (impl_->rigid_body_count + block_size - 1U) / block_size;
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
        error = cudaMemcpyAsync(coupling->previous_surface, body.surface_positions,
            body.surface_vertex_count * sizeof(Vec3), cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) return cuda_failure(error, "previous fluid soft-body skin copy failed");
    }
    for (auto &owner : impl_->smokes) {
        if (!owner || !owner->alive || owner->count == 0U ||
            owner->grid.resolution != 0U) continue;
        auto &smoke = *owner;
        smoke_emit_cells<<<(smoke.options.capacity + block_size - 1U) /
            block_size, block_size, 0, stream>>>(smoke.positions, smoke.ages,
            smoke.count, smoke.options, smoke.keys[0], smoke.indices[0]);
        error = cub::DeviceRadixSort::SortPairs(
            smoke.sort_workspace, smoke.sort_workspace_size,
            smoke.keys[0], smoke.keys[1], smoke.indices[0], smoke.indices[1],
            smoke.options.capacity, 0, 64, stream);
        if (error != cudaSuccess)
            return cuda_failure(error, "smoke neighbor index failed");
        smoke.index_dirty = false;
    }
    // The Eulerian air step precedes deformable integration, so every
    // coupled surface reads the same projected field for this frame.
    for (auto &owner : impl_->smokes) {
        if (!owner || !owner->alive || owner->grid.resolution == 0U) continue;
        auto &smoke = *owner;
        auto &grid = smoke.grid;
        const auto blocks = (grid.cell_count + 255U) / 256U;
        SmokeGridField field{grid.resolution, grid.height,
            grid.minimum, grid.spacing, grid.velocity, grid.density,
            grid.pressure[0]};
        for (int axis = 0; axis < 3; ++axis)
            field.face[axis] = grid.face_velocity[axis][0];
        field.vorticity = grid.vorticity;
        field.strain = grid.strain;
        const auto largest = std::max({grid.cell_count, grid.face_count[0],
            grid.face_count[1], grid.face_count[2]});
        smoke_grid_clear_obstacles<<<(largest + 255U) / 256U,
            256U, 0, stream>>>(grid.solid, grid.cell_count,
            grid.face_open[0], grid.face_open[1], grid.face_open[2],
            grid.face_wall_velocity[0], grid.face_wall_velocity[1],
            grid.face_wall_velocity[2],
            grid.face_count[0],
            grid.face_count[1], grid.face_count[2]);
        cudaMemsetAsync(grid.density_accumulator,0,
            grid.cell_count*sizeof(unsigned long long),stream);
        cudaMemsetAsync(grid.temperature_accumulator,0,
            grid.cell_count*sizeof(unsigned long long),stream);
        if (smoke.count != 0U)
            smoke_grid_splat_particles<<<
                (smoke.count + 127U) / 128U, 128U, 0, stream>>>(
                smoke.positions, smoke.ages, smoke.thermal_lift,
                smoke.count, smoke.options, field,grid.density_accumulator,
                grid.temperature_accumulator);
        smoke_grid_resolve_particle_fields<<<blocks,256U,0,stream>>>(
            grid.density_accumulator,grid.temperature_accumulator,
            grid.density,grid.temperature,grid.cell_count);
        bool moving_triangle_boundary = false;
        for (const auto &coupling : impl_->smoke_rigid_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            std::uint32_t body{};
            status = impl_->validate_handle(coupling->options.body, body);
            if (!status) return status;
            moving_triangle_boundary |=
                impl_->parameters[body].motion != MotionType::static_body;
        }
        for (const auto &coupling : impl_->smoke_soft_body_couplings)
            moving_triangle_boundary |= coupling && coupling->alive &&
                coupling->options.enabled &&
                coupling->options.smoke.index < impl_->smokes.size() &&
                impl_->smokes[coupling->options.smoke.index].get() == &smoke;
        for (const auto &coupling : impl_->smoke_cloth_couplings)
            moving_triangle_boundary |= coupling && coupling->alive &&
                coupling->options.enabled &&
                coupling->options.smoke.index < impl_->smokes.size() &&
                impl_->smokes[coupling->options.smoke.index].get() == &smoke;
        const int raster_passes = moving_triangle_boundary ||
            !grid.static_metadata_valid ? 2 : 1;
        for (const auto &coupling : impl_->smoke_rigid_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            std::uint32_t body{};
            status = impl_->validate_handle(coupling->options.body, body);
            if (!status) return status;
            const auto &mesh = impl_->meshes[impl_->parameters[body].mesh.index];
            const auto triangles = mesh.index_count / 3U;
            for (int pass = 0; pass < raster_passes; ++pass)
                smoke_grid_raster_rigid<<<(triangles + 127U) / 128U,
                    128U, 0, stream>>>(mesh,
                    impl_->states[impl_->current_state], body, field, grid.solid,
                    grid.face_open[0], grid.face_open[1], grid.face_open[2],
                    grid.face_wall_velocity[0], grid.face_wall_velocity[1],
                    grid.face_wall_velocity[2], grid.face_normal[0],
                    grid.face_normal[1], grid.face_normal[2],
                    grid.face_nearest_triangle[0],
                    grid.face_nearest_triangle[1],
                    grid.face_nearest_triangle[2],pass!=0);
        }
        for (const auto &coupling : impl_->smoke_soft_body_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            const auto &body = *impl_->soft_bodies[
                coupling->options.soft_body.index];
            const auto triangles = body.surface_index_count / 3U;
            for (int pass = 0; pass < raster_passes; ++pass)
                smoke_grid_raster_soft<<<(triangles + 127U) / 128U,
                    128U, 0, stream>>>(body.surface_positions,
                    body.surface_bindings, body.velocities,
                    body.surface_indices, body.surface_index_count,
                    field, grid.solid, grid.face_open[0], grid.face_open[1],
                    grid.face_open[2], grid.face_wall_velocity[0],
                    grid.face_wall_velocity[1], grid.face_wall_velocity[2],
                    grid.face_normal[0], grid.face_normal[1],
                    grid.face_normal[2], grid.face_nearest_triangle[0],
                    grid.face_nearest_triangle[1],
                    grid.face_nearest_triangle[2],pass!=0);
        }
        for (const auto &coupling : impl_->smoke_cloth_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            const auto &cloth = *impl_->cloths[coupling->options.cloth.index];
            const auto triangles = cloth.index_count / 3U;
            for (int pass = 0; pass < raster_passes; ++pass)
                smoke_grid_raster_deformable<<<(triangles + 127U) / 128U,
                    128U, 0, stream>>>(cloth.positions, cloth.velocities,
                    cloth.indices, cloth.index_count, field, grid.solid,
                    grid.face_open[0], grid.face_open[1], grid.face_open[2],
                    grid.face_wall_velocity[0], grid.face_wall_velocity[1],
                    grid.face_wall_velocity[2], grid.face_normal[0],
                    grid.face_normal[1], grid.face_normal[2],
                    grid.face_nearest_triangle[0],
                    grid.face_nearest_triangle[1],
                    grid.face_nearest_triangle[2],pass!=0);
        }
        grid.static_metadata_valid = !moving_triangle_boundary;

        // Fixed inlet and far-field normal velocities are zero-aperture
        // pressure boundaries. Downstream faces stay open with p=0 outside.
        for (int axis = 0; axis < 3; ++axis)
            smoke_grid_mark_domain_boundaries<<<
                (grid.face_count[axis] + 255U) / 256U, 256U, 0, stream>>>(
                axis, field, grid.face_open[axis],
                grid.face_wall_velocity[axis], smoke.options);

        // Coarsen the cut-face apertures once; all pressure V-cycles reuse
        // the same geometry for this frame.
        int fine_n = int(grid.resolution), fine_h = int(grid.height);
        const float *fine_open[3] = {grid.face_open[0], grid.face_open[1],
                                     grid.face_open[2]};
        for (auto &level : grid.coarse) {
            for (int axis = 0; axis < 3; ++axis) {
                const auto count = smoke_grid_face_count(axis,
                    int(level.n), int(level.height));
                smoke_grid_restrict_open<<<(count + 255U) / 256U,
                    256U, 0, stream>>>(axis, fine_open[axis], fine_n, fine_h,
                    level.open[axis], int(level.n), int(level.height));
                fine_open[axis] = level.open[axis];
            }
            fine_n = int(level.n); fine_h = int(level.height);
        }

        // RK2 MacCormack self-advection on the staggered faces.
        for (int axis = 0; axis < 3; ++axis)
            smoke_grid_advect_face<<<
                (grid.face_count[axis] + 255U) / 256U, 256U, 0, stream>>>(
                axis, field, grid.face_velocity[axis][0],
                grid.face_velocity[axis][1], options.timestep);
        SmokeGridField predicted = field;
        for (int axis = 0; axis < 3; ++axis)
            predicted.face[axis] = grid.face_velocity[axis][1];
        for (int axis = 0; axis < 3; ++axis)
            smoke_grid_advect_face<<<
                (grid.face_count[axis] + 255U) / 256U, 256U, 0, stream>>>(
                axis, predicted, grid.face_velocity[axis][1],
                grid.face_reverse[axis], -options.timestep);
        for (int axis = 0; axis < 3; ++axis)
            smoke_grid_correct_face<<<
                (grid.face_count[axis] + 255U) / 256U, 256U, 0, stream>>>(
                axis, field, grid.face_velocity[axis][0],
                grid.face_velocity[axis][1], grid.face_reverse[axis],
                grid.face_velocity[axis][1], options.timestep);

        smoke_grid_cell_diagnostics<<<blocks, 256U, 0, stream>>>(
            predicted, nullptr, grid.vorticity, grid.strain, nullptr);
        smoke_grid_subgrid_force<<<blocks, 256U, 0, stream>>>(predicted,
            grid.vorticity, grid.subgrid_force,
            smoke.options.vorticity_confinement);
        for (int axis = 0; axis < 3; ++axis)
            smoke_grid_apply_face_forces<<<
                (grid.face_count[axis] + 255U) / 256U, 256U, 0, stream>>>(
                axis, predicted, grid.face_velocity[axis][1],
                grid.face_velocity[axis][0], grid.face_open[axis],
                grid.face_wall_velocity[axis], grid.strain,
                grid.subgrid_force, grid.temperature, smoke.options,
                options.timestep, options.gravity);
        SmokeGridField forced = field;
        for (int axis = 0; axis < 3; ++axis) {
            forced.face[axis] = grid.face_velocity[axis][0];
            smoke_grid_apply_face_boundaries<<<
                (grid.face_count[axis] + 255U) / 256U, 256U, 0, stream>>>(
                axis, forced, grid.face_velocity[axis][0],
                grid.face_open[axis], grid.face_wall_velocity[axis],
                smoke.options);
        }

        cudaMemsetAsync(grid.rhs_max, 0, sizeof(float), stream);
        cudaMemsetAsync(grid.pressure_converged, 0,
            sizeof(unsigned int), stream);
        smoke_grid_divergence<<<blocks, 256U, 0, stream>>>(forced,
            grid.divergence, grid.rhs_max, options.timestep);

        int level_n[4] = {int(grid.resolution), int(grid.coarse[0].n),
                          int(grid.coarse[1].n), int(grid.coarse[2].n)};
        int level_h[4] = {int(grid.height), int(grid.coarse[0].height),
                          int(grid.coarse[1].height), int(grid.coarse[2].height)};
        float level_spacing[4] = {grid.spacing, grid.spacing*2.0F,
                                  grid.spacing*4.0F, grid.spacing*8.0F};
        float *level_pressure[4][2] = {
            {grid.pressure[0], grid.pressure[1]},
            {grid.coarse[0].pressure[0], grid.coarse[0].pressure[1]},
            {grid.coarse[1].pressure[0], grid.coarse[1].pressure[1]},
            {grid.coarse[2].pressure[0], grid.coarse[2].pressure[1]}};
        float *level_rhs[4] = {grid.divergence, grid.coarse[0].rhs,
            grid.coarse[1].rhs, grid.coarse[2].rhs};
        float *level_residual[4] = {grid.residual, grid.coarse[0].residual,
            grid.coarse[1].residual, grid.coarse[2].residual};
        float *level_open[4][3] = {
            {grid.face_open[0],grid.face_open[1],grid.face_open[2]},
            {grid.coarse[0].open[0],grid.coarse[0].open[1],grid.coarse[0].open[2]},
            {grid.coarse[1].open[0],grid.coarse[1].open[1],grid.coarse[1].open[2]},
            {grid.coarse[2].open[0],grid.coarse[2].open[1],grid.coarse[2].open[2]}};
        const auto smooth = [&](int level, int iterations) {
            const auto count = std::uint32_t(level_n[level] * level_h[level] *
                                             level_n[level]);
            for (int iteration = 0; iteration < iterations; ++iteration) {
                smoke_grid_pressure_smooth<<<(count + 255U) / 256U,
                    256U, 0, stream>>>(level_n[level], level_h[level],
                    level_spacing[level], level_rhs[level],
                    level_pressure[level][0], level_pressure[level][1],
                    level_open[level][0], level_open[level][1],
                    level_open[level][2], grid.pressure_converged);
                std::swap(level_pressure[level][0], level_pressure[level][1]);
            }
        };
        // One V-cycle costs about 4.6 fine-grid Jacobi sweeps after accounting
        // for the geometrically smaller levels. Keep the authored value as a
        // fine-grid-equivalent work ceiling and stop earlier on the GPU when
        // the relative residual reaches the requested tolerance.
        const auto cycles = std::max(1U,
            (smoke.options.grid_pressure_iterations + 4U) / 5U);
        for (std::uint32_t cycle = 0U; cycle < cycles; ++cycle) {
            for (int level = 0; level < 3; ++level) {
                smooth(level, 2);
                const auto count = std::uint32_t(level_n[level] *
                    level_h[level] * level_n[level]);
                smoke_grid_pressure_residual<<<(count + 255U) / 256U,
                    256U, 0, stream>>>(level_n[level], level_h[level],
                    level_spacing[level], level_rhs[level],
                    level_pressure[level][0], level_residual[level],
                    level_open[level][0], level_open[level][1],
                    level_open[level][2], nullptr, grid.pressure_converged);
                const auto coarse_count = std::uint32_t(level_n[level+1] *
                    level_h[level+1] * level_n[level+1]);
                smoke_grid_restrict_residual<<<
                    (coarse_count + 255U) / 256U, 256U, 0, stream>>>(
                    level_residual[level], level_n[level], level_h[level],
                    level_rhs[level+1], level_n[level+1], level_h[level+1]);
                cudaMemsetAsync(level_pressure[level+1][0], 0,
                    coarse_count * sizeof(float), stream);
                cudaMemsetAsync(level_pressure[level+1][1], 0,
                    coarse_count * sizeof(float), stream);
            }
            smooth(3, 12);
            for (int level = 2; level >= 0; --level) {
                const auto fine_count = std::uint32_t(level_n[level] *
                    level_h[level] * level_n[level]);
                smoke_grid_prolong_add<<<(fine_count + 255U) / 256U,
                    256U, 0, stream>>>(level_pressure[level+1][0],
                    level_n[level+1], level_h[level+1],
                    level_pressure[level][0], level_n[level], level_h[level]);
                smooth(level, 2);
            }
            smoke_grid_reset_residual<<<1U,1U,0,stream>>>(
                grid.residual_max,grid.pressure_converged);
            smoke_grid_pressure_residual<<<blocks, 256U, 0, stream>>>(
                level_n[0], level_h[0], level_spacing[0], level_rhs[0],
                level_pressure[0][0], level_residual[0], level_open[0][0],
                level_open[0][1], level_open[0][2], grid.residual_max,
                grid.pressure_converged);
            smoke_grid_compare_residual<<<1U,1U,0,stream>>>(grid.rhs_max,
                grid.residual_max, smoke.options.grid_pressure_tolerance,
                grid.pressure_converged);
        }
        grid.pressure[0] = level_pressure[0][0];
        grid.pressure[1] = level_pressure[0][1];
        for (int level = 0; level < 3; ++level) {
            grid.coarse[level].pressure[0] = level_pressure[level+1][0];
            grid.coarse[level].pressure[1] = level_pressure[level+1][1];
        }
        for (int axis = 0; axis < 3; ++axis) {
            smoke_grid_project_face<<<
                (grid.face_count[axis] + 255U) / 256U, 256U, 0, stream>>>(
                axis, forced, grid.pressure[0], grid.face_velocity[axis][0],
                grid.face_open[axis], grid.face_wall_velocity[axis],
                options.timestep);
        }
        SmokeGridField projected = forced;
        projected.pressure = grid.pressure[0];
        projected.vorticity = grid.vorticity;
        smoke_grid_cell_diagnostics<<<blocks, 256U, 0, stream>>>(projected,
            grid.velocity, grid.vorticity, grid.strain, grid.divergence);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return cuda_failure(error, "smoke grid step failed");
    }
    if (std::any_of(impl_->smokes.begin(), impl_->smokes.end(),
        [](const auto &smoke) {
            return smoke && smoke->alive && smoke->grid.resolution != 0U;
        })) {
        status = record_timing_stage(TimingStage::smoke_grid);
        if (!status) return status;
    }
    if (debug_enabled && impl_->rigid_body_count != 0U) {
        capture_rigid_inputs_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->accumulators, impl_->debug_applied_forces,
            impl_->debug_applied_torques, impl_->rigid_body_count);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return cuda_failure(error,
                                "rigid debug input capture launch failed");
    }
    const bool has_cloth = std::any_of(impl_->cloths.begin(), impl_->cloths.end(),
        [](const auto &cloth) { return cloth && cloth->alive; });
    const bool has_soft_body = std::any_of(
        impl_->soft_bodies.begin(), impl_->soft_bodies.end(),
        [](const auto &body) { return body && body->alive; });
    const bool has_rope_soft_body = std::any_of(
        impl_->rope_soft_body_couplings.begin(),
        impl_->rope_soft_body_couplings.end(),
        [](const auto &coupling) {
            return coupling && coupling->alive && coupling->options.enabled;
        });
    const bool has_smoke = std::any_of(impl_->smokes.begin(), impl_->smokes.end(),
        [](const auto &smoke) { return smoke && smoke->alive; });
    bool any_moving_body = false;
    if (impl_->fluid_count != 0U || has_smoke) {
        for (std::uint32_t body = 0U; body < impl_->rigid_body_count; ++body)
            any_moving_body |= impl_->parameters[body].motion !=
                               MotionType::static_body;
        if (any_moving_body) {
            error = cudaMemcpyAsync(impl_->fluid_previous_states,
                impl_->states[impl_->current_state],
                impl_->rigid_body_count * sizeof(RigidBodyState),
                cudaMemcpyDeviceToDevice, stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid body state copy failed");
        }
    }
    const float substep_timestep =
        options.timestep / static_cast<float>(options.substeps);
    impl_->rigid_solve_kernels_per_substep = 6U;
    const auto coupled_cloth = [&](std::uint32_t index) {
        return std::any_of(impl_->soft_cloth_couplings.begin(),
            impl_->soft_cloth_couplings.end(), [&](const auto &coupling) {
                return coupling && coupling->alive && coupling->options.enabled &&
                       coupling->options.cloth.index == index;
            });
    };
    const auto coupled_soft = [&](std::uint32_t index) {
        return std::any_of(impl_->soft_cloth_couplings.begin(),
            impl_->soft_cloth_couplings.end(), [&](const auto &coupling) {
                return coupling && coupling->alive && coupling->options.enabled &&
                       coupling->options.soft_body.index == index;
            });
    };
    const auto advance_cloth = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        for (std::uint32_t cloth_index = 0U;
             cloth_index < impl_->cloths.size(); ++cloth_index) {
            const auto &cloth_pointer = impl_->cloths[cloth_index];
            if (!cloth_pointer || !cloth_pointer->alive) continue;
            ClothStorage &cloth = *cloth_pointer;
            const std::uint32_t blocks =
                (cloth.vertex_count + block_size - 1U) / block_size;
            for (const auto &owner : impl_->smoke_cloth_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    owner->options.cloth.index != cloth_index ||
                    owner->options.wind_drag == 0.0F) continue;
                const auto &smoke = *impl_->smokes[owner->options.smoke.index];
                if (smoke.count == 0U) continue;
                if (smoke.grid.resolution != 0U) {
                    error = cudaMemsetAsync(cloth.smoke_forces, 0,
                        cloth.vertex_count * sizeof(Vec3), stream);
                    if (error != cudaSuccess)
                        return cuda_failure(error,
                            "cloth smoke force clear failed");
                    const auto triangles = cloth.index_count / 3U;
                    smoke_grid_cloth_force<<<
                        (triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions,
                        cloth.velocities, cloth.inverse_masses, cloth.indices,
                        cloth.index_count,
                        {smoke.grid.resolution, smoke.grid.height,
                         smoke.grid.minimum, smoke.grid.spacing,
                         smoke.grid.velocity, smoke.grid.density,
                         smoke.grid.pressure[0],
                         {smoke.grid.face_velocity[0][0],
                          smoke.grid.face_velocity[1][0],
                          smoke.grid.face_velocity[2][0]},
                        smoke.grid.vorticity, smoke.grid.strain},
                        owner->options.wind_drag,
                        smoke.options.rest_number_density,
                        smoke.options.grid_kinematic_viscosity,
                        smoke.options.grid_les_coefficient,
                        cloth.smoke_forces);
                    smoke_grid_apply_cloth_force<<<blocks, block_size, 0,
                        stream>>>(cloth.velocities, cloth.inverse_masses,
                        cloth.smoke_forces, cloth.vertex_count,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep);
                } else {
                    smoke_cloth_wind<<<blocks, block_size, 0, stream>>>(
                        cloth.positions, cloth.velocities, cloth.inverse_masses,
                        cloth.vertex_count, smoke.options,
                        smoke.positions, smoke.velocities,
                        smoke.keys[1], smoke.indices[1], {},
                        owner->options.wind_drag,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep);
                }
            }
            deformable_predict<<<blocks, block_size, 0, stream>>>(
                cloth.positions, cloth.previous, cloth.velocities,
                cloth.inverse_masses, cloth.vertex_count, options.gravity,
                substep_timestep, cloth.velocity_damping);
            Status cloth_status = record_timing_stage(
                TimingStage::cloth_prediction);
            if (!cloth_status) return cloth_status;
            for (std::uint32_t iteration = 0U;
                 iteration < cloth.solver_iterations; ++iteration) {
                deformable_project_links<<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.scratch, cloth.inverse_masses,
                    cloth.offsets, cloth.neighbors, cloth.bond_active,
                    cloth.vertex_count, substep_timestep, 0.0F);
                std::swap(cloth.positions, cloth.scratch);
                if (cloth.preserve_volume) {
                    cloth_project_volume<<<1U, k_cloth_volume_threads, 0, stream>>>(
                        cloth.positions, cloth.inverse_masses, cloth.indices,
                        cloth.vertex_count, cloth.index_count / 3U,
                        cloth.volume_corner_offsets, cloth.volume_corner_indices,
                        cloth.volume_gradients, cloth.target_volume,
                        cloth.orientation, cloth.volume_compliance,
                        substep_timestep, cloth.volume_lambda,
                        iteration == 0U);
                }
                cloth_status = record_timing_stage(
                    TimingStage::cloth_constraints);
                if (!cloth_status) return cloth_status;
            }
            if (impl_->rigid_body_count != 0U) {
                const cudaError_t clear_error = cudaMemsetAsync(
                    impl_->fluid_body_contact_flags, 0,
                    impl_->rigid_body_count * sizeof(std::uint32_t), stream);
                if (clear_error != cudaSuccess)
                    return cuda_failure(clear_error,
                                        "cloth contact flags clear failed");
                deformable_collide<false><<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.previous,
                    cloth.inverse_masses, cloth.vertex_count,
                    cloth.thickness, substep_timestep,
                    impl_->parameters, previous_states,
                    impl_->states[impl_->current_state],
                    impl_->meshes, impl_->rigid_body_count,
                    cloth.body_impulses, cloth.rigid_contact_forces,
                    nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr,
                    impl_->fluid_body_contact_flags,
                    false, true, 20.0F);
                reduce_point_body_impulses<<<impl_->rigid_body_count,
                                             block_size, 0, stream>>>(
                    cloth.body_impulses, cloth.count, impl_->parameters,
                    impl_->states[impl_->current_state],
                    impl_->fluid_body_contact_flags, impl_->rigid_body_count);
                if (cloth.surface_positions != nullptr) {
                    const std::uint32_t triangles = cloth.index_count / 3U;
                    cloth_update_surface<<<
                        (triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                }
                cloth_constrain_bodies<<<block_count, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.indices, cloth.vertex_sources,
                    cloth.surface_positions,
                    cloth.inverse_masses,
                    cloth.vertex_count, cloth.index_count / 3U,
                    cloth.thickness, substep_timestep,
                    cloth.contact_friction, impl_->parameters,
                    previous_states, impl_->states[impl_->current_state],
                    impl_->meshes, impl_->ids, impl_->rigid_body_count,
                    {cloth_index, cloth.generation}, impl_->paint_fields,
                    impl_->options.paint_field_capacity, impl_->paint_rules,
                    impl_->options.paint_rule_capacity,
                    cloth.body_corrections);
                cloth_apply_body_corrections<<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.inverse_masses,
                    cloth.vertex_count,
                    cloth.body_corrections, impl_->rigid_body_count);
            } else {
                deformable_collide<false><<<blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.velocities, cloth.previous,
                    cloth.inverse_masses, cloth.vertex_count,
                    cloth.thickness, substep_timestep,
                    impl_->parameters, previous_states,
                    impl_->states[impl_->current_state],
                    impl_->meshes, 0U, cloth.body_impulses,
                    cloth.rigid_contact_forces, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr,
                    impl_->fluid_body_contact_flags, false, true, 20.0F);
            }
            if (cloth.free_triangle_nodes)
                cloth_limit_strain<<<1U, 128U, 0, stream>>>(cloth.positions,
                    cloth.scratch, cloth.inverse_masses, cloth.offsets,
                    cloth.neighbors, cloth.free_triangle_nodes, cloth.vertex_count);
            cloth_status = record_timing_stage(TimingStage::cloth_contacts);
            if (!cloth_status) return cloth_status;
            if (cloth.surface_positions != nullptr) {
                const std::uint32_t triangles = cloth.index_count / 3U;
                if (!coupled_cloth(cloth_index)) cloth_break_bonds<<<
                    (cloth.bond_count + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions, cloth.bonds,
                    cloth.bond_active, cloth.bond_damage, cloth.bond_count,
                    cloth.break_strain, cloth.fracture_persistence_substeps,
                    cloth.body_impulses, cloth.impact_break_impulse);
                cloth_update_surface<<<
                    (triangles + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions,
                    cloth.indices,
                    cloth.surface_positions, triangles);
            }
        }
        const cudaError_t cloth_error = cudaPeekAtLastError();
        if (cloth_error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(cloth_error, "cloth kernel launch failed");
        }
        return success();
    };
    const auto advance_soft_bodies = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        for (const auto &body_pointer : impl_->soft_bodies) {
            if (!body_pointer || !body_pointer->alive) continue;
            SoftBodyStorage &body = *body_pointer;
            const std::uint32_t blocks =
                (body.node_count + block_size - 1U) / block_size;
            bool grid_snapshot_ready = false;
            for (const auto &owner : impl_->smoke_soft_body_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    owner->options.soft_body.index >= impl_->soft_bodies.size() ||
                    impl_->soft_bodies[owner->options.soft_body.index].get() != &body ||
                    owner->options.wind_drag == 0.0F) continue;
                const auto &smoke = *impl_->smokes[owner->options.smoke.index];
                if (smoke.count == 0U) continue;
                if (smoke.grid.resolution != 0U) {
                    if (!grid_snapshot_ready) {
                        const cudaError_t copy_error = cudaMemcpyAsync(
                            body.velocity_scratch, body.velocities,
                            body.node_count * sizeof(Vec3),
                            cudaMemcpyDeviceToDevice, stream);
                        if (copy_error != cudaSuccess)
                            return cuda_failure(copy_error,
                                "soft-body air velocity snapshot failed");
                        grid_snapshot_ready = true;
                    }
                    const auto triangles = body.surface_index_count / 3U;
                    smoke_grid_soft_body_force<<<
                        (triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.surface_positions,
                        body.surface_indices, body.surface_index_count,
                        body.surface_bindings, body.velocity_scratch,
                        body.velocities,
                        body.inverse_masses,
                        {smoke.grid.resolution, smoke.grid.height,
                         smoke.grid.minimum,
                         smoke.grid.spacing, smoke.grid.velocity,
                         smoke.grid.density, smoke.grid.pressure[0],
                         {smoke.grid.face_velocity[0][0],
                          smoke.grid.face_velocity[1][0],
                          smoke.grid.face_velocity[2][0]},
                         smoke.grid.vorticity, smoke.grid.strain},
                        owner->options.wind_drag,
                        smoke.options.rest_number_density,
                        smoke.options.grid_kinematic_viscosity,
                        smoke.options.grid_les_coefficient,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep);
                } else {
                    smoke_soft_body_wind<<<blocks, block_size, 0, stream>>>(
                        body.positions, body.velocities, body.inverse_masses,
                        body.node_count, smoke.options,
                        smoke.positions, smoke.velocities,
                        smoke.keys[1], smoke.indices[1], {},
                        owner->options.wind_drag,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep, body.maximum_speed);
                }
            }
            deformable_predict<<<blocks, block_size, 0, stream>>>(
                body.positions, body.previous, body.velocities,
                body.inverse_masses, body.node_count, options.gravity,
                substep_timestep, body.velocity_damping);
            cudaError_t friction_clear_error = cudaMemsetAsync(
                body.contact_momentum_delta, 0,
                body.node_count * sizeof(Vec3), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body momentum accumulator clear failed");
            friction_clear_error = cudaMemsetAsync(
                body.contact_friction_delta, 0,
                body.node_count * sizeof(Vec3), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body friction accumulator clear failed");
            friction_clear_error = cudaMemsetAsync(
                body.contact_normal_delta, 0,
                body.node_count * sizeof(float), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body normal accumulator clear failed");
            friction_clear_error = cudaMemsetAsync(
                body.dynamic_contact_flag, 0,
                sizeof(std::uint32_t), stream);
            if (friction_clear_error != cudaSuccess)
                return cuda_failure(friction_clear_error,
                    "soft-body dynamic contact flag clear failed");
            soft_body_measure_momentum<<<1U, block_size, 0, stream>>>(
                body.velocities, body.inverse_masses, body.node_count,
                body.predicted_momentum);
            Status body_status = record_timing_stage(
                TimingStage::soft_body_prediction);
            if (!body_status) return body_status;
            const auto resolve_contacts = [&]() noexcept -> Status {
                cudaError_t clear_error = cudaMemsetAsync(
                    body.contact_count, 0, sizeof(std::uint32_t), stream);
                if (clear_error != cudaSuccess)
                    return cuda_failure(clear_error,
                        "soft-body contact count clear failed");
                if (impl_->rigid_body_count != 0U) {
                    clear_error = cudaMemsetAsync(
                        impl_->fluid_body_contact_flags, 0,
                        impl_->rigid_body_count * sizeof(std::uint32_t),
                        stream);
                    if (clear_error != cudaSuccess)
                        return cuda_failure(clear_error,
                            "soft-body contact flags clear failed");
                }
                deformable_collide<true><<<blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.previous,
                    body.inverse_masses, body.node_count, body.node_radius,
                    substep_timestep, impl_->parameters, previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->rigid_body_count, body.body_impulses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms,
                    body.contact_momentum_delta,
                    body.contact_normal_delta,
                    body.contact_count, body.dynamic_contact_flag,
                    impl_->fluid_body_contact_flags,
                    false, false, body.maximum_speed,
                    body.body_position_corrections);
                return success();
            };
            const auto apply_contact_traction = [&]() noexcept {
                soft_body_apply_contact_friction<<<
                    blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.inverse_masses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.body_impulses,
                    impl_->parameters,
                    body.contact_momentum_delta,
                    body.contact_friction_delta,
                    body.contact_normal_delta,
                    body.contact_count, body.node_count, body.movable_mass,
                    options.gravity, substep_timestep, body.contact_friction,
                    body.maximum_speed);
            };
            const auto finish_contact_pass = [&]() noexcept -> Status {
                apply_contact_traction();
                if (impl_->rigid_body_count != 0U)
                    reduce_point_body_impulses<<<
                        impl_->rigid_body_count, block_size, 0, stream>>>(
                        body.body_impulses, body.count, impl_->parameters,
                        impl_->states[impl_->current_state],
                        impl_->fluid_body_contact_flags,
                        impl_->rigid_body_count,
                        body.body_position_corrections);
                return record_timing_stage(TimingStage::soft_body_contacts);
            };
            // Contact and graph constraints form one position solve. Revisit
            // the triangle boundary after every two graph passes so spring
            // projection cannot strand nodes across a wall, while the next
            // passes distribute each contact correction through the volume.
            body_status = resolve_contacts();
            if (!body_status) return body_status;
            body_status = finish_contact_pass();
            if (!body_status) return body_status;
            for (std::uint32_t iteration = 0U;
                 iteration < body.solver_iterations; ++iteration) {
                if(body.warp_neighbors) {
                    constexpr std::uint32_t groups_per_block = 8U;
                    const std::uint32_t group_blocks =
                        (body.node_count + groups_per_block - 1U) /
                        groups_per_block;
                    deformable_project_links_warp<<<
                        group_blocks, block_size, 0, stream>>>(
                        body.positions, body.scratch, body.inverse_masses,
                        body.offsets, body.warp_neighbors,
                        body.minimum_rest_lengths, body.node_count,
                        body.stretch_compliance, substep_timestep,
                        body.maximum_projection_fraction);
                } else
                    deformable_project_links<<<blocks, block_size, 0, stream>>>(
                        body.positions, body.scratch, body.inverse_masses,
                        body.offsets, body.neighbors, body.bond_active,
                        body.node_count, substep_timestep,
                        body.maximum_projection_fraction);
                std::swap(body.positions, body.scratch);
                if (iteration + 1U == body.solver_iterations &&
                    body.shape_matching_stiffness > 0.0F) {
                    soft_body_project_rest_shape<<<1U, 128U, 0, stream>>>(
                        body.positions, body.scratch, body.rest_positions,
                        body.inverse_masses, body.node_count,
                        body.movable_mass, body.shape_rest_center,
                        body.shape_inverse_rest, body.shape_orientation,
                        body.shape_matching_stiffness,
                        body.shape_maximum_projection,
                        body.dynamic_contact_flag);
                }
                body_status = record_timing_stage(
                    TimingStage::soft_body_constraints);
                if (!body_status) return body_status;
                if ((iteration + 1U) % 2U == 0U ||
                    iteration + 1U == body.solver_iterations) {
                    body_status = resolve_contacts();
                    if (!body_status) return body_status;
                    if (iteration + 1U != body.solver_iterations) {
                        body_status = finish_contact_pass();
                        if (!body_status) return body_status;
                    }
                }
            }
            soft_body_finalize_velocities<<<blocks, block_size, 0, stream>>>(
                body.positions, body.previous, body.velocities,
                body.inverse_masses, body.node_count,
                1.0F / substep_timestep,
                body.constraint_velocity_response, body.maximum_speed);
            if (body.warp_neighbors) {
                constexpr std::uint32_t groups_per_block = 8U;
                const std::uint32_t group_blocks =
                    (body.node_count + groups_per_block - 1U) /
                    groups_per_block;
                soft_body_damp_springs_warp<<<
                    group_blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.velocity_scratch,
                    body.inverse_masses, body.offsets, body.warp_neighbors,
                    body.node_count, body.spring_damping,
                    body.maximum_speed);
            } else {
                soft_body_damp_springs<<<blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.velocity_scratch,
                    body.inverse_masses, body.offsets, body.neighbors,
                    body.bond_active, body.node_count, body.spring_damping,
                    body.maximum_speed);
            }
            std::swap(body.velocities, body.velocity_scratch);
            body_status = finish_contact_pass();
            if (!body_status) return body_status;
            soft_body_restore_momentum<<<1U, block_size, 0, stream>>>(
                body.velocities, body.inverse_masses,
                body.contact_momentum_delta, body.node_count,
                body.movable_mass, body.predicted_momentum,
                body.dynamic_contact_flag,
                body.maximum_speed);
            // Traction changes positions too. Finish with nonpenetration so
            // neither friction nor a competing collider becomes next step's
            // already-invalid sweep origin. This pass adds no second impulse.
            // Recovery enforces geometry with a small skin. The full node
            // radius is the normal solver's contact target, but overlapping
            // safety margins in a pinch must not push a node through a solid.
            for (std::uint32_t pass = 0U;
                 impl_->rigid_body_count != 0U &&
                 pass < k_soft_contact_cleanup_passes; ++pass) {
                if (pass % 2U == 0U) {
                    soft_body_update_surface<<<
                        (body.surface_vertex_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.positions, body.rest_positions,
                        body.surface_rest_positions, body.surface_bindings,
                        body.surface_positions, body.surface_vertex_count);
                    const std::uint32_t triangles = body.surface_index_count / 3U;
                    soft_body_surface_contacts<<<
                        (triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.surface_positions,
                        body.surface_indices, triangles, impl_->parameters,
                        impl_->states[impl_->current_state], impl_->meshes,
                        impl_->rigid_body_count, body.surface_corner_corrections);
                    soft_body_apply_surface_contacts<<<blocks, block_size, 0, stream>>>(
                        body.positions, body.surface_corner_corrections,
                        body.surface_node_offsets, body.surface_node_influences,
                        body.node_count);
                }
                deformable_collide<true, true><<<blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.previous,
                    body.inverse_masses, body.node_count,
                    body.node_radius * 0.01F,
                    substep_timestep, impl_->parameters, previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->rigid_body_count, body.body_impulses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.contact_momentum_delta,
                    body.contact_normal_delta, body.contact_count,
                    body.dynamic_contact_flag, impl_->fluid_body_contact_flags,
                    pass + 1U == k_soft_contact_cleanup_passes,
                    false, body.maximum_speed);
            }
            const std::uint32_t surface_blocks =
                (body.surface_vertex_count + block_size - 1U) / block_size;
            soft_body_update_surface<<<surface_blocks, block_size, 0, stream>>>(
                body.positions, body.rest_positions,
                body.surface_rest_positions, body.surface_bindings,
                body.surface_positions, body.surface_vertex_count);
            body_status = record_timing_stage(TimingStage::soft_body_contact_cleanup);
            if (!body_status) return body_status;
        }
        const cudaError_t body_error = cudaPeekAtLastError();
        if (body_error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(body_error, "soft-body kernel launch failed");
        }
        return success();
    };
    const auto advance_soft_cloth = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        std::uint32_t iterations = 0U;
        for (const auto &coupling : impl_->soft_cloth_couplings)
            if (coupling && coupling->alive && coupling->options.enabled)
                iterations = std::max(iterations, coupling->options.solver_iterations);
        for (const auto &body : impl_->soft_bodies) if (body && body->alive) {
            const auto clear = cudaMemsetAsync(body->cloth_forces, 0,
                body->node_count * sizeof(Vec3), stream);
            if (clear != cudaSuccess) return cuda_failure(clear, "soft-cloth force clear failed");
        }
        for (const auto &cloth : impl_->cloths) if (cloth && cloth->alive) {
            const auto clear = cudaMemsetAsync(cloth->soft_body_forces, 0,
                cloth->vertex_count * sizeof(Vec3), stream);
            if (clear != cudaSuccess) return cuda_failure(clear, "cloth-soft force clear failed");
        }
        if (iterations == 0U) return success();
        impl_->soft_cloth_kernels_per_substep = 0U;
        for (std::uint32_t pass = 0U; pass < iterations; ++pass) {
            for (const auto &owner : impl_->soft_cloth_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled) continue;
                auto &coupling = *owner;
                auto &body = *impl_->soft_bodies[coupling.options.soft_body.index];
                auto &cloth = *impl_->cloths[coupling.options.cloth.index];
                const auto clear = cudaMemsetAsync(coupling.cloth_contact_counts, 0,
                    cloth.vertex_count * sizeof(std::uint32_t), stream);
                if (clear != cudaSuccess) return cuda_failure(clear, "soft-cloth count clear failed");
                const float distance = coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance : body.node_radius + cloth.thickness;
                const auto blocks = (body.node_count + block_size - 1U) / block_size;
                if (cloth.surface_positions != nullptr) {
                    const auto triangles = cloth.index_count / 3U;
                    cloth_update_surface<<<(triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
                soft_cloth_detect<<<blocks, block_size, 0, stream>>>(
                    body.positions, body.previous, body.velocities, body.inverse_masses,
                    body.node_count, cloth.positions, cloth.previous, cloth.velocities,
                    cloth.inverse_masses, cloth.indices, cloth.surface_positions,
                    cloth.index_count / 3U, distance, coupling.options.friction,
                    coupling.contacts, coupling.cloth_contact_counts);
                soft_cloth_apply_soft<<<blocks, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.cloth_forces, body.inverse_masses,
                    body.node_count, coupling.contacts, coupling.cloth_contact_counts,
                    substep_timestep, body.maximum_speed);
                soft_cloth_apply_cloth<<<
                    (cloth.vertex_count + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions, cloth.velocities,
                    cloth.soft_body_forces, cloth.inverse_masses, cloth.vertex_count,
                    coupling.contacts, body.node_count, coupling.cloth_contact_counts,
                    substep_timestep);
                impl_->soft_cloth_kernels_per_substep += 3U;
            }
            for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
                if (!coupled_cloth(index)) continue;
                auto &cloth = *impl_->cloths[index];
                // Sample once per substep, before projection erases impact strain.
                if (pass == 0U && cloth.surface_positions != nullptr) {
                    cloth_break_bonds<<<(cloth.bond_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions, cloth.bonds,
                        cloth.bond_active, cloth.bond_damage, cloth.bond_count,
                        cloth.break_strain, cloth.fracture_persistence_substeps,
                        cloth.body_impulses, cloth.impact_break_impulse);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
                if (pass + 1U < iterations) {
                    deformable_project_links<<<
                        (cloth.vertex_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions, cloth.scratch,
                        cloth.inverse_masses, cloth.offsets, cloth.neighbors,
                        cloth.bond_active, cloth.vertex_count, substep_timestep, 0.0F);
                    std::swap(cloth.positions, cloth.scratch);
                    ++impl_->soft_cloth_kernels_per_substep;
                    if (cloth.preserve_volume) {
                        cloth_project_volume<<<1U, k_cloth_volume_threads, 0, stream>>>(
                            cloth.positions, cloth.inverse_masses, cloth.indices,
                            cloth.vertex_count, cloth.index_count / 3U,
                            cloth.volume_corner_offsets, cloth.volume_corner_indices,
                            cloth.volume_gradients, cloth.target_volume,
                            cloth.orientation, cloth.volume_compliance,
                            substep_timestep, cloth.volume_lambda, false);
                        ++impl_->soft_cloth_kernels_per_substep;
                    }
                }
                if (cloth.surface_positions != nullptr) {
                    const auto triangles = cloth.index_count / 3U;
                    cloth_limit_strain<<<1U, 128U, 0, stream>>>(cloth.positions,
                        cloth.scratch, cloth.inverse_masses, cloth.offsets,
                        cloth.neighbors, cloth.free_triangle_nodes, cloth.vertex_count);
                    ++impl_->soft_cloth_kernels_per_substep;
                    cloth_update_surface<<<(triangles + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
            }
            if (pass + 1U < iterations)
                for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
                    if (!coupled_soft(index)) continue;
                    auto &body = *impl_->soft_bodies[index];
                    deformable_project_links<<<
                        (body.node_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.positions, body.scratch,
                        body.inverse_masses, body.offsets, body.neighbors,
                        body.bond_active, body.node_count, substep_timestep,
                        body.maximum_projection_fraction);
                    std::swap(body.positions, body.scratch);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
        }
        for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
            if (!coupled_cloth(index) || impl_->rigid_body_count == 0U) continue;
            auto &cloth = *impl_->cloths[index];
            deformable_collide<false, true><<<
                (cloth.vertex_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
                cloth.positions, cloth.velocities, cloth.previous, cloth.inverse_masses,
                cloth.vertex_count, cloth.thickness, substep_timestep,
                impl_->parameters, previous_states, impl_->states[impl_->current_state],
                impl_->meshes, impl_->rigid_body_count, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true, false, 20.0F);
            ++impl_->soft_cloth_kernels_per_substep;
            if (cloth.surface_positions != nullptr) {
                const auto triangles = cloth.index_count / 3U;
                cloth_update_surface<<<(triangles + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(cloth.positions,
                    cloth.indices,
                    cloth.surface_positions, triangles);
                ++impl_->soft_cloth_kernels_per_substep;
            }
        }
        for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
            if (!coupled_soft(index)) continue;
            auto &body = *impl_->soft_bodies[index];
            if (impl_->rigid_body_count != 0U) {
                deformable_collide<true, true><<<
                    (body.node_count + block_size - 1U) / block_size, block_size, 0, stream>>>(
                    body.positions, body.velocities, body.previous, body.inverse_masses,
                    body.node_count, body.node_radius * 0.01F, substep_timestep,
                    impl_->parameters, previous_states, impl_->states[impl_->current_state],
                    impl_->meshes, impl_->rigid_body_count, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true, false,
                    body.maximum_speed);
                ++impl_->soft_cloth_kernels_per_substep;
            }
            soft_body_update_surface<<<
                (body.surface_vertex_count + block_size - 1U) / block_size,
                block_size, 0, stream>>>(body.positions, body.rest_positions,
                body.surface_rest_positions, body.surface_bindings,
                body.surface_positions, body.surface_vertex_count);
            ++impl_->soft_cloth_kernels_per_substep;
        }
        const auto launch_error = cudaPeekAtLastError();
        if (launch_error != cudaSuccess)
            return cuda_failure(launch_error, "soft-body cloth contact launch failed");
        return record_timing_stage(TimingStage::soft_body_cloth_contacts);
    };
    const auto advance_ropes = [&](const RigidBodyState *previous_states, bool first_substep) -> Status {
        if(first_substep)for(const auto &body:impl_->soft_bodies) {
            if(!body || !body->alive)continue;
            const auto clear=cudaMemsetAsync(body->rope_forces,0,
                body->node_count*sizeof(Vec3),stream);
            if(clear!=cudaSuccess)return cuda_failure(clear,"soft-body rope force clear failed");
        }
        if(first_substep)for(const auto &body:impl_->cloths) {
            if(!body || !body->alive || !body->rope_forces)continue;
            const auto clear=cudaMemsetAsync(body->rope_forces,0,
                body->vertex_count*sizeof(Vec3),stream);
            if(clear!=cudaSuccess)return cuda_failure(clear,"cloth rope force clear failed");
        }
        for (unsigned rope_index=0;rope_index<impl_->ropes.size();++rope_index) {
            const auto &rope=impl_->ropes[rope_index];
            if(!rope || !rope->alive)continue;
            RopeSoftTarget soft_targets[2]{};
            RopeSoftBodyCouplingStorage *couplings[2]{};
            unsigned coupled=0;
            for(const auto &owner:impl_->rope_soft_body_couplings) {
                if(!owner || !owner->alive || !owner->options.enabled ||
                   !(owner->options.rope==RopeId{rope_index,rope->generation}))continue;
                if(coupled>=2U)return failure(StatusCode::capacity_exceeded,
                    "a rope supports at most two soft-body contact targets");
                auto &item=*owner;
                auto &body=*impl_->soft_bodies[item.options.soft_body.index];
                const auto clear=cudaMemsetAsync(item.node_impulses,0,
                    body.node_count*sizeof(Vec3),stream);
                if(clear!=cudaSuccess)return cuda_failure(clear,"rope soft-body impulse clear failed");
                if(first_substep) {
                    auto e=cudaMemsetAsync(item.contact_count,0,sizeof(std::uint32_t),stream);
                    if(e!=cudaSuccess)return cuda_failure(e,"rope soft-body count clear failed");
                    e=cudaMemsetAsync(item.maximum_penetration,0,sizeof(float),stream);
                    if(e!=cudaSuccess)return cuda_failure(e,"rope soft-body depth clear failed");
                }
                auto refit_error=cudaMemsetAsync(item.ready,0,
                    item.tree_count*sizeof(std::uint32_t),stream);
                if(refit_error!=cudaSuccess)
                    return cuda_failure(refit_error,"rope soft-body BVH clear failed");
                fluid_soft_refit<<<(item.tree_count+block_size-1U)/block_size,
                    block_size,0,stream>>>(body.surface_positions,
                    item.previous_surface,body.surface_indices,item.order,
                    item.tree,item.parents,item.ready,item.tree_count,item.bounds);
                for(unsigned end=0;end<2;++end)if(item.anchor_triangle[end]!=~0U)
                    rope_soft_sample_anchor<<<1,1,0,stream>>>(rope->data,end,
                        body.surface_positions,body.velocities,body.surface_indices,
                        body.surface_bindings,item.anchor_triangle[end],
                        item.anchor_weights[end],item.anchor_offset[end]);
                soft_targets[coupled]={body.surface_positions,item.previous_surface,
                    body.velocities,item.bounds,
                    item.tree,item.order,
                    body.surface_indices,body.surface_bindings,body.inverse_masses,
                    item.node_impulses,item.contact_count,item.maximum_penetration,
                    body.surface_index_count/3U,body.surface_vertex_count,
                    item.options.contact_distance>0?item.options.contact_distance:rope->data.options.radius,
                    item.options.friction,item.orientation,
                    item.options.attach_first,item.options.attach_last};
                couplings[coupled++]=&item;
            }
            for(const auto &item:impl_->rope_cloth_couplings) {
                if(!item.alive || !item.options.enabled ||
                   !(item.options.rope==RopeId{rope_index,rope->generation}))continue;
                auto &cloth=*impl_->cloths[item.options.cloth.index];
                for(unsigned end=0;end<2;++end) {
                    const unsigned vertex=end?item.options.last_vertex:
                        item.options.first_vertex;
                    if(vertex==UINT32_MAX)continue;
                    rope_cloth_sample_anchor<<<1,1,0,stream>>>(rope->data,end,
                        cloth.positions,cloth.velocities,cloth.inverse_masses,
                        vertex,item.options.anchor_effective_mass);
                }
            }
            int first=-1,last=-1;unsigned dense;
            if(rope->data.options.first.enabled) {
                auto status=impl_->validate_handle(rope->data.options.first.body,dense);
                if(!status)return status;first=int(dense);
            }
            if(rope->data.options.last.enabled) {
                auto status=impl_->validate_handle(rope->data.options.last.body,dense);
                if(!status)return status;last=int(dense);
            }
            for (const auto &owner : impl_->smoke_rope_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    !(owner->options.rope == RopeId{rope_index, rope->generation}) ||
                    owner->options.wind_drag == 0.0F) continue;
                const auto &smoke = *impl_->smokes[owner->options.smoke.index];
                if (smoke.count == 0U) continue;
                smoke_rope_wind<<<
                    (rope->data.count + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(rope->data, smoke.options,
                    smoke.positions, smoke.velocities,
                    smoke.keys[1], smoke.indices[1],
                    {smoke.grid.resolution, smoke.grid.height,
                     smoke.grid.minimum, smoke.grid.spacing,
                     smoke.grid.velocity, smoke.grid.density,
                     smoke.grid.pressure[0],
                     {smoke.grid.face_velocity[0][0],
                      smoke.grid.face_velocity[1][0],
                      smoke.grid.face_velocity[2][0]},
                     smoke.grid.vorticity, smoke.grid.strain},
                    owner->options.wind_drag,
                    owner->options.maximum_wind_acceleration,
                    substep_timestep, first, last);
            }
            rope_advance<<<1,128,0,stream>>>(rope->data,substep_timestep,options.gravity,first,last,
                impl_->parameters,impl_->states[impl_->current_state],previous_states,
                impl_->meshes,impl_->rigid_body_count,first_substep,
                soft_targets[0],soft_targets[1]);
            auto error=cudaPeekAtLastError();
            if(error!=cudaSuccess)return cuda_failure(error,"rope solver launch failed");
            for(const auto &item:impl_->rope_cloth_couplings) {
                if(!item.alive || !item.options.enabled ||
                   !(item.options.rope==RopeId{rope_index,rope->generation}))continue;
                auto &cloth=*impl_->cloths[item.options.cloth.index];
                for(unsigned end=0;end<2;++end) {
                    const unsigned vertex=end?item.options.last_vertex:
                        item.options.first_vertex;
                    if(vertex==UINT32_MAX)continue;
                    rope_cloth_apply_anchor<<<1,1,0,stream>>>(rope->data,end,
                        cloth.positions,cloth.velocities,cloth.rope_forces,
                        cloth.inverse_masses,vertex,
                        item.options.maximum_cloth_acceleration,
                        substep_timestep,1.0F/options.timestep);
                }
                if(cloth.surface_positions) {
                    const unsigned triangles=cloth.index_count/3U;
                    cloth_update_surface<<<(triangles+block_size-1U)/block_size,
                        block_size,0,stream>>>(cloth.positions,cloth.indices,
                        cloth.surface_positions,triangles);
                }
            }
            auto status=record_timing_stage(TimingStage::rope_solve);
            if(!status)return status;
            for(unsigned index=0;index<coupled;++index) {
                auto &item=*couplings[index];
                auto &body=*impl_->soft_bodies[item.options.soft_body.index];
                for(unsigned end=0;end<2;++end)if(item.anchor_triangle[end]!=~0U)
                    rope_soft_scatter_anchor<<<1,256,0,stream>>>(rope->data,end,
                        body.positions,body.node_count,
                        item.options.anchor_support_radius_scale*body.node_radius,
                        item.options.anchor_contact_support_radius_scale*
                            body.node_radius,item.contact_count,
                        body.surface_indices,body.surface_bindings,
                        item.anchor_triangle[end],item.anchor_weights[end],
                        item.node_impulses);
                rope_soft_apply<<<(body.node_count+block_size-1U)/block_size,
                    block_size,0,stream>>>(body.positions,body.velocities,
                    body.rope_forces,body.inverse_masses,item.node_impulses,
                    body.node_count,item.options.maximum_soft_body_acceleration,
                    body.maximum_speed,substep_timestep,1.0F/options.timestep);
                soft_body_update_surface<<<(body.surface_vertex_count+block_size-1U)/block_size,
                    block_size,0,stream>>>(body.positions,body.rest_positions,
                    body.surface_rest_positions,body.surface_bindings,
                    body.surface_positions,body.surface_vertex_count);
                error=cudaMemcpyAsync(item.previous_surface,body.surface_positions,
                    body.surface_vertex_count*sizeof(Vec3),cudaMemcpyDeviceToDevice,stream);
                if(error!=cudaSuccess)
                    return cuda_failure(error,"rope soft-body skin snapshot failed");
                error=cudaPeekAtLastError();
                if(error!=cudaSuccess)return cuda_failure(error,"rope soft-body coupling launch failed");
                status=record_timing_stage(TimingStage::rope_soft_body_contacts,
                    2U+unsigned(item.anchor_triangle[0]!=~0U)+
                    unsigned(item.anchor_triangle[1]!=~0U));
                if(!status)return status;
            }
        }
        return success();
    };
    for (std::uint32_t substep = 0; substep < options.substeps; ++substep) {
        if (impl_->rigid_body_count == 0U) {
            break;
        }
        const std::uint32_t output_state = 1U - impl_->current_state;
        integrate_rigid_bodies_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->accumulators, impl_->targets,
            impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, options.gravity, substep_timestep,
            options.substeps - substep, substep == 0U);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error, "rigid integration kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_integration);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        const std::uint32_t pair_count =
            impl_->rigid_body_count * impl_->rigid_body_count;
        const std::uint32_t pair_block_count =
            (pair_count + block_size - 1U) / block_size;
        const std::uint32_t contact_block_count =
            std::min(pair_count, 128U);
        compute_rigid_world_bounds_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->states[impl_->current_state],
            impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes,
            impl_->rigid_world_bounds);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid world-bounds kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_world_bounds);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        if (impl_->rigid_constraint_count != 0U) {
            build_fixed_collision_groups_kernel<<<1U, 1U, 0, stream>>>(
                impl_->rigid_constraints, impl_->options.rigid_constraint_capacity,
                impl_->ids, impl_->rigid_body_count,
                impl_->fixed_contact_projection);
        }
        broad_phase_rigid_pairs_kernel<<<pair_block_count, block_size, 0,
                                         stream>>>(
            impl_->parameters, impl_->rigid_world_bounds,
            impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->meshes, impl_->ids, impl_->rigid_constraints,
            impl_->options.rigid_constraint_capacity,
            impl_->rigid_constraint_count != 0U
                ? impl_->fixed_contact_projection : nullptr,
            impl_->rigid_body_count, impl_->rigid_active_pair_flags);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid broad-phase kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_pair_filter,
                                    impl_->rigid_constraint_count != 0U ? 2U : 1U);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        const auto pair_indices =
            thrust::make_counting_iterator<std::uint32_t>(0U);
        error = cub::DeviceSelect::Flagged(
            impl_->rigid_broad_phase_workspace,
            impl_->rigid_broad_phase_workspace_size, pair_indices,
            impl_->rigid_active_pair_flags, impl_->rigid_active_pairs,
            impl_->rigid_active_pair_count, static_cast<int>(pair_count),
            stream);
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid broad-phase compaction failed");
        }
        status = record_timing_stage(TimingStage::rigid_pair_compaction);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        const std::uint32_t shared_leaf_capacity =
            impl_->rigid_leaf_manifolds != nullptr ? k_shared_rigid_leaf_capacity : 0U;
        generate_rigid_leaf_pairs_kernel<<<contact_block_count, block_size,
                                           shared_leaf_capacity * sizeof(WorldAabb),
                                           stream>>>(
            impl_->parameters, impl_->states[impl_->current_state],
            impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes,
            impl_->options.triangle_mesh_capacity,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->rigid_leaf_pairs,
            impl_->rigid_leaf_pair_counts,
            impl_->rigid_leaf_pair_slot_capacity,
            impl_->rigid_leaf_pairs_per_slot, shared_leaf_capacity);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid leaf-pair kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_leaf_pair_generation);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        // Many-body worlds parallelize across pairs. Few-body worlds spread
        // each dense pair across blocks and reduce in the same candidate order.
        // One shared reduction manifold accompanies the per-thread scratch.
        constexpr std::uint32_t contact_evaluation_threads =
            33U * sizeof(ContactManifold) <= 48U * 1024U ? 32U : 16U;
        const std::uint32_t blocks_per_pair =
            impl_->rigid_leaf_manifolds != nullptr ? k_rigid_leaf_blocks_per_pair : 1U;
        finalize_rigid_contact_manifolds_kernel<<<contact_block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes, impl_->ids, impl_->rigid_constraints,
            impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->rigid_leaf_pair_counts, substep_timestep, impl_->rigid_manifolds, true);
        evaluate_rigid_leaf_pairs_kernel<<<
            contact_block_count * blocks_per_pair, contact_evaluation_threads,
            contact_evaluation_threads * sizeof(ContactManifold),
            stream>>>(impl_->parameters,
                      impl_->states[impl_->current_state],
                      impl_->states[output_state],
                      impl_->rigid_body_count, impl_->meshes, impl_->ids,
                      impl_->rigid_constraints,
                      impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
                      impl_->rigid_active_pairs,
                      impl_->rigid_active_pair_count,
                      impl_->rigid_leaf_pairs, impl_->rigid_leaf_pair_counts,
                      impl_->rigid_leaf_pairs_per_slot,
                      impl_->rigid_leaf_manifolds, blocks_per_pair,
                      substep_timestep, impl_->rigid_manifolds);
        if (impl_->rigid_leaf_manifolds != nullptr) {
            reduce_rigid_leaf_manifolds_kernel<<<contact_block_count, 32U, 0, stream>>>(
                impl_->parameters, impl_->rigid_body_count,
                impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
                impl_->rigid_leaf_pair_counts, impl_->rigid_leaf_pairs_per_slot,
                impl_->rigid_leaf_manifolds, impl_->rigid_manifolds);
        }
        finalize_rigid_contact_manifolds_kernel<<<
            contact_block_count, block_size, 0, stream>>>(
                impl_->parameters, impl_->states[impl_->current_state],
                impl_->states[output_state], impl_->rigid_body_count,
                impl_->meshes, impl_->ids, impl_->rigid_constraints,
                impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
                impl_->rigid_active_pairs,
                impl_->rigid_active_pair_count,
                impl_->rigid_leaf_pair_counts, substep_timestep,
                impl_->rigid_manifolds, false);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid leaf contact kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_contact_evaluation);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        if (impl_->rigid_contact_revision != impl_->revision)
            impl_->rigid_contact_epoch += 2U;
        impl_->rigid_contact_revision = impl_->revision;
        ++impl_->rigid_contact_epoch;
        prepare_parallel_contact_events_kernel<<<1U, 1U, 0, stream>>>(
            impl_->rigid_body_count, impl_->rigid_manifolds,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->ids, impl_->rigid_contact_event_offsets,
            impl_->rigid_contact_events, impl_->rigid_contact_capacity,
            impl_->rigid_contact_count,
            collect_rigid_contacts, substep == 0U);
        load_rigid_contact_cache_kernel<<<contact_block_count, block_size, 0, stream>>>(
            impl_->rigid_manifolds, impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->states[output_state], impl_->ids, impl_->rigid_body_count,
            impl_->rigid_contact_cache, impl_->rigid_contact_cache_slots,
            impl_->rigid_leaf_pair_slot_capacity, impl_->rigid_contact_epoch, substep_timestep);
        const unsigned avbd_grid = std::min(impl_->rigid_contact_grid_limit,
            std::max(1U, (impl_->rigid_body_count + 31U) / 32U));
        prepare_avbd_kernel<<<1U, 128U, 0, stream>>>(
            impl_->parameters, impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, impl_->rigid_manifolds, impl_->rigid_active_pairs,
            impl_->rigid_active_pair_count, impl_->rigid_constraints,
            impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
            impl_->ids, impl_->avbd_bodies, impl_->rigid_contact_schedule,
            substep_timestep, options.gravity, options.rigid_contact_pass_limit, avbd_grid);
        const auto launch_avbd = [&](auto... arguments) {
            if (avbd_grid == 1U) {
                solve_avbd_kernel<<<1U, 32U, 0, stream>>>(arguments...);
                return cudaGetLastError();
            }
            void *kernel_arguments[]{static_cast<void *>(&arguments)...};
            return cudaLaunchCooperativeKernel(reinterpret_cast<void *>(solve_avbd_kernel),
                dim3(avbd_grid), dim3(32U), kernel_arguments, 0, stream);
        };
        error = launch_avbd(
            impl_->parameters, impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, impl_->rigid_manifolds, impl_->rigid_active_pairs,
            impl_->rigid_active_pair_count, impl_->rigid_constraints,
            impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
            impl_->avbd_bodies, impl_->rigid_contact_schedule, substep_timestep);
        if (error != cudaSuccess) return cuda_failure(error, "launch AVBD vertex solve");
        clamp_rigid_speeds_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->states[output_state],
            impl_->rigid_body_count);
        save_rigid_contact_cache_kernel<<<contact_block_count, block_size, 0, stream>>>(
            impl_->rigid_manifolds, impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->states[output_state], impl_->ids, impl_->rigid_body_count,
            impl_->rigid_contact_cache, impl_->rigid_contact_cache_slots,
            impl_->rigid_leaf_pair_slot_capacity, impl_->rigid_contact_epoch, substep_timestep,
            impl_->rigid_contact_event_offsets, impl_->rigid_contact_events,
            collect_rigid_contacts ? impl_->rigid_contact_capacity : 0U);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error,
                                "rigid contact solve kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_contact_solve);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
        impl_->current_state = output_state;
        // Couple the sheet to the rigid state from this same substep.
        if (has_cloth) {
            status = advance_cloth(impl_->states[1U - impl_->current_state]);
            if (!status) return status;
        }
        if (has_rope_soft_body) {
            status = advance_ropes(impl_->states[1U - impl_->current_state], substep == 0);
            if (!status) return status;
        }
        if (has_soft_body) {
            status = advance_soft_bodies(
                impl_->states[1U - impl_->current_state]);
            if (!status) return status;
        }
        status = advance_soft_cloth(impl_->states[1U - impl_->current_state]);
        if (!status) return status;
        if (!has_rope_soft_body) {
            status = advance_ropes(impl_->states[1U - impl_->current_state], substep == 0);
            if (!status) return status;
        }
    }

    if (impl_->rigid_body_count > 0U) {
        clear_rigid_inputs_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->accumulators, impl_->targets, impl_->rigid_body_count);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error, "rigid input-clear kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_input_clear);
        if (!status) {
            cudaStreamSynchronize(stream);
            return status;
        }
    }

    if (impl_->rigid_body_count == 0U) {
        for (std::uint32_t substep = 0U; substep < options.substeps; ++substep) {
            if (has_cloth) {
                status = advance_cloth(nullptr);
                if (!status) return status;
            }
            if (has_rope_soft_body) {
                status = advance_ropes(nullptr, substep == 0);
                if (!status) return status;
            }
            if (has_soft_body) {
                status = advance_soft_bodies(nullptr);
                if (!status) return status;
            }
            status = advance_soft_cloth(nullptr);
            if (!status) return status;
            if (!has_rope_soft_body) {
                status = advance_ropes(nullptr, substep == 0);
                if (!status) return status;
            }
        }
    }

    if (any_moving_body && impl_->fluid_count != 0U) {
        fluid_body_bounds_kernel<<<block_count, block_size, 0, stream>>>(
            impl_->parameters, impl_->fluid_previous_states,
            impl_->states[impl_->current_state], impl_->meshes,
            impl_->rigid_body_count, impl_->fluid_body_bounds);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return cuda_failure(error, "fluid body bounds launch failed");
    }

    // Fluid storage is capacity-sized and all sort/selection workspaces were
    // reserved at creation. No CUDA allocations occur during a frame.
    for (const auto &body : impl_->soft_bodies) {
        if (!body || !body->alive) continue;
        error = cudaMemsetAsync(body->fluid_forces, 0, body->node_count * sizeof(Vec3), stream);
        if (error != cudaSuccess) return cuda_failure(error, "soft-body fluid force clear failed");
    }
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        error = cudaMemsetAsync(coupling->contact_count, 0, sizeof(std::uint32_t), stream);
        if (error == cudaSuccess) error = cudaMemsetAsync(coupling->maximum_penetration, 0,
            sizeof(float), stream);
        if (error != cudaSuccess) return cuda_failure(error, "fluid soft-body diagnostics clear failed");
    }
    for (const auto &coupling : impl_->fluid_rope_couplings) {
        if (!coupling || !coupling->alive) continue;
        error = cudaMemsetAsync(coupling->contact_count, 0, sizeof(std::uint32_t), stream);
        if (error == cudaSuccess) error = cudaMemsetAsync(coupling->maximum_penetration, 0,
            sizeof(float), stream);
        if (error != cudaSuccess) return cuda_failure(error, "fluid rope diagnostics clear failed");
    }
    for (std::uint32_t fluid_index = 0U;
         fluid_index < impl_->fluids.size(); ++fluid_index) {
        if (!impl_->fluids[fluid_index] || !impl_->fluids[fluid_index]->alive)
            continue;
        FluidStorage &fluid = *impl_->fluids[fluid_index];
        const FluidId fluid_id{fluid_index, fluid.generation};
        std::uint32_t live = *fluid.count;
        const std::uint32_t first_spawned = live;
        bool spawned = false;
        const std::uint32_t blocks =
            (fluid.options.capacity + block_size - 1U) / block_size;
        float source_cell_size = fluid.options.support_radius;
        bool has_sources = false;
        for (const auto &slot : impl_->particle_sources) {
            if (!slot.alive || !slot.options.enabled || !(slot.options.fluid == fluid_id)) continue;
            has_sources = true;
            source_cell_size = std::max(source_cell_size, slot.data->spacing);
        }
        if (has_sources) {
            if (fluid.next_id > UINT32_MAX - (fluid.options.capacity - live))
                return failure(StatusCode::capacity_exceeded, "stable fluid particle ID range exhausted");
            fluid_emit_cells<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.count, fluid.options.capacity,
                1.0F/source_cell_size, fluid.keys[0], fluid.indices[0]);
            error = cub::DeviceRadixSort::SortPairs(
                fluid.sort_workspace, fluid.sort_workspace_size,
                fluid.keys[0], fluid.keys[1], fluid.indices[0], fluid.indices[1],
                fluid.options.capacity, 0, 64, stream);
            if (error != cudaSuccess) return cuda_failure(error, "fluid source spatial index failed");
        }
        for (ParticleSourceSlot &slot : impl_->particle_sources) {
            if (!slot.alive || !slot.options.enabled ||
                !(slot.options.fluid == fluid_id)) continue;
            const auto &data = *slot.data;
            fluid_source_vacancies<<<(data.count+block_size-1U)/block_size,block_size,0,stream>>>(
                data.points, data.count, data.spacing, fluid.positions,
                fluid.keys[1], fluid.indices[1], fluid.options.capacity,
                1.0F/source_cell_size, data.vacant);
            fluid_source_emit<<<1U,1U,0,stream>>>(
                data.points, data.vacant, data.count, data.spacing,
                slot.options.initial_velocity, fluid.positions, fluid.velocities,
                fluid.ids, fluid.foam, fluid.temperatures,
                slot.options.initial_temperature, fluid.count,
                fluid.options.capacity,
                first_spawned, fluid.next_id, data.capacity_misses);
        }
        if (has_sources) {
            // One readback per fluid, never one per sample. No frame allocation.
            error = cudaStreamSynchronize(stream);
            if (error != cudaSuccess) return cuda_failure(error, "fluid source emission failed");
            live = *fluid.count;
            const auto amount = live - first_spawned;
            spawned = amount != 0;
            fluid.next_id += amount;
            fluid.emitted_count += amount;
            impl_->emitted_particle_count += amount;
            for (const auto &slot : impl_->particle_sources)
                if (slot.alive && slot.options.enabled && slot.options.fluid == fluid_id)
                    impl_->spawn_capacity_miss_count += *slot.data->capacity_misses;
            status = record_timing_stage(TimingStage::fluid_spawn);
            if (!status) return status;
        }
        if (live == 0U) continue;
        const std::uint32_t iterations =
            options.substeps * fluid.options.solver_iterations;
        const float dt = options.timestep / static_cast<float>(iterations);
        const float diameter = 2.0F * fluid.options.particle_radius;
        const float particle_mass = fluid.options.rest_density *
            (fluid.options.rest_particle_volume > 0.0F
                ? fluid.options.rest_particle_volume
                : diameter * diameter * diameter);
        const std::uint32_t body_words =
            (impl_->options.rigid_body_capacity + 63U) / 64U;
        if (any_moving_body) {
            error = cudaMemsetAsync(impl_->fluid_body_masks, 0,
                k_fluid_body_buckets * body_words * sizeof(unsigned long long),
                stream);
            if (error == cudaSuccess)
                error = cudaMemsetAsync(impl_->fluid_global_body_masks, 0,
                    body_words * sizeof(unsigned long long), stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid body index clear failed");
            fluid_index_body_cells<<<block_count, block_size, 0, stream>>>(
                impl_->parameters, impl_->fluid_body_bounds,
                impl_->rigid_body_count, body_words,
                fluid.options.particle_radius + fluid.options.maximum_speed * dt,
                impl_->fluid_body_masks, impl_->fluid_global_body_masks);
            status = record_timing_stage(TimingStage::fluid_body_index);
            if (!status) return status;
        }
        if (collect_fluid_contacts) {
            error = cudaMemsetAsync(fluid.contact_flags, 0,
                fluid.options.capacity * sizeof(std::uint8_t), stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid event flags clear failed");
        }
        for (std::uint32_t iteration = 0U; iteration < iterations;
             ++iteration) {
            fluid_emit_cells<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.count, fluid.options.capacity,
                1.0F / fluid.options.support_radius,
                fluid.keys[0], fluid.indices[0]);
            error = cub::DeviceRadixSort::SortPairs(
                fluid.sort_workspace, fluid.sort_workspace_size,
                fluid.keys[0], fluid.keys[1], fluid.indices[0],
                fluid.indices[1], fluid.options.capacity, 0, 64, stream);
            if (error != cudaSuccess) {
                cudaStreamSynchronize(stream);
                return cuda_failure(error, "fluid neighbor cell sort failed");
            }
            status = record_timing_stage(TimingStage::fluid_neighbor_sort);
            if (!status) return status;
            fluid_compute_forces<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.velocities, fluid.count,
                fluid.keys[1], fluid.indices[1], fluid.foam,
                fluid.options, normalized_or(multiply(options.gravity, -1.0F),
                                             {0.0F, 1.0F, 0.0F}),
                fluid.forces, fluid.foam_source,
                impl_->fluid_neighbor_overflow,
                impl_->fluid_maximum_neighbor_count);
            status = record_timing_stage(TimingStage::fluid_neighbor_forces);
            if (!status) return status;
            for (FluidClothCouplingResource &coupling :
                 impl_->fluid_cloth_couplings) {
                if (!coupling.alive || !coupling.options.enabled ||
                    !(coupling.options.fluid == fluid_id)) continue;
                if (coupling.options.cloth.index >= impl_->cloths.size())
                    continue;
                const auto &cloth_pointer =
                    impl_->cloths[coupling.options.cloth.index];
                if (!cloth_pointer || !cloth_pointer->alive ||
                    cloth_pointer->generation !=
                        coupling.options.cloth.generation)
                    continue;
                ClothStorage &cloth = *cloth_pointer;
                FluidClothCouplingOptions resolved = coupling.options;
                if (resolved.contact_distance == 0.0F)
                    resolved.contact_distance =
                        fluid.options.particle_radius + cloth.thickness;
                if (resolved.interaction_radius == 0.0F)
                    resolved.interaction_radius = fluid.options.support_radius;
                error = cudaMemsetAsync(cloth.fluid_forces, 0,
                    cloth.vertex_count * sizeof(Vec3), stream);
                if (error != cudaSuccess)
                    return cuda_failure(error,
                        "fluid-cloth reaction clear failed");
                fluid_cloth_containment_forces<<<
                    blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.velocities, fluid.count,
                        fluid.forces, fluid.foam_source, particle_mass,
                        cloth.positions, cloth.velocities, cloth.indices,
                        cloth.index_count / 3U, cloth.orientation, resolved,
                        cloth.fluid_forces);
                const std::uint32_t cloth_blocks =
                    (cloth.vertex_count + block_size - 1U) / block_size;
                cloth_apply_fluid_forces<<<
                    cloth_blocks, block_size, 0, stream>>>(
                        cloth.positions, cloth.velocities,
                        cloth.inverse_masses, cloth.fluid_forces,
                        cloth.vertex_count, dt);
                deformable_project_links<<<cloth_blocks, block_size, 0, stream>>>(
                    cloth.positions, cloth.scratch, cloth.inverse_masses,
                    cloth.offsets, cloth.neighbors, cloth.bond_active,
                    cloth.vertex_count, dt, 0.0F);
                std::swap(cloth.positions, cloth.scratch);
                if (cloth.preserve_volume) {
                    cloth_project_volume<<<1U, k_cloth_volume_threads, 0, stream>>>(
                        cloth.positions, cloth.inverse_masses, cloth.indices,
                        cloth.vertex_count, cloth.index_count / 3U,
                        cloth.volume_corner_offsets, cloth.volume_corner_indices,
                        cloth.volume_gradients, cloth.target_volume,
                        cloth.orientation, cloth.volume_compliance, dt,
                        cloth.volume_lambda, true);
                }
                status = record_timing_stage(
                    TimingStage::fluid_cloth_contacts);
                if (!status) return status;
            }
            fluid_integrate<<<blocks, block_size, 0, stream>>>(
                fluid.positions, fluid.velocities, fluid.previous,
                fluid.foam, fluid.forces, fluid.foam_source,
                fluid.count, fluid.options,
                options.gravity, dt);
            status = record_timing_stage(TimingStage::fluid_integration);
            if (!status) return status;
            for (const auto &owner : impl_->fluid_rope_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    !(owner->options.fluid == fluid_id)) continue;
                auto &rope = impl_->ropes[owner->options.rope.index]->data;
                const float distance = owner->options.contact_distance > 0.0F
                    ? owner->options.contact_distance
                    : fluid.options.particle_radius + rope.options.radius;
                error = cudaMemsetAsync(owner->node_impulses, 0,
                    rope.count * sizeof(Vec3), stream);
                if (error != cudaSuccess) return cuda_failure(error, "fluid rope impulse clear failed");
                fluid_rope_contacts<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.count, particle_mass,
                    rope, distance, owner->options.friction, owner->node_impulses,
                    owner->contact_count, owner->maximum_penetration, true);
                fluid_rope_apply<<<(rope.count + block_size - 1U) / block_size,
                    block_size, 0, stream>>>(rope,
                    rope.options.first.enabled ? 0 : -1,
                    rope.options.last.enabled ? 0 : -1,
                    owner->node_impulses, owner->options.maximum_rope_acceleration,
                    dt, 1.0F / options.timestep);
                status = record_timing_stage(TimingStage::fluid_rope_contacts, 2U);
                if (!status) return status;
            }
            for (const auto &owner : impl_->fluid_soft_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    !(owner->options.fluid == fluid_id)) continue;
                auto &coupling = *owner;
                auto &body = *impl_->soft_bodies[coupling.options.soft_body.index];
                const auto node_blocks = (body.node_count + block_size - 1U) / block_size;
                const auto skin_blocks = (body.surface_vertex_count + block_size - 1U) / block_size;
                const float distance = coupling.options.contact_distance > 0
                    ? coupling.options.contact_distance : fluid.options.particle_radius;
                const auto refit_surface = [&]() -> Status {
                    const auto clear = cudaMemsetAsync(coupling.ready, 0,
                        coupling.tree_count * sizeof(std::uint32_t), stream);
                    if (clear != cudaSuccess) return cuda_failure(clear, "soft surface BVH clear failed");
                    fluid_soft_refit<<<(coupling.tree_count + block_size - 1U) / block_size,
                        block_size, 0, stream>>>(body.surface_positions, coupling.previous_surface,
                        body.surface_indices, coupling.triangle_order, coupling.tree, coupling.parents,
                        coupling.ready, coupling.tree_count, coupling.bounds);
                    return success();
                };
                for (std::uint32_t pass = 0; pass < coupling.options.solver_iterations; ++pass) {
                    error = cudaMemsetAsync(coupling.counts, 0,
                        body.node_count * sizeof(std::uint32_t), stream);
                    if (error == cudaSuccess) error = cudaMemsetAsync(coupling.position_deltas, 0,
                        body.node_count * sizeof(Vec3), stream);
                    if (error == cudaSuccess) error = cudaMemsetAsync(coupling.impulses, 0,
                        body.node_count * sizeof(Vec3), stream);
                    if (error != cudaSuccess) return cuda_failure(error, "fluid soft-body scratch clear failed");
                    status = refit_surface();
                    if (!status) return status;
                    fluid_soft_detect<<<blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.previous, fluid.velocities, fluid.count,
                        body.surface_positions, coupling.previous_surface, body.surface_indices,
                        body.surface_index_count, body.surface_bindings, body.velocities,
                        coupling.orientation, distance, coupling.bounds, pass == 0,
                        coupling.tree, coupling.triangle_order,
                        coupling.contacts, coupling.counts, coupling.contact_count,
                        coupling.maximum_penetration);
                    fluid_soft_solve<<<blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.velocities, fluid.forces, fluid.foam, fluid.count,
                        coupling.contacts, coupling.counts, body.inverse_masses,
                        body.velocities, body.maximum_speed,
                        1.0F / particle_mass, coupling.options.friction,
                        0.2F * body.node_radius, dt, coupling.position_deltas, coupling.impulses);
                    fluid_soft_apply<<<node_blocks, block_size, 0, stream>>>(
                        body.positions, body.velocities, body.inverse_masses,
                        coupling.position_deltas, coupling.impulses, body.fluid_forces,
                        body.node_count, 1.0F / options.timestep);
                    deformable_project_links<<<node_blocks, block_size, 0, stream>>>(
                        body.positions, body.scratch, body.inverse_masses, body.offsets,
                        body.neighbors, body.bond_active, body.node_count, dt,
                        body.maximum_projection_fraction);
                    std::swap(body.positions, body.scratch);
                    // Keep fluid reaction corrections on the valid side of
                    // passive/dynamic rigid geometry before updating the skin.
                    if (impl_->rigid_body_count != 0U)
                        deformable_collide<true, true><<<node_blocks, block_size, 0, stream>>>(
                            body.positions, body.velocities, body.previous, body.inverse_masses,
                            body.node_count, body.node_radius * 0.01F, dt, impl_->parameters,
                            nullptr, impl_->states[impl_->current_state], impl_->meshes,
                            impl_->rigid_body_count, body.body_impulses, body.rigid_contact_forces,
                            body.contact_normals, body.contact_arms, body.contact_momentum_delta,
                            body.contact_normal_delta, body.contact_count, body.dynamic_contact_flag,
                            impl_->fluid_body_contact_flags, false, false, body.maximum_speed);
                    soft_body_update_surface<<<skin_blocks, block_size, 0, stream>>>(
                        body.positions, body.rest_positions, body.surface_rest_positions,
                        body.surface_bindings, body.surface_positions, body.surface_vertex_count);
                }
                status = refit_surface();
                if (!status) return status;
                fluid_soft_recover<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.previous, fluid.count, body.surface_positions,
                    coupling.previous_surface, body.surface_indices, body.surface_index_count,
                    coupling.orientation, distance, coupling.bounds, true,
                    coupling.tree, coupling.triangle_order);
                error = cudaMemcpyAsync(coupling.previous_surface, body.surface_positions,
                    body.surface_vertex_count * sizeof(Vec3), cudaMemcpyDeviceToDevice, stream);
                if (error != cudaSuccess) return cuda_failure(error, "fluid soft-body skin copy failed");
                status = record_timing_stage(TimingStage::fluid_soft_body_contacts,
                    (6U + (impl_->rigid_body_count != 0U ? 1U : 0U)) *
                    coupling.options.solver_iterations + 2U);
                if (!status) return status;
            }
            for (const FluidClothCouplingResource &coupling :
                 impl_->fluid_cloth_couplings) {
                if (!coupling.alive || !coupling.options.enabled ||
                    !(coupling.options.fluid == fluid_id) ||
                    coupling.options.cloth.index >= impl_->cloths.size())
                    continue;
                const auto &cloth_pointer =
                    impl_->cloths[coupling.options.cloth.index];
                if (!cloth_pointer || !cloth_pointer->alive ||
                    cloth_pointer->generation !=
                        coupling.options.cloth.generation)
                    continue;
                const ClothStorage &cloth = *cloth_pointer;
                const float contact_distance =
                    coupling.options.contact_distance > 0.0F
                        ? coupling.options.contact_distance
                        : fluid.options.particle_radius + cloth.thickness;
                fluid_project_inside_cloth<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.count,
                    contact_distance, cloth.positions, cloth.velocities,
                    cloth.indices, cloth.index_count / 3U,
                    cloth.orientation);
            }
            if (any_moving_body) {
                error = cudaMemsetAsync(impl_->fluid_body_contact_flags, 0,
                    impl_->rigid_body_count * sizeof(std::uint32_t), stream);
                if (error != cudaSuccess)
                    return cuda_failure(error, "fluid contact flags clear failed");
                fluid_detect_moving_contacts<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.previous, fluid.count,
                    fluid.options.particle_radius,
                    impl_->parameters,
                    impl_->fluid_previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->fluid_body_bounds, impl_->rigid_body_count,
                    impl_->fluid_body_masks, impl_->fluid_global_body_masks,
                    body_words, iteration == 0U, fluid.moving_contacts,
                    impl_->fluid_body_contact_flags);
                fluid_resolve_moving_contacts<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.foam,
                    fluid.count, fluid.options.particle_radius,
                    particle_mass, dt, fluid.options.maximum_speed,
                    impl_->parameters, impl_->states[impl_->current_state],
                    impl_->meshes, fluid.moving_contacts,
                    impl_->fluid_body_contact_flags, fluid.body_impulses,
                    collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags, fluid_id, impl_->ids,
                    impl_->paint_rule_count != 0U,
                    impl_->paint_fields, impl_->options.paint_field_capacity,
                    impl_->paint_rules, impl_->options.paint_rule_capacity);
                reduce_point_body_impulses<<<impl_->rigid_body_count,
                                             block_size, 0, stream>>>(
                    fluid.body_impulses, fluid.count, impl_->parameters,
                    impl_->states[impl_->current_state],
                    impl_->fluid_body_contact_flags,
                    impl_->rigid_body_count);
                status = record_timing_stage(TimingStage::fluid_moving_contacts, 3U);
                if (!status) return status;
            }
            bool any_static_body = false;
            for (std::uint32_t body = 0U; body < impl_->rigid_body_count;
                 ++body) {
                if (impl_->parameters[body].motion != MotionType::static_body)
                    continue;
                any_static_body = true;
                fluid_static_contacts<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.previous,
                    fluid.foam, fluid.count, fluid.options.particle_radius,
                    fluid.options.support_radius, first_spawned,
                    spawned && iteration == 0U,
                    normalized_or(multiply(options.gravity, -1.0F),
                                  {0.0F, 1.0F, 0.0F}),
                    impl_->parameters, impl_->states[impl_->current_state],
                    impl_->meshes, body, particle_mass,
                    collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags, fluid_id, impl_->ids[body],
                    impl_->paint_rule_count != 0U,
                    impl_->paint_fields, impl_->options.paint_field_capacity,
                    impl_->paint_rules, impl_->options.paint_rule_capacity);
            }
            if (any_static_body) {
                status = record_timing_stage(TimingStage::fluid_static_contacts);
                if (!status) return status;
            }
            // A rigid boundary can push water back into a neighboring soft
            // face. Finish with current-skin recovery, not the old surface or
            // an infinite triangle plane. No extra kinetic impulse is added.
            for (const auto &coupling : impl_->fluid_soft_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    !(coupling->options.fluid == fluid_id)) continue;
                const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
                const float distance = coupling->options.contact_distance > 0
                    ? coupling->options.contact_distance : fluid.options.particle_radius;
                for (std::uint32_t recovery = 0; recovery < 3; ++recovery)
                    fluid_soft_recover<<<blocks, block_size, 0, stream>>>(
                        fluid.positions, fluid.positions, fluid.count, body.surface_positions,
                        body.surface_positions, body.surface_indices, body.surface_index_count,
                        coupling->orientation, distance, coupling->bounds, false,
                        coupling->tree, coupling->triangle_order);
                status = record_timing_stage(TimingStage::fluid_soft_body_contacts, 3U);
                if (!status) return status;
            }
            bool any_destroy_plane = false;
            for (const DestroyPlaneSlot &slot : impl_->destroy_planes) {
                if (!slot.alive || !slot.options.enabled ||
                    !(slot.options.fluid == fluid_id)) continue;
                fluid_destroy_flags<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.previous, fluid.count,
                    fluid.options.capacity, slot.options,
                    any_destroy_plane, fluid.keep);
                any_destroy_plane = true;
            }
            if (any_destroy_plane) {
                const auto sequence =
                    thrust::make_counting_iterator<std::uint32_t>(0U);
                error = cub::DeviceSelect::Flagged(
                    fluid.select_workspace, fluid.select_workspace_size,
                    sequence, fluid.keep, fluid.selected, fluid.count,
                    fluid.options.capacity, stream);
                if (error != cudaSuccess) {
                    cudaStreamSynchronize(stream);
                    return cuda_failure(error, "fluid particle compaction failed");
                }
                fluid_gather<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.ids, fluid.foam,
                    fluid.temperatures,
                    fluid.contact_samples, fluid.contact_flags,
                    collect_fluid_contacts,
                    fluid.selected, fluid.count, fluid.options.capacity,
                    fluid.next_positions, fluid.next_velocities,
                    fluid.next_ids, fluid.next_foam,
                    fluid.next_temperatures,
                    fluid.next_contact_samples, fluid.next_contact_flags);
                std::swap(fluid.positions, fluid.next_positions);
                std::swap(fluid.velocities, fluid.next_velocities);
                std::swap(fluid.ids, fluid.next_ids);
                std::swap(fluid.foam, fluid.next_foam);
                std::swap(fluid.temperatures, fluid.next_temperatures);
                std::swap(fluid.contact_samples, fluid.next_contact_samples);
                std::swap(fluid.contact_flags, fluid.next_contact_flags);
                status = record_timing_stage(
                    TimingStage::fluid_outflow_compaction);
                if (!status) return status;
            }
        }
        if (collect_fluid_contacts) {
            const auto sequence =
                thrust::make_counting_iterator<std::uint32_t>(0U);
            error = cub::DeviceSelect::Flagged(
                fluid.select_workspace, fluid.select_workspace_size,
                sequence, fluid.contact_flags, fluid.selected,
                fluid.contact_count, fluid.options.capacity, stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "fluid contact selection failed");
            fluid_reserve_contact_events<<<1U, 1U, 0, stream>>>(
                fluid.contact_count, fluid.contact_offset,
                impl_->fluid_contact_count, impl_->fluid_contact_overflow,
                impl_->options.contact_capacity);
            fluid_gather_contact_events<<<blocks, block_size, 0, stream>>>(
                fluid.selected, fluid.contact_count, fluid.contact_offset,
                fluid.contact_samples, fluid.ids, impl_->ids, fluid_id,
                impl_->fluid_contact_events, impl_->options.contact_capacity);
            status = record_timing_stage(TimingStage::fluid_contact_events);
            if (!status) return status;
        }
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return cuda_failure(error, "fluid kernel launch failed");
        }
    }
    for (auto &owner : impl_->fluid_smoke_couplings) {
        if (!owner || !owner->alive) continue;
        auto &coupling = *owner;
        auto &fluid = *impl_->fluids[coupling.options.fluid.index];
        auto &smoke = *impl_->smokes[coupling.options.smoke.index];
        const auto live = *fluid.count;
        if (live == 0U) continue;
        if (smoke.grid.resolution == 0U && smoke.count != 0U &&
            smoke.index_dirty) {
            smoke_emit_cells<<<(smoke.options.capacity + block_size - 1U) /
                block_size, block_size, 0, stream>>>(
                smoke.positions, smoke.ages, smoke.count, smoke.options,
                smoke.keys[0], smoke.indices[0]);
            error = cub::DeviceRadixSort::SortPairs(
                smoke.sort_workspace, smoke.sort_workspace_size,
                smoke.keys[0], smoke.keys[1], smoke.indices[0],
                smoke.indices[1], smoke.options.capacity, 0, 64, stream);
            if (error != cudaSuccess)
                return cuda_failure(error, "smoke phase-transfer index failed");
            smoke.index_dirty = false;
        }
        *coupling.converted = 0U;
        const auto blocks = (fluid.options.capacity + block_size - 1U) / block_size;
        if (smoke.count != 0U && coupling.options.wind_drag > 0.0F) {
            if (smoke.grid.resolution != 0U)
                fluid_smoke_grid_drag<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.count,
                    coupling.options, smoke.options,
                    {smoke.grid.resolution, smoke.grid.height,
                     smoke.grid.minimum, smoke.grid.spacing,
                     smoke.grid.velocity, smoke.grid.density,
                     smoke.grid.pressure[0],
                     {smoke.grid.face_velocity[0][0],
                      smoke.grid.face_velocity[1][0],
                      smoke.grid.face_velocity[2][0]},
                     smoke.grid.vorticity, smoke.grid.strain},
                    options.timestep);
            else
                fluid_smoke_drag<<<blocks, block_size, 0, stream>>>(
                    fluid.positions, fluid.velocities, fluid.count,
                    coupling.options, smoke.options, smoke.positions,
                    smoke.velocities, smoke.keys[1], smoke.indices[1],
                    options.timestep);
        }
        fluid_smoke_exchange<<<blocks, block_size, 0, stream>>>(
            fluid.positions, fluid.velocities, fluid.temperatures,
            fluid.count, fluid.options.particle_radius, coupling.options,
            smoke.options, smoke.positions, smoke.velocities,
            smoke.ages, smoke.thermal_lift,
            smoke.number_densities, smoke.pressures, smoke.vorticities,
            smoke.next_slot,
            coupling.converted, fluid.keep, options.timestep,
            options.gravity);
        error = cudaGetLastError();
        if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
            return cuda_failure(error, "fluid smoke exchange failed");
        const auto converted = std::min(*coupling.converted, smoke.options.capacity);
        if (converted == 0U) {
            status = record_timing_stage(TimingStage::fluid_smoke_exchange);
            if (!status) return status;
            continue;
        }
        smoke.next_slot = (smoke.next_slot + converted) % smoke.options.capacity;
        smoke.count = std::min(smoke.options.capacity, smoke.count + converted);
        smoke.emitted += converted;
        smoke.index_dirty = true;
        impl_->boiled_particle_count += converted;
        fluid_clear_inactive_keep<<<blocks, block_size, 0, stream>>>(
            fluid.keep, fluid.count, fluid.options.capacity);
        const auto sequence = thrust::make_counting_iterator<std::uint32_t>(0U);
        error = cub::DeviceSelect::Flagged(
            fluid.select_workspace, fluid.select_workspace_size,
            sequence, fluid.keep, fluid.selected, fluid.count,
            fluid.options.capacity, stream);
        if (error != cudaSuccess)
            return cuda_failure(error, "boiled fluid compaction failed");
        fluid_gather<<<blocks, block_size, 0, stream>>>(
            fluid.positions, fluid.velocities, fluid.ids, fluid.foam,
            fluid.temperatures, fluid.contact_samples, fluid.contact_flags,
            false, fluid.selected, fluid.count, fluid.options.capacity,
            fluid.next_positions, fluid.next_velocities, fluid.next_ids,
            fluid.next_foam, fluid.next_temperatures,
            fluid.next_contact_samples, fluid.next_contact_flags);
        std::swap(fluid.positions, fluid.next_positions);
        std::swap(fluid.velocities, fluid.next_velocities);
        std::swap(fluid.ids, fluid.next_ids);
        std::swap(fluid.foam, fluid.next_foam);
        std::swap(fluid.temperatures, fluid.next_temperatures);
        status = record_timing_stage(TimingStage::fluid_smoke_exchange, 3U);
        if (!status) return status;
    }
    for (auto &owner : impl_->smokes) {
        if (!owner || !owner->alive) continue;
        auto &smoke = *owner;
        if (smoke.count != 0U) {
            const auto blocks = (smoke.count + 127U) / 128U;
            if (smoke.grid.resolution != 0U) {
                smoke_grid_trace<<<blocks, 128U, 0, stream>>>(
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages, smoke.thermal_lift,
                    smoke.number_densities, smoke.pressures,
                    smoke.vorticities, smoke.count, smoke.options,
                    {smoke.grid.resolution, smoke.grid.height,
                     smoke.grid.minimum, smoke.grid.spacing,
                     smoke.grid.velocity, smoke.grid.density,
                     smoke.grid.pressure[0],
                     {smoke.grid.face_velocity[0][0],
                      smoke.grid.face_velocity[1][0],
                      smoke.grid.face_velocity[2][0]},
                     smoke.grid.vorticity, smoke.grid.strain},
                    options.timestep, options.gravity);
            } else {
                smoke_emit_cells<<<(smoke.options.capacity + 127U) / 128U,
                    128U, 0, stream>>>(smoke.positions, smoke.ages, smoke.count,
                    smoke.options, smoke.keys[0], smoke.indices[0]);
                error = cub::DeviceRadixSort::SortPairs(
                    smoke.sort_workspace, smoke.sort_workspace_size,
                    smoke.keys[0], smoke.keys[1], smoke.indices[0],
                    smoke.indices[1], smoke.options.capacity, 0, 64, stream);
                if (error != cudaSuccess)
                    return cuda_failure(error,
                        "smoke pressure neighbor sort failed");
                smoke.index_dirty = false;
                smoke_density_pressure<<<blocks, 128U, 0, stream>>>(
                    smoke.positions, smoke.ages, smoke.count, smoke.options,
                    smoke.keys[1], smoke.indices[1], smoke.number_densities,
                    smoke.pressures, options.timestep);
                smoke_compute_vorticity<<<blocks, 128U, 0, stream>>>(
                    smoke.positions, smoke.velocities, smoke.ages, smoke.count,
                    smoke.options, smoke.keys[1], smoke.indices[1],
                    smoke.number_densities, smoke.vorticities,
                    smoke.vorticity_magnitudes);
                smoke_pair_forces<<<blocks, 128U, 0, stream>>>(
                    smoke.positions, smoke.velocities, smoke.ages, smoke.count,
                    smoke.options, smoke.keys[1], smoke.indices[1],
                    smoke.number_densities, smoke.pressures,
                    smoke.vorticities, smoke.vorticity_magnitudes,
                    smoke.accelerations);
                smoke_advect<<<blocks, 128U, 0, stream>>>(
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages,
                    smoke.thermal_lift, smoke.count,
                    smoke.options, smoke.accelerations,
                    options.timestep, options.gravity);
            }
            for (const auto &coupling : impl_->smoke_soft_body_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                const auto &body = *impl_->soft_bodies[
                    coupling->options.soft_body.index];
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius + body.node_radius;
                    smoke_deformable_bounds<<<1U, 1U, 0, stream>>>(
                        body.surface_positions, body.surface_vertex_count,
                        coupling->minimum, coupling->maximum);
                    smoke_soft_body_contact<<<
                        (smoke.count + 127U) / 128U, 128U, 0, stream>>>(
                        smoke.positions, smoke.previous_positions,
                        smoke.velocities, smoke.ages,
                        smoke.count, smoke.options.lifetime,
                        body.surface_positions, body.surface_bindings,
                        body.surface_indices, body.surface_index_count,
                        body.velocities, coupling->minimum,
                        coupling->maximum, clearance,
                        {smoke.grid.resolution, smoke.grid.height,
                         smoke.grid.minimum, smoke.grid.spacing,
                         smoke.grid.velocity, smoke.grid.density,
                         smoke.grid.pressure[0],
                         {smoke.grid.face_velocity[0][0],
                          smoke.grid.face_velocity[1][0],
                          smoke.grid.face_velocity[2][0]},
                         smoke.grid.vorticity, smoke.grid.strain});
            }
            for (const auto &coupling : impl_->smoke_cloth_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                const auto &cloth = *impl_->cloths[coupling->options.cloth.index];
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius + cloth.thickness;
                smoke_deformable_bounds<<<1U, 1U, 0, stream>>>(
                    cloth.positions, cloth.vertex_count,
                    coupling->minimum, coupling->maximum);
                smoke_cloth_contact<<<
                    (smoke.count + 127U) / 128U, 128U, 0, stream>>>(
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages, smoke.pressures,
                    smoke.count, smoke.options.lifetime,
                    cloth.positions, cloth.velocities,
                    cloth.indices, cloth.index_count,
                    coupling->minimum, coupling->maximum, clearance,
                    smoke.options.pressure_stiffness,
                    smoke.options.maximum_speed,
                    {smoke.grid.resolution, smoke.grid.height,
                     smoke.grid.minimum, smoke.grid.spacing,
                     smoke.grid.velocity, smoke.grid.density,
                     smoke.grid.pressure[0],
                     {smoke.grid.face_velocity[0][0],
                      smoke.grid.face_velocity[1][0],
                      smoke.grid.face_velocity[2][0]},
                     smoke.grid.vorticity, smoke.grid.strain});
            }
            for (const auto &coupling : impl_->smoke_rigid_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                std::uint32_t body = 0U;
                status = impl_->validate_handle(coupling->options.body, body);
                if (!status) return status;
                const auto &mesh = impl_->meshes[impl_->parameters[body].mesh.index];
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius;
                if (smoke.grid.resolution != 0U &&
                    coupling->options.air_density > 0.0F)
                    smoke_grid_rigid_force<<<1U, 128U, 0, stream>>>(mesh,
                        body, impl_->parameters,
                        impl_->states[impl_->current_state],
                        {smoke.grid.resolution, smoke.grid.height,
                         smoke.grid.minimum, smoke.grid.spacing,
                         smoke.grid.velocity, smoke.grid.density,
                         smoke.grid.pressure[0],
                         {smoke.grid.face_velocity[0][0],
                          smoke.grid.face_velocity[1][0],
                          smoke.grid.face_velocity[2][0]},
                         smoke.grid.vorticity, smoke.grid.strain},
                        coupling->options.air_density,
                        coupling->options.drag_coefficient,
                        smoke.options.rest_number_density,
                        smoke.options.grid_kinematic_viscosity,
                        smoke.options.grid_les_coefficient,
                        options.timestep);
                if (coupling->options.tracer_contact)
                    smoke_rigid_contact<<<
                        (smoke.count + 127U) / 128U, 128U, 0, stream>>>(
                        smoke.positions, smoke.previous_positions,
                        smoke.velocities, smoke.ages, smoke.pressures, smoke.count,
                        smoke.options.lifetime, mesh,
                        impl_->states[impl_->current_state],
                        any_moving_body ? impl_->fluid_previous_states
                                        : impl_->states[impl_->current_state],
                        body, clearance,
                        (smoke.grid.resolution == 0U ?
                         coupling->options.air_density : 0.0F) *
                            std::pow(2.0F * smoke.options.particle_radius, 3.0F),
                        coupling->options.drag_coefficient,
                        smoke.options.pressure_stiffness, options.timestep,
                        smoke.options.maximum_speed,
                        {smoke.grid.resolution, smoke.grid.height,
                         smoke.grid.minimum,
                         smoke.grid.spacing, smoke.grid.velocity,
                         smoke.grid.density, smoke.grid.pressure[0],
                         {smoke.grid.face_velocity[0][0],
                          smoke.grid.face_velocity[1][0],
                          smoke.grid.face_velocity[2][0]},
                         smoke.grid.vorticity, smoke.grid.strain},
                        smoke.rigid_impulses);
                if (smoke.grid.resolution == 0U &&
                    coupling->options.air_density > 0.0F &&
                    coupling->options.tracer_contact)
                    smoke_apply_rigid_impulses<<<1U, 128U, 0, stream>>>(
                        smoke.rigid_impulses, smoke.count, body,
                        impl_->parameters,
                        impl_->states[impl_->current_state]);
            }
            for (const auto &coupling : impl_->smoke_rope_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                const auto &rope = impl_->ropes[coupling->options.rope.index]->data;
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius + rope.options.radius;
                smoke_deformable_bounds<<<1U, 1U, 0, stream>>>(
                    rope.positions, rope.count,
                    coupling->minimum, coupling->maximum);
                smoke_rope_contact<<<
                    (smoke.count + 127U) / 128U, 128U, 0, stream>>>(
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages,
                    smoke.count, smoke.options.lifetime, rope,
                    coupling->minimum, coupling->maximum, clearance);
            }
            status = record_timing_stage(TimingStage::smoke_advection, 1U);
            if (!status) return status;
        }
        const double exact = double(smoke.emission_fraction) +
            double(smoke.options.particles_per_second) * options.timestep;
        const auto requested = static_cast<std::uint32_t>(std::min(
            std::floor(exact), double(smoke.options.capacity)));
        smoke.emission_fraction = float(exact - std::floor(exact));
        if (requested != 0U) {
            smoke_emit<<<(requested + 127U) / 128U, 128U, 0, stream>>>(
                smoke.positions, smoke.previous_positions,
                smoke.velocities, smoke.ages,
                smoke.thermal_lift, smoke.number_densities,
                smoke.pressures, smoke.vorticities, smoke.options,
                smoke.next_slot, requested, smoke.emitted);
            smoke.next_slot = (smoke.next_slot + requested) % smoke.options.capacity;
            smoke.count = std::min(smoke.options.capacity, smoke.count + requested);
            smoke.emitted += requested;
            status = record_timing_stage(TimingStage::smoke_emission, 1U);
            if (!status) return status;
        }
        smoke.time = float(std::fmod(
            double(smoke.time) + double(options.timestep), 1000.0));
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure(error, "smoke kernel launch failed");
    }
    if (options.collect_kernel_timings && timing_boundary == 1U) {
        error = cudaEventRecord(impl_->timing_events[timing_boundary++], stream);
        if (error != cudaSuccess)
            return cuda_failure(error, "failed to finish empty kernel timing");
    }
    error = cudaEventRecord(frame->event, stream);
    if (error != cudaSuccess) {
        cudaStreamSynchronize(stream);
        return cuda_failure(error, "failed to record frame completion event");
    }

    ++impl_->frame_index;
    ++impl_->revision;
    impl_->rigid_contact_revision = impl_->revision;
    if (options.collect_kernel_timings) {
        impl_->timing_available = true;
        impl_->timing_boundary_count = timing_boundary;
        impl_->timing_frame_index = impl_->frame_index;
    }
    impl_->frame = frame;
    token_impl->completion = std::move(frame);
    completion.impl_ = std::move(token_impl);
    return success();
}

Status World::step(StepOptions options, cudaStream_t stream) noexcept {
    FrameToken completion;
    Status status = step_async(options, completion, stream);
    if (!status) {
        return status;
    }
    return completion.wait();
}

ContactDeviceView World::contacts() const noexcept {
    if (!impl_ || (impl_->frame && !impl_->frame->acknowledged)) return {};
    const std::uint32_t count = *impl_->fluid_contact_count;
    return {{impl_->fluid_contact_events, count}, count,
            *impl_->fluid_contact_overflow != 0U, impl_->frame_index};
}

RigidContactDeviceView World::rigid_contacts() const noexcept {
    if (!impl_ || (impl_->frame && !impl_->frame->acknowledged)) {
        return {};
    }
    return {{impl_->rigid_contact_events, *impl_->rigid_contact_count},
            *impl_->rigid_contact_count, impl_->frame_index};
}

Status World::physics_debug_frame(
    PhysicsDebugFrameView &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->options.physics_debug.frame_capacity == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture was not enabled at world creation");
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    if (impl_->debug_frame_count == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture has no completed frame");
    }
    const std::size_t index =
        (impl_->debug_next_frame + impl_->debug_frames.size() - 1U) %
        impl_->debug_frames.size();
    const PhysicsDebugFrame &frame = impl_->debug_frames[index];
    output.frame_index = frame.frame_index;
    output.timestep = frame.timestep;
    output.gravity = frame.gravity;
    output.maximum_fluid_neighbor_count =
        frame.maximum_fluid_neighbor_count;
    output.rigid_bodies = {
        frame.rigid_bodies.data(), frame.rigid_bodies.size()};
    output.fluid_particles = {
        frame.fluid_particles.data(), frame.fluid_particles.size()};
    output.cloth_vertices = {
        frame.cloth_vertices.data(), frame.cloth_vertices.size()};
    output.soft_body_nodes = {
        frame.soft_body_nodes.data(), frame.soft_body_nodes.size()};
    output.rope_nodes = {frame.rope_nodes.data(), frame.rope_nodes.size()};
    output.rigid_contacts = {
        frame.rigid_contacts.data(), frame.rigid_contacts.size()};
    output.fluid_contacts = {
        frame.fluid_contacts.data(), frame.fluid_contacts.size()};
    return success();
}

Status World::copy_physics_debug_capture(
    PhysicsDebugCapture &output) const noexcept {
    output.frames.clear();
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->options.physics_debug.frame_capacity == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture was not enabled at world creation");
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    try {
        output.frames.reserve(impl_->debug_frame_count);
        const std::size_t first =
            (impl_->debug_next_frame + impl_->debug_frames.size() -
             impl_->debug_frame_count) % impl_->debug_frames.size();
        for (std::size_t offset = 0U;
             offset < impl_->debug_frame_count; ++offset) {
            output.frames.push_back(impl_->debug_frames[
                (first + offset) % impl_->debug_frames.size()]);
        }
    } catch (...) {
        output.frames.clear();
        return failure(StatusCode::out_of_memory,
                       "physics debug capture copy failed");
    }
    return success();
}

Status World::collect_step_timings(WorldStepTimings &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) {
        return status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    if (!impl_->timing_available || impl_->timing_boundary_count < 2U) {
        output.frame_index = impl_->frame_index;
        return success();
    }

    output.frame_index = impl_->timing_frame_index;
    output.available = true;
    cudaError_t error = cudaEventElapsedTime(
        &output.total_gpu_milliseconds, impl_->timing_events[0],
        impl_->timing_events[impl_->timing_boundary_count - 1U]);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to collect total kernel timing");
    }
    for (std::size_t index = 0; index < impl_->timing_stages.size(); ++index) {
        float milliseconds = 0.0F;
        error = cudaEventElapsedTime(&milliseconds, impl_->timing_events[index],
                                     impl_->timing_events[index + 1U]);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to collect kernel stage timing");
        }
        KernelTiming *timing = nullptr;
        bool contact_generation_stage = false;
        switch (impl_->timing_stages[index]) {
        case TimingStage::rigid_integration:
            timing = &output.rigid_integration;
            break;
        case TimingStage::rigid_world_bounds:
            timing = &output.rigid_world_bounds;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_pair_filter:
            timing = &output.rigid_pair_filter;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_pair_compaction:
            timing = &output.rigid_pair_compaction;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_leaf_pair_generation:
            timing = &output.rigid_leaf_pair_generation;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_contact_evaluation:
            timing = &output.rigid_contact_evaluation;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_contact_solve:
            timing = &output.rigid_contact_solve;
            break;
        case TimingStage::rigid_input_clear:
            timing = &output.rigid_input_clear;
            break;
        case TimingStage::rope_solve:
            timing = &output.rope_solve;
            break;
        case TimingStage::smoke_advection:
            timing = &output.smoke_advection;
            break;
        case TimingStage::smoke_grid:
            timing = &output.smoke_grid;
            break;
        case TimingStage::smoke_emission:
            timing = &output.smoke_emission;
            break;
        case TimingStage::cloth_prediction:
            timing = &output.cloth_prediction;
            break;
        case TimingStage::cloth_constraints:
            timing = &output.cloth_constraints;
            break;
        case TimingStage::cloth_contacts:
            timing = &output.cloth_contacts;
            break;
        case TimingStage::soft_body_prediction:
            timing = &output.soft_body_prediction;
            break;
        case TimingStage::soft_body_constraints:
            timing = &output.soft_body_constraints;
            break;
        case TimingStage::soft_body_contacts:
        case TimingStage::soft_body_contact_cleanup:
            timing = &output.soft_body_contacts;
            break;
        case TimingStage::soft_body_cloth_contacts:
            timing = &output.soft_body_cloth_contacts;
            break;
        case TimingStage::fluid_cloth_contacts:
            timing = &output.fluid_cloth_contacts;
            break;
        case TimingStage::fluid_soft_body_contacts:
            timing = &output.fluid_soft_body_contacts;
            break;
        case TimingStage::fluid_rope_contacts:
            timing = &output.fluid_rope_contacts;
            break;
        case TimingStage::rope_soft_body_contacts:
            timing = &output.rope_soft_body_contacts;
            break;
        case TimingStage::fluid_spawn:
            timing = &output.fluid_spawn;
            break;
        case TimingStage::fluid_neighbor_sort:
            timing = &output.fluid_neighbor_sort;
            break;
        case TimingStage::fluid_neighbor_forces:
            timing = &output.fluid_neighbor_forces;
            break;
        case TimingStage::fluid_integration:
            timing = &output.fluid_integration;
            break;
        case TimingStage::fluid_static_contacts:
            timing = &output.fluid_static_contacts;
            break;
        case TimingStage::fluid_body_index:
            timing = &output.fluid_body_index;
            break;
        case TimingStage::fluid_moving_contacts:
            timing = &output.fluid_moving_contacts;
            break;
        case TimingStage::fluid_contact_events:
            timing = &output.fluid_contact_events;
            break;
        case TimingStage::fluid_outflow_compaction:
            timing = &output.fluid_outflow_compaction;
            break;
        case TimingStage::fluid_smoke_exchange:
            timing = &output.fluid_smoke_exchange;
            break;
        }
        timing->total_milliseconds += milliseconds;
        const std::uint32_t launches =
            impl_->timing_launch_counts[index] != 0U ? impl_->timing_launch_counts[index] :
            impl_->timing_stages[index] == TimingStage::soft_body_cloth_contacts
                ? impl_->soft_cloth_kernels_per_substep
                : impl_->timing_stages[index] == TimingStage::soft_body_contact_cleanup
                ? 2U + (impl_->rigid_body_count != 0U
                    ? k_soft_contact_cleanup_passes * 5U / 2U : 0U)
                : impl_->timing_stages[index] == TimingStage::rigid_contact_solve
                ? impl_->rigid_solve_kernels_per_substep
                : impl_->timing_stages[index] ==
                          TimingStage::rigid_contact_evaluation
                    ? 3U + (impl_->rigid_leaf_manifolds != nullptr ? 1U : 0U) : 1U;
        timing->launch_count += launches;
        if (contact_generation_stage) {
            output.rigid_contact_generation.total_milliseconds += milliseconds;
            output.rigid_contact_generation.launch_count += launches;
        }
    }
    return success();
}

Status World::collect_statistics(WorldStatistics &output,
                                 cudaStream_t stream) const noexcept {
    (void)stream;
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "world is not initialized");
    }
    Status device_status = impl_->require_current_device();
    if (!device_status) {
        return device_status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        Status status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    const std::size_t capacity = impl_->options.rigid_body_capacity;
    output = {};
    output.frame_index = impl_->frame_index;
    output.fluid_count = impl_->fluid_count;
    output.boiled_particle_count = impl_->boiled_particle_count;
    for (const auto &smoke : impl_->smokes) {
        if (!smoke || !smoke->alive) continue;
        ++output.smoke_system_count;
        output.smoke_particle_count += smoke->count;
        output.emitted_smoke_particle_count += smoke->emitted;
        output.allocated_bytes += static_cast<std::size_t>(smoke->options.capacity) *
            (2U * sizeof(Vec3) + 2U * sizeof(float));
    }
    for (const auto &cloth : impl_->cloths) {
        if (!cloth || !cloth->alive) continue;
        ++output.cloth_count;
        output.cloth_vertex_count += cloth->vertex_count;
        output.allocated_bytes +=
            static_cast<std::size_t>(cloth->vertex_capacity) *
                (6U * sizeof(Vec3) + sizeof(float) + sizeof(FluidBodyImpulse)) +
            cloth->index_count * sizeof(std::uint32_t) +
            (static_cast<std::size_t>(cloth->vertex_capacity) + 1U) *
                sizeof(std::uint32_t) +
            cloth->neighbor_capacity * sizeof(DeformableNeighbor) +
            cloth->bond_count * (sizeof(ClothBond) +
                2U * sizeof(std::uint8_t)) +
            (cloth->surface_positions != nullptr
                ? cloth->index_count * (sizeof(Vec3) +
                    3U * sizeof(std::uint32_t)) +
                    cloth->vertex_capacity * (sizeof(std::uint32_t) + sizeof(std::uint8_t)) : 0U) +
            impl_->options.rigid_body_capacity *
                sizeof(ClothBodyCorrection) +
            (cloth->volume_gradients != nullptr
                ? static_cast<std::size_t>(cloth->vertex_count) * sizeof(Vec3) +
                    (static_cast<std::size_t>(cloth->vertex_count) + 1U +
                        cloth->index_count) * sizeof(std::uint32_t) +
                    sizeof(float) : 0U) +
            (cloth->fluid_forces != nullptr
                ? static_cast<std::size_t>(cloth->vertex_capacity) * sizeof(Vec3)
                : 0U) +
            sizeof(std::uint32_t);
    }
    for (const auto &body : impl_->soft_bodies) {
        if (!body || !body->alive) continue;
        ++output.soft_body_count;
        output.soft_body_node_count += body->node_count;
        output.allocated_bytes +=
            static_cast<std::size_t>(body->node_count) *
                (14U * sizeof(Vec3) + 2U * sizeof(float) +
                 sizeof(FluidBodyImpulse)) +
            static_cast<std::size_t>(body->bond_count) *
                (sizeof(SoftBodyBond) + sizeof(std::uint8_t)) +
            (static_cast<std::size_t>(body->node_count) + 1U) *
                sizeof(std::uint32_t) +
            static_cast<std::size_t>(body->neighbor_count) *
                sizeof(DeformableNeighbor) +
            (body->warp_neighbors != nullptr
                ? static_cast<std::size_t>(body->neighbor_count) *
                    sizeof(SoftBodyNeighbor) +
                    static_cast<std::size_t>(body->node_count) * sizeof(float)
                : 0U) +
            static_cast<std::size_t>(body->surface_vertex_count) *
                (2U * sizeof(Vec3) + sizeof(SoftBodySurfaceBinding)) +
            static_cast<std::size_t>(body->surface_index_count) *
                sizeof(std::uint32_t) + 3U * sizeof(std::uint32_t) +
                sizeof(Vec3) + sizeof(Quaternion);
    }
    for (const auto &coupling : impl_->soft_cloth_couplings) {
        if (!coupling || !coupling->alive) continue;
        output.allocated_bytes +=
            impl_->soft_bodies[coupling->options.soft_body.index]->node_count *
                sizeof(SoftClothContact) +
            impl_->cloths[coupling->options.cloth.index]->vertex_capacity *
                sizeof(std::uint32_t);
    }
    output.emitted_particle_count = impl_->emitted_particle_count;
    for(const auto &rope:impl_->ropes) {
        if(!rope || !rope->alive)continue;
        ++output.rope_count;output.rope_node_count+=rope->data.count;
        output.allocated_bytes+=rope->data.count*(10*sizeof(Vec3)+2*sizeof(float))+
            2*impl_->options.rigid_body_capacity*sizeof(Vec3);
    }
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
        const auto &fluid = *impl_->fluids[coupling->options.fluid.index];
        output.allocated_bytes += fluid.options.capacity * sizeof(FluidSoftContact) +
            body.node_count * (sizeof(std::uint32_t) + 2U * sizeof(Vec3)) +
            body.surface_vertex_count * sizeof(Vec3) + 2U * sizeof(Vec3) +
            sizeof(std::uint32_t) + sizeof(float);
        output.fluid_soft_body_contact_count += *coupling->contact_count;
        output.maximum_fluid_soft_body_penetration = std::max(
            output.maximum_fluid_soft_body_penetration, *coupling->maximum_penetration);
    }
    for (const auto &coupling : impl_->fluid_rope_couplings) {
        if (!coupling || !coupling->alive) continue;
        output.allocated_bytes += impl_->ropes[coupling->options.rope.index]->data.count *
            sizeof(Vec3) + sizeof(std::uint32_t) + sizeof(float);
        output.fluid_rope_contact_count += *coupling->contact_count;
        output.maximum_fluid_rope_penetration = std::max(
            output.maximum_fluid_rope_penetration, *coupling->maximum_penetration);
    }
    for (const auto &coupling : impl_->rope_soft_body_couplings) {
        if (!coupling || !coupling->alive) continue;
        output.allocated_bytes += impl_->soft_bodies[coupling->options.soft_body.index]->node_count *
            sizeof(Vec3) + impl_->soft_bodies[coupling->options.soft_body.index]->surface_vertex_count*
            sizeof(Vec3) + 2U*sizeof(Vec3) +
            coupling->tree_count*(sizeof(BvhNode)+2U*sizeof(std::uint32_t))+
            impl_->soft_bodies[coupling->options.soft_body.index]->surface_index_count/3U*
                sizeof(std::uint32_t)+sizeof(std::uint32_t)+sizeof(float);
        output.rope_soft_body_contact_count += *coupling->contact_count;
        output.maximum_rope_soft_body_penetration = std::max(
            output.maximum_rope_soft_body_penetration,*coupling->maximum_penetration);
    }
    output.destroyed_particle_count = impl_->destroyed_particle_count;
    output.spawn_capacity_miss_count = impl_->spawn_capacity_miss_count;
    for (const auto &source : impl_->particle_sources)
        if (source.alive)
            output.allocated_bytes += source.data->count * (sizeof(Vec3) + sizeof(std::uint8_t)) + sizeof(std::uint32_t);
    output.contact_count = *impl_->fluid_contact_count;
    output.contact_overflow_count = *impl_->fluid_contact_overflow;
    output.maximum_fluid_neighbor_count =
        *impl_->fluid_maximum_neighbor_count;
    for (const auto &fluid : impl_->fluids) {
        if (!fluid || !fluid->alive) continue;
        const std::uint32_t live = *fluid->count;
        output.particle_count += live;
        output.destroyed_particle_count +=
            fluid->initial_count + fluid->emitted_count - live;
        const std::size_t capacity = fluid->options.capacity;
        output.allocated_bytes += capacity *
            (6U * sizeof(Vec3) + 5U * sizeof(std::uint32_t) +
             5U * sizeof(float) + 3U * sizeof(std::uint8_t) +
             2U * sizeof(std::uint64_t) + sizeof(FluidBodyImpulse) +
             2U * sizeof(FluidContactSample)) +
             fluid->sort_workspace_size + fluid->select_workspace_size +
             3U * sizeof(std::uint32_t);
    }
    output.rigid_body_count = impl_->rigid_body_count;
    output.rigid_constraint_count = impl_->rigid_constraint_count;
    output.triangle_mesh_count = impl_->triangle_mesh_count;
    if (impl_->frame_index != 0U && impl_->rigid_body_count != 0U) {
        const auto &schedule = *impl_->rigid_contact_schedule;
        output.rigid_contact_island_count = schedule.island_count;
        output.rigid_contact_early_exit_count = schedule.early_exit_count;
        output.rigid_contact_maximum_passes = schedule.maximum_passes;
        output.rigid_contact_color_count = schedule.remaining;
        output.rigid_contact_overflow_pairs = schedule.counts[28];
        output.rigid_contact_grid_blocks = schedule.blocks;
        output.rigid_contact_candidate_pairs = schedule.candidates;
        output.rigid_contact_live_pairs = schedule.contacts;
    }
    output.allocated_bytes +=
        capacity * (sizeof(BodyParameters) + sizeof(BodyAccumulator) + sizeof(AvbdBody) +
                    sizeof(KinematicTarget) + sizeof(RigidBodyId) +
                    4U * sizeof(RigidBodyState) +
                    (impl_->fixed_contact_projection != nullptr
                         ? sizeof(FixedContactProjection)
                         : 0U) +
                    (impl_->debug_applied_forces != nullptr
                         ? 2U * sizeof(Vec3) + sizeof(PhysicsDebugRigidSample) : 0U) +
                    sizeof(std::uint8_t)) +
        capacity * (capacity - 1U) / 2U * sizeof(ContactManifold) +
        impl_->rigid_leaf_pair_slot_capacity * sizeof(CachedContactPair) +
        capacity * capacity * sizeof(std::uint32_t) +
        sizeof(ContactSchedule) +
        capacity * (capacity - 1U) / 2U * sizeof(std::uint32_t) +
        capacity * (2U * sizeof(WorldAabb) + sizeof(std::uint32_t)) +
        capacity * capacity *
            (sizeof(std::uint8_t) + sizeof(std::uint32_t)) +
        sizeof(std::uint32_t) + impl_->rigid_broad_phase_workspace_size +
        impl_->rigid_leaf_pair_capacity * sizeof(LeafPair) +
        (impl_->rigid_leaf_manifolds != nullptr
             ? impl_->rigid_leaf_pair_capacity * sizeof(ContactManifold) : 0U) +
        capacity * capacity * sizeof(std::uint32_t) +
        impl_->rigid_contact_capacity * sizeof(RigidContactEvent) +
        impl_->options.contact_capacity * sizeof(ContactEvent) +
        k_fluid_body_buckets *
            ((capacity + 63U) / 64U) * sizeof(unsigned long long) +
        ((capacity + 63U) / 64U) * sizeof(unsigned long long) +
        4U * sizeof(std::uint32_t) +
        impl_->options.triangle_mesh_capacity * sizeof(TriangleMeshResource) +
        impl_->options.paint_field_capacity * sizeof(PaintFieldResource) +
        impl_->options.paint_rule_capacity * sizeof(PaintRuleResource) +
        impl_->options.rigid_constraint_capacity *
            sizeof(RigidConstraintResource);
    output.allocated_bytes +=
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
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index) {
        const PaintFieldResource &field = impl_->paint_fields[index];
        if (!field.alive) continue;
        output.allocated_bytes +=
            field.options.vertex_uvs.size * sizeof(Vec2) +
            static_cast<std::size_t>(field.options.width) *
                field.options.height * sizeof(std::uint32_t);
    }
    for (std::uint32_t index = 0;
         index < impl_->options.triangle_mesh_capacity; ++index) {
        if (impl_->meshes[index].alive) {
            output.allocated_bytes +=
                impl_->meshes[index].vertex_count * sizeof(Vec3) +
                impl_->meshes[index].index_count * sizeof(std::uint32_t) +
                impl_->meshes[index].bvh_node_count * sizeof(BvhNode) +
                impl_->meshes[index].bvh_leaf_count * sizeof(std::uint32_t) +
                (impl_->meshes[index].solid_planes != nullptr
                     ? (impl_->meshes[index].index_count / 3U +
                        impl_->meshes[index].solid_unique_plane_count) * sizeof(CollisionPlane) : 0U) +
                (impl_->meshes[index].shell_normals != nullptr
                     ? impl_->meshes[index].index_count / 3U * sizeof(Vec3) : 0U);
        }
    }
    return success();
}

int World::device_ordinal() const noexcept {
    return impl_ ? impl_->device_ordinal : -1;
}

} // namespace parallel_mater
