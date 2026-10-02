// SPDX-License-Identifier: MIT
// Weakly compressible smoke particles: local pressure, viscosity, measured
// vorticity, and triangle-mesh contact. Integration stays in the API.
#include "smoke_grid.cuh"
struct SmokeStorage {
    SmokeOptions options{};
    SmokeGridStorage grid{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *positions{};
    Vec3 *previous_positions{};
    Vec3 *velocities{};
    float *ages{};
    float *thermal_lift{};
    float *number_densities{};
    float *pressures{};
    Vec3 *accelerations{};
    Vec3 *vorticities{};
    float *vorticity_magnitudes{};
    std::uint64_t *keys[2]{};
    std::uint32_t *indices[2]{};
    std::uint8_t *sort_workspace{};
    std::size_t sort_workspace_size{};
    FluidBodyImpulse *rigid_impulses{};
    std::uint32_t count{};
    std::uint32_t next_slot{};
    std::uint64_t emitted{};
    bool index_dirty{};
    float emission_fraction{};
    float time{};
    ~SmokeStorage() {
        release_managed(positions);
        release_managed(previous_positions);
        release_managed(velocities);
        release_managed(ages);
        release_managed(thermal_lift);
        release_managed(number_densities);
        release_managed(pressures);
        release_managed(accelerations);
        release_managed(vorticities);
        release_managed(vorticity_magnitudes);
        for (auto &key : keys) release_managed(key);
        for (auto &index : indices) release_managed(index);
        release_managed(sort_workspace);
        release_managed(rigid_impulses);
    }
};

template <class Options> struct SmokeDeformableCouplingSlot {
    Options options{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *minimum{};
    Vec3 *maximum{};
    ~SmokeDeformableCouplingSlot() {
        release_managed(minimum);
        release_managed(maximum);
    }
};

using SmokeSoftBodyCouplingSlot =
    SmokeDeformableCouplingSlot<SmokeSoftBodyCouplingOptions>;
using SmokeClothCouplingSlot =
    SmokeDeformableCouplingSlot<SmokeClothCouplingOptions>;
using SmokeRopeCouplingSlot =
    SmokeDeformableCouplingSlot<SmokeRopeCouplingOptions>;
struct SmokeRigidCouplingSlot {
    SmokeRigidCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};

__host__ __device__ std::uint32_t smoke_hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

__device__ float smoke_random(std::uint32_t seed) {
    return float(smoke_hash(seed) & 0x00ffffffU) / 16777216.0F;
}

__global__ void smoke_emit_cells(const Vec3 *positions, const float *ages,
    std::uint32_t count, SmokeOptions options, std::uint64_t *keys,
    std::uint32_t *indices) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= options.capacity) return;
    indices[particle] = particle;
    if (particle >= count || ages[particle] >= options.lifetime) {
        keys[particle] = k_fluid_empty_cell;
        return;
    }
    const float inverse = 1.0F / (3.0F * options.particle_radius);
    const Vec3 point = positions[particle];
    keys[particle] = fluid_cell_key(
        __float2int_rd(point.x * inverse),
        __float2int_rd(point.y * inverse),
        __float2int_rd(point.z * inverse));
}

// Compact positive kernel: density is local occupancy in support volumes.
// The same sorted grid serves particle pressure and local structure coupling.
struct SmokeFlowSample { Vec3 velocity{}; float number_density{}; };

__device__ SmokeFlowSample smoke_sample_flow(Vec3 point,
    const Vec3 *positions, const Vec3 *velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    SmokeOptions options) {
    SmokeFlowSample sample{};
    const float support = 3.0F * options.particle_radius;
    const float inverse = 1.0F / support;
    const int cx = __float2int_rd(point.x * inverse);
    const int cy = __float2int_rd(point.y * inverse);
    const int cz = __float2int_rd(point.z * inverse);
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const auto key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (std::uint32_t item = fluid_lower_bound(
                         keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item) {
                    const auto other = indices[item];
                    const float distance = vector_length(subtract(
                        point, positions[other]));
                    if (distance >= support) continue;
                    const float q = 1.0F - distance * inverse;
                    const float weight = q * q * q;
                    sample.number_density += weight;
                    sample.velocity = add(sample.velocity,
                        multiply(velocities[other], weight));
                }
            }
    if (sample.number_density > 1.0e-6F)
        sample.velocity = multiply(sample.velocity,
            1.0F / sample.number_density);
    return sample;
}

__global__ void smoke_density_pressure(const Vec3 *positions,
    const float *ages, std::uint32_t count, SmokeOptions options,
    const std::uint64_t *keys, const std::uint32_t *indices,
    float *densities, float *pressures, float dt) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    if (ages[particle] >= options.lifetime) {
        densities[particle] = pressures[particle] = 0.0F;
        return;
    }
    const SmokeFlowSample local = smoke_sample_flow(positions[particle],
        positions, positions, keys, indices, options);
    densities[particle] = local.number_density;
    const float crowding = options.pressure_stiffness * fmaxf(0.0F,
        local.number_density / options.rest_number_density - 1.0F);
    // Contact-generated stagnation pressure propagates through neighboring
    // particles for a few frames, then decays in the absence of new impacts.
    pressures[particle] = fminf(100.0F, crowding +
        pressures[particle] * expf(-dt / 0.05F));
}

__global__ void smoke_pair_forces(const Vec3 *positions,
    const Vec3 *velocities, const float *ages, std::uint32_t count,
    SmokeOptions options, const std::uint64_t *keys,
    const std::uint32_t *indices, const float *densities,
    const float *pressures, const Vec3 *vorticities,
    const float *vorticity_magnitudes, Vec3 *accelerations) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    accelerations[particle] = {};
    if (ages[particle] >= options.lifetime) return;
    const Vec3 point = positions[particle];
    const Vec3 velocity = velocities[particle];
    const float support = 3.0F * options.particle_radius;
    const float inverse = 1.0F / support;
    const int cx = __float2int_rd(point.x * inverse);
    const int cy = __float2int_rd(point.y * inverse);
    const int cz = __float2int_rd(point.z * inverse);
    Vec3 acceleration{};
    Vec3 confinement_gradient{};
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const auto key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (std::uint32_t item = fluid_lower_bound(
                         keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item) {
                    const auto other = indices[item];
                    if (other == particle) continue;
                    const Vec3 delta = subtract(point, positions[other]);
                    const float distance2 = length_squared(delta);
                    if (distance2 >= support * support ||
                        distance2 < 1.0e-12F) continue;
                    const float distance = sqrtf(distance2);
                    const float q = 1.0F - distance * inverse;
                    const float pair_density = fmaxf(1.0F,
                        sqrtf(densities[particle] * densities[other]));
                    const float pressure = (pressures[particle] +
                        pressures[other]) * 0.5F;
                    acceleration = add(acceleration, multiply(delta,
                        3.0F * pressure * q * q /
                        (support * distance * pair_density)));
                    acceleration = add(acceleration, multiply(
                        subtract(velocities[other], velocity),
                        options.viscosity * q * q / pair_density));
                    if (options.vorticity_confinement > 0.0F) {
                        const Vec3 grad = multiply(delta,
                            -3.0F * q * q /
                            (support * distance * pair_density));
                        confinement_gradient = add(confinement_gradient,
                            multiply(grad, vorticity_magnitudes[other] -
                                vorticity_magnitudes[particle]));
                    }
                }
            }
    if (length_squared(confinement_gradient) > 1.0e-10F)
        acceleration = add(acceleration, multiply(cross(
            normalized_or(confinement_gradient, {1.0F, 0.0F, 0.0F}),
            vorticities[particle]),
            options.vorticity_confinement * support));
    accelerations[particle] = clamp_length(acceleration, 50.0F);
}

__global__ void smoke_compute_vorticity(const Vec3 *positions,
    const Vec3 *velocities, const float *ages, std::uint32_t count,
    SmokeOptions options, const std::uint64_t *keys,
    const std::uint32_t *indices, const float *densities,
    Vec3 *vorticities, float *magnitudes) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    vorticities[particle] = {};
    magnitudes[particle] = 0.0F;
    if (ages[particle] >= options.lifetime ||
        options.vorticity_confinement == 0.0F) return;
    const Vec3 point = positions[particle];
    const float support = 3.0F * options.particle_radius;
    const float inverse = 1.0F / support;
    const int cx = __float2int_rd(point.x * inverse);
    const int cy = __float2int_rd(point.y * inverse);
    const int cz = __float2int_rd(point.z * inverse);
    Vec3 curl{};
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const auto key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (std::uint32_t item = fluid_lower_bound(
                         keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item) {
                    const auto other = indices[item];
                    if (other == particle) continue;
                    const Vec3 delta = subtract(point, positions[other]);
                    const float distance2 = length_squared(delta);
                    if (distance2 >= support * support ||
                        distance2 < 1.0e-12F) continue;
                    const float distance = sqrtf(distance2);
                    const float q = 1.0F - distance * inverse;
                    const float pair_density = fmaxf(1.0F,
                        sqrtf(densities[particle] * densities[other]));
                    const Vec3 gradient = multiply(delta,
                        -3.0F * q * q /
                        (support * distance * pair_density));
                    curl = add(curl, cross(subtract(
                        velocities[other], velocities[particle]), gradient));
                }
            }
    vorticities[particle] = curl;
    magnitudes[particle] = vector_length(curl);
}

__global__ void smoke_advect(Vec3 *positions, Vec3 *previous_positions,
    Vec3 *velocities, float *ages,
    float *thermal_lift,
    std::uint32_t count, SmokeOptions options,
    const Vec3 *accelerations, float dt, Vec3 gravity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || ages[index] >= options.lifetime) return;
    Vec3 point = positions[index];
    previous_positions[index] = point;
    Vec3 velocity = velocities[index];
    const Vec3 lift = multiply(
        normalized_or(multiply(gravity, -1.0F), {0.0F, 1.0F, 0.0F}),
        options.buoyancy + thermal_lift[index]);
    velocity = add(velocity, multiply(add(accelerations[index], lift), dt));
    const float response = 1.0F - expf(-options.response * dt);
    velocity = clamp_length(add(velocity, multiply(
        subtract(options.wind, velocity), response)), options.maximum_speed);
    point = add(point, multiply(velocity, dt));
    positions[index] = point;
    velocities[index] = velocity;
    ages[index] += dt;
    thermal_lift[index] *= expf(-dt / 3.0F);
}

__global__ void smoke_emit(Vec3 *positions, Vec3 *previous_positions,
    Vec3 *velocities, float *ages,
    float *thermal_lift, float *densities, float *pressures,
    Vec3 *vorticities,
    SmokeOptions options, std::uint32_t first_slot,
    std::uint32_t count, std::uint64_t first_serial) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t slot = (first_slot + index) % options.capacity;
    const std::uint32_t serial = std::uint32_t(first_serial + index);
    const float y = 2.0F * smoke_random(serial ^ 0x132aef41U) - 1.0F;
    const float z = 2.0F * smoke_random(serial ^ 0xa385c9d3U) - 1.0F;
    positions[slot] = add(options.emitter_center, {0.0F,
        y * options.emitter_half_extents.x,
        z * options.emitter_half_extents.y});
    previous_positions[slot] = positions[slot];
    velocities[slot] = options.initial_velocity;
    ages[slot] = 0.0F;
    thermal_lift[slot] = 0.0F;
    densities[slot] = 0.0F;
    pressures[slot] = 0.0F;
    vorticities[slot] = {};
}

__device__ Vec3 smoke_wind_delta(Vec3 position, Vec3 velocity,
    const Vec3 *smoke_positions, const Vec3 *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    SmokeOptions smoke, SmokeGridField grid,
    float drag, float maximum_acceleration, float dt) {
    if (grid.n != 0U && smoke_grid_contains(position, grid)) {
        Vec3 air{};
        float density{};
        smoke_grid_sample(position, grid, air, density);
        const float response = 1.0F - expf(-drag *
            clamp_scalar(density*smoke.rest_number_density,0.0F,1.0F)*dt);
        return clamp_length(multiply(subtract(air, velocity), response),
                            maximum_acceleration * dt);
    }
    const SmokeFlowSample local = smoke_sample_flow(position, smoke_positions,
        smoke_velocities, keys, indices, smoke);
    if (local.number_density <= 1.0e-6F) return {};
    const float occupancy = fminf(1.0F,
        local.number_density / smoke.rest_number_density);
    const float response = 1.0F - expf(-drag * occupancy * dt);
    return clamp_length(multiply(subtract(local.velocity, velocity), response),
                        maximum_acceleration * dt);
}

__global__ void smoke_soft_body_wind(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    std::uint32_t node_count, SmokeOptions smoke,
    const Vec3 *smoke_positions, const Vec3 *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    SmokeGridField grid,
    float drag, float maximum_acceleration, float dt,
    float maximum_speed) {
    const auto node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count || inverse_masses[node] == 0.0F) return;
    velocities[node] = clamp_length(add(velocities[node], smoke_wind_delta(
        positions[node], velocities[node], smoke_positions, smoke_velocities,
        keys, indices, smoke, grid, drag, maximum_acceleration, dt)), maximum_speed);
}

__global__ void smoke_deformable_bounds(
    const Vec3 *surface, std::uint32_t count, Vec3 *minimum, Vec3 *maximum) {
    if (blockIdx.x || threadIdx.x) return;
    Vec3 low = surface[0], high = surface[0];
    for (std::uint32_t index = 1U; index < count; ++index) {
        const Vec3 point = surface[index];
        low.x = fminf(low.x, point.x); low.y = fminf(low.y, point.y);
        low.z = fminf(low.z, point.z);
        high.x = fmaxf(high.x, point.x); high.y = fmaxf(high.y, point.y);
        high.z = fmaxf(high.z, point.z);
    }
    *minimum = low;
    *maximum = high;
}

__global__ void smoke_cloth_wind(
    const Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    std::uint32_t vertex_count, SmokeOptions smoke,
    const Vec3 *smoke_positions, const Vec3 *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    SmokeGridField grid, float drag, float maximum_acceleration, float dt) {
    const auto vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count || inverse_masses[vertex] == 0.0F) return;
    velocities[vertex] = add(velocities[vertex], smoke_wind_delta(
        positions[vertex], velocities[vertex], smoke_positions,
        smoke_velocities, keys, indices, smoke, grid, drag,
        maximum_acceleration, dt));
}

__global__ void smoke_rope_wind(
    RopeData rope, SmokeOptions smoke,
    const Vec3 *smoke_positions, const Vec3 *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    SmokeGridField grid, float drag, float maximum_acceleration, float dt,
    int first, int last) {
    const auto node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= rope.count ||
        rope_anchor_body(rope, node, first, last) >= 0 ||
        rope_soft_anchor(rope, node)) return;
    rope.velocities[node] = clamp_length(add(rope.velocities[node],
        smoke_wind_delta(rope.positions[node], rope.velocities[node],
            smoke_positions, smoke_velocities, keys, indices, smoke,
            grid, drag, maximum_acceleration, dt)), rope.options.maximum_speed);
}

__global__ void smoke_rope_contact(
    Vec3 *positions, const Vec3 *previous_positions,
    Vec3 *velocities, const float *ages,
    std::uint32_t count, float lifetime, RopeData rope,
    const Vec3 *minimum, const Vec3 *maximum, float clearance) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count || ages[particle] >= lifetime) return;
    const Vec3 point = positions[particle];
    const Vec3 before = previous_positions[particle];
    const Vec3 low = *minimum, high = *maximum;
    if (fmaxf(point.x, before.x) < low.x - clearance ||
        fminf(point.x, before.x) > high.x + clearance ||
        fmaxf(point.y, before.y) < low.y - clearance ||
        fminf(point.y, before.y) > high.y + clearance ||
        fmaxf(point.z, before.z) < low.z - clearance ||
        fminf(point.z, before.z) > high.z + clearance) return;
    float earliest = 2.0F, nearest2 = clearance * clearance;
    std::uint32_t segment = rope.count;
    Vec3 smoke_hit{}, rope_hit{};
    const Vec3 path = subtract(point, before);
    const float path2 = length_squared(path);
    for (std::uint32_t index = 0U; index + 1U < rope.count; ++index) {
        const Vec3 a = rope.positions[index], b = rope.positions[index + 1U];
        if (fmaxf(point.x, before.x) < fminf(a.x, b.x) - clearance ||
            fminf(point.x, before.x) > fmaxf(a.x, b.x) + clearance ||
            fmaxf(point.y, before.y) < fminf(a.y, b.y) - clearance ||
            fminf(point.y, before.y) > fmaxf(a.y, b.y) + clearance ||
            fmaxf(point.z, before.z) < fminf(a.z, b.z) - clearance ||
            fminf(point.z, before.z) > fmaxf(a.z, b.z) + clearance) continue;
        Vec3 on_path{}, on_rope{};
        closest_segments(before, point, a, b, on_path, on_rope);
        const float distance2 = length_squared(subtract(on_path, on_rope));
        if (distance2 >= clearance * clearance) continue;
        const float fraction = path2 > 1.0e-12F
            ? clamp_scalar(dot(subtract(on_path, before), path) / path2, 0.0F, 1.0F)
            : 0.0F;
        if (fraction > earliest + 1.0e-6F ||
            (fabsf(fraction - earliest) <= 1.0e-6F && distance2 >= nearest2))
            continue;
        earliest = fraction;
        nearest2 = distance2;
        segment = index;
        smoke_hit = on_path;
        rope_hit = on_rope;
    }
    if (segment == rope.count) return;
    const Vec3 a = rope.positions[segment], b = rope.positions[segment + 1U];
    const Vec3 axis = subtract(b, a);
    const float fraction = clamp_scalar(dot(subtract(rope_hit, a), axis) /
        fmaxf(length_squared(axis), 1.0e-12F), 0.0F, 1.0F);
    const Vec3 rope_velocity = add(
        multiply(rope.velocities[segment], 1.0F - fraction),
        multiply(rope.velocities[segment + 1U], fraction));
    Vec3 normal = subtract(smoke_hit, rope_hit);
    if (length_squared(normal) < 1.0e-10F)
        normal = subtract(before, rope_hit);
    if (length_squared(normal) < 1.0e-10F) {
        const Vec3 relative = subtract(velocities[particle], rope_velocity);
        normal = subtract(multiply(relative, -1.0F),
            multiply(axis, -dot(relative, axis) /
                fmaxf(length_squared(axis), 1.0e-12F)));
    }
    normal = normalized_or(normal, {0.0F, 1.0F, 0.0F});
    positions[particle] = add(rope_hit, multiply(normal, clearance));
    Vec3 relative = subtract(velocities[particle], rope_velocity);
    relative = subtract(relative,
        multiply(normal, fminf(0.0F, dot(relative, normal))));
    velocities[particle] = add(rope_velocity, relative);
}

__global__ void smoke_rigid_contact(
    Vec3 *positions, const Vec3 *previous_positions,
    Vec3 *velocities, const float *ages, float *pressures,
    std::uint32_t count, float lifetime,
    TriangleMeshResource mesh, const RigidBodyState *states,
    const RigidBodyState *previous_states, std::uint32_t body,
    float clearance, float particle_mass, float wall_drag,
    float pressure_stiffness,
    float dt, float maximum_speed, SmokeGridField grid,
    FluidBodyImpulse *impulses) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count) return;
    impulses[particle] = {};
    if (ages[particle] >= lifetime) return;
    const float boundary = 3.0F * clearance;
    const RigidBodyState state = states[body];
    const RigidBodyState previous_state = previous_states[body];
    const Vec3 before = inverse_rotate(previous_state.orientation,
        subtract(previous_positions[particle], previous_state.position));
    const Vec3 point = inverse_rotate(state.orientation,
        subtract(positions[particle], state.position));
    if (fmaxf(point.x, before.x) < mesh.minimum.x - boundary ||
        fminf(point.x, before.x) > mesh.maximum.x + boundary ||
        fmaxf(point.y, before.y) < mesh.minimum.y - boundary ||
        fminf(point.y, before.y) > mesh.maximum.y + boundary ||
        fmaxf(point.z, before.z) < mesh.minimum.z - boundary ||
        fminf(point.z, before.z) > mesh.maximum.z + boundary) return;
    float nearest2 = boundary * boundary;
    float earliest = 2.0F;
    bool found = false, swept = false;
    Vec3 nearest{}, face_normal{};
    const Vec3 path = subtract(point, before);
    std::uint32_t stack[64]{0U};
    int pending = mesh.bvh_node_count == 0U ? 0 : 1;
    while (pending != 0) {
        const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
        if (!fluid_segment_bounds(before, point, node, boundary)) continue;
        if (node.triangle_count == 0U) {
            if (pending + 2 > 64) continue;
            stack[pending++] = node.right;
            stack[pending++] = node.left;
            continue;
        }
        for (std::uint32_t item = 0U; item < node.triangle_count; ++item) {
            const auto base = (node.first_triangle + item) * 3U;
            const Vec3 a = mesh.vertices[mesh.indices[base]];
            const Vec3 b = mesh.vertices[mesh.indices[base + 1U]];
            const Vec3 c = mesh.vertices[mesh.indices[base + 2U]];
            const Vec3 raw_normal = cross(subtract(b, a), subtract(c, a));
            if (length_squared(raw_normal) < 1.0e-12F) continue;
            const float side_before = dot(subtract(before, a), raw_normal);
            const float side_after = dot(subtract(point, a), raw_normal);
            if (side_before * side_after < 0.0F) {
                const float fraction = side_before / (side_before - side_after);
                if (fraction < earliest) {
                    const Vec3 hit = add(before, multiply(path, fraction));
                    Vec3 weights{};
                    const Vec3 on_face = fluid_closest_triangle_barycentric(
                        hit, a, b, c, weights);
                    if (length_squared(subtract(hit, on_face)) < 1.0e-8F) {
                        earliest = fraction;
                        nearest = on_face;
                        face_normal = normalized_or(raw_normal,
                                                    {1.0F, 0.0F, 0.0F});
                        swept = true;
                        found = true;
                    }
                }
            }
            if (swept) continue;
            Vec3 weights{};
            const Vec3 candidate = fluid_closest_triangle_barycentric(
                point, a, b, c, weights);
            const float distance2 = length_squared(subtract(point, candidate));
            if (distance2 < nearest2) {
                nearest2 = distance2;
                nearest = candidate;
                face_normal = normalized_or(raw_normal,
                                            {1.0F, 0.0F, 0.0F});
                found = true;
            }
        }
    }
    if (!found) return;
    float side = dot(subtract(before, nearest), face_normal);
    if (fabsf(side) < 1.0e-5F)
        side = dot(subtract(point, nearest), face_normal);
    const Vec3 local_normal = side >= 0.0F
        ? face_normal : multiply(face_normal, -1.0F);
    const Vec3 normal = rotate(state.orientation, local_normal);
    const Vec3 arm = rotate(state.orientation, nearest);
    const bool touching = swept || nearest2 < clearance * clearance;
    const Vec3 surface_point=add(state.position,arm);
    const Vec3 trace_position=add(surface_point,multiply(normal,
        grid.n!=0U?1.5F*grid.spacing:clearance));
    if(touching)positions[particle]=trace_position;
    const Vec3 surface_velocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    const Vec3 old_velocity = velocities[particle];
    const Vec3 tracer_velocity=grid.n!=0U&&touching&&
        smoke_grid_contains(trace_position,grid)?
        smoke_grid_sample_velocity(trace_position,grid):old_velocity;
    Vec3 relative = subtract(tracer_velocity, surface_velocity);
    const float normal_inflow = fmaxf(0.0F, -dot(relative, normal));
    if (touching)
        relative = subtract(relative,
            multiply(normal, fminf(0.0F, dot(relative, normal))));
    if(grid.n!=0U&&touching){
        // Contact is only a safeguard for interpolation error in grid mode.
        // Redirect the penetrative component along the resolved streamline
        // instead of numerically deleting tracer speed at a thin surface.
        const float source_speed=fmaxf(vector_length(subtract(
            tracer_velocity,surface_velocity)),vector_length(subtract(
            old_velocity,surface_velocity)));
        const float tangent_speed=vector_length(relative);
        if(tangent_speed>1.0e-6F&&tangent_speed<source_speed)
            relative=multiply(relative,source_speed/tangent_speed);
    }
    const float distance = touching ? clearance : sqrtf(nearest2);
    const float q = fmaxf(0.0F, 1.0F - distance / boundary);
    // No-slip applies at the surface, not at a particle center one radius
    // away. Relax tangential velocity over time without freezing a particle
    // on every repeated contact.
    if(grid.n==0U){
        const float response=1.0F-expf(-wall_drag*q*q*dt);
        relative=multiply(relative,1.0F-response);
    }
    if (touching && grid.n == 0U)
        pressures[particle] = fmaxf(pressures[particle],
            10.0F * pressure_stiffness * normal_inflow * normal_inflow);
    const Vec3 new_velocity = clamp_length(
        add(surface_velocity, relative), maximum_speed);
    velocities[particle] = new_velocity;
    const Vec3 reaction = multiply(subtract(old_velocity, new_velocity),
                                   particle_mass);
    impulses[particle] = {reaction, cross(arm, reaction), body};
}

__global__ void smoke_apply_rigid_impulses(
    const FluidBodyImpulse *impulses, std::uint32_t count,
    std::uint32_t body, const BodyParameters *parameters,
    RigidBodyState *states) {
    if (blockIdx.x != 0U || parameters[body].motion != MotionType::dynamic)
        return;
    __shared__ Vec3 linear[128], angular[128];
    Vec3 local_linear{}, local_angular{};
    for (std::uint32_t particle = threadIdx.x; particle < count;
         particle += blockDim.x) {
        const FluidBodyImpulse item = impulses[particle];
        if (item.body != body) continue;
        local_linear = add(local_linear, item.linear);
        local_angular = add(local_angular, item.angular);
    }
    linear[threadIdx.x] = local_linear;
    angular[threadIdx.x] = local_angular;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
        if (threadIdx.x < stride) {
            linear[threadIdx.x] = add(linear[threadIdx.x],
                linear[threadIdx.x + stride]);
            angular[threadIdx.x] = add(angular[threadIdx.x],
                angular[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0U) return;
    RigidBodyState state = states[body];
    const BodyParameters settings = parameters[body];
    state.linear_velocity = clamp_length(add(state.linear_velocity,
        multiply(linear[0], settings.inverse_mass)),
        settings.maximum_linear_speed);
    state.angular_velocity = clamp_length(add(state.angular_velocity,
        inverse_inertia_world(settings, state, angular[0])),
        settings.maximum_angular_speed);
    states[body] = state;
}

__global__ void smoke_cloth_contact(
    Vec3 *positions, const Vec3 *previous_positions,
    Vec3 *velocities, const float *ages, float *pressures,
    std::uint32_t count, float lifetime,
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *indices, std::uint32_t index_count,
    const Vec3 *minimum, const Vec3 *maximum, float clearance,
    float pressure_stiffness, float maximum_speed, SmokeGridField grid) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count || ages[particle] >= lifetime) return;
    const Vec3 point = positions[particle];
    const Vec3 before = previous_positions[particle];
    const Vec3 low = *minimum, high = *maximum;
    if (fmaxf(point.x, before.x) < low.x-clearance ||
        fminf(point.x, before.x) > high.x+clearance ||
        fmaxf(point.y, before.y) < low.y-clearance ||
        fminf(point.y, before.y) > high.y+clearance ||
        fmaxf(point.z, before.z) < low.z-clearance ||
        fminf(point.z, before.z) > high.z+clearance) return;
    float nearest2 = clearance * clearance;
    std::uint32_t best = index_count;
    Vec3 nearest{}, face_normal{}, weights{};
    float earliest = 2.0F;
    std::uint32_t swept_best = index_count;
    Vec3 swept_nearest{}, swept_normal{}, swept_weights{};
    for (std::uint32_t base = 0U; base < index_count; base += 3U) {
        const Vec3 a = cloth_positions[indices[base]];
        const Vec3 b = cloth_positions[indices[base+1U]];
        const Vec3 c = cloth_positions[indices[base+2U]];
        const Vec3 normal = cross(subtract(b, a), subtract(c, a));
        if (length_squared(normal) < 1.0e-12F) continue;
        const float side_before = dot(subtract(before, a), normal);
        const float side_after = dot(subtract(point, a), normal);
        if (side_before * side_after < 0.0F) {
            const float fraction = side_before / (side_before - side_after);
            if (fraction < earliest) {
                const Vec3 hit = add(before,
                    multiply(subtract(point, before), fraction));
                Vec3 hit_weights{};
                const Vec3 hit_nearest = fluid_closest_triangle_barycentric(
                    hit, a, b, c, hit_weights);
                if (length_squared(subtract(hit, hit_nearest)) < 1.0e-8F) {
                    earliest = fraction;
                    swept_best = base;
                    swept_nearest = hit_nearest;
                    swept_normal = normalized_or(normal, {1.0F, 0.0F, 0.0F});
                    swept_weights = hit_weights;
                }
            }
        }
        Vec3 barycentric{};
        const Vec3 candidate = fluid_closest_triangle_barycentric(
            point, a, b, c, barycentric);
        const float distance2 = length_squared(subtract(point, candidate));
        if (distance2 < nearest2) {
            nearest2 = distance2;
            best = base;
            nearest = candidate;
            face_normal = normalized_or(normal, {1.0F, 0.0F, 0.0F});
            weights = barycentric;
        }
    }
    if (swept_best != index_count) {
        best = swept_best;
        nearest = swept_nearest;
        face_normal = swept_normal;
        weights = swept_weights;
    }
    if (best == index_count) return;
    // Restore the side occupied before this step, even when the tracer
    // crossed a thin moving sheet during advection.
    float side = dot(subtract(before, nearest), face_normal);
    if (fabsf(side) < 1.0e-5F)
        side = dot(subtract(point, nearest), face_normal);
    const Vec3 normal = side >= 0.0F ? face_normal : multiply(face_normal, -1.0F);
    const Vec3 trace_position=add(nearest,multiply(normal,
        grid.n!=0U?1.5F*grid.spacing:clearance));
    positions[particle]=trace_position;
    const Vec3 cloth_velocity = add(
        multiply(cloth_velocities[indices[best]], weights.x),
        add(multiply(cloth_velocities[indices[best+1U]], weights.y),
            multiply(cloth_velocities[indices[best+2U]], weights.z)));
    const Vec3 incoming = subtract(grid.n!=0U&&
        smoke_grid_contains(trace_position,grid)?
        smoke_grid_sample_velocity(trace_position,grid):velocities[particle],
        cloth_velocity);
    Vec3 relative = incoming;
    const float normal_inflow = fmaxf(0.0F, -dot(relative, normal));
    relative = subtract(relative,
        multiply(normal, fminf(0.0F, dot(relative, normal))));
    if(grid.n!=0U){
        const float source_speed=fmaxf(vector_length(incoming),vector_length(
            subtract(velocities[particle],cloth_velocity)));
        const float tangent_speed=vector_length(relative);
        if(tangent_speed>1.0e-6F&&tangent_speed<source_speed)
            relative=multiply(relative,source_speed/tangent_speed);
    }
    if(grid.n==0U){
        pressures[particle]=fmaxf(pressures[particle],
            5.0F*pressure_stiffness*normal_inflow*normal_inflow);
        // Particle-only mode retains its finite-sheet pressure release.
        const Vec3 center=multiply(add(low,high),0.5F);
        const Vec3 offset=subtract(nearest,center);
        const Vec3 tangent=subtract(offset,multiply(normal,dot(offset,normal)));
        if(pressures[particle]>0.1F&&length_squared(tangent)>1.0e-10F){
            const Vec3 outward=normalized_or(tangent,{0.0F,1.0F,0.0F});
            const float current=dot(relative,outward);
            const float target=fminf(maximum_speed,sqrtf(
                length_squared(incoming)+0.5F*pressures[particle]));
            relative=add(relative,multiply(outward,
                fmaxf(0.0F,target-current)));
        }
    }
    velocities[particle] = add(cloth_velocity,
        clamp_length(relative, maximum_speed));
}

__global__ void smoke_soft_body_contact(
    Vec3 *positions, const Vec3 *previous_positions,
    Vec3 *velocities, const float *ages,
    std::uint32_t count, float lifetime,
    const Vec3 *surface, const SoftBodySurfaceBinding *bindings,
    const std::uint32_t *indices,std::uint32_t index_count,
    const Vec3 *node_velocities,
    const Vec3 *minimum, const Vec3 *maximum, float clearance,
    SmokeGridField grid) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count || ages[particle] >= lifetime) return;
    const Vec3 point = positions[particle];
    const Vec3 before = previous_positions[particle];
    const Vec3 low = *minimum, high = *maximum;
    if(fmaxf(point.x,before.x)<low.x-clearance||
       fminf(point.x,before.x)>high.x+clearance||
       fmaxf(point.y,before.y)<low.y-clearance||
       fminf(point.y,before.y)>high.y+clearance||
       fmaxf(point.z,before.z)<low.z-clearance||
       fminf(point.z,before.z)>high.z+clearance)return;
    float nearest2=clearance*clearance,earliest=2.0F;
    std::uint32_t best=index_count;
    Vec3 nearest{},face_normal{},weights{};
    const Vec3 path=subtract(point,before);
    for(std::uint32_t base=0;base<index_count;base+=3U){
        const Vec3 a=surface[indices[base]],b=surface[indices[base+1U]];
        const Vec3 c=surface[indices[base+2U]];
        const Vec3 raw_normal=cross(subtract(b,a),subtract(c,a));
        if(length_squared(raw_normal)<1.0e-12F)continue;
        const float side_before=dot(subtract(before,a),raw_normal);
        const float side_after=dot(subtract(point,a),raw_normal);
        if(side_before*side_after<0.0F){
            const float fraction=side_before/(side_before-side_after);
            if(fraction<earliest){
                const Vec3 hit=add(before,multiply(path,fraction));
                Vec3 hit_weights{};
                const Vec3 on_face=fluid_closest_triangle_barycentric(
                    hit,a,b,c,hit_weights);
                if(length_squared(subtract(hit,on_face))<1.0e-8F){
                    earliest=fraction;best=base;nearest=on_face;
                    face_normal=normalized_or(raw_normal,{1,0,0});
                    weights=hit_weights;
                }
            }
        }
        if(earliest<=1.0F)continue;
        Vec3 barycentric{};
        const Vec3 candidate=fluid_closest_triangle_barycentric(
            point,a,b,c,barycentric);
        const float distance2=length_squared(subtract(point,candidate));
        if(distance2<nearest2){
            nearest2=distance2;best=base;nearest=candidate;
            face_normal=normalized_or(raw_normal,{1,0,0});weights=barycentric;
        }
    }
    if(best==index_count)return;
    float side=dot(subtract(before,nearest),face_normal);
    if(fabsf(side)<1.0e-5F)side=dot(subtract(point,nearest),face_normal);
    const Vec3 normal=side>=0.0F?face_normal:multiply(face_normal,-1.0F);
    const Vec3 trace_position=add(nearest,multiply(normal,
        grid.n!=0U?1.5F*grid.spacing:clearance));
    positions[particle]=trace_position;
    const Vec3 va=smoke_soft_surface_velocity(
        bindings[indices[best]],node_velocities);
    const Vec3 vb=smoke_soft_surface_velocity(
        bindings[indices[best+1U]],node_velocities);
    const Vec3 vc=smoke_soft_surface_velocity(
        bindings[indices[best+2U]],node_velocities);
    const Vec3 body_velocity=add(multiply(va,weights.x),
        add(multiply(vb,weights.y),multiply(vc,weights.z)));
    Vec3 relative=subtract(grid.n!=0U&&smoke_grid_contains(trace_position,grid)?
        smoke_grid_sample_velocity(trace_position,grid):velocities[particle],
        body_velocity);
    const float source_speed=fmaxf(vector_length(relative),vector_length(
        subtract(velocities[particle],body_velocity)));
    relative=subtract(relative,multiply(normal,
        fminf(0.0F,dot(relative,normal))));
    const float tangent_speed=vector_length(relative);
    if(grid.n!=0U&&tangent_speed>1.0e-6F&&tangent_speed<source_speed)
        relative=multiply(relative,source_speed/tangent_speed);
    velocities[particle]=add(body_velocity,relative);
}
