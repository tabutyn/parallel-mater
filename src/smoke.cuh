// SPDX-License-Identifier: MIT
// Dilute smoke tracers: a potential-flow sphere deflection plus a translating
// alternating vortex street. All integration and emission remain in the API.
struct SmokeStorage {
    SmokeOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *positions{};
    Vec3 *previous_positions{};
    Vec3 *velocities{};
    float *ages{};
    float *thermal_lift{};
    std::uint32_t count{};
    std::uint32_t next_slot{};
    std::uint64_t emitted{};
    float emission_fraction{};
    float time{};
    ~SmokeStorage() {
        release_managed(positions);
        release_managed(previous_positions);
        release_managed(velocities);
        release_managed(ages);
        release_managed(thermal_lift);
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
    Vec3 local_area_vector{};
    std::vector<Vec3> local_triangle_areas{};
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

__device__ Vec3 smoke_velocity_field(Vec3 point, Vec3 center,
    const SmokeOptions &options, float time, Vec3 gravity) {
    const Vec3 relative = subtract(point, center);
    const float radius = options.obstacle_radius;
    const float distance2 = fmaxf(length_squared(relative), radius * radius * 1.001F);
    const float distance = sqrtf(distance2);
    const float ratio = radius * radius * radius / (distance2 * distance);
    // Analytic incompressible potential flow has zero normal velocity at the
    // sphere, and accelerates the stream around its sides.
    Vec3 velocity = subtract(
        multiply(options.wind, 1.0F + 0.5F * ratio),
        multiply(relative, 1.5F * ratio * dot(options.wind, relative) / distance2));
    if (relative.x > 0.15F * radius) {
        const float spacing = 1.25F * radius;
        const float phase = fmodf(fmaxf(time * options.wind.x, 0.0F), spacing);
        const float width = radius;
        for (unsigned vortex = 0U; vortex < 5U; ++vortex) {
            const float sign = vortex & 1U ? -1.0F : 1.0F;
            const float dx = relative.x - (0.75F * radius +
                float(vortex) * spacing + phase);
            const float dy = relative.y - sign * 0.42F * radius;
            const float dz = relative.z - sign * 0.20F * radius;
            const float envelope = expf(-(dx * dx + dy * dy +
                0.4F * dz * dz) / (width * width));
            const float strength = sign * options.wake_strength * envelope / width;
            velocity = add(velocity, {strength * -dy, strength * dx,
                0.38F * strength * -dx});
        }
    }
    velocity = add(velocity, multiply(
        normalized_or(multiply(gravity, -1.0F), {0.0F, 1.0F, 0.0F}),
        options.buoyancy));
    return velocity;
}

__global__ void smoke_advect(Vec3 *positions, Vec3 *previous_positions,
    Vec3 *velocities, float *ages,
    float *thermal_lift,
    std::uint32_t count, SmokeOptions options,
    const RigidBodyState *states, const RigidBodyState *previous_states,
    bool moving_body, std::uint32_t obstacle, float time, float dt,
    Vec3 gravity) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || ages[index] >= options.lifetime) return;
    Vec3 point = positions[index];
    previous_positions[index] = point;
    Vec3 velocity = velocities[index];
    const Vec3 center = states[obstacle].position;
    const Vec3 previous_center = moving_body
        ? previous_states[obstacle].position : center;
    const Vec3 center_path = subtract(center, previous_center);
    const Vec3 obstacle_velocity = multiply(center_path, 1.0F / dt);
    Vec3 desired = smoke_velocity_field(point, center, options, time, gravity);
    desired = add(desired, multiply(
        normalized_or(multiply(gravity, -1.0F), {0.0F, 1.0F, 0.0F}),
        thermal_lift[index]));
    const float response = 1.0F - expf(-options.response * dt);
    velocity = clamp_length(add(velocity,
        multiply(subtract(desired, velocity), response)), options.maximum_speed);
    const Vec3 previous = point;
    point = add(point, multiply(velocity, dt));
    const float clearance = options.obstacle_radius + 0.15F * options.particle_radius;
    const Vec3 path = subtract(point, previous);
    // Solve the swept collision in the obstacle's translating frame.
    const Vec3 start = subtract(previous, previous_center);
    const Vec3 relative_path = subtract(path, center_path);
    const float path2 = length_squared(relative_path);
    const float projection = dot(start, relative_path);
    const float discriminant = projection * projection - path2 *
        (length_squared(start) - clearance * clearance);
    const bool swept_hit = length_squared(start) >= clearance * clearance &&
        path2 > 1.0e-12F && projection < 0.0F && discriminant >= 0.0F &&
        -projection - sqrtf(discriminant) <= path2;
    if (length_squared(subtract(point, center)) < clearance * clearance || swept_hit) {
        float collision_fraction = 1.0F;
        if (swept_hit) {
            collision_fraction = fmaxf(0.0F,
                (-projection - sqrtf(discriminant)) / path2);
            point = add(previous, multiply(path, collision_fraction));
        }
        const Vec3 contact_center = add(previous_center,
            multiply(center_path, collision_fraction));
        const Vec3 normal = normalized_or(subtract(point, contact_center),
            {-1.0F, 0.0F, 0.0F});
        point = add(contact_center, multiply(normal, clearance));
        Vec3 relative_velocity = subtract(velocity, obstacle_velocity);
        relative_velocity = subtract(relative_velocity,
            multiply(normal, fminf(0.0F, dot(relative_velocity, normal))));
        velocity = add(relative_velocity, obstacle_velocity);
        if (swept_hit)
            point = add(point, multiply(velocity,
                dt * (1.0F - collision_fraction)));
        const Vec3 final_separation = subtract(point, center);
        if (length_squared(final_separation) < clearance * clearance)
            point = add(center, multiply(normalized_or(final_separation, normal),
                clearance));
    }
    positions[index] = point;
    velocities[index] = velocity;
    ages[index] += dt;
    thermal_lift[index] *= expf(-dt / 3.0F);
}

__global__ void smoke_emit(Vec3 *positions, Vec3 *previous_positions,
    Vec3 *velocities, float *ages,
    float *thermal_lift,
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
}

__device__ Vec3 smoke_wind_delta(Vec3 position, Vec3 velocity,
    SmokeOptions smoke, Vec3 obstacle_center, float time, float drag,
    float maximum_acceleration, float dt, Vec3 gravity) {
    const Vec3 desired = smoke_velocity_field(
        position, obstacle_center, smoke, time, gravity);
    const float response = 1.0F - expf(-drag * dt);
    return clamp_length(multiply(subtract(desired, velocity), response),
                        maximum_acceleration * dt);
}

__global__ void smoke_soft_body_wind(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    std::uint32_t node_count, SmokeOptions smoke,
    const RigidBodyState *states, std::uint32_t obstacle,
    float time, float drag, float maximum_acceleration, float dt,
    float maximum_speed, Vec3 gravity) {
    const auto node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count || inverse_masses[node] == 0.0F) return;
    velocities[node] = clamp_length(add(velocities[node], smoke_wind_delta(
        positions[node], velocities[node], smoke, states[obstacle].position,
        time, drag, maximum_acceleration, dt, gravity)), maximum_speed);
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
    const RigidBodyState *states, std::uint32_t obstacle,
    float time, float drag, float maximum_acceleration, float dt,
    Vec3 gravity) {
    const auto vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count || inverse_masses[vertex] == 0.0F) return;
    velocities[vertex] = add(velocities[vertex], smoke_wind_delta(
        positions[vertex], velocities[vertex], smoke,
        states[obstacle].position, time, drag, maximum_acceleration,
        dt, gravity));
}

__global__ void smoke_rope_wind(
    RopeData rope, SmokeOptions smoke,
    const RigidBodyState *states, std::uint32_t obstacle,
    float time, float drag, float maximum_acceleration, float dt,
    Vec3 gravity,
    int first, int last) {
    const auto node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= rope.count ||
        rope_anchor_body(rope, node, first, last) >= 0 ||
        rope_soft_anchor(rope, node)) return;
    rope.velocities[node] = clamp_length(add(rope.velocities[node],
        smoke_wind_delta(rope.positions[node], rope.velocities[node], smoke,
            states[obstacle].position, time, drag, maximum_acceleration,
            dt, gravity)), rope.options.maximum_speed);
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
    Vec3 *velocities, const float *ages,
    std::uint32_t count, float lifetime,
    TriangleMeshResource mesh, const RigidBodyState *states,
    const RigidBodyState *previous_states, std::uint32_t body,
    float clearance,
    float edge_flow_speed, float maximum_speed) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count || ages[particle] >= lifetime) return;
    const RigidBodyState state = states[body];
    const RigidBodyState previous_state = previous_states[body];
    const Vec3 before = inverse_rotate(previous_state.orientation,
        subtract(previous_positions[particle], previous_state.position));
    const Vec3 point = inverse_rotate(state.orientation,
        subtract(positions[particle], state.position));
    if (fmaxf(point.x, before.x) < mesh.minimum.x - clearance ||
        fminf(point.x, before.x) > mesh.maximum.x + clearance ||
        fmaxf(point.y, before.y) < mesh.minimum.y - clearance ||
        fminf(point.y, before.y) > mesh.maximum.y + clearance ||
        fmaxf(point.z, before.z) < mesh.minimum.z - clearance ||
        fminf(point.z, before.z) > mesh.maximum.z + clearance) return;
    float nearest2 = clearance * clearance;
    float earliest = 2.0F;
    bool found = false, swept = false;
    Vec3 nearest{}, face_normal{};
    const Vec3 path = subtract(point, before);
    for (std::uint32_t base = 0U; base < mesh.index_count; base += 3U) {
        const Vec3 a = mesh.vertices[mesh.indices[base]];
        const Vec3 b = mesh.vertices[mesh.indices[base + 1U]];
        const Vec3 c = mesh.vertices[mesh.indices[base + 2U]];
        const Vec3 low = component_min(component_min(a, b), c);
        const Vec3 high = component_max(component_max(a, b), c);
        if (fmaxf(point.x, before.x) < low.x - clearance ||
            fminf(point.x, before.x) > high.x + clearance ||
            fmaxf(point.y, before.y) < low.y - clearance ||
            fminf(point.y, before.y) > high.y + clearance ||
            fmaxf(point.z, before.z) < low.z - clearance ||
            fminf(point.z, before.z) > high.z + clearance) continue;
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
                    face_normal = normalized_or(raw_normal, {1.0F, 0.0F, 0.0F});
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
            face_normal = normalized_or(raw_normal, {1.0F, 0.0F, 0.0F});
            found = true;
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
    positions[particle] = add(state.position,
        rotate(state.orientation, add(nearest,
            multiply(local_normal, clearance))));
    const Vec3 surface_velocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    Vec3 relative = subtract(velocities[particle], surface_velocity);
    relative = subtract(relative,
        multiply(normal, fminf(0.0F, dot(relative, normal))));
    const Vec3 from_center = subtract(nearest, mesh.bounding_center);
    const Vec3 projected = subtract(from_center,
        multiply(local_normal, dot(from_center, local_normal)));
    if (length_squared(projected) > 1.0e-8F && edge_flow_speed > 0.0F) {
        const Vec3 toward_edge = rotate(state.orientation,
            normalized_or(projected, {0.0F, 1.0F, 0.0F}));
        relative = add(relative, multiply(toward_edge,
            fmaxf(0.0F, edge_flow_speed - dot(relative, toward_edge))));
        relative = clamp_length(relative, maximum_speed);
    }
    velocities[particle] = add(surface_velocity, relative);
}

__global__ void smoke_cloth_contact(
    Vec3 *positions, const Vec3 *previous_positions,
    Vec3 *velocities, const float *ages,
    std::uint32_t count, float lifetime,
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *indices, std::uint32_t index_count,
    const Vec3 *minimum, const Vec3 *maximum, float clearance,
    float edge_flow_speed, float maximum_speed) {
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
    positions[particle] = add(nearest, multiply(normal, clearance));
    const Vec3 cloth_velocity = add(
        multiply(cloth_velocities[indices[best]], weights.x),
        add(multiply(cloth_velocities[indices[best+1U]], weights.y),
            multiply(cloth_velocities[indices[best+2U]], weights.z)));
    Vec3 relative = subtract(velocities[particle], cloth_velocity);
    relative = subtract(relative,
        multiply(normal, fminf(0.0F, dot(relative, normal))));
    // A no-through response alone leaves a steady carrier wind pushing every
    // tracer back into the same face. Redirect that blocked flow along the
    // local tangent, away from the finite sheet's center, so it can clear an
    // edge. The same rule works on either side and on moving/rotated cloth.
    const Vec3 center = multiply(add(low, high), 0.5F);
    const Vec3 from_center = subtract(nearest, center);
    const Vec3 projected = subtract(from_center,
        multiply(normal, dot(from_center, normal)));
    if (length_squared(projected) > 1.0e-8F && edge_flow_speed > 0.0F) {
        const Vec3 toward_edge = normalized_or(projected, {0.0F, 1.0F, 0.0F});
        const float outward_speed = dot(relative, toward_edge);
        relative = add(relative, multiply(toward_edge,
            fmaxf(0.0F, edge_flow_speed - outward_speed)));
        relative = clamp_length(relative, maximum_speed);
    }
    velocities[particle] = add(cloth_velocity, relative);
}

__global__ void smoke_soft_body_contact(
    Vec3 *positions, Vec3 *velocities, const float *ages,
    std::uint32_t count, float lifetime,
    const Vec3 *surface, const SoftBodySurfaceBinding *bindings,
    std::uint32_t surface_count, const Vec3 *node_velocities,
    const Vec3 *minimum, const Vec3 *maximum, float clearance) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count || ages[particle] >= lifetime) return;
    const Vec3 point = positions[particle];
    const Vec3 low = *minimum, high = *maximum;
    if (point.x < low.x-clearance || point.x > high.x+clearance ||
        point.y < low.y-clearance || point.y > high.y+clearance ||
        point.z < low.z-clearance || point.z > high.z+clearance) return;
    float nearest = clearance * clearance;
    std::uint32_t vertex = surface_count;
    for (std::uint32_t index = 0U; index < surface_count; ++index) {
        const float distance2 = length_squared(subtract(point, surface[index]));
        if (distance2 < nearest) { nearest = distance2; vertex = index; }
    }
    if (vertex == surface_count) return;
    const Vec3 normal = normalized_or(subtract(point, surface[vertex]),
                                       {-1.0F, 0.0F, 0.0F});
    positions[particle] = add(surface[vertex], multiply(normal, clearance));
    const Vec3 body_velocity = node_velocities[bindings[vertex].nodes[0]];
    Vec3 relative = subtract(velocities[particle], body_velocity);
    relative = subtract(relative,
        multiply(normal, fminf(0.0F, dot(relative, normal))));
    velocities[particle] = add(body_velocity, relative);
}
