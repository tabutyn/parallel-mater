// SPDX-License-Identifier: MIT
// Included after RopeData and the shared vector/contact helpers in world.cu.
struct FluidRopeCouplingStorage {
    FluidRopeCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *node_impulses{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    void release() noexcept {
        release_managed(node_impulses);
        release_managed(contact_count);
        release_managed(maximum_penetration);
    }
    ~FluidRopeCouplingStorage() { release(); }
};

__global__ void fluid_rope_contacts(
    Vec3 *positions, Vec3 *velocities, const std::uint32_t *particle_count,
    float particle_mass, RopeData rope, float distance, float friction,
    Vec3 *node_impulses, std::uint32_t *contact_count,
    float *maximum_penetration, bool transfer_impulse) {
    const unsigned particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *particle_count) return;
    const Vec3 position = positions[particle];
    float nearest = distance * distance;
    unsigned segment = rope.count;
    float fraction = 0.0F;
    Vec3 closest{};
    // The tight segment AABB eliminates most distance calculations even for a
    // long rope, without a per-frame acceleration structure or host readback.
    for (unsigned i = 0; i + 1 < rope.count; ++i) {
        const Vec3 a = rope.positions[i], b = rope.positions[i + 1];
        if (position.x < fminf(a.x, b.x) - distance ||
            position.x > fmaxf(a.x, b.x) + distance ||
            position.y < fminf(a.y, b.y) - distance ||
            position.y > fmaxf(a.y, b.y) + distance ||
            position.z < fminf(a.z, b.z) - distance ||
            position.z > fmaxf(a.z, b.z) + distance) continue;
        const Vec3 ab = subtract(b, a);
        const float t = clamp_scalar(dot(subtract(position, a), ab) /
            fmaxf(length_squared(ab), 1.0e-12F), 0.0F, 1.0F);
        const Vec3 point = add(a, multiply(ab, t));
        const float squared = length_squared(subtract(position, point));
        if (squared >= nearest) continue;
        nearest = squared;
        segment = i;
        fraction = t;
        closest = point;
    }
    if (segment == rope.count) return;
    const float separation = sqrtf(nearest);
    const Vec3 rope_velocity = add(multiply(rope.velocities[segment], 1.0F - fraction),
                                   multiply(rope.velocities[segment + 1], fraction));
    const Vec3 relative = subtract(velocities[particle], rope_velocity);
    const Vec3 normal = separation > 1.0e-6F
        ? multiply(subtract(position, closest), 1.0F / separation)
        : normalized_or(multiply(relative, -1.0F), {0.0F, 1.0F, 0.0F});
    const float penetration = distance - separation;
    positions[particle] = add(position, multiply(normal, penetration));
    const float approach = fminf(dot(relative, normal), 0.0F);
    const Vec3 tangent = subtract(relative, multiply(normal, dot(relative, normal)));
    const Vec3 velocity_change = subtract(multiply(tangent, -friction),
                                           multiply(normal, approach));
    velocities[particle] = add(velocities[particle], velocity_change);
    if (!transfer_impulse) return;
    atomicAdd(contact_count, 1U);
    atomicMax(reinterpret_cast<int *>(maximum_penetration),
              __float_as_int(penetration));
    const Vec3 reaction = multiply(velocity_change, -particle_mass);
    atomic_add(node_impulses + segment, multiply(reaction, 1.0F - fraction));
    atomic_add(node_impulses + segment + 1, multiply(reaction, fraction));
}

__global__ void fluid_rope_apply(
    RopeData rope, int first, int last, const Vec3 *node_impulses,
    float maximum_acceleration, float dt, float frame_inverse_dt) {
    const unsigned node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= rope.count || rope_anchor_body(rope, node, first, last) >= 0) return;
    const float mass = rope.options.mass / static_cast<float>(rope.count);
    const Vec3 velocity_change = clamp_length(
        multiply(node_impulses[node], 1.0F / mass), maximum_acceleration * dt);
    rope.velocities[node] = clamp_length(add(rope.velocities[node], velocity_change),
                                         rope.options.maximum_speed);
    rope.fluid_contact_forces[node] = add(rope.fluid_contact_forces[node],
        multiply(velocity_change, mass * frame_inverse_dt));
}
