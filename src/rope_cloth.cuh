// SPDX-License-Identifier: MIT
// Shared rope endpoint/cloth vertex joint. World owns lifetime and ordering.
struct RopeClothCouplingStorage {
    RopeClothCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
};

__global__ void rope_cloth_sample_anchor(RopeData rope, unsigned end,
    const Vec3 *positions, const Vec3 *velocities,
    const float *inverse_masses, unsigned vertex, float effective_mass) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    rope.soft_anchor_positions[end] = positions[vertex];
    rope.soft_anchor_velocities[end] = velocities[vertex];
    rope.soft_anchor_impulses[end] = {};
    rope.soft_anchor_inverse_masses[end] =
        fminf(inverse_masses[vertex], 1.0F / effective_mass);
}

__global__ void rope_cloth_apply_anchor(RopeData rope, unsigned end,
    Vec3 *positions, Vec3 *velocities, Vec3 *forces,
    const float *inverse_masses, unsigned vertex,
    float maximum_acceleration, float dt, float frame_inverse_dt) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    const float inverse_mass = inverse_masses[vertex];
    if (inverse_mass <= 0.0F) return;
    const Vec3 correction = clamp_length(
        subtract(rope.soft_anchor_positions[end], positions[vertex]),
        maximum_acceleration * dt * dt);
    const Vec3 delta_velocity = multiply(correction, 1.0F / dt);
    velocities[vertex] = add(velocities[vertex], delta_velocity);
    positions[vertex] = add(positions[vertex], correction);
    forces[vertex] = add(forces[vertex],
        multiply(delta_velocity, frame_inverse_dt / inverse_mass));
    const unsigned node = end ? rope.count - 1U : 0U;
    rope.positions[node] = rope.soft_anchor_positions[end] = positions[vertex];
    rope.velocities[node] = rope.soft_anchor_velocities[end] = velocities[vertex];
}
