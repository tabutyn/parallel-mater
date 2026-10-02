// SPDX-License-Identifier: MIT
// Rope and smoke coupling kernels.

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
