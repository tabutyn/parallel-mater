// SPDX-License-Identifier: MIT
// Cloth and smoke coupling kernels.

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
