// SPDX-License-Identifier: MIT
// Fluid geometry queries and rigid-body contact kernels.

__device__ Vec3 fluid_closest_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) noexcept {
    const Vec3 ab = subtract(b, a), ac = subtract(c, a), ap = subtract(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) return a;
    const Vec3 bp = subtract(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F)
        return add(a, multiply(ab, d1 / (d1 - d3)));
    const Vec3 cp = subtract(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F)
        return add(a, multiply(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F)
        return add(b, multiply(subtract(c, b),
                               (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    const float denominator = va + vb + vc;
    return denominator > 1.0e-12F
        ? add(a, add(multiply(ab, vb / denominator),
                     multiply(ac, vc / denominator))) : a;
}

__device__ Vec3 fluid_closest_triangle_barycentric(
    Vec3 p, Vec3 a, Vec3 b, Vec3 c, Vec3 &weights) noexcept {
    const Vec3 ab = subtract(b, a), ac = subtract(c, a), ap = subtract(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) {
        weights = {1.0F, 0.0F, 0.0F};
        return a;
    }
    const Vec3 bp = subtract(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) {
        weights = {0.0F, 1.0F, 0.0F};
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        const float v = d1 / (d1 - d3);
        weights = {1.0F - v, v, 0.0F};
        return add(a, multiply(ab, v));
    }
    const Vec3 cp = subtract(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) {
        weights = {0.0F, 0.0F, 1.0F};
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        const float w = d2 / (d2 - d6);
        weights = {1.0F - w, 0.0F, w};
        return add(a, multiply(ac, w));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        weights = {0.0F, 1.0F - w, w};
        return add(b, multiply(subtract(c, b), w));
    }
    const float denominator = va + vb + vc;
    if (denominator <= 1.0e-12F) {
        weights = {1.0F, 0.0F, 0.0F};
        return a;
    }
    const float inverse = 1.0F / denominator;
    const float v = vb * inverse, w = vc * inverse;
    weights = {1.0F - v - w, v, w};
    return add(a, add(multiply(ab, v), multiply(ac, w)));
}

__device__ void atomic_add(Vec3 *destination, Vec3 value) noexcept {
    atomicAdd(&destination->x, value.x);
    atomicAdd(&destination->y, value.y);
    atomicAdd(&destination->z, value.z);
}

__device__ bool fluid_segment_bounds(Vec3 a, Vec3 b, const BvhNode &node,
                                     float radius) noexcept {
    float lower = 0.0F, upper = 1.0F;
    const float starts[3]{a.x, a.y, a.z};
    const float ends[3]{b.x, b.y, b.z};
    const float minima[3]{node.minimum.x, node.minimum.y, node.minimum.z};
    const float maxima[3]{node.maximum.x, node.maximum.y, node.maximum.z};
    for (int axis = 0; axis < 3; ++axis) {
        const float delta = ends[axis] - starts[axis];
        const float minimum = minima[axis] - radius;
        const float maximum = maxima[axis] + radius;
        if (fabsf(delta) < 1.0e-9F) {
            if (starts[axis] < minimum || starts[axis] > maximum) return false;
        } else {
            const float first = (minimum - starts[axis]) / delta;
            const float second = (maximum - starts[axis]) / delta;
            lower = fmaxf(lower, fminf(first, second));
            upper = fminf(upper, fmaxf(first, second));
            if (lower > upper) return false;
        }
    }
    return true;
}

__device__ __noinline__ void stamp_paint_at_contact(
    Vec3 local_particle, float particle_radius, FluidId source,
    RigidBodyId target, const TriangleMeshResource *meshes,
    const PaintFieldResource *fields, std::uint32_t field_capacity,
    const PaintRuleResource *rules, std::uint32_t rule_capacity) noexcept {
    for (std::uint32_t rule_index = 0; rule_index < rule_capacity;
         ++rule_index) {
        const PaintRuleResource rule = rules[rule_index];
        if (!rule.alive || !rule.options.enabled ||
            rule.options.source.index != source.index ||
            rule.options.source.generation != source.generation ||
            rule.options.target.index >= field_capacity) continue;
        const PaintFieldResource field = fields[rule.options.target.index];
        if (!field.alive || field.options.cloth.generation != 0U ||
            field.generation != rule.options.target.generation ||
            field.options.body.index != target.index ||
            field.options.body.generation != target.generation) continue;
        const TriangleMeshResource mesh = meshes[field.options.mesh.index];
        const float reach = particle_radius + rule.options.reach;
        const float reach_squared = reach * reach;
        float best_distance = reach_squared;
        Vec2 best_uv{};
        std::uint32_t best_side = 0U;
        std::uint32_t stack[64]{};
        int pending = mesh.bvh_node_count == 0U ? 0 : 1;
        while (pending != 0) {
            const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
            if (!fluid_segment_bounds(local_particle, local_particle,
                                      node, reach)) continue;
            if (node.triangle_count == 0U) {
                if (pending + 2 > 64) continue;
                stack[pending++] = node.right;
                stack[pending++] = node.left;
                continue;
            }
            for (std::uint32_t item = 0; item < node.triangle_count; ++item) {
                const std::uint32_t base =
                    (node.first_triangle + item) * 3U;
                const std::uint32_t ia = mesh.indices[base];
                const std::uint32_t ib = mesh.indices[base + 1U];
                const std::uint32_t ic = mesh.indices[base + 2U];
                const Vec3 a = mesh.vertices[ia], b = mesh.vertices[ib];
                const Vec3 c = mesh.vertices[ic];
                const Vec3 nearest = fluid_closest_triangle(
                    local_particle, a, b, c);
                const Vec3 delta = subtract(local_particle, nearest);
                const float distance = length_squared(delta);
                if (distance >= best_distance) continue;
                const Vec3 ab = subtract(b, a), ac = subtract(c, a);
                const Vec3 ap = subtract(nearest, a);
                const float d00 = dot(ab, ab), d01 = dot(ab, ac);
                const float d11 = dot(ac, ac), d20 = dot(ap, ab);
                const float d21 = dot(ap, ac);
                const float divisor = d00 * d11 - d01 * d01;
                if (divisor <= 1.0e-12F) continue;
                const float v = (d11*d20 - d01*d21) / divisor;
                const float w = (d00*d21 - d01*d20) / divisor;
                const float u = 1.0F - v - w;
                best_uv = {u*field.uvs[ia].x + v*field.uvs[ib].x +
                               w*field.uvs[ic].x,
                           u*field.uvs[ia].y + v*field.uvs[ib].y +
                               w*field.uvs[ic].y};
                best_side = dot(cross(ab, ac), delta) >= 0.0F ? 1U : 2U;
                best_distance = distance;
            }
        }
        if (best_side == 0U) continue;
        const int width = static_cast<int>(field.options.width);
        const int height = static_cast<int>(field.options.height);
        int x = static_cast<int>(floorf(best_uv.x * width)) % width;
        if (x < 0) x += width;
        const int y = max(0, min(height - 1,
            static_cast<int>(floorf(best_uv.y * height))));
        atomicOr(field.pixels + y * width + x, best_side);
    }
}

__global__ void fluid_static_contacts(
    Vec3 *positions, Vec3 *velocities, const Vec3 *previous, float *foam,
    const std::uint32_t *count, float radius, float spawn_clearance,
    std::uint32_t first_spawned, bool recover_spawn, Vec3 up,
    const BodyParameters *parameters, const RigidBodyState *states,
    const TriangleMeshResource *meshes, std::uint32_t body_index,
    float particle_mass, bool collect_contacts,
    FluidContactSample *samples, std::uint8_t *contact_flags,
    FluidId fluid_id, RigidBodyId body_id, bool apply_paint,
    const PaintFieldResource *paint_fields, std::uint32_t field_capacity,
    const PaintRuleResource *paint_rules, std::uint32_t rule_capacity) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    const BodyParameters body = parameters[body_index];
    const RigidBodyState state = states[body_index];
    const TriangleMeshResource mesh = meshes[body.mesh.index];
    if (particle >= *count || mesh.bvh_node_count == 0U) return;
    const Vec3 origin = inverse_rotate(state.orientation,
        subtract(previous[particle], state.position));
    Vec3 position = inverse_rotate(state.orientation,
        subtract(positions[particle], state.position));
    Vec3 velocity = inverse_rotate(state.orientation, velocities[particle]);
    const bool newly_spawned = recover_spawn && particle >= first_spawned;
    const float query_radius = newly_spawned
        ? fmaxf(radius, spawn_clearance) : radius;
    const Vec3 local_up = newly_spawned
        ? inverse_rotate(state.orientation, up) : Vec3{};
    float best_penetration = 0.0F;
    Vec3 best_normal{};
    Vec3 best_contact{};
    std::uint32_t stack[64]{};
    int pending = 1;
    while (pending != 0) {
        const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
        if (!fluid_segment_bounds(origin, position, node, query_radius)) continue;
        if (node.triangle_count == 0U) {
            if (pending + 2 > 64) continue;
            stack[pending++] = node.right;
            stack[pending++] = node.left;
            continue;
        }
        for (std::uint32_t item = 0U; item < node.triangle_count; ++item) {
            const std::uint32_t triangle =
                (node.first_triangle + item) * 3U;
            const Vec3 a = mesh.vertices[mesh.indices[triangle]];
            const Vec3 b = mesh.vertices[mesh.indices[triangle + 1U]];
            const Vec3 c = mesh.vertices[mesh.indices[triangle + 2U]];
            const Vec3 face = normalized_or(cross(subtract(b, a),
                                                  subtract(c, a)),
                                            {0.0F, 1.0F, 0.0F});
            const Vec3 closest = fluid_closest_triangle(position, a, b, c);
            const Vec3 delta = subtract(position, closest);
            const float distance = vector_length(delta);
            Vec3 normal = distance > 1.0e-6F
                ? multiply(delta, 1.0F / distance)
                : multiply(face, dot(subtract(origin, a), face) >= 0.0F
                                     ? 1.0F : -1.0F);
            float penetration = radius - distance;
            // An open triangle also catches a particle that crosses between
            // samples, even if its endpoint is already beyond the radius.
            const float before = dot(subtract(origin, a), face);
            const float after = dot(subtract(position, a), face);
            if (before * after < 0.0F) {
                const float fraction = before / (before - after);
                const Vec3 crossing = add(origin,
                    multiply(subtract(position, origin), fraction));
                if (length_squared(subtract(fluid_closest_triangle(
                        crossing, a, b, c), crossing)) < radius * radius) {
                    normal = multiply(face, before > 0.0F ? 1.0F : -1.0F);
                    penetration = fmaxf(penetration,
                        radius + fabsf(after));
                }
            }
            // A flow plane can overlap an open terrain mesh. Recover only
            // newly emitted particles close to a floor-facing triangle;
            // subsequent motion still uses swept, two-sided contacts.
            if (newly_spawned && fabsf(dot(face, local_up)) > 0.7F) {
                const Vec3 floor_normal = dot(face, local_up) > 0.0F
                    ? face : multiply(face, -1.0F);
                const float side = dot(subtract(position, a), floor_normal);
                if (side < radius && side > -spawn_clearance &&
                    distance < spawn_clearance) {
                    normal = floor_normal;
                    penetration = fmaxf(penetration, radius - side);
                }
            }
            if (penetration > best_penetration) {
                best_penetration = penetration;
                best_normal = normal;
                best_contact = closest;
            }
        }
    }
    if (best_penetration > 0.0F) {
        if (apply_paint && rule_capacity != 0U)
            stamp_paint_at_contact(position, radius, fluid_id, body_id,
                meshes, paint_fields, field_capacity, paint_rules,
                rule_capacity);
        position = add(position, multiply(best_normal, best_penetration));
        const float incoming = dot(velocity, best_normal);
        const float normal_impulse = incoming < 0.0F
            ? -incoming * (1.0F + body.restitution) * particle_mass : 0.0F;
        if (incoming < 0.0F) {
            velocity = subtract(velocity,
                multiply(best_normal, incoming * (1.0F + body.restitution)));
            const Vec3 tangent = subtract(velocity,
                multiply(best_normal, dot(velocity, best_normal)));
            const float tangent_speed = vector_length(tangent);
            if (tangent_speed > k_epsilon) {
                // Coulomb friction is limited by this contact's normal
                // impulse, as in fluid_moving_contacts. A fixed fractional
                // cut on every solver pass overdamps water at rest on walls.
                const float friction_speed = fminf(tangent_speed,
                    body.friction * normal_impulse / particle_mass);
                velocity = subtract(velocity, multiply(tangent,
                    friction_speed / tangent_speed));
            }
            foam[particle] = fmaxf(foam[particle],
                                  fminf(1.0F, -incoming * 0.35F));
        }
        positions[particle] = add(state.position,
                                  rotate(state.orientation, position));
        velocities[particle] = rotate(state.orientation, velocity);
        if (collect_contacts &&
            (contact_flags[particle] == 0U ||
             (contact_flags[particle] == 1U &&
              normal_impulse > samples[particle].normal_impulse))) {
            samples[particle] = {
                transform_point(state, best_contact),
                rotate(state.orientation, best_normal),
                normal_impulse, body_index};
            contact_flags[particle] = 1U;
        }
    }
}

__global__ void fluid_body_bounds_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous,
    const RigidBodyState *current, const TriangleMeshResource *meshes,
    std::uint32_t count, WorldAabb *bounds) {
    const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= count || parameters[body].motion == MotionType::static_body)
        return;
    const TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
    Vec3 minimum{}, maximum{};
    transformed_motion_bounds(mesh.minimum, mesh.maximum,
        bounds_transform(previous[body]), bounds_transform(current[body]),
        true, parameters[body].collision_margin, minimum, maximum);
    // Endpoint AABBs alone can miss the middle of a fast rotation.
    const float rotation_reach = rotational_motion_bound(
        previous[body], current[body], mesh);
    const Vec3 expansion{rotation_reach, rotation_reach, rotation_reach};
    minimum = subtract(minimum, expansion);
    maximum = add(maximum, expansion);
    bounds[body] = {minimum, maximum};
}

__global__ void fluid_index_body_cells(
    const BodyParameters *parameters, const WorldAabb *bounds,
    std::uint32_t body_count, std::uint32_t words, float padding,
    unsigned long long *masks, unsigned long long *global_masks) {
    const std::uint32_t body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= body_count || parameters[body].motion == MotionType::static_body)
        return;
    const WorldAabb box = bounds[body];
    const int x0 = __float2int_rd(box.minimum.x - padding);
    const int y0 = __float2int_rd(box.minimum.y - padding);
    const int z0 = __float2int_rd(box.minimum.z - padding);
    const int x1 = __float2int_rd(box.maximum.x + padding);
    const int y1 = __float2int_rd(box.maximum.y + padding);
    const int z1 = __float2int_rd(box.maximum.z + padding);
    const unsigned long long bit = 1ULL << (body & 63U);
    const std::uint32_t word = body / 64U;
    if (static_cast<std::int64_t>(x1) - x0 > 3 ||
        static_cast<std::int64_t>(y1) - y0 > 3 ||
        static_cast<std::int64_t>(z1) - z0 > 3) {
        atomicOr(global_masks + word, bit);
        return;
    }
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                atomicOr(masks + fluid_body_bucket(x, y, z) * words + word,
                         bit);
}

__global__ void fluid_detect_moving_contacts(
    const Vec3 *positions, const Vec3 *previous,
    const std::uint32_t *count, float radius,
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states, const TriangleMeshResource *meshes,
    const WorldAabb *bounds, std::uint32_t body_count,
    const unsigned long long *masks, const unsigned long long *global_masks,
    std::uint32_t words, bool first_iteration,
    FluidMovingContact *contacts, std::uint32_t *contact_counts) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    contacts[particle] = {};
    const Vec3 start = previous[particle];
    const Vec3 end = positions[particle];
    float best_penetration = 0.0F;
    Vec3 best_normal{}, best_contact{};
    std::uint32_t best_body = k_invalid_dense;
    const std::uint32_t bucket = fluid_body_bucket(
        __float2int_rd(end.x), __float2int_rd(end.y),
        __float2int_rd(end.z));
    for (std::uint32_t word = 0U; word < words; ++word) {
        unsigned long long candidates =
            masks[bucket * words + word] | global_masks[word];
        while (candidates != 0ULL) {
            const std::uint32_t bit =
                static_cast<std::uint32_t>(__ffsll(candidates) - 1);
            candidates &= candidates - 1ULL;
            const std::uint32_t body_index = word * 64U + bit;
            if (body_index >= body_count) continue;
            const BodyParameters body = parameters[body_index];
            if (body.motion == MotionType::static_body) continue;
            const WorldAabb box = bounds[body_index];
            if (fmaxf(start.x, end.x) + radius < box.minimum.x ||
                fminf(start.x, end.x) - radius > box.maximum.x ||
                fmaxf(start.y, end.y) + radius < box.minimum.y ||
                fminf(start.y, end.y) - radius > box.maximum.y ||
                fmaxf(start.z, end.z) + radius < box.minimum.z ||
                fminf(start.z, end.z) - radius > box.maximum.z) continue;
            const RigidBodyState state = states[body_index];
            const RigidBodyState old = first_iteration
                ? previous_states[body_index] : state;
            const Vec3 origin = inverse_rotate(old.orientation,
                subtract(start, old.position));
            const Vec3 position = inverse_rotate(state.orientation,
                subtract(end, state.position));
            const TriangleMeshResource mesh = meshes[body.mesh.index];
            if (mesh.bvh_node_count == 0U) continue;
            std::uint32_t stack[64]{};
            int pending = 1;
            while (pending != 0) {
                const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
                if (!fluid_segment_bounds(origin, position, node, radius))
                    continue;
                if (node.triangle_count == 0U) {
                    if (pending + 2 > 64) continue;
                    stack[pending++] = node.right;
                    stack[pending++] = node.left;
                    continue;
                }
                for (std::uint32_t item = 0U; item < node.triangle_count;
                     ++item) {
                    const std::uint32_t triangle =
                        (node.first_triangle + item) * 3U;
                    const Vec3 a = mesh.vertices[mesh.indices[triangle]];
                    const Vec3 b = mesh.vertices[mesh.indices[triangle + 1U]];
                    const Vec3 c = mesh.vertices[mesh.indices[triangle + 2U]];
                    const Vec3 face = normalized_or(cross(subtract(b, a),
                        subtract(c, a)), {0.0F, 1.0F, 0.0F});
                    const Vec3 closest = fluid_closest_triangle(position, a, b, c);
                    const Vec3 delta = subtract(position, closest);
                    const float distance = vector_length(delta);
                    Vec3 normal = distance > 1.0e-6F
                        ? multiply(delta, 1.0F / distance)
                        : multiply(face, dot(subtract(origin, a), face) >= 0.0F
                                             ? 1.0F : -1.0F);
                    float penetration = radius - distance;
                    const float before = dot(subtract(origin, a), face);
                    const float after = dot(subtract(position, a), face);
                    if (before * after < 0.0F) {
                        const float fraction = before / (before - after);
                        const Vec3 crossing = add(origin,
                            multiply(subtract(position, origin), fraction));
                        if (length_squared(subtract(fluid_closest_triangle(
                                crossing, a, b, c), crossing)) < radius * radius) {
                            normal = multiply(face, before > 0.0F ? 1.0F : -1.0F);
                            penetration = fmaxf(penetration,
                                                radius + fabsf(after));
                        }
                    }
                    if (penetration > best_penetration) {
                        best_penetration = penetration;
                        best_normal = rotate(state.orientation, normal);
                        best_contact = transform_point(state, closest);
                        best_body = body_index;
                    }
                }
            }
        }
    }
    if (best_body == k_invalid_dense) return;
    contacts[particle] = {best_normal, best_contact,
                          best_penetration, best_body};
    atomicAdd(contact_counts + best_body, 1U);
}

__global__ void fluid_resolve_moving_contacts(
    Vec3 *positions, Vec3 *velocities, float *foam,
    const std::uint32_t *count, float radius, float particle_mass,
    float timestep, float maximum_speed,
    const BodyParameters *parameters, const RigidBodyState *states,
    const TriangleMeshResource *meshes,
    const FluidMovingContact *contacts, const std::uint32_t *contact_counts,
    FluidBodyImpulse *impulses, bool collect_contacts,
    FluidContactSample *samples, std::uint8_t *particle_contact_flags,
    FluidId fluid_id, const RigidBodyId *body_ids, bool apply_paint,
    const PaintFieldResource *paint_fields, std::uint32_t field_capacity,
    const PaintRuleResource *paint_rules, std::uint32_t rule_capacity) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    impulses[particle] = {};
    const Vec3 end = positions[particle];
    const FluidMovingContact contact = contacts[particle];
    const Vec3 best_normal = contact.normal;
    const Vec3 best_contact = contact.point;
    const float best_penetration = contact.penetration;
    const std::uint32_t best_body = contact.body;
    if (best_body == k_invalid_dense) return;
    const BodyParameters body = parameters[best_body];
    const RigidBodyState state = states[best_body];
    if (apply_paint && rule_capacity != 0U) {
        const Vec3 local_particle = inverse_rotate(state.orientation,
            subtract(end, state.position));
        stamp_paint_at_contact(local_particle, radius, fluid_id,
            body_ids[best_body], meshes, paint_fields, field_capacity,
            paint_rules, rule_capacity);
    }
    positions[particle] = add(end, multiply(best_normal, best_penetration));
    if (collect_contacts && particle_contact_flags[particle] != 2U) {
        samples[particle] = {best_contact, best_normal, 0.0F, best_body};
        particle_contact_flags[particle] = 2U;
    }
    Vec3 velocity = velocities[particle];
    const Vec3 arm = subtract(best_contact, state.position);
    const Vec3 body_velocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    const Vec3 relative = subtract(velocity, body_velocity);
    const float incoming = dot(relative, best_normal);
    // A resting particle can be pushed into a moving surface without having
    // negative normal velocity. Share a bounded overlap-recovery impulse with
    // the body instead of silently moving only the particle.
    const float recovery_speed = fminf(maximum_speed,
        fminf(0.5F * radius, 0.2F * best_penetration) / timestep);
    if (incoming >= recovery_speed) return;
    const Vec3 normal_cross = cross(arm, best_normal);
    // Jacobi contacts all saw the same pre-solve body velocity. A light body
    // must share its effective mass across simultaneous impacts, otherwise it
    // receives N full reactions. For a heavy body, isolated pair contacts are
    // already well conditioned; blend by the particle/body mass ratio to
    // avoid unnecessarily weakening its established contact response.
    const float batch_size = 1.0F +
        float(contact_counts[best_body] - 1U) *
        fminf(1.0F, 4.0F * particle_mass * body.inverse_mass);
    const float normal_denominator = 1.0F / particle_mass +
        batch_size * (body.inverse_mass +
            dot(cross(inverse_inertia_world(body, state,
                normal_cross), arm), best_normal));
    if (normal_denominator <= k_epsilon) return;
    const float normal_impulse =
        (recovery_speed - incoming) / normal_denominator;
    if (collect_contacts &&
        (particle_contact_flags[particle] != 2U ||
         normal_impulse > samples[particle].normal_impulse)) {
        samples[particle] = {best_contact, best_normal, normal_impulse,
                             best_body};
        particle_contact_flags[particle] = 2U;
    }
    Vec3 impulse = multiply(best_normal, normal_impulse);
    velocity = add(velocity, multiply(impulse, 1.0F / particle_mass));
    const Vec3 tangent_velocity = subtract(relative,
        multiply(best_normal, incoming));
    const float tangent_speed = vector_length(tangent_velocity);
    if (tangent_speed > k_epsilon) {
        const Vec3 tangent = multiply(tangent_velocity, 1.0F / tangent_speed);
        const float tangent_denominator = 1.0F / particle_mass +
            batch_size * (body.inverse_mass +
                dot(cross(inverse_inertia_world(body, state,
                    cross(arm, tangent)), arm), tangent));
        if (tangent_denominator > k_epsilon) {
            const float tangent_impulse = fminf(
                tangent_speed / tangent_denominator,
                body.friction * normal_impulse);
            const Vec3 friction = multiply(tangent, -tangent_impulse);
            impulse = add(impulse, friction);
            velocity = add(velocity, multiply(friction,
                                              1.0F / particle_mass));
        }
    }
    velocities[particle] = velocity;
    foam[particle] = fmaxf(foam[particle],
                          fminf(1.0F, -incoming * 0.35F));
    impulses[particle] = {multiply(impulse, -1.0F),
                          multiply(cross(arm, impulse), -1.0F), best_body};
}

__global__ void reduce_point_body_impulses(
    const FluidBodyImpulse *impulses, const std::uint32_t *particle_count,
    const BodyParameters *parameters, RigidBodyState *states,
    const std::uint32_t *contact_flags, std::uint32_t body_count,
    const Vec3 *position_corrections = nullptr) {
    const std::uint32_t body = blockIdx.x;
    if (body >= body_count || contact_flags[body] == 0U ||
        parameters[body].motion != MotionType::dynamic)
        return;
    __shared__ Vec3 linear[128];
    __shared__ Vec3 angular[128];
    __shared__ Vec3 position[128];
    Vec3 local_linear{}, local_angular{}, local_position{};
    for (std::uint32_t particle = threadIdx.x; particle < *particle_count;
         particle += blockDim.x) {
        const FluidBodyImpulse impulse = impulses[particle];
        if (impulse.body != body) continue;
        local_linear = add(local_linear, impulse.linear);
        local_angular = add(local_angular, impulse.angular);
        if (position_corrections != nullptr)
            local_position = add(local_position, position_corrections[particle]);
    }
    linear[threadIdx.x] = local_linear;
    angular[threadIdx.x] = local_angular;
    position[threadIdx.x] = local_position;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
        if (threadIdx.x < stride) {
            linear[threadIdx.x] = add(linear[threadIdx.x],
                                     linear[threadIdx.x + stride]);
            angular[threadIdx.x] = add(angular[threadIdx.x],
                                       angular[threadIdx.x + stride]);
            position[threadIdx.x] = add(position[threadIdx.x],
                                        position[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        RigidBodyState state = states[body];
        const BodyParameters options = parameters[body];
        state.position = add(state.position, position[0]);
        state.linear_velocity = clamp_length(add(state.linear_velocity,
            multiply(linear[0], options.inverse_mass)),
            options.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(state.angular_velocity,
            inverse_inertia_world(options, state, angular[0])),
            options.maximum_angular_speed);
        states[body] = state;
    }
}
