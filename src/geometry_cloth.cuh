// SPDX-License-Identifier: MIT
// Cloth geometry and rigid-body contact kernels.

__device__ __noinline__ void stamp_rigid_cloth_paint(
    RigidBodyId rigid_id, ClothId cloth_id, Vec3 center, Vec3 contact,
    const Vec3 *positions, const std::uint32_t *indices,
    const std::uint32_t *vertex_sources,
    std::uint32_t triangle_count, const PaintFieldResource *fields,
    std::uint32_t field_capacity, const PaintRuleResource *rules,
    std::uint32_t rule_capacity) {
    for (std::uint32_t index = 0U; index < rule_capacity; ++index) {
        const PaintRuleResource rule = rules[index];
        if (!rule.alive || !rule.options.enabled ||
            rule.options.rigid_source.index != rigid_id.index ||
            rule.options.rigid_source.generation != rigid_id.generation ||
            rule.options.target.index >= field_capacity) continue;
        const PaintFieldResource field = fields[rule.options.target.index];
        if (!field.alive || field.generation != rule.options.target.generation ||
            field.options.cloth.index != cloth_id.index ||
            field.options.cloth.generation != cloth_id.generation) continue;
        const int width = static_cast<int>(field.options.width);
        const int height = static_cast<int>(field.options.height);
        const float radius_squared = rule.options.brush_radius *
                                     rule.options.brush_radius;
        for (std::uint32_t triangle = 0U; triangle < triangle_count;
             ++triangle) {
            const std::uint32_t base = triangle * 3U;
            const std::uint32_t ia = indices[base];
            const std::uint32_t ib = indices[base + 1U];
            const std::uint32_t ic = indices[base + 2U];
            if (ia == ib) continue;
            const Vec3 a = positions[ia], b = positions[ib], c = positions[ic];
            if (length_squared(subtract(contact,
                fluid_closest_triangle(contact, a, b, c))) > radius_squared)
                continue;
            const Vec2 ua = field.uvs[vertex_sources ? vertex_sources[ia] : ia],
                       ub = field.uvs[vertex_sources ? vertex_sources[ib] : ib],
                       uc = field.uvs[vertex_sources ? vertex_sources[ic] : ic];
            const float e0x = ub.x - ua.x, e0y = ub.y - ua.y;
            const float e1x = uc.x - ua.x, e1y = uc.y - ua.y;
            const float determinant = e0x * e1y - e0y * e1x;
            if (fabsf(determinant) < 1.0e-10F) continue;
            const float min_u = fminf(ua.x, fminf(ub.x, uc.x));
            const float max_u = fmaxf(ua.x, fmaxf(ub.x, uc.x));
            const float min_v = fminf(ua.y, fminf(ub.y, uc.y));
            const float max_v = fmaxf(ua.y, fmaxf(ub.y, uc.y));
            const int x0 = max(0, static_cast<int>(floorf(min_u * width)));
            const int x1 = min(width - 1,
                static_cast<int>(floorf(max_u * width)));
            const int y0 = max(0, static_cast<int>(floorf(min_v * height)));
            const int y1 = min(height - 1,
                static_cast<int>(floorf(max_v * height)));
            if (x0 > x1 || y0 > y1) continue;
            const std::uint32_t side = dot(cross(subtract(b, a),
                subtract(c, a)), subtract(center, contact)) >= 0.0F
                ? 1U : 2U;
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const float qx = (static_cast<float>(x) + 0.5F) /
                                     width - ua.x;
                    const float qy = (static_cast<float>(y) + 0.5F) /
                                     height - ua.y;
                    const float v = (qx * e1y - qy * e1x) / determinant;
                    const float w = (e0x * qy - e0y * qx) / determinant;
                    const float u = 1.0F - v - w;
                    if (u < -1.0e-4F || v < -1.0e-4F || w < -1.0e-4F)
                        continue;
                    const Vec3 point = add(multiply(a, u),
                        add(multiply(b, v), multiply(c, w)));
                    if (length_squared(subtract(point, contact)) <=
                        radius_squared)
                        atomicOr(field.pixels + y * width + x, side);
                }
            }
        }
    }
}

__global__ void cloth_constrain_bodies(
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *cloth_indices, const std::uint32_t *vertex_sources,
    const Vec3 *surface_positions,
    const float *cloth_inverse_masses, std::uint32_t vertex_count,
    std::uint32_t triangle_count, float thickness, float dt,
    float contact_friction,
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    RigidBodyState *states, const TriangleMeshResource *meshes,
    const RigidBodyId *body_ids, std::uint32_t body_count,
    ClothId cloth_id, const PaintFieldResource *paint_fields,
    std::uint32_t field_capacity, const PaintRuleResource *paint_rules,
    std::uint32_t rule_capacity, ClothBodyCorrection *corrections) {
    const std::uint32_t body_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (body_index >= body_count) return;
    corrections[body_index] = {};
    if (parameters[body_index].motion != MotionType::dynamic) return;
    const BodyParameters body = parameters[body_index];
    RigidBodyState state = states[body_index];
    const TriangleMeshResource mesh = meshes[body.mesh.index];
    const Vec3 center = transform_point(state, mesh.bounding_center);
    const Vec3 previous_center = transform_point(previous_states[body_index],
                                                  mesh.bounding_center);
    const float radius = mesh.bounding_radius + body.collision_margin + thickness;
    float best_penetration = 0.0F;
    Vec3 best_normal{}, best_contact{};
    std::uint32_t best_triangle = 0U;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t base = triangle * 3U;
        const Vec3 a = surface_positions != nullptr
            ? surface_positions[base] : cloth_positions[cloth_indices[base]];
        const Vec3 b = surface_positions != nullptr
            ? surface_positions[base + 1U]
            : cloth_positions[cloth_indices[base + 1U]];
        const Vec3 c = surface_positions != nullptr
            ? surface_positions[base + 2U]
            : cloth_positions[cloth_indices[base + 2U]];
        const Vec3 ab = subtract(b, a), ac = subtract(c, a);
        const Vec3 face = cross(ab, ac);
        if (length_squared(face) < 1.0e-12F) continue;
        const Vec3 normal = normalized_or(face, {0.0F, 0.0F, 1.0F});
        const Vec3 nearest = fluid_closest_triangle(center, a, b, c);
        const Vec3 delta = subtract(center, nearest);
        const float distance = vector_length(delta);
        Vec3 contact_normal = distance > 1.0e-6F
            ? multiply(delta, 1.0F / distance)
            : multiply(normal, dot(subtract(previous_center, a), normal) >= 0.0F
                                   ? 1.0F : -1.0F);
        float penetration = radius - distance;
        const float before = dot(subtract(previous_center, a), normal);
        const float after = dot(subtract(center, a), normal);
        if (before * after < 0.0F) {
            const float fraction = before / (before - after);
            const Vec3 crossing = add(previous_center,
                multiply(subtract(center, previous_center), fraction));
            if (length_squared(subtract(fluid_closest_triangle(
                    crossing, a, b, c), crossing)) < radius * radius) {
                contact_normal = multiply(normal, before > 0.0F ? 1.0F : -1.0F);
                penetration = fmaxf(penetration, radius + fabsf(after));
            }
        } else if (penetration > 0.0F && before * after > 0.0F) {
            contact_normal = multiply(normal, before > 0.0F ? 1.0F : -1.0F);
        }
        if (penetration > best_penetration) {
            best_penetration = penetration;
            best_normal = contact_normal;
            best_contact = nearest;
            best_triangle = triangle;
        }
    }
    if (best_penetration <= 0.0F) return;
    const std::uint32_t first = best_triangle * 3U;
    const std::uint32_t a = cloth_indices[first];
    const std::uint32_t b = cloth_indices[first + 1U];
    const std::uint32_t c = cloth_indices[first + 2U];
    if (rule_capacity != 0U)
        stamp_rigid_cloth_paint(body_ids[body_index], cloth_id, center,
            best_contact, cloth_positions, cloth_indices, vertex_sources, triangle_count,
            paint_fields, field_capacity, paint_rules, rule_capacity);
    // Fracturing cloth follows the node-contact response: the conservative
    // triangle-radius constraint otherwise cancels the body's incoming speed
    // on every substep while bonds fail. Keep the triangle query for paint.
    if (surface_positions != nullptr) return;
    const float free_fraction =
        ((cloth_inverse_masses[a] > 0.0F ? 1.0F : 0.0F) +
         (cloth_inverse_masses[b] > 0.0F ? 1.0F : 0.0F) +
         (cloth_inverse_masses[c] > 0.0F ? 1.0F : 0.0F)) / 3.0F;
    constexpr float cloth_share = 0.5F;
    const float cloth_shift = cloth_share * best_penetration;
    const Vec3 arm = subtract(best_contact, state.position);
    const float incoming = dot(state.linear_velocity, best_normal);
    const Vec3 angular_axis = cross(arm, best_normal);
    // The conservative body-radius constraint must remove inward center
    // velocity even if spin makes the contact-point velocity nearly zero.
    const float normal_impulse = incoming < 0.0F && body.inverse_mass > 0.0F
        ? -incoming / body.inverse_mass : 0.0F;
    const float support_radius = fmaxf(0.15F, mesh.bounding_radius * 0.8F);
    const float inverse_support_squared = 1.0F /
        (support_radius * support_radius);
    float weight_sum = 0.0F;
    float maximum_weighted_inverse_mass = 0.0F;
    float weighted_inverse_mass_squared = 0.0F;
    Vec3 weighted_cloth_velocity{};
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
        const float inverse_mass = cloth_inverse_masses[vertex];
        if (inverse_mass == 0.0F) continue;
        const float squared = length_squared(subtract(
            cloth_positions[vertex], best_contact));
        const float weight = fmaxf(0.0F,
            1.0F - squared * inverse_support_squared);
        const float weighted = weight * weight;
        weight_sum += weighted;
        weighted_inverse_mass_squared += inverse_mass * weighted * weighted;
        weighted_cloth_velocity = add(weighted_cloth_velocity,
                                      multiply(cloth_velocities[vertex], weighted));
        maximum_weighted_inverse_mass = fmaxf(
            maximum_weighted_inverse_mass, inverse_mass * weighted);
    }
    // Bound the velocity kick of every contacted cloth vertex to one tenth
    // of its collision thickness per substep. Excess impact is dissipated.
    const float maximum_cloth_impulse = maximum_weighted_inverse_mass > 0.0F
        ? 0.1F * thickness * weight_sum /
              (dt * maximum_weighted_inverse_mass)
        : normal_impulse;
    const float cloth_impulse = fminf(normal_impulse,
                                     maximum_cloth_impulse);
    Vec3 tangent_impulse{};
    if (weight_sum > 1.0e-8F && contact_friction > 0.0F) {
        const Vec3 cloth_velocity = multiply(weighted_cloth_velocity,
                                             1.0F / weight_sum);
        const Vec3 body_velocity = add(state.linear_velocity,
                                      cross(state.angular_velocity, arm));
        const Vec3 relative = subtract(body_velocity, cloth_velocity);
        const float separating_speed = dot(relative, best_normal);
        const Vec3 tangent = subtract(relative,
            multiply(best_normal, separating_speed));
        const float tangent_speed = vector_length(tangent);
        // Friction must not turn an outward-moving body into a cloth tether.
        const float release_weight = fmaxf(0.0F,
            1.0F - fmaxf(incoming, 0.0F) / 0.5F);
        if (release_weight > 0.0F && separating_speed <= 0.0F &&
            tangent_speed > 1.0e-6F) {
            const Vec3 direction = multiply(tangent, 1.0F / tangent_speed);
            const Vec3 angular_axis = cross(arm, direction);
            const float cloth_inverse_mass =
                weighted_inverse_mass_squared / (weight_sum * weight_sum);
            const float effective_inverse_mass = body.inverse_mass +
                dot(cross(inverse_inertia_world(body, state, angular_axis), arm),
                    direction) + cloth_inverse_mass;
            // Resting contact can have little incoming speed despite a finite
            // positional correction; use that correction as normal load.
            const float correction_impulse = best_penetration > 0.0F &&
                    body.inverse_mass > 0.0F
                ? 0.2F * best_penetration / (dt * body.inverse_mass) : 0.0F;
            const float friction_limit = contact_friction * release_weight *
                fmaxf(normal_impulse, correction_impulse);
            const float magnitude = effective_inverse_mass > k_epsilon
                ? fminf(tangent_speed / effective_inverse_mass, friction_limit)
                : 0.0F;
            tangent_impulse = multiply(direction, -magnitude);
        }
    }
    // Limit the cloth-side kick independently of the rigid-body friction.
    const Vec3 cloth_tangent_impulse = clamp_length(tangent_impulse,
                                                    maximum_cloth_impulse);
    corrections[body_index] = {
        multiply(best_normal, -cloth_shift),
        subtract(multiply(best_normal, -cloth_impulse),
                 cloth_tangent_impulse), best_contact,
        support_radius, weight_sum, {a, b, c}, true};
    state.position = add(state.position,
        multiply(best_normal,
                 best_penetration - cloth_shift * free_fraction));
    if (normal_impulse > 0.0F) {
        state.linear_velocity = clamp_length(add(state.linear_velocity,
            multiply(best_normal, normal_impulse * body.inverse_mass)),
            body.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(state.angular_velocity,
            inverse_inertia_world(body, state,
                multiply(angular_axis, normal_impulse))),
            body.maximum_angular_speed);
    }
    state.linear_velocity = clamp_length(add(state.linear_velocity,
        multiply(tangent_impulse, body.inverse_mass)),
        body.maximum_linear_speed);
    state.angular_velocity = clamp_length(add(state.angular_velocity,
        inverse_inertia_world(body, state, cross(arm, tangent_impulse))),
        body.maximum_angular_speed);
    states[body_index] = state;
}

__global__ void cloth_apply_body_corrections(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    std::uint32_t count,
    const ClothBodyCorrection *corrections, std::uint32_t body_count) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count || inverse_masses[vertex] == 0.0F) return;
    Vec3 offset{};
    Vec3 velocity = velocities[vertex];
    for (std::uint32_t body = 0U; body < body_count; ++body) {
        const ClothBodyCorrection correction = corrections[body];
        if (!correction.active) continue;
        if (correction.vertices[0] == vertex ||
            correction.vertices[1] == vertex ||
            correction.vertices[2] == vertex)
            offset = add(offset, correction.offset);
        if (correction.weight_sum > 1.0e-8F) {
            const float squared = length_squared(subtract(
                positions[vertex], correction.contact));
            const float support_squared = correction.support_radius *
                correction.support_radius;
            const float weight = fmaxf(0.0F, 1.0F - squared / support_squared);
            velocity = add(velocity, multiply(correction.impulse,
                inverse_masses[vertex] * weight * weight /
                    correction.weight_sum));
        }
    }
    positions[vertex] = add(positions[vertex], offset);
    velocities[vertex] = clamp_length(velocity, 20.0F);
}
