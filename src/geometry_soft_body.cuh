// SPDX-License-Identifier: MIT
// Soft-body geometry and rigid-body contact kernels.

template <bool couple_dynamic, bool position_only = false>
__global__ void deformable_collide(
    Vec3 *positions, Vec3 *velocities, const Vec3 *previous,
    const float *inverse_masses, std::uint32_t count, float thickness,
    float dt, const BodyParameters *parameters,
    const RigidBodyState *previous_states, const RigidBodyState *states,
    const TriangleMeshResource *meshes,
    std::uint32_t body_count, FluidBodyImpulse *impulses,
    Vec3 *contact_forces, Vec3 *contact_normals, Vec3 *contact_arms,
    Vec3 *contact_momentum_delta,
    float *accumulated_normal_delta,
    std::uint32_t *deformable_contact_count,
    std::uint32_t *dynamic_contact_flag,
    std::uint32_t *contact_flags, bool static_only,
    bool reconstruct_free_velocities, float maximum_speed,
    Vec3 *body_position_corrections = nullptr) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    if constexpr (!position_only) {
        if (body_position_corrections != nullptr)
            body_position_corrections[vertex] = {};
        impulses[vertex] = {};
        contact_forces[vertex] = {};
        if (contact_normals != nullptr) contact_normals[vertex] = {};
        if constexpr (couple_dynamic) contact_arms[vertex] = {};
    }
    if (inverse_masses[vertex] == 0.0F) return;
    const Vec3 start = previous[vertex];
    Vec3 end = positions[vertex];
    float best_penetration = 0.0F;
    Vec3 best_normal{}, best_contact{};
    std::uint32_t best_body = k_invalid_dense;
    for (std::uint32_t body_index = 0U; body_index < body_count; ++body_index) {
        const BodyParameters body = parameters[body_index];
        if (static_only && body.motion != MotionType::static_body) continue;
        const RigidBodyState state = states[body_index];
        const RigidBodyState previous_state = previous_states != nullptr
            ? previous_states[body_index] : state;
        const TriangleMeshResource mesh = meshes[body.mesh.index];
        if (mesh.bvh_node_count == 0U) continue;
        // Sweep in collider-local space at both ends. Reusing the current body
        // transform for the start sample lets a moving closed body engulf a
        // node without the relative segment ever crossing its surface.
        const Vec3 local_start = inverse_rotate(previous_state.orientation,
            subtract(start, previous_state.position));
        const Vec3 local_end = inverse_rotate(state.orientation,
            subtract(end, state.position));
        if (!fluid_segment_bounds(local_start, local_end, mesh.bvh_nodes[0],
                                  thickness + body.collision_margin)) continue;
        if constexpr (couple_dynamic) {
            if (mesh.solid_planes != nullptr) {
                float nearest_side = -FLT_MAX;
                CollisionPlane nearest_plane{};
                for (std::uint32_t triangle = 0U;
                     triangle < mesh.index_count / 3U; ++triangle) {
                    const CollisionPlane plane = mesh.solid_planes[triangle];
                    const float side = dot(plane.normal, local_end) - plane.offset;
                    if (side > nearest_side) {
                        nearest_side = side;
                        nearest_plane = plane;
                    }
                    if (side > 0.0F) break;
                }
                if (nearest_side <= 0.0F) {
                    const float penetration = thickness +
                        body.collision_margin - nearest_side;
                    if (penetration > best_penetration) {
                        best_penetration = penetration;
                        best_normal = rotate(state.orientation,
                                              nearest_plane.normal);
                        best_contact = transform_point(state, subtract(local_end,
                            multiply(nearest_plane.normal, nearest_side)));
                        best_body = body_index;
                    }
                    // A solid's interior cannot choose the inward normal of
                    // its closest triangle, even after an earlier contact or
                    // spring projection left the node behind that triangle.
                    continue;
                }
            }
        }
        std::uint32_t stack[64]{};
        int pending = 1;
        while (pending != 0) {
            const BvhNode &node = mesh.bvh_nodes[stack[--pending]];
            if (!fluid_segment_bounds(local_start, local_end, node,
                                      thickness + body.collision_margin)) continue;
            if (node.triangle_count == 0U) {
                if (pending + 2 > 64) continue;
                stack[pending++] = node.right;
                stack[pending++] = node.left;
                continue;
            }
            for (std::uint32_t item = 0U; item < node.triangle_count; ++item) {
                const std::uint32_t base = (node.first_triangle + item) * 3U;
                const Vec3 a = mesh.vertices[mesh.indices[base]];
                const Vec3 b = mesh.vertices[mesh.indices[base + 1U]];
                const Vec3 c = mesh.vertices[mesh.indices[base + 2U]];
                const Vec3 nearest = fluid_closest_triangle(local_end, a, b, c);
                const Vec3 delta = subtract(local_end, nearest);
                const float distance = vector_length(delta);
                const bool solid = couple_dynamic && mesh.solid_planes != nullptr;
                const Vec3 face = solid ? mesh.solid_planes[base / 3U].normal
                    : normalized_or(cross(subtract(b, a),
                        subtract(c, a)), {0.0F, 1.0F, 0.0F});
                Vec3 normal = distance > 1.0e-6F
                    ? multiply(delta, 1.0F / distance)
                    : multiply(face, dot(subtract(local_start, a), face) >= 0.0F
                                         ? 1.0F : -1.0F);
                float penetration = thickness + body.collision_margin - distance;
                const float before = dot(subtract(local_start, a), face);
                const float after = dot(subtract(local_end, a), face);
                if (before * after < 0.0F && (!solid || before > 0.0F)) {
                    const float fraction = before / (before - after);
                    const Vec3 crossing = add(local_start,
                        multiply(subtract(local_end, local_start), fraction));
                    if (length_squared(subtract(fluid_closest_triangle(
                            crossing, a, b, c), crossing)) <
                            thickness * thickness) {
                        normal = multiply(face, before > 0.0F ? 1.0F : -1.0F);
                        penetration = fmaxf(penetration,
                            thickness + body.collision_margin + fabsf(after));
                    }
                }
                if (penetration > best_penetration) {
                    best_penetration = penetration;
                    best_normal = rotate(state.orientation, normal);
                    best_contact = transform_point(state, nearest);
                    best_body = body_index;
                }
            }
        }
    }
    if (best_body != k_invalid_dense) {
        if constexpr (position_only) {
            positions[vertex] = add(end, multiply(best_normal, best_penetration));
            return;
        }
        const BodyParameters body = parameters[best_body];
        const RigidBodyState state = states[best_body];
        if constexpr (couple_dynamic) {
            if (body.motion == MotionType::dynamic)
                atomicExch(dynamic_contact_flag, 1U);
        }
        if (contact_normals != nullptr) contact_normals[vertex] = best_normal;
        if (accumulated_normal_delta != nullptr)
            accumulated_normal_delta[vertex] += best_penetration;
        if (deformable_contact_count != nullptr)
            atomicAdd(deformable_contact_count, 1U);
        Vec3 velocity = multiply(subtract(end, start), 1.0F / dt);
        float node_share = 1.0F;
        if constexpr (couple_dynamic)
            node_share = inverse_masses[vertex] /
                         (inverse_masses[vertex] + body.inverse_mass);
        end = add(end, multiply(best_normal, best_penetration * node_share));
        const Vec3 arm = subtract(best_contact, state.position);
        if constexpr (couple_dynamic) {
            impulses[vertex].body = best_body;
            contact_arms[vertex] = arm;
            atomicExch(contact_flags + best_body, 1U);
        }
        const Vec3 body_velocity = add(state.linear_velocity,
            cross(state.angular_velocity, arm));
        const Vec3 relative = subtract(velocity, body_velocity);
        const float incoming = dot(relative, best_normal);
        if (incoming < 0.0F) {
            const float mass = 1.0F / fmaxf(inverse_masses[vertex], 1.0e-6F);
            const Vec3 arm_cross = cross(arm, best_normal);
            const float denominator = 1.0F / mass + body.inverse_mass +
                dot(cross(inverse_inertia_world(body, state, arm_cross), arm),
                    best_normal);
            if (denominator > k_epsilon) {
                const Vec3 impulse = multiply(best_normal,
                    -(1.0F + body.restitution) * incoming / denominator);
                const Vec3 velocity_delta = multiply(impulse, 1.0F / mass);
                velocity = add(velocity, velocity_delta);
                if constexpr (couple_dynamic)
                    contact_momentum_delta[vertex] = add(
                        contact_momentum_delta[vertex], impulse);
                // Soft-body velocity is reconstructed from its projected
                // position. Encode dynamic-body impulses in that position so
                // the node retains the same impulse whose opposite is applied
                // to the rigid body after this contact pass.
                if constexpr (couple_dynamic) {
                    if (body.motion == MotionType::dynamic)
                        end = add(end, multiply(velocity_delta, dt));
                }
                impulses[vertex] = {multiply(impulse, -1.0F),
                    multiply(cross(arm, impulse), -1.0F), best_body};
                contact_forces[vertex] = multiply(impulse, 1.0F / dt);
                if constexpr (!couple_dynamic)
                    atomicExch(contact_flags + best_body, 1U);
            }
        }
        if constexpr (couple_dynamic)
            if (body_position_corrections != nullptr)
                body_position_corrections[vertex] = multiply(best_normal,
                    -best_penetration * body.inverse_mass /
                    (inverse_masses[vertex] + body.inverse_mass));
        positions[vertex] = end;
        velocities[vertex] = clamp_length(velocity, maximum_speed);
    } else if (reconstruct_free_velocities) {
        velocities[vertex] = clamp_length(
            multiply(subtract(end, start), 1.0F / dt), maximum_speed);
    }
}

struct SoftTriangleHalfspaceSource {
    const CollisionPlane *planes{};
    Quaternion orientation{};
    Vec3 local_corners[3]{};
    float margin{};
    __host__ __device__ solver::TriangleHalfspace<Vec3> operator[](unsigned face) const {
        const auto plane = planes[face];
        solver::TriangleHalfspace<Vec3> result{};
        result.normal = rotate(orientation,plane.normal);
        for (unsigned corner = 0; corner < 3; ++corner)
            result.depths[corner] = margin + plane.offset - dot(plane.normal,local_corners[corner]);
        return result;
    }
};

__global__ void soft_body_surface_contacts(
    const Vec3 *surface, const std::uint32_t *indices,
    std::uint32_t triangle_count, const BodyParameters *parameters,
    const RigidBodyState *states, const TriangleMeshResource *meshes,
    std::uint32_t body_count, Vec3 *corrections) {
    const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= triangle_count) return;
    const std::uint32_t base = triangle * 3U;
    const Vec3 world[3]{surface[indices[base]], surface[indices[base + 1U]],
                        surface[indices[base + 2U]]};
    for (std::uint32_t corner = 0U; corner < 3U; ++corner)
        corrections[base + corner] = {};
    float depths[2]{};
    Vec3 normals[2]{};
    float corner_depths[2][3]{};
    unsigned contact_bodies[2]{};
    unsigned contact_count = 0U;
    for (std::uint32_t body = 0U; body < body_count; ++body) {
        const TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
        if (mesh.solid_planes == nullptr) continue;
        const RigidBodyState state = states[body];
        Vec3 p[3];
        for (std::uint32_t corner = 0U; corner < 3U; ++corner)
            p[corner] = inverse_rotate(state.orientation,
                                      subtract(world[corner], state.position));
        const float margin = parameters[body].collision_margin;
        const Vec3 expansion{margin, margin, margin};
        if (!bounds_overlap(subtract(component_min(p[0], component_min(p[1], p[2])), expansion),
                            add(component_max(p[0], component_max(p[1], p[2])), expansion),
                            mesh.minimum, mesh.maximum)) continue;
        float separation = -FLT_MAX;
        CollisionPlane support{};
        for (std::uint32_t face = 0U; face < mesh.index_count / 3U; ++face) {
            const CollisionPlane plane = mesh.solid_planes[face];
            const float minimum_side = fminf(dot(plane.normal, p[0]),
                fminf(dot(plane.normal, p[1]), dot(plane.normal, p[2]))) -
                plane.offset;
            if (minimum_side > separation) {
                separation = minimum_side;
                support = plane;
            }
            if (separation >= margin) break;
        }
        if (separation >= margin ||
            (contact_count == 2U && margin - separation <= depths[1])) continue;
        // Face planes alone are conservative near edges. Require actual
        // triangle proximity, or a vertex inside the closed solid.
        bool contact = false;
        for (std::uint32_t corner = 0U; corner < 3U && !contact; ++corner) {
            bool inside = true;
            for (std::uint32_t face = 0U; face < mesh.index_count / 3U; ++face) {
                const CollisionPlane plane = mesh.solid_planes[face];
                if (dot(plane.normal, p[corner]) > plane.offset) {
                    inside = false;
                    break;
                }
            }
            contact = inside;
        }
        for (std::uint32_t face = 0U; face < mesh.index_count && !contact; face += 3U) {
            const Vec3 a = mesh.vertices[mesh.indices[face]];
            const Vec3 b = mesh.vertices[mesh.indices[face + 1U]];
            const Vec3 c = mesh.vertices[mesh.indices[face + 2U]];
            if (!triangle_bounds_overlap(p[0], p[1], p[2], a, b, c, margin)) continue;
            Vec3 on_soft{}, on_rigid{};
            closest_triangle_pair(p[0], p[1], p[2], a, b, c, on_soft, on_rigid);
            contact = length_squared(subtract(on_soft, on_rigid)) < margin * margin;
        }
        if (!contact) continue;
        const unsigned slot = contact_count == 0U ||
            margin - separation > depths[0] ? 0U : 1U;
        if (slot == 0U && contact_count != 0U) {
            depths[1] = depths[0]; normals[1] = normals[0];
            contact_bodies[1] = contact_bodies[0];
            for (unsigned corner = 0U; corner < 3U; ++corner)
                corner_depths[1][corner] = corner_depths[0][corner];
        }
        depths[slot] = margin - separation;
        contact_bodies[slot] = body;
        normals[slot] = rotate(state.orientation, support.normal);
        for (std::uint32_t corner = 0U; corner < 3U; ++corner)
            corner_depths[slot][corner] =
                margin + support.offset - dot(support.normal, p[corner]);
        contact_count = contact_count == 0U ? 1U : 2U;
    }
    for (unsigned corner = 0U; corner < 3U && contact_count != 0U; ++corner) {
        corrections[base + corner] = multiply(normals[0],fmaxf(0.0F,corner_depths[0][corner]));
    }
    if (contact_count == 2U) {
        SoftTriangleHalfspaceSource sources[2]{};
        solver::TriangleHalfspace<Vec3> seeds[2]{};
        unsigned faces[2]{};
        for (unsigned side = 0; side < 2; ++side) {
            const unsigned body = contact_bodies[side];
            const auto mesh = meshes[parameters[body].mesh.index];
            const auto state = states[body];
            sources[side].planes = mesh.solid_planes + mesh.index_count/3U;
            sources[side].orientation = state.orientation;
            sources[side].margin = parameters[body].collision_margin;
            faces[side] = mesh.solid_unique_plane_count;
            seeds[side].normal = normals[side];
            for (unsigned corner = 0; corner < 3; ++corner) {
                sources[side].local_corners[corner] = inverse_rotate(state.orientation,subtract(world[corner],state.position));
                seeds[side].depths[corner] = corner_depths[side][corner];
            }
        }
        Vec3 paired[3]{};
        // A fixed nearly-opposing pair can extrapolate a tiny overlap into a
        // very large correction. Search the actual convex faces, retaining
        // ONE separating pair for the whole triangle; no displacement cap or
        // extra cleanup pass is needed. Invalid geometry keeps the existing
        // deepest-contact fallback.
        if (solver::project_convex_pair_triangle(sources[0],faces[0],sources[1],faces[1],seeds[0],seeds[1],paired))
            for (unsigned corner = 0; corner < 3; ++corner) corrections[base + corner] = paired[corner];
    }
}

__global__ void soft_body_apply_surface_contacts(
    Vec3 *positions, const Vec3 *corrections, const std::uint32_t *offsets,
    const SoftSurfaceInfluence *influences, std::uint32_t node_count) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) return;
    Vec3 correction{};
    for (std::uint32_t entry = offsets[node]; entry < offsets[node + 1U]; ++entry) {
        const SoftSurfaceInfluence influence = influences[entry];
        const Vec3 delta = multiply(corrections[influence.corner], influence.factor);
        const float squared = length_squared(delta);
        if (squared > 1.0e-16F) {
            // Satisfy adjacent face constraints together. Keeping only the
            // longest correction can discard a different contact normal.
            const float remaining = fmaxf(0.0F,
                1.0F - dot(correction, delta) / squared);
            correction = add(correction, multiply(delta, remaining));
        }
    }
    positions[node] = add(positions[node], correction);
}

// The vertex-side contact above deforms the sheet and transfers momentum.
// A second, triangle-side constraint keeps a fast rigid collider from slipping
// between intact, nonfracturing cloth vertices. The broad-phase sphere is
// conservative for any mesh; tearable cloth uses node contacts instead.
