// SPDX-License-Identifier: MIT
// Rigid geometry, contacts, and constraint solver internals.

struct BodyParameters {
    MotionType motion{};
    TriangleMeshId mesh{};
    float inverse_mass{};
    Vec3 inverse_inertia_local{};
    float friction{};
    float restitution{};
    float linear_damping{};
    float angular_damping{};
    float maximum_linear_speed{};
    float maximum_angular_speed{};
    float collision_margin{};
    std::uint64_t user_data{};
};

struct BodyAccumulator {
    Vec3 force{};
    Vec3 torque{};
    Vec3 impulse{};
    Vec3 angular_impulse{};
};

struct KinematicTarget {
    RigidBodyState state{};
    bool active{};
};

struct RigidConstraintResource {
    RigidConstraintOptions options{};
    RigidConstraintState state{};
    std::uint32_t generation{1U};
    bool alive{};
};

struct HingeContactFrame {
    Vec3 anchor{};
    Vec3 axis{};
    Vec3 local_anchor{};
    bool present{};
    bool fixed{};
};

struct Contact {
    Vec3 normal{};
    Vec3 point{};
    // Positive inside solver rest offset; negative while speculative.
    float penetration{};
    HingeContactFrame body_hinge{};
    HingeContactFrame collider_hinge{};
};

struct ContactManifold {
    Contact contacts[8]{};
    std::uint32_t count{};
};

constexpr float k_rigid_surface_tolerance = 1.0e-5F;
constexpr float k_rigid_rest_offset = 1.0e-3F;

__host__ __device__ float rigid_rest_offset(float margin) noexcept {
    return fminf(margin, k_rigid_rest_offset);
}

struct LeafPair {
    std::uint32_t body_first{};
    std::uint32_t body_count{};
    std::uint32_t collider_first{};
    std::uint32_t collider_count{};
};

constexpr std::uint32_t k_max_leaf_pairs_per_body_pair = 512U;
// Keep the fast leaf-pair cache proportional to body capacity. Dense worlds
// retain exact contacts through the serial fallback instead of reserving one
// 512-entry cache for every possible body pair.
constexpr std::uint32_t k_leaf_pair_cache_slots_per_body = 8U;
constexpr std::uint32_t k_minimum_leaf_pair_cache_slots = 4'096U;
constexpr std::uint32_t k_leaf_pair_overflow =
    std::numeric_limits<std::uint32_t>::max();
// 24 colors cover the 1,000-body pile-up without serial overflow. Fewer
// rounds pay for their saved launches with much costlier overflow work.
constexpr std::uint32_t k_contact_color_count = 24U;
constexpr std::uint8_t k_contact_color_overflow = 0xffU;

struct AppliedContactImpulse {
    float normal{};
    Vec3 friction{};
};

struct BvhNode {
    Vec3 minimum{};
    Vec3 maximum{};
    std::uint32_t left{};
    std::uint32_t right{};
    std::uint32_t first_triangle{};
    std::uint32_t triangle_count{};
};

struct WorldAabb {
    Vec3 minimum{};
    Vec3 maximum{};
};

struct CollisionPlane {
    Vec3 normal{};
    float offset{};
};

struct TriangleMeshResource {
    Vec3 *vertices{};
    std::uint32_t *indices{};
    std::uint32_t vertex_count{};
    std::uint32_t index_count{};
    std::uint32_t generation{};
    bool alive{};
    Vec3 minimum{};
    Vec3 maximum{};
    Vec3 bounding_center{};
    float bounding_radius{};
    Vec3 unit_inertia{};
    BvhNode *bvh_nodes{};
    std::uint32_t bvh_node_count{};
    std::uint32_t *bvh_leaves{};
    std::uint32_t bvh_leaf_count{};
    // Present only when the authored triangles form a closed convex solid.
    CollisionPlane *solid_planes{};
};

__device__ float rotational_motion_bound(
    const RigidBodyState &previous, const RigidBodyState &current,
    const TriangleMeshResource &mesh) noexcept {
    const float orientation_dot = clamp_scalar(
        fabsf(previous.orientation.x * current.orientation.x +
              previous.orientation.y * current.orientation.y +
              previous.orientation.z * current.orientation.z +
              previous.orientation.w * current.orientation.w),
        0.0F, 1.0F);
    const float sine_half_angle =
        sqrtf(fmaxf(0.0F, 1.0F - orientation_dot * orientation_dot));
    const Vec3 maximum_absolute{
        fmaxf(fabsf(mesh.minimum.x), fabsf(mesh.maximum.x)),
        fmaxf(fabsf(mesh.minimum.y), fabsf(mesh.maximum.y)),
        fmaxf(fabsf(mesh.minimum.z), fabsf(mesh.maximum.z))};
    return 2.0F * vector_length(maximum_absolute) * sine_half_angle;
}

__device__ bool requires_swept_contact(
    const RigidBodyState &previous, const RigidBodyState &current,
    const TriangleMeshResource &mesh, float threshold) noexcept {
    const float translation =
        vector_length(subtract(current.position, previous.position));
    return translation + rotational_motion_bound(previous, current, mesh) >
           threshold;
}

__device__ bool requires_swept_pair_contact(
    const RigidBodyState &previous_body, const RigidBodyState &body,
    const TriangleMeshResource &body_mesh,
    const RigidBodyState &previous_collider,
    const RigidBodyState &collider,
    const TriangleMeshResource &collider_mesh, float threshold) noexcept {
    const Vec3 relative_translation = subtract(
        subtract(body.position, previous_body.position),
        subtract(collider.position, previous_collider.position));
    const float body_rotation =
        rotational_motion_bound(previous_body, body, body_mesh);
    const float collider_rotation = rotational_motion_bound(
        previous_collider, collider, collider_mesh);
    return vector_length(relative_translation) + body_rotation +
               collider_rotation > threshold;
}

__device__ bool bounding_spheres_may_contact(
    const RigidBodyState &previous_body, const RigidBodyState &body,
    const TriangleMeshResource &body_mesh,
    const RigidBodyState &previous_collider,
    const RigidBodyState &collider,
    const TriangleMeshResource &collider_mesh, float margin) noexcept {
    const Vec3 previous_relative = subtract(
        add(previous_body.position,
            rotate(previous_body.orientation, body_mesh.bounding_center)),
        add(previous_collider.position,
            rotate(previous_collider.orientation,
                   collider_mesh.bounding_center)));
    const Vec3 current_relative = subtract(
        add(body.position, rotate(body.orientation, body_mesh.bounding_center)),
        add(collider.position,
            rotate(collider.orientation, collider_mesh.bounding_center)));
    const Vec3 movement = subtract(current_relative, previous_relative);
    const float squared_movement = length_squared(movement);
    const float time = squared_movement > k_epsilon * k_epsilon
        ? clamp_scalar(-dot(previous_relative, movement) / squared_movement,
                       0.0F, 1.0F)
        : 0.0F;
    const Vec3 nearest = add(previous_relative, multiply(movement, time));
    const float radius = body_mesh.bounding_radius +
        collider_mesh.bounding_radius + margin + 1.0e-5F +
        rotational_motion_bound(previous_body, body, body_mesh) +
        rotational_motion_bound(previous_collider, collider, collider_mesh);
    return length_squared(nearest) <= radius * radius;
}

__host__ __device__ void closest_segments(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2,
                                          Vec3 &c1, Vec3 &c2) noexcept {
    const Vec3 d1 = subtract(q1, p1);
    const Vec3 d2 = subtract(q2, p2);
    const Vec3 r = subtract(p1, p2);
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    float s = 0.0F;
    float t = 0.0F;

    const float squared_epsilon = k_epsilon * k_epsilon;
    if (a <= squared_epsilon && e <= squared_epsilon) {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= squared_epsilon) {
        t = clamp_scalar(f / e, 0.0F, 1.0F);
    } else {
        const float c = dot(d1, r);
        if (e <= squared_epsilon) {
            s = clamp_scalar(-c / a, 0.0F, 1.0F);
        } else {
            const float b = dot(d1, d2);
            const float denominator = a * e - b * b;
            // The denominator has units length^4. An absolute tolerance
            // classified ordinary centimetre-scale edges as parallel.
            if (fabsf(denominator) > k_epsilon * a * e) {
                s = clamp_scalar((b * f - c * e) / denominator, 0.0F, 1.0F);
            }
            t = (b * s + f) / e;
            if (t < 0.0F) {
                t = 0.0F;
                s = clamp_scalar(-c / a, 0.0F, 1.0F);
            } else if (t > 1.0F) {
                t = 1.0F;
                s = clamp_scalar((b - c) / a, 0.0F, 1.0F);
            }
        }
    }
    c1 = add(p1, multiply(d1, s));
    c2 = add(p2, multiply(d2, t));
}

__host__ __device__ Vec3 closest_on_triangle(Vec3 point, Vec3 a, Vec3 b,
                                             Vec3 c) noexcept {
    const Vec3 ab = subtract(b, a);
    const Vec3 ac = subtract(c, a);
    const Vec3 ap = subtract(point, a);
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) {
        return a;
    }
    const Vec3 bp = subtract(point, b);
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) {
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        return add(a, multiply(ab, d1 / (d1 - d3)));
    }
    const Vec3 cp = subtract(point, c);
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) {
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        return add(a, multiply(ac, d2 / (d2 - d6)));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && d4 - d3 >= 0.0F && d5 - d6 >= 0.0F) {
        return add(b, multiply(subtract(c, b),
                               (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    const float inverse = 1.0F / (va + vb + vc);
    return add(a, add(multiply(ab, vb * inverse),
                      multiply(ac, vc * inverse)));
}

__host__ __device__ bool point_in_triangle(Vec3 point, Vec3 a, Vec3 b,
                                           Vec3 c, Vec3 normal) noexcept {
    // These edge tests and |normal|^2 both have units length^4. A fixed
    // tolerance admits points far outside small triangles (gear teeth),
    // producing false intersections and recovery torques through real gaps.
    const float tolerance = -1.0e-5F * length_squared(normal);
    return dot(cross(subtract(b, a), subtract(point, a)), normal) >= tolerance &&
           dot(cross(subtract(c, b), subtract(point, b)), normal) >= tolerance &&
           dot(cross(subtract(a, c), subtract(point, c)), normal) >= tolerance;
}

__host__ __device__ void consider_closest_pair(
    Vec3 on_segment, Vec3 on_triangle, float &best_squared,
    Vec3 &segment_point, Vec3 &triangle_point) noexcept {
    const float squared = length_squared(subtract(on_segment, on_triangle));
    if (squared < best_squared) {
        best_squared = squared;
        segment_point = on_segment;
        triangle_point = on_triangle;
    }
}

__host__ __device__ Vec3 transform_point(const RigidBodyState &state,
                                         Vec3 point) noexcept {
    return add(state.position, rotate(state.orientation, point));
}

__host__ __device__ float component(Vec3 value, std::uint32_t axis) noexcept {
    return axis == 0U ? value.x : axis == 1U ? value.y : value.z;
}

__host__ __device__ Vec3 basis_axis(std::uint32_t axis) noexcept {
    return axis == 0U ? Vec3{1.0F, 0.0F, 0.0F}
                      : axis == 1U ? Vec3{0.0F, 1.0F, 0.0F}
                                   : Vec3{0.0F, 0.0F, 1.0F};
}

__host__ __device__ bool same_rigid_body_id(
    RigidBodyId left, RigidBodyId right) noexcept {
    return left.index == right.index && left.generation == right.generation;
}

struct BoundsTransform {
    Vec3 position{};
    Vec3 axis_x{};
    Vec3 axis_y{};
    Vec3 axis_z{};
};

__device__ BoundsTransform bounds_transform(
    const RigidBodyState &state) noexcept {
    return {state.position,
            rotate(state.orientation, {1.0F, 0.0F, 0.0F}),
            rotate(state.orientation, {0.0F, 1.0F, 0.0F}),
            rotate(state.orientation, {0.0F, 0.0F, 1.0F})};
}

__host__ __device__ Vec3 component_min(Vec3 first, Vec3 second) noexcept {
    return {fminf(first.x, second.x), fminf(first.y, second.y),
            fminf(first.z, second.z)};
}

__host__ __device__ Vec3 component_max(Vec3 first, Vec3 second) noexcept {
    return {fmaxf(first.x, second.x), fmaxf(first.y, second.y),
            fmaxf(first.z, second.z)};
}

__device__ void transformed_bounds(Vec3 local_minimum, Vec3 local_maximum,
                                   const BoundsTransform &transform, float margin,
                                   Vec3 &minimum, Vec3 &maximum) noexcept {
    const Vec3 local_center = multiply(add(local_minimum, local_maximum), 0.5F);
    const Vec3 local_half = multiply(subtract(local_maximum, local_minimum), 0.5F);
    const Vec3 world_center =
        add(transform.position,
            add(multiply(transform.axis_x, local_center.x),
                add(multiply(transform.axis_y, local_center.y),
                    multiply(transform.axis_z, local_center.z))));
    const Vec3 world_half{
        fabsf(transform.axis_x.x) * local_half.x +
            fabsf(transform.axis_y.x) * local_half.y +
            fabsf(transform.axis_z.x) * local_half.z,
        fabsf(transform.axis_x.y) * local_half.x +
            fabsf(transform.axis_y.y) * local_half.y +
            fabsf(transform.axis_z.y) * local_half.z,
        fabsf(transform.axis_x.z) * local_half.x +
            fabsf(transform.axis_y.z) * local_half.y +
            fabsf(transform.axis_z.z) * local_half.z};
    const Vec3 expansion{margin, margin, margin};
    minimum = subtract(subtract(world_center, world_half), expansion);
    maximum = add(add(world_center, world_half), expansion);
}

__device__ void transformed_motion_bounds(
    Vec3 local_minimum, Vec3 local_maximum,
    const BoundsTransform &previous_transform,
    const BoundsTransform &current_transform, bool swept, float margin,
    Vec3 &minimum, Vec3 &maximum) noexcept {
    transformed_bounds(local_minimum, local_maximum, current_transform, margin,
                       minimum, maximum);
    if (!swept) {
        return;
    }
    Vec3 previous_minimum{};
    Vec3 previous_maximum{};
    transformed_bounds(local_minimum, local_maximum, previous_transform, margin,
                       previous_minimum, previous_maximum);
    minimum = component_min(minimum, previous_minimum);
    maximum = component_max(maximum, previous_maximum);
}

__host__ __device__ bool bounds_overlap(Vec3 minimum_a, Vec3 maximum_a,
                                        Vec3 minimum_b,
                                        Vec3 maximum_b) noexcept {
    return minimum_a.x <= maximum_b.x && maximum_a.x >= minimum_b.x &&
           minimum_a.y <= maximum_b.y && maximum_a.y >= minimum_b.y &&
           minimum_a.z <= maximum_b.z && maximum_a.z >= minimum_b.z;
}

__host__ __device__ bool triangle_bounds_overlap(
    Vec3 a0, Vec3 a1, Vec3 a2, Vec3 b0, Vec3 b1, Vec3 b2,
    float margin) noexcept {
    const Vec3 expansion{margin, margin, margin};
    const Vec3 minimum_a =
        subtract(component_min(a0, component_min(a1, a2)), expansion);
    const Vec3 maximum_a =
        add(component_max(a0, component_max(a1, a2)), expansion);
    const Vec3 minimum_b = component_min(b0, component_min(b1, b2));
    const Vec3 maximum_b = component_max(b0, component_max(b1, b2));
    return bounds_overlap(minimum_a, maximum_a, minimum_b, maximum_b);
}

__host__ __device__ bool segment_hits_triangle(
    Vec3 first, Vec3 second, Vec3 a, Vec3 b, Vec3 c,
    Vec3 &intersection) noexcept {
    const Vec3 normal = cross(subtract(b, a), subtract(c, a));
    const Vec3 direction = subtract(second, first);
    const float denominator = dot(normal, direction);
    if (length_squared(normal) <= k_epsilon * k_epsilon ||
        fabsf(denominator) <= k_epsilon) {
        return false;
    }
    const float amount = dot(normal, subtract(a, first)) / denominator;
    if (amount < 0.0F || amount > 1.0F) {
        return false;
    }
    intersection = add(first, multiply(direction, amount));
    return point_in_triangle(intersection, a, b, c, normal);
}

__host__ __device__ void closest_triangle_pair(
    Vec3 a0, Vec3 a1, Vec3 a2, Vec3 b0, Vec3 b1, Vec3 b2,
    Vec3 &point_a, Vec3 &point_b) noexcept {
    const Vec3 a[3]{a0, a1, a2};
    const Vec3 b[3]{b0, b1, b2};
    Vec3 intersection{};
    for (std::uint32_t edge = 0U; edge < 3U; ++edge) {
        if (segment_hits_triangle(
                a[edge], a[(edge + 1U) % 3U], b0, b1, b2,
                intersection)) {
            point_a = point_b = intersection;
            return;
        }
    }
    for (std::uint32_t edge = 0U; edge < 3U; ++edge) {
        if (segment_hits_triangle(
                b[edge], b[(edge + 1U) % 3U], a0, a1, a2,
                intersection)) {
            point_a = point_b = intersection;
            return;
        }
    }

    float best_squared = FLT_MAX;
    for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
        const Vec3 on_triangle =
            closest_on_triangle(a[vertex], b0, b1, b2);
        consider_closest_pair(a[vertex], on_triangle, best_squared,
                              point_a, point_b);
    }
    for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
        const Vec3 on_triangle =
            closest_on_triangle(b[vertex], a0, a1, a2);
        consider_closest_pair(on_triangle, b[vertex], best_squared,
                              point_a, point_b);
    }
    for (std::uint32_t edge_a = 0U; edge_a < 3U; ++edge_a) {
        for (std::uint32_t edge_b = 0U; edge_b < 3U; ++edge_b) {
            Vec3 on_a{};
            Vec3 on_b{};
            closest_segments(a[edge_a], a[(edge_a + 1U) % 3U],
                             b[edge_b], b[(edge_b + 1U) % 3U],
                             on_a, on_b);
            consider_closest_pair(on_a, on_b, best_squared,
                                  point_a, point_b);
        }
    }
}

__device__ void add_manifold_contact(ContactManifold &manifold,
                                     Contact candidate,
                                     float point_spacing) noexcept {
    const float minimum_spacing_squared = point_spacing * point_spacing;
    for (std::uint32_t index = 0; index < manifold.count; ++index) {
        if (length_squared(subtract(candidate.point,
                                    manifold.contacts[index].point)) <
            minimum_spacing_squared) {
            if (candidate.penetration >
                    manifold.contacts[index].penetration) {
                manifold.contacts[index] = candidate;
            }
            return;
        }
    }
    if (manifold.count < 8U) {
        manifold.contacts[manifold.count++] = candidate;
        return;
    }
    std::uint32_t shallowest = 0U;
    for (std::uint32_t index = 1; index < manifold.count; ++index) {
        if (manifold.contacts[index].penetration <
            manifold.contacts[shallowest].penetration) {
            shallowest = index;
        }
    }
    if (candidate.penetration > manifold.contacts[shallowest].penetration) {
        manifold.contacts[shallowest] = candidate;
    }
}

__device__ float contact_normal_speed(
    const RigidBodyState &body_state,
    const RigidBodyState &collider_state, Vec3 point, Vec3 normal) noexcept {
    const Vec3 body_velocity = add(
        body_state.linear_velocity,
        cross(body_state.angular_velocity,
              subtract(point, body_state.position)));
    const Vec3 collider_velocity = add(
        collider_state.linear_velocity,
        cross(collider_state.angular_velocity,
              subtract(point, collider_state.position)));
    return dot(subtract(body_velocity, collider_velocity), normal);
}

__device__ bool contact_reaches_rest_offset(
    const RigidBodyState &body_state,
    const RigidBodyState &collider_state, Vec3 point, Vec3 normal,
    float distance, float rest_offset, float timestep) noexcept {
    constexpr float velocity_tolerance = 1.0e-5F;
    if (distance <= rest_offset + k_rigid_surface_tolerance) {
        return true;
    }
    const float normal_speed = contact_normal_speed(
        body_state, collider_state, point, normal);
    if (normal_speed > velocity_tolerance) {
        return false;
    }
    return -normal_speed * timestep + rest_offset +
               k_rigid_surface_tolerance >=
           distance;
}

__device__ void collide_triangle_ranges(
    const RigidBodyState &body_state, Vec3 body_reference,
    const HingeContactFrame &body_hinge,
    const TriangleMeshResource &body_mesh,
    std::uint32_t body_first, std::uint32_t body_count,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh, std::uint32_t collider_first,
    std::uint32_t collider_count, const HingeContactFrame &collider_hinge,
    float margin,
    float timestep,
    ContactManifold &manifold) noexcept {
    const float rest_offset = rigid_rest_offset(margin);
    for (std::uint32_t body_triangle = body_first;
         body_triangle < body_first + body_count; ++body_triangle) {
        const std::uint32_t body_index = body_triangle * 3U;
        const Vec3 a0 = transform_point(
            body_state, body_mesh.vertices[body_mesh.indices[body_index]]);
        const Vec3 a1 = transform_point(
            body_state, body_mesh.vertices[body_mesh.indices[body_index + 1U]]);
        const Vec3 a2 = transform_point(
            body_state, body_mesh.vertices[body_mesh.indices[body_index + 2U]]);
        for (std::uint32_t collider_triangle = collider_first;
             collider_triangle < collider_first + collider_count;
             ++collider_triangle) {
            const std::uint32_t collider_index = collider_triangle * 3U;
            const Vec3 b0 = transform_point(
                collider_state,
                collider_mesh.vertices[collider_mesh.indices[collider_index]]);
            const Vec3 b1 = transform_point(
                collider_state,
                collider_mesh.vertices[collider_mesh.indices[collider_index + 1U]]);
            const Vec3 b2 = transform_point(
                collider_state,
                collider_mesh.vertices[collider_mesh.indices[collider_index + 2U]]);
            if (!triangle_bounds_overlap(a0, a1, a2, b0, b1, b2, margin)) {
                continue;
            }
            Vec3 point_a{};
            Vec3 point_b{};
            closest_triangle_pair(a0, a1, a2, b0, b1, b2, point_a, point_b);
            const Vec3 delta = subtract(point_a, point_b);
            const float squared = length_squared(delta);
            if (squared > margin * margin) {
                continue;
            }
            const Vec3 collider_normal = normalized_or(
                cross(subtract(b1, b0), subtract(b2, b0)),
                {0.0F, 1.0F, 0.0F});
            const Vec3 surface_delta = subtract(body_reference, point_b);
            const Vec3 fallback = dot(collider_normal, surface_delta) >= 0.0F
                ? collider_normal
                : multiply(collider_normal, -1.0F);
            const float distance = sqrtf(fmaxf(squared, 0.0F));
            const Vec3 point = multiply(add(point_a, point_b), 0.5F);
            // Once two triangle surfaces are within the contact tolerance,
            // their closest-point delta is numerical noise rather than a
            // reliable direction.  Keep the normal on the body's known side
            // of the contacted triangle so resting bodies cannot slowly
            // migrate through a zero-thickness surface.
            const Vec3 normal = distance <= k_rigid_surface_tolerance
                ? fallback
                : normalized_or(delta, fallback);
            if (!contact_reaches_rest_offset(
                    body_state, collider_state, point, normal, distance,
                    rest_offset, timestep)) {
                continue;
            }
            float penetration = rest_offset - distance +
                                k_rigid_surface_tolerance;
            if (distance <= k_rigid_surface_tolerance) {
                // Triangle intersection has no reliable closest-point depth.
                // Recover by one search shell; swept contacts carry exact
                // remaining travel for fast impacts.
                penetration = margin + k_rigid_surface_tolerance;
            }
            const Contact contact{
                normal, point, penetration, body_hinge, collider_hinge};
            add_manifold_contact(manifold, contact, fmaxf(margin * 2.0F, 1.0e-4F));
        }
    }
}

__device__ void collide_triangle_ranges_swept(
    const RigidBodyState &previous_body_state,
    const RigidBodyState &body_state, Vec3 previous_body_reference,
    Vec3 body_reference, const HingeContactFrame &body_hinge,
    const TriangleMeshResource &body_mesh, std::uint32_t body_first,
    std::uint32_t body_count,
    const RigidBodyState &previous_collider_state,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh,
    std::uint32_t collider_first, std::uint32_t collider_count, float margin,
    float timestep, bool body_moves, bool collider_moves,
    const HingeContactFrame &collider_hinge,
    ContactManifold &manifold) noexcept {
    if (!body_moves && !collider_moves) {
        return;
    }
    const float rest_offset = rigid_rest_offset(margin);
    for (std::uint32_t body_triangle = body_first;
         body_triangle < body_first + body_count; ++body_triangle) {
        const std::uint32_t body_index = body_triangle * 3U;
        Vec3 previous_a[3]{};
        Vec3 current_a[3]{};
        for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
            const Vec3 local =
                body_mesh.vertices[body_mesh.indices[body_index + vertex]];
            previous_a[vertex] = transform_point(previous_body_state, local);
            current_a[vertex] = transform_point(body_state, local);
        }
        Vec3 swept_a_minimum = component_min(previous_a[0], current_a[0]);
        Vec3 swept_a_maximum = component_max(previous_a[0], current_a[0]);
        for (std::uint32_t vertex = 1U; vertex < 3U; ++vertex) {
            swept_a_minimum = component_min(
                swept_a_minimum,
                component_min(previous_a[vertex], current_a[vertex]));
            swept_a_maximum = component_max(
                swept_a_maximum,
                component_max(previous_a[vertex], current_a[vertex]));
        }
        for (std::uint32_t collider_triangle = collider_first;
             collider_triangle < collider_first + collider_count;
             ++collider_triangle) {
            const std::uint32_t collider_index = collider_triangle * 3U;
            Vec3 previous_b[3]{};
            Vec3 current_b[3]{};
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                const Vec3 local = collider_mesh.vertices[
                    collider_mesh.indices[collider_index + vertex]];
                previous_b[vertex] =
                    transform_point(previous_collider_state, local);
                current_b[vertex] = transform_point(collider_state, local);
            }
            Vec3 swept_b_minimum = component_min(previous_b[0], current_b[0]);
            Vec3 swept_b_maximum = component_max(previous_b[0], current_b[0]);
            for (std::uint32_t vertex = 1U; vertex < 3U; ++vertex) {
                swept_b_minimum = component_min(
                    swept_b_minimum,
                    component_min(previous_b[vertex], current_b[vertex]));
                swept_b_maximum = component_max(
                    swept_b_maximum,
                    component_max(previous_b[vertex], current_b[vertex]));
            }
            if (!bounds_overlap(
                    {swept_a_minimum.x - margin,
                     swept_a_minimum.y - margin,
                     swept_a_minimum.z - margin},
                    {swept_a_maximum.x + margin,
                     swept_a_maximum.y + margin,
                     swept_a_maximum.z + margin},
                    swept_b_minimum, swept_b_maximum)) {
                continue;
            }
            Vec3 delta_a[3]{};
            Vec3 delta_b[3]{};
            float speed_bound = 0.0F;
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                delta_a[vertex] = subtract(current_a[vertex], previous_a[vertex]);
                delta_b[vertex] = subtract(current_b[vertex], previous_b[vertex]);
                speed_bound = fmaxf(speed_bound,
                                    vector_length(delta_a[vertex]));
            }
            float collider_speed = 0.0F;
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                collider_speed = fmaxf(collider_speed,
                                        vector_length(delta_b[vertex]));
            }
            speed_bound += collider_speed;
            // Distance between moving triangles depends on their relative
            // motion. Subtracting any common translation preserves a safe
            // Lipschitz bound for all barycentric point pairs.
            const Vec3 common_motion = delta_b[0];
            float relative_body_speed = 0.0F;
            float relative_collider_speed = 0.0F;
            for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                relative_body_speed = fmaxf(
                    relative_body_speed,
                    vector_length(subtract(delta_a[vertex], common_motion)));
                relative_collider_speed = fmaxf(
                    relative_collider_speed,
                    vector_length(subtract(delta_b[vertex], common_motion)));
            }
            speed_bound = fminf(speed_bound,
                                relative_body_speed + relative_collider_speed);
            if (speed_bound <= k_epsilon) {
                continue;
            }

            float time = 0.0F;
            for (std::uint32_t iteration = 0U; iteration < 32U;
                 ++iteration) {
                Vec3 a[3]{};
                Vec3 b[3]{};
                for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
                    a[vertex] = add(previous_a[vertex],
                                    multiply(delta_a[vertex], time));
                    b[vertex] = add(previous_b[vertex],
                                    multiply(delta_b[vertex], time));
                }
                Vec3 point_a{};
                Vec3 point_b{};
                closest_triangle_pair(a[0], a[1], a[2], b[0], b[1], b[2],
                                      point_a, point_b);
                const Vec3 delta = subtract(point_a, point_b);
                const float distance =
                    sqrtf(fmaxf(0.0F, length_squared(delta)));
                if (distance <= rest_offset + k_rigid_surface_tolerance) {
                    if (iteration == 0U) {
                        break;
                    }
                    const Vec3 collider_normal = normalized_or(
                        cross(subtract(b[1], b[0]), subtract(b[2], b[0])),
                        {0.0F, 1.0F, 0.0F});
                    const Vec3 body_center = add(
                        previous_body_state.position,
                        multiply(subtract(body_state.position,
                                          previous_body_state.position),
                                 time));
                    Vec3 fallback =
                        dot(collider_normal,
                            subtract(body_center, point_b)) >= 0.0F
                            ? collider_normal
                            : multiply(collider_normal, -1.0F);
                    if (body_hinge.present) {
                        const Vec3 reference = add(
                            previous_body_reference,
                            multiply(subtract(body_reference,
                                              previous_body_reference),
                                     time));
                        fallback =
                            dot(collider_normal,
                                subtract(reference, point_b)) >= 0.0F
                                ? collider_normal
                                : multiply(collider_normal, -1.0F);
                    }
                    const Vec3 normal =
                        distance <= k_rigid_surface_tolerance
                            ? fallback
                            : normalized_or(delta, fallback);
                    const Vec3 point = multiply(add(point_a, point_b), 0.5F);
                    const float normal_speed = contact_normal_speed(
                        body_state, collider_state, point, normal);
                    if (normal_speed > 1.0e-5F) {
                        break;
                    }
                    const float remaining = fmaxf(
                        0.0F,
                        -normal_speed * timestep * (1.0F - time) - distance);
                    add_manifold_contact(
                        manifold,
                        {normal, point, remaining + rest_offset +
                             k_rigid_surface_tolerance,
                         body_hinge, collider_hinge},
                        fmaxf(margin * 2.0F, 1.0e-4F));
                    break;
                }
                float advancement =
                    (distance - rest_offset) /
                    (speed_bound + k_epsilon) * 0.9F;
                advancement = fmaxf(advancement, 1.0e-5F);
                time += advancement;
                if (time > 1.0F) {
                    break;
                }
            }
        }
    }
}

__device__ ContactManifold collide_meshes(
    const BodyParameters &body, const RigidBodyState &previous_body_state,
    const RigidBodyState &body_state, Vec3 previous_body_reference,
    Vec3 body_reference, const HingeContactFrame &body_hinge,
    const TriangleMeshResource &body_mesh, const BodyParameters &collider,
    const RigidBodyState &previous_collider_state,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh,
    const HingeContactFrame &collider_hinge,
    float timestep) noexcept {
    ContactManifold manifold{};
    const float margin = body.collision_margin + collider.collision_margin;
    const bool swept = requires_swept_pair_contact(
        previous_body_state, body_state, body_mesh,
        previous_collider_state, collider_state, collider_mesh, margin);
    const BoundsTransform previous_body_transform =
        swept ? bounds_transform(previous_body_state) : BoundsTransform{};
    const BoundsTransform previous_collider_transform =
        swept ? bounds_transform(previous_collider_state) : BoundsTransform{};
    Vec3 body_minimum{};
    Vec3 body_maximum{};
    Vec3 collider_minimum{};
    Vec3 collider_maximum{};
    const BoundsTransform body_transform = bounds_transform(body_state);
    const BoundsTransform collider_transform = bounds_transform(collider_state);
    transformed_motion_bounds(body_mesh.minimum, body_mesh.maximum,
                              previous_body_transform, body_transform, swept,
                              margin, body_minimum, body_maximum);
    transformed_motion_bounds(collider_mesh.minimum, collider_mesh.maximum,
                              previous_collider_transform, collider_transform,
                              swept, 0.0F, collider_minimum, collider_maximum);
    if (!bounds_overlap(body_minimum, body_maximum, collider_minimum,
                        collider_maximum)) {
        return manifold;
    }

    struct NodePair {
        std::uint32_t body{};
        std::uint32_t collider{};
    };
    NodePair stack[256]{{0U, 0U}};
    std::uint32_t stack_size = 1U;
    bool overflow = body_mesh.bvh_node_count == 0U ||
                    collider_mesh.bvh_node_count == 0U;
    while (stack_size > 0U && !overflow) {
        const NodePair pair = stack[--stack_size];
        const BvhNode &body_node = body_mesh.bvh_nodes[pair.body];
        const BvhNode &collider_node = collider_mesh.bvh_nodes[pair.collider];
        transformed_motion_bounds(
            body_node.minimum, body_node.maximum, previous_body_transform,
            body_transform, swept, margin, body_minimum, body_maximum);
        transformed_motion_bounds(
            collider_node.minimum, collider_node.maximum,
            previous_collider_transform, collider_transform, swept, 0.0F,
            collider_minimum, collider_maximum);
        if (!bounds_overlap(body_minimum, body_maximum, collider_minimum,
                            collider_maximum)) {
            continue;
        }
        const bool body_leaf = body_node.triangle_count != 0U;
        const bool collider_leaf = collider_node.triangle_count != 0U;
        if (body_leaf && collider_leaf) {
            collide_triangle_ranges(
                body_state, body_reference, body_hinge, body_mesh,
                body_node.first_triangle,
                body_node.triangle_count, collider_state, collider_mesh,
                collider_node.first_triangle, collider_node.triangle_count,
                collider_hinge, margin, timestep, manifold);
            if (swept) {
                collide_triangle_ranges_swept(
                    previous_body_state, body_state, previous_body_reference,
                    body_reference, body_hinge, body_mesh,
                    body_node.first_triangle, body_node.triangle_count,
                    previous_collider_state, collider_state, collider_mesh,
                    collider_node.first_triangle, collider_node.triangle_count,
                    margin, timestep, true, true, collider_hinge, manifold);
            }
            continue;
        }

        const std::uint32_t required = body_leaf || collider_leaf ? 2U : 4U;
        if (stack_size + required > 256U) {
            overflow = true;
            break;
        }
        if (body_leaf) {
            stack[stack_size++] = {pair.body, collider_node.right};
            stack[stack_size++] = {pair.body, collider_node.left};
        } else if (collider_leaf) {
            stack[stack_size++] = {body_node.right, pair.collider};
            stack[stack_size++] = {body_node.left, pair.collider};
        } else {
            stack[stack_size++] = {body_node.right, collider_node.right};
            stack[stack_size++] = {body_node.right, collider_node.left};
            stack[stack_size++] = {body_node.left, collider_node.right};
            stack[stack_size++] = {body_node.left, collider_node.left};
        }
    }
    if (overflow) {
        manifold = {};
        collide_triangle_ranges(
            body_state, body_reference, body_hinge, body_mesh, 0U,
            body_mesh.index_count / 3U,
            collider_state, collider_mesh, 0U,
            collider_mesh.index_count / 3U, collider_hinge, margin, timestep,
            manifold);
        if (swept) {
            collide_triangle_ranges_swept(
                previous_body_state, body_state, previous_body_reference,
                body_reference, body_hinge, body_mesh, 0U,
                body_mesh.index_count / 3U, previous_collider_state,
                collider_state, collider_mesh, 0U,
                collider_mesh.index_count / 3U, margin, timestep, true, true,
                collider_hinge, manifold);
        }
    }
    return manifold;
}

__host__ __device__ Vec3 inverse_inertia_world(
    const BodyParameters &parameters, const RigidBodyState &state,
    Vec3 world_vector) noexcept {
    const Vec3 local = inverse_rotate(state.orientation, world_vector);
    const Vec3 transformed{local.x * parameters.inverse_inertia_local.x,
                           local.y * parameters.inverse_inertia_local.y,
                           local.z * parameters.inverse_inertia_local.z};
    return rotate(state.orientation, transformed);
}

__device__ float fixed_hinge_inverse_moment(
    const BodyParameters &parameters, const RigidBodyState &state,
    const HingeContactFrame &hinge) noexcept {
    if (!hinge.fixed || parameters.inverse_mass <= k_epsilon) return 0.0F;
    const Vec3 local_axis = inverse_rotate(state.orientation, hinge.axis);
    const float center_moment =
        local_axis.x * local_axis.x /
            fmaxf(parameters.inverse_inertia_local.x, k_epsilon) +
        local_axis.y * local_axis.y /
            fmaxf(parameters.inverse_inertia_local.y, k_epsilon) +
        local_axis.z * local_axis.z /
            fmaxf(parameters.inverse_inertia_local.z, k_epsilon);
    const Vec3 center_arm = subtract(state.position, hinge.anchor);
    const Vec3 perpendicular = subtract(
        center_arm, multiply(hinge.axis, dot(center_arm, hinge.axis)));
    const float pivot_moment = center_moment +
        length_squared(perpendicular) / parameters.inverse_mass;
    return pivot_moment > k_epsilon ? 1.0F / pivot_moment : 0.0F;
}

__device__ Vec3 contact_point_velocity(
    const RigidBodyState &state, const HingeContactFrame &hinge,
    Vec3 point) noexcept {
    if (!hinge.fixed) {
        return add(state.linear_velocity,
                   cross(state.angular_velocity,
                         subtract(point, state.position)));
    }
    const Vec3 angular = multiply(
        hinge.axis, dot(state.angular_velocity, hinge.axis));
    return cross(angular, subtract(point, hinge.anchor));
}

__device__ float contact_direction_inverse_mass(
    const BodyParameters &parameters, const RigidBodyState &state,
    const HingeContactFrame &hinge, Vec3 point, Vec3 direction) noexcept {
    if (hinge.fixed) {
        const float jacobian = dot(
            cross(hinge.axis, subtract(point, hinge.anchor)), direction);
        return jacobian * jacobian *
            fixed_hinge_inverse_moment(parameters, state, hinge);
    }
    const Vec3 arm = subtract(point, state.position);
    const Vec3 angular = cross(arm, direction);
    return parameters.inverse_mass +
        dot(cross(inverse_inertia_world(parameters, state, angular), arm),
            direction);
}

__device__ void apply_contact_velocity_impulse(
    const BodyParameters &parameters, RigidBodyState &state,
    const HingeContactFrame &hinge, Vec3 point, Vec3 impulse) noexcept {
    if (parameters.inverse_mass <= 0.0F) return;
    if (hinge.fixed) {
        const float angular_impulse = dot(
            hinge.axis, cross(subtract(point, hinge.anchor), impulse));
        const Vec3 angular_delta = multiply(
            hinge.axis, angular_impulse *
                fixed_hinge_inverse_moment(parameters, state, hinge));
        state.angular_velocity = add(state.angular_velocity, angular_delta);
        state.linear_velocity = add(
            state.linear_velocity,
            cross(angular_delta, subtract(state.position, hinge.anchor)));
        return;
    }
    state.linear_velocity = add(
        state.linear_velocity, multiply(impulse, parameters.inverse_mass));
    state.angular_velocity = add(
        state.angular_velocity,
        inverse_inertia_world(parameters, state,
                              cross(subtract(point, state.position), impulse)));
}

__device__ AppliedContactImpulse apply_contact_impulse(
    const BodyParameters &body, RigidBodyState &state,
    const BodyParameters &collider, RigidBodyState &collider_state,
    const Contact &contact, float timestep) noexcept {
    AppliedContactImpulse applied{};
    const Vec3 body_velocity = contact_point_velocity(
        state, contact.body_hinge, contact.point);
    const Vec3 collider_velocity = contact_point_velocity(
        collider_state, contact.collider_hinge, contact.point);
    Vec3 relative_velocity = subtract(body_velocity, collider_velocity);
    const float normal_speed = dot(relative_velocity, contact.normal);
    const float separation = fmaxf(0.0F, -contact.penetration);
    float target_speed = separation > k_rigid_surface_tolerance
        ? -separation / fmaxf(timestep, k_epsilon)
        : 0.0F;
    const float restitution = fminf(body.restitution, collider.restitution);
    if (separation <= k_rigid_surface_tolerance &&
        normal_speed < 0.0F) {
        target_speed = -restitution * normal_speed;
    }
    if (normal_speed >= target_speed) {
        return applied;
    }

    const float denominator =
        contact_direction_inverse_mass(
            body, state, contact.body_hinge, contact.point, contact.normal) +
        contact_direction_inverse_mass(
            collider, collider_state, contact.collider_hinge,
            contact.point, contact.normal);
    if (denominator <= k_epsilon) {
        return applied;
    }

    const float normal_impulse = (target_speed - normal_speed) / denominator;
    applied.normal = normal_impulse;
    const Vec3 normal_vector = multiply(contact.normal, normal_impulse);
    apply_contact_velocity_impulse(
        body, state, contact.body_hinge, contact.point, normal_vector);
    apply_contact_velocity_impulse(
        collider, collider_state, contact.collider_hinge, contact.point,
        multiply(normal_vector, -1.0F));

    if (separation > k_rigid_surface_tolerance) {
        return applied;
    }
    relative_velocity = subtract(
        contact_point_velocity(state, contact.body_hinge, contact.point),
        contact_point_velocity(collider_state, contact.collider_hinge,
                               contact.point));
    Vec3 tangent = subtract(relative_velocity,
                            multiply(contact.normal,
                                     dot(relative_velocity, contact.normal)));
    const float tangent_length = vector_length(tangent);
    if (tangent_length <= k_epsilon) {
        return applied;
    }
    tangent = multiply(tangent, 1.0F / tangent_length);
    const float tangent_denominator =
        contact_direction_inverse_mass(
            body, state, contact.body_hinge, contact.point, tangent) +
        contact_direction_inverse_mass(
            collider, collider_state, contact.collider_hinge,
            contact.point, tangent);
    if (tangent_denominator <= k_epsilon) {
        return applied;
    }
    float tangent_impulse = -dot(relative_velocity, tangent) / tangent_denominator;
    const float friction_limit =
        sqrtf(body.friction * collider.friction) * normal_impulse;
    tangent_impulse =
        clamp_scalar(tangent_impulse, -friction_limit, friction_limit);
    const Vec3 tangent_vector = multiply(tangent, tangent_impulse);
    applied.friction = tangent_vector;
    apply_contact_velocity_impulse(
        body, state, contact.body_hinge, contact.point, tangent_vector);
    apply_contact_velocity_impulse(
        collider, collider_state, contact.collider_hinge, contact.point,
        multiply(tangent_vector, -1.0F));
    return applied;
}

__device__ void apply_orientation_correction(
    RigidBodyState &state, Vec3 world_rotation) noexcept {
    const Quaternion rotation{world_rotation.x, world_rotation.y,
                              world_rotation.z, 0.0F};
    const Quaternion derivative = quaternion_multiply(
        rotation, state.orientation);
    state.orientation = normalized_quaternion({
        state.orientation.x + 0.5F * derivative.x,
        state.orientation.y + 0.5F * derivative.y,
        state.orientation.z + 0.5F * derivative.z,
        state.orientation.w + 0.5F * derivative.w});
}

__device__ void apply_contact_position_correction(
    const BodyParameters &body, RigidBodyState &state,
    const BodyParameters &collider, RigidBodyState &collider_state,
    const Contact &contact, float penetration) noexcept {
    const auto position_inverse_mass = [&](
        const BodyParameters &parameters, const RigidBodyState &body_state,
        const HingeContactFrame &hinge) {
        if (!hinge.fixed) return parameters.inverse_mass;
        const float jacobian = dot(
            cross(hinge.axis, subtract(contact.point, hinge.anchor)),
            contact.normal);
        return jacobian * jacobian *
            fixed_hinge_inverse_moment(parameters, body_state, hinge);
    };
    const float denominator =
        position_inverse_mass(body, state, contact.body_hinge) +
        position_inverse_mass(collider, collider_state,
                              contact.collider_hinge);
    if (denominator <= k_epsilon || penetration <= 0.0F) {
        return;
    }
    const Vec3 correction = multiply(contact.normal,
                                     penetration / denominator);
    const auto apply = [&](const BodyParameters &parameters,
                           RigidBodyState &body_state,
                           const HingeContactFrame &hinge,
                           Vec3 body_correction) {
        if (parameters.inverse_mass <= 0.0F) return;
        if (!hinge.fixed) {
            body_state.position = add(
                body_state.position,
                multiply(body_correction, parameters.inverse_mass));
            return;
        }
        const float angular_correction = dot(
            hinge.axis,
            cross(subtract(contact.point, hinge.anchor), body_correction)) *
            fixed_hinge_inverse_moment(parameters, body_state, hinge);
        apply_orientation_correction(
            body_state, multiply(hinge.axis, angular_correction));
        body_state.position = subtract(
            hinge.anchor,
            rotate(body_state.orientation, hinge.local_anchor));
    };
    apply(body, state, contact.body_hinge, correction);
    apply(collider, collider_state, contact.collider_hinge,
          multiply(correction, -1.0F));
}

__device__ void resolve_contacts(
    const BodyParameters &body, RigidBodyState &state,
    const BodyParameters &collider, RigidBodyState &collider_state,
    const Contact *contacts, std::uint32_t contact_count,
    float timestep, bool correct_position, RigidContactEvent *debug_events,
    std::uint32_t debug_event_count) noexcept {
    if (contact_count == 0U) {
        return;
    }
    const float inverse_mass_sum = body.inverse_mass + collider.inverse_mass;
    if (correct_position && inverse_mass_sum > k_epsilon) {
        std::uint32_t correction_count = 0U;
        for (std::uint32_t index = 0U; index < contact_count; ++index) {
            if (contacts[index].penetration > 0.0F) {
                ++correction_count;
            }
        }
        const float contact_weight = correction_count > 0U
            ? 1.0F / static_cast<float>(correction_count)
            : 0.0F;
        for (std::uint32_t index = 0; index < contact_count; ++index) {
            if (contacts[index].penetration <= 0.0F) {
                continue;
            }
            apply_contact_position_correction(
                body, state, collider, collider_state, contacts[index],
                (fminf(contacts[index].penetration,
                       contacts[index].body_hinge.fixed ||
                               contacts[index].collider_hinge.fixed
                           ? k_rigid_rest_offset
                           : contacts[index].penetration) +
                 k_rigid_surface_tolerance) * contact_weight);
        }
    }
    for (std::uint32_t index = 0; index < contact_count; ++index) {
        const AppliedContactImpulse applied = apply_contact_impulse(
            body, state, collider, collider_state, contacts[index], timestep);
        if (debug_events != nullptr && index < debug_event_count) {
            debug_events[index].normal_impulse += applied.normal;
            debug_events[index].friction_impulse =
                add(debug_events[index].friction_impulse, applied.friction);
        }
    }
}

__device__ Vec3 quaternion_delta_velocity(Quaternion from, Quaternion to,
                                          float timestep) noexcept {
    Quaternion delta = quaternion_multiply(to, conjugate(from));
    if (delta.w < 0.0F) {
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    }
    delta = normalized_quaternion(delta);
    const float vector_size = sqrtf(delta.x * delta.x + delta.y * delta.y +
                                    delta.z * delta.z);
    if (vector_size <= k_epsilon || timestep <= 0.0F) {
        return {};
    }
    const float angle = 2.0F * atan2f(vector_size, clamp_scalar(delta.w, -1.0F, 1.0F));
    const float scale = angle / (vector_size * timestep);
    return {delta.x * scale, delta.y * scale, delta.z * scale};
}

__global__ void integrate_rigid_bodies_kernel(
    const BodyParameters *parameters, const BodyAccumulator *accumulators,
    const KinematicTarget *targets, const RigidBodyState *input,
    RigidBodyState *output, std::uint32_t count, Vec3 gravity, float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const BodyParameters body = parameters[index];
    const RigidBodyState previous = input[index];
    RigidBodyState next = previous;
    if (body.motion == MotionType::static_body) {
        next.linear_velocity = {};
        next.angular_velocity = {};
        output[index] = next;
        return;
    }

    if (body.motion == MotionType::kinematic) {
        if (targets[index].active) {
            const float fraction = 1.0F / static_cast<float>(remaining_substeps);
            next.position = add(previous.position,
                                multiply(subtract(targets[index].state.position,
                                                  previous.position),
                                         fraction));
            Quaternion target = targets[index].state.orientation;
            const float orientation_dot = previous.orientation.x * target.x +
                                          previous.orientation.y * target.y +
                                          previous.orientation.z * target.z +
                                          previous.orientation.w * target.w;
            if (orientation_dot < 0.0F) {
                target = {-target.x, -target.y, -target.z, -target.w};
            }
            next.orientation = normalized_quaternion(
                {previous.orientation.x + (target.x - previous.orientation.x) * fraction,
                 previous.orientation.y + (target.y - previous.orientation.y) * fraction,
                 previous.orientation.z + (target.z - previous.orientation.z) * fraction,
                 previous.orientation.w + (target.w - previous.orientation.w) * fraction});
            next.linear_velocity = multiply(
                subtract(next.position, previous.position), 1.0F / timestep);
            next.angular_velocity = quaternion_delta_velocity(
                previous.orientation, next.orientation, timestep);
        } else {
            next.linear_velocity = {};
            next.angular_velocity = {};
        }
        output[index] = next;
        return;
    }

    const BodyAccumulator accumulator = accumulators[index];
    next.linear_velocity =
        add(next.linear_velocity,
            multiply(add(gravity, multiply(accumulator.force, body.inverse_mass)),
                     timestep));
    const Vec3 angular_acceleration =
        inverse_inertia_world(body, next, accumulator.torque);
    next.angular_velocity =
        add(next.angular_velocity, multiply(angular_acceleration, timestep));
    if (apply_impulses) {
        next.linear_velocity =
            add(next.linear_velocity,
                multiply(accumulator.impulse, body.inverse_mass));
        next.angular_velocity =
            add(next.angular_velocity,
                inverse_inertia_world(body, next, accumulator.angular_impulse));
    }

    next.linear_velocity = multiply(
        next.linear_velocity, 1.0F / (1.0F + body.linear_damping * timestep));
    next.angular_velocity = multiply(
        next.angular_velocity, 1.0F / (1.0F + body.angular_damping * timestep));
    next.linear_velocity =
        clamp_length(next.linear_velocity, body.maximum_linear_speed);
    next.angular_velocity =
        clamp_length(next.angular_velocity, body.maximum_angular_speed);
    next.position = add(next.position, multiply(next.linear_velocity, timestep));

    const Quaternion angular{next.angular_velocity.x, next.angular_velocity.y,
                             next.angular_velocity.z, 0.0F};
    const Quaternion derivative = quaternion_multiply(angular, next.orientation);
    next.orientation = normalized_quaternion(
        {next.orientation.x + 0.5F * derivative.x * timestep,
         next.orientation.y + 0.5F * derivative.y * timestep,
         next.orientation.z + 0.5F * derivative.z * timestep,
         next.orientation.w + 0.5F * derivative.w * timestep});
    output[index] = next;
}

__device__ std::uint32_t find_rigid_body_dense(
    RigidBodyId id, const RigidBodyId *ids, std::uint32_t count) noexcept {
    for (std::uint32_t index = 0U; index < count; ++index)
        if (ids[index].index == id.index &&
            ids[index].generation == id.generation) return index;
    return k_invalid_dense;
}

__device__ float solve_linear_constraint_axis(
    const BodyParameters &a, RigidBodyState &state_a,
    const BodyParameters &b, RigidBodyState &state_b,
    Vec3 arm_a, Vec3 arm_b, Vec3 axis, float error, float timestep,
    float stiffness, float damping, bool spring) noexcept {
    const Vec3 velocity_a = add(
        state_a.linear_velocity, cross(state_a.angular_velocity, arm_a));
    const Vec3 velocity_b = add(
        state_b.linear_velocity, cross(state_b.angular_velocity, arm_b));
    const float relative_velocity = dot(subtract(velocity_b, velocity_a), axis);
    const Vec3 angular_a = cross(
        inverse_inertia_world(a, state_a, cross(arm_a, axis)), arm_a);
    const Vec3 angular_b = cross(
        inverse_inertia_world(b, state_b, cross(arm_b, axis)), arm_b);
    const float denominator = a.inverse_mass + b.inverse_mass +
        dot(add(angular_a, angular_b), axis);
    if (denominator <= k_epsilon) return 0.0F;
    const float impulse = spring
        ? -(relative_velocity + stiffness * error * timestep) /
              (denominator + damping * timestep)
        : -(relative_velocity + 0.35F * error / timestep) / denominator;
    const Vec3 vector = multiply(axis, impulse);
    if (a.inverse_mass > 0.0F) {
        state_a.linear_velocity = subtract(
            state_a.linear_velocity, multiply(vector, a.inverse_mass));
        state_a.angular_velocity = subtract(
            state_a.angular_velocity,
            inverse_inertia_world(a, state_a, cross(arm_a, vector)));
    }
    if (b.inverse_mass > 0.0F) {
        state_b.linear_velocity = add(
            state_b.linear_velocity, multiply(vector, b.inverse_mass));
        state_b.angular_velocity = add(
            state_b.angular_velocity,
            inverse_inertia_world(b, state_b, cross(arm_b, vector)));
    }
    return fabsf(impulse);
}

__device__ float solve_angular_constraint_axis(
    const BodyParameters &a, RigidBodyState &state_a,
    const BodyParameters &b, RigidBodyState &state_b,
    Vec3 axis, float error, float timestep, float stiffness, float damping,
    bool spring) noexcept {
    const float relative_velocity = dot(
        subtract(state_b.angular_velocity, state_a.angular_velocity), axis);
    const Vec3 inverse_a = inverse_inertia_world(a, state_a, axis);
    const Vec3 inverse_b = inverse_inertia_world(b, state_b, axis);
    const float denominator = dot(add(inverse_a, inverse_b), axis);
    if (denominator <= k_epsilon) return 0.0F;
    const float impulse = spring
        ? -(relative_velocity + stiffness * error * timestep) /
              (denominator + damping * timestep)
        : -(relative_velocity + 0.30F * error / timestep) / denominator;
    if (a.inverse_mass > 0.0F)
        state_a.angular_velocity = subtract(
            state_a.angular_velocity, multiply(inverse_a, impulse));
    if (b.inverse_mass > 0.0F)
        state_b.angular_velocity = add(
            state_b.angular_velocity, multiply(inverse_b, impulse));
    return fabsf(impulse);
}

__device__ float solve_motor_axis(
    const BodyParameters &a, RigidBodyState &state_a,
    const BodyParameters &b, RigidBodyState &state_b, Vec3 axis,
    Vec3 arm_a, Vec3 arm_b, float target_velocity, float maximum_impulse,
    bool angular) noexcept {
    float denominator = 0.0F;
    float relative_velocity = 0.0F;
    if (angular) {
        relative_velocity = dot(
            subtract(state_b.angular_velocity, state_a.angular_velocity), axis);
        denominator = dot(add(inverse_inertia_world(a, state_a, axis),
                              inverse_inertia_world(b, state_b, axis)), axis);
    } else {
        const Vec3 velocity_a = add(
            state_a.linear_velocity, cross(state_a.angular_velocity, arm_a));
        const Vec3 velocity_b = add(
            state_b.linear_velocity, cross(state_b.angular_velocity, arm_b));
        relative_velocity = dot(subtract(velocity_b, velocity_a), axis);
        denominator = a.inverse_mass + b.inverse_mass +
            dot(add(cross(inverse_inertia_world(a, state_a, cross(arm_a, axis)),
                          arm_a),
                    cross(inverse_inertia_world(b, state_b, cross(arm_b, axis)),
                          arm_b)), axis);
    }
    if (denominator <= k_epsilon || maximum_impulse <= 0.0F) return 0.0F;
    const float impulse = clamp_scalar(
        (target_velocity - relative_velocity) / denominator,
        -maximum_impulse, maximum_impulse);
    if (angular) {
        if (a.inverse_mass > 0.0F)
            state_a.angular_velocity = subtract(
                state_a.angular_velocity,
                multiply(inverse_inertia_world(a, state_a, axis), impulse));
        if (b.inverse_mass > 0.0F)
            state_b.angular_velocity = add(
                state_b.angular_velocity,
                multiply(inverse_inertia_world(b, state_b, axis), impulse));
    } else {
        const Vec3 vector = multiply(axis, impulse);
        if (a.inverse_mass > 0.0F) {
            state_a.linear_velocity = subtract(
                state_a.linear_velocity, multiply(vector, a.inverse_mass));
            state_a.angular_velocity = subtract(
                state_a.angular_velocity,
                inverse_inertia_world(a, state_a, cross(arm_a, vector)));
        }
        if (b.inverse_mass > 0.0F) {
            state_b.linear_velocity = add(
                state_b.linear_velocity, multiply(vector, b.inverse_mass));
            state_b.angular_velocity = add(
                state_b.angular_velocity,
                inverse_inertia_world(b, state_b, cross(arm_b, vector)));
        }
    }
    return fabsf(impulse);
}

__device__ Vec3 relative_rotation_vector(
    Quaternion frame_a, Quaternion frame_b) noexcept {
    Quaternion relative = normalized_quaternion(
        quaternion_multiply(conjugate(frame_a), frame_b));
    if (relative.w < 0.0F)
        relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    const float size = sqrtf(relative.x * relative.x + relative.y * relative.y +
                             relative.z * relative.z);
    if (size <= k_epsilon) return {};
    const float angle = 2.0F * atan2f(
        size, clamp_scalar(relative.w, -1.0F, 1.0F));
    return {relative.x * angle / size, relative.y * angle / size,
            relative.z * angle / size};
}

__host__ __device__ bool axis_enabled(
    std::uint8_t mask, std::uint32_t axis) noexcept {
    return (mask & static_cast<std::uint8_t>(1U << axis)) != 0U;
}

__device__ float limit_error(float value, float lower, float upper) noexcept {
    return value < lower ? value - lower : value > upper ? value - upper : 0.0F;
}

__global__ void solve_rigid_constraints_kernel(
    RigidConstraintResource *constraints, std::uint32_t capacity,
    const RigidBodyId *ids, const BodyParameters *parameters,
    RigidBodyState *states, std::uint32_t body_count, float timestep) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    std::uint32_t iterations = 0U;
    for (std::uint32_t index = 0U; index < capacity; ++index) {
        RigidConstraintResource &constraint = constraints[index];
        if (!constraint.alive) continue;
        // Preserve the impulse that broke a constraint for later diagnostics.
        if (!constraint.state.broken)
            constraint.state.applied_impulse = 0.0F;
        constraint.state.enabled = constraint.options.enabled && !constraint.state.broken;
        if (constraint.state.enabled)
            iterations = constraint.options.solver_iterations > iterations
                ? constraint.options.solver_iterations : iterations;
    }
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        for (std::uint32_t index = 0U; index < capacity; ++index) {
            RigidConstraintResource &constraint = constraints[index];
            RigidConstraintOptions &options = constraint.options;
            if (!constraint.alive || !constraint.state.enabled ||
                iteration >= options.solver_iterations) continue;
            const std::uint32_t dense_a = find_rigid_body_dense(
                options.body_a, ids, body_count);
            const std::uint32_t dense_b = find_rigid_body_dense(
                options.body_b, ids, body_count);
            if (dense_a == k_invalid_dense || dense_b == k_invalid_dense) continue;
            const BodyParameters &a = parameters[dense_a];
            const BodyParameters &b = parameters[dense_b];
            RigidBodyState &state_a = states[dense_a];
            RigidBodyState &state_b = states[dense_b];
            const Vec3 arm_a = rotate(state_a.orientation, options.local_anchor_a);
            const Vec3 arm_b = rotate(state_b.orientation, options.local_anchor_b);
            const Vec3 anchor_error = subtract(
                add(state_b.position, arm_b), add(state_a.position, arm_a));
            const Quaternion frame_a = normalized_quaternion(
                quaternion_multiply(state_a.orientation,
                                    options.local_orientation_a));
            const Quaternion frame_b = normalized_quaternion(
                quaternion_multiply(state_b.orientation,
                                    options.local_orientation_b));
            const Vec3 rotation_error = relative_rotation_vector(frame_a, frame_b);
            const Vec3 hinge_alignment_error = cross(
                rotate(frame_a, {0.0F, 0.0F, 1.0F}),
                rotate(frame_b, {0.0F, 0.0F, 1.0F}));
            float applied = 0.0F;
            for (std::uint32_t axis_index = 0U; axis_index < 3U; ++axis_index) {
                const Vec3 world_axis = rotate(frame_a, basis_axis(axis_index));
                const bool generic = options.type == RigidConstraintType::generic ||
                    options.type == RigidConstraintType::generic_spring;
                const bool motor = options.type == RigidConstraintType::motor;
                const bool linear_lock =
                    options.type == RigidConstraintType::fixed ||
                    options.type == RigidConstraintType::point ||
                    options.type == RigidConstraintType::hinge ||
                    ((options.type == RigidConstraintType::slider ||
                      options.type == RigidConstraintType::piston) && axis_index != 0U) ||
                    (motor && (axis_index != 0U || !options.motor.linear_enabled));
                const bool linear_spring =
                    options.type == RigidConstraintType::generic_spring &&
                    axis_enabled(options.linear_springs.axes, axis_index);
                const bool linear_limit = generic &&
                    axis_enabled(options.linear_limits.axes, axis_index);
                float linear_error = dot(anchor_error, world_axis);
                if (linear_limit && !linear_lock && !linear_spring)
                    linear_error = limit_error(
                        linear_error, component(options.linear_limits.lower, axis_index),
                        component(options.linear_limits.upper, axis_index));
                if (linear_lock || linear_spring ||
                    (linear_limit && linear_error != 0.0F)) {
                    applied += solve_linear_constraint_axis(
                        a, state_a, b, state_b, arm_a, arm_b, world_axis,
                        linear_error, timestep,
                        component(options.linear_springs.stiffness, axis_index),
                        component(options.linear_springs.damping, axis_index),
                        linear_spring && !linear_lock);
                }

                const bool angular_lock =
                    options.type == RigidConstraintType::fixed ||
                    options.type == RigidConstraintType::slider ||
                    (options.type == RigidConstraintType::hinge && axis_index != 2U) ||
                    (options.type == RigidConstraintType::piston && axis_index != 0U) ||
                    (motor && (axis_index != 0U || !options.motor.angular_enabled));
                const bool angular_spring =
                    options.type == RigidConstraintType::generic_spring &&
                    axis_enabled(options.angular_springs.axes, axis_index);
                const bool angular_limit = generic &&
                    axis_enabled(options.angular_limits.axes, axis_index);
                float angular_error =
                    options.type == RigidConstraintType::hinge &&
                            axis_index != 2U
                        ? dot(hinge_alignment_error, world_axis)
                        : component(rotation_error, axis_index);
                if (angular_limit && !angular_lock && !angular_spring)
                    angular_error = limit_error(
                        angular_error, component(options.angular_limits.lower, axis_index),
                        component(options.angular_limits.upper, axis_index));
                if (angular_lock || angular_spring ||
                    (angular_limit && angular_error != 0.0F)) {
                    applied += solve_angular_constraint_axis(
                        a, state_a, b, state_b, world_axis, angular_error,
                        timestep,
                        component(options.angular_springs.stiffness, axis_index),
                        component(options.angular_springs.damping, axis_index),
                        angular_spring && !angular_lock);
                }
            }
            if (options.type == RigidConstraintType::hinge &&
                axis_enabled(options.angular_limits.axes, 2U)) {
                const float error = limit_error(
                    rotation_error.z, options.angular_limits.lower.z,
                    options.angular_limits.upper.z);
                if (error != 0.0F)
                    applied += solve_angular_constraint_axis(
                        a, state_a, b, state_b,
                        rotate(frame_a, {0.0F, 0.0F, 1.0F}), error,
                        timestep, 0.0F, 0.0F, false);
            }
            if (options.type == RigidConstraintType::slider &&
                axis_enabled(options.linear_limits.axes, 0U)) {
                const float value = dot(
                    anchor_error, rotate(frame_a, {1.0F, 0.0F, 0.0F}));
                const float error = limit_error(
                    value, options.linear_limits.lower.x,
                    options.linear_limits.upper.x);
                if (error != 0.0F)
                    applied += solve_linear_constraint_axis(
                        a, state_a, b, state_b, arm_a, arm_b,
                        rotate(frame_a, {1.0F, 0.0F, 0.0F}), error,
                        timestep, 0.0F, 0.0F, false);
            }
            if (options.type == RigidConstraintType::piston) {
                const Vec3 piston_axis = rotate(frame_a, {1.0F, 0.0F, 0.0F});
                if (axis_enabled(options.linear_limits.axes, 0U)) {
                    const float error = limit_error(
                        dot(anchor_error, piston_axis), options.linear_limits.lower.x,
                        options.linear_limits.upper.x);
                    if (error != 0.0F)
                        applied += solve_linear_constraint_axis(
                            a, state_a, b, state_b, arm_a, arm_b, piston_axis,
                            error, timestep, 0.0F, 0.0F, false);
                }
                if (axis_enabled(options.angular_limits.axes, 0U)) {
                    const float error = limit_error(
                        rotation_error.x, options.angular_limits.lower.x,
                        options.angular_limits.upper.x);
                    if (error != 0.0F)
                        applied += solve_angular_constraint_axis(
                            a, state_a, b, state_b, piston_axis, error,
                            timestep, 0.0F, 0.0F, false);
                }
            }
            if (options.type == RigidConstraintType::motor) {
                const Vec3 motor_axis = rotate(frame_a, {1.0F, 0.0F, 0.0F});
                const float inverse_iterations =
                    1.0F / static_cast<float>(options.solver_iterations);
                if (options.motor.linear_enabled)
                    applied += solve_motor_axis(
                        a, state_a, b, state_b, motor_axis, arm_a, arm_b,
                        options.motor.linear_target_velocity,
                        options.motor.linear_maximum_impulse * inverse_iterations,
                        false);
                if (options.motor.angular_enabled)
                    applied += solve_motor_axis(
                        a, state_a, b, state_b, motor_axis, arm_a, arm_b,
                        options.motor.angular_target_velocity,
                        options.motor.angular_maximum_impulse * inverse_iterations,
                        true);
            }
            constraint.state.applied_impulse += applied;
            if (options.breaking_impulse_threshold > 0.0F &&
                constraint.state.applied_impulse >
                    options.breaking_impulse_threshold) {
                constraint.state.broken = true;
                constraint.state.enabled = false;
            }
        }
    }
    // Velocity constraints keep hinge motion tangent to the anchor, but a
    // finite quaternion step follows the chord of that arc. Project the
    // positional anchor error once per substep so off-centre hinges cannot
    // accumulate drift during full rotations or after contact impulses.
    for (std::uint32_t index = 0U; index < capacity; ++index) {
        const RigidConstraintResource &constraint = constraints[index];
        if (!constraint.alive || !constraint.state.enabled ||
            constraint.options.type != RigidConstraintType::hinge) {
            continue;
        }
        const std::uint32_t dense_a = find_rigid_body_dense(
            constraint.options.body_a, ids, body_count);
        const std::uint32_t dense_b = find_rigid_body_dense(
            constraint.options.body_b, ids, body_count);
        if (dense_a == k_invalid_dense || dense_b == k_invalid_dense) {
            continue;
        }
        const BodyParameters &a = parameters[dense_a];
        const BodyParameters &b = parameters[dense_b];
        const float inverse_mass_sum = a.inverse_mass + b.inverse_mass;
        if (inverse_mass_sum <= k_epsilon) {
            continue;
        }
        RigidBodyState &state_a = states[dense_a];
        RigidBodyState &state_b = states[dense_b];
        const Vec3 anchor_a = add(
            state_a.position,
            rotate(state_a.orientation,
                   constraint.options.local_anchor_a));
        const Vec3 anchor_b = add(
            state_b.position,
            rotate(state_b.orientation,
                   constraint.options.local_anchor_b));
        const Vec3 error = subtract(anchor_b, anchor_a);
        state_a.position = add(
            state_a.position,
            multiply(error, a.inverse_mass / inverse_mass_sum));
        state_b.position = subtract(
            state_b.position,
            multiply(error, b.inverse_mass / inverse_mass_sum));
    }
}

__device__ bool constrained_collision_disabled(
    RigidBodyId first, RigidBodyId second,
    const RigidConstraintResource *constraints,
    std::uint32_t constraint_capacity) noexcept {
    for (std::uint32_t index = 0U; index < constraint_capacity; ++index) {
        const RigidConstraintResource &constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken || !constraint.options.disable_collisions)
            continue;
        const bool forward = same_rigid_body_id(
            constraint.options.body_a, first) && same_rigid_body_id(
            constraint.options.body_b, second);
        const bool reverse = same_rigid_body_id(
            constraint.options.body_a, second) && same_rigid_body_id(
            constraint.options.body_b, first);
        if (forward || reverse) return true;
    }
    return false;
}

__device__ HingeContactFrame rigid_hinge_contact_frame(
    const BodyParameters *parameters, const RigidBodyId *ids,
    std::uint32_t body_count, std::uint32_t dense,
    const RigidBodyState *states, const RigidBodyState &state,
    const RigidConstraintResource *constraints,
    std::uint32_t constraint_capacity, Vec3 &reference) noexcept {
    // A compound mesh can have its body origin far from its rotating part.
    // Its hinge anchor is the stable inside/outside reference for coincident
    // triangle contacts, where the closest-point delta has no direction.
    HingeContactFrame result{};
    if (parameters[dense].inverse_mass <= 0.0F) {
        reference = state.position;
        return result;
    }
    const RigidBodyId body = ids[dense];
    for (std::uint32_t index = 0U; index < constraint_capacity; ++index) {
        const RigidConstraintResource &constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken ||
            constraint.options.type != RigidConstraintType::hinge)
            continue;
        const bool is_a = same_rigid_body_id(
            constraint.options.body_a, body);
        const bool is_b = same_rigid_body_id(
            constraint.options.body_b, body);
        if (!is_a && !is_b) continue;
        result.local_anchor = is_a
            ? constraint.options.local_anchor_a
            : constraint.options.local_anchor_b;
        const Quaternion local_orientation = is_a
            ? constraint.options.local_orientation_a
            : constraint.options.local_orientation_b;
        result.anchor = add(
            state.position, rotate(state.orientation, result.local_anchor));
        result.axis = normalized_or(
            rotate(quaternion_multiply(state.orientation, local_orientation),
                   {0.0F, 0.0F, 1.0F}),
            {0.0F, 0.0F, 1.0F});
        result.present = true;
        const RigidBodyId other = is_a
            ? constraint.options.body_b : constraint.options.body_a;
        const std::uint32_t other_dense = find_rigid_body_dense(
            other, ids, body_count);
        result.fixed = other_dense != k_invalid_dense &&
            parameters[other_dense].motion == MotionType::static_body;
        if (result.fixed) {
            const Vec3 other_local_anchor = is_a
                ? constraint.options.local_anchor_b
                : constraint.options.local_anchor_a;
            const Quaternion other_local_orientation = is_a
                ? constraint.options.local_orientation_b
                : constraint.options.local_orientation_a;
            const RigidBodyState &other_state = states[other_dense];
            result.anchor = add(
                other_state.position,
                rotate(other_state.orientation, other_local_anchor));
            result.axis = normalized_or(
                rotate(quaternion_multiply(other_state.orientation,
                                           other_local_orientation),
                       {0.0F, 0.0F, 1.0F}),
                result.axis);
        }
        reference = result.anchor;
        return result;
    }
    reference = state.position;
    return result;
}

__global__ void compute_rigid_world_bounds_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states,
    std::uint32_t count, const TriangleMeshResource *meshes,
    WorldAabb *world_bounds) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const TriangleMeshResource &mesh = meshes[parameters[index].mesh.index];
    const bool swept = requires_swept_contact(
        previous_states[index], states[index], mesh,
        parameters[index].collision_margin);
    const BoundsTransform current_transform = bounds_transform(states[index]);
    BoundsTransform previous_transform{};
    if (swept) {
        previous_transform = bounds_transform(previous_states[index]);
    }
    transformed_motion_bounds(
        mesh.minimum, mesh.maximum, previous_transform, current_transform,
        swept, 0.0F,
        world_bounds[index].minimum, world_bounds[index].maximum);
}

__global__ void broad_phase_rigid_pairs_kernel(
    const BodyParameters *parameters, const WorldAabb *world_bounds,
    const RigidBodyState *previous_states, const RigidBodyState *states,
    const TriangleMeshResource *meshes, const RigidBodyId *ids,
    const RigidConstraintResource *constraints,
    std::uint32_t constraint_capacity,
    std::uint32_t count, std::uint8_t *active_flags) {
    const std::uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= count * count) {
        return;
    }
    const std::uint32_t index = pair / count;
    const std::uint32_t collider_index = pair % count;
    bool active = parameters[index].motion == MotionType::dynamic &&
                  index != collider_index;
    if (active && parameters[collider_index].motion == MotionType::dynamic &&
        collider_index < index) {
        active = false;
    }
    if (active && constrained_collision_disabled(
            ids[index], ids[collider_index], constraints,
            constraint_capacity)) active = false;
    if (active) {
        const float margin = parameters[index].collision_margin +
                             parameters[collider_index].collision_margin;
        const WorldAabb body = world_bounds[index];
        const WorldAabb collider = world_bounds[collider_index];
        active = bounds_overlap(
            {body.minimum.x - margin, body.minimum.y - margin,
             body.minimum.z - margin},
            {body.maximum.x + margin, body.maximum.y + margin,
             body.maximum.z + margin},
            collider.minimum, collider.maximum);
        if (active) {
            active = bounding_spheres_may_contact(
                previous_states[index], states[index],
                meshes[parameters[index].mesh.index],
                previous_states[collider_index], states[collider_index],
                meshes[parameters[collider_index].mesh.index], margin);
        }
    }
    active_flags[pair] = active ? 1U : 0U;
}

__global__ void generate_rigid_leaf_pairs_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states, std::uint32_t count,
    const TriangleMeshResource *meshes,
    std::uint32_t mesh_capacity, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, LeafPair *leaf_pairs,
    std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity) {
    for (std::uint32_t active_index = blockIdx.x;
         active_index < *active_pair_count; active_index += gridDim.x) {
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        if (threadIdx.x == 0U) {
            leaf_pair_counts[pair] = 0U;
        }
        __syncthreads();
        if (active_index >= leaf_pair_cache_slot_capacity) {
            if (threadIdx.x == 0U) {
                leaf_pair_counts[pair] = k_leaf_pair_overflow;
            }
            __syncthreads();
            continue;
        }
        const TriangleMeshId body_mesh_id = parameters[index].mesh;
        const TriangleMeshId collider_mesh_id =
            parameters[collider_index].mesh;
        if (body_mesh_id.index >= mesh_capacity ||
            collider_mesh_id.index >= mesh_capacity) {
            continue;
        }
        const TriangleMeshResource &body_mesh = meshes[body_mesh_id.index];
        const TriangleMeshResource &collider_mesh =
            meshes[collider_mesh_id.index];
        if (!body_mesh.alive ||
            body_mesh.generation != body_mesh_id.generation ||
            !collider_mesh.alive ||
            collider_mesh.generation != collider_mesh_id.generation) {
            continue;
        }

        const float margin = parameters[index].collision_margin +
                             parameters[collider_index].collision_margin;
        const BoundsTransform body_transform = bounds_transform(states[index]);
        const BoundsTransform collider_transform =
            bounds_transform(states[collider_index]);
        const bool swept = requires_swept_pair_contact(
            previous_states[index], states[index], body_mesh,
            previous_states[collider_index], states[collider_index],
            collider_mesh, k_rigid_surface_tolerance);
        BoundsTransform previous_body_transform{};
        BoundsTransform previous_collider_transform{};
        if (swept) {
            previous_body_transform = bounds_transform(previous_states[index]);
            previous_collider_transform =
                bounds_transform(previous_states[collider_index]);
        }
        Vec3 body_minimum{};
        Vec3 body_maximum{};
        Vec3 collider_minimum{};
        Vec3 collider_maximum{};
        const std::uint64_t leaf_pair_count =
            static_cast<std::uint64_t>(body_mesh.bvh_leaf_count) *
            collider_mesh.bvh_leaf_count;
        std::uint32_t local_count = 0U;
        for (std::uint64_t leaf_pair = threadIdx.x;
             leaf_pair < leaf_pair_count; leaf_pair += blockDim.x) {
            const std::uint32_t body_leaf = static_cast<std::uint32_t>(
                leaf_pair / collider_mesh.bvh_leaf_count);
            const std::uint32_t collider_leaf = static_cast<std::uint32_t>(
                leaf_pair % collider_mesh.bvh_leaf_count);
            const BvhNode &body_node =
                body_mesh.bvh_nodes[body_mesh.bvh_leaves[body_leaf]];
            const BvhNode &collider_node = collider_mesh.bvh_nodes[
                collider_mesh.bvh_leaves[collider_leaf]];
            transformed_motion_bounds(
                body_node.minimum, body_node.maximum, previous_body_transform,
                body_transform, swept, margin, body_minimum,
                body_maximum);
            transformed_motion_bounds(
                collider_node.minimum, collider_node.maximum,
                previous_collider_transform, collider_transform,
                swept, 0.0F, collider_minimum, collider_maximum);
            if (bounds_overlap(body_minimum, body_maximum, collider_minimum,
                               collider_maximum)) {
                ++local_count;
            }
        }

        __shared__ std::uint32_t offsets[128];
        __shared__ std::uint32_t candidate_count;
        offsets[threadIdx.x] = local_count;
        __syncthreads();
        if (threadIdx.x == 0U) {
            std::uint32_t prefix = 0U;
            for (std::uint32_t thread = 0U; thread < blockDim.x; ++thread) {
                const std::uint32_t count_for_thread = offsets[thread];
                offsets[thread] = prefix;
                prefix += count_for_thread;
            }
            candidate_count = prefix;
            leaf_pair_counts[pair] =
                prefix > k_max_leaf_pairs_per_body_pair ? k_leaf_pair_overflow
                                                        : prefix;
        }
        __syncthreads();
        if (candidate_count > k_max_leaf_pairs_per_body_pair) {
            continue;
        }

        LeafPair *pair_candidates =
            leaf_pairs + static_cast<std::size_t>(active_index) *
                             k_max_leaf_pairs_per_body_pair;
        std::uint32_t output_index = offsets[threadIdx.x];
        for (std::uint64_t leaf_pair = threadIdx.x;
             leaf_pair < leaf_pair_count; leaf_pair += blockDim.x) {
            const std::uint32_t body_leaf = static_cast<std::uint32_t>(
                leaf_pair / collider_mesh.bvh_leaf_count);
            const std::uint32_t collider_leaf = static_cast<std::uint32_t>(
                leaf_pair % collider_mesh.bvh_leaf_count);
            const BvhNode &body_node =
                body_mesh.bvh_nodes[body_mesh.bvh_leaves[body_leaf]];
            const BvhNode &collider_node = collider_mesh.bvh_nodes[
                collider_mesh.bvh_leaves[collider_leaf]];
            transformed_motion_bounds(
                body_node.minimum, body_node.maximum, previous_body_transform,
                body_transform, swept, margin, body_minimum,
                body_maximum);
            transformed_motion_bounds(
                collider_node.minimum, collider_node.maximum,
                previous_collider_transform, collider_transform,
                swept, 0.0F, collider_minimum, collider_maximum);
            if (!bounds_overlap(body_minimum, body_maximum, collider_minimum,
                                collider_maximum)) {
                continue;
            }
            pair_candidates[output_index++] = {
                body_node.first_triangle, body_node.triangle_count,
                collider_node.first_triangle, collider_node.triangle_count};
        }
        __syncthreads();
    }
}

__global__ void evaluate_rigid_leaf_pairs_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    RigidBodyState *states, std::uint32_t count,
    const TriangleMeshResource *meshes, const RigidBodyId *ids,
    const RigidConstraintResource *constraints,
    std::uint32_t constraint_capacity, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const LeafPair *leaf_pairs, const std::uint32_t *leaf_pair_counts,
    float timestep, ContactManifold *manifolds) {
    for (std::uint32_t active_index = blockIdx.x;
         active_index < *active_pair_count; active_index += gridDim.x) {
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        ContactManifold &output = manifolds[active_index];
        const std::uint32_t candidate_count = leaf_pair_counts[pair];
        if (candidate_count == 0U) {
            if (threadIdx.x == 0U) {
                output = {};
            }
            __syncthreads();
            continue;
        }
        const TriangleMeshResource &body_mesh =
            meshes[parameters[index].mesh.index];
        const TriangleMeshResource &collider_mesh =
            meshes[parameters[collider_index].mesh.index];
        if (candidate_count == k_leaf_pair_overflow) {
            if (threadIdx.x == 0U) {
                output = {};
            }
            __syncthreads();
            continue;
        }

        extern __shared__ ContactManifold partials[];
        __shared__ ContactManifold reduced;
        if (threadIdx.x == 0U) {
            reduced = {};
        }
        __syncthreads();
        const LeafPair *pair_candidates =
            leaf_pairs + static_cast<std::size_t>(active_index) *
                             k_max_leaf_pairs_per_body_pair;
        const float separation = fmaxf(
            (parameters[index].collision_margin +
             parameters[collider_index].collision_margin) *
                2.0F,
            1.0e-4F);
        const float collision_margin =
            parameters[index].collision_margin +
            parameters[collider_index].collision_margin;
        const bool swept = requires_swept_pair_contact(
            previous_states[index], states[index], body_mesh,
            previous_states[collider_index], states[collider_index],
            collider_mesh, collision_margin);
        Vec3 previous_body_reference{};
        Vec3 body_reference{};
        rigid_hinge_contact_frame(
            parameters, ids, count, index, previous_states,
            previous_states[index], constraints, constraint_capacity,
            previous_body_reference);
        const HingeContactFrame body_hinge = rigid_hinge_contact_frame(
            parameters, ids, count, index, states, states[index], constraints,
            constraint_capacity, body_reference);
        Vec3 collider_reference{};
        const HingeContactFrame collider_hinge = rigid_hinge_contact_frame(
            parameters, ids, count, collider_index, states,
            states[collider_index], constraints, constraint_capacity,
            collider_reference);
        for (std::uint32_t wave = 0U; wave < candidate_count;
             wave += blockDim.x) {
            ContactManifold local{};
            const std::uint32_t candidate_index = wave + threadIdx.x;
            if (candidate_index < candidate_count) {
                const LeafPair candidate = pair_candidates[candidate_index];
                collide_triangle_ranges(
                    states[index], body_reference, body_hinge,
                    body_mesh,
                    candidate.body_first,
                    candidate.body_count, states[collider_index],
                    collider_mesh, candidate.collider_first,
                    candidate.collider_count, collider_hinge,
                    collision_margin, timestep, local);
                if (swept) {
                    collide_triangle_ranges_swept(
                        previous_states[index], states[index],
                        previous_body_reference, body_reference,
                        body_hinge, body_mesh,
                        candidate.body_first, candidate.body_count,
                        previous_states[collider_index],
                        states[collider_index], collider_mesh,
                        candidate.collider_first, candidate.collider_count,
                        collision_margin, timestep, true, true,
                        collider_hinge, local);
                }
            }
            partials[threadIdx.x] = local;
            __syncthreads();
            if (threadIdx.x == 0U) {
                const std::uint32_t remaining = candidate_count - wave;
                const std::uint32_t wave_count =
                    blockDim.x < remaining ? blockDim.x : remaining;
                for (std::uint32_t item = 0U; item < wave_count; ++item) {
                    for (std::uint32_t contact = 0U;
                         contact < partials[item].count; ++contact) {
                        add_manifold_contact(reduced,
                                             partials[item].contacts[contact],
                                             separation);
                    }
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0U) {
            output = reduced;
        }
        __syncthreads();
    }
}

// Serial BVH traversal needs a large stack. Isolate it from normal pair work.
__global__ void evaluate_overflow_rigid_pairs_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states, std::uint32_t count,
    const TriangleMeshResource *meshes, const RigidBodyId *ids,
    const RigidConstraintResource *constraints,
    std::uint32_t constraint_capacity, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint32_t *leaf_pair_counts, float timestep,
    ContactManifold *manifolds) {
    for (std::uint32_t active_index = blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        const std::uint32_t pair = active_pairs[active_index];
        if (leaf_pair_counts[pair] != k_leaf_pair_overflow) {
            continue;
        }
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        Vec3 previous_body_reference{};
        Vec3 body_reference{};
        rigid_hinge_contact_frame(
            parameters, ids, count, index, previous_states,
            previous_states[index], constraints, constraint_capacity,
            previous_body_reference);
        const HingeContactFrame body_hinge = rigid_hinge_contact_frame(
            parameters, ids, count, index, states, states[index], constraints,
            constraint_capacity, body_reference);
        Vec3 collider_reference{};
        const HingeContactFrame collider_hinge = rigid_hinge_contact_frame(
            parameters, ids, count, collider_index, states,
            states[collider_index], constraints, constraint_capacity,
            collider_reference);
        manifolds[active_index] = collide_meshes(
            parameters[index], previous_states[index], states[index],
            previous_body_reference, body_reference, body_hinge,
            meshes[parameters[index].mesh.index], parameters[collider_index],
            previous_states[collider_index], states[collider_index],
            meshes[parameters[collider_index].mesh.index], collider_hinge,
            timestep);
    }
}

__global__ void prepare_parallel_contact_events_kernel(
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const RigidBodyId *ids,
    std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    if (reset_events) {
        *event_count = 0U;
    }
    if (!collect_events) {
        return;
    }
    std::uint32_t cursor = 0U;
    for (std::uint32_t active_index = 0U;
         active_index < *active_pair_count; ++active_index) {
        const ContactManifold &manifold = manifolds[active_index];
        event_offsets[active_index] = cursor;
        const std::uint32_t remaining = cursor < event_capacity
            ? event_capacity - cursor : 0U;
        const std::uint32_t retained = manifold.count < remaining
            ? manifold.count : remaining;
        if (retained > 0U) {
            const std::uint32_t pair = active_pairs[active_index];
            const std::uint32_t body_index = pair / count;
            const std::uint32_t collider_index = pair % count;
            for (std::uint32_t contact_index = 0U;
                 contact_index < retained; ++contact_index) {
                const Contact &contact = manifold.contacts[contact_index];
                events[cursor + contact_index] = {
                    ids[body_index], ids[collider_index], contact.point,
                    contact.normal, fmaxf(0.0F, contact.penetration),
                    0.0F, {}};
            }
        }
        cursor += manifold.count;
    }
    // A later substep with no contacts must not erase an earlier event from
    // this frame; the original serial path retained it as well.
    if (cursor > 0U) {
        *event_count = cursor < event_capacity ? cursor : event_capacity;
    }
}

__global__ void initialize_parallel_colors_kernel(
    const std::uint32_t *active_pair_count, std::uint8_t *pair_colors,
    std::uint32_t *color_state) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        color_state[0] = 0U;
        color_state[1] = 0U;
    }
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        pair_colors[active_index] = k_contact_color_overflow;
    }
}

__global__ void reset_parallel_color_owners_kernel(
    std::uint32_t *owners, std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        owners[index] = 0xffffffffU;
    }
}

// Each body's lowest-priority uncolored contact wins this color round.
__host__ __device__ std::uint32_t contact_color_priority(
    std::uint32_t pair) noexcept {
    // Odd multiplication permutes uint32 values; nearby body IDs do not
    // monopolize all rounds, and priorities remain deterministic and unique.
    return pair * 2654435761U + 1013904223U;
}

__global__ void find_parallel_color_owners_kernel(
    const BodyParameters *parameters, std::uint32_t count,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const std::uint8_t *pair_colors,
    std::uint32_t *owners) {
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        if (pair_colors[active_index] != k_contact_color_overflow ||
            manifolds[active_index].count == 0U) {
            continue;
        }
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        const std::uint32_t priority = contact_color_priority(pair);
        atomicMin(&owners[index], priority);
        if (parameters[collider_index].motion == MotionType::dynamic) {
            atomicMin(&owners[collider_index], priority);
        }
    }
}

__global__ void assign_parallel_contact_colors_kernel(
    const BodyParameters *parameters, std::uint32_t count,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const std::uint32_t *owners,
    std::uint8_t *pair_colors, std::uint32_t *color_state,
    std::uint32_t color, std::uint32_t color_round_count) {
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        if (pair_colors[active_index] != k_contact_color_overflow ||
            manifolds[active_index].count == 0U) {
            continue;
        }
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        const bool dynamic_collider =
            parameters[collider_index].motion == MotionType::dynamic;
        const std::uint32_t priority = contact_color_priority(pair);
        if (owners[index] == priority &&
            (!dynamic_collider || owners[collider_index] == priority)) {
            pair_colors[active_index] = static_cast<std::uint8_t>(color);
            atomicMax(&color_state[0], color + 1U);
        } else if (color + 1U == color_round_count) {
            atomicAdd(&color_state[1], 1U);
        }
    }
}

__global__ void color_small_rigid_contacts_kernel(
    const BodyParameters *parameters, std::uint32_t count,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, std::uint8_t *pair_colors,
    std::uint32_t *owners, std::uint32_t *color_state,
    std::uint32_t color_round_count) {
    if (blockIdx.x != 0U) return;
    const std::uint32_t active_count = *active_pair_count;
    for (std::uint32_t color = 0U; color < color_round_count; ++color) {
        for (std::uint32_t body = threadIdx.x; body < count;
             body += blockDim.x)
            owners[body] = UINT32_MAX;
        __syncthreads();
        for (std::uint32_t active_index = threadIdx.x;
             active_index < active_count; active_index += blockDim.x) {
            if (pair_colors[active_index] != k_contact_color_overflow ||
                manifolds[active_index].count == 0U) continue;
            const std::uint32_t pair = active_pairs[active_index];
            const std::uint32_t first = pair / count;
            const std::uint32_t second = pair % count;
            const std::uint32_t priority = contact_color_priority(pair);
            atomicMin(&owners[first], priority);
            if (parameters[second].motion == MotionType::dynamic)
                atomicMin(&owners[second], priority);
        }
        __syncthreads();
        for (std::uint32_t active_index = threadIdx.x;
             active_index < active_count; active_index += blockDim.x) {
            if (pair_colors[active_index] != k_contact_color_overflow ||
                manifolds[active_index].count == 0U) continue;
            const std::uint32_t pair = active_pairs[active_index];
            const std::uint32_t first = pair / count;
            const std::uint32_t second = pair % count;
            const bool dynamic_second =
                parameters[second].motion == MotionType::dynamic;
            const std::uint32_t priority = contact_color_priority(pair);
            if (owners[first] == priority &&
                (!dynamic_second || owners[second] == priority)) {
                pair_colors[active_index] = static_cast<std::uint8_t>(color);
                atomicMax(&color_state[0], color + 1U);
            } else if (color + 1U == color_round_count) {
                atomicAdd(&color_state[1], 1U);
            }
        }
        __syncthreads();
    }
}

__device__ void resolve_active_rigid_contact_pair(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, std::uint32_t active_index,
    float timestep, bool correct_position) {
    const std::uint32_t pair = active_pairs[active_index];
    const std::uint32_t index = pair / count;
    const std::uint32_t collider_index = pair % count;
    const ContactManifold &manifold = manifolds[active_index];
    RigidContactEvent *pair_events = nullptr;
    std::uint32_t retained = 0U;
    if (event_capacity > 0U) {
        const std::uint32_t offset = event_offsets[active_index];
        if (offset < event_capacity) {
            pair_events = events + offset;
            const std::uint32_t remaining = event_capacity - offset;
            retained = manifold.count < remaining
                ? manifold.count : remaining;
        }
    }
    resolve_contacts(parameters[index], states[index],
                     parameters[collider_index], states[collider_index],
                     manifold.contacts, manifold.count, timestep,
                     correct_position,
                     pair_events, retained);
}

__global__ void resolve_colored_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint8_t *pair_colors, const std::uint32_t *color_state,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity,
    std::uint32_t color, float timestep, bool correct_position) {
    if (color >= color_state[0]) {
        return;
    }
    for (std::uint32_t active_index =
             blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        if (pair_colors[active_index] != color) {
            continue;
        }
        resolve_active_rigid_contact_pair(parameters, states, count,
            manifolds, active_pairs, event_offsets, events, event_capacity,
            active_index, timestep, correct_position);
    }
}

__global__ void resolve_uncolored_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint8_t *pair_colors, const std::uint32_t *color_state,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity,
    float timestep, bool correct_position) {
    if (blockIdx.x != 0U || threadIdx.x != 0U || color_state[1] == 0U) {
        return;
    }
    for (std::uint32_t active_index = 0U;
         active_index < *active_pair_count; ++active_index) {
        if (pair_colors[active_index] != k_contact_color_overflow ||
            manifolds[active_index].count == 0U) {
            continue;
        }
        resolve_active_rigid_contact_pair(parameters, states, count,
            manifolds, active_pairs, event_offsets, events, event_capacity,
            active_index, timestep, correct_position);
    }
}

// One block can synchronize between colors without a kernel launch per round.
// Pair colors are body-disjoint, so contacts within a color remain parallel.
__global__ void resolve_small_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint8_t *pair_colors, const std::uint32_t *color_state,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, float timestep) {
    if (blockIdx.x != 0U) return;
    const std::uint32_t active_count = *active_pair_count;
    const std::uint32_t used_colors = color_state[0];
    for (std::uint32_t pass = 0U; pass < 8U; ++pass) {
        for (std::uint32_t color = 0U; color < used_colors; ++color) {
            for (std::uint32_t active_index = threadIdx.x;
                 active_index < active_count; active_index += blockDim.x) {
                if (pair_colors[active_index] != color) continue;
                resolve_active_rigid_contact_pair(parameters, states, count,
                    manifolds, active_pairs, event_offsets, events,
                    event_capacity, active_index, timestep, pass == 0U);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0U && color_state[1] != 0U) {
            for (std::uint32_t active_index = 0U;
                 active_index < active_count; ++active_index) {
                if (pair_colors[active_index] != k_contact_color_overflow ||
                    manifolds[active_index].count == 0U) continue;
                resolve_active_rigid_contact_pair(parameters, states, count,
                    manifolds, active_pairs, event_offsets, events,
                    event_capacity, active_index, timestep, pass == 0U);
            }
        }
        __syncthreads();
    }
}

__global__ void clamp_rigid_speeds_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || parameters[index].motion != MotionType::dynamic) {
        return;
    }
    states[index].linear_velocity = clamp_length(
        states[index].linear_velocity, parameters[index].maximum_linear_speed);
    states[index].angular_velocity = clamp_length(
        states[index].angular_velocity, parameters[index].maximum_angular_speed);
}

__global__ void clear_rigid_inputs_kernel(BodyAccumulator *accumulators,
                                          KinematicTarget *targets,
                                          std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    accumulators[index] = {};
    targets[index].active = false;
}

__global__ void capture_rigid_inputs_kernel(
    const BodyAccumulator *accumulators, Vec3 *forces, Vec3 *torques,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    forces[index] = accumulators[index].force;
    torques[index] = accumulators[index].torque;
}
