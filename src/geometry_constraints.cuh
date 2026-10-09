// SPDX-License-Identifier: MIT
// Rigid integration, collision geometry, and AVBD resource internals.

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

struct RigidConstraintGeometry {
    std::uint32_t dense_a{};
    std::uint32_t dense_b{};
};

struct RigidConstraintResource {
    RigidConstraintOptions options{};
    RigidConstraintState state{};
    std::uint32_t generation{1U};
    bool alive{};
    // Dense endpoints are refreshed each substep. AVBD evaluates moving joint
    // anchors and tangent Jacobians at the current pose, not a frozen geometry.
    RigidConstraintGeometry geometry{};
    avbd::Row avbd_rows[18]{};
    std::uint32_t avbd_count{}, avbd_next_a{}, avbd_next_b{};
};

// Collision filtering only; fixed groups do not merge mass or project poses.
struct FixedContactProjection {
    std::uint32_t root{};
};

struct HingeContactFrame {
    Vec3 anchor{};
    Vec3 axis{};
    Vec3 local_anchor{};
    float inverse_mass{};
    float inverse_moment{};
    bool present{};
    bool fixed{};
    // A static piston/slider leaves axial translation free. Only pistons
    // additionally allow rotation about that axis.
    bool axial{};
    bool axial_rotation{};
    bool fixed_member{};
    bool point_member{};
    bool static_body{};
};

struct Contact {
    Vec3 normal{};
    Vec3 point{};
    // Positive inside solver rest offset; negative while speculative.
    float penetration{};
    HingeContactFrame body_hinge{};
    HingeContactFrame collider_hinge{};
    float accumulated_normal_impulse{};
    Vec3 accumulated_friction_impulse{};
    float initial_normal_speed{};
    bool warm_started{};
    bool persistent{};
    float impact_fraction{1.0F};
    // Final AVBD forces converted to impulses for contact-event publication.
    float reported_normal_impulse{};
    Vec3 reported_friction_impulse{};
    avbd::Dual avbd_dual[3]{};
    Vec3 avbd_error{};
    Vec3 avbd_anchor_a{}, avbd_anchor_b{};
    bool avbd_cached{};
};

struct ContactManifold {
    Contact contacts[8]{};
    std::uint32_t count{};
    Vec3 initial_relative_position{};
    bool face_patch{};
    std::uint32_t response_slot{0xffffffffU};
    std::uint32_t avbd_next_a{}, avbd_next_b{};
};

__device__ bool guided_static_pair(const HingeContactFrame &body,
                                  const HingeContactFrame &collider) noexcept {
    return (body.axial && collider.static_body) ||
           (collider.axial && body.static_body);
}

__device__ bool guided_static_contact(const Contact &contact) noexcept {
    return guided_static_pair(contact.body_hinge, contact.collider_hinge);
}

struct CachedContact {
    Vec3 local_point{};
    Vec3 normal{};
    float normal_impulse{};
    Vec3 friction_impulse{};
    avbd::Dual avbd_dual[3]{};
    Vec3 avbd_anchor_a{}, avbd_anchor_b{};
};

struct CachedContactPair {
    std::uint32_t pair{};
    RigidBodyId body{}, collider{};
    std::uint64_t epoch{};
    float timestep{};
    std::uint32_t count{};
    CachedContact contacts[8]{};
};

struct ContactSchedule {
    // Include the overflow lane (28 also covers all pairs in tiny worlds).
    std::uint32_t counts[29]{}, offsets[29]{};
    std::uint32_t budget{}, remaining{};
    std::uint32_t island_count{}, early_exit_count{}, maximum_passes{}, blocks{};
    std::uint32_t candidates{}, contacts{};
};

constexpr float k_rigid_surface_tolerance = 1.0e-5F;
constexpr float k_rigid_rest_offset = 1.0e-3F;
constexpr float k_rigid_guided_rest_offset = 1.0e-4F;

__host__ __device__ float rigid_rest_offset(float margin) noexcept {
    return fminf(margin, k_rigid_rest_offset);
}

struct LeafPair {
    std::uint32_t body_first{};
    std::uint32_t body_count{};
    std::uint32_t collider_first{};
    std::uint32_t collider_count{};
};

// Iterate the same strided Cartesian product without a 64-bit division and
// remainder for every candidate. Quotients are computed once per thread.
struct RigidLeafPairCursor {
    std::uint32_t body_leaf{};
    std::uint32_t collider_leaf{};
    std::uint32_t body_stride{};
    std::uint32_t collider_stride{};
    std::uint32_t collider_count{};

    __device__ explicit RigidLeafPairCursor(std::uint32_t count) noexcept
        : body_leaf(threadIdx.x / count), collider_leaf(threadIdx.x % count),
          body_stride(blockDim.x / count), collider_stride(blockDim.x % count),
          collider_count(count) {}

    __device__ void advance() noexcept {
        body_leaf += body_stride;
        collider_leaf += collider_stride;
        if (collider_leaf >= collider_count) {
            collider_leaf -= collider_count;
            ++body_leaf;
        }
    }
};

constexpr std::uint32_t k_max_leaf_pairs_per_body_pair = 512U;
// Few-body scenes need parallelism within a dense mesh pair. Bound the extra
// scratch by world capacity; many-body worlds retain their lean pair cache.
constexpr std::uint32_t k_small_rigid_leaf_body_capacity = 8U;
constexpr std::uint32_t k_small_rigid_leaf_pair_capacity = 4096U;
constexpr std::uint32_t k_shared_rigid_leaf_capacity = 1024U;
constexpr std::uint32_t k_rigid_leaf_blocks_per_pair = 16U;
// Keep the fast leaf-pair cache proportional to body capacity. Dense worlds
// retain exact contacts through the serial fallback instead of reserving one
// 512-entry cache for every possible body pair.
constexpr std::uint32_t k_leaf_pair_cache_slots_per_body = 8U;
constexpr std::uint32_t k_minimum_leaf_pair_cache_slots = 4'096U;
constexpr std::uint32_t k_leaf_pair_overflow =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t k_leaf_pair_face_patch = k_leaf_pair_overflow - 1U;
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
    // The first index_count/3 entries retain triangle order. A deduplicated
    // tail is used only by paired soft-surface support-plane searches.
    CollisionPlane *solid_planes{};
    std::uint32_t solid_unique_plane_count{};
    // Outward normals for closed convex connected shells, including mirrored
    // pieces in a concave compound. Other triangles have a zero normal.
    Vec3 *shell_normals{};
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
                                             Vec3 c, bool stable_face = false) noexcept {
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
    // The region tests established an interior face projection. Rebuilding
    // it from barycentric coordinates loses tangential precision on long,
    // thin triangles; at micron-scale gaps that error becomes a false contact
    // normal. Orthogonal projection retains the input's tangential position.
    if (stable_face) {
        const Vec3 normal = cross(ab, ac);
        return subtract(point, multiply(normal, dot(ap, normal) / length_squared(normal)));
    }
    const float inverse = 1.0F / (va + vb + vc);
    return add(a, add(multiply(ab, vb * inverse), multiply(ac, vc * inverse)));
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
    Vec3 &point_a, Vec3 &point_b, bool stable_face = false) noexcept {
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
            closest_on_triangle(a[vertex], b0, b1, b2, stable_face);
        consider_closest_pair(a[vertex], on_triangle, best_squared,
                              point_a, point_b);
    }
    for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
        const Vec3 on_triangle =
            closest_on_triangle(b[vertex], a0, a1, a2, stable_face);
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
    const bool guided = guided_static_contact(candidate);
    const auto guide_direction = [](const Contact &contact) {
        const auto &frame = contact.body_hinge.axial
            ? contact.body_hinge : contact.collider_hinge;
        return Vec3{dot(contact.normal, frame.axis) * sqrtf(frame.inverse_mass),
            frame.axial_rotation ? dot(frame.axis,
                cross(subtract(contact.point, frame.anchor), contact.normal)) *
                    sqrtf(frame.inverse_moment) : 0.0F,
            0.0F};
    };
    const Vec3 candidate_direction = guided ? guide_direction(candidate) : Vec3{};
    const float candidate_length = guided ? vector_length(candidate_direction) : 1.0F;
    if (guided && candidate_length * candidate_length <= k_epsilon) return;
    if (guided && candidate.impact_fraction > 0.0F) {
        // Later triangle intersections can lie inside an obstacle already
        // hit earlier in the sweep. Keep existing support and the first new
        // impact, not incompatible normals from beyond that impact.
        float first_impact = 1.0F;
        for (std::uint32_t row = 0; row < manifold.count; ++row)
            if (manifold.contacts[row].impact_fraction > 0.0F)
                first_impact = fminf(first_impact, manifold.contacts[row].impact_fraction);
        constexpr float simultaneous = 1.0e-4F;
        if (candidate.impact_fraction > first_impact + simultaneous) return;
        if (candidate.impact_fraction < first_impact - simultaneous) {
            std::uint32_t kept = 0;
            for (std::uint32_t row = 0; row < manifold.count; ++row)
                if (manifold.contacts[row].impact_fraction == 0.0F)
                    manifold.contacts[kept++] = manifold.contacts[row];
            manifold.count = kept;
        }
    }
    const float minimum_spacing_squared = point_spacing * point_spacing;
    for (std::uint32_t index = 0; index < manifold.count; ++index) {
        if (guided) {
            // Repeated features must not fill every row with the same
            // axial stop and discard the face that stops rotation. Reduce in
            // the guide's actual translation/twist Jacobian, not world space.
            const Vec3 direction = guide_direction(manifold.contacts[index]);
            const float size = vector_length(direction);
            if (dot(candidate_direction, direction) >
                0.9999F * candidate_length * size) {
                if (candidate.penetration / candidate_length >
                    manifold.contacts[index].penetration / size)
                    manifold.contacts[index] = candidate;
                return;
            }
            continue;
        }
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

// Refine broad face contacts on small closed convex meshes. The bounded face
// count keeps clipping storage and per-pair work fixed; curved/concave meshes
// and edge impacts retain the triangle/BVH path.
__device__ bool convex_face_manifold(
    const RigidBodyState &body_state, const TriangleMeshResource &body_mesh,
    const HingeContactFrame &body_hinge,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh,
    const HingeContactFrame &collider_hinge, float margin,
    ContactManifold &output) noexcept {
    constexpr std::uint32_t maximum_faces = 32U;
    if (!body_mesh.solid_planes || !collider_mesh.solid_planes ||
        body_mesh.index_count > maximum_faces * 3U ||
        collider_mesh.index_count > maximum_faces * 3U) return false;

    float best_separation = -FLT_MAX;
    std::uint32_t reference_face = 0U;
    bool reference_is_body = false;
    for (std::uint32_t side = 0U; side < 2U; ++side) {
        const auto &reference_mesh = side == 0U ? collider_mesh : body_mesh;
        const auto &reference_state = side == 0U ? collider_state : body_state;
        const auto &incident_mesh = side == 0U ? body_mesh : collider_mesh;
        const auto &incident_state = side == 0U ? body_state : collider_state;
        for (std::uint32_t face = 0U; face < reference_mesh.index_count / 3U; ++face) {
            const CollisionPlane plane = reference_mesh.solid_planes[face];
            const Vec3 normal = rotate(reference_state.orientation, plane.normal);
            const Vec3 incident_normal = inverse_rotate(incident_state.orientation, normal);
            float support = FLT_MAX;
            for (std::uint32_t vertex = 0U; vertex < incident_mesh.vertex_count; ++vertex)
                support = fminf(support, dot(incident_normal, incident_mesh.vertices[vertex]));
            const float separation = support + dot(normal,
                subtract(incident_state.position, reference_state.position)) - plane.offset;
            if (separation > best_separation) {
                best_separation = separation;
                reference_face = face;
                reference_is_body = side != 0U;
            }
        }
    }
    if (best_separation > margin) {
        output = {};
        return true;
    }
    const auto &reference_mesh = reference_is_body ? body_mesh : collider_mesh;
    const auto &reference_state = reference_is_body ? body_state : collider_state;
    const auto &incident_mesh = reference_is_body ? collider_mesh : body_mesh;
    const auto &incident_state = reference_is_body ? collider_state : body_state;
    const CollisionPlane reference = reference_mesh.solid_planes[reference_face];
    const Vec3 outward = rotate(reference_state.orientation, reference.normal);
    const Vec3 incident_axis = inverse_rotate(incident_state.orientation, outward);
    float alignment = 1.0F;
    std::uint32_t incident_face = 0U;
    for (std::uint32_t face = 0U; face < incident_mesh.index_count / 3U; ++face) {
        const float value = dot(incident_axis, incident_mesh.solid_planes[face].normal);
        if (value < alignment) {
            alignment = value;
            incident_face = face;
        }
    }
    if (alignment > -0.98F) return false;

    ContactManifold manifold{};
    const Vec3 normal = multiply(outward, reference_is_body ? -1.0F : 1.0F);
    const CollisionPlane incident = incident_mesh.solid_planes[incident_face];
    for (std::uint32_t triangle = 0U; triangle < incident_mesh.index_count / 3U; ++triangle) {
        const CollisionPlane face = incident_mesh.solid_planes[triangle];
        if (dot(face.normal, incident.normal) < 0.99999F ||
            fabsf(face.offset - incident.offset) > k_rigid_surface_tolerance) continue;
        Vec3 polygon[maximum_faces + 4U]{};
        Vec3 clipped[maximum_faces + 4U]{};
        std::uint32_t count = 3U;
        for (std::uint32_t corner = 0U; corner < 3U; ++corner)
            polygon[corner] = inverse_rotate(reference_state.orientation,
                subtract(transform_point(incident_state,
                    incident_mesh.vertices[incident_mesh.indices[triangle * 3U + corner]]),
                    reference_state.position));
        // Clip the incident triangle against the reference solid's side faces.
        // Only the supporting face is expanded for speculative contacts.
        for (std::uint32_t plane_index = 0U;
             plane_index < reference_mesh.index_count / 3U && count != 0U; ++plane_index) {
            const CollisionPlane plane = reference_mesh.solid_planes[plane_index];
            const float offset = plane.offset +
                (dot(plane.normal, reference.normal) > 0.99999F ? margin : 0.0F);
            std::uint32_t clipped_count = 0U;
            Vec3 previous = polygon[count - 1U];
            float previous_distance = dot(plane.normal, previous) - offset;
            for (std::uint32_t vertex = 0U; vertex < count; ++vertex) {
                const Vec3 current = polygon[vertex];
                const float distance = dot(plane.normal, current) - offset;
                if ((distance <= 0.0F) != (previous_distance <= 0.0F))
                    clipped[clipped_count++] = add(previous,
                        multiply(subtract(current, previous),
                                 previous_distance / (previous_distance - distance)));
                if (distance <= 0.0F) clipped[clipped_count++] = current;
                previous = current;
                previous_distance = distance;
            }
            count = clipped_count;
            for (std::uint32_t vertex = 0U; vertex < count; ++vertex)
                polygon[vertex] = clipped[vertex];
        }
        for (std::uint32_t vertex = 0U; vertex < count; ++vertex) {
            const float distance = dot(reference.normal, polygon[vertex]) - reference.offset;
            const Vec3 point = transform_point(reference_state,
                subtract(polygon[vertex], multiply(reference.normal, distance * 0.5F)));
            // Solid face normals remain defined at zero distance, so these
            // contacts need no artificial gap between authored coplanar faces.
            // Another contact in the stack can change closing velocity during
            // this solve. Keep the whole speculative band, even when both
            // bodies initially fall at the same speed.
            if (distance <= margin)
                add_manifold_contact(manifold,
                    {normal, point, -distance,
                     body_hinge, collider_hinge}, fmaxf(margin * 2.0F, 1.0e-4F));
        }
    }
    // Parallel supporting faces with no overlap are separated, including
    // adjacent corners that only coincide within floating-point roundoff.
    // Falling back to intersecting triangle faces invents penetration there.
    if (manifold.count == 0U && alignment > -0.999999F) return false;
    manifold.face_patch = true;
    output = manifold;
    return true;
}

__device__ Vec3 guided_triangle_normal(
    Vec3 a0, Vec3 a1, Vec3 a2, Vec3 b0, Vec3 b1, Vec3 b2,
    Vec3 point_a, Vec3 point_b, Vec3 fallback) noexcept {
    const Vec3 delta = subtract(point_a, point_b);
    Vec3 normal = normalized_or(delta, fallback);
    // Subtracting world-space points a few microns apart loses precision.
    // On a face use its geometric normal; tiny axial components on a purely
    // radial contact otherwise become large guide-space corrections.
    const Vec3 faces[2] = {
        normalized_or(cross(subtract(a1, a0), subtract(a2, a0)), normal),
        normalized_or(cross(subtract(b1, b0), subtract(b2, b0)), normal)};
    // At a coincident vertex/face hit the tiny point gap cannot classify its
    // normal, and the previous separating plane may still describe an edge.
    // A closest point strictly inside a triangle identifies its face without
    // inventing a face at a genuine corner. Use the same scale-relative edge
    // tolerance as point_in_triangle, with a positive interior margin.
    const Vec3 vertices[2][3] = {{a0, a1, a2}, {b0, b1, b2}};
    const Vec3 points[2] = {point_a, point_b};
    float face_alignment = -1.0F;
    Vec3 interior_normal{};
    for (std::uint32_t index = 0U; index < 2U; ++index) {
        const Vec3 raw_normal = cross(subtract(vertices[index][1], vertices[index][0]),
                                      subtract(vertices[index][2], vertices[index][0]));
        const float tolerance = 1.0e-5F * length_squared(raw_normal);
        bool interior = tolerance > 0.0F;
        for (std::uint32_t edge = 0U; edge < 3U; ++edge) {
            interior = interior && dot(cross(
                subtract(vertices[index][(edge + 1U) % 3U], vertices[index][edge]),
                subtract(points[index], vertices[index][edge])), raw_normal) > tolerance;
        }
        const float alignment = fabsf(dot(faces[index], normal));
        if (interior && alignment > face_alignment) {
            face_alignment = alignment;
            const float projection = dot(delta, faces[index]);
            const float side = fabsf(projection) > k_epsilon
                ? projection : dot(fallback, faces[index]);
            interior_normal = side >= 0.0F ? faces[index] : multiply(faces[index], -1.0F);
        }
    }
    if (face_alignment >= 0.0F) return interior_normal;
    float best_error = 4.0F * k_epsilon * k_epsilon;
    if (length_squared(delta) > k_epsilon * k_epsilon) {
        for (const Vec3 face : faces) {
            const float projection = dot(delta, face);
            const float error = length_squared(subtract(delta, multiply(face, projection)));
            if (error < best_error) {
                best_error = error;
                normal = projection >= 0.0F ? face : multiply(face, -1.0F);
            }
        }
    }
    return normal;
}

__device__ bool triangle_pair_face_contact(
    const Vec3 *a, const Vec3 *b, Vec3 separation) noexcept {
    const Vec3 normal = normalized_or(separation, {});
    const Vec3 face_a = normalized_or(cross(subtract(a[1], a[0]), subtract(a[2], a[0])), {});
    const Vec3 face_b = normalized_or(cross(subtract(b[1], b[0]), subtract(b[2], b[0])), {});
    return fabsf(dot(normal, face_a)) > 0.9999F || fabsf(dot(normal, face_b)) > 0.9999F;
}

__device__ void collide_triangle_ranges(
    const RigidBodyState &previous_body_state,
    const RigidBodyState &body_state, Vec3 body_reference,
    const HingeContactFrame &body_hinge,
    const TriangleMeshResource &body_mesh,
    std::uint32_t body_first, std::uint32_t body_count,
    const RigidBodyState &previous_collider_state,
    const RigidBodyState &collider_state,
    const TriangleMeshResource &collider_mesh, std::uint32_t collider_first,
    std::uint32_t collider_count, const HingeContactFrame &collider_hinge,
    float margin,
    float timestep,
    ContactManifold &manifold) noexcept {
    const bool guided_pair = guided_static_pair(body_hinge, collider_hinge);
    if (guided_pair) {
        // Swept entry contacts are authoritative. A tooth can cross a thin
        // face and end close to its opposite side in the predicted pose.
        // Mixing that endpoint's reversed normal with the entry normal
        // creates incompatible rows and pushes the tooth through its stop.
        // The sweep also handles contacts present at the start of the step.
        return;
    }
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
            const Vec3 plane_projection = subtract(point_a, multiply(
                collider_normal, dot(subtract(point_a, b0), collider_normal)));
            const bool on_triangle_face = length_squared(subtract(
                plane_projection, closest_on_triangle(plane_projection, b0, b1, b2)))
                <= k_rigid_surface_tolerance * k_rigid_surface_tolerance;
            // Large floor triangles lose a few ulps in closest-point
            // subtraction. An interior face contact has the exact face
            // normal, even when its tiny nonzero separation is noisy.
            const bool small_convex = body_mesh.solid_planes != nullptr &&
                                      body_mesh.index_count <= 96U;
            Vec3 normal =
                (distance <= k_rigid_surface_tolerance ||
                 (small_convex && on_triangle_face))
                    ? fallback : normalized_or(delta, fallback);
            // A concave body's origin may lie in its cavity, on the wrong
            // side of the contacting face. For a convex body against such a
            // surface, use that surface's face and the convex body's entry
            // side, independently of which body was allocated first. Using
            // the convex body's own intersecting facets can push it inward.
            bool convex_surface_face = false;
            float face_separation = 0.0F;
            const bool convex_a = body_mesh.solid_planes != nullptr;
            const bool convex_b = collider_mesh.solid_planes != nullptr;
            if (convex_a != convex_b &&
                distance <= k_rigid_surface_tolerance) {
                const Vec3 face0 = convex_a ? b0 : a0;
                const Vec3 face1 = convex_a ? b1 : a1;
                const Vec3 face2 = convex_a ? b2 : a2;
                const auto &surface_state = convex_a ? collider_state : body_state;
                const auto &previous_surface = convex_a ? previous_collider_state : previous_body_state;
                const auto &previous_convex = convex_a ? previous_body_state : previous_collider_state;
                Vec3 outward = normalized_or(cross(subtract(face1, face0), subtract(face2, face0)), {});
                const Vec3 local_normal = inverse_rotate(surface_state.orientation, outward);
                const Vec3 old_normal = rotate(previous_surface.orientation, local_normal);
                const Vec3 old_point = transform_point(previous_surface,
                    inverse_rotate(surface_state.orientation, subtract(face0, surface_state.position)));
                if (dot(old_normal, subtract(previous_convex.position, old_point)) < 0.0F)
                    outward = multiply(outward, -1.0F);
                const Vec3 incident_point = convex_a ? point_a : point_b;
                const Vec3 projection = subtract(incident_point,
                    multiply(outward, dot(subtract(incident_point, face0), outward)));
                convex_surface_face = length_squared(outward) > 0.5F &&
                    length_squared(subtract(projection,
                        closest_on_triangle(projection, face0, face1, face2))) <=
                    k_rigid_surface_tolerance * k_rigid_surface_tolerance;
                if (convex_surface_face) {
                    normal = multiply(outward, convex_a ? 1.0F : -1.0F);
                    face_separation = fminf(dot(subtract(convex_a ? a0 : b0, face0), outward),
                        fminf(dot(subtract(convex_a ? a1 : b1, face0), outward),
                              dot(subtract(convex_a ? a2 : b2, face0), outward)));
                }
            }
            if (!contact_reaches_rest_offset(
                    body_state, collider_state, point, normal, distance,
                    rest_offset, timestep)) {
                continue;
            }
            float penetration = rest_offset - distance +
                                k_rigid_surface_tolerance;
            if (convex_surface_face) {
                // Measure actual signed depth, not one complete search
                // margin on every substep of a resting intersection.
                penetration = fminf(margin, rest_offset - face_separation) +
                              k_rigid_surface_tolerance;
            } else if (distance <= k_rigid_surface_tolerance) {
                // Triangle intersection has no reliable closest-point depth.
                if (body_hinge.fixed_member ||
                    collider_hinge.fixed_member ||
                    small_convex) {
                    // Use body vertices behind the contacted triangle's
                    // plane for closed solids and welded members. The search
                    // margin is only a cap, not the depth of touching faces.
                    const float intersection_depth = fmaxf(
                        0.0F,
                        -fminf(dot(subtract(a0, point_b), normal),
                               fminf(dot(subtract(a1, point_b), normal),
                                     dot(subtract(a2, point_b), normal))));
                    penetration = fminf(
                        margin, intersection_depth + rest_offset) +
                        k_rigid_surface_tolerance;
                } else {
                    // Preserve the established conservative recovery for
                    // general rigid, cloth, soft-body, and rope contacts.
                    penetration = margin + k_rigid_surface_tolerance;
                }
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
    // Static guides use entry-only swept features so an exit face cannot
    // contradict the entry plane. Moving pairs retain ordinary two-body
    // geometry; both cases feed the same AVBD response.
    const bool guided = guided_static_pair(body_hinge, collider_hinge);
    const float rest_offset = guided
        ? fminf(margin, k_rigid_guided_rest_offset) : rigid_rest_offset(margin);
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
            bool starts_near_contact = false;
            Vec3 separating_normal{};
            bool have_separating_normal = false;
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
                                      point_a, point_b, guided);
                const Vec3 delta = subtract(point_a, point_b);
                const float distance =
                    sqrtf(fmaxf(0.0F, length_squared(delta)));
                // Stand-off stabilizes supporting faces, but rounding a
                // sharp tooth tip by that amount can obstruct a genuinely
                // clear sliding path. Edge/vertex sweeps test the surface.
                float contact_offset = rest_offset;
                if (guided &&
                    !triangle_pair_face_contact(a, b, delta)) {
                    const auto &guide = body_hinge.axial ? body_hinge : collider_hinge;
                    const Vec3 direction = normalized_or(delta, {});
                    const float axial = dot(direction, guide.axis);
                    const float angular = dot(guide.axis,
                        cross(subtract(point_a, guide.anchor), direction));
                    // Preserve clearance at a pure axial/rotational stop.
                    // Only the rounded transition between those directions
                    // must not obstruct the other, physically free motion.
                    if (fabsf(axial) > 1.0e-3F && fabsf(angular) > 1.0e-3F)
                        contact_offset = 0.0F;
                }
                if (iteration == 0U)
                    starts_near_contact = distance <= contact_offset + 5.0F * k_rigid_surface_tolerance;
                if (distance <= contact_offset + k_rigid_surface_tolerance) {
                    if (iteration == 0U && !guided) {
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
                    const Vec3 outward_a = body_mesh.shell_normals
                        ? rotate(body_state.orientation, body_mesh.shell_normals[body_triangle]) : Vec3{};
                    const Vec3 outward_b = collider_mesh.shell_normals
                        ? rotate(collider_state.orientation, collider_mesh.shell_normals[collider_triangle]) : Vec3{};
                    if (guided && length_squared(outward_b) > 0.5F) fallback = outward_b;
                    else if (guided && length_squared(outward_a) > 0.5F) fallback = multiply(outward_a, -1.0F);
                    // Match the discrete face normal. Normalizing the tiny
                    // closest-point gap on a large triangle amplifies roundoff
                    // into a spurious tangential impact impulse.
                    const Vec3 plane_projection = subtract(point_a, multiply(
                        collider_normal, dot(subtract(point_a, b[0]), collider_normal)));
                    const bool on_triangle_face = length_squared(subtract(
                        plane_projection, closest_on_triangle(
                            plane_projection, b[0], b[1], b[2]))) <=
                        k_rigid_surface_tolerance * k_rigid_surface_tolerance;
                    const bool small_convex = body_mesh.solid_planes != nullptr &&
                                              body_mesh.index_count <= 96U;
                    // An edge can advance onto a face before impact. Refresh
                    // its geometric normal; retaining the prior edge normal
                    // would round a sharp corner and obstruct free sliding.
                    // Truly coincident features still use the prior plane.
                    const Vec3 normal = guided ? guided_triangle_normal(a[0], a[1], a[2], b[0], b[1], b[2], point_a, point_b,
                            have_separating_normal ? separating_normal : fallback)
                        : (distance <= k_rigid_surface_tolerance ||
                           (small_convex && on_triangle_face))
                            ? fallback : normalized_or(delta, fallback);
                    if (guided && (dot(normal, outward_a) > 1.0e-3F ||
                                   dot(normal, outward_b) < -1.0e-3F)) break;
                    const Vec3 point = multiply(add(point_a, point_b), 0.5F);
                    const float normal_speed = contact_normal_speed(
                        body_state, collider_state, point, normal);
                    if (normal_speed > 1.0e-5F) {
                        break;
                    }
                    const float remaining =
                        -normal_speed * timestep * (1.0F - time) - distance;
                    // A guide sweep can start at an existing resting contact.
                    // Clamp after adding the rest offset: clamping the travel
                    // first adds an entire gap on every tiny support motion.
                    const float penetration = guided
                        ? fmaxf(0.0F, remaining + contact_offset)
                        : fmaxf(0.0F, remaining) + rest_offset;
                    Contact contact{normal, point, penetration + k_rigid_surface_tolerance,
                                    body_hinge, collider_hinge};
                    contact.impact_fraction = starts_near_contact ? 0.0F : time;
                    add_manifold_contact(manifold, contact,
                        fmaxf(margin * 2.0F, 1.0e-4F));
                    break;
                }
                separating_normal = guided_triangle_normal(
                    a[0], a[1], a[2], b[0], b[1], b[2], point_a, point_b,
                    normalized_or(delta, {0.0F, 1.0F, 0.0F}));
                have_separating_normal = true;
                float advancement =
                    (distance - contact_offset) /
                    (speed_bound + k_epsilon) * 0.9F;
                if (guided) {
                    // Euclidean speed is a very loose bound at a shallow
                    // corner: fast axial travel can hide a slow angular hit
                    // until the iteration budget expires. A separating plane
                    // remains valid until its projected vertex gap closes.
                    // Affine swept vertices make this a conservative bound,
                    // including edge/face changes and unequal vertex speeds.
                    const Vec3 plane = guided_triangle_normal(
                        a[0], a[1], a[2], b[0], b[1], b[2], point_a, point_b,
                        separating_normal);
                    float minimum_a = FLT_MAX, maximum_b = -FLT_MAX;
                    float minimum_speed_a = FLT_MAX, maximum_speed_b = -FLT_MAX;
                    for (std::uint32_t vertex = 0; vertex < 3U; ++vertex) {
                        minimum_a = fminf(minimum_a, dot(subtract(a[vertex], point_b), plane));
                        maximum_b = fmaxf(maximum_b, dot(subtract(b[vertex], point_b), plane));
                        minimum_speed_a = fminf(minimum_speed_a, dot(delta_a[vertex], plane));
                        maximum_speed_b = fmaxf(maximum_speed_b, dot(delta_b[vertex], plane));
                    }
                    const float gap = minimum_a - maximum_b - contact_offset;
                    const float closing_speed = maximum_speed_b - minimum_speed_a;
                    if (gap > k_rigid_surface_tolerance && closing_speed <= 0.0F) break;
                    if (gap > 0.0F && closing_speed > k_epsilon)
                        advancement = fmaxf(advancement, 0.9F * gap / closing_speed);
                }
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
    const bool swept = guided_static_pair(body_hinge, collider_hinge) || requires_swept_pair_contact(
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
                previous_body_state, body_state, body_reference, body_hinge, body_mesh,
                body_node.first_triangle,
                body_node.triangle_count, previous_collider_state, collider_state, collider_mesh,
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
            previous_body_state, body_state, body_reference, body_hinge, body_mesh, 0U,
            body_mesh.index_count / 3U,
            previous_collider_state, collider_state, collider_mesh, 0U,
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

// Geometry-only response metric for reducing guided contact candidates.
// It never applies a velocity or pose projection; AVBD handles the response.
__device__ float fixed_hinge_inverse_moment(
    const BodyParameters &parameters, const RigidBodyState &state,
    const HingeContactFrame &hinge) noexcept {
    if ((!hinge.fixed && !hinge.axial_rotation) ||
        parameters.inverse_mass <= k_epsilon) return 0.0F;
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

__device__ Vec3 quaternion_delta_velocity(Quaternion from, Quaternion to,
                                          float timestep) noexcept {
    Quaternion delta = quaternion_multiply(to, conjugate(from));
    if (delta.w < 0.0F) {
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    }
    delta = normalized_quaternion(delta);
    const float vector_size = sqrtf(delta.x * delta.x + delta.y * delta.y +
                                    delta.z * delta.z);
    if (timestep <= 0.0F) {
        return {};
    }
    // The quaternion logarithm has the continuous limit 2*v at identity.
    // Dropping tiny rotations loses constraint impulses during reconstruction.
    if (vector_size <= k_epsilon)
        return {delta.x * (2.0F / timestep), delta.y * (2.0F / timestep),
                delta.z * (2.0F / timestep)};
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

__device__ Vec3 relative_rotation_vector(
    Quaternion frame_a, Quaternion frame_b) noexcept {
    Quaternion relative = normalized_quaternion(
        quaternion_multiply(conjugate(frame_a), frame_b));
    if (relative.w < 0.0F)
        relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    const float size = sqrtf(relative.x * relative.x + relative.y * relative.y +
                             relative.z * relative.z);
    if (size <= k_epsilon)
        return {2.0F * relative.x, 2.0F * relative.y, 2.0F * relative.z};
    const float angle = 2.0F * atan2f(
        size, clamp_scalar(relative.w, -1.0F, 1.0F));
    return {relative.x * angle / size, relative.y * angle / size,
            relative.z * angle / size};
}

__host__ __device__ bool axis_enabled(
    std::uint8_t mask, std::uint32_t axis) noexcept {
    return (mask & static_cast<std::uint8_t>(1U << axis)) != 0U;
}

__device__ std::uint32_t fixed_projection_root(
    const FixedContactProjection *groups, std::uint32_t body) noexcept {
    while (groups[body].root != body) body = groups[body].root;
    return body;
}

// Rebuild before each broad phase so edits, broken joints, and dense-body
// compaction cannot leave stale collision exclusions. Only collision-disabled
// fixed edges are transitive; articulated joints still suppress direct pairs.
__global__ void build_fixed_collision_groups_kernel(
    const RigidConstraintResource *constraints, std::uint32_t capacity,
    const RigidBodyId *ids, std::uint32_t count,
    FixedContactProjection *groups) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    for (std::uint32_t body = 0U; body < count; ++body)
        groups[body].root = body;
    for (std::uint32_t index = 0U; index < capacity; ++index) {
        const auto &constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken || !constraint.options.disable_collisions ||
            constraint.options.type != RigidConstraintType::fixed) continue;
        auto a = find_rigid_body_dense(constraint.options.body_a, ids, count);
        auto b = find_rigid_body_dense(constraint.options.body_b, ids, count);
        if (a == k_invalid_dense || b == k_invalid_dense) continue;
        a = fixed_projection_root(groups, a);
        b = fixed_projection_root(groups, b);
        if (a != b) groups[a > b ? a : b].root = a < b ? a : b;
    }
    for (std::uint32_t body = 0U; body < count; ++body)
        groups[body].root = fixed_projection_root(groups, body);
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
    result.static_body = parameters[dense].motion == MotionType::static_body;
    if (parameters[dense].inverse_mass <= 0.0F) {
        reference = state.position;
        return result;
    }
    const RigidBodyId body = ids[dense];
    std::uint32_t incident_count = 0U;
    for (std::uint32_t index = 0U; index < constraint_capacity; ++index) {
        const RigidConstraintResource &constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken)
            continue;
        const bool is_a = same_rigid_body_id(
            constraint.options.body_a, body);
        const bool is_b = same_rigid_body_id(
            constraint.options.body_b, body);
        if (!is_a && !is_b) continue;
        ++incident_count;
        if (constraint.options.type == RigidConstraintType::fixed)
            result.fixed_member = true;
        if (constraint.options.type == RigidConstraintType::point)
            result.point_member = true;
        const auto type = constraint.options.type;
        const bool axial = type == RigidConstraintType::piston ||
                           type == RigidConstraintType::slider;
        const RigidBodyId other = is_a
            ? constraint.options.body_b : constraint.options.body_a;
        const std::uint32_t other_dense = find_rigid_body_dense(
            other, ids, body_count);
        const bool static_anchor = other_dense != k_invalid_dense &&
            parameters[other_dense].motion == MotionType::static_body;
        // Breakable and moving-anchor joints do not have a permanent static
        // guide for feature selection. All joint responses still use AVBD.
        if (result.present || (type != RigidConstraintType::hinge &&
            !(axial && static_anchor &&
              constraint.options.breaking_impulse_threshold <= 0.0F)))
            continue;
        const Vec3 local_axis = axial ? Vec3{1.0F, 0.0F, 0.0F}
                                     : Vec3{0.0F, 0.0F, 1.0F};
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
                   local_axis), local_axis);
        result.present = true;
        result.fixed = static_anchor && !axial;
        result.axial = static_anchor && axial;
        result.axial_rotation = result.axial && type == RigidConstraintType::piston;
        if (static_anchor) {
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
                       local_axis),
                result.axis);
        }
    }
    // Guide-space feature reduction applies only to one unbreakable static
    // guide; mechanisms with additional joints keep ordinary geometry.
    if (result.axial && incident_count != 1U) {
        result.axial = result.axial_rotation = result.present = false;
    }
    if (result.axial) {
        result.inverse_mass = parameters[dense].inverse_mass;
        result.inverse_moment = fixed_hinge_inverse_moment(parameters[dense], state, result);
    }
    reference = result.present && !result.axial ? result.anchor : state.position;
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
    const FixedContactProjection *fixed_groups,
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
    if (active && fixed_groups != nullptr &&
        fixed_groups[index].root == fixed_groups[collider_index].root)
        active = false;
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
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) {
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
            collider_mesh, margin);
        BoundsTransform previous_body_transform{};
        BoundsTransform previous_collider_transform{};
        if (swept) {
            previous_body_transform = bounds_transform(previous_states[index]);
            previous_collider_transform =
                bounds_transform(previous_states[collider_index]);
        }
        // Dense meshes otherwise transform each leaf once for every leaf of
        // the other mesh, twice. Cache identical bounds cooperatively while
        // retaining the original candidate order and conservative sweep.
        extern __shared__ WorldAabb cached_bounds[];
        const std::uint32_t total_leaves =
            body_mesh.bvh_leaf_count + collider_mesh.bvh_leaf_count;
        const bool cache_bounds = total_leaves <= shared_leaf_capacity;
        if (cache_bounds) {
            for (std::uint32_t leaf = threadIdx.x; leaf < total_leaves;
                 leaf += blockDim.x) {
                const bool body = leaf < body_mesh.bvh_leaf_count;
                const TriangleMeshResource &mesh = body ? body_mesh : collider_mesh;
                const std::uint32_t local_leaf = body
                    ? leaf : leaf - body_mesh.bvh_leaf_count;
                const BvhNode &node = mesh.bvh_nodes[mesh.bvh_leaves[local_leaf]];
                transformed_motion_bounds(
                    node.minimum, node.maximum,
                    body ? previous_body_transform : previous_collider_transform,
                    body ? body_transform : collider_transform,
                    swept, body ? margin : 0.0F,
                    cached_bounds[leaf].minimum, cached_bounds[leaf].maximum);
            }
        }
        __syncthreads();
        Vec3 body_minimum{};
        Vec3 body_maximum{};
        Vec3 collider_minimum{};
        Vec3 collider_maximum{};
        std::uint32_t local_count = 0U;
        for (RigidLeafPairCursor cursor(collider_mesh.bvh_leaf_count);
             cursor.body_leaf < body_mesh.bvh_leaf_count; cursor.advance()) {
            const std::uint32_t body_leaf = cursor.body_leaf;
            const std::uint32_t collider_leaf = cursor.collider_leaf;
            const BvhNode &body_node =
                body_mesh.bvh_nodes[body_mesh.bvh_leaves[body_leaf]];
            const BvhNode &collider_node = collider_mesh.bvh_nodes[
                collider_mesh.bvh_leaves[collider_leaf]];
            if (cache_bounds) {
                body_minimum = cached_bounds[body_leaf].minimum;
                body_maximum = cached_bounds[body_leaf].maximum;
                collider_minimum = cached_bounds[
                    body_mesh.bvh_leaf_count + collider_leaf].minimum;
                collider_maximum = cached_bounds[
                    body_mesh.bvh_leaf_count + collider_leaf].maximum;
            } else {
                transformed_motion_bounds(
                    body_node.minimum, body_node.maximum, previous_body_transform,
                    body_transform, swept, margin, body_minimum, body_maximum);
                transformed_motion_bounds(
                    collider_node.minimum, collider_node.maximum,
                    previous_collider_transform, collider_transform,
                    swept, 0.0F, collider_minimum, collider_maximum);
            }
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
            leaf_pair_counts[pair] = prefix > leaf_pairs_per_slot
                ? k_leaf_pair_overflow : prefix;
        }
        __syncthreads();
        if (candidate_count > leaf_pairs_per_slot) {
            continue;
        }

        LeafPair *pair_candidates =
            leaf_pairs + static_cast<std::size_t>(active_index) *
                             leaf_pairs_per_slot;
        std::uint32_t output_index = offsets[threadIdx.x];
        for (RigidLeafPairCursor cursor(collider_mesh.bvh_leaf_count);
             cursor.body_leaf < body_mesh.bvh_leaf_count; cursor.advance()) {
            const std::uint32_t body_leaf = cursor.body_leaf;
            const std::uint32_t collider_leaf = cursor.collider_leaf;
            const BvhNode &body_node =
                body_mesh.bvh_nodes[body_mesh.bvh_leaves[body_leaf]];
            const BvhNode &collider_node = collider_mesh.bvh_nodes[
                collider_mesh.bvh_leaves[collider_leaf]];
            if (cache_bounds) {
                body_minimum = cached_bounds[body_leaf].minimum;
                body_maximum = cached_bounds[body_leaf].maximum;
                collider_minimum = cached_bounds[
                    body_mesh.bvh_leaf_count + collider_leaf].minimum;
                collider_maximum = cached_bounds[
                    body_mesh.bvh_leaf_count + collider_leaf].maximum;
            } else {
                transformed_motion_bounds(
                    body_node.minimum, body_node.maximum, previous_body_transform,
                    body_transform, swept, margin, body_minimum, body_maximum);
                transformed_motion_bounds(
                    collider_node.minimum, collider_node.maximum,
                    previous_collider_transform, collider_transform,
                    swept, 0.0F, collider_minimum, collider_maximum);
            }
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
    std::uint32_t leaf_pairs_per_slot,
    ContactManifold *leaf_manifolds, std::uint32_t blocks_per_pair,
    float timestep, ContactManifold *manifolds) {
    for (std::uint32_t active_index = blockIdx.x / blocks_per_pair;
         active_index < *active_pair_count;
         active_index += gridDim.x / blocks_per_pair) {
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        ContactManifold &output = manifolds[active_index];
        const std::uint32_t candidate_count = leaf_pair_counts[pair];
        if (candidate_count == k_leaf_pair_face_patch) continue;
        if (candidate_count == 0U) {
            if (threadIdx.x == 0U && blockIdx.x % blocks_per_pair == 0U) {
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
            if (threadIdx.x == 0U && blockIdx.x % blocks_per_pair == 0U) {
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
                             leaf_pairs_per_slot;
        const float separation = fmaxf(
            (parameters[index].collision_margin +
             parameters[collider_index].collision_margin) *
                2.0F,
            1.0e-4F);
        const float collision_margin =
            parameters[index].collision_margin +
            parameters[collider_index].collision_margin;
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
        // Guided mechanisms can have sharp features much smaller than the
        // broad-phase search margin. Sweep even their short resting motions.
        const bool swept = guided_static_pair(body_hinge, collider_hinge) ||
            requires_swept_pair_contact(previous_states[index], states[index], body_mesh,
                previous_states[collider_index], states[collider_index],
                collider_mesh, collision_margin);
        for (std::uint32_t wave = (blockIdx.x % blocks_per_pair) * blockDim.x;
             wave < candidate_count; wave += blockDim.x * blocks_per_pair) {
            ContactManifold local{};
            const std::uint32_t candidate_index = wave + threadIdx.x;
            if (candidate_index < candidate_count) {
                const LeafPair candidate = pair_candidates[candidate_index];
                collide_triangle_ranges(
                    previous_states[index], states[index], body_reference, body_hinge,
                    body_mesh,
                    candidate.body_first,
                    candidate.body_count, previous_states[collider_index], states[collider_index],
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
            if (leaf_manifolds != nullptr) {
                if (candidate_index < candidate_count)
                    leaf_manifolds[static_cast<std::size_t>(active_index) *
                                       leaf_pairs_per_slot + candidate_index] = local;
                continue;
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
        if (leaf_manifolds == nullptr && threadIdx.x == 0U) {
            output = reduced;
        }
        __syncthreads();
    }
}

// A fixed candidate-order reduction matches the single-block evaluator;
// parallelizing triangle work must not reorder the contact manifold.
__global__ void reduce_rigid_leaf_manifolds_kernel(
    const BodyParameters *parameters, std::uint32_t count,
    const std::uint32_t *active_pairs, const std::uint32_t *active_pair_count,
    const std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    const ContactManifold *leaf_manifolds, ContactManifold *manifolds) {
    const std::uint32_t active_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_index >= *active_pair_count) return;
    const std::uint32_t pair = active_pairs[active_index];
    const std::uint32_t candidate_count = leaf_pair_counts[pair];
    if (candidate_count == k_leaf_pair_overflow || candidate_count == k_leaf_pair_face_patch) return;
    const float separation = fmaxf(
        (parameters[pair / count].collision_margin +
         parameters[pair % count].collision_margin) * 2.0F, 1.0e-4F);
    ContactManifold reduced{};
    for (std::uint32_t candidate = 0U; candidate < candidate_count; ++candidate) {
        const ContactManifold &local = leaf_manifolds[
            static_cast<std::size_t>(active_index) * leaf_pairs_per_slot + candidate];
        for (std::uint32_t contact = 0U; contact < local.count; ++contact)
            add_manifold_contact(reduced, local.contacts[contact], separation);
    }
    manifolds[active_index] = reduced;
}

__device__ void initialize_contact_solve(ContactManifold &manifold,
    const RigidBodyState &body, const RigidBodyState &collider, bool persistent) noexcept {
    manifold.initial_relative_position = subtract(body.position, collider.position);
    for (std::uint32_t point = 0U; point < manifold.count; ++point) {
        auto &contact = manifold.contacts[point];
        contact.persistent = guided_static_contact(contact) || (persistent && !contact.body_hinge.present &&
            !contact.collider_hinge.present && !contact.body_hinge.fixed_member &&
            !contact.collider_hinge.fixed_member);
        contact.initial_normal_speed = contact_normal_speed(
            body, collider, contact.point, contact.normal);
    }
}

// Try convex faces before triangle evaluation, marking handled pairs so that
// neither the triangle evaluator nor its reduction repeats their work. A
// second call initializes every final manifold and handles BVH overflow.
__global__ void finalize_rigid_contact_manifolds_kernel(
    const BodyParameters *parameters, const RigidBodyState *previous_states,
    const RigidBodyState *states, std::uint32_t count,
    const TriangleMeshResource *meshes, const RigidBodyId *ids,
    const RigidConstraintResource *constraints,
    std::uint32_t constraint_capacity, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep,
    ContactManifold *manifolds, bool prepare_faces) {
    for (std::uint32_t active_index = blockIdx.x * blockDim.x + threadIdx.x;
         active_index < *active_pair_count;
         active_index += gridDim.x * blockDim.x) {
        const std::uint32_t pair = active_pairs[active_index];
        const std::uint32_t index = pair / count;
        const std::uint32_t collider_index = pair % count;
        const auto &body_mesh = meshes[parameters[index].mesh.index];
        const auto &collider_mesh = meshes[parameters[collider_index].mesh.index];
        const float margin = parameters[index].collision_margin +
                             parameters[collider_index].collision_margin;
        // Curved/concave meshes keep their triangle path, and hinge/fixed
        // members keep their established impulse response.
        const bool persistent = body_mesh.solid_planes && body_mesh.index_count <= 96U &&
            (parameters[collider_index].motion != MotionType::dynamic ||
             (collider_mesh.solid_planes && collider_mesh.index_count <= 96U));
        if (!prepare_faces && leaf_pair_counts[pair] != k_leaf_pair_overflow) {
            initialize_contact_solve(manifolds[active_index],
                                     states[index], states[collider_index], persistent);
            continue;
        }
        const bool face_pair = prepare_faces && body_mesh.solid_planes && collider_mesh.solid_planes &&
            body_mesh.index_count <= 96U && collider_mesh.index_count <= 96U &&
            !requires_swept_pair_contact(previous_states[index], states[index], body_mesh,
                previous_states[collider_index], states[collider_index], collider_mesh, margin);
        if (prepare_faces && !face_pair) continue;
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
        if (prepare_faces) {
            if (guided_static_pair(body_hinge, collider_hinge)) continue;
            if (convex_face_manifold(states[index], body_mesh, body_hinge,
                states[collider_index], collider_mesh, collider_hinge, margin,
                manifolds[active_index])) leaf_pair_counts[pair] = k_leaf_pair_face_patch;
            continue;
        }
        manifolds[active_index] = collide_meshes(
            parameters[index], previous_states[index], states[index],
            previous_body_reference, body_reference, body_hinge,
            meshes[parameters[index].mesh.index], parameters[collider_index],
            previous_states[collider_index], states[collider_index],
            meshes[parameters[collider_index].mesh.index], collider_hinge,
            timestep);
        initialize_contact_solve(manifolds[active_index],
                                 states[index], states[collider_index], persistent);
    }
}

__global__ void load_rigid_contact_cache_kernel(
    ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_count, const RigidBodyState *states,
    const RigidBodyId *ids, std::uint32_t body_count,
    const CachedContactPair *cache, const std::uint32_t *slots,
    std::uint32_t capacity, std::uint64_t epoch, float timestep) {
    for (std::uint32_t active = blockIdx.x * blockDim.x + threadIdx.x;
         active < *active_count; active += blockDim.x * gridDim.x) {
        const std::uint32_t pair = active_pairs[active], slot = slots[pair];
        const std::uint32_t body = pair / body_count, collider = pair % body_count;
        if (slot >= capacity) continue;
        const auto &saved = cache[slot];
        if (saved.pair != pair || saved.epoch + 1U != epoch ||
            saved.body.index != ids[body].index || saved.body.generation != ids[body].generation ||
            saved.collider.index != ids[collider].index || saved.collider.generation != ids[collider].generation ||
            fabsf(saved.timestep - timestep) > 1.0e-7F) continue;
        auto &manifold = manifolds[active];
        std::uint32_t used = 0U;
        for (std::uint32_t point = 0U; point < manifold.count; ++point) {
            auto &contact = manifold.contacts[point];
            const Vec3 local = inverse_rotate(states[body].orientation,
                                             subtract(contact.point, states[body].position));
            float nearest = 0.02F * 0.02F;
            std::uint32_t match = 8U;
            for (std::uint32_t previous = 0U; previous < saved.count; ++previous) {
                if ((used & (1U << previous)) != 0U ||
                    dot(contact.normal, saved.contacts[previous].normal) < 0.99F) continue;
                const float distance = length_squared(subtract(local, saved.contacts[previous].local_point));
                if (distance < nearest) { nearest = distance; match = previous; }
            }
            if (match == 8U) continue;
            used |= 1U << match;
            const auto &previous = saved.contacts[match];
            for (unsigned axis = 0; axis < 3; ++axis)
                contact.avbd_dual[axis] = previous.avbd_dual[axis];
            const Vec3 tangent0 = normalized_or(cross(contact.normal,
                fabsf(contact.normal.x) < 0.57735F ? Vec3{1, 0, 0} : Vec3{0, 1, 0}), {0, 0, 1});
            const Vec3 tangent1 = cross(contact.normal, tangent0);
            contact.avbd_dual[1].lambda = dot(previous.friction_impulse, tangent0) / timestep;
            contact.avbd_dual[2].lambda = dot(previous.friction_impulse, tangent1) / timestep;
            contact.avbd_anchor_a = previous.avbd_anchor_a;
            contact.avbd_anchor_b = previous.avbd_anchor_b;
            contact.avbd_cached = true;
            contact.accumulated_normal_impulse = previous.normal_impulse;
            const Vec3 friction = previous.friction_impulse;
            contact.accumulated_friction_impulse = subtract(friction,
                multiply(contact.normal, dot(friction, contact.normal)));
        }
    }
}

__global__ void save_rigid_contact_cache_kernel(
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_count, const RigidBodyState *states,
    const RigidBodyId *ids, std::uint32_t body_count,
    CachedContactPair *cache, std::uint32_t *slots,
    std::uint32_t capacity, std::uint64_t epoch, float timestep,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity) {
    for (std::uint32_t active = blockIdx.x * blockDim.x + threadIdx.x;
         active < *active_count; active += blockDim.x * gridDim.x) {
        const std::uint32_t pair = active_pairs[active];
        const std::uint32_t body = pair / body_count, collider = pair % body_count;
        const auto &manifold = manifolds[active];
        if (event_capacity > 0U) {
            const auto offset = event_offsets[active];
            for (std::uint32_t point = 0U; point < manifold.count &&
                 offset < event_capacity && point < event_capacity - offset; ++point) {
                const auto &contact = manifold.contacts[point];
                auto &event = events[offset + point];
                event.normal_impulse = contact.persistent
                    ? contact.accumulated_normal_impulse : contact.reported_normal_impulse;
                event.friction_impulse = contact.persistent
                    ? contact.accumulated_friction_impulse : contact.reported_friction_impulse;
            }
        }
        if (manifold.response_slot >= capacity) continue;
        auto &saved = cache[manifold.response_slot];
        saved.pair = pair; saved.body = ids[body]; saved.collider = ids[collider];
        saved.epoch = epoch; saved.timestep = timestep; saved.count = manifold.count;
        for (std::uint32_t point = 0U; point < manifold.count; ++point) {
            const auto &contact = manifold.contacts[point];
            saved.contacts[point] = {
                inverse_rotate(states[body].orientation, subtract(contact.point, states[body].position)),
                contact.normal, contact.accumulated_normal_impulse, contact.accumulated_friction_impulse};
            for (unsigned axis = 0; axis < 3; ++axis)
                saved.contacts[point].avbd_dual[axis] = contact.avbd_dual[axis];
            saved.contacts[point].avbd_anchor_a = contact.avbd_anchor_a;
            saved.contacts[point].avbd_anchor_b = contact.avbd_anchor_b;
        }
        slots[pair] = manifold.response_slot;
    }
}

__global__ void prepare_parallel_contact_events_kernel(
    std::uint32_t count, ContactManifold *manifolds,
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
    std::uint32_t cursor = 0U;
    std::uint32_t response_slot = 0U;
    for (std::uint32_t active_index = 0U;
         active_index < *active_pair_count; ++active_index) {
        auto &manifold = manifolds[active_index];
        // Cache actual contact patches, not empty broad-phase candidates.
        manifold.response_slot = manifold.count != 0U ? response_slot++ : 0xffffffffU;
        if (!collect_events) continue;
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

__global__ void gather_rigid_debug_samples_kernel(
    const RigidBodyId *ids, const RigidBodyState *states,
    const Vec3 *forces, const Vec3 *torques, std::uint32_t count,
    PhysicsDebugRigidSample *samples) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) samples[index] = {ids[index], states[index], forces[index], torques[index]};
}

__global__ void capture_rigid_inputs_kernel(
    const BodyAccumulator *accumulators, Vec3 *forces, Vec3 *torques,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    forces[index] = accumulators[index].force;
    torques[index] = accumulators[index].torque;
}
