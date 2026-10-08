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

struct RigidConstraintAxisGeometry {
    Vec3 axis{};
    Vec3 inverse_angular_a{};
    Vec3 inverse_angular_b{};
    float linear_denominator{};
    float angular_denominator{};
};

struct RigidConstraintGeometry {
    std::uint32_t dense_a{};
    std::uint32_t dense_b{};
    Vec3 arm_a{};
    Vec3 arm_b{};
    Vec3 anchor_error{};
    Vec3 rotation_error{};
    Vec3 hinge_alignment_error{};
    Vec3 piston_alignment_error{};
    RigidConstraintAxisGeometry axes[3]{};
};

struct RigidConstraintResource {
    RigidConstraintOptions options{};
    RigidConstraintState state{};
    std::uint32_t generation{1U};
    bool alive{};
    // Positions and orientations stay fixed during the velocity iterations.
    // Rebuild once per substep, including after edits and body compaction.
    RigidConstraintGeometry geometry{};
};

struct FixedContactProjection {
    std::uint32_t root{};
    bool movable{};
    Vec3 translation{};
};

struct RigidCompound {
    std::uint32_t root{};
    std::uint32_t member_count{};
    bool eligible{};
    bool blocked{};
    Vec3 center{};
    float inverse_mass{};
    // Rows of the world-space inverse inertia tensor about center.
    Vec3 inverse_inertia[3]{};
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
    // Transient rows have no retained lambda; collect their diagnostic sum
    // here, then publish all event records once after contact/joint solving.
    float reported_normal_impulse{};
    Vec3 reported_friction_impulse{};
};

struct ContactManifold {
    Contact contacts[8]{};
    std::uint32_t count{};
    Vec3 initial_relative_position{};
    bool face_patch{};
    std::uint32_t response_slot{0xffffffffU};
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
};

struct CachedContactPair {
    std::uint32_t pair{};
    RigidBodyId body{}, collider{};
    std::uint64_t epoch{};
    float timestep{};
    std::uint32_t count{};
    CachedContact contacts[8]{};
};

// Optional bounded effective-mass cache. Every contact uses the same row solver;
// cache capacity, world size and unrelated joints never select different math.
struct ContactResponseRow {
    Vec3 arm_a{}, arm_b{};
    float normal_inverse_mass{};
    Vec3 tangents[2]{};
    Vec3 angular_a[3]{}, angular_b[3]{}; // normal, tangent X, tangent Y
    float tangent_xx{}, tangent_xy{}, tangent_yy{};
};

struct ContactResponsePatch {
    solver::ContactMaterial material{};
    Vec3 position_a{}, position_b{};
    Quaternion orientation_a{}, orientation_b{};
    float normal_inverse_mass[8]{};
    bool ready{};
    bool rows_ready{};
    Vec3 inertia_a[3]{}, inertia_b[3]{}; // world-space tensor columns
    ContactResponseRow rows[8]{};
};

struct ContactSolveDelta {
    float velocity{}, position{}, impulse{};
};

struct ContactResidual {
    unsigned velocity_change{}, position_change{}, impulse_change{};
};

struct ContactIsland {
    std::uint32_t parent{};
    std::uint32_t budget{};
    std::uint32_t active_root{};
    Vec3 previous_linear{}, previous_angular{}, previous_position{};
    ContactResidual residual{};
    solver::ContactConvergence convergence{};
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
    // Entry-only contacts and post-solve trajectory integration form one
    // solver path, restricted to guides hitting static obstacles. A moving
    // gear needs the regular two-body manifold and position recovery too.
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
                    const Vec3 normal = guided && have_separating_normal &&
                                             distance <= k_rigid_surface_tolerance
                        ? separating_normal
                        : guided ? guided_triangle_normal(a[0], a[1], a[2], b[0], b[1], b[2], point_a, point_b,
                            have_separating_normal ? separating_normal : fallback)
                        : distance <= k_rigid_surface_tolerance
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

__device__ Vec3 contact_point_velocity(
    const RigidBodyState &state, const HingeContactFrame &hinge,
    Vec3 point) noexcept {
    if (!hinge.fixed && !hinge.axial) {
        return add(state.linear_velocity,
                   cross(state.angular_velocity,
                         subtract(point, state.position)));
    }
    const Vec3 angular = multiply(hinge.axis,
        hinge.fixed || hinge.axial_rotation
            ? dot(state.angular_velocity, hinge.axis) : 0.0F);
    return add(hinge.axial
                   ? multiply(hinge.axis, dot(state.linear_velocity, hinge.axis))
                   : Vec3{},
               cross(angular, subtract(point, hinge.anchor)));
}

__device__ Vec3 compound_inverse_inertia_world(
    const RigidCompound &compound, Vec3 value) noexcept {
    return {dot(compound.inverse_inertia[0], value),
            dot(compound.inverse_inertia[1], value),
            dot(compound.inverse_inertia[2], value)};
}

__device__ float contact_direction_inverse_mass(
    const BodyParameters &parameters, const RigidBodyState &state,
    const HingeContactFrame &hinge, Vec3 point, Vec3 direction,
    const RigidCompound *compounds, std::uint32_t index, bool unit_direction = true) noexcept {
    if (compounds != nullptr && compounds[index].eligible) {
        const RigidCompound &compound = compounds[compounds[index].root];
        const Vec3 arm = subtract(point, compound.center);
        const Vec3 angular = cross(arm, direction);
        return compound.inverse_mass * (unit_direction ? 1.0F : length_squared(direction)) +
            dot(cross(compound_inverse_inertia_world(compound, angular), arm),
                direction);
    }
    if (hinge.fixed || hinge.axial) {
        const float jacobian = dot(
            cross(hinge.axis, subtract(point, hinge.anchor)), direction);
        const float axial = hinge.axial ? dot(hinge.axis, direction) : 0.0F;
        return parameters.inverse_mass * axial * axial + jacobian * jacobian *
            fixed_hinge_inverse_moment(parameters, state, hinge);
    }
    const Vec3 arm = subtract(point, state.position);
    const Vec3 angular = cross(arm, direction);
    return parameters.inverse_mass * (unit_direction ? 1.0F : length_squared(direction)) +
        dot(cross(inverse_inertia_world(parameters, state, angular), arm),
            direction);
}

__device__ void apply_contact_velocity_impulse(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, std::uint32_t index,
    const HingeContactFrame &hinge, Vec3 point, Vec3 impulse,
    const RigidCompound *compounds, RigidBodyState *local_state = nullptr) noexcept {
    const BodyParameters &body = parameters[index];
    RigidBodyState &state = local_state != nullptr ? *local_state : states[index];
    if (compounds != nullptr && compounds[index].eligible) {
        const std::uint32_t root = compounds[index].root;
        const RigidCompound &compound = compounds[root];
        const Vec3 linear_delta = multiply(impulse, compound.inverse_mass);
        const Vec3 angular_delta = compound_inverse_inertia_world(
            compound, cross(subtract(point, compound.center), impulse));
        for (std::uint32_t member = 0U; member < count; ++member) {
            if (!compounds[member].eligible ||
                compounds[member].root != root) continue;
            states[member].linear_velocity = add(
                states[member].linear_velocity,
                add(linear_delta,
                    cross(angular_delta,
                          subtract(states[member].position,
                                   compound.center))));
            states[member].angular_velocity = add(
                states[member].angular_velocity, angular_delta);
        }
        return;
    }
    if (body.inverse_mass <= 0.0F) return;
    if (hinge.fixed || hinge.axial) {
        const float angular_impulse = dot(
            hinge.axis, cross(subtract(point, hinge.anchor), impulse));
        const Vec3 angular_delta = multiply(
            hinge.axis, angular_impulse *
                fixed_hinge_inverse_moment(body, state, hinge));
        state.angular_velocity = add(state.angular_velocity, angular_delta);
        state.linear_velocity = add(
            state.linear_velocity,
            add(hinge.axial
                    ? multiply(hinge.axis, body.inverse_mass * dot(impulse, hinge.axis))
                    : Vec3{},
                cross(angular_delta, subtract(state.position, hinge.anchor))));
        return;
    }
    state.linear_velocity = add(
        state.linear_velocity, multiply(impulse, body.inverse_mass));
    state.angular_velocity = add(
        state.angular_velocity,
        inverse_inertia_world(body, state,
                              cross(subtract(point, state.position), impulse)));
}

__device__ bool contact_response_matches(
    const ContactResponsePatch &patch, const RigidBodyState &a,
    const RigidBodyState &b) noexcept {
    const auto same = [](Quaternion x, Quaternion y) {
        return x.x == y.x && x.y == y.y && x.z == y.z && x.w == y.w;
    };
    return patch.ready && same(patch.orientation_a, a.orientation) &&
        same(patch.orientation_b, b.orientation) &&
        patch.position_a.x == a.position.x && patch.position_a.y == a.position.y &&
        patch.position_a.z == a.position.z && patch.position_b.x == b.position.x &&
        patch.position_b.y == b.position.y && patch.position_b.z == b.position.z;
}

__device__ solver::ContactMaterial prepare_contact_material(
    const BodyParameters *parameters, unsigned a, unsigned b, const Contact &contact) {
    return {
        fminf(parameters[a].restitution, parameters[b].restitution),
        sqrtf(parameters[a].friction * parameters[b].friction),
        contact.persistent && !guided_static_contact(contact)
            ? rigid_rest_offset(parameters[a].collision_margin + parameters[b].collision_margin)
            : k_rigid_surface_tolerance,
        contact.body_hinge.fixed_member || contact.collider_hinge.fixed_member};
}

template<bool FreePair = false>
__device__ __forceinline__ void prepare_contact_response_row(
    const BodyParameters *parameters, const RigidBodyState &state_a, const RigidBodyState &state_b,
    unsigned a, unsigned b, const Contact &contact, const RigidCompound *compounds,
    ContactResponseRow &row, const ContactResponsePatch *patch) {
        row.arm_a = subtract(contact.point,
            !FreePair && (contact.body_hinge.fixed || contact.body_hinge.axial)
                ? contact.body_hinge.anchor : state_a.position);
        row.arm_b = subtract(contact.point,
            !FreePair && (contact.collider_hinge.fixed || contact.collider_hinge.axial)
                ? contact.collider_hinge.anchor : state_b.position);
        if (!FreePair) {
            row.normal_inverse_mass = contact_direction_inverse_mass(
            parameters[a], state_a, contact.body_hinge, contact.point,
            contact.normal, compounds, a) + contact_direction_inverse_mass(
            parameters[b], state_b, contact.collider_hinge, contact.point,
            contact.normal, compounds, b);
        // Unretained impacts preserve their established global arithmetic;
        // prepared basis interpolation is for rows with stable feature history.
            return;
        }
        row.tangents[0] = normalized_or(cross(contact.normal,
            fabsf(contact.normal.x) < 0.5F ? Vec3{1, 0, 0} : Vec3{0, 1, 0}), {0, 0, 1});
        row.tangents[1] = cross(contact.normal, row.tangents[0]);
        const auto angular = [&](unsigned index, const HingeContactFrame &hinge, Vec3 arm, Vec3 direction) {
            const auto &state = index == a ? state_a : state_b;
            if (!FreePair && compounds != nullptr && compounds[index].eligible) {
                const auto &compound = compounds[compounds[index].root];
                return compound_inverse_inertia_world(compound, cross(subtract(contact.point, compound.center), direction));
            }
            if (!FreePair && (hinge.fixed || hinge.axial))
                return multiply(hinge.axis, dot(hinge.axis, cross(arm, direction)) *
                    fixed_hinge_inverse_moment(parameters[index], state, hinge));
            const auto moment = cross(arm, direction);
            if (patch != nullptr) {
                const auto *tensor = index == a ? patch->inertia_a : patch->inertia_b;
                return add(add(multiply(tensor[0], moment.x), multiply(tensor[1], moment.y)),
                           multiply(tensor[2], moment.z));
            }
            return inverse_inertia_world(parameters[index], state, moment);
        };
        for (unsigned axis = 0U; axis < 3U; ++axis) {
            const auto direction = axis == 0U ? contact.normal : row.tangents[axis - 1U];
            row.angular_a[axis] = angular(a, contact.body_hinge, row.arm_a, direction);
            row.angular_b[axis] = angular(b, contact.collider_hinge, row.arm_b, direction);
        }
        const auto normal_mass = [&](unsigned index, const HingeContactFrame &hinge, Vec3 arm, Vec3 angular_delta) {
            if (!FreePair && ((compounds != nullptr && compounds[index].eligible) || hinge.fixed || hinge.axial))
                return contact_direction_inverse_mass(parameters[index], index == a ? state_a : state_b,
                    hinge, contact.point, contact.normal, compounds, index);
            return parameters[index].inverse_mass + dot(cross(angular_delta, arm), contact.normal);
        };
        row.normal_inverse_mass = normal_mass(a, contact.body_hinge, row.arm_a, row.angular_a[0]) +
            normal_mass(b, contact.collider_hinge, row.arm_b, row.angular_b[0]);
        const auto point_response = [&](unsigned index, const HingeContactFrame &hinge,
                                         Vec3 arm, Vec3 direction, Vec3 angular_delta) {
            if (!FreePair && compounds != nullptr && compounds[index].eligible) {
                const auto &compound = compounds[compounds[index].root];
                return add(multiply(direction, compound.inverse_mass),
                    cross(angular_delta, subtract(contact.point, compound.center)));
            }
            if (!FreePair && (hinge.fixed || hinge.axial))
                return add(hinge.axial ? multiply(hinge.axis, parameters[index].inverse_mass * dot(direction, hinge.axis)) : Vec3{},
                    cross(angular_delta, arm));
            return add(multiply(direction, parameters[index].inverse_mass), cross(angular_delta, arm));
        };
        const auto tangent_response = [&](unsigned axis) {
            return add(point_response(a, contact.body_hinge, row.arm_a, row.tangents[axis], row.angular_a[axis + 1U]),
                       point_response(b, contact.collider_hinge, row.arm_b, row.tangents[axis], row.angular_b[axis + 1U]));
        };
        const auto tangent_x = tangent_response(0);
        row.tangent_xx = dot(tangent_x, row.tangents[0]);
        row.tangent_xy = dot(tangent_x, row.tangents[1]);
        row.tangent_yy = dot(tangent_response(1), row.tangents[1]);
}

template<bool FreePair>
__device__ __forceinline__ void prepare_contact_response(
    const BodyParameters *parameters, const RigidBodyState &state_a, const RigidBodyState &state_b,
    std::uint32_t a, std::uint32_t b, const Contact *contacts,
    std::uint32_t contact_count, const RigidCompound *compounds,
    ContactResponsePatch &patch) noexcept {
    // Translation changes contact arms, not world-space inertia. Retain each
    // tensor until its orientation changes; projection must not rebuild six
    // quaternion transforms per row on every sweep.
    const auto same = [](Quaternion x, Quaternion y) {
        return x.x == y.x && x.y == y.y && x.z == y.z && x.w == y.w;
    };
    if (FreePair) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            const Vec3 direction{axis == 0 ? 1.0F : 0.0F, axis == 1 ? 1.0F : 0.0F, axis == 2 ? 1.0F : 0.0F};
            if (!patch.ready || !same(patch.orientation_a, state_a.orientation))
                patch.inertia_a[axis] = inverse_inertia_world(parameters[a], state_a, direction);
            if (!patch.ready || !same(patch.orientation_b, state_b.orientation))
                patch.inertia_b[axis] = inverse_inertia_world(parameters[b], state_b, direction);
        }
    }
    patch.orientation_a = state_a.orientation;
    patch.orientation_b = state_b.orientation;
    patch.position_a = state_a.position;
    patch.position_b = state_b.position;
    if (!patch.ready) patch.material = prepare_contact_material(parameters, a, b, contacts[0]);
    (void)contact_count;
    (void)compounds;
    patch.ready = true;
}

// Body-response adapter only; it never chooses contact-solving equations.
// Preserve the operation order of constrained and compound impulse application,
// and cache normal effective mass only while the bodies' current poses match.
struct RigidContactAdapter {
    const BodyParameters *parameters;
    RigidBodyState *states;
    std::uint32_t count, a, b;
    const Contact &contact;
    const RigidCompound *compounds;
    float normal_inverse_mass;
    struct Response { Vec3 direction{}; float inverse_mass{}; };

    __device__ Vec3 relative_velocity() const noexcept {
        return subtract(contact_point_velocity(states[a], contact.body_hinge, contact.point),
                        contact_point_velocity(states[b], contact.collider_hinge, contact.point));
    }
    __device__ Response normal_response() const noexcept {
        return {contact.normal, normal_inverse_mass};
    }
    __device__ Response response(Vec3 direction) const noexcept {
        return {direction,
            contact_direction_inverse_mass(parameters[a], states[a], contact.body_hinge,
                contact.point, direction, compounds, a, !contact.persistent) +
            contact_direction_inverse_mass(parameters[b], states[b], contact.collider_hinge,
                contact.point, direction, compounds, b, !contact.persistent)};
    }
    __device__ void apply(const Response &response, float magnitude) const noexcept {
        const auto impulse = multiply(response.direction, magnitude);
        apply_contact_velocity_impulse(parameters, states, count, a,
            contact.body_hinge, contact.point, impulse, compounds);
        apply_contact_velocity_impulse(parameters, states, count, b,
            contact.collider_hinge, contact.point, multiply(impulse, -1.0F), compounds);
    }
    __device__ void apply_friction(Vec3 impulse) const noexcept {
        apply({impulse, 0.0F}, 1.0F);
    }
};

// Storage/response specialization, not a second solver: free pairs need only
// two private velocities and their prepared Jacobian. Keeping global world
// pointers out of this adapter lets the compiler retain those velocities in
// registers across the shared normal/friction row.
struct FreeContactAdapter {
    RigidBodyState &a, &b;
    float inverse_a, inverse_b;
    const ContactResponseRow &prepared;
    Vec3 normal;
    struct Response { Vec3 direction{}; float inverse_mass{}; Vec3 angular_a{}, angular_b{}; };

    __device__ Vec3 relative_velocity() const noexcept {
        return subtract(add(a.linear_velocity, cross(a.angular_velocity, prepared.arm_a)),
                        add(b.linear_velocity, cross(b.angular_velocity, prepared.arm_b)));
    }
    __device__ Response normal_response() const noexcept {
        return {normal, prepared.normal_inverse_mass, prepared.angular_a[0], prepared.angular_b[0]};
    }
    __device__ Response response(Vec3 direction) const noexcept {
        const float x = dot(direction, prepared.tangents[0]);
        const float y = dot(direction, prepared.tangents[1]);
        return {direction, prepared.tangent_xx * x * x + 2.0F * prepared.tangent_xy * x * y + prepared.tangent_yy * y * y,
            add(multiply(prepared.angular_a[1], x), multiply(prepared.angular_a[2], y)),
            add(multiply(prepared.angular_b[1], x), multiply(prepared.angular_b[2], y))};
    }
    __device__ void apply(const Response &response, float magnitude) const noexcept {
        const auto impulse = multiply(response.direction, magnitude);
        if (inverse_a > 0.0F) {
            a.linear_velocity = add(a.linear_velocity, multiply(impulse, inverse_a));
            a.angular_velocity = add(a.angular_velocity, multiply(response.angular_a, magnitude));
        }
        if (inverse_b > 0.0F) {
            b.linear_velocity = add(b.linear_velocity, multiply(impulse, -inverse_b));
            b.angular_velocity = add(b.angular_velocity, multiply(response.angular_b, -magnitude));
        }
    }
    __device__ void apply_friction(Vec3 impulse) const noexcept { apply(response(impulse), 1.0F); }
};

__device__ __forceinline__ AppliedContactImpulse apply_contact_impulse(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, std::uint32_t body_index,
    std::uint32_t collider_index, Contact &contact, float timestep,
    const RigidCompound *compounds, float normal_inverse_mass,
    const solver::ContactMaterial &material) noexcept {
    RigidContactAdapter adapter{parameters, states, count, body_index, collider_index,
        contact, compounds, normal_inverse_mass};
    const auto applied = solver::solve_contact_velocity(contact, material, timestep, adapter);
    return {applied.normal, applied.friction};
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
        if (!hinge.fixed && !hinge.axial) return parameters.inverse_mass;
        const float jacobian = dot(
            cross(hinge.axis, subtract(contact.point, hinge.anchor)),
            contact.normal);
        const float axial = hinge.axial ? dot(hinge.axis, contact.normal) : 0.0F;
        return parameters.inverse_mass * axial * axial + jacobian * jacobian *
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
        if (!hinge.fixed && !hinge.axial) {
            body_state.position = add(
                body_state.position,
                multiply(body_correction, parameters.inverse_mass));
            return;
        }
        const float angular_correction = dot(
            hinge.axis,
            cross(subtract(contact.point, hinge.anchor), body_correction)) *
            fixed_hinge_inverse_moment(parameters, body_state, hinge);
        const Vec3 current_anchor = add(body_state.position,
            rotate(body_state.orientation, hinge.local_anchor));
        const Vec3 anchor = hinge.axial
            ? add(current_anchor, multiply(hinge.axis,
                  parameters.inverse_mass * dot(body_correction, hinge.axis)))
            : hinge.anchor;
        apply_orientation_correction(
            body_state, multiply(hinge.axis, angular_correction));
        body_state.position = subtract(
            anchor,
            rotate(body_state.orientation, hinge.local_anchor));
    };
    apply(body, state, contact.body_hinge, correction);
    apply(collider, collider_state, contact.collider_hinge,
          multiply(correction, -1.0F));
}

template<bool FreePair = false>
__device__ __forceinline__ float resolve_contact_rows(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t body_count, std::uint32_t body_index,
    std::uint32_t collider_index,
    Contact *contacts, std::uint32_t contact_count,
    Vec3 initial_relative_position,
    float timestep, bool correct_position, RigidContactEvent *debug_events,
    std::uint32_t debug_event_count,
    const RigidCompound *compounds, bool warm_start_only,
    ContactResponsePatch *response,
    RigidBodyState &state, RigidBodyState &collider_state) noexcept {
    if (contact_count == 0U) {
        return 0.0F;
    }
    const BodyParameters &body = parameters[body_index];
    const BodyParameters &collider = parameters[collider_index];
    for (std::uint32_t point = 0U; point < contact_count; ++point) {
        Contact &contact = contacts[point];
        if (!contact.persistent || contact.warm_started) continue;
        contact.warm_started = true;
        const Vec3 impulse = add(multiply(contact.normal, contact.accumulated_normal_impulse),
                                 contact.accumulated_friction_impulse);
        apply_contact_velocity_impulse(parameters, states, body_count, body_index,
            contact.body_hinge, contact.point, impulse, compounds, &state);
        apply_contact_velocity_impulse(parameters, states, body_count, collider_index,
            contact.collider_hinge, contact.point, multiply(impulse, -1.0F), compounds, &collider_state);
    }
    if (warm_start_only) return 0.0F;
    const float inverse_mass_sum = body.inverse_mass + collider.inverse_mass;
    const bool guided = guided_static_contact(contacts[0]);
    const bool translational_projection = contacts[0].persistent && !guided;
    if (!guided && (correct_position || translational_projection) &&
        inverse_mass_sum > k_epsilon) {
        std::uint32_t correction_count = contact_count;
        if (!translational_projection) {
            correction_count = 0U;
            for (std::uint32_t index = 0U; index < contact_count; ++index)
                if (contacts[index].penetration > 0.0F &&
                    !contacts[index].body_hinge.fixed_member &&
                    !contacts[index].collider_hinge.fixed_member) ++correction_count;
        }
        const float contact_weight = correction_count > 0U
            ? 1.0F / static_cast<float>(correction_count) : 0.0F;
        for (std::uint32_t index = 0; index < contact_count; ++index) {
            if (contacts[index].body_hinge.fixed_member ||
                contacts[index].collider_hinge.fixed_member) {
                continue;
            }
            const bool hinged = contacts[index].body_hinge.fixed ||
                                contacts[index].collider_hinge.fixed;
            const float penetration = contacts[index].penetration -
                (!contacts[index].persistent ? 0.0F : dot(subtract(
                    subtract(state.position, collider_state.position),
                    initial_relative_position), contacts[index].normal));
            if (penetration <= 0.0F) continue;
            apply_contact_position_correction(
                body, state, collider, collider_state, contacts[index],
                (fminf(penetration,
                       hinged ? k_rigid_rest_offset : penetration) +
                 (contacts[index].persistent ? 0.0F : k_rigid_surface_tolerance)) *
                    contact_weight);
        }
    }
    // Position/joint projection can change lever arms and inertia. Refresh
    // after projection, never freeze a Jacobian at an earlier pose.
    const bool unchanged_pose = response != nullptr && contact_response_matches(*response, state, collider_state);
    const bool reuse_response = unchanged_pose && (!FreePair || response->rows_ready);
    const bool retain_response = !FreePair || unchanged_pose;
    if (response != nullptr && !reuse_response)
        prepare_contact_response<FreePair>(parameters, state, collider_state, body_index, collider_index,
            contacts, contact_count, compounds, *response);
    const auto material = response != nullptr ? response->material
        : prepare_contact_material(parameters, body_index, collider_index, contacts[0]);
    float impulse_change = 0.0F;
    for (std::uint32_t index = 0; index < contact_count; ++index) {
        // Keep every row private across normal and friction solving, including
        // triangle contacts. Only stable features retain their impulse history.
        auto row = [&]() {
            if constexpr (FreePair) {
                const auto &contact = contacts[index];
                return solver::ContactRow<Vec3>{contact.normal, contact.penetration, true,
                    contact.initial_normal_speed, contact.accumulated_normal_impulse,
                    contact.accumulated_friction_impulse};
            } else return Contact{contacts[index]};
        }();
        // Bounded-cache overflow needs one row of scratch, not a full patch
        // on every thread's stack. Cached and uncached rows use identical math.
        ContactResponseRow local_response{};
        if (reuse_response) {
            if constexpr (FreePair) local_response = response->rows[index];
            else local_response.normal_inverse_mass = response->normal_inverse_mass[index];
        }
        if (!reuse_response) {
            if constexpr (FreePair)
                prepare_contact_response_row<true>(parameters, state, collider_state,
                    body_index, collider_index, contacts[index], compounds, local_response, response);
            else prepare_contact_response_row(parameters, state, collider_state,
                body_index, collider_index, row, compounds, local_response, response);
        }
        AppliedContactImpulse applied{};
        if constexpr (FreePair) {
            FreeContactAdapter adapter{state, collider_state, body.inverse_mass,
                collider.inverse_mass, local_response, row.normal};
            const auto impulse = solver::solve_contact_velocity(row, material, timestep, adapter);
            applied = {impulse.normal, impulse.friction};
        } else {
            applied = apply_contact_impulse(parameters, states, body_count,
                body_index, collider_index, row, timestep, compounds,
                local_response.normal_inverse_mass, material);
        }
        if (response != nullptr && retain_response && !reuse_response) {
            if constexpr (FreePair) response->rows[index] = local_response;
            else response->normal_inverse_mass[index] = local_response.normal_inverse_mass;
        }
        const float change = fmaxf(fabsf(applied.normal), fmaxf(fabsf(applied.friction.x),
            fmaxf(fabsf(applied.friction.y), fabsf(applied.friction.z))));
        impulse_change = isfinite(change) ? fmaxf(impulse_change, change) : __int_as_float(0x7f800000);
        if (row.persistent) {
            contacts[index].accumulated_normal_impulse = row.accumulated_normal_impulse;
            contacts[index].accumulated_friction_impulse = row.accumulated_friction_impulse;
        }
        if (!row.persistent && debug_events != nullptr && index < debug_event_count) {
            contacts[index].reported_normal_impulse += applied.normal;
            contacts[index].reported_friction_impulse =
                add(contacts[index].reported_friction_impulse, applied.friction);
        }
    }
    // Admit full Jacobian rows only after the same pose is observed twice.
    // Moving projections still use freshly prepared rows, but do not stream
    // patches to global memory that the next color immediately invalidates.
    if (response != nullptr) response->rows_ready = retain_response;
    return impulse_change;
}

__device__ __noinline__ float resolve_free_contact_patch(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t body_count, std::uint32_t body_index,
    std::uint32_t collider_index, Contact *contacts, std::uint32_t contact_count,
    Vec3 initial_relative_position, float timestep, bool correct_position,
    RigidContactEvent *debug_events, std::uint32_t debug_event_count,
    bool warm_start_only, ContactResponsePatch *response) noexcept {
    RigidBodyState a = states[body_index], b = states[collider_index];
    const float change = resolve_contact_rows<true>(parameters, states, body_count, body_index, collider_index,
        contacts, contact_count, initial_relative_position, timestep, correct_position,
        debug_events, debug_event_count, nullptr, warm_start_only, response, a, b);
    if (parameters[body_index].inverse_mass > 0.0F) states[body_index] = a;
    if (parameters[collider_index].inverse_mass > 0.0F) states[collider_index] = b;
    return change;
}

__device__ __forceinline__ float resolve_contacts(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t body_count, std::uint32_t body_index,
    std::uint32_t collider_index, Contact *contacts, std::uint32_t contact_count,
    Vec3 initial_relative_position, float timestep, bool correct_position,
    RigidContactEvent *debug_events, std::uint32_t debug_event_count,
    const RigidCompound *compounds, bool warm_start_only,
    ContactResponsePatch *response) noexcept {
    if (contact_count == 0U) return 0.0F;
    if (contacts[0].persistent && (compounds == nullptr ||
        (!compounds[body_index].eligible && !compounds[collider_index].eligible))) {
        // Colored pairs own their dynamic bodies. Keep the two states local
        // across a patch's rows instead of round-tripping every impulse through
        // global memory. Static/kinematic colliders are shared read-only.
        // Transient rows retain their established global-state store order;
        // localizing them changes floating-point contraction in coupled scenes.
        // This is storage policy: both branches call the same row implementation.
        const auto &first = contacts[0];
        if (!first.body_hinge.fixed && !first.body_hinge.axial &&
            !first.collider_hinge.fixed && !first.collider_hinge.axial) {
            return resolve_free_contact_patch(parameters, states, body_count, body_index, collider_index,
                contacts, contact_count, initial_relative_position, timestep,
                correct_position, debug_events, debug_event_count, warm_start_only, response);
        }
        const BodyParameters pair_parameters[]{parameters[body_index], parameters[collider_index]};
        RigidBodyState pair_states[]{states[body_index], states[collider_index]};
        const float impulse_change = resolve_contact_rows(pair_parameters, pair_states, 2U, 0U, 1U,
            contacts, contact_count, initial_relative_position, timestep,
            correct_position, debug_events, debug_event_count, nullptr, warm_start_only, response, pair_states[0], pair_states[1]);
        if (parameters[body_index].inverse_mass > 0.0F) states[body_index] = pair_states[0];
        if (parameters[collider_index].inverse_mass > 0.0F) states[collider_index] = pair_states[1];
        return impulse_change;
    } else {
        return resolve_contact_rows(parameters, states, body_count, body_index, collider_index,
            contacts, contact_count, initial_relative_position, timestep,
            correct_position, debug_events, debug_event_count, compounds, warm_start_only, response,
            states[body_index], states[collider_index]);
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

__device__ HingeContactFrame rigid_hinge_contact_frame(
    const BodyParameters *, const RigidBodyId *, std::uint32_t, std::uint32_t,
    const RigidBodyState *, const RigidBodyState &,
    const RigidConstraintResource *, std::uint32_t, Vec3 &,
    Quaternion * = nullptr) noexcept;

__device__ void advance_guided_pose(
    RigidBodyState &next, const RigidBodyState &previous,
    const HingeContactFrame &frame, Quaternion axial_orientation,
    float timestep) noexcept {
    next.position = add(previous.position, multiply(next.linear_velocity, timestep));
    const Quaternion angular{next.angular_velocity.x, next.angular_velocity.y,
                             next.angular_velocity.z, 0.0F};
    const Quaternion derivative = quaternion_multiply(angular, previous.orientation);
    next.orientation = normalized_quaternion({
        previous.orientation.x + 0.5F * derivative.x * timestep,
        previous.orientation.y + 0.5F * derivative.y * timestep,
        previous.orientation.z + 0.5F * derivative.z * timestep,
        previous.orientation.w + 0.5F * derivative.w * timestep});
    if (frame.axial_rotation) {
        const Quaternion relative = quaternion_multiply(next.orientation, conjugate(axial_orientation));
        const float twist = dot({relative.x, relative.y, relative.z}, frame.axis);
        const Quaternion rotation = normalized_quaternion({
            frame.axis.x * twist, frame.axis.y * twist, frame.axis.z * twist, relative.w});
        next.orientation = normalized_quaternion(quaternion_multiply(rotation, axial_orientation));
    } else {
        next.orientation = axial_orientation;
    }
    const Vec3 anchor = add(next.position, rotate(next.orientation, frame.local_anchor));
    next.position = subtract(add(frame.anchor,
        multiply(frame.axis, dot(subtract(anchor, frame.anchor), frame.axis))),
        rotate(next.orientation, frame.local_anchor));
}

__global__ void finalize_guided_bodies_kernel(
    const BodyParameters *parameters, const RigidBodyId *ids,
    const RigidBodyState *previous, RigidBodyState *states, std::uint32_t count,
    const RigidConstraintResource *constraints, std::uint32_t capacity, float timestep,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || parameters[index].motion != MotionType::dynamic) return;
    Vec3 reference{};
    Quaternion orientation{};
    const auto frame = rigid_hinge_contact_frame(parameters, ids, count, index,
        previous, previous[index], constraints, capacity, reference, &orientation);
    if (frame.axial) {
        // The pose used for contact detection is a predictor. Advance the
        // actual guide coordinates with the constrained velocity. Independent
        // positional pushes can rotate a tooth into another face that was
        // clear in the predictor, injecting potential energy during recovery.
        float impact = 1.0F;
        for (std::uint32_t active = 0; active < *active_pair_count; ++active) {
            const auto pair = active_pairs[active];
            if (pair / count != index && pair % count != index) continue;
            const auto &manifold = manifolds[active];
            for (std::uint32_t row = 0; row < manifold.count; ++row) {
                const auto &contact = manifold.contacts[row];
                if (contact.accumulated_normal_impulse > k_epsilon)
                    impact = fminf(impact, contact.impact_fraction);
            }
        }
        if (impact < 1.0F) {
            RigidBodyState hit = previous[index];
            hit.position = add(hit.position, multiply(
                subtract(states[index].position, hit.position), impact));
            const auto a = hit.orientation;
            auto b = states[index].orientation;
            if (a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w < 0.0F)
                b = {-b.x, -b.y, -b.z, -b.w};
            hit.orientation = normalized_quaternion({
                a.x + (b.x-a.x)*impact, a.y + (b.y-a.y)*impact,
                a.z + (b.z-a.z)*impact, a.w + (b.w-a.w)*impact});
            advance_guided_pose(states[index], hit, frame, orientation,
                                timestep * (1.0F-impact));
        }
    }
}

__global__ void integrate_rigid_bodies_kernel(
    const BodyParameters *parameters, const BodyAccumulator *accumulators,
    const KinematicTarget *targets, const RigidBodyState *input,
    RigidBodyState *output, std::uint32_t count, Vec3 gravity, float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses,
    const RigidBodyId *ids, const RigidConstraintResource *constraints,
    std::uint32_t constraint_capacity) {
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
    Vec3 reference{};
    Quaternion axial_orientation{};
    const HingeContactFrame frame = rigid_hinge_contact_frame(
        parameters, ids, count, index, input, previous, constraints,
        constraint_capacity, reference, &axial_orientation);
    if (frame.axial) {
        // Project momentum before predicting contact geometry. Otherwise a
        // supported piston first falls sideways into its teeth, and removing
        // that forbidden motion after collision leaves spurious axial kicks.
        const Vec3 local_omega = inverse_rotate(next.orientation, next.angular_velocity);
        const Vec3 angular_momentum = rotate(next.orientation, {
            local_omega.x / fmaxf(body.inverse_inertia_local.x, k_epsilon),
            local_omega.y / fmaxf(body.inverse_inertia_local.y, k_epsilon),
            local_omega.z / fmaxf(body.inverse_inertia_local.z, k_epsilon)});
        const Vec3 arm = subtract(next.position, frame.anchor);
        const float spin = dot(frame.axis, add(angular_momentum,
            cross(arm, multiply(next.linear_velocity, 1.0F / body.inverse_mass)))) *
            fixed_hinge_inverse_moment(body, next, frame);
        next.angular_velocity = multiply(frame.axis, spin);
        next.linear_velocity = add(
            multiply(frame.axis, dot(next.linear_velocity, frame.axis)),
            cross(next.angular_velocity, arm));
    }
    next.position = add(next.position, multiply(next.linear_velocity, timestep));

    const Quaternion angular{next.angular_velocity.x, next.angular_velocity.y,
                             next.angular_velocity.z, 0.0F};
    const Quaternion derivative = quaternion_multiply(angular, next.orientation);
    next.orientation = normalized_quaternion(
        {next.orientation.x + 0.5F * derivative.x * timestep,
         next.orientation.y + 0.5F * derivative.y * timestep,
         next.orientation.z + 0.5F * derivative.z * timestep,
         next.orientation.w + 0.5F * derivative.w * timestep});
    if (frame.axial) {
        if (frame.axial_rotation) {
            const Quaternion relative = quaternion_multiply(
                next.orientation, conjugate(axial_orientation));
            const float twist = dot({relative.x, relative.y, relative.z}, frame.axis);
            const Quaternion rotation = normalized_quaternion({
                frame.axis.x * twist, frame.axis.y * twist,
                frame.axis.z * twist, relative.w});
            next.orientation = normalized_quaternion(
                quaternion_multiply(rotation, axial_orientation));
        } else {
            next.orientation = axial_orientation;
        }
        // Rotation follows an arc, not the Euler chord. Keep the body's
        // joint anchor on the guide without changing its free axial travel.
        const Vec3 anchor = add(next.position, rotate(next.orientation, frame.local_anchor));
        next.position = subtract(add(frame.anchor,
            multiply(frame.axis, dot(subtract(anchor, frame.anchor), frame.axis))),
            rotate(next.orientation, frame.local_anchor));
    }
    output[index] = next;
}

__device__ std::uint32_t find_rigid_body_dense(
    RigidBodyId id, const RigidBodyId *ids, std::uint32_t count) noexcept {
    for (std::uint32_t index = 0U; index < count; ++index)
        if (ids[index].index == id.index &&
            ids[index].generation == id.generation) return index;
    return k_invalid_dense;
}

__device__ std::uint32_t rigid_compound_root(
    const RigidCompound *compounds, std::uint32_t body) noexcept {
    while (compounds[body].root != body) body = compounds[body].root;
    return body;
}

struct SymmetricMatrix3 {
    float xx{}, xy{}, xz{}, yy{}, yz{}, zz{};
};

__device__ void add_inertia_axis(
    SymmetricMatrix3 &matrix, Vec3 axis, float moment) noexcept {
    matrix.xx += moment * axis.x * axis.x;
    matrix.xy += moment * axis.x * axis.y;
    matrix.xz += moment * axis.x * axis.z;
    matrix.yy += moment * axis.y * axis.y;
    matrix.yz += moment * axis.y * axis.z;
    matrix.zz += moment * axis.z * axis.z;
}

__device__ Vec3 multiply_symmetric(
    const SymmetricMatrix3 &matrix, Vec3 value) noexcept {
    return {matrix.xx * value.x + matrix.xy * value.y + matrix.xz * value.z,
            matrix.xy * value.x + matrix.yy * value.y + matrix.yz * value.z,
            matrix.xz * value.x + matrix.yz * value.y + matrix.zz * value.z};
}

__device__ bool invert_symmetric(
    const SymmetricMatrix3 &matrix, Vec3 (&inverse)[3]) noexcept {
    const float c00 = matrix.yy * matrix.zz - matrix.yz * matrix.yz;
    const float c01 = matrix.xz * matrix.yz - matrix.xy * matrix.zz;
    const float c02 = matrix.xy * matrix.yz - matrix.xz * matrix.yy;
    const float c11 = matrix.xx * matrix.zz - matrix.xz * matrix.xz;
    const float c12 = matrix.xy * matrix.xz - matrix.xx * matrix.yz;
    const float c22 = matrix.xx * matrix.yy - matrix.xy * matrix.xy;
    const float determinant =
        matrix.xx * c00 + matrix.xy * c01 + matrix.xz * c02;
    if (!isfinite(determinant) || fabsf(determinant) <= k_epsilon) return false;
    const float scale = 1.0F / determinant;
    inverse[0] = {c00 * scale, c01 * scale, c02 * scale};
    inverse[1] = {c01 * scale, c11 * scale, c12 * scale};
    inverse[2] = {c02 * scale, c12 * scale, c22 * scale};
    return true;
}

__device__ bool compound_fixed_edge(
    const RigidConstraintResource &constraint,
    const BodyParameters *parameters, std::uint32_t a,
    std::uint32_t b) noexcept {
    return constraint.options.type == RigidConstraintType::fixed &&
        constraint.options.disable_collisions &&
        constraint.options.breaking_impulse_threshold <= 0.0F &&
        parameters[a].motion == MotionType::dynamic &&
        parameters[b].motion == MotionType::dynamic;
}

// Rebuild after integration so newly attached/released bodies take effect on
// the next substep. Logical bodies remain collision surfaces; eligible welded
// components share one aggregate mass, inertia, and rigid twist.
__global__ void build_rigid_compounds_kernel(
    const RigidConstraintResource *constraints, std::uint32_t capacity,
    const RigidBodyId *ids, const BodyParameters *parameters,
    RigidBodyState *states, std::uint32_t count,
    RigidCompound *compounds) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    for (std::uint32_t body = 0U; body < count; ++body)
        compounds[body] = {.root = body};

    for (std::uint32_t index = 0U; index < capacity; ++index) {
        const auto &constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken) continue;
        const auto a = find_rigid_body_dense(
            constraint.options.body_a, ids, count);
        const auto b = find_rigid_body_dense(
            constraint.options.body_b, ids, count);
        if (a == k_invalid_dense || b == k_invalid_dense ||
            !compound_fixed_edge(constraint, parameters, a, b)) continue;
        const auto root_a = rigid_compound_root(compounds, a);
        const auto root_b = rigid_compound_root(compounds, b);
        if (root_a != root_b)
            compounds[root_a > root_b ? root_a : root_b].root =
                root_a < root_b ? root_a : root_b;
    }
    for (std::uint32_t body = 0U; body < count; ++body)
        compounds[body].root = rigid_compound_root(compounds, body);

    // Any incident joint needing general solver semantics keeps its whole
    // fixed component on the general path.
    for (std::uint32_t index = 0U; index < capacity; ++index) {
        const auto &constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken) continue;
        const auto a = find_rigid_body_dense(
            constraint.options.body_a, ids, count);
        const auto b = find_rigid_body_dense(
            constraint.options.body_b, ids, count);
        if (a == k_invalid_dense || b == k_invalid_dense ||
            compound_fixed_edge(constraint, parameters, a, b)) continue;
        compounds[compounds[a].root].blocked = true;
        compounds[compounds[b].root].blocked = true;
    }
    for (std::uint32_t body = 0U; body < count; ++body)
        ++compounds[compounds[body].root].member_count;

    for (std::uint32_t root = 0U; root < count; ++root) {
        RigidCompound &compound = compounds[root];
        if (compound.root != root || compound.member_count < 2U ||
            compound.blocked) continue;
        // Advance one member pose, then rebuild every child transform from
        // fixed-joint frames. This removes integration drift while retaining
        // each member as an independent collision surface and public handle.
        compounds[root].eligible = true;
        for (std::uint32_t pass = 1U; pass < compound.member_count; ++pass) {
            bool changed = false;
            for (std::uint32_t index = 0U; index < capacity; ++index) {
                const auto &constraint = constraints[index];
                if (!constraint.alive || !constraint.options.enabled ||
                    constraint.state.broken) continue;
                const auto a = find_rigid_body_dense(
                    constraint.options.body_a, ids, count);
                const auto b = find_rigid_body_dense(
                    constraint.options.body_b, ids, count);
                if (a == k_invalid_dense || b == k_invalid_dense ||
                    compounds[a].root != root ||
                    compounds[b].root != root ||
                    !compound_fixed_edge(constraint, parameters, a, b) ||
                    compounds[a].eligible == compounds[b].eligible) continue;
                const RigidConstraintOptions &options = constraint.options;
                if (compounds[a].eligible) {
                    states[b].orientation = normalized_quaternion(
                        quaternion_multiply(
                            quaternion_multiply(states[a].orientation,
                                                options.local_orientation_a),
                            conjugate(options.local_orientation_b)));
                    states[b].position = subtract(
                        add(states[a].position,
                            rotate(states[a].orientation,
                                   options.local_anchor_a)),
                        rotate(states[b].orientation,
                               options.local_anchor_b));
                    compounds[b].eligible = true;
                } else {
                    states[a].orientation = normalized_quaternion(
                        quaternion_multiply(
                            quaternion_multiply(states[b].orientation,
                                                options.local_orientation_b),
                            conjugate(options.local_orientation_a)));
                    states[a].position = subtract(
                        add(states[b].position,
                            rotate(states[b].orientation,
                                   options.local_anchor_b)),
                        rotate(states[a].orientation,
                               options.local_anchor_a));
                    compounds[a].eligible = true;
                }
                changed = true;
            }
            if (!changed) break;
        }
        bool connected = true;
        for (std::uint32_t member = 0U; member < count; ++member)
            if (compounds[member].root == root &&
                !compounds[member].eligible) connected = false;
        if (!connected) {
            for (std::uint32_t member = 0U; member < count; ++member)
                if (compounds[member].root == root)
                    compounds[member].eligible = false;
            continue;
        }
        float mass_sum = 0.0F;
        Vec3 weighted_center{};
        Vec3 linear_momentum{};
        for (std::uint32_t member = 0U; member < count; ++member) {
            if (compounds[member].root != root) continue;
            const float inverse_mass = parameters[member].inverse_mass;
            if (inverse_mass <= k_epsilon) {
                compound.blocked = true;
                break;
            }
            const float mass = 1.0F / inverse_mass;
            mass_sum += mass;
            weighted_center = add(
                weighted_center, multiply(states[member].position, mass));
            linear_momentum = add(
                linear_momentum,
                multiply(states[member].linear_velocity, mass));
        }
        if (compound.blocked || mass_sum <= k_epsilon) {
            for (std::uint32_t member = 0U; member < count; ++member)
                if (compounds[member].root == root)
                    compounds[member].eligible = false;
            continue;
        }
        compound.center = multiply(weighted_center, 1.0F / mass_sum);

        SymmetricMatrix3 inertia{};
        Vec3 angular_momentum{};
        for (std::uint32_t member = 0U; member < count; ++member) {
            if (compounds[member].root != root) continue;
            const BodyParameters &body = parameters[member];
            const RigidBodyState &state = states[member];
            const float mass = 1.0F / body.inverse_mass;
            const Vec3 local_moment{
                1.0F / fmaxf(body.inverse_inertia_local.x, k_epsilon),
                1.0F / fmaxf(body.inverse_inertia_local.y, k_epsilon),
                1.0F / fmaxf(body.inverse_inertia_local.z, k_epsilon)};
            SymmetricMatrix3 member_inertia{};
            add_inertia_axis(member_inertia,
                             rotate(state.orientation, {1.0F, 0.0F, 0.0F}),
                             local_moment.x);
            add_inertia_axis(member_inertia,
                             rotate(state.orientation, {0.0F, 1.0F, 0.0F}),
                             local_moment.y);
            add_inertia_axis(member_inertia,
                             rotate(state.orientation, {0.0F, 0.0F, 1.0F}),
                             local_moment.z);
            inertia.xx += member_inertia.xx;
            inertia.xy += member_inertia.xy;
            inertia.xz += member_inertia.xz;
            inertia.yy += member_inertia.yy;
            inertia.yz += member_inertia.yz;
            inertia.zz += member_inertia.zz;
            const Vec3 arm = subtract(state.position, compound.center);
            const float radius_squared = dot(arm, arm);
            inertia.xx += mass * (radius_squared - arm.x * arm.x);
            inertia.xy -= mass * arm.x * arm.y;
            inertia.xz -= mass * arm.x * arm.z;
            inertia.yy += mass * (radius_squared - arm.y * arm.y);
            inertia.yz -= mass * arm.y * arm.z;
            inertia.zz += mass * (radius_squared - arm.z * arm.z);
            angular_momentum = add(
                angular_momentum,
                add(multiply_symmetric(member_inertia,
                                       state.angular_velocity),
                    cross(arm,
                          multiply(state.linear_velocity, mass))));
        }
        if (!invert_symmetric(inertia, compound.inverse_inertia)) {
            for (std::uint32_t member = 0U; member < count; ++member)
                if (compounds[member].root == root)
                    compounds[member].eligible = false;
            continue;
        }
        compound.inverse_mass = 1.0F / mass_sum;
        compound.eligible = true;
        const Vec3 linear_velocity =
            multiply(linear_momentum, compound.inverse_mass);
        const Vec3 angular_velocity =
            compound_inverse_inertia_world(compound, angular_momentum);
        for (std::uint32_t member = 0U; member < count; ++member) {
            if (compounds[member].root != root) continue;
            compounds[member].eligible = true;
            states[member].angular_velocity = angular_velocity;
            states[member].linear_velocity = add(
                linear_velocity,
                cross(angular_velocity,
                      subtract(states[member].position, compound.center)));
        }
    }
}

__device__ float solve_linear_constraint_axis(
    const BodyParameters &a, RigidBodyState &state_a,
    const BodyParameters &b, RigidBodyState &state_b,
    Vec3 arm_a, Vec3 arm_b, const RigidConstraintAxisGeometry &geometry,
    float error, float timestep,
    float stiffness, float damping, bool spring) noexcept {
    const Vec3 axis = geometry.axis;
    const Vec3 velocity_a = add(
        state_a.linear_velocity, cross(state_a.angular_velocity, arm_a));
    const Vec3 velocity_b = add(
        state_b.linear_velocity, cross(state_b.angular_velocity, arm_b));
    const float relative_velocity = dot(subtract(velocity_b, velocity_a), axis);
    const float denominator = geometry.linear_denominator;
    if (denominator <= k_epsilon) return 0.0F;
    const float spring_response = timestep * (damping + stiffness * timestep);
    const float impulse = spring
        ? -(spring_response * relative_velocity +
            stiffness * error * timestep) /
              (1.0F + spring_response * denominator)
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
    const RigidConstraintAxisGeometry &geometry,
    float error, float timestep, float stiffness, float damping,
    bool spring) noexcept {
    const Vec3 axis = geometry.axis;
    const float relative_velocity = dot(
        subtract(state_b.angular_velocity, state_a.angular_velocity), axis);
    const Vec3 inverse_a = geometry.inverse_angular_a;
    const Vec3 inverse_b = geometry.inverse_angular_b;
    const float denominator = geometry.angular_denominator;
    if (denominator <= k_epsilon) return 0.0F;
    const float spring_response = timestep * (damping + stiffness * timestep);
    const float impulse = spring
        ? -(spring_response * relative_velocity +
            stiffness * error * timestep) /
              (1.0F + spring_response * denominator)
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
    const BodyParameters &b, RigidBodyState &state_b,
    const RigidConstraintAxisGeometry &geometry,
    Vec3 arm_a, Vec3 arm_b, float target_velocity, float maximum_impulse,
    bool angular) noexcept {
    const Vec3 axis = geometry.axis;
    float denominator = 0.0F;
    float relative_velocity = 0.0F;
    if (angular) {
        relative_velocity = dot(
            subtract(state_b.angular_velocity, state_a.angular_velocity), axis);
        denominator = geometry.angular_denominator;
    } else {
        const Vec3 velocity_a = add(
            state_a.linear_velocity, cross(state_a.angular_velocity, arm_a));
        const Vec3 velocity_b = add(
            state_b.linear_velocity, cross(state_b.angular_velocity, arm_b));
        relative_velocity = dot(subtract(velocity_b, velocity_a), axis);
        denominator = geometry.linear_denominator;
    }
    if (denominator <= k_epsilon || maximum_impulse <= 0.0F) return 0.0F;
    const float impulse = clamp_scalar(
        (target_velocity - relative_velocity) / denominator,
        -maximum_impulse, maximum_impulse);
    if (angular) {
        if (a.inverse_mass > 0.0F)
            state_a.angular_velocity = subtract(
                state_a.angular_velocity,
                multiply(geometry.inverse_angular_a, impulse));
        if (b.inverse_mass > 0.0F)
            state_b.angular_velocity = add(
                state_b.angular_velocity,
                multiply(geometry.inverse_angular_b, impulse));
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

__device__ __forceinline__ void resolve_active_rigid_contact_pair(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, std::uint32_t active_index,
    float timestep, bool correct_position,
    const RigidCompound *compounds, bool warm_start_only = false,
    ContactResponsePatch *response = nullptr, ContactSolveDelta *delta = nullptr);

__global__ void solve_rigid_constraints_kernel(
    RigidConstraintResource *constraints, std::uint32_t capacity,
    const RigidBodyId *ids, const BodyParameters *parameters,
    RigidBodyState *states, std::uint32_t body_count, float timestep,
    ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, const RigidCompound *compounds,
    ContactResponsePatch *responses, std::uint32_t response_capacity) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    bool iterated_contacts = false;
    std::uint32_t iterations = 0U;
    for (std::uint32_t index = 0U; index < capacity; ++index) {
        RigidConstraintResource &constraint = constraints[index];
        if (!constraint.alive) continue;
        // Preserve the impulse that broke a constraint for later diagnostics.
        if (!constraint.state.broken)
            constraint.state.applied_impulse = 0.0F;
        constraint.state.enabled = constraint.options.enabled && !constraint.state.broken;
        if (constraint.state.enabled) {
            const RigidConstraintOptions &options = constraint.options;
            RigidConstraintGeometry &geometry = constraint.geometry;
            geometry.dense_a = find_rigid_body_dense(options.body_a, ids, body_count);
            geometry.dense_b = find_rigid_body_dense(options.body_b, ids, body_count);
            if (geometry.dense_a == k_invalid_dense ||
                geometry.dense_b == k_invalid_dense) continue;
            const bool absorbed =
                compounds != nullptr &&
                options.type == RigidConstraintType::fixed &&
                options.disable_collisions &&
                options.breaking_impulse_threshold <= 0.0F &&
                compounds[geometry.dense_a].eligible &&
                compounds[geometry.dense_b].eligible &&
                compounds[geometry.dense_a].root ==
                    compounds[geometry.dense_b].root;
            if (absorbed) {
                geometry.dense_a = k_invalid_dense;
                geometry.dense_b = k_invalid_dense;
                continue;
            }
            iterations = constraint.options.solver_iterations > iterations
                ? constraint.options.solver_iterations : iterations;
            iterated_contacts |=
                options.type == RigidConstraintType::fixed ||
                options.type == RigidConstraintType::point;
            const BodyParameters &a = parameters[geometry.dense_a];
            const BodyParameters &b = parameters[geometry.dense_b];
            const RigidBodyState &state_a = states[geometry.dense_a];
            const RigidBodyState &state_b = states[geometry.dense_b];
            geometry.arm_a = rotate(state_a.orientation, options.local_anchor_a);
            geometry.arm_b = rotate(state_b.orientation, options.local_anchor_b);
            geometry.anchor_error = subtract(
                add(state_b.position, geometry.arm_b),
                add(state_a.position, geometry.arm_a));
            const Quaternion frame_a = normalized_quaternion(
                quaternion_multiply(state_a.orientation, options.local_orientation_a));
            const Quaternion frame_b = normalized_quaternion(
                quaternion_multiply(state_b.orientation, options.local_orientation_b));
            geometry.rotation_error = relative_rotation_vector(frame_a, frame_b);
            geometry.hinge_alignment_error = cross(
                rotate(frame_a, {0.0F, 0.0F, 1.0F}),
                rotate(frame_b, {0.0F, 0.0F, 1.0F}));
            // Locked swing must not include the permitted X twist. The full
            // quaternion logarithm changes branch at a half-turn and couples
            // that free twist into Y/Z correction, amplifying tiny tilts.
            geometry.piston_alignment_error = cross(
                rotate(frame_a, {1.0F, 0.0F, 0.0F}),
                rotate(frame_b, {1.0F, 0.0F, 0.0F}));
            for (std::uint32_t axis_index = 0U; axis_index < 3U; ++axis_index) {
                RigidConstraintAxisGeometry &row = geometry.axes[axis_index];
                row.axis = rotate(frame_a, basis_axis(axis_index));
                row.inverse_angular_a = inverse_inertia_world(a, state_a, row.axis);
                row.inverse_angular_b = inverse_inertia_world(b, state_b, row.axis);
                row.angular_denominator = dot(
                    add(row.inverse_angular_a, row.inverse_angular_b), row.axis);
                const Vec3 angular_a = cross(
                    inverse_inertia_world(a, state_a, cross(geometry.arm_a, row.axis)),
                    geometry.arm_a);
                const Vec3 angular_b = cross(
                    inverse_inertia_world(b, state_b, cross(geometry.arm_b, row.axis)),
                    geometry.arm_b);
                row.linear_denominator = a.inverse_mass + b.inverse_mass +
                    dot(add(angular_a, angular_b), row.axis);
            }
        }
    }
    // General fixed and point joints must converge with contacts. Otherwise
    // the final joint impulse can restore inward velocity after the ordinary
    // contact pass and drive a constrained body through a support surface.
    // Compound-fixed contacts already use aggregate mass/inertia in the
    // contact solver.
    const std::uint32_t contact_sweeps = iterated_contacts ? 8U : 1U;
    for (std::uint32_t iteration = 0U;
         iteration < iterations * contact_sweeps; ++iteration) {
        // Contact and weld impulses must converge together. Solving all floor
        // contacts before the joints lets a heavy parent pull its light ground
        // supports downward again, discarding their support impulse each step.
        for (std::uint32_t active = 0U; iterated_contacts && active < *active_pair_count; ++active) {
            const ContactManifold &manifold = manifolds[active];
            if (manifold.count == 0U ||
                (!manifold.contacts[0].body_hinge.fixed_member &&
                 !manifold.contacts[0].collider_hinge.fixed_member &&
                 !manifold.contacts[0].body_hinge.point_member &&
                 !manifold.contacts[0].collider_hinge.point_member)) continue;
            const std::uint32_t pair = active_pairs[active];
            const std::uint32_t body = pair / body_count;
            const std::uint32_t collider = pair % body_count;
            if (compounds != nullptr &&
                (compounds[body].eligible ||
                 compounds[collider].eligible)) continue;
            resolve_active_rigid_contact_pair(
                parameters, states, body_count, manifolds, active_pairs,
                event_offsets, events, event_capacity, active, timestep, false,
                compounds, false,
                manifold.response_slot < response_capacity ? responses + manifold.response_slot : nullptr);
        }
        for (std::uint32_t index = 0U; index < capacity; ++index) {
            RigidConstraintResource &constraint = constraints[index];
            RigidConstraintOptions &options = constraint.options;
            if (!constraint.alive || !constraint.state.enabled ||
                iteration >= options.solver_iterations * contact_sweeps) continue;
            const RigidConstraintGeometry &geometry = constraint.geometry;
            const std::uint32_t dense_a = geometry.dense_a;
            const std::uint32_t dense_b = geometry.dense_b;
            if (dense_a == k_invalid_dense || dense_b == k_invalid_dense) continue;
            const BodyParameters &a = parameters[dense_a];
            const BodyParameters &b = parameters[dense_b];
            // Work on disjoint local states so each axis does not force
            // alias-sensitive global reloads. Publish before the next
            // constraint to retain Gauss-Seidel ordering.
            RigidBodyState state_a = states[dense_a];
            RigidBodyState state_b = states[dense_b];
            const Vec3 arm_a = geometry.arm_a;
            const Vec3 arm_b = geometry.arm_b;
            const Vec3 anchor_error = geometry.anchor_error;
            const Vec3 rotation_error = geometry.rotation_error;
            const Vec3 hinge_alignment_error = geometry.hinge_alignment_error;
            float applied = 0.0F;
            for (std::uint32_t axis_index = 0U; axis_index < 3U; ++axis_index) {
                const auto &row = geometry.axes[axis_index];
                const Vec3 world_axis = row.axis;
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
                const float linear_error = dot(anchor_error, world_axis);
                // A spring is a physical impulse for this substep, not a
                // convergence row. Reapplying it each solver iteration turns
                // authored damping into an effectively rigid constraint.
                if (linear_spring && iteration == 0U) {
                    applied += solve_linear_constraint_axis(
                        a, state_a, b, state_b, arm_a, arm_b, row,
                        linear_error, timestep,
                        component(options.linear_springs.stiffness, axis_index),
                        component(options.linear_springs.damping, axis_index),
                        true);
                }
                const float linear_limit_error = linear_limit
                    ? limit_error(
                          linear_error,
                          component(options.linear_limits.lower, axis_index),
                          component(options.linear_limits.upper, axis_index))
                    : 0.0F;
                if (linear_lock || linear_limit_error != 0.0F) {
                    applied += solve_linear_constraint_axis(
                        a, state_a, b, state_b, arm_a, arm_b, row,
                        linear_lock ? linear_error : linear_limit_error,
                        timestep, 0.0F, 0.0F, false);
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
                const float angular_error =
                    options.type == RigidConstraintType::hinge &&
                            axis_index != 2U
                        ? dot(hinge_alignment_error, world_axis)
                        : options.type == RigidConstraintType::piston &&
                                  axis_index != 0U
                            ? dot(geometry.piston_alignment_error, world_axis)
                            : component(rotation_error, axis_index);
                if (angular_spring && iteration == 0U) {
                    applied += solve_angular_constraint_axis(
                        a, state_a, b, state_b, row, angular_error,
                        timestep,
                        component(options.angular_springs.stiffness, axis_index),
                        component(options.angular_springs.damping, axis_index),
                        true);
                }
                const float angular_limit_error = angular_limit
                    ? limit_error(
                          angular_error,
                          component(options.angular_limits.lower, axis_index),
                          component(options.angular_limits.upper, axis_index))
                    : 0.0F;
                if (angular_lock || angular_limit_error != 0.0F) {
                    applied += solve_angular_constraint_axis(
                        a, state_a, b, state_b, row,
                        angular_lock ? angular_error : angular_limit_error,
                        timestep, 0.0F, 0.0F, false);
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
                        geometry.axes[2], error,
                        timestep, 0.0F, 0.0F, false);
            }
            if (options.type == RigidConstraintType::slider &&
                axis_enabled(options.linear_limits.axes, 0U)) {
                const float value = dot(
                    anchor_error, geometry.axes[0].axis);
                const float error = limit_error(
                    value, options.linear_limits.lower.x,
                    options.linear_limits.upper.x);
                if (error != 0.0F)
                    applied += solve_linear_constraint_axis(
                        a, state_a, b, state_b, arm_a, arm_b,
                        geometry.axes[0], error,
                        timestep, 0.0F, 0.0F, false);
            }
            if (options.type == RigidConstraintType::piston) {
                const Vec3 piston_axis = geometry.axes[0].axis;
                if (axis_enabled(options.linear_limits.axes, 0U)) {
                    const float error = limit_error(
                        dot(anchor_error, piston_axis), options.linear_limits.lower.x,
                        options.linear_limits.upper.x);
                    if (error != 0.0F)
                        applied += solve_linear_constraint_axis(
                            a, state_a, b, state_b, arm_a, arm_b, geometry.axes[0],
                            error, timestep, 0.0F, 0.0F, false);
                }
                if (axis_enabled(options.angular_limits.axes, 0U)) {
                    const float error = limit_error(
                        rotation_error.x, options.angular_limits.lower.x,
                        options.angular_limits.upper.x);
                    if (error != 0.0F)
                        applied += solve_angular_constraint_axis(
                            a, state_a, b, state_b, geometry.axes[0], error,
                            timestep, 0.0F, 0.0F, false);
                }
            }
            if (options.type == RigidConstraintType::motor) {
                const float inverse_iterations =
                    1.0F / static_cast<float>(
                        options.solver_iterations * contact_sweeps);
                if (options.motor.linear_enabled)
                    applied += solve_motor_axis(
                        a, state_a, b, state_b, geometry.axes[0], arm_a, arm_b,
                        options.motor.linear_target_velocity,
                        options.motor.linear_maximum_impulse * inverse_iterations,
                        false);
                if (options.motor.angular_enabled)
                    applied += solve_motor_axis(
                        a, state_a, b, state_b, geometry.axes[0], arm_a, arm_b,
                        options.motor.angular_target_velocity,
                        options.motor.angular_maximum_impulse * inverse_iterations,
                        true);
            }
            states[dense_a] = state_a;
            states[dense_b] = state_b;
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

// Split positional recovery for free welded groups against immovable surfaces.
// Translate the whole component together: correcting only its light contact
// body breaks the weld, while velocity-only recovery leaves visible overlap
// during fast impacts. This correction does not add kinetic energy.
__global__ void project_fixed_ground_contacts_kernel(
    const RigidConstraintResource *constraints, std::uint32_t capacity,
    const RigidBodyId *ids, const BodyParameters *parameters,
    RigidBodyState *states, std::uint32_t count,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, FixedContactProjection *groups) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    for (std::uint32_t body = 0U; body < count; ++body)
        groups[body] = {body, true, {}};
    bool welded = false;
    for (std::uint32_t index = 0U; index < capacity; ++index) {
        const auto &constraint = constraints[index];
        if (!constraint.alive || !constraint.state.enabled ||
            constraint.options.type != RigidConstraintType::fixed) continue;
        std::uint32_t a = find_rigid_body_dense(constraint.options.body_a, ids, count);
        std::uint32_t b = find_rigid_body_dense(constraint.options.body_b, ids, count);
        if (a == k_invalid_dense || b == k_invalid_dense) continue;
        a = fixed_projection_root(groups, a);
        b = fixed_projection_root(groups, b);
        if (a != b) groups[a > b ? a : b].root = a < b ? a : b;
        welded = true;
    }
    if (!welded) return;
    for (std::uint32_t body = 0U; body < count; ++body) {
        groups[body].root = fixed_projection_root(groups, body);
        if (parameters[body].motion != MotionType::dynamic)
            groups[groups[body].root].movable = false;
    }
    // A group tied to a hinge or other non-fixed joint cannot translate freely.
    // Leave those contacts to the coupled velocity solve.
    for (std::uint32_t index = 0U; index < capacity; ++index) {
        const auto &constraint = constraints[index];
        if (!constraint.alive || !constraint.state.enabled ||
            constraint.options.type == RigidConstraintType::fixed) continue;
        const auto a = find_rigid_body_dense(constraint.options.body_a, ids, count);
        const auto b = find_rigid_body_dense(constraint.options.body_b, ids, count);
        if (a != k_invalid_dense) groups[groups[a].root].movable = false;
        if (b != k_invalid_dense) groups[groups[b].root].movable = false;
    }
    for (std::uint32_t pass = 0U; pass < 8U; ++pass) {
        for (std::uint32_t active = 0U; active < *active_pair_count; ++active) {
            const auto pair = active_pairs[active];
            const auto a = pair / count;
            const auto b = pair % count;
            const auto &manifold = manifolds[active];
            if (manifold.count == 0U) continue;
            const bool first = parameters[a].motion == MotionType::dynamic &&
                parameters[b].motion != MotionType::dynamic &&
                manifold.contacts[0].body_hinge.fixed_member;
            const bool second = parameters[b].motion == MotionType::dynamic &&
                parameters[a].motion != MotionType::dynamic &&
                manifold.contacts[0].collider_hinge.fixed_member;
            if (!first && !second) continue;
            auto &group = groups[groups[first ? a : b].root];
            if (!group.movable) continue;
            for (std::uint32_t point = 0U; point < manifold.count; ++point) {
                const auto &contact = manifold.contacts[point];
                const Vec3 normal = multiply(contact.normal, first ? 1.0F : -1.0F);
                const float depth = contact.penetration - dot(normal, group.translation);
                if (depth > 0.0F)
                    group.translation = add(group.translation,
                        multiply(normal, depth + k_rigid_surface_tolerance));
            }
        }
    }
    for (std::uint32_t body = 0U; body < count; ++body) {
        states[body].position = add(states[body].position,
                                    groups[groups[body].root].translation);
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
    std::uint32_t constraint_capacity, Vec3 &reference,
    Quaternion *axial_orientation) noexcept {
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
        // Breakable and moving-anchor joints retain the general solver so
        // their reaction impulses and both endpoint motions remain explicit.
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
            if (axial_orientation != nullptr)
                *axial_orientation = normalized_quaternion(quaternion_multiply(
                    quaternion_multiply(other_state.orientation, other_local_orientation),
                    conjugate(local_orientation)));
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
    // Reduced coordinates are exact for one unbreakable static guide. A
    // mechanism with additional incident joints needs the general solver.
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
    std::uint32_t capacity, std::uint64_t epoch, float timestep,
    ContactResponsePatch *responses) {
    for (std::uint32_t active = blockIdx.x * blockDim.x + threadIdx.x;
         active < *active_count; active += blockDim.x * gridDim.x) {
        const std::uint32_t pair = active_pairs[active], slot = slots[pair];
        const std::uint32_t body = pair / body_count, collider = pair % body_count;
        const auto response_slot = manifolds[active].response_slot;
        if (response_slot < capacity) {
            responses[response_slot].ready = false;
            responses[response_slot].rows_ready = false;
        }
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
            if (!contact.persistent) continue;
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
    std::uint32_t *owners, const RigidCompound *compounds) {
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
        const std::uint32_t owner_index =
            compounds != nullptr && compounds[index].eligible
                ? compounds[index].root : index;
        const std::uint32_t collider_owner =
            compounds != nullptr && compounds[collider_index].eligible
                ? compounds[collider_index].root : collider_index;
        const std::uint32_t priority = contact_color_priority(pair);
        atomicMin(&owners[owner_index], priority);
        if (parameters[collider_index].motion == MotionType::dynamic) {
            atomicMin(&owners[collider_owner], priority);
        }
    }
}

__global__ void assign_parallel_contact_colors_kernel(
    const BodyParameters *parameters, std::uint32_t count,
    const ContactManifold *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const std::uint32_t *owners,
    std::uint8_t *pair_colors, std::uint32_t *color_state,
    std::uint32_t color, std::uint32_t color_round_count,
    const RigidCompound *compounds) {
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
        const std::uint32_t owner_index =
            compounds != nullptr && compounds[index].eligible
                ? compounds[index].root : index;
        const std::uint32_t collider_owner =
            compounds != nullptr && compounds[collider_index].eligible
                ? compounds[collider_index].root : collider_index;
        const bool dynamic_collider =
            parameters[collider_index].motion == MotionType::dynamic;
        const std::uint32_t priority = contact_color_priority(pair);
        if (owners[owner_index] == priority &&
            (!dynamic_collider || owners[collider_owner] == priority)) {
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
    std::uint32_t color_round_count, const RigidCompound *compounds) {
    if (blockIdx.x != 0U) return;
    const std::uint32_t active_count = *active_pair_count;
    // Same mutual-minimum ordering as multi-block coloring. Scene features
    // must not choose another dependent pair order or contact response.
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
            const std::uint32_t first_owner =
                compounds != nullptr && compounds[first].eligible
                    ? compounds[first].root : first;
            const std::uint32_t second_owner =
                compounds != nullptr && compounds[second].eligible
                    ? compounds[second].root : second;
            const std::uint32_t priority = contact_color_priority(pair);
            atomicMin(&owners[first_owner], priority);
            if (parameters[second].motion == MotionType::dynamic)
                atomicMin(&owners[second_owner], priority);
        }
        __syncthreads();
        for (std::uint32_t active_index = threadIdx.x;
             active_index < active_count; active_index += blockDim.x) {
            if (pair_colors[active_index] != k_contact_color_overflow ||
                manifolds[active_index].count == 0U) continue;
            const std::uint32_t pair = active_pairs[active_index];
            const std::uint32_t first = pair / count;
            const std::uint32_t second = pair % count;
            const std::uint32_t first_owner =
                compounds != nullptr && compounds[first].eligible
                    ? compounds[first].root : first;
            const std::uint32_t second_owner =
                compounds != nullptr && compounds[second].eligible
                    ? compounds[second].root : second;
            const bool dynamic_second =
                parameters[second].motion == MotionType::dynamic;
            const std::uint32_t priority = contact_color_priority(pair);
            if (owners[first_owner] == priority &&
                (!dynamic_second || owners[second_owner] == priority)) {
                pair_colors[active_index] = static_cast<std::uint8_t>(color);
                atomicMax(&color_state[0], color + 1U);
            } else if (color + 1U == color_round_count) {
                atomicAdd(&color_state[1], 1U);
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void resolve_active_rigid_contact_pair(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, std::uint32_t active_index,
    float timestep, bool correct_position,
    const RigidCompound *compounds, bool warm_start_only,
    ContactResponsePatch *response, ContactSolveDelta *delta) {
    const std::uint32_t pair = active_pairs[active_index];
    const std::uint32_t index = pair / count;
    const std::uint32_t collider_index = pair % count;
    ContactManifold &manifold = manifolds[active_index];
    const RigidBodyState before_a = delta != nullptr ? states[index] : RigidBodyState{};
    const RigidBodyState before_b = delta != nullptr ? states[collider_index] : RigidBodyState{};
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
    const float impulse_change = resolve_contacts(parameters, states, count, index, collider_index,
                     manifold.contacts, manifold.count,
                     manifold.initial_relative_position, timestep,
                     correct_position,
                     pair_events, retained, compounds, warm_start_only, response);
    if (delta != nullptr) {
        delta->impulse = impulse_change;
        const auto magnitude = [](Vec3 v) {
            return isfinite(v.x) && isfinite(v.y) && isfinite(v.z)
                ? fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z))) : __int_as_float(0x7f800000);
        };
        delta->velocity = fmaxf(
            fmaxf(magnitude(subtract(states[index].linear_velocity, before_a.linear_velocity)),
                  magnitude(subtract(states[collider_index].linear_velocity, before_b.linear_velocity))),
            fmaxf(magnitude(subtract(states[index].angular_velocity, before_a.angular_velocity)),
                  magnitude(subtract(states[collider_index].angular_velocity, before_b.angular_velocity))));
        delta->position = fmaxf(magnitude(subtract(states[index].position, before_a.position)),
                               magnitude(subtract(states[collider_index].position, before_b.position)));
    }
}


__device__ std::uint32_t contact_island_root(ContactIsland *islands, std::uint32_t body) {
    while (islands[body].parent != body) body = islands[body].parent;
    return body;
}

// Construction is deterministic and serial; solving is grid-parallel. Static
// ground does not join otherwise independent islands. Joint edges do.
__device__ void join_contact_islands(ContactIsland *islands, const BodyParameters *parameters,
                                    std::uint32_t a, std::uint32_t b) {
    if (parameters[a].motion != MotionType::dynamic || parameters[b].motion != MotionType::dynamic) return;
    a = contact_island_root(islands, a);
    b = contact_island_root(islands, b);
    islands[max(a, b)].parent = min(a, b);
}

__global__ void prepare_rigid_contact_schedule_kernel(
    const BodyParameters *parameters, std::uint32_t count, const ContactManifold *manifolds,
    const std::uint32_t *active_pairs, const std::uint32_t *active_pair_count,
    const std::uint8_t *pair_colors, std::uint32_t *color_work,
    ContactSchedule *schedule, ContactIsland *islands,
    const RigidConstraintResource *constraints, std::uint32_t constraint_capacity,
    const RigidBodyId *ids, unsigned solve_blocks) {
    const auto rank = threadIdx.x;
    const auto stride = blockDim.x;
    const std::uint32_t active_count = *active_pair_count;
    constexpr unsigned overflow = 28U;
    for (unsigned color = rank; color <= overflow; color += stride) schedule->counts[color] = 0U;
    for (unsigned body = rank; body < count; body += stride) {
        islands[body] = {};
        islands[body].parent = body;
    }
    __syncthreads();
    if (rank == 0U) {
        schedule->budget = 8U;
        schedule->candidates = active_count;
        schedule->contacts = 0U;
        schedule->island_count = schedule->early_exit_count = schedule->maximum_passes = 0U;
        schedule->blocks = solve_blocks;
        for (std::uint32_t active = 0U; active < active_count; ++active) {
            const auto &manifold = manifolds[active];
            if (manifold.count == 0U) continue;
            ++schedule->contacts;
            const auto pair = active_pairs[active];
            join_contact_islands(islands, parameters, pair / count, pair % count);
            schedule->budget = max(schedule->budget, solver::contact_iteration_budget(manifold.face_patch));
        }
        for (unsigned joint = 0U; joint < constraint_capacity; ++joint) {
            const auto &constraint = constraints[joint];
            if (!constraint.alive || !constraint.options.enabled || constraint.state.broken) continue;
            const auto a = find_rigid_body_dense(constraint.options.body_a, ids, count);
            const auto b = find_rigid_body_dense(constraint.options.body_b, ids, count);
            if (a != k_invalid_dense && b != k_invalid_dense) join_contact_islands(islands, parameters, a, b);
        }
        for (unsigned body = 0U; body < count; ++body)
            islands[body].parent = contact_island_root(islands, body);
    }
    __syncthreads();
    for (unsigned active = rank; active < active_count; active += stride) {
        if (manifolds[active].count == 0U) continue;
        const unsigned color = pair_colors[active] == k_contact_color_overflow ? overflow : pair_colors[active];
        atomicAdd(&schedule->counts[color], 1U);
        const auto pair = active_pairs[active];
        const auto body = parameters[pair / count].motion == MotionType::dynamic ? pair / count : pair % count;
        // Preserve the established maximum budget; islands may independently
        // stop only on measured convergence, not merely by their shape type.
        atomicMax(&islands[islands[body].parent].budget, schedule->budget);
    }
    __syncthreads();
    if (rank == 0U) {
        std::uint32_t offset = 0U;
        for (unsigned body = 0U; body < count; ++body) {
            auto &island = islands[body];
            if (island.parent != body || island.budget == 0U) continue;
            islands[schedule->island_count++].active_root = body;
        }
        for (unsigned color = 0U; color <= overflow; ++color) {
            schedule->offsets[color] = offset;
            offset += schedule->counts[color];
            schedule->counts[color] = 0U;
        }
        schedule->remaining = schedule->island_count;
        schedule->blocks = schedule->island_count == 0U ? 0U : solve_blocks;
    }
    __syncthreads();
    for (unsigned active = rank; active < active_count; active += stride) {
        if (manifolds[active].count == 0U) continue;
        const unsigned color = pair_colors[active] == k_contact_color_overflow ? overflow : pair_colors[active];
        if (color == overflow) continue;
        color_work[schedule->offsets[color] + atomicAdd(&schedule->counts[color], 1U)] = active;
    }
    __syncthreads();
    if (rank == 0U) for (unsigned active = 0U; active < active_count; ++active) {
        if (manifolds[active].count == 0U || pair_colors[active] != k_contact_color_overflow) continue;
        color_work[schedule->offsets[overflow] + schedule->counts[overflow]++] = active;
    }
}

// Aggregate same-island residuals within a warp before touching global island
// counters. Zero changes need no atomic transaction at all.
__device__ void reduce_contact_delta(ContactResidual &residual, ContactSolveDelta delta) {
    const unsigned peers = __match_any_sync(__activemask(), reinterpret_cast<unsigned long long>(&residual));
    unsigned velocity = __float_as_uint(delta.velocity);
    unsigned position = __float_as_uint(delta.position);
    unsigned impulse = __float_as_uint(delta.impulse);
#if __CUDA_ARCH__ >= 800
    velocity = __reduce_max_sync(peers, velocity);
    position = __reduce_max_sync(peers, position);
    impulse = __reduce_max_sync(peers, impulse);
#else
    for (unsigned lanes = peers; lanes != 0U; lanes &= lanes - 1U) {
        const int lane = __ffs(lanes) - 1;
        velocity = max(velocity, __shfl_sync(peers, __float_as_uint(delta.velocity), lane));
        position = max(position, __shfl_sync(peers, __float_as_uint(delta.position), lane));
        impulse = max(impulse, __shfl_sync(peers, __float_as_uint(delta.impulse), lane));
    }
#endif
    if ((threadIdx.x & 31U) == static_cast<unsigned>(__ffs(peers) - 1)) {
        if (velocity != 0U) atomicMax(&residual.velocity_change, velocity);
        if (position != 0U) atomicMax(&residual.position_change, position);
        if (impulse != 0U) atomicMax(&residual.impulse_change, impulse);
    }
}

// Colors stay globally ordered, but even a single connected stack uses the
// whole resident grid. Small blocks spread narrow colors across SMs. A device
// without cooperative launch runs this exact kernel as one block instead.
__global__ void resolve_rigid_contacts_kernel(
    const BodyParameters *parameters, RigidBodyState *states,
    std::uint32_t count, ContactManifold *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *event_offsets, RigidContactEvent *events,
    std::uint32_t event_capacity, float timestep,
    const RigidCompound *compounds, ContactResponsePatch *responses,
    std::uint32_t response_capacity, const std::uint32_t *color_work,
    ContactSchedule *schedule, ContactIsland *islands, const std::uint32_t *color_state) {
    __shared__ ContactResidual local_residuals[256];
    const auto residual_for = [&](ContactIsland &island) -> ContactResidual & {
        return gridDim.x == 1U && count <= 256U
            ? local_residuals[&island - islands] : island.residual;
    };
    const auto grid = cooperative_groups::this_grid();
    const auto synchronize = [&]() {
        if (gridDim.x == 1U) __syncthreads();
        else grid.sync();
    };
    const unsigned rank = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned pass = 0U; pass <= schedule->budget && schedule->remaining != 0U; ++pass) {
        for (unsigned work = rank; work < schedule->island_count; work += stride) {
            auto &island = islands[islands[work].active_root];
            residual_for(island) = {};
        }
        if (pass >= 7U) {
            for (unsigned body = rank; body < count; body += stride) {
                islands[body].previous_linear = states[body].linear_velocity;
                islands[body].previous_angular = states[body].angular_velocity;
                islands[body].previous_position = states[body].position;
            }
        }
        synchronize();
        const auto solve_pair = [&](unsigned active_index) {
            const auto pair = active_pairs[active_index];
            const auto body = parameters[pair / count].motion == MotionType::dynamic ? pair / count : pair % count;
            auto &island = islands[islands[body].parent];
            if (island.convergence.finished) return;
            ContactSolveDelta delta{};
            resolve_active_rigid_contact_pair(parameters, states, count,
                manifolds, active_pairs, event_offsets, events, event_capacity,
                active_index, timestep, pass == 1U, compounds, pass == 0U,
                manifolds[active_index].response_slot < response_capacity
                    ? responses + manifolds[active_index].response_slot : nullptr,
                pass >= 7U ? &delta : nullptr);
            // Net body changes alone can hide opposing contact updates. An
            // impact must also settle within each pair before its island stops.
            if (pass >= 7U) reduce_contact_delta(residual_for(island), delta);
        };
        for (unsigned color = 0U; color < color_state[0]; ++color) {
            for (unsigned work = rank; work < schedule->counts[color]; work += stride)
                solve_pair(color_work[schedule->offsets[color] + work]);
            synchronize();
        }
        if (rank == 0U) for (unsigned work = 0U; work < schedule->counts[28]; ++work)
            solve_pair(color_work[schedule->offsets[28] + work]);
        synchronize();
        if (pass >= 7U) {
            const auto magnitude = [](Vec3 v) {
                return isfinite(v.x) && isfinite(v.y) && isfinite(v.z)
                    ? fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z))) : __int_as_float(0x7f800000);
            };
            for (unsigned body = rank; body < count; body += stride) {
                if (parameters[body].motion != MotionType::dynamic) continue;
                auto &island = islands[islands[body].parent];
                if (island.budget == 0U || island.convergence.finished) continue;
                const auto &previous = islands[body];
                const float velocity = fmaxf(magnitude(subtract(states[body].linear_velocity, previous.previous_linear)),
                    magnitude(subtract(states[body].angular_velocity, previous.previous_angular)));
                const float position = magnitude(subtract(states[body].position, previous.previous_position));
                reduce_contact_delta(residual_for(island), {velocity, position, 0.0F});
            }
        }
        synchronize();
        if (pass >= 7U) for (unsigned work = rank; work < schedule->island_count; work += stride) {
            auto &island = islands[islands[work].active_root];
            if (island.convergence.finished) continue;
            const auto &residual = residual_for(island);
            island.convergence.observe(pass, island.budget,
                __uint_as_float(residual.velocity_change), __uint_as_float(residual.position_change),
                __uint_as_float(residual.impulse_change));
            if (island.convergence.finished) {
                atomicMax(&schedule->maximum_passes, pass);
                if (pass < island.budget) atomicAdd(&schedule->early_exit_count, 1U);
                atomicSub(&schedule->remaining, 1U);
            }
        }
        synchronize();
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
