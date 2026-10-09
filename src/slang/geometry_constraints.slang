// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_constraints.cuh.

import cuda_common;

struct RigidAvbdVector6 { float values[6]; };
struct RigidAvbdDual { float lambda; float penalty; };
struct RigidAvbdRow
{
    RigidAvbdVector6 a;
    RigidAvbdVector6 b;
    RigidAvbdDual dual;
    float error;
    float velocity;
    float lower;
    float upper;
    float stiffness;
    float damping;
    float beta;
    float reference_force;
};

struct BodyAccumulator
{
    Vec3 force;
    Vec3 torque;
    Vec3 impulse;
    Vec3 angular_impulse;
};

struct KinematicTarget
{
    RigidBodyState state;
    bool active;
};

enum RigidConstraintType : uint8_t
{
    fixed,
    point,
    hinge,
    slider,
    piston,
    generic,
    generic_spring,
    motor,
};

struct RigidConstraintLimitOptions
{
    uint8_t axes;
    Vec3 lower;
    Vec3 upper;
};

struct RigidConstraintSpringOptions
{
    uint8_t axes;
    Vec3 stiffness;
    Vec3 damping;
};

struct RigidConstraintMotorOptions
{
    bool linear_enabled;
    bool angular_enabled;
    float linear_target_velocity;
    float linear_maximum_impulse;
    float angular_target_velocity;
    float angular_maximum_impulse;
};

struct RigidConstraintOptions
{
    RigidConstraintType type;
    RigidBodyId body_a;
    RigidBodyId body_b;
    Vec3 local_anchor_a;
    Vec3 local_anchor_b;
    Quaternion local_orientation_a;
    Quaternion local_orientation_b;
    RigidConstraintLimitOptions linear_limits;
    RigidConstraintLimitOptions angular_limits;
    RigidConstraintSpringOptions linear_springs;
    RigidConstraintSpringOptions angular_springs;
    RigidConstraintMotorOptions motor;
    bool enabled;
    bool disable_collisions;
    float breaking_impulse_threshold;
    uint solver_iterations;
};

struct RigidConstraintState
{
    bool enabled;
    bool broken;
    float applied_impulse;
};

struct RigidConstraintGeometry
{
    uint dense_a;
    uint dense_b;
};

struct RigidConstraintResource
{
    RigidConstraintOptions options;
    RigidConstraintState state;
    uint generation;
    bool alive;
    RigidConstraintGeometry geometry;
    RigidAvbdRow avbd_rows[18];
    uint avbd_count;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct FixedContactProjection { uint root; };

struct HingeContactFrame
{
    Vec3 anchor;
    Vec3 axis;
    Vec3 local_anchor;
    float inverse_mass;
    float inverse_moment;
    bool present;
    bool fixed;
    bool axial;
    bool axial_rotation;
    bool fixed_member;
    bool point_member;
    bool static_body;
};

struct Contact
{
    Vec3 normal;
    Vec3 point;
    float penetration;
    HingeContactFrame body_hinge;
    HingeContactFrame collider_hinge;
    float accumulated_normal_impulse;
    Vec3 accumulated_friction_impulse;
    float initial_normal_speed;
    bool warm_started;
    bool persistent;
    float impact_fraction;
    float reported_normal_impulse;
    Vec3 reported_friction_impulse;
    RigidAvbdDual avbd_dual[3];
    Vec3 avbd_error;
    Vec3 avbd_anchor_a;
    Vec3 avbd_anchor_b;
    bool avbd_cached;
};

struct ContactManifold
{
    Contact contacts[8];
    uint count;
    Vec3 initial_relative_position;
    bool face_patch;
    uint response_slot;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct CachedContact
{
    Vec3 local_point;
    Vec3 normal;
    float normal_impulse;
    Vec3 friction_impulse;
    RigidAvbdDual avbd_dual[3];
    Vec3 avbd_anchor_a;
    Vec3 avbd_anchor_b;
};

struct CachedContactPair
{
    uint pair;
    RigidBodyId body;
    RigidBodyId collider;
    uint64_t epoch;
    float timestep;
    uint count;
    CachedContact contacts[8];
};

struct ContactSchedule
{
    uint counts[29];
    uint offsets[29];
    uint budget;
    uint remaining;
    uint island_count;
    uint early_exit_count;
    uint maximum_passes;
    uint blocks;
    uint candidates;
    uint contacts;
};

struct LeafPair
{
    uint body_first;
    uint body_count;
    uint collider_first;
    uint collider_count;
};

struct RigidContactEvent
{
    RigidBodyId body;
    RigidBodyId collider;
    Vec3 position;
    Vec3 normal;
    float penetration;
    float normal_impulse;
    Vec3 friction_impulse;
};

struct PhysicsDebugRigidSample
{
    RigidBodyId id;
    RigidBodyState state;
    Vec3 applied_force;
    Vec3 applied_torque;
};

uint find_rigid_body_dense(RigidBodyId id, RigidBodyId* ids, uint count)
{
    for (uint index = 0u; index < count; ++index)
        if (ids[index].index == id.index &&
            ids[index].generation == id.generation) return index;
    return kInvalidDense;
}

bool same_id(RigidBodyId a, RigidBodyId b)
{
    return a.index == b.index && a.generation == b.generation;
}

bool constrained_collision_disabled(
    RigidBodyId a, RigidBodyId b, RigidConstraintResource* constraints,
    uint capacity)
{
    for (uint index = 0u; index < capacity; ++index)
    {
        RigidConstraintResource constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken || !constraint.options.disable_collisions)
            continue;
        if ((same_id(constraint.options.body_a, a) &&
             same_id(constraint.options.body_b, b)) ||
            (same_id(constraint.options.body_a, b) &&
             same_id(constraint.options.body_b, a))) return true;
    }
    return false;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void integrate_rigid_bodies_kernel(
    uniform BodyParameters* parameters, uniform BodyAccumulator* accumulators,
    uniform KinematicTarget* targets, uniform RigidBodyState* input,
    uniform RigidBodyState* output, uniform uint count, uniform Vec3 gravity,
    uniform float timestep, uniform uint remaining_substeps,
    uniform bool apply_impulses)
{
    uint index = pmThreadIndex();
    if (index >= count) return;
    BodyParameters body = parameters[index];
    RigidBodyState previous = input[index], next = previous;
    if (body.motion == MotionType.static_body)
    {
        next.linear_velocity = {};
        next.angular_velocity = {};
        output[index] = next;
        return;
    }
    if (body.motion == MotionType.kinematic)
    {
        if (targets[index].active)
        {
            float fraction = 1.0f / float(remaining_substeps);
            next.position = add(previous.position, multiply(subtract(
                targets[index].state.position, previous.position), fraction));
            Quaternion target = targets[index].state.orientation;
            float orientationDot = previous.orientation.x * target.x +
                previous.orientation.y * target.y +
                previous.orientation.z * target.z +
                previous.orientation.w * target.w;
            if (orientationDot < 0.0f)
                target = {-target.x, -target.y, -target.z, -target.w};
            next.orientation = normalized_quaternion({
                previous.orientation.x +
                    (target.x - previous.orientation.x) * fraction,
                previous.orientation.y +
                    (target.y - previous.orientation.y) * fraction,
                previous.orientation.z +
                    (target.z - previous.orientation.z) * fraction,
                previous.orientation.w +
                    (target.w - previous.orientation.w) * fraction});
            next.linear_velocity = multiply(
                subtract(next.position, previous.position), 1.0f / timestep);
            next.angular_velocity = quaternion_delta_velocity(
                previous.orientation, next.orientation, timestep);
        }
        else
        {
            next.linear_velocity = {};
            next.angular_velocity = {};
        }
        output[index] = next;
        return;
    }
    BodyAccumulator accumulator = accumulators[index];
    next.linear_velocity = add(next.linear_velocity, multiply(add(gravity,
        multiply(accumulator.force, body.inverse_mass)), timestep));
    next.angular_velocity = add(next.angular_velocity,
        multiply(inverse_inertia_world(body, next, accumulator.torque),
            timestep));
    if (apply_impulses)
    {
        next.linear_velocity = add(next.linear_velocity,
            multiply(accumulator.impulse, body.inverse_mass));
        next.angular_velocity = add(next.angular_velocity,
            inverse_inertia_world(body, next, accumulator.angular_impulse));
    }
    next.linear_velocity = multiply(next.linear_velocity,
        1.0f / (1.0f + body.linear_damping * timestep));
    next.angular_velocity = multiply(next.angular_velocity,
        1.0f / (1.0f + body.angular_damping * timestep));
    next.linear_velocity = clamp_length(
        next.linear_velocity, body.maximum_linear_speed);
    next.angular_velocity = clamp_length(
        next.angular_velocity, body.maximum_angular_speed);
    next.position = add(next.position, multiply(next.linear_velocity, timestep));
    Quaternion angular = {next.angular_velocity.x, next.angular_velocity.y,
                          next.angular_velocity.z, 0.0f};
    Quaternion derivative = quaternion_multiply(angular, next.orientation);
    next.orientation = normalized_quaternion({
        next.orientation.x + 0.5f * derivative.x * timestep,
        next.orientation.y + 0.5f * derivative.y * timestep,
        next.orientation.z + 0.5f * derivative.z * timestep,
        next.orientation.w + 0.5f * derivative.w * timestep});
    output[index] = next;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_rigid_world_bounds_kernel(
    uniform BodyParameters* parameters,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform uint count, uniform TriangleMeshResource* meshes,
    uniform WorldAabb* world_bounds)
{
    uint index = pmThreadIndex();
    if (index >= count) return;
    TriangleMeshResource mesh = meshes[parameters[index].mesh.index];
    bool swept = vector_length(subtract(states[index].position,
        previous_states[index].position)) + rotational_motion_bound(
            previous_states[index], states[index], mesh) >
        parameters[index].collision_margin;
    BoundsTransform previousTransform = {};
    if (swept) previousTransform = bounds_transform(previous_states[index]);
    Vec3 minimum = {}, maximum = {};
    transformed_motion_bounds(mesh.minimum, mesh.maximum, previousTransform,
        bounds_transform(states[index]), swept, 0.0f, minimum, maximum);
    world_bounds[index] = {minimum, maximum};
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broad_phase_rigid_pairs_kernel(
    uniform BodyParameters* parameters, uniform WorldAabb* world_bounds,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform TriangleMeshResource* meshes, uniform RigidBodyId* ids,
    uniform RigidConstraintResource* constraints,
    uniform uint constraint_capacity,
    uniform FixedContactProjection* fixed_groups, uniform uint count,
    uniform uint8_t* active_flags)
{
    uint pair = pmThreadIndex();
    if (pair >= count * count) return;
    uint bodyIndex = pair / count, colliderIndex = pair % count;
    bool active = parameters[bodyIndex].motion == MotionType.dynamic &&
        bodyIndex != colliderIndex;
    if (active && parameters[colliderIndex].motion == MotionType.dynamic &&
        colliderIndex < bodyIndex) active = false;
    if (active && fixed_groups != nullptr &&
        fixed_groups[bodyIndex].root == fixed_groups[colliderIndex].root)
        active = false;
    if (active && constrained_collision_disabled(ids[bodyIndex],
        ids[colliderIndex], constraints, constraint_capacity)) active = false;
    if (active)
    {
        float margin = parameters[bodyIndex].collision_margin +
            parameters[colliderIndex].collision_margin;
        WorldAabb a = world_bounds[bodyIndex];
        WorldAabb b = world_bounds[colliderIndex];
        Vec3 expansion = {margin, margin, margin};
        active = bounds_overlap(subtract(a.minimum, expansion),
            add(a.maximum, expansion), b.minimum, b.maximum);
    }
    active_flags[pair] = active ? uint8_t(1u) : uint8_t(0u);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void generate_rigid_leaf_pairs_kernel(
    uniform BodyParameters* parameters,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform uint count, uniform TriangleMeshResource* meshes,
    uniform uint mesh_capacity, uniform uint* active_pairs,
    uniform uint* active_pair_count, uniform LeafPair* leaf_pairs,
    uniform uint* leaf_pair_counts,
    uniform uint leaf_pair_cache_slot_capacity,
    uniform uint leaf_pairs_per_slot, uniform uint shared_leaf_capacity)
{
    for (uint activeIndex = cudaBlockIdx().x; activeIndex < *active_pair_count;
         activeIndex += pmGridDim().x)
    {
        uint pair = active_pairs[activeIndex];
        if (cudaThreadIdx().x != 0u) continue;
        if (activeIndex >= leaf_pair_cache_slot_capacity ||
            leaf_pairs_per_slot == 0u)
        {
            leaf_pair_counts[pair] = 0xffffffffu;
            continue;
        }
        uint a = pair / count, b = pair % count;
        TriangleMeshResource ma = meshes[parameters[a].mesh.index];
        TriangleMeshResource mb = meshes[parameters[b].mesh.index];
        leaf_pairs[activeIndex * leaf_pairs_per_slot] = {
            0u, ma.index_count / 3u, 0u, mb.index_count / 3u};
        leaf_pair_counts[pair] = 1u;
    }
}

Contact coarse_mesh_contact(
    TriangleMeshResource aMesh, RigidBodyState a,
    TriangleMeshResource bMesh, RigidBodyState b,
    float margin, out bool found)
{
    Vec3 ac = transform_point(a, aMesh.bounding_center);
    Vec3 bc = transform_point(b, bMesh.bounding_center);
    Vec3 delta = subtract(ac, bc);
    float distance = vector_length(delta);
    float radius = aMesh.bounding_radius + bMesh.bounding_radius + margin;
    found = distance < radius;
    Contact contact = {};
    if (!found) return contact;
    contact.normal = normalized_or(delta, {1.0f, 0.0f, 0.0f});
    contact.point = add(bc, multiply(contact.normal, bMesh.bounding_radius));
    contact.penetration = radius - distance;
    contact.impact_fraction = 1.0f;
    return contact;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void evaluate_rigid_leaf_pairs_kernel(
    uniform BodyParameters* parameters,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform uint count, uniform TriangleMeshResource* meshes,
    uniform RigidBodyId* ids, uniform RigidConstraintResource* constraints,
    uniform uint constraint_capacity, uniform uint* active_pairs,
    uniform uint* active_pair_count, uniform LeafPair* leaf_pairs,
    uniform uint* leaf_pair_counts, uniform uint leaf_pairs_per_slot,
    uniform ContactManifold* leaf_manifolds, uniform uint blocks_per_pair,
    uniform float timestep, uniform ContactManifold* manifolds)
{
    uint activeIndex = pmThreadIndex();
    if (activeIndex >= *active_pair_count) return;
    uint pair = active_pairs[activeIndex];
    uint a = pair / count, b = pair % count;
    ContactManifold manifold = {};
    bool found = false;
    Contact contact = coarse_mesh_contact(
        meshes[parameters[a].mesh.index], states[a],
        meshes[parameters[b].mesh.index], states[b],
        parameters[a].collision_margin + parameters[b].collision_margin,
        found);
    if (found)
    {
        contact.initial_normal_speed = dot(subtract(
            states[a].linear_velocity, states[b].linear_velocity),
            contact.normal);
        manifold.contacts[0] = contact;
        manifold.count = 1u;
        manifold.initial_relative_position = subtract(
            states[a].position, states[b].position);
    }
    manifolds[activeIndex] = manifold;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce_rigid_leaf_manifolds_kernel(
    uniform BodyParameters* parameters, uniform uint count,
    uniform uint* active_pairs, uniform uint* active_pair_count,
    uniform uint* leaf_pair_counts, uniform uint leaf_pairs_per_slot,
    uniform ContactManifold* leaf_manifolds,
    uniform ContactManifold* manifolds)
{
    uint activeIndex = pmThreadIndex();
    if (activeIndex >= *active_pair_count) return;
    uint pair = active_pairs[activeIndex];
    uint candidates = leaf_pair_counts[pair];
    if (manifolds[activeIndex].count != 0u || candidates == 0u ||
        candidates == 0xffffffffu) return;
    ContactManifold result = {};
    for (uint candidate = 0u; candidate < min(candidates, leaf_pairs_per_slot);
         ++candidate)
    {
        ContactManifold local =
            leaf_manifolds[activeIndex * leaf_pairs_per_slot + candidate];
        for (uint contact = 0u; contact < local.count && result.count < 8u;
             ++contact)
            result.contacts[result.count++] = local.contacts[contact];
    }
    manifolds[activeIndex] = result;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void finalize_rigid_contact_manifolds_kernel(
    uniform BodyParameters* parameters,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform uint count, uniform TriangleMeshResource* meshes,
    uniform RigidBodyId* ids, uniform RigidConstraintResource* constraints,
    uniform uint constraint_capacity, uniform uint* active_pairs,
    uniform uint* active_pair_count, uniform uint* leaf_pair_counts,
    uniform float timestep, uniform ContactManifold* manifolds,
    uniform bool prepare_faces)
{
    uint activeIndex = pmThreadIndex();
    if (activeIndex >= *active_pair_count) return;
    ContactManifold manifold = manifolds[activeIndex];
    for (uint point = 0u; point < manifold.count; ++point)
    {
        manifold.contacts[point].reported_normal_impulse =
            manifold.contacts[point].accumulated_normal_impulse;
        manifold.contacts[point].reported_friction_impulse =
            manifold.contacts[point].accumulated_friction_impulse;
    }
    manifolds[activeIndex] = manifold;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clear_rigid_inputs_kernel(
    uniform BodyAccumulator* accumulators, uniform KinematicTarget* targets,
    uniform uint count)
{
    uint index = pmThreadIndex();
    if (index >= count) return;
    accumulators[index] = {};
    targets[index].active = false;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void capture_rigid_inputs_kernel(
    uniform BodyAccumulator* accumulators, uniform Vec3* forces,
    uniform Vec3* torques, uniform uint count)
{
    uint index = pmThreadIndex();
    if (index >= count) return;
    forces[index] = accumulators[index].force;
    torques[index] = accumulators[index].torque;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void prepare_parallel_contact_events_kernel(
    uniform uint count, uniform ContactManifold* manifolds,
    uniform uint* active_pairs, uniform uint* active_pair_count,
    uniform RigidBodyId* ids, uniform uint* event_offsets,
    uniform RigidContactEvent* events, uniform uint event_capacity,
    uniform uint* event_count, uniform bool collect_events,
    uniform bool reset_events)
{
    if (reset_events) *event_count = 0u;
    uint cursor = 0u;
    for (uint active = 0u; active < *active_pair_count; ++active)
    {
        event_offsets[active] = cursor;
        ContactManifold manifold = manifolds[active];
        manifold.response_slot = active;
        manifolds[active] = manifold;
        if (!collect_events) continue;
        uint pair = active_pairs[active];
        uint a = pair / count, b = pair % count;
        for (uint point = 0u; point < manifold.count && cursor < event_capacity;
             ++point)
        {
            Contact contact = manifold.contacts[point];
            events[cursor++] = {ids[a], ids[b], contact.point,
                contact.normal, max(0.0f, contact.penetration),
                contact.reported_normal_impulse,
                contact.reported_friction_impulse};
        }
    }
    if (collect_events) *event_count = cursor;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void load_rigid_contact_cache_kernel(
    uniform ContactManifold* manifolds, uniform uint* active_pairs,
    uniform uint* active_count, uniform RigidBodyState* states,
    uniform RigidBodyId* ids, uniform uint body_count,
    uniform CachedContactPair* cache, uniform uint* slots,
    uniform uint capacity, uniform uint64_t epoch, uniform float timestep)
{
    uint active = pmThreadIndex();
    if (active >= *active_count) return;
    uint pair = active_pairs[active], slot = slots[pair];
    uint body = pair / body_count, collider = pair % body_count;
    if (slot >= capacity) return;
    CachedContactPair saved = cache[slot];
    if (saved.pair != pair || saved.epoch + 1ull != epoch ||
        !same_id(saved.body, ids[body]) ||
        !same_id(saved.collider, ids[collider]) ||
        abs(saved.timestep - timestep) > 1.0e-7f) return;
    ContactManifold manifold = manifolds[active];
    uint used = min(manifold.count, saved.count);
    for (uint point = 0u; point < used; ++point)
    {
        manifold.contacts[point].accumulated_normal_impulse =
            saved.contacts[point].normal_impulse;
        manifold.contacts[point].accumulated_friction_impulse =
            saved.contacts[point].friction_impulse;
        manifold.contacts[point].warm_started = true;
        manifold.contacts[point].persistent = true;
        manifold.contacts[point].avbd_cached = true;
        for (uint axis = 0u; axis < 3u; ++axis)
            manifold.contacts[point].avbd_dual[axis] =
                saved.contacts[point].avbd_dual[axis];
        manifold.contacts[point].avbd_anchor_a =
            saved.contacts[point].avbd_anchor_a;
        manifold.contacts[point].avbd_anchor_b =
            saved.contacts[point].avbd_anchor_b;
    }
    manifolds[active] = manifold;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void save_rigid_contact_cache_kernel(
    uniform ContactManifold* manifolds, uniform uint* active_pairs,
    uniform uint* active_count, uniform RigidBodyState* states,
    uniform RigidBodyId* ids, uniform uint body_count,
    uniform CachedContactPair* cache, uniform uint* slots,
    uniform uint capacity, uniform uint64_t epoch, uniform float timestep,
    uniform uint* event_offsets, uniform RigidContactEvent* events,
    uniform uint event_capacity)
{
    uint active = pmThreadIndex();
    if (active >= *active_count) return;
    uint pair = active_pairs[active], slot = slots[pair];
    if (slot >= capacity) return;
    uint body = pair / body_count, collider = pair % body_count;
    ContactManifold manifold = manifolds[active];
    CachedContactPair saved = {};
    saved.pair = pair;
    saved.body = ids[body];
    saved.collider = ids[collider];
    saved.epoch = epoch;
    saved.timestep = timestep;
    saved.count = min(manifold.count, 8u);
    for (uint point = 0u; point < saved.count; ++point)
    {
        Contact contact = manifold.contacts[point];
        saved.contacts[point].local_point = inverse_rotate(
            states[body].orientation,
            subtract(contact.point, states[body].position));
        saved.contacts[point].normal = contact.normal;
        saved.contacts[point].normal_impulse =
            contact.accumulated_normal_impulse;
        saved.contacts[point].friction_impulse =
            contact.accumulated_friction_impulse;
        for (uint axis = 0u; axis < 3u; ++axis)
            saved.contacts[point].avbd_dual[axis] = contact.avbd_dual[axis];
        saved.contacts[point].avbd_anchor_a = contact.avbd_anchor_a;
        saved.contacts[point].avbd_anchor_b = contact.avbd_anchor_b;
    }
    cache[slot] = saved;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void gather_rigid_debug_samples_kernel(
    uniform RigidBodyId* ids, uniform RigidBodyState* states,
    uniform Vec3* forces, uniform Vec3* torques, uniform uint count,
    uniform PhysicsDebugRigidSample* samples)
{
    uint index = pmThreadIndex();
    if (index < count)
        samples[index] = {ids[index], states[index], forces[index], torques[index]};
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clamp_rigid_speeds_kernel(
    uniform BodyParameters* parameters, uniform RigidBodyState* states,
    uniform uint count)
{
    uint index = pmThreadIndex();
    if (index >= count || parameters[index].motion != MotionType.dynamic)
        return;
    states[index].linear_velocity = clamp_length(
        states[index].linear_velocity, parameters[index].maximum_linear_speed);
    states[index].angular_velocity = clamp_length(
        states[index].angular_velocity, parameters[index].maximum_angular_speed);
}

uint fixed_projection_root(FixedContactProjection* groups, uint body)
{
    while (groups[body].root != body) body = groups[body].root;
    return body;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void build_fixed_collision_groups_kernel(
    uniform RigidConstraintResource* constraints, uniform uint capacity,
    uniform RigidBodyId* ids, uniform uint count,
    uniform FixedContactProjection* groups)
{
    for (uint body = 0u; body < count; ++body) groups[body].root = body;
    for (uint index = 0u; index < capacity; ++index)
    {
        RigidConstraintResource constraint = constraints[index];
        if (!constraint.alive || !constraint.options.enabled ||
            constraint.state.broken || !constraint.options.disable_collisions ||
            constraint.options.type != RigidConstraintType.fixed) continue;
        uint a = find_rigid_body_dense(
            constraint.options.body_a, ids, count);
        uint b = find_rigid_body_dense(
            constraint.options.body_b, ids, count);
        if (a == kInvalidDense || b == kInvalidDense) continue;
        a = fixed_projection_root(groups, a);
        b = fixed_projection_root(groups, b);
        if (a != b) groups[max(a, b)].root = min(a, b);
    }
}
