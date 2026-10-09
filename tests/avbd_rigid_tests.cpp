// SPDX-License-Identifier: MIT
// Focused public-API regressions for the production AVBD adapters. Every case
// prints physical errors, and --case NAME permits short solver experiments.
#include <parallel_mater/parallel_mater.hpp>
#include <parallel_mater/solver/avbd.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
using namespace parallel_mater;
constexpr float pi = 3.14159265358979323846F;
unsigned failures{};
std::uint32_t pass_limit{};
bool trace_contacts{};

void check(bool condition, const char *message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
void require(Status status, const char *operation) {
    if (!status) {
        std::cerr << "ERROR: " << operation << ": " << (status.message ? status.message : "unknown") << '\n';
        std::exit(1);
    }
}
void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::cerr << "ERROR: " << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}
Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 mul(Vec3 a, float value) { return {a.x * value, a.y * value, a.z * value}; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
float length(Vec3 a) { return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
Quaternion conjugate(Quaternion a) { return {-a.x, -a.y, -a.z, a.w}; }
Quaternion product(Quaternion a, Quaternion b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
            a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
            a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
            a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}
Quaternion rotation(Vec3 axis, float angle) {
    return {axis.x*std::sin(angle/2), axis.y*std::sin(angle/2), axis.z*std::sin(angle/2), std::cos(angle/2)};
}
Vec3 rotate(Quaternion q, Vec3 v) {
    const Vec3 u{q.x, q.y, q.z};
    return add(v, mul(add(mul(cross(u, v), q.w), cross(u, cross(u, v))), 2));
}
float rotation_difference(Quaternion a, Quaternion b) {
    const auto q = product(a, conjugate(b));
    return 2*std::atan2(length({q.x, q.y, q.z}), std::fabs(q.w));
}
bool finite(RigidBodyState state) {
    return std::isfinite(length(state.position)) && std::isfinite(length(state.linear_velocity)) &&
           std::isfinite(length(state.angular_velocity)) && std::isfinite(state.orientation.w) &&
           std::isfinite(state.orientation.x) && std::isfinite(state.orientation.y) && std::isfinite(state.orientation.z);
}
TriangleMeshId mesh(World &world, const std::vector<Vec3> &vertices,
                    const std::vector<std::uint32_t> &indices) {
    Vec3 *device_vertices{}; std::uint32_t *device_indices{};
    require_cuda(cudaMalloc(reinterpret_cast<void **>(&device_vertices), vertices.size()*sizeof(Vec3)), "mesh allocation");
    require_cuda(cudaMalloc(reinterpret_cast<void **>(&device_indices), indices.size()*sizeof(std::uint32_t)), "index allocation");
    require_cuda(cudaMemcpy(device_vertices, vertices.data(), vertices.size()*sizeof(Vec3), cudaMemcpyHostToDevice), "mesh upload");
    require_cuda(cudaMemcpy(device_indices, indices.data(), indices.size()*sizeof(std::uint32_t), cudaMemcpyHostToDevice), "index upload");
    TriangleMeshId id{};
    require(world.add_triangle_mesh({device_vertices, vertices.size()}, {device_indices, indices.size()}, id), "add mesh");
    require_cuda(cudaFree(device_vertices), "mesh free"); require_cuda(cudaFree(device_indices), "index free");
    return id;
}
TriangleMeshId box(World &world, float half = 0.1F) {
    return mesh(world, {{-half,-half,-half},{half,-half,-half},{half,half,-half},{-half,half,-half},
                        {-half,-half,half},{half,-half,half},{half,half,half},{-half,half,half}},
                {0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0});
}
TriangleMeshId plane(World &world) {
    return mesh(world, {{-10,0,-10},{10,0,-10},{10,0,10},{-10,0,10}}, {0,2,1,0,3,2});
}
RigidBodyId body(World &world, RigidBodyOptions options) {
    RigidBodyId id{}; require(world.add_rigid_body(options, id), "add body"); return id;
}
RigidBodyState state(World &world, RigidBodyId id) {
    RigidBodyState value{}; require(world.read_rigid_body_state(id, value), "read body"); return value;
}
void step(World &world, Vec3 gravity = {}, float dt = 1.0F/60.0F, std::uint32_t substeps = 4) {
    require(world.step({.timestep = dt, .substeps = substeps, .gravity = gravity,
                        .rigid_contact_pass_limit = pass_limit}), "step AVBD world");
}
std::uint32_t reported_passes(World &world) {
    WorldStatistics statistics{};
    require(world.collect_statistics(statistics), "collect AVBD iteration count");
    return statistics.rigid_contact_maximum_passes;
}
std::uint32_t phase_budget() { return pass_limit ? pass_limit : avbd::default_iterations; }

void test_stack() {
    World world;
    require(World::create({.rigid_body_capacity = 6, .triangle_mesh_capacity = 2}, world), "create stack world");
    const auto floor_mesh = plane(world), cube = box(world, 0.25F);
    (void)body(world, {.motion = MotionType::static_body, .mesh = floor_mesh});
    std::array<RigidBodyId, 5> boxes{};
    for (unsigned i = 0; i < boxes.size(); ++i)
        boxes[i] = body(world, {.mesh = cube, .initial_state = {.position = {0, 0.251F + 0.502F*i, 0}},
                               .linear_damping = 0, .angular_damping = 0});
    float minimum_clearance = 1, maximum_drift = 0, final_speed = 0;
    for (unsigned frame = 0; frame < 240; ++frame) {
        step(world, {0,-9.81F,0});
        for (unsigned i = 0; i < boxes.size(); ++i) {
            const auto s = state(world, boxes[i]);
            check(finite(s), "stack pose must remain finite");
            minimum_clearance = std::min(minimum_clearance, s.position.y - (0.25F + 0.5F*i));
            maximum_drift = std::max(maximum_drift, std::hypot(s.position.x, s.position.z));
            if (frame >= 220) final_speed = std::max(final_speed, length(s.linear_velocity));
        }
    }
    std::cout << "stack: clearance=" << minimum_clearance << " lateral=" << maximum_drift << " final_speed=" << final_speed
              << " passes=" << reported_passes(world) << '\n';
    check(minimum_clearance > -0.02F, "five-box stack must not compress or cross the floor by 2 cm");
    check(maximum_drift < 0.03F, "resting stack must not develop lateral drift");
    check(final_speed < 0.15F, "resting stack must settle rather than sustain motion");
}

void test_rotated_joint(RigidConstraintType type) {
    World world;
    require(World::create({.rigid_body_capacity = 2, .rigid_constraint_capacity = 1,
                          .triangle_mesh_capacity = 1}, world), "create rotated joint world");
    const auto cube = box(world);
    const Quaternion qa = rotation({0,0,1}, 0.4F), qb = rotation({0,0,1}, -0.7F);
    const Quaternion frame = product(rotation({0,1,0}, 0.5F), rotation({0,0,1}, 0.8F));
    const Vec3 pa{0,3,0}, pb{1,3,0.2F}, pivot{0.6F,3.2F,0.1F};
    const auto anchor = body(world, {.motion = MotionType::static_body, .mesh = cube,
                                    .initial_state = {.position = pa, .orientation = qa}});
    const auto moving = body(world, {.mesh = cube,
        .initial_state = {.position = pb, .orientation = qb, .angular_velocity = {0,0,2}},
        .linear_damping = 0, .angular_damping = 0});
    RigidConstraintOptions options{.type = type, .body_a = anchor, .body_b = moving,
        .local_anchor_a = rotate(conjugate(qa), sub(pivot,pa)),
        .local_anchor_b = rotate(conjugate(qb), sub(pivot,pb)),
        .local_orientation_a = product(conjugate(qa), frame),
        .local_orientation_b = product(conjugate(qb), frame)};
    RigidConstraintId joint{}; require(world.add_rigid_constraint(options, joint), "add rotated joint");
    float maximum_error = 0;
    RigidBodyState s{};
    for (unsigned i = 0; i < 120; ++i) {
        step(world);
        s = state(world, moving);
        maximum_error = std::max(maximum_error, length(sub(add(s.position, rotate(s.orientation, options.local_anchor_b)), pivot)));
    }
    const auto angular_error = rotation_difference(s.orientation, qb);
    std::cout << (type == RigidConstraintType::fixed ? "fixed" : "point") << ": anchor_error=" << maximum_error
              << " rotation=" << angular_error << " speed=" << length(s.linear_velocity) << '\n';
    check(finite(s) && maximum_error < 0.015F, "rotated local frames must keep off-center anchors coincident");
    if (type == RigidConstraintType::fixed) {
        check(angular_error < 0.02F, "fixed joint must preserve authored relative orientation");
        check(length(s.linear_velocity) < 0.02F && length(s.angular_velocity) < 0.02F,
              "fixed joint must remove forbidden motion");
    } else check(angular_error > 0.1F, "point joint must retain permitted rotation");
}

void test_spring() {
    for (float stiffness : {25.0F, 100.0F}) {
        World world;
        require(World::create({.rigid_body_capacity = 2, .rigid_constraint_capacity = 1,
                              .triangle_mesh_capacity = 1}, world), "create spring world");
        const auto cube = box(world);
        const auto anchor = body(world, {.motion = MotionType::static_body, .mesh = cube,
                                        .initial_state = {.position = {0,2,0}}});
        const auto moving = body(world, {.mesh = cube, .initial_state = {.position = {1,2,0}},
                                         .linear_damping = 0, .angular_damping = 0});
        RigidConstraintId joint{};
        require(world.add_rigid_constraint({.type = RigidConstraintType::generic_spring,
            .body_a = anchor, .body_b = moving, .local_anchor_a = {1,0,0},
            .linear_springs = {.axes = rigid_constraint_axis_y, .stiffness = {0,stiffness,0},
                              .damping = {0,2*std::sqrt(stiffness),0}}}, joint), "add compliant spring");
        for (unsigned i = 0; i < 120; ++i) step(world, {0,-10,0}, 1.0F/30.0F);
        const auto s = state(world, moving);
        const float sag = 2-s.position.y, expected = 10/stiffness;
        std::cout << "spring: stiffness=" << stiffness << " sag=" << sag << " expected=" << expected
                  << " velocity=" << s.linear_velocity.y << '\n';
        check(finite(s) && std::fabs(sag-expected) < 0.005F, "finite spring equilibrium must remain mg/k, not harden to a rigid joint");
        check(std::fabs(s.linear_velocity.y) < 0.01F, "critically damped spring must settle");
    }
}

void test_motor() {
    World world;
    require(World::create({.rigid_body_capacity = 2, .rigid_constraint_capacity = 1,
                          .triangle_mesh_capacity = 1}, world), "create motor world");
    const auto cube = box(world);
    const auto anchor = body(world, {.motion = MotionType::static_body, .mesh = cube});
    const auto initial = rotation({0,1,0}, 170*pi/180);
    const auto moving = body(world, {.mesh = cube, .initial_state = {.orientation = initial, .angular_velocity = {0,4,0}},
                                     .linear_damping = 0, .angular_damping = 0});
    const auto frame = rotation({0,0,1}, pi/2);
    RigidConstraintId joint{};
    require(world.add_rigid_constraint({.type = RigidConstraintType::motor, .body_a = anchor, .body_b = moving,
        .local_orientation_a = frame, .local_orientation_b = frame,
        .motor = {.angular_enabled = true, .angular_target_velocity = 4, .angular_maximum_impulse = 1}}, joint), "add rotated motor");
    float minimum_speed = 100, maximum_speed = 0, transverse = 0, angle = 0;
    Quaternion previous = initial;
    for (unsigned i = 0; i < 240; ++i) {
        step(world);
        const auto s = state(world, moving);
        auto delta = product(s.orientation, conjugate(previous));
        if (delta.w < 0) delta = {-delta.x,-delta.y,-delta.z,-delta.w};
        angle += 2*std::atan2(delta.y, delta.w);
        previous = s.orientation;
        minimum_speed = std::min(minimum_speed, s.angular_velocity.y);
        maximum_speed = std::max(maximum_speed, s.angular_velocity.y);
        transverse = std::max(transverse, std::hypot(s.angular_velocity.x, s.angular_velocity.z));
    }
    const auto passes = reported_passes(world);
    std::cout << "motor: angle=" << angle << " speed_range=" << minimum_speed << ',' << maximum_speed << " transverse=" << transverse
              << " passes=" << passes << '\n';
    check(angle > 15.0F && angle < 17.0F, "motor must drive multiple complete revolutions through the quaternion branch cut");
    check(minimum_speed > 3.8F && maximum_speed < 4.2F && transverse < 0.03F,
          "rotated motor must preserve target speed and axis across pi");
    check(passes == phase_budget(), "motion without impact must not run the impact correction phase");
}

RigidBodyState lifecycle_result(unsigned mode, bool warm, RigidBodyState &transition) {
    World world;
    require(World::create({.rigid_body_capacity = 2, .rigid_constraint_capacity = 1,
                          .triangle_mesh_capacity = 1}, world), "create lifecycle world");
    const auto cube = box(world);
    const auto anchor = body(world, {.motion = MotionType::static_body, .mesh = cube});
    const RigidBodyState initial = warm ? RigidBodyState{.position = {1,0,0}} : transition;
    const auto moving = body(world, {.mesh = cube, .initial_state = initial,
                                     .linear_damping = 0, .angular_damping = 0});
    RigidConstraintOptions options{.type = RigidConstraintType::fixed, .body_a = anchor, .body_b = moving,
                                    .local_anchor_a = {1,0,0}};
    RigidConstraintId joint{};
    if (warm) {
        require(world.add_rigid_constraint(options, joint), "add preload joint");
        for (unsigned i = 0; i < 60; ++i) step(world, {0,-10,0});
        transition = state(world, moving);
    }
    if (mode == 2) {
        require(world.set_rigid_body_state(anchor, {.position = {5,4,-2}}), "teleport anchor");
        require(world.set_rigid_body_state(moving, {.position = {6,4,-2}}), "teleport constrained body");
        if (!warm) require(world.add_rigid_constraint(options, joint), "add fresh teleported joint");
    } else {
        // Do not reset the body here: doing so would independently invalidate
        // its dual cache and hide a missing remove/update invalidation.
        options.type = RigidConstraintType::generic_spring;
        options.local_anchor_a = {1.25F,0,0};
        options.linear_springs = {.axes = rigid_constraint_axis_x, .stiffness = {25,0,0}, .damping = {2,0,0}};
        if (warm && mode == 1) require(world.update_rigid_constraint(joint, options), "update fixed to spring");
        else {
            RigidConstraintId removed = joint;
            if (warm) require(world.remove_rigid_constraint(joint), "remove preloaded joint");
            require(world.add_rigid_constraint(options, joint), "add reused spring");
            if (warm) check(removed.index == joint.index && removed.generation != joint.generation,
                            "constraint slot reuse must issue a new generation");
        }
    }
    for (unsigned i = 0; i < 10; ++i) step(world);
    return state(world, moving);
}
void test_lifecycle() {
    for (unsigned mode = 0; mode < 3; ++mode) {
        RigidBodyState transition{};
        const auto warm = lifecycle_result(mode, true, transition), fresh = lifecycle_result(mode, false, transition);
        const float position_error = length(sub(warm.position,fresh.position));
        const float velocity_error = length(sub(warm.linear_velocity,fresh.linear_velocity));
        std::cout << "lifecycle: mode=" << mode << " position_error=" << position_error << " velocity_error=" << velocity_error << '\n';
        check(finite(warm) && position_error < 1e-5F && velocity_error < 1e-4F,
              "joint reuse, update and body teleport must not retain old dual forces");
    }
}

void test_ccd() {
    for (const float speed : {12.0F, 120.0F}) for (const float restitution : {0.0F, 0.5F, 1.0F}) {
        World world;
        require(World::create({.rigid_body_capacity = 2, .triangle_mesh_capacity = 2}, world), "create CCD world");
        const auto floor_mesh = plane(world), cube = box(world);
        (void)body(world, {.motion = MotionType::static_body, .mesh = floor_mesh, .friction = 0, .restitution = restitution});
        const float initial_y = speed == 12 ? 0.25F : 1.5F;
        const auto moving = body(world, {.mesh = cube, .initial_state = {.position = {0,initial_y,0}, .linear_velocity = {0,-speed,0}},
                                         .friction = 0, .restitution = restitution, .linear_damping = 0,
                                         .angular_damping = 0, .maximum_linear_speed = 200});
        step(world, {}, 1.0F/60.0F, 1);
        const auto s = state(world, moving);
        const auto passes = reported_passes(world);
        std::cout << "ccd: speed=" << speed << " restitution=" << restitution << " y=" << s.position.y
                  << " velocity=" << s.linear_velocity.y << " expected=" << restitution*speed << " passes=" << passes << '\n';
        check(finite(s) && s.position.y >= 0.099F, "CCD must keep a crossing mesh on the incoming side of a thin plane");
        check(std::fabs(s.linear_velocity.y-restitution*speed) < std::max(0.2F, speed*0.02F),
              "CCD restitution must use impact velocity, independent of time of impact");
        check(passes == 2*phase_budget(), "impact statistics must include both primary and correction AVBD iterations");
    }
}

struct IslandResult {
    RigidBodyState contact{}, spring{};
    RigidConstraintState joint{};
    WorldStatistics statistics{};
};
IslandResult island_result(bool impact) {
    World world;
    require(World::create({.rigid_body_capacity = 5, .rigid_constraint_capacity = 1,
                          .triangle_mesh_capacity = 2}, world), "create independent impact islands");
    const auto floor_mesh = plane(world), cube = box(world);
    (void)body(world, {.motion = MotionType::static_body, .mesh = floor_mesh});
    const auto resting = body(world, {.mesh = cube,
        .initial_state = {.position = {-4,0.1F,0}, .linear_velocity = {0.2F,-0.05F,0}},
        .linear_damping = 0, .angular_damping = 0});
    const auto anchor = body(world, {.motion = MotionType::static_body, .mesh = cube,
        .initial_state = {.position = {-4,1,0}}});
    const auto spring = body(world, {.mesh = cube, .initial_state = {.position = {-3,1,0}},
        .linear_damping = 0, .angular_damping = 0});
    RigidConstraintId joint{};
    require(world.add_rigid_constraint({.type = RigidConstraintType::generic_spring,
        .body_a = anchor, .body_b = spring, .local_anchor_a = {0.75F,0,0},
        .linear_springs = {.axes = rigid_constraint_axis_x, .stiffness = {40,0,0}, .damping = {2,0,0}}}, joint),
        "add isolated spring");
    (void)body(world, {.mesh = cube, .initial_state = {.position = {4,0.25F,0},
        .linear_velocity = {0,impact ? -12.0F : 0.0F,0}}, .linear_damping = 0, .angular_damping = 0});
    step(world, {}, 1.0F/60.0F, 1);
    IslandResult result{state(world, resting), state(world, spring)};
    require(world.read_rigid_constraint_state(joint, result.joint), "read isolated spring impulse");
    require(world.collect_statistics(result.statistics), "read independent island statistics");
    return result;
}
void test_impact_islands() {
    const auto alone = island_result(false), impacted = island_result(true);
    const float contact_position = length(sub(alone.contact.position, impacted.contact.position));
    const float contact_velocity = length(sub(alone.contact.linear_velocity, impacted.contact.linear_velocity));
    const float spring_position = length(sub(alone.spring.position, impacted.spring.position));
    const float spring_velocity = length(sub(alone.spring.linear_velocity, impacted.spring.linear_velocity));
    const float spring_impulse = std::fabs(alone.joint.applied_impulse-impacted.joint.applied_impulse);
    std::cout << "impact-islands: contact_delta=" << contact_position << ',' << contact_velocity
              << " spring_delta=" << spring_position << ',' << spring_velocity << " impulse_delta=" << spring_impulse
              << " island_counts=" << alone.statistics.rigid_contact_island_count << ','
              << impacted.statistics.rigid_contact_island_count << '\n';
    check(contact_position < 1e-6F && contact_velocity < 1e-5F &&
          rotation_difference(alone.contact.orientation, impacted.contact.orientation) < 1e-6F,
          "an unrelated impact must not alter a quiet contact island sharing its static floor");
    check(spring_position < 1e-6F && spring_velocity < 1e-5F && spring_impulse < 1e-6F,
          "an unrelated impact must not change spring motion or its reported impulse");
    check(alone.statistics.rigid_contact_island_count == 2 && impacted.statistics.rigid_contact_island_count == 3,
          "static floor neighbors must not merge independent dynamic contact/joint islands");
}

void test_impact_spring() {
    World world;
    require(World::create({.rigid_body_capacity = 3, .rigid_constraint_capacity = 1,
                          .triangle_mesh_capacity = 2}, world), "create impacting spring world");
    const auto floor_mesh = plane(world), cube = box(world);
    (void)body(world, {.motion = MotionType::static_body, .mesh = floor_mesh, .friction = 0});
    const auto anchor = body(world, {.motion = MotionType::static_body, .mesh = cube,
        .initial_state = {.position = {0,0.25F,0}}});
    const auto moving = body(world, {.mesh = cube,
        .initial_state = {.position = {0.25F,0.25F,0}, .linear_velocity = {0,-12,0}},
        .friction = 0, .linear_damping = 0, .angular_damping = 0});
    RigidConstraintId joint{};
    require(world.add_rigid_constraint({.type = RigidConstraintType::generic_spring,
        .body_a = anchor, .body_b = moving,
        .linear_springs = {.axes = rigid_constraint_axis_x, .stiffness = {100,0,0}}}, joint), "add impacting spring");
    constexpr float dt = 1.0F/60.0F;
    require(world.step({.timestep = dt, .substeps = 1, .gravity = {},
        .collect_rigid_contacts = trace_contacts, .rigid_contact_pass_limit = pass_limit}), "step impacting spring");
    const auto s = state(world, moving);
    RigidConstraintState constraint{};
    require(world.read_rigid_constraint_state(joint, constraint), "read impacting spring diagnostic");
    const float expected_impulse = 100*s.position.x*dt;
    std::cout << "impact-spring: x=" << s.position.x << " vx=" << s.linear_velocity.x
              << " reported_impulse=" << constraint.applied_impulse << " expected=" << expected_impulse << '\n';
    if (trace_contacts) {
        const auto view = world.rigid_contacts();
        std::vector<RigidContactEvent> events(view.event_count);
        if (!events.empty()) require_cuda(cudaMemcpy(events.data(), view.events.data,
            events.size()*sizeof(RigidContactEvent), cudaMemcpyDeviceToHost), "read impacting spring contacts");
        std::cout << "impact-spring-state: position=" << s.position.x << ',' << s.position.y << ',' << s.position.z
                  << " velocity=" << s.linear_velocity.x << ',' << s.linear_velocity.y << ',' << s.linear_velocity.z
                  << " orientation=" << s.orientation.x << ',' << s.orientation.y << ',' << s.orientation.z << ',' << s.orientation.w
                  << " omega=" << s.angular_velocity.x << ',' << s.angular_velocity.y << ',' << s.angular_velocity.z
                  << " passes=" << reported_passes(world) << " contacts=" << events.size() << '\n';
        Vec3 contact_impulse{};
        for (const auto &event : events) {
            const auto impulse = add(mul(event.normal, event.normal_impulse), event.friction_impulse);
            if (event.body.index == moving.index) contact_impulse = add(contact_impulse, impulse);
            else if (event.collider.index == moving.index) contact_impulse = sub(contact_impulse, impulse);
            std::cout << "impact-spring-contact: body=" << event.body.index << " collider=" << event.collider.index
                      << " normal=" << event.normal.x << ',' << event.normal.y << ',' << event.normal.z
                      << " point=" << event.position.x << ',' << event.position.y << ',' << event.position.z
                      << " penetration=" << event.penetration << " impulse=" << event.normal_impulse
                      << " friction=" << event.friction_impulse.x << ',' << event.friction_impulse.y << ',' << event.friction_impulse.z << '\n';
        }
        std::cout << "impact-spring-contact-sum: " << contact_impulse.x << ',' << contact_impulse.y << ',' << contact_impulse.z << '\n';
    }
    check(finite(s) && std::fabs(s.linear_velocity.y) < 0.01F && s.position.y >= 0.099F,
          "spring-connected impact must still stabilize normal velocity");
    check(length(s.angular_velocity) < (phase_budget() >= 32 ? 0.01F : 0.5F),
          "a symmetric frictionless floor impact must not create large spurious spin");
    check(std::fabs(constraint.applied_impulse-expected_impulse) < 0.001F &&
          std::fabs(s.linear_velocity.x+expected_impulse) < 0.001F,
          "impact correction must retain a finite spring's physical force exactly once in motion and diagnostics");
}

void test_overlap() {
    for (float initial_y : {0.05F, 0.08F}) {
        World world;
        require(World::create({.rigid_body_capacity = 2, .triangle_mesh_capacity = 2}, world), "create overlap world");
        const auto floor_mesh = plane(world), cube = box(world);
        (void)body(world, {.motion = MotionType::static_body, .mesh = floor_mesh});
        const auto moving = body(world, {.mesh = cube, .initial_state = {.position = {0,initial_y,0}},
            .linear_damping = 0, .angular_damping = 0});
        float maximum_speed = 0, maximum_height = 0;
        RigidBodyState s{};
        for (unsigned frame = 0; frame < 30; ++frame) {
            const bool trace = trace_contacts && frame < 5;
            require(world.step({.timestep = 1.0F/60.0F, .substeps = 1, .gravity = {},
                .collect_rigid_contacts = trace, .rigid_contact_pass_limit = pass_limit}), "step pre-existing overlap");
            s = state(world, moving);
            maximum_speed = std::max(maximum_speed, length(s.linear_velocity));
            maximum_height = std::max(maximum_height, s.position.y);
            if (trace) {
                const auto view = world.rigid_contacts();
                std::vector<RigidContactEvent> events(view.event_count);
                if (!events.empty()) require_cuda(cudaMemcpy(events.data(), view.events.data,
                    events.size()*sizeof(RigidContactEvent), cudaMemcpyDeviceToHost), "read overlap contact diagnostics");
                std::cout << "overlap-frame: initial_y=" << initial_y << " frame=" << frame
                          << " position=" << s.position.x << ',' << s.position.y << ',' << s.position.z
                          << " velocity=" << s.linear_velocity.x << ',' << s.linear_velocity.y << ',' << s.linear_velocity.z
                          << " omega=" << s.angular_velocity.x << ',' << s.angular_velocity.y << ',' << s.angular_velocity.z
                          << " passes=" << reported_passes(world) << " contacts=" << events.size() << '\n';
                for (const auto &event : events)
                    std::cout << "overlap-contact: body=" << event.body.index << " collider=" << event.collider.index
                              << " normal=" << event.normal.x << ',' << event.normal.y << ',' << event.normal.z
                              << " point=" << event.position.x << ',' << event.position.y << ',' << event.position.z
                              << " penetration=" << event.penetration << " impulse=" << event.normal_impulse
                              << " friction=" << event.friction_impulse.x << ',' << event.friction_impulse.y << ',' << event.friction_impulse.z << '\n';
            }
        }
        std::cout << "overlap: initial_y=" << initial_y << " final_y=" << s.position.y
                  << " max_y=" << maximum_height << " max_speed=" << maximum_speed << '\n';
        check(finite(s) && s.position.y >= 0.099F, "zero-velocity overlap must recover nonpenetrating geometry");
        check(maximum_speed < 0.01F && maximum_height < 0.105F,
              "repairing a pre-existing overlap must not create ejection kinetic energy");
    }
}

void test_skin() {
    // Narrowphase constructs the rest offset from the SUM of pair margins.
    // Unequal and sub-millimeter margins must share that same solver contract.
    constexpr std::array<std::array<float,2>,4> margins{{
        {0.005F,0.005F}, {1e-6F,0.005F}, {0.005F,1e-6F}, {0.0002F,0.0003F}}};
    for (const auto &margin : margins)
    for (float initial_y : {0.1F, 0.1005F, 0.1015F}) for (unsigned iterations : {1U,4U,10U}) {
        World world;
        require(World::create({.rigid_body_capacity = 2, .triangle_mesh_capacity = 2}, world), "create stationary skin world");
        const auto floor_mesh = plane(world), cube = box(world);
        (void)body(world, {.motion = MotionType::static_body, .mesh = floor_mesh,
                          .collision_margin = margin[0]});
        const auto moving = body(world, {.mesh = cube, .initial_state = {.position = {0,initial_y,0}},
            .linear_damping = 0, .angular_damping = 0, .collision_margin = margin[1]});
        float drift = 0, speed = 0, angular_speed = 0;
        bool all_finite = true;
        for (unsigned frame = 0; frame < 30; ++frame) {
            require(world.step({.timestep = 1.0F/60.0F, .substeps = 1, .gravity = {},
                .rigid_contact_pass_limit = iterations}), "step stationary collision skin");
            const auto s = state(world, moving);
            all_finite = all_finite && finite(s);
            drift = std::max(drift, length(sub(s.position, {0,initial_y,0})));
            speed = std::max(speed, length(s.linear_velocity));
            angular_speed = std::max(angular_speed, length(s.angular_velocity));
        }
        std::cout << "skin: y=" << initial_y << " margins=" << margin[0] << ',' << margin[1]
                  << " iterations=" << iterations << " drift=" << drift << " speed=" << speed
                  << " spin=" << angular_speed << '\n';
        check(all_finite && drift < 1e-6F && speed < 1e-6F, "a stationary touching or separated body must not gain motion from collision skin");
        check(angular_speed < 1e-6F, "collision skin must not spin a stationary body with unequal pair margins");
        check(reported_passes(world) == iterations, "collision skin alone must not trigger an impact phase");
    }
}

struct MomentumResult {
    RigidBodyState a{}, b{};
    Vec3 momentum{}, center{};
};
MomentumResult momentum_result(unsigned iterations, bool supported_neighbor) {
    World world;
    require(World::create({.rigid_body_capacity = 4, .triangle_mesh_capacity = 2}, world), "create momentum world");
    const auto floor_mesh = plane(world), cube = box(world);
    (void)body(world, {.motion = MotionType::static_body, .mesh = floor_mesh});
    constexpr float mass_a = 1, mass_b = 3, dt = 1.0F/60.0F;
    constexpr Vec3 gravity{0.3F,-9.81F,0.1F};
    const RigidBodyState initial_a{.position = {-0.095F,2,0}, .linear_velocity = {2,0.25F,-0.3F}};
    const RigidBodyState initial_b{.position = {0.095F,2,0}, .linear_velocity = {-0.5F,-0.125F,0.1F}};
    const auto a = body(world, {.mesh = cube, .initial_state = initial_a, .mass = mass_a,
                                .linear_damping = 0, .angular_damping = 0});
    const auto b = body(world, {.mesh = cube, .initial_state = initial_b, .mass = mass_b,
                                .linear_damping = 0, .angular_damping = 0});
    // Keep the same live body count/IDs in both worlds. This box either has
    // no constraints, or belongs to a separate floor-supported island.
    (void)body(world, {.mesh = cube, .initial_state = {.position = {4,supported_neighbor ? 0.1F : 10.0F,0}},
                      .linear_damping = 0, .angular_damping = 0});
    require(world.step({.timestep = dt, .substeps = 1, .gravity = gravity,
                        .rigid_contact_pass_limit = iterations}), "step unequal-mass momentum collision");
    MomentumResult result{state(world,a), state(world,b)};
    result.momentum = add(mul(result.a.linear_velocity,mass_a), mul(result.b.linear_velocity,mass_b));
    result.center = mul(add(mul(result.a.position,mass_a), mul(result.b.position,mass_b)), 1/(mass_a+mass_b));
    const Vec3 initial_momentum = add(mul(initial_a.linear_velocity,mass_a), mul(initial_b.linear_velocity,mass_b));
    const Vec3 expected_momentum = add(initial_momentum, mul(gravity,(mass_a+mass_b)*dt));
    const Vec3 initial_center = mul(add(mul(initial_a.position,mass_a), mul(initial_b.position,mass_b)), 1/(mass_a+mass_b));
    const Vec3 expected_center = add(initial_center, mul(expected_momentum, dt/(mass_a+mass_b)));
    const Vec3 momentum_error = sub(result.momentum,expected_momentum);
    const float center_error = length(sub(result.center,expected_center));
    WorldStatistics statistics{};
    require(world.collect_statistics(statistics), "read momentum island statistics");
    std::cout << "momentum: iterations=" << iterations << " supported_neighbor=" << supported_neighbor
              << " error=" << momentum_error.x << ',' << momentum_error.y << ',' << momentum_error.z
              << " center_error=" << center_error << " islands=" << statistics.rigid_contact_island_count << '\n';
    check(finite(result.a) && finite(result.b), "unequal-mass collision must remain finite at every iteration budget");
    check(statistics.rigid_contact_live_pairs >= (supported_neighbor ? 2U : 1U) &&
          length(sub(result.a.linear_velocity,add(initial_a.linear_velocity,mul(gravity,dt)))) > 1e-3F,
          "momentum fixture must exercise a real dynamic collision response");
    check(length(momentum_error) < 1e-4F,
          "closed dynamic island must conserve linear momentum including external gravity at any iteration budget");
    check(center_error < 2e-6F, "internal collision corrections must preserve the island's inertial center-of-mass trajectory");
    return result;
}
void test_momentum() {
    for (unsigned iterations : {1U,4U,10U}) {
        const auto isolated = momentum_result(iterations, false);
        const auto neighbor = momentum_result(iterations, true);
        const float position_error = std::max(length(sub(isolated.a.position,neighbor.a.position)),
                                              length(sub(isolated.b.position,neighbor.b.position)));
        const float velocity_error = std::max(length(sub(isolated.a.linear_velocity,neighbor.a.linear_velocity)),
                                              length(sub(isolated.b.linear_velocity,neighbor.b.linear_velocity)));
        std::cout << "momentum-isolation: iterations=" << iterations << " position_delta=" << position_error
                  << " velocity_delta=" << velocity_error << '\n';
        check(position_error < 1e-6F && velocity_error < 1e-5F,
              "a separate floor-supported island must not change a closed dynamic island's momentum solve");
    }
}

struct ScheduleFrame {
    std::vector<RigidBodyState> bodies;
    std::vector<RigidConstraintState> joints;
    std::vector<RigidContactEvent> contacts;
    WorldStatistics statistics{};
};
ScheduleFrame schedule_frame(World &world, const std::vector<RigidBodyId> &bodies,
                             const std::vector<RigidConstraintId> &joints = {}) {
    ScheduleFrame result;
    for (const auto id : bodies) result.bodies.push_back(state(world,id));
    for (const auto id : joints) {
        RigidConstraintState value{};
        require(world.read_rigid_constraint_state(id,value), "read scheduler joint impulse");
        result.joints.push_back(value);
    }
    const auto view = world.rigid_contacts();
    result.contacts.resize(view.event_count);
    if (!result.contacts.empty()) require_cuda(cudaMemcpy(result.contacts.data(), view.events.data,
        result.contacts.size()*sizeof(RigidContactEvent), cudaMemcpyDeviceToHost), "read scheduler contact impulses");
    require(world.collect_statistics(result.statistics), "read scheduler graph statistics");
    return result;
}
void compare_schedule(const ScheduleFrame &reference, const ScheduleFrame &candidate,
                      bool same_graph = true) {
    check(reference.bodies.size() == candidate.bodies.size() && reference.joints.size() == candidate.joints.size()
        && reference.contacts.size() == candidate.contacts.size(), "scheduler cache must preserve diagnostic output counts");
    float position_error = 0, velocity_error = 0, angular_error = 0, impulse_error = 0, contact_error = 0;
    for (unsigned i = 0; i < std::min(reference.bodies.size(),candidate.bodies.size()); ++i) {
        const auto &a = reference.bodies[i], &b = candidate.bodies[i];
        check(finite(a) && finite(b), "cached and uncached scheduling must produce finite states");
        position_error = std::max(position_error,length(sub(a.position,b.position)));
        velocity_error = std::max({velocity_error,length(sub(a.linear_velocity,b.linear_velocity)),
                                  length(sub(a.angular_velocity,b.angular_velocity))});
        angular_error = std::max(angular_error,rotation_difference(a.orientation,b.orientation));
    }
    for (unsigned i = 0; i < std::min(reference.joints.size(),candidate.joints.size()); ++i) {
        const auto &a = reference.joints[i], &b = candidate.joints[i];
        check(a.enabled == b.enabled && a.broken == b.broken, "scheduler cache must preserve joint lifecycle state");
        impulse_error = std::max(impulse_error,std::fabs(a.applied_impulse-b.applied_impulse));
    }
    for (unsigned i = 0; i < std::min(reference.contacts.size(),candidate.contacts.size()); ++i) {
        const auto &a = reference.contacts[i], &b = candidate.contacts[i];
        check(a.body.index == b.body.index && a.collider.index == b.collider.index,
              "scheduler cache must preserve canonical contact order and endpoints");
        contact_error = std::max({contact_error,length(sub(a.position,b.position)),length(sub(a.normal,b.normal)),
                                 std::fabs(a.penetration-b.penetration)});
        impulse_error = std::max({impulse_error,std::fabs(a.normal_impulse-b.normal_impulse),
                                 length(sub(a.friction_impulse,b.friction_impulse))});
    }
    std::cout << "schedule-errors: position=" << position_error << " velocity=" << velocity_error
              << " angle=" << angular_error << " impulse=" << impulse_error << " contact=" << contact_error << '\n';
    check(position_error < 1e-6F && velocity_error < 1e-5F && angular_error < 1e-6F,
          "scheduler metadata placement must not change the physical body solve");
    check(impulse_error < 1e-6F && contact_error < 1e-6F,
          "scheduler metadata placement must not change joint or contact diagnostics");
    if (same_graph) {
        const auto &a = reference.statistics, &b = candidate.statistics;
        check(a.rigid_contact_island_count == b.rigid_contact_island_count &&
              a.rigid_contact_color_count == b.rigid_contact_color_count &&
              a.rigid_contact_live_pairs == b.rigid_contact_live_pairs &&
              a.rigid_contact_candidate_pairs == b.rigid_contact_candidate_pairs &&
              a.rigid_contact_maximum_passes == b.rigid_contact_maximum_passes &&
              a.rigid_contact_early_exit_count == b.rigid_contact_early_exit_count &&
              a.rigid_contact_overflow_pairs == b.rigid_contact_overflow_pairs,
              "cached and uncached scheduler metadata must yield identical graph statistics");
    }
}
std::vector<ScheduleFrame> schedule_assembly(unsigned body_padding, unsigned joint_padding) {
    World world;
    require(World::create({.rigid_body_capacity = body_padding+5,
        .rigid_constraint_capacity = joint_padding+1, .triangle_mesh_capacity = 2},world), "create scheduler boundary world");
    const auto floor_mesh = plane(world), cube = box(world);
    for (unsigned i = 0; i < body_padding; ++i)
        (void)body(world,{.motion = MotionType::static_body, .mesh = cube,
                         .initial_state = {.position = {100+0.5F*i,20,0}}});
    (void)body(world,{.motion = MotionType::static_body, .mesh = floor_mesh});
    const auto lower = body(world,{.mesh = cube, .initial_state = {.position = {0,0.1005F,0}},
                                  .linear_damping = 0, .angular_damping = 0});
    const auto upper = body(world,{.mesh = cube, .initial_state = {.position = {0.015F,0.301F,0},
                                  .linear_velocity = {0.12F,0,0}}, .linear_damping = 0, .angular_damping = 0});
    const auto anchor = body(world,{.motion = MotionType::static_body, .mesh = cube,
                                   .initial_state = {.position = {3,2,0}}});
    const auto moving = body(world,{.mesh = cube, .initial_state = {.position = {3.35F,2,0}},
                                   .linear_damping = 0, .angular_damping = 0});
    RigidConstraintOptions spring{.type = RigidConstraintType::generic_spring, .body_a = anchor, .body_b = moving,
        .local_anchor_a = {0.25F,0,0},
        .linear_springs = {.axes = rigid_constraint_axis_x, .stiffness = {80,0,0}, .damping = {3,0,0}}};
    spring.enabled = false;
    for (unsigned i = 0; i < joint_padding; ++i) {
        RigidConstraintId unused{};
        require(world.add_rigid_constraint(spring,unused), "pad disabled scheduler joints");
    }
    spring.enabled = true;
    RigidConstraintId joint{}; require(world.add_rigid_constraint(spring,joint), "add scheduler spring");
    std::vector<ScheduleFrame> result;
    for (unsigned frame = 0; frame < 3; ++frame) {
        require(world.step({.timestep = 1.0F/60.0F, .substeps = 1, .gravity = {0,-9.81F,0},
            .collect_rigid_contacts = true, .rigid_contact_pass_limit = pass_limit}), "step scheduler boundary assembly");
        auto snapshot = schedule_frame(world,{lower,upper,moving},{joint});
        for (auto &event : snapshot.contacts) {
            check(event.body.index >= body_padding && event.collider.index >= body_padding,
                  "far static scheduler padding must not introduce contacts");
            event.body.index -= body_padding;
            event.collider.index -= body_padding;
        }
        const auto &s = snapshot.statistics;
        std::cout << "schedule-body: padding=" << body_padding << " joint_slot=" << joint_padding << " frame=" << frame
                  << " colors=" << s.rigid_contact_color_count << " islands=" << s.rigid_contact_island_count
                  << " live=" << s.rigid_contact_live_pairs << '\n';
        check(s.rigid_contact_island_count == 2 && s.rigid_contact_color_count == 2 && s.rigid_contact_live_pairs >= 2,
              "scheduler boundary fixture must exercise connected contact vertices and a separate joint island");
        check(!snapshot.contacts.empty() && snapshot.joints[0].applied_impulse > 0,
              "scheduler boundary fixture must exercise real contact and joint diagnostics");
        result.push_back(std::move(snapshot));
    }
    return result; // World is destroyed before the next large padded world is created.
}
std::vector<ScheduleFrame> schedule_contact_prefix(unsigned moving_count) {
    constexpr unsigned floor_count = 47, dense_body_count = 22;
    World world;
    require(World::create({.rigid_body_capacity = floor_count+moving_count, .triangle_mesh_capacity = 2},world),
            "create contact-prefix scheduler world");
    const auto floor_mesh = plane(world), cube = box(world);
    // Redundant coplanar supports create >1024 real graph edges without
    // overlapping dynamic bodies or coupling otherwise independent islands.
    for (unsigned i = 0; i < floor_count; ++i)
        (void)body(world,{.motion = MotionType::static_body, .mesh = floor_mesh});
    RigidBodyId selected{};
    for (unsigned i = 0; i < moving_count; ++i) {
        const unsigned physical_index = moving_count == 1 ? dense_body_count-1 : i;
        selected = body(world,{.mesh = cube, .initial_state = {.position = {-4+0.35F*physical_index,0.1005F,0},
            .linear_velocity = {0.1F,-0.03F,0}}, .linear_damping = 0, .angular_damping = 0});
    }
    std::vector<ScheduleFrame> result;
    for (unsigned frame = 0; frame < 3; ++frame) {
        require(world.step({.timestep = 1.0F/60.0F, .substeps = 1, .gravity = {0,-9.81F,0},
            .collect_rigid_contacts = true, .rigid_contact_pass_limit = pass_limit}), "step contact-prefix scheduler world");
        auto snapshot = schedule_frame(world,{selected});
        std::erase_if(snapshot.contacts,[&](const auto &event) {
            return event.body.index != selected.index && event.collider.index != selected.index;
        });
        for (auto &event : snapshot.contacts) {
            if (event.body.index == selected.index) event.body.index = floor_count;
            if (event.collider.index == selected.index) event.collider.index = floor_count;
        }
        const auto &s = snapshot.statistics;
        std::cout << "schedule-contacts: movers=" << moving_count << " frame=" << frame
                  << " colors=" << s.rigid_contact_color_count << " islands=" << s.rigid_contact_island_count
                  << " live=" << s.rigid_contact_live_pairs << '\n';
        check(s.rigid_contact_live_pairs == floor_count*moving_count &&
              s.rigid_contact_candidate_pairs == floor_count*moving_count &&
              s.rigid_contact_island_count == moving_count && s.rigid_contact_color_count == 1,
              "contact-prefix fixture must retain every support edge without merging islands through static floors");
        check(!snapshot.contacts.empty(), "contact-prefix fixture must publish contact impulses");
        result.push_back(std::move(snapshot));
    }
    return result;
}
void test_schedule() {
    // CUDA caches body/contact[0,1024), joint[0,128); Metal uses 512/64.
    // These boundaries are storage choices only; traversal and solve order
    // agree. The public-API fixture also records Metal's smaller boundaries
    // for reuse when this CUDA-upload test harness gains a Metal runner.
    const auto reference = schedule_assembly(0,0);
    for (const auto padding : std::array<std::array<unsigned,2>,8>{{
            {509,0},{512,0},{1021,0},{1024,0},{0,63},{0,64},{0,127},{0,128}}}) {
        const auto candidate = schedule_assembly(padding[0],padding[1]);
        for (unsigned frame = 0; frame < reference.size(); ++frame)
            compare_schedule(reference[frame],candidate[frame]);
    }
    const auto isolated = schedule_contact_prefix(1), dense = schedule_contact_prefix(22);
    for (unsigned frame = 0; frame < isolated.size(); ++frame)
        compare_schedule(isolated[frame],dense[frame],false);
}
} // namespace

int main(int argc, char **argv) {
    const char *selected = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--case") == 0 && i+1 < argc) selected = argv[++i];
        else if (std::strcmp(argv[i], "--passes") == 0 && i+1 < argc) pass_limit = std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--trace") == 0) trace_contacts = true;
        else { std::cerr << "usage: avbd-rigid-tests [--case stack|fixed|point|spring|motor|lifecycle|ccd|impact-islands|impact-spring|overlap|skin|momentum|schedule] [--passes N] [--trace]\n"; return 2; }
    }
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    bool matched = false;
    const auto run = [&](const char *name, auto test) {
        if (!selected || std::strcmp(selected,name) == 0) { matched = true; test(); }
    };
    run("stack", test_stack);
    run("fixed", [] { test_rotated_joint(RigidConstraintType::fixed); });
    run("point", [] { test_rotated_joint(RigidConstraintType::point); });
    run("spring", test_spring);
    run("motor", test_motor);
    run("lifecycle", test_lifecycle);
    run("ccd", test_ccd);
    run("impact-islands", test_impact_islands);
    run("impact-spring", test_impact_spring);
    run("overlap", test_overlap);
    run("skin", test_skin);
    run("momentum", test_momentum);
    run("schedule", test_schedule);
    if (!matched) { std::cerr << "unknown AVBD case\n"; return 2; }
    std::cout << "AVBD production regression failures: " << failures << '\n';
    return failures ? 1 : 0;
}
