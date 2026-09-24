// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(parallel_mater::Status status, const char *message) {
    if (!status) {
        std::cerr << "FAIL: " << message << ": "
                  << (status.message ? status.message : "unknown") << '\n';
        ++failures;
    }
}

template <class T>
T *upload(const T *source, std::size_t count) {
    T *device = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device), count * sizeof(T)) !=
            cudaSuccess ||
        cudaMemcpy(device, source, count * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        ++failures;
        return nullptr;
    }
    return device;
}

} // namespace

int main() {
    using namespace parallel_mater;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    // Exact two-particle brute-force reference: q=1-0.5/1=0.5,
    // acceleration=20*q*q=5, so one 0.1 s step changes speed by 0.5.
    World pair_world;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, pair_world),
          "create pair world");
    const std::array<FluidParticle, 2> pair{{
        {{0.0F, 0.0F, 0.0F}, {}},
        {{0.5F, 0.0F, 0.0F}, {}}}};
    FluidParticle *pair_input = upload(pair.data(), pair.size());
    FluidId pair_id{};
    check(pair_world.add_fluid({.capacity = 4U,
                                .particle_radius = 0.1F,
                                .support_radius = 1.0F,
                                .solver_iterations = 1U,
                                .repulsion = 20.0F,
                                .viscosity = 0.0F,
                                .velocity_damping = 0.0F,
                                .maximum_speed = 100.0F},
                               {pair_input, pair.size()}, pair_id),
          "add pair fluid");
    cudaFree(pair_input);
    check(pair_world.step({.timestep = 0.1F, .substeps = 1U,
                           .gravity = {}, .collect_kernel_timings = true}),
          "step pair fluid");
    WorldStepTimings pair_timings{};
    check(pair_world.collect_step_timings(pair_timings),
          "collect fluid stage timings");
    check(pair_timings.available && pair_timings.total_gpu_milliseconds > 0.0F &&
              pair_timings.fluid_neighbor_sort.launch_count == 1U &&
              pair_timings.fluid_neighbor_forces.launch_count == 1U &&
              pair_timings.fluid_integration.launch_count == 1U,
          "fluid timings identify sort, neighbor and integration stages");
    FluidDeviceView pair_view{};
    check(pair_world.fluid_view(pair_id, pair_view), "view pair fluid");
    check(pair_view.particle_count == 2U, "pair retains two particles");
    std::array<Vec3, 2> pair_positions{}, pair_velocities{};
    if (pair_view.particle_count == 2U) {
        check(cudaMemcpy(pair_positions.data(), pair_view.positions.data,
                         sizeof(pair_positions), cudaMemcpyDeviceToHost) ==
                  cudaSuccess, "read pair positions");
        check(cudaMemcpy(pair_velocities.data(), pair_view.velocities.data,
                         sizeof(pair_velocities), cudaMemcpyDeviceToHost) ==
                  cudaSuccess, "read pair velocities");
        check(std::fabs(pair_velocities[0].x + 0.5F) < 0.02F &&
              std::fabs(pair_velocities[1].x - 0.5F) < 0.02F &&
              std::fabs(pair_positions[0].x + 0.05F) < 0.02F &&
              std::fabs(pair_positions[1].x - 0.55F) < 0.02F,
              "sorted-cell repulsion matches brute-force pair");
    }

    World overflow_world;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, overflow_world),
          "create neighbor-overflow world");
    const std::array<FluidParticle, 3> close_particles{{
        {{0.0F, 0.0F, 0.0F}, {}},
        {{0.2F, 0.0F, 0.0F}, {}},
        {{0.4F, 0.0F, 0.0F}, {}}}};
    FluidParticle *close_input = upload(close_particles.data(),
                                        close_particles.size());
    FluidId overflow_id{};
    check(overflow_world.add_fluid({.capacity = 3U,
                                    .particle_radius = 0.1F,
                                    .support_radius = 1.0F,
                                    .solver_iterations = 1U,
                                    .maximum_neighbors = 1U},
                                   {close_input, close_particles.size()},
                                   overflow_id), "add dense fluid");
    cudaFree(close_input);
    check(overflow_world.step({.timestep = 0.01F,
                               .substeps = 1U, .gravity = {}}).code ==
              StatusCode::capacity_exceeded,
          "neighbor overflow must be reported without silent truncation");

    World invalid_world;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, invalid_world),
          "create invalid-input world");
    const FluidParticle nonfinite{
        {std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F}, {}};
    FluidParticle *nonfinite_input = upload(&nonfinite, 1U);
    FluidId invalid_id{};
    check(invalid_world.add_fluid({.capacity = 1U},
                                  {nonfinite_input, 1U}, invalid_id).code ==
              StatusCode::invalid_argument,
          "nonfinite device-resident initial fluid must be rejected");
    cudaFree(nonfinite_input);

    World lifecycle;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, lifecycle),
          "create lifecycle world");
    FluidId lifecycle_id{};
    check(lifecycle.add_fluid({.capacity = 32U,
                               .particle_radius = 0.05F,
                               .support_radius = 0.2F,
                               .solver_iterations = 1U}, {}, lifecycle_id),
          "add empty lifecycle fluid");
    ParticleSpawnPlaneId spawn_id{};
    ParticleSpawnPlaneOptions spawn{
        .fluid = lifecycle_id,
        .plane = {.half_extents = {0.5F, 0.5F}},
        .particles_per_second = 120.0F,
        .initial_velocity = {0.0F, -1.0F, 0.0F}};
    check(lifecycle.add_particle_spawn_plane(spawn, spawn_id),
          "add spawn plane");
    ParticleDestroyPlaneId destroy_id{};
    check(lifecycle.add_particle_destroy_plane(
        {.fluid = lifecycle_id,
         .plane = {.center = {0.0F, -0.05F, 0.0F},
                   .half_extents = {1.0F, 1.0F}},
         .crossing = CrossingDirection::against_normal}, destroy_id),
          "add destroy plane");
    for (int frame = 0; frame < 6; ++frame)
        check(lifecycle.step({.timestep = 1.0F / 60.0F,
                              .substeps = 1U, .gravity = {}}),
              "step lifecycle fluid");
    WorldStatistics lifecycle_stats{};
    check(lifecycle.collect_statistics(lifecycle_stats),
          "collect lifecycle statistics");
    check(lifecycle_stats.emitted_particle_count == 12U &&
          lifecycle_stats.destroyed_particle_count > 0U,
          "inflow and swept outflow update lifecycle counters");
    spawn.enabled = false;
    check(lifecycle.update_particle_spawn_plane(spawn_id, spawn),
          "disable lifecycle inflow");
    for (int frame = 0; frame < 8; ++frame)
        check(lifecycle.step({.timestep = 1.0F / 60.0F,
                              .substeps = 1U, .gravity = {}}),
              "drain lifecycle fluid");
    check(lifecycle.collect_statistics(lifecycle_stats),
          "collect drained lifecycle statistics");
    check(lifecycle_stats.particle_count == 0U &&
          lifecycle_stats.destroyed_particle_count == 12U,
          "stable compaction drains every emitted particle");

    World contact_world;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, contact_world),
          "create static-contact world");
    const std::array<Vec3, 4> vertices{{
        {-2.0F, 0.0F, -2.0F}, {2.0F, 0.0F, -2.0F},
        {2.0F, 0.0F, 2.0F}, {-2.0F, 0.0F, 2.0F}}};
    const std::array<std::uint32_t, 6> triangles{{0U, 1U, 2U, 0U, 2U, 3U}};
    Vec3 *device_vertices = upload(vertices.data(), vertices.size());
    std::uint32_t *device_triangles = upload(triangles.data(), triangles.size());
    TriangleMeshId mesh{};
    check(contact_world.add_triangle_mesh(
        {device_vertices, vertices.size()},
        {device_triangles, triangles.size()}, mesh), "add passive triangles");
    cudaFree(device_vertices);
    cudaFree(device_triangles);
    RigidBodyId body{};
    check(contact_world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = mesh}, body),
          "add passive collider");
    const FluidParticle drop{{0.0F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}};
    FluidParticle *drop_input = upload(&drop, 1U);
    FluidId drop_id{};
    check(contact_world.add_fluid({.capacity = 2U,
                                   .particle_radius = 0.1F,
                                   .support_radius = 0.2F,
                                   .solver_iterations = 1U,
                                   .maximum_speed = 100.0F},
                                  {drop_input, 1U}, drop_id),
          "add falling particle");
    cudaFree(drop_input);
    check(contact_world.step({.timestep = 0.1F, .substeps = 1U,
                              .gravity = {}}), "step swept passive contact");
    FluidDeviceView drop_view{};
    check(contact_world.fluid_view(drop_id, drop_view),
          "view contacted particle");
    Vec3 final_position{};
    float final_foam = 0.0F;
    check(cudaMemcpy(&final_position, drop_view.positions.data,
                     sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read contacted position");
    check(cudaMemcpy(&final_foam, drop_view.foam.data,
                     sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read contacted foam");
    check(final_position.y >= 0.099F && final_foam > 0.5F,
          "swept triangle contact prevents tunneling and emits foam");
    ParticleSpawnPlaneId embedded_spawn{};
    check(contact_world.add_particle_spawn_plane(
        {.fluid = drop_id,
         .plane = {.center = {0.0F, -0.02F, 0.0F},
                   .half_extents = {0.01F, 0.01F}},
         .particles_per_second = 10.0F}, embedded_spawn),
          "add slightly embedded inflow");
    check(contact_world.step({.timestep = 0.1F, .substeps = 1U,
                              .gravity = {}}),
          "step embedded inflow against passive triangles");
    check(contact_world.fluid_view(drop_id, drop_view),
          "view recovered inflow particle");
    Vec3 recovered_position{};
    check(drop_view.particle_count == 2U &&
              cudaMemcpy(&recovered_position, drop_view.positions.data + 1U,
                         sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read recovered inflow particle");
    check(recovered_position.y >= 0.099F,
          "embedded inflow starts above floor regardless of triangle winding");

    // Repeated resting contacts in four substeps retain one event, not four.
    World resting_events;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U,
                         .contact_capacity = 1U}, resting_events),
          "create per-frame contact world");
    device_vertices = upload(vertices.data(), vertices.size());
    device_triangles = upload(triangles.data(), triangles.size());
    TriangleMeshId resting_mesh{};
    check(resting_events.add_triangle_mesh(
        {device_vertices, vertices.size()},
        {device_triangles, triangles.size()}, resting_mesh),
        "add resting triangle mesh");
    cudaFree(device_vertices);
    cudaFree(device_triangles);
    RigidBodyId resting_body{};
    check(resting_events.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = resting_mesh},
        resting_body), "add resting floor");
    const FluidParticle resting_particle{{0.0F, 0.1F, 0.0F}, {}};
    FluidParticle *resting_input = upload(&resting_particle, 1U);
    FluidId resting_fluid{};
    check(resting_events.add_fluid({.capacity = 1U,
                                    .particle_radius = 0.1F,
                                    .support_radius = 0.2F,
                                    .solver_iterations = 1U},
                                   {resting_input, 1U}, resting_fluid),
          "add resting particle");
    cudaFree(resting_input);
    check(resting_events.step({.timestep = 0.1F, .substeps = 4U,
                               .collect_fluid_contacts = true}),
          "step four resting contacts");
    check(resting_events.contacts().event_count == 1U &&
          resting_events.contacts().overflowed == 0U,
          "one particle produces at most one event per frame");

    // A swept particle impact transfers equal and opposite normal impulse
    // to a dynamic triangle body, without an analytic shape shortcut.
    World coupled_world;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, coupled_world),
          "create coupled world");
    Vec3 *coupled_vertices = upload(vertices.data(), vertices.size());
    std::uint32_t *coupled_indices = upload(triangles.data(), triangles.size());
    TriangleMeshId coupled_mesh{};
    check(coupled_world.add_triangle_mesh(
        {coupled_vertices, vertices.size()},
        {coupled_indices, triangles.size()}, coupled_mesh),
        "add dynamic triangle mesh");
    cudaFree(coupled_vertices);
    cudaFree(coupled_indices);
    RigidBodyId coupled_body{};
    check(coupled_world.add_rigid_body(
        {.motion = MotionType::dynamic, .mesh = coupled_mesh,
         .linear_damping = 0.0F, .angular_damping = 0.0F},
        coupled_body), "add dynamic triangle body");
    FluidParticle *coupled_input = upload(&drop, 1U);
    FluidId coupled_fluid{};
    check(coupled_world.add_fluid({.capacity = 1U,
                                   .particle_radius = 0.1F,
                                   .rest_density = 125.0F,
                                   .support_radius = 0.2F,
                                   .solver_iterations = 1U,
                                   .velocity_damping = 0.0F,
                                   .maximum_speed = 100.0F},
                                  {coupled_input, 1U}, coupled_fluid),
          "add unit-mass impact particle");
    cudaFree(coupled_input);
    check(coupled_world.step({.timestep = 0.1F, .substeps = 1U,
                              .gravity = {}, .collect_kernel_timings = true,
                              .collect_fluid_contacts = true}),
          "step dynamic fluid impact");
    WorldStepTimings coupled_timings{};
    check(coupled_world.collect_step_timings(coupled_timings),
          "collect moving contact timing");
    check(coupled_timings.fluid_moving_contacts.launch_count == 1U,
          "moving contacts have their own timing stage");
    FluidDeviceView coupled_view{};
    check(coupled_world.fluid_view(coupled_fluid, coupled_view),
          "view coupled fluid");
    Vec3 coupled_velocity{};
    check(cudaMemcpy(&coupled_velocity, coupled_view.velocities.data,
                     sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read coupled particle velocity");
    RigidBodyState coupled_state{};
    check(coupled_world.read_rigid_body_state(coupled_body, coupled_state),
          "read coupled body state");
    check(coupled_velocity.y > -4.0F && coupled_state.linear_velocity.y < -1.0F &&
          std::fabs(coupled_velocity.y + coupled_state.linear_velocity.y + 5.0F)
              < 0.05F,
          "particle and body exchange balanced normal momentum");
    const ContactDeviceView coupled_events = coupled_world.contacts();
    ContactEvent coupled_event{};
    check(coupled_events.event_count == 1U && !coupled_events.overflowed &&
          cudaMemcpy(&coupled_event, coupled_events.events.data,
                     sizeof(ContactEvent), cudaMemcpyDeviceToHost) == cudaSuccess,
          "retain deterministic fluid-rigid impact event");
    check(coupled_event.fluid == coupled_fluid &&
          coupled_event.rigid_body == coupled_body &&
          coupled_event.stable_particle_id == 0U &&
          coupled_event.normal.y > 0.9F &&
          coupled_event.normal_impulse > 1.0F,
          "contact event names the particle, body, normal, and impulse");

    World limited_events;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U,
                         .contact_capacity = 1U}, limited_events),
          "create limited-contact world");
    coupled_vertices = upload(vertices.data(), vertices.size());
    coupled_indices = upload(triangles.data(), triangles.size());
    TriangleMeshId limited_mesh{};
    check(limited_events.add_triangle_mesh(
        {coupled_vertices, vertices.size()},
        {coupled_indices, triangles.size()}, limited_mesh),
        "add limited-contact triangle mesh");
    cudaFree(coupled_vertices);
    cudaFree(coupled_indices);
    RigidBodyId limited_body{};
    check(limited_events.add_rigid_body(
        {.motion = MotionType::dynamic, .mesh = limited_mesh}, limited_body),
        "add limited-contact body");
    const std::array<FluidParticle, 2> twin_impacts{{
        {{-1.0F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}},
        {{1.0F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}}}};
    FluidParticle *twin_input = upload(twin_impacts.data(), twin_impacts.size());
    FluidId twin_fluid{};
    check(limited_events.add_fluid({.capacity = 2U,
                                    .particle_radius = 0.1F,
                                    .rest_density = 125.0F,
                                    .support_radius = 0.2F,
                                    .solver_iterations = 1U,
                                    .maximum_speed = 100.0F},
                                   {twin_input, twin_impacts.size()}, twin_fluid),
          "add twin impacts");
    cudaFree(twin_input);
    check(limited_events.step({.timestep = 0.1F, .substeps = 1U,
                               .gravity = {}, .collect_fluid_contacts = true}),
          "step overflowing contacts");
    const ContactDeviceView limited_view = limited_events.contacts();
    ContactEvent retained{};
    check(limited_view.event_count == 1U && limited_view.overflowed &&
          cudaMemcpy(&retained, limited_view.events.data,
                     sizeof(ContactEvent), cudaMemcpyDeviceToHost) == cudaSuccess &&
          retained.stable_particle_id == 0U,
          "contact capacity retains first stable particle and reports overflow");
    WorldStatistics limited_stats{};
    check(limited_events.collect_statistics(limited_stats) &&
          limited_stats.contact_count == 1U &&
          limited_stats.contact_overflow_count == 1U,
          "contact statistics report retained and dropped events");

    World moving_plane;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, moving_plane),
          "create moving-plane world");
    coupled_vertices = upload(vertices.data(), vertices.size());
    coupled_indices = upload(triangles.data(), triangles.size());
    TriangleMeshId moving_mesh{};
    check(moving_plane.add_triangle_mesh(
        {coupled_vertices, vertices.size()},
        {coupled_indices, triangles.size()}, moving_mesh),
        "add kinematic triangle mesh");
    cudaFree(coupled_vertices);
    cudaFree(coupled_indices);
    RigidBodyId moving_body{};
    check(moving_plane.add_rigid_body(
        {.motion = MotionType::kinematic, .mesh = moving_mesh,
         .initial_state = {.position = {0.0F, 1.0F, 0.0F}}},
        moving_body), "add kinematic triangle body");
    const FluidParticle stationary{{0.0F, 0.2F, 0.0F}, {}};
    FluidParticle *stationary_input = upload(&stationary, 1U);
    FluidId stationary_fluid{};
    check(moving_plane.add_fluid({.capacity = 1U,
                                  .particle_radius = 0.1F,
                                  .support_radius = 0.2F,
                                  .solver_iterations = 1U,
                                  .maximum_speed = 100.0F},
                                 {stationary_input, 1U}, stationary_fluid),
          "add stationary particle");
    cudaFree(stationary_input);
    check(moving_plane.set_kinematic_target(moving_body,
        {.position = {0.0F, -1.0F, 0.0F}}), "sweep kinematic plane");
    check(moving_plane.step({.timestep = 0.1F, .substeps = 1U,
                             .gravity = {}, .collect_fluid_contacts = true}),
          "step moving plane through particle");
    FluidDeviceView stationary_view{};
    check(moving_plane.fluid_view(stationary_fluid, stationary_view),
          "view kinematic-contact particle");
    Vec3 swept_position{}, swept_velocity{};
    check(cudaMemcpy(&swept_position, stationary_view.positions.data,
                     sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess &&
          cudaMemcpy(&swept_velocity, stationary_view.velocities.data,
                     sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read kinematic-contact particle");
    check(swept_position.y < -1.05F && swept_velocity.y < -10.0F &&
          moving_plane.contacts().event_count == 1U,
          "swept moving triangle carries a particle and reports contact");

    // The same authored-triangle path must work on curved fixtures; there is
    // no special analytic-cylinder collision API in ParallelMater.
    constexpr std::uint32_t sides = 16U;
    constexpr float pi = 3.14159265358979323846F;
    std::vector<Vec3> cylinder_vertices;
    std::vector<std::uint32_t> cylinder_indices;
    for (std::uint32_t side = 0U; side < sides; ++side) {
        const float angle = 2.0F * pi * side / sides;
        cylinder_vertices.push_back({std::cos(angle), -1.0F,
                                     std::sin(angle)});
        cylinder_vertices.push_back({std::cos(angle), 1.0F,
                                     std::sin(angle)});
    }
    for (std::uint32_t side = 0U; side < sides; ++side) {
        const std::uint32_t a = side * 2U;
        const std::uint32_t b = ((side + 1U) % sides) * 2U;
        cylinder_indices.insert(cylinder_indices.end(),
                                {a, b, b + 1U, a, b + 1U, a + 1U});
    }
    World cylinder_world;
    check(World::create({.rigid_body_capacity = 1U,
                         .triangle_mesh_capacity = 1U}, cylinder_world),
          "create triangle-cylinder world");
    Vec3 *cylinder_device_vertices = upload(cylinder_vertices.data(),
                                            cylinder_vertices.size());
    std::uint32_t *cylinder_device_indices = upload(
        cylinder_indices.data(), cylinder_indices.size());
    TriangleMeshId cylinder_mesh{};
    check(cylinder_world.add_triangle_mesh(
        {cylinder_device_vertices, cylinder_vertices.size()},
        {cylinder_device_indices, cylinder_indices.size()}, cylinder_mesh),
          "add passive triangle cylinder");
    cudaFree(cylinder_device_vertices);
    cudaFree(cylinder_device_indices);
    RigidBodyId cylinder_body{};
    check(cylinder_world.add_rigid_body(
        {.motion = MotionType::static_body, .mesh = cylinder_mesh},
        cylinder_body), "add passive cylinder body");
    const FluidParticle cylinder_drop{{1.3F, 0.0F, 0.0F},
                                      {-5.0F, 0.0F, 0.0F}};
    FluidParticle *cylinder_input = upload(&cylinder_drop, 1U);
    FluidId cylinder_fluid{};
    check(cylinder_world.add_fluid({.capacity = 1U,
                                    .particle_radius = 0.1F,
                                    .support_radius = 0.2F,
                                    .solver_iterations = 1U,
                                    .maximum_speed = 100.0F},
                                   {cylinder_input, 1U}, cylinder_fluid),
          "add particle aimed at cylinder");
    cudaFree(cylinder_input);
    check(cylinder_world.step({.timestep = 0.1F, .substeps = 1U,
                               .gravity = {}}),
          "step passive triangle-cylinder impact");
    FluidDeviceView cylinder_view{};
    check(cylinder_world.fluid_view(cylinder_fluid, cylinder_view),
          "view cylinder-contact particle");
    Vec3 cylinder_result{};
    check(cudaMemcpy(&cylinder_result, cylinder_view.positions.data,
                     sizeof(Vec3), cudaMemcpyDeviceToHost) == cudaSuccess,
          "read cylinder-contact position");
    check(cylinder_result.x > 1.05F,
          "particle must not tunnel through authored cylinder triangles");

    return failures == 0 ? 0 : 1;
}
