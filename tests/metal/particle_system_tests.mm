// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace parallel_mater::metal;

namespace {
bool require(bool condition, const char *message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

template <typename T> const T *contents(BufferSpan<const T> span) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)span.buffer;
    return reinterpret_cast<const T *>(
        static_cast<const std::byte *>(buffer.contents) + span.byte_offset);
}

bool finite(Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}
} // namespace

int main() {
    WorldOptions capacities{};
    capacities.rigid_body_capacity = 2U;
    capacities.triangle_mesh_capacity = 1U;
    World world;
    Status status = World::create(capacities, world);
    if (!require(status.ok(), status.message ? status.message : "World create failed"))
        return 1;

    std::array<FluidParticle, 2> water{{
        {{-0.03F, 1.0F, 0.0F}, {}, 20.0F},
        {{0.03F, 1.0F, 0.0F}, {}, 20.0F},
    }};
    FluidId fluid{};
    status = world.add_fluid(
        {.capacity = 8U,
         .particle_radius = 0.03F,
         .support_radius = 0.12F,
         .repulsion = 20.0F,
         .maximum_speed = 10.0F},
        HostSpan<const FluidParticle>{water.data(), water.size()}, fluid);
    if (!require(status.ok(), "Fluid creation failed")) return 1;

    SmokeId smoke{};
    status = world.add_smoke(
        {.capacity = 32U,
         .emitter_center = {0.0F, 0.0F, 0.0F},
         .initial_velocity = {0.5F, 0.0F, 0.0F},
         .wind = {0.5F, 0.0F, 0.0F},
         .particles_per_second = 60.0F,
         .lifetime = 2.0F,
         .grid_resolution = 16U,
         .grid_vertical_resolution = 8U,
         .grid_pressure_iterations = 4U,
         .grid_minimum = {-0.25F, -1.0F, -1.0F},
         .grid_edge_length = 2.0F},
        smoke);
    if (!require(status.ok(), "Smoke creation failed")) return 1;

    std::array<Vec3, 4> cloth_vertices{{
        {-1.0F, 0.0F, -1.0F}, {3.0F, 0.0F, -1.0F},
        {-1.0F, 4.0F, -1.0F}, {-1.0F, 0.0F, 3.0F},
    }};
    std::array<std::uint32_t, 12> cloth_indices{{
        0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
    }};
    std::array<float, 4> cloth_inverse{{0.0F, 0.0F, 0.0F, 0.0F}};
    ClothId cloth{};
    status = world.add_cloth(
        {.vertices = {cloth_vertices.data(), cloth_vertices.size()},
         .triangle_indices = {cloth_indices.data(), cloth_indices.size()},
         .inverse_masses = {cloth_inverse.data(), cloth_inverse.size()},
         .solver_iterations = 8U,
         .preserve_volume = true},
        cloth);
    if (!require(status.ok(), "Cloth creation failed")) return 1;

    std::array<Vec3, 4> soft_nodes{{
        {2.0F, 1.0F, 0.0F}, {2.3F, 1.0F, 0.0F},
        {2.0F, 1.3F, 0.0F}, {2.0F, 1.0F, 0.3F},
    }};
    const auto distance = [&](std::uint32_t a, std::uint32_t b) {
        const Vec3 d{soft_nodes[b].x - soft_nodes[a].x,
                     soft_nodes[b].y - soft_nodes[a].y,
                     soft_nodes[b].z - soft_nodes[a].z};
        return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    };
    std::array<SoftBodyBond, 6> soft_bonds{{
        {0U, 1U, distance(0U, 1U)}, {0U, 2U, distance(0U, 2U)},
        {0U, 3U, distance(0U, 3U)}, {1U, 2U, distance(1U, 2U)},
        {1U, 3U, distance(1U, 3U)}, {2U, 3U, distance(2U, 3U)},
    }};
    std::array<SoftBodySurfaceBinding, 4> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].nodes[0] = index;
        bindings[index].weights[0] = 1.0F;
    }
    std::array<std::uint32_t, 12> soft_indices{{
        0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
    }};
    SoftBodyId soft{};
    status = world.add_soft_body(
        {.nodes = {soft_nodes.data(), soft_nodes.size()},
         .bonds = {soft_bonds.data(), soft_bonds.size()},
         .surface_vertices = {soft_nodes.data(), soft_nodes.size()},
         .surface_triangle_indices = {soft_indices.data(), soft_indices.size()},
         .surface_bindings = {bindings.data(), bindings.size()},
         .solver_iterations = 8U},
        soft);
    if (!require(status.ok(), "Soft-body creation failed")) return 1;

    std::array<Vec3, 2> rope_line{{cloth_vertices[0],
                                   {-2.0F, 0.0F, -1.0F}}};
    RopeId rope{};
    status = world.add_rope(
        {.centerline = {rope_line.data(), rope_line.size()},
         .node_spacing = 0.04F,
         .radius = 0.02F,
         .solver_iterations = 16U},
        rope);
    if (!require(status.ok(), status.message ? status.message : "Rope creation failed"))
        return 1;

    FluidSmokeCouplingId fluid_smoke{};
    status = world.add_fluid_smoke_coupling(
        {.fluid = fluid, .smoke = smoke}, fluid_smoke);
    if (!require(status.ok(), "Fluid-smoke coupling failed")) return 1;
    SmokeSoftBodyCouplingId smoke_soft{};
    status = world.add_smoke_soft_body_coupling(
        {.smoke = smoke, .soft_body = soft}, smoke_soft);
    if (!require(status.ok(), "Smoke-soft coupling failed")) return 1;
    SmokeClothCouplingId smoke_cloth{};
    status = world.add_smoke_cloth_coupling(
        {.smoke = smoke, .cloth = cloth}, smoke_cloth);
    if (!require(status.ok(), "Smoke-cloth coupling failed")) return 1;
    SmokeRopeCouplingId smoke_rope{};
    status = world.add_smoke_rope_coupling(
        {.smoke = smoke, .rope = rope}, smoke_rope);
    if (!require(status.ok(), "Smoke-rope coupling failed")) return 1;
    FluidClothCouplingId fluid_cloth{};
    status = world.add_fluid_cloth_coupling(
        {.fluid = fluid, .cloth = cloth}, fluid_cloth);
    if (!require(status.ok(), "Fluid-cloth coupling failed")) return 1;
    FluidSoftBodyCouplingId fluid_soft{};
    status = world.add_fluid_soft_body_coupling(
        {.fluid = fluid, .soft_body = soft}, fluid_soft);
    if (!require(status.ok(), "Fluid-soft coupling failed")) return 1;
    FluidRopeCouplingId fluid_rope{};
    status = world.add_fluid_rope_coupling(
        {.fluid = fluid, .rope = rope}, fluid_rope);
    if (!require(status.ok(), "Fluid-rope coupling failed")) return 1;
    FluidRopeCouplingId duplicate_fluid_rope{};
    const Status duplicate_fluid_rope_status =
        world.add_fluid_rope_coupling(
            {.fluid = fluid, .rope = rope}, duplicate_fluid_rope);
    if (!require(duplicate_fluid_rope_status.code ==
                     StatusCode::invalid_argument,
                 "Duplicate fluid-rope coupling was accepted"))
        return 1;
    SoftBodyClothCouplingId soft_cloth{};
    status = world.add_soft_body_cloth_coupling(
        {.soft_body = soft, .cloth = cloth}, soft_cloth);
    if (!require(status.ok(), "Soft-cloth coupling failed")) return 1;
    RopeSoftBodyCouplingId rope_soft{};
    status = world.add_rope_soft_body_coupling(
        {.rope = rope, .soft_body = soft}, rope_soft);
    if (!require(status.ok(), "Rope-soft coupling failed")) return 1;
    RopeClothCouplingId rope_cloth{};
    status = world.add_rope_cloth_coupling(
        {.rope = rope, .cloth = cloth, .first_vertex = 0U}, rope_cloth);
    if (!require(status.ok(), "Rope-cloth coupling failed")) return 1;
    const Status changed_rope_cloth_endpoint =
        world.update_rope_cloth_coupling(
            rope_cloth,
            {.rope = rope, .cloth = cloth, .last_vertex = 1U});
    if (!require(changed_rope_cloth_endpoint.code ==
                     StatusCode::invalid_argument,
                 "Rope-cloth update changed attachment endpoints"))
        return 1;

    for (std::uint32_t frame = 0; frame < 10U; ++frame) {
        status = world.step({.timestep = 1.0F / 60.0F,
                             .substeps = 2U,
                             .gravity = {0.0F, -9.81F, 0.0F},
                             .collect_kernel_timings = frame == 9U});
        if (!require(status.ok(), status.message ? status.message : "Particle step failed"))
            return 1;
    }

    WorldStepTimings particle_timings{};
    status = world.collect_step_timings(particle_timings);
    const std::uint64_t fluid_iterations =
        particle_timings.fluid_neighbor_sort.launch_count;
    const bool timings_complete =
        status.ok() && particle_timings.available &&
                     fluid_iterations >= 8U && fluid_iterations % 4U == 0U &&
                     particle_timings.fluid_neighbor_forces.launch_count ==
                         fluid_iterations &&
                     particle_timings.fluid_integration.launch_count ==
                         fluid_iterations &&
                     particle_timings.smoke_grid.launch_count >= 2U &&
                     particle_timings.smoke_advection.launch_count >= 1U &&
                     particle_timings.smoke_emission.launch_count == 1U &&
                     particle_timings.fluid_smoke_exchange.launch_count ==
                         1U &&
                     particle_timings.cloth_prediction.launch_count ==
                         fluid_iterations / 4U &&
                     particle_timings.cloth_constraints.launch_count >= 2U &&
                     particle_timings.soft_body_prediction.launch_count ==
                         fluid_iterations / 4U &&
                     particle_timings.soft_body_constraints.launch_count >=
                         2U &&
                     particle_timings.rope_solve.launch_count >= 2U &&
                     particle_timings.fluid_cloth_contacts.launch_count ==
                         2U * fluid_iterations &&
                     particle_timings.fluid_soft_body_contacts.launch_count ==
                         (6U * FluidSoftBodyCouplingOptions{}
                                   .solver_iterations +
                          3U) * fluid_iterations &&
                     particle_timings.fluid_rope_contacts.launch_count ==
                         fluid_iterations;
    if (!timings_complete && status.ok() && particle_timings.available)
        std::cerr << "fluid timing counts: sort="
                  << particle_timings.fluid_neighbor_sort.launch_count
                  << " force="
                  << particle_timings.fluid_neighbor_forces.launch_count
                  << " integrate="
                  << particle_timings.fluid_integration.launch_count
                  << " cloth-predict="
                  << particle_timings.cloth_prediction.launch_count
                  << " cloth="
                  << particle_timings.fluid_cloth_contacts.launch_count
                  << " soft-predict="
                  << particle_timings.soft_body_prediction.launch_count
                  << " soft="
                  << particle_timings.fluid_soft_body_contacts.launch_count
                  << " rope="
                  << particle_timings.fluid_rope_contacts.launch_count << '\n';
    if (!require(timings_complete,
                 "Metal particle-system per-phase timings are incomplete"))
        return 1;

    FluidDeviceView fluid_view{};
    ClothDeviceView cloth_view{};
    SoftBodyDeviceView soft_view{};
    RopeDeviceView rope_view{};
    SmokeDeviceView smoke_view{};
    status = world.fluid_view(fluid, fluid_view);
    const bool fluid_valid =
        status.ok() && fluid_view.particle_count == water.size() &&
        finite(contents(fluid_view.positions)[0]) &&
        finite(contents(fluid_view.accelerations)[0]);
    if (!require(fluid_valid, "Fluid view/state is invalid"))
        return 1;
    status = world.smoke_view(smoke, smoke_view);
    if (!require(status.ok() && smoke_view.particle_count > 0U &&
                     finite(contents(smoke_view.positions)[0]) &&
                     smoke_view.grid_density.size == 2048U,
                 "Smoke did not emit finite particles"))
        return 1;
    bool occupied_grid_cell = false;
    for (std::uint64_t index = 0; index < smoke_view.grid_density.size; ++index)
        occupied_grid_cell |= contents(smoke_view.grid_density)[index] > 0.0F;
    if (!require(occupied_grid_cell, "Grid smoke did not splat density"))
        return 1;
    status = world.cloth_view(cloth, cloth_view);
    if (!require(status.ok() && cloth_view.vertex_count == cloth_vertices.size() &&
                     finite(contents(cloth_view.positions)[2]),
                 "Cloth state is invalid"))
        return 1;
    status = world.soft_body_view(soft, soft_view);
    if (!require(status.ok() && soft_view.node_count == soft_nodes.size() &&
                     finite(contents(soft_view.positions)[0]),
                 "Soft-body state is invalid"))
        return 1;
    status = world.rope_view(rope, rope_view);
    if (!require(status.ok() && rope_view.positions.size >= 2U &&
                     finite(contents(rope_view.positions)[0]),
                 "Rope state is invalid"))
        return 1;

    status = world.remove_fluid(fluid);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Coupled fluid was removed"))
        return 1;
    status = world.remove_fluid_rope_coupling(fluid_rope);
    if (!require(status.ok(), "Coupling removal failed")) return 1;
    status = world.remove_fluid_rope_coupling(fluid_rope);
    if (!require(status.code == StatusCode::invalid_handle,
                 "Removed coupling handle did not become stale"))
        return 1;

    WorldStatistics statistics{};
    status = world.collect_statistics(statistics);
    if (!require(status.ok() && statistics.fluid_count == 1U &&
                     statistics.smoke_system_count == 1U &&
                     statistics.cloth_count == 1U &&
                     statistics.soft_body_count == 1U &&
                     statistics.rope_count == 1U,
                 "Particle statistics are incomplete"))
        return 1;

    World lifecycle;
    status = World::create({.fluid_capacity = 2U}, lifecycle);
    if (!require(status.ok(), "Lifecycle world creation failed")) return 1;
    FluidId emitted_fluid{};
    status = lifecycle.add_fluid(
        {.capacity = 1U}, HostSpan<const FluidParticle>{}, emitted_fluid);
    if (!require(status.ok(), "Empty source fluid creation failed")) return 1;
    std::array<Vec3, 3> source_vertices{{
        {-0.04F, 0.0F, -0.04F}, {0.04F, 0.0F, -0.04F},
        {0.0F, 0.0F, 0.04F},
    }};
    const std::array<Vec3, 3> capacity_source_vertices{{
        {-0.4F, 0.0F, -0.4F}, {0.4F, 0.0F, -0.4F},
        {0.0F, 0.0F, 0.4F},
    }};
    std::array<std::uint32_t, 3> source_indices{{0U, 1U, 2U}};
    ParticleSourceId source{};
    status = lifecycle.add_particle_source(
        {{capacity_source_vertices.data(), capacity_source_vertices.size()},
         {source_indices.data(), source_indices.size()}, 0.09F},
        {.fluid = emitted_fluid,
         .initial_velocity = {0.0F, 1.0F, 0.0F}},
        source);
    if (!require(status.ok(), "Particle source creation failed")) return 1;
    status = lifecycle.step({.timestep = 1.0F / 60.0F,
                             .substeps = 1U,
                             .gravity = {}});
    FluidDeviceView emitted_view{};
    status = status ? lifecycle.fluid_view(emitted_fluid, emitted_view) : status;
    if (!require(status.ok() && emitted_view.particle_count > 0U,
                 "Particle source did not emit"))
        return 1;
    WorldStatistics lifecycle_statistics{};
    status = lifecycle.collect_statistics(lifecycle_statistics);
    if (!require(status.ok() &&
                     lifecycle_statistics.emitted_particle_count ==
                         emitted_view.particle_count &&
                     lifecycle_statistics.destroyed_particle_count == 0U &&
                     lifecycle_statistics.spawn_capacity_miss_count > 0U,
                 "Particle source lifetime statistics are invalid"))
        return 1;
    status = lifecycle.update_particle_source(
        source, {.fluid = emitted_fluid,
                 .initial_velocity = {0.0F, 1.0F, 0.0F},
                 .enabled = false});
    if (!require(status.ok(), "Particle source update failed")) return 1;
    ParticleDestroyPlaneId destroy{};
    status = lifecycle.add_particle_destroy_plane(
        {.fluid = emitted_fluid,
         .plane = {.center = {0.0F, 0.02F, 0.0F},
                   .orientation = {0.0F, 0.0F, 0.0F, 3.0F},
                   .half_extents = {1.0F, 1.0F}},
         .crossing = CrossingDirection::along_normal},
        destroy);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Destroy plane accepted CUDA-invalid quaternion magnitude"))
        return 1;
    status = lifecycle.add_particle_destroy_plane(
        {.fluid = emitted_fluid,
         .plane = {.center = {0.0F, 0.02F, 0.0F},
                   .orientation = {0.0F, 0.0F, 0.0F, 1.5F},
                   .half_extents = {1.0F, 1.0F}},
         .crossing = CrossingDirection::along_normal},
        destroy);
    if (!require(status.ok(), "Destroy plane creation failed")) return 1;
    status = lifecycle.step({.timestep = 1.0F / 30.0F,
                             .substeps = 1U,
                             .gravity = {}});
    status = status ? lifecycle.fluid_view(emitted_fluid, emitted_view) : status;
    if (!require(status.ok() && emitted_view.particle_count == 0U,
                 "Destroy plane did not compact crossed particles"))
        return 1;
    status = lifecycle.collect_statistics(lifecycle_statistics);
    const std::uint64_t emitted_lifetime =
        lifecycle_statistics.emitted_particle_count;
    const std::uint64_t capacity_miss_lifetime =
        lifecycle_statistics.spawn_capacity_miss_count;
    if (!require(status.ok() &&
                     lifecycle_statistics.destroyed_particle_count ==
                         emitted_lifetime,
                 "Destroyed-particle statistics omit removed particles"))
        return 1;
    const FluidParticle retarget_particle{
        {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 20.0F};
    FluidId retarget_fluid{};
    status = lifecycle.add_fluid(
        {.capacity = 1U, .velocity_damping = 0.0F},
        {&retarget_particle, 1U}, retarget_fluid);
    status = status ? lifecycle.update_particle_destroy_plane(
                          destroy,
                          {.fluid = retarget_fluid,
                           .plane = {.center = {0.0F, 0.02F, 0.0F},
                                     .orientation = {0.0F, 0.0F, 0.0F, 1.5F},
                                     .half_extents = {1.0F, 1.0F}},
                           .crossing = CrossingDirection::along_normal})
                    : status;
    status = status ? lifecycle.remove_fluid(emitted_fluid) : status;
    if (!require(status.ok(), status.message ? status.message
                                             : "Destroy-plane retarget failed"))
        return 1;
    status = lifecycle.remove_particle_source(source);
    if (!require(status.code == StatusCode::invalid_handle,
                 "Removing a fluid did not invalidate its particle source"))
        return 1;
    status = lifecycle.step({.timestep = 1.0F / 30.0F,
                             .substeps = 1U,
                             .gravity = {}});
    status = status ? lifecycle.fluid_view(retarget_fluid, emitted_view) : status;
    if (!require(status.ok() && emitted_view.particle_count == 0U,
                 "Retargeted destroy plane retained stale fluid buffers"))
        return 1;
    status = lifecycle.remove_fluid(retarget_fluid);
    if (!require(status.ok(),
                 "Retargeted destroy plane blocked owning-fluid removal"))
        return 1;
    status = lifecycle.remove_particle_destroy_plane(destroy);
    if (!require(status.code == StatusCode::invalid_handle,
                 "Removing a fluid did not invalidate its destroy plane"))
        return 1;
    status = lifecycle.collect_statistics(lifecycle_statistics);
    if (!require(status.ok() && lifecycle_statistics.fluid_count == 0U &&
                     lifecycle_statistics.emitted_particle_count ==
                         emitted_lifetime &&
                     lifecycle_statistics.destroyed_particle_count ==
                         emitted_lifetime + 1U &&
                     lifecycle_statistics.spawn_capacity_miss_count ==
                         capacity_miss_lifetime,
                 "Fluid removal discarded lifetime statistics"))
        return 1;

    constexpr std::uint32_t compaction_count = 257U;
    std::vector<FluidParticle> compaction_particles(compaction_count);
    for (std::uint32_t index = 0U; index < compaction_count; ++index) {
        compaction_particles[index] = {
            {(static_cast<float>(index) - 128.0F) * 0.01F,
             (index & 1U) == 0U ? -0.01F : -0.02F, 0.0F},
            {0.0F, (index & 1U) == 0U ? 1.0F : 0.0F, 0.0F},
            20.0F};
    }
    World compaction_world;
    status = World::create({.fluid_capacity = 1U}, compaction_world);
    FluidId compaction_fluid{};
    status = status
                 ? compaction_world.add_fluid(
                       {.capacity = compaction_count,
                        .particle_radius = 0.001F,
                        .support_radius = 0.003F,
                        .solver_iterations = 1U,
                        .repulsion = 0.0F,
                        .viscosity = 0.0F,
                        .velocity_damping = 0.0F,
                        .normal_damping = 0.0F},
                       HostSpan<const FluidParticle>{
                           compaction_particles.data(),
                           compaction_particles.size()},
                       compaction_fluid)
                 : status;
    ParticleDestroyPlaneId compaction_plane{};
    status = status
                 ? compaction_world.add_particle_destroy_plane(
                       {.fluid = compaction_fluid,
                        .plane = {.center = {},
                                  .orientation = {0.0F, 0.0F, 0.0F, 1.0F},
                                  .half_extents = {2.0F, 1.0F}},
                        .crossing = CrossingDirection::along_normal},
                       compaction_plane)
                 : status;
    status = status ? compaction_world.step(
                          {.timestep = 0.02F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    FluidDeviceView compaction_view{};
    status = status ? compaction_world.fluid_view(
                          compaction_fluid, compaction_view)
                    : status;
    bool stable_compaction = status.ok() &&
                             compaction_view.particle_count == 128U;
    if (stable_compaction) {
        const auto *ids = contents(compaction_view.stable_particle_ids);
        for (std::uint32_t index = 0U;
             index < compaction_view.particle_count; ++index)
            stable_compaction &= ids[index] == index * 2U + 1U;
    }
    WorldStatistics compaction_statistics{};
    status = status ? compaction_world.collect_statistics(
                          compaction_statistics)
                    : status;
    if (!require(status.ok() && stable_compaction &&
                     compaction_statistics.destroyed_particle_count == 129U,
                 "Cross-threadgroup destroy compaction lost stable IDs"))
        return 1;

    constexpr std::array<Vec3, 8> cube{{
        {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
        {1.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 1.0F},
        {1.0F, 1.0F, 1.0F}, {0.0F, 1.0F, 1.0F},
    }};
    constexpr std::array<std::uint32_t, 36> cube_indices{{
        0U, 2U, 1U, 0U, 3U, 2U, 4U, 5U, 6U, 4U, 6U, 7U,
        0U, 1U, 5U, 0U, 5U, 4U, 1U, 2U, 6U, 1U, 6U, 5U,
        2U, 3U, 7U, 2U, 7U, 6U, 3U, 0U, 4U, 3U, 4U, 7U,
    }};
    const FluidGeometrySource geometry{
        {cube.data(), cube.size()},
        {cube_indices.data(), cube_indices.size()}, {}, {}, 0.2F};
    World geometry_world;
    status = World::create({}, geometry_world);
    FluidId geometry_fluid{};
    status = status ? geometry_world.add_fluid_geometry(
                          {.capacity = 0U}, geometry, geometry_fluid)
                    : status;
    if (!require(status.code == StatusCode::invalid_argument,
                 "Metal accepted zero-capacity fluid geometry"))
        return 1;
    status = geometry_world.add_fluid_geometry(
        {.capacity = 10U}, geometry, geometry_fluid);
    FluidDeviceView geometry_view{};
    status = status ? geometry_world.fluid_view(geometry_fluid, geometry_view)
                    : status;
    if (!require(status.ok() && geometry_view.particle_count == 10U,
                 "Metal fluid geometry did not use CUDA-style thinning"))
        return 1;

    WorldOptions spawn_contact_capacities{};
    spawn_contact_capacities.rigid_body_capacity = 1U;
    spawn_contact_capacities.triangle_mesh_capacity = 1U;
    World spawn_contact_world;
    status = World::create(spawn_contact_capacities, spawn_contact_world);
    FluidId spawn_contact_fluid{};
    status = status ? spawn_contact_world.add_fluid(
                          {.capacity = 8U},
                          HostSpan<const FluidParticle>{},
                          spawn_contact_fluid)
                    : status;
    TriangleMeshId spawn_contact_mesh{};
    status = status ? spawn_contact_world.add_triangle_mesh(
                          HostSpan<const Vec3>{source_vertices.data(),
                                               source_vertices.size()},
                          HostSpan<const std::uint32_t>{
                              source_indices.data(), source_indices.size()},
                          spawn_contact_mesh)
                    : status;
    RigidBodyId spawn_contact_body{};
    status = status ? spawn_contact_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = spawn_contact_mesh},
                          spawn_contact_body)
                    : status;
    ParticleSourceId spawn_contact_source{};
    status = status ? spawn_contact_world.add_particle_source(
                          {{source_vertices.data(), source_vertices.size()},
                           {source_indices.data(), source_indices.size()},
                           0.09F},
                          {.fluid = spawn_contact_fluid},
                          spawn_contact_source)
                    : status;
    status = status ? spawn_contact_world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    FluidDeviceView spawn_contact_view{};
    status = status ? spawn_contact_world.fluid_view(
                          spawn_contact_fluid, spawn_contact_view)
                    : status;
    bool spawned_particles_hit_rigid = status.ok() &&
                                       spawn_contact_view.particle_count > 0U;
    if (spawned_particles_hit_rigid) {
        const Vec3 *spawned_positions =
            contents(spawn_contact_view.positions);
        // CUDA treats a newly emitted particle on a floor-facing triangle as
        // spawn recovery and chooses gravity-up rather than source winding.
        for (std::uint32_t index = 0U;
             index < spawn_contact_view.particle_count; ++index)
            spawned_particles_hit_rigid &= spawned_positions[index].y >
                                           0.01F;
    }
    if (!spawned_particles_hit_rigid && status.ok()) {
        std::cerr << "same-frame spawn/contact count="
                  << spawn_contact_view.particle_count;
        if (spawn_contact_view.particle_count != 0U)
            std::cerr << " first_y="
                      << contents(spawn_contact_view.positions)[0].y;
        std::cerr << '\n';
    }
    if (!require(spawned_particles_hit_rigid,
                 "Particles spawned on-GPU missed same-frame rigid recovery"))
        return 1;

    const auto source_count_for_substeps = [&](std::uint32_t substeps) {
        World source_world;
        Status source_status = World::create({}, source_world);
        FluidId source_fluid{};
        source_status = source_status
                            ? source_world.add_fluid(
                                  {.capacity = 64U,
                                   .particle_radius = 0.01F,
                                   .support_radius = 0.09F,
                                   .solver_iterations = 1U,
                                   .repulsion = 0.0F,
                                   .viscosity = 0.0F,
                                   .velocity_damping = 0.0F,
                                   .maximum_speed = 100.0F},
                                  HostSpan<const FluidParticle>{},
                                  source_fluid)
                            : source_status;
        ParticleSourceId fast_source{};
        source_status = source_status
                            ? source_world.add_particle_source(
                                  {{source_vertices.data(),
                                    source_vertices.size()},
                                   {source_indices.data(),
                                    source_indices.size()},
                                   0.09F},
                                  {.fluid = source_fluid,
                                   .initial_velocity =
                                       {0.0F, 10.0F, 0.0F}},
                                  fast_source)
                            : source_status;
        source_status = source_status
                            ? source_world.step(
                                  {.timestep = 0.2F,
                                   .substeps = substeps,
                                   .gravity = {}})
                            : source_status;
        FluidDeviceView source_view{};
        source_status = source_status
                            ? source_world.fluid_view(source_fluid,
                                                      source_view)
                            : source_status;
        return std::pair{source_status, source_view.particle_count};
    };
    const auto single_source_frame = source_count_for_substeps(1U);
    const auto split_source_frame = source_count_for_substeps(4U);
    if (!require(single_source_frame.first.ok() &&
                     split_source_frame.first.ok() &&
                     single_source_frame.second > 0U &&
                     split_source_frame.second == single_source_frame.second,
                 "Particle source emitted once per substep instead of once per frame"))
        return 1;

    World phase_world;
    status = World::create({}, phase_world);
    if (!require(status.ok(), "Phase-transfer world creation failed")) return 1;
    const std::array<FluidParticle, 1> hot_water{{
        {{0.0F, 0.0F, 0.0F}, {}, 99.0F},
    }};
    FluidId phase_fluid{};
    status = phase_world.add_fluid(
        {.capacity = 4U},
        HostSpan<const FluidParticle>{hot_water.data(), hot_water.size()},
        phase_fluid);
    SmokeId phase_smoke{};
    status = status ? phase_world.add_smoke(
                          {.capacity = 4U,
                           .emitter_center = {100.0F, 100.0F, 100.0F},
                           .particles_per_second = 1.0F,
                           .grid_resolution = 16U,
                           .grid_vertical_resolution = 8U,
                           .grid_pressure_iterations = 8U,
                           .grid_minimum = {-1.0F, -1.0F, -1.0F},
                           .grid_edge_length = 2.0F},
                          phase_smoke)
                    : status;
    FluidSmokeCouplingId phase_coupling{};
    status = status ? phase_world.add_fluid_smoke_coupling(
                          {.fluid = phase_fluid,
                           .smoke = phase_smoke,
                           .heater = {.center = {},
                                      .half_extents = {1.0F, 1.0F}},
                           .heater_temperature = 500.0F,
                           .boiling_temperature = 100.0F,
                           .heat_transfer_rate = 1000.0F,
                           .steam_rise_speed = 2.0F},
                          phase_coupling)
                    : status;
    if (!require(status.ok(), "Fluid-smoke phase coupling creation failed"))
        return 1;
    status = phase_world.step(
        {.timestep = 1.0F / 60.0F, .substeps = 1U, .gravity = {}});
    FluidDeviceView phase_fluid_view{};
    SmokeDeviceView phase_smoke_view{};
    status = status ? phase_world.fluid_view(phase_fluid, phase_fluid_view)
                    : status;
    status = status ? phase_world.smoke_view(phase_smoke, phase_smoke_view)
                    : status;
    WorldStatistics phase_statistics{};
    status = status ? phase_world.collect_statistics(phase_statistics) : status;
    if (!require(status.ok() && phase_fluid_view.particle_count == 0U &&
                     phase_smoke_view.particle_count == 1U &&
                     contents(phase_smoke_view.velocities)[0].y > 1.9F &&
                     phase_statistics.boiled_particle_count == 1U &&
                     phase_statistics.destroyed_particle_count == 1U,
                 "Heated fluid did not convert deterministically to smoke"))
        return 1;
    status = phase_world.step(
        {.timestep = 1.0F / 60.0F, .substeps = 1U, .gravity = {}});
    status = status ? phase_world.smoke_view(phase_smoke, phase_smoke_view)
                    : status;
    float maximum_thermal_lift = 0.0F;
    if (status.ok()) {
        const float *temperature = contents(phase_smoke_view.grid_temperature);
        const float *density = contents(phase_smoke_view.grid_density);
        for (std::uint64_t cell = 0U;
             cell < phase_smoke_view.grid_temperature.size; ++cell) {
            if (density[cell] > 1.0e-7F)
                maximum_thermal_lift = std::max(
                    maximum_thermal_lift, temperature[cell] / density[cell]);
        }
    }
    if (!require(status.ok() && maximum_thermal_lift > 1.5F,
                 "Boiled smoke thermal lift did not enter the grid"))
        return 1;

    constexpr std::uint32_t boiling_fluid_count = 257U;
    constexpr std::uint32_t boiling_smoke_capacity = 80U;
    std::vector<FluidParticle> boiling_water(boiling_fluid_count);
    for (std::uint32_t index = 0U; index < boiling_fluid_count; ++index) {
        boiling_water[index] = {
            {(static_cast<float>(index) - 128.0F) * 0.01F, 0.0F, 0.0F},
            {}, 150.0F};
    }
    World boiling_world;
    status = World::create({.fluid_capacity = 1U,
                            .smoke_capacity = 1U,
                            .fluid_smoke_coupling_capacity = 1U},
                           boiling_world);
    FluidId boiling_fluid{};
    status = status
                 ? boiling_world.add_fluid(
                       {.capacity = boiling_fluid_count,
                        .particle_radius = 0.001F,
                        .support_radius = 0.003F,
                        .solver_iterations = 1U,
                        .repulsion = 0.0F,
                        .viscosity = 0.0F,
                        .velocity_damping = 0.0F},
                       HostSpan<const FluidParticle>{boiling_water.data(),
                                                     boiling_water.size()},
                       boiling_fluid)
                 : status;
    SmokeId boiling_smoke{};
    status = status
                 ? boiling_world.add_smoke(
                       {.capacity = boiling_smoke_capacity,
                        .emitter_center = {100.0F, 100.0F, 100.0F},
                        .particles_per_second = 1.0F,
                        .buoyancy = 0.0F,
                        .response = 0.0F},
                       boiling_smoke)
                 : status;
    FluidSmokeCouplingId boiling_coupling{};
    status = status
                 ? boiling_world.add_fluid_smoke_coupling(
                       {.fluid = boiling_fluid,
                        .smoke = boiling_smoke,
                        .heater = {.center = {100.0F, 100.0F, 100.0F}},
                        .heat_transfer_rate = 0.0F,
                        .wind_drag = 0.0F},
                       boiling_coupling)
                 : status;
    status = status ? boiling_world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    FluidDeviceView boiling_fluid_view{};
    SmokeDeviceView boiling_smoke_view{};
    status = status ? boiling_world.fluid_view(boiling_fluid,
                                                boiling_fluid_view)
                    : status;
    status = status ? boiling_world.smoke_view(boiling_smoke,
                                                boiling_smoke_view)
                    : status;
    bool stable_boiling = status.ok() &&
        boiling_fluid_view.particle_count ==
            boiling_fluid_count - boiling_smoke_capacity &&
        boiling_smoke_view.particle_count == boiling_smoke_capacity;
    std::uint32_t first_bad_boiling_id = UINT32_MAX;
    if (stable_boiling) {
        const auto *ids = contents(boiling_fluid_view.stable_particle_ids);
        for (std::uint32_t index = 0U;
             index < boiling_fluid_view.particle_count; ++index) {
            if (ids[index] != index + boiling_smoke_capacity) {
                first_bad_boiling_id = index;
                stable_boiling = false;
                break;
            }
        }
    }
    WorldStatistics boiling_statistics{};
    status = status ? boiling_world.collect_statistics(boiling_statistics)
                    : status;
    if (!(status.ok() && stable_boiling &&
          boiling_statistics.boiled_particle_count ==
              boiling_smoke_capacity &&
          boiling_statistics.destroyed_particle_count ==
              boiling_smoke_capacity)) {
        std::cerr << "Boiling status="
                  << (status.message == nullptr ? "unknown" : status.message)
                  << " fluid=" << boiling_fluid_view.particle_count
                  << " smoke=" << boiling_smoke_view.particle_count
                  << " boiled="
                  << boiling_statistics.boiled_particle_count
                  << " destroyed="
                  << boiling_statistics.destroyed_particle_count
                  << " bad_id_index=" << first_bad_boiling_id << '\n';
    }
    if (!require(status.ok() && stable_boiling &&
                     boiling_statistics.boiled_particle_count ==
                         boiling_smoke_capacity &&
                     boiling_statistics.destroyed_particle_count ==
                         boiling_smoke_capacity,
                 "Cross-threadgroup boiling lost stable excess water"))
        return 1;

    World grid_drag_world;
    status = World::create({}, grid_drag_world);
    const std::array<FluidParticle, 1> grid_water{{
        // Outside the tracer's 3r support but inside its quadratic grid
        // deposit, so only the projected-grid coupling can move this water.
        {{0.12F, 0.0F, 0.0F}, {}, 20.0F},
    }};
    FluidId grid_fluid{};
    status = status ? grid_drag_world.add_fluid(
                          {.capacity = 1U,
                           .velocity_damping = 0.0F,
                           .maximum_speed = 10.0F},
                          {grid_water.data(), grid_water.size()}, grid_fluid)
                    : status;
    SmokeId grid_smoke{};
    status = status ? grid_drag_world.add_smoke(
                          {.capacity = 2U,
                           .emitter_center = {},
                           .emitter_half_extents = {0.001F, 0.001F},
                           .initial_velocity = {-1.0F, 0.0F, 0.0F},
                           .wind = {1.0F, 0.0F, 0.0F},
                           .particles_per_second = 60.0F,
                           .lifetime = 2.0F,
                           .particle_radius = 0.025F,
                           .buoyancy = 0.0F,
                           .response = 0.0F,
                           .grid_resolution = 16U,
                           .grid_vertical_resolution = 16U,
                           .grid_pressure_iterations = 8U,
                           .grid_minimum = {-1.0F, -1.0F, -1.0F},
                           .grid_edge_length = 2.0F},
                          grid_smoke)
                    : status;
    FluidSmokeCouplingId grid_drag{};
    status = status ? grid_drag_world.add_fluid_smoke_coupling(
                          {.fluid = grid_fluid,
                           .smoke = grid_smoke,
                           .heater = {.center = {100.0F, 100.0F, 100.0F},
                                      .half_extents = {1.0F, 1.0F}},
                           .heat_transfer_rate = 0.0F,
                           .wind_drag = 100.0F},
                          grid_drag)
                    : status;
    status = status ? grid_drag_world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    status = status ? grid_drag_world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    FluidDeviceView grid_fluid_view{};
    status = status ? grid_drag_world.fluid_view(grid_fluid,
                                                  grid_fluid_view)
                    : status;
    if (!require(status.ok() && grid_fluid_view.particle_count == 1U &&
                     std::abs(contents(grid_fluid_view.velocities)[0].x) >
                         1.0e-4F,
                 "Fluid did not sample projected grid-smoke velocity"))
        return 1;
    return 0;
}
