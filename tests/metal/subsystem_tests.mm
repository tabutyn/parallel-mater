// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
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

float distance(Vec3 first, Vec3 second) {
    const float x = first.x - second.x;
    const float y = first.y - second.y;
    const float z = first.z - second.z;
    return std::sqrt(x * x + y * y + z * z);
}

std::uint32_t smoke_hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

float smoke_random(std::uint32_t seed) {
    return static_cast<float>(smoke_hash(seed) & 0x00ffffffU) /
           16777216.0F;
}

std::uint64_t fluid_cell_key(int x, int y, int z) {
    constexpr int bias = 1 << 20;
    x = std::clamp(x, -bias, bias - 1);
    y = std::clamp(y, -bias, bias - 1);
    z = std::clamp(z, -bias, bias - 1);
    return (static_cast<std::uint64_t>(x + bias) << 42U) |
           (static_cast<std::uint64_t>(y + bias) << 21U) |
           static_cast<std::uint64_t>(z + bias);
}

std::vector<Vec3> fluid_reference_accelerations(
    const std::vector<FluidParticle> &particles, const FluidOptions &options) {
    struct CellEntry {
        std::uint64_t key{};
        std::uint32_t index{};
    };
    std::vector<CellEntry> cells;
    cells.reserve(particles.size());
    const float inverse_radius = 1.0F / options.support_radius;
    for (std::uint32_t index = 0U; index < particles.size(); ++index) {
        const Vec3 position = particles[index].position;
        cells.push_back({
            fluid_cell_key(
                static_cast<int>(std::floor(position.x * inverse_radius)),
                static_cast<int>(std::floor(position.y * inverse_radius)),
                static_cast<int>(std::floor(position.z * inverse_radius))),
            index});
    }
    std::stable_sort(cells.begin(), cells.end(),
                     [](const CellEntry &first, const CellEntry &second) {
                         return first.key < second.key;
                     });

    std::vector<Vec3> result(particles.size());
    const float support_squared =
        options.support_radius * options.support_radius;
    constexpr int bias = 1 << 20;
    for (std::uint32_t index = 0U; index < particles.size(); ++index) {
        const Vec3 position = particles[index].position;
        const Vec3 velocity = particles[index].velocity;
        const int center_x =
            static_cast<int>(std::floor(position.x * inverse_radius));
        const int center_y =
            static_cast<int>(std::floor(position.y * inverse_radius));
        const int center_z =
            static_cast<int>(std::floor(position.z * inverse_radius));
        Vec3 acceleration{};
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = center_x + dx;
                    const int y = center_y + dy;
                    const int z = center_z + dz;
                    if (x < -bias || x >= bias || y < -bias || y >= bias ||
                        z < -bias || z >= bias)
                        continue;
                    const std::uint64_t key = fluid_cell_key(x, y, z);
                    auto item = std::lower_bound(
                        cells.begin(), cells.end(), key,
                        [](const CellEntry &entry, std::uint64_t value) {
                            return entry.key < value;
                        });
                    for (; item != cells.end() && item->key == key; ++item) {
                        const std::uint32_t other = item->index;
                        if (other == index) continue;
                        const Vec3 other_position = particles[other].position;
                        const Vec3 delta{position.x - other_position.x,
                                         position.y - other_position.y,
                                         position.z - other_position.z};
                        const float squared = delta.x * delta.x +
                                              delta.y * delta.y +
                                              delta.z * delta.z;
                        if (squared >= support_squared) continue;
                        const float pair_distance =
                            std::sqrt(std::max(squared, 1.0e-12F));
                        const Vec3 normal = squared > 1.0e-12F
                            ? Vec3{delta.x / pair_distance,
                                   delta.y / pair_distance,
                                   delta.z / pair_distance}
                            : (index < other ? Vec3{-1.0F, 0.0F, 0.0F}
                                             : Vec3{1.0F, 0.0F, 0.0F});
                        const float weight =
                            1.0F - pair_distance * inverse_radius;
                        const Vec3 relative_velocity{
                            particles[other].velocity.x - velocity.x,
                            particles[other].velocity.y - velocity.y,
                            particles[other].velocity.z - velocity.z};
                        const float radial_speed =
                            relative_velocity.x * normal.x +
                            relative_velocity.y * normal.y +
                            relative_velocity.z * normal.z;
                        const float pair_acceleration =
                            options.repulsion *
                                (1'000.0F / options.rest_density) * weight *
                                weight +
                            options.normal_damping * radial_speed;
                        acceleration.x += normal.x * pair_acceleration;
                        acceleration.y += normal.y * pair_acceleration;
                        acceleration.z += normal.z * pair_acceleration;
                        acceleration.x += relative_velocity.x *
                                          (options.viscosity * weight);
                        acceleration.y += relative_velocity.y *
                                          (options.viscosity * weight);
                        acceleration.z += relative_velocity.z *
                                          (options.viscosity * weight);
                    }
                }
            }
        }
        result[index] = acceleration;
    }
    return result;
}

bool step(World &world, std::uint32_t frames, Vec3 gravity,
          std::uint32_t substeps = 2U) {
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        const Status status = world.step(
            {.timestep = 1.0F / 60.0F,
             .substeps = substeps,
             .gravity = gravity});
        if (!require(status.ok(), status.message ? status.message
                                                 : "Metal step failed"))
            return false;
    }
    return true;
}

int fluid_test() {
    World world;
    Status status = World::create({}, world);
    if (!require(status.ok(), "Fluid world creation failed")) return 1;
    std::array<FluidParticle, 2> particles{{
        {{-0.02F, 0.0F, 0.0F}, {}, 20.0F},
        {{0.02F, 0.0F, 0.0F}, {}, 20.0F},
    }};
    FluidId fluid{};
    status = world.add_fluid(
        {.capacity = 8U,
         .particle_radius = 0.02F,
         .support_radius = 0.12F,
         .repulsion = 50.0F,
         .velocity_damping = 0.0F,
         .maximum_speed = 20.0F},
        HostSpan<const FluidParticle>{particles.data(), particles.size()},
        fluid);
    if (!require(status.ok(), "Fluid creation failed")) return 1;
    if (!step(world, 2U, {})) return 1;
    FluidDeviceView view{};
    status = world.fluid_view(fluid, view);
    if (!require(status.ok() && view.particle_count == 2U,
                 "Fluid count changed unexpectedly"))
        return 1;
    const Vec3 *positions = contents(view.positions);
    const Vec3 *accelerations = contents(view.accelerations);
    if (!require(distance(positions[0], positions[1]) > 0.04F &&
                     accelerations[0].x < 0.0F &&
                     accelerations[1].x > 0.0F &&
                     finite(accelerations[0]) && finite(accelerations[1]),
                 "Fluid neighbor force did not separate the pair"))
        return 1;

    World capped_world;
    status = World::create({}, capped_world);
    FluidId capped_fluid{};
    status = status ? capped_world.add_fluid(
                          {.capacity = 2U,
                           .particle_radius = 0.02F,
                           .support_radius = 0.12F,
                           .solver_iterations = 1U,
                           .repulsion = 50.0F,
                           .viscosity = 0.0F,
                           .velocity_damping = 0.0F,
                           .maximum_speed = 20.0F,
                           .maximum_pair_acceleration = 1.0F},
                          HostSpan<const FluidParticle>{
                              particles.data(), particles.size()},
                          capped_fluid)
                    : status;
    status = status ? capped_world.step(
                          {.timestep = 0.1F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    FluidDeviceView capped_view{};
    status = status ? capped_world.fluid_view(capped_fluid, capped_view)
                    : status;
    if (!require(status.ok() && capped_view.particle_count == 2U &&
                     std::abs(contents(capped_view.velocities)[0].x + 0.1F) <
                         0.01F &&
                     std::abs(contents(capped_view.velocities)[1].x - 0.1F) <
                         0.01F,
                 "Fluid pair-acceleration cap diverged from CUDA behavior"))
        return 1;

    World invalid_world;
    status = World::create({}, invalid_world);
    const FluidParticle nonfinite_particle{
        {std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F}, {}, 20.0F};
    FluidId invalid_fluid{};
    const Status nonfinite_status = status
        ? invalid_world.add_fluid(
              {.capacity = 1U},
              HostSpan<const FluidParticle>{&nonfinite_particle, 1U},
              invalid_fluid)
        : status;
    if (!require(nonfinite_status.code == StatusCode::invalid_argument,
                 "Non-finite initial fluid particle was accepted"))
        return 1;

    World sorted_world;
    status = World::create({}, sorted_world);
    if (!require(status.ok(), "Sorted-fluid world creation failed")) return 1;
    FluidOptions sorted_options{
        .capacity = 64U,
        .particle_radius = 0.02F,
        .support_radius = 0.12F,
        .solver_iterations = 1U,
        .maximum_neighbors = 128U,
        .repulsion = 50.0F,
        .viscosity = 0.7F,
        .velocity_damping = 0.0F,
        .maximum_speed = 1'000.0F,
        .normal_damping = 0.2F,
    };
    std::vector<FluidParticle> sorted_particles(64U);
    for (std::uint32_t index = 0U; index < sorted_particles.size(); ++index) {
        const std::uint32_t logical = (index * 37U) % 64U;
        const std::uint32_t x = logical % 4U;
        const std::uint32_t y = (logical / 4U) % 4U;
        const std::uint32_t z = logical / 16U;
        sorted_particles[index] = {
            {0.07F * (static_cast<float>(x) - 1.5F),
             0.07F * (static_cast<float>(y) - 1.5F),
             0.07F * (static_cast<float>(z) - 1.5F)},
            {0.03F * (static_cast<float>(z) - 1.5F),
             -0.02F * (static_cast<float>(x) - 1.5F),
             0.01F * (static_cast<float>(y) - 1.5F)},
            20.0F};
    }
    const std::vector<Vec3> expected =
        fluid_reference_accelerations(sorted_particles, sorted_options);
    FluidId sorted_fluid{};
    status = sorted_world.add_fluid(
        sorted_options,
        HostSpan<const FluidParticle>{sorted_particles.data(),
                                      sorted_particles.size()},
        sorted_fluid);
    status = status ? sorted_world.step({.timestep = 1.0e-5F,
                                         .substeps = 1U,
                                         .gravity = {},
                                         .collect_kernel_timings = true})
                    : status;
    FluidDeviceView sorted_view{};
    status = status ? sorted_world.fluid_view(sorted_fluid, sorted_view)
                    : status;
    bool matches_reference = status.ok() &&
                             sorted_view.particle_count ==
                                 sorted_particles.size();
    if (matches_reference) {
        const Vec3 *actual = contents(sorted_view.accelerations);
        const std::uint32_t *ids =
            contents(sorted_view.stable_particle_ids);
        for (std::uint32_t index = 0U; index < sorted_particles.size();
             ++index) {
            const float scale = std::max(
                {std::abs(expected[index].x), std::abs(expected[index].y),
                 std::abs(expected[index].z), 1.0F});
            const float tolerance = 1.0e-3F + 1.0e-4F * scale;
            matches_reference &= ids[index] == index;
            matches_reference &=
                std::abs(actual[index].x - expected[index].x) <= tolerance &&
                std::abs(actual[index].y - expected[index].y) <= tolerance &&
                std::abs(actual[index].z - expected[index].z) <= tolerance;
        }
    }
    WorldStepTimings sorted_timings{};
    status = status ? sorted_world.collect_step_timings(sorted_timings)
                    : status;
    if (!require(status.ok() && matches_reference &&
                     sorted_timings.available &&
                     sorted_timings.fluid_neighbor_sort.launch_count == 1U &&
                     sorted_timings.fluid_neighbor_forces.launch_count == 1U,
                 "Fluid spatial sort differs from the stable CUDA-order reference"))
        return 1;

    FluidOptions block_sorted_options = sorted_options;
    block_sorted_options.capacity = 513U;
    std::vector<FluidParticle> block_sorted_particles(
        block_sorted_options.capacity);
    for (std::uint32_t index = 0U;
         index < block_sorted_particles.size(); ++index) {
        const std::uint32_t logical = (index * 257U) % 513U;
        const std::uint32_t x = logical % 9U;
        const std::uint32_t y = (logical / 9U) % 8U;
        const std::uint32_t z = logical / 72U;
        block_sorted_particles[index] = {
            {0.07F * (static_cast<float>(x) - 4.0F),
             0.07F * (static_cast<float>(y) - 3.5F),
             0.07F * (static_cast<float>(z) - 3.5F)},
            {0.003F * (static_cast<float>(z) - 3.5F),
             -0.002F * (static_cast<float>(x) - 4.0F),
             0.001F * (static_cast<float>(y) - 3.5F)},
            20.0F};
    }
    const std::vector<Vec3> block_sorted_expected =
        fluid_reference_accelerations(block_sorted_particles,
                                      block_sorted_options);
    World block_sorted_world;
    status = World::create({}, block_sorted_world);
    FluidId block_sorted_fluid{};
    status = status ? block_sorted_world.add_fluid(
                          block_sorted_options,
                          HostSpan<const FluidParticle>{
                              block_sorted_particles.data(),
                              block_sorted_particles.size()},
                          block_sorted_fluid)
                    : status;
    status = status ? block_sorted_world.step(
                          {.timestep = 1.0e-5F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    FluidDeviceView block_sorted_view{};
    status = status ? block_sorted_world.fluid_view(
                          block_sorted_fluid, block_sorted_view)
                    : status;
    bool block_sorted_match = status.ok() &&
                              block_sorted_view.particle_count ==
                                  block_sorted_particles.size();
    if (block_sorted_match) {
        const Vec3 *actual = contents(block_sorted_view.accelerations);
        const std::uint32_t *ids =
            contents(block_sorted_view.stable_particle_ids);
        for (std::uint32_t index = 0U;
             index < block_sorted_particles.size(); ++index) {
            const Vec3 expected_acceleration = block_sorted_expected[index];
            const float scale = std::max(
                {std::abs(expected_acceleration.x),
                 std::abs(expected_acceleration.y),
                 std::abs(expected_acceleration.z), 1.0F});
            const float tolerance = 1.0e-3F + 1.0e-4F * scale;
            block_sorted_match &= ids[index] == index;
            block_sorted_match &=
                std::abs(actual[index].x - expected_acceleration.x) <=
                    tolerance &&
                std::abs(actual[index].y - expected_acceleration.y) <=
                    tolerance &&
                std::abs(actual[index].z - expected_acceleration.z) <=
                    tolerance;
        }
    }
    if (!require(block_sorted_match,
                 "Multi-block Metal radix sort differs from the stable CUDA-order reference"))
        return 1;

    FluidOptions iterated_options = sorted_options;
    iterated_options.solver_iterations = 4U;
    FluidOptions substepped_options = sorted_options;
    substepped_options.solver_iterations = 1U;
    World iterated_world;
    World substepped_world;
    status = World::create({}, iterated_world);
    status = status ? World::create({}, substepped_world) : status;
    FluidId iterated_fluid{};
    FluidId substepped_fluid{};
    const HostSpan<const FluidParticle> initial{
        sorted_particles.data(), sorted_particles.size()};
    status = status ? iterated_world.add_fluid(
                          iterated_options, initial, iterated_fluid)
                    : status;
    status = status ? substepped_world.add_fluid(
                          substepped_options, initial, substepped_fluid)
                    : status;
    status = status ? iterated_world.step({.timestep = 1.0F / 120.0F,
                                           .substeps = 1U,
                                           .gravity = {},
                                           .collect_kernel_timings = true})
                    : status;
    status = status ? substepped_world.step(
                          {.timestep = 1.0F / 120.0F,
                           .substeps = 4U,
                           .gravity = {}})
                    : status;
    FluidDeviceView iterated_view{};
    FluidDeviceView substepped_view{};
    status = status ? iterated_world.fluid_view(iterated_fluid, iterated_view)
                    : status;
    status = status ? substepped_world.fluid_view(
                          substepped_fluid, substepped_view)
                    : status;
    bool iteration_match = status.ok() &&
                           iterated_view.particle_count ==
                               substepped_view.particle_count;
    if (iteration_match) {
        const Vec3 *iterated_positions = contents(iterated_view.positions);
        const Vec3 *iterated_velocities = contents(iterated_view.velocities);
        const Vec3 *substepped_positions = contents(substepped_view.positions);
        const Vec3 *substepped_velocities =
            contents(substepped_view.velocities);
        for (std::uint32_t index = 0U; index < iterated_view.particle_count;
             ++index) {
            iteration_match &=
                distance(iterated_positions[index],
                         substepped_positions[index]) <= 2.0e-6F &&
                distance(iterated_velocities[index],
                         substepped_velocities[index]) <= 2.0e-5F;
        }
    }
    WorldStepTimings iteration_timings{};
    status = status ? iterated_world.collect_step_timings(iteration_timings)
                    : status;
    if (!require(status.ok() && iteration_match &&
                     iteration_timings.fluid_neighbor_sort.launch_count ==
                         iterated_options.solver_iterations &&
                     iteration_timings.fluid_neighbor_forces.launch_count ==
                         iterated_options.solver_iterations &&
                     iteration_timings.fluid_integration.launch_count ==
                         iterated_options.solver_iterations,
                 "Fluid solver iterations do not match equivalent substeps"))
        return 1;

    World overflow_world;
    status = World::create({}, overflow_world);
    const std::array<FluidParticle, 3> dense_particles{{
        {{-0.01F, 0.0F, 0.0F}, {}, 20.0F},
        {{0.0F, 0.0F, 0.0F}, {}, 20.0F},
        {{0.01F, 0.0F, 0.0F}, {}, 20.0F},
    }};
    FluidId overflow_fluid{};
    status = status ? overflow_world.add_fluid(
                          {.capacity = 3U,
                           .particle_radius = 0.01F,
                           .support_radius = 0.08F,
                           .solver_iterations = 1U,
                           .maximum_neighbors = 1U,
                           .repulsion = 0.0F,
                           .velocity_damping = 0.0F},
                          HostSpan<const FluidParticle>{
                              dense_particles.data(), dense_particles.size()},
                          overflow_fluid)
                    : status;
    FrameToken overflow_completion;
    status = status ? overflow_world.step_async(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}},
                          overflow_completion)
                    : status;
    if (!require(status.ok(), "Fluid overflow frame submission failed"))
        return 1;
    status = overflow_completion.wait();
    WorldStatistics overflow_statistics{};
    const Status statistics_status =
        overflow_world.collect_statistics(overflow_statistics);
    if (!require(status.code == StatusCode::capacity_exceeded &&
                     statistics_status.ok() &&
                     overflow_statistics.maximum_fluid_neighbor_count == 2U,
                 "Fluid neighbor overflow was not propagated from the GPU"))
        return 1;
    const Status remove_status = overflow_world.remove_fluid(overflow_fluid);
    const Status clear_frame_status = remove_status
        ? overflow_world.step({.timestep = 1.0F / 60.0F,
                               .substeps = 1U,
                               .gravity = {}})
        : remove_status;
    const Status retained_overflow = overflow_completion.wait();
    if (!require(clear_frame_status.ok() &&
                     retained_overflow.code == StatusCode::capacity_exceeded,
                 "Completed frame lost its neighbor-overflow snapshot"))
        return 1;

    const std::array<Vec3, 4> floor_vertices{{
        {-2.0F, 0.0F, -2.0F}, {2.0F, 0.0F, -2.0F},
        {2.0F, 0.0F, 2.0F}, {-2.0F, 0.0F, 2.0F},
    }};
    const std::array<std::uint32_t, 6> floor_indices{{
        0U, 2U, 1U, 0U, 3U, 2U,
    }};
    const FluidParticle fast_drop{
        {0.0F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}, 20.0F};
    const FluidOptions contact_options{
        .capacity = 1U,
        .particle_radius = 0.1F,
        .rest_density = 125.0F,
        .support_radius = 0.2F,
        .solver_iterations = 1U,
        .repulsion = 0.0F,
        .viscosity = 0.0F,
        .velocity_damping = 0.0F,
        .maximum_speed = 100.0F,
    };
    {
        World contact_world;
        status = World::create({}, contact_world);
        TriangleMeshId mesh{};
        status = status ? contact_world.add_triangle_mesh(
                              {floor_vertices.data(), floor_vertices.size()},
                              {floor_indices.data(), floor_indices.size()}, mesh)
                        : status;
        RigidBodyId floor{};
        status = status ? contact_world.add_rigid_body(
                              {.motion = MotionType::static_body, .mesh = mesh},
                              floor)
                        : status;
        FluidId drop{};
        status = status ? contact_world.add_fluid(
                              contact_options, {&fast_drop, 1U}, drop)
                        : status;
        status = status ? contact_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {},
                               .collect_fluid_contacts = true})
                        : status;
        FluidDeviceView drop_view{};
        status = status ? contact_world.fluid_view(drop, drop_view) : status;
        if (!require(
                status.ok() && contents(drop_view.positions)[0].y >= 0.099F &&
                    contents(drop_view.foam)[0] > 0.5F &&
                    contact_world.contacts().event_count == 1U,
                "Swept static triangle contact did not stop a fast fluid particle"))
            return 1;
    }
    {
        World overflow_contact_world;
        status = World::create({.contact_capacity = 1U},
                               overflow_contact_world);
        TriangleMeshId mesh{};
        status = status ? overflow_contact_world.add_triangle_mesh(
                              {floor_vertices.data(), floor_vertices.size()},
                              {floor_indices.data(), floor_indices.size()}, mesh)
                        : status;
        RigidBodyId floor{};
        status = status ? overflow_contact_world.add_rigid_body(
                              {.motion = MotionType::static_body, .mesh = mesh},
                              floor)
                        : status;
        const std::array<FluidParticle, 4> drops{{
            {{-0.6F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}, 20.0F},
            {{-0.2F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}, 20.0F},
            {{0.2F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}, 20.0F},
            {{0.6F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}, 20.0F},
        }};
        FluidId fluid{};
        FluidOptions options = contact_options;
        options.capacity = static_cast<std::uint32_t>(drops.size());
        status = status ? overflow_contact_world.add_fluid(
                              options,
                              {drops.data(), drops.size()}, fluid)
                        : status;
        status = status ? overflow_contact_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {},
                               .collect_fluid_contacts = true})
                        : status;
        WorldStatistics statistics{};
        status = status ? overflow_contact_world.collect_statistics(statistics)
                        : status;
        const ContactDeviceView contacts = overflow_contact_world.contacts();
        if (!require(status.ok() && contacts.event_count == 1U &&
                         contacts.overflowed &&
                         statistics.contact_count == 1U &&
                         statistics.contact_overflow_count == 3U,
                     "Fluid contact statistics collapsed the overflow count"))
            return 1;
    }
    {
        World moving_world;
        status = World::create({}, moving_world);
        TriangleMeshId mesh{};
        status = status ? moving_world.add_triangle_mesh(
                              {floor_vertices.data(), floor_vertices.size()},
                              {floor_indices.data(), floor_indices.size()}, mesh)
                        : status;
        RigidBodyId floor{};
        status = status ? moving_world.add_rigid_body(
                              {.motion = MotionType::kinematic,
                               .mesh = mesh,
                               .initial_state = {
                                   .position = {0.0F, 1.0F, 0.0F}}},
                              floor)
                        : status;
        const FluidParticle stationary{{0.0F, 0.2F, 0.0F}, {}, 20.0F};
        FluidId fluid{};
        status = status ? moving_world.add_fluid(
                              contact_options, {&stationary, 1U}, fluid)
                        : status;
        status = status ? moving_world.set_kinematic_target(
                              floor,
                              {.position = {0.0F, -1.0F, 0.0F}})
                        : status;
        status = status ? moving_world.step(
                              {.timestep = 0.1F,
                               .substeps = 4U,
                               .gravity = {},
                               .collect_fluid_contacts = true})
                        : status;
        FluidDeviceView fluid_view{};
        status = status ? moving_world.fluid_view(fluid, fluid_view) : status;
        if (status.ok() && fluid_view.particle_count != 0U &&
            !(contents(fluid_view.positions)[0].y < -1.05F &&
              contents(fluid_view.velocities)[0].y < -10.0F &&
              moving_world.contacts().event_count == 1U))
            std::cerr << "fluid frame sweep y="
                      << contents(fluid_view.positions)[0].y
                      << " vy=" << contents(fluid_view.velocities)[0].y
                      << " contacts="
                      << moving_world.contacts().event_count << '\n';
        if (!require(status.ok() &&
                         contents(fluid_view.positions)[0].y < -1.05F &&
                         contents(fluid_view.velocities)[0].y < -10.0F &&
                         moving_world.contacts().event_count == 1U,
                     "Swept kinematic triangle did not carry a fluid particle"))
            return 1;
    }
    {
        World reaction_world;
        status = World::create({}, reaction_world);
        TriangleMeshId mesh{};
        status = status ? reaction_world.add_triangle_mesh(
                              {floor_vertices.data(), floor_vertices.size()},
                              {floor_indices.data(), floor_indices.size()}, mesh)
                        : status;
        RigidBodyId body{};
        status = status ? reaction_world.add_rigid_body(
                              {.motion = MotionType::dynamic,
                               .mesh = mesh,
                               .mass = 1.0F,
                               .linear_damping = 0.0F,
                               .angular_damping = 0.0F},
                              body)
                        : status;
        FluidId fluid{};
        status = status ? reaction_world.add_fluid(
                              contact_options, {&fast_drop, 1U}, fluid)
                        : status;
        status = status ? reaction_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {},
                               .collect_fluid_contacts = true})
                        : status;
        FluidDeviceView fluid_view{};
        RigidBodyState body_state{};
        status = status ? reaction_world.fluid_view(fluid, fluid_view) : status;
        status = status ? reaction_world.read_rigid_body_state(body, body_state)
                        : status;
        const float particle_velocity = status.ok()
                                            ? contents(fluid_view.velocities)[0].y
                                            : 0.0F;
        if (!require(status.ok() && particle_velocity > -4.0F &&
                         body_state.linear_velocity.y < -1.0F &&
                         std::abs(particle_velocity +
                                  body_state.linear_velocity.y + 5.0F) <
                             0.05F &&
                         std::abs(body_state.position.y) < 1.0e-5F,
                     "Fluid and dynamic triangle did not exchange balanced momentum"))
            return 1;
        const ContactDeviceView contact_view = reaction_world.contacts();
        const bool contact_matches = contact_view.event_count == 1U &&
            !contact_view.overflowed && contact_view.events.size == 1U &&
            contents(contact_view.events)[0].fluid == fluid &&
            contents(contact_view.events)[0].rigid_body == body &&
            contents(contact_view.events)[0].stable_particle_id == 0U &&
            contents(contact_view.events)[0].normal.y > 0.9F &&
            contents(contact_view.events)[0].normal_impulse > 1.0F;
        if (!require(contact_matches,
                     "Fluid contact event lost CUDA identity or impulse semantics"))
            return 1;
    }
    {
        World corner_world;
        status = World::create({}, corner_world);
        TriangleMeshId mesh{};
        status = status ? corner_world.add_triangle_mesh(
                              {floor_vertices.data(), floor_vertices.size()},
                              {floor_indices.data(), floor_indices.size()}, mesh)
                        : status;
        RigidBodyId floor{};
        status = status ? corner_world.add_rigid_body(
                              {.motion = MotionType::static_body,
                               .mesh = mesh},
                              floor)
                        : status;
        RigidBodyId moving_ceiling{};
        status = status ? corner_world.add_rigid_body(
                              {.motion = MotionType::kinematic,
                               .mesh = mesh,
                               .initial_state = {
                                   .position = {0.0F, 1.0F, 0.0F}}},
                              moving_ceiling)
                        : status;
        const FluidParticle stationary{{0.0F, 0.2F, 0.0F}, {}, 20.0F};
        FluidId fluid{};
        status = status ? corner_world.add_fluid(
                              contact_options, {&stationary, 1U}, fluid)
                        : status;
        status = status ? corner_world.set_kinematic_target(
                              moving_ceiling,
                              {.position = {0.0F, 0.15F, 0.0F}})
                        : status;
        status = status ? corner_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {},
                               .collect_fluid_contacts = true})
                        : status;
        FluidDeviceView fluid_view{};
        status = status ? corner_world.fluid_view(fluid, fluid_view) : status;
        if (!require(
                status.ok() && contents(fluid_view.positions)[0].y >= 0.099F &&
                    corner_world.contacts().event_count == 1U &&
                    contents(corner_world.contacts().events)[0].rigid_body ==
                        moving_ceiling,
                "Moving rigid contact suppressed the following static boundary"))
            return 1;
    }
    {
        World overlap_world;
        status = World::create({}, overlap_world);
        TriangleMeshId mesh{};
        status = status ? overlap_world.add_triangle_mesh(
                              {floor_vertices.data(), floor_vertices.size()},
                              {floor_indices.data(), floor_indices.size()}, mesh)
                        : status;
        RigidBodyId body{};
        status = status ? overlap_world.add_rigid_body(
                              {.motion = MotionType::dynamic,
                               .mesh = mesh,
                               .mass = 1.0F,
                               .linear_damping = 0.0F,
                               .angular_damping = 0.0F},
                              body)
                        : status;
        const FluidParticle overlapping{{0.0F, 0.05F, 0.0F}, {}, 20.0F};
        FluidId fluid{};
        status = status ? overlap_world.add_fluid(
                              contact_options,
                              HostSpan<const FluidParticle>{&overlapping, 1U},
                              fluid)
                        : status;
        status = status ? overlap_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        FluidDeviceView fluid_view{};
        RigidBodyState body_state{};
        status = status ? overlap_world.fluid_view(fluid, fluid_view) : status;
        status = status ? overlap_world.read_rigid_body_state(body, body_state)
                        : status;
        const float particle_velocity = status.ok()
                                            ? contents(fluid_view.velocities)[0].y
                                            : 0.0F;
        if (!require(status.ok() && particle_velocity > 0.01F &&
                         body_state.linear_velocity.y < -0.01F &&
                         std::abs(particle_velocity +
                                  body_state.linear_velocity.y) < 0.01F,
                     "Resting fluid overlap did not share recovery momentum"))
            return 1;
    }
    {
        World batch_world;
        status = World::create({}, batch_world);
        TriangleMeshId mesh{};
        status = status ? batch_world.add_triangle_mesh(
                              {floor_vertices.data(), floor_vertices.size()},
                              {floor_indices.data(), floor_indices.size()}, mesh)
                        : status;
        RigidBodyId body{};
        status = status ? batch_world.add_rigid_body(
                              {.motion = MotionType::dynamic,
                               .mesh = mesh,
                               .mass = 1.0F,
                               .linear_damping = 0.0F,
                               .angular_damping = 0.0F},
                              body)
                        : status;
        const std::array<FluidParticle, 2> impacts{{
            {{-1.0F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}, 20.0F},
            {{1.0F, 0.2F, 0.0F}, {0.0F, -5.0F, 0.0F}, 20.0F},
        }};
        FluidOptions batch_options = contact_options;
        batch_options.capacity = 2U;
        FluidId fluid{};
        status = status ? batch_world.add_fluid(
                              batch_options,
                              {impacts.data(), impacts.size()}, fluid)
                        : status;
        status = status ? batch_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        FluidDeviceView fluid_view{};
        RigidBodyState body_state{};
        status = status ? batch_world.fluid_view(fluid, fluid_view) : status;
        status = status ? batch_world.read_rigid_body_state(body, body_state)
                        : status;
        float momentum = body_state.linear_velocity.y;
        if (status.ok())
            momentum += contents(fluid_view.velocities)[0].y +
                        contents(fluid_view.velocities)[1].y;
        if (!require(status.ok() && body_state.linear_velocity.y < -1.0F &&
                         body_state.linear_velocity.y > -3.5F &&
                         std::abs(momentum + 10.0F) < 0.05F,
                     "Batched fluid impacts over-accelerated a shared rigid body"))
            return 1;
    }
    {
        constexpr std::uint32_t sides = 16U;
        constexpr float pi = 3.14159265358979323846F;
        std::vector<Vec3> cylinder_vertices;
        std::vector<std::uint32_t> cylinder_indices;
        cylinder_vertices.reserve(2U * sides);
        cylinder_indices.reserve(6U * sides);
        for (std::uint32_t side = 0U; side < sides; ++side) {
            const float angle = 2.0F * pi * side / sides;
            cylinder_vertices.push_back(
                {std::cos(angle), -1.0F, std::sin(angle)});
            cylinder_vertices.push_back(
                {std::cos(angle), 1.0F, std::sin(angle)});
        }
        for (std::uint32_t side = 0U; side < sides; ++side) {
            const std::uint32_t first = side * 2U;
            const std::uint32_t next = ((side + 1U) % sides) * 2U;
            cylinder_indices.insert(cylinder_indices.end(),
                                    {first, next, next + 1U,
                                     first, next + 1U, first + 1U});
        }
        World cylinder_world;
        status = World::create({}, cylinder_world);
        TriangleMeshId cylinder_mesh{};
        status = status ? cylinder_world.add_triangle_mesh(
                              HostSpan<const Vec3>{
                                  cylinder_vertices.data(),
                                  cylinder_vertices.size()},
                              HostSpan<const std::uint32_t>{
                                  cylinder_indices.data(),
                                  cylinder_indices.size()},
                              cylinder_mesh)
                        : status;
        RigidBodyId cylinder_body{};
        status = status ? cylinder_world.add_rigid_body(
                              {.motion = MotionType::static_body,
                               .mesh = cylinder_mesh},
                              cylinder_body)
                        : status;
        const FluidParticle cylinder_drop{
            {1.3F, 0.0F, 0.0F}, {-5.0F, 0.0F, 0.0F}, 20.0F};
        FluidId cylinder_fluid{};
        status = status ? cylinder_world.add_fluid(
                              contact_options,
                              HostSpan<const FluidParticle>{
                                  &cylinder_drop, 1U},
                              cylinder_fluid)
                        : status;
        status = status ? cylinder_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        FluidDeviceView cylinder_view{};
        status = status ? cylinder_world.fluid_view(
                              cylinder_fluid, cylinder_view)
                        : status;
        if (!require(status.ok() &&
                         contents(cylinder_view.positions)[0].x > 1.05F,
                     "Fluid tunneled through authored cylinder triangles"))
            return 1;
    }
    return 0;
}

int cloth_test() {
    World world;
    Status status = World::create({}, world);
    if (!require(status.ok(), "Cloth world creation failed")) return 1;
    const std::array<Vec3, 3> zero_edge_vertices{{
        {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F},
    }};
    const std::array<std::uint32_t, 3> zero_edge_indices{{0U, 1U, 2U}};
    ClothId cloth{};
    status = world.add_cloth(
        {.vertices = {zero_edge_vertices.data(), zero_edge_vertices.size()},
         .triangle_indices = {zero_edge_indices.data(),
                              zero_edge_indices.size()}},
        cloth);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Metal accepted a CUDA-invalid zero-length cloth edge"))
        return 1;
    const std::array<Vec3, 4> vertices{{
        {-0.5F, 1.0F, 0.0F}, {0.5F, 1.0F, 0.0F},
        {-0.5F, 0.5F, 0.0F}, {0.5F, 0.5F, 0.0F},
    }};
    const std::array<std::uint32_t, 6> indices{{0U, 2U, 1U, 1U, 2U, 3U}};
    const std::array<float, 4> inverse_masses{{0.0F, 0.0F, 20.0F, 20.0F}};
    status = world.add_cloth(
        {.vertices = {vertices.data(), vertices.size()},
         .triangle_indices = {indices.data(), indices.size()},
         .inverse_masses = {inverse_masses.data(), inverse_masses.size()},
         .solver_iterations = 12U},
        cloth);
    if (!require(status.ok(), "Cloth creation failed")) return 1;
    if (!step(world, 30U, {0.0F, -9.81F, 0.0F}, 4U)) return 1;
    ClothDeviceView view{};
    status = world.cloth_view(cloth, view);
    const Vec3 *positions = status ? contents(view.positions) : nullptr;
    float maximum_strain = 0.0F;
    bool found_bending = false;
    if (status.ok()) {
        const ClothBond *bonds = contents(view.bonds);
        for (std::uint64_t index = 0; index < view.bonds.size; ++index) {
            found_bending |= bonds[index].bending;
            maximum_strain = std::max(
                maximum_strain,
                std::abs(distance(positions[bonds[index].first],
                                  positions[bonds[index].second]) /
                             bonds[index].rest_length -
                         1.0F));
        }
    }
    if (!require(status.ok() && view.vertex_count == vertices.size() &&
                     distance(positions[0], vertices[0]) < 1.0e-6F &&
                     distance(positions[1], vertices[1]) < 1.0e-6F &&
                     found_bending && maximum_strain < 0.02F &&
                     finite(positions[2]) &&
                     finite(positions[3]),
                 "Cloth pins or constrained motion are invalid"))
        return 1;

    World fracture_world;
    status = World::create(
        {.physics_debug = {.frame_capacity = 1U, .frame_stride = 1U}},
        fracture_world);
    if (!require(status.ok(), "Fracture world creation failed")) return 1;
    const std::array<Vec3, 4> fracture_vertices{{
        {0.0F, 1.0F, 0.0F}, {0.5F, 1.0F, 0.0F},
        {0.0F, 1.0F, 0.5F}, {0.5F, 1.0F, 0.5F},
    }};
    const std::array<std::uint32_t, 6> fracture_indices{{
        0U, 2U, 1U, 1U, 2U, 3U,
    }};
    const std::array<float, 4> fracture_inverse{{20.0F, 0.0F, 20.0F, 20.0F}};
    ClothId fracture_cloth{};
    status = fracture_world.add_cloth(
        {.vertices = {fracture_vertices.data(), fracture_vertices.size()},
         .triangle_indices = {fracture_indices.data(),
                              fracture_indices.size()},
         .inverse_masses = {fracture_inverse.data(), fracture_inverse.size()},
         .stretch_compliance = 1.0F,
         .solver_iterations = 1U,
         .break_strain = 0.01F,
         .fracture_persistence_substeps = 2U},
        fracture_cloth);
    if (!require(status.ok(), "Fracture cloth creation failed")) return 1;
    status = fracture_world.step(
        {.timestep = 1.0F / 60.0F,
         .substeps = 1U,
         .gravity = {0.0F, -1000.0F, 0.0F}});
    ClothDeviceView fracture_view{};
    status = status ? fracture_world.cloth_view(fracture_cloth, fracture_view)
                    : status;
    bool first_pass_active = status.ok();
    if (status.ok())
        for (std::uint64_t index = 0; index < fracture_view.active_bonds.size;
             ++index)
            first_pass_active &= contents(fracture_view.active_bonds)[index] != 0U;
    status = status ? fracture_world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {0.0F, -1000.0F, 0.0F}})
                    : status;
    status = status ? fracture_world.cloth_view(fracture_cloth, fracture_view)
                    : status;
    bool broke = false;
    if (status.ok())
        for (std::uint64_t index = 0; index < fracture_view.active_bonds.size;
             ++index)
            broke |= contents(fracture_view.active_bonds)[index] == 0U;
    status = status ? fracture_world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    status = status ? fracture_world.cloth_view(fracture_cloth, fracture_view)
                    : status;
    bool split_topology = fracture_view.vertex_count > fracture_vertices.size();
    for (std::uint64_t corner = 0;
         corner < fracture_view.triangle_indices.size; ++corner)
        split_topology |= contents(fracture_view.triangle_indices)[corner] >=
                          fracture_vertices.size();
    PhysicsDebugFrameView fracture_debug{};
    status = status ? fracture_world.physics_debug_frame(fracture_debug)
                    : status;
    if (!require(status.ok() && first_pass_active && broke && split_topology &&
                     fracture_view.vertex_source_indices.size ==
                         fracture_view.vertex_count &&
                     fracture_view.surface_positions.size ==
                         fracture_indices.size() &&
                     fracture_debug.cloth_vertices.size ==
                         fracture_view.vertex_count,
                 "Cloth fracture persistence or physical split is invalid"))
        return 1;

    World contact_world;
    status = World::create({}, contact_world);
    const std::array<Vec3, 3> floor_vertices{{
        {-1.0F, 0.0F, -1.0F}, {1.0F, 0.0F, -1.0F},
        {0.0F, 0.0F, 1.0F},
    }};
    const std::array<std::uint32_t, 3> floor_indices{{0U, 2U, 1U}};
    TriangleMeshId floor_mesh{};
    status = status ? contact_world.add_triangle_mesh(
                          {floor_vertices.data(), floor_vertices.size()},
                          {floor_indices.data(), floor_indices.size()},
                          floor_mesh)
                    : status;
    RigidBodyId floor{};
    status = status ? contact_world.add_rigid_body(
                          {.motion = MotionType::static_body,
                           .mesh = floor_mesh},
                          floor)
                    : status;
    const std::array<Vec3, 3> contact_vertices{{
        {-0.1F, 0.005F, -0.1F}, {0.1F, 0.005F, -0.1F},
        {0.0F, 0.005F, 0.1F},
    }};
    const std::array<std::uint32_t, 3> contact_indices{{0U, 1U, 2U}};
    const std::array<float, 3> contact_inverse{{10.0F, 10.0F, 10.0F}};
    ClothId contact_cloth{};
    status = status ? contact_world.add_cloth(
                          {.vertices = {contact_vertices.data(),
                                        contact_vertices.size()},
                           .triangle_indices = {contact_indices.data(),
                                                contact_indices.size()},
                           .inverse_masses = {contact_inverse.data(),
                                              contact_inverse.size()},
                           .thickness = 0.02F,
                           .contact_friction = 0.0F,
                           .solver_iterations = 1U,
                           .impact_break_impulse = 1.0e-6F},
                          contact_cloth)
                    : status;
    if (!require(status.ok(), status.message ? status.message
                                             : "Rigid-cloth contact setup failed"))
        return 1;
    if (!step(contact_world, 1U, {0.0F, -9.81F, 0.0F}, 2U)) return 1;
    ClothDeviceView contact_view{};
    status = contact_world.cloth_view(contact_cloth, contact_view);
    bool rigid_force = false;
    bool impact_broke = false;
    for (std::uint64_t vertex = 0;
         vertex < contact_view.rigid_contact_forces.size; ++vertex)
        rigid_force |=
            distance(contents(contact_view.rigid_contact_forces)[vertex], {}) >
            1.0e-4F;
    for (std::uint64_t bond = 0; bond < contact_view.active_bonds.size; ++bond)
        impact_broke |= contents(contact_view.active_bonds)[bond] == 0U;
    if (!require(status.ok() && rigid_force && impact_broke,
                 "Rigid-cloth impact diagnostics or fracture are missing"))
        return 1;

    {
        // The tetrahedron crosses the middle of a large cloth face while all
        // three cloth nodes remain far outside the rigid mesh. This exercises
        // the body-against-face phase independently of node-against-body
        // contact.
        World surface_world;
        status = World::create({}, surface_world);
        const std::array<Vec3, 3> surface_vertices{{
            {-1.0F, 0.0F, -1.0F}, {1.0F, 0.0F, -1.0F},
            {0.0F, 0.0F, 1.0F},
        }};
        const std::array<std::uint32_t, 3> surface_indices{{0U, 2U, 1U}};
        const std::array<float, 3> surface_inverse{{10.0F, 10.0F, 10.0F}};
        ClothId surface_cloth{};
        status = status ? surface_world.add_cloth(
                              {.vertices = {surface_vertices.data(),
                                            surface_vertices.size()},
                               .triangle_indices = {surface_indices.data(),
                                                    surface_indices.size()},
                               .inverse_masses = {surface_inverse.data(),
                                                  surface_inverse.size()},
                               .thickness = 0.02F,
                               .contact_friction = 0.0F,
                               .solver_iterations = 1U},
                              surface_cloth)
                        : status;
        const std::array<Vec3, 4> body_vertices{{
            {-0.05F, -0.05F, -0.05F}, {0.05F, -0.05F, -0.05F},
            {0.0F, 0.05F, -0.05F}, {0.0F, 0.0F, 0.05F},
        }};
        const std::array<std::uint32_t, 12> body_indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        TriangleMeshId body_mesh{};
        status = status ? surface_world.add_triangle_mesh(
                              {body_vertices.data(), body_vertices.size()},
                              {body_indices.data(), body_indices.size()},
                              body_mesh)
                        : status;
        RigidBodyId moving_body{};
        status = status ? surface_world.add_rigid_body(
                              {.motion = MotionType::dynamic,
                               .mesh = body_mesh,
                               .initial_state = {
                                   .position = {0.0F, 0.2F, 0.0F},
                                   .linear_velocity = {0.0F, -5.0F, 0.0F}},
                               .mass = 1.0F,
                               .linear_damping = 0.0F,
                               .angular_damping = 0.0F,
                               .collision_margin = 0.0F},
                              moving_body)
                        : status;
        status = status ? surface_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        ClothDeviceView surface_view{};
        RigidBodyState body_state{};
        status = status ? surface_world.cloth_view(surface_cloth, surface_view)
                        : status;
        status = status ? surface_world.read_rigid_body_state(
                              moving_body, body_state)
                        : status;
        float mean_cloth_y = 0.0F;
        if (status.ok())
            for (std::uint64_t vertex = 0U;
                 vertex < surface_view.positions.size; ++vertex)
                mean_cloth_y += contents(surface_view.positions)[vertex].y /
                                surface_view.positions.size;
        if (!require(status.ok() && finite(body_state.position) &&
                         finite(body_state.linear_velocity) &&
                         body_state.linear_velocity.y > -0.1F &&
                         body_state.position.y > -0.2F &&
                         mean_cloth_y < -0.05F,
                     "Rigid body crossed a cloth triangle interior"))
            return 1;
    }
    return 0;
}

int soft_body_test() {
    World world;
    Status status = World::create({}, world);
    if (!require(status.ok(), "Soft-body world creation failed")) return 1;
    const std::array<Vec3, 4> nodes{{
        {0.0F, 1.0F, 0.0F}, {0.3F, 1.0F, 0.0F},
        {0.0F, 1.3F, 0.0F}, {0.0F, 1.0F, 0.3F},
    }};
    const auto rest = [&](std::uint32_t a, std::uint32_t b) {
        return distance(nodes[a], nodes[b]);
    };
    const std::array<SoftBodyBond, 6> bonds{{
        {0U, 1U, rest(0U, 1U)}, {0U, 2U, rest(0U, 2U)},
        {0U, 3U, rest(0U, 3U)}, {1U, 2U, rest(1U, 2U)},
        {1U, 3U, rest(1U, 3U)}, {2U, 3U, rest(2U, 3U)},
    }};
    std::array<Vec3, 4> surface = nodes;
    for (Vec3 &point : surface) point.z += 0.1F;
    const std::array<std::uint32_t, 12> indices{{
        0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
    }};
    std::array<SoftBodySurfaceBinding, 4> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].nodes[0] = index;
        bindings[index].weights[0] = 1.0F;
    }
    SoftBodyId soft{};
    const std::array<SoftBodyBond, 3> isolated_bonds{{
        bonds[0], bonds[1], bonds[3],
    }};
    status = world.add_soft_body(
        {.nodes = {nodes.data(), nodes.size()},
         .bonds = {isolated_bonds.data(), isolated_bonds.size()},
         .surface_vertices = {surface.data(), surface.size()},
         .surface_triangle_indices = {indices.data(), indices.size()},
         .surface_bindings = {bindings.data(), bindings.size()}},
        soft);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Metal accepted an isolated soft-body node"))
        return 1;
    const std::array<float, 4> fixed_inverse_masses{};
    status = world.add_soft_body(
        {.nodes = {nodes.data(), nodes.size()},
         .bonds = {bonds.data(), bonds.size()},
         .inverse_masses = {fixed_inverse_masses.data(),
                            fixed_inverse_masses.size()},
         .surface_vertices = {surface.data(), surface.size()},
         .surface_triangle_indices = {indices.data(), indices.size()},
         .surface_bindings = {bindings.data(), bindings.size()}},
        soft);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Metal accepted a soft body with no movable nodes"))
        return 1;
    const std::array<float, 4> planar_inverse_masses{{1.0F, 1.0F, 1.0F,
                                                      0.0F}};
    status = world.add_soft_body(
        {.nodes = {nodes.data(), nodes.size()},
         .bonds = {bonds.data(), bonds.size()},
         .inverse_masses = {planar_inverse_masses.data(),
                            planar_inverse_masses.size()},
         .surface_vertices = {surface.data(), surface.size()},
         .surface_triangle_indices = {indices.data(), indices.size()},
         .surface_bindings = {bindings.data(), bindings.size()},
         .shape_matching_stiffness = 0.5F},
        soft);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Shape matching accepted a planar movable rest lattice"))
        return 1;
    status = world.add_soft_body(
        {.nodes = {nodes.data(), nodes.size()},
         .bonds = {bonds.data(), bonds.size()},
         .surface_vertices = {surface.data(), surface.size()},
         .surface_triangle_indices = {indices.data(), indices.size()},
         .surface_bindings = {bindings.data(), bindings.size()},
         .spring_damping = 0.85F,
         .shape_matching_stiffness = 0.8F,
         .solver_iterations = 12U},
        soft);
    if (!require(status.ok(), status.message ? status.message
                                             : "Soft-body creation failed"))
        return 1;
    if (!step(world, 20U, {0.0F, -9.81F, 0.0F}, 4U)) return 1;
    SoftBodyDeviceView view{};
    status = world.soft_body_view(soft, view);
    const Vec3 *positions = status ? contents(view.positions) : nullptr;
    const Vec3 *skinned = status ? contents(view.surface_positions) : nullptr;
    if (!require(status.ok() && finite(positions[0]) && finite(skinned[0]) &&
                     std::abs((skinned[0].z - positions[0].z) - 0.1F) <
                         2.0e-3F,
                 "Soft-body shape matching or delta skinning is invalid"))
        return 1;

    const std::array<Vec3, 8> box_vertices{{
        {-0.5F, -0.5F, -0.5F}, {0.5F, -0.5F, -0.5F},
        {0.5F, 0.5F, -0.5F}, {-0.5F, 0.5F, -0.5F},
        {-0.5F, -0.5F, 0.5F}, {0.5F, -0.5F, 0.5F},
        {0.5F, 0.5F, 0.5F}, {-0.5F, 0.5F, 0.5F},
    }};
    const std::array<std::uint32_t, 36> box_indices{{
        0U, 2U, 1U, 0U, 3U, 2U, 4U, 5U, 6U, 4U, 6U, 7U,
        0U, 1U, 5U, 0U, 5U, 4U, 3U, 7U, 6U, 3U, 6U, 2U,
        0U, 4U, 7U, 0U, 7U, 3U, 1U, 2U, 6U, 1U, 6U, 5U,
    }};
    const std::array<Vec3, 4> trapped_nodes{{
        {-0.05F, -0.05F, -0.05F}, {0.05F, -0.05F, -0.05F},
        {-0.05F, 0.05F, -0.05F}, {-0.05F, -0.05F, 0.05F},
    }};
    const auto trapped_rest = [&](std::uint32_t a, std::uint32_t b) {
        return distance(trapped_nodes[a], trapped_nodes[b]);
    };
    const std::array<SoftBodyBond, 6> trapped_bonds{{
        {0U, 1U, trapped_rest(0U, 1U)},
        {0U, 2U, trapped_rest(0U, 2U)},
        {0U, 3U, trapped_rest(0U, 3U)},
        {1U, 2U, trapped_rest(1U, 2U)},
        {1U, 3U, trapped_rest(1U, 3U)},
        {2U, 3U, trapped_rest(2U, 3U)},
    }};
    const std::array<std::uint32_t, 12> trapped_indices{{
        0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
    }};
    std::array<SoftBodySurfaceBinding, 4> trapped_bindings{};
    for (std::uint32_t index = 0U; index < trapped_bindings.size(); ++index) {
        trapped_bindings[index].nodes[0] = index;
        trapped_bindings[index].weights[0] = 1.0F;
    }
    {
        World containment_world;
        status = World::create({}, containment_world);
        TriangleMeshId box_mesh{};
        status = status ? containment_world.add_triangle_mesh(
                              {box_vertices.data(), box_vertices.size()},
                              {box_indices.data(), box_indices.size()}, box_mesh)
                        : status;
        RigidBodyId box{};
        status = status ? containment_world.add_rigid_body(
                              {.motion = MotionType::static_body,
                               .mesh = box_mesh,
                               .collision_margin = 0.0F},
                              box)
                        : status;
        SoftBodyId trapped{};
        status = status ? containment_world.add_soft_body(
                              {.nodes = {trapped_nodes.data(),
                                         trapped_nodes.size()},
                               .bonds = {trapped_bonds.data(),
                                         trapped_bonds.size()},
                               .surface_vertices = {trapped_nodes.data(),
                                                    trapped_nodes.size()},
                               .surface_triangle_indices = {
                                   trapped_indices.data(),
                                   trapped_indices.size()},
                               .surface_bindings = {trapped_bindings.data(),
                                                    trapped_bindings.size()},
                               .node_radius = 0.05F,
                               .solver_iterations = 1U},
                              trapped)
                        : status;
        status = status ? containment_world.step(
                              {.timestep = 1.0F / 60.0F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        SoftBodyDeviceView trapped_view{};
        status = status ? containment_world.soft_body_view(
                              trapped, trapped_view)
                        : status;
        bool all_outside = status.ok();
        if (status.ok())
            for (std::uint64_t node = 0U;
                 node < trapped_view.positions.size; ++node) {
                const Vec3 position = contents(trapped_view.positions)[node];
                all_outside &= std::max({std::abs(position.x),
                                         std::abs(position.y),
                                         std::abs(position.z)}) >= 0.549F;
            }
        if (!require(status.ok() && all_outside,
                     "Closed rigid mesh did not recover trapped soft nodes"))
            return 1;
    }
    {
        World reaction_world;
        status = World::create({}, reaction_world);
        const std::array<Vec3, 3> plane_vertices{{
            {-1.0F, 0.0F, -1.0F}, {1.0F, 0.0F, -1.0F},
            {0.0F, 0.0F, 1.0F},
        }};
        const std::array<std::uint32_t, 3> plane_indices{{0U, 2U, 1U}};
        TriangleMeshId plane_mesh{};
        status = status ? reaction_world.add_triangle_mesh(
                              {plane_vertices.data(), plane_vertices.size()},
                              {plane_indices.data(), plane_indices.size()},
                              plane_mesh)
                        : status;
        RigidBodyId plane{};
        status = status ? reaction_world.add_rigid_body(
                              {.motion = MotionType::dynamic,
                               .mesh = plane_mesh,
                               .mass = 1.0F,
                               .linear_damping = 0.0F,
                               .angular_damping = 0.0F,
                               .collision_margin = 0.0F},
                              plane)
                        : status;
        const std::array<Vec3, 4> overlap_nodes{{
            {-0.03F, 0.01F, -0.03F}, {0.03F, 0.01F, -0.03F},
            {-0.03F, 0.01F, 0.03F}, {-0.03F, 0.04F, -0.03F},
        }};
        const auto overlap_rest = [&](std::uint32_t a, std::uint32_t b) {
            return distance(overlap_nodes[a], overlap_nodes[b]);
        };
        const std::array<SoftBodyBond, 6> overlap_bonds{{
            {0U, 1U, overlap_rest(0U, 1U)},
            {0U, 2U, overlap_rest(0U, 2U)},
            {0U, 3U, overlap_rest(0U, 3U)},
            {1U, 2U, overlap_rest(1U, 2U)},
            {1U, 3U, overlap_rest(1U, 3U)},
            {2U, 3U, overlap_rest(2U, 3U)},
        }};
        SoftBodyId overlap{};
        status = status ? reaction_world.add_soft_body(
                              {.nodes = {overlap_nodes.data(),
                                         overlap_nodes.size()},
                               .bonds = {overlap_bonds.data(),
                                         overlap_bonds.size()},
                               .surface_vertices = {overlap_nodes.data(),
                                                    overlap_nodes.size()},
                               .surface_triangle_indices = {
                                   trapped_indices.data(),
                                   trapped_indices.size()},
                               .surface_bindings = {trapped_bindings.data(),
                                                    trapped_bindings.size()},
                               .node_mass = 1.0F,
                               .node_radius = 0.05F,
                               .solver_iterations = 1U},
                              overlap)
                        : status;
        status = status ? reaction_world.step(
                              {.timestep = 1.0F / 60.0F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        RigidBodyState plane_state{};
        status = status ? reaction_world.read_rigid_body_state(
                              plane, plane_state)
                        : status;
        if (!require(status.ok() && plane_state.position.y < -0.02F,
                     "Soft-rigid overlap did not share position with the body"))
            return 1;
    }
    return 0;
}

int rope_test() {
    {
        World limit_world;
        Status limit_status = World::create({.rope_capacity = 1U}, limit_world);
        const std::array<Vec3, 2> line{{
            {0.0F, 0.0F, 0.0F}, {0.0F, -0.2F, 0.0F},
        }};
        RopeId rope{};
        limit_status = limit_status
                           ? limit_world.add_rope(
                                 {.centerline = {line.data(), line.size()},
                                  .node_spacing = 0.05F,
                                  .radius = 0.025F,
                                  .maximum_substep_timestep = 1.0e-8F},
                                 rope)
                           : limit_status;
        limit_status = limit_status
                           ? limit_world.step({.timestep = 1.0F / 60.0F})
                           : limit_status;
        if (!require(
                limit_status.code == StatusCode::invalid_argument,
                "Rope timestep limit exceeded CUDA's 1024-substep ceiling"))
            return 1;
    }
    {
        World crossing_world;
        Status crossing_status = World::create(
            {.rigid_body_capacity = 1U,
             .triangle_mesh_capacity = 1U,
             .rope_capacity = 1U},
            crossing_world);
        const std::array<Vec3, 4> vertices{{
            {-0.05F, -0.05F, -0.05F}, {0.05F, -0.05F, -0.05F},
            {0.0F, 0.05F, -0.05F}, {0.0F, 0.0F, 0.05F},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        TriangleMeshId mesh{};
        crossing_status = crossing_status
                              ? crossing_world.add_triangle_mesh(
                                    {vertices.data(), vertices.size()},
                                    {indices.data(), indices.size()}, mesh)
                              : crossing_status;
        RigidBodyId body{};
        crossing_status = crossing_status
                              ? crossing_world.add_rigid_body(
                                    {.motion = MotionType::static_body,
                                     .mesh = mesh},
                                    body)
                              : crossing_status;
        const std::array<Vec3, 2> crossing_line{{
            {-0.2F, 0.0F, 0.0F}, {0.2F, 0.0F, 0.0F},
        }};
        RopeId crossing_rope{};
        crossing_status = crossing_status
                              ? crossing_world.add_rope(
                                    {.centerline = {crossing_line.data(),
                                                    crossing_line.size()},
                                     .node_spacing = 0.05F,
                                     .radius = 0.025F},
                                    crossing_rope)
                              : crossing_status;
        if (!require(
                crossing_status.code == StatusCode::invalid_argument,
                "Metal accepted a rope rest centerline through a rigid collider"))
            return 1;
    }
    {
        World segment_world;
        Status segment_status = World::create(
            {.rigid_body_capacity = 1U,
             .triangle_mesh_capacity = 1U,
             .rope_capacity = 1U},
            segment_world);
        const std::array<Vec3, 3> vertices{{
            {-0.001F, -0.008F, -0.001F},
            {0.001F, -0.008F, -0.001F},
            {0.0F, -0.008F, 0.001F},
        }};
        const std::array<std::uint32_t, 3> indices{{0U, 2U, 1U}};
        TriangleMeshId mesh{};
        segment_status = segment_status
                             ? segment_world.add_triangle_mesh(
                                   {vertices.data(), vertices.size()},
                                   {indices.data(), indices.size()}, mesh)
                             : segment_status;
        RigidBodyId body{};
        segment_status = segment_status
                             ? segment_world.add_rigid_body(
                                   {.motion = MotionType::static_body,
                                    .mesh = mesh},
                                   body)
                             : segment_status;
        const std::array<Vec3, 2> line{{
            {-0.01F, 0.0F, 0.0F}, {0.01F, 0.0F, 0.0F},
        }};
        RopeId rope{};
        segment_status = segment_status
                             ? segment_world.add_rope(
                                   {.centerline = {line.data(), line.size()},
                                    .node_spacing = 0.02F,
                                    .radius = 0.01F,
                                    .velocity_damping = 0.0F,
                                    .maximum_substep_timestep = 1.0F / 60.0F,
                                    .solver_iterations = 2U,
                                    .self_collision = false},
                                   rope)
                             : segment_status;
        if (!require(segment_status.ok(),
                     segment_status.message
                         ? segment_status.message
                         : "Rope segment-contact setup failed"))
            return 1;
        if (!step(segment_world, 1U, {}, 1U)) return 1;
        RopeDeviceView view{};
        segment_status = segment_world.rope_view(rope, view);
        bool segment_force = false;
        float mean_height = 0.0F;
        if (segment_status.ok()) {
            for (std::uint64_t node = 0U; node < view.positions.size; ++node) {
                mean_height += contents(view.positions)[node].y;
                segment_force |=
                    distance(contents(view.contact_forces)[node], {}) >
                    1.0e-4F;
            }
            mean_height /= static_cast<float>(view.positions.size);
        }
        if (!require(segment_status.ok() && segment_force &&
                         mean_height > 1.0e-4F,
                     "Rope capsule segment contact was not solved in-loop"))
            return 1;
    }
    {
        // Both rope nodes finish clear of the small body. Contact is possible
        // only if rigid-attached endpoints retain their pre-integration
        // positions for the swept-node test.
        World sweep_world;
        Status sweep_status = World::create(
            {.rigid_body_capacity = 3U,
             .triangle_mesh_capacity = 3U,
             .rope_capacity = 1U},
            sweep_world);
        const std::array<Vec3, 4> anchor_vertices{{
            {-0.02F, -0.02F, -0.02F}, {0.02F, -0.02F, -0.02F},
            {0.0F, 0.02F, -0.02F}, {0.0F, 0.0F, 0.02F},
        }};
        const std::array<std::uint32_t, 12> tetra_indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        TriangleMeshId anchor_mesh{};
        sweep_status = sweep_status
                           ? sweep_world.add_triangle_mesh(
                                 {anchor_vertices.data(),
                                  anchor_vertices.size()},
                                 {tetra_indices.data(), tetra_indices.size()},
                                 anchor_mesh)
                           : sweep_status;
        RigidBodyId anchor{};
        sweep_status = sweep_status
                           ? sweep_world.add_rigid_body(
                                 {.motion = MotionType::kinematic,
                                  .mesh = anchor_mesh,
                                  .initial_state = {
                                      .position = {0.0F, 10.0F, 0.0F}}},
                                 anchor)
                           : sweep_status;
        TriangleMeshId last_anchor_mesh{};
        sweep_status = sweep_status
                           ? sweep_world.add_triangle_mesh(
                                 {anchor_vertices.data(),
                                  anchor_vertices.size()},
                                 {tetra_indices.data(), tetra_indices.size()},
                                 last_anchor_mesh)
                           : sweep_status;
        RigidBodyId last_anchor{};
        sweep_status = sweep_status
                           ? sweep_world.add_rigid_body(
                                 {.motion = MotionType::kinematic,
                                  .mesh = last_anchor_mesh,
                                  .initial_state = {
                                      .position = {0.0F, 11.0F, 0.0F}}},
                                 last_anchor)
                           : sweep_status;
        TriangleMeshId target_mesh{};
        sweep_status = sweep_status
                           ? sweep_world.add_triangle_mesh(
                                 {anchor_vertices.data(),
                                  anchor_vertices.size()},
                                 {tetra_indices.data(), tetra_indices.size()},
                                 target_mesh)
                           : sweep_status;
        RigidBodyId target{};
        sweep_status = sweep_status
                           ? sweep_world.add_rigid_body(
                                 {.motion = MotionType::dynamic,
                                  .mesh = target_mesh,
                                  .mass = 0.1F},
                                 target)
                           : sweep_status;
        const std::array<Vec3, 2> line{{
            {-0.2F, 0.0F, 0.0F}, {-0.1F, 0.0F, 0.0F},
        }};
        RopeId rope{};
        sweep_status = sweep_status
                           ? sweep_world.add_rope(
                                 {.centerline = {line.data(), line.size()},
                                  .node_spacing = 0.1F,
                                  .radius = 0.05F,
                                  .mass = 0.1F,
                                  .velocity_damping = 0.0F,
                                  .maximum_substep_timestep = 1.0F / 60.0F,
                                  .solver_iterations = 2U,
                                  .self_collision = false,
                                  .first = {
                                      .body = anchor,
                                      .local_anchor = {-0.2F, -10.0F, 0.0F},
                                      .enabled = true},
                                  .last = {
                                      .body = last_anchor,
                                      .local_anchor = {-0.1F, -11.0F, 0.0F},
                                      .enabled = true}},
                                 rope)
                           : sweep_status;
        sweep_status = sweep_status
                           ? sweep_world.set_kinematic_target(
                                 anchor,
                                 {.position = {0.3F, 10.0F, 0.0F}})
                           : sweep_status;
        sweep_status = sweep_status
                           ? sweep_world.set_kinematic_target(
                                 last_anchor,
                                 {.position = {0.3F, 11.0F, 0.0F}})
                           : sweep_status;
        sweep_status = sweep_status
                           ? sweep_world.step(
                                 {.timestep = 1.0F / 60.0F,
                                  .substeps = 1U,
                                  .gravity = {}})
                           : sweep_status;
        RigidBodyState target_state{};
        RopeDeviceView sweep_view{};
        sweep_status = sweep_status
                           ? sweep_world.read_rigid_body_state(target,
                                                               target_state)
                           : sweep_status;
        sweep_status = sweep_status
                           ? sweep_world.rope_view(rope, sweep_view)
                           : sweep_status;
        bool swept_force = false;
        if (sweep_status.ok())
            for (std::uint64_t node = 0U;
                 node < sweep_view.contact_forces.size; ++node)
                swept_force |=
                    distance(contents(sweep_view.contact_forces)[node], {}) >
                    1.0e-4F;
        if (!require(sweep_status.ok() && swept_force &&
                         distance(target_state.position, {}) > 1.0e-5F,
                     "Rigid rope anchor sweep missed a thin dynamic body"))
            return 1;
    }
    {
        // A taut rope must release a unilateral floor normal in the same
        // nonlinear solve when its kinematic endpoint lifts.  Retaining that
        // normal pins the first free nodes and leaves a large axial error.
        World release_world;
        Status release_status = World::create(
            {.rigid_body_capacity = 2U,
             .triangle_mesh_capacity = 2U,
             .rope_capacity = 1U},
            release_world);
        const std::array<Vec3, 4> floor_vertices{{
            {-2.0F, 0.0F, -2.0F}, {2.0F, 0.0F, -2.0F},
            {2.0F, 0.0F, 2.0F}, {-2.0F, 0.0F, 2.0F},
        }};
        const std::array<std::uint32_t, 6> floor_indices{{
            0U, 2U, 1U, 0U, 3U, 2U,
        }};
        TriangleMeshId floor_mesh{};
        release_status = release_status
                             ? release_world.add_triangle_mesh(
                                   {floor_vertices.data(),
                                    floor_vertices.size()},
                                   {floor_indices.data(), floor_indices.size()},
                                   floor_mesh)
                             : release_status;
        RigidBodyId floor{};
        release_status = release_status
                             ? release_world.add_rigid_body(
                                   {.motion = MotionType::static_body,
                                    .mesh = floor_mesh},
                                   floor)
                             : release_status;
        const std::array<Vec3, 4> anchor_vertices{{
            {-0.05F, -0.05F, -0.05F}, {0.05F, -0.05F, -0.05F},
            {0.0F, 0.05F, -0.05F}, {0.0F, 0.0F, 0.05F},
        }};
        const std::array<std::uint32_t, 12> anchor_indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        TriangleMeshId anchor_mesh{};
        release_status = release_status
                             ? release_world.add_triangle_mesh(
                                   {anchor_vertices.data(),
                                    anchor_vertices.size()},
                                   {anchor_indices.data(),
                                    anchor_indices.size()},
                                   anchor_mesh)
                             : release_status;
        RigidBodyId anchor{};
        release_status = release_status
                             ? release_world.add_rigid_body(
                                   {.motion = MotionType::kinematic,
                                    .mesh = anchor_mesh,
                                    .initial_state = {
                                        .position = {10.0F, 0.0F, 0.0F}}},
                                   anchor)
                             : release_status;
        const std::array<Vec3, 2> line{{
            {0.0F, 0.026F, 0.0F}, {0.4F, 0.026F, 0.0F},
        }};
        RopeId rope{};
        release_status = release_status
                             ? release_world.add_rope(
                                   {.centerline = {line.data(), line.size()},
                                    .node_spacing = 0.05F,
                                    .radius = 0.025F,
                                    .mass = 0.1F,
                                    .velocity_damping = 0.0F,
                                    .maximum_substep_timestep = 1.0F / 60.0F,
                                    .solver_iterations = 2U,
                                    .self_collision = false,
                                    .first = {
                                        .body = anchor,
                                        .local_anchor = {-10.0F, 0.026F, 0.0F},
                                        .enabled = true}},
                                   rope)
                             : release_status;
        if (!require(release_status.ok(),
                     release_status.message
                         ? release_status.message
                         : "Rope contact-release setup failed"))
            return 1;
        if (!step(release_world, 8U, {0.0F, -9.81F, 0.0F}, 1U)) return 1;
        release_status = release_world.set_kinematic_target(
            anchor, {.position = {10.0F, 0.25F, 0.0F}});
        release_status = release_status
                             ? release_world.step(
                                   {.timestep = 1.0F / 60.0F,
                                    .substeps = 1U,
                                    .gravity = {0.0F, -9.81F, 0.0F}})
                             : release_status;
        RopeDeviceView release_view{};
        release_status = release_status
                             ? release_world.rope_view(rope, release_view)
                             : release_status;
        float release_strain = 0.0F;
        if (release_status.ok()) {
            const Vec3 *release_positions = contents(release_view.positions);
            const float *release_rest = contents(release_view.rest_lengths);
            for (std::uint64_t edge = 0U;
                 edge < release_view.rest_lengths.size; ++edge)
                release_strain = std::max(
                    release_strain,
                    std::abs(distance(release_positions[edge],
                                      release_positions[edge + 1U]) /
                                 release_rest[edge] -
                             1.0F));
            if (!require(release_positions[1].y > 0.035F &&
                             release_strain < 0.03F,
                         "Rope floor contact did not release under tension"))
                return 1;
        }
        if (!require(release_status.ok(),
                     release_status.message
                         ? release_status.message
                         : "Rope contact-release step failed"))
            return 1;
    }
    WorldOptions options{};
    options.rigid_body_capacity = 1U;
    options.triangle_mesh_capacity = 1U;
    World world;
    Status status = World::create(options, world);
    if (!require(status.ok(), "Rope world creation failed")) return 1;
    const std::array<Vec3, 4> mesh_vertices{{
        {-0.05F, -0.05F, -0.05F}, {0.05F, -0.05F, -0.05F},
        {0.0F, 0.05F, -0.05F}, {0.0F, 0.0F, 0.05F},
    }};
    const std::array<std::uint32_t, 12> mesh_indices{{
        0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
    }};
    TriangleMeshId mesh{};
    status = world.add_triangle_mesh({mesh_vertices.data(), mesh_vertices.size()},
                                     {mesh_indices.data(), mesh_indices.size()},
                                     mesh);
    if (!require(status.ok(), "Rope anchor mesh creation failed")) return 1;
    RigidBodyId anchor{};
    status = world.add_rigid_body(
        {.motion = MotionType::static_body,
         .mesh = mesh,
         .initial_state = {.position = {10.0F, 2.0F, 0.0F}}},
        anchor);
    if (!require(status.ok(), "Rope anchor body creation failed")) return 1;
    const std::array<Vec3, 2> line{{{0.0F, 2.0F, 0.0F},
                                    {0.0F, 0.0F, 0.0F}}};
    RopeId rope{};
    status = world.add_rope(
        {.centerline = {line.data(), line.size()},
         .node_spacing = 0.1F,
         .radius = 0.025F,
         .mass = 0.2F,
         .first = {.body = anchor,
                   .local_anchor = {-10.0F, 0.0F, 0.0F},
                   .enabled = true}},
        rope);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Rope accepted spacing larger than its diameter"))
        return 1;
    status = world.add_rope(
        {.centerline = {line.data(), line.size()},
         .node_spacing = 0.05F,
         .radius = 0.025F,
         .mass = 0.2F,
         .first = {.body = anchor,
                   .local_anchor = {-9.8F, 0.0F, 0.0F},
                   .enabled = true}},
        rope);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Rope accepted a displaced rigid attachment"))
        return 1;
    status = world.add_rope(
        {.centerline = {line.data(), line.size()},
         .node_spacing = 0.05F,
         .radius = 0.025F,
         .mass = 0.2F,
         .solver_iterations = 33U,
         .first = {.body = anchor,
                   .local_anchor = {-10.0F, 0.0F, 0.0F},
                   .enabled = true}},
        rope);
    if (!require(status.ok(), status.message ? status.message
                                             : "Rope creation failed"))
        return 1;
    if (!step(world, 60U, {0.0F, -9.81F, 0.0F}, 2U)) return 1;
    RopeDeviceView view{};
    status = world.rope_view(rope, view);
    const Vec3 *positions = status ? contents(view.positions) : nullptr;
    const Vec3 *forces = status ? contents(view.constraint_forces) : nullptr;
    const float *rest_lengths = status ? contents(view.rest_lengths) : nullptr;
    float maximum_strain = 0.0F;
    bool finite_forces = true;
    if (status.ok()) {
        for (std::uint64_t edge = 0; edge < view.rest_lengths.size; ++edge) {
            maximum_strain = std::max(
                maximum_strain,
                std::abs(distance(positions[edge], positions[edge + 1U]) /
                             rest_lengths[edge] -
                         1.0F));
        }
        for (std::uint64_t node = 0; node < view.positions.size; ++node)
            finite_forces &= finite(forces[node]);
    }
    if (!require(status.ok() && distance(positions[0], line[0]) < 1.0e-4F &&
                     maximum_strain < 0.02F && finite_forces,
                 "Rope attachment or direct stretch solve is invalid"))
        return 1;
    return 0;
}

int smoke_test() {
    {
        World auto_domain_world;
        Status domain_status = World::create({}, auto_domain_world);
        SmokeId auto_smoke{};
        domain_status = domain_status ? auto_domain_world.add_smoke(
            {.capacity = 2U,
             .emitter_center = {1.0F, 2.0F, 3.0F},
             .wind = {-2.0F, 0.0F, 4.0F},
             .particles_per_second = 1.0F,
             .lifetime = 2.0F,
             .grid_resolution = 16U,
             .grid_vertical_resolution = 8U,
             .grid_pressure_iterations = 4U,
             .grid_edge_length = 0.001F},
            auto_smoke) : domain_status;
        if (!require(domain_status.code == StatusCode::invalid_argument,
                     "Smoke accepted a CUDA-invalid grid cell spacing"))
            return 1;
        domain_status = auto_domain_world.add_smoke(
            {.capacity = 2U,
             .emitter_center = {1.0F, 2.0F, 3.0F},
             .wind = {-2.0F, 0.0F, 4.0F},
             .particles_per_second = 1.0F,
             .lifetime = 2.0F,
             .grid_resolution = 16U,
             .grid_vertical_resolution = 8U,
             .grid_pressure_iterations = 4U},
            auto_smoke);
        SmokeDeviceView auto_view{};
        domain_status = domain_status ? auto_domain_world.smoke_view(
                                            auto_smoke, auto_view)
                                      : domain_status;
        if (!require(domain_status.ok() &&
                         std::abs(auto_view.grid_spacing - 0.6875F) < 1.0e-6F &&
                         std::abs(auto_view.grid_minimum.x + 6.5F) < 1.0e-6F &&
                         std::abs(auto_view.grid_minimum.y + 0.75F) < 1.0e-6F &&
                         std::abs(auto_view.grid_minimum.z - 1.5F) < 1.0e-6F,
                     "Automatic smoke grid domain differs from CUDA"))
            return 1;
    }
    {
        // CUDA emits after advection: a particle created by this frame starts
        // at age zero and has not yet advanced along its initial velocity.
        World emission_world;
        Status emission_status = World::create({}, emission_world);
        SmokeId emitted_smoke{};
        emission_status = emission_status
                              ? emission_world.add_smoke(
                                    {.capacity = 1U,
                                     .emitter_center = {2.0F, 0.0F, 0.0F},
                                     .initial_velocity = {3.0F, 0.0F, 0.0F},
                                     .wind = {},
                                     .particles_per_second = 10.0F,
                                     .lifetime = 2.0F,
                                     .buoyancy = 0.0F,
                                     .response = 0.0F},
                                    emitted_smoke)
                              : emission_status;
        emission_status = emission_status
                              ? emission_world.step(
                                    {.timestep = 0.1F,
                                     .substeps = 1U,
                                     .gravity = {},
                                     .collect_kernel_timings = true})
                              : emission_status;
        SmokeDeviceView emitted_view{};
        WorldStepTimings emission_timings{};
        emission_status = emission_status
                              ? emission_world.smoke_view(
                                    emitted_smoke, emitted_view)
                              : emission_status;
        emission_status = emission_status
                              ? emission_world.collect_step_timings(
                                    emission_timings)
                              : emission_status;
        if (!require(
                emission_status.ok() && emitted_view.particle_count == 1U &&
                    std::abs(contents(emitted_view.positions)[0].x - 2.0F) <
                        1.0e-6F &&
                    contents(emitted_view.ages)[0] == 0.0F &&
                    emission_timings.smoke_emission.launch_count == 1U,
                "Smoke emission ran before advection or was not timed"))
            return 1;
    }
    {
        // CUDA quantizes every quadratic B-spline contribution to 24 fixed
        // fractional bits before its 64-bit cell accumulation.  This case
        // spans many Metal radix blocks and checks the complete field against
        // that CPU oracle, including cells receiving hundreds of values.
        constexpr std::uint32_t capacity = 257U;
        constexpr std::uint32_t resolution = 16U;
        constexpr std::uint32_t vertical = 8U;
        constexpr Vec3 minimum{-2.0F, -1.0F, -2.0F};
        constexpr float spacing = 0.25F;
        constexpr float rest_density = 12.0F;
        constexpr float scale = 16777216.0F;
        World splat_world;
        Status splat_status = World::create({}, splat_world);
        SmokeId splat_smoke{};
        splat_status = splat_status
                           ? splat_world.add_smoke(
                                 {.capacity = capacity,
                                  .emitter_center = {},
                                  .emitter_half_extents = {0.2F, 0.2F},
                                  .initial_velocity = {},
                                  .wind = {},
                                  .particles_per_second =
                                      static_cast<float>(capacity),
                                  .lifetime = 10.0F,
                                  .buoyancy = 0.0F,
                                  .response = 0.0F,
                                  .rest_number_density = rest_density,
                                  .grid_resolution = resolution,
                                  .grid_vertical_resolution = vertical,
                                  .grid_pressure_iterations = 4U,
                                  .grid_minimum = minimum,
                                  .grid_edge_length = 4.0F},
                                 splat_smoke)
                           : splat_status;
        splat_status = splat_status
                           ? splat_world.step({.timestep = 1.0F,
                                               .substeps = 1U,
                                               .gravity = {}})
                           : splat_status;
        SmokeDeviceView before_splat{};
        splat_status = splat_status
                           ? splat_world.smoke_view(splat_smoke, before_splat)
                           : splat_status;
        std::vector<Vec3> deposited_positions{};
        if (splat_status.ok() && before_splat.particle_count == capacity)
            deposited_positions.assign(
                contents(before_splat.positions),
                contents(before_splat.positions) + capacity);
        else if (splat_status.ok())
            splat_status = {StatusCode::internal_error, 0,
                            "Smoke splat oracle did not fill its ring"};

        const std::uint32_t cell_count =
            resolution * vertical * resolution;
        std::vector<std::uint64_t> fixed_density(cell_count, 0U);
        const auto axis_weight = [](float value, int offset) {
            if (offset < 0)
                return 0.5F * (0.5F - value) * (0.5F - value);
            if (offset == 0) return 0.75F - value * value;
            return 0.5F * (0.5F + value) * (0.5F + value);
        };
        for (const Vec3 position : deposited_positions) {
            const std::array<float, 3> g{{
                (position.x - minimum.x) / spacing - 0.5F,
                (position.y - minimum.y) / spacing - 0.5F,
                (position.z - minimum.z) / spacing - 0.5F,
            }};
            const std::array<int, 3> center{{
                static_cast<int>(std::floor(g[0] + 0.5F)),
                static_cast<int>(std::floor(g[1] + 0.5F)),
                static_cast<int>(std::floor(g[2] + 0.5F)),
            }};
            const std::array<float, 3> delta{{
                g[0] - static_cast<float>(center[0]),
                g[1] - static_cast<float>(center[1]),
                g[2] - static_cast<float>(center[2]),
            }};
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int x = center[0] + dx;
                        const int y = center[1] + dy;
                        const int z = center[2] + dz;
                        if (x < 0 || y < 0 || z < 0 ||
                            x >= static_cast<int>(resolution) ||
                            y >= static_cast<int>(vertical) ||
                            z >= static_cast<int>(resolution))
                            continue;
                        const float weight =
                            axis_weight(delta[0], dx) *
                            axis_weight(delta[1], dy) *
                            axis_weight(delta[2], dz) / rest_density;
                        const std::uint32_t cell =
                            static_cast<std::uint32_t>(x) + resolution *
                                (static_cast<std::uint32_t>(y) + vertical *
                                     static_cast<std::uint32_t>(z));
                        fixed_density[cell] += static_cast<std::uint64_t>(
                            weight * scale + 0.5F);
                    }
        }
        std::vector<float> expected_density(cell_count, 0.0F);
        for (std::uint32_t cell = 0U; cell < cell_count; ++cell)
            expected_density[cell] =
                static_cast<float>(fixed_density[cell]) / scale;

        splat_status = splat_status
                           ? splat_world.step({.timestep = 0.001F,
                                               .substeps = 1U,
                                               .gravity = {}})
                           : splat_status;
        SmokeDeviceView after_splat{};
        splat_status = splat_status
                           ? splat_world.smoke_view(splat_smoke, after_splat)
                           : splat_status;
        bool cuda_quantized_density =
            splat_status.ok() &&
            after_splat.grid_density.size == expected_density.size() &&
            after_splat.grid_temperature.size == expected_density.size();
        std::uint64_t maximum_fixed_difference = 0U;
        if (cuda_quantized_density) {
            const float *actual = contents(after_splat.grid_density);
            const float *temperature = contents(after_splat.grid_temperature);
            for (std::uint32_t cell = 0U; cell < cell_count; ++cell) {
                const std::uint64_t actual_fixed =
                    static_cast<std::uint64_t>(actual[cell] * scale + 0.5F);
                const std::uint64_t difference =
                    actual_fixed > fixed_density[cell]
                        ? actual_fixed - fixed_density[cell]
                        : fixed_density[cell] - actual_fixed;
                maximum_fixed_difference =
                    std::max(maximum_fixed_difference, difference);
                cuda_quantized_density &= std::isfinite(actual[cell]) &&
                    temperature[cell] == 0.0F;
            }
        }
        // CPU and GPU contraction may move an individual contribution by one
        // fixed-point unit.  The accumulated field remains within four units
        // here (2.4e-7), far inside the cross-backend floating tolerance.
        cuda_quantized_density &= maximum_fixed_difference <= 4U;
        if (!cuda_quantized_density && splat_status.ok())
            std::cerr << "smoke splat maximum fixed-point difference: "
                      << maximum_fixed_difference << '\n';
        if (!require(cuda_quantized_density,
                     "Smoke splat differs from CUDA's fixed-point oracle"))
            return 1;
    }
    {
        // CUDA retains the fractional emission count on the host, evaluates
        // each frame with a double intermediate, and launches only when the
        // floored request is non-zero. The emitted slot and serial sequences
        // also drive the exact deterministic Y/Z offsets.
        World schedule_world;
        Status schedule_status = World::create({}, schedule_world);
        constexpr std::uint32_t capacity = 64U;
        constexpr float rate = 13.7F;
        constexpr Vec3 center{2.0F, -3.0F, 4.0F};
        constexpr Vec2 extents{0.4F, 0.7F};
        SmokeId scheduled_smoke{};
        schedule_status = schedule_status
                              ? schedule_world.add_smoke(
                                    {.capacity = capacity,
                                     .emitter_center = center,
                                     .emitter_half_extents = extents,
                                     .initial_velocity = {},
                                     .wind = {},
                                     .particles_per_second = rate,
                                     .lifetime = 1000.0F,
                                     .buoyancy = 0.0F,
                                     .response = 0.0F},
                                    scheduled_smoke)
                              : schedule_status;
        const std::array<float, 12> timesteps{{
            0.01F, 0.02F, 0.031F, 0.07F, 0.11F, 0.003F,
            0.2F, 0.05F, 0.017F, 0.13F, 0.09F, 0.04F,
        }};
        float fraction = 0.0F;
        std::uint32_t expected_count = 0U;
        for (const float timestep : timesteps) {
            const std::uint32_t old_count = expected_count;
            const double exact = static_cast<double>(fraction) +
                static_cast<double>(rate) * timestep;
            const std::uint32_t requested =
                static_cast<std::uint32_t>(std::min(
                    std::floor(exact), static_cast<double>(capacity)));
            fraction = static_cast<float>(exact - std::floor(exact));
            expected_count = std::min(capacity,
                                      expected_count + requested);
            schedule_status = schedule_status
                                  ? schedule_world.step(
                                        {.timestep = timestep,
                                         .substeps = 3U,
                                         .gravity = {},
                                         .collect_kernel_timings = true})
                                  : schedule_status;
            SmokeDeviceView schedule_view{};
            WorldStepTimings schedule_timings{};
            schedule_status = schedule_status
                                  ? schedule_world.smoke_view(
                                        scheduled_smoke, schedule_view)
                                  : schedule_status;
            schedule_status = schedule_status
                                  ? schedule_world.collect_step_timings(
                                        schedule_timings)
                                  : schedule_status;
            if (!require(
                    schedule_status.ok() &&
                        schedule_view.particle_count == expected_count &&
                        schedule_timings.smoke_emission.launch_count ==
                            (requested == 0U ? 0U : 1U),
                    "Smoke emission schedule differs from CUDA"))
                return 1;
            for (std::uint32_t serial = old_count;
                 serial < expected_count; ++serial) {
                const Vec3 position = contents(schedule_view.positions)[serial];
                const float expected_y = center.y + extents.x *
                    (2.0F * smoke_random(serial ^ 0x132aef41U) - 1.0F);
                const float expected_z = center.z + extents.y *
                    (2.0F * smoke_random(serial ^ 0xa385c9d3U) - 1.0F);
                if (!require(
                        std::abs(position.x - center.x) < 1.0e-6F &&
                            std::abs(position.y - expected_y) < 1.0e-6F &&
                            std::abs(position.z - expected_z) < 1.0e-6F,
                        "Smoke emitter hash sequence differs from CUDA"))
                    return 1;
            }
        }
    }
    {
        struct SmokeClockSample {
            Status status{};
            Vec3 position{};
            Vec3 velocity{};
            float age{};
            WorldStepTimings timings{};
        };
        const auto sample = [](std::uint32_t substeps) {
            SmokeClockSample result{};
            World clock_world;
            result.status = World::create({}, clock_world);
            SmokeId clock_smoke{};
            result.status = result.status
                                ? clock_world.add_smoke(
                                      {.capacity = 2U,
                                       .emitter_half_extents =
                                           {0.001F, 0.001F},
                                       .initial_velocity = {},
                                       .wind = {5.0F, 0.0F, 0.0F},
                                       .particles_per_second = 1.0F,
                                       .lifetime = 10.0F,
                                       .buoyancy = 0.0F,
                                       .response = 2.0F,
                                       .maximum_speed = 100.0F},
                                      clock_smoke)
                                : result.status;
            result.status = result.status
                                ? clock_world.step(
                                      {.timestep = 1.0F,
                                       .substeps = substeps,
                                       .gravity = {}})
                                : result.status;
            result.status = result.status
                                ? clock_world.step(
                                      {.timestep = 0.2F,
                                       .substeps = substeps,
                                       .gravity = {},
                                       .collect_kernel_timings = true})
                                : result.status;
            SmokeDeviceView clock_view{};
            result.status = result.status
                                ? clock_world.smoke_view(clock_smoke,
                                                         clock_view)
                                : result.status;
            result.status = result.status
                                ? clock_world.collect_step_timings(
                                      result.timings)
                                : result.status;
            if (result.status.ok() && clock_view.particle_count == 1U) {
                result.position = contents(clock_view.positions)[0];
                result.velocity = contents(clock_view.velocities)[0];
                result.age = contents(clock_view.ages)[0];
            } else if (result.status.ok()) {
                result.status = {StatusCode::internal_error, 0,
                                 "Smoke clock emitted an unexpected count"};
            }
            return result;
        };
        const SmokeClockSample single = sample(1U);
        const SmokeClockSample split = sample(4U);
        if (!require(
                single.status.ok() && split.status.ok() &&
                    std::memcmp(&single.position, &split.position,
                                sizeof(Vec3)) == 0 &&
                    std::memcmp(&single.velocity, &split.velocity,
                                sizeof(Vec3)) == 0 &&
                    std::memcmp(&single.age, &split.age, sizeof(float)) == 0 &&
                    single.timings.smoke_advection.launch_count == 1U &&
                    split.timings.smoke_advection.launch_count == 1U &&
                    single.timings.smoke_grid.launch_count == 0U &&
                    split.timings.smoke_grid.launch_count == 0U,
                "Smoke advection clock depends on world substeps"))
            return 1;

        struct SmokeGridClockSample {
            Status status{};
            std::vector<Vec3> velocity{};
            std::vector<float> pressure{};
            WorldStepTimings timings{};
        };
        const auto sample_grid = [](std::uint32_t substeps) {
            SmokeGridClockSample result{};
            World clock_world;
            result.status = World::create({}, clock_world);
            SmokeId clock_smoke{};
            result.status = result.status
                                ? clock_world.add_smoke(
                                      {.capacity = 2U,
                                       .emitter_half_extents =
                                           {0.001F, 0.001F},
                                       .initial_velocity = {},
                                       .wind = {1.0F, 0.0F, 0.0F},
                                       .particles_per_second = 1.0F,
                                       .lifetime = 10.0F,
                                       .buoyancy = 0.0F,
                                       .response = 0.0F,
                                       .grid_resolution = 16U,
                                       .grid_vertical_resolution = 8U,
                                       .grid_pressure_iterations = 8U,
                                       .grid_minimum =
                                           {-2.0F, -1.0F, -2.0F},
                                       .grid_edge_length = 4.0F},
                                      clock_smoke)
                                : result.status;
            result.status = result.status
                                ? clock_world.step(
                                      {.timestep = 1.0F,
                                       .substeps = substeps,
                                       .gravity = {}})
                                : result.status;
            result.status = result.status
                                ? clock_world.step(
                                      {.timestep = 0.05F,
                                       .substeps = substeps,
                                       .gravity = {},
                                       .collect_kernel_timings = true})
                                : result.status;
            SmokeDeviceView clock_view{};
            result.status = result.status
                                ? clock_world.smoke_view(clock_smoke,
                                                         clock_view)
                                : result.status;
            result.status = result.status
                                ? clock_world.collect_step_timings(
                                      result.timings)
                                : result.status;
            if (result.status.ok()) {
                result.velocity.assign(
                    contents(clock_view.grid_velocity),
                    contents(clock_view.grid_velocity) +
                        clock_view.grid_velocity.size);
                result.pressure.assign(
                    contents(clock_view.grid_pressure),
                    contents(clock_view.grid_pressure) +
                        clock_view.grid_pressure.size);
            }
            return result;
        };
        const SmokeGridClockSample single_grid = sample_grid(1U);
        const SmokeGridClockSample split_grid = sample_grid(4U);
        if (!require(
                single_grid.status.ok() && split_grid.status.ok() &&
                    single_grid.velocity.size() ==
                        split_grid.velocity.size() &&
                    single_grid.pressure.size() ==
                        split_grid.pressure.size() &&
                    std::memcmp(single_grid.velocity.data(),
                                split_grid.velocity.data(),
                                single_grid.velocity.size() *
                                    sizeof(Vec3)) == 0 &&
                    std::memcmp(single_grid.pressure.data(),
                                split_grid.pressure.data(),
                                single_grid.pressure.size() *
                                    sizeof(float)) == 0 &&
                    single_grid.timings.smoke_grid.launch_count == 2U &&
                    split_grid.timings.smoke_grid.launch_count == 2U &&
                    single_grid.timings.smoke_advection.launch_count == 1U &&
                    split_grid.timings.smoke_advection.launch_count == 1U,
                "Smoke grid clock depends on world substeps"))
            return 1;
    }
    {
        struct ParticleSmokeCapture {
            Status status{};
            std::vector<Vec3> positions{};
            std::vector<Vec3> velocities{};
            std::vector<float> ages{};
            std::vector<float> densities{};
            std::vector<float> pressures{};
            std::vector<Vec3> vorticities{};
            WorldStepTimings timings{};
        };
        const auto capture = [] {
            ParticleSmokeCapture result{};
            World replay_world;
            result.status = World::create({}, replay_world);
            SmokeId replay_smoke{};
            result.status = result.status
                                ? replay_world.add_smoke(
                                      {.capacity = 257U,
                                       .emitter_half_extents =
                                           {0.04F, 0.04F},
                                       .initial_velocity =
                                           {0.2F, -0.1F, 0.05F},
                                       .wind = {0.4F, 0.2F, -0.1F},
                                       .particles_per_second = 1.0F,
                                       .lifetime = 1000.0F,
                                       .particle_radius = 0.02F,
                                       .buoyancy = 0.1F,
                                       .response = 0.7F,
                                       .rest_number_density = 12.0F,
                                       .pressure_stiffness = 8.0F,
                                       .viscosity = 0.2F,
                                       .vorticity_confinement = 0.3F,
                                       .maximum_speed = 20.0F},
                                      replay_smoke)
                                : result.status;
            result.status = result.status
                                ? replay_world.step(
                                      {.timestep = 257.0F,
                                       .substeps = 1U,
                                       .gravity = {}})
                                : result.status;
            result.status = result.status
                                ? replay_world.step(
                                      {.timestep = 0.01F,
                                       .substeps = 1U,
                                       .gravity = {0.0F, -9.81F, 0.0F}})
                                : result.status;
            result.status = result.status
                                ? replay_world.step(
                                      {.timestep = 0.01F,
                                       .substeps = 1U,
                                       .gravity = {0.0F, -9.81F, 0.0F},
                                       .collect_kernel_timings = true})
                                : result.status;
            SmokeDeviceView replay_view{};
            result.status = result.status
                                ? replay_world.smoke_view(replay_smoke,
                                                          replay_view)
                                : result.status;
            result.status = result.status
                                ? replay_world.collect_step_timings(
                                      result.timings)
                                : result.status;
            if (!result.status || replay_view.particle_count != 257U) {
                if (result.status)
                    result.status = {StatusCode::internal_error, 0,
                                     "Parallel smoke replay count differs"};
                return result;
            }
            const auto copy = [](auto span, auto &destination) {
                destination.assign(contents(span),
                                   contents(span) + span.size);
            };
            copy(replay_view.positions, result.positions);
            copy(replay_view.velocities, result.velocities);
            copy(replay_view.ages, result.ages);
            copy(replay_view.number_densities, result.densities);
            copy(replay_view.pressures, result.pressures);
            copy(replay_view.vorticities, result.vorticities);
            return result;
        };
        const ParticleSmokeCapture baseline = capture();
        bool byte_identical = baseline.status.ok() &&
                              baseline.timings.available &&
                              baseline.timings.smoke_advection.launch_count ==
                                  1U;
        const auto same_bytes = [](const auto &first, const auto &second) {
            using Value = typename std::decay_t<decltype(first)>::value_type;
            return first.size() == second.size() &&
                   std::memcmp(first.data(), second.data(),
                               first.size() * sizeof(Value)) == 0;
        };
        for (std::uint32_t run = 1U; run < 10U && byte_identical; ++run) {
            const ParticleSmokeCapture replay = capture();
            byte_identical = replay.status.ok() &&
                same_bytes(baseline.positions, replay.positions) &&
                same_bytes(baseline.velocities, replay.velocities) &&
                same_bytes(baseline.ages, replay.ages) &&
                same_bytes(baseline.densities, replay.densities) &&
                same_bytes(baseline.pressures, replay.pressures) &&
                same_bytes(baseline.vorticities, replay.vorticities);
        }
        if (!require(
                byte_identical,
                "Parallel particle-smoke replay was not byte deterministic"))
            return 1;
    }
    World world;
    Status status = World::create({}, world);
    if (!require(status.ok(), "Smoke world creation failed")) return 1;
    SmokeId smoke{};
    SmokeOptions viscous_options{
        .capacity = 64U,
        .initial_velocity = {0.2F, 0.0F, 0.0F},
        .wind = {0.5F, 0.0F, 0.0F},
        .particles_per_second = 120.0F,
        .lifetime = 2.0F,
        .grid_resolution = 16U,
        .grid_vertical_resolution = 8U,
        .grid_pressure_iterations = 8U,
        .grid_kinematic_viscosity = 0.05F,
        .grid_les_coefficient = 0.3F,
        .grid_minimum = {-1.0F, -1.0F, -1.0F},
        .grid_edge_length = 2.0F};
    status = world.add_smoke(viscous_options, smoke);
    if (!require(status.ok(), "Smoke creation failed")) return 1;
    if (!step(world, 10U, {})) return 1;
    SmokeDeviceView view{};
    status = world.smoke_view(smoke, view);
    bool occupied = false;
    bool finite_grid = true;
    float maximum_projected_divergence = 0.0F;
    if (status.ok()) {
        for (std::uint64_t cell = 0; cell < view.grid_density.size; ++cell) {
            occupied |= contents(view.grid_density)[cell] > 0.0F;
            finite_grid &= std::isfinite(contents(view.grid_pressure)[cell]) &&
                           std::isfinite(contents(view.grid_divergence)[cell]);
            maximum_projected_divergence = std::max(
                maximum_projected_divergence,
                std::abs(contents(view.grid_divergence)[cell]));
        }
    }
    const bool valid_grid =
        status.ok() && view.particle_count > 0U && occupied && finite_grid &&
        finite(contents(view.positions)[0]) &&
        std::isfinite(view.grid_pressure_relative_residual) &&
        view.grid_pressure_relative_residual <= 1.0F &&
        maximum_projected_divergence < 1.0F;
    if (!valid_grid && status.ok())
        std::cerr << "smoke grid count=" << view.particle_count
                  << " occupied=" << occupied
                  << " finite=" << finite_grid
                  << " residual=" << view.grid_pressure_relative_residual
                  << " divergence=" << maximum_projected_divergence << '\n';
    if (!require(valid_grid, "Smoke particles or pressure grid are invalid"))
        return 1;

    World replay_world;
    status = World::create({}, replay_world);
    SmokeId replay_smoke{};
    status = status ? replay_world.add_smoke(viscous_options, replay_smoke)
                    : status;
    if (status.ok() && !step(replay_world, 10U, {})) return 1;
    SmokeDeviceView replay_view{};
    status = status ? replay_world.smoke_view(replay_smoke, replay_view)
                    : status;
    const bool replay_equal = status.ok() &&
        replay_view.particle_count == view.particle_count &&
        replay_view.positions.size == view.positions.size &&
        replay_view.grid_pressure.size == view.grid_pressure.size &&
        replay_view.grid_velocity.size == view.grid_velocity.size &&
        replay_view.grid_divergence.size == view.grid_divergence.size &&
        std::memcmp(contents(replay_view.positions), contents(view.positions),
                    view.positions.size * sizeof(Vec3)) == 0 &&
        std::memcmp(contents(replay_view.grid_pressure),
                    contents(view.grid_pressure),
                    view.grid_pressure.size * sizeof(float)) == 0 &&
        std::memcmp(contents(replay_view.grid_velocity),
                    contents(view.grid_velocity),
                    view.grid_velocity.size * sizeof(Vec3)) == 0 &&
        std::memcmp(contents(replay_view.grid_divergence),
                    contents(view.grid_divergence),
                    view.grid_divergence.size * sizeof(float)) == 0 &&
        std::memcmp(&replay_view.grid_pressure_relative_residual,
                    &view.grid_pressure_relative_residual,
                    sizeof(float)) == 0;
    if (!require(replay_equal,
                 "Smoke multigrid replay was not byte deterministic"))
        return 1;

    World low_pressure_world;
    status = World::create({}, low_pressure_world);
    SmokeOptions low_pressure_options = viscous_options;
    low_pressure_options.grid_pressure_iterations = 4U;
    SmokeId low_pressure_smoke{};
    status = status ? low_pressure_world.add_smoke(
                          low_pressure_options, low_pressure_smoke)
                    : status;
    if (status.ok() && !step(low_pressure_world, 10U, {})) return 1;
    SmokeDeviceView low_pressure_view{};
    status = status ? low_pressure_world.smoke_view(
                          low_pressure_smoke, low_pressure_view)
                    : status;
    float low_pressure_divergence = 0.0F;
    if (status.ok()) {
        const float *divergence =
            contents(low_pressure_view.grid_divergence);
        for (std::uint64_t cell = 0U;
             cell < low_pressure_view.grid_divergence.size; ++cell)
            low_pressure_divergence =
                std::max(low_pressure_divergence,
                         std::abs(divergence[cell]));
    }
    const bool pressure_improved =
        status.ok() &&
        view.grid_pressure_relative_residual <=
            low_pressure_view.grid_pressure_relative_residual + 1.0e-5F &&
        maximum_projected_divergence <=
            low_pressure_divergence + 1.0e-5F;
    if (!pressure_improved && status.ok())
        std::cerr << "pressure residual low="
                  << low_pressure_view.grid_pressure_relative_residual
                  << " high=" << view.grid_pressure_relative_residual
                  << " divergence low=" << low_pressure_divergence
                  << " high=" << maximum_projected_divergence << '\n';
    if (!require(
            pressure_improved,
            "Additional smoke pressure iterations did not improve projection"))
        return 1;

    World inviscid_world;
    status = World::create({}, inviscid_world);
    SmokeOptions inviscid_options = viscous_options;
    inviscid_options.grid_kinematic_viscosity = 0.0F;
    inviscid_options.grid_les_coefficient = 0.0F;
    SmokeId inviscid_smoke{};
    status = status ? inviscid_world.add_smoke(
                          inviscid_options, inviscid_smoke)
                    : status;
    if (status.ok() && !step(inviscid_world, 10U, {})) return 1;
    SmokeDeviceView inviscid_view{};
    status = status ? inviscid_world.smoke_view(
                          inviscid_smoke, inviscid_view)
                    : status;
    double velocity_difference = 0.0;
    if (status.ok() &&
        inviscid_view.grid_velocity.size == view.grid_velocity.size) {
        const Vec3 *viscous_velocity = contents(view.grid_velocity);
        const Vec3 *inviscid_velocity = contents(inviscid_view.grid_velocity);
        for (std::uint64_t cell = 0U; cell < view.grid_velocity.size; ++cell) {
            velocity_difference +=
                std::abs(viscous_velocity[cell].x -
                         inviscid_velocity[cell].x) +
                std::abs(viscous_velocity[cell].y -
                         inviscid_velocity[cell].y) +
                std::abs(viscous_velocity[cell].z -
                         inviscid_velocity[cell].z);
        }
    }
    if (!require(status.ok() && velocity_difference > 1.0e-4,
                 "Smoke grid viscosity and LES options had no effect"))
        return 1;

    World gravity_world;
    status = World::create({}, gravity_world);
    SmokeId gravity_smoke{};
    status = status ? gravity_world.add_smoke(
                          {.capacity = 4U,
                           .emitter_half_extents = {0.001F, 0.001F},
                           .initial_velocity = {},
                           .wind = {},
                           .particles_per_second = 60.0F,
                           .lifetime = 2.0F,
                           .buoyancy = 1.0F,
                           .response = 0.0F},
                          gravity_smoke)
                    : status;
    if (!require(status.ok(), "Smoke gravity-direction setup failed"))
        return 1;
    if (!step(gravity_world, 2U, {9.81F, 0.0F, 0.0F}, 1U)) return 1;
    SmokeDeviceView gravity_view{};
    status = gravity_world.smoke_view(gravity_smoke, gravity_view);
    const Vec3 gravity_velocity =
        status.ok() && gravity_view.particle_count != 0U
            ? contents(gravity_view.velocities)[0]
            : Vec3{};
    if (!require(status.ok() && gravity_view.particle_count == 2U &&
                     gravity_velocity.x < -0.015F &&
                     std::abs(gravity_velocity.y) < 1.0e-6F &&
                     contents(gravity_view.ages)[1] == 0.0F,
                 "Smoke buoyancy did not oppose the configured gravity"))
        return 1;

    {
        World sweep_world;
        status = World::create({}, sweep_world);
        SmokeId swept_smoke{};
        status = status ? sweep_world.add_smoke(
                              {.capacity = 1U,
                               .emitter_center = {0.0F, 0.2F, 0.0F},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {0.0F, -5.0F, 0.0F},
                               .wind = {0.0F, -5.0F, 0.0F},
                               .particles_per_second = 5.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.1F,
                               .buoyancy = 0.0F,
                               .response = 0.0F,
                               .maximum_speed = 100.0F},
                              swept_smoke)
                        : status;
        const std::array<Vec3, 4> plane_vertices{{
            {-2.0F, 0.0F, -2.0F}, {2.0F, 0.0F, -2.0F},
            {2.0F, 0.0F, 2.0F}, {-2.0F, 0.0F, 2.0F},
        }};
        const std::array<std::uint32_t, 6> plane_indices{{
            0U, 2U, 1U, 0U, 3U, 2U,
        }};
        TriangleMeshId plane_mesh{};
        status = status ? sweep_world.add_triangle_mesh(
                              {plane_vertices.data(), plane_vertices.size()},
                              {plane_indices.data(), plane_indices.size()},
                              plane_mesh)
                        : status;
        RigidBodyId plane{};
        status = status ? sweep_world.add_rigid_body(
                              {.motion = MotionType::dynamic,
                               .mesh = plane_mesh,
                               .mass = 1.0F,
                               .linear_damping = 0.0F,
                               .angular_damping = 0.0F,
                               .collision_margin = 0.0F},
                              plane)
                        : status;
        SmokeRigidCouplingId coupling{};
        status = status ? sweep_world.add_smoke_rigid_coupling(
                              {.smoke = swept_smoke,
                               .body = plane,
                               .air_density = 125.0F,
                               .drag_coefficient = 0.0F,
                               .contact_distance = 0.1F,
                               .tracer_contact = true},
                              coupling)
                        : status;
        status = status ? sweep_world.step(
                              {.timestep = 0.2F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        status = status ? sweep_world.step(
                              {.timestep = 0.1F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        SmokeDeviceView swept_view{};
        RigidBodyState plane_state{};
        status = status ? sweep_world.smoke_view(swept_smoke, swept_view)
                        : status;
        status = status ? sweep_world.read_rigid_body_state(
                              plane, plane_state)
                        : status;
        const Vec3 tracer_position =
            status.ok() && swept_view.particle_count != 0U
                ? contents(swept_view.positions)[0]
                : Vec3{};
        const Vec3 tracer_velocity =
            status.ok() && swept_view.particle_count != 0U
                ? contents(swept_view.velocities)[0]
                : Vec3{};
        if (!(status.ok() && swept_view.particle_count == 1U &&
              tracer_position.y >= 0.099F &&
              tracer_velocity.y > -1.0e-4F &&
              std::abs(plane_state.linear_velocity.y + 5.0F) < 0.05F))
            std::cerr << "smoke sweep status="
                      << static_cast<int>(status.code)
                      << " count=" << swept_view.particle_count
                      << " y=" << tracer_position.y
                      << " vy=" << tracer_velocity.y
                      << " body_vy=" << plane_state.linear_velocity.y
                      << '\n';
        if (!require(
                status.ok() && swept_view.particle_count == 1U &&
                    tracer_position.y >= 0.099F &&
                    tracer_velocity.y > -1.0e-4F &&
                    std::abs(plane_state.linear_velocity.y + 5.0F) < 0.05F,
                "Swept smoke-rigid contact or volume-scaled reaction is invalid"))
            return 1;
    }
    {
        World moving_world;
        status = World::create({}, moving_world);
        SmokeId smoke{};
        status = status ? moving_world.add_smoke(
                              {.capacity = 1U,
                               .emitter_center = {0.0F, 0.2F, 0.0F},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {},
                               .wind = {},
                               .particles_per_second = 5.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.1F,
                               .buoyancy = 0.0F,
                               .response = 0.0F,
                               .maximum_speed = 100.0F},
                              smoke)
                        : status;
        const std::array<Vec3, 4> plane_vertices{{
            {-2.0F, 0.0F, -2.0F}, {2.0F, 0.0F, -2.0F},
            {2.0F, 0.0F, 2.0F}, {-2.0F, 0.0F, 2.0F},
        }};
        const std::array<std::uint32_t, 6> plane_indices{{
            0U, 2U, 1U, 0U, 3U, 2U,
        }};
        TriangleMeshId mesh{};
        status = status ? moving_world.add_triangle_mesh(
                              {plane_vertices.data(), plane_vertices.size()},
                              {plane_indices.data(), plane_indices.size()}, mesh)
                        : status;
        RigidBodyId plane{};
        status = status ? moving_world.add_rigid_body(
                              {.motion = MotionType::kinematic,
                               .mesh = mesh,
                               .initial_state = {
                                   .position = {0.0F, 1.0F, 0.0F}},
                               .collision_margin = 0.0F},
                              plane)
                        : status;
        SmokeRigidCouplingId coupling{};
        status = status ? moving_world.add_smoke_rigid_coupling(
                              {.smoke = smoke,
                               .body = plane,
                               .air_density = 0.0F,
                               .drag_coefficient = 0.0F,
                               .contact_distance = 0.1F,
                               .tracer_contact = true},
                              coupling)
                        : status;
        status = status ? moving_world.step(
                              {.timestep = 0.2F,
                               .substeps = 1U,
                               .gravity = {}})
                        : status;
        status = status ? moving_world.set_kinematic_target(
                              plane,
                              {.position = {0.0F, -1.0F, 0.0F}})
                        : status;
        status = status ? moving_world.step(
                              {.timestep = 0.1F,
                               .substeps = 4U,
                               .gravity = {}})
                        : status;
        SmokeDeviceView view{};
        status = status ? moving_world.smoke_view(smoke, view) : status;
        const Vec3 position = status.ok() && view.particle_count != 0U
                                  ? contents(view.positions)[0]
                                  : Vec3{};
        const Vec3 velocity = status.ok() && view.particle_count != 0U
                                  ? contents(view.velocities)[0]
                                  : Vec3{};
        if (!require(status.ok() && view.particle_count == 1U &&
                         position.y < -1.099F && velocity.y < -19.0F,
                     "Kinematic rigid sweep did not carry a smoke tracer"))
            return 1;
    }

    World obstacle_world;
    status = World::create({}, obstacle_world);
    SmokeOptions obstacle_options = viscous_options;
    obstacle_options.emitter_center = {-0.35F, -0.5F, 0.0F};
    obstacle_options.initial_velocity = {1.0F, 0.0F, 0.0F};
    obstacle_options.wind = {1.0F, 0.0F, 0.0F};
    SmokeId obstacle_smoke{};
    status = status ? obstacle_world.add_smoke(
                          obstacle_options, obstacle_smoke)
                    : status;
    const std::array<Vec3, 8> obstacle_vertices{{
        {-0.25F, -0.25F, -0.25F}, {0.25F, -0.25F, -0.25F},
        {0.25F, -0.25F, 0.25F},   {-0.25F, -0.25F, 0.25F},
        {-0.25F, 0.25F, -0.25F},  {0.25F, 0.25F, -0.25F},
        {0.25F, 0.25F, 0.25F},    {-0.25F, 0.25F, 0.25F},
    }};
    const std::array<std::uint32_t, 36> obstacle_indices{{
        0U, 1U, 2U, 0U, 2U, 3U, 4U, 6U, 5U, 4U, 7U, 6U,
        0U, 4U, 5U, 0U, 5U, 1U, 1U, 5U, 6U, 1U, 6U, 2U,
        2U, 6U, 7U, 2U, 7U, 3U, 3U, 7U, 4U, 3U, 4U, 0U,
    }};
    {
        World cache_world;
        Status cache_status = World::create(
            {.smoke_capacity = 1U,
             .smoke_rigid_coupling_capacity = 1U,
             .rigid_body_capacity = 1U,
             .triangle_mesh_capacity = 1U},
            cache_world);
        SmokeId cache_smoke{};
        cache_status = cache_status
                           ? cache_world.add_smoke(
                                 {.capacity = 1U,
                                  .emitter_center = {},
                                  .initial_velocity = {},
                                  .wind = {},
                                  .particles_per_second = 1.0F,
                                  .lifetime = 10.0F,
                                  .buoyancy = 0.0F,
                                  .response = 0.0F,
                                  .grid_resolution = 16U,
                                  .grid_vertical_resolution = 8U,
                                  .grid_pressure_iterations = 4U,
                                  .grid_minimum = {-1.0F, -1.0F, -1.0F},
                                  .grid_edge_length = 2.0F},
                                 cache_smoke)
                           : cache_status;
        TriangleMeshId cache_mesh{};
        cache_status = cache_status
                           ? cache_world.add_triangle_mesh(
                                 {obstacle_vertices.data(),
                                  obstacle_vertices.size()},
                                 {obstacle_indices.data(),
                                  obstacle_indices.size()},
                                 cache_mesh)
                           : cache_status;
        RigidBodyId cache_body{};
        cache_status = cache_status
                           ? cache_world.add_rigid_body(
                                 {.motion = MotionType::static_body,
                                  .mesh = cache_mesh,
                                  .initial_state = {
                                      .position = {-0.45F, -0.5F, 0.0F}}},
                                 cache_body)
                           : cache_status;
        SmokeRigidCouplingId cache_coupling{};
        cache_status = cache_status
                           ? cache_world.add_smoke_rigid_coupling(
                                 {.smoke = cache_smoke,
                                  .body = cache_body,
                                  .air_density = 0.0F,
                                  .tracer_contact = false},
                                 cache_coupling)
                           : cache_status;
        const auto capture_static = [&](WorldStepTimings &timings,
                                        std::vector<std::uint32_t> &solid) {
            cache_status = cache_status
                               ? cache_world.step(
                                     {.timestep = 1.0F / 60.0F,
                                      .substeps = 1U,
                                      .gravity = {},
                                      .collect_kernel_timings = true})
                               : cache_status;
            SmokeDeviceView view{};
            cache_status = cache_status
                               ? cache_world.smoke_view(cache_smoke, view)
                               : cache_status;
            cache_status = cache_status
                               ? cache_world.collect_step_timings(timings)
                               : cache_status;
            if (cache_status)
                solid.assign(contents(view.grid_solid),
                             contents(view.grid_solid) +
                                 view.grid_solid.size);
        };
        WorldStepTimings first_timings{}, second_timings{}, moved_timings{},
            recached_timings{};
        std::vector<std::uint32_t> first_solid, second_solid, moved_solid,
            recached_solid;
        capture_static(first_timings, first_solid);
        capture_static(second_timings, second_solid);
        cache_status = cache_status
                           ? cache_world.set_rigid_body_state(
                                 cache_body,
                                 {.position = {0.45F, -0.5F, 0.0F}})
                           : cache_status;
        capture_static(moved_timings, moved_solid);
        capture_static(recached_timings, recached_solid);
        if (!require(
                cache_status.ok() && first_timings.available &&
                    second_timings.available && moved_timings.available &&
                    recached_timings.available &&
                    first_timings.smoke_grid.launch_count ==
                        second_timings.smoke_grid.launch_count + 1U &&
                    moved_timings.smoke_grid.launch_count ==
                        recached_timings.smoke_grid.launch_count + 1U &&
                    first_solid == second_solid &&
                    moved_solid == recached_solid &&
                    first_solid != moved_solid,
                "Static smoke metadata was not cached or invalidated"))
            return 1;
    }
    TriangleMeshId obstacle_mesh{};
    status = status ? obstacle_world.add_triangle_mesh(
                          {obstacle_vertices.data(), obstacle_vertices.size()},
                          {obstacle_indices.data(), obstacle_indices.size()},
                          obstacle_mesh)
                    : status;
    RigidBodyId obstacle_body{};
    status = status ? obstacle_world.add_rigid_body(
                          {.mesh = obstacle_mesh,
                           .initial_state = {
                               .position = {0.0F, -0.5F, 0.0F},
                               .linear_velocity = {0.25F, 0.0F, 0.0F}},
                           .linear_damping = 0.0F},
                          obstacle_body)
                    : status;
    SmokeRigidCouplingId zero_density_coupling{};
    status = status ? obstacle_world.add_smoke_rigid_coupling(
                          {.smoke = obstacle_smoke,
                           .body = obstacle_body,
                           .air_density = 0.0F,
                           .enabled = false},
                          zero_density_coupling)
                    : status;
    status = status ? obstacle_world.remove_smoke_rigid_coupling(
                          zero_density_coupling)
                    : status;
    SmokeRigidCouplingId obstacle_coupling{};
    status = status ? obstacle_world.add_smoke_rigid_coupling(
                          {.smoke = obstacle_smoke,
                           .body = obstacle_body,
                           .air_density = 30.0F,
                           .drag_coefficient = 4.0F,
                           .tracer_contact = false},
                          obstacle_coupling)
                    : status;
    if (!require(status.ok(), status.message ? status.message
                                              : "Grid obstacle setup failed"))
        return 1;
    SmokeRigidCouplingId duplicate_coupling{};
    status = obstacle_world.add_smoke_rigid_coupling(
        {.smoke = obstacle_smoke, .body = obstacle_body},
        duplicate_coupling);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Duplicate smoke-rigid coupling was accepted"))
        return 1;
    if (!step(obstacle_world, 2U, {}, 1U)) return 1;
    SmokeDeviceView obstacle_view{};
    status = obstacle_world.smoke_view(obstacle_smoke, obstacle_view);
    RigidBodyState obstacle_state{};
    status = status ? obstacle_world.read_rigid_body_state(
                          obstacle_body, obstacle_state)
                    : status;
    std::uint32_t solid_cells = 0U;
    float solid_x_velocity = 0.0F;
    if (status.ok()) {
        const auto *solid = contents(obstacle_view.grid_solid);
        const auto *velocity = contents(obstacle_view.grid_velocity);
        for (std::uint64_t cell = 0U; cell < obstacle_view.grid_solid.size;
             ++cell) {
            if (solid[cell] == 0U) continue;
            ++solid_cells;
            solid_x_velocity += velocity[cell].x;
        }
    }
    if (!require(status.ok() && solid_cells > 0U &&
                     std::abs(obstacle_state.linear_velocity.x - 0.25F) >
                         1.0e-5F &&
                     solid_x_velocity /
                             static_cast<float>(solid_cells) >
                         0.1F,
                 "Grid smoke did not voxelize or react on a moving rigid body"))
        return 1;
    status = obstacle_world.remove_smoke_rigid_coupling(obstacle_coupling);
    if (!require(status.ok(), "Grid obstacle coupling removal failed"))
        return 1;
    if (!step(obstacle_world, 1U, {}, 1U)) return 1;
    status = obstacle_world.smoke_view(obstacle_smoke, obstacle_view);
    bool stale_solid = false;
    if (status.ok()) {
        const auto *solid = contents(obstacle_view.grid_solid);
        for (std::uint64_t cell = 0U; cell < obstacle_view.grid_solid.size;
             ++cell)
            stale_solid |= solid[cell] != 0U;
    }
    if (!require(status.ok() && !stale_solid,
                 "Removed rigid smoke obstacle left stale grid cells"))
        return 1;
    {
        struct OverlapCapture {
            Status status{};
            std::vector<std::uint32_t> solid{};
            std::vector<Vec3> velocity{};
        };
        const auto capture_overlap = [&] {
            OverlapCapture capture{};
            World world;
            capture.status = World::create(
                {.smoke_rigid_coupling_capacity = 2U,
                 .rigid_body_capacity = 2U,
                 .triangle_mesh_capacity = 1U},
                world);
            SmokeId smoke{};
            capture.status = capture.status
                                 ? world.add_smoke(
                                       {.capacity = 1U,
                                        .emitter_center = {0.8F, 0.0F, 0.0F},
                                        .emitter_half_extents =
                                            {0.001F, 0.001F},
                                        .initial_velocity = {},
                                        .wind = {},
                                        .particles_per_second = 1.0F,
                                        .lifetime = 2.0F,
                                        .buoyancy = 0.0F,
                                        .response = 0.0F,
                                        .grid_resolution = 16U,
                                        .grid_vertical_resolution = 8U,
                                        .grid_pressure_iterations = 4U,
                                        .grid_minimum =
                                            {-1.0F, -0.5F, -1.0F},
                                        .grid_edge_length = 2.0F},
                                       smoke)
                                 : capture.status;
            TriangleMeshId mesh{};
            capture.status = capture.status
                                 ? world.add_triangle_mesh(
                                       {obstacle_vertices.data(),
                                        obstacle_vertices.size()},
                                       {obstacle_indices.data(),
                                        obstacle_indices.size()},
                                       mesh)
                                 : capture.status;
            const std::array<float, 2> speeds{{0.75F, -0.75F}};
            for (const float speed : speeds) {
                RigidBodyId body{};
                capture.status = capture.status
                                     ? world.add_rigid_body(
                                           {.motion = MotionType::kinematic,
                                            .mesh = mesh,
                                            .initial_state = {
                                                .linear_velocity =
                                                    {speed, 0.0F, 0.0F}},
                                            .collision_margin = 0.0F},
                                           body)
                                     : capture.status;
                SmokeRigidCouplingId coupling{};
                capture.status = capture.status
                                     ? world.add_smoke_rigid_coupling(
                                           {.smoke = smoke,
                                            .body = body,
                                            .air_density = 0.0F,
                                            .drag_coefficient = 0.0F,
                                            .tracer_contact = false},
                                           coupling)
                                     : capture.status;
            }
            capture.status = capture.status
                                 ? world.step({.timestep = 1.0F / 60.0F,
                                               .substeps = 1U,
                                               .gravity = {}})
                                 : capture.status;
            SmokeDeviceView view{};
            capture.status = capture.status
                                 ? world.smoke_view(smoke, view)
                                 : capture.status;
            if (capture.status.ok()) {
                capture.solid.assign(contents(view.grid_solid),
                                     contents(view.grid_solid) +
                                         view.grid_solid.size);
                capture.velocity.assign(contents(view.grid_velocity),
                                        contents(view.grid_velocity) +
                                            view.grid_velocity.size);
            }
            return capture;
        };
        const OverlapCapture baseline = capture_overlap();
        bool stable_overlap = baseline.status.ok();
        for (std::uint32_t replay = 0U; replay < 4U; ++replay) {
            const OverlapCapture sample = capture_overlap();
            stable_overlap &= sample.status.ok() &&
                sample.solid.size() == baseline.solid.size() &&
                sample.velocity.size() == baseline.velocity.size() &&
                std::memcmp(sample.solid.data(), baseline.solid.data(),
                            baseline.solid.size() * sizeof(std::uint32_t)) ==
                    0 &&
                std::memcmp(sample.velocity.data(), baseline.velocity.data(),
                            baseline.velocity.size() * sizeof(Vec3)) == 0;
        }
        std::uint32_t overlap_cells = 0U;
        float overlap_velocity = 0.0F;
        if (baseline.status.ok()) {
            for (std::size_t cell = 0U; cell < baseline.solid.size(); ++cell) {
                if (baseline.solid[cell] == 0U) continue;
                ++overlap_cells;
                overlap_velocity += baseline.velocity[cell].x;
            }
        }
        if (!require(
                stable_overlap && overlap_cells != 0U &&
                    overlap_velocity / static_cast<float>(overlap_cells) >
                        0.1F,
                "Overlapping smoke obstacles lack stable triangle tie-breaking"))
            return 1;
    }
    return 0;
}

int coupling_test() {
    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 4> vertices{{
            {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
            {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        const std::array<float, 4> inverse_masses{{0.0F, 0.0F, 0.0F,
                                                   0.0F}};
        ClothId cloth{};
        status = status ? world.add_cloth(
                              {.vertices = {vertices.data(), vertices.size()},
                               .triangle_indices = {indices.data(),
                                                    indices.size()},
                               .inverse_masses = {inverse_masses.data(),
                                                  inverse_masses.size()},
                               .preserve_volume = true},
                              cloth)
                        : status;
        const std::array<FluidParticle, 1> particles{{
            {{0.34F, 0.34F, 0.36F}, {}, 20.0F},
        }};
        FluidId fluid{};
        status = status ? world.add_fluid(
                              {.capacity = 1U,
                               .particle_radius = 0.01F,
                               .support_radius = 0.05F,
                               .repulsion = 0.0F,
                               .velocity_damping = 0.0F},
                              HostSpan<const FluidParticle>{particles.data(),
                                                            particles.size()},
                              fluid)
                        : status;
        FluidClothCouplingId coupling{};
        status = status ? world.add_fluid_cloth_coupling(
                              {.fluid = fluid,
                               .cloth = cloth,
                               .contact_distance = 0.02F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Fluid-cloth setup failed"))
            return 1;
        if (!step(world, 1U, {}, 1U)) return 1;
        FluidDeviceView fluid_view{};
        ClothDeviceView cloth_view{};
        status = world.fluid_view(fluid, fluid_view);
        status = status ? world.cloth_view(cloth, cloth_view) : status;
        const Vec3 point = contents(fluid_view.positions)[0];
        bool has_reaction = false;
        for (std::uint64_t node = 0; node < cloth_view.fluid_contact_forces.size;
             ++node) {
            const Vec3 force = contents(cloth_view.fluid_contact_forces)[node];
            has_reaction |= std::abs(force.x) + std::abs(force.y) +
                                std::abs(force.z) >
                            1.0e-5F;
        }
        if (!require(status.ok() && point.x + point.y + point.z < 0.99F &&
                         has_reaction,
                     "Fluid-cloth containment ignored the triangle surface"))
            return 1;
    }

    {
        World world;
        Status status = World::create(
            {.fluid_capacity = 2U,
             .cloth_capacity = 2U,
             .fluid_cloth_coupling_capacity = 1U},
            world);
        const std::array<Vec3, 4> vertices{{
            {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
            {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        const std::array<float, 4> inverse_masses{};
        ClothOptions cloth_options{
            .vertices = {vertices.data(), vertices.size()},
            .triangle_indices = {indices.data(), indices.size()},
            .inverse_masses = {inverse_masses.data(), inverse_masses.size()},
            .preserve_volume = true};
        ClothId old_cloth{}, new_cloth{};
        status = status ? world.add_cloth(cloth_options, old_cloth) : status;
        status = status ? world.add_cloth(cloth_options, new_cloth) : status;
        FluidId old_fluid{}, new_fluid{};
        status = status ? world.add_fluid(
                              {.capacity = 1U},
                              HostSpan<const FluidParticle>{}, old_fluid)
                        : status;
        const std::array<FluidParticle, 1> particles{{
            {{0.34F, 0.34F, 0.36F}, {}, 20.0F},
        }};
        status = status ? world.add_fluid(
                              {.capacity = 1U,
                               .particle_radius = 0.01F,
                               .support_radius = 0.05F,
                               .repulsion = 0.0F,
                               .velocity_damping = 0.0F},
                              {particles.data(), particles.size()}, new_fluid)
                        : status;
        FluidClothCouplingId coupling{};
        status = status ? world.add_fluid_cloth_coupling(
                              {.fluid = old_fluid, .cloth = old_cloth},
                              coupling)
                        : status;
        status = status ? world.update_fluid_cloth_coupling(
                              coupling,
                              {.fluid = new_fluid, .cloth = new_cloth,
                               .contact_distance = 0.02F})
                        : status;
        status = status ? world.remove_fluid(old_fluid) : status;
        status = status ? world.remove_cloth(old_cloth) : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Fluid-cloth endpoint update failed"))
            return 1;
        if (!step(world, 1U, {}, 1U)) return 1;
        FluidDeviceView view{};
        status = world.fluid_view(new_fluid, view);
        const Vec3 point = status.ok() ? contents(view.positions)[0] : Vec3{};
        if (!require(status.ok() && point.x + point.y + point.z < 0.99F,
                     "Updated fluid-cloth table retained stale endpoints"))
            return 1;
        status = world.remove_fluid(new_fluid);
        if (!require(status.code == StatusCode::invalid_argument,
                     "Updated fluid-cloth endpoint lost its reference guard"))
            return 1;
        status = world.remove_fluid_cloth_coupling(coupling);
        status = status ? world.remove_fluid(new_fluid) : status;
        status = status ? world.remove_cloth(new_cloth) : status;
        if (!require(status.ok(), "Updated fluid-cloth lifecycle failed"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 4> nodes{{
            {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
            {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
        }};
        const auto rest = [&](std::uint32_t a, std::uint32_t b) {
            return distance(nodes[a], nodes[b]);
        };
        const std::array<SoftBodyBond, 6> bonds{{
            {0U, 1U, rest(0U, 1U)}, {0U, 2U, rest(0U, 2U)},
            {0U, 3U, rest(0U, 3U)}, {1U, 2U, rest(1U, 2U)},
            {1U, 3U, rest(1U, 3U)}, {2U, 3U, rest(2U, 3U)},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        const std::array<float, 4> inverse_masses{{1.0e-6F, 1.0e-6F,
                                                   1.0e-6F, 1.0e-6F}};
        std::array<SoftBodySurfaceBinding, 4> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].nodes[0] = index;
            bindings[index].weights[0] = 1.0F;
        }
        SoftBodyId soft{};
        status = status ? world.add_soft_body(
                              {.nodes = {nodes.data(), nodes.size()},
                               .bonds = {bonds.data(), bonds.size()},
                               .inverse_masses = {inverse_masses.data(),
                                                  inverse_masses.size()},
                               .surface_vertices = {nodes.data(), nodes.size()},
                               .surface_triangle_indices = {indices.data(),
                                                            indices.size()},
                               .surface_bindings = {bindings.data(),
                                                    bindings.size()}},
                              soft)
                        : status;
        const std::array<FluidParticle, 1> particles{{
            {{0.32F, 0.32F, 0.32F}, {-1.0F, -1.0F, -1.0F}, 20.0F},
        }};
        FluidId fluid{};
        status = status ? world.add_fluid(
                              {.capacity = 1U,
                               .particle_radius = 0.05F,
                               .support_radius = 0.1F,
                               .repulsion = 0.0F,
                               .velocity_damping = 0.0F},
                              HostSpan<const FluidParticle>{particles.data(),
                                                            particles.size()},
                              fluid)
                        : status;
        FluidSoftBodyCouplingId coupling{};
        status = status ? world.add_fluid_soft_body_coupling(
                              {.fluid = fluid, .soft_body = soft}, coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Fluid-soft setup failed"))
            return 1;
        if (!step(world, 1U, {}, 1U)) return 1;
        FluidDeviceView fluid_view{};
        SoftBodyDeviceView soft_view{};
        status = world.fluid_view(fluid, fluid_view);
        status = status ? world.soft_body_view(soft, soft_view) : status;
        const Vec3 point = contents(fluid_view.positions)[0];
        bool has_reaction = false;
        for (std::uint64_t node = 0; node < soft_view.fluid_contact_forces.size;
             ++node) {
            const Vec3 force = contents(soft_view.fluid_contact_forces)[node];
            has_reaction |= std::abs(force.x) + std::abs(force.y) +
                                std::abs(force.z) >
                            1.0e-5F;
        }
        WorldStatistics diagnostics{};
        status = status ? world.collect_statistics(diagnostics) : status;
        if (!require(status.ok() && point.x + point.y + point.z > 1.03F &&
                         has_reaction &&
                         diagnostics.fluid_soft_body_contact_count > 0U &&
                         diagnostics.maximum_fluid_soft_body_penetration >
                             0.0F,
                     "Fluid-soft contact ignored the closed skinned surface"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 3> cloth_vertices{{
            {-1.0F, -1.0F, 0.0F}, {1.0F, -1.0F, 0.0F},
            {0.0F, 1.0F, 0.0F},
        }};
        const std::array<std::uint32_t, 3> cloth_indices{{0U, 1U, 2U}};
        const std::array<float, 3> cloth_inverse{{0.0F, 0.0F, 0.0F}};
        ClothId cloth{};
        status = status ? world.add_cloth(
                              {.vertices = {cloth_vertices.data(),
                                            cloth_vertices.size()},
                               .triangle_indices = {cloth_indices.data(),
                                                    cloth_indices.size()},
                               .inverse_masses = {cloth_inverse.data(),
                                                  cloth_inverse.size()}},
                              cloth)
                        : status;
        const std::array<Vec3, 4> nodes{{
            {0.0F, 0.0F, 0.02F}, {0.2F, 0.0F, 0.2F},
            {0.0F, 0.2F, 0.2F}, {0.0F, 0.0F, 0.4F},
        }};
        const auto rest = [&](std::uint32_t a, std::uint32_t b) {
            return distance(nodes[a], nodes[b]);
        };
        const std::array<SoftBodyBond, 6> bonds{{
            {0U, 1U, rest(0U, 1U)}, {0U, 2U, rest(0U, 2U)},
            {0U, 3U, rest(0U, 3U)}, {1U, 2U, rest(1U, 2U)},
            {1U, 3U, rest(1U, 3U)}, {2U, 3U, rest(2U, 3U)},
        }};
        const std::array<std::uint32_t, 12> soft_indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        std::array<SoftBodySurfaceBinding, 4> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].nodes[0] = index;
            bindings[index].weights[0] = 1.0F;
        }
        SoftBodyId soft{};
        status = status ? world.add_soft_body(
                              {.nodes = {nodes.data(), nodes.size()},
                               .bonds = {bonds.data(), bonds.size()},
                               .surface_vertices = {nodes.data(), nodes.size()},
                               .surface_triangle_indices = {soft_indices.data(),
                                                            soft_indices.size()},
                               .surface_bindings = {bindings.data(),
                                                    bindings.size()}},
                              soft)
                        : status;
        SoftBodyClothCouplingId coupling{};
        status = status ? world.add_soft_body_cloth_coupling(
                              {.soft_body = soft,
                               .cloth = cloth,
                               .contact_distance = 0.075F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Soft-cloth setup failed"))
            return 1;
        if (!step(world, 1U, {}, 1U)) return 1;
        SoftBodyDeviceView soft_view{};
        status = world.soft_body_view(soft, soft_view);
        if (!require(status.ok() &&
                         contents(soft_view.positions)[0].z > 0.07F,
                     "Soft-cloth contact ignored the cloth triangle interior"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 3> cloth_vertices{{
            {0.0F, -1.0F, -1.0F}, {0.0F, 1.0F, -1.0F},
            {0.0F, 0.0F, 1.0F},
        }};
        const std::array<std::uint32_t, 3> cloth_indices{{0U, 1U, 2U}};
        const std::array<float, 3> cloth_inverse{{0.0F, 0.0F, 0.0F}};
        ClothId cloth{};
        status = status ? world.add_cloth(
                              {.vertices = {cloth_vertices.data(),
                                            cloth_vertices.size()},
                               .triangle_indices = {cloth_indices.data(),
                                                    cloth_indices.size()},
                               .inverse_masses = {cloth_inverse.data(),
                                                  cloth_inverse.size()}},
                              cloth)
                        : status;
        SmokeId smoke{};
        status = status ? world.add_smoke(
                              {.capacity = 4U,
                               .emitter_center = {-0.01F, 0.0F, 0.0F},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {1.0F, 0.0F, 0.0F},
                               .wind = {1.0F, 0.0F, 0.0F},
                               .particles_per_second = 60.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.025F,
                               .response = 0.0F,
                               .grid_resolution = 16U,
                               .grid_vertical_resolution = 16U,
                               .grid_minimum = {-1.0F, -1.0F, -1.0F},
                               .grid_edge_length = 2.0F},
                              smoke)
                        : status;
        SmokeClothCouplingId coupling{};
        status = status ? world.add_smoke_cloth_coupling(
                              {.smoke = smoke,
                               .cloth = cloth,
                               .wind_drag = 0.0F,
                               .contact_distance = 0.05F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Smoke-cloth setup failed"))
            return 1;
        if (!step(world, 2U, {}, 1U)) return 1;
        SmokeDeviceView smoke_view{};
        status = world.smoke_view(smoke, smoke_view);
        bool cloth_voxelized = false;
        if (status.ok())
            for (std::uint64_t cell = 0U;
                 cell < smoke_view.grid_solid.size; ++cell)
                cloth_voxelized |=
                    contents(smoke_view.grid_solid)[cell] != 0U;
        if (!require(status.ok() && smoke_view.particle_count == 2U &&
                         contents(smoke_view.positions)[0].x < -0.049F &&
                         contents(smoke_view.velocities)[0].x < 0.1F &&
                         cloth_voxelized,
                     "Smoke tracer crossed through the cloth triangle"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 3> vertices{{
            {-0.3F, -0.3F, 0.0F}, {0.3F, -0.3F, 0.0F},
            {0.0F, 0.3F, 0.0F},
        }};
        const std::array<std::uint32_t, 3> indices{{0U, 1U, 2U}};
        const std::array<float, 3> inverse{{1.0F, 1.0F, 1.0F}};
        ClothId cloth{};
        status = status ? world.add_cloth(
                              {.vertices = {vertices.data(), vertices.size()},
                               .triangle_indices = {indices.data(),
                                                    indices.size()},
                               .inverse_masses = {inverse.data(),
                                                  inverse.size()},
                               .velocity_damping = 0.0F,
                               .solver_iterations = 1U},
                              cloth)
                        : status;
        SmokeId smoke{};
        status = status ? world.add_smoke(
                              {.capacity = 2U,
                               .emitter_center = {0.0F, 0.0F, 0.1875F},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {1.0F, 0.0F, 0.0F},
                               .wind = {1.0F, 0.0F, 0.0F},
                               .particles_per_second = 60.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.01F,
                               .buoyancy = 0.0F,
                               .response = 0.0F,
                               .grid_resolution = 16U,
                               .grid_vertical_resolution = 16U,
                               .grid_kinematic_viscosity = 1.0F,
                               .grid_les_coefficient = 0.0F,
                               .grid_minimum = {-1.0F, -1.0F, -1.0F},
                               .grid_edge_length = 2.0F},
                              smoke)
                        : status;
        SmokeClothCouplingId coupling{};
        status = status ? world.add_smoke_cloth_coupling(
                              {.smoke = smoke,
                               .cloth = cloth,
                               .wind_drag = 4.0F,
                               .maximum_wind_acceleration = 100.0F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Grid smoke-cloth force setup failed"))
            return 1;
        if (!step(world, 2U, {}, 1U)) return 1;
        ClothDeviceView view{};
        status = world.cloth_view(cloth, view);
        float mean_x_velocity = 0.0F;
        if (status.ok())
            for (std::uint64_t vertex = 0U; vertex < view.velocities.size;
                 ++vertex)
                mean_x_velocity += contents(view.velocities)[vertex].x;
        mean_x_velocity /= static_cast<float>(view.velocities.size);
        if (!require(status.ok() && mean_x_velocity > 1.0e-4F,
                     "Projected smoke grid did not load the cloth surface"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 4> nodes{{
            {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
            {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
        }};
        const auto rest = [&](std::uint32_t a, std::uint32_t b) {
            return distance(nodes[a], nodes[b]);
        };
        const std::array<SoftBodyBond, 6> bonds{{
            {0U, 1U, rest(0U, 1U)}, {0U, 2U, rest(0U, 2U)},
            {0U, 3U, rest(0U, 3U)}, {1U, 2U, rest(1U, 2U)},
            {1U, 3U, rest(1U, 3U)}, {2U, 3U, rest(2U, 3U)},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        const std::array<float, 4> inverse_masses{{1.0e-6F, 1.0e-6F,
                                                   1.0e-6F, 1.0e-6F}};
        std::array<SoftBodySurfaceBinding, 4> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].nodes[0] = index;
            bindings[index].weights[0] = 1.0F;
        }
        SoftBodyId soft{};
        status = status ? world.add_soft_body(
                              {.nodes = {nodes.data(), nodes.size()},
                               .bonds = {bonds.data(), bonds.size()},
                               .inverse_masses = {inverse_masses.data(),
                                                  inverse_masses.size()},
                               .surface_vertices = {nodes.data(), nodes.size()},
                               .surface_triangle_indices = {indices.data(),
                                                            indices.size()},
                               .surface_bindings = {bindings.data(),
                                                    bindings.size()}},
                              soft)
                        : status;
        SmokeId smoke{};
        status = status ? world.add_smoke(
                              {.capacity = 4U,
                               .emitter_center = {0.2F, 0.2F, -0.01F},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {0.0F, 0.0F, 1.0F},
                               .wind = {0.0F, 0.0F, 1.0F},
                               .particles_per_second = 60.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.025F,
                               .response = 0.0F,
                               .grid_resolution = 16U,
                               .grid_vertical_resolution = 16U,
                               .grid_minimum = {-1.0F, -1.0F, -1.0F},
                               .grid_edge_length = 2.0F},
                              smoke)
                        : status;
        SmokeSoftBodyCouplingId coupling{};
        status = status ? world.add_smoke_soft_body_coupling(
                              {.smoke = smoke,
                               .soft_body = soft,
                               .wind_drag = 0.0F,
                               .contact_distance = 0.05F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Smoke-soft setup failed"))
            return 1;
        if (!step(world, 2U, {}, 1U)) return 1;
        SmokeDeviceView smoke_view{};
        status = world.smoke_view(smoke, smoke_view);
        bool soft_voxelized = false;
        if (status.ok())
            for (std::uint64_t cell = 0U;
                 cell < smoke_view.grid_solid.size; ++cell)
                soft_voxelized |=
                    contents(smoke_view.grid_solid)[cell] != 0U;
        if (!require(status.ok() && smoke_view.particle_count == 2U &&
                         contents(smoke_view.positions)[0].z < -0.049F &&
                         contents(smoke_view.velocities)[0].z < 0.1F &&
                         soft_voxelized,
                     "Smoke tracer crossed through the soft-body skin"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 4> nodes{{
            {-0.3F, -0.3F, 0.0F}, {0.3F, -0.3F, 0.0F},
            {0.0F, 0.3F, 0.0F}, {0.0F, 0.0F, -0.4F},
        }};
        const auto rest = [&](std::uint32_t a, std::uint32_t b) {
            return distance(nodes[a], nodes[b]);
        };
        const std::array<SoftBodyBond, 6> bonds{{
            {0U, 1U, rest(0U, 1U)}, {0U, 2U, rest(0U, 2U)},
            {0U, 3U, rest(0U, 3U)}, {1U, 2U, rest(1U, 2U)},
            {1U, 3U, rest(1U, 3U)}, {2U, 3U, rest(2U, 3U)},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 1U, 2U, 0U, 3U, 1U,
            1U, 3U, 2U, 2U, 3U, 0U,
        }};
        const std::array<float, 4> inverse{{1.0F, 1.0F, 1.0F, 1.0F}};
        std::array<SoftBodySurfaceBinding, 4> bindings{};
        for (std::uint32_t index = 0U; index < bindings.size(); ++index) {
            bindings[index].nodes[0] = index;
            bindings[index].weights[0] = 1.0F;
        }
        SoftBodyId soft{};
        status = status ? world.add_soft_body(
                              {.nodes = {nodes.data(), nodes.size()},
                               .bonds = {bonds.data(), bonds.size()},
                               .inverse_masses = {inverse.data(),
                                                  inverse.size()},
                               .surface_vertices = {nodes.data(), nodes.size()},
                               .surface_triangle_indices = {indices.data(),
                                                            indices.size()},
                               .surface_bindings = {bindings.data(),
                                                    bindings.size()},
                               .node_radius = 0.01F,
                               .velocity_damping = 0.0F,
                               .spring_damping = 0.0F,
                               .solver_iterations = 1U},
                              soft)
                        : status;
        SmokeId smoke{};
        status = status ? world.add_smoke(
                              {.capacity = 2U,
                               .emitter_center = {0.0F, 0.0F, 0.1875F},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {1.0F, 0.0F, 0.0F},
                               .wind = {1.0F, 0.0F, 0.0F},
                               .particles_per_second = 60.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.01F,
                               .buoyancy = 0.0F,
                               .response = 0.0F,
                               .grid_resolution = 16U,
                               .grid_vertical_resolution = 16U,
                               .grid_kinematic_viscosity = 1.0F,
                               .grid_les_coefficient = 0.0F,
                               .grid_minimum = {-1.0F, -1.0F, -1.0F},
                               .grid_edge_length = 2.0F},
                              smoke)
                        : status;
        SmokeSoftBodyCouplingId coupling{};
        status = status ? world.add_smoke_soft_body_coupling(
                              {.smoke = smoke,
                               .soft_body = soft,
                               .wind_drag = 4.0F,
                               .maximum_wind_acceleration = 100.0F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Grid smoke-soft force setup failed"))
            return 1;
        if (!step(world, 2U, {}, 1U)) return 1;
        SoftBodyDeviceView view{};
        status = world.soft_body_view(soft, view);
        float mean_x_velocity = 0.0F;
        if (status.ok())
            for (std::uint64_t node = 0U; node < view.velocities.size; ++node)
                mean_x_velocity += contents(view.velocities)[node].x;
        mean_x_velocity /= static_cast<float>(view.velocities.size);
        if (!require(status.ok() && mean_x_velocity > 1.0e-4F,
                     "Projected smoke grid did not load the soft-body skin"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 2> line{{{-0.1F, 0.0F, 0.0F},
                                        {0.1F, 0.0F, 0.0F}}};
        RopeId rope{};
        status = status ? world.add_rope(
                              {.centerline = {line.data(), line.size()},
                               .node_spacing = 0.02F,
                               .radius = 0.02F,
                               .mass = 0.2F,
                               .velocity_damping = 0.0F,
                               .self_collision = false},
                              rope)
                        : status;
        const std::array<FluidParticle, 1> water{{
            {{0.0F, 0.025F, 0.0F}, {0.0F, -1.0F, 0.0F}, 20.0F},
        }};
        FluidId fluid{};
        status = status ? world.add_fluid(
                              {.capacity = 1U,
                               .particle_radius = 0.02F,
                               .support_radius = 0.08F,
                               .repulsion = 0.0F,
                               .viscosity = 0.0F,
                               .velocity_damping = 0.0F},
                              HostSpan<const FluidParticle>{water.data(),
                                                            water.size()},
                              fluid)
                        : status;
        FluidRopeCouplingId coupling{};
        status = status ? world.add_fluid_rope_coupling(
                              {.fluid = fluid, .rope = rope}, coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Fluid-rope setup failed"))
            return 1;
        if (!step(world, 1U, {}, 4U)) return 1;
        FluidDeviceView fluid_view{};
        RopeDeviceView rope_view{};
        status = world.fluid_view(fluid, fluid_view);
        status = status ? world.rope_view(rope, rope_view) : status;
        const std::uint64_t middle = rope_view.positions.size / 2U;
        const Vec3 water_position = contents(fluid_view.positions)[0];
        const Vec3 water_velocity = contents(fluid_view.velocities)[0];
        const Vec3 rope_position = contents(rope_view.positions)[middle];
        const Vec3 rope_velocity = contents(rope_view.velocities)[middle];
        bool reported_force = false;
        for (std::uint64_t node = 0;
             node < rope_view.fluid_contact_forces.size; ++node)
            reported_force |=
                distance(contents(rope_view.fluid_contact_forces)[node], {}) >
                1.0e-4F;
        WorldStatistics diagnostics{};
        status = status ? world.collect_statistics(diagnostics) : status;
        if (!require(status.ok() &&
                         water_position.y - rope_position.y >= 0.039F &&
                         water_velocity.y > -0.1F && rope_velocity.y < -0.001F &&
                         reported_force &&
                         diagnostics.fluid_rope_contact_count > 0U &&
                         diagnostics.maximum_fluid_rope_penetration > 0.0F,
                     "Fluid-rope capsule response is not bidirectional"))
            return 1;
    }

    {
        struct FluidRopeMassSample {
            bool valid{};
            float maximum_rope_speed{};
        };
        const auto sample = [](float rest_particle_volume) {
            World world;
            Status status = World::create({}, world);
            const std::array<Vec3, 2> line{{{-0.1F, 0.0F, 0.0F},
                                            {0.1F, 0.0F, 0.0F}}};
            RopeId rope{};
            status = status ? world.add_rope(
                                  {.centerline = {line.data(), line.size()},
                                   .node_spacing = 0.02F,
                                   .radius = 0.02F,
                                   .mass = 0.2F,
                                   .velocity_damping = 0.0F,
                                   .maximum_substep_timestep = 1.0F / 60.0F,
                                   .maximum_speed = 100.0F,
                                   .self_collision = false},
                                  rope)
                            : status;
            const std::array<FluidParticle, 1> water{{
                {{0.0F, 0.025F, 0.0F}, {0.0F, -1.0F, 0.0F}, 20.0F},
            }};
            FluidId fluid{};
            status = status ? world.add_fluid(
                                  {.capacity = 1U,
                                   .particle_radius = 0.02F,
                                   .support_radius = 0.08F,
                                   .repulsion = 0.0F,
                                   .viscosity = 0.0F,
                                   .velocity_damping = 0.0F,
                                   .rest_particle_volume =
                                       rest_particle_volume},
                                  HostSpan<const FluidParticle>{water.data(),
                                                                water.size()},
                                  fluid)
                            : status;
            FluidRopeCouplingId coupling{};
            status = status ? world.add_fluid_rope_coupling(
                                  {.fluid = fluid,
                                   .rope = rope,
                                   .maximum_rope_acceleration = 10'000.0F},
                                  coupling)
                            : status;
            if (!status.ok() || !step(world, 1U, {}, 1U))
                return FluidRopeMassSample{};
            RopeDeviceView rope_view{};
            status = world.rope_view(rope, rope_view);
            WorldStatistics diagnostics{};
            status = status ? world.collect_statistics(diagnostics) : status;
            float maximum_rope_speed = 0.0F;
            if (status.ok())
                for (std::uint64_t node = 0U;
                     node < rope_view.velocities.size; ++node)
                    maximum_rope_speed = std::max(
                        maximum_rope_speed,
                        std::fabs(contents(rope_view.velocities)[node].y));
            return FluidRopeMassSample{
                status.ok() && diagnostics.fluid_rope_contact_count > 0U,
                maximum_rope_speed};
        };

        const FluidRopeMassSample light = sample(1.0e-6F);
        const FluidRopeMassSample heavy = sample(1.0e-4F);
        if (!require(light.valid && heavy.valid &&
                         light.maximum_rope_speed > 1.0e-5F &&
                         heavy.maximum_rope_speed >
                             20.0F * light.maximum_rope_speed,
                     "Fluid-rope reaction does not scale with CUDA particle mass"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 2> line{{{-0.5F, 0.0F, 0.0F},
                                        {0.5F, 0.0F, 0.0F}}};
        RopeId rope{};
        status = status ? world.add_rope(
                              {.centerline = {line.data(), line.size()},
                               .node_spacing = 0.04F,
                               .radius = 0.02F,
                               .velocity_damping = 0.0F,
                               .self_collision = false},
                              rope)
                        : status;
        SmokeId smoke{};
        status = status ? world.add_smoke(
                              {.capacity = 4U,
                               .emitter_center = {},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {1.0F, 0.0F, 0.0F},
                               .wind = {1.0F, 0.0F, 0.0F},
                               .particles_per_second = 60.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.02F,
                               .response = 0.0F},
                              smoke)
                        : status;
        SmokeRopeCouplingId coupling{};
        status = status ? world.add_smoke_rope_coupling(
                              {.smoke = smoke,
                               .rope = rope,
                               .wind_drag = 10.0F,
                               .maximum_wind_acceleration = 100.0F,
                               .contact_distance = 0.05F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Smoke-rope setup failed"))
            return 1;
        if (!step(world, 2U, {}, 2U)) return 1;
        SmokeDeviceView smoke_view{};
        RopeDeviceView rope_view{};
        status = world.smoke_view(smoke, smoke_view);
        status = status ? world.rope_view(rope, rope_view) : status;
        bool wind_reached_rope = false;
        for (std::uint64_t node = 0; node < rope_view.velocities.size; ++node)
            wind_reached_rope |= contents(rope_view.velocities)[node].x > 1.0e-5F;
        bool smoke_cleared_rope = false;
        for (std::uint32_t particle = 0; particle < smoke_view.particle_count;
             ++particle) {
            const Vec3 point = contents(smoke_view.positions)[particle];
            smoke_cleared_rope |=
                std::sqrt(point.y * point.y + point.z * point.z) >= 0.049F;
        }
        if (!require(status.ok() && wind_reached_rope && smoke_cleared_rope,
                     "Smoke-rope wind or capsule deflection is missing"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 2> line{{{-0.25F, 0.0F, 0.0F},
                                        {0.25F, 0.0F, 0.0F}}};
        RopeId rope{};
        status = status ? world.add_rope(
                              {.centerline = {line.data(), line.size()},
                               .node_spacing = 0.02F,
                               .radius = 0.01F,
                               .velocity_damping = 0.0F,
                               .self_collision = false},
                              rope)
                        : status;
        SmokeId smoke{};
        status = status ? world.add_smoke(
                              {.capacity = 2U,
                               .emitter_center = {},
                               .emitter_half_extents = {0.001F, 0.001F},
                               .initial_velocity = {1.0F, 0.0F, 0.0F},
                               .wind = {1.0F, 0.0F, 0.0F},
                               .particles_per_second = 60.0F,
                               .lifetime = 2.0F,
                               .particle_radius = 0.01F,
                               .buoyancy = 0.0F,
                               .response = 0.0F,
                               .grid_resolution = 16U,
                               .grid_vertical_resolution = 16U,
                               .grid_minimum = {-1.0F, -1.0F, -1.0F},
                               .grid_edge_length = 2.0F},
                              smoke)
                        : status;
        SmokeRopeCouplingId coupling{};
        status = status ? world.add_smoke_rope_coupling(
                              {.smoke = smoke,
                               .rope = rope,
                               .wind_drag = 10.0F,
                               .maximum_wind_acceleration = 100.0F,
                               .contact_distance = 0.02F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Grid smoke-rope setup failed"))
            return 1;
        if (!step(world, 2U, {}, 1U)) return 1;
        RopeDeviceView view{};
        status = world.rope_view(rope, view);
        bool grid_wind_reached_rope = false;
        if (status.ok())
            for (std::uint64_t node = 0U; node < view.velocities.size; ++node)
                grid_wind_reached_rope |=
                    contents(view.velocities)[node].x > 1.0e-5F;
        if (!require(status.ok() && grid_wind_reached_rope,
                     "Projected smoke grid did not drive rope nodes"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 2> line{{{0.0F, -0.5F, 0.0F},
                                        {0.0F, 0.5F, 0.0F}}};
        RopeId rope{};
        status = status ? world.add_rope(
                              {.centerline = {line.data(), line.size()},
                               .node_spacing = 0.02F,
                               .radius = 0.01F,
                               .velocity_damping = 0.0F,
                               .self_collision = false},
                              rope)
                        : status;
        SmokeId smoke{};
        status = status ? world.add_smoke(
                              {.capacity = 16U,
                               .emitter_center = {-0.1F, 0.0F, 0.0F},
                               .emitter_half_extents = {1.0e-8F, 1.0e-8F},
                               .initial_velocity = {100.0F, 0.0F, 0.0F},
                               .wind = {100.0F, 0.0F, 0.0F},
                               .particles_per_second = 480.0F,
                               .lifetime = 1.0F,
                               .particle_radius = 0.01F,
                               .buoyancy = 0.0F,
                               .response = 0.0F,
                               .maximum_speed = 100.0F},
                              smoke)
                        : status;
        SmokeRopeCouplingId coupling{};
        status = status ? world.add_smoke_rope_coupling(
                              {.smoke = smoke,
                               .rope = rope,
                               .wind_drag = 0.0F,
                               .maximum_wind_acceleration = 0.0F,
                               .contact_distance = 0.025F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Swept smoke-rope setup failed"))
            return 1;
        if (!step(world, 2U, {}, 2U)) return 1;
        SmokeDeviceView view{};
        status = world.smoke_view(smoke, view);
        bool stayed_on_entry_side = view.particle_count != 0U;
        for (std::uint32_t particle = 0; particle < view.particle_count;
             ++particle) {
            if (contents(view.ages)[particle] >= view.lifetime) continue;
            stayed_on_entry_side &=
                contents(view.positions)[particle].x <= -0.024F;
        }
        if (!require(status.ok() && stayed_on_entry_side,
                     "Fast smoke tracer tunneled through a rope capsule"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 3> cloth_vertices{{
            {0.0F, 0.0F, 0.0F}, {0.2F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.2F},
        }};
        const std::array<std::uint32_t, 3> cloth_indices{{0U, 1U, 2U}};
        const std::array<float, 3> cloth_inverse{{20.0F, 0.0F, 0.0F}};
        ClothId cloth{};
        status = status ? world.add_cloth(
                              {.vertices = {cloth_vertices.data(),
                                            cloth_vertices.size()},
                               .triangle_indices = {cloth_indices.data(),
                                                    cloth_indices.size()},
                               .inverse_masses = {cloth_inverse.data(),
                                                  cloth_inverse.size()},
                               .solver_iterations = 1U},
                              cloth)
                        : status;
        const std::array<Vec3, 2> line{{cloth_vertices[0],
                                        {0.2F, 0.1F, 0.0F}}};
        RopeId rope{};
        status = status ? world.add_rope(
                              {.centerline = {line.data(), line.size()},
                               .node_spacing = 0.02F,
                               .self_collision = false},
                              rope)
                        : status;
        RopeClothCouplingId coupling{};
        status = status ? world.add_rope_cloth_coupling(
                              {.rope = rope,
                               .cloth = cloth,
                               .first_vertex = 0U},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Rope-cloth setup failed"))
            return 1;
        if (!step(world, 1U, {0.0F, -9.81F, 0.0F}, 2U)) return 1;
        ClothDeviceView cloth_view{};
        RopeDeviceView rope_view{};
        status = world.cloth_view(cloth, cloth_view);
        status = status ? world.rope_view(rope, rope_view) : status;
        const Vec3 cloth_anchor = contents(cloth_view.positions)[0];
        const Vec3 rope_anchor = contents(rope_view.positions)[0];
        const bool reported_force =
            distance(contents(cloth_view.rope_contact_forces)[0], {}) >
            1.0e-4F;
        const bool tension_reached_anchor =
            distance(contents(rope_view.constraint_forces)[0], {}) >
            1.0e-4F;
        if (!require(status.ok() && cloth_anchor.y < -1.0e-5F &&
                         distance(cloth_anchor, rope_anchor) < 1.0e-5F &&
                         reported_force && tension_reached_anchor,
                     "Rope-cloth anchor did not participate in the rope solve"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 4> nodes{{
            {0.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F},
            {0.0F, 2.0F, 0.0F}, {0.0F, 0.0F, 2.0F},
        }};
        const auto rest = [&](std::uint32_t a, std::uint32_t b) {
            return distance(nodes[a], nodes[b]);
        };
        const std::array<SoftBodyBond, 6> bonds{{
            {0U, 1U, rest(0U, 1U)}, {0U, 2U, rest(0U, 2U)},
            {0U, 3U, rest(0U, 3U)}, {1U, 2U, rest(1U, 2U)},
            {1U, 3U, rest(1U, 3U)}, {2U, 3U, rest(2U, 3U)},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        std::array<SoftBodySurfaceBinding, 4> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].nodes[0] = index;
            bindings[index].weights[0] = 1.0F;
        }
        const std::array<float, 4> inverse_masses{{1.0e-6F, 1.0e-6F,
                                                   1.0e-6F, 1.0e-6F}};
        SoftBodyId soft{};
        status = status ? world.add_soft_body(
                              {.nodes = {nodes.data(), nodes.size()},
                               .bonds = {bonds.data(), bonds.size()},
                               .inverse_masses = {inverse_masses.data(),
                                                  inverse_masses.size()},
                               .surface_vertices = {nodes.data(), nodes.size()},
                               .surface_triangle_indices = {indices.data(),
                                                            indices.size()},
                               .surface_bindings = {bindings.data(),
                                                    bindings.size()},
                               .solver_iterations = 2U},
                              soft)
                        : status;
        const std::array<Vec3, 2> line{{{0.3F, 0.3F, -0.02F},
                                        {0.7F, 0.3F, -0.02F}}};
        RopeId rope{};
        status = status ? world.add_rope(
                              {.centerline = {line.data(), line.size()},
                               .node_spacing = 0.04F,
                               .radius = 0.02F,
                               .self_collision = false},
                              rope)
                        : status;
        RopeSoftBodyCouplingId coupling{};
        status = status ? world.add_rope_soft_body_coupling(
                              {.rope = rope,
                               .soft_body = soft,
                               .contact_distance = 0.05F},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Rope-soft setup failed"))
            return 1;
        if (!step(world, 1U, {}, 2U)) return 1;
        SoftBodyDeviceView soft_view{};
        RopeDeviceView rope_view{};
        status = world.soft_body_view(soft, soft_view);
        status = status ? world.rope_view(rope, rope_view) : status;
        bool cleared_face = true;
        bool reported_contact = false;
        bool contact_reached_stretch_solve = false;
        for (std::uint64_t node = 0; node < rope_view.positions.size; ++node) {
            cleared_face &= contents(rope_view.positions)[node].z <= -0.049F;
            reported_contact |=
                distance(contents(rope_view.soft_body_contact_forces)[node],
                         {}) > 1.0e-4F;
            contact_reached_stretch_solve |=
                distance(contents(rope_view.constraint_forces)[node], {}) >
                1.0e-4F;
        }
        WorldStatistics diagnostics{};
        status = status ? world.collect_statistics(diagnostics) : status;
        if (!require(status.ok() && cleared_face && reported_contact &&
                         contact_reached_stretch_solve &&
                         diagnostics.rope_soft_body_contact_count > 0U &&
                         diagnostics.maximum_rope_soft_body_penetration >
                             0.0F,
                     "Rope capsule missed a soft-body face interior"))
            return 1;
    }

    {
        World world;
        Status status = World::create({}, world);
        const std::array<Vec3, 4> nodes{{
            {0.0F, 0.0F, 0.0F}, {0.2F, 0.0F, 0.0F},
            {0.0F, 0.2F, 0.0F}, {0.0F, 0.0F, 0.2F},
        }};
        const auto rest = [&](std::uint32_t a, std::uint32_t b) {
            return distance(nodes[a], nodes[b]);
        };
        const std::array<SoftBodyBond, 6> bonds{{
            {0U, 1U, rest(0U, 1U)}, {0U, 2U, rest(0U, 2U)},
            {0U, 3U, rest(0U, 3U)}, {1U, 2U, rest(1U, 2U)},
            {1U, 3U, rest(1U, 3U)}, {2U, 3U, rest(2U, 3U)},
        }};
        const std::array<std::uint32_t, 12> indices{{
            0U, 2U, 1U, 0U, 1U, 3U, 1U, 2U, 3U, 2U, 0U, 3U,
        }};
        std::array<SoftBodySurfaceBinding, 4> bindings{};
        for (std::uint32_t index = 0; index < bindings.size(); ++index) {
            bindings[index].nodes[0] = index;
            bindings[index].weights[0] = 1.0F;
        }
        const std::array<float, 4> inverse_masses{{1.0e-6F, 1.0e-6F,
                                                   1.0e-6F, 1.0e-6F}};
        SoftBodyId soft{};
        status = status ? world.add_soft_body(
                              {.nodes = {nodes.data(), nodes.size()},
                               .bonds = {bonds.data(), bonds.size()},
                               .inverse_masses = {inverse_masses.data(),
                                                  inverse_masses.size()},
                               .surface_vertices = {nodes.data(), nodes.size()},
                               .surface_triangle_indices = {indices.data(),
                                                            indices.size()},
                               .surface_bindings = {bindings.data(),
                                                    bindings.size()},
                               .velocity_damping = 20.0F,
                               .solver_iterations = 2U},
                              soft)
                        : status;
        const std::array<Vec3, 2> line{{{0.05F, 0.05F, -0.02F},
                                        {0.3F, 0.05F, -0.02F}}};
        RopeId rope{};
        status = status ? world.add_rope(
                              {.centerline = {line.data(), line.size()},
                               .node_spacing = 0.02F,
                               .self_collision = false},
                              rope)
                        : status;
        RopeSoftBodyCouplingId coupling{};
        status = status ? world.add_rope_soft_body_coupling(
                              {.rope = rope,
                               .soft_body = soft,
                               .contact_distance = 0.005F,
                               .attach_first = true},
                              coupling)
                        : status;
        if (!require(status.ok(), status.message ? status.message
                                                 : "Rope-soft anchor setup failed"))
            return 1;
        if (!step(world, 1U, {0.0F, -9.81F, 0.0F}, 2U)) return 1;
        SoftBodyDeviceView soft_view{};
        RopeDeviceView rope_view{};
        status = world.soft_body_view(soft, soft_view);
        status = status ? world.rope_view(rope, rope_view) : status;
        std::uint32_t loaded_nodes = 0U;
        for (std::uint64_t node = 0; node < soft_view.rope_contact_forces.size;
             ++node) {
            const float force = distance(
                contents(soft_view.rope_contact_forces)[node], {});
            if (force > 1.0e-6F)
                ++loaded_nodes;
        }
        const Vec3 translated_anchor{
            contents(soft_view.positions)[0].x + line[0].x,
            contents(soft_view.positions)[0].y + line[0].y,
            contents(soft_view.positions)[0].z + line[0].z};
        if (!require(status.ok() &&
                         distance(contents(rope_view.positions)[0],
                                  translated_anchor) < 1.0e-5F &&
                         loaded_nodes >= 2U,
                     "Rope-soft surface anchor lost its offset or distribution"))
            return 1;
    }
    return 0;
}

int paint_test() {
    World world;
    Status status = World::create(
        {.fluid_capacity = 2U,
         .rigid_body_capacity = 2U,
         .triangle_mesh_capacity = 2U,
         .paint_field_capacity = 2U,
         .paint_rule_capacity = 2U,
         .cloth_capacity = 2U},
        world);
    if (!require(status.ok(), "Paint world creation failed")) return 1;

    const std::array<Vec3, 3> triangle{{
        {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 1.0F},
    }};
    const std::array<std::uint32_t, 3> indices{{0U, 1U, 2U}};
    TriangleMeshId mesh{};
    status = world.add_triangle_mesh(
        {triangle.data(), triangle.size()}, {indices.data(), indices.size()},
        mesh);
    RigidBodyId body{};
    status = status ? world.add_rigid_body(
                          {.motion = MotionType::static_body, .mesh = mesh},
                          body)
                    : status;
    const std::array<Vec2, 3> point_uvs{{
        {0.99F, 0.5F}, {0.99F, 0.5F}, {0.99F, 0.5F},
    }};
    PaintFieldId field{};
    status = status ? world.add_paint_field(
                          {.body = body,
                           .mesh = mesh,
                           .vertex_uvs = {point_uvs.data(), point_uvs.size()},
                           .width = 64U,
                           .height = 64U},
                          field)
                    : status;
    const FluidParticle particle{
        {0.2F, 0.02F, 0.2F}, {0.0F, -1.0F, 0.0F}, 20.0F};
    FluidId fluid{};
    status = status ? world.add_fluid(
                          {.capacity = 1U,
                           .particle_radius = 0.05F,
                           .support_radius = 0.12F,
                           .solver_iterations = 1U},
                          {&particle, 1U}, fluid)
                    : status;
    PaintRuleId rule{};
    status = status ? world.add_paint_rule(
                          {.source = fluid, .target = field}, rule)
                    : status;
    if (!require(status.ok(), "Fluid-rigid paint setup failed")) return 1;
    if (!require(!world.remove_paint_field(field).ok(),
                 "Referenced paint field was removable"))
        return 1;
    status = world.step(
        {.timestep = 1.0F / 60.0F,
         .substeps = 1U,
         .gravity = {},
         .collect_fluid_contacts = true});
    PaintFieldDeviceView view{};
    status = status ? world.paint_field_view(field, view) : status;
    const ContactDeviceView contact_view = world.contacts();
    const std::uint32_t paint_index = 32U * 64U + 63U;
    if (!require(status.ok() && view.width == 64U && view.height == 64U &&
                     contents(view.pixels)[paint_index] == 2U &&
                     contact_view.event_count == 1U &&
                     contact_view.events.size == 1U &&
                     contents(contact_view.events)[0].fluid == fluid &&
                     contents(contact_view.events)[0].rigid_body == body &&
                     contents(contact_view.events)[0].stable_particle_id == 0U &&
                     finite(contents(contact_view.events)[0].position) &&
                     finite(contents(contact_view.events)[0].normal) &&
                     contents(contact_view.events)[0].normal.y > 0.9F &&
                     contents(contact_view.events)[0].normal_impulse > 0.0F,
                 "Fluid contact paint or retained event is invalid"))
        return 1;
    status = world.clear_paint_field(field);
    if (!require(status.ok() && contents(view.pixels)[paint_index] == 0U,
                 "Paint field clear failed"))
        return 1;
    status = world.remove_paint_rule(rule);
    status = status ? world.remove_fluid(fluid) : status;
    std::vector<FluidParticle> concurrent_paint(65U);
    for (std::uint32_t index = 0U; index < concurrent_paint.size(); ++index)
        concurrent_paint[index] = {
            {0.2F, (index & 1U) == 0U ? -0.02F : 0.02F, 0.2F}, {},
            20.0F};
    status = status ? world.add_fluid(
                          {.capacity = 65U,
                           .particle_radius = 0.05F,
                           .support_radius = 0.12F,
                           .solver_iterations = 1U,
                           .repulsion = 0.0F,
                           .viscosity = 0.0F,
                           .velocity_damping = 0.0F},
                          HostSpan<const FluidParticle>{
                              concurrent_paint.data(),
                              concurrent_paint.size()},
                          fluid)
                    : status;
    status = status ? world.add_paint_rule(
                          {.source = fluid, .target = field}, rule)
                    : status;
    status = status ? world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    if (!require(status.ok() && contents(view.pixels)[paint_index] == 3U,
                 "Concurrent fluid paint lost a two-sided atomic bit"))
        return 1;
    status = world.remove_paint_rule(rule);
    status = status ? world.remove_fluid(fluid) : status;
    status = status ? world.remove_paint_field(field) : status;
    status = status ? world.remove_rigid_body(body) : status;
    status = status ? world.remove_triangle_mesh(mesh) : status;
    if (!require(status.ok(), "Fluid-rigid paint teardown failed")) return 1;

    const std::array<Vec3, 3> cloth_vertices{{
        {-0.5F, 0.0F, -0.5F}, {0.5F, 0.0F, -0.5F},
        {-0.5F, 0.0F, 0.5F},
    }};
    const std::array<float, 3> fixed{{0.0F, 0.0F, 0.0F}};
    ClothId cloth{};
    status = world.add_cloth(
        {.vertices = {cloth_vertices.data(), cloth_vertices.size()},
         .triangle_indices = {indices.data(), indices.size()},
         .inverse_masses = {fixed.data(), fixed.size()},
         .thickness = 0.02F,
         .solver_iterations = 1U},
        cloth);
    const std::array<Vec2, 3> cloth_uvs{{
        {0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F},
    }};
    status = status ? world.add_paint_field(
                          {.cloth = cloth,
                           .vertex_uvs = {cloth_uvs.data(), cloth_uvs.size()},
                           .width = 32U,
                           .height = 32U},
                          field)
                    : status;
    const std::array<Vec3, 3> source_vertices{{
        {-0.05F, 0.0F, -0.05F}, {0.05F, 0.0F, -0.05F},
        {0.0F, 0.0F, 0.05F},
    }};
    TriangleMeshId source_mesh{};
    status = status ? world.add_triangle_mesh(
                          {source_vertices.data(), source_vertices.size()},
                          {indices.data(), indices.size()}, source_mesh)
                    : status;
    RigidBodyId source_body{};
    status = status ? world.add_rigid_body(
                          {.motion = MotionType::dynamic,
                           .mesh = source_mesh,
                           .initial_state = {.position = {0.0F, 0.01F, 0.0F}}},
                          source_body)
                    : status;
    status = status ? world.add_paint_rule(
                          {.rigid_source = source_body,
                           .target = field,
                           .brush_radius = 0.2F},
                          rule)
                    : status;
    status = status ? world.step(
                          {.timestep = 1.0F / 60.0F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    status = status ? world.paint_field_view(field, view) : status;
    bool stamped = false;
    if (status.ok())
        for (std::uint64_t index = 0U; index < view.pixels.size; ++index)
            stamped |= contents(view.pixels)[index] != 0U;
    if (!require(status.ok() && stamped,
                 "Rigid contact did not paint the cloth UV field"))
        return 1;
    status = world.remove_paint_rule(rule);
    status = status ? world.remove_paint_field(field) : status;
    status = status ? world.remove_cloth(cloth) : status;
    status = status ? world.remove_rigid_body(source_body) : status;
    status = status ? world.remove_triangle_mesh(source_mesh) : status;
    if (!require(status.ok(), "Rigid-cloth paint teardown failed")) return 1;
    return 0;
}

int debug_test() {
    World disabled;
    Status status = World::create({}, disabled);
    PhysicsDebugFrameView disabled_view{};
    status = status ? disabled.physics_debug_frame(disabled_view) : status;
    if (!require(status.code == StatusCode::not_supported,
                 "Disabled physics debug capture did not report unsupported"))
        return 1;

    World world;
    status = World::create(
        {.fluid_capacity = 1U,
         .rigid_body_capacity = 1U,
         .triangle_mesh_capacity = 1U,
         .physics_debug = {.frame_capacity = 2U, .frame_stride = 1U}},
        world);
    const std::array<Vec3, 3> vertices{{
        {-1.0F, 0.0F, -1.0F}, {1.0F, 0.0F, -1.0F},
        {0.0F, 0.0F, 1.0F},
    }};
    const std::array<std::uint32_t, 3> indices{{0U, 2U, 1U}};
    TriangleMeshId mesh{};
    status = status ? world.add_triangle_mesh(
                          {vertices.data(), vertices.size()},
                          {indices.data(), indices.size()}, mesh)
                    : status;
    RigidBodyId body{};
    status = status ? world.add_rigid_body(
                          {.motion = MotionType::static_body, .mesh = mesh},
                          body)
                    : status;
    const FluidParticle particle{{0.0F, 0.02F, 0.0F}, {}, 20.0F};
    FluidId fluid{};
    status = status ? world.add_fluid(
                          {.capacity = 1U,
                           .particle_radius = 0.05F,
                           .support_radius = 0.12F,
                           .solver_iterations = 1U},
                          {&particle, 1U}, fluid)
                    : status;
    status = status ? world.step(
                          {.timestep = 0.01F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    PhysicsDebugFrameView view{};
    status = status ? world.physics_debug_frame(view) : status;
    if (!require(status.ok() && view.frame_index == 1U &&
                     std::abs(view.timestep - 0.01F) < 1.0e-7F &&
                     view.rigid_bodies.size == 1U &&
                     view.rigid_bodies.data[0].id == body &&
                     view.fluid_particles.size == 1U &&
                     view.fluid_particles.data[0].fluid == fluid &&
                     view.fluid_contacts.size == 1U,
                 "Latest Metal physics debug frame is incomplete"))
        return 1;
    status = world.step(
        {.timestep = 0.02F, .substeps = 1U, .gravity = {}});
    status = status ? world.step(
                          {.timestep = 0.03F,
                           .substeps = 1U,
                           .gravity = {}})
                    : status;
    PhysicsDebugCapture capture{};
    status = status ? world.copy_physics_debug_capture(capture) : status;
    if (!require(status.ok() && capture.frames.size() == 2U &&
                     capture.frames[0].frame_index == 2U &&
                     capture.frames[1].frame_index == 3U &&
                     std::abs(capture.frames[0].timestep - 0.02F) < 1.0e-7F &&
                     std::abs(capture.frames[1].timestep - 0.03F) < 1.0e-7F,
                 "Metal physics debug ring is not chronological"))
        return 1;
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "expected subsystem name\n";
        return 2;
    }
    const std::string_view subsystem = argv[1];
    if (subsystem == "fluid") return fluid_test();
    if (subsystem == "cloth") return cloth_test();
    if (subsystem == "soft-body") return soft_body_test();
    if (subsystem == "rope") return rope_test();
    if (subsystem == "smoke") return smoke_test();
    if (subsystem == "couplings") return coupling_test();
    if (subsystem == "paint") return paint_test();
    if (subsystem == "debug") return debug_test();
    std::cerr << "unknown subsystem: " << subsystem << '\n';
    return 2;
}
