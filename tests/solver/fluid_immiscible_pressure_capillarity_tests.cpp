// SPDX-License-Identifier: MIT
#include "simulation/fluid_dispatch.hpp"
#include "simulation/primitives.hpp"
#include "simulation/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

using namespace parallel_mater;
using namespace parallel_mater::simulation;

namespace {
struct Phase {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    std::vector<Vec3> accelerations;
    std::vector<float> densities;
    std::vector<Vec3> normals;
    std::vector<float> curvatures;
    std::vector<std::uint32_t> flags;
};

float length(Vec3 value) {
    return std::sqrt(value.x * value.x + value.y * value.y +
        value.z * value.z);
}

Vec3 scaled(Vec3 value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 added(Vec3 left, Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 cross(Vec3 left, Vec3 right) {
    return {left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x};
}

void check_close(float left, float right, float relative_tolerance) {
    CHECK(std::fabs(left - right) <= relative_tolerance *
        std::max(1.0e-6F, std::max(std::fabs(left), std::fabs(right))));
}

void check_close(Vec3 left, Vec3 right, float tolerance) {
    CHECK(std::fabs(left.x - right.x) <= tolerance);
    CHECK(std::fabs(left.y - right.y) <= tolerance);
    CHECK(std::fabs(left.z - right.z) <= tolerance);
}

float particle_mass(const FluidDispatchConstants &constants) {
    const float diameter = 2.0F * constants.particle_radius;
    const float volume = constants.rest_particle_volume > 0.0F
        ? constants.rest_particle_volume
        : diameter * diameter * diameter;
    return constants.rest_density * volume;
}

float pressure_gradient(float distance, float support_radius) {
    const float support_squared = support_radius * support_radius;
    const float q_squared = std::max(
        0.0F, 1.0F - distance * distance / support_squared);
    return 9.40008882636F * distance * q_squared * q_squared /
        (support_squared * support_squared * support_radius);
}

FluidDispatchConstants material(float density, float volume) {
    FluidDispatchConstants result{};
    result.particle_radius = 0.01F;
    result.support_radius = 0.5F;
    result.rest_density = density;
    result.rest_particle_volume = volume;
    result.maximum_neighbors = 32U;
    return result;
}

void prepare(Phase &phase, float density, Vec3 normal = {1.0F, 0.0F, 0.0F},
    float curvature = 0.0F,
    std::uint32_t flags = k_fluid_interface_sample) {
    const auto count = phase.positions.size();
    phase.velocities.resize(count);
    phase.accelerations.assign(count, {});
    phase.densities.assign(count, density);
    phase.normals.assign(count, normal);
    phase.curvatures.assign(count, curvature);
    phase.flags.assign(count, flags);
}

void apply(Phase &first, Phase &second,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants, float surface_tension) {
    const auto first_count = static_cast<std::uint32_t>(first.positions.size());
    const auto second_count = static_cast<std::uint32_t>(second.positions.size());
    const auto total = first_count + second_count;
    CHECK(first.velocities.size() == first_count);
    CHECK(second.velocities.size() == second_count);
    CHECK(first.densities.size() == first_count);
    CHECK(second.densities.size() == second_count);
    first.accelerations.assign(first_count, {});
    second.accelerations.assign(second_count, {});

    std::vector<std::uint64_t> keys[2];
    std::vector<std::uint32_t> identities[2];
    for (unsigned slot = 0U; slot < 2U; ++slot) {
        keys[slot].resize(total);
        identities[slot].resize(total);
    }
    const float support = std::max(first_constants.support_radius,
        second_constants.support_radius);
    fluid_pair_cell_keys((total + 127U) / 128U, 128U, {},
        first.positions.data(), &first_count, first_count,
        second.positions.data(), &second_count, second_count,
        1.0F / support, keys[0].data(), identities[0].data());
    std::vector<std::byte> workspace(radix_sort_scratch_bytes(total));
    CHECK(radix_sort_pairs_u64_u32({}, keys[0].data(), keys[1].data(),
        identities[0].data(), identities[1].data(), total,
        workspace.data(), workspace.size()) == PM_SUCCESS);
    fluid_immiscible_pair_pressure_capillarity(
        (total + 127U) / 128U, 128U, {},
        first.positions.data(), first.velocities.data(), &first_count,
        first_count, first.accelerations.data(), first.densities.data(),
        first.normals.data(), first.curvatures.data(), first.flags.data(),
        second.positions.data(), second.velocities.data(), &second_count,
        second_count, second.accelerations.data(), second.densities.data(),
        second.normals.data(), second.curvatures.data(), second.flags.data(),
        keys[1].data(), identities[1].data(), total,
        first_constants, second_constants, surface_tension);
}

void check_conservation(const Phase &first, const Phase &second,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) {
    Vec3 net_force{};
    Vec3 net_torque{};
    float activity = 0.0F;
    const auto accumulate = [&](const Phase &phase, float mass) {
        for (std::size_t index = 0U; index < phase.positions.size(); ++index) {
            const Vec3 force = scaled(phase.accelerations[index], mass);
            net_force = added(net_force, force);
            net_torque = added(net_torque,
                cross(phase.positions[index], force));
            activity += length(force);
        }
    };
    accumulate(first, particle_mass(first_constants));
    accumulate(second, particle_mass(second_constants));
    CHECK(activity >= 1.0e-8F);
    CHECK(length(net_force) <= std::max(1.0e-10F, 5.0e-5F * activity));
    CHECK(length(net_torque) <= std::max(1.0e-10F, 5.0e-5F * activity));
}

Phase reversed_phase(const Phase &source) {
    Phase result = source;
    for (auto &normal : result.normals)
        normal = scaled(normal, -1.0F);
    for (auto &curvature : result.curvatures)
        curvature = -curvature;
    return result;
}

void check_endpoint_reversal(Phase first, Phase second,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants, float surface_tension) {
    apply(first, second, first_constants, second_constants, surface_tension);
    auto reversed_first = reversed_phase(second);
    auto reversed_second = reversed_phase(first);
    apply(reversed_first, reversed_second, second_constants, first_constants,
        surface_tension);
    for (std::size_t index = 0U; index < first.accelerations.size(); ++index)
        check_close(first.accelerations[index],
            reversed_second.accelerations[index], 2.0e-5F);
    for (std::size_t index = 0U; index < second.accelerations.size(); ++index)
        check_close(second.accelerations[index],
            reversed_first.accelerations[index], 2.0e-5F);
}

void test_contact_pressure_conservation_and_reversal() {
    auto first_material = material(800.0F, 0.002F);
    first_material.particle_radius = 0.08F;
    first_material.repulsion = 20.0F;
    first_material.normal_damping = 0.6F;
    auto second_material = material(1'200.0F, 0.001F);
    second_material.particle_radius = 0.07F;
    second_material.repulsion = 70.0F;
    second_material.normal_damping = 1.4F;

    Phase first{{{-0.05F, 0.00F, 0.00F},
                    {0.00F, 0.12F, 0.04F}},
        {{0.03F, -0.02F, 0.01F}, {-0.02F, 0.01F, 0.00F}}};
    Phase second{{{0.09F, 0.02F, 0.00F},
                     {0.02F, 0.20F, 0.06F}},
        {{-0.02F, 0.01F, 0.00F}, {0.01F, -0.03F, 0.02F}}};
    prepare(first, 1'050.0F, {0.995037F, 0.0995037F, 0.0F}, 3.0F);
    prepare(second, 1'450.0F, {0.995037F, 0.0995037F, 0.0F}, 3.0F);
    apply(first, second, first_material, second_material, 0.05F);
    check_conservation(first, second, first_material, second_material);
    check_endpoint_reversal(first, second, first_material, second_material,
        0.05F);

    // Contact alone keeps the phases apart inside the radius sum.
    first = {{{0.0F, 0.0F, 0.0F}}};
    second = {{{0.10F, 0.0F, 0.0F}}};
    prepare(first, first_material.rest_density, {}, 0.0F, 0U);
    prepare(second, second_material.rest_density, {}, 0.0F, 0U);
    apply(first, second, first_material, second_material, 0.0F);
    CHECK(first.accelerations[0].x < 0.0F);
    CHECK(second.accelerations[0].x > 0.0F);
    check_conservation(first, second, first_material, second_material);

    // Outside contact, heterogeneous pressure still transfers normally.
    first.positions[0] = {0.0F, 0.0F, 0.0F};
    second.positions[0] = {0.20F, 0.0F, 0.0F};
    first.densities[0] = 1'050.0F;
    second.densities[0] = 1'450.0F;
    apply(first, second, first_material, second_material, 0.0F);
    CHECK(first.accelerations[0].x < 0.0F);
    CHECK(second.accelerations[0].x > 0.0F);
    check_conservation(first, second, first_material, second_material);
}

void test_planar_and_degenerate_interfaces_add_no_capillarity() {
    auto first_material = material(800.0F, 0.002F);
    auto second_material = material(1'200.0F, 0.001F);
    first_material.repulsion = 20.0F;
    second_material.repulsion = 70.0F;
    Phase first{{{0.0F, 0.0F, 0.0F}}};
    Phase second{{{0.2F, 0.0F, 0.0F}}};
    prepare(first, 1'050.0F);
    prepare(second, 1'450.0F);

    auto baseline_first = first;
    auto baseline_second = second;
    apply(baseline_first, baseline_second,
        first_material, second_material, 0.0F);
    apply(first, second, first_material, second_material, 0.08F);
    check_close(first.accelerations[0], baseline_first.accelerations[0], 0.0F);
    check_close(second.accelerations[0], baseline_second.accelerations[0], 0.0F);

    first.curvatures[0] = 4.0F;
    second.curvatures[0] = 4.0F;
    second.flags[0] |= k_fluid_interface_degenerate;
    apply(first, second, first_material, second_material, 0.08F);
    check_close(first.accelerations[0], baseline_first.accelerations[0], 0.0F);
    check_close(second.accelerations[0], baseline_second.accelerations[0], 0.0F);
}

void test_young_laplace_radius_and_tension_sweeps() {
    const auto first_material = material(800.0F, 0.002F);
    const auto second_material = material(1'200.0F, 0.001F);
    constexpr float distance = 0.2F;
    const float volume_product = first_material.rest_particle_volume *
        second_material.rest_particle_volume;
    const float gradient = pressure_gradient(
        distance, first_material.support_radius);

    for (const float radius : {0.5F, 1.0F, 2.0F}) {
        for (const float tension : {0.02F, 0.07F}) {
            Phase first{{{0.0F, 0.0F, 0.0F}}};
            Phase second{{{distance, 0.0F, 0.0F}}};
            prepare(first, first_material.rest_density,
                {1.0F, 0.0F, 0.0F}, 2.0F / radius);
            prepare(second, second_material.rest_density,
                {1.0F, 0.0F, 0.0F}, 2.0F / radius);
            apply(first, second, first_material, second_material, tension);

            const float force = first.accelerations[0].x *
                particle_mass(first_material);
            CHECK(force > 0.0F);
            const float recovered_jump = force / (volume_product * gradient);
            const float expected_jump = tension * 2.0F / radius;
            check_close(recovered_jump, expected_jump, 2.0e-5F);
            check_conservation(first, second,
                first_material, second_material);
            check_endpoint_reversal(first, second,
                first_material, second_material, tension);
        }
    }

    Phase first{{{0.0F, 0.0F, 0.0F}}};
    Phase second{{{distance, 0.0F, 0.0F}}};
    prepare(first, first_material.rest_density,
        {1.0F, 0.0F, 0.0F}, 4.0F);
    prepare(second, second_material.rest_density,
        {1.0F, 0.0F, 0.0F}, 4.0F);
    apply(first, second, first_material, second_material, 0.0F);
    check_close(first.accelerations[0], {}, 0.0F);
    check_close(second.accelerations[0], {}, 0.0F);
}
} // namespace

int main() {
    test_contact_pressure_conservation_and_reversal();
    test_planar_and_degenerate_interfaces_add_no_capillarity();
    test_young_laplace_radius_and_tension_sweeps();
    return 0;
}
