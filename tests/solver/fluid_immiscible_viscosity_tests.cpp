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
    std::vector<Vec3> normals;
    std::vector<std::uint32_t> flags;
};

Vec3 added(Vec3 left, Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 scaled(Vec3 value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

float dot(Vec3 left, Vec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

float length(Vec3 value) {
    return std::sqrt(dot(value, value));
}

void check_close(float left, float right, float relative_tolerance) {
    CHECK(std::fabs(left - right) <= relative_tolerance *
        std::max(1.0e-6F, std::max(std::fabs(left), std::fabs(right))));
}

void check_close(Vec3 left, Vec3 right, float relative_tolerance) {
    check_close(left.x, right.x, relative_tolerance);
    check_close(left.y, right.y, relative_tolerance);
    check_close(left.z, right.z, relative_tolerance);
}

float particle_mass(const FluidDispatchConstants &constants) {
    const float diameter = 2.0F * constants.particle_radius;
    const float volume = constants.rest_particle_volume > 0.0F
        ? constants.rest_particle_volume
        : diameter * diameter * diameter;
    return constants.rest_density * volume;
}

FluidDispatchConstants material(float density, float volume,
    float viscosity, float support = 0.5F) {
    FluidDispatchConstants result{};
    result.particle_radius = 0.04F;
    result.support_radius = support;
    result.viscosity = viscosity;
    result.rest_density = density;
    result.rest_particle_volume = volume;
    result.maximum_neighbors = 16U;
    return result;
}

void prepare(Phase &phase, Vec3 normal = {1.0F, 0.0F, 0.0F},
    std::uint32_t flags = k_fluid_interface_sample) {
    const auto count = phase.positions.size();
    phase.accelerations.assign(count, {});
    phase.normals.assign(count, normal);
    phase.flags.assign(count, flags);
}

void apply(Phase &first, Phase &second,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) {
    const auto first_count = static_cast<std::uint32_t>(first.positions.size());
    const auto second_count = static_cast<std::uint32_t>(second.positions.size());
    const auto total = first_count + second_count;
    CHECK(first.velocities.size() == first_count);
    CHECK(second.velocities.size() == second_count);
    CHECK(first.normals.size() == first_count);
    CHECK(second.normals.size() == second_count);
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
    fluid_immiscible_pair_viscosity(
        (total + 127U) / 128U, 128U, {},
        first.positions.data(), first.velocities.data(), &first_count,
        first_count, first.accelerations.data(), first.normals.data(),
        first.flags.data(), second.positions.data(), second.velocities.data(),
        &second_count, second_count, second.accelerations.data(),
        second.normals.data(), second.flags.data(), keys[1].data(),
        identities[1].data(), total, first_constants, second_constants);
}

void check_momentum_and_dissipation(const Phase &first, const Phase &second,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) {
    Vec3 net_force{};
    float power = 0.0F;
    float activity = 0.0F;
    const auto accumulate = [&](const Phase &phase, float mass) {
        for (std::size_t index = 0U; index < phase.positions.size(); ++index) {
            const Vec3 force = scaled(phase.accelerations[index], mass);
            net_force = added(net_force, force);
            power += dot(phase.velocities[index], force);
            activity += length(force);
        }
    };
    accumulate(first, particle_mass(first_constants));
    accumulate(second, particle_mass(second_constants));
    CHECK(activity >= 1.0e-6F);
    CHECK(length(net_force) <= std::max(1.0e-7F, 5.0e-5F * activity));
    CHECK(power < 0.0F);
}

void check_endpoint_reversal(Phase first, Phase second,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) {
    apply(first, second, first_constants, second_constants);
    Phase reversed_first = second;
    Phase reversed_second = first;
    for (auto &normal : reversed_first.normals)
        normal = scaled(normal, -1.0F);
    for (auto &normal : reversed_second.normals)
        normal = scaled(normal, -1.0F);
    apply(reversed_first, reversed_second,
        second_constants, first_constants);
    CHECK(first.accelerations.size() == reversed_second.accelerations.size());
    CHECK(second.accelerations.size() == reversed_first.accelerations.size());
    for (std::size_t index = 0U; index < first.accelerations.size(); ++index)
        check_close(first.accelerations[index],
            reversed_second.accelerations[index], 2.0e-6F);
    for (std::size_t index = 0U; index < second.accelerations.size(); ++index)
        check_close(second.accelerations[index],
            reversed_first.accelerations[index], 2.0e-6F);
}

void test_harmonic_tangential_traction_sweeps() {
    struct Materials {
        FluidDispatchConstants first;
        FluidDispatchConstants second;
    };
    const Materials cases[]{
        {material(650.0F, 0.0030F, 0.40F, 0.40F),
            material(1'400.0F, 0.0007F, 0.15F, 0.50F)},
        {material(1'000.0F, 0.0010F, 0.25F, 0.45F),
            material(500.0F, 0.0040F, 0.70F, 0.35F)},
        {material(1'800.0F, 0.0004F, 0.10F, 0.30F),
            material(700.0F, 0.0025F, 0.55F, 0.55F)},
    };

    for (const auto &materials : cases) {
        Phase first{{{0.0F, 0.0F, 0.0F}}, {{1.0F, 0.5F, -0.25F}}};
        Phase second{{{0.2F, 0.0F, 0.0F}}, {{-0.5F, 0.25F, 0.75F}}};
        prepare(first);
        prepare(second);
        apply(first, second, materials.first, materials.second);

        const float first_mu = materials.first.rest_density *
            materials.first.viscosity;
        const float second_mu = materials.second.rest_density *
            materials.second.viscosity;
        const float pair_mu = 2.0F * first_mu * second_mu /
            (first_mu + second_mu);
        const float first_volume = particle_mass(materials.first) /
            materials.first.rest_density;
        const float second_volume = particle_mass(materials.second) /
            materials.second.rest_density;
        const float pair_volume = 2.0F * first_volume * second_volume /
            (first_volume + second_volume);
        const float support = std::max(materials.first.support_radius,
            materials.second.support_radius);
        const float coefficient = pair_mu * pair_volume *
            (1.0F - 0.2F / support);
        const Vec3 tangential_relative{0.0F, -0.25F, 1.0F};
        const Vec3 common_force = scaled(tangential_relative, coefficient);
        check_close(first.accelerations[0], scaled(common_force,
            1.0F / particle_mass(materials.first)), 2.0e-6F);
        check_close(second.accelerations[0], scaled(common_force,
            -1.0F / particle_mass(materials.second)), 2.0e-6F);
        check_close(first.accelerations[0].x, 0.0F, 0.0F);
        check_close(second.accelerations[0].x, 0.0F, 0.0F);
        check_momentum_and_dissipation(
            first, second, materials.first, materials.second);
        check_endpoint_reversal(
            first, second, materials.first, materials.second);
    }
}

void test_zero_viscosity_and_interface_limits() {
    auto first_material = material(800.0F, 0.002F, 0.3F);
    auto second_material = material(1'200.0F, 0.001F, 0.8F);
    Phase first{{{0.0F, 0.0F, 0.0F}}, {{0.0F, 1.0F, 0.0F}}};
    Phase second{{{0.2F, 0.0F, 0.0F}}, {{0.0F, -1.0F, 0.5F}}};
    prepare(first);
    prepare(second);

    auto inviscid_first = first_material;
    inviscid_first.viscosity = 0.0F;
    apply(first, second, inviscid_first, second_material);
    check_close(first.accelerations[0], {}, 0.0F);
    check_close(second.accelerations[0], {}, 0.0F);

    auto inviscid_second = second_material;
    inviscid_second.viscosity = 0.0F;
    apply(first, second, first_material, inviscid_second);
    check_close(first.accelerations[0], {}, 0.0F);
    check_close(second.accelerations[0], {}, 0.0F);

    first.flags[0] = 0U;
    apply(first, second, first_material, second_material);
    check_close(first.accelerations[0], {}, 0.0F);
    check_close(second.accelerations[0], {}, 0.0F);
    first.flags[0] = k_fluid_interface_sample;
    second.flags[0] |= k_fluid_interface_degenerate;
    apply(first, second, first_material, second_material);
    check_close(first.accelerations[0], {}, 0.0F);
    check_close(second.accelerations[0], {}, 0.0F);
}

void test_normal_relative_motion_adds_no_traction() {
    const auto first_material = material(800.0F, 0.002F, 0.3F);
    const auto second_material = material(1'200.0F, 0.001F, 0.8F);
    Phase first{{{0.0F, 0.0F, 0.0F}}, {{1.5F, 0.0F, 0.0F}}};
    Phase second{{{0.2F, 0.0F, 0.0F}}, {{-0.5F, 0.0F, 0.0F}}};
    prepare(first);
    prepare(second);
    apply(first, second, first_material, second_material);
    check_close(first.accelerations[0], {}, 0.0F);
    check_close(second.accelerations[0], {}, 0.0F);
}

void test_common_force_limit_respects_endpoint_masses() {
    auto first_material = material(800.0F, 0.002F, 0.3F);
    auto second_material = material(1'200.0F, 0.001F, 0.8F);
    first_material.maximum_pair_acceleration = 0.05F;
    second_material.maximum_pair_acceleration = 0.02F;
    Phase first{{{0.0F, 0.0F, 0.0F}}, {{0.0F, 100.0F, 0.0F}}};
    Phase second{{{0.2F, 0.0F, 0.0F}}, {{0.0F, -100.0F, 0.0F}}};
    prepare(first);
    prepare(second);
    apply(first, second, first_material, second_material);
    CHECK(length(first.accelerations[0]) <=
        first_material.maximum_pair_acceleration + 1.0e-6F);
    check_close(length(second.accelerations[0]),
        second_material.maximum_pair_acceleration, 2.0e-6F);
    check_momentum_and_dissipation(
        first, second, first_material, second_material);
    check_endpoint_reversal(
        first, second, first_material, second_material);
}
} // namespace

int main() {
    test_harmonic_tangential_traction_sweeps();
    test_zero_viscosity_and_interface_limits();
    test_normal_relative_motion_adds_no_traction();
    test_common_force_limit_respects_endpoint_masses();
    return 0;
}
