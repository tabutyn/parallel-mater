// SPDX-License-Identifier: MIT
#include "simulation/fluid_dispatch.hpp"
#include "simulation/primitives.hpp"
#include "simulation/resource_plan.hpp"
#include "simulation/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

using namespace parallel_mater;
using namespace parallel_mater::simulation;

namespace {
struct Result {
    std::vector<std::uint32_t> first_total;
    std::vector<std::uint32_t> first_cross;
    std::vector<std::uint32_t> second_total;
    std::vector<std::uint32_t> second_cross;
    std::vector<std::uint32_t> sorted_identities;
    std::uint32_t first_overflow{};
    std::uint32_t second_overflow{};
    std::uint32_t first_maximum{};
    std::uint32_t second_maximum{};
};

struct FluidState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    std::vector<Vec3> previous;
    std::vector<Vec3> forces;
    std::vector<float> foam;
    std::vector<float> foam_source;
};

void compute_self_forces(FluidState &state,
    FluidDispatchConstants constants) {
    const auto count = static_cast<std::uint32_t>(state.positions.size());
    constants.count = count;
    std::vector<std::uint64_t> keys[2];
    std::vector<std::uint32_t> indices[2];
    for (unsigned slot = 0U; slot < 2U; ++slot) {
        keys[slot].resize(count);
        indices[slot].resize(count);
    }
    std::vector<std::byte> workspace(radix_sort_scratch_bytes(count));
    std::uint32_t live = count;
    fluid_cell_keys((count + 127U) / 128U, 128U, {},
        state.positions.data(), &live, count,
        1.0F / constants.support_radius, keys[0].data(), indices[0].data());
    CHECK(radix_sort_pairs_u64_u32({}, keys[0].data(), keys[1].data(),
        indices[0].data(), indices[1].data(), count,
        workspace.data(), workspace.size()) == PM_SUCCESS);
    state.previous.resize(count);
    state.forces.resize(count);
    state.foam.assign(count, 0.0F);
    state.foam_source.resize(count);
    std::uint32_t overflow{};
    std::uint32_t maximum{};
    const float up[3]{0.0F, 1.0F, 0.0F};
    fluid_forces((count + 127U) / 128U, 128U, {},
        state.positions.data(), state.velocities.data(), state.forces.data(),
        state.foam.data(), &live, constants, up, state.foam_source.data(),
        keys[1].data(), indices[1].data(), &overflow, &maximum);
    CHECK(overflow == 0U);
}

void integrate(FluidState &state, FluidDispatchConstants constants) {
    auto count = static_cast<std::uint32_t>(state.positions.size());
    constants.count = count;
    integrate_fluid((count + 127U) / 128U, 128U, {},
        state.positions.data(), state.velocities.data(), state.forces.data(),
        state.foam.data(), &count, constants, state.previous.data(),
        state.foam_source.data());
}

void check_close(Vec3 left, Vec3 right, float tolerance) {
    CHECK(std::fabs(left.x - right.x) <= tolerance);
    CHECK(std::fabs(left.y - right.y) <= tolerance);
    CHECK(std::fabs(left.z - right.z) <= tolerance);
}

Result count_neighbors(const std::vector<Vec3> &first,
    std::uint32_t first_count, std::uint32_t first_limit,
    const std::vector<Vec3> &second, std::uint32_t second_count,
    std::uint32_t second_limit, float radius) {
    const auto first_capacity = static_cast<std::uint32_t>(first.size());
    const auto second_capacity = static_cast<std::uint32_t>(second.size());
    const auto total = first_capacity + second_capacity;
    std::vector<std::uint64_t> keys[2];
    std::vector<std::uint32_t> identities[2];
    keys[0].resize(total);
    keys[1].resize(total);
    identities[0].resize(total);
    identities[1].resize(total);
    fluid_pair_cell_keys((total + 127U) / 128U, 128U, {},
        first.data(), &first_count, first_capacity,
        second.data(), &second_count, second_capacity,
        1.0F / radius, keys[0].data(), identities[0].data());
    std::vector<std::byte> workspace(radix_sort_scratch_bytes(total));
    CHECK(radix_sort_pairs_u64_u32({}, keys[0].data(), keys[1].data(),
        identities[0].data(), identities[1].data(), total,
        workspace.data(), workspace.size()) == PM_SUCCESS);

    Result result;
    result.first_total.assign(first_capacity, 99U);
    result.first_cross.assign(first_capacity, 99U);
    result.second_total.assign(second_capacity, 99U);
    result.second_cross.assign(second_capacity, 99U);
    result.sorted_identities = identities[1];
    fluid_pair_neighbor_counts((total + 127U) / 128U, 128U, {},
        first.data(), &first_count, first_capacity, first_limit,
        second.data(), &second_count, second_capacity, second_limit,
        keys[1].data(), identities[1].data(), total, radius,
        result.first_total.data(), result.first_cross.data(),
        result.second_total.data(), result.second_cross.data(),
        &result.first_overflow, &result.second_overflow,
        &result.first_maximum, &result.second_maximum);
    return result;
}

void apply_pair_viscosity(FluidState &first, FluidState &second,
    const FluidDispatchConstants &constants) {
    const auto first_count = static_cast<std::uint32_t>(first.positions.size());
    const auto second_count = static_cast<std::uint32_t>(second.positions.size());
    const auto total = first_count + second_count;
    std::vector<std::uint64_t> keys[2];
    std::vector<std::uint32_t> identities[2];
    for (unsigned slot = 0U; slot < 2U; ++slot) {
        keys[slot].resize(total);
        identities[slot].resize(total);
    }
    fluid_pair_cell_keys((total + 127U) / 128U, 128U, {},
        first.positions.data(), &first_count, first_count,
        second.positions.data(), &second_count, second_count,
        1.0F / constants.support_radius, keys[0].data(), identities[0].data());
    std::vector<std::byte> workspace(radix_sort_scratch_bytes(total));
    CHECK(radix_sort_pairs_u64_u32({}, keys[0].data(), keys[1].data(),
        identities[0].data(), identities[1].data(), total,
        workspace.data(), workspace.size()) == PM_SUCCESS);
    fluid_pair_viscosity((total + 127U) / 128U, 128U, {},
        first.positions.data(), first.velocities.data(), &first_count,
        first_count, first.forces.data(), second.positions.data(),
        second.velocities.data(), &second_count, second_count,
        second.forces.data(), keys[1].data(), identities[1].data(), total,
        constants);
}

Vec3 momentum(const FluidState &first, const FluidState &second,
    float timestep) {
    Vec3 result{};
    for (std::size_t index = 0U; index < first.velocities.size(); ++index) {
        result.x += first.velocities[index].x + timestep * first.forces[index].x;
        result.y += first.velocities[index].y + timestep * first.forces[index].y;
        result.z += first.velocities[index].z + timestep * first.forces[index].z;
    }
    for (std::size_t index = 0U; index < second.velocities.size(); ++index) {
        result.x += second.velocities[index].x + timestep * second.forces[index].x;
        result.y += second.velocities[index].y + timestep * second.forces[index].y;
        result.z += second.velocities[index].z + timestep * second.forces[index].z;
    }
    return result;
}

float kinetic_energy(const FluidState &first, const FluidState &second,
    float timestep) {
    float result = 0.0F;
    const auto accumulate = [&](const FluidState &state) {
        for (std::size_t index = 0U; index < state.velocities.size(); ++index) {
            const Vec3 velocity{
                state.velocities[index].x + timestep * state.forces[index].x,
                state.velocities[index].y + timestep * state.forces[index].y,
                state.velocities[index].z + timestep * state.forces[index].z};
            result += 0.5F * (velocity.x * velocity.x +
                velocity.y * velocity.y + velocity.z * velocity.z);
        }
    };
    accumulate(first);
    accumulate(second);
    return result;
}

void test_mixed_populations_and_limits() {
    const std::vector<Vec3> first{{0.0F, 0.0F, 0.0F},
        {0.75F, 0.0F, 0.0F}, {100.0F, 0.0F, 0.0F}};
    const std::vector<Vec3> second{{0.25F, 0.0F, 0.0F},
        {1.5F, 0.0F, 0.0F}, {-100.0F, 0.0F, 0.0F}};
    const auto result = count_neighbors(first, 2U, 1U, second, 2U, 1U, 1.0F);
    CHECK((result.first_total == std::vector<std::uint32_t>{2U, 3U, 0U}));
    CHECK((result.first_cross == std::vector<std::uint32_t>{1U, 2U, 0U}));
    CHECK((result.second_total == std::vector<std::uint32_t>{2U, 1U, 0U}));
    CHECK((result.second_cross == std::vector<std::uint32_t>{2U, 1U, 0U}));
    CHECK(result.first_overflow == 2U);
    CHECK(result.second_overflow == 1U);
    CHECK(result.first_maximum == 3U);
    CHECK(result.second_maximum == 2U);
    CHECK(std::find(result.sorted_identities.begin(),
        result.sorted_identities.end(), 0U) != result.sorted_identities.end());
    CHECK(std::find(result.sorted_identities.begin(),
        result.sorted_identities.end(), 0x80000000U) !=
        result.sorted_identities.end());
}

void test_cell_and_support_boundaries() {
    const std::vector<Vec3> first{{-0.01F, 0.0F, 0.0F},
        {50.0F, 0.0F, 0.0F}};
    const std::vector<Vec3> second{{0.01F, 0.0F, 0.0F},
        {0.99F, 0.0F, 0.0F}};
    const auto result = count_neighbors(first, 1U, 8U, second, 2U, 8U, 1.0F);
    CHECK((result.first_total == std::vector<std::uint32_t>{1U, 0U}));
    CHECK((result.first_cross == std::vector<std::uint32_t>{1U, 0U}));
    CHECK((result.second_total == std::vector<std::uint32_t>{2U, 1U}));
    CHECK((result.second_cross == std::vector<std::uint32_t>{1U, 0U}));
    CHECK(result.first_overflow == 0U && result.second_overflow == 0U);
}

void test_empty_live_populations() {
    const std::vector<Vec3> first{{0.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F}};
    const std::vector<Vec3> second{{0.1F, 0.0F, 0.0F},
        {2.0F, 0.0F, 0.0F}};
    auto result = count_neighbors(first, 0U, 1U, second, 1U, 1U, 1.0F);
    CHECK((result.first_total == std::vector<std::uint32_t>{0U, 0U}));
    CHECK((result.second_total == std::vector<std::uint32_t>{0U, 0U}));
    CHECK(result.first_overflow == 0U && result.second_overflow == 0U);
    result = count_neighbors(first, 1U, 1U, second, 0U, 1U, 1.0F);
    CHECK((result.first_total == std::vector<std::uint32_t>{0U, 0U}));
    CHECK((result.second_total == std::vector<std::uint32_t>{0U, 0U}));
    result = count_neighbors(first, 0U, 1U, second, 0U, 1U, 1.0F);
    CHECK((result.first_cross == std::vector<std::uint32_t>{0U, 0U}));
    CHECK((result.second_cross == std::vector<std::uint32_t>{0U, 0U}));
}

void test_bounded_storage_plan() {
    StorageRequirements requirements{};
    CHECK(fluid_pair_neighbor_index_storage_requirements(3U, 5U, requirements));
    CHECK(requirements.persistent != 0U);
    CHECK(requirements.scratch >= radix_sort_scratch_bytes(8U));
    FluidPairNeighborIndexPlan plan{};
    StoragePlanner planner;
    CHECK(visit_fluid_pair_neighbor_index_storage(plan, 3U, 5U, planner));
    CHECK(plan.total_capacity == 8U);
    CHECK(plan.sort_workspace_size == radix_sort_scratch_bytes(8U));
    CHECK(planner.requirements.persistent == requirements.persistent);
    CHECK(planner.requirements.scratch == requirements.scratch);
    CHECK(!fluid_pair_neighbor_index_storage_requirements(
        0U, 5U, requirements));
    CHECK(!fluid_pair_neighbor_index_storage_requirements(
        std::numeric_limits<std::uint32_t>::max(), 1U, requirements));
}

void test_pressure_contact_matches_combined_and_balances_momentum() {
    FluidDispatchConstants constants{};
    constants.timestep = 0.01F;
    constants.count = 4U;
    constants.particle_radius = 0.1F;
    constants.support_radius = 1.0F;
    constants.repulsion = 40.0F;
    constants.viscosity = 0.0F;
    constants.normal_damping = 2.0F;
    constants.velocity_damping = 0.0F;
    constants.maximum_speed = 100.0F;
    constants.rest_density = 1'000.0F;
    constants.maximum_neighbors = 16U;

    FluidState first{{{-0.55F, 0.0F, 0.0F},
                          {-0.15F, 0.0F, 0.0F}},
        {{0.02F, 0.0F, 0.0F}, {-0.01F, 0.0F, 0.0F}}};
    FluidState second{{{-0.15F, 0.0F, 0.0F},
                           {0.65F, 0.0F, 0.0F}},
        {{-0.02F, 0.0F, 0.0F}, {0.01F, 0.0F, 0.0F}}};
    FluidState combined;
    combined.positions = first.positions;
    combined.positions.insert(combined.positions.end(),
        second.positions.begin(), second.positions.end());
    combined.velocities = first.velocities;
    combined.velocities.insert(combined.velocities.end(),
        second.velocities.begin(), second.velocities.end());

    compute_self_forces(first, constants);
    compute_self_forces(second, constants);
    compute_self_forces(combined, constants);
    const auto first_self = first.forces;
    const auto second_self = second.forces;

    const auto first_count = static_cast<std::uint32_t>(first.positions.size());
    const auto second_count = static_cast<std::uint32_t>(second.positions.size());
    const auto total = first_count + second_count;
    std::vector<std::uint64_t> keys[2];
    std::vector<std::uint32_t> identities[2];
    for (unsigned slot = 0U; slot < 2U; ++slot) {
        keys[slot].resize(total);
        identities[slot].resize(total);
    }
    fluid_pair_cell_keys((total + 127U) / 128U, 128U, {},
        first.positions.data(), &first_count, first_count,
        second.positions.data(), &second_count, second_count,
        1.0F / constants.support_radius, keys[0].data(), identities[0].data());
    std::vector<std::byte> workspace(radix_sort_scratch_bytes(total));
    CHECK(radix_sort_pairs_u64_u32({}, keys[0].data(), keys[1].data(),
        identities[0].data(), identities[1].data(), total,
        workspace.data(), workspace.size()) == PM_SUCCESS);
    fluid_pair_pressure_contact((total + 127U) / 128U, 128U, {},
        first.positions.data(), first.velocities.data(), &first_count,
        first_count, first.forces.data(), second.positions.data(),
        second.velocities.data(), &second_count, second_count,
        second.forces.data(), keys[1].data(), identities[1].data(), total,
        constants);

    Vec3 residual{};
    float exchanged = 0.0F;
    for (std::uint32_t index = 0U; index < first_count; ++index) {
        const Vec3 cross{first.forces[index].x - first_self[index].x,
            first.forces[index].y - first_self[index].y,
            first.forces[index].z - first_self[index].z};
        residual.x += cross.x;
        residual.y += cross.y;
        residual.z += cross.z;
        exchanged += std::sqrt(cross.x * cross.x + cross.y * cross.y +
            cross.z * cross.z);
        check_close(first.forces[index], combined.forces[index], 1.0e-4F);
    }
    for (std::uint32_t index = 0U; index < second_count; ++index) {
        const Vec3 cross{second.forces[index].x - second_self[index].x,
            second.forces[index].y - second_self[index].y,
            second.forces[index].z - second_self[index].z};
        residual.x += cross.x;
        residual.y += cross.y;
        residual.z += cross.z;
        exchanged += std::sqrt(cross.x * cross.x + cross.y * cross.y +
            cross.z * cross.z);
        check_close(second.forces[index],
            combined.forces[first_count + index], 1.0e-4F);
    }
    const float residual_length = std::sqrt(residual.x * residual.x +
        residual.y * residual.y + residual.z * residual.z);
    const float diameter = 2.0F * constants.particle_radius;
    const float particle_mass = constants.rest_density *
        diameter * diameter * diameter;
    const float exchanged_impulse =
        particle_mass * constants.timestep * exchanged;
    const float residual_impulse =
        particle_mass * constants.timestep * residual_length;
    CHECK(exchanged_impulse >= 1.0e-4F);
    CHECK(residual_impulse <=
        std::max(1.0e-7F, 5.0e-5F * exchanged_impulse));

    integrate(first, constants);
    integrate(second, constants);
    integrate(combined, constants);
    for (std::uint32_t index = 0U; index < first_count; ++index) {
        check_close(first.positions[index], combined.positions[index], 1.0e-6F);
        check_close(first.velocities[index], combined.velocities[index], 1.0e-5F);
        CHECK(std::fabs(first.foam[index] - combined.foam[index]) <= 1.0e-6F);
    }
    for (std::uint32_t index = 0U; index < second_count; ++index) {
        check_close(second.positions[index],
            combined.positions[first_count + index], 1.0e-6F);
        check_close(second.velocities[index],
            combined.velocities[first_count + index], 1.0e-5F);
        CHECK(std::fabs(second.foam[index] - combined.foam[first_count + index]) <=
            1.0e-6F);
    }
}

void test_viscosity_matches_combined_population() {
    FluidDispatchConstants constants{};
    constants.timestep = 0.02F;
    constants.count = 4U;
    constants.particle_radius = 0.1F;
    constants.support_radius = 1.0F;
    constants.viscosity = 0.6F;
    constants.maximum_speed = 100.0F;
    constants.rest_density = 1'000.0F;
    constants.maximum_neighbors = 16U;

    FluidState first{{{-0.55F, 0.0F, 0.0F},
                          {-0.15F, 0.0F, 0.0F}},
        {{0.10F, 0.0F, 0.0F}, {-0.04F, 0.0F, 0.0F}}};
    FluidState second{{{0.15F, 0.0F, 0.0F},
                           {0.65F, 0.0F, 0.0F}},
        {{-0.08F, 0.0F, 0.0F}, {0.02F, 0.0F, 0.0F}}};
    FluidState combined;
    combined.positions = first.positions;
    combined.positions.insert(combined.positions.end(),
        second.positions.begin(), second.positions.end());
    combined.velocities = first.velocities;
    combined.velocities.insert(combined.velocities.end(),
        second.velocities.begin(), second.velocities.end());

    compute_self_forces(first, constants);
    compute_self_forces(second, constants);
    compute_self_forces(combined, constants);
    apply_pair_viscosity(first, second, constants);

    for (std::size_t index = 0U; index < first.forces.size(); ++index)
        check_close(first.forces[index], combined.forces[index], 1.0e-5F);
    for (std::size_t index = 0U; index < second.forces.size(); ++index)
        check_close(second.forces[index],
            combined.forces[first.forces.size() + index], 1.0e-5F);

    integrate(first, constants);
    integrate(second, constants);
    integrate(combined, constants);
    for (std::size_t index = 0U; index < first.positions.size(); ++index) {
        check_close(first.positions[index], combined.positions[index], 1.0e-6F);
        check_close(first.velocities[index], combined.velocities[index], 1.0e-6F);
    }
    for (std::size_t index = 0U; index < second.positions.size(); ++index) {
        check_close(second.positions[index],
            combined.positions[first.positions.size() + index], 1.0e-6F);
        check_close(second.velocities[index],
            combined.velocities[first.positions.size() + index], 1.0e-6F);
    }
}

void test_viscosity_analytic_and_isolated_limits() {
    FluidDispatchConstants constants{};
    constants.timestep = 0.1F;
    constants.particle_radius = 0.1F;
    constants.support_radius = 1.0F;
    constants.viscosity = 0.4F;
    constants.maximum_speed = 100.0F;
    constants.rest_density = 1'000.0F;
    constants.maximum_neighbors = 4U;
    FluidState first{{{0.0F, 0.0F, 0.0F}}, {{1.0F, 0.0F, 0.0F}}};
    FluidState second{{{0.5F, 0.0F, 0.0F}}, {{-1.0F, 0.0F, 0.0F}}};
    first.forces.assign(1U, {});
    second.forces.assign(1U, {});
    apply_pair_viscosity(first, second, constants);
    check_close(first.forces[0], {-0.4F, 0.0F, 0.0F}, 1.0e-6F);
    check_close(second.forces[0], {0.4F, 0.0F, 0.0F}, 1.0e-6F);
    check_close(momentum(first, second, constants.timestep), {}, 1.0e-6F);
    CHECK(kinetic_energy(first, second, constants.timestep) < 1.0F);

    constants.viscosity = 0.0F;
    first.forces.assign(1U, {});
    second.forces.assign(1U, {});
    apply_pair_viscosity(first, second, constants);
    check_close(first.forces[0], {}, 0.0F);
    check_close(second.forces[0], {}, 0.0F);

    constants.viscosity = 0.4F;
    second.positions[0] = {constants.support_radius, 0.0F, 0.0F};
    apply_pair_viscosity(first, second, constants);
    check_close(first.forces[0], {}, 0.0F);
    check_close(second.forces[0], {}, 0.0F);
}

void test_viscosity_timestep_and_population_mass_ratio_sweeps() {
    for (const std::uint32_t second_count : {1U, 2U, 4U}) {
        for (const float timestep : {0.001F, 0.01F, 0.05F}) {
            FluidDispatchConstants constants{};
            constants.timestep = timestep;
            constants.particle_radius = 0.1F;
            constants.support_radius = 1.0F;
            constants.viscosity = 0.4F;
            constants.maximum_speed = 100.0F;
            constants.rest_density = 1'000.0F;
            constants.maximum_neighbors = 8U;
            FluidState first{{{0.0F, 0.0F, 0.0F}},
                {{1.0F, 0.25F, 0.0F}}};
            FluidState second;
            for (std::uint32_t index = 0U; index < second_count; ++index) {
                second.positions.push_back(
                    {0.2F, 0.05F * static_cast<float>(index), 0.0F});
                second.velocities.push_back(
                    {-1.0F / static_cast<float>(second_count),
                     -0.25F / static_cast<float>(second_count), 0.0F});
            }
            first.forces.assign(1U, {});
            second.forces.assign(second_count, {});
            const Vec3 before = momentum(first, second, 0.0F);
            const float energy_before = kinetic_energy(first, second, 0.0F);
            apply_pair_viscosity(first, second, constants);
            const Vec3 after = momentum(first, second, timestep);
            const float energy_after = kinetic_energy(first, second, timestep);
            check_close(before, after, 2.0e-6F);
            CHECK(energy_after <= energy_before + 1.0e-6F);
        }
    }
}
} // namespace

int main() {
    test_mixed_populations_and_limits();
    test_cell_and_support_boundaries();
    test_empty_live_populations();
    test_bounded_storage_plan();
    test_pressure_contact_matches_combined_and_balances_momentum();
    test_viscosity_matches_combined_population();
    test_viscosity_analytic_and_isolated_limits();
    test_viscosity_timestep_and_population_mass_ratio_sweeps();
    std::puts("PASS: two-fluid neighbor index and balanced force exchange");
}
