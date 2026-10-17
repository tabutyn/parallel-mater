// SPDX-License-Identifier: MIT
#include "simulation/fluid_dispatch.hpp"
#include "simulation/primitives.hpp"
#include "simulation/resource_plan.hpp"
#include "simulation/types.hpp"

#include <algorithm>
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
} // namespace

int main() {
    test_mixed_populations_and_limits();
    test_cell_and_support_boundaries();
    test_empty_live_populations();
    test_bounded_storage_plan();
    std::puts("PASS: shared two-fluid neighbor index and bounded endpoint accounting");
}
