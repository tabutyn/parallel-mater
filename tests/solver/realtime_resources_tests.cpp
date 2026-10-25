// SPDX-License-Identifier: MIT
#include "simulation/realtime_resources.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>

using namespace parallel_mater::simulation;

#define CHECK(value) do { if (!(value)) {                                  \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value);              \
    std::abort();                                                           \
} } while (false)

namespace {

constexpr std::uint32_t executable = realtime_domain_execution_phase_mask |
    realtime_phase_bit(RealtimeResourcePhase::boundary) |
    realtime_phase_bit(RealtimeResourcePhase::commit);

RealtimeResourcePlan plan() {
    RealtimeResourcePlan result{};
    result.families = {{{8U, 8U}, {3U, 96U}, {2U, 48U}, {2U, 32U},
                        {4U, 64U}, {2U, 128U}, {2U, 512U}}};
    result.shared_smoke_slots = 2U;
    return result;
}

std::array<RealtimeResourceRegistration, realtime_resource_family_count>
empty_registration() {
    return {};
}

RealtimeResourceRegistration registration(
    std::uint32_t slots, std::uint64_t elements,
    std::uint64_t identity, std::uint32_t phases = executable) {
    return {slots, elements, identity, phases};
}

void plan_and_binding_contract() {
    ProductionRealtimeResources resources;
    auto invalid = plan();
    invalid.families[0].elements = 0U;
    CHECK(resources.configure(invalid) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(resources.configure(plan()) == PM_SUCCESS);
    CHECK(resources.configured());
    CHECK(resources.plan().shared_smoke_slots == 2U);
    CHECK(resources.configure(plan()) == PM_ERROR_INVALID_STATE);
    for (const auto &binding : realtime_phase_bindings) {
        CHECK(!binding.permits_host_readback);
        CHECK(!binding.permits_host_wait);
    }
    CHECK(realtime_phase_bindings.front().reads_public);
    CHECK(realtime_phase_bindings.back().publishes_public);
}

void empty_sparse_and_mixed_registration() {
    ProductionRealtimeResources resources;
    CHECK(resources.configure(plan()) == PM_SUCCESS);
    auto domains = empty_registration();
    CHECK(resources.stage(1U, 1U, 1U, 1U, 1U, domains) == PM_SUCCESS);
    CHECK(resources.execution_status() == PM_SUCCESS);
    CHECK(resources.publish_spatial() == PM_SUCCESS);
    CHECK(resources.publish_spatial() == PM_ERROR_INVALID_STATE);
    CHECK(resources.commit() == PM_SUCCESS);
    CHECK(resources.current()->families == domains);

    domains[realtime_index(RealtimeResourceFamily::liquid)] =
        registration(1U, 8U, 0x1001U, 0U);
    CHECK(resources.stage(2U, 2U, 2U, 2U, 2U, domains) == PM_SUCCESS);
    CHECK(resources.execution_status() == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(resources.abort() == PM_SUCCESS);
    CHECK(resources.current()->token.tick == 1U);

    domains = empty_registration();
    for (std::size_t index = 0U; index < domains.size(); ++index) {
        const auto &capacity = resources.plan().families[index];
        domains[index] = registration(
            index == realtime_index(RealtimeResourceFamily::particle_smoke) ||
                    index == realtime_index(RealtimeResourceFamily::grid_smoke)
                ? 1U : capacity.slots,
            index == realtime_index(RealtimeResourceFamily::particle_smoke) ||
                    index == realtime_index(RealtimeResourceFamily::grid_smoke)
                ? capacity.elements / 2U : capacity.elements,
            0x2000U + index);
    }
    CHECK(resources.stage(2U, 2U, 2U, 2U, 2U, domains) == PM_SUCCESS);
    CHECK(resources.publish_spatial() == PM_SUCCESS);
    CHECK(resources.commit() == PM_SUCCESS);
    CHECK(resources.current()->families == domains);
}

void failure_is_atomic_and_capacity_reusable() {
    ProductionRealtimeResources resources;
    CHECK(resources.configure(plan()) == PM_SUCCESS);
    auto domains = empty_registration();
    domains[realtime_index(RealtimeResourceFamily::rigid)] =
        registration(2U, 2U, 0x3001U);
    CHECK(resources.stage(1U, 1U, 1U, 1U, 1U, domains) == PM_SUCCESS);
    CHECK(resources.publish_spatial() == PM_SUCCESS);
    CHECK(resources.commit() == PM_SUCCESS);
    const auto completed = *resources.current();
    const auto capacity = resources.plan();

    domains[realtime_index(RealtimeResourceFamily::liquid)] =
        registration(4U, 8U, 0x3002U);
    CHECK(resources.stage(2U, 2U, 2U, 2U, 2U, domains) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(!resources.pending());
    CHECK(*resources.current() == completed);
    CHECK(resources.plan().families == capacity.families);

    domains[realtime_index(RealtimeResourceFamily::liquid)] =
        registration(1U, 97U, 0x3002U);
    CHECK(resources.stage(2U, 2U, 2U, 2U, 2U, domains) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    domains[realtime_index(RealtimeResourceFamily::liquid)] = {};
    domains[realtime_index(RealtimeResourceFamily::rigid)].identity_hash++;
    CHECK(resources.stage(2U, 2U, 2U, 2U, 2U, domains) == PM_SUCCESS);
    const auto failed_generation = resources.pending()->token.generation;
    CHECK(resources.abort() == PM_SUCCESS);
    CHECK(*resources.current() == completed);
    CHECK(resources.stage(2U, 2U, 2U, 2U, 2U, domains) == PM_SUCCESS);
    CHECK(resources.pending()->token.generation > failed_generation);
    CHECK(resources.publish_spatial() == PM_SUCCESS);
    CHECK(resources.commit() == PM_SUCCESS);
}

void smoke_sharing_and_generation_invalidation() {
    ProductionRealtimeResources resources;
    CHECK(resources.configure(plan()) == PM_SUCCESS);
    auto domains = empty_registration();
    domains[realtime_index(RealtimeResourceFamily::particle_smoke)] =
        registration(2U, 4U, 0x4001U);
    domains[realtime_index(RealtimeResourceFamily::grid_smoke)] =
        registration(1U, 8U, 0x4002U);
    CHECK(resources.stage(1U, 1U, 1U, 1U, 1U, domains) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    domains[realtime_index(RealtimeResourceFamily::particle_smoke)].live_slots =
        1U;
    CHECK(resources.stage(1U, 1U, 1U, 1U, 1U, domains) == PM_SUCCESS);
    const auto staged = resources.pending()->token;
    resources.invalidate(1U);
    CHECK(!resources.pending());
    CHECK(!resources.current());
    CHECK(resources.stage(1U, 2U, 1U, 2U, 2U, domains) ==
          PM_ERROR_INVALID_STATE);
    CHECK(resources.stage(2U, 2U, 2U, 2U, 2U, domains) == PM_SUCCESS);
    CHECK(resources.pending()->token.generation > staged.generation);
}

} // namespace

int main() {
    plan_and_binding_contract();
    empty_sparse_and_mixed_registration();
    failure_is_atomic_and_capacity_reusable();
    smoke_sharing_and_generation_invalidation();
    std::puts("PASS: realtime resources stage every domain atomically");
}
