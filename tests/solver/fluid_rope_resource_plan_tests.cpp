// SPDX-License-Identifier: MIT
#include "simulation/memory_arena.hpp"
#include "simulation/resource_plan.hpp"

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
struct FluidRopePlan {
    std::uint32_t *contribution_nodes{};
    Vec3 *contribution_impulses{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
};

struct ArenaAllocator {
    static constexpr bool materialize = true;
    MemoryArena &arena;

    template<class T>
    Status operator()(T *&pointer, std::size_t count,
                      MemoryClass memory_class) noexcept {
        pointer = static_cast<T *>(
            arena.allocate(count * sizeof(T), alignof(T), memory_class));
        return pointer != nullptr
            ? Status{}
            : Status{StatusCode::out_of_memory, 0,
                     "exact fluid rope plan allocation failed"};
    }
};
} // namespace

int main() {
    FluidRopePlan planned{};
    StoragePlanner planner;
    CHECK(visit_fluid_rope_storage(planned, 3U, planner));
    CHECK(planner.requirements.persistent != 0U);
    CHECK(planner.requirements.scratch == 0U);

    std::vector<std::byte> backing(planner.requirements.persistent + 1U);
    MemoryArena arena(backing.data() + 1U, planner.requirements.persistent,
                      nullptr, 0U);
    FluidRopePlan allocated{};
    ArenaAllocator allocator{arena};
    CHECK(visit_fluid_rope_storage(allocated, 3U, allocator));
    CHECK(allocated.contribution_nodes != nullptr);
    CHECK(allocated.contribution_impulses != nullptr);
    CHECK(allocated.contact_count != nullptr);
    CHECK(allocated.maximum_penetration != nullptr);
    CHECK(arena.persistent_used() <= planner.requirements.persistent);

    FluidRopePlan overflow{};
    StoragePlanner overflow_planner;
    CHECK(visit_fluid_rope_storage(
              overflow, std::numeric_limits<std::size_t>::max(),
              overflow_planner).code == StatusCode::out_of_memory);

    std::puts("PASS: per-particle fluid-rope contribution plan and overflow");
}
