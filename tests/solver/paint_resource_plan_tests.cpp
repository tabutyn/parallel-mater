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
                     "exact paint plan allocation failed"};
    }
};

template<class T>
bool aligned(const T *pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0U;
}
} // namespace

int main() {
    StorageRequirements requirements{};
    CHECK(paint_storage_requirements(7U, 3U, 5U, requirements));
    CHECK(requirements.persistent != 0U);
    CHECK(requirements.scratch == 0U);

    std::vector<std::byte> backing(requirements.persistent + 1U);
    MemoryArena arena(backing.data() + 1U, requirements.persistent, nullptr, 0U);
    PaintStoragePlan allocated{};
    ArenaAllocator allocator{arena};
    CHECK(visit_paint_storage(allocated, 7U, 3U, 5U, allocator));
    CHECK(allocated.vertex_capacity == 7U);
    CHECK(allocated.pixel_capacity == 15U);
    CHECK(aligned(allocated.uvs));
    CHECK(aligned(allocated.pixels));
    CHECK(aligned(allocated.rollback_pixels));
    CHECK(arena.persistent_used() ==
          7U * sizeof(Vec2) + 30U * sizeof(std::uint32_t));
    CHECK(arena.persistent_used() <= requirements.persistent);
    CHECK(arena.scratch_used() == 0U);

    PaintStoragePlan invalid{};
    StoragePlanner invalid_planner;
    CHECK(visit_paint_storage(invalid, 0U, 3U, 5U, invalid_planner).code ==
          StatusCode::invalid_argument);
    CHECK(visit_paint_storage(invalid, 7U, 0U, 5U, invalid_planner).code ==
          StatusCode::invalid_argument);

    StorageRequirements overflow_requirements{1U, 1U};
    CHECK(paint_storage_requirements(
              7U, std::numeric_limits<std::size_t>::max(), 2U,
              overflow_requirements).code == StatusCode::out_of_memory);
    CHECK(overflow_requirements.persistent == 0U);
    CHECK(overflow_requirements.scratch == 0U);
    CHECK(paint_storage_requirements(
              std::numeric_limits<std::size_t>::max(), 1U, 1U,
              overflow_requirements).code == StatusCode::out_of_memory);

    std::puts("PASS: exact typed paint plan, alignment and overflow");
}
