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
                     "exact geometry plan allocation failed"};
    }
};

template<class T>
bool aligned(const T *pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0U;
}
} // namespace

int main() {
    StorageRequirements requirements{};
    CHECK(geometry_storage_requirements(8U, 36U, requirements));
    CHECK(requirements.persistent != 0U);
    CHECK(requirements.scratch == 0U);

    // Offset the arena base deliberately. Planner padding must make the exact
    // queried byte count sufficient without relying on allocator alignment.
    std::vector<std::byte> backing(requirements.persistent + 1U);
    MemoryArena arena(backing.data() + 1U, requirements.persistent, nullptr, 0U);
    GeometryStoragePlan allocated{};
    ArenaAllocator allocator{arena};
    CHECK(visit_geometry_storage(allocated, 8U, 36U, allocator));
    CHECK(allocated.vertex_capacity == 8U);
    CHECK(allocated.index_capacity == 36U);
    CHECK(allocated.bvh_node_capacity == 23U);
    CHECK(allocated.bvh_leaf_capacity == 12U);
    CHECK(allocated.solid_plane_capacity == 24U);
    CHECK(allocated.shell_normal_capacity == 12U);
    CHECK(aligned(allocated.vertices));
    CHECK(aligned(allocated.indices));
    CHECK(aligned(allocated.bvh_nodes));
    CHECK(aligned(allocated.bvh_leaves));
    CHECK(aligned(allocated.solid_planes));
    CHECK(aligned(allocated.shell_normals));
    const std::size_t materialized_bytes =
        8U * sizeof(Vec3) + 36U * sizeof(std::uint32_t) +
        23U * sizeof(BvhNode) + 12U * sizeof(std::uint32_t) +
        24U * sizeof(CollisionPlane) + 12U * sizeof(Vec3);
    CHECK(arena.persistent_used() == materialized_bytes);
    CHECK(arena.persistent_used() <= requirements.persistent);
    CHECK(arena.scratch_used() == 0U);

    GeometryStoragePlan invalid{};
    StoragePlanner invalid_planner;
    CHECK(visit_geometry_storage(invalid, 2U, 3U, invalid_planner).code ==
          StatusCode::invalid_argument);
    CHECK(visit_geometry_storage(invalid, 3U, 4U, invalid_planner).code ==
          StatusCode::invalid_argument);

    GeometryStoragePlan overflow{};
    StoragePlanner overflow_planner;
    CHECK(visit_geometry_storage(
              overflow, std::numeric_limits<std::size_t>::max(), 3U,
              overflow_planner).code == StatusCode::out_of_memory);
    overflow = {};
    overflow_planner = {};
    CHECK(visit_geometry_storage(
              overflow, 3U, std::numeric_limits<std::size_t>::max(),
              overflow_planner).code == StatusCode::out_of_memory);

    std::puts("PASS: exact typed geometry plan, alignment, topology and overflow");
}
