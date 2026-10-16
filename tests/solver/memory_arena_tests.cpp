// SPDX-License-Identifier: MIT
#include "simulation/memory_arena.hpp"
#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); } } while(false)
using namespace parallel_mater::simulation;
int main() {
    alignas(256) std::array<std::byte, 1024> persistent{}, scratch{};
    MemoryArena arena(persistent.data(), persistent.size(), scratch.data(), scratch.size());
    auto *left = arena.allocate(256, 256, MemoryClass::persistent);
    auto *middle = arena.allocate(256, 256, MemoryClass::persistent);
    auto *right = arena.allocate(512, 256, MemoryClass::persistent);
    CHECK(left && middle && right);
    std::memset(left, 0x12, 256); std::memset(right, 0x34, 512);
    CHECK(!arena.allocate(1, 1, MemoryClass::persistent));
    CHECK(!arena.release(static_cast<std::byte *>(middle) + 1));
    for (unsigned i = 0; i < 4096; ++i) {
        CHECK(release_bound(middle));
        middle = arena.allocate(256, 256, MemoryClass::persistent);
        CHECK(middle && arena.persistent_used() == 1024);
        CHECK(static_cast<unsigned char *>(left)[0] == 0x12);
        CHECK(static_cast<unsigned char *>(right)[511] == 0x34);
    }
    CHECK(arena.release(left) && arena.release(middle));
    CHECK(!arena.release(middle));
    auto *merged = arena.allocate(512, 256, MemoryClass::persistent);
    CHECK(merged == left); // Adjacent free gaps coalesce.
    CHECK(arena.release(merged) && arena.release(right));
    CHECK(arena.persistent_used() == 0);
    CHECK(!arena.allocate(std::numeric_limits<std::size_t>::max(), 16, MemoryClass::persistent));
    CHECK(!arena.allocate(8, 3, MemoryClass::persistent));
    CHECK(arena.persistent_used() == 0);
    { ScopedMemoryArena scope(&arena);
      auto *temporary = allocate_bound(1024, 1, MemoryClass::scratch);
      CHECK(temporary && arena.scratch_used() == 1024);
      CHECK(release_bound(temporary) && arena.scratch_used() == 0); }
    CHECK(!has_active_arena());
    CHECK(!release_bound(reinterpret_cast<void *>(1)));
    // Absolute alignment, even when the bound address is not aligned.
    MemoryArena unaligned(persistent.data() + 1, 1023, nullptr, 0);
    auto *aligned = unaligned.allocate(256, 256, MemoryClass::persistent);
    CHECK(aligned && reinterpret_cast<std::uintptr_t>(aligned) % 256 == 0);
    CHECK(unaligned.release(aligned));
    std::puts("PASS: arena reclamation, coalescing, live-data isolation and failure bounds");
}
