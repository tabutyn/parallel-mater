// SPDX-License-Identifier: MIT
#include "memory_arena.hpp"
#include "device/device.hpp"

#include <algorithm>
#include <limits>

namespace parallel_mater::simulation {
namespace {
thread_local MemoryArena *active_arena{};
std::mutex arenas_mutex;
std::vector<MemoryArena *> arenas;

bool in_range(const void *pointer, const void *base, std::size_t size) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto begin = reinterpret_cast<std::uintptr_t>(base);
    return pointer && base && address >= begin && address - begin < size;
}
} // namespace

MemoryArena::MemoryArena(void *persistent, std::size_t persistent_size,
                         void *scratch, std::size_t scratch_size,
                         device::Services *services)
    : services_(services),
      persistent_{static_cast<std::byte *>(persistent), persistent_size, 0, {}},
      scratch_{static_cast<std::byte *>(scratch), scratch_size, 0, {}} {
    std::lock_guard lock(arenas_mutex);
    arenas.push_back(this);
}

MemoryArena::~MemoryArena() {
    std::lock_guard lock(arenas_mutex);
    std::erase(arenas, this);
}

void *MemoryArena::allocate(std::size_t bytes, std::size_t alignment,
                            MemoryClass memory_class) noexcept {
    if (!bytes || !alignment || (alignment & (alignment - 1))) return nullptr;
    std::lock_guard lock(mutex_);
    auto &pool = memory_class == MemoryClass::scratch ? scratch_ : persistent_;
    if (!pool.base || bytes > pool.size) return nullptr;
    // Sorted live allocations implicitly coalesce every freed gap. Metadata
    // lives on the host: no allocation headers are written into device storage.
    std::size_t offset = 0;
    const auto base = reinterpret_cast<std::uintptr_t>(pool.base);
    for (std::size_t index = 0; index <= pool.allocations.size(); ++index) {
        if (offset > std::numeric_limits<std::uintptr_t>::max() - base)
            return nullptr;
        const auto address = base + offset;
        const auto padding = (alignment - (address & (alignment - 1))) & (alignment - 1);
        const auto end = index == pool.allocations.size()
            ? pool.size : pool.allocations[index].offset;
        if (offset <= end && padding <= end - offset && bytes <= end - offset - padding) {
            const auto aligned = offset + padding;
            try {
                pool.allocations.insert(pool.allocations.begin() + index, {aligned, bytes});
            } catch (...) { return nullptr; }
            void *result = pool.base + aligned;
            if (services_ && services_->write_host(result, bytes) != PM_SUCCESS) {
                pool.allocations.erase(pool.allocations.begin() + index);
                return nullptr;
            }
            pool.used += bytes;
            return result;
        }
        if (index < pool.allocations.size())
            offset = pool.allocations[index].offset + pool.allocations[index].size;
    }
    return nullptr;
}

bool MemoryArena::release(const void *pointer) noexcept {
    if (!pointer) return false;
    std::lock_guard lock(mutex_);
    for (auto *pool : {&persistent_, &scratch_}) {
        if (!in_range(pointer, pool->base, pool->size)) continue;
        const auto offset = reinterpret_cast<std::uintptr_t>(pointer) -
                            reinterpret_cast<std::uintptr_t>(pool->base);
        const auto found = std::lower_bound(pool->allocations.begin(), pool->allocations.end(),
            offset, [](const Allocation &allocation, std::size_t value) {
                return allocation.offset < value;
            });
        if (found == pool->allocations.end() || found->offset != offset) continue;
        pool->used -= found->size;
        pool->allocations.erase(found); // No allocation or failure on release.
        return true;
    }
    return false;
}

bool MemoryArena::contains(const void *pointer) const noexcept {
    return in_range(pointer, persistent_.base, persistent_.size) ||
           in_range(pointer, scratch_.base, scratch_.size);
}

std::size_t MemoryArena::persistent_used() const noexcept {
    std::lock_guard lock(mutex_);
    return persistent_.used;
}
std::size_t MemoryArena::scratch_used() const noexcept {
    std::lock_guard lock(mutex_);
    return scratch_.used;
}
void MemoryArena::reset() noexcept {
    std::lock_guard lock(mutex_);
    persistent_.allocations.clear();
    scratch_.allocations.clear();
    persistent_.used = scratch_.used = 0;
}

ScopedMemoryArena::ScopedMemoryArena(MemoryArena *arena) noexcept
    : previous_(active_arena) { active_arena = arena; }
ScopedMemoryArena::~ScopedMemoryArena() { active_arena = previous_; }
void *allocate_bound(std::size_t bytes, std::size_t alignment,
                     MemoryClass memory_class) noexcept {
    return active_arena ? active_arena->allocate(bytes, alignment, memory_class) : nullptr;
}
bool has_active_arena() noexcept { return active_arena != nullptr; }
bool is_bound_pointer(const void *pointer) noexcept {
    std::lock_guard lock(arenas_mutex);
    return std::any_of(arenas.begin(), arenas.end(),
        [pointer](const auto *arena) { return arena->contains(pointer); });
}
bool release_bound(const void *pointer) noexcept {
    std::lock_guard lock(arenas_mutex);
    if (active_arena && active_arena->release(pointer)) return true;
    for (auto *arena : arenas)
        if (arena != active_arena && arena->release(pointer)) return true;
    return false;
}
} // namespace parallel_mater::simulation
