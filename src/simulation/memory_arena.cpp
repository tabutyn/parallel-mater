// SPDX-License-Identifier: MIT
#include "memory_arena.hpp"

#include <algorithm>
#include <mutex>
#include <vector>

namespace parallel_mater::simulation {
namespace {

struct Range {
    std::uintptr_t begin{};
    std::uintptr_t end{};
};

thread_local MemoryArena *active_arena{};
std::mutex ranges_mutex;
std::vector<Range> ranges;

void register_range(const void *pointer, std::size_t size) {
    if (pointer == nullptr || size == 0U) return;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    std::lock_guard lock(ranges_mutex);
    ranges.push_back({begin, begin + size});
}

void unregister_range(const void *pointer, std::size_t size) {
    if (pointer == nullptr || size == 0U) return;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    const auto end = begin + size;
    std::lock_guard lock(ranges_mutex);
    const auto found = std::find_if(ranges.begin(), ranges.end(),
        [begin, end](const Range &range) {
            return range.begin == begin && range.end == end;
        });
    if (found != ranges.end()) ranges.erase(found);
}

[[nodiscard]] void *bump(std::byte *base, std::size_t size,
                         std::size_t &offset, std::size_t bytes,
                         std::size_t alignment) noexcept {
    if (base == nullptr || bytes == 0U || alignment == 0U ||
        (alignment & (alignment - 1U)) != 0U)
        return nullptr;
    const std::size_t aligned =
        (offset + alignment - 1U) & ~(alignment - 1U);
    if (aligned > size || bytes > size - aligned) return nullptr;
    void *result = base + aligned;
    offset = aligned + bytes;
    return result;
}

} // namespace

MemoryArena::MemoryArena(void *persistent, std::size_t persistent_size,
                         void *scratch, std::size_t scratch_size)
    : persistent_(static_cast<std::byte *>(persistent)),
      persistent_size_(persistent_size), scratch_(static_cast<std::byte *>(scratch)),
      scratch_size_(scratch_size) {
    register_range(persistent_, persistent_size_);
    register_range(scratch_, scratch_size_);
}

MemoryArena::~MemoryArena() {
    unregister_range(scratch_, scratch_size_);
    unregister_range(persistent_, persistent_size_);
}

void *MemoryArena::allocate(std::size_t bytes, std::size_t alignment,
                            MemoryClass memory_class) noexcept {
    return memory_class == MemoryClass::scratch
        ? bump(scratch_, scratch_size_, scratch_offset_, bytes, alignment)
        : bump(persistent_, persistent_size_, persistent_offset_, bytes,
               alignment);
}

std::size_t MemoryArena::persistent_used() const noexcept {
    return persistent_offset_;
}

std::size_t MemoryArena::scratch_used() const noexcept {
    return scratch_offset_;
}

void MemoryArena::reset() noexcept {
    persistent_offset_ = 0U;
    scratch_offset_ = 0U;
}

ScopedMemoryArena::ScopedMemoryArena(MemoryArena *arena) noexcept
    : previous_(active_arena) {
    active_arena = arena;
}

ScopedMemoryArena::~ScopedMemoryArena() { active_arena = previous_; }

void *allocate_bound(std::size_t bytes, std::size_t alignment,
                     MemoryClass memory_class) noexcept {
    return active_arena == nullptr
        ? nullptr : active_arena->allocate(bytes, alignment, memory_class);
}

bool has_active_arena() noexcept { return active_arena != nullptr; }

bool is_bound_pointer(const void *pointer) noexcept {
    if (pointer == nullptr) return false;
    const auto value = reinterpret_cast<std::uintptr_t>(pointer);
    std::lock_guard lock(ranges_mutex);
    return std::any_of(ranges.begin(), ranges.end(),
        [value](const Range &range) {
            return value >= range.begin && value < range.end;
        });
}

} // namespace parallel_mater::simulation
