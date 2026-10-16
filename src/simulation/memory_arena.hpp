// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace parallel_mater::device { class Services; }

namespace parallel_mater::simulation {

enum class MemoryClass : std::uint8_t { persistent, scratch };

class MemoryArena {
  public:
    MemoryArena(void *persistent, std::size_t persistent_size, void *scratch,
                std::size_t scratch_size, device::Services *services = nullptr);
    ~MemoryArena();
    MemoryArena(const MemoryArena &) = delete;
    MemoryArena &operator=(const MemoryArena &) = delete;

    [[nodiscard]] void *allocate(std::size_t bytes, std::size_t alignment,
                                 MemoryClass memory_class) noexcept;
    // Retire device accesses first. Only exact allocation addresses are accepted.
    [[nodiscard]] bool release(const void *pointer) noexcept;
    [[nodiscard]] bool contains(const void *pointer) const noexcept;
    [[nodiscard]] std::size_t persistent_used() const noexcept;
    [[nodiscard]] std::size_t scratch_used() const noexcept;
    void reset() noexcept;

  private:
    struct Allocation { std::size_t offset{}, size{}; };
    struct Pool {
        std::byte *base{};
        std::size_t size{}, used{};
        std::vector<Allocation> allocations;
    };
    device::Services *services_{};
    mutable std::mutex mutex_;
    Pool persistent_;
    Pool scratch_;
};

class ScopedMemoryArena {
  public:
    explicit ScopedMemoryArena(MemoryArena *arena) noexcept;
    ~ScopedMemoryArena();
    ScopedMemoryArena(const ScopedMemoryArena &) = delete;
    ScopedMemoryArena &operator=(const ScopedMemoryArena &) = delete;

  private:
    MemoryArena *previous_{};
};

[[nodiscard]] void *allocate_bound(std::size_t bytes, std::size_t alignment,
                                   MemoryClass memory_class) noexcept;
[[nodiscard]] bool has_active_arena() noexcept;
[[nodiscard]] bool is_bound_pointer(const void *pointer) noexcept;
[[nodiscard]] bool release_bound(const void *pointer) noexcept;

} // namespace parallel_mater::simulation
