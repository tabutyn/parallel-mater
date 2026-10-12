// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace parallel_mater::simulation {

enum class MemoryClass : std::uint8_t { persistent, scratch };

class MemoryArena {
  public:
    MemoryArena(void *persistent, std::size_t persistent_size, void *scratch,
                std::size_t scratch_size);
    ~MemoryArena();
    MemoryArena(const MemoryArena &) = delete;
    MemoryArena &operator=(const MemoryArena &) = delete;

    [[nodiscard]] void *allocate(std::size_t bytes, std::size_t alignment,
                                 MemoryClass memory_class) noexcept;
    [[nodiscard]] std::size_t persistent_used() const noexcept;
    [[nodiscard]] std::size_t scratch_used() const noexcept;
    void reset() noexcept;

  private:
    std::byte *persistent_{};
    std::size_t persistent_size_{};
    std::size_t persistent_offset_{};
    std::byte *scratch_{};
    std::size_t scratch_size_{};
    std::size_t scratch_offset_{};
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

} // namespace parallel_mater::simulation
