// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <utility>

namespace parallel_mater::simulation {

// Fixed-capacity runtime tables retain only the generation after an owner is
// retired. Resetting the owner releases its host-side auxiliary allocations
// while the slot metadata continues to reject stale handles.
template <class T>
class OwnerSlot {
public:
    OwnerSlot() noexcept = default;
    OwnerSlot(OwnerSlot &&) noexcept = default;
    OwnerSlot &operator=(OwnerSlot &&) noexcept = default;
    OwnerSlot(const OwnerSlot &) = delete;
    OwnerSlot &operator=(const OwnerSlot &) = delete;

    OwnerSlot &operator=(std::unique_ptr<T> owner) noexcept {
        if (owner) owner->generation = generation_;
        owner_ = std::move(owner);
        return *this;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(owner_);
    }
    [[nodiscard]] T *get() noexcept { return owner_.get(); }
    [[nodiscard]] T *get() const noexcept { return owner_.get(); }
    [[nodiscard]] T &operator*() noexcept { return *owner_; }
    [[nodiscard]] T &operator*() const noexcept { return *owner_; }
    [[nodiscard]] T *operator->() noexcept { return owner_.get(); }
    [[nodiscard]] T *operator->() const noexcept { return owner_.get(); }

    [[nodiscard]] std::uint32_t generation() const noexcept {
        return generation_;
    }

    void retire() noexcept {
        if (!owner_) return;
        ++generation_;
        if (generation_ == 0U) generation_ = 1U;
        owner_.reset();
    }

private:
    std::unique_ptr<T> owner_{};
    std::uint32_t generation_{1U};
};

} // namespace parallel_mater::simulation
