// SPDX-License-Identifier: MIT
#pragma once

// Type-erased native ownership used at the common engine handoff boundary.
// Borrowed objects deliberately cannot carry a release callback.
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace parallel_mater::device::compute {

enum class NativeApi : std::uint8_t {
    none,
    cuda,
    metal,
    d3d12,
    vulkan,
    mock,
};

enum class NativeObjectKind : std::uint8_t {
    instance,
    physical_device,
    device,
    queue,
    buffer,
    completion,
};

enum class NativeOwnership : std::uint8_t { borrowed, owned };

class NativeObject final {
public:
    using Releaser = void (*)(void *context, std::uintptr_t handle) noexcept;

    NativeObject() = default;
    ~NativeObject() { reset(); }
    NativeObject(const NativeObject &) = delete;
    NativeObject &operator=(const NativeObject &) = delete;

    NativeObject(NativeObject &&other) noexcept { move_from(other); }
    NativeObject &operator=(NativeObject &&other) noexcept {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }

    [[nodiscard]] static NativeObject borrowed(NativeApi api, NativeObjectKind kind,
                                                std::uintptr_t handle) {
        if (api == NativeApi::none || !handle)
            throw std::invalid_argument("invalid borrowed native object");
        return NativeObject(api, kind, NativeOwnership::borrowed, handle, nullptr, nullptr);
    }

    [[nodiscard]] static NativeObject owned(NativeApi api, NativeObjectKind kind,
                                             std::uintptr_t handle,
                                             void *context, Releaser releaser) {
        if (api == NativeApi::none || !handle || !releaser)
            throw std::invalid_argument("owned native object requires handle and releaser");
        return NativeObject(api, kind, NativeOwnership::owned, handle, context, releaser);
    }

    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != 0; }
    [[nodiscard]] NativeApi api() const noexcept { return api_; }
    [[nodiscard]] NativeObjectKind kind() const noexcept { return kind_; }
    [[nodiscard]] NativeOwnership ownership() const noexcept { return ownership_; }
    [[nodiscard]] std::uintptr_t handle() const noexcept { return handle_; }

    void reset() noexcept {
        if (handle_ && ownership_ == NativeOwnership::owned) releaser_(context_, handle_);
        handle_ = 0;
        context_ = nullptr;
        releaser_ = nullptr;
    }

private:
    NativeObject(NativeApi api, NativeObjectKind kind, NativeOwnership ownership,
                 std::uintptr_t handle, void *context, Releaser releaser) noexcept
        : api_(api), kind_(kind), ownership_(ownership), handle_(handle), context_(context),
          releaser_(releaser) {}

    void move_from(NativeObject &other) noexcept {
        api_ = other.api_;
        kind_ = other.kind_;
        ownership_ = other.ownership_;
        handle_ = std::exchange(other.handle_, 0);
        context_ = std::exchange(other.context_, nullptr);
        releaser_ = std::exchange(other.releaser_, nullptr);
    }

    NativeApi api_{NativeApi::none};
    NativeObjectKind kind_{NativeObjectKind::device};
    NativeOwnership ownership_{NativeOwnership::borrowed};
    std::uintptr_t handle_{};
    void *context_{};
    Releaser releaser_{};
};

} // namespace parallel_mater::device::compute
