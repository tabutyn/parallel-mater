// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace parallel_mater::device {

struct Event {
    std::uintptr_t value{};

    [[nodiscard]] explicit operator bool() const noexcept { return value != 0U; }
    friend bool operator==(Event left, Event right) noexcept {
        return left.value == right.value;
    }
};

// Opaque queue token shared with the simulation scheduler. Only a concrete
// device adapter may interpret the native handle.
struct ExecutionQueue {
    void *native_handle{};

    [[nodiscard]] explicit operator bool() const noexcept {
        return native_handle != nullptr;
    }
};

enum class CopyDirection : std::uint8_t {
    automatic,
    host_to_device,
    device_to_host,
    device_to_device,
};

[[nodiscard]] PmResult from_driver_error(std::int32_t error) noexcept;

struct Description {
    int ordinal{};
    std::uint32_t driver_version{};
    std::uint32_t vendor_id{0x10deU};
    std::uint32_t device_id{};
    std::uint8_t uuid[16]{};
    std::string name;
    PmDeviceSize total_memory{};
};

class Services {
  public:
    virtual ~Services() = default;

    [[nodiscard]] virtual int ordinal() const noexcept = 0;
    [[nodiscard]] virtual void *native_device_handle() const noexcept = 0;
    [[nodiscard]] virtual void *native_queue_handle() const noexcept = 0;
    [[nodiscard]] virtual void *native_allocation_handle(
        const void *allocation) const noexcept {
        return const_cast<void *>(allocation);
    }
    [[nodiscard]] virtual std::uint64_t native_allocation_usage() const noexcept {
        return 0U;
    }
    [[nodiscard]] virtual Event epoch() const noexcept = 0;
    [[nodiscard]] virtual PmResult create_event(
        bool enable_timing, Event &output) noexcept = 0;
    virtual void destroy_event(Event event) noexcept = 0;
    [[nodiscard]] virtual PmResult record_event(Event event) noexcept = 0;
    [[nodiscard]] virtual PmResult enqueue_wait(Event event) noexcept = 0;
    [[nodiscard]] virtual PmResult wait_event(Event event) noexcept = 0;
    [[nodiscard]] virtual PmResult query_event(
        Event event, bool &ready) noexcept = 0;
    [[nodiscard]] virtual PmResult elapsed_time(
        Event begin, Event end, float &milliseconds) noexcept = 0;
    [[nodiscard]] virtual PmResult allocate(PmDeviceSize size,
                                            bool host_visible,
                                            void **output) noexcept = 0;
    virtual void free(void *allocation) noexcept = 0;
    [[nodiscard]] virtual PmResult activate() noexcept = 0;
    [[nodiscard]] virtual PmResult copy(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection direction = CopyDirection::automatic) noexcept = 0;
    [[nodiscard]] virtual PmResult copy_and_wait(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection direction = CopyDirection::automatic) noexcept = 0;
    // Takes ownership of `source` and releases it after the queued upload.
    [[nodiscard]] virtual PmResult upload(
        void *destination, std::unique_ptr<std::byte[]> source,
        PmDeviceSize size) noexcept = 0;
    [[nodiscard]] virtual PmResult wait_idle() noexcept = 0;
    [[nodiscard]] virtual PmResult memory_budget(PmDeviceSize &available,
                                                 PmDeviceSize &total) noexcept = 0;
};

[[nodiscard]] inline ExecutionQueue execution_queue(
    Services &services) noexcept {
    return {services.native_queue_handle()};
}

[[nodiscard]] PmResult enumerate(std::uint32_t *count,
                                 Description *descriptions) noexcept;
[[nodiscard]] std::unique_ptr<Services> create_cuda(
    int ordinal, PmResult &result) noexcept;
[[nodiscard]] std::unique_ptr<Services> create_cpu(
    int ordinal, PmResult &result) noexcept;
[[nodiscard]] std::unique_ptr<Services> create_metal(
    int ordinal, PmResult &result) noexcept;
[[nodiscard]] std::unique_ptr<Services> create_d3d12(
    int ordinal, PmResult &result) noexcept;

} // namespace parallel_mater::device
