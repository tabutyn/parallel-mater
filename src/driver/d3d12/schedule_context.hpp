// SPDX-License-Identifier: MIT
#pragma once

#include "slang_command.hpp"

#include <parallel_mater/parallel_mater.h>

#include "device/device.hpp"
#include <chrono>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace parallel_mater::d3d12 {

struct FenceEvent {
    void *stream{};
    std::uint64_t fence_value{};
    std::chrono::steady_clock::time_point recorded{};
    bool ready{};
    bool timing{};
};

class ScheduleContext {
  public:
    ScheduleContext();
    ~ScheduleContext();
    ScheduleContext(const ScheduleContext &) = delete;
    ScheduleContext &operator=(const ScheduleContext &) = delete;

    [[nodiscard]] PmResult initialize(IDXGIAdapter1 *adapter) noexcept;
    [[nodiscard]] PmResult allocate(PmDeviceSize size, void **host) noexcept;
    void free(void *host) noexcept;

    [[nodiscard]] PmResult copy_host(void *destination, const void *source,
                                     PmDeviceSize size) noexcept;
    [[nodiscard]] PmResult copy(void *destination, const void *source,
        PmDeviceSize size, device::CopyDirection direction) noexcept;
    [[nodiscard]] PmResult read_host(void *pointer, PmDeviceSize size) noexcept;
    [[nodiscard]] PmResult write_host(void *pointer, PmDeviceSize size) noexcept;
    [[nodiscard]] PmResult map_host(void *pointer, PmDeviceSize size) noexcept;
    void unmap_host(void *allocation) noexcept;
    void prepare_host_access() noexcept;
    [[nodiscard]] PmResult wait_idle() noexcept;
    [[nodiscard]] PmResult memory_budget(PmDeviceSize &available,
                                         PmDeviceSize &total) noexcept;

    [[nodiscard]] ID3D12Device *device() const noexcept;
    [[nodiscard]] ID3D12CommandQueue *queue() const noexcept;
    [[nodiscard]] ID3D12Resource *native_resource(
        const void *host) const noexcept;

    [[nodiscard]] GraphicsBufferAddress address(
        const void *pointer) noexcept;
    [[nodiscard]] GraphicsBufferAddress translated_meshes(
        const void *meshes, const void *parameters,
        std::uint32_t body_count,
        std::uint32_t mesh_capacity = 0U,
        const void *paint_fields = nullptr,
        std::uint32_t paint_field_capacity = 0U) noexcept;
    [[nodiscard]] GraphicsBufferAddress translated_paint_fields(
        const void *fields, std::uint32_t capacity) noexcept;
    [[nodiscard]] GraphicsBufferAddress translated_paint_rules(
        const void *rules, std::uint32_t capacity) noexcept;

    [[nodiscard]] PmResult dispatch(
        std::string_view module, std::string_view entry,
        device::DispatchSize grid, device::DispatchSize block,
        std::span<const std::byte> constants) noexcept;
    [[nodiscard]] PmResult copy_device(
        void *destination, const void *source, std::size_t size) noexcept;
    [[nodiscard]] PmResult clear_device(
        void *destination, int value, std::size_t size) noexcept;
    [[nodiscard]] PmResult synchronize() noexcept;
    [[nodiscard]] PmResult record(FenceEvent *event) noexcept;
    [[nodiscard]] PmResult wait(FenceEvent *event) noexcept;
    [[nodiscard]] PmResult query(FenceEvent *event) noexcept;
    [[nodiscard]] PmResult error(bool take) noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] ScheduleContext *context(void *stream) noexcept;

} // namespace parallel_mater::d3d12
