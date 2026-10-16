// SPDX-License-Identifier: MIT
#pragma once

#include "slang_command.hpp"

#include <parallel_mater/parallel_mater.h>

#include <cuda_runtime_api.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace parallel_mater::d3d12 {

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
    [[nodiscard]] PmResult wait_idle() noexcept;
    [[nodiscard]] PmResult memory_budget(PmDeviceSize &available,
                                         PmDeviceSize &total) noexcept;

    [[nodiscard]] ID3D12Device *device() const noexcept;
    [[nodiscard]] ID3D12CommandQueue *queue() const noexcept;
    [[nodiscard]] ID3D12Resource *native_resource(
        const void *host) const noexcept;

    [[nodiscard]] GraphicsBufferAddress address(
        const void *pointer) noexcept;
    [[nodiscard]] GraphicsBufferAddress widened_flags(
        const std::uint8_t *pointer, std::uint32_t count) noexcept;
    [[nodiscard]] GraphicsBufferAddress translated_meshes(
        const void *meshes, const void *parameters,
        std::uint32_t body_count,
        std::uint32_t mesh_capacity = 0U) noexcept;

    [[nodiscard]] cudaError_t dispatch(
        std::string_view module, std::string_view entry,
        dim3 grid, dim3 block,
        std::span<const std::byte> constants) noexcept;
    [[nodiscard]] cudaError_t copy_device(
        void *destination, const void *source, std::size_t size) noexcept;
    [[nodiscard]] cudaError_t clear_device(
        void *destination, int value, std::size_t size) noexcept;
    [[nodiscard]] cudaError_t synchronize() noexcept;
    [[nodiscard]] cudaError_t record(cudaEvent_t event) noexcept;
    [[nodiscard]] cudaError_t wait(cudaEvent_t event) noexcept;
    [[nodiscard]] cudaError_t query(cudaEvent_t event) noexcept;
    [[nodiscard]] cudaError_t error(bool take) noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] inline ScheduleContext *context(cudaStream_t stream) noexcept {
    return static_cast<ScheduleContext *>(stream);
}

} // namespace parallel_mater::d3d12
