// SPDX-License-Identifier: MIT
#pragma once
#include "device/device.hpp"
#include <array>
#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

namespace parallel_mater::vulkan {
// Owned compute context. The first qualification path uses coherent mapped
// buffers so the shared scheduler's host preparation remains valid on UMA and
// discrete GPUs. Shader pointers always contain VkDeviceAddress, never host VA.
class ScheduleContext {
  public:
    ScheduleContext();
    ~ScheduleContext();
    ScheduleContext(const ScheduleContext &) = delete;
    ScheduleContext &operator=(const ScheduleContext &) = delete;
    PmResult initialize(int ordinal) noexcept;
    PmResult allocate(std::size_t size, void **output) noexcept;
    void free(void *allocation) noexcept;
    void activate() noexcept;
    std::uint64_t address(const void *pointer) noexcept;
    std::uint64_t translated_meshes(const void *meshes, const void *parameters,
                                    std::uint32_t count,
                                    std::uint32_t capacity = 0) noexcept;
    void dispatch(std::string_view module, std::string_view entry,
                  std::span<const std::uint32_t> shader, dim3 grid, dim3 block,
                  dim3 shader_block, std::span<const std::byte> constants) noexcept;
    cudaError_t copy(void *destination, const void *source, std::size_t size,
                     cudaMemcpyKind kind) noexcept;
    cudaError_t clear(void *destination, int value, std::size_t size) noexcept;
    cudaError_t synchronize() noexcept;
    cudaError_t record(cudaEvent_t event) noexcept;
    cudaError_t wait(cudaEvent_t event) noexcept;
    cudaError_t query(cudaEvent_t event) noexcept;
    cudaError_t error(bool take) noexcept;
    cudaError_t elapsed(cudaEvent_t begin, cudaEvent_t end, float *ms) noexcept;
    PmResult memory_budget(PmDeviceSize &available, PmDeviceSize &total) noexcept;
    int ordinal() const noexcept;
    VkDevice device() const noexcept;
    VkInstance instance() const noexcept;
    VkPhysicalDevice physical() const noexcept;
    VkBuffer buffer(const void *pointer) const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
ScheduleContext *context(cudaStream_t stream = nullptr) noexcept;
std::vector<device::Description> enumerate_devices();
} // namespace parallel_mater::vulkan
