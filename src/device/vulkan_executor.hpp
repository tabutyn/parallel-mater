// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace parallel_mater::device {

// Native execution service for a future Slang/Vulkan driver.  It deliberately
// knows nothing about physics domains or schedules: callers provide SPIR-V,
// resource layouts, descriptor sets, and dispatch dimensions.
class VulkanExecutor final {
  public:
    VulkanExecutor(const VulkanExecutor &) = delete;
    VulkanExecutor &operator=(const VulkanExecutor &) = delete;
    ~VulkanExecutor();

    [[nodiscard]] static PmResult create(
        VkDevice device, VkQueue queue, std::uint32_t queue_family,
        std::unique_ptr<VulkanExecutor> &output) noexcept;

    [[nodiscard]] PmResult create_compute_pipeline(
        std::span<const std::uint32_t> spirv, const char *entry_point,
        VkPipelineLayout layout, VkPipeline &pipeline) noexcept;
    void destroy_pipeline(VkPipeline pipeline) noexcept;

    [[nodiscard]] PmResult allocate_command_buffer(
        VkCommandBuffer &command_buffer) noexcept;
    void free_command_buffer(VkCommandBuffer command_buffer) noexcept;
    [[nodiscard]] PmResult reset_command_buffer(
        VkCommandBuffer command_buffer) noexcept;
    [[nodiscard]] PmResult begin_command_buffer(
        VkCommandBuffer command_buffer) noexcept;
    [[nodiscard]] PmResult end_command_buffer(
        VkCommandBuffer command_buffer) noexcept;

    void cmd_dispatch(VkCommandBuffer command_buffer, VkPipeline pipeline,
                      VkPipelineLayout layout,
                      std::span<const VkDescriptorSet> descriptor_sets,
                      std::span<const std::byte> push_constants,
                      std::uint32_t x, std::uint32_t y,
                      std::uint32_t z) noexcept;
    void cmd_compute_barrier(VkCommandBuffer command_buffer) noexcept;

    [[nodiscard]] PmResult queue_submit(
        VkCommandBuffer command_buffer, VkSemaphore wait_semaphore,
        std::uint64_t wait_value, VkSemaphore signal_semaphore,
        std::uint64_t signal_value, VkFence fence) noexcept;
    [[nodiscard]] PmResult wait_idle() noexcept;

  private:
    VulkanExecutor(VkDevice device, VkQueue queue,
                   VkCommandPool command_pool) noexcept;

    VkDevice device_{};
    VkQueue queue_{};
    VkCommandPool command_pool_{};
};

} // namespace parallel_mater::device
