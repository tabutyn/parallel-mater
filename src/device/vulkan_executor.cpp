// SPDX-License-Identifier: MIT
#include "vulkan_executor.hpp"

#include <new>

namespace parallel_mater::device {
namespace {

PmResult from_vulkan(VkResult result) noexcept {
    switch (result) {
    case VK_SUCCESS: return PM_SUCCESS;
    case VK_NOT_READY: return PM_NOT_READY;
    case VK_TIMEOUT: return PM_TIMEOUT;
    case VK_INCOMPLETE: return PM_INCOMPLETE;
    case VK_ERROR_OUT_OF_HOST_MEMORY: return PM_ERROR_OUT_OF_HOST_MEMORY;
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    case VK_ERROR_DEVICE_LOST: return PM_ERROR_DEVICE_LOST;
    case VK_ERROR_FEATURE_NOT_PRESENT: return PM_ERROR_FEATURE_NOT_PRESENT;
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return PM_ERROR_FORMAT_NOT_SUPPORTED;
    default: return PM_ERROR_INVALID_STATE;
    }
}

} // namespace

VulkanExecutor::VulkanExecutor(VkDevice device, VkQueue queue,
                               VkCommandPool command_pool) noexcept
    : device_(device), queue_(queue), command_pool_(command_pool) {}

VulkanExecutor::~VulkanExecutor() {
    if (command_pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, command_pool_, nullptr);
}

PmResult VulkanExecutor::create(VkDevice device, VkQueue queue,
                                std::uint32_t queue_family,
                                std::unique_ptr<VulkanExecutor> &output) noexcept {
    output.reset();
    if (device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE)
        return PM_ERROR_INVALID_ARGUMENT;
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    info.queueFamilyIndex = queue_family;
    VkCommandPool pool{};
    const VkResult result = vkCreateCommandPool(device, &info, nullptr, &pool);
    if (result != VK_SUCCESS) return from_vulkan(result);
    try {
        output.reset(new VulkanExecutor(device, queue, pool));
        return PM_SUCCESS;
    } catch (const std::bad_alloc &) {
        vkDestroyCommandPool(device, pool, nullptr);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
}

PmResult VulkanExecutor::create_compute_pipeline(
    std::span<const std::uint32_t> spirv, const char *entry_point,
    VkPipelineLayout layout, VkPipeline &pipeline) noexcept {
    pipeline = VK_NULL_HANDLE;
    if (spirv.empty() || entry_point == nullptr || *entry_point == '\0' ||
        layout == VK_NULL_HANDLE)
        return PM_ERROR_INVALID_ARGUMENT;
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size_bytes();
    module_info.pCode = spirv.data();
    VkShaderModule module{};
    VkResult result = vkCreateShaderModule(device_, &module_info, nullptr, &module);
    if (result != VK_SUCCESS) return from_vulkan(result);
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = entry_point;
    pipeline_info.layout = layout;
    result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1U,
                                      &pipeline_info, nullptr, &pipeline);
    vkDestroyShaderModule(device_, module, nullptr);
    return from_vulkan(result);
}

void VulkanExecutor::destroy_pipeline(VkPipeline pipeline) noexcept {
    if (pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pipeline, nullptr);
}

PmResult VulkanExecutor::allocate_command_buffer(
    VkCommandBuffer &command_buffer) noexcept {
    command_buffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = command_pool_;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1U;
    return from_vulkan(vkAllocateCommandBuffers(device_, &info, &command_buffer));
}

void VulkanExecutor::free_command_buffer(VkCommandBuffer command_buffer) noexcept {
    if (command_buffer != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device_, command_pool_, 1U, &command_buffer);
}

PmResult VulkanExecutor::reset_command_buffer(
    VkCommandBuffer command_buffer) noexcept {
    if (command_buffer == VK_NULL_HANDLE) return PM_ERROR_INVALID_ARGUMENT;
    return from_vulkan(vkResetCommandBuffer(command_buffer, 0U));
}

PmResult VulkanExecutor::begin_command_buffer(
    VkCommandBuffer command_buffer) noexcept {
    if (command_buffer == VK_NULL_HANDLE) return PM_ERROR_INVALID_ARGUMENT;
    VkCommandBufferBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return from_vulkan(vkBeginCommandBuffer(command_buffer, &info));
}

PmResult VulkanExecutor::end_command_buffer(
    VkCommandBuffer command_buffer) noexcept {
    return command_buffer == VK_NULL_HANDLE
        ? PM_ERROR_INVALID_ARGUMENT
        : from_vulkan(vkEndCommandBuffer(command_buffer));
}

void VulkanExecutor::cmd_dispatch(
    VkCommandBuffer command_buffer, VkPipeline pipeline,
    VkPipelineLayout layout, std::span<const VkDescriptorSet> descriptor_sets,
    std::span<const std::byte> push_constants, std::uint32_t x,
    std::uint32_t y, std::uint32_t z) noexcept {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (!descriptor_sets.empty())
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout, 0U,
                                static_cast<std::uint32_t>(descriptor_sets.size()),
                                descriptor_sets.data(), 0U, nullptr);
    if (!push_constants.empty())
        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0U, static_cast<std::uint32_t>(push_constants.size()),
                           push_constants.data());
    vkCmdDispatch(command_buffer, x, y, z);
}

void VulkanExecutor::cmd_compute_barrier(
    VkCommandBuffer command_buffer) noexcept {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &barrier,
                         0U, nullptr, 0U, nullptr);
}

PmResult VulkanExecutor::queue_submit(
    VkCommandBuffer command_buffer, VkSemaphore wait_semaphore,
    std::uint64_t wait_value, VkSemaphore signal_semaphore,
    std::uint64_t signal_value, VkFence fence) noexcept {
    if (command_buffer == VK_NULL_HANDLE) return PM_ERROR_INVALID_ARGUMENT;
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkTimelineSemaphoreSubmitInfo timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.waitSemaphoreValueCount =
        wait_semaphore == VK_NULL_HANDLE ? 0U : 1U;
    timeline.pWaitSemaphoreValues = &wait_value;
    timeline.signalSemaphoreValueCount =
        signal_semaphore == VK_NULL_HANDLE ? 0U : 1U;
    timeline.pSignalSemaphoreValues = &signal_value;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &timeline;
    submit.waitSemaphoreCount = timeline.waitSemaphoreValueCount;
    submit.pWaitSemaphores = &wait_semaphore;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command_buffer;
    submit.signalSemaphoreCount = timeline.signalSemaphoreValueCount;
    submit.pSignalSemaphores = &signal_semaphore;
    return from_vulkan(vkQueueSubmit(queue_, 1U, &submit, fence));
}

PmResult VulkanExecutor::wait_idle() noexcept {
    return from_vulkan(vkQueueWaitIdle(queue_));
}

} // namespace parallel_mater::device
