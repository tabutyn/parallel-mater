// SPDX-License-Identifier: MIT
#include "device.hpp"

#include "driver/vulkan/schedule_context.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace parallel_mater::device {
namespace {

class VulkanServices final : public Services {
  public:
    explicit VulkanServices(int ordinal) : ordinal_(ordinal) {}

    [[nodiscard]] PmResult initialize() noexcept {
        PmResult result = context_.initialize(ordinal_);
        if (result != PM_SUCCESS)
            return result;
        cudaEvent_t event{};
        const cudaError_t created = cudaEventCreateWithFlags(&event, cudaEventDefault);
        if (created != cudaSuccess)
            return from_driver_error(created);
        const cudaError_t recorded = cudaEventRecord(event, &context_);
        if (recorded != cudaSuccess) {
            (void)cudaEventDestroy(event);
            return from_driver_error(recorded);
        }
        epoch_ = to_event(event);
        return PM_SUCCESS;
    }

    ~VulkanServices() override {
        (void)context_.synchronize();
        destroy_event(epoch_);
    }

    [[nodiscard]] int ordinal() const noexcept override { return ordinal_; }
    [[nodiscard]] void *native_device_handle() const noexcept override {
        return context_.device();
    }
    void describe_native_device(PmNativeDeviceInfo &info) const noexcept override {
        info.instance = reinterpret_cast<PmNativeHandle>(context_.instance());
        info.physicalDevice = reinterpret_cast<PmNativeHandle>(context_.physical());
    }
    [[nodiscard]] void *native_queue_handle() const noexcept override {
        return const_cast<vulkan::ScheduleContext *>(&context_);
    }
    [[nodiscard]] void *
    native_allocation_handle(const void *allocation) const noexcept override {
        return reinterpret_cast<void *>(context_.buffer(allocation));
    }
    [[nodiscard]] std::uint64_t native_allocation_usage() const noexcept override {
        return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    [[nodiscard]] Event epoch() const noexcept override { return epoch_; }

    [[nodiscard]] PmResult create_event(bool enable_timing,
                                        Event &output) noexcept override {
        cudaEvent_t event{};
        const cudaError_t result = cudaEventCreateWithFlags(
            &event, enable_timing ? cudaEventDefault : cudaEventDisableTiming);
        if (result != cudaSuccess)
            return from_driver_error(result);
        output = to_event(event);
        return PM_SUCCESS;
    }

    void destroy_event(Event event) noexcept override {
        if (event)
            (void)cudaEventDestroy(to_cuda(event));
    }

    [[nodiscard]] PmResult record_event(Event event) noexcept override {
        return event ? from_driver_error(cudaEventRecord(to_cuda(event), &context_))
                     : PM_ERROR_INVALID_ARGUMENT;
    }

    [[nodiscard]] PmResult enqueue_wait(Event event) noexcept override {
        return wait_event(event);
    }

    [[nodiscard]] PmResult wait_event(Event event) noexcept override {
        return event ? from_driver_error(cudaEventSynchronize(to_cuda(event)))
                     : PM_ERROR_INVALID_ARGUMENT;
    }

    [[nodiscard]] PmResult query_event(Event event, bool &ready) noexcept override {
        if (!event)
            return PM_ERROR_INVALID_ARGUMENT;
        const cudaError_t result = cudaEventQuery(to_cuda(event));
        ready = result == cudaSuccess;
        return result == cudaSuccess || result == cudaErrorNotReady
                   ? PM_SUCCESS
                   : from_driver_error(result);
    }

    [[nodiscard]] PmResult elapsed_time(Event begin, Event end,
                                        float &milliseconds) noexcept override {
        if (!begin || !end)
            return PM_ERROR_INVALID_ARGUMENT;
        return from_driver_error(
            cudaEventElapsedTime(&milliseconds, to_cuda(begin), to_cuda(end)));
    }

    [[nodiscard]] PmResult allocate(PmDeviceSize size, bool,
                                    void **output) noexcept override {
        return context_.allocate(size, output);
    }

    void free(void *allocation) noexcept override { context_.free(allocation); }

    [[nodiscard]] PmResult activate() noexcept override {
        context_.activate();
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult copy(void *destination, const void *source, PmDeviceSize size,
                                CopyDirection) noexcept override {
        return from_driver_error(
            context_.copy(destination, source, size, cudaMemcpyDefault));
    }

    [[nodiscard]] PmResult copy_and_wait(void *destination, const void *source,
                                         PmDeviceSize size,
                                         CopyDirection direction) noexcept override {
        const PmResult copied = copy(destination, source, size, direction);
        if (copied != PM_SUCCESS)
            return copied;
        return from_driver_error(context_.synchronize());
    }

    [[nodiscard]] PmResult upload(void *destination, std::unique_ptr<std::byte[]> source,
                                  PmDeviceSize size) noexcept override {
        if (!source)
            return PM_ERROR_INVALID_ARGUMENT;
        return from_driver_error(
            context_.copy(destination, source.get(), size, cudaMemcpyHostToDevice));
    }

    [[nodiscard]] PmResult wait_idle() noexcept override {
        return from_driver_error(context_.synchronize());
    }

    [[nodiscard]] PmResult clear(void *destination, int value,
                                 PmDeviceSize size) noexcept override {
        return from_driver_error(context_.clear(destination, value,
            static_cast<std::size_t>(size)));
    }
    [[nodiscard]] PmResult last_error(bool take) noexcept override {
        return from_driver_error(context_.error(take));
    }

    [[nodiscard]] PmResult memory_budget(PmDeviceSize &available,
                                         PmDeviceSize &total) noexcept override {
        return context_.memory_budget(available, total);
    }

  private:
    [[nodiscard]] static Event to_event(cudaEvent_t event) noexcept {
        return {reinterpret_cast<std::uintptr_t>(event)};
    }
    [[nodiscard]] static cudaEvent_t to_cuda(Event event) noexcept {
        return reinterpret_cast<cudaEvent_t>(event.value);
    }

    int ordinal_{};
    vulkan::ScheduleContext context_;
    Event epoch_{};
};

} // namespace

PmResult from_driver_error(std::int32_t error) noexcept {
    switch (static_cast<cudaError_t>(error)) {
    case cudaSuccess:
        return PM_SUCCESS;
    case cudaErrorMemoryAllocation:
        return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidResourceHandle:
        return PM_ERROR_INVALID_ARGUMENT;
    case cudaErrorNotSupported:
        return PM_ERROR_FEATURE_NOT_PRESENT;
    case cudaErrorLaunchFailure:
    case cudaErrorIllegalAddress:
        return PM_ERROR_DEVICE_LOST;
    default:
        return PM_ERROR_INVALID_STATE;
    }
}

PmResult enumerate(std::uint32_t *count, Description *descriptions) noexcept {
    if (count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    try {
        auto available = vulkan::enumerate_devices();
        if (!descriptions) {
            *count = static_cast<std::uint32_t>(available.size());
            return PM_SUCCESS;
        }
        auto written = std::min(*count, static_cast<std::uint32_t>(available.size()));
        std::copy_n(available.begin(), written, descriptions);
        *count = written;
        return written == available.size() ? PM_SUCCESS : PM_INCOMPLETE;
    } catch (...) {
        return PM_ERROR_INVALID_STATE;
    }
}
std::unique_ptr<Services> create_vulkan(int ordinal, PmResult &result) noexcept {
    try {
        auto services = std::make_unique<VulkanServices>(ordinal);
        result = services->initialize();
        if (result != PM_SUCCESS)
            return {};
        return services;
    } catch (...) {
        result = PM_ERROR_INVALID_STATE;
        return {};
    }
}

} // namespace parallel_mater::device
