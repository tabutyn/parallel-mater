// SPDX-License-Identifier: MIT
#if !__has_feature(objc_arc)
#error "Metal device services require ARC"
#endif
#include "device.hpp"
#include "driver/metal/schedule_context.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>

namespace parallel_mater::device {
namespace {
class MetalServices final : public Services {
public:
    explicit MetalServices(std::unique_ptr<compute::Device> gpu) : context_(std::move(gpu)) {
        context_.activate();
        if (create_event(false, epoch_) != PM_SUCCESS || record_event(epoch_) != PM_SUCCESS) {
            destroy_event(epoch_);
            throw std::runtime_error("Metal epoch creation failed");
        }
    }
    ~MetalServices() override { (void)wait_idle(); destroy_event(epoch_); }
    int ordinal() const noexcept override { return 0; }
    void *native_device_handle() const noexcept override { return context_.device().native_device(); }
    void describe_native_queue(PmNativeQueueInfo &info) const noexcept override {
        info.queue = static_cast<PmNativeHandle>(reinterpret_cast<std::uintptr_t>(
            context_.device().native_queue()));
        info.familyIndex = 0U;
        info.queueIndex = 0U;
    }
    // Internal schedule token, not an exported native queue handle.
    void *native_queue_handle() const noexcept override { return &context_; }
    void *native_allocation_handle(const void *pointer) const noexcept override {
        try { return context_.native_resource(pointer); } catch (...) { return nullptr; }
    }
    Event epoch() const noexcept override { return epoch_; }
    PmResult activate() noexcept override { context_.activate(); return PM_SUCCESS; }
    PmResult create_event(bool timing, Event &out) noexcept override {
        out = {};
        if (timing) return PM_ERROR_FEATURE_NOT_PRESENT;
        cudaEvent_t event{};
        const auto result = cudaEventCreateWithFlags(&event, timing ? cudaEventDefault : cudaEventDisableTiming);
        out = {reinterpret_cast<std::uintptr_t>(event)};
        return from_driver_error(result);
    }
    static cudaEvent_t event(Event value) noexcept { return reinterpret_cast<cudaEvent_t>(value.value); }
    void destroy_event(Event value) noexcept override { if (value) (void)cudaEventDestroy(event(value)); }
    PmResult record_event(Event value) noexcept override { return from_driver_error(cudaEventRecord(event(value), &context_)); }
    PmResult enqueue_wait(Event value) noexcept override { return wait_event(value); }
    PmResult wait_event(Event value) noexcept override { return from_driver_error(cudaEventSynchronize(event(value))); }
    PmResult query_event(Event value, bool &ready) noexcept override {
        const auto result = cudaEventQuery(event(value));
        ready = result == cudaSuccess;
        return result == cudaErrorNotReady ? PM_SUCCESS : from_driver_error(result);
    }
    PmResult elapsed_time(Event, Event, float &) noexcept override {
        return PM_ERROR_FEATURE_NOT_PRESENT;
    }
    PmResult allocate(PmDeviceSize bytes, bool, void **out) noexcept override {
        if (!out || !bytes || bytes > SIZE_MAX) return PM_ERROR_INVALID_ARGUMENT;
        *out = nullptr;
        return from_driver_error(context_.checked([&] { *out = context_.allocate(bytes); }, false));
    }
    void free(void *pointer) noexcept override { (void)context_.checked([&] { context_.free(pointer); }, false); }
    PmResult copy(void *dst, const void *src, PmDeviceSize bytes, CopyDirection) noexcept override {
        if (bytes > SIZE_MAX) return PM_ERROR_INVALID_ARGUMENT;
        return from_driver_error(context_.checked([&] { context_.copy(dst, src, bytes); }, false));
    }
    PmResult copy_and_wait(void *dst, const void *src, PmDeviceSize bytes, CopyDirection direction) noexcept override {
        const auto result = copy(dst, src, bytes, direction);
        return result == PM_SUCCESS ? wait_idle() : result;
    }
    PmResult upload(void *dst, std::unique_ptr<std::byte[]> source, PmDeviceSize bytes) noexcept override {
        if (!source) return PM_ERROR_INVALID_ARGUMENT;
        return copy(dst, source.get(), bytes, CopyDirection::host_to_device);
    }
    PmResult wait_idle() noexcept override { return from_driver_error(context_.checked([&] { context_.wait(); }, false)); }
    PmResult clear(void *destination, int value, PmDeviceSize bytes) noexcept override {
        if (bytes > SIZE_MAX) return PM_ERROR_INVALID_ARGUMENT;
        return from_driver_error(context_.checked([&] { context_.clear(destination, value, bytes); }, false));
    }
    PmResult last_error(bool take) noexcept override {
        return from_driver_error(context_.error(take));
    }
    PmResult memory_budget(PmDeviceSize &available, PmDeviceSize &total) noexcept override {
        id<MTLDevice> gpu = (__bridge id<MTLDevice>)native_device_handle();
        total = gpu.recommendedMaxWorkingSetSize;
        available = total > gpu.currentAllocatedSize ? total - gpu.currentAllocatedSize : 0;
        return PM_SUCCESS;
    }
private:
    mutable metal::ScheduleContext context_;
    Event epoch_{};
};
}
PmResult from_driver_error(std::int32_t error) noexcept {
    switch (static_cast<cudaError_t>(error)) {
    case cudaSuccess: return PM_SUCCESS;
    case cudaErrorMemoryAllocation: return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    case cudaErrorNotSupported: return PM_ERROR_FEATURE_NOT_PRESENT;
    case cudaErrorInvalidValue: case cudaErrorInvalidResourceHandle: return PM_ERROR_INVALID_ARGUMENT;
    case cudaErrorLaunchFailure: case cudaErrorIllegalAddress: return PM_ERROR_DEVICE_LOST;
    default: return PM_ERROR_INVALID_STATE;
    }
}
PmResult enumerate(std::uint32_t *count, Description *descriptions) noexcept {
    if (!count) return PM_ERROR_INVALID_ARGUMENT;
    try {
        auto gpu = compute::create_metal_device();
        if (!gpu) { *count = 0; return PM_SUCCESS; }
        if (!descriptions) { *count = 1; return PM_SUCCESS; }
        if (*count == 0) return PM_INCOMPLETE;
        id<MTLDevice> native = (__bridge id<MTLDevice>)gpu->native_device();
        auto &out = descriptions[0];
        out = {};
        out.driver_version = 202618U;
        out.vendor_id = 0x106b;
        out.name = gpu->name() + " - Metal GPU preview (qualification incomplete)";
        out.total_memory = native.recommendedMaxWorkingSetSize;
        const auto registry = native.registryID;
        std::memcpy(out.uuid, &registry, sizeof(registry));
        std::memcpy(out.uuid + sizeof(registry), "PMMETAL1", 8);
        *count = 1;
        return PM_SUCCESS;
    } catch (...) { return PM_ERROR_INVALID_STATE; }
}
std::unique_ptr<Services> create_metal(int ordinal, PmResult &result) noexcept {
    if (ordinal != 0) { result = PM_ERROR_INVALID_ARGUMENT; return {}; }
    try {
        auto services = std::make_unique<MetalServices>(compute::create_metal_device());
        result = PM_SUCCESS; return services;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Metal driver creation failed: %s\n", error.what());
        result = PM_ERROR_INVALID_STATE; return {};
    }
}
} // namespace parallel_mater::device
