// SPDX-License-Identifier: MIT
#include "cuda_device.hpp"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <new>

namespace parallel_mater::device {
namespace {

[[nodiscard]] constexpr cudaMemcpyKind cuda_copy_kind(
    CopyDirection direction) noexcept {
    switch (direction) {
    case CopyDirection::automatic: return cudaMemcpyDefault;
    case CopyDirection::host_to_device: return cudaMemcpyHostToDevice;
    case CopyDirection::device_to_host: return cudaMemcpyDeviceToHost;
    case CopyDirection::device_to_device: return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyDefault;
}

class CudaServices final : public Services {
  public:
    explicit CudaServices(int ordinal) noexcept : ordinal_(ordinal) {}

    [[nodiscard]] PmResult initialize() noexcept {
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        cudaError_t error{};
        error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (error != cudaSuccess) return from_cuda(error);
        error = cudaEventCreateWithFlags(&epoch_, cudaEventDefault);
        if (error != cudaSuccess) return from_cuda(error);
        error = cudaEventRecord(epoch_, stream_);
        if (error != cudaSuccess) return from_cuda(error);
        if (cuCtxGetCurrent(&context_) != CUDA_SUCCESS || context_ == nullptr)
            return PM_ERROR_INVALID_STATE;
        error = cudaDeviceGetAttribute(&concurrent_managed_access_,
            cudaDevAttrConcurrentManagedAccess, ordinal_);
        if (error != cudaSuccess) return from_cuda(error);
        return PM_SUCCESS;
    }

    ~CudaServices() override {
        (void)cudaSetDevice(ordinal_);
        if (stream_ != nullptr) (void)cudaStreamSynchronize(stream_);
        if (epoch_ != nullptr) (void)cudaEventDestroy(epoch_);
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }

    [[nodiscard]] int ordinal() const noexcept override { return ordinal_; }
    [[nodiscard]] void *native_device_handle() const noexcept override {
        return context_;
    }
    [[nodiscard]] void *native_queue_handle() const noexcept override {
        return stream_;
    }
    void describe_native_queue(PmNativeQueueInfo &info) const noexcept override {
        info.queue = static_cast<PmNativeHandle>(
            reinterpret_cast<std::uintptr_t>(stream_));
        info.familyIndex = 0U;
        info.queueIndex = 0U;
    }

    [[nodiscard]] Event epoch() const noexcept override {
        return to_event(epoch_);
    }

    [[nodiscard]] PmResult create_event(bool enable_timing,
                                        Event &output) noexcept override {
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        cudaEvent_t event{};
        const cudaError_t error = cudaEventCreateWithFlags(
            &event, enable_timing ? cudaEventDefault : cudaEventDisableTiming);
        if (error != cudaSuccess) return from_cuda(error);
        output = to_event(event);
        return PM_SUCCESS;
    }

    void destroy_event(Event event) noexcept override {
        if (event && activate() == PM_SUCCESS)
            (void)cudaEventDestroy(to_cuda(event));
    }

    [[nodiscard]] PmResult record_event(Event event) noexcept override {
        if (!event) return PM_ERROR_INVALID_ARGUMENT;
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        return from_cuda(cudaEventRecord(to_cuda(event), stream_));
    }

    [[nodiscard]] PmResult enqueue_wait(Event event) noexcept override {
        if (!event) return PM_ERROR_INVALID_ARGUMENT;
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        return from_cuda(cudaStreamWaitEvent(stream_, to_cuda(event), 0U));
    }

    [[nodiscard]] PmResult wait_event(Event event) noexcept override {
        if (!event) return PM_ERROR_INVALID_ARGUMENT;
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        return from_cuda(cudaEventSynchronize(to_cuda(event)));
    }

    [[nodiscard]] PmResult query_event(Event event,
                                       bool &ready) noexcept override {
        if (!event) return PM_ERROR_INVALID_ARGUMENT;
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        const cudaError_t status = cudaEventQuery(to_cuda(event));
        if (status == cudaSuccess) {
            ready = true;
            return PM_SUCCESS;
        }
        if (status == cudaErrorNotReady) {
            ready = false;
            return PM_SUCCESS;
        }
        ready = false;
        return from_cuda(status);
    }

    [[nodiscard]] PmResult elapsed_time(Event begin, Event end,
                                        float &milliseconds) noexcept override {
        if (!begin || !end) return PM_ERROR_INVALID_ARGUMENT;
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        return from_cuda(cudaEventElapsedTime(
            &milliseconds, to_cuda(begin), to_cuda(end)));
    }

    [[nodiscard]] PmResult allocate(PmDeviceSize size, bool host_visible,
                                    void **output) noexcept override {
        if (output == nullptr || size == 0U || size > SIZE_MAX)
            return PM_ERROR_INVALID_ARGUMENT;
        *output = nullptr;
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        const cudaError_t error = host_visible
            ? cudaMallocManaged(output, static_cast<std::size_t>(size),
                                cudaMemAttachGlobal)
            : cudaMalloc(output, static_cast<std::size_t>(size));
        return from_cuda(error);
    }

    void free(void *allocation) noexcept override {
        if (allocation != nullptr && activate() == PM_SUCCESS)
            (void)cudaFree(allocation);
    }

    [[nodiscard]] PmResult activate() noexcept override {
        return from_cuda(cudaSetDevice(ordinal_));
    }

    [[nodiscard]] PmResult copy(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection direction) noexcept override {
        if (destination == nullptr || source == nullptr || size > SIZE_MAX)
            return PM_ERROR_INVALID_ARGUMENT;
        if (size == 0U) return PM_SUCCESS;
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        return from_cuda(cudaMemcpyAsync(destination, source,
            static_cast<std::size_t>(size), cuda_copy_kind(direction), stream_));
    }

    [[nodiscard]] PmResult copy_and_wait(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection direction) noexcept override {
        const PmResult result = copy(destination, source, size, direction);
        if (result != PM_SUCCESS) return result;
        return wait_idle();
    }

    [[nodiscard]] PmResult upload(
        void *destination, std::unique_ptr<std::byte[]> source,
        PmDeviceSize size) noexcept override {
        if (!source) return PM_ERROR_INVALID_ARGUMENT;
        if (size == 0U) return PM_SUCCESS;
        const PmResult result = copy(destination, source.get(), size,
                                     CopyDirection::host_to_device);
        if (result != PM_SUCCESS) return result;

        std::byte *owned = source.release();
        const cudaError_t callback_result = cudaLaunchHostFunc(
            stream_,
            [](void *storage) { delete[] static_cast<std::byte *>(storage); },
            owned);
        if (callback_result == cudaSuccess) return PM_SUCCESS;

        // The copy was accepted but ownership could not be attached to the
        // stream. Drain it before releasing the source buffer.
        (void)cudaStreamSynchronize(stream_);
        delete[] owned;
        return from_cuda(callback_result);
    }

    [[nodiscard]] PmResult wait_idle() noexcept override {
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        return from_cuda(cudaStreamSynchronize(stream_));
    }

    [[nodiscard]] PmResult clear(void *destination, int value,
                                 PmDeviceSize size) noexcept override {
        return from_cuda(cudaMemsetAsync(destination, value,
            static_cast<std::size_t>(size), stream_));
    }
    [[nodiscard]] PmResult read_host(void *, PmDeviceSize size) noexcept override {
        if (!size) return PM_SUCCESS;
        const auto current = activate();
        if (current != PM_SUCCESS) return current;
        // Windows/basic Unified Memory forbids any CPU managed access while
        // the device is active, including unrelated queued buffer copies.
        // Concurrent-UM systems need only the owning queue dependency.
        return from_cuda(concurrent_managed_access_ ? cudaStreamSynchronize(stream_)
                                                   : cudaDeviceSynchronize());
    }
    [[nodiscard]] PmResult map_host(void *pointer, PmDeviceSize size) noexcept override {
        return read_host(pointer, size);
    }
    [[nodiscard]] PmResult last_error(bool take) noexcept override {
        return from_cuda(take ? cudaGetLastError() : cudaPeekAtLastError());
    }
    [[nodiscard]] PmResult scheduling_units(std::uint32_t &units) noexcept override {
        int cooperative{}, multiprocessors{};
        auto result = cudaDeviceGetAttribute(&cooperative,
            cudaDevAttrCooperativeLaunch, ordinal_);
        if (result == cudaSuccess)
            result = cudaDeviceGetAttribute(&multiprocessors,
                cudaDevAttrMultiProcessorCount, ordinal_);
        units = cooperative ? static_cast<std::uint32_t>(std::max(1, multiprocessors)) : 1U;
        return from_cuda(result);
    }

    [[nodiscard]] PmResult memory_budget(PmDeviceSize &available,
                                         PmDeviceSize &total) noexcept override {
        const PmResult current = activate();
        if (current != PM_SUCCESS) return current;
        std::size_t free_bytes = 0U;
        std::size_t total_bytes = 0U;
        const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (error != cudaSuccess) return from_cuda(error);
        available = free_bytes;
        total = total_bytes;
        return PM_SUCCESS;
    }

  private:
    [[nodiscard]] static Event to_event(cudaEvent_t event) noexcept {
        return {reinterpret_cast<std::uintptr_t>(event)};
    }

    [[nodiscard]] static cudaEvent_t to_cuda(Event event) noexcept {
        return reinterpret_cast<cudaEvent_t>(event.value);
    }

    int ordinal_{};
    int concurrent_managed_access_{};
    CUcontext context_{};
    cudaStream_t stream_{};
    cudaEvent_t epoch_{};
};

} // namespace

PmResult from_driver_error(std::int32_t error) noexcept {
    switch (static_cast<cudaError_t>(error)) {
    case cudaSuccess: return PM_SUCCESS;
    case cudaErrorMemoryAllocation: return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidResourceHandle: return PM_ERROR_INVALID_ARGUMENT;
    case cudaErrorNotSupported: return PM_ERROR_FEATURE_NOT_PRESENT;
    case cudaErrorLaunchFailure:
    case cudaErrorIllegalAddress: return PM_ERROR_DEVICE_LOST;
    default: return PM_ERROR_INVALID_STATE;
    }
}

PmResult enumerate(std::uint32_t *count, Description *descriptions) noexcept {
    if (count == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        *count = 0U;
        // Instance creation and physical-device enumeration remain valid on a
        // machine with no usable execution target. This matches the command
        // API model: target availability is discovered after instance setup.
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
            error == cudaErrorInitializationError ||
            error == cudaErrorSystemDriverMismatch) {
            (void)cudaGetLastError();
            return PM_SUCCESS;
        }
        return from_cuda(error);
    }
    const std::uint32_t available = static_cast<std::uint32_t>(device_count);
    if (descriptions == nullptr) {
        *count = available;
        return PM_SUCCESS;
    }
    const std::uint32_t written = std::min(*count, available);
    int driver = 0;
    (void)cudaDriverGetVersion(&driver);
    for (std::uint32_t index = 0U; index < written; ++index) {
        cudaDeviceProp properties{};
        error = cudaGetDeviceProperties(&properties, static_cast<int>(index));
        if (error != cudaSuccess) return from_cuda(error);
        Description &output = descriptions[index];
        output.ordinal = static_cast<int>(index);
        output.driver_version = static_cast<std::uint32_t>(driver);
        output.device_id = static_cast<std::uint32_t>(index);
        output.name = properties.name;
        output.total_memory = static_cast<PmDeviceSize>(properties.totalGlobalMem);
        static_assert(sizeof properties.uuid.bytes == sizeof output.uuid);
        std::memcpy(output.uuid, properties.uuid.bytes, sizeof output.uuid);
    }
    *count = written;
    return written == available ? PM_SUCCESS : PM_INCOMPLETE;
}

std::unique_ptr<Services> create_cuda(int ordinal, PmResult &result) noexcept {
    try {
        auto services = std::make_unique<CudaServices>(ordinal);
        result = services->initialize();
        if (result != PM_SUCCESS) return {};
        return services;
    } catch (const std::bad_alloc &) {
        result = PM_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

} // namespace parallel_mater::device
