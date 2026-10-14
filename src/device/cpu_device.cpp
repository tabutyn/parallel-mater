// SPDX-License-Identifier: MIT
#include "cuda_device.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <sys/sysctl.h>

namespace parallel_mater::device {
namespace {

[[nodiscard]] std::uint64_t installed_memory() noexcept {
    std::uint64_t bytes{};
    std::size_t size = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0
        ? bytes : (8ULL << 30U);
}

[[nodiscard]] std::string processor_name() {
    std::array<char, 256> name{};
    std::size_t size = name.size();
    if (sysctlbyname("machdep.cpu.brand_string", name.data(), &size,
                     nullptr, 0) == 0 && name.front() != '\0')
        return name.data();
    return "Apple Silicon";
}

class CpuServices final : public Services {
  public:
    [[nodiscard]] PmResult initialize() noexcept {
        cudaEvent_t epoch{};
        const cudaError_t created = cudaEventCreateWithFlags(
            &epoch, cudaEventDisableTiming);
        if (created != cudaSuccess) return from_driver_error(created);
        const cudaError_t recorded = cudaEventRecord(epoch, nullptr);
        if (recorded != cudaSuccess) {
            (void)cudaEventDestroy(epoch);
            return from_driver_error(recorded);
        }
        epoch_ = to_event(epoch);
        return PM_SUCCESS;
    }

    ~CpuServices() override { destroy_event(epoch_); }

    [[nodiscard]] int ordinal() const noexcept override { return 0; }
    [[nodiscard]] void *native_device_handle() const noexcept override {
        return nullptr;
    }
    [[nodiscard]] void *native_queue_handle() const noexcept override {
        return nullptr;
    }
    [[nodiscard]] Event epoch() const noexcept override { return epoch_; }

    [[nodiscard]] PmResult create_event(bool enable_timing,
                                        Event &output) noexcept override {
        cudaEvent_t event{};
        const cudaError_t error = cudaEventCreateWithFlags(
            &event, enable_timing ? cudaEventDefault : cudaEventDisableTiming);
        if (error != cudaSuccess) return from_driver_error(error);
        output = to_event(event);
        return PM_SUCCESS;
    }

    void destroy_event(Event event) noexcept override {
        if (event) (void)cudaEventDestroy(to_cuda(event));
    }

    [[nodiscard]] PmResult record_event(Event event) noexcept override {
        return event ? from_driver_error(cudaEventRecord(to_cuda(event)))
                     : PM_ERROR_INVALID_ARGUMENT;
    }
    [[nodiscard]] PmResult enqueue_wait(Event event) noexcept override {
        return wait_event(event);
    }
    [[nodiscard]] PmResult wait_event(Event event) noexcept override {
        return event ? from_driver_error(cudaEventSynchronize(to_cuda(event)))
                     : PM_ERROR_INVALID_ARGUMENT;
    }
    [[nodiscard]] PmResult query_event(Event event,
                                       bool &ready) noexcept override {
        if (!event) return PM_ERROR_INVALID_ARGUMENT;
        const cudaError_t result = cudaEventQuery(to_cuda(event));
        ready = result == cudaSuccess;
        return result == cudaSuccess || result == cudaErrorNotReady
            ? PM_SUCCESS : from_driver_error(result);
    }
    [[nodiscard]] PmResult elapsed_time(Event begin, Event end,
                                        float &milliseconds) noexcept override {
        if (!begin || !end) return PM_ERROR_INVALID_ARGUMENT;
        return from_driver_error(cudaEventElapsedTime(
            &milliseconds, to_cuda(begin), to_cuda(end)));
    }

    [[nodiscard]] PmResult allocate(PmDeviceSize size, bool,
                                    void **output) noexcept override {
        if (output == nullptr || size == 0U || size > SIZE_MAX)
            return PM_ERROR_INVALID_ARGUMENT;
        return from_driver_error(cudaMalloc(
            output, static_cast<std::size_t>(size)));
    }
    void free(void *allocation) noexcept override {
        if (allocation != nullptr) (void)cudaFree(allocation);
    }
    [[nodiscard]] PmResult activate() noexcept override { return PM_SUCCESS; }
    [[nodiscard]] PmResult copy(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection) noexcept override {
        if ((destination == nullptr || source == nullptr) && size != 0U)
            return PM_ERROR_INVALID_ARGUMENT;
        if (size > SIZE_MAX) return PM_ERROR_INVALID_ARGUMENT;
        return from_driver_error(cudaMemcpy(
            destination, source, static_cast<std::size_t>(size),
            cudaMemcpyDefault));
    }
    [[nodiscard]] PmResult copy_and_wait(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection direction) noexcept override {
        return copy(destination, source, size, direction);
    }
    [[nodiscard]] PmResult upload(
        void *destination, std::unique_ptr<std::byte[]> source,
        PmDeviceSize size) noexcept override {
        if (!source) return PM_ERROR_INVALID_ARGUMENT;
        return copy(destination, source.get(), size,
                    CopyDirection::host_to_device);
    }
    [[nodiscard]] PmResult wait_idle() noexcept override { return PM_SUCCESS; }
    [[nodiscard]] PmResult memory_budget(PmDeviceSize &available,
                                         PmDeviceSize &total) noexcept override {
        total = installed_memory();
        // The serial driver uses pageable process memory.  Report a conservative
        // budget rather than pretending all unified memory is available.
        available = total / 2U;
        return PM_SUCCESS;
    }

  private:
    [[nodiscard]] static Event to_event(cudaEvent_t event) noexcept {
        return {reinterpret_cast<std::uintptr_t>(event)};
    }
    [[nodiscard]] static cudaEvent_t to_cuda(Event event) noexcept {
        return reinterpret_cast<cudaEvent_t>(event.value);
    }

    Event epoch_{};
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
    if (descriptions == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    Description &output = descriptions[0];
    output.ordinal = 0;
    output.driver_version = 202618U;
    output.vendor_id = 0x106bU;
    output.device_id = 0U;
    output.name = "Parallel Mater Slang CPU (" + processor_name() + ")";
    output.total_memory = installed_memory();
    constexpr std::array<std::uint8_t, 16> uuid{
        0x50, 0x4d, 0x53, 0x4c, 0x41, 0x4e, 0x47, 0x43,
        0x50, 0x55, 0x4d, 0x41, 0x43, 0x4f, 0x53, 0x01};
    std::copy(uuid.begin(), uuid.end(), output.uuid);
    *count = 1U;
    return PM_SUCCESS;
}

std::unique_ptr<Services> create_cpu(int ordinal, PmResult &result) noexcept {
    if (ordinal != 0) {
        result = PM_ERROR_INVALID_ARGUMENT;
        return {};
    }
    try {
        auto services = std::make_unique<CpuServices>();
        result = services->initialize();
        if (result != PM_SUCCESS) return {};
        return services;
    } catch (const std::bad_alloc &) {
        result = PM_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

} // namespace parallel_mater::device
