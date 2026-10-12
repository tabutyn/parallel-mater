// SPDX-License-Identifier: MIT
#include "device.hpp"

#include <algorithm>
#include <cstring>
#include <new>

namespace parallel_mater::device {
namespace {

class CudaServices final : public Services {
  public:
    explicit CudaServices(int ordinal) noexcept : ordinal_(ordinal) {}

    [[nodiscard]] PmResult initialize() noexcept {
        cudaError_t error = cudaSetDevice(ordinal_);
        if (error != cudaSuccess) return from_cuda(error);
        error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (error != cudaSuccess) return from_cuda(error);
        error = cudaEventCreateWithFlags(&epoch_, cudaEventDefault);
        if (error != cudaSuccess) return from_cuda(error);
        error = cudaEventRecord(epoch_, stream_);
        if (error != cudaSuccess) return from_cuda(error);
        return PM_SUCCESS;
    }

    ~CudaServices() override {
        if (stream_ != nullptr) (void)cudaStreamSynchronize(stream_);
        if (epoch_ != nullptr) (void)cudaEventDestroy(epoch_);
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }

    [[nodiscard]] int ordinal() const noexcept override { return ordinal_; }
    [[nodiscard]] cudaStream_t stream() const noexcept override { return stream_; }
    [[nodiscard]] cudaEvent_t epoch() const noexcept override { return epoch_; }

    [[nodiscard]] PmResult allocate(PmDeviceSize size, bool host_visible,
                                    void **output) noexcept override {
        if (output == nullptr || size == 0U || size > SIZE_MAX)
            return PM_ERROR_INVALID_ARGUMENT;
        *output = nullptr;
        const cudaError_t error = host_visible
            ? cudaMallocManaged(output, static_cast<std::size_t>(size),
                                cudaMemAttachGlobal)
            : cudaMalloc(output, static_cast<std::size_t>(size));
        return from_cuda(error);
    }

    void free(void *allocation) noexcept override {
        if (allocation != nullptr) (void)cudaFree(allocation);
    }

    [[nodiscard]] PmResult wait_idle() noexcept override {
        return from_cuda(cudaStreamSynchronize(stream_));
    }

    [[nodiscard]] PmResult memory_budget(PmDeviceSize &available,
                                         PmDeviceSize &total) noexcept override {
        std::size_t free_bytes = 0U;
        std::size_t total_bytes = 0U;
        const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (error != cudaSuccess) return from_cuda(error);
        available = free_bytes;
        total = total_bytes;
        return PM_SUCCESS;
    }

  private:
    int ordinal_{};
    cudaStream_t stream_{};
    cudaEvent_t epoch_{};
};

} // namespace

PmResult from_cuda(cudaError_t error) noexcept {
    switch (error) {
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
