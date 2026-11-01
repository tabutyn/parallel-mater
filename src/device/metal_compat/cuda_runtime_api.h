// SPDX-License-Identifier: MIT
#pragma once

// CUDA-shaped host vocabulary retained by the shared runtime while the
// Metal adapter supplies the actual queue, resources, and completion fence.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include "device/compute.hpp"

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

struct dim3 {
    unsigned int x{1U};
    unsigned int y{1U};
    unsigned int z{1U};

    constexpr dim3() noexcept = default;
    constexpr dim3(unsigned int x_value, unsigned int y_value = 1U,
                   unsigned int z_value = 1U) noexcept
        : x(x_value), y(y_value), z(z_value) {}
};

enum cudaError_t : int {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidDevice = 10,
    cudaErrorInsufficientDriver = 35,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchFailure = 719,
    cudaErrorNotSupported = 801,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorNoDevice = 100,
};

enum cudaMemcpyKind : int {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaDeviceAttr : int {
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrCooperativeLaunch = 95,
};

inline constexpr unsigned int cudaMemAttachGlobal = 1U;
inline constexpr unsigned int cudaEventDefault = 0U;
inline constexpr unsigned int cudaEventDisableTiming = 2U;

using cudaStream_t = void *;

struct pm_metal_cuda_event {
    cudaStream_t stream{};
    parallel_mater::device::compute::SubmissionPtr submission;
    std::chrono::steady_clock::time_point recorded{};
    bool ready{};
    bool timing{};
};
using cudaEvent_t = pm_metal_cuda_event *;

cudaError_t pmMetalMemcpyAsync(void *destination, const void *source,
                              std::size_t size, cudaMemcpyKind kind,
                              cudaStream_t stream) noexcept;
cudaError_t pmMetalMemsetAsync(void *destination, int value,
                              std::size_t size, cudaStream_t stream) noexcept;
cudaError_t pmMetalStreamSynchronize(cudaStream_t stream) noexcept;
cudaError_t pmMetalRecordEvent(cudaEvent_t event,
                              cudaStream_t stream) noexcept;
cudaError_t pmMetalWaitEvent(cudaEvent_t event) noexcept;
cudaError_t pmMetalQueryEvent(cudaEvent_t event) noexcept;
cudaError_t pmMetalElapsedTime(float *milliseconds, cudaEvent_t begin,
                              cudaEvent_t end) noexcept;
cudaError_t pmMetalPeekError(cudaStream_t stream, bool take) noexcept;

cudaError_t pmMetalAllocate(void **output, std::size_t size) noexcept;
cudaError_t pmMetalFree(void *pointer) noexcept;
inline cudaError_t cudaMalloc(void **output, std::size_t size) noexcept {
    return pmMetalAllocate(output, size);
}

template <typename T>
inline cudaError_t cudaMalloc(T **output, std::size_t size) noexcept {
    return cudaMalloc(reinterpret_cast<void **>(output), size);
}

inline cudaError_t cudaMallocManaged(
    void **output, std::size_t size,
    unsigned int = cudaMemAttachGlobal) noexcept {
    return cudaMalloc(output, size);
}

template <typename T>
inline cudaError_t cudaMallocManaged(
    T **output, std::size_t size,
    unsigned int flags = cudaMemAttachGlobal) noexcept {
    return cudaMallocManaged(reinterpret_cast<void **>(output), size, flags);
}

inline cudaError_t cudaFree(void *value) noexcept {
    return pmMetalFree(value);
}

inline cudaError_t cudaMemcpy(void *destination, const void *source,
                              std::size_t size, cudaMemcpyKind) noexcept {
    return pmMetalMemcpyAsync(destination, source, size, cudaMemcpyDefault, nullptr);
}

inline cudaError_t cudaMemcpyAsync(void *destination, const void *source,
                                   std::size_t size, cudaMemcpyKind kind,
                                   cudaStream_t stream = nullptr) noexcept {
    return stream == nullptr
        ? cudaMemcpy(destination, source, size, kind)
        : pmMetalMemcpyAsync(destination, source, size, kind, stream);
}

inline cudaError_t cudaMemset(void *destination, int value,
                              std::size_t size) noexcept {
    return pmMetalMemsetAsync(destination, value, size, nullptr);
}

inline cudaError_t cudaMemsetAsync(void *destination, int value,
                                   std::size_t size,
                                   cudaStream_t stream = nullptr) noexcept {
    return stream == nullptr ? cudaMemset(destination, value, size)
                             : pmMetalMemsetAsync(destination, value, size,
                                                 stream);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) noexcept {
    return stream == nullptr ? cudaSuccess : pmMetalStreamSynchronize(stream);
}

inline cudaError_t cudaDeviceSynchronize() noexcept { return pmMetalStreamSynchronize(nullptr); }
inline cudaError_t cudaGetDevice(int *device) noexcept {
    if (device == nullptr) return cudaErrorInvalidValue;
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attribute,
                                          int) noexcept {
    if (value == nullptr) return cudaErrorInvalidValue;
    *value = attribute == cudaDevAttrMultiProcessorCount ? 1 : 0;
    return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
                                            unsigned int flags) noexcept {
    if (event == nullptr) return cudaErrorInvalidValue;
    auto *created = new (std::nothrow) pm_metal_cuda_event{};
    if (created == nullptr) return cudaErrorMemoryAllocation;
    created->timing = (flags & cudaEventDisableTiming) == 0U;
    *event = created;
    return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t *event) noexcept {
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) noexcept {
    delete event;
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event,
                                   cudaStream_t stream = nullptr) noexcept {
    if (event == nullptr || stream == nullptr)
        return cudaErrorInvalidResourceHandle;
    return pmMetalRecordEvent(event, stream);
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) noexcept {
    return event == nullptr ? cudaErrorInvalidResourceHandle
                            : pmMetalWaitEvent(event);
}

inline cudaError_t cudaEventQuery(cudaEvent_t event) noexcept {
    return event == nullptr ? cudaErrorInvalidResourceHandle
                            : pmMetalQueryEvent(event);
}

inline cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t begin,
                                        cudaEvent_t end) noexcept {
    return pmMetalElapsedTime(milliseconds, begin, end);
}

inline cudaError_t cudaPeekAtLastError() noexcept {
    return pmMetalPeekError(nullptr, false);
}
inline cudaError_t cudaGetLastError() noexcept {
    return pmMetalPeekError(nullptr, true);
}
