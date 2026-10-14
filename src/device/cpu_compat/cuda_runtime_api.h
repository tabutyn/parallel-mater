// SPDX-License-Identifier: MIT
#pragma once

// Host-runtime compatibility used only by the serial Slang CPU adapter.  The
// shared scheduler predates the driver split and still spells its queue and
// transfer primitives with CUDA names.  Keeping that vocabulary here avoids a
// second scheduler; no CUDA library or device code is linked into this driver.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

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
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchFailure = 719,
    cudaErrorNotSupported = 801,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorInvalidResourceHandle = 400,
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

struct pm_cpu_cuda_event {
    std::chrono::steady_clock::time_point recorded{};
    bool ready{};
    bool timing{};
};
using cudaEvent_t = pm_cpu_cuda_event *;

inline cudaError_t cudaMalloc(void **output, std::size_t size) noexcept {
    if (output == nullptr || size == 0U) return cudaErrorInvalidValue;
    *output = std::malloc(size);
    return *output == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

template <typename T>
inline cudaError_t cudaMalloc(T **output, std::size_t size) noexcept {
    return cudaMalloc(reinterpret_cast<void **>(output), size);
}

inline cudaError_t cudaMallocManaged(void **output, std::size_t size,
                                     unsigned int = cudaMemAttachGlobal) noexcept {
    return cudaMalloc(output, size);
}

template <typename T>
inline cudaError_t cudaMallocManaged(T **output, std::size_t size,
                                     unsigned int flags = cudaMemAttachGlobal) noexcept {
    return cudaMallocManaged(reinterpret_cast<void **>(output), size, flags);
}

inline cudaError_t cudaFree(void *value) noexcept {
    std::free(value);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void *destination, const void *source,
                              std::size_t size, cudaMemcpyKind) noexcept {
    if ((destination == nullptr || source == nullptr) && size != 0U)
        return cudaErrorInvalidValue;
    if (size != 0U) std::memmove(destination, source, size);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void *destination, const void *source,
                                   std::size_t size, cudaMemcpyKind kind,
                                   cudaStream_t = nullptr) noexcept {
    return cudaMemcpy(destination, source, size, kind);
}

inline cudaError_t cudaMemset(void *destination, int value,
                              std::size_t size) noexcept {
    if (destination == nullptr && size != 0U) return cudaErrorInvalidValue;
    if (size != 0U) std::memset(destination, value, size);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void *destination, int value,
                                   std::size_t size,
                                   cudaStream_t = nullptr) noexcept {
    return cudaMemset(destination, value, size);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t) noexcept {
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() noexcept { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int *device) noexcept {
    if (device == nullptr) return cudaErrorInvalidValue;
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attribute,
                                          int) noexcept {
    if (value == nullptr) return cudaErrorInvalidValue;
    *value = attribute == cudaDevAttrCooperativeLaunch ? 1 : 1;
    return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
                                            unsigned int flags) noexcept {
    if (event == nullptr) return cudaErrorInvalidValue;
    auto *created = new (std::nothrow) pm_cpu_cuda_event{};
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
                                   cudaStream_t = nullptr) noexcept {
    if (event == nullptr) return cudaErrorInvalidResourceHandle;
    event->recorded = std::chrono::steady_clock::now();
    event->ready = true;
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) noexcept {
    return event == nullptr ? cudaErrorInvalidResourceHandle : cudaSuccess;
}

inline cudaError_t cudaEventQuery(cudaEvent_t event) noexcept {
    if (event == nullptr) return cudaErrorInvalidResourceHandle;
    return event->ready ? cudaSuccess : cudaErrorNotReady;
}

inline cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t begin,
                                        cudaEvent_t end) noexcept {
    if (milliseconds == nullptr || begin == nullptr || end == nullptr)
        return cudaErrorInvalidValue;
    *milliseconds = std::chrono::duration<float, std::milli>(
        end->recorded - begin->recorded).count();
    return cudaSuccess;
}

inline cudaError_t cudaPeekAtLastError() noexcept { return cudaSuccess; }
inline cudaError_t cudaGetLastError() noexcept { return cudaSuccess; }
