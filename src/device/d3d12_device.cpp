// SPDX-License-Identifier: MIT
#include "device.hpp"

#include "driver/d3d12/schedule_context.hpp"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace parallel_mater::device {
namespace {

[[nodiscard]] bool supports_schedule(IDXGIAdapter1 *adapter) noexcept {
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&device))))
        return false;
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_ROOT_SIGNATURE root{
        D3D_ROOT_SIGNATURE_VERSION_1_1};
    return SUCCEEDED(device->CheckFeatureSupport(
               D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) &&
        model.HighestShaderModel >= D3D_SHADER_MODEL_6_6 &&
        SUCCEEDED(device->CheckFeatureSupport(
            D3D12_FEATURE_ROOT_SIGNATURE, &root, sizeof(root))) &&
        root.HighestVersion >= D3D_ROOT_SIGNATURE_VERSION_1_1;
}

[[nodiscard]] std::vector<ComPtr<IDXGIAdapter1>> adapters() {
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(0U, IID_PPV_ARGS(&factory)))) return {};
    std::vector<ComPtr<IDXGIAdapter1>> result;
    for (UINT index = 0U;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(adapter->GetDesc1(&description)) ||
            (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0U ||
            !supports_schedule(adapter.Get()))
            continue;
        result.push_back(std::move(adapter));
    }
    return result;
}

class D3D12Services final : public Services {
  public:
    D3D12Services(int ordinal, ComPtr<IDXGIAdapter1> adapter)
        : ordinal_(ordinal), adapter_(std::move(adapter)) {}

    [[nodiscard]] PmResult initialize() noexcept {
        PmResult result = context_.initialize(adapter_.Get());
        if (result != PM_SUCCESS) return result;
        cudaEvent_t event{};
        const cudaError_t created = cudaEventCreateWithFlags(
            &event, cudaEventDisableTiming);
        if (created != cudaSuccess) return from_driver_error(created);
        const cudaError_t recorded = cudaEventRecord(event, &context_);
        if (recorded != cudaSuccess) {
            (void)cudaEventDestroy(event);
            return from_driver_error(recorded);
        }
        epoch_ = to_event(event);
        return PM_SUCCESS;
    }

    ~D3D12Services() override {
        (void)context_.wait_idle();
        destroy_event(epoch_);
    }

    [[nodiscard]] int ordinal() const noexcept override { return ordinal_; }
    [[nodiscard]] void *native_device_handle() const noexcept override {
        return context_.device();
    }
    [[nodiscard]] void *native_queue_handle() const noexcept override {
        return const_cast<d3d12::ScheduleContext *>(&context_);
    }
    [[nodiscard]] void *native_allocation_handle(
        const void *allocation) const noexcept override {
        return context_.native_resource(allocation);
    }
    [[nodiscard]] std::uint64_t native_allocation_usage() const noexcept override {
        return D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    [[nodiscard]] Event epoch() const noexcept override { return epoch_; }

    [[nodiscard]] PmResult create_event(bool enable_timing,
                                        Event &output) noexcept override {
        cudaEvent_t event{};
        const cudaError_t result = cudaEventCreateWithFlags(
            &event, enable_timing ? cudaEventDefault : cudaEventDisableTiming);
        if (result != cudaSuccess) return from_driver_error(result);
        output = to_event(event);
        return PM_SUCCESS;
    }

    void destroy_event(Event event) noexcept override {
        if (event) (void)cudaEventDestroy(to_cuda(event));
    }

    [[nodiscard]] PmResult record_event(Event event) noexcept override {
        return event ? from_driver_error(
                           cudaEventRecord(to_cuda(event), &context_))
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
        return context_.allocate(size, output);
    }

    void free(void *allocation) noexcept override { context_.free(allocation); }

    [[nodiscard]] PmResult activate() noexcept override { return PM_SUCCESS; }

    [[nodiscard]] PmResult copy(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection) noexcept override {
        return context_.copy_host(destination, source, size);
    }

    [[nodiscard]] PmResult copy_and_wait(
        void *destination, const void *source, PmDeviceSize size,
        CopyDirection direction) noexcept override {
        const PmResult copied = copy(destination, source, size, direction);
        if (copied != PM_SUCCESS) return copied;
        return context_.wait_idle();
    }

    [[nodiscard]] PmResult upload(
        void *destination, std::unique_ptr<std::byte[]> source,
        PmDeviceSize size) noexcept override {
        if (!source) return PM_ERROR_INVALID_ARGUMENT;
        return context_.copy_host(destination, source.get(), size);
    }

    [[nodiscard]] PmResult wait_idle() noexcept override {
        return context_.wait_idle();
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
    ComPtr<IDXGIAdapter1> adapter_;
    d3d12::ScheduleContext context_;
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
    std::vector<ComPtr<IDXGIAdapter1>> available;
    try {
        available = adapters();
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (descriptions == nullptr) {
        *count = static_cast<std::uint32_t>(available.size());
        return PM_SUCCESS;
    }
    const std::uint32_t written = std::min(
        *count, static_cast<std::uint32_t>(available.size()));
    for (std::uint32_t index = 0U; index < written; ++index) {
        DXGI_ADAPTER_DESC1 native{};
        if (FAILED(available[index]->GetDesc1(&native)))
            return PM_ERROR_INVALID_STATE;
        Description &output = descriptions[index];
        output.ordinal = static_cast<int>(index);
        output.driver_version = 1U;
        output.vendor_id = native.VendorId;
        output.device_id = native.DeviceId;
        output.total_memory = native.DedicatedVideoMemory;
        char name[128]{};
        const int converted = WideCharToMultiByte(
            CP_UTF8, 0U, native.Description, -1, name,
            static_cast<int>(sizeof(name)), nullptr, nullptr);
        output.name = converted > 0 ? name : "Parallel Mater D3D12";
        std::memcpy(output.uuid, &native.AdapterLuid,
                    std::min(sizeof(output.uuid), sizeof(native.AdapterLuid)));
    }
    *count = written;
    return written == available.size() ? PM_SUCCESS : PM_INCOMPLETE;
}

std::unique_ptr<Services> create_d3d12(
    int ordinal, PmResult &result) noexcept {
    try {
        auto available = adapters();
        if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= available.size()) {
            result = PM_ERROR_INVALID_ARGUMENT;
            return {};
        }
        auto services = std::make_unique<D3D12Services>(
            ordinal, available[static_cast<std::size_t>(ordinal)]);
        result = services->initialize();
        if (result != PM_SUCCESS) return {};
        return services;
    } catch (const std::bad_alloc &) {
        result = PM_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

} // namespace parallel_mater::device
