// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_driver.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "slang_command.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

struct PmInstance_T;
struct PmDevice_T;

struct PmPhysicalDevice_T {
    PmInstance_T *instance{};
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 description{};
};

struct PmInstance_T {
    ComPtr<IDXGIFactory6> factory;
    std::vector<std::unique_ptr<PmPhysicalDevice_T>> physicals;
};

struct PmQueue_T {
    PmDevice_T *device{};
};

struct PmDevice_T {
    PmPhysicalDevice_T *physical{};
    ComPtr<ID3D12Device> native;
    ComPtr<ID3D12CommandQueue> command_queue;
    parallel_mater::d3d12::SlangCommandPath slang;
    PmQueue_T queue{};
    std::mutex idle_mutex;
    std::mutex pending_mutex;
    std::vector<PmCommandBuffer> pending_commands;
    std::atomic<PmDeviceSize> application_bytes{};
};

struct PmDeviceMemory_T {
    PmDevice_T *device{};
    ComPtr<ID3D12Resource> resource;
    PmDeviceSize size{};
    std::uint32_t type{};
    D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
    void *mapped{};
};

struct PmBuffer_T {
    PmDevice_T *device{};
    PmDeviceSize size{};
    PmFlags usage{};
    std::uint32_t queue_family{};
    PmDeviceMemory memory{};
    PmDeviceSize offset{};
    std::uint32_t descriptor_index{
        parallel_mater::d3d12::invalid_descriptor};
};

enum class CommandState : std::uint8_t {
    initial,
    recording,
    executable,
    pending,
};

struct PmCommandPool_T {
    PmDevice_T *device{};
    PmFlags flags{};
};

struct PmCommandBuffer_T {
    PmDevice_T *device{};
    PmCommandPool pool{};
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    CommandState state{CommandState::initial};
};

struct PmFence_T {
    PmDevice_T *device{};
    ComPtr<ID3D12Fence> native;
    HANDLE event{};
    std::uint64_t target{1U};
    bool submitted{};
    std::vector<PmCommandBuffer> pending;

    ~PmFence_T() {
        if (event != nullptr) CloseHandle(event);
    }
};

namespace {

constexpr std::uint32_t k_driver_version = 1U;
constexpr std::uint32_t k_memory_default = 0U;
constexpr std::uint32_t k_memory_upload = 1U;
constexpr std::uint32_t k_memory_readback = 2U;
constexpr std::uint32_t k_memory_type_count = 3U;
template <typename T, typename... Args>
T *make(Args &&...args) noexcept {
    return new (std::nothrow) T(std::forward<Args>(args)...);
}

template <typename T>
bool valid_device_object(PmDevice device, T *object) noexcept {
    return device != nullptr && object != nullptr && object->device == device;
}

bool valid_structure(PmStructureType actual, PmStructureType expected,
                     const void *next) noexcept {
    return actual == expected && next == nullptr;
}

bool fits(PmDeviceSize offset, PmDeviceSize size,
          PmDeviceSize capacity) noexcept {
    return offset <= capacity && size <= capacity - offset;
}

PmDeviceSize resolved_size(PmDeviceSize offset, PmDeviceSize size,
                           PmDeviceSize capacity) noexcept {
    return size == PM_WHOLE_SIZE && offset <= capacity
        ? capacity - offset : size;
}

D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) noexcept {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    result.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    result.CreationNodeMask = 1U;
    result.VisibleNodeMask = 1U;
    return result;
}

D3D12_RESOURCE_DESC buffer_description(PmDeviceSize size,
                                       D3D12_RESOURCE_FLAGS flags) noexcept {
    D3D12_RESOURCE_DESC result{};
    result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    result.Alignment = 0U;
    result.Width = std::max<PmDeviceSize>(size, 1U);
    result.Height = 1U;
    result.DepthOrArraySize = 1U;
    result.MipLevels = 1U;
    result.Format = DXGI_FORMAT_UNKNOWN;
    result.SampleDesc.Count = 1U;
    result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    result.Flags = flags;
    return result;
}

PmResult result_from_hresult(HRESULT result) noexcept {
    if (result == E_OUTOFMEMORY) return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == DXGI_ERROR_DEVICE_REMOVED ||
        result == DXGI_ERROR_DEVICE_RESET ||
        result == DXGI_ERROR_DEVICE_HUNG)
        return PM_ERROR_DEVICE_LOST;
    return PM_ERROR_INVALID_STATE;
}

std::string utf8(const wchar_t *value) {
    if (value == nullptr || *value == L'\0') return {};
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) return {};
    std::string result(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), required,
                        nullptr, nullptr);
    result.pop_back();
    return result;
}

std::uint32_t buffer_memory_bits(PmFlags usage) noexcept {
    std::uint32_t bits = 1U << k_memory_default;
    const PmFlags non_transfer = usage &
        ~(PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    if (non_transfer == 0U &&
        (usage & PM_BUFFER_TRANSFER_DST_BIT) == 0U)
        bits |= 1U << k_memory_upload;
    if (non_transfer == 0U &&
        (usage & PM_BUFFER_TRANSFER_SRC_BIT) == 0U)
        bits |= 1U << k_memory_readback;
    return bits;
}

void transition(PmCommandBuffer commands, PmDeviceMemory memory,
                D3D12_RESOURCE_STATES desired) noexcept {
    if (memory->type != k_memory_default || memory->state == desired) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = memory->resource.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = memory->state;
    barrier.Transition.StateAfter = desired;
    commands->list->ResourceBarrier(1U, &barrier);
    memory->state = desired;
}

DWORD wait_milliseconds(std::uint64_t nanoseconds) noexcept {
    if (nanoseconds == PM_TIMEOUT_INFINITE) return INFINITE;
    const std::uint64_t milliseconds =
        (nanoseconds + UINT64_C(999999)) / UINT64_C(1000000);
    return static_cast<DWORD>(std::min<std::uint64_t>(
        milliseconds, static_cast<std::uint64_t>(INFINITE - 1U)));
}

PmResult wait_native_fence(ID3D12Fence *fence, std::uint64_t value,
                           HANDLE event, std::uint64_t timeout) noexcept {
    if (fence->GetCompletedValue() >= value) return PM_SUCCESS;
    const HRESULT set = fence->SetEventOnCompletion(value, event);
    if (FAILED(set)) return result_from_hresult(set);
    const DWORD result = WaitForSingleObject(event, wait_milliseconds(timeout));
    if (result == WAIT_OBJECT_0) return PM_SUCCESS;
    if (result == WAIT_TIMEOUT) return PM_TIMEOUT;
    return PM_ERROR_DEVICE_LOST;
}

void complete_commands(PmDevice device,
                       const std::vector<PmCommandBuffer> &completed) noexcept {
    std::lock_guard lock(device->pending_mutex);
    for (PmCommandBuffer commands : completed) {
        commands->state = CommandState::executable;
        device->pending_commands.erase(std::remove(
            device->pending_commands.begin(), device->pending_commands.end(),
            commands), device->pending_commands.end());
    }
}

void complete_all_commands(PmDevice device) noexcept {
    std::lock_guard lock(device->pending_mutex);
    for (PmCommandBuffer commands : device->pending_commands)
        commands->state = CommandState::executable;
    device->pending_commands.clear();
}

PmResult wait_idle(PmDevice device) noexcept {
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    std::lock_guard lock(device->idle_mutex);
    ComPtr<ID3D12Fence> fence;
    HRESULT result = device->native->CreateFence(
        0U, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(result)) return result_from_hresult(result);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    result = device->command_queue->Signal(fence.Get(), 1U);
    const PmResult waited = FAILED(result)
        ? result_from_hresult(result)
        : wait_native_fence(fence.Get(), 1U, event, PM_TIMEOUT_INFINITE);
    CloseHandle(event);
    if (waited == PM_SUCCESS) complete_all_commands(device);
    return waited;
}

bool supports_slang_dxil(ID3D12Device *device) noexcept {
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

struct ClearRigidInputsConstants {
    parallel_mater::d3d12::GraphicsBufferAddress accumulators;
    std::uint32_t accumulator_padding{};
    parallel_mater::d3d12::GraphicsBufferAddress targets;
    std::uint32_t count{};
};
static_assert(sizeof(ClearRigidInputsConstants) == 32U);
static_assert(offsetof(ClearRigidInputsConstants, targets) == 16U);
static_assert(offsetof(ClearRigidInputsConstants, count) == 28U);

PmResult qualify_slang_command_path(PmDevice device) noexcept {
    using parallel_mater::d3d12::DispatchConstants;
    using parallel_mater::d3d12::invalid_descriptor;

    if (device->slang.kernel_count() == 0U ||
        !device->slang.contains("geometry", "pm_clear_rigid_inputs") ||
        !device->slang.contains("cloth", "cloth_update_surface") ||
        !device->slang.contains(
            "geometry_constraints", "solve_avbd_kernel"))
        return PM_ERROR_FEATURE_NOT_PRESENT;

    constexpr std::uint64_t storage_size = 512U;
    const D3D12_HEAP_PROPERTIES default_heap =
        heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_HEAP_PROPERTIES upload_heap =
        heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_HEAP_PROPERTIES readback_heap =
        heap_properties(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC storage_description = buffer_description(
        storage_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const D3D12_RESOURCE_DESC staging_description = buffer_description(
        storage_size, D3D12_RESOURCE_FLAG_NONE);

    ComPtr<ID3D12Resource> storage;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> readback;
    ComPtr<ID3D12Resource> constants;
    HRESULT result = device->native->CreateCommittedResource(
        &default_heap, D3D12_HEAP_FLAG_NONE, &storage_description,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&storage));
    if (FAILED(result)) return result_from_hresult(result);
    result = device->native->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &staging_description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    if (FAILED(result)) return result_from_hresult(result);
    result = device->native->CreateCommittedResource(
        &readback_heap, D3D12_HEAP_FLAG_NONE, &staging_description,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
    if (FAILED(result)) return result_from_hresult(result);
    result = device->native->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &staging_description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&constants));
    if (FAILED(result)) return result_from_hresult(result);

    D3D12_RANGE no_read{0U, 0U};
    void *mapped = nullptr;
    result = upload->Map(0U, &no_read, &mapped);
    if (FAILED(result)) return result_from_hresult(result);
    std::memset(mapped, 0x7f, storage_size);
    upload->Unmap(0U, nullptr);

    struct DescriptorLease {
        parallel_mater::d3d12::SlangCommandPath *path{};
        std::uint32_t value{invalid_descriptor};
        ~DescriptorLease() {
            if (path != nullptr) path->release_descriptor(value);
        }
    };
    DescriptorLease accumulators{&device->slang};
    DescriptorLease targets{&device->slang};
    result = device->slang.create_raw_buffer_view(
        storage.Get(), 0U, 48U, &accumulators.value);
    if (FAILED(result)) return result_from_hresult(result);
    result = device->slang.create_raw_buffer_view(
        storage.Get(), 256U, 56U, &targets.value);
    if (FAILED(result)) return result_from_hresult(result);

    std::byte *constant_data = nullptr;
    result = constants->Map(
        0U, &no_read, reinterpret_cast<void **>(&constant_data));
    if (FAILED(result)) return result_from_hresult(result);
    std::memset(constant_data, 0, storage_size);
    const DispatchConstants dispatch{1U, 1U, 1U, 256U};
    const ClearRigidInputsConstants entry{
        {accumulators.value, 0U, 1U}, 0U,
        {targets.value, 0U, 1U}, 1U};
    std::memcpy(constant_data, &dispatch, sizeof(dispatch));
    std::memcpy(constant_data + 256U, &entry, sizeof(entry));
    constants->Unmap(0U, nullptr);

    ComPtr<ID3D12CommandAllocator> allocator;
    result = device->native->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator));
    if (FAILED(result)) return result_from_hresult(result);
    ComPtr<ID3D12GraphicsCommandList> list;
    result = device->native->CreateCommandList(
        0U, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(), nullptr,
        IID_PPV_ARGS(&list));
    if (FAILED(result)) return result_from_hresult(result);

    list->CopyBufferRegion(
        storage.Get(), 0U, upload.Get(), 0U, storage_size);
    D3D12_RESOURCE_BARRIER to_uav{};
    to_uav.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_uav.Transition.pResource = storage.Get();
    to_uav.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    to_uav.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    to_uav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    list->ResourceBarrier(1U, &to_uav);

    result = device->slang.record_dispatch(
        list.Get(), "geometry", "pm_clear_rigid_inputs",
        constants->GetGPUVirtualAddress(),
        constants->GetGPUVirtualAddress() + 256U, 1U);
    if (FAILED(result)) return result_from_hresult(result);

    D3D12_RESOURCE_BARRIER visibility{};
    visibility.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    visibility.UAV.pResource = storage.Get();
    list->ResourceBarrier(1U, &visibility);
    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = storage.Get();
    to_copy.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    to_copy.Transition.StateBefore =
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1U, &to_copy);
    list->CopyBufferRegion(
        readback.Get(), 0U, storage.Get(), 0U, storage_size);
    result = list->Close();
    if (FAILED(result)) return result_from_hresult(result);

    ID3D12CommandList *lists[]{list.Get()};
    device->command_queue->ExecuteCommandLists(1U, lists);
    ComPtr<ID3D12Fence> fence;
    result = device->native->CreateFence(
        0U, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(result)) return result_from_hresult(result);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    result = device->command_queue->Signal(fence.Get(), 1U);
    const PmResult waited = FAILED(result)
        ? result_from_hresult(result)
        : wait_native_fence(
            fence.Get(), 1U, event, PM_TIMEOUT_INFINITE);
    CloseHandle(event);
    if (waited != PM_SUCCESS) return waited;

    const std::uint8_t *bytes = nullptr;
    D3D12_RANGE read_range{0U, storage_size};
    result = readback->Map(
        0U, &read_range,
        reinterpret_cast<void **>(const_cast<std::uint8_t **>(&bytes)));
    if (FAILED(result)) return result_from_hresult(result);
    const bool accumulators_cleared = std::all_of(
        bytes, bytes + 48U, [](std::uint8_t value) { return value == 0U; });
    const bool target_preserved = std::all_of(
        bytes + 256U, bytes + 308U,
        [](std::uint8_t value) { return value == 0x7fU; });
    const bool target_disabled = std::all_of(
        bytes + 308U, bytes + 312U,
        [](std::uint8_t value) { return value == 0U; });
    D3D12_RANGE no_write{0U, 0U};
    readback->Unmap(0U, &no_write);
    return accumulators_cleared && target_preserved && target_disabled
        ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
}
PmResult drv_enumerate_instance_version(std::uint32_t *version) noexcept {
    if (version == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *version = PM_API_VERSION_0_2;
    return PM_SUCCESS;
}

PmResult drv_enumerate_device_extensions(PmPhysicalDevice physical,
                                         const char *layer,
                                         std::uint32_t *count,
                                         PmExtensionProperties *) noexcept {
    if (physical == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (layer != nullptr) return PM_ERROR_LAYER_NOT_PRESENT;
    *count = 0U;
    return PM_SUCCESS;
}

PmResult drv_create_instance(const PmInstanceCreateInfo *info,
                             const PmAllocationCallbacks *,
                             PmInstance *output) noexcept {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (info == nullptr || !valid_structure(
            info->sType, PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, info->pNext) ||
        info->apiVersion > PM_API_VERSION_0_2)
        return PM_ERROR_INVALID_ARGUMENT;
    auto created = std::unique_ptr<PmInstance_T>(make<PmInstance_T>());
    if (!created) return PM_ERROR_OUT_OF_HOST_MEMORY;
    HRESULT result = CreateDXGIFactory2(0U, IID_PPV_ARGS(&created->factory));
    if (FAILED(result)) return result_from_hresult(result);
    try {
        std::vector<std::unique_ptr<PmPhysicalDevice_T>> software;
        for (UINT index = 0U;; ++index) {
            ComPtr<IDXGIAdapter1> adapter;
            result = created->factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&adapter));
            if (result == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(result)) continue;
            ComPtr<ID3D12Device> probe;
            if (FAILED(D3D12CreateDevice(
                    adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                    IID_PPV_ARGS(&probe))) || !supports_slang_dxil(probe.Get()))
                continue;
            auto physical = std::make_unique<PmPhysicalDevice_T>();
            physical->instance = created.get();
            physical->adapter = std::move(adapter);
            physical->adapter->GetDesc1(&physical->description);
            if ((physical->description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0U)
                software.push_back(std::move(physical));
            else
                created->physicals.push_back(std::move(physical));
        }
        if (created->physicals.empty())
            created->physicals = std::move(software);
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    *output = created.release();
    return PM_SUCCESS;
}

void drv_destroy_instance(PmInstance instance) noexcept { delete instance; }

PmResult drv_enumerate_physical_devices(PmInstance instance,
                                        std::uint32_t *count,
                                        PmPhysicalDevice *devices) noexcept {
    if (instance == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    const std::uint32_t available =
        static_cast<std::uint32_t>(instance->physicals.size());
    if (devices == nullptr) {
        *count = available;
        return PM_SUCCESS;
    }
    const std::uint32_t written = std::min(*count, available);
    for (std::uint32_t index = 0U; index < written; ++index)
        devices[index] = instance->physicals[index].get();
    *count = written;
    return written == available ? PM_SUCCESS : PM_INCOMPLETE;
}

void drv_get_physical_properties(
    PmPhysicalDevice physical, PmPhysicalDeviceProperties *properties) noexcept {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    properties->apiVersion = PM_API_VERSION_0_2;
    properties->driverVersion = k_driver_version;
    properties->vendorId = physical->description.VendorId;
    properties->deviceId = physical->description.DeviceId;
    std::memcpy(properties->deviceUuid, &physical->description.AdapterLuid,
                sizeof(physical->description.AdapterLuid));
    const std::string name = utf8(physical->description.Description);
    const std::size_t copy_size = std::min(
        name.size(), sizeof(properties->deviceName) - 1U);
    std::memcpy(properties->deviceName, name.data(), copy_size);
    properties->deviceName[copy_size] = '\0';
}

PmResult drv_get_queue_families(PmPhysicalDevice physical,
                                std::uint32_t *count,
                                PmQueueFamilyProperties *properties) noexcept {
    if (physical == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (properties == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    properties[0] = {PM_QUEUE_TRANSFER_BIT, 1U, 0U, 0.0F};
    *count = 1U;
    return PM_SUCCESS;
}

void drv_get_features(PmPhysicalDevice physical,
                      PmPhysicalDeviceFeatures *features) noexcept {
    if (features == nullptr) return;
    *features = {};
    if (physical != nullptr) features->memoryBudget = PM_TRUE;
}

bool no_physics_features(PmPhysicalDeviceFeatures features) noexcept {
    features.memoryBudget = PM_FALSE;
    const auto *values = reinterpret_cast<const PmBool32 *>(&features);
    constexpr std::size_t count =
        sizeof(PmPhysicalDeviceFeatures) / sizeof(PmBool32);
    static_assert(sizeof(PmPhysicalDeviceFeatures) % sizeof(PmBool32) == 0U);
    for (std::size_t index = 0U; index < count; ++index) {
        if (values[index] != PM_FALSE) return false;
    }
    return true;
}

PmResult drv_create_device(PmPhysicalDevice physical,
                           const PmDeviceCreateInfo *info,
                           PmDevice *output) noexcept {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (physical == nullptr || info == nullptr || !valid_structure(
            info->sType, PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO, info->pNext) ||
        info->queueCreateInfoCount != 1U || info->pQueueInfos == nullptr ||
        info->pQueueInfos[0].queueFamilyIndex != 0U ||
        info->pQueueInfos[0].queueCount != 1U ||
        !no_physics_features(info->enabledFeatures))
        return PM_ERROR_INVALID_ARGUMENT;
    auto created = std::unique_ptr<PmDevice_T>(make<PmDevice_T>());
    if (!created) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->physical = physical;
    HRESULT result = D3D12CreateDevice(
        physical->adapter.Get(), D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&created->native));
    if (FAILED(result)) return result_from_hresult(result);
    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    result = created->native->CreateCommandQueue(
        &queue_description, IID_PPV_ARGS(&created->command_queue));
    if (FAILED(result)) return result_from_hresult(result);
    created->queue.device = created.get();
    result = created->slang.initialize(created->native.Get());
    if (FAILED(result)) return result_from_hresult(result);
    const PmResult qualification =
        qualify_slang_command_path(created.get());
    if (qualification != PM_SUCCESS) return qualification;
    *output = created.release();
    return PM_SUCCESS;
}

void drv_destroy_device(PmDevice device) noexcept {
    if (device == nullptr) return;
    (void)wait_idle(device);
    delete device;
}

void drv_get_device_queue(PmDevice device, std::uint32_t family,
                          std::uint32_t index, PmQueue *queue) noexcept {
    if (queue == nullptr) return;
    *queue = device != nullptr && family == 0U && index == 0U
        ? &device->queue : nullptr;
}

void drv_get_memory_properties(PmPhysicalDevice physical,
                               PmMemoryProperties *properties) noexcept {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    properties->heapCount = 1U;
    properties->heaps[0] = {
        physical->description.DedicatedVideoMemory,
        PM_MEMORY_DEVICE_LOCAL_BIT};
    properties->typeCount = k_memory_type_count;
    properties->types[k_memory_default] = {PM_MEMORY_DEVICE_LOCAL_BIT, 0U};
    properties->types[k_memory_upload] = {
        PM_MEMORY_HOST_VISIBLE_BIT | PM_MEMORY_HOST_COHERENT_BIT, 0U};
    properties->types[k_memory_readback] = {
        PM_MEMORY_HOST_VISIBLE_BIT | PM_MEMORY_HOST_COHERENT_BIT |
            PM_MEMORY_HOST_CACHED_BIT,
        0U};
}

PmResult drv_allocate_memory(PmDevice device,
                             const PmMemoryAllocateInfo *info,
                             PmDeviceMemory *output) noexcept {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (device == nullptr || info == nullptr || !valid_structure(
            info->sType, PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, info->pNext) ||
        info->allocationSize == 0U ||
        info->memoryTypeIndex >= k_memory_type_count)
        return PM_ERROR_INVALID_ARGUMENT;
    auto created = std::unique_ptr<PmDeviceMemory_T>(make<PmDeviceMemory_T>());
    if (!created) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->size = info->allocationSize;
    created->type = info->memoryTypeIndex;
    D3D12_HEAP_TYPE heap_type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (created->type == k_memory_upload) {
        heap_type = D3D12_HEAP_TYPE_UPLOAD;
        state = D3D12_RESOURCE_STATE_GENERIC_READ;
        flags = D3D12_RESOURCE_FLAG_NONE;
    } else if (created->type == k_memory_readback) {
        heap_type = D3D12_HEAP_TYPE_READBACK;
        state = D3D12_RESOURCE_STATE_COPY_DEST;
        flags = D3D12_RESOURCE_FLAG_NONE;
    }
    const D3D12_HEAP_PROPERTIES heap = heap_properties(heap_type);
    const D3D12_RESOURCE_DESC description =
        buffer_description(created->size, flags);
    const HRESULT result = device->native->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &description, state, nullptr,
        IID_PPV_ARGS(&created->resource));
    if (FAILED(result)) return result_from_hresult(result);
    created->state = state;
    device->application_bytes.fetch_add(created->size,
                                        std::memory_order_relaxed);
    *output = created.release();
    return PM_SUCCESS;
}

void drv_free_memory(PmDevice device, PmDeviceMemory memory) noexcept {
    if (!valid_device_object(device, memory)) return;
    if (memory->mapped != nullptr) memory->resource->Unmap(0U, nullptr);
    device->application_bytes.fetch_sub(memory->size,
                                        std::memory_order_relaxed);
    delete memory;
}

PmResult drv_map_memory(PmDevice device, PmDeviceMemory memory,
                        PmDeviceSize offset, PmDeviceSize size,
                        void **output) noexcept {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    const PmDeviceSize mapped_size = memory == nullptr
        ? 0U : resolved_size(offset, size, memory->size);
    if (!valid_device_object(device, memory) ||
        memory->type == k_memory_default || memory->mapped != nullptr ||
        !fits(offset, mapped_size, memory->size))
        return PM_ERROR_INVALID_ARGUMENT;
    D3D12_RANGE read_range{};
    if (memory->type == k_memory_readback)
        read_range = {static_cast<SIZE_T>(offset),
                      static_cast<SIZE_T>(offset + mapped_size)};
    HRESULT result = memory->resource->Map(0U, &read_range, &memory->mapped);
    if (FAILED(result)) return result_from_hresult(result);
    *output = static_cast<std::byte *>(memory->mapped) + offset;
    return PM_SUCCESS;
}

void drv_unmap_memory(PmDevice device, PmDeviceMemory memory) noexcept {
    if (!valid_device_object(device, memory) || memory->mapped == nullptr) return;
    memory->resource->Unmap(0U, nullptr);
    memory->mapped = nullptr;
}

PmResult drv_mapped_ranges(PmDevice device, std::uint32_t count,
                           const PmMappedMemoryRange *ranges) noexcept {
    if (device == nullptr || (count != 0U && ranges == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto &range = ranges[index];
        if (!valid_device_object(device, range.memory) ||
            range.memory->type == k_memory_default)
            return PM_ERROR_INVALID_ARGUMENT;
        const PmDeviceSize size = resolved_size(
            range.offset, range.size, range.memory->size);
        if (!fits(range.offset, size, range.memory->size))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}

PmResult drv_create_buffer(PmDevice device, const PmBufferCreateInfo *info,
                           PmBuffer *output) noexcept {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (device == nullptr || info == nullptr || !valid_structure(
            info->sType, PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO, info->pNext) ||
        info->size == 0U ||
        (info->initialQueueFamilyIndex != 0U &&
         info->initialQueueFamilyIndex != PM_QUEUE_FAMILY_IGNORED))
        return PM_ERROR_INVALID_ARGUMENT;
    auto created = std::unique_ptr<PmBuffer_T>(make<PmBuffer_T>());
    if (!created) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->size = info->size;
    created->usage = info->usage;
    created->queue_family = 0U;
    *output = created.release();
    return PM_SUCCESS;
}

void drv_destroy_buffer(PmDevice device, PmBuffer buffer) noexcept {
    if (!valid_device_object(device, buffer)) return;
    device->slang.release_descriptor(buffer->descriptor_index);
    delete buffer;
}

void drv_get_buffer_requirements(PmDevice device, PmBuffer buffer,
                                 PmMemoryRequirements *requirements) noexcept {
    if (requirements == nullptr) return;
    *requirements = {};
    if (!valid_device_object(device, buffer)) return;
    requirements->size = buffer->size;
    requirements->alignment = 256U;
    requirements->memoryTypeBits = buffer_memory_bits(buffer->usage);
    requirements->requiresDedicatedAllocation = PM_FALSE;
}

PmResult drv_bind_buffer(PmDevice device, PmBuffer buffer,
                         PmDeviceMemory memory, PmDeviceSize offset) noexcept {
    if (!valid_device_object(device, buffer) ||
        !valid_device_object(device, memory) || buffer->memory != nullptr ||
        offset % 256U != 0U || !fits(offset, buffer->size, memory->size) ||
        (buffer_memory_bits(buffer->usage) & (1U << memory->type)) == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    const PmFlags device_usage = buffer->usage &
        ~(PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    if (memory->type == k_memory_default && device_usage != 0U) {
        const HRESULT result = device->slang.create_raw_buffer_view(
            memory->resource.Get(), offset, buffer->size,
            &buffer->descriptor_index);
        if (FAILED(result)) return result_from_hresult(result);
    }
    buffer->memory = memory;
    buffer->offset = offset;
    return PM_SUCCESS;
}

PmResult drv_create_command_pool(PmDevice device,
                                 const PmCommandPoolCreateInfo *info,
                                 PmCommandPool *output) noexcept {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (device == nullptr || info == nullptr || !valid_structure(
            info->sType, PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            info->pNext) || info->queueFamilyIndex != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto created = std::unique_ptr<PmCommandPool_T>(make<PmCommandPool_T>());
    if (!created) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->flags = info->flags;
    *output = created.release();
    return PM_SUCCESS;
}

void drv_destroy_command_pool(PmDevice device, PmCommandPool pool) noexcept {
    if (valid_device_object(device, pool)) delete pool;
}

PmResult drv_reset_command_pool(PmDevice device, PmCommandPool pool) noexcept {
    return valid_device_object(device, pool)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}

PmResult drv_allocate_command_buffers(PmDevice device, PmCommandPool pool,
                                      std::uint32_t count,
                                      PmCommandBuffer *outputs) noexcept {
    if (!valid_device_object(device, pool) || count == 0U || outputs == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    std::fill_n(outputs, count, nullptr);
    for (std::uint32_t index = 0U; index < count; ++index) {
        auto created = std::unique_ptr<PmCommandBuffer_T>(
            make<PmCommandBuffer_T>());
        if (!created) return PM_ERROR_OUT_OF_HOST_MEMORY;
        created->device = device;
        created->pool = pool;
        HRESULT result = device->native->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            IID_PPV_ARGS(&created->allocator));
        if (FAILED(result)) return result_from_hresult(result);
        result = device->native->CreateCommandList(
            0U, D3D12_COMMAND_LIST_TYPE_COMPUTE, created->allocator.Get(),
            nullptr, IID_PPV_ARGS(&created->list));
        if (FAILED(result)) return result_from_hresult(result);
        result = created->list->Close();
        if (FAILED(result)) return result_from_hresult(result);
        outputs[index] = created.release();
    }
    return PM_SUCCESS;
}

void drv_free_command_buffers(PmDevice device, PmCommandPool pool,
                              std::uint32_t count,
                              const PmCommandBuffer *buffers) noexcept {
    if (!valid_device_object(device, pool) ||
        (count != 0U && buffers == nullptr))
        return;
    for (std::uint32_t index = 0U; index < count; ++index)
        if (valid_device_object(device, buffers[index]) &&
            buffers[index]->pool == pool &&
            buffers[index]->state != CommandState::pending)
            delete buffers[index];
}

PmResult drv_begin_command_buffer(PmCommandBuffer commands,
                                  const PmCommandBufferBeginInfo *info) noexcept {
    if (commands == nullptr || info == nullptr || !valid_structure(
            info->sType, PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            info->pNext) || commands->state == CommandState::recording ||
        commands->state == CommandState::pending)
        return PM_ERROR_INVALID_ARGUMENT;
    HRESULT result = commands->allocator->Reset();
    if (SUCCEEDED(result))
        result = commands->list->Reset(commands->allocator.Get(), nullptr);
    if (FAILED(result)) return result_from_hresult(result);
    commands->state = CommandState::recording;
    return PM_SUCCESS;
}

PmResult drv_end_command_buffer(PmCommandBuffer commands) noexcept {
    if (commands == nullptr || commands->state != CommandState::recording)
        return PM_ERROR_INVALID_STATE;
    const HRESULT result = commands->list->Close();
    if (FAILED(result)) return result_from_hresult(result);
    commands->state = CommandState::executable;
    return PM_SUCCESS;
}

PmResult drv_reset_command_buffer(PmCommandBuffer commands) noexcept {
    if (commands == nullptr || commands->state == CommandState::pending ||
        (commands->pool->flags & PM_COMMAND_POOL_RESET_BUFFER_BIT) == 0U)
        return PM_ERROR_INVALID_STATE;
    commands->state = CommandState::initial;
    return PM_SUCCESS;
}

PmResult drv_cmd_copy_buffer(PmCommandBuffer commands, PmBuffer source,
                             PmBuffer destination, std::uint32_t count,
                             const PmBufferCopy *regions) noexcept {
    if (commands == nullptr || commands->state != CommandState::recording ||
        !valid_device_object(commands->device, source) ||
        !valid_device_object(commands->device, destination) ||
        source->memory == nullptr || destination->memory == nullptr ||
        (source->usage & PM_BUFFER_TRANSFER_SRC_BIT) == 0U ||
        (destination->usage & PM_BUFFER_TRANSFER_DST_BIT) == 0U ||
        count == 0U || regions == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    transition(commands, source->memory, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(commands, destination->memory, D3D12_RESOURCE_STATE_COPY_DEST);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto &region = regions[index];
        if (region.size == 0U ||
            !fits(region.srcOffset, region.size, source->size) ||
            !fits(region.dstOffset, region.size, destination->size))
            return PM_ERROR_INVALID_ARGUMENT;
        commands->list->CopyBufferRegion(
            destination->memory->resource.Get(),
            destination->offset + region.dstOffset,
            source->memory->resource.Get(), source->offset + region.srcOffset,
            region.size);
    }
    return PM_SUCCESS;
}

PmResult drv_cmd_pipeline_barrier(
    PmCommandBuffer commands, std::uint32_t memory_count,
    const PmMemoryBarrier *memory_barriers, std::uint32_t buffer_count,
    const PmBufferMemoryBarrier *buffer_barriers) noexcept {
    if (commands == nullptr || commands->state != CommandState::recording ||
        (memory_count != 0U && memory_barriers == nullptr) ||
        (buffer_count != 0U && buffer_barriers == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    (void)memory_barriers;
    if (memory_count != 0U) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        commands->list->ResourceBarrier(1U, &barrier);
    }
    for (std::uint32_t index = 0U; index < buffer_count; ++index) {
        const auto &source = buffer_barriers[index];
        if (!valid_device_object(commands->device, source.range.buffer) ||
            source.range.buffer->memory == nullptr ||
            (source.srcQueueFamilyIndex != PM_QUEUE_FAMILY_IGNORED &&
             source.srcQueueFamilyIndex != 0U) ||
            (source.dstQueueFamilyIndex != PM_QUEUE_FAMILY_IGNORED &&
             source.dstQueueFamilyIndex != 0U))
            return PM_ERROR_FEATURE_NOT_PRESENT;
        if (source.range.buffer->memory->type != k_memory_default) continue;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = source.range.buffer->memory->resource.Get();
        commands->list->ResourceBarrier(1U, &barrier);
    }
    return PM_SUCCESS;
}

PmResult drv_create_fence(PmDevice device, PmFlags flags,
                          PmFence *output) noexcept {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (device == nullptr || (flags & ~PM_FENCE_SIGNALED_BIT) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto created = std::unique_ptr<PmFence_T>(make<PmFence_T>());
    if (!created) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    const UINT64 initial = (flags & PM_FENCE_SIGNALED_BIT) != 0U ? 1U : 0U;
    HRESULT result = device->native->CreateFence(
        initial, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&created->native));
    if (FAILED(result)) return result_from_hresult(result);
    created->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (created->event == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->target = 1U;
    created->submitted = initial != 0U;
    *output = created.release();
    return PM_SUCCESS;
}

void drv_destroy_fence(PmDevice device, PmFence fence) noexcept {
    if (valid_device_object(device, fence)) delete fence;
}

PmResult drv_get_fence_status(PmDevice device, PmFence fence) noexcept {
    if (!valid_device_object(device, fence)) return PM_ERROR_INVALID_ARGUMENT;
    if (fence->native->GetCompletedValue() < fence->target)
        return PM_NOT_READY;
    complete_commands(device, fence->pending);
    fence->pending.clear();
    return PM_SUCCESS;
}

PmResult drv_reset_fences(PmDevice device, std::uint32_t count,
                          const PmFence *fences) noexcept {
    if (device == nullptr || count == 0U || fences == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < count; ++index) {
        PmFence fence = fences[index];
        if (!valid_device_object(device, fence) ||
            fence->native->GetCompletedValue() < fence->target)
            return PM_ERROR_INVALID_STATE;
    }
    for (std::uint32_t index = 0U; index < count; ++index) {
        PmFence fence = fences[index];
        complete_commands(device, fence->pending);
        fence->pending.clear();
        fence->target = fence->native->GetCompletedValue() + 1U;
        fence->submitted = false;
    }
    return PM_SUCCESS;
}

PmResult drv_wait_for_fences(PmDevice device, std::uint32_t count,
                             const PmFence *fences, PmBool32 wait_all,
                             std::uint64_t timeout) noexcept {
    if (device == nullptr || count == 0U || fences == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < count; ++index)
        if (!valid_device_object(device, fences[index]) ||
            !fences[index]->submitted)
            return PM_ERROR_INVALID_STATE;
    if (wait_all == PM_TRUE) {
        const auto begin = std::chrono::steady_clock::now();
        for (std::uint32_t index = 0U; index < count; ++index) {
            std::uint64_t remaining = timeout;
            if (timeout != PM_TIMEOUT_INFINITE) {
                const auto elapsed = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin).count();
                if (elapsed >= static_cast<std::int64_t>(timeout))
                    return PM_TIMEOUT;
                remaining -= static_cast<std::uint64_t>(elapsed);
            }
            const PmResult result = wait_native_fence(
                fences[index]->native.Get(), fences[index]->target,
                fences[index]->event, remaining);
            if (result != PM_SUCCESS) return result;
            complete_commands(device, fences[index]->pending);
            fences[index]->pending.clear();
        }
        return PM_SUCCESS;
    }
    const auto begin = std::chrono::steady_clock::now();
    for (;;) {
        for (std::uint32_t index = 0U; index < count; ++index)
            if (fences[index]->native->GetCompletedValue() >=
                fences[index]->target) {
                complete_commands(device, fences[index]->pending);
                fences[index]->pending.clear();
                return PM_SUCCESS;
            }
        if (timeout == 0U) return PM_TIMEOUT;
        if (timeout != PM_TIMEOUT_INFINITE &&
            static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count()) >= timeout)
            return PM_TIMEOUT;
        std::this_thread::yield();
    }
}

PmResult drv_queue_submit(PmQueue queue, const PmSubmitInfo *info,
                          PmFence fence) noexcept {
    if (queue == nullptr || info == nullptr || !valid_structure(
            info->sType, PM_STRUCTURE_TYPE_SUBMIT_INFO, info->pNext) ||
        info->waitCount != 0U || info->signalCount != 0U ||
        (info->commandBufferCount != 0U && info->pCommandBuffers == nullptr) ||
        (fence != nullptr && (!valid_device_object(queue->device, fence) ||
                              fence->submitted)))
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<ID3D12CommandList *> native;
    try {
        native.reserve(info->commandBufferCount);
        for (std::uint32_t index = 0U; index < info->commandBufferCount;
             ++index) {
            PmCommandBuffer commands = info->pCommandBuffers[index];
            if (!valid_device_object(queue->device, commands) ||
                commands->state != CommandState::executable)
                return PM_ERROR_INVALID_STATE;
            native.push_back(commands->list.Get());
        }
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    {
        std::lock_guard lock(queue->device->pending_mutex);
        try {
            queue->device->pending_commands.reserve(
                queue->device->pending_commands.size() +
                info->commandBufferCount);
            if (fence != nullptr)
                fence->pending.assign(info->pCommandBuffers,
                                      info->pCommandBuffers +
                                          info->commandBufferCount);
        } catch (const std::bad_alloc &) {
            return PM_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    if (!native.empty())
        queue->device->command_queue->ExecuteCommandLists(
            static_cast<UINT>(native.size()), native.data());
    if (fence != nullptr) {
        const HRESULT result = queue->device->command_queue->Signal(
            fence->native.Get(), fence->target);
        if (FAILED(result)) {
            fence->pending.clear();
            return result_from_hresult(result);
        }
        fence->submitted = true;
    }
    {
        std::lock_guard lock(queue->device->pending_mutex);
        for (std::uint32_t index = 0U; index < info->commandBufferCount;
             ++index) {
            PmCommandBuffer commands = info->pCommandBuffers[index];
            commands->state = CommandState::pending;
            queue->device->pending_commands.push_back(commands);
        }
    }
    return PM_SUCCESS;
}

PmResult drv_queue_wait_idle(PmQueue queue) noexcept {
    if (queue == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    return wait_idle(queue->device);
}

PmResult drv_device_wait_idle(PmDevice device) noexcept {
    return wait_idle(device);
}

PmResult drv_get_interop_properties(
    PmPhysicalDevice physical, PmNativeBackend backend,
    PmNativeInteropProperties *properties) noexcept {
    if (physical == nullptr || properties == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    *properties = {};
    return backend == PM_NATIVE_BACKEND_D3D12
        ? PM_SUCCESS : PM_ERROR_FEATURE_NOT_PRESENT;
}

PmResult drv_get_native_device(PmDevice device,
                               PmNativeDeviceInfo *output) noexcept {
    if (device == nullptr || output == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    *output = {};
    output->backend = PM_NATIVE_BACKEND_D3D12;
    output->physicalDevice = reinterpret_cast<PmNativeHandle>(
        device->physical->adapter.Get());
    output->device = reinterpret_cast<PmNativeHandle>(device->native.Get());
    return PM_SUCCESS;
}

PmResult drv_get_native_buffer(PmDevice device, PmBuffer buffer,
                               PmNativeBufferInfo *output) noexcept {
    if (!valid_device_object(device, buffer) || output == nullptr ||
        buffer->memory == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    *output = {};
    output->buffer = reinterpret_cast<PmNativeHandle>(
        buffer->memory->resource.Get());
    output->offset = buffer->offset;
    output->size = buffer->size;
    output->usage = buffer->usage;
    output->nativeUsage = static_cast<std::uint64_t>(buffer->memory->state);
    output->owningQueueFamilyIndex = buffer->queue_family;
    return PM_SUCCESS;
}

PmResult drv_get_memory_budget(PmDevice device, PmMemoryBudget *budget) noexcept {
    if (device == nullptr || budget == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    *budget = {};
    ComPtr<IDXGIAdapter3> adapter;
    const HRESULT as_result = device->physical->adapter.As(&adapter);
    if (FAILED(as_result)) return result_from_hresult(as_result);
    DXGI_QUERY_VIDEO_MEMORY_INFO information{};
    const HRESULT result = adapter->QueryVideoMemoryInfo(
        0U, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &information);
    if (FAILED(result)) return result_from_hresult(result);
    budget->heapCount = 1U;
    budget->budget[0] = information.Budget;
    budget->usage[0] = information.CurrentUsage;
    budget->applicationAllocationBytes =
        device->application_bytes.load(std::memory_order_relaxed);
    return PM_SUCCESS;
}

void drv_get_limits(PmPhysicalDevice physical,
                    PmPhysicalDeviceLimits *limits) noexcept {
    if (limits == nullptr) return;
    *limits = {};
    if (physical == nullptr) return;
    limits->maxAllocationSize = physical->description.DedicatedVideoMemory;
    limits->maxBufferSize = physical->description.DedicatedVideoMemory;
    limits->nonCoherentAtomSize = 1U;
    limits->maxMemoryAllocations = 1U << 20U;
}

void drv_get_format_properties(PmPhysicalDevice physical, PmFormat format,
                               PmFormatProperties *properties) noexcept {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    switch (format) {
    case PM_FORMAT_R32_UINT:
    case PM_FORMAT_R32_SFLOAT:
        properties->elementSize = properties->elementAlignment = 4U;
        break;
    case PM_FORMAT_R32G32B32_SFLOAT:
        properties->elementSize = 12U;
        properties->elementAlignment = 4U;
        break;
    case PM_FORMAT_R32G32B32A32_SFLOAT:
        properties->elementSize = properties->elementAlignment = 16U;
        break;
    default: break;
    }
}

#define PM_D3D12_LOOKUP(public_name, implementation)                         \
    if (std::strcmp(name, #public_name) == 0)                                \
        return reinterpret_cast<PFN_pmVoidFunction>(&implementation)

PFN_pmVoidFunction lookup(const char *name) noexcept {
    if (name == nullptr) return nullptr;
    PM_D3D12_LOOKUP(pmEnumerateInstanceVersion, drv_enumerate_instance_version);
    PM_D3D12_LOOKUP(pmEnumerateDeviceExtensionProperties,
                    drv_enumerate_device_extensions);
    PM_D3D12_LOOKUP(pmCreateInstance, drv_create_instance);
    PM_D3D12_LOOKUP(pmDestroyInstance, drv_destroy_instance);
    PM_D3D12_LOOKUP(pmEnumeratePhysicalDevices,
                    drv_enumerate_physical_devices);
    PM_D3D12_LOOKUP(pmGetPhysicalDeviceProperties,
                    drv_get_physical_properties);
    PM_D3D12_LOOKUP(pmGetPhysicalDeviceQueueFamilies, drv_get_queue_families);
    PM_D3D12_LOOKUP(pmGetPhysicalDeviceFeatures, drv_get_features);
    PM_D3D12_LOOKUP(pmCreateDevice, drv_create_device);
    PM_D3D12_LOOKUP(pmDestroyDevice, drv_destroy_device);
    PM_D3D12_LOOKUP(pmGetDeviceQueue, drv_get_device_queue);
    PM_D3D12_LOOKUP(pmGetPhysicalDeviceMemoryProperties,
                    drv_get_memory_properties);
    PM_D3D12_LOOKUP(pmAllocateMemory, drv_allocate_memory);
    PM_D3D12_LOOKUP(pmFreeMemory, drv_free_memory);
    PM_D3D12_LOOKUP(pmMapMemory, drv_map_memory);
    PM_D3D12_LOOKUP(pmUnmapMemory, drv_unmap_memory);
    PM_D3D12_LOOKUP(pmFlushMappedMemoryRanges, drv_mapped_ranges);
    PM_D3D12_LOOKUP(pmInvalidateMappedMemoryRanges, drv_mapped_ranges);
    PM_D3D12_LOOKUP(pmCreateBuffer, drv_create_buffer);
    PM_D3D12_LOOKUP(pmDestroyBuffer, drv_destroy_buffer);
    PM_D3D12_LOOKUP(pmGetBufferMemoryRequirements,
                    drv_get_buffer_requirements);
    PM_D3D12_LOOKUP(pmBindBufferMemory, drv_bind_buffer);
    PM_D3D12_LOOKUP(pmCreateCommandPool, drv_create_command_pool);
    PM_D3D12_LOOKUP(pmDestroyCommandPool, drv_destroy_command_pool);
    PM_D3D12_LOOKUP(pmResetCommandPool, drv_reset_command_pool);
    PM_D3D12_LOOKUP(pmAllocateCommandBuffers, drv_allocate_command_buffers);
    PM_D3D12_LOOKUP(pmFreeCommandBuffers, drv_free_command_buffers);
    PM_D3D12_LOOKUP(pmBeginCommandBuffer, drv_begin_command_buffer);
    PM_D3D12_LOOKUP(pmEndCommandBuffer, drv_end_command_buffer);
    PM_D3D12_LOOKUP(pmResetCommandBuffer, drv_reset_command_buffer);
    PM_D3D12_LOOKUP(pmCmdCopyBuffer, drv_cmd_copy_buffer);
    PM_D3D12_LOOKUP(pmCmdPipelineBarrier, drv_cmd_pipeline_barrier);
    PM_D3D12_LOOKUP(pmQueueSubmit, drv_queue_submit);
    PM_D3D12_LOOKUP(pmQueueWaitIdle, drv_queue_wait_idle);
    PM_D3D12_LOOKUP(pmDeviceWaitIdle, drv_device_wait_idle);
    PM_D3D12_LOOKUP(pmCreateFence, drv_create_fence);
    PM_D3D12_LOOKUP(pmDestroyFence, drv_destroy_fence);
    PM_D3D12_LOOKUP(pmGetFenceStatus, drv_get_fence_status);
    PM_D3D12_LOOKUP(pmResetFences, drv_reset_fences);
    PM_D3D12_LOOKUP(pmWaitForFences, drv_wait_for_fences);
    PM_D3D12_LOOKUP(pmGetPhysicalDeviceInteropProperties,
                    drv_get_interop_properties);
    PM_D3D12_LOOKUP(pmGetNativeDevice, drv_get_native_device);
    PM_D3D12_LOOKUP(pmGetNativeBuffer, drv_get_native_buffer);
    PM_D3D12_LOOKUP(pmGetDeviceMemoryBudget, drv_get_memory_budget);
    PM_D3D12_LOOKUP(pmGetPhysicalDeviceLimits, drv_get_limits);
    PM_D3D12_LOOKUP(pmGetPhysicalDeviceFormatProperties,
                    drv_get_format_properties);
    return nullptr;
}

#undef PM_D3D12_LOOKUP

} // namespace

#define PM_DRIVER_EXPORT extern "C" PM_DRIVER_API_ATTR

PM_DRIVER_EXPORT PmResult PMAPI_CALL
pm_driverNegotiateLoaderInterfaceVersion(std::uint32_t *version) {
    if (version == nullptr || *version == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    *version = std::min(*version, PM_DRIVER_INTERFACE_VERSION);
    return PM_SUCCESS;
}

PM_DRIVER_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetInstanceProcAddr(
    PmInstance instance, const char *name) {
    if (instance == nullptr && name != nullptr &&
        std::strcmp(name, "pmEnumerateInstanceVersion") != 0 &&
        std::strcmp(name, "pmCreateInstance") != 0)
        return nullptr;
    return lookup(name);
}

PM_DRIVER_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetDeviceProcAddr(
    PmDevice device, const char *name) {
    return device == nullptr ? nullptr : lookup(name);
}

PM_DRIVER_EXPORT PFN_pmVoidFunction PMAPI_CALL
pm_driverGetPhysicalDeviceProcAddr(PmInstance instance, const char *name) {
    return instance == nullptr ? nullptr : lookup(name);
}
