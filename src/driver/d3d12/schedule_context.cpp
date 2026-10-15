// SPDX-License-Identifier: MIT
#include "schedule_context.hpp"

#include "simulation/geometry_storage.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <malloc.h>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace parallel_mater::d3d12 {
namespace {

thread_local ScheduleContext *active_context{};

[[nodiscard]] D3D12_HEAP_PROPERTIES heap_properties(
    D3D12_HEAP_TYPE type) noexcept {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    result.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    result.CreationNodeMask = 1U;
    result.VisibleNodeMask = 1U;
    return result;
}

[[nodiscard]] D3D12_RESOURCE_DESC buffer_description(
    std::uint64_t size, D3D12_RESOURCE_FLAGS flags) noexcept {
    D3D12_RESOURCE_DESC result{};
    result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    result.Width = std::max<std::uint64_t>(4U, (size + 3U) & ~3ULL);
    result.Height = 1U;
    result.DepthOrArraySize = 1U;
    result.MipLevels = 1U;
    result.Format = DXGI_FORMAT_UNKNOWN;
    result.SampleDesc.Count = 1U;
    result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    result.Flags = flags;
    return result;
}

[[nodiscard]] cudaError_t cuda_error(HRESULT result) noexcept {
    if (SUCCEEDED(result)) return cudaSuccess;
    if (result == E_OUTOFMEMORY) return cudaErrorMemoryAllocation;
    if (result == E_INVALIDARG) return cudaErrorInvalidValue;
    if (result == DXGI_ERROR_DEVICE_REMOVED ||
        result == DXGI_ERROR_DEVICE_RESET || result == DXGI_ERROR_DEVICE_HUNG)
        return cudaErrorLaunchFailure;
    return cudaErrorInvalidResourceHandle;
}

[[nodiscard]] PmResult pm_error(HRESULT result) noexcept {
    const cudaError_t converted = cuda_error(result);
    switch (converted) {
    case cudaSuccess: return PM_SUCCESS;
    case cudaErrorMemoryAllocation: return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    case cudaErrorInvalidValue: return PM_ERROR_INVALID_ARGUMENT;
    case cudaErrorLaunchFailure: return PM_ERROR_DEVICE_LOST;
    default: return PM_ERROR_INVALID_STATE;
    }
}

[[nodiscard]] PmResult wait_fence(ID3D12Fence *fence, std::uint64_t value,
                                  HANDLE event) noexcept {
    if (fence == nullptr || event == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (fence->GetCompletedValue() >= value) return PM_SUCCESS;
    const HRESULT armed = fence->SetEventOnCompletion(value, event);
    if (FAILED(armed)) return pm_error(armed);
    return WaitForSingleObject(event, INFINITE) == WAIT_OBJECT_0
        ? PM_SUCCESS : PM_ERROR_DEVICE_LOST;
}

struct GraphicsTriangleMeshResource {
    GraphicsBufferAddress vertices{};
    GraphicsBufferAddress indices{};
    std::uint32_t vertex_count{};
    std::uint32_t index_count{};
    std::uint32_t generation{};
    std::uint32_t alive{};
    Vec3 minimum{};
    Vec3 maximum{};
    Vec3 bounding_center{};
    float bounding_radius{};
    Vec3 unit_inertia{};
    GraphicsBufferAddress bvh_nodes{};
    std::uint32_t bvh_node_count{};
    GraphicsBufferAddress bvh_leaves{};
    std::uint32_t bvh_leaf_count{};
    GraphicsBufferAddress solid_planes{};
    std::uint32_t solid_unique_plane_count{};
    GraphicsBufferAddress shell_normals{};
};
static_assert(sizeof(GraphicsTriangleMeshResource) == 152U);

} // namespace

struct ScheduleContext::Impl {
    struct Allocation {
        void *host{};
        std::size_t size{};
        ComPtr<ID3D12Resource> storage;
        ComPtr<ID3D12Resource> upload;
        ComPtr<ID3D12Resource> readback;
        std::byte *upload_data{};
        std::byte *readback_data{};
        std::uint32_t descriptor{invalid_descriptor};
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
    };

    enum class AuxiliaryKind : std::uint8_t { widened_flags, meshes };
    struct Auxiliary {
        const void *key{};
        AuxiliaryKind kind{};
        std::uint32_t count{};
        std::size_t size{};
        ComPtr<ID3D12Resource> storage;
        ComPtr<ID3D12Resource> upload;
        ComPtr<ID3D12Resource> readback;
        std::byte *upload_data{};
        std::byte *readback_data{};
        std::uint32_t descriptor{invalid_descriptor};
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
    };

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event{};
    std::uint64_t next_fence{1U};
    std::uint64_t submitted_fence{};
    SlangCommandPath slang;
    ComPtr<ID3D12Resource> constants;
    std::byte *constant_data{};
    std::size_t constant_capacity{16U * 1024U * 1024U};
    std::size_t constant_offset{};
    std::vector<Allocation> allocations;
    std::vector<Auxiliary> auxiliary;
    std::vector<ComPtr<ID3D12Resource>> transient_uploads;
    bool open{};
    bool upload_pending{true};
    cudaError_t last_error{cudaSuccess};
    std::mutex mutex;

    ~Impl() {
        if (fence_event != nullptr) CloseHandle(fence_event);
        if (constants && constant_data != nullptr) constants->Unmap(0U, nullptr);
        for (auto &item : auxiliary) {
            if (item.upload && item.upload_data != nullptr)
                item.upload->Unmap(0U, nullptr);
            if (item.readback && item.readback_data != nullptr)
                item.readback->Unmap(0U, nullptr);
            slang.release_descriptor(item.descriptor);
        }
        for (auto &item : allocations) {
            if (item.upload && item.upload_data != nullptr)
                item.upload->Unmap(0U, nullptr);
            if (item.readback && item.readback_data != nullptr)
                item.readback->Unmap(0U, nullptr);
            slang.release_descriptor(item.descriptor);
            _aligned_free(item.host);
        }
    }

    [[nodiscard]] Allocation *find(const void *pointer,
                                   std::size_t size = 0U) noexcept {
        if (pointer == nullptr) return nullptr;
        const auto value = reinterpret_cast<std::uintptr_t>(pointer);
        for (auto &item : allocations) {
            const auto begin = reinterpret_cast<std::uintptr_t>(item.host);
            if (value >= begin && value - begin <= item.size &&
                size <= item.size - (value - begin))
                return &item;
        }
        return nullptr;
    }

    [[nodiscard]] const Allocation *find(const void *pointer) const noexcept {
        return const_cast<Impl *>(this)->find(pointer);
    }

    void transition(ID3D12Resource *resource, D3D12_RESOURCE_STATES &state,
                    D3D12_RESOURCE_STATES desired) noexcept {
        if (state == desired) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = state;
        barrier.Transition.StateAfter = desired;
        commands->ResourceBarrier(1U, &barrier);
        state = desired;
    }

    void uav_barrier(ID3D12Resource *resource = nullptr) noexcept {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = resource;
        commands->ResourceBarrier(1U, &barrier);
    }

    [[nodiscard]] HRESULT create_buffer_set(
        std::size_t size, ComPtr<ID3D12Resource> &storage,
        ComPtr<ID3D12Resource> &upload, ComPtr<ID3D12Resource> &readback,
        std::byte **upload_data, std::byte **readback_data,
        std::uint32_t *descriptor) noexcept {
        const auto storage_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        const auto upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
        const auto readback_heap = heap_properties(D3D12_HEAP_TYPE_READBACK);
        const auto storage_desc = buffer_description(
            size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const auto staging_desc = buffer_description(size, D3D12_RESOURCE_FLAG_NONE);
        HRESULT result = device->CreateCommittedResource(
            &storage_heap, D3D12_HEAP_FLAG_NONE, &storage_desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&storage));
        if (FAILED(result)) return result;
        result = device->CreateCommittedResource(
            &upload_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
        if (FAILED(result)) return result;
        result = device->CreateCommittedResource(
            &readback_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
        if (FAILED(result)) return result;
        D3D12_RANGE no_read{0U, 0U};
        result = upload->Map(0U, &no_read, reinterpret_cast<void **>(upload_data));
        if (FAILED(result)) return result;
        D3D12_RANGE read_range{0U, staging_desc.Width};
        result = readback->Map(
            0U, &read_range, reinterpret_cast<void **>(readback_data));
        if (FAILED(result)) return result;
        return slang.create_raw_buffer_view(storage.Get(), 0U,
                                             storage_desc.Width, descriptor);
    }

    [[nodiscard]] cudaError_t wait_submitted() noexcept {
        if (submitted_fence == 0U) return cudaSuccess;
        const PmResult waited = wait_fence(
            fence.Get(), submitted_fence, fence_event);
        return waited == PM_SUCCESS ? cudaSuccess : cudaErrorLaunchFailure;
    }

    [[nodiscard]] cudaError_t begin() noexcept {
        if (open) return cudaSuccess;
        const cudaError_t waited = wait_submitted();
        if (waited != cudaSuccess) return last_error = waited;
        HRESULT result = allocator->Reset();
        if (SUCCEEDED(result)) result = commands->Reset(allocator.Get(), nullptr);
        if (FAILED(result)) return last_error = cuda_error(result);
        open = true;
        if (!upload_pending) return cudaSuccess;
        for (auto &item : allocations) {
            std::memcpy(item.upload_data, item.host, item.size);
            transition(item.storage.Get(), item.state,
                       D3D12_RESOURCE_STATE_COPY_DEST);
            commands->CopyBufferRegion(item.storage.Get(), 0U,
                                       item.upload.Get(), 0U, item.size);
            transition(item.storage.Get(), item.state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        upload_pending = false;
        return cudaSuccess;
    }

    [[nodiscard]] cudaError_t submit(std::uint64_t &value) noexcept {
        if (!open) {
            const cudaError_t started = begin();
            if (started != cudaSuccess) return started;
        }
        HRESULT result = commands->Close();
        if (FAILED(result)) return last_error = cuda_error(result);
        ID3D12CommandList *lists[]{commands.Get()};
        queue->ExecuteCommandLists(1U, lists);
        value = next_fence++;
        result = queue->Signal(fence.Get(), value);
        if (FAILED(result)) return last_error = cuda_error(result);
        submitted_fence = value;
        open = false;
        return cudaSuccess;
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS reserve_constants(
        const void *source, std::size_t size) noexcept {
        const std::size_t aligned = (constant_offset + 255U) & ~255ULL;
        if (source == nullptr || size == 0U || aligned > constant_capacity ||
            size > constant_capacity - aligned)
            return 0U;
        std::memcpy(constant_data + aligned, source, size);
        constant_offset = aligned + size;
        return constants->GetGPUVirtualAddress() + aligned;
    }

    [[nodiscard]] cudaError_t readback_all() noexcept {
        const cudaError_t waited = wait_submitted();
        if (waited != cudaSuccess) return waited;
        transient_uploads.clear();
        HRESULT result = allocator->Reset();
        if (SUCCEEDED(result)) result = commands->Reset(allocator.Get(), nullptr);
        if (FAILED(result)) return last_error = cuda_error(result);
        open = true;
        for (auto &item : allocations) {
            transition(item.storage.Get(), item.state,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->CopyBufferRegion(item.readback.Get(), 0U,
                                       item.storage.Get(), 0U, item.size);
            transition(item.storage.Get(), item.state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        for (auto &item : auxiliary) {
            if (item.kind != AuxiliaryKind::widened_flags) continue;
            transition(item.storage.Get(), item.state,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->CopyBufferRegion(item.readback.Get(), 0U,
                                       item.storage.Get(), 0U, item.size);
            transition(item.storage.Get(), item.state,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        std::uint64_t value{};
        const cudaError_t submitted = submit(value);
        if (submitted != cudaSuccess) return submitted;
        const cudaError_t completed = wait_submitted();
        if (completed != cudaSuccess) return completed;
        for (auto &item : allocations)
            std::memcpy(item.host, item.readback_data, item.size);
        for (auto &item : auxiliary) {
            if (item.kind != AuxiliaryKind::widened_flags) continue;
            auto *destination = const_cast<std::uint8_t *>(
                static_cast<const std::uint8_t *>(item.key));
            const auto *source = reinterpret_cast<const std::uint32_t *>(
                item.readback_data);
            for (std::uint32_t index = 0U; index < item.count; ++index)
                destination[index] = source[index] != 0U ? 1U : 0U;
        }
        for (auto &item : auxiliary) {
            if (item.upload && item.upload_data != nullptr)
                item.upload->Unmap(0U, nullptr);
            if (item.readback && item.readback_data != nullptr)
                item.readback->Unmap(0U, nullptr);
            slang.release_descriptor(item.descriptor);
        }
        auxiliary.clear();
        constant_offset = 0U;
        upload_pending = true;
        return cudaSuccess;
    }
};

ScheduleContext::ScheduleContext() : impl_(std::make_unique<Impl>()) {}
ScheduleContext::~ScheduleContext() = default;

PmResult ScheduleContext::initialize(IDXGIAdapter1 *adapter) noexcept {
    if (adapter == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    impl_->adapter = adapter;
    HRESULT result = D3D12CreateDevice(
        adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&impl_->device));
    if (FAILED(result)) return pm_error(result);
    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    result = impl_->device->CreateCommandQueue(
        &queue_description, IID_PPV_ARGS(&impl_->queue));
    if (FAILED(result)) return pm_error(result);
    result = impl_->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&impl_->allocator));
    if (FAILED(result)) return pm_error(result);
    result = impl_->device->CreateCommandList(
        0U, D3D12_COMMAND_LIST_TYPE_COMPUTE, impl_->allocator.Get(), nullptr,
        IID_PPV_ARGS(&impl_->commands));
    if (FAILED(result)) return pm_error(result);
    result = impl_->commands->Close();
    if (FAILED(result)) return pm_error(result);
    result = impl_->device->CreateFence(
        0U, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&impl_->fence));
    if (FAILED(result)) return pm_error(result);
    impl_->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (impl_->fence_event == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    result = impl_->slang.initialize(impl_->device.Get());
    if (FAILED(result)) return pm_error(result);
    const auto upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    const auto description = buffer_description(
        impl_->constant_capacity, D3D12_RESOURCE_FLAG_NONE);
    result = impl_->device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&impl_->constants));
    if (FAILED(result)) return pm_error(result);
    D3D12_RANGE no_read{0U, 0U};
    result = impl_->constants->Map(
        0U, &no_read, reinterpret_cast<void **>(&impl_->constant_data));
    return pm_error(result);
}

PmResult ScheduleContext::allocate(PmDeviceSize size, void **host) noexcept {
    if (host == nullptr || size == 0U || size > SIZE_MAX)
        return PM_ERROR_INVALID_ARGUMENT;
    *host = nullptr;
    void *shadow = _aligned_malloc(static_cast<std::size_t>(size), 256U);
    if (shadow == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    std::memset(shadow, 0, static_cast<std::size_t>(size));
    Impl::Allocation allocation{};
    allocation.host = shadow;
    allocation.size = static_cast<std::size_t>(size);
    const HRESULT result = impl_->create_buffer_set(
        allocation.size, allocation.storage, allocation.upload,
        allocation.readback, &allocation.upload_data,
        &allocation.readback_data, &allocation.descriptor);
    if (FAILED(result)) {
        _aligned_free(shadow);
        return pm_error(result);
    }
    try {
        impl_->allocations.push_back(std::move(allocation));
    } catch (const std::bad_alloc &) {
        _aligned_free(shadow);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    *host = shadow;
    impl_->upload_pending = true;
    return PM_SUCCESS;
}

void ScheduleContext::free(void *host) noexcept {
    if (host == nullptr) return;
    (void)synchronize();
    const auto found = std::find_if(
        impl_->allocations.begin(), impl_->allocations.end(),
        [host](const Impl::Allocation &item) { return item.host == host; });
    if (found == impl_->allocations.end()) return;
    if (found->upload_data != nullptr) found->upload->Unmap(0U, nullptr);
    if (found->readback_data != nullptr) found->readback->Unmap(0U, nullptr);
    impl_->slang.release_descriptor(found->descriptor);
    _aligned_free(found->host);
    impl_->allocations.erase(found);
}

PmResult ScheduleContext::copy_host(void *destination, const void *source,
                                    PmDeviceSize size) noexcept {
    if ((destination == nullptr || source == nullptr) && size != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    if (size > SIZE_MAX) return PM_ERROR_INVALID_ARGUMENT;
    if (size != 0U)
        std::memmove(destination, source, static_cast<std::size_t>(size));
    impl_->upload_pending = true;
    return PM_SUCCESS;
}

PmResult ScheduleContext::wait_idle() noexcept {
    return synchronize() == cudaSuccess ? PM_SUCCESS : PM_ERROR_DEVICE_LOST;
}

PmResult ScheduleContext::memory_budget(PmDeviceSize &available,
                                        PmDeviceSize &total) noexcept {
    ComPtr<IDXGIAdapter3> queried;
    const HRESULT converted = impl_->adapter.As(&queried);
    if (FAILED(converted)) return pm_error(converted);
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    const HRESULT result = queried->QueryVideoMemoryInfo(
        0U, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
    if (FAILED(result)) return pm_error(result);
    available = info.Budget > info.CurrentUsage
        ? info.Budget - info.CurrentUsage : 0U;
    total = info.Budget;
    return PM_SUCCESS;
}

ID3D12Device *ScheduleContext::device() const noexcept {
    return impl_->device.Get();
}

ID3D12CommandQueue *ScheduleContext::queue() const noexcept {
    return impl_->queue.Get();
}

ID3D12Resource *ScheduleContext::native_resource(
    const void *host) const noexcept {
    const auto *allocation = impl_->find(host);
    return allocation == nullptr ? nullptr : allocation->storage.Get();
}

GraphicsBufferAddress ScheduleContext::address(const void *pointer) noexcept {
    if (pointer == nullptr) return {};
    auto *allocation = impl_->find(pointer);
    if (allocation == nullptr) {
        impl_->last_error = cudaErrorInvalidValue;
        return {};
    }
    const auto byte_offset = reinterpret_cast<std::uintptr_t>(pointer) -
        reinterpret_cast<std::uintptr_t>(allocation->host);
    if (byte_offset > std::numeric_limits<std::uint32_t>::max()) {
        impl_->last_error = cudaErrorInvalidValue;
        return {};
    }
    return {allocation->descriptor, static_cast<std::uint32_t>(byte_offset), 1U};
}

GraphicsBufferAddress ScheduleContext::widened_flags(
    const std::uint8_t *pointer, std::uint32_t count) noexcept {
    if (pointer == nullptr) return {};
    for (const auto &item : impl_->auxiliary)
        if (item.key == pointer &&
            item.kind == Impl::AuxiliaryKind::widened_flags &&
            item.count >= count)
            return {item.descriptor, 0U, 1U};
    if (impl_->begin() != cudaSuccess) return {};
    Impl::Auxiliary item{};
    item.key = pointer;
    item.kind = Impl::AuxiliaryKind::widened_flags;
    item.count = count;
    item.size = std::max<std::size_t>(1U, count) * sizeof(std::uint32_t);
    const HRESULT created = impl_->create_buffer_set(
        item.size, item.storage, item.upload, item.readback,
        &item.upload_data, &item.readback_data, &item.descriptor);
    if (FAILED(created)) {
        impl_->last_error = cuda_error(created);
        return {};
    }
    const auto result = GraphicsBufferAddress{item.descriptor, 0U, 1U};
    try {
        impl_->auxiliary.push_back(std::move(item));
    } catch (const std::bad_alloc &) {
        impl_->last_error = cudaErrorMemoryAllocation;
        return {};
    }
    return result;
}

GraphicsBufferAddress ScheduleContext::translated_meshes(
    const void *mesh_pointer, const void *parameter_pointer,
    std::uint32_t body_count, std::uint32_t mesh_capacity) noexcept {
    if (mesh_pointer == nullptr) return {};
    const auto *meshes = static_cast<const simulation::TriangleMeshResource *>(
        mesh_pointer);
    const auto *parameters = static_cast<const simulation::BodyParameters *>(
        parameter_pointer);
    std::uint32_t required = mesh_capacity;
    for (std::uint32_t index = 0U; index < body_count; ++index)
        required = std::max(required, parameters[index].mesh.index + 1U);
    required = std::max(1U, required);
    for (const auto &item : impl_->auxiliary)
        if (item.key == mesh_pointer && item.kind == Impl::AuxiliaryKind::meshes &&
            item.count >= required)
            return {item.descriptor, 0U, 1U};
    if (impl_->begin() != cudaSuccess) return {};
    std::vector<GraphicsTriangleMeshResource> translated;
    try {
        translated.resize(required);
    } catch (const std::bad_alloc &) {
        impl_->last_error = cudaErrorMemoryAllocation;
        return {};
    }
    for (std::uint32_t index = 0U; index < required; ++index) {
        const auto &source = meshes[index];
        auto &destination = translated[index];
        destination.vertices = address(source.vertices);
        destination.indices = address(source.indices);
        destination.vertex_count = source.vertex_count;
        destination.index_count = source.index_count;
        destination.generation = source.generation;
        destination.alive = source.alive;
        destination.minimum = source.minimum;
        destination.maximum = source.maximum;
        destination.bounding_center = source.bounding_center;
        destination.bounding_radius = source.bounding_radius;
        destination.unit_inertia = source.unit_inertia;
        destination.bvh_nodes = address(source.bvh_nodes);
        destination.bvh_node_count = source.bvh_node_count;
        destination.bvh_leaves = address(source.bvh_leaves);
        destination.bvh_leaf_count = source.bvh_leaf_count;
        destination.solid_planes = address(source.solid_planes);
        destination.solid_unique_plane_count = source.solid_unique_plane_count;
        destination.shell_normals = address(source.shell_normals);
    }
    Impl::Auxiliary item{};
    item.key = mesh_pointer;
    item.kind = Impl::AuxiliaryKind::meshes;
    item.count = required;
    item.size = translated.size() * sizeof(translated[0]);
    const HRESULT created = impl_->create_buffer_set(
        item.size, item.storage, item.upload, item.readback,
        &item.upload_data, &item.readback_data, &item.descriptor);
    if (FAILED(created)) {
        impl_->last_error = cuda_error(created);
        return {};
    }
    std::memcpy(item.upload_data, translated.data(), item.size);
    impl_->transition(item.storage.Get(), item.state,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->commands->CopyBufferRegion(item.storage.Get(), 0U,
                                      item.upload.Get(), 0U, item.size);
    impl_->transition(item.storage.Get(), item.state,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto result = GraphicsBufferAddress{item.descriptor, 0U, 1U};
    try {
        impl_->auxiliary.push_back(std::move(item));
    } catch (const std::bad_alloc &) {
        impl_->last_error = cudaErrorMemoryAllocation;
        return {};
    }
    return result;
}

cudaError_t ScheduleContext::dispatch(
    std::string_view module, std::string_view entry, dim3 grid, dim3 block,
    std::span<const std::byte> entry_constants) noexcept {
    active_context = this;
    const cudaError_t started = impl_->begin();
    if (started != cudaSuccess) return started;
    const DispatchConstants global{
        grid.x, grid.y, grid.z, block.x};
    const auto global_address = impl_->reserve_constants(&global, sizeof(global));
    const auto entry_address = impl_->reserve_constants(
        entry_constants.data(), entry_constants.size());
    if (global_address == 0U || entry_address == 0U)
        return impl_->last_error = cudaErrorMemoryAllocation;
    const HRESULT result = impl_->slang.record_dispatch(
        impl_->commands.Get(), module, entry, global_address, entry_address,
        grid.x, grid.y, grid.z);
    if (FAILED(result)) return impl_->last_error = cuda_error(result);
    impl_->uav_barrier();
    return cudaSuccess;
}

cudaError_t ScheduleContext::copy_device(
    void *destination, const void *source, std::size_t size) noexcept {
    if (size == 0U) return cudaSuccess;
    auto *target = impl_->find(destination, size);
    auto *input = impl_->find(source, size);
    if (target == nullptr || input == nullptr)
        return impl_->last_error = cudaErrorInvalidValue;
    const cudaError_t started = impl_->begin();
    if (started != cudaSuccess) return started;
    const auto target_offset = reinterpret_cast<std::uintptr_t>(destination) -
        reinterpret_cast<std::uintptr_t>(target->host);
    const auto source_offset = reinterpret_cast<std::uintptr_t>(source) -
        reinterpret_cast<std::uintptr_t>(input->host);
    impl_->transition(input->storage.Get(), input->state,
                      D3D12_RESOURCE_STATE_COPY_SOURCE);
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->commands->CopyBufferRegion(
        target->storage.Get(), target_offset, input->storage.Get(),
        source_offset, size);
    impl_->transition(input->storage.Get(), input->state,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return cudaSuccess;
}

cudaError_t ScheduleContext::clear_device(
    void *destination, int value, std::size_t size) noexcept {
    if (size == 0U) return cudaSuccess;
    auto *target = impl_->find(destination, size);
    if (target == nullptr)
        return impl_->last_error = cudaErrorInvalidValue;
    const auto offset = reinterpret_cast<std::uintptr_t>(destination) -
        reinterpret_cast<std::uintptr_t>(target->host);
    const cudaError_t started = impl_->begin();
    if (started != cudaSuccess) return started;
    ComPtr<ID3D12Resource> upload;
    const auto upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    const auto description = buffer_description(size, D3D12_RESOURCE_FLAG_NONE);
    HRESULT created = impl_->device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    if (FAILED(created)) return impl_->last_error = cuda_error(created);
    void *mapped{};
    D3D12_RANGE no_read{0U, 0U};
    created = upload->Map(0U, &no_read, &mapped);
    if (FAILED(created)) return impl_->last_error = cuda_error(created);
    std::memset(mapped, value, size);
    upload->Unmap(0U, nullptr);
    try {
        impl_->transient_uploads.push_back(upload);
    } catch (const std::bad_alloc &) {
        return impl_->last_error = cudaErrorMemoryAllocation;
    }
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->commands->CopyBufferRegion(target->storage.Get(), offset,
                                      upload.Get(), 0U, size);
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl_->uav_barrier(target->storage.Get());
    return cudaSuccess;
}

cudaError_t ScheduleContext::synchronize() noexcept {
    active_context = this;
    if (impl_->open || impl_->upload_pending) {
        if (!impl_->open) {
            const cudaError_t started = impl_->begin();
            if (started != cudaSuccess) return started;
        }
        std::uint64_t ignored{};
        const cudaError_t submitted = impl_->submit(ignored);
        if (submitted != cudaSuccess) return submitted;
    }
    if (impl_->submitted_fence == 0U) return cudaSuccess;
    return impl_->readback_all();
}

cudaError_t ScheduleContext::record(cudaEvent_t event) noexcept {
    if (event == nullptr) return cudaErrorInvalidResourceHandle;
    active_context = this;
    std::uint64_t value{};
    const cudaError_t submitted = impl_->submit(value);
    if (submitted != cudaSuccess) return submitted;
    event->stream = this;
    event->fence_value = value;
    event->recorded = std::chrono::steady_clock::now();
    event->ready = false;
    return cudaSuccess;
}

cudaError_t ScheduleContext::wait(cudaEvent_t event) noexcept {
    if (event == nullptr || event->stream != this)
        return cudaErrorInvalidResourceHandle;
    const PmResult waited = wait_fence(
        impl_->fence.Get(), event->fence_value, impl_->fence_event);
    if (waited != PM_SUCCESS) return cudaErrorLaunchFailure;
    event->ready = true;
    return synchronize();
}

cudaError_t ScheduleContext::query(cudaEvent_t event) noexcept {
    if (event == nullptr || event->stream != this)
        return cudaErrorInvalidResourceHandle;
    if (event->ready || impl_->fence->GetCompletedValue() >= event->fence_value) {
        event->ready = true;
        return cudaSuccess;
    }
    return cudaErrorNotReady;
}

cudaError_t ScheduleContext::error(bool take) noexcept {
    const cudaError_t result = impl_->last_error;
    if (take) impl_->last_error = cudaSuccess;
    return result;
}

} // namespace parallel_mater::d3d12

cudaError_t pmD3D12MemcpyAsync(
    void *destination, const void *source, std::size_t size,
    cudaMemcpyKind kind, cudaStream_t stream) noexcept {
    auto *execution = parallel_mater::d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    if (kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyDefault)
        return execution->copy_device(destination, source, size);
    if (kind == cudaMemcpyDeviceToHost) {
        const cudaError_t waited = execution->synchronize();
        if (waited != cudaSuccess) return waited;
        return cudaMemcpy(destination, source, size, kind);
    }
    if (kind == cudaMemcpyHostToDevice) {
        const cudaError_t waited = execution->synchronize();
        if (waited != cudaSuccess) return waited;
        return cudaMemcpy(destination, source, size, kind);
    }
    return cudaMemcpy(destination, source, size, kind);
}

cudaError_t pmD3D12MemsetAsync(
    void *destination, int value, std::size_t size,
    cudaStream_t stream) noexcept {
    auto *execution = parallel_mater::d3d12::context(stream);
    return execution == nullptr ? cudaErrorInvalidResourceHandle
                                : execution->clear_device(destination, value,
                                                          size);
}

cudaError_t pmD3D12StreamSynchronize(cudaStream_t stream) noexcept {
    auto *execution = parallel_mater::d3d12::context(stream);
    return execution == nullptr ? cudaErrorInvalidResourceHandle
                                : execution->synchronize();
}

cudaError_t pmD3D12RecordEvent(cudaEvent_t event,
                              cudaStream_t stream) noexcept {
    auto *execution = parallel_mater::d3d12::context(stream);
    return execution == nullptr ? cudaErrorInvalidResourceHandle
                                : execution->record(event);
}

cudaError_t pmD3D12WaitEvent(cudaEvent_t event) noexcept {
    auto *execution = event == nullptr
        ? nullptr : parallel_mater::d3d12::context(event->stream);
    return execution == nullptr ? cudaErrorInvalidResourceHandle
                                : execution->wait(event);
}

cudaError_t pmD3D12QueryEvent(cudaEvent_t event) noexcept {
    auto *execution = event == nullptr
        ? nullptr : parallel_mater::d3d12::context(event->stream);
    return execution == nullptr ? cudaErrorInvalidResourceHandle
                                : execution->query(event);
}

cudaError_t pmD3D12PeekError(cudaStream_t stream, bool take) noexcept {
    auto *execution = stream == nullptr
        ? parallel_mater::d3d12::active_context
        : parallel_mater::d3d12::context(stream);
    return execution == nullptr ? cudaSuccess : execution->error(take);
}
