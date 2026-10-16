// SPDX-License-Identifier: MIT
#include "schedule_context.hpp"

#include "simulation/geometry_storage.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
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

[[nodiscard]] PmResult translate_hresult(HRESULT result) noexcept {
    if (SUCCEEDED(result)) return PM_SUCCESS;
    if (result == E_OUTOFMEMORY) return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == E_INVALIDARG) return PM_ERROR_INVALID_ARGUMENT;
    if (result == DXGI_ERROR_DEVICE_REMOVED ||
        result == DXGI_ERROR_DEVICE_RESET || result == DXGI_ERROR_DEVICE_HUNG)
        return PM_ERROR_DEVICE_LOST;
    return PM_ERROR_INVALID_STATE;
}

[[nodiscard]] PmResult pm_error(HRESULT result) noexcept {
    const PmResult converted = translate_hresult(result);
    switch (converted) {
    case PM_SUCCESS: return PM_SUCCESS;
    case PM_ERROR_OUT_OF_DEVICE_MEMORY: return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    case PM_ERROR_INVALID_ARGUMENT: return PM_ERROR_INVALID_ARGUMENT;
    case PM_ERROR_DEVICE_LOST: return PM_ERROR_DEVICE_LOST;
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
    struct Range { std::size_t offset{}, size{}; };
    struct Allocation {
        void *host{};
        std::size_t size{};
        ComPtr<ID3D12Resource> storage;
        ComPtr<ID3D12Resource> readback;
        std::vector<std::byte> host_baseline;
        std::byte *readback_data{};
        std::uint32_t descriptor{invalid_descriptor};
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
        std::vector<Range> dirty;
        std::size_t host_extent{};
        Range mapped{};
        bool gpu_written{};
    };

    enum class AuxiliaryKind : std::uint8_t { meshes };
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
    ComPtr<ID3D12Resource> copy_scratch;
    std::size_t copy_scratch_size{};
    D3D12_RESOURCE_STATES copy_scratch_state{D3D12_RESOURCE_STATE_COPY_DEST};
    std::byte *constant_data{};
    std::size_t constant_capacity{16U * 1024U * 1024U};
    std::size_t constant_offset{};
    std::vector<Allocation> allocations;
    std::vector<Auxiliary> auxiliary;
    std::vector<ComPtr<ID3D12Resource>> transient_uploads;
    bool open{};
    bool host_scan_pending{true};
    PmResult last_error{PM_SUCCESS};
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
        if (upload_data != nullptr) {
            result = device->CreateCommittedResource(
                &upload_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
            if (FAILED(result)) return result;
        }
        result = device->CreateCommittedResource(
            &readback_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
        if (FAILED(result)) return result;
        D3D12_RANGE no_read{0U, 0U};
        if (upload_data != nullptr) {
            result = upload->Map(0U, &no_read, reinterpret_cast<void **>(upload_data));
            if (FAILED(result)) return result;
        }
        D3D12_RANGE read_range{0U, staging_desc.Width};
        result = readback->Map(
            0U, &read_range, reinterpret_cast<void **>(readback_data));
        if (FAILED(result)) return result;
        return slang.create_raw_buffer_view(storage.Get(), 0U,
                                             storage_desc.Width, descriptor);
    }

    [[nodiscard]] PmResult wait_submitted() noexcept {
        if (submitted_fence == 0U) return PM_SUCCESS;
        const PmResult waited = wait_fence(
            fence.Get(), submitted_fence, fence_event);
        return waited == PM_SUCCESS ? PM_SUCCESS : PM_ERROR_DEVICE_LOST;
    }

    [[nodiscard]] PmResult begin() noexcept {
        if (open) return flush_uploads();
        const PmResult waited = wait_submitted();
        if (waited != PM_SUCCESS) return last_error = waited;
        transient_uploads.clear();
        constant_offset = 0U;
        HRESULT result = allocator->Reset();
        if (SUCCEEDED(result)) result = commands->Reset(allocator.Get(), nullptr);
        if (FAILED(result)) return last_error = translate_hresult(result);
        open = true;
        // Host metadata remains writable between submissions. Compare only
        // registered host extents; unchanged GPU state is never uploaded.
        if (host_scan_pending) for (auto &item : allocations) {
            const auto *host = static_cast<const std::byte *>(item.host);
            for (std::size_t chunk = 0U; chunk < item.host_extent; chunk += 4096U) {
                const auto end = std::min(chunk + 4096U, item.host_extent);
                if (std::memcmp(host + chunk, item.host_baseline.data() + chunk, end - chunk) == 0)
                    continue;
                for (std::size_t index = chunk; index < end;) {
                    if (host[index] == item.host_baseline[index]) { ++index; continue; }
                    const auto first = index++;
                    while (index < end && host[index] != item.host_baseline[index]) ++index;
                    try { item.dirty.push_back({first, index - first}); }
                    catch (...) { return last_error = PM_ERROR_OUT_OF_HOST_MEMORY; }
                }
            }
        }
        host_scan_pending = false;
        return flush_uploads();
    }

    [[nodiscard]] PmResult flush_uploads() noexcept {
        for (auto &item : allocations) {
            if (item.dirty.empty()) continue;
            std::sort(item.dirty.begin(), item.dirty.end(),
                [](Range a, Range b) { return a.offset < b.offset; });
            std::size_t merged = 0U;
            for (const auto range : item.dirty) {
                if (merged && range.offset <= item.dirty[merged - 1U].offset +
                                                item.dirty[merged - 1U].size) {
                    auto &previous = item.dirty[merged - 1U];
                    previous.size = std::max(previous.offset + previous.size,
                                             range.offset + range.size) - previous.offset;
                } else item.dirty[merged++] = range;
            }
            item.dirty.resize(merged);
            std::size_t staging_size{};
            for (const auto range : item.dirty) staging_size += range.size;
            // One immutable staging allocation per batch, not per changed byte
            // run. Disjoint CPU writes must not copy stale GPU-owned gaps.
            ComPtr<ID3D12Resource> staging;
            const auto heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
            const auto description = buffer_description(staging_size, D3D12_RESOURCE_FLAG_NONE);
            auto result = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                &description, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&staging));
            if (FAILED(result)) return last_error = translate_hresult(result);
            void *mapped{};
            const D3D12_RANGE no_read{0U, 0U};
            result = staging->Map(0U, &no_read, &mapped);
            if (FAILED(result)) return last_error = translate_hresult(result);
            try { transient_uploads.push_back(staging); }
            catch (...) { staging->Unmap(0U, nullptr); return last_error = PM_ERROR_OUT_OF_HOST_MEMORY; }
            std::size_t staging_offset{};
            transition(item.storage.Get(), item.state, D3D12_RESOURCE_STATE_COPY_DEST);
            for (const auto range : item.dirty) {
                const auto *source = static_cast<const std::byte *>(item.host) + range.offset;
                std::memcpy(static_cast<std::byte *>(mapped) + staging_offset, source, range.size);
                std::memcpy(item.host_baseline.data() + range.offset, source, range.size);
                commands->CopyBufferRegion(item.storage.Get(), range.offset,
                    staging.Get(), staging_offset, range.size);
                staging_offset += range.size;
            }
            staging->Unmap(0U, nullptr);
            transition(item.storage.Get(), item.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            item.dirty.clear();
        }
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult submit(std::uint64_t &value) noexcept {
        if (!open) {
            const PmResult started = begin();
            if (started != PM_SUCCESS) return started;
        }
        const auto uploaded = flush_uploads();
        if (uploaded != PM_SUCCESS) return uploaded;
        HRESULT result = commands->Close();
        if (FAILED(result)) return last_error = translate_hresult(result);
        ID3D12CommandList *lists[]{commands.Get()};
        queue->ExecuteCommandLists(1U, lists);
        value = next_fence++;
        result = queue->Signal(fence.Get(), value);
        if (FAILED(result)) return last_error = translate_hresult(result);
        submitted_fence = value;
        open = false;
        return PM_SUCCESS;
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

    [[nodiscard]] PmResult download(Allocation &item, Range range, void *destination) noexcept {
        const auto started = begin();
        if (started != PM_SUCCESS) return started;
        transition(item.storage.Get(), item.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        commands->CopyBufferRegion(item.readback.Get(), range.offset,
                                   item.storage.Get(), range.offset, range.size);
        transition(item.storage.Get(), item.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        std::uint64_t value{};
        const auto submitted = submit(value);
        if (submitted != PM_SUCCESS) return submitted;
        const PmResult waited = wait_submitted();
        if (waited != PM_SUCCESS) return waited;
        const auto *source = item.readback_data + range.offset;
        std::memmove(destination, source, range.size);
        if (destination == static_cast<std::byte *>(item.host) + range.offset)
            std::memcpy(item.host_baseline.data() + range.offset, source, range.size);
        return PM_SUCCESS;
    }

    void release_auxiliary() noexcept {
        for (auto &item : auxiliary) {
            if (item.upload && item.upload_data != nullptr)
                item.upload->Unmap(0U, nullptr);
            if (item.readback && item.readback_data != nullptr)
                item.readback->Unmap(0U, nullptr);
            slang.release_descriptor(item.descriptor);
        }
        auxiliary.clear();
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
    // UPLOAD mappings may be write-combined: never read them for CPU dirty
    // tracking. Keep the last host values in ordinary cacheable memory.
    try { allocation.host_baseline.resize(allocation.size); }
    catch (...) { _aligned_free(shadow); return PM_ERROR_OUT_OF_HOST_MEMORY; }
    ComPtr<ID3D12Resource> unused_upload;
    const HRESULT result = impl_->create_buffer_set(
        allocation.size, allocation.storage, unused_upload,
        allocation.readback, nullptr,
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
    return PM_SUCCESS;
}

void ScheduleContext::free(void *host) noexcept {
    if (host == nullptr) return;
    (void)synchronize();
    const auto found = std::find_if(
        impl_->allocations.begin(), impl_->allocations.end(),
        [host](const Impl::Allocation &item) { return item.host == host; });
    if (found == impl_->allocations.end()) return;
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
    return write_host(destination, size);
}

PmResult ScheduleContext::wait_idle() noexcept {
    return synchronize();
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
        impl_->last_error = PM_ERROR_INVALID_ARGUMENT;
        return {};
    }
    const auto byte_offset = reinterpret_cast<std::uintptr_t>(pointer) -
        reinterpret_cast<std::uintptr_t>(allocation->host);
    if (byte_offset > std::numeric_limits<std::uint32_t>::max()) {
        impl_->last_error = PM_ERROR_INVALID_ARGUMENT;
        return {};
    }
    allocation->gpu_written = true;
    return {allocation->descriptor, static_cast<std::uint32_t>(byte_offset), 1U};
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
    if (impl_->begin() != PM_SUCCESS) return {};
    std::vector<GraphicsTriangleMeshResource> translated;
    try {
        translated.resize(required);
    } catch (const std::bad_alloc &) {
        impl_->last_error = PM_ERROR_OUT_OF_DEVICE_MEMORY;
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
        impl_->last_error = translate_hresult(created);
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
        impl_->last_error = PM_ERROR_OUT_OF_DEVICE_MEMORY;
        return {};
    }
    return result;
}

PmResult ScheduleContext::dispatch(
    std::string_view module, std::string_view entry, device::DispatchSize grid, device::DispatchSize block,
    std::span<const std::byte> entry_constants) noexcept {
    active_context = this;
    if (!impl_->slang.matches_group_size(module, entry, block.x, block.y, block.z)) {
        std::fprintf(stderr, "D3D12 Slang launch size mismatch: %.*s.%.*s (%u,%u,%u)\n",
            static_cast<int>(module.size()), module.data(),
            static_cast<int>(entry.size()), entry.data(), block.x, block.y, block.z);
        return impl_->last_error = PM_ERROR_INVALID_ARGUMENT;
    }
    const PmResult started = impl_->begin();
    if (started != PM_SUCCESS) return started;
    const DispatchConstants global{
        grid.x, grid.y, grid.z, block.x};
    const auto global_address = impl_->reserve_constants(&global, sizeof(global));
    const auto entry_address = impl_->reserve_constants(
        entry_constants.data(), entry_constants.size());
    if (global_address == 0U || entry_address == 0U)
        return impl_->last_error = PM_ERROR_OUT_OF_DEVICE_MEMORY;
    const HRESULT result = impl_->slang.record_dispatch(
        impl_->commands.Get(), module, entry, global_address, entry_address,
        grid.x, grid.y, grid.z);
    if (FAILED(result)) return impl_->last_error = translate_hresult(result);
    impl_->uav_barrier();
    return PM_SUCCESS;
}

PmResult ScheduleContext::copy_device(
    void *destination, const void *source, std::size_t size) noexcept {
    if (size == 0U) return PM_SUCCESS;
    auto *target = impl_->find(destination, size);
    auto *input = impl_->find(source, size);
    if (target == nullptr || input == nullptr)
        return impl_->last_error = PM_ERROR_INVALID_ARGUMENT;
    const PmResult started = impl_->begin();
    if (started != PM_SUCCESS) return started;
    const auto target_offset = reinterpret_cast<std::uintptr_t>(destination) -
        reinterpret_cast<std::uintptr_t>(target->host);
    const auto source_offset = reinterpret_cast<std::uintptr_t>(source) -
        reinterpret_cast<std::uintptr_t>(input->host);
    if (target == input) {
        if (target_offset == source_offset) return PM_SUCCESS;
        // Legacy D3D12 barriers cannot put a buffer in COPY_SOURCE and
        // COPY_DEST simultaneously. Arena-to-arena moves use GPU scratch,
        // including overlapping ranges; no host round trip is involved.
        if (size > impl_->copy_scratch_size) {
            ComPtr<ID3D12Resource> replacement;
            const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
            const auto description = buffer_description(size, D3D12_RESOURCE_FLAG_NONE);
            const auto created = impl_->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                &description, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&replacement));
            if (FAILED(created)) return impl_->last_error = translate_hresult(created);
            try { if (impl_->copy_scratch) impl_->transient_uploads.push_back(impl_->copy_scratch); }
            catch (...) { return impl_->last_error = PM_ERROR_OUT_OF_HOST_MEMORY; }
            impl_->copy_scratch = std::move(replacement);
            impl_->copy_scratch_size = size;
            impl_->copy_scratch_state = D3D12_RESOURCE_STATE_COPY_DEST;
        }
        impl_->transition(input->storage.Get(), input->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        impl_->transition(impl_->copy_scratch.Get(), impl_->copy_scratch_state, D3D12_RESOURCE_STATE_COPY_DEST);
        impl_->commands->CopyBufferRegion(impl_->copy_scratch.Get(), 0U, input->storage.Get(), source_offset, size);
        impl_->transition(impl_->copy_scratch.Get(), impl_->copy_scratch_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        impl_->transition(target->storage.Get(), target->state, D3D12_RESOURCE_STATE_COPY_DEST);
        impl_->commands->CopyBufferRegion(target->storage.Get(), target_offset, impl_->copy_scratch.Get(), 0U, size);
        impl_->transition(target->storage.Get(), target->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        target->gpu_written = true;
        return PM_SUCCESS;
    }
    impl_->transition(input->storage.Get(), input->state,
                      D3D12_RESOURCE_STATE_COPY_SOURCE);
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->commands->CopyBufferRegion(
        target->storage.Get(), target_offset, input->storage.Get(),
        source_offset, size);
    target->gpu_written = true;
    impl_->transition(input->storage.Get(), input->state,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return PM_SUCCESS;
}

PmResult ScheduleContext::clear_device(
    void *destination, int value, std::size_t size) noexcept {
    if (size == 0U) return PM_SUCCESS;
    auto *target = impl_->find(destination, size);
    if (target == nullptr)
        return impl_->last_error = PM_ERROR_INVALID_ARGUMENT;
    const auto offset = reinterpret_cast<std::uintptr_t>(destination) -
        reinterpret_cast<std::uintptr_t>(target->host);
    const PmResult started = impl_->begin();
    if (started != PM_SUCCESS) return started;
    ComPtr<ID3D12Resource> upload;
    const auto upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    const auto description = buffer_description(size, D3D12_RESOURCE_FLAG_NONE);
    HRESULT created = impl_->device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &description,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    if (FAILED(created)) return impl_->last_error = translate_hresult(created);
    void *mapped{};
    D3D12_RANGE no_read{0U, 0U};
    created = upload->Map(0U, &no_read, &mapped);
    if (FAILED(created)) return impl_->last_error = translate_hresult(created);
    std::memset(mapped, value, size);
    upload->Unmap(0U, nullptr);
    try {
        impl_->transient_uploads.push_back(upload);
    } catch (const std::bad_alloc &) {
        return impl_->last_error = PM_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    impl_->commands->CopyBufferRegion(target->storage.Get(), offset,
                                      upload.Get(), 0U, size);
    target->gpu_written = true;
    impl_->transition(target->storage.Get(), target->state,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl_->uav_barrier(target->storage.Get());
    return PM_SUCCESS;
}

PmResult ScheduleContext::synchronize() noexcept {
    active_context = this;
    if (impl_->open) {
        std::uint64_t ignored{};
        const PmResult submitted = impl_->submit(ignored);
        if (submitted != PM_SUCCESS) return submitted;
    }
    const auto waited = impl_->wait_submitted();
    if (waited != PM_SUCCESS) return waited;
    // Only explicitly mapped application outputs become host coherent.
    // Simulation arenas stay resident; waiting never invalidates all of them.
    for (auto &item : impl_->allocations) {
        if (!item.gpu_written || item.mapped.size == 0U) continue;
        const auto result = impl_->download(item, item.mapped,
            static_cast<std::byte *>(item.host) + item.mapped.offset);
        if (result != PM_SUCCESS) return result;
        item.gpu_written = false;
    }
    impl_->release_auxiliary();
    return PM_SUCCESS;
}

PmResult ScheduleContext::record(FenceEvent *event) noexcept {
    if (event == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    active_context = this;
    std::uint64_t value{};
    const PmResult submitted = impl_->submit(value);
    if (submitted != PM_SUCCESS) return submitted;
    event->stream = this;
    event->fence_value = value;
    event->recorded = std::chrono::steady_clock::now();
    event->ready = false;
    return PM_SUCCESS;
}

PmResult ScheduleContext::wait(FenceEvent *event) noexcept {
    if (event == nullptr || event->stream != this)
        return PM_ERROR_INVALID_ARGUMENT;
    const PmResult waited = wait_fence(
        impl_->fence.Get(), event->fence_value, impl_->fence_event);
    if (waited != PM_SUCCESS) return PM_ERROR_DEVICE_LOST;
    event->ready = true;
    return synchronize();
}

PmResult ScheduleContext::query(FenceEvent *event) noexcept {
    if (event == nullptr || event->stream != this)
        return PM_ERROR_INVALID_ARGUMENT;
    if (event->ready || impl_->fence->GetCompletedValue() >= event->fence_value) {
        event->ready = true;
        return PM_SUCCESS;
    }
    return PM_NOT_READY;
}

PmResult ScheduleContext::error(bool take) noexcept {
    const PmResult result = impl_->last_error;
    if (take) impl_->last_error = PM_SUCCESS;
    return result;
}

ScheduleContext *context(void *stream) noexcept {
    return stream == nullptr ? active_context
                             : static_cast<ScheduleContext *>(stream);
}

PmResult ScheduleContext::copy(void *destination, const void *source,
    PmDeviceSize size, device::CopyDirection direction) noexcept {
    if (size == 0U) return PM_SUCCESS;
    if (size > SIZE_MAX || destination == nullptr || source == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (direction == device::CopyDirection::device_to_device ||
        (direction == device::CopyDirection::automatic &&
         impl_->find(destination, size) && impl_->find(source, size)))
        return copy_device(destination, source, static_cast<std::size_t>(size));
    auto *input = impl_->find(source, size);
    if (direction == device::CopyDirection::device_to_host ||
        (direction == device::CopyDirection::automatic && input)) {
        if (!input) return PM_ERROR_INVALID_ARGUMENT;
        const auto offset = reinterpret_cast<std::uintptr_t>(source) -
            reinterpret_cast<std::uintptr_t>(input->host);
        return impl_->download(*input, {offset, static_cast<std::size_t>(size)}, destination);
    }
    return copy_host(destination, source, size);
}

PmResult ScheduleContext::read_host(void *pointer, PmDeviceSize size) noexcept {
    if (size == 0U) return PM_SUCCESS;
    auto *allocation = impl_->find(pointer, size);
    if (!allocation) return PM_ERROR_INVALID_ARGUMENT;
    const auto offset = reinterpret_cast<std::uintptr_t>(pointer) -
        reinterpret_cast<std::uintptr_t>(allocation->host);
    return impl_->download(*allocation, {offset, static_cast<std::size_t>(size)}, pointer);
}

PmResult ScheduleContext::write_host(void *pointer, PmDeviceSize size) noexcept {
    if (size == 0U) return PM_SUCCESS;
    auto *allocation = impl_->find(pointer, size);
    if (!allocation) return PM_ERROR_INVALID_ARGUMENT;
    const auto offset = reinterpret_cast<std::uintptr_t>(pointer) -
        reinterpret_cast<std::uintptr_t>(allocation->host);
    try { allocation->dirty.push_back({offset, static_cast<std::size_t>(size)}); }
    catch (...) { return impl_->last_error = PM_ERROR_OUT_OF_HOST_MEMORY; }
    allocation->host_extent = std::max(allocation->host_extent,
                                      offset + static_cast<std::size_t>(size));
    impl_->host_scan_pending = true;
    return PM_SUCCESS;
}

PmResult ScheduleContext::map_host(void *pointer, PmDeviceSize size) noexcept {
    auto *allocation = impl_->find(pointer, size);
    if (!allocation) return PM_ERROR_INVALID_ARGUMENT;
    const auto result = allocation->gpu_written ? read_host(pointer, size) : write_host(pointer, size);
    if (result != PM_SUCCESS) return result;
    const auto offset = reinterpret_cast<std::uintptr_t>(pointer) -
        reinterpret_cast<std::uintptr_t>(allocation->host);
    allocation->mapped = {offset, static_cast<std::size_t>(size)};
    return PM_SUCCESS;
}

void ScheduleContext::unmap_host(void *pointer) noexcept {
    if (auto *allocation = impl_->find(pointer)) allocation->mapped = {};
}

void ScheduleContext::prepare_host_access() noexcept {
    impl_->host_scan_pending = true;
}

} // namespace parallel_mater::d3d12
