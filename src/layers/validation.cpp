// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_layer.h>
#include <parallel_mater/pm_layer_args.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

#if defined(_WIN32)
#define PM_LAYER_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_LAYER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct Memory {
    PmDeviceSize size{};
    uint32_t type{};
    bool mapped{};
};
struct Buffer {
    PmDeviceSize size{};
    PmDeviceMemory memory{};
    bool bound{};
    PmMemoryRequirements requirements{};
};
struct Device {
    std::unordered_map<PmDeviceMemory, Memory> memories;
    std::unordered_map<PmBuffer, Buffer> buffers;
};
std::mutex mutex;
std::unordered_map<PmDevice, Device> devices;
bool tracking_available = true;

bool fits(PmDeviceSize offset, PmDeviceSize size, PmDeviceSize capacity) {
    return offset <= capacity && size <= capacity - offset;
}
bool range_fits(PmDeviceSize offset, PmDeviceSize size, PmDeviceSize capacity) {
    return offset <= capacity && (size == PM_WHOLE_SIZE || fits(offset, size, capacity));
}
Memory *memory(PmDevice device, PmDeviceMemory value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto m = d->second.memories.find(value);
    return m == d->second.memories.end() ? nullptr : &m->second;
}
Buffer *buffer(PmDevice device, PmBuffer value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto b = d->second.buffers.find(value);
    return b == d->second.buffers.end() ? nullptr : &b->second;
}
template<class T> PmResult validate(const T &) { return PM_SUCCESS; }
template<class T> void record(const T &) {}

PmResult validate(const PmLayerArgs_pmAllocateMemory &a) {
    return a.pInfo && a.pMemory &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO &&
        !a.pInfo->pNext && a.pInfo->allocationSize != 0
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmAllocateMemory &a) {
    if (*a.pMemory)
        devices[a.device].memories[*a.pMemory] = {a.pInfo->allocationSize, a.pInfo->memoryTypeIndex};
}
PmResult validate(const PmLayerArgs_pmCreateBuffer &a) {
    return a.pInfo && a.pBuffer &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO &&
        !a.pInfo->pNext && a.pInfo->size != 0
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCreateBuffer &a) {
    if (*a.pBuffer) devices[a.device].buffers[*a.pBuffer] = {a.pInfo->size};
}
void record(const PmLayerArgs_pmImportNativeBuffer &a) {
    if (a.pInfo && a.pBuffer && *a.pBuffer)
        devices[a.device].buffers[*a.pBuffer] = {a.pInfo->size, nullptr, true};
}
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
void record(const PmLayerArgs_pmImportVulkanBufferEXT &a) {
    if (a.pInfo && a.pBuffer && *a.pBuffer)
        devices[a.device].buffers[*a.pBuffer] = {a.pInfo->size, nullptr, true};
}
#endif
PmResult validate(const PmLayerArgs_pmGetBufferMemoryRequirements &a) {
    return a.pRequirements && (!tracking_available || buffer(a.device, a.buffer))
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmGetBufferMemoryRequirements &a) {
    if (auto *b = buffer(a.device, a.buffer)) b->requirements = *a.pRequirements;
}
PmResult validate(const PmLayerArgs_pmBindBufferMemory &a) {
    if (!tracking_available) return PM_SUCCESS;
    const auto *b = buffer(a.device, a.buffer);
    const auto *m = memory(a.device, a.memory);
    if (!b || !m) return PM_ERROR_INVALID_ARGUMENT;
    if (b->bound) return PM_ERROR_INVALID_STATE;
    const auto &r = b->requirements;
    const auto size = r.size ? r.size : b->size;
    if (!fits(a.offset, size, m->size) ||
        (r.alignment && a.offset % r.alignment != 0) ||
        (r.memoryTypeBits && (m->type >= 32 || !(r.memoryTypeBits & (1U << m->type)))))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmBindBufferMemory &a) {
    if (auto *b = buffer(a.device, a.buffer)) { b->memory = a.memory; b->bound = true; }
}
PmResult validate(const PmLayerArgs_pmMapMemory &a) {
    if (!a.ppData) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *m = memory(a.device, a.memory);
    if (!m || !range_fits(a.offset, a.size, m->size)) return PM_ERROR_INVALID_ARGUMENT;
    return m->mapped ? PM_ERROR_INVALID_STATE : PM_SUCCESS;
}
void record(const PmLayerArgs_pmMapMemory &a) {
    if (auto *m = memory(a.device, a.memory)) m->mapped = true;
}
PmResult validate(const PmLayerArgs_pmUnmapMemory &a) {
    return !tracking_available || memory(a.device, a.memory)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmUnmapMemory &a) {
    if (auto *m = memory(a.device, a.memory)) m->mapped = false;
}
PmResult ranges(PmDevice device, uint32_t count, const PmMappedMemoryRange *values,
                bool require_mapped) {
    if (count && !values) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    for (uint32_t i = 0; i < count; ++i) {
        const auto *m = memory(device, values[i].memory);
        if (!m || (require_mapped && !m->mapped) ||
            !range_fits(values[i].offset, values[i].size, m->size))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}
PmResult validate(const PmLayerArgs_pmFlushMappedMemoryRanges &a) {
    return ranges(a.device, a.count, a.pRanges, false);
}
PmResult validate(const PmLayerArgs_pmInvalidateMappedMemoryRanges &a) {
    return ranges(a.device, a.count, a.pRanges, true);
}
PmResult validate(const PmLayerArgs_pmDestroyBuffer &a) {
    return !a.buffer || !tracking_available || buffer(a.device, a.buffer)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmDestroyBuffer &a) {
    if (const auto d = devices.find(a.device); d != devices.end())
        d->second.buffers.erase(a.buffer);
}
PmResult validate(const PmLayerArgs_pmFreeMemory &a) {
    if (!a.memory || !tracking_available) return PM_SUCCESS;
    if (!memory(a.device, a.memory)) return PM_ERROR_INVALID_ARGUMENT;
    for (const auto &[handle, b] : devices.at(a.device).buffers) {
        (void)handle;
        if (b.memory == a.memory) return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmFreeMemory &a) {
    if (const auto d = devices.find(a.device); d != devices.end())
        d->second.memories.erase(a.memory);
}
void record(const PmLayerArgs_pmDestroyDevice &a) { devices.erase(a.device); }

template<class Function>
PmResult dispatch(const PmLayerCallInfo &call, Function function) {
    if (!call.pCommandName || !call.pCommandName[0] || !call.primaryHandle || !call.pArguments)
        return PM_ERROR_INVALID_ARGUMENT;
#include "layer_dispatch.inc"
    // An older layer must allow commands added by a newer compatible loader.
    return PM_SUCCESS;
}
} // namespace

PM_LAYER_EXPORT PmResult PMAPI_CALL
pm_layerNegotiateLoaderInterfaceVersion(uint32_t *version) {
    if (!version) return PM_ERROR_INVALID_ARGUMENT;
    // A v1 loader cannot supply the typed payload required by this layer.
    if (*version < 2U) return PM_ERROR_INCOMPATIBLE_VERSION;
    *version = PM_LAYER_INTERFACE_VERSION;
    return PM_SUCCESS;
}

PM_LAYER_EXPORT PmResult PMAPI_CALL pm_layerPreCall(const PmLayerCallInfo *call) {
    if (!call) return PM_ERROR_INVALID_ARGUMENT;
    std::lock_guard lock(mutex);
    return dispatch(*call, [](const auto &a) { return validate(a); });
}

PM_LAYER_EXPORT void PMAPI_CALL pm_layerPostCall(
    const PmLayerCallInfo *call, PmResult result) {
    if (!call || result != PM_SUCCESS) return;
    std::lock_guard lock(mutex);
    if (!tracking_available) return;
    try {
        (void)dispatch(*call, [](const auto &a) { record(a); return PM_SUCCESS; });
    } catch (const std::bad_alloc &) {
        // A diagnostic must not unwind across the C ABI or turn a successful
        // resource creation into a failure that leaks the driver's resource.
        devices.clear();
        tracking_available = false;
        std::fputs("Parallel Mater validation: host memory exhausted; object tracking disabled\n", stderr);
    }
}
