// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater/pm_layer.h>
#include <parallel_mater/pm_layer_args.h>
#include <cstdio>
#include <cstring>

extern "C" PmResult PMAPI_CALL pm_layerNegotiateLoaderInterfaceVersion(uint32_t *);
extern "C" PmResult PMAPI_CALL pm_layerPreCall(const PmLayerCallInfo *);
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%d: %s\n", __LINE__, #x); return 1; } } while(false)

int main(int argc, char **argv) {
    const bool validation = argc < 2 || std::strcmp(argv[1], "plain") != 0;
    uint32_t version = 1;
    CHECK(pm_layerNegotiateLoaderInterfaceVersion(&version) == PM_ERROR_INCOMPATIBLE_VERSION);
    version = 999;
    CHECK(pm_layerNegotiateLoaderInterfaceVersion(&version) == PM_SUCCESS);
    CHECK(version == 2);
    PmLayerArgs_pmMapMemory bad_arguments{};
    PmLayerCallInfo bad_call{"pmMapMemory", &bad_arguments, 0, &bad_arguments};
    CHECK(pm_layerPreCall(&bad_call) == PM_ERROR_INVALID_ARGUMENT);

    PmInstanceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_2;
    info.enableValidation = validation ? PM_TRUE : PM_FALSE;
    PmInstance instance{};
    CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_SUCCESS);
    uint32_t count = 1;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    PmDevice device{}, other{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    CHECK(pmCreateDevice(physical, &device_info, &other) == PM_SUCCESS);
    const auto invalid = validation ? PM_ERROR_INVALID_ARGUMENT : PM_SUCCESS;
    const auto invalid_state = validation ? PM_ERROR_INVALID_STATE : PM_SUCCESS;

    PmMemoryAllocateInfo allocation{PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, 512, 1};
    PmDeviceMemory memory{};
    CHECK(pmAllocateMemory(device, &allocation, &memory) == PM_SUCCESS);
    PmBufferCreateInfo buffer_info{PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 256,
                                  PM_BUFFER_TRANSFER_SRC_BIT, 0};
    PmBuffer buffer{};
    CHECK(pmCreateBuffer(device, &buffer_info, &buffer) == PM_SUCCESS);
    PmMemoryRequirements requirements{};
    pmGetBufferMemoryRequirements(device, buffer, &requirements);
    CHECK(requirements.size == 256);
    CHECK(pmBindBufferMemory(other, buffer, memory, 0) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, 1) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, 512) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, UINT64_MAX - 1) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, 256) == PM_SUCCESS);
    CHECK(pmBindBufferMemory(device, buffer, memory, 0) == invalid_state);

    void *mapped{};
    CHECK(pmMapMemory(other, memory, 0, 1, &mapped) == invalid);
    CHECK(pmMapMemory(device, memory, UINT64_MAX - 2, 8, &mapped) == invalid);
    CHECK(pmMapMemory(device, memory, 0, PM_WHOLE_SIZE, &mapped) == PM_SUCCESS);
    CHECK(pmMapMemory(device, memory, 0, 1, &mapped) == invalid_state);
    PmMappedMemoryRange range{memory, 513, 1};
    CHECK(pmFlushMappedMemoryRanges(device, 1, &range) == invalid);
    CHECK(pmInvalidateMappedMemoryRanges(device, 1, &range) == invalid);
    range = {memory, 0, PM_WHOLE_SIZE};
    CHECK(pmFlushMappedMemoryRanges(device, 1, &range) == PM_SUCCESS);
    CHECK(pmInvalidateMappedMemoryRanges(device, 1, &range) == PM_SUCCESS);
    pmUnmapMemory(device, memory);
    CHECK(pmInvalidateMappedMemoryRanges(device, 1, &range) == invalid);

    allocation.allocationSize = 13; // driver failure must not mutate layer state
    PmDeviceMemory failed{};
    CHECK(pmAllocateMemory(device, &allocation, &failed) == PM_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK(!failed);
    if (validation) {
        PmBuffer failed_buffer{};
        CHECK(pmCreateBuffer(device, nullptr, &failed_buffer) == PM_ERROR_INVALID_ARGUMENT);
        pmFreeMemory(device, memory); // rejected: live buffer still bound
        CHECK(pmMapMemory(device, memory, 0, 1, &mapped) == PM_SUCCESS);
        pmUnmapMemory(device, memory);
    }
    pmDestroyBuffer(device, buffer);
    CHECK(pmBindBufferMemory(device, buffer, memory, 0) == invalid);
    pmFreeMemory(device, memory);
    CHECK(pmMapMemory(device, memory, 0, 1, &mapped) == invalid);
    pmDestroyDevice(other);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts(validation ? "PASS: layer rejects invalid calls before permissive vendor driver" :
                           "PASS: permissive driver accepts negative controls without layer");
}
