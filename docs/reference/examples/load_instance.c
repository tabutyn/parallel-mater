/* Compile-only example for the command API contract. */
#include "../api/parallel_mater.h"

PmResult create_instance_with_loader(
    PFN_pmGetInstanceProcAddr getProc,
    const PmInstanceCreateInfo* info,
    const PmAllocationCallbacks* allocator,
    PmInstance* instance)
{
    if (!instance) return PM_ERROR_INVALID_ARGUMENT;
    *instance = NULL;
    if (!getProc || !info) return PM_ERROR_INVALID_ARGUMENT;

    PFN_pmCreateInstance createInstance =
        (PFN_pmCreateInstance)getProc(NULL, "pmCreateInstance");
    PFN_pmEnumerateInstanceVersion enumerateVersion =
        (PFN_pmEnumerateInstanceVersion)getProc(NULL, "pmEnumerateInstanceVersion");
    if (!createInstance || !enumerateVersion)
        return PM_ERROR_FEATURE_NOT_PRESENT;

    uint32_t supportedVersion = 0;
    PmResult result = enumerateVersion(&supportedVersion);
    if (result != PM_SUCCESS) return result;
    if (supportedVersion < info->apiVersion)
        return PM_ERROR_INCOMPATIBLE_VERSION;
    return createInstance(info, allocator, instance);
}

PFN_pmQueueSubmit resolve_queue_submit(
    PFN_pmGetDeviceProcAddr getProc, PmDevice device)
{
    if (!getProc || !device) return NULL;
    return (PFN_pmQueueSubmit)getProc(device, "pmQueueSubmit");
}
