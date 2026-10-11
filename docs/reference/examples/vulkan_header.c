/* Optional extension header must coexist with the Vulkan SDK in C and C++. */
#include "../api/parallel_mater_vulkan.h"

PmResult (PMAPI_CALL *const checked_import_device)(
    PmInstance, const PmVulkanDeviceImportInfo*, PmDevice*) =
    pmImportVulkanDeviceEXT;

PmResult (PMAPI_CALL *const checked_import_semaphore)(
    PmDevice, VkSemaphore, PmSemaphore*) = pmImportVulkanSemaphoreEXT;
