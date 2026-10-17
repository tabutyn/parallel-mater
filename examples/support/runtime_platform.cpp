// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/runtime_platform.hpp>

#include <parallel_mater/parallel_mater.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fstream>
#include <sys/utsname.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/utsname.h>
#endif

namespace parallel_mater::gallery {
namespace {

struct SelectedDevice {
    PmInstance instance{};
    PmPhysicalDevice physical{};
    PmPhysicalDeviceFeatures features{};
    PmDevice device{};

    ~SelectedDevice() {
        if (device != nullptr) pmDestroyDevice(device);
        if (instance != nullptr) pmDestroyInstance(instance);
    }
};

[[nodiscard]] bool exposes_backend(PmPhysicalDevice physical,
                                   PmNativeBackend backend) noexcept {
    PmNativeInteropProperties properties{};
    return pmGetPhysicalDeviceInteropProperties(
               physical, backend, &properties) == PM_SUCCESS;
}

bool open_selected_device(SelectedDevice &selected,
                          std::optional<PmNativeBackend> backend) {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.pApplicationName = "parallel-mater-gallery-platform";
    if (pmCreateInstance(&instance_info, nullptr, &selected.instance) !=
        PM_SUCCESS)
        return false;
    std::uint32_t count{};
    if (pmEnumeratePhysicalDevices(selected.instance, &count, nullptr) !=
            PM_SUCCESS || count == 0U)
        return false;
    std::vector<PmPhysicalDevice> physicals(count);
    if (pmEnumeratePhysicalDevices(
            selected.instance, &count, physicals.data()) != PM_SUCCESS)
        return false;
    selected.physical = backend ? nullptr : physicals.front();
    for (PmPhysicalDevice candidate : physicals) {
        PmPhysicalDeviceFeatures features{};
        pmGetPhysicalDeviceFeatures(candidate, &features);
        if (features.rigidBodies == PM_TRUE &&
            (!backend || exposes_backend(candidate, *backend))) {
            selected.physical = candidate;
            selected.features = features;
            break;
        }
    }
    if (selected.physical == nullptr) return false;
    if (selected.features.rigidBodies != PM_TRUE)
        pmGetPhysicalDeviceFeatures(selected.physical, &selected.features);
    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue;
    device_info.enabledFeatures = selected.features;
    return pmCreateDevice(selected.physical, &device_info, &selected.device) ==
        PM_SUCCESS;
}

#if defined(__APPLE__)
[[nodiscard]] std::string system_string(const char *name) {
    std::size_t length = 0U;
    if (sysctlbyname(name, nullptr, &length, nullptr, 0U) != 0 || length < 2U)
        return {};
    std::string result(length, '\0');
    if (sysctlbyname(name, result.data(), &length, nullptr, 0U) != 0)
        return {};
    while (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
}
#endif

} // namespace

std::vector<GalleryBackendOption> platform_gallery_backends() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.pApplicationName = "parallel-mater-gallery-backends";
    PmInstance instance{};
    if (pmCreateInstance(&instance_info, nullptr, &instance) != PM_SUCCESS)
        return {};
    struct InstanceOwner {
        PmInstance value{};
        ~InstanceOwner() { if (value != nullptr) pmDestroyInstance(value); }
    } owner{instance};
    std::uint32_t count{};
    if (pmEnumeratePhysicalDevices(instance, &count, nullptr) != PM_SUCCESS ||
        count == 0U)
        return {};
    std::vector<PmPhysicalDevice> physicals(count);
    if (pmEnumeratePhysicalDevices(instance, &count, physicals.data()) !=
        PM_SUCCESS)
        return {};

    constexpr std::array backends{
        PM_NATIVE_BACKEND_CUDA, PM_NATIVE_BACKEND_D3D12,
        PM_NATIVE_BACKEND_METAL, PM_NATIVE_BACKEND_VULKAN};
    std::vector<GalleryBackendOption> result;
    for (PmNativeBackend backend : backends) {
        for (PmPhysicalDevice physical : physicals) {
            if (!exposes_backend(physical, backend)) continue;
            PmPhysicalDeviceFeatures features{};
            pmGetPhysicalDeviceFeatures(physical, &features);
            if (features.rigidBodies != PM_TRUE) continue;
            PmPhysicalDeviceProperties properties{};
            pmGetPhysicalDeviceProperties(physical, &properties);
            result.push_back({backend, properties.deviceName, features});
            break;
        }
    }
    return result;
}

std::optional<PmPhysicalDeviceFeatures> platform_gallery_features(
    std::optional<PmNativeBackend> backend) {
    SelectedDevice selected;
    if (!open_selected_device(selected, backend)) return std::nullopt;
    return selected.features;
}

HardwareIdentity platform_hardware_identity(
    std::optional<PmNativeBackend> backend) {
    SelectedDevice selected;
    if (!open_selected_device(selected, backend)) return {};
    PmPhysicalDeviceProperties properties{};
    PmMemoryProperties memory{};
    PmNativeDeviceInfo native{};
    pmGetPhysicalDeviceProperties(selected.physical, &properties);
    pmGetPhysicalDeviceMemoryProperties(selected.physical, &memory);
    const PmResult native_result = pmGetNativeDevice(selected.device, &native);

    HardwareIdentity result;
    const char *machine = std::getenv("COMPUTERNAME");
    if (machine == nullptr) machine = std::getenv("HOSTNAME");
    result.machine_model = machine != nullptr ? machine : properties.deviceName;
    result.cpu_model = "unknown CPU";
#if defined(__linux__)
    struct utsname system {};
    if (uname(&system) == 0) {
        if (result.machine_model.empty()) result.machine_model = system.nodename;
        result.operating_system = std::string(system.sysname) + ' ' +
            system.release;
    } else {
        result.operating_system = "Linux";
    }
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (!line.starts_with("model name")) continue;
        const auto separator = line.find(':');
        if (separator != std::string::npos) {
            result.cpu_model = line.substr(separator + 1U);
            result.cpu_model.erase(
                0U, result.cpu_model.find_first_not_of(" \t"));
        }
        break;
    }
#elif defined(_WIN32)
    result.operating_system = "Windows";
    if (const char *processor = std::getenv("PROCESSOR_IDENTIFIER"))
        result.cpu_model = processor;
#elif defined(__APPLE__)
    result.machine_model = system_string("hw.model");
    result.cpu_model = system_string("machdep.cpu.brand_string");
    if (result.cpu_model.empty()) result.cpu_model = system_string("hw.chip");
    struct utsname system {};
    if (uname(&system) == 0)
        result.operating_system = std::string(system.sysname) + ' ' +
            system.release;
    else
        result.operating_system = "macOS";
#else
    result.operating_system = "unknown";
#endif
    result.gpu_model = properties.deviceName;
    result.gpu_variant = "loader device " +
        std::to_string(properties.vendorId) + ':' +
        std::to_string(properties.deviceId);
    for (std::uint32_t index = 0U; index < memory.heapCount; ++index)
        if ((memory.heaps[index].flags & PM_MEMORY_DEVICE_LOCAL_BIT) != 0U)
            result.memory_bytes = std::max(
                result.memory_bytes, memory.heaps[index].size);
    result.backend = native_result == PM_SUCCESS
        ? gallery_backend_name(native.backend) : "parallel-mater-loader";
    result.driver = "Parallel Mater driver " +
        std::to_string(properties.driverVersion);
    result.power_mode = "driver-default";
    return result;
}

std::uint64_t platform_available_device_memory(
    std::optional<PmNativeBackend> backend) noexcept {
    SelectedDevice selected;
    if (!open_selected_device(selected, backend)) return 0U;
    PmMemoryBudget budget{};
    if (pmGetDeviceMemoryBudget(selected.device, &budget) != PM_SUCCESS ||
        budget.heapCount == 0U)
        return 0U;
    return budget.budget[0] > budget.usage[0]
        ? budget.budget[0] - budget.usage[0] : 0U;
}

} // namespace parallel_mater::gallery
