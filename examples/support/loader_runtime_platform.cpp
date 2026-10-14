// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/runtime_platform.hpp>

#include <parallel_mater/parallel_mater.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <sys/sysctl.h>
#include <sys/utsname.h>

namespace parallel_mater::gallery {
namespace {

class LoaderInstance {
  public:
    LoaderInstance() {
        PmInstanceCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.apiVersion = PM_API_VERSION_0_2;
        info.pApplicationName = "parallel-mater-gallery-device-query";
        if (pmCreateInstance(&info, nullptr, &value_) != PM_SUCCESS)
            value_ = nullptr;
    }
    ~LoaderInstance() {
        if (value_ != nullptr) pmDestroyInstance(value_);
    }
    LoaderInstance(const LoaderInstance &) = delete;
    LoaderInstance &operator=(const LoaderInstance &) = delete;
    [[nodiscard]] PmInstance get() const noexcept { return value_; }

  private:
    PmInstance value_{};
};

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

[[nodiscard]] PmPhysicalDevice first_physical_device(PmInstance instance) {
    if (instance == nullptr) return nullptr;
    std::uint32_t count = 0U;
    if (pmEnumeratePhysicalDevices(instance, &count, nullptr) != PM_SUCCESS ||
        count == 0U)
        return nullptr;
    std::vector<PmPhysicalDevice> devices(count);
    if (pmEnumeratePhysicalDevices(instance, &count, devices.data()) != PM_SUCCESS ||
        count == 0U)
        return nullptr;
    return devices.front();
}

[[nodiscard]] std::uint64_t device_memory(PmPhysicalDevice physical) noexcept {
    if (physical == nullptr) return 0U;
    PmMemoryProperties memory{};
    pmGetPhysicalDeviceMemoryProperties(physical, &memory);
    std::uint64_t result = 0U;
    for (std::uint32_t index = 0U;
         index < std::min(memory.heapCount, 16U); ++index)
        result = std::max(result, static_cast<std::uint64_t>(memory.heaps[index].size));
    return result;
}

} // namespace

HardwareIdentity platform_hardware_identity() {
    HardwareIdentity result;
    result.machine_model = system_string("hw.model");
    result.cpu_model = system_string("machdep.cpu.brand_string");
    if (result.cpu_model.empty()) result.cpu_model = system_string("hw.chip");
    if (result.cpu_model.empty()) result.cpu_model = "Apple Silicon";

    struct utsname system {};
    if (uname(&system) == 0)
        result.operating_system =
            std::string(system.sysname) + ' ' + system.release;
    else
        result.operating_system = "macOS";

    LoaderInstance instance;
    const PmPhysicalDevice physical = first_physical_device(instance.get());
    if (physical == nullptr) return result;
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(physical, &properties);
    result.gpu_model = properties.deviceName;
    result.gpu_variant = "vendor " + std::to_string(properties.vendorId) +
        ", device " + std::to_string(properties.deviceId);
    result.memory_bytes = device_memory(physical);
    result.backend = "parallel-mater-loader";
    result.driver = "driver " + std::to_string(properties.driverVersion);
    result.power_mode = "driver-default";
    return result;
}

std::uint64_t platform_available_device_memory() noexcept {
    LoaderInstance instance;
    return device_memory(first_physical_device(instance.get()));
}

} // namespace parallel_mater::gallery
