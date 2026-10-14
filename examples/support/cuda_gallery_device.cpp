// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace parallel_mater::gallery {

std::uint64_t gallery_device_available_memory() noexcept {
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess
        ? static_cast<std::uint64_t>(free_bytes)
        : 0U;
}

HardwareIdentity gallery_device_hardware_identity() {
    cudaDeviceProp properties{};
    int device = 0;
    int driver = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return {};
    cudaDriverGetVersion(&driver);

    HardwareIdentity result;
    const char *machine = std::getenv("COMPUTERNAME");
    if (machine == nullptr) machine = std::getenv("HOSTNAME");
    result.machine_model = machine != nullptr ? machine : "";
    result.cpu_model = "unknown CPU";
#if defined(__linux__)
    {
        std::ifstream product_name("/sys/devices/virtual/dmi/id/product_name");
        std::string value;
        if (std::getline(product_name, value)) {
            const auto last = value.find_last_not_of(" \t\r\n");
            if (last != std::string::npos)
                result.machine_model = value.substr(0U, last + 1U);
        }
    }
    struct utsname system {};
    const bool have_system_identity = uname(&system) == 0;
    if (result.machine_model.empty() && have_system_identity)
        result.machine_model = system.nodename;
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            const std::string prefix = "model name";
            if (!line.starts_with(prefix)) continue;
            const auto separator = line.find(':');
            if (separator != std::string::npos) {
                result.cpu_model = line.substr(separator + 1U);
                result.cpu_model.erase(
                    0U, result.cpu_model.find_first_not_of(" \t"));
            }
            break;
        }
    }
#elif defined(_WIN32)
    if (const char *processor = std::getenv("PROCESSOR_IDENTIFIER"))
        result.cpu_model = processor;
#endif
    if (result.machine_model.empty()) result.machine_model = properties.name;
    result.gpu_model = properties.name;
    result.gpu_variant = std::to_string(properties.multiProcessorCount) +
        " SM, compute " + std::to_string(properties.major) + '.' +
        std::to_string(properties.minor);
    result.memory_bytes = properties.totalGlobalMem;
    result.backend = "cuda";
#if defined(_WIN32)
    result.operating_system = "Windows";
#elif defined(__linux__)
    result.operating_system = have_system_identity
        ? std::string(system.sysname) + ' ' + system.release
        : "Linux";
#else
    result.operating_system = "unknown";
#endif
    result.driver = "CUDA driver " + std::to_string(driver);
    result.power_mode = "driver-default";
    return result;
}

} // namespace parallel_mater::gallery
