// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/json.hpp>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using parallel_mater::conformance::Json;

struct Arguments {
    std::string contract;
    std::filesystem::path root;
    std::filesystem::path output;
};

Arguments parse(int argc, char **argv) {
    Arguments result{};
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--contract" && index + 1 < argc)
            result.contract = argv[++index];
        else if (argument == "--root" && index + 1 < argc)
            result.root = argv[++index];
        else if (argument == "--output" && index + 1 < argc)
            result.output = argv[++index];
        else
            throw std::runtime_error("unknown or incomplete argument: " + argument);
    }
    if (result.contract.empty() || result.root.empty() || result.output.empty())
        throw std::runtime_error(
            "usage: pm-android-device-contracts --contract "
            "rigid-quality|fluid-quality|adverse-lifecycle|vulkan-info "
            "--root DIR --output FILE");
    return result;
}

void write(const std::filesystem::path &path, const std::string &contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream) throw std::runtime_error("cannot write " + path.string());
}

std::string vulkan_version(std::uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
        std::to_string(VK_API_VERSION_MINOR(version)) + "." +
        std::to_string(VK_API_VERSION_PATCH(version));
}

std::string vulkan_info() {
    std::uint32_t instance_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&instance_version) != VK_SUCCESS)
        throw std::runtime_error("cannot query Vulkan instance version");
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "Parallel Mater qualification";
    application.apiVersion = std::min(instance_version, VK_API_VERSION_1_1);
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &application;
    VkInstance instance{};
    if (vkCreateInstance(&create, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("cannot create Vulkan qualification instance");
    std::uint32_t count{};
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS ||
        count == 0U) {
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("no Vulkan physical device");
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("cannot enumerate Vulkan physical devices");
    }
    VkPhysicalDeviceDriverProperties driver{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    vkGetPhysicalDeviceProperties2(devices.front(), &properties);
    vkDestroyInstance(instance, nullptr);

    Json result = Json::object();
    result["schema"] = "parallel-mater-vulkan-device-info/v1";
    result["instance_api_version"] = vulkan_version(instance_version);
    result["api_version"] = vulkan_version(properties.properties.apiVersion);
    result["api_version_raw"] =
        static_cast<std::uint64_t>(properties.properties.apiVersion);
    result["driver_version_raw"] =
        static_cast<std::uint64_t>(properties.properties.driverVersion);
    result["vendor_id"] =
        static_cast<std::uint64_t>(properties.properties.vendorID);
    result["device_id"] =
        static_cast<std::uint64_t>(properties.properties.deviceID);
    result["device_name"] = properties.properties.deviceName;
    result["driver_id"] = static_cast<std::uint64_t>(driver.driverID);
    result["driver_name"] = driver.driverName;
    result["driver_info"] = driver.driverInfo;
    return result.serialize();
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto arguments = parse(argc, argv);
        if (arguments.contract == "rigid-quality") {
            write(arguments.output,
                  parallel_mater::conformance::conformance_rigid_brick_quality(
                      arguments.root));
        } else if (arguments.contract == "fluid-quality") {
            const auto temporary = arguments.output.string() + ".temporary";
            write(arguments.output,
                  parallel_mater::conformance::conformance_fluid_pair_quality(
                      temporary));
            std::filesystem::remove(temporary);
        } else if (arguments.contract == "adverse-lifecycle") {
            write(arguments.output,
                  parallel_mater::conformance::conformance_adverse_lifecycle(
                      arguments.root));
        } else if (arguments.contract == "vulkan-info") {
            write(arguments.output, vulkan_info());
        } else {
            throw std::runtime_error("unknown contract: " + arguments.contract);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "android qualification: " << error.what() << '\n';
        return 2;
    }
}
