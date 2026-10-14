// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater/pm_driver.h>
#include <parallel_mater/pm_layer.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace parallel_mater::loader {

struct Driver;
struct Layer;
using LayerChain = std::vector<std::shared_ptr<Layer>>;
struct LoaderDispatch final {};
extern const LoaderDispatch loader_dispatch;

enum class Scope : std::uint8_t { instance, physical, device };

struct Route {
    std::shared_ptr<Driver> driver;
    void *native{};
    void *context{};
    Scope scope{Scope::device};
    std::shared_ptr<LayerChain> layers;

    explicit operator bool() const noexcept {
        return driver != nullptr && native != nullptr;
    }
};

struct Layer {
    ~Layer();
    Layer(const Layer &) = delete;
    Layer &operator=(const Layer &) = delete;

    static std::shared_ptr<Layer> open(const std::filesystem::path &path,
                                       std::string &error);
    PmResult pre(const char *command, const void *handle) const;
    void post(const char *command, const void *handle, PmResult result) const;

    std::filesystem::path path;
    void *library{};
    PFN_pmLayerPreCall pre_call{};
    PFN_pmLayerPostCall post_call{};

  private:
    Layer() = default;
};

struct Driver {
    ~Driver();
    Driver(const Driver &) = delete;
    Driver &operator=(const Driver &) = delete;

    static std::shared_ptr<Driver> open(const std::filesystem::path &path,
                                        std::string &error);
    PFN_pmVoidFunction resolve(const Route &route, const char *name) const;
    PFN_pmVoidFunction instance_proc(PmInstance instance,
                                     const char *name) const;
    PFN_pmVoidFunction device_proc(PmDevice device, const char *name) const;

    std::filesystem::path path;
    std::uint32_t interface_version{};
    void *library{};
    PFN_pmDriverGetInstanceProcAddr get_instance_proc{};
    PFN_pmDriverGetDeviceProcAddr get_device_proc{};
    PFN_pmDriverGetPhysicalDeviceProcAddr get_physical_proc{};

  private:
    Driver() = default;
};

std::vector<std::shared_ptr<Driver>> discover_drivers();
std::shared_ptr<LayerChain> load_validation_layers(bool enabled);
PFN_pmVoidFunction lookup_loader_proc(const char *name) noexcept;
PmResult pre_call(const Route &route, const char *command) noexcept;
void post_call(const Route &route, const char *command, PmResult result) noexcept;

template <typename T>
T failure() noexcept {
    if constexpr (std::is_same_v<T, void>) return;
    else if constexpr (std::is_same_v<T, PmResult>) return PM_ERROR_INVALID_ARGUMENT;
    else return {};
}

template <typename T>
T unsupported() noexcept {
    if constexpr (std::is_same_v<T, void>) return;
    else if constexpr (std::is_same_v<T, PmResult>) return PM_ERROR_FEATURE_NOT_PRESENT;
    else return {};
}

Route route(PmPhysicalDevice value) noexcept;
Route route(PmDevice value) noexcept;
Route route(PmQueue value) noexcept;
Route route(PmCommandBuffer value) noexcept;

} // namespace parallel_mater::loader

struct PmInstance_T {
    const parallel_mater::loader::LoaderDispatch *dispatch{};
    struct Child {
        std::shared_ptr<parallel_mater::loader::Driver> driver;
        PmInstance native{};
    };
    std::vector<Child> children;
    std::vector<std::unique_ptr<PmPhysicalDevice_T>> physical_devices;
    std::shared_ptr<parallel_mater::loader::LayerChain> layers;
};

struct PmPhysicalDevice_T {
    const parallel_mater::loader::LoaderDispatch *dispatch{};
    std::shared_ptr<parallel_mater::loader::Driver> driver;
    PmPhysicalDevice native{};
    PmInstance native_instance{};
    std::shared_ptr<parallel_mater::loader::LayerChain> layers;
};

struct PmDevice_T {
    const parallel_mater::loader::LoaderDispatch *dispatch{};
    std::shared_ptr<parallel_mater::loader::Driver> driver;
    PmDevice native{};
    PmInstance native_instance{};
    std::shared_ptr<parallel_mater::loader::LayerChain> layers;
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::unique_ptr<PmQueue_T>> queues;
};

struct PmQueue_T {
    const parallel_mater::loader::LoaderDispatch *dispatch{};
    std::shared_ptr<parallel_mater::loader::Driver> driver;
    PmQueue native{};
    PmDevice native_device{};
    std::shared_ptr<parallel_mater::loader::LayerChain> layers;
};

struct PmCommandBuffer_T {
    const parallel_mater::loader::LoaderDispatch *dispatch{};
    std::shared_ptr<parallel_mater::loader::Driver> driver;
    PmCommandBuffer native{};
    PmDevice native_device{};
    std::shared_ptr<parallel_mater::loader::LayerChain> layers;
};
