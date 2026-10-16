// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "normal";
    PmInstanceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    for (const std::uint32_t version : {1U, PM_API_VERSION_0_2 + 1U, 0xffffffffU}) {
        info.apiVersion = version;
        CHECK(pmCreateInstance(&info, nullptr, &instance) ==
              PM_ERROR_INCOMPATIBLE_VERSION);
        CHECK(instance == nullptr);
    }
    info.apiVersion = PM_API_VERSION_0_2;
    if (std::strcmp(mode, "reject-driver") == 0) {
        CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_ERROR_DRIVER_NOT_FOUND);
        CHECK(instance == nullptr);
        return 0;
    }
    const char *layer = "PM_LAYER_PARALLEL_MATER_validation";
    if (std::strcmp(mode, "normal") != 0) {
        info.enabledLayerCount = 1;
        info.ppEnabledLayerNames = &layer;
    }
    if (std::strcmp(mode, "reject-layer") == 0) {
        CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_ERROR_LAYER_NOT_PRESENT);
        CHECK(instance == nullptr);
        return 0;
    }
    CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t count{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, nullptr) == PM_SUCCESS);
    CHECK(count >= 1U);
    std::vector<PmPhysicalDevice> physicals(count);
    CHECK(pmEnumeratePhysicalDevices(instance, &count, physicals.data()) ==
          PM_SUCCESS);
    bool found_mock = false;
    for (PmPhysicalDevice physical : physicals) {
        PmPhysicalDeviceProperties properties{};
        pmGetPhysicalDeviceProperties(physical, &properties);
        found_mock |= properties.vendorId == 0x504dU;
    }
    CHECK(found_mock);
    pmDestroyInstance(instance);
    // Zero remains the documented default; the legacy layer spelling is accepted
    // but enumeration must advertise the project-owned name.
    info.apiVersion = 0;
    layer = "PM_LAYER_KHRONOS_validation";
    CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_SUCCESS);
    pmDestroyInstance(instance);
    PmLayerProperties advertised{};
    count = 1;
    CHECK(pmEnumerateInstanceLayerProperties(&count, &advertised) == PM_SUCCESS);
    CHECK(std::strcmp(advertised.layerName, "PM_LAYER_PARALLEL_MATER_validation") == 0);
    std::puts("PASS: loader aggregates an independently installed vendor driver");
    return 0;
}
