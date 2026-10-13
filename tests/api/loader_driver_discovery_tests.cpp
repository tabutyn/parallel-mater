// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

int main() {
    PmInstanceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
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
    std::puts("PASS: loader aggregates an independently installed vendor driver");
    return 0;
}
