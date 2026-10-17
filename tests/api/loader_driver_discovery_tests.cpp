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
    PmPhysicalDevice mock_physical{};
    for (PmPhysicalDevice physical : physicals) {
        PmPhysicalDeviceProperties properties{};
        pmGetPhysicalDeviceProperties(physical, &properties);
        if (properties.vendorId == 0x504dU) {
            found_mock = true;
            mock_physical = physical;
        }
    }
    CHECK(found_mock);
    std::uint32_t interaction_count{};
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              mock_physical, &interaction_count, nullptr) == PM_SUCCESS);
    CHECK(interaction_count == 1U);
    PmInteractionCapability interaction{};
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              mock_physical, &interaction_count, &interaction) == PM_SUCCESS);
    CHECK(interaction.key.firstFamily == PM_INTERACTION_FAMILY_FLUID);
    CHECK(interaction.key.secondFamily == PM_INTERACTION_FAMILY_RIGID_BODY);
    CHECK(interaction.key.mode == PM_INTERACTION_MODE_CONTACT);
    CHECK(interaction.maxGraphEdgesPerSimulation == 64U);
    PmInteractionCapability queried{};
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              mock_physical, &interaction.key, &queried) == PM_SUCCESS);
    CHECK(queried.maxGraphEdgesPerSimulation == 64U);
    auto unsupported = interaction.key;
    unsupported.mode = PM_INTERACTION_MODE_PHASE;
    queried.maxGraphEdgesPerSimulation = 99U;
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              mock_physical, &unsupported, &queried) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(queried.maxGraphEdgesPerSimulation == 0U);
    auto unsupported_material = interaction.key;
    unsupported_material.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_GRID;
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              mock_physical, &unsupported_material, &queried) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    auto malformed_pair = interaction.key;
    malformed_pair.firstFamily = PM_INTERACTION_FAMILY_RIGID_BODY;
    malformed_pair.secondFamily = PM_INTERACTION_FAMILY_FLUID;
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              mock_physical, &malformed_pair, &queried) ==
          PM_ERROR_INVALID_ARGUMENT);
    auto malformed = interaction.key;
    malformed.mode = static_cast<PmInteractionMode>(99);
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              mock_physical, &malformed, &queried) ==
          PM_ERROR_INVALID_ARGUMENT);
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
