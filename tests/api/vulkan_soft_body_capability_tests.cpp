// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdint>
#include <cstdio>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

namespace {

PmResult query(PmPhysicalDevice physical, PmInteractionFamily first,
               PmInteractionFamily second,
               PmInteractionCapability &capability) {
    const PmInteractionCapabilityKey key{
        first, second, PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    return pmGetPhysicalDeviceInteractionCapability(
        physical, &key, &capability);
}

} // namespace

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);

    std::uint32_t physical_count = 1U;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, &physical) ==
          PM_SUCCESS);
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);
    CHECK(features.rigidBodies && features.particleLiquids && features.cloth &&
          features.softBodies && features.domainCouplings);

    PmInteractionCapability capability{};
    CHECK(query(physical, PM_INTERACTION_FAMILY_FLUID,
                PM_INTERACTION_FAMILY_SOFT_BODY, capability) == PM_SUCCESS);
    CHECK(capability.maxGraphEdgesPerSimulation ==
          PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY);
    CHECK(query(physical, PM_INTERACTION_FAMILY_SOFT_BODY,
                PM_INTERACTION_FAMILY_RIGID_BODY, capability) == PM_SUCCESS);
    CHECK(capability.maxGraphEdgesPerSimulation ==
          PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE);
    CHECK(query(physical, PM_INTERACTION_FAMILY_CLOTH,
                PM_INTERACTION_FAMILY_SOFT_BODY, capability) ==
          PM_ERROR_FEATURE_NOT_PRESENT);

    pmDestroyInstance(instance);
    std::puts("PASS: Vulkan exposes only qualified soft-body interactions");
    return 0;
}
