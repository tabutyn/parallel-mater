// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <utility>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

namespace {

PmResult query(PmPhysicalDevice physical, PmInteractionFamily first,
               PmInteractionFamily second, PmInteractionScope scope,
               PmInteractionMode mode, PmInteractionDirection direction,
               PmInteractionMaterialProfile profile,
               PmInteractionCapability &capability) {
    const PmInteractionCapabilityKey key{
        first, second, scope, mode, direction, profile};
    return pmGetPhysicalDeviceInteractionCapability(
        physical, &key, &capability);
}

} // namespace

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);

    std::uint32_t physical_count = 1U;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, &physical) ==
          PM_SUCCESS);
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);
    CHECK(features.rigidBodies && features.particleLiquids && features.cloth &&
          features.rope && features.softBodies && features.smoke &&
          features.domainCouplings);

    constexpr std::array profiles{
        PM_INTERACTION_MATERIAL_SMOKE_PARTICLE,
        PM_INTERACTION_MATERIAL_SMOKE_GRID};
    PmInteractionCapability capability{};
    for (const auto profile : profiles) {
        for (const auto mode : {PM_INTERACTION_MODE_PRESSURE,
                                PM_INTERACTION_MODE_VISCOSITY}) {
            CHECK(query(physical, PM_INTERACTION_FAMILY_SMOKE,
                        PM_INTERACTION_FAMILY_SMOKE,
                        PM_INTERACTION_SCOPE_SELF, mode,
                        PM_INTERACTION_DIRECTION_BIDIRECTIONAL, profile,
                        capability) == PM_SUCCESS);
            CHECK(capability.maxGraphEdgesPerSimulation ==
                  PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE);
        }
        for (const auto mode : {PM_INTERACTION_MODE_CONTACT,
                                PM_INTERACTION_MODE_DRAG}) {
            CHECK(query(physical, PM_INTERACTION_FAMILY_SMOKE,
                        PM_INTERACTION_FAMILY_RIGID_BODY,
                        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, mode,
                        PM_INTERACTION_DIRECTION_BIDIRECTIONAL, profile,
                        capability) == PM_SUCCESS);
            CHECK(capability.maxGraphEdgesPerSimulation ==
                  PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY);
        }
        const std::array fluid_modes{
            std::pair{PM_INTERACTION_MODE_DRAG,
                      PM_INTERACTION_DIRECTION_SECOND_TO_FIRST},
            std::pair{PM_INTERACTION_MODE_THERMAL,
                      PM_INTERACTION_DIRECTION_EXTERNAL_TO_FIRST},
            std::pair{PM_INTERACTION_MODE_PHASE,
                      PM_INTERACTION_DIRECTION_FIRST_TO_SECOND_TRANSFER}};
        for (const auto [mode, direction] : fluid_modes) {
            CHECK(query(physical, PM_INTERACTION_FAMILY_FLUID,
                        PM_INTERACTION_FAMILY_SMOKE,
                        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, mode, direction,
                        profile, capability) == PM_SUCCESS);
            CHECK(capability.maxGraphEdgesPerSimulation ==
                  PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY);
        }
        for (const auto deformable : {PM_INTERACTION_FAMILY_CLOTH,
                                      PM_INTERACTION_FAMILY_SOFT_BODY,
                                      PM_INTERACTION_FAMILY_ROPE}) {
            CHECK(query(physical, deformable, PM_INTERACTION_FAMILY_SMOKE,
                        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
                        PM_INTERACTION_MODE_CONTACT,
                        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND, profile,
                        capability) == PM_SUCCESS);
            CHECK(capability.maxGraphEdgesPerSimulation ==
                  PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY);
            CHECK(query(physical, deformable, PM_INTERACTION_FAMILY_SMOKE,
                        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
                        PM_INTERACTION_MODE_DRAG,
                        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST, profile,
                        capability) == PM_SUCCESS);
            CHECK(capability.maxGraphEdgesPerSimulation ==
                  PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY);
        }
    }

    pmDestroyInstance(instance);
    std::puts("PASS: Vulkan exposes qualified smoke interactions");
    return 0;
}
