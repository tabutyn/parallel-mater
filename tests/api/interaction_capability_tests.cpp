// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

namespace {

bool same_key(const PmInteractionCapabilityKey &left,
              const PmInteractionCapabilityKey &right) {
    return left.firstFamily == right.firstFamily &&
           left.secondFamily == right.secondFamily &&
           left.scope == right.scope && left.mode == right.mode &&
           left.direction == right.direction &&
           left.materialProfile == right.materialProfile;
}

} // namespace

int main() {
    PmInstanceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t physical_count = 1U;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, &physical) ==
          PM_SUCCESS);

    std::uint32_t count{};
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              physical, &count, nullptr) == PM_SUCCESS);
    CHECK(count >= 2U);
    PmInteractionCapability short_record{};
    std::uint32_t short_count = 1U;
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              physical, &short_count, &short_record) == PM_INCOMPLETE);
    CHECK(short_count == 1U);

    std::vector<PmInteractionCapability> records(count);
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              physical, &count, records.data()) == PM_SUCCESS);
    CHECK(count == records.size());
    bool rigid_contact = false;
    bool rigid_attachment = false;
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto &record = records[index];
        for (std::size_t prior = 0; prior < index; ++prior)
            CHECK(!same_key(record.key, records[prior].key));
        PmInteractionCapability queried{};
        CHECK(pmGetPhysicalDeviceInteractionCapability(
                  physical, &record.key, &queried) == PM_SUCCESS);
        CHECK(same_key(record.key, queried.key));
        CHECK(record.maxGraphEdgesPerSimulation ==
              queried.maxGraphEdgesPerSimulation);
        CHECK(!(record.key.firstFamily == PM_INTERACTION_FAMILY_SMOKE &&
                record.key.secondFamily == PM_INTERACTION_FAMILY_SMOKE &&
                record.key.scope == PM_INTERACTION_SCOPE_SEPARATE_SYSTEM));
        rigid_contact |=
            record.key.firstFamily == PM_INTERACTION_FAMILY_RIGID_BODY &&
            record.key.secondFamily == PM_INTERACTION_FAMILY_RIGID_BODY &&
            record.key.scope == PM_INTERACTION_SCOPE_SAME_DOMAIN &&
            record.key.mode == PM_INTERACTION_MODE_CONTACT;
        rigid_attachment |=
            record.key.firstFamily == PM_INTERACTION_FAMILY_RIGID_BODY &&
            record.key.secondFamily == PM_INTERACTION_FAMILY_RIGID_BODY &&
            record.key.scope == PM_INTERACTION_SCOPE_SAME_DOMAIN &&
            record.key.mode == PM_INTERACTION_MODE_ATTACHMENT;
    }
    CHECK(rigid_contact && rigid_attachment);

    PmInteractionCapabilityKey unsupported{
        PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SEPARATE_SYSTEM,
        PM_INTERACTION_MODE_PRESSURE,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_EQUAL};
    PmInteractionCapability queried{};
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &unsupported, &queried) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    for (const auto profile : {PM_INTERACTION_MATERIAL_MISCIBLE,
                               PM_INTERACTION_MATERIAL_IMMISCIBLE_INTERFACE}) {
        unsupported.materialProfile = profile;
        for (const auto mode : {PM_INTERACTION_MODE_CONTACT,
                                PM_INTERACTION_MODE_PRESSURE,
                                PM_INTERACTION_MODE_VISCOSITY,
                                PM_INTERACTION_MODE_INTERFACE}) {
            unsupported.mode = mode;
            CHECK(pmGetPhysicalDeviceInteractionCapability(
                      physical, &unsupported, &queried) ==
                  PM_ERROR_FEATURE_NOT_PRESENT);
        }
    }
    unsupported.mode = static_cast<PmInteractionMode>(999);
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &unsupported, &queried) == PM_ERROR_INVALID_ARGUMENT);
    unsupported.mode = PM_INTERACTION_MODE_CONTACT;
    unsupported.firstFamily = PM_INTERACTION_FAMILY_RIGID_BODY;
    unsupported.secondFamily = PM_INTERACTION_FAMILY_FLUID;
    unsupported.scope = PM_INTERACTION_SCOPE_DISTINCT_FAMILY;
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &unsupported, &queried) == PM_ERROR_INVALID_ARGUMENT);

    // The public identities exist before any driver may advertise a
    // separate-system smoke operator.
    unsupported.firstFamily = PM_INTERACTION_FAMILY_SMOKE;
    unsupported.secondFamily = PM_INTERACTION_FAMILY_SMOKE;
    unsupported.scope = PM_INTERACTION_SCOPE_SEPARATE_SYSTEM;
    unsupported.direction = PM_INTERACTION_DIRECTION_BIDIRECTIONAL;
    for (const auto profile : {PM_INTERACTION_MATERIAL_SMOKE_PARTICLE,
                               PM_INTERACTION_MATERIAL_SMOKE_GRID}) {
        unsupported.materialProfile = profile;
        for (const auto mode : {PM_INTERACTION_MODE_PRESSURE,
                                PM_INTERACTION_MODE_VISCOSITY,
                                PM_INTERACTION_MODE_THERMAL}) {
            unsupported.mode = mode;
            CHECK(pmGetPhysicalDeviceInteractionCapability(
                      physical, &unsupported, &queried) ==
                  PM_ERROR_FEATURE_NOT_PRESENT);
        }
    }
    unsupported.materialProfile =
        PM_INTERACTION_MATERIAL_SMOKE_PARTICLE_GRID;
    for (const auto mode : {PM_INTERACTION_MODE_DRAG,
                            PM_INTERACTION_MODE_THERMAL}) {
        unsupported.mode = mode;
        CHECK(pmGetPhysicalDeviceInteractionCapability(
                  physical, &unsupported, &queried) ==
              PM_ERROR_FEATURE_NOT_PRESENT);
    }

    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);
    if (features.rigidBodies && features.particleLiquids &&
        features.liquidRigidCoupling) {
        const float priority = 1.0F;
        const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
        PmDeviceCreateInfo device_info{};
        device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1U;
        device_info.pQueueInfos = &queue_info;
        device_info.enabledFeatures.rigidBodies = PM_TRUE;
        device_info.enabledFeatures.particleLiquids = PM_TRUE;
        device_info.enabledFeatures.liquidRigidCoupling = PM_TRUE;
        PmDevice device{};
        CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);

        PmCouplingState edge{};
        edge.type = PM_COUPLING_LIQUID_RIGID;
        edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_CONTAINMENT_BIT;
        edge.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
        edge.parameters.mechanical.twoWay = PM_TRUE;
        PmPhysicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
        pipeline_info.couplingCount = 1U;
        pipeline_info.pCouplings = &edge;
        PmPhysicsPipeline pipeline{};
        CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
              PM_SUCCESS);
        pmDestroyPhysicsPipeline(device, pipeline);

        edge.modeFlags = 0U;
        CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
              PM_ERROR_INVALID_ARGUMENT);
        edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
        CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
              PM_ERROR_FEATURE_NOT_PRESENT);
        edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_CONTAINMENT_BIT;
        edge.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_GRID;
        CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
              PM_ERROR_FEATURE_NOT_PRESENT);
        pmDestroyDevice(device);
    }

    pmDestroyInstance(instance);
    std::puts("PASS: discovery and graph descriptors share exact capabilities");
    return 0;
}
