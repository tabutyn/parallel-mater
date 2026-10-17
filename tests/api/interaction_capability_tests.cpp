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
    unsupported.materialProfile = PM_INTERACTION_MATERIAL_MISCIBLE;
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &unsupported, &queried) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    unsupported.mode = static_cast<PmInteractionMode>(999);
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &unsupported, &queried) == PM_ERROR_INVALID_ARGUMENT);
    unsupported.mode = PM_INTERACTION_MODE_CONTACT;
    unsupported.firstFamily = PM_INTERACTION_FAMILY_RIGID_BODY;
    unsupported.secondFamily = PM_INTERACTION_FAMILY_FLUID;
    unsupported.scope = PM_INTERACTION_SCOPE_DISTINCT_FAMILY;
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &unsupported, &queried) == PM_ERROR_INVALID_ARGUMENT);

    pmDestroyInstance(instance);
    std::puts("PASS: reference driver exposes exact interaction capabilities");
    return 0;
}
