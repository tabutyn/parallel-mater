// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <cmath>

namespace parallel_mater::api {
namespace {

struct RequiredInteraction {
    PmInteractionMode mode;
    PmInteractionDirection direction;
};

struct CouplingContract {
    PmInteractionFamily first;
    PmInteractionFamily second;
    PmInteractionModeFlags mode_flags;
    bool smoke_profile;
    RequiredInteraction interactions[3];
    std::uint32_t interaction_count;
};

constexpr CouplingContract coupling_contracts[] = {
    {PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_RIGID_BODY,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_CONTAINMENT_BIT,
     false, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL},
             {PM_INTERACTION_MODE_CONTAINMENT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 2U},
    {PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_PRESSURE_BIT |
         PM_INTERACTION_MODE_VISCOSITY_BIT,
     false, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL},
             {PM_INTERACTION_MODE_PRESSURE, PM_INTERACTION_DIRECTION_BIDIRECTIONAL},
             {PM_INTERACTION_MODE_VISCOSITY, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 3U},
    {PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_CLOTH,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_CONTAINMENT_BIT,
     false, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL},
             {PM_INTERACTION_MODE_CONTAINMENT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 2U},
    {PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_ROPE,
     PM_INTERACTION_MODE_CONTACT_BIT, false,
     {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 1U},
    {PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SOFT_BODY,
     PM_INTERACTION_MODE_CONTACT_BIT, false,
     {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 1U},
    {PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SMOKE,
     PM_INTERACTION_MODE_DRAG_BIT | PM_INTERACTION_MODE_THERMAL_BIT |
         PM_INTERACTION_MODE_PHASE_BIT,
     true, {{PM_INTERACTION_MODE_DRAG, PM_INTERACTION_DIRECTION_SECOND_TO_FIRST},
            {PM_INTERACTION_MODE_THERMAL, PM_INTERACTION_DIRECTION_EXTERNAL_TO_FIRST},
            {PM_INTERACTION_MODE_PHASE, PM_INTERACTION_DIRECTION_FIRST_TO_SECOND_TRANSFER}}, 3U},
    {PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_SOFT_BODY,
     PM_INTERACTION_MODE_CONTACT_BIT, false,
     {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 1U},
    {PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_ROPE,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_ATTACHMENT_BIT,
     false, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL},
             {PM_INTERACTION_MODE_ATTACHMENT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 2U},
    {PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_ROPE,
     PM_INTERACTION_MODE_ATTACHMENT_BIT, false,
     {{PM_INTERACTION_MODE_ATTACHMENT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 1U},
    {PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_RIGID_BODY,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_DRAG_BIT,
     true, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_BIDIRECTIONAL},
            {PM_INTERACTION_MODE_DRAG, PM_INTERACTION_DIRECTION_BIDIRECTIONAL}}, 2U},
    {PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_SMOKE,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_DRAG_BIT,
     true, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_FIRST_TO_SECOND},
            {PM_INTERACTION_MODE_DRAG, PM_INTERACTION_DIRECTION_SECOND_TO_FIRST}}, 2U},
    {PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_SMOKE,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_DRAG_BIT,
     true, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_FIRST_TO_SECOND},
            {PM_INTERACTION_MODE_DRAG, PM_INTERACTION_DIRECTION_SECOND_TO_FIRST}}, 2U},
    {PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_SMOKE,
     PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_DRAG_BIT,
     true, {{PM_INTERACTION_MODE_CONTACT, PM_INTERACTION_DIRECTION_FIRST_TO_SECOND},
            {PM_INTERACTION_MODE_DRAG, PM_INTERACTION_DIRECTION_SECOND_TO_FIRST}}, 2U},
};

constexpr PmInteractionModeFlags all_mode_flags =
    PM_INTERACTION_MODE_CONTACT_BIT | PM_INTERACTION_MODE_ATTACHMENT_BIT |
    PM_INTERACTION_MODE_DRAG_BIT | PM_INTERACTION_MODE_CONTAINMENT_BIT |
    PM_INTERACTION_MODE_PRESSURE_BIT | PM_INTERACTION_MODE_VISCOSITY_BIT |
    PM_INTERACTION_MODE_THERMAL_BIT | PM_INTERACTION_MODE_PHASE_BIT;

bool zero_equal_material_parameters(
    const PmCouplingParameters &parameters) noexcept {
    const auto &mechanical = parameters.mechanical;
    const auto &smoke = parameters.smoke;
    const auto &attachment = parameters.attachment;
    const auto &thermal = parameters.thermal;
    return mechanical.contactDistance == 0.0F &&
        mechanical.stiffness == 0.0F && mechanical.damping == 0.0F &&
        mechanical.friction == 0.0F && mechanical.tangentialDrag == 0.0F &&
        mechanical.maximumForce == 0.0F &&
        mechanical.maximumAcceleration == 0.0F &&
        mechanical.twoWay == PM_TRUE && smoke.windDrag == 0.0F &&
        smoke.maximumWindAcceleration == 0.0F && smoke.airDensity == 0.0F &&
        smoke.dragCoefficient == 0.0F && smoke.contactDistance == 0.0F &&
        smoke.tracerContact == PM_FALSE && attachment.firstVertex == 0U &&
        attachment.lastVertex == 0U && attachment.effectiveMass == 0.0F &&
        attachment.supportRadiusScale == 0.0F &&
        attachment.contactSupportRadiusScale == 0.0F &&
        attachment.maximumAcceleration == 0.0F &&
        attachment.attachFirst == PM_FALSE &&
        attachment.attachLast == PM_FALSE && thermal.heaterCenter[0] == 0.0F &&
        thermal.heaterCenter[1] == 0.0F && thermal.heaterCenter[2] == 0.0F &&
        thermal.heaterOrientation[0] == 0.0F &&
        thermal.heaterOrientation[1] == 0.0F &&
        thermal.heaterOrientation[2] == 0.0F &&
        thermal.heaterOrientation[3] == 0.0F &&
        thermal.heaterHalfExtents[0] == 0.0F &&
        thermal.heaterHalfExtents[1] == 0.0F &&
        thermal.heaterTemperature == 0.0F &&
        thermal.boilingTemperature == 0.0F &&
        thermal.heatTransferRate == 0.0F && thermal.windDrag == 0.0F &&
        thermal.steamRiseSpeed == 0.0F;
}

static_assert(sizeof(coupling_contracts) / sizeof(coupling_contracts[0]) ==
              PM_COUPLING_SMOKE_SOFT_BODY + 1U);

} // namespace

PmResult validate_coupling(PmDevice device, const PmCouplingState &coupling) noexcept {
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (coupling.type < PM_COUPLING_LIQUID_RIGID || coupling.type > PM_COUPLING_SMOKE_SOFT_BODY)
        return PM_ERROR_INVALID_ARGUMENT;
    if (coupling.modeFlags == 0U || (coupling.modeFlags & ~all_mode_flags) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    if (coupling.materialProfile < PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE ||
        coupling.materialProfile > PM_INTERACTION_MATERIAL_THERMAL_PHASE)
        return PM_ERROR_INVALID_ARGUMENT;

    const auto &contract = coupling_contracts[static_cast<std::size_t>(coupling.type)];
    if (coupling.modeFlags != contract.mode_flags)
        return PM_ERROR_FEATURE_NOT_PRESENT;
    const bool valid_profile = coupling.type == PM_COUPLING_LIQUID_LIQUID
        ? coupling.materialProfile == PM_INTERACTION_MATERIAL_EQUAL
        : contract.smoke_profile
        ? coupling.materialProfile == PM_INTERACTION_MATERIAL_SMOKE_PARTICLE ||
              coupling.materialProfile == PM_INTERACTION_MATERIAL_SMOKE_GRID
        : coupling.materialProfile == PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    if (!valid_profile) return PM_ERROR_FEATURE_NOT_PRESENT;

    for (std::uint32_t index = 0U; index < contract.interaction_count; ++index) {
        const PmInteractionCapabilityKey key{
            contract.first, contract.second,
            coupling.type == PM_COUPLING_LIQUID_LIQUID
                ? PM_INTERACTION_SCOPE_SEPARATE_SYSTEM
                : PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
            contract.interactions[index].mode,
            contract.interactions[index].direction,
            coupling.materialProfile};
        PmInteractionCapability capability{};
        const PmResult result = find_interaction_capability(
            device->enabled, key, capability);
        if (result != PM_SUCCESS) return result;
        if (capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
            return PM_ERROR_FEATURE_NOT_PRESENT;
    }
    switch (coupling.type) {
    case PM_COUPLING_LIQUID_LIQUID:
        if (!zero_equal_material_parameters(coupling.parameters))
            return PM_ERROR_FEATURE_NOT_PRESENT;
        break;
    case PM_COUPLING_LIQUID_RIGID:
    case PM_COUPLING_LIQUID_CLOTH:
    case PM_COUPLING_LIQUID_ROPE:
    case PM_COUPLING_LIQUID_SOFT_BODY:
    case PM_COUPLING_SOFT_BODY_CLOTH:
        if (coupling.parameters.mechanical.twoWay > PM_TRUE) return PM_ERROR_INVALID_ARGUMENT;
        if (!coupling.parameters.mechanical.twoWay) return PM_ERROR_FEATURE_NOT_PRESENT;
        break;
    default: break;
    }
    // Cloth support radius is derived from the liquid material. Acceleration
    // is a different unit and this pair does not implement that cap.
    if (coupling.type == PM_COUPLING_LIQUID_CLOTH && coupling.parameters.mechanical.maximumAcceleration != 0)
        return PM_ERROR_FEATURE_NOT_PRESENT;
    return PM_SUCCESS;
}
} // namespace parallel_mater::api

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreatePhysicsPipeline(
    PmDevice device, const PmPhysicsPipelineCreateInfo *info,
    PmPhysicsPipeline *pipeline) {
    if (pipeline == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *pipeline = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO,
            info->pNext) ||
        (info->fluidMaterialCount != 0U && info->pFluidMaterials == nullptr) ||
        (info->couplingCount != 0U && info->pCouplings == nullptr) ||
        (info->clothMaterialCount != 0U && info->pClothMaterials == nullptr) ||
        (info->ropeMaterialCount != 0U && info->pRopeMaterials == nullptr) ||
        (info->softBodyMaterialCount != 0U &&
         info->pSoftBodyMaterials == nullptr) ||
        (info->smokeMaterialCount != 0U && info->pSmokeMaterials == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmPhysicsPipeline_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->serial = device->next_pipeline_serial.fetch_add(1, std::memory_order_relaxed);
    created->rigid = info->pRigidState != nullptr
        ? *info->pRigidState : PmRigidSolverState{10U, PM_FALSE, 0.02F, 0.02F, 0.5F};
    if (created->rigid.rigidIterations == 0U ||
        created->rigid.rigidIterations > 64U) {
        parallel_mater::api::destroy_object(device, created);
        return PM_ERROR_INVALID_ARGUMENT;
    }
    if (created->rigid.enableSleeping != PM_FALSE) {
        const auto result = created->rigid.enableSleeping == PM_TRUE
            ? PM_ERROR_FEATURE_NOT_PRESENT : PM_ERROR_INVALID_ARGUMENT;
        parallel_mater::api::destroy_object(device, created);
        return result;
    }
    try {
        if (info->fluidMaterialCount != 0U)
            created->fluids.assign(info->pFluidMaterials,
                                   info->pFluidMaterials + info->fluidMaterialCount);
        if (info->couplingCount != 0U)
            created->couplings.assign(info->pCouplings,
                                      info->pCouplings + info->couplingCount);
        if (info->clothMaterialCount != 0U)
            created->cloths.assign(info->pClothMaterials,
                                   info->pClothMaterials + info->clothMaterialCount);
        if (info->ropeMaterialCount != 0U)
            created->ropes.assign(info->pRopeMaterials,
                                  info->pRopeMaterials + info->ropeMaterialCount);
        if (info->softBodyMaterialCount != 0U)
            created->soft_bodies.assign(
                info->pSoftBodyMaterials,
                info->pSoftBodyMaterials + info->softBodyMaterialCount);
        if (info->smokeMaterialCount != 0U)
            created->smokes.assign(info->pSmokeMaterials,
                                   info->pSmokeMaterials + info->smokeMaterialCount);
    } catch (const std::bad_alloc &) {
        parallel_mater::api::destroy_object(device, created);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (const auto &fluid : created->fluids) {
        if (!(fluid.restDensity > 0.0F && fluid.particleRadius > 0.0F &&
              fluid.supportRadius >= fluid.particleRadius &&
              fluid.maximumSpeed > 0.0F && fluid.densityIterations != 0U)) {
            parallel_mater::api::destroy_object(device, created);
            return PM_ERROR_INVALID_ARGUMENT;
        }
    }
    for (const auto &coupling : created->couplings) {
        const auto result = parallel_mater::api::validate_coupling(device, coupling);
        if (result != PM_SUCCESS) {
            parallel_mater::api::destroy_object(device, created);
            return result;
        }
    }
    *pipeline = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyPhysicsPipeline(
    PmDevice device, PmPhysicsPipeline pipeline) {
    if (device == nullptr || pipeline == nullptr || pipeline->device != device) return;
    (void)pmDeviceWaitIdle(device);
    parallel_mater::api::destroy_object(device, pipeline);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdBindPhysicsPipeline(
    PmCommandBuffer command_buffer, PmPhysicsPipeline pipeline) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (pipeline == nullptr || pipeline->device != command_buffer->device)
        return PM_ERROR_INVALID_ARGUMENT;
    command_buffer->bound_pipeline = pipeline;
    return PM_SUCCESS;
}
