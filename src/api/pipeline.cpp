// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <cmath>

namespace parallel_mater::api {
PmResult validate_coupling(PmDevice device, const PmCouplingState &coupling) noexcept {
    if (coupling.type < PM_COUPLING_LIQUID_RIGID || coupling.type > PM_COUPLING_SMOKE_SOFT_BODY)
        return PM_ERROR_INVALID_ARGUMENT;
    if (coupling.type == PM_COUPLING_LIQUID_LIQUID) return PM_ERROR_FEATURE_NOT_PRESENT;
    if (coupling.type != PM_COUPLING_LIQUID_RIGID && !device->enabled.domainCouplings)
        return PM_ERROR_FEATURE_NOT_PRESENT;
    switch (coupling.type) {
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
