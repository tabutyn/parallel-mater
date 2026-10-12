// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater/parallel_mater.hpp>

#include "../device/device.hpp"
#include "../simulation/memory_arena.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct PmInstance_T;
struct PmDevice_T;

struct PmPhysicalDevice_T {
    PmInstance_T *instance{};
    parallel_mater::device::Description description;
};

struct PmQueue_T {
    PmDevice_T *device{};
    std::uint32_t family{};
    std::uint32_t index{};
    std::mutex mutex;
};

struct PmInstance_T {
    PmAllocationCallbacks allocator{};
    bool has_allocator{};
    bool validation{};
    std::vector<std::unique_ptr<PmPhysicalDevice_T>> physical_devices;
};

struct PmDevice_T {
    PmPhysicalDevice_T *physical{};
    PmPhysicalDeviceFeatures enabled{};
    std::unique_ptr<parallel_mater::device::Services> services;
    PmQueue_T queue{};
    std::atomic<PmDeviceSize> application_bytes{};
    std::atomic<PmDeviceSize> library_host_bytes{};
};

struct PmDeviceMemory_T {
    PmDevice_T *device{};
    void *allocation{};
    PmDeviceSize size{};
    std::uint32_t memory_type{};
    bool mapped{};
};

struct PmBuffer_T {
    PmDevice_T *device{};
    PmDeviceSize size{};
    PmFlags usage{};
    std::uint32_t queue_family{};
    PmDeviceMemory memory{};
    PmDeviceSize memory_offset{};
    void *native_pointer{};
    bool imported{};
};

struct PmGeometry_T {
    PmDevice_T *device{};
    PmGeometryCreateInfo info{};
    PmDeviceMemory memory{};
    PmDeviceSize memory_offset{};
    std::unique_ptr<parallel_mater::simulation::MemoryArena> arena;
    bool built{};
};

struct PmSimulation_T {
    PmDevice_T *device{};
    PmSimulationCreateInfo info{};
    std::vector<PmFluidCapacity> fluids;
    std::vector<PmClothCapacity> cloth_capacities;
    std::vector<PmRopeCapacity> rope_capacities;
    std::vector<PmSoftBodyCapacity> soft_body_capacities;
    std::vector<PmSmokeCapacity> smoke_capacities;
    std::vector<PmPaintFieldCapacity> paint_field_capacities;
    PmSimulationMemoryRequirements requirements{};
    parallel_mater::WorldOptions world_options{};
    PmSimulationMemoryBinding binding{};
    std::unique_ptr<parallel_mater::simulation::MemoryArena> arena;
    std::unique_ptr<parallel_mater::World> runtime;
    std::unordered_map<PmGeometry, parallel_mater::TriangleMeshId> geometries;
    std::vector<std::optional<parallel_mater::RigidBodyId>> rigid_bodies;
    std::vector<parallel_mater::RigidBodyId> rigid_dense_ids;
    std::vector<std::uint32_t> rigid_dense_indices;
    bool rigid_dense_mapping_dirty{true};
    std::vector<std::optional<parallel_mater::RigidConstraintId>> constraints;
    std::vector<std::optional<parallel_mater::FluidId>> fluid_ids;
    std::vector<std::optional<parallel_mater::ClothId>> cloths;
    std::vector<std::optional<parallel_mater::RopeId>> ropes;
    std::vector<std::optional<parallel_mater::SoftBodyId>> soft_bodies;
    std::vector<std::optional<parallel_mater::SmokeId>> smokes;
    std::vector<std::optional<parallel_mater::ParticleSourceId>> emitters;
    std::vector<std::optional<parallel_mater::ParticleDestroyPlaneId>> destroyers;
    std::vector<std::optional<parallel_mater::PaintFieldId>> paint_fields;
    std::vector<std::optional<parallel_mater::PaintRuleId>> paint_rules;
    bool couplings_initialized{};
    std::uint64_t submitted_steps{};
    std::uint32_t last_substeps{};
    PmResult execution_result{PM_SUCCESS};
};

struct PmPhysicsPipeline_T {
    PmDevice_T *device{};
    PmRigidSolverState rigid{};
    std::vector<PmLiquidMaterialState> fluids;
    std::vector<PmCouplingState> couplings;
    std::vector<PmClothMaterialState> cloths;
    std::vector<PmRopeMaterialState> ropes;
    std::vector<PmSoftBodyMaterialState> soft_bodies;
    std::vector<PmSmokeMaterialState> smokes;
};

enum class PmCommandState : std::uint8_t { initial, recording, executable, pending };

struct PmCommandBuffer_T {
    PmDevice_T *device{};
    struct PmCommandPool_T *pool{};
    PmCommandState state{PmCommandState::initial};
    PmFlags usage{};
    PmResult latched_error{PM_SUCCESS};
    PmPhysicsPipeline bound_pipeline{};
    bool statistics_open{};
    PmQueryPool statistics_pool{};
    std::uint32_t statistics_query{};
    PmSimulation statistics_simulation{};
    cudaEvent_t completion{};
    std::vector<std::function<PmResult(cudaStream_t)>> commands;
};

struct PmCommandPool_T {
    PmDevice_T *device{};
    PmFlags flags{};
    std::uint32_t queue_family{};
    std::vector<PmCommandBuffer> buffers;
};

struct PmSemaphore_T {
    PmDevice_T *device{};
    std::mutex mutex;
    std::condition_variable changed;
    std::uint64_t value{};
    std::map<std::uint64_t, cudaEvent_t> events;
    bool imported{};
    PmNativeHandle native{};
};

struct PmFence_T {
    PmDevice_T *device{};
    cudaEvent_t event{};
    bool submitted{};
    bool initially_signaled{};
};

struct PmQuerySlot {
    cudaEvent_t event{};
    bool available{};
    PmStepStatistics statistics{};
};

struct PmQueryPool_T {
    PmDevice_T *device{};
    PmQueryType type{};
    std::vector<PmQuerySlot> slots;
};

namespace parallel_mater::api {

template <typename T, typename... Args>
[[nodiscard]] T *make_object(PmDevice_T *device, Args &&...args) noexcept {
    try {
        auto *value = new T(std::forward<Args>(args)...);
        if (device != nullptr)
            device->library_host_bytes.fetch_add(sizeof(T), std::memory_order_relaxed);
        return value;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

template <typename T>
void destroy_object(PmDevice_T *device, T *value) noexcept {
    if (value == nullptr) return;
    delete value;
    if (device != nullptr)
        device->library_host_bytes.fetch_sub(sizeof(T), std::memory_order_relaxed);
}

[[nodiscard]] inline bool valid_structure(PmStructureType actual,
                                          PmStructureType expected,
                                          const void *next) noexcept {
    return actual == expected && next == nullptr;
}

[[nodiscard]] inline bool range_fits(PmDeviceSize offset, PmDeviceSize size,
                                     PmDeviceSize capacity) noexcept {
    return offset <= capacity && size <= capacity - offset;
}

[[nodiscard]] inline PmDeviceSize resolved_size(PmDeviceSize offset,
                                                PmDeviceSize size,
                                                PmDeviceSize capacity) noexcept {
    return size == PM_WHOLE_SIZE && offset <= capacity ? capacity - offset : size;
}

[[nodiscard]] inline void *buffer_pointer(PmBuffer buffer,
                                          PmDeviceSize offset = 0U) noexcept {
    if (buffer == nullptr || buffer->native_pointer == nullptr || offset > buffer->size)
        return nullptr;
    return static_cast<std::byte *>(buffer->native_pointer) + offset;
}

[[nodiscard]] inline PmResult require_recording(PmCommandBuffer buffer) noexcept {
    if (buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (buffer->state != PmCommandState::recording) return PM_ERROR_INVALID_STATE;
    return buffer->latched_error;
}

inline void refresh_command_state(PmCommandBuffer buffer) noexcept {
    if (buffer == nullptr || buffer->state != PmCommandState::pending ||
        buffer->completion == nullptr)
        return;
    const cudaError_t status = cudaEventQuery(buffer->completion);
    if (status == cudaSuccess) buffer->state = PmCommandState::executable;
    else if (status != cudaErrorNotReady) (void)cudaGetLastError();
}

[[nodiscard]] inline PmResult from_status(
    const parallel_mater::Status &status) noexcept {
    using parallel_mater::StatusCode;
    switch (status.code) {
    case StatusCode::success: return PM_SUCCESS;
    case StatusCode::invalid_argument:
    case StatusCode::invalid_handle: return PM_ERROR_INVALID_ARGUMENT;
    case StatusCode::not_supported: return PM_ERROR_FEATURE_NOT_PRESENT;
    case StatusCode::capacity_exceeded: return PM_ERROR_CAPACITY_EXCEEDED;
    case StatusCode::busy: return PM_ERROR_INVALID_STATE;
    case StatusCode::out_of_memory: return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    case StatusCode::cuda_failure: return parallel_mater::device::from_cuda(
        status.cuda_error);
    case StatusCode::internal_error: return PM_ERROR_INVALID_STATE;
    }
    return PM_ERROR_INVALID_STATE;
}

inline PmResult record(PmCommandBuffer buffer,
                       std::function<PmResult(cudaStream_t)> command) noexcept {
    const PmResult state = require_recording(buffer);
    if (state != PM_SUCCESS) return state;
    try {
        buffer->commands.push_back(std::move(command));
        return PM_SUCCESS;
    } catch (const std::bad_alloc &) {
        buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
        return buffer->latched_error;
    }
}

[[nodiscard]] PmPhysicalDeviceFeatures supported_features() noexcept;
[[nodiscard]] PFN_pmVoidFunction lookup_proc(const char *name) noexcept;
[[nodiscard]] PmResult initialize_couplings(
    PmSimulation simulation, PmPhysicsPipeline pipeline) noexcept;
[[nodiscard]] PmResult register_geometry(
    PmSimulation simulation, PmGeometry geometry, cudaStream_t stream,
    parallel_mater::TriangleMeshId &output) noexcept;

} // namespace parallel_mater::api
