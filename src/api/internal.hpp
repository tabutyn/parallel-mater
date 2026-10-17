// SPDX-License-Identifier: MIT
#pragma once

#if defined(PARALLEL_MATER_DRIVER_IMPLEMENTATION)
#include "driver_remap.h"
#endif

#include <parallel_mater/parallel_mater.h>
#include <simulation/runtime.hpp>

#include "../device/device.hpp"
#include "../device/submission.hpp"
#include "../simulation/memory_arena.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include <variant>
#include <type_traits>

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
    std::unique_ptr<parallel_mater::device::SubmissionQueue> submissions;
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
    std::atomic<std::uint64_t> next_pipeline_serial{1};
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
    PmNativeHandle native_buffer{};
    PmDeviceSize native_offset{};
    std::uint64_t native_usage{};
    void *native_pointer{};
    bool imported{};
};

struct PmGeometry_T {
    PmDevice_T *device{};
    PmGeometryCreateInfo info{};
    PmMemoryRequirements requirements{};
    PmDeviceMemory memory{};
    PmDeviceSize memory_offset{};
    std::unique_ptr<parallel_mater::simulation::MemoryArena> arena;
    bool built{};
};

struct PmDomainMemory {
    using Arena = parallel_mater::simulation::MemoryArena;
    std::vector<std::unique_ptr<Arena>> fluids, cloths, ropes, soft_bodies, smokes;
};

using PmRuntimeCoupling = std::variant<
    parallel_mater::FluidClothCouplingId,
    parallel_mater::FluidRopeCouplingId,
    parallel_mater::FluidSoftBodyCouplingId,
    parallel_mater::FluidSmokeCouplingId,
    parallel_mater::SoftBodyClothCouplingId,
    parallel_mater::RopeSoftBodyCouplingId,
    parallel_mater::RopeClothCouplingId,
    parallel_mater::SmokeRigidCouplingId,
    parallel_mater::SmokeClothCouplingId,
    parallel_mater::SmokeRopeCouplingId,
    parallel_mater::SmokeSoftBodyCouplingId>;

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
    parallel_mater::SimulationOptions simulation_options{};
    PmSimulationMemoryBinding binding{};
    std::unique_ptr<parallel_mater::simulation::MemoryArena> arena;
    PmDomainMemory domain_memory;
    std::unique_ptr<parallel_mater::SimulationRuntime> runtime;
    std::unordered_map<PmGeometry, parallel_mater::TriangleMeshId> geometries;
    std::vector<std::optional<parallel_mater::RigidBodyId>> rigid_bodies;
    // Public geometry handles outlive every recorded/pending command that uses
    // them. This parallel slot table lets replacement retire the runtime mesh
    // after its final rigid reference without retaining all historical meshes.
    std::vector<PmGeometry> rigid_body_geometries;
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
    // Endpoint resources borrow a public fluid slot. Track that slot beside
    // the private generation so fluid replacement can invalidate only its
    // dependants after the runtime retires them.
    std::vector<std::uint32_t> emitter_fluid_indices;
    std::vector<std::optional<parallel_mater::ParticleDestroyPlaneId>> destroyers;
    std::vector<std::uint32_t> destroyer_fluid_indices;
    std::vector<std::optional<parallel_mater::PaintFieldId>> paint_fields;
    // Rigid paint targets borrow public geometry just like bodies do. Keeping
    // the public handle per slot lets replacement retire the private mesh only
    // after the final body or paint reference disappears.
    std::vector<PmGeometry> paint_field_geometries;
    std::vector<std::optional<parallel_mater::PaintRuleId>> paint_rules;
    std::vector<PmCouplingState> requested_couplings;
    bool explicit_couplings{};
    std::vector<PmRuntimeCoupling> active_couplings;
    std::vector<PmCouplingState> active_coupling_states;
    std::uint64_t coupling_pipeline_serial{};
    bool couplings_initialized{};
    std::uint64_t submitted_steps{};
    std::uint32_t last_substeps{};
    std::atomic<PmResult> execution_result{PM_SUCCESS};
    std::atomic<std::uint32_t> pending_submissions{};
};

struct PmPhysicsPipeline_T {
    PmDevice_T *device{};
    std::uint64_t serial{};
    PmRigidSolverState rigid{};
    std::vector<PmLiquidMaterialState> fluids;
    std::vector<PmCouplingState> couplings;
    std::vector<PmClothMaterialState> cloths;
    std::vector<PmRopeMaterialState> ropes;
    std::vector<PmSoftBodyMaterialState> soft_bodies;
    std::vector<PmSmokeMaterialState> smokes;
};

struct PmQueryWrite {
    PmQueryPool pool{};
    std::uint32_t first{};
    std::uint32_t count{};
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
    std::shared_ptr<parallel_mater::device::SubmissionCompletion> completion;
    std::vector<PmQueryWrite> query_writes;
    std::vector<PmSimulation> simulation_resources;
    std::vector<std::function<PmResult(parallel_mater::device::Services &)>> commands;
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
    std::uint64_t value{};
    std::map<std::uint64_t, std::shared_ptr<parallel_mater::device::SubmissionCompletion>> signals;
    bool imported{};
    PmNativeHandle native{};
};

struct PmFence_T {
    PmDevice_T *device{};
    std::shared_ptr<parallel_mater::device::SubmissionCompletion> completion;
    bool submitted{};
    bool initially_signaled{};
};

struct PmQuerySlot {
    std::shared_ptr<parallel_mater::device::SubmissionCompletion> completion;
    parallel_mater::device::Event event{};
    bool available{};
    PmStepStatistics statistics{};
};

struct PmQueryPool_T {
    std::mutex mutex;
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

[[nodiscard]] inline PmResult require_device_ownership(PmBuffer buffer) noexcept {
    return buffer != nullptr && buffer->queue_family == 0U
        ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
}

[[nodiscard]] inline PmResult require_recording(PmCommandBuffer buffer) noexcept {
    if (buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (buffer->state != PmCommandState::recording) return PM_ERROR_INVALID_STATE;
    return buffer->latched_error;
}

inline void refresh_command_state(PmCommandBuffer buffer) noexcept {
    if (buffer == nullptr || buffer->state != PmCommandState::pending ||
        !buffer->completion)
        return;
    if (buffer->completion->status() != PM_NOT_READY)
        buffer->state = PmCommandState::executable;
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
    case StatusCode::backend_failure:
        if (status.backend_error < 0)
            return static_cast<PmResult>(status.backend_error);
        return parallel_mater::device::from_driver_error(status.backend_error);
    case StatusCode::internal_error: return PM_ERROR_INVALID_STATE;
    }
    return PM_ERROR_INVALID_STATE;
}

inline PmResult record(PmCommandBuffer buffer,
                       std::function<PmResult(
                           parallel_mater::device::Services &)> command) noexcept {
    const PmResult state = require_recording(buffer);
    if (state != PM_SUCCESS) return state;
#if defined(PARALLEL_MATER_TEST_FAILURE_DRIVER)
    if (std::getenv("PM_TEST_FAIL_RECORDING") != nullptr) {
        buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
        return buffer->latched_error;
    }
#endif
    try {
        buffer->commands.push_back(std::move(command));
        return PM_SUCCESS;
    } catch (const std::bad_alloc &) {
        buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
        return buffer->latched_error;
    }
}

inline PmResult record_simulation(PmCommandBuffer buffer, PmSimulation simulation,
    std::function<PmResult(device::Services &)> command) noexcept {
    const auto state = require_recording(buffer);
    if (state != PM_SUCCESS) return state;
    try {
        buffer->simulation_resources.push_back(simulation);
    } catch (const std::bad_alloc &) {
        return buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    return record(buffer, std::move(command));
}

inline PmResult record_query(PmCommandBuffer buffer, PmQueryPool pool,
    std::uint32_t first, std::uint32_t count,
    std::function<PmResult(device::Services &)> command) noexcept {
    try {
        buffer->query_writes.push_back({pool, first, count});
    } catch (const std::bad_alloc &) {
        return buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    return record(buffer, std::move(command));
}

[[nodiscard]] PmPhysicalDeviceFeatures supported_features() noexcept;
[[nodiscard]] PmResult find_interaction_capability(
    const PmPhysicalDeviceFeatures &features,
    const PmInteractionCapabilityKey &key,
    PmInteractionCapability &capability) noexcept;
[[nodiscard]] PFN_pmVoidFunction lookup_proc(const char *name) noexcept;
[[nodiscard]] PmResult validate_coupling(PmDevice device, const PmCouplingState &coupling) noexcept;
[[nodiscard]] PmResult clear_couplings(PmSimulation simulation) noexcept;
[[nodiscard]] PmResult initialize_couplings(
    PmSimulation simulation, PmPhysicsPipeline pipeline) noexcept;
[[nodiscard]] PmResult register_geometry(
    PmSimulation simulation, PmGeometry geometry,
    parallel_mater::device::Services &services,
    parallel_mater::TriangleMeshId &output) noexcept;
[[nodiscard]] PmResult retire_geometry_if_unused(
    PmSimulation simulation, PmGeometry geometry) noexcept;

} // namespace parallel_mater::api
