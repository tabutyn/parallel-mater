// SPDX-License-Identifier: MIT
#include <parallel_mater/vulkan.hpp>

#include "parallel_mater_vulkan_clear_spv.hpp"
#include "parallel_mater_vulkan_integrate_spv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace parallel_mater::vulkan {
namespace {

constexpr std::uint32_t k_workgroup_size = 64U;

Status success() noexcept { return {}; }
Status invalid_argument(const char *message) noexcept {
    return {StatusCode::invalid_argument, VK_SUCCESS, message};
}
Status not_supported(const char *message) noexcept {
    return {StatusCode::not_supported, VK_SUCCESS, message};
}
Status invalid_handle(const char *message) noexcept {
    return {StatusCode::invalid_handle, VK_SUCCESS, message};
}
Status capacity_exceeded(const char *message) noexcept {
    return {StatusCode::capacity_exceeded, VK_SUCCESS, message};
}
Status busy(const char *message) noexcept {
    return {StatusCode::busy, VK_SUCCESS, message};
}
Status out_of_memory(VkResult result, const char *message) noexcept {
    return {StatusCode::out_of_memory, result, message};
}
Status vulkan_failure(VkResult result, const char *message) noexcept {
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY ||
        result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        return out_of_memory(result, message);
    }
    return {StatusCode::vulkan_failure, result, message};
}
Status internal_error(const char *message) noexcept {
    return {StatusCode::internal_error, VK_SUCCESS, message};
}

bool finite(Vec3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool finite(Quaternion value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

bool finite(RigidBodyState value) noexcept {
    return finite(value.position) && finite(value.orientation) &&
           finite(value.linear_velocity) && finite(value.angular_velocity);
}

Vec3 add(Vec3 left, Vec3 right) noexcept {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 subtract(Vec3 left, Vec3 right) noexcept {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 multiply(Vec3 value, float scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 cross(Vec3 left, Vec3 right) noexcept {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

float length_squared(Vec3 value) noexcept {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

Quaternion normalized(Quaternion value) noexcept {
    const float inverse = 1.0F / std::sqrt(
        value.x * value.x + value.y * value.y + value.z * value.z +
        value.w * value.w);
    return {value.x * inverse, value.y * inverse, value.z * inverse,
            value.w * inverse};
}

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) noexcept {
    return alignment <= 1U ? value
                           : (value + alignment - 1U) & ~(alignment - 1U);
}

struct GpuRigidParameters {
    std::uint32_t motion{};
    std::uint32_t has_kinematic_target{};
    float inverse_mass{};
    float inverse_inertia_x{};
    float inverse_inertia_y{};
    float inverse_inertia_z{};
    float linear_damping{};
    float angular_damping{};
    float maximum_linear_speed{};
    float maximum_angular_speed{};
};

struct StepPushConstants {
    std::uint32_t body_count{};
    std::uint32_t substeps{};
    float timestep{};
    float gravity_x{};
    float gravity_y{};
    float gravity_z{};
};

static_assert(sizeof(Vec3) == 12U);
static_assert(sizeof(Quaternion) == 16U);
static_assert(sizeof(RigidBodyId) == 8U);
static_assert(sizeof(RigidBodyState) == 52U);
static_assert(sizeof(GpuRigidParameters) == 40U);
static_assert(sizeof(StepPushConstants) == 24U);

struct DeviceState {
    NativeContext native{};
    bool owned{};
    PFN_vkGetSemaphoreCounterValue get_semaphore_counter_value{};
    PFN_vkWaitSemaphores wait_semaphores{};

    ~DeviceState() {
        if (owned && native.device != VK_NULL_HANDLE) {
            vkDestroyDevice(native.device, nullptr);
        }
        if (owned && native.instance != VK_NULL_HANDLE) {
            vkDestroyInstance(native.instance, nullptr);
        }
    }
};

struct CompletionState {
    std::shared_ptr<DeviceState> device{};
    VkSemaphore semaphore{VK_NULL_HANDLE};
    std::uint64_t value{};

    ~CompletionState() {
        if (semaphore == VK_NULL_HANDLE || !device ||
            device->native.device == VK_NULL_HANDLE) {
            return;
        }
        if (value != 0U) {
            VkSemaphoreWaitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait_info.semaphoreCount = 1U;
            wait_info.pSemaphores = &semaphore;
            wait_info.pValues = &value;
            (void)device->wait_semaphores(device->native.device, &wait_info,
                                          UINT64_MAX);
        }
        vkDestroySemaphore(device->native.device, semaphore, nullptr);
    }
};

bool completion_ready(const std::shared_ptr<CompletionState> &completion,
                      std::uint64_t value) noexcept {
    if (!completion || value == 0U) return true;
    std::uint64_t counter = 0U;
    return completion->device->get_semaphore_counter_value(
               completion->device->native.device, completion->semaphore,
               &counter) == VK_SUCCESS &&
           counter >= value;
}

Status wait_for_completion(const std::shared_ptr<CompletionState> &completion,
                           std::uint64_t value) noexcept {
    if (!completion || value == 0U) return success();
    VkSemaphoreWaitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait_info.semaphoreCount = 1U;
    wait_info.pSemaphores = &completion->semaphore;
    wait_info.pValues = &value;
    const VkResult result = completion->device->wait_semaphores(
        completion->device->native.device, &wait_info, UINT64_MAX);
    return result == VK_SUCCESS
        ? success()
        : vulkan_failure(result, "Vulkan frame completion wait failed");
}

struct BufferArena {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize allocation_size{};
    void *mapped{};
    bool coherent{};
};

struct BufferLayout {
    VkDeviceSize ids{};
    VkDeviceSize states{};
    VkDeviceSize previous_states{};
    VkDeviceSize parameters{};
    VkDeviceSize forces{};
    VkDeviceSize torques{};
    VkDeviceSize targets{};
    VkDeviceSize impulses{};
    VkDeviceSize angular_impulses{};
    VkDeviceSize total{};
};

struct MeshSlot {
    std::vector<Vec3> vertices{};
    std::vector<std::uint32_t> indices{};
    Vec3 minimum{};
    Vec3 maximum{};
    std::uint32_t generation{1U};
    bool alive{};
};

struct HandleSlot {
    std::uint32_t dense_index{};
    std::uint32_t generation{1U};
    bool alive{};
};

std::uint32_t find_memory_type(
    VkPhysicalDevice physical_device, std::uint32_t allowed,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred) noexcept {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    std::uint32_t fallback = UINT32_MAX;
    for (std::uint32_t index = 0U; index < properties.memoryTypeCount; ++index) {
        if ((allowed & (1U << index)) == 0U) continue;
        const VkMemoryPropertyFlags flags =
            properties.memoryTypes[index].propertyFlags;
        if ((flags & required) != required) continue;
        if ((flags & preferred) == preferred) return index;
        if (fallback == UINT32_MAX) fallback = index;
    }
    return fallback;
}

Status create_buffer_arena(
    const std::shared_ptr<DeviceState> &device, VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred, BufferArena &output) noexcept {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(device->native.device, &buffer_info,
                                     nullptr, &output.buffer);
    if (result != VK_SUCCESS) {
        return vulkan_failure(result, "Could not create Vulkan buffer arena");
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device->native.device, output.buffer,
                                  &requirements);
    const std::uint32_t memory_type = find_memory_type(
        device->native.physical_device, requirements.memoryTypeBits,
        required, preferred);
    if (memory_type == UINT32_MAX) {
        vkDestroyBuffer(device->native.device, output.buffer, nullptr);
        output.buffer = VK_NULL_HANDLE;
        return not_supported("Vulkan device has no compatible buffer memory type");
    }
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(device->native.physical_device,
                                        &memory_properties);
    output.coherent =
        (memory_properties.memoryTypes[memory_type].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;

    VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = memory_type;
    result = vkAllocateMemory(device->native.device, &allocate_info, nullptr,
                              &output.memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device->native.device, output.buffer, nullptr);
        output.buffer = VK_NULL_HANDLE;
        return vulkan_failure(result, "Could not allocate Vulkan buffer memory");
    }
    output.allocation_size = requirements.size;
    result = vkBindBufferMemory(device->native.device, output.buffer,
                                output.memory, 0U);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device->native.device, output.buffer, nullptr);
        vkFreeMemory(device->native.device, output.memory, nullptr);
        output = {};
        return vulkan_failure(result, "Could not bind Vulkan buffer memory");
    }
    if ((required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0U) {
        result = vkMapMemory(device->native.device, output.memory, 0U,
                             VK_WHOLE_SIZE, 0U, &output.mapped);
        if (result != VK_SUCCESS) {
            vkDestroyBuffer(device->native.device, output.buffer, nullptr);
            vkFreeMemory(device->native.device, output.memory, nullptr);
            output = {};
            return vulkan_failure(result, "Could not map Vulkan buffer memory");
        }
    }
    return success();
}

void destroy_buffer_arena(VkDevice device, BufferArena &arena) noexcept {
    if (arena.mapped != nullptr) vkUnmapMemory(device, arena.memory);
    if (arena.buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, arena.buffer, nullptr);
    if (arena.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, arena.memory, nullptr);
    arena = {};
}

Status create_timeline_completion(
    const std::shared_ptr<DeviceState> &device,
    std::shared_ptr<CompletionState> &output) {
    auto state = std::make_shared<CompletionState>();
    state->device = device;
    VkSemaphoreTypeCreateInfo type_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue = 0U;
    VkSemaphoreCreateInfo create_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    create_info.pNext = &type_info;
    const VkResult result = vkCreateSemaphore(
        device->native.device, &create_info, nullptr, &state->semaphore);
    if (result != VK_SUCCESS) {
        return vulkan_failure(result,
                              "Could not create Vulkan timeline semaphore");
    }
    output = std::move(state);
    return success();
}

PFN_vkGetPhysicalDeviceFeatures2 load_get_physical_device_features2(
    VkInstance instance) noexcept {
    auto function = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
    if (!function) {
        function = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            vkGetInstanceProcAddr(instance,
                                  "vkGetPhysicalDeviceFeatures2KHR"));
    }
    return function;
}

Status load_device_functions(DeviceState &device) noexcept {
    device.get_semaphore_counter_value =
        reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(vkGetDeviceProcAddr(
            device.native.device, "vkGetSemaphoreCounterValue"));
    if (!device.get_semaphore_counter_value) {
        device.get_semaphore_counter_value =
            reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(vkGetDeviceProcAddr(
                device.native.device, "vkGetSemaphoreCounterValueKHR"));
    }
    device.wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(
        vkGetDeviceProcAddr(device.native.device, "vkWaitSemaphores"));
    if (!device.wait_semaphores) {
        device.wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(
            vkGetDeviceProcAddr(device.native.device, "vkWaitSemaphoresKHR"));
    }
    return device.get_semaphore_counter_value && device.wait_semaphores
        ? success()
        : not_supported(
              "Vulkan loader does not expose timeline semaphore functions");
}

Status validate_physical_device(VkInstance instance,
                                VkPhysicalDevice physical_device,
                                std::uint32_t queue_family_index,
                                bool require_timeline) {
    if (physical_device == VK_NULL_HANDLE)
        return invalid_argument("Vulkan physical device is null");
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_2)
        return not_supported("ParallelMater Vulkan requires Vulkan 1.2");
    if (properties.limits.maxComputeWorkGroupInvocations < k_workgroup_size ||
        properties.limits.maxComputeWorkGroupSize[0] < k_workgroup_size)
        return not_supported("Vulkan compute workgroup limit is below 64");

    std::uint32_t count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    if (queue_family_index >= count)
        return invalid_argument("Vulkan queue family index is invalid");
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count,
                                             families.data());
    if (families[queue_family_index].queueCount == 0U ||
        (families[queue_family_index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0U)
        return not_supported("Vulkan queue family does not support compute");

    if (require_timeline) {
        const auto get_features = load_get_physical_device_features2(instance);
        if (!get_features)
            return not_supported(
                "Vulkan loader does not expose physical-device feature queries");
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        VkPhysicalDeviceFeatures2 features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &timeline;
        get_features(physical_device, &features);
        if (timeline.timelineSemaphore != VK_TRUE)
            return not_supported("Vulkan device lacks timeline semaphores");
    }
    return success();
}

Status create_owned_device(std::shared_ptr<DeviceState> &output) {
    auto state = std::make_shared<DeviceState>();
    state->owned = true;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "ParallelMater";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.pEngineName = "ParallelMater";
    application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;

    VkResult result = vkCreateInstance(&instance_info, nullptr,
                                       &state->native.instance);
    if (result != VK_SUCCESS)
        return vulkan_failure(result, "Could not create Vulkan 1.2 instance");
    const VkInstance instance = state->native.instance;

    std::uint32_t device_count = 0U;
    result = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0U) {
        return result == VK_SUCCESS
            ? not_supported("No Vulkan physical device is available")
            : vulkan_failure(result, "Could not enumerate Vulkan devices");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    result = vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    if (result != VK_SUCCESS) {
        return vulkan_failure(result, "Could not enumerate Vulkan devices");
    }

    struct Candidate {
        VkPhysicalDevice device{VK_NULL_HANDLE};
        std::uint32_t family{};
        int score{-1};
    } best;
    const auto get_features = load_get_physical_device_features2(instance);
    if (!get_features) {
        return not_supported(
            "Vulkan loader does not expose physical-device feature queries");
    }
    for (VkPhysicalDevice physical : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_2) continue;
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        VkPhysicalDeviceFeatures2 features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &timeline;
        get_features(physical, &features);
        if (timeline.timelineSemaphore != VK_TRUE ||
            properties.limits.maxComputeWorkGroupInvocations < k_workgroup_size ||
            properties.limits.maxComputeWorkGroupSize[0] < k_workgroup_size)
            continue;
        std::uint32_t family_count = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count,
                                                 families.data());
        for (std::uint32_t family = 0U; family < family_count; ++family) {
            if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0U ||
                families[family].queueCount == 0U)
                continue;
            int score = 0;
            switch (properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 500; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 400; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score = 300; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: score = 200; break;
            default: score = 100; break;
            }
            if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U)
                score += 10;
            if (score > best.score) best = {physical, family, score};
        }
    }
    if (best.device == VK_NULL_HANDLE) {
        return not_supported(
            "No Vulkan 1.2 compute device with timeline semaphores is available");
    }

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = best.family;
    queue_info.queueCount = 1U;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &timeline;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueCreateInfos = &queue_info;
    result = vkCreateDevice(best.device, &device_info, nullptr,
                            &state->native.device);
    if (result != VK_SUCCESS) {
        return vulkan_failure(result, "Could not create Vulkan compute device");
    }
    const VkDevice logical = state->native.device;
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(logical, best.family, 0U, &queue);
    state->native = {instance, best.device, logical, queue, best.family, true};
    Status status = load_device_functions(*state);
    if (!status) return status;
    output = std::move(state);
    return success();
}

Status create_borrowed_device(NativeContext context,
                              std::shared_ptr<DeviceState> &output) {
    if (context.instance == VK_NULL_HANDLE ||
        context.physical_device == VK_NULL_HANDLE ||
        context.device == VK_NULL_HANDLE || context.queue == VK_NULL_HANDLE)
        return invalid_argument("Borrowed Vulkan context has null handles");
    if (!context.timeline_semaphore_enabled)
        return not_supported(
            "Borrowed Vulkan device must enable timelineSemaphore");
    Status status = validate_physical_device(
        context.instance, context.physical_device, context.queue_family_index,
        true);
    if (!status) return status;
    auto state = std::make_shared<DeviceState>();
    state->native = context;
    status = load_device_functions(*state);
    if (!status) return status;
    output = std::move(state);
    return success();
}

} // namespace

struct FrameToken::Impl {
    std::shared_ptr<CompletionState> completion{};
};

FrameToken::FrameToken() noexcept = default;
FrameToken::~FrameToken() = default;
FrameToken::FrameToken(FrameToken &&) noexcept = default;
FrameToken &FrameToken::operator=(FrameToken &&) noexcept = default;

bool FrameToken::pending() const noexcept {
    return impl_ != nullptr && impl_->completion != nullptr &&
           !completion_ready(impl_->completion, impl_->completion->value);
}

bool FrameToken::ready() const noexcept { return !pending(); }

Status FrameToken::wait() noexcept {
    return impl_ == nullptr
        ? success()
        : wait_for_completion(impl_->completion, impl_->completion->value);
}

NativeCompletion FrameToken::native_completion() const noexcept {
    if (impl_ == nullptr || !impl_->completion) return {};
    return {impl_->completion->semaphore, impl_->completion->value};
}

struct World::Impl {
    WorldOptions options{};
    std::shared_ptr<DeviceState> device{};
    VkPhysicalDeviceProperties device_properties{};
    VkQueueFamilyProperties queue_properties{};
    BufferLayout layout{};
    BufferArena device_arena{};
    BufferArena upload_arena{};
    BufferArena readback_arena{};
    VkCommandPool command_pool{VK_NULL_HANDLE};
    VkCommandBuffer command_buffer{VK_NULL_HANDLE};
    VkFence upload_fence{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptor_set_layout{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
    VkPipeline integrate_pipeline{VK_NULL_HANDLE};
    VkPipeline clear_pipeline{VK_NULL_HANDLE};
    VkQueryPool query_pool{VK_NULL_HANDLE};
    bool timestamps_supported{};
    bool faulted{};
    VkResult fault_result{VK_SUCCESS};
    mutable std::mutex mutex{};

    std::vector<MeshSlot> meshes{};
    std::vector<HandleSlot> rigid_slots{};
    std::vector<RigidBodyId> ids{};
    std::vector<RigidBodyState> states{};
    std::vector<RigidBodyState> previous_states{};
    std::vector<GpuRigidParameters> parameters{};
    std::vector<Vec3> forces{};
    std::vector<Vec3> torques{};
    std::vector<RigidBodyState> targets{};
    std::vector<Vec3> impulses{};
    std::vector<Vec3> angular_impulses{};
    std::vector<TriangleMeshId> rigid_meshes{};
    std::uint32_t rigid_body_count{};
    std::uint32_t triangle_mesh_count{};
    std::uint64_t revision{};
    std::uint64_t frame_index{};
    std::shared_ptr<CompletionState> latest_completion{};
    std::uint64_t latest_value{};
    std::uint64_t finalized_value{};
    bool last_frame_profiled{};
    WorldStepTimings timings{};
    FrameToken synchronous_completion{};

    ~Impl() {
        if (!device) return;
        (void)wait_for_completion(latest_completion, latest_value);
        const VkDevice logical = device->native.device;
        if (query_pool != VK_NULL_HANDLE)
            vkDestroyQueryPool(logical, query_pool, nullptr);
        if (integrate_pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(logical, integrate_pipeline, nullptr);
        if (clear_pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(logical, clear_pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(logical, pipeline_layout, nullptr);
        if (descriptor_pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(logical, descriptor_pool, nullptr);
        if (descriptor_set_layout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(logical, descriptor_set_layout,
                                         nullptr);
        if (upload_fence != VK_NULL_HANDLE)
            vkDestroyFence(logical, upload_fence, nullptr);
        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(logical, command_pool, nullptr);
        destroy_buffer_arena(logical, readback_arena);
        destroy_buffer_arena(logical, upload_arena);
        destroy_buffer_arena(logical, device_arena);
    }

    [[nodiscard]] Status mark_failure(VkResult result,
                                      const char *message) noexcept {
        if (result == VK_ERROR_DEVICE_LOST) {
            faulted = true;
            fault_result = result;
        }
        return vulkan_failure(result, message);
    }

    [[nodiscard]] Status healthy() const noexcept {
        return faulted
            ? vulkan_failure(fault_result,
                             "Vulkan World is unusable after device loss")
            : success();
    }

    [[nodiscard]] bool valid(TriangleMeshId id) const noexcept {
        return id.index < meshes.size() && meshes[id.index].alive &&
               meshes[id.index].generation == id.generation;
    }

    [[nodiscard]] bool valid(RigidBodyId id,
                             std::uint32_t &dense_index) const noexcept {
        if (id.index >= rigid_slots.size()) return false;
        const HandleSlot &slot = rigid_slots[id.index];
        if (!slot.alive || slot.generation != id.generation) return false;
        dense_index = slot.dense_index;
        return dense_index < rigid_body_count && ids[dense_index] == id;
    }

    [[nodiscard]] Status flush_upload() noexcept {
        if (upload_arena.coherent) return success();
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = upload_arena.memory;
        range.offset = 0U;
        range.size = VK_WHOLE_SIZE;
        const VkResult result =
            vkFlushMappedMemoryRanges(device->native.device, 1U, &range);
        return result == VK_SUCCESS
            ? success()
            : mark_failure(result, "Could not flush Vulkan upload memory");
    }

    [[nodiscard]] Status invalidate_readback() noexcept {
        if (readback_arena.coherent) return success();
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = readback_arena.memory;
        range.offset = 0U;
        range.size = VK_WHOLE_SIZE;
        const VkResult result =
            vkInvalidateMappedMemoryRanges(device->native.device, 1U, &range);
        return result == VK_SUCCESS
            ? success()
            : mark_failure(result, "Could not invalidate Vulkan readback memory");
    }

    template <typename T>
    void copy_to_upload(VkDeviceSize offset, const std::vector<T> &values,
                        std::uint32_t count) noexcept {
        if (count == 0U) return;
        std::memcpy(static_cast<std::byte *>(upload_arena.mapped) + offset,
                    values.data(), sizeof(T) * count);
    }

    void pack_upload() noexcept {
        std::memset(upload_arena.mapped, 0,
                    static_cast<std::size_t>(layout.total));
        copy_to_upload(layout.ids, ids, rigid_body_count);
        copy_to_upload(layout.states, states, rigid_body_count);
        copy_to_upload(layout.previous_states, previous_states,
                       rigid_body_count);
        copy_to_upload(layout.parameters, parameters, rigid_body_count);
        copy_to_upload(layout.forces, forces, rigid_body_count);
        copy_to_upload(layout.torques, torques, rigid_body_count);
        copy_to_upload(layout.targets, targets, rigid_body_count);
        copy_to_upload(layout.impulses, impulses, rigid_body_count);
        copy_to_upload(layout.angular_impulses, angular_impulses,
                       rigid_body_count);
    }

    [[nodiscard]] Status finalize(bool wait) noexcept {
        Status status = healthy();
        if (!status || latest_value == 0U || finalized_value == latest_value)
            return status;
        if (wait) {
            status = wait_for_completion(latest_completion, latest_value);
            if (!status) {
                if (status.vulkan_result == VK_ERROR_DEVICE_LOST) {
                    faulted = true;
                    fault_result = status.vulkan_result;
                }
                return status;
            }
        } else {
            std::uint64_t counter = 0U;
            const VkResult result = device->get_semaphore_counter_value(
                device->native.device, latest_completion->semaphore, &counter);
            if (result != VK_SUCCESS)
                return mark_failure(result,
                                    "Could not query Vulkan frame completion");
            if (counter < latest_value)
                return busy("A Vulkan frame is still in flight");
        }
        status = invalidate_readback();
        if (!status) return status;
        if (rigid_body_count != 0U) {
            std::memcpy(states.data(),
                        static_cast<const std::byte *>(readback_arena.mapped) +
                            layout.states,
                        sizeof(RigidBodyState) * rigid_body_count);
            std::memcpy(previous_states.data(),
                        static_cast<const std::byte *>(readback_arena.mapped) +
                            layout.previous_states,
                        sizeof(RigidBodyState) * rigid_body_count);
        }
        timings = {};
        timings.frame_index = frame_index;
        timings.available = last_frame_profiled;
        if (last_frame_profiled) {
            std::array<std::uint64_t, 4U> values{};
            const VkResult result = vkGetQueryPoolResults(
                device->native.device, query_pool, 0U,
                static_cast<std::uint32_t>(values.size()), sizeof(values),
                values.data(), sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT);
            if (result != VK_SUCCESS)
                return mark_failure(result,
                                    "Could not read Vulkan timestamp queries");
            const double milliseconds_per_tick =
                static_cast<double>(device_properties.limits.timestampPeriod) /
                1.0e6;
            timings.rigid_integration = {
                static_cast<float>((values[1] - values[0]) *
                                   milliseconds_per_tick),
                rigid_body_count == 0U ? 0U : 1U};
            timings.rigid_input_clear = {
                static_cast<float>((values[3] - values[2]) *
                                   milliseconds_per_tick),
                rigid_body_count == 0U ? 0U : 1U};
            timings.total_gpu_milliseconds = static_cast<float>(
                (values[3] - values[0]) * milliseconds_per_tick);
        }
        finalized_value = latest_value;
        return success();
    }

    [[nodiscard]] Status upload_now() noexcept {
        Status status = healthy();
        if (!status) return status;
        status = finalize(false);
        if (!status && status.code != StatusCode::busy) return status;
        if (status.code == StatusCode::busy) return status;
        pack_upload();
        status = flush_upload();
        if (!status) return status;
        VkResult result = vkResetCommandBuffer(command_buffer, 0U);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not reset Vulkan upload command buffer");
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffer, &begin);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not begin Vulkan upload command buffer");
        VkBufferCopy copy{0U, 0U, layout.total};
        vkCmdCopyBuffer(command_buffer, upload_arena.buffer,
                        device_arena.buffer, 1U, &copy);
        result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not end Vulkan upload command buffer");
        result = vkResetFences(device->native.device, 1U, &upload_fence);
        if (result != VK_SUCCESS)
            return mark_failure(result, "Could not reset Vulkan upload fence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1U;
        submit.pCommandBuffers = &command_buffer;
        result = vkQueueSubmit(device->native.queue, 1U, &submit, upload_fence);
        if (result == VK_SUCCESS) {
            result = vkWaitForFences(device->native.device, 1U, &upload_fence,
                                     VK_TRUE, UINT64_MAX);
        }
        return result == VK_SUCCESS
            ? success()
            : mark_failure(result, "Vulkan state upload failed");
    }

    [[nodiscard]] Status initialize() {
        const std::uint32_t capacity = options.rigid_body_capacity;
        vkGetPhysicalDeviceProperties(device->native.physical_device,
                                      &device_properties);
        std::uint32_t family_count = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(device->native.physical_device,
                                                 &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device->native.physical_device,
                                                 &family_count,
                                                 families.data());
        queue_properties = families[device->native.queue_family_index];

        if (device_properties.limits.maxPerStageDescriptorStorageBuffers < 9U ||
            device_properties.limits.maxDescriptorSetStorageBuffers < 9U)
            return not_supported(
                "Vulkan device exposes fewer than nine storage buffers");
        if (device_properties.limits.maxPushConstantsSize <
            sizeof(StepPushConstants))
            return not_supported(
                "Vulkan push-constant limit is below rigid-step ABI size");
        const std::uint64_t workgroups =
            (static_cast<std::uint64_t>(capacity) + k_workgroup_size - 1U) /
            k_workgroup_size;
        if (workgroups > device_properties.limits.maxComputeWorkGroupCount[0])
            return not_supported(
                "Rigid capacity exceeds Vulkan compute dispatch limit");

        const VkDeviceSize alignment = std::max<VkDeviceSize>(
            1U, device_properties.limits.minStorageBufferOffsetAlignment);
        VkDeviceSize cursor = 0U;
        auto place = [&](VkDeviceSize bytes, VkDeviceSize &offset) {
            cursor = align_up(cursor, alignment);
            offset = cursor;
            cursor += bytes;
        };
        place(sizeof(RigidBodyId) * capacity, layout.ids);
        place(sizeof(RigidBodyState) * capacity, layout.states);
        place(sizeof(RigidBodyState) * capacity, layout.previous_states);
        place(sizeof(GpuRigidParameters) * capacity, layout.parameters);
        place(sizeof(Vec3) * capacity, layout.forces);
        place(sizeof(Vec3) * capacity, layout.torques);
        place(sizeof(RigidBodyState) * capacity, layout.targets);
        place(sizeof(Vec3) * capacity, layout.impulses);
        place(sizeof(Vec3) * capacity, layout.angular_impulses);
        layout.total = align_up(cursor, alignment);

        const std::array<VkDeviceSize, 9U> ranges{
            sizeof(RigidBodyId) * capacity,
            sizeof(RigidBodyState) * capacity,
            sizeof(RigidBodyState) * capacity,
            sizeof(GpuRigidParameters) * capacity,
            sizeof(Vec3) * capacity,
            sizeof(Vec3) * capacity,
            sizeof(RigidBodyState) * capacity,
            sizeof(Vec3) * capacity,
            sizeof(Vec3) * capacity,
        };
        for (VkDeviceSize range : ranges) {
            if (range > device_properties.limits.maxStorageBufferRange)
                return not_supported(
                    "Rigid capacity exceeds Vulkan storage-buffer range");
        }

        Status status = create_buffer_arena(
            device, layout.total,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            0U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, device_arena);
        if (!status) return status;
        status = create_buffer_arena(
            device, layout.total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, upload_arena);
        if (!status) return status;
        status = create_buffer_arena(
            device, layout.total, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, readback_arena);
        if (!status) return status;

        VkCommandPoolCreateInfo command_pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = device->native.queue_family_index;
        VkResult result = vkCreateCommandPool(
            device->native.device, &command_pool_info, nullptr, &command_pool);
        if (result != VK_SUCCESS)
            return mark_failure(result, "Could not create Vulkan command pool");
        VkCommandBufferAllocateInfo command_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = command_pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1U;
        result = vkAllocateCommandBuffers(device->native.device, &command_info,
                                          &command_buffer);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not allocate Vulkan command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        result = vkCreateFence(device->native.device, &fence_info, nullptr,
                               &upload_fence);
        if (result != VK_SUCCESS)
            return mark_failure(result, "Could not create Vulkan upload fence");

        std::array<VkDescriptorSetLayoutBinding, 9U> bindings{};
        for (std::uint32_t index = 0U; index < bindings.size(); ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1U;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo descriptor_layout_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptor_layout_info.bindingCount =
            static_cast<std::uint32_t>(bindings.size());
        descriptor_layout_info.pBindings = bindings.data();
        result = vkCreateDescriptorSetLayout(
            device->native.device, &descriptor_layout_info, nullptr,
            &descriptor_set_layout);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not create Vulkan descriptor layout");
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9U};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1U;
        pool_info.poolSizeCount = 1U;
        pool_info.pPoolSizes = &pool_size;
        result = vkCreateDescriptorPool(device->native.device, &pool_info,
                                        nullptr, &descriptor_pool);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not create Vulkan descriptor pool");
        VkDescriptorSetAllocateInfo descriptor_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        descriptor_info.descriptorPool = descriptor_pool;
        descriptor_info.descriptorSetCount = 1U;
        descriptor_info.pSetLayouts = &descriptor_set_layout;
        result = vkAllocateDescriptorSets(device->native.device,
                                          &descriptor_info, &descriptor_set);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not allocate Vulkan descriptor set");

        const std::array<VkDeviceSize, 9U> offsets{
            layout.ids, layout.states, layout.previous_states,
            layout.parameters, layout.forces, layout.torques, layout.targets,
            layout.impulses, layout.angular_impulses,
        };
        std::array<VkDescriptorBufferInfo, 9U> buffer_infos{};
        std::array<VkWriteDescriptorSet, 9U> writes{};
        for (std::uint32_t index = 0U; index < writes.size(); ++index) {
            buffer_infos[index] = {device_arena.buffer, offsets[index],
                                   ranges[index]};
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = descriptor_set;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1U;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &buffer_infos[index];
        }
        vkUpdateDescriptorSets(device->native.device,
                               static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0U, nullptr);

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = sizeof(StepPushConstants);
        VkPipelineLayoutCreateInfo pipeline_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1U;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1U;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        result = vkCreatePipelineLayout(device->native.device,
                                        &pipeline_layout_info, nullptr,
                                        &pipeline_layout);
        if (result != VK_SUCCESS)
            return mark_failure(result,
                                "Could not create Vulkan pipeline layout");

        auto create_pipeline = [&](const unsigned char *bytes,
                                   std::size_t size, const char *entry,
                                   VkPipeline &pipeline) -> Status {
            VkShaderModuleCreateInfo shader_info{
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = size;
            shader_info.pCode = reinterpret_cast<const std::uint32_t *>(bytes);
            VkShaderModule module = VK_NULL_HANDLE;
            VkResult local_result = vkCreateShaderModule(
                device->native.device, &shader_info, nullptr, &module);
            if (local_result != VK_SUCCESS)
                return mark_failure(local_result,
                                    "Could not create Vulkan shader module");
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = entry;
            VkComputePipelineCreateInfo pipeline_info{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout;
            local_result = vkCreateComputePipelines(
                device->native.device, VK_NULL_HANDLE, 1U, &pipeline_info,
                nullptr, &pipeline);
            vkDestroyShaderModule(device->native.device, module, nullptr);
            return local_result == VK_SUCCESS
                ? success()
                : mark_failure(local_result,
                               "Could not create Vulkan compute pipeline");
        };
        status = create_pipeline(parallel_mater_vulkan_integrate_spv,
                                 parallel_mater_vulkan_integrate_spv_size,
                                 "pm_rigid_integrate", integrate_pipeline);
        if (!status) return status;
        status = create_pipeline(parallel_mater_vulkan_clear_spv,
                                 parallel_mater_vulkan_clear_spv_size,
                                 "pm_rigid_clear_inputs", clear_pipeline);
        if (!status) return status;

        timestamps_supported = queue_properties.timestampValidBits != 0U;
        if (timestamps_supported) {
            VkQueryPoolCreateInfo query_info{
                VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query_info.queryCount = 4U;
            result = vkCreateQueryPool(device->native.device, &query_info,
                                       nullptr, &query_pool);
            if (result != VK_SUCCESS)
                return mark_failure(result,
                                    "Could not create Vulkan timestamp pool");
        }

        meshes.resize(options.triangle_mesh_capacity);
        rigid_slots.resize(capacity);
        ids.resize(capacity);
        states.resize(capacity);
        previous_states.resize(capacity);
        parameters.resize(capacity);
        forces.resize(capacity);
        torques.resize(capacity);
        targets.resize(capacity);
        impulses.resize(capacity);
        angular_impulses.resize(capacity);
        rigid_meshes.resize(capacity);
        return success();
    }
};

namespace {

bool valid_state(RigidBodyState state) noexcept {
    return finite(state) &&
           (state.orientation.x * state.orientation.x +
            state.orientation.y * state.orientation.y +
            state.orientation.z * state.orientation.z +
            state.orientation.w * state.orientation.w) > 1.0e-20F;
}

std::uint32_t next_generation(std::uint32_t generation) noexcept {
    ++generation;
    return generation == 0U ? 1U : generation;
}

} // namespace

World::World() noexcept = default;
World::~World() = default;
World::World(World &&) noexcept = default;
World &World::operator=(World &&) noexcept = default;

Status World::create(WorldOptions options, World &output) noexcept {
    if (options.rigid_body_capacity == 0U ||
        options.triangle_mesh_capacity == 0U)
        return invalid_argument("Vulkan capacities must be nonzero");
    if (options.rigid_sleeping)
        return not_supported("Vulkan rigid sleeping is not implemented");
    if (options.physics_debug.frame_capacity != 0U)
        return not_supported("Vulkan physics-debug capture is not implemented");
    try {
        std::shared_ptr<DeviceState> device;
        Status status = create_owned_device(device);
        if (!status) return status;
        auto impl = std::make_unique<Impl>();
        impl->options = options;
        impl->device = std::move(device);
        status = impl->initialize();
        if (!status) return status;
        output.impl_ = std::move(impl);
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory(VK_ERROR_OUT_OF_HOST_MEMORY,
                             "Could not allocate Vulkan World resources");
    } catch (...) {
        return internal_error("Unexpected owned Vulkan World creation failure");
    }
}

Status World::create(WorldOptions options, NativeContext context,
                     World &output) noexcept {
    if (options.rigid_body_capacity == 0U ||
        options.triangle_mesh_capacity == 0U)
        return invalid_argument("Vulkan capacities must be nonzero");
    if (options.rigid_sleeping)
        return not_supported("Vulkan rigid sleeping is not implemented");
    if (options.physics_debug.frame_capacity != 0U)
        return not_supported("Vulkan physics-debug capture is not implemented");
    try {
        std::shared_ptr<DeviceState> device;
        Status status = create_borrowed_device(context, device);
        if (!status) return status;
        auto impl = std::make_unique<Impl>();
        impl->options = options;
        impl->device = std::move(device);
        status = impl->initialize();
        if (!status) return status;
        output.impl_ = std::move(impl);
        return success();
    } catch (const std::bad_alloc &) {
        return out_of_memory(VK_ERROR_OUT_OF_HOST_MEMORY,
                             "Could not allocate Vulkan World resources");
    } catch (...) {
        return internal_error(
            "Unexpected borrowed Vulkan World creation failure");
    }
}

Status World::add_triangle_mesh(
    HostSpan<const Vec3> vertices,
    HostSpan<const std::uint32_t> triangle_indices,
    TriangleMeshId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Triangle meshes cannot change while a frame is in flight")
        : status;
    if (vertices.data == nullptr || triangle_indices.data == nullptr ||
        vertices.size < 3U || triangle_indices.size < 3U ||
        triangle_indices.size % 3U != 0U)
        return invalid_argument(
            "Triangle mesh requires vertices and complete triangle indices");
    if (impl_->triangle_mesh_count >= impl_->options.triangle_mesh_capacity)
        return capacity_exceeded("Triangle mesh capacity is exhausted");
    for (std::uint64_t index = 0U; index < vertices.size; ++index) {
        if (!finite(vertices.data[index]))
            return invalid_argument(
                "Triangle mesh contains a non-finite vertex");
    }
    for (std::uint64_t index = 0U; index < triangle_indices.size; index += 3U) {
        const std::uint32_t first = triangle_indices.data[index];
        const std::uint32_t second = triangle_indices.data[index + 1U];
        const std::uint32_t third = triangle_indices.data[index + 2U];
        if (first >= vertices.size || second >= vertices.size ||
            third >= vertices.size)
            return invalid_argument(
                "Triangle mesh index is outside the vertex span");
        const Vec3 area = cross(subtract(vertices.data[second],
                                         vertices.data[first]),
                                subtract(vertices.data[third],
                                         vertices.data[first]));
        if (length_squared(area) <= 1.0e-12F)
            return invalid_argument(
                "Triangle mesh contains a degenerate triangle");
    }
    std::uint32_t slot_index = UINT32_MAX;
    for (std::uint32_t index = 0U; index < impl_->meshes.size(); ++index) {
        if (!impl_->meshes[index].alive) {
            slot_index = index;
            break;
        }
    }
    if (slot_index == UINT32_MAX)
        return internal_error("No free triangle mesh slot was found");
    try {
        MeshSlot &mesh = impl_->meshes[slot_index];
        mesh.vertices.assign(vertices.data, vertices.data + vertices.size);
        mesh.indices.assign(triangle_indices.data,
                            triangle_indices.data + triangle_indices.size);
        mesh.minimum = mesh.maximum = mesh.vertices.front();
        for (Vec3 vertex : mesh.vertices) {
            mesh.minimum.x = std::min(mesh.minimum.x, vertex.x);
            mesh.minimum.y = std::min(mesh.minimum.y, vertex.y);
            mesh.minimum.z = std::min(mesh.minimum.z, vertex.z);
            mesh.maximum.x = std::max(mesh.maximum.x, vertex.x);
            mesh.maximum.y = std::max(mesh.maximum.y, vertex.y);
            mesh.maximum.z = std::max(mesh.maximum.z, vertex.z);
        }
        mesh.alive = true;
        ++impl_->triangle_mesh_count;
        ++impl_->revision;
        output = {slot_index, mesh.generation};
        return success();
    } catch (const std::bad_alloc &) {
        MeshSlot &mesh = impl_->meshes[slot_index];
        mesh.vertices.clear();
        mesh.indices.clear();
        mesh.alive = false;
        return out_of_memory(VK_ERROR_OUT_OF_HOST_MEMORY,
                             "Could not copy triangle mesh host data");
    } catch (...) {
        return internal_error("Unexpected triangle mesh registration failure");
    }
}

Status World::remove_triangle_mesh(TriangleMeshId mesh_id) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Triangle meshes cannot change while a frame is in flight")
        : status;
    if (!impl_->valid(mesh_id))
        return invalid_handle("Triangle mesh handle is stale");
    for (std::uint32_t index = 0U; index < impl_->rigid_body_count; ++index) {
        if (impl_->rigid_meshes[index] == mesh_id)
            return invalid_argument(
                "Triangle mesh is still referenced by a rigid body");
    }
    MeshSlot &mesh = impl_->meshes[mesh_id.index];
    mesh.vertices.clear();
    mesh.indices.clear();
    mesh.alive = false;
    mesh.generation = next_generation(mesh.generation);
    --impl_->triangle_mesh_count;
    ++impl_->revision;
    return success();
}

Status World::add_rigid_body(RigidBodyOptions options,
                             RigidBodyId &output) noexcept {
    output = {};
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Rigid bodies cannot change while a frame is in flight")
        : status;
    if (!impl_->valid(options.mesh))
        return invalid_handle("Rigid body triangle mesh handle is stale");
    if (!valid_state(options.initial_state) ||
        options.motion > MotionType::dynamic ||
        !std::isfinite(options.mass) || options.mass <= 0.0F ||
        !finite(options.inertia_diagonal) || options.inertia_diagonal.x < 0.0F ||
        options.inertia_diagonal.y < 0.0F ||
        options.inertia_diagonal.z < 0.0F ||
        !std::isfinite(options.friction) || options.friction < 0.0F ||
        !std::isfinite(options.restitution) || options.restitution < 0.0F ||
        options.restitution > 1.0F ||
        !std::isfinite(options.collision_margin) ||
        options.collision_margin < 0.0F ||
        !std::isfinite(options.linear_damping) ||
        options.linear_damping < 0.0F ||
        !std::isfinite(options.angular_damping) ||
        options.angular_damping < 0.0F ||
        !std::isfinite(options.maximum_linear_speed) ||
        options.maximum_linear_speed <= 0.0F ||
        !std::isfinite(options.maximum_angular_speed) ||
        options.maximum_angular_speed <= 0.0F)
        return invalid_argument("Rigid body options are invalid");
    if (impl_->rigid_body_count >= impl_->options.rigid_body_capacity)
        return capacity_exceeded("Rigid body capacity is exhausted");

    Vec3 inertia = options.inertia_diagonal;
    if (inertia.x == 0.0F && inertia.y == 0.0F && inertia.z == 0.0F) {
        const MeshSlot &mesh = impl_->meshes[options.mesh.index];
        const Vec3 half = multiply(subtract(mesh.maximum, mesh.minimum), 0.5F);
        inertia = {
            options.mass * std::max((half.y * half.y + half.z * half.z) / 3.0F,
                                    1.0e-6F),
            options.mass * std::max((half.x * half.x + half.z * half.z) / 3.0F,
                                    1.0e-6F),
            options.mass * std::max((half.x * half.x + half.y * half.y) / 3.0F,
                                    1.0e-6F),
        };
    } else if (!(inertia.x > 0.0F && inertia.y > 0.0F && inertia.z > 0.0F)) {
        return invalid_argument(
            "Explicit rigid body inertia must be positive on every axis");
    }

    std::uint32_t slot_index = UINT32_MAX;
    for (std::uint32_t index = 0U; index < impl_->rigid_slots.size(); ++index) {
        if (!impl_->rigid_slots[index].alive) {
            slot_index = index;
            break;
        }
    }
    if (slot_index == UINT32_MAX)
        return internal_error("No free rigid body handle slot was found");
    HandleSlot &slot = impl_->rigid_slots[slot_index];
    const std::uint32_t dense = impl_->rigid_body_count;
    slot.alive = true;
    slot.dense_index = dense;
    if (slot.generation == 0U) slot.generation = 1U;
    const RigidBodyId id{slot_index, slot.generation};
    impl_->ids[dense] = id;
    impl_->states[dense] = options.initial_state;
    impl_->states[dense].orientation = normalized(options.initial_state.orientation);
    impl_->previous_states[dense] = impl_->states[dense];
    impl_->parameters[dense] = {
        static_cast<std::uint32_t>(options.motion), 0U,
        options.motion == MotionType::dynamic ? 1.0F / options.mass : 0.0F,
        options.motion == MotionType::dynamic ? 1.0F / inertia.x : 0.0F,
        options.motion == MotionType::dynamic ? 1.0F / inertia.y : 0.0F,
        options.motion == MotionType::dynamic ? 1.0F / inertia.z : 0.0F,
        options.linear_damping, options.angular_damping,
        options.maximum_linear_speed, options.maximum_angular_speed,
    };
    impl_->forces[dense] = {};
    impl_->torques[dense] = {};
    impl_->targets[dense] = {};
    impl_->impulses[dense] = {};
    impl_->angular_impulses[dense] = {};
    impl_->rigid_meshes[dense] = options.mesh;
    ++impl_->rigid_body_count;
    ++impl_->revision;
    status = impl_->upload_now();
    if (!status) {
        --impl_->rigid_body_count;
        --impl_->revision;
        slot.alive = false;
        return status;
    }
    output = id;
    return success();
}

Status World::remove_rigid_body(RigidBodyId body) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Rigid bodies cannot change while a frame is in flight")
        : status;
    std::uint32_t dense = 0U;
    if (!impl_->valid(body, dense))
        return invalid_handle("Rigid body handle is stale");
    const std::uint32_t last = impl_->rigid_body_count - 1U;
    const HandleSlot old_slot = impl_->rigid_slots[body.index];
    const RigidBodyId removed_id = impl_->ids[dense];
    const RigidBodyState removed_state = impl_->states[dense];
    const RigidBodyState removed_previous = impl_->previous_states[dense];
    const GpuRigidParameters removed_parameters = impl_->parameters[dense];
    const Vec3 removed_force = impl_->forces[dense];
    const Vec3 removed_torque = impl_->torques[dense];
    const RigidBodyState removed_target = impl_->targets[dense];
    const Vec3 removed_impulse = impl_->impulses[dense];
    const Vec3 removed_angular_impulse = impl_->angular_impulses[dense];
    const TriangleMeshId removed_mesh = impl_->rigid_meshes[dense];
    if (dense != last) {
        impl_->ids[dense] = impl_->ids[last];
        impl_->states[dense] = impl_->states[last];
        impl_->previous_states[dense] = impl_->previous_states[last];
        impl_->parameters[dense] = impl_->parameters[last];
        impl_->forces[dense] = impl_->forces[last];
        impl_->torques[dense] = impl_->torques[last];
        impl_->targets[dense] = impl_->targets[last];
        impl_->impulses[dense] = impl_->impulses[last];
        impl_->angular_impulses[dense] = impl_->angular_impulses[last];
        impl_->rigid_meshes[dense] = impl_->rigid_meshes[last];
        impl_->rigid_slots[impl_->ids[dense].index].dense_index = dense;
    }
    --impl_->rigid_body_count;
    HandleSlot &slot = impl_->rigid_slots[body.index];
    slot.alive = false;
    slot.generation = next_generation(slot.generation);
    ++impl_->revision;
    status = impl_->upload_now();
    if (!status) {
        ++impl_->rigid_body_count;
        --impl_->revision;
        impl_->rigid_slots[body.index] = old_slot;
        if (dense != last) {
            impl_->rigid_slots[impl_->ids[dense].index].dense_index = last;
            impl_->ids[last] = impl_->ids[dense];
            impl_->states[last] = impl_->states[dense];
            impl_->previous_states[last] = impl_->previous_states[dense];
            impl_->parameters[last] = impl_->parameters[dense];
            impl_->forces[last] = impl_->forces[dense];
            impl_->torques[last] = impl_->torques[dense];
            impl_->targets[last] = impl_->targets[dense];
            impl_->impulses[last] = impl_->impulses[dense];
            impl_->angular_impulses[last] = impl_->angular_impulses[dense];
            impl_->rigid_meshes[last] = impl_->rigid_meshes[dense];
        }
        impl_->ids[dense] = removed_id;
        impl_->states[dense] = removed_state;
        impl_->previous_states[dense] = removed_previous;
        impl_->parameters[dense] = removed_parameters;
        impl_->forces[dense] = removed_force;
        impl_->torques[dense] = removed_torque;
        impl_->targets[dense] = removed_target;
        impl_->impulses[dense] = removed_impulse;
        impl_->angular_impulses[dense] = removed_angular_impulse;
        impl_->rigid_meshes[dense] = removed_mesh;
        return status;
    }
    return success();
}

Status World::set_rigid_body_state(RigidBodyId body,
                                   RigidBodyState state) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Rigid body state cannot change during a frame") : status;
    if (!valid_state(state))
        return invalid_argument("Rigid body state is invalid");
    std::uint32_t dense = 0U;
    if (!impl_->valid(body, dense))
        return invalid_handle("Rigid body handle is stale");
    const RigidBodyState old = impl_->states[dense];
    const RigidBodyState old_previous = impl_->previous_states[dense];
    const Vec3 old_force = impl_->forces[dense];
    const Vec3 old_torque = impl_->torques[dense];
    const Vec3 old_impulse = impl_->impulses[dense];
    const Vec3 old_angular_impulse = impl_->angular_impulses[dense];
    const std::uint32_t old_target =
        impl_->parameters[dense].has_kinematic_target;
    state.orientation = normalized(state.orientation);
    impl_->states[dense] = state;
    impl_->previous_states[dense] = state;
    impl_->forces[dense] = {};
    impl_->torques[dense] = {};
    impl_->impulses[dense] = {};
    impl_->angular_impulses[dense] = {};
    impl_->parameters[dense].has_kinematic_target = 0U;
    ++impl_->revision;
    status = impl_->upload_now();
    if (!status) {
        impl_->states[dense] = old;
        impl_->previous_states[dense] = old_previous;
        impl_->forces[dense] = old_force;
        impl_->torques[dense] = old_torque;
        impl_->impulses[dense] = old_impulse;
        impl_->angular_impulses[dense] = old_angular_impulse;
        impl_->parameters[dense].has_kinematic_target = old_target;
        --impl_->revision;
    }
    return status;
}

Status World::set_kinematic_target(RigidBodyId body,
                                   RigidBodyState target) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Kinematic targets cannot change during a frame") : status;
    if (!valid_state(target))
        return invalid_argument("Kinematic target is invalid");
    std::uint32_t dense = 0U;
    if (!impl_->valid(body, dense))
        return invalid_handle("Rigid body handle is stale");
    if (impl_->parameters[dense].motion !=
        static_cast<std::uint32_t>(MotionType::kinematic))
        return invalid_argument("Rigid body is not kinematic");
    target.orientation = normalized(target.orientation);
    impl_->targets[dense] = target;
    impl_->parameters[dense].has_kinematic_target = 1U;
    return success();
}

Status World::apply_force(RigidBodyId body, Vec3 force,
                          Vec3 world_point) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Forces cannot change during a frame") : status;
    if (!finite(force) || !finite(world_point))
        return invalid_argument("Force and world point must be finite");
    std::uint32_t dense = 0U;
    if (!impl_->valid(body, dense))
        return invalid_handle("Rigid body handle is stale");
    if (impl_->parameters[dense].motion !=
        static_cast<std::uint32_t>(MotionType::dynamic))
        return invalid_argument("Forces require a dynamic rigid body");
    impl_->forces[dense] = add(impl_->forces[dense], force);
    impl_->torques[dense] = add(
        impl_->torques[dense],
        cross(subtract(world_point, impl_->states[dense].position), force));
    return success();
}

Status World::apply_central_acceleration(
    HostSpan<RigidBodyId> bodies, Vec3 acceleration) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Accelerations cannot change during a frame") : status;
    if (!finite(acceleration) || (bodies.size != 0U && bodies.data == nullptr) ||
        bodies.size > impl_->rigid_body_count)
        return invalid_argument("Central acceleration batch is invalid");
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense = 0U;
        if (!impl_->valid(bodies.data[index], dense))
            return invalid_handle("Rigid body handle is stale");
        if (impl_->parameters[dense].motion !=
            static_cast<std::uint32_t>(MotionType::dynamic))
            return invalid_argument(
                "Central acceleration requires dynamic bodies");
        for (std::uint64_t prior = 0U; prior < index; ++prior) {
            if (bodies.data[prior] == bodies.data[index])
                return invalid_argument(
                    "Central acceleration body is duplicated");
        }
    }
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense = 0U;
        (void)impl_->valid(bodies.data[index], dense);
        const float mass = 1.0F / impl_->parameters[dense].inverse_mass;
        impl_->forces[dense] = add(
            impl_->forces[dense], multiply(acceleration, mass));
    }
    return success();
}

Status World::apply_impulse(RigidBodyId body, Vec3 impulse,
                            Vec3 world_point) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Impulses cannot change during a frame") : status;
    if (!finite(impulse) || !finite(world_point))
        return invalid_argument("Impulse and world point must be finite");
    std::uint32_t dense = 0U;
    if (!impl_->valid(body, dense))
        return invalid_handle("Rigid body handle is stale");
    if (impl_->parameters[dense].motion !=
        static_cast<std::uint32_t>(MotionType::dynamic))
        return invalid_argument("Impulses require a dynamic rigid body");
    impl_->impulses[dense] = add(impl_->impulses[dense], impulse);
    impl_->angular_impulses[dense] = add(
        impl_->angular_impulses[dense],
        cross(subtract(world_point, impl_->states[dense].position), impulse));
    return success();
}

Status World::rigid_body_view(RigidBodyDeviceView &output) const noexcept {
    output = {};
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Rigid body view requires a completed frame") : status;
    output.ids = {impl_->device_arena.buffer, impl_->layout.ids,
                  impl_->rigid_body_count};
    output.states = {impl_->device_arena.buffer, impl_->layout.states,
                     impl_->rigid_body_count};
    output.previous_states = {impl_->device_arena.buffer,
                              impl_->layout.previous_states,
                              impl_->rigid_body_count};
    output.revision = impl_->revision;
    return success();
}

Status World::read_rigid_body_state(RigidBodyId body,
                                    RigidBodyState &output) const noexcept {
    output = {};
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Rigid body readback requires a completed frame") : status;
    std::uint32_t dense = 0U;
    if (!impl_->valid(body, dense))
        return invalid_handle("Rigid body handle is stale");
    output = impl_->states[dense];
    return success();
}

Status World::step_async(StepOptions options,
                         FrameToken &completion) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->healthy();
    if (!status) return status;
    status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Only one Vulkan frame may be in flight") : status;
    if (!std::isfinite(options.timestep) || options.timestep <= 0.0F ||
        options.substeps == 0U || options.substeps > 1024U ||
        !finite(options.gravity))
        return invalid_argument("Vulkan step options are invalid");
    if (options.collect_rigid_contacts || options.collect_fluid_contacts)
        return not_supported("Vulkan contact collection is not implemented");
    if (options.collect_kernel_timings && !impl_->timestamps_supported)
        return not_supported(
            "Vulkan queue does not support compute timestamps");
    if (completion.pending())
        return busy("FrameToken still represents pending Vulkan work");

    try {
        if (!completion.impl_)
            completion.impl_ = std::make_unique<FrameToken::Impl>();
        if (!completion.impl_->completion ||
            completion.impl_->completion->device != impl_->device) {
            std::shared_ptr<CompletionState> timeline;
            status = create_timeline_completion(impl_->device, timeline);
            if (!status) return status;
            completion.impl_->completion = std::move(timeline);
        }
    } catch (const std::bad_alloc &) {
        return out_of_memory(VK_ERROR_OUT_OF_HOST_MEMORY,
                             "Could not allocate Vulkan frame token");
    }

    impl_->pack_upload();
    status = impl_->flush_upload();
    if (!status) return status;
    VkResult result = vkResetCommandBuffer(impl_->command_buffer, 0U);
    if (result != VK_SUCCESS)
        return impl_->mark_failure(result,
            "Could not reset Vulkan step command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(impl_->command_buffer, &begin);
    if (result != VK_SUCCESS)
        return impl_->mark_failure(result,
            "Could not begin Vulkan step command buffer");

    VkBufferCopy upload_copy{0U, 0U, impl_->layout.total};
    vkCmdCopyBuffer(impl_->command_buffer, impl_->upload_arena.buffer,
                    impl_->device_arena.buffer, 1U, &upload_copy);
    VkBufferMemoryBarrier upload_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    upload_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    upload_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    upload_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    upload_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    upload_barrier.buffer = impl_->device_arena.buffer;
    upload_barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(impl_->command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U,
                         0U, nullptr, 1U, &upload_barrier, 0U, nullptr);

    const bool profiled = options.collect_kernel_timings;
    if (profiled) {
        vkCmdResetQueryPool(impl_->command_buffer, impl_->query_pool, 0U, 4U);
        vkCmdWriteTimestamp(impl_->command_buffer,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            impl_->query_pool, 0U);
    }
    const StepPushConstants push{
        impl_->rigid_body_count, options.substeps, options.timestep,
        options.gravity.x, options.gravity.y, options.gravity.z,
    };
    const std::uint32_t groups =
        (impl_->rigid_body_count + k_workgroup_size - 1U) / k_workgroup_size;
    vkCmdBindPipeline(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      impl_->integrate_pipeline);
    vkCmdBindDescriptorSets(impl_->command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            impl_->pipeline_layout, 0U, 1U,
                            &impl_->descriptor_set, 0U, nullptr);
    vkCmdPushConstants(impl_->command_buffer, impl_->pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(push), &push);
    if (groups != 0U)
        vkCmdDispatch(impl_->command_buffer, groups, 1U, 1U);
    if (profiled) {
        vkCmdWriteTimestamp(impl_->command_buffer,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            impl_->query_pool, 1U);
    }

    VkBufferMemoryBarrier compute_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    compute_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    compute_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_barrier.buffer = impl_->device_arena.buffer;
    compute_barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(impl_->command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U,
                         0U, nullptr, 1U, &compute_barrier, 0U, nullptr);
    if (profiled) {
        vkCmdWriteTimestamp(impl_->command_buffer,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            impl_->query_pool, 2U);
    }
    vkCmdBindPipeline(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      impl_->clear_pipeline);
    if (groups != 0U)
        vkCmdDispatch(impl_->command_buffer, groups, 1U, 1U);
    if (profiled) {
        vkCmdWriteTimestamp(impl_->command_buffer,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            impl_->query_pool, 3U);
    }

    VkBufferMemoryBarrier readback_barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    readback_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    readback_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    readback_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readback_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readback_barrier.buffer = impl_->device_arena.buffer;
    readback_barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(impl_->command_buffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0U,
                         0U, nullptr, 1U, &readback_barrier, 0U, nullptr);
    VkBufferCopy readback_copy{0U, 0U, impl_->layout.total};
    vkCmdCopyBuffer(impl_->command_buffer, impl_->device_arena.buffer,
                    impl_->readback_arena.buffer, 1U, &readback_copy);
    result = vkEndCommandBuffer(impl_->command_buffer);
    if (result != VK_SUCCESS)
        return impl_->mark_failure(result,
            "Could not end Vulkan step command buffer");

    const std::uint64_t signal_value =
        completion.impl_->completion->value + 1U;
    VkTimelineSemaphoreSubmitInfo timeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.signalSemaphoreValueCount = 1U;
    timeline.pSignalSemaphoreValues = &signal_value;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &impl_->command_buffer;
    submit.signalSemaphoreCount = 1U;
    submit.pSignalSemaphores = &completion.impl_->completion->semaphore;
    result = vkQueueSubmit(impl_->device->native.queue, 1U, &submit,
                           VK_NULL_HANDLE);
    if (result != VK_SUCCESS)
        return impl_->mark_failure(result, "Vulkan frame submission failed");

    completion.impl_->completion->value = signal_value;
    impl_->latest_completion = completion.impl_->completion;
    impl_->latest_value = signal_value;
    impl_->last_frame_profiled = profiled;
    ++impl_->frame_index;
    ++impl_->revision;
    for (std::uint32_t index = 0U; index < impl_->rigid_body_count; ++index) {
        impl_->forces[index] = {};
        impl_->torques[index] = {};
        impl_->impulses[index] = {};
        impl_->angular_impulses[index] = {};
        impl_->parameters[index].has_kinematic_target = 0U;
    }
    return success();
}

Status World::step(StepOptions options) noexcept {
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    Status status = step_async(options, impl_->synchronous_completion);
    if (!status) return status;
    status = impl_->synchronous_completion.wait();
    if (!status) {
        std::lock_guard lock(impl_->mutex);
        if (status.vulkan_result == VK_ERROR_DEVICE_LOST) {
            impl_->faulted = true;
            impl_->fault_result = VK_ERROR_DEVICE_LOST;
        }
        return status;
    }
    std::lock_guard lock(impl_->mutex);
    return impl_->finalize(true);
}

Status World::collect_step_timings(WorldStepTimings &output) const noexcept {
    output = {};
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Step timings require a completed frame") : status;
    output = impl_->timings;
    return success();
}

Status World::collect_statistics(WorldStatistics &output) const noexcept {
    output = {};
    if (!impl_) return invalid_argument("Vulkan World is not initialized");
    std::lock_guard lock(impl_->mutex);
    Status status = impl_->finalize(false);
    if (!status) return status.code == StatusCode::busy
        ? busy("Statistics require a completed frame") : status;
    output.frame_index = impl_->frame_index;
    output.rigid_body_count = impl_->rigid_body_count;
    output.triangle_mesh_count = impl_->triangle_mesh_count;
    output.allocated_bytes = static_cast<std::size_t>(
        impl_->device_arena.allocation_size + impl_->upload_arena.allocation_size +
        impl_->readback_arena.allocation_size);
    return success();
}

NativeContext World::native_context() const noexcept {
    return impl_ ? impl_->device->native : NativeContext{};
}

} // namespace parallel_mater::vulkan
