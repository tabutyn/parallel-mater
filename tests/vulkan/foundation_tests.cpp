// SPDX-License-Identifier: MIT
#include <parallel_mater/vulkan.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::vulkan;

namespace {

[[noreturn]] void fail(const char *message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void require(bool condition, const char *message) {
    if (!condition) fail(message);
}

void require(Status status, const char *message) {
    if (!status) {
        std::cerr << message << ": "
                  << (status.message ? status.message : "unknown")
                  << " (VkResult " << status.vulkan_result << ")\n";
        std::exit(1);
    }
}

bool near(float left, float right, float tolerance = 2.0e-4F) {
    return std::abs(left - right) <= tolerance;
}

struct BorrowedContext {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    std::uint32_t family{};
    VkDebugUtilsMessengerEXT messenger{VK_NULL_HANDLE};
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger{};
    std::atomic<std::uint32_t> validation_errors{};

    ~BorrowedContext() {
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (messenger != VK_NULL_HANDLE && destroy_messenger)
            destroy_messenger(instance, messenger, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }
};

VKAPI_ATTR VkBool32 VKAPI_CALL validation_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *user_data) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        auto *errors = static_cast<std::atomic<std::uint32_t> *>(user_data);
        ++(*errors);
        std::cerr << "Vulkan validation: "
                  << (data && data->pMessage ? data->pMessage : "unknown")
                  << '\n';
    }
    return VK_FALSE;
}

bool make_borrowed(BorrowedContext &context) {
    std::uint32_t extension_count = 0U;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count,
                                           extensions.data());
    bool debug_utils = false;
    for (const auto &extension : extensions) {
        if (std::string_view(extension.extensionName) ==
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
            debug_utils = true;
    }
    if (!debug_utils) return false;
    const char *instance_extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "ParallelMater Vulkan tests";
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    instance_info.enabledExtensionCount = 1U;
    instance_info.ppEnabledExtensionNames = instance_extensions;
    if (vkCreateInstance(&instance_info, nullptr, &context.instance) != VK_SUCCESS)
        return false;
    auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(context.instance,
                              "vkCreateDebugUtilsMessengerEXT"));
    context.destroy_messenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(context.instance,
                                  "vkDestroyDebugUtilsMessengerEXT"));
    if (!create_messenger || !context.destroy_messenger) return false;
    VkDebugUtilsMessengerCreateInfoEXT messenger_info{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_info.pfnUserCallback = validation_callback;
    messenger_info.pUserData = &context.validation_errors;
    if (create_messenger(context.instance, &messenger_info, nullptr,
                         &context.messenger) != VK_SUCCESS)
        return false;
    std::uint32_t physical_count = 0U;
    if (vkEnumeratePhysicalDevices(context.instance, &physical_count, nullptr) !=
            VK_SUCCESS ||
        physical_count == 0U)
        return false;
    std::vector<VkPhysicalDevice> devices(physical_count);
    if (vkEnumeratePhysicalDevices(context.instance, &physical_count,
                                   devices.data()) != VK_SUCCESS)
        return false;
    for (VkPhysicalDevice physical : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_2) continue;
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        VkPhysicalDeviceFeatures2 features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &timeline;
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (!timeline.timelineSemaphore) continue;
        std::uint32_t family_count = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count,
                                                 families.data());
        for (std::uint32_t family = 0U; family < family_count; ++family) {
            if (families[family].queueCount == 0U ||
                !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
                continue;
            const float priority = 1.0F;
            VkDeviceQueueCreateInfo queue_info{
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue_info.queueFamilyIndex = family;
            queue_info.queueCount = 1U;
            queue_info.pQueuePriorities = &priority;
            timeline.timelineSemaphore = VK_TRUE;
            VkDeviceCreateInfo device_info{
                VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            device_info.pNext = &timeline;
            device_info.queueCreateInfoCount = 1U;
            device_info.pQueueCreateInfos = &queue_info;
            VkDevice logical = VK_NULL_HANDLE;
            if (vkCreateDevice(physical, &device_info, nullptr, &logical) !=
                VK_SUCCESS)
                continue;
            context.physical = physical;
            context.device = logical;
            context.family = family;
            vkGetDeviceQueue(logical, family, 0U, &context.queue);
            return true;
        }
    }
    return false;
}

std::uint32_t host_visible_memory_type(VkPhysicalDevice physical,
                                       std::uint32_t allowed,
                                       bool &coherent) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    std::uint32_t fallback = UINT32_MAX;
    for (std::uint32_t index = 0U; index < properties.memoryTypeCount; ++index) {
        if (!(allowed & (1U << index))) continue;
        const auto flags = properties.memoryTypes[index].propertyFlags;
        if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
            coherent = true;
            return index;
        }
        fallback = index;
    }
    coherent = false;
    return fallback;
}

RigidBodyState copy_device_state(
    BorrowedContext &native, BufferSpan<const RigidBodyState> source,
    std::uint32_t index) {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = sizeof(RigidBodyState);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    require(vkCreateBuffer(native.device, &buffer_info, nullptr, &buffer) ==
                VK_SUCCESS,
            "create state staging buffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(native.device, buffer, &requirements);
    bool coherent = false;
    const std::uint32_t memory_type = host_visible_memory_type(
        native.physical, requirements.memoryTypeBits, coherent);
    require(memory_type != UINT32_MAX, "find state staging memory");
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    require(vkAllocateMemory(native.device, &allocation, nullptr, &memory) ==
                VK_SUCCESS,
            "allocate state staging memory");
    require(vkBindBufferMemory(native.device, buffer, memory, 0U) == VK_SUCCESS,
            "bind state staging memory");

    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = native.family;
    VkCommandPool pool = VK_NULL_HANDLE;
    require(vkCreateCommandPool(native.device, &pool_info, nullptr, &pool) ==
                VK_SUCCESS,
            "create state copy command pool");
    VkCommandBufferAllocateInfo command_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1U;
    VkCommandBuffer command = VK_NULL_HANDLE;
    require(vkAllocateCommandBuffers(native.device, &command_info, &command) ==
                VK_SUCCESS,
            "allocate state copy command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS,
            "begin state copy command");
    VkBufferCopy copy{source.byte_offset + index * sizeof(RigidBodyState), 0U,
                      sizeof(RigidBodyState)};
    vkCmdCopyBuffer(command, source.buffer, buffer, 1U, &copy);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0U, 0U, nullptr, 1U,
                         &barrier, 0U, nullptr);
    require(vkEndCommandBuffer(command) == VK_SUCCESS,
            "end state copy command");
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    require(vkCreateFence(native.device, &fence_info, nullptr, &fence) ==
                VK_SUCCESS,
            "create state copy fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command;
    require(vkQueueSubmit(native.queue, 1U, &submit, fence) == VK_SUCCESS,
            "submit state copy");
    require(vkWaitForFences(native.device, 1U, &fence, VK_TRUE, UINT64_MAX) ==
                VK_SUCCESS,
            "wait state copy");
    void *mapped = nullptr;
    require(vkMapMemory(native.device, memory, 0U, VK_WHOLE_SIZE, 0U, &mapped) ==
                VK_SUCCESS,
            "map state staging memory");
    if (!coherent) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = memory;
        range.size = VK_WHOLE_SIZE;
        require(vkInvalidateMappedMemoryRanges(native.device, 1U, &range) ==
                    VK_SUCCESS,
                "invalidate state staging memory");
    }
    const RigidBodyState result = *static_cast<RigidBodyState *>(mapped);
    vkUnmapMemory(native.device, memory);
    vkDestroyFence(native.device, fence, nullptr);
    vkDestroyCommandPool(native.device, pool, nullptr);
    vkDestroyBuffer(native.device, buffer, nullptr);
    vkFreeMemory(native.device, memory, nullptr);
    return result;
}

void gpu_wait(VkDevice device, VkQueue queue, NativeCompletion completion) {
    VkTimelineSemaphoreSubmitInfo timeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = 1U;
    timeline.pWaitSemaphoreValues = &completion.value;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline;
    submit.waitSemaphoreCount = 1U;
    submit.pWaitSemaphores = &completion.semaphore;
    submit.pWaitDstStageMask = &stage;
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    require(vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS,
            "create GPU-wait fence");
    require(vkQueueSubmit(queue, 1U, &submit, fence) == VK_SUCCESS,
            "submit GPU-side timeline wait");
    require(vkWaitForFences(device, 1U, &fence, VK_TRUE, UINT64_MAX) ==
                VK_SUCCESS,
            "wait GPU-side timeline fence");
    vkDestroyFence(device, fence, nullptr);
}

TriangleMeshId add_test_mesh(World &world) {
    const std::array<Vec3, 4U> vertices{{
        {-1.0F, -1.0F, 0.0F}, {1.0F, -1.0F, 0.0F},
        {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
    }};
    const std::array<std::uint32_t, 12U> indices{
        0U, 1U, 2U, 0U, 3U, 1U, 1U, 3U, 2U, 2U, 3U, 0U,
    };
    TriangleMeshId mesh{};
    require(world.add_triangle_mesh({vertices.data(), vertices.size()},
                                    {indices.data(), indices.size()}, mesh),
            "add triangle mesh");
    return mesh;
}

void exercise_owned() {
    WorldOptions options{};
    options.rigid_body_capacity = 2U;
    options.triangle_mesh_capacity = 1U;
    World world;
    Status status = World::create(options, world);
    if (status.code == StatusCode::not_supported) std::exit(77);
    require(status, "create owned Vulkan world");
    require(world.native_context().device != VK_NULL_HANDLE,
            "owned native context");
    TriangleMeshId mesh = add_test_mesh(world);
    RigidBodyOptions body_options{};
    body_options.mesh = mesh;
    body_options.motion = MotionType::static_body;
    RigidBodyId body{};
    require(world.add_rigid_body(body_options, body), "add owned rigid body");
    require(world.step({}), "step owned world");
}

void exercise_borrowed(BorrowedContext &native) {
    WorldOptions options{};
    options.rigid_body_capacity = 3U;
    options.triangle_mesh_capacity = 1U;
    World world;
    require(World::create(options,
                          {native.instance, native.physical, native.device,
                           native.queue, native.family, true},
                          world),
            "create borrowed Vulkan world");
    const TriangleMeshId mesh = add_test_mesh(world);

    RigidBodyOptions invalid_body_options{};
    invalid_body_options.mesh = mesh;
    invalid_body_options.mass = 0.0F;
    RigidBodyId invalid_body{};
    require(world.add_rigid_body(invalid_body_options, invalid_body).code ==
                StatusCode::invalid_argument,
            "invalid body rejected transactionally");

    RigidBodyOptions dynamic_options{};
    dynamic_options.mesh = mesh;
    dynamic_options.mass = 2.0F;
    dynamic_options.linear_damping = 0.0F;
    dynamic_options.angular_damping = 0.0F;
    RigidBodyId dynamic{};
    require(world.add_rigid_body(dynamic_options, dynamic), "add dynamic body");
    require(world.apply_impulse(dynamic, {2.0F, 0.0F, 0.0F}, {}),
            "apply impulse");
    require(world.apply_force(dynamic, {4.0F, 0.0F, 0.0F}, {}),
            "apply force");
    const std::array<RigidBodyId, 1U> accelerated{dynamic};
    require(world.apply_central_acceleration(
                {accelerated.data(), accelerated.size()}, {0.0F, 0.0F, 3.0F}),
            "apply acceleration");

    RigidBodyOptions static_options{};
    static_options.mesh = mesh;
    static_options.motion = MotionType::static_body;
    static_options.initial_state.position = {4.0F, 5.0F, 6.0F};
    static_options.initial_state.linear_velocity = {10.0F, 10.0F, 10.0F};
    RigidBodyId fixed{};
    require(world.add_rigid_body(static_options, fixed), "add static body");

    RigidBodyOptions kinematic_options{};
    kinematic_options.mesh = mesh;
    kinematic_options.motion = MotionType::kinematic;
    RigidBodyId kinematic{};
    require(world.add_rigid_body(kinematic_options, kinematic),
            "add kinematic body");
    RigidBodyId overflow{};
    require(world.add_rigid_body(dynamic_options, overflow).code ==
                StatusCode::capacity_exceeded,
            "rigid capacity enforced");
    RigidBodyState target{};
    target.position = {0.0F, 2.0F, 0.0F};
    require(world.set_kinematic_target(kinematic, target),
            "set kinematic target");

    FrameToken token;
    StepOptions step{};
    step.timestep = 0.1F;
    step.substeps = 1U;
    step.gravity = {0.0F, -10.0F, 0.0F};
    require(world.step_async(step, token), "submit async step");
    const NativeCompletion completion = token.native_completion();
    require(completion.semaphore != VK_NULL_HANDLE && completion.value != 0U,
            "native timeline completion");
    gpu_wait(native.device, native.queue, completion);
    require(token.ready(), "timeline token ready after GPU wait");

    RigidBodyState state{};
    require(world.read_rigid_body_state(dynamic, state), "read dynamic state");
    require(near(state.linear_velocity.x, 1.2F) &&
                near(state.linear_velocity.y, -1.0F) &&
                near(state.linear_velocity.z, 0.3F),
            "dynamic velocity formula");
    require(near(state.position.x, 0.12F) && near(state.position.y, -0.1F) &&
                near(state.position.z, 0.03F),
            "dynamic position formula");
    require(world.read_rigid_body_state(fixed, state), "read static state");
    require(near(state.position.x, 4.0F) && near(state.position.y, 5.0F) &&
                near(state.position.z, 6.0F) && near(state.linear_velocity.x, 0.0F),
            "static body formula");
    require(world.read_rigid_body_state(kinematic, state),
            "read kinematic state");
    require(near(state.position.y, 2.0F) && near(state.linear_velocity.y, 20.0F),
            "kinematic body formula");
    RigidBodyDeviceView first_view{};
    require(world.rigid_body_view(first_view), "acquire first rigid view");
    const RigidBodyState previous =
        copy_device_state(native, first_view.previous_states, 0U);
    require(near(previous.position.x, 0.0F) &&
                near(previous.position.y, 0.0F) &&
                near(previous.linear_velocity.x, 0.0F) &&
                near(previous.linear_velocity.y, 0.0F),
            "previous state preserves pre-step dynamic state");

    const RigidBodyState first_result = [&] {
        RigidBodyState value{};
        require(world.read_rigid_body_state(dynamic, value),
                "capture deterministic result");
        return value;
    }();
    require(world.set_rigid_body_state(dynamic, {}), "reset dynamic state");
    require(world.apply_impulse(dynamic, {2.0F, 0.0F, 0.0F}, {}),
            "reapply impulse");
    require(world.apply_force(dynamic, {4.0F, 0.0F, 0.0F}, {}),
            "reapply force");
    require(world.apply_central_acceleration(
                {accelerated.data(), accelerated.size()}, {0.0F, 0.0F, 3.0F}),
            "reapply acceleration");
    require(world.step_async(step, token), "reuse async token");
    require(token.wait(), "wait reused token");
    require(world.read_rigid_body_state(dynamic, state),
            "read deterministic replay");
    require(near(state.position.x, first_result.position.x) &&
                near(state.position.y, first_result.position.y) &&
                near(state.position.z, first_result.position.z) &&
                near(state.linear_velocity.x, first_result.linear_velocity.x),
            "deterministic replay");

    StepOptions no_loads = step;
    no_loads.gravity = {};
    require(world.step_async(no_loads, token), "step consumed inputs");
    require(token.wait(), "wait consumed-input step");
    require(world.read_rigid_body_state(dynamic, state),
            "read consumed-input step");
    require(near(state.linear_velocity.x, 1.2F) &&
                near(state.linear_velocity.z, 0.3F),
            "force impulse and acceleration consumed once");

    RigidBodyDeviceView view{};
    require(world.rigid_body_view(view), "acquire rigid device view");
    require(view.ids.buffer != VK_NULL_HANDLE && view.ids.size == 3U &&
                view.states.byte_offset != view.previous_states.byte_offset,
            "rigid device view contract");
    WorldStatistics statistics{};
    require(world.collect_statistics(statistics), "collect statistics");
    require(statistics.frame_index == 3U && statistics.rigid_body_count == 3U &&
                statistics.triangle_mesh_count == 1U &&
                statistics.allocated_bytes != 0U,
            "statistics values");

    require(world.remove_rigid_body(fixed), "remove rigid body");
    require(world.read_rigid_body_state(fixed, state).code ==
                StatusCode::invalid_handle,
            "stale handle rejected");
    require(world.remove_triangle_mesh(mesh).code == StatusCode::invalid_argument,
            "referenced mesh removal rejected");

    RigidBodyOptions clamped_options{};
    clamped_options.mesh = mesh;
    clamped_options.initial_state.linear_velocity = {10.0F, 0.0F, 0.0F};
    clamped_options.linear_damping = 1.0F;
    clamped_options.maximum_linear_speed = 2.0F;
    RigidBodyId clamped{};
    require(world.add_rigid_body(clamped_options, clamped),
            "add damped speed-limited body");
    require(world.step(no_loads), "step damped speed-limited body");
    require(world.read_rigid_body_state(clamped, state),
            "read damped speed-limited body");
    require(near(state.linear_velocity.x, 2.0F) && near(state.position.x, 0.2F),
            "damping then speed clamp formula");

    StepOptions timed_step = no_loads;
    timed_step.collect_kernel_timings = true;
    const Status timed_status = world.step(timed_step);
    if (timed_status) {
        WorldStepTimings timings{};
        require(world.collect_step_timings(timings), "collect GPU timings");
        require(timings.available &&
                    timings.rigid_integration.launch_count == 1U &&
                    timings.rigid_input_clear.launch_count == 1U,
                "GPU timing values available");
    } else {
        require(timed_status.code == StatusCode::not_supported,
                "timing failure maps to not_supported");
    }

    StepOptions contacts{};
    contacts.collect_rigid_contacts = true;
    require(world.step_async(contacts, token).code == StatusCode::not_supported,
            "contact collection rejected");
}

} // namespace

int main() {
    World uninitialized;
    RigidBodyState ignored{};
    require(uninitialized.read_rigid_body_state({}, ignored).code ==
                StatusCode::invalid_argument,
            "uninitialized world rejected");
    WorldOptions unsupported{};
    unsupported.rigid_sleeping = true;
    World rejected;
    require(World::create(unsupported, rejected).code == StatusCode::not_supported,
            "sleeping rejected");

    exercise_owned();
    BorrowedContext native;
    if (!make_borrowed(native)) return 77;
    WorldOptions borrowed_options{};
    World rejected_borrowed;
    require(World::create(
                borrowed_options,
                {native.instance, native.physical, native.device, native.queue,
                 native.family, false},
                rejected_borrowed)
                .code == StatusCode::not_supported,
            "borrowed context without timeline declaration rejected");
    require(rejected_borrowed.native_context().device == VK_NULL_HANDLE,
            "failed creation leaves output unchanged");
    exercise_borrowed(native);
    require(native.validation_errors.load() == 0U,
            "Vulkan validation emitted errors");

    VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore semaphore = VK_NULL_HANDLE;
    require(vkCreateSemaphore(native.device, &semaphore_info, nullptr,
                              &semaphore) == VK_SUCCESS,
            "borrowed device remains alive after World destruction");
    vkDestroySemaphore(native.device, semaphore, nullptr);
    return 0;
}
