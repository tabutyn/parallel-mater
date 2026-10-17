// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(expression) do { if (!(expression)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression); return 1; \
} } while (false)

int main() {
    std::uint32_t version = 0U;
    CHECK(pmEnumerateInstanceVersion(&version) == PM_SUCCESS);
    CHECK(version == PM_API_VERSION_0_2);
    CHECK(pmGetInstanceProcAddr(nullptr, "pmCreateInstance") != nullptr);
    CHECK(pmGetInstanceProcAddr(nullptr, "pmQueueSubmit") == nullptr);
    std::uint32_t layer_count = 0U;
    CHECK(pmEnumerateInstanceLayerProperties(&layer_count, nullptr) == PM_SUCCESS);
    CHECK(layer_count == 1U);
    PmLayerProperties layer{};
    CHECK(pmEnumerateInstanceLayerProperties(&layer_count, &layer) == PM_SUCCESS);
    CHECK(std::strcmp(layer.layerName, "PM_LAYER_PARALLEL_MATER_validation") == 0);

    const char *missing_instance_extension = "PM_EXT_missing";
    PmInstanceCreateInfo missing_instance_info{};
    missing_instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    missing_instance_info.apiVersion = PM_API_VERSION_0_2;
    missing_instance_info.enabledExtensionCount = 1U;
    missing_instance_info.ppEnabledExtensionNames = &missing_instance_extension;
    PmInstance missing_instance{};
    CHECK(pmCreateInstance(&missing_instance_info, nullptr, &missing_instance) ==
          PM_ERROR_EXTENSION_NOT_PRESENT);
    CHECK(missing_instance == nullptr);

    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    const char *validation_layer = "PM_LAYER_PARALLEL_MATER_validation";
    instance_info.enabledLayerCount = 1U;
    instance_info.ppEnabledLayerNames = &validation_layer;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);

    std::uint32_t physical_count = 0U;
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, nullptr) == PM_SUCCESS);
    if (physical_count == 0U) {
        pmDestroyInstance(instance);
        return 77;
    }
    PmPhysicalDevice physical{};
    std::uint32_t one = 1U;
    CHECK(pmEnumeratePhysicalDevices(instance, &one, &physical) ==
          (physical_count == 1U ? PM_SUCCESS : PM_INCOMPLETE));

    PmPhysicalDeviceFeatures supported{};
    pmGetPhysicalDeviceFeatures(physical, &supported);
    CHECK(supported.rigidBodies == PM_TRUE);
    CHECK(supported.rigidConstraints == PM_TRUE);
    CHECK(supported.liquidLiquidCoupling == PM_FALSE);
    CHECK(supported.timelineSemaphores == PM_TRUE);
    CHECK(supported.timestampQueries == PM_TRUE);
    std::uint32_t interaction_count{};
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              physical, &interaction_count, nullptr) == PM_SUCCESS);
    CHECK(interaction_count >= 2U);
    std::array<PmInteractionCapability, 1> short_interactions{};
    std::uint32_t short_count = 1U;
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              physical, &short_count, short_interactions.data()) == PM_INCOMPLETE);
    CHECK(short_count == 1U);
    std::vector<PmInteractionCapability> interactions(interaction_count);
    CHECK(pmEnumeratePhysicalDeviceInteractionCapabilities(
              physical, &interaction_count, interactions.data()) == PM_SUCCESS);
    bool found_rigid_contact = false;
    for (const auto &interaction : interactions) {
        PmInteractionCapability queried{};
        CHECK(pmGetPhysicalDeviceInteractionCapability(
                  physical, &interaction.key, &queried) == PM_SUCCESS);
        CHECK(queried.maxGraphEdgesPerSimulation ==
              interaction.maxGraphEdgesPerSimulation);
        found_rigid_contact |=
            interaction.key.firstFamily == PM_INTERACTION_FAMILY_RIGID_BODY &&
            interaction.key.secondFamily == PM_INTERACTION_FAMILY_RIGID_BODY &&
            interaction.key.scope == PM_INTERACTION_SCOPE_SAME_DOMAIN &&
            interaction.key.mode == PM_INTERACTION_MODE_CONTACT;
    }
    CHECK(found_rigid_contact);
    PmInteractionCapabilityKey unsupported_interaction{
        PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SEPARATE_SYSTEM,
        PM_INTERACTION_MODE_PRESSURE,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_EQUAL};
    PmInteractionCapability interaction_result{};
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &unsupported_interaction, &interaction_result) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    auto malformed_interaction = unsupported_interaction;
    malformed_interaction.mode = static_cast<PmInteractionMode>(99);
    CHECK(pmGetPhysicalDeviceInteractionCapability(
              physical, &malformed_interaction, &interaction_result) ==
          PM_ERROR_INVALID_ARGUMENT);
    PmNativeInteropProperties cuda_interop{};
    CHECK(pmGetPhysicalDeviceInteropProperties(
              physical, PM_NATIVE_BACKEND_CUDA, &cuda_interop) == PM_SUCCESS);
    CHECK(cuda_interop.deviceImport == PM_FALSE);
    CHECK(cuda_interop.bufferImport == PM_TRUE);
    CHECK(cuda_interop.timelineImport == PM_FALSE);
    PmNativeInteropProperties unavailable_interop{PM_TRUE, PM_TRUE, PM_TRUE};
    CHECK(pmGetPhysicalDeviceInteropProperties(
              physical, PM_NATIVE_BACKEND_VULKAN, &unavailable_interop) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(unavailable_interop.deviceImport == PM_FALSE);
    CHECK(unavailable_interop.bufferImport == PM_FALSE);
    CHECK(unavailable_interop.timelineImport == PM_FALSE);

    float priority = 1.0F;
    PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.timelineSemaphores = PM_TRUE;
    device_info.enabledFeatures.timestampQueries = PM_TRUE;
    device_info.enabledFeatures.memoryBudget = PM_TRUE;
    device_info.enabledFeatures.rigidBodies = PM_TRUE;
    device_info.enabledFeatures.rigidConstraints = PM_TRUE;
    device_info.enabledFeatures.particleLiquids = PM_TRUE;
    device_info.enabledFeatures.cloth = PM_TRUE;
    std::uint32_t device_extension_count{};
    CHECK(pmEnumerateDeviceExtensionProperties(
              physical, nullptr, &device_extension_count, nullptr) == PM_SUCCESS);
    std::array<PmExtensionProperties, 4> device_extensions{};
    const char *enabled_device_extension = nullptr;
    CHECK(device_extension_count <= device_extensions.size());
    if (device_extension_count != 0U) {
        CHECK(pmEnumerateDeviceExtensionProperties(
                  physical, nullptr, &device_extension_count,
                  device_extensions.data()) == PM_SUCCESS);
        enabled_device_extension = device_extensions[0].extensionName;
        device_info.enabledExtensionCount = 1U;
        device_info.ppEnabledExtensionNames = &enabled_device_extension;
    }
    const char *missing_device_extension = "PM_EXT_missing";
    PmDeviceCreateInfo missing_device_info = device_info;
    missing_device_info.enabledExtensionCount = 1U;
    missing_device_info.ppEnabledExtensionNames = &missing_device_extension;
    PmDevice missing_device{};
    CHECK(pmCreateDevice(physical, &missing_device_info, &missing_device) ==
          PM_ERROR_EXTENSION_NOT_PRESENT);
    CHECK(missing_device == nullptr);
    PmDevice device{};
    const PmResult create_device = pmCreateDevice(physical, &device_info, &device);
    if (create_device != PM_SUCCESS)
        std::fprintf(stderr, "pmCreateDevice returned %d\n",
                     static_cast<int>(create_device));
    CHECK(create_device == PM_SUCCESS);
    CHECK(pmGetDeviceProcAddr(device, "pmQueueSubmit") != nullptr);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    CHECK(queue != nullptr);

    PmMemoryAllocateInfo allocation{};
    allocation.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = 256U;
    allocation.memoryTypeIndex = 1U;
    PmDeviceMemory source_memory{}, destination_memory{};
    CHECK(pmAllocateMemory(device, &allocation, &source_memory) == PM_SUCCESS);
    CHECK(pmAllocateMemory(device, &allocation, &destination_memory) == PM_SUCCESS);

    PmBufferCreateInfo buffer_info{};
    buffer_info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = 256U;
    buffer_info.usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT |
        PM_BUFFER_GEOMETRY_INPUT_BIT | PM_BUFFER_PARTICLE_INPUT_BIT |
        PM_BUFFER_SIMULATION_OUTPUT_BIT;
    PmBuffer source{}, destination{};
    CHECK(pmCreateBuffer(device, &buffer_info, &source) == PM_SUCCESS);
    CHECK(pmCreateBuffer(device, &buffer_info, &destination) == PM_SUCCESS);
    CHECK(pmBindBufferMemory(device, source, source_memory, 0U) == PM_SUCCESS);
    CHECK(pmBindBufferMemory(device, destination, destination_memory, 0U) == PM_SUCCESS);

    void *mapped_source = nullptr;
    void *mapped_destination = nullptr;
    CHECK(pmMapMemory(device, source_memory, 0U, PM_WHOLE_SIZE,
                      &mapped_source) == PM_SUCCESS);
    CHECK(pmMapMemory(device, destination_memory, 0U, PM_WHOLE_SIZE,
                      &mapped_destination) == PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 64U; ++index)
        static_cast<std::uint32_t *>(mapped_source)[index] = index * 17U;
    std::memset(mapped_destination, 0, 256U);

    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1U, &commands) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    PmBufferCopy copy{0U, 0U, 256U};
    CHECK(pmCmdCopyBuffer(commands, source, destination, 1U, &copy) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);

    PmSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    PmSemaphore semaphore{};
    CHECK(pmCreateSemaphore(device, &semaphore_info, &semaphore) == PM_SUCCESS);
    PmSemaphoreSubmitInfo signal{semaphore, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &commands;
    submit.signalCount = 1U;
    submit.pSignals = &signal;
    PmFence fence{};
    CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);

    // Invalid synchronization must reject the entire submit before the copy
    // is launched, leaving the command buffer executable for a valid retry.
    PmSemaphoreSubmitInfo invalid_signal{
        semaphore, 0U, PM_STAGE_ALL_COMMANDS_BIT};
    submit.pSignals = &invalid_signal;
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_ERROR_INVALID_ARGUMENT);
    PmFence signaled_fence{};
    CHECK(pmCreateFence(device, PM_FENCE_SIGNALED_BIT, &signaled_fence) ==
          PM_SUCCESS);
    submit.pSignals = &signal;
    CHECK(pmQueueSubmit(queue, &submit, signaled_fence) ==
          PM_ERROR_INVALID_STATE);
    pmDestroyFence(device, signaled_fence);
    PmSemaphoreSubmitInfo self_wait{
        semaphore, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    submit.waitCount = 1U;
    submit.pWaits = &self_wait;
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_ERROR_INVALID_ARGUMENT);
    submit.waitCount = 0U;
    submit.pWaits = nullptr;

    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmWaitSemaphore(device, semaphore, 1U, PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 64U; ++index)
        CHECK(static_cast<std::uint32_t *>(mapped_destination)[index] == index * 17U);

    PmNativeDeviceInfo native_device{};
    PmNativeBufferInfo native_buffer{};
    CHECK(pmGetNativeDevice(device, &native_device) == PM_SUCCESS);
    CHECK(native_device.backend == PM_NATIVE_BACKEND_CUDA);
    CHECK(native_device.device != 0U);
    CHECK(pmGetNativeBuffer(device, destination, &native_buffer) == PM_SUCCESS);
    CHECK(native_buffer.buffer != 0U && native_buffer.size == 256U);
    CHECK(native_buffer.offset == 0U && native_buffer.nativeUsage == 0U);
    PmNativeBufferInfo misaligned_import = native_buffer;
    misaligned_import.offset = 1U;
    PmBuffer imported{};
    CHECK(pmImportNativeBuffer(device, &misaligned_import, &imported) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(imported == nullptr);
    PmNativeBufferInfo borrowed_import = native_buffer;
    borrowed_import.owningQueueFamilyIndex = PM_QUEUE_FAMILY_EXTERNAL;
    CHECK(pmImportNativeBuffer(device, &borrowed_import, &imported) == PM_SUCCESS);
    PmNativeBufferInfo borrowed_round_trip{};
    CHECK(pmGetNativeBuffer(device, imported, &borrowed_round_trip) == PM_SUCCESS);
    CHECK(borrowed_round_trip.buffer == borrowed_import.buffer);
    CHECK(borrowed_round_trip.offset == borrowed_import.offset);
    CHECK(borrowed_round_trip.owningQueueFamilyIndex ==
          PM_QUEUE_FAMILY_EXTERNAL);
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    PmBufferMemoryBarrier visibility{};
    visibility.range = {source, 0U, 256U};
    visibility.srcQueueFamilyIndex = PM_QUEUE_FAMILY_IGNORED;
    visibility.dstQueueFamilyIndex = PM_QUEUE_FAMILY_IGNORED;
    CHECK(pmCmdPipelineBarrier(commands, 0U, nullptr, 1U, &visibility) ==
          PM_SUCCESS);
    PmBufferMemoryBarrier acquire{};
    acquire.range = {imported, 0U, 256U};
    acquire.srcQueueFamilyIndex = PM_QUEUE_FAMILY_EXTERNAL;
    acquire.dstQueueFamilyIndex = 0U;
    CHECK(pmCmdPipelineBarrier(commands, 0U, nullptr, 1U, &acquire) ==
          PM_SUCCESS);
    CHECK(pmCmdCopyBuffer(commands, source, imported, 1U, &copy) == PM_SUCCESS);
    PmBufferMemoryBarrier release = acquire;
    release.srcQueueFamilyIndex = 0U;
    release.dstQueueFamilyIndex = PM_QUEUE_FAMILY_EXTERNAL;
    CHECK(pmCmdPipelineBarrier(commands, 0U, nullptr, 1U, &release) ==
          PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    submit.signalCount = 0U;
    submit.pSignals = nullptr;
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmGetNativeBuffer(device, imported, &borrowed_round_trip) == PM_SUCCESS);
    CHECK(borrowed_round_trip.owningQueueFamilyIndex ==
          PM_QUEUE_FAMILY_EXTERNAL);
    pmDestroyBuffer(device, imported);
    PmNativeHandle native_semaphore = 1U;
    CHECK(pmGetNativeSemaphore(device, semaphore, &native_semaphore) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(native_semaphore == 0U);
    PmSemaphore imported_semaphore{};
    CHECK(pmImportNativeSemaphore(device, 1U, &imported_semaphore) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(imported_semaphore == nullptr);
    PmMemoryBudget budget{};
    CHECK(pmGetDeviceMemoryBudget(device, &budget) == PM_SUCCESS);
    CHECK(budget.applicationAllocationBytes >= 512U);

    // Exercise the physics path through the command API, including a Slang
    // AVBD step and portable state export.
    const float cube_vertices[8][3] = {
        {-0.5F, -0.5F, -0.5F}, {0.5F, -0.5F, -0.5F},
        {0.5F, 0.5F, -0.5F}, {-0.5F, 0.5F, -0.5F},
        {-0.5F, -0.5F, 0.5F}, {0.5F, -0.5F, 0.5F},
        {0.5F, 0.5F, 0.5F}, {-0.5F, 0.5F, 0.5F}};
    const std::uint32_t cube_indices[36] = {
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};
    std::memcpy(mapped_source, cube_vertices, sizeof cube_vertices);
    std::memcpy(static_cast<std::byte *>(mapped_source) + sizeof cube_vertices,
                cube_indices, sizeof cube_indices);
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {{source, 0U, sizeof cube_vertices},
                              PM_FORMAT_R32G32B32_SFLOAT, 8U, 12U};
    geometry_info.indices = {{source, sizeof cube_vertices, sizeof cube_indices},
                             PM_FORMAT_R32_UINT, 36U, 4U};
    PmGeometry geometry{};
    CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
    PmDeviceMemory geometry_memory{};
    PmMemoryRequirements geometry_requirements{};
    pmGetGeometryMemoryRequirements(device, geometry, &geometry_requirements);
    CHECK(geometry_requirements.size >= sizeof cube_vertices + sizeof cube_indices);
    CHECK(geometry_requirements.alignment == 256U);
    CHECK((geometry_requirements.memoryTypeBits & 0x2U) != 0U);
    allocation.allocationSize = geometry_requirements.size;
    allocation.memoryTypeIndex = 1U;
    CHECK(pmAllocateMemory(device, &allocation, &geometry_memory) == PM_SUCCESS);
    CHECK(pmBindGeometryMemory(device, geometry, geometry_memory, 0U) == PM_SUCCESS);

    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 2U;
    simulation_info.contactCapacity = 32U;
    const PmFluidCapacity fluid_capacity{2U, 16U};
    simulation_info.fluidCount = 1U;
    simulation_info.pFluidCapacities = &fluid_capacity;
    const PmClothCapacity cloth_capacity{3U, 3U, 3U};
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &cloth_capacity;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) == PM_SUCCESS);
    PmDeviceMemory persistent_memory{}, scratch_memory{};
    PmSimulationMemoryRequirements simulation_requirements{};
    pmGetSimulationMemoryRequirements(device, simulation,
                                      &simulation_requirements);
    CHECK(simulation_requirements.persistent.size > 256U);
    CHECK(simulation_requirements.scratch.size != 0U);
    CHECK(simulation_requirements.scratch.alignment != 0U);
    CHECK(simulation_requirements.scratch.size % simulation_requirements.scratch.alignment == 0U);
    allocation.memoryTypeIndex = 1U;
    allocation.allocationSize = simulation_requirements.persistent.size;
    CHECK(pmAllocateMemory(device, &allocation, &persistent_memory) == PM_SUCCESS);
    allocation.allocationSize = simulation_requirements.scratch.size;
    CHECK(pmAllocateMemory(device, &allocation, &scratch_memory) == PM_SUCCESS);
    PmSimulationMemoryBinding simulation_binding{
        persistent_memory, 0U, scratch_memory, 0U};
    const PmResult bind_simulation =
        pmBindSimulationMemory(device, simulation, &simulation_binding);
    if (bind_simulation != PM_SUCCESS)
        std::fprintf(stderr, "pmBindSimulationMemory returned %d\n",
                     static_cast<int>(bind_simulation));
    CHECK(bind_simulation == PM_SUCCESS);

    PmRigidSolverState rigid_solver{10U, PM_FALSE, 0.02F, 0.02F, 0.5F};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid_solver;
    const PmLiquidMaterialState liquid_material{
        .restDensity = 1000.0F,
        .particleRadius = 0.025F,
        .supportRadius = 0.1F,
        .restParticleVolume = 0.000125F,
        .pressureStiffness = 50.0F,
        .kinematicViscosity = 0.02F,
        .normalDamping = 0.0F,
        .velocityDamping = 0.2F,
        .maximumSpeed = 8.0F,
        .maximumPairAcceleration = 0.0F,
        .densityIterations = 4U};
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &liquid_material;
    const PmClothMaterialState cloth_material{
        0.02F, 0.025F, 1.0e-6F, 0.1F, 5.0F, 0.4F,
        8U, 0.0F, 4U, 0.0F, PM_FALSE, 0.0F, 1.0e-7F};
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth_material;
    PmCouplingState unsupported_coupling{};
    unsupported_coupling.type = PM_COUPLING_LIQUID_LIQUID;
    unsupported_coupling.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    unsupported_coupling.materialProfile = PM_INTERACTION_MATERIAL_EQUAL;
    PmPhysicsPipelineCreateInfo unsupported_pipeline_info = pipeline_info;
    unsupported_pipeline_info.couplingCount = 1U;
    unsupported_pipeline_info.pCouplings = &unsupported_coupling;
    PmPhysicsPipeline unsupported_pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &unsupported_pipeline_info,
                                  &unsupported_pipeline) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(unsupported_pipeline == nullptr);
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) == PM_SUCCESS);
    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_STEP_STATISTICS;
    query_info.queryCount = 1U;
    PmQueryPool statistics_query{};
    CHECK(pmCreateQueryPool(device, &query_info, &statistics_query) == PM_SUCCESS);

    PmRigidBodyInfo bodies[2]{};
    for (auto &body : bodies) {
        body.active = PM_TRUE;
        body.geometry = geometry;
        body.state.orientation[3] = 1.0F;
        body.mass = 1.0F;
        body.inertiaDiagonal[0] = body.inertiaDiagonal[1] =
            body.inertiaDiagonal[2] = 1.0F / 6.0F;
        body.friction = 0.5F;
        body.linearDamping = body.angularDamping = 0.05F;
        body.maximumLinearSpeed = body.maximumAngularSpeed = 100.0F;
        body.collisionMargin = 0.005F;
    }
    bodies[0].motionType = PM_MOTION_STATIC;
    bodies[0].state.position[1] = -1.0F;
    bodies[1].motionType = PM_MOTION_DYNAMIC;
    bodies[1].state.position[1] = 2.0F;
    PmStepInfo step{1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F}, 0U};
    PmBufferRange output_range{destination, 0U, 2U * sizeof(PmRigidState)};
    auto *particle_states = reinterpret_cast<PmParticleState *>(
        static_cast<std::byte *>(mapped_destination) + 128U);
    particle_states[0].position[0] = -0.1F;
    particle_states[0].position[1] = 1.0F;
    particle_states[0].temperatureCelsius = 20.0F;
    particle_states[1].position[0] = 0.1F;
    particle_states[1].position[1] = 1.0F;
    particle_states[1].temperatureCelsius = 20.0F;
    PmBufferView particle_input{{destination, 128U,
                                 2U * sizeof(PmParticleState)},
                                PM_FORMAT_PARTICLE_STATE, 2U,
                                sizeof(PmParticleState)};
    PmBufferRange particle_output{
        destination, 128U, sizeof(PmDomainOutputHeader) +
            2U * sizeof(PmFluidParticleOutput)};
    const PmFloat3 cloth_vertices[3]{{-0.5F, 1.0F, 0.0F},
                                     {0.5F, 1.0F, 0.0F},
                                     {0.0F, 2.0F, 0.0F}};
    const std::uint32_t cloth_indices[3]{0U, 1U, 2U};
    const float cloth_inverse_masses[3]{1.0F, 1.0F, 1.0F};
    const PmClothInfo cloth{PM_TRUE, 0U, 3U, cloth_vertices, 3U,
                            cloth_indices, cloth_inverse_masses};

    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBuildGeometry(commands, geometry) == PM_SUCCESS);
    CHECK(pmCmdResetSimulation(commands, simulation) == PM_SUCCESS);
    CHECK(pmCmdWriteRigidBodies(commands, simulation, 0U, 2U, bodies) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdWriteCloths(commands, simulation, 0U, 1U, &cloth) == PM_SUCCESS);
    CHECK(pmCmdSetFluidParticles(commands, simulation, 0U, &particle_input) ==
          PM_SUCCESS);
    PmRigidInput rigid_input{};
    rigid_input.bodyIndex = 1U;
    rigid_input.torque[2] = 0.5F;
    rigid_input.angularImpulse[2] = 0.25F;
    CHECK(pmCmdApplyRigidInputs(commands, simulation, 1U, &rigid_input) ==
          PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, statistics_query, 0U, 1U) == PM_SUCCESS);
    CHECK(pmCmdBeginStatisticsQuery(commands, statistics_query, 0U) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdEndStatisticsQuery(commands, statistics_query, 0U) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(commands, simulation, 0U, 2U, &output_range) ==
          PM_SUCCESS);
    CHECK(pmCmdCopyFluidParticles(commands, simulation, 0U, &particle_output) ==
          PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    submit.signalCount = 0U;
    submit.pSignals = nullptr;
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    const auto *states = static_cast<const PmRigidState *>(mapped_destination);
    CHECK(states[0].position[1] == -1.0F);
    CHECK(states[1].position[1] < 2.0F);
    CHECK(states[1].angularVelocity[2] > 0.0F);
    const auto *particle_header = reinterpret_cast<const PmDomainOutputHeader *>(
        static_cast<const std::byte *>(mapped_destination) + 128U);
    CHECK(particle_header->activeCount == 2U);
    CHECK(particle_header->capacity == 2U);
    const auto *particle_output_states =
        reinterpret_cast<const PmFluidParticleOutput *>(
            static_cast<const std::byte *>(mapped_destination) + 128U +
            sizeof(PmDomainOutputHeader));
    CHECK(particle_output_states[0].position[1] < 1.0F);
    CHECK(particle_output_states[0].stableId == 0U);
    PmResult execution = PM_ERROR_INVALID_STATE;
    CHECK(pmGetSimulationStatus(device, simulation, &execution) == PM_SUCCESS);
    CHECK(execution == PM_SUCCESS);
    PmStepStatistics step_statistics{};
    CHECK(pmGetQueryPoolResults(device, statistics_query, 0U, 1U,
        sizeof(step_statistics), &step_statistics, sizeof(step_statistics),
        PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
    CHECK(step_statistics.stepCount >= 1U);
    CHECK(step_statistics.peakActiveBodies == 2U);

    // The gallery's F key enables contact collection and copies the resulting
    // events through the loader API. Keep this path covered because its count
    // buffers may reside in GPU-owned managed-memory pages on Windows.
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    step.flags = PM_STEP_COLLECT_CONTACT_EVENTS_BIT;
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    const PmBufferRange contact_output{destination, 0U, 256U};
    CHECK(pmCmdCopyContactEvents(commands, simulation, &contact_output) ==
          PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    const auto *contact_header =
        static_cast<const PmDomainOutputHeader *>(mapped_destination);
    CHECK(contact_header->activeCount <= contact_header->capacity);
    CHECK(contact_header->capacity == simulation_info.contactCapacity);
    CHECK(contact_header->revision >= 2U);
    CHECK(pmGetSimulationStatus(device, simulation, &execution) == PM_SUCCESS);
    CHECK(execution == PM_SUCCESS);

    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    const PmBufferRange cloth_output{destination, 0U, 256U};
    CHECK(pmCmdCopyClothState(commands, simulation, 0U, &cloth_output) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    const auto *cloth_header = static_cast<const PmDomainOutputHeader *>(
        mapped_destination);
    CHECK(cloth_header->activeCount == 3U);
    const auto *cloth_states = reinterpret_cast<const PmClothVertexState *>(
        static_cast<const std::byte *>(mapped_destination) +
        sizeof(PmDomainOutputHeader));
    CHECK(cloth_states[0].position.y < 1.0F);

    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroyQueryPool(device, statistics_query);
    pmDestroySimulation(device, simulation);
    pmDestroyGeometry(device, geometry);
    pmFreeMemory(device, scratch_memory);
    pmFreeMemory(device, persistent_memory);
    pmFreeMemory(device, geometry_memory);

    pmDestroyFence(device, fence);
    pmDestroySemaphore(device, semaphore);
    pmFreeCommandBuffers(device, pool, 1U, &commands);
    pmDestroyCommandPool(device, pool);
    pmUnmapMemory(device, destination_memory);
    pmUnmapMemory(device, source_memory);
    pmDestroyBuffer(device, destination);
    pmDestroyBuffer(device, source);
    pmFreeMemory(device, destination_memory);
    pmFreeMemory(device, source_memory);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: command API CUDA rigid, fluid, and cloth simulation");
}
