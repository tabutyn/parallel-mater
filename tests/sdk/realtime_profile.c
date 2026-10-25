/* SPDX-License-Identifier: MIT
 * Installed-SDK realtime profile execution and observation gate. */
#include <parallel_mater/parallel_mater.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression) do {                                              \
    if (!(expression)) {                                                    \
        fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
        return 1;                                                           \
    }                                                                       \
} while (0)

typedef struct AllocationList {
    PmDeviceMemory values[8];
    uint32_t count;
} AllocationList;

static PmResult create_device(PmPhysicalDevice physical, int enable_profile,
                              PmDevice* device) {
    static const float priority = 1.0F;
    static const char* extension =
        PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME;
    PmDeviceQueueCreateInfo queue = {0U, 1U, &priority};
    PmDeviceCreateInfo info = {0};
    info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1U;
    info.pQueueInfos = &queue;
    info.enabledFeatures.rigidBodies = PM_TRUE;
    info.enabledFeatures.rigidConstraints = PM_TRUE;
    info.enabledFeatures.statisticsQueries = PM_TRUE;
    info.enabledExtensionCount = enable_profile ? 1U : 0U;
    info.ppEnabledExtensionNames = enable_profile ? &extension : NULL;
    return pmCreateDevice(physical, &info, device);
}

static PmDeviceMemory allocate_memory(PmDevice device, PmDeviceSize size,
                                      AllocationList* allocations) {
    PmMemoryAllocateInfo info = {0};
    PmDeviceMemory memory = NULL;
    info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = size;
    info.memoryTypeIndex = 1U;
    if (allocations->count == 8U ||
        pmAllocateMemory(device, &info, &memory) != PM_SUCCESS)
        return NULL;
    allocations->values[allocations->count++] = memory;
    return memory;
}

static PmBuffer create_mapped_buffer(PmDevice device, PmDeviceSize size,
                                     AllocationList* allocations,
                                     void** mapped) {
    PmBufferCreateInfo info = {0};
    PmBuffer buffer = NULL;
    PmDeviceMemory memory = allocate_memory(device, size, allocations);
    if (memory == NULL) return NULL;
    info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT |
        PM_BUFFER_GEOMETRY_INPUT_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;
    if (pmCreateBuffer(device, &info, &buffer) != PM_SUCCESS ||
        pmBindBufferMemory(device, buffer, memory, 0U) != PM_SUCCESS ||
        pmMapMemory(device, memory, 0U, size, mapped) != PM_SUCCESS)
        return NULL;
    return buffer;
}

static int submit_and_wait(PmDevice device, PmQueue queue,
                           PmCommandBuffer command, PmFence fence) {
    PmSubmitInfo submit = {0};
    PmResult result;
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &command;
    CHECK(pmEndCommandBuffer(command) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    result = pmWaitForFences(device, 1U, &fence, PM_TRUE,
                             PM_TIMEOUT_INFINITE);
    if (result != PM_SUCCESS)
        fprintf(stderr, "submission completion returned %d\n", (int)result);
    CHECK(result == PM_SUCCESS);
    return 0;
}

static int write_observation(const char* path, int validation,
                             const char* candidate_id,
                             const char* candidate_sha256,
                             const char* candidate_revision,
                             const PmRealtimeStepStatistics* statistics,
                             const PmRigidState* state) {
    static const char* phases[PM_REALTIME_WORK_PHASE_COUNT] = {
        "boundary", "predict", "index", "candidates", "domain",
        "coupling", "commit"
    };
    FILE* output = fopen(path, "wb");
    uint32_t index;
    uint64_t spatial_updates = statistics->total.spatialIndexBuilds +
        statistics->total.spatialIndexRefits;
    if (output == NULL) return 1;
    fprintf(output,
        "{\n"
        "  \"schema\": \"parallel-mater-realtime-observation/v1\",\n"
        "  \"source\": \"installed-public-api\",\n"
        "  \"driver_private_state_used\": false,\n"
        "  \"validation_enabled\": %s,\n"
        "  \"profile\": {\"name\": \"realtime-single-step\", "
        "\"version\": {\"major\": 1, \"minor\": 0}},\n"
        "  \"candidate\": {\"manifest_role\": \"qualification-input\", "
        "\"execution_identity_claimed\": false, \"id\": \"%s\", "
        "\"manifest_sha256\": \"%s\", "
        "\"source_revision\": \"%s\", "
        "\"runtime_variant\": \"cpu-runtime\"},\n"
        "  \"case_id\": \"f8-kernel-entrypoint-execution/cpu\",\n"
        "  \"public_api_calls\": [\n"
        "    \"pmEnumerateExecutionProfiles\", \"pmCreateDevice\", "
        "\"pmCreateSimulation\", \"pmCreatePhysicsPipeline\",\n"
        "    \"pmCreateQueryPool\", \"pmBeginCommandBuffer\", "
        "\"pmCmdBuildGeometry\", \"pmCmdWriteRigidBodies\",\n"
        "    \"pmCmdResetQueryPool\", \"pmCmdBeginStatisticsQuery\", "
        "\"pmCmdBindPhysicsPipeline\", \"pmCmdStepSimulation\",\n"
        "    \"pmCmdCopyRigidStates\", \"pmCmdEndStatisticsQuery\", "
        "\"pmEndCommandBuffer\", \"pmQueueSubmit\",\n"
        "    \"pmWaitForFences\", \"pmGetSimulationStatus\", "
        "\"pmGetQueryPoolResults\"\n"
        "  ],\n"
        "  \"steps\": [{\"timestep_seconds\": %.9g, \"substeps\": 1}],\n"
        "  \"simulation_status\": \"PM_SUCCESS\",\n"
        "  \"statistics\": {\"step_count\": %llu, "
        "\"substeps_executed\": %llu, \"execution_result\": "
        "\"PM_SUCCESS\", \"telemetry_schema_version\": %u, "
        "\"profile_version\": %u, \"budget_version\": %u},\n"
        "  \"copied_public_state\": {\"body_count\": 1, "
        "\"position\": [%.9g, %.9g, %.9g], "
        "\"linear_velocity\": [%.9g, %.9g, %.9g]},\n"
        "  \"measurements\": {},\n"
        "  \"tick_work\": [{\n"
        "    \"unit\": \"per-tick\", \"status\": \"success\",\n"
        "    \"dispatches\": %llu, \"phase_dispatches\": {",
        validation ? "true" : "false", candidate_id, candidate_sha256,
        candidate_revision, (double)(1.0F / 60.0F),
        (unsigned long long)statistics->step.stepCount,
        (unsigned long long)statistics->step.substepsExecuted,
        statistics->schemaVersion, statistics->profileVersion,
        statistics->budgetVersion,
        (double)state->position[0], (double)state->position[1],
        (double)state->position[2], (double)state->linearVelocity[0],
        (double)state->linearVelocity[1], (double)state->linearVelocity[2],
        (unsigned long long)statistics->total.dispatches);
    for (index = 0U; index < PM_REALTIME_WORK_PHASE_COUNT; ++index) {
        fprintf(output, "%s\"%s\": %llu", index == 0U ? "" : ", ",
                phases[index],
                (unsigned long long)statistics->phases[index].dispatches);
    }
    fprintf(output,
        "},\n"
        "    \"workgroups\": %llu, \"barriers\": %llu, "
        "\"queue_submissions\": %llu,\n"
        "    \"intra_tick_host_readbacks\": %llu, "
        "\"bytes_moved\": %llu,\n"
        "    \"spatial_index_updates\": %llu, "
        "\"spatial_index_publications\": %llu, "
        "\"time_advances\": %llu,\n"
        "    \"active_iteration_budgets\": "
        "[\"rigid_contact_and_joint_sweeps\"],\n"
        "    \"coupling_edges\": %llu, \"iterations\": {"
        "\"rigid_contact_and_joint_sweeps\": %llu, "
        "\"coupling_sweeps_total\": %llu}\n"
        "  }]\n"
        "}\n",
        (unsigned long long)statistics->total.workgroups,
        (unsigned long long)statistics->total.barriers,
        (unsigned long long)statistics->total.queueSubmissions,
        (unsigned long long)statistics->total.intraTickHostReadbacks,
        (unsigned long long)statistics->total.bytesMoved,
        (unsigned long long)spatial_updates,
        (unsigned long long)statistics->total.spatialIndexPublications,
        (unsigned long long)statistics->timeAdvances,
        (unsigned long long)statistics->couplingEdgeCount,
        (unsigned long long)statistics->domainIterations[0],
        (unsigned long long)statistics->couplingSweeps);
    return fclose(output) == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    const int validation = argc >= 2 && strcmp(argv[1], "validation") == 0;
    const char* observation_path = argc >= 3 ? argv[2] : NULL;
    const char* candidate_id = argc >= 4 ? argv[3] : NULL;
    const char* candidate_sha256 = argc >= 5 ? argv[4] : NULL;
    const char* candidate_revision = argc >= 6 ? argv[5] : NULL;
    PmInstanceCreateInfo instance_info = {0};
    PmInstance instance = NULL;
    PmPhysicalDevice physical = NULL;
    PmDevice device = NULL;
    PmQueue queue = NULL;
    PmSimulation simulation = NULL;
    PmGeometry geometry = NULL;
    PmPhysicsPipeline pipeline = NULL;
    PmQueryPool query_pool = NULL;
    PmCommandPool command_pool = NULL;
    PmCommandBuffer command = NULL;
    PmFence fence = NULL;
    PmBuffer source = NULL;
    PmBuffer output = NULL;
    AllocationList allocations = {{0}, 0U};
    void* source_map = NULL;
    void* output_map = NULL;
    uint32_t count = 1U;
    uint32_t index;
    CHECK(argc == 6 && observation_path && candidate_id &&
          candidate_sha256 && candidate_revision);

    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.enableValidation = validation ? PM_TRUE : PM_FALSE;
    CHECK(pmCreateInstance(&instance_info, NULL, &instance) == PM_SUCCESS);
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);
    CHECK(count == 1U && physical != NULL);

    count = 0U;
    CHECK(pmEnumerateExecutionProfiles(physical, &count, NULL) == PM_SUCCESS);
    CHECK(count == 1U);
    {
        PmExecutionProfileProperties property = {0};
        CHECK(pmEnumerateExecutionProfiles(
                  physical, &count, &property) == PM_SUCCESS);
        CHECK(strcmp(property.profileName,
                     PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP) == 0);
        CHECK(property.versionMajor == 1U && property.versionMinor == 0U);
    }

    {
        PmDevice legacy = NULL;
        PmSimulationProfileCreateInfo profile = {0};
        PmSimulationCreateInfo simulation_info = {0};
        profile.sType = PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO;
        profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
        profile.versionMajor = 1U;
        simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
        simulation_info.pNext = &profile;
        simulation_info.rigidBodyCapacity = 1U;
        CHECK(create_device(physical, 0, &legacy) == PM_SUCCESS);
        CHECK(pmCreateSimulation(legacy, &simulation_info, &simulation) ==
              PM_ERROR_EXTENSION_NOT_PRESENT);
        pmDestroyDevice(legacy);
        CHECK(create_device(physical, 1, &device) == PM_SUCCESS);
        profile.pProfileName = "unknown-profile";
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_ERROR_FEATURE_NOT_PRESENT);
        profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
        profile.versionMinor = 1U;
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_ERROR_INCOMPATIBLE_VERSION);
        profile.versionMinor = 0U;
        profile.pNext = &profile;
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_ERROR_INVALID_ARGUMENT);
        profile.pNext = NULL;
        simulation_info.couplingCapacity = 1U;
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_ERROR_FEATURE_NOT_PRESENT);
    }
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    CHECK(queue != NULL);

    source = create_mapped_buffer(device, 512U, &allocations, &source_map);
    output = create_mapped_buffer(device, 512U, &allocations, &output_map);
    CHECK(source != NULL && output != NULL);
    {
        static const float vertices[8][3] = {
            {-0.5F,-0.5F,-0.5F},{0.5F,-0.5F,-0.5F},
            {0.5F,0.5F,-0.5F},{-0.5F,0.5F,-0.5F},
            {-0.5F,-0.5F,0.5F},{0.5F,-0.5F,0.5F},
            {0.5F,0.5F,0.5F},{-0.5F,0.5F,0.5F}};
        static const uint32_t indices[36] = {
            0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
            3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
        PmGeometryCreateInfo geometry_info = {0};
        PmMemoryRequirements geometry_memory = {0};
        memcpy(source_map, vertices, sizeof(vertices));
        memcpy((unsigned char*)source_map + sizeof(vertices),
               indices, sizeof(indices));
        geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
        geometry_info.vertices.range.buffer = source;
        geometry_info.vertices.range.size = sizeof(vertices);
        geometry_info.vertices.format = PM_FORMAT_R32G32B32_SFLOAT;
        geometry_info.vertices.count = 8U;
        geometry_info.vertices.stride = 12U;
        geometry_info.indices.range.buffer = source;
        geometry_info.indices.range.offset = sizeof(vertices);
        geometry_info.indices.range.size = sizeof(indices);
        geometry_info.indices.format = PM_FORMAT_R32_UINT;
        geometry_info.indices.count = 36U;
        geometry_info.indices.stride = 4U;
        CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
        pmGetGeometryMemoryRequirements(device, geometry, &geometry_memory);
        CHECK(pmBindGeometryMemory(device, geometry,
              allocate_memory(device, geometry_memory.size, &allocations), 0U) ==
              PM_SUCCESS);
    }
    {
        PmSimulationProfileCreateInfo profile = {0};
        PmSimulationCreateInfo simulation_info = {0};
        PmSimulationMemoryRequirements requirements = {0};
        PmSimulationMemoryBinding binding = {0};
        profile.sType = PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO;
        profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
        profile.versionMajor = 1U;
        simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
        simulation_info.pNext = &profile;
        simulation_info.rigidBodyCapacity = 1U;
        simulation_info.contactCapacity = 8U;
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_SUCCESS);
        pmGetSimulationMemoryRequirements(device, simulation, &requirements);
        binding.persistentMemory = allocate_memory(
            device, requirements.persistent.size, &allocations);
        binding.scratchMemory = allocate_memory(
            device, requirements.scratch.size, &allocations);
        CHECK(binding.persistentMemory && binding.scratchMemory);
        CHECK(pmBindSimulationMemory(device, simulation, &binding) == PM_SUCCESS);
    }
    {
        PmPhysicsPipelineCreateInfo info = {0};
        info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
        CHECK(pmCreatePhysicsPipeline(device, &info, &pipeline) == PM_SUCCESS);
    }
    {
        PmQueryPoolCreateInfo info = {0};
        info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = PM_QUERY_REALTIME_STEP_STATISTICS;
        info.queryCount = 1U;
        CHECK(pmCreateQueryPool(device, &info, &query_pool) == PM_SUCCESS);
    }
    {
        PmCommandPoolCreateInfo info = {0};
        info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
        CHECK(pmCreateCommandPool(device, &info, &command_pool) == PM_SUCCESS);
        CHECK(pmAllocateCommandBuffers(device, command_pool, 1U, &command) ==
              PM_SUCCESS);
    }
    CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);

    {
        PmCommandBufferBeginInfo begin = {0};
        PmRigidBodyInfo body = {0};
        begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        body.active = PM_TRUE;
        body.motionType = PM_MOTION_DYNAMIC;
        body.geometry = geometry;
        body.state.position[1] = 2.0F;
        body.state.orientation[3] = 1.0F;
        body.mass = 1.0F;
        body.inertiaDiagonal[0] = body.inertiaDiagonal[1] =
            body.inertiaDiagonal[2] = 1.0F / 6.0F;
        body.maximumLinearSpeed = body.maximumAngularSpeed = 100.0F;
        body.collisionMargin = 0.005F;
        CHECK(pmBeginCommandBuffer(command, &begin) == PM_SUCCESS);
        CHECK(pmCmdBuildGeometry(command, geometry) == PM_SUCCESS);
        CHECK(pmCmdWriteRigidBodies(command, simulation, 0U, 1U, &body) ==
              PM_SUCCESS);
        CHECK(submit_and_wait(device, queue, command, fence) == 0);
        CHECK(pmResetCommandBuffer(command) == PM_SUCCESS);
        CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    }
    {
        PmCommandBufferBeginInfo begin = {0};
        PmStepInfo step = {0};
        PmBufferRange output_range = {output, 0U, sizeof(PmRigidState)};
        PmRealtimeStepStatistics statistics = {0};
        PmResult execution = PM_ERROR_INVALID_STATE;
        const PmRigidState* state;
        begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        step.timestep = 1.0F / 60.0F;
        step.substeps = 1U;
        step.gravity[1] = -9.81F;
        CHECK(pmBeginCommandBuffer(command, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(command, query_pool, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(command, query_pool, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(command, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(command, simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyRigidStates(command, simulation, 0U, 1U,
                                   &output_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(command, query_pool, 0U) == PM_SUCCESS);
        CHECK(submit_and_wait(device, queue, command, fence) == 0);
        CHECK(pmGetSimulationStatus(device, simulation, &execution) == PM_SUCCESS);
        CHECK(execution == PM_SUCCESS);
        CHECK(pmGetQueryPoolResults(device, query_pool, 0U, 1U,
              sizeof(statistics), &statistics, sizeof(statistics),
              PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        state = (const PmRigidState*)output_map;
        CHECK(statistics.step.stepCount == 1U);
        CHECK(statistics.step.substepsExecuted == 1U);
        CHECK(statistics.step.executionResult == PM_SUCCESS);
        CHECK(statistics.schemaVersion == 1U &&
              statistics.profileVersion == 1U &&
              statistics.budgetVersion == 1U);
        CHECK(statistics.phaseCount == PM_REALTIME_WORK_PHASE_COUNT);
        CHECK(statistics.tick == 1U);
        CHECK(statistics.total.dispatches > 0U);
        CHECK(statistics.total.queueSubmissions == 1U);
        CHECK(statistics.total.intraTickHostReadbacks == 0U);
        CHECK(statistics.total.spatialIndexBuilds +
              statistics.total.spatialIndexRefits == 1U);
        CHECK(statistics.total.spatialIndexPublications == 1U);
        CHECK(statistics.timeAdvances == 1U);
        CHECK(statistics.domainIterations[0] == 8U);
        CHECK(statistics.couplingEdgeCount == 0U &&
              statistics.couplingSweeps == 0U);
        CHECK(isfinite(state->position[1]) && state->position[1] < 2.0F);
        CHECK(write_observation(observation_path, validation, candidate_id,
              candidate_sha256, candidate_revision, &statistics, state) == 0);
    }

    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, command_pool, 1U, &command);
    pmDestroyCommandPool(device, command_pool);
    pmDestroyQueryPool(device, query_pool);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyGeometry(device, geometry);
    pmDestroyBuffer(device, output);
    pmDestroyBuffer(device, source);
    for (index = 0U; index < allocations.count; ++index)
        pmFreeMemory(device, allocations.values[index]);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    puts(validation
        ? "PASS: installed validated realtime public observation"
        : "PASS: installed realtime public observation");
    return 0;
}
