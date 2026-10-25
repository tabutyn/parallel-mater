// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(value) do { if (!(value)) {                                  \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value);             \
    std::abort();                                                           \
} } while (false)

int main(int argc, char **argv) {
    const bool production = argc == 2 &&
        std::strcmp(argv[1], "--production") == 0;
    CHECK(argc == 1 || production);
    constexpr std::uint32_t body_count = 8U;
    constexpr std::uint32_t constraint_count = 8U;
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t physical_count = 1U;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(
        instance, &physical_count, &physical) == PM_SUCCESS);

    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    const char *extension = PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME;
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    pmGetPhysicalDeviceFeatures(physical, &device_info.enabledFeatures);
    device_info.enabledExtensionCount = 1U;
    device_info.ppEnabledExtensionNames = &extension;
    PmDevice device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);

    std::vector<PmDeviceMemory> memory;
    const auto allocate = [&](PmDeviceSize size) {
        PmMemoryAllocateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.memoryTypeIndex = 1U;
        info.allocationSize = size;
        PmDeviceMemory result{};
        CHECK(pmAllocateMemory(device, &info, &result) == PM_SUCCESS);
        memory.push_back(result);
        return result;
    };
    const auto buffer = [&](PmDeviceSize size, void **mapped) {
        const auto allocation = allocate(size);
        PmBufferCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = PM_BUFFER_TRANSFER_SRC_BIT |
            PM_BUFFER_TRANSFER_DST_BIT | PM_BUFFER_GEOMETRY_INPUT_BIT |
            PM_BUFFER_PARTICLE_INPUT_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;
        PmBuffer result{};
        CHECK(pmCreateBuffer(device, &info, &result) == PM_SUCCESS);
        CHECK(pmBindBufferMemory(device, result, allocation, 0U) == PM_SUCCESS);
        CHECK(pmMapMemory(device, allocation, 0U, size, mapped) == PM_SUCCESS);
        return result;
    };

    void *source_map{}, *output_map{}, *fluid_output_map{}, *cloth_output_map{};
    const auto source = buffer(512U, &source_map);
    const auto output = buffer(
        body_count * sizeof(PmRigidState), &output_map);
    const auto fluid_output = buffer(1024U, &fluid_output_map);
    const auto cloth_output = buffer(1024U, &cloth_output_map);
    const float vertices[8][3] = {
        {-0.5F,-0.5F,-0.5F},{0.5F,-0.5F,-0.5F},
        {0.5F,0.5F,-0.5F},{-0.5F,0.5F,-0.5F},
        {-0.5F,-0.5F,0.5F},{0.5F,-0.5F,0.5F},
        {0.5F,0.5F,0.5F},{-0.5F,0.5F,0.5F}};
    const std::uint32_t indices[36] = {
        0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
        3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
    std::memcpy(source_map, vertices, sizeof vertices);
    std::memcpy(static_cast<std::byte *>(source_map) + sizeof vertices,
                indices, sizeof indices);
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {{source, 0U, sizeof vertices},
                              PM_FORMAT_R32G32B32_SFLOAT, 8U, 12U};
    geometry_info.indices = {{source, sizeof vertices, sizeof indices},
                             PM_FORMAT_R32_UINT, 36U, 4U};
    PmGeometry geometry{};
    CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements geometry_memory{};
    pmGetGeometryMemoryRequirements(device, geometry, &geometry_memory);
    CHECK(pmBindGeometryMemory(
        device, geometry, allocate(geometry_memory.size), 0U) == PM_SUCCESS);

    PmSimulationProfileCreateInfo profile{};
    profile.sType = PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO;
    profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
    profile.versionMajor = 1U;
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.pNext = &profile;
    simulation_info.rigidBodyCapacity = body_count;
    simulation_info.constraintCapacity = constraint_count;
    simulation_info.contactCapacity = 2048U;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        allocate(requirements.persistent.size), 0U,
        allocate(requirements.scratch.size), 0U};
    CHECK(pmBindSimulationMemory(device, simulation, &binding) == PM_SUCCESS);

    const std::array fluid_capacities{
        PmFluidCapacity{4U, 4U}, PmFluidCapacity{7U, 7U}};
    const PmClothCapacity cloth_capacity{3U, 3U, 3U};
    const PmRopeCapacity rope_capacity{2U};
    const PmSoftBodyCapacity soft_capacity{1U, 1U, 3U, 3U};
    const PmSmokeCapacity smoke_capacity{4U, 8U};
    PmSimulationCreateInfo mixed_info = simulation_info;
    mixed_info.rigidBodyCapacity = 1U;
    mixed_info.constraintCapacity = 0U;
    mixed_info.fluidCount = device_info.enabledFeatures.particleLiquids
        ? static_cast<std::uint32_t>(fluid_capacities.size()) : 0U;
    mixed_info.pFluidCapacities = fluid_capacities.data();
    mixed_info.clothCount = device_info.enabledFeatures.cloth;
    mixed_info.pClothCapacities = &cloth_capacity;
    mixed_info.ropeCount = device_info.enabledFeatures.rope;
    mixed_info.pRopeCapacities = &rope_capacity;
    mixed_info.softBodyCount = device_info.enabledFeatures.softBodies;
    mixed_info.pSoftBodyCapacities = &soft_capacity;
    mixed_info.smokeCount = device_info.enabledFeatures.smoke;
    mixed_info.pSmokeCapacities = &smoke_capacity;
    PmSimulation mixed_simulation{};
    CHECK(pmCreateSimulation(device, &mixed_info, &mixed_simulation) ==
          PM_SUCCESS);
    PmSimulationMemoryRequirements mixed_requirements{};
    pmGetSimulationMemoryRequirements(
        device, mixed_simulation, &mixed_requirements);
    CHECK(mixed_requirements.persistent.size != 0U);
    CHECK(mixed_requirements.scratch.size != 0U);
    const PmSimulationMemoryBinding mixed_binding{
        allocate(mixed_requirements.persistent.size), 0U,
        allocate(mixed_requirements.scratch.size), 0U};
    CHECK(pmBindSimulationMemory(
        device, mixed_simulation, &mixed_binding) == PM_SUCCESS);

    PmRigidSolverState rigid_solver{};
    rigid_solver.rigidIterations = 8U;
    rigid_solver.enableSleeping = PM_FALSE;
    PmLiquidMaterialState liquid_material{};
    liquid_material.restDensity = 1000.0F;
    liquid_material.particleRadius = 0.025F;
    liquid_material.supportRadius = 0.1F;
    liquid_material.restParticleVolume = 0.000125F;
    liquid_material.pressureStiffness = 50.0F;
    liquid_material.velocityDamping = 0.2F;
    liquid_material.maximumSpeed = 2.9F;
    liquid_material.densityIterations = 6U;
    const std::array liquid_materials{liquid_material, liquid_material};
    PmClothMaterialState cloth_material{};
    cloth_material.vertexMass = 1.0F;
    cloth_material.thickness = 0.01F;
    cloth_material.velocityDamping = 0.2F;
    cloth_material.contactFriction = 0.2F;
    cloth_material.solverIterations = 8U;
    cloth_material.fracturePersistenceSubsteps = 1U;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid_solver;
    pipeline_info.fluidMaterialCount =
        static_cast<std::uint32_t>(liquid_materials.size());
    pipeline_info.pFluidMaterials = liquid_materials.data();
    pipeline_info.clothMaterialCount = mixed_info.clothCount;
    pipeline_info.pClothMaterials = &cloth_material;
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
          PM_SUCCESS);
    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_REALTIME_STEP_STATISTICS;
    query_info.queryCount = 1U;
    PmQueryPool query{};
    CHECK(pmCreateQueryPool(device, &query_info, &query) == PM_SUCCESS);
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1U, &commands) == PM_SUCCESS);
    PmFence fence{};
    CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &commands;

    std::uint32_t execution_index{};
    const auto execute = [&](PmResult expected) {
        ++execution_index;
        CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
        const auto waited = pmWaitForFences(
            device, 1U, &fence, PM_TRUE, PM_TIMEOUT_INFINITE);
        if (waited != expected)
            std::fprintf(stderr, "execution=%u waited=%d expected=%d\n",
                         execution_index, int(waited), int(expected));
        CHECK(waited == expected);
        CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    };
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBuildGeometry(commands, geometry) == PM_SUCCESS);
    execute(PM_SUCCESS);
    PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, 0U};
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    execute(PM_SUCCESS);

    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(
        commands, mixed_simulation, &step) == PM_SUCCESS);
    execute(PM_SUCCESS);
    if (mixed_info.fluidCount != 0U) {
        auto *particle = reinterpret_cast<PmParticleState *>(
            static_cast<std::byte *>(source_map) + 256U);
        *particle = {};
        particle->position[1] = 1.0F;
        PmBufferView particle_input{
            {source, 256U, sizeof(PmParticleState)},
            PM_FORMAT_PARTICLE_STATE, 1U, sizeof(PmParticleState)};
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdSetFluidParticles(
            commands, mixed_simulation, 0U, &particle_input) == PM_SUCCESS);
        PmBufferView empty_input{
            {source, 384U, 0U}, PM_FORMAT_PARTICLE_STATE, 0U,
            sizeof(PmParticleState)};
        CHECK(pmCmdSetFluidParticles(
            commands, mixed_simulation, 1U, &empty_input) == PM_SUCCESS);
        execute(PM_SUCCESS);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        const PmBufferRange fluid_range{fluid_output, 0U, 512U};
        CHECK(pmCmdCopyFluidParticles(
            commands, mixed_simulation, 0U, &fluid_range) == PM_SUCCESS);
        const PmBufferRange empty_range{fluid_output, 512U, 512U};
        CHECK(pmCmdCopyFluidParticles(
            commands, mixed_simulation, 1U, &empty_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        const auto *fluid_header =
            static_cast<const PmDomainOutputHeader *>(fluid_output_map);
        const auto *fluid_particles =
            reinterpret_cast<const PmFluidParticleOutput *>(
                fluid_header + 1U);
        CHECK(fluid_header->activeCount == 1U);
        CHECK(fluid_particles[0].position[1] < 1.0F);
        CHECK(std::isfinite(fluid_particles[0].position[1]));
        const auto *empty_header = reinterpret_cast<
            const PmDomainOutputHeader *>(
                static_cast<const std::byte *>(fluid_output_map) + 512U);
        CHECK(empty_header->activeCount == 0U);

        auto *packed = reinterpret_cast<PmParticleState *>(
            static_cast<std::byte *>(source_map) + 256U);
        for (std::uint32_t index = 0U; index < 7U; ++index) {
            packed[index] = {};
            packed[index].position[0] =
                static_cast<float>(index % 4U) * 0.04F;
            packed[index].position[1] =
                1.0F + static_cast<float>(index / 4U) * 0.04F;
        }
        const PmBufferView first_population{
            {source, 256U, 4U * sizeof(PmParticleState)},
            PM_FORMAT_PARTICLE_STATE, 4U, sizeof(PmParticleState)};
        const PmBufferView second_population{
            {source, 256U + 4U * sizeof(PmParticleState),
             3U * sizeof(PmParticleState)},
            PM_FORMAT_PARTICLE_STATE, 3U, sizeof(PmParticleState)};
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdSetFluidParticles(commands, mixed_simulation, 0U,
                                     &first_population) == PM_SUCCESS);
        CHECK(pmCmdSetFluidParticles(commands, mixed_simulation, 1U,
                                     &second_population) == PM_SUCCESS);
        execute(PM_SUCCESS);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        execute(PM_SUCCESS);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyFluidParticles(
            commands, mixed_simulation, 0U, &fluid_range) == PM_SUCCESS);
        CHECK(pmCmdCopyFluidParticles(
            commands, mixed_simulation, 1U, &empty_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        execute(PM_SUCCESS);
        PmRealtimeStepStatistics liquid_statistics{};
        CHECK(pmGetQueryPoolResults(
            device, query, 0U, 1U, sizeof liquid_statistics,
            &liquid_statistics, sizeof liquid_statistics,
            PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        CHECK(liquid_statistics.timeAdvances == 1U);
        CHECK(liquid_statistics.domainIterations[1] == 6U);
        CHECK(liquid_statistics.total.spatialIndexBuilds == 1U);
        CHECK(liquid_statistics.total.spatialIndexPublications == 1U);
        CHECK(liquid_statistics.total.intraTickHostReadbacks == 0U);
        fluid_header =
            static_cast<const PmDomainOutputHeader *>(fluid_output_map);
        fluid_particles = reinterpret_cast<const PmFluidParticleOutput *>(
            fluid_header + 1U);
        CHECK(fluid_header->activeCount == 4U);
        empty_header = reinterpret_cast<const PmDomainOutputHeader *>(
            static_cast<const std::byte *>(fluid_output_map) + 512U);
        CHECK(empty_header->activeCount == 3U);
        for (std::uint32_t index = 0U; index < 4U; ++index) {
            CHECK(std::isfinite(fluid_particles[index].position[0]));
            CHECK(std::isfinite(fluid_particles[index].position[1]));
            CHECK(std::abs(fluid_particles[index].position[0]) < 2.0F);
            CHECK(std::abs(fluid_particles[index].position[1]) < 2.0F);
        }
        if (!production) {
            std::array<std::byte, 1024U> liquid_before_failure{};
            std::memcpy(liquid_before_failure.data(), fluid_output_map,
                        liquid_before_failure.size());
            CHECK(setenv(
                "PM_TEST_REALTIME_LIMIT_MINUS_ONE", "workgroups", 1) == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
            CHECK(pmCmdStepSimulation(
                commands, mixed_simulation, &step) == PM_SUCCESS);
            execute(PM_ERROR_CAPACITY_EXCEEDED);
            CHECK(unsetenv("PM_TEST_REALTIME_LIMIT_MINUS_ONE") == 0);

            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdCopyFluidParticles(
                commands, mixed_simulation, 0U, &fluid_range) == PM_SUCCESS);
            CHECK(pmCmdCopyFluidParticles(
                commands, mixed_simulation, 1U, &empty_range) == PM_SUCCESS);
            execute(PM_SUCCESS);
            CHECK(std::memcmp(liquid_before_failure.data(), fluid_output_map,
                              liquid_before_failure.size()) == 0);
        }

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        execute(PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        execute(PM_SUCCESS);

    }
    if (mixed_info.clothCount != 0U) {
        const std::uint32_t cloth_indices[3]{0U, 1U, 2U};
        const float pinned_masses[3]{0.0F, 0.0F, 1.0F};
        const PmFloat3 hanging_vertices[3]{
            {-0.1F, 1.0F, 0.0F}, {0.1F, 1.0F, 0.0F},
            {0.0F, 0.9F, 0.1F}};
        const PmClothInfo hanging_cloth{
            PM_TRUE, 0U, 3U, hanging_vertices, 3U, cloth_indices,
            pinned_masses};
        const PmBufferRange cloth_range{cloth_output, 0U, 1024U};

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        execute(PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteCloths(
            commands, mixed_simulation, 0U, 1U, &hanging_cloth) == PM_SUCCESS);
        execute(PM_SUCCESS);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyClothState(
            commands, mixed_simulation, 0U, &cloth_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        execute(PM_SUCCESS);
        PmRealtimeStepStatistics cloth_statistics{};
        CHECK(pmGetQueryPoolResults(
            device, query, 0U, 1U, sizeof cloth_statistics,
            &cloth_statistics, sizeof cloth_statistics,
            PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        CHECK(cloth_statistics.timeAdvances == 1U);
        CHECK(cloth_statistics.domainIterations[2] == 8U);
        CHECK(cloth_statistics.total.spatialIndexBuilds == 1U);
        CHECK(cloth_statistics.total.spatialIndexPublications == 1U);
        CHECK(cloth_statistics.total.intraTickHostReadbacks == 0U);
        const auto *cloth_header =
            static_cast<const PmDomainOutputHeader *>(cloth_output_map);
        const auto *cloth_vertices =
            reinterpret_cast<const PmClothVertexState *>(cloth_header + 1U);
        CHECK(cloth_header->activeCount == 3U);
        for (std::uint32_t vertex = 0U; vertex < 2U; ++vertex) {
            CHECK(cloth_vertices[vertex].position.x ==
                  hanging_vertices[vertex].x);
            CHECK(cloth_vertices[vertex].position.y ==
                  hanging_vertices[vertex].y);
            CHECK(cloth_vertices[vertex].position.z ==
                  hanging_vertices[vertex].z);
            CHECK(cloth_vertices[vertex].velocity.x == 0.0F);
            CHECK(cloth_vertices[vertex].velocity.y == 0.0F);
            CHECK(cloth_vertices[vertex].velocity.z == 0.0F);
        }
        CHECK(std::isfinite(cloth_vertices[2].position.y));
        CHECK(std::isfinite(cloth_vertices[2].velocity.y));

        if (!production) {
            std::array<std::byte, 1024U> cloth_before_failure{};
            std::memcpy(cloth_before_failure.data(), cloth_output_map,
                        cloth_before_failure.size());
            CHECK(setenv(
                "PM_TEST_REALTIME_LIMIT_MINUS_ONE", "workgroups", 1) == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
            CHECK(pmCmdStepSimulation(
                commands, mixed_simulation, &step) == PM_SUCCESS);
            execute(PM_ERROR_CAPACITY_EXCEEDED);
            CHECK(unsetenv("PM_TEST_REALTIME_LIMIT_MINUS_ONE") == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdCopyClothState(
                commands, mixed_simulation, 0U, &cloth_range) == PM_SUCCESS);
            execute(PM_SUCCESS);
            CHECK(std::memcmp(cloth_before_failure.data(), cloth_output_map,
                              cloth_before_failure.size()) == 0);
        }

        PmRigidBodyInfo floor{};
        floor.active = PM_TRUE;
        floor.geometry = geometry;
        floor.motionType = PM_MOTION_STATIC;
        floor.mass = 1.0F;
        floor.inertiaDiagonal[0] = floor.inertiaDiagonal[1] =
            floor.inertiaDiagonal[2] = 1.0F / 6.0F;
        floor.state.orientation[3] = 1.0F;
        floor.maximumLinearSpeed = floor.maximumAngularSpeed = 100.0F;
        floor.collisionMargin = 0.005F;
        const PmFloat3 contact_vertices[3]{
            {-0.1F, 0.505F, 0.0F}, {0.1F, 0.505F, 0.0F},
            {0.0F, 0.505F, 0.1F}};
        const float dynamic_masses[3]{1.0F, 1.0F, 1.0F};
        const PmClothInfo contact_cloth{
            PM_TRUE, 0U, 3U, contact_vertices, 3U, cloth_indices,
            dynamic_masses};
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteRigidBodies(
            commands, mixed_simulation, 0U, 1U, &floor) == PM_SUCCESS);
        execute(PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteCloths(
            commands, mixed_simulation, 0U, 1U, &contact_cloth) == PM_SUCCESS);
        execute(PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyClothState(
            commands, mixed_simulation, 0U, &cloth_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        cloth_header =
            static_cast<const PmDomainOutputHeader *>(cloth_output_map);
        cloth_vertices = reinterpret_cast<const PmClothVertexState *>(
            cloth_header + 1U);
        CHECK(cloth_header->activeCount == 3U);
        for (std::uint32_t vertex = 0U; vertex < 3U; ++vertex) {
            CHECK(std::isfinite(cloth_vertices[vertex].position.y));
            CHECK(std::isfinite(cloth_vertices[vertex].velocity.y));
        }

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        execute(PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteCloths(
            commands, mixed_simulation, 0U, 1U, &hanging_cloth) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyClothState(
            commands, mixed_simulation, 0U, &cloth_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        cloth_header =
            static_cast<const PmDomainOutputHeader *>(cloth_output_map);
        CHECK(cloth_header->activeCount == 3U);
    }
    pmDestroySimulation(device, mixed_simulation);

    std::array<PmRigidBodyInfo, body_count> bodies{};
    std::array<PmRigidConstraintInfo, constraint_count> constraints{};
    for (std::uint32_t index = 0U; index < body_count; ++index) {
        auto &body = bodies[index];
        body.active = PM_TRUE;
        body.geometry = geometry;
        body.motionType = PM_MOTION_DYNAMIC;
        body.mass = 1.0F;
        body.inertiaDiagonal[0] = body.inertiaDiagonal[1] =
            body.inertiaDiagonal[2] = 1.0F / 6.0F;
        body.state.position[0] = static_cast<float>(index % 4U) * 2.0F;
        body.state.position[1] =
            2.0F + static_cast<float>(index / 4U) * 2.0F;
        body.state.orientation[3] = 1.0F;
        body.maximumLinearSpeed = body.maximumAngularSpeed = 100.0F;
        body.collisionMargin = 0.005F;
        auto &joint = constraints[index];
        joint.active = PM_TRUE;
        joint.type = PM_CONSTRAINT_POINT;
        joint.bodyA = index;
        joint.bodyB = (index + 1U) % body_count;
        const float next_x = static_cast<float>(joint.bodyB % 4U) * 2.0F;
        const float next_y =
            2.0F + static_cast<float>(joint.bodyB / 4U) * 2.0F;
        joint.frameA.position[0] =
            (next_x - body.state.position[0]) * 0.5F;
        joint.frameA.position[1] =
            (next_y - body.state.position[1]) * 0.5F;
        joint.frameB.position[0] = -joint.frameA.position[0];
        joint.frameB.position[1] = -joint.frameA.position[1];
        joint.frameA.orientation[3] = 1.0F;
        joint.frameB.orientation[3] = 1.0F;
        joint.solverIterations = 8U;
    }
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdWriteRigidBodies(
        commands, simulation, 0U, body_count, bodies.data()) == PM_SUCCESS);
    CHECK(pmCmdWriteRigidConstraints(commands, simulation, 0U,
        constraint_count, constraints.data()) == PM_SUCCESS);
    execute(PM_SUCCESS);

    PmStepInfo invalid_step{1.0F / 60.0F, 2U,
                            {0.0F, -9.81F, 0.0F}, 0U};
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &invalid_step) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);

    PmBufferRange output_range{
        output, 0U, body_count * sizeof(PmRigidState)};
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
    CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(
        commands, simulation, 0U, body_count, &output_range) == PM_SUCCESS);
    CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
    execute(PM_SUCCESS);
    PmRealtimeStepStatistics statistics{};
    CHECK(pmGetQueryPoolResults(device, query, 0U, 1U,
        sizeof statistics, &statistics, sizeof statistics,
        PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
    CHECK(statistics.step.executionResult == PM_SUCCESS);
    CHECK(statistics.step.stepCount == 1U);
    CHECK(statistics.step.substepsExecuted == 1U);
    CHECK(statistics.schemaVersion == 1U);
    CHECK(statistics.profileVersion == 1U);
    CHECK(statistics.budgetVersion == 1U);
    CHECK(statistics.phaseCount == PM_REALTIME_WORK_PHASE_COUNT);
    CHECK(statistics.tick == 2U);
    CHECK(statistics.total.dispatches != 0U);
    CHECK(statistics.total.workgroups != 0U);
    if (!production) {
        CHECK(statistics.total.dispatches == 20U);
        CHECK(statistics.total.workgroups == 1421U);
    }
    CHECK(statistics.total.numericalIterations == 8U);
    CHECK(statistics.total.spatialIndexBuilds == 1U);
    CHECK(statistics.total.spatialIndexPublications == 1U);
    CHECK(statistics.total.queueSubmissions == 1U);
    CHECK(statistics.total.intraTickHostReadbacks == 0U);
    CHECK(statistics.timeAdvances == 1U);
    CHECK(statistics.domainIterations[0] == 8U);
    CHECK(statistics.couplingEdgeCount == 0U);
    CHECK(statistics.couplingSweeps == 0U);
    std::uint64_t phase_dispatches{};
    for (std::uint32_t index = 0U;
         index < PM_REALTIME_WORK_PHASE_COUNT; ++index)
        phase_dispatches += statistics.phases[index].dispatches;
    CHECK(phase_dispatches == statistics.total.dispatches);
    const auto first = *static_cast<const PmRigidState *>(output_map);
    CHECK(first.position[1] < 2.0F);

    if (!production) {
    CHECK(setenv("PM_TEST_REALTIME_LIMIT_MINUS_ONE", "workgroups", 1) == 0);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    execute(PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(unsetenv("PM_TEST_REALTIME_LIMIT_MINUS_ONE") == 0);

    constexpr std::array fields{
        "dispatches", "workgroups", "bytes_moved", "index_builds",
        "index_publications", "numerical_iterations", "barriers",
        "submissions"};
    bool input_written = false;
    for (const char *field : fields) {
        CHECK(setenv("PM_TEST_REALTIME_ZERO_LIMIT", field, 1) == 0);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        if (!input_written) {
            PmRigidInput input{};
            input.bodyIndex = 0U;
            input.linearImpulse[0] = 1.0F;
            CHECK(pmCmdApplyRigidInputs(commands, simulation, 1U, &input) ==
                  PM_SUCCESS);
            input_written = true;
        }
        CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
        execute(PM_ERROR_CAPACITY_EXCEEDED);
    }
    CHECK(unsetenv("PM_TEST_REALTIME_ZERO_LIMIT") == 0);
    CHECK(setenv("PM_TEST_REALTIME_ZERO_DOMAIN_DISPATCHES", "1", 1) == 0);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    execute(PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(unsetenv("PM_TEST_REALTIME_ZERO_DOMAIN_DISPATCHES") == 0);
    CHECK(setenv("PM_TEST_REALTIME_ZERO_COMMIT_SUBMISSIONS", "1", 1) == 0);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    execute(PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(unsetenv("PM_TEST_REALTIME_ZERO_COMMIT_SUBMISSIONS") == 0);

    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(
        commands, simulation, 0U, body_count, &output_range) == PM_SUCCESS);
    execute(PM_SUCCESS);
    CHECK(std::memcmp(output_map, &first, sizeof first) == 0);
    }

    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(
        commands, simulation, 0U, body_count, &output_range) == PM_SUCCESS);
    execute(PM_SUCCESS);
    const auto second = *static_cast<const PmRigidState *>(output_map);
    CHECK(second.position[1] < first.position[1]);
    float total_x_velocity{};
    const auto *states = static_cast<const PmRigidState *>(output_map);
    for (std::uint32_t index = 0U; index < body_count; ++index)
        total_x_velocity += states[index].linearVelocity[0];
    if (!production) CHECK(total_x_velocity > 0.5F);

    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, pool, 1U, &commands);
    pmDestroyCommandPool(device, pool);
    pmDestroyQueryPool(device, query);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyGeometry(device, geometry);
    pmDestroyBuffer(device, output);
    pmDestroyBuffer(device, fluid_output);
    pmDestroyBuffer(device, cloth_output);
    pmDestroyBuffer(device, source);
    for (auto allocation : memory) pmFreeMemory(device, allocation);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts(production
        ? "PASS: production realtime resources are fixed and atomic"
        : "PASS: native CPU realtime schedule is fixed and atomic");
}
