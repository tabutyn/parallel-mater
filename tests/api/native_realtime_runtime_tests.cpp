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

    void *source_map{}, *output_map{}, *fluid_output_map{}, *cloth_output_map{},
        *soft_output_map{}, *rope_output_map{}, *smoke_output_map{};
    const auto source = buffer(512U, &source_map);
    const auto output = buffer(
        body_count * sizeof(PmRigidState), &output_map);
    const auto fluid_output = buffer(1024U, &fluid_output_map);
    const auto cloth_output = buffer(1024U, &cloth_output_map);
    const auto soft_output = buffer(1024U, &soft_output_map);
    const auto rope_output = buffer(1024U, &rope_output_map);
    const auto smoke_output = buffer(4096U, &smoke_output_map);
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
    const PmRopeCapacity rope_capacity{6U};
    const PmSoftBodyCapacity soft_capacity{8U, 16U, 8U, 18U};
    const PmSmokeCapacity smoke_capacity{32U, 16U * 8U * 16U};
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
    PmRopeMaterialState rope_material{};
    rope_material.nodeSpacing = 0.1F;
    rope_material.radius = 0.05F;
    rope_material.mass = 0.4F;
    rope_material.stretchCompliance = 0.0F;
    rope_material.velocityDamping = 0.2F;
    rope_material.maximumSubstepTimestep = 1.0F / 240.0F;
    rope_material.friction = 0.2F;
    rope_material.maximumSpeed = 2.9F;
    rope_material.solverIterations = 3U;
    rope_material.selfCollision = PM_FALSE;
    PmSoftBodyMaterialState soft_material{};
    soft_material.nodeMass = 1.0F;
    soft_material.nodeRadius = 0.01F;
    soft_material.velocityDamping = 0.2F;
    soft_material.contactFriction = 0.2F;
    soft_material.shapeMatchingStiffness = 0.5F;
    soft_material.maximumProjectionFraction = 0.2F;
    soft_material.constraintVelocityResponse = 1.0F;
    soft_material.maximumSpeed = 2.9F;
    soft_material.solverIterations = 8U;
    PmSmokeMaterialState particle_smoke_material{};
    particle_smoke_material.particleRadius = 0.05F;
    particle_smoke_material.buoyancy = 0.2F;
    particle_smoke_material.response = 1.0F;
    particle_smoke_material.restNumberDensity = 12.0F;
    particle_smoke_material.pressureStiffness = 2.0F;
    particle_smoke_material.viscosity = 0.1F;
    particle_smoke_material.maximumSpeed = 2.0F;
    PmSmokeMaterialState grid_smoke_material = particle_smoke_material;
    grid_smoke_material.gridResolution = 16U;
    grid_smoke_material.gridVerticalResolution = 8U;
    grid_smoke_material.gridPressureIterations = 12U;
    grid_smoke_material.gridKinematicViscosity = 0.01F;
    grid_smoke_material.gridLesCoefficient = 0.1F;
    grid_smoke_material.gridPressureTolerance = 0.001F;
    const std::array smoke_materials{
        particle_smoke_material, grid_smoke_material};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid_solver;
    pipeline_info.fluidMaterialCount =
        static_cast<std::uint32_t>(liquid_materials.size());
    pipeline_info.pFluidMaterials = liquid_materials.data();
    pipeline_info.clothMaterialCount = mixed_info.clothCount;
    pipeline_info.pClothMaterials = &cloth_material;
    pipeline_info.ropeMaterialCount = mixed_info.ropeCount;
    pipeline_info.pRopeMaterials = &rope_material;
    pipeline_info.softBodyMaterialCount = mixed_info.softBodyCount;
    pipeline_info.pSoftBodyMaterials = &soft_material;
    pipeline_info.smokeMaterialCount = mixed_info.smokeCount == 0U
        ? 0U : static_cast<std::uint32_t>(smoke_materials.size());
    pipeline_info.pSmokeMaterials = smoke_materials.data();
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
    if (mixed_info.softBodyCount != 0U) {
        const PmFloat3 nodes[5]{
            {0.0F, 1.2F, 0.0F}, {-0.05F, 1.0F, 0.0F},
            {0.05F, 1.0F, 0.0F}, {0.0F, 1.1F, 0.05F},
            {0.0F, 1.1F, -0.05F}};
        const float inverse_masses[5]{0.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        std::array<PmSoftBodyBond, 10> bonds{};
        std::uint32_t bond_count{};
        for (std::uint32_t first = 0U; first < 5U; ++first)
            for (std::uint32_t second = first + 1U; second < 5U; ++second) {
                const float x = nodes[first].x - nodes[second].x;
                const float y = nodes[first].y - nodes[second].y;
                const float z = nodes[first].z - nodes[second].z;
                bonds[bond_count++] = {
                    first, second, std::sqrt(x * x + y * y + z * z)};
            }
        const PmFloat3 surface_vertices[4]{
            nodes[1], nodes[2], nodes[3], nodes[4]};
        const std::uint32_t surface_indices[12]{
            0U, 2U, 1U, 0U, 1U, 3U,
            0U, 3U, 2U, 1U, 2U, 3U};
        std::array<PmSoftBodySurfaceBinding, 4> bindings{};
        for (std::uint32_t vertex = 0U; vertex < bindings.size(); ++vertex) {
            bindings[vertex].nodes[0] = vertex + 1U;
            bindings[vertex].weights[0] = 1.0F;
        }
        const PmSoftBodyInfo body{PM_TRUE, 0U, 5U, nodes,
            static_cast<std::uint32_t>(bonds.size()), bonds.data(),
            inverse_masses, 4U, surface_vertices, 12U, surface_indices,
            bindings.data()};
        const PmBufferRange soft_range{soft_output, 0U, 1024U};

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        execute(PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteSoftBodies(
            commands, mixed_simulation, 0U, 1U, &body) == PM_SUCCESS);
        execute(PM_SUCCESS);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopySoftBodyState(
            commands, mixed_simulation, 0U, &soft_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        execute(PM_SUCCESS);
        PmRealtimeStepStatistics soft_statistics{};
        CHECK(pmGetQueryPoolResults(
            device, query, 0U, 1U, sizeof soft_statistics,
            &soft_statistics, sizeof soft_statistics,
            PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        CHECK(soft_statistics.timeAdvances == 1U);
        CHECK(soft_statistics.domainIterations[3] == 8U);
        CHECK(soft_statistics.total.spatialIndexBuilds == 1U);
        CHECK(soft_statistics.total.spatialIndexPublications == 1U);
        CHECK(soft_statistics.total.intraTickHostReadbacks == 0U);
        const auto *soft_header =
            static_cast<const PmDomainOutputHeader *>(soft_output_map);
        const auto *soft_nodes =
            reinterpret_cast<const PmSoftBodyNodeState *>(soft_header + 1U);
        CHECK(soft_header->activeCount == 5U);
        CHECK(soft_nodes[0].position.x == nodes[0].x);
        CHECK(soft_nodes[0].position.y == nodes[0].y);
        CHECK(soft_nodes[0].position.z == nodes[0].z);
        CHECK(soft_nodes[0].velocity.x == 0.0F);
        CHECK(soft_nodes[0].velocity.y == 0.0F);
        CHECK(soft_nodes[0].velocity.z == 0.0F);
        for (std::uint32_t node = 1U; node < 5U; ++node) {
            CHECK(std::isfinite(soft_nodes[node].position.y));
            CHECK(std::isfinite(soft_nodes[node].velocity.y));
        }
        for (const auto &bond : bonds) {
            const auto &first = soft_nodes[bond.first].position;
            const auto &second = soft_nodes[bond.second].position;
            const float x = first.x - second.x;
            const float y = first.y - second.y;
            const float z = first.z - second.z;
            CHECK(std::abs(std::sqrt(x * x + y * y + z * z) -
                           bond.restLength) < 0.03F);
        }

        if (!production) {
            std::array<std::byte, 1024U> soft_before_failure{};
            std::memcpy(soft_before_failure.data(), soft_output_map,
                        soft_before_failure.size());
            CHECK(setenv(
                "PM_TEST_REALTIME_LIMIT_MINUS_ONE", "workgroups", 1) == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
            CHECK(pmCmdStepSimulation(
                commands, mixed_simulation, &step) == PM_SUCCESS);
            execute(PM_ERROR_CAPACITY_EXCEEDED);
            CHECK(unsetenv("PM_TEST_REALTIME_LIMIT_MINUS_ONE") == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdCopySoftBodyState(
                commands, mixed_simulation, 0U, &soft_range) == PM_SUCCESS);
            execute(PM_SUCCESS);
            CHECK(std::memcmp(soft_before_failure.data(), soft_output_map,
                              soft_before_failure.size()) == 0);
        }

        std::array<PmFloat3, 5> replacement_nodes{};
        std::array<PmFloat3, 4> replacement_surface{};
        for (std::uint32_t node = 0U; node < replacement_nodes.size(); ++node) {
            replacement_nodes[node] = nodes[node];
            replacement_nodes[node].x += 0.2F;
        }
        for (std::uint32_t vertex = 0U;
             vertex < replacement_surface.size(); ++vertex) {
            replacement_surface[vertex] = surface_vertices[vertex];
            replacement_surface[vertex].x += 0.2F;
        }
        const PmSoftBodyInfo replacement{PM_TRUE, 0U, 5U,
            replacement_nodes.data(), static_cast<std::uint32_t>(bonds.size()),
            bonds.data(), inverse_masses, 4U, replacement_surface.data(),
            12U, surface_indices, bindings.data()};
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteSoftBodies(
            commands, mixed_simulation, 0U, 1U, &replacement) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopySoftBodyState(
            commands, mixed_simulation, 0U, &soft_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        soft_header =
            static_cast<const PmDomainOutputHeader *>(soft_output_map);
        soft_nodes = reinterpret_cast<const PmSoftBodyNodeState *>(
            soft_header + 1U);
        CHECK(soft_header->activeCount == 5U);
        CHECK(soft_nodes[0].position.x == replacement_nodes[0].x);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        execute(PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteSoftBodies(
            commands, mixed_simulation, 0U, 1U, &body) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopySoftBodyState(
            commands, mixed_simulation, 0U, &soft_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        soft_header =
            static_cast<const PmDomainOutputHeader *>(soft_output_map);
        CHECK(soft_header->activeCount == 5U);
    }
    if (mixed_info.ropeCount != 0U) {
        PmRigidBodyInfo anchor{};
        anchor.active = PM_TRUE;
        anchor.geometry = geometry;
        anchor.motionType = PM_MOTION_STATIC;
        anchor.mass = 1.0F;
        anchor.inertiaDiagonal[0] = anchor.inertiaDiagonal[1] =
            anchor.inertiaDiagonal[2] = 1.0F / 6.0F;
        anchor.state.position[1] = 1.0F;
        anchor.state.orientation[3] = 1.0F;
        anchor.maximumLinearSpeed = anchor.maximumAngularSpeed = 100.0F;
        anchor.collisionMargin = 0.005F;
        const PmFloat3 centerline[2]{
            {0.5F, 1.0F, 0.0F}, {1.0F, 1.0F, 0.0F}};
        PmRopeInfo rope{};
        rope.active = PM_TRUE;
        rope.pointCount = 2U;
        rope.pCenterline = centerline;
        rope.first = {0U, {0.5F, 0.0F, 0.0F}, PM_TRUE};
        const PmBufferRange rope_range{rope_output, 0U, 1024U};

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteRigidBodies(
            commands, mixed_simulation, 0U, 1U, &anchor) == PM_SUCCESS);
        CHECK(pmCmdWriteRopes(
            commands, mixed_simulation, 0U, 1U, &rope) == PM_SUCCESS);
        execute(PM_SUCCESS);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyRopeState(
            commands, mixed_simulation, 0U, &rope_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        execute(PM_SUCCESS);
        PmRealtimeStepStatistics rope_statistics{};
        CHECK(pmGetQueryPoolResults(
            device, query, 0U, 1U, sizeof rope_statistics,
            &rope_statistics, sizeof rope_statistics,
            PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        CHECK(rope_statistics.timeAdvances == 1U);
        CHECK(rope_statistics.domainIterations[4] == 8U);
        CHECK(rope_statistics.total.spatialIndexBuilds == 1U);
        CHECK(rope_statistics.total.spatialIndexPublications == 1U);
        CHECK(rope_statistics.total.intraTickHostReadbacks == 0U);
        CHECK(rope_statistics.couplingEdgeCount == 2U);
        CHECK(rope_statistics.couplingSweeps == 4U);
        const auto *rope_header =
            static_cast<const PmDomainOutputHeader *>(rope_output_map);
        const auto *rope_nodes = reinterpret_cast<const PmRopeNodeState *>(
            rope_header + 1U);
        CHECK(rope_header->activeCount == 6U);
        CHECK(rope_nodes[0].position.x == centerline[0].x);
        CHECK(rope_nodes[0].position.y == centerline[0].y);
        CHECK(rope_nodes[0].position.z == centerline[0].z);
        CHECK(rope_nodes[0].velocity.x == 0.0F);
        CHECK(rope_nodes[0].velocity.y == 0.0F);
        CHECK(rope_nodes[0].velocity.z == 0.0F);
        for (std::uint32_t node = 1U; node < 6U; ++node) {
            CHECK(std::isfinite(rope_nodes[node].position.x));
            CHECK(std::isfinite(rope_nodes[node].position.y));
            CHECK(std::isfinite(rope_nodes[node].velocity.y));
            const auto &first_node = rope_nodes[node - 1U].position;
            const auto &second_node = rope_nodes[node].position;
            const float x = first_node.x - second_node.x;
            const float y = first_node.y - second_node.y;
            const float z = first_node.z - second_node.z;
            CHECK(std::abs(std::sqrt(x * x + y * y + z * z) - 0.1F) <
                  0.03F);
        }

        if (!production) {
            std::array<std::byte, 1024U> rope_before_failure{};
            std::memcpy(rope_before_failure.data(), rope_output_map,
                        rope_before_failure.size());
            CHECK(setenv(
                "PM_TEST_REALTIME_LIMIT_MINUS_ONE", "workgroups", 1) == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
            CHECK(pmCmdStepSimulation(
                commands, mixed_simulation, &step) == PM_SUCCESS);
            execute(PM_ERROR_CAPACITY_EXCEEDED);
            CHECK(unsetenv("PM_TEST_REALTIME_LIMIT_MINUS_ONE") == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdCopyRopeState(
                commands, mixed_simulation, 0U, &rope_range) == PM_SUCCESS);
            execute(PM_SUCCESS);
            CHECK(std::memcmp(rope_before_failure.data(), rope_output_map,
                              rope_before_failure.size()) == 0);
        }

        const PmFloat3 shorter_centerline[2]{
            centerline[0], {0.9F, 1.0F, 0.0F}};
        rope.pCenterline = shorter_centerline;
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteRopes(
            commands, mixed_simulation, 0U, 1U, &rope) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyRopeState(
            commands, mixed_simulation, 0U, &rope_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        rope_header =
            static_cast<const PmDomainOutputHeader *>(rope_output_map);
        CHECK(rope_header->activeCount == 5U);

        rope.pCenterline = centerline;
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteRigidBodies(
            commands, mixed_simulation, 0U, 1U, &anchor) == PM_SUCCESS);
        CHECK(pmCmdWriteRopes(
            commands, mixed_simulation, 0U, 1U, &rope) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyRopeState(
            commands, mixed_simulation, 0U, &rope_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        rope_header =
            static_cast<const PmDomainOutputHeader *>(rope_output_map);
        CHECK(rope_header->activeCount == 6U);
    }
    if (mixed_info.clothCount != 0U && mixed_info.softBodyCount != 0U &&
        mixed_info.ropeCount != 0U) {
        PmRigidBodyInfo floor{};
        floor.active = PM_TRUE;
        floor.geometry = geometry;
        floor.motionType = PM_MOTION_STATIC;
        floor.mass = 1.0F;
        floor.inertiaDiagonal[0] = floor.inertiaDiagonal[1] =
            floor.inertiaDiagonal[2] = 1.0F / 6.0F;
        floor.state.position[1] = -1.0F;
        floor.state.orientation[3] = 1.0F;
        floor.maximumLinearSpeed = floor.maximumAngularSpeed = 100.0F;
        floor.collisionMargin = 0.005F;

        const std::uint32_t cloth_indices[3]{0U, 1U, 2U};
        const float cloth_inverse_masses[3]{1.0F, 1.0F, 1.0F};
        const PmFloat3 cloth_vertices[3]{{-0.1F, 1.0F, 0.0F},
            {0.0F, 1.0F, 0.0F}, {-0.05F, 0.9F, 0.05F}};
        const PmClothInfo cloth{PM_TRUE, 0U, 3U, cloth_vertices, 3U,
            cloth_indices, cloth_inverse_masses};

        const PmFloat3 soft_nodes_input[5]{{0.2F, 1.15F, 0.0F},
            {0.15F, 1.0F, 0.0F}, {0.25F, 1.0F, 0.0F},
            {0.2F, 1.1F, 0.05F}, {0.2F, 1.1F, -0.05F}};
        const float soft_inverse_masses[5]{0.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        std::array<PmSoftBodyBond, 10> soft_bonds{};
        std::uint32_t soft_bond_count{};
        for (std::uint32_t first = 0U; first < 5U; ++first)
            for (std::uint32_t second = first + 1U; second < 5U; ++second) {
                const float x = soft_nodes_input[first].x -
                    soft_nodes_input[second].x;
                const float y = soft_nodes_input[first].y -
                    soft_nodes_input[second].y;
                const float z = soft_nodes_input[first].z -
                    soft_nodes_input[second].z;
                soft_bonds[soft_bond_count++] = {first, second,
                    std::sqrt(x * x + y * y + z * z)};
            }
        const PmFloat3 soft_surface[4]{soft_nodes_input[1],
            soft_nodes_input[2], soft_nodes_input[3], soft_nodes_input[4]};
        const std::uint32_t soft_surface_indices[12]{0U, 2U, 1U,
            0U, 1U, 3U, 0U, 3U, 2U, 1U, 2U, 3U};
        std::array<PmSoftBodySurfaceBinding, 4> soft_bindings{};
        for (std::uint32_t vertex = 0U; vertex < soft_bindings.size(); ++vertex) {
            soft_bindings[vertex].nodes[0] = vertex + 1U;
            soft_bindings[vertex].weights[0] = 1.0F;
        }
        const PmSoftBodyInfo soft_body{PM_TRUE, 0U, 5U, soft_nodes_input,
            static_cast<std::uint32_t>(soft_bonds.size()), soft_bonds.data(),
            soft_inverse_masses, 4U, soft_surface, 12U,
            soft_surface_indices, soft_bindings.data()};

        const PmFloat3 rope_centerline[2]{cloth_vertices[0], soft_surface[0]};
        PmRopeInfo rope{};
        rope.active = PM_TRUE;
        rope.pointCount = 2U;
        rope.pCenterline = rope_centerline;

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteRigidBodies(
            commands, mixed_simulation, 0U, 1U, &floor) == PM_SUCCESS);
        CHECK(pmCmdWriteCloths(
            commands, mixed_simulation, 0U, 1U, &cloth) == PM_SUCCESS);
        CHECK(pmCmdWriteSoftBodies(
            commands, mixed_simulation, 0U, 1U, &soft_body) == PM_SUCCESS);
        CHECK(pmCmdWriteRopes(
            commands, mixed_simulation, 0U, 1U, &rope) == PM_SUCCESS);
        execute(PM_SUCCESS);

        std::array<PmCouplingStateInfo, 3> couplings{};
        auto &rope_soft = couplings[0];
        rope_soft.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
        rope_soft.type = PM_COUPLING_ROPE_SOFT_BODY;
        rope_soft.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_ATTACHMENT_BIT;
        rope_soft.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
        rope_soft.parameters.mechanical.contactDistance = 0.02F;
        rope_soft.parameters.mechanical.friction = 0.2F;
        rope_soft.parameters.mechanical.twoWay = PM_TRUE;
        rope_soft.parameters.attachment.supportRadiusScale = 2.0F;
        rope_soft.parameters.attachment.contactSupportRadiusScale = 1.0F;
        rope_soft.parameters.attachment.maximumAcceleration = 10.0F;
        rope_soft.parameters.attachment.attachLast = PM_TRUE;
        auto &soft_cloth = couplings[1];
        soft_cloth.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
        soft_cloth.type = PM_COUPLING_SOFT_BODY_CLOTH;
        soft_cloth.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
        soft_cloth.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
        soft_cloth.parameters.mechanical.contactDistance = 0.02F;
        soft_cloth.parameters.mechanical.friction = 0.2F;
        soft_cloth.parameters.mechanical.twoWay = PM_TRUE;
        auto &rope_cloth = couplings[2];
        rope_cloth.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
        rope_cloth.type = PM_COUPLING_ROPE_CLOTH;
        rope_cloth.modeFlags = PM_INTERACTION_MODE_ATTACHMENT_BIT;
        rope_cloth.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
        rope_cloth.parameters.attachment.firstVertex = 0U;
        rope_cloth.parameters.attachment.lastVertex = UINT32_MAX;
        rope_cloth.parameters.attachment.effectiveMass = 1.0F;
        rope_cloth.parameters.attachment.maximumAcceleration = 10.0F;

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdSetCouplings2(commands, mixed_simulation,
            static_cast<std::uint32_t>(couplings.size()), couplings.data()) ==
            PM_SUCCESS);
        execute(PM_SUCCESS);

        const PmBufferRange cloth_range{cloth_output, 0U, 1024U};
        const PmBufferRange soft_range{soft_output, 0U, 1024U};
        const PmBufferRange rope_range{rope_output, 0U, 1024U};
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyClothState(
            commands, mixed_simulation, 0U, &cloth_range) == PM_SUCCESS);
        CHECK(pmCmdCopySoftBodyState(
            commands, mixed_simulation, 0U, &soft_range) == PM_SUCCESS);
        CHECK(pmCmdCopyRopeState(
            commands, mixed_simulation, 0U, &rope_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        execute(PM_SUCCESS);
        PmRealtimeStepStatistics coupling_statistics{};
        CHECK(pmGetQueryPoolResults(device, query, 0U, 1U,
            sizeof coupling_statistics, &coupling_statistics,
            sizeof coupling_statistics, PM_QUERY_RESULT_WAIT_BIT) ==
            PM_SUCCESS);
        CHECK(coupling_statistics.couplingEdgeCount == 7U);
        CHECK(coupling_statistics.couplingSweeps == 8U);
        CHECK(coupling_statistics.timeAdvances == 1U);
        CHECK(coupling_statistics.total.spatialIndexBuilds == 1U);
        CHECK(coupling_statistics.total.spatialIndexPublications == 1U);
        CHECK(coupling_statistics.total.intraTickHostReadbacks == 0U);
        const auto *coupled_cloth_header =
            static_cast<const PmDomainOutputHeader *>(cloth_output_map);
        const auto *coupled_cloth =
            reinterpret_cast<const PmClothVertexState *>(
                coupled_cloth_header + 1U);
        const auto *coupled_soft_header =
            static_cast<const PmDomainOutputHeader *>(soft_output_map);
        const auto *coupled_soft =
            reinterpret_cast<const PmSoftBodyNodeState *>(
                coupled_soft_header + 1U);
        const auto *coupled_rope_header =
            static_cast<const PmDomainOutputHeader *>(rope_output_map);
        const auto *coupled_rope =
            reinterpret_cast<const PmRopeNodeState *>(
                coupled_rope_header + 1U);
        CHECK(coupled_cloth_header->activeCount == 3U);
        CHECK(coupled_soft_header->activeCount == 5U);
        CHECK(coupled_rope_header->activeCount != 0U);
        CHECK(std::isfinite(coupled_cloth[0].position.x));
        CHECK(std::isfinite(coupled_soft[1].position.x));
        CHECK(std::isfinite(coupled_rope[0].position.x));

        if (!production) {
            std::array<std::byte, 1024U> cloth_before_failure{};
            std::array<std::byte, 1024U> soft_before_failure{};
            std::array<std::byte, 1024U> rope_before_failure{};
            std::memcpy(cloth_before_failure.data(), cloth_output_map,
                        cloth_before_failure.size());
            std::memcpy(soft_before_failure.data(), soft_output_map,
                        soft_before_failure.size());
            std::memcpy(rope_before_failure.data(), rope_output_map,
                        rope_before_failure.size());
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
            CHECK(pmCmdCopySoftBodyState(
                commands, mixed_simulation, 0U, &soft_range) == PM_SUCCESS);
            CHECK(pmCmdCopyRopeState(
                commands, mixed_simulation, 0U, &rope_range) == PM_SUCCESS);
            execute(PM_SUCCESS);
            CHECK(std::memcmp(cloth_before_failure.data(), cloth_output_map,
                              cloth_before_failure.size()) == 0);
            CHECK(std::memcmp(soft_before_failure.data(), soft_output_map,
                              soft_before_failure.size()) == 0);
            CHECK(std::memcmp(rope_before_failure.data(), rope_output_map,
                              rope_before_failure.size()) == 0);
        }

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdSetCouplings2(
            commands, mixed_simulation, 0U, nullptr) == PM_SUCCESS);
        execute(PM_SUCCESS);
    }
    if (mixed_info.smokeCount != 0U) {
        PmSmokeInfo smoke{};
        smoke.active = PM_TRUE;
        smoke.particleCapacity = smoke_capacity.particleCapacity;
        smoke.emitterCenter = {0.0F, 0.0F, 0.0F};
        smoke.emitterHalfExtents = {0.5F, 0.5F};
        smoke.initialVelocity = {0.5F, 0.0F, 0.0F};
        smoke.wind = smoke.initialVelocity;
        smoke.particlesPerSecond = 120.0F;
        smoke.lifetime = 2.0F;
        smoke.gridMinimum = {-4.0F, -2.0F, -4.0F};
        smoke.gridEdgeLength = 8.0F;
        const PmBufferRange smoke_range{smoke_output, 0U, 4096U};

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteSmoke(
            commands, mixed_simulation, 0U, 1U, &smoke) == PM_SUCCESS);
        execute(PM_SUCCESS);

        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopySmokeState(
            commands, mixed_simulation, 0U, &smoke_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        execute(PM_SUCCESS);
        PmRealtimeStepStatistics particle_smoke_statistics{};
        CHECK(pmGetQueryPoolResults(
            device, query, 0U, 1U, sizeof particle_smoke_statistics,
            &particle_smoke_statistics, sizeof particle_smoke_statistics,
            PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        CHECK(particle_smoke_statistics.timeAdvances == 1U);
        CHECK(particle_smoke_statistics.domainIterations[5] == 6U);
        CHECK(particle_smoke_statistics.total.spatialIndexBuilds == 1U);
        CHECK(particle_smoke_statistics.total.spatialIndexPublications == 1U);
        CHECK(particle_smoke_statistics.total.intraTickHostReadbacks == 0U);
        const auto *smoke_header =
            static_cast<const PmDomainOutputHeader *>(smoke_output_map);
        const auto *smoke_particles =
            reinterpret_cast<const PmSmokeParticleState *>(smoke_header + 1U);
        CHECK(smoke_header->activeCount == 2U);
        for (std::uint32_t particle = 0U;
             particle < smoke_header->activeCount; ++particle) {
            const auto &state = smoke_particles[particle];
            const float speed = std::sqrt(
                state.velocity.x * state.velocity.x +
                state.velocity.y * state.velocity.y +
                state.velocity.z * state.velocity.z);
            CHECK(std::isfinite(state.position.x));
            CHECK(std::isfinite(state.position.y));
            CHECK(std::isfinite(state.position.z));
            CHECK(std::isfinite(state.numberDensity));
            CHECK(std::isfinite(state.pressure));
            CHECK(state.age == 1.0F / 60.0F);
            CHECK(speed <= particle_smoke_material.maximumSpeed + 1.0e-4F);
        }

        if (!production) {
            std::array<std::byte, 4096U> smoke_before_failure{};
            std::memcpy(smoke_before_failure.data(), smoke_output_map,
                        smoke_before_failure.size());
            CHECK(setenv(
                "PM_TEST_REALTIME_LIMIT_MINUS_ONE", "workgroups", 1) == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
            CHECK(pmCmdStepSimulation(
                commands, mixed_simulation, &step) == PM_SUCCESS);
            execute(PM_ERROR_CAPACITY_EXCEEDED);
            CHECK(unsetenv("PM_TEST_REALTIME_LIMIT_MINUS_ONE") == 0);
            CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
            CHECK(pmCmdCopySmokeState(
                commands, mixed_simulation, 0U, &smoke_range) == PM_SUCCESS);
            execute(PM_SUCCESS);
            CHECK(std::memcmp(smoke_before_failure.data(), smoke_output_map,
                              smoke_before_failure.size()) == 0);
        }

        smoke.materialIndex = 1U;
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteSmoke(
            commands, mixed_simulation, 0U, 1U, &smoke) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopySmokeState(
            commands, mixed_simulation, 0U, &smoke_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        execute(PM_SUCCESS);
        PmRealtimeStepStatistics grid_smoke_statistics{};
        CHECK(pmGetQueryPoolResults(
            device, query, 0U, 1U, sizeof grid_smoke_statistics,
            &grid_smoke_statistics, sizeof grid_smoke_statistics,
            PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        CHECK(grid_smoke_statistics.timeAdvances == 1U);
        CHECK(grid_smoke_statistics.domainIterations[5] == 12U);
        CHECK(grid_smoke_statistics.total.spatialIndexBuilds == 1U);
        CHECK(grid_smoke_statistics.total.spatialIndexPublications == 1U);
        CHECK(grid_smoke_statistics.total.intraTickHostReadbacks == 0U);
        smoke_header =
            static_cast<const PmDomainOutputHeader *>(smoke_output_map);
        CHECK(smoke_header->activeCount == 0U);

        smoke.materialIndex = 0U;
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetSimulation(commands, mixed_simulation) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteSmoke(
            commands, mixed_simulation, 0U, 1U, &smoke) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(
            commands, mixed_simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopySmokeState(
            commands, mixed_simulation, 0U, &smoke_range) == PM_SUCCESS);
        execute(PM_SUCCESS);
        smoke_header =
            static_cast<const PmDomainOutputHeader *>(smoke_output_map);
        CHECK(smoke_header->activeCount == 2U);
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
    pmDestroyBuffer(device, soft_output);
    pmDestroyBuffer(device, rope_output);
    pmDestroyBuffer(device, smoke_output);
    pmDestroyBuffer(device, source);
    for (auto allocation : memory) pmFreeMemory(device, allocation);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts(production
        ? "PASS: production realtime resources are fixed and atomic"
        : "PASS: native CPU realtime schedule is fixed and atomic");
}
