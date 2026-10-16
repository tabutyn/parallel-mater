// SPDX-License-Identifier: MIT
// Public API only: analytic free fall and pair momentum, not a solver clone.
#include <parallel_mater/parallel_mater.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t count = 1;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS && count == 1);
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);
    CHECK(features.particleLiquids && features.liquidRigidCoupling);
    CHECK(!features.liquidLiquidCoupling);
    const float priority = 1;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.particleLiquids = PM_TRUE;
    device_info.enabledFeatures.rigidBodies = PM_TRUE;
    device_info.enabledFeatures.liquidRigidCoupling = PM_TRUE;
    PmDevice device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0, 0, &queue);
    std::vector<PmDeviceMemory> allocations;
    const auto allocate = [&](PmDeviceSize size) {
        PmMemoryAllocateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize = size;
        info.memoryTypeIndex = 1;
        PmDeviceMemory memory{};
        CHECK(pmAllocateMemory(device, &info, &memory) == PM_SUCCESS);
        allocations.push_back(memory);
        return memory;
    };
    const auto buffer = [&](void **mapped) {
        PmBufferCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = 65536;
        info.usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT |
                     PM_BUFFER_PARTICLE_INPUT_BIT | PM_BUFFER_GEOMETRY_INPUT_BIT |
                     PM_BUFFER_SIMULATION_OUTPUT_BIT;
        PmBuffer result{};
        CHECK(pmCreateBuffer(device, &info, &result) == PM_SUCCESS);
        const auto memory = allocate(info.size);
        CHECK(pmBindBufferMemory(device, result, memory, 0) == PM_SUCCESS);
        CHECK(pmMapMemory(device, memory, 0, info.size, mapped) == PM_SUCCESS);
        return result;
    };
    void *input{}, *output{};
    const auto source = buffer(&input), destination = buffer(&output);
    const float cube[8][3] = {
        {-0.5F,-0.5F,-0.5F},{0.5F,-0.5F,-0.5F},{0.5F,0.5F,-0.5F},{-0.5F,0.5F,-0.5F},
        {-0.5F,-0.5F,0.5F},{0.5F,-0.5F,0.5F},{0.5F,0.5F,0.5F},{-0.5F,0.5F,0.5F}};
    const std::uint32_t indices[36] = {
        0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
    std::memcpy(static_cast<std::byte *>(input) + 32768, cube, sizeof(cube));
    std::memcpy(static_cast<std::byte *>(input) + 32768 + sizeof(cube), indices, sizeof(indices));
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {{source, 32768, sizeof(cube)}, PM_FORMAT_R32G32B32_SFLOAT, 8, 12};
    geometry_info.indices = {{source, 32768 + sizeof(cube), sizeof(indices)}, PM_FORMAT_R32_UINT, 36, 4};
    PmGeometry geometry{};
    CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements geometry_requirements{};
    pmGetGeometryMemoryRequirements(device, geometry, &geometry_requirements);
    CHECK(pmBindGeometryMemory(device, geometry, allocate(geometry_requirements.size), 0) == PM_SUCCESS);
    constexpr std::uint32_t capacity = 600, active = 513;
    const PmFluidCapacity fluid_capacity{capacity, capacity};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.fluidCount = 1;
    simulation_info.rigidBodyCapacity = 1;
    simulation_info.contactCapacity = 16;
    simulation_info.couplingCapacity = 1;
    simulation_info.pFluidCapacities = &fluid_capacity;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        allocate(requirements.persistent.size), 0, allocate(requirements.scratch.size), 0};
    CHECK(pmBindSimulationMemory(device, simulation, &binding) == PM_SUCCESS);
    PmLiquidMaterialState material{};
    material.restDensity = 1000;
    material.particleRadius = 0.025F;
    material.supportRadius = 0.1F;
    material.pressureStiffness = 50;
    material.maximumSpeed = 100;
    material.densityIterations = 1;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 1;
    pipeline_info.pFluidMaterials = &material;
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_LIQUID_RIGID;
    coupling.parameters.mechanical.twoWay = PM_TRUE;
    pipeline_info.couplingCount = 1;
    pipeline_info.pCouplings = &coupling;
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) == PM_SUCCESS);
    coupling.type = PM_COUPLING_LIQUID_LIQUID;
    PmPhysicsPipeline unsupported{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &unsupported) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(!unsupported);
    coupling.type = PM_COUPLING_LIQUID_RIGID;
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1, &commands) == PM_SUCCESS);
    PmFence fence{};
    CHECK(pmCreateFence(device, 0, &fence) == PM_SUCCESS);
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    const PmBufferRange range{destination, 0,
        sizeof(PmDomainOutputHeader) + capacity * sizeof(PmFluidParticleOutput)};
    auto *particles = static_cast<PmParticleState *>(input);
    auto *header = static_cast<PmDomainOutputHeader *>(output);
    auto *states = reinterpret_cast<PmFluidParticleOutput *>(header + 1);
    bool built_geometry = false;
    const auto run = [&](std::uint32_t live, std::uint32_t steps, float gravity,
                         const PmRigidBodyInfo *body = nullptr) {
        CHECK(pmResetFences(device, 1, &fence) == PM_SUCCESS);
        CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        if (body) {
            if (!built_geometry) {
                CHECK(pmCmdBuildGeometry(commands, geometry) == PM_SUCCESS);
                built_geometry = true;
            }
            CHECK(pmCmdWriteRigidBodies(commands, simulation, 0, 1, body) == PM_SUCCESS);
        }
        const PmBufferView view{{source, 0, live * sizeof(PmParticleState)},
            PM_FORMAT_PARTICLE_STATE, live, sizeof(PmParticleState)};
        CHECK(pmCmdSetFluidParticles(commands, simulation, 0, &view) == PM_SUCCESS);
        const PmStepInfo step{1.0F / 60.0F, 1U, {0, gravity, 0}, 0};
        for (std::uint32_t frame = 0; frame < steps; ++frame)
            CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyFluidParticles(commands, simulation, 0, &range) == PM_SUCCESS);
        if (body) {
            const PmBufferRange rigid_range{destination, 61440, sizeof(PmRigidState)};
            CHECK(pmCmdCopyRigidStates(commands, simulation, 0, 1, &rigid_range) == PM_SUCCESS);
        }
        PmHitBox box{};
        box.orientation[3] = 1;
        box.halfExtents[0] = 200;
        box.halfExtents[1] = box.halfExtents[2] = 3;
        const PmBufferRange query_range{destination, 32768,
            sizeof(PmHitBoxOutputHeader) + sizeof(std::uint32_t) + capacity * sizeof(PmHitBoxParticle)};
        CHECK(pmCmdQueryHitBox(commands, simulation, &box, &query_range) == PM_SUCCESS);
        CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
        const auto completed = pmWaitForFences(device, 1, &fence, PM_TRUE, PM_TIMEOUT_INFINITE);
        if (completed != PM_SUCCESS)
            std::fprintf(stderr, "liquid case live=%u steps=%u rigid=%d: %d\n",
                live, steps, body != nullptr, int(completed));
        CHECK(completed == PM_SUCCESS);
        CHECK(header->activeCount == live && header->capacity == capacity);
        const auto *hits = reinterpret_cast<const PmHitBoxOutputHeader *>(
            static_cast<const std::byte *>(output) + 32768);
        CHECK(hits->rigidBodyCount == (body ? 1U : 0U) && hits->particleCount == live);
        PmResult execution{};
        CHECK(pmGetSimulationStatus(device, simulation, &execution) == PM_SUCCESS);
        CHECK(execution == PM_SUCCESS);
    };
    for (std::uint32_t i = 0; i < active; ++i) {
        particles[i] = {};
        particles[i].position[0] = 0.2F * i;
        particles[i].position[1] = 2;
        particles[i].temperatureCelsius = 20;
    }
    // Non-multiple workgroup size plus unused capacity exercises sort sentinels.
    run(active, 10, -9.81F);
    for (std::uint32_t i = 0; i < active; ++i) {
        CHECK(states[i].stableId == i);
        CHECK(std::abs(states[i].position[0] - particles[i].position[0]) < 1e-5F);
        CHECK(std::abs(states[i].position[1] - (2.0F - 9.81F * 55.0F / 3600.0F)) < 1e-5F);
        CHECK(std::abs(states[i].velocity[1] + 9.81F / 6.0F) < 1e-5F);
        CHECK(states[i].temperatureCelsius == 20);
    }
    particles[0] = {}; particles[1] = {};
    particles[0].position[0] = -0.025F;
    particles[1].position[0] = 0.025F;
    run(2, 1, 0);
    CHECK(states[0].velocity[0] < -0.01F && states[1].velocity[0] > 0.01F);
    CHECK(std::abs(states[0].velocity[0] + states[1].velocity[0]) < 1e-6F);
    CHECK(std::abs(states[0].position[0] + states[1].position[0]) < 1e-6F);
    CHECK(std::abs(states[0].acceleration[0] + states[1].acceleration[0]) < 1e-5F);
    PmRigidBodyInfo body{};
    body.active = PM_TRUE;
    body.geometry = geometry;
    body.motionType = PM_MOTION_STATIC;
    body.mass = 1;
    body.state.orientation[3] = 1;
    for (auto &value : body.inertiaDiagonal) value = 1.0F / 6.0F;
    body.maximumLinearSpeed = body.maximumAngularSpeed = 100;
    body.collisionMargin = 0.005F;
    particles[0] = {};
    particles[0].position[1] = 0.54F;
    particles[0].velocity[1] = -1;
    run(1, 1, 0, &body);
    CHECK(states[0].position[1] >= 0.524F);
    CHECK(states[0].velocity[1] >= -1e-4F);
    body.motionType = PM_MOTION_DYNAMIC;
    run(1, 1, 0, &body);
    const auto *rigid = reinterpret_cast<const PmRigidState *>(
        static_cast<const std::byte *>(output) + 61440);
    CHECK(rigid->linearVelocity[1] < -1e-3F);
    constexpr float particle_mass = 1000.0F * 0.05F * 0.05F * 0.05F;
    CHECK(std::abs(particle_mass * states[0].velocity[1] + rigid->linearVelocity[1] + particle_mass) < 1e-4F);
    CHECK(pmDeviceWaitIdle(device) == PM_SUCCESS);
    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, pool, 1, &commands);
    pmDestroyCommandPool(device, pool);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyGeometry(device, geometry);
    pmDestroyBuffer(device, destination);
    pmDestroyBuffer(device, source);
    for (auto memory : allocations) pmFreeMemory(device, memory);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: liquid free fall, sort, queries, pair momentum and two-way rigid contact");
}
