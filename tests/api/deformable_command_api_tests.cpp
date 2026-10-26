// SPDX-License-Identifier: MIT
// Analytic translation and pinned invariance through the public command API.
#include <parallel_mater/parallel_mater.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t count = 1;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS && count == 1);
    const float priority = 1;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.cloth = PM_TRUE;
    device_info.enabledFeatures.softBodies = PM_TRUE;
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
    PmBufferCreateInfo buffer_info{};
    buffer_info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = 16384;
    buffer_info.usage = PM_BUFFER_TRANSFER_DST_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;
    PmBuffer output{};
    CHECK(pmCreateBuffer(device, &buffer_info, &output) == PM_SUCCESS);
    const auto output_memory = allocate(buffer_info.size);
    CHECK(pmBindBufferMemory(device, output, output_memory, 0) == PM_SUCCESS);
    void *mapped{};
    CHECK(pmMapMemory(device, output_memory, 0, buffer_info.size, &mapped) == PM_SUCCESS);
    const PmClothCapacity cloth_capacity{3, 3, 3};
    // Nine nodes crosses two native SIMD workgroups and leaves a partial tail.
    const PmSoftBodyCapacity soft_capacity{9, 36, 9, 12};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.clothCount = simulation_info.softBodyCount = 1;
    simulation_info.pClothCapacities = &cloth_capacity;
    simulation_info.pSoftBodyCapacities = &soft_capacity;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        allocate(requirements.persistent.size), 0, allocate(requirements.scratch.size), 0};
    CHECK(pmBindSimulationMemory(device, simulation, &binding) == PM_SUCCESS);
    PmClothMaterialState cloth_material{};
    cloth_material.vertexMass = 1;
    cloth_material.thickness = 0.01F;
    cloth_material.solverIterations = 4;
    cloth_material.fracturePersistenceSubsteps = 1;
    PmSoftBodyMaterialState soft_material{};
    soft_material.nodeMass = 1;
    soft_material.nodeRadius = 0.01F;
    soft_material.springDamping = 0.1F;
    soft_material.shapeMatchingStiffness = 0.5F;
    soft_material.maximumProjectionFraction = 0.2F;
    soft_material.constraintVelocityResponse = 1;
    soft_material.maximumSpeed = 100;
    soft_material.solverIterations = 4;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.clothMaterialCount = pipeline_info.softBodyMaterialCount = 1;
    pipeline_info.pClothMaterials = &cloth_material;
    pipeline_info.pSoftBodyMaterials = &soft_material;
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) == PM_SUCCESS);
    const PmFloat3 cloth_vertices[3]{{-1, 2, 0}, {1, 2, 0}, {0, 2, 1}};
    const std::uint32_t cloth_indices[3]{0, 1, 2};
    const PmFloat3 nodes[9]{{0, 5, 0}, {1, 5, 0}, {0, 6, 0}, {0, 5, 1},
        {1, 6, 0}, {0, 6, 1}, {1, 5, 1}, {1, 6, 1}, {0.5F, 5.5F, 0.5F}};
    const std::uint32_t surface_indices[12]{0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
    std::array<PmSoftBodySurfaceBinding, 9> surface_bindings{};
    for (std::uint32_t i = 0; i < 9; ++i) {
        surface_bindings[i].nodes[0] = i;
        surface_bindings[i].weights[0] = 1;
    }
    std::vector<PmSoftBodyBond> bonds;
    for (std::uint32_t i = 0; i < 9; ++i)
        for (std::uint32_t j = i + 1; j < 9; ++j) {
            const auto a = nodes[i], b = nodes[j];
            bonds.push_back({i, j, std::sqrt((a.x-b.x)*(a.x-b.x) +
                (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z))});
        }
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
    for (bool pinned : {false, true}) {
        std::array<float, 3> cloth_masses;
        std::array<float, 9> soft_masses;
        cloth_masses.fill(pinned ? 0 : 1);
        soft_masses.fill(1);
        if (pinned) soft_masses[0] = 0; // Soft bodies require at least one movable node.
        const PmClothInfo cloth{PM_TRUE, 0, 3, cloth_vertices, 3, cloth_indices, cloth_masses.data()};
        const PmSoftBodyInfo soft{PM_TRUE, 0, 9, nodes, 36, bonds.data(), soft_masses.data(),
            9, nodes, 12, surface_indices, surface_bindings.data()};
        CHECK(pmResetFences(device, 1, &fence) == PM_SUCCESS);
        CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteCloths(commands, simulation, 0, 1, &cloth) == PM_SUCCESS);
        CHECK(pmCmdWriteSoftBodies(commands, simulation, 0, 1, &soft) == PM_SUCCESS);
        const PmStepInfo step{1.0F / 60.0F, 1, {0, -9.81F, 0}, 0};
        for (int frame = 0; frame < 10; ++frame)
            CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
        const PmBufferRange cloth_output{output, 0, 4096}, soft_output{output, 4096, 4096};
        const PmBufferRange surface_output{output, 8192, 4096};
        CHECK(pmCmdCopyClothState(commands, simulation, 0, &cloth_output) == PM_SUCCESS);
        CHECK(pmCmdCopySoftBodyState(commands, simulation, 0, &soft_output) == PM_SUCCESS);
        CHECK(pmCmdCopySoftBodySurface(commands, simulation, 0, &surface_output) == PM_SUCCESS);
        CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
        const auto completed = pmWaitForFences(device, 1, &fence, PM_TRUE, PM_TIMEOUT_INFINITE);
        if (completed != PM_SUCCESS) std::fprintf(stderr, "deformable pinned=%d: %d\n", pinned, int(completed));
        CHECK(completed == PM_SUCCESS);
        const auto *cloth_header = static_cast<const PmDomainOutputHeader *>(mapped);
        const auto *soft_header = reinterpret_cast<const PmDomainOutputHeader *>(static_cast<const std::byte *>(mapped) + 4096);
        const auto *surface_header = reinterpret_cast<const PmDomainOutputHeader *>(static_cast<const std::byte *>(mapped) + 8192);
        CHECK(cloth_header->activeCount == 3 && soft_header->activeCount == 9 && surface_header->activeCount == 9);
        const auto verify = [&](PmFloat3 position, PmFloat3 velocity, PmFloat3 original) {
            const float displacement = pinned ? 0 : -9.81F * 55.0F / 3600.0F;
            CHECK(std::abs(position.x - original.x) < 1e-4F && std::abs(position.z - original.z) < 1e-4F);
            CHECK(std::abs(position.y - original.y - displacement) < 1e-4F);
            CHECK(std::abs(velocity.x) < 1e-4F && std::abs(velocity.z) < 1e-4F);
            CHECK(std::abs(velocity.y - (pinned ? 0 : -9.81F / 6.0F)) < 1e-4F);
        };
        const auto *cloth_states = reinterpret_cast<const PmClothVertexState *>(cloth_header + 1);
        const auto *soft_states = reinterpret_cast<const PmSoftBodyNodeState *>(soft_header + 1);
        const auto *surface = reinterpret_cast<const PmFloat3 *>(surface_header + 1);
        for (int i = 0; i < 3; ++i) verify(cloth_states[i].position, cloth_states[i].velocity, cloth_vertices[i]);
        for (int i = 0; i < 9; ++i) {
            if (!pinned || i == 0) verify(soft_states[i].position, soft_states[i].velocity, nodes[i]);
            CHECK(std::isfinite(soft_states[i].velocity.y));
            CHECK(std::abs(surface[i].x - soft_states[i].position.x) < 1e-4F);
            CHECK(std::abs(surface[i].y - soft_states[i].position.y) < 1e-4F);
            CHECK(std::abs(surface[i].z - soft_states[i].position.z) < 1e-4F);
        }
        for (const auto &bond : bonds) {
            const auto a = soft_states[bond.first].position, b = soft_states[bond.second].position;
            const float length = std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
            CHECK(std::abs(length - bond.restLength) < 0.02F);
        }
    }
    CHECK(pmDeviceWaitIdle(device) == PM_SUCCESS);
    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, pool, 1, &commands);
    pmDestroyCommandPool(device, pool);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyBuffer(device, output);
    for (auto memory : allocations) pmFreeMemory(device, memory);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: cloth/soft-body analytic translation, pinned nodes, surface exports and replacement");
}
