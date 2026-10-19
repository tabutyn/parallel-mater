// SPDX-License-Identifier: MIT
// Public-SDK fluid-cloth graph capacity, replacement, removal, and rollback.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using test::Context;

constexpr std::array<PmFloat3, 4> cloth_vertices{{
    {0.0F, 0.0F, 0.0F},
    {1.0F, 0.0F, 0.0F},
    {0.0F, 1.0F, 0.0F},
    {0.0F, 0.0F, 1.0F},
}};
constexpr std::array<std::uint32_t, 12> cloth_indices{{
    0U, 2U, 1U,
    0U, 1U, 3U,
    0U, 3U, 2U,
    1U, 2U, 3U,
}};
constexpr std::array<float, 4> pinned_inverse_masses{{0.0F, 0.0F, 0.0F, 0.0F}};

struct ResourceObservation {
    std::string id;
    std::string state_sha256;
};

struct Checkpoint {
    std::string id;
    std::string outcome;
    std::string graph_sha256;
    std::vector<ResourceObservation> resources;
};

[[nodiscard]] const char *driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult result = pmGetNativeDevice(device, &native);
    if (result == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (result != PM_SUCCESS) return "unknown";
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    return "unknown";
}

[[nodiscard]] bool supports(PmPhysicalDevice physical,
                            PmInteractionMode mode) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_FAMILY_CLOTH,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        mode,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    PmInteractionCapability capability{};
    return pmGetPhysicalDeviceInteractionCapability(
               physical, &key, &capability) == PM_SUCCESS &&
        capability.maxGraphEdgesPerSimulation >= 1U;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, const float value[3]) {
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
        stream << ',';
        append_float(stream, value[axis]);
    }
}

void append_vector(std::ostringstream &stream, PmFloat3 value) {
    const float components[3]{value.x, value.y, value.z};
    append_vector(stream, components);
}

[[nodiscard]] PmCouplingState edge(float stiffness,
                                   std::uint32_t fluid = 0U) {
    PmCouplingState result{};
    result.type = PM_COUPLING_LIQUID_CLOTH;
    result.first = fluid;
    result.second = 0U;
    result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_CONTAINMENT_BIT;
    result.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    result.parameters.mechanical.contactDistance = 0.05F;
    result.parameters.mechanical.stiffness = stiffness;
    result.parameters.mechanical.damping = 1.0F;
    result.parameters.mechanical.tangentialDrag = 0.2F;
    result.parameters.mechanical.maximumForce = 10.0F;
    result.parameters.mechanical.twoWay = PM_TRUE;
    return result;
}

[[nodiscard]] std::string graph_fingerprint(const PmCouplingState *edges,
                                            std::size_t count) {
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-cloth-containment-graph/v1;count="
              << count;
    for (std::size_t index = 0U; index < count; ++index) {
        const auto &value = edges[index];
        canonical << ";edge=" << index << ','
                  << static_cast<unsigned>(value.type) << ','
                  << value.first << ',' << value.second << ','
                  << value.modeFlags << ','
                  << static_cast<unsigned>(value.materialProfile)
                  << ",contact=";
        append_float(canonical, value.parameters.mechanical.contactDistance);
        canonical << ",stiffness=";
        append_float(canonical, value.parameters.mechanical.stiffness);
        canonical << ",damping=";
        append_float(canonical, value.parameters.mechanical.damping);
        canonical << ",drag=";
        append_float(canonical, value.parameters.mechanical.tangentialDrag);
        canonical << ",maximum-force=";
        append_float(canonical, value.parameters.mechanical.maximumForce);
        canonical << ",two-way="
                  << value.parameters.mechanical.twoWay;
    }
    return sha256(canonical.str());
}

[[nodiscard]] std::string fluid_fingerprint(
    const PmDomainOutputHeader *header) {
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-public-state/v1;count="
              << header->activeCount << ";capacity=" << header->capacity;
    const auto *particles =
        reinterpret_cast<const PmFluidParticleOutput *>(header + 1);
    for (std::uint32_t index = 0U; index < header->activeCount; ++index) {
        canonical << ";particle=" << particles[index].stableId << ",position";
        append_vector(canonical, particles[index].position);
        canonical << ",velocity";
        append_vector(canonical, particles[index].velocity);
        canonical << ",acceleration";
        append_vector(canonical, particles[index].acceleration);
    }
    return sha256(canonical.str());
}

[[nodiscard]] std::string cloth_fingerprint(
    const PmDomainOutputHeader *header) {
    std::ostringstream canonical;
    canonical << "parallel-mater-cloth-public-state/v1;count="
              << header->activeCount << ";capacity=" << header->capacity;
    const auto *vertices =
        reinterpret_cast<const PmClothVertexState *>(header + 1);
    for (std::uint32_t index = 0U; index < header->activeCount; ++index) {
        canonical << ";vertex=" << index << ",position";
        append_vector(canonical, vertices[index].position);
        canonical << ",velocity";
        append_vector(canonical, vertices[index].velocity);
        canonical << ",liquid-force";
        append_vector(canonical, vertices[index].liquidContactForce);
    }
    return sha256(canonical.str());
}

[[nodiscard]] Json resource_json(const ResourceObservation &resource) {
    Json result = Json::object();
    result["id"] = resource.id;
    result["active"] = true;
    result["generation"] = static_cast<std::uint64_t>(1U);
    result["state_sha256"] = resource.state_sha256;
    return result;
}

[[nodiscard]] std::string result_json(
    const std::filesystem::path &root, std::string driver,
    std::string device, ExecutionStatus status,
    const std::vector<Checkpoint> &checkpoints, std::string reason = {}) {
    Json observations = Json::array();
    for (const auto &checkpoint : checkpoints) {
        Json resources = Json::array();
        for (const auto &resource : checkpoint.resources)
            resources.push_back(resource_json(resource));
        Json observation = Json::object();
        observation["id"] = checkpoint.id;
        observation["outcome"] = checkpoint.outcome;
        observation["active_graph_sha256"] = checkpoint.graph_sha256;
        observation["resources"] = std::move(resources);
        observations.push_back(std::move(observation));
    }
    Json families = Json::array();
    families.push_back("fluid");
    families.push_back("cloth");
    Json unrelated = Json::array();
    Json attempt = Json::object();
    attempt["fixture"] = "fluid-cloth-containment-public-lifecycle";
    attempt["status"] = status == ExecutionStatus::completed ? "completed" :
        status == ExecutionStatus::unsupported ? "unsupported" : "failed";
    if (!reason.empty()) attempt["reason"] = reason;
    Json lifecycle_case = Json::object();
    lifecycle_case["id"] =
        "fluid-cloth-containment-capacity-replacement-removal-rollback";
    lifecycle_case["execution_status"] = attempt["status"];
    lifecycle_case["attempt"] = std::move(attempt);
    lifecycle_case["families"] = std::move(families);
    lifecycle_case["unrelated_resources"] = std::move(unrelated);
    lifecycle_case["observations"] = std::move(observations);
    Json cases = Json::array();
    cases.push_back(std::move(lifecycle_case));
    Json driver_value = Json::object();
    driver_value["name"] = std::move(driver);
    driver_value["device"] = std::move(device);
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = "fluid-cloth-containment-lifecycle-v1";
    result["profile_sha256"] = sha256_contract_file(
        root /
        "conformance/v1/lifecycle/fluid-cloth-containment-lifecycle-v1.json");
    result["execution_status"] = status == ExecutionStatus::completed ?
        "completed" : status == ExecutionStatus::unsupported ?
        "unsupported" : "failed";
    result["driver"] = std::move(driver_value);
    result["cases"] = std::move(cases);
    return result.serialize();
}

} // namespace

std::string conformance_fluid_cloth_containment_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    if (status != nullptr) *status = ExecutionStatus::failed;
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.cloth = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    if (!context.open(needed)) {
        if (status != nullptr) *status = ExecutionStatus::unsupported;
        return result_json(root, "unavailable", "unavailable",
                           ExecutionStatus::unsupported, {});
    }
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    const std::string selected_driver = driver_name(context.device);
    if (!supports(context.physical, PM_INTERACTION_MODE_CONTACT) ||
        !supports(context.physical, PM_INTERACTION_MODE_CONTAINMENT)) {
        if (status != nullptr) *status = ExecutionStatus::unsupported;
        return result_json(root, selected_driver, properties.deviceName,
                           ExecutionStatus::unsupported, {});
    }

    PmParticleState particle{};
    particle.position[0] = 0.25F;
    particle.position[1] = 0.25F;
    particle.position[2] = 0.25F;
    particle.temperatureCelsius = 20.0F;
    const auto input = context.buffer(sizeof(particle), PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(input.mapped, &particle, sizeof(particle));
    context.flush(input);
    const PmBufferView particles{
        {input.handle, 0U, sizeof(particle)}, PM_FORMAT_PARTICLE_STATE,
        1U, sizeof(PmParticleState)};

    const PmFluidCapacity fluid_capacity{1U, 16U};
    const PmClothCapacity cloth_capacity{4U, 12U, 12U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = 1U;
    create.pFluidCapacities = &fluid_capacity;
    create.clothCount = 1U;
    create.pClothCapacities = &cloth_capacity;
    create.couplingCapacity = 1U;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &create, &simulation) ==
             PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) ==
             PM_SUCCESS);

    PmLiquidMaterialState fluid{};
    fluid.restDensity = 1000.0F;
    fluid.particleRadius = 0.025F;
    fluid.supportRadius = 0.1F;
    fluid.restParticleVolume = 0.001F;
    fluid.maximumSpeed = 10.0F;
    fluid.maximumPairAcceleration = 100.0F;
    fluid.densityIterations = 1U;
    PmClothMaterialState cloth{};
    cloth.vertexMass = 1.0F;
    cloth.thickness = 0.025F;
    cloth.stretchCompliance = 1.0e-6F;
    cloth.bendingCompliance = 0.1F;
    cloth.solverIterations = 4U;
    cloth.fracturePersistenceSubsteps = 1U;
    cloth.preserveVolume = PM_TRUE;
    cloth.volumeCompliance = 1.0e-7F;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &fluid;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const PmClothInfo cloth_info{
        PM_TRUE, 0U, static_cast<std::uint32_t>(cloth_vertices.size()),
        cloth_vertices.data(), static_cast<std::uint32_t>(cloth_indices.size()),
        cloth_indices.data(), pinned_inverse_masses.data()};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &particles) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U,
                              &cloth_info) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmDeviceSize fluid_bytes = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    const PmDeviceSize cloth_bytes = sizeof(PmDomainOutputHeader) +
        cloth_vertices.size() * sizeof(PmClothVertexState);
    const auto output = context.buffer(fluid_bytes + cloth_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange fluid_output{output.handle, 0U, fluid_bytes};
    const PmBufferRange cloth_output{output.handle, fluid_bytes, cloth_bytes};
    const auto baseline = edge(100.0F);
    const auto replacement = edge(120.0F);
    const std::array<PmCouplingState, 2> overflowing{{replacement, baseline}};
    const auto invalid = edge(120.0F, 1U);
    const std::string baseline_hash = graph_fingerprint(&baseline, 1U);
    const std::string replacement_hash = graph_fingerprint(&replacement, 1U);
    const std::string empty_hash = graph_fingerprint(nullptr, 0U);
    std::vector<Checkpoint> checkpoints;

    const auto capture = [&](std::string id, std::string outcome,
                             std::string graph_hash) {
        context.begin();
        PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                         &fluid_output) == PM_SUCCESS);
        PM_CHECK(pmCmdCopyClothState(context.commands, simulation, 0U,
                                     &cloth_output) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(output);
        const auto *fluid_header =
            static_cast<const PmDomainOutputHeader *>(output.mapped);
        const auto *cloth_header = reinterpret_cast<const PmDomainOutputHeader *>(
            static_cast<const std::byte *>(output.mapped) + fluid_bytes);
        PM_CHECK(fluid_header->activeCount == 1U);
        PM_CHECK(cloth_header->activeCount == cloth_vertices.size());
        checkpoints.push_back({std::move(id), std::move(outcome),
                               std::move(graph_hash),
                               {{"fluid", fluid_fingerprint(fluid_header)},
                                {"cloth", cloth_fingerprint(cloth_header)}}});
    };
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    const auto submit_graph = [&](const PmCouplingState *edges,
                                  std::uint32_t count,
                                  PmResult record_result,
                                  PmResult submit_result) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, simulation, count, edges) ==
                 record_result);
        if (record_result == PM_SUCCESS)
            PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                     PM_SUCCESS);
        PM_CHECK(context.submit() == submit_result);
    };

    submit_graph(&baseline, 1U, PM_SUCCESS, PM_SUCCESS);
    capture("baseline", "PM_SUCCESS", baseline_hash);
    submit_graph(&replacement, 1U, PM_SUCCESS, PM_SUCCESS);
    capture("replacement", "PM_SUCCESS", replacement_hash);
    submit_graph(overflowing.data(),
                 static_cast<std::uint32_t>(overflowing.size()),
                 PM_ERROR_CAPACITY_EXCEEDED, PM_SUCCESS);
    capture("rejected-capacity", "PM_ERROR_CAPACITY_EXCEEDED",
            replacement_hash);
    submit_graph(&invalid, 1U, PM_SUCCESS, PM_ERROR_INVALID_STATE);
    capture("partial-failure", "PM_ERROR_INVALID_STATE", replacement_hash);
    submit_graph(nullptr, 0U, PM_SUCCESS, PM_SUCCESS);
    capture("removal", "PM_SUCCESS", empty_hash);
    submit_graph(&baseline, 1U, PM_SUCCESS, PM_SUCCESS);
    capture("recovery", "PM_SUCCESS", baseline_hash);

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    if (status != nullptr) *status = ExecutionStatus::completed;
    return result_json(root, selected_driver, properties.deviceName,
                       ExecutionStatus::completed, checkpoints);
}

} // namespace parallel_mater::conformance
