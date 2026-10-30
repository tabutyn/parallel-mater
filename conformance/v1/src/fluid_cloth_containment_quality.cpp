// SPDX-License-Identifier: MIT
// Public-SDK fluid containment quality fixture. No driver-private state.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

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
constexpr std::array<float, 4> cloth_inverse_masses{{1.0F, 1.0F, 1.0F, 1.0F}};
constexpr float timestep = 1.0F / 60.0F;

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

[[nodiscard]] Json vector_json(const float value[3]) {
    Json result = Json::array();
    for (std::size_t axis = 0U; axis < 3U; ++axis)
        result.push_back(static_cast<double>(value[axis]));
    return result;
}

[[nodiscard]] Json vector_json(PmFloat3 value) {
    const float components[3]{value.x, value.y, value.z};
    return vector_json(components);
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
        capability.maxGraphEdgesPerSimulation !=
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE;
}

[[nodiscard]] PmLiquidMaterialState fluid_material() {
    PmLiquidMaterialState material{};
    material.restDensity = 1000.0F;
    material.particleRadius = 0.05F;
    material.supportRadius = 0.2F;
    material.restParticleVolume = 0.001F;
    material.maximumSpeed = 5.0F;
    material.maximumPairAcceleration = 1000.0F;
    material.densityIterations = 1U;
    return material;
}

[[nodiscard]] PmClothMaterialState cloth_material() {
    PmClothMaterialState material{};
    material.vertexMass = 1.0F;
    material.thickness = 0.05F;
    material.stretchCompliance = 4.0e-6F;
    material.bendingCompliance = 0.01F;
    material.solverIterations = 8U;
    material.fracturePersistenceSubsteps = 1U;
    material.preserveVolume = PM_TRUE;
    material.volumeCompliance = 1.0e-7F;
    return material;
}

[[nodiscard]] PmCouplingState containment_edge(float stiffness) {
    PmCouplingState edge{};
    edge.type = PM_COUPLING_LIQUID_CLOTH;
    edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_CONTAINMENT_BIT;
    edge.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    edge.parameters.mechanical.contactDistance = 0.1F;
    edge.parameters.mechanical.stiffness = stiffness;
    edge.parameters.mechanical.damping = 10.0F;
    edge.parameters.mechanical.tangentialDrag = 0.0F;
    edge.parameters.mechanical.maximumForce = 100.0F;
    edge.parameters.mechanical.twoWay = PM_TRUE;
    return edge;
}

[[nodiscard]] Json fluid_state_json(const PmFluidParticleOutput &particle) {
    Json result = Json::object();
    result["stable_id"] = static_cast<std::uint64_t>(particle.stableId);
    result["position_m"] = vector_json(particle.position);
    result["velocity_m_s"] = vector_json(particle.velocity);
    result["acceleration_m_s2"] = vector_json(particle.acceleration);
    result["foam"] = static_cast<double>(particle.foam);
    return result;
}

[[nodiscard]] Json cloth_state_json(const PmClothVertexState &vertex,
                                    std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(vertex.position);
    result["velocity_m_s"] = vector_json(vertex.velocity);
    result["liquid_contact_force_n"] = vector_json(vertex.liquidContactForce);
    return result;
}

[[nodiscard]] Json settings_json(const PmParticleState &particle) {
    const auto fluid = fluid_material();
    const auto cloth = cloth_material();
    const auto edge = containment_edge(100.0F);
    Json vertices = Json::array();
    for (const auto vertex : cloth_vertices) vertices.push_back(vector_json(vertex));
    Json indices = Json::array();
    for (const auto index : cloth_indices)
        indices.push_back(static_cast<std::uint64_t>(index));
    Json initial_particles = Json::array();
    Json initial = Json::object();
    initial["stable_id"] = static_cast<std::uint64_t>(0U);
    initial["position_m"] = vector_json(particle.position);
    initial["velocity_m_s"] = vector_json(particle.velocity);
    initial_particles.push_back(std::move(initial));

    Json result = Json::object();
    result["id"] = "closed-tetrahedron-outward-pulse";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["fluid_particle_count"] = static_cast<std::uint64_t>(1U);
    result["fluid_rest_density_kg_m3"] =
        static_cast<double>(fluid.restDensity);
    result["fluid_particle_radius_m"] =
        static_cast<double>(fluid.particleRadius);
    result["fluid_support_radius_m"] =
        static_cast<double>(fluid.supportRadius);
    result["fluid_rest_particle_volume_m3"] =
        static_cast<double>(fluid.restParticleVolume);
    result["fluid_maximum_speed_m_s"] =
        static_cast<double>(fluid.maximumSpeed);
    result["cloth_vertex_mass_kg"] = static_cast<double>(cloth.vertexMass);
    result["cloth_thickness_m"] = static_cast<double>(cloth.thickness);
    result["cloth_stretch_compliance_m_per_n"] =
        static_cast<double>(cloth.stretchCompliance);
    result["cloth_bending_compliance_m_per_n"] =
        static_cast<double>(cloth.bendingCompliance);
    result["cloth_solver_iterations"] =
        static_cast<std::uint64_t>(cloth.solverIterations);
    result["cloth_preserve_volume"] = true;
    result["cloth_target_volume_m3"] =
        static_cast<double>(cloth.targetVolume);
    result["cloth_volume_compliance_m3_per_pa"] =
        static_cast<double>(cloth.volumeCompliance);
    result["contact_distance_m"] =
        static_cast<double>(edge.parameters.mechanical.contactDistance);
    result["stiffness_n_per_m"] =
        static_cast<double>(edge.parameters.mechanical.stiffness);
    result["damping_n_s_per_m"] =
        static_cast<double>(edge.parameters.mechanical.damping);
    result["tangential_drag_n_s_per_m"] =
        static_cast<double>(edge.parameters.mechanical.tangentialDrag);
    result["maximum_force_n"] =
        static_cast<double>(edge.parameters.mechanical.maximumForce);
    result["two_way"] = true;
    result["cloth_vertices_m"] = std::move(vertices);
    result["cloth_triangle_indices"] = std::move(indices);
    result["initial_particles"] = std::move(initial_particles);
    result["oracle"] =
        "public-state-closed-manifold-signed-distance-and-reciprocal-force";
    return result;
}

} // namespace

std::string conformance_fluid_cloth_containment_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    if (status != nullptr) *status = ExecutionStatus::failed;
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.cloth = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    if (!context.open(needed)) {
        if (status != nullptr) *status = ExecutionStatus::unsupported;
        return {};
    }

    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    if (!supports(context.physical, PM_INTERACTION_MODE_CONTACT) ||
        !supports(context.physical, PM_INTERACTION_MODE_CONTAINMENT)) {
        if (status != nullptr) *status = ExecutionStatus::unsupported;
        return {};
    }

    PmParticleState particle{};
    particle.position[0] = 0.05F;
    particle.position[1] = 0.2F;
    particle.position[2] = 0.2F;
    particle.velocity[0] = -1.0F;
    particle.temperatureCelsius = 20.0F;
    const auto input = context.buffer(sizeof(particle), PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(input.mapped, &particle, sizeof(particle));
    context.flush(input);

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

    const auto fluid = fluid_material();
    const auto cloth = cloth_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &fluid;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const PmBufferView particles{
        {input.handle, 0U, sizeof(particle)}, PM_FORMAT_PARTICLE_STATE,
        1U, sizeof(PmParticleState)};
    const PmClothInfo cloth_info{
        PM_TRUE, 0U, static_cast<std::uint32_t>(cloth_vertices.size()),
        cloth_vertices.data(), static_cast<std::uint32_t>(cloth_indices.size()),
        cloth_indices.data(), cloth_inverse_masses.data()};
    const auto edge = containment_edge(100.0F);
    const PmDeviceSize fluid_bytes = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    const PmDeviceSize cloth_bytes = sizeof(PmDomainOutputHeader) +
        cloth_vertices.size() * sizeof(PmClothVertexState);
    const auto output = context.buffer(2U * (fluid_bytes + cloth_bytes),
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmDeviceSize initial_fluid_offset = 0U;
    const PmDeviceSize initial_cloth_offset = fluid_bytes;
    const PmDeviceSize final_fluid_offset = fluid_bytes + cloth_bytes;
    const PmDeviceSize final_cloth_offset = 2U * fluid_bytes + cloth_bytes;
    const PmBufferRange initial_fluid{
        output.handle, initial_fluid_offset, fluid_bytes};
    const PmBufferRange initial_cloth{
        output.handle, initial_cloth_offset, cloth_bytes};
    const PmBufferRange final_fluid{
        output.handle, final_fluid_offset, fluid_bytes};
    const PmBufferRange final_cloth{
        output.handle, final_cloth_offset, cloth_bytes};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &particles) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U,
                              &cloth_info) == PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &edge) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                     &initial_fluid) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyClothState(context.commands, simulation, 0U,
                                 &initial_cloth) == PM_SUCCESS);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                     &final_fluid) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyClothState(context.commands, simulation, 0U,
                                 &final_cloth) == PM_SUCCESS);
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error(
            "fluid-cloth containment submission failed: " +
            std::to_string(submitted));
    context.invalidate(output);

    const auto header = [&](PmDeviceSize offset) {
        return reinterpret_cast<const PmDomainOutputHeader *>(
            static_cast<const std::byte *>(output.mapped) + offset);
    };
    const auto *initial_fluid_header = header(initial_fluid_offset);
    const auto *initial_cloth_header = header(initial_cloth_offset);
    const auto *final_fluid_header = header(final_fluid_offset);
    const auto *final_cloth_header = header(final_cloth_offset);
    if (initial_fluid_header->activeCount != 1U ||
        final_fluid_header->activeCount != 1U ||
        initial_cloth_header->activeCount != cloth_vertices.size() ||
        final_cloth_header->activeCount != cloth_vertices.size())
        throw std::runtime_error("fluid-cloth public output count mismatch");
    const auto *initial_fluid_state =
        reinterpret_cast<const PmFluidParticleOutput *>(initial_fluid_header + 1);
    const auto *final_fluid_state =
        reinterpret_cast<const PmFluidParticleOutput *>(final_fluid_header + 1);
    const auto *initial_cloth_state =
        reinterpret_cast<const PmClothVertexState *>(initial_cloth_header + 1);
    const auto *final_cloth_state =
        reinterpret_cast<const PmClothVertexState *>(final_cloth_header + 1);

    Json initial_particles = Json::array();
    initial_particles.push_back(fluid_state_json(initial_fluid_state[0]));
    Json final_particles = Json::array();
    final_particles.push_back(fluid_state_json(final_fluid_state[0]));
    Json initial_vertices = Json::array();
    Json final_vertices = Json::array();
    for (std::uint32_t index = 0U; index < cloth_vertices.size(); ++index) {
        initial_vertices.push_back(cloth_state_json(initial_cloth_state[index], index));
        final_vertices.push_back(cloth_state_json(final_cloth_state[index], index));
    }

    Json fixture = settings_json(particle);
    fixture["initial_fluid_state"] = std::move(initial_particles);
    fixture["final_fluid_state"] = std::move(final_particles);
    fixture["initial_cloth_state"] = std::move(initial_vertices);
    fixture["final_cloth_state"] = std::move(final_vertices);
    Json driver = Json::object();
    driver["name"] = driver_name(context.device);
    driver["device"] = properties.deviceName;
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = "fluid-cloth-containment-v1";
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/fluid-cloth-containment-v1.json");
    result["execution_status"] = "completed";
    result["driver"] = std::move(driver);
    result["fixture"] = std::move(fixture);

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    if (status != nullptr) *status = ExecutionStatus::completed;
    return result.serialize();
}

} // namespace parallel_mater::conformance
