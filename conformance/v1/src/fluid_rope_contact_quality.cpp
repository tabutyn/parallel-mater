// SPDX-License-Identifier: MIT
// Public-SDK fluid/rope contact quality fixture. No driver-private state.
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

constexpr float timestep = 1.0F / 60.0F;
constexpr float fluid_maximum_speed = 5.999F;
constexpr float rope_maximum_speed = 2.999F;
constexpr float rope_stretch_compliance = 1.0e-3F;

constexpr std::array<PmFloat3, 2> rope_centerline{{
    {0.0F, 0.0F, 0.0F},
    {0.0F, 0.1F, 0.0F},
}};

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

[[nodiscard]] bool supports(PmPhysicalDevice physical) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        PM_INTERACTION_MODE_CONTACT,
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
    material.maximumSpeed = fluid_maximum_speed;
    material.maximumPairAcceleration = 1000.0F;
    material.densityIterations = 1U;
    return material;
}

[[nodiscard]] PmRopeMaterialState rope_material() {
    PmRopeMaterialState material{};
    material.nodeSpacing = 0.1F;
    material.radius = 0.05F;
    material.mass = 4.0F;
    material.stretchCompliance = rope_stretch_compliance;
    material.maximumSubstepTimestep = timestep;
    material.maximumSpeed = rope_maximum_speed;
    material.solverIterations = 1U;
    return material;
}

[[nodiscard]] PmCouplingState contact_edge() {
    PmCouplingState edge{};
    edge.type = PM_COUPLING_LIQUID_ROPE;
    edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    edge.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    edge.parameters.mechanical.contactDistance = 0.1F;
    edge.parameters.mechanical.friction = 0.0F;
    edge.parameters.mechanical.maximumAcceleration = 1000.0F;
    edge.parameters.mechanical.twoWay = PM_TRUE;
    return edge;
}

[[nodiscard]] Json fluid_state_json(const PmFluidParticleOutput &particle) {
    Json result = Json::object();
    result["stable_id"] = static_cast<std::uint64_t>(particle.stableId);
    result["position_m"] = vector_json(particle.position);
    result["velocity_m_s"] = vector_json(particle.velocity);
    result["acceleration_m_s2"] = vector_json(particle.acceleration);
    return result;
}

[[nodiscard]] Json rope_state_json(const PmRopeNodeState &node,
                                   std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(node.position);
    result["velocity_m_s"] = vector_json(node.velocity);
    result["liquid_contact_force_n"] = vector_json(node.liquidContactForce);
    return result;
}

[[nodiscard]] Json settings_json(const PmParticleState &particle) {
    const auto fluid = fluid_material();
    const auto rope = rope_material();
    const auto edge = contact_edge();
    Json centerline = Json::array();
    for (const auto point : rope_centerline)
        centerline.push_back(vector_json(point));
    Json initial_particles = Json::array();
    Json initial = Json::object();
    initial["stable_id"] = static_cast<std::uint64_t>(0U);
    initial["position_m"] = vector_json(particle.position);
    initial["velocity_m_s"] = vector_json(particle.velocity);
    initial_particles.push_back(std::move(initial));

    Json result = Json::object();
    result["id"] = "quarter-segment-normal-impact";
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
    result["rope_node_count"] =
        static_cast<std::uint64_t>(rope_centerline.size());
    result["rope_node_spacing_m"] = static_cast<double>(rope.nodeSpacing);
    result["rope_radius_m"] = static_cast<double>(rope.radius);
    result["rope_total_mass_kg"] = static_cast<double>(rope.mass);
    result["rope_stretch_compliance_m_per_n"] =
        static_cast<double>(rope.stretchCompliance);
    result["rope_velocity_damping"] =
        static_cast<double>(rope.velocityDamping);
    result["rope_maximum_substep_timestep_seconds"] =
        static_cast<double>(rope.maximumSubstepTimestep);
    result["rope_friction"] = static_cast<double>(rope.friction);
    result["rope_maximum_speed_m_s"] =
        static_cast<double>(rope.maximumSpeed);
    result["rope_solver_iterations"] =
        static_cast<std::uint64_t>(rope.solverIterations);
    result["rope_self_collision"] = false;
    result["rope_first_attached"] = false;
    result["rope_last_attached"] = false;
    result["rope_centerline_m"] = std::move(centerline);
    result["contact_distance_m"] =
        static_cast<double>(edge.parameters.mechanical.contactDistance);
    result["friction"] =
        static_cast<double>(edge.parameters.mechanical.friction);
    result["maximum_rope_acceleration_m_s2"] =
        static_cast<double>(edge.parameters.mechanical.maximumAcceleration);
    result["two_way"] = true;
    result["initial_particles"] = std::move(initial_particles);
    result["oracle"] =
        "public-state-segment-clearance-reciprocal-impulse-distribution-and-unattached-endpoints";
    return result;
}

} // namespace

std::string conformance_fluid_rope_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    if (status != nullptr) *status = ExecutionStatus::failed;
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.rope = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    if (!context.open(needed)) {
        if (status != nullptr) *status = ExecutionStatus::unsupported;
        return {};
    }

    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    if (!supports(context.physical)) {
        if (status != nullptr) *status = ExecutionStatus::unsupported;
        return {};
    }

    PmParticleState particle{};
    particle.position[0] = -0.09F;
    particle.position[1] = 0.025F;
    particle.velocity[0] = 1.0F;
    particle.temperatureCelsius = 20.0F;
    const auto input = context.buffer(sizeof(particle),
                                      PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(input.mapped, &particle, sizeof(particle));
    context.flush(input);

    const PmFluidCapacity fluid_capacity{1U, 16U};
    const PmRopeCapacity rope_capacity{
        static_cast<std::uint32_t>(rope_centerline.size())};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = 1U;
    create.pFluidCapacities = &fluid_capacity;
    create.ropeCount = 1U;
    create.pRopeCapacities = &rope_capacity;
    create.couplingCapacity = 1U;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &create, &simulation) ==
             PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding memory{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &memory) ==
             PM_SUCCESS);

    const auto fluid = fluid_material();
    const auto rope = rope_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &fluid;
    pipeline_info.ropeMaterialCount = 1U;
    pipeline_info.pRopeMaterials = &rope;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const PmBufferView particles{
        {input.handle, 0U, sizeof(particle)}, PM_FORMAT_PARTICLE_STATE,
        1U, sizeof(PmParticleState)};
    PmRopeInfo rope_info{};
    rope_info.active = PM_TRUE;
    rope_info.materialIndex = 0U;
    rope_info.pointCount = static_cast<std::uint32_t>(rope_centerline.size());
    rope_info.pCenterline = rope_centerline.data();
    const auto edge = contact_edge();

    const PmDeviceSize fluid_bytes = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    const PmDeviceSize rope_bytes = sizeof(PmDomainOutputHeader) +
        rope_centerline.size() * sizeof(PmRopeNodeState);
    const PmDeviceSize initial_fluid_offset = 0U;
    const PmDeviceSize initial_rope_offset = fluid_bytes;
    const PmDeviceSize final_fluid_offset = fluid_bytes + rope_bytes;
    const PmDeviceSize final_rope_offset = final_fluid_offset + fluid_bytes;
    const auto output = context.buffer(final_rope_offset + rope_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange initial_fluid{
        output.handle, initial_fluid_offset, fluid_bytes};
    const PmBufferRange initial_rope{
        output.handle, initial_rope_offset, rope_bytes};
    const PmBufferRange final_fluid{
        output.handle, final_fluid_offset, fluid_bytes};
    const PmBufferRange final_rope{
        output.handle, final_rope_offset, rope_bytes};

    const auto submit_stage = [&](const char *label, auto record) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) ==
                 PM_SUCCESS);
        record();
        const PmResult submitted = context.submit();
        if (submitted != PM_SUCCESS)
            throw std::runtime_error(std::string("fluid-rope ") + label +
                                     " failed: " +
                                     std::to_string(submitted));
    };
    submit_stage("fluid initialization", [&] {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                        &particles) == PM_SUCCESS);
    });
    submit_stage("rope initialization", [&] {
        PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0U, 1U,
                                 &rope_info) == PM_SUCCESS);
    });
    submit_stage("coupling initialization", [&] {
        PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &edge) ==
                 PM_SUCCESS);
    });
    submit_stage("initial capture", [&] {
        PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                         &initial_fluid) == PM_SUCCESS);
        PM_CHECK(pmCmdCopyRopeState(context.commands, simulation, 0U,
                                    &initial_rope) == PM_SUCCESS);
    });

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                     &final_fluid) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRopeState(context.commands, simulation, 0U,
                                &final_rope) == PM_SUCCESS);
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("fluid-rope contact submission failed: " +
                                 std::to_string(submitted));
    context.invalidate(output);

    const auto header = [&](PmDeviceSize offset) {
        return reinterpret_cast<const PmDomainOutputHeader *>(
            static_cast<const std::byte *>(output.mapped) + offset);
    };
    const auto *initial_fluid_header = header(initial_fluid_offset);
    const auto *initial_rope_header = header(initial_rope_offset);
    const auto *final_fluid_header = header(final_fluid_offset);
    const auto *final_rope_header = header(final_rope_offset);
    if (initial_fluid_header->activeCount != 1U ||
        final_fluid_header->activeCount != 1U ||
        initial_rope_header->activeCount != rope_centerline.size() ||
        final_rope_header->activeCount != rope_centerline.size())
        throw std::runtime_error("fluid-rope public output count mismatch");

    const auto *initial_fluid_state =
        reinterpret_cast<const PmFluidParticleOutput *>(
            initial_fluid_header + 1);
    const auto *final_fluid_state =
        reinterpret_cast<const PmFluidParticleOutput *>(final_fluid_header + 1);
    const auto *initial_rope_state =
        reinterpret_cast<const PmRopeNodeState *>(initial_rope_header + 1);
    const auto *final_rope_state =
        reinterpret_cast<const PmRopeNodeState *>(final_rope_header + 1);

    Json initial_particles = Json::array();
    initial_particles.push_back(fluid_state_json(initial_fluid_state[0]));
    Json final_particles = Json::array();
    final_particles.push_back(fluid_state_json(final_fluid_state[0]));
    Json initial_nodes = Json::array();
    Json final_nodes = Json::array();
    for (std::uint32_t index = 0U; index < rope_centerline.size(); ++index) {
        initial_nodes.push_back(rope_state_json(initial_rope_state[index], index));
        final_nodes.push_back(rope_state_json(final_rope_state[index], index));
    }

    Json fixture = settings_json(particle);
    fixture["initial_fluid_state"] = std::move(initial_particles);
    fixture["final_fluid_state"] = std::move(final_particles);
    fixture["initial_rope_node_state"] = std::move(initial_nodes);
    fixture["final_rope_node_state"] = std::move(final_nodes);
    Json driver = Json::object();
    driver["name"] = driver_name(context.device);
    driver["device"] = properties.deviceName;
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = "fluid-rope-contact-v1";
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/fluid-rope-contact-v1.json");
    result["execution_status"] = "completed";
    result["driver"] = std::move(driver);
    result["fixture"] = std::move(fixture);

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    if (status != nullptr) *status = ExecutionStatus::completed;
    return result.serialize();
}

} // namespace parallel_mater::conformance
