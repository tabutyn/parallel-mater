// SPDX-License-Identifier: MIT
// Public-SDK fluid/soft-body contact quality fixture. No driver-private state.
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

constexpr std::array<PmFloat3, 4> soft_nodes{{
    {0.0F, 0.0F, 0.0F},
    {1.0F, 0.0F, 0.0F},
    {0.0F, 1.0F, 0.0F},
    {0.0F, 0.0F, 1.0F},
}};
constexpr std::array<float, 4> soft_inverse_masses{{
    0.5F, 0.5F, 0.5F, 0.5F}};
constexpr std::array<PmSoftBodyBond, 6> soft_bonds{{
    {0U, 1U, 1.0F},
    {0U, 2U, 1.0F},
    {0U, 3U, 1.0F},
    {1U, 2U, 1.41421356237F},
    {1U, 3U, 1.41421356237F},
    {2U, 3U, 1.41421356237F},
}};
constexpr std::array<PmFloat3, 10> surface_vertices{{
    soft_nodes[0], soft_nodes[1], soft_nodes[2], soft_nodes[3],
    {0.5F, 0.0F, 0.0F}, {0.0F, 0.5F, 0.0F},
    {0.0F, 0.0F, 0.5F}, {0.5F, 0.5F, 0.0F},
    {0.5F, 0.0F, 0.5F}, {0.0F, 0.5F, 0.5F},
}};
constexpr std::array<std::uint32_t, 48> surface_indices{{
    0U, 5U, 4U, 5U, 2U, 7U, 4U, 7U, 1U, 5U, 7U, 4U,
    0U, 4U, 6U, 4U, 1U, 8U, 6U, 8U, 3U, 4U, 8U, 6U,
    0U, 6U, 5U, 6U, 3U, 9U, 5U, 9U, 2U, 6U, 9U, 5U,
    1U, 7U, 8U, 7U, 2U, 9U, 8U, 9U, 3U, 7U, 9U, 8U,
}};

[[nodiscard]] std::array<PmSoftBodySurfaceBinding, 10>
surface_bindings() {
    std::array<PmSoftBodySurfaceBinding, 10> result{};
    for (std::uint32_t index = 0U; index < 4U; ++index) {
        result[index].nodes[0] = index;
        result[index].weights[0] = 1.0F;
    }
    constexpr std::array<std::array<std::uint32_t, 2>, 6> edges{{
        {{0U, 1U}}, {{0U, 2U}}, {{0U, 3U}},
        {{1U, 2U}}, {{1U, 3U}}, {{2U, 3U}},
    }};
    for (std::uint32_t index = 0U; index < edges.size(); ++index) {
        auto &binding = result[index + 4U];
        binding.nodes[0] = edges[index][0];
        binding.nodes[1] = edges[index][1];
        binding.weights[0] = binding.weights[1] = 0.5F;
    }
    return result;
}

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
        PM_INTERACTION_FAMILY_SOFT_BODY,
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
    material.maximumSpeed = 10.0F;
    material.maximumPairAcceleration = 1000.0F;
    material.densityIterations = 1U;
    return material;
}

[[nodiscard]] PmSoftBodyMaterialState soft_material() {
    PmSoftBodyMaterialState material{};
    material.nodeMass = 2.0F;
    material.nodeRadius = 0.05F;
    material.stretchCompliance = 1.0F;
    material.maximumProjectionFraction = 0.2F;
    material.maximumSpeed = 10.0F;
    material.solverIterations = 1U;
    return material;
}

[[nodiscard]] PmCouplingState contact_edge() {
    PmCouplingState edge{};
    edge.type = PM_COUPLING_LIQUID_SOFT_BODY;
    edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    edge.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    edge.parameters.mechanical.contactDistance = 0.1F;
    edge.parameters.mechanical.friction = 0.0F;
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

[[nodiscard]] Json soft_state_json(const PmSoftBodyNodeState &node,
                                   std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(node.position);
    result["velocity_m_s"] = vector_json(node.velocity);
    result["liquid_contact_force_n"] = vector_json(node.liquidContactForce);
    return result;
}

[[nodiscard]] Json settings_json(const PmParticleState &particle,
                                 const std::array<PmSoftBodySurfaceBinding, 10>
                                     &bindings) {
    const auto fluid = fluid_material();
    const auto soft = soft_material();
    const auto edge = contact_edge();
    Json nodes = Json::array();
    for (const auto node : soft_nodes) nodes.push_back(vector_json(node));
    Json inverse_masses = Json::array();
    for (const auto mass : soft_inverse_masses)
        inverse_masses.push_back(static_cast<double>(mass));
    Json bonds = Json::array();
    for (const auto &bond : soft_bonds) {
        Json item = Json::object();
        item["first"] = static_cast<std::uint64_t>(bond.first);
        item["second"] = static_cast<std::uint64_t>(bond.second);
        item["rest_length_m"] = static_cast<double>(bond.restLength);
        bonds.push_back(std::move(item));
    }
    Json vertices = Json::array();
    for (const auto vertex : surface_vertices)
        vertices.push_back(vector_json(vertex));
    Json indices = Json::array();
    for (const auto index : surface_indices)
        indices.push_back(static_cast<std::uint64_t>(index));
    Json bindings_json = Json::array();
    for (const auto &binding : bindings) {
        Json item = Json::object();
        Json binding_nodes = Json::array();
        Json weights = Json::array();
        for (std::size_t slot = 0U; slot < 4U; ++slot) {
            binding_nodes.push_back(
                static_cast<std::uint64_t>(binding.nodes[slot]));
            weights.push_back(static_cast<double>(binding.weights[slot]));
        }
        item["nodes"] = std::move(binding_nodes);
        item["weights"] = std::move(weights);
        bindings_json.push_back(std::move(item));
    }
    Json initial_particles = Json::array();
    Json initial = Json::object();
    initial["stable_id"] = static_cast<std::uint64_t>(0U);
    initial["position_m"] = vector_json(particle.position);
    initial["velocity_m_s"] = vector_json(particle.velocity);
    initial_particles.push_back(std::move(initial));
    Json expected_weights = Json::array();
    for (const float weight : {0.6F, 0.0F, 0.2F, 0.2F})
        expected_weights.push_back(static_cast<double>(weight));

    Json result = Json::object();
    result["id"] = "subdivided-tetrahedron-normal-impact";
    result["timestep_seconds"] = 0.01;
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
    result["soft_node_mass_kg"] = static_cast<double>(soft.nodeMass);
    result["soft_node_radius_m"] = static_cast<double>(soft.nodeRadius);
    result["soft_stretch_compliance_m_per_n"] =
        static_cast<double>(soft.stretchCompliance);
    result["soft_velocity_damping"] =
        static_cast<double>(soft.velocityDamping);
    result["soft_spring_damping"] = static_cast<double>(soft.springDamping);
    result["soft_contact_friction"] =
        static_cast<double>(soft.contactFriction);
    result["soft_shape_matching_stiffness"] =
        static_cast<double>(soft.shapeMatchingStiffness);
    result["soft_maximum_projection_fraction"] =
        static_cast<double>(soft.maximumProjectionFraction);
    result["soft_constraint_velocity_response"] =
        static_cast<double>(soft.constraintVelocityResponse);
    result["soft_maximum_speed_m_s"] =
        static_cast<double>(soft.maximumSpeed);
    result["soft_solver_iterations"] =
        static_cast<std::uint64_t>(soft.solverIterations);
    result["coupling_solver_iterations"] = static_cast<std::uint64_t>(4U);
    result["contact_distance_m"] =
        static_cast<double>(edge.parameters.mechanical.contactDistance);
    result["friction"] =
        static_cast<double>(edge.parameters.mechanical.friction);
    result["two_way"] = true;
    result["soft_nodes_m"] = std::move(nodes);
    result["soft_inverse_masses_kg_inverse"] = std::move(inverse_masses);
    result["soft_bonds"] = std::move(bonds);
    result["surface_vertices_m"] = std::move(vertices);
    result["surface_triangle_indices"] = std::move(indices);
    result["surface_bindings"] = std::move(bindings_json);
    result["minimum_binding_incidence_per_node"] =
        static_cast<std::uint64_t>(4U);
    result["expected_contact_node_weights"] = std::move(expected_weights);
    result["initial_particles"] = std::move(initial_particles);
    result["oracle"] =
        "public-state-convex-clearance-reciprocal-impulse-and-unique-bound-node-effective-mass";
    return result;
}

} // namespace

std::string conformance_fluid_soft_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    if (status != nullptr) *status = ExecutionStatus::failed;
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.softBodies = PM_TRUE;
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
    particle.position[0] = -0.1F;
    particle.position[1] = 0.2F;
    particle.position[2] = 0.2F;
    particle.velocity[0] = 1.0F;
    particle.temperatureCelsius = 20.0F;
    const auto input = context.buffer(sizeof(particle), PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(input.mapped, &particle, sizeof(particle));
    context.flush(input);

    const PmFluidCapacity fluid_capacity{1U, 16U};
    const PmSoftBodyCapacity soft_capacity{
        static_cast<std::uint32_t>(soft_nodes.size()),
        static_cast<std::uint32_t>(soft_bonds.size()),
        static_cast<std::uint32_t>(surface_vertices.size()),
        static_cast<std::uint32_t>(surface_indices.size())};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = 1U;
    create.pFluidCapacities = &fluid_capacity;
    create.softBodyCount = 1U;
    create.pSoftBodyCapacities = &soft_capacity;
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
    const auto soft = soft_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &fluid;
    pipeline_info.softBodyMaterialCount = 1U;
    pipeline_info.pSoftBodyMaterials = &soft;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const PmBufferView particles{
        {input.handle, 0U, sizeof(particle)}, PM_FORMAT_PARTICLE_STATE,
        1U, sizeof(PmParticleState)};
    const auto bindings = surface_bindings();
    const PmSoftBodyInfo body{
        PM_TRUE, 0U, static_cast<std::uint32_t>(soft_nodes.size()),
        soft_nodes.data(), static_cast<std::uint32_t>(soft_bonds.size()),
        soft_bonds.data(), soft_inverse_masses.data(),
        static_cast<std::uint32_t>(surface_vertices.size()),
        surface_vertices.data(),
        static_cast<std::uint32_t>(surface_indices.size()),
        surface_indices.data(), bindings.data()};
    const auto edge = contact_edge();

    const PmDeviceSize fluid_bytes = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    const PmDeviceSize soft_bytes = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState);
    const PmDeviceSize surface_bytes = sizeof(PmDomainOutputHeader) +
        surface_vertices.size() * sizeof(PmFloat3);
    const PmDeviceSize initial_fluid_offset = 0U;
    const PmDeviceSize initial_soft_offset = fluid_bytes;
    const PmDeviceSize initial_surface_offset = fluid_bytes + soft_bytes;
    const PmDeviceSize final_fluid_offset = fluid_bytes + soft_bytes + surface_bytes;
    const PmDeviceSize final_soft_offset = final_fluid_offset + fluid_bytes;
    const PmDeviceSize final_surface_offset = final_soft_offset + soft_bytes;
    const auto output = context.buffer(final_surface_offset + surface_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange initial_fluid{
        output.handle, initial_fluid_offset, fluid_bytes};
    const PmBufferRange initial_soft{
        output.handle, initial_soft_offset, soft_bytes};
    const PmBufferRange initial_surface{
        output.handle, initial_surface_offset, surface_bytes};
    const PmBufferRange final_fluid{
        output.handle, final_fluid_offset, fluid_bytes};
    const PmBufferRange final_soft{
        output.handle, final_soft_offset, soft_bytes};
    const PmBufferRange final_surface{
        output.handle, final_surface_offset, surface_bytes};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &particles) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0U, 1U,
                                  &body) == PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &edge) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                     &initial_fluid) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodyState(context.commands, simulation, 0U,
                                    &initial_soft) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodySurface(context.commands, simulation, 0U,
                                      &initial_surface) == PM_SUCCESS);
    const PmStepInfo step{0.01F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                     &final_fluid) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodyState(context.commands, simulation, 0U,
                                    &final_soft) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodySurface(context.commands, simulation, 0U,
                                      &final_surface) == PM_SUCCESS);
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("fluid-soft contact submission failed: " +
                                 std::to_string(submitted));
    context.invalidate(output);

    const auto header = [&](PmDeviceSize offset) {
        return reinterpret_cast<const PmDomainOutputHeader *>(
            static_cast<const std::byte *>(output.mapped) + offset);
    };
    const auto *initial_fluid_header = header(initial_fluid_offset);
    const auto *initial_soft_header = header(initial_soft_offset);
    const auto *initial_surface_header = header(initial_surface_offset);
    const auto *final_fluid_header = header(final_fluid_offset);
    const auto *final_soft_header = header(final_soft_offset);
    const auto *final_surface_header = header(final_surface_offset);
    if (initial_fluid_header->activeCount != 1U ||
        final_fluid_header->activeCount != 1U ||
        initial_soft_header->activeCount != soft_nodes.size() ||
        final_soft_header->activeCount != soft_nodes.size() ||
        initial_surface_header->activeCount != surface_vertices.size() ||
        final_surface_header->activeCount != surface_vertices.size())
        throw std::runtime_error("fluid-soft public output count mismatch");

    const auto *initial_fluid_state =
        reinterpret_cast<const PmFluidParticleOutput *>(
            initial_fluid_header + 1);
    const auto *final_fluid_state =
        reinterpret_cast<const PmFluidParticleOutput *>(final_fluid_header + 1);
    const auto *initial_soft_state =
        reinterpret_cast<const PmSoftBodyNodeState *>(initial_soft_header + 1);
    const auto *final_soft_state =
        reinterpret_cast<const PmSoftBodyNodeState *>(final_soft_header + 1);
    const auto *initial_surface_state =
        reinterpret_cast<const PmFloat3 *>(initial_surface_header + 1);
    const auto *final_surface_state =
        reinterpret_cast<const PmFloat3 *>(final_surface_header + 1);

    Json initial_particles = Json::array();
    initial_particles.push_back(fluid_state_json(initial_fluid_state[0]));
    Json final_particles = Json::array();
    final_particles.push_back(fluid_state_json(final_fluid_state[0]));
    Json initial_nodes = Json::array();
    Json final_nodes = Json::array();
    for (std::uint32_t index = 0U; index < soft_nodes.size(); ++index) {
        initial_nodes.push_back(soft_state_json(initial_soft_state[index], index));
        final_nodes.push_back(soft_state_json(final_soft_state[index], index));
    }
    Json initial_skin = Json::array();
    Json final_skin = Json::array();
    for (std::uint32_t index = 0U; index < surface_vertices.size(); ++index) {
        initial_skin.push_back(vector_json(initial_surface_state[index]));
        final_skin.push_back(vector_json(final_surface_state[index]));
    }

    Json fixture = settings_json(particle, bindings);
    fixture["initial_fluid_state"] = std::move(initial_particles);
    fixture["final_fluid_state"] = std::move(final_particles);
    fixture["initial_soft_node_state"] = std::move(initial_nodes);
    fixture["final_soft_node_state"] = std::move(final_nodes);
    fixture["initial_surface_state_m"] = std::move(initial_skin);
    fixture["final_surface_state_m"] = std::move(final_skin);
    Json driver = Json::object();
    driver["name"] = driver_name(context.device);
    driver["device"] = properties.deviceName;
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = "fluid-soft-contact-v1";
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/fluid-soft-contact-v1.json");
    result["execution_status"] = "completed";
    result["driver"] = std::move(driver);
    result["fixture"] = std::move(fixture);

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    if (status != nullptr) *status = ExecutionStatus::completed;
    return result.serialize();
}

} // namespace parallel_mater::conformance
