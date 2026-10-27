// SPDX-License-Identifier: MIT
#include "lifecycle_fixture.hpp"
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using parallel_mater::conformance::Json;
using parallel_mater::conformance::sha256;
using parallel_mater::conformance::sha256_contract_file;
using parallel_mater::test::Context;
using parallel_mater::test::GraphLifecycleResult;
using parallel_mater::test::LifecycleCheckpointObservation;
using parallel_mater::test::LifecycleResourceObservation;

namespace {

constexpr std::array<PmFloat3, 4> tetra_vertices{{
    {0.0F, 0.0F, 0.0F},
    {0.2F, 0.0F, 0.0F},
    {0.0F, 0.2F, 0.0F},
    {0.0F, 0.0F, 0.2F},
}};
constexpr std::array<std::uint32_t, 12> tetra_indices{{
    0U, 2U, 1U,
    0U, 1U, 3U,
    0U, 3U, 2U,
    1U, 2U, 3U,
}};
constexpr std::array<float, 4> pinned_masses{{0.0F, 0.0F, 0.0F, 0.0F}};
constexpr std::array<float, 4> movable_masses{{1.0F, 1.0F, 1.0F, 1.0F}};

PmRigidBodyInfo rigid_body(PmGeometry geometry) {
    PmRigidBodyInfo body{};
    body.active = PM_TRUE;
    body.motionType = PM_MOTION_STATIC;
    body.geometry = geometry;
    body.state.position[0] = 20.0F;
    body.state.orientation[3] = 1.0F;
    body.friction = 0.5F;
    body.maximumLinearSpeed = 100.0F;
    body.maximumAngularSpeed = 100.0F;
    body.collisionMargin = 0.005F;
    return body;
}

std::array<PmCouplingState, 12> coupling_graph(bool updated) {
    std::array<PmCouplingState, 12> edges{};
    std::size_t index = 0U;
    const float delta = updated ? 0.002F : 0.0F;

    auto &liquid_rigid = edges[index++];
    liquid_rigid.type = PM_COUPLING_LIQUID_RIGID;
    liquid_rigid.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_CONTAINMENT_BIT;
    liquid_rigid.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    liquid_rigid.parameters.mechanical.contactDistance = 0.01F;
    liquid_rigid.parameters.mechanical.friction = updated ? 0.3F : 0.2F;
    liquid_rigid.parameters.mechanical.maximumAcceleration = 10.0F;
    liquid_rigid.parameters.mechanical.twoWay = PM_TRUE;

    auto &liquid_cloth = edges[index++];
    liquid_cloth.type = PM_COUPLING_LIQUID_CLOTH;
    liquid_cloth.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_CONTAINMENT_BIT;
    liquid_cloth.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    liquid_cloth.parameters.mechanical.contactDistance = 0.01F + delta;
    liquid_cloth.parameters.mechanical.stiffness = updated ? 120.0F : 100.0F;
    liquid_cloth.parameters.mechanical.damping = 1.0F;
    liquid_cloth.parameters.mechanical.tangentialDrag = 0.2F;
    liquid_cloth.parameters.mechanical.maximumForce = 10.0F;
    liquid_cloth.parameters.mechanical.twoWay = PM_TRUE;

    auto &liquid_rope = edges[index++];
    liquid_rope.type = PM_COUPLING_LIQUID_ROPE;
    liquid_rope.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    liquid_rope.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    liquid_rope.parameters.mechanical.contactDistance = 0.01F + delta;
    liquid_rope.parameters.mechanical.friction = 0.2F;
    liquid_rope.parameters.mechanical.maximumAcceleration =
        updated ? 12.0F : 10.0F;
    liquid_rope.parameters.mechanical.twoWay = PM_TRUE;

    auto &liquid_soft = edges[index++];
    liquid_soft.type = PM_COUPLING_LIQUID_SOFT_BODY;
    liquid_soft.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    liquid_soft.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    liquid_soft.parameters.mechanical.contactDistance = 0.01F + delta;
    liquid_soft.parameters.mechanical.friction = updated ? 0.25F : 0.2F;
    liquid_soft.parameters.mechanical.twoWay = PM_TRUE;

    auto &liquid_smoke = edges[index++];
    liquid_smoke.type = PM_COUPLING_LIQUID_SMOKE;
    liquid_smoke.modeFlags = PM_INTERACTION_MODE_DRAG_BIT |
        PM_INTERACTION_MODE_THERMAL_BIT | PM_INTERACTION_MODE_PHASE_BIT;
    liquid_smoke.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    liquid_smoke.parameters.thermal.heaterOrientation[3] = 1.0F;
    liquid_smoke.parameters.thermal.heaterHalfExtents[0] = 0.1F;
    liquid_smoke.parameters.thermal.heaterHalfExtents[1] = 0.1F;
    liquid_smoke.parameters.thermal.heaterTemperature = 120.0F;
    liquid_smoke.parameters.thermal.boilingTemperature = 100.0F;
    liquid_smoke.parameters.thermal.heatTransferRate = updated ? 0.3F : 0.2F;
    liquid_smoke.parameters.thermal.windDrag = 0.1F;
    liquid_smoke.parameters.thermal.steamRiseSpeed = 1.0F;

    auto &soft_cloth = edges[index++];
    soft_cloth.type = PM_COUPLING_SOFT_BODY_CLOTH;
    soft_cloth.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    soft_cloth.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    soft_cloth.parameters.mechanical.contactDistance = 0.01F + delta;
    soft_cloth.parameters.mechanical.friction = updated ? 0.3F : 0.2F;
    soft_cloth.parameters.mechanical.twoWay = PM_TRUE;

    auto &rope_soft = edges[index++];
    rope_soft.type = PM_COUPLING_ROPE_SOFT_BODY;
    rope_soft.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_ATTACHMENT_BIT;
    rope_soft.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    rope_soft.parameters.mechanical.contactDistance = 0.01F + delta;
    rope_soft.parameters.mechanical.friction = 0.2F;
    rope_soft.parameters.attachment.maximumAcceleration = 10.0F;
    rope_soft.parameters.attachment.supportRadiusScale = updated ? 3.0F : 2.0F;
    rope_soft.parameters.attachment.contactSupportRadiusScale = 1.0F;

    auto &rope_cloth = edges[index++];
    rope_cloth.type = PM_COUPLING_ROPE_CLOTH;
    rope_cloth.modeFlags = PM_INTERACTION_MODE_ATTACHMENT_BIT;
    rope_cloth.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    rope_cloth.parameters.attachment.firstVertex = 0U;
    rope_cloth.parameters.attachment.lastVertex =
        std::numeric_limits<std::uint32_t>::max();
    rope_cloth.parameters.attachment.effectiveMass = updated ? 0.2F : 0.1F;
    rope_cloth.parameters.attachment.maximumAcceleration = 10.0F;

    auto smoke_edge = [&](PmCouplingState &edge, PmCouplingType type,
                          float wind_drag) {
        edge.type = type;
        edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_DRAG_BIT;
        edge.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
        edge.parameters.smoke.windDrag = wind_drag;
        edge.parameters.smoke.maximumWindAcceleration = 10.0F;
        edge.parameters.smoke.airDensity = 1.2F;
        edge.parameters.smoke.dragCoefficient = updated ? 1.1F : 1.0F;
        edge.parameters.smoke.contactDistance = 0.01F + delta;
        edge.parameters.smoke.tracerContact = PM_TRUE;
    };
    smoke_edge(edges[index++], PM_COUPLING_SMOKE_RIGID,
               updated ? 0.3F : 0.2F);
    smoke_edge(edges[index++], PM_COUPLING_SMOKE_CLOTH,
               updated ? 0.4F : 0.3F);
    smoke_edge(edges[index++], PM_COUPLING_SMOKE_ROPE,
               updated ? 0.5F : 0.4F);
    smoke_edge(edges[index++], PM_COUPLING_SMOKE_SOFT_BODY,
               updated ? 0.6F : 0.5F);
    PM_CHECK(index == edges.size());
    return edges;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_state_float(std::ostringstream &stream, float value) {
    if (!std::isfinite(value)) {
        stream << (std::isnan(value) ? "nan," : value < 0.0F ? "-inf," : "inf,");
        return;
    }
    stream << std::dec << std::llround(static_cast<double>(value) * 10000.0)
           << ',';
}

void append_state_vec3(std::ostringstream &stream, const PmFloat3 &value) {
    append_state_float(stream, value.x);
    append_state_float(stream, value.y);
    append_state_float(stream, value.z);
}

template <typename State, typename AppendState>
std::string domain_fingerprint(std::string_view family,
                               const PmDomainOutputHeader *header,
                               AppendState append_state) {
    std::ostringstream canonical;
    canonical << "parallel-mater-domain-public-state/v1;family=" << family
              << ";count=" << header->activeCount
              << ";capacity=" << header->capacity;
    const auto *states = reinterpret_cast<const State *>(header + 1);
    for (std::uint32_t index = 0U; index < header->activeCount; ++index) {
        canonical << ";state=" << index << ',';
        append_state(canonical, states[index]);
    }
    return sha256(canonical.str());
}

std::string fluid_fingerprint(const PmDomainOutputHeader *header) {
    return domain_fingerprint<PmFluidParticleOutput>(
        "liquid", header, [](std::ostringstream &stream,
                             const PmFluidParticleOutput &state) {
            for (float value : state.position) append_state_float(stream, value);
            stream << std::dec << state.stableId;
        });
}

std::string cloth_fingerprint(const PmDomainOutputHeader *header) {
    return domain_fingerprint<PmClothVertexState>(
        "cloth", header, [](std::ostringstream &stream,
                            const PmClothVertexState &state) {
            append_state_vec3(stream, state.position);
        });
}

std::string rope_fingerprint(const PmDomainOutputHeader *header) {
    return domain_fingerprint<PmRopeNodeState>(
        "rope", header, [](std::ostringstream &stream,
                           const PmRopeNodeState &state) {
            append_state_vec3(stream, state.position);
        });
}

std::string soft_body_fingerprint(const PmDomainOutputHeader *header) {
    return domain_fingerprint<PmSoftBodyNodeState>(
        "soft-body", header, [](std::ostringstream &stream,
                                const PmSoftBodyNodeState &state) {
            append_state_vec3(stream, state.position);
        });
}

std::string smoke_fingerprint(const PmDomainOutputHeader *header) {
    return domain_fingerprint<PmSmokeParticleState>(
        "smoke", header, [](std::ostringstream &stream,
                            const PmSmokeParticleState &state) {
            append_state_vec3(stream, state.position);
        });
}

std::string graph_fingerprint(const PmCouplingState *edges,
                              std::size_t count) {
    std::ostringstream canonical;
    canonical << "parallel-mater-coupling-graph/v1;count=" << count;
    for (std::size_t index = 0U; index < count; ++index) {
        const auto &edge = edges[index];
        canonical << ";edge=" << index << ",type=" << edge.type
                  << ",first=" << edge.first << ",second=" << edge.second
                  << ",modes=" << edge.modeFlags
                  << ",material=" << edge.materialProfile << ",mechanical=";
        for (float value : {edge.parameters.mechanical.contactDistance,
                            edge.parameters.mechanical.stiffness,
                            edge.parameters.mechanical.damping,
                            edge.parameters.mechanical.friction,
                            edge.parameters.mechanical.tangentialDrag,
                            edge.parameters.mechanical.maximumForce,
                            edge.parameters.mechanical.maximumAcceleration})
            append_float(canonical, value);
        canonical << ',' << edge.parameters.mechanical.twoWay << ",smoke=";
        for (float value : {edge.parameters.smoke.windDrag,
                            edge.parameters.smoke.maximumWindAcceleration,
                            edge.parameters.smoke.airDensity,
                            edge.parameters.smoke.dragCoefficient,
                            edge.parameters.smoke.contactDistance})
            append_float(canonical, value);
        canonical << ',' << edge.parameters.smoke.tracerContact
                  << ",attachment=" << edge.parameters.attachment.firstVertex
                  << ',' << edge.parameters.attachment.lastVertex << ',';
        for (float value : {edge.parameters.attachment.effectiveMass,
                            edge.parameters.attachment.supportRadiusScale,
                            edge.parameters.attachment.contactSupportRadiusScale,
                            edge.parameters.attachment.maximumAcceleration})
            append_float(canonical, value);
        canonical << ',' << edge.parameters.attachment.attachFirst << ','
                  << edge.parameters.attachment.attachLast << ",thermal=";
        for (float value : edge.parameters.thermal.heaterCenter)
            append_float(canonical, value);
        for (float value : edge.parameters.thermal.heaterOrientation)
            append_float(canonical, value);
        for (float value : edge.parameters.thermal.heaterHalfExtents)
            append_float(canonical, value);
        for (float value : {edge.parameters.thermal.heaterTemperature,
                            edge.parameters.thermal.boilingTemperature,
                            edge.parameters.thermal.heatTransferRate,
                            edge.parameters.thermal.windDrag,
                            edge.parameters.thermal.steamRiseSpeed})
            append_float(canonical, value);
    }
    return sha256(canonical.str());
}

const PmDomainOutputHeader *output_header(
    const parallel_mater::test::Buffer &buffer, PmDeviceSize offset) {
    return reinterpret_cast<const PmDomainOutputHeader *>(
        static_cast<const std::byte *>(buffer.mapped) + offset);
}

void record_checkpoint(GraphLifecycleResult *result, std::string id,
                       std::string outcome, std::string graph_sha256,
                       const std::array<const PmDomainOutputHeader *, 5> &state,
                       std::uint64_t liquid_generation) {
    if (result == nullptr) return;
    LifecycleCheckpointObservation checkpoint;
    checkpoint.id = std::move(id);
    checkpoint.outcome = std::move(outcome);
    checkpoint.active_graph_sha256 = std::move(graph_sha256);
    checkpoint.resources = {
        LifecycleResourceObservation{
            "liquid", true, liquid_generation, fluid_fingerprint(state[0])},
        LifecycleResourceObservation{
            "rope", true, 1U, rope_fingerprint(state[1])},
        LifecycleResourceObservation{
            "soft-body", true, 1U, soft_body_fingerprint(state[2])},
        LifecycleResourceObservation{
            "smoke", true, 1U, smoke_fingerprint(state[3])},
        LifecycleResourceObservation{
            "cloth", true, 1U, cloth_fingerprint(state[4])},
    };
    result->observations.push_back(std::move(checkpoint));
}

#ifndef PM_CONFORMANCE_EMBEDDED

Json resource_json(const LifecycleResourceObservation &resource) {
    Json value = Json::object();
    value["id"] = resource.id;
    value["active"] = resource.active;
    value["generation"] = resource.generation;
    value["state_sha256"] = resource.state_sha256;
    return value;
}

std::string lifecycle_result_json(const GraphLifecycleResult &fixture,
                                  std::string_view driver_name,
                                  const std::filesystem::path &profile) {
    Json observations = Json::array();
    for (const auto &checkpoint : fixture.observations) {
        Json resources = Json::array();
        for (const auto &resource : checkpoint.resources)
            resources.push_back(resource_json(resource));
        Json value = Json::object();
        value["id"] = checkpoint.id;
        value["outcome"] = checkpoint.outcome;
        value["active_graph_sha256"] = checkpoint.active_graph_sha256;
        value["resources"] = std::move(resources);
        observations.push_back(std::move(value));
    }
    Json families = Json::array();
    for (const char *family : {"liquid", "rope", "soft-body"})
        families.push_back(family);
    Json unrelated = Json::array();
    for (const char *family : {"smoke", "cloth"}) unrelated.push_back(family);
    Json graph = Json::object();
    graph["id"] = "three-family-partial-rollback";
    graph["execution_status"] = fixture.status == 0 ? "completed" :
        fixture.status == 77 ? "unsupported" : "failed";
    graph["families"] = std::move(families);
    graph["unrelated_resources"] = std::move(unrelated);
    graph["observations"] = std::move(observations);
    Json cases = Json::array();
    cases.push_back(std::move(graph));
    Json driver = Json::object();
    driver["name"] = std::string(driver_name);
    driver["device"] = fixture.device.empty() ? "unavailable" : fixture.device;
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = "adverse-capacity-three-family-v1";
    result["profile_sha256"] = sha256_contract_file(profile);
    result["execution_status"] = fixture.status == 0 ? "completed" :
        fixture.status == 77 ? "unsupported" : "failed";
    result["driver"] = std::move(driver);
    result["cases"] = std::move(cases);
    return result.serialize();
}

#endif

} // namespace

int parallel_mater_interaction_graph_lifecycle_fixture(
    GraphLifecycleResult *result) {
    GraphLifecycleResult discarded_result;
    if (result == nullptr) result = &discarded_result;
    *result = {};
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.rigidBodies = PM_TRUE;
    needed.particleLiquids = PM_TRUE;
    needed.liquidRigidCoupling = PM_TRUE;
    needed.cloth = PM_TRUE;
    needed.rope = PM_TRUE;
    needed.softBodies = PM_TRUE;
    needed.smoke = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    if (!context.open(needed)) {
        result->status = 77;
        return 77;
    }
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    result->device = properties.deviceName;

    const auto geometry_source = context.buffer(
        sizeof(tetra_vertices) + sizeof(tetra_indices),
        PM_BUFFER_GEOMETRY_INPUT_BIT);
    std::memcpy(geometry_source.mapped, tetra_vertices.data(),
                sizeof(tetra_vertices));
    std::memcpy(static_cast<std::byte *>(geometry_source.mapped) +
                    sizeof(tetra_vertices),
                tetra_indices.data(), sizeof(tetra_indices));
    context.flush(geometry_source);
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {
        {geometry_source.handle, 0U, sizeof(tetra_vertices)},
        PM_FORMAT_R32G32B32_SFLOAT,
        static_cast<std::uint32_t>(tetra_vertices.size()), sizeof(PmFloat3)};
    geometry_info.indices = {
        {geometry_source.handle, sizeof(tetra_vertices), sizeof(tetra_indices)},
        PM_FORMAT_R32_UINT,
        static_cast<std::uint32_t>(tetra_indices.size()), sizeof(std::uint32_t)};
    PmGeometry geometry{};
    PM_CHECK(pmCreateGeometry(context.device, &geometry_info, &geometry) ==
             PM_SUCCESS);
    PmMemoryRequirements geometry_requirements{};
    pmGetGeometryMemoryRequirements(context.device, geometry,
                                    &geometry_requirements);
    PM_CHECK(pmBindGeometryMemory(context.device, geometry,
                                  context.allocate(geometry_requirements), 0U) ==
             PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBuildGeometry(context.commands, geometry) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmFluidCapacity fluid_capacity{4U, 16U};
    const PmClothCapacity cloth_capacity{4U, 12U, 12U};
    const PmRopeCapacity rope_capacity{16U};
    const PmSoftBodyCapacity soft_capacity{4U, 6U, 4U, 12U};
    const PmSmokeCapacity smoke_capacity{4U, 1U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.rigidBodyCapacity = 1U;
    create.contactCapacity = 16U;
    create.fluidCount = create.clothCount = create.ropeCount =
        create.softBodyCount = create.smokeCount = 1U;
    create.pFluidCapacities = &fluid_capacity;
    create.pClothCapacities = &cloth_capacity;
    create.pRopeCapacities = &rope_capacity;
    create.pSoftBodyCapacities = &soft_capacity;
    create.pSmokeCapacities = &smoke_capacity;
    create.couplingCapacity = 12U;
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

    PmLiquidMaterialState fluid_material{};
    fluid_material.restDensity = 1000.0F;
    fluid_material.particleRadius = 0.025F;
    fluid_material.supportRadius = 0.1F;
    fluid_material.maximumSpeed = 100.0F;
    fluid_material.densityIterations = 1U;
    PmClothMaterialState cloth_material{};
    cloth_material.vertexMass = 1.0F;
    cloth_material.thickness = 0.01F;
    cloth_material.solverIterations = 4U;
    cloth_material.fracturePersistenceSubsteps = 1U;
    cloth_material.preserveVolume = PM_TRUE;
    cloth_material.volumeCompliance = 1.0e-7F;
    PmRopeMaterialState rope_material{};
    rope_material.nodeSpacing = 0.02F;
    rope_material.radius = 0.01F;
    rope_material.mass = 1.0F;
    rope_material.maximumSubstepTimestep = 1.0F / 60.0F;
    rope_material.maximumSpeed = 100.0F;
    rope_material.solverIterations = 4U;
    PmSoftBodyMaterialState soft_material{};
    soft_material.nodeMass = 1.0F;
    soft_material.nodeRadius = 0.01F;
    soft_material.maximumProjectionFraction = 0.2F;
    soft_material.constraintVelocityResponse = 1.0F;
    soft_material.maximumSpeed = 100.0F;
    soft_material.solverIterations = 4U;
    PmSmokeMaterialState smoke_material{};
    smoke_material.particleRadius = 0.02F;
    smoke_material.restNumberDensity = 12.0F;
    smoke_material.maximumSpeed = 100.0F;

    const auto graph = coupling_graph(false);
    const auto updated_graph = coupling_graph(true);
    auto pipeline_info = PmPhysicsPipelineCreateInfo{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = pipeline_info.clothMaterialCount =
        pipeline_info.ropeMaterialCount = pipeline_info.softBodyMaterialCount =
            pipeline_info.smokeMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &fluid_material;
    pipeline_info.pClothMaterials = &cloth_material;
    pipeline_info.pRopeMaterials = &rope_material;
    pipeline_info.pSoftBodyMaterials = &soft_material;
    pipeline_info.pSmokeMaterials = &smoke_material;
    PmPhysicsPipeline empty_pipeline{}, graph_pipeline{}, updated_pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &empty_pipeline) == PM_SUCCESS);
    pipeline_info.couplingCount = static_cast<std::uint32_t>(graph.size());
    pipeline_info.pCouplings = graph.data();
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &graph_pipeline) == PM_SUCCESS);
    pipeline_info.pCouplings = updated_graph.data();
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &updated_pipeline) == PM_SUCCESS);

    std::array<PmParticleState, 2> particle_states{};
    particle_states[0].position[0] = particle_states[0].position[1] =
        particle_states[0].position[2] = 10.0F;
    particle_states[1].position[0] = particle_states[1].position[1] =
        particle_states[1].position[2] = 11.0F;
    const auto particle_input = context.buffer(
        sizeof(particle_states), PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(particle_input.mapped, particle_states.data(),
                sizeof(particle_states));
    context.flush(particle_input);
    const PmBufferView particles{
        {particle_input.handle, 0U, sizeof(PmParticleState)},
        PM_FORMAT_PARTICLE_STATE, 1U, sizeof(PmParticleState)};
    const PmBufferView replacement_particles{
        {particle_input.handle, sizeof(PmParticleState), sizeof(PmParticleState)},
        PM_FORMAT_PARTICLE_STATE, 1U, sizeof(PmParticleState)};
    const PmClothInfo cloth{
        PM_TRUE, 0U, static_cast<std::uint32_t>(tetra_vertices.size()),
        tetra_vertices.data(), static_cast<std::uint32_t>(tetra_indices.size()),
        tetra_indices.data(), pinned_masses.data()};
    const PmFloat3 rope_points[2]{{0.0F, 0.0F, 0.0F},
                                  {-0.08F, 0.0F, 0.0F}};
    PmRopeInfo rope{};
    rope.active = PM_TRUE;
    rope.pointCount = 2U;
    rope.pCenterline = rope_points;
    std::array<PmSoftBodyBond, 6> bonds{};
    std::size_t bond_index = 0U;
    for (std::uint32_t first = 0U; first < tetra_vertices.size(); ++first) {
        for (std::uint32_t second = first + 1U;
             second < tetra_vertices.size(); ++second) {
            const auto dx = tetra_vertices[first].x - tetra_vertices[second].x;
            const auto dy = tetra_vertices[first].y - tetra_vertices[second].y;
            const auto dz = tetra_vertices[first].z - tetra_vertices[second].z;
            bonds[bond_index++] = {
                first, second, std::sqrt(dx * dx + dy * dy + dz * dz)};
        }
    }
    std::array<PmSoftBodySurfaceBinding, 4> skin{};
    for (std::uint32_t index = 0U; index < skin.size(); ++index) {
        skin[index].nodes[0] = index;
        skin[index].weights[0] = 1.0F;
    }
    std::array<PmFloat3, 4> soft_vertices = tetra_vertices;
    for (auto &vertex : soft_vertices) vertex.x += 5.0F;
    const PmSoftBodyInfo soft{
        PM_TRUE, 0U, static_cast<std::uint32_t>(soft_vertices.size()),
        soft_vertices.data(), static_cast<std::uint32_t>(bonds.size()),
        bonds.data(), movable_masses.data(),
        static_cast<std::uint32_t>(soft_vertices.size()), soft_vertices.data(),
        static_cast<std::uint32_t>(tetra_indices.size()), tetra_indices.data(),
        skin.data()};
    const PmFloat3 stable_rope_points[2]{{0.0F, 0.0F, 0.0F},
                                         {-0.02F, 0.0F, 0.0F}};
    PmRopeInfo stable_rope = rope;
    stable_rope.pCenterline = stable_rope_points;
    PmSmokeInfo smoke{};
    smoke.active = PM_TRUE;
    smoke.particleCapacity = 4U;
    smoke.emitterCenter = {30.0F, 0.0F, 0.0F};
    smoke.emitterHalfExtents = {0.1F, 0.1F};
    smoke.particlesPerSecond = 1.0F;
    smoke.lifetime = 1.0F;
    smoke.gridMinimum = {29.0F, -1.0F, -1.0F};
    smoke.gridEdgeLength = 2.0F;
    const auto body = rigid_body(geometry);
    const PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, 0.0F, 0.0F}, 0U};

    const auto submit_endpoint = [&](const char *name, auto record) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty_pipeline) ==
                 PM_SUCCESS);
        record();
        const auto result = context.submit();
        if (result != PM_SUCCESS)
            std::fprintf(stderr, "%s endpoint submission: %d\n", name, result);
        PM_CHECK(result == PM_SUCCESS);
    };
    submit_endpoint("reset", [&] {
        PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    });
    submit_endpoint("fluid", [&] {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                        &particles) == PM_SUCCESS);
    });
    submit_endpoint("cloth", [&] {
        PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U, &cloth) ==
                 PM_SUCCESS);
    });
    submit_endpoint("rope", [&] {
        PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0U, 1U, &rope) ==
                 PM_SUCCESS);
    });
    submit_endpoint("soft body", [&] {
        PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0U, 1U,
                                      &soft) == PM_SUCCESS);
    });
    submit_endpoint("smoke", [&] {
        PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U, &smoke) ==
                 PM_SUCCESS);
    });
    submit_endpoint("rigid body", [&] {
        PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                       &body) == PM_SUCCESS);
    });

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, graph_pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    const auto initial_result = context.submit();
    if (initial_result != PM_SUCCESS)
        std::fprintf(stderr, "initial graph submission: %d\n", initial_result);
    PM_CHECK(initial_result == PM_SUCCESS);

    const auto submit_pipeline = [&](PmPhysicsPipeline pipeline) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    };
    // Pipeline policy creates, updates, clears and recreates every edge.
    submit_pipeline(updated_pipeline);
    submit_pipeline(empty_pipeline);
    submit_pipeline(graph_pipeline);

    const auto submit_graph = [&](const PmCouplingState *edges,
                                  std::uint32_t count,
                                  PmResult expected = PM_SUCCESS) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty_pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, simulation, count, edges) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == expected);
    };
    // Explicit graph policy independently creates, updates and clears all edges.
    submit_graph(updated_graph.data(),
                 static_cast<std::uint32_t>(updated_graph.size()));
    submit_graph(graph.data(), static_cast<std::uint32_t>(graph.size()));
    submit_graph(nullptr, 0U);
    submit_graph(graph.data(), static_cast<std::uint32_t>(graph.size()));

    const auto replace_and_step = [&](const char *name,
                                      auto record_replacement) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty_pipeline) ==
                 PM_SUCCESS);
        record_replacement();
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        const PmResult submission = context.submit();
        if (submission != PM_SUCCESS)
            throw std::runtime_error(std::string(name) +
                " replacement submission: " + std::to_string(submission));
    };
    // Each write retires borrowed generations. The explicit graph must rebind
    // every incident pair before the following step.
    replace_and_step("fluid", [&] {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                        &particles) == PM_SUCCESS);
    });
    replace_and_step("rigid body", [&] {
        PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                       &body) == PM_SUCCESS);
    });
    replace_and_step("cloth", [&] {
        PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U,
                                  &cloth) == PM_SUCCESS);
    });
    replace_and_step("rope", [&] {
        PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0U, 1U, &rope) ==
                 PM_SUCCESS);
    });
    replace_and_step("soft body", [&] {
        PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0U, 1U,
                                      &soft) == PM_SUCCESS);
    });
    replace_and_step("smoke", [&] {
        PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U, &smoke) ==
                 PM_SUCCESS);
    });

    // Fluid, rope and soft body form a three-family graph. Smoke-cloth is an
    // unrelated edge that must survive the failed replacement rollback.
    std::array<PmCouplingState, 4> four_family_graph{
        graph[2], graph[3], graph[6], graph[9]};
    std::array<PmCouplingState, 4> four_family_update{
        updated_graph[2], updated_graph[3], updated_graph[6], updated_graph[9]};
    const std::string installed_graph_sha = graph_fingerprint(
        four_family_graph.data(), four_family_graph.size());
    const std::string updated_graph_sha = graph_fingerprint(
        four_family_update.data(), four_family_update.size());
    PM_CHECK(installed_graph_sha != updated_graph_sha);

    // Start the formal profile from freshly installed endpoint generations.
    submit_endpoint("formal reset", [&] {
        PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    });
    submit_endpoint("formal fluid", [&] {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                        &particles) == PM_SUCCESS);
    });
    submit_endpoint("formal cloth", [&] {
        PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U, &cloth) ==
                 PM_SUCCESS);
    });
    submit_endpoint("formal rope", [&] {
        PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0U, 1U,
                                 &stable_rope) == PM_SUCCESS);
    });
    submit_endpoint("formal soft body", [&] {
        PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0U, 1U,
                                      &soft) == PM_SUCCESS);
    });
    submit_endpoint("formal smoke", [&] {
        PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U, &smoke) ==
                 PM_SUCCESS);
    });

    const PmDeviceSize fluid_size = sizeof(PmDomainOutputHeader) +
        fluid_capacity.particleCapacity * sizeof(PmFluidParticleOutput);
    const PmDeviceSize rope_size = sizeof(PmDomainOutputHeader) +
        rope_capacity.nodeCapacity * sizeof(PmRopeNodeState);
    const PmDeviceSize soft_size = sizeof(PmDomainOutputHeader) +
        soft_capacity.nodeCapacity * sizeof(PmSoftBodyNodeState);
    const PmDeviceSize smoke_size = sizeof(PmDomainOutputHeader) +
        smoke_capacity.particleCapacity * sizeof(PmSmokeParticleState);
    const PmDeviceSize cloth_size = sizeof(PmDomainOutputHeader) +
        cloth_capacity.vertexCapacity * sizeof(PmClothVertexState);
    const PmDeviceSize fluid_offset = 0U;
    const PmDeviceSize rope_offset = fluid_offset + fluid_size;
    const PmDeviceSize soft_offset = rope_offset + rope_size;
    const PmDeviceSize smoke_offset = soft_offset + soft_size;
    const PmDeviceSize cloth_offset = smoke_offset + smoke_size;
    const auto state_output = context.buffer(
        cloth_offset + cloth_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange fluid_output{state_output.handle, fluid_offset, fluid_size};
    const PmBufferRange rope_output{state_output.handle, rope_offset, rope_size};
    const PmBufferRange soft_output{state_output.handle, soft_offset, soft_size};
    const PmBufferRange smoke_output{state_output.handle, smoke_offset, smoke_size};
    const PmBufferRange cloth_output{state_output.handle, cloth_offset, cloth_size};

    const auto capture = [&](std::string id, std::string outcome,
                             const std::string &graph_sha,
                             std::uint64_t liquid_generation) {
        context.begin();
        PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                         &fluid_output) == PM_SUCCESS);
        PM_CHECK(pmCmdCopyRopeState(context.commands, simulation, 0U,
                                    &rope_output) == PM_SUCCESS);
        PM_CHECK(pmCmdCopySoftBodyState(context.commands, simulation, 0U,
                                        &soft_output) == PM_SUCCESS);
        PM_CHECK(pmCmdCopySmokeState(context.commands, simulation, 0U,
                                     &smoke_output) == PM_SUCCESS);
        PM_CHECK(pmCmdCopyClothState(context.commands, simulation, 0U,
                                     &cloth_output) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(state_output);
        record_checkpoint(
            result, std::move(id), std::move(outcome), graph_sha,
            {output_header(state_output, fluid_offset),
             output_header(state_output, rope_offset),
             output_header(state_output, soft_offset),
             output_header(state_output, smoke_offset),
             output_header(state_output, cloth_offset)},
            liquid_generation);
    };

    submit_graph(four_family_graph.data(),
                 static_cast<std::uint32_t>(four_family_graph.size()));
    capture("installed", "PM_SUCCESS", installed_graph_sha, 1U);
    submit_graph(four_family_update.data(),
                 static_cast<std::uint32_t>(four_family_update.size()));
    capture("updated", "PM_SUCCESS", updated_graph_sha, 1U);
    auto invalid_replacement = four_family_update;
    invalid_replacement[1].parameters.mechanical.friction = -1.0F;
    // Descriptor/capability preflight accepts this graph. Runtime creation
    // installs the first edge, rejects the second, clears the partial graph,
    // and restores the previously active graph including smoke-cloth.
    submit_graph(invalid_replacement.data(),
                 static_cast<std::uint32_t>(invalid_replacement.size()),
                 PM_ERROR_INVALID_ARGUMENT);
    capture("partial-failure", "PM_ERROR_INVALID_ARGUMENT", updated_graph_sha,
            1U);
    submit_graph(four_family_graph.data(),
                 static_cast<std::uint32_t>(four_family_graph.size()));
    capture("recovery", "PM_SUCCESS", installed_graph_sha, 1U);
    replace_and_step("formal fluid", [&] {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                        &replacement_particles) == PM_SUCCESS);
    });
    capture("replace-liquid", "PM_SUCCESS", installed_graph_sha, 2U);

    PM_CHECK(result->observations.size() == 5U);
    const auto same_resources = [&](std::size_t first, std::size_t second,
                                    std::size_t begin = 0U) {
        for (std::size_t index = begin;
             index < result->observations[first].resources.size(); ++index) {
            const auto &a = result->observations[first].resources[index];
            const auto &b = result->observations[second].resources[index];
            if (a.generation != b.generation ||
                a.state_sha256 != b.state_sha256)
                std::fprintf(stderr,
                    "checkpoint resource drift: %zu/%zu %s %s %s\n",
                    first, second, a.id.c_str(), a.state_sha256.c_str(),
                    b.state_sha256.c_str());
            PM_CHECK(a.generation == b.generation);
            PM_CHECK(a.state_sha256 == b.state_sha256);
        }
    };
    same_resources(0U, 1U);
    same_resources(1U, 2U);
    same_resources(0U, 3U);
    PM_CHECK(result->observations[3U].resources[0U].generation !=
             result->observations[4U].resources[0U].generation);
    PM_CHECK(result->observations[3U].resources[0U].state_sha256 !=
             result->observations[4U].resources[0U].state_sha256);
    same_resources(3U, 4U, 1U);
    submit_graph(nullptr, 0U);

    pmDestroySimulation(context.device, simulation);
    for (const auto pipeline :
         {empty_pipeline, graph_pipeline, updated_pipeline})
        pmDestroyPhysicsPipeline(context.device, pipeline);
    pmDestroyGeometry(context.device, geometry);
    result->status = 0;
    std::printf(
        "PASS: all 12 interaction pairs rebind; 5 graph rollback checkpoints "
        "on %s (vendor=%04x device=%04x)\n",
        properties.deviceName, properties.vendorId, properties.deviceId);
    return 0;
}

#ifndef PM_CONFORMANCE_EMBEDDED
int main(int argc, char **argv) {
    std::filesystem::path output;
    std::filesystem::path profile = PM_LIFECYCLE_PROFILE_PATH;
    std::string driver_name = "selected-driver";
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if ((option == "--output" || option == "--profile" ||
             option == "--driver-name") && index + 1 >= argc) {
            std::fprintf(stderr, "%.*s requires a value\n",
                         static_cast<int>(option.size()), option.data());
            return 2;
        }
        if (option == "--output") output = argv[++index];
        else if (option == "--profile") profile = argv[++index];
        else if (option == "--driver-name") driver_name = argv[++index];
        else {
            std::fprintf(stderr, "unknown option: %.*s\n",
                         static_cast<int>(option.size()), option.data());
            return 2;
        }
    }
    GraphLifecycleResult result;
    const int status = parallel_mater_interaction_graph_lifecycle_fixture(&result);
    if (!output.empty()) {
        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());
        std::ofstream stream(output, std::ios::binary);
        if (!stream) {
            std::fprintf(stderr, "cannot write lifecycle result: %s\n",
                         output.string().c_str());
            return 2;
        }
        stream << lifecycle_result_json(result, driver_name, profile);
    }
    return status;
}
#endif
