// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

using parallel_mater::test::Context;

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

} // namespace

int main() {
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
    if (!context.open(needed)) return 77;

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

    PmParticleState particle{};
    particle.position[0] = particle.position[1] = particle.position[2] = 10.0F;
    const auto particle_input = context.buffer(
        sizeof(particle), PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(particle_input.mapped, &particle, sizeof(particle));
    context.flush(particle_input);
    const PmBufferView particles{
        {particle_input.handle, 0U, sizeof(particle)},
        PM_FORMAT_PARTICLE_STATE, 1U, sizeof(particle)};
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
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};

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

    const auto replace_and_step = [&](auto record_replacement) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty_pipeline) ==
                 PM_SUCCESS);
        record_replacement();
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    };
    // Each write retires borrowed generations. The explicit graph must rebind
    // every incident pair before the following step.
    replace_and_step([&] {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                        &particles) == PM_SUCCESS);
    });
    replace_and_step([&] {
        PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                       &body) == PM_SUCCESS);
    });
    replace_and_step([&] {
        PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U,
                                  &cloth) == PM_SUCCESS);
    });
    replace_and_step([&] {
        PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0U, 1U, &rope) ==
                 PM_SUCCESS);
    });
    replace_and_step([&] {
        PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0U, 1U,
                                      &soft) == PM_SUCCESS);
    });
    replace_and_step([&] {
        PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U, &smoke) ==
                 PM_SUCCESS);
    });

    // Fluid, rope and soft body form a three-family graph. Smoke-cloth is an
    // unrelated edge that must survive the failed replacement rollback.
    std::array<PmCouplingState, 4> four_family_graph{
        graph[2], graph[3], graph[6], graph[9]};
    std::array<PmCouplingState, 4> four_family_update{
        updated_graph[2], updated_graph[3], updated_graph[6], updated_graph[9]};
    submit_graph(four_family_graph.data(),
                 static_cast<std::uint32_t>(four_family_graph.size()));
    submit_graph(four_family_update.data(),
                 static_cast<std::uint32_t>(four_family_update.size()));
    auto invalid_replacement = four_family_update;
    invalid_replacement[1].parameters.mechanical.friction = -1.0F;
    // Descriptor/capability preflight accepts this graph. Runtime creation
    // installs the first edge, rejects the second, clears the partial graph,
    // and restores the previously active graph including smoke-cloth.
    submit_graph(invalid_replacement.data(),
                 static_cast<std::uint32_t>(invalid_replacement.size()),
                 PM_ERROR_INVALID_ARGUMENT);
    submit_graph(four_family_graph.data(),
                 static_cast<std::uint32_t>(four_family_graph.size()));
    replace_and_step([&] {
        PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U, &smoke) ==
                 PM_SUCCESS);
    });
    submit_graph(nullptr, 0U);

    pmDestroySimulation(context.device, simulation);
    for (const auto pipeline :
         {empty_pipeline, graph_pipeline, updated_pipeline})
        pmDestroyPhysicsPipeline(context.device, pipeline);
    pmDestroyGeometry(context.device, geometry);
    std::puts("PASS: all 12 interaction pairs rebind; three-family rollback recovers");
}
