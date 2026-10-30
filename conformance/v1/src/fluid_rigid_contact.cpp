// SPDX-License-Identifier: MIT
// Public-SDK fluid/rigid contact quality and graph lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <algorithm>
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
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

constexpr float timestep = 1.0F / 60.0F;
constexpr float particle_radius = 0.05F;
constexpr float particle_mass = 1.0F;
constexpr float rigid_mass = 2.0F;
constexpr float cube_half_extent = 0.1F;
constexpr float rigid_collision_margin = 0.005F;
constexpr float cube_inertia =
    (1.0F / 6.0F) * rigid_mass * (2.0F * cube_half_extent) *
    (2.0F * cube_half_extent);
constexpr std::uint32_t contact_capacity = 8U;

constexpr std::array<PmFloat3, 8U> cube_vertices{{
    {-cube_half_extent, -cube_half_extent, -cube_half_extent},
    { cube_half_extent, -cube_half_extent, -cube_half_extent},
    { cube_half_extent,  cube_half_extent, -cube_half_extent},
    {-cube_half_extent,  cube_half_extent, -cube_half_extent},
    {-cube_half_extent, -cube_half_extent,  cube_half_extent},
    { cube_half_extent, -cube_half_extent,  cube_half_extent},
    { cube_half_extent,  cube_half_extent,  cube_half_extent},
    {-cube_half_extent,  cube_half_extent,  cube_half_extent},
}};

constexpr std::array<std::uint32_t, 36U> cube_indices{{
    0U, 3U, 2U, 0U, 2U, 1U, // -Z
    4U, 5U, 6U, 4U, 6U, 7U, // +Z
    0U, 4U, 7U, 0U, 7U, 3U, // -X, impact face
    1U, 2U, 6U, 1U, 6U, 5U, // +X
    0U, 1U, 5U, 0U, 5U, 4U, // -Y
    3U, 7U, 6U, 3U, 6U, 2U, // +Y
}};

struct Snapshot {
    PmFluidParticleOutput particle{};
    PmRigidState rigid{};
    std::vector<PmContactEvent> contacts{};
};

struct FixtureObjects {
    PmGeometry geometry{};
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer fluid_output{};
    Buffer rigid_output{};
    Buffer contact_output{};
    PmBufferRange fluid_range{};
    PmBufferRange rigid_range{};
    PmBufferRange contact_range{};
};

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query fluid-rigid contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown fluid-rigid contact backend");
}

[[nodiscard]] Json driver_json(const Context &context) {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    Json result = Json::object();
    result["name"] = native_driver_name(context.device);
    result["device"] = properties.deviceName;
    result["vendor_id"] = static_cast<std::uint64_t>(properties.vendorId);
    result["device_id"] = static_cast<std::uint64_t>(properties.deviceId);
    result["driver_version"] = std::to_string(properties.driverVersion);
    return result;
}

[[nodiscard]] Json unavailable_driver() {
    Json result = Json::object();
    result["name"] = "unavailable";
    result["device"] = "unavailable";
    return result;
}

[[nodiscard]] PmPhysicalDeviceFeatures required_features() {
    PmPhysicalDeviceFeatures needed{};
    needed.rigidBodies = PM_TRUE;
    needed.particleLiquids = PM_TRUE;
    needed.liquidRigidCoupling = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_contact_capability(const Context &context) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error("unsupported: fluid-rigid contact capability");
}

[[nodiscard]] PmLiquidMaterialState fluid_material() {
    PmLiquidMaterialState material{};
    material.restDensity = 1'000.0F;
    material.particleRadius = particle_radius;
    material.supportRadius = 0.2F;
    material.restParticleVolume = particle_mass / material.restDensity;
    material.maximumSpeed = 5.0F;
    material.maximumPairAcceleration = 1'000.0F;
    material.densityIterations = 1U;
    return material;
}

[[nodiscard]] PmRigidBodyInfo rigid_body(PmGeometry geometry) {
    PmRigidBodyInfo body{};
    body.active = PM_TRUE;
    body.motionType = PM_MOTION_DYNAMIC;
    body.geometry = geometry;
    body.state.orientation[3] = 1.0F;
    body.mass = rigid_mass;
    body.inertiaDiagonal[0] = cube_inertia;
    body.inertiaDiagonal[1] = cube_inertia;
    body.inertiaDiagonal[2] = cube_inertia;
    body.friction = 0.0F;
    body.restitution = 0.0F;
    body.linearDamping = 0.0F;
    body.angularDamping = 0.0F;
    body.maximumLinearSpeed = 5.0F;
    body.maximumAngularSpeed = 10.0F;
    body.collisionMargin = rigid_collision_margin;
    return body;
}

// API 0.2 exposes CONTACT and CONTAINMENT as a fixed descriptor bundle for
// liquid/rigid. This fixture uses an exterior, one-face impact and qualifies
// only CONTACT; it makes no closed-volume retention observation.
[[nodiscard]] PmCouplingState contact_coupling(float distance) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_LIQUID_RIGID;
    coupling.first = 0U;
    coupling.second = 0U;
    coupling.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_CONTAINMENT_BIT;
    coupling.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    coupling.parameters.mechanical.contactDistance = distance;
    coupling.parameters.mechanical.friction = 0.0F;
    coupling.parameters.mechanical.maximumAcceleration = 1'000.0F;
    coupling.parameters.mechanical.twoWay = PM_TRUE;
    return coupling;
}

[[nodiscard]] PmGeometry create_geometry(Context &context) {
    const PmDeviceSize vertex_bytes = sizeof(cube_vertices);
    const PmDeviceSize index_bytes = sizeof(cube_indices);
    const Buffer input = context.buffer(
        vertex_bytes + index_bytes, PM_BUFFER_GEOMETRY_INPUT_BIT);
    std::memcpy(input.mapped, cube_vertices.data(), vertex_bytes);
    std::memcpy(static_cast<std::byte *>(input.mapped) + vertex_bytes,
                cube_indices.data(), index_bytes);
    context.flush(input);

    PmGeometryCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    info.vertices = {{input.handle, 0U, vertex_bytes},
                     PM_FORMAT_R32G32B32_SFLOAT,
                     static_cast<std::uint32_t>(cube_vertices.size()),
                     sizeof(PmFloat3)};
    info.indices = {{input.handle, vertex_bytes, index_bytes},
                    PM_FORMAT_R32_UINT,
                    static_cast<std::uint32_t>(cube_indices.size()),
                    sizeof(std::uint32_t)};
    PmGeometry geometry{};
    PM_CHECK(pmCreateGeometry(context.device, &info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements requirements{};
    pmGetGeometryMemoryRequirements(context.device, geometry, &requirements);
    PM_CHECK(pmBindGeometryMemory(context.device, geometry,
                                  context.allocate(requirements), 0U) ==
             PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBuildGeometry(context.commands, geometry) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    return geometry;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context,
                                            std::uint32_t coupling_capacity) {
    FixtureObjects fixture{};
    fixture.geometry = create_geometry(context);
    constexpr PmFluidCapacity fluid_capacity{1U, 4U};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    simulation_info.contactCapacity = contact_capacity;
    simulation_info.fluidCount = 1U;
    simulation_info.pFluidCapacities = &fluid_capacity;
    simulation_info.couplingCapacity = coupling_capacity;
    PM_CHECK(pmCreateSimulation(context.device, &simulation_info,
                                &fixture.simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, fixture.simulation,
                                      &requirements);
    const PmSimulationMemoryBinding binding{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, fixture.simulation,
                                    &binding) == PM_SUCCESS);

    const PmLiquidMaterialState liquid = fluid_material();
    const PmRigidSolverState rigid{4U, PM_FALSE, 0.0F, 0.0F, 0.0F};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid;
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &liquid;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize fluid_bytes = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    constexpr PmDeviceSize rigid_bytes = sizeof(PmRigidState);
    constexpr PmDeviceSize contact_bytes = sizeof(PmDomainOutputHeader) +
        contact_capacity * sizeof(PmContactEvent);
    constexpr PmFlags output_usage = PM_BUFFER_SIMULATION_OUTPUT_BIT |
        PM_BUFFER_TRANSFER_DST_BIT;
    fixture.fluid_output = context.buffer(fluid_bytes, output_usage);
    fixture.rigid_output = context.buffer(rigid_bytes, output_usage);
    fixture.contact_output = context.buffer(contact_bytes, output_usage);
    fixture.fluid_range = {fixture.fluid_output.handle, 0U, fluid_bytes};
    fixture.rigid_range = {fixture.rigid_output.handle, 0U, rigid_bytes};
    fixture.contact_range = {fixture.contact_output.handle, 0U, contact_bytes};
    return fixture;
}

void record_endpoint_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, fixture.simulation, 0U,
                                    &fixture.fluid_range) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRigidStates(context.commands, fixture.simulation, 0U, 1U,
                                 &fixture.rigid_range) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture,
                                     bool include_contacts) {
    context.invalidate(fixture.fluid_output);
    context.invalidate(fixture.rigid_output);
    const auto *fluid_header = static_cast<const PmDomainOutputHeader *>(
        fixture.fluid_output.mapped);
    if (fluid_header->activeCount != 1U)
        throw std::runtime_error("fluid-rigid fluid output count mismatch");
    Snapshot result{};
    result.particle = reinterpret_cast<const PmFluidParticleOutput *>(
        fluid_header + 1)[0];
    result.rigid = *static_cast<const PmRigidState *>(fixture.rigid_output.mapped);
    if (include_contacts) {
        context.invalidate(fixture.contact_output);
        const auto *header = static_cast<const PmDomainOutputHeader *>(
            fixture.contact_output.mapped);
        if (header->activeCount > contact_capacity)
            throw std::runtime_error("fluid-rigid contact output count mismatch");
        const auto *events = reinterpret_cast<const PmContactEvent *>(header + 1);
        result.contacts.assign(events, events + header->activeCount);
    }
    return result;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          float particle_x, float particle_velocity_x,
                          Snapshot *snapshot = nullptr) {
    PmParticleState particle{};
    particle.position[0] = particle_x;
    particle.velocity[0] = particle_velocity_x;
    particle.temperatureCelsius = 20.0F;
    const Buffer input = context.buffer(
        sizeof(particle), PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(input.mapped, &particle, sizeof(particle));
    context.flush(input);
    const PmBufferView view{{input.handle, 0U, sizeof(particle)},
                            PM_FORMAT_PARTICLE_STATE, 1U,
                            sizeof(PmParticleState)};
    const PmRigidBodyInfo body = rigid_body(fixture.geometry);
    const auto submit_stage = [&](const char *label, const auto &record) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        record();
        const PmResult submitted = context.submit();
        if (submitted != PM_SUCCESS)
            throw std::runtime_error(std::string("fluid-rigid ") + label +
                                     " failed: " +
                                     std::to_string(submitted));
    };
    submit_stage("fluid initialization", [&] {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, fixture.simulation, 0U,
                                        &view) == PM_SUCCESS);
    });
    submit_stage("rigid initialization", [&] {
        PM_CHECK(pmCmdWriteRigidBodies(context.commands, fixture.simulation, 0U,
                                       1U, &body) == PM_SUCCESS);
    });
    if (snapshot != nullptr) {
        submit_stage("initial capture", [&] {
            record_endpoint_copies(context, fixture);
        });
        *snapshot = read_snapshot(context, fixture, false);
    }
}

struct QualityRun {
    Json driver{};
    Snapshot initial{};
    Snapshot final{};
};

[[nodiscard]] QualityRun run_quality_fixture() {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid-rigid contact features");
    require_contact_capability(context);
    FixtureObjects fixture = create_fixture(context, 1U);
    Snapshot initial{};
    initialize_endpoints(context, fixture, -0.155F, 1.0F, &initial);
    const PmCouplingState coupling = contact_coupling(particle_radius);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F},
                          PM_STEP_COLLECT_CONTACT_EVENTS_BIT};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation, &step) ==
             PM_SUCCESS);
    record_endpoint_copies(context, fixture);
    PM_CHECK(pmCmdCopyContactEvents(context.commands, fixture.simulation,
                                   &fixture.contact_range) == PM_SUCCESS);
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("fluid-rigid contact submission failed: " +
                                 std::to_string(submitted));
    Snapshot final = read_snapshot(context, fixture, true);
    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
    return {std::move(driver), std::move(initial), std::move(final)};
}

[[nodiscard]] Json vector_json(PmFloat3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

[[nodiscard]] Json vector_json(const float value[3]) {
    return vector_json({value[0], value[1], value[2]});
}

[[nodiscard]] Json quaternion_json(const float value[4]) {
    Json result = Json::array();
    for (std::size_t index = 0U; index < 4U; ++index)
        result.push_back(static_cast<double>(value[index]));
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot,
                                 bool include_contacts) {
    Json fluid = Json::object();
    fluid["stable_id"] = static_cast<std::uint64_t>(snapshot.particle.stableId);
    fluid["position_m"] = vector_json(snapshot.particle.position);
    fluid["velocity_m_s"] = vector_json(snapshot.particle.velocity);
    fluid["acceleration_m_s2"] = vector_json(snapshot.particle.acceleration);
    Json rigid = Json::object();
    rigid["position_m"] = vector_json(snapshot.rigid.position);
    rigid["orientation_xyzw"] = quaternion_json(snapshot.rigid.orientation);
    rigid["linear_velocity_m_s"] = vector_json(snapshot.rigid.linearVelocity);
    rigid["angular_velocity_rad_s"] = vector_json(snapshot.rigid.angularVelocity);
    Json result = Json::object();
    result["fluid"] = std::move(fluid);
    result["rigid"] = std::move(rigid);
    if (include_contacts) {
        Json contacts = Json::array();
        for (const auto &contact : snapshot.contacts) {
            Json item = Json::object();
            item["type"] = contact.type == PM_CONTACT_LIQUID_RIGID
                ? "LIQUID_RIGID" : "RIGID_RIGID";
            item["first"] = static_cast<std::uint64_t>(contact.first);
            item["second"] = static_cast<std::uint64_t>(contact.second);
            item["position_m"] = vector_json(contact.position);
            item["normal"] = vector_json(contact.normal);
            item["normal_impulse_n_s"] =
                static_cast<double>(contact.normalImpulse);
            contacts.push_back(std::move(item));
        }
        result["contact_events"] = std::move(contacts);
    }
    return result;
}

[[nodiscard]] Json vertices_json() {
    Json result = Json::array();
    for (const auto vertex : cube_vertices)
        result.push_back(vector_json(vertex));
    return result;
}

[[nodiscard]] Json indices_json() {
    Json result = Json::array();
    for (const auto index : cube_indices)
        result.push_back(static_cast<std::uint64_t>(index));
    return result;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, const float value[3]) {
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
        if (axis != 0U) stream << ',';
        append_float(stream, value[axis]);
    }
}

[[nodiscard]] std::string graph_fingerprint(
    const PmCouplingState *coupling) {
    if (coupling == nullptr)
        return sha256("parallel-mater-fluid-rigid-contact-graph/v1;inactive");
    const auto &mechanical = coupling->parameters.mechanical;
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-rigid-contact-graph/v1;type="
              << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";qualified-mode=CONTACT;direction=BIDIRECTIONAL;material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";contact=";
    append_float(canonical, mechanical.contactDistance);
    canonical << ";friction=";
    append_float(canonical, mechanical.friction);
    canonical << ";maximum-acceleration=";
    append_float(canonical, mechanical.maximumAcceleration);
    canonical << ";two-way=" << mechanical.twoWay;
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-rigid-contact-endpoints/v1;particle="
              << snapshot.particle.stableId << ",position=";
    append_vector(canonical, snapshot.particle.position);
    canonical << ",velocity=";
    append_vector(canonical, snapshot.particle.velocity);
    canonical << ";rigid-position=";
    append_vector(canonical, snapshot.rigid.position);
    canonical << ";rigid-linear=";
    append_vector(canonical, snapshot.rigid.linearVelocity);
    canonical << ";rigid-angular=";
    append_vector(canonical, snapshot.rigid.angularVelocity);
    return sha256(canonical.str());
}

[[nodiscard]] Json resource_json(std::string id, bool active,
                                 std::uint64_t generation,
                                 std::string fingerprint) {
    Json result = Json::object();
    result["id"] = std::move(id);
    result["active"] = active;
    result["generation"] = generation;
    result["state_sha256"] = std::move(fingerprint);
    return result;
}

void add_checkpoint(Json &observations, std::string id, std::string outcome,
                    const PmCouplingState *coupling,
                    std::uint64_t graph_generation,
                    const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        "fluid-rigid-contact-graph", coupling != nullptr, graph_generation,
        graph_fingerprint(coupling)));
    resources.push_back(resource_json(
        "fluid-rigid-endpoints", true, 1U,
        endpoint_fingerprint(endpoints)));
    Json checkpoint = Json::object();
    checkpoint["id"] = std::move(id);
    checkpoint["outcome"] = std::move(outcome);
    checkpoint["resources"] = std::move(resources);
    observations.push_back(std::move(checkpoint));
}

struct LifecycleRun {
    Json driver{};
    Json observations{};
};

[[nodiscard]] LifecycleRun run_lifecycle_fixture() {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid-rigid contact features");
    require_contact_capability(context);
    FixtureObjects fixture = create_fixture(context, 1U);
    initialize_endpoints(context, fixture, -1.0F, 0.0F);
    const PmCouplingState baseline = contact_coupling(0.05F);
    const PmCouplingState replacement = contact_coupling(0.06F);
    const PmCouplingState recovery = contact_coupling(0.07F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();

    const auto mutate = [&](const PmCouplingState *couplings,
                            std::uint32_t count, PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, count,
                                  couplings) == expected);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        record_endpoint_copies(context, fixture);
        PM_CHECK(context.submit() == PM_SUCCESS);
        return read_snapshot(context, fixture, false);
    };

    add_checkpoint(observations, "baseline", "PM_SUCCESS", &baseline, 1U,
                   mutate(&baseline, 1U, PM_SUCCESS));
    add_checkpoint(observations, "replacement", "PM_SUCCESS", &replacement,
                   2U, mutate(&replacement, 1U, PM_SUCCESS));
    add_checkpoint(observations, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   mutate(overflow.data(),
                          static_cast<std::uint32_t>(overflow.size()),
                          PM_ERROR_CAPACITY_EXCEEDED));
    add_checkpoint(observations, "removal", "PM_SUCCESS", nullptr, 3U,
                   mutate(nullptr, 0U, PM_SUCCESS));
    add_checkpoint(observations, "recovery", "PM_SUCCESS", &recovery, 4U,
                   mutate(&recovery, 1U, PM_SUCCESS));

    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_fluid_rigid_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "fluid-rigid-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/fluid-rigid-contact-v1.json");
    try {
        QualityRun run = run_quality_fixture();
        Json modes = Json::array();
        modes.push_back("CONTACT");
        modes.push_back("CONTAINMENT");
        Json impact_face = Json::array();
        for (const std::uint64_t index : {0U, 4U, 7U, 3U})
            impact_face.push_back(index);
        Json fixture = Json::object();
        fixture["id"] = "centered-exterior-cube-impact";
        fixture["timestep_seconds"] = static_cast<double>(timestep);
        fixture["step_count"] = static_cast<std::uint64_t>(1U);
        fixture["substeps"] = static_cast<std::uint64_t>(1U);
        fixture["gravity_m_s2"] = vector_json(PmFloat3{});
        fixture["declared_mode_flags"] = std::move(modes);
        fixture["exercised_mode"] = "CONTACT";
        fixture["containment_claimed"] = false;
        fixture["fluid_rest_density_kg_m3"] = 1'000.0;
        fixture["fluid_particle_radius_m"] =
            static_cast<double>(particle_radius);
        fixture["fluid_support_radius_m"] = 0.2;
        fixture["fluid_rest_particle_volume_m3"] = 0.001;
        fixture["fluid_particle_mass_kg"] = static_cast<double>(particle_mass);
        fixture["fluid_maximum_speed_m_s"] = 5.0;
        fixture["particle_initial_position_m"] =
            vector_json(PmFloat3{-0.155F, 0.0F, 0.0F});
        fixture["particle_initial_velocity_m_s"] =
            vector_json(PmFloat3{1.0F, 0.0F, 0.0F});
        fixture["rigid_geometry_vertices_m"] = vertices_json();
        fixture["rigid_geometry_triangle_indices"] = indices_json();
        fixture["impact_face_vertex_indices"] = std::move(impact_face);
        fixture["rigid_initial_position_m"] = vector_json(PmFloat3{});
        const float orientation[4]{0.0F, 0.0F, 0.0F, 1.0F};
        fixture["rigid_initial_orientation_xyzw"] =
            quaternion_json(orientation);
        fixture["rigid_mass_kg"] = static_cast<double>(rigid_mass);
        fixture["rigid_inertia_diagonal_kg_m2"] = vector_json(
            PmFloat3{cube_inertia, cube_inertia, cube_inertia});
        fixture["rigid_friction"] = 0.0;
        fixture["rigid_restitution"] = 0.0;
        fixture["rigid_linear_damping_per_s"] = 0.0;
        fixture["rigid_angular_damping_per_s"] = 0.0;
        fixture["rigid_maximum_linear_speed_m_s"] = 5.0;
        fixture["rigid_maximum_angular_speed_rad_s"] = 10.0;
        fixture["rigid_collision_margin_m"] =
            static_cast<double>(rigid_collision_margin);
        fixture["coupling_contact_distance_m"] =
            static_cast<double>(particle_radius);
        fixture["coupling_friction"] = 0.0;
        fixture["coupling_maximum_acceleration_m_s2"] = 1'000.0;
        fixture["coupling_two_way"] = true;
        fixture["oracle"] =
            "public-exterior-clearance-contact-event-reciprocal-normal-impulse";
        fixture["initial"] = snapshot_json(run.initial, false);
        fixture["final"] = snapshot_json(run.final, true);
        Json fixtures = Json::array();
        fixtures.push_back(std::move(fixture));
        result["execution_status"] = "completed";
        result["driver"] = std::move(run.driver);
        result["fixtures"] = std::move(fixtures);
        if (status != nullptr) *status = ExecutionStatus::completed;
    } catch (const std::exception &error) {
        const bool unsupported = unsupported_error(error);
        result["execution_status"] = unsupported ? "unsupported" : "failed";
        result["reason"] = error.what();
        result["driver"] = unavailable_driver();
        result["fixtures"] = Json::array();
        if (status != nullptr) *status = unsupported ?
            ExecutionStatus::unsupported : ExecutionStatus::failed;
    }
    return result.serialize();
}

std::string conformance_fluid_rigid_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "fluid-rigid-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/fluid-rigid-contact-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("fluid");
        families.push_back("rigid_body");
        Json unrelated = Json::array();
        unrelated.push_back("fluid-rigid-endpoints");
        Json attempt = Json::object();
        attempt["fixture"] = "fluid-rigid-contact-public-lifecycle";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "fluid-rigid-contact-capacity-replacement-removal-rollback";
        item["observations"] = std::move(run.observations);
        item["unrelated_resources"] = std::move(unrelated);
        Json cases = Json::array();
        cases.push_back(std::move(item));
        result["execution_status"] = "completed";
        result["driver"] = std::move(run.driver);
        result["cases"] = std::move(cases);
        if (status != nullptr) *status = ExecutionStatus::completed;
    } catch (const std::exception &error) {
        const bool unsupported = unsupported_error(error);
        result["execution_status"] = unsupported ? "unsupported" : "failed";
        result["reason"] = error.what();
        result["driver"] = unavailable_driver();
        result["cases"] = Json::array();
        if (status != nullptr) *status = unsupported ?
            ExecutionStatus::unsupported : ExecutionStatus::failed;
    }
    return result.serialize();
}

} // namespace parallel_mater::conformance
