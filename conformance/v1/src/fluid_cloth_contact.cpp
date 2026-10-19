// SPDX-License-Identifier: MIT
// Public-SDK fluid/cloth contact quality and graph lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr float quality_timestep = 0.005F;
constexpr std::uint32_t quality_steps = 100U;
constexpr float particle_mass = 1.0F;
constexpr float cloth_vertex_mass = 1.0F;
constexpr float contact_distance = 0.05F;
constexpr float contact_stiffness = 200.0F;
constexpr float contact_damping = 25.0F;
constexpr float maximum_force = 100.0F;
constexpr std::array<PmFloat3, 3U> triangle{{
    {-1.0F, 0.0F, -1.0F},
    {0.0F, 0.0F, 1.0F},
    {1.0F, 0.0F, -1.0F}}};
constexpr std::array<std::uint32_t, 3U> triangle_indices{{0U, 1U, 2U}};

struct Snapshot {
    PmFluidParticleOutput particle{};
    std::array<PmClothVertexState, 3U> cloth{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange fluid_output{};
    PmBufferRange cloth_output{};
};

[[nodiscard]] PmFloat3 add(PmFloat3 a, PmFloat3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] PmFloat3 subtract(PmFloat3 a, PmFloat3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] PmFloat3 multiply(PmFloat3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

[[nodiscard]] float dot(PmFloat3 a, PmFloat3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] PmFloat3 cross(PmFloat3 a, PmFloat3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

[[nodiscard]] PmFloat3 normalized(PmFloat3 value) {
    const float length = std::sqrt(dot(value, value));
    if (!(length > 0.0F)) throw std::runtime_error("degenerate cloth triangle");
    return multiply(value, 1.0F / length);
}

[[nodiscard]] PmFloat3 cloth_normal(const Snapshot &snapshot) {
    return normalized(cross(
        subtract(snapshot.cloth[1].position, snapshot.cloth[0].position),
        subtract(snapshot.cloth[2].position, snapshot.cloth[0].position)));
}

[[nodiscard]] float signed_separation(const Snapshot &snapshot) {
    return dot(subtract(
        {snapshot.particle.position[0], snapshot.particle.position[1],
         snapshot.particle.position[2]},
        snapshot.cloth[0].position), cloth_normal(snapshot));
}

[[nodiscard]] PmFloat3 particle_velocity(const Snapshot &snapshot) {
    return {snapshot.particle.velocity[0], snapshot.particle.velocity[1],
            snapshot.particle.velocity[2]};
}

[[nodiscard]] PmFloat3 endpoint_momentum(const Snapshot &snapshot) {
    PmFloat3 result = multiply(particle_velocity(snapshot), particle_mass);
    for (const auto &vertex : snapshot.cloth)
        result = add(result, multiply(vertex.velocity, cloth_vertex_mass));
    return result;
}

[[nodiscard]] PmFloat3 mean_cloth_velocity(const Snapshot &snapshot) {
    PmFloat3 result{};
    for (const auto &vertex : snapshot.cloth)
        result = add(result, vertex.velocity);
    return multiply(result, 1.0F / static_cast<float>(snapshot.cloth.size()));
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

[[nodiscard]] Json vertices_json(
    const std::array<PmFloat3, 3U> &vertices) {
    Json result = Json::array();
    for (const auto vertex : vertices) result.push_back(vector_json(vertex));
    return result;
}

[[nodiscard]] Json indices_json() {
    Json result = Json::array();
    for (const auto index : triangle_indices)
        result.push_back(static_cast<std::uint64_t>(index));
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json vertices = Json::array();
    Json velocities = Json::array();
    Json contact_forces = Json::array();
    for (const auto &vertex : snapshot.cloth) {
        vertices.push_back(vector_json(vertex.position));
        velocities.push_back(vector_json(vertex.velocity));
        contact_forces.push_back(vector_json(vertex.liquidContactForce));
    }
    Json result = Json::object();
    result["particle_position_m"] = vector_json(snapshot.particle.position);
    result["particle_velocity_m_s"] = vector_json(snapshot.particle.velocity);
    result["particle_acceleration_m_s2"] =
        vector_json(snapshot.particle.acceleration);
    result["cloth_vertices_m"] = std::move(vertices);
    result["cloth_velocities_m_s"] = std::move(velocities);
    result["cloth_liquid_contact_forces_n"] = std::move(contact_forces);
    result["cloth_normal"] = vector_json(cloth_normal(snapshot));
    result["signed_separation_m"] =
        static_cast<double>(signed_separation(snapshot));
    result["contact_violation_m"] = static_cast<double>(std::max(
        0.0F, contact_distance + signed_separation(snapshot)));
    result["endpoint_momentum_kg_m_s"] =
        vector_json(endpoint_momentum(snapshot));
    result["fluid_normal_velocity_m_s"] = static_cast<double>(
        dot(particle_velocity(snapshot), cloth_normal(snapshot)));
    result["cloth_mean_normal_velocity_m_s"] = static_cast<double>(
        dot(mean_cloth_velocity(snapshot), cloth_normal(snapshot)));
    return result;
}

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query fluid-cloth contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown fluid-cloth contact backend");
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
    needed.particleLiquids = PM_TRUE;
    needed.cloth = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

[[nodiscard]] PmLiquidMaterialState fluid_material() {
    PmLiquidMaterialState material{};
    material.restDensity = 1'000.0F;
    material.particleRadius = 0.025F;
    material.supportRadius = 0.20F;
    material.restParticleVolume = particle_mass / material.restDensity;
    material.pressureStiffness = 1.0F;
    material.kinematicViscosity = 0.0F;
    material.normalDamping = 0.0F;
    material.velocityDamping = 0.0F;
    material.maximumSpeed = 2.0F;
    material.maximumPairAcceleration = 0.0F;
    material.densityIterations = 1U;
    return material;
}

[[nodiscard]] PmClothMaterialState cloth_material() {
    PmClothMaterialState material{};
    material.vertexMass = cloth_vertex_mass;
    material.thickness = 0.01F;
    material.stretchCompliance = 0.0F;
    material.bendingCompliance = 0.0F;
    material.velocityDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.solverIterations = 1U;
    material.fracturePersistenceSubsteps = 1U;
    material.preserveVolume = PM_FALSE;
    material.volumeCompliance = 1.0e-7F;
    return material;
}

[[nodiscard]] PmCouplingState contact_coupling(float stiffness) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_LIQUID_CLOTH;
    coupling.first = 0U;
    coupling.second = 0U;
    coupling.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    coupling.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    coupling.parameters.mechanical.contactDistance = contact_distance;
    coupling.parameters.mechanical.stiffness = stiffness;
    coupling.parameters.mechanical.damping = contact_damping;
    coupling.parameters.mechanical.tangentialDrag = 0.0F;
    coupling.parameters.mechanical.maximumForce = maximum_force;
    coupling.parameters.mechanical.twoWay = PM_TRUE;
    return coupling;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context,
                                            std::uint32_t coupling_capacity) {
    constexpr PmFluidCapacity fluid_capacity{1U, 4U};
    constexpr PmClothCapacity cloth_capacity{3U, 3U, 3U};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.fluidCount = 1U;
    simulation_info.pFluidCapacities = &fluid_capacity;
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &cloth_capacity;
    simulation_info.couplingCapacity = coupling_capacity;
    FixtureObjects fixture{};
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
    const PmClothMaterialState cloth = cloth_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &liquid;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize fluid_size = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    constexpr PmDeviceSize cloth_size = sizeof(PmDomainOutputHeader) +
        3U * sizeof(PmClothVertexState);
    fixture.output = context.buffer(
        fluid_size + cloth_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.fluid_output = {fixture.output.handle, 0U, fluid_size};
    fixture.cloth_output = {fixture.output.handle, fluid_size, cloth_size};
    return fixture;
}

[[nodiscard]] PmClothInfo cloth_info(const float *inverse_masses) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(triangle.size()),
            triangle.data(),
            static_cast<std::uint32_t>(triangle_indices.size()),
            triangle_indices.data(), inverse_masses};
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, fixture.simulation, 0U,
                                    &fixture.fluid_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyClothState(context.commands, fixture.simulation, 0U,
                                &fixture.cloth_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *fluid_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *cloth_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.cloth_output.offset);
    if (fluid_header->activeCount != 1U || cloth_header->activeCount != 3U)
        throw std::runtime_error("fluid-cloth fixture output count mismatch");
    Snapshot result{};
    result.particle =
        reinterpret_cast<const PmFluidParticleOutput *>(fluid_header + 1)[0];
    const auto *cloth =
        reinterpret_cast<const PmClothVertexState *>(cloth_header + 1);
    for (std::size_t index = 0U; index < result.cloth.size(); ++index)
        result.cloth[index] = cloth[index];
    return result;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          float particle_y, const float *inverse_masses,
                          Snapshot *snapshot = nullptr) {
    PmParticleState particle{};
    particle.position[1] = particle_y;
    const Buffer input = context.buffer(
        sizeof(particle), PM_BUFFER_PARTICLE_INPUT_BIT);
    *static_cast<PmParticleState *>(input.mapped) = particle;
    context.flush(input);
    const PmBufferView particle_view{
        {input.handle, 0U, sizeof(particle)}, PM_FORMAT_PARTICLE_STATE,
        1U, sizeof(particle)};
    const PmClothInfo cloth = cloth_info(inverse_masses);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, fixture.simulation, 0U,
                                    &particle_view) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                             &cloth) == PM_SUCCESS);
    if (snapshot != nullptr) record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    if (snapshot != nullptr) *snapshot = read_snapshot(context, fixture);
}

struct QualityRun {
    Json driver{};
    Snapshot initial{};
    Snapshot first{};
    Snapshot final{};
};

[[nodiscard]] QualityRun run_quality_fixture() {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid-cloth contact features");
    FixtureObjects fixture = create_fixture(context, 1U);
    constexpr std::array<float, 3U> inverse_masses{{1.0F, 1.0F, 1.0F}};
    Snapshot initial{};
    initialize_endpoints(context, fixture, 0.01F, inverse_masses.data(),
                         &initial);

    const PmCouplingState coupling = contact_coupling(contact_stiffness);
    const PmStepInfo step{quality_timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation, &step) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    const Snapshot first = read_snapshot(context, fixture);

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 1U; index < quality_steps; ++index)
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    const Snapshot final = read_snapshot(context, fixture);
    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), initial, first, final};
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, PmFloat3 value) {
    append_float(stream, value.x);
    stream << ',';
    append_float(stream, value.y);
    stream << ',';
    append_float(stream, value.z);
}

[[nodiscard]] std::string graph_fingerprint(
    const PmCouplingState *coupling) {
    if (coupling == nullptr)
        return sha256("parallel-mater-fluid-cloth-contact-graph/v1;inactive");
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-cloth-contact-graph/v1;type="
              << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";direction=BIDIRECTIONAL"
              << ";material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";contact=";
    append_float(canonical,
                 coupling->parameters.mechanical.contactDistance);
    canonical << ";stiffness=";
    append_float(canonical, coupling->parameters.mechanical.stiffness);
    canonical << ";damping=";
    append_float(canonical, coupling->parameters.mechanical.damping);
    canonical << ";drag=";
    append_float(canonical, coupling->parameters.mechanical.tangentialDrag);
    canonical << ";maximum-force=";
    append_float(canonical, coupling->parameters.mechanical.maximumForce);
    canonical << ";two-way="
              << coupling->parameters.mechanical.twoWay;
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-cloth-contact-endpoints/v1;particle-position=";
    append_vector(canonical, {snapshot.particle.position[0],
                              snapshot.particle.position[1],
                              snapshot.particle.position[2]});
    canonical << ";particle-velocity=";
    append_vector(canonical, particle_velocity(snapshot));
    for (std::size_t index = 0U; index < snapshot.cloth.size(); ++index) {
        canonical << ";cloth=" << index << ",position=";
        append_vector(canonical, snapshot.cloth[index].position);
        canonical << ",velocity=";
        append_vector(canonical, snapshot.cloth[index].velocity);
    }
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
        "fluid-cloth-contact-graph", coupling != nullptr, graph_generation,
        graph_fingerprint(coupling)));
    resources.push_back(resource_json(
        "fluid-cloth-endpoints", true, 1U,
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
        throw std::runtime_error("unsupported: fluid-cloth contact features");
    FixtureObjects fixture = create_fixture(context, 1U);
    constexpr std::array<float, 3U> pinned{{0.0F, 0.0F, 0.0F}};
    initialize_endpoints(context, fixture, -1.0F, pinned.data());
    const PmCouplingState baseline = contact_coupling(100.0F);
    const PmCouplingState replacement = contact_coupling(150.0F);
    const PmCouplingState recovery = contact_coupling(175.0F);
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
        record_copies(context, fixture);
        PM_CHECK(context.submit() == PM_SUCCESS);
        return read_snapshot(context, fixture);
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
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}
} // namespace

std::string conformance_fluid_cloth_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "fluid-cloth-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/fluid-cloth-contact-v1.json");
    try {
        QualityRun run = run_quality_fixture();
        Json mode_flags = Json::array();
        mode_flags.push_back("CONTACT");
        Json gravity = Json::array();
        gravity.push_back(0.0);
        gravity.push_back(0.0);
        gravity.push_back(0.0);
        Json fixture = Json::object();
        fixture["id"] = "fluid-cloth-open-triangle-contact";
        fixture["timestep_seconds"] = static_cast<double>(quality_timestep);
        fixture["step_count"] = static_cast<std::uint64_t>(quality_steps);
        fixture["substeps"] = static_cast<std::uint64_t>(1U);
        fixture["gravity_m_s2"] = std::move(gravity);
        fixture["mode_flags"] = std::move(mode_flags);
        fixture["cloth_preserve_volume"] = false;
        fixture["cloth_vertices_m"] = vertices_json(triangle);
        fixture["cloth_triangle_indices"] = indices_json();
        fixture["cloth_vertex_mass_kg"] =
            static_cast<double>(cloth_vertex_mass);
        fixture["particle_initial_position_m"] =
            vector_json(PmFloat3{0.0F, 0.01F, 0.0F});
        fixture["particle_initial_velocity_m_s"] = vector_json(PmFloat3{});
        fixture["particle_mass_kg"] = static_cast<double>(particle_mass);
        fixture["contact_distance_m"] = static_cast<double>(contact_distance);
        fixture["stiffness_n_m"] = static_cast<double>(contact_stiffness);
        fixture["damping_n_s_m"] = static_cast<double>(contact_damping);
        fixture["tangential_drag_n_s_m"] = 0.0;
        fixture["maximum_force_n"] = static_cast<double>(maximum_force);
        fixture["oracle"] =
            "open-surface-signed-distance-and-reciprocal-normal-momentum";
        fixture["initial"] = snapshot_json(run.initial);
        fixture["first_step"] = snapshot_json(run.first);
        fixture["final"] = snapshot_json(run.final);
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

std::string conformance_fluid_cloth_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "fluid-cloth-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/fluid-cloth-contact-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("fluid");
        families.push_back("cloth");
        Json unrelated = Json::array();
        unrelated.push_back("fluid-cloth-endpoints");
        Json attempt = Json::object();
        attempt["fixture"] = "fluid-cloth-contact-public-lifecycle";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "fluid-cloth-contact-capacity-replacement-removal-rollback";
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
