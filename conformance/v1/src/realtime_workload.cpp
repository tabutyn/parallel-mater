// SPDX-License-Identifier: MIT
// Installed-public fixed-tick workload producer.  This translation unit uses
// only parallel_mater.h; no driver-private handles or implementation state.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {

using Clock = std::chrono::steady_clock;
using test::Buffer;
using test::Context;

constexpr PmFlags input_usage = PM_BUFFER_TRANSFER_SRC_BIT |
    PM_BUFFER_TRANSFER_DST_BIT | PM_BUFFER_GEOMETRY_INPUT_BIT |
    PM_BUFFER_PARTICLE_INPUT_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;

struct Workload {
    std::string_view id;
    std::uint32_t rigid{}, constraints{}, contacts{};
    std::vector<PmFluidCapacity> fluids;
    std::vector<PmClothCapacity> cloths;
    std::vector<PmSoftBodyCapacity> soft_bodies;
    std::vector<PmRopeCapacity> ropes;
    std::vector<PmSmokeCapacity> smoke;
    std::uint32_t couplings{};
};

[[nodiscard]] Workload workload(std::string_view id) {
    if (id == "rigid-wall")
        return {id, 386U, 0U, 65'536U};
    if (id == "fluid-rigid")
        return {id, 65U, 0U, 65'536U,
                {{30'000U, 128U}}, {}, {}, {}, {}, 1U};
    if (id == "deformable-mixed")
        return {id, 0U, 0U, 65'536U, {},
                {{289U, 1'536U, 1'536U}, {289U, 1'536U, 1'536U}},
                {{912U, 130'033U, 4U, 12U}}, {{72U}}, {}, 3U};
    if (id == "smoke-coupled")
        return {id, 4U, 0U, 16'384U, {}, {}, {}, {},
                {{4'500U, 1U}}, 1U};
    if (id == "six-family-mixed")
        return {id, 32U, 0U, 32'768U,
                {{2'048U, 128U}, {2'048U, 128U}},
                {{1'024U, 5'766U, 5'766U}},
                {{512U, 1'024U, 4U, 12U}}, {{512U}},
                {{2'048U, 1U}}, 16U};
    throw std::runtime_error("unknown realtime workload: " + std::string(id));
}

[[nodiscard]] PmPhysicalDeviceFeatures required_features(const Workload &value) {
    PmPhysicalDeviceFeatures result{};
    result.rigidBodies = value.rigid != 0U;
    result.particleLiquids = !value.fluids.empty();
    result.liquidRigidCoupling = value.id == "fluid-rigid" ||
        value.id == "six-family-mixed";
    result.liquidLiquidCoupling = value.fluids.size() > 1U;
    result.cloth = !value.cloths.empty();
    result.rope = !value.ropes.empty();
    result.softBodies = !value.soft_bodies.empty();
    result.smoke = !value.smoke.empty();
    result.domainCouplings = value.couplings != 0U;
    result.timestampQueries = PM_TRUE;
    result.statisticsQueries = PM_TRUE;
    return result;
}

[[nodiscard]] double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

[[nodiscard]] Json distribution(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("workload has no timing samples");
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double fraction) {
        const double rank = std::ceil(fraction * static_cast<double>(values.size()));
        const auto index = static_cast<std::size_t>(std::max(1.0, rank) - 1.0);
        return values[std::min(index, values.size() - 1U)];
    };
    Json result = Json::object();
    result["p50"] = percentile(0.50);
    result["p95"] = percentile(0.95);
    result["p99"] = percentile(0.99);
    result["maximum"] = values.back();
    return result;
}

[[nodiscard]] Json samples_json(const std::vector<double> &values) {
    Json result = Json::array();
    for (double value : values) result.push_back(value);
    return result;
}

[[nodiscard]] Json counts_json(const Workload &value) {
    Json result = Json::object();
    const auto set = [&](const char *name, std::uint64_t count) {
        result[name] = count;
    };
    if (value.id == "rigid-wall") {
        set("rigid_bodies_live", 386U);
        set("rigid_bodies_capacity", 386U);
        set("constraints_live", 0U);
        set("constraints_capacity", 0U);
        set("contact_capacity", 65'536U);
    } else if (value.id == "fluid-rigid") {
        set("rigid_bodies_live", 65U);
        set("rigid_bodies_capacity", 65U);
        set("fluid_systems", 1U);
        set("fluid_particles_live", 30'000U);
        set("fluid_particles_capacity", 30'000U);
        set("neighbors_per_particle_capacity", 128U);
        set("couplings_live", 1U);
        set("couplings_capacity", 1U);
        set("contact_capacity", 65'536U);
    } else if (value.id == "deformable-mixed") {
        set("cloth_systems", 2U);
        set("cloth_vertices_live", 578U);
        set("cloth_vertices_capacity", 578U);
        set("cloth_triangle_indices_live", 3'072U);
        set("cloth_triangle_indices_capacity", 3'072U);
        set("soft_body_systems", 1U);
        set("soft_nodes_live", 912U);
        set("soft_nodes_capacity", 912U);
        set("soft_bonds_live", 130'033U);
        set("soft_bonds_capacity", 130'033U);
        set("rope_systems", 1U);
        set("rope_nodes_live", 72U);
        set("rope_nodes_capacity", 72U);
        set("couplings_live", 3U);
        set("couplings_capacity", 3U);
        set("contact_capacity", 65'536U);
    } else if (value.id == "smoke-coupled") {
        set("smoke_systems", 1U);
        set("smoke_particles_live", 4'500U);
        set("smoke_particles_capacity", 4'500U);
        set("rigid_bodies_live", 4U);
        set("rigid_bodies_capacity", 4U);
        set("couplings_live", 1U);
        set("couplings_capacity", 1U);
        set("contact_capacity", 16'384U);
    } else {
        set("rigid_bodies_live", 32U);
        set("rigid_bodies_capacity", 32U);
        set("fluid_systems", 2U);
        set("fluid_particles_live", 4'096U);
        set("fluid_particles_capacity", 4'096U);
        set("cloth_systems", 1U);
        set("cloth_vertices_live", 1'024U);
        set("cloth_vertices_capacity", 1'024U);
        set("soft_body_systems", 1U);
        set("soft_nodes_live", 512U);
        set("soft_nodes_capacity", 512U);
        set("rope_systems", 1U);
        set("rope_nodes_live", 512U);
        set("rope_nodes_capacity", 512U);
        set("smoke_systems", 1U);
        set("smoke_particles_live", 2'048U);
        set("smoke_particles_capacity", 2'048U);
        set("couplings_live", 16U);
        set("couplings_capacity", 16U);
        set("contact_capacity", 32'768U);
    }
    return result;
}

struct WorkloadRuntime {
    Context context;
    Workload definition;
    PmGeometry geometry{};
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    PmQueryPool timestamps{};
    Buffer particle_input{};
    std::vector<PmRigidBodyInfo> rigid;
    std::vector<PmParticleState> particles;
    std::vector<PmFloat3> cloth_vertices;
    std::vector<std::uint32_t> cloth_indices;
    std::vector<float> cloth_inverse_masses;
    std::vector<PmClothInfo> cloth;
    std::vector<PmFloat3> soft_nodes;
    std::vector<PmSoftBodyBond> soft_bonds;
    std::vector<float> soft_inverse_masses;
    std::vector<PmFloat3> soft_surface_vertices;
    std::vector<std::uint32_t> soft_surface_indices;
    std::vector<PmSoftBodySurfaceBinding> soft_surface_bindings;
    std::vector<PmSoftBodyInfo> soft;
    std::vector<PmFloat3> rope_points;
    std::vector<PmRopeInfo> rope;
    std::vector<PmSmokeInfo> smoke;
    std::vector<PmCouplingStateInfo> couplings;
    float timestamp_period_ns{};

    explicit WorkloadRuntime(Workload configured)
        : definition(std::move(configured)) {}

    ~WorkloadRuntime() {
        if (context.device != nullptr) {
            (void)pmDeviceWaitIdle(context.device);
            if (timestamps != nullptr)
                pmDestroyQueryPool(context.device, timestamps);
            if (pipeline != nullptr)
                pmDestroyPhysicsPipeline(context.device, pipeline);
            if (simulation != nullptr)
                pmDestroySimulation(context.device, simulation);
            if (geometry != nullptr)
                pmDestroyGeometry(context.device, geometry);
        }
    }

    void open() {
        if (!context.open(required_features(definition)))
            throw std::runtime_error("no physical device supports workload " +
                                     std::string(definition.id));
        std::uint32_t queue_family_count{};
        PM_CHECK(pmGetPhysicalDeviceQueueFamilies(
            context.physical, &queue_family_count, nullptr) == PM_SUCCESS);
        std::vector<PmQueueFamilyProperties> queue_families(queue_family_count);
        PM_CHECK(pmGetPhysicalDeviceQueueFamilies(context.physical,
            &queue_family_count, queue_families.data()) == PM_SUCCESS);
        if (queue_families.empty())
            throw std::runtime_error("physical device has no queue family");
        timestamp_period_ns = queue_families.front().timestampPeriodNanoseconds;
        if (!(timestamp_period_ns > 0.0F))
            throw std::runtime_error("physical device has no timestamp period");
        create_geometry();
        create_simulation();
        create_pipeline();
        create_queries();
        initialize_resources();
    }

    void create_geometry() {
        if (definition.rigid == 0U) return;
        constexpr std::array<PmFloat3, 8U> vertices{{
            {-0.5F,-0.5F,-0.5F},{0.5F,-0.5F,-0.5F},
            {0.5F,0.5F,-0.5F},{-0.5F,0.5F,-0.5F},
            {-0.5F,-0.5F,0.5F},{0.5F,-0.5F,0.5F},
            {0.5F,0.5F,0.5F},{-0.5F,0.5F,0.5F}}};
        constexpr std::array<std::uint32_t, 36U> indices{{
            0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
            3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5}};
        const PmDeviceSize vertex_bytes = sizeof(vertices);
        const PmDeviceSize index_bytes = sizeof(indices);
        auto input = context.buffer(vertex_bytes + index_bytes, input_usage);
        std::copy(vertices.begin(), vertices.end(),
                  static_cast<PmFloat3 *>(input.mapped));
        auto *index_output = reinterpret_cast<std::uint32_t *>(
            static_cast<std::byte *>(input.mapped) + vertex_bytes);
        std::copy(indices.begin(), indices.end(), index_output);
        context.flush(input);
        PmGeometryCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
        info.vertices = {{input.handle, 0U, vertex_bytes},
                         PM_FORMAT_R32G32B32_SFLOAT,
                         static_cast<std::uint32_t>(vertices.size()),
                         sizeof(PmFloat3)};
        info.indices = {{input.handle, vertex_bytes, index_bytes},
                        PM_FORMAT_R32_UINT,
                        static_cast<std::uint32_t>(indices.size()),
                        sizeof(std::uint32_t)};
        PM_CHECK(pmCreateGeometry(context.device, &info, &geometry) == PM_SUCCESS);
        PmMemoryRequirements requirements{};
        pmGetGeometryMemoryRequirements(context.device, geometry, &requirements);
        PM_CHECK(pmBindGeometryMemory(context.device, geometry,
                                     context.allocate(requirements), 0U) == PM_SUCCESS);
        context.begin();
        PM_CHECK(pmCmdBuildGeometry(context.commands, geometry) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }

    void create_simulation() {
        PmSimulationCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
        info.rigidBodyCapacity = definition.rigid;
        info.constraintCapacity = definition.constraints;
        info.contactCapacity = definition.contacts;
        info.fluidCount = static_cast<std::uint32_t>(definition.fluids.size());
        info.pFluidCapacities = definition.fluids.data();
        info.clothCount = static_cast<std::uint32_t>(definition.cloths.size());
        info.pClothCapacities = definition.cloths.data();
        info.softBodyCount = static_cast<std::uint32_t>(definition.soft_bodies.size());
        info.pSoftBodyCapacities = definition.soft_bodies.data();
        info.ropeCount = static_cast<std::uint32_t>(definition.ropes.size());
        info.pRopeCapacities = definition.ropes.data();
        info.smokeCount = static_cast<std::uint32_t>(definition.smoke.size());
        info.pSmokeCapacities = definition.smoke.data();
        info.couplingCapacity = definition.couplings;
        PM_CHECK(pmCreateSimulation(context.device, &info, &simulation) == PM_SUCCESS);
        PmSimulationMemoryRequirements requirements{};
        pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
        const PmSimulationMemoryBinding binding{
            context.allocate(requirements.persistent), 0U,
            context.allocate(requirements.scratch), 0U};
        PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) ==
                 PM_SUCCESS);
    }

    void create_pipeline() {
        PmRigidSolverState rigid_state{};
        rigid_state.rigidIterations = 8U;
        std::vector<PmLiquidMaterialState> liquids(definition.fluids.size());
        for (auto &material : liquids) {
            material.restDensity = 1000.0F;
            material.particleRadius = 0.025F;
            material.supportRadius = 0.09F;
            material.restParticleVolume = 0.000125F;
            material.pressureStiffness = 50.0F;
            material.maximumSpeed = 2.6F;
            material.densityIterations = 6U;
        }
        std::vector<PmClothMaterialState> cloth_materials(definition.cloths.size());
        for (auto &material : cloth_materials) {
            material.vertexMass = 1.0F;
            material.thickness = 0.01F;
            material.solverIterations = 8U;
            material.fracturePersistenceSubsteps = 1U;
        }
        std::vector<PmSoftBodyMaterialState> soft_materials(
            definition.soft_bodies.size());
        for (auto &material : soft_materials) {
            material.nodeMass = 1.0F;
            material.nodeRadius = 0.01F;
            material.shapeMatchingStiffness = 0.5F;
            material.maximumProjectionFraction = 0.2F;
            material.constraintVelocityResponse = 1.0F;
            material.maximumSpeed = 1.0F;
            material.solverIterations = 8U;
        }
        std::vector<PmRopeMaterialState> rope_materials(definition.ropes.size());
        for (auto &material : rope_materials) {
            material.nodeSpacing = 0.05F;
            material.radius = 0.025F;
            material.mass = 1.0F;
            material.maximumSubstepTimestep = 1.0F / 60.0F;
            material.maximumSpeed = 1.4F;
            material.solverIterations = 8U;
        }
        std::vector<PmSmokeMaterialState> smoke_materials(definition.smoke.size());
        for (auto &material : smoke_materials) {
            material.particleRadius = 0.025F;
            material.response = 1.0F;
            material.restNumberDensity = 12.0F;
            material.pressureStiffness = 2.0F;
            material.maximumSpeed = 2.0F;
        }
        PmPhysicsPipelineCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
        info.pRigidState = definition.rigid == 0U ? nullptr : &rigid_state;
        info.fluidMaterialCount = static_cast<std::uint32_t>(liquids.size());
        info.pFluidMaterials = liquids.data();
        info.clothMaterialCount = static_cast<std::uint32_t>(cloth_materials.size());
        info.pClothMaterials = cloth_materials.data();
        info.softBodyMaterialCount = static_cast<std::uint32_t>(soft_materials.size());
        info.pSoftBodyMaterials = soft_materials.data();
        info.ropeMaterialCount = static_cast<std::uint32_t>(rope_materials.size());
        info.pRopeMaterials = rope_materials.data();
        info.smokeMaterialCount = static_cast<std::uint32_t>(smoke_materials.size());
        info.pSmokeMaterials = smoke_materials.data();
        PM_CHECK(pmCreatePhysicsPipeline(context.device, &info, &pipeline) == PM_SUCCESS);
    }

    void create_queries() {
        const PmQueryPoolCreateInfo info{
            PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr,
            PM_QUERY_TIMESTAMP, 2U};
        PM_CHECK(pmCreateQueryPool(context.device, &info, &timestamps) == PM_SUCCESS);
    }

    void make_rigid() {
        rigid.resize(definition.rigid);
        for (std::uint32_t index = 0U; index < definition.rigid; ++index) {
            auto &body = rigid[index];
            body.active = PM_TRUE;
            body.motionType = index == 0U ? PM_MOTION_STATIC : PM_MOTION_DYNAMIC;
            body.geometry = geometry;
            body.state.position[0] =
                ((!definition.fluids.empty() || !definition.smoke.empty()) &&
                 index != 0U ? 25.0F : 0.0F) +
                static_cast<float>(index % 16U) * 3.0F;
            body.state.position[1] = index == 0U ? -1.0F
                : 1.0F + static_cast<float>((index / 16U) % 24U) * 3.0F;
            body.state.position[2] = static_cast<float>(index / 384U) * 3.0F;
            body.state.orientation[3] = 1.0F;
            body.mass = 1.0F;
            body.inertiaDiagonal[0] = body.inertiaDiagonal[1] =
                body.inertiaDiagonal[2] = 1.0F / 6.0F;
            body.friction = 0.5F;
            body.maximumLinearSpeed = body.maximumAngularSpeed = 20.0F;
            body.collisionMargin = 0.005F;
        }
    }

    void make_fluids() {
        std::size_t total{};
        for (const auto &capacity : definition.fluids)
            total += capacity.particleCapacity;
        if (total == 0U) return;
        particles.resize(total);
        for (std::size_t index = 0U; index < total; ++index) {
            auto &particle = particles[index];
            particle.position[0] = static_cast<float>(index % 64U) * 0.2F;
            particle.position[1] = 4.0F +
                static_cast<float>((index / 64U) % 64U) * 0.2F;
            particle.position[2] = static_cast<float>(index / (64U * 64U)) * 0.2F;
            particle.temperatureCelsius = 20.0F;
        }
        particle_input = context.buffer(
            static_cast<PmDeviceSize>(particles.size() * sizeof(PmParticleState)),
            input_usage);
        std::copy(particles.begin(), particles.end(),
                  static_cast<PmParticleState *>(particle_input.mapped));
        context.flush(particle_input);
    }

    void make_cloth() {
        std::size_t vertex_offset{}, index_offset{};
        cloth.resize(definition.cloths.size());
        for (std::size_t system = 0U; system < definition.cloths.size(); ++system) {
            const auto capacity = definition.cloths[system];
            const auto side = static_cast<std::uint32_t>(
                std::sqrt(static_cast<double>(capacity.vertexCapacity)));
            if (side * side != capacity.vertexCapacity)
                throw std::runtime_error("cloth workload requires square grids");
            const std::size_t base_vertex = cloth_vertices.size();
            for (std::uint32_t y = 0U; y < side; ++y)
                for (std::uint32_t x = 0U; x < side; ++x)
                    cloth_vertices.push_back({
                        static_cast<float>(x) * 0.05F,
                        10.0F + static_cast<float>(system),
                        static_cast<float>(y) * 0.05F});
            for (std::uint32_t y = 0U; y + 1U < side; ++y)
                for (std::uint32_t x = 0U; x + 1U < side; ++x) {
                    const auto a = y * side + x;
                    const auto b = a + 1U;
                    const auto c = a + side;
                    const auto d = c + 1U;
                    cloth_indices.insert(cloth_indices.end(), {a, c, b, b, c, d});
                }
            cloth_inverse_masses.insert(cloth_inverse_masses.end(),
                                        capacity.vertexCapacity, 0.0F);
            auto &info = cloth[system];
            info.active = PM_TRUE;
            info.materialIndex = static_cast<std::uint32_t>(system);
            info.vertexCount = capacity.vertexCapacity;
            info.pVertices = cloth_vertices.data() + base_vertex;
            info.triangleIndexCount = capacity.triangleIndexCapacity;
            info.pTriangleIndices = cloth_indices.data() + index_offset;
            info.pInverseMasses = cloth_inverse_masses.data() + vertex_offset;
            vertex_offset += capacity.vertexCapacity;
            index_offset += capacity.triangleIndexCapacity;
        }
        // Vector growth can invalidate pointers assigned above.
        vertex_offset = index_offset = 0U;
        for (std::size_t system = 0U; system < cloth.size(); ++system) {
            cloth[system].pVertices = cloth_vertices.data() + vertex_offset;
            cloth[system].pTriangleIndices = cloth_indices.data() + index_offset;
            cloth[system].pInverseMasses = cloth_inverse_masses.data() + vertex_offset;
            vertex_offset += definition.cloths[system].vertexCapacity;
            index_offset += definition.cloths[system].triangleIndexCapacity;
        }
    }

    void make_soft_bodies() {
        soft.resize(definition.soft_bodies.size());
        std::size_t node_offset{}, bond_offset{}, surface_vertex_offset{},
            surface_index_offset{};
        for (std::size_t system = 0U; system < definition.soft_bodies.size(); ++system) {
            const auto capacity = definition.soft_bodies[system];
            const std::size_t base_node = soft_nodes.size();
            const auto side = static_cast<std::uint32_t>(std::ceil(std::cbrt(
                static_cast<double>(capacity.nodeCapacity))));
            for (std::uint32_t node = 0U; node < capacity.nodeCapacity; ++node)
                soft_nodes.push_back({
                    static_cast<float>(node % side) * 0.04F,
                    12.0F + static_cast<float>((node / side) % side) * 0.04F,
                    static_cast<float>(node / (side * side)) * 0.04F});
            for (std::uint32_t bond = 0U; bond < capacity.bondCapacity; ++bond) {
                const auto first = bond % capacity.nodeCapacity;
                auto second = (first + 1U + bond / capacity.nodeCapacity) %
                    capacity.nodeCapacity;
                if (second == first) second = (second + 1U) % capacity.nodeCapacity;
                const auto &a = soft_nodes[base_node + first];
                const auto &b = soft_nodes[base_node + second];
                const float x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
                soft_bonds.push_back({first, second, std::sqrt(x*x + y*y + z*z)});
            }
            soft_inverse_masses.insert(soft_inverse_masses.end(),
                                       capacity.nodeCapacity, 1.0F);
            const std::size_t surface_base = soft_surface_vertices.size();
            const std::array<std::uint32_t, 4U> surface_nodes{
                0U, 1U, side, side * side};
            for (const auto node : surface_nodes)
                soft_surface_vertices.push_back(soft_nodes[base_node + node]);
            soft_surface_indices.insert(soft_surface_indices.end(), {
                0U, 2U, 1U, 0U, 1U, 3U,
                0U, 3U, 2U, 1U, 2U, 3U});
            for (std::uint32_t vertex = 0U; vertex < 4U; ++vertex) {
                PmSoftBodySurfaceBinding binding{};
                binding.nodes[0] = surface_nodes[vertex];
                binding.weights[0] = 1.0F;
                soft_surface_bindings.push_back(binding);
            }
            auto &info = soft[system];
            info.active = PM_TRUE;
            info.materialIndex = static_cast<std::uint32_t>(system);
            info.nodeCount = capacity.nodeCapacity;
            info.pNodes = soft_nodes.data() + base_node;
            info.bondCount = capacity.bondCapacity;
            info.pBonds = soft_bonds.data() + bond_offset;
            info.pInverseMasses = soft_inverse_masses.data() + node_offset;
            info.surfaceVertexCount = 4U;
            info.pSurfaceVertices = soft_surface_vertices.data() + surface_base;
            info.surfaceTriangleIndexCount = 12U;
            info.pSurfaceTriangleIndices =
                soft_surface_indices.data() + surface_index_offset;
            info.pSurfaceBindings =
                soft_surface_bindings.data() + surface_vertex_offset;
            node_offset += capacity.nodeCapacity;
            bond_offset += capacity.bondCapacity;
            surface_vertex_offset += 4U;
            surface_index_offset += 12U;
        }
        node_offset = bond_offset = surface_vertex_offset =
            surface_index_offset = 0U;
        for (std::size_t system = 0U; system < soft.size(); ++system) {
            soft[system].pNodes = soft_nodes.data() + node_offset;
            soft[system].pBonds = soft_bonds.data() + bond_offset;
            soft[system].pInverseMasses = soft_inverse_masses.data() + node_offset;
            soft[system].pSurfaceVertices =
                soft_surface_vertices.data() + surface_vertex_offset;
            soft[system].pSurfaceTriangleIndices =
                soft_surface_indices.data() + surface_index_offset;
            soft[system].pSurfaceBindings =
                soft_surface_bindings.data() + surface_vertex_offset;
            node_offset += definition.soft_bodies[system].nodeCapacity;
            bond_offset += definition.soft_bodies[system].bondCapacity;
            surface_vertex_offset += 4U;
            surface_index_offset += 12U;
        }
    }

    void make_ropes() {
        rope.resize(definition.ropes.size());
        std::size_t offset{};
        for (std::size_t system = 0U; system < definition.ropes.size(); ++system) {
            const auto count = definition.ropes[system].nodeCapacity;
            const float length = std::nextafter(
                static_cast<float>(count - 1U) * 0.05F, 0.0F);
            rope_points.push_back({0.0F, 10.0F, 0.0F});
            rope_points.push_back({length, 10.0F, 0.0F});
            auto &info = rope[system];
            info.active = PM_TRUE;
            info.materialIndex = static_cast<std::uint32_t>(system);
            info.pointCount = 2U;
            info.pCenterline = rope_points.data() + offset;
            offset += 2U;
        }
        offset = 0U;
        for (std::size_t system = 0U; system < rope.size(); ++system) {
            rope[system].pCenterline = rope_points.data() + offset;
            offset += 2U;
        }
    }

    void make_smoke() {
        smoke.resize(definition.smoke.size());
        for (std::size_t system = 0U; system < smoke.size(); ++system) {
            auto &info = smoke[system];
            info.active = PM_TRUE;
            info.materialIndex = static_cast<std::uint32_t>(system);
            info.particleCapacity = definition.smoke[system].particleCapacity;
            info.emitterCenter = {0.0F, 16.0F, 0.0F};
            info.emitterHalfExtents = {4.0F, 4.0F};
            info.initialVelocity = {1.6F, 0.0F, 0.0F};
            info.wind = info.initialVelocity;
            info.particlesPerSecond = static_cast<float>(
                definition.smoke[system].particleCapacity);
            info.lifetime = 10'000.0F;
        }
    }

    static PmCouplingStateInfo mechanical(PmCouplingType type,
                                           std::uint32_t first = 0U,
                                           std::uint32_t second = 0U) {
        PmCouplingStateInfo result{};
        result.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
        result.type = type;
        result.first = first;
        result.second = second;
        result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
        result.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
        result.parameters.mechanical.contactDistance = 0.02F;
        result.parameters.mechanical.stiffness = 100.0F;
        result.parameters.mechanical.damping = 1.0F;
        result.parameters.mechanical.maximumForce = 10.0F;
        result.parameters.mechanical.maximumAcceleration = 10.0F;
        result.parameters.mechanical.twoWay = PM_TRUE;
        return result;
    }

    static PmCouplingStateInfo smoke_coupling(PmCouplingType type) {
        auto result = mechanical(type);
        result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_DRAG_BIT;
        result.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
        result.parameters.smoke.windDrag = 1.0F;
        result.parameters.smoke.maximumWindAcceleration = 10.0F;
        result.parameters.smoke.airDensity = 1.5F;
        result.parameters.smoke.dragCoefficient = 2.0F;
        result.parameters.smoke.contactDistance = 0.04F;
        result.parameters.smoke.tracerContact = PM_TRUE;
        return result;
    }

    static PmCouplingStateInfo liquid_rigid(std::uint32_t fluid = 0U) {
        auto result = mechanical(PM_COUPLING_LIQUID_RIGID, fluid);
        result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_CONTAINMENT_BIT;
        result.parameters.mechanical.stiffness = 0.0F;
        result.parameters.mechanical.damping = 0.0F;
        result.parameters.mechanical.maximumForce = 0.0F;
        return result;
    }

    static PmCouplingStateInfo liquid_cloth(std::uint32_t fluid) {
        auto result = mechanical(PM_COUPLING_LIQUID_CLOTH, fluid);
        result.parameters.mechanical.maximumAcceleration = 0.0F;
        return result;
    }

    static PmCouplingStateInfo liquid_pair() {
        PmCouplingStateInfo result{};
        result.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
        result.type = PM_COUPLING_LIQUID_LIQUID;
        result.first = 0U;
        result.second = 1U;
        result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_PRESSURE_BIT |
            PM_INTERACTION_MODE_VISCOSITY_BIT;
        result.materialProfile = PM_INTERACTION_MATERIAL_EQUAL;
        result.parameters.mechanical.twoWay = PM_TRUE;
        return result;
    }

    static PmCouplingStateInfo liquid_smoke(std::uint32_t fluid) {
        PmCouplingStateInfo result{};
        result.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
        result.type = PM_COUPLING_LIQUID_SMOKE;
        result.first = fluid;
        result.modeFlags = PM_INTERACTION_MODE_DRAG_BIT |
            PM_INTERACTION_MODE_THERMAL_BIT | PM_INTERACTION_MODE_PHASE_BIT;
        result.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
        result.parameters.thermal.heaterOrientation[3] = 1.0F;
        result.parameters.thermal.heaterHalfExtents[0] = 0.1F;
        result.parameters.thermal.heaterHalfExtents[1] = 0.1F;
        result.parameters.thermal.heaterTemperature = 120.0F;
        result.parameters.thermal.boilingTemperature = 100.0F;
        result.parameters.thermal.heatTransferRate = 0.2F;
        result.parameters.thermal.windDrag = 0.1F;
        result.parameters.thermal.steamRiseSpeed = 1.0F;
        return result;
    }

    static PmCouplingStateInfo rope_soft() {
        auto result = mechanical(PM_COUPLING_ROPE_SOFT_BODY);
        result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_ATTACHMENT_BIT;
        result.parameters.mechanical.friction = 0.2F;
        result.parameters.attachment.supportRadiusScale = 2.0F;
        result.parameters.attachment.contactSupportRadiusScale = 1.0F;
        result.parameters.attachment.maximumAcceleration = 10.0F;
        return result;
    }

    static PmCouplingStateInfo rope_cloth() {
        PmCouplingStateInfo result{};
        result.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
        result.type = PM_COUPLING_ROPE_CLOTH;
        result.modeFlags = PM_INTERACTION_MODE_ATTACHMENT_BIT;
        result.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
        result.parameters.attachment.lastVertex = UINT32_MAX;
        result.parameters.attachment.effectiveMass = 1.0F;
        result.parameters.attachment.maximumAcceleration = 10.0F;
        return result;
    }

    void make_couplings() {
        if (definition.id == "fluid-rigid") {
            couplings.push_back(liquid_rigid());
        } else if (definition.id == "deformable-mixed") {
            couplings.push_back(mechanical(PM_COUPLING_SOFT_BODY_CLOTH));
            couplings.push_back(rope_soft());
            couplings.push_back(rope_cloth());
        } else if (definition.id == "smoke-coupled") {
            couplings.push_back(smoke_coupling(PM_COUPLING_SMOKE_RIGID));
        } else if (definition.id == "six-family-mixed") {
            couplings.push_back(liquid_rigid(0U));
            couplings.push_back(liquid_rigid(1U));
            couplings.push_back(liquid_pair());
            couplings.push_back(liquid_cloth(0U));
            couplings.push_back(liquid_cloth(1U));
            couplings.push_back(mechanical(PM_COUPLING_LIQUID_ROPE, 0U));
            couplings.push_back(mechanical(PM_COUPLING_LIQUID_ROPE, 1U));
            couplings.push_back(mechanical(PM_COUPLING_LIQUID_SOFT_BODY, 0U));
            couplings.push_back(mechanical(PM_COUPLING_LIQUID_SOFT_BODY, 1U));
            couplings.push_back(liquid_smoke(0U));
            couplings.push_back(smoke_coupling(PM_COUPLING_SMOKE_ROPE));
            couplings.push_back(mechanical(PM_COUPLING_SOFT_BODY_CLOTH));
            couplings.push_back(rope_soft());
            couplings.push_back(rope_cloth());
            couplings.push_back(smoke_coupling(PM_COUPLING_SMOKE_RIGID));
            couplings.push_back(smoke_coupling(PM_COUPLING_SMOKE_CLOTH));
        }
        if (couplings.size() != definition.couplings)
            throw std::runtime_error("workload coupling count mismatch");
    }

    void initialize_resources() {
        make_rigid();
        make_fluids();
        make_cloth();
        make_soft_bodies();
        make_ropes();
        make_smoke();
        make_couplings();
        const auto update = [&](const char *label, auto encode) {
            context.begin();
            PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) ==
                     PM_SUCCESS);
            encode();
            const PmResult execution = context.submit();
            if (execution != PM_SUCCESS)
                throw std::runtime_error(std::string("workload ") + label +
                    " submission failed: " +
                    std::to_string(static_cast<int>(execution)));
        };
        if (!rigid.empty()) update("rigid initialization", [&] {
            PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U,
                static_cast<std::uint32_t>(rigid.size()), rigid.data()) == PM_SUCCESS);
        });
        if (!definition.fluids.empty()) update("fluid initialization", [&] {
            std::size_t offset{};
            for (std::uint32_t index = 0U; index < definition.fluids.size(); ++index) {
                const auto count = definition.fluids[index].particleCapacity;
                const PmDeviceSize bytes =
                    static_cast<PmDeviceSize>(count) * sizeof(PmParticleState);
                const PmBufferView view{{particle_input.handle,
                    static_cast<PmDeviceSize>(offset * sizeof(PmParticleState)), bytes},
                    PM_FORMAT_PARTICLE_STATE, count, sizeof(PmParticleState)};
                PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                                 &view) == PM_SUCCESS);
                offset += count;
            }
        });
        if (!cloth.empty()) update("cloth initialization", [&] {
            PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U,
                static_cast<std::uint32_t>(cloth.size()), cloth.data()) == PM_SUCCESS);
        });
        if (!soft.empty()) update("soft-body initialization", [&] {
            PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0U,
                static_cast<std::uint32_t>(soft.size()), soft.data()) == PM_SUCCESS);
        });
        if (!rope.empty()) update("rope initialization", [&] {
            const auto result = pmCmdWriteRopes(context.commands, simulation, 0U,
                static_cast<std::uint32_t>(rope.size()), rope.data());
            if (result != PM_SUCCESS)
                throw std::runtime_error("rope command recording failed: " +
                    std::to_string(static_cast<int>(result)));
        });
        if (!smoke.empty()) update("smoke initialization", [&] {
            PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U,
                static_cast<std::uint32_t>(smoke.size()), smoke.data()) == PM_SUCCESS);
        });
        if (!couplings.empty()) update("coupling initialization", [&] {
            PM_CHECK(pmCmdSetCouplings2(context.commands, simulation,
                static_cast<std::uint32_t>(couplings.size()), couplings.data()) ==
                PM_SUCCESS);
        });
    }

    struct TickTiming {
        double cpu_ms{};
        double gpu_ms{};
        double physics_ms{};
        double frame_ms{};
        PmResult result{PM_ERROR_INVALID_STATE};
    };

    [[nodiscard]] TickTiming tick() {
        const auto frame_start = Clock::now();
        context.begin();
        PM_CHECK(pmCmdResetQueryPool(context.commands, timestamps, 0U, 2U) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdWriteTimestamp(context.commands, PM_STAGE_ALL_COMMANDS_BIT,
                                     timestamps, 0U) == PM_SUCCESS);
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        const PmStepInfo step{1.0F / 60.0F, 1U, {}, 0U};
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
        PM_CHECK(pmCmdWriteTimestamp(context.commands, PM_STAGE_ALL_COMMANDS_BIT,
                                     timestamps, 1U) == PM_SUCCESS);
        const auto encoded = Clock::now();
        const PmResult result = context.submit();
        const auto completed = Clock::now();
        std::uint64_t values[2]{};
        if (result == PM_SUCCESS)
            PM_CHECK(pmGetQueryPoolResults(context.device, timestamps, 0U, 2U,
                sizeof(values), values, sizeof(values[0]),
                PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        const double gpu_ms = result == PM_SUCCESS && values[1] >= values[0]
            ? static_cast<double>(values[1] - values[0]) *
                static_cast<double>(timestamp_period_ns) / 1'000'000.0
            : 0.0;
        return {milliseconds(encoded - frame_start), gpu_ms,
                milliseconds(completed - frame_start),
                milliseconds(completed - frame_start), result};
    }
};

} // namespace

std::vector<std::string> conformance_realtime_workload_ids() {
    return {"rigid-wall", "fluid-rigid", "deformable-mixed",
            "smoke-coupled", "six-family-mixed"};
}

std::string conformance_realtime_workload(
    const std::string &workload_id, std::uint32_t warmup_ticks,
    std::uint32_t measured_ticks) {
    if (warmup_ticks == 0U || measured_ticks == 0U)
        throw std::runtime_error("workload tick counts must be positive");
    WorkloadRuntime runtime{workload(workload_id)};
    runtime.open();
    for (std::uint32_t tick = 0U; tick < warmup_ticks; ++tick) {
        const auto timing = runtime.tick();
        if (timing.result != PM_SUCCESS)
            throw std::runtime_error("workload warmup tick " +
                std::to_string(tick) + " failed: " +
                std::to_string(static_cast<int>(timing.result)));
    }
    std::vector<double> cpu, gpu, physics, frame;
    cpu.reserve(measured_ticks);
    gpu.reserve(measured_ticks);
    physics.reserve(measured_ticks);
    frame.reserve(measured_ticks);
    const auto measured_start = Clock::now();
    for (std::uint32_t tick = 0U; tick < measured_ticks; ++tick) {
        const auto timing = runtime.tick();
        if (timing.result != PM_SUCCESS)
            throw std::runtime_error("workload measured tick " +
                std::to_string(tick) + " failed: " +
                std::to_string(static_cast<int>(timing.result)));
        cpu.push_back(timing.cpu_ms);
        gpu.push_back(timing.gpu_ms);
        physics.push_back(timing.physics_ms);
        frame.push_back(timing.frame_ms);
    }
    const double wall_seconds = std::chrono::duration<double>(
        Clock::now() - measured_start).count();
    Json raw = Json::object();
    raw["cpu_ms"] = samples_json(cpu);
    raw["gpu_ms"] = samples_json(gpu);
    raw["physics_ms"] = samples_json(physics);
    raw["frame_ms"] = samples_json(frame);
    Json timings = Json::object();
    timings["cpu_ms"] = distribution(cpu);
    timings["gpu_ms"] = distribution(gpu);
    timings["physics_ms"] = distribution(physics);
    timings["frame_ms"] = distribution(frame);
    Json result = Json::object();
    result["schema"] = "parallel-mater-realtime-workload-capture/v1";
    result["source"] = "installed-public-api";
    result["driver_private_state_used"] = false;
    result["id"] = workload_id;
    result["counts"] = counts_json(runtime.definition);
    result["warmup_ticks"] = static_cast<std::uint64_t>(warmup_ticks);
    result["measured_ticks"] = static_cast<std::uint64_t>(measured_ticks);
    result["failed_ticks"] = static_cast<std::uint64_t>(0U);
    result["simulated_to_wall_time_ratio"] = wall_seconds > 0.0
        ? (static_cast<double>(measured_ticks) / 60.0) / wall_seconds : 0.0;
    result["timings"] = std::move(timings);
    result["raw_samples"] = std::move(raw);
    return result.serialize();
}

} // namespace parallel_mater::conformance
