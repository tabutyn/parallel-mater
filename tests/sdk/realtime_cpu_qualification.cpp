// SPDX-License-Identifier: MIT
// F8-23 installed-public-SDK CPU qualification capture.
#include <parallel_mater/parallel_mater.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#define CHECK(value) do { if (!(value)) {                                  \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value);             \
    std::abort();                                                          \
} } while (false)

namespace {

constexpr std::uint32_t body_count = 8U;
constexpr std::uint32_t constraint_count = 8U;
constexpr std::uint32_t contact_capacity = 2048U;
constexpr std::uint32_t cold_ticks = 120U;
constexpr std::uint32_t warmup_ticks = 600U;
constexpr std::uint32_t measured_ticks = 7200U;
constexpr double timestep_seconds = 1.0 / 60.0;

struct Work {
    std::uint64_t dispatches{};
    std::uint64_t workgroups{};
    std::uint64_t barriers{};
    std::uint64_t queue_submissions{};
    std::uint64_t bytes_moved{};
    std::uint64_t spatial_updates{};
    std::uint64_t spatial_publications{};
    std::uint64_t time_advances{};
    std::uint64_t rigid_iterations{};
    std::uint64_t coupling_edges{};
    std::uint64_t coupling_sweeps{};
    std::array<std::uint64_t, PM_REALTIME_WORK_PHASE_COUNT> phases{};

    [[nodiscard]] bool operator==(const Work &) const = default;
};

struct Summary {
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};
    double maximum_ms{};
};

[[nodiscard]] double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto rank = std::max<std::size_t>(
        1U, static_cast<std::size_t>(std::ceil(fraction * values.size())));
    return values[rank - 1U];
}

[[nodiscard]] Summary summarize(const std::vector<double> &values) {
    return {percentile(values, 0.50), percentile(values, 0.95),
            percentile(values, 0.99), *std::max_element(values.begin(), values.end())};
}

void write_samples(std::ofstream &output, const std::vector<double> &values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_summary(std::ofstream &output, const Summary &summary) {
    output << "{\"p50_ms\":" << summary.p50_ms
           << ",\"p95_ms\":" << summary.p95_ms
           << ",\"p99_ms\":" << summary.p99_ms
           << ",\"maximum_ms\":" << summary.maximum_ms << '}';
}

void write_work(std::ofstream &output, const Work &work) {
    static constexpr std::array phase_names{
        "boundary", "predict", "index", "candidates", "domain",
        "coupling", "commit"};
    output << "{\"dispatches\":" << work.dispatches
           << ",\"workgroups\":" << work.workgroups
           << ",\"barriers\":" << work.barriers
           << ",\"queue_submissions\":" << work.queue_submissions
           << ",\"intra_tick_host_readbacks\":0"
           << ",\"bytes_moved\":" << work.bytes_moved
           << ",\"spatial_index_updates\":" << work.spatial_updates
           << ",\"spatial_index_publications\":" << work.spatial_publications
           << ",\"time_advances\":" << work.time_advances
           << ",\"active_iteration_budgets\":[\"rigid_contact_and_joint_sweeps\"]"
           << ",\"iterations\":{\"rigid_contact_and_joint_sweeps\":"
           << work.rigid_iterations << ",\"coupling_sweeps_total\":"
           << work.coupling_sweeps << "},\"coupling_edges\":"
           << work.coupling_edges << ",\"phase_dispatches\":{";
    for (std::size_t index = 0; index < work.phases.size(); ++index) {
        if (index != 0U) output << ',';
        output << '\"' << phase_names[index] << "\":" << work.phases[index];
    }
    output << "}}";
}

[[nodiscard]] Work work_from(const PmRealtimeStepStatistics &statistics) {
    Work result{};
    result.dispatches = statistics.total.dispatches;
    result.workgroups = statistics.total.workgroups;
    result.barriers = statistics.total.barriers;
    result.queue_submissions = statistics.total.queueSubmissions;
    result.bytes_moved = statistics.total.bytesMoved;
    result.spatial_updates = statistics.total.spatialIndexBuilds +
        statistics.total.spatialIndexRefits;
    result.spatial_publications = statistics.total.spatialIndexPublications;
    result.time_advances = statistics.timeAdvances;
    result.rigid_iterations = statistics.domainIterations[0];
    result.coupling_edges = statistics.couplingEdgeCount;
    result.coupling_sweeps = statistics.couplingSweeps;
    for (std::size_t index = 0; index < result.phases.size(); ++index)
        result.phases[index] = statistics.phases[index].dispatches;
    return result;
}

[[nodiscard]] bool valid_work(const PmRealtimeStepStatistics &statistics,
                              std::uint64_t expected_tick) {
    const Work work = work_from(statistics);
    std::uint64_t phase_dispatches = 0U;
    for (const auto dispatches : work.phases) phase_dispatches += dispatches;
    return statistics.step.stepCount == 1U &&
        statistics.step.substepsExecuted == 1U &&
        statistics.step.executionResult == PM_SUCCESS &&
        statistics.schemaVersion == 1U && statistics.profileVersion == 1U &&
        statistics.budgetVersion == 1U &&
        statistics.phaseCount == PM_REALTIME_WORK_PHASE_COUNT &&
        statistics.tick == expected_tick && work.dispatches > 0U &&
        work.dispatches == phase_dispatches && work.queue_submissions == 1U &&
        statistics.total.intraTickHostReadbacks == 0U &&
        work.spatial_updates == 1U && work.spatial_publications == 1U &&
        work.time_advances == 1U && work.rigid_iterations == 8U &&
        work.coupling_edges == 0U && work.coupling_sweeps == 0U;
}

} // namespace

int main(int argc, char **argv) {
    CHECK(argc == 6);
    const std::string output_path = argv[1];
    const std::string candidate_id = argv[2];
    const std::string candidate_sha256 = argv[3];
    const std::string candidate_revision = argv[4];
    const std::string driver_sha256 = argv[5];

    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t physical_count = 1U;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, &physical) ==
          PM_SUCCESS);
    CHECK(physical_count == 1U && physical != nullptr);
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(physical, &properties);

    std::uint32_t profile_count = 0U;
    CHECK(pmEnumerateExecutionProfiles(physical, &profile_count, nullptr) ==
          PM_SUCCESS);
    CHECK(profile_count == 1U);
    PmExecutionProfileProperties profile_property{};
    CHECK(pmEnumerateExecutionProfiles(
              physical, &profile_count, &profile_property) == PM_SUCCESS);
    CHECK(std::strcmp(profile_property.profileName,
                      PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP) == 0);
    CHECK(profile_property.versionMajor == 1U &&
          profile_property.versionMinor == 0U);

    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    const char *extension = PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME;
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.rigidBodies = PM_TRUE;
    device_info.enabledFeatures.rigidConstraints = PM_TRUE;
    device_info.enabledFeatures.statisticsQueries = PM_TRUE;
    device_info.enabledExtensionCount = 1U;
    device_info.ppEnabledExtensionNames = &extension;
    PmDevice device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    CHECK(queue != nullptr);

    std::vector<PmDeviceMemory> allocations;
    const auto allocate = [&](PmDeviceSize size) {
        PmMemoryAllocateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.memoryTypeIndex = 1U;
        info.allocationSize = size;
        PmDeviceMemory result{};
        CHECK(pmAllocateMemory(device, &info, &result) == PM_SUCCESS);
        allocations.push_back(result);
        return result;
    };
    const auto buffer = [&](PmDeviceSize size, void **mapped) {
        const auto memory = allocate(size);
        PmBufferCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT |
            PM_BUFFER_GEOMETRY_INPUT_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;
        PmBuffer result{};
        CHECK(pmCreateBuffer(device, &info, &result) == PM_SUCCESS);
        CHECK(pmBindBufferMemory(device, result, memory, 0U) == PM_SUCCESS);
        CHECK(pmMapMemory(device, memory, 0U, size, mapped) == PM_SUCCESS);
        return result;
    };

    void *source_map{}, *output_map{};
    const PmBuffer source = buffer(512U, &source_map);
    const PmBuffer output = buffer(
        body_count * sizeof(PmRigidState), &output_map);
    CHECK(source != nullptr && output != nullptr);
    static constexpr float vertices[8][3] = {
        {-0.5F,-0.5F,-0.5F},{0.5F,-0.5F,-0.5F},
        {0.5F,0.5F,-0.5F},{-0.5F,0.5F,-0.5F},
        {-0.5F,-0.5F,0.5F},{0.5F,-0.5F,0.5F},
        {0.5F,0.5F,0.5F},{-0.5F,0.5F,0.5F}};
    static constexpr std::uint32_t indices[36] = {
        0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
        3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
    std::memcpy(source_map, vertices, sizeof vertices);
    std::memcpy(static_cast<std::byte *>(source_map) + sizeof vertices,
                indices, sizeof indices);
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {{source, 0U, sizeof vertices},
                              PM_FORMAT_R32G32B32_SFLOAT, 8U, 12U};
    geometry_info.indices = {{source, sizeof vertices, sizeof indices},
                             PM_FORMAT_R32_UINT, 36U, 4U};
    PmGeometry geometry{};
    CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements geometry_memory{};
    pmGetGeometryMemoryRequirements(device, geometry, &geometry_memory);
    CHECK(pmBindGeometryMemory(
              device, geometry, allocate(geometry_memory.size), 0U) == PM_SUCCESS);

    PmSimulationProfileCreateInfo profile{};
    profile.sType = PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO;
    profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
    profile.versionMajor = 1U;
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.pNext = &profile;
    simulation_info.rigidBodyCapacity = body_count;
    simulation_info.constraintCapacity = constraint_count;
    simulation_info.contactCapacity = contact_capacity;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        allocate(requirements.persistent.size), 0U,
        allocate(requirements.scratch.size), 0U};
    CHECK(pmBindSimulationMemory(device, simulation, &binding) == PM_SUCCESS);

    PmRigidSolverState rigid_state{};
    rigid_state.rigidIterations = 8U;
    rigid_state.enableSleeping = PM_FALSE;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid_state;
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
          PM_SUCCESS);
    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_REALTIME_STEP_STATISTICS;
    query_info.queryCount = 1U;
    PmQueryPool query{};
    CHECK(pmCreateQueryPool(device, &query_info, &query) == PM_SUCCESS);
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1U, &commands) == PM_SUCCESS);
    PmFence fence{};
    CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &commands;
    const char *submission_stage = "initialization";

    const auto submit_and_wait = [&] {
        CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
        const PmResult waited = pmWaitForFences(
            device, 1U, &fence, PM_TRUE, PM_TIMEOUT_INFINITE);
        if (waited != PM_SUCCESS)
            std::fprintf(stderr, "%s submission completion returned %d\n",
                         submission_stage, static_cast<int>(waited));
        return waited;
    };
    const auto reset = [&] {
        CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
        return 0;
    };

    std::array<PmRigidBodyInfo, body_count> bodies{};
    std::array<PmRigidConstraintInfo, constraint_count> constraints{};
    for (std::uint32_t index = 0U; index < body_count; ++index) {
        auto &body = bodies[index];
        body.active = PM_TRUE;
        body.motionType = PM_MOTION_DYNAMIC;
        body.geometry = geometry;
        body.state.position[0] = static_cast<float>(index % 4U) * 2.0F;
        body.state.position[1] = 2.0F + static_cast<float>(index / 4U) * 2.0F;
        body.state.orientation[3] = 1.0F;
        body.mass = 1.0F;
        body.inertiaDiagonal[0] = body.inertiaDiagonal[1] =
            body.inertiaDiagonal[2] = 1.0F / 6.0F;
        body.maximumLinearSpeed = body.maximumAngularSpeed = 100.0F;
        body.collisionMargin = 0.005F;
        auto &joint = constraints[index];
        joint.active = PM_TRUE;
        joint.type = PM_CONSTRAINT_POINT;
        joint.bodyA = index;
        joint.bodyB = (index + 1U) % body_count;
        const float next_x = static_cast<float>(joint.bodyB % 4U) * 2.0F;
        const float next_y = 2.0F +
            static_cast<float>(joint.bodyB / 4U) * 2.0F;
        joint.frameA.position[0] = (next_x - body.state.position[0]) * 0.5F;
        joint.frameA.position[1] = (next_y - body.state.position[1]) * 0.5F;
        joint.frameB.position[0] = -joint.frameA.position[0];
        joint.frameB.position[1] = -joint.frameA.position[1];
        joint.frameA.orientation[3] = 1.0F;
        joint.frameB.orientation[3] = 1.0F;
        joint.solverIterations = 8U;
    }
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBuildGeometry(commands, geometry) == PM_SUCCESS);
    CHECK(pmCmdWriteRigidBodies(
              commands, simulation, 0U, body_count, bodies.data()) == PM_SUCCESS);
    CHECK(pmCmdWriteRigidConstraints(commands, simulation, 0U,
              constraint_count, constraints.data()) == PM_SUCCESS);
    CHECK(submit_and_wait() == PM_SUCCESS);
    CHECK(reset() == 0);

    // Known-invalid input must be rejected synchronously and submit no work.
    PmStepInfo invalid_step{1.0F / 60.0F, 2U,
                            {0.0F, -9.81F, 0.0F}, 0U};
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &invalid_step) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);

    std::vector<double> cold_samples;
    std::vector<double> measured_samples;
    cold_samples.reserve(cold_ticks);
    measured_samples.reserve(measured_ticks);
    std::vector<Work> unique_work;
    std::uint64_t verified_ticks = 0U;
    std::uint64_t work_mismatches = 0U;
    bool all_states_finite = true;
    PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, 0U};
    const PmBufferRange output_range{
        output, 0U, body_count * sizeof(PmRigidState)};
    std::chrono::steady_clock::time_point sustained_start{};
    std::chrono::steady_clock::time_point sustained_end{};
    for (std::uint32_t tick = 0U;
         tick < warmup_ticks + measured_ticks; ++tick) {
        submission_stage = "profile tick";
        if (tick == warmup_ticks)
            sustained_start = std::chrono::steady_clock::now();
        const auto started = std::chrono::steady_clock::now();
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdResetQueryPool(commands, query, 0U, 1U) == PM_SUCCESS);
        CHECK(pmCmdBeginStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
        CHECK(pmCmdCopyRigidStates(commands, simulation, 0U, body_count,
                                   &output_range) == PM_SUCCESS);
        CHECK(pmCmdEndStatisticsQuery(commands, query, 0U) == PM_SUCCESS);
        const PmResult completion = submit_and_wait();
        const auto completed = std::chrono::steady_clock::now();
        if (completion != PM_SUCCESS) {
            PmRealtimeStepStatistics debug{};
            const PmResult query_result = pmGetQueryPoolResults(
                device, query, 0U, 1U, sizeof debug, &debug,
                sizeof debug, PM_QUERY_RESULT_WAIT_BIT);
            std::fprintf(stderr,
                "query=%d execution=%d dispatches=%llu workgroups=%llu "
                "iterations=%llu builds=%llu refits=%llu publications=%llu\n",
                static_cast<int>(query_result),
                static_cast<int>(debug.step.executionResult),
                static_cast<unsigned long long>(debug.total.dispatches),
                static_cast<unsigned long long>(debug.total.workgroups),
                static_cast<unsigned long long>(debug.total.numericalIterations),
                static_cast<unsigned long long>(debug.total.spatialIndexBuilds),
                static_cast<unsigned long long>(debug.total.spatialIndexRefits),
                static_cast<unsigned long long>(
                    debug.total.spatialIndexPublications));
            std::ofstream failure(output_path, std::ios::binary);
            CHECK(failure.good());
            failure << "{\n"
                "  \"schema\": \"parallel-mater-realtime-cpu-capture/v1\",\n"
                "  \"status\": \"failed-required-row\",\n"
                "  \"source\": \"installed-public-api\",\n"
                "  \"driver_private_state_used\": false,\n"
                "  \"candidate\": {\"id\": \"" << candidate_id
                    << "\", \"manifest_sha256\": \"" << candidate_sha256
                    << "\", \"source_revision\": \"" << candidate_revision
                    << "\", \"runtime_variant\": \"cpu-runtime\"},\n"
                "  \"binary\": {\"driver_sha256\": \"" << driver_sha256
                    << "\"},\n"
                "  \"device\": {\"name\": \"" << properties.deviceName
                    << "\", \"vendor_id\": " << properties.vendorId
                    << ", \"device_id\": " << properties.deviceId
                    << ", \"driver_version\": " << properties.driverVersion
                    << "},\n"
                "  \"profile\": {\"name\": \"realtime-single-step\", "
                    "\"version\": {\"major\": 1, \"minor\": 0}},\n"
                "  \"workload\": {\"tier\": \"cpu-reference-v1\", "
                    "\"id\": \"rigid-brick-oracle\", \"counts\": {"
                    "\"rigid_bodies_live\": 8, \"rigid_bodies_capacity\": 8, "
                    "\"constraints_live\": 8, \"constraints_capacity\": 8, "
                    "\"contact_capacity\": 2048}},\n"
                "  \"failure\": {\"tick\": " << tick + 1U
                    << ", \"fence_result\": " << static_cast<int>(completion)
                    << ", \"query_result\": " << static_cast<int>(query_result)
                    << ", \"execution_result\": "
                    << static_cast<int>(debug.step.executionResult)
                    << ", \"partial_work\": {\"dispatches\": "
                    << debug.total.dispatches << ", \"workgroups\": "
                    << debug.total.workgroups << ", \"numerical_iterations\": "
                    << debug.total.numericalIterations
                    << ", \"spatial_index_builds\": "
                    << debug.total.spatialIndexBuilds
                    << ", \"spatial_index_refits\": "
                    << debug.total.spatialIndexRefits
                    << ", \"spatial_index_publications\": "
                    << debug.total.spatialIndexPublications << "}},\n"
                "  \"required_result\": \"PM_SUCCESS for all 7800 ticks\",\n"
                "  \"observed_result\": \"PM_ERROR_CAPACITY_EXCEEDED on first tick\"\n"
                "}\n";
            failure.close();
            return 2;
        }
        if (tick + 1U == warmup_ticks + measured_ticks)
            sustained_end = completed;
        const double milliseconds =
            std::chrono::duration<double, std::milli>(completed - started).count();
        if (tick < cold_ticks) cold_samples.push_back(milliseconds);
        if (tick >= warmup_ticks) measured_samples.push_back(milliseconds);

        PmResult status = PM_ERROR_INVALID_STATE;
        CHECK(pmGetSimulationStatus(device, simulation, &status) == PM_SUCCESS);
        CHECK(status == PM_SUCCESS);
        PmRealtimeStepStatistics statistics{};
        CHECK(pmGetQueryPoolResults(device, query, 0U, 1U,
                  sizeof statistics, &statistics, sizeof statistics,
                  PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        if (!valid_work(statistics, static_cast<std::uint64_t>(tick) + 1U))
            ++work_mismatches;
        else
            ++verified_ticks;
        const Work work = work_from(statistics);
        if (std::find(unique_work.begin(), unique_work.end(), work) ==
            unique_work.end()) unique_work.push_back(work);
        const auto *states = static_cast<const PmRigidState *>(output_map);
        for (std::uint32_t body = 0U; body < body_count; ++body) {
            const auto finite3 = [](const float *value) {
                return std::isfinite(value[0]) && std::isfinite(value[1]) &&
                    std::isfinite(value[2]);
            };
            all_states_finite = all_states_finite && finite3(states[body].position) &&
                finite3(states[body].linearVelocity) &&
                finite3(states[body].angularVelocity) &&
                std::isfinite(states[body].orientation[0]) &&
                std::isfinite(states[body].orientation[1]) &&
                std::isfinite(states[body].orientation[2]) &&
                std::isfinite(states[body].orientation[3]);
        }
        CHECK(reset() == 0);
    }
    CHECK(cold_samples.size() == cold_ticks);
    CHECK(measured_samples.size() == measured_ticks);
    CHECK(verified_ticks == warmup_ticks + measured_ticks);
    CHECK(work_mismatches == 0U && all_states_finite);
    const double sustained_wall_seconds =
        std::chrono::duration<double>(sustained_end - sustained_start).count();
    const double simulated_to_wall_ratio =
        measured_ticks * timestep_seconds / sustained_wall_seconds;
    const Summary cold_summary = summarize(cold_samples);
    const Summary sustained_summary = summarize(measured_samples);

    std::ofstream result(output_path, std::ios::binary);
    CHECK(result.good());
    result << std::setprecision(17)
           << "{\n  \"schema\": \"parallel-mater-realtime-cpu-capture/v1\",\n"
           << "  \"status\": \"passed\",\n"
           << "  \"source\": \"installed-public-api\",\n"
           << "  \"driver_private_state_used\": false,\n"
           << "  \"candidate\": {\"id\": \"" << candidate_id
           << "\", \"manifest_sha256\": \"" << candidate_sha256
           << "\", \"source_revision\": \"" << candidate_revision
           << "\", \"runtime_variant\": \"cpu-runtime\"},\n"
           << "  \"binary\": {\"driver_sha256\": \"" << driver_sha256
           << "\"},\n"
           << "  \"device\": {\"name\": \"" << properties.deviceName
           << "\", \"vendor_id\": " << properties.vendorId
           << ", \"device_id\": " << properties.deviceId
           << ", \"driver_version\": " << properties.driverVersion << "},\n"
           << "  \"profile\": {\"name\": \"realtime-single-step\", "
              "\"version\": {\"major\": 1, \"minor\": 0}},\n"
           << "  \"workload\": {\"tier\": \"cpu-reference-v1\", "
              "\"id\": \"rigid-brick-oracle\", \"counts\": {"
              "\"rigid_bodies_live\": 8, \"rigid_bodies_capacity\": 8, "
              "\"constraints_live\": 8, \"constraints_capacity\": 8, "
              "\"contact_capacity\": 2048}},\n"
           << "  \"steps\": {\"timestep_seconds\": " << timestep_seconds
           << ", \"substeps\": 1, \"successful_ticks\": "
           << warmup_ticks + measured_ticks << "},\n"
           << "  \"negative_cases\": {\"invalid_substeps_rejected_synchronously\": true},\n"
           << "  \"cold_start\": {\"ticks\": " << cold_ticks
           << ", \"summary\": ";
    write_summary(result, cold_summary);
    result << ", \"host_physics_samples_ms\": ";
    write_samples(result, cold_samples);
    result << "},\n  \"sustained\": {\"warmup_ticks\": " << warmup_ticks
           << ", \"measured_ticks\": " << measured_ticks
           << ", \"retained_samples\": " << measured_samples.size()
           << ", \"failed_ticks\": 0, \"wall_seconds\": "
           << sustained_wall_seconds
           << ", \"simulated_to_wall_time_ratio\": "
           << simulated_to_wall_ratio << ", \"host_physics_summary\": ";
    write_summary(result, sustained_summary);
    result << ", \"host_physics_samples_ms\": ";
    write_samples(result, measured_samples);
    result << ", \"gpu_timing\": {\"status\": \"unavailable\", "
              "\"samples\": []}, \"frame_timing\": {\"status\": "
              "\"not-applicable-headless-reference-target\", \"samples\": []}},\n"
           << "  \"work_accounting\": {\"verified_ticks\": "
           << verified_ticks << ", \"mismatches\": " << work_mismatches
           << ", \"unique_records\": [";
    for (std::size_t index = 0; index < unique_work.size(); ++index) {
        if (index != 0U) result << ',';
        write_work(result, unique_work[index]);
    }
    result << "]},\n  \"all_public_states_finite\": true\n}\n";
    CHECK(result.good());
    result.close();

    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, pool, 1U, &commands);
    pmDestroyCommandPool(device, pool);
    pmDestroyQueryPool(device, query);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyGeometry(device, geometry);
    pmDestroyBuffer(device, output);
    pmDestroyBuffer(device, source);
    for (const auto memory : allocations) pmFreeMemory(device, memory);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: F8-23 installed CPU qualification capture");
    return 0;
}
