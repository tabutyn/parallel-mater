// SPDX-License-Identifier: MIT
#include <parallel_mater/d3d12.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace parallel_mater::d3d12;

namespace {
bool require(bool condition, const char *message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}
}

int main() {
    FrameToken empty;
    if (!require(empty.ready() && !empty.pending() && empty.wait().ok(),
                 "Default token must be complete")) return 1;

    World zero_constraints;
    Status status = World::create({.rigid_constraint_capacity = 0U},
                                  zero_constraints);
    if (!require(status.ok(), "Zero constraint capacity was rejected")) return 1;

    World world;
    status = World::create({}, world);
    if (!require(status.ok(), status.message ? status.message : "World create failed"))
        return 1;
    const NativeContext native = world.native_context();
    if (!require(native.device != nullptr && native.direct_queue != nullptr,
                 "Native D3D12 context is missing")) return 1;

    FluidId fluid{};
    status = world.add_fluid({}, HostSpan<const FluidParticle>{}, fluid);
    if (!require(status.code == StatusCode::not_supported,
                 "Non-rigid gate must fail transactionally")) return 1;

    FrameToken invalid_token;
    if (!require(world.step_async({.timestep = 0.0F}, invalid_token).code ==
                     StatusCode::invalid_argument,
                 "Zero timestep was accepted")) return 1;
    if (!require(world.step_async({.substeps = 1025U}, invalid_token).code ==
                     StatusCode::invalid_argument,
                 "Too many substeps were accepted")) return 1;

    constexpr std::array<Vec3, 8> vertices{{
        {-0.5F,-0.5F,-0.5F},{0.5F,-0.5F,-0.5F},
        {0.5F,0.5F,-0.5F},{-0.5F,0.5F,-0.5F},
        {-0.5F,-0.5F,0.5F},{0.5F,-0.5F,0.5F},
        {0.5F,0.5F,0.5F},{-0.5F,0.5F,0.5F}}};
    constexpr std::array<std::uint32_t, 36> indices{{
        0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
        2,3,7,2,7,6,0,4,7,0,7,3,1,2,6,1,6,5}};
    TriangleMeshId mesh{};
    status = world.add_triangle_mesh(
        HostSpan<const Vec3>{vertices.data(), vertices.size()},
        HostSpan<const std::uint32_t>{indices.data(), indices.size()}, mesh);
    if (!require(status.ok(), "Could not create test mesh")) return 1;
    RigidBodyId body{};
    RigidBodyOptions body_options{};
    body_options.mesh = mesh;
    body_options.initial_state.position = {0.0F, 2.0F, 0.0F};
    status = world.add_rigid_body(body_options, body);
    if (!require(status.ok(), "Could not create rigid body")) return 1;

    FrameToken completion;
    status = world.step_async({.timestep=1.0F/60.0F, .substeps=4U,
                               .collect_kernel_timings=true,
                               .collect_rigid_contacts=true}, completion);
    if (!require(status.ok(), status.message ? status.message : "Submit failed") ||
        !require(completion.wait().ok(), "Frame fence failed")) return 1;
    RigidBodyState state{};
    status = world.read_rigid_body_state(body, state);
    if (!require(status.ok() && state.position.y < 2.0F,
                 "DirectCompute integration did not advance")) return 1;

    WorldStepTimings timings{};
    status = world.collect_step_timings(timings);
    if (!require(status.ok() && timings.available && timings.frame_index == 1U,
                 "Timestamp timings unavailable")) return 1;
    const float timing_sum=timings.rigid_integration.total_milliseconds+
        timings.rigid_contact_generation.total_milliseconds+
        timings.rigid_contact_solve.total_milliseconds+
        timings.rigid_input_clear.total_milliseconds;
    if(!require(timings.rigid_integration.launch_count==4U &&
                timings.rigid_contact_generation.launch_count>0U &&
                timings.rigid_contact_solve.launch_count>0U,
                "GPU phase launch counts are missing") ||
       !require(std::fabs(timing_sum-timings.total_gpu_milliseconds)<0.001F,
                "GPU phase timings do not add up")) return 1;
    WorldStatistics statistics{};
    status = world.collect_statistics(statistics);
    if (!require(status.ok() && statistics.frame_index == 1U &&
                     statistics.rigid_body_count == 1U,
                 "Statistics are invalid")) return 1;
    return 0;
}
