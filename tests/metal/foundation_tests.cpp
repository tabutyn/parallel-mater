// SPDX-License-Identifier: MIT
#include <parallel_mater/metal.hpp>

#include <iostream>

using namespace parallel_mater::metal;

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

} // namespace

int main() {
    FrameToken empty_token;
    if (!require(empty_token.ready() && !empty_token.pending() &&
                     empty_token.wait().ok(),
                 "A default frame token must be complete")) {
        return 1;
    }

    World unconstrained_world;
    Status status = World::create(
        {.rigid_constraint_capacity = 0U}, unconstrained_world);
    if (!require(status.ok(),
                 "Metal rejected CUDA-valid zero constraint capacity")) {
        return 1;
    }

    World world;
    status = World::create({}, world);
    if (!require(status.ok(), status.message == nullptr ? "World creation failed"
                                                        : status.message)) {
        return 1;
    }
    const NativeContext native = world.native_context();
    if (!require(native.device != nullptr && native.command_queue != nullptr,
                 "World did not expose its retained Metal context")) {
        return 1;
    }

    FluidId fluid{};
    status = world.add_fluid({}, HostSpan<const FluidParticle>{}, fluid);
    if (!require(status.code == StatusCode::invalid_argument,
                 "A zero-capacity fluid must fail validation")) {
        return 1;
    }

    FrameToken invalid_token;
    status = world.step_async({.timestep = 0.0F}, invalid_token);
    if (!require(status.code == StatusCode::invalid_argument,
                 "A zero timestep must fail validation")) {
        return 1;
    }
    status = world.step_async({.substeps = 1'025U}, invalid_token);
    if (!require(status.code == StatusCode::invalid_argument,
                 "Metal accepted more than CUDA's 1024 step substeps")) {
        return 1;
    }

    FrameToken completion;
    status = world.step_async({.collect_kernel_timings = true}, completion);
    if (!require(status.ok(), status.message == nullptr ? "Submission failed"
                                                        : status.message) ||
        !require(completion.wait().ok(), "Metal frame failed") ||
        !require(completion.ready() && !completion.pending(),
                 "Completed token reports pending")) {
        return 1;
    }

    WorldStepTimings timings{};
    status = world.collect_step_timings(timings);
    if (!require(status.ok() && timings.available && timings.frame_index == 1U,
                 "Profiled frame timings are unavailable")) {
        return 1;
    }

    WorldStatistics statistics{};
    status = world.collect_statistics(statistics);
    if (!require(status.ok() && statistics.frame_index == 1U,
                 "Foundation statistics are invalid")) {
        return 1;
    }
    status = world.step_async({}, completion);
    if (!require(status.ok() && completion.wait().ok(),
                 "A completed frame token could not be reused")) {
        return 1;
    }
    status = world.step({});
    status = status ? world.collect_statistics(statistics) : status;
    if (!require(status.ok() && statistics.frame_index == 3U,
                 "Synchronous step did not reuse its preallocated token")) {
        return 1;
    }
    return 0;
}
