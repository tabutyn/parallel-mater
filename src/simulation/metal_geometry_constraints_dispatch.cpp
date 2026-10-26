// SPDX-License-Identifier: MIT
#include "geometry_constraints_dispatch.hpp"
#include "avbd_schedule.hpp"
#include "rigid_realtime.hpp"
#include "driver/metal/schedule_context.hpp"

namespace parallel_mater::simulation {
PmResult geometry_constraints_occupancy(int *blocks, std::uint32_t) noexcept {
    if (!blocks) return PM_ERROR_INVALID_ARGUMENT;
    *blocks = 1; return PM_SUCCESS;
}
PmResult prepare_geometry_constraints(device::ExecutionQueue stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *predicted, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations, std::uint32_t profile_version,
    std::uint32_t) noexcept {
    return metal::launch(stream, "prepare_avbd_kernel", 1U,
        parameters, previous, predicted, body_count, manifolds,
        active_pairs, active_pair_count, constraints, constraint_capacity,
        ids, bodies, schedule, timestep, gravity, base_iterations,
        requested_iterations, profile_version,
        std::max(1U, (body_count + 31U) / 32U));
}
PmResult solve_geometry_constraints(device::ExecutionQueue stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *states, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t) noexcept {
    auto *execution = metal::context(stream);
    if (!execution) return PM_ERROR_INVALID_ARGUMENT;
    return execution->checked_result([&] {
        // Explicit preview limitation: read the prepared work bounds instead of
        // encoding O(body_count * maximum_iterations) empty GPU dispatches.
        // All solver equations and state updates still execute on the GPU.
        execution->wait();
        const auto &prepared = *static_cast<const ContactSchedule *>(schedule);
        PmResult result = PM_SUCCESS;
        encode_avbd_schedule(prepared.budget, prepared.remaining,
            [&](AvbdStage operation, std::uint32_t phase, std::uint32_t iteration, std::uint32_t color) {
                if (result != PM_SUCCESS) return;
                result = metal::launch(stream, "solve_avbd_stage_kernel",
                    std::max(1U, (body_count + 31U) / 32U),
                    parameters, previous, states, body_count, manifolds,
                    active_pairs, active_pair_count, constraints, constraint_capacity,
                    bodies, schedule, timestep, static_cast<std::uint32_t>(operation),
                    phase, iteration, color);
            }, prepared.offsets[0] == rigid_realtime_profile_version
                ? 1U : 2U);
        if (result != PM_SUCCESS) throw std::runtime_error("Metal staged AVBD dispatch failed");
    });
}

PmResult solve_geometry_constraints_fixed_realtime(
    device::ExecutionQueue stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *states,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t) noexcept {
    auto *execution = metal::context(stream);
    if (!execution) return PM_ERROR_INVALID_ARGUMENT;
    return execution->checked_result([&] {
        const auto first_dispatch = execution->dispatch_count;
        const auto result = metal::launch(stream, "solve_avbd_kernel", 1U,
            parameters, previous, states, body_count, manifolds, active_pairs,
            active_pair_count, constraints, constraint_capacity, bodies,
            schedule, timestep);
        if (result != PM_SUCCESS)
            throw std::runtime_error(
                "Metal fixed realtime cooperative AVBD dispatch failed");
        const auto issued = execution->dispatch_count - first_dispatch;
        const auto expected =
            rigid_realtime_cooperative_solve_dispatches(body_count);
        if (issued != expected)
            throw std::runtime_error(
                "Metal fixed realtime AVBD dispatch accounting mismatch");
    });
}
} // namespace parallel_mater::simulation
