// SPDX-License-Identifier: MIT
#include "geometry_constraints_dispatch.hpp"
#include "avbd_schedule.hpp"
#include "driver/metal/schedule_context.hpp"

namespace parallel_mater::simulation {
cudaError_t geometry_constraints_occupancy(int *blocks, std::uint32_t) noexcept {
    if (!blocks) return cudaErrorInvalidValue;
    *blocks = 1; return cudaSuccess;
}
cudaError_t prepare_geometry_constraints(cudaStream_t stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *predicted, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations, std::uint32_t) noexcept {
    return metal::launch(stream, "prepare_avbd_kernel", 1U,
        parameters, previous, predicted, body_count, manifolds,
        active_pairs, active_pair_count, constraints, constraint_capacity,
        ids, bodies, schedule, timestep, gravity, base_iterations,
        requested_iterations, std::max(1U, (body_count + 31U) / 32U));
}
cudaError_t solve_geometry_constraints(cudaStream_t stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *states, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t) noexcept {
    auto *execution = metal::context(stream);
    if (!execution) return cudaErrorInvalidResourceHandle;
    return execution->checked([&] {
        // Explicit preview limitation: read the prepared work bounds instead of
        // encoding O(body_count * maximum_iterations) empty GPU dispatches.
        // All solver equations and state updates still execute on the GPU.
        execution->wait();
        const auto &prepared = *static_cast<const ContactSchedule *>(schedule);
        cudaError_t result = cudaSuccess;
        encode_avbd_schedule(prepared.budget, prepared.remaining,
            [&](AvbdStage operation, std::uint32_t phase, std::uint32_t iteration, std::uint32_t color) {
                if (result != cudaSuccess) return;
                result = metal::launch(stream, "solve_avbd_stage_kernel",
                    std::max(1U, (body_count + 31U) / 32U),
                    parameters, previous, states, body_count, manifolds,
                    active_pairs, active_pair_count, constraints, constraint_capacity,
                    bodies, schedule, timestep, static_cast<std::uint32_t>(operation),
                    phase, iteration, color);
            });
        if (result != cudaSuccess) throw std::runtime_error("Metal staged AVBD dispatch failed");
    });
}
} // namespace parallel_mater::simulation
