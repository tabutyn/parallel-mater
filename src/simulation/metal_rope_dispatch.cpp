// SPDX-License-Identifier: MIT
#include "rope_dispatch.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
void advance_rope(cudaStream_t stream, const RopeDispatchData &rope,
    float timestep, Vec3 gravity, int first, int last,
    const void *body_parameters, void *body_states,
    const void *previous_body_states, const void *meshes,
    std::uint32_t body_count, bool first_substep,
    const RopeSoftTargetDispatch &soft_a,
    const RopeSoftTargetDispatch &soft_b) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked([&] { execution->meshes(meshes, body_parameters, body_count); }) != cudaSuccess) return;
    }
    (void)metal::launch(stream, "rope_advance", 1U, metal::rope_record(rope),
        timestep, gravity, first, last, body_parameters, body_states,
        previous_body_states, meshes, body_count, first_substep,
        metal::target_record(soft_a), metal::target_record(soft_b));
}
} // namespace parallel_mater::simulation
