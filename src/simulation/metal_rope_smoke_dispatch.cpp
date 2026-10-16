// SPDX-License-Identifier: MIT
#include "smoke_coupling_dispatch.hpp"

#include "metal_dispatch_options.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
using metal_bindings::smoke_options;

} // namespace

void apply_smoke_wind_to_rope(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count,
    float rope_maximum_speed, bool first_attached, bool last_attached,
    bool soft_first, bool soft_last, const SmokeDispatchOptions &smoke,
    const void *smoke_positions, const void *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    const SmokeGridDispatchField &smoke_grid, float drag,
    float maximum_acceleration, float timestep) noexcept {
    (void)metal::launch(stream, "smoke_rope_wind", grid,
        rope_positions,
        rope_velocities, rope_count, rope_maximum_speed,
        first_attached, last_attached, soft_first, soft_last, smoke_options(smoke),
        smoke_positions,
        smoke_velocities,
        keys,
        indices, metal::grid_record(smoke_grid), drag,
        maximum_acceleration, timestep);
}

void collide_smoke_with_rope(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    std::uint32_t count, const SmokeDispatchOptions &smoke,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, const void *minimum, const void *maximum,
    float clearance) noexcept {
    (void)metal::launch(stream, "smoke_rope_contact", grid,
        positions,
        previous_positions,
        velocities, ages, count,
        smoke.lifetime,
        rope_positions,
        rope_velocities, rope_count,
        minimum,
        maximum, clearance);
}

} // namespace parallel_mater::simulation
