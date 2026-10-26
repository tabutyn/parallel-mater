// SPDX-License-Identifier: MIT
#pragma once

// Private runtime storage. Execution lives in the Slang dispatch module.
struct FluidSmokeCouplingStorage {
    FluidSmokeCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t *converted{};
    Vec3 *realtime_phase_positions{};
    Vec3 *realtime_phase_velocities{};
    Vec3 *realtime_phase_vorticities{};
    float *realtime_phase_ages{};
    float *realtime_phase_thermal_lift{};
    float *realtime_phase_densities{};
    float *realtime_phase_pressures{};
    std::uint32_t realtime_pending_phase_count{};
    std::uint32_t realtime_phase_source_generation{};
    std::uint32_t realtime_phase_destination_generation{};
    std::uint32_t realtime_phase_apply_count{};
    std::uint32_t realtime_phase_first_slot{};
    bool realtime_phase_discovery{};
    ~FluidSmokeCouplingStorage() {
        release_managed(converted);
        release_managed(realtime_phase_positions);
        release_managed(realtime_phase_velocities);
        release_managed(realtime_phase_vorticities);
        release_managed(realtime_phase_ages);
        release_managed(realtime_phase_thermal_lift);
        release_managed(realtime_phase_densities);
        release_managed(realtime_phase_pressures);
    }
};
