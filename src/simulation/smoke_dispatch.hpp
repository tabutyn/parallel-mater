// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace parallel_mater::simulation {

// Backend-neutral arguments shared by smoke domain and coupling dispatch.
// Keep only fields consumed by generated kernels at this boundary.
struct SmokeDispatchOptions {
    std::uint32_t capacity{};
    float particle_radius{};
    float lifetime{};
    float rest_number_density{};
    float pressure_stiffness{};
    float maximum_speed{};
};

struct SmokeDomainDispatchOptions {
    std::uint32_t capacity{};
    float emitter_center[3]{};
    float emitter_half_extents[2]{};
    float initial_velocity[3]{};
    float wind[3]{};
    float particles_per_second{};
    float lifetime{};
    float particle_radius{};
    float buoyancy{};
    float response{};
    float rest_number_density{};
    float pressure_stiffness{};
    float viscosity{};
    float vorticity_confinement{};
    float maximum_speed{};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
    std::uint32_t grid_pressure_iterations{};
    float grid_kinematic_viscosity{};
    float grid_les_coefficient{};
    float grid_pressure_tolerance{};
    float grid_minimum[3]{};
    float grid_edge_length{};
};

struct SmokeGridDispatchField {
    std::uint32_t resolution{};
    std::uint32_t height{};
    float minimum[3]{};
    float spacing{};
    const void *velocity{};
    const float *density{};
    const float *pressure{};
    const float *face_velocity[3]{};
    const void *vorticity{};
    const float *strain{};
};

} // namespace parallel_mater::simulation
