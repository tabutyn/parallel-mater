// SPDX-License-Identifier: MIT
#pragma once
#include "types.hpp"
#include "smoke_dispatch.hpp"

namespace parallel_mater::simulation::metal_bindings {
inline Vec3 vector(const float value[3]) { return {value[0], value[1], value[2]}; }
inline SmokeOptions smoke_options(const SmokeDomainDispatchOptions &value) noexcept {
    SmokeOptions result{};
    result.capacity = value.capacity;
    result.emitter_center = vector(value.emitter_center);
    result.emitter_half_extents = {
        value.emitter_half_extents[0], value.emitter_half_extents[1]};
    result.initial_velocity = vector(value.initial_velocity);
    result.wind = vector(value.wind);
    result.particles_per_second = value.particles_per_second;
    result.lifetime = value.lifetime;
    result.particle_radius = value.particle_radius;
    result.buoyancy = value.buoyancy;
    result.response = value.response;
    result.rest_number_density = value.rest_number_density;
    result.pressure_stiffness = value.pressure_stiffness;
    result.viscosity = value.viscosity;
    result.vorticity_confinement = value.vorticity_confinement;
    result.maximum_speed = value.maximum_speed;
    result.grid_resolution = value.grid_resolution;
    result.grid_vertical_resolution = value.grid_vertical_resolution;
    result.grid_pressure_iterations = value.grid_pressure_iterations;
    result.grid_kinematic_viscosity = value.grid_kinematic_viscosity;
    result.grid_les_coefficient = value.grid_les_coefficient;
    result.grid_pressure_tolerance = value.grid_pressure_tolerance;
    result.grid_minimum = vector(value.grid_minimum);
    result.grid_edge_length = value.grid_edge_length;
    return result;
}
inline SmokeOptions smoke_options(const SmokeDispatchOptions &value) {
    SmokeOptions result{};
    result.capacity = value.capacity;
    result.particle_radius = value.particle_radius;
    result.lifetime = value.lifetime;
    result.rest_number_density = value.rest_number_density;
    result.pressure_stiffness = value.pressure_stiffness;
    result.maximum_speed = value.maximum_speed;
    return result;
}
} // namespace parallel_mater::simulation::metal_bindings
