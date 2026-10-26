// SPDX-License-Identifier: MIT
#include <parallel_mater_scene/scene.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

namespace parallel_mater::gallery {
namespace {

constexpr float fixed_timestep = 1.0F / 60.0F;
constexpr float gallery_gravity_magnitude = 9.81F;
constexpr float minimum_material_response = 0.06F;
constexpr float maximum_material_response = 0.99F;
constexpr std::uint32_t smoke_grid_pressure_cycles = 12U;

[[nodiscard]] float below(float boundary) noexcept {
    return std::nextafter(boundary, 0.0F);
}

[[nodiscard]] bool clamp_compliance(
    float &compliance, const std::vector<float> &inverse_masses,
    float fallback_mass, std::string_view domain, std::string &error) {
    if (compliance == 0.0F) return true;

    float minimum_inverse = std::numeric_limits<float>::infinity();
    float maximum_inverse = 0.0F;
    if (inverse_masses.empty()) {
        if (!std::isfinite(fallback_mass) || fallback_mass <= 0.0F) {
            error = std::string(domain) + " has invalid fixed-tick mass";
            return false;
        }
        minimum_inverse = maximum_inverse = 1.0F / fallback_mass;
    } else {
        for (const float inverse : inverse_masses) {
            if (!std::isfinite(inverse) || inverse < 0.0F) {
                error = std::string(domain) +
                    " has invalid fixed-tick inverse mass";
                return false;
            }
            if (inverse == 0.0F) continue;
            minimum_inverse = std::min(minimum_inverse, inverse);
            maximum_inverse = std::max(maximum_inverse, inverse);
        }
        if (maximum_inverse == 0.0F) return true;
    }

    // A live constraint may join one moving node to a pin, or two moving
    // nodes. This conservative interval therefore covers every authored link
    // without inspecting or rewriting topology.
    const double dt2 = static_cast<double>(fixed_timestep) * fixed_timestep;
    const double minimum_ratio =
        1.0 / maximum_material_response - 1.0;
    const double maximum_ratio =
        1.0 / minimum_material_response - 1.0;
    const float lower = static_cast<float>(
        minimum_ratio * dt2 * (2.0 * maximum_inverse));
    const float upper = static_cast<float>(
        maximum_ratio * dt2 * minimum_inverse);
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
        error = std::string(domain) +
            " mass ratio has no fixed-tick compliance envelope";
        return false;
    }
    compliance = std::clamp(compliance, lower, upper);
    return true;
}

[[nodiscard]] float shortest_bond(
    const SoftBodyDefinition &body) noexcept {
    float result = std::numeric_limits<float>::infinity();
    for (const SoftBodyBond &bond : body.bonds)
        if (std::isfinite(bond.rest_length) && bond.rest_length > 0.0F)
            result = std::min(result, bond.rest_length);
    return result;
}

[[nodiscard]] float smoke_grid_spacing(const SmokeOptions &smoke) noexcept {
    if (smoke.grid_resolution == 0U) return 0.0F;
    if (smoke.grid_edge_length > 0.0F)
        return smoke.grid_edge_length /
            static_cast<float>(smoke.grid_resolution);
    const float x = std::abs(smoke.wind.x) * smoke.lifetime + 3.0F;
    const float z = std::abs(smoke.wind.z) * smoke.lifetime + 3.0F;
    return std::max(x, z) / static_cast<float>(smoke.grid_resolution);
}

} // namespace

bool apply_gallery_fixed_tick_material_contract(
    SceneDefinition &scene, std::string &error) {
    error.clear();

    for (const ClothDefinition &cloth : scene.cloths)
        if (cloth.break_strain > 0.0F || cloth.impact_break_impulse > 0.0F) {
            error = "fixed-tick-60hz-v1 does not implement cloth fracture";
            return false;
        }

    if (scene.fluid_options.capacity != 0U) {
        if (scene.fluid_options.rest_particle_volume <= 0.0F) {
            const float diameter = 2.0F * scene.fluid_options.particle_radius;
            scene.fluid_options.rest_particle_volume =
                diameter * diameter * diameter;
        }
        const float maximum = below(
            0.5F * scene.fluid_options.support_radius / fixed_timestep);
        scene.fluid_options.maximum_speed = std::min(
            scene.fluid_options.maximum_speed, maximum);
    }

    for (ClothDefinition &cloth : scene.cloths) {
        if (!clamp_compliance(cloth.stretch_compliance,
                              cloth.inverse_masses, cloth.vertex_mass,
                              "cloth stretch", error) ||
            !clamp_compliance(cloth.bending_compliance,
                              cloth.inverse_masses, cloth.vertex_mass,
                              "cloth bend", error))
            return false;
    }

    const float minimum_characteristic_length = std::nextafter(
        4.0F * gallery_gravity_magnitude * scene.gravity_scale *
            fixed_timestep * fixed_timestep,
        std::numeric_limits<float>::infinity());
    for (SoftBodyDefinition &body : scene.soft_bodies) {
        for (SoftBodyBond &bond : body.bonds)
            if (std::isfinite(bond.rest_length) && bond.rest_length > 0.0F)
                bond.rest_length = std::max(
                    bond.rest_length, minimum_characteristic_length);
        const float characteristic = shortest_bond(body);
        if (!std::isfinite(characteristic)) {
            error = "soft body has no fixed-tick characteristic length";
            return false;
        }
        body.maximum_speed = std::min(body.maximum_speed,
            below(0.5F * characteristic / fixed_timestep));
        body.node_radius = std::min(body.node_radius,
            below(0.5F * characteristic));
        body.spring_damping = 0.0F;
        if (body.shape_matching_stiffness != 0.0F)
            body.shape_matching_stiffness = std::clamp(
                body.shape_matching_stiffness, 0.05F, 0.995F);
        if (!clamp_compliance(body.stretch_compliance,
                              body.inverse_masses, body.node_mass,
                              "soft-body stretch", error))
            return false;
    }

    for (RopeDefinition &rope : scene.ropes)
        rope.options.maximum_speed = std::min(
            rope.options.maximum_speed,
            below(0.5F * rope.options.node_spacing / fixed_timestep));

    if (scene.has_smoke) {
        auto &smoke = scene.smoke_options;
        if (smoke.grid_resolution == 0U) {
            smoke.maximum_speed = std::min(smoke.maximum_speed,
                below(1.5F * smoke.particle_radius / fixed_timestep));
        } else {
            const float spacing = smoke_grid_spacing(smoke);
            if (!std::isfinite(spacing) || spacing <= 0.0F) {
                error = "smoke grid has no fixed-tick cell spacing";
                return false;
            }
            smoke.maximum_speed = std::min(smoke.maximum_speed,
                below(spacing / fixed_timestep));
            smoke.grid_pressure_iterations = smoke_grid_pressure_cycles;
        }
    }

    return true;
}

} // namespace parallel_mater::gallery
