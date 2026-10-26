// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

constexpr float dt = 1.0F / 60.0F;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

float response(float compliance, float effective_inverse_mass) {
    if (compliance == 0.0F) return 1.0F;
    return 1.0F / (1.0F + compliance /
        (dt * dt * effective_inverse_mass));
}

void require_compliance(float compliance,
                        const std::vector<float> &inverse_masses,
                        float fallback_mass, const std::string &label) {
    if (compliance == 0.0F) return;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = 0.0F;
    if (inverse_masses.empty()) {
        minimum = maximum = 1.0F / fallback_mass;
    } else {
        for (const float inverse : inverse_masses) {
            if (inverse == 0.0F) continue;
            minimum = std::min(minimum, inverse);
            maximum = std::max(maximum, inverse);
        }
        if (maximum == 0.0F) return;
    }
    require(response(compliance, minimum) >= 0.05F &&
            response(compliance, 2.0F * maximum) <= 0.995F,
            label + " is outside the fixed-tick response envelope");
}

float shortest_bond(const parallel_mater::gallery::SoftBodyDefinition &body) {
    float shortest = std::numeric_limits<float>::infinity();
    for (const auto &bond : body.bonds)
        shortest = std::min(shortest, bond.rest_length);
    return shortest;
}

} // namespace

int main() {
    using namespace parallel_mater::gallery;
    const std::filesystem::path assets{PARALLEL_MATER_GALLERY_ASSET_DIR};
    std::size_t supported{};
    std::size_t unavailable{};

    for (const GalleryEntry &entry : gallery_entries) {
        SceneDefinition scene;
        std::string error;
        require(load_glb_scene(assets / entry.asset_file, scene, error),
                std::string(entry.name) + ": " + error);
        const bool applied =
            apply_gallery_fixed_tick_material_contract(scene, error);
        if (!gallery_fixed_tick_supported(entry.context)) {
            require(!applied && error.find("cloth fracture") != std::string::npos,
                    std::string(entry.name) +
                        ": unsupported fracture was not explicit");
            ++unavailable;
            continue;
        }
        require(applied, std::string(entry.name) + ": " + error);
        ++supported;

        if (scene.fluid_options.capacity != 0U)
            require(scene.fluid_options.maximum_speed * dt <=
                        0.5F * scene.fluid_options.support_radius,
                    std::string(entry.name) + ": fluid speed envelope");
        for (const ClothDefinition &cloth : scene.cloths) {
            require_compliance(cloth.stretch_compliance,
                cloth.inverse_masses, cloth.vertex_mass,
                std::string(entry.name) + ": cloth stretch");
            require_compliance(cloth.bending_compliance,
                cloth.inverse_masses, cloth.vertex_mass,
                std::string(entry.name) + ": cloth bend");
        }
        for (const SoftBodyDefinition &body : scene.soft_bodies) {
            const float characteristic = shortest_bond(body);
            require(characteristic >= 0.005F &&
                    9.81F * scene.gravity_scale * dt * dt <=
                        0.25F * characteristic &&
                    body.spring_damping == 0.0F &&
                    (body.shape_matching_stiffness == 0.0F ||
                     (body.shape_matching_stiffness >= 0.05F &&
                      body.shape_matching_stiffness <= 0.995F)) &&
                    body.maximum_speed * dt <= 0.5F * characteristic &&
                    body.node_radius <= 0.5F * characteristic,
                    std::string(entry.name) + ": soft-body envelope");
            require_compliance(body.stretch_compliance,
                body.inverse_masses, body.node_mass,
                std::string(entry.name) + ": soft-body stretch");
        }
        for (const RopeDefinition &rope : scene.ropes)
            require(rope.options.maximum_speed * dt <=
                        0.5F * rope.options.node_spacing,
                    std::string(entry.name) + ": rope speed envelope");
        if (scene.has_smoke && scene.smoke_options.grid_resolution != 0U)
            require(scene.smoke_options.grid_pressure_iterations == 12U,
                    std::string(entry.name) +
                        ": smoke pressure-cycle envelope");
    }

    require(supported == 26U && unavailable == 2U,
            "fixed-tick Gallery availability count changed");
    std::cout << "PASS: 26 Gallery scenes migrated; 2 fracture scenes explicit\n";
    return 0;
}
