// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/render_scene.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace gallery = parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(float left, float right) {
    return std::fabs(left - right) < 0.0001F;
}
} // namespace

int main() {
    try {
        gallery::RenderSceneFrame previous;
        previous.revision = 7U;
        previous.rigid_states.push_back({
            .position = {0.0F, 0.0F, 0.0F},
            .orientation = {0.0F, 0.0F, 0.0F, 1.0F},
        });
        previous.fluid_particles = {
            {.position = {0.0F, 0.0F, 0.0F}, .stable_id = 1U},
            {.position = {10.0F, 0.0F, 0.0F}, .stable_id = 2U},
        };
        previous.rope_nodes = {{{0.0F, 0.0F, 0.0F},
                                {2.0F, 0.0F, 0.0F}}};
        previous.smoke_particles = {
            {.position = {1.0F, 0.0F, 0.0F}},
            {.position = {2.0F, 0.0F, 0.0F}},
        };
        gallery::reset_render_scene_snapshots(previous);

        gallery::RenderSceneFrame current;
        current.revision = 8U;
        current.rigid_states.push_back({
            .position = {10.0F, 0.0F, 0.0F},
            .orientation = {0.0F, 0.0F, 0.70710678F, 0.70710678F},
        });
        // Stable IDs deliberately reorder. ID 3 is a birth; ID 1 is removed.
        current.fluid_particles = {
            {.position = {14.0F, 0.0F, 0.0F}, .stable_id = 2U},
            {.position = {30.0F, 0.0F, 0.0F}, .stable_id = 3U},
        };
        current.rope_nodes = {{{2.0F, 0.0F, 0.0F},
                               {4.0F, 0.0F, 0.0F}}};
        current.smoke_particles = {
            {.position = {3.0F, 0.0F, 0.0F}},
            {.position = {4.0F, 0.0F, 0.0F}},
        };
        gallery::advance_render_scene_snapshots(previous, current);

        const auto rigid = gallery::presentation_rigid_state(
            current, 0U, 0.5F);
        require(near(rigid.position.x, 5.0F) &&
                    near(rigid.orientation.z, 0.38268343F) &&
                    near(rigid.orientation.w, 0.92387953F),
                "Rigid position/orientation interpolation changed");
        require(near(gallery::presentation_fluid_position(
                         current, 0U, 0.5F).x, 12.0F) &&
                    near(gallery::presentation_fluid_position(
                         current, 1U, 0.5F).x, 30.0F),
                "Stable-ID birth/removal interpolation changed");
        require(near(gallery::presentation_domain_position(
                         current.previous_rope_nodes, current.rope_nodes,
                         0U, 1U, 0.5F).x, 3.0F),
                "Stable topology did not interpolate");
        require(near(gallery::presentation_smoke_position(
                         current, 0U, 0.5F).x, 2.0F),
                "Stable smoke output did not interpolate");

        current.rope_nodes[0].push_back({99.0F, 0.0F, 0.0F});
        require(near(gallery::presentation_domain_position(
                         current.previous_rope_nodes, current.rope_nodes,
                         0U, 1U, 0.0F).x, 4.0F),
                "Topology change blended stale element slots");
        current.smoke_particles.push_back(
            {.position = {77.0F, 0.0F, 0.0F}});
        require(near(gallery::presentation_smoke_position(
                         current, 0U, 0.0F).x, 3.0F),
                "Smoke spawn blended stale element slots");

        // Equivalent antipodal quaternion representations take the short path
        // instead of collapsing through a zero quaternion.
        current.rigid_states[0].orientation = {0.0F, 0.0F, 0.0F, -1.0F};
        current.previous_rigid_states[0].orientation =
            {0.0F, 0.0F, 0.0F, 1.0F};
        const auto antipodal = gallery::presentation_rigid_state(
            current, 0U, 0.5F);
        require(near(std::fabs(antipodal.orientation.w), 1.0F),
                "Antipodal orientation interpolation took the long path");

        gallery::reset_render_scene_snapshots(current);
        require(current.previous_revision == current.revision &&
                    current.previous_rigid_states.size() ==
                        current.rigid_states.size() &&
                    near(current.previous_rigid_states[0].position.x,
                         current.rigid_states[0].position.x) &&
                    current.previous_fluid_particles.size() ==
                        current.fluid_particles.size(),
                "Reset did not seed identical presentation snapshots");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
