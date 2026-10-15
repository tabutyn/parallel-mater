// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <render/render_scene.hpp>
#include <scene_builder.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    using namespace parallel_mater::gallery;
    SceneDefinition scene;
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_TEST_SCENE_PATH, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    RenderSceneFrame frame;
    frame.rigid_states.reserve(scene.rigid_bodies.size());
    for (const RigidBodyDefinition &body : scene.rigid_bodies)
        frame.rigid_states.push_back(body.options.initial_state);
    frame.previous_rigid_states = frame.rigid_states;

    RendererHandle renderer;
    CommandSceneInstance instance;
    if (!create_renderer(scene, instance, 320U, 180U, renderer, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::vector<std::uint32_t> pixels;
    if (!renderer.render(frame, {}, pixels, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (pixels.size() != 320U * 180U) return 1;
    const auto [minimum, maximum] = std::minmax_element(pixels.begin(), pixels.end());
    if (minimum == pixels.end() || *minimum == *maximum) {
        std::cerr << "software renderer produced a uniform frame\n";
        return 1;
    }

    // The shared pixel buffer is bottom-up: OpenGL consumes it directly,
    // while file and D3D12 presenters reverse its rows. Keep geometry aligned
    // with overlays and those presentation adapters.
    SceneDefinition orientation_scene;
    orientation_scene.meshes.push_back(TriangleMesh{
        "upper triangle",
        {{{-0.5F, 0.4F, 0.0F}}, {{0.5F, 0.4F, 0.0F}},
         {{0.0F, 1.2F, 0.0F}}},
        {0U, 1U, 2U},
        {1.0F, 0.2F, 0.1F}});
    RigidBodyDefinition upper_body;
    upper_body.name = "upper body";
    upper_body.mesh_indices = {0U};
    orientation_scene.rigid_bodies.push_back(upper_body);
    RenderSceneFrame orientation_frame;
    orientation_frame.rigid_states.push_back({});
    orientation_frame.previous_rigid_states = orientation_frame.rigid_states;
    RendererHandle orientation_renderer;
    if (!create_renderer(orientation_scene, instance, 96U, 96U,
                         orientation_renderer, error) ||
        !orientation_renderer.render(
            orientation_frame,
            Camera{{0.0F, 0.0F, 5.0F}, {0.0F, 0.0F, 0.0F},
                   {0.0F, 1.0F, 0.0F}, 48.0F},
            pixels, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const auto bright_pixels = [&](std::uint32_t first_row,
                                   std::uint32_t end_row) {
        std::size_t count = 0U;
        for (std::uint32_t row = first_row; row < end_row; ++row) {
            for (std::uint32_t column = 0U; column < 96U; ++column) {
                const std::uint32_t value = pixels[row * 96U + column];
                const unsigned luminance = (value & 0xffU) +
                    ((value >> 8U) & 0xffU) + ((value >> 16U) & 0xffU);
                count += luminance > 100U;
            }
        }
        return count;
    };
    if (bright_pixels(48U, 96U) <= bright_pixels(0U, 48U)) {
        std::cerr << "software renderer inverted vertical image orientation\n";
        return 1;
    }
    return 0;
}
