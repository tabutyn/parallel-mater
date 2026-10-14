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
    return 0;
}
