// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

// Host-only contract check for assets freshly produced by Blender tests.
int main(int argc, char **argv) {
    if (argc != 7) return 2;
    parallel_mater::gallery::SceneDefinition scene;
    std::string error;
    if (!parallel_mater::gallery::load_glb_scene(argv[1], scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const std::array<std::size_t, 5> counts{
        scene.rigid_bodies.size(), scene.cloths.size(), scene.spawn_planes.size(),
        scene.destroy_planes.size(), scene.initial_particles.empty() ? 0U : 1U};
    for (std::size_t index = 0; index < counts.size(); ++index) {
        if (counts[index] != std::strtoul(argv[index + 2], nullptr, 10)) {
            std::cerr << "Exported scene count mismatch at field " << index << '\n';
            return 1;
        }
    }
    return 0;
}
