// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

// Host-only contract check for assets freshly produced by Blender tests.
int main(int argc, char **argv) {
    if (argc != 10) return 2;
    parallel_mater::gallery::SceneDefinition scene;
    std::string error;
    if (!parallel_mater::gallery::load_glb_scene(argv[1], scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const std::array<std::size_t, 8> counts{
        scene.rigid_bodies.size(), scene.rigid_constraints.size(),
        scene.cloths.size(), scene.particle_sources.size(),
        scene.destroy_planes.size(), scene.initial_particles.empty() ? 0U : 1U,
        scene.soft_bodies.size(), scene.ropes.size()};
    for (std::size_t index = 0; index < counts.size(); ++index) {
        if (counts[index] != std::strtoul(argv[index + 2], nullptr, 10)) {
            std::cerr << "Exported scene count mismatch at field " << index << '\n';
            return 1;
        }
    }
    return 0;
}
