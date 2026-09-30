// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include "../../examples/support/vector_math.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::gallery;
namespace math = parallel_mater::gallery::math;

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void require(Status status, const char *message) {
    if (!status) throw std::runtime_error(std::string(message) + ": " + status.message);
}
template <class T> static std::vector<T> read(DeviceSpan<const T> data) {
    std::vector<T> output(data.size);
    require(cudaMemcpy(output.data(), data.data, data.size * sizeof(T),
        cudaMemcpyDeviceToHost) == cudaSuccess, "device read failed");
    return output;
}

int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        SceneDefinition scene;
        std::string error;
        require(load_glb_scene(PARALLEL_MATER_ROPE_CLOTH_SCENE_PATH, scene, error),
            error.c_str());
        require(scene.ropes.size() == 4 && scene.cloths.size() == 1,
            "expected four post-to-cloth ropes and one sheet");
        const auto &mesh = scene.meshes[scene.cloths[0].mesh_index];
        require(mesh.vertices.size() >= 200 && mesh.indices.size() >= 1200,
            "pre-Cloth Simple subdivision was not exported");
        std::set<unsigned> corners;
        for (const auto &rope : scene.ropes) {
            require(rope.first_body >= 0 && rope.last_cloth == 0 &&
                rope.last_cloth_vertex < mesh.vertices.size() &&
                rope.first_cloth < 0 && rope.last_body < 0,
                "post/cloth endpoint inference failed");
            corners.insert(rope.last_cloth_vertex);
        }
        require(corners.size() == 4, "cloth ropes must bind four distinct corners");

        World world;
        SceneInstance instance;
        require(create_scene_world(scene, world, instance), "create bridge world");
        require(instance.rope_cloth_couplings.size() == 4,
            "four rope-cloth API joints were not registered");
        require(!world.remove_cloth(instance.cloths[0]),
            "coupled cloth was removed");
        require(!world.remove_rope(instance.ropes[0]),
            "coupled rope was removed");
        RopeClothCouplingId duplicate{};
        require(!world.add_rope_cloth_coupling({
            .rope = instance.ropes[0], .cloth = instance.cloths[0],
            .last_vertex = scene.ropes[0].last_cloth_vertex}, duplicate),
            "duplicate rope-cloth endpoint accepted");
        if (argc > 1) {
            const float limit = std::stof(argv[1]);
            for (unsigned index = 0; index < instance.rope_cloth_couplings.size(); ++index)
                require(world.update_rope_cloth_coupling(
                    instance.rope_cloth_couplings[index], {
                        .rope = instance.ropes[index], .cloth = instance.cloths[0],
                        .last_vertex = scene.ropes[index].last_cloth_vertex,
                        .maximum_cloth_acceleration = limit}),
                    "set trial rope-cloth impulse limit");
        }

        float peak_force = 0.0F;
        float peak_gap = 0.0F;
        float peak_strain = 0.0F;
        unsigned peak_frame = 0U, peak_rope = 0U, peak_edge = 0U;
        float peak_edge_length = 0.0F, peak_edge_rest = 0.0F;
        Vec3 peak_a{}, peak_b{};
        const unsigned frames = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 300U;
        for (unsigned frame = 0; frame < frames; ++frame) {
            require(world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                .gravity = {0.0F, -9.81F, 0.0F}}), "step bridge");
            if (frame % 30U != 29U) continue;
            ClothDeviceView cloth{};
            require(world.cloth_view(instance.cloths[0], cloth), "cloth view");
            auto positions = read(cloth.positions);
            auto forces = read(cloth.rope_contact_forces);
            require(positions.size() == mesh.vertices.size(),
                "cloth topology changed unexpectedly");
            for (unsigned vertex : corners) {
                const auto &point = positions[vertex];
                require(std::isfinite(point.x) && std::isfinite(point.y) &&
                    std::isfinite(point.z), "non-finite cloth corner");
                peak_force = std::max(peak_force, math::length(forces[vertex]));
            }
            for (unsigned index = 0; index < scene.ropes.size(); ++index) {
                RopeDeviceView rope{};
                require(world.rope_view(instance.ropes[index], rope), "rope view");
                const auto nodes = read(rope.positions);
                const auto rest = read(rope.rest_lengths);
                peak_gap = std::max(peak_gap, math::length(math::subtract(
                    nodes.back(), positions[scene.ropes[index].last_cloth_vertex])));
                for (unsigned edge = 0; edge < rest.size(); ++edge) {
                    const float length = math::length(math::subtract(
                        nodes[edge + 1], nodes[edge]));
                    const float strain = std::abs(length / rest[edge] - 1.0F);
                    if (strain > peak_strain) {
                        peak_strain = strain;
                        peak_frame = frame;
                        peak_rope = index;
                        peak_edge = edge;
                        peak_edge_length = length;
                        peak_edge_rest = rest[edge];
                        peak_a = nodes[edge];
                        peak_b = nodes[edge + 1];
                    }
                }
            }
        }
        ClothDeviceView cloth{};
        require(world.cloth_view(instance.cloths[0], cloth), "final cloth view");
        const auto attached = read(cloth.positions);
        float attached_height = 0.0F;
        for (unsigned vertex : corners) attached_height += attached[vertex].y / 4.0F;
        std::cout << "precheck peak_force=" << peak_force
                  << " peak_gap=" << peak_gap
                  << " peak_strain=" << peak_strain
                  << " peak_frame=" << peak_frame
                  << " peak_rope=" << peak_rope
                  << " peak_edge=" << peak_edge
                  << " edge_length=" << peak_edge_length
                  << " edge_rest=" << peak_edge_rest
                  << " a=(" << peak_a.x << ',' << peak_a.y << ',' << peak_a.z << ')'
                  << " b=(" << peak_b.x << ',' << peak_b.y << ',' << peak_b.z << ')'
                  << " attached_height=" << attached_height << '\n';
        require(peak_force > 0.1F, "rope tension did not reach cloth API force view");
        require(peak_gap < 0.03F, "rope endpoint separated from cloth vertex");
        require(peak_strain < 0.10F, "suspended rope stretched excessively");
        require(attached_height > 0.15F, "rope failed to suspend cloth corners");

        World loose_world;
        SceneInstance loose;
        require(create_scene_world(scene, loose_world, loose), "create loose comparison");
        for (unsigned index = 0; index < loose.rope_cloth_couplings.size(); ++index) {
            auto options = RopeClothCouplingOptions{
                .rope = loose.ropes[index], .cloth = loose.cloths[0],
                .last_vertex = scene.ropes[index].last_cloth_vertex,
                .enabled = false};
            require(loose_world.update_rope_cloth_coupling(
                loose.rope_cloth_couplings[index], options),
                "disable comparison coupling");
        }
        for (unsigned frame = 0; frame < 120; ++frame)
            require(loose_world.step({.timestep = 1.0F / 60.0F,
                .substeps = 4U, .gravity = {0.0F, -9.81F, 0.0F}}),
                "step loose comparison");
        require(loose_world.cloth_view(loose.cloths[0], cloth), "loose cloth view");
        const auto loose_positions = read(cloth.positions);
        float loose_height = 0.0F;
        for (unsigned vertex : corners) loose_height += loose_positions[vertex].y / 4.0F;
        require(attached_height > loose_height + 0.10F,
            "coupled cloth was not suspended above loose cloth");
        std::cout << "bridge_vertices=" << mesh.vertices.size()
                  << " peak_rope_force=" << peak_force
                  << " peak_gap=" << peak_gap
                  << " peak_strain=" << peak_strain
                  << " attached_height=" << attached_height
                  << " loose_height=" << loose_height << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
