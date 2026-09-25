// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace parallel_mater;
using namespace parallel_mater::gallery;

bool check(Status status, const char *operation) {
    if (status) return true;
    std::cerr << operation << ": " <<
        (status.message ? status.message : "unknown") << '\n';
    return false;
}

float length(Vec3 a, Vec3 b) {
    const float x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
    return std::sqrt(x * x + y * y + z * z);
}

bool run_tear() {
    SceneDefinition scene{};
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH,
                        scene, error)) {
        std::cerr << error << '\n';
        return false;
    }
    if (scene.cloths.size() != 1U || scene.cloths[0].tear_ratio <= 1.0F)
        return false;
    World world;
    if (!check(World::create({.rigid_body_capacity = 2U,
                              .triangle_mesh_capacity = 4U,
                              .cloth_capacity = 1U}, world), "create tear world"))
        return false;
    SceneInstance instance{};
    if (!check(instantiate_scene(scene, world, instance), "instantiate tear"))
        return false;
    std::size_t active = 0U;
    for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index)
        if (scene.rigid_bodies[index].options.motion == MotionType::dynamic)
            active = index;
    RigidBodyState initial_body{}, settled_body{}, final_body{};
    if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                          initial_body), "read tear body"))
        return false;
    for (int frame = 0; frame < 120; ++frame)
        if (!check(world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                               .gravity = {0.0F, -9.81F, 0.0F}}),
                   "settle tear ball")) return false;
    if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                          settled_body), "read settled ball"))
        return false;
    ClothDeviceView view{};
    if (!check(world.cloth_view(instance.cloths[0], view), "view tear"))
        return false;
    std::vector<Vec3> positions(view.vertex_count);
    std::vector<std::uint32_t> indices(view.triangle_indices.size);
    const auto read_indices = [&]() {
        return cudaMemcpy(indices.data(), view.triangle_indices.data,
            indices.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    if (!read_indices()) return false;
    std::uint32_t settled_torn = 0U;
    for (std::size_t base = 0U; base < indices.size(); base += 3U)
        settled_torn += indices[base] == indices[base + 1U];
    int first_cut_frame = -1;
    float first_cut_body_z = INFINITY;
    for (int frame = 0; frame < 300; ++frame) {
        if (!check(world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                               .gravity = {0.0F, -6.93671752F,
                                           -6.93671752F}}),
                   "roll tear ball")) return false;
        if (first_cut_frame >= 0) continue;
        if (!read_indices()) return false;
        for (std::size_t base = 0U; base < indices.size(); base += 3U) {
            if (indices[base] != indices[base + 1U]) continue;
            RigidBodyState cut_body{};
            if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                                  cut_body), "read cut body"))
                return false;
            first_cut_frame = frame;
            first_cut_body_z = cut_body.position.z;
            break;
        }
    }
    if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                          final_body), "read torn body"))
        return false;
    if (cudaMemcpy(positions.data(), view.positions.data,
                   positions.size() * sizeof(Vec3),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        !read_indices()) return false;
    const TriangleMesh &mesh = scene.meshes[scene.cloths[0].mesh_index];
    std::uint32_t torn = 0U;
    float maximum_ratio = 0.0F;
    float cut_min_x = INFINITY, cut_max_x = -INFINITY;
    float cut_min_y = INFINITY, cut_max_y = -INFINITY;
    for (std::size_t base = 0U; base < indices.size(); base += 3U) {
        if (indices[base] == indices[base + 1U]) {
            ++torn;
            const auto &a = mesh.vertices[mesh.indices[base]].position;
            const auto &b = mesh.vertices[mesh.indices[base + 1U]].position;
            const auto &c = mesh.vertices[mesh.indices[base + 2U]].position;
            const float x = (a.x + b.x + c.x) / 3.0F;
            const float y = (a.y + b.y + c.y) / 3.0F;
            cut_min_x = std::min(cut_min_x, x);
            cut_max_x = std::max(cut_max_x, x);
            cut_min_y = std::min(cut_min_y, y);
            cut_max_y = std::max(cut_max_y, y);
            continue;
        }
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const auto a = indices[base + edge];
            const auto b = indices[base + (edge + 1U) % 3U];
            maximum_ratio = std::max(maximum_ratio,
                length(positions[a], positions[b]) /
                length(mesh.vertices[a].position, mesh.vertices[b].position));
        }
    }
    std::cout << "Tear before_roll_removed_triangles=" << settled_torn
              << " removed_triangles=" << torn
              << " total_triangles=" << indices.size() / 3U
              << " first_cut_frame=" << first_cut_frame
              << " first_cut_body_z=" << first_cut_body_z
              << " contact_cut_scale=" << scene.cloths[0].contact_cut_radius_scale
              << " cut_bounds_x=" << cut_min_x << ".." << cut_max_x
              << " cut_bounds_y=" << cut_min_y << ".." << cut_max_y
              << " remaining_max_edge_ratio=" << maximum_ratio
              << " initial_body_y=" << initial_body.position.y
              << " settled_body_y=" << settled_body.position.y
              << " final_body_y=" << final_body.position.y
              << " initial_body_z=" << initial_body.position.z
              << " settled_body_z=" << settled_body.position.z
              << " final_body_z=" << final_body.position.z << '\n';
    SceneDefinition no_impact = scene;
    no_impact.rigid_bodies.erase(std::remove_if(
        no_impact.rigid_bodies.begin(), no_impact.rigid_bodies.end(),
        [](const RigidBodyDefinition &body) {
            return body.options.motion == MotionType::dynamic;
        }), no_impact.rigid_bodies.end());
    World baseline;
    if (!check(World::create({.rigid_body_capacity = 1U,
                              .triangle_mesh_capacity = 4U,
                              .cloth_capacity = 1U}, baseline),
               "create no-impact world")) return false;
    SceneInstance baseline_instance{};
    if (!check(instantiate_scene(no_impact, baseline, baseline_instance),
               "instantiate no-impact cloth")) return false;
    for (int frame = 0; frame < 180; ++frame)
        if (!check(baseline.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                                  .gravity = {0.0F, -6.93671752F,
                                              -6.93671752F}}),
                   "step no-impact cloth")) return false;
    ClothDeviceView baseline_view{};
    if (!check(baseline.cloth_view(baseline_instance.cloths[0], baseline_view),
               "view no-impact cloth")) return false;
    std::vector<std::uint32_t> baseline_indices(baseline_view.triangle_indices.size);
    if (cudaMemcpy(baseline_indices.data(), baseline_view.triangle_indices.data,
                   baseline_indices.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    std::uint32_t baseline_torn = 0U;
    for (std::size_t base = 0U; base < baseline_indices.size(); base += 3U)
        baseline_torn += baseline_indices[base] == baseline_indices[base + 1U];
    std::cout << "No-impact removed_triangles=" << baseline_torn << '\n';
    const float width = cut_max_x - cut_min_x;
    const float height = cut_max_y - cut_min_y;
    return length(initial_body.linear_velocity, {}) < 1.0e-5F &&
           settled_torn == 0U && first_cut_frame >= 0 &&
           first_cut_body_z < 1.0F && torn > 0U &&
           torn < (indices.size() / 3U) / 4U &&
           baseline_torn == 0U &&
           scene.cloths[0].contact_cut_radius_scale > 0.0F &&
           width < 1.5F && height < 1.5F &&
           initial_body.position.y > settled_body.position.y &&
           std::abs(settled_body.position.z - initial_body.position.z) < 0.2F &&
           final_body.position.z < -0.5F &&
           maximum_ratio <= scene.cloths[0].tear_ratio + 0.05F;
}

bool run_paint() {
    SceneDefinition scene{};
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_CLOTH_PAINT_SCENE_PATH,
                        scene, error)) {
        std::cerr << error << '\n';
        return false;
    }
    if (scene.cloths.size() != 1U || !scene.cloths[0].paintable ||
        scene.cloths[0].tear_ratio != 0.0F ||
        scene.cloths[0].paint_source.empty() ||
        !scene.initial_particles.empty()) return false;
    World world;
    if (!check(World::create({.rigid_body_capacity = 2U,
                              .triangle_mesh_capacity = 5U,
                              .paint_field_capacity = 1U,
                              .paint_rule_capacity = 1U,
                              .cloth_capacity = 1U}, world),
               "create paint world")) return false;
    SceneInstance instance{};
    if (!check(instantiate_scene(scene, world, instance),
               "instantiate paint")) return false;
    if (instance.paint_bindings.size() != 1U || instance.has_fluid)
        return false;
    PaintFieldDeviceView view{};
    if (!check(world.paint_field_view(instance.paint_bindings[0].field, view),
               "view initial paint")) return false;
    std::vector<std::uint32_t> pixels(view.pixels.size);
    if (cudaMemcpy(pixels.data(), view.pixels.data,
                   pixels.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        std::any_of(pixels.begin(), pixels.end(),
                    [](std::uint32_t value) { return value != 0U; }))
        return false;
    ClothDeviceView cloth_view{};
    if (!check(world.cloth_view(instance.cloths[0], cloth_view),
               "view initial paint cloth")) return false;
    std::vector<Vec3> initial(cloth_view.vertex_count);
    if (cudaMemcpy(initial.data(), cloth_view.positions.data,
                   initial.size() * sizeof(Vec3),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (int frame = 0; frame < 180; ++frame)
        if (!check(world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                               .gravity = {0.0F, -9.81F, 0.0F}}),
                   "step paint")) return false;
    if (!check(world.cloth_view(instance.cloths[0], cloth_view),
               "view final paint cloth")) return false;
    std::vector<Vec3> final(cloth_view.vertex_count);
    std::vector<std::uint32_t> triangles(cloth_view.triangle_indices.size);
    if (cudaMemcpy(final.data(), cloth_view.positions.data,
                   final.size() * sizeof(Vec3),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(triangles.data(), cloth_view.triangle_indices.data,
                   triangles.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    float free_motion = 0.0F, pinned_motion = 0.0F;
    for (std::size_t index = 0U; index < final.size(); ++index) {
        const float motion = length(initial[index], final[index]);
        if (scene.cloths[0].inverse_masses[index] == 0.0F)
            pinned_motion = std::max(pinned_motion, motion);
        else free_motion = std::max(free_motion, motion);
    }
    if (!check(world.paint_field_view(instance.paint_bindings[0].field, view),
               "view paint")) return false;
    if (cudaMemcpy(pixels.data(), view.pixels.data,
                   pixels.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    const auto painted = std::count_if(pixels.begin(), pixels.end(),
                                      [](std::uint32_t value) { return value != 0U; });
    const auto front_painted = std::count_if(pixels.begin(), pixels.end(),
                                      [](std::uint32_t value) { return (value & 1U) != 0U; });
    const auto back_painted = std::count_if(pixels.begin(), pixels.end(),
                                      [](std::uint32_t value) { return (value & 2U) != 0U; });
    std::cout << "Rigid-cloth painted_texels=" << painted
              << " front=" << front_painted << " back=" << back_painted
              << " free_motion=" << free_motion
              << " pinned_motion=" << pinned_motion << '\n';
    return front_painted >= 500U && painted >= 500U &&
           free_motion > 0.02F &&
           pinned_motion < 1.0e-5F &&
           triangles == scene.meshes[scene.cloths[0].mesh_index].indices;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    return run_tear() && run_paint() ? 0 : 1;
}
