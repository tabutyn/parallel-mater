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

bool run_api_fracture() {
    const std::vector<Vec3> vertices{{-0.5F, 1.0F, 0.0F},
                                     { 0.5F, 1.0F, 0.0F},
                                     {-0.5F, 0.0F, 0.0F},
                                     { 0.5F, 0.0F, 0.0F}};
    const std::vector<std::uint32_t> triangles{0U, 2U, 1U, 1U, 2U, 3U};
    const std::vector<float> inverse_masses{0.0F, 0.0F, 1.0F, 1.0F};
    World world;
    if (!check(World::create({.rigid_body_capacity = 1U,
                              .cloth_capacity = 1U}, world),
               "create API fracture world")) return false;
    ClothId cloth{};
    std::vector<Vec3> degenerate = vertices;
    degenerate[1] = degenerate[0];
    if (world.add_cloth({.vertices = {degenerate.data(), degenerate.size()},
                         .triangle_indices = {triangles.data(), triangles.size()}},
                        cloth).code != StatusCode::invalid_argument) return false;
    if (!check(world.add_cloth({.vertices = {vertices.data(), vertices.size()},
                               .triangle_indices = {triangles.data(), triangles.size()},
                               .inverse_masses = {inverse_masses.data(), inverse_masses.size()},
                               .stretch_compliance = 0.2F,
                               .bending_compliance = 0.2F,
                               .velocity_damping = 0.0F,
                               .break_strain = 0.05F,
                               .fracture_persistence_substeps = 1U}, cloth),
               "add API fracture cloth")) return false;
    ClothDeviceView view{};
    if (!check(world.cloth_view(cloth, view), "view API fracture cloth") ||
        view.vertex_count != vertices.size() ||
        view.triangle_indices.size != triangles.size() ||
        view.surface_positions.size != triangles.size() ||
        view.surface_triangle_indices.size != triangles.size() ||
        view.surface_source_indices.size != triangles.size() ||
        view.bonds.size == 0U ||
        view.active_bonds.size != view.bonds.size) return false;
    for (int frame = 0; frame < 12; ++frame)
        if (!check(world.step({.gravity = {0.0F, -50.0F, 0.0F}}),
                   "step API fracture cloth")) return false;
    std::vector<std::uint8_t> active(view.active_bonds.size);
    std::vector<ClothBond> bonds(view.bonds.size);
    std::vector<std::uint32_t> source(view.surface_source_indices.size);
    std::vector<std::uint32_t> surface_triangles(view.surface_triangle_indices.size);
    std::vector<std::uint32_t> original_triangles(view.triangle_indices.size);
    std::vector<Vec3> surface(view.surface_positions.size);
    if (cudaMemcpy(active.data(), view.active_bonds.data,
                   active.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(bonds.data(), view.bonds.data,
                   bonds.size() * sizeof(ClothBond), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(source.data(), view.surface_source_indices.data,
                   source.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(surface_triangles.data(), view.surface_triangle_indices.data,
                   surface_triangles.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(original_triangles.data(), view.triangle_indices.data,
                   original_triangles.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(surface.data(), view.surface_positions.data,
                   surface.size() * sizeof(Vec3), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    const std::size_t broken = std::count(active.begin(), active.end(), std::uint8_t{0U});
    const bool valid_bonds = std::all_of(bonds.begin(), bonds.end(),
        [&](const ClothBond &bond) {
            return bond.first < vertices.size() && bond.second < vertices.size() &&
                   bond.rest_length > 0.0F;
        });
    const bool stable_surface = std::all_of(surface.begin(), surface.end(),
        [](Vec3 point) {
            return std::isfinite(point.x) && std::isfinite(point.y) &&
                   std::isfinite(point.z);
        });
    std::cout << "API fracture broken_bonds=" << broken << "/" << bonds.size()
              << " retained_triangles=" << original_triangles.size() / 3U << '\n';
    for (std::size_t corner = 0U; corner < triangles.size(); ++corner)
        if (source[corner] != triangles[corner] ||
            surface_triangles[corner] != corner) return false;
    return broken > 0U && broken < bonds.size() && valid_bonds && stable_surface &&
           original_triangles == triangles;
}

bool run_tear() {
    SceneDefinition scene{};
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH,
                        scene, error)) {
        std::cerr << error << '\n';
        return false;
    }
    if (scene.cloths.size() != 1U || scene.cloths[0].break_strain <= 0.0F)
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
    ClothDeviceView view{};
    if (!check(world.cloth_view(instance.cloths[0], view), "view tear"))
        return false;
    const TriangleMesh &mesh = scene.meshes[scene.cloths[0].mesh_index];
    if (view.surface_positions.size != mesh.indices.size() ||
        view.surface_triangle_indices.size != mesh.indices.size() ||
        view.active_bonds.size == 0U) return false;
    std::vector<std::uint8_t> bonds(view.active_bonds.size);
    const auto broken_bonds = [&]() {
        if (cudaMemcpy(bonds.data(), view.active_bonds.data,
                       bonds.size() * sizeof(std::uint8_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return static_cast<std::size_t>(-1);
        return static_cast<std::size_t>(std::count(
            bonds.begin(), bonds.end(), std::uint8_t{0U}));
    };
    for (int frame = 0; frame < 120; ++frame)
        if (!check(world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                               .gravity = {0.0F, -9.81F, 0.0F}}),
                   "settle tear ball")) return false;
    if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                          settled_body), "read settled ball"))
        return false;
    const std::size_t settled_broken = broken_bonds();
    int first_break_frame = -1;
    int first_pass_frame = -1;
    float first_break_body_z = INFINITY;
    float first_pass_velocity_z = 0.0F;
    float remote_max_ratio = 0.0F, contact_max_ratio = 0.0F;
    std::vector<Vec3> physical(view.vertex_count);
    for (int frame = 0; frame < 300; ++frame) {
        if (!check(world.step({.timestep = 1.0F / 60.0F, .substeps = 4U,
                               .gravity = {0.0F, -6.93671752F,
                                           -6.93671752F}}),
                   "roll tear ball")) return false;
        RigidBodyState sampled_body{};
        if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                               sampled_body), "sample tear body") ||
            cudaMemcpy(physical.data(), view.positions.data,
                       physical.size() * sizeof(Vec3),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (first_pass_frame < 0 && sampled_body.position.z < -0.5F) {
            first_pass_frame = frame;
            first_pass_velocity_z = sampled_body.linear_velocity.z;
        }
        for (std::size_t base = 0U; base < mesh.indices.size(); base += 3U)
            for (std::size_t edge = 0U; edge < 3U; ++edge) {
                const auto a = mesh.indices[base + edge];
                const auto b = mesh.indices[base + (edge + 1U) % 3U];
                const float ratio = length(physical[a], physical[b]) /
                    length(mesh.vertices[a].position, mesh.vertices[b].position);
                if (sampled_body.position.z > 1.0F)
                    remote_max_ratio = std::max(remote_max_ratio, ratio);
                else contact_max_ratio = std::max(contact_max_ratio, ratio);
            }
        if (first_break_frame < 0 && broken_bonds() > settled_broken) {
            RigidBodyState cut_body{};
            if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                                  cut_body), "read cut body"))
                return false;
            first_break_frame = frame;
            first_break_body_z = cut_body.position.z;
        }
    }
    if (!check(world.read_rigid_body_state(instance.rigid_bodies[active],
                                          final_body), "read torn body"))
        return false;
    const std::size_t broken = broken_bonds();
    std::vector<Vec3> surface(view.surface_positions.size);
    std::vector<std::uint32_t> indices(view.triangle_indices.size);
    if (cudaMemcpy(surface.data(), view.surface_positions.data,
                   surface.size() * sizeof(Vec3),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(indices.data(), view.triangle_indices.data,
                   indices.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    float maximum_ratio = 0.0F;
    for (std::size_t base = 0U; base < indices.size(); base += 3U) {
        if (indices[base] != mesh.indices[base] ||
            indices[base + 1U] != mesh.indices[base + 1U] ||
            indices[base + 2U] != mesh.indices[base + 2U]) return false;
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const auto a = mesh.indices[base + edge];
            const auto b = mesh.indices[base + (edge + 1U) % 3U];
            maximum_ratio = std::max(maximum_ratio,
                length(surface[base + edge],
                       surface[base + (edge + 1U) % 3U]) /
                length(mesh.vertices[a].position, mesh.vertices[b].position));
        }
    }
    std::cout << "Tear settled_broken=" << settled_broken
              << " broken_bonds=" << broken << "/" << bonds.size()
              << " retained_triangles=" << indices.size() / 3U
              << " first_break_frame=" << first_break_frame
              << " first_break_body_z=" << first_break_body_z
              << " first_pass_frame=" << first_pass_frame
              << " first_pass_velocity_z=" << first_pass_velocity_z
              << " remote_max_ratio=" << remote_max_ratio
              << " contact_max_ratio=" << contact_max_ratio
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
    std::vector<std::uint8_t> baseline_bonds(baseline_view.active_bonds.size);
    if (cudaMemcpy(baseline_bonds.data(), baseline_view.active_bonds.data,
                   baseline_bonds.size() * sizeof(std::uint8_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    const std::size_t baseline_broken = std::count(
        baseline_bonds.begin(), baseline_bonds.end(), std::uint8_t{0U});
    std::cout << "No-impact broken_bonds=" << baseline_broken << '\n';
    return length(initial_body.linear_velocity, {}) < 1.0e-5F &&
           settled_broken == 0U && first_break_frame >= 0 &&
           first_break_body_z < 1.0F && broken > 0U &&
           first_pass_frame > first_break_frame &&
           first_pass_frame - first_break_frame <= 24 &&
           first_pass_velocity_z < -1.0F &&
           broken < bonds.size() / 3U && baseline_broken == 0U &&
           initial_body.position.y > settled_body.position.y &&
           std::abs(settled_body.position.z - initial_body.position.z) < 0.2F &&
           final_body.position.z < -0.5F &&
           maximum_ratio <= 1.15F;
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
        scene.cloths[0].break_strain != 0.0F ||
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
    return run_api_fracture() && run_tear() && run_paint() ? 0 : 1;
}
