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

bool require(parallel_mater::Status status, const char *operation) {
    if (status) return true;
    std::cerr << operation << ": " <<
        (status.message != nullptr ? status.message : "unknown") << '\n';
    return false;
}

bool read_cloth(const parallel_mater::World &world,
                parallel_mater::ClothId id,
                std::vector<parallel_mater::Vec3> &positions) {
    parallel_mater::ClothDeviceView view{};
    if (!require(world.cloth_view(id, view), "borrow cloth")) return false;
    positions.resize(view.vertex_count);
    return cudaMemcpy(positions.data(), view.positions.data,
                      positions.size() * sizeof(positions[0]),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

float distance(parallel_mater::Vec3 a, parallel_mater::Vec3 b) {
    const float x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
    return std::sqrt(x * x + y * y + z * z);
}

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    SceneDefinition scene{};
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_CLOTH_SCENE_PATH, scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (scene.cloths.size() != 1U || scene.rigid_bodies.size() != 2U) {
        std::cerr << "Cloth scene needs one cloth and two rigid bodies\n";
        return 1;
    }
    const ClothDefinition &cloth = scene.cloths.front();
    const TriangleMesh &mesh = scene.meshes[cloth.mesh_index];
    const std::size_t pinned = std::count(cloth.inverse_masses.begin(),
                                           cloth.inverse_masses.end(), 0.0F);
    if (mesh.vertices.size() != 1089U || pinned != 66U) {
        std::cerr << "Expected 33x33 cloth and 66 pinned edge vertices; got "
                  << mesh.vertices.size() << " and " << pinned << '\n';
        return 1;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    World world;
    if (!require(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U,
                                .cloth_capacity = 1U}, world),
                 "create cloth world")) return 1;
    SceneInstance instance{};
    if (!require(instantiate_scene(scene, world, instance),
                 "instantiate cloth scene")) return 1;
    std::vector<Vec3> initial;
    if (!read_cloth(world, instance.cloths.front(), initial)) return 1;
    std::size_t active = 0U;
    for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index)
        if (scene.rigid_bodies[index].options.motion == MotionType::dynamic)
            active = index;
    RigidBodyState sphere_initial{};
    if (!require(world.read_rigid_body_state(instance.rigid_bodies[active],
                                             sphere_initial), "read active body"))
        return 1;
    const StepOptions step{.timestep = 1.0F / 60.0F,
                           .substeps = 4U,
                           .gravity = {0.0F, -6.93671752F, -6.93671752F}};
    for (int frame = 0; frame < 600; ++frame)
        if (!require(world.step(step), "step cloth world")) return 1;
    std::vector<Vec3> final;
    if (!read_cloth(world, instance.cloths.front(), final)) return 1;
    float pinned_motion = 0.0F, free_motion = 0.0F;
    for (std::size_t index = 0U; index < final.size(); ++index) {
        if (!std::isfinite(final[index].x) || !std::isfinite(final[index].y) ||
            !std::isfinite(final[index].z)) {
            std::cerr << "Cloth vertex became non-finite\n";
            return 1;
        }
        const float motion = distance(initial[index], final[index]);
        if (cloth.inverse_masses[index] == 0.0F)
            pinned_motion = std::max(pinned_motion, motion);
        else free_motion = std::max(free_motion, motion);
    }
    RigidBodyState sphere_final{};
    if (!require(world.read_rigid_body_state(instance.rigid_bodies[active],
                                             sphere_final), "read final active body"))
        return 1;
    std::cout << "Cloth vertices=" << final.size() << " pinned=" << pinned
              << " pinned_motion=" << pinned_motion
              << " free_motion=" << free_motion
              << " sphere_initial_z=" << sphere_initial.position.z
              << " sphere_final_z=" << sphere_final.position.z
              << " sphere_final_y=" << sphere_final.position.y << '\n';
    if (pinned_motion > 1.0e-5F || free_motion < 0.02F ||
        sphere_final.position.z >= sphere_initial.position.z - 0.1F)
        return 1;

    SceneDefinition rigid_only = scene;
    rigid_only.cloths.clear();
    World no_cloth_world;
    if (!require(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U,
                                .cloth_capacity = 0U}, no_cloth_world),
                 "create rigid-only comparison world")) return 1;
    SceneInstance no_cloth_instance{};
    if (!require(instantiate_scene(rigid_only, no_cloth_world, no_cloth_instance),
                 "instantiate rigid-only comparison")) return 1;
    for (int frame = 0; frame < 600; ++frame)
        if (!require(no_cloth_world.step(step), "step rigid-only comparison"))
            return 1;
    RigidBodyState no_cloth_sphere{};
    if (!require(no_cloth_world.read_rigid_body_state(
            no_cloth_instance.rigid_bodies[active], no_cloth_sphere),
            "read rigid-only sphere")) return 1;
    std::cout << "Rigid-only sphere_final_z=" << no_cloth_sphere.position.z
              << " cloth_impact_delta_z=" <<
                  sphere_final.position.z - no_cloth_sphere.position.z << '\n';
    if (sphere_final.position.z - no_cloth_sphere.position.z < 0.5F) {
        std::cerr << "Cloth did not stop the rigid body\n";
        return 1;
    }

    SceneDefinition cloth_only = scene;
    cloth_only.rigid_bodies.erase(cloth_only.rigid_bodies.begin() + active);
    World no_sphere_world;
    if (!require(World::create({.rigid_body_capacity = 1U,
                                .triangle_mesh_capacity = 1U,
                                .cloth_capacity = 1U}, no_sphere_world),
                 "create cloth-only comparison world")) return 1;
    SceneInstance no_sphere_instance{};
    if (!require(instantiate_scene(cloth_only, no_sphere_world,
                                   no_sphere_instance),
                 "instantiate cloth-only comparison")) return 1;
    for (int frame = 0; frame < 600; ++frame)
        if (!require(no_sphere_world.step(step), "step cloth-only comparison"))
            return 1;
    std::vector<Vec3> no_sphere_positions;
    if (!read_cloth(no_sphere_world, no_sphere_instance.cloths.front(),
                    no_sphere_positions)) return 1;
    float body_to_cloth_motion = 0.0F;
    for (std::size_t index = 0U; index < final.size(); ++index)
        if (cloth.inverse_masses[index] != 0.0F)
            body_to_cloth_motion = std::max(body_to_cloth_motion,
                distance(final[index], no_sphere_positions[index]));
    std::cout << "Rigid impact cloth displacement=" << body_to_cloth_motion
              << '\n';
    if (body_to_cloth_motion < 0.01F) {
        std::cerr << "Rigid body did not deform the cloth\n";
        return 1;
    }

    const ClothId removed = instance.cloths.front();
    if (!require(world.remove_cloth(removed), "remove cloth")) return 1;
    ClothDeviceView stale{};
    if (world.cloth_view(removed, stale).code != StatusCode::invalid_handle) {
        std::cerr << "Removed cloth handle remained valid\n";
        return 1;
    }
    std::vector<Vec3> rest;
    rest.reserve(mesh.vertices.size());
    for (const Vertex &vertex : mesh.vertices) rest.push_back(vertex.position);
    ClothId replacement{};
    if (!require(world.add_cloth({
            .vertices = {rest.data(), rest.size()},
            .triangle_indices = {mesh.indices.data(), mesh.indices.size()},
            .inverse_masses = {cloth.inverse_masses.data(),
                               cloth.inverse_masses.size()},
            .vertex_mass = cloth.vertex_mass,
            .thickness = cloth.thickness}, replacement),
            "reuse cloth slot")) return 1;
    if (replacement.index != removed.index ||
        replacement.generation == removed.generation) {
        std::cerr << "Reused cloth slot did not advance generation\n";
        return 1;
    }
    return 0;
}
