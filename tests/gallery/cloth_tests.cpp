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

struct MotionMetric {
    float squared_sum{};
    float peak{};
    std::uint64_t count{};

    void observe(float value) {
        squared_sum += value * value;
        peak = std::max(peak, value);
        ++count;
    }

    [[nodiscard]] float rms() const {
        return count != 0U ? std::sqrt(squared_sum / static_cast<float>(count))
                           : 0.0F;
    }
};

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
    MotionMetric settled_speed, settled_velocity_jump, contact_patch_motion;
    MotionMetric nearest_normal_jump;
    std::uint32_t nearest_vertex_switches = 0U;
    std::size_t previous_nearest_vertex = initial.size();
    Vec3 previous_nearest_normal{};
    std::vector<Vec3> sampled_cloth;
    std::vector<Vec3> previous_sampled_cloth;
    RigidBodyState previous_sphere = sphere_initial;
    for (int frame = 0; frame < 600; ++frame) {
        if (!require(world.step(step), "step cloth world")) return 1;
        RigidBodyState current_sphere{};
        if (!require(world.read_rigid_body_state(instance.rigid_bodies[active],
                                                 current_sphere),
                     "sample active body")) return 1;
        if (frame >= 240) {
            settled_speed.observe(distance(current_sphere.linear_velocity, {}));
            settled_velocity_jump.observe(distance(
                current_sphere.linear_velocity,
                previous_sphere.linear_velocity));
            if (!read_cloth(world, instance.cloths.front(), sampled_cloth))
                return 1;
            if (!previous_sampled_cloth.empty()) {
                for (std::size_t index = 0U;
                     index < sampled_cloth.size(); ++index) {
                    if (distance(sampled_cloth[index],
                                 current_sphere.position) > 1.0F) continue;
                    contact_patch_motion.observe(distance(
                        sampled_cloth[index], previous_sampled_cloth[index]));
                }
            }
            float nearest_squared = 1.0e30F;
            std::size_t nearest_vertex = sampled_cloth.size();
            for (std::size_t index = 0U; index < sampled_cloth.size(); ++index) {
                const Vec3 delta{
                    sampled_cloth[index].x - current_sphere.position.x,
                    sampled_cloth[index].y - current_sphere.position.y,
                    sampled_cloth[index].z - current_sphere.position.z};
                const float squared = delta.x * delta.x +
                    delta.y * delta.y + delta.z * delta.z;
                if (squared < nearest_squared) {
                    nearest_squared = squared;
                    nearest_vertex = index;
                }
            }
            const Vec3 nearest_normal{
                (current_sphere.position.x - sampled_cloth[nearest_vertex].x) /
                    std::sqrt(nearest_squared),
                (current_sphere.position.y - sampled_cloth[nearest_vertex].y) /
                    std::sqrt(nearest_squared),
                (current_sphere.position.z - sampled_cloth[nearest_vertex].z) /
                    std::sqrt(nearest_squared)};
            if (previous_nearest_vertex != initial.size()) {
                nearest_vertex_switches += nearest_vertex !=
                    previous_nearest_vertex;
                nearest_normal_jump.observe(distance(
                    nearest_normal, previous_nearest_normal));
            }
            previous_nearest_vertex = nearest_vertex;
            previous_nearest_normal = nearest_normal;
            previous_sampled_cloth = sampled_cloth;
        }
        previous_sphere = current_sphere;
    }
    std::cout << "contact settled_peak_speed=" << settled_speed.peak
              << " settled_rms_velocity_jump=" << settled_velocity_jump.rms()
              << " nearest_vertex_switches=" << nearest_vertex_switches
              << " nearest_normal_rms_jump=" << nearest_normal_jump.rms()
              << " contact_patch_rms_displacement=" <<
                 contact_patch_motion.rms()
              << " contact_patch_peak_displacement=" <<
                 contact_patch_motion.peak
              << '\n';
    if (settled_velocity_jump.rms() > 0.06F ||
        contact_patch_motion.rms() > 0.002F ||
        contact_patch_motion.peak > 0.025F) {
        std::cerr << "Ball–cloth contact did not settle smoothly\n";
        return 1;
    }
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
              << " sphere_final_x=" << sphere_final.position.x
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

    World escape_world;
    if (!require(World::create({.rigid_body_capacity = 2U,
                                .triangle_mesh_capacity = 2U,
                                .cloth_capacity = 1U}, escape_world),
                 "create cloth release world")) return 1;
    SceneInstance escape_instance{};
    if (!require(instantiate_scene(scene, escape_world, escape_instance),
                 "instantiate cloth release world")) return 1;
    for (int frame = 0; frame < 600; ++frame)
        if (!require(escape_world.step(step), "settle cloth release world"))
            return 1;
    RigidBodyState escape_before{};
    if (!require(escape_world.read_rigid_body_state(
            escape_instance.rigid_bodies[active], escape_before),
            "read cloth release body")) return 1;
    if (!require(no_cloth_world.set_rigid_body_state(
            no_cloth_instance.rigid_bodies[active], escape_before),
            "set free release body")) return 1;
    // The settled sphere presses against the sheet's lower fold in -Y;
    // push it away from that local contact with gravity disabled.
    const Vec3 outward_impulse{0.0F,
        1.5F * scene.rigid_bodies[active].options.mass, 0.0F};
    if (!require(escape_world.apply_impulse(
            escape_instance.rigid_bodies[active], outward_impulse,
            escape_before.position), "impulse cloth release body") ||
        !require(no_cloth_world.apply_impulse(
            no_cloth_instance.rigid_bodies[active], outward_impulse,
            escape_before.position), "impulse free release body")) return 1;
    const StepOptions release_step{.timestep = 1.0F / 60.0F,
                                   .substeps = 4U,
                                   .gravity = {}};
    RigidBodyState escape_after_30{}, free_after_30{};
    for (int frame = 0; frame < 120; ++frame) {
        if (!require(escape_world.step(release_step),
                     "step cloth release body") ||
            !require(no_cloth_world.step(release_step),
                     "step free release body")) return 1;
        if (frame == 29) {
            if (!require(escape_world.read_rigid_body_state(
                    escape_instance.rigid_bodies[active], escape_after_30),
                    "read 30-frame cloth release") ||
                !require(no_cloth_world.read_rigid_body_state(
                    no_cloth_instance.rigid_bodies[active], free_after_30),
                    "read 30-frame free release")) return 1;
        }
    }
    RigidBodyState escape_after{}, free_after{};
    if (!require(escape_world.read_rigid_body_state(
            escape_instance.rigid_bodies[active], escape_after),
            "read cloth release result") ||
        !require(no_cloth_world.read_rigid_body_state(
            no_cloth_instance.rigid_bodies[active], free_after),
            "read free release result")) return 1;
    std::vector<Vec3> released_cloth;
    if (!read_cloth(escape_world, escape_instance.cloths.front(),
                    released_cloth)) return 1;
    float release_separation = 1.0e30F;
    for (const Vec3 vertex : released_cloth)
        release_separation = std::min(release_separation,
            distance(vertex, escape_after.position));
    std::cout << "Outward impulse 30-frame cloth_dy=" <<
        escape_after_30.position.y - escape_before.position.y
              << " free_dy=" <<
        free_after_30.position.y - escape_before.position.y
              << " 120-frame cloth_dy=" <<
        escape_after.position.y - escape_before.position.y
              << " free_dy=" <<
        free_after.position.y - escape_before.position.y
              << " nearest_cloth_distance=" << release_separation << '\n';
    const float free_30_displacement = free_after_30.position.y -
        escape_before.position.y;
    const float cloth_30_displacement = escape_after_30.position.y -
        escape_before.position.y;
    const float free_120_displacement = free_after.position.y -
        escape_before.position.y;
    const float cloth_120_displacement = escape_after.position.y -
        escape_before.position.y;
    if (free_30_displacement < 0.5F ||
        cloth_30_displacement < 0.9F * free_30_displacement ||
        cloth_120_displacement < 0.75F * free_120_displacement ||
        release_separation < 0.8F) {
        std::cerr << "Cloth trapped the separating rigid body\n";
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

    // Push the collected sphere sideways while a matching cloth-only world
    // receives the same gravity. Subtracting that control isolates the pull
    // transferred by body/cloth friction from gravity-driven sheet motion.
    const StepOptions lateral_step{
        .timestep = 1.0F / 60.0F,
        .substeps = 4U,
        .gravity = {5.6638F, -5.6638F, -5.6638F}};
    for (int frame = 0; frame < 30; ++frame) {
        if (!require(world.step(lateral_step), "step lateral cloth contact") ||
            !require(no_sphere_world.step(lateral_step),
                     "step lateral cloth-only control")) return 1;
    }
    RigidBodyState lateral_sphere{};
    if (!require(world.read_rigid_body_state(instance.rigid_bodies[active],
                                             lateral_sphere),
                 "read lateral sphere")) return 1;
    std::vector<Vec3> lateral_positions, lateral_control_positions;
    if (!read_cloth(world, instance.cloths.front(), lateral_positions) ||
        !read_cloth(no_sphere_world, no_sphere_instance.cloths.front(),
                    lateral_control_positions)) return 1;
    float local_x_pull = 0.0F;
    std::uint32_t local_count = 0U;
    for (std::size_t index = 0U; index < final.size(); ++index) {
        if (cloth.inverse_masses[index] == 0.0F ||
            distance(final[index], sphere_final.position) > 0.8F) continue;
        const float body_shift = lateral_positions[index].x - final[index].x;
        const float control_shift = lateral_control_positions[index].x -
            no_sphere_positions[index].x;
        local_x_pull += body_shift - control_shift;
        ++local_count;
    }
    if (local_count != 0U) local_x_pull /= static_cast<float>(local_count);
    std::cout << "Lateral sphere_dx=" <<
        lateral_sphere.position.x - sphere_final.position.x
              << " local_cloth_extra_dx=" << local_x_pull
              << " nearby_vertices=" << local_count << '\n';

    const float friction_dx = lateral_sphere.position.x -
        sphere_final.position.x;
    if (local_count < 100U || friction_dx < 0.15F ||
        local_x_pull < 0.015F) {
        std::cerr << "Ball did not slide while dragging the cloth\n";
        return 1;
    }
    const StepOptions reverse_step{
        .timestep = 1.0F / 60.0F,
        .substeps = 4U,
        .gravity = {-5.6638F, -5.6638F, -5.6638F}};
    for (int frame = 0; frame < 60; ++frame) {
        if (!require(world.step(reverse_step), "step reverse cloth contact") ||
            !require(no_sphere_world.step(reverse_step),
                     "step reverse cloth-only control")) return 1;
    }
    RigidBodyState reverse_sphere{};
    if (!require(world.read_rigid_body_state(instance.rigid_bodies[active],
                                             reverse_sphere),
                 "read reverse sphere")) return 1;
    std::vector<Vec3> reverse_positions, reverse_control_positions;
    if (!read_cloth(world, instance.cloths.front(), reverse_positions) ||
        !read_cloth(no_sphere_world, no_sphere_instance.cloths.front(),
                    reverse_control_positions)) return 1;
    float reverse_local_x_pull = 0.0F;
    std::uint32_t reverse_local_count = 0U;
    for (std::size_t index = 0U; index < final.size(); ++index) {
        if (cloth.inverse_masses[index] == 0.0F ||
            distance(lateral_positions[index], lateral_sphere.position) >
                0.8F) continue;
        const float body_shift = reverse_positions[index].x -
            lateral_positions[index].x;
        const float control_shift = reverse_control_positions[index].x -
            lateral_control_positions[index].x;
        reverse_local_x_pull += body_shift - control_shift;
        ++reverse_local_count;
    }
    if (reverse_local_count != 0U)
        reverse_local_x_pull /= static_cast<float>(reverse_local_count);
    std::cout << "Reverse sphere_dx=" <<
        reverse_sphere.position.x - lateral_sphere.position.x
              << " local_cloth_extra_dx=" << reverse_local_x_pull
              << " nearby_vertices=" << reverse_local_count << '\n';
    if (reverse_local_count < 100U ||
        reverse_sphere.position.x - lateral_sphere.position.x > -0.2F ||
        reverse_local_x_pull > -0.015F) {
        std::cerr << "Reverse motion did not drag the cloth back\n";
        return 1;
    }

    SceneDefinition isolated_cloth = scene;
    isolated_cloth.rigid_bodies.clear();
    World isolated_world;
    if (!require(World::create({.rigid_body_capacity = 1U,
                                .triangle_mesh_capacity = 1U,
                                .cloth_capacity = 1U}, isolated_world),
                 "create isolated cloth world")) return 1;
    SceneInstance isolated_instance{};
    if (!require(instantiate_scene(isolated_cloth, isolated_world,
                                   isolated_instance),
                 "instantiate isolated cloth")) return 1;
    for (int frame = 0; frame < 120; ++frame)
        if (!require(isolated_world.step(step), "step isolated cloth"))
            return 1;
    std::vector<Vec3> isolated_positions;
    if (!read_cloth(isolated_world, isolated_instance.cloths.front(),
                    isolated_positions)) return 1;
    float isolated_pinned_motion = 0.0F, isolated_free_motion = 0.0F;
    for (std::size_t index = 0U; index < initial.size(); ++index) {
        const float motion = distance(initial[index], isolated_positions[index]);
        if (!std::isfinite(motion)) {
            std::cerr << "Isolated cloth vertex became non-finite\n";
            return 1;
        }
        if (cloth.inverse_masses[index] == 0.0F)
            isolated_pinned_motion = std::max(isolated_pinned_motion, motion);
        else isolated_free_motion = std::max(isolated_free_motion, motion);
    }
    if (isolated_pinned_motion > 1.0e-5F ||
        isolated_free_motion < 0.01F) {
        std::cerr << "Isolated cloth did not advance with its pins fixed\n";
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
