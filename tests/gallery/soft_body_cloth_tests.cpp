// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include "../../examples/support/vector_math.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::gallery;
namespace math = parallel_mater::gallery::math;

void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
void check(Status status, const char *message) {
    if (!status) throw std::runtime_error(std::string(message) + ": " + status.message);
}
template<class T> std::vector<T> read(DeviceSpan<const T> span) {
    std::vector<T> result(span.size);
    check(cudaMemcpy(result.data(), span.data, span.size * sizeof(T),
                     cudaMemcpyDeviceToHost) == cudaSuccess, "device read failed");
    return result;
}
float length(Vec3 v) { return std::sqrt(math::dot(v, v)); }
Vec3 mean(const std::vector<Vec3> &points) {
    Vec3 sum{};
    for (Vec3 p : points) sum = math::add(sum, p);
    return math::multiply(sum, 1.0F / points.size());
}

// Winding test against the actual deformed closed skin, not a bounding sphere.
bool inside_skin(Vec3 point, const std::vector<Vec3> &skin,
                 const std::vector<std::uint32_t> &indices, Vec3 lo, Vec3 hi) {
    if (point.x < lo.x || point.x > hi.x || point.y < lo.y || point.y > hi.y ||
        point.z < lo.z || point.z > hi.z) return false;
    double angle = 0;
    for (std::size_t i = 0; i < indices.size(); i += 3) {
        const Vec3 a = math::subtract(skin[indices[i]], point);
        const Vec3 b = math::subtract(skin[indices[i + 1]], point);
        const Vec3 c = math::subtract(skin[indices[i + 2]], point);
        const double x = length(a), y = length(b), z = length(c);
        angle += 2 * std::atan2(static_cast<double>(math::dot(a, math::cross(b,c))),
            x*y*z + math::dot(a,b)*z + math::dot(b,c)*x + math::dot(c,a)*y);
    }
    return std::abs(angle) > 6.283185307179586;
}

// During the initial drop the bridge is a height field. Check its actual
// deformed triangles, not its rest plane or the arena's much lower floor.
float bridge_clearance(const std::vector<Vec3> &points,
                       const std::vector<Vec3> &cloth,
                       const std::vector<std::uint32_t> &indices) {
    float clearance = 100.0F;
    for (Vec3 point : points) {
        for (std::size_t triangle = 0; triangle < indices.size(); triangle += 3) {
            const Vec3 a = cloth[indices[triangle]];
            const Vec3 b = math::subtract(cloth[indices[triangle + 1]], a);
            const Vec3 c = math::subtract(cloth[indices[triangle + 2]], a);
            const Vec3 p = math::subtract(point, a);
            const float determinant = b.x * c.z - b.z * c.x;
            if (std::fabs(determinant) < 1.0e-10F) continue;
            const float u = (p.x * c.z - p.z * c.x) / determinant;
            const float v = (b.x * p.z - b.z * p.x) / determinant;
            if (u < 0 || v < 0 || u + v > 1.0F) continue;
            clearance = std::min(clearance, p.y - u * b.y - v * c.y);
        }
    }
    return clearance;
}

void translate_soft(SceneDefinition &scene, Vec3 target) {
    auto &body = scene.soft_bodies[0];
    const Vec3 shift = math::subtract(target, mean(body.nodes));
    for (Vec3 &point : body.nodes) point = math::add(point, shift);
    for (auto &vertex : scene.meshes[body.mesh_index].vertices)
        vertex.position = math::add(vertex.position, shift);
}

void verify_controls(const SceneDefinition &authored, std::size_t bridge,
                     std::size_t curtain) {
    // Remove rigid geometry to exercise substep scheduling for two deformables
    // alone. Disabling the public coupling must let the sphere fall freely.
    std::vector<Vec3> replay_positions;
    for (bool enabled : {false, true, true}) {
        SceneDefinition scene = authored;
        scene.rigid_bodies.clear();
        scene.cloths = {authored.cloths[bridge]};
        World world;
        SceneInstance instance;
        check(create_scene_world(scene, world, instance), "create support control");
        check(world.update_soft_body_cloth_coupling(instance.soft_body_cloth_couplings[0],
            {.soft_body = instance.soft_bodies[0], .cloth = instance.cloths[0],
             .enabled = enabled}), "set coupling control");
        const auto started = std::chrono::steady_clock::now();
        for (unsigned frame = 0; frame < 180; ++frame)
            check(world.step({}), "step support control");
        check(cudaDeviceSynchronize() == cudaSuccess, "finish support control");
        const float milliseconds = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - started).count() / 180.0F;
        SoftBodyDeviceView view;
        check(world.soft_body_view(instance.soft_bodies[0], view), "support control view");
        const auto positions = read(view.positions);
        const float height = mean(positions).y;
        std::cout << "coupling enabled=" << enabled << " center_y=" << height
                  << " unprofiled_step_ms=" << milliseconds << std::endl;
        check(enabled ? height > -0.2F : height < -1.0F,
              "coupling enable switch did not change unsupported fall");
        if (enabled) {
            if (replay_positions.empty()) replay_positions = positions;
            else for (std::size_t node = 0; node < positions.size(); ++node)
                check(length(math::subtract(positions[node], replay_positions[node])) == 0.0F,
                      "soft-cloth replay is nondeterministic");
        }
    }
    // The same curtain with fracture disabled must resist the sphere from
    // either winding side. Place the body close enough that it cannot simply
    // travel around an edge before the contact has been measured.
    const auto &curtain_mesh = authored.meshes[authored.cloths[curtain].mesh_index];
    const float plane_z = curtain_mesh.vertices.front().position.z;
    for (float side : {1.0F, -1.0F}) {
        SceneDefinition scene = authored;
        scene.cloths = {authored.cloths[curtain]};
        scene.cloths[0].break_strain = 0.0F;
        scene.cloths[0].impact_break_impulse = 0.0F;
        translate_soft(scene, {0, 0.65F, plane_z + side * 0.70F});
        World world;
        SceneInstance instance;
        check(create_scene_world(scene, world, instance), "create intact curtain control");
        unsigned contact_frames = 0;
        for (unsigned frame = 0; frame < 180; ++frame) {
            check(world.step({.gravity = {0, -6.936718F, -side * 6.936718F}}),
                  "step intact curtain");
            SoftBodyDeviceView soft;
            check(world.soft_body_view(instance.soft_bodies[0], soft), "curtain soft view");
            const auto forces = read(soft.cloth_contact_forces);
            contact_frames += std::any_of(forces.begin(), forces.end(),
                [](Vec3 force) { return length(force) > 1.0e-5F; });
        }
        SoftBodyDeviceView soft;
        ClothDeviceView cloth;
        check(world.soft_body_view(instance.soft_bodies[0], soft), "intact soft view");
        check(world.cloth_view(instance.cloths[0], cloth), "intact cloth view");
        const float separation = side * (mean(read(soft.positions)).z - plane_z);
        const auto active = read(cloth.active_bonds);
        std::cout << "intact curtain side=" << side << " separation=" << separation
                  << " contact_frames=" << contact_frames << std::endl;
        check(contact_frames > 20U && separation > -0.10F &&
              std::count(active.begin(), active.end(), 0U) == 0U,
              "intact two-sided curtain contact failed");
    }
}

int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        SceneDefinition scene;
        std::string error;
        check(load_glb_scene(PARALLEL_MATER_SOFT_BODY_CLOTH_SCENE_PATH, scene, error),
              error.c_str());
        check(scene.soft_bodies.size() == 1U && scene.cloths.size() == 2U,
              "scene needs one soft body and two cloths");
        const std::size_t bridge = scene.cloths[0].break_strain == 0.0F ? 0U : 1U;
        const std::size_t curtain = 1U - bridge;
        const bool smoke = argc > 1 && std::string(argv[1]) == "--smoke";
        check(scene.cloths[bridge].break_strain == 0.0F &&
              scene.cloths[curtain].break_strain > 0.0F,
              "cloth fracture must be authored independently");
        World world;
        SceneInstance instance;
        check(create_scene_world(scene, world, instance, {.frame_capacity = 2U}),
              "create soft-cloth scene");
        check(instance.soft_body_cloth_couplings.size() == 2U, "coupling count");
        SoftBodyClothCouplingId duplicate{};
        check(world.add_soft_body_cloth_coupling(
            {.soft_body = instance.soft_bodies[0], .cloth = instance.cloths[0]},
            duplicate).code == StatusCode::invalid_argument, "duplicate coupling accepted");
        check(world.remove_cloth(instance.cloths[0]).code == StatusCode::invalid_argument,
              "referenced cloth removed");
        check(world.remove_soft_body(instance.soft_bodies[0]).code == StatusCode::invalid_argument,
              "referenced soft body removed");
        check(world.update_soft_body_cloth_coupling(instance.soft_body_cloth_couplings[0],
            {.soft_body = instance.soft_bodies[0], .cloth = instance.cloths[0],
             .solver_iterations = 0U}).code == StatusCode::invalid_argument,
            "zero coupling iterations accepted");
        check(world.update_soft_body_cloth_coupling(instance.soft_body_cloth_couplings[0],
            {.soft_body = instance.soft_bodies[0], .cloth = instance.cloths[1]}).code ==
              StatusCode::invalid_argument, "coupling endpoint change accepted");
        // Mixed requests use the highest shared pass count. A lower request
        // must not stop constraining a body that its other pair still moves.
        check(world.update_soft_body_cloth_coupling(instance.soft_body_cloth_couplings[bridge],
            {.soft_body = instance.soft_bodies[0], .cloth = instance.cloths[bridge],
             .friction = scene.cloths[bridge].contact_friction,
             .solver_iterations = 2U}), "set mixed coupling iterations");
        float bridge_sag = 0.0F, pin_error = 0.0F, maximum_speed = 0.0F;
        float force_balance = 0.0F, settled_height = 0.0F, minimum_settled_y = 100.0F;
        float maximum_strain[2]{};
        float minimum_bridge_clearance = 100.0F;
        float minimum_node_clearance = 100.0F;
        std::uint32_t clearance_frame = 0U;
        std::uint32_t contact_frames = 0U, curtain_contact_frames = 0U;
        std::uint32_t bridge_breaks = 0U, curtain_breaks = 0U;
        std::uint32_t inside_triangle_frames = 0U;
        float surface_mismatch = 0.0F;
        Vec3 center{};
        for (std::uint32_t frame = 0U; frame < (smoke ? 240U : 1200U); ++frame) {
            const StepOptions step{.timestep = 1.0F / 60.0F, .substeps = 4U,
                .gravity = frame < 240U || frame >= 900U ? Vec3{0, -9.81F, 0}
                                      : Vec3{0, -6.936718F, -6.936718F},
                .collect_kernel_timings = frame == 239U};
            check(world.step(step), "step soft-cloth scene");
            SoftBodyDeviceView soft;
            check(world.soft_body_view(instance.soft_bodies[0], soft), "soft view");
            const auto points = read(soft.positions);
            const auto velocities = read(soft.velocities);
            const auto forces = read(soft.cloth_contact_forces);
            const auto skin = read(soft.surface_positions);
            const auto skin_indices = read(soft.surface_triangle_indices);
            Vec3 lo=skin[0], hi=lo;
            for (Vec3 p:skin) {
                lo={std::min(lo.x,p.x), std::min(lo.y,p.y), std::min(lo.z,p.z)};
                hi={std::max(hi.x,p.x), std::max(hi.y,p.y), std::max(hi.z,p.z)};
            }
            if (frame == 239U) {
                PhysicsDebugFrameView capture;
                check(world.physics_debug_frame(capture), "coupling debug capture");
                check(capture.soft_body_nodes.size == soft.node_count,
                      "capture omitted soft nodes");
                for (std::size_t node = 0U; node < forces.size(); ++node)
                    check(length(math::subtract(forces[node],
                        capture.soft_body_nodes.data[node].cloth_contact_force)) < 1.0e-5F,
                        "capture omitted soft-cloth forces");
            }
            center = mean(points);
            Vec3 total_force{};
            float force_sum = 0.0F;
            for (std::size_t node = 0U; node < points.size(); ++node) {
                check(std::isfinite(points[node].x) && std::isfinite(points[node].y) &&
                      std::isfinite(points[node].z) && std::isfinite(length(velocities[node])),
                      "nonfinite soft body");
                maximum_speed = std::max(maximum_speed, length(velocities[node]));
                total_force = math::add(total_force, forces[node]);
                force_sum += length(forces[node]);
                if (frame >= 180U && frame < 240U)
                    minimum_settled_y = std::min(minimum_settled_y, points[node].y);
            }
            contact_frames += force_sum > 1.0e-5F;
            for (std::size_t sheet = 0U; sheet < 2U; ++sheet) {
                ClothDeviceView cloth;
                check(world.cloth_view(instance.cloths[sheet], cloth), "cloth view");
                const auto cloth_points = read(cloth.positions);
                const auto cloth_forces = read(cloth.soft_body_contact_forces);
                const auto active = read(cloth.active_bonds);
                const auto bonds = read(cloth.bonds);
                const auto sources = read(cloth.vertex_source_indices);
                const auto &definition = scene.cloths[sheet];
                const auto &rest = scene.meshes[definition.mesh_index].vertices;
                check(cloth.triangle_indices.size ==
                      scene.meshes[definition.mesh_index].indices.size(),
                      "fracture removed cloth triangles");
                if (sheet == bridge && frame < 240U) {
                    minimum_node_clearance = std::min(minimum_node_clearance,
                        bridge_clearance(points, cloth_points,
                            scene.meshes[definition.mesh_index].indices));
                    auto samples = points;
                    const auto skin = read(soft.surface_positions);
                    samples.insert(samples.end(), skin.begin(), skin.end());
                    const auto &triangles = scene.meshes[scene.soft_bodies[0].mesh_index].indices;
                    for (std::size_t triangle = 0; triangle < triangles.size(); triangle += 3) {
                        const Vec3 a = skin[triangles[triangle]];
                        const Vec3 b = skin[triangles[triangle + 1]];
                        const Vec3 c = skin[triangles[triangle + 2]];
                        samples.push_back(math::multiply(math::add(a, math::add(b, c)), 1.0F / 3.0F));
                        samples.push_back(math::multiply(math::add(a, b), 0.5F));
                        samples.push_back(math::multiply(math::add(b, c), 0.5F));
                        samples.push_back(math::multiply(math::add(c, a), 0.5F));
                    }
                    const float clearance = bridge_clearance(samples, cloth_points,
                        scene.meshes[definition.mesh_index].indices);
                    if (clearance < minimum_bridge_clearance) {
                        minimum_bridge_clearance = clearance;
                        clearance_frame = frame;
                    }
                }
                float sheet_force = 0.0F;
                for (std::size_t node = 0U; node < cloth_points.size(); ++node) {
                    const auto source = sources.empty() ? node : sources[node];
                    check(std::isfinite(length(cloth_points[node])), "nonfinite cloth");
                    if (definition.inverse_masses[source] == 0.0F)
                        pin_error = std::max(pin_error,
                            length(math::subtract(cloth_points[node], rest[source].position)));
                    if (sheet == bridge && frame < 240U)
                        bridge_sag = std::max(bridge_sag,
                            rest[node].position.y - cloth_points[node].y);
                    total_force = math::add(total_force, cloth_forces[node]);
                    sheet_force += length(cloth_forces[node]);
                }
                const auto broken = static_cast<std::uint32_t>(
                    std::count(active.begin(), active.end(), 0U));
                for (std::size_t edge = 0U; edge < bonds.size(); ++edge) {
                    if (active[edge] == 0U) continue;
                    const auto bond = bonds[edge];
                    maximum_strain[sheet] = std::max(maximum_strain[sheet],
                        length(math::subtract(cloth_points[bond.first],
                                              cloth_points[bond.second])) / bond.rest_length - 1.0F);
                }
                if (sheet == bridge) bridge_breaks = broken;
                else {
                    curtain_breaks = broken;
                    curtain_contact_frames += sheet_force > 1.0e-5F;
                    check(curtain_breaks == 0U || curtain_contact_frames > 0U,
                          "curtain broke before contact");
                    const auto surface = read(cloth.surface_positions);
                    const auto indices = read(cloth.triangle_indices);
                    const auto masses = read(cloth.inverse_masses);
                    double original_mass = 0, split_mass = 0;
                    for (float inverse:definition.inverse_masses)
                        if (inverse>0) original_mass += 1.0/inverse;
                    for (float inverse:masses) if (inverse>0) split_mass += 1.0/inverse;
                    check(std::abs(split_mass-original_mass) < original_mass*1.0e-5,
                          "tearing changed cloth mass");
                    for (std::size_t i=0; i<indices.size(); i+=3) {
                        bool inside = true;
                        Vec3 centroid{};
                        for (std::size_t c=0; c<3; ++c) {
                            check(sources[indices[i+c]] == scene.meshes[definition.mesh_index].indices[i+c],
                                  "tear changed authored triangle/source mapping");
                            surface_mismatch = std::max(surface_mismatch,
                                length(math::subtract(surface[i+c],cloth_points[indices[i+c]])));
                            inside &= inside_skin(surface[i+c],skin,skin_indices,lo,hi);
                            centroid = math::add(centroid,surface[i+c]);
                        }
                        inside_triangle_frames += inside && inside_skin(
                            math::multiply(centroid,1.0F/3),skin,skin_indices,lo,hi);
                    }
                }
            }
            force_balance = std::max(force_balance,
                length(total_force) / std::max(1.0F, force_sum));
            if (frame == 239U) {
                settled_height = center.y;
                WorldStepTimings timings;
                check(world.collect_step_timings(timings), "timings");
                check(timings.soft_body_cloth_contacts.launch_count > 0U,
                      "missing coupling timing");
                std::cout << "settled center=" << center.x << ',' << center.y << ',' << center.z
                          << " min_y=" << minimum_settled_y << " sag=" << bridge_sag
                          << " gpu_ms=" << timings.total_gpu_milliseconds
                          << " coupling_gpu_ms=" << timings.soft_body_cloth_contacts.total_milliseconds
                          << std::endl;
            }
            if ((frame + 1U) % 180U == 0U)
                std::cout << "frame=" << frame + 1U << " center=" << center.x << ','
                          << center.y << ',' << center.z << " broken=" << bridge_breaks
                          << ',' << curtain_breaks << std::endl;
        }
        std::cout << "soft-cloth bridge_sag=" << bridge_sag << " pin_error=" << pin_error
                  << " maximum_speed=" << maximum_speed << " force_balance=" << force_balance
                  << " contact_frames=" << contact_frames
                  << " curtain_contact_frames=" << curtain_contact_frames
                  << " bridge_breaks=" << bridge_breaks << " curtain_breaks=" << curtain_breaks
                  << " maximum_strain=" << maximum_strain[bridge] << ',' << maximum_strain[curtain]
                  << " bridge_clearance=" << minimum_bridge_clearance
                  << " node_clearance=" << minimum_node_clearance
                  << " clearance_frame=" << clearance_frame
                  << " final_z=" << center.z << std::endl;
        std::cout << "inside_triangle_frames=" << inside_triangle_frames
                  << " physical_surface_mismatch=" << surface_mismatch << std::endl;
        check(inside_triangle_frames == 0U, "torn triangle trapped inside soft skin");
        check(surface_mismatch < 1.0e-7F, "cloth surface detached from physical vertices");
        check(settled_height > 0.0F && minimum_settled_y > -0.45F && bridge_sag > 0.005F,
              "cloth failed to support the soft sphere over the pit");
        check(minimum_bridge_clearance > -0.005F,
              "soft nodes or sampled skin crossed the loaded bridge");
        check(pin_error < 1.0e-6F && bridge_breaks == 0U, "bridge pins or bonds failed");
        check(contact_frames > 30U && force_balance < 1.0e-3F, "unbalanced coupling forces");
        check(smoke || (curtain_contact_frames > 0U && curtain_breaks > 0U && center.z < -3.4F),
              "sphere failed to tear and pass through curtain");
        check(maximum_speed <= scene.soft_bodies[0].maximum_speed + 1.0e-4F,
              "soft body exceeded speed cap");
        for (auto coupling : instance.soft_body_cloth_couplings) {
            check(world.remove_soft_body_cloth_coupling(coupling), "remove coupling");
            check(world.remove_soft_body_cloth_coupling(coupling).code == StatusCode::invalid_handle,
                  "stale coupling accepted");
        }
        SoftBodyClothCouplingId replacement;
        check(world.add_soft_body_cloth_coupling(
            {.soft_body = instance.soft_bodies[0], .cloth = instance.cloths[0]},
            replacement), "reuse coupling slot");
        check(!(replacement == instance.soft_body_cloth_couplings[0]),
              "coupling generation was reused");
        check(world.remove_soft_body_cloth_coupling(replacement), "remove replacement");
        check(world.remove_soft_body(instance.soft_bodies[0]), "remove uncoupled soft body");
        if (!smoke) verify_controls(scene, bridge, curtain);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
