// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <scene_builder.hpp>
#include "backend_comparison.hpp"
#include "rigid_mesh_checks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

using namespace parallel_mater;
using namespace parallel_mater::gallery;

[[nodiscard]] bool finite(const PmRigidState &state) noexcept {
    for (float value : {state.position[0], state.position[1], state.position[2],
                        state.orientation[0], state.orientation[1],
                        state.orientation[2], state.orientation[3],
                        state.linearVelocity[0], state.linearVelocity[1],
                        state.linearVelocity[2], state.angularVelocity[0],
                        state.angularVelocity[1], state.angularVelocity[2]})
        if (!std::isfinite(value)) return false;
    return true;
}

[[nodiscard]] bool finite(const PmFloat3 &value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

template <std::size_t Size>
[[nodiscard]] bool finite(const float (&value)[Size]) noexcept {
    for (const float component : value)
        if (!std::isfinite(component)) return false;
    return true;
}

[[nodiscard]] bool validate_frame(const SceneDefinition &scene,
                                  const CommandSceneFrame &frame,
                                  std::string &error) {
    if (frame.rigid_states.size() != scene.rigid_bodies.size()) {
        error = "rigid output count does not match the scene";
        return false;
    }
    for (const PmRigidState &state : frame.rigid_states) {
        if (!finite(state)) {
            error = "rigid output contains a non-finite value";
            return false;
        } else if (std::fabs(state.position[0]) > 1'000.0F ||
                 std::fabs(state.position[1]) > 1'000.0F ||
                 std::fabs(state.position[2]) > 1'000.0F ||
                 std::fabs(state.linearVelocity[0]) > 1'000.0F ||
                 std::fabs(state.linearVelocity[1]) > 1'000.0F ||
                 std::fabs(state.linearVelocity[2]) > 1'000.0F ||
                 std::fabs(state.angularVelocity[0]) > 1'000.0F ||
                 std::fabs(state.angularVelocity[1]) > 1'000.0F ||
                 std::fabs(state.angularVelocity[2]) > 1'000.0F) {
            error = "rigid output escaped the stability bounds";
            return false;
        }
        float norm = 0.0F;
        for (float component : state.orientation) norm += component * component;
        if (std::abs(norm - 1.0F) > 0.001F) {
            error = "rigid orientation lost unit length";
            return false;
        }
    }
    for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
        const auto &body = scene.rigid_bodies[index];
        if (body.options.motion != MotionType::static_body) continue;
        const auto &position = body.options.initial_state.position;
        const auto &actual = frame.rigid_states[index];
        if (std::abs(actual.position[0] - position.x) > 1.0e-6F ||
            std::abs(actual.position[1] - position.y) > 1.0e-6F ||
            std::abs(actual.position[2] - position.z) > 1.0e-6F) {
            error = "static rigid body moved: " + body.name;
            return false;
        }
    }
    for (const PmFluidParticleOutput &particle : frame.fluid_particles)
        if (!finite(particle.position) || !finite(particle.velocity)) {
            error = "fluid output contains a non-finite value";
            return false;
        }
    for (const auto &cloth : frame.cloth_vertices)
        for (const PmClothVertexState &vertex : cloth)
            if (!finite(vertex.position) || !finite(vertex.velocity)) {
                error = "cloth output contains a non-finite value";
                return false;
            }
    if (frame.cloth_vertices.size() != scene.cloths.size()) {
        error = "cloth output count does not match the scene";
        return false;
    }
    for (std::size_t cloth = 0U; cloth < scene.cloths.size(); ++cloth) {
        const auto &definition = scene.cloths[cloth];
        const auto &vertices = scene.meshes.at(definition.mesh_index).vertices;
        if (frame.cloth_vertices[cloth].size() < vertices.size()) {
            error = "cloth lost original vertices";
            return false;
        }
        for (std::size_t vertex = 0U; vertex < definition.inverse_masses.size(); ++vertex) {
            if (definition.inverse_masses[vertex] != 0.0F) continue;
            const auto &expected = vertices.at(vertex).position;
            const auto &actual = frame.cloth_vertices[cloth][vertex].position;
            if (std::abs(actual.x - expected.x) > 1.0e-5F ||
                std::abs(actual.y - expected.y) > 1.0e-5F ||
                std::abs(actual.z - expected.z) > 1.0e-5F) {
                error = "pinned cloth vertex moved";
                return false;
            }
        }
    }
    for (const auto &contact : frame.contacts) {
        if (!finite(contact.position) || !finite(contact.normal) ||
            !finite(contact.frictionImpulse) || !std::isfinite(contact.penetration) ||
            !std::isfinite(contact.normalImpulse) || contact.normalImpulse < -1.0e-6F) {
            error = "contact record has an invalid position, normal or impulse";
            return false;
        }
    }
    for (const auto &rope : frame.rope_nodes)
        for (const PmRopeNodeState &node : rope)
            if (!finite(node.position) || !finite(node.velocity)) {
                error = "rope output contains a non-finite value";
                return false;
            }
    for (const auto &body : frame.soft_body_nodes)
        for (const PmSoftBodyNodeState &node : body)
            if (!finite(node.position) || !finite(node.velocity)) {
                error = "soft-body output contains a non-finite value";
                return false;
            }
    return true;
}

bool validate_torque_free_rotation(const TriangleMesh &mesh,
                                  std::optional<PmNativeBackend> backend,
                                  std::string &error) {
    for (const Vec3 spin : {Vec3{}, Vec3{0.3F, -0.7F, 1.1F}}) {
        SceneDefinition scene;
        scene.meshes.push_back(mesh);
        RigidBodyDefinition body;
        body.name = "TorqueFree";
        body.mesh_indices = {0U};
        body.options.inertia_diagonal = {1.0F, 1.0F, 1.0F};
        body.options.linear_damping = 0.0F;
        body.options.angular_damping = 0.0F;
        body.options.initial_state.orientation = {0.3F, 0.4F, 0.5F, std::sqrt(0.5F)};
        body.options.initial_state.angular_velocity = spin;
        scene.rigid_bodies.push_back(body);
        CommandScene simulation;
        CommandSceneInstance instance;
        CommandSceneFrame frame;
        if (!CommandScene::create(scene, 2U, simulation, instance, error, backend)) return false;
        float maximum_drift{};
        for (unsigned step = 0U; step < 60U; ++step) {
            if (!simulation.step(1.0F / 60.0F, 1U, {}, frame, error)) return false;
            const auto &actual = frame.rigid_states.at(0U);
            const float expected_scale = std::pow(0.95F,
                static_cast<float>(step + 1U));
            maximum_drift = std::max({maximum_drift,
                std::abs(actual.angularVelocity[0] - spin.x * expected_scale),
                std::abs(actual.angularVelocity[1] - spin.y * expected_scale),
                std::abs(actual.angularVelocity[2] - spin.z * expected_scale)});
        }
        std::cout << "Torque-free damping error=" << maximum_drift << " rad/s\n";
        if (maximum_drift > 1.0e-5F) {
            error = "AVBD torque-free damping changed the angular axis or rate";
            return false;
        }
    }
    return true;
}

bool validate_swept_face_impact_case(std::optional<PmNativeBackend> backend,
                                    float start, unsigned substeps, std::string &error) {
    SceneDefinition scene;
    TriangleMesh box;
    box.vertices = {{{-0.25F, -0.25F, -0.25F}}, {{0.25F, -0.25F, -0.25F}},
                    {{0.25F, 0.25F, -0.25F}}, {{-0.25F, 0.25F, -0.25F}},
                    {{-0.25F, -0.25F, 0.25F}}, {{0.25F, -0.25F, 0.25F}},
                    {{0.25F, 0.25F, 0.25F}}, {{-0.25F, 0.25F, 0.25F}}};
    box.indices = {0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,
                   1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0};
    scene.meshes.push_back(box);
    RigidBodyDefinition body;
    body.name = "CenteredTarget";
    body.mesh_indices = {0U};
    body.options.linear_damping = 0.0F;
    body.options.angular_damping = 0.0F;
    body.options.inertia_diagonal = {1.0F / 24.0F, 1.0F / 24.0F, 1.0F / 24.0F};
    scene.rigid_bodies.push_back(body);
    body.name = "TranslatingStriker";
    body.options.motion = MotionType::kinematic;
    body.options.initial_state.position = {start, 0.0F, 0.0F};
    scene.rigid_bodies.push_back(body);
    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    RigidBodyState target;
    target.position = {1.0F, 0.0F, 0.0F};
    if (!CommandScene::create(scene, 2U, simulation, instance, error, backend) ||
        !simulation.set_kinematic_target(instance.rigid_bodies.at(1U), target, error) ||
        !simulation.step(1.0F / 60.0F, substeps, {}, frame, error, true)) return false;
    const auto &actual = frame.rigid_states.at(0U);
    float transverse_speed = std::max(std::abs(actual.linearVelocity[1]),
                                     std::abs(actual.linearVelocity[2]));
    float angular_speed = std::max({std::abs(actual.angularVelocity[0]),
        std::abs(actual.angularVelocity[1]), std::abs(actual.angularVelocity[2])});
    std::cout << "Swept face impact: start=" << start << " substeps=" << substeps
              << " forward=" << actual.linearVelocity[0]
              << " transverse=" << transverse_speed << " angular=" << angular_speed << '\n';
    if (actual.linearVelocity[0] <= 1.0F || transverse_speed > 1.0e-3F ||
        angular_speed > 0.01F) {
        error = "centered convex sweep missed the target or invented lateral motion";
        return false;
    }
    return true;
}

bool validate_swept_face_impact(std::optional<PmNativeBackend> backend,
                               std::string &error) {
    return validate_swept_face_impact_case(backend, -1.0F, 1U, error) &&
        validate_swept_face_impact_case(backend, -0.5F, 1U, error);
}

[[nodiscard]] TriangleMesh planar_sweep_box(Vec3 half) {
    TriangleMesh box;
    box.vertices = {{{-half.x, -half.y, -half.z}},
                    {{ half.x, -half.y, -half.z}},
                    {{ half.x,  half.y, -half.z}},
                    {{-half.x,  half.y, -half.z}},
                    {{-half.x, -half.y,  half.z}},
                    {{ half.x, -half.y,  half.z}},
                    {{ half.x,  half.y,  half.z}},
                    {{-half.x,  half.y,  half.z}}};
    box.indices = {0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,
                   1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0};
    return box;
}

bool run_planar_sweep(std::optional<PmNativeBackend> backend, float target_tilt,
                      PmRigidState &state, std::vector<PmContactEvent> &contacts,
                      std::string &error) {
    SceneDefinition scene;
    scene.meshes.push_back(planar_sweep_box({0.25F, 0.25F, 0.25F}));
    scene.meshes.push_back(planar_sweep_box({0.2F, 0.2F, 0.2F}));
    RigidBodyDefinition body;
    body.name = "PlanarSweepDynamic";
    body.mesh_indices = {0U};
    body.options.initial_state.position = {0.0F, 2.0F, 0.0F};
    body.options.mass = 2.0F;
    body.options.inertia_diagonal = {1.0F / 12.0F, 1.0F / 12.0F,
                                     1.0F / 12.0F};
    body.options.linear_damping = 0.0F;
    body.options.angular_damping = 0.0F;
    scene.rigid_bodies.push_back(body);
    body = {};
    body.name = "PlanarSweepKinematic";
    body.mesh_indices = {1U};
    body.options.motion = MotionType::kinematic;
    body.options.initial_state.position = {-1.0F, 1.0F, 0.0F};
    scene.rigid_bodies.push_back(body);

    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    if (!CommandScene::create(scene, 2U, simulation, instance, error, backend))
        return false;
    PmRigidInput impulse{};
    impulse.bodyIndex = 0U;
    impulse.linearImpulse[0] = 1.5F;
    if (!simulation.apply_rigid_inputs({impulse}, error)) return false;
    for (unsigned step = 0U; step < 20U; ++step)
        if (!simulation.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F},
                             frame, error, true)) return false;
    RigidBodyState target;
    target.position = {1.0F, 1.5F, 0.0F};
    target.orientation = {0.0F, std::sin(0.5F * target_tilt), 0.0F,
                          std::cos(0.5F * target_tilt)};
    if (!simulation.set_kinematic_target(1U, target, error) ||
        !simulation.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F},
                         frame, error, true)) return false;
    state = frame.rigid_states.at(0U);
    contacts = std::move(frame.contacts);
    return true;
}

bool validate_planar_sweep_symmetry(std::optional<PmNativeBackend> backend,
                                    std::string &error) {
    PmRigidState planar{};
    std::vector<PmContactEvent> planar_contacts;
    if (!run_planar_sweep(backend, 0.0F, planar, planar_contacts, error))
        return false;
    const float leakage = std::max({std::abs(planar.position[2]),
        std::abs(planar.orientation[0]), std::abs(planar.orientation[1]),
        std::abs(planar.linearVelocity[2]),
        std::abs(planar.angularVelocity[0]),
        std::abs(planar.angularVelocity[1])});
    if (leakage > 1.0e-5F || planar_contacts.size() != 4U) {
        error = "planar sweep retained a tessellation contact or leaked out of plane";
        return false;
    }
    std::array<float, 4> contact_z{};
    for (std::size_t index = 0U; index < planar_contacts.size(); ++index) {
        contact_z[index] = planar_contacts[index].position.z;
        if (std::abs(planar_contacts[index].normal.z) > 1.0e-6F) {
            error = "planar sweep generated an out-of-plane contact normal";
            return false;
        }
    }
    std::sort(contact_z.begin(), contact_z.end());
    if (std::abs(contact_z[0] + contact_z[3]) > 1.0e-5F ||
        std::abs(contact_z[1] + contact_z[2]) > 1.0e-5F) {
        error = "planar sweep contact patch lost mirror pairing";
        return false;
    }

    PmRigidState perturbed{};
    std::vector<PmContactEvent> perturbed_contacts;
    if (!run_planar_sweep(backend, 2.0e-2F, perturbed,
                          perturbed_contacts, error)) return false;
    const float three_dimensional_response = std::max(
        std::abs(perturbed.angularVelocity[0]),
        std::abs(perturbed.angularVelocity[1]));
    if (three_dimensional_response <= 1.0e-4F) {
        error = "face-patch reduction discarded a real three-dimensional response";
        return false;
    }
    std::cout << "Planar sweep symmetry leakage=" << leakage
              << " rad/s, perturbed response="
              << three_dimensional_response << " rad/s\n";
    return true;
}

bool validate_face_contact_normals(const TriangleMesh &mesh,
                                  std::optional<PmNativeBackend> backend,
                                  std::string &error) {
    SceneDefinition scene;
    TriangleMesh refined = mesh;
    // Exercise the triangle path, not the <=96-index convex-face fast path.
    while (refined.indices.size() <= 96U) {
        std::vector<std::uint32_t> indices;
        for (std::size_t triangle = 0U; triangle < refined.indices.size(); triangle += 3U) {
            const auto a = refined.indices[triangle];
            const auto b = refined.indices[triangle + 1U];
            const auto c = refined.indices[triangle + 2U];
            const auto midpoint = [&](std::uint32_t first, std::uint32_t second) {
                const auto p = refined.vertices[first].position;
                const auto q = refined.vertices[second].position;
                const auto index = static_cast<std::uint32_t>(refined.vertices.size());
                Vertex vertex;
                vertex.position = {(p.x + q.x) * 0.5F, (p.y + q.y) * 0.5F,
                                   (p.z + q.z) * 0.5F};
                refined.vertices.push_back(vertex);
                return index;
            };
            const auto ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            indices.insert(indices.end(), {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
        }
        refined.indices = std::move(indices);
    }
    const TriangleMesh surface_box = refined;
    // A small dent makes this a closed non-convex surface, like the authored
    // point-scene spheres. Convex/concave classification must not reverse the
    // reaction of the flat ground underneath it.
    auto highest = std::max_element(refined.vertices.begin(), refined.vertices.end(),
        [](const Vertex &a, const Vertex &b) { return a.position.y < b.position.y; })->position;
    for (const auto &vertex : refined.vertices)
        if (vertex.position.y == highest.y &&
            vertex.position.x * vertex.position.x + vertex.position.z * vertex.position.z <
                highest.x * highest.x + highest.z * highest.z) highest = vertex.position;
    for (auto &vertex : refined.vertices)
        if (vertex.position.x == highest.x && vertex.position.y == highest.y &&
            vertex.position.z == highest.z) vertex.position.y -= 0.02F;
    scene.meshes.push_back(refined);
    TriangleMesh floor;
    floor.vertices = {{{-10.0F, 0.0F, -10.0F}}, {{10.0F, 0.0F, -10.0F}},
                      {{10.0F, 0.0F, 10.0F}}, {{-10.0F, 0.0F, 10.0F}},
                      {{-10.0F, -0.3F, -10.0F}}, {{10.0F, -0.3F, -10.0F}},
                      {{10.0F, -0.3F, 10.0F}}, {{-10.0F, -0.3F, 10.0F}}};
    floor.indices = {0U, 2U, 1U, 0U, 3U, 2U, 4U, 5U, 6U, 4U, 6U, 7U,
                     0U, 1U, 5U, 0U, 5U, 4U, 1U, 2U, 6U, 1U, 6U, 5U,
                     2U, 3U, 7U, 2U, 7U, 6U, 3U, 0U, 4U, 3U, 4U, 7U};
    scene.meshes.push_back(floor);
    float minimum_y = refined.vertices.front().position.y;
    for (const auto &vertex : refined.vertices)
        minimum_y = std::min(minimum_y, vertex.position.y);
    RigidBodyDefinition body;
    body.name = "RefinedBox";
    body.mesh_indices = {0U};
    body.options.initial_state.position = {-2.2F, -0.001F - minimum_y, 0.137F};
    scene.rigid_bodies.push_back(body);
    body = {};
    body.name = "Plane";
    body.mesh_indices = {1U};
    body.options.motion = MotionType::static_body;
    scene.rigid_bodies.push_back(body);
    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    if (!CommandScene::create(scene, 2U, simulation, instance, error, backend)) return false;
    std::size_t contacts{};
    float minimum_clearance = 0.0F;
    for (unsigned step = 0U; step < 60U; ++step) {
        if (!simulation.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, frame, error, true))
            return false;
        const auto &state = frame.rigid_states.at(0U);
        const Quaternion orientation{state.orientation[0], state.orientation[1],
            state.orientation[2], state.orientation[3]};
        for (const auto &vertex : refined.vertices)
            minimum_clearance = std::min(minimum_clearance, state.position[1] +
                rigid_mesh_checks::rotate(orientation, vertex.position).y);
        if (minimum_clearance < -0.002F) {
            error = "planar floor penetration=" + std::to_string(-minimum_clearance) +
                " m at frame " + std::to_string(step);
            return false;
        }
        for (const auto &contact : frame.contacts) {
            ++contacts;
            if (std::abs(contact.normal.x) > 1.0e-6F ||
                std::abs(contact.normal.z) > 1.0e-6F ||
                std::abs(contact.normal.y - 1.0F) > 1.0e-6F) {
                error = "planar ground contact generated normal " +
                    std::to_string(contact.normal.x) + "," + std::to_string(contact.normal.y) +
                    "," + std::to_string(contact.normal.z) + " at frame " + std::to_string(step) +
                    " body y=" + std::to_string(frame.rigid_states.at(0U).position[1]);
                return false;
            }
        }
    }
    if (contacts == 0U) { error = "face-normal regression produced no contacts"; return false; }
    std::cout << "Planar face normals passed (" << contacts << " contacts), penetration="
              << -minimum_clearance << " m\n";
    // Adjacent closed boxes share coplanar top/bottom/front/back surfaces.
    // Those co-facing surfaces are not opposing collision boundaries.
    SceneDefinition seam_scene;
    seam_scene.meshes = {surface_box, surface_box};
    float minimum_x = surface_box.vertices.front().position.x;
    float maximum_x = minimum_x;
    float minimum_y_seam = surface_box.vertices.front().position.y;
    float maximum_y_seam = minimum_y_seam;
    for (const auto &vertex : surface_box.vertices) {
        minimum_x = std::min(minimum_x, vertex.position.x);
        maximum_x = std::max(maximum_x, vertex.position.x);
        minimum_y_seam = std::min(minimum_y_seam, vertex.position.y);
        maximum_y_seam = std::max(maximum_y_seam, vertex.position.y);
    }
    body = {};
    body.name = "SeamDynamic";
    body.mesh_indices = {0U};
    seam_scene.rigid_bodies.push_back(body);
    body.name = "SeamStatic";
    body.mesh_indices = {1U};
    body.options.motion = MotionType::static_body;
    body.options.initial_state.position.x = maximum_x - minimum_x - 0.001F;
    body.options.initial_state.position.y = 0.9F * (maximum_y_seam - minimum_y_seam);
    seam_scene.rigid_bodies.push_back(body);
    for (bool open : {false, true}) {
        if (open) {
            // An uncapped volume still has exterior support faces. Missing
            // faces elsewhere must not make its front/back seam collide.
            auto &indices = seam_scene.meshes[0].indices;
            std::vector<std::uint32_t> remaining;
            for (std::size_t i = 0; i < indices.size(); i += 3U) {
                bool top = true;
                for (std::size_t corner = 0; corner < 3U; ++corner)
                    top = top && surface_box.vertices[indices[i + corner]].position.y ==
                        maximum_y_seam;
                if (!top) remaining.insert(remaining.end(),
                    indices.begin() + i, indices.begin() + i + 3U);
            }
            indices = std::move(remaining);
            // An extruded concave side wall is not a support plane. Its
            // interior edges must still not collide with the other box's
            // front/back surfaces, as in the authored gear/rack mesh pair.
            for (auto &vertex : seam_scene.meshes[0].vertices)
                if (vertex.position.x == maximum_x &&
                    vertex.position.y > minimum_y_seam &&
                    vertex.position.y < maximum_y_seam)
                    vertex.position.x -= 0.02F;
        }
        CommandScene seam_simulation;
        if (!CommandScene::create(seam_scene, 2U, seam_simulation, instance, error, backend) ||
            !seam_simulation.step(1.0F / 60.0F, 1U, {}, frame, error, true)) return false;
        if (frame.contacts.empty()) { error = "surface seam produced no contacts"; return false; }
        for (const auto &contact : frame.contacts)
            if (std::abs(contact.normal.z) > 1.0e-6F) {
                error = std::string(open ? "open" : "closed") +
                    " co-facing surface seam generated normal " +
                    std::to_string(contact.normal.x) + "," + std::to_string(contact.normal.y) +
                    "," + std::to_string(contact.normal.z);
                return false;
            }
        std::cout << (open ? "Open" : "Closed") << " co-facing surface seams passed\n";
    }
    // A truly flat open sheet has no certified inside. Keep collision from
    // both sides, irrespective of its authored triangle winding.
    for (unsigned probe = 0U; probe < 4U; ++probe) {
        const float side = probe == 0U ? -1.0F : 1.0F;
        SceneDefinition sheet_scene;
        sheet_scene.meshes = {surface_box, floor};
        if (probe < 2U) {
            sheet_scene.meshes[1].indices.resize(6U);
            sheet_scene.meshes[1].vertices.resize(4U);
        } else {
            // A vertical open sheet must keep its real boundary-edge impact
            // against the closed floor; only interior/tangent seams are culled.
            // Also align with the floor's internal diagonal. Contact must
            // depend on the whole collider, not its arbitrary triangulation.
            const float diagonal = probe == 3U ? 0.1F : 0.0F;
            sheet_scene.meshes[0].vertices = {
                {{-diagonal, minimum_y_seam, -0.1F}},
                {{-diagonal, maximum_y_seam, -0.1F}},
                {{diagonal, maximum_y_seam, 0.1F}},
                {{diagonal, minimum_y_seam, 0.1F}}};
            sheet_scene.meshes[0].indices = {0U, 1U, 2U, 0U, 2U, 3U};
        }
        body = {};
        body.name = "SheetProbe";
        body.mesh_indices = {0U};
        body.options.linear_damping = 0.0F;
        body.options.angular_damping = 0.0F;
        body.options.initial_state.position.y = side > 0.0F
            ? 0.02F - minimum_y_seam : -0.02F - maximum_y_seam;
        body.options.initial_state.linear_velocity.y = -side;
        sheet_scene.rigid_bodies.push_back(body);
        body = {};
        body.name = "TwoSidedSheet";
        body.mesh_indices = {1U};
        body.options.motion = MotionType::static_body;
        sheet_scene.rigid_bodies.push_back(body);
        CommandScene sheet_simulation;
        if (!CommandScene::create(sheet_scene, 2U, sheet_simulation, instance, error, backend))
            return false;
        bool saw_contact = false;
        for (unsigned step = 0U; step < 30U; ++step) {
            if (!sheet_simulation.step(1.0F / 60.0F, 1U, {}, frame, error, true)) return false;
            for (const auto &contact : frame.contacts) {
                saw_contact = true;
                if (contact.normal.y * side < 0.99F) {
                    error = "open sheet reversed a two-sided contact";
                    return false;
                }
            }
            const auto &state = frame.rigid_states[0];
            const Quaternion orientation{state.orientation[0], state.orientation[1],
                state.orientation[2], state.orientation[3]};
            for (const auto &vertex : sheet_scene.meshes[0].vertices) {
                const float height = state.position[1] +
                    rigid_mesh_checks::rotate(orientation, vertex.position).y;
                if (height * side < -0.002F) {
                    error = "open-surface probe " + std::to_string(probe) +
                        " penetration=" + std::to_string(-height * side) +
                        " m at frame " + std::to_string(step);
                    return false;
                }
            }
        }
        if (!saw_contact) { error = "two-sided sheet contact missing"; return false; }
        std::cout << "Open-surface probe " << probe << " passed\n";
    }
    std::cout << "Two-sided open sheet contacts passed\n";
    return true;
}

} // namespace

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    const std::filesystem::path asset_root{PARALLEL_MATER_GALLERY_ASSET_DIR};
    std::optional<PmNativeBackend> backend;
    std::optional<PmNativeBackend> reference_backend;
    std::uint32_t requested_frames = 0U;
    bool rigid_only = false;
    bool trace_comparison = false;
    std::string selected_scene;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--rigid-only") rigid_only = true;
        else if (option == "--trace-comparison") trace_comparison = true;
        else if (option == "--scene" && argument + 1 < argc)
            selected_scene = argv[++argument];
        else if ((option == "--backend" || option == "--compare-backend") && argument + 1 < argc) {
            const std::string value = argv[++argument];
            auto &selected = option == "--backend" ? backend : reference_backend;
            if (value == "d3d12") selected = PM_NATIVE_BACKEND_D3D12;
            else if (value == "cuda") selected = PM_NATIVE_BACKEND_CUDA;
            else if (value == "vulkan") selected = PM_NATIVE_BACKEND_VULKAN;
            else { std::cerr << "Unknown backend: " << value << '\n'; return 1; }
        } else if (option == "--frames" && argument + 1 < argc) {
            try { requested_frames = std::stoul(argv[++argument]); }
            catch (...) { return 1; }
            if (requested_frames == 0U) return 1;
        } else { std::cerr << "Unknown option: " << option << '\n'; return 1; }
    }
    std::string error;
    std::size_t passed = 0U;
    for (std::size_t index = 0U; index < gallery_entries.size(); ++index) {
        if (rigid_only && index > gallery_context_index(GalleryContext::dump))
            break;
        const GalleryEntry &entry = gallery_entries[index];
        if (!selected_scene.empty() && selected_scene != entry.asset_file)
            continue;
        SceneDefinition authored;
        if (!load_glb_scene(asset_root / entry.asset_file, authored, error)) {
            std::cerr << entry.name << ": load failed: " << error << '\n';
            return 1;
        }

        SceneDefinition scene = std::move(authored);
        if (entry.context == GalleryContext::rigid_body) {
            SceneDefinition generated;
            if (!make_brick_scene(scene, {8U, 1.0F, 1U}, generated, error)) {
                std::cerr << entry.name << ": brick generation failed: "
                          << error << '\n';
                return 1;
            }
            scene = std::move(generated);
        } else if (entry.context == GalleryContext::dump) {
            if (!configure_dump_payload(scene, 10U, error)) {
                std::cerr << entry.name << ": payload generation failed: "
                          << error << '\n';
                return 1;
            }
        }
        if (is_fluid_context(entry.context))
            scene.fluid_options.capacity = 100U;
        if (!apply_gallery_fixed_tick_material_contract(scene, error)) {
            std::cerr << entry.name << ": fixed-tick material migration failed: "
                      << error << '\n';
            return 1;
        }
        std::vector<rigid_mesh_checks::MeshSamples> brick_geometry;
        float maximum_brick_penetration{};
        std::uint32_t maximum_brick_penetration_frame{};
        if (entry.context == GalleryContext::rigid_body)
            for (std::size_t body = 0U; body < scene.rigid_bodies.size(); ++body)
                brick_geometry.emplace_back(scene, body);

        CommandScene simulation;
        CommandSceneInstance instance;
        CommandSceneFrame frame;
        CommandScene reference;
        CommandSceneInstance reference_instance;
        CommandSceneFrame reference_frame;
        if (!CommandScene::create(scene, 2U, simulation, instance, error, backend)) {
            if (index == 0U && error.find("physical device") != std::string::npos) {
                std::cout << "SKIP: no Parallel Mater device\n";
                return 77;
            }
            std::cerr << entry.name << ": create failed: " << error << '\n';
            return 1;
        }
        if (reference_backend && !CommandScene::create(scene, 2U, reference,
                reference_instance, error, reference_backend)) {
            std::cerr << entry.name << ": reference create failed: " << error << '\n';
            return 1;
        }
        if (simulation.read_output(frame, error)) {
            std::cerr << entry.name << ": published output before the first completed step\n";
            return 1;
        }
        const auto &buffers = instance.render_buffers;
        if (buffers.device != simulation.device() || buffers.queue != simulation.queue() ||
            !buffers.output || buffers.rigid.size != scene.rigid_bodies.size() * sizeof(PmRigidState) ||
            buffers.cloth_surfaces.size() != scene.cloths.size() ||
            buffers.soft_surfaces.size() != scene.soft_bodies.size() ||
            buffers.ropes.size() != scene.ropes.size()) {
            std::cerr << entry.name << ": invalid common GPU output bindings\n";
            return 1;
        }
        const std::uint32_t frame_count = requested_frames != 0U
            ? requested_frames
            : (index <= gallery_context_index(GalleryContext::dump) ? 120U : 1U);
        std::cout << entry.name << ": starting " << frame_count << " frames\n" << std::flush;
        for (std::uint32_t frame_index = 0U; frame_index < frame_count;
             ++frame_index) {
            if (!simulation.step(1.0F / 60.0F, 1U,
                                 {0.0F, -9.81F, 0.0F}, frame, error, true) ||
                !validate_frame(scene, frame, error)) {
                std::cerr << entry.name << ": frame " << frame_index
                          << " failed: " << error << '\n';
                return 1;
            }
            if ((frame_index + 1U) % 60U == 0U)
                std::cout << entry.name << ": " << frame_index + 1U
                          << " frames submitted\n" << std::flush;
            if (entry.context == GalleryContext::constraint_point) {
                for (std::size_t body = 0U; body < scene.rigid_bodies.size(); ++body) {
                    if (!scene.rigid_bodies[body].name.starts_with("Ground")) continue;
                    for (const auto &contact : frame.contacts) {
                        if (std::abs(contact.position.x) >= 5.0F ||
                            std::abs(contact.position.z) >= 5.0F) continue;
                        const auto ground = instance.rigid_bodies[body];
                        const float upward = contact.second == ground ? contact.normal.y
                            : contact.first == ground ? -contact.normal.y : 1.0F;
                        if (upward < 0.99F) {
                            std::cerr << "Point-scene ground reaction reversed at frame "
                                      << frame_index << " normal.y=" << upward << '\n';
                            return 1;
                        }
                    }
                }
            }
            if (!brick_geometry.empty()) {
                const auto state = [&](std::size_t body) {
                    const auto &value = frame.rigid_states[body];
                    return RigidBodyState{{value.position[0],value.position[1],value.position[2]},
                        {value.orientation[0],value.orientation[1],value.orientation[2],value.orientation[3]},
                        {}, {}};
                };
                for (std::size_t a = 0U; a < brick_geometry.size(); ++a)
                    for (std::size_t b = a + 1U; b < brick_geometry.size(); ++b) {
                        if (!scene.rigid_bodies[a].name.starts_with("Brick_") ||
                            !scene.rigid_bodies[b].name.starts_with("Brick_")) continue;
                        const float depth = std::max(
                            brick_geometry[a].penetration(state(a), brick_geometry[b], state(b)),
                            brick_geometry[b].penetration(state(b), brick_geometry[a], state(a)));
                        if (depth > maximum_brick_penetration) {
                            maximum_brick_penetration = depth;
                            maximum_brick_penetration_frame = frame_index;
                        }
                        if (depth > 0.002F) {
                            std::cerr << "Brick separation failed at frame " << frame_index
                                      << " bodies=" << a << ',' << b << " penetration=" << depth << '\n';
                            for (const auto body : {a, b}) {
                                const auto &value = frame.rigid_states[body];
                                std::cerr << scene.rigid_bodies[body].name << " position=";
                                for (float v : value.position) std::cerr << v << ',';
                                std::cerr << " orientation=";
                                for (float v : value.orientation) std::cerr << v << ',';
                                std::cerr << " velocity=";
                                for (float v : value.linearVelocity) std::cerr << v << ',';
                                std::cerr << " angular=";
                                for (float v : value.angularVelocity) std::cerr << v << ',';
                                std::cerr << '\n';
                            }
                            for (const auto &contact : frame.contacts)
                                if ((contact.first == a && contact.second == b) ||
                                    (contact.first == b && contact.second == a)) {
                                    std::cerr << "contact penetration=" << contact.penetration
                                              << " impulse=" << contact.normalImpulse << " normal=";
                                    for (float v : {contact.normal.x, contact.normal.y, contact.normal.z}) std::cerr << v << ',';
                                    std::cerr << " position=";
                                    for (float v : {contact.position.x, contact.position.y, contact.position.z}) std::cerr << v << ',';
                                    std::cerr << '\n';
                                }
                            return 1;
                        }
                    }
            }
            if (reference_backend &&
                (!reference.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, reference_frame, error, true) ||
                 !validate_frame(scene, reference_frame, error) ||
                 !test::compare_frames(frame, reference_frame, error, trace_comparison))) {
                std::cerr << entry.name << ": backend comparison frame " << frame_index
                          << " failed: " << error << '\n';
                return 1;
            }
        }

        if (requested_frames >= 120U &&
            (entry.context == GalleryContext::peg_paint ||
             entry.context == GalleryContext::cloth_paint)) {
            const auto painted = std::count_if(
                frame.paint_fields.begin(), frame.paint_fields.end(),
                [](const auto &field) {
                    return std::any_of(field.begin(), field.end(),
                                       [](std::uint32_t pixel) {
                                           return pixel != 0U;
                                       });
                });
            if (painted == 0) {
                std::cerr << entry.name
                          << ": paint field remained empty after "
                          << requested_frames << " frames\n";
                return 1;
            }
        }

        if (entry.context == GalleryContext::constraint_point) {
            for (std::uint32_t constraint : instance.rigid_constraints)
                if (!simulation.set_constraint_active(constraint, false, error)) {
                    std::cerr << entry.name << ": release failed: " << error << '\n';
                    return 1;
                }
            if (!simulation.step(1.0F / 60.0F, 1U,
                                 {0.0F, -9.81F, 0.0F}, frame, error)) {
                std::cerr << entry.name << ": released step failed: "
                          << error << '\n';
                return 1;
            }
            for (std::uint32_t constraint : instance.rigid_constraints)
                if (!simulation.set_constraint_active(constraint, true, error)) {
                    std::cerr << entry.name << ": restore failed: " << error << '\n';
                    return 1;
                }
        }

        // Native drawing must not require deformable CPU vectors. A later
        // diagnostic read must recover the same latest output without stepping.
        if (!simulation.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, frame,
                             error, true, false) || frame.full_host_output ||
            !frame.fluid_particles.empty() || !frame.cloth_vertices.empty() ||
            !simulation.read_output(frame, error) || !frame.full_host_output ||
            !validate_frame(scene, frame, error)) {
            std::cerr << entry.name << ": GPU-only/diagnostic read failed: " << error << '\n';
            return 1;
        }
        // Apple drawing needs no per-frame host snapshot at all. Explicit
        // diagnostic access must still recover that exact completed output.
        CommandSceneFrame gpu_only;
        if (!simulation.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, gpu_only,
                error, false, false, false) || gpu_only.full_host_output ||
            gpu_only.revision != 0 || !gpu_only.rigid_states.empty() ||
            !gpu_only.fluid_particles.empty() || !gpu_only.cloth_vertices.empty() ||
            gpu_only.statistics.stepCount == 0U ||
            !simulation.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, gpu_only,
                error, false, false, false) || gpu_only.full_host_output ||
            gpu_only.revision != 0 || !gpu_only.rigid_states.empty() ||
            !simulation.read_output(gpu_only, error) || !validate_frame(scene, gpu_only, error)) {
            std::cerr << entry.name << ": no-host-output step failed: " << error << '\n';
            return 1;
        }
        if (!simulation.reset(error)) {
            std::cerr << entry.name << ": reset failed: " << error << '\n';
            return 1;
        }
        if (simulation.read_output(frame, error)) {
            std::cerr << entry.name << ": reset retained completed output\n";
            return 1;
        }
        // Reset must recreate the selected driver and resume submission.
        if (backend) {
            PmNativeDeviceInfo native{};
            if (pmGetNativeDevice(simulation.device(), &native) != PM_SUCCESS || native.backend != *backend) {
                std::cerr << entry.name << ": reset changed the selected backend\n";
                return 1;
            }
        }
        if (!simulation.step(1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F},
                             frame, error, true) ||
            !validate_frame(scene, frame, error)) {
            std::cerr << entry.name << ": reset step failed: " << error << '\n';
            return 1;
        }
        ++passed;
        if (!brick_geometry.empty())
            std::cout << "Independent brick penetration=" << maximum_brick_penetration << " m\n";
        if (!brick_geometry.empty())
            std::cout << "Independent brick penetration frame="
                      << maximum_brick_penetration_frame << '\n';
        if (!brick_geometry.empty() && !validate_torque_free_rotation(
                scene.meshes.at(scene.rigid_bodies.back().mesh_indices.front()), backend, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        if (!brick_geometry.empty() && !validate_swept_face_impact(backend, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        if (!brick_geometry.empty() && !validate_planar_sweep_symmetry(
                backend, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        if (!brick_geometry.empty() && !validate_face_contact_normals(
                scene.meshes.at(scene.rigid_bodies.back().mesh_indices.front()), backend, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        std::cout << entry.name << ": " << frame_count
                  << " frames and reset passed" << std::endl;
    }

    std::cout << passed
              << " public-command gallery scenes passed\n";
    if (passed == 0U) std::cerr << "No gallery scene matched the requested filters\n";
    return passed == 0U ? 1 : 0;
}
