// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <scene_builder.hpp>
#include "backend_comparison.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

using namespace parallel_mater;
using namespace parallel_mater::gallery;

constexpr std::array scene_files{
    "RigidBody.glb", "ConstraintFixed.glb", "ConstraintPoint.glb",
    "ConstraintHinge.glb", "ConstraintPiston.glb", "ConstraintGeneric.glb",
    "ConstraintMotorSpring.glb", "DumpTruck.glb", "Fluid.glb",
    "FluidRigid.glb", "Pegs.glb", "Cloth.glb", "ClothTear.glb",
    "ClothPaint.glb", "ClothWater.glb", "Softbody.glb",
    "SoftbodyRigidBody.glb", "SoftbodyCloth.glb", "SoftbodyFluid.glb",
    "Rope.glb", "RopeFluid.glb", "RopeSoftbody.glb", "RopeCloth.glb",
    "Smoke.glb", "SmokeWater.glb", "SmokeSoftbody.glb",
    "SmokeCloth.glb", "SmokeRope.glb"};

static_assert(scene_files.size() == gallery_entries.size());

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
        if (!selected_scene.empty() && selected_scene != scene_files[index])
            continue;
        const GalleryEntry &entry = gallery_entries[index];
        SceneDefinition authored;
        if (!load_glb_scene(asset_root / scene_files[index], authored, error)) {
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
        if (entry.has_fluid) scene.fluid_options.capacity = 100U;

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
        const std::uint32_t frame_count = requested_frames != 0U
            ? requested_frames
            : (index <= gallery_context_index(GalleryContext::dump) ? 120U : 1U);
        std::cout << entry.name << ": starting " << frame_count << " frames\n" << std::flush;
        for (std::uint32_t frame_index = 0U; frame_index < frame_count;
             ++frame_index) {
            if (!simulation.step(1.0F / 60.0F, 4U,
                                 {0.0F, -9.81F, 0.0F}, frame, error, true) ||
                !validate_frame(scene, frame, error)) {
                std::cerr << entry.name << ": frame " << frame_index
                          << " failed: " << error << '\n';
                return 1;
            }
            if ((frame_index + 1U) % 60U == 0U)
                std::cout << entry.name << ": " << frame_index + 1U
                          << " frames submitted\n" << std::flush;
            if (reference_backend &&
                (!reference.step(1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F}, reference_frame, error, true) ||
                 !validate_frame(scene, reference_frame, error) ||
                 !test::compare_frames(frame, reference_frame, error, trace_comparison))) {
                std::cerr << entry.name << ": backend comparison frame " << frame_index
                          << " failed: " << error << '\n';
                return 1;
            }
        }

        if (entry.context == GalleryContext::constraint_point) {
            for (std::uint32_t constraint : instance.rigid_constraints)
                if (!simulation.set_constraint_active(constraint, false, error)) {
                    std::cerr << entry.name << ": release failed: " << error << '\n';
                    return 1;
                }
            if (!simulation.step(1.0F / 60.0F, 4U,
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
        if (!simulation.step(1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F}, frame,
                             error, true, false) || frame.full_host_output ||
            !frame.fluid_particles.empty() || !frame.cloth_vertices.empty() ||
            !simulation.read_output(frame, error) || !frame.full_host_output ||
            !validate_frame(scene, frame, error)) {
            std::cerr << entry.name << ": GPU-only/diagnostic read failed: " << error << '\n';
            return 1;
        }
        if (!simulation.reset(error)) {
            std::cerr << entry.name << ": reset failed: " << error << '\n';
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
        if (!simulation.step(1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F},
                             frame, error, true) ||
            !validate_frame(scene, frame, error)) {
            std::cerr << entry.name << ": reset step failed: " << error << '\n';
            return 1;
        }
        ++passed;
        std::cout << entry.name << ": " << frame_count
                  << " frames and reset passed" << std::endl;
    }

    std::cout << passed
              << " public-command gallery scenes passed\n";
    if (passed == 0U) std::cerr << "No gallery scene matched the requested filters\n";
    return passed == 0U ? 1 : 0;
}
