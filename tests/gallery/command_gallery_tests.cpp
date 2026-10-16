// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <scene_builder.hpp>

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
    std::uint32_t requested_frames = 0U;
    bool rigid_only = false;
    std::string selected_scene;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--rigid-only") rigid_only = true;
        else if (option == "--scene" && argument + 1 < argc)
            selected_scene = argv[++argument];
        else if (option == "--backend" && argument + 1 < argc) {
            const std::string value = argv[++argument];
            if (value == "d3d12") backend = PM_NATIVE_BACKEND_D3D12;
            else if (value == "cuda") backend = PM_NATIVE_BACKEND_CUDA;
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
        if (!CommandScene::create(scene, 2U, simulation, instance, error, backend)) {
            if (index == 0U && error.find("physical device") != std::string::npos) {
                std::cout << "SKIP: no Parallel Mater device\n";
                return 77;
            }
            std::cerr << entry.name << ": create failed: " << error << '\n';
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

        if (!simulation.reset(error)) {
            std::cerr << entry.name << ": reset failed: " << error << '\n';
            return 1;
        }
        // Reset must recreate the selected driver and resume submission.
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
    return 0;
}
