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
    for (const PmRigidState &state : frame.rigid_states)
        if (!finite(state)) {
            error = "rigid output contains a non-finite value";
            return false;
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

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    const std::filesystem::path asset_root{PARALLEL_MATER_GALLERY_ASSET_DIR};
    std::string error;
    for (std::size_t index = 0U; index < gallery_entries.size(); ++index) {
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
        if (!CommandScene::create(scene, 2U, simulation, instance, error)) {
            if (index == 0U && error.find("physical device") != std::string::npos) {
                std::cout << "SKIP: no Parallel Mater device\n";
                return 77;
            }
            std::cerr << entry.name << ": create failed: " << error << '\n';
            return 1;
        }
        if (!simulation.step(1.0F / 60.0F, 4U,
                             {0.0F, -9.81F, 0.0F}, frame, error, true) ||
            !validate_frame(scene, frame, error)) {
            std::cerr << entry.name << ": step failed: " << error << '\n';
            return 1;
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
    }

    std::cout << gallery_entries.size()
              << " public-command gallery scenes passed\n";
    return 0;
}
