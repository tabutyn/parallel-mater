// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>
#include <render/render_scene.hpp>
#include <scene_builder.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool okay, const std::string &message) {
    if (!okay) throw std::runtime_error(message);
}

void require(parallel_mater::Status status) {
    require(static_cast<bool>(status),
            status.message != nullptr ? status.message : "physics operation failed");
}

} // namespace

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    try {
        SceneDefinition materials;
        std::string error;
        require(load_glb_scene(PARALLEL_MATER_ALPHA_SCENE_PATH, materials, error),
                "load material fixture: " + error);
        constexpr std::array expected{false, false, true, false,
                                      true, true, true, true};
        require(materials.meshes.size() == expected.size() &&
                    materials.rigid_bodies.size() == 1U &&
                    materials.rigid_bodies[0].mesh_indices.size() == expected.size(),
                "mixed-material primitives must all remain collision geometry");
        for (std::size_t i = 0; i < expected.size(); ++i) {
            require(materials.meshes[i].visible == expected[i],
                    "incorrect material visibility for primitive " + std::to_string(i));
            require(materials.meshes[i].indices.size() == 6U,
                    "alpha must not discard triangles");
        }

        SceneDefinition pegs;
        require(load_glb_scene(PARALLEL_MATER_PEGS_SCENE_PATH, pegs, error),
                "load Pegs: " + error);
        std::size_t invisible = 0U;
        for (const auto &body : pegs.rigid_bodies)
            for (const auto mesh : body.mesh_indices)
                if (!pegs.meshes[mesh].visible) {
                    ++invisible;
                    require(body.options.motion == MotionType::static_body &&
                                !pegs.meshes[mesh].indices.empty(),
                            "Pegs invisible cylinder must retain passive triangles");
                }
        require(invisible == 1U, "Pegs must import its alpha-zero cylinder");

        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "Material import passed; SKIP: CUDA unavailable\n";
            return 77;
        }

        // Invisible upper floor, visible lower floor, and a falling rigid body
        // and fluid particle. No scene-specific collision behavior is involved.
        SceneDefinition scene;
        scene.meshes = {materials.meshes[0], materials.meshes[6],
                        make_dump_scene(10U).meshes[2]};
        scene.meshes[0].base_color = {1.0F, 0.0F, 0.0F};
        scene.meshes[1].base_color = {0.0F, 1.0F, 0.0F};
        scene.rigid_bodies = {
            {"invisible floor", {.motion = MotionType::static_body}, {0U}},
            {"visible floor", {.motion = MotionType::static_body,
                .initial_state = {.position = {0.0F, -1.0F, 0.0F}}}, {1U}},
            {"falling body", {.motion = MotionType::dynamic,
                .initial_state = {.position = {0.5F, 0.5F, 0.0F}}}, {2U}}};
        scene.fluid_options.capacity = 1U;
        scene.initial_particles = {{{-0.5F, 0.5F, 0.0F}, {}}};
        CommandScene simulation;
        CommandSceneInstance instance;
        CommandSceneFrame frame;
        require(CommandScene::create(scene, 0U, simulation, instance, error),
                error);
        for (int step = 0; step < 120; ++step)
            require(simulation.step(1.0F / 60.0F, 4U,
                                    {0.0F, -9.81F, 0.0F}, frame, error),
                    error);
        require(frame.rigid_states.size() == scene.rigid_bodies.size() &&
                    frame.fluid_particles.size() == 1U,
                "read public simulation outputs");
        const float rigid_y = frame.rigid_states[2].position[1];
        const float particle_y = frame.fluid_particles[0].position[1];
        require(std::isfinite(rigid_y) && rigid_y > 0.08F &&
                    rigid_y < 0.15F && std::isfinite(particle_y) &&
                    particle_y > 0.0F && particle_y < 0.06F,
                "invisible floor must stop both rigid bodies and fluid");

        const Camera camera{.eye = {0.0F, 3.0F, 3.0F}, .target = {}};
        auto render = [&](const SceneDefinition &definition) {
            GalleryRenderer renderer;
            require(GalleryRenderer::create(
                definition, instance, 96U, 96U, renderer, error), error);
            RenderSceneFrame render_frame;
            require(make_render_scene_frame(definition, frame, render_frame,
                                            error), error);
            // Repeated render also exercises instance updates: visibility must
            // not revert to opaque when rigid transforms are refreshed.
            std::array<std::vector<std::uint32_t>, 2> images;
            require(renderer.render(render_frame, camera, images[0], error), error);
            require(renderer.render(render_frame, camera, images[1], error,
                                    nullptr, FluidRenderMode::particles), error);
            return images;
        };
        const auto hidden = render(scene);
        SceneDefinition reference = scene;
        reference.rigid_bodies[0].mesh_indices.clear(); // Renderer only.
        require(hidden == render(reference),
                "alpha-zero faces must not affect color, water, or particle depth");
        scene.meshes[0].visible = true;
        require(hidden != render(scene),
                "opaque control must cover the lower floor");

        SceneDefinition water_cloth;
        require(load_glb_scene(PARALLEL_MATER_CLOTH_WATER_SCENE_PATH,
                               water_cloth, error),
                "load ClothWater: " + error);
        CommandScene water_simulation;
        CommandSceneInstance water_instance;
        CommandSceneFrame water_frame;
        require(CommandScene::create(water_cloth, 0U, water_simulation,
                                     water_instance, error), error);
        for (int step = 0; step < 10; ++step)
            require(water_simulation.step(1.0F / 60.0F, 4U,
                    {0.0F, -9.81F, 0.0F}, water_frame, error), error);
        const auto render_water = [&](SceneDefinition definition,
                                      FluidRenderMode mode) {
            GalleryRenderer renderer;
            require(GalleryRenderer::create(
                definition, water_instance, 96U, 96U, renderer, error),
                error);
            RenderSceneFrame render_frame;
            require(make_render_scene_frame(definition, water_frame,
                                            render_frame, error), error);
            std::vector<std::uint32_t> image;
            require(renderer.render(render_frame, camera, image, error,
                                    nullptr, mode), error);
            return image;
        };
        const auto transparent = render_water(
            water_cloth, FluidRenderMode::surface);
        SceneDefinition opaque_cloth = water_cloth;
        opaque_cloth.cloths[0].contains_fluid = false;
        require(transparent != render_water(
                    opaque_cloth, FluidRenderMode::surface),
                "contained-fluid cloth must use the transparent skin layer");
        require(transparent != render_water(
                    water_cloth, FluidRenderMode::wireframe),
                "wireframe mode must hide the transparent skin and show particles");

        std::cout << "Material alpha import, rendering, and collisions passed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
