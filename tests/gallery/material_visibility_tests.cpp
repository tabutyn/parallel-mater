// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>

#include <cuda_runtime_api.h>

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
        World world;
        require(World::create({.rigid_body_capacity = 3U,
                               .triangle_mesh_capacity = 3U}, world));
        SceneInstance instance;
        require(instantiate_scene(scene, world, instance));
        for (int frame = 0; frame < 120; ++frame) require(world.step({}));
        RigidBodyState rigid;
        require(world.read_rigid_body_state(instance.rigid_bodies[2], rigid));
        FluidDeviceView fluid;
        require(world.fluid_view(instance.fluid, fluid));
        Vec3 particle;
        require(fluid.particle_count == 1U &&
                    cudaMemcpy(&particle, fluid.positions.data, sizeof(particle),
                               cudaMemcpyDeviceToHost) == cudaSuccess,
                "read particle on invisible floor");
        require(std::isfinite(rigid.position.y) && rigid.position.y > 0.08F &&
                    rigid.position.y < 0.15F && std::isfinite(particle.y) &&
                    particle.y > 0.0F && particle.y < 0.06F,
                "invisible floor must stop both rigid bodies and fluid");

        const Camera camera{.eye = {0.0F, 3.0F, 3.0F}, .target = {}};
        auto render = [&](const SceneDefinition &definition) {
            OptixRenderer renderer;
            require(OptixRenderer::create(definition, PARALLEL_MATER_OPTIX_PTX_PATH,
                                         96U, 96U, renderer, error), error);
            // Repeated render also exercises instance updates: visibility must
            // not revert to opaque when rigid transforms are refreshed.
            std::array<std::vector<std::uint32_t>, 2> images;
            require(renderer.render(world, instance, camera, images[0], error), error);
            require(renderer.render(world, instance, camera, images[1], error,
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

        std::cout << "Material alpha import, rendering, and collisions passed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
