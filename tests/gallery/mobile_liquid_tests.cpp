// SPDX-License-Identifier: MIT
#include "scene_builder.hpp"
#include <parallel_mater_gallery/device_profiles.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#ifndef PM_TEST_BACKEND
#define PM_TEST_BACKEND "slang-cpu"
#endif

#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " << #x << ": " << error << '\n'; std::abort(); } } while (false)

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    std::string error;
    // Analytic free fall exercises integration and CPU entry-point indexing.
    // 130 particles crosses both 64-lane and 128-lane boundaries.
    SceneDefinition freefall;
    freefall.fluid_options.capacity = 130;
    freefall.fluid_options.velocity_damping = 0;
    freefall.fluid_options.maximum_speed = 2.5F;
    freefall.fluid_options.rest_particle_volume =
        0.045F * 0.045F * 0.045F;
    for (int i = 0; i < 130; ++i)
        freefall.initial_particles.push_back({.position = {float(i), 10, 0}});
    CommandScene simulation;
    CommandSceneInstance instance;
    CHECK(CommandScene::create(freefall, 0, simulation, instance, error));
    CommandSceneFrame frame;
    for (int i = 0; i < 6; ++i)
        CHECK(simulation.step(1.0F / 60, 1, {0, -9.81F, 0}, frame, error));
    CHECK(frame.fluid_particles.size() == 130);
    for (const auto &particle : frame.fluid_particles) {
        CHECK(std::fabs(particle.velocity[1] + 0.981F) < 1.0e-4F);
        CHECK(particle.position[1] < 10 && particle.position[1] > 9.9F);
    }

    DeviceProfileCatalog profiles;
    CHECK(load_device_profiles(PM_PROFILE_CATALOG, profiles, error));
    const auto *limits = select_workload_limits(profiles, "Android", "Pixel 7", PM_TEST_BACKEND);
    CHECK(limits);
    for (const char *file : {"Fluid.glb", "FluidRigid.glb"}) {
        SceneDefinition scene;
        CHECK(load_glb_scene(std::filesystem::path(PM_SCENE_ROOT) / file, scene, error));
        const auto original_bodies = scene.rigid_bodies.size();
        if (original_bodies > limits->coupled_bodies + 1) {
            // A reduction must fail atomically when a body-index graph exists.
            auto constrained = scene;
            constrained.rigid_constraints.push_back({.body_a = 1, .body_b = 2});
            CHECK(!apply_liquid_workload_limits(constrained, *limits, error));
            CHECK(constrained.rigid_bodies.size() == original_bodies);
            CHECK(constrained.fluid_options.capacity == scene.fluid_options.capacity);
        }
        CHECK(apply_liquid_workload_limits(scene, *limits, error));
        CHECK(scene.rigid_bodies.size() == (original_bodies == 1 ? 1 : limits->coupled_bodies + 1));
        CHECK(std::count_if(scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
            [](const auto &body) { return body.options.motion == MotionType::static_body; }) == 1);
        CommandScene preview;
        CommandSceneInstance bindings;
        CHECK(CommandScene::create(scene, 0, preview, bindings, error));
        for (int i = 0; i < 30; ++i) {
            CHECK(preview.step(1.0F / 60, 1, {0, -9.81F, 0}, frame, error));
            CHECK(!frame.fluid_particles.empty());
            CHECK(frame.fluid_particles.size() <= limits->fluid_particles);
            for (const auto &particle : frame.fluid_particles)
                for (float coordinate : particle.position) CHECK(std::isfinite(coordinate));
            CHECK(frame.rigid_states.size() == scene.rigid_bodies.size());
            for (const auto &body : frame.rigid_states)
                for (float coordinate : body.position) CHECK(std::isfinite(coordinate));
        }
        CHECK(preview.reset(error));
        CHECK(preview.step(1.0F / 60, 1, {0, -9.81F, 0}, frame, error));
    }
}
