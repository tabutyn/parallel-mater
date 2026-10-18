// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/session.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace gallery = parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        const std::filesystem::path root = PM_GALLERY_SESSION_TEST_ROOT;
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Runtime count was not initially zero");
        {
            gallery::GallerySessionConfig config{
                .assets = root / "examples/assets",
                .profiles = root /
                    "tests/gallery/assets/session_contract_profiles.json",
                .platform = "ContractTest",
                .model = "test-host",
                .backend = PM_GALLERY_SESSION_TEST_BACKEND,
#ifdef PM_GALLERY_SESSION_TEST_NATIVE_BACKEND
                .native_backend = PM_GALLERY_SESSION_TEST_NATIVE_BACKEND,
#endif
            };
            gallery::GallerySession session(std::move(config));
            session.resize(96U, 72U);
            require(gallery::GalleryRuntime::live_runtime_count() == 1U,
                    "Session did not own exactly one live runtime");

            require(session.catalog().size() == gallery::gallery_entries.size(),
                    "Shared catalog changed");
            require(session.ready() && session.selection() == 0U,
                    "Session did not select its initial scene");

            session.frame();
            require(session.frames == 1U && session.timing.latest() &&
                        session.timing.latest()->physics_steps == 1U,
                    "Running frame did not advance exactly once");

            session.set_paused(true);
            session.frame();
            require(session.frames == 1U && session.timing.latest() &&
                        session.timing.latest()->physics_steps == 0U,
                    "Paused frame advanced physics");

            session.steer(5.0F, -5.0F);
            session.set_active(false);
            session.frame();
            require(session.frames == 1U && !session.timing.latest(),
                    "Inactive session advanced or retained timing");
            session.set_active(true);
            session.set_paused(false);
            session.frame();
            require(session.frames == 2U && session.timing.latest() &&
                        session.timing.latest()->physics_steps == 1U,
                    "Foreground session did not resume with one normal step");

            session.select(2U); // Constraint: Point in bounded catalog.
            require(session.selection() == 2U && session.frames == 0U &&
                        !session.timing.latest(),
                    "Scene selection retained stale frame state");
            require(gallery::GalleryRuntime::live_runtime_count() == 1U,
                    "Scene replacement retained more than one runtime");
            session.frame();
            require(session.primary_action() ==
                        gallery::GalleryActionOutcome::applied &&
                        session.controller_state().constraints_released,
                    "Mobile semantic action did not release point constraints");
            session.frame();
            require(session.frames == 2U,
                    "Point release did not preserve the live scene");
            require(session.primary_action() ==
                        gallery::GalleryActionOutcome::reset_requested,
                    "Mobile semantic action did not request point restore");
            require(session.selection() == 2U && session.frames == 0U &&
                        !session.timing.latest() &&
                        !session.controller_state().constraints_released,
                    "Point restore did not reset the selected scene");

            const auto selected = session.selection();
            bool rejected = false;
            try {
                session.select(999U);
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            require(rejected && session.ready() &&
                        session.selection() == selected &&
                        gallery::GalleryRuntime::live_runtime_count() == 1U,
                    "Rejected selection replaced the live scene");

            session.frame();
            session.reset();
            require(session.selection() == selected && session.frames == 0U &&
                        !session.timing.latest(),
                    "Reset changed selection or retained frame state");

            if (session.backend_name() == "vulkan") {
                for (const gallery::GalleryContext context : {
                         gallery::GalleryContext::cloth,
                         gallery::GalleryContext::cloth_tear,
                         gallery::GalleryContext::cloth_paint,
                         gallery::GalleryContext::water_cloth}) {
                    const auto item = std::find_if(
                        session.catalog().begin(), session.catalog().end(),
                        [context](const auto &candidate) {
                            return candidate.scene.context == context;
                        });
                    require(item != session.catalog().end() && item->available,
                            "Vulkan cloth scene was not advertised");
                    session.select(static_cast<unsigned>(
                        item - session.catalog().begin()));
                    (void)session.qualify(3U);
                    require(session.ready() && session.selection() ==
                                static_cast<unsigned>(
                                    item - session.catalog().begin()),
                            "Vulkan cloth qualification lost its scene");
                }
            }

            std::cout << "Gallery session contract passed on "
                      << session.backend_name() << '\n';
        }
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Session teardown did not release its runtime exactly once");

        gallery::GalleryRuntimeCreateInfo valid;
        valid.asset = root / "examples/assets/RigidBody.glb";
        valid.width = 96U;
        valid.height = 72U;
#ifdef PM_GALLERY_SESSION_TEST_NATIVE_BACKEND
        valid.backend = PM_GALLERY_SESSION_TEST_NATIVE_BACKEND;
#endif
        gallery::GalleryRuntime runtime;
        std::string failure;
        require(runtime.replace(valid, failure),
                "Direct runtime creation failed");
        require(runtime.ready() &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Direct runtime did not own one scene");
        const PmDevice preserved_device = runtime.simulation.device();

        auto missing = valid;
        missing.asset = root / "examples/assets/Missing.glb";
        require(!runtime.replace(missing, failure) && runtime.ready() &&
                    runtime.simulation.device() == preserved_device &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Preflight failure did not preserve the live scene");

        auto rejected_before_release = valid;
        rejected_before_release.replacement =
            gallery::GalleryReplacementPolicy::release_before_device_create;
        rejected_before_release.configure_scene =
            [](gallery::SceneDefinition &, std::string &error) {
                error = "forced preflight failure";
                return false;
            };
        require(!runtime.replace(rejected_before_release, failure) &&
                    runtime.ready() &&
                    runtime.simulation.device() == preserved_device &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Release-first policy destroyed the scene during preflight");

        auto failed_after_release = valid;
        failed_after_release.replacement =
            gallery::GalleryReplacementPolicy::release_before_device_create;
        failed_after_release.prepare_runtime =
            [](gallery::GalleryRuntime &, std::string &error) {
                error = "forced post-create failure";
                return false;
            };
        require(!runtime.replace(failed_after_release, failure) &&
                    !runtime.ready() &&
                    gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Post-create failure retained stale runtime resources");
        require(!runtime.reset(failure) && !runtime.ready() &&
                    gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Retry failure leaked a runtime");

        require(runtime.replace(valid, failure) &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Runtime did not recover after replacement failure");
        runtime.clear();
        runtime.clear();
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Repeated teardown released the runtime more than once");

        gallery::GalleryRuntimeCreateInfo kinematic = valid;
        kinematic.asset = root / "examples/assets/PassiveActive.glb";
        kinematic.create_host_renderer = false;
        gallery::GalleryRuntime kinematic_runtime;
        require(kinematic_runtime.replace(kinematic, failure) &&
                    kinematic_runtime.controller_state().moves_kinematic &&
                    kinematic_runtime.kinematic_index <
                        kinematic_runtime.instance.rigid_bodies.size(),
                "Rigid semantic action did not bind an authored kinematic body");
        const auto initial_kinematic =
            kinematic_runtime.kinematic_target.position;
        require(kinematic_runtime.controlled_step({
                    .right = 1.0F,
                    .forward = 1.0F,
                    .camera = kinematic_runtime.camera.camera(),
                }, {}, failure) &&
                    kinematic_runtime.kinematic_target.position.x >
                        initial_kinematic.x &&
                    kinematic_runtime.kinematic_target.position.z <
                        initial_kinematic.z,
                "Shared directional action did not move the kinematic target");
        kinematic_runtime.clear();

        gallery::GalleryRuntimeCreateInfo desktop_point = valid;
        desktop_point.context = gallery::GalleryContext::constraint_point;
        desktop_point.asset = root / "examples/assets/Celestial.glb";
        desktop_point.create_host_renderer = false;
        gallery::GalleryRuntime desktop_runtime;
        require(desktop_runtime.replace(desktop_point, failure),
                "Desktop adapter point runtime creation failed");
        require(desktop_runtime.controller_state().primary ==
                    gallery::GalleryPrimaryAction::release_restore_constraints &&
                    desktop_runtime.primary_action(failure) ==
                    gallery::GalleryActionOutcome::applied &&
                    desktop_runtime.controller_state().constraints_released,
                "Desktop semantic action did not match mobile release");
        require(desktop_runtime.primary_action(failure) ==
                    gallery::GalleryActionOutcome::reset_requested &&
                    desktop_runtime.reset(failure) &&
                    !desktop_runtime.controller_state().constraints_released,
                "Desktop semantic action did not match mobile reset");
        desktop_runtime.clear();
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Desktop adapter controller test leaked its runtime");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        if (std::string_view(error.what()) == "no Parallel Mater physical device") {
            std::cout << "SKIP: no selected Gallery physics device\n";
            return 77;
        }
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
