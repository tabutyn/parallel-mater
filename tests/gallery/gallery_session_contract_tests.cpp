// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/session.hpp>

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
            session.release();
            session.frame();
            require(session.frames == 2U,
                    "Point release did not preserve the live scene");
            session.release(); // Restore is a scene reset.
            require(session.selection() == 2U && session.frames == 0U &&
                        !session.timing.latest(),
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
