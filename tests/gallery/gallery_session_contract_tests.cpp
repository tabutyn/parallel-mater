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
        gallery::GallerySessionConfig config{
            .assets = root / "examples/assets",
            .profiles = root / "tests/gallery/assets/session_contract_profiles.json",
            .platform = "ContractTest",
            .model = "test-host",
            .backend = PM_GALLERY_SESSION_TEST_BACKEND,
#ifdef PM_GALLERY_SESSION_TEST_NATIVE_BACKEND
            .native_backend = PM_GALLERY_SESSION_TEST_NATIVE_BACKEND,
#endif
            .scenes = gallery::GallerySceneSet::rigid_liquid,
        };
        gallery::GallerySession session(std::move(config));
        session.resize(96U, 72U);

        const auto names = gallery::GallerySession::scene_names(
            gallery::GallerySceneSet::rigid_liquid);
        require(names.size() == 8U, "Rigid/liquid catalog changed");
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

        session.select(2U); // Constraint: Point in the bounded catalog.
        require(session.selection() == 2U && session.frames == 0U &&
                    !session.timing.latest(),
                "Scene selection retained stale frame state");
        session.frame();
        session.release();
        session.frame();
        require(session.frames == 2U,
                "Point release did not preserve the live scene");
        session.release(); // Restore is defined as a scene reset.
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
        require(rejected && session.ready() && session.selection() == selected,
                "Rejected selection replaced the live scene");

        session.frame();
        session.reset();
        require(session.selection() == selected && session.frames == 0U &&
                    !session.timing.latest(),
                "Reset changed selection or retained frame state");

        std::cout << "Gallery session contract passed on "
                  << session.backend_name() << '\n';
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
