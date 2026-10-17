// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/session.hpp>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
using parallel_mater::gallery::GallerySession;
int main() {
    try {
        const auto root = std::filesystem::path(PM_IPAD_TEST_RESOURCES);
        GallerySession session({.assets = root / "assets", .profiles = root / "config/device-profiles.json",
            .platform = "iPadOS", .model = "iPad13,8", .backend = "metal", .native_backend = PM_NATIVE_BACKEND_METAL});
        const auto &catalog = session.catalog();
        if (catalog.size() != 28) throw std::runtime_error("Scene catalog mismatch");
        session.resize(480, 360);
        for (unsigned i = 0; i < catalog.size(); ++i) {
            if (!catalog[i].available) continue;
            session.select(i);
            for (unsigned step = 0; step < 3; ++step) session.frame();
            session.validate_snapshot();
            if (session.frames != 3 || !session.timing.latest() || session.timing.latest()->physics_steps != 1)
                throw std::runtime_error("Frame accounting mismatch");
            session.set_paused(true); session.frame();
            if (session.frames != 3 || session.timing.latest()->physics_steps != 0)
                throw std::runtime_error("Paused frame advanced physics");
            session.set_paused(false);
            session.steer(1, -1);
            session.set_active(false);
            session.frame();
            if (session.frames != 3 || session.timing.latest())
                throw std::runtime_error("Background session advanced or retained timing");
            session.set_active(true);
            bool presented = false;
            session.frame([&](const GallerySession &) { presented = true; });
            if (!presented || !session.timing.latest()->present_ms || session.frames != 4)
                throw std::runtime_error("Foreground/presentation recovery failed");
            const auto completed = session.timing.latest()->frame;
            try { session.frame([](const GallerySession &) { throw std::runtime_error("display failure"); }); }
            catch (const std::runtime_error &) {}
            if (session.timing.latest()->frame != completed)
                throw std::runtime_error("Failed presentation published a frame");
            session.reset();
            if (session.frames != 0 || session.timing.latest())
                throw std::runtime_error("Reset retained stale timing");
            std::printf("PASS iPad session: %s\n",
                        std::string(catalog[i].scene.name).c_str());
        }
        bool rejected = false;
        const auto old_selection = session.selection();
        try { session.select(999); } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected || session.selection() != old_selection)
            throw std::runtime_error("Invalid scene replaced live state");
        session.select(2); (void)session.primary_action(); session.frame();
        (void)session.primary_action();
        if (session.frames != 0) throw std::runtime_error("Point restore did not reset");
        // Exercise Android budgets through the same catalog/session on the
        // available host GPU; this is not Android hardware qualification.
        GallerySession android({.assets = root / "assets", .profiles = root / "config/device-profiles.json",
            .platform = "Android", .model = "Pixel 7", .backend = "metal", .native_backend = PM_NATIVE_BACKEND_METAL});
        if (android.catalog().size() != catalog.size() ||
            android.limits.fluid_particles != 128)
            throw std::runtime_error("Android session profile/catalog changed");
        for (unsigned i = 0; i < android.catalog().size(); ++i) {
            if (!android.catalog()[i].available) continue;
            android.select(i); android.frame(); android.validate_snapshot();
        }
        GallerySession gpu({.assets = root / "assets", .profiles = root / "config/device-profiles.json",
            .platform = "iPadOS", .model = "iPad13,8", .backend = "metal", .native_backend = PM_NATIVE_BACKEND_METAL,
            .gpu_rendering = true});
        gpu.resize(480, 360);
        const auto *requested = std::getenv("PM_GALLERY_QUALIFY_FRAMES");
        const unsigned qualification_frames = requested ? static_cast<unsigned>(std::stoul(requested)) : 3;
        for (unsigned i = 0; i < catalog.size(); ++i) {
            if (!catalog[i].available) continue;
            gpu.select(i); (void)gpu.qualify(qualification_frames);
            gpu.validate_snapshot();
            gpu.frame();
            if (!gpu.pixels.empty()) throw std::runtime_error("Interactive GPU frame performed image readback");
            std::printf("PASS direct GPU rendering: %s\n",
                        std::string(catalog[i].scene.name).c_str());
        }
        return 0;
    } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
