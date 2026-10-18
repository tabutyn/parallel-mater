// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/mobile_qualification.hpp>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
using parallel_mater::gallery::GallerySession;
int main() {
    try {
        const auto root = std::filesystem::path(PM_IPAD_TEST_RESOURCES);
        parallel_mater::gallery::MobileGallerySession mobile({
            .assets = root / "assets",
            .profiles = root / "config/device-profiles.json",
            .platform = "iPadOS",
            .model = "iPad13,8",
            .backend = "metal",
        });
        const auto &catalog = mobile.gallery().catalog();
        if (catalog.size() != 28) throw std::runtime_error("Scene catalog mismatch");
        if (!mobile.attach_surface(480U, 360U))
            throw std::runtime_error(mobile.state().failure);
        const auto common = parallel_mater::gallery::qualify_mobile_gallery(
            mobile, {
                .frames_per_scene = 3U,
                .width = 480U,
                .height = 360U,
                .nonce = "host-metal-contract",
            });
        if (!common.completed || !common.passed || common.scenes.size() != 28U ||
            !common.failures.empty())
            throw std::runtime_error("Common mobile qualification failed on Metal");
        std::printf("PASS common iPad qualification: %zu scenes\n",
                    common.scenes.size());
        // Exercise Android budgets through the same catalog/session on the
        // available host GPU; this is not Android hardware qualification.
        GallerySession android({.assets = root / "assets", .profiles = root / "config/device-profiles.json",
            .platform = "Android", .model = "Pixel 7", .backend = "metal"});
        if (android.catalog().size() != catalog.size() ||
            android.limits.fluid_particles != 128)
            throw std::runtime_error("Android session profile/catalog changed");
        for (unsigned i = 0; i < android.catalog().size(); ++i) {
            if (!android.catalog()[i].available) continue;
            android.select(i); android.frame(); android.validate_snapshot();
        }
        GallerySession gpu({.assets = root / "assets", .profiles = root / "config/device-profiles.json",
            .platform = "iPadOS", .model = "iPad13,8", .backend = "metal",
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
