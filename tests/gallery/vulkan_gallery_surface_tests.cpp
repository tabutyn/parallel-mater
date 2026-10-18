// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_catalog.hpp>
#include <parallel_mater_gallery/session.hpp>
#include <parallel_mater_gallery/window.hpp>

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace gallery = parallel_mater::gallery;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

gallery::GallerySessionConfig configuration() {
    const auto root = std::filesystem::path(PM_GALLERY_TEST_ROOT);
    return {
        .assets = root / "examples/assets",
        .profiles = root / "config/device-profiles.json",
        .platform = "Desktop",
        .model = "*",
        .backend = "vulkan",
        .native_backend = PM_NATIVE_BACKEND_VULKAN,
        .gpu_rendering = true,
        .workload_limits = gallery::GalleryWorkloadLimits{
            .name = "vulkan-surface-test",
            .platform = "Desktop",
            .machine_model = "*",
            .backend = "vulkan",
            .brick_count = 4U,
            .fluid_particles = 64U,
            .coupled_bodies = std::numeric_limits<std::uint32_t>::max(),
            .smoke_particles = 64U,
            .dump_spheres = 8U,
            .render_width = 96U,
            .render_height = 72U,
            .target_fps = 60U,
        },
        .bricks = gallery::BrickSceneConfig{4U, 1.5F, 2U},
        .dump_spheres = 8U,
        .fluid_particles = 64U,
        .initial_scene = static_cast<unsigned>(gallery::gallery_context_index(
            gallery::GalleryContext::constraint_point)),
        .replacement = gallery::GalleryReplacementPolicy::preserve_until_ready,
        .create_host_renderer = false,
        .prime_scene = true,
    };
}

bool unavailable(const std::string &message) {
    return message.find("No compatible Parallel Mater devices") !=
               std::string::npos ||
        message.find("no Vulkan") != std::string::npos ||
        message.find("enumerate devices") != std::string::npos ||
        message.find("GLFW initialization failed") != std::string::npos ||
        message.find("GLFW window creation failed") != std::string::npos;
}

} // namespace

int main() {
    try {
        {
            gallery::GallerySession session(configuration());
            const PmNativeDeviceInfo device = session.native_render_device();
            std::string error;

            gallery::GalleryWindow first;
            if (!gallery::GalleryWindow::create(
                    480U, 360U, 96U, 72U, "Vulkan Gallery surface test",
                    first, error, PM_NATIVE_BACKEND_VULKAN)) {
                if (unavailable(error)) {
                    std::fprintf(stderr, "SKIP Vulkan Gallery surface: %s\n",
                                 error.c_str());
                    return 77;
                }
                throw std::runtime_error(error);
            }
            auto target = first.native_render_target(device, 96U, 72U, error);
            require(error.empty() && target.handle != 0U,
                    "Could not create first Vulkan Gallery surface: " + error);
            session.set_render_target(target);
            session.frame();

            // Detach before the platform destroys its surface, then prove the
            // same renderer remains usable offscreen.
            session.set_render_target({});
            session.frame();
            first = gallery::GalleryWindow{};

            gallery::GalleryWindow replacement;
            require(gallery::GalleryWindow::create(
                        640U, 400U, 80U, 60U,
                        "Vulkan Gallery replacement surface", replacement,
                        error, PM_NATIVE_BACKEND_VULKAN),
                    "Could not create replacement Vulkan Gallery window: " +
                        error);
            session.resize(80U, 60U);
            target = replacement.native_render_target(device, 80U, 60U, error);
            require(error.empty() && target.handle != 0U,
                    "Could not create replacement Vulkan Gallery surface: " +
                        error);
            session.set_render_target(target);
            session.frame();
            session.set_render_target({});
        }
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Vulkan Gallery surface test leaked a runtime");
        std::puts(
            "PASS Vulkan Gallery surface: attach, detach, offscreen, replacement, resize, teardown");
        return 0;
    } catch (const std::exception &error) {
        if (unavailable(error.what())) {
            std::fprintf(stderr, "SKIP Vulkan Gallery surface: %s\n",
                         error.what());
            return 77;
        }
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
