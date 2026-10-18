// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_catalog.hpp>
#include <parallel_mater_gallery/session.hpp>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace gallery = parallel_mater::gallery;

namespace {

void require(bool condition, const char *message) {
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
            .name = "vulkan-renderer-test",
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
        message.find("enumerate devices") != std::string::npos;
}

} // namespace

int main() {
    try {
        {
            gallery::GallerySession session(configuration());
            session.resize(96U, 72U);
            PmNativeDeviceInfo public_device{};
            PmNativeQueueInfo public_queue{};
            require(pmGetNativeDevice(session.runtime().simulation.device(),
                                      &public_device) == PM_SUCCESS,
                    "Vulkan test could not query public native device");
            require(pmGetNativeQueue(session.runtime().simulation.queue(),
                                     &public_queue) == PM_SUCCESS &&
                        public_queue.queue != 0U,
                    "Vulkan test could not query public native queue");
            const PmNativeDeviceInfo renderer_device =
                session.native_render_device();
            require(public_device.backend == PM_NATIVE_BACKEND_VULKAN &&
                        renderer_device.instance == public_device.instance &&
                        renderer_device.physicalDevice ==
                            public_device.physicalDevice &&
                        renderer_device.device == public_device.device,
                    "Gallery renderer did not use the physics Vulkan device");

            session.frame();
            require(session.pixels.empty(),
                    "ordinary Vulkan frame performed image readback");
            for (unsigned frame = 0U; frame < 6U; ++frame) session.frame();
            require(session.pixels.empty(),
                    "frames-in-flight path performed image readback");

            constexpr std::uint32_t overlay_color = 0xffff00ffU;
            std::vector<std::uint32_t> overlay(96U * 72U, 0U);
            std::fill_n(overlay.begin(), 96U * 8U, overlay_color);
            gallery::GpuRenderRequest overlay_request;
            std::string overlay_error;
            require(gallery::make_gpu_render_request(
                        session.runtime().instance.render_buffers,
                        session.runtime().gpu_frame_generation,
                        session.camera().camera(),
                        {PM_NATIVE_BACKEND_VULKAN, public_device.device, 0U,
                         96U, 72U},
                        {&session.pixels}, nullptr, overlay, overlay_request,
                        overlay_error),
                    "Vulkan overlay request creation failed");
            require(session.runtime().native_renderer->render(
                        overlay_request, overlay_error),
                    "Vulkan overlay readback failed");
            require(session.pixels.front() == overlay_color &&
                        session.pixels[71U * 96U] != overlay_color,
                    "Vulkan diagnostic readback is vertically inverted");

            (void)session.qualify(2U);
            require(session.pixels.size() == 96U * 72U,
                    "offscreen Vulkan readback has wrong dimensions");
            require(std::any_of(session.pixels.begin(), session.pixels.end(),
                                [](std::uint32_t pixel) {
                                    return pixel != 0xff1c140eU && pixel != 0U;
                                }),
                    "offscreen Vulkan image contains no rendered output");

            session.resize(80U, 60U);
            (void)session.qualify(2U);
            require(session.pixels.size() == 80U * 60U,
                    "resized Vulkan readback has wrong dimensions");
            session.frame();
            require(session.pixels.empty(),
                    "post-resize ordinary frame performed image readback");
        }
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Vulkan Gallery teardown leaked a runtime");
        std::puts("PASS Vulkan Gallery renderer: direct output, offscreen, resize, frame slots, teardown");
        return 0;
    } catch (const std::exception &error) {
        if (unavailable(error.what())) {
            std::fprintf(stderr, "SKIP Vulkan Gallery renderer: %s\n", error.what());
            return 77;
        }
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
