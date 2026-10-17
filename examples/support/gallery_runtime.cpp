// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_runtime.hpp>

#include <atomic>
#include <utility>

namespace parallel_mater::gallery {
namespace {
std::atomic_size_t live_runtimes{};
}

struct GalleryRuntime::LifetimeToken {
    LifetimeToken() noexcept { live_runtimes.fetch_add(1U); }
    ~LifetimeToken() { live_runtimes.fetch_sub(1U); }
};

GalleryRuntime::GalleryRuntime() noexcept = default;
GalleryRuntime::~GalleryRuntime() = default;
GalleryRuntime::GalleryRuntime(GalleryRuntime &&) noexcept = default;
GalleryRuntime &GalleryRuntime::operator=(GalleryRuntime &&) noexcept = default;

std::size_t GalleryRuntime::live_runtime_count() noexcept {
    return live_runtimes.load();
}

void GalleryRuntime::clear() noexcept { *this = GalleryRuntime{}; }

bool GalleryRuntime::replace(GalleryRuntimeCreateInfo info,
                             std::string &error) {
    error.clear();
    GalleryRuntime next;
    next.context = info.context;
    if (!load_glb_scene(info.asset, next.scene, error)) {
        error = "scene load failed: " + error;
        return false;
    }
    if (info.configure_scene && !info.configure_scene(next.scene, error)) {
        if (error.empty()) error = "scene configuration failed";
        return false;
    }

    const bool release_first = info.replacement ==
        GalleryReplacementPolicy::release_before_device_create;
    if (release_first) clear();
    const auto retain_retry = [&] {
        if (release_first) recipe_ = info;
    };

    if (!CommandScene::create(next.scene, info.debug_frame_capacity,
                              next.simulation, next.instance, error,
                              info.backend)) {
        if (error.empty()) error = "scene creation failed";
        retain_retry();
        return false;
    }
    next.lifetime_ = std::make_unique<LifetimeToken>();
    if (info.create_host_renderer &&
        !create_renderer(next.scene, next.instance, info.width, info.height,
                         next.renderer, error)) {
        error = "renderer creation failed: " + error;
        retain_retry();
        return false;
    }
    if (info.prepare_runtime && !info.prepare_runtime(next, error)) {
        if (error.empty()) error = "runtime preparation failed";
        retain_retry();
        return false;
    }
    next.recipe_ = std::move(info);
    *this = std::move(next);
    error.clear();
    return true;
}

bool GalleryRuntime::reset(std::string &error) {
    if (!recipe_) {
        error = "Gallery scene has no reset recipe";
        return false;
    }
    GalleryRuntimeCreateInfo recipe = *recipe_;
    return replace(std::move(recipe), error);
}

bool GalleryRuntime::step(const GalleryStepOptions &options,
                          std::string &error) {
    if (!ready()) {
        error = "Choose a scene or Reset to continue";
        return false;
    }
    if (!simulation.step(options.timestep, options.substeps, options.gravity,
                         frame, error,
                         options.collect_rigid_contacts ||
                             options.collect_fluid_contacts,
                         options.read_deformables, options.read_host))
        return false;
    gpu_ready = true;
    if (!options.read_host) return true;
    return make_render_scene_frame(scene, frame, render_frame, error);
}

} // namespace parallel_mater::gallery
