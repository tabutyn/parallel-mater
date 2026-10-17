// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/device_profiles.hpp>
#include <parallel_mater_gallery/frame_timing.hpp>
#include <parallel_mater_gallery/gallery_catalog.hpp>
#include <parallel_mater_gallery/gallery_runtime.hpp>
#include <parallel_mater/parallel_mater.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parallel_mater::gallery {
struct GallerySessionConfig {
    std::filesystem::path assets, profiles;
    std::string platform, model, backend;
    std::optional<PmNativeBackend> native_backend;
    bool gpu_rendering{};
};

// One worker owns the session. Platform clients supply paths, input and an
// optional presentation callback; no platform solver, controller or clock.
class GallerySession {
public:
    explicit GallerySession(GallerySessionConfig config);
    ~GallerySession();
    void select(unsigned index);
    void reset();
    void resize(unsigned width, unsigned height);
    void orbit(float x, float y);
    void zoom(float amount);
    void focus();
    GalleryActionOutcome primary_action();
    void release();
    void frame(const std::function<void(const GallerySession &)> &present = {});
    void set_paused(bool value);
    void set_active(bool value);
    void steer(float right, float up);
    void set_bricks(unsigned count);
    void validate_snapshot() const;
    // Destructive diagnostic: exercise current scene, then reset it. The caller
    // supplies scheduling/reporting; all platforms run these same checks.
    FrameTimingStatistics qualify(unsigned frame_count);
    std::string details() const;
    std::string status() const;
    [[nodiscard]] const GalleryCatalog &catalog() const noexcept {
        return catalog_;
    }
    std::string scene_file() const;
    std::string device_name() const { return device_; }
    static std::string build_revision();
    std::string backend_name() const { return config_.backend; }
    std::uint64_t allocated_bytes() const;
    std::size_t moving_bodies() const;
    std::size_t particle_count() const;
    GalleryControllerState controller_state() const {
        return runtime_.controller_state();
    }
    bool ready() const { return runtime_.ready(); }
    bool active() const { return active_; }
    bool is_paused() const { return paused_; }
    unsigned selection() const { return selected_; }
    void set_presentation_layer(void *layer) { presentation_layer_ = layer; }
    void *native_render_device() const;
    GalleryWorkloadLimits limits;
    unsigned width{}, height{};
    std::uint64_t frames{};
    std::vector<std::uint32_t> pixels;
    FrameTimingCollector timing;
private:
    GalleryRuntime runtime_;
    GallerySessionConfig config_;
    GalleryCatalog catalog_{make_gallery_catalog()};
    unsigned selected_{};
    unsigned bricks_{};
    bool paused_{}, active_{true};
    float right_{}, up_{};
    bool released_{};
    std::string device_;
    void *presentation_layer_{};
    bool diagnostics_{};
    const GalleryCatalogItem &catalog_item(unsigned index) const;
    const GalleryEntry &entry(unsigned index) const {
        return catalog_item(index).scene;
    }
    void require_scene() const;
};
}
