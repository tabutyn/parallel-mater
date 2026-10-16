// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/device_profiles.hpp>
#include <parallel_mater_gallery/frame_timing.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater/parallel_mater.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parallel_mater::gallery {
// Curated qualification scopes, not different simulation implementations.
enum class GallerySceneSet { all_existing, rigid_liquid };
struct GallerySessionConfig {
    std::filesystem::path assets, profiles;
    std::string platform, model, backend;
    std::optional<PmNativeBackend> native_backend;
    GallerySceneSet scenes{GallerySceneSet::all_existing};
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
    static std::vector<std::string> scene_names(GallerySceneSet scenes = GallerySceneSet::all_existing);
    std::string scene_file() const;
    std::string device_name() const { return device_; }
    static std::string build_revision();
    std::string backend_name() const { return config_.backend; }
    std::uint64_t allocated_bytes() const;
    std::size_t moving_bodies() const;
    std::size_t particle_count() const;
    bool ready() const { return bool(scene_); }
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
    struct Scene;
    std::unique_ptr<Scene> scene_;
    GallerySessionConfig config_;
    unsigned selected_{};
    unsigned bricks_{};
    bool paused_{}, active_{true};
    float right_{}, up_{};
    bool released_{};
    std::string device_;
    void *presentation_layer_{};
    bool diagnostics_{};
    const GalleryEntry &entry(unsigned index) const;
    void require_scene() const;
};
}
