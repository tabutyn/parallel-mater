// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/device_profiles.hpp>
#include <parallel_mater_gallery/frame_policy.hpp>
#include <parallel_mater_gallery/frame_timing.hpp>
#include <parallel_mater_gallery/gallery_catalog.hpp>
#include <parallel_mater_gallery/gallery_runtime.hpp>
#include <parallel_mater/parallel_mater.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace parallel_mater::gallery {
struct GallerySessionConfig {
    std::filesystem::path assets, profiles;
    std::string platform, model, backend;
    std::optional<PmNativeBackend> native_backend;
    bool gpu_rendering{};
    std::optional<GalleryWorkloadLimits> workload_limits;
    std::optional<BrickSceneConfig> bricks;
    std::uint32_t dump_spheres{};
    std::uint32_t fluid_particles{};
    unsigned initial_scene{};
    std::filesystem::path rigid_body_asset;
    GalleryReplacementPolicy replacement{
        GalleryReplacementPolicy::release_before_device_create};
    std::optional<bool> create_host_renderer;
    bool prime_scene{true};
    std::function<bool(GalleryContext, const SceneDefinition &, std::string &)>
        validate_scene;
};

struct GallerySceneParameters {
    BrickSceneConfig bricks{};
    std::uint32_t dump_spheres{};
    std::uint32_t fluid_particles{};
};

// Per-display-frame requirements supplied by a platform input adapter. The
// session still owns cadence, catch-up limits, solver step size, readback
// admission, renderer ordering, timing publication, and failure cancellation.
struct GalleryFramePlan {
    GalleryControlInput controls{};
    bool simulation_enabled{true};
    bool collect_rigid_contacts{};
    bool collect_fluid_contacts{};
    std::uint32_t rigid_contact_pass_limit{};
    bool host_diagnostics{};
};

// One worker owns the session. Platform clients supply paths, input, monotonic
// frame timestamps, and an optional presentation callback; no platform solver,
// controller, or timing policy.
class GallerySession {
public:
    using FrameClock = GalleryFramePolicy::Clock;
    using FrameRenderer =
        std::function<void(GallerySession &, float interpolation_alpha)>;
    using Presenter = std::function<void(const GallerySession &)>;

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
    void frame(const Presenter &present = {});
    void frame_at(FrameClock::time_point timestamp,
                  const Presenter &present = {});
    void frame_at(FrameClock::time_point timestamp, GalleryFramePlan plan,
                  const FrameRenderer &render, const Presenter &present = {});
    void set_paused(bool value);
    void set_active(bool value);
    void steer(float right, float up);
    void set_bricks(unsigned count);
    void set_scene_parameters(GallerySceneParameters parameters) noexcept;
    [[nodiscard]] GallerySceneParameters scene_parameters() const noexcept;
    // Explicit headless/calibration step. Interactive shells use frame_at so
    // they cannot create a second scheduler or timing path.
    [[nodiscard]] bool diagnostic_step(const GalleryControlInput &input,
                                       GalleryStepOptions options,
                                       std::string &error);
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
    [[nodiscard]] GalleryContext context() const noexcept {
        return runtime_.context;
    }
    [[nodiscard]] const SceneDefinition &scene() const noexcept {
        return runtime_.scene;
    }
    [[nodiscard]] const CommandSceneInstance &scene_instance() const noexcept {
        return runtime_.instance;
    }
    [[nodiscard]] const CommandSceneFrame &output() const noexcept {
        return runtime_.frame;
    }
    CameraController &camera() noexcept { return runtime_.camera; }
    const CameraController &camera() const noexcept { return runtime_.camera; }
    bool ready() const { return runtime_.ready(); }
    bool active() const { return active_; }
    bool is_paused() const { return paused_; }
    unsigned selection() const { return selected_; }
    std::uint64_t revision() const noexcept { return revision_; }
    void set_render_target(GpuRenderTarget target) noexcept;
    void detach_render_target() noexcept;
    PmNativeDeviceInfo native_physics_device() const;
    PmNativeDeviceInfo native_render_device() const;
    PmNativeQueueInfo native_render_queue() const;
    [[nodiscard]] bool has_native_renderer() const noexcept;
    [[nodiscard]] bool gpu_output_ready() const noexcept {
        return runtime_.gpu_ready;
    }
    [[nodiscard]] NativeRenderImage native_image() const noexcept;
    [[nodiscard]] bool render_host(
        std::vector<std::uint32_t> &rgba, std::string &error,
        RendererTimings *renderer_timings = nullptr,
        FluidRenderMode fluid_mode = FluidRenderMode::surface,
        bool show_smoke_particles = true,
        float rigid_interpolation_alpha = 1.0F);
    [[nodiscard]] bool render_native(
        GpuRenderTarget target, std::span<const std::uint32_t> overlay,
        std::vector<std::uint32_t> *readback, std::string &error);
    [[nodiscard]] bool read_host_image(
        std::vector<std::uint32_t> &rgba, std::string &error);
    [[nodiscard]] bool read_host_output(std::string &error);
    [[nodiscard]] bool advance_host_visuals(std::string &error);
    [[nodiscard]] bool paint_coverage(
        std::uint64_t &painted_texels, std::string &error) const;
    [[nodiscard]] bool set_rigid_state(
        std::size_t scene_body_index, const RigidBodyState &state,
        std::string &error);
    [[nodiscard]] std::size_t fixed_attached_count() const noexcept;
    [[nodiscard]] std::size_t fixed_constraint_count() const noexcept;
    [[nodiscard]] const FrameTimingCollector &frame_timing() const noexcept {
        return timing;
    }
    [[nodiscard]] float frame_timestep() const noexcept {
        return frame_policy_.timestep();
    }
    [[nodiscard]] std::uint32_t frame_substeps() const noexcept {
        return frame_policy_.substeps();
    }
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
    GallerySceneParameters parameters_{};
    std::uint64_t revision_{};
    bool paused_{}, active_{true};
    float right_{}, up_{};
    bool released_{};
    std::string device_;
    GpuRenderTarget render_target_{};
    bool diagnostics_{};
    GalleryFramePolicy frame_policy_{};
    const GalleryCatalogItem &catalog_item(unsigned index) const;
    const GalleryEntry &entry(unsigned index) const {
        return catalog_item(index).scene;
    }
    void require_scene() const;
    [[nodiscard]] bool controlled_step(const GalleryControlInput &input,
                                       GalleryStepOptions options,
                                       std::string &error);
};
}
