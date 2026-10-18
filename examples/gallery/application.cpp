// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/arrow_forces.hpp>
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/fixed_collector.hpp>
#include <parallel_mater_gallery/frame_policy.hpp>
#include <parallel_mater_gallery/frame_timing.hpp>
#include <parallel_mater_gallery/gallery_debug.hpp>
#include <parallel_mater_gallery/gallery_input.hpp>
#include <parallel_mater_gallery/gallery_runtime.hpp>
#include <parallel_mater_gallery/session.hpp>
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/physics_debug.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/runtime_platform.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/surface_query.hpp>
#include <parallel_mater_gallery/window.hpp>
#ifdef PM_GALLERY_NATIVE_GPU_RENDERER
#include <parallel_mater_gallery/gpu_renderer.hpp>
#endif
#include "scene_builder.hpp"
#include "render/render_scene.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
extern "C" const char *parallel_mater_windows_gallery_build_stamp() noexcept;
#endif

namespace {

using parallel_mater::RigidBodyState;
using parallel_mater::Vec3;
using parallel_mater::gallery::CameraController;
using parallel_mater::gallery::CameraDragMode;
using parallel_mater::gallery::GalleryControlPolicy;
using parallel_mater::gallery::GalleryCountKind;
using parallel_mater::gallery::GalleryContext;
using parallel_mater::gallery::GalleryCatalog;
using parallel_mater::gallery::GalleryEntry;
using parallel_mater::gallery::GalleryReplacementPolicy;
using parallel_mater::gallery::GallerySceneParameters;
using parallel_mater::gallery::GallerySession;
using parallel_mater::gallery::GallerySessionConfig;
using parallel_mater::gallery::GalleryWorkloadLimits;
using parallel_mater::gallery::GalleryStepOptions;
using parallel_mater::gallery::GalleryDebugState;
using parallel_mater::gallery::GalleryFramePlan;
using parallel_mater::gallery::GalleryFramePolicy;
using parallel_mater::gallery::GalleryInputState;
using parallel_mater::gallery::GalleryKey;
using parallel_mater::gallery::GallerySceneSource;
using parallel_mater::gallery::GalleryWindow;
using parallel_mater::gallery::gallery_entries;
using parallel_mater::gallery::gallery_entry;
using parallel_mater::gallery::gallery_context_index;
using parallel_mater::gallery::gallery_backend_name;
using parallel_mater::gallery::is_fluid_context;
using parallel_mater::gallery::is_cloth_context;
using parallel_mater::gallery::is_soft_body_context;
using parallel_mater::gallery::is_smoke_context;
using parallel_mater::gallery::uses_rigid_gravity;
using parallel_mater::gallery::RendererHandle;
using parallel_mater::gallery::SmokeDebugMode;
using parallel_mater::gallery::SceneDefinition;
using parallel_mater::gallery::CommandScene;
using parallel_mater::gallery::CommandSceneFrame;
using parallel_mater::gallery::CommandSceneInstance;
using parallel_mater::gallery::RenderSceneFrame;
using parallel_mater::gallery::StaticTriangleSurface;
using parallel_mater::gallery::SurfaceSelection;
using parallel_mater::gallery::BrickSceneConfig;
using parallel_mater::gallery::brick_minimum_count;
using parallel_mater::gallery::brick_maximum_count;
using parallel_mater::gallery::brick_minimum_scale;
using parallel_mater::gallery::brick_maximum_scale;
using parallel_mater::gallery::brick_maximum_planes;
using parallel_mater::gallery::brick_render_width;
using parallel_mater::gallery::brick_render_height;
using parallel_mater::gallery::validate_brick_config;

// UI frame input; translated by the common runtime into public commands.
using StepOptions = GalleryStepOptions;

constexpr float k_timestep =
    1.0F / static_cast<float>(GalleryFramePolicy::default_target_fps);
constexpr float k_gravity = 9.81F;
constexpr float k_pi = 3.14159265358979323846F;
constexpr std::uint32_t k_default_dump_spheres =
    parallel_mater::gallery::default_dump_payload_count;
constexpr std::uint32_t k_default_fluid_particles = 30'000U;

parallel_mater::gallery::CameraPreset camera_preset_for(
    GalleryContext context, BrickSceneConfig bricks) {
    return context == GalleryContext::rigid_body
        ? parallel_mater::gallery::brick_camera_preset(bricks)
        : gallery_entry(context).camera;
}

struct Options {
    std::optional<parallel_mater::gallery::GalleryWorkloadLimits> workload_limits;
    bool fluid_particles_overridden{};
    bool dump_spheres_overridden{};
    std::filesystem::path executable_path{};
    std::filesystem::path scene{PARALLEL_MATER_DEFAULT_SCENE_PATH};
    std::filesystem::path headless_output{};
    std::filesystem::path interactive_smoke_output{};
    std::filesystem::path physics_capture_output{};
    std::filesystem::path frame_timings_output{};
    int frames{240};
    std::uint32_t width{brick_render_width};
    std::uint32_t height{brick_render_height};
#ifdef PARALLEL_MATER_METAL_PREVIEW
    GalleryContext initial_context{GalleryContext::constraint_point};
#else
    GalleryContext initial_context{GalleryContext::rigid_body};
#endif
    BrickSceneConfig bricks{};
    std::filesystem::path profiles_file{PARALLEL_MATER_DEVICE_PROFILES_PATH};
    std::uint32_t dump_spheres{k_default_dump_spheres};
    std::uint32_t fluid_particles{k_default_fluid_particles};
    std::uint32_t contact_pass_limit{};
    std::uint32_t headless_cloth_tilt_degrees{};
    std::uint32_t headless_cloth_tilt_after_frames{};
    std::uint32_t headless_constraint_action_after_frames{};
    bool headless_cloth_tilt_left{};
    bool headless_motor_forward{};
    bool headless_motor_right{};
    bool fluid_particle_view{};
    bool trace_fluid_escapes{};
    bool cloth_debug{};
    bool calibrate{};
    bool bricks_overridden{};
    bool profiles_file_overridden{};
    bool print_brick_profile{};
    std::optional<PmNativeBackend> backend{};
};

std::filesystem::path device_profiles_path(const Options &options);

struct InputState {
    GalleryInputState shared{};
    CameraController *camera{};
    bool replace_count_value{};
    bool count_value_invalid{};
    std::string count_value{};
    std::array<std::string, 3> brick_values{};
    std::size_t brick_field{};
    bool brick_dialog{};
};

enum class KeyAction : std::size_t {
    escape,
    enter,
    backspace,
    scenes,
    particle_count,
    action,
    up,
    down,
    reset,
    timing,
    primary_debug,
    normals,
    rigid_forces,
    fluid_forces,
    bonds,
    velocities,
    capture,
    action_count,
};

class KeyEdges {
  public:
    explicit KeyEdges(GLFWwindow *window) noexcept : window_(window), previous_instance_(active_) {
        active_ = this;
        previous_callback_ = glfwSetKeyCallback(window, key_event);
    }
    ~KeyEdges() {
        glfwSetKeyCallback(window_, previous_callback_);
        active_ = previous_instance_;
    }
    KeyEdges(const KeyEdges &) = delete;
    KeyEdges &operator=(const KeyEdges &) = delete;

    void update() noexcept {
        // Retain ordered press edges across a long GPU frame. Polling only the
        // final key state loses a press and release received in the same poll.
        current_ = KeyAction::action_count;
        if (pending_count_ != 0U) {
            current_ = pending_[pending_begin_];
            pending_begin_ = (pending_begin_ + 1U) % pending_.size();
            --pending_count_;
        }
    }

    [[nodiscard]] bool pressed(KeyAction action) const noexcept {
        return current_ == action;
    }

  private:
    struct Binding { int first; int second; };
    static constexpr std::size_t action_count =
        static_cast<std::size_t>(KeyAction::action_count);
    static constexpr std::array<Binding, action_count> bindings_{
        Binding{GLFW_KEY_ESCAPE, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_ENTER, GLFW_KEY_KP_ENTER},
        Binding{GLFW_KEY_BACKSPACE, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_TAB, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_P, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_SPACE, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_UP, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_DOWN, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_R, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_F, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_V, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_Z, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_X, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_C, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_B, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_N, GLFW_KEY_UNKNOWN},
        Binding{GLFW_KEY_M, GLFW_KEY_UNKNOWN}};
    static void key_event(GLFWwindow *window, int key, int scan, int action, int modifiers) {
        auto *self = active_;
        if (self == nullptr || self->window_ != window) return;
        if (action == GLFW_PRESS && self->pending_count_ < self->pending_.size()) {
            for (std::size_t index = 0U; index < bindings_.size(); ++index) {
                if (key != bindings_[index].first && key != bindings_[index].second) continue;
                self->pending_[(self->pending_begin_ + self->pending_count_++) % self->pending_.size()] =
                    static_cast<KeyAction>(index);
                break;
            }
        }
        if (self->previous_callback_) self->previous_callback_(window, key, scan, action, modifiers);
    }
    inline static thread_local KeyEdges *active_{};
    GLFWwindow *window_{};
    KeyEdges *previous_instance_{};
    GLFWkeyfun previous_callback_{};
    std::array<KeyAction, 256U> pending_{};
    std::size_t pending_begin_{}, pending_count_{};
    KeyAction current_{KeyAction::action_count};
};

[[nodiscard]] std::string backend_title(PmNativeBackend backend) {
    switch (backend) {
    case PM_NATIVE_BACKEND_CUDA: return "CUDA";
    case PM_NATIVE_BACKEND_D3D12: return "DIRECT3D 12";
    case PM_NATIVE_BACKEND_VULKAN: return "VULKAN";
    case PM_NATIVE_BACKEND_METAL: return "METAL";
    }
    return "UNKNOWN";
}

[[nodiscard]] std::string build_revision_label() {
    std::string revision = PARALLEL_MATER_GALLERY_GIT_COMMIT;
    if (revision.size() > 12U) revision.resize(12U);
#if defined(_WIN32)
    revision += "; ";
    revision += parallel_mater_windows_gallery_build_stamp();
#endif
    return revision;
}

[[nodiscard]] std::string build_identity_label() {
    std::string identity = PARALLEL_MATER_GALLERY_GIT_COMMIT;
#if defined(_WIN32)
    identity += "; built ";
    identity += parallel_mater_windows_gallery_build_stamp();
#endif
    return identity;
}

[[nodiscard]] bool choose_backend(
    GalleryWindow &presentation, std::uint32_t width, std::uint32_t height,
    const std::vector<parallel_mater::gallery::GalleryBackendOption> &backends,
    PmNativeBackend &selected_backend, std::string &error) {
    if (backends.empty()) {
        error = "no gallery-capable Parallel Mater backend was found";
        return false;
    }
    std::vector<std::string> names;
    std::vector<std::string> details;
    std::vector<std::uint32_t> pixels;
    try {
        names.reserve(backends.size());
        details.reserve(backends.size());
        for (const auto &backend : backends) {
            names.push_back(backend_title(backend.backend));
            details.push_back(
                backend.device_name + " / " +
                (backend.features.cloth == PM_TRUE &&
                 backend.features.rope == PM_TRUE &&
                 backend.features.softBodies == PM_TRUE &&
                 backend.features.smoke == PM_TRUE
                     ? "FULL GALLERY"
                     : backend.features.particleLiquids == PM_TRUE
                         ? "RIGID + LIQUID PREVIEW"
                         : "RIGID + CONSTRAINTS"));
        }
        pixels.assign(static_cast<std::size_t>(width) * height,
                      0xff120d08U);
    } catch (const std::bad_alloc &) {
        error = "backend menu allocation failed";
        return false;
    }
    std::size_t selection{};
    GLFWwindow *window = presentation.native_handle();
    KeyEdges keys(window);
    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        keys.update();
        if (keys.pressed(KeyAction::up))
            selection = (selection + backends.size() - 1U) % backends.size();
        if (keys.pressed(KeyAction::down))
            selection = (selection + 1U) % backends.size();
        if (keys.pressed(KeyAction::escape)) {
            error = "backend selection cancelled";
            return false;
        }
        if (keys.pressed(KeyAction::enter)) {
            selected_backend = backends[selection].backend;
            return true;
        }
        std::fill(pixels.begin(), pixels.end(), 0xff120d08U);
        parallel_mater::gallery::draw_backend_overlay(
            pixels, width, height, names, details, selection);
        if (!presentation.present(pixels, error)) return false;
    }
    error = "backend selection window closed";
    return false;
}

#include "application_runtime.cpp.inc"

#include "application_input.cpp.inc"

#include "application_validation.cpp.inc"

#include "application_calibration.cpp.inc"

static int run_gallery_application(
    int argc, char **argv, parallel_mater::gallery::FrameTimingCsv &timing_csv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    Options options;
    std::error_code executable_error;
#if defined(__linux__)
    options.executable_path = std::filesystem::read_symlink("/proc/self/exe", executable_error);
    if (executable_error)
#endif
    options.executable_path = std::filesystem::absolute(argv[0], executable_error);
    if (executable_error) options.executable_path = argv[0];
    if (!parse_options(argc, argv, options)) {
        std::cerr << "Invalid arguments. Use --help.\n";
        return 2;
    }
    if (options.calibrate && !options.headless_output.empty()) {
        std::cerr << "--calibrate requires the interactive gallery path\n";
        return 2;
    }
    if (!options.interactive_smoke_output.empty() &&
        (options.frames < 3 || options.calibrate || !options.headless_output.empty())) {
        std::cerr << "--interactive-smoke requires at least 3 frames and cannot combine with headless/calibration\n";
        return 2;
    }
    std::string error;
    std::string window_title =
#ifdef PARALLEL_MATER_METAL_PREVIEW
        "ParallelMater Gallery - Apple GPU Preview (qualification incomplete; Metal graphics)";
#elif defined(PARALLEL_MATER_SLANG_TEST_BUILD)
        "ParallelMater Gallery - Slang Production Test";
#else
        "ParallelMater Gallery";
#endif
    window_title += " " PARALLEL_MATER_GALLERY_VERSION " [" +
        build_revision_label() + "]";
    const auto backends = platform_gallery_backends();
    if (options.backend &&
        std::none_of(backends.begin(), backends.end(), [&](const auto &item) {
            return item.backend == *options.backend;
        })) {
        std::cerr << "Requested backend "
                  << gallery_backend_name(*options.backend)
                  << " is not available\n";
        return 1;
    }
    const bool interactive = options.headless_output.empty() &&
        !options.print_brick_profile;
    if (!options.backend && backends.size() == 1U)
        options.backend = backends.front().backend;
    if (interactive && !options.backend && !backends.empty()) {
        GalleryWindow selector;
        if (!GalleryWindow::create(1280U, 720U, options.width, options.height,
                                   window_title.c_str(), selector, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        PmNativeBackend selected{};
        if (!choose_backend(selector, options.width, options.height,
                            backends, selected, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        options.backend = selected;
    }
    if (options.backend) window_title += " - " + backend_title(*options.backend);
    if (options.headless_output.empty()) {
        const HardwareIdentity hardware =
            platform_hardware_identity(options.backend);
        if (hardware.gpu_model.empty()) {
            std::cerr << "Could not identify the current Parallel Mater device\n";
            return 1;
        }
        DeviceProfileCatalog catalog;
        std::string profile_error;
        if (!load_device_profiles(device_profiles_path(options), catalog, profile_error)) {
            std::cerr << "Device profile catalog unavailable: " << profile_error << '\n';
        }
        const char *platform =
#if defined(_WIN32)
            "Windows";
#elif defined(__APPLE__)
            "macOS";
#else
            "Linux";
#endif
        if (const auto *limits = select_workload_limits(catalog, platform,
                hardware.machine_model, hardware.backend)) {
            options.workload_limits = *limits;
            options.width = limits->render_width;
            options.height = limits->render_height;
            if (!options.fluid_particles_overridden) options.fluid_particles = limits->fluid_particles;
            if (!options.dump_spheres_overridden) options.dump_spheres = limits->dump_spheres;
        }
        const char *selection_label = "command line";
        if (!options.bricks_overridden) {
            const auto selection = select_startup_bricks(
                catalog, hardware, slang_solver_version, options.width, options.height);
            options.bricks = selection.scene;
            selection_label = brick_selection_label(selection.source);
#ifdef PARALLEL_MATER_METAL_PREVIEW
            if (selection.source == BrickSelectionSource::fallback) {
                options.bricks.brick_count = 16U;
                selection_label = "preview default; not performance-qualified";
            }
#endif
            if (options.workload_limits) {
                options.bricks.brick_count = options.workload_limits->brick_count;
                selection_label = options.workload_limits->name.c_str();
            }
        }
        std::cout << "Brick startup: " << hardware.gpu_model << " (" << hardware.backend
                  << ") -> " << options.bricks.brick_count << " bricks, scale "
                  << options.bricks.brick_scale << ", " << options.bricks.wall_planes
                  << " walls [" << selection_label << "]\n";
    }
    if (options.print_brick_profile) return 0;
    if (!options.frame_timings_output.empty() &&
        !timing_csv.open(options.frame_timings_output, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    GallerySession session(desktop_session_config(
        options, options.initial_context));
    const GalleryCatalog &scene_catalog = session.catalog();
    // The target-detach guard below releases swapchains before this platform
    // surface, while the session keeps its borrowed native instance alive.
    GalleryWindow presentation;
    if (interactive &&
        !GalleryWindow::create(1280U, 720U, options.width, options.height,
                               window_title.c_str(), presentation, error,
                               options.backend)) {
        std::cerr << error << '\n';
        return 1;
    }
    struct NativeTargetDetach {
        GallerySession &session;
        ~NativeTargetDetach() { session.detach_render_target(); }
    } native_target_detach{session};
    std::cout << "Gallery scene ready index=" << session.selection()
              << " revision=" << session.revision() << '\n' << std::flush;
    std::cout << "Physics device: " << session.device_name() << '\n';
    std::cout << "Build revision: " << PARALLEL_MATER_GALLERY_GIT_COMMIT << '\n';
    std::uint32_t dump_spheres = options.dump_spheres;
    std::uint32_t fluid_particles = options.fluid_particles;
    std::optional<CalibrationResult> last_calibration;

    const StepOptions step_options{.timestep = session.frame_timestep(),
                                   .substeps = session.frame_substeps(),
                                   .gravity = initial_scene_gravity(
                                       session.context(),
                                       session.scene().gravity_scale),
                                   .rigid_contact_pass_limit = options.contact_pass_limit};
    std::vector<std::uint32_t> pixels;
    InputState input_state;
    input_state.camera = &session.camera();
    input_state.shared.current = gallery_context_index(session.context());
    input_state.shared.selected = input_state.shared.current;
    if (!options.headless_output.empty()) {
        const bool measure_frames = !options.frame_timings_output.empty();
        FrameTimingCollector headless_frame_timings;
        RendererTimings headless_render_timings{};
        const auto render_headless = [&] {
            return session.render_host(
                pixels, error, &headless_render_timings,
                options.cloth_debug ? FluidRenderMode::wireframe :
                options.fluid_particle_view ? FluidRenderMode::particles :
                                             FluidRenderMode::surface);
        };
        StepOptions headless_step = step_options;
        const bool fixed_collection = session.controller_state().directional ==
            GalleryControlPolicy::collector_gravity;
        headless_step.collect_rigid_contacts = fixed_collection;
        Vec3 fixed_headless_gravity = step_options.gravity;
        if (gallery_entry(session.context()).controls ==
                GalleryControlPolicy::cloth_gravity &&
            options.headless_cloth_tilt_degrees != 0U) {
            const float angle = static_cast<float>(
                options.headless_cloth_tilt_degrees) * k_pi / 180.0F;
            const float magnitude = k_gravity * session.scene().gravity_scale;
            headless_step.gravity = {0.0F, -magnitude * std::cos(angle),
                                    -magnitude * std::sin(angle)};
        } else if (uses_rigid_gravity(
                       gallery_entry(session.context()).controls) &&
                   !fixed_collection &&
                   options.headless_cloth_tilt_degrees != 0U) {
            const float angle = static_cast<float>(
                options.headless_cloth_tilt_degrees) * k_pi / 180.0F;
            const float magnitude = k_gravity * session.scene().gravity_scale;
            headless_step.gravity = {magnitude * std::sin(angle),
                                     -magnitude * std::cos(angle), 0.0F};
        } else if (fixed_collection &&
                   options.headless_cloth_tilt_degrees != 0U) {
            const float angle = static_cast<float>(
                options.headless_cloth_tilt_degrees) * k_pi / 180.0F;
            const float magnitude = k_gravity * session.scene().gravity_scale;
            fixed_headless_gravity = {magnitude * std::sin(angle),
                                      -magnitude * std::cos(angle), 0.0F};
        }
        FluidEscapeTrace escape_trace{};
        StaticTriangleSurface floor_index{};
        if (options.trace_fluid_escapes) {
            if (session.scene_instance().fluid == PM_INVALID_INDEX ||
                !StaticTriangleSurface::create(session.scene(), floor_index,
                                               error)) {
                std::cerr << "Fluid escape trace needs a passive collider: "
                          << error << '\n';
                return 1;
            }
            for (const auto &body : session.scene().rigid_bodies) {
                if (body.options.motion != parallel_mater::gallery::MotionType::static_body)
                    continue;
                std::cout << "Passive body " << body.name << " collision_meshes="
                          << body.collision_mesh_indices.size() << " render_meshes="
                          << body.mesh_indices.size() << '\n';
            }
            const Vec3 passive_minimum = floor_index.minimum();
            const Vec3 passive_maximum = floor_index.maximum();
            std::cout << "Passive mesh bounds x=" << passive_minimum.x << ".."
                      << passive_maximum.x << " y=" << passive_minimum.y << ".."
                      << passive_maximum.y << " z=" << passive_minimum.z << ".."
                      << passive_maximum.z << '\n';
            std::cout << "Passive floor triangles="
                      << floor_index.triangle_count() << '\n';
            for (const auto &spawn : session.scene().particle_sources) {
                std::cout << "Fluid source triangles=" << spawn.indices.size()/3
                          << " spacing=" << spawn.spacing << " velocity=("
                          << spawn.options.initial_velocity.x << ','
                          << spawn.options.initial_velocity.y << ','
                          << spawn.options.initial_velocity.z << ")\n";
            }
        }
        for (int frame = 0; frame < options.frames; ++frame) {
            if (measure_frames) {
                headless_frame_timings.begin_frame();
                headless_frame_timings.begin_phase(FramePhase::physics);
            }
            if (options.headless_constraint_action_after_frames != 0U &&
                frame == static_cast<int>(
                    options.headless_constraint_action_after_frames)) {
                try {
                    (void)session.primary_action();
                } catch (const std::exception &failure) {
                    std::cerr << "scene action failed: " << failure.what()
                              << '\n';
                    return 1;
                }
            }
            StepOptions frame_step = headless_step;
            parallel_mater::gallery::GalleryControlInput controls{
                .right = options.headless_motor_right ? 1.0F : 0.0F,
                .forward = options.headless_motor_forward ? 1.0F : 0.0F,
                .camera = session.camera().camera(),
            };
            if (frame < static_cast<int>(options.headless_cloth_tilt_after_frames))
                frame_step.gravity = step_options.gravity;
            else if (options.headless_cloth_tilt_left &&
                     gallery_entry(session.context()).controls ==
                         GalleryControlPolicy::cloth_gravity) {
                controls.right = -1.0F;
            }
            if (fixed_collection) {
                frame_step.gravity =
                    frame < static_cast<int>(
                                options.headless_cloth_tilt_after_frames)
                    ? step_options.gravity : fixed_headless_gravity;
            }
            if (options.headless_cloth_tilt_degrees != 0U ||
                fixed_collection)
                controls.gravity_override = frame_step.gravity;
            if (!session.diagnostic_step(controls, frame_step, error)) {
                std::cerr << "Headless frame " << frame + 1
                          << " failed: " << error << "; peak contacts="
                          << session.output().statistics.peakContactCount << '\n';
                return 1;
            }
            if (measure_frames) {
                headless_frame_timings.end_phase();
                headless_frame_timings.begin_phase(FramePhase::render);
                if (!render_headless()) {
                    std::cerr << "Render failed: " << error << '\n';
                    return 1;
                }
                const auto rendered = FrameTimingCollector::Clock::now();
                headless_frame_timings.end_phase(rendered);
                headless_frame_timings.finish_frame(rendered);
                if (!timing_csv.write(*headless_frame_timings.latest(),
                                     gallery_entry(session.context()).name,
                                     "headless", error)) {
                    std::cerr << error << '\n';
                    return 1;
                }
            } else if (session.scene_instance().fluid != PM_INVALID_INDEX &&
                frame + 1 < options.frames &&
                !session.advance_host_visuals(error)) {
                std::cerr << "Visual update failed: " << error << '\n';
                return 1;
            }
            if (options.trace_fluid_escapes &&
                !trace_fluid_escapes(session.output(), floor_index, frame + 1,
                                     escape_trace))
                return 1;
        }
        if (options.trace_fluid_escapes)
            std::cout << "Fluid escape summary first_frame="
                      << escape_trace.first_frame
                      << " peak_below=" << escape_trace.peak_below
                      << " final_below=" << escape_trace.final_below
                      << " first_local_frame="
                      << escape_trace.first_local_frame
                      << " peak_below_local="
                      << escape_trace.peak_below_local
                      << " final_below_local="
                      << escape_trace.final_below_local
                      << " lowest_y=" << escape_trace.lowest_y << '\n';
        if (options.trace_fluid_escapes &&
            (escape_trace.peak_below != 0U ||
             escape_trace.peak_below_local != 0U)) return 1;
        if (fixed_collection) {
            std::cout << "Fixed collector attached="
                      << session.fixed_attached_count()
                      << " generated_constraints="
                      << session.fixed_constraint_count()
                      << '\n';
        }
        if (session.scene_instance().fluid != PM_INVALID_INDEX) {
            std::cout << "Fluid particles="
                      << session.output().fluid_particles.size()
                      << " peak="
                      << session.output().statistics.peakParticleCount
                      << " capacity_overflows="
                      << session.output().statistics.overflowCount << '\n';
            if (!session.output().fluid_particles.empty()) {
                const auto first = session.output().fluid_particles.front().position;
                Vec3 low{first[0], first[1], first[2]};
                Vec3 high = low;
                for (const auto &particle : session.output().fluid_particles) {
                    const Vec3 point{particle.position[0], particle.position[1],
                                     particle.position[2]};
                    low.x = std::min(low.x, point.x);
                    low.y = std::min(low.y, point.y);
                    low.z = std::min(low.z, point.z);
                    high.x = std::max(high.x, point.x);
                    high.y = std::max(high.y, point.y);
                    high.z = std::max(high.z, point.z);
                }
                std::cout << "Fluid bounds x=" << low.x << ".." << high.x
                          << " y=" << low.y << ".." << high.y
                          << " z=" << low.z << ".." << high.z << '\n';
            }
        }
        if (session.scene_instance().smoke != PM_INVALID_INDEX) {
            std::cout << "Smoke particles="
                      << session.output().smoke_particles.size() << '\n';
        }
        if (!options.physics_capture_output.empty()) {
            if (!write_physics_debug_capture(
                    session.output(), options.physics_capture_output, error)) {
                std::cerr << "Physics capture failed: " << error << '\n';
                return 1;
            }
            std::cout << "Physics capture "
                      << options.physics_capture_output << '\n';
        }
        if (!measure_frames && !render_headless()) {
            std::cerr << "Render failed: " << error << '\n';
            return 1;
        }
        if (!session.read_host_image(pixels, error)) {
            std::cerr << "Render capture failed: " << error << '\n';
            return 1;
        }
        if (options.cloth_debug) {
            if (session.scene_instance().cloths.empty()) {
                std::cerr << "Cloth debug overlay needs a cloth resource\n";
                return 1;
            }
            for (std::uint32_t cloth = 0U;
                 cloth < session.scene_instance().cloths.size(); ++cloth) {
                if (!draw_cloth_debug_overlay(
                        pixels, session.width, session.height,
                        session.scene(), session.output(), cloth,
                        session.camera().camera(),
                        {.normals = true, .rigid_contact_forces = true,
                         .fluid_contact_forces = true, .wireframe = true,
                         .bonds = true}, error)) {
                    std::cerr << "Cloth debug overlay failed: " << error << '\n';
                    return 1;
                }
            }
            std::cout << "Cloth debug surfaces=" << session.scene_instance().cloths.size() << '\n';
            draw_physics_debug_overlay(
                pixels, session.width, session.height,
                session.output(), session.camera().camera(),
                {.contact_normals = true, .rigid_forces = true,
                 .fluid_forces = true, .velocities = true});
        }
        if (session.context() == GalleryContext::peg_paint ||
            session.context() == GalleryContext::cloth_paint) {
            std::uint64_t painted = 0U;
            const std::uint64_t minimum =
                session.context() == GalleryContext::peg_paint ? 100U :
                options.headless_cloth_tilt_degrees != 0U ? 1U : 0U;
            if (!session.paint_coverage(painted, error) ||
                painted < minimum) {
                std::cerr << "Paint coverage failed: " << painted << " texels; need "
                          << minimum << ". " << error << '\n';
                return 1;
            }
            std::cout << "Painted texels=" << painted << '\n';
        }
        if (session.scene_instance().fluid != PM_INVALID_INDEX &&
            !options.fluid_particle_view &&
            !options.cloth_debug)
            std::cout << "Fluid surface outliers="
                      << headless_render_timings.surface_excluded_particle_count
                      << " surface_gpu_ms="
                      << headless_render_timings.surface_gpu_milliseconds
                      << " foam_patches="
                      << headless_render_timings.foam_patch_count
                      << " foam_cpu_ms="
                      << headless_render_timings.foam_wall_milliseconds << '\n';
        if (!validate_render(pixels, session.width,
                             session.height, error)) {
            std::cerr << "Render validation failed: " << error << '\n';
            return 1;
        }
        if (session.context() == GalleryContext::cloth_tear) {
            const auto *bytes = reinterpret_cast<const std::uint8_t *>(
                pixels.data());
            std::size_t cloth_pixels = 0U;
            for (std::size_t index = 0U; index < pixels.size(); ++index) {
                const unsigned red = bytes[4U * index];
                const unsigned green = bytes[4U * index + 1U];
                const unsigned blue = bytes[4U * index + 2U];
                cloth_pixels += blue > red + 25U && blue > green + 20U;
            }
            std::cout << "Visible cloth pixels=" << cloth_pixels << '\n';
            const auto short_edge = std::min(session.width, session.height);
            const auto projected_area = static_cast<std::size_t>(short_edge) *
                                        static_cast<std::size_t>(short_edge);
            if (cloth_pixels < projected_area / 50U) {
                std::cerr << "Cloth Tear render lost the sheet\n";
                if (write_ppm(options.headless_output, pixels,
                              session.width, session.height))
                    std::cerr << "Failed render saved for diagnosis: " << options.headless_output << '\n';
                return 1;
            }
        }
        if (!write_ppm(options.headless_output, pixels, session.width,
                       session.height)) {
            std::cerr << "Failed to write " << options.headless_output << '\n';
            return 1;
        }
        std::cout << "Rendered " << options.headless_output << '\n';
        return 0;
    }

    GLFWwindow *window = presentation.native_handle();
    glfwSetWindowUserPointer(window, &input_state);
    glfwSetMouseButtonCallback(window, &mouse_button);
    glfwSetCursorPosCallback(window, &cursor_position);
    glfwSetScrollCallback(window, &scroll);
    glfwSetCharCallback(window, &character_input);

    if (options.calibrate) {
        CalibrationResult result;
        if (!calibrate_bricks(options, presentation, result, error)) {
            if (result.metrics.duration_seconds > 0.0)
                save_local_calibration(result);
            std::cerr << "Calibration failed: " << error << '\n';
            return 1;
        }
        last_calibration = result;
        save_local_calibration(result);
        return 0;
    }

    KeyEdges keys(window);
    GalleryDebugState debug;
    debug.reset(options.fluid_particle_view);
    using FrameClock = FrameTimingCollector::Clock;
    bool capture_requested = false;
    std::string capture_status;
    bool capture_status_failure = false;
    FrameClock::time_point capture_status_until{};
    unsigned displayed_frames = 0;
    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        const auto frame_time = FrameClock::now();
        glfwPollEvents();
        keys.update();
        if (!options.interactive_smoke_output.empty() && displayed_frames == 1)
            input_state.shared.key(GalleryKey::fps);
        if (keys.pressed(KeyAction::timing))
            input_state.shared.key(GalleryKey::fps);

        if (input_state.shared.parameters_visible) {
            if (input_state.brick_dialog &&
                keys.pressed(KeyAction::fluid_forces)) {
                CalibrationResult result;
                std::string calibration_error;
                bool calibration_succeeded = false;
                if (calibrate_bricks(options, presentation, result,
                                     calibration_error)) {
                    calibration_succeeded = true;
                    last_calibration = result;
                    save_local_calibration(result);
                    input_state.brick_values = {
                        std::to_string(result.config.brick_count),
                        std::to_string(result.config.brick_scale),
                        std::to_string(result.config.wall_planes)};
                    input_state.shared.parameters_visible = false;
                    input_state.count_value_invalid = false;
                } else {
                    if (result.metrics.duration_seconds > 0.0)
                        save_local_calibration(result);
                    input_state.count_value_invalid = true;
                    std::cerr << "Calibration failed: "
                              << calibration_error << '\n';
                }
                if (calibration_succeeded &&
                    !select_desktop_scene(session, options,
                                          GalleryContext::rigid_body,
                                          dump_spheres, fluid_particles,
                                          error)) {
                    std::cerr << error << '\n';
                    break;
                }
                // Calibration uses a separate session, so restarting this
                // scene also resets its common frame clock and timing history.
            }
            if (input_state.brick_dialog &&
                keys.pressed(KeyAction::primary_debug)) {
                std::string save_error;
                if (!last_calibration.has_value() ||
                    last_calibration->config != options.bricks) {
                    save_error =
                        "calibrate this exact configuration before verification";
                } else if (verify_and_save_profile(
                               options, *last_calibration, save_error)) {
                    std::cout << "Verified device profile saved\n";
                    input_state.shared.parameters_visible = false;
                }
                if (!save_error.empty())
                    std::cerr << "Profile not saved: " << save_error << '\n';
            }
            if (keys.pressed(KeyAction::escape)) {
                input_state.shared.key(GalleryKey::escape);
                input_state.count_value_invalid = false;
            }
            if (keys.pressed(KeyAction::backspace)) {
                if (input_state.replace_count_value) {
                    if (input_state.brick_dialog)
                        input_state.brick_values[input_state.brick_field].clear();
                    else
                        input_state.count_value.clear();
                    input_state.replace_count_value = false;
                } else {
                    std::string &value = input_state.brick_dialog
                        ? input_state.brick_values[input_state.brick_field]
                        : input_state.count_value;
                    if (!value.empty()) value.pop_back();
                }
                input_state.count_value_invalid = false;
            }
            if (keys.pressed(KeyAction::scenes)) {
                session.camera().end_drag();
                input_state.shared.key(GalleryKey::catalog);
                input_state.count_value_invalid = false;
            }
            if (keys.pressed(KeyAction::enter)) {
                std::uint32_t requested = 0U;
                const GalleryEntry &entry = gallery_entry(session.context());
                const bool fluid_dialog =
                    entry.count_kind == GalleryCountKind::fluid_particles;
                BrickSceneConfig requested_bricks = options.bricks;
                const bool bricks_valid = !input_state.brick_dialog ||
                    (parse_count(input_state.brick_values[0], brick_minimum_count,
                                 brick_maximum_count, requested_bricks.brick_count) &&
                     parse_scale(input_state.brick_values[1], requested_bricks.brick_scale) &&
                     parse_count(input_state.brick_values[2], 1U, brick_maximum_planes,
                                 requested_bricks.wall_planes) &&
                     validate_brick_config(requested_bricks, error));
                if (!bricks_valid || (!input_state.brick_dialog &&
                    !parse_count(input_state.count_value, entry.minimum_count,
                                 entry.maximum_count, requested))) {
                    input_state.count_value_invalid = true;
                } else {
                    const BrickSceneConfig previous_bricks = options.bricks;
                    if (input_state.brick_dialog) options.bricks = requested_bricks;
                    const std::uint32_t requested_dump =
                        fluid_dialog || input_state.brick_dialog
                            ? dump_spheres : requested;
                    const std::uint32_t requested_fluid =
                        fluid_dialog ? requested : fluid_particles;
                    if (select_desktop_scene(
                            session, options, session.context(),
                            requested_dump, requested_fluid, error)) {
                        dump_spheres = requested_dump;
                        fluid_particles = requested_fluid;
                        input_state.shared.parameters_visible = false;
                        input_state.count_value_invalid = false;
                        if (input_state.brick_dialog)
                            options.bricks_overridden = true;
                    } else {
                        options.bricks = previous_bricks;
                        std::cerr << "Scene restart failed: " << error << '\n';
                        input_state.count_value_invalid = true;
                    }
                }
            }
        } else {
            if (keys.pressed(KeyAction::escape)) {
                input_state.shared.key(GalleryKey::escape);
                if (input_state.shared.close_requested) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
            }
            if (keys.pressed(KeyAction::scenes)) {
                session.camera().end_drag();
                input_state.shared.key(GalleryKey::catalog);
            }
            if (input_state.shared.catalog_visible) {
                std::size_t selected = input_state.shared.selected;
                if (keys.pressed(KeyAction::up)) {
                    if (selected != 0U) --selected;
                }
                if (keys.pressed(KeyAction::down)) {
                    selected = std::min(gallery_entries.size() - 1U,
                                        selected + 1U);
                }
                input_state.shared.selected = selected;
                if (keys.pressed(KeyAction::enter)) {
                    const auto &catalog_selection = scene_catalog[selected];
                    const GalleryContext context_selection =
                        catalog_selection.scene.context;
                    const bool context_changed =
                        context_selection != session.context();
                    if (!catalog_selection.available) {
                        // The catalog explains why this row is disabled.
                    } else if (!context_changed ||
                               select_desktop_scene(
                                   session, options, context_selection,
                                   dump_spheres, fluid_particles, error)) {
                        if (context_changed) {
                            debug.reset(options.fluid_particle_view);
                            input_state.shared.current = selected;
                        }
                        input_state.shared.catalog_visible = false;
                    } else {
                        std::cerr << "Scene switch failed: " << error << '\n';
                    }
                }
            } else if (gallery_entry(session.context()).count_kind !=
                           GalleryCountKind::none &&
                       keys.pressed(KeyAction::particle_count)) {
                session.camera().end_drag();
                input_state.shared.key(GalleryKey::parameters);
                input_state.brick_dialog = gallery_entry(session.context()).count_kind ==
                    GalleryCountKind::brick_scene;
                input_state.brick_field = 0U;
                input_state.brick_values = {std::to_string(options.bricks.brick_count),
                    std::to_string(options.bricks.brick_scale),
                    std::to_string(options.bricks.wall_planes)};
                input_state.count_value = std::to_string(
                    gallery_entry(session.context()).count_kind ==
                            GalleryCountKind::fluid_particles
                        ? fluid_particles
                        : parallel_mater::gallery::dump_payload_count(session.scene()));
                input_state.replace_count_value = true;
                input_state.count_value_invalid = false;
            }
            if (!input_state.shared.catalog_visible &&
                keys.pressed(KeyAction::reset)) {
                if (!reset_desktop_scene(session, error)) {
                    std::cerr << "Scene reset failed: " << error << '\n';
                }
            }
            if (!input_state.shared.catalog_visible &&
                keys.pressed(KeyAction::action)) {
                try {
                    (void)session.primary_action();
                } catch (const std::exception &failure) {
                    std::cerr << "scene action failed: " << failure.what()
                              << '\n';
                    break;
                }
            }
            const bool smoke_debug = is_smoke_context(session.context());
            if (keys.pressed(KeyAction::primary_debug)) {
                if (smoke_debug)
                    debug.toggle_smoke(SmokeDebugMode::density_temperature);
                else
                    debug.toggle_primary(session.context());
            }
            if (keys.pressed(KeyAction::normals)) {
                if (smoke_debug) debug.toggle_smoke(SmokeDebugMode::grid);
                else debug.normals = !debug.normals;
            }
            if (keys.pressed(KeyAction::rigid_forces)) {
                if (smoke_debug) debug.toggle_smoke(SmokeDebugMode::velocity);
                else debug.rigid_forces = !debug.rigid_forces;
            }
            if (keys.pressed(KeyAction::fluid_forces)) {
                if (smoke_debug) debug.toggle_smoke(SmokeDebugMode::pressure);
                else debug.fluid_forces = !debug.fluid_forces;
            }
            if (keys.pressed(KeyAction::bonds)) {
                if (smoke_debug) debug.toggle_smoke(SmokeDebugMode::vorticity);
                else debug.cloth_bonds = !debug.cloth_bonds;
            }
            if (keys.pressed(KeyAction::velocities)) {
                if (smoke_debug) debug.toggle_smoke(SmokeDebugMode::divergence);
                else debug.velocities = !debug.velocities;
            }
            if (keys.pressed(KeyAction::capture))
                capture_requested = true;
        }

        const bool simulation_enabled =
            !input_state.shared.parameters_visible &&
            !input_state.shared.catalog_visible;
        const GalleryEntry &entry = gallery_entry(session.context());
        GalleryFramePlan frame_plan;
        frame_plan.simulation_enabled = simulation_enabled;
        frame_plan.controls.camera = session.camera().camera();
        if (simulation_enabled) {
            frame_plan.controls = directional_input(
                window, entry.controls != GalleryControlPolicy::motor_drive);
            frame_plan.controls.camera = session.camera().camera();
            if (entry.controls == GalleryControlPolicy::peg_gravity) {
                frame_plan.controls.right +=
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) -
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS);
                frame_plan.controls.forward +=
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) -
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS);
            }
        }
        frame_plan.collect_rigid_contacts = debug.rigid_contacts;
        frame_plan.collect_fluid_contacts = debug.rigid_contacts;
        frame_plan.rigid_contact_pass_limit = options.contact_pass_limit;
        frame_plan.host_diagnostics = capture_requested || debug.structure ||
            debug.cloth_bonds || debug.rigid_contacts ||
            debug.vectors_visible() ||
            debug.smoke_mode != SmokeDebugMode::none;

        bool gpu_frame = false;
        try {
            session.frame_at(
                frame_time, frame_plan,
                [&](GallerySession &frame, float interpolation_alpha) {
                    // A debug mode can be enabled while simulation is paused.
                    // Materialize the latest output without submitting physics.
                    if (!frame.output().full_host_output &&
                        (frame.gpu_output_ready() ||
                         frame.output().revision != 0U) &&
                        frame_plan.host_diagnostics &&
                        !frame.read_host_output(error)) {
                        throw std::runtime_error(
                            "read gallery diagnostics failed: " + error);
                    }

                    const bool capture_frame_ready =
                        frame.output().revision != 0U ||
                        frame.gpu_output_ready() ||
                        !frame.output().rigid_states.empty() ||
                        !frame.output().fluid_particles.empty() ||
                        !frame.output().cloth_vertices.empty() ||
                        !frame.output().soft_body_nodes.empty() ||
                        !frame.output().rope_nodes.empty() ||
                        !frame.output().smoke_particles.empty();
                    if (capture_requested && capture_frame_ready) {
                        std::filesystem::path capture_path;
                        if (save_physics_debug_capture(
                                frame.output(), capture_path, error)) {
                            capture_status = "CAPTURED " + capture_path.string();
                            capture_status_failure = false;
                            std::cout << "Physics capture " << capture_path << '\n';
                        } else {
                            capture_status = "CAPTURE FAILED: " + error;
                            capture_status_failure = true;
                            std::cerr << "Physics capture failed: " << error << '\n';
                        }
                        capture_status_until = FrameClock::now() +
                            std::chrono::seconds(4);
                        capture_requested = false;
                    }

                    const Camera current_camera = frame.camera().camera();
                    gpu_frame = frame.has_native_renderer() &&
                        frame.gpu_output_ready();
                    if (gpu_frame) {
                        pixels.assign(std::size_t(frame.width) * frame.height, 0U);
                    } else if (!frame.render_host(
                                   pixels, error, nullptr,
                                   debug.fluid_render_mode(frame.context()),
                                   debug.smoke_mode == SmokeDebugMode::none,
                                   interpolation_alpha)) {
                        throw std::runtime_error("Render failed: " + error);
                    }

                    if (is_smoke_context(frame.context()) &&
                        debug.smoke_mode != SmokeDebugMode::none &&
                        !draw_smoke_debug_overlay(
                            pixels, frame.width, frame.height, frame.output(),
                            current_camera, debug.smoke_mode, error)) {
                        throw std::runtime_error(
                            "Smoke grid debug overlay failed: " + error);
                    }
                    if (is_cloth_context(frame.context()) &&
                        (debug.normals || debug.structure || debug.cloth_bonds)) {
                        for (std::uint32_t cloth = 0U;
                             cloth < frame.scene_instance().cloths.size(); ++cloth) {
                            if (!draw_cloth_debug_overlay(
                                    pixels, frame.width, frame.height,
                                    frame.scene(), frame.output(), cloth,
                                    current_camera,
                                    {.normals = debug.normals,
                                     .wireframe = debug.structure,
                                     .bonds = debug.cloth_bonds}, error)) {
                                throw std::runtime_error(
                                    "Cloth debug overlay failed: " + error);
                            }
                        }
                    }
                    if (is_soft_body_context(frame.context()) &&
                        (debug.normals || debug.structure || debug.cloth_bonds)) {
                        for (std::uint32_t body = 0U;
                             body < frame.scene_instance().soft_bodies.size(); ++body) {
                            if (!draw_soft_body_debug_overlay(
                                    pixels, frame.width, frame.height,
                                    frame.scene(), frame.output(), body,
                                    current_camera,
                                    {.normals = debug.normals,
                                     .wireframe = debug.cloth_bonds,
                                     .bonds = debug.structure}, error)) {
                                throw std::runtime_error(
                                    "Soft-body debug overlay failed: " + error);
                            }
                        }
                    }
                    if (debug.structure || debug.cloth_bonds) {
                        for (std::uint32_t rope = 0U;
                             rope < frame.scene_instance().ropes.size(); ++rope) {
                            if (!draw_rope_debug_overlay(
                                    pixels, frame.width, frame.height,
                                    frame.output(), rope, current_camera, error)) {
                                throw std::runtime_error(
                                    "Rope debug failed: " + error);
                            }
                        }
                    }
                    if (debug.vectors_visible()) {
                        draw_physics_debug_overlay(
                            pixels, frame.width, frame.height, frame.output(),
                            current_camera,
                            {.contact_normals = debug.normals,
                             .rigid_forces = debug.rigid_forces,
                             .fluid_forces = debug.fluid_forces,
                             .velocities = debug.velocities});
                    }
                    if (debug.rigid_contacts &&
                        !draw_rigid_contact_overlay(
                            pixels, frame.width, frame.height,
                            frame.output().contacts, current_camera, error)) {
                        throw std::runtime_error(
                            "Rigid contact overlay failed: " + error);
                    }
                    if (input_state.shared.show_fps) {
                        draw_frame_timing_overlay(
                            pixels, frame.width, frame.height,
                            frame.frame_timing());
                    }
                    if (input_state.shared.catalog_visible) {
                        draw_context_overlay(
                            pixels, frame.width, frame.height,
                            scene_catalog[input_state.shared.selected]
                                .scene.context,
                            scene_catalog);
                    }
                    if (input_state.shared.parameters_visible) {
                        if (input_state.brick_dialog) {
                            draw_brick_settings_overlay(
                                pixels, frame.width, frame.height,
                                input_state.brick_values,
                                input_state.brick_field,
                                input_state.count_value_invalid);
                        } else {
                            draw_count_overlay(
                                pixels, frame.width, frame.height,
                                frame.context(), input_state.count_value,
                                input_state.count_value_invalid);
                        }
                    }
                    if (!capture_status.empty() &&
                        FrameClock::now() < capture_status_until) {
                        draw_status_overlay(
                            pixels, frame.width, frame.height,
                            capture_status, capture_status_failure);
                    }

                    if (gpu_frame) {
                        const bool capture_image =
                            !options.interactive_smoke_output.empty() &&
                            displayed_frames + 1 ==
                                static_cast<unsigned>(options.frames);
                        error.clear();
                        const auto target = presentation.native_render_target(
                            frame.native_render_device(), frame.width,
                            frame.height, error);
                        if (!error.empty() || !frame.render_native(
                                target, pixels,
                                capture_image ? &pixels : nullptr, error)) {
                            throw std::runtime_error(
                                "Presentation failed: " + error);
                        }
                    }
                },
                [&](const GallerySession &frame) {
                    if (!gpu_frame &&
                        !presentation.present(
                            pixels, error, frame.native_image())) {
                        throw std::runtime_error(
                            "Presentation failed: " + error);
                    }
                });
        } catch (const std::exception &failure) {
            std::cerr << failure.what() << '\n';
            break;
        }
        if (!session.frame_timing().latest() ||
            !timing_csv.write(*session.frame_timing().latest(),
                              gallery_entry(session.context()).name,
                              "interactive", error)) {
            if (!error.empty()) std::cerr << error << '\n';
            return 1;
        }
        ++displayed_frames;
        if (!options.interactive_smoke_output.empty() && displayed_frames == static_cast<unsigned>(options.frames)) {
            if (!write_ppm(options.interactive_smoke_output, pixels,
                           session.width, session.height)) return 1;
            std::cout << "Interactive smoke: " << displayed_frames
                      << " frames, renderer=";
            if (gpu_frame)
                std::cout << gallery_backend_name(
                    session.native_render_device().backend)
                          << "-direct-fenced";
            else
                std::cout << "software";
            std::cout << '\n';
            return 0;
        }
    }
    return 0;
}

int parallel_mater_gallery_application(int argc, char **argv) {
    parallel_mater::gallery::FrameTimingCsv timing_csv;
    int result = 1;
    try { result = run_gallery_application(argc, argv, timing_csv); }
    catch (const std::exception &failure) { std::cerr << "Gallery failed: " << failure.what() << '\n'; }
    std::string error;
    if (!timing_csv.close(error)) {
        std::cerr << error << '\n';
        return 1;
    }
    return result;
}
