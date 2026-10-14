// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/arrow_forces.hpp>
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/fixed_collector.hpp>
#include <parallel_mater_gallery/gallery_debug.hpp>
#include <parallel_mater_gallery/gallery_input.hpp>
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/physics_debug.hpp>
#include <parallel_mater_gallery/physics_frame_budget.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/runtime_platform.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/surface_query.hpp>
#include <parallel_mater_gallery/window.hpp>
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
#include <numeric>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using parallel_mater::RigidBodyState;
using parallel_mater::RigidConstraintOptions;
using parallel_mater::RigidConstraintState;
using parallel_mater::RigidConstraintType;
using parallel_mater::MotionType;
using parallel_mater::Quaternion;
using parallel_mater::StepOptions;
using parallel_mater::Vec3;
using parallel_mater::gallery::CameraController;
using parallel_mater::gallery::CameraDragMode;
using parallel_mater::gallery::screen_space_gravity;
using parallel_mater::gallery::FixedContactCollector;
using parallel_mater::gallery::steer_gravity;
using parallel_mater::gallery::peg_paint_gravity_tilt_degrees;
using parallel_mater::gallery::GalleryControlPolicy;
using parallel_mater::gallery::GalleryCountKind;
using parallel_mater::gallery::GalleryContext;
using parallel_mater::gallery::GalleryEntry;
using parallel_mater::gallery::GalleryDebugState;
using parallel_mater::gallery::GalleryInputState;
using parallel_mater::gallery::GalleryKey;
using parallel_mater::gallery::GallerySceneSource;
using parallel_mater::gallery::GalleryWindow;
using parallel_mater::gallery::gallery_entries;
using parallel_mater::gallery::gallery_entry;
using parallel_mater::gallery::gallery_context_index;
using parallel_mater::gallery::is_fluid_context;
using parallel_mater::gallery::is_cloth_context;
using parallel_mater::gallery::is_soft_body_context;
using parallel_mater::gallery::is_smoke_context;
using parallel_mater::gallery::toggles_constraint;
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

constexpr float k_timestep = 1.0F / 60.0F;
constexpr std::uint32_t k_maximum_catch_up_steps = 4U;
constexpr double k_maximum_frame_delta = 0.25;
constexpr float k_kinematic_speed = 2.0F;
constexpr float k_gravity = 9.81F;
constexpr float k_cloth_gravity_tilt_degrees = 45.0F;
constexpr float k_rigid_gravity_tilt_degrees = 30.0F;
constexpr float k_collector_gravity_tilt_degrees =
    parallel_mater::gallery::collector_gravity_tilt_degrees;
constexpr float k_pi = 3.14159265358979323846F;
constexpr float k_motor_speed = 8.0F;
constexpr float k_motor_steering_angle = 25.0F * k_pi / 180.0F;
constexpr float k_motor_steering_speed = 90.0F * k_pi / 180.0F;
constexpr std::uint32_t k_default_dump_spheres =
    parallel_mater::gallery::default_dump_payload_count;
constexpr std::uint32_t k_default_fluid_particles = 30'000U;

parallel_mater::gallery::CameraPreset camera_preset_for(
    GalleryContext context, BrickSceneConfig bricks) {
    return context == GalleryContext::rigid_body
        ? parallel_mater::gallery::brick_camera_preset(bricks)
        : gallery_entry(context).camera;
}

[[nodiscard]] constexpr std::uint32_t scene_substeps(
    GalleryContext context) noexcept {
    return context == GalleryContext::dump ? 8U : 4U;
}

struct Options {
    std::filesystem::path executable_path{};
    std::filesystem::path scene{PARALLEL_MATER_DEFAULT_SCENE_PATH};
    std::filesystem::path headless_output{};
    std::filesystem::path physics_capture_output{};
    int frames{240};
    std::uint32_t width{brick_render_width};
    std::uint32_t height{brick_render_height};
    GalleryContext initial_context{GalleryContext::rigid_body};
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
};

struct InputState {
    GalleryInputState shared{};
    bool replace_count_value{};
    bool count_value_invalid{};
    std::string count_value{};
    std::array<std::string, 3> brick_values{};
    std::size_t brick_field{};
    bool brick_dialog{};
};

struct DirectionalInput {
    float x{};
    float z{};
};

struct SuspensionSteeringJoint {
    std::size_t constraint_index{};
    Quaternion authored_local_orientation_a{};
    bool front{};
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
    void update(GLFWwindow *window) noexcept {
        previous_ = down_;
        for (std::size_t index = 0U; index < bindings_.size(); ++index) {
            const Binding binding = bindings_[index];
            down_[index] = glfwGetKey(window, binding.first) == GLFW_PRESS ||
                (binding.second != GLFW_KEY_UNKNOWN &&
                 glfwGetKey(window, binding.second) == GLFW_PRESS);
        }
    }

    [[nodiscard]] bool pressed(KeyAction action) const noexcept {
        const std::size_t index = static_cast<std::size_t>(action);
        return down_[index] && !previous_[index];
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
    std::array<bool, action_count> down_{};
    std::array<bool, action_count> previous_{};
};

#include "application_runtime.cpp.inc"

#include "application_input.cpp.inc"

#include "application_validation.cpp.inc"

#include "application_calibration.cpp.inc"

int parallel_mater_gallery_application(int argc, char **argv) {
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
    if (options.headless_output.empty()) {
        const HardwareIdentity hardware = platform_hardware_identity();
        if (hardware.gpu_model.empty()) {
            std::cerr << "Could not identify the current Parallel Mater device\n";
            return 1;
        }
        DeviceProfileCatalog catalog;
        const char *selection_label = "command line";
        if (!options.bricks_overridden) {
            std::string profile_error;
            if (!load_device_profiles(device_profiles_path(options), catalog, profile_error)) {
                std::cerr << "Device profile catalog unavailable: " << profile_error << '\n';
                catalog = {};
            }
            const auto selection = select_startup_bricks(
                catalog, hardware, slang_solver_version, options.width, options.height);
            options.bricks = selection.scene;
            selection_label = brick_selection_label(selection.source);
        }
        std::cout << "Brick startup: " << hardware.gpu_model << " (" << hardware.backend
                  << ") -> " << options.bricks.brick_count << " bricks, scale "
                  << options.bricks.brick_scale << ", " << options.bricks.wall_planes
                  << " walls [" << selection_label << "]\n";
    }
    if (options.print_brick_profile) return 0;
    std::string error;
    GallerySession session;
    if (!session.rebuild(options, options.initial_context, options.dump_spheres,
                         options.fluid_particles, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    GalleryRuntime &runtime = session.runtime;
    std::uint32_t &dump_spheres = session.dump_spheres;
    std::uint32_t &fluid_particles = session.fluid_particles;
    Vec3 &peg_gravity = session.peg_gravity;
    Vec3 &cloth_gravity = session.cloth_gravity;
    auto &timings = session.timings;
    auto &statistics = session.statistics;
    RendererTimings &renderer_timings = session.renderer_timings;
    std::optional<CalibrationResult> last_calibration;

    const StepOptions step_options{.timestep = k_timestep,
                                   .substeps = scene_substeps(runtime.context),
                                   .gravity = initial_scene_gravity(
                                       runtime.context,
                                       runtime.scene.gravity_scale),
                                   .rigid_contact_pass_limit = options.contact_pass_limit};
    std::vector<std::uint32_t> pixels;
    InputState input_state;
    input_state.shared.current = gallery_context_index(runtime.context);
    input_state.shared.selected = input_state.shared.current;
    input_state.shared.camera.set_preset(
        camera_preset_for(runtime.context, options.bricks));
    if (!options.headless_output.empty()) {
        StepOptions headless_step = step_options;
        const bool fixed_collection = runtime.fixed_collector.active();
        headless_step.collect_rigid_contacts = fixed_collection;
        Vec3 fixed_headless_gravity = step_options.gravity;
        if (gallery_entry(runtime.context).controls ==
                GalleryControlPolicy::cloth_gravity &&
            options.headless_cloth_tilt_degrees != 0U) {
            const float angle = static_cast<float>(
                options.headless_cloth_tilt_degrees) * k_pi / 180.0F;
            const float magnitude = k_gravity * runtime.scene.gravity_scale;
            headless_step.gravity = {0.0F, -magnitude * std::cos(angle),
                                    -magnitude * std::sin(angle)};
        } else if (uses_rigid_gravity(
                       gallery_entry(runtime.context).controls) &&
                   !fixed_collection &&
                   options.headless_cloth_tilt_degrees != 0U) {
            const float angle = static_cast<float>(
                options.headless_cloth_tilt_degrees) * k_pi / 180.0F;
            const float magnitude = k_gravity * runtime.scene.gravity_scale;
            headless_step.gravity = {magnitude * std::sin(angle),
                                     -magnitude * std::cos(angle), 0.0F};
        } else if (fixed_collection &&
                   options.headless_cloth_tilt_degrees != 0U) {
            const float angle = static_cast<float>(
                options.headless_cloth_tilt_degrees) * k_pi / 180.0F;
            const float magnitude = k_gravity * runtime.scene.gravity_scale;
            fixed_headless_gravity = {magnitude * std::sin(angle),
                                      -magnitude * std::cos(angle), 0.0F};
        }
        FluidEscapeTrace escape_trace{};
        StaticTriangleSurface floor_index{};
        if (options.trace_fluid_escapes) {
            if (runtime.instance.fluid == PM_INVALID_INDEX ||
                !StaticTriangleSurface::create(runtime.scene, floor_index,
                                               error)) {
                std::cerr << "Fluid escape trace needs a passive collider: "
                          << error << '\n';
                return 1;
            }
            for (const auto &body : runtime.scene.rigid_bodies) {
                if (body.options.motion != parallel_mater::MotionType::static_body)
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
            for (const auto &spawn : runtime.scene.particle_sources) {
                std::cout << "Fluid source triangles=" << spawn.indices.size()/3
                          << " spacing=" << spawn.spacing << " velocity=("
                          << spawn.options.initial_velocity.x << ','
                          << spawn.options.initial_velocity.y << ','
                          << spawn.options.initial_velocity.z << ")\n";
            }
        }
        for (int frame = 0; frame < options.frames; ++frame) {
            if (options.headless_constraint_action_after_frames != 0U &&
                frame == static_cast<int>(
                    options.headless_constraint_action_after_frames)) {
                if (runtime.context == GalleryContext::dump) runtime.dump_bed.toggle();
                else if (toggles_constraint(gallery_entry(runtime.context).controls) &&
                         !toggle_constraints(runtime)) return 1;
            }
            if (gallery_entry(runtime.context).controls ==
                GalleryControlPolicy::motor_drive) {
                const DirectionalInput motor_input{
                    options.headless_motor_right ? 1.0F : 0.0F,
                    options.headless_motor_forward ? -1.0F : 0.0F};
                if (!drive_motors(runtime, motor_input) ||
                    !steer_motor_suspension(runtime, motor_input.x)) return 1;
            }
            if (runtime.context == GalleryContext::dump &&
                !runtime.dump_bed.advance(runtime.simulation, runtime.scene,
                    runtime.instance, k_timestep, error)) {
                std::cerr << "tilt dump bucket failed: " << error << '\n';
                return 1;
            }
            StepOptions frame_step = headless_step;
            if (frame < static_cast<int>(options.headless_cloth_tilt_after_frames))
                frame_step.gravity = step_options.gravity;
            else if (options.headless_cloth_tilt_left &&
                     gallery_entry(runtime.context).controls ==
                         GalleryControlPolicy::cloth_gravity) {
                headless_step.gravity = steer_gravity(
                    headless_step.gravity, input_state.shared.camera.camera(),
                    -1.0F, 0.0F, k_gravity * runtime.scene.gravity_scale,
                    k_cloth_gravity_tilt_degrees, k_timestep);
                frame_step.gravity = headless_step.gravity;
            }
            if (fixed_collection) {
                frame_step.gravity =
                    frame < static_cast<int>(
                                options.headless_cloth_tilt_after_frames)
                    ? step_options.gravity : fixed_headless_gravity;
                if (!runtime.fixed_collector.apply_loose_gravity(
                        runtime.simulation, runtime.scene, runtime.instance,
                        step_options.gravity, frame_step.gravity, error)) {
                    std::cerr << "apply loose fixed-scene gravity failed: "
                              << error << '\n';
                    return 1;
                }
            }
            if (!apply_gravity_tilt_overrides(
                    runtime, frame_step.gravity, error)) {
                std::cerr << "preserve authored rigid gravity failed: "
                          << error << '\n';
                return 1;
            }
            if (!step_runtime(runtime, frame_step, error)) {
                std::cerr << "Headless frame " << frame + 1
                          << " failed: " << error << "; peak contacts="
                          << runtime.frame.statistics.peakContactCount << '\n';
                return 1;
            }
            if (fixed_collection &&
                !runtime.fixed_collector.collect(
                    runtime.simulation, runtime.scene, runtime.instance,
                    runtime.frame, error)) {
                std::cerr << "collect fixed contacts failed: " << error << '\n';
                return 1;
            }
            if (runtime.instance.fluid != PM_INVALID_INDEX &&
                frame + 1 < options.frames &&
                !runtime.renderer.advance_visuals(runtime.render_frame, error)) {
                std::cerr << "Visual update failed: " << error << '\n';
                return 1;
            }
            if (options.trace_fluid_escapes &&
                !trace_fluid_escapes(runtime.frame, floor_index, frame + 1,
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
                      << runtime.fixed_collector.attached_count()
                      << " generated_constraints="
                      << runtime.fixed_collector.generated_constraint_count()
                      << '\n';
        }
        if (runtime.instance.fluid != PM_INVALID_INDEX) {
            std::cout << "Fluid particles="
                      << runtime.frame.fluid_particles.size()
                      << " peak="
                      << runtime.frame.statistics.peakParticleCount
                      << " capacity_overflows="
                      << runtime.frame.statistics.overflowCount << '\n';
            if (!runtime.frame.fluid_particles.empty()) {
                const auto first = runtime.frame.fluid_particles.front().position;
                Vec3 low{first[0], first[1], first[2]};
                Vec3 high = low;
                for (const auto &particle : runtime.frame.fluid_particles) {
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
        if (runtime.instance.smoke != PM_INVALID_INDEX) {
            std::cout << "Smoke particles="
                      << runtime.frame.smoke_particles.size() << '\n';
        }
        if (!options.physics_capture_output.empty()) {
            if (!write_physics_debug_capture(
                    runtime.frame, options.physics_capture_output, error)) {
                std::cerr << "Physics capture failed: " << error << '\n';
                return 1;
            }
            std::cout << "Physics capture "
                      << options.physics_capture_output << '\n';
        }
        RendererTimings headless_render_timings{};
        if (!runtime.renderer.render(runtime.render_frame,
                                     input_state.shared.camera.camera(),
                                     pixels, error, &headless_render_timings,
                                     options.cloth_debug
                                         ? FluidRenderMode::wireframe
                                         : options.fluid_particle_view
                                         ? FluidRenderMode::particles
                                         : FluidRenderMode::surface)) {
            std::cerr << "Render failed: " << error << '\n';
            return 1;
        }
        if (options.cloth_debug) {
            if (runtime.instance.cloths.empty()) {
                std::cerr << "Cloth debug overlay needs a cloth resource\n";
                return 1;
            }
            for (std::uint32_t cloth = 0U;
                 cloth < runtime.instance.cloths.size(); ++cloth) {
                if (!draw_cloth_debug_overlay(
                        pixels, runtime.renderer.width(), runtime.renderer.height(),
                        runtime.scene, runtime.frame, cloth,
                        input_state.shared.camera.camera(),
                        {.normals = true, .rigid_contact_forces = true,
                         .fluid_contact_forces = true, .wireframe = true,
                         .bonds = true}, error)) {
                    std::cerr << "Cloth debug overlay failed: " << error << '\n';
                    return 1;
                }
            }
            std::cout << "Cloth debug surfaces=" << runtime.instance.cloths.size() << '\n';
            draw_physics_debug_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                runtime.frame, input_state.shared.camera.camera(),
                {.contact_normals = true, .rigid_forces = true,
                 .fluid_forces = true, .velocities = true});
        }
        if (runtime.context == GalleryContext::peg_paint ||
            runtime.context == GalleryContext::cloth_paint) {
            std::uint64_t painted = 0U;
            const std::uint64_t minimum =
                runtime.context == GalleryContext::peg_paint ? 100U :
                options.headless_cloth_tilt_degrees != 0U ? 1U : 0U;
            if (!runtime.renderer.paint_coverage(painted, error) ||
                painted < minimum) {
                std::cerr << "Paint coverage failed: " << error << '\n';
                return 1;
            }
            std::cout << "Painted texels=" << painted << '\n';
        }
        if (runtime.instance.fluid != PM_INVALID_INDEX &&
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
        if (!validate_render(pixels, runtime.renderer.width(),
                             runtime.renderer.height(), error)) {
            std::cerr << "Render validation failed: " << error << '\n';
            return 1;
        }
        if (runtime.context == GalleryContext::cloth_tear) {
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
            if (cloth_pixels < pixels.size() / 50U) {
                std::cerr << "Cloth Tear render lost the sheet\n";
                return 1;
            }
        }
        if (!write_ppm(options.headless_output, pixels, runtime.renderer.width(),
                       runtime.renderer.height())) {
            std::cerr << "Failed to write " << options.headless_output << '\n';
            return 1;
        }
        std::cout << "Rendered " << options.headless_output << '\n';
        return 0;
    }

    const char *window_title =
#ifdef PARALLEL_MATER_SLANG_TEST_BUILD
        "ParallelMater Gallery - Slang Production Test";
#else
        "ParallelMater Gallery";
#endif
    GalleryWindow presentation;
    if (!GalleryWindow::create(1280U, 720U, options.width, options.height,
                               window_title, presentation, error)) {
        std::cerr << error << '\n';
        return 1;
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

    KeyEdges keys;
    bool timing_visible = false;
    GalleryDebugState debug;
    debug.reset(options.fluid_particle_view);
    using FrameClock = std::chrono::steady_clock;
    bool capture_requested = false;
    std::string capture_status;
    bool capture_status_failure = false;
    FrameClock::time_point capture_status_until{};
    auto previous_frame_time = FrameClock::now();
    double physics_accumulator = 0.0;
    std::uint64_t observed_session_revision = session.revision;
    float rigid_interpolation_alpha = 1.0F;
    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        const auto frame_time = FrameClock::now();
        const double frame_delta = std::min(
            k_maximum_frame_delta,
            std::chrono::duration<double>(
                frame_time - previous_frame_time).count());
        previous_frame_time = frame_time;
        glfwPollEvents();
        keys.update(window);

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
                    !session.rebuild(options, GalleryContext::rigid_body,
                                     options.dump_spheres,
                                     options.fluid_particles, error)) {
                    std::cerr << error << '\n';
                    break;
                }
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
                input_state.shared.key(GalleryKey::catalog);
                input_state.count_value_invalid = false;
            }
            if (keys.pressed(KeyAction::enter)) {
                std::uint32_t requested = 0U;
                const GalleryEntry &entry = gallery_entry(runtime.context);
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
                    if (session.rebuild(
                            options, runtime.context,
                            fluid_dialog || input_state.brick_dialog
                                ? dump_spheres : requested,
                            fluid_dialog ? requested : fluid_particles,
                            error)) {
                        input_state.shared.parameters_visible = false;
                        input_state.count_value_invalid = false;
                        if (input_state.brick_dialog) {
                            options.bricks_overridden = true;
                            input_state.shared.camera.set_preset(
                                camera_preset_for(runtime.context,
                                                  options.bricks));
                        }
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
                    const GalleryContext context_selection =
                        gallery_entries[selected].context;
                    const bool context_changed =
                        context_selection != runtime.context;
                    if (!context_changed ||
                        session.rebuild(options, context_selection, dump_spheres,
                                        fluid_particles, error)) {
                        if (context_changed) {
                            debug.reset(options.fluid_particle_view);
                            input_state.shared.current = selected;
                            input_state.shared.camera.set_preset(
                                camera_preset_for(runtime.context, options.bricks));
                        }
                        input_state.shared.catalog_visible = false;
                    } else {
                        std::cerr << "Scene switch failed: " << error << '\n';
                    }
                }
            } else if (gallery_entry(runtime.context).count_kind !=
                           GalleryCountKind::none &&
                       keys.pressed(KeyAction::particle_count)) {
                input_state.shared.key(GalleryKey::parameters);
                input_state.brick_dialog = gallery_entry(runtime.context).count_kind ==
                    GalleryCountKind::brick_scene;
                input_state.brick_field = 0U;
                input_state.brick_values = {std::to_string(options.bricks.brick_count),
                    std::to_string(options.bricks.brick_scale),
                    std::to_string(options.bricks.wall_planes)};
                input_state.count_value = std::to_string(
                    gallery_entry(runtime.context).count_kind ==
                            GalleryCountKind::fluid_particles
                        ? fluid_particles
                        : parallel_mater::gallery::dump_payload_count(runtime.scene));
                input_state.replace_count_value = true;
                input_state.count_value_invalid = false;
            }
            if (!input_state.shared.catalog_visible &&
                keys.pressed(KeyAction::reset)) {
                if (!session.rebuild(options, runtime.context, dump_spheres,
                                     fluid_particles, error)) {
                    std::cerr << "Scene reset failed: " << error << '\n';
                }
            }
            if (!input_state.shared.catalog_visible &&
                keys.pressed(KeyAction::action)) {
                if (runtime.context == GalleryContext::dump) runtime.dump_bed.toggle();
                else if (toggles_constraint(gallery_entry(runtime.context).controls) &&
                         !toggle_constraints(runtime)) break;
            }
            if (keys.pressed(KeyAction::timing)) {
                timing_visible = !timing_visible;
            }
            const bool smoke_debug = is_smoke_context(runtime.context);
            if (keys.pressed(KeyAction::primary_debug)) {
                if (smoke_debug)
                    debug.toggle_smoke(SmokeDebugMode::density_temperature);
                else
                    debug.toggle_primary(runtime.context);
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

        if (session.revision != observed_session_revision) {
            observed_session_revision = session.revision;
            physics_accumulator = 0.0;
            rigid_interpolation_alpha = 1.0F;
        } else if (input_state.shared.parameters_visible ||
                   input_state.shared.catalog_visible) {
            physics_accumulator = 0.0;
            rigid_interpolation_alpha = 1.0F;
        } else {
            physics_accumulator += frame_delta;
            const GalleryEntry &entry = gallery_entry(runtime.context);
            const DirectionalInput directional = input_state.shared.catalog_visible
                ? DirectionalInput{}
                : directional_input(
                      window, entry.controls != GalleryControlPolicy::motor_drive);
            // An authored force controller owns the arrow keys. It must not
            // simultaneously tilt gravity or drive an unrelated kinematic body.
            const DirectionalInput steering = runtime.arrow_forces.active()
                ? DirectionalInput{} : directional;
            std::uint32_t physics_steps = 0U;
            parallel_mater::gallery::PhysicsFrameBudget physics_budget(k_maximum_catch_up_steps);
            bool step_failed = false;
            while (physics_accumulator >= k_timestep &&
                   physics_budget.can_step()) {
                const auto physics_started = FrameClock::now();
                if (entry.controls == GalleryControlPolicy::motor_drive &&
                    (!drive_motors(runtime, steering) ||
                     !steer_motor_suspension(runtime, steering.x))) {
                    step_failed = true;
                    break;
                }
                if (runtime.context == GalleryContext::dump &&
                    !runtime.dump_bed.advance(runtime.simulation, runtime.scene,
                        runtime.instance, k_timestep, error)) {
                    std::cerr << "tilt dump bucket failed: " << error << '\n';
                    step_failed = true;
                    break;
                }
                if (runtime.kinematic_index <
                    runtime.instance.rigid_bodies.size()) {
                    runtime.kinematic_target.position.x +=
                        steering.x * k_kinematic_speed * k_timestep;
                    runtime.kinematic_target.position.z +=
                        steering.z * k_kinematic_speed * k_timestep;
                    if (!runtime.simulation.set_kinematic_target(
                            runtime.instance.rigid_bodies[
                                runtime.kinematic_index],
                            runtime.kinematic_target, error)) {
                        std::cerr << "move kinematic body failed: "
                                  << error << '\n';
                        step_failed = true;
                        break;
                    }
                }
                StepOptions interactive_step = step_options;
                interactive_step.substeps = scene_substeps(runtime.context);
                interactive_step.gravity = {
                    0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F};
                if (entry.controls == GalleryControlPolicy::cloth_gravity) {
                    cloth_gravity = steer_gravity(
                        cloth_gravity, input_state.shared.camera.camera(),
                        steering.x, -steering.z,
                        k_gravity * runtime.scene.gravity_scale,
                        k_cloth_gravity_tilt_degrees, k_timestep);
                    interactive_step.gravity = cloth_gravity;
                } else if (entry.controls ==
                           GalleryControlPolicy::collector_gravity) {
                    interactive_step.collect_rigid_contacts = true;
                    interactive_step.gravity = collector_gravity_for(
                        steering, runtime.scene.gravity_scale,
                        input_state.shared.camera.camera());
                    const Vec3 loose_gravity{
                        0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F};
                    if (!runtime.fixed_collector.apply_loose_gravity(
                            runtime.simulation, runtime.scene,
                            runtime.instance, loose_gravity,
                            interactive_step.gravity, error)) {
                        std::cerr << "apply loose fixed-scene gravity failed: "
                                  << error << '\n';
                        step_failed = true;
                        break;
                    }
                } else if (uses_rigid_gravity(entry.controls)) {
                    interactive_step.gravity = gravity_for(
                        steering, runtime.scene.gravity_scale,
                        input_state.shared.camera.camera());
                } else if (entry.controls ==
                           GalleryControlPolicy::peg_gravity) {
                    const float right = steering.x + (!input_state.shared.catalog_visible ?
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_D) == GLFW_PRESS) -
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_A) == GLFW_PRESS) : 0.0F);
                    const float forward = -steering.z + (!input_state.shared.catalog_visible ?
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_W) == GLFW_PRESS) -
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_S) == GLFW_PRESS) : 0.0F);
                    peg_gravity = steer_gravity(
                        peg_gravity, input_state.shared.camera.camera(), right,
                        forward, k_gravity * runtime.scene.gravity_scale,
                        peg_paint_gravity_tilt_degrees, k_timestep);
                    interactive_step.gravity = peg_gravity;
                }
                if (!apply_gravity_tilt_overrides(
                        runtime, interactive_step.gravity, error)) {
                    std::cerr << "preserve authored rigid gravity failed: "
                              << error << '\n';
                    step_failed = true;
                    break;
                }
                interactive_step.collect_kernel_timings = timing_visible;
                interactive_step.collect_rigid_contacts =
                    interactive_step.collect_rigid_contacts ||
                    debug.rigid_contacts || timing_visible;
                interactive_step.collect_fluid_contacts =
                    debug.rigid_contacts || timing_visible;
                if (!runtime.arrow_forces.apply(
                        runtime.simulation,
                        input_state.shared.camera.camera(), directional.x,
                        -directional.z, error)) {
                    std::cerr << "apply screen-space arrow force failed: "
                              << error << '\n';
                    step_failed = true;
                    break;
                }
                if (!step_runtime(runtime, interactive_step, error)) {
                    std::cerr << "step gallery failed: " << error << '\n';
                    step_failed = true;
                    break;
                }
                if (entry.controls ==
                        GalleryControlPolicy::collector_gravity &&
                    !runtime.fixed_collector.collect(
                        runtime.simulation, runtime.scene, runtime.instance,
                        runtime.frame, error)) {
                    std::cerr << "collect fixed contacts failed: "
                              << error << '\n';
                    step_failed = true;
                    break;
                }
                physics_accumulator -= k_timestep;
                ++physics_steps;
                physics_budget.record_step(std::chrono::duration<double>(
                    FrameClock::now() - physics_started).count());
            }
            if (step_failed) break;
            if (!physics_budget.can_step() &&
                physics_accumulator >= k_timestep) {
                physics_accumulator = std::fmod(
                    physics_accumulator, static_cast<double>(k_timestep));
            }
            rigid_interpolation_alpha = static_cast<float>(std::clamp(
                physics_accumulator / static_cast<double>(k_timestep),
                0.0, 1.0));
            if (physics_steps != 0U && timing_visible) {
                // Per-stage timestamp queries will replace this compatibility
                // presentation as command-query coverage is expanded.
                timings = {};
                statistics = {};
                statistics.frame_index = runtime.frame.statistics.stepCount;
                statistics.particle_count = static_cast<std::uint32_t>(
                    runtime.frame.fluid_particles.size());
                statistics.smoke_particle_count = static_cast<std::uint32_t>(
                    runtime.frame.smoke_particles.size());
                statistics.rigid_body_count = static_cast<std::uint32_t>(
                    runtime.frame.rigid_states.size());
                statistics.contact_count = static_cast<std::uint32_t>(
                    runtime.frame.contacts.size());
                statistics.rigid_contact_maximum_passes =
                    static_cast<std::uint32_t>(
                        runtime.frame.statistics.poseIterations +
                        runtime.frame.statistics.impactIterations);
                statistics.contact_overflow_count = static_cast<std::uint32_t>(
                    runtime.frame.statistics.overflowCount);
            }
        }

        const bool capture_frame_ready = runtime.frame.revision != 0U ||
            !runtime.frame.rigid_states.empty() ||
            !runtime.frame.fluid_particles.empty() ||
            !runtime.frame.cloth_vertices.empty() ||
            !runtime.frame.soft_body_nodes.empty() ||
            !runtime.frame.rope_nodes.empty() ||
            !runtime.frame.smoke_particles.empty();
        if (capture_requested && capture_frame_ready) {
            std::filesystem::path capture_path;
            if (save_physics_debug_capture(runtime.frame, capture_path,
                                           error)) {
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

        const Camera current_camera = input_state.shared.camera.camera();
        if (!runtime.renderer.render(runtime.render_frame,
                                     current_camera, pixels, error,
                                     timing_visible ? &renderer_timings : nullptr,
                                     debug.fluid_render_mode(runtime.context),
                                     debug.smoke_mode == SmokeDebugMode::none,
                                     rigid_interpolation_alpha)) {
            std::cerr << "Render failed: " << error << '\n';
            break;
        }
        if (is_smoke_context(runtime.context) &&
            debug.smoke_mode != SmokeDebugMode::none) {
            if (!draw_smoke_debug_overlay(
                    pixels, runtime.renderer.width(), runtime.renderer.height(),
                    runtime.frame, current_camera, debug.smoke_mode, error)) {
                std::cerr << "Smoke grid debug overlay failed: " << error << '\n';
                break;
            }
        }
        if (is_cloth_context(runtime.context) &&
            (debug.normals || debug.structure || debug.cloth_bonds)) {
            bool cloth_debug_ok = true;
            for (std::uint32_t cloth = 0U;
                 cloth < runtime.instance.cloths.size(); ++cloth) {
                if (!draw_cloth_debug_overlay(
                        pixels, runtime.renderer.width(),
                        runtime.renderer.height(), runtime.scene, runtime.frame,
                        cloth, current_camera,
                        {.normals = debug.normals,
                         .wireframe = debug.structure,
                         .bonds = debug.cloth_bonds}, error)) {
                    cloth_debug_ok = false;
                    break;
                }
            }
            if (!cloth_debug_ok) {
                std::cerr << "Cloth debug overlay failed: " << error << '\n';
                break;
            }
        }
        if (is_soft_body_context(runtime.context) &&
            (debug.normals || debug.structure || debug.cloth_bonds)) {
            bool soft_debug_ok = true;
            for (std::uint32_t body = 0U;
                 body < runtime.instance.soft_bodies.size(); ++body) {
                if (!draw_soft_body_debug_overlay(
                        pixels, runtime.renderer.width(),
                        runtime.renderer.height(), runtime.scene, runtime.frame,
                        body, current_camera,
                        {.normals = debug.normals,
                         .wireframe = debug.cloth_bonds,
                         .bonds = debug.structure}, error)) {
                    soft_debug_ok = false;
                    break;
                }
            }
            if (!soft_debug_ok) {
                std::cerr << "Soft-body debug overlay failed: " << error << '\n';
                break;
            }
        }
        if (debug.structure || debug.cloth_bonds) {
            for (std::uint32_t rope = 0U;
                 rope < runtime.instance.ropes.size(); ++rope) {
                if (!draw_rope_debug_overlay(
                        pixels, runtime.renderer.width(),
                        runtime.renderer.height(), runtime.frame, rope,
                        current_camera, error)) {
                    std::cerr << "Rope debug failed: " << error << '\n';return 1;
                }
            }
        }
        if (debug.vectors_visible()) {
            draw_physics_debug_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                runtime.frame, current_camera,
                {.contact_normals = debug.normals,
                 .rigid_forces = debug.rigid_forces,
                 .fluid_forces = debug.fluid_forces,
                 .velocities = debug.velocities});
        }
        if (debug.rigid_contacts &&
            !draw_rigid_contact_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                runtime.frame.contacts, current_camera, error)) {
            std::cerr << "Rigid contact overlay failed: " << error << '\n';
            break;
        }
        if (timing_visible) {
            if (runtime.context == GalleryContext::smoke)
                draw_smoke_timing_overlay(
                    pixels, runtime.renderer.width(), runtime.renderer.height(),
                    timings, renderer_timings, statistics,
                    runtime.scene.smoke_options.capacity);
            else if (is_fluid_context(runtime.context))
                draw_fluid_timing_overlay(
                    pixels, runtime.renderer.width(), runtime.renderer.height(),
                    timings, renderer_timings, statistics, fluid_particles);
            else if (is_soft_body_context(runtime.context))
                draw_soft_body_timing_overlay(
                    pixels, runtime.renderer.width(),
                    runtime.renderer.height(), timings);
            else if (is_cloth_context(runtime.context))
                draw_cloth_timing_overlay(pixels, runtime.renderer.width(),
                                          runtime.renderer.height(), timings);
            else
                draw_timing_overlay(pixels, runtime.renderer.width(),
                                    runtime.renderer.height(), timings, statistics);
        }
        if (input_state.shared.catalog_visible) {
            draw_context_overlay(pixels, runtime.renderer.width(),
                                 runtime.renderer.height(),
                                 gallery_entries[input_state.shared.selected].context);
        }
        if (input_state.shared.parameters_visible) {
            if (input_state.brick_dialog)
                draw_brick_settings_overlay(
                    pixels, runtime.renderer.width(), runtime.renderer.height(),
                    input_state.brick_values, input_state.brick_field,
                    input_state.count_value_invalid);
            else
                draw_count_overlay(
                    pixels, runtime.renderer.width(), runtime.renderer.height(),
                    runtime.context, input_state.count_value,
                    input_state.count_value_invalid);
        }
        if (!capture_status.empty() && FrameClock::now() < capture_status_until)
            draw_status_overlay(pixels, runtime.renderer.width(),
                                runtime.renderer.height(), capture_status,
                                capture_status_failure);

        if (!presentation.present(pixels, error)) {
            std::cerr << "Presentation failed: " << error << '\n';
            return 1;
        }
    }
    return 0;
}
