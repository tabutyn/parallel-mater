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
using parallel_mater::Status;
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

struct GalleryRuntime {
    GalleryContext context{GalleryContext::rigid_body};
    SceneDefinition scene{};
    CommandScene simulation{};
    CommandSceneInstance instance{};
    CommandSceneFrame frame{};
    RenderSceneFrame render_frame{};
    RendererHandle renderer{};
    FixedContactCollector fixed_collector{};
    parallel_mater::gallery::ArrowForces arrow_forces{};
    parallel_mater::gallery::DumpTruckBed dump_bed{};
    std::vector<SuspensionSteeringJoint> suspension_steering_joints{};
    float steering_angle{};
    std::vector<std::uint32_t> gravity_tilt_bodies{};
    std::size_t kinematic_index{std::numeric_limits<std::size_t>::max()};
    RigidBodyState kinematic_target{};
};

[[nodiscard]] RigidBodyState rigid_state(const PmRigidState &source) noexcept {
    return {{source.position[0], source.position[1], source.position[2]},
            {source.orientation[0], source.orientation[1],
             source.orientation[2], source.orientation[3]},
            {source.linearVelocity[0], source.linearVelocity[1],
             source.linearVelocity[2]},
            {source.angularVelocity[0], source.angularVelocity[1],
             source.angularVelocity[2]}};
}

[[nodiscard]] bool apply_acceleration(
    GalleryRuntime &runtime, const std::vector<std::uint32_t> &bodies,
    Vec3 acceleration, std::string &error) {
    if (bodies.empty()) return true;
    std::vector<PmRigidInput> inputs;
    try {
        inputs.reserve(bodies.size());
        for (std::uint32_t body : bodies) {
            if (body >= runtime.scene.rigid_bodies.size()) {
                error = "rigid acceleration body index is invalid";
                return false;
            }
            PmRigidInput input{};
            input.bodyIndex = body;
            const float mass = runtime.scene.rigid_bodies[body].options.mass;
            input.force[0] = acceleration.x * mass;
            input.force[1] = acceleration.y * mass;
            input.force[2] = acceleration.z * mass;
            inputs.push_back(input);
        }
    } catch (const std::bad_alloc &) {
        error = "rigid acceleration allocation failed";
        return false;
    }
    return runtime.simulation.apply_rigid_inputs(inputs, error);
}

[[nodiscard]] bool step_runtime(GalleryRuntime &runtime,
                                const StepOptions &options,
                                std::string &error) {
    if (!runtime.simulation.step(options.timestep, options.substeps,
                                 options.gravity, runtime.frame, error,
                                 options.collect_rigid_contacts ||
                                     options.collect_fluid_contacts))
        return false;
    return parallel_mater::gallery::make_render_scene_frame(
        runtime.scene, runtime.frame, runtime.render_frame, error);
}

[[nodiscard]] const GalleryEntry *entry_for_option(
    std::string_view option) noexcept {
    const auto found = std::find_if(
        gallery_entries.begin(), gallery_entries.end(),
        [&](const GalleryEntry &entry) {
            return !entry.command_line_option.empty() &&
                   entry.command_line_option == option;
        });
    return found == gallery_entries.end() ? nullptr : &*found;
}

[[nodiscard]] parallel_mater::Quaternion rotation_z(float radians) {
    return {0.0F, 0.0F, std::sin(radians * 0.5F),
            std::cos(radians * 0.5F)};
}

[[nodiscard]] parallel_mater::Quaternion conjugate(
    parallel_mater::Quaternion value) noexcept {
    return {-value.x, -value.y, -value.z, value.w};
}

[[nodiscard]] parallel_mater::Quaternion multiply(
    parallel_mater::Quaternion left,
    parallel_mater::Quaternion right) noexcept {
    return {
        left.w * right.x + left.x * right.w + left.y * right.z -
            left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w +
            left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x +
            left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y -
            left.z * right.z};
}

[[nodiscard]] parallel_mater::Vec3 rotate(
    parallel_mater::Quaternion rotation,
    parallel_mater::Vec3 point) noexcept {
    const parallel_mater::Quaternion vector{point.x, point.y, point.z, 0.0F};
    const parallel_mater::Quaternion result =
        multiply(multiply(rotation, vector), conjugate(rotation));
    return {result.x, result.y, result.z};
}

[[nodiscard]] parallel_mater::Vec3 subtract(
    parallel_mater::Vec3 left, parallel_mater::Vec3 right) noexcept {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

[[nodiscard]] parallel_mater::Vec3 midpoint(
    parallel_mater::Vec3 left, parallel_mater::Vec3 right) noexcept {
    return {(left.x + right.x) * 0.5F, (left.y + right.y) * 0.5F,
            (left.z + right.z) * 0.5F};
}

struct FluidEscapeTrace {
    int first_frame{-1};
    int first_local_frame{-1};
    std::uint32_t peak_below{};
    std::uint32_t final_below{};
    std::uint32_t peak_below_local{};
    std::uint32_t final_below_local{};
    float lowest_y{std::numeric_limits<float>::max()};
    std::vector<std::array<parallel_mater::Vec3, 64>> recent_positions{};
    std::vector<std::uint8_t> recent_counts{};
};

[[nodiscard]] bool trace_fluid_escapes(const CommandSceneFrame &simulation,
                                       const StaticTriangleSurface &floor,
                                       int frame, FluidEscapeTrace &trace) {
    std::vector<parallel_mater::Vec3> positions;
    positions.reserve(simulation.fluid_particles.size());
    for (const auto &particle : simulation.fluid_particles)
        positions.push_back({particle.position[0], particle.position[1],
                             particle.position[2]});
    const parallel_mater::Vec3 mesh_minimum = floor.minimum();
    const parallel_mater::Vec3 mesh_maximum = floor.maximum();
    const float threshold = mesh_minimum.y;
    std::uint32_t below = 0U;
    std::uint32_t below_local = 0U;
    std::size_t first = positions.size();
    std::size_t first_local = positions.size();
    float first_local_surface = 0.0F;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        trace.lowest_y = std::min(trace.lowest_y, positions[i].y);
        const auto surface = floor.height(
            positions[i].x, positions[i].z, SurfaceSelection::lowest);
        if (surface && positions[i].y < *surface - 0.001F) {
            ++below_local;
            if (first_local == positions.size()) {
                first_local = i;
                first_local_surface = *surface;
            }
        }
        if (positions[i].y < threshold) {
            ++below;
            if (first == positions.size()) first = i;
        }
    }
    trace.peak_below = std::max(trace.peak_below, below);
    trace.final_below = below;
    trace.peak_below_local = std::max(trace.peak_below_local, below_local);
    trace.final_below_local = below_local;
    std::vector<std::uint32_t> stable_ids;
    if ((trace.first_frame < 0 || trace.first_local_frame < 0) &&
        !positions.empty()) {
        stable_ids.reserve(simulation.fluid_particles.size());
        for (const auto &particle : simulation.fluid_particles)
            stable_ids.push_back(particle.stableId);
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const std::uint32_t id = stable_ids[i];
            if (id >= trace.recent_positions.size()) {
                trace.recent_positions.resize(static_cast<std::size_t>(id) + 1U);
                trace.recent_counts.resize(static_cast<std::size_t>(id) + 1U);
            }
            trace.recent_positions[id][frame % 64] = positions[i];
            trace.recent_counts[id] = std::min<std::uint8_t>(
                64U, static_cast<std::uint8_t>(trace.recent_counts[id] + 1U));
        }
    }
    if (below_local != 0U && trace.first_local_frame < 0) {
        trace.first_local_frame = frame;
        const auto p = positions[first_local];
        std::cout << "First local floor penetration frame=" << frame
                  << " id=" << stable_ids[first_local] << " position=("
                  << p.x << ',' << p.y << ',' << p.z << ") floor_y="
                  << first_local_surface << " below_local=" << below_local
                  << '\n';
    }
    if (below != 0U && trace.first_frame < 0) {
        trace.first_frame = frame;
        const std::uint32_t stable_id = stable_ids[first];
        const auto p = positions[first];
        const bool within_xz_bounds = p.x >= mesh_minimum.x &&
            p.x <= mesh_maximum.x && p.z >= mesh_minimum.z &&
            p.z <= mesh_maximum.z;
        std::cout << "First fluid escape frame=" << frame
                  << " id=" << stable_id << " position=(" << p.x << ','
                  << p.y << ',' << p.z << ") below=" << below
                  << " within_mesh_xz_bounds=" << within_xz_bounds << '\n';
        const int history_count = trace.recent_counts[stable_id];
        for (int sample_frame = frame - history_count + 1;
             sample_frame <= frame; ++sample_frame) {
            const auto sample =
                trace.recent_positions[stable_id][sample_frame % 64];
            float normal_y = 0.0F;
            const auto surface = floor.height(
                sample.x, sample.z, SurfaceSelection::highest, &normal_y);
            std::cout << "  particle " << stable_id << " frame=" << sample_frame
                      << " y=" << sample.y << " surface_y=";
            if (surface) std::cout << *surface << " normal_y=" << normal_y;
            else std::cout << "none";
            std::cout << " x=" << sample.x << " z=" << sample.z << '\n';
        }
    }
    return true;
}

[[nodiscard]] DirectionalInput directional_input(
    GLFWwindow *window, bool normalize_diagonal = true) {
    DirectionalInput input{
        static_cast<float>(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) -
            static_cast<float>(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS),
        static_cast<float>(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) -
            static_cast<float>(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)};
    const float length = std::sqrt(input.x * input.x + input.z * input.z);
    if (normalize_diagonal && length > 1.0F) {
        input.x /= length;
        input.z /= length;
    }
    return input;
}

[[nodiscard]] parallel_mater::Vec3 gravity_for(
    DirectionalInput input, float gravity_scale,
    parallel_mater::gallery::Camera camera) {
    return screen_space_gravity(camera, input.x, -input.z,
                                k_gravity * gravity_scale,
                                k_rigid_gravity_tilt_degrees);
}

[[nodiscard]] bool apply_gravity_tilt_overrides(
    GalleryRuntime &runtime, Vec3 world_gravity,
    std::string &error) noexcept {
    if (runtime.gravity_tilt_bodies.empty()) return true;
    const Vec3 vertical_gravity{
        0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F};
    const Vec3 compensation{
        vertical_gravity.x - world_gravity.x,
        vertical_gravity.y - world_gravity.y,
        vertical_gravity.z - world_gravity.z};
    if (compensation.x == 0.0F && compensation.y == 0.0F &&
        compensation.z == 0.0F) return true;
    return apply_acceleration(runtime, runtime.gravity_tilt_bodies,
                              compensation, error);
}

[[nodiscard]] parallel_mater::Vec3 collector_gravity_for(
    DirectionalInput input, float gravity_scale,
    parallel_mater::gallery::Camera camera) {
    return screen_space_gravity(camera, input.x, -input.z,
                                k_gravity * gravity_scale,
                                k_collector_gravity_tilt_degrees);
}

[[nodiscard]] bool parse_positive(std::string_view value, int &output) {
    int parsed = 0;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed <= 0 || parsed > 100'000) {
        return false;
    }
    output = parsed;
    return true;
}

[[nodiscard]] bool parse_count(std::string_view value, std::uint32_t minimum,
                               std::uint32_t maximum, std::uint32_t &output) {
    std::uint32_t parsed = 0U;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed < minimum || parsed > maximum) {
        return false;
    }
    output = parsed;
    return true;
}

[[nodiscard]] bool parse_scale(std::string_view value, float &output) {
    float parsed = 0.0F;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        !std::isfinite(parsed) || parsed < brick_minimum_scale ||
        parsed > brick_maximum_scale) return false;
    output = parsed;
    return true;
}

[[nodiscard]] bool parse_options(int argc, char **argv, Options &output) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--scene" && index + 1 < argc) {
            output.scene = argv[++index];
        } else if (argument == "--headless" && index + 1 < argc) {
            output.headless_output = argv[++index];
        } else if (argument == "--frames" && index + 1 < argc) {
            if (!parse_positive(argv[++index], output.frames)) {
                return false;
            }
        } else if (argument == "--contact-passes" && index + 1 < argc) {
            if (!parse_count(argv[++index], 0U, 64U, output.contact_pass_limit)) return false;
        } else if (argument == "--dump-spheres" && index + 1 < argc) {
            const GalleryEntry &entry = gallery_entry(GalleryContext::dump);
            if (!parse_count(argv[++index], entry.minimum_count,
                             entry.maximum_count, output.dump_spheres)) {
                return false;
            }
            output.initial_context = GalleryContext::dump;
        } else if (argument == "--brick-count" && index + 1 < argc) {
            if (!parse_count(argv[++index], brick_minimum_count,
                             brick_maximum_count, output.bricks.brick_count)) return false;
            output.initial_context = GalleryContext::rigid_body;
            output.bricks_overridden = true;
        } else if (argument == "--brick-scale" && index + 1 < argc) {
            if (!parse_scale(argv[++index], output.bricks.brick_scale)) return false;
            output.initial_context = GalleryContext::rigid_body;
            output.bricks_overridden = true;
        } else if (argument == "--brick-planes" && index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, brick_maximum_planes,
                             output.bricks.wall_planes)) return false;
            output.initial_context = GalleryContext::rigid_body;
            output.bricks_overridden = true;
        } else if (argument == "--profiles-file" && index + 1 < argc) {
            output.profiles_file = argv[++index];
            output.profiles_file_overridden = true;
        } else if (argument == "--calibrate") {
            output.calibrate = true;
            output.initial_context = GalleryContext::rigid_body;
        } else if (argument == "--print-brick-profile") {
            output.print_brick_profile = true;
        } else if (argument == "--fluid-particles" && index + 1 < argc) {
            const GalleryEntry &entry = gallery_entry(GalleryContext::fluid);
            if (!parse_count(argv[++index], entry.minimum_count,
                             entry.maximum_count,
                             output.fluid_particles)) return false;
            if (!is_fluid_context(output.initial_context))
                output.initial_context = GalleryContext::fluid;
        } else if (const GalleryEntry *entry = entry_for_option(argument)) {
            output.initial_context = entry->context;
        } else if (argument == "--cloth-tilt-degrees" && index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 45U,
                             output.headless_cloth_tilt_degrees)) return false;
        } else if (argument == "--gravity-tilt-degrees" && index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 80U,
                             output.headless_cloth_tilt_degrees)) return false;
        } else if (argument == "--cloth-tilt-after-frames" && index + 1 < argc) {
            if (!parse_count(argv[++index], 0U, 100000U,
                             output.headless_cloth_tilt_after_frames)) return false;
        } else if (argument == "--cloth-tilt-left") {
            output.headless_cloth_tilt_left = true;
        } else if ((argument == "--constraint-action-after-frames" || argument == "--dump-after-frames") &&
                   index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 100000U,
                             output.headless_constraint_action_after_frames))
                return false;
        } else if (argument == "--motor-forward") {
            output.headless_motor_forward = true;
        } else if (argument == "--motor-right") {
            output.headless_motor_right = true;
        } else if (argument == "--fluid-particle-view") {
            if (!is_fluid_context(output.initial_context))
                output.initial_context = GalleryContext::fluid;
            output.fluid_particle_view = true;
        } else if (argument == "--trace-fluid-escapes") {
            if (!is_fluid_context(output.initial_context))
                output.initial_context = GalleryContext::fluid;
            output.trace_fluid_escapes = true;
        } else if (argument == "--water-cloth-debug" || argument == "--cloth-debug") {
            if (argument == "--water-cloth-debug")
                output.initial_context = GalleryContext::water_cloth;
            output.cloth_debug = true;
        } else if (argument == "--physics-capture" && index + 1 < argc) {
            output.physics_capture_output = argv[++index];
        } else if (argument == "--help") {
            std::cout << "parallel-mater-gallery [--scene file.glb] "
                         "[--brick-count 1..4096] [--brick-scale 0.5..2] "
                         "[--brick-planes 1..16] [--calibrate] "
                         "[--profiles-file file.json] "
                         "[--print-brick-profile] "
                         "[--dump-spheres N] [";
            bool first = true;
            for (const GalleryEntry &entry : gallery_entries) {
                if (entry.command_line_option.empty()) continue;
                std::cout << (first ? "" : "|") << entry.command_line_option;
                first = false;
            }
            std::cout << "] [--fluid-particles N] [--contact-passes 0..64] "
                         "[--gravity-tilt-degrees 1..80 (headless)] "
                         "[--cloth-tilt-after-frames N (headless)] "
                         "[--cloth-tilt-left (headless)] "
                         "[--constraint-action-after-frames N (headless)] "
                         "[--dump-after-frames N (headless)] "
                         "[--motor-forward (headless)] "
                         "[--motor-right (headless)] "
                         "[--fluid-particle-view] [--trace-fluid-escapes] "
                         "[--cloth-debug | --water-cloth-debug] "
                         "[--physics-capture output.log] "
                         "[--headless output.ppm] "
                         "[--frames N]\n";
            std::exit(0);
        } else {
            return false;
        }
    }
    std::string brick_error;
    return (!output.print_brick_profile || output.headless_output.empty()) &&
        (!output.trace_fluid_escapes || !output.headless_output.empty()) &&
        validate_brick_config(output.bricks, brick_error);
}

[[nodiscard]] bool require(Status status, const char *operation) {
    if (status) {
        return true;
    }
    std::cerr << operation << " failed: "
              << (status.message != nullptr ? status.message : "unknown")
              << '\n';
    return false;
}

[[nodiscard]] bool toggle_constraints(GalleryRuntime &runtime) {
    if (runtime.scene.rigid_constraints.empty() ||
        runtime.scene.rigid_constraints.size() !=
            runtime.instance.rigid_constraints.size()) {
        std::cerr << "Constraint toggle scene needs matching constraints\n";
        return false;
    }

    bool enable = false;
    for (std::uint32_t slot : runtime.instance.rigid_constraints)
        enable = enable || slot >= runtime.frame.constraint_states.size() ||
            runtime.frame.constraint_states[slot].enabled == PM_FALSE;

    std::string command_error;
    if (enable && runtime.context == GalleryContext::constraint_point) {
        // Released arms can be metres away. Restore the authored assembly
        // before enabling its joints instead of injecting a snap-back impulse.
        for (std::size_t index = 0; index < runtime.scene.rigid_bodies.size(); ++index) {
            const auto &body = runtime.scene.rigid_bodies[index];
            if (body.options.motion == MotionType::dynamic &&
                !runtime.simulation.set_rigid_state(
                    runtime.instance.rigid_bodies[index],
                    body.options.initial_state, command_error)) {
                std::cerr << "restore point assembly failed: "
                          << command_error << '\n';
                return false;
            }
        }
    }

    for (std::size_t index = 0U;
         index < runtime.scene.rigid_constraints.size(); ++index) {
        auto &definition = runtime.scene.rigid_constraints[index];
        auto updated = definition;
        updated.options.enabled = enable;
        if (enable && updated.options.type != RigidConstraintType::point) {
            if (definition.body_a >= runtime.frame.rigid_states.size() ||
                definition.body_b >= runtime.frame.rigid_states.size()) {
                std::cerr << "constraint body output is unavailable\n";
                return false;
            }
            const RigidBodyState state_a = rigid_state(
                runtime.frame.rigid_states[definition.body_a]);
            const RigidBodyState state_b = rigid_state(
                runtime.frame.rigid_states[definition.body_b]);
            const Vec3 anchor = midpoint(state_a.position, state_b.position);
            const Quaternion world_orientation = state_a.orientation;
            updated.options.local_anchor_a = rotate(conjugate(state_a.orientation),
                                            subtract(anchor, state_a.position));
            updated.options.local_anchor_b = rotate(conjugate(state_b.orientation),
                                            subtract(anchor, state_b.position));
            updated.options.local_orientation_a =
                multiply(conjugate(state_a.orientation), world_orientation);
            updated.options.local_orientation_b =
                multiply(conjugate(state_b.orientation), world_orientation);
        }
        if (!runtime.simulation.write_constraint(
                runtime.instance.rigid_constraints[index], updated,
                command_error)) {
            std::cerr << (enable ? "enable" : "disable")
                      << " constraint failed: " << command_error << '\n';
            return false;
        }
        definition.options = updated.options;
    }
    return true;
}

[[nodiscard]] bool drive_motors(GalleryRuntime &runtime,
                                DirectionalInput input) {
    const float forward = -input.z;
    const float target_velocity = -forward * k_motor_speed;
    for (std::size_t index = 0U;
         index < runtime.scene.rigid_constraints.size(); ++index) {
        auto &definition = runtime.scene.rigid_constraints[index];
        if (definition.options.type != RigidConstraintType::motor) continue;
        if (definition.options.motor.angular_target_velocity == target_velocity)
            continue;
        auto updated = definition;
        updated.options.motor.angular_target_velocity = target_velocity;
        std::string error;
        if (!runtime.simulation.write_constraint(
                runtime.instance.rigid_constraints[index], updated, error)) {
            std::cerr << "drive motor constraint failed: " << error << '\n';
            return false;
        }
        definition.options = updated.options;
    }
    return true;
}

[[nodiscard]] bool steer_motor_suspension(GalleryRuntime &runtime,
                                           float right_input) {
    const float target = right_input * k_motor_steering_angle;
    const float maximum_step = k_motor_steering_speed * k_timestep;
    const float previous = runtime.steering_angle;
    runtime.steering_angle += std::clamp(
        target - runtime.steering_angle, -maximum_step, maximum_step);
    if (runtime.steering_angle == previous) return true;
    const auto angles = parallel_mater::gallery::axle_steering_angles(
        runtime.steering_angle, 1.0F);
    for (const auto &steering : runtime.suspension_steering_joints) {
        auto &definition =
            runtime.scene.rigid_constraints[steering.constraint_index];
        auto updated = definition;
        const float angle = steering.front ? angles.front : angles.rear;
        updated.options.local_orientation_a = multiply(
            steering.authored_local_orientation_a, rotation_z(angle));
        std::string error;
        if (!runtime.simulation.write_constraint(
                runtime.instance.rigid_constraints[
                    steering.constraint_index], updated, error)) {
            std::cerr << "steer suspension constraint failed: "
                      << error << '\n';
            return false;
        }
        definition.options = updated.options;
    }
    return true;
}

[[nodiscard]] parallel_mater::Vec3 initial_scene_gravity(
    GalleryContext context, float scale) {
    (void)context;
    return {0.0F, -k_gravity * scale, 0.0F};
}

void mouse_button(GLFWwindow *window, int button, int action, int modifiers) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (action == GLFW_RELEASE) {
        input->shared.camera.end_drag();
        return;
    }
    if (action != GLFW_PRESS || input->shared.parameters_visible ||
        input->shared.catalog_visible) return;
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    input->shared.camera.begin_drag((modifiers & GLFW_MOD_SHIFT) != 0
                                 ? CameraDragMode::pan
                                 : CameraDragMode::orbit,
                             x, y);
}

void cursor_position(GLFWwindow *window, double x, double y) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    int height = 0;
    glfwGetWindowSize(window, nullptr, &height);
    input->shared.camera.move_cursor(
        x, y, height,
        !input->shared.parameters_visible && !input->shared.catalog_visible);
}

void scroll(GLFWwindow *window, double, double offset) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (input->shared.parameters_visible || input->shared.catalog_visible) {
        return;
    }
    input->shared.camera.zoom(offset);
}

void character_input(GLFWwindow *window, unsigned int codepoint) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    const bool digit = codepoint >= '0' && codepoint <= '9';
    const bool scale_point = input->brick_dialog && input->brick_field == 1U &&
        codepoint == '.';
    if (!input->shared.parameters_visible || (!digit && !scale_point)) {
        return;
    }
    if (input->replace_count_value) {
        if (input->brick_dialog) input->brick_values[input->brick_field].clear();
        else input->count_value.clear();
        input->replace_count_value = false;
    }
    std::string &value = input->brick_dialog
        ? input->brick_values[input->brick_field] : input->count_value;
    if (value.size() < 6U && (!scale_point || value.find('.') == std::string::npos)) {
        value.push_back(static_cast<char>(codepoint));
        input->count_value_invalid = false;
    }
}

[[nodiscard]] bool write_ppm(const std::filesystem::path &path,
                             const std::vector<std::uint32_t> &pixels,
                             std::uint32_t width, std::uint32_t height) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output << "P6\n" << width << ' ' << height << "\n255\n";
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(pixels.data());
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t source_row = height - row - 1U;
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::size_t offset =
                (static_cast<std::size_t>(source_row) * width + column) * 4U;
            output.write(reinterpret_cast<const char *>(bytes + offset), 3);
        }
    }
    return static_cast<bool>(output);
}

[[nodiscard]] bool validate_render(
    const std::vector<std::uint32_t> &pixels, std::uint32_t width,
    std::uint32_t height, std::string &error) {
    const std::size_t expected =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixels.size() != expected || pixels.empty()) {
        error = "renderer returned an unexpected pixel count";
        return false;
    }
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(pixels.data());
    unsigned minimum_luminance = 3U * 255U;
    unsigned maximum_luminance = 0U;
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        const unsigned luminance = static_cast<unsigned>(bytes[index * 4U]) +
                                   static_cast<unsigned>(bytes[index * 4U + 1U]) +
                                   static_cast<unsigned>(bytes[index * 4U + 2U]);
        minimum_luminance = std::min(minimum_luminance, luminance);
        maximum_luminance = std::max(maximum_luminance, luminance);
    }
    if (maximum_luminance < 48U || maximum_luminance - minimum_luminance < 96U) {
        error = "renderer produced a black or nearly uniform frame";
        return false;
    }
    return true;
}

[[nodiscard]] bool build_runtime(const Options &options,
                                 GalleryContext context,
                                 std::uint32_t dump_spheres,
                                 std::uint32_t fluid_particles,
                                 GalleryRuntime &output,
                                 std::string &error) {
    GalleryRuntime next{};
    next.context = context;
    const GalleryEntry &entry = gallery_entry(context);
    {
        const std::array scene_paths{
            options.scene,
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH),
            std::filesystem::path(
                PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_FLUID_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_FLUID_RIGID_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_PEGS_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CLOTH_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CLOTH_PAINT_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CLOTH_WATER_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SOFT_BODY_SCENE_PATH),
            std::filesystem::path(
                PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SOFT_BODY_CLOTH_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SOFT_BODY_FLUID_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_ROPE_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_ROPE_FLUID_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_ROPE_SOFT_BODY_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_ROPE_CLOTH_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SMOKE_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SMOKE_WATER_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SMOKE_CLOTH_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH)};
        const std::filesystem::path &scene_path = scene_paths[
            static_cast<std::size_t>(entry.source)];
        if (!parallel_mater::gallery::load_glb_scene(scene_path, next.scene,
                                                      error)) {
            error = "scene load failed: " + error;
            return false;
        }
        if (context == GalleryContext::rigid_body) {
            SceneDefinition generated;
            if (!parallel_mater::gallery::make_brick_scene(
                    next.scene, options.bricks, generated, error)) {
                error = "brick scene generation failed: " + error;
                return false;
            }
            next.scene = std::move(generated);
        }
        if (entry.has_fluid)
            next.scene.fluid_options.capacity = fluid_particles;
    }
    if (context == GalleryContext::dump) {
        if (!parallel_mater::gallery::configure_dump_payload(next.scene, dump_spheres, error)) return false;
        if (!next.dump_bed.initialize(next.scene)) {
            error = "dump truck needs its authored DumpLift joint";
            return false;
        }
    }
    if (entry.controls == GalleryControlPolicy::motor_drive) {
        for (std::size_t spring_index = 0U;
             spring_index < next.scene.rigid_constraints.size();
             ++spring_index) {
            const auto &spring = next.scene.rigid_constraints[spring_index];
            if (spring.options.type != RigidConstraintType::generic_spring)
                continue;
            const auto motor = std::find_if(
                next.scene.rigid_constraints.begin(),
                next.scene.rigid_constraints.end(), [&](const auto &candidate) {
                    return candidate.options.type == RigidConstraintType::motor &&
                           candidate.body_a == spring.body_b;
                });
            if (motor == next.scene.rigid_constraints.end() ||
                (motor->name.find("Front") == std::string::npos &&
                 motor->name.find("Rear") == std::string::npos)) {
                error = spring.name +
                    ": suspension hub needs a named Front or Rear motor";
                return false;
            }
            next.suspension_steering_joints.push_back({
                spring_index, spring.options.local_orientation_a,
                motor->name.find("Front") != std::string::npos});
        }
        if (next.suspension_steering_joints.size() != 4U) {
            error = "vehicle scene needs four steerable suspension hubs";
            return false;
        }
    }
    if (context == GalleryContext::rigid_body) {
        const std::uint64_t free_bytes =
            parallel_mater::gallery::platform_available_device_memory();
        if (free_bytes != 0U &&
            parallel_mater::gallery::estimated_rigid_contact_bytes(
                static_cast<std::uint32_t>(next.scene.rigid_bodies.size())) >
                free_bytes * 7U / 10U) {
            error = "brick scene exceeds the available GPU memory budget";
            return false;
        }
    }
    const std::uint32_t debug_frames =
        context == GalleryContext::constraint_fixed ? 300U : 30U;
    bool created = CommandScene::create(next.scene, debug_frames,
                                        next.simulation, next.instance, error);
    if (created)
        created = next.arrow_forces.initialize(next.scene, next.instance, error);
    if (created) {
        next.gravity_tilt_bodies.reserve(next.scene.rigid_bodies.size());
        for (std::size_t index = 0U;
             index < next.scene.rigid_bodies.size(); ++index) {
            const auto &body = next.scene.rigid_bodies[index];
            if (body.options.motion == MotionType::dynamic &&
                !body.follows_gravity_tilt) {
                next.gravity_tilt_bodies.push_back(
                    next.instance.rigid_bodies[index]);
            }
        }
    }
    if (created && context == GalleryContext::constraint_fixed) {
        created = next.fixed_collector.initialize(
            next.scene, next.instance, error);
    }
    if (!created) {
        if (error.empty()) error = "scene creation failed";
        return false;
    }
    if (context == GalleryContext::rigid_body) {
        const auto dynamic_count = std::count_if(
            next.scene.rigid_bodies.begin(), next.scene.rigid_bodies.end(),
            [](const auto &body) {
                return body.options.motion == MotionType::dynamic;
            });
        std::cout << "Rigid gravity tilt: "
                  << dynamic_count - next.gravity_tilt_bodies.size()
                  << " follow, " << next.gravity_tilt_bodies.size()
                  << " keep vertical\n";
    }
    if (next.arrow_forces.active()) {
        for (const auto &body : next.scene.rigid_bodies)
            if (body.arrow_force > 0.0F)
                std::cout << "Arrow force: " << body.source_name << " "
                          << body.arrow_force
                          << " N (camera-relative ground plane; gravity stays vertical)\n";
    }
    if (!parallel_mater::gallery::create_renderer(
            next.scene, next.instance, options.width, options.height,
            next.renderer, error)) {
        error = "renderer creation failed: " + error;
        return false;
    }
    for (std::size_t index = 0; index < next.scene.rigid_bodies.size(); ++index) {
        if (next.scene.rigid_bodies[index].options.motion ==
            parallel_mater::MotionType::kinematic) {
            next.kinematic_index = index;
            next.kinematic_target =
                next.scene.rigid_bodies[index].options.initial_state;
            break;
        }
    }
    output = std::move(next);
    error.clear();
    return true;
}

struct GallerySession {
    GalleryRuntime runtime{};
    std::uint32_t dump_spheres{k_default_dump_spheres};
    std::uint32_t fluid_particles{k_default_fluid_particles};
    parallel_mater::Vec3 peg_gravity{};
    parallel_mater::Vec3 cloth_gravity{};
    parallel_mater::gallery::GalleryStepTimings timings{};
    parallel_mater::gallery::GalleryStatistics statistics{};
    parallel_mater::gallery::RendererTimings renderer_timings{};
    std::uint64_t revision{};

    [[nodiscard]] bool rebuild(const Options &options, GalleryContext context,
                               std::uint32_t requested_dump_spheres,
                               std::uint32_t requested_fluid_particles,
                               std::string &error) {
        GalleryRuntime replacement;
        if (!build_runtime(options, context, requested_dump_spheres,
                           requested_fluid_particles, replacement, error)) {
            return false;
        }
        runtime = std::move(replacement);
        dump_spheres = requested_dump_spheres;
        fluid_particles = requested_fluid_particles;
        peg_gravity = initial_scene_gravity(context, runtime.scene.gravity_scale);
        cloth_gravity = peg_gravity;
        timings = {};
        statistics = {};
        renderer_timings = {};
        ++revision;
        return true;
    }
};

struct CalibrationResult {
    BrickSceneConfig config{};
    parallel_mater::gallery::CalibrationMetrics metrics{};
    parallel_mater::gallery::HardwareIdentity hardware{};
};

std::filesystem::path device_profiles_path(const Options &options) {
    if (options.profiles_file_overridden) return options.profiles_file;
    if (!options.executable_path.empty()) {
        const auto bundled = options.executable_path.parent_path() /
            "config/device-profiles.json";
        std::error_code error;
        if (std::filesystem::is_regular_file(bundled, error)) return bundled;
    }
    return options.profiles_file;
}

bool launch_calibration_ball(GalleryRuntime &runtime, bool off_center,
                             std::string &error) {
    for (std::size_t index = 0U; index < runtime.scene.rigid_bodies.size(); ++index) {
        if (runtime.scene.rigid_bodies[index].source_name != "Icosphere") continue;
        RigidBodyState state = runtime.scene.rigid_bodies[index].options.initial_state;
        state.position = {off_center ? 1.0F : 0.0F, 1.05F, 3.0F};
        state.linear_velocity = {0.0F, 0.0F, -10.0F};
        state.angular_velocity = {};
        return runtime.simulation.set_rigid_state(
            runtime.instance.rigid_bodies[index], state, error);
    }
    error = "brick calibration scene has no ball";
    return false;
}

bool stable_calibration_scene(GalleryRuntime &runtime, bool quiet,
                              float scale, std::string &error) {
    if (runtime.frame.rigid_states.size() < runtime.instance.rigid_bodies.size()) {
        error = "brick calibration rigid output is incomplete";
        return false;
    }
    unsigned displaced_bricks = 0U;
    for (std::size_t index = 0U; index < runtime.instance.rigid_bodies.size(); ++index) {
        const RigidBodyState state = rigid_state(runtime.frame.rigid_states[index]);
        if (!std::isfinite(state.position.x) || !std::isfinite(state.position.y) ||
            !std::isfinite(state.position.z) ||
            std::fabs(state.position.x) > 1'000.0F ||
            std::fabs(state.position.y) > 1'000.0F ||
            std::fabs(state.position.z) > 1'000.0F) {
            error = "brick calibration produced an unstable body";
            return false;
        }
        const float speed = std::sqrt(
            state.linear_velocity.x * state.linear_velocity.x +
            state.linear_velocity.y * state.linear_velocity.y +
            state.linear_velocity.z * state.linear_velocity.z);
        if (!std::isfinite(speed) || speed > 50.0F) {
            error = "brick calibration produced excessive kinetic energy";
            return false;
        }
        const auto &body = runtime.scene.rigid_bodies[index];
        if (body.source_name != "Layer1" && body.source_name != "Layer2")
            continue;
        const auto initial = body.options.initial_state.position;
        const float displacement = std::hypot(
            std::hypot(state.position.x - initial.x,
                       state.position.y - initial.y),
            state.position.z - initial.z);
        if (quiet && displacement > std::max(0.03F, 0.04F * scale)) {
            error = "brick wall lost quiet support";
            return false;
        }
        displaced_bricks += !quiet && displacement > 0.03F;
    }
    if (!quiet && displaced_bricks == 0U) {
        error = "calibration impact did not reach the brick wall";
        return false;
    }
    return true;
}

void present_calibration_frame(GLFWwindow *window,
                               const std::vector<std::uint32_t> &pixels,
                               std::uint32_t width, std::uint32_t height) {
    int framebuffer_width = 0, framebuffer_height = 0;
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
    glViewport(0, 0, framebuffer_width, framebuffer_height);
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2f(-1.0F, -1.0F);
    glPixelZoom(static_cast<float>(framebuffer_width) / width,
                static_cast<float>(framebuffer_height) / height);
    glDrawPixels(static_cast<int>(width), static_cast<int>(height), GL_RGBA,
                 GL_UNSIGNED_BYTE, pixels.data());
    glfwSwapBuffers(window);
}

bool run_calibration_trial(const Options &base, const BrickSceneConfig &config,
                           GLFWwindow *window, std::uint32_t quiet_frames,
                           std::uint32_t collision_frames, bool off_center,
                           std::vector<parallel_mater::gallery::CalibrationSample> &samples,
                           bool &stable, std::string &error) {
    Options candidate = base;
    candidate.bricks = config;
    GalleryRuntime runtime;
    if (!build_runtime(candidate, GalleryContext::rigid_body,
                       candidate.dump_spheres, candidate.fluid_particles,
                       runtime, error)) return false;
    const StepOptions step{.timestep = k_timestep, .substeps = 4U,
                           .gravity = {0.0F, -k_gravity, 0.0F}};
    for (std::uint32_t frame = 0U; frame < 30U; ++frame) {
        if (!step_runtime(runtime, step, error)) {
            error = "calibration settle failed: " + error;
            return false;
        }
    }
    std::vector<std::uint32_t> pixels;
    CameraController camera;
    camera.set_preset(camera_preset_for(GalleryContext::rigid_body, config));
    auto measure = [&](bool collision) {
        const auto begin = std::chrono::steady_clock::now();
        if (!step_runtime(runtime, step, error)) {
            error = "calibration step failed: " + error;
            return false;
        }
        if (!runtime.renderer.render(runtime.render_frame,
                                     camera.camera(), pixels, error)) return false;
        present_calibration_frame(window, pixels, candidate.width, candidate.height);
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        samples.push_back({milliseconds, collision, true});
        glfwPollEvents();
        return glfwWindowShouldClose(window) == GLFW_FALSE;
    };
    for (std::uint32_t frame = 0U; frame < quiet_frames; ++frame)
        if (!measure(false)) { if (error.empty()) error = "calibration cancelled"; return false; }
    std::string correctness_error;
    stable = stable && stable_calibration_scene(
        runtime, true, config.brick_scale, correctness_error);
    if (!launch_calibration_ball(runtime, off_center, error)) return false;
    for (std::uint32_t frame = 0U; frame < collision_frames; ++frame)
        if (!measure(true)) { if (error.empty()) error = "calibration cancelled"; return false; }
    correctness_error.clear();
    stable = stable && stable_calibration_scene(
        runtime, false, config.brick_scale, correctness_error);
    return true;
}

double measured_seconds(
    const std::vector<parallel_mater::gallery::CalibrationSample> &samples) {
    return std::accumulate(samples.begin(), samples.end(), 0.0,
        [](double total, const auto &sample) {
            return total + sample.milliseconds / 1'000.0;
        });
}

bool calibrate_bricks(Options &options, GLFWwindow *window,
                      CalibrationResult &output, std::string &error) {
    glfwSwapInterval(0);
    const auto &presets = parallel_mater::gallery::brick_calibration_presets();
    std::optional<std::size_t> best;
    for (std::size_t index = 0U; index < presets.size(); ++index) {
        std::vector<parallel_mater::gallery::CalibrationSample> samples;
        bool stable = true;
        std::string trial_error;
        const auto begin = std::chrono::steady_clock::now();
        if (!run_calibration_trial(options, presets[index], window, 60U, 120U,
                                   index % 2U != 0U, samples, stable,
                                   trial_error)) {
            if (trial_error.find("memory budget") != std::string::npos) break;
            glfwSwapInterval(1); error = trial_error; return false;
        }
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        auto metrics = parallel_mater::gallery::summarize_calibration(
            samples, duration, stable, false);
        const double work = measured_seconds(samples);
        metrics.simulation_progress_ratio = work > 0.0
            ? std::min(1.0, samples.size() * static_cast<double>(k_timestep) / work)
            : 0.0;
        std::cout << "Calibration " << presets[index].brick_count
                  << " bricks: collision p95 "
                  << metrics.collision_p95_milliseconds << " ms, max "
                  << metrics.collision_maximum_milliseconds << " ms\n";
        if (index == 0U) {
            output = {.config = presets[index], .metrics = metrics,
                      .hardware =
                          parallel_mater::gallery::platform_hardware_identity()};
        }
        if (!parallel_mater::gallery::calibration_passes(metrics)) {
            if (index == 0U) {
                glfwSwapInterval(1);
                error = "minimum one-brick preset failed the 1080p frame budget";
                return false;
            }
            if (best.has_value()) break;
            continue;
        }
        best = index;
    }
    if (!best.has_value()) {
        glfwSwapInterval(1); error = "no brick preset met the 1080p frame budget";
        return false;
    }
    for (std::size_t index = *best + 1U; index-- > 0U;) {
        std::vector<parallel_mater::gallery::CalibrationSample> samples;
        bool stable = true;
        bool off_center = false;
        const auto begin = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - begin).count() < 180.0) {
            if (!run_calibration_trial(options, presets[index], window, 60U, 300U,
                                       off_center, samples, stable, error)) {
                glfwSwapInterval(1); return false;
            }
            if (!stable) break;
            off_center = !off_center;
        }
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        auto metrics = parallel_mater::gallery::summarize_calibration(
            samples, duration, stable, false);
        const double work = measured_seconds(samples);
        metrics.simulation_progress_ratio = work > 0.0
            ? std::min(1.0, samples.size() * static_cast<double>(k_timestep) / work)
            : 0.0;
        if (parallel_mater::gallery::calibration_passes(metrics)) {
            options.bricks = presets[index];
            output = {.config = presets[index], .metrics = metrics,
                      .hardware =
                          parallel_mater::gallery::platform_hardware_identity()};
            glfwSwapInterval(1);
            return true;
        }
        if (index == 0U) break;
    }
    glfwSwapInterval(1);
    error = "no brick preset passed sustained 1080p qualification";
    return false;
}

void save_local_calibration(const CalibrationResult &result) {
    auto path = parallel_mater::gallery::default_local_profiles_path();
    path = path.parent_path() / "latest-brick-calibration.json";
    std::error_code filesystem_error;
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error) return;
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) return;
    output << "{\n  \"verified\": false,\n  \"backend\": \""
           << result.hardware.backend << "\",\n"
           << "  \"gpu_model\": \"" << result.hardware.gpu_model << "\",\n"
           << "  \"gpu_variant\": \"" << result.hardware.gpu_variant << "\",\n"
           << "  \"machine_model\": \"" << result.hardware.machine_model << "\",\n"
           << "  \"cpu_model\": \"" << result.hardware.cpu_model << "\",\n"
           << "  \"memory_bytes\": " << result.hardware.memory_bytes << ",\n"
           << "  \"operating_system\": \""
           << result.hardware.operating_system << "\",\n"
           << "  \"driver\": \"" << result.hardware.driver << "\",\n"
           << "  \"power_mode\": \"" << result.hardware.power_mode << "\",\n"
           << "  \"width\": " << brick_render_width << ",\n"
           << "  \"height\": " << brick_render_height << ",\n"
           << "  \"scene_version\": "
           << parallel_mater::gallery::brick_scene_version << ",\n"
           << "  \"solver_version\": \""
           << parallel_mater::gallery::slang_solver_version << "\",\n"
           << "  \"build_revision\": \""
           << PARALLEL_MATER_GALLERY_GIT_COMMIT << "\",\n"
           << "  \"brick_count\": " << result.config.brick_count << ",\n"
           << "  \"brick_scale\": " << result.config.brick_scale << ",\n"
           << "  \"wall_planes\": " << result.config.wall_planes << ",\n"
           << "  \"collision_p95_milliseconds\": "
           << result.metrics.collision_p95_milliseconds << ",\n"
           << "  \"collision_maximum_milliseconds\": "
           << result.metrics.collision_maximum_milliseconds << ",\n"
           << "  \"quiet_p95_milliseconds\": "
           << result.metrics.quiet_p95_milliseconds << ",\n"
           << "  \"quiet_maximum_milliseconds\": "
           << result.metrics.quiet_maximum_milliseconds << ",\n"
           << "  \"duration_seconds\": "
           << result.metrics.duration_seconds << ",\n"
           << "  \"simulation_progress_ratio\": "
           << result.metrics.simulation_progress_ratio << ",\n"
           << "  \"stable\": "
           << (result.metrics.stable ? "true" : "false") << ",\n"
           << "  \"no_dropped_steps\": "
           << (result.metrics.no_dropped_steps ? "true" : "false") << "\n}\n";
    output.close();
    if (!output) return;
    std::filesystem::rename(temporary, path, filesystem_error);
    if (filesystem_error) std::filesystem::remove(temporary);
}

bool verify_and_save_profile(const Options &options,
                             const CalibrationResult &result,
                             std::string &error) {
    parallel_mater::gallery::VerifiedBrickProfile profile;
    profile.hardware = result.hardware;
    profile.scene = result.config;
    profile.solver_version = parallel_mater::gallery::slang_solver_version;
    profile.build_revision = PARALLEL_MATER_GALLERY_GIT_COMMIT;
    profile.metrics = result.metrics;
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream timestamp;
    timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    profile.verified_at = timestamp.str();
    const char *user = std::getenv("USER");
    if (user == nullptr) user = std::getenv("USERNAME");
    profile.verifier = user != nullptr ? user : "human-verified";
    const std::filesystem::path destination = options.profiles_file_overridden
        ? options.profiles_file
        : parallel_mater::gallery::default_local_profiles_path().parent_path() /
              "verified-profile-export.json";
    const bool saved = parallel_mater::gallery::save_verified_profile(
        destination, profile, error);
    if (saved) std::cout << "Verified profile written to " << destination << '\n';
    return saved;
}

} // namespace

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

    if (glfwInit() == GLFW_FALSE) {
        std::cerr << "GLFW initialization failed\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow *window = glfwCreateWindow(
        1280, 720,
#ifdef PARALLEL_MATER_SLANG_TEST_BUILD
        "ParallelMater Gallery - Slang Production Test",
#else
        "ParallelMater Gallery",
#endif
        nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "GLFW window creation failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(window, &input_state);
    glfwSetMouseButtonCallback(window, &mouse_button);
    glfwSetCursorPosCallback(window, &cursor_position);
    glfwSetScrollCallback(window, &scroll);
    glfwSetCharCallback(window, &character_input);
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    if (options.calibrate) {
        CalibrationResult result;
        if (!calibrate_bricks(options, window, result, error)) {
            if (result.metrics.duration_seconds > 0.0)
                save_local_calibration(result);
            std::cerr << "Calibration failed: " << error << '\n';
            glfwDestroyWindow(window);
            glfwTerminate();
            return 1;
        }
        last_calibration = result;
        save_local_calibration(result);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    }

    KeyEdges keys;
    bool timing_visible = false;
    GalleryDebugState debug;
    debug.reset(options.fluid_particle_view);
    bool capture_requested = false;
    using FrameClock = std::chrono::steady_clock;
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
                if (calibrate_bricks(options, window, result,
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
            if (physics_steps != 0U && capture_requested) {
                std::filesystem::path capture_path;
                if (save_physics_debug_capture(runtime.frame, capture_path,
                                               error)) {
                    std::cout << "Physics capture " << capture_path << '\n';
                } else {
                    std::cerr << "Physics capture failed: " << error << '\n';
                }
                capture_requested = false;
            }
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

        int framebuffer_width = 0, framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        glViewport(0, 0, framebuffer_width, framebuffer_height);
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2f(-1.0F, -1.0F);
        glPixelZoom(static_cast<float>(framebuffer_width) /
                        static_cast<float>(options.width),
                    static_cast<float>(framebuffer_height) /
                        static_cast<float>(options.height));
        glDrawPixels(static_cast<int>(options.width),
                     static_cast<int>(options.height), GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
