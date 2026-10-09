// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/arrow_forces.hpp>
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/fixed_collector.hpp>
#include <parallel_mater_gallery/gallery_debug.hpp>
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/physics_debug.hpp>
#include <parallel_mater_gallery/physics_frame_budget.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/surface_query.hpp>

#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>

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
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace {

using parallel_mater::RigidBodyId;
using parallel_mater::RigidBodyState;
using parallel_mater::RigidConstraintOptions;
using parallel_mater::RigidConstraintState;
using parallel_mater::RigidConstraintType;
using parallel_mater::MotionType;
using parallel_mater::Quaternion;
using parallel_mater::SoftBodyDeviceView;
using parallel_mater::SoftBodyId;
using parallel_mater::SmokeDeviceView;
using parallel_mater::Status;
using parallel_mater::StepOptions;
using parallel_mater::World;
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
using parallel_mater::gallery::OptixRenderer;
using parallel_mater::gallery::SmokeDebugMode;
using parallel_mater::gallery::SceneDefinition;
using parallel_mater::gallery::SceneInstance;
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
    return context == GalleryContext::constraint_hinge || context == GalleryContext::dump ? 8U : 4U;
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
};

struct InputState {
    CameraController camera{};
    bool count_dialog_visible{};
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
    World world{};
    SceneInstance instance{};
    OptixRenderer renderer{};
    FixedContactCollector fixed_collector{};
    parallel_mater::gallery::ArrowForces arrow_forces{};
    parallel_mater::gallery::DumpTruckBed dump_bed{};
    std::vector<SuspensionSteeringJoint> suspension_steering_joints{};
    float steering_angle{};
    std::vector<RigidBodyId> gravity_tilt_bodies{};
    std::size_t kinematic_index{std::numeric_limits<std::size_t>::max()};
    RigidBodyState kinematic_target{};
};

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

[[nodiscard]] bool trace_fluid_escapes(const World &world,
                                       const SceneInstance &instance,
                                       const StaticTriangleSurface &floor,
                                       int frame, FluidEscapeTrace &trace) {
    parallel_mater::FluidDeviceView view{};
    const Status status = world.fluid_view(instance.fluid, view);
    if (!status) {
        std::cerr << "Fluid escape diagnostic view failed: "
                  << (status.message ? status.message : "unknown") << '\n';
        return false;
    }
    std::vector<parallel_mater::Vec3> positions(view.particle_count);
    if (!positions.empty()) {
        const cudaError_t copy = cudaMemcpy(
            positions.data(), view.positions.data,
            positions.size() * sizeof(positions[0]), cudaMemcpyDeviceToHost);
        if (copy != cudaSuccess) {
            std::cerr << "Fluid escape diagnostic readback failed: "
                      << cudaGetErrorString(copy) << '\n';
            return false;
        }
    }
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
        stable_ids.resize(positions.size());
        const cudaError_t copy = cudaMemcpy(
            stable_ids.data(), view.stable_particle_ids.data,
            stable_ids.size() * sizeof(stable_ids[0]), cudaMemcpyDeviceToHost);
        if (copy != cudaSuccess) {
            std::cerr << "Fluid escape ID readback failed: "
                      << cudaGetErrorString(copy) << '\n';
            return false;
        }
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

[[nodiscard]] Status apply_gravity_tilt_overrides(
    GalleryRuntime &runtime, Vec3 world_gravity) noexcept {
    if (runtime.gravity_tilt_bodies.empty()) return {};
    const Vec3 vertical_gravity{
        0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F};
    const Vec3 compensation{
        vertical_gravity.x - world_gravity.x,
        vertical_gravity.y - world_gravity.y,
        vertical_gravity.z - world_gravity.z};
    if (compensation.x == 0.0F && compensation.y == 0.0F &&
        compensation.z == 0.0F) return {};
    return runtime.world.apply_central_acceleration(
        {runtime.gravity_tilt_bodies.data(),
         runtime.gravity_tilt_bodies.size()}, compensation);
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
                         "[--dump-spheres N] [";
            bool first = true;
            for (const GalleryEntry &entry : gallery_entries) {
                if (entry.command_line_option.empty()) continue;
                std::cout << (first ? "" : "|") << entry.command_line_option;
                first = false;
            }
            std::cout << "] [--fluid-particles N] "
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
    return (!output.trace_fluid_escapes || !output.headless_output.empty()) &&
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
    for (const auto id : runtime.instance.rigid_constraints) {
        RigidConstraintState state{};
        if (!require(runtime.world.read_rigid_constraint_state(id, state),
                     "read constraint state")) return false;
        enable = enable || !state.enabled;
    }

    if (enable && runtime.context == GalleryContext::constraint_point) {
        // Released arms can be metres away. Restore the authored assembly
        // before enabling its joints instead of injecting a snap-back impulse.
        for (std::size_t index = 0; index < runtime.scene.rigid_bodies.size(); ++index) {
            const auto &body = runtime.scene.rigid_bodies[index];
            if (body.options.motion == MotionType::dynamic &&
                !require(runtime.world.set_rigid_body_state(
                             runtime.instance.rigid_bodies[index], body.options.initial_state),
                         "restore point assembly")) return false;
        }
    }

    for (std::size_t index = 0U;
         index < runtime.scene.rigid_constraints.size(); ++index) {
        auto &definition = runtime.scene.rigid_constraints[index];
        RigidConstraintOptions options = definition.options;
        options.body_a = runtime.instance.rigid_bodies[definition.body_a];
        options.body_b = runtime.instance.rigid_bodies[definition.body_b];
        options.enabled = enable;
        if (enable && options.type != RigidConstraintType::point) {
            RigidBodyState state_a{}, state_b{};
            if (!require(runtime.world.read_rigid_body_state(
                             options.body_a, state_a),
                         "read first constraint body") ||
                !require(runtime.world.read_rigid_body_state(
                             options.body_b, state_b),
                         "read second constraint body")) return false;
            const Vec3 anchor = midpoint(state_a.position, state_b.position);
            const Quaternion world_orientation = state_a.orientation;
            options.local_anchor_a = rotate(conjugate(state_a.orientation),
                                            subtract(anchor, state_a.position));
            options.local_anchor_b = rotate(conjugate(state_b.orientation),
                                            subtract(anchor, state_b.position));
            options.local_orientation_a =
                multiply(conjugate(state_a.orientation), world_orientation);
            options.local_orientation_b =
                multiply(conjugate(state_b.orientation), world_orientation);
        }
        if (!require(runtime.world.update_rigid_constraint(
                         runtime.instance.rigid_constraints[index], options),
                     enable ? "enable constraint" : "disable constraint"))
            return false;
        definition.options = options;
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
        RigidConstraintOptions options = definition.options;
        options.body_a = runtime.instance.rigid_bodies[definition.body_a];
        options.body_b = runtime.instance.rigid_bodies[definition.body_b];
        options.motor.angular_target_velocity = target_velocity;
        if (!require(runtime.world.update_rigid_constraint(
                         runtime.instance.rigid_constraints[index], options),
                     "drive motor constraint")) return false;
        definition.options = options;
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
        RigidConstraintOptions options = definition.options;
        options.body_a = runtime.instance.rigid_bodies[definition.body_a];
        options.body_b = runtime.instance.rigid_bodies[definition.body_b];
        const float angle = steering.front ? angles.front : angles.rear;
        options.local_orientation_a = multiply(
            steering.authored_local_orientation_a, rotation_z(angle));
        if (!require(runtime.world.update_rigid_constraint(
                         runtime.instance.rigid_constraints[
                             steering.constraint_index], options),
                     "steer suspension constraint")) return false;
        definition.options = options;
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
        input->camera.end_drag();
        return;
    }
    if (action != GLFW_PRESS || input->count_dialog_visible) return;
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    input->camera.begin_drag((modifiers & GLFW_MOD_SHIFT) != 0
                                 ? CameraDragMode::pan
                                 : CameraDragMode::orbit,
                             x, y);
}

void cursor_position(GLFWwindow *window, double x, double y) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    int height = 0;
    glfwGetWindowSize(window, nullptr, &height);
    input->camera.move_cursor(x, y, height,
                              !input->count_dialog_visible);
}

void scroll(GLFWwindow *window, double, double offset) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (input->count_dialog_visible) {
        return;
    }
    input->camera.zoom(offset);
}

void character_input(GLFWwindow *window, unsigned int codepoint) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    const bool digit = codepoint >= '0' && codepoint <= '9';
    const bool scale_point = input->brick_dialog && input->brick_field == 1U &&
        codepoint == '.';
    if (!input->count_dialog_visible || (!digit && !scale_point)) {
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
    parallel_mater::WorldOptions world_options{};
    if (context == GalleryContext::rigid_body) {
        std::size_t free_bytes = 0U, total_bytes = 0U;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
            parallel_mater::gallery::estimated_metal_contact_bytes(
                static_cast<std::uint32_t>(next.scene.rigid_bodies.size())) >
                static_cast<std::uint64_t>(free_bytes) * 7U / 10U) {
            error = "brick scene exceeds the available GPU memory budget";
            return false;
        }
    }
    Status create_status = parallel_mater::gallery::scene_world_options(
        next.scene, world_options,
        {.frame_capacity = context == GalleryContext::constraint_fixed ? 300U : 30U,
         .frame_stride = 1U});
    if (create_status && context == GalleryContext::constraint_fixed) {
        world_options.rigid_constraint_capacity =
            FixedContactCollector::constraint_capacity(next.scene);
    }
    if (create_status)
        create_status = World::create(world_options, next.world);
    if (create_status)
        create_status = parallel_mater::gallery::instantiate_scene(
            next.scene, next.world, next.instance);
    if (create_status)
        create_status = next.arrow_forces.initialize(next.scene, next.instance);
    if (create_status) {
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
    if (create_status && context == GalleryContext::constraint_fixed) {
        create_status = next.fixed_collector.initialize(
            next.scene, next.instance);
    }
    if (!create_status) {
        error = create_status.message != nullptr ? create_status.message
                                                 : "scene creation failed";
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
    if (!OptixRenderer::create(next.scene, next.world, next.instance,
                               PARALLEL_MATER_OPTIX_PTX_PATH,
                               options.width, options.height, next.renderer,
                               error)) {
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
    parallel_mater::WorldStepTimings timings{};
    parallel_mater::WorldStatistics statistics{};
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

parallel_mater::gallery::HardwareIdentity cuda_hardware_identity() {
    cudaDeviceProp properties{};
    int device = 0;
    int driver = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&properties, device);
    cudaDriverGetVersion(&driver);
    parallel_mater::gallery::HardwareIdentity result;
    const char *machine = std::getenv("COMPUTERNAME");
    if (machine == nullptr) machine = std::getenv("HOSTNAME");
    result.machine_model = machine != nullptr ? machine : properties.name;
    result.cpu_model = "unknown CPU";
#if defined(__linux__)
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            const std::string prefix = "model name";
            if (!line.starts_with(prefix)) continue;
            const auto separator = line.find(':');
            if (separator != std::string::npos) {
                result.cpu_model = line.substr(separator + 1U);
                result.cpu_model.erase(0U, result.cpu_model.find_first_not_of(" \t"));
            }
            break;
        }
    }
#elif defined(_WIN32)
    if (const char *processor = std::getenv("PROCESSOR_IDENTIFIER"))
        result.cpu_model = processor;
#endif
    result.gpu_model = properties.name;
    result.gpu_variant = std::to_string(properties.multiProcessorCount) +
        " SM, compute " + std::to_string(properties.major) + '.' +
        std::to_string(properties.minor);
    result.memory_bytes = properties.totalGlobalMem;
    result.backend = "cuda";
#if defined(_WIN32)
    result.operating_system = "Windows";
#elif defined(__linux__)
    struct utsname system{};
    result.operating_system = uname(&system) == 0
        ? std::string(system.sysname) + ' ' + system.release
        : "Linux";
#else
    result.operating_system = "unknown";
#endif
    result.driver = "CUDA driver " + std::to_string(driver);
    result.power_mode = "driver-default";
    return result;
}

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
        const Status status = runtime.world.set_rigid_body_state(
            runtime.instance.rigid_bodies[index], state);
        if (status) return true;
        error = status.message != nullptr ? status.message : "could not launch ball";
        return false;
    }
    error = "brick calibration scene has no ball";
    return false;
}

bool stable_calibration_scene(GalleryRuntime &runtime, bool quiet,
                              float scale, std::string &error) {
    unsigned displaced_bricks = 0U;
    for (std::size_t index = 0U; index < runtime.instance.rigid_bodies.size(); ++index) {
        RigidBodyState state;
        const Status status = runtime.world.read_rigid_body_state(
            runtime.instance.rigid_bodies[index], state);
        if (!status) {
            error = status.message != nullptr ? status.message : "could not read body state";
            return false;
        }
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
    for (std::uint32_t frame = 0U; frame < 30U; ++frame)
        if (!runtime.world.step(step)) { error = "calibration settle failed"; return false; }
    std::vector<std::uint32_t> pixels;
    CameraController camera;
    camera.set_preset(camera_preset_for(GalleryContext::rigid_body, config));
    auto measure = [&](bool collision) {
        const auto begin = std::chrono::steady_clock::now();
        const Status status = runtime.world.step(step);
        if (!status) {
            error = status.message != nullptr ? status.message : "calibration step failed";
            return false;
        }
        if (!runtime.renderer.render(runtime.world, runtime.instance,
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
                      .hardware = cuda_hardware_identity()};
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
                      .hardware = cuda_hardware_identity()};
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
    output << "{\n  \"verified\": false,\n  \"backend\": \"cuda\",\n"
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
           << "  \"solver_version\": \"cuda-rigid-v1\",\n"
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
    profile.solver_version = "cuda-rigid-v1";
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

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    Options options;
    std::error_code executable_error;
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
    if (options.headless_output.empty() && !options.bricks_overridden) {
        DeviceProfileCatalog catalog;
        std::string profile_error;
        if (load_device_profiles(device_profiles_path(options), catalog, profile_error)) {
            const HardwareIdentity hardware = cuda_hardware_identity();
            if (const auto *profile = find_matching_profile(
                    catalog, hardware, brick_render_width,
                    brick_render_height, "cuda-rigid-v1")) {
                options.bricks = profile->scene;
                std::cout << "Using verified brick profile: "
                          << options.bricks.brick_count << " bricks, "
                          << options.bricks.wall_planes << " walls\n";
            }
        } else {
            std::cerr << "Device profile catalog unavailable: "
                      << profile_error << '\n';
        }
    }
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
    WorldStepTimings &timings = session.timings;
    WorldStatistics &statistics = session.statistics;
    RendererTimings &renderer_timings = session.renderer_timings;
    std::optional<CalibrationResult> last_calibration;

    const StepOptions step_options{.timestep = k_timestep,
                                   .substeps = scene_substeps(runtime.context),
                                   .gravity = initial_scene_gravity(
                                       runtime.context,
                                       runtime.scene.gravity_scale)};
    std::vector<std::uint32_t> pixels;
    InputState input_state;
    input_state.camera.set_preset(camera_preset_for(runtime.context, options.bricks));
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
            if (!runtime.instance.has_fluid ||
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
                !require(runtime.dump_bed.advance(runtime.world, runtime.scene,
                            runtime.instance, k_timestep), "tilt dump bucket")) return 1;
            StepOptions frame_step = headless_step;
            if (frame < static_cast<int>(options.headless_cloth_tilt_after_frames))
                frame_step.gravity = step_options.gravity;
            else if (options.headless_cloth_tilt_left &&
                     gallery_entry(runtime.context).controls ==
                         GalleryControlPolicy::cloth_gravity) {
                headless_step.gravity = steer_gravity(
                    headless_step.gravity, input_state.camera.camera(),
                    -1.0F, 0.0F, k_gravity * runtime.scene.gravity_scale,
                    k_cloth_gravity_tilt_degrees, k_timestep);
                frame_step.gravity = headless_step.gravity;
            }
            if (fixed_collection) {
                frame_step.gravity =
                    frame < static_cast<int>(
                                options.headless_cloth_tilt_after_frames)
                    ? step_options.gravity : fixed_headless_gravity;
                if (!require(runtime.fixed_collector.apply_loose_gravity(
                                 runtime.world, runtime.scene,
                                 runtime.instance, step_options.gravity,
                                 frame_step.gravity),
                             "apply loose fixed-scene gravity")) {
                    return 1;
                }
            }
            if (!require(apply_gravity_tilt_overrides(
                             runtime, frame_step.gravity),
                         "preserve authored rigid gravity")) {
                return 1;
            }
            const Status frame_status = runtime.world.step(frame_step);
            if (!frame_status) {
                WorldStatistics failure_statistics{};
                (void)runtime.world.collect_statistics(failure_statistics);
                std::cerr << "Headless frame " << frame + 1
                          << " failed; peak fluid neighbors="
                          << failure_statistics.maximum_fluid_neighbor_count
                          << '\n';
                (void)require(frame_status, "step headless gallery");
                return 1;
            }
            if (fixed_collection &&
                !require(runtime.fixed_collector.collect(
                             runtime.world, runtime.scene, runtime.instance),
                         "collect fixed contacts")) {
                return 1;
            }
            if (runtime.instance.has_fluid &&
                frame + 1 < options.frames &&
                !runtime.renderer.advance_visuals(
                    runtime.world, runtime.instance, error)) {
                std::cerr << "Visual update failed: " << error << '\n';
                return 1;
            }
            if (options.trace_fluid_escapes &&
                !trace_fluid_escapes(runtime.world, runtime.instance,
                                     floor_index, frame + 1, escape_trace))
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
        if (runtime.instance.has_fluid) {
            parallel_mater::WorldStatistics statistics{};
            if (!require(runtime.world.collect_statistics(statistics),
                         "collect headless fluid statistics")) return 1;
            std::cout << "Fluid particles=" << statistics.particle_count
                      << " emitted=" << statistics.emitted_particle_count
                      << " outflowed=" <<
                          statistics.destroyed_particle_count -
                              statistics.boiled_particle_count
                      << " boiled=" << statistics.boiled_particle_count
                      << " capacity_misses="
                      << statistics.spawn_capacity_miss_count << '\n';
            parallel_mater::FluidDeviceView fluid_view{};
            if (!require(runtime.world.fluid_view(runtime.instance.fluid,
                                                  fluid_view),
                         "borrow headless fluid view")) return 1;
            if (fluid_view.particle_count != 0U) {
                std::vector<parallel_mater::Vec3> points(fluid_view.particle_count);
                const cudaError_t copy = cudaMemcpy(
                    points.data(), fluid_view.positions.data,
                    points.size() * sizeof(points[0]), cudaMemcpyDeviceToHost);
                if (copy != cudaSuccess) {
                    std::cerr << "Fluid diagnostic readback failed\n";
                    return 1;
                }
                parallel_mater::Vec3 low = points.front(), high = points.front();
                for (const auto &point : points) {
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
        if (runtime.instance.has_smoke) {
            parallel_mater::SmokeDeviceView smoke{};
            if (!require(runtime.world.smoke_view(runtime.instance.smoke, smoke),
                         "borrow headless smoke view")) return 1;
            std::cout << "Smoke particles=" << smoke.particle_count << '\n';
        }
        if (!options.physics_capture_output.empty()) {
            if (!write_physics_debug_capture(
                    runtime.world, options.physics_capture_output, error)) {
                std::cerr << "Physics capture failed: " << error << '\n';
                return 1;
            }
            std::cout << "Physics capture "
                      << options.physics_capture_output << '\n';
        }
        RendererTimings headless_render_timings{};
        if (!runtime.renderer.render(runtime.world, runtime.instance,
                                     input_state.camera.camera(),
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
            for (ClothId id : runtime.instance.cloths) {
                ClothDeviceView cloth{};
                if (!require(runtime.world.cloth_view(id, cloth),
                             "borrow headless cloth debug view") ||
                    !draw_cloth_debug_overlay(
                        pixels, runtime.renderer.width(), runtime.renderer.height(),
                        cloth, input_state.camera.camera(),
                        {.normals = true, .rigid_contact_forces = true,
                         .fluid_contact_forces = true, .wireframe = true,
                         .bonds = true}, error)) {
                    std::cerr << "Cloth debug overlay failed: " << error << '\n';
                    return 1;
                }
            }
            std::cout << "Cloth debug surfaces=" << runtime.instance.cloths.size() << '\n';
            PhysicsDebugFrameView debug_frame{};
            if (!require(runtime.world.physics_debug_frame(debug_frame),
                         "borrow headless physics debug frame")) return 1;
            draw_physics_debug_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                debug_frame, input_state.camera.camera(),
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
        if (runtime.instance.has_fluid && !options.fluid_particle_view &&
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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(
        960, 540,
        "ParallelMater Gallery", nullptr, nullptr);
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
    bool context_visible = false;
    GalleryContext context_selection = runtime.context;
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

        if (input_state.count_dialog_visible) {
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
                    input_state.count_dialog_visible = false;
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
                    input_state.count_dialog_visible = false;
                }
                if (!save_error.empty())
                    std::cerr << "Profile not saved: " << save_error << '\n';
            }
            if (keys.pressed(KeyAction::escape)) {
                input_state.count_dialog_visible = false;
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
            if (input_state.brick_dialog && keys.pressed(KeyAction::scenes)) {
                input_state.brick_field = (input_state.brick_field + 1U) % 3U;
                input_state.replace_count_value = true;
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
                        input_state.count_dialog_visible = false;
                        input_state.count_value_invalid = false;
                        if (input_state.brick_dialog) {
                            options.bricks_overridden = true;
                            input_state.camera.set_preset(
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
                if (context_visible) {
                    context_visible = false;
                } else {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
            }
            if (keys.pressed(KeyAction::scenes)) {
                context_visible = !context_visible;
                context_selection = runtime.context;
            }
            if (context_visible) {
                std::size_t selected = gallery_context_index(context_selection);
                if (keys.pressed(KeyAction::up)) {
                    if (selected != 0U) --selected;
                }
                if (keys.pressed(KeyAction::down)) {
                    selected = std::min(gallery_entries.size() - 1U,
                                        selected + 1U);
                }
                context_selection = gallery_entries[selected].context;
                if (keys.pressed(KeyAction::enter)) {
                    const bool context_changed =
                        context_selection != runtime.context;
                    if (!context_changed ||
                        session.rebuild(options, context_selection, dump_spheres,
                                        fluid_particles, error)) {
                        if (context_changed) {
                            debug.reset(options.fluid_particle_view);
                            input_state.camera.set_preset(
                                camera_preset_for(runtime.context, options.bricks));
                        }
                        context_visible = false;
                    } else {
                        std::cerr << "Scene switch failed: " << error << '\n';
                    }
                }
            } else if (gallery_entry(runtime.context).count_kind !=
                           GalleryCountKind::none &&
                       keys.pressed(KeyAction::particle_count)) {
                input_state.count_dialog_visible = true;
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
            if (!context_visible && keys.pressed(KeyAction::reset)) {
                if (!session.rebuild(options, runtime.context, dump_spheres,
                                     fluid_particles, error)) {
                    std::cerr << "Scene reset failed: " << error << '\n';
                }
            }
            if (!context_visible && keys.pressed(KeyAction::action)) {
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
        } else if (input_state.count_dialog_visible) {
            physics_accumulator = 0.0;
            rigid_interpolation_alpha = 1.0F;
        } else {
            physics_accumulator += frame_delta;
            const GalleryEntry &entry = gallery_entry(runtime.context);
            const DirectionalInput directional = context_visible
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
                    !require(runtime.dump_bed.advance(runtime.world, runtime.scene,
                                runtime.instance, k_timestep), "tilt dump bucket")) {
                    step_failed = true;
                    break;
                }
                if (runtime.kinematic_index <
                    runtime.instance.rigid_bodies.size()) {
                    runtime.kinematic_target.position.x +=
                        steering.x * k_kinematic_speed * k_timestep;
                    runtime.kinematic_target.position.z +=
                        steering.z * k_kinematic_speed * k_timestep;
                    if (!require(runtime.world.set_kinematic_target(
                                     runtime.instance.rigid_bodies[
                                         runtime.kinematic_index],
                                     runtime.kinematic_target),
                                 "move kinematic body")) {
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
                        cloth_gravity, input_state.camera.camera(),
                        steering.x, -steering.z,
                        k_gravity * runtime.scene.gravity_scale,
                        k_cloth_gravity_tilt_degrees, k_timestep);
                    interactive_step.gravity = cloth_gravity;
                } else if (entry.controls ==
                           GalleryControlPolicy::collector_gravity) {
                    interactive_step.collect_rigid_contacts = true;
                    interactive_step.gravity = collector_gravity_for(
                        steering, runtime.scene.gravity_scale,
                        input_state.camera.camera());
                    const Vec3 loose_gravity{
                        0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F};
                    if (!require(runtime.fixed_collector.apply_loose_gravity(
                                     runtime.world, runtime.scene,
                                     runtime.instance, loose_gravity,
                                     interactive_step.gravity),
                                 "apply loose fixed-scene gravity")) {
                        step_failed = true;
                        break;
                    }
                } else if (uses_rigid_gravity(entry.controls)) {
                    interactive_step.gravity = gravity_for(
                        steering, runtime.scene.gravity_scale,
                        input_state.camera.camera());
                } else if (entry.controls ==
                           GalleryControlPolicy::peg_gravity) {
                    const float right = steering.x + (!context_visible ?
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_D) == GLFW_PRESS) -
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_A) == GLFW_PRESS) : 0.0F);
                    const float forward = -steering.z + (!context_visible ?
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_W) == GLFW_PRESS) -
                        static_cast<float>(glfwGetKey(
                            window, GLFW_KEY_S) == GLFW_PRESS) : 0.0F);
                    peg_gravity = steer_gravity(
                        peg_gravity, input_state.camera.camera(), right,
                        forward, k_gravity * runtime.scene.gravity_scale,
                        peg_paint_gravity_tilt_degrees, k_timestep);
                    interactive_step.gravity = peg_gravity;
                }
                if (!require(apply_gravity_tilt_overrides(
                                 runtime, interactive_step.gravity),
                             "preserve authored rigid gravity")) {
                    step_failed = true;
                    break;
                }
                interactive_step.collect_kernel_timings = timing_visible;
                if (!require(runtime.arrow_forces.apply(runtime.world,
                                 input_state.camera.camera(), directional.x, -directional.z),
                             "apply screen-space arrow force")) {
                    step_failed = true;
                    break;
                }
                if (!require(runtime.world.step(interactive_step),
                             "step gallery")) {
                    step_failed = true;
                    break;
                }
                if (entry.controls ==
                        GalleryControlPolicy::collector_gravity &&
                    !require(runtime.fixed_collector.collect(
                                 runtime.world, runtime.scene,
                                 runtime.instance),
                             "collect fixed contacts")) {
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
                if (save_physics_debug_capture(runtime.world, capture_path,
                                               error)) {
                    std::cout << "Physics capture " << capture_path << '\n';
                } else {
                    std::cerr << "Physics capture failed: " << error << '\n';
                }
                capture_requested = false;
            }
            if (physics_steps != 0U && timing_visible &&
                !require(runtime.world.collect_step_timings(timings),
                         "collect timings")) {
                break;
            }
            if (physics_steps != 0U && timing_visible &&
                is_fluid_context(runtime.context) &&
                !require(runtime.world.collect_statistics(statistics),
                         "collect fluid statistics")) break;
        }

        const Camera current_camera = input_state.camera.camera();
        if (!runtime.renderer.render(runtime.world, runtime.instance,
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
            SmokeDeviceView smoke{};
            if (!require(runtime.world.smoke_view(runtime.instance.smoke, smoke),
                         "borrow smoke grid debug view") ||
                !draw_smoke_grid_debug_overlay(
                    pixels, runtime.renderer.width(), runtime.renderer.height(),
                    smoke, current_camera, debug.smoke_mode, error)) {
                std::cerr << "Smoke grid debug overlay failed: " << error << '\n';
                break;
            }
        }
        if (is_cloth_context(runtime.context) &&
            (debug.normals || debug.structure || debug.cloth_bonds)) {
            bool cloth_debug_ok = true;
            for (ClothId id : runtime.instance.cloths) {
                ClothDeviceView cloth{};
                if (!require(runtime.world.cloth_view(id, cloth),
                             "borrow cloth debug view") ||
                    !draw_cloth_debug_overlay(
                        pixels, runtime.renderer.width(),
                        runtime.renderer.height(), cloth, current_camera,
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
            for (SoftBodyId id : runtime.instance.soft_bodies) {
                SoftBodyDeviceView body{};
                if (!require(runtime.world.soft_body_view(id, body),
                             "borrow soft-body debug view") ||
                    !draw_soft_body_debug_overlay(
                        pixels, runtime.renderer.width(),
                        runtime.renderer.height(), body, current_camera,
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
            for (auto id:runtime.instance.ropes) {
                parallel_mater::RopeDeviceView view;
                if(!require(runtime.world.rope_view(id,view),"rope debug view") ||
                   !draw_rope_debug_overlay(pixels,runtime.renderer.width(),runtime.renderer.height(),view,current_camera,error)) {
                    std::cerr << "Rope debug failed: " << error << '\n';return 1;
                }
            }
        }
        if (debug.vectors_visible()) {
            PhysicsDebugFrameView debug_frame{};
            const Status debug_status =
                runtime.world.physics_debug_frame(debug_frame);
            if (!debug_status) {
                std::cerr << "Physics debug overlay failed: "
                          << (debug_status.message != nullptr
                                  ? debug_status.message : "unknown") << '\n';
                break;
            }
            draw_physics_debug_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                debug_frame, current_camera,
                {.contact_normals = debug.normals,
                 .rigid_forces = debug.rigid_forces,
                 .fluid_forces = debug.fluid_forces,
                 .velocities = debug.velocities});
        }
        if (debug.rigid_contacts &&
            !draw_rigid_contact_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                runtime.world.rigid_contacts(), current_camera, error)) {
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
                                    runtime.renderer.height(), timings);
        }
        if (context_visible) {
            draw_context_overlay(pixels, runtime.renderer.width(),
                                 runtime.renderer.height(), context_selection);
        }
        if (input_state.count_dialog_visible) {
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
