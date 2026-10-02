// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/gallery_debug.hpp>
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/physics_debug.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_gallery/surface_query.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using parallel_mater::RigidBodyState;
using parallel_mater::RigidConstraintOptions;
using parallel_mater::RigidConstraintState;
using parallel_mater::RigidConstraintType;
using parallel_mater::Quaternion;
using parallel_mater::SoftBodyDeviceView;
using parallel_mater::SoftBodyId;
using parallel_mater::SmokeDeviceView;
using parallel_mater::Status;
using parallel_mater::World;
using parallel_mater::Vec3;
using parallel_mater::gallery::CameraController;
using parallel_mater::gallery::CameraDragMode;
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
using parallel_mater::gallery::OptixRenderer;
using parallel_mater::gallery::SmokeDebugMode;
using parallel_mater::gallery::SceneDefinition;
using parallel_mater::gallery::SceneInstance;
using parallel_mater::gallery::StaticTriangleSurface;
using parallel_mater::gallery::SurfaceSelection;

constexpr float k_timestep = 1.0F / 60.0F;
constexpr float k_kinematic_speed = 2.0F;
constexpr float k_gravity = 9.81F;
constexpr float k_cloth_gravity_tilt_degrees = 45.0F;
constexpr float k_gravity_tilt_tangent = 0.577350269F;
constexpr float k_pi = 3.14159265358979323846F;
constexpr float k_dump_initial_angle = k_pi * 0.25F;
constexpr float k_dump_final_angle = -k_pi * 0.25F;
constexpr float k_dump_rotation_speed = k_pi * 0.25F;
constexpr float k_motor_speed = 8.0F;
constexpr std::uint32_t k_default_dump_spheres = 100U;
constexpr std::uint32_t k_default_fluid_particles = 30'000U;

struct Options {
    std::filesystem::path scene{PARALLEL_MATER_DEFAULT_SCENE_PATH};
    std::filesystem::path headless_output{};
    std::filesystem::path physics_capture_output{};
    int frames{240};
    std::uint32_t width{960U};
    std::uint32_t height{720U};
    GalleryContext initial_context{GalleryContext::rigid_body};
    std::uint32_t dump_spheres{k_default_dump_spheres};
    std::uint32_t fluid_particles{k_default_fluid_particles};
    std::uint32_t headless_cloth_tilt_degrees{};
    std::uint32_t headless_cloth_tilt_after_frames{};
    std::uint32_t headless_constraint_action_after_frames{};
    bool headless_cloth_tilt_left{};
    bool headless_motor_forward{};
    bool fluid_particle_view{};
    bool trace_fluid_escapes{};
    bool cloth_debug{};
};

struct InputState {
    CameraController camera{};
    bool count_dialog_visible{};
    bool replace_count_value{};
    bool count_value_invalid{};
    std::string count_value{};
};

struct DirectionalInput {
    float x{};
    float z{};
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

[[nodiscard]] DirectionalInput directional_input(GLFWwindow *window) {
    DirectionalInput input{
        static_cast<float>(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) -
            static_cast<float>(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS),
        static_cast<float>(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) -
            static_cast<float>(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)};
    const float length = std::sqrt(input.x * input.x + input.z * input.z);
    if (length > 1.0F) {
        input.x /= length;
        input.z /= length;
    }
    return input;
}

[[nodiscard]] parallel_mater::Vec3 gravity_for(DirectionalInput input) {
    const float horizontal_squared = input.x * input.x + input.z * input.z;
    if (horizontal_squared == 0.0F) {
        return {0.0F, -k_gravity, 0.0F};
    }
    const float inverse = 1.0F /
        std::sqrt(1.0F + k_gravity_tilt_tangent * k_gravity_tilt_tangent);
    return {input.x * k_gravity * k_gravity_tilt_tangent * inverse,
            -k_gravity * inverse,
            input.z * k_gravity * k_gravity_tilt_tangent * inverse};
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
        } else if (argument == "--fluid-particles" && index + 1 < argc) {
            const GalleryEntry &entry = gallery_entry(GalleryContext::fluid);
            if (!parse_count(argv[++index], entry.minimum_count,
                             entry.maximum_count,
                             output.fluid_particles)) return false;
            if (!is_fluid_context(output.initial_context))
                output.initial_context = GalleryContext::fluid;
        } else if (const GalleryEntry *entry = entry_for_option(argument)) {
            output.initial_context = entry->context;
        } else if ((argument == "--cloth-tilt-degrees" ||
                    argument == "--gravity-tilt-degrees") && index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 45U,
                             output.headless_cloth_tilt_degrees)) return false;
        } else if (argument == "--cloth-tilt-after-frames" && index + 1 < argc) {
            if (!parse_count(argv[++index], 0U, 100000U,
                             output.headless_cloth_tilt_after_frames)) return false;
        } else if (argument == "--cloth-tilt-left") {
            output.headless_cloth_tilt_left = true;
        } else if (argument == "--constraint-action-after-frames" &&
                   index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 100000U,
                             output.headless_constraint_action_after_frames))
                return false;
        } else if (argument == "--motor-forward") {
            output.headless_motor_forward = true;
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
                         "[--dump-spheres N] [";
            bool first = true;
            for (const GalleryEntry &entry : gallery_entries) {
                if (entry.command_line_option.empty()) continue;
                std::cout << (first ? "" : "|") << entry.command_line_option;
                first = false;
            }
            std::cout << "] [--fluid-particles N] "
                         "[--gravity-tilt-degrees 1..45 (headless)] "
                         "[--cloth-tilt-after-frames N (headless)] "
                         "[--cloth-tilt-left (headless)] "
                         "[--constraint-action-after-frames N (headless)] "
                         "[--motor-forward (headless)] "
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
    return !output.trace_fluid_escapes || !output.headless_output.empty();
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

[[nodiscard]] bool toggle_constraint(GalleryRuntime &runtime) {
    if (runtime.scene.rigid_constraints.size() != 1U ||
        runtime.instance.rigid_constraints.size() != 1U) {
        std::cerr << "Constraint toggle scene needs exactly one constraint\n";
        return false;
    }
    auto &definition = runtime.scene.rigid_constraints.front();
    RigidConstraintState constraint_state{};
    if (!require(runtime.world.read_rigid_constraint_state(
                     runtime.instance.rigid_constraints.front(),
                     constraint_state),
                 "read constraint state")) return false;

    RigidConstraintOptions options = definition.options;
    options.body_a = runtime.instance.rigid_bodies[definition.body_a];
    options.body_b = runtime.instance.rigid_bodies[definition.body_b];
    options.enabled = !constraint_state.enabled;
    if (options.enabled) {
        RigidBodyState state_a{}, state_b{};
        if (!require(runtime.world.read_rigid_body_state(options.body_a, state_a),
                     "read first constraint body") ||
            !require(runtime.world.read_rigid_body_state(options.body_b, state_b),
                     "read second constraint body")) return false;
        const Vec3 anchor = options.type == RigidConstraintType::point
            ? state_b.position : midpoint(state_a.position, state_b.position);
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
                     runtime.instance.rigid_constraints.front(), options),
                 options.enabled ? "enable constraint" : "disable constraint"))
        return false;
    definition.options = options;
    return true;
}

[[nodiscard]] bool drive_motors(GalleryRuntime &runtime,
                                DirectionalInput input) {
    const float forward = -input.z;
    const float left_speed = -(forward + input.x) * k_motor_speed;
    const float right_speed = -(forward - input.x) * k_motor_speed;
    for (std::size_t index = 0U;
         index < runtime.scene.rigid_constraints.size(); ++index) {
        auto &definition = runtime.scene.rigid_constraints[index];
        if (definition.options.type != RigidConstraintType::motor) continue;
        const float target_velocity =
            definition.name.find("Left") != std::string::npos
                ? left_speed : right_speed;
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
    if (!input->count_dialog_visible || codepoint < '0' || codepoint > '9') {
        return;
    }
    if (input->replace_count_value) {
        input->count_value.clear();
        input->replace_count_value = false;
    }
    if (input->count_value.size() < 6U) {
        input->count_value.push_back(static_cast<char>(codepoint));
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
    if (entry.source == GallerySceneSource::procedural_dump) {
        next.scene = parallel_mater::gallery::make_dump_scene(dump_spheres);
    } else {
        const std::array scene_paths{
            options.scene,
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_SLIDER_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH),
            std::filesystem::path(
                PARALLEL_MATER_CONSTRAINT_GENERIC_SPRING_SCENE_PATH),
            std::filesystem::path(PARALLEL_MATER_CONSTRAINT_MOTOR_SCENE_PATH),
            std::filesystem::path{},
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
        if (entry.has_fluid)
            next.scene.fluid_options.capacity = fluid_particles;
    }
    const Status create_status = parallel_mater::gallery::create_scene_world(
        next.scene, next.world, next.instance,
        {.frame_capacity = 30U, .frame_stride = 1U});
    if (!create_status) {
        error = create_status.message != nullptr ? create_status.message
                                                 : "scene creation failed";
        return false;
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
    float dump_angle{k_dump_initial_angle};
    parallel_mater::Vec3 peg_gravity{};
    parallel_mater::Vec3 cloth_gravity{};
    parallel_mater::WorldStepTimings timings{};
    parallel_mater::WorldStatistics statistics{};
    parallel_mater::gallery::RendererTimings renderer_timings{};

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
        dump_angle = k_dump_initial_angle;
        peg_gravity = initial_scene_gravity(context, runtime.scene.gravity_scale);
        cloth_gravity = peg_gravity;
        timings = {};
        statistics = {};
        renderer_timings = {};
        return true;
    }
};

} // namespace

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;

    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "Invalid arguments. Use --help.\n";
        return 2;
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
    float &dump_angle = session.dump_angle;
    Vec3 &peg_gravity = session.peg_gravity;
    Vec3 &cloth_gravity = session.cloth_gravity;
    WorldStepTimings &timings = session.timings;
    WorldStatistics &statistics = session.statistics;
    RendererTimings &renderer_timings = session.renderer_timings;

    const StepOptions step_options{.timestep = k_timestep,
                                   .substeps = 4U,
                                   .gravity = initial_scene_gravity(
                                       runtime.context,
                                       runtime.scene.gravity_scale)};
    std::vector<std::uint32_t> pixels;
    InputState input_state;
    input_state.camera.set_preset(gallery_entry(runtime.context).camera);
    if (!options.headless_output.empty()) {
        StepOptions headless_step = step_options;
        if (gallery_entry(runtime.context).controls ==
                GalleryControlPolicy::cloth_gravity &&
            options.headless_cloth_tilt_degrees != 0U) {
            const float angle = static_cast<float>(
                options.headless_cloth_tilt_degrees) * k_pi / 180.0F;
            const float magnitude = k_gravity * runtime.scene.gravity_scale;
            headless_step.gravity = {0.0F, -magnitude * std::cos(angle),
                                    -magnitude * std::sin(angle)};
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
        float headless_dump_angle = k_dump_initial_angle;
        for (int frame = 0; frame < options.frames; ++frame) {
            if (options.headless_constraint_action_after_frames != 0U &&
                frame == static_cast<int>(
                    options.headless_constraint_action_after_frames) &&
                !toggle_constraint(runtime)) return 1;
            if (options.headless_motor_forward &&
                gallery_entry(runtime.context).controls ==
                    GalleryControlPolicy::tank_motor &&
                !drive_motors(runtime, {0.0F, -1.0F})) return 1;
            if (gallery_entry(runtime.context).controls ==
                    GalleryControlPolicy::dump_rotation &&
                runtime.kinematic_index < runtime.instance.rigid_bodies.size()) {
                headless_dump_angle = std::max(
                    k_dump_final_angle,
                    headless_dump_angle - k_dump_rotation_speed * k_timestep);
                runtime.kinematic_target.orientation =
                    rotation_z(headless_dump_angle);
                if (!require(runtime.world.set_kinematic_target(
                                 runtime.instance.rigid_bodies[
                                     runtime.kinematic_index],
                                 runtime.kinematic_target),
                             "rotate headless DUMP hopper")) {
                    return 1;
                }
            }
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
        static_cast<int>(options.width), static_cast<int>(options.height),
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

    KeyEdges keys;
    bool timing_visible = false;
    GalleryDebugState debug;
    debug.reset(options.fluid_particle_view);
    bool capture_requested = false;
    bool context_visible = false;
    GalleryContext context_selection = runtime.context;
    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        keys.update(window);

        if (input_state.count_dialog_visible) {
            if (keys.pressed(KeyAction::escape)) {
                input_state.count_dialog_visible = false;
                input_state.count_value_invalid = false;
            }
            if (keys.pressed(KeyAction::backspace)) {
                if (input_state.replace_count_value) {
                    input_state.count_value.clear();
                    input_state.replace_count_value = false;
                } else if (!input_state.count_value.empty()) {
                    input_state.count_value.pop_back();
                }
                input_state.count_value_invalid = false;
            }
            if (keys.pressed(KeyAction::enter)) {
                std::uint32_t requested = 0U;
                const GalleryEntry &entry = gallery_entry(runtime.context);
                const bool fluid_dialog =
                    entry.count_kind == GalleryCountKind::fluid_particles;
                if (!parse_count(input_state.count_value, entry.minimum_count,
                                 entry.maximum_count,
                                 requested)) {
                    input_state.count_value_invalid = true;
                } else {
                    if (session.rebuild(
                            options, runtime.context,
                            fluid_dialog ? dump_spheres : requested,
                            fluid_dialog ? requested : fluid_particles,
                            error)) {
                        input_state.count_dialog_visible = false;
                        input_state.count_value_invalid = false;
                    } else {
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
                                gallery_entry(runtime.context).camera);
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
                input_state.count_value = std::to_string(
                    gallery_entry(runtime.context).count_kind ==
                            GalleryCountKind::fluid_particles
                        ? fluid_particles : dump_spheres);
                input_state.replace_count_value = true;
                input_state.count_value_invalid = false;
            }
            if (!context_visible && keys.pressed(KeyAction::reset)) {
                if (!session.rebuild(options, runtime.context, dump_spheres,
                                     fluid_particles, error)) {
                    std::cerr << "Scene reset failed: " << error << '\n';
                }
            }
            if (!context_visible && keys.pressed(KeyAction::action) &&
                gallery_entry(runtime.context).controls ==
                    GalleryControlPolicy::constraint_toggle &&
                !toggle_constraint(runtime)) {
                break;
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

        if (!input_state.count_dialog_visible) {
            const DirectionalInput directional =
                context_visible ? DirectionalInput{} : directional_input(window);
            const GalleryEntry &entry = gallery_entry(runtime.context);
            if (entry.controls == GalleryControlPolicy::tank_motor &&
                !drive_motors(runtime, directional)) {
                break;
            }
            if (runtime.kinematic_index < runtime.instance.rigid_bodies.size()) {
                if (entry.controls == GalleryControlPolicy::dump_rotation) {
                    if (!context_visible &&
                        glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                        dump_angle = std::max(
                            k_dump_final_angle,
                            dump_angle - k_dump_rotation_speed * k_timestep);
                    }
                    runtime.kinematic_target.orientation =
                        rotation_z(dump_angle);
                } else {
                    runtime.kinematic_target.position.x +=
                        directional.x * k_kinematic_speed * k_timestep;
                    runtime.kinematic_target.position.z +=
                        directional.z * k_kinematic_speed * k_timestep;
                }
                if (!require(runtime.world.set_kinematic_target(
                                 runtime.instance.rigid_bodies[
                                     runtime.kinematic_index],
                                 runtime.kinematic_target),
                             "move kinematic body")) {
                    break;
                }
            }
            StepOptions interactive_step = step_options;
            interactive_step.gravity = {0.0F,
                -k_gravity * runtime.scene.gravity_scale, 0.0F};
            if (entry.controls == GalleryControlPolicy::cloth_gravity) {
                cloth_gravity = steer_gravity(
                    cloth_gravity, input_state.camera.camera(),
                    directional.x, -directional.z,
                    k_gravity * runtime.scene.gravity_scale,
                    k_cloth_gravity_tilt_degrees, k_timestep);
                interactive_step.gravity = cloth_gravity;
            } else if (entry.controls == GalleryControlPolicy::rigid_gravity) {
                interactive_step.gravity = gravity_for(directional);
            } else if (entry.controls == GalleryControlPolicy::peg_gravity) {
                const float right = directional.x + (!context_visible ?
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) -
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                    : 0.0F);
                const float forward = -directional.z + (!context_visible ?
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) -
                    static_cast<float>(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                    : 0.0F);
                peg_gravity = steer_gravity(
                    peg_gravity, input_state.camera.camera(), right, forward,
                    k_gravity * runtime.scene.gravity_scale,
                    peg_paint_gravity_tilt_degrees, k_timestep);
                interactive_step.gravity = peg_gravity;
            }
            interactive_step.collect_kernel_timings = timing_visible;
            if (!require(runtime.world.step(interactive_step), "step gallery")) {
                break;
            }
            if (capture_requested) {
                std::filesystem::path capture_path;
                if (save_physics_debug_capture(runtime.world, capture_path,
                                               error)) {
                    std::cout << "Physics capture " << capture_path << '\n';
                } else {
                    std::cerr << "Physics capture failed: " << error << '\n';
                }
                capture_requested = false;
            }
            if (timing_visible &&
                !require(runtime.world.collect_step_timings(timings),
                         "collect timings")) {
                break;
            }
            if (timing_visible && is_fluid_context(runtime.context) &&
                !require(runtime.world.collect_statistics(statistics),
                         "collect fluid statistics")) break;
        }

        const Camera current_camera = input_state.camera.camera();
        if (!runtime.renderer.render(runtime.world, runtime.instance,
                                     current_camera, pixels, error,
                                     timing_visible ? &renderer_timings : nullptr,
                                     debug.fluid_render_mode(runtime.context),
                                     debug.smoke_mode == SmokeDebugMode::none)) {
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
            draw_count_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                runtime.context, input_state.count_value,
                input_state.count_value_invalid);
        }

        glViewport(0, 0, static_cast<int>(options.width),
                   static_cast<int>(options.height));
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2f(-1.0F, -1.0F);
        glDrawPixels(static_cast<int>(options.width),
                     static_cast<int>(options.height), GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
