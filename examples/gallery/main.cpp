// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
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
using parallel_mater::Status;
using parallel_mater::World;
using parallel_mater::gallery::GalleryContext;
using parallel_mater::gallery::OptixRenderer;
using parallel_mater::gallery::SceneDefinition;
using parallel_mater::gallery::SceneInstance;

constexpr float k_timestep = 1.0F / 60.0F;
constexpr float k_kinematic_speed = 2.0F;
constexpr float k_gravity = 9.81F;
constexpr float k_gravity_tilt_tangent = 0.577350269F;
constexpr float k_pi = 3.14159265358979323846F;
constexpr float k_dump_initial_angle = k_pi * 0.25F;
constexpr float k_dump_final_angle = -k_pi * 0.25F;
constexpr float k_dump_rotation_speed = k_pi * 0.25F;
constexpr std::uint32_t k_minimum_dump_spheres = 10U;
constexpr std::uint32_t k_maximum_dump_spheres = 1'000U;
constexpr std::uint32_t k_default_dump_spheres = 100U;

struct Options {
    std::filesystem::path scene{PARALLEL_MATER_DEFAULT_SCENE_PATH};
    std::filesystem::path headless_output{};
    int frames{240};
    std::uint32_t width{960U};
    std::uint32_t height{720U};
    GalleryContext initial_context{GalleryContext::rigid_body};
    std::uint32_t dump_spheres{k_default_dump_spheres};
};

struct OrbitState {
    float yaw{0.62F};
    float pitch{0.32F};
    float distance{11.0F};
    bool dragging{};
    double previous_x{};
    double previous_y{};
};

struct InputState {
    OrbitState orbit{};
    bool dump_dialog_visible{};
    bool replace_dump_value{};
    bool dump_value_invalid{};
    std::string dump_value{};
};

struct DirectionalInput {
    float x{};
    float z{};
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

[[nodiscard]] parallel_mater::Quaternion rotation_z(float radians) {
    return {0.0F, 0.0F, std::sin(radians * 0.5F),
            std::cos(radians * 0.5F)};
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

[[nodiscard]] bool parse_dump_count(std::string_view value,
                                    std::uint32_t &output) {
    std::uint32_t parsed = 0U;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed < k_minimum_dump_spheres ||
        parsed > k_maximum_dump_spheres) {
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
            if (!parse_dump_count(argv[++index], output.dump_spheres)) {
                return false;
            }
            output.initial_context = GalleryContext::dump;
        } else if (argument == "--help") {
            std::cout << "parallel-mater-gallery [--scene file.glb] "
                         "[--dump-spheres N] [--headless output.ppm] "
                         "[--frames N]\n";
            std::exit(0);
        } else {
            return false;
        }
    }
    return true;
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

[[nodiscard]] parallel_mater::gallery::Camera
camera(const OrbitState &orbit, GalleryContext context) {
    parallel_mater::gallery::Camera result;
    result.target = context == GalleryContext::dump
                        ? parallel_mater::Vec3{0.5F, 2.2F, 0.0F}
                        : parallel_mater::Vec3{0.0F, 1.8F, 0.0F};
    const float horizontal = orbit.distance * std::cos(orbit.pitch);
    result.eye = {result.target.x + horizontal * std::sin(orbit.yaw),
                  result.target.y + orbit.distance * std::sin(orbit.pitch),
                  result.target.z + horizontal * std::cos(orbit.yaw)};
    return result;
}

void mouse_button(GLFWwindow *window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    input->orbit.dragging = action == GLFW_PRESS;
    glfwGetCursorPos(window, &input->orbit.previous_x,
                     &input->orbit.previous_y);
}

void cursor_position(GLFWwindow *window, double x, double y) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    OrbitState &orbit = input->orbit;
    if (orbit.dragging && !input->dump_dialog_visible) {
        orbit.yaw -= static_cast<float>(x - orbit.previous_x) * 0.006F;
        orbit.pitch = std::clamp(
            orbit.pitch + static_cast<float>(y - orbit.previous_y) * 0.006F,
            -1.35F, 1.35F);
    }
    orbit.previous_x = x;
    orbit.previous_y = y;
}

void scroll(GLFWwindow *window, double, double offset) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (input->dump_dialog_visible) {
        return;
    }
    input->orbit.distance = std::clamp(
        input->orbit.distance * std::exp(static_cast<float>(-offset) * 0.1F),
        3.0F, 30.0F);
}

void character_input(GLFWwindow *window, unsigned int codepoint) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (!input->dump_dialog_visible || codepoint < '0' || codepoint > '9') {
        return;
    }
    if (input->replace_dump_value) {
        input->dump_value.clear();
        input->replace_dump_value = false;
    }
    if (input->dump_value.size() < 4U) {
        input->dump_value.push_back(static_cast<char>(codepoint));
        input->dump_value_invalid = false;
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
                                 GalleryRuntime &output,
                                 std::string &error) {
    GalleryRuntime next{};
    next.context = context;
    if (context == GalleryContext::dump) {
        next.scene = parallel_mater::gallery::make_dump_scene(dump_spheres);
    } else if (context == GalleryContext::rigid_body) {
        if (!parallel_mater::gallery::load_glb_scene(options.scene, next.scene,
                                                      error)) {
            error = "scene load failed: " + error;
            return false;
        }
    } else {
        error = "fluid scene is not available";
        return false;
    }

    const std::size_t mesh_capacity =
        next.scene.meshes.size() + next.scene.collision_meshes.size();
    if (next.scene.rigid_bodies.empty() ||
        next.scene.rigid_bodies.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        mesh_capacity == 0U ||
        mesh_capacity > std::numeric_limits<std::uint32_t>::max()) {
        error = "scene exceeds world capacity range";
        return false;
    }
    const Status create_status = World::create(
        {.rigid_body_capacity =
             static_cast<std::uint32_t>(next.scene.rigid_bodies.size()),
         .triangle_mesh_capacity = static_cast<std::uint32_t>(mesh_capacity)},
        next.world);
    if (!create_status) {
        error = create_status.message != nullptr ? create_status.message
                                                 : "world creation failed";
        return false;
    }
    const Status instantiate_status = parallel_mater::gallery::instantiate_scene(
        next.scene, next.world, next.instance);
    if (!instantiate_status) {
        error = instantiate_status.message != nullptr
                    ? instantiate_status.message
                    : "scene instantiation failed";
        return false;
    }
    if (!OptixRenderer::create(next.scene, PARALLEL_MATER_OPTIX_PTX_PATH,
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

[[nodiscard]] bool reset_runtime(GalleryRuntime &runtime) {
    for (std::size_t index = 0; index < runtime.instance.rigid_bodies.size();
         ++index) {
        if (!require(runtime.world.set_rigid_body_state(
                         runtime.instance.rigid_bodies[index],
                         runtime.scene.rigid_bodies[index].options.initial_state),
                     "reset body")) {
            return false;
        }
    }
    if (runtime.kinematic_index < runtime.scene.rigid_bodies.size()) {
        runtime.kinematic_target =
            runtime.scene.rigid_bodies[runtime.kinematic_index]
                .options.initial_state;
    }
    return true;
}

[[nodiscard]] int context_index(GalleryContext context) {
    switch (context) {
    case GalleryContext::rigid_body: return 0;
    case GalleryContext::dump: return 1;
    case GalleryContext::fluid: return 2;
    }
    return 0;
}

[[nodiscard]] GalleryContext context_from_index(int index) {
    switch (std::clamp(index, 0, 2)) {
    case 1: return GalleryContext::dump;
    case 2: return GalleryContext::fluid;
    default: return GalleryContext::rigid_body;
    }
}

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
    GalleryRuntime runtime;
    if (!build_runtime(options, options.initial_context, options.dump_spheres,
                       runtime, error)) {
        std::cerr << error << '\n';
        return 1;
    }

    constexpr StepOptions step_options{.timestep = k_timestep,
                                       .substeps = 4U,
                                       .gravity = {0.0F, -9.81F, 0.0F}};
    std::vector<std::uint32_t> pixels;
    InputState input_state;
    if (!options.headless_output.empty()) {
        float headless_dump_angle = k_dump_initial_angle;
        for (int frame = 0; frame < options.frames; ++frame) {
            if (runtime.context == GalleryContext::dump &&
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
            if (!require(runtime.world.step(step_options),
                         "step headless gallery")) {
                return 1;
            }
        }
        if (!runtime.renderer.render(runtime.world, runtime.instance,
                                     camera(input_state.orbit, runtime.context),
                                     pixels, error)) {
            std::cerr << "Render failed: " << error << '\n';
            return 1;
        }
        if (!validate_render(pixels, runtime.renderer.width(),
                             runtime.renderer.height(), error)) {
            std::cerr << "Render validation failed: " << error << '\n';
            return 1;
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

    bool reset_was_down = false;
    bool timing_was_down = false;
    bool debug_was_down = false;
    bool tab_was_down = false;
    bool p_was_down = false;
    bool up_was_down = false;
    bool down_was_down = false;
    bool enter_was_down = false;
    bool backspace_was_down = false;
    bool escape_was_down = false;
    bool timing_visible = false;
    bool debug_visible = false;
    bool context_visible = false;
    GalleryContext context_selection = runtime.context;
    std::uint32_t dump_spheres = options.dump_spheres;
    float dump_angle = k_dump_initial_angle;
    WorldStepTimings timings{};

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        const bool escape_down =
            glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
        const bool enter_down =
            glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_KP_ENTER) == GLFW_PRESS;
        const bool backspace_down =
            glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS;
        const bool tab_down = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
        const bool p_down = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
        const bool up_down = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
        const bool down_down = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
        const bool reset_down = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        const bool timing_down = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
        const bool debug_down = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;

        if (input_state.dump_dialog_visible) {
            if (escape_down && !escape_was_down) {
                input_state.dump_dialog_visible = false;
                input_state.dump_value_invalid = false;
            }
            if (backspace_down && !backspace_was_down) {
                if (input_state.replace_dump_value) {
                    input_state.dump_value.clear();
                    input_state.replace_dump_value = false;
                } else if (!input_state.dump_value.empty()) {
                    input_state.dump_value.pop_back();
                }
                input_state.dump_value_invalid = false;
            }
            if (enter_down && !enter_was_down) {
                std::uint32_t requested = 0U;
                if (!parse_dump_count(input_state.dump_value, requested)) {
                    input_state.dump_value_invalid = true;
                } else {
                    GalleryRuntime replacement;
                    if (build_runtime(options, GalleryContext::dump, requested,
                                      replacement, error)) {
                        runtime = std::move(replacement);
                        dump_spheres = requested;
                        dump_angle = k_dump_initial_angle;
                        input_state.dump_dialog_visible = false;
                        input_state.dump_value_invalid = false;
                        timings = {};
                    } else {
                        std::cerr << "DUMP restart failed: " << error << '\n';
                        input_state.dump_value_invalid = true;
                    }
                }
            }
        } else {
            if (escape_down && !escape_was_down) {
                if (context_visible) {
                    context_visible = false;
                } else {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
            }
            if (tab_down && !tab_was_down) {
                context_visible = !context_visible;
                context_selection = runtime.context;
            }
            if (context_visible) {
                int selected = context_index(context_selection);
                if (up_down && !up_was_down) {
                    selected = std::max(0, selected - 1);
                }
                if (down_down && !down_was_down) {
                    selected = std::min(2, selected + 1);
                }
                context_selection = context_from_index(selected);
                if (enter_down && !enter_was_down &&
                    context_selection != GalleryContext::fluid) {
                    GalleryRuntime replacement;
                    if (context_selection == runtime.context ||
                        build_runtime(options, context_selection, dump_spheres,
                                      replacement, error)) {
                        if (context_selection != runtime.context) {
                            runtime = std::move(replacement);
                            dump_angle = k_dump_initial_angle;
                            timings = {};
                        }
                        context_visible = false;
                    } else {
                        std::cerr << "Scene switch failed: " << error << '\n';
                    }
                }
            } else if (runtime.context == GalleryContext::dump && p_down &&
                       !p_was_down) {
                input_state.dump_dialog_visible = true;
                input_state.dump_value = std::to_string(dump_spheres);
                input_state.replace_dump_value = true;
                input_state.dump_value_invalid = false;
            }
            if (!context_visible && reset_down && !reset_was_down) {
                if (!reset_runtime(runtime)) {
                    break;
                }
                if (runtime.context == GalleryContext::dump) {
                    dump_angle = k_dump_initial_angle;
                }
            }
            if (timing_down && !timing_was_down) {
                timing_visible = !timing_visible;
            }
            if (debug_down && !debug_was_down) {
                debug_visible = !debug_visible;
            }
        }

        reset_was_down = reset_down;
        timing_was_down = timing_down;
        debug_was_down = debug_down;
        tab_was_down = tab_down;
        p_was_down = p_down;
        up_was_down = up_down;
        down_was_down = down_down;
        enter_was_down = enter_down;
        backspace_was_down = backspace_down;
        escape_was_down = escape_down;

        if (!input_state.dump_dialog_visible) {
            const DirectionalInput directional =
                context_visible ? DirectionalInput{} : directional_input(window);
            if (runtime.kinematic_index < runtime.instance.rigid_bodies.size()) {
                if (runtime.context == GalleryContext::dump) {
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
            if (runtime.context == GalleryContext::rigid_body) {
                interactive_step.gravity = gravity_for(directional);
            }
            interactive_step.collect_kernel_timings = timing_visible;
            interactive_step.collect_rigid_contacts = debug_visible;
            if (!require(runtime.world.step(interactive_step), "step gallery")) {
                break;
            }
            if (timing_visible &&
                !require(runtime.world.collect_step_timings(timings),
                         "collect timings")) {
                break;
            }
        }

        const Camera current_camera = camera(input_state.orbit, runtime.context);
        if (!runtime.renderer.render(runtime.world, runtime.instance,
                                     current_camera, pixels, error)) {
            std::cerr << "Render failed: " << error << '\n';
            break;
        }
        if (debug_visible && !draw_rigid_contact_overlay(
                                 pixels, runtime.renderer.width(),
                                 runtime.renderer.height(),
                                 runtime.world.rigid_contacts(), current_camera,
                                 error)) {
            std::cerr << "Debug overlay failed: " << error << '\n';
            break;
        }
        if (timing_visible) {
            draw_timing_overlay(pixels, runtime.renderer.width(),
                                runtime.renderer.height(), timings);
        }
        if (context_visible) {
            draw_context_overlay(pixels, runtime.renderer.width(),
                                 runtime.renderer.height(), context_selection);
        }
        if (input_state.dump_dialog_visible) {
            draw_dump_count_overlay(
                pixels, runtime.renderer.width(), runtime.renderer.height(),
                input_state.dump_value, input_state.dump_value_invalid);
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
