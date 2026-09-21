// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path scene{PARALLEL_MATER_DEFAULT_SCENE_PATH};
    std::filesystem::path headless_output{};
    int frames{240};
    std::uint32_t width{960U};
    std::uint32_t height{720U};
};

struct OrbitState {
    float yaw{0.62F};
    float pitch{0.32F};
    float distance{11.0F};
    bool dragging{};
    double previous_x{};
    double previous_y{};
};

struct DirectionalInput {
    float x{};
    float z{};
};

constexpr float k_timestep = 1.0F / 60.0F;
constexpr float k_kinematic_speed = 2.0F;
constexpr float k_gravity = 9.81F;
constexpr float k_gravity_tilt_tangent = 0.577350269F; // 30 degrees.

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
    char *end = nullptr;
    const long parsed = std::strtol(value.data(), &end, 10);
    if (end != value.data() + value.size() || parsed <= 0 || parsed > 100'000) {
        return false;
    }
    output = static_cast<int>(parsed);
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
        } else if (argument == "--help") {
            std::cout << "parallel-mater-gallery [--scene file.glb] "
                         "[--headless output.ppm] [--frames N]\n";
            std::exit(0);
        } else {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool require(parallel_mater::Status status,
                           const char *operation) {
    if (status) {
        return true;
    }
    std::cerr << operation << " failed: "
              << (status.message != nullptr ? status.message : "unknown") << '\n';
    return false;
}

[[nodiscard]] parallel_mater::gallery::Camera camera(const OrbitState &orbit) {
    parallel_mater::gallery::Camera result;
    result.target = {0.0F, 1.8F, 0.0F};
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
    auto *orbit = static_cast<OrbitState *>(glfwGetWindowUserPointer(window));
    orbit->dragging = action == GLFW_PRESS;
    glfwGetCursorPos(window, &orbit->previous_x, &orbit->previous_y);
}

void cursor_position(GLFWwindow *window, double x, double y) {
    auto *orbit = static_cast<OrbitState *>(glfwGetWindowUserPointer(window));
    if (orbit->dragging) {
        orbit->yaw -= static_cast<float>(x - orbit->previous_x) * 0.006F;
        orbit->pitch = std::clamp(
            orbit->pitch + static_cast<float>(y - orbit->previous_y) * 0.006F,
            -1.35F, 1.35F);
    }
    orbit->previous_x = x;
    orbit->previous_y = y;
}

void scroll(GLFWwindow *window, double, double offset) {
    auto *orbit = static_cast<OrbitState *>(glfwGetWindowUserPointer(window));
    orbit->distance =
        std::clamp(orbit->distance * std::exp(static_cast<float>(-offset) * 0.1F),
                   3.0F, 30.0F);
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

[[nodiscard]] bool reset_scene(
    const parallel_mater::gallery::SceneDefinition &definition,
    const parallel_mater::gallery::SceneInstance &instance,
    parallel_mater::World &world) {
    for (std::size_t index = 0; index < instance.rigid_bodies.size(); ++index) {
        if (!require(world.set_rigid_body_state(
                         instance.rigid_bodies[index],
                         definition.rigid_bodies[index].options.initial_state),
                     "reset body")) {
            return false;
        }
    }
    return true;
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
    SceneDefinition scene;
    std::string error;
    if (!load_glb_scene(options.scene, scene, error)) {
        std::cerr << "Scene load failed: " << error << '\n';
        return 1;
    }
    World world;
    if (!require(World::create(
                     {.rigid_body_capacity =
                          static_cast<std::uint32_t>(scene.rigid_bodies.size())},
                     world),
                 "create world")) {
        return 1;
    }
    SceneInstance instance;
    if (!require(instantiate_scene(scene, world, instance), "instantiate scene")) {
        return 1;
    }
    OptixRenderer renderer;
    if (!OptixRenderer::create(scene, PARALLEL_MATER_OPTIX_PTX_PATH,
                               options.width, options.height, renderer, error)) {
        std::cerr << "Renderer creation failed: " << error << '\n';
        return 1;
    }

    constexpr StepOptions step_options{.timestep = k_timestep,
                                       .substeps = 4U,
                                       .gravity = {0.0F, -9.81F, 0.0F}};
    std::vector<std::uint32_t> pixels;
    OrbitState orbit;
    if (!options.headless_output.empty()) {
        for (int frame = 0; frame < options.frames; ++frame) {
            if (!require(world.step(step_options), "step headless gallery")) {
                return 1;
            }
        }
        if (!renderer.render(world, instance, camera(orbit), pixels, error)) {
            std::cerr << "Render failed: " << error << '\n';
            return 1;
        }
        if (!validate_render(pixels, renderer.width(), renderer.height(), error)) {
            std::cerr << "Render validation failed: " << error << '\n';
            return 1;
        }
        if (!write_ppm(options.headless_output, pixels, renderer.width(),
                       renderer.height())) {
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
        "ParallelMater — Blender-authored rigid scene", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "GLFW window creation failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(window, &orbit);
    glfwSetMouseButtonCallback(window, &mouse_button);
    glfwSetCursorPosCallback(window, &cursor_position);
    glfwSetScrollCallback(window, &scroll);
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    std::size_t kinematic_index = scene.rigid_bodies.size();
    RigidBodyState kinematic_target{};
    for (std::size_t index = 0; index < scene.rigid_bodies.size(); ++index) {
        if (scene.rigid_bodies[index].options.motion == MotionType::kinematic) {
            kinematic_index = index;
            kinematic_target = scene.rigid_bodies[index].options.initial_state;
            break;
        }
    }

    bool reset_was_down = false;
    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        const bool reset_down = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        if (reset_down && !reset_was_down && !reset_scene(scene, instance, world)) {
            break;
        }
        if (reset_down && !reset_was_down &&
            kinematic_index < scene.rigid_bodies.size()) {
            kinematic_target =
                scene.rigid_bodies[kinematic_index].options.initial_state;
        }
        reset_was_down = reset_down;

        const DirectionalInput input = directional_input(window);
        if (kinematic_index < instance.rigid_bodies.size()) {
            kinematic_target.position.x +=
                input.x * k_kinematic_speed * k_timestep;
            kinematic_target.position.z +=
                input.z * k_kinematic_speed * k_timestep;
            if (!require(world.set_kinematic_target(
                             instance.rigid_bodies[kinematic_index],
                             kinematic_target),
                         "move kinematic body")) {
                break;
            }
        }
        StepOptions interactive_step = step_options;
        interactive_step.gravity = gravity_for(input);
        if (!require(world.step(interactive_step), "step gallery")) {
            break;
        }
        if (!renderer.render(world, instance, camera(orbit), pixels, error)) {
            std::cerr << "Render failed: " << error << '\n';
            break;
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
