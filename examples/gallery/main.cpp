// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/scene.hpp>

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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using parallel_mater::RigidBodyState;
using parallel_mater::Status;
using parallel_mater::World;
using parallel_mater::gallery::CameraController;
using parallel_mater::gallery::CameraDragMode;
using parallel_mater::gallery::CameraPreset;
using parallel_mater::gallery::GalleryContext;
using parallel_mater::gallery::is_fluid_context;
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
constexpr std::uint32_t k_minimum_fluid_particles = 100U;
constexpr std::uint32_t k_maximum_fluid_particles = 100'000U;
constexpr std::uint32_t k_default_fluid_particles = 30'000U;

struct Options {
    std::filesystem::path scene{PARALLEL_MATER_DEFAULT_SCENE_PATH};
    std::filesystem::path headless_output{};
    int frames{240};
    std::uint32_t width{960U};
    std::uint32_t height{720U};
    GalleryContext initial_context{GalleryContext::rigid_body};
    std::uint32_t dump_spheres{k_default_dump_spheres};
    std::uint32_t fluid_particles{k_default_fluid_particles};
    bool fluid_particle_view{};
    bool trace_fluid_escapes{};
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

[[nodiscard]] parallel_mater::Vec3 rotate_point(
    parallel_mater::Quaternion q, parallel_mater::Vec3 p) {
    const parallel_mater::Vec3 t{
        2.0F * (q.y * p.z - q.z * p.y),
        2.0F * (q.z * p.x - q.x * p.z),
        2.0F * (q.x * p.y - q.y * p.x)};
    return {p.x + q.w * t.x + q.y * t.z - q.z * t.y,
            p.y + q.w * t.y + q.z * t.x - q.x * t.z,
            p.z + q.w * t.z + q.x * t.y - q.y * t.x};
}

[[nodiscard]] bool passive_mesh_bounds(const SceneDefinition &scene,
                                       parallel_mater::Vec3 &minimum,
                                       parallel_mater::Vec3 &maximum) {
    minimum = {std::numeric_limits<float>::max(),
               std::numeric_limits<float>::max(),
               std::numeric_limits<float>::max()};
    maximum = {-std::numeric_limits<float>::max(),
               -std::numeric_limits<float>::max(),
               -std::numeric_limits<float>::max()};
    bool found = false;
    for (const auto &body : scene.rigid_bodies) {
        if (body.options.motion != parallel_mater::MotionType::static_body)
            continue;
        const auto &meshes = body.collision_mesh_indices.empty()
            ? scene.meshes : scene.collision_meshes;
        const auto &indices = body.collision_mesh_indices.empty()
            ? body.mesh_indices : body.collision_mesh_indices;
        for (std::uint32_t mesh_index : indices) {
            for (const auto &vertex : meshes[mesh_index].vertices) {
                const auto local = rotate_point(
                    body.options.initial_state.orientation, vertex.position);
                const auto center = body.options.initial_state.position;
                const parallel_mater::Vec3 p{
                    center.x + local.x, center.y + local.y, center.z + local.z};
                minimum.x = std::min(minimum.x, p.x);
                minimum.y = std::min(minimum.y, p.y);
                minimum.z = std::min(minimum.z, p.z);
                maximum.x = std::max(maximum.x, p.x);
                maximum.y = std::max(maximum.y, p.y);
                maximum.z = std::max(maximum.z, p.z);
                found = true;
            }
        }
    }
    return found;
}

[[nodiscard]] std::optional<float> passive_surface_height(
    const SceneDefinition &scene, parallel_mater::Vec3 position,
    float *normal_y = nullptr) {
    std::optional<float> highest;
    for (const auto &body : scene.rigid_bodies) {
        if (body.options.motion != parallel_mater::MotionType::static_body)
            continue;
        const auto &meshes = body.collision_mesh_indices.empty()
            ? scene.meshes : scene.collision_meshes;
        const auto &indices = body.collision_mesh_indices.empty()
            ? body.mesh_indices : body.collision_mesh_indices;
        const auto world_vertex = [&](parallel_mater::Vec3 local) {
            const auto rotated = rotate_point(
                body.options.initial_state.orientation, local);
            const auto origin = body.options.initial_state.position;
            return parallel_mater::Vec3{origin.x + rotated.x,
                                        origin.y + rotated.y,
                                        origin.z + rotated.z};
        };
        for (std::uint32_t mesh_index : indices) {
            const auto &mesh = meshes[mesh_index];
            for (std::size_t triangle = 0; triangle < mesh.indices.size();
                 triangle += 3U) {
                const auto a = world_vertex(
                    mesh.vertices[mesh.indices[triangle]].position);
                const auto b = world_vertex(
                    mesh.vertices[mesh.indices[triangle + 1U]].position);
                const auto c = world_vertex(
                    mesh.vertices[mesh.indices[triangle + 2U]].position);
                const float abx = b.x - a.x, abz = b.z - a.z;
                const float acx = c.x - a.x, acz = c.z - a.z;
                const float determinant = abx * acz - abz * acx;
                if (std::fabs(determinant) < 1.0e-8F) continue;
                const float px = position.x - a.x, pz = position.z - a.z;
                const float u = (px * acz - pz * acx) / determinant;
                const float v = (abx * pz - abz * px) / determinant;
                if (u < -1.0e-4F || v < -1.0e-4F ||
                    u + v > 1.0001F) continue;
                const float height = a.y + u * (b.y - a.y) + v * (c.y - a.y);
                if (!highest || height > *highest) {
                    highest = height;
                    if (normal_y != nullptr) {
                        const float ny = abz * acx - abx * acz;
                        const float nx = (b.y - a.y) * acz -
                                         (c.y - a.y) * abz;
                        const float nz = abx * (c.y - a.y) -
                                         acx * (b.y - a.y);
                        *normal_y = ny / std::sqrt(nx * nx + ny * ny + nz * nz);
                    }
                }
            }
        }
    }
    return highest;
}

struct PassiveFloorIndex {
    struct Triangle {
        parallel_mater::Vec3 a{}, b{}, c{};
    };
    float minimum_x{}, minimum_z{};
    float cell_size{0.35F};
    std::size_t columns{}, rows{};
    std::vector<Triangle> triangles{};
    std::vector<std::vector<std::uint32_t>> cells{};

    [[nodiscard]] std::optional<float> height(float x, float z) const {
        const int column = static_cast<int>(std::floor((x - minimum_x) /
                                                        cell_size));
        const int row = static_cast<int>(std::floor((z - minimum_z) /
                                                     cell_size));
        if (column < 0 || row < 0 ||
            static_cast<std::size_t>(column) >= columns ||
            static_cast<std::size_t>(row) >= rows) return std::nullopt;
        std::optional<float> lowest;
        for (const std::uint32_t index :
             cells[static_cast<std::size_t>(row) * columns + column]) {
            const auto &triangle = triangles[index];
            const auto a = triangle.a, b = triangle.b, c = triangle.c;
            const float abx = b.x - a.x, abz = b.z - a.z;
            const float acx = c.x - a.x, acz = c.z - a.z;
            const float determinant = abx * acz - abz * acx;
            const float px = x - a.x, pz = z - a.z;
            const float u = (px * acz - pz * acx) / determinant;
            const float v = (abx * pz - abz * px) / determinant;
            if (u < -1.0e-4F || v < -1.0e-4F || u + v > 1.0001F)
                continue;
            const float y = a.y + u * (b.y - a.y) + v * (c.y - a.y);
            if (!lowest || y < *lowest) lowest = y;
        }
        return lowest;
    }
};

[[nodiscard]] PassiveFloorIndex build_passive_floor_index(
    const SceneDefinition &scene, parallel_mater::Vec3 minimum,
    parallel_mater::Vec3 maximum) {
    PassiveFloorIndex index{};
    index.minimum_x = minimum.x;
    index.minimum_z = minimum.z;
    index.columns = static_cast<std::size_t>(
        std::ceil((maximum.x - minimum.x) / index.cell_size)) + 1U;
    index.rows = static_cast<std::size_t>(
        std::ceil((maximum.z - minimum.z) / index.cell_size)) + 1U;
    index.cells.resize(index.columns * index.rows);
    for (const auto &body : scene.rigid_bodies) {
        if (body.options.motion != parallel_mater::MotionType::static_body)
            continue;
        const auto &meshes = body.collision_mesh_indices.empty()
            ? scene.meshes : scene.collision_meshes;
        const auto &mesh_indices = body.collision_mesh_indices.empty()
            ? body.mesh_indices : body.collision_mesh_indices;
        const auto world_vertex = [&](parallel_mater::Vec3 local) {
            const auto p = rotate_point(body.options.initial_state.orientation,
                                        local);
            const auto origin = body.options.initial_state.position;
            return parallel_mater::Vec3{p.x + origin.x, p.y + origin.y,
                                        p.z + origin.z};
        };
        for (const std::uint32_t mesh_index : mesh_indices) {
            const auto &mesh = meshes[mesh_index];
            for (std::size_t offset = 0; offset < mesh.indices.size();
                 offset += 3U) {
                const auto a = world_vertex(
                    mesh.vertices[mesh.indices[offset]].position);
                const auto b = world_vertex(
                    mesh.vertices[mesh.indices[offset + 1U]].position);
                const auto c = world_vertex(
                    mesh.vertices[mesh.indices[offset + 2U]].position);
                const float nx = (b.y - a.y) * (c.z - a.z) -
                                 (b.z - a.z) * (c.y - a.y);
                const float ny = (b.z - a.z) * (c.x - a.x) -
                                 (b.x - a.x) * (c.z - a.z);
                const float nz = (b.x - a.x) * (c.y - a.y) -
                                 (b.y - a.y) * (c.x - a.x);
                const float normal_length = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (normal_length < 1.0e-8F ||
                    std::fabs(ny) < normal_length * 0.2F) continue;
                const std::uint32_t triangle = static_cast<std::uint32_t>(
                    index.triangles.size());
                index.triangles.push_back({a, b, c});
                const float left = std::min({a.x, b.x, c.x});
                const float right = std::max({a.x, b.x, c.x});
                const float front = std::min({a.z, b.z, c.z});
                const float back = std::max({a.z, b.z, c.z});
                const auto first_column = static_cast<std::size_t>(
                    std::max(0.0F, std::floor((left - minimum.x) /
                                                index.cell_size)));
                const auto last_column = std::min(index.columns - 1U,
                    static_cast<std::size_t>(std::max(0.0F,
                        std::floor((right - minimum.x) / index.cell_size))));
                const auto first_row = static_cast<std::size_t>(
                    std::max(0.0F, std::floor((front - minimum.z) /
                                                index.cell_size)));
                const auto last_row = std::min(index.rows - 1U,
                    static_cast<std::size_t>(std::max(0.0F,
                        std::floor((back - minimum.z) / index.cell_size))));
                for (std::size_t row = first_row; row <= last_row; ++row)
                    for (std::size_t column = first_column;
                         column <= last_column; ++column)
                        index.cells[row * index.columns + column].push_back(
                            triangle);
            }
        }
    }
    return index;
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
                                       const SceneDefinition &scene,
                                       const PassiveFloorIndex &floor,
                                       parallel_mater::Vec3 mesh_minimum,
                                       parallel_mater::Vec3 mesh_maximum,
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
    const float threshold = mesh_minimum.y;
    std::uint32_t below = 0U;
    std::uint32_t below_local = 0U;
    std::size_t first = positions.size();
    std::size_t first_local = positions.size();
    float first_local_surface = 0.0F;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        trace.lowest_y = std::min(trace.lowest_y, positions[i].y);
        const auto surface = floor.height(positions[i].x, positions[i].z);
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
            const auto surface = passive_surface_height(scene, sample,
                                                         &normal_y);
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
            if (!parse_count(argv[++index], k_minimum_dump_spheres,
                             k_maximum_dump_spheres, output.dump_spheres)) {
                return false;
            }
            output.initial_context = GalleryContext::dump;
        } else if (argument == "--fluid-particles" && index + 1 < argc) {
            if (!parse_count(argv[++index], k_minimum_fluid_particles,
                             k_maximum_fluid_particles,
                             output.fluid_particles)) return false;
            if (output.initial_context != GalleryContext::fluid_rigid)
                output.initial_context = GalleryContext::fluid;
        } else if (argument == "--fluid") {
            output.initial_context = GalleryContext::fluid;
        } else if (argument == "--fluid-rigid") {
            output.initial_context = GalleryContext::fluid_rigid;
        } else if (argument == "--fluid-particle-view") {
            if (output.initial_context != GalleryContext::fluid_rigid)
                output.initial_context = GalleryContext::fluid;
            output.fluid_particle_view = true;
        } else if (argument == "--trace-fluid-escapes") {
            if (output.initial_context != GalleryContext::fluid_rigid)
                output.initial_context = GalleryContext::fluid;
            output.trace_fluid_escapes = true;
        } else if (argument == "--help") {
            std::cout << "parallel-mater-gallery [--scene file.glb] "
                         "[--dump-spheres N] [--fluid|--fluid-rigid] "
                         "[--fluid-particles N] "
                         "[--fluid-particle-view] [--trace-fluid-escapes] "
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

[[nodiscard]] CameraPreset camera_preset(GalleryContext context) {
    if (context == GalleryContext::dump)
        return {.target = {0.5F, 2.2F, 0.0F}};
    if (is_fluid_context(context))
        return {.target = {-2.0F, -1.0F, -2.0F},
                .distance_scale = 1.6F};
    return {};
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
    if (context == GalleryContext::dump) {
        next.scene = parallel_mater::gallery::make_dump_scene(dump_spheres);
    } else if (context == GalleryContext::rigid_body ||
               is_fluid_context(context)) {
        const std::filesystem::path scene_path =
            context == GalleryContext::fluid
                ? std::filesystem::path(PARALLEL_MATER_FLUID_SCENE_PATH)
                : context == GalleryContext::fluid_rigid
                    ? std::filesystem::path(PARALLEL_MATER_FLUID_RIGID_SCENE_PATH)
                    : options.scene;
        if (!parallel_mater::gallery::load_glb_scene(scene_path, next.scene,
                                                      error)) {
            error = "scene load failed: " + error;
            return false;
        }
        if (is_fluid_context(context))
            next.scene.fluid_options.capacity = fluid_particles;
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

[[nodiscard]] int context_index(GalleryContext context) {
    switch (context) {
    case GalleryContext::rigid_body: return 0;
    case GalleryContext::dump: return 1;
    case GalleryContext::fluid: return 2;
    case GalleryContext::fluid_rigid: return 3;
    }
    return 0;
}

[[nodiscard]] GalleryContext context_from_index(int index) {
    switch (std::clamp(index, 0, 3)) {
    case 1: return GalleryContext::dump;
    case 2: return GalleryContext::fluid;
    case 3: return GalleryContext::fluid_rigid;
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
                       options.fluid_particles,
                       runtime, error)) {
        std::cerr << error << '\n';
        return 1;
    }

    constexpr StepOptions step_options{.timestep = k_timestep,
                                       .substeps = 4U,
                                       .gravity = {0.0F, -9.81F, 0.0F}};
    std::vector<std::uint32_t> pixels;
    InputState input_state;
    input_state.camera.set_preset(camera_preset(runtime.context));
    if (!options.headless_output.empty()) {
        parallel_mater::Vec3 passive_minimum{}, passive_maximum{};
        FluidEscapeTrace escape_trace{};
        PassiveFloorIndex floor_index{};
        if (options.trace_fluid_escapes) {
            if (!runtime.instance.has_fluid ||
                !passive_mesh_bounds(runtime.scene, passive_minimum,
                                     passive_maximum)) {
                std::cerr << "Fluid escape trace needs a passive collider\n";
                return 1;
            }
            for (const auto &body : runtime.scene.rigid_bodies) {
                if (body.options.motion != parallel_mater::MotionType::static_body)
                    continue;
                std::cout << "Passive body " << body.name << " collision_meshes="
                          << body.collision_mesh_indices.size() << " render_meshes="
                          << body.mesh_indices.size() << '\n';
            }
            std::cout << "Passive mesh bounds x=" << passive_minimum.x << ".."
                      << passive_maximum.x << " y=" << passive_minimum.y << ".."
                      << passive_maximum.y << " z=" << passive_minimum.z << ".."
                      << passive_maximum.z << '\n';
            floor_index = build_passive_floor_index(
                runtime.scene, passive_minimum, passive_maximum);
            std::cout << "Passive floor triangles="
                      << floor_index.triangles.size() << '\n';
            for (const auto &spawn : runtime.scene.spawn_planes) {
                std::cout << "Fluid inflow center=(" << spawn.plane.center.x
                          << ',' << spawn.plane.center.y << ','
                          << spawn.plane.center.z << ") half_extents=("
                          << spawn.plane.half_extents.x << ','
                          << spawn.plane.half_extents.y << ") velocity=("
                          << spawn.initial_velocity.x << ','
                          << spawn.initial_velocity.y << ','
                          << spawn.initial_velocity.z << ")\n";
            }
        }
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
            if (options.trace_fluid_escapes &&
                !trace_fluid_escapes(runtime.world, runtime.instance,
                                     runtime.scene, floor_index,
                                     passive_minimum, passive_maximum, frame + 1,
                                     escape_trace)) return 1;
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
                      << " outflowed=" << statistics.destroyed_particle_count
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
        RendererTimings headless_render_timings{};
        if (!runtime.renderer.render(runtime.world, runtime.instance,
                                     input_state.camera.camera(),
                                     pixels, error, &headless_render_timings,
                                     options.fluid_particle_view
                                         ? FluidRenderMode::particles
                                         : FluidRenderMode::surface)) {
            std::cerr << "Render failed: " << error << '\n';
            return 1;
        }
        if (runtime.instance.has_fluid && !options.fluid_particle_view)
            std::cout << "Fluid surface outliers="
                      << headless_render_timings.surface_excluded_particle_count
                      << " surface_gpu_ms="
                      << headless_render_timings.surface_gpu_milliseconds << '\n';
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
    bool fluid_particle_view = options.fluid_particle_view;
    bool context_visible = false;
    GalleryContext context_selection = runtime.context;
    std::uint32_t dump_spheres = options.dump_spheres;
    std::uint32_t fluid_particles = options.fluid_particles;
    float dump_angle = k_dump_initial_angle;
    WorldStepTimings timings{};
    WorldStatistics statistics{};
    RendererTimings renderer_timings{};

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

        if (input_state.count_dialog_visible) {
            if (escape_down && !escape_was_down) {
                input_state.count_dialog_visible = false;
                input_state.count_value_invalid = false;
            }
            if (backspace_down && !backspace_was_down) {
                if (input_state.replace_count_value) {
                    input_state.count_value.clear();
                    input_state.replace_count_value = false;
                } else if (!input_state.count_value.empty()) {
                    input_state.count_value.pop_back();
                }
                input_state.count_value_invalid = false;
            }
            if (enter_down && !enter_was_down) {
                std::uint32_t requested = 0U;
                const bool fluid_dialog = is_fluid_context(runtime.context);
                const std::uint32_t minimum = fluid_dialog
                    ? k_minimum_fluid_particles : k_minimum_dump_spheres;
                const std::uint32_t maximum = fluid_dialog
                    ? k_maximum_fluid_particles : k_maximum_dump_spheres;
                if (!parse_count(input_state.count_value, minimum, maximum,
                                 requested)) {
                    input_state.count_value_invalid = true;
                } else {
                    GalleryRuntime replacement;
                    if (build_runtime(options, runtime.context,
                                      fluid_dialog ? dump_spheres : requested,
                                      fluid_dialog ? requested : fluid_particles,
                                      replacement, error)) {
                        runtime = std::move(replacement);
                        if (fluid_dialog) fluid_particles = requested;
                        else dump_spheres = requested;
                        dump_angle = k_dump_initial_angle;
                        input_state.count_dialog_visible = false;
                        input_state.count_value_invalid = false;
                        timings = {};
                    } else {
                        std::cerr << "Scene restart failed: " << error << '\n';
                        input_state.count_value_invalid = true;
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
                    selected = std::min(3, selected + 1);
                }
                context_selection = context_from_index(selected);
                if (enter_down && !enter_was_down) {
                    GalleryRuntime replacement;
                    if (context_selection == runtime.context ||
                        build_runtime(options, context_selection, dump_spheres,
                                      fluid_particles,
                                      replacement, error)) {
                        if (context_selection != runtime.context) {
                            runtime = std::move(replacement);
                            input_state.camera.set_preset(
                                camera_preset(runtime.context));
                            dump_angle = k_dump_initial_angle;
                            timings = {};
                        }
                        context_visible = false;
                    } else {
                        std::cerr << "Scene switch failed: " << error << '\n';
                    }
                }
            } else if (runtime.context != GalleryContext::rigid_body &&
                       p_down && !p_was_down) {
                input_state.count_dialog_visible = true;
                input_state.count_value = std::to_string(
                    is_fluid_context(runtime.context)
                        ? fluid_particles : dump_spheres);
                input_state.replace_count_value = true;
                input_state.count_value_invalid = false;
            }
            if (!context_visible && reset_down && !reset_was_down) {
                GalleryRuntime replacement;
                if (build_runtime(options, runtime.context, dump_spheres,
                                  fluid_particles, replacement, error)) {
                    runtime = std::move(replacement);
                    dump_angle = k_dump_initial_angle;
                    timings = {};
                    statistics = {};
                    renderer_timings = {};
                } else {
                    std::cerr << "Scene reset failed: " << error << '\n';
                }
            }
            if (timing_down && !timing_was_down) {
                timing_visible = !timing_visible;
            }
            if (debug_down && !debug_was_down) {
                if (is_fluid_context(runtime.context))
                    fluid_particle_view = !fluid_particle_view;
                else
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

        if (!input_state.count_dialog_visible) {
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
            interactive_step.collect_rigid_contacts =
                debug_visible && !is_fluid_context(runtime.context);
            interactive_step.collect_fluid_contacts =
                timing_visible && is_fluid_context(runtime.context);
            if (!require(runtime.world.step(interactive_step), "step gallery")) {
                break;
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
                                     fluid_particle_view
                                         ? FluidRenderMode::particles
                                         : FluidRenderMode::surface)) {
            std::cerr << "Render failed: " << error << '\n';
            break;
        }
        if (debug_visible && !is_fluid_context(runtime.context) &&
            !draw_rigid_contact_overlay(
                                 pixels, runtime.renderer.width(),
                                 runtime.renderer.height(),
                                 runtime.world.rigid_contacts(), current_camera,
                                 error)) {
            std::cerr << "Debug overlay failed: " << error << '\n';
            break;
        }
        if (timing_visible) {
            if (is_fluid_context(runtime.context))
                draw_fluid_timing_overlay(
                    pixels, runtime.renderer.width(), runtime.renderer.height(),
                    timings, renderer_timings, statistics, fluid_particles);
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
