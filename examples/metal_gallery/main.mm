// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include "parallel_mater_gallery_build_info.hpp"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>
#include <sys/sysctl.h>

namespace {

#include "parallel_mater_gallery_metallib.inc"

using parallel_mater::Quaternion;
using parallel_mater::Vec3;
using parallel_mater::gallery::Camera;
using parallel_mater::gallery::CameraController;
using parallel_mater::gallery::CameraDragMode;
using parallel_mater::gallery::CameraPreset;
using parallel_mater::gallery::GalleryControlPolicy;
using parallel_mater::gallery::GalleryCountKind;
using parallel_mater::gallery::GalleryContext;
using parallel_mater::gallery::GalleryColor;
using parallel_mater::gallery::GalleryEntry;
using parallel_mater::gallery::GallerySceneSource;
using parallel_mater::gallery::gallery_context_index;
using parallel_mater::gallery::gallery_entries;
using parallel_mater::gallery::gallery_entry;
using parallel_mater::gallery::screen_space_gravity;
using parallel_mater::gallery::peg_paint_gravity_tilt_degrees;
using parallel_mater::gallery::steer_gravity;
using parallel_mater::gallery::toggles_constraint;
using parallel_mater::gallery::uses_rigid_gravity;
using parallel_mater::gallery::BrickSceneConfig;
using parallel_mater::gallery::brick_minimum_count;
using parallel_mater::gallery::brick_maximum_count;
using parallel_mater::gallery::brick_minimum_scale;
using parallel_mater::gallery::brick_maximum_scale;
using parallel_mater::gallery::brick_maximum_planes;
using parallel_mater::gallery::brick_render_width;
using parallel_mater::gallery::brick_render_height;
using parallel_mater::gallery::validate_brick_config;
using parallel_mater::metal::BufferSpan;
using parallel_mater::metal::ClothDeviceView;
using parallel_mater::metal::FluidDeviceView;
using parallel_mater::metal::FrameToken;
using parallel_mater::metal::MotionType;
using parallel_mater::metal::PaintFieldDeviceView;
using parallel_mater::metal::RigidBodyId;
using parallel_mater::metal::RigidBodyDeviceView;
using parallel_mater::metal::RigidConstraintId;
using parallel_mater::metal::RigidConstraintOptions;
using parallel_mater::metal::RigidConstraintState;
using parallel_mater::metal::RigidConstraintType;
using parallel_mater::metal::RigidContactEvent;
using parallel_mater::metal::RigidBodyState;
using parallel_mater::metal::RopeDeviceView;
using parallel_mater::metal::SmokeDeviceView;
using parallel_mater::metal::SoftBodyDeviceView;
using parallel_mater::metal::Status;
using parallel_mater::metal::World;
using parallel_mater::metal::WorldOptions;
using parallel_mater::metal::WorldStatistics;
using MetalScene = parallel_mater::metal::gallery::SceneDefinition;
using MetalInstance = parallel_mater::metal::gallery::SceneInstance;
using MetalMesh = parallel_mater::metal::gallery::TriangleMesh;

constexpr float k_timestep = 1.0F / 60.0F;
constexpr float k_kinematic_speed = 2.0F;
constexpr float k_gravity = 9.81F;
constexpr float k_cloth_gravity_tilt_degrees = 45.0F;
constexpr float k_rigid_gravity_tilt_degrees = 30.0F;
constexpr float k_collector_gravity_tilt_degrees =
    parallel_mater::gallery::collector_gravity_tilt_degrees;
constexpr float k_pi = 3.14159265358979323846F;
constexpr float k_dump_initial_angle = k_pi * 0.25F;
constexpr float k_dump_final_angle = -k_pi * 0.25F;
constexpr float k_dump_rotation_speed = k_pi * 0.25F;
constexpr float k_motor_speed = 8.0F;
constexpr std::uint32_t k_default_dump_spheres = 100U;
constexpr std::uint32_t k_default_fluid_particles = 30'000U;

CameraPreset camera_preset_for(GalleryContext context,
                               BrickSceneConfig bricks) {
    return context == GalleryContext::rigid_body
        ? parallel_mater::gallery::brick_camera_preset(bricks)
        : gallery_entry(context).camera;
}

struct Options {
    std::filesystem::path scene{PARALLEL_MATER_DEFAULT_SCENE_PATH};
    std::filesystem::path output{};
    GalleryContext context{GalleryContext::rigid_body};
    std::uint32_t frames{240U};
    std::uint32_t width{brick_render_width};
    std::uint32_t height{brick_render_height};
    BrickSceneConfig bricks{};
    std::filesystem::path profiles_file{PARALLEL_MATER_DEVICE_PROFILES_PATH};
    std::uint32_t dump_spheres{k_default_dump_spheres};
    std::uint32_t fluid_particles{k_default_fluid_particles};
    bool frames_set{};
    bool headless{};
    bool validate{};
    bool list_scenes{};
    bool all_scenes{};
    bool scene_overridden{};
    bool help{};
    bool version{};
    bool calibrate{};
    bool bricks_overridden{};
    bool profiles_file_overridden{};
};

struct GalleryVertex {
    float position[3]{};
    float normal[3]{};
    float color[3]{};
    float uv[2]{};
    float checkerboard{};
};

struct ParticleVertex {
    float position[3]{};
    float color[4]{};
    float radius{};
};

struct UiVertex {
    float position[2]{};
    float color[4]{};
};

struct Matrix4 {
    float values[16]{};
};

struct GalleryUniforms {
    Matrix4 view_projection{};
    float viewport_height{};
    float eye[3]{};
    float camera_u[3]{};
    float camera_v[3]{};
    float camera_w[3]{};
};

struct UiUniforms {
    float viewport[2]{};
};

struct alignas(16) RigidInstance {
    float position[3]{};
    float padding{};
    float orientation[4]{};
};

struct RigidBatch {
    std::uint32_t mesh_index{};
    std::vector<GalleryVertex> vertices{};
    std::vector<RigidInstance> instances{};
};

struct RenderFrame {
    std::vector<GalleryVertex> triangles{};
    std::vector<RigidBatch> rigid_batches{};
    std::vector<ParticleVertex> particles{};
    std::vector<ParticleVertex> smoke_particles{};
};

struct GalleryOverlay {
    std::optional<GalleryContext> picker_selection{};
    bool count_dialog_visible{};
    GalleryContext context{GalleryContext::rigid_body};
    std::string_view count_value{};
    std::array<std::string_view, 3> brick_values{};
    std::size_t brick_field{};
    bool brick_dialog{};
    bool count_value_invalid{};
    std::string_view status{};
};

enum class MetalSmokeDebugMode : std::uint8_t {
    none,
    grid,
    velocity,
    pressure,
    density_temperature,
    vorticity,
    divergence,
};

struct MetalDebugState {
    bool particle_view{};
    bool normals{};
    bool rigid_forces{};
    bool fluid_forces{};
    bool cloth_bonds{};
    bool velocities{};
    bool structure{};
    bool rigid_contacts{};
    MetalSmokeDebugMode smoke_mode{MetalSmokeDebugMode::none};

    void reset() noexcept { *this = {}; }

    void toggle_smoke(MetalSmokeDebugMode mode) noexcept {
        smoke_mode = smoke_mode == mode ? MetalSmokeDebugMode::none : mode;
    }

    void toggle_primary(GalleryContext context) noexcept {
        const GalleryEntry &entry = gallery_entry(context);
        if (entry.has_cloth || entry.has_soft_body || entry.has_rope)
            structure = !structure;
        if (entry.has_fluid) particle_view = !particle_view;
        if (!entry.has_cloth && !entry.has_fluid && !entry.has_soft_body &&
            !entry.has_rope) {
            rigid_contacts = !rigid_contacts;
        }
    }
};

static_assert(sizeof(GalleryVertex) == 48U);
static_assert(sizeof(ParticleVertex) == 32U);
static_assert(sizeof(UiVertex) == 24U);
static_assert(sizeof(GalleryUniforms) == 116U);
static_assert(sizeof(UiUniforms) == 8U);
static_assert(sizeof(RigidInstance) == 32U);

bool parse_u32(std::string_view text, std::uint32_t &output) {
    const char *begin = text.data();
    const char *end = begin + text.size();
    const auto result = std::from_chars(begin, end, output);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parse_float(std::string_view text, float &output) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), output);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
        std::isfinite(output);
}

std::optional<GalleryContext> context_from_option(std::string_view option) {
    if (option == "--rigid") return GalleryContext::rigid_body;
    if (option == "--dump") return GalleryContext::dump;
    for (const GalleryEntry &entry : gallery_entries) {
        if (!entry.command_line_option.empty() &&
            option == entry.command_line_option) {
            return entry.context;
        }
    }
    return std::nullopt;
}

void print_help() {
    std::cout
        << "ParallelMater Metal gallery\n"
           "Usage: parallel-mater-metal-gallery [options] [scene option]\n"
           "  --headless              render without a window\n"
           "  --frames N              simulate N frames (headless default: 240)\n"
           "  --output PATH           write the final frame as PPM\n"
           "  --width N --height N    render dimensions\n"
           "  --scene PATH            override the rigid-body GLB\n"
           "  --scene-index N         select gallery entry 0..28\n"
           "  --all-scenes            run every gallery entry\n"
           "  --particles N           fluid capacity\n"
           "  --dump-spheres N        procedural dump sphere count\n"
           "  --brick-count N         procedural wall bricks (1..4096)\n"
           "  --brick-scale N         brick scale (0.5..2.0)\n"
           "  --brick-planes N        separated walls (1..16)\n"
           "  --calibrate             run device calibration\n"
           "  --profiles-file PATH    tracked device profile catalog\n"
           "  --validate              check physics and rendered output\n"
           "  --list-scenes           print the gallery registry\n"
           "  --version               print build identity\n"
           "  --help                  show this message\n\n"
           "Interactive controls: Tab opens scenes; Up/Down selects and Enter\n"
           "loads; mouse orbits/pans/zooms; arrows run each scene control;\n"
           "Space performs the scene action; P edits supported scene counts;\n"
           "F shows GPU timing; V/Z/X/C/B/N select debug views; R reloads;\n"
           "Escape closes the active page or exits.\n\n"
           "Scene options:\n";
    for (std::size_t index = 0U; index < gallery_entries.size(); ++index) {
        const GalleryEntry &entry = gallery_entries[index];
        std::cout << "  " << index << ": "
                  << (entry.command_line_option.empty()
                          ? (entry.context == GalleryContext::dump
                                 ? "--dump"
                                 : "--rigid")
                          : entry.command_line_option)
                  << "  " << entry.name << '\n';
    }
}

std::string build_label() {
    std::string label{PARALLEL_MATER_GALLERY_GIT_COMMIT};
    if (std::string_view{PARALLEL_MATER_GALLERY_GIT_STATUS} == "dirty")
        label += "-dirty";
    return label;
}

void print_version() {
    std::cout << "ParallelMater Metal Gallery "
              << PARALLEL_MATER_GALLERY_VERSION << " (" << build_label()
              << ", built " << PARALLEL_MATER_GALLERY_BUILD_TIME << ")\n";
}

bool parse_options(int argc, char **argv, Options &output) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help") {
            output.help = true;
        } else if (argument == "--version") {
            output.version = true;
        } else if (argument == "--list-scenes") {
            output.list_scenes = true;
        } else if (argument == "--headless") {
            output.headless = true;
        } else if (argument == "--validate") {
            output.validate = true;
        } else if (argument == "--all-scenes") {
            output.all_scenes = true;
            output.headless = true;
        } else if (argument == "--calibrate") {
            output.calibrate = true;
            output.context = GalleryContext::rigid_body;
        } else if (argument == "--brick-scale") {
            if (index + 1 >= argc || !parse_float(argv[++index], output.bricks.brick_scale))
                return false;
            output.context = GalleryContext::rigid_body;
            output.bricks_overridden = true;
        } else if (argument == "--scene" || argument == "--output" ||
                   argument == "--frames" || argument == "--width" ||
                   argument == "--height" || argument == "--scene-index" ||
                   argument == "--particles" ||
                   argument == "--dump-spheres" ||
                   argument == "--brick-count" ||
                   argument == "--brick-planes" ||
                   argument == "--profiles-file") {
            if (index + 1 >= argc) return false;
            const std::string_view value{argv[++index]};
            if (argument == "--scene") {
                output.scene = value;
                output.scene_overridden = true;
            } else if (argument == "--output") {
                output.output = value;
            } else if (argument == "--profiles-file") {
                output.profiles_file = value;
                output.profiles_file_overridden = true;
            } else {
                std::uint32_t number{};
                if (!parse_u32(value, number)) return false;
                if (argument == "--frames") {
                    output.frames = number;
                    output.frames_set = true;
                } else if (argument == "--width") {
                    output.width = number;
                } else if (argument == "--height") {
                    output.height = number;
                } else if (argument == "--particles") {
                    output.fluid_particles = number;
                } else if (argument == "--dump-spheres") {
                    output.dump_spheres = number;
                } else if (argument == "--brick-count") {
                    output.bricks.brick_count = number;
                    output.context = GalleryContext::rigid_body;
                    output.bricks_overridden = true;
                } else if (argument == "--brick-planes") {
                    output.bricks.wall_planes = number;
                    output.context = GalleryContext::rigid_body;
                    output.bricks_overridden = true;
                } else {
                    if (number >= gallery_entries.size()) return false;
                    output.context = gallery_entries[number].context;
                }
            }
        } else if (const auto context = context_from_option(argument)) {
            output.context = *context;
        } else {
            return false;
        }
    }
    std::string brick_error;
    if (output.width == 0U || output.height == 0U ||
        output.fluid_particles == 0U || output.dump_spheres == 0U) {
        return false;
    }
    if (!validate_brick_config(output.bricks, brick_error)) return false;
    if (output.all_scenes && !output.frames_set) output.frames = 1U;
    return true;
}

bool check(Status status, std::string_view operation, std::string &error) {
    if (status) return true;
    error = std::string(operation) + " failed: " +
        (status.message != nullptr ? status.message : "unknown error") +
        " (Metal error " + std::to_string(status.metal_error) + ")";
    return false;
}

Vec3 add(Vec3 left, Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 subtract(Vec3 left, Vec3 right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 scale(Vec3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

float dot(Vec3 left, Vec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3 cross(Vec3 left, Vec3 right) {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

Vec3 normalize(Vec3 value) {
    const float length = std::sqrt(std::max(0.0F, dot(value, value)));
    return length > 1.0e-8F ? scale(value, 1.0F / length)
                            : Vec3{0.0F, 1.0F, 0.0F};
}

Vec3 rotate(Quaternion orientation, Vec3 point) {
    const Vec3 vector{orientation.x, orientation.y, orientation.z};
    const Vec3 twice_cross = scale(cross(vector, point), 2.0F);
    return add(point, add(scale(twice_cross, orientation.w),
                          cross(vector, twice_cross)));
}

Quaternion conjugate(Quaternion value) {
    return {-value.x, -value.y, -value.z, value.w};
}

Quaternion multiply(Quaternion left, Quaternion right) {
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

Quaternion rotation_z(float radians) {
    return {0.0F, 0.0F, std::sin(radians * 0.5F),
            std::cos(radians * 0.5F)};
}

Vec3 midpoint(Vec3 left, Vec3 right) {
    return scale(add(left, right), 0.5F);
}

Matrix4 multiply(Matrix4 left, Matrix4 right) {
    Matrix4 output{};
    for (std::size_t column = 0U; column < 4U; ++column)
        for (std::size_t row = 0U; row < 4U; ++row)
            for (std::size_t inner = 0U; inner < 4U; ++inner)
                output.values[column * 4U + row] +=
                    left.values[inner * 4U + row] *
                    right.values[column * 4U + inner];
    return output;
}

Camera camera_for_preset(CameraPreset preset) {
    CameraController controller;
    controller.set_preset(preset);
    return controller.camera();
}

Matrix4 view_projection(float aspect, Camera camera) {
    constexpr float pi = 3.14159265358979323846F;
    constexpr float near_plane = 0.05F;
    constexpr float far_plane = 120.0F;
    const Vec3 forward = normalize(subtract(camera.target, camera.eye));
    const Vec3 side = normalize(cross(forward, camera.up));
    const Vec3 up = normalize(cross(side, forward));
    Matrix4 view{{
        side.x, up.x, -forward.x, 0.0F,
        side.y, up.y, -forward.y, 0.0F,
        side.z, up.z, -forward.z, 0.0F,
        -dot(side, camera.eye), -dot(up, camera.eye),
        dot(forward, camera.eye), 1.0F,
    }};
    const float focal = 1.0F / std::tan(
        camera.vertical_field_of_view_degrees * pi / 360.0F);
    Matrix4 projection{{
        focal / aspect, 0.0F, 0.0F, 0.0F,
        0.0F, focal, 0.0F, 0.0F,
        0.0F, 0.0F, far_plane / (near_plane - far_plane), -1.0F,
        0.0F, 0.0F,
        (far_plane * near_plane) / (near_plane - far_plane), 0.0F,
    }};
    return multiply(projection, view);
}

template <typename T>
const T *span_data(BufferSpan<const T> span) {
    if (span.size == 0U) return nullptr;
    if (span.buffer == nullptr) return nullptr;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)span.buffer;
    if (buffer == nil || buffer.contents == nullptr) return nullptr;
    const std::uint64_t bytes = span.size * sizeof(T);
    if (span.byte_offset > buffer.length ||
        bytes > buffer.length - span.byte_offset) {
        return nullptr;
    }
    return reinterpret_cast<const T *>(
        static_cast<const std::uint8_t *>(buffer.contents) +
        span.byte_offset);
}

bool finite(Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool finite(RigidBodyState state) {
    return finite(state.position) && finite(state.linear_velocity) &&
           finite(state.angular_velocity) &&
           std::isfinite(state.orientation.x) &&
           std::isfinite(state.orientation.y) &&
           std::isfinite(state.orientation.z) &&
           std::isfinite(state.orientation.w);
}

class FixedContactCollector {
  public:
    bool initialize(const MetalScene &scene, const MetalInstance &instance,
                    std::string &error) {
        collector_body_index_ = invalid_index;
        generated_constraints_.clear();
        try {
            members_.assign(scene.rigid_bodies.size(), std::uint8_t{});
            generated_constraints_.reserve(scene.rigid_bodies.size());
            gravity_bodies_.reserve(scene.rigid_bodies.size());
        } catch (...) {
            error = "fixed collector state allocation failed";
            return false;
        }
        if (scene.rigid_bodies.size() != instance.rigid_bodies.size()) {
            error = "fixed collector body bindings do not match scene";
            return false;
        }
        for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
            const auto &body = scene.rigid_bodies[index];
            if (body.source_name == "Large" || body.name == "Large") {
                if (collector_body_index_ != invalid_index) {
                    error = "fixed collector body name is ambiguous";
                    return false;
                }
                collector_body_index_ = index;
            }
        }
        if (collector_body_index_ == invalid_index) {
            error = "fixed collector body was not found";
            return false;
        }
        members_[collector_body_index_] = 1U;
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto &constraint : scene.rigid_constraints) {
                if (constraint.options.type != RigidConstraintType::fixed ||
                    !constraint.options.enabled ||
                    constraint.body_a >= members_.size() ||
                    constraint.body_b >= members_.size()) {
                    continue;
                }
                const bool member_a = members_[constraint.body_a] != 0U;
                const bool member_b = members_[constraint.body_b] != 0U;
                if (member_a == member_b) continue;
                members_[member_a ? constraint.body_b : constraint.body_a] = 1U;
                changed = true;
            }
        }
        return true;
    }

    [[nodiscard]] bool active() const noexcept {
        return collector_body_index_ != invalid_index;
    }

    [[nodiscard]] static std::uint32_t constraint_capacity(
        const MetalScene &scene) noexcept {
        return static_cast<std::uint32_t>(std::max<std::size_t>(
            1U, std::max(scene.rigid_constraints.size(),
                         scene.rigid_bodies.size())));
    }

    bool apply_loose_gravity(World &world, const MetalScene &scene,
                             const MetalInstance &instance,
                             Vec3 loose_gravity, Vec3 world_gravity,
                             std::string &error) {
        if (!active() || members_.size() != scene.rigid_bodies.size() ||
            members_.size() != instance.rigid_bodies.size()) {
            error = "fixed collector bindings are invalid";
            return false;
        }
        const Vec3 acceleration = subtract(loose_gravity, world_gravity);
        if (acceleration.x == 0.0F && acceleration.y == 0.0F &&
            acceleration.z == 0.0F) {
            return true;
        }
        gravity_bodies_.clear();
        for (std::size_t index = 0U; index < members_.size(); ++index) {
            if (members_[index] == 0U &&
                scene.rigid_bodies[index].options.motion == MotionType::dynamic) {
                gravity_bodies_.push_back(instance.rigid_bodies[index]);
            }
        }
        return check(world.apply_central_acceleration(
                         {gravity_bodies_.data(), gravity_bodies_.size()},
                         acceleration),
                     "apply loose-body collector gravity", error);
    }

    bool collect(World &world, const MetalScene &scene,
                 const MetalInstance &instance, std::string &error) {
        if (!active() || members_.size() != scene.rigid_bodies.size() ||
            members_.size() != instance.rigid_bodies.size()) {
            error = "fixed collector bindings are invalid";
            return false;
        }
        const auto contacts = world.rigid_contacts();
        const RigidContactEvent *events = span_data(contacts.events);
        if (contacts.event_count != 0U && events == nullptr) {
            error = "fixed collector contact view is unavailable";
            return false;
        }
        const auto body_index = [&](RigidBodyId id) noexcept {
            const auto found = std::find(instance.rigid_bodies.begin(),
                                         instance.rigid_bodies.end(), id);
            return found == instance.rigid_bodies.end()
                ? invalid_index
                : static_cast<std::size_t>(found -
                                           instance.rigid_bodies.begin());
        };
        RigidBodyDeviceView state_view{};
        const RigidBodyId *dense_ids = nullptr;
        const RigidBodyState *dense_states = nullptr;
        for (std::uint32_t contact_index = 0U;
             contact_index < contacts.event_count; ++contact_index) {
            const RigidContactEvent &contact = events[contact_index];
            if (contact.penetration <= 0.0F &&
                contact.normal_impulse <= 0.0F) {
                continue;
            }
            const std::size_t first = body_index(contact.body);
            const std::size_t second = body_index(contact.collider);
            if (first == invalid_index || second == invalid_index ||
                first == second ||
                (first != collector_body_index_ &&
                 second != collector_body_index_)) {
                continue;
            }
            const std::size_t target = first == collector_body_index_
                ? second : first;
            if (members_[target] != 0U ||
                scene.rigid_bodies[target].options.motion !=
                    MotionType::dynamic) {
                continue;
            }
            if (dense_ids == nullptr) {
                if (!check(world.rigid_body_view(state_view),
                           "read fixed collector body view", error)) {
                    return false;
                }
                dense_ids = span_data(state_view.ids);
                dense_states = span_data(state_view.states);
                if (state_view.ids.size != state_view.states.size ||
                    (state_view.ids.size != 0U &&
                     (dense_ids == nullptr || dense_states == nullptr))) {
                    error = "fixed collector body view is unavailable";
                    return false;
                }
            }
            const auto find_state = [&](RigidBodyId id) {
                std::uint64_t index = 0U;
                while (index < state_view.ids.size && dense_ids[index] != id)
                    ++index;
                return index;
            };
            const std::uint64_t collector_dense =
                find_state(instance.rigid_bodies[collector_body_index_]);
            const std::uint64_t target_dense =
                find_state(instance.rigid_bodies[target]);
            if (collector_dense == state_view.ids.size ||
                target_dense == state_view.ids.size) {
                error = "fixed collector body handle is stale";
                return false;
            }
            const RigidBodyState &collector_state =
                dense_states[collector_dense];
            const RigidBodyState &target_state = dense_states[target_dense];
            const Quaternion collector_inverse =
                conjugate(collector_state.orientation);
            const Quaternion target_inverse =
                conjugate(target_state.orientation);
            RigidConstraintOptions options{
                .type = RigidConstraintType::fixed,
                .body_a = instance.rigid_bodies[collector_body_index_],
                .body_b = instance.rigid_bodies[target],
                .local_anchor_a = rotate(
                    collector_inverse,
                    subtract(contact.position, collector_state.position)),
                .local_anchor_b = rotate(
                    target_inverse,
                    subtract(contact.position, target_state.position)),
                .local_orientation_a = {},
                .local_orientation_b = multiply(
                    target_inverse, collector_state.orientation),
                .enabled = true,
                .disable_collisions = true,
                .solver_iterations = 16U};
            RigidConstraintId constraint{};
            if (!check(world.add_rigid_constraint(options, constraint),
                       "attach fixed collector body", error)) {
                return false;
            }
            generated_constraints_.push_back(constraint);
            members_[target] = 1U;
        }
        return true;
    }

  private:
    static constexpr std::size_t invalid_index =
        std::numeric_limits<std::size_t>::max();
    std::size_t collector_body_index_{invalid_index};
    std::vector<std::uint8_t> members_{};
    std::vector<RigidConstraintId> generated_constraints_{};
    std::vector<RigidBodyId> gravity_bodies_{};
};

void append_vertex(RenderFrame &frame, Vec3 position, Vec3 normal,
                   Vec3 color, parallel_mater::Vec2 uv = {},
                   bool checkerboard = false) {
    frame.triangles.push_back(
        {{position.x, position.y, position.z},
         {normal.x, normal.y, normal.z},
         {color.x, color.y, color.z},
         {uv.x, uv.y}, checkerboard ? 1.0F : 0.0F});
}

float paint_cubic_weight(float value) {
    value = std::abs(value);
    if (value <= 1.0F)
        return (4.0F - 6.0F * value * value +
                3.0F * value * value * value) / 6.0F;
    if (value < 2.0F)
        return (2.0F - value) * (2.0F - value) * (2.0F - value) /
               6.0F;
    return 0.0F;
}

float paint_amount(const PaintFieldDeviceView *view,
                   parallel_mater::Vec2 uv) {
    if (view == nullptr || view->width == 0U || view->height == 0U ||
        view->pixels.size !=
            static_cast<std::uint64_t>(view->width) * view->height) {
        return 0.0F;
    }
    const std::uint32_t *pixels = span_data(view->pixels);
    if (pixels == nullptr) return 0.0F;
    const int width = static_cast<int>(view->width);
    const int height = static_cast<int>(view->height);
    const float fx = (uv.x - std::floor(uv.x)) * width - 0.5F;
    const float fy = std::clamp(uv.y, 0.0F, 1.0F) * height - 0.5F;
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    float paint = 0.0F;
    float total_weight = 0.0F;
    for (int dy = -1; dy <= 2; ++dy) {
        const int y = std::clamp(y0 + dy, 0, height - 1);
        const float wy = paint_cubic_weight(fy - static_cast<float>(y0 + dy));
        for (int dx = -1; dx <= 2; ++dx) {
            int x = (x0 + dx) % width;
            if (x < 0) x += width;
            const float weight = wy *
                paint_cubic_weight(fx - static_cast<float>(x0 + dx));
            paint += weight * ((pixels[y * width + x] & 3U) != 0U
                                   ? 1.0F : 0.0F);
            total_weight += weight;
        }
    }
    return total_weight > 0.0F
        ? std::clamp(paint / total_weight, 0.0F, 1.0F) : 0.0F;
}

Vec3 painted_color(Vec3 base, const PaintFieldDeviceView *paint,
                   parallel_mater::Vec2 uv) {
    const float amount = paint_amount(paint, uv);
    constexpr Vec3 wet_blue{0.025F, 0.36F, 0.94F};
    return add(scale(base, 1.0F - amount), scale(wet_blue, amount));
}

bool append_rigid_mesh(const MetalMesh &mesh, RigidBodyState state,
                       RenderFrame &frame,
                       const PaintFieldDeviceView *paint = nullptr) {
    for (const std::uint32_t vertex_index : mesh.indices) {
        if (vertex_index >= mesh.vertices.size()) return false;
        const auto &source = mesh.vertices[vertex_index];
        append_vertex(frame,
                      add(state.position,
                          rotate(state.orientation, source.position)),
                      normalize(rotate(state.orientation, source.normal)),
                      painted_color(mesh.base_color, paint, source.uv),
                      source.uv);
    }
    return true;
}

bool append_rigid_instance(const MetalMesh &mesh, std::uint32_t mesh_index,
                           RigidBodyState state, RenderFrame &frame) {
    auto batch = std::find_if(
        frame.rigid_batches.begin(), frame.rigid_batches.end(),
        [mesh_index](const RigidBatch &candidate) {
            return candidate.mesh_index == mesh_index;
        });
    if (batch == frame.rigid_batches.end()) {
        frame.rigid_batches.push_back({mesh_index});
        batch = frame.rigid_batches.end() - 1;
        batch->vertices.reserve(mesh.indices.size());
        for (const std::uint32_t vertex_index : mesh.indices) {
            if (vertex_index >= mesh.vertices.size()) return false;
            const auto &source = mesh.vertices[vertex_index];
            batch->vertices.push_back(
                {{source.position.x, source.position.y, source.position.z},
                 {source.normal.x, source.normal.y, source.normal.z},
                 {mesh.base_color.x, mesh.base_color.y, mesh.base_color.z},
                 {source.uv.x, source.uv.y},
                 0.0F});
        }
    }
    batch->instances.push_back(
        {{state.position.x, state.position.y, state.position.z},
         0.0F,
         {state.orientation.x, state.orientation.y,
          state.orientation.z, state.orientation.w}});
    return true;
}

bool append_surface(const Vec3 *positions, std::size_t position_count,
                    const std::uint32_t *indices, std::size_t index_count,
                    Vec3 color, RenderFrame &frame,
                    const MetalMesh *material_mesh = nullptr,
                    const PaintFieldDeviceView *paint = nullptr) {
    if ((position_count != 0U && positions == nullptr) ||
        (index_count != 0U && indices == nullptr)) {
        return false;
    }
    for (std::size_t index = 0U; index + 2U < index_count; index += 3U) {
        const std::uint32_t a = indices[index];
        const std::uint32_t b = indices[index + 1U];
        const std::uint32_t c = indices[index + 2U];
        if (a >= position_count || b >= position_count ||
            c >= position_count) {
            return false;
        }
        const Vec3 normal = normalize(cross(subtract(positions[b], positions[a]),
                                            subtract(positions[c], positions[a])));
        const auto vertex_color = [&](std::uint32_t vertex) {
            return material_mesh != nullptr &&
                           vertex < material_mesh->vertices.size()
                ? painted_color(color, paint,
                                material_mesh->vertices[vertex].uv)
                : color;
        };
        append_vertex(frame, positions[a], normal, vertex_color(a));
        append_vertex(frame, positions[b], normal, vertex_color(b));
        append_vertex(frame, positions[c], normal, vertex_color(c));
    }
    return true;
}

struct Runtime {
    GalleryContext context{GalleryContext::rigid_body};
    MetalScene scene{};
    World world{};
    MetalInstance instance{};
    std::size_t kinematic_index{std::numeric_limits<std::size_t>::max()};
    RigidBodyState kinematic_target{};
    FixedContactCollector fixed_collector{};
    std::vector<RigidBodyId> gravity_tilt_bodies{};
    bool collect_rigid_contacts{};
    bool collect_kernel_timings{};
};

bool borrow_paint(Runtime &runtime, std::uint32_t body_index,
                  std::uint32_t mesh_index, PaintFieldDeviceView &view,
                  bool &found, std::string &error) {
    found = false;
    view = {};
    const auto binding = std::find_if(
        runtime.instance.paint_bindings.begin(),
        runtime.instance.paint_bindings.end(),
        [&](const MetalInstance::PaintBinding &candidate) {
            return candidate.body_index == body_index &&
                   candidate.mesh_index == mesh_index;
        });
    if (binding == runtime.instance.paint_bindings.end()) return true;
    if (!check(runtime.world.paint_field_view(binding->field, view),
               "read paint render view", error)) {
        return false;
    }
    found = true;
    return true;
}

struct DirectionalInput {
    float x{};
    float z{};
};

DirectionalInput directional_input(GLFWwindow *window, bool enabled) {
    if (!enabled) return {};
    DirectionalInput input{
        static_cast<float>(
            glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) -
            static_cast<float>(
                glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS),
        static_cast<float>(
            glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) -
            static_cast<float>(
                glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)};
    const float length = std::hypot(input.x, input.z);
    if (length > 1.0F) {
        input.x /= length;
        input.z /= length;
    }
    return input;
}

Vec3 gravity_for(DirectionalInput input, float gravity_scale, Camera camera) {
    return screen_space_gravity(camera, input.x, -input.z,
                                k_gravity * gravity_scale,
                                k_rigid_gravity_tilt_degrees);
}

Vec3 collector_gravity_for(DirectionalInput input, float gravity_scale,
                           Camera camera) {
    return screen_space_gravity(camera, input.x, -input.z,
                                k_gravity * gravity_scale,
                                k_collector_gravity_tilt_degrees);
}

bool apply_gravity_tilt_overrides(Runtime &runtime, Vec3 world_gravity,
                                  std::string &error) {
    if (runtime.gravity_tilt_bodies.empty()) return true;
    const Vec3 vertical_gravity{
        0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F};
    const Vec3 compensation{
        vertical_gravity.x - world_gravity.x,
        vertical_gravity.y - world_gravity.y,
        vertical_gravity.z - world_gravity.z};
    if (compensation.x == 0.0F && compensation.y == 0.0F &&
        compensation.z == 0.0F) {
        return true;
    }
    return check(runtime.world.apply_central_acceleration(
                     {runtime.gravity_tilt_bodies.data(),
                      runtime.gravity_tilt_bodies.size()}, compensation),
                 "preserve authored rigid gravity", error);
}

bool toggle_constraints(Runtime &runtime, std::string &error) {
    if (runtime.scene.rigid_constraints.empty() ||
        runtime.scene.rigid_constraints.size() !=
            runtime.instance.rigid_constraints.size()) {
        error = "constraint toggle scene needs matching constraints";
        return false;
    }
    bool enable = false;
    for (const auto id : runtime.instance.rigid_constraints) {
        RigidConstraintState state;
        if (!check(runtime.world.read_rigid_constraint_state(id, state),
                   "read constraint state", error)) {
            return false;
        }
        enable = enable || !state.enabled;
    }
    for (std::size_t index = 0U;
         index < runtime.scene.rigid_constraints.size(); ++index) {
        auto &definition = runtime.scene.rigid_constraints[index];
        RigidConstraintOptions options = definition.options;
        options.body_a = runtime.instance.rigid_bodies[definition.body_a];
        options.body_b = runtime.instance.rigid_bodies[definition.body_b];
        options.enabled = enable;
        if (enable && options.type != RigidConstraintType::point) {
            RigidBodyState state_a;
            RigidBodyState state_b;
            if (!check(runtime.world.read_rigid_body_state(
                           options.body_a, state_a),
                       "read first constraint body", error) ||
                !check(runtime.world.read_rigid_body_state(
                           options.body_b, state_b),
                       "read second constraint body", error)) {
                return false;
            }
            const Vec3 anchor = midpoint(state_a.position, state_b.position);
            const Quaternion world_orientation = state_a.orientation;
            options.local_anchor_a = rotate(
                conjugate(state_a.orientation),
                subtract(anchor, state_a.position));
            options.local_anchor_b = rotate(
                conjugate(state_b.orientation),
                subtract(anchor, state_b.position));
            options.local_orientation_a = multiply(
                conjugate(state_a.orientation), world_orientation);
            options.local_orientation_b = multiply(
                conjugate(state_b.orientation), world_orientation);
        }
        if (!check(runtime.world.update_rigid_constraint(
                       runtime.instance.rigid_constraints[index], options),
                   enable ? "enable constraint" : "disable constraint",
                   error)) {
            return false;
        }
        definition.options = options;
    }
    return true;
}

bool drive_motors(Runtime &runtime, DirectionalInput input,
                  std::string &error) {
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
        if (definition.options.motor.angular_target_velocity ==
            target_velocity) {
            continue;
        }
        RigidConstraintOptions options = definition.options;
        options.body_a = runtime.instance.rigid_bodies[definition.body_a];
        options.body_b = runtime.instance.rigid_bodies[definition.body_b];
        options.motor.angular_target_velocity = target_velocity;
        if (!check(runtime.world.update_rigid_constraint(
                       runtime.instance.rigid_constraints[index], options),
                   "drive motor constraint", error)) {
            return false;
        }
        definition.options = options;
    }
    return true;
}

std::filesystem::path bundled_scene_path(
    const std::filesystem::path &configured_path) {
    if (configured_path.empty()) return {};
    @autoreleasepool {
        NSString *resource_path = [NSBundle mainBundle].resourcePath;
        if (resource_path != nil) {
            const auto candidate =
                std::filesystem::path(resource_path.UTF8String) / "assets" /
                configured_path.filename();
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error))
                return candidate;
        }
    }
    return configured_path;
}

std::filesystem::path device_profiles_path(const Options &options) {
    if (options.profiles_file_overridden) return options.profiles_file;
    @autoreleasepool {
        NSString *resource_path = [NSBundle mainBundle].resourcePath;
        if (resource_path != nil) {
            const auto bundled = std::filesystem::path(resource_path.UTF8String) /
                "config/device-profiles.json";
            std::error_code error;
            if (std::filesystem::is_regular_file(bundled, error)) return bundled;
        }
    }
    return options.profiles_file;
}

std::string sysctl_string(const char *name) {
    std::size_t size = 0U;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0U) != 0 || size == 0U)
        return {};
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0U) != 0) return {};
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

std::string system_profiler_value(NSString *data_type, NSString *property) {
    @autoreleasepool {
        NSTask *task = [[NSTask alloc] init];
        task.executableURL = [NSURL fileURLWithPath:@"/usr/sbin/system_profiler"];
        task.arguments = @[data_type, @"-json"];
        NSPipe *pipe = [NSPipe pipe];
        task.standardOutput = pipe;
        task.standardError = [NSFileHandle fileHandleWithNullDevice];
        NSError *launch_error = nil;
        if (![task launchAndReturnError:&launch_error])
            return {};
        [task waitUntilExit];
        NSData *data = [pipe.fileHandleForReading readDataToEndOfFile];
        if (task.terminationStatus == 0 && data.length != 0U) {
            NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                options:0 error:nil];
            NSArray *records = root[data_type];
            NSDictionary *record = records.count != 0U ? records[0] : nil;
            NSString *value = record[property];
            if ([value isKindOfClass:[NSString class]] && value.length != 0U)
                return value.UTF8String;
        }
        return {};
    }
}

std::string metal_gpu_variant(id<MTLDevice> device) {
    const std::string cores = system_profiler_value(
        @"SPDisplaysDataType", @"sppci_cores");
    return !cores.empty() ? cores + "-core GPU"
        : device != nil ? std::string(device.name.UTF8String) : "Apple GPU";
}

parallel_mater::gallery::HardwareIdentity metal_hardware_identity(
    id<MTLDevice> device) {
    parallel_mater::gallery::HardwareIdentity result;
    result.machine_model = sysctl_string("hw.model");
    if (result.machine_model.empty())
        result.machine_model = system_profiler_value(
            @"SPHardwareDataType", @"machine_model");
    result.cpu_model = sysctl_string("machdep.cpu.brand_string");
    if (result.cpu_model.empty())
        result.cpu_model = system_profiler_value(
            @"SPHardwareDataType", @"chip_type");
    if (result.cpu_model.empty())
        result.cpu_model = device != nil ? device.name.UTF8String : "Apple CPU";
    result.gpu_model = device != nil ? device.name.UTF8String : "Apple GPU";
    // Metal does not expose the core bin, but system_profiler does. Keep the
    // exact bin in the matching key so 8-core and 10-core Airs revalidate.
    result.gpu_variant = metal_gpu_variant(device);
    result.memory_bytes = NSProcessInfo.processInfo.physicalMemory;
    result.backend = "metal";
    result.operating_system =
        NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String;
    result.driver = "Metal " + result.operating_system;
    result.power_mode = NSProcessInfo.processInfo.isLowPowerModeEnabled
        ? "low-power" : "normal";
    return result;
}

std::filesystem::path scene_path(const Options &options,
                                 GallerySceneSource source) {
    const std::array paths{
        options.scene,
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH),
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
        std::filesystem::path(PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH),
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
    static_assert(paths.size() == static_cast<std::size_t>(GallerySceneSource::smoke_rope) + 1U);
    const auto configured_path = paths[static_cast<std::size_t>(source)];
    if (source == GallerySceneSource::default_scene &&
        options.scene_overridden) {
        return configured_path;
    }
    return bundled_scene_path(configured_path);
}

bool build_runtime(const Options &options, GalleryContext context,
                   Runtime &output, std::string &error) {
    Runtime next;
    next.context = context;
    const GalleryEntry &entry = gallery_entry(context);
    if (entry.source == GallerySceneSource::procedural_dump) {
        next.scene =
            parallel_mater::metal::gallery::make_dump_scene(
                options.dump_spheres);
    } else if (!parallel_mater::metal::gallery::load_glb_scene(
                   scene_path(options, entry.source), next.scene, error)) {
        error = "scene load failed: " + error;
        return false;
    }
    if (context == GalleryContext::rigid_body) {
        MetalScene generated;
        if (!parallel_mater::metal::gallery::make_brick_scene(
                next.scene, options.bricks, generated, error)) {
            error = "brick scene generation failed: " + error;
            return false;
        }
        next.scene = std::move(generated);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        const std::uint64_t recommended = device != nil
            ? static_cast<std::uint64_t>(device.recommendedMaxWorkingSetSize) : 0U;
        const std::uint64_t allocated = device != nil
            ? static_cast<std::uint64_t>(device.currentAllocatedSize) : 0U;
        const std::uint64_t available = recommended > allocated
            ? recommended - allocated : recommended;
        if (available != 0U &&
            parallel_mater::gallery::estimated_metal_contact_bytes(
                static_cast<std::uint32_t>(next.scene.rigid_bodies.size())) >
                available * 7U / 10U) {
            error = "brick scene exceeds the available Metal memory budget";
            return false;
        }
    }
    if (entry.has_fluid)
        next.scene.fluid_options.capacity = options.fluid_particles;

    WorldOptions world_options{};
    if (!check(parallel_mater::metal::gallery::scene_world_options(
                   next.scene, world_options),
               "derive gallery world capacities", error)) {
        return false;
    }
    world_options.rigid_sleeping = true;
    if (entry.controls == GalleryControlPolicy::collector_gravity) {
        world_options.rigid_constraint_capacity =
            FixedContactCollector::constraint_capacity(next.scene);
    }
    if (!check(World::create(world_options, next.world),
               "create gallery world", error) ||
        !check(parallel_mater::metal::gallery::instantiate_scene(
                   next.scene, next.world, next.instance),
               "instantiate gallery scene", error)) {
        return false;
    }
    if (entry.controls == GalleryControlPolicy::collector_gravity &&
        !next.fixed_collector.initialize(next.scene, next.instance, error)) {
        return false;
    }
    next.gravity_tilt_bodies.reserve(next.scene.rigid_bodies.size());
    for (std::size_t index = 0U;
         index < next.scene.rigid_bodies.size(); ++index) {
        const auto &body = next.scene.rigid_bodies[index];
        if (body.options.motion == MotionType::dynamic &&
            !body.follows_gravity_tilt) {
            next.gravity_tilt_bodies.push_back(
                next.instance.rigid_bodies[index]);
        }
        if (body.options.motion == MotionType::kinematic) {
            next.kinematic_index = index;
            next.kinematic_target =
                next.scene.rigid_bodies[index].options.initial_state;
            break;
        }
    }
    output = std::move(next);
    return true;
}

bool read_rigid_states(Runtime &runtime,
                       std::vector<RigidBodyState> &states,
                       std::string &error) {
    states.resize(runtime.instance.rigid_bodies.size());
    RigidBodyDeviceView view;
    if (!check(runtime.world.rigid_body_view(view),
               "read rigid-body view", error)) {
        return false;
    }
    const auto *ids = span_data(view.ids);
    const auto *device_states = span_data(view.states);
    if (view.states.size != view.ids.size ||
        (view.states.size != 0U && (ids == nullptr || device_states == nullptr))) {
        error = "rigid-body render view is unavailable or invalid";
        return false;
    }
    bool dense_order = states.size() == view.ids.size;
    for (std::size_t index = 0U; dense_order && index < states.size(); ++index)
        dense_order = ids[index] == runtime.instance.rigid_bodies[index];
    if (dense_order) {
        std::copy_n(device_states, states.size(), states.begin());
        return true;
    }
    for (std::size_t index = 0U; index < states.size(); ++index) {
        const auto handle = runtime.instance.rigid_bodies[index];
        std::size_t dense_index = 0U;
        while (dense_index < view.ids.size && ids[dense_index] != handle)
            ++dense_index;
        if (dense_index == view.ids.size) {
            error = "rigid-body render handle is missing from the device view";
            return false;
        }
        states[index] = device_states[dense_index];
    }
    return true;
}

bool assemble_frame(Runtime &runtime, RenderFrame &frame,
                    std::vector<RigidBodyState> &rigid_states,
                    std::string &error,
                    const MetalDebugState *debug = nullptr) {
    frame.triangles.clear();
    frame.rigid_batches.clear();
    frame.particles.clear();
    frame.smoke_particles.clear();
    if (!read_rigid_states(runtime, rigid_states, error)) return false;

    for (std::size_t body_index = 0U;
         body_index < runtime.scene.rigid_bodies.size(); ++body_index) {
        const auto &body = runtime.scene.rigid_bodies[body_index];
        for (const std::uint32_t mesh_index : body.mesh_indices) {
            if (mesh_index >= runtime.scene.meshes.size()) {
                error = "rigid render mesh index is out of range";
                return false;
            }
            const MetalMesh &mesh = runtime.scene.meshes[mesh_index];
            PaintFieldDeviceView paint;
            bool has_paint = false;
            if (!borrow_paint(runtime, static_cast<std::uint32_t>(body_index),
                              mesh_index, paint, has_paint, error)) {
                return false;
            }
            const bool appended = !mesh.visible ||
                (has_paint
                     ? append_rigid_mesh(mesh, rigid_states[body_index],
                                         frame, &paint)
                     : append_rigid_instance(
                           mesh, mesh_index, rigid_states[body_index], frame));
            if (!appended) {
                error = "rigid render mesh contains an invalid index";
                return false;
            }
        }
    }

    for (std::size_t index = 0U;
         index < runtime.instance.cloths.size(); ++index) {
        ClothDeviceView view;
        if (!check(runtime.world.cloth_view(runtime.instance.cloths[index],
                                            view),
                   "read cloth render view", error)) {
            return false;
        }
        const auto &definition = runtime.scene.cloths[index];
        const MetalMesh *material =
            definition.mesh_index < runtime.scene.meshes.size()
            ? &runtime.scene.meshes[definition.mesh_index] : nullptr;
        const Vec3 color = material != nullptr
            ? material->base_color
            : Vec3{0.75F, 0.55F, 0.25F};
        PaintFieldDeviceView paint;
        bool has_paint = false;
        if (!borrow_paint(runtime, UINT32_MAX, definition.mesh_index,
                          paint, has_paint, error)) {
            return false;
        }
        if (!append_surface(
                span_data(view.surface_positions),
                static_cast<std::size_t>(view.surface_positions.size),
                span_data(view.surface_triangle_indices),
                static_cast<std::size_t>(view.surface_triangle_indices.size),
                color, frame, material, has_paint ? &paint : nullptr)) {
            error = "cloth render view is unavailable or invalid";
            return false;
        }
        if (debug != nullptr && debug->structure) {
            const Vec3 *positions = span_data(view.positions);
            if (view.vertex_count != 0U && positions == nullptr) {
                error = "cloth structure positions are unavailable";
                return false;
            }
            for (std::uint32_t vertex = 0U; vertex < view.vertex_count;
                 ++vertex) {
                frame.particles.push_back(
                    {{positions[vertex].x, positions[vertex].y,
                      positions[vertex].z},
                     {1.0F, 0.62F, 0.12F, 1.0F}, 0.014F});
            }
        }
    }

    for (std::size_t index = 0U;
         index < runtime.instance.soft_bodies.size(); ++index) {
        SoftBodyDeviceView view;
        if (!check(runtime.world.soft_body_view(
                       runtime.instance.soft_bodies[index], view),
                   "read soft-body render view", error)) {
            return false;
        }
        const auto &definition = runtime.scene.soft_bodies[index];
        const Vec3 color = definition.mesh_index < runtime.scene.meshes.size()
            ? runtime.scene.meshes[definition.mesh_index].base_color
            : Vec3{0.62F, 0.38F, 0.92F};
        if (!append_surface(
                span_data(view.surface_positions),
                static_cast<std::size_t>(view.surface_positions.size),
                span_data(view.surface_triangle_indices),
                static_cast<std::size_t>(
                    view.surface_triangle_indices.size),
                color, frame)) {
            error = "soft-body render view is unavailable or invalid";
            return false;
        }
        if (debug != nullptr && debug->structure) {
            const Vec3 *positions = span_data(view.positions);
            if (view.node_count != 0U && positions == nullptr) {
                error = "soft-body structure positions are unavailable";
                return false;
            }
            for (std::uint32_t node = 0U; node < view.node_count; ++node) {
                frame.particles.push_back(
                    {{positions[node].x, positions[node].y,
                      positions[node].z},
                     {0.92F, 0.35F, 1.0F, 1.0F}, 0.016F});
            }
        }
    }

    for (std::size_t index = 0U;
         index < runtime.instance.ropes.size(); ++index) {
        RopeDeviceView view;
        if (!check(runtime.world.rope_view(runtime.instance.ropes[index],
                                           view),
                   "read rope render view", error)) {
            return false;
        }
        const Vec3 *positions = span_data(view.positions);
        if (view.positions.size != 0U && positions == nullptr) {
            error = "rope render positions are unavailable";
            return false;
        }
        const auto &definition = runtime.scene.ropes[index];
        if (definition.mesh_index >= runtime.scene.meshes.size()) {
            error = "rope render mesh index is out of range";
            return false;
        }
        MetalMesh mesh = runtime.scene.meshes[definition.mesh_index];
        std::vector<Vec3> nodes(
            positions, positions + static_cast<std::size_t>(
                                      view.positions.size));
        parallel_mater::metal::gallery::update_rope_render_mesh(
            nodes, view.radius, mesh);
        const RigidBodyState identity{{}, {0.0F, 0.0F, 0.0F, 1.0F}, {}, {}};
        if (!append_rigid_mesh(mesh, identity, frame)) {
            error = "rope render mesh contains an invalid index";
            return false;
        }
        if (debug != nullptr && debug->structure) {
            for (std::uint64_t node = 0U; node < view.positions.size; ++node) {
                frame.particles.push_back(
                    {{positions[node].x, positions[node].y,
                      positions[node].z},
                     {1.0F, 0.48F, 0.08F, 1.0F},
                     std::max(0.012F, view.radius * 0.72F)});
            }
        }
    }

    if (runtime.instance.has_fluid) {
        FluidDeviceView view;
        if (!check(runtime.world.fluid_view(runtime.instance.fluid, view),
                   "read fluid render view", error)) {
            return false;
        }
        const Vec3 *positions = span_data(view.positions);
        const float *foam = span_data(view.foam);
        const float *temperatures = span_data(view.temperatures);
        if (view.particle_count != 0U && positions == nullptr) {
            error = "fluid render positions are unavailable";
            return false;
        }
        frame.particles.reserve(frame.particles.size() +
                                view.particle_count);
        for (std::uint32_t index = 0U; index < view.particle_count; ++index) {
            const float f = foam != nullptr ? std::clamp(foam[index], 0.0F, 1.0F)
                                            : 0.0F;
            const float temperature = temperatures != nullptr
                ? temperatures[index] : 20.0F;
            const float heat = std::clamp((temperature - 30.0F) / 450.0F,
                                          0.0F, 1.0F);
            const Vec3 color{
                0.04F + 0.86F * f + 0.45F * heat,
                0.28F + 0.68F * f,
                0.78F + 0.20F * f - 0.45F * heat};
            frame.particles.push_back(
                {{positions[index].x, positions[index].y,
                  positions[index].z},
                 {color.x, color.y, color.z, 1.0F},
                 view.particle_radius *
                     (debug != nullptr && debug->particle_view
                          ? 0.72F : 1.35F)});
        }
    }

    if (runtime.instance.has_smoke) {
        SmokeDeviceView view;
        if (!check(runtime.world.smoke_view(runtime.instance.smoke, view),
                   "read smoke render view", error)) {
            return false;
        }
        const Vec3 *positions = span_data(view.positions);
        const float *ages = span_data(view.ages);
        const Vec3 *velocities = span_data(view.velocities);
        const float *densities = span_data(view.number_densities);
        const float *pressures = span_data(view.pressures);
        const Vec3 *vorticities = span_data(view.vorticities);
        if (view.particle_count != 0U &&
            (positions == nullptr || ages == nullptr)) {
            error = "smoke render positions are unavailable";
            return false;
        }
        for (std::uint32_t index = 0U; index < view.particle_count; ++index) {
            if (ages[index] >= view.lifetime) continue;
            const float life = view.lifetime > 0.0F
                ? std::clamp(1.0F - ages[index] / view.lifetime,
                             0.0F, 1.0F)
                : 1.0F;
            Vec3 color{0.52F + 0.18F * life,
                       0.55F + 0.18F * life,
                       0.60F + 0.18F * life};
            if (debug != nullptr) {
                if (debug->smoke_mode ==
                        MetalSmokeDebugMode::density_temperature &&
                    densities != nullptr) {
                    const float value = std::tanh(
                        std::max(0.0F, densities[index]) * 0.2F);
                    color = {0.08F + 0.92F * value,
                             0.18F + 0.45F * value,
                             1.0F - 0.72F * value};
                } else if (debug->smoke_mode ==
                               MetalSmokeDebugMode::pressure &&
                           pressures != nullptr) {
                    const float value = std::tanh(
                        std::abs(pressures[index]) * 0.08F);
                    color = pressures[index] >= 0.0F
                        ? Vec3{1.0F, 0.2F + 0.5F * (1.0F - value),
                               0.12F}
                        : Vec3{0.12F, 0.45F, 1.0F};
                } else if (debug->smoke_mode ==
                               MetalSmokeDebugMode::velocity &&
                           velocities != nullptr) {
                    const float value = std::tanh(
                        std::sqrt(dot(velocities[index], velocities[index])));
                    color = {0.08F, 0.35F + 0.65F * value, 1.0F};
                } else if (debug->smoke_mode ==
                               MetalSmokeDebugMode::vorticity &&
                           vorticities != nullptr) {
                    const float value = std::tanh(
                        std::sqrt(dot(vorticities[index], vorticities[index])));
                    color = {0.72F + 0.28F * value, 0.12F,
                             0.78F + 0.22F * value};
                }
            }
            frame.smoke_particles.push_back(
                {{positions[index].x, positions[index].y,
                  positions[index].z},
                 {color.x, color.y, color.z, 0.18F + 0.38F * life},
                 view.particle_radius * 2.0F});
        }
        if (debug != nullptr &&
            debug->smoke_mode != MetalSmokeDebugMode::none &&
            view.grid_resolution != 0U &&
            view.grid_vertical_resolution != 0U) {
            const std::uint64_t cell_count =
                static_cast<std::uint64_t>(view.grid_resolution) *
                view.grid_vertical_resolution * view.grid_resolution;
            const float *grid_density = span_data(view.grid_density);
            const float *grid_pressure = span_data(view.grid_pressure);
            const float *grid_temperature = span_data(view.grid_temperature);
            const float *grid_divergence = span_data(view.grid_divergence);
            const Vec3 *grid_velocity = span_data(view.grid_velocity);
            const Vec3 *grid_vorticity = span_data(view.grid_vorticity);
            const std::uint32_t *grid_solid = span_data(view.grid_solid);
            const std::uint64_t stride = std::max<std::uint64_t>(
                1U, (cell_count + 29'999U) / 30'000U);
            const std::uint64_t layer =
                static_cast<std::uint64_t>(view.grid_resolution) *
                view.grid_vertical_resolution;
            for (std::uint64_t cell = 0U; cell < cell_count;
                 cell += stride) {
                float value = 0.0F;
                Vec3 color{0.15F, 0.75F, 1.0F};
                switch (debug->smoke_mode) {
                case MetalSmokeDebugMode::grid:
                    value = grid_solid != nullptr && grid_solid[cell] != 0U
                        ? 1.0F
                        : (grid_density != nullptr
                               ? std::tanh(std::max(0.0F,
                                                   grid_density[cell]) * 0.2F)
                               : 0.0F);
                    color = grid_solid != nullptr && grid_solid[cell] != 0U
                        ? Vec3{1.0F, 0.45F, 0.10F}
                        : Vec3{0.10F, 0.72F, 1.0F};
                    break;
                case MetalSmokeDebugMode::velocity:
                    if (grid_velocity != nullptr)
                        value = std::tanh(std::sqrt(dot(
                            grid_velocity[cell], grid_velocity[cell])));
                    color = {0.08F, 0.35F + 0.65F * value, 1.0F};
                    break;
                case MetalSmokeDebugMode::pressure:
                    if (grid_pressure != nullptr)
                        value = std::tanh(std::abs(grid_pressure[cell]) * 0.08F);
                    color = grid_pressure != nullptr && grid_pressure[cell] < 0.0F
                        ? Vec3{0.12F, 0.45F, 1.0F}
                        : Vec3{1.0F, 0.18F, 0.10F};
                    break;
                case MetalSmokeDebugMode::density_temperature:
                    if (grid_density != nullptr)
                        value = std::tanh(
                            std::max(0.0F, grid_density[cell]) * 0.2F);
                    if (grid_temperature != nullptr &&
                        grid_density != nullptr && grid_density[cell] > 1.0e-6F) {
                        const float heat = std::tanh(std::abs(
                            grid_temperature[cell] / grid_density[cell]) *
                            0.05F);
                        color = {0.12F + 0.88F * heat,
                                 0.20F + 0.55F * value,
                                 1.0F - 0.80F * heat};
                    }
                    break;
                case MetalSmokeDebugMode::vorticity:
                    if (grid_vorticity != nullptr)
                        value = std::tanh(std::sqrt(dot(
                            grid_vorticity[cell], grid_vorticity[cell])));
                    color = {0.80F + 0.20F * value, 0.10F, 0.90F};
                    break;
                case MetalSmokeDebugMode::divergence:
                    if (grid_divergence != nullptr)
                        value = std::tanh(std::abs(grid_divergence[cell]));
                    color = grid_divergence != nullptr &&
                                    grid_divergence[cell] < 0.0F
                        ? Vec3{0.10F, 0.42F, 1.0F}
                        : Vec3{1.0F, 0.18F, 0.10F};
                    break;
                case MetalSmokeDebugMode::none: break;
                }
                if (value < 0.025F) continue;
                const std::uint64_t z = cell / layer;
                const std::uint64_t y =
                    (cell / view.grid_resolution) %
                    view.grid_vertical_resolution;
                const std::uint64_t x = cell % view.grid_resolution;
                const Vec3 position{
                    view.grid_minimum.x + (static_cast<float>(x) + 0.5F) *
                        view.grid_spacing,
                    view.grid_minimum.y + (static_cast<float>(y) + 0.5F) *
                        view.grid_spacing,
                    view.grid_minimum.z + (static_cast<float>(z) + 0.5F) *
                        view.grid_spacing};
                frame.smoke_particles.push_back(
                    {{position.x, position.y, position.z},
                     {color.x, color.y, color.z,
                      0.18F + 0.48F * value},
                     view.grid_spacing * 0.18F});
            }
        }
    }
    if (debug != nullptr && debug->rigid_contacts) {
        const auto contacts = runtime.world.rigid_contacts();
        const RigidContactEvent *events = span_data(contacts.events);
        if (contacts.event_count != 0U && events == nullptr) {
            error = "rigid contact debug view is unavailable";
            return false;
        }
        for (std::uint32_t index = 0U; index < contacts.event_count; ++index) {
            const RigidContactEvent &contact = events[index];
            frame.particles.push_back(
                {{contact.position.x, contact.position.y, contact.position.z},
                 {1.0F, 0.18F, 0.06F, 1.0F}, 0.035F});
        }
    }
    return true;
}

std::array<std::uint8_t, 7> ui_glyph(char character) {
    switch (character) {
    case 'A': return {14, 17, 17, 31, 17, 17, 17};
    case 'B': return {30, 17, 17, 30, 17, 17, 30};
    case 'C': return {14, 17, 16, 16, 16, 17, 14};
    case 'D': return {30, 17, 17, 17, 17, 17, 30};
    case 'E': return {31, 16, 16, 30, 16, 16, 31};
    case 'F': return {31, 16, 16, 30, 16, 16, 16};
    case 'G': return {14, 17, 16, 23, 17, 17, 14};
    case 'H': return {17, 17, 17, 31, 17, 17, 17};
    case 'I': return {14, 4, 4, 4, 4, 4, 14};
    case 'J': return {7, 2, 2, 2, 18, 18, 12};
    case 'K': return {17, 18, 20, 24, 20, 18, 17};
    case 'L': return {16, 16, 16, 16, 16, 16, 31};
    case 'M': return {17, 27, 21, 21, 17, 17, 17};
    case 'N': return {17, 25, 21, 19, 17, 17, 17};
    case 'O': return {14, 17, 17, 17, 17, 17, 14};
    case 'P': return {30, 17, 17, 30, 16, 16, 16};
    case 'Q': return {14, 17, 17, 17, 21, 18, 13};
    case 'R': return {30, 17, 17, 30, 20, 18, 17};
    case 'S': return {15, 16, 16, 14, 1, 1, 30};
    case 'T': return {31, 4, 4, 4, 4, 4, 4};
    case 'U': return {17, 17, 17, 17, 17, 17, 14};
    case 'V': return {17, 17, 17, 17, 17, 10, 4};
    case 'W': return {17, 17, 17, 21, 21, 21, 10};
    case 'X': return {17, 17, 10, 4, 10, 17, 17};
    case 'Y': return {17, 17, 10, 4, 4, 4, 4};
    case 'Z': return {31, 1, 2, 4, 8, 16, 31};
    case '0': return {14, 17, 19, 21, 25, 17, 14};
    case '1': return {4, 12, 4, 4, 4, 4, 14};
    case '2': return {14, 17, 1, 2, 4, 8, 31};
    case '3': return {30, 1, 1, 14, 1, 1, 30};
    case '4': return {2, 6, 10, 18, 31, 2, 2};
    case '5': return {31, 16, 16, 30, 1, 1, 30};
    case '6': return {14, 16, 16, 30, 17, 17, 14};
    case '7': return {31, 1, 2, 4, 8, 8, 8};
    case '8': return {14, 17, 17, 14, 17, 17, 14};
    case '9': return {14, 17, 17, 15, 1, 1, 14};
    case '.': return {0, 0, 0, 0, 0, 12, 12};
    case ':': return {0, 12, 12, 0, 12, 12, 0};
    case '-': return {0, 0, 0, 31, 0, 0, 0};
    case '/': return {1, 2, 2, 4, 8, 8, 16};
    default: return {};
    }
}

void append_ui_rect(std::vector<UiVertex> &vertices, float left, float top,
                    float right, float bottom, GalleryColor color) {
    const float red = color.red / 255.0F;
    const float green = color.green / 255.0F;
    const float blue = color.blue / 255.0F;
    const float alpha = color.alpha / 255.0F;
    const auto vertex = [&](float x, float y) {
        return UiVertex{{x, y}, {red, green, blue, alpha}};
    };
    vertices.push_back(vertex(left, top));
    vertices.push_back(vertex(left, bottom));
    vertices.push_back(vertex(right, bottom));
    vertices.push_back(vertex(left, top));
    vertices.push_back(vertex(right, bottom));
    vertices.push_back(vertex(right, top));
}

void append_ui_text(std::vector<UiVertex> &vertices, int x, int y,
                    std::string_view value, GalleryColor color,
                    int scale = 2) {
    for (char character : value) {
        if (character >= 'a' && character <= 'z')
            character = static_cast<char>(character - 'a' + 'A');
        const auto rows = ui_glyph(character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[row] & (1U << (4 - column))) == 0U) continue;
                append_ui_rect(vertices,
                               static_cast<float>(x + column * scale),
                               static_cast<float>(y + row * scale),
                               static_cast<float>(x + (column + 1) * scale),
                               static_cast<float>(y + (row + 1) * scale),
                               color);
            }
        }
        x += 6 * scale;
    }
}

void build_scene_picker(std::vector<UiVertex> &vertices,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext selection) {
    vertices.clear();
    const int center = static_cast<int>(width) / 2;
    constexpr int row_height = 68;
    const int count = static_cast<int>(gallery_entries.size());
    const int visible = std::clamp(
        (static_cast<int>(height) - 110) / row_height, 1, count);
    const int selected =
        static_cast<int>(gallery_context_index(selection));
    const int first =
        std::clamp(selected - visible / 2, 0, count - visible);
    const int panel_height = 81 + row_height * visible;
    const int top =
        std::max(14, (static_cast<int>(height) - panel_height) / 2);
    append_ui_rect(vertices, static_cast<float>(center - 255),
                   static_cast<float>(top),
                   static_cast<float>(center + 255),
                   static_cast<float>(top + panel_height),
                   {4, 10, 16, 230});
    append_ui_text(vertices, center - 225, top + 24, "SCENES",
                   {110, 225, 255, 255}, 3);
    append_ui_text(vertices, center - 70, top + 34,
                   "UP DOWN SELECT   ENTER LOAD   TAB CLOSE",
                   {185, 220, 235, 255}, 1);

    int y = top + 78;
    for (int index = first; index < first + visible; ++index) {
        const GalleryEntry &entry =
            gallery_entries[static_cast<std::size_t>(index)];
        if (selection == entry.context) {
            append_ui_rect(vertices, static_cast<float>(center - 226),
                           static_cast<float>(y - 6),
                           static_cast<float>(center + 226),
                           static_cast<float>(y + 60),
                           {105, 255, 155, 255});
        }
        append_ui_rect(vertices, static_cast<float>(center - 220),
                       static_cast<float>(y),
                       static_cast<float>(center + 220),
                       static_cast<float>(y + 54), entry.background);
        append_ui_rect(vertices, static_cast<float>(center - 198),
                       static_cast<float>(y + 8),
                       static_cast<float>(center - 160),
                       static_cast<float>(y + 46), entry.icon);
        append_ui_text(vertices, center - 135, y + 6, entry.name,
                       {245, 247, 250, 255}, 2);
        append_ui_text(vertices, center - 135, y + 31, entry.help,
                       {105, 255, 155, 255}, 1);
        y += row_height;
    }
}

void build_count_dialog(std::vector<UiVertex> &vertices,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, std::string_view value,
                        bool invalid) {
    vertices.clear();
    const GalleryEntry &entry = gallery_entry(context);
    const bool fluid =
        entry.count_kind == GalleryCountKind::fluid_particles;
    const int center_x = static_cast<int>(width) / 2;
    const int center_y = static_cast<int>(height) / 2;
    append_ui_rect(vertices, static_cast<float>(center_x - 260),
                   static_cast<float>(center_y - 118),
                   static_cast<float>(center_x + 260),
                   static_cast<float>(center_y + 118), {4, 10, 16, 242});
    append_ui_text(vertices, center_x - 220, center_y - 88,
                   fluid ? "FLUID PARTICLE CAP" : "DUMP SPHERES",
                   fluid ? GalleryColor{35, 150, 255, 255}
                         : GalleryColor{245, 130, 45, 255},
                   2);
    append_ui_rect(vertices, static_cast<float>(center_x - 220),
                   static_cast<float>(center_y - 30),
                   static_cast<float>(center_x + 220),
                   static_cast<float>(center_y + 20),
                   invalid ? GalleryColor{105, 20, 20, 255}
                           : GalleryColor{27, 38, 48, 255});
    append_ui_text(vertices, center_x - 198, center_y - 17,
                   std::string(fluid ? "MAX " : "COUNT ") +
                       std::string(value),
                   {245, 247, 250, 255}, 2);
    const std::string range = invalid
        ? "USE " + std::to_string(entry.minimum_count) + '-' +
              std::to_string(entry.maximum_count)
        : "MIN " + std::to_string(entry.minimum_count) + "  MAX " +
              std::to_string(entry.maximum_count);
    append_ui_text(vertices, center_x - 220, center_y + 42, range,
                   invalid ? GalleryColor{255, 105, 105, 255}
                           : GalleryColor{160, 190, 210, 255},
                   1);
    append_ui_text(vertices, center_x - 220, center_y + 72,
                   "ENTER APPLY  ESC CANCEL", {160, 190, 210, 255}, 1);
}

void build_brick_dialog(std::vector<UiVertex> &vertices,
                        std::uint32_t width, std::uint32_t height,
                        const std::array<std::string_view, 3> &values,
                        std::size_t selected, bool invalid) {
    vertices.clear();
    const int center_x = static_cast<int>(width) / 2;
    const int center_y = static_cast<int>(height) / 2;
    append_ui_rect(vertices, center_x - 300.0F, center_y - 180.0F,
                   center_x + 300.0F, center_y + 180.0F, {4, 10, 16, 242});
    append_ui_text(vertices, center_x - 260, center_y - 145,
                   "BRICK WALL SETTINGS", {105, 255, 155, 255}, 2);
    constexpr std::array labels{"COUNT  ", "SCALE  ", "WALLS  "};
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int y = center_y - 82 + static_cast<int>(index) * 58;
        append_ui_rect(vertices, center_x - 260.0F, static_cast<float>(y),
                       center_x + 260.0F, static_cast<float>(y + 42),
                       index == selected
                           ? (invalid ? GalleryColor{105, 20, 20, 255}
                                      : GalleryColor{45, 72, 88, 255})
                           : GalleryColor{27, 38, 48, 255});
        append_ui_text(vertices, center_x - 238, y + 12,
                       std::string(labels[index]) + std::string(values[index]),
                       {245, 247, 250, 255}, 2);
    }
    append_ui_text(vertices, center_x - 260, center_y + 106,
        invalid ? "COUNT 1-4096  SCALE .5-2  WALLS 1-16"
                : "TAB FIELD  ENTER APPLY AND RESTART",
        invalid ? GalleryColor{255, 105, 105, 255}
                : GalleryColor{160, 190, 210, 255}, 1);
    append_ui_text(vertices, center_x - 260, center_y + 138,
        "C AUTO-CALIBRATE  V VERIFY AND SAVE  ESC CANCEL",
        {160, 190, 210, 255}, 1);
}

void build_status_overlay(std::vector<UiVertex> &vertices,
                          std::string_view status) {
    vertices.clear();
    if (status.empty()) return;
    const int width = static_cast<int>(status.size()) * 12 + 28;
    append_ui_rect(vertices, 14.0F, 14.0F, static_cast<float>(14 + width),
                   48.0F, {4, 10, 16, 210});
    append_ui_text(vertices, 28, 24, status, {185, 235, 255, 255}, 2);
}

class Renderer {
  public:
    Renderer() = default;
    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;

    ~Renderer() {
        if (inflight_semaphore_ == nullptr) return;
        for (std::size_t index = 0U; index < k_buffer_count; ++index)
            dispatch_semaphore_wait(inflight_semaphore_,
                                    DISPATCH_TIME_FOREVER);
        for (std::size_t index = 0U; index < k_buffer_count; ++index)
            dispatch_semaphore_signal(inflight_semaphore_);
    }

    bool finish(std::string &error) {
        for (std::size_t index = 0U; index < k_buffer_count; ++index) {
            if (pending_commands_[index] == nil) continue;
            [pending_commands_[index] waitUntilCompleted];
            if (!check_command(pending_commands_[index], error)) return false;
            pending_commands_[index] = nil;
        }
        return true;
    }

    bool create(GLFWwindow *window, id<MTLDevice> device,
                std::string &error) {
        @autoreleasepool {
            device_ = device;
            command_queue_ = [device_ newCommandQueue];
            if (device_ == nil || command_queue_ == nil) {
                error = "could not create the gallery command queue";
                return false;
            }
            dispatch_data_t data = dispatch_data_create(
                parallel_mater_metal_gallery_metallib_data,
                static_cast<std::size_t>(
                    parallel_mater_metal_gallery_metallib_data_len),
                dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *native_error = nil;
            id<MTLLibrary> library =
                [device_ newLibraryWithData:data error:&native_error];
            if (library == nil) {
                error = native_error.localizedDescription.UTF8String;
                return false;
            }
            id<MTLFunction> triangle_vertex =
                [library newFunctionWithName:@"gallery_vertex"];
            id<MTLFunction> rigid_vertex =
                [library newFunctionWithName:@"gallery_rigid_vertex"];
            id<MTLFunction> triangle_fragment =
                [library newFunctionWithName:@"gallery_fragment"];
            id<MTLFunction> sky_vertex =
                [library newFunctionWithName:@"gallery_sky_vertex"];
            id<MTLFunction> sky_fragment =
                [library newFunctionWithName:@"gallery_sky_fragment"];
            id<MTLFunction> particle_vertex =
                [library newFunctionWithName:@"gallery_particle_vertex"];
            id<MTLFunction> particle_fragment =
                [library newFunctionWithName:@"gallery_particle_fragment"];
            id<MTLFunction> ui_vertex =
                [library newFunctionWithName:@"gallery_ui_vertex"];
            id<MTLFunction> ui_fragment =
                [library newFunctionWithName:@"gallery_ui_fragment"];
            if (triangle_vertex == nil || rigid_vertex == nil ||
                triangle_fragment == nil ||
                sky_vertex == nil || sky_fragment == nil ||
                particle_vertex == nil || particle_fragment == nil ||
                ui_vertex == nil || ui_fragment == nil) {
                error = "embedded Metal gallery functions are incomplete";
                return false;
            }
            if (!create_pipeline(triangle_vertex, triangle_fragment, false,
                                 triangle_pipeline_, native_error) ||
                !create_pipeline(rigid_vertex, triangle_fragment, false,
                                 rigid_pipeline_, native_error) ||
                !create_pipeline(sky_vertex, sky_fragment, false,
                                 sky_pipeline_, native_error) ||
                !create_pipeline(particle_vertex, particle_fragment, true,
                                 particle_pipeline_, native_error) ||
                !create_pipeline(ui_vertex, ui_fragment, true,
                                 ui_pipeline_, native_error)) {
                error = native_error != nil
                    ? native_error.localizedDescription.UTF8String
                    : "could not create a Metal gallery pipeline";
                return false;
            }
            MTLDepthStencilDescriptor *depth_descriptor =
                [[MTLDepthStencilDescriptor alloc] init];
            depth_descriptor.depthCompareFunction = MTLCompareFunctionLess;
            depth_descriptor.depthWriteEnabled = YES;
            depth_state_ =
                [device_ newDepthStencilStateWithDescriptor:depth_descriptor];
            depth_descriptor.depthWriteEnabled = NO;
            transparent_depth_state_ =
                [device_ newDepthStencilStateWithDescriptor:depth_descriptor];
            depth_descriptor.depthCompareFunction = MTLCompareFunctionAlways;
            ui_depth_state_ =
                [device_ newDepthStencilStateWithDescriptor:depth_descriptor];
            if (depth_state_ == nil || transparent_depth_state_ == nil ||
                ui_depth_state_ == nil) {
                error = "could not create the gallery depth state";
                return false;
            }
            inflight_semaphore_ =
                dispatch_semaphore_create(k_buffer_count);
            if (window != nullptr) {
                NSWindow *native_window = glfwGetCocoaWindow(window);
                NSView *view = native_window.contentView;
                layer_ = [CAMetalLayer layer];
                layer_.device = device_;
                layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
                layer_.framebufferOnly = YES;
                layer_.contentsScale = native_window.backingScaleFactor;
                view.wantsLayer = YES;
                view.layer = layer_;
            }
            return true;
        }
    }

    bool draw(GLFWwindow *window, const RenderFrame &frame,
              const GalleryEntry &entry, Camera camera,
              const GalleryOverlay &overlay,
              std::string &error) {
        @autoreleasepool {
            (void)window;
            const int width = static_cast<int>(brick_render_width);
            const int height = static_cast<int>(brick_render_height);
            layer_.drawableSize = CGSizeMake(width, height);
            id<CAMetalDrawable> drawable = [layer_ nextDrawable];
            if (drawable == nil) return true;
            const std::size_t slot = next_slot_++ % k_buffer_count;
            dispatch_semaphore_wait(inflight_semaphore_,
                                    DISPATCH_TIME_FOREVER);
            if (pending_commands_[slot] != nil &&
                !check_command(pending_commands_[slot], error)) {
                dispatch_semaphore_signal(inflight_semaphore_);
                return false;
            }
            pending_commands_[slot] = nil;
            if (!ensure_depth(static_cast<std::uint32_t>(width),
                              static_cast<std::uint32_t>(height), error) ||
                !upload(frame, overlay,
                        static_cast<std::uint32_t>(width),
                        static_cast<std::uint32_t>(height), slot, error)) {
                dispatch_semaphore_signal(inflight_semaphore_);
                return false;
            }
            id<MTLCommandBuffer> command_buffer =
                [command_queue_ commandBuffer];
            if (!encode(command_buffer, drawable.texture, depth_texture_,
                        static_cast<std::uint32_t>(width),
                        static_cast<std::uint32_t>(height), frame, entry,
                        camera, slot, ui_vertices_.size(), error)) {
                dispatch_semaphore_signal(inflight_semaphore_);
                return false;
            }
            [command_buffer presentDrawable:drawable];
            dispatch_semaphore_t semaphore = inflight_semaphore_;
            [command_buffer addCompletedHandler:
                ^(id<MTLCommandBuffer>) {
                    dispatch_semaphore_signal(semaphore);
                }];
            pending_commands_[slot] = command_buffer;
            [command_buffer commit];
            return true;
        }
    }

    bool capture(std::uint32_t width, std::uint32_t height,
                 const RenderFrame &frame, const GalleryEntry &entry,
                 Camera camera,
                 std::vector<std::uint32_t> &rgba, std::string &error) {
        @autoreleasepool {
            if (!drain(error)) return false;
            if (!ensure_depth(width, height, error) ||
                !ensure_offscreen(width, height, error) ||
                !upload(frame, {}, width, height, 0U, error)) {
                return false;
            }
            const std::size_t row_bytes =
                static_cast<std::size_t>(width) * 4U;
            const std::size_t total_bytes =
                row_bytes * static_cast<std::size_t>(height);
            if (readback_buffer_ == nil ||
                readback_buffer_.length < total_bytes) {
                readback_buffer_ = [device_
                    newBufferWithLength:total_bytes
                               options:MTLResourceStorageModeShared];
            }
            if (readback_buffer_ == nil) {
                error = "could not allocate the gallery readback buffer";
                return false;
            }
            id<MTLCommandBuffer> command_buffer =
                [command_queue_ commandBuffer];
            if (!encode(command_buffer, offscreen_texture_, depth_texture_,
                        width, height, frame, entry, camera, 0U, 0U, error)) {
                return false;
            }
            id<MTLBlitCommandEncoder> blit =
                [command_buffer blitCommandEncoder];
            [blit copyFromTexture:offscreen_texture_
                     sourceSlice:0U
                     sourceLevel:0U
                    sourceOrigin:MTLOriginMake(0U, 0U, 0U)
                      sourceSize:MTLSizeMake(width, height, 1U)
                        toBuffer:readback_buffer_
               destinationOffset:0U
          destinationBytesPerRow:row_bytes
        destinationBytesPerImage:total_bytes];
            [blit endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (!check_command(command_buffer, error)) return false;
            rgba.resize(static_cast<std::size_t>(width) * height);
            const auto *source = static_cast<const std::uint8_t *>(
                readback_buffer_.contents);
            auto *destination =
                reinterpret_cast<std::uint8_t *>(rgba.data());
            for (std::size_t pixel = 0U; pixel < rgba.size(); ++pixel) {
                destination[pixel * 4U] = source[pixel * 4U + 2U];
                destination[pixel * 4U + 1U] = source[pixel * 4U + 1U];
                destination[pixel * 4U + 2U] = source[pixel * 4U];
                destination[pixel * 4U + 3U] = source[pixel * 4U + 3U];
            }
            return true;
        }
    }

  private:
    struct CachedRigidMesh {
        std::uint32_t mesh_index{};
        std::uint64_t hash{};
        std::size_t vertex_count{};
        id<MTLBuffer> __strong buffer{nil};
    };

    bool create_pipeline(id<MTLFunction> vertex, id<MTLFunction> fragment,
                         bool blending,
                         id<MTLRenderPipelineState> __strong &output,
                         NSError * __strong &error) {
        MTLRenderPipelineDescriptor *descriptor =
            [[MTLRenderPipelineDescriptor alloc] init];
        descriptor.vertexFunction = vertex;
        descriptor.fragmentFunction = fragment;
        descriptor.colorAttachments[0].pixelFormat =
            MTLPixelFormatBGRA8Unorm;
        descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        if (blending) {
            descriptor.colorAttachments[0].blendingEnabled = YES;
            descriptor.colorAttachments[0].sourceRGBBlendFactor =
                MTLBlendFactorSourceAlpha;
            descriptor.colorAttachments[0].destinationRGBBlendFactor =
                MTLBlendFactorOneMinusSourceAlpha;
            descriptor.colorAttachments[0].sourceAlphaBlendFactor =
                MTLBlendFactorOne;
            descriptor.colorAttachments[0].destinationAlphaBlendFactor =
                MTLBlendFactorOneMinusSourceAlpha;
        }
        NSError *pipeline_error = nil;
        output = [device_ newRenderPipelineStateWithDescriptor:descriptor
                                                         error:&pipeline_error];
        error = pipeline_error;
        return output != nil;
    }

    bool ensure_depth(std::uint32_t width, std::uint32_t height,
                      std::string &error) {
        if (depth_texture_ != nil && width_ == width && height_ == height)
            return true;
        width_ = width;
        height_ = height;
        MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                         width:width
                                        height:height
                                     mipmapped:NO];
        descriptor.usage = MTLTextureUsageRenderTarget;
        descriptor.storageMode = MTLStorageModePrivate;
        depth_texture_ = [device_ newTextureWithDescriptor:descriptor];
        if (depth_texture_ == nil) {
            error = "could not allocate the gallery depth texture";
            return false;
        }
        return true;
    }

    bool ensure_offscreen(std::uint32_t width, std::uint32_t height,
                          std::string &error) {
        if (offscreen_texture_ != nil &&
            offscreen_texture_.width == width &&
            offscreen_texture_.height == height) {
            return true;
        }
        MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:width
                                        height:height
                                     mipmapped:NO];
        descriptor.usage = MTLTextureUsageRenderTarget |
                           MTLTextureUsageShaderRead;
        descriptor.storageMode = MTLStorageModePrivate;
        offscreen_texture_ = [device_ newTextureWithDescriptor:descriptor];
        if (offscreen_texture_ == nil) {
            error = "could not allocate the offscreen gallery texture";
            return false;
        }
        return true;
    }

    bool ensure_buffer(id<MTLBuffer> __strong &buffer, std::size_t bytes,
                       std::string_view name, std::string &error) {
        bytes = std::max<std::size_t>(bytes, 256U);
        if (buffer != nil && buffer.length >= bytes) return true;
        std::size_t capacity = 256U;
        while (capacity < bytes &&
               capacity <= std::numeric_limits<std::size_t>::max() / 2U) {
            capacity *= 2U;
        }
        if (capacity < bytes ||
            capacity > std::numeric_limits<NSUInteger>::max()) {
            error = std::string(name) + " buffer is too large";
            return false;
        }
        buffer = [device_
            newBufferWithLength:capacity
                       options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            error = "could not allocate the " + std::string(name) + " buffer";
            return false;
        }
        return true;
    }

    bool upload(const RenderFrame &frame,
                const GalleryOverlay &overlay,
                std::uint32_t width, std::uint32_t height,
                std::size_t slot, std::string &error) {
        if (overlay.picker_selection.has_value())
            build_scene_picker(ui_vertices_, width, height,
                               *overlay.picker_selection);
        else if (overlay.count_dialog_visible) {
            if (overlay.brick_dialog)
                build_brick_dialog(ui_vertices_, width, height,
                                   overlay.brick_values, overlay.brick_field,
                                   overlay.count_value_invalid);
            else
                build_count_dialog(ui_vertices_, width, height, overlay.context,
                                   overlay.count_value,
                                   overlay.count_value_invalid);
        }
        else
            build_status_overlay(ui_vertices_, overlay.status);
        const std::size_t triangle_bytes =
            frame.triangles.size() * sizeof(GalleryVertex);
        const std::size_t particle_bytes =
            frame.particles.size() * sizeof(ParticleVertex);
        const std::size_t smoke_particle_bytes =
            frame.smoke_particles.size() * sizeof(ParticleVertex);
        const std::size_t ui_bytes = ui_vertices_.size() * sizeof(UiVertex);
        rigid_instances_.clear();
        rigid_mesh_indices_.clear();
        rigid_instance_offsets_.clear();
        std::size_t rigid_instance_count = 0U;
        for (const RigidBatch &batch : frame.rigid_batches)
            rigid_instance_count += batch.instances.size();
        rigid_instances_.reserve(rigid_instance_count);
        for (const RigidBatch &batch : frame.rigid_batches) {
            std::uint64_t hash = 1469598103934665603ULL;
            const auto *bytes = reinterpret_cast<const std::uint8_t *>(
                batch.vertices.data());
            for (std::size_t index = 0U;
                 index < batch.vertices.size() * sizeof(GalleryVertex);
                 ++index) {
                hash ^= bytes[index];
                hash *= 1099511628211ULL;
            }
            auto cached = std::find_if(
                rigid_mesh_cache_.begin(), rigid_mesh_cache_.end(),
                [&](const CachedRigidMesh &candidate) {
                    return candidate.mesh_index == batch.mesh_index &&
                        candidate.hash == hash &&
                        candidate.vertex_count == batch.vertices.size();
                });
            if (cached == rigid_mesh_cache_.end()) {
                const std::size_t bytes_size =
                    batch.vertices.size() * sizeof(GalleryVertex);
                id<MTLBuffer> buffer = bytes_size == 0U
                    ? [device_ newBufferWithLength:1U
                                           options:MTLResourceStorageModeShared]
                    : [device_ newBufferWithBytes:batch.vertices.data()
                                             length:bytes_size
                                            options:MTLResourceStorageModeShared];
                if (buffer == nil) {
                    error = "could not allocate a static rigid mesh buffer";
                    return false;
                }
                rigid_mesh_cache_.push_back(
                    {batch.mesh_index, hash, batch.vertices.size(), buffer});
                cached = rigid_mesh_cache_.end() - 1;
            }
            rigid_mesh_indices_.push_back(static_cast<std::size_t>(
                cached - rigid_mesh_cache_.begin()));
            rigid_instance_offsets_.push_back(rigid_instances_.size());
            rigid_instances_.insert(rigid_instances_.end(),
                                    batch.instances.begin(),
                                    batch.instances.end());
        }
        const std::size_t rigid_instance_bytes =
            rigid_instances_.size() * sizeof(RigidInstance);
        if (!ensure_buffer(triangle_buffers_[slot], triangle_bytes,
                           "triangle", error) ||
            !ensure_buffer(rigid_instance_buffers_[slot],
                           rigid_instance_bytes, "rigid instance", error) ||
            !ensure_buffer(particle_buffers_[slot], particle_bytes,
                           "particle", error) ||
            !ensure_buffer(smoke_particle_buffers_[slot],
                           smoke_particle_bytes, "smoke particle", error) ||
            !ensure_buffer(ui_buffers_[slot], ui_bytes, "UI", error)) {
            return false;
        }
        if (triangle_bytes != 0U)
            std::memcpy(triangle_buffers_[slot].contents,
                        frame.triangles.data(),
                        triangle_bytes);
        if (rigid_instance_bytes != 0U)
            std::memcpy(rigid_instance_buffers_[slot].contents,
                        rigid_instances_.data(), rigid_instance_bytes);
        if (particle_bytes != 0U)
            std::memcpy(particle_buffers_[slot].contents,
                        frame.particles.data(),
                        particle_bytes);
        if (smoke_particle_bytes != 0U)
            std::memcpy(smoke_particle_buffers_[slot].contents,
                        frame.smoke_particles.data(), smoke_particle_bytes);
        if (ui_bytes != 0U)
            std::memcpy(ui_buffers_[slot].contents, ui_vertices_.data(),
                        ui_bytes);
        return true;
    }

    bool encode(id<MTLCommandBuffer> command_buffer,
                id<MTLTexture> color, id<MTLTexture> depth,
                std::uint32_t width, std::uint32_t height,
                const RenderFrame &frame, const GalleryEntry &entry,
                Camera camera, std::size_t slot,
                std::size_t ui_vertex_count,
                std::string &error) {
        if (command_buffer == nil || color == nil || depth == nil) {
            error = "could not allocate Metal gallery frame resources";
            return false;
        }
        MTLRenderPassDescriptor *pass =
            [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = color;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(
            entry.background.red / 255.0,
            entry.background.green / 255.0,
            entry.background.blue / 255.0, 1.0);
        pass.depthAttachment.texture = depth;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionDontCare;
        pass.depthAttachment.clearDepth = 1.0;
        id<MTLRenderCommandEncoder> encoder =
            [command_buffer renderCommandEncoderWithDescriptor:pass];
        if (encoder == nil) {
            error = "could not encode the Metal gallery frame";
            return false;
        }
        const float aspect = static_cast<float>(width) /
                             static_cast<float>(height);
        const Vec3 forward = normalize(subtract(camera.target, camera.eye));
        const Vec3 right = normalize(cross(forward, camera.up));
        const Vec3 corrected_up = normalize(cross(right, forward));
        constexpr float radians = 3.14159265358979323846F / 180.0F;
        const float vertical_scale = std::tan(
            camera.vertical_field_of_view_degrees * radians * 0.5F);
        const Vec3 camera_u = scale(right, vertical_scale * aspect);
        const Vec3 camera_v = scale(corrected_up, vertical_scale);
        const GalleryUniforms uniforms{
            view_projection(aspect, camera), static_cast<float>(height),
            {camera.eye.x, camera.eye.y, camera.eye.z},
            {camera_u.x, camera_u.y, camera_u.z},
            {camera_v.x, camera_v.y, camera_v.z},
            {forward.x, forward.y, forward.z}};
        [encoder setDepthStencilState:ui_depth_state_];
        [encoder setRenderPipelineState:sky_pipeline_];
        [encoder setVertexBytes:&uniforms
                         length:sizeof(uniforms)
                        atIndex:1];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0 vertexCount:3];
        [encoder setDepthStencilState:depth_state_];
        [encoder setCullMode:MTLCullModeNone];
        if (!frame.triangles.empty()) {
            [encoder setRenderPipelineState:triangle_pipeline_];
            [encoder setVertexBuffer:triangle_buffers_[slot]
                             offset:0 atIndex:0];
            [encoder setVertexBytes:&uniforms
                             length:sizeof(uniforms)
                            atIndex:1];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:frame.triangles.size()];
        }
        if (!frame.rigid_batches.empty()) {
            [encoder setRenderPipelineState:rigid_pipeline_];
            [encoder setVertexBytes:&uniforms
                             length:sizeof(uniforms)
                            atIndex:1];
            for (std::size_t batch_index = 0U;
                 batch_index < frame.rigid_batches.size(); ++batch_index) {
                const RigidBatch &batch = frame.rigid_batches[batch_index];
                [encoder setVertexBuffer:
                             rigid_mesh_cache_[
                                 rigid_mesh_indices_[batch_index]].buffer
                                 offset:0 atIndex:0];
                [encoder setVertexBuffer:rigid_instance_buffers_[slot]
                                 offset:rigid_instance_offsets_[batch_index] *
                                        sizeof(RigidInstance)
                                atIndex:2];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                            vertexStart:0
                            vertexCount:batch.vertices.size()
                          instanceCount:batch.instances.size()];
            }
        }
        if (!frame.particles.empty()) {
            [encoder setRenderPipelineState:particle_pipeline_];
            [encoder setVertexBuffer:particle_buffers_[slot]
                             offset:0 atIndex:0];
            [encoder setVertexBytes:&uniforms
                             length:sizeof(uniforms)
                            atIndex:1];
            [encoder drawPrimitives:MTLPrimitiveTypePoint
                        vertexStart:0
                        vertexCount:frame.particles.size()];
        }
        if (!frame.smoke_particles.empty()) {
            [encoder setDepthStencilState:transparent_depth_state_];
            [encoder setRenderPipelineState:particle_pipeline_];
            [encoder setVertexBuffer:smoke_particle_buffers_[slot]
                             offset:0 atIndex:0];
            [encoder setVertexBytes:&uniforms
                             length:sizeof(uniforms)
                            atIndex:1];
            [encoder drawPrimitives:MTLPrimitiveTypePoint
                        vertexStart:0
                        vertexCount:frame.smoke_particles.size()];
        }
        if (ui_vertex_count != 0U) {
            const UiUniforms ui_uniforms{{static_cast<float>(width),
                                          static_cast<float>(height)}};
            [encoder setDepthStencilState:ui_depth_state_];
            [encoder setRenderPipelineState:ui_pipeline_];
            [encoder setVertexBuffer:ui_buffers_[slot]
                             offset:0 atIndex:0];
            [encoder setVertexBytes:&ui_uniforms
                             length:sizeof(ui_uniforms)
                            atIndex:1];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0 vertexCount:ui_vertex_count];
        }
        [encoder endEncoding];
        return true;
    }

    bool drain(std::string &error) {
        if (inflight_semaphore_ == nullptr) return true;
        for (std::size_t index = 0U; index < k_buffer_count; ++index)
            dispatch_semaphore_wait(inflight_semaphore_,
                                    DISPATCH_TIME_FOREVER);
        bool result = true;
        for (std::size_t index = 0U; index < k_buffer_count; ++index) {
            if (pending_commands_[index] != nil &&
                !check_command(pending_commands_[index], error)) {
                result = false;
            }
            pending_commands_[index] = nil;
            dispatch_semaphore_signal(inflight_semaphore_);
        }
        return result;
    }

    bool check_command(id<MTLCommandBuffer> command_buffer,
                       std::string &error) {
        if (command_buffer.status != MTLCommandBufferStatusError) return true;
        error = command_buffer.error != nil
            ? command_buffer.error.localizedDescription.UTF8String
            : "Metal gallery command failed";
        return false;
    }

    static constexpr std::size_t k_buffer_count = 3U;
    id<MTLDevice> device_{nil};
    id<MTLCommandQueue> command_queue_{nil};
    id<MTLRenderPipelineState> triangle_pipeline_{nil};
    id<MTLRenderPipelineState> rigid_pipeline_{nil};
    id<MTLRenderPipelineState> sky_pipeline_{nil};
    id<MTLRenderPipelineState> particle_pipeline_{nil};
    id<MTLRenderPipelineState> ui_pipeline_{nil};
    id<MTLDepthStencilState> depth_state_{nil};
    id<MTLDepthStencilState> transparent_depth_state_{nil};
    id<MTLDepthStencilState> ui_depth_state_{nil};
    id<MTLBuffer> __strong triangle_buffers_[k_buffer_count]{};
    id<MTLBuffer> __strong rigid_instance_buffers_[k_buffer_count]{};
    id<MTLBuffer> __strong particle_buffers_[k_buffer_count]{};
    id<MTLBuffer> __strong smoke_particle_buffers_[k_buffer_count]{};
    id<MTLBuffer> __strong ui_buffers_[k_buffer_count]{};
    id<MTLCommandBuffer> __strong pending_commands_[k_buffer_count]{};
    id<MTLBuffer> readback_buffer_{nil};
    id<MTLTexture> depth_texture_{nil};
    id<MTLTexture> offscreen_texture_{nil};
    CAMetalLayer *layer_{nil};
    dispatch_semaphore_t inflight_semaphore_{nullptr};
    std::size_t next_slot_{};
    std::vector<UiVertex> ui_vertices_{};
    std::vector<CachedRigidMesh> rigid_mesh_cache_{};
    std::vector<RigidInstance> rigid_instances_{};
    std::vector<std::size_t> rigid_mesh_indices_{};
    std::vector<std::size_t> rigid_instance_offsets_{};
    std::uint32_t width_{};
    std::uint32_t height_{};
};

bool validate_render(const std::vector<std::uint32_t> &pixels,
                     std::string &error) {
    if (pixels.empty()) {
        error = "renderer produced no pixels";
        return false;
    }
    const auto *bytes =
        reinterpret_cast<const std::uint8_t *>(pixels.data());
    unsigned minimum = 3U * 255U;
    unsigned maximum = 0U;
    for (std::size_t index = 0U; index < pixels.size(); ++index) {
        const unsigned luminance = bytes[index * 4U] +
                                   bytes[index * 4U + 1U] +
                                   bytes[index * 4U + 2U];
        minimum = std::min(minimum, luminance);
        maximum = std::max(maximum, luminance);
    }
    if (maximum < 48U || maximum - minimum < 48U) {
        error = "renderer produced a black or nearly uniform frame";
        return false;
    }
    return true;
}

bool write_ppm(const std::filesystem::path &path,
               const std::vector<std::uint32_t> &pixels,
               std::uint32_t width, std::uint32_t height) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    output << "P6\n" << width << ' ' << height << "\n255\n";
    const auto *bytes =
        reinterpret_cast<const std::uint8_t *>(pixels.data());
    for (std::uint32_t row = 0U; row < height; ++row) {
        for (std::uint32_t column = 0U; column < width; ++column) {
            const std::size_t offset =
                (static_cast<std::size_t>(row) * width + column) * 4U;
            output.write(reinterpret_cast<const char *>(bytes + offset), 3);
        }
    }
    return static_cast<bool>(output);
}

std::string file_stem(const GalleryEntry &entry) {
    std::string result;
    for (const char character : entry.name) {
        if (character >= 'A' && character <= 'Z')
            result.push_back(static_cast<char>(character - 'A' + 'a'));
        else if ((character >= 'a' && character <= 'z') ||
                 (character >= '0' && character <= '9'))
            result.push_back(character);
        else if (result.empty() || result.back() != '-')
            result.push_back('-');
    }
    while (!result.empty() && result.back() == '-') result.pop_back();
    return result;
}

std::filesystem::path capture_path(const Options &options,
                                   const GalleryEntry &entry,
                                   bool all_scenes) {
    if (!all_scenes) return options.output;
    const std::filesystem::path directory =
        options.output.empty() ? std::filesystem::path("metal-gallery-captures")
                               : options.output;
    return directory / (file_stem(entry) + ".ppm");
}

bool validate_runtime(Runtime &runtime,
                      const std::vector<RigidBodyState> &states,
                      std::string &error) {
    if (states.size() != runtime.scene.rigid_bodies.size()) {
        error = "rigid state count does not match the scene";
        return false;
    }
    for (const RigidBodyState state : states) {
        if (!finite(state)) {
            error = "gallery physics produced a non-finite rigid state";
            return false;
        }
    }
    WorldStatistics statistics;
    if (!check(runtime.world.collect_statistics(statistics),
               "collect gallery statistics", error)) {
        return false;
    }
    if (statistics.rigid_body_count != runtime.instance.rigid_bodies.size() ||
        statistics.cloth_count != runtime.instance.cloths.size() ||
        statistics.soft_body_count != runtime.instance.soft_bodies.size() ||
        statistics.rope_count != runtime.instance.ropes.size() ||
        statistics.fluid_count !=
            (runtime.instance.has_fluid ? 1U : 0U) ||
        statistics.smoke_system_count !=
            (runtime.instance.has_smoke ? 1U : 0U)) {
        error = "gallery world statistics do not match the scene";
        return false;
    }
    return true;
}

std::uint32_t scene_substeps(GalleryContext context) {
    return context == GalleryContext::constraint_hinge ? 8U : 4U;
}

bool step_runtime(Runtime &runtime, Vec3 gravity, std::string &error) {
    if (runtime.fixed_collector.active() &&
        !runtime.fixed_collector.apply_loose_gravity(
            runtime.world, runtime.scene, runtime.instance,
            {0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F}, gravity,
            error)) {
        return false;
    }
    if (!apply_gravity_tilt_overrides(runtime, gravity, error)) return false;
    if (!check(runtime.world.step(
                   {.timestep = k_timestep,
                    .substeps = scene_substeps(runtime.context),
                    .gravity = gravity,
                    .collect_kernel_timings = runtime.collect_kernel_timings,
                    .collect_rigid_contacts =
                        runtime.fixed_collector.active() ||
                        runtime.collect_rigid_contacts}),
               "step gallery world", error)) {
        return false;
    }
    return !runtime.fixed_collector.active() ||
        runtime.fixed_collector.collect(
            runtime.world, runtime.scene, runtime.instance, error);
}

bool step_runtime_async(Runtime &runtime, Vec3 gravity,
                        FrameToken &completion, std::string &error) {
    if (runtime.fixed_collector.active() &&
        !runtime.fixed_collector.apply_loose_gravity(
            runtime.world, runtime.scene, runtime.instance,
            {0.0F, -k_gravity * runtime.scene.gravity_scale, 0.0F}, gravity,
            error)) {
        return false;
    }
    if (!apply_gravity_tilt_overrides(runtime, gravity, error)) return false;
    return check(runtime.world.step_async(
                     {.timestep = k_timestep,
                      .substeps = scene_substeps(runtime.context),
                      .gravity = gravity,
                      .collect_kernel_timings = runtime.collect_kernel_timings,
                      .collect_rigid_contacts =
                          runtime.fixed_collector.active() ||
                          runtime.collect_rigid_contacts},
                     completion),
                 "submit gallery world step", error);
}

Vec3 default_gravity(const Runtime &runtime) {
    return {0.0F, -9.81F * runtime.scene.gravity_scale, 0.0F};
}

bool run_headless_context(const Options &options, GalleryContext context,
                          Renderer &renderer, id<MTLDevice> device,
                          bool all_scenes, std::string &error) {
    Runtime runtime;
    if (!build_runtime(options, context, runtime, error)) return false;
    if (device == nil) {
        const auto native = runtime.world.native_context();
        device = (__bridge id<MTLDevice>)native.device;
        if (!renderer.create(nullptr, device, error)) return false;
    }
    for (std::uint32_t frame = 0U; frame < options.frames; ++frame)
        if (!step_runtime(runtime, default_gravity(runtime), error))
            return false;

    RenderFrame render_frame;
    std::vector<RigidBodyState> states;
    if (!assemble_frame(runtime, render_frame, states, error)) return false;
    std::vector<std::uint32_t> pixels;
    const GalleryEntry &entry = gallery_entry(context);
    if (!renderer.capture(options.width, options.height, render_frame, entry,
                          camera_for_preset(entry.camera), pixels, error)) {
        return false;
    }
    if (options.validate &&
        (!validate_runtime(runtime, states, error) ||
         !validate_render(pixels, error))) {
        return false;
    }
    const std::filesystem::path path =
        capture_path(options, entry, all_scenes);
    if (!path.empty()) {
        std::error_code filesystem_error;
        std::filesystem::create_directories(path.parent_path(),
                                             filesystem_error);
        if ((filesystem_error && !path.parent_path().empty()) ||
            !write_ppm(path, pixels, options.width, options.height)) {
            error = "could not write " + path.string();
            return false;
        }
    }
    WorldStatistics statistics;
    if (!check(runtime.world.collect_statistics(statistics),
               "collect gallery statistics", error)) {
        return false;
    }
    std::size_t rigid_instances = 0U;
    std::size_t rigid_triangles = 0U;
    for (const RigidBatch &batch : render_frame.rigid_batches) {
        rigid_instances += batch.instances.size();
        rigid_triangles +=
            (batch.vertices.size() / 3U) * batch.instances.size();
    }
    std::cout << "PASS " << entry.name << ": " << options.frames
              << " frames, "
              << render_frame.triangles.size() / 3U + rigid_triangles
              << " triangles, " << rigid_instances << " rigid instances, "
              << render_frame.particles.size() +
                     render_frame.smoke_particles.size()
              << " particles\n";
    return true;
}

bool run_headless(const Options &options, std::string &error) {
    Renderer renderer;
    id<MTLDevice> device = nil;
    if (options.all_scenes) {
        const std::filesystem::path directory =
            options.output.empty()
                ? std::filesystem::path("metal-gallery-captures")
                : options.output;
        std::error_code filesystem_error;
        std::filesystem::create_directories(directory, filesystem_error);
        if (filesystem_error) {
            error = "could not create capture directory " +
                    directory.string();
            return false;
        }
        for (const GalleryEntry &entry : gallery_entries) {
            if (!run_headless_context(options, entry.context, renderer,
                                      device, true, error)) {
                error = std::string(entry.name) + ": " + error;
                return false;
            }
            if (device == nil) {
                // Renderer has retained the default device after first create.
                device = MTLCreateSystemDefaultDevice();
            }
        }
        return true;
    }
    return run_headless_context(options, options.context, renderer, device,
                                false, error);
}

struct InteractiveInput {
    CameraController camera{};
    bool ui_visible{};
    bool count_dialog_visible{};
    bool replace_count_value{};
    bool count_value_invalid{};
    std::string count_value{};
    std::array<std::string, 3> brick_values{};
    std::size_t brick_field{};
    bool brick_dialog{};
};

void mouse_button(GLFWwindow *window, int button, int action, int modifiers) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    auto *input = static_cast<InteractiveInput *>(
        glfwGetWindowUserPointer(window));
    if (input == nullptr) return;
    if (action == GLFW_RELEASE) {
        input->camera.end_drag();
        return;
    }
    if (action != GLFW_PRESS || input->ui_visible) return;
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    input->camera.begin_drag((modifiers & GLFW_MOD_SHIFT) != 0
                                 ? CameraDragMode::pan
                                 : CameraDragMode::orbit,
                             x, y);
}

void cursor_position(GLFWwindow *window, double x, double y) {
    auto *input = static_cast<InteractiveInput *>(
        glfwGetWindowUserPointer(window));
    if (input == nullptr) return;
    int height = 0;
    glfwGetWindowSize(window, nullptr, &height);
    input->camera.move_cursor(x, y, height, !input->ui_visible);
}

void scroll(GLFWwindow *window, double, double offset) {
    auto *input = static_cast<InteractiveInput *>(
        glfwGetWindowUserPointer(window));
    if (input != nullptr && !input->ui_visible)
        input->camera.zoom(offset);
}

void character_input(GLFWwindow *window, unsigned int codepoint) {
    auto *input = static_cast<InteractiveInput *>(
        glfwGetWindowUserPointer(window));
    const bool digit = codepoint >= '0' && codepoint <= '9';
    const bool scale_point = input != nullptr && input->brick_dialog &&
        input->brick_field == 1U && codepoint == '.';
    if (input == nullptr || !input->count_dialog_visible ||
        (!digit && !scale_point)) {
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

bool launch_calibration_ball(Runtime &runtime, bool off_center,
                             std::string &error) {
    for (std::size_t index = 0U; index < runtime.scene.rigid_bodies.size(); ++index) {
        if (runtime.scene.rigid_bodies[index].source_name != "Icosphere") continue;
        RigidBodyState state =
            runtime.scene.rigid_bodies[index].options.initial_state;
        state.position = {off_center ? 1.0F : 0.0F, 1.05F, 3.0F};
        state.linear_velocity = {0.0F, 0.0F, -10.0F};
        state.angular_velocity = {};
        return check(runtime.world.set_rigid_body_state(
                         runtime.instance.rigid_bodies[index], state),
                     "launch calibration ball", error);
    }
    error = "brick calibration scene has no ball";
    return false;
}

bool calibration_scene_correct(const Runtime &runtime,
                               const std::vector<RigidBodyState> &states,
                               bool quiet, float scale) {
    if (states.size() != runtime.scene.rigid_bodies.size()) return false;
    unsigned displaced_bricks = 0U;
    for (std::size_t index = 0U; index < states.size(); ++index) {
        const auto &state = states[index];
        const auto &body = runtime.scene.rigid_bodies[index];
        if (!finite(state)) return false;
        const float speed = std::sqrt(
            state.linear_velocity.x * state.linear_velocity.x +
            state.linear_velocity.y * state.linear_velocity.y +
            state.linear_velocity.z * state.linear_velocity.z);
        if (!std::isfinite(speed) || speed > 50.0F) return false;
        if (body.source_name != "Layer1" && body.source_name != "Layer2")
            continue;
        const auto initial = body.options.initial_state.position;
        const float displacement = std::hypot(
            std::hypot(state.position.x - initial.x,
                       state.position.y - initial.y),
            state.position.z - initial.z);
        if (quiet && displacement > std::max(0.03F, 0.04F * scale))
            return false;
        displaced_bricks += !quiet && displacement > 0.03F;
    }
    return quiet || displaced_bricks != 0U;
}

struct CalibrationResult {
    BrickSceneConfig config{};
    parallel_mater::gallery::CalibrationMetrics metrics{};
    parallel_mater::gallery::HardwareIdentity hardware{};
};

bool run_calibration_trial(const Options &base, const BrickSceneConfig &config,
                           GLFWwindow *window, Renderer &renderer,
                           std::uint32_t quiet_frames,
                           std::uint32_t collision_frames,
                           bool off_center,
                           std::vector<parallel_mater::gallery::CalibrationSample> &samples,
                           bool &stable, std::string &error) {
    Options candidate = base;
    candidate.bricks = config;
    Runtime runtime;
    if (!build_runtime(candidate, GalleryContext::rigid_body, runtime, error))
        return false;
    for (std::uint32_t frame = 0U; frame < 30U; ++frame)
        if (!step_runtime(runtime, default_gravity(runtime), error)) return false;

    RenderFrame render_frame;
    std::vector<RigidBodyState> states;
    auto measure = [&](bool collision) {
        const auto begin = std::chrono::steady_clock::now();
        if (!step_runtime(runtime, default_gravity(runtime), error) ||
            !assemble_frame(runtime, render_frame, states, error) ||
            !renderer.draw(window, render_frame,
                gallery_entry(GalleryContext::rigid_body),
                camera_for_preset(camera_preset_for(
                    GalleryContext::rigid_body, config)),
                {.context = GalleryContext::rigid_body,
                 .status = collision ? "CALIBRATING COLLISION"
                                     : "CALIBRATING QUIET"}, error) ||
            !renderer.finish(error)) return false;
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        samples.push_back({milliseconds, collision, true});
        glfwPollEvents();
        return glfwWindowShouldClose(window) == GLFW_FALSE;
    };
    for (std::uint32_t frame = 0U; frame < quiet_frames; ++frame)
        if (!measure(false)) { error = "brick calibration cancelled"; return false; }
    stable = stable && calibration_scene_correct(
        runtime, states, true, config.brick_scale);
    if (!launch_calibration_ball(runtime, off_center, error)) return false;
    for (std::uint32_t frame = 0U; frame < collision_frames; ++frame)
        if (!measure(true)) { error = "brick calibration cancelled"; return false; }
    std::string validation_error;
    stable = stable && validate_runtime(runtime, states, validation_error) &&
        calibration_scene_correct(runtime, states, false, config.brick_scale);
    return true;
}

bool calibrate_bricks(Options &options, GLFWwindow *window, Renderer &renderer,
                      id<MTLDevice> device, CalibrationResult &output,
                      std::string &error) {
    const auto &presets = parallel_mater::gallery::brick_calibration_presets();
    std::optional<std::size_t> best;
    for (std::size_t index = 0U; index < presets.size(); ++index) {
        std::vector<parallel_mater::gallery::CalibrationSample> samples;
        bool stable = true;
        std::string trial_error;
        const auto begin = std::chrono::steady_clock::now();
        if (!run_calibration_trial(options, presets[index], window, renderer,
                                   60U, 120U, index % 2U != 0U,
                                   samples, stable, trial_error)) {
            if (trial_error.find("memory budget") != std::string::npos) break;
            error = trial_error;
            return false;
        }
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        auto metrics = parallel_mater::gallery::summarize_calibration(
            samples, duration, stable, false);
        const double measured_seconds = std::accumulate(
            samples.begin(), samples.end(), 0.0,
            [](double total, const auto &sample) {
                return total + sample.milliseconds / 1'000.0;
            });
        metrics.simulation_progress_ratio = measured_seconds > 0.0
            ? std::min(1.0, samples.size() * static_cast<double>(k_timestep) /
                              measured_seconds)
            : 0.0;
        std::cout << "Calibration " << presets[index].brick_count << " bricks, "
                  << presets[index].wall_planes << " walls: collision p95 "
                  << metrics.collision_p95_milliseconds << " ms, max "
                  << metrics.collision_maximum_milliseconds << " ms\n";
        if (index == 0U) {
            output = {.config = presets[index], .metrics = metrics,
                      .hardware = metal_hardware_identity(device)};
        }
        if (!parallel_mater::gallery::calibration_passes(metrics)) {
            if (index == 0U) {
                error = "minimum one-brick preset failed the 1080p frame budget";
                return false;
            }
            if (best.has_value()) break;
            continue;
        }
        best = index;
    }
    if (!best.has_value()) {
        error = "no brick preset met the 1080p frame budget";
        return false;
    }

    // Qualify the chosen preset under repeated impacts for three minutes.
    for (std::size_t index = *best + 1U; index-- > 0U;) {
        std::vector<parallel_mater::gallery::CalibrationSample> samples;
        bool stable = true;
        const auto begin = std::chrono::steady_clock::now();
        bool off_center = false;
        while (std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - begin).count() < 180.0) {
            if (!run_calibration_trial(options, presets[index], window, renderer,
                                       60U, 300U, off_center, samples, stable,
                                       error))
                return false;
            if (!stable) break;
            off_center = !off_center;
        }
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        auto metrics = parallel_mater::gallery::summarize_calibration(
            samples, duration, stable, false);
        const double measured_seconds = std::accumulate(
            samples.begin(), samples.end(), 0.0,
            [](double total, const auto &sample) {
                return total + sample.milliseconds / 1'000.0;
            });
        metrics.simulation_progress_ratio = measured_seconds > 0.0
            ? std::min(1.0, samples.size() * static_cast<double>(k_timestep) /
                              measured_seconds)
            : 0.0;
        if (parallel_mater::gallery::calibration_passes(metrics)) {
            options.bricks = presets[index];
            output = {.config = presets[index], .metrics = metrics,
                      .hardware = metal_hardware_identity(device)};
            std::cout << "Qualified " << presets[index].brick_count
                      << " bricks at 1080p: collision p95 "
                      << metrics.collision_p95_milliseconds << " ms, max "
                      << metrics.collision_maximum_milliseconds << " ms\n";
            return true;
        }
        if (index == 0U) break;
    }
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
    output << "{\n  \"verified\": false,\n  \"backend\": \"metal\",\n"
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
           << parallel_mater::gallery::metal_rigid_solver_version << "\",\n"
           << "  \"build_revision\": \"" << build_label() << "\",\n"
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
    if (!parallel_mater::gallery::calibration_passes(result.metrics)) {
        error = "current configuration has no passing calibration";
        return false;
    }
    @autoreleasepool {
        NSAlert *alert = [[NSAlert alloc] init];
        alert.messageText = @"Verify brick simulation";
        alert.informativeText =
            @"Confirm that the wall remained stable and impacts looked correct. Enter your name before saving this measured profile.";
        [alert addButtonWithTitle:@"Verify and Save"];
        [alert addButtonWithTitle:@"Cancel"];
        NSTextField *verifier = [[NSTextField alloc]
            initWithFrame:NSMakeRect(0, 0, 320, 24)];
        verifier.placeholderString = @"Verifier name";
        alert.accessoryView = verifier;
        if ([alert runModal] != NSAlertFirstButtonReturn) {
            error = "verification cancelled";
            return false;
        }
        const std::string verifier_name = verifier.stringValue.UTF8String;
        if (verifier_name.empty()) {
            error = "verifier name is required";
            return false;
        }

        std::filesystem::path destination = options.profiles_file;
        if (!options.profiles_file_overridden) {
            NSSavePanel *panel = [NSSavePanel savePanel];
            panel.title = @"Update a tracked catalog or export a verified profile";
            panel.prompt = @"Save Verified Profile";
            panel.nameFieldStringValue = @"verified-profile-export.json";
            if ([panel runModal] != NSModalResponseOK || panel.URL == nil) {
                error = "profile file selection cancelled";
                return false;
            }
            destination = panel.URL.fileSystemRepresentation;
        }

        parallel_mater::gallery::VerifiedBrickProfile profile;
        profile.hardware = result.hardware;
        profile.scene = result.config;
        profile.solver_version = parallel_mater::gallery::metal_rigid_solver_version;
        profile.build_revision = build_label();
        profile.metrics = result.metrics;
        NSISO8601DateFormatter *formatter = [[NSISO8601DateFormatter alloc] init];
        profile.verified_at = [formatter stringFromDate:[NSDate date]].UTF8String;
        profile.verifier = verifier_name;
        return parallel_mater::gallery::save_verified_profile(
            destination, profile, error);
    }
}

bool run_interactive(const Options &options, std::string &error) {
    if (glfwInit() != GLFW_TRUE) {
        error = "GLFW initialization failed";
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(
        960, 540,
        "ParallelMater Metal Gallery", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        error = "Metal gallery window creation failed";
        return false;
    }
    NSWindow *native_window = glfwGetCocoaWindow(window);
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [native_window makeKeyAndOrderFront:nil];
    [NSApp activate];
    glfwFocusWindow(window);
    glfwPollEvents();
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
    InteractiveInput input;
    input.camera.set_preset(camera_preset_for(options.context, options.bricks));
    glfwSetWindowUserPointer(window, &input);
    glfwSetMouseButtonCallback(window, &mouse_button);
    glfwSetCursorPosCallback(window, &cursor_position);
    glfwSetScrollCallback(window, &scroll);
    glfwSetCharCallback(window, &character_input);

    Options interactive_options = options;
    const auto device = MTLCreateSystemDefaultDevice();
    Renderer renderer;
    if (!renderer.create(window, device, error)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }
    std::optional<CalibrationResult> last_calibration;
    if (interactive_options.calibrate) {
        CalibrationResult result;
        if (!calibrate_bricks(interactive_options, window, renderer, device,
                              result, error)) {
            if (result.metrics.duration_seconds > 0.0)
                save_local_calibration(result);
            glfwDestroyWindow(window);
            glfwTerminate();
            return false;
        }
        last_calibration = result;
        save_local_calibration(result);
        glfwDestroyWindow(window);
        glfwTerminate();
        return true;
    }
    Runtime runtime;
    if (!build_runtime(interactive_options, options.context, runtime, error)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }

    RenderFrame frame;
    std::vector<RigidBodyState> states;
    MetalDebugState debug;
    bool timing_visible = false;
    float latest_gpu_milliseconds = 0.0F;
    if (!assemble_frame(runtime, frame, states, error, &debug)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }
    FrameToken physics_completion;
    bool physics_in_flight = false;
    bool picker_visible = false;
    GalleryContext picker_selection = runtime.context;
    bool tab_latch = false;
    bool reset_latch = false;
    bool action_latch = false;
    bool escape_latch = false;
    bool enter_latch = false;
    bool backspace_latch = false;
    bool count_latch = false;
    bool timing_latch = false;
    bool primary_debug_latch = false;
    bool normals_latch = false;
    bool rigid_forces_latch = false;
    bool fluid_forces_latch = false;
    bool bonds_latch = false;
    bool velocities_latch = false;
    bool up_latch = false;
    bool down_latch = false;
    std::uint32_t completed_frames = 0U;
    std::uint32_t reported_completed_frames = 0U;
    float dump_angle = k_dump_initial_angle;
    Vec3 peg_gravity = default_gravity(runtime);
    Vec3 cloth_gravity = peg_gravity;
    auto update_title = [&](std::optional<double> fps = std::nullopt,
                            std::optional<double> simulation_hz =
                                std::nullopt) {
        const GalleryEntry &entry = gallery_entry(runtime.context);
        std::string title =
            "ParallelMater Metal " + build_label() + " — " +
            std::string(entry.name) + " [" +
            std::to_string(gallery_context_index(runtime.context) + 1U) +
            "/" + std::to_string(gallery_entries.size()) + "]";
        if (fps.has_value())
            title += " — " +
                     std::to_string(static_cast<int>(*fps + 0.5)) + " FPS";
        if (simulation_hz.has_value())
            title += " / " + std::to_string(
                static_cast<int>(*simulation_hz + 0.5)) + " SIM Hz";
        glfwSetWindowTitle(window, title.c_str());
        if (!fps.has_value())
            std::cout << "Scene " << gallery_context_index(runtime.context)
                      << ": " << entry.name << " — " << entry.help << '\n';
    };
    update_title();

    using Clock = std::chrono::steady_clock;
    auto previous_time = Clock::now();
    auto fps_start = previous_time;
    std::uint32_t presented_frames = 0U;
    double physics_accumulator = 0.0;
    auto finish_physics = [&](bool refresh_frame) {
        if (!physics_in_flight) return true;
        if (!check(physics_completion.wait(), "finish gallery world step",
                   error)) {
            return false;
        }
        physics_in_flight = false;
        ++completed_frames;
        if (runtime.fixed_collector.active() &&
            !runtime.fixed_collector.collect(
                runtime.world, runtime.scene, runtime.instance, error)) {
            return false;
        }
        if (timing_visible) {
            parallel_mater::WorldStepTimings timings{};
            if (!check(runtime.world.collect_step_timings(timings),
                       "collect gallery timings", error)) {
                return false;
            }
            if (timings.available)
                latest_gpu_milliseconds = timings.total_gpu_milliseconds;
        }
        return !refresh_frame ||
               assemble_frame(runtime, frame, states, error, &debug);
    };
    auto adopt_runtime = [&](Runtime &&replacement) {
        runtime = std::move(replacement);
        input.camera.set_preset(camera_preset_for(
            runtime.context, interactive_options.bricks));
        completed_frames = 0U;
        reported_completed_frames = 0U;
        dump_angle = k_dump_initial_angle;
        peg_gravity = default_gravity(runtime);
        cloth_gravity = peg_gravity;
        physics_accumulator = 0.0;
        update_title();
        return assemble_frame(runtime, frame, states, error, &debug);
    };

    while (!glfwWindowShouldClose(window)) {
        if (options.frames_set && completed_frames >= options.frames) break;
        glfwPollEvents();
        const auto now = Clock::now();
        const double frame_delta = std::clamp(
            std::chrono::duration<double>(now - previous_time).count(),
            0.0, 0.1);
        previous_time = now;
        if (physics_in_flight && physics_completion.ready() &&
            !finish_physics(true)) {
            glfwDestroyWindow(window);
            glfwTerminate();
            return false;
        }

        const bool tab = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
        const bool reset = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        const bool action = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        const bool escape =
            glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
        const bool enter =
            glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_KP_ENTER) == GLFW_PRESS;
        const bool backspace =
            glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS;
        const bool edit_count =
            glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
        const bool timing = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;
        const bool primary_debug =
            glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;
        const bool normals = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;
        const bool rigid_forces =
            glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
        const bool fluid_forces =
            glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
        const bool bonds = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
        const bool velocities =
            glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS;
        const bool up = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
        const bool down = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;

        if (input.count_dialog_visible) {
            if (input.brick_dialog && fluid_forces && !fluid_forces_latch) {
                if (!finish_physics(false)) {
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    return false;
                }
                CalibrationResult result;
                std::string calibration_error;
                bool calibration_succeeded = false;
                if (calibrate_bricks(interactive_options, window, renderer,
                                     device, result, calibration_error)) {
                    calibration_succeeded = true;
                    last_calibration = result;
                    save_local_calibration(result);
                    input.brick_values = {
                        std::to_string(result.config.brick_count),
                        std::to_string(result.config.brick_scale),
                        std::to_string(result.config.wall_planes)};
                    input.count_dialog_visible = false;
                    input.count_value_invalid = false;
                } else {
                    if (result.metrics.duration_seconds > 0.0)
                        save_local_calibration(result);
                    input.count_value_invalid = true;
                    std::cerr << "Calibration failed: " << calibration_error << '\n';
                }
                if (calibration_succeeded) {
                    Runtime replacement;
                    if (!build_runtime(interactive_options,
                                       GalleryContext::rigid_body, replacement,
                                       error) ||
                        !adopt_runtime(std::move(replacement))) {
                        glfwDestroyWindow(window);
                        glfwTerminate();
                        return false;
                    }
                }
            }
            if (input.brick_dialog && primary_debug && !primary_debug_latch) {
                std::string save_error;
                if (!last_calibration.has_value() ||
                    last_calibration->config != interactive_options.bricks) {
                    save_error = "calibrate this exact configuration before verification";
                } else if (verify_and_save_profile(
                               interactive_options, *last_calibration,
                               save_error)) {
                    std::cout << "Verified device profile saved\n";
                    input.count_dialog_visible = false;
                }
                if (!save_error.empty())
                    std::cerr << "Profile not saved: " << save_error << '\n';
            }
            if (escape && !escape_latch) {
                input.count_dialog_visible = false;
                input.count_value_invalid = false;
            }
            if (backspace && !backspace_latch) {
                if (input.replace_count_value) {
                    if (input.brick_dialog)
                        input.brick_values[input.brick_field].clear();
                    else
                        input.count_value.clear();
                    input.replace_count_value = false;
                } else {
                    std::string &value = input.brick_dialog
                        ? input.brick_values[input.brick_field]
                        : input.count_value;
                    if (!value.empty()) value.pop_back();
                }
                input.count_value_invalid = false;
            }
            if (input.brick_dialog && tab && !tab_latch) {
                input.brick_field = (input.brick_field + 1U) % 3U;
                input.replace_count_value = true;
                input.count_value_invalid = false;
            }
            if (enter && !enter_latch) {
                std::uint32_t requested = 0U;
                const GalleryEntry &entry = gallery_entry(runtime.context);
                BrickSceneConfig requested_bricks = interactive_options.bricks;
                std::string validation_error;
                const bool bricks_valid = !input.brick_dialog ||
                    (parse_u32(input.brick_values[0], requested_bricks.brick_count) &&
                     parse_float(input.brick_values[1], requested_bricks.brick_scale) &&
                     parse_u32(input.brick_values[2], requested_bricks.wall_planes) &&
                     validate_brick_config(requested_bricks, validation_error));
                if (!bricks_valid || (!input.brick_dialog &&
                    (!parse_u32(input.count_value, requested) ||
                     requested < entry.minimum_count ||
                     requested > entry.maximum_count))) {
                    input.count_value_invalid = true;
                } else if (!finish_physics(false)) {
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    return false;
                } else {
                    Options requested_options = interactive_options;
                    if (input.brick_dialog) {
                        requested_options.bricks = requested_bricks;
                        requested_options.bricks_overridden = true;
                    } else if (entry.count_kind == GalleryCountKind::fluid_particles)
                        requested_options.fluid_particles = requested;
                    else
                        requested_options.dump_spheres = requested;
                    Runtime replacement;
                    std::string rebuild_error;
                    if (!build_runtime(requested_options, runtime.context,
                                       replacement, rebuild_error)) {
                        input.count_value_invalid = true;
                        std::cerr << "Scene restart failed: "
                                  << rebuild_error << '\n';
                    } else {
                        interactive_options = std::move(requested_options);
                        input.count_dialog_visible = false;
                        input.count_value_invalid = false;
                        if (!adopt_runtime(std::move(replacement))) {
                            glfwDestroyWindow(window);
                            glfwTerminate();
                            return false;
                        }
                    }
                }
            }
        } else if (escape && !escape_latch) {
            if (picker_visible)
                picker_visible = false;
            else
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (tab && !tab_latch) {
            picker_visible = !picker_visible;
            picker_selection = runtime.context;
            physics_accumulator = 0.0;
        }
        if (!input.count_dialog_visible && picker_visible) {
            std::size_t selected = gallery_context_index(picker_selection);
            if (up && !up_latch && selected != 0U) --selected;
            if (down && !down_latch)
                selected = std::min(gallery_entries.size() - 1U,
                                    selected + 1U);
            picker_selection = gallery_entries[selected].context;
            if (enter && !enter_latch) {
                const GalleryContext requested = picker_selection;
                if (requested != runtime.context) {
                    if (!finish_physics(false)) {
                        glfwDestroyWindow(window);
                        glfwTerminate();
                        return false;
                    }
                    Runtime replacement;
                    if (!build_runtime(interactive_options, requested, replacement,
                                       error)) {
                        glfwDestroyWindow(window);
                        glfwTerminate();
                        return false;
                    }
                    debug.reset();
                    if (!adopt_runtime(std::move(replacement))) {
                        glfwDestroyWindow(window);
                        glfwTerminate();
                        return false;
                    }
                }
                picker_visible = false;
                physics_accumulator = 0.0;
            }
        } else if (!input.count_dialog_visible &&
                   reset && !reset_latch) {
            if (!finish_physics(false)) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
            Runtime replacement;
            if (!build_runtime(interactive_options, runtime.context, replacement,
                               error)) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
            if (!adopt_runtime(std::move(replacement))) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
        } else if (!input.count_dialog_visible && !picker_visible &&
                   edit_count && !count_latch &&
                   gallery_entry(runtime.context).count_kind !=
                       GalleryCountKind::none) {
            input.count_dialog_visible = true;
            input.brick_dialog = gallery_entry(runtime.context).count_kind ==
                GalleryCountKind::brick_scene;
            input.brick_field = 0U;
            input.brick_values = {
                std::to_string(interactive_options.bricks.brick_count),
                std::to_string(interactive_options.bricks.brick_scale),
                std::to_string(interactive_options.bricks.wall_planes)};
            input.count_value = std::to_string(
                gallery_entry(runtime.context).count_kind ==
                        GalleryCountKind::fluid_particles
                    ? interactive_options.fluid_particles
                    : interactive_options.dump_spheres);
            input.replace_count_value = true;
            input.count_value_invalid = false;
            physics_accumulator = 0.0;
        }
        if (!picker_visible && !input.count_dialog_visible &&
            action && !action_latch &&
            toggles_constraint(
                gallery_entry(runtime.context).controls)) {
            if (!finish_physics(true) ||
                !toggle_constraints(runtime, error)) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
        }
        bool debug_changed = false;
        if (!input.count_dialog_visible) {
            if (timing && !timing_latch) timing_visible = !timing_visible;
            const bool smoke =
                parallel_mater::gallery::is_smoke_context(runtime.context);
            if (primary_debug && !primary_debug_latch) {
                if (smoke)
                    debug.toggle_smoke(
                        MetalSmokeDebugMode::density_temperature);
                else
                    debug.toggle_primary(runtime.context);
                debug_changed = true;
            }
            if (normals && !normals_latch) {
                if (smoke)
                    debug.toggle_smoke(MetalSmokeDebugMode::grid);
                else
                    debug.normals = !debug.normals;
                debug_changed = true;
            }
            if (rigid_forces && !rigid_forces_latch) {
                if (smoke)
                    debug.toggle_smoke(MetalSmokeDebugMode::velocity);
                else
                    debug.rigid_forces = !debug.rigid_forces;
                debug_changed = true;
            }
            if (fluid_forces && !fluid_forces_latch) {
                if (smoke)
                    debug.toggle_smoke(MetalSmokeDebugMode::pressure);
                else
                    debug.fluid_forces = !debug.fluid_forces;
                debug_changed = true;
            }
            if (bonds && !bonds_latch) {
                if (smoke)
                    debug.toggle_smoke(MetalSmokeDebugMode::vorticity);
                else
                    debug.cloth_bonds = !debug.cloth_bonds;
                debug_changed = true;
            }
            if (velocities && !velocities_latch) {
                if (smoke)
                    debug.toggle_smoke(MetalSmokeDebugMode::divergence);
                else
                    debug.velocities = !debug.velocities;
                debug_changed = true;
            }
        }
        runtime.collect_rigid_contacts = debug.rigid_contacts;
        runtime.collect_kernel_timings = timing_visible;
        if (debug_changed && !physics_in_flight &&
            !assemble_frame(runtime, frame, states, error, &debug)) {
            glfwDestroyWindow(window);
            glfwTerminate();
            return false;
        }
        input.ui_visible = picker_visible || input.count_dialog_visible;
        tab_latch = tab;
        reset_latch = reset;
        action_latch = action;
        escape_latch = escape;
        enter_latch = enter;
        backspace_latch = backspace;
        count_latch = edit_count;
        timing_latch = timing;
        primary_debug_latch = primary_debug;
        normals_latch = normals;
        rigid_forces_latch = rigid_forces;
        fluid_forces_latch = fluid_forces;
        bonds_latch = bonds;
        velocities_latch = velocities;
        up_latch = up;
        down_latch = down;

        const DirectionalInput directional =
            directional_input(window, !input.ui_visible);
        if (input.ui_visible) {
            physics_accumulator = 0.0;
        } else {
            physics_accumulator = std::min(
                physics_accumulator + frame_delta,
                4.0 * static_cast<double>(k_timestep));
        }
        if (!physics_in_flight && physics_accumulator >= k_timestep &&
            (!options.frames_set || completed_frames < options.frames)) {
            const GalleryEntry &entry = gallery_entry(runtime.context);
            if (entry.controls == GalleryControlPolicy::tank_motor &&
                !drive_motors(runtime, directional, error)) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
            if (runtime.kinematic_index < runtime.instance.rigid_bodies.size() &&
                (entry.controls == GalleryControlPolicy::dump_rotation ||
                 directional.x != 0.0F || directional.z != 0.0F)) {
                if (entry.controls == GalleryControlPolicy::dump_rotation) {
                    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
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
                if (!check(runtime.world.set_kinematic_target(
                               runtime.instance.rigid_bodies[
                                   runtime.kinematic_index],
                               runtime.kinematic_target),
                           "move kinematic body", error)) {
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    return false;
                }
            }
            Vec3 gravity = default_gravity(runtime);
            if (entry.controls == GalleryControlPolicy::cloth_gravity) {
                cloth_gravity = steer_gravity(
                    cloth_gravity, input.camera.camera(), directional.x,
                    -directional.z,
                    k_gravity * runtime.scene.gravity_scale,
                    k_cloth_gravity_tilt_degrees, k_timestep);
                gravity = cloth_gravity;
            } else if (entry.controls ==
                       GalleryControlPolicy::collector_gravity) {
                gravity = collector_gravity_for(
                    directional, runtime.scene.gravity_scale,
                    input.camera.camera());
            } else if (uses_rigid_gravity(entry.controls)) {
                gravity = gravity_for(directional,
                                      runtime.scene.gravity_scale,
                                      input.camera.camera());
            } else if (entry.controls == GalleryControlPolicy::peg_gravity) {
                const float right = directional.x +
                    static_cast<float>(
                        glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) -
                    static_cast<float>(
                        glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS);
                const float forward = -directional.z +
                    static_cast<float>(
                        glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) -
                    static_cast<float>(
                        glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS);
                peg_gravity = steer_gravity(
                    peg_gravity, input.camera.camera(), right, forward,
                    k_gravity * runtime.scene.gravity_scale,
                    peg_paint_gravity_tilt_degrees, k_timestep);
                gravity = peg_gravity;
            }
            if (!step_runtime_async(runtime, gravity, physics_completion,
                                    error)) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
            physics_in_flight = true;
            physics_accumulator -= k_timestep;
        }
        std::string status_text;
        if (timing_visible) {
            status_text = "F TIMING  GPU " +
                std::to_string(static_cast<int>(
                    latest_gpu_milliseconds + 0.5F)) + " MS";
        }
        const auto append_status = [&](std::string_view value) {
            if (!status_text.empty()) status_text += "  ";
            status_text += value;
        };
        switch (debug.smoke_mode) {
        case MetalSmokeDebugMode::grid: append_status("Z GRID"); break;
        case MetalSmokeDebugMode::velocity: append_status("X VELOCITY"); break;
        case MetalSmokeDebugMode::pressure: append_status("C PRESSURE"); break;
        case MetalSmokeDebugMode::density_temperature:
            append_status("V DENSITY TEMPERATURE");
            break;
        case MetalSmokeDebugMode::vorticity: append_status("B VORTICITY"); break;
        case MetalSmokeDebugMode::divergence: append_status("N DIVERGENCE"); break;
        case MetalSmokeDebugMode::none:
            if (debug.particle_view) append_status("V PARTICLES");
            if (debug.structure) append_status("V STRUCTURE");
            if (debug.rigid_contacts) append_status("V CONTACTS");
            if (debug.normals) append_status("Z NORMALS");
            if (debug.rigid_forces) append_status("X RIGID FORCES");
            if (debug.fluid_forces) append_status("C FLUID FORCES");
            if (debug.cloth_bonds) append_status("B BONDS");
            if (debug.velocities) append_status("N VELOCITIES");
            break;
        }
        if (!renderer.draw(
                window, frame, gallery_entry(runtime.context),
                input.camera.camera(),
                {.picker_selection = picker_visible
                        ? std::optional<GalleryContext>{picker_selection}
                        : std::nullopt,
                 .count_dialog_visible = input.count_dialog_visible,
                 .context = runtime.context,
                 .count_value = input.count_value,
                 .brick_values = {input.brick_values[0], input.brick_values[1],
                                  input.brick_values[2]},
                 .brick_field = input.brick_field,
                 .brick_dialog = input.brick_dialog,
                 .count_value_invalid = input.count_value_invalid,
                 .status = status_text},
                error)) {
            glfwDestroyWindow(window);
            glfwTerminate();
            return false;
        }
        ++presented_frames;
        const double fps_interval =
            std::chrono::duration<double>(now - fps_start).count();
        if (fps_interval >= 0.5) {
            update_title(
                static_cast<double>(presented_frames) / fps_interval,
                static_cast<double>(completed_frames -
                                    reported_completed_frames) /
                    fps_interval);
            reported_completed_frames = completed_frames;
            fps_start = now;
            presented_frames = 0U;
        }
    }
    if (!finish_physics(false)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "Invalid arguments. Use --help.\n";
        return 2;
    }
    if (options.calibrate && (options.headless || options.all_scenes)) {
        std::cerr << "--calibrate requires the interactive gallery path\n";
        return 2;
    }
    if (options.version) {
        print_version();
        return 0;
    }
    if (options.help || options.list_scenes) {
        print_help();
        return 0;
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::cerr << "No Metal device is available\n";
            return 1;
        }
        std::cout << "Metal device: " << device.name.UTF8String << '\n';
        if (!options.headless && !options.all_scenes &&
            !options.bricks_overridden) {
            parallel_mater::gallery::DeviceProfileCatalog catalog;
            std::string profile_error;
            if (parallel_mater::gallery::load_device_profiles(
                    device_profiles_path(options), catalog, profile_error)) {
                const auto hardware = metal_hardware_identity(device);
                if (const auto *profile =
                        parallel_mater::gallery::find_matching_profile(
                            catalog, hardware, brick_render_width,
                            brick_render_height, parallel_mater::gallery::metal_rigid_solver_version)) {
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
    }
    std::string error;
    const bool success = options.headless || options.all_scenes
        ? run_headless(options, error)
        : run_interactive(options, error);
    if (!success) {
        std::cerr << "Metal gallery failed: " << error << '\n';
        return 1;
    }
    return 0;
}
