// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>

#include "render/render_scene.hpp"
#include "vector_math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace parallel_mater::gallery {
namespace {

using math::add;
using math::cross;
using math::dot;
using math::multiply;
using math::normalize_or;
using math::subtract;

struct Projected {
    float x{};
    float y{};
    float depth{};
    bool visible{};
};

struct CameraBasis {
    Vec3 eye{};
    Vec3 forward{};
    Vec3 right{};
    Vec3 up{};
    float horizontal{};
    float vertical{};
};

CameraBasis camera_basis(Camera camera, std::uint32_t width,
                         std::uint32_t height) noexcept {
    constexpr float radians = 3.14159265358979323846F / 180.0F;
    CameraBasis result{};
    result.eye = camera.eye;
    result.forward = normalize_or(subtract(camera.target, camera.eye),
                                  {0.0F, 0.0F, -1.0F});
    result.right = normalize_or(cross(result.forward, camera.up),
                                {1.0F, 0.0F, 0.0F});
    result.up = normalize_or(cross(result.right, result.forward),
                             {0.0F, 1.0F, 0.0F});
    result.vertical = std::tan(
        camera.vertical_field_of_view_degrees * radians * 0.5F);
    result.horizontal = result.vertical * static_cast<float>(width) /
        static_cast<float>(height);
    return result;
}

Projected project(Vec3 point, const CameraBasis &camera,
                  std::uint32_t width, std::uint32_t height) noexcept {
    const Vec3 relative = subtract(point, camera.eye);
    const float depth = dot(relative, camera.forward);
    if (depth <= 0.01F) return {};
    const float ndc_x = dot(relative, camera.right) /
        (depth * camera.horizontal);
    const float ndc_y = dot(relative, camera.up) /
        (depth * camera.vertical);
    return {(ndc_x * 0.5F + 0.5F) * static_cast<float>(width),
            (ndc_y * 0.5F + 0.5F) * static_cast<float>(height),
            depth,
            std::fabs(ndc_x) <= 2.0F && std::fabs(ndc_y) <= 2.0F};
}

Vec3 rotate(Quaternion quaternion, Vec3 vector) noexcept {
    const Vec3 q{quaternion.x, quaternion.y, quaternion.z};
    const Vec3 twice = multiply(cross(q, vector), 2.0F);
    return add(vector, add(multiply(twice, quaternion.w), cross(q, twice)));
}

Quaternion normalized(Quaternion value) noexcept {
    const float size = std::sqrt(value.x * value.x + value.y * value.y +
                                 value.z * value.z + value.w * value.w);
    if (size <= 1.0e-8F) return {};
    return {value.x / size, value.y / size, value.z / size, value.w / size};
}

RigidBodyState interpolate(RigidBodyState previous, RigidBodyState current,
                           float alpha) noexcept {
    alpha = std::clamp(alpha, 0.0F, 1.0F);
    const auto blend = [alpha](float first, float second) {
        return first + (second - first) * alpha;
    };
    RigidBodyState result = current;
    result.position = {blend(previous.position.x, current.position.x),
                       blend(previous.position.y, current.position.y),
                       blend(previous.position.z, current.position.z)};
    result.orientation = normalized({
        blend(previous.orientation.x, current.orientation.x),
        blend(previous.orientation.y, current.orientation.y),
        blend(previous.orientation.z, current.orientation.z),
        blend(previous.orientation.w, current.orientation.w)});
    return result;
}

std::uint32_t packed(Vec3 color) noexcept {
    const auto channel = [](float value) {
        return static_cast<std::uint32_t>(
            std::clamp(value, 0.0F, 1.0F) * 255.0F + 0.5F);
    };
    return channel(color.x) | (channel(color.y) << 8U) |
        (channel(color.z) << 16U) | 0xff000000U;
}

float edge(Projected first, Projected second, float x, float y) noexcept {
    return (x - first.x) * (second.y - first.y) -
        (y - first.y) * (second.x - first.x);
}

void draw_triangle(Projected a, Projected b, Projected c, Vec3 color,
                   std::uint32_t width, std::uint32_t height,
                   std::vector<float> &depth,
                   std::vector<std::uint32_t> &rgba) noexcept {
    if (!a.visible && !b.visible && !c.visible) return;
    const float area = edge(a, b, c.x, c.y);
    if (std::fabs(area) <= 1.0e-5F) return;
    const int low_x = std::max(0, static_cast<int>(std::floor(
        std::min({a.x, b.x, c.x}))));
    const int high_x = std::min(static_cast<int>(width) - 1,
        static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
    const int low_y = std::max(0, static_cast<int>(std::floor(
        std::min({a.y, b.y, c.y}))));
    const int high_y = std::min(static_cast<int>(height) - 1,
        static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));
    if (low_x > high_x || low_y > high_y) return;
    const float inverse_area = 1.0F / area;
    const std::uint32_t fill = packed(color);
    for (int y = low_y; y <= high_y; ++y) {
        for (int x = low_x; x <= high_x; ++x) {
            const float sample_x = static_cast<float>(x) + 0.5F;
            const float sample_y = static_cast<float>(y) + 0.5F;
            const float wa = edge(b, c, sample_x, sample_y) * inverse_area;
            const float wb = edge(c, a, sample_x, sample_y) * inverse_area;
            const float wc = 1.0F - wa - wb;
            if (wa < 0.0F || wb < 0.0F || wc < 0.0F) continue;
            const float candidate = wa * a.depth + wb * b.depth + wc * c.depth;
            const std::size_t index = static_cast<std::size_t>(y) * width +
                static_cast<std::uint32_t>(x);
            if (candidate >= depth[index]) continue;
            depth[index] = candidate;
            rgba[index] = fill;
        }
    }
}

void draw_mesh(const TriangleMesh &mesh, const std::vector<Vec3> *positions,
               const RigidBodyState *transform, const CameraBasis &camera,
               std::uint32_t width, std::uint32_t height,
               std::vector<float> &depth,
               std::vector<std::uint32_t> &rgba) noexcept {
    if (!mesh.visible || mesh.indices.size() < 3U) return;
    const auto point = [&](std::uint32_t index) {
        Vec3 value = positions != nullptr && index < positions->size()
            ? (*positions)[index] : mesh.vertices[index].position;
        if (transform != nullptr)
            value = add(rotate(transform->orientation, value),
                        transform->position);
        return value;
    };
    constexpr Vec3 light = {0.349F, 0.814F, 0.465F};
    for (std::size_t triangle = 0U; triangle + 2U < mesh.indices.size();
         triangle += 3U) {
        const std::uint32_t ia = mesh.indices[triangle];
        const std::uint32_t ib = mesh.indices[triangle + 1U];
        const std::uint32_t ic = mesh.indices[triangle + 2U];
        if (ia >= mesh.vertices.size() || ib >= mesh.vertices.size() ||
            ic >= mesh.vertices.size())
            continue;
        const Vec3 a = point(ia);
        const Vec3 b = point(ib);
        const Vec3 c = point(ic);
        const Vec3 normal = normalize_or(
            cross(subtract(b, a), subtract(c, a)), {0.0F, 1.0F, 0.0F});
        const float shade = 0.28F + 0.72F * std::max(0.0F, dot(normal, light));
        draw_triangle(project(a, camera, width, height),
                      project(b, camera, width, height),
                      project(c, camera, width, height),
                      multiply(mesh.base_color, shade), width, height,
                      depth, rgba);
    }
}

void draw_particle(Vec3 position, float radius, Vec3 color,
                   const CameraBasis &camera, std::uint32_t width,
                   std::uint32_t height, std::vector<float> &depth,
                   std::vector<std::uint32_t> &rgba) noexcept {
    const Projected center = project(position, camera, width, height);
    if (!center.visible) return;
    const int pixels = std::clamp(static_cast<int>(
        radius * static_cast<float>(height) /
        (center.depth * camera.vertical)), 1, 12);
    const std::uint32_t fill = packed(color);
    const int center_x = static_cast<int>(center.x);
    const int center_y = static_cast<int>(center.y);
    for (int y = -pixels; y <= pixels; ++y) {
        for (int x = -pixels; x <= pixels; ++x) {
            if (x * x + y * y > pixels * pixels) continue;
            const int screen_x = center_x + x;
            const int screen_y = center_y + y;
            if (screen_x < 0 || screen_y < 0 ||
                screen_x >= static_cast<int>(width) ||
                screen_y >= static_cast<int>(height))
                continue;
            const std::size_t index = static_cast<std::size_t>(screen_y) *
                width + static_cast<std::uint32_t>(screen_x);
            if (center.depth >= depth[index]) continue;
            depth[index] = center.depth;
            rgba[index] = fill;
        }
    }
}

class SoftwareRenderer final : public Renderer {
  public:
    SoftwareRenderer(SceneDefinition scene, std::uint32_t width,
                     std::uint32_t height)
        : scene_(std::move(scene)), width_(width), height_(height),
          depth_(static_cast<std::size_t>(width) * height) {}

    bool render(const RenderSceneFrame &frame, Camera camera,
                std::vector<std::uint32_t> &rgba, std::string &error,
                RendererTimings *timings, FluidRenderMode,
                bool show_smoke_particles,
                float rigid_interpolation_alpha) override {
        error.clear();
        const auto begin = std::chrono::steady_clock::now();
        try {
            rgba.resize(static_cast<std::size_t>(width_) * height_);
            depth_.assign(depth_.size(), std::numeric_limits<float>::infinity());
        } catch (const std::bad_alloc &) {
            error = "software renderer: out of host memory";
            return false;
        }
        for (std::uint32_t y = 0U; y < height_; ++y) {
            const float t = static_cast<float>(y) /
                static_cast<float>(std::max(height_ - 1U, 1U));
            const Vec3 color{0.035F + 0.04F * t,
                             0.055F + 0.055F * t,
                             0.075F + 0.075F * t};
            std::fill_n(rgba.begin() + static_cast<std::size_t>(y) * width_,
                        width_, packed(color));
        }
        const CameraBasis basis = camera_basis(camera, width_, height_);
        for (std::size_t body_index = 0U;
             body_index < scene_.rigid_bodies.size(); ++body_index) {
            const auto &body = scene_.rigid_bodies[body_index];
            RigidBodyState state = body.options.initial_state;
            if (body_index < frame.rigid_states.size()) {
                state = frame.rigid_states[body_index];
                if (body_index < frame.previous_rigid_states.size())
                    state = interpolate(frame.previous_rigid_states[body_index],
                                        state, rigid_interpolation_alpha);
            }
            for (std::uint32_t mesh_index : body.mesh_indices)
                if (mesh_index < scene_.meshes.size())
                    draw_mesh(scene_.meshes[mesh_index], nullptr, &state, basis,
                              width_, height_, depth_, rgba);
        }
        for (std::size_t index = 0U; index < scene_.cloths.size(); ++index) {
            const auto mesh_index = scene_.cloths[index].mesh_index;
            if (mesh_index >= scene_.meshes.size()) continue;
            const std::vector<Vec3> *positions =
                index < frame.cloth_surface_vertices.size()
                ? &frame.cloth_surface_vertices[index] : nullptr;
            draw_mesh(scene_.meshes[mesh_index], positions, nullptr, basis,
                      width_, height_, depth_, rgba);
        }
        for (std::size_t index = 0U; index < scene_.soft_bodies.size(); ++index) {
            const auto mesh_index = scene_.soft_bodies[index].mesh_index;
            if (mesh_index >= scene_.meshes.size()) continue;
            const std::vector<Vec3> *positions =
                index < frame.soft_body_surface_vertices.size()
                ? &frame.soft_body_surface_vertices[index] : nullptr;
            draw_mesh(scene_.meshes[mesh_index], positions, nullptr, basis,
                      width_, height_, depth_, rgba);
        }
        for (const auto &particle : frame.fluid_particles)
            draw_particle(particle.position,
                          std::max(frame.fluid_particle_radius, 0.02F),
                          {0.05F, 0.42F + 0.45F * particle.foam, 0.92F},
                          basis, width_, height_, depth_, rgba);
        if (show_smoke_particles) {
            for (const auto &particle : frame.smoke_particles)
                draw_particle(particle.position,
                              std::max(frame.smoke_particle_radius, 0.03F),
                              {0.58F, 0.61F, 0.65F}, basis, width_, height_,
                              depth_, rgba);
        }
        for (const auto &rope : frame.rope_nodes)
            for (Vec3 node : rope)
                draw_particle(node, 0.025F, {0.85F, 0.62F, 0.18F}, basis,
                              width_, height_, depth_, rgba);
        painted_texels_ = 0U;
        for (const auto &field : frame.paint_fields)
            painted_texels_ += static_cast<std::uint64_t>(std::count_if(
                field.begin(), field.end(), [](std::uint32_t value) {
                    return value != 0U;
                }));
        if (timings != nullptr) {
            *timings = {};
            timings->particle_count =
                static_cast<std::uint32_t>(frame.fluid_particles.size());
            timings->smoke_particle_count =
                static_cast<std::uint32_t>(frame.smoke_particles.size());
            timings->total_wall_milliseconds =
                std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
        }
        return true;
    }

    bool advance_visuals(const RenderSceneFrame &, std::string &error) override {
        error.clear();
        return true;
    }

    bool paint_coverage(std::uint64_t &painted_texels,
                        std::string &error) const override {
        error.clear();
        painted_texels = painted_texels_;
        return true;
    }

    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }

  private:
    SceneDefinition scene_;
    std::uint32_t width_{};
    std::uint32_t height_{};
    std::vector<float> depth_;
    std::uint64_t painted_texels_{};
};

} // namespace

bool create_renderer(const SceneDefinition &scene,
                     const CommandSceneInstance &, std::uint32_t width,
                     std::uint32_t height, RendererHandle &output,
                     std::string &error) {
    error.clear();
    if (scene.meshes.empty() || width == 0U || height == 0U) {
        error = "renderer requires scene geometry and nonzero image dimensions";
        return false;
    }
    try {
        output.implementation_ = std::make_unique<SoftwareRenderer>(
            scene, width, height);
        return true;
    } catch (const std::bad_alloc &) {
        error = "software renderer creation: out of host memory";
        return false;
    }
}

} // namespace parallel_mater::gallery
