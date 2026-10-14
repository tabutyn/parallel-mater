// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/renderer.hpp>

#include <render/render_scene.hpp>
#include <scene_builder.hpp>

#include "vector_math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace parallel_mater::gallery {
namespace {

using math::cross;
using math::dot;
using math::normalize_or;

constexpr float k_pi = 3.14159265358979323846F;
constexpr float k_near_plane = 0.025F;

struct ProjectedVertex {
    float x{};
    float y{};
    float depth{};
    bool visible{};
};

struct View {
    Vec3 eye{};
    Vec3 right{};
    Vec3 up{};
    Vec3 forward{};
    float focal{};
    float aspect{};
};

[[nodiscard]] Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 multiply(Vec3 value, float scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] Quaternion normalized(Quaternion value) noexcept {
    const float magnitude = std::sqrt(value.x * value.x + value.y * value.y +
                                      value.z * value.z + value.w * value.w);
    if (magnitude <= 1.0e-12F) return {0.0F, 0.0F, 0.0F, 1.0F};
    return {value.x / magnitude, value.y / magnitude,
            value.z / magnitude, value.w / magnitude};
}

[[nodiscard]] Quaternion interpolate(Quaternion before, Quaternion after,
                                     float alpha) noexcept {
    if (before.x * after.x + before.y * after.y + before.z * after.z +
            before.w * after.w < 0.0F)
        after = {-after.x, -after.y, -after.z, -after.w};
    return normalized({before.x + (after.x - before.x) * alpha,
                       before.y + (after.y - before.y) * alpha,
                       before.z + (after.z - before.z) * alpha,
                       before.w + (after.w - before.w) * alpha});
}

[[nodiscard]] Vec3 rotate(Quaternion orientation, Vec3 value) noexcept {
    const Vec3 vector{orientation.x, orientation.y, orientation.z};
    const Vec3 twice = multiply(cross(vector, value), 2.0F);
    return add(value, add(multiply(twice, orientation.w), cross(vector, twice)));
}

[[nodiscard]] RigidBodyState interpolate(const RigidBodyState &before,
                                         const RigidBodyState &after,
                                         float alpha) noexcept {
    return {
        add(before.position,
            multiply(subtract(after.position, before.position), alpha)),
        interpolate(before.orientation, after.orientation, alpha),
        {}, {}};
}

[[nodiscard]] std::uint32_t color(Vec3 value, float brightness = 1.0F,
                                  float alpha = 1.0F) noexcept {
    const auto channel = [brightness](float item) {
        return static_cast<std::uint32_t>(std::clamp(
            item * brightness * 255.0F, 0.0F, 255.0F));
    };
    return channel(value.x) | channel(value.y) << 8U |
        channel(value.z) << 16U |
        static_cast<std::uint32_t>(std::clamp(alpha * 255.0F, 0.0F, 255.0F))
            << 24U;
}

[[nodiscard]] float edge(float ax, float ay, float bx, float by,
                         float px, float py) noexcept {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

class SoftwareRenderer final : public Renderer {
  public:
    SoftwareRenderer(const SceneDefinition &scene, std::uint32_t width,
                     std::uint32_t height)
        : scene_(scene), width_(width), height_(height) {}

    [[nodiscard]] bool render(
        const RenderSceneFrame &frame, Camera camera,
        std::vector<std::uint32_t> &rgba, std::string &error,
        RendererTimings *timings, FluidRenderMode,
        bool show_smoke_particles, float rigid_interpolation_alpha) override {
        using clock = std::chrono::steady_clock;
        const auto started = clock::now();
        error.clear();
        try {
            rgba.resize(static_cast<std::size_t>(width_) * height_);
            depth_.assign(rgba.size(), std::numeric_limits<float>::infinity());
            for (std::uint32_t y = 0U; y < height_; ++y) {
                const float blend = static_cast<float>(y) /
                    static_cast<float>(std::max(height_ - 1U, 1U));
                const Vec3 background{
                    0.025F + 0.025F * blend,
                    0.055F + 0.055F * blend,
                    0.075F + 0.075F * blend};
                std::fill_n(rgba.begin() + static_cast<std::size_t>(y) * width_,
                            width_, color(background));
            }

            const View view = make_view(camera);
            draw_rigid_bodies(frame, view, rigid_interpolation_alpha, rgba);
            draw_deformables(frame, view, rgba);
            for (const auto &nodes : frame.rope_nodes)
                draw_polyline(nodes, 0.035F, {0.72F, 0.52F, 0.22F}, view, rgba);
            for (const RenderFluidParticle &particle : frame.fluid_particles)
                draw_point(particle.position,
                           std::max(frame.fluid_particle_radius, 0.025F),
                           {0.08F, 0.42F, 0.92F}, view, rgba);
            if (show_smoke_particles) {
                for (const RenderSmokeParticle &particle : frame.smoke_particles)
                    draw_point(particle.position,
                               std::max(frame.smoke_particle_radius, 0.035F),
                               {0.62F, 0.66F, 0.70F}, view, rgba);
            }

            painted_texels_ = 0U;
            for (const auto &field : frame.paint_fields)
                painted_texels_ += static_cast<std::uint64_t>(std::count_if(
                    field.begin(), field.end(),
                    [](std::uint32_t value) { return value != 0U; }));
            if (timings != nullptr) {
                *timings = {};
                timings->particle_count =
                    static_cast<std::uint32_t>(frame.fluid_particles.size());
                timings->smoke_particle_count =
                    static_cast<std::uint32_t>(frame.smoke_particles.size());
                timings->particle_view = true;
                timings->total_wall_milliseconds =
                    std::chrono::duration<float, std::milli>(clock::now() - started)
                        .count();
                timings->raytrace_wall_milliseconds =
                    timings->total_wall_milliseconds;
            }
        } catch (const std::bad_alloc &) {
            error = "software renderer: out of host memory";
            return false;
        }
        return true;
    }

    [[nodiscard]] bool advance_visuals(const RenderSceneFrame &,
                                       std::string &error) override {
        error.clear();
        return true;
    }

    [[nodiscard]] bool paint_coverage(std::uint64_t &painted_texels,
                                      std::string &error) const override {
        error.clear();
        painted_texels = painted_texels_;
        return true;
    }

    [[nodiscard]] std::uint32_t width() const noexcept override { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept override { return height_; }

  private:
    [[nodiscard]] View make_view(Camera camera) const noexcept {
        View result;
        result.eye = camera.eye;
        result.forward = normalize_or(subtract(camera.target, camera.eye),
                                      {0.0F, 0.0F, -1.0F});
        result.right = normalize_or(cross(result.forward, camera.up),
                                    {1.0F, 0.0F, 0.0F});
        result.up = normalize_or(cross(result.right, result.forward),
                                 {0.0F, 1.0F, 0.0F});
        result.focal = 1.0F / std::tan(
            camera.vertical_field_of_view_degrees * k_pi / 360.0F);
        result.aspect = static_cast<float>(width_) /
            static_cast<float>(std::max(height_, 1U));
        return result;
    }

    [[nodiscard]] ProjectedVertex project(Vec3 point,
                                          const View &view) const noexcept {
        const Vec3 relative = subtract(point, view.eye);
        const float z = dot(relative, view.forward);
        if (z <= k_near_plane) return {};
        const float x = dot(relative, view.right) * view.focal /
            (z * view.aspect);
        const float y = dot(relative, view.up) * view.focal / z;
        return {(x * 0.5F + 0.5F) * static_cast<float>(width_),
                (y * 0.5F + 0.5F) * static_cast<float>(height_), z, true};
    }

    void draw_rigid_bodies(const RenderSceneFrame &frame, const View &view,
                           float alpha,
                           std::vector<std::uint32_t> &rgba) {
        const bool have_states =
            frame.rigid_states.size() == scene_.rigid_bodies.size();
        for (std::size_t body_index = 0U;
             body_index < scene_.rigid_bodies.size(); ++body_index) {
            const RigidBodyDefinition &body = scene_.rigid_bodies[body_index];
            RigidBodyState state = body.options.initial_state;
            if (have_states) {
                state = frame.rigid_states[body_index];
                if (frame.previous_rigid_states.size() ==
                    frame.rigid_states.size())
                    state = interpolate(frame.previous_rigid_states[body_index],
                                        state, std::clamp(alpha, 0.0F, 1.0F));
            }
            for (std::uint32_t mesh_index : body.mesh_indices) {
                if (mesh_index < scene_.meshes.size())
                    draw_mesh(scene_.meshes[mesh_index], nullptr, &state,
                              view, rgba);
            }
        }
    }

    void draw_deformables(const RenderSceneFrame &frame, const View &view,
                          std::vector<std::uint32_t> &rgba) {
        for (std::size_t index = 0U; index < scene_.cloths.size(); ++index) {
            const std::uint32_t mesh_index = scene_.cloths[index].mesh_index;
            if (mesh_index >= scene_.meshes.size()) continue;
            const std::vector<Vec3> *positions = nullptr;
            if (index < frame.cloth_surface_vertices.size() &&
                !frame.cloth_surface_vertices[index].empty())
                positions = &frame.cloth_surface_vertices[index];
            else if (index < frame.cloth_vertices.size())
                positions = &frame.cloth_vertices[index];
            draw_mesh(scene_.meshes[mesh_index], positions, nullptr, view, rgba);
        }
        for (std::size_t index = 0U; index < scene_.soft_bodies.size(); ++index) {
            const std::uint32_t mesh_index = scene_.soft_bodies[index].mesh_index;
            if (mesh_index >= scene_.meshes.size()) continue;
            const std::vector<Vec3> *positions = nullptr;
            if (index < frame.soft_body_surface_vertices.size() &&
                !frame.soft_body_surface_vertices[index].empty())
                positions = &frame.soft_body_surface_vertices[index];
            else if (index < frame.soft_body_nodes.size())
                positions = &frame.soft_body_nodes[index];
            draw_mesh(scene_.meshes[mesh_index], positions, nullptr, view, rgba);
        }
    }

    void draw_mesh(const TriangleMesh &mesh, const std::vector<Vec3> *positions,
                   const RigidBodyState *state, const View &view,
                   std::vector<std::uint32_t> &rgba) {
        if (!mesh.visible) return;
        const auto position = [&](std::uint32_t index) {
            Vec3 point = positions != nullptr && index < positions->size()
                ? (*positions)[index] : mesh.vertices[index].position;
            return state != nullptr
                ? add(rotate(state->orientation, point), state->position)
                : point;
        };
        for (std::size_t triangle = 0U; triangle + 2U < mesh.indices.size();
             triangle += 3U) {
            const std::uint32_t ia = mesh.indices[triangle];
            const std::uint32_t ib = mesh.indices[triangle + 1U];
            const std::uint32_t ic = mesh.indices[triangle + 2U];
            if (ia >= mesh.vertices.size() || ib >= mesh.vertices.size() ||
                ic >= mesh.vertices.size())
                continue;
            const Vec3 a = position(ia);
            const Vec3 b = position(ib);
            const Vec3 c = position(ic);
            const Vec3 normal = normalize_or(
                cross(subtract(b, a), subtract(c, a)), {0.0F, 1.0F, 0.0F});
            const float light = 0.28F + 0.72F * std::max(
                0.0F, dot(normal, normalize_or(Vec3{-0.35F, 0.8F, 0.45F},
                                                  {0.0F, 1.0F, 0.0F})));
            draw_triangle(project(a, view), project(b, view), project(c, view),
                          color(mesh.base_color, light), rgba);
        }
    }

    void draw_triangle(ProjectedVertex a, ProjectedVertex b, ProjectedVertex c,
                       std::uint32_t packed,
                       std::vector<std::uint32_t> &rgba) {
        if (!a.visible || !b.visible || !c.visible) return;
        const float area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
        if (std::abs(area) < 1.0e-7F) return;
        const int left = std::max(0, static_cast<int>(std::floor(
            std::min({a.x, b.x, c.x}))));
        const int right = std::min(static_cast<int>(width_) - 1,
            static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
        const int bottom = std::max(0, static_cast<int>(std::floor(
            std::min({a.y, b.y, c.y}))));
        const int top = std::min(static_cast<int>(height_) - 1,
            static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));
        if (left > right || bottom > top) return;
        const float inverse_area = 1.0F / area;
        for (int y = bottom; y <= top; ++y) {
            for (int x = left; x <= right; ++x) {
                const float px = static_cast<float>(x) + 0.5F;
                const float py = static_cast<float>(y) + 0.5F;
                const float wa = edge(b.x, b.y, c.x, c.y, px, py) * inverse_area;
                const float wb = edge(c.x, c.y, a.x, a.y, px, py) * inverse_area;
                const float wc = 1.0F - wa - wb;
                if (wa < 0.0F || wb < 0.0F || wc < 0.0F) continue;
                const float depth = wa * a.depth + wb * b.depth + wc * c.depth;
                const std::size_t pixel = static_cast<std::size_t>(y) * width_ +
                    static_cast<std::uint32_t>(x);
                if (depth >= depth_[pixel]) continue;
                depth_[pixel] = depth;
                rgba[pixel] = packed;
            }
        }
    }

    void draw_point(Vec3 position, float radius, Vec3 tint, const View &view,
                    std::vector<std::uint32_t> &rgba) {
        const ProjectedVertex center = project(position, view);
        if (!center.visible) return;
        const int screen_radius = std::clamp(static_cast<int>(std::ceil(
            radius * view.focal * static_cast<float>(height_) /
            (2.0F * center.depth))), 1, 32);
        const int center_x = static_cast<int>(std::round(center.x));
        const int center_y = static_cast<int>(std::round(center.y));
        const int radius_squared = screen_radius * screen_radius;
        for (int y = center_y - screen_radius; y <= center_y + screen_radius; ++y) {
            if (y < 0 || y >= static_cast<int>(height_)) continue;
            for (int x = center_x - screen_radius; x <= center_x + screen_radius; ++x) {
                if (x < 0 || x >= static_cast<int>(width_)) continue;
                const int dx = x - center_x;
                const int dy = y - center_y;
                if (dx * dx + dy * dy > radius_squared) continue;
                const float sphere = std::sqrt(std::max(
                    0.0F, 1.0F - static_cast<float>(dx * dx + dy * dy) /
                        static_cast<float>(std::max(radius_squared, 1))));
                const float depth = center.depth - radius * sphere;
                const std::size_t pixel = static_cast<std::size_t>(y) * width_ +
                    static_cast<std::uint32_t>(x);
                if (depth >= depth_[pixel]) continue;
                depth_[pixel] = depth;
                rgba[pixel] = color(tint, 0.55F + 0.45F * sphere);
            }
        }
    }

    void draw_polyline(const std::vector<Vec3> &points, float radius, Vec3 tint,
                       const View &view, std::vector<std::uint32_t> &rgba) {
        if (points.empty()) return;
        draw_point(points.front(), radius, tint, view, rgba);
        for (std::size_t index = 1U; index < points.size(); ++index) {
            const Vec3 delta = subtract(points[index], points[index - 1U]);
            const float length = std::sqrt(dot(delta, delta));
            const int samples = std::max(1, static_cast<int>(
                std::ceil(length / std::max(radius, 0.01F))));
            for (int sample = 1; sample <= samples; ++sample)
                draw_point(add(points[index - 1U],
                               multiply(delta, static_cast<float>(sample) /
                                                   static_cast<float>(samples))),
                           radius, tint, view, rgba);
        }
    }

    SceneDefinition scene_;
    std::uint32_t width_{};
    std::uint32_t height_{};
    std::vector<float> depth_{};
    std::uint64_t painted_texels_{};
};

} // namespace

bool create_renderer(const SceneDefinition &scene,
                     const CommandSceneInstance &,
                     std::uint32_t width, std::uint32_t height,
                     RendererHandle &output, std::string &error) {
    error.clear();
    if (width == 0U || height == 0U) {
        error = "software renderer dimensions must be nonzero";
        return false;
    }
    try {
        output.implementation_ =
            std::make_unique<SoftwareRenderer>(scene, width, height);
    } catch (const std::bad_alloc &) {
        error = "create software renderer: out of host memory";
        return false;
    }
    return true;
}

} // namespace parallel_mater::gallery
