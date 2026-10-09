// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/bitmap_font.hpp>

#include "vector_math.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

namespace parallel_mater::gallery {
namespace {

using math::add;
using math::cross;
using math::dot;
using math::length;
using math::multiply;
using math::subtract;

struct Color {
    std::uint8_t red{};
    std::uint8_t green{};
    std::uint8_t blue{};
    std::uint8_t alpha{255U};
};

[[nodiscard]] constexpr Color color(GalleryColor value) noexcept {
    return {value.red, value.green, value.blue, value.alpha};
}

[[nodiscard]] std::uint32_t packed(Color color) {
    return static_cast<std::uint32_t>(color.red) |
           static_cast<std::uint32_t>(color.green) << 8U |
           static_cast<std::uint32_t>(color.blue) << 16U | 0xff000000U;
}

void pixel(std::vector<std::uint32_t> &rgba, std::uint32_t width,
           std::uint32_t height, int x, int y, Color color) {
    if (x < 0 || y < 0 || x >= static_cast<int>(width) ||
        y >= static_cast<int>(height)) {
        return;
    }
    const std::size_t index =
        static_cast<std::size_t>(height - 1U - static_cast<std::uint32_t>(y)) *
            width +
        static_cast<std::uint32_t>(x);
    if (color.alpha == 255U) {
        rgba[index] = packed(color);
        return;
    }
    const std::uint32_t old = rgba[index];
    const unsigned inverse = 255U - color.alpha;
    const auto blend = [&](unsigned source, unsigned destination) {
        return (source * color.alpha + destination * inverse) / 255U;
    };
    rgba[index] = static_cast<std::uint32_t>(
                      blend(color.red, old & 0xffU)) |
                  static_cast<std::uint32_t>(
                      blend(color.green, (old >> 8U) & 0xffU))
                      << 8U |
                  static_cast<std::uint32_t>(
                      blend(color.blue, (old >> 16U) & 0xffU))
                      << 16U |
                  0xff000000U;
}

void rectangle(std::vector<std::uint32_t> &rgba, std::uint32_t width,
               std::uint32_t height, int left, int top, int right, int bottom,
               Color color) {
    for (int y = top; y < bottom; ++y) {
        for (int x = left; x < right; ++x) {
            pixel(rgba, width, height, x, y, color);
        }
    }
}

void line(std::vector<std::uint32_t> &rgba, std::uint32_t width,
          std::uint32_t height, int x0, int y0, int x1, int y1, Color color) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    for (;;) {
        pixel(rgba, width, height, x0, y0, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int twice = 2 * error;
        if (twice >= dy) {
            error += dy;
            x0 += sx;
        }
        if (twice <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

void text(std::vector<std::uint32_t> &rgba, std::uint32_t width,
          std::uint32_t height, int x, int y, std::string_view value,
          Color color, int scale = 2) {
    for (const char character : value) {
        const auto rows = glyph(character >= 'a' && character <= 'z'
                                    ? static_cast<char>(character - 'a' + 'A')
                                    : character);
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[row] & (1U << (4 - column))) == 0U) {
                    continue;
                }
                rectangle(rgba, width, height, x + column * scale,
                          y + row * scale, x + (column + 1) * scale,
                          y + (row + 1) * scale, color);
            }
        }
        x += 6 * scale;
    }
}

[[nodiscard]] Vec3 normalized(Vec3 value) {
    return math::normalize_or(value, {}, 1.0e-6F);
}

struct ScreenPoint {
    int x{};
    int y{};
    bool visible{};
};

[[nodiscard]] ScreenPoint project(Vec3 point, Camera camera,
                                  std::uint32_t width,
                                  std::uint32_t height) {
    const Vec3 forward = normalized(subtract(camera.target, camera.eye));
    const Vec3 right = normalized(cross(forward, camera.up));
    const Vec3 up = normalized(cross(right, forward));
    const Vec3 relative = subtract(point, camera.eye);
    const float depth = dot(relative, forward);
    if (depth <= 0.01F) {
        return {};
    }
    constexpr float radians = 3.14159265358979323846F / 180.0F;
    const float vertical =
        std::tan(camera.vertical_field_of_view_degrees * radians * 0.5F);
    const float horizontal = vertical * static_cast<float>(width) /
                             static_cast<float>(height);
    const float ndc_x = dot(relative, right) / (depth * horizontal);
    const float ndc_y = dot(relative, up) / (depth * vertical);
    if (std::fabs(ndc_x) > 1.2F || std::fabs(ndc_y) > 1.2F) {
        return {};
    }
    return {static_cast<int>((ndc_x * 0.5F + 0.5F) * width),
            static_cast<int>((0.5F - ndc_y * 0.5F) * height), true};
}

void arrow(std::vector<std::uint32_t> &rgba, std::uint32_t width,
           std::uint32_t height, ScreenPoint start, ScreenPoint end,
           Color color) {
    if (!start.visible || !end.visible) {
        return;
    }
    line(rgba, width, height, start.x, start.y, end.x, end.y, color);
    const float dx = static_cast<float>(end.x - start.x);
    const float dy = static_cast<float>(end.y - start.y);
    const float size = std::sqrt(dx * dx + dy * dy);
    if (size <= 1.0F) {
        return;
    }
    const float ux = dx / size;
    const float uy = dy / size;
    const int left_x = static_cast<int>(end.x - ux * 8.0F + uy * 4.0F);
    const int left_y = static_cast<int>(end.y - uy * 8.0F - ux * 4.0F);
    const int right_x = static_cast<int>(end.x - ux * 8.0F - uy * 4.0F);
    const int right_y = static_cast<int>(end.y - uy * 8.0F + ux * 4.0F);
    line(rgba, width, height, end.x, end.y, left_x, left_y, color);
    line(rgba, width, height, end.x, end.y, right_x, right_y, color);
}

void timing_row(std::vector<std::uint32_t> &rgba, std::uint32_t width,
                std::uint32_t height, int y, const char *label,
                KernelTiming timing) {
    char value[96]{};
    std::snprintf(value, sizeof(value), "%-12s %7.3f MS  %u X", label,
                  timing.total_milliseconds, timing.launch_count);
    text(rgba, width, height, 32, y, value, {225, 235, 242, 255}, 2);
}

struct TimingRow {
    const char *label{};
    KernelTiming timing{};
};

void timing_panel(std::vector<std::uint32_t> &rgba, std::uint32_t width,
                  std::uint32_t height, int right, int bottom,
                  std::string_view title, bool available,
                  std::span<const TimingRow> rows, int row_step, int total_y,
                  float total_milliseconds) {
    rectangle(rgba, width, height, 18, 18, right, bottom, {5, 12, 18, 215});
    text(rgba, width, height, 32, 30, title, {80, 220, 255, 255}, 2);
    if (!available) {
        text(rgba, width, height, 32, 58, "NO TIMING SAMPLE",
             {255, 190, 70, 255}, 2);
        return;
    }
    int y = 58;
    for (const TimingRow &row : rows) {
        timing_row(rgba, width, height, y, row.label, row.timing);
        y += row_step;
    }
    char total[96]{};
    std::snprintf(total, sizeof(total), "TOTAL        %7.3f MS",
                  total_milliseconds);
    text(rgba, width, height, 32, total_y, total, {100, 255, 155, 255}, 2);
}

} // namespace

void draw_timing_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         const WorldStepTimings &timings) {
    const std::array rows{
        TimingRow{"INTEGRATE", timings.rigid_integration},
        TimingRow{"BOUNDS", timings.rigid_world_bounds},
        TimingRow{"PAIR FILTER", timings.rigid_pair_filter},
        TimingRow{"PAIR COMPACT", timings.rigid_pair_compaction},
        TimingRow{"LEAF PAIRS", timings.rigid_leaf_pair_generation},
        TimingRow{"TRI CONTACT", timings.rigid_contact_evaluation},
        TimingRow{"SOLVE", timings.rigid_contact_solve},
        TimingRow{"CLEAR", timings.rigid_input_clear},
        TimingRow{"ROPE", timings.rope_solve}};
    timing_panel(rgba, width, height, 390, 294, "GPU KERNELS",
                 timings.available, rows, 20, 244,
                 timings.total_gpu_milliseconds);
}

void draw_smoke_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &physics,
                               const RendererTimings &renderer,
                               const WorldStatistics &statistics,
                               std::uint32_t capacity) {
    const std::array rows{
        TimingRow{"AIR GRID", physics.smoke_grid},
        TimingRow{"ADVECT + WAKE", physics.smoke_advection},
        TimingRow{"EMIT", physics.smoke_emission}};
    timing_panel(rgba, width, height, 465, 270, "SMOKE GPU KERNELS",
                 physics.available, rows, 24, 138,
                 physics.total_gpu_milliseconds);
    if (!physics.available) return;
    char line[128]{};
    std::snprintf(line, sizeof(line), "RENDER WALL   %7.3f MS",
                  renderer.total_wall_milliseconds);
    text(rgba, width, height, 32, 166, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "SMOKE SPLATS  %7.3f MS",
                  renderer.foam_wall_milliseconds);
    text(rgba, width, height, 32, 188, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "SLOTS %u / MAX %u",
                  statistics.smoke_particle_count, capacity);
    text(rgba, width, height, 32, 210, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "EMITTED %llu",
                  static_cast<unsigned long long>(
                      statistics.emitted_smoke_particle_count));
    text(rgba, width, height, 32, 232, line, {225, 235, 242, 255}, 2);
}

void draw_cloth_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &timings) {
    const std::array rows{
        TimingRow{"RIGID STEP",
            {timings.rigid_integration.total_milliseconds +
             timings.rigid_contact_generation.total_milliseconds +
             timings.rigid_contact_solve.total_milliseconds,
             timings.rigid_integration.launch_count +
             timings.rigid_contact_generation.launch_count +
             timings.rigid_contact_solve.launch_count}},
        TimingRow{"PREDICT", timings.cloth_prediction},
        TimingRow{"LINKS", timings.cloth_constraints},
        TimingRow{"CONTACTS", timings.cloth_contacts}};
    timing_panel(rgba, width, height, 410, 230, "CLOTH GPU KERNELS",
                 timings.available, rows, 26, 177,
                 timings.total_gpu_milliseconds);
}

void draw_soft_body_timing_overlay(std::vector<std::uint32_t> &rgba,
                                   std::uint32_t width, std::uint32_t height,
                                   const WorldStepTimings &timings) {
    const std::array rows{
        TimingRow{"RIGID STEP",
            {timings.rigid_integration.total_milliseconds +
             timings.rigid_contact_generation.total_milliseconds +
             timings.rigid_contact_solve.total_milliseconds,
             timings.rigid_integration.launch_count +
             timings.rigid_contact_generation.launch_count +
             timings.rigid_contact_solve.launch_count}},
        TimingRow{"PREDICT", timings.soft_body_prediction},
        TimingRow{"SPRINGS", timings.soft_body_constraints},
        TimingRow{"CONTACTS", timings.soft_body_contacts},
        TimingRow{"CLOTH STEP",
            {timings.cloth_prediction.total_milliseconds +
             timings.cloth_constraints.total_milliseconds +
             timings.cloth_contacts.total_milliseconds,
             timings.cloth_prediction.launch_count +
             timings.cloth_constraints.launch_count +
             timings.cloth_contacts.launch_count}},
        TimingRow{"SOFT CLOTH", timings.soft_body_cloth_contacts}};
    timing_panel(rgba, width, height, 430, 282, "SOFT BODY GPU KERNELS",
                 timings.available, rows, 26, 229,
                 timings.total_gpu_milliseconds);
}

void draw_fluid_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &physics,
                               const RendererTimings &renderer,
                               const WorldStatistics &statistics,
                               std::uint32_t capacity) {
    const std::array rows{
        TimingRow{"SPAWN", physics.fluid_spawn},
        TimingRow{"CELL SORT", physics.fluid_neighbor_sort},
        TimingRow{"NEIGHBORS", physics.fluid_neighbor_forces},
        TimingRow{"INTEGRATE", physics.fluid_integration},
        TimingRow{"TRI COLLIDE", physics.fluid_static_contacts},
        TimingRow{"BODY INDEX", physics.fluid_body_index},
        TimingRow{"MOVING TRI", physics.fluid_moving_contacts},
        TimingRow{"FLUID CLOTH", physics.fluid_cloth_contacts},
        TimingRow{"FLUID SOFT BODY", physics.fluid_soft_body_contacts},
        TimingRow{"FLUID ROPE", physics.fluid_rope_contacts},
        TimingRow{"SOFT CLOTH", physics.soft_body_cloth_contacts},
        TimingRow{"SOFT PREDICT", physics.soft_body_prediction},
        TimingRow{"SOFT SPRINGS", physics.soft_body_constraints},
        TimingRow{"SOFT RIGID", physics.soft_body_contacts},
        TimingRow{"EVENTS", physics.fluid_contact_events},
        TimingRow{"OUTFLOW", physics.fluid_outflow_compaction},
        TimingRow{"SMOKE HEAT", physics.fluid_smoke_exchange}};
    const int total_y = 58 + static_cast<int>(rows.size()) * 18 + 8;
    const int details_y = total_y + 24;
    timing_panel(rgba, width, height, 480, details_y + 238,
                 renderer.particle_view ? "FLUID PARTICLE TIMINGS"
                                        : "FLUID SURFACE TIMINGS",
                 physics.available, rows, 18, total_y,
                 physics.total_gpu_milliseconds);
    if (!physics.available) return;
    char line[128]{};
    if (renderer.particle_view)
        std::snprintf(line, sizeof(line), "SURFACE GPU       OFF");
    else
        std::snprintf(line, sizeof(line), "SURFACE GPU    %7.3f MS",
                      renderer.surface_gpu_milliseconds);
    text(rgba, width, height, 32, details_y, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "OPTIX + COPY   %7.3f MS",
                  renderer.raytrace_wall_milliseconds);
    text(rgba, width, height, 32, details_y + 20, line, {225, 235, 242, 255}, 2);
    if (renderer.particle_view)
        std::snprintf(line, sizeof(line), "SPRITES CPU   %7.3f MS",
                      renderer.foam_wall_milliseconds);
    else
        std::snprintf(line, sizeof(line), "FOAM CPU      %7.3f MS  %u PATCHES",
                      renderer.foam_wall_milliseconds,
                      renderer.foam_patch_count);
    text(rgba, width, height, 32, details_y + 40, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "RENDER WALL    %7.3f MS",
                  renderer.total_wall_milliseconds);
    text(rgba, width, height, 32, details_y + 64, line, {100, 255, 155, 255}, 2);
    std::snprintf(line, sizeof(line), "LIVE %u / MAX %u",
                  statistics.particle_count, capacity);
    text(rgba, width, height, 32, details_y + 96, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "EMITTED %llu  OUTFLOW %llu",
                  static_cast<unsigned long long>(statistics.emitted_particle_count),
                  static_cast<unsigned long long>(
                      statistics.destroyed_particle_count -
                      statistics.boiled_particle_count));
    text(rgba, width, height, 32, details_y + 116, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "CAPACITY MISSED %llu",
                  static_cast<unsigned long long>(statistics.spawn_capacity_miss_count));
    text(rgba, width, height, 32, details_y + 136, line,
         statistics.spawn_capacity_miss_count
             ? Color{255, 190, 70, 255} : Color{225, 235, 242, 255}, 2);
    if (statistics.boiled_particle_count != 0U) {
        std::snprintf(line, sizeof(line), "BOILED TO SMOKE %llu",
                      static_cast<unsigned long long>(statistics.boiled_particle_count));
        text(rgba, width, height, 32, details_y + 208, line,
             {255, 190, 70, 255}, 2);
    }
    std::snprintf(line, sizeof(line), "SURFACE OUTLIERS %u",
                  renderer.surface_excluded_particle_count);
    text(rgba, width, height, 32, details_y + 160, line,
         renderer.surface_excluded_particle_count
             ? Color{255, 190, 70, 255} : Color{225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "CONTACTS %u  OVERFLOW %u",
                  statistics.contact_count, statistics.contact_overflow_count);
    text(rgba, width, height, 32, details_y + 184, line,
         statistics.contact_overflow_count
             ? Color{255, 190, 70, 255} : Color{225, 235, 242, 255}, 2);
    if (statistics.soft_body_count != 0U) {
        std::snprintf(line, sizeof(line), "WATER SOFT %u  MAX DEPTH %.4f",
            statistics.fluid_soft_body_contact_count,
            statistics.maximum_fluid_soft_body_penetration);
        text(rgba, width, height, 32, details_y + 208, line, {225, 235, 242, 255}, 2);
    }
    if (statistics.rope_count != 0U) {
        std::snprintf(line, sizeof(line), "WATER ROPE %u  MAX DEPTH %.4f",
            statistics.fluid_rope_contact_count,
            statistics.maximum_fluid_rope_penetration);
        text(rgba, width, height, 32, details_y + 228, line, {225, 235, 242, 255}, 2);
    }
}

bool draw_rigid_contact_overlay(std::vector<std::uint32_t> &rgba,
                                std::uint32_t width, std::uint32_t height,
                                RigidContactDeviceView contacts, Camera camera,
                                std::string &error) {
    error.clear();
    std::vector<RigidContactEvent> host(contacts.event_count);
    if (!host.empty()) {
        const cudaError_t copy = cudaMemcpy(
            host.data(), contacts.events.data,
            host.size() * sizeof(RigidContactEvent), cudaMemcpyDeviceToHost);
        if (copy != cudaSuccess) {
            error = std::string("copy rigid contacts: ") + cudaGetErrorString(copy);
            return false;
        }
    }
    for (const RigidContactEvent &contact : host) {
        const ScreenPoint origin = project(contact.position, camera, width, height);
        if (!origin.visible) {
            continue;
        }
        for (int offset = -3; offset <= 3; ++offset) {
            pixel(rgba, width, height, origin.x + offset, origin.y,
                  {255, 70, 70, 255});
            pixel(rgba, width, height, origin.x, origin.y + offset,
                  {255, 70, 70, 255});
        }
        const float normal_length =
            std::clamp(0.16F + contact.normal_impulse * 0.04F, 0.16F, 0.45F);
        arrow(rgba, width, height, origin,
              project(add(contact.position,
                          multiply(contact.normal, normal_length)),
                      camera, width, height),
              {70, 255, 110, 255});
        const float friction_size = length(contact.friction_impulse);
        if (friction_size > 1.0e-6F) {
            const float arrow_length =
                std::clamp(0.12F + friction_size * 0.08F, 0.12F, 0.4F);
            arrow(rgba, width, height, origin,
                  project(add(contact.position,
                              multiply(normalized(contact.friction_impulse),
                                       arrow_length)),
                          camera, width, height),
                  {255, 190, 45, 255});
        }
    }
    rectangle(rgba, width, height, 18, static_cast<int>(height) - 42, 520,
              static_cast<int>(height) - 14, {5, 12, 18, 205});
    text(rgba, width, height, 28, static_cast<int>(height) - 35,
         "RED CONTACT  GREEN NORMAL  ORANGE FRICTION", {235, 240, 245, 255},
         1);
    return true;
}

void draw_physics_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, PhysicsDebugFrameView frame, Camera camera,
    PhysicsDebugVisualizationOptions options) {
    constexpr std::size_t maximum_vectors = 700U;
    const auto draw_vector = [&](Vec3 origin, Vec3 value, Color color,
                                 float scale) {
        const float magnitude = length(value);
        if (!(magnitude > 1.0e-5F) || !std::isfinite(magnitude)) return;
        const float arrow_length = std::clamp(
            scale * std::log1p(magnitude), 0.025F, 0.35F);
        arrow(rgba, width, height,
              project(origin, camera, width, height),
              project(add(origin, multiply(value, arrow_length / magnitude)),
                      camera, width, height), color);
    };
    const auto stride_for = [](std::uint64_t count) {
        return std::max<std::uint64_t>(
            1U, (count + maximum_vectors - 1U) / maximum_vectors);
    };
    if (options.contact_normals) {
        for (std::uint64_t index = 0U; index < frame.rigid_contacts.size;
             index += stride_for(frame.rigid_contacts.size)) {
            const RigidContactEvent &contact = frame.rigid_contacts.data[index];
            draw_vector(contact.position, contact.normal,
                        {48, 255, 95, 245}, 0.12F);
        }
        for (std::uint64_t index = 0U; index < frame.fluid_contacts.size;
             index += stride_for(frame.fluid_contacts.size)) {
            const ContactEvent &contact = frame.fluid_contacts.data[index];
            draw_vector(contact.position, contact.normal,
                        {55, 238, 255, 235}, 0.12F);
        }
    }
    if (options.rigid_forces) {
        for (std::uint64_t index = 0U; index < frame.rigid_bodies.size;
             index += stride_for(frame.rigid_bodies.size)) {
            const PhysicsDebugRigidSample &sample =
                frame.rigid_bodies.data[index];
            draw_vector(sample.state.position, sample.applied_force,
                        {255, 88, 64, 245}, 0.035F);
        }
        for (std::uint64_t index = 0U; index < frame.cloth_vertices.size;
             index += stride_for(frame.cloth_vertices.size)) {
            const PhysicsDebugClothSample &sample =
                frame.cloth_vertices.data[index];
            draw_vector(sample.position, sample.rigid_contact_force,
                        {52, 135, 255, 238}, 0.025F);
            draw_vector(sample.position, sample.soft_body_contact_force,
                        {205, 110, 255, 238}, 0.025F);
        }
        for (std::uint64_t index = 0U; index < frame.soft_body_nodes.size;
             index += stride_for(frame.soft_body_nodes.size)) {
            const PhysicsDebugSoftBodySample &sample =
                frame.soft_body_nodes.data[index];
            draw_vector(sample.position, sample.rigid_contact_force,
                        {145, 92, 255, 238}, 0.025F);
            draw_vector(sample.position, sample.cloth_contact_force,
                        {205, 110, 255, 238}, 0.025F);
        }
        const float inverse_timestep = frame.timestep > 0.0F
            ? 1.0F / frame.timestep : 0.0F;
        for (std::uint64_t index = 0U; index < frame.rigid_contacts.size;
             index += stride_for(frame.rigid_contacts.size)) {
            const RigidContactEvent &contact = frame.rigid_contacts.data[index];
            draw_vector(contact.position,
                        add(multiply(contact.normal,
                                     contact.normal_impulse * inverse_timestep),
                            multiply(contact.friction_impulse,
                                     inverse_timestep)),
                        {255, 180, 35, 238}, 0.025F);
        }
    }
    if (options.fluid_forces) {
        for (std::uint64_t index = 0; index < frame.soft_body_nodes.size;
             index += stride_for(frame.soft_body_nodes.size)) {
            const auto &sample = frame.soft_body_nodes.data[index];
            draw_vector(sample.position, sample.fluid_contact_force,
                        {255, 225, 30, 238}, 0.025F);
        }
        for (std::uint64_t index = 0U; index < frame.fluid_particles.size;
             index += stride_for(frame.fluid_particles.size)) {
            const PhysicsDebugFluidSample &sample =
                frame.fluid_particles.data[index];
            draw_vector(sample.position, sample.acceleration,
                        {20, 210, 255, 225}, 0.018F);
        }
        for (std::uint64_t index = 0U; index < frame.cloth_vertices.size;
             index += stride_for(frame.cloth_vertices.size)) {
            const PhysicsDebugClothSample &sample =
                frame.cloth_vertices.data[index];
            draw_vector(sample.position, sample.fluid_contact_force,
                        {255, 225, 30, 238}, 0.025F);
        }
    }
    if (options.velocities) {
        for (std::uint64_t index = 0U; index < frame.rigid_bodies.size;
             index += stride_for(frame.rigid_bodies.size)) {
            const PhysicsDebugRigidSample &sample =
                frame.rigid_bodies.data[index];
            draw_vector(sample.state.position, sample.state.linear_velocity,
                        {255, 80, 230, 238}, 0.085F);
        }
        for (std::uint64_t index = 0U; index < frame.fluid_particles.size;
             index += stride_for(frame.fluid_particles.size)) {
            const PhysicsDebugFluidSample &sample =
                frame.fluid_particles.data[index];
            draw_vector(sample.position, sample.velocity,
                        {255, 80, 230, 215}, 0.085F);
        }
        for (std::uint64_t index = 0U; index < frame.cloth_vertices.size;
             index += stride_for(frame.cloth_vertices.size)) {
            const PhysicsDebugClothSample &sample =
                frame.cloth_vertices.data[index];
            draw_vector(sample.position, sample.velocity,
                        {255, 80, 230, 220}, 0.085F);
        }
        for (std::uint64_t index = 0U; index < frame.soft_body_nodes.size;
             index += stride_for(frame.soft_body_nodes.size)) {
            const PhysicsDebugSoftBodySample &sample =
                frame.soft_body_nodes.data[index];
            draw_vector(sample.position, sample.velocity,
                        {190, 95, 255, 220}, 0.085F);
        }
    }
    for(std::uint64_t i=0;i<frame.rope_nodes.size;i+=stride_for(frame.rope_nodes.size)) {
        const auto &sample=frame.rope_nodes.data[i];
        if(options.velocities)draw_vector(sample.position,sample.velocity,{190,95,255,220},0.085F);
        if(options.rigid_forces) {
            draw_vector(sample.position,sample.constraint_force,{255,170,40,235},0.02F);
            draw_vector(sample.position,sample.contact_force,{255,70,70,235},0.02F);
        }
        if(options.fluid_forces)
            draw_vector(sample.position,sample.fluid_contact_force,{255,225,30,238},0.025F);
        if(options.contact_normals)draw_vector(sample.position,
            math::normalize_or(sample.contact_force,{}),{48,255,95,245},0.12F);
    }
    if (options.contact_normals || options.rigid_forces ||
        options.fluid_forces || options.velocities) {
        rectangle(rgba, width, height, 18, static_cast<int>(height) - 62,
                  690, static_cast<int>(height) - 14, {5, 12, 18, 205});
        text(rgba, width, height, 28, static_cast<int>(height) - 55,
             "Z NORMALS  X CONTACT FORCES  C FLUID FORCES  N VELOCITIES",
             {235, 240, 245, 255}, 1);
        char summary[128]{};
        std::snprintf(summary, sizeof(summary),
                      "FRAME %llu  MAX FLUID NEIGHBORS %u",
                      static_cast<unsigned long long>(frame.frame_index),
                      frame.maximum_fluid_neighbor_count);
        text(rgba, width, height, 28, static_cast<int>(height) - 35,
             summary, {130, 220, 255, 255}, 1);
    }
}

bool draw_smoke_grid_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, SmokeDeviceView smoke, Camera camera,
    SmokeDebugMode mode, std::string &error) {
    error.clear();
    if (mode == SmokeDebugMode::none) return true;
    const std::uint64_t expected = std::uint64_t(smoke.grid_resolution) *
        smoke.grid_vertical_resolution * smoke.grid_resolution;
    if (smoke.grid_resolution == 0U ||
        smoke.grid_vertical_resolution == 0U || expected == 0U ||
        expected > std::numeric_limits<std::size_t>::max() ||
        !(smoke.grid_spacing > 0.0F) ||
        !std::isfinite(smoke.grid_spacing)) {
        error = "smoke grid debug view is unavailable";
        return false;
    }
    const auto valid = [expected](auto span) {
        return span.size == expected && span.data != nullptr;
    };
    if (!valid(smoke.grid_velocity) || !valid(smoke.grid_pressure) ||
        !valid(smoke.grid_density) || !valid(smoke.grid_temperature) ||
        !valid(smoke.grid_solid) || !valid(smoke.grid_vorticity) ||
        !valid(smoke.grid_divergence)) {
        error = "smoke grid debug fields disagree with grid dimensions";
        return false;
    }
    const auto copy = [&](auto span, auto &host, const char *label) {
        using Value = typename std::decay_t<decltype(host)>::value_type;
        host.resize(static_cast<std::size_t>(span.size));
        const cudaError_t result = cudaMemcpy(
            host.data(), span.data, host.size() * sizeof(Value),
            cudaMemcpyDeviceToHost);
        if (result == cudaSuccess) return true;
        error = std::string("copy smoke grid ") + label + ": " +
                cudaGetErrorString(result);
        return false;
    };

    const std::uint32_t n = smoke.grid_resolution;
    const std::uint32_t h = smoke.grid_vertical_resolution;
    const auto index = [n, h](std::uint32_t x, std::uint32_t y,
                              std::uint32_t z) {
        return static_cast<std::size_t>(x) + std::size_t(n) *
            (std::size_t(y) + std::size_t(h) * z);
    };
    const auto center = [&](std::uint32_t x, std::uint32_t y,
                            std::uint32_t z) {
        return add(smoke.grid_minimum,
                   multiply(Vec3{float(x) + 0.5F, float(y) + 0.5F,
                                 float(z) + 0.5F},
                            smoke.grid_spacing));
    };
    const auto on_slice = [n, h](std::uint32_t x, std::uint32_t y,
                                  std::uint32_t z) {
        return x == n / 2U || y == h / 2U || z == n / 2U;
    };
    const auto draw_point = [&](Vec3 position, Color value, int radius = 1) {
        const ScreenPoint point = project(position, camera, width, height);
        if (!point.visible) return;
        rectangle(rgba, width, height, point.x - radius, point.y - radius,
                  point.x + radius + 1, point.y + radius + 1, value);
    };
    const auto draw_world_line = [&](Vec3 a, Vec3 b, Color value) {
        const ScreenPoint first = project(a, camera, width, height);
        const ScreenPoint second = project(b, camera, width, height);
        if (first.visible && second.visible)
            line(rgba, width, height, first.x, first.y, second.x, second.y,
                 value);
    };
    const Vec3 low = smoke.grid_minimum;
    const Vec3 high = add(low, multiply(
        Vec3{float(n), float(h), float(n)}, smoke.grid_spacing));
    const std::array<Vec3, 8> corners{{
        {low.x, low.y, low.z}, {high.x, low.y, low.z},
        {low.x, high.y, low.z}, {high.x, high.y, low.z},
        {low.x, low.y, high.z}, {high.x, low.y, high.z},
        {low.x, high.y, high.z}, {high.x, high.y, high.z}}};
    constexpr std::array<std::array<int, 2>, 12> edges{{
        {{0,1}},{{0,2}},{{1,3}},{{2,3}},{{4,5}},{{4,6}},
        {{5,7}},{{6,7}},{{0,4}},{{1,5}},{{2,6}},{{3,7}}}};
    for (const auto edge : edges)
        draw_world_line(corners[edge[0]], corners[edge[1]],
                        {40, 220, 255, 170});

    const auto signed_color = [](float value, float scale,
                                  std::uint8_t alpha = 205U) {
        const float t = std::clamp(value / std::max(scale, 1.0e-12F),
                                   -1.0F, 1.0F);
        if (t < 0.0F) {
            const float amount = -t;
            return Color{
                static_cast<std::uint8_t>(235.0F - 205.0F * amount),
                static_cast<std::uint8_t>(235.0F - 105.0F * amount),
                255U, alpha};
        }
        return Color{255U,
            static_cast<std::uint8_t>(235.0F - 185.0F * t),
            static_cast<std::uint8_t>(235.0F - 210.0F * t), alpha};
    };
    const auto vector_color = [](Vec3 value, float scale) {
        const float magnitude = length(value);
        if (!(magnitude > 1.0e-8F) || !std::isfinite(magnitude))
            return Color{0U, 0U, 0U, 0U};
        const Vec3 direction = multiply(value, 1.0F / magnitude);
        const float intensity = std::sqrt(std::clamp(
            magnitude / std::max(scale, 1.0e-8F), 0.0F, 1.0F));
        const auto channel = [intensity](float component) {
            return static_cast<std::uint8_t>(std::clamp(
                (0.5F + 0.5F * component) * (80.0F + 175.0F * intensity),
                0.0F, 255.0F));
        };
        return Color{channel(direction.x), channel(direction.y),
                     channel(direction.z),
                     static_cast<std::uint8_t>(55.0F + 190.0F * intensity)};
    };

    float scale = 1.0F;
    const char *title = "GRID";
    if (mode == SmokeDebugMode::grid) {
        std::vector<std::uint32_t> solid;
        if (!copy(smoke.grid_solid, solid, "solid cells")) return false;
        const std::uint32_t sx = std::max(1U, n / 16U);
        const std::uint32_t sy = std::max(1U, h / 8U);
        const float mx = low.x + float(n / 2U) * smoke.grid_spacing;
        const float my = low.y + float(h / 2U) * smoke.grid_spacing;
        const float mz = low.z + float(n / 2U) * smoke.grid_spacing;
        for (std::uint32_t x = 0U; x <= n; x += sx) {
            const float px = low.x + float(x) * smoke.grid_spacing;
            draw_world_line({px, low.y, mz}, {px, high.y, mz},
                            {55, 190, 220, 70});
            draw_world_line({px, my, low.z}, {px, my, high.z},
                            {55, 190, 220, 70});
        }
        for (std::uint32_t y = 0U; y <= h; y += sy) {
            const float py = low.y + float(y) * smoke.grid_spacing;
            draw_world_line({low.x, py, mz}, {high.x, py, mz},
                            {55, 190, 220, 70});
            draw_world_line({mx, py, low.z}, {mx, py, high.z},
                            {55, 190, 220, 70});
        }
        for (std::uint32_t z = 0U; z <= n; z += sx) {
            const float pz = low.z + float(z) * smoke.grid_spacing;
            draw_world_line({low.x, my, pz}, {high.x, my, pz},
                            {55, 190, 220, 70});
        }
        for (std::uint32_t z = 0U; z < n; ++z)
            for (std::uint32_t y = 0U; y < h; ++y)
                for (std::uint32_t x = 0U; x < n; ++x)
                    if (solid[index(x,y,z)] != 0U)
                        draw_point(center(x,y,z), {255, 125, 25, 235}, 2);
    } else if (mode == SmokeDebugMode::velocity ||
               mode == SmokeDebugMode::vorticity) {
        std::vector<Vec3> values;
        if (!copy(mode == SmokeDebugMode::velocity ? smoke.grid_velocity
                                                   : smoke.grid_vorticity,
                  values, mode == SmokeDebugMode::velocity ? "velocity"
                                                           : "vorticity"))
            return false;
        scale = 1.0e-8F;
        for (std::uint32_t z = 0U; z < n; ++z)
            for (std::uint32_t y = 0U; y < h; ++y)
                for (std::uint32_t x = 0U; x < n; ++x)
                    if (on_slice(x,y,z))
                        scale = std::max(scale, length(values[index(x,y,z)]));
        for (std::uint32_t z = 0U; z < n; ++z)
            for (std::uint32_t y = 0U; y < h; ++y)
                for (std::uint32_t x = 0U; x < n; ++x) {
                    if (!on_slice(x,y,z)) continue;
                    const Vec3 value = values[index(x,y,z)];
                    const Color mapped = vector_color(value, scale);
                    if (mapped.alpha != 0U) draw_point(center(x,y,z), mapped);
                }
        if (mode == SmokeDebugMode::velocity) {
            const std::uint32_t step = std::max(2U, n / 24U);
            const std::uint32_t z = n / 2U;
            for (std::uint32_t y = 0U; y < h; y += step)
                for (std::uint32_t x = 0U; x < n; x += step) {
                    const Vec3 value = values[index(x,y,z)];
                    const float magnitude = length(value);
                    if (!(magnitude > 1.0e-5F)) continue;
                    const Vec3 origin = center(x,y,z);
                    const Vec3 endpoint = add(origin, multiply(
                        value, 2.5F * smoke.grid_spacing /
                            std::max(scale, 1.0e-8F)));
                    arrow(rgba, width, height,
                          project(origin, camera, width, height),
                          project(endpoint, camera, width, height),
                          {255, 255, 255, 205});
                }
            title = "VELOCITY SIGNED RGB XYZ";
        } else {
            title = "VORTICITY SIGNED RGB XYZ";
        }
    } else if (mode == SmokeDebugMode::density_temperature) {
        std::vector<float> density;
        std::vector<float> temperature;
        if (!copy(smoke.grid_density, density, "density") ||
            !copy(smoke.grid_temperature, temperature, "temperature"))
            return false;
        scale = 1.0e-8F;
        float heat_scale = 1.0e-8F;
        for (std::size_t cell = 0U; cell < density.size(); ++cell) {
            scale = std::max(scale, density[cell]);
            if (density[cell] > 1.0e-8F)
                heat_scale = std::max(heat_scale,
                                      temperature[cell] / density[cell]);
        }
        for (std::uint32_t z = 0U; z < n; ++z)
            for (std::uint32_t y = 0U; y < h; ++y)
                for (std::uint32_t x = 0U; x < n; ++x) {
                    const std::size_t cell = index(x,y,z);
                    if (density[cell] < scale * 0.01F) continue;
                    const float amount = std::sqrt(std::clamp(
                        density[cell] / scale, 0.0F, 1.0F));
                    const float heat = std::clamp(
                        temperature[cell] /
                            std::max(density[cell] * heat_scale, 1.0e-8F),
                        0.0F, 1.0F);
                    draw_point(center(x,y,z),
                        {static_cast<std::uint8_t>(25.0F + 230.0F * heat),
                         static_cast<std::uint8_t>(185.0F - 85.0F * heat),
                         static_cast<std::uint8_t>(255.0F - 225.0F * heat),
                         static_cast<std::uint8_t>(50.0F + 205.0F * amount)}, 2);
                }
        title = "DENSITY BLUE  HEAT RED";
    } else {
        std::vector<float> values;
        const auto source = mode == SmokeDebugMode::pressure
            ? smoke.grid_pressure : smoke.grid_divergence;
        if (!copy(source, values, mode == SmokeDebugMode::pressure
                                      ? "pressure" : "divergence"))
            return false;
        scale = 1.0e-12F;
        for (std::uint32_t z = 0U; z < n; ++z)
            for (std::uint32_t y = 0U; y < h; ++y)
                for (std::uint32_t x = 0U; x < n; ++x)
                    if (on_slice(x,y,z))
                        scale = std::max(scale,
                            std::fabs(values[index(x,y,z)]));
        for (std::uint32_t z = 0U; z < n; ++z)
            for (std::uint32_t y = 0U; y < h; ++y)
                for (std::uint32_t x = 0U; x < n; ++x) {
                    if (!on_slice(x,y,z)) continue;
                    draw_point(center(x,y,z),
                               signed_color(values[index(x,y,z)], scale));
                }
        title = mode == SmokeDebugMode::pressure
            ? "PRESSURE BLUE LOW  RED HIGH"
            : "DIVERGENCE BLUE NEG  RED POS";
    }

    const int top = static_cast<int>(height) - 80;
    rectangle(rgba, width, height, 14, top, std::min<int>(width - 14, 930),
              static_cast<int>(height) - 10, {5, 12, 18, 220});
    text(rgba, width, height, 24, top + 8,
         "SMOKE  Z GRID  X VELOCITY  C PRESSURE  V DENSITY HEAT  B VORTICITY  N DIVERGENCE",
         {235, 240, 245, 255}, 1);
    char summary[160]{};
    if (mode == SmokeDebugMode::grid) {
        std::snprintf(summary, sizeof(summary),
                      "GRID %uX%uX%u  ORANGE SOLID CUT CELLS",
                      n, h, n);
    } else {
        std::snprintf(summary, sizeof(summary), "%s  MAX %.4f", title, scale);
    }
    text(rgba, width, height, 24, top + 27, summary,
         {130, 220, 255, 255}, 1);
    const int bar_y = top + 48;
    if (mode == SmokeDebugMode::pressure ||
        mode == SmokeDebugMode::divergence) {
        for (int x = 0; x < 240; ++x)
            rectangle(rgba, width, height, 24 + x, bar_y, 25 + x, bar_y + 10,
                      signed_color(float(x) / 119.5F - 1.0F, 1.0F, 255U));
    } else if (mode == SmokeDebugMode::density_temperature) {
        for (int x = 0; x < 240; ++x) {
            const float heat = float(x) / 239.0F;
            rectangle(rgba, width, height, 24 + x, bar_y, 25 + x, bar_y + 10,
                {static_cast<std::uint8_t>(25.0F + 230.0F * heat),
                 static_cast<std::uint8_t>(185.0F - 85.0F * heat),
                 static_cast<std::uint8_t>(255.0F - 225.0F * heat), 255U});
        }
    } else if (mode == SmokeDebugMode::velocity ||
               mode == SmokeDebugMode::vorticity) {
        rectangle(rgba, width, height, 24, bar_y, 94, bar_y + 10,
                  {255, 55, 55, 255});
        rectangle(rgba, width, height, 94, bar_y, 164, bar_y + 10,
                  {55, 255, 55, 255});
        rectangle(rgba, width, height, 164, bar_y, 234, bar_y + 10,
                  {55, 55, 255, 255});
    } else {
        rectangle(rgba, width, height, 24, bar_y, 144, bar_y + 10,
                  {40, 220, 255, 255});
        rectangle(rgba, width, height, 144, bar_y, 264, bar_y + 10,
                  {255, 125, 25, 255});
    }
    return true;
}

bool draw_cloth_debug_overlay(std::vector<std::uint32_t> &rgba,
                              std::uint32_t width, std::uint32_t height,
                              ClothDeviceView cloth, Camera camera,
                              ClothDebugOptions options, std::string &error) {
    error.clear();
    const auto copy = [&](auto span, auto &host, const char *label) {
        using Value = typename std::decay_t<decltype(host)>::value_type;
        host.resize(span.size);
        if (host.empty()) return true;
        const cudaError_t result = cudaMemcpy(
            host.data(), span.data, host.size() * sizeof(Value),
            cudaMemcpyDeviceToHost);
        if (result == cudaSuccess) return true;
        error = std::string("copy ") + label + ": " +
                cudaGetErrorString(result);
        return false;
    };
    std::vector<Vec3> positions;
    std::vector<Vec3> surface;
    std::vector<std::uint32_t> triangles;
    if (!copy(cloth.positions, positions, "cloth positions")) return false;
    const bool fractured_surface = cloth.surface_positions.size != 0U;
    const auto &triangle_positions = fractured_surface ? surface : positions;
    if (options.wireframe || options.normals) {
        if (fractured_surface != (cloth.surface_triangle_indices.size != 0U)) {
            error = "cloth debug surface positions and indices disagree";
            return false;
        }
        // Authored connectivity belongs to the physical graph. After tearing,
        // draw the same triangle-local surface that the solid renderer uses.
        if (fractured_surface &&
            !copy(cloth.surface_positions, surface, "cloth surface positions"))
            return false;
        if (!copy(fractured_surface ? cloth.surface_triangle_indices
                                    : cloth.triangle_indices,
                  triangles, "cloth surface triangles")) return false;
        if (triangles.size() % 3U != 0U) {
            error = "cloth debug triangle index count is not divisible by three";
            return false;
        }
        for (const std::uint32_t index : triangles) {
            if (index >= triangle_positions.size()) {
                error = "cloth debug triangle index is out of range";
                return false;
            }
        }
    }
    if (options.wireframe) {
        const Color wire{26, 230, 255, 225};
        for (std::size_t triangle = 0U; triangle < triangles.size();
             triangle += 3U) {
            const std::uint32_t indices[3]{triangles[triangle],
                triangles[triangle + 1U], triangles[triangle + 2U]};
            for (int edge = 0; edge < 3; ++edge) {
                const ScreenPoint first = project(
                    triangle_positions[indices[edge]], camera, width, height);
                const ScreenPoint second = project(
                    triangle_positions[indices[(edge + 1) % 3]], camera, width, height);
                if (first.visible && second.visible)
                    line(rgba, width, height, first.x, first.y,
                         second.x, second.y, wire);
            }
        }
    }
    if (options.bonds) {
        std::vector<ClothBond> bonds;
        std::vector<std::uint8_t> active;
        if (!copy(cloth.bonds, bonds, "cloth bonds") ||
            !copy(cloth.active_bonds, active, "cloth active bonds"))
            return false;
        const std::size_t stride = std::max<std::size_t>(
            1U, (bonds.size() + 2'499U) / 2'500U);
        for (std::size_t index = 0U; index < bonds.size(); index += stride) {
            if (index < active.size() && active[index] == 0U) continue;
            const ClothBond &bond = bonds[index];
            if (bond.first >= positions.size() ||
                bond.second >= positions.size()) continue;
            const ScreenPoint first = project(
                positions[bond.first], camera, width, height);
            const ScreenPoint second = project(
                positions[bond.second], camera, width, height);
            if (!first.visible || !second.visible) continue;
            line(rgba, width, height, first.x, first.y, second.x, second.y,
                 {255, 155, 25, 220});
        }
    }
    std::vector<Vec3> normals;
    if (options.normals) {
        normals.assign(triangle_positions.size(), {});
        for (std::size_t triangle = 0U; triangle < triangles.size();
             triangle += 3U) {
            const std::uint32_t a = triangles[triangle];
            const std::uint32_t b = triangles[triangle + 1U];
            const std::uint32_t c = triangles[triangle + 2U];
            const Vec3 face = cross(subtract(triangle_positions[b], triangle_positions[a]),
                                    subtract(triangle_positions[c], triangle_positions[a]));
            normals[a] = add(normals[a], face);
            normals[b] = add(normals[b], face);
            normals[c] = add(normals[c], face);
        }
        for (Vec3 &normal : normals) normal = normalized(normal);
    }
    const auto draw_vectors = [&](const std::vector<Vec3> &origins,
                                  const std::vector<Vec3> &vectors,
                                  float fixed_length, Color color) {
        const std::size_t count = std::min(origins.size(), vectors.size());
        for (std::size_t index = 0U; index < count; ++index) {
            const float magnitude = length(vectors[index]);
            if (!(magnitude > 1.0e-5F) || !std::isfinite(magnitude)) continue;
            const float arrow_length = fixed_length > 0.0F
                ? fixed_length
                : std::clamp(0.003F * magnitude, 0.012F, 0.14F);
            const Vec3 endpoint = add(
                origins[index], multiply(vectors[index],
                                            arrow_length / magnitude));
            arrow(rgba, width, height,
                  project(origins[index], camera, width, height),
                  project(endpoint, camera, width, height), color);
        }
    };
    if (options.normals)
        draw_vectors(triangle_positions, normals, 0.065F, {31, 255, 56, 242});
    if (options.rigid_contact_forces) {
        std::vector<Vec3> forces;
        if (!copy(cloth.rigid_contact_forces, forces,
                  "cloth rigid contact forces")) return false;
        draw_vectors(positions, forces, 0.0F, {31, 122, 255, 242});
        if (!copy(cloth.soft_body_contact_forces, forces,
                  "cloth soft-body contact forces")) return false;
        draw_vectors(positions, forces, 0.0F, {205, 110, 255, 242});
    }
    if (options.fluid_contact_forces) {
        std::vector<Vec3> forces;
        if (!copy(cloth.fluid_contact_forces, forces,
                  "cloth fluid contact forces")) return false;
        draw_vectors(positions, forces, 0.0F, {255, 219, 20, 242});
    }
    if (options.normals || options.rigid_contact_forces ||
        options.fluid_contact_forces || options.wireframe || options.bonds) {
        rectangle(rgba, width, height, 18, static_cast<int>(height) - 88,
                  620, static_cast<int>(height) - 66, {5, 12, 18, 205});
        text(rgba, width, height, 28, static_cast<int>(height) - 81,
             "CLOTH  Z NORMALS  V WIREFRAME  B BONDS",
             {235, 240, 245, 255}, 1);
    }
    return true;
}

bool draw_rope_debug_overlay(std::vector<std::uint32_t> &rgba,
    std::uint32_t width,std::uint32_t height,RopeDeviceView rope,Camera camera,std::string &error) {
    std::vector<Vec3> nodes(rope.positions.size);
    const auto status=cudaMemcpy(nodes.data(),rope.positions.data,nodes.size()*sizeof(Vec3),cudaMemcpyDeviceToHost);
    if(status!=cudaSuccess){error=cudaGetErrorString(status);return false;}
    for(std::size_t i=1;i<nodes.size();++i) {
        const auto a=project(nodes[i-1],camera,width,height),b=project(nodes[i],camera,width,height);
        if(a.visible && b.visible)line(rgba,width,height,a.x,a.y,b.x,b.y,{60,240,255,245});
    }
    return true;
}

bool draw_soft_body_debug_overlay(std::vector<std::uint32_t> &rgba,
                                  std::uint32_t width, std::uint32_t height,
                                  SoftBodyDeviceView body, Camera camera,
                                  ClothDebugOptions options,
                                  std::string &error) {
    error.clear();
    const auto copy = [&](auto span, auto &host, const char *label) {
        using Value = typename std::decay_t<decltype(host)>::value_type;
        host.resize(span.size);
        if (host.empty()) return true;
        const cudaError_t result = cudaMemcpy(
            host.data(), span.data, host.size() * sizeof(Value),
            cudaMemcpyDeviceToHost);
        if (result == cudaSuccess) return true;
        error = std::string("copy ") + label + ": " +
                cudaGetErrorString(result);
        return false;
    };
    std::vector<Vec3> nodes;
    std::vector<Vec3> surface;
    std::vector<std::uint32_t> triangles;
    if (!copy(body.positions, nodes, "soft-body nodes") ||
        !copy(body.surface_positions, surface, "soft-body surface") ||
        !copy(body.surface_triangle_indices, triangles,
              "soft-body triangles")) return false;
    if (triangles.size() % 3U != 0U) {
        error = "soft-body triangle index count is not divisible by three";
        return false;
    }
    if (options.wireframe || options.normals) {
        const Color wire{186, 105, 255, 225};
        std::vector<Vec3> normals(surface.size(), Vec3{});
        for (std::size_t triangle = 0U; triangle < triangles.size();
             triangle += 3U) {
            const std::uint32_t ids[3]{triangles[triangle],
                triangles[triangle + 1U], triangles[triangle + 2U]};
            if (ids[0] >= surface.size() || ids[1] >= surface.size() ||
                ids[2] >= surface.size()) {
                error = "soft-body triangle index is out of range";
                return false;
            }
            if (options.wireframe) {
                for (std::uint32_t edge = 0U; edge < 3U; ++edge) {
                    const ScreenPoint first = project(
                        surface[ids[edge]], camera, width, height);
                    const ScreenPoint second = project(
                        surface[ids[(edge + 1U) % 3U]], camera, width, height);
                    if (first.visible && second.visible)
                        line(rgba, width, height, first.x, first.y,
                             second.x, second.y, wire);
                }
            }
            if (options.normals) {
                const Vec3 face = cross(
                    subtract(surface[ids[1]], surface[ids[0]]),
                    subtract(surface[ids[2]], surface[ids[0]]));
                for (std::uint32_t corner = 0U; corner < 3U; ++corner)
                    normals[ids[corner]] = add(normals[ids[corner]], face);
            }
        }
        if (options.normals) {
            const std::size_t stride = std::max<std::size_t>(
                1U, (surface.size() + 699U) / 700U);
            for (std::size_t vertex = 0U; vertex < surface.size();
                 vertex += stride) {
                const Vec3 normal = normalized(normals[vertex]);
                arrow(rgba, width, height,
                    project(surface[vertex], camera, width, height),
                    project(add(surface[vertex], multiply(normal, 0.065F)),
                            camera, width, height),
                    {31, 255, 56, 242});
            }
        }
    }
    if (options.bonds) {
        std::vector<SoftBodyBond> bonds;
        if (!copy(body.bonds, bonds, "soft-body bonds")) return false;
        for (std::size_t index = 0U; index < bonds.size(); ++index) {
            const SoftBodyBond &bond = bonds[index];
            if (bond.first >= nodes.size() || bond.second >= nodes.size())
                continue;
            const ScreenPoint first = project(
                nodes[bond.first], camera, width, height);
            const ScreenPoint second = project(
                nodes[bond.second], camera, width, height);
            if (first.visible && second.visible)
                line(rgba, width, height, first.x, first.y,
                     second.x, second.y, {255, 155, 25, 190});
        }
    }
    if (options.rigid_contact_forces) {
        for (auto source : {body.rigid_contact_forces, body.cloth_contact_forces}) {
            std::vector<Vec3> forces;
            if (!copy(source, forces, "soft-body contact forces")) return false;
            const std::size_t count = std::min(nodes.size(), forces.size());
            for (std::size_t node = 0U; node < count; ++node) {
                const float magnitude = length(forces[node]);
                if (!(magnitude > 1.0e-5F)) continue;
                const float arrow_length =
                    std::clamp(0.003F * magnitude, 0.012F, 0.14F);
                arrow(rgba, width, height,
                    project(nodes[node], camera, width, height),
                    project(add(nodes[node], multiply(forces[node],
                        arrow_length / magnitude)), camera, width, height),
                    {31, 122, 255, 242});
            }
        }
    }
    if (options.normals || options.rigid_contact_forces ||
        options.wireframe || options.bonds) {
        rectangle(rgba, width, height, 18, static_cast<int>(height) - 88,
                  650, static_cast<int>(height) - 66, {5, 12, 18, 205});
        text(rgba, width, height, 28, static_cast<int>(height) - 81,
             "SOFT BODY  Z NORMALS  V SPRINGS  B SURFACE",
             {235, 240, 245, 255}, 1);
    }
    return true;
}

void draw_context_overlay(std::vector<std::uint32_t> &rgba,
                          std::uint32_t width, std::uint32_t height,
                          GalleryContext selection) {
    const int center = static_cast<int>(width) / 2;
    constexpr int row_height = 68;
    const int count = static_cast<int>(gallery_entries.size());
    const int visible = std::clamp((static_cast<int>(height) - 110) / row_height, 1, count);
    const int selected = static_cast<int>(gallery_context_index(selection));
    const int first = std::clamp(selected - visible / 2, 0, count - visible);
    const int panel_height = 81 + row_height * visible;
    const int top = std::max(14, (static_cast<int>(height) - panel_height) / 2);
    rectangle(rgba, width, height, center - 255, top, center + 255,
              top + panel_height,
              {4, 10, 16, 230});
    text(rgba, width, height, center - 225, top + 24, "SCENES",
         {110, 225, 255, 255}, 3);
    text(rgba, width, height, center - 70, top + 34,
         "Z X C V B N DEBUG   M CAPTURE", {185, 220, 235, 255}, 1);

    const auto row = [&](int y, GalleryContext context, Color background,
                         Color icon, std::string_view name,
                         std::string_view state, Color state_color) {
        if (selection == context) {
            rectangle(rgba, width, height, center - 226, y - 6, center + 226,
                      y + 60, {105, 255, 155, 255});
        }
        rectangle(rgba, width, height, center - 220, y, center + 220, y + 54,
                  background);
        rectangle(rgba, width, height, center - 198, y + 8, center - 160,
                  y + 46, icon);
        text(rgba, width, height, center - 135, y + 6, name,
             {245, 247, 250, 255}, 2);
        text(rgba, width, height, center - 135, y + 31, state, state_color, 1);
    };

    int y = top + 78;
    for (int index = first; index < first + visible; ++index) {
        const GalleryEntry &entry = gallery_entries[static_cast<std::size_t>(index)];
        row(y, entry.context, color(entry.background), color(entry.icon),
            entry.name, entry.help, {105, 255, 155, 255});
        y += row_height;
    }
}

void draw_count_overlay(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, const std::string &value,
                        bool invalid) {
    const GalleryEntry &entry = gallery_entry(context);
    const bool fluid = entry.count_kind == GalleryCountKind::fluid_particles;
    const int center_x = static_cast<int>(width) / 2;
    const int center_y = static_cast<int>(height) / 2;
    rectangle(rgba, width, height, center_x - 260, center_y - 118,
              center_x + 260, center_y + 118, {4, 10, 16, 242});
    text(rgba, width, height, center_x - 220, center_y - 88,
         fluid ? "FLUID PARTICLE CAP" : "DUMP SPHERES",
         fluid ? Color{35, 150, 255, 255} : Color{245, 130, 45, 255}, 2);
    rectangle(rgba, width, height, center_x - 220, center_y - 30,
              center_x + 220, center_y + 20,
              invalid ? Color{105, 20, 20, 255} : Color{27, 38, 48, 255});
    text(rgba, width, height, center_x - 198, center_y - 17,
         (fluid ? "MAX " : "COUNT ") + value, {245, 247, 250, 255}, 2);
    text(rgba, width, height, center_x - 220, center_y + 42,
         invalid ? ("USE " + std::to_string(entry.minimum_count) + '-' +
                    std::to_string(entry.maximum_count))
                 : ("MIN " + std::to_string(entry.minimum_count) + "  MAX " +
                    std::to_string(entry.maximum_count)),
         invalid ? Color{255, 105, 105, 255} : Color{160, 190, 210, 255}, 1);
    text(rgba, width, height, center_x - 220, center_y + 72,
         "ENTER APPLY  ESC CANCEL", {160, 190, 210, 255}, 1);
}

} // namespace parallel_mater::gallery
