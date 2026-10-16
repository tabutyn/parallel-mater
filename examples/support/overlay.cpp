// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/overlay.hpp>
#include <parallel_mater_gallery/runtime_platform.hpp>
#include <parallel_mater_gallery/scene.hpp>
#include <scene_builder.hpp>
#include <parallel_mater_gallery/bitmap_font.hpp>

#include "vector_math.hpp"

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

struct MenuScale {
    float value{1.0F};

    [[nodiscard]] int pixels(int amount) const {
        return static_cast<int>(std::lround(static_cast<float>(amount) * value));
    }

    [[nodiscard]] int font(int amount) const {
        return std::max(1, pixels(amount));
    }
};

[[nodiscard]] MenuScale menu_scale(std::uint32_t width,
                                   std::uint32_t height) {
    // The native window presents at 1280x720 while the renderer may use a
    // larger device profile. Scale both layout and glyphs before that image
    // is downsampled so menus keep their intended presentation size.
    constexpr float presentation_width = 1280.0F;
    constexpr float presentation_height = 720.0F;
    return {std::max(0.25F,
                     std::min(static_cast<float>(width) / presentation_width,
                              static_cast<float>(height) /
                                  presentation_height))};
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
                GalleryKernelTiming timing) {
    char value[96]{};
    std::snprintf(value, sizeof(value), "%-12s %7.3f MS  %u X", label,
                  timing.total_milliseconds, timing.launch_count);
    text(rgba, width, height, 32, y, value, {225, 235, 242, 255}, 2);
}

struct TimingRow {
    const char *label{};
    GalleryKernelTiming timing{};
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
                         const GalleryStepTimings &timings,
                         const GalleryStatistics &statistics) {
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
    timing_panel(rgba, width, height, 420, 326, "GPU TIMINGS / X: LAUNCHES",
                 timings.available, rows, 20, 244,
                 timings.total_gpu_milliseconds);
    if (timings.available) {
        char passes[64]{};
        std::snprintf(passes, sizeof(passes), "AVBD ITERATIONS MAX: %u",
                      statistics.rigid_contact_maximum_passes);
        text(rgba, width, height, 32, 274, passes, {225, 235, 242, 255}, 2);
        text(rgba, width, height, 32, 298, "LAST SUBSTEP / NO WARM START",
             {225, 235, 242, 255}, 2);
    }
}

void draw_smoke_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const GalleryStepTimings &physics,
                               const RendererTimings &renderer,
                               const GalleryStatistics &statistics,
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
                               const GalleryStepTimings &timings) {
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
                                   const GalleryStepTimings &timings) {
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
                               const GalleryStepTimings &physics,
                               const RendererTimings &renderer,
                               const GalleryStatistics &statistics,
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

void draw_context_overlay(std::vector<std::uint32_t> &rgba,
                          std::uint32_t width, std::uint32_t height,
                          GalleryContext selection,
                          const PmPhysicalDeviceFeatures *features) {
    const MenuScale ui = menu_scale(width, height);
    const int center = static_cast<int>(width) / 2;
    const int row_height = ui.pixels(68);
    const int count = static_cast<int>(gallery_entries.size());
    const int visible = std::clamp(
        (static_cast<int>(height) - ui.pixels(110)) / row_height, 1, count);
    const int selected = static_cast<int>(gallery_context_index(selection));
    const int first = std::clamp(selected - visible / 2, 0, count - visible);
    const int panel_height = ui.pixels(81) + row_height * visible;
    const int top = std::max(ui.pixels(14),
                             (static_cast<int>(height) - panel_height) / 2);
    rectangle(rgba, width, height, center - ui.pixels(255), top,
              center + ui.pixels(255),
              top + panel_height,
              {4, 10, 16, 230});
    text(rgba, width, height, center - ui.pixels(225), top + ui.pixels(24),
         "SCENES", {110, 225, 255, 255}, ui.font(3));
    text(rgba, width, height, center - ui.pixels(70), top + ui.pixels(34),
         "Z X C V B N DEBUG   M CAPTURE", {185, 220, 235, 255}, ui.font(1));

    const auto row = [&](int y, GalleryContext context, Color background,
                         Color icon, std::string_view name,
                         std::string_view state, Color state_color) {
        if (selection == context) {
            rectangle(rgba, width, height, center - ui.pixels(226),
                      y - ui.pixels(6), center + ui.pixels(226),
                      y + ui.pixels(60), {105, 255, 155, 255});
        }
        rectangle(rgba, width, height, center - ui.pixels(220), y,
                  center + ui.pixels(220), y + ui.pixels(54), background);
        rectangle(rgba, width, height, center - ui.pixels(198),
                  y + ui.pixels(8), center - ui.pixels(160),
                  y + ui.pixels(46), icon);
        text(rgba, width, height, center - ui.pixels(135), y + ui.pixels(6),
             name, {245, 247, 250, 255}, ui.font(2));
        text(rgba, width, height, center - ui.pixels(135), y + ui.pixels(31),
             state, state_color, ui.font(1));
    };

    int y = top + ui.pixels(78);
    for (int index = first; index < first + visible; ++index) {
        const GalleryEntry &entry = gallery_entries[static_cast<std::size_t>(index)];
        const bool available = !features || gallery_backend_supports(entry.context, *features);
        row(y, entry.context,
            available ? color(entry.background) : Color{25, 30, 35, 255},
            available ? color(entry.icon) : Color{70, 75, 80, 255}, entry.name,
            available ? entry.help : "UNAVAILABLE ON THIS BACKEND",
            available ? Color{105, 255, 155, 255} : Color{180, 185, 190, 255});
        y += row_height;
    }
}

void draw_backend_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, std::span<const std::string> names,
    std::span<const std::string> details, std::size_t selection) {
    if (names.empty() || names.size() != details.size()) return;
    const MenuScale ui = menu_scale(width, height);
    const int center = static_cast<int>(width) / 2;
    const int row_height = ui.pixels(88);
    const int panel_height = ui.pixels(130) +
        row_height * static_cast<int>(names.size());
    const int top = std::max(ui.pixels(14),
        (static_cast<int>(height) - panel_height) / 2);
    rectangle(rgba, width, height, center - ui.pixels(330), top,
              center + ui.pixels(330), top + panel_height,
              {4, 10, 16, 242});
    text(rgba, width, height, center - ui.pixels(292),
         top + ui.pixels(28), "SELECT PHYSICS BACKEND",
         {110, 225, 255, 255}, ui.font(3));
    text(rgba, width, height, center - ui.pixels(292),
         top + ui.pixels(67), "UP DOWN SELECT   ENTER START",
         {185, 220, 235, 255}, ui.font(1));
    for (std::size_t index = 0U; index < names.size(); ++index) {
        const int y = top + ui.pixels(105) +
            static_cast<int>(index) * row_height;
        if (index == selection) {
            rectangle(rgba, width, height, center - ui.pixels(296),
                      y - ui.pixels(5), center + ui.pixels(296),
                      y + ui.pixels(72), {105, 255, 155, 255});
        }
        rectangle(rgba, width, height, center - ui.pixels(290), y,
                  center + ui.pixels(290), y + ui.pixels(62),
                  index == selection ? Color{28, 64, 50, 255}
                                     : Color{27, 38, 48, 255});
        text(rgba, width, height, center - ui.pixels(262),
             y + ui.pixels(8), names[index], {245, 247, 250, 255},
             ui.font(2));
        text(rgba, width, height, center - ui.pixels(262),
             y + ui.pixels(37), details[index],
             index == selection ? Color{130, 255, 185, 255}
                                : Color{160, 190, 210, 255},
             ui.font(1));
    }
}

void draw_count_overlay(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, const std::string &value,
                        bool invalid) {
    const MenuScale ui = menu_scale(width, height);
    const GalleryEntry &entry = gallery_entry(context);
    const bool fluid = entry.count_kind == GalleryCountKind::fluid_particles;
    const int center_x = static_cast<int>(width) / 2;
    const int center_y = static_cast<int>(height) / 2;
    rectangle(rgba, width, height, center_x - ui.pixels(260),
              center_y - ui.pixels(118), center_x + ui.pixels(260),
              center_y + ui.pixels(118), {4, 10, 16, 242});
    text(rgba, width, height, center_x - ui.pixels(220),
         center_y - ui.pixels(88),
         fluid ? "FLUID PARTICLE CAP" : "DUMP SPHERES",
         fluid ? Color{35, 150, 255, 255} : Color{245, 130, 45, 255},
         ui.font(2));
    rectangle(rgba, width, height, center_x - ui.pixels(220),
              center_y - ui.pixels(30), center_x + ui.pixels(220),
              center_y + ui.pixels(20),
              invalid ? Color{105, 20, 20, 255} : Color{27, 38, 48, 255});
    text(rgba, width, height, center_x - ui.pixels(198),
         center_y - ui.pixels(17), (fluid ? "MAX " : "COUNT ") + value,
         {245, 247, 250, 255}, ui.font(2));
    text(rgba, width, height, center_x - ui.pixels(220),
         center_y + ui.pixels(42),
         invalid ? ("USE " + std::to_string(entry.minimum_count) + '-' +
                    std::to_string(entry.maximum_count))
                 : ("MIN " + std::to_string(entry.minimum_count) + "  MAX " +
                    std::to_string(entry.maximum_count)),
         invalid ? Color{255, 105, 105, 255} : Color{160, 190, 210, 255},
         ui.font(1));
    text(rgba, width, height, center_x - ui.pixels(220),
         center_y + ui.pixels(72), "ENTER APPLY  ESC CANCEL",
         {160, 190, 210, 255}, ui.font(1));
}

void draw_brick_settings_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const std::array<std::string, 3> &values,
    std::size_t selected, bool invalid) {
    const MenuScale ui = menu_scale(width, height);
    const int center_x = static_cast<int>(width) / 2;
    const int center_y = static_cast<int>(height) / 2;
    rectangle(rgba, width, height, center_x - ui.pixels(300),
              center_y - ui.pixels(180), center_x + ui.pixels(300),
              center_y + ui.pixels(180), {4, 10, 16, 242});
    text(rgba, width, height, center_x - ui.pixels(260),
         center_y - ui.pixels(145), "BRICK WALL SETTINGS",
         {105, 255, 155, 255}, ui.font(2));
    constexpr std::array labels{"COUNT  ", "SCALE  ", "WALLS  "};
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int y = center_y - ui.pixels(82) +
                      static_cast<int>(index) * ui.pixels(58);
        rectangle(rgba, width, height, center_x - ui.pixels(260), y,
                  center_x + ui.pixels(260), y + ui.pixels(42),
                  index == selected
                      ? (invalid ? Color{105, 20, 20, 255}
                                 : Color{45, 72, 88, 255})
                      : Color{27, 38, 48, 255});
        text(rgba, width, height, center_x - ui.pixels(238),
             y + ui.pixels(12),
             std::string(labels[index]) + values[index],
             {245, 247, 250, 255}, ui.font(2));
    }
    text(rgba, width, height, center_x - ui.pixels(260),
         center_y + ui.pixels(106),
         invalid ? "COUNT 1-4096  SCALE .5-2  WALLS 1-16"
                 : "TAB FIELD  ENTER APPLY AND RESTART",
         invalid ? Color{255, 105, 105, 255} : Color{160, 190, 210, 255},
         ui.font(1));
    text(rgba, width, height, center_x - ui.pixels(260),
         center_y + ui.pixels(138),
         "C AUTO-CALIBRATE  V VERIFY AND SAVE  ESC CANCEL",
         {160, 190, 210, 255}, ui.font(1));
}

void draw_status_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         std::string_view message, bool failure) {
    if (message.empty() || width < 80U || height < 54U) return;
    constexpr int scale = 2;
    constexpr int glyph_width = 6 * scale;
    const std::size_t maximum_characters =
        static_cast<std::size_t>((width - 48U) / glyph_width);
    message = message.substr(0U, maximum_characters);
    const int panel_width = static_cast<int>(message.size()) * glyph_width + 28;
    rectangle(rgba, width, height, 16, 16, 16 + panel_width, 50,
              failure ? Color{72, 18, 18, 235} : Color{8, 52, 34, 235});
    text(rgba, width, height, 30, 26, message,
         failure ? Color{255, 135, 135, 255}
                 : Color{120, 255, 180, 255}, scale);
}

bool draw_rigid_contact_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const std::vector<PmContactEvent> &contacts,
    Camera camera, std::string &error) {
    error.clear();
    for (const PmContactEvent &contact : contacts) {
        if (contact.type != PM_CONTACT_RIGID_RIGID) continue;
        const Vec3 position{contact.position.x, contact.position.y,
                            contact.position.z};
        const Vec3 normal{contact.normal.x, contact.normal.y,
                          contact.normal.z};
        const Vec3 friction{contact.frictionImpulse.x,
                            contact.frictionImpulse.y,
                            contact.frictionImpulse.z};
        const ScreenPoint origin = project(position, camera, width, height);
        if (!origin.visible) continue;
        for (int offset = -3; offset <= 3; ++offset) {
            pixel(rgba, width, height, origin.x + offset, origin.y,
                  {255, 70, 70, 255});
            pixel(rgba, width, height, origin.x, origin.y + offset,
                  {255, 70, 70, 255});
        }
        arrow(rgba, width, height, origin,
              project(add(position, multiply(normal, std::clamp(
                  0.16F + contact.normalImpulse * 0.04F, 0.16F, 0.45F))),
                  camera, width, height), {70, 255, 110, 255});
        const float friction_size = length(friction);
        if (friction_size > 1.0e-6F)
            arrow(rgba, width, height, origin,
                  project(add(position, multiply(
                      friction, std::clamp(0.12F + friction_size * 0.08F,
                                           0.12F, 0.4F) / friction_size)),
                      camera, width, height), {255, 190, 45, 255});
    }
    return true;
}

namespace {

[[nodiscard]] Vec3 host(PmFloat3 value) noexcept {
    return {value.x, value.y, value.z};
}

template <typename State>
[[nodiscard]] std::vector<Vec3> positions_of(
    const std::vector<State> &states) {
    std::vector<Vec3> output;
    output.reserve(states.size());
    for (const State &state : states) output.push_back(host(state.position));
    return output;
}

void draw_indexed_edges(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        const std::vector<Vec3> &positions,
                        const std::vector<std::uint32_t> &indices,
                        Camera camera, Color value) {
    for (std::size_t triangle = 0U; triangle + 2U < indices.size();
         triangle += 3U) {
        const std::uint32_t ids[3]{indices[triangle], indices[triangle + 1U],
                                   indices[triangle + 2U]};
        if (ids[0] >= positions.size() || ids[1] >= positions.size() ||
            ids[2] >= positions.size()) continue;
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const ScreenPoint first = project(
                positions[ids[edge]], camera, width, height);
            const ScreenPoint second = project(
                positions[ids[(edge + 1U) % 3U]], camera, width, height);
            if (first.visible && second.visible)
                line(rgba, width, height, first.x, first.y,
                     second.x, second.y, value);
        }
    }
}

void draw_normals(std::vector<std::uint32_t> &rgba,
                  std::uint32_t width, std::uint32_t height,
                  const std::vector<Vec3> &positions,
                  const std::vector<std::uint32_t> &indices,
                  Camera camera) {
    std::vector<Vec3> normals(positions.size());
    for (std::size_t triangle = 0U; triangle + 2U < indices.size();
         triangle += 3U) {
        const std::uint32_t a = indices[triangle];
        const std::uint32_t b = indices[triangle + 1U];
        const std::uint32_t c = indices[triangle + 2U];
        if (a >= positions.size() || b >= positions.size() ||
            c >= positions.size()) continue;
        const Vec3 face = cross(subtract(positions[b], positions[a]),
                                subtract(positions[c], positions[a]));
        normals[a] = add(normals[a], face);
        normals[b] = add(normals[b], face);
        normals[c] = add(normals[c], face);
    }
    const std::size_t stride = std::max<std::size_t>(
        1U, (positions.size() + 699U) / 700U);
    for (std::size_t vertex = 0U; vertex < positions.size(); vertex += stride) {
        const Vec3 normal = normalized(normals[vertex]);
        if (length(normal) <= 1.0e-6F) continue;
        arrow(rgba, width, height,
              project(positions[vertex], camera, width, height),
              project(add(positions[vertex], multiply(normal, 0.065F)),
                      camera, width, height), {31, 255, 56, 242});
    }
}

void draw_force(std::vector<std::uint32_t> &rgba,
                std::uint32_t width, std::uint32_t height, Vec3 position,
                Vec3 force, Camera camera, Color value) {
    const float magnitude = length(force);
    if (!(magnitude > 1.0e-5F) || !std::isfinite(magnitude)) return;
    const float amount = std::clamp(0.003F * magnitude, 0.012F, 0.14F);
    arrow(rgba, width, height, project(position, camera, width, height),
          project(add(position, multiply(force, amount / magnitude)),
                  camera, width, height), value);
}

[[nodiscard]] std::vector<std::uint32_t> surface_indices(
    const TriangleMesh &mesh, std::size_t surface_size) {
    if (surface_size == mesh.indices.size()) {
        std::vector<std::uint32_t> sequential(surface_size);
        for (std::size_t index = 0; index < surface_size; ++index)
            sequential[index] = static_cast<std::uint32_t>(index);
        return sequential;
    }
    return mesh.indices;
}

} // namespace

bool draw_cloth_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const SceneDefinition &scene,
    const CommandSceneFrame &frame, std::uint32_t cloth, Camera camera,
    ClothDebugOptions options, std::string &error) {
    error.clear();
    if (cloth >= scene.cloths.size() || cloth >= frame.cloth_vertices.size() ||
        scene.cloths[cloth].mesh_index >= scene.meshes.size()) {
        error = "cloth debug output is unavailable";
        return false;
    }
    const auto &states = frame.cloth_vertices[cloth];
    const auto &mesh = scene.meshes[scene.cloths[cloth].mesh_index];
    const std::vector<Vec3> nodes = positions_of(states);
    std::vector<Vec3> surface;
    if (cloth < frame.cloth_surface_vertices.size()) {
        surface.reserve(frame.cloth_surface_vertices[cloth].size());
        for (PmFloat3 value : frame.cloth_surface_vertices[cloth])
            surface.push_back(host(value));
    }
    if (surface.empty()) surface = nodes;
    const std::vector<std::uint32_t> indices =
        surface_indices(mesh, surface.size());
    if (options.wireframe)
        draw_indexed_edges(rgba, width, height, surface, indices, camera,
                           {26, 230, 255, 225});
    if (options.normals)
        draw_normals(rgba, width, height, surface, indices, camera);
    if (options.bonds)
        draw_indexed_edges(rgba, width, height, nodes, mesh.indices, camera,
                           {255, 155, 25, 220});
    for (std::size_t index = 0U; index < states.size(); ++index) {
        if (options.rigid_contact_forces) {
            draw_force(rgba, width, height, nodes[index],
                       host(states[index].rigidContactForce), camera,
                       {31, 122, 255, 242});
            draw_force(rgba, width, height, nodes[index],
                       host(states[index].softBodyContactForce), camera,
                       {205, 110, 255, 242});
        }
        if (options.fluid_contact_forces)
            draw_force(rgba, width, height, nodes[index],
                       host(states[index].liquidContactForce), camera,
                       {255, 219, 20, 242});
    }
    return true;
}

bool draw_soft_body_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const SceneDefinition &scene,
    const CommandSceneFrame &frame, std::uint32_t body, Camera camera,
    ClothDebugOptions options, std::string &error) {
    error.clear();
    if (body >= scene.soft_bodies.size() ||
        body >= frame.soft_body_nodes.size() ||
        body >= frame.soft_body_surface_vertices.size() ||
        scene.soft_bodies[body].mesh_index >= scene.meshes.size()) {
        error = "soft-body debug output is unavailable";
        return false;
    }
    const auto &definition = scene.soft_bodies[body];
    const auto &states = frame.soft_body_nodes[body];
    const std::vector<Vec3> nodes = positions_of(states);
    std::vector<Vec3> surface;
    surface.reserve(frame.soft_body_surface_vertices[body].size());
    for (PmFloat3 value : frame.soft_body_surface_vertices[body])
        surface.push_back(host(value));
    const auto &indices = scene.meshes[definition.mesh_index].indices;
    if (options.wireframe)
        draw_indexed_edges(rgba, width, height, surface, indices, camera,
                           {186, 105, 255, 225});
    if (options.normals)
        draw_normals(rgba, width, height, surface, indices, camera);
    if (options.bonds) {
        for (const SoftBodyBond &bond : definition.bonds) {
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
    if (options.rigid_contact_forces)
        for (std::size_t index = 0; index < states.size(); ++index) {
            draw_force(rgba, width, height, nodes[index],
                       host(states[index].rigidContactForce), camera,
                       {31, 122, 255, 242});
            draw_force(rgba, width, height, nodes[index],
                       host(states[index].clothContactForce), camera,
                       {205, 110, 255, 242});
        }
    return true;
}

bool draw_rope_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const CommandSceneFrame &frame,
    std::uint32_t rope, Camera camera, std::string &error) {
    error.clear();
    if (rope >= frame.rope_nodes.size()) {
        error = "rope debug output is unavailable";
        return false;
    }
    const std::vector<Vec3> nodes = positions_of(frame.rope_nodes[rope]);
    for (std::size_t index = 1U; index < nodes.size(); ++index) {
        const ScreenPoint first = project(nodes[index - 1U], camera,
                                          width, height);
        const ScreenPoint second = project(nodes[index], camera,
                                           width, height);
        if (first.visible && second.visible)
            line(rgba, width, height, first.x, first.y,
                 second.x, second.y, {60, 240, 255, 245});
    }
    return true;
}

void draw_physics_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const CommandSceneFrame &frame, Camera camera,
    PhysicsDebugVisualizationOptions options) {
    for (const PmDebugRecord &record : frame.debug_records) {
        const Vec3 position = host(record.position);
        const Vec3 velocity = host(record.velocity);
        const Vec3 force = host(record.force);
        if (options.velocities)
            draw_force(rgba, width, height, position, velocity, camera,
                       {80, 220, 255, 242});
        if ((options.rigid_forces && record.type == PM_DEBUG_RIGID) ||
            (options.fluid_forces && record.type != PM_DEBUG_RIGID))
            draw_force(rgba, width, height, position, force, camera,
                       {255, 219, 20, 242});
    }
    if (options.contact_normals) {
        std::string ignored;
        (void)draw_rigid_contact_overlay(rgba, width, height, frame.contacts,
                                         camera, ignored);
    }
}

bool draw_smoke_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const CommandSceneFrame &frame, Camera camera,
    SmokeDebugMode mode, std::string &error) {
    error.clear();
    if (mode == SmokeDebugMode::none) return true;
    for (const PmSmokeParticleState &particle : frame.smoke_particles) {
        const Vec3 position = host(particle.position);
        const ScreenPoint point = project(position, camera, width, height);
        if (!point.visible) continue;
        Color value{40, 220, 255, 220};
        if (mode == SmokeDebugMode::pressure) {
            const float amount = std::clamp(std::fabs(particle.pressure),
                                            0.0F, 1.0F);
            value = particle.pressure < 0.0F
                ? Color{60, 130, 255, 220}
                : Color{static_cast<std::uint8_t>(80 + 175 * amount),
                        70, 45, 220};
        } else if (mode == SmokeDebugMode::density_temperature) {
            const float amount = std::clamp(particle.numberDensity,
                                            0.0F, 1.0F);
            value = {static_cast<std::uint8_t>(40 + 210 * amount), 120,
                     static_cast<std::uint8_t>(255 - 180 * amount), 220};
        }
        rectangle(rgba, width, height, point.x - 2, point.y - 2,
                  point.x + 3, point.y + 3, value);
        if (mode == SmokeDebugMode::velocity)
            draw_force(rgba, width, height, position,
                       host(particle.velocity), camera,
                       {255, 255, 255, 205});
        else if (mode == SmokeDebugMode::vorticity)
            draw_force(rgba, width, height, position,
                       host(particle.vorticity), camera,
                       {255, 125, 25, 205});
    }
    return true;
}

} // namespace parallel_mater::gallery
