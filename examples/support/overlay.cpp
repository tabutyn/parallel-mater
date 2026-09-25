// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/overlay.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string_view>

namespace parallel_mater::gallery {
namespace {

struct Color {
    std::uint8_t red{};
    std::uint8_t green{};
    std::uint8_t blue{};
    std::uint8_t alpha{255U};
};

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

[[nodiscard]] std::array<std::uint8_t, 7> glyph(char character) {
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

[[nodiscard]] Vec3 subtract(Vec3 first, Vec3 second) {
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

[[nodiscard]] Vec3 add(Vec3 first, Vec3 second) {
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

[[nodiscard]] Vec3 multiply(Vec3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

[[nodiscard]] float dot(Vec3 first, Vec3 second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

[[nodiscard]] Vec3 cross(Vec3 first, Vec3 second) {
    return {first.y * second.z - first.z * second.y,
            first.z * second.x - first.x * second.z,
            first.x * second.y - first.y * second.x};
}

[[nodiscard]] float length(Vec3 value) {
    return std::sqrt(std::max(dot(value, value), 0.0F));
}

[[nodiscard]] Vec3 normalized(Vec3 value) {
    const float size = length(value);
    return size > 1.0e-6F ? multiply(value, 1.0F / size) : Vec3{};
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

} // namespace

void draw_timing_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         const WorldStepTimings &timings) {
    rectangle(rgba, width, height, 18, 18, 390, 274, {5, 12, 18, 215});
    text(rgba, width, height, 32, 30, "GPU KERNELS", {80, 220, 255, 255}, 2);
    if (!timings.available) {
        text(rgba, width, height, 32, 58, "NO TIMING SAMPLE", {255, 190, 70, 255},
             2);
        return;
    }
    timing_row(rgba, width, height, 58, "INTEGRATE",
               timings.rigid_integration);
    timing_row(rgba, width, height, 78, "BOUNDS",
               timings.rigid_world_bounds);
    timing_row(rgba, width, height, 98, "PAIR FILTER",
               timings.rigid_pair_filter);
    timing_row(rgba, width, height, 118, "PAIR COMPACT",
               timings.rigid_pair_compaction);
    timing_row(rgba, width, height, 138, "LEAF PAIRS",
               timings.rigid_leaf_pair_generation);
    timing_row(rgba, width, height, 158, "TRI CONTACT",
               timings.rigid_contact_evaluation);
    timing_row(rgba, width, height, 178, "SOLVE",
               timings.rigid_contact_solve);
    timing_row(rgba, width, height, 198, "CLEAR",
               timings.rigid_input_clear);
    char total[96]{};
    std::snprintf(total, sizeof(total), "TOTAL        %7.3f MS",
                  timings.total_gpu_milliseconds);
    text(rgba, width, height, 32, 224, total, {100, 255, 155, 255}, 2);
}

void draw_cloth_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &timings) {
    rectangle(rgba, width, height, 18, 18, 410, 230, {5, 12, 18, 215});
    text(rgba, width, height, 32, 30, "CLOTH GPU KERNELS",
         {80, 220, 255, 255}, 2);
    if (!timings.available) {
        text(rgba, width, height, 32, 58, "NO TIMING SAMPLE",
             {255, 190, 70, 255}, 2);
        return;
    }
    timing_row(rgba, width, height, 58, "RIGID STEP",
        {timings.rigid_integration.total_milliseconds +
         timings.rigid_contact_generation.total_milliseconds +
         timings.rigid_contact_solve.total_milliseconds,
         timings.rigid_integration.launch_count +
         timings.rigid_contact_generation.launch_count +
         timings.rigid_contact_solve.launch_count});
    timing_row(rgba, width, height, 84, "PREDICT", timings.cloth_prediction);
    timing_row(rgba, width, height, 110, "LINKS", timings.cloth_constraints);
    timing_row(rgba, width, height, 136, "CONTACTS", timings.cloth_contacts);
    char total[96]{};
    std::snprintf(total, sizeof(total), "TOTAL        %7.3f MS",
                  timings.total_gpu_milliseconds);
    text(rgba, width, height, 32, 177, total, {100, 255, 155, 255}, 2);
}

void draw_fluid_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &physics,
                               const RendererTimings &renderer,
                               const WorldStatistics &statistics,
                               std::uint32_t capacity) {
    rectangle(rgba, width, height, 18, 18, 480, 482,
              {5, 12, 18, 225});
    text(rgba, width, height, 32, 30,
         renderer.particle_view ? "FLUID PARTICLE TIMINGS"
                                : "FLUID SURFACE TIMINGS",
         {80, 220, 255, 255}, 2);
    if (!physics.available) {
        text(rgba, width, height, 32, 58, "NO TIMING SAMPLE",
             {255, 190, 70, 255}, 2);
        return;
    }
    timing_row(rgba, width, height, 58, "SPAWN", physics.fluid_spawn);
    timing_row(rgba, width, height, 78, "CELL SORT",
               physics.fluid_neighbor_sort);
    timing_row(rgba, width, height, 98, "NEIGHBORS",
               physics.fluid_neighbor_forces);
    timing_row(rgba, width, height, 118, "INTEGRATE",
               physics.fluid_integration);
    timing_row(rgba, width, height, 138, "TRI COLLIDE",
               physics.fluid_static_contacts);
    timing_row(rgba, width, height, 158, "BODY INDEX",
               physics.fluid_body_index);
    timing_row(rgba, width, height, 178, "MOVING TRI",
               physics.fluid_moving_contacts);
    timing_row(rgba, width, height, 198, "EVENTS",
               physics.fluid_contact_events);
    timing_row(rgba, width, height, 218, "OUTFLOW",
               physics.fluid_outflow_compaction);
    char line[128]{};
    std::snprintf(line, sizeof(line), "PHYSICS GPU    %7.3f MS",
                  physics.total_gpu_milliseconds);
    text(rgba, width, height, 32, 244, line, {100, 255, 155, 255}, 2);
    if (renderer.particle_view)
        std::snprintf(line, sizeof(line), "SURFACE GPU       OFF");
    else
        std::snprintf(line, sizeof(line), "SURFACE GPU    %7.3f MS",
                      renderer.surface_gpu_milliseconds);
    text(rgba, width, height, 32, 268, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "OPTIX + COPY   %7.3f MS",
                  renderer.raytrace_wall_milliseconds);
    text(rgba, width, height, 32, 288, line, {225, 235, 242, 255}, 2);
    if (renderer.particle_view)
        std::snprintf(line, sizeof(line), "SPRITES CPU   %7.3f MS",
                      renderer.foam_wall_milliseconds);
    else
        std::snprintf(line, sizeof(line), "FOAM CPU      %7.3f MS  %u PATCHES",
                      renderer.foam_wall_milliseconds,
                      renderer.foam_patch_count);
    text(rgba, width, height, 32, 308, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "RENDER WALL    %7.3f MS",
                  renderer.total_wall_milliseconds);
    text(rgba, width, height, 32, 332, line, {100, 255, 155, 255}, 2);
    std::snprintf(line, sizeof(line), "LIVE %u / MAX %u",
                  statistics.particle_count, capacity);
    text(rgba, width, height, 32, 364, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "EMITTED %llu  OUTFLOW %llu",
                  static_cast<unsigned long long>(statistics.emitted_particle_count),
                  static_cast<unsigned long long>(statistics.destroyed_particle_count));
    text(rgba, width, height, 32, 384, line, {225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "CAPACITY MISSED %llu",
                  static_cast<unsigned long long>(statistics.spawn_capacity_miss_count));
    text(rgba, width, height, 32, 404, line,
         statistics.spawn_capacity_miss_count
             ? Color{255, 190, 70, 255} : Color{225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "SURFACE OUTLIERS %u",
                  renderer.surface_excluded_particle_count);
    text(rgba, width, height, 32, 428, line,
         renderer.surface_excluded_particle_count
             ? Color{255, 190, 70, 255} : Color{225, 235, 242, 255}, 2);
    std::snprintf(line, sizeof(line), "CONTACTS %u  OVERFLOW %u",
                  statistics.contact_count, statistics.contact_overflow_count);
    text(rgba, width, height, 32, 452, line,
         statistics.contact_overflow_count
             ? Color{255, 190, 70, 255} : Color{225, 235, 242, 255}, 2);
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

void draw_context_overlay(std::vector<std::uint32_t> &rgba,
                          std::uint32_t width, std::uint32_t height,
                          GalleryContext selection) {
    const int center = static_cast<int>(width) / 2;
    const int top = std::max(14, static_cast<int>(height) / 2 - 312);
    rectangle(rgba, width, height, center - 255, top, center + 255, top + 625,
              {4, 10, 16, 230});
    text(rgba, width, height, center - 225, top + 24, "SCENES",
         {110, 225, 255, 255}, 3);

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

    row(top + 78, GalleryContext::rigid_body, {48, 55, 63, 235},
        {170, 176, 184, 255}, "RIGID BODY", "AVAILABLE",
        {105, 255, 155, 255});
    row(top + 146, GalleryContext::dump, {62, 38, 22, 235},
        {245, 130, 45, 255}, "DUMP", "AVAILABLE  P EDITS SPHERES",
        {105, 255, 155, 255});
    row(top + 214, GalleryContext::fluid, {12, 42, 65, 235},
        {35, 150, 255, 255}, "FLUID", "P CAP  V PARTICLES  R RESET",
        {105, 255, 155, 255});
    row(top + 282, GalleryContext::fluid_rigid, {25, 52, 64, 235},
        {35, 190, 230, 255}, "FLUID RIGID", "64 FREE SPHERES  P CAP",
        {105, 255, 155, 255});
    row(top + 350, GalleryContext::peg_paint, {44, 28, 61, 235},
        {42, 145, 255, 255}, "PEG PAINT", "ARROWS GRAVITY  P CAP",
        {105, 255, 155, 255});
    row(top + 418, GalleryContext::cloth, {40, 42, 58, 235},
        {236, 188, 96, 255}, "CLOTH", "ARROWS GRAVITY  R RESET",
        {105, 255, 155, 255});
    row(top + 486, GalleryContext::cloth_tear, {53, 37, 48, 235},
        {255, 126, 111, 255}, "CLOTH TEAR", "45 DEG GRAVITY  R RESET",
        {105, 255, 155, 255});
    row(top + 554, GalleryContext::cloth_paint, {28, 49, 55, 235},
        {65, 177, 240, 255}, "CLOTH PAINT", "FLUID CONTACT PAINT  P CAP",
        {105, 255, 155, 255});
}

void draw_count_overlay(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, const std::string &value,
                        bool invalid) {
    const bool fluid = is_fluid_context(context);
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
         fluid ? (invalid ? "USE 100-100000" : "MIN 100  MAX 100000")
               : (invalid ? "USE 10-1000" : "MIN 10  MAX 1000"),
         invalid ? Color{255, 105, 105, 255} : Color{160, 190, 210, 255}, 1);
    text(rgba, width, height, center_x - 220, center_y + 72,
         "ENTER APPLY  ESC CANCEL", {160, 190, 210, 255}, 1);
}

} // namespace parallel_mater::gallery
