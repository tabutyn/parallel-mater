// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace parallel_mater {
namespace {

Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 subtract(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 multiply(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
Vec3 rotate(Quaternion q, Vec3 v) {
    const Vec3 axis{q.x, q.y, q.z};
    const Vec3 t = multiply(cross(axis, v), 2.0F);
    return add(v, add(multiply(t, q.w), cross(axis, t)));
}
bool finite(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
struct Triangle { Vec3 a, b, c; };

} // namespace

Status sample_fluid_geometry(FluidGeometrySource source,
                             std::vector<FluidParticle> &output) noexcept {
    const std::size_t original_count = output.size();
    if (source.vertices.data == nullptr || source.triangle_indices.data == nullptr ||
        source.vertices.size < 4U || source.triangle_indices.size < 12U ||
        source.triangle_indices.size % 3U != 0U ||
        !std::isfinite(source.spacing) || source.spacing <= 0.0F ||
        !finite(source.transform.position) || !finite(source.initial_velocity) ||
        !std::isfinite(source.transform.orientation.x) ||
        !std::isfinite(source.transform.orientation.y) ||
        !std::isfinite(source.transform.orientation.z) ||
        !std::isfinite(source.transform.orientation.w)) {
        return {StatusCode::invalid_argument, cudaSuccess,
                "fluid geometry source is invalid"};
    }
    const Quaternion q = source.transform.orientation;
    const float q2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if (q2 < 1.0e-12F || std::fabs(q2 - 1.0F) > 1.0e-3F)
        return {StatusCode::invalid_argument, cudaSuccess,
                "fluid geometry orientation must be normalized"};
    try {
        std::vector<Vec3> points;
        points.reserve(source.vertices.size);
        Vec3 minimum{std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max()};
        Vec3 maximum{-minimum.x, -minimum.y, -minimum.z};
        for (std::uint64_t i = 0; i < source.vertices.size; ++i) {
            const Vec3 local = source.vertices.data[i];
            if (!finite(local))
                return {StatusCode::invalid_argument, cudaSuccess,
                        "fluid geometry contains a non-finite vertex"};
            const Vec3 p = add(source.transform.position, rotate(q, local));
            points.push_back(p);
            minimum = {std::min(minimum.x, p.x), std::min(minimum.y, p.y),
                       std::min(minimum.z, p.z)};
            maximum = {std::max(maximum.x, p.x), std::max(maximum.y, p.y),
                       std::max(maximum.z, p.z)};
        }
        std::vector<Triangle> triangles;
        triangles.reserve(source.triangle_indices.size / 3U);
        for (std::uint64_t i = 0; i < source.triangle_indices.size; i += 3U) {
            const auto a = source.triangle_indices.data[i];
            const auto b = source.triangle_indices.data[i+1U];
            const auto c = source.triangle_indices.data[i+2U];
            if (a >= points.size() || b >= points.size() || c >= points.size())
                return {StatusCode::invalid_argument, cudaSuccess,
                        "fluid geometry index is outside the vertex buffer"};
            triangles.push_back({points[a], points[b], points[c]});
        }
        const Vec3 extent = subtract(maximum, minimum);
        const float row_height = source.spacing * std::sqrt(3.0F) * 0.5F;
        const float layer_height = source.spacing * std::sqrt(2.0F / 3.0F);
        if (!finite(extent) || std::min({extent.x, extent.y, extent.z}) <
                source.spacing || extent.x / source.spacing > 254.0F ||
            extent.y / layer_height > 254.0F || extent.z / row_height > 254.0F)
            return {StatusCode::invalid_argument, cudaSuccess,
                    "fluid geometry must be a bounded three-dimensional volume"};
        const std::array<std::uint32_t, 3> dimensions{
            static_cast<std::uint32_t>(std::ceil(extent.x/source.spacing))+1U,
            static_cast<std::uint32_t>(std::ceil(extent.y/layer_height))+1U,
            static_cast<std::uint32_t>(std::ceil(extent.z/row_height))+1U};
        if (static_cast<std::uint64_t>(dimensions[0])*dimensions[1]*
                dimensions[2] > 1'000'000U)
            return {StatusCode::capacity_exceeded, cudaSuccess,
                    "fluid geometry sampling grid is too large"};
        const Vec3 direction = multiply(Vec3{1.0F, 0.371F, 0.173F},
            1.0F / std::sqrt(1.0F + 0.371F*0.371F + 0.173F*0.173F));
        std::vector<float> crossings;
        crossings.reserve(triangles.size());
        for (std::uint32_t y = 0; y < dimensions[1]; ++y)
            for (std::uint32_t z = 0; z < dimensions[2]; ++z)
                for (std::uint32_t x = 0; x < dimensions[0]; ++x) {
                    const bool shifted_layer = (y & 1U) != 0U;
                    const bool shifted_row = (z & 1U) != 0U;
                    const Vec3 point{
                        minimum.x + (x + 0.5F + (shifted_row ? 0.5F : 0.0F) +
                                     (shifted_layer ? 0.5F : 0.0F))*source.spacing,
                        minimum.y + (y + 0.5F)*layer_height,
                        minimum.z + (z + 0.5F)*row_height +
                            (shifted_layer ? row_height/3.0F : 0.0F)};
                    if (point.x > maximum.x || point.y > maximum.y ||
                        point.z > maximum.z) continue;
                    crossings.clear();
                    for (const Triangle &t : triangles) {
                        const Vec3 first = subtract(t.b, t.a);
                        const Vec3 second = subtract(t.c, t.a);
                        const Vec3 p = cross(direction, second);
                        const float determinant = dot(first, p);
                        if (std::fabs(determinant) < 1.0e-8F) continue;
                        const Vec3 from_vertex = subtract(point, t.a);
                        const float u = dot(from_vertex, p)/determinant;
                        if (u < -1.0e-6F || u > 1.0F + 1.0e-6F) continue;
                        const Vec3 qv = cross(from_vertex, first);
                        const float v = dot(direction, qv)/determinant;
                        if (v < -1.0e-6F || u+v > 1.0F + 1.0e-6F) continue;
                        const float distance = dot(second, qv)/determinant;
                        if (distance > 1.0e-5F) crossings.push_back(distance);
                    }
                    std::sort(crossings.begin(), crossings.end());
                    std::size_t unique = 0U;
                    for (const float distance : crossings)
                        if (unique == 0U ||
                            distance-crossings[unique-1U] > 1.0e-4F)
                            crossings[unique++] = distance;
                    if (unique % 2U != 0U)
                        output.push_back({point, source.initial_velocity});
                }
        if (output.size() == original_count)
            return {StatusCode::invalid_argument, cudaSuccess,
                    "fluid geometry contains no particle centers"};
        return {};
    } catch (...) {
        output.resize(original_count);
        return {StatusCode::out_of_memory, cudaSuccess,
                "fluid geometry sampling allocation failed"};
    }
}

} // namespace parallel_mater
